"""analysis 层测试：occupancy 算术、区间重叠、roofline。"""

import pytest

from kernoscope.analyze.occupancy import (achieved_occupancy_from_intervals,
                                          theoretical_occupancy)
from kernoscope.analyze.roofline import MachinePeaks, SpeedOfLight, gemm_workload


def test_theoretical_occupancy_register_limited():
    r = theoretical_occupancy(39, 0, 0, 128, "sm_89")
    assert r.blocks_per_sm == 12
    assert r.warps_per_sm == 48
    assert r.occupancy_pct == pytest.approx(100.0)


def test_theoretical_occupancy_smem_limited_matches_cuda_api():
    # 与真机 cudaOccupancyMaxActiveBlocksPerMultiprocessor 实测一致：32KB smem + 128 线程 -> 3 blocks/SM
    r = theoretical_occupancy(10, 32768, 0, 128, "sm_89")
    assert r.blocks_per_sm == 3
    assert r.occupancy_pct == pytest.approx(25.0)
    assert "shared_mem" in r.limiter


def test_achieved_occupancy_interval_overlap_exact_case():
    # 单 SM 上 3 个区间完全重叠 -> 平均并发 3.0；128 线程/块 = 4 warps；sm_89 上限 48 warps
    records = [(0, 0, 1000), (0, 0, 1000), (0, 0, 1000)]
    out = achieved_occupancy_from_intervals(records, n_sm=1, threads_per_block=128, arch="sm_89")
    assert out["avg_concurrent_blocks"] == pytest.approx(3.0)
    assert out["occupancy_pct"] == pytest.approx(25.0)


def test_achieved_occupancy_staggered_intervals():
    # A=[0,100] B=[50,150]：并发序列为 1 (0-50) / 2 (50-100) / 1 (100-150)
    # 面积 = 50*1 + 50*2 + 50*1 = 200，时间跨度 150 -> 平均并发 = 4/3
    records = [(0, 0, 100), (0, 50, 150)]
    out = achieved_occupancy_from_intervals(records, n_sm=1, threads_per_block=256, arch="sm_89")
    assert out["avg_concurrent_blocks"] == pytest.approx(4.0 / 3.0)


def test_achieved_occupancy_ignores_cross_sm_naive_use():
    # 不同 SM 独立统计：SM0 两个重叠，SM1 一个
    records = [(0, 0, 100), (0, 0, 100), (1, 0, 100)]
    out = achieved_occupancy_from_intervals(records, n_sm=2, threads_per_block=128, arch="sm_89")
    assert out["avg_concurrent_blocks"] == pytest.approx(1.5)
    assert out["samples"] == 2


def test_gemm_workload_and_roofline():
    flops, bytes_ = gemm_workload(4096, 4096, 4096, "bf16")
    assert flops == pytest.approx(2 * 4096 ** 3)
    assert bytes_ == pytest.approx(3 * 4096 * 4096 * 2)

    peaks = MachinePeaks.from_spec(171.7, 936.0, "bf16")
    sol = SpeedOfLight(duration_ns=951_000.0, flops=flops, bytes_=bytes_, peaks=peaks)
    assert sol.achieved_tflops == pytest.approx(144.5, rel=0.01)
    assert sol.pct_compute == pytest.approx(84.2, rel=0.01)
    assert sol.bound == "compute"
    assert sol.ideal_ns_compute > sol.ideal_ns_memory


def test_peaks_from_machine_profile():
    prof = {
        "gpu": "NVIDIA GeForce RTX 4090",
        "measured_dram": {"copy_gbs": 909.0, "read_gbs": 900.0, "write_gbs": 934.0},
        "measured_matmul_tflops": {
            "torch.bfloat16": {"tflops": 171.7},
            "torch.float16": {"tflops": 168.7},
            "torch.float32": {"tflops": 54.0},
        },
    }
    p = MachinePeaks.from_profile(prof, "bf16")
    assert p.compute_tflops == pytest.approx(171.7)
    # 取 copy 速率（读写混合的代表值）；取 max 会落在 write 峰值上、高估流量约 4%
    assert p.bandwidth_gbs == pytest.approx(909.0)
    p32 = MachinePeaks.from_profile(prof, "fp32")
    assert p32.compute_tflops == pytest.approx(54.0)  # 实测值，而不是规格 82.6


def test_machine_peaks_prefers_copy_bandwidth():
    """取 max 会落在 write 峰值上、系统性高估 DRAM 流量；默认应取 copy 速率。"""
    from kernoscope.analyze.roofline import MachinePeaks

    peaks = MachinePeaks.from_profile({
        "gpu": "X",
        "measured_dram": {"copy_gbs": 906.7, "read_gbs": 897.1, "write_gbs": 935.8},
        "measured_matmul_tflops": {"torch.bfloat16": {"tflops": 171.7}},
    })
    assert peaks.bandwidth_gbs == pytest.approx(906.7)

    # 没有 copy 项时回退到 max，而不是 0
    fallback = MachinePeaks.from_profile({"measured_dram": {"read_gbs": 800.0, "write_gbs": 900.0}})
    assert fallback.bandwidth_gbs == pytest.approx(900.0)
