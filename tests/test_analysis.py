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


def test_over_peak_is_a_red_flag():
    """R-38 的负样本：达成值 > 标定峰值必须报红灯，而不是变成一个漂亮的百分比。

    教程第 6 章用「162.3 TFLOPS ≈ 98% fp16 峰值」做裁决；峰值若被取成 145.7，
    同一份数据就变成 114% —— 一个物理上不可能的结论被当成裁决依据。
    """
    flops, bytes_ = gemm_workload(4096, 4096, 4096, "fp16")
    # 峰值被人为取小（例如拿 fp32 档的读数当 fp16 用）：达成值必然"超过"它
    peaks = MachinePeaks.from_profile({
        "gpu": "X",
        "measured_dram": {"copy_gbs": 909.0, "read_gbs": 900.0, "write_gbs": 934.0},
        "measured_matmul_tflops": {"torch.float16": {"tflops": 145.7}},
    }, "fp16")
    sol = SpeedOfLight(duration_ns=900_000.0, flops=flops, bytes_=bytes_, peaks=peaks)
    # 校验一下这个场景确实超过峰值（否则用例本身没意义）
    assert sol.pct_compute > 100.0
    assert sol.over_peak
    note = sol.over_peak_note()
    assert "物理上不可能" in note and "峰值取小" in note

    # 正常情况（达成 84%）绝不能报红灯
    ok = SpeedOfLight(duration_ns=951_000.0, flops=flops, bytes_=bytes_, peaks=peaks)
    assert not ok.over_peak and ok.over_peak_note() == ""

    # 容差内的轻微超出（标定口径本身有 3~4% 差异）**不报红灯，但必须给提示**（D20）
    mild = SpeedOfLight(duration_ns=928_000.0, flops=flops, bytes_=bytes_, peaks=peaks)
    assert 100.0 <= mild.pct_compute <= 102.0 + 1e-6
    assert not mild.over_peak
    assert mild.near_peak, "101% 已经物理上说不通，不能和 95% 长得一模一样"
    near = mild.near_peak_note()
    assert near and "口径差" in near and "达标" in near
    assert mild.over_peak_note() == ""
    # 正常情况连提示都不该有
    assert not ok.near_peak and ok.near_peak_note() == ""
    # 红灯时不再重复出提示（避免同一件事说两遍）
    assert sol.over_peak and not sol.near_peak and sol.near_peak_note() == ""


# ---------------------------------------------------------------------------
# bf16 与 fp16 是两个**不同的分母**（真机实测，RTX 2080 Ti / 2026-10-12）
# ---------------------------------------------------------------------------
# 该卡 bf16 走 fp32 路径只有 6.8 TFLOP/s，fp16 张量核 49.4 —— 差 7.3 倍。旧实现把 bf16 映射到
# `torch.bfloat16` 后再与 `torch.float16` 取 max()，于是 **bf16 的 roofline 分母变成了 49.4**：
# 利用率低报 7.3 倍，同时"物理上不可能"守卫的上限被抬高 7.3 倍。RTX 4090 上两条目只差 1%
# （169.0 vs 167.4），所以这个错在主力机上完全看不出来。
TURING_2080TI_PEAKS = {
    "gpu": "NVIDIA GeForce RTX 2080 Ti",
    "measured_dram": {"copy_gbs": 534.6, "read_gbs": 532.4, "write_gbs": 515.1},
    "measured_matmul_tflops": {
        "torch.bfloat16": {"tflops": 6.8, "acc_caliber": "bf16 (fp32 accumulate)",
                           "precision_settings": {}, "method": "torch.matmul"},
        "torch.float16": {"tflops": 49.4, "acc_caliber": "fp16 (fp32 accumulate)",
                          "precision_settings": {}, "method": "torch.matmul"},
        "torch.float32": {"tflops": 11.2, "acc_caliber": "fp32 (fp32 accumulate)",
                          "precision_settings": {}, "method": "torch.matmul"},
    },
}


def test_bf16_and_fp16_peaks_are_not_conflated():
    """bf16 的分母必须是 bf16 的实测值，不能拿 fp16 的顶替。"""
    from kernoscope.analyze.roofline import MachinePeaks

    bf = MachinePeaks.from_profile(TURING_2080TI_PEAKS, "bf16")
    assert bf.compute_tflops == pytest.approx(6.8), "bf16 的分母被 fp16 顶替了"
    fp = MachinePeaks.from_profile(TURING_2080TI_PEAKS, "fp16")
    assert fp.compute_tflops == pytest.approx(49.4)
    # 同机任何 dtype 的上限是 49.4 —— 它只服务"物理上不可能"守卫
    assert bf.ceiling_tflops == pytest.approx(49.4)
    # 两个数在日志里必须都出现，否则读者以为 roofline 用的是同机上限
    label = bf.source_label()
    assert "compute=6.8" in label and "49.4" in label


def test_missing_dtype_entry_is_refused_not_substituted():
    """标定里没有本次 dtype 的条目时：**拒绝**算 compute roof，而不是拿另一个 dtype 顶替。"""
    from kernoscope.analyze.roofline import MachinePeaks

    only_fp16 = {"gpu": "X", "measured_dram": {"copy_gbs": 900.0},
                 "measured_matmul_tflops": {"torch.float16": {"tflops": 49.4}}}
    bf = MachinePeaks.from_profile(only_fp16, "bf16")
    assert not bf.caliber_ok, "拿 fp16 的峰值顶替 bf16 时必须拒绝"
    assert "顶替" in bf.caliber_note or "没有 dtype=bf16" in bf.caliber_note


def test_over_peak_guard_uses_the_same_machine_ceiling_not_the_dtype_peak():
    """守卫回答的是"这台机器有没有可能做到"，分母必须取同机最大者。

    否则峰值取小会**误报**：本用例里达成值 20 TFLOPS 高于 bf16 的 6.8，但远低于同机 fp16 的
    49.4 —— 它是可能的（自定义 kernel 用 fp16 路径做 16 位数学），不该报"物理上不可能"。
    真正不可能的是超过 49.4。
    """
    from kernoscope.analyze.roofline import MachinePeaks

    peaks = MachinePeaks.from_profile(TURING_2080TI_PEAKS, "bf16")
    # 1 ms 内 2e10 FLOP ⇒ 20 TFLOP/s（在 bf16 的 6.8 与同机上限 49.4 之间）；
    # 字节取 1e8 ⇒ 100 GB/s，远低于带宽峰值，避免带宽侧把用例带偏。
    sol = SpeedOfLight(duration_ns=1_000_000.0, flops=2e10, bytes_=1e8, peaks=peaks)
    assert sol.achieved_tflops == pytest.approx(20.0)
    assert sol.pct_compute == pytest.approx(294.1, rel=1e-2)      # 相对本 dtype 峰值：远超
    assert sol.pct_compute_ceiling == pytest.approx(40.5, rel=1e-2)  # 相对同机上限：没超
    assert not sol.over_peak, "40.5% 的上限不该被报成物理上不可能"

    # 超过同机上限（49.4）才是不可能是：1 ms 内 6e10 FLOP = 60 TFLOP/s
    sol2 = SpeedOfLight(duration_ns=1_000_000.0, flops=6e10, bytes_=1e8, peaks=peaks)
    assert sol2.pct_compute_ceiling > 102.0
    assert sol2.over_peak
    note = sol2.over_peak_note()
    assert "同机任何 dtype 的上限" in note
    # 两个分母不同时必须点名本 dtype 的峰值，否则读者会把 7.3 倍的差当成噪声
    assert "本次 dtype=bf16" in note and "6.8" in note
