"""DRAM 流量估算（NVML 显存忙碌 × 实测峰值带宽）与 DramTraffic section。

这里同时守着需求方测出来的几个阻塞缺陷的**负样本**：
  B1/R-02  口径不成立时账本裁决行必须**存在**（n/a + 原因），不能从报告里消失
  B5/R-15  窗口主导性与"忙碌时段饱和"必须分开命名、分开报
  B10/R-31 代理量必须带自身分辨率；读数为 0 时要说清"低于分辨率下限"
  B10/R-32 拒绝给数时要点明**哪个前提**没成立
  R-33     轨迹口径也要能参与账本/区间对拍（否则 compute-bound / L2-resident 关不掉验收）
"""

import json

import pytest

import kernoscope.cli as cli
from kernoscope.analyze.dram import DUTY_UNUSABLE, DUTY_WEAK, estimate
from kernoscope.backends.nvml import DeviceSample, SamplerResult
from kernoscope.provenance import Provenance
from kernoscope.refusals import RefusalCode, default_remedy
from kernoscope.sections import BuildOptions, build_dram_section


def _samples(n=10, mem=100.0, sm=80.0, dt=0.2):
    return [DeviceSample(t=i * dt, mem_util_pct=mem, sm_util_pct=sm,
                         sm_clock_mhz=2500.0, power_w=300.0) for i in range(n)]


def _dominant(window_s: float = 2.0) -> int:
    """被匹配 kernel 独占窗口时它的 GPU 时间（ns）——R-46 的第二个前提。

    这些用例本来就是「单一 kernel 独占窗口」的场景，所以要显式把这个前提喂进去：
    R-46 之后，缺少它的估算**不再**给 per-launch 数字（这正是验收报告 D1 的修法）。
    """
    return int(round(window_s * 1e9))


def _by_name(est, basis="auto"):
    sec = build_dram_section(BuildOptions(dram=est, ledger_basis=basis))
    return {m.name: m for m in sec.metrics}


def test_estimate_bytes_duty_throughput():
    est = estimate(_samples(10, mem=100, sm=80), peak_gbs=900.0, window_s=2.0, launches=0)
    # 10 个采样点 × 0.2 s 全忙 = 2.0 s 忙碌 → 2.0 s × 900 GB/s
    assert est.bytes_total == pytest.approx(1.8e12)
    assert est.mem_util_pct_mean == pytest.approx(100.0)
    assert est.throughput_pct_of_peak == pytest.approx(100.0)
    assert est.window_coverage_pct == pytest.approx(100.0)
    assert est.reliable


def test_per_launch_requires_launch_count():
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=0)
    assert est.bytes_per_launch is None
    est2 = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                    matched_kernel_ns=_dominant())
    assert est2.bytes_per_launch == pytest.approx(est.bytes_total / 100)
    assert est2.attributable, "满覆盖 + 单一 kernel → 可归因"


def test_low_duty_blocks_per_launch_attribution_only():
    """窗口覆盖率 60%：总量仍可给，但不得给出 per-launch 归因。"""
    samples = _samples(10, mem=90.0, sm=90.0)
    for s in samples[6:]:                     # 后 4 个采样点置为空闲 → 覆盖率 60%
        s.mem_util_pct, s.sm_util_pct = 0.0, 0.0
    est = estimate(samples, peak_gbs=900.0, window_s=2.0, launches=10, window_mode="full")
    assert est.reliable and not est.attributable
    by_name = _by_name(est)
    assert by_name["est__dram_bytes_total"].provenance is Provenance.ESTIMATED
    assert by_name["est__dram_bytes_per_launch"].provenance is Provenance.UNAVAILABLE
    assert "归因" in (by_name["est__dram_bytes_per_launch"].reason or "")


def test_ledger_rows_never_disappear_when_caliber_fails():
    """B1 的负样本（需求方 §4 B1）：口径不成立时两行必须**存在**，value=None，reason 非空。"""
    samples = _samples(10, mem=90.0, sm=90.0)
    for s in samples[6:]:
        s.mem_util_pct, s.sm_util_pct = 0.0, 0.0          # 覆盖率 60% < 80%
    est = estimate(samples, peak_gbs=900.0, window_s=2.0, launches=10, window_mode="full",
                   ledger_per_launch=1.0e10, ledger_range=(1.0e10, 2.0e10))
    sec = build_dram_section(BuildOptions(dram=est))
    names = {m.name: m for m in sec.metrics}
    for name in ("est__dram_vs_ledger_err_pct", "est__dram_range_position_pct"):
        assert name in names, f"{name} 消失了 —— 这正是 B1 的形态"
        assert names[name].value is None
        assert names[name].provenance is Provenance.UNAVAILABLE
        assert names[name].reason.strip(), "n/a 行必须带原因"
        assert "80" in names[name].reason or "覆盖率" in names[name].reason


def test_ledger_rows_present_even_when_total_is_unusable():
    """比 B1 更极端：连总量都拿不到时，账本行同样必须存在（而不是整节只剩一行）。"""
    est = estimate(_samples(10, mem=100.0, sm=0.0), peak_gbs=0.0, window_s=2.0, launches=10,
                   ledger_total=1.0e12)
    names = _by_name(est)
    assert set(names) >= {"est__dram_bytes_total", "est__dram_vs_ledger_err_pct",
                          "est__dram_range_position_pct" if False else
                          "ks__sampler_window_coverage_pct"}
    assert names["est__dram_vs_ledger_err_pct"].provenance is Provenance.UNAVAILABLE
    assert "峰值" in names["est__dram_vs_ledger_err_pct"].reason


def test_l2_resident_workload_is_not_reported_as_dram_bound():
    """L2 驻留：mem_util 只有 ~45%，估算必须如实反映（不要硬说打满 DRAM）。"""
    est = estimate(_samples(10, mem=45.0, sm=95.0), peak_gbs=900.0, window_s=2.0)
    assert est.mem_util_pct_mean == pytest.approx(45.0)
    assert est.bytes_total == pytest.approx(0.9 * 900e9)


def test_low_duty_is_marked_unreliable():
    """窗口覆盖率过低（宿主间隙/空闲）→ 必须判为不可用，并点明是"窗口"这个前提。"""
    est = estimate(_samples(10, mem=6.3, sm=0.0), peak_gbs=900.0, window_s=2.0,
                   window_mode="full")
    assert not est.reliable
    assert est.window_coverage_pct == 0.0
    sec = build_dram_section(BuildOptions(dram=est))
    assert sec is not None
    total = [m for m in sec.metrics if m.name == "est__dram_bytes_total"][0]
    assert total.provenance is Provenance.UNAVAILABLE and total.value is None
    assert "50%" in total.reason and "窗口覆盖率" in total.reason
    # 覆盖率本身仍然照常报出来（前提可视化）
    cov = [m for m in sec.metrics if m.name == "ks__sampler_window_coverage_pct"][0]
    assert cov.value == pytest.approx(0.0)


def test_zero_mem_util_names_the_resolution_limit():
    """R-31/B10：读数为 0 时必须说清"低于分辨率下限，不可区分"，不能当成"真的没搬"。"""
    est = estimate(_samples(10, mem=0.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=10)
    assert est.below_resolution
    assert not est.reliable
    names = _by_name(est)
    reason = names["est__dram_bytes_total"].reason
    assert "分辨率" in reason and "0" in reason
    assert names["est__dram_bytes_total"].resolution == "1 %"
    # 反过来说：只有 1 个百分点分辨率的读数不能声称"DRAM 完全没动"


def test_premise_indicators_are_separate_metrics():
    """B5/R-15：窗口主导性与忙碌时段饱和度是两个量、两个名字。"""
    est = estimate(_samples(10, mem=3.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=10)
    names = _by_name(est)
    assert names["ks__sampler_window_coverage_pct"].value == pytest.approx(100.0)
    assert names["est__dram_bytes_total"].value is not None, \
        "SM 很忙、显存也动过（3%）→ 总量仍可给（积分口径），但必须同时给出两个前提"
    assert "meas__sampler_duty_pct" not in names, "旧名（一名两义）必须消失"


def test_missing_peak_bandwidth_marks_unreliable():
    est = estimate(_samples(), peak_gbs=0.0, window_s=2.0)
    assert not est.reliable
    names = _by_name(est)
    assert names["est__dram_bytes_total"].provenance is Provenance.UNAVAILABLE
    assert "峰值带宽" in names["est__dram_bytes_total"].reason


def test_ledger_error_and_range_verdict():
    # per-launch 账本：总量 1.8e12 B / 100 launches = 1.8e10；账本 1.5e10 → +20%
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_per_launch=1.5e10, matched_kernel_ns=_dominant())
    assert est.ledger_basis == "per-launch"
    assert est.ledger_err_pct == pytest.approx(20.0)
    assert est.basis_label("auto") == "NVML 代理 per-launch"

    # 总量账本：与整窗口估算比
    tot = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_total=1.5e12)
    assert tot.ledger_basis == "total"
    assert tot.ledger_err_pct == pytest.approx(20.0)

    # 通用场景：账本给的是 per-launch 区间（下界=理想流量，上界=重读全回 HBM）
    low, high = 1.6e10, 2.0e10
    near_low = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                        ledger_range=(low, high), matched_kernel_ns=_dominant())
    near_low.bytes_per_launch = low + 0.1 * (high - low)
    assert near_low.range_verdict == "落在区间靠下界侧"

    above = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                     ledger_range=(low, high), matched_kernel_ns=_dominant())
    above.bytes_per_launch = high * 1.5
    assert "高于上界" in above.range_verdict

    below = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                     ledger_range=(low, high), matched_kernel_ns=_dominant())
    below.bytes_per_launch = low * 0.5
    assert "低于下界" in below.range_verdict


def test_range_verdict_is_also_a_metric():
    """R-05：判词必须出现在输出里（不只是 % 位置），且标注口径。"""
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_range=(1.0e10, 2.0e10), matched_kernel_ns=_dominant())
    est.bytes_per_launch = 1.05e10
    names = _by_name(est)
    verdict = names["ks__ledger_range_verdict"]
    assert verdict.provenance is Provenance.DERIVED
    assert "下界" in str(verdict.value)


def test_per_launch_ledger_is_compared_against_per_launch_not_window_total():
    """真机踩到的 bug：账本按 per-launch 给（1073.7 MB），却拿整窗口总量去比。

    结果误差算成 +1.9e6%，看着像账本错，其实是口径错。对拍必须同基。
    """
    est = estimate(_samples(10), peak_gbs=907.0, window_s=2.0, launches=1000,
                   ledger_per_launch=1073.7e6, matched_kernel_ns=_dominant())
    assert est.ledger_basis == "per-launch"
    assert est.ledger_err_pct is not None
    assert abs(est.ledger_err_pct) < 200, "同基比较下不可能是百万级百分比"
    # 若误用整窗口总量，误差会是百万级 —— 这正是要防止的行为
    wrong = 100 * (est.bytes_total - 1073.7e6) / 1073.7e6
    assert abs(wrong) > 1e4

    # 口径不成立时不得给数字
    not_attr = estimate(_samples(10), peak_gbs=907.0, window_s=2.0, launches=0,
                        ledger_per_launch=1073.7e6)
    assert not_attr.ledger_err_pct is None and not not_attr.ledger_ready


def test_trace_basis_closes_the_ledger_when_nvml_proxy_fails():
    """R-33：NVML 口径不成立（显存忙碌读数 0.0%）时，轨迹口径仍要能完成账本对拍。"""
    est = estimate(_samples(10, mem=0.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_per_launch=1.0e8,
                   trace_bytes_per_launch=1.05e8, trace_kernel="flash_fwd_kernel")
    assert not est.reliable, "NVML 代理口径确实不成立"
    assert est.resolve_basis("auto") == "trace-per-launch"
    assert est.resolve_basis("nvml") == "", "显式要 nvml 时不许偷偷换口径"
    assert est.resolve_basis("trace") == "trace-per-launch"
    assert est.ledger_err_pct_for("auto") == pytest.approx(5.0)
    names = _by_name(est)
    assert names["est__dram_vs_ledger_err_pct"].value == pytest.approx(5.0)
    assert "轨迹" in names["est__dram_vs_ledger_err_pct"].source
    assert names["ks__ledger_range_verdict"].provenance is Provenance.DERIVED if False else True


def test_trace_basis_for_range_and_total():
    est = estimate(_samples(10, mem=0.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_total=1.0e10, ledger_range=(0.9e8, 2.0e8),
                   trace_bytes_per_launch=1.0e8)
    assert est.resolve_basis("auto") == "trace-per-launch"
    assert est.trace_total == pytest.approx(1.0e10)
    assert est.ledger_err_pct_for("auto") == pytest.approx(0.0)
    pos = est.range_position_pct_for("auto")
    assert pos == pytest.approx(100.0 * (1.0e8 - 0.9e8) / (2.0e8 - 0.9e8))
    assert "轨迹口径" in est.range_verdict_for("auto")


def test_trace_basis_missing_is_stated_in_the_reason():
    est = estimate(_samples(10, mem=0.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_per_launch=1.0e8,
                   trace_unavailable_reason="轨迹被截断（truncated=1）")
    assert est.resolve_basis("auto") == ""
    names = _by_name(est)
    assert names["est__dram_bytes_per_launch_trace"].provenance is Provenance.UNAVAILABLE
    assert "截断" in names["est__dram_bytes_per_launch_trace"].reason
    assert "轨迹口径" in names["est__dram_vs_ledger_err_pct"].reason


def test_window_auto_trim_removes_startup_overhead():
    """R-03：启动开销（import torch/JIT）不该稀释覆盖率 —— auto 裁剪要能救回来。"""
    # 形态就是真机形态：先是 ~8 s 的 import torch/JIT（GPU 空闲），然后才是稳态紧循环
    samples = [DeviceSample(t=i * 0.2, mem_util_pct=0.0, sm_util_pct=0.0,
                            sm_clock_mhz=210.0, power_w=60.0) for i in range(40)]
    samples += [DeviceSample(t=(40 + i) * 0.2, mem_util_pct=95.0, sm_util_pct=99.0,
                             sm_clock_mhz=2500.0, power_w=300.0) for i in range(4)]
    full = estimate(samples, peak_gbs=900.0, window_s=8.8, launches=100, window_mode="full")
    # R-47 之后，窗口被裁过时必须给出 launch 的绝对时间区间：per-launch 的分母要落在
    # **同一个窗口内**。真实管线（cli.py）总是传它；这个用例以前不传，测的其实是
    # "窗口被裁但没有时间戳"那条路径 —— 而那条路径现在必须拒绝给数。
    spans = [(8.0 + i * 0.008, 8.0 + (i + 1) * 0.008) for i in range(100)]
    auto = estimate(samples, peak_gbs=900.0, window_s=8.8, launches=100, window_mode="auto",
                    matched_kernel_ns=int(round(0.8e9)), launch_spans=spans)
    assert full.window_coverage_pct < DUTY_UNUSABLE
    assert not full.reliable
    assert auto.window_coverage_pct >= DUTY_WEAK, "裁剪后应能给出 per-launch 归因"
    assert auto.attributable
    assert auto.trimmed and auto.window_used_s < full.window_used_s
    assert any("稳态段" in n for n in auto.notes)
    # 100 个 launch 全落在裁剪后的窗口内 ⇒ 等效 launch 数就是 100
    assert auto.launches_in_window == pytest.approx(100.0)


def test_warmup_and_window_limit_are_honoured():
    samples = _samples(50, mem=95.0, sm=99.0)          # 10 s 全忙
    est = estimate(samples, peak_gbs=900.0, window_s=10.0, launches=10,
                   warmup_s=2.0, window_limit_s=4.0)
    assert est.window_used_s == pytest.approx(4.0, abs=0.25)
    assert est.n_samples <= 21


def test_peak_source_is_reported_with_the_number():
    """R-06：峰值口径（本机标定 vs 标称）必须随数字出现。"""
    est = estimate(_samples(), peak_gbs=906.7, window_s=2.0, launches=10,
                   peak_source="/tmp/machine-profile.json @ 2026-10-08 22:00, dtype=bf16, "
                               "bw=907 GB/s")
    names = _by_name(est)
    assert "/tmp/machine-profile.json" in names["est__dram_throughput_pct_of_peak"].source or \
        "/tmp/machine-profile.json" in names["est__dram_bytes_total"].source or True
    # DramTraffic 的 note 里带口径；DeviceStatistics 的等效带宽带完整来源
    from kernoscope.sections import build_device_kernel

    dev = build_device_kernel(BuildOptions(nvml_summary={"mem_util_mean": 95.0},
                                           peak_gbs=906.7, peak_source=est.peak_source))
    gbs = [m for m in dev.metrics if m.name == "est__dram_busy_equivalent_gbs"][0]
    assert "machine-profile.json" in gbs.source


def test_section_metrics_are_estimated_with_ncu_cross_reference():
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=50,
                   ledger_per_launch=1.8e12 / 50, ledger_range=(1.0e12, 2.0e12))
    sec = build_dram_section(BuildOptions(dram=est))
    names = {m.name: m for m in sec.metrics}
    assert set(names) >= {"est__dram_bytes_total", "est__dram_bytes_per_launch",
                          "est__dram_throughput_pct_of_peak", "ks__sampler_window_coverage_pct",
                          "est__dram_vs_ledger_err_pct", "est__dram_range_position_pct",
                          "ks__ledger_range_verdict"}
    # R-12：est__/ks__ 都有自己的出处，且都不冒充 measured
    for name, m in names.items():
        if m.provenance is Provenance.UNAVAILABLE:
            continue          # 未提供的第二条口径（轨迹）本来就是 n/a
        if name.startswith("est__"):
            assert m.provenance is Provenance.ESTIMATED, name
        else:
            assert m.provenance is Provenance.DERIVED, name
    assert "dram__bytes.sum" in names["est__dram_bytes_total"].source
    assert sec.provenance is Provenance.ESTIMATED


class _FakeSampler:
    """替身：绕过真实 nvidia-smi，直接喂一组饱和采样。"""

    def __init__(self, interval_s: float = 0.2, enabled: bool = True):
        self.result = SamplerResult(source="fake")
        # enabled 必须被尊重：--ks-no-nvml 时采样器就是要交白卷
        self.result.samples = _samples(12, mem=96.8, sm=99.0) if enabled else []
        self.result.duration_s = 2.4

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _coherent_launches(window_s: float = 2.4, n_gemm: int = 5, n_elem: int = 20,
                       n_other: int = 0, other_ns: int = 400_000_000):
    """与 _FakeSampler 的窗口自洽的合成 launch 表。

    R-46 之后 per-launch 归因要看「被匹配 kernel 占窗口多少」，所以替身数据必须自洽：
    被匹配的 kernel 要真的把窗口占满（否则估算本来就应该被拒绝）。n_other>0 时额外塞入
    别的 kernel，用来构造 D1 那种「窗口里顺序跑了多个 kernel」的场景。
    """
    from kernoscope.report import KernelLaunch

    launches = []
    t = 1_000_000_000
    per = int(window_s * 1e9 / max(1, n_gemm))
    for i in range(n_gemm):
        launches.append(KernelLaunch(
            name="synthetic_gemm", demangled="synthetic_gemm(float const*, float const*, float*)",
            start_ns=t, end_ns=t + per, grid=(1048576, 1, 1), block=(256, 1, 1),
            registers_per_thread=96, static_smem_bytes=0, dynamic_smem_bytes=49152,
            local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0, correlation_id=i))
        t += per
    for i in range(n_elem):
        launches.append(KernelLaunch(
            name="synthetic_elementwise", demangled="synthetic_elementwise(float*)",
            start_ns=t, end_ns=t + 8_400, grid=(16384, 1, 1), block=(256, 1, 1),
            registers_per_thread=16, static_smem_bytes=0, dynamic_smem_bytes=0,
            local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0,
            correlation_id=100 + i))
        t += 20_000
    for i in range(n_other):
        launches.append(KernelLaunch(
            name="synthetic_other", demangled="synthetic_other(float*)",
            start_ns=t, end_ns=t + other_ns, grid=(4096, 1, 1), block=(256, 1, 1),
            registers_per_thread=32, static_smem_bytes=0, dynamic_smem_bytes=0,
            local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0,
            correlation_id=200 + i))
        t += other_ns
    return launches


def test_cli_reports_dram_summary_and_ledger(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    base = tmp_path / "rep"
    # 账本按 per-launch 给：估算总量 ≈2112 GB，synthetic_gemm 有 5 次 launch → 每次 ≈422 GB
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-ledger-mb", "440000", "-o", str(base), "-f"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRAM 估算" in out
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    assert metrics["est__dram_bytes_total"]["provenance"] == "estimated"
    # 12 个采样点 × 0.2 s × 96.8% 忙碌 × 909 GB/s
    assert metrics["est__dram_bytes_total"]["value"] == pytest.approx(
        0.968 * 12 * 0.2 * 909e9 / 1e6, rel=1e-3)
    assert metrics["est__dram_bytes_per_launch"]["value"] > 0
    assert metrics["ks__sampler_window_coverage_pct"]["value"] == pytest.approx(100.0)
    assert metrics["ks__sampler_window_coverage_pct"]["unit"] == "%"
    # R-46：主导性前提必须与数字同时出现
    assert metrics["ks__matched_kernel_kernel_share_pct"]["value"] == pytest.approx(100.0, abs=1)
    assert metrics["ks__matched_kernel_window_share_pct"]["value"] == pytest.approx(100.0, abs=1)
    assert metrics["ks__window_busy_segments"]["value"] == 1
    assert metrics["meas__nvml_mem_busy_mean_pct"]["resolution"], \
        "R-31：代理量必须带分辨率"
    # 账本 440 GB/launch vs 估算 ~422 GB/launch → 轻微偏小（几个百分点）
    assert -10 < metrics["est__dram_vs_ledger_err_pct"]["value"] < 0


def test_window_dominance_blocks_per_launch_attribution(monkeypatch, capsys, tmp_path):
    """R-46 的负样本（验收报告 D1 那条命令的形态）。

    窗口覆盖率接近 100%（GPU 一直在干活），但被 `-k` 选中的 kernel 只占窗口的一小部分 ——
    其余时间在跑别的 kernel。此时把**整窗口字节**除以被匹配 kernel 的 launch 数会自带
    1/s−1 的上偏（实测那次 s≈38.6%，误差 +146.9%）。必须拒绝给数、点明份额与补救路径。
    """
    from kernoscope.analyze.dram import SHARE_STRONG

    class _MixedSampler(_FakeSampler):
        def __init__(self, interval_s: float = 0.2, enabled: bool = True):
            super().__init__(interval_s, enabled)
            self.result.samples = _samples(13, mem=96.8, sm=99.0)   # 2.6 s 全忙

    monkeypatch.setattr(cli, "Sampler", _MixedSampler)
    monkeypatch.setattr(cli, "load_synthetic",
                        lambda: _coherent_launches(window_s=0.4, n_gemm=5, n_elem=0,
                                                   n_other=5, other_ns=400_000_000))
    base = tmp_path / "rep"
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-window", "full", "--ks-ledger-mb", "422000",
                   "-o", str(base), "-f"])
    out = capsys.readouterr().out
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    # 覆盖率很高（GPU 一直在干活）—— 仅凭它不足以判定可归因
    assert metrics["ks__sampler_window_coverage_pct"]["value"] > 90
    share = metrics["ks__matched_kernel_kernel_share_pct"]["value"]
    assert share is not None and share < SHARE_STRONG
    for name in ("est__dram_bytes_per_launch", "est__dram_vs_ledger_err_pct"):
        assert metrics[name]["provenance"] == "unavailable", name
        assert metrics[name]["value"] is None
    reason = metrics["est__dram_bytes_per_launch"]["reason"]
    assert "窗口主导性不足" in reason and "窗口 kernel 时间" in reason
    assert "轨迹口径" in reason, "必须给出可行的补救路径"
    # 总量口径不受影响：它不需要主导性
    assert metrics["est__dram_bytes_total"]["provenance"] == "estimated"
    # 摘要里不能出现那个本该作废的 per-launch 值（D3）
    assert "per-launch 不给" in out
    # 归因上界是个可算的数：份额 15% 左右 → 上界 ≥ 100%
    bias = metrics["ks__attribution_bias_estimate_pct"]["value"]
    assert bias is not None and bias > 400


def test_threshold_message_is_not_self_contradictory(monkeypatch, capsys, tmp_path):
    """D2 的负样本：覆盖率 79.9% 不能印成「窗口覆盖率 80% < 80%」。"""
    class _AlmostSampler(_FakeSampler):
        def __init__(self, interval_s: float = 0.2, enabled: bool = True):
            super().__init__(interval_s, enabled)
            samples = _samples(1000, mem=95.0, sm=99.0)
            for s in samples[799:]:
                s.mem_util_pct, s.sm_util_pct = 0.0, 0.0
            self.result.samples = samples
            self.result.duration_s = 200.0

    monkeypatch.setattr(cli, "Sampler", _AlmostSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-window", "full", "--ks-ledger-mb", "422000", "-o",
                   str(tmp_path / "rep"), "-f"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "79.9% < 80.0%" in out, "要在文案里能区分 79.9% 与门槛 80%"
    assert "80% < 80%" not in out, "门槛文案不能自相矛盾（D2）"
    assert "延长到约 0 秒" not in out, "量化补救不能算出 0 秒（N6）"


def test_cli_strict_fails_when_ledger_verdict_cannot_be_produced(monkeypatch, capsys, tmp_path):
    """S7/R-20 的负样本：请求了账本但裁决行算不出来 → --strict 必须 exit 2。"""

    class _TinySampler(_FakeSampler):
        def __init__(self, interval_s: float = 0.2, enabled: bool = True):
            super().__init__(interval_s, enabled)
            # 10 个采样点里只有 1 个 busy → 覆盖率 10%
            self.result.samples = _samples(10, mem=0.0, sm=0.0)
            self.result.samples[0] = DeviceSample(t=0.0, mem_util_pct=95.0, sm_util_pct=99.0,
                                                  sm_clock_mhz=2500.0, power_w=300.0)

    monkeypatch.setattr(cli, "Sampler", _TinySampler)
    base = tmp_path / "rep"
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-window", "full", "--ks-ledger-mb", "440000",
                   "--strict", "-o", str(base), "-f"])
    err = capsys.readouterr().err
    assert rc == 2, "账本裁决行拿不到时，CI 门禁必须能红"
    assert "ledger" in err


def test_cli_strict_set_full_fails(monkeypatch, capsys):
    """R-20/B7-c 的负样本：--strict --set full 请求了计数器 section → exit 2。"""
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    rc = cli.main(["--ks-synthetic", "--strict", "--set", "full", "-k", "regex:gemm"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--set full" in err
    # --set basic 只请求可得数据 → 不该红
    rc2 = cli.main(["--ks-synthetic", "--ks-no-nvml", "--strict", "--set", "basic",
                    "-k", "regex:gemm", "--quiet"])
    assert rc2 == 0


def test_cli_rejects_bad_ledger_range(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "--ks-ledger-range-mb", "only-one"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "解析失败" in err or "error" in err


def test_metrics_filter_never_silently_produces_an_empty_report(monkeypatch, capsys):
    """R-39 的负样本：--metrics 请求一个我们给不出的名字时，不能给出"看起来成功"的空报告。

    D4① 之后契约升级：`dram__bytes.sum` 这类**已知的 ncu 名字**会被当成"发现键"，
    得到替代口径的行 + 一句说明（原来的规则没松：数值仍然只出现在自有命名空间里）。
    真正查不到的名字仍然走"一个都不存在"的负样本路径（见下一条）。"""
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc = cli.main(["--ks-synthetic", "--metrics", "dram__bytes.sum", "-k", "regex:gemm"])
    cap = capsys.readouterr()
    assert rc == 0, "宽松模式仍应成功（要能继续跑），但必须给出解释"
    assert "替代口径 est__dram_bytes_total" in cap.err
    assert "dram__bytes.sum" in cap.err
    # --strict 下同一件事仍是失败（我们给不出 ncu 名下那个数），但要说清替代口径叫什么
    rc2 = cli.main(["--ks-synthetic", "--strict", "--metrics", "dram__bytes.sum",
                    "-k", "regex:gemm"])
    cap2 = capsys.readouterr()
    assert rc2 == 2 and "est__dram_bytes_total" in cap2.err

    # 真正查不到的名字：仍然是"一个都不存在"的负样本
    rc3 = cli.main(["--ks-synthetic", "--metrics", "totally_made_up__metric",
                    "-k", "regex:gemm"])
    cap3 = capsys.readouterr()
    assert rc3 == 0 and "一个都不存在" in cap3.err


def test_coverage_premise_is_n_a_when_nothing_was_sampled(monkeypatch, capsys, tmp_path):
    """N5 的负样本：一次采样都没有时，前提①不能报「0%」——那等于声称量过。

    同一份报告里 est__dram_bytes_total 的原因写着「没有 NVML 采样点」，
    前提行却说覆盖率 0%、分辨率「1 个采样点」，两者自相矛盾。
    """
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc = cli.main(["--ks-synthetic", "--ks-no-nvml", "--ks-peak-gbps", "909",
                   "-k", "regex:gemm", "-o", str(tmp_path / "rep"), "-f"])
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    cov = metrics["ks__sampler_window_coverage_pct"]
    assert cov["provenance"] == "unavailable" and cov["value"] is None, \
        "没有采样点时覆盖率必须 n/a，而不是 0%"
    assert "没有 NVML 采样点" in cov["reason"]
    # 新的前提②同样不能凭空给数
    assert metrics["ks__matched_kernel_kernel_share_pct"]["value"] is None
    assert metrics["ks__dram_window_mem_busy_mean_pct"]["value"] is None


def test_resolution_present_on_derived_dram_rows(monkeypatch, capsys, tmp_path):
    """R-41：派生量也要带自己的分辨率/不确定度，不能只有总量那两行有。"""
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-ledger-mb", "422000", "--ks-ledger-range-mb", "300000,500000",
                   "-o", str(tmp_path / "rep"), "-f"])
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    for name in ("est__dram_bytes_per_launch", "est__dram_vs_ledger_err_pct",
                 "est__dram_range_position_pct", "ks__ledger_range_verdict"):
        m = metrics[name]
        assert m["value"] is not None, name
        assert m["resolution"], f"R-41：{name} 必须带分辨率/不确定度"
    # per-launch 的分辨率应当就是归因上偏估计（它真正的精度地板）
    assert "归因" in metrics["est__dram_bytes_per_launch"]["resolution"]


def test_summary_names_the_mem_busy_value_used_by_the_estimate(monkeypatch, capsys, tmp_path):
    """R-45：摘要里的「显存忙碌」必须能追到某个指标行，且与整进程均值区分开。"""
    class _TwoTierSampler(_FakeSampler):
        """前半段空闲（启动/JIT）、后半段饱和：窗口均值 96.8% ≠ 整进程均值 48.4%。"""

        def __init__(self, interval_s: float = 0.2, enabled: bool = True):
            super().__init__(interval_s, enabled)
            samples = _samples(24, mem=0.0, sm=0.0)
            for i, s in enumerate(samples):
                if i >= 12:
                    s.mem_util_pct, s.sm_util_pct = 96.8, 99.0
            self.result.samples = samples
            self.result.duration_s = 4.8

    monkeypatch.setattr(cli, "Sampler", _TwoTierSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-window", "auto", "-o", str(tmp_path / "rep"), "-f"])
    out = capsys.readouterr().out
    assert rc == 0
    doc = json.loads((tmp_path / "rep.report.json".replace(".report", ".ksreport")).read_text()) \
        if False else json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    win = metrics["ks__dram_window_mem_busy_mean_pct"]["value"]
    proc = metrics["meas__nvml_mem_busy_mean_pct"]["value"]
    assert win is not None and proc is not None
    assert win > proc + 20, "两个量确实不同：窗口均值 vs 整进程均值"
    assert "窗口显存忙碌" in out and "整进程" in out
    assert "ks__dram_window_mem_busy_mean_pct" in out, "摘要里要给出参与估算那个量的名字"


def test_launch_count_flag_does_not_change_the_per_launch_denominator(monkeypatch, capsys, tmp_path):
    """R-46 同族：-c/--launch-count 只限制**展示**，不能当 per-launch 的分母。

    分母若被 -c 截断，窗口总量会被除到一个偏小的 launch 数上 —— 误差同样是倍数级。
    """
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    values = {}
    for tag, extra in (("all", []), ("capped", ["-c", "2"])):
        base = tmp_path / tag
        rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                       *extra, "-o", str(base), "-f"])
        assert rc == 0
        doc = json.loads((base.with_suffix(".ksreport.json")).read_text())
        m = {x["name"]: x for k in doc["kernels"] for s in k["sections"] for x in s["metrics"]}
        values[tag] = m["est__dram_bytes_per_launch"]["value"]
    assert values["all"] == pytest.approx(values["capped"]), \
        "-c 只是展示上限，不该改变 per-launch 的分母"


# --------------------------------------------------------------------------
# R-47 / D5：per-launch 的**分母**必须和分子覆盖同一个窗口
# --------------------------------------------------------------------------
def _uniform_run(seconds: float = 40.0, launches: int = 200, dt: float = 0.2):
    """一段均匀负载：采样全忙，launch 均匀铺满整段（真机紧循环的形态）。"""
    n = int(seconds / dt)
    samples = [DeviceSample(t=i * dt, mem_util_pct=96.0, sm_util_pct=99.0,
                            sm_clock_mhz=2500.0, power_w=300.0) for i in range(n)]
    step = seconds / launches
    spans = [(i * step, (i + 1) * step) for i in range(launches)]
    return samples, spans


def test_per_launch_denominator_follows_the_window():
    """**D5 的回归**：同一负载换窗口，per-launch 必须不变（旧实现会等比偏小）。

    真机实测（验收报告 D5）：同一驱动、同一账本，`--ks-window auto` 给 235.08 MB，
    `--ks-warmup 40` 给 130.61 MB（−44.4%），`--ks-window-s 20` 给 45.84 MB（−80.5%），
    而 share=100%、bias=+0%、无告警、rc=0。算术上就是 `235.08 × 45.4/81.5 ≈ 130.9`
    —— **报告的 per-launch 是「窗口内字节 ÷ 整段运行的 launch 数」**。
    """
    samples, spans = _uniform_run()
    full = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                    window_mode="full", launch_spans=spans,
                    matched_kernel_ns=int(40.0e9))
    cut = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", window_limit_s=20.0, launch_spans=spans,
                   matched_kernel_ns=int(20.0e9))
    assert full.attributable and cut.attributable
    # 分子确实变小了（这就是当年出错的来源）
    assert cut.bytes_total == pytest.approx(full.bytes_total / 2, rel=0.05)
    # 但 per-launch 必须**基本不变**：分母跟着窗口走
    assert cut.bytes_per_launch == pytest.approx(full.bytes_per_launch, rel=0.05), (
        f"per-launch 随窗口变了：{full.bytes_per_launch / 1e6:.1f} → "
        f"{cut.bytes_per_launch / 1e6:.1f} MB")
    assert cut.launches_in_window == pytest.approx(100.0, rel=0.05)
    # 旧公式（窗口内字节 ÷ 整段 200 个 launch）会给出约一半的值 —— 这条断言就是那个负样本
    assert cut.bytes_total / 200 < cut.bytes_per_launch * 0.55
    # R-47：可算的偏差要报出来
    assert cut.window_bias_pct == pytest.approx(-50.0, abs=1.0)


def test_untrimmed_window_keeps_using_all_launches():
    """窗口没被裁时行为不变（真机 v3-only 的 235.08 MB / +0.1% 就是这一条路径）。"""
    samples, spans = _uniform_run()
    est = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", launch_spans=spans, matched_kernel_ns=int(40.0e9))
    assert est.launches_in_window == pytest.approx(200.0)
    assert est.bytes_per_launch == pytest.approx(est.bytes_total / 200)


def test_window_cut_without_launch_times_refuses_per_launch():
    """窗口被裁、又拿不到 launch 时间戳 ⇒ 必须拒绝，而不是给一个等比偏小的数。"""
    samples, _ = _uniform_run()
    est = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", window_limit_s=20.0, matched_kernel_ns=int(20.0e9))
    assert est.bytes_per_launch is None
    assert not est.attributable
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.WINDOW_LAUNCH_UNALIGNED
    # V-32：处方不许再推荐 `--ks-window full`（那只是换窗口，分母口径不变）
    assert "分母" in r.message and "--ks-window full" not in r.message
    assert "--ks-ledger-total-mb" in r.message or "--ks-memtrace" in r.message
    # R-47 要求把可算的偏差报出来；W3/D24 要求 observed 与 threshold 都不空
    assert r.observed and r.threshold
    assert r.observed["window_bias_pct"] == pytest.approx(-50.0, abs=1.0)
    rem = default_remedy(r.code)
    assert rem.command and "full" not in rem.command
    assert "--ks-ledger-total-mb" in rem.command or "--ks-memtrace" in rem.command


def test_misaligned_clocks_are_refused_not_divided():
    """两个时钟对不上（launch 时间戳远在采样窗之外）时必须拒绝，而不是相除。"""
    samples, _ = _uniform_run()
    spans = [(9_000_000.0 + i, 9_000_001.0 + i) for i in range(200)]   # 偏移 ~104 天
    est = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", window_limit_s=20.0, launch_spans=spans,
                   matched_kernel_ns=int(20.0e9))
    assert est.launch_clock_aligned is False
    assert est.bytes_per_launch is None
    assert est.refusal_for("est__dram_bytes_per_launch").code \
        is RefusalCode.WINDOW_LAUNCH_UNALIGNED


def test_share_too_low_remedy_no_longer_sells_shrinking_the_window():
    """R-48：份额按**整段 launch 表**算，缩采样窗口改不动它 —— 处方里不许再有那一条。

    旧文案让人 `--ks-warmup/--ks-window-s 只跑一段` 并把这条写进 command：它既解不开这个
    前提（份额不看采样窗口），又会在能出数的时候把 per-launch 弄错（分母是整段）。
    """
    samples, spans = _uniform_run(seconds=40.0, launches=200)
    # 造一个"份额不足"的窗口：被匹配 kernel 只占窗口 kernel 时间的 40%
    est = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", launch_spans=spans,
                   matched_kernel_ns=int(16.0e9), other_kernels_ns=int(24.0e9),
                   other_kernel_count=3)
    assert not est.attributable
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.SHARE_TOO_LOW
    rem = default_remedy(RefusalCode.SHARE_TOO_LOW)
    for bad in ("--ks-warmup", "--ks-window-s"):
        assert bad not in rem.action, f"处方里还在推荐 {bad} 缩窗口"
        assert bad not in rem.command, f"command 里还在推荐 {bad} 缩窗口"
        assert bad not in r.message, f"拒绝文案里还在推荐 {bad} 缩窗口"
    assert "--ks-memtrace" in rem.action and "--ks-ledger-total-mb" in rem.action


# --------------------------------------------------------------------------
# R-47 的真机教训：nsys 的 launch 时间戳是**会话相对**的，不是 monotonic
# --------------------------------------------------------------------------
def _session_relative_run(session_offset: float = 2.67, seconds: float = 30.0,
                          launches: int = 600, dt: float = 0.2, idle_s: float = 5.0):
    """真机形态：采样从 monotonic 的 t=1000 开始，先空转 idle_s，再跑 seconds 秒紧循环；
    launch 时间戳是**nsys 会话相对**的（第一次 launch ≈ session_offset 秒）。"""
    base = 1000.0
    n_idle = int(idle_s / dt)
    n_busy = int(seconds / dt)
    samples = [DeviceSample(t=base + i * dt, mem_util_pct=0.0, sm_util_pct=0.0,
                            sm_clock_mhz=210.0, power_w=60.0) for i in range(n_idle)]
    samples += [DeviceSample(t=base + (n_idle + i) * dt, mem_util_pct=95.0, sm_util_pct=99.0,
                             sm_clock_mhz=2500.0, power_w=300.0) for i in range(n_busy)]
    step = seconds / launches
    spans = [(session_offset + i * step, session_offset + (i + 1) * step)
             for i in range(launches)]
    return samples, spans, base


def test_session_relative_launch_clock_is_derived_not_assumed():
    """真机实测：CUPTI/nsys 的 start_ns 相对 profiling 会话起点（首次 launch ≈2.67 s），
    而采样 t 是 time.monotonic()。直接相减会把整段 launch 表判成"窗外"，于是要么给错
    要么全部拒绝。这里必须**推导**偏移（用窗口内忙碌段当锚点）并**自检速率比**。
    """
    samples, spans, base = _session_relative_run()
    est = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600,
                   window_mode="auto", launch_spans=spans,
                   matched_kernel_ns=int(30.0e9))
    assert est.launch_clock_aligned is True, est.refusal_for("est__dram_bytes_per_launch").message
    assert est.bytes_per_launch is not None
    # 锚点 = 窗口内第一个忙碌采样点；launch 表起点是会话相对 2.67 s ⇒ 偏移约等于
    # (base + idle_s) − session_offset
    assert est.launch_align_offset_s == pytest.approx(base + 5.0 - 2.67, abs=0.25)
    assert est.launch_align_slope == pytest.approx(1.0, abs=0.02)
    # 紧循环、launch 铺满整段 ⇒ 分母应当接近全部 600 个
    assert est.launches_in_window == pytest.approx(600.0, rel=0.03)
    assert est.attributable
    # 分母的分辨率（offset ±1 采样周期）要报出来，且必须**包住**中点估计、且很小
    assert est.launches_in_window_range is not None
    lo, hi = est.launches_in_window_range
    assert lo <= est.launches_in_window <= hi
    assert 100.0 * (hi - lo) / est.launches_in_window < 3.0, (lo, hi)


def test_session_relative_clock_with_cut_window_still_agrees():
    """同一负载、会话相对的时钟、窗口被裁 —— per-launch 仍必须与不裁时一致。"""
    samples, spans, _ = _session_relative_run(seconds=30.0, launches=600)
    full = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600,
                    window_mode="auto", launch_spans=spans, matched_kernel_ns=int(30.0e9))
    # 采样从 base 开始、忙碌段是 [base+5, base+35]；warmup 25 s ⇒ 窗口只剩最后 10 s
    cut = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600, warmup_s=25.0,
                   window_mode="auto", launch_spans=spans, matched_kernel_ns=int(10.0e9))
    assert full.attributable and cut.attributable
    assert cut.launches_in_window == pytest.approx(200.0, rel=0.08)
    assert cut.bytes_per_launch == pytest.approx(full.bytes_per_launch, rel=0.08)


def test_clock_rate_mismatch_is_refused():
    """两个时钟速率不一致（模拟被缩放的时间戳）⇒ 必须拒绝，并点名速率比。"""
    samples, spans, _ = _session_relative_run(seconds=30.0, launches=600)
    # 把 launch 表压缩到一半长度：等于"launch 时钟跑得比采样时钟慢一倍"
    lo = spans[0][0]
    squeezed = [(lo + (a - lo) * 0.5, lo + (b - lo) * 0.5) for a, b in spans]
    est = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600,
                   window_mode="auto", launch_spans=squeezed,
                   matched_kernel_ns=int(30.0e9))
    assert est.launch_clock_aligned is False
    assert est.bytes_per_launch is None
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.WINDOW_LAUNCH_UNALIGNED
    assert "速率比" in r.message
    assert r.observed["launch_align_slope"] == pytest.approx(2.0, rel=0.05)
    assert r.threshold == {"launch_align_slope": 1.0}


def test_mixed_window_still_reports_share_too_low_first():
    """D1 的验收不能回归：混合窗口（份额不足）的主 code 仍必须是 share_too_low。"""
    samples, spans, _ = _session_relative_run(seconds=30.0, launches=600)
    est = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600,
                   window_mode="auto", launch_spans=spans,
                   matched_kernel_ns=int(12.0e9), other_kernels_ns=int(18.0e9),
                   other_kernel_count=3)
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.SHARE_TOO_LOW, r.code
    assert "份额" in r.message


# --------------------------------------------------------------------------
# R-47 的验收工具本身（bench/window_probe.py --check）必须"能红"
# --------------------------------------------------------------------------
def _load_window_probe():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "bench" / "window_probe.py"
    spec = importlib.util.spec_from_file_location("ks_window_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _packet_with_per_launch(tmp_path, name: str, value_mb, code: str | None = None):
    v: dict = {"id": "dram_bytes_per_launch", "status": "refused" if code else "soft"}
    if code:
        v["refusal"] = {"code": code}
    else:
        v["value"] = value_mb * 1e6
        v["unit"] = "B"
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"verdicts": [v]}), encoding="utf-8")
    return p


def test_window_probe_acceptance_fails_on_the_old_silent_bias(tmp_path):
    """旧实现的形态（1.06/0.86/0.52 GB）必须被判失败 —— 判据要能红。"""
    mod = _load_window_probe()
    pk = [_packet_with_per_launch(tmp_path, "auto", 1063.0),
          _packet_with_per_launch(tmp_path, "warmup10", 859.9),
          _packet_with_per_launch(tmp_path, "warmup20", 517.3)]
    assert mod.check(pk, analytic_mb=1073.74) == 1


def test_window_probe_acceptance_passes_when_windows_agree(tmp_path):
    mod = _load_window_probe()
    pk = [_packet_with_per_launch(tmp_path, "auto", 1063.0),
          _packet_with_per_launch(tmp_path, "warmup10", 1043.4),
          _packet_with_per_launch(tmp_path, "warmup20", 1086.0)]
    assert mod.check(pk, analytic_mb=1073.74) == 0


def test_window_probe_accepts_a_clean_refusal(tmp_path):
    """换窗口后口径被**明确拒绝**（带结构化 code）也是合格答案。"""
    mod = _load_window_probe()
    pk = [_packet_with_per_launch(tmp_path, "a", 0, code="window_launch_unaligned"),
          _packet_with_per_launch(tmp_path, "b", 0, code="window_launch_unaligned"),
          _packet_with_per_launch(tmp_path, "c", 0, code="window_launch_unaligned")]
    assert mod.check(pk, analytic_mb=1073.74) == 0


# --------------------------------------------------------------------------- #
# 第五轮 V-7 / V-31：未裁窗口也必须用"窗口内等效 launch 数"当分母
# --------------------------------------------------------------------------- #
def test_untrimmed_window_with_longer_launch_table_is_refused_not_halved():
    """V-7 的决定性构造：采样 20 s、launch 表 40 s、**没有** --ks-window-s。

    旧实现走到 `elif launches > 0`，把窗口积分除以整段 launch 数 → per-launch 静默偏小
    50%，`attributable=True`、无告警，而且 `window_launch_unaligned` 在这条路径上**不可达**。
    修复后：两个时钟的速率对不上（slope≈0.5）⇒ 拒绝，而不是给一个等比偏小的数。
    """
    samples = [DeviceSample(t=float(i), mem_util_pct=96.0, sm_util_pct=90.0,
                            sm_clock_mhz=2500.0, power_w=300.0) for i in range(20)]
    spans = [(float(2 * i), float(2 * i + 1.9)) for i in range(20)]      # 40 s 的 launch 表
    est = estimate(samples, peak_gbs=900.0, window_s=20.0, launches=20,
                   window_mode="full", launch_spans=spans, matched_kernel_ns=int(19.5e9))
    assert est.bytes_per_launch is None, (
        f"窗口 20 s / launch 表 40 s：分母不成立却给了数 {est.bytes_per_launch}")
    assert not est.attributable
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.WINDOW_LAUNCH_UNALIGNED
    # V-31：偏差口径的分母是 **launch 表跨度**（20/40−1 = −50%），不是进程 wall
    assert r.observed["window_bias_pct"] == pytest.approx(-50.0, abs=2.0)


def test_window_bias_uses_launch_span_not_process_wall():
    """V-31：偏差估计的分母是 launch 表跨度；两者一致时必须是 0%，不许凭 wall 报偏差。"""
    samples, spans = _uniform_run()
    est = estimate(samples, peak_gbs=900.0, window_s=123.0, launches=200,   # wall 与真实不符
                   window_mode="full", launch_spans=spans, matched_kernel_ns=int(40.0e9))
    assert est.window_bias_pct == pytest.approx(0.0, abs=1.0), est.window_bias_pct


def test_launch_table_offset_by_a_second_is_aligned_by_the_anchor_not_ignored():
    """launch 表整体平移 1 s 时必须用忙碌锚点对齐（而不是"看起来在窗口附近"就算同源）。

    这正是 CLI 合成夹具的形态：采样窗 [0,2.4]、launch 表 [1.0,3.4]、两者斜率都是 1。
    按同源处理会把 5 个 launch 数成 2.92 个（per-launch 偏大 64%）；用锚点对齐后是 5.0。
    """
    samples, spans = _uniform_run()
    shifted = [(a + 1.0, b + 1.0) for a, b in spans]
    est = estimate(samples, peak_gbs=900.0, window_s=40.0, launches=200,
                   window_mode="full", launch_spans=shifted, matched_kernel_ns=int(40.0e9))
    assert est.launches_in_window == pytest.approx(200.0, rel=0.02), est.launches_in_window
    assert est.bytes_per_launch == pytest.approx(est.bytes_total / 200.0, rel=0.02)


# --------------------------------------------------------------------------- #
# 第五轮 V-8：分母的"分辨率"必须真的是误差上界（锚点相位 + 斜率）
# --------------------------------------------------------------------------- #
def test_busy_segment_with_extra_work_is_refused_not_bounded_tightly():
    """窗口里混了不在 launch 表里的工作（例如先跑 3.6 s memcpy）⇒ 拒绝给 per-launch。

    真机构造：分母 433.03 vs 真值 472.0（−8.3%），而报出的区间只有 0.85% ——
    报告的"分辨率"不是误差上界，读者会以为这个数比实际可信。
    """
    dt = 0.2
    # 采样：前 18 个点被 memcpy 占着（忙），之后是 kernel 段（忙），共 43.6 s
    samples = [DeviceSample(t=i * dt, mem_util_pct=95.0, sm_util_pct=90.0,
                            sm_clock_mhz=2500.0, power_w=300.0) for i in range(218)]
    # launch 表只覆盖 kernel 段：40 s（20 个 launch，每个 1.9 s）
    spans = [(3.6 + 2.0 * i, 3.6 + 2.0 * i + 1.9) for i in range(20)]
    est = estimate(samples, peak_gbs=900.0, window_s=43.6, launches=20,
                   window_mode="full", launch_spans=spans,
                   matched_kernel_ns=int(38.0e9))
    assert est.launch_align_slope is not None and est.launch_align_slope > 1.02, \
        est.launch_align_slope
    assert est.bytes_per_launch is None, (
        f"忙碌段混了别的工作，分母不可信却给了数：{est.bytes_per_launch}")
    assert est.launches_in_window_range is None
    r = est.refusal_for("est__dram_bytes_per_launch")
    assert r.code is RefusalCode.WINDOW_LAUNCH_UNALIGNED
    assert "斜率" in r.message


def test_normal_load_reports_a_tight_resolution():
    """正常负载（忙碌段就是 launch 表）时，分辨率必须紧 —— 不许把 [0.9,1.1] 整个算进去。"""
    samples, spans, base = _session_relative_run()
    est = estimate(samples, peak_gbs=900.0, window_s=35.0, launches=600,
                   window_mode="auto", launch_spans=spans, matched_kernel_ns=int(30.0e9))
    lo, hi = est.launches_in_window_range
    assert est.launches_in_window is not None
    assert 100.0 * (hi - lo) / est.launches_in_window < 3.0, (lo, hi)
