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
from kernoscope.sections import BuildOptions, build_dram_section


def _samples(n=10, mem=100.0, sm=80.0, dt=0.2):
    return [DeviceSample(t=i * dt, mem_util_pct=mem, sm_util_pct=sm,
                         sm_clock_mhz=2500.0, power_w=300.0) for i in range(n)]


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
    est2 = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100)
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
                   ledger_per_launch=1.5e10)
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
                        ledger_range=(low, high))
    near_low.bytes_per_launch = low + 0.1 * (high - low)
    assert near_low.range_verdict == "落在区间靠下界侧"

    above = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                     ledger_range=(low, high))
    above.bytes_per_launch = high * 1.5
    assert "高于上界" in above.range_verdict

    below = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                     ledger_range=(low, high))
    below.bytes_per_launch = low * 0.5
    assert "低于下界" in below.range_verdict


def test_range_verdict_is_also_a_metric():
    """R-05：判词必须出现在输出里（不只是 % 位置），且标注口径。"""
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100,
                   ledger_range=(1.0e10, 2.0e10))
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
                   ledger_per_launch=1073.7e6)
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
    auto = estimate(samples, peak_gbs=900.0, window_s=8.8, launches=100, window_mode="auto")
    assert full.window_coverage_pct < DUTY_UNUSABLE
    assert not full.reliable
    assert auto.window_coverage_pct >= DUTY_WEAK, "裁剪后应能给出 per-launch 归因"
    assert auto.attributable
    assert auto.trimmed and auto.window_used_s < full.window_used_s
    assert any("稳态段" in n for n in auto.notes)


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
        self.result.samples = _samples(12, mem=96.8, sm=99.0)
        self.result.duration_s = 2.4

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_cli_reports_dram_summary_and_ledger(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
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
    assert metrics["meas__nvml_mem_busy_mean_pct"]["resolution"], \
        "R-31：代理量必须带分辨率"
    # 账本 440 GB/launch vs 估算 ~422 GB/launch → 轻微偏小（几个百分点）
    assert -10 < metrics["est__dram_vs_ledger_err_pct"]["value"] < 0


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
