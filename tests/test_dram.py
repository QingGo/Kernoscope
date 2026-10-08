"""DRAM 流量估算（NVML 显存忙碌 × 实测峰值带宽）与 DramTraffic section。"""

import json

import pytest

import kernoscope.cli as cli
from kernoscope.analyze.dram import estimate
from kernoscope.backends.nvml import DeviceSample, SamplerResult
from kernoscope.provenance import Provenance
from kernoscope.sections import BuildOptions, build_dram_section


def _samples(n=10, mem=100.0, sm=80.0, dt=0.2):
    return [DeviceSample(t=i * dt, mem_util_pct=mem, sm_util_pct=sm,
                         sm_clock_mhz=2500.0, power_w=300.0) for i in range(n)]


def test_estimate_bytes_duty_throughput():
    est = estimate(_samples(10, mem=100, sm=80), peak_gbs=900.0, window_s=2.0, launches=0)
    # 10 个采样点 × 0.2 s 全忙 = 2.0 s 忙碌 → 2.0 s × 900 GB/s
    assert est.bytes_total == pytest.approx(1.8e12)
    assert est.busy_pct_mean == pytest.approx(100.0)
    assert est.throughput_pct_of_peak == pytest.approx(100.0)
    assert est.duty_pct == pytest.approx(100.0)
    assert est.reliable


def test_per_launch_requires_launch_count():
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=0)
    assert est.bytes_per_launch is None
    est2 = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=100)
    assert est2.bytes_per_launch == pytest.approx(est.bytes_total / 100)
    assert est2.attributable, "满占空比 + 单一 kernel → 可归因"


def test_low_duty_blocks_per_launch_attribution_only():
    """占空比 60%：总量仍可给，但不得给出 per-launch 归因。"""
    samples = _samples(10, mem=90.0, sm=90.0)
    for s in samples[6:]:                     # 后 4 个采样点置为空闲 → 占空比 60%
        s.mem_util_pct, s.sm_util_pct = 0.0, 0.0
    est = estimate(samples, peak_gbs=900.0, window_s=2.0, launches=10)
    assert est.reliable and not est.attributable
    sec = build_dram_section(BuildOptions(dram=est))
    by_name = {m.name: m for m in sec.metrics}
    assert by_name["est__dram_bytes_total"].provenance is Provenance.ESTIMATED
    assert by_name["est__dram_bytes_per_launch"].provenance is Provenance.UNAVAILABLE
    assert "归因" in (by_name["est__dram_bytes_per_launch"].reason or "")


def test_l2_resident_workload_is_not_reported_as_dram_bound():
    """L2 驻留：mem_util 只有 ~45%，估算必须如实反映（不要硬说打满 DRAM）。"""
    est = estimate(_samples(10, mem=45.0, sm=95.0), peak_gbs=900.0, window_s=2.0)
    assert est.busy_pct_mean == pytest.approx(45.0)
    assert est.bytes_total == pytest.approx(0.9 * 900e9)


def test_low_duty_is_marked_unreliable():
    """占空比过低（宿主间隙/空闲）时代理会被稀释 → 必须判为不可用。"""
    est = estimate(_samples(10, mem=6.3, sm=0.0), peak_gbs=900.0, window_s=2.0)
    assert not est.reliable
    assert any("饱和" in n for n in est.notes)
    sec = build_dram_section(BuildOptions(dram=est))
    assert sec is not None
    m = sec.metrics[0]
    assert m.provenance is Provenance.UNAVAILABLE and m.value is None
    assert "饱和" in (m.reason or "")


def test_missing_peak_bandwidth_marks_unreliable():
    est = estimate(_samples(), peak_gbs=0.0, window_s=2.0)
    assert not est.reliable
    assert any("峰值带宽" in n for n in est.notes)
    sec = build_dram_section(BuildOptions(dram=est))
    assert sec.metrics[0].provenance is Provenance.UNAVAILABLE


def test_ledger_error_and_range_verdict():
    # 估算 1.8e12 B，账本 1.5e12 B → 误差 +20%
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, ledger_bytes=1.5e12)
    assert est.ledger_err_pct == pytest.approx(20.0)

    # 通用场景：账本给的是区间（下界=理想流量，上界=假设重读全回 HBM）
    low, high = 100e6, 400e6
    near_low = estimate(_samples(), peak_gbs=900.0, window_s=2.0, ledger_range=(low, high))
    near_low.bytes_total = low + 0.1 * (high - low)
    assert near_low.range_verdict == "落在区间靠下界侧"

    above = estimate(_samples(), peak_gbs=900.0, window_s=2.0, ledger_range=(low, high))
    above.bytes_total = high * 1.5
    assert "高于上界" in above.range_verdict

    below = estimate(_samples(), peak_gbs=900.0, window_s=2.0, ledger_range=(low, high))
    below.bytes_total = low * 0.5
    assert "低于下界" in below.range_verdict


def test_section_metrics_are_estimated_with_ncu_cross_reference():
    est = estimate(_samples(), peak_gbs=900.0, window_s=2.0, launches=50,
                   ledger_bytes=1.8e12, ledger_range=(1.0e12, 2.0e12))
    sec = build_dram_section(BuildOptions(dram=est))
    names = {m.name: m for m in sec.metrics}
    assert set(names) >= {"est__dram_bytes_total", "est__dram_bytes_per_launch",
                          "est__dram_throughput_pct_of_peak", "meas__sampler_duty_pct",
                          "est__dram_vs_ledger_err_pct", "est__dram_range_position_pct"}
    assert all(m.provenance is Provenance.ESTIMATED for m in sec.metrics)
    # note 里说明与 ncu 指标的对应关系（不占用 ncu 的指标名）
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
    # 账本给成与估算同量级（2110 GB），这样误差百分比才有意义
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                   "--ks-ledger-mb", "2200000", "-o", str(base), "-f"])
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
    assert metrics["meas__sampler_duty_pct"]["value"] == pytest.approx(100.0)
    assert metrics["meas__sampler_duty_pct"]["unit"] == "%"
    # 账本 2200 GB vs 估算 ~2112 GB → 轻微偏小（-4% 量级）
    assert -10 < metrics["est__dram_vs_ledger_err_pct"]["value"] < 0


def test_cli_rejects_bad_ledger_range(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    rc = cli.main(["--ks-synthetic", "--ks-peak-gbps", "909", "--ks-ledger-range-mb", "only-one"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "解析失败" in err or "error" in err
