"""provenance 四规则的强制测试 —— 这是项目的核心不变量。"""

import pytest

from kernoscope.provenance import Metric, Provenance, ProvenanceError, Section


def test_r1_unavailable_must_be_none_and_have_reason():
    with pytest.raises(ProvenanceError, match="R1"):
        Metric("sm__throughput.avg.pct_of_peak_sustained_elapsed", 42.0, Provenance.UNAVAILABLE,
               reason="counters")
    with pytest.raises(ProvenanceError, match="R1"):
        Metric("x", None, Provenance.UNAVAILABLE)  # 缺 reason
    m = Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed", "counters restricted")
    assert m.value is None
    assert m.render_value() == "n/a"


def test_r2_counter_namespace_requires_exact():
    # 计数器命名空间下，任何非 exact 的取值都必须被拒绝
    for prov in (Provenance.DERIVED, Provenance.MEASURED, Provenance.ESTIMATED):
        with pytest.raises(ProvenanceError, match="R2"):
            Metric("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0, prov)
    # exact 允许（虽然我们实际拿不到，规则本身允许声明）
    assert Metric.exact("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0).value == 1.0


def test_r3_non_exact_must_use_our_prefix():
    with pytest.raises(ProvenanceError, match="R3"):
        Metric("throughput_pct", 50.0, Provenance.ESTIMATED)  # 名字像 ncu 但不是我们的前缀
    assert Metric.estimated("est__throughput_pct", 50.0).value == 50.0
    assert Metric.measured("meas__achieved_occupancy_pct", 81.0).value == 81.0
    assert Metric.derived("model__x", 1.0).value == 1.0
    # exact 不受 R3 限制（launch__ / gpu__time_duration 这类我们同源可得）
    assert Metric.exact("launch__registers_per_thread", 96).value == 96


def test_r4_gpu_duration_only_exact():
    with pytest.raises(ProvenanceError, match="R4"):
        Metric("gpu__time_duration.sum", 1.0, Provenance.MEASURED)
    assert Metric.exact("gpu__time_duration.sum", 1.0).unit == ""


def test_section_provenance_ignores_na_rows():
    """含 exact 数据 + 一条 n/a 的 section，头部应显示 exact（n/a 单独计数）。"""
    s = Section("SpeedOfLight")
    s.add(Metric.exact("gpu__time_duration.sum", 1.0))
    s.add(Metric.estimated("est__achieved_tflops", 2.0))
    s.add(Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed", "counters"))
    assert s.provenance is Provenance.ESTIMATED   # 有值指标里最保守的那个
    assert s.n_unavailable == 1
    assert s.has_values

    only_na = Section("SchedulerStats").add(
        Metric.unavailable("ks__schedulerstats_status", "counters"))
    assert only_na.provenance is Provenance.UNAVAILABLE
    assert not only_na.has_values

    s2 = Section("LaunchStats").add(Metric.exact("launch__registers_per_thread", 8))
    assert s2.provenance is Provenance.EXACT


def test_only_exact_is_cross_comparable():
    assert Provenance.EXACT.is_numeric_trustworthy
    for p in (Provenance.DERIVED, Provenance.MEASURED, Provenance.ESTIMATED,
              Provenance.UNAVAILABLE):
        assert not p.is_numeric_trustworthy
