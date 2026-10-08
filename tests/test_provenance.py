"""provenance 四规则的强制测试 —— 这是项目的核心不变量。"""

import pytest

from kernoscope.refusals import RefusalCode
from kernoscope.provenance import Metric, Provenance, ProvenanceError, Section


def test_r1_unavailable_must_be_none_and_have_reason():
    with pytest.raises(ProvenanceError, match="R1"):
        Metric("sm__throughput.avg.pct_of_peak_sustained_elapsed", 42.0, Provenance.UNAVAILABLE,
               reason="counters")
    with pytest.raises(ProvenanceError, match="R1"):
        Metric("x", None, Provenance.UNAVAILABLE)  # 缺 reason
    m = Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed",
                                     "counters restricted", code=RefusalCode.NO_COUNTERS)
    assert m.value is None
    assert m.render_value() == "n/a"


def test_r2_counter_namespace_requires_exact():
    # 计数器命名空间下，任何非 exact 的取值都必须被拒绝
    for prov in (Provenance.DERIVED, Provenance.MEASURED, Provenance.ESTIMATED):
        with pytest.raises(ProvenanceError, match="R2"):
            Metric("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0, prov)


def test_r2_hardened_counter_names_cannot_carry_any_value():
    """R-11 的负样本（需求方 §3 R-11）：连 `Metric.exact` 也不能在计数器名下填数。

    旧实现只禁止"counter 名 + 非 exact"，于是 `Metric.exact("sm__…", 1.0)` 合法 ——
    而那正是"在 ncu 指标名下面填一个数"的缺口。现在计数器名必须登记在
    EXACT_COUNTER_ALLOWLIST（当前为空），所以有值的计数器指标在类型层面不可构造。
    """
    with pytest.raises(ProvenanceError, match="no counter backend"):
        Metric.exact("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0)
    with pytest.raises(ProvenanceError, match="no counter backend"):
        Metric.exact("dram__bytes.sum", 1024)
    # 正确姿势：n/a + 原因
    assert Metric.unavailable("dram__bytes.sum", "counters restricted",
                                  code=RefusalCode.NO_COUNTERS).value is None
    # 将来真接上计数器时：登记白名单即可（这里验证机制本身可用）
    from kernoscope import provenance

    try:
        provenance.EXACT_COUNTER_ALLOWLIST = frozenset({"dram__bytes.sum"})
        assert Metric.exact("dram__bytes.sum", 1024).value == 1024
    finally:
        provenance.EXACT_COUNTER_ALLOWLIST = frozenset()


def test_r12_meas_prefix_must_be_a_real_measurement():
    """R-12：`meas__` 承诺"这是实测读数"，出处就必须是 measured。"""
    assert Metric.measured("meas__nvml_mem_busy_mean_pct", 3.2).value == 3.2
    for prov, ctor in (("estimated", Metric.estimated), ("derived", Metric.derived)):
        with pytest.raises(ProvenanceError, match="R-12"):
            ctor("meas__sampler_duty_pct", 60.0)
    # 由实测样本再算出来的统计量请用 ks__ + derived
    assert Metric.derived("ks__sampler_window_coverage_pct", 60.0).value == 60.0


def test_r31_resolution_travels_with_the_number():
    m = Metric.measured("meas__nvml_mem_busy_mean_pct", 0.0, unit="%",
                        resolution="1 %（整数百分比）")
    assert m.resolution.startswith("1 %")
    assert "resolution" in m.render_line()


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
    s.add(Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed", "counters",
                                 code=RefusalCode.NO_COUNTERS))
    assert s.provenance is Provenance.ESTIMATED   # 有值指标里最保守的那个
    assert s.n_unavailable == 1
    assert s.has_values

    only_na = Section("SchedulerStats").add(
        Metric.unavailable("ks__schedulerstats_status", "counters",
                                 code=RefusalCode.NO_COUNTERS))
    assert only_na.provenance is Provenance.UNAVAILABLE
    assert not only_na.has_values

    s2 = Section("LaunchStats").add(Metric.exact("launch__registers_per_thread", 8))
    assert s2.provenance is Provenance.EXACT


def test_only_exact_is_cross_comparable():
    assert Provenance.EXACT.is_numeric_trustworthy
    for p in (Provenance.DERIVED, Provenance.MEASURED, Provenance.ESTIMATED,
              Provenance.UNAVAILABLE):
        assert not p.is_numeric_trustworthy
