"""真值对拍（I1）与 `.ncu-rep` 导出（I2/M3）的契约测试。

这一族功能的失败模式很特别：**它不会崩，它会很安静地骗人**。所以测试的重点是
"什么时候必须拒绝给数"和"写进 `.ncu-rep` 的东西能不能被署名、会不会被误当计数器读数"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernoscope.analyze.truth import (PAIRS, compare, comparison_metrics, match_action,
                                      truth_metrics)
from kernoscope.backends.ncu_rep import NcuRepAction, NcuRepData
from kernoscope.ncrep.substitute import (ATTRIBUTION_MARK, SubstituteRefused, assert_writable,
                                         build_plan, render_comment, select_metrics,
                                         write_substitute)
from kernoscope.provenance import COUNTER_NAMESPACES, Metric, Provenance


# --------------------------------------------------------------------------
# 测试替身（与真实 dataclass 同构）
# --------------------------------------------------------------------------
class Launch:
    def __init__(self, grid=(12288, 1, 1), block=(256, 1, 1)):
        self.grid, self.block = grid, block


class Section:
    def __init__(self, name):
        self.name, self.metrics = name, []

    def add(self, m):
        self.metrics.append(m)


class KernelReport:
    def __init__(self, name, metrics=(), launches=None):
        self.name = name
        self.launches = launches if launches is not None else [Launch()]
        self.sections = []
        if metrics:
            s = Section("Duration")
            for m in metrics:
                s.add(m)
            self.sections.append(s)


def _action(kernel="addConstDouble(int, double *, double, double *)", **metrics):
    base = {"gpu__time_duration.sum": 89728.0, "launch__registers_per_thread": 16.0,
            "lts__t_sector_hit_rate.pct": 50.0}
    base.update(metrics)
    return NcuRepAction(kernel_name=kernel, grid=(12288, 1, 1), block=(256, 1, 1),
                        metrics=base, rules=[("Issue Slot Utilization", 9.4)],
                        function_name=kernel.split("(")[0], demangled_name=kernel)


def _data(*actions):
    return NcuRepData(path="/tmp/x.ncu-rep", actions=list(actions), tool_version="2025.3.1")


# --------------------------------------------------------------------------
# 对拍
# --------------------------------------------------------------------------
def test_comparison_uses_the_declared_unit_conversion():
    """真值 ns → 我们的 ms：这条系数是本模块最容易错的地方（10⁶ 级），必须钉住。"""
    ours = {"ks__launch_time_mean": 0.09}          # ms
    cmp = compare(_data(_action()), ours, ours_by_name=ours)
    row = next(r for r in cmp.rows if r.ours_name == "ks__launch_time_mean")
    assert row.truth == pytest.approx(89728.0 * 1e-6, rel=1e-12)      # = 0.089728 ms
    assert row.rel_err_pct == pytest.approx(100.0 * (0.09 - 0.089728) / 0.089728, rel=1e-9)
    assert row.rel_err_pct > 0                                        # 我们给的值略大
    assert "ns" in row.basis and "ms" in row.basis


def test_missing_truth_metric_gives_a_reason_not_a_number():
    """报告里没有这个指标（不同架构/采集配置）—— 不许拿别的指标顶替。"""
    data = _data(NcuRepAction("k", (1, 1, 1), (1, 1, 1), {"launch__registers_per_thread": 32.0},
                              [], "", ""))
    cmp = compare(data, {"ks__launch_time_mean": 0.09}, ours_by_name={"ks__launch_time_mean": 0.09})
    row = next(r for r in cmp.rows if r.truth_name == "gpu__time_duration.sum")
    assert row.rel_err_pct is None and "没有这个指标" in row.reason
    assert cmp.max_abs_err_pct is None


def test_missing_our_value_gives_a_reason_too():
    cmp = compare(_data(_action()), {}, ours_by_name={})
    row = next(r for r in cmp.rows if r.ours_name == "ks__launch_time_mean")
    assert row.rel_err_pct is None and "我们的报告里没有" in row.reason


def test_zero_truth_refuses_the_percentage():
    """真值为 0 时相对偏差无定义（除以 0）—— 这是"基线为 0"的老坑，不许返回 inf/100%。"""
    act = _action(**{"gpu__time_duration.sum": 0.0})
    cmp = compare(_data(act), {"ks__launch_time_mean": 0.1},
                  ours_by_name={"ks__launch_time_mean": 0.1})
    row = next(r for r in cmp.rows if r.ours_name == "ks__launch_time_mean")
    assert row.rel_err_pct is None and "除以 0" in row.reason


def test_identity_pairs_are_marked_as_self_checks():
    """"两边同源应当逐位相同"的那两行是解析路径自检 —— 它们的存在本身是设计意图。"""
    identity = {ours for truth, ours, factor, _ in PAIRS if factor == 1.0
                and truth.startswith("launch__")}
    assert identity, "至少要有一条 factor=1 的自检行"
    cmp = compare(_data(_action()), {"launch__registers_per_thread": 16.0},
                  ours_by_name={"launch__registers_per_thread": 16.0})
    row = next(r for r in cmp.rows if r.ours_name == "launch__registers_per_thread")
    assert row.rel_err_pct == pytest.approx(0.0)
    assert "自检" in row.basis


def test_truth_metrics_carry_the_ncu_prefix_and_are_real_exact_metrics():
    """`ncu__` 前缀不是"计数器命名空间"：它表示"与 ncu 同源的导入值"，因此不触发 R2 允许表。"""
    rows = truth_metrics(_data(_action()))
    assert rows and all(n.startswith("ncu__") for n, _, _ in rows)
    for name, value, source in rows:
        m = Metric.exact(name, value, source=source)
        assert m.provenance is Provenance.EXACT
    # 反面：真值去掉前缀就会落进计数器命名空间，被 R2 拦下 —— 这正是前缀存在的理由
    with pytest.raises(Exception):
        Metric.exact("dram__bytes.sum", 123.0)
        # 若将来把 dram__bytes.sum 登记进 EXACT_COUNTER_ALLOWLIST，这条会失效，
        # 那时应改成断言 allowlist 的显式登记（见 provenance.EXACT_COUNTER_ALLOWLIST）


def test_comparison_metric_names_are_namespaced_and_constructible():
    cmp = compare(_data(_action()), {"ks__launch_time_mean": 0.09},
                  ours_by_name={"ks__launch_time_mean": 0.09})
    for name, value, unit, note in comparison_metrics(cmp):
        assert name.startswith("ks__truth_vs_substitute_pct[")
        assert unit == "%"
        if value is not None:
            assert Metric.derived(name, value, unit, note=note).provenance is Provenance.DERIVED


def test_action_matching_is_conservative():
    """`addConstDouble` 与 `addConstDouble3` 绝不许互相匹配（对拍表里放错 kernel 的数 = 假百分比）。"""
    data = _data(_action("addConstDouble3(int, double3 *, double3 *)",
                         function_name="addConstDouble3"))
    assert match_action(data, "addConstDouble(int, double *, double, double *)") is None
    assert match_action(data, "addConstDouble3(int, double3 *, double3 *)") is not None


# --------------------------------------------------------------------------
# 写 .ncu-rep：选择与署名
# --------------------------------------------------------------------------
def _kr_with(metrics, packet=None, launches=None):
    kr = KernelReport("addConstDouble(int, double *, double, double *)", metrics, launches)
    return kr


def _m(name, value, prov=Provenance.DERIVED, reason=None):
    if prov is Provenance.UNAVAILABLE:
        from kernoscope.refusals import RefusalCode

        return Metric.unavailable(name, reason or "不给数", code=RefusalCode.NO_NVML)
    return Metric(name, value, prov)


class _RawRow:
    """绕过 Metric 的类型校验，模拟"某个后端的 metric 落在计数器命名空间里"。

    真 `Metric.exact("dram__bytes.sum", 1.0)` 在**构造**时就被 R2 拦下（这正是我们想要的），
    所以这里用一个只有字段的替身来验证 `select_metrics` 的第二道防线也生效。
    """

    def __init__(self, name, value, prov):
        self.name, self.value, self.provenance, self.source = name, value, prov, "raw"


def test_selection_refuses_counter_namespace_and_unavailable_values():
    kr = _kr_with([
        _m("ks__launch_time_mean", 0.09),
        _RawRow("dram__bytes.sum", 1.0, Provenance.EXACT),      # 计数器命名空间：绝不写
        _m("meas__window_energy_j", 4200.0, Provenance.MEASURED),
        _m("est__dram_bytes_per_launch", 2.5e7, Provenance.ESTIMATED),
        _RawRow("sm__throughput.avg.pct_of_peak_sustained_elapsed", None, Provenance.UNAVAILABLE),
    ])
    metrics, prov, units, dropped, by_ns, over = select_metrics(kr)
    names = [m.name for m in metrics]
    assert "dram__bytes.sum" not in names
    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" not in names
    assert names[0] == "meas__window_energy_j"        # measured 排在 estimated 之前
    assert prov["ks__launch_time_mean"] == "derived"
    assert over == 0 and dropped == [] and by_ns == []
    with pytest.raises(SubstituteRefused):
        assert_writable([type("M", (), {"name": "dram__bytes.sum"})()])


def test_selection_drops_metrics_whose_premise_failed_and_refuses_if_nothing_is_left():
    """被拒判词引用过的指标不写；全被丢掉时**拒绝生成文件**（空报告比没有报告更糟）。"""
    kr = _kr_with([_m("est__dram_bytes_per_launch", 2.5e7, Provenance.ESTIMATED)])
    packet = {"verdicts": [{"id": "dram_bytes", "status": "refused",
                            "metrics": ["est__dram_bytes_per_launch"]}]}
    metrics, _, _, dropped, _, _ = select_metrics(kr, packet)
    assert metrics == [] and dropped == ["est__dram_bytes_per_launch"]
    with pytest.raises(SubstituteRefused) as e:
        build_plan(kr, packet=packet, tool_version="0.2.0")
    assert "一个可署名的值都不剩" in str(e.value)
    # 同一条指标：前提成立（没有 refused 判词）时必须能写出去
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="0.2.0")
    assert [m.name for m in plan.metrics] == ["est__dram_bytes_per_launch"]


def test_kernel_without_launches_is_refused_instead_of_inventing_an_identity():
    kr = _kr_with([_m("ks__launch_time_mean", 0.09)], launches=[])
    with pytest.raises(SubstituteRefused) as e:
        build_plan(kr, packet={"verdicts": []})
    assert "身份" in str(e.value)


def test_same_source_ncu_rows_are_written_with_ncu_units_and_labels():
    """同源行要写进 ncu 名下，**并且按 ncu 单位换算 + 用 ncu 的显示标签**。

    两条都由真机实验逼出来：
      * 单位：我们是 ms、ncu 是 ns —— 少乘 1e6 就是 R-51（MB 被当字节）那类错误的翻版；
      * 标签：`ncu --import --page details` 的表按 **Label** 查 ncu 自带的节定义，
        Label 认不出来整行就消失（把真报告的 "Grid Size" 换成我们的指标名 → 行没了）。
    """
    from kernoscope.ncrep.ncu_rows import NCU_ROWS, consensus_check

    assert consensus_check() == [], "行表与节表必须自洽"
    labels = {r.name: r.label for r in NCU_ROWS}
    assert labels["gpu__time_duration.sum"] == "Duration"
    assert labels["launch__registers_per_thread"] == "Registers Per Thread"

    kr = _kr_with([_m("gpu__time_duration.sum", 0.089728, Provenance.EXACT),
                   _m("launch__registers_per_thread", 16.0, Provenance.EXACT),
                   _m("ks__launch_time_mean", 0.089728)])
    metrics, _, units, dropped, by_ns, _ = select_metrics(kr)
    by_name = {m.name: m for m in metrics}
    assert "gpu__time_duration.sum" in by_name and "gpu__time_duration.sum" not in by_ns
    assert by_name["gpu__time_duration.sum"].value == pytest.approx(0.089728 * 1e6)  # ms → ns
    assert units["gpu__time_duration.sum"] == "ns"
    assert units["launch__registers_per_thread"] == "register/thread"
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="0.2.0")
    assert "written in ncu units" in plan.comment


def test_comment_is_self_describing_for_someone_who_never_heard_of_kernoscope():
    plan = build_plan(_kr_with([
        _m("launch__registers_per_thread", 16.0, Provenance.EXACT),
        _m("ks__cost_per_mtok", 12.5),
        _m("est__dram_bytes_per_launch", 2.5e7, Provenance.ESTIMATED),
        _m("meas__window_energy_j", 4200.0, Provenance.MEASURED),
    ]), packet={"verdicts": []}, tool_version="0.2.0")
    c = plan.comment
    assert ATTRIBUTION_MARK in c
    assert "NOT HARDWARE PERFORMANCE COUNTERS" in c
    assert "kernoscope 0.2.0" in c
    assert "exact=1" in c and "derived=1" in c and "estimated=1" in c and "measured=1" in c
    assert "ks__=derived" in c


def test_write_substitute_emits_a_v7_container_plus_a_human_sidecar(tmp_path):
    kr = _kr_with([_m("ks__cost_per_mtok", 12.5), _m("meas__window_energy_j", 4200.0,
                                                     Provenance.MEASURED)])
    out = tmp_path / "sub" / "rep.ncu-rep"
    res = write_substitute(kr, str(out), packet={"verdicts": []}, tool_version="0.2.0")
    raw = out.read_bytes()
    assert raw[:4] == b"NVR\x00"                     # 魔数
    assert raw[4:8] == (2).to_bytes(4, "little")     # FileHeader 长度
    assert raw[8] == 0x08 and raw[9] == 7            # FileHeader.Version = 7
    assert res.bytes_written == len(raw) and res.metrics_written == 2

    side = Path(res.sidecar_path)
    assert side.exists()
    text = side.read_text(encoding="utf-8")
    assert ATTRIBUTION_MARK in text and "不是硬件计数器读数" in text
    assert "`ks__cost_per_mtok`" in text and "`meas__window_energy_j`" in text
    assert "| 12.5 |" in text


def test_dry_run_writes_nothing(tmp_path):
    kr = _kr_with([_m("ks__cost_per_mtok", 12.5)])
    out = tmp_path / "rep.ncu-rep"
    res = write_substitute(kr, str(out), packet={"verdicts": []}, dry_run=True)
    assert res.metrics_written == 1 and res.bytes_written == 0
    assert not out.exists() and not Path(res.sidecar_path).exists()


def test_dropped_and_over_cap_are_reported_not_silently_lost():
    """静默丢值就是另一种撒谎：超上限与被前提挡住都要写进署名。"""
    metrics = [_m(f"ks__m{i:02d}", float(i)) for i in range(40)]
    kr = _kr_with(metrics)
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="0.2.0", limit=10)
    assert len(plan.metrics) == 10 and plan.dropped_over_cap == 30
    assert "omitted by size cap: 30" in plan.comment

    kr2 = _kr_with([_m("est__x", 1.0, Provenance.ESTIMATED), _m("ks__y", 2.0)])
    packet = {"verdicts": [{"id": "roofline", "status": "refused", "metrics": ["est__x"]}]}
    plan2 = build_plan(kr2, packet=packet, tool_version="0.2.0")
    assert plan2.dropped_by_refusal == ["est__x"]
    assert "premises failed" in plan2.comment
    assert "est__x" not in render_comment(plan2).split("premises failed", 1)[0].split("values:")[0]
