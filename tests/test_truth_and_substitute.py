"""真值对拍（I1）与 `.ncu-rep` 导出（I2/M3）的契约测试。

这一族功能的失败模式很特别：**它不会崩，它会很安静地骗人**。所以测试的重点是
"什么时候必须拒绝给数"和"写进 `.ncu-rep` 的东西能不能被署名、会不会被误当计数器读数"。

**真报告口径未验证**（R-107/R-90）：对拍那一半用的是与真实 dataclass 同构的**替身**
（`tests/test_ncu_rep.py` 则用假 `ncu_report` 模块）—— 仓库里没有真 ncu 报告，
所以这里验的是**包装逻辑**，不是真报告上的对拍结果。README/CHANGELOG 的措辞由
`tests/test_claim_maturity.py` 钉住，别把它改回"已验证"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernoscope.analyze.truth import (PAIRS, compare, comparison_metrics, match_action,
                                      truth_metrics)
from kernoscope.backends.ncu_rep import NcuRepAction, NcuRepData
from kernoscope.ncrep.substitute import (ATTRIBUTION_JSON_MARK, ATTRIBUTION_JSON_SCHEMA,
                                         ATTRIBUTION_MARK, SubstituteRefused,
                                         assert_writable, attribution_data, build_plan,
                                         render_comment, render_sidecar, select_metrics,
                                         write_substitute)
from kernoscope.ncrep.source_class import SourceClass, allows_ncu_name
from kernoscope.ncrep.ncu_rows import row_for
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


def _m(name, value, prov=Provenance.DERIVED, reason=None, unit=""):
    """`unit` 必须给**真实**单位：R-110 之后空串不再被当成"按假设换算"，
    所以夹具里给 `gpu__time_duration.sum` 这类同源行必须写 `"ms"`（与 `sections.py` 一致）。"""
    if prov is Provenance.UNAVAILABLE:
        from kernoscope.refusals import RefusalCode

        return Metric.unavailable(name, reason or "不给数", code=RefusalCode.NO_NVML)
    return Metric(name, value, prov, unit=unit)


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
    metrics, prov, units, dropped, by_ns, by_sc, our_units, over = select_metrics(kr)
    names = [m.name for m in metrics]
    assert "dram__bytes.sum" not in names
    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" not in names
    assert names[0] == "meas__window_energy_j"        # measured 排在 estimated 之前
    assert prov["ks__launch_time_mean"] == "derived"
    assert over == 0 and dropped == [] and by_ns == [] and by_sc == []
    with pytest.raises(SubstituteRefused):
        assert_writable([type("M", (), {"name": "dram__bytes.sum"})()])


def test_selection_drops_metrics_whose_premise_failed_and_refuses_if_nothing_is_left():
    """被拒判词引用过的指标不写；全被丢掉时**拒绝生成文件**（空报告比没有报告更糟）。"""
    kr = _kr_with([_m("est__dram_bytes_per_launch", 2.5e7, Provenance.ESTIMATED)])
    packet = {"verdicts": [{"id": "dram_bytes", "status": "refused",
                            "metrics": ["est__dram_bytes_per_launch"]}]}
    metrics, _, _, dropped, _, _, _, _ = select_metrics(kr, packet)
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

    kr = _kr_with([_m("gpu__time_duration.sum", 0.089728, Provenance.EXACT, unit="ms"),
                   _m("launch__registers_per_thread", 16.0, Provenance.EXACT),
                   _m("ks__launch_time_mean", 0.089728, unit="ms")])
    metrics, _, units, dropped, by_ns, _, _, _ = select_metrics(kr)
    by_name = {m.name: m for m in metrics}
    assert "gpu__time_duration.sum" in by_name and "gpu__time_duration.sum" not in by_ns
    # 选择阶段**不改值**（换算只允许在 build_report 一处发生；双换算曾把 ms→ns 变成 ×1e12）
    assert by_name["gpu__time_duration.sum"].value == pytest.approx(0.089728)
    assert units["gpu__time_duration.sum"] == "ns"
    assert units["launch__registers_per_thread"] == "register/thread"
    # 写文件时换算一次，且只有一次
    from kernoscope.ncrep.profile import build_report
    from kernoscope.ncrep.ncu_rows import convert
    assert convert("gpu__time_duration.sum", 0.089728) == pytest.approx((89728.0, "ns"))
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


# --------------------------------------------------------------------------- #
# 第五轮验收 V-11 / V-12：署名散文不许宣称不存在的能力、不许举结构上做不到的例子
# --------------------------------------------------------------------------- #
def test_sidecar_does_not_claim_details_renders_table_rows():
    """V-11：真 ncu 判定 details 页只出节标题 + 规则文本，**没有表格行**。

    所以署名文件与 docstring 都不许写"会出现在 … 表里"。这条测试把口径钉住：
    必须明确写出"表格行不会渲染"，并指出完整数值在哪。
    """
    from kernoscope.ncrep.ncu_rows import convert
    from kernoscope.ncrep.substitute import render_sidecar

    text = render_sidecar(_build_plan_for_sidecar(), ncu_rep_path="out.ncu-rep")
    assert "表格行不会渲染" in text
    assert "--page raw" in text
    # 单位换算必须同时写清"我们的值 × 系数"，读者才能复核（而不是只给 ncu 单位）
    assert "ms × 1e6" in text or "× 1e6" in text
    # 白名单里的行结构上不可能出现在 by_namespace 里 —— 散文不许拿它当"被丢弃"的例子
    from kernoscope.ncrep.profile import build_report
    from kernoscope.ncrep.ncu_rows import row_for
    assert row_for("gpu__time_duration.sum") is not None
    assert row_for("launch__local_mem_per_thread") is None
    assert convert("gpu__time_duration.sum", 1.0, "ms")[0] == 1e6


def _build_plan_for_sidecar():
    """构造一个 plan：含同源行 + 一个真的会被命名空间规则丢弃的名字。"""
    from kernoscope.ncrep.substitute import build_plan

    class _M:
        def __init__(self, name, value, unit=""):
            from kernoscope.provenance import Provenance
            self.name, self.value, self.unit = name, value, unit
            self.provenance = Provenance.EXACT
            self.source = "nvml"

    class _Sec:
        def __init__(self, metrics):
            self.name = "Duration"
            self.metrics = metrics

    class _L:
        grid = (1024, 1, 1)
        block = (256, 1, 1)

    class _K:
        name = "k(float *)"
        launches = [_L()]
        sections = [_Sec([_M("gpu__time_duration.sum", 4.755, "ms"),
                          _M("ks__launch_time_mean", 4.755, "ms"),
                          _M("launch__local_mem_per_thread", 0.0, "B"),
                          _M("launch__waves_per_multiprocessor", 1.0, "B"),
                          _M("gpu__time_duration.avg", 2.0, "ms")])]

    return build_plan(_K(), packet={"verdicts": []}, tool_version="test")


#: 这三条都必须被**命名空间规则**丢弃：它们不在 `NCU_ROWS` 白名单里，也不带自有前缀。
#: （`gpu__time_duration.sum` 是白名单里的同源行，能写；`ks__launch_time_mean` 带自有前缀。）
NAMESPACE_DROPPED = frozenset({"launch__local_mem_per_thread",
                               "launch__waves_per_multiprocessor",
                               "gpu__time_duration.avg"})


def test_sidecar_names_a_value_that_can_actually_be_dropped():
    """V-12 + R-105/W-11：`by_namespace` 的**丢弃集合必须逐个相等**。

    旧正向测试只断言"例子里那个名字在集合里"（`in`）。把 `select_metrics` 的过滤**窄化**
    成只保护例子名（`if m.name != "launch__local_mem_per_thread"`）时，
    `launch__waves_per_multiprocessor` / `gpu__time_duration.avg` 会**漏进** `.ncu-rep`
    —— 而 `in` 断言照样绿（18 passed，报告实测）。所以这里改成集合相等 + 反向不泄漏。
    """
    plan = _build_plan_for_sidecar()
    assert set(plan.dropped_by_namespace) == set(NAMESPACE_DROPPED), (
        "被命名空间规则丢弃的名字集合必须逐个相等（只含例子名说明过滤被窄化了）："
        f"{sorted(plan.dropped_by_namespace)} vs {sorted(NAMESPACE_DROPPED)}")
    # 同源行（白名单里的 exact）**不**该出现在丢弃集合里 —— 结构上不可能
    assert "gpu__time_duration.sum" not in plan.dropped_by_namespace
    written = {m.name for m in plan.metrics}
    assert not (written & NAMESPACE_DROPPED), (
        f"命名空间名泄漏进了 .ncu-rep：{sorted(written & NAMESPACE_DROPPED)}")
    assert written == {"gpu__time_duration.sum", "ks__launch_time_mean"}, sorted(written)
    from kernoscope.ncrep.substitute import render_sidecar
    text = render_sidecar(plan, ncu_rep_path="out.ncu-rep")
    assert "launch__local_mem_per_thread" in text


# --------------------------------------------------------------------------- #
# R-93③/R-108：署名表整数按整数印；署名文件名按自家约定 `<path>.ATTRIBUTION.txt`
# --------------------------------------------------------------------------- #
def test_sidecar_prints_integers_as_integers_not_scientific_notation():
    """`1048576` 被 `{:.6g}` 印成 `1.04858e+06`（R-93③ / R-108 的原话）。

    旧行为回来的话这条会红：整数（含同源行换算后的 ns）必须是整数写法，且与 details 页
    共用**同一个**格式函数 —— 同一份产物里两种写法本身就是不自洽。
    """
    from kernoscope.ncrep import profile, substitute
    from kernoscope.ncrep.substitute import render_sidecar

    assert substitute.format_value is profile.format_value, (
        "署名表与 details 页必须共用同一套数字格式（别各自抄一份 `:.6g`）")

    kr = _kr_with([
        _m("launch__thread_count", 1048576.0, Provenance.EXACT),    # 旧行为：1.04858e+06
        _m("gpu__time_duration.sum", 22.5248, Provenance.EXACT, unit="ms"),  # ×1e6 → 22524800 ns
    ])
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="test")
    text = render_sidecar(plan, ncu_rep_path="out.ncu-rep")

    assert "| `launch__thread_count` | 1,048,576 |" in text, text
    assert "| `gpu__time_duration.sum` | 22,524,800" in text, text
    assert "1.04858e+06" not in text and "2.25248e+07" not in text, (
        "整数不许用科学计数法（同模块的 profile.format_value 早就这么写了）：\n" + text)
    # 换算系数也是整数；`1e+06` 同样是指数记法
    assert "× 1,000,000）" in text, text


def test_sidecar_name_follows_the_path_attribution_convention(tmp_path):
    """约定是 `<path>.ATTRIBUTION.txt`（`path` = 那份 `.ncu-rep` 的路径）。

    R-108 记的错法是 `file.with_suffix(".ATTRIBUTION.txt")` ⇒ `rep.ATTRIBUTION.txt`：
    读的人按约定去找 `rep.ncu-rep.ATTRIBUTION.txt` 会找不到（归档的 m6 证据现在就是这个错法）。
    这条钉住"路径**原样拼接**"——改回 with_suffix、或让写法随路径后缀变化，都会红。
    """
    kr = _kr_with([_m("ks__cost_per_mtok", 12.5)])
    out = tmp_path / "rep.ncu-rep"
    res = write_substitute(kr, str(out), packet={"verdicts": []}, tool_version="test")
    assert res.sidecar_path == str(out) + ".ATTRIBUTION.txt", res.sidecar_path
    assert Path(res.sidecar_path).name == "rep.ncu-rep.ATTRIBUTION.txt"
    assert Path(res.sidecar_path).exists()
    assert not (tmp_path / "rep.ATTRIBUTION.txt").exists(), (
        "不许把 `.ncu-rep` 后缀替换掉：署名文件名必须是 `<path>.ATTRIBUTION.txt`")
    # 文件头的 `file:` 行要指向那份报告：归档时一改名就看不出它属于哪份报告了
    assert f"file: {out}" in Path(res.sidecar_path).read_text(encoding="utf-8")


def test_empty_unit_is_not_converted_by_assumption():
    """R-110（第六轮 W-13）：工厂默认的 `unit=""` 不能被当成"按假设换算"。

    修前 `select_metrics` 用 `m.unit or None` 把空串变成"调用方拿不到单位"，于是
    `gpu__time_duration.sum` 会被无条件 ×1e6 写出去（端到端曾写出 9.51e11 ns）。
    单位不明的量必须**不写**，并出现在"因命名空间/单位规则未写入"的名单里。
    """
    from kernoscope.ncrep.substitute import select_metrics

    kr = _kr_with([
        _m("gpu__time_duration.sum", 9.51, Provenance.EXACT),        # 单位缺失 ⇒ 不许换算
        _m("launch__thread_count", 1048576.0, Provenance.EXACT),     # from_units 含 "" ⇒ 不受影响
    ])
    metrics, _, _, _, by_ns, _, _, _ = select_metrics(kr)
    names = {m.name for m in metrics}
    assert "gpu__time_duration.sum" not in names, \
        "单位不明的同源行不许按假设换算后写出去"
    assert "gpu__time_duration.sum" in by_ns, by_ns
    assert "launch__thread_count" in names, \
        "`from_units` 里本来就含空串的行不受影响（否则这条修复会误伤）"


# --------------------------------------------------------------------------- #
# N3：把"同源才写 ncu 名"从散文政策变成类型系统（`source_class` 四档）
# --------------------------------------------------------------------------- #
def _demote(monkeypatch, name: str, cls: SourceClass):
    """把白名单里某一行的来源档位降级 —— 用来驱动"规则被回退就变红"。

    走 `ncu_rows._BY_NAME`（`row_for()` 的唯一后端）而不是复制一份表：
    替换的是**真的查找路径**，所以测的是生产代码，不是测试自己搭的假表。
    """
    from dataclasses import replace

    from kernoscope.ncrep import ncu_rows

    row = ncu_rows.row_for(name)
    assert row is not None, f"{name} 必须在 NCU_ROWS 白名单里"
    monkeypatch.setitem(ncu_rows._BY_NAME, name, replace(row, source_class=cls))


def _parse_json_block(text: str) -> dict:
    """从旁路署名里取出机器可读块（委托生产者的读法，避免下游各自 split 一次）。"""
    from kernoscope.ncrep.substitute import parse_attribution_json

    return parse_attribution_json(text)


def test_allows_ncu_name_is_exactly_documented_and_unit_ok():
    """规则的机检形式：**只有 documented 且单位相符**才允许 ncu 名。

    这条覆盖四档 × 单位两态：把 `allows_ncu_name` 改成"只要 unit_ok 就行"
    （即撤掉档位这道门）时，`documented-adjacent`/`undocumented-observed`/
    `vendor-tool-output` 那 6 个组合会立刻变红。
    """
    assert allows_ncu_name(SourceClass.DOCUMENTED, True) is True
    assert allows_ncu_name("documented", True) is True       # str 子类：JSON/CLI 值也能用
    for cls in (SourceClass.DOCUMENTED_ADJACENT, SourceClass.UNDOCUMENTED_OBSERVED,
                SourceClass.VENDOR_TOOL_OUTPUT):
        assert allows_ncu_name(cls, True) is False, f"{cls.value} 不许挂 ncu 名"
        assert allows_ncu_name(cls, False) is False
    # 单位是独立的一道门：来源再权威，单位不符也不许写（V-9 不许被这条规则绕过）
    assert allows_ncu_name(SourceClass.DOCUMENTED, False) is False


def test_default_source_class_keeps_todays_rows_documented():
    """`source_class` 有默认值 ⇒ 今天的行行为逐位不变（不能借这次改动放松边界）。"""
    from kernoscope.ncrep.ncu_rows import NCU_ROWS

    assert NCU_ROWS and all(r.source_class is SourceClass.DOCUMENTED for r in NCU_ROWS)
    assert all(allows_ncu_name(r.source_class, True) for r in NCU_ROWS), \
        "白名单里的每一行都必须允许用 ncu 名（否则该行永远写不出去）"


def test_undocumented_row_may_not_be_written_under_its_ncu_name(monkeypatch):
    """**规则回退就红**：把一行降级成 `undocumented-observed`，它就不许用那个 ncu 名。

    回退检查：删掉 `select_metrics` 里的档位判断（或把 `allows_ncu_name` 改成只看单位），
    这条立刻红 —— 因为 `gpu__time_duration.sum` 会重新被写出去。
    """
    _demote(monkeypatch, "gpu__time_duration.sum", SourceClass.UNDOCUMENTED_OBSERVED)
    kr = _kr_with([_m("gpu__time_duration.sum", 0.089728, Provenance.EXACT, unit="ms"),
                   _m("ks__launch_time_mean", 0.089728, Provenance.DERIVED, unit="ms")])
    metrics, _, _, _, by_ns, by_sc, _, _ = select_metrics(kr)
    names = {m.name for m in metrics}
    assert "gpu__time_duration.sum" not in names, \
        "非 documented 的行不许写进 ncu 名字空间（档位规则被回退了）"
    assert "gpu__time_duration.sum" not in by_ns, \
        "它不是被命名空间规则挡下的（它确实在白名单里），是被**档位**挡下的"
    assert [d.name for d in by_sc] == ["gpu__time_duration.sum"]
    assert by_sc[0].source_class is SourceClass.UNDOCUMENTED_OBSERVED
    assert "undocumented-observed" in by_sc[0].reason, by_sc[0].reason
    # 同一份报告里的自有命名空间值照写（丢弃是逐条的，不是整份放弃）
    assert "ks__launch_time_mean" in names


def test_demoted_row_is_refused_not_renamed_into_another_ncu_name(monkeypatch):
    """被档位挡下的值**不许**改挂到别的 ncu 名（"换个名字写出去"是最坏的回退）。"""
    _demote(monkeypatch, "launch__grid_size", SourceClass.DOCUMENTED_ADJACENT)
    kr = _kr_with([_m("launch__grid_size", 4096.0, Provenance.EXACT),
                   _m("launch__block_size", 128.0, Provenance.EXACT)])
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="test")
    written = {m.name for m in plan.metrics}
    assert written == {"launch__block_size"}, written
    assert [d.name for d in plan.dropped_by_source_class] == ["launch__grid_size"]
    # 机器块里也只出现允许的那个 ncu 名
    json_block = _parse_json_block(render_sidecar(plan, ncu_rep_path="out.ncu-rep"))
    assert [w["ncu_name"] for w in json_block["written"]] == ["launch__block_size"]
    assert [d["name"] for d in json_block["not_written_by_source_class"]] == ["launch__grid_size"]


def test_demoted_row_alone_refuses_the_whole_file(monkeypatch):
    """只剩一个"档位不相容"的值 ⇒ 拒绝生成文件（空报告比没有报告更糟，I2 的老纪律）。"""
    _demote(monkeypatch, "launch__block_size", SourceClass.VENDOR_TOOL_OUTPUT)
    kr = _kr_with([_m("launch__block_size", 128.0, Provenance.EXACT)])
    with pytest.raises(SubstituteRefused) as e:
        build_plan(kr, packet={"verdicts": []}, tool_version="test")
    assert "一个可署名的值都不剩" in str(e.value)


def test_assert_writable_also_refuses_a_demoted_row(monkeypatch):
    """第二道防线也认档位：**不经过 `select_metrics`** 直接送进来的行照样被挡下。"""
    from kernoscope.ncrep.profile import RepMetric

    _demote(monkeypatch, "launch__thread_count", SourceClass.DOCUMENTED_ADJACENT)
    with pytest.raises(SubstituteRefused) as e:
        assert_writable([RepMetric("launch__thread_count", 1024.0)])
    assert "N3" in str(e.value) and "documented-adjacent" in str(e.value)


def test_assert_writable_still_refuses_a_unit_mismatch_for_a_documented_row():
    """档位对、单位错，仍然拒绝 —— 两道门是"与"，不是"或"（V-9 + N3）。"""
    from kernoscope.ncrep.profile import RepMetric

    with pytest.raises(SubstituteRefused) as e:
        assert_writable([RepMetric("gpu__time_duration.sum", 1.0, "us")])
    assert "单位" in str(e.value)


def test_attribution_json_covers_every_written_value_and_blocks_ncu_names():
    """机检署名：每个写入值都有条目；**非 documented 的值绝不出现在 ncu 名下**。"""
    plan = _build_plan_for_sidecar()
    data = _parse_json_block(render_sidecar(plan, ncu_rep_path="out.ncu-rep"))
    written_names = {m.name for m in plan.metrics}
    entries = {w["name"]: w for w in data["written"]}
    assert written_names == set(entries), "每个写入值都必须有署名条目（名字集合相等）"
    assert len(data["written"]) == len(plan.metrics), "条目数必须等于写入值数（不许少也不许多）"
    for name, entry in entries.items():
        assert entry["provenance"] == plan.provenance_by_name[name]
        if entry["ncu_name"] is not None:
            row = row_for(entry["ncu_name"])
            assert row is not None and row.source_class is SourceClass.DOCUMENTED
    # 反向：`ncu_name` 不是 documented 的行，在文件里就不该存在
    assert data["not_written_by_source_class"] == []


def test_every_written_value_carries_its_origin_in_the_machine_block():
    """出处必须进机器块（`origin` = `Metric.source` 原文）：审计工具要能追到通道。"""
    from dataclasses import replace

    metric = _m("ks__cost_per_mtok", 12.5)
    kr = _kr_with([replace(metric, source="NVML 采样 window_energy")])
    plan = build_plan(kr, packet={"verdicts": []}, tool_version="test")
    data = _parse_json_block(render_sidecar(plan, ncu_rep_path="out.ncu-rep"))
    assert data["written"] == [{
        "name": "ks__cost_per_mtok", "value": 12.5, "ncu_name": None, "unit": "",
        "our_unit": "", "source_class": None, "provenance": "derived",
        "origin": "NVML 采样 window_energy",
    }]


def test_parse_attribution_json_refuses_a_malformed_block():
    """读法也要在同一个地方定义：标记缺失/块不完整/schema 不对 ⇒ 抛错，不返回空字典。

    "读不到"与"一个值都没写"必须可区分 —— 静默返回 `{}` 会让审计工具把
    「署名丢了」读成「没有值得署名的东西」。
    """
    from kernoscope.ncrep.substitute import parse_attribution_json

    good = render_sidecar(_build_plan_for_sidecar(), ncu_rep_path="out.ncu-rep")
    assert parse_attribution_json(good)["schema"] == ATTRIBUTION_JSON_SCHEMA
    for broken in ("# 没有标记的署名",                                   # 标记丢了
                   good.split("```json", 1)[0],                          # 块被截断
                   good.replace(ATTRIBUTION_JSON_SCHEMA, "someone-elses/9")):  # schema 不对
        with pytest.raises(ValueError):
            parse_attribution_json(broken)


def test_writing_a_demoted_row_is_refused_and_nothing_is_written(monkeypatch, tmp_path):
    """端到端：写了档位不相容的值 ⇒ 拒绝 + 磁盘上不留 `.ncu-rep` / 署名文件。

    `.ATTRIBUTION.txt` 只在成功写出报告后才存在，所以"被拒的产物没有署名文件"
    正是我们要的行为：一份没有署名的 `.ncu-rep` 才是真正危险的东西。
    """
    _demote(monkeypatch, "launch__grid_size", SourceClass.UNDOCUMENTED_OBSERVED)
    kr = _kr_with([_m("launch__grid_size", 4096.0, Provenance.EXACT)])
    out = tmp_path / "sub" / "rep.ncu-rep"
    with pytest.raises(SubstituteRefused):
        write_substitute(kr, str(out), packet={"verdicts": []}, tool_version="test")
    assert not out.exists()
    assert not Path(str(out) + ".ATTRIBUTION.txt").exists()

    # 同一个值如果留在它自己的名字下，就必须能写出去（拒绝的是"借 ncu 名"，不是这个数）
    plan = build_plan(_kr_with([_m("ks__grid_size", 4096.0, Provenance.EXACT)]),
                      packet={"verdicts": []}, tool_version="test")
    assert [m.name for m in plan.metrics] == ["ks__grid_size"]
