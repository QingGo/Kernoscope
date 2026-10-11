"""Which of our values may be written under **ncu's own metric names**（M6）。

背景（真机实验得出，见 `docs` 之外的提交信息与 README §4.4）：

* `ncu --import --page details` 的表格**不是**按我们写的 `Sections[].Header.Metrics[].Name`
  渲染的 —— 它按 **`Label`** 去查 Nsight Compute 自带的节定义（`sections/*.section` 编译进去的
  那张表）：Label 认不出来，那一行就整个消失；Label 认出来但 Name 在报告里找不到，就渲染成
  `(!) n/a`。这是真机二分实验的结论（把真报告里一个 Header.Metrics 的 Label 从
  "Grid Size" 改成我们的指标名 → 该表立刻少一行；改回真标签 → 行回来）。
* 但**只用这些还不够**：details 页的**表格行**只在"报告的指标集完整到能被 ncu 自己的节定义
  解析"时才渲染（真机二分：把真报告的 839 条 MetricResults 全搬进来才出行；只给子集 ——
  哪怕名字/标签/单位全对 —— 一行都不出）。替代口径报告**结构上不可能**满足这个条件，
  所以那道题的正解不是继续伺候它的渲染逻辑，而是：
  ① 用 ncu 的节标识（`LaunchStats` / `SpeedOfLight`）+ **ncu 的显示标签** + **ncu 的单位**，
     让我们的同源行在**任何**按 ncu 语义读文件的工具里都自洽；
  ② 每个节挂一条 `RuleResult`（details 页**会**渲染它的 Message 散文）→ 同源值以文本出现；
  ③ 完整数值走 `--page raw` / Python API / 旁路署名 / 我们自己的查看器。
  ⇒ 现实口径：details 页给"节标题 + 同源值文本 + 署名"，**不给表格行**（README §11⑧）。

这条纪律与"绝不占用计数器命名空间"并不冲突，因为这里只允许**同源**的量：
  我们确实从 CUPTI activity / cubin 拿到的、与 ncu 语义完全相同的那些数
  （kernel 时长、launch 几何、寄存器/静态共享内存）。它们本来就是 ncu 会给出的数。
**代理/模型/成本类的量永远不会走这张表** —— 它们留在我们自己的命名空间
（`ks__`/`est__`/`meas__`）里，在 `--page raw`、Python API、旁路署名与 RuleResult 里可见。

标签与单位来自 NVIDIA 随 Nsight Compute 发布的 `sections/*.section`（那是
`ProfilerSection` 的文本格式定义），我们只取**指标名 → 显示标签**这一对短事实，
其余（描述文本、Sets、Filter）一律自己生成，不复制。

**名字与来源必须相容（N3）**：白名单只是"我们有这一行"的登记，不是"可以用 ncu 名"的许可。
许可由 `source_class` 判定：只有 `documented`（NVIDIA 公开文档/头文件定义了该量与单位）
且单位相符的行才允许写进 ncu 名字空间；`documented-adjacent`/`undocumented-observed`/
`vendor-tool-output` 的行即使登记在这里，也只许进 `ks__`/`est__`/`meas__`
（见 `source_class.allows_ncu_name`）。`consensus_check()` 会拒绝"登记了却不可用"的行 ——
登记一行却写不出去，是表与规则脱节的信号，必须炸而不是静默。
"""

from __future__ import annotations

from dataclasses import dataclass

from .source_class import SourceClass, allows_ncu_name


@dataclass(frozen=True)
class NcuRow:
    """一行"同源可写"的 ncu 指标。

    Args:
        name: ncu 的指标名，也是我们报告里那个指标的名字（同源 ⇒ 同名）。
        label: ncu 自带的显示标签 —— **必须逐字一致**，否则 details 页不渲染这一行。
        unit: 写进文件的单位（ncu 的单位）。
        scale: 把**我们的值**换算成 ncu 单位要乘的系数。
        section: 它属于哪个 ncu 节标识。
        from_units: 我们报告里这个指标**允许**的单位（别名集合）。
            这是换算的安全带（第五轮 V-9）：`scale` 只在"我们的单位确实是这个"时才成立，
            所以换算前必须核对单位。空字符串表示"无量纲/计数"。
        source_class: 这个量的来源分类（N3）。默认 `documented` —— 现有行都是
            CUPTI activity / cubin 这类"公开文档定义了量与单位"的同源量，
            所以默认值让今天的行为逐位不变。挂上非 `documented` 的档就等于
            **声明这一行不许再用 ncu 名**（登记与许可就此分开）。
    """

    name: str
    label: str
    unit: str
    scale: float
    section: str
    from_units: tuple[str, ...] = ("",)
    source_class: SourceClass = SourceClass.DOCUMENTED


@dataclass(frozen=True)
class NcuSection:
    identifier: str
    display_name: str
    order: int
    #: 我们在这一节里愿意/能够提供的行（顺序即渲染顺序）。
    rows: tuple[str, ...]


#: 允许写进 ncu 命名空间的行。**白名单，不是黑名单**：
#: 单位语义必须两边完全一致（或有一个明确的换算），否则不进来。
#: 白名单是"我们有这一行"的登记；能否用 ncu 名还要看 `source_class`（N3 规则）。
#:
#: `gpu__time_duration.sum` 是这份白名单里唯一需要换算的行：我们报告里的单位是 **ms**
#: （CUPTI 时间戳换成毫秒便于人读），而 ncu 的是 **ns** —— 所以写文件时 ×1e6。
#: 早先的做法是"干脆不写它"，那是把**单位问题**当成了**身份问题**：同源的量只要换算正确，
#: 写进 ncu 名下恰恰是它该在的地方（用户要的就是在 ncu-ui 里看到 Duration）。
NCU_ROWS: tuple[NcuRow, ...] = (
    NcuRow("gpu__time_duration.sum", "Duration", "ns", 1e6, "SpeedOfLight", ("ms",)),
    NcuRow("launch__grid_size", "Grid Size", "", 1.0, "LaunchStats"),
    NcuRow("launch__block_size", "Block Size", "", 1.0, "LaunchStats"),
    NcuRow("launch__registers_per_thread", "Registers Per Thread", "register/thread", 1.0,
           "LaunchStats"),
    NcuRow("launch__shared_mem_per_block_static", "Static Shared Memory Per Block",
           "byte/block", 1.0, "LaunchStats", ("B", "byte", "bytes")),
    NcuRow("launch__thread_count", "Threads", "thread", 1.0, "LaunchStats"),
    NcuRow("launch__shared_mem_per_block_dynamic", "Dynamic Shared Memory Per Block",
           "byte/block", 1.0, "LaunchStats", ("B", "byte", "bytes")),
)

#: 我们会生成的 ncu 节（只列我们能填的行；ncu 的节定义里有更多行，我们没有就不写）。
#: `Order` 与 `DisplayName` 取自随包发布的 section 定义（顺序影响 details 页里的先后）。
NCU_SECTIONS: tuple[NcuSection, ...] = (
    NcuSection("SpeedOfLight", "GPU Speed Of Light Throughput", 20,
               ("gpu__time_duration.sum",)),
    NcuSection("LaunchStats", "Launch Statistics", 70,
               ("launch__grid_size", "launch__block_size", "launch__registers_per_thread",
                "launch__shared_mem_per_block_static",
                "launch__shared_mem_per_block_dynamic", "launch__thread_count")),
)

_BY_NAME: dict[str, NcuRow] = {row.name: row for row in NCU_ROWS}


def row_for(name: str) -> NcuRow | None:
    """这个指标名是否是"同源可写"的一行；不是就返回 None（**不猜**）。"""
    return _BY_NAME.get(name)


def section_of(name: str) -> str:
    row = _BY_NAME.get(name)
    return row.section if row is not None else ""


def source_class_of(name: str) -> SourceClass | None:
    """这一行的来源分类；名字不在表里返回 None（**不猜**）。"""
    row = _BY_NAME.get(name)
    return row.source_class if row is not None else None


class UnitMismatch(ValueError):
    """我们的单位与这张表假设的单位不一致 —— 换算系数不成立，拒绝换算。

    为什么要有它（第五轮 V-9）：`gpu__time_duration.sum` 无条件 ×1e6 只在"我们的单位确实是
    ms"时才对；上游哪天改成 ns，就会静默写出 1000× 的数。宁可炸，也不要写一个乘错的数字。
    """


def unit_ok(name: str, unit: str | None) -> bool:
    """这个单位能不能按本表的系数换算（None = 没带单位信息，按表里的假设走）。"""
    row = _BY_NAME.get(name)
    if row is None:
        return False
    if unit is None or unit == "" and "" in row.from_units:
        return True
    return unit in row.from_units


def ncu_name_refusal(name: str, unit: str | None) -> str:
    """名字/来源/单位这个组合**为什么**不许用 ncu 名；允许时返回空串。

    给"机器可读的丢弃原因"用一句话说清是哪一道门挡下的 ——
    审计者不必复算就能区分**来源档位**不符与**单位**不符。
    """
    row = _BY_NAME.get(name)
    if row is None:
        return f"{name!r} 不在 NCU_ROWS 白名单里，没有同源行定义"
    if row.source_class is not SourceClass.DOCUMENTED:
        return (f"{name!r} 的 source_class={row.source_class.value}"
                f"（{row.source_class.justification}）—— 只有 documented 才允许用 ncu 名")
    if not unit_ok(name, unit):
        return (f"{name!r} 的单位 {unit!r} 与换算表 {row.from_units} 不符"
                f"（ncu 单位 {row.unit!r}）—— 单位不符就不写，宁可不写也不写乘错的数")
    return ""


def allows_ncu_row_name(name: str, unit: str | None) -> bool:
    """白名单里的这一行**此刻**是否允许写 ncu 名字空间（来源档位 + 单位两道门一起判）。"""
    row = _BY_NAME.get(name)
    if row is None:
        return False
    return allows_ncu_name(row.source_class, unit_ok(name, unit))


def convert(name: str, value: float, unit: str | None = None) -> tuple[float, str]:
    """把我们的值换算成 ncu 单位；返回 ``(换算后的值, ncu 单位)``。

    Args:
        unit: 我们的值**实际**的单位。给了就核对（不一致抛 :class:`UnitMismatch`）；
            给 ``None`` 表示调用方拿不到单位信息，按表里的假设换算（老调用点兼容）。

    名字不在白名单里也抛 —— 调用方必须先 `row_for()` 判过，绝不允许"顺手写一个名字"。
    """
    row = _BY_NAME[name]
    if unit is not None and unit not in row.from_units:
        raise UnitMismatch(
            f"{name}: 我们的单位是 {unit!r}，本表的换算系数按 {row.from_units} 成立"
            f"（ncu 单位 {row.unit!r}）—— 单位不符时不换算，宁可不写")
    return float(value) * row.scale, row.unit


def consensus_check() -> list[str]:
    """自查：节表与行表必须自洽（同一个名字只属于一个节、节里的名字都有行）。

    放在代码里而不是文档里：这张表是"我们敢用 ncu 名字"的全部依据，
    一旦有人手改出不一致（例如把某行挂到两个节），必须立刻炸而不是静默。

    还多一条（N3）：**登记在这里的行必须真的允许用 ncu 名** ——
    来源档位不是 `documented`，或者单位集合筛不出任何可换算的单位，
    这一行就永远写不出去。那种"登记了却不可用"的状态是表与规则脱节的信号，
    必须立刻炸，而不是等到某天有人对着它写 ncu 名。
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    # 遍历**查找表当前的行对象**（而不是模块级元组里那一份）：自查的对象必须是
    # `row_for()` 真正会返回的东西，否则表被替换/降级时自查会对着旧对象说"一切正常"。
    for row in _BY_NAME.values():
        if row.name in seen:
            problems.append(f"{row.name} 出现在多个节：{seen[row.name]} / {row.section}")
        seen[row.name] = row.section
        if not row.label:
            problems.append(f"{row.name} 没有 ncu 显示标签（details 页会整行消失）")
        if row.scale <= 0:
            problems.append(f"{row.name} 的换算系数非法：{row.scale}")
        if not row.from_units:
            problems.append(f"{row.name} 没声明 from_units —— 换算没有安全带（V-9）")
        if not isinstance(row.source_class, SourceClass):
            problems.append(f"{row.name} 的 source_class 非法：{row.source_class!r}")
        elif not row.source_class.allows_ncu_namespace:
            problems.append(
                f"{row.name} 登记在 NCU_ROWS 里，source_class 却是 {row.source_class.value}"
                " —— 这一行永远不许用 ncu 名，不该登记为同源行（N3）")
        elif not any(unit_ok(row.name, u) for u in (*row.from_units, None)):
            problems.append(
                f"{row.name} 的 from_units={row.from_units} 里没有任何可换算的单位"
                " —— 单位这道门永远关着，等于写不出去")
    for sec in NCU_SECTIONS:
        for name in sec.rows:
            if name not in _BY_NAME:
                problems.append(f"节 {sec.identifier} 引用了没有行定义的名字 {name}")
            elif _BY_NAME[name].section != sec.identifier:
                problems.append(f"{name} 的行定义说它属于 {_BY_NAME[name].section}，"
                                f"却挂在 {sec.identifier}")
    return problems
