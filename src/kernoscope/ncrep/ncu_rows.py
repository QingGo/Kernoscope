"""Which of our values may be written under **ncu's own metric names**（M6）。

背景（真机实验得出，见 `docs` 之外的提交信息与 README §4.4）：

* `ncu --import --page details` 的表格**不是**按我们写的 `Sections[].Header.Metrics[].Name`
  渲染的 —— 它按 **`Label`** 去查 Nsight Compute 自带的节定义（`sections/*.section` 编译进去的
  那张表）：Label 认不出来，那一行就整个消失；Label 认出来但 Name 在报告里找不到，就渲染成
  `(!) n/a`。这是真机二分实验的结论（把真报告里一个 Header.Metrics 的 Label 从
  "Grid Size" 改成我们的指标名 → 该表立刻少一行；改回真标签 → 行回来）。
* 所以想让 details（以及 ncu-ui 的表格视图）显示我们的数，**必须**：
  ① 用 ncu 的节标识（`LaunchStats` / `SpeedOfLight`）；
  ② 每一行用 **ncu 的显示标签**；
  ③ 值写在 **ncu 的同名指标**下面，且按 ncu 的**单位**换算。

这条纪律与"绝不占用计数器命名空间"并不冲突，因为这里只允许**同源**的量：
  我们确实从 CUPTI activity / cubin 拿到的、与 ncu 语义完全相同的那些数
  （kernel 时长、launch 几何、寄存器/静态共享内存）。它们本来就是 ncu 会给出的数。
**代理/模型/成本类的量永远不会走这张表** —— 它们留在我们自己的命名空间
（`ks__`/`est__`/`meas__`）里，在 `--page raw`、Python API、旁路署名与 RuleResult 里可见。

标签与单位来自 NVIDIA 随 Nsight Compute 发布的 `sections/*.section`（那是
`ProfilerSection` 的文本格式定义），我们只取**指标名 → 显示标签**这一对短事实，
其余（描述文本、Sets、Filter）一律自己生成，不复制。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NcuRow:
    """一行"同源可写"的 ncu 指标。

    Args:
        name: ncu 的指标名，也是我们报告里那个指标的名字（同源 ⇒ 同名）。
        label: ncu 自带的显示标签 —— **必须逐字一致**，否则 details 页不渲染这一行。
        unit: 写进文件的单位（ncu 的单位）。
        scale: 把**我们的值**换算成 ncu 单位要乘的系数。
        section: 它属于哪个 ncu 节标识。
    """

    name: str
    label: str
    unit: str
    scale: float
    section: str


@dataclass(frozen=True)
class NcuSection:
    identifier: str
    display_name: str
    order: int
    #: 我们在这一节里愿意/能够提供的行（顺序即渲染顺序）。
    rows: tuple[str, ...]


#: 允许写进 ncu 命名空间的行。**白名单，不是黑名单**：
#: 单位语义必须两边完全一致（或有一个明确的换算），否则不进来。
#:
#: `gpu__time_duration.sum` 是这份白名单里唯一需要换算的行：我们报告里的单位是 **ms**
#: （CUPTI 时间戳换成毫秒便于人读），而 ncu 的是 **ns** —— 所以写文件时 ×1e6。
#: 早先的做法是"干脆不写它"，那是把**单位问题**当成了**身份问题**：同源的量只要换算正确，
#: 写进 ncu 名下恰恰是它该在的地方（用户要的就是在 ncu-ui 里看到 Duration）。
NCU_ROWS: tuple[NcuRow, ...] = (
    NcuRow("gpu__time_duration.sum", "Duration", "ns", 1e6, "SpeedOfLight"),
    NcuRow("launch__grid_size", "Grid Size", "", 1.0, "LaunchStats"),
    NcuRow("launch__block_size", "Block Size", "", 1.0, "LaunchStats"),
    NcuRow("launch__registers_per_thread", "Registers Per Thread", "register/thread", 1.0,
           "LaunchStats"),
    NcuRow("launch__shared_mem_per_block_static", "Static Shared Memory Per Block",
           "byte/block", 1.0, "LaunchStats"),
)

#: 我们会生成的 ncu 节（只列我们能填的行；ncu 的节定义里有更多行，我们没有就不写）。
#: `Order` 与 `DisplayName` 取自随包发布的 section 定义（顺序影响 details 页里的先后）。
NCU_SECTIONS: tuple[NcuSection, ...] = (
    NcuSection("SpeedOfLight", "GPU Speed Of Light Throughput", 20,
               ("gpu__time_duration.sum",)),
    NcuSection("LaunchStats", "Launch Statistics", 70,
               ("launch__grid_size", "launch__block_size", "launch__registers_per_thread",
                "launch__shared_mem_per_block_static")),
)

_BY_NAME: dict[str, NcuRow] = {row.name: row for row in NCU_ROWS}


def row_for(name: str) -> NcuRow | None:
    """这个指标名是否是"同源可写"的一行；不是就返回 None（**不猜**）。"""
    return _BY_NAME.get(name)


def section_of(name: str) -> str:
    row = _BY_NAME.get(name)
    return row.section if row is not None else ""


def convert(name: str, value: float) -> tuple[float, str]:
    """把我们的值换算成 ncu 单位；返回 ``(换算后的值, ncu 单位)``。

    名字不在白名单里就抛 —— 调用方必须先 `row_for()` 判过，绝不允许"顺手写一个名字"。
    """
    row = _BY_NAME[name]
    return float(value) * row.scale, row.unit


def consensus_check() -> list[str]:
    """自查：节表与行表必须自洽（同一个名字只属于一个节、节里的名字都有行）。

    放在代码里而不是文档里：这张表是"我们敢用 ncu 名字"的全部依据，
    一旦有人手改出不一致（例如把某行挂到两个节），必须立刻炸而不是静默。
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for row in NCU_ROWS:
        if row.name in seen:
            problems.append(f"{row.name} 出现在多个节：{seen[row.name]} / {row.section}")
        seen[row.name] = row.section
        if not row.label:
            problems.append(f"{row.name} 没有 ncu 显示标签（details 页会整行消失）")
        if row.scale <= 0:
            problems.append(f"{row.name} 的换算系数非法：{row.scale}")
    for sec in NCU_SECTIONS:
        for name in sec.rows:
            if name not in _BY_NAME:
                problems.append(f"节 {sec.identifier} 引用了没有行定义的名字 {name}")
            elif _BY_NAME[name].section != sec.identifier:
                problems.append(f"{name} 的行定义说它属于 {_BY_NAME[name].section}，"
                                f"却挂在 {sec.identifier}")
    return problems
