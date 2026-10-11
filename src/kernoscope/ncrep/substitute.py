"""把我们的替代口径写成 `.ncu-rep`（I2/M3）—— **带署名、带前提、带拒绝**。

D3=A 的决策：为了让人能继续用他们已有的 Nsight Compute 工作流（`ncu --import`、ncu-ui、
以及那些只吃 `.ncu-rep` 的下游脚本），我们把替代口径的数值写进一个合法的 `.ncu-rep`。
但这条路的全部风险都在一个地方：**读的人会以为这些数字是硬件计数器读数**。所以：

  1. **署名是强制的**（`ncrep.profile.build_report` 里 provider/comment 为空就抛）；
     `comment` 里必须写清"这不是计数器读数、每个值的出处、以及它的前提"。
  2. **不进 ncu 计数器命名空间**：写进去的名字只能是我们的 `ks__` / `est__` / `meas__` /
     `model__`（以及 `gpu__time_duration` / `launch__` 这类与 ncu 同源的 exact）。
     这一条在 `select_metrics()` 里过滤，并由 `assert_writable()` 再兜一次 —— 
     双重防线的理由是：**写错一次，这个文件就会作为"ncu 报告"在别人的流程里流传**。
  3. **前提不成立的值不写**：决策包里任何一条 `status=refused` 的判词所引用的指标名，
     都不许出现在文件里（它的前提已经被我们自己推翻了）。要是过滤后一个值都不剩，
     就**拒绝生成文件**——生成一个空的/全是垃圾的"ncu 报告"比不生成更糟。
  4. **旁路署名**（sidecar）：官方 Python API 读不到 comment，`ncu --import` 的 details 页
     也不渲染它（2025.3.1 实测）。所以同时在 `<file>.ATTRIBUTION.txt` 里放一份**人能读**的
     完整署名与指标表 —— 文件内署名 + 文件外署名，两条路都留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..provenance import COUNTER_NAMESPACES, OUR_PREFIXES, Provenance
from .profile import RepMetric, RepSource, build_report
from .writer import FILE_VERSION

# 署名标记：下游/审计可以 grep 这个字符串判断"这份报告是不是替代口径产物"。
ATTRIBUTION_MARK = "kernoscope-substitute-v1"
# 写进文件的指标上限：`.ncu-rep` 的大小不是问题，问题是"读的人要能核"。
# 超过上限时按"与决策最相关"的顺序截断（exact/measured 优先，然后 derived，最后 estimated），
# 并把被截断的条数写进 comment —— 静默丢值就是另一种撒谎。
MAX_SUBSTITUTE_METRICS = 32


# 允许写进 `.ncu-rep` 的 ncu 名字（**白名单，不是黑名单**）。
#
# 为什么必须白名单：读 `.ncu-rep` 的人（包括 ncu-ui 与我们自己的导入路径）会**按 ncu 的
# 单位约定**理解这些名字。我们的 `gpu__time_duration.sum` 是 **ms**，而 ncu 同名的真值是
# **ns** —— 把它写进 `.ncu-rep`，读的人会差 10⁶ 倍。本仓库为同一类错误付过一次代价
# （R-51：MB 被当字节，7.27 TB 写成 7.3 MB），所以这次在源头挡住：
# 只有"两边单位语义完全相同"的**计数量**（个/字节）才进白名单。
SUBSTITUTE_SAFE_NCU_NAMES: frozenset[str] = frozenset({
    "launch__registers_per_thread",          # 个/线程
    "launch__shared_mem_per_block_static",    # 字节
    "launch__grid_size",                      # 个
    "launch__block_size",                     # 个
})


class SubstituteRefused(RuntimeError):
    """拒绝生成 `.ncu-rep`：没有可署名的、前提成立的值。"""


@dataclass
class SubstitutePlan:
    """一次导出的完整计划（可先算后看：`--export-ncu-rep` 的 dry-run 也用它）。"""

    source: RepSource
    metrics: list[RepMetric]
    provenance_by_name: dict[str, str] = field(default_factory=dict)
    dropped_by_refusal: list[str] = field(default_factory=list)
    dropped_by_namespace: list[str] = field(default_factory=list)
    dropped_over_cap: int = 0
    premises: list[str] = field(default_factory=list)
    comment: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.metrics)


def _is_counter_name(name: str) -> bool:
    return any(name.startswith(p) for p in COUNTER_NAMESPACES)


def _refused_metric_names(packet: dict[str, Any] | None) -> set[str]:
    """决策包里被拒判词引用的指标名 —— 这些值的前提已被我们自己推翻。"""
    if not packet:
        return set()
    out: set[str] = set()
    for v in packet.get("verdicts", []) or []:
        if str(v.get("status")) == "refused":
            for n in v.get("metrics", []) or []:
                out.add(str(n))
    return out


def assert_writable(metrics: Sequence[RepMetric]) -> None:
    """第二道防线：进文件的名字不许落在 ncu 计数器命名空间里。"""
    bad = [m.name for m in metrics if _is_counter_name(m.name)]
    if bad:
        raise SubstituteRefused(
            "拒绝写出：这些名字落在 ncu 计数器命名空间里（{0}）。我们拿不到计数器，"
            "把替代值挂在这些名字下面就是在伪造 ncu 读数".format(", ".join(sorted(bad)[:6])))


def _provenance_rank(p: Provenance) -> int:
    """截断顺序：与 ncu 同源的 exact 最该保留（可对拍），estimated 最后。"""
    return {Provenance.EXACT: 0, Provenance.MEASURED: 1, Provenance.DERIVED: 2,
            Provenance.ESTIMATED: 3, Provenance.UNAVAILABLE: 9}[p]


def select_metrics(kr: Any, packet: dict[str, Any] | None = None, *,
                   limit: int = MAX_SUBSTITUTE_METRICS
                   ) -> tuple[list[RepMetric], dict[str, str], list[str], list[str], int]:
    """从 kernel 报告里挑出**可以写进 `.ncu-rep`** 的指标。

    过滤规则（每条都有对应测试）：
      * `unavailable` 不写（没有值就是没有值）；
      * 名字落在 ncu 计数器命名空间里的不写（我们给不出计数器）；
      * 名字在 ncu 命名空间里但**不**在 `SUBSTITUTE_SAFE_NCU_NAMES` 白名单里的不写
        （单位语义可能不同：`gpu__time_duration.sum` 我们是 ms、ncu 是 ns）；
      * 被拒判词引用过的不写（前提不成立）；
      * 值不是数字的不写（字符串型判词无法进 `.ncu-rep` 的 double 通道）。
    返回 (metrics, 名字→出处, 因前提不成立被丢弃的名字, 因命名空间规则被丢弃的名字,
    因超上限被丢弃的条数)。
    """
    refused = _refused_metric_names(packet)
    picked: list[tuple[Provenance, str, float, str, str]] = []
    dropped: list[str] = []
    by_namespace: list[str] = []
    for sec in getattr(kr, "sections", []) or []:
        for m in getattr(sec, "metrics", []) or []:
            if m.provenance is Provenance.UNAVAILABLE or m.value is None:
                continue
            if _is_counter_name(m.name):
                continue
            if not m.name.startswith(OUR_PREFIXES) and m.name not in SUBSTITUTE_SAFE_NCU_NAMES:
                by_namespace.append(m.name)
                continue
            if m.name in refused:
                dropped.append(m.name)
                continue
            if isinstance(m.value, bool) or not isinstance(m.value, (int, float)):
                continue
            picked.append((m.provenance, m.name, float(m.value), sec.name, m.source or ""))
    # 同一名字出现多次（多个 section）时保留第一次：`.ncu-rep` 里同名指标只能有一个值，
    # 而"哪个 section 的那个值"必须是我们能说清的（第一个即主口径）。
    seen: set[str] = set()
    uniq = []
    for row in picked:
        if row[1] in seen:
            continue
        seen.add(row[1])
        uniq.append(row)
    uniq.sort(key=lambda r: (_provenance_rank(r[0]), r[1]))
    over = max(0, len(uniq) - limit)
    uniq = uniq[:limit]
    metrics = [RepMetric(name, value, "") for _, name, value, _, _ in uniq]
    prov = {row[1]: row[0].value for row in uniq}
    return metrics, prov, dropped, by_namespace, over


def build_plan(kr: Any, *, packet: dict[str, Any] | None = None,
               tool_version: str = "", limit: int = MAX_SUBSTITUTE_METRICS) -> SubstitutePlan:
    """把"这份 kernel 报告能怎么进 `.ncu-rep`"算出来（纯函数：不写文件）。"""
    launches = list(getattr(kr, "launches", []) or [])
    if not launches:
        raise SubstituteRefused(
            "拒绝写出：这个 kernel 没有 launch 记录，写不出 kernel 身份"
            "（GridSize/BlockSize 是 `.ncu-rep` 的必填字段，编一组数字就是在造身份）")
    launch = launches[0]
    source = RepSource(kernel_name=str(getattr(kr, "name", "") or ""),
                       grid=tuple(launch.grid), block=tuple(launch.block))
    metrics, prov_by_name, dropped, by_namespace, over = select_metrics(kr, packet, limit=limit)
    plan = SubstitutePlan(source=source, metrics=metrics, provenance_by_name=prov_by_name,
                          dropped_by_refusal=dropped, dropped_by_namespace=by_namespace,
                          dropped_over_cap=over)
    if not plan.ok:
        raise SubstituteRefused(
            "拒绝写出：过滤掉「前提不成立」「单位语义不确定（非自有命名空间）」与不可用的指标后，"
            "一个可署名的值都不剩（被丢弃：{0}）".format(
                ", ".join(sorted(dropped + by_namespace)[:6]) or "无"))
    plan.premises = _premises(kr, packet)
    plan.comment = render_comment(plan, tool_version=tool_version)
    return plan


def _premises(kr: Any, packet: dict[str, Any] | None) -> list[str]:
    """把支撑这些数字的前提抄出来（人读；署名的一部分）。"""
    out: list[str] = []
    meta = getattr(kr, "meta", None)
    if isinstance(meta, dict):
        for key in ("window_mode", "calibration_source"):
            if meta.get(key):
                out.append(f"{key}={meta[key]}")
    if packet:
        for v in packet.get("verdicts", []) or []:
            for p in v.get("premises", []) or []:
                if p.get("ok"):
                    unit = p.get("unit") or ""
                    thr = p.get("threshold")
                    out.append(f"{p.get('id')}={p.get('observed')}{unit}"
                               + (f" (门槛 {thr}{unit})" if thr is not None else ""))
    # 去重但保序
    seen: set[str] = set()
    uniq = [x for x in out if not (x in seen or seen.add(x))]
    return uniq


def render_comment(plan: SubstitutePlan, *, tool_version: str = "") -> str:
    """文件内的署名：**这句话是给不知道 kernoscope 的人看的**，所以必须自证。"""
    counts: dict[str, int] = {}
    for p in plan.provenance_by_name.values():
        counts[p] = counts.get(p, 0) + 1
    prov = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    lines = [
        f"{ATTRIBUTION_MARK} -- produced by kernoscope"
        + (f" {tool_version}" if tool_version else ""),
        "NOT HARDWARE PERFORMANCE COUNTERS. These values are substitute calibers "
        "(NVML device sampling, CUPTI kernel activity, address-arithmetic traces, "
        "models) computed on a machine where NVIDIA performance counters are not "
        "accessible. Do not cite them as ncu counter readings.",
        f"values: {len(plan.metrics)} ({prov})",
        "namespaces: ks__=derived, est__=estimated(model/proxy), meas__=our measured probe, "
        "gpu__time_duration/launch__=same source as ncu (exact)",
    ]
    if plan.dropped_by_refusal:
        lines.append("omitted (their own premises failed, so we refuse to publish them): "
                     + ", ".join(plan.dropped_by_refusal[:8]))
    if plan.dropped_by_namespace:
        lines.append("omitted (not our own metric namespace: a .ncu-rep reader would assume "
                     "ncu's unit convention for these names, e.g. ns vs our ms): "
                     + ", ".join(plan.dropped_by_namespace[:8]))
    if plan.dropped_over_cap:
        lines.append(f"omitted by size cap: {plan.dropped_over_cap} more values "
                     f"(see the sidecar for the full list of what exists on this machine)")
    if plan.premises:
        lines.append("premises: " + "; ".join(plan.premises[:6]))
    return "\n".join(lines)


def render_sidecar(plan: SubstitutePlan, *, tool_version: str = "", ncu_rep_path: str = "") -> str:
    """文件外的署名（人能读、可 diff、可归档）：完整署名 + 全部写进去的值。"""
    lines = [f"# {ATTRIBUTION_MARK} attribution / 署名", ""]
    lines.append(f"file: {ncu_rep_path or '(未指定)'}")
    lines.append(f"producer: kernoscope {tool_version}".rstrip())
    lines.append(f"kernel: {plan.source.kernel_name}")
    lines.append(f"grid: {plan.source.grid}  block: {plan.source.block}")
    lines.append("")
    lines.append("## 这些数字不是硬件计数器读数")
    lines.append("")
    lines.append("它们来自替代口径（NVML 设备采样 / CUPTI kernel 活动 / 访存轨迹地址算术 / 模型）。"
                 "在原生 `ncu` 无法运行的机器上，这是我们能给出的最接近的证据；"
                 "出处逐条标注，**不要把 `ks__`/`est__`/`meas__` 当作 ncu 计数器读数引用**。")
    lines.append("")
    lines.append("## 写进报告的值")
    lines.append("")
    lines.append("| metric | value | provenance |")
    lines.append("|---|---|---|")
    for m in plan.metrics:
        lines.append(f"| `{m.name}` | {m.value:.6g} | {plan.provenance_by_name.get(m.name, '')} |")
    if plan.dropped_by_namespace:
        lines.append("")
        lines.append("## 因命名空间规则而未写入的值")
        lines.append("")
        lines.append("这些名字属于 ncu 的命名空间，但两边的**单位语义不确定**"
                     "（例如 `gpu__time_duration.sum`：我们是 ms，ncu 是 ns）。"
                     "写进 `.ncu-rep` 会被按 ncu 的单位理解，所以宁可不写："
                     + ", ".join(f"`{n}`" for n in plan.dropped_by_namespace))
    if plan.dropped_by_refusal:
        lines.append("")
        lines.append("## 因前提不成立而未写入的值")
        lines.append("")
        lines.append("这些指标在决策包里被判词**拒绝**（前提不成立），所以不写进报告："
                     + ", ".join(f"`{n}`" for n in plan.dropped_by_refusal))
    if plan.dropped_over_cap:
        lines.append("")
        lines.append(f"另有 {plan.dropped_over_cap} 个值因数量上限未写入。")
    if plan.premises:
        lines.append("")
        lines.append("## 前提")
        lines.append("")
        for p in plan.premises:
            lines.append(f"- {p}")
    return "\n".join(lines) + "\n"


@dataclass
class SubstituteResult:
    ncu_rep_path: str
    sidecar_path: str
    metrics_written: int
    bytes_written: int
    plan: SubstitutePlan


def write_substitute(kr: Any, out_path: str, *, packet: dict[str, Any] | None = None,
                     tool_version: str = "", provider: str = "kernoscope",
                     dry_run: bool = False) -> SubstituteResult:
    """写 `.ncu-rep` + 旁路署名。任何一条前提不成立就抛 `SubstituteRefused`。

    `provider` 与 `comment` 一起构成文件内署名：**两个都不能为空**（由 `build_report` 强制），
    所以"悄悄地写一份没有署名的 ncu 报告"在代码层面不可写。
    """
    if FILE_VERSION != 7:
        raise SubstituteRefused(
            f"拒绝写出：本工具只会写 v7 报告（当前 FILE_VERSION={FILE_VERSION}）；"
            "格式版本变化必须重新验证 reader 能不能读")
    plan = build_plan(kr, packet=packet, tool_version=tool_version)
    assert_writable(plan.metrics)
    committer = f"{provider}"
    if not plan.comment.strip() or not committer.strip():
        raise SubstituteRefused("拒绝写出：署名不完整（文件内 comment/provider 都为必填）")
    sidecar_path = out_path + ".ATTRIBUTION.txt"
    if dry_run:
        return SubstituteResult(out_path, sidecar_path, len(plan.metrics), 0, plan)
    data = build_report(plan.source, plan.metrics, provider=committer, comment=plan.comment)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(data)
    Path(sidecar_path).write_text(
        render_sidecar(plan, tool_version=tool_version, ncu_rep_path=out_path),
        encoding="utf-8")
    return SubstituteResult(out_path, sidecar_path, len(plan.metrics), len(data), plan)
