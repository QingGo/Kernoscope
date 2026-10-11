"""决策包（W2，机器视图）与一屏（W4，人视图）—— **同一个证据模型的两副面孔**。

问题：一次 `kernoscope` 运行默认产出上百 MB JSON。信息很全，但 agent 要的是
「我该改哪一行代码」，人要看的是「一句话结论 + 前提 + 下一步」。两者都不能从
原始报告里直接读出来。

做法：从 `Report` 抽出**判词（verdict）**这一层，两个视图都只由它渲染：

    Report（原始读数，可能很大）
      └─ packet（判词 + 前提 + 未测到 + 代价，默认 < 4 KB）
           ├─ render_onepage()  人看的一屏（散文在这里生成）
           └─ --json            机器读的结构化包（只有数字与短标识）

**包与视图的分工**（这是能不能压进 4 KB 的关键）：
  * 包只放**每次都变**的东西：判词 id、状态、动态数值（facts）、前提观测值、代价。
  * 静态文本（问题描述、前提标签、拒绝原因的人话）不进包 —— 它们按 id 查表：
    `capabilities.QUESTIONS`、`packet.PREMISE_LABELS`、`refusals.CODE_TITLES`。
    （中文散文在 UTF-8 里 3 字节/字；把静态句子抄进包，等于用预算买重复。）
  * 散文只在 `render_onepage` 里生成。所以「人视图与机器视图分叉」在结构上不可能：
    人视图没有第二个数据来源。

三条纪律：
  1. **大数组永不默认**：`unmeasured` 按 code 归组计数（最多留 2 个例子、6 组），
     需要逐条时读完整报告（包里给出路径与大小）。
  2. **判词带前提**：结论必须列出支撑它的前提值（observed vs threshold）；前提不成立
     时状态只能是 refused —— 不允许「结论在这里、前提在别处」。
  3. **代价可见**：wall 时间、包大小、完整报告大小、launch 数都在 `cost` 里（C 的度量）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import __version__
from .provenance import Metric, Provenance, fmt_bytes, to_bytes
from .refusals import Refusal, RefusalCode, Remedy, code_title, default_remedy

PACKET_SCHEMA_VERSION = 1
# v1.1（追加式，向后兼容）：加 `actionable` 与 `schema_revision`。主版本不动是因为
# **读 v1.0 包的代码仍然能读 v1.1**：新字段是"多给一条已经能推出来的结论"，不是改语义。
# 改了语义才升 PACKET_SCHEMA_VERSION（那时 `from_dict` 要按版本分支，并写进 CHANGELOG）。
PACKET_SCHEMA_REVISION = "1.2"
PACKET_SIZE_BUDGET_BYTES = 4096          # 默认包的尺寸预算（测试守着，防止再长成 150 MB）
UNMEASURED_EXAMPLES_PER_CODE = 2
MAX_UNMEASURED_GROUPS = 6                 # 归组之后仍然只默认给前 6 组
MAX_NEXT_ACTIONS = 4
MAX_VERDICTS = 10
FACT_STR_LIMIT = 48                       # facts 里的字符串（kernel 名等）限长

OK, SOFT, REFUSED = "ok", "soft", "refused"
STATUSES = (OK, SOFT, REFUSED)

# 前提的可读标签是**静态**的（同一 id 每次一样），所以不进包：渲染时查这张表。
# 前提 id → 它不成立时该报哪个 code（拒绝必须可编程：让人知道去修哪一条）
PREMISE_CODES: dict[str, RefusalCode] = {
    "window_coverage_pct": RefusalCode.COVERAGE_TOO_LOW,
    "matched_kernel_kernel_share_pct": RefusalCode.SHARE_TOO_LOW,
    "kernel_time_pct_of_wall": RefusalCode.PREMISE_FAILED,
    "kernel_share_of_total": RefusalCode.PREMISE_FAILED,
    "trace_vs_nvml_pct": RefusalCode.PREMISE_FAILED,
    "over_peak": RefusalCode.CALIBRATION_CALIBER_MISMATCH,
}

PREMISE_LABELS: dict[str, str] = {
    "kernel_share_of_total": "该 kernel 占全部 kernel 时间的比例",
    "kernel_time_pct_of_wall": "kernel 时间占进程 wall 的比例",
    "window_coverage_pct": "窗口是否被被测负载主导",
    "window_mem_busy_pct": "参与积分的窗口内显存忙碌均值",
    "matched_kernel_kernel_share_pct": "被匹配 kernel 是否占窗口 kernel 时间 ≥95%",
    "trace_vs_nvml_pct": "轨迹口径 / NVML 口径",
    "over_peak": "达成值是否超过本机实测峰值",
}


def premise_label(pid: str) -> str:
    return PREMISE_LABELS.get(pid, pid)


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Premise:
    """一条支撑结论的前提。observed 与 threshold 必须同单位，否则读者无法比较。"""

    id: str
    observed: Any
    threshold: Any = None
    ok: bool = True
    unit: str = ""

    @property
    def label(self) -> str:
        return premise_label(self.id)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "observed": self.observed, "ok": self.ok}
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.unit:
            d["unit"] = self.unit
        return d


@dataclass
class Verdict:
    """一条判词。

    `facts` 是**动态**部分（数字、名字），由它 + verdict id 就能在视图侧生成完整句子；
    `claim` 是内存里的人话版本（测试与日志用），**不进包**（散文 3 字节/字，占不起）。
    """

    id: str
    question: str
    status: str
    facts: dict[str, Any] = field(default_factory=dict)
    value: float | int | str | None = None
    unit: str = ""
    provenance: str = ""
    band: dict[str, Any] | None = None      # 误差带（来自 selftest；没有就是没有）
    mdd: float | None = None                # 最小可分辨差异（同单位）
    premises: list[Premise] = field(default_factory=list)
    refusal: Refusal | None = None
    next: str = ""
    metrics: tuple[str, ...] = ()
    claim: str = ""                         # 仅内存：视图侧重新生成，避免两处文案

    @property
    def failed_premises(self) -> list[Premise]:
        return [p for p in self.premises if p.ok is False]

    @property
    def is_actionable(self) -> bool:
        """能不能把这条判词当作行动依据（v1.1 起的显式字段）。"""
        return self.status != REFUSED and not self.failed_premises

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad verdict status {self.status!r}")
        if self.status == REFUSED and self.refusal is None:
            raise ValueError(f"refused verdict {self.id!r} must carry a structured refusal")
        if self.status == REFUSED and self.next:
            object.__setattr__(self, "next", "")   # 拒绝时「下一步」就是 remedy，不重复
        # 构造期**不**报错：前提是构建之后才逐步填齐的。执法放在两个地方：
        #   * build() 里统一降级（正常路径，见 _downgrade_broken_premises）；
        #   * to_dict() 里断言（任何旁路都会立刻炸，而不是把不一致的包发出去）。

    def to_dict(self) -> dict[str, Any]:
        if self.status != REFUSED and self.failed_premises:
            # 前提不成立却把结论标成 ok/soft —— 这正是外部 agent 会踩的坑（它只看 status）。
            raise ValueError(
                f"verdict {self.id!r} is {self.status} but premise(s) "
                f"{[p.id for p in self.failed_premises]} failed - build() must downgrade it "
                f"to refused before serialization")
        d: dict[str, Any] = {"id": self.id, "status": self.status,
                             # 「能不能拿它当行动依据」的**单字段**答案：三个条件
                             # （不是 refused、前提全成立、还在包里）里前两个在这里合取，
                             # 第三个由"它出现在包里"本身保证。agent 因此不必自己推。
                             "actionable": self.is_actionable}
        if self.facts:
            d["facts"] = {k: (_clip(v, FACT_STR_LIMIT) if isinstance(v, str) else v)
                          for k, v in self.facts.items()}
        if self.value is not None:
            d["value"] = self.value
            if self.unit:
                d["unit"] = self.unit
        if self.provenance:
            d["provenance"] = self.provenance
        if self.band is not None:
            d["band"] = self.band
        if self.mdd is not None:
            d["mdd"] = self.mdd
        if self.premises:
            d["premises"] = [p.to_dict() for p in self.premises]
        if self.refusal is not None:
            # 归一化：文案与 remedy 只在 packet["refusals"] 里存一份（按 code 索引）
            d["refusal"] = {"code": self.refusal.code.value}
            if self.refusal.observed:
                d["refusal"]["observed"] = dict(self.refusal.observed)
            if self.refusal.threshold:
                d["refusal"]["threshold"] = dict(self.refusal.threshold)
        if self.next:
            d["next"] = _clip(self.next, 120)
        if self.metrics:
            d["metrics"] = list(self.metrics)[:3]
        return d


# --------------------------------------------------------------------------
# 从 Report 里读指标
# --------------------------------------------------------------------------
def find_metric(rep: Any, name: str) -> Metric | None:
    for k in getattr(rep, "kernels", []):
        for s in getattr(k, "sections", []):
            for m in s.metrics:
                if m.name == name:
                    return m
    return None


def metric_value(rep: Any, name: str) -> float | int | None:
    m = find_metric(rep, name)
    if m is None or m.value is None:
        return None
    return m.value  # type: ignore[return-value]


def metric_bytes(rep: Any, name: str) -> float | None:
    """把指标值折成**字节**，单位只从 `Metric.unit` 一处读（provenance.to_bytes）。

    决策包里 `dram_bytes` 一族的契约是字节（`unit="B"`、`facts.mb`、claim 都从它派生），
    而真机 metric 的 unit 是 MB。历史上这里写的是 `float(m.value) / 1e6` —— 把 MB 当字节，
    于是 7.27 TB 被写成 7.3 MB。所以：**任何按字节消费 dram 指标的地方都必须走这个函数**。
    """
    m = find_metric(rep, name)
    if m is None or m.value is None:
        return None
    return to_bytes(float(m.value), getattr(m, "unit", ""))  # type: ignore[arg-type]


def _prov(rep: Any, name: str) -> str:
    m = find_metric(rep, name)
    return m.provenance.value if m is not None else ""


def _ms(ns: float) -> float:
    return round(ns / 1e6, 3)


# --------------------------------------------------------------------------
# 判词构造：每条只在数据支持时出现（不造空结论）
# --------------------------------------------------------------------------
def _verdict_kernel_time(rep: Any) -> Verdict | None:
    kernels = [k for k in getattr(rep, "kernels", []) if k.name != "(device)" and k.total_ns]
    if not kernels:
        return None
    kernels.sort(key=lambda k: k.total_ns, reverse=True)
    top = kernels[0]
    total = sum(k.total_ns for k in kernels)
    share = 100.0 * top.total_ns / total if total else 0.0
    wall = getattr(rep, "wall_time_ns", 0) or 0
    premises = [Premise("kernel_share_of_total", round(share, 1), unit="%")]
    if wall:
        premises.append(Premise("kernel_time_pct_of_wall",
                                round(100.0 * total / wall, 1), unit="%"))
    return Verdict(
        id="kernel_time", question="哪个 kernel 最耗时、每次跑多久", status=OK,
        facts={"top": top.name, "top_ms": _ms(top.total_ns), "launches": top.launch_count,
               "share_pct": round(share, 1), "total_ms": _ms(total),
               "n_kernels": len(kernels)},
        value=_ms(top.total_ns), unit="ms",
        provenance=_prov(rep, "gpu__time_duration.sum") or "exact", premises=premises,
        metrics=("gpu__time_duration.sum", "ks__launch_time_mean"),
        next="用 -k 只选这个 kernel 再跑一次，拿到它的 DRAM/occupancy/roofline 判词",
        claim=f"最耗时的是 {top.name}（{_ms(top.total_ns)} ms / {top.launch_count} 次，"
              f"占 kernel 时间 {share:.1f}%），进程内 kernel 时间合计 {_ms(total)} ms")


def _refused(id_: str, question: str, m: Metric, premises: list[Premise],
             metrics: tuple[str, ...]) -> Verdict:
    return Verdict(id=id_, question=question, status=REFUSED, refusal=m.refusal,
                   premises=premises, metrics=metrics,
                   claim=f"不给数字：{code_title(m.refusal.code) if m.refusal else ''}"
                         f" —— {(m.reason or '')[:120]}")


def _verdict_dram_total(rep: Any) -> Verdict | None:
    m = find_metric(rep, "est__dram_bytes_total")
    if m is None:
        return None
    premises: list[Premise] = []
    cov = metric_value(rep, "ks__sampler_window_coverage_pct")
    if cov is not None:
        premises.append(Premise("window_coverage_pct", cov, 50.0, float(cov) >= 50.0, "%"))
    busy = metric_value(rep, "ks__dram_window_mem_busy_mean_pct")
    if busy is not None:
        premises.append(Premise("window_mem_busy_pct", busy, unit="%"))
    if m.provenance is Provenance.UNAVAILABLE:
        return _refused("dram_bytes", "DRAM 搬了多少字节", m, premises,
                        ("est__dram_bytes_total",))
    nbytes = metric_bytes(rep, "est__dram_bytes_total")
    if nbytes is None:          # 不变量上不可达；真到了这里，宁可不给判词也不给错的量级
        return None
    return Verdict(
        id="dram_bytes", question="DRAM 搬了多少字节 / 是不是 DRAM bound", status=SOFT,
        facts={"mb": round(nbytes / 1e6, 1)},
        value=round(nbytes, 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_total", "ks__sampler_window_coverage_pct"),
        next="与你的账本对拍（--ks-ledger-total-mb）确认这个量级是否符合预期",
        claim=f"整窗口 DRAM 约 {fmt_bytes(nbytes)}（代理口径：NVML 显存忙碌时间 × "
              f"本机实测峰值），前提见 premises")


def _verdict_per_launch(rep: Any) -> Verdict | None:
    m = find_metric(rep, "est__dram_bytes_per_launch")
    if m is None:
        return None
    share = metric_value(rep, "ks__matched_kernel_kernel_share_pct")
    cov = metric_value(rep, "ks__sampler_window_coverage_pct")
    premises: list[Premise] = []
    if cov is not None:
        premises.append(Premise("window_coverage_pct", cov, 80.0, float(cov) >= 80.0, "%"))
    premises.append(Premise("matched_kernel_kernel_share_pct", share, 95.0,
                            share is not None and float(share) >= 95.0, "%"))
    if m.provenance is Provenance.UNAVAILABLE:
        return _refused("dram_bytes_per_launch", "每次 launch 搬多少字节", m, premises,
                        ("est__dram_bytes_per_launch",))
    nbytes = metric_bytes(rep, "est__dram_bytes_per_launch")
    if nbytes is None:
        return None
    bias = metric_value(rep, "ks__attribution_bias_estimate_pct")
    facts: dict[str, Any] = {"mb": round(nbytes / 1e6, 2)}
    if bias:
        facts["bias_pct"] = round(float(bias))
    return Verdict(
        id="dram_bytes_per_launch", question="每次 launch 搬多少字节", status=SOFT,
        facts=facts, value=round(nbytes, 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_per_launch", "ks__attribution_bias_estimate_pct"),
        next="改一次 kernel 再跑 diff：差值小于 MDD 就是噪声，不要当成改进",
        claim=f"每次 launch 约 {fmt_bytes(nbytes)}（代理口径）"
              + (f"；归因上偏估计 +{bias:.0f}%" if bias else ""))


def _verdict_trace(rep: Any) -> Verdict | None:
    m = find_metric(rep, "est__dram_bytes_per_launch_trace")
    if m is None:
        return None
    premises: list[Premise] = []
    ratio = metric_value(rep, "ks__dram_trace_vs_nvml_pct")
    if ratio is not None:
        premises.append(Premise("trace_vs_nvml_pct", ratio, unit="%"))
    if m.provenance is Provenance.UNAVAILABLE:
        return _refused("dram_bytes_trace", "DRAM 字节的第二条口径（访存轨迹）", m, premises,
                        ("est__dram_bytes_per_launch_trace",))
    nbytes = metric_bytes(rep, "est__dram_bytes_per_launch_trace")
    if nbytes is None:
        return None
    return Verdict(
        id="dram_bytes_trace", question="DRAM 字节的第二条口径（访存轨迹）", status=SOFT,
        facts={"mb": round(nbytes / 1e6, 2)},
        value=round(nbytes, 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_per_launch_trace",),
        next="两条口径差得远时先查采样代表性与 L2 模型参数，不要挑好看的那个",
        claim=f"轨迹口径每次 launch 约 {fmt_bytes(nbytes)}（L2 命中模型）")


def _verdict_ledger(rep: Any) -> Verdict | None:
    err = metric_value(rep, "est__dram_vs_ledger_err_pct")
    pos = metric_value(rep, "ks__ledger_range_verdict")
    if err is None and pos is None:
        return None
    if err is None:
        return Verdict(
            id="ledger_verdict", question="估算落在你给的账本区间哪一侧", status=SOFT,
            facts={"position": _clip(str(pos), FACT_STR_LIMIT)}, provenance="derived",
            metrics=("ks__ledger_range_verdict",),
            next="误差在误差带内就别改代码；超出再看口径是不是可比",
            claim=str(pos))
    return Verdict(
        id="ledger_verdict", question="估算与账本的相对误差", status=SOFT,
        facts={"err_pct": round(float(err), 1)}, value=round(float(err), 1), unit="%",
        provenance="derived", metrics=("est__dram_vs_ledger_err_pct",),
        next="误差超出本机误差带才值得继续优化；否则先改测量口径",
        claim=f"与账本相差 {float(err):+.1f}%（同口径对拍）")


def _verdict_occupancy(rep: Any) -> Verdict | None:
    theo = metric_value(rep, "ks__theoretical_occupancy_pct")
    ach = metric_value(rep, "meas__achieved_occupancy_pct")
    if theo is None and ach is None:
        return None
    limiter = metric_value(rep, "ks__occupancy_limiter")
    facts: dict[str, Any] = {}
    if theo is not None:
        facts["theoretical_pct"] = theo
    if ach is not None:
        facts["achieved_pct"] = ach
    if limiter:
        facts["limiter"] = _clip(str(limiter), FACT_STR_LIMIT)
    parts = []
    if theo is not None:
        parts.append(f"理论上界 {theo}%")
    if ach is not None:
        parts.append(f"实测达到 {ach}%")
    if limiter:
        parts.append(f"限制因素 {limiter}")
    return Verdict(
        id="occupancy", question="occupancy 上界与实际达到值",
        status=OK if ach is not None else SOFT, facts=facts,
        value=ach if ach is not None else theo, unit="%",
        provenance=_prov(rep, "meas__achieved_occupancy_pct") or "exact",
        metrics=("ks__theoretical_occupancy_pct", "meas__achieved_occupancy_pct"),
        next=("实测远低于理论上界时查 tail effect / 小 grid；接近上界时别再调 occupancy"
              if ach is not None else "装上 NVBit 探针可拿到实测 occupancy（不只是上界）"),
        claim="；".join(parts))


def _verdict_roofline(rep: Any) -> Verdict | None:
    bound = metric_value(rep, "est__bound")
    eff = metric_value(rep, "est__roofline_efficiency_pct")
    gbs = metric_value(rep, "est__achieved_gbs")
    if bound is None and eff is None:
        return None
    facts: dict[str, Any] = {}
    if bound is not None:
        facts["bound"] = _clip(str(bound), FACT_STR_LIMIT)
    if eff is not None:
        facts["efficiency_pct"] = eff
    if gbs is not None:
        facts["gbs"] = gbs
    bits = []
    if bound is not None:
        bits.append(f"判定为 {bound} bound")
    if eff is not None:
        bits.append(f"离 roofline {100.0 - float(eff):.1f}%")
    if gbs is not None:
        bits.append(f"达成带宽 {gbs} GB/s")
    # 「下一步」必须看数据：达成率已经很高时还说"去查访问模式"是模板话术，会把 agent
    # 送去抠已经不存在的空间（这是 W8 冷启动评测里被测 agent 亲自指出的矛盾）。
    if eff is not None and float(eff) >= 90.0:
        nxt = (f"已达 roofline {float(eff):.1f}%：先核对工作量/账本输入（--ks-gemm/--ks-flops）"
               f"是否正确；若要再进一步，只能减少总字节（算子融合 / 提高 L2 复用）")
    elif bound is not None and str(bound) == "memory":
        nxt = "memory bound 且达成率低 → 先查访问模式（L2 命中/合并效率）与工作集是否驻留"
    elif bound is not None:
        nxt = "compute bound → 查指令混合与发射槽利用率上界（--ks-instr-hist）"
    else:
        nxt = "先确认工作量输入与标定口径，再看达成率"
    return Verdict(
        id="roofline", question="离 roofline 多远、compute 还是 memory bound", status=SOFT,
        facts=facts, value=eff, unit="%", provenance="estimated",
        metrics=("est__bound", "est__roofline_efficiency_pct", "est__achieved_gbs"),
        next=nxt, claim="；".join(str(b) for b in bits))


def _verdict_issue_slot(rep: Any) -> Verdict | None:
    util = metric_value(rep, "est__issue_slot_utilization_pct")
    direction = metric_value(rep, "est__stall_direction")
    if util is None and direction is None:
        return None
    facts: dict[str, Any] = {}
    if util is not None:
        facts["util_pct"] = util
    if direction is not None:
        facts["direction"] = _clip(str(direction), FACT_STR_LIMIT)
    return Verdict(
        id="issue_slot", question="发射槽用了多少、瓶颈方向", status=SOFT, facts=facts,
        value=util, unit="%", provenance="estimated",
        metrics=("est__issue_slot_utilization_pct", "est__stall_direction"),
        next="方向是 latency 就先加并行度/减少依赖链；是 throughput 才谈指令精简",
        claim=("发射槽利用率上界 " + (f"{float(util):.1f}%" if util is not None else "n/a")
               + (f"；瓶颈方向 {direction}" if direction else "")))


def _verdict_l2(rep: Any) -> Verdict | None:
    hit = metric_value(rep, "est__l2_hit_rate_pct")
    ws = metric_value(rep, "est__working_set_bytes")
    if hit is None and ws is None:
        return None
    facts: dict[str, Any] = {}
    if hit is not None:
        facts["hit_pct"] = hit
    if ws is not None:
        facts["working_set_mib"] = round(float(ws) / 2 ** 20, 1)
    bits = []
    if hit is not None:
        bits.append(f"L2 命中率模型 {hit}%")
    if ws is not None:
        bits.append(f"工作集 {float(ws) / 2 ** 20:.1f} MiB")
    return Verdict(
        id="l2_locality", question="L1/L2 命中率与工作集", status=SOFT, facts=facts,
        value=hit, unit="%", provenance="estimated",
        metrics=("est__l2_hit_rate_pct", "est__working_set_bytes"),
        next="命中的部分不该算进 DRAM 账：对拍 DRAM 前先确认工作集能不能驻留",
        claim="；".join(bits))


def _verdict_over_peak(rep: Any) -> Verdict | None:
    warn = metric_value(rep, "ks__over_peak_warning")
    if not warn:
        return None
    m = find_metric(rep, "ks__over_peak_warning")
    return Verdict(
        id="over_peak", question="达成值是否超过标定峰值（物理上不可能）", status=REFUSED,
        premises=[Premise("over_peak", 1.0, None, False)],
        refusal=Refusal(RefusalCode.CALIBRATION_CALIBER_MISMATCH,
                        (m.note if m is not None else "") or "achieved > measured peak"),
        metrics=("ks__over_peak_warning",),
        next="重跑标定（tools/machine-profile.py），并确认 --ks-gemm/--ks-flops 的工作量正确",
        claim="达成值超过本机实测峰值：本次的百分比不成立，先查标定与工作量输入")


def _verdict_static_mix(rep: Any) -> Verdict | None:
    total = metric_value(rep, "ks__static_sass_instructions")
    if total is None:
        return None
    mix: dict[str, float] = {}
    for name in ("ks__static_sass_share_fp", "ks__static_sass_share_memory",
                 "ks__static_sass_share_control", "ks__static_sass_share_int"):
        v = metric_value(rep, name)
        if v is not None:
            mix[name.rsplit("_", 1)[-1]] = v
    return Verdict(
        id="static_mix", question="静态指令混合（哪些指令占多数）", status=OK,
        facts={"total": int(total), "mix": mix}, value=int(total), provenance="exact",
        metrics=("ks__static_sass_instructions",),
        next="静态混合只说明代码长相：动态占比要靠动态指令计数（--ks-instr-hist）",
        claim=f"SASS 共 {int(total)} 条"
              + ("；" + "，".join(f"{k} {v}%" for k, v in mix.items()) if mix else ""))


# 判词目录（catalog）：**决策包里可能出现哪些判词 id** 的单一来源。
# 生成类文档（给 agent 的参考层）与测试都读它，避免"文档漏了一个判词"这种事。
# 有测试守它与 BUILDERS 实际能产出的 id 完全一致。
VERDICT_IDS: tuple[str, ...] = (
    "kernel_time",
    "dram_bytes",
    "dram_bytes_per_launch",
    "dram_bytes_trace",
    "ledger_verdict",
    "occupancy",
    "roofline",
    "issue_slot",
    "l2_locality",
    "over_peak",
    "static_mix",
)

BUILDERS = (_verdict_kernel_time, _verdict_dram_total, _verdict_per_launch, _verdict_trace,
            _verdict_ledger, _verdict_occupancy, _verdict_roofline, _verdict_issue_slot,
            _verdict_l2, _verdict_over_peak, _verdict_static_mix)


# --------------------------------------------------------------------------
# 未测到的指标：按 code 归组，绝不默认铺开大数组
# --------------------------------------------------------------------------
def group_unmeasured(rep: Any) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for k in getattr(rep, "kernels", []):
        for s in getattr(k, "sections", []):
            for m in s.metrics:
                if m.provenance is not Provenance.UNAVAILABLE:
                    continue
                code = m.refusal.code.value if m.refusal else RefusalCode.LEGACY_REPORT.value
                g = groups.setdefault(code, {"code": code, "count": 0, "examples": []})
                g["count"] += 1
                if (len(g["examples"]) < UNMEASURED_EXAMPLES_PER_CODE
                        and m.name not in g["examples"]):
                    g["examples"].append(_clip(m.name, 46))
    return sorted(groups.values(), key=lambda g: (-g["count"], g["code"]))


def _window_mode(rep: Any) -> str:
    """只留窗口**模式**：整段窗口描述里有 wall 时间等每次都变的数字。"""
    w = str((getattr(rep, "meta", {}) or {}).get("window", ""))
    return w.split(" ")[0].replace("mode=", "") if w else ""


def protocol_hash(rep: Any, extra: dict[str, Any] | None = None) -> str:
    """本次测量的**协议**哈希：协议不同 ⇒ 两次结果不可比（diff 用）。

    协议 = 会影响数字口径的东西：工具版本、schema、标定来源、精度、窗口模式、
    warmup、kernel 选择、峰值来源、设备。**不包含**时间戳、主机名、wall 时间
    —— 那些每次都变，会把「同协议」误判成不同。
    """
    meta = getattr(rep, "meta", {}) or {}
    device = getattr(rep, "device", None)
    payload = {
        "tool_version": getattr(rep, "tool_version", __version__),
        "packet_schema": PACKET_SCHEMA_VERSION,
        "report_schema": getattr(rep, "schema_version", None),
        "dtype": meta.get("dtype"),
        "window_mode": _window_mode(rep),
        "warmup_s": meta.get("warmup_s"),
        "window_s": meta.get("window_s"),
        "calibration": meta.get("calibration") or meta.get("calibration_source"),
        "peaks": meta.get("peak_source"),
        "kernel": meta.get("kernel_filter"),
        "ledger_basis": meta.get("ledger_basis"),
        "device": (getattr(device, "name", "") if device is not None else ""),
        "compute_capability": (getattr(device, "compute_capability", "")
                               if device is not None else ""),
        "synthetic": bool(meta.get("synthetic")),
        "degraded": meta.get("degraded") or "",
    }
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# 组包
# --------------------------------------------------------------------------
def _propagate_over_peak(verdicts: list[Verdict]) -> None:
    """`over_peak` 成立 ⇒ 由它算出来的 roofline 判词不能当依据（R-59/D18）。

    真机包实测：`roofline status=soft actionable=True value=24041.6 %` 与
    `over_peak status=refused` **并存** —— 按 v1.1 自己定的规矩（"读包的人用 actionable
    判断可依据性"），那个物理不可能的达成率是"可依据"的，`diff` 还会拿它报 improved。
    `ledger_verdict` 早就会继承 per-launch 的拒绝（同一个模式），roofline 漏了。
    """
    op = next((v for v in verdicts if v.id == "over_peak"), None)
    roof = next((v for v in verdicts if v.id == "roofline"), None)
    if op is None or op.status != REFUSED or roof is None or roof.status == REFUSED:
        return
    if roof.value is not None:
        roof.facts = {**roof.facts, "uncitable_value": roof.value}
    roof.status = REFUSED
    roof.value = None
    roof.refusal = Refusal(
        RefusalCode.CALIBRATION_CALIBER_MISMATCH,
        "达成值超过本机实测峰值（物理上不可能）⇒ roofline 的达成率不成立；"
        "先核对标定口径（dtype/尺寸）与 --ks-gemm/--ks-flops 的工作量，再谈优化方向")
    # 旧 hint 是"已达 roofline 91.2%…"（在没被推翻时是对的）；被推翻后必须换成
    # 与拒绝一致的动作，否则同一行里一边说"不成立"一边说"已达"。
    roof.next = ("先核对标定口径（dtype/尺寸）与工作量输入（--ks-gemm/--ks-flops），"
                 "再谈优化方向")


def _downgrade_broken_premises(verdicts: list[Verdict]) -> None:
    """前提不成立的判词**必须**变成 refused（外部评测暴露的漏洞，见下）。

    真事：一个 0.9 s 窗口里 4 个 kernel，被匹配 kernel 只占 41%（门槛 95%），
    `dram_bytes_per_launch` 的 status 却还是 `soft` —— agent 只看 status，于是把它
    （以及由它推出的账本误差 +146.9%）当成结论。值本身没错，错的是**允许它带着一个
    不成立的前提被引用**。所以这里统一降级：status=refused + 按前提给 code + 保留历史值
    在 facts 里（读者仍能看到"它曾经算出过多少"，但明确不可引用）。
    """
    for v in verdicts:
        failed = v.failed_premises
        if v.status == REFUSED or not failed:
            continue
        main = failed[0]
        code = PREMISE_CODES.get(main.id, RefusalCode.PREMISE_FAILED)
        msg = (f"前提不成立（{premise_label(main.id)}：实测 {main.observed}"
               f"{main.unit}，门槛 {main.threshold}{main.unit}）⇒ 这个数字不可引用；"
               f"数值 {v.value} {v.unit} 只作历史记录")
        facts = dict(v.facts)
        if v.value is not None:
            facts["uncitable_value"] = v.value
        v.status = REFUSED
        v.refusal = Refusal(code, msg, observed={main.id: main.observed},
                            threshold={main.id: main.threshold})
        v.value = None
        v.facts = facts


def build(rep: Any, *, capabilities: Any = None, selfcheck: dict[str, Any] | None = None,
          basis: str = "", cost_extra: dict[str, Any] | None = None,
          report_path: str = "") -> dict[str, Any]:
    """由 Report 组出决策包（不读文件、不跑命令，纯函数便于测试）。"""
    verdicts_all = [v for fn in BUILDERS if (v := fn(rep)) is not None]
    verdicts = verdicts_all[:MAX_VERDICTS]
    # R-53/D10：上限截断必须**留痕** —— 否则读者（人或 agent）不知道包里少了一条，
    # 而 `cost.trimmed` 只记预算裁剪、记不到这一步（它在预算裁剪之前就发生了）。
    verdicts_more = ({"count": len(verdicts_all) - MAX_VERDICTS,
                      "ids": [v.id for v in verdicts_all[MAX_VERDICTS:]]}
                     if len(verdicts_all) > MAX_VERDICTS else None)
    _downgrade_broken_premises(verdicts)
    _propagate_over_peak(verdicts)

    band = ((selfcheck or {}).get("band") or {}).get("dram_bytes_pct")
    mdd_dram = ((selfcheck or {}).get("mdd") or {}).get("dram_bytes_pct")
    mdd_kernel = ((selfcheck or {}).get("mdd") or {}).get("kernel_time_pct")
    for v in verdicts:
        if v.id.startswith("dram_bytes"):
            if band:
                v.band = dict(band)
            if mdd_dram is not None:
                v.mdd = float(mdd_dram)
        elif v.id in ("kernel_time", "occupancy") and mdd_kernel is not None:
            v.mdd = float(mdd_kernel)

    unmeasured_all = group_unmeasured(rep)
    unmeasured = unmeasured_all[:MAX_UNMEASURED_GROUPS]
    unmeasured_more = None
    if len(unmeasured_all) > len(unmeasured):
        unmeasured_more = {"groups": len(unmeasured_all) - len(unmeasured),
                           "metrics": sum(g["count"] for g in unmeasured_all[len(unmeasured):])}

    # 账本裁决继承它所用的那条口径：per-launch 对拍用的是 per-launch 数字，
    # 而那个数字不可归因时，误差百分比同样不可引用（外部评测里 agent 正是在这里踩的）。
    if basis.startswith("per-launch") or basis in ("", "auto"):
        pl = next((v for v in verdicts if v.id == "dram_bytes_per_launch"), None)
        lg = next((v for v in verdicts if v.id == "ledger_verdict"), None)
        if (pl is not None and pl.status == REFUSED and lg is not None
                and lg.value is not None):
            lg.facts = {**lg.facts, "uncitable_value": lg.value}
            lg.status = REFUSED
            lg.refusal = Refusal(
                pl.refusal.code if pl.refusal else RefusalCode.PREMISE_FAILED,
                "账本对拍用的就是那个不可归因的 per-launch 数字 ⇒ 误差百分比不可引用；"
                "要么缩小窗口让它可归因，要么改用 --ks-ledger-total-mb 走整窗口口径")
            lg.value = None

    env: dict[str, Any] = {}
    if capabilities is not None:
        env = {"tier": capabilities.tier,
               "tier_name": capabilities.tier_obj.name,
               "counter_state": capabilities.env.counter_state,
               "blocked_by": sorted({d["code"] for d in capabilities.next_actions()})[:6]}

    cost: dict[str, Any] = {
        "wall_s": round((getattr(rep, "wall_time_ns", 0) or 0) / 1e9, 3),
        "launches": sum(len(getattr(k, "launches", [])) for k in getattr(rep, "kernels", [])),
        "unmeasured_metrics": sum(g["count"] for g in unmeasured_all),
        "selftest_band": bool(band),
    }
    if report_path and os.path.exists(report_path):
        cost["report_bytes"] = os.path.getsize(report_path)
    if cost_extra:
        cost.update(cost_extra)

    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "schema_revision": PACKET_SCHEMA_REVISION,
        "tool": "kernoscope",
        "tool_version": getattr(rep, "tool_version", __version__),
        "protocol": {"hash": protocol_hash(rep), "basis": basis,
                     "window_mode": _window_mode(rep)},
        "env": env,
        "verdicts": [v.to_dict() for v in verdicts],
        "unmeasured": unmeasured,
        "refusals": _refusal_index(verdicts, unmeasured_all, capabilities,
                                   _env_refusals(rep)),
        "next": _next_actions(verdicts, unmeasured),
        "cost": cost,
    }
    if unmeasured_more:
        packet["unmeasured_more"] = unmeasured_more
    if verdicts_more:
        packet["verdicts_more"] = verdicts_more
    if getattr(rep, "meta", {}).get("synthetic"):
        packet["synthetic"] = True
    for d in _env_refusals(rep):
        packet.setdefault("environment_refusals", []).append(
            {"code": d.get("code"), "message": _clip(str(d.get("message", "")), 120)})
    if report_path:
        packet["report_path"] = report_path
    enforce_budget(packet)
    return packet


def enforce_budget(packet: dict[str, Any],
                   limit: int = PACKET_SIZE_BUDGET_BYTES) -> list[str]:
    """把包压进尺寸预算，并如实记录**裁掉了什么**（返回被裁剪的步骤名）。

    尺寸是 A 的一部分，所以它是一条**硬**约束：不管报告里有多少判词与拒绝，
    默认包都不许超过预算。做法是有序降级（先去掉最能"再生"的东西）；
    下面 `steps` 里的顺序**就是**实际顺序（这条注释与实现逐字一致，改一处就改两处）：

        未测到的例子 > 判词的指标名 > 下一步指针 > 未测到归组数 > 成立的前提 >
        remedy 命令 > 处置文本 > 判词条数 > 判词 facts > 拒绝条目数 > 判词硬上限 >
        环境降级诊断的 message > 环境降级诊断条目数

    每一步都只删"能从完整报告里再取回来"的东西，并把步骤名写进 `cost.trimmed` ——
    读者（人或 agent）因此知道包是摘要，而不是以为这就是全部。最后一级丢掉的
    拒绝 code 会列在 `refusals_omitted` 里，所以"少了什么"也是可查的。
    """
    trimmed: list[str] = []
    # `cost.trimmed` 自己也是包的一部分，所以先放好空列表再量，之后每应用一步就把它
    # 记进去再量 —— 否则会出现「刚量完 4090 B，写进裁剪步骤后变成 4119 B」这种
    # 自我欺骗（真事：加 actionable 字段后正好卡在这个缝上）。
    packet.setdefault("cost", {})["trimmed"] = []

    def _size() -> int:
        return len(dumps(packet).encode("utf-8"))

    if _size() <= limit:
        return trimmed

    def _drop_passing_premises() -> None:
        """只保留**不成立**的前提（成立的前提值能从报告里再看）。"""
        for v in packet["verdicts"]:
            ps = v.get("premises") or []
            if len(ps) > 1:
                v["premises"] = [p for p in ps if not p.get("ok")] or ps[:1]

    def _shorten_refusal_actions() -> None:
        """把每条处置文本截短：保留 code 与"往哪走"的骨架。

        完整处置在报告的 refusals 表与参考层文档里（agent 需要细看时按 code 去查）。
        """
        for r in (packet.get("refusals") or {}).values():
            a = r.get("action")
            if isinstance(a, str) and len(a) > REFUSAL_ACTION_LIMIT:
                r["action"] = a[:REFUSAL_ACTION_LIMIT - 1] + "…"

    def _cap_refusal_entries() -> None:
        """只保留前几条拒绝条目，**把丢掉的 code 列出来**（列不全时明确标出）。

        为什么可以丢条目但不许悄悄丢：读者仍然知道"曾经有哪些 code"与"还差几个"
        （`refusals_omitted`），以及完整处置在哪 —— 这比让整包超预算
        （agent 按 <4 KB 规划就失效）要好。
        """
        refs = packet.get("refusals") or {}
        if len(refs) <= REFUSAL_KEEP:
            return
        # R-65/N2：**先保被引用的**。旧实现按字母序留 8 条，于是 `unmeasured`（另按 3 组
        # 截断）引用的 code 可能整条消失，又不在 `refusals_omitted` 里 —— 一屏会印
        # "还有 N 个 `no_occupancy_probe`"却不给办法，而受限环境正是命中它的场景
        # （47 个构造场景里 47 个都有悬空 code）。
        cited = _cited_codes(packet)
        keep = [c for c in sorted(refs) if c in cited][:REFUSAL_KEEP]
        for c in sorted(refs):
            if len(keep) >= REFUSAL_KEEP:
                break
            if c not in keep:
                keep.append(c)
        keep_set = set(keep)
        dropped = [c for c in sorted(refs) if c not in keep_set]
        packet["refusals"] = {c: refs[c] for c in keep}
        # 丢掉的名字**先列被引用的**（否则不变量在列表截断时会破），再列其余的
        ordered = ([c for c in dropped if c in cited] + [c for c in dropped if c not in cited])
        # 上限自适应：被丢掉的**被引用** code 必须全部列出来（不变量优先于字节），
        # 其余名字才允许只留计数。真实被引用集合很小（未测组 ≤3 + 判词 + 环境缺件）。
        limit = max(REFUSAL_LIST_CODES, len([c for c in dropped if c in cited]))
        omitted: dict[str, Any] = {"count": len(dropped), "codes": ordered[:limit]}
        if len(dropped) > limit:
            omitted["codes_truncated"] = True
        packet["refusals_omitted"] = omitted

    def _drop_refusal_action_text() -> None:
        """最后一级：保留条目骨架（code / 代价），丢掉处置散文。

        处置的完整文本在报告的 refusals 表与参考层文档里（`remedy.sh <code>` 一条命令就能
        查到），所以包里留 code + 代价即可 —— 读者仍然知道"这类拒绝该怎么办、要花多少"。
        """
        for r in (packet.get("refusals") or {}).values():
            r.pop("action", None)

    def _trim_environment_messages() -> None:
        """环境降级诊断的 message 最先丢（code 与 remedy 才是机器契约）。"""
        for e in packet.get("environment_refusals") or []:
            e.pop("message", None)

    def _cap_environment_entries() -> None:
        """R-70/D9：`environment_refusals` 此前**完全不参与预算**（30 条 × 120 字 → 7012 B
        仍带 `over_budget=True` 发出）。现在只留前几条，丢掉的 code 列进
        `environment_refusals_more`（它们的 remedy 仍在 `refusals` 表里，所以照样查得到）。
        """
        env = packet.get("environment_refusals") or []
        if len(env) <= ENVIRONMENT_KEEP:
            return
        dropped = [e.get("code") for e in env[ENVIRONMENT_KEEP:] if e.get("code")]
        packet["environment_refusals"] = env[:ENVIRONMENT_KEEP]
        packet["environment_refusals_more"] = {"count": len(env) - ENVIRONMENT_KEEP,
                                               "codes": dropped[:REFUSAL_LIST_CODES]}
        if len(dropped) > REFUSAL_LIST_CODES:
            packet["environment_refusals_more"]["codes_truncated"] = True

    def _cap_verdicts_hard() -> None:
        packet["verdicts"] = (packet.get("verdicts") or [])[:VERDICT_KEEP_HARD]

    steps = [
        ("unmeasured_examples", lambda: [g.pop("examples", None) for g in packet["unmeasured"]]),
        ("verdict_metrics", lambda: [v.pop("metrics", None) for v in packet["verdicts"]]),
        ("next_pointers", lambda: packet.pop("next", None)),
        ("unmeasured_groups", lambda: _cap_list(packet, "unmeasured", 3)),
        ("passing_premises", _drop_passing_premises),
        ("remedy_commands",
         lambda: [r.pop("cmd", None) for r in (packet.get("refusals") or {}).values()]),
        # R-52：上面几步只删了 refusal 的 `cmd` 子字段，**条目本身不参与裁剪** —— 受限环境
        # 天然产生十几个 code，包会常态化突破预算（实测 45 个 code → 8883 B）。
        # 所以再加两级：先把每条处置文本截短（code 与"怎么办"的骨架保留），
        # 最后才丢条目、并把丢掉的 **code 列出来**（读者知道少了什么、去哪看）。
        ("refusal_actions", _shorten_refusal_actions),
        ("verdicts", lambda: _cap_list(packet, "verdicts", 6)),
        ("facts", lambda: [v.pop("facts", None) for v in packet["verdicts"]]),
        ("refusal_entries", _cap_refusal_entries),
        ("refusal_action_text", _drop_refusal_action_text),
        ("verdicts_hard_cap", _cap_verdicts_hard),
        # 环境降级诊断：message 先丢，再限条目数（R-70）
        ("environment_messages", _trim_environment_messages),
        ("environment_entries", _cap_environment_entries),
    ]
    for name, fn in steps:
        fn()
        packet["cost"]["trimmed"].append(name)
        if _size() <= limit:
            break
    if _size() > limit:
        # 极限压缩后仍然超预算（只在把 limit 调到远小于默认值时发生）：如实标记，
        # 不去改 `packet_bytes` 那个数 —— 它必须等于真实字节数（有测试守）。
        packet["cost"]["over_budget"] = True
    return list(packet["cost"]["trimmed"])


REFUSAL_ACTION_LIMIT = 72     # 截短后的处置文本上限（完整版在报告与参考层里）
REFUSAL_KEEP = 8              # 最多保留几条拒绝条目（丢掉的名字列进 refusals_omitted）
ENVIRONMENT_KEEP = 4          # 最多保留几条环境降级诊断（其余 code 列进 *_more）
REFUSAL_LIST_CODES = 4        # refusals_omitted 里最多列几个 code（再多就只给计数）
VERDICT_KEEP_HARD = 4         # 最后一级：判词只留最关键的四条


def _cap_list(packet: dict[str, Any], key: str, n: int) -> None:
    items = packet.get(key) or []
    if len(items) > n:
        packet[key] = items[:n]
        if key == "unmeasured":
            packet["unmeasured_more"] = {
                "groups": len(items) - n,
                "metrics": sum(g.get("count", 0) for g in items[n:]),
            }


def _next_actions(verdicts: Sequence[Verdict],
                  unmeasured: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """下一步：`{for, code}` —— 具体动作与代价按 code 去 `refusals` 里查。

    只给指针不给文本：同一句 remedy 会在多个判词上重复，而包的预算要花在**判词**上。
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for v in verdicts:
        if v.status == REFUSED and v.refusal is not None:
            code = v.refusal.code.value
            if code not in seen:
                seen.add(code)
                out.append({"for": v.id, "code": code})
        elif v.status != REFUSED and v.next and v.id in ("dram_bytes", "dram_bytes_per_launch",
                                                         "kernel_time"):
            out.append({"for": v.id, "hint": _clip(v.next, 90)})
    for g in unmeasured:
        if g["code"] not in seen:
            seen.add(g["code"])
            out.append({"for": "unmeasured", "code": g["code"]})
    return out[:MAX_NEXT_ACTIONS]


_COST_PAREN = ("（代价接近 0：采样在后台线程里做）", "（代价接近 0）")


def short_action(action: str, limit: int = 46) -> str:
    """把 remedy 动作压成一句短动作（代价单独在 cost_s 里，不必重复在句子里）。"""
    text = action or ""
    for frag in _COST_PAREN:
        text = text.replace(frag, "")
    return _clip(text.strip("；， "), limit)


def _refusal_entry(code: str, severity: str, rem: Remedy) -> dict[str, Any]:
    """拒绝表里的一条：**动作**优先于人话解释（人话在渲染时按 code 查 CODE_TITLES）。"""
    e: dict[str, Any] = {"action": short_action(rem.action), "cost_kind": rem.cost_kind}
    if severity != "blocking":
        e["sev"] = severity
    if rem.cost_s:
        e["cost_s"] = round(rem.cost_s, 1)
    if rem.command:
        e["cmd"] = _clip(rem.command, 56)
    return {k: v for k, v in e.items() if v not in ("", None)}


def _cited_codes(packet: dict[str, Any]) -> set[str]:
    """包里**被引用到**的拒绝码：判词、未测项、下一步、环境拒绝。

    不变量（R-65）：这个集合里的每个 code 都必须能在 `refusals` 或
    `refusals_omitted.codes` 里查到 —— 否则读者看到"还有 N 个某 code"，却查不到处置。
    """
    cited: set[str] = set()
    for v in packet.get("verdicts") or []:
        code = ((v.get("refusal") or {}) or {}).get("code")
        if code:
            cited.add(code)
    for g in packet.get("unmeasured") or []:
        if g.get("code"):
            cited.add(g["code"])
    for n in packet.get("next") or []:
        if n.get("code"):
            cited.add(n["code"])
    for e in packet.get("environment_refusals") or []:
        if e.get("code"):
            cited.add(e["code"])
    for c in (packet.get("environment_refusals_more") or {}).get("codes") or []:
        cited.add(c)
    return cited


def _env_refusals(rep: Any) -> list[dict[str, Any]]:
    """报告里的降级诊断（缺 nsys / 缺 NVBit / 缺标定…）—— 它们也必须进拒绝表。"""
    return list((getattr(rep, "meta", {}) or {}).get("environment_refusals", []) or [])


def _refusal_index(verdicts: Sequence[Verdict], unmeasured: Sequence[dict[str, Any]],
                   capabilities: Any = None,
                   environment_refusals: Sequence[dict[str, Any]] | None = None
                   ) -> dict[str, dict[str, Any]]:
    """按 code 归一的拒绝表（`{code: {...}}`）：文本、remedy、代价只存一份。

    `environment_refusals` 也要进来（R-65）：顶层那个数组的 code 曾经**从不**进表，
    于是"缺 nsys"这类被引用到的 code 在包里查不到处置（结构性的悬空）。
    """
    index: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        if v.refusal is None:
            continue
        code = v.refusal.code.value
        if code not in index:
            index[code] = _refusal_entry(code, v.refusal.severity,
                                         v.refusal.remedy_or_default)
    for g in unmeasured:
        # 兜底：即使 capabilities 里没有对应条目，也必须有「怎么办」——
        # 拒绝表里缺一句 remedy，读者就只能干瞪眼（而这正是 W3 要消灭的状态）。
        code = g["code"]
        if not index.get(code):
            try:
                rc = RefusalCode(code)
            except ValueError:
                rc = RefusalCode.INTERNAL_UNKNOWN
            index[code] = _refusal_entry(code, "blocking", default_remedy(rc))
    if capabilities is not None:
        for a in capabilities.answers:
            if a.refusal is None:
                continue
            code = a.refusal.code.value
            if not index.get(code):
                index[code] = _refusal_entry(code, a.refusal.severity,
                                             a.refusal.remedy_or_default)
    for d in environment_refusals or []:
        code = d.get("code")
        if not code or index.get(code):
            continue
        try:
            rc = RefusalCode(code)
        except ValueError:
            rc = RefusalCode.INTERNAL_UNKNOWN
        index[code] = _refusal_entry(code, "blocking", default_remedy(rc))
    return index


def verdict_is_actionable(verdict: dict[str, Any]) -> bool:
    """这条判词能不能当行动依据（读包者的统一入口）。

    优先用显式字段 `actionable`（v1.1+）；v1.0 的包没有这个字段，就按同一套规则推：
    不是 refused、且**列出来的**前提全部成立。没列前提的判词（例如只有一条实测事实的
    结论）不因此变得不可依据 —— 但一旦列了前提，任何一条不成立就必须整条作废。
    """
    if "actionable" in verdict:
        return bool(verdict["actionable"])
    if verdict.get("status") == REFUSED:
        return False
    return all(p.get("ok") for p in (verdict.get("premises") or []))


def refusal_index(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """按 code 查拒绝详情（包里文本只存一份，查表是唯一的读法）。"""
    r = packet.get("refusals") or {}
    return {k: v for k, v in r.items() if isinstance(v, dict)}


# --------------------------------------------------------------------------
# 序列化（紧凑 + 自我计量）
# --------------------------------------------------------------------------
def dumps(packet: dict[str, Any], *, stable_size: bool = True, pretty: bool = False) -> str:
    """序列化，并把自身大小写回 `cost.packet_bytes`（稳定后返回）。

    默认**紧凑**（无缩进）：这个包是给机器读的，缩进只是把字节花在空白上；
    人要看的内容由 `render_onepage` 负责（同一份数据的另一种排版）。

    尺寸自己是**成本指标**的一部分（C 的直接度量）。先写占位再迭代到不动点，
    免得「报了 3.9 KB 实际 4.1 KB」这种自我欺骗。
    """
    def _dump() -> str:
        if pretty:
            return json.dumps(packet, indent=2, ensure_ascii=False)
        return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))

    if not stable_size:
        return _dump()
    packet.setdefault("cost", {})
    for _ in range(6):
        text = _dump()
        size = len(text.encode("utf-8"))
        if packet["cost"].get("packet_bytes") == size:
            return text
        packet["cost"]["packet_bytes"] = size
    return _dump()


def packet_bytes(packet: dict[str, Any]) -> int:
    return len(dumps(packet).encode("utf-8"))


def write_packet(packet: dict[str, Any], path: str) -> tuple[str, int]:
    """落盘并返回 `(路径, 字节数)`。

    D30②：返回的字节数必须**等于文件里真实的字节数**。旧实现写 `text + "\\n"` 却返回
    `len(text)`，于是包比 `cost.packet_bytes` 说的多 1 B —— 而 `packet_bytes` 是"这个包
    花了多少上下文"的那个数，正是读者拿来跟 4 KB 预算比的。多一个换行的收益（POSIX 文本
    惯例）抵不过"数字与文件对不上"的代价，所以不再追加换行。
    """
    data = dumps(packet).encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return path, len(data)


def load_packet(path: str) -> dict[str, Any]:
    """读决策包；也接受**完整报告**（`*.ksreport.json`）—— 现场回读时不该要求转换。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.loads(f.read())
    if d.get("tool") == "kernoscope" and "kernels" in d and "verdicts" not in d:
        from .report import Report

        return build(Report.from_dict(d), report_path=path)
    return d


def _clip(text: Any, limit: int) -> Any:
    if not isinstance(text, str):
        return text
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# W4：人视图（一屏）—— 只由 packet 渲染，散文在这里生成
# --------------------------------------------------------------------------
_STATUS_MARK = {OK: "ok  ", SOFT: "soft", REFUSED: "no  "}


def _bytes_phrase(v: dict[str, Any]) -> str:
    """判词里的字节量级：从 `value`（字节）走统一的单位阶梯。

    不要拿 `facts.mb` 再拼一个 " MB" —— 旧实现把 MB 当成字节，于是同一句话印出
    "约 7.3 MB"，而报告里那一行是 7.273 TB（差 10⁶）。散文只由这里生成，所以这里也必须
    只认 `value` 一个来源。
    """
    if v.get("value") is not None:
        return fmt_bytes(float(v["value"]))
    return "?"


def phrase_verdict(v: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    """判词的一句话（散文只在这里生成，包与报告里都不存它）。"""
    fid = v.get("id", "")
    f = v.get("facts") or {}
    if v.get("status") == REFUSED:
        # 只给「哪一类拒绝」：怎么办由下面的 remedy 行负责（同一句话不印两遍）
        code = (v.get("refusal") or {}).get("code", "?")
        return f"不给数字：{code_title(code)}"
    if fid == "kernel_time":
        return (f"最耗时的是 {f.get('top', '?')}（{f.get('top_ms')} ms / {f.get('launches')} 次，"
                f"占 kernel 时间 {f.get('share_pct')}%），进程内 kernel 时间合计 "
                f"{f.get('total_ms')} ms")
    if fid == "dram_bytes":
        return f"整窗口 DRAM 约 {_bytes_phrase(v)}（代理口径：NVML 显存忙碌 × 本机实测峰值）"
    if fid == "dram_bytes_per_launch":
        return (f"每次 launch 约 {_bytes_phrase(v)}（代理口径）"
                + (f"；归因上偏估计 +{f['bias_pct']}%" if f.get("bias_pct") else ""))
    if fid == "dram_bytes_trace":
        return f"轨迹口径每次 launch 约 {_bytes_phrase(v)}（L2 命中模型）"
    if fid == "ledger_verdict":
        if "err_pct" in f:
            return f"与账本相差 {float(f['err_pct']):+.1f}%（同口径对拍）"
        return str(f.get("position", "账本区间判词"))
    if fid == "occupancy":
        bits = []
        if "theoretical_pct" in f:
            bits.append(f"理论上界 {f['theoretical_pct']}%")
        if "achieved_pct" in f:
            bits.append(f"实测达到 {f['achieved_pct']}%")
        if f.get("limiter"):
            bits.append(f"限制因素 {f['limiter']}")
        return "；".join(bits)
    if fid == "roofline":
        bits = []
        if f.get("bound"):
            bits.append(f"判定为 {f['bound']} bound")
        if "efficiency_pct" in f:
            bits.append(f"离 roofline {100.0 - float(f['efficiency_pct']):.1f}%")
        if "gbs" in f:
            bits.append(f"达成带宽 {f['gbs']} GB/s")
        return "；".join(bits)
    if fid == "issue_slot":
        return ("发射槽利用率上界 "
                + (f"{f['util_pct']}%" if "util_pct" in f else "n/a")
                + (f"；瓶颈方向 {f['direction']}" if f.get("direction") else ""))
    if fid == "l2_locality":
        bits = []
        if "hit_pct" in f:
            bits.append(f"L2 命中率模型 {f['hit_pct']}%")
        if "working_set_mib" in f:
            bits.append(f"工作集 {f['working_set_mib']} MiB")
        return "；".join(bits)
    if fid == "static_mix":
        mix = f.get("mix") or {}
        return (f"SASS 共 {f.get('total')} 条"
                + ("；" + "，".join(f"{k} {v}%" for k, v in mix.items()) if mix else ""))
    if fid == "over_peak":
        return "达成值超过本机实测峰值：本次百分比不成立，先查标定与工作量输入"
    return v.get("id", "")


def render_onepage(packet: dict[str, Any], width: int = 96) -> str:
    """结论优先的一屏：判词 + 前提值 + 下一步 + 什么没测到 + 代价。

    纪律：本函数**只读 packet**。任何需要读 Report 才能显示的东西都不许出现在这里 ——
    那是「两个视图分叉」的入口。
    """
    out: list[str] = []
    index = refusal_index(packet)
    env = packet.get("env") or {}
    head = (f"Kernoscope {packet.get('tool_version')} · "
            f"协议 {str((packet.get('protocol') or {}).get('hash', ''))[7:19]}")
    if env:
        head += f" · 层 {env.get('tier')} ({env.get('tier_name')})"
    if packet.get("synthetic"):
        head += " · **合成数据（不是测量）**"
    out.append(head)
    out.append("-" * min(width, 96))

    verdicts = packet.get("verdicts") or []
    if not verdicts:
        out.append("没有任何可下结论的判词：见下面「没测到什么」。")
    for v in verdicts:
        mark = _STATUS_MARK.get(v.get("status"), "?   ")
        line = f"[{mark}] {phrase_verdict(v, index)}"
        if v.get("value") is not None and v.get("unit"):
            line += f"  ({v['value']} {v['unit']}"
            if v.get("provenance"):
                line += f", {v['provenance']}"
            if v.get("band"):
                line += f", 误差带 {v['band'].get('low')}…{v['band'].get('high')}%"
            if v.get("mdd") is not None:
                line += f", MDD {v['mdd']}%"
            line += ")"
        out.append(line)
        for p in v.get("premises") or []:
            label = premise_label(str(p.get("id", "")))
            thr = p.get("threshold")
            unit = p.get("unit", "")
            if p.get("observed") is None:
                out.append(f"        前提 {label}：**无法判定**"
                           + (f"（门槛 {thr}{unit}）" if thr is not None else "")
                           + " → " + ("满足" if p.get("ok") else "**不满足**"))
            elif thr is None:
                out.append(f"        前提 {label}：{p['observed']}{unit}")
            else:
                flag = "满足" if p.get("ok") else "**不满足**"
                out.append(f"        前提 {label}：{p['observed']}{unit}"
                           f"（门槛 {thr}{unit}）→ {flag}")
        if v.get("status") == REFUSED:
            code = (v.get("refusal") or {}).get("code", "?")
            r = index.get(code) or {}
            cost = _cost_text(r)
            if r.get("action"):
                out.append(f"        怎么办：{r['action']}"
                           + (f"（代价 {cost}）" if cost and r.get("cost_kind")
                              not in (None, "unavailable") else "")
                           + (f"  [{r['cmd']}]" if r.get("cmd") else ""))
        elif v.get("next"):
            out.append(f"        下一步：{v['next']}")

    env_ref = packet.get("environment_refusals") or []
    um = packet.get("unmeasured") or []
    if env_ref:
        out.append("")
        out.append("本机缺件（不是这次没测，而是这台机器没有）：")
        for d in env_ref:
            code = d.get("code", "?")
            r = index.get(code) or {}
            out.append(f"  - {code}（{code_title(code)}）"
                       + (f"；怎么办：{r['action']}" if r.get("action") else "")
                       + (f"  [{r['cmd']}]" if r.get("cmd") else ""))
    if um or env.get("blocked_by"):
        more = packet.get("unmeasured_more") or {}
        total = sum(g["count"] for g in um) + int(more.get("metrics") or 0)
        out.append("")
        out.append(f"没测到什么（共 {total} 个指标，按原因归组）：")
        for g in um:
            r = index.get(g["code"]) or {}
            cost = _cost_text(r)
            out.append(f"  - {g['count']} 个 {g['code']}（{code_title(g['code'])}）"
                       + (f"（例：{', '.join(g['examples'])}）" if g.get("examples") else "")
                       + (f"；怎么办：{r['action']}" if r.get("action") else "")
                       + (f"（代价 {cost}）" if cost and r.get("cost_kind")
                          not in (None, "unavailable") else ""))
        for code in (env.get("blocked_by") or []):
            if code not in {g["code"] for g in um}:
                r = index.get(code) or {}
                out.append(f"  - 本机缺件 {code}（{code_title(code)}）"
                           + (f"；怎么办：{r['action']}" if r.get("action") else ""))
        if more:
            out.append(f"  … 其余 {more.get('groups')} 组 / {more.get('metrics')} 个指标见 "
                       f"--packet（机器视图）")

    cost = packet.get("cost") or {}
    bits = [f"wall {cost.get('wall_s')}s", f"包 {cost.get('packet_bytes')}B"]
    if cost.get("report_bytes"):
        bits.append(f"完整报告 {cost['report_bytes'] / 1024:.0f} KB")
    if cost.get("launches"):
        bits.append(f"{cost['launches']} 次 launch")
    if not cost.get("selftest_band"):
        bits.append("无本机误差带（跑 kernoscope selftest 后 diff 才有 MDD）")
    if cost.get("trimmed"):
        # R-70/D10：只说"裁了哪一步"不够 —— 读者要知道**少了什么**。所以把三个
        # "*_more" 的计数与 id 一起印出来（它们本来就是为这个存在的机器契约字段）。
        bits.append("包已按尺寸预算裁剪（" + ", ".join(cost["trimmed"]) + "）")
    more = packet.get("verdicts_more") or {}
    if more:
        bits.append(f"判词超上限少 {more.get('count')} 条（{', '.join(more.get('ids') or [])}）")
    omitted = packet.get("refusals_omitted") or {}
    if omitted:
        codes = ", ".join(omitted.get("codes") or [])
        bits.append(f"拒绝条目省了 {omitted.get('count')} 条"
                    + (f"（{codes}{'…' if omitted.get('codes_truncated') else ''}）"
                       if codes else ""))
    un_more = packet.get("unmeasured_more") or {}
    if un_more:
        bits.append(f"未测项合并了 {un_more.get('groups')} 组"
                    f"（共 {un_more.get('metrics')} 个指标）")
    out.append("")
    out.append("本次代价：" + " · ".join(str(b) for b in bits))
    if cost.get("report_path") or False:
        pass
    nxt = packet.get("next") or []
    if nxt:
        out.append("建议的下一步：")
        for n in nxt[:MAX_NEXT_ACTIONS]:
            code = n.get("code")
            if code:
                r = index.get(code) or {}
                out.append(f"  - {n.get('for', '')}：{r.get('action') or code_title(code)}"
                           + (f"  [{r['cmd']}]" if r.get("cmd") else ""))
            else:
                out.append(f"  - {n.get('for', '')}：{n.get('hint', '')}")
    return "\n".join(out)


def _cost_text(rem: dict[str, Any]) -> str:
    kind = rem.get("cost_kind")
    if kind == "unavailable":
        return "本机不可行"
    if kind == "manual":
        return "需人工"
    s = float(rem.get("cost_s") or 0.0)
    if s <= 0:
        return "≈0"
    return f"≈{s:.0f}s" if s < 90 else f"≈{s / 60:.0f}min"


__all__ = ["MAX_VERDICTS", "OK", "PACKET_SCHEMA_REVISION", "PACKET_SCHEMA_VERSION",
           "VERDICT_IDS",
           "PACKET_SIZE_BUDGET_BYTES", "verdict_is_actionable",
           "PREMISE_LABELS", "REFUSED", "SOFT", "Premise", "Verdict", "build", "dumps",
           "find_metric", "group_unmeasured", "load_packet", "metric_value", "packet_bytes",
           "phrase_verdict", "premise_label", "protocol_hash", "refusal_index",
           "render_onepage", "write_packet"]
