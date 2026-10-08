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
from .provenance import Metric, Provenance
from .refusals import Refusal, RefusalCode, Remedy, code_title, default_remedy

PACKET_SCHEMA_VERSION = 1
PACKET_SIZE_BUDGET_BYTES = 4096          # 默认包的尺寸预算（测试守着，防止再长成 150 MB）
UNMEASURED_EXAMPLES_PER_CODE = 2
MAX_UNMEASURED_GROUPS = 6                 # 归组之后仍然只默认给前 6 组
MAX_NEXT_ACTIONS = 4
MAX_VERDICTS = 10
FACT_STR_LIMIT = 48                       # facts 里的字符串（kernel 名等）限长

OK, SOFT, REFUSED = "ok", "soft", "refused"
STATUSES = (OK, SOFT, REFUSED)

# 前提的可读标签是**静态**的（同一 id 每次一样），所以不进包：渲染时查这张表。
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

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad verdict status {self.status!r}")
        if self.status == REFUSED and self.refusal is None:
            raise ValueError(f"refused verdict {self.id!r} must carry a structured refusal")
        if self.status == REFUSED and self.next:
            object.__setattr__(self, "next", "")   # 拒绝时「下一步」就是 remedy，不重复

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "status": self.status}
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
    val = float(m.value)  # type: ignore[arg-type]
    return Verdict(
        id="dram_bytes", question="DRAM 搬了多少字节 / 是不是 DRAM bound", status=SOFT,
        facts={"mb": round(val / 1e6, 1)}, value=round(val, 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_total", "ks__sampler_window_coverage_pct"),
        next="与你的账本对拍（--ks-ledger-total-mb）确认这个量级是否符合预期",
        claim=f"整窗口 DRAM 约 {val / 1e6:.1f} MB（代理口径：NVML 显存忙碌时间 × "
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
    bias = metric_value(rep, "ks__attribution_bias_estimate_pct")
    facts: dict[str, Any] = {"mb": round(float(m.value) / 1e6, 2)}
    if bias:
        facts["bias_pct"] = round(float(bias))
    return Verdict(
        id="dram_bytes_per_launch", question="每次 launch 搬多少字节", status=SOFT,
        facts=facts, value=round(float(m.value), 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_per_launch", "ks__attribution_bias_estimate_pct"),
        next="改一次 kernel 再跑 diff：差值小于 MDD 就是噪声，不要当成改进",
        claim=f"每次 launch 约 {float(m.value) / 1e6:.2f} MB（代理口径）"
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
    return Verdict(
        id="dram_bytes_trace", question="DRAM 字节的第二条口径（访存轨迹）", status=SOFT,
        facts={"mb": round(float(m.value) / 1e6, 2)}, value=round(float(m.value), 1), unit="B",
        provenance=m.provenance.value, premises=premises,
        metrics=("est__dram_bytes_per_launch_trace",),
        next="两条口径差得远时先查采样代表性与 L2 模型参数，不要挑好看的那个",
        claim=f"轨迹口径每次 launch 约 {float(m.value) / 1e6:.2f} MB（L2 命中模型）")


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
    return Verdict(
        id="roofline", question="离 roofline 多远、compute 还是 memory bound", status=SOFT,
        facts=facts, value=eff, unit="%", provenance="estimated",
        metrics=("est__bound", "est__roofline_efficiency_pct", "est__achieved_gbs"),
        next=("memory bound 且达成率低 → 先查访问模式（L2 命中/合并效率）；"
              "compute bound → 查指令混合与发射槽"),
        claim="；".join(str(b) for b in bits))


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
def build(rep: Any, *, capabilities: Any = None, selfcheck: dict[str, Any] | None = None,
          basis: str = "", cost_extra: dict[str, Any] | None = None,
          report_path: str = "") -> dict[str, Any]:
    """由 Report 组出决策包（不读文件、不跑命令，纯函数便于测试）。"""
    verdicts = [v for fn in BUILDERS if (v := fn(rep)) is not None][:MAX_VERDICTS]

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
        "tool": "kernoscope",
        "tool_version": getattr(rep, "tool_version", __version__),
        "protocol": {"hash": protocol_hash(rep), "basis": basis,
                     "window_mode": _window_mode(rep)},
        "env": env,
        "verdicts": [v.to_dict() for v in verdicts],
        "unmeasured": unmeasured,
        "refusals": _refusal_index(verdicts, unmeasured_all, capabilities),
        "next": _next_actions(verdicts, unmeasured),
        "cost": cost,
    }
    if unmeasured_more:
        packet["unmeasured_more"] = unmeasured_more
    if getattr(rep, "meta", {}).get("synthetic"):
        packet["synthetic"] = True
    for d in (getattr(rep, "meta", {}) or {}).get("environment_refusals", []) or []:
        packet.setdefault("environment_refusals", []).append(
            {"code": d.get("code"), "message": _clip(str(d.get("message", "")), 120)})
    if report_path:
        packet["report_path"] = report_path
    enforce_budget(packet)
    return packet


def enforce_budget(packet: dict[str, Any],
                   limit: int = PACKET_SIZE_BUDGET_BYTES) -> list[str]:
    """把包压进尺寸预算，并如实记录**裁掉了什么**（返回被裁剪的步骤名）。

    尺寸是 C 的一部分，所以它是一条**硬**约束：不管报告里有多少判词与拒绝，
    默认包都不许超过预算。做法是有序降级（先去掉最能"再生"的东西）：

        例子 > 归组数 > 下一步条数 > 判词指标名 > remedy 命令 > 非失败前提 > 判词条数

    每一步都只删"能从完整报告里再取回来"的东西，并把步骤名写进 `cost.trimmed` ——
    读者（人或 agent）因此知道包是摘要，而不是以为这就是全部。
    """
    trimmed: list[str] = []

    def _size() -> int:
        # 必须按**最终**序列化量：`cost.packet_bytes` 自己也是包的一部分，
        # 若不计入它，就会出现「包 4121 B 却认为自己没超 4096」这种自我欺骗。
        return len(dumps(packet).encode("utf-8"))

    if _size() <= limit:
        packet.setdefault("cost", {})["trimmed"] = []
        return trimmed

    def _step(name: str, fn) -> bool:
        fn()
        trimmed.append(name)
        return _size() <= limit

    def _drop_passing_premises() -> None:
        """只保留**不成立**的前提（成立的前提值能从报告里再看）。"""
        for v in packet["verdicts"]:
            ps = v.get("premises") or []
            if len(ps) > 1:
                v["premises"] = [p for p in ps if not p.get("ok")] or ps[:1]

    steps = [
        ("unmeasured_examples", lambda: [g.pop("examples", None) for g in packet["unmeasured"]]),
        ("next_pointers", lambda: packet.pop("next", None)),
        ("verdict_metrics", lambda: [v.pop("metrics", None) for v in packet["verdicts"]]),
        ("unmeasured_groups", lambda: _cap_list(packet, "unmeasured", 3)),
        ("passing_premises", _drop_passing_premises),
        ("remedy_commands",
         lambda: [r.pop("cmd", None) for r in (packet.get("refusals") or {}).values()]),
        ("verdicts", lambda: _cap_list(packet, "verdicts", 6)),
        ("facts", lambda: [v.pop("facts", None) for v in packet["verdicts"]]),
    ]
    for name, fn in steps:
        if _step(name, fn):
            break
    packet.setdefault("cost", {})["trimmed"] = trimmed
    if _size() > limit:  # pragma: no cover - 走到这里说明上面的步骤不够，得再加一步
        packet["cost"]["over_budget"] = _size()
    return trimmed


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


def _refusal_index(verdicts: Sequence[Verdict], unmeasured: Sequence[dict[str, Any]],
                   capabilities: Any = None) -> dict[str, dict[str, Any]]:
    """按 code 归一的拒绝表（`{code: {...}}`）：文本、remedy、代价只存一份。"""
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
    return index


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
    text = dumps(packet)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path, len(text.encode("utf-8"))


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
        return f"整窗口 DRAM 约 {f.get('mb')} MB（代理口径：NVML 显存忙碌 × 本机实测峰值）"
    if fid == "dram_bytes_per_launch":
        return (f"每次 launch 约 {f.get('mb')} MB（代理口径）"
                + (f"；归因上偏估计 +{f['bias_pct']}%" if f.get("bias_pct") else ""))
    if fid == "dram_bytes_trace":
        return f"轨迹口径每次 launch 约 {f.get('mb')} MB（L2 命中模型）"
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
        bits.append("包已按尺寸预算裁剪（" + ", ".join(cost["trimmed"]) + "）")
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


__all__ = ["MAX_VERDICTS", "OK", "PACKET_SCHEMA_VERSION", "PACKET_SIZE_BUDGET_BYTES",
           "PREMISE_LABELS", "REFUSED", "SOFT", "Premise", "Verdict", "build", "dumps",
           "find_metric", "group_unmeasured", "load_packet", "metric_value", "packet_bytes",
           "phrase_verdict", "premise_label", "protocol_hash", "refusal_index",
           "render_onepage", "write_packet"]
