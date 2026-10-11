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
    # 环境前提（第六轮 W-6：这三条此前不在两张表里 ⇒ 降级文案**直接印 id**）
    "env_gpu_exclusive": RefusalCode.ENV_SHARED_GPU,
    "env_gpu_busy_pct": RefusalCode.COVERAGE_TOO_LOW,   # 同义：窗口没被负载主导，只是观测量不同
    "env_no_throttle_pct": RefusalCode.ENV_THROTTLED,
}

PREMISE_LABELS: dict[str, str] = {
    "kernel_share_of_total": "该 kernel 占全部 kernel 时间的比例",
    "kernel_time_pct_of_wall": "kernel 时间占进程 wall 的比例",
    "window_coverage_pct": "窗口是否被被测负载主导",
    "window_mem_busy_pct": "参与积分的窗口内显存忙碌均值",
    "matched_kernel_kernel_share_pct": "被匹配 kernel 是否占窗口 kernel 时间 ≥95%",
    "trace_vs_nvml_pct": "轨迹口径 / NVML 口径",
    "over_peak": "达成值是否超过本机实测峰值",
    # 环境前提：把"这钱是在什么环境下花的"变成前提，环境不干净时结论自动降级
    "env_gpu_exclusive": "同卡有没有别的计算进程（多租户会把读数稀释）",
    "env_gpu_busy_pct": "采样窗口里 GPU 有多少时间在干活（宿主间隙会稀释比值类口径）",
    "env_no_throttle_pct": "窗口内降频时间占比（功率封顶/热降频会让能效读数偏乐观或偏保守）",
    # 成本账（C1）：结论的两个分母，必须随结论一起出现
    "window_energy_j": "窗口内实测能耗（NVML 功率积分/能量累加器）",
    "tokens_per_s": "窗口内吞吐（调用方给的 token 数 ÷ 窗口秒数）",
}

# 判词 → 它**会**挂上的前提（单一真源）。三方共用：
#   * `tools/gen_agent_refs.py` 用它生成 `references/verdicts.md` 的「需要的前提」列；
#   * 测试用它断言"代码实际发出的前提 ⊆ 文档列出的前提"（第六轮 W-6：生成文档此前读的是
#     生成器自己的手写表，所以 `cost_per_token` 在文档里写着「无强制前提」，而代码会给它挂
#     三条环境前提，`--gates` 却 9/9）；
#   * 新增前提时改这里一处就够。
VERDICT_PREMISES: dict[str, tuple[str, ...]] = {
    "kernel_time": ("kernel_share_of_total", "kernel_time_pct_of_wall"),
    # 成本判词：两个分母 + 三条环境前提（环境不干净时它必须降级，这是它存在的理由）
    "cost_per_token": ("window_energy_j", "tokens_per_s", "env_gpu_exclusive",
                       "env_gpu_busy_pct", "env_no_throttle_pct"),
    "truth_vs_substitute": (),
    "saving_estimate": (),
    "dram_bytes": ("window_coverage_pct", "window_mem_busy_pct"),
    "dram_bytes_per_launch": ("window_coverage_pct", "matched_kernel_kernel_share_pct"),
    "dram_bytes_trace": ("trace_vs_nvml_pct",),
    "ledger_verdict": (),          # 继承 `dram_bytes_per_launch` 的前提（见生成器的说明列）
    "occupancy": (),
    "roofline": (),
    "issue_slot": (),
    "l2_locality": (),
    "over_peak": ("over_peak",),
    "static_mix": (),
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
    # **不进包**的判词身份：这条判词属于哪个 kernel（未截断的名字）。
    # 为什么不能复用 `facts["kernel"]`：那是 `_clip(name, 48)` 的**显示串**，而 C++ 模板实例化
    # 名字前 48 字符撞车是常态（真机 A/B：两个不同 kernel 前 48 字相同）—— 于是一个便宜 kernel
    # 的超峰会否掉另一个 kernel 合法的 91% roofline（第六轮 W-4）。截断只用于显示，不用于比较。
    owner_key: str = field(default="", repr=False)

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


def find_metric_in(kernel: Any, name: str) -> Metric | None:
    """只在**指定 kernel** 里找指标（V-6：跨 kernel 传染的根治手段）。"""
    for s in getattr(kernel, "sections", []):
        for m in s.metrics:
            if m.name == name:
                return m
    return None


def kernel_with(rep: Any, name: str, *, prefer: Any = None) -> Any | None:
    """哪个 kernel 有这个指标（`prefer` 命中就优先）；没有返回 None。

    判词必须绑在**具体 kernel** 上：同一份报告里 `ks__over_peak_warning` 可能只属于
    某个便宜 kernel，而 roofline 属于真正耗时的那个 —— 全局"找到第一个"会把前者的问题
    传染给后者（第五轮 V-6）。
    """
    if prefer is not None and find_metric_in(prefer, name) is not None:
        return prefer
    for k in getattr(rep, "kernels", []):
        if getattr(k, "name", "") == "(device)":
            continue
        if find_metric_in(k, name) is not None:
            return k
    return None


def metric_value_in(kernel: Any, name: str) -> float | int | None:
    m = find_metric_in(kernel, name)
    return None if m is None or m.value is None else m.value


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
def _top_kernel(rep: Any) -> Any | None:
    """耗时最大的真 kernel（判词的默认主体）。"""
    ks = [k for k in getattr(rep, "kernels", [])
          if getattr(k, "name", "") != "(device)" and getattr(k, "total_ns", 0)]
    return max(ks, key=lambda k: k.total_ns) if ks else None


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
    # V-6：roofline 必须绑在**它自己那个 kernel** 上。全局"第一个 est__roofline_efficiency_pct"
    # 会把别的 kernel 的超峰传染进来（真机可达：一个便宜 kernel 超峰 → 另一个 kernel 合法的
    # 91% roofline 被判 refused）。
    top = _top_kernel(rep)
    owner = (kernel_with(rep, "est__roofline_efficiency_pct", prefer=top)
             or kernel_with(rep, "est__bound", prefer=top)
             or top)
    bound = metric_value_in(owner, "est__bound") if owner is not None else None
    eff = metric_value_in(owner, "est__roofline_efficiency_pct") if owner is not None else None
    gbs = metric_value_in(owner, "est__achieved_gbs") if owner is not None else None
    if bound is None and eff is None:
        return None
    facts: dict[str, Any] = {}
    if owner is not None and getattr(owner, "name", ""):
        facts["kernel"] = _clip(str(owner.name), FACT_STR_LIMIT)
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
        owner_key=str(getattr(owner, "name", "") or "") if owner is not None else "",
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


# 环境前提的**门槛**（R-100③：此前是裸字面量 `0.0/80.0/5.0`，README 零提及）。
# 改这三个数会改变"什么算环境干净"，所以它们是接口的一部分，不是魔法数字。
ENV_OTHER_PROCS_MAX = 0.0     # 同卡其他计算进程数上限（0 = 必须独占）
ENV_GPU_BUSY_MIN_PCT = 80.0   # 窗口里 GPU 干活时间占比下限
ENV_NO_THROTTLE_MAX_PCT = 5.0  # 降频时间占比上限


def _metric_unreadable(rep: Any, name: str) -> bool:
    """这个指标**存在但读不到**（unavailable），与"这个指标根本没有"区分开（R-96④）。"""
    m = find_metric(rep, name)
    return m is not None and m.provenance is Provenance.UNAVAILABLE


def _environment_premises(rep: Any) -> list[Premise]:
    """把"这钱是在什么环境下花的"变成前提（R-61 延伸）。

    不重放 ⇒ 数字是真实时间，但**真实时间只在环境可比时才能互相比较**。三条：
      * 同卡有没有别的计算进程（多租户：数字被稀释）
      * GPU 空转占比（宿主间隙：比值类口径被稀释）
      * 降频占比（功率封顶/热降频：能效数字偏乐观或偏保守）
    任一不成立 → 结论自动降级（`premises.ok=False` 时 `Verdict` 会降级），
    而不是假装环境干净。
    """
    out: list[Premise] = []
    others = metric_value(rep, "meas__other_procs_on_gpu")
    if others is not None:
        out.append(Premise("env_gpu_exclusive", float(others), ENV_OTHER_PROCS_MAX,
                           float(others) <= ENV_OTHER_PROCS_MAX, "个"))
    elif _metric_unreadable(rep, "meas__other_procs_on_gpu"):
        # 探测失败 ⇒ 前提"不成立"（不是缺席）。缺席会让 cost_per_token 回到 actionable=True，
        # 那正是 R-96 要修的那种假绿。
        out.append(Premise("env_gpu_exclusive", None, ENV_OTHER_PROCS_MAX, False, "个"))
    idle = metric_value(rep, "meas__gpu_idle_pct")
    if idle is not None:
        out.append(Premise("env_gpu_busy_pct", round(100.0 - float(idle), 1),
                           ENV_GPU_BUSY_MIN_PCT,
                           (100.0 - float(idle)) >= ENV_GPU_BUSY_MIN_PCT, "%"))
    thr = metric_value(rep, "meas__throttle_limited_pct")
    if thr is not None:
        out.append(Premise("env_no_throttle_pct", round(float(thr), 1),
                           ENV_NO_THROTTLE_MAX_PCT,
                           float(thr) < ENV_NO_THROTTLE_MAX_PCT, "%"))
    elif _metric_unreadable(rep, "meas__throttle_limited_pct"):
        out.append(Premise("env_no_throttle_pct", None, ENV_NO_THROTTLE_MAX_PCT, False, "%"))
    return out


def _verdict_cost(rep: Any) -> Verdict | None:
    """成本判词（C1）—— 这个工具存在的理由：把秒换算成钱。

    三种状态都要能出现，且**拒绝本身是输出**：
      * 有 token 数 + 价格 → 给 ¥/Mtoken（外加 token/s 与 J/token）；
      * 只有一半输入     → 给算得出来的那一半（例如只要功率与电价就有 ¥/h），
                            另一半用带 code 的拒绝说清楚该加哪个开关；
      * 都没有           → refused + remedy（`no_tokens` / `no_price` / `no_energy`），
                            agent 据此知道下一步该传什么，而不是猜。
    """
    per_mtok = find_metric(rep, "ks__cost_per_mtok")
    per_hour = find_metric(rep, "ks__cost_per_hour")
    if per_mtok is None and per_hour is None:
        return None
    tokens_s = metric_value(rep, "ks__tokens_per_s")
    j_per_token = metric_value(rep, "ks__j_per_token")
    energy = find_metric(rep, "meas__window_energy_j")
    kernel_cost = metric_value(rep, "est__kernel_cost_per_hour")
    top = next((k.name for k in getattr(rep, "kernels", [])
                if any(m.name == "est__kernel_cost_per_hour"
                       for s in getattr(k, "sections", []) for m in s.metrics)), "")
    premises: list[Premise] = []
    if energy is not None and energy.value is not None:
        premises.append(Premise("window_energy_j", round(float(energy.value), 1), unit="J"))
    if tokens_s is not None:
        premises.append(Premise("tokens_per_s", round(float(tokens_s), 1), unit="token/s"))
    premises.extend(_environment_premises(rep))

    # ---- ① 单位成本可用：这是主结论 ----
    if per_mtok is not None and per_mtok.value is not None:
        facts: dict[str, Any] = {"per_mtok": round(float(per_mtok.value), 4),
                                 "per_hour": (round(float(per_hour.value), 4)
                                              if per_hour is not None and per_hour.value is not None
                                              else None),
                                 "tokens_s": round(float(tokens_s), 1) if tokens_s else None,
                                 "j_per_token": (float(f"{float(j_per_token):.4g}")
                                                 if j_per_token is not None else None)}
        if kernel_cost is not None and top:
            facts["top_kernel"] = _clip(top, FACT_STR_LIMIT)
            facts["top_kernel_per_hour"] = round(float(kernel_cost), 4)
        return Verdict(
            id="cost_per_token", question="这次推理/训练每百万 token 花多少钱", status=OK,
            facts=facts, value=round(float(per_mtok.value), 4), unit=per_mtok.unit,
            provenance=_prov(rep, "ks__cost_per_mtok") or "derived", premises=premises,
            metrics=("ks__cost_per_mtok", "ks__tokens_per_s", "meas__window_energy_j"),
            next="改完重跑同一条命令，用 kernoscope diff 比 ks__cost_per_mtok（越小越好）；"
                 "先说清改的是哪个 kernel（est__kernel_cost_per_hour 大的那个）",
            claim=(f"单位成本 {float(per_mtok.value):.4g} {per_mtok.unit}"
                   + (f"（{float(tokens_s):,.0f} token/s、{float(j_per_token):.3g} J/token）"
                      if tokens_s and j_per_token is not None else "")))

    # ---- ② 只有每小时成本：仍是有用的部分结论 ----
    if per_hour is not None and per_hour.value is not None:
        facts = {"per_hour": round(float(per_hour.value), 4),
                 "tokens_s": round(float(tokens_s), 1) if tokens_s else None}
        if kernel_cost is not None and top:
            facts["top_kernel"] = _clip(top, FACT_STR_LIMIT)
            facts["top_kernel_per_hour"] = round(float(kernel_cost), 4)
        return Verdict(
            id="cost_per_token", question="这次推理/训练每百万 token 花多少钱", status=SOFT,
            facts=facts, value=round(float(per_hour.value), 4), unit=per_hour.unit,
            provenance=_prov(rep, "ks__cost_per_hour") or "derived", premises=premises,
            metrics=("ks__cost_per_hour", "meas__window_energy_j"),
            next="想把它换算成每百万 token：补 --ks-tokens <每步 token 数>（有价格就够了）",
            claim=f"每小时成本 {float(per_hour.value):.4g} {per_hour.unit}"
                  "（还差每步 token 数才能给单位成本）")

    # ---- ③ 给不出钱：拒绝 + 该加哪个开关 ----
    ref_metric = next((m for m in (per_mtok, per_hour, energy)
                       if m is not None and getattr(m, "refusal", None) is not None), None)
    if ref_metric is None:
        return None
    return _refused("cost_per_token", "这次推理/训练每百万 token 花多少钱", ref_metric,
                    premises, ("ks__cost_per_mtok", "ks__cost_per_hour",
                               "meas__window_energy_j"))


def _verdict_truth(rep: Any) -> Verdict | None:
    """真值对拍判词（I1）—— 只有在导入了 `.ncu-rep` 时才有。

    这条判词回答的是"我们的替代口径到底准不准"：它把导入的真值（`ncu__*`，与 ncu 同源）
    与我们自己的值逐条相比。**一条对拍都没有时也出判词**（refused + 原因），
    因为"对不上"本身是结论 —— 沉默会让读的人以为对拍过了。
    """
    t = (getattr(rep, "meta", {}) or {}).get("ncu_truth")
    if not t:
        return None
    compared = int(t.get("pairs_compared") or 0)
    total = int(t.get("pairs_total") or 0)
    facts = {"ncu": _clip(str(t.get("ncu_version") or "?"), FACT_STR_LIMIT),
             "compared": compared, "pairs": total,
             "actions": int(t.get("actions") or 0)}
    if t.get("max_abs_err_pct") is not None:
        facts["max_abs_err_pct"] = round(float(t["max_abs_err_pct"]), 3)
    if compared == 0:
        return Verdict(
            id="truth_vs_substitute", question="我们的替代口径 vs 真值（导入的 .ncu-rep）",
            status=REFUSED, facts=facts,
            refusal=Refusal(RefusalCode.METRIC_ABSENT,
                            "一条对拍都没成功：要么这份报告里没有我们用来对拍的指标"
                            "（不同架构/采集配置的指标集不同），要么我们这次没算出对应的替代值；"
                            "用 --json 看 ncu_truth.comparisons[].pairs[].reason"),
            metrics=("ncu__gpu__time_duration.sum",),
            next="换一份含 gpu__time_duration / launch__registers_per_thread 的报告，"
                 "或补齐我们的替代值前提（标定/账本）再对拍",
            claim=f"导入了 {t.get('actions')} 个 action 的真值，但没有一条能与我们逐条对拍")
    return Verdict(
        id="truth_vs_substitute", question="我们的替代口径 vs 真值（导入的 .ncu-rep）",
        status=OK, facts=facts, value=facts.get("max_abs_err_pct"), unit="%",
        provenance="exact", premises=[],
        metrics=("ncu__gpu__time_duration.sum", "ks__truth_vs_substitute_pct"),
        next="偏差大的那条口径要写进结论的限制里（别把代理值当计数器读数引用）",
        claim=f"与 ncu {t.get('ncu_version')} 的真值对拍 {compared}/{total} 条，"
              f"最大相对偏差 {facts.get('max_abs_err_pct')}%")


def _discovery_ranking(rep: Any, *, limit: int = 3) -> dict[str, Any] | None:
    """**发现层的第一级**：不依赖标定、不依赖价格，按**实测时间占比**排出的头寸清单（M13）。

    为什么必须有它：唯一会"排头寸"的判词是 `saving_estimate`，而它的第二级（保守可省区间、
    折算成钱）需要先跑一次本机标定 + 工作量输入 + 价格 —— 第一次上手的人/agent 通常都没有，
    于是"钱在哪、先改什么"**在最需要它的时候缺席**（实测：合成包里整条判词不出现）。
    而排序本身**不需要**任何前提：CUPTI activity 的时长是 `exact` 的。

    诚实边界：这里的数字是**时间占比**，不是节省比例、不是钱。所以它带 `basis: time_share`、
    `money: unavailable`，并由 `needs` 说明升级到第二级还缺什么 —— 绝不让读者把它当钱。
    """
    kernels = [k for k in (getattr(rep, "kernels", []) or [])
               if float(getattr(k, "total_ns", 0) or 0) > 0]
    if not kernels:
        return None
    kernels.sort(key=lambda k: float(k.total_ns), reverse=True)
    total = sum(float(k.total_ns) for k in kernels)
    if total <= 0:
        return None
    needs = []
    if metric_value(rep, "est__achieved_gbs") is None:
        needs.append("calibration")            # 本机实测峰值（tools/machine-profile.py）
    if metric_value(rep, "ks__cost_per_hour") is None:
        needs.append("price")                  # --ks-price-per-hour / --ks-price-per-kwh
    # facts 只放**标量**：一屏契约要求"包里的每个 fact 都出现在人视图里"，列表/嵌套结构
    # 会让那条契约失效（或逼渲染层去猜）。所以 top 压成一行人类可读、机器可 split 的字符串。
    top_line = "；".join(f"{_clip(str(k.name), 40)} {100.0 * float(k.total_ns) / total:.1f}%"
                        for k in kernels[:limit])
    return {
        "basis": "time_share",
        "money": "unavailable",
        "needs": ",".join(needs) or "workload",
        "n": len(kernels),
        "top_kernel": _clip(str(kernels[0].name), FACT_STR_LIMIT),
        "top_pct": round(100.0 * float(kernels[0].total_ns) / total, 1),
        "top": top_line,
    }


def _verdict_saving(rep: Any) -> Verdict | None:
    """处方判词（C2）—— "先改哪个、能省多少"，只有在算得出头寸时才出。

    三条边界（与 analyze/prescription.py 同源，这里只负责呈现）：
      * 比例的分母必须是**本机实测峰值**（规格表/覆盖值算不出头寸）；
      * 没有价格就没有钱（比例照给，状态为 soft）；
      * 头寸重叠不相加（由处方模块保证，包里的 total 就是它给的那个数）。
    """
    p = (getattr(rep, "meta", {}) or {}).get("prescription")
    disc = _discovery_ranking(rep)
    if not p:
        # 第二级（保守可省区间 + 钱）算不出来时，**第一级不许消失**（M13）：
        # 按实测时间占比排出的头寸仍然可行动，并明说缺什么才能升级。
        if disc is None:
            return None
        return Verdict(
            id="saving_estimate", question="先改哪里（按实测占比排的头寸）", status=SOFT,
            facts=disc, value=disc["top_pct"], unit="%", provenance="exact",
            metrics=("gpu__time_duration.sum",),
            next=("先看占比最高的那一条；要拿到「保守可省区间 + 钱」需要："
                  + ("本机标定（`tools/machine-profile.py` → --ks-calibration）"
                     if "calibration" in disc["needs"] else "")
                  + ("；" if "calibration" in disc["needs"] and "price" in disc["needs"] else "")
                  + ("价格（--ks-price-per-hour 或 --ks-price-per-kwh）"
                     if "price" in disc["needs"] else "")),
            claim=(f"按**实测时间占比**排的头寸：{disc['top_kernel']} 占 {disc['top_pct']}%"
                   f"（共 {disc['n']} 个 kernel）。这是时长占比，**不是**节省比例、也不是钱 —— "
                   f"要钱与区间还需：{disc['needs']}"))
    ranked = list(p.get("ranked") or [])
    if not ranked:
        code = p.get("refusal") or p.get("money_refusal") or RefusalCode.NO_WORKLOAD.value
        try:
            code_enum = RefusalCode(str(code))
        except ValueError:
            code_enum = RefusalCode.NO_WORKLOAD
        return Verdict(
            id="saving_estimate", question="先改哪里、能省多少钱", status=REFUSED,
            facts=(disc or {}),          # 拒绝也可以带"第一级"的实测排序（拒绝≠空）
            refusal=Refusal(code_enum, str(p.get("refusal_message")
                                          or p.get("money_refusal_message")
                                          or "没有可辩护的头寸：缺达成值或本机实测峰值")),
            metrics=("ks__prescription_headroom_n",),
            next="给 --ks-gemm M,N,K（或 --ks-flops/--ks-bytes）并加 --ks-calibration "
                 "machine-profile.json，才有实测分母来算头寸",
            claim="不给节省估计：没有可辩护的头寸（缺少达成值或本机实测峰值）")
    top = ranked[0]
    facts: dict[str, Any] = {"n": len(ranked), "top_kernel": _clip(str(top.get("kernel", "")),
                                                                  FACT_STR_LIMIT),
                             "confidence": top.get("confidence", "")}
    if top.get("pct_lo") is not None:
        facts["top_pct_lo"] = round(float(top["pct_lo"]), 1)
        facts["top_pct_hi"] = round(float(top["pct_hi"]), 1)
    if p.get("total_per_hour") is not None:
        facts["total_per_hour"] = round(float(p["total_per_hour"]), 4)
        facts["currency"] = p.get("currency", "CNY")
    if p.get("total_per_mtok") is not None:
        facts["total_per_mtok"] = round(float(p["total_per_mtok"]), 4)
    if p.get("money_refusal"):
        facts["no_money"] = p["money_refusal"]
    status = OK if p.get("total_per_hour") is not None else SOFT
    unit = f"{p.get('currency', 'CNY')}/h"
    return Verdict(
        id="saving_estimate", question="先改哪里、能省多少钱", status=status, facts=facts,
        value=facts.get("total_per_hour"), unit=unit if facts.get("total_per_hour") else "",
        provenance="estimated", premises=[],
        metrics=("ks__prescription_total_saving_per_hour",),
        next="按 Priority 顺序改一处再重跑同一条命令；用 kernoscope diff 确认变化超过 MDD，"
             "再用 ks__cost_per_mtok 看钱有没有降",
        claim=(f"头寸 {len(ranked)} 条：最大一条是 {facts.get('top_kernel')}"
               f"（保守 {facts.get('top_pct_lo')}%~上界 {facts.get('top_pct_hi')}%）"
               + (f"，合计可省 {facts.get('total_per_hour')} {unit}"
                  if facts.get("total_per_hour") else "；钱给不出（缺价格或份额）")))


def _verdict_over_peak(rep: Any) -> Verdict | None:
    owner = kernel_with(rep, "ks__over_peak_warning")
    warn = metric_value_in(owner, "ks__over_peak_warning") if owner is not None else None
    if not warn:
        # W-12：`kernel_with` 会跳过 `(device)` 伪 kernel。若这个警告只挂在设备级
        # （`-i` 回读老报告时可达），旧实现会让 `over_peak` 判词**静默消失** —— 相对旧版
        # `find_metric` 是回归。这里退回设备级读取（`find_metric` 不跳过 `(device)`）。
        m = find_metric(rep, "ks__over_peak_warning")
        if m is None or m.value is None:
            return None
        return Verdict(
            id="over_peak", question="达成值是否超过标定峰值（物理上不可能）", status=REFUSED,
            facts={"kernel": "(device)"},
            owner_key=str(getattr(getattr(rep, "device", None), "name", "") or "(device)"),
            premises=[Premise("over_peak", 1.0, None, False)],
            refusal=Refusal(RefusalCode.CALIBRATION_CALIBER_MISMATCH,
                            (m.note if m is not None else "") or "achieved > measured peak"),
            metrics=("ks__over_peak_warning",),
            next="重跑标定（tools/machine-profile.py），并确认 --ks-gemm/--ks-flops 的工作量正确",
            claim="达成值超过本机实测峰值：本次的百分比不成立，先查标定与工作量输入")
    m = find_metric_in(owner, "ks__over_peak_warning")
    return Verdict(
        id="over_peak", question="达成值是否超过标定峰值（物理上不可能）", status=REFUSED,
        # V-6：超峰是**某个 kernel** 的事，必须点名，否则会被当成整份报告的结论
        facts={"kernel": _clip(str(getattr(owner, "name", "")), FACT_STR_LIMIT)},
        owner_key=str(getattr(owner, "name", "") or ""),
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
    "cost_per_token",
    "truth_vs_substitute",
    "saving_estimate",
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

BUILDERS = (_verdict_kernel_time, _verdict_cost, _verdict_truth, _verdict_saving,
            _verdict_dram_total, _verdict_per_launch,
            _verdict_trace, _verdict_ledger, _verdict_occupancy, _verdict_roofline,
            _verdict_issue_slot, _verdict_l2, _verdict_over_peak, _verdict_static_mix)


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


def _protocol_inputs(rep: Any) -> dict[str, Any]:
    """协议输入的**可读副本**（V-26）：标定与轨迹的指纹，供案件目录等消费者直接用。

    只放稳定量（文件名 + 内容指纹），不含绝对路径；没有就不放这个键。
    """
    meta = getattr(rep, "meta", {}) or {}
    out: dict[str, Any] = {}
    for key in ("calibration", "memtrace"):
        v = meta.get(key)
        if isinstance(v, dict) and v:
            # `_calibration_identity` 给的是 "文件名:摘要" 这样的字符串，
            # `_memtrace_identity` 给的是 {name, sha256} —— 两种都照带，只剥掉绝对路径。
            out[key] = {k: v[k] for k in sorted(v) if k not in ("path", "abspath")}
        elif isinstance(v, str) and v:
            out[key] = v
    return out


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
# 不能被截断挤掉的"守卫"判词：它们一旦消失，别的判词会带着**已被推翻的前提**留在包里
# （`over_peak` 不成立 ⇒ roofline 的达成率不可引用）。与结论四条一起进优先集合。
GUARD_VERDICT_IDS: tuple[str, ...] = ("over_peak",)


def _priority_ids() -> tuple[str, ...]:
    """结论四条 + 守卫（延迟读取：`PRIMARY_FACT_IDS` 定义在文件靠后的裁剪区）。"""
    return PRIMARY_FACT_IDS + GUARD_VERDICT_IDS


def _select_verdicts(vs: list[Verdict], n: int) -> tuple[list[Verdict], list[str]]:
    """按优先级选留 n 条：**结论与守卫先保**，其余按原顺序补足。

    返回 `(选留的判词, 被丢掉的 id)`；两个列表都保持**原顺序**（外部看到的顺序不变，
    `verdicts_more.ids` 也仍是原序）。`build()` 的 `MAX_VERDICTS` 与 `enforce_budget`
    的预算裁剪共用这一条策略 —— 两处若各写一套，迟早会分叉。
    """
    if len(vs) <= n:
        return list(vs), []
    priority = [i for i, v in enumerate(vs) if v.id in _priority_ids()]
    rest = [i for i, v in enumerate(vs) if v.id not in _priority_ids()]
    keep = set((priority + rest)[:n])
    return ([v for i, v in enumerate(vs) if i in keep],
            [v.id for i, v in enumerate(vs) if i not in keep])


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
    # V-6：只有当超峰与 roofline 属于**同一个 kernel** 时才传播。身份用 `owner_key`
    # （未截断名，**不是** facts 里那个 48 字显示串 —— 见 Verdict.owner_key 的注释）；
    # 两边都拿不到身份时保守传播（宁可拒绝一个可疑的达成率，也不要放行一个物理不可能的）。
    op_k = op.owner_key or (op.facts or {}).get("kernel") or ""
    roof_k = roof.owner_key or (roof.facts or {}).get("kernel") or ""
    if op_k and roof_k and op_k != roof_k:
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
        obs = ("**读不到**（未知，不是 0）" if main.observed is None
               else f"实测 {main.observed}{main.unit}")
        msg = (f"前提不成立（{premise_label(main.id)}：{obs}"
               f"，门槛 {main.threshold}{main.unit}）⇒ 这个数字不可引用；"
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
    # 前提降级与"物理不可能"传播必须在**截断之前**做完：它们是判词之间的**逻辑约束**
    # （over_peak 不成立 ⇒ roofline 的达成率不可引用），而不是显示层的事。截断在前的话，
    # over_peak 一旦被 MAX_VERDICTS 挤掉，roofline 就带着一个已被推翻的前提留在包里。
    _downgrade_broken_premises(verdicts_all)
    _propagate_over_peak(verdicts_all)
    # 上限截断**按优先级**选留（结论四条 + 守卫 `over_peak` 先保），输出顺序不变。
    # 位置截断会让"多了一条判词"顺手把守卫挤出去：M13 让 `saving_estimate` 常态化出现，
    # 实测就把 `over_peak` 顶出了 10 条上限（roofline 于是带着一个已被推翻的前提留在包里）。
    verdicts, dropped = _select_verdicts(verdicts_all, MAX_VERDICTS)
    # R-53/D10：上限截断必须**留痕** —— 否则读者（人或 agent）不知道包里少了一条，
    # 而 `cost.trimmed` 只记预算裁剪、记不到这一步（它在预算裁剪之前就发生了）。
    verdicts_more = ({"count": len(dropped), "ids": dropped} if dropped else None)

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
        # V-26①：`protocol_hash` 读的那批输入（标定/轨迹指纹）也要能**被消费者看到** ——
        # 案件目录的 `manifest.protocol_inputs` 此前恒空，因为包只带了 hash 没带输入。
        # 形状与 `case.py` 已经在读的键一致（扁平），消费者零改动。
        "protocol": {"hash": protocol_hash(rep), "basis": basis,
                     "window_mode": _window_mode(rep),
                     **_protocol_inputs(rep)},
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
        remedy 命令 > 处置文本截短 > **次要判词的 facts** > 拒绝条目数 >
        拒绝处置散文 > 环境降级诊断的 message > 环境降级诊断条目数 >
        **判词条数** > **判词硬上限** > **结论四条的 facts（最后一级）**

    最后一级的顺序是真机教出来的：8 个拒绝码的机器上，facts 早于拒绝条目被裁掉，
    于是一屏出现「最耗时的是 ?（None ms）」—— 把"什么最贵/多少钱/先改哪里"裁没了，
    却留着 8 条拒绝的处置散文。结论比处置散文重要，所以结论的 facts 放到最后。

    每一步都只删"能从完整报告里再取回来"的东西，并**只把真正改动了包的步骤**写进
    `cost.trimmed`（R-110：旧实现连 no-op 步骤也记，"裁剪步骤"因此不等于"真的裁了什么"）。
    条目级留痕有三个出口，所以"少了什么"永远可查：
      * 丢掉的拒绝 code → `refusals_omitted.codes`；
      * 丢掉的判词 id → `verdicts_more.ids`（保持原序；**先保结论四条**，见 `_cap_verdicts`）；
      * 被裁掉 facts 的判词 id → `cost.trimmed_ids`（`"<步骤>:<id>"`，R-106）。
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

    def _record_dropped_verdicts(dropped_ids: list[str]) -> None:
        """把预算裁掉的判词**记下来**（V-1：结论不许静默消失）。

        `verdicts_more` 本来就是为"包里少了判词"而存在的字段；此前只有 `build()` 的
        `MAX_VERDICTS` 截断会写它，预算裁剪这一步不写 —— 于是一屏上少了 `occupancy`
        这种判词（还带着"物理不可能"守卫），读者与 agent 都无从知道。
        这里把两处截断的 id 合并进同一个字段（`count` 是总数），并**按原序**输出。
        """
        ids = [i for i in dropped_ids if i]
        if not ids:
            return
        more = packet.get("verdicts_more") or {"count": 0, "ids": []}
        more["count"] = int(more.get("count") or 0) + len(ids)
        more["ids"] = sorted(list(more.get("ids") or []) + ids,
                             key=lambda i: _rank.get(str(i), len(_rank)))
        packet["verdicts_more"] = more

    # `verdicts_more.ids` 必须按**判词在包里出现过的原序**输出（R-110）。裁剪会分几轮
    # （先 6 条、再 4 条），每轮各自"按当前顺序"记，合起来就不是原序了。所以先建一张排名表：
    # `build()` 已经记下的 id 排在最前（它们的相对顺序本来就是原序），再排当前包里的顺序。
    _rank: dict[str, int] = {}
    for i, vid in enumerate((packet.get("verdicts_more") or {}).get("ids") or []):
        _rank.setdefault(str(vid), i)
    for i, v in enumerate(packet.get("verdicts") or [], start=len(_rank)):
        _rank.setdefault(str(v.get("id") or ""), i)

    def _record_trimmed_items(step: str, ids: list[str]) -> None:
        """把**被裁掉的条目 id** 记进 `cost.trimmed_ids`（第六轮 R-106/R-110）。

        为什么：`cost.trimmed` 只记步骤名，于是"facts 被裁"这件事知道**发生了**，却不知道
        **是谁的 facts 没了**（集群②在 2400 个场景里命中 663 次 facts 被裁，全都只有步骤名）。
        记成 `"<步骤>:<id>"` 的扁平列表：既省字节，又能直接 grep。
        """
        ids = [i for i in ids if i]
        if not ids:
            return
        got = packet["cost"].setdefault("trimmed_ids", [])
        got.extend(f"{step}:{i}" for i in ids)

    def _drop_facts(predicate) -> None:
        """按条件丢 facts，并**逐个记下 id**（不记的话"结论的数字没了"无从察觉）。"""
        dropped = [v.get("id") for v in packet.get("verdicts") or []
                   if v.get("facts") is not None and predicate(v)]
        for v in packet.get("verdicts") or []:
            if v.get("facts") is not None and predicate(v):
                v.pop("facts", None)
        return dropped

    def _cap_verdicts(n: int) -> None:
        """按预算保留 n 条判词，**先保结论四条**（R-106）。

        旧实现是 `vs[:n]` —— 靠位置，只因为 `BUILDERS` 的顺序恰好把结论排在前面才没丢；
        集群②改一下顺序就构造出"结论四条整条掉出包"的红。这里改成按 `PRIMARY_FACT_IDS`
        显式排序选留，而**输出顺序保持不变**（外部消费者看到的判词顺序不变）。
        """
        vs = packet.get("verdicts") or []
        if len(vs) <= n:
            return
        priority = [i for i, v in enumerate(vs) if v.get("id") in _priority_ids()]
        rest = [i for i, v in enumerate(vs) if v.get("id") not in _priority_ids()]
        keep = set((priority + rest)[:n])
        # 丢掉的是"原来顺序里没被选中的"，所以 `verdicts_more.ids` 保持原序（R-110）
        _record_dropped_verdicts([v.get("id") for i, v in enumerate(vs) if i not in keep])
        packet["verdicts"] = [v for i, v in enumerate(vs) if i in keep]

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
        # 真机教训（拒绝密集的机器上包被裁到 facts 全丢）：facts 是判词的**动态内容**，
        # 而 `kernel_time` / `cost_per_token` / `saving_estimate` / `truth_vs_substitute`
        # 这四条是"什么最贵、多少钱、先改哪里"的结论本身 —— 一屏上出现
        # "最耗时的是 ?（None ms）"就等于把结论裁没了。所以先只裁次要判词的 facts，
        # 到下一个更狠的级别才动这四条。
        ("facts_secondary", lambda: _record_trimmed_items(
            "facts_secondary",
            _drop_facts(lambda v: v.get("id") not in PRIMARY_FACT_IDS))),
        # V-1（第五轮 P0）：**拒绝散文先裁、判词最后裁**。
        # 旧顺序把 `verdicts` 排在 `refusal_entries`/`refusal_action_text` 之前，
        # 与函数自己的 docstring（"结论比处置散文重要"）相反：真机 9 条判词 → 包 6 条，
        # 被删的正是 `occupancy`/`roofline`/`over_peak`（含"物理不可能"守卫），
        # 而 8 条拒绝的处置散文一条不少。处置文本随时能从报告与参考层再取回
        # （`remedy.sh <code>`），判词不能。
        ("refusal_entries", _cap_refusal_entries),
        ("refusal_action_text", _drop_refusal_action_text),
        # 环境降级诊断：message 先丢，再限条目数（R-70）
        ("environment_messages", _trim_environment_messages),
        ("environment_entries", _cap_environment_entries),
        # 最后才动结论条数，而且**每一次都留痕**（`verdicts_more`）
        ("verdicts", lambda: _cap_verdicts(6)),
        ("verdicts_hard_cap", lambda: _cap_verdicts(VERDICT_KEEP_HARD)),
        # **最后一级**：到这一步才允许把"什么最贵/多少钱/先改哪里/真值偏差"的 facts 丢掉。
        # 读者仍然能拿到判词 id/status/value/unit 与全部拒绝码，只是细节要回完整报告看。
        ("facts_primary", lambda: _record_trimmed_items("facts_primary", _drop_facts(lambda v: True))),
    ]
    for name, fn in steps:
        before = dumps(packet)
        fn()
        # `cost.trimmed` 的语义（R-110）：**实际改动了包的步骤**，按应用顺序。
        # 旧实现把 no-op 步骤也记进去，"裁剪步骤"于是不等于"真的裁了什么"。
        # 判据用内容比较而不是字节数：有一步（裁判词）会同时**删**判词、**加** `verdicts_more`
        # 记账，净字节可能变大 —— 按字节数判会漏记。
        if dumps(packet) != before:
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
# 这几条判词的 facts 是"什么最贵/多少钱/先改哪里/真值偏差"的结论本身，最后才裁。
PRIMARY_FACT_IDS: tuple[str, ...] = ("kernel_time", "cost_per_token", "saving_estimate",
                                     "truth_vs_substitute")


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
    if fid == "cost_per_token":
        # C1：把成本判词说成人话。单位成本与"每小时成本"是两个不同的口径，缺哪个说哪个 ——
        # 只印 id（旧行为）会让一屏上出现一行光秃秃的 "cost_per_token"。
        bits = []
        if f.get("tokens_s"):
            bits.append(f"{float(f['tokens_s']):,.0f} token/s")
        if f.get("j_per_token") is not None:
            bits.append(f"{float(f['j_per_token']):.3g} J/token")
        if f.get("per_mtok") is not None:
            bits.append(f"单位成本 {f['per_mtok']} {v.get('unit', '')}".strip())
        elif f.get("per_hour") is not None:
            bits.append(f"每小时成本 {f['per_hour']} {v.get('unit', '')}".strip())
        if f.get("top_kernel"):
            # 单位要从"每百万 token"里取出币种再拼 /h：直接拼 unit + "/h" 会得到
            # "CNY/Mtoken/h" 这种不存在的单位（真机一屏上出现过）。
            cur = (v.get("unit") or "CNY/Mtoken").split("/")[0] or "CNY"
            bits.append(f"最贵的是 {f['top_kernel']}"
                        f"（{f.get('top_kernel_per_hour')} {cur}/h 量级）".strip())
        return "；".join(bits) or "成本账"
    if fid == "saving_estimate":
        if f.get("basis") == "time_share":
            # 发现层第一级（M13）：按实测时间占比排的头寸，**明说这不是钱**
            # 人话里带上机器字段（basis/money）：一屏契约要求"包里的每个 fact 都能在
            # 人视图里看到"，藏起来就等于两个视图分叉。
            return (f"按**实测时间占比**排的头寸（basis={f.get('basis')}；"
                    f"money={f.get('money')}）：{f.get('top')}"
                    f"（共 {f.get('n')} 个 kernel；最大一条 {f.get('top_pct')}%）"
                    f"；这是时长占比**不是钱**，要钱与区间还需补：{f.get('needs')}")
        if f.get("top_pct_lo") is not None:
            return (f"先改 {f.get('top_kernel')}（保守可省 {f['top_pct_lo']}%"
                    f"~上界 {f['top_pct_hi']}%）"
                    + (f"；合计可省 {f['total_per_hour']} {f.get('currency', 'CNY')}/h"
                       if f.get("total_per_hour") is not None else "；钱给不出"))
        return f"头寸 {f.get('n')} 条，但都还没有可辩护的区间"
    if fid == "truth_vs_substitute":
        return (f"与 ncu {f.get('ncu', '?')} 的真值对拍 {f.get('compared')}/{f.get('pairs')} 条"
                + (f"，最大相对偏差 {f['max_abs_err_pct']}%"
                   if f.get("max_abs_err_pct") is not None else ""))
    if not f and v.get("value") is not None:
        # facts 被尺寸预算裁掉时也要说人话：`value`/`unit` 还在包里，别印一个光秃秃的 id
        # （真机一屏曾经因为 facts 全丢而出现"最耗时的是 ?（None ms）"）。
        return (f"{v.get('id')}：{v.get('value')} {v.get('unit', '')}"
                f"（facts 被尺寸预算裁掉，细节见完整报告）").replace("  ", " ")
    return v.get("id", "")


def _one_line_summary(packet: dict[str, Any]) -> str:
    """一屏顶部的**一行结论**：最能决定下一步的那件事。

    派生规则（确定、可读）：
      * 有 `refused` 判词 → "N 条判词被拒（code…），它们不能当依据"；
      * 否则全部成立 → "M 条判词成立"；再补一句"最该动的一步"（取 next_ranked 第一条）。
    永远不编造"应该改哪一行代码" —— 那是 C2（收益预估/处方排序）的活，且必须由测量推出。
    """
    verdicts = packet.get("verdicts") or []
    refused = [v for v in verdicts if v.get("status") == REFUSED]
    ok = [v for v in verdicts if v.get("status") != REFUSED]
    if not verdicts:
        return "没有可下结论的判词：先看「没测到什么」"
    parts: list[str] = []
    if refused:
        codes = []
        for v in refused:
            c = (v.get("refusal") or {}).get("code")
            if c and c not in codes:
                codes.append(c)
        parts.append(f"{len(refused)}/{len(verdicts)} 条判词被拒（{', '.join(codes[:3])}）"
                     f"，它们不能当依据")
    if ok:
        parts.append(f"{len(ok)} 条成立")
    ranked = next_ranked(packet, limit=1)
    if ranked and ranked[0].get("code"):
        parts.append(f"先解 {ranked[0]['code']}")
    return "；".join(parts)


def next_ranked(packet: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """「路径」：按**挡住决策的程度**排序的下一步。

    排序规则（确定、可测，不是模板话术）：
      1. `refused` 的判词（它们的数不能当依据 ⇒ 挡路最狠），按判词顺序（判词顺序本身就是
         "先看哪个"的意图：kernel_time → dram_bytes → …）；
      2. 其余判词自带的 `next` 提示（已经成立，但可以更省/更准）；
      3. 未测到的组（缺数据源 ⇒ 换口径或补依赖）。
    同一 code / 同一提示只出现一次。
    """
    index = refusal_index(packet)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        key = item.get("code") or item.get("hint") or item.get("for", "")
        if key in seen:
            return
        seen.add(key)
        out.append(item)

    for v in packet.get("verdicts") or []:
        if v.get("status") == REFUSED:
            code = (v.get("refusal") or {}).get("code", "?")
            r = index.get(code) or {}
            add({"for": v.get("id"), "code": code, "action": r.get("action"),
                 "cmd": r.get("cmd"), "kind": "refused"})
    for v in packet.get("verdicts") or []:
        if v.get("status") != REFUSED and v.get("next"):
            add({"for": v.get("id"), "hint": v["next"], "kind": "advisory"})
    for g in packet.get("unmeasured") or []:
        add({"for": g.get("code"), "code": g.get("code"),
             "action": (index.get(g.get("code")) or {}).get("action"),
             "hint": f"{g.get('count')} 个指标没测到", "kind": "unmeasured"})
    return out[:limit]


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
    # C5：三段式（结论 → 发现 → 路径）的**结论**一行 —— agent 先读这一行就够决定要不要往下看
    _summary = _one_line_summary(packet)
    if _summary:
        out.append(f"结论：{_summary}")
        out.append("")

    verdicts = packet.get("verdicts") or []
    out.append("-- 发现（判词与前提）--")
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
    out.append("-- 路径（下一步，按挡住决策的程度排序）--")
    for i, it in enumerate(next_ranked(packet), 1):
        if it.get("code"):
            out.append(f"  {i}. [{it['for']}] 先解 {it['code']}"
                       + (f"：{it['action']}" if it.get("action") else "")
                       + (f"  [{it['cmd']}]" if it.get("cmd") else ""))
        elif it.get("hint"):
            out.append(f"  {i}. [{it['for']}] {_clip(it['hint'], 88)}")
    if not next_ranked(packet):
        out.append("  （没有待办：判词都成立；改代码后再跑 diff）")
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
