"""处方：**哪个改动能省多少钱、我们有多确定** —— 预期节省 + 排序。

为什么需要这个模块：`cost.py` 只回答"这个 kernel 现在花了多少钱"，`dram.py` / `roofline.py`
只回答"它离本机实测峰值还有多远"。人（或 agent）真正要做的决策是第三种问题：
**先把力气花在哪一个改动上**。所以这里把"剩余空间"换算成"预期节省"，再按节省排序。

这个模块的全部难度不在算术，而在**不撒谎**。下面每一条纪律都对应一个曾经的失败模式：

① 没有价格就没有钱（rule 1）
   钱只可能来自调用方给的价格。拿不到价格时给 `saving_per_hour is None` + 结构化拒绝，
   绝不用"典型电价 0.6 元/度"顶替 —— 编出来的货币数字会被直接拿去做租卡决策。

② 没有标定就没有区间（rule 2）
   剩余空间的分母必须是**本机实测峰值**（`machine-profile`），不是规格表。
   规格表峰值（例如 4090 的 82.6 TFLOP/s fp32 对实测 ~54）会让"达成 65%"这类结论整体偏错，
   于是区间也跟着偏。规格覆盖值（`MachinePeaks.from_spec`）一律拒绝用于头寸估计。

③ 区间必须两端都有说法（rule 3，算术见下）
   区间不能是"±20%"这种拍出来的数：保守端必须能被一个可辩护的论证达到，
   乐观端必须**显式标注为上界**（"这是上界，不是预测"）。算术全文：

   记 P = 本机实测峰值，A = 达成值（两者同口径、同单位）。记缺口比例
   u = (P − A)/P。**时间节省**的物理含义：当前搬/发一个单位工作量要 1/A 秒，
   把速率提到 X 之后要 1/X 秒，于是省下的**运行时间比例** = 1 − A/X。
   比例与金额用的是同一个数（"省下多少运行时间"），所以可以直接相乘。

     区间下界（保守，唯一可辩护的论证："把缺口关掉一半"）：
       新速率 X = A + (P − A)/2 = (A + P)/2
       时间节省 = 1 − A/X = (P − A)/(P + A) = u/(2 − u)
     区间上界（乐观，**上界不是预测**：假设该资源被完全填满，且没有别的瓶颈顶上来）：
       新速率 X = P ⇒ 时间节省 = 1 − A/P = u

   （一般式：关掉缺口比例 c ⇒ 时间节省 = c·u/(1 − u + c·u)；c = 1/2 即下界、c = 1 即上界。
     公式与常量 `CONSERVATIVE_CLOSE_FRACTION` 绑在一起，改一个必须改另一个。）

   注意 A=2400、P≈2985.19（u=0.1960）时：下界 10.81%、上界 19.61% —— 两端都是上式的精确值。

   金额（与比例同数）：
       saving_per_hour(保守) = 可归因成本/h × u/(2 − u)
       saving_per_hour(乐观) = 可归因成本/h × u

④ 排序按"能省多少"，不许静默跨置信度混排（rule 4）
   排序键分三段，顺序固定：没有数（区间为 None）的条目永远排在最后 → 按保守端金额降序 →
   同金额再看保守度。每一条都带 `confidence`（measured / estimated / unknown），
   读者一眼能分清"实测的结论"与"模型的结论"。

⑤ 重叠的节省不许相加（rule 5）
   同一个 kernel 上的两个头寸（例如 DRAM 头寸与发射槽头寸）**几乎一定重叠**：
   填满 DRAM 会让 kernel 变快，而那同时也会改变发射槽利用率。
   把两个百分比按同一个分母相加，就是把"同一段时间"卖了两遍 —— 这是这类功能最经典的撒谎方式。
   所以：只有**成对不重叠**的条目才求和；同一 kernel 上的多个条目取最大者封顶，
   并把"为什么没有相加"和"相加会是多少"一起写进 note（透明，而不是悄悄丢掉）。

⑥ 一个数字一个名字，名字必须能过 provenance 关卡（rule 6）
   `to_metrics()` 产出的每个 `est__`/`ks__` 名字都必须能构造成真的 `provenance.Metric`。
   `meas__` 只在真的是实测读数时用；由实测再算出来的比例/金额一律 `ks__` + derived。
   **绝不把模型结果挂在 ncu 计数器名（`dram__*` / `sm__*`）下面。**

⑦ 什么都不成立时，空的处方就是诚实答案（rule 7）
   `ranked == []` + 一条结构化拒绝，比"凑一个 5% 的通用建议"有用得多 ——
   前者告诉你缺哪个输入、怎么补；后者会让你去改一个不需要改的东西。

「我们说不了」的地方（这些东西写文档时要照实写）：
  * 没给价格 / kernel 无法按份额归因 → 钱一律不给（`no_price` / `premise_failed`）。
  * 没标定、用的是规格覆盖值、标定口径与本次不可比 → 区间与钱都不给
    （`no_calibration` / `calibration_caliber_mismatch`）。
  * 只有一侧数字（有峰值没达成值，或相反）→ 基向量为空、confidence=unknown，条目排在最后。
  * `saving_per_mtok` 只在被匹配 kernel **主导窗口**（份额 ≥ `DOMINANT_SHARE_PCT`）时才给：
    否则"kernel 变快"不一定变成"端到端吞吐变高"，per-token 节省就是无根据的。
  * 达成值 ≥ 峰值（口径不一致的典型征兆）→ 头寸为 0，并出一条 `premise_failed`，
    而不是报一个正的头寸。

两个拒绝位是**两件事**，`Headroom` 与 `Prescription` 都刻意分成两个字段：
    `refusal`       连**比例**都给不出（缺标定/缺达成值/口径不成立）⇒ 没有区间；
    `money_refusal` 比例能给，但**钱**给不出（缺价格/缺 kernel 份额/缺 token 数）。
合成一个字段会逼出"要么丢掉比例、要么丢掉拒绝"的二选一 —— 两者都是撒谎：比例是物理量
（不需要钱），而"为什么没有钱"必须留下来（否则读者以为我们忘了乘价格）。

用法（本模块是纯函数，不读文件、不跑命令，所以能被测试完整覆盖）：

    cost = attribute(window_estimate, share_pct, steps=n)   # kernoscope.analyze.cost
    calib = Calibration(bandwidth_gbs=1005.0, source="machine-profile.json", measured=True)
    entries = [
        bandwidth_headroom("kcopy", achieved_gbs=420.0, calib=calib, cost=cost),
        bandwidth_headroom("kgemm", achieved_gbs=None, calib=Calibration(), cost=cost),
    ]
    pres = prescribe_or_refuse(entries, currency=cost.inputs.currency)
    for name, value, unit, source in pres.to_metrics():
        ...   # value is None -> Metric.unavailable(name, reason, code=...)，见 to_metrics 文档
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..provenance import Provenance
from ..refusals import Refusal, RefusalCode, Remedy, combined_refusal
from .cost import CostEstimate

# ---------------------------------------------------------------------------
# 常量：每一个都写清楚为什么是这个数（否则后人只能靠猜）
# ---------------------------------------------------------------------------

# 保守端的论证："把差距关掉一半"。为什么是一半而不是某个更漂亮的数：
# 它是**可辩护的最小论证**——"你已经做到过的速率"与"本机实测达到过的速率"之间的中点，
# 不需要任何额外假设。想要更强的主张（例如关掉 3/4）就必须给出额外的证据。
CONSERVATIVE_CLOSE_FRACTION = 0.5

# 达成值必须低于峰值的比例：98% 以上的"头寸"落在标定自身的口径离散（真机 3~4%）之内，
# 报出来只是噪声。
MIN_GAP_FRACTION = 0.02

# per-token 节省的门槛：只有被匹配 kernel 占窗口时间 ≥ 这个比例时，
# "它变快"才等于"整段变快"。低于它，省下的是 GPU 时间，不是 token 成本。
DOMINANT_SHARE_PCT = 50.0

# 资源轴。重叠判定用的就是它：同一 (kernel, 资源) 的两个条目是同一份节省的两种说法。
RESOURCE_DRAM = "dram_bandwidth"
RESOURCE_ISSUE = "issue_slots"

CONF_MEASURED = "measured"
CONF_ESTIMATED = "estimated"
CONF_UNKNOWN = "unknown"

# 达成值出处 → confidence。只有"真的读到的量"才算 measured；
# 由实测再算的比例/代理（derived/estimated）一律降级成 estimated。
# 注意：这个映射**不包含标定文件**，因为能进本模块的 Calibration 必须 already measured
# （见 Calibration.from_spec 的拒绝），标定不是"候选条目"的一部分。
_CONFIDENCE_BY_PROVENANCE: dict[str, str] = {
    Provenance.MEASURED.value: CONF_MEASURED,
    Provenance.DERIVED.value: CONF_ESTIMATED,
    Provenance.ESTIMATED.value: CONF_ESTIMATED,
    Provenance.UNAVAILABLE.value: CONF_UNKNOWN,
}

_MISSING_BASIS = {
    "achieved": Remedy("给出被测 kernel 的达成值：DRAM 口径用 --ks-calibration + "
                       "--ks-window auto 让 NVML 代理算出「达成 GB/s / 实测峰值」，"
                       "发射槽口径用 --ks-instr-hist 给出动态指令数与上界利用率",
                       command="kernoscope --ks-calibration machine-profile.json "
                               "--ks-window auto --ks-instr-hist opcode_hist.txt -- python bench.py",
                       cost_s=600.0, then_answers=("prescription",)),
    "peak_gbs": Remedy("跑一次本机标定并把结果传进来（规格表峰值不能用来估计能省多少）",
                       command="python tools/machine-profile.py --out machine-profile.json",
                       cost_s=300.0, then_answers=("prescription",)),
    "peak_issue": Remedy("给出本机实测可达的发射槽利用率上限（标定/探针），"
                         "不能用「每周期 4 条」这个硬件上限顶替 —— 那是规格，不是实测",
                         command="python tools/machine-profile.py --out machine-profile.json",
                         cost_s=300.0, then_answers=("prescription",)),
}


def _budget_refusal(what: str) -> Refusal:
    """头寸估计缺前提时的统一拒绝（补什么、花多少时间，都在 remedy 里）。"""
    return Refusal(
        RefusalCode.NO_WORKLOAD,
        f"算不出「{what}」的头寸：这个口径的分母（本机实测可达值）与分子（本次达成值）"
        f"没有同时具备。缺哪一侧就说哪一侧 —— 我们不拿规格表或典型值顶替",
        observed={"achieved": None, "measured_peak": None},
        remedy=_MISSING_BASIS.get(what),
        severity="degraded")


def _calibration_refusal(what: str, measured: bool, caliber_ok: bool,
                         note: str = "") -> Refusal:
    """没有可信标定时的拒绝：区分「没标定」与「标定口径不可比」两种。"""
    if not measured or not caliber_ok:
        return Refusal(
            RefusalCode.CALIBRATION_CALIBER_MISMATCH,
            f"「{what}」的分母来自规格表或命令行覆盖，不是本机标定：规格峰值与实测可差 30% 以上"
            f"（4090 fp32 规格 82.6 对实测 ~54 TFLOP/s），用它算头寸会系统性偏错。"
            f"{note}".strip(),
            observed={"peak_source": "spec/override", "caliber_ok": caliber_ok},
            remedy=_MISSING_BASIS.get(what))
    return Refusal(
        RefusalCode.NO_CALIBRATION,
        f"缺少本机实测峰值：没有它，「{what}」的剩余空间没有分母（也就不用谈省多少）",
        observed={"measured_peak": None},
        remedy=_MISSING_BASIS.get(what))


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """本机实测峰值（头寸估计的**唯一**合法分母）。

    `measured=False` 表示这个数来自 `--ks-peak-*` 覆盖或规格表（对应
    `roofline.MachinePeaks.from_spec`）。它仍然可以用于展示，但**不用于头寸估计** ——
    规格与实测的差距正是本工具存在的理由。
    """

    bandwidth_gbs: float | None = None          # 本机实测 DRAM 峰值（machine-profile）
    issue_util_pct: float | None = None         # 本机实测可达发射槽利用率上限（%）
    source: str = ""
    calibrated_at: str = ""
    measured: bool = True
    caliber_ok: bool = True
    caliber_note: str = ""

    @staticmethod
    def from_spec(bandwidth_gbs: float | None = None,
                  issue_util_pct: float | None = None, source: str = "spec/override",
                  note: str = "") -> "Calibration":
        """规格表 / `--ks-peak-*` 覆盖：可以展示，不能算头寸。"""
        return Calibration(bandwidth_gbs=bandwidth_gbs, issue_util_pct=issue_util_pct,
                           source=source, measured=False, caliber_ok=False,
                           caliber_note=note or "峰值来自规格表或命令行覆盖，不是本机标定")

    def source_label(self) -> str:
        when = f" @ {self.calibrated_at}" if self.calibrated_at else ""
        bits: list[str] = []
        if self.bandwidth_gbs:
            bits.append(f"实测峰值带宽 {self.bandwidth_gbs:.0f} GB/s")
        if self.issue_util_pct:
            bits.append(f"实测可达发射槽利用率上界 {self.issue_util_pct:.0f}%")
        return f"{self.source or 'machine-profile'}{when}（" + "，".join(bits) + "）"


@dataclass
class Headroom:
    """一个"改动方向"的预期节省。**每个数都带它的前提与出处。**"""

    kernel: str
    basis: str                      # 人读口径（含达成值与峰值）
    resource: str = ""              # 重叠判定用的资源轴
    achieved_pct: float | None = None       # 达成值占本机实测峰值的百分比；None=未知
    saving_pct_range: tuple[float, float] | None = None   # 保守..乐观（上界）
    saving_per_hour: float | None = None    # 区间下界对应的金额（保守端）
    saving_per_mtok: float | None = None
    confidence: str = CONF_UNKNOWN          # measured | estimated | unknown
    premises: list[str] = field(default_factory=list)
    # 两个拒绝位是**两件事**，刻意分开：
    #   refusal        —— 连**比例**都给不出（缺峰值/缺达成值/口径不成立）⇒ 没有区间；
    #   money_refusal  —— 比例能给，但**钱**给不出（缺价格 / 缺 kernel 份额 / 缺 token 数）。
    # 合成一个字段会逼出"要么丢掉比例、要么丢掉拒绝"的二选一 —— 两者都是撒谎：
    # 比例是物理量（不需要钱），而"为什么没有钱"必须留下来（否则读者以为我们忘了乘价格）。
    refusal: Refusal | None = None
    money_refusal: Refusal | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 不变量：有金额就必须有区间（金额 = 区间下界 × 可归因成本）。
        # 反过来不成立（有区间没价格是合法的：比例不需要钱）。
        if self.saving_per_hour is not None and self.saving_pct_range is None:
            raise ValueError(
                f"{self.kernel}: 有金额却没有区间 —— 金额只能是区间下界乘上可归因成本，"
                f"没有区间的金额一定是编的")
        if self.saving_per_hour is not None and self.confidence == CONF_UNKNOWN:
            raise ValueError(
                f"{self.kernel}: 有金额却标 unknown 置信度 —— 有价格能算钱就至少是 estimated")
        # 不变量：条目级拒绝 ⇔ 连区间都没有（W3 的"判决只能有一个"）。
        if self.refusal is not None and self.saving_pct_range is not None:
            raise ValueError(
                f"{self.kernel}: 同一个条目既拒绝又给区间 —— 判决只能有一个"
                f"（给不出钱请用 money_refusal，别把比例也丢掉）")
        if self.refusal is not None and self.saving_per_hour is not None:
            raise ValueError(f"{self.kernel}: 同一个条目既拒绝又给金额")
        # 钱给不出时必须有 money_refusal（"我们没算"与"给不出"是两件事）。
        if self.saving_per_hour is None and self.money_refusal is None \
                and self.saving_pct_range is not None:
            raise ValueError(
                f"{self.kernel}: 有区间却没有金额、也没有 money_refusal —— "
                f"缺什么必须写出来（no_price / premise_failed），不许静默地不给钱")

    # ---- 阅读辅助 ----
    @property
    def has_range(self) -> bool:
        return self.saving_pct_range is not None

    @property
    def has_number(self) -> bool:
        """这条头寸到底有没有给出**可排序的数**。

        只有区间没有金额是合法的（没价格）：比例本身就是"能省多少"的答案，
        只是还不能换算成钱。所以排序判据是「有没有区间」，不是「有没有钱」。
        """
        return self.saving_pct_range is not None

    @property
    def has_money(self) -> bool:
        return self.saving_per_hour is not None

    @property
    def conservative_pct(self) -> float | None:
        return self.saving_pct_range[0] if self.saving_pct_range else None

    @property
    def optimistic_pct(self) -> float | None:
        return self.saving_pct_range[1] if self.saving_pct_range else None


@dataclass
class Prescription:
    """一份排序后的处方：先做哪个、能省多少、有多确定、为什么现在还不能给钱。"""

    ranked: list[Headroom] = field(default_factory=list)
    total_saving_per_hour: float | None = None
    total_saving_per_mtok: float | None = None
    notes: list[str] = field(default_factory=list)
    refusal: Refusal | None = None           # 整份处方给不出**比例**时的拒绝
    # 比例能给、但**钱**给不出时的整份理由（缺价格/缺份额/缺 token 数）。
    # 与 Headroom 一样分成两个字段：合成一个就会逼出"要么丢掉比例、要么丢掉拒绝"。
    money_refusal: Refusal | None = None
    currency: str = "CNY"

    def to_metrics(self) -> list[tuple[str, float | None, str, str]]:
        """(name, value, unit, source/note) —— 供 section 组装。

        **value 为 None 的条目也要输出**（B1：口径不成立时"拒绝"不许退化成"消失"），
        并在 source/note 的第一段带上 `unavailable: code=<refusal code>`，
        组装方据此用 `Metric.unavailable(..., code=...)` 构造那一行（W3：拒绝必须带 code）。
        有值的名字一律 `est__` / `ks__`，绝不占用 ncu 的计数器命名空间（R2/R3）。
        """
        out: list[tuple[str, float | None, str, str]] = []
        cur = self.currency or "CNY"
        top_code = self.refusal.code.value if self.refusal is not None else ""
        money_code = self.money_refusal.code.value if self.money_refusal is not None else ""
        # 一句话口径：所有行共用，让读者知道分母是"本机实测"而不是规格表。
        how = ("按【保守=关掉一半差距；乐观=完全填满，是上界不是预测】换算；"
               "分母必须来自本机实测峰值")

        def _na(code: str, msg: str) -> str:
            return f"unavailable: code={code} | {msg}"

        out.append(("ks__prescription_headroom_n", float(len(self.ranked)), "个",
                    "排序后的头寸条目数（含只有区间、以及连区间都没有的条目）"))
        if self.total_saving_per_hour is not None:
            out.append(("ks__prescription_total_saving_per_hour",
                        self.total_saving_per_hour, f"{cur}/h",
                        f"只把**成对不重叠**的条目相加；重叠的按最大者封顶。{how}"))
        else:
            reason = (self.refusal or self.money_refusal)
            out.append(("ks__prescription_total_saving_per_hour", None, f"{cur}/h",
                        _na(top_code or money_code or RefusalCode.NO_PRICE.value,
                            "钱一律不给：" + (reason.message[:140] if reason is not None else
                                          "没有任何条目同时具备区间与可归因价格"))))
        if self.total_saving_per_mtok is not None:
            out.append(("ks__prescription_total_saving_per_mtok",
                        self.total_saving_per_mtok, f"{cur}/Mtoken",
                        "只在被匹配 kernel 主导窗口时才给：否则省下的是 GPU 时间，不是 token 成本"))
        else:
            out.append(("ks__prescription_total_saving_per_mtok", None, f"{cur}/Mtoken",
                        _na(top_code or money_code or RefusalCode.PREMISE_FAILED.value,
                            "端到端 token 成本给不出：没有价格、没有 token 数，"
                            "或被匹配 kernel 没有主导窗口（省下的是 GPU 时间，不是 token 成本）")))
        for i, h in enumerate(self.ranked, start=1):
            base = f"prescription__headroom_{i}__{_slug(h.kernel)}"
            out.append((f"ks__{base}__kernel", h.kernel, "", h.basis))
            if h.achieved_pct is not None:
                out.append((f"est__{base}__achieved_pct", h.achieved_pct, "%", h.basis))
            else:
                out.append((f"est__{base}__achieved_pct", None, "%",
                            _na(h.refusal.code.value if h.refusal else
                                RefusalCode.NO_WORKLOAD.value,
                                "达成值不可得：没有它就连比例也给不出")))
            if h.saving_pct_range is not None:
                lo, hi = h.saving_pct_range
                out.append((f"ks__{base}__saving_pct_lo", lo, "%",
                            "保守端：把差距关掉一半 —— " + how))
                out.append((f"ks__{base}__saving_pct_hi", hi, "%",
                            "**乐观端是上界**（假设该资源被完全填满且无其他瓶颈顶上来），"
                            "不是预测值"))
            else:
                code = h.refusal.code.value if h.refusal else RefusalCode.NO_WORKLOAD.value
                out.append((f"ks__{base}__saving_pct_lo", None, "%",
                            _na(code, "没有实测峰值/达成值，区间给不出")))
                out.append((f"ks__{base}__saving_pct_hi", None, "%",
                            _na(code, "同上：上界也不给，避免读者拿它当预测")))
            out.append((f"ks__{base}__confidence", h.confidence, "",
                        "measured=真的读到的量；estimated=模型/代理/由实测再算；unknown=数字不全"))
            money_code = h.money_refusal.code.value if h.money_refusal else (
                top_code or RefusalCode.NO_PRICE.value)
            out.append((f"ks__{base}__saving_per_hour", h.saving_per_hour, f"{cur}/h",
                        h.basis if h.saving_per_hour is not None
                        else _na(money_code,
                                 "没有价格或无法按份额归因：金额一律不给，"
                                 "绝不按典型电价/典型租价顶替一个货币数字")))
            # per-token 有两个额外前提（价格 + kernel 主导窗口），所以它单独报：
            # 有每小时金额但没有 per-token 时，原因必须点明是哪一个前提没成立。
            if h.saving_per_mtok is not None:
                mtok_note = h.basis + "；per-token 口径：被匹配 kernel 主导窗口"
            else:
                mtok_code = (h.money_refusal.code.value if h.money_refusal is not None
                             else RefusalCode.PREMISE_FAILED.value)
                mtok_note = _na(mtok_code,
                                f"per-token 节省要求「有 ks__cost_per_mtok」且"
                                f"「被匹配 kernel 主导窗口（份额 ≥ "
                                f"{DOMINANT_SHARE_PCT:.0f}%）」，否则不给")
            out.append((f"ks__{base}__saving_per_mtok", h.saving_per_mtok,
                        f"{cur}/Mtoken", mtok_note))
        out.append(("ks__prescription_note", "；".join(self.notes) if self.notes else None,
                    "", "口径说明（含「重叠为什么不相加」）"))
        return out


# ---------------------------------------------------------------------------
# 纯算术（全部可手算复核；测试里的期望值就是按这些式子在注释里算出来的）
# ---------------------------------------------------------------------------

def gap_fraction(achieved: float | None, peak: float | None) -> float | None:
    """u = (P − A)/P —— **达成值相对峰值的缺口比例**（正数 = 还有空间）。

    缺口的定义放在这里、只放一次，是为了让"保守端的物理含义"可验证：
    取 A_half = A + u·(P−A)/2（关掉一半缺口 × 一半，即 A_half = (A+P)/2）后，
    真实时间节省 = 1 − A/A_half = (P−A)/(P+A) —— 见 saving_pct_conservative。

    A ≥ P 时返回 None：物理上不可能，只可能是两侧口径不同，由调用方出 premise_failed。
    """
    if achieved is None or peak is None or peak <= 0 or achieved <= 0:
        return None
    if achieved >= peak:
        return None
    return (peak - achieved) / peak


def saving_pct_conservative(u: float) -> float:
    """保守端（可辩护的论证）：**把缺口关掉 1/2** ⇒ 时间节省 = u/(2−u)。

    推导（一般式，设关掉缺口的比例为 c）：
        u = (P−A)/P，则把速率提到 A_c = A + c·(P−A) 后，u_c = (P−A_c)/P = (1−c)·u。
        同一个单位工作量：当前花 1/A，之后花 1/A_c，于是**时间节省比例**
            1 − A/A_c = c·u / (1 − u + c·u)
        （用 A_c/A = (1−u_c)/(1−u) = (1−(1−c)u)/(1−u) 代入。）
    取 c = 1/2（`CONSERVATIVE_CLOSE_FRACTION`）化简得 c·u/(1−u+c·u) = u/(2−u)。

    与 `saving_pct_optimistic`（c = 1）是同一个物理含义（"省下多少运行时间"），
    所以比例与金额可以直接相乘，不需要第二套口径。
    """
    # 闭式解只在 c = 1/2 时成立。改常量而不改公式会静默算错，所以这里硬拦一道。
    if CONSERVATIVE_CLOSE_FRACTION != 0.5:      # pragma: no cover - 防御性
        raise AssertionError(
            "saving_pct_conservative 的闭式解只在 CONSERVATIVE_CLOSE_FRACTION == 0.5 时成立；"
            "改常量就必须同时改用一般式 c·u/(1−u+c·u)")
    return 100.0 * u / (2.0 - u)


def saving_pct_optimistic(u: float) -> float:
    """乐观端（**上界**）：该资源被完全填满 ⇒ 时间节省 = u = (P−A)/P。"""
    return 100.0 * u


def _money(fraction_pct: float, attributable_cost_per_hour: float | None
           ) -> float | None:
    """金额 = 可归因成本 × 比例。**没有可归因成本就没有金额**（不猜价格、不猜份额）。"""
    if attributable_cost_per_hour is None:
        return None
    return attributable_cost_per_hour * fraction_pct / 100.0


# ---------------------------------------------------------------------------
# 单个头寸的构造
# ---------------------------------------------------------------------------

def bandwidth_headroom(kernel: str, *, achieved_gbs: float | None = None,
                       calib: Calibration | None = None,
                       achieved_provenance: str = Provenance.ESTIMATED.value,
                       achieved_note: str = "",
                       cost: CostEstimate | None = None,
                       kernel_share_pct: float | None = None,
                       cost_per_mtok: float | None = None) -> Headroom:
    """DRAM 头寸：达成 GB/s 对**本机实测峰值**带宽。

    达成值在受限机器上通常来自 NVML 代理（`est__dram_throughput_pct_of_peak`），
    所以默认出处是 estimated ⇒ `confidence = "estimated"`。DRAM 的 `caliber_ok`
    （累加口径）不影响带宽侧（见 `roofline.MachinePeaks`），所以只查 `measured`。
    """
    calib = calib or Calibration()
    basis = _bandwidth_basis(achieved_gbs, calib, achieved_note)
    pct = _achieved_pct(achieved_gbs, calib.bandwidth_gbs)
    calibrated = bool(calib.bandwidth_gbs and calib.bandwidth_gbs > 0)
    if not calibrated or not calib.measured:
        # 拒绝里区分「没有峰值」（no_calibration）与「峰值是规格/覆盖值」
        # （calibration_caliber_mismatch）—— 两者的补救动作不同，见 _calibration_refusal。
        return Headroom(
            kernel=kernel, resource=RESOURCE_DRAM, basis=basis, achieved_pct=pct,
            confidence=CONF_UNKNOWN, premises=_bandwidth_premises(pct),
            refusal=_calibration_refusal("DRAM 峰值带宽（GB/s）", calib.measured,
                                         calib.caliber_ok, calib.caliber_note),
            notes=[calib.caliber_note] if calib.caliber_note else [])
    return _assemble(kernel=kernel, resource=RESOURCE_DRAM, basis=basis,
                     achieved=pct, peak_pct=100.0,
                     accuracy_provenance=achieved_provenance,
                     premises=_bandwidth_premises(pct),
                     cost=cost, kernel_share_pct=kernel_share_pct,
                     cost_per_mtok=cost_per_mtok, prefix_notes=[])


def issue_headroom(kernel: str, *, achieved_pct: float | None = None,
                   calib: Calibration | None = None,
                   achieved_provenance: str = Provenance.ESTIMATED.value,
                   note: str = "",
                   cost: CostEstimate | None = None,
                   kernel_share_pct: float | None = None,
                   cost_per_mtok: float | None = None) -> Headroom:
    """发射槽头寸：本机可达发射槽利用率上界 vs 本次达成。

    为什么这个头寸特别保守：`analyze.issue` 的利用率是**上界口径**（每 SM 4 调度器、
    每周期各发 1 条是硬件上限），而这里的分母要求"本机实测可达"。拿硬件上限当分母会让
    头寸虚高，所以没有实测分母时宁可拒绝。
    """
    calib = calib or Calibration()
    basis = _issue_basis(achieved_pct, calib, note)
    calibrated = bool(calib.issue_util_pct and calib.issue_util_pct > 0)
    if not calib.measured or not calibrated:
        return Headroom(
            kernel=kernel, resource=RESOURCE_ISSUE, basis=basis, achieved_pct=achieved_pct,
            confidence=CONF_UNKNOWN,
            premises=_issue_premises(achieved_pct),
            refusal=_calibration_refusal("发射槽利用率上界（%）", calib.measured,
                                         calib.caliber_ok, calib.caliber_note),
            notes=[calib.caliber_note] if calib.caliber_note else [])
    return _assemble(kernel=kernel, resource=RESOURCE_ISSUE, basis=basis,
                     achieved=achieved_pct, peak_pct=calib.issue_util_pct,
                     accuracy_provenance=achieved_provenance,
                     premises=_issue_premises(achieved_pct),
                     cost=cost, kernel_share_pct=kernel_share_pct,
                     cost_per_mtok=cost_per_mtok, prefix_notes=[])


def _assemble(*, kernel: str, resource: str, basis: str, achieved: float | None,
              peak_pct: float, accuracy_provenance: str, premises: list[str],
              cost: CostEstimate | None, kernel_share_pct: float | None,
              cost_per_mtok: float | None, prefix_notes: list[str]) -> Headroom:
    """把（达成值、实测分母、成本账）拼成一个头寸 —— 唯一做算术的地方。"""
    confidence = _CONFIDENCE_BY_PROVENANCE.get(
        str(getattr(accuracy_provenance, "value", accuracy_provenance)),
        CONF_ESTIMATED)
    # 可归因成本：要么调用方给了 kernel 份额（cost 只记着**窗口级**每小时成本），
    # 要么 cost 已经是分摊后的 kernel 账（kernel_share_pct 由 attribute() 写进去了）。
    share = _resolve_share(cost, kernel_share_pct)
    attr_cost, cost_ref = _attributable_cost(cost, share)

    if achieved is None:
        return Headroom(
            kernel=kernel, resource=resource, basis=basis, achieved_pct=None,
            saving_pct_range=None, saving_per_hour=None, saving_per_mtok=None,
            confidence=CONF_UNKNOWN, premises=premises,
            refusal=_budget_refusal("achieved"), notes=list(prefix_notes))
    if achieved >= peak_pct * (1.0 - MIN_GAP_FRACTION):
        # 达成值 ≥ 峰值（或落在口径离散之内）：头寸为 0。
        # 这不是"没有优化空间"，而是**口径不可信**的信号（R-38 的同一个红灯），
        # 所以出 premise_failed 而不是一句"已经很好了"。
        gap = 100.0 - achieved
        return Headroom(
            kernel=kernel, resource=resource, basis=basis, achieved_pct=achieved,
            saving_pct_range=None, confidence=CONF_UNKNOWN, premises=premises,
            refusal=Refusal(
                RefusalCode.PREMISE_FAILED,
                f"达成值 {achieved:.1f}% 已达到/接近本机实测分母 {peak_pct:.1f}%："
                f"头寸为 0（差 {gap:.1f} 个百分点，落在标定自身的口径离散之内）。"
                f"先复核口径（dtype/尺寸/工作量输入），不要把它当成'没有优化空间' —— "
                f"达成值超过实测峰值在物理上不可能，只可能是两侧口径不同",
                observed={"achieved_pct": round(achieved, 3),
                          "measured_peak_pct": round(peak_pct, 3)},
                threshold={"min_gap_pct": 100.0 * MIN_GAP_FRACTION}),
            notes=list(prefix_notes))

    u = (peak_pct - achieved) / peak_pct
    pct_pair = (saving_pct_conservative(u), saving_pct_optimistic(u))
    per_hour = _money(pct_pair[0], attr_cost)
    # per-token 的基数：调用方给的优先；没给就用成本账里的 `ks__cost_per_mtok`。
    # **不做二次按份额折算**：调用方传进来的成本账本来就对应本次被测负载
    # （`attribute()` 已经把每小时口径折到那个 kernel 上了），再折一次份额就是重复打折。
    if cost_per_mtok is None and cost is not None and cost.cost_per_mtok is not None:
        cost_per_mtok = cost.cost_per_mtok
    per_mtok = None
    extra_notes: list[str] = []
    if attr_cost is not None and share is not None:
        # per-token 的额外前提：被匹配 kernel 必须主导窗口。否则"它变快"省的是 GPU 时间，
        # 不一定是 token 成本（吞吐可能被别的环节挡住）。
        if share >= DOMINANT_SHARE_PCT and cost_per_mtok is not None:
            per_mtok = cost_per_mtok * pct_pair[0] / 100.0
        else:
            extra_notes.append(
                f"per-token 节省未给出：被匹配 kernel 只占窗口 {share:.0f}%"
                f"（门槛 {DOMINANT_SHARE_PCT:.0f}%）—— 省下的是 GPU 时间，"
                f"不一定是端到端 token 成本，所以不给这个数"
                if share < DOMINANT_SHARE_PCT else
                "per-token 节省未给出：没有 ks__cost_per_mtok（缺 token 数或价格）")
    # 注意 `refusal` 在这里**只可能**是 None：达成值与分母都成立 ⇒ 区间一定给得出。
    # "为什么没有钱"放 money_refusal，这样比例与拒绝都不用被牺牲（见字段注释）。
    return Headroom(
        kernel=kernel, resource=resource, basis=basis, achieved_pct=achieved,
        saving_pct_range=pct_pair, saving_per_hour=per_hour, saving_per_mtok=per_mtok,
        confidence=confidence, premises=premises, refusal=None, money_refusal=cost_ref,
        notes=list(prefix_notes) + extra_notes)


def _resolve_share(cost: CostEstimate | None, given: float | None) -> float | None:
    """调用方给的份额优先；没给就看成本账自己是不是分摊后的 kernel 账。"""
    if given is not None:
        return float(max(0.0, min(100.0, given)))
    if cost is not None and cost.kernel_share_pct is not None:
        return float(max(0.0, min(100.0, cost.kernel_share_pct)))
    return None


def _attributable_cost(cost: CostEstimate | None,
                       share: float | None) -> tuple[float | None, Refusal | None]:
    """(可归因到该 kernel 的每小时成本, 给不出钱的原因)。

    两种口径分开判，因为补救办法不同：
      * 完全没有价格（也没有小时成本）→ `no_price`（去给价格）；
      * 有价格、但没有 kernel 份额（无法归因）→ `premise_failed`（去让窗口里只有它）。
    """
    if cost is None or (cost.cost_per_hour is None and cost.kernel_cost_per_hour is None):
        return None, Refusal(
            RefusalCode.NO_PRICE,
            "没有价格（--ks-price-per-kwh 电费口径 / --ks-price-per-hour 租约口径）："
            "「省多少」需要「现在花多少」，我们不按典型电价或典型租价顶替一个货币数字",
            observed={"cost_per_hour": None})
    if share is None:
        return None, Refusal(
            RefusalCode.PREMISE_FAILED,
            "有价格，但没有该 kernel 的窗口份额（或窗口里不止一个 kernel）："
            "整机的每小时成本不能直接乘到某一个 kernel 的头寸上 —— "
            "要么让被测区间里只有这一个 kernel，要么给出份额",
            observed={"kernel_share_pct": None, "cost_per_hour": cost.cost_per_hour})
    if cost.kernel_cost_per_hour is not None and cost.kernel_share_pct is not None:
        return float(cost.kernel_cost_per_hour), None
    if cost.cost_per_hour is not None:
        return float(cost.cost_per_hour) * share / 100.0, None
    return None, Refusal(
        RefusalCode.PREMISE_FAILED,
        "成本账里既没有窗口级每小时成本，也没有 kernel 级每小时成本：金额无从归因",
        observed={"cost_per_hour": cost.cost_per_hour,
                  "kernel_cost_per_hour": cost.kernel_cost_per_hour})


# ---- 口径文本（basis / premises）：读者必须能从一句话里看出分母是什么 ----

def _achieved_pct(achieved_gbs: float | None, peak_gbs: float | None) -> float | None:
    if achieved_gbs is None or peak_gbs is None or peak_gbs <= 0:
        return None
    return 100.0 * achieved_gbs / peak_gbs


def _bandwidth_basis(achieved_gbs: float | None, calib: Calibration,
                     extra: str = "") -> str:
    if achieved_gbs is None:
        return f"DRAM 带宽口径：达成值不可得（缺 NVML 代理/轨迹字节）；{calib.source_label()}"
    if not calib.bandwidth_gbs:
        return (f"DRAM 带宽口径：达成 {achieved_gbs:.1f} GB/s / **无本机实测峰值**"
                f"（规格表峰值不能当分母）")
    # 措辞必须区分"实测峰值"与"规格/覆盖值"：后者可以用来展示比例，但**不能**当分母
    # 算头寸 —— 一句话里把这件事说错，读者就会把规格值当成实测值（这是最贵的一种错）。
    label = "实测峰值" if (calib.measured and calib.caliber_ok) else "规格/覆盖峰值（不可用于头寸）"
    return (f"DRAM 带宽口径：达成 {achieved_gbs:.1f} GB/s / {label} "
            f"{calib.bandwidth_gbs:.0f} GB/s = {100.0 * achieved_gbs / calib.bandwidth_gbs:.1f}%"
            + (f"；{extra}" if extra else ""))


def _issue_basis(achieved_pct: float | None, calib: Calibration, extra: str = "") -> str:
    if achieved_pct is None:
        return (f"发射槽口径：达成值不可得（缺 --ks-instr-hist 的动态指令数/时钟）；"
                f"{calib.source_label()}")
    if not calib.issue_util_pct:
        return (f"发射槽口径：达成 {achieved_pct:.1f}% / **无本机实测可达上界**"
                f"（每周期 4 条是硬件规格，不是实测）")
    return (f"发射槽口径：达成 {achieved_pct:.1f}% / 本机实测可达上界 "
            f"{calib.issue_util_pct:.0f}%" + (f"；{extra}" if extra else ""))


def _bandwidth_premises(pct: float | None) -> list[str]:
    return [
        "该 kernel 是**访存受限**的：否则填满 DRAM 不会让它变快（计算/延迟瓶颈会顶上来）",
        "达成/峰值两侧同口径（读或读写混合要一致）—— 标定用 copy 速率时最接近 DRAM bound 负载",
        "窗口内功率与时钟大致稳定：抢占/降频会让'达成值'本身漂移",
        (f"达成值来自 NVML 代理（整数百分比、200 ms 区间平均）而非计数器读数"
         if pct is not None else "达成值尚不可得"),
    ]


def _issue_premises(pct: float | None) -> list[str]:
    return [
        "达成值是**上界口径**（每 SM 4 调度器 × 每周期 1 条）：真实利用率只会更低，"
        "所以这里的头寸是保守方向",
        "减少指令数会同时改变访存行为：把'发射槽头寸'与'DRAM 头寸'当两笔独立的钱是错的",
        "分母必须是**本机实测可达**的上界；拿硬件规格上限做分母会让头寸虚高",
    ]


# ---------------------------------------------------------------------------
# 排序与总计：本模块最容易撒谎的两处
# ---------------------------------------------------------------------------

def rank(entries: Sequence[Headroom]) -> list[Headroom]:
    """按"能省多少"排序，**不静默跨置信度混排**。

    排序键（降序）：
      ① 有没有数：连区间都没有的条目永远排最后（它们在排序上不参与"谁更值"）；
      ② 保守端金额（没有金额就用保守端比例）：这是"能省多少"的代理；
      ③ 置信度：measured 优于 estimated —— 同金额时先做我们更确定的那个；
      ④ 原顺序：稳定，保证同输入必然同输出。
    """
    def key(item: tuple[int, Headroom]) -> tuple:
        i, h = item
        has_number = 1 if h.has_number else 0
        weight = h.saving_per_hour if h.saving_per_hour is not None else (
            h.conservative_pct or 0.0)
        conf = 1 if h.confidence == CONF_MEASURED else 0
        return (-has_number, -float(weight), -conf, i)

    return [h for _, h in sorted(enumerate(entries), key=key)]


def _overlap_reason(a: Headroom, b: Headroom) -> str | None:
    """两个头寸是否可能重叠（重叠 = 同一份节省的两种说法）。"""
    if a.kernel == b.kernel:
        return f"同一个 kernel（{a.kernel}）上的两个头寸"
    if a.resource and b.resource and a.resource == b.resource:
        return f"不同 kernel（{a.kernel} / {b.kernel}）抢占同一条资源（{a.resource}）"
    return None


def _totals(entries: Sequence[Headroom]) -> tuple[float | None, list[str]]:
    """只把**成对不重叠**的条目相加；重叠的按最大者封顶，并说明为什么。

    为什么不能简单地全部相加：同一个 kernel 上"DRAM 头寸 30%"与"发射槽头寸 25%"
    乘的是**同一段 kernel 时间**，相加等于把同一段时间卖了两遍。
    跨 kernel 但抢同一条资源的两个头寸同理（它们不能同时都拿到 100% 的 DRAM）。
    跨 kernel 且资源不同的条目才可以相加 —— 那是两段不同的 GPU 时间。
    """
    money = [h for h in entries if h.has_money]
    if not money:
        return None, []
    by_kernel: dict[str, list[Headroom]] = {}
    for h in money:
        by_kernel.setdefault(h.kernel, []).append(h)
    disjoint: list[Headroom] = []
    capped: list[Headroom] = []
    for group in by_kernel.values():
        best = max(group, key=lambda h: float(h.saving_per_hour or 0.0))
        disjoint.append(best)
        capped.extend(h for h in group if h is not best)
    notes: list[str] = []
    if capped:
        raw = sum(float(h.saving_per_hour or 0.0) for h in money)
        kept = sum(float(h.saving_per_hour or 0.0) for h in disjoint)
        which = "；".join(f"{h.kernel}·{h.resource or '口径'}" for h in capped)
        notes.append(
            f"**重叠的节省没有相加**：{which} 与同 kernel 上的最大头寸重叠"
            f"（同一份节省的两种说法），按最大者封顶。"
            f"全部相加会得到 {raw:.4g}/h，那是把同一段时间卖了两遍；"
            f"这里的合计 {kept:.4g}/h 只含成对不重叠的条目")
    # 跨 kernel 但同资源的条目：它们**也**可能重叠（共享同一条 DRAM）。这里不封顶
    # （跨 kernel 的时间是两段不同的时间），但必须写出来 —— 否则读者会把合计当保证。
    for i, a in enumerate(disjoint):
        for b in disjoint[i + 1:]:
            reason = _overlap_reason(a, b)
            if reason and a.kernel != b.kernel:
                notes.append(
                    f"合计里 {a.kernel} 与 {b.kernel} 各自算了一份节省，但{reason}："
                    f"两者都按'独占该资源'估的上界，合起来看应当偏保守地理解")
    if len(disjoint) > 1:
        notes.append(
            f"合计 = {len(disjoint)} 条**成对不重叠**条目的保守端之和"
            f"（不同 kernel、不同资源轴；每条都按保守端计）")
    return sum(float(h.saving_per_hour or 0.0) for h in disjoint), notes


def _per_mtok_total(entries: Sequence[Headroom]) -> tuple[float | None, list[str]]:
    """per-token 合计：同样只加不重叠的条目（规则与金额一致，不另开一套口径）。"""
    vals = [h for h in entries if h.saving_per_mtok is not None and h.kernel]
    if not vals:
        return None, []
    by_kernel: dict[str, list[Headroom]] = {}
    for h in vals:
        by_kernel.setdefault(h.kernel, []).append(h)
    kept = [max(g, key=lambda h: float(h.saving_per_mtok or 0.0))
            for g in by_kernel.values()]
    notes: list[str] = []
    if len(kept) < len(vals):
        notes.append("per-token 合计同样只取每个 kernel 的最大头寸：重叠的 token 成本不能重复计")
    return sum(float(h.saving_per_mtok or 0.0) for h in kept), notes


def prescribe(entries: Iterable[Headroom] | Headroom, *,
              currency: str = "CNY") -> Prescription:
    """把若干头寸条目整理成一份**排序 + 合计合法**的处方。"""
    if isinstance(entries, Headroom):
        items = [entries]
    else:
        items = list(entries)
    ranked = rank(items)
    total_hour, notes_h = _totals(ranked)
    total_mtok, notes_t = _per_mtok_total(ranked)
    notes = notes_h + notes_t
    if not items:
        notes.append("没有任何头寸条目：处方为空（这是诚实答案 —— 缺的输入在 capabilities/拒绝里）")
    if items and not any(h.has_range for h in items):
        notes.append("没有任何条目能给出区间：本机标定或达成值至少缺一侧，"
                     "所以这份处方的结论是「先补测量」，不是「改代码」")
        # 一条区间都没有时**不许**给出合计（哪怕某条碰巧有金额）：
        # 合计是"这份处方一共能省多少"的承诺，没有区间就没有这个承诺。
        total_hour = None
        total_mtok = None
    # 有钱算不出来时，把条目层面的理由提升到整份处方：读者不必自己去翻每一条。
    money_refusals = [h.money_refusal for h in ranked if h.money_refusal is not None]
    money_ref = combined_refusal(money_refusals) if money_refusals else None
    if money_ref is not None and total_hour is None:
        notes.append(f"没有给出钱：{money_ref.code.value} —— {money_ref.message[:160]}")
    return Prescription(ranked=ranked, total_saving_per_hour=total_hour,
                        total_saving_per_mtok=total_mtok, notes=notes,
                        money_refusal=money_ref, currency=currency or "CNY")


def prescribe_or_refuse(entries: Iterable[Headroom], *, currency: str = "CNY") -> Prescription:
    """一份**什么都给不出**的处方 = `ranked == []` + 一条结构化拒绝（rule 7）。

    "没有可辩护的头寸"与"我们没算"是两件事，但都不该退化成空输出：
    读者必须看到"为什么没有"（缺价格？缺标定？没有达成值？）以及怎么补。
    """
    items = list(entries)
    pres = prescribe(items, currency=currency)
    if any(h.has_range for h in items):
        return pres
    refusals = [h.refusal for h in items if h.refusal is not None]
    ref = combined_refusal(refusals) if refusals else (
        pres.money_refusal or Refusal(
            RefusalCode.NO_WORKLOAD,
            "没有任何可辩护的头寸：既没有本机实测峰值（分母），也没有达成值（分子），"
            "或没有价格（乘数）",
            observed={"n_entries": len(items)}))
    pres.ranked = []
    pres.total_saving_per_hour = None
    pres.total_saving_per_mtok = None
    pres.refusal = ref
    pres.notes = pres.notes + [
        f"处方为空：给不出可辩护的头寸（{ref.code.value}）。"
        f"空的处方是诚实答案 —— 先按 remedy 补齐输入，而不是去改一个我们指不出来的东西"]
    return pres


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """kernel 名 → 指标名里可用的一段（只留字母数字下划线，避免指标名里有空格/冒号）。"""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in (name or "kernel"))
    return out.strip("_") or "kernel"
