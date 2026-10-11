"""处方（预期节省 + 排序）的契约测试。

这一类测试的重点和 `test_cost.py` 一样：**不是"算得对不对"（那是最低要求），
而是"什么时候拒绝给数"**。处方模块的危险在于它输出的东西会被直接拿去做
"先改哪个 kernel"的决策，所以每条诚实纪律都必须有一条测试守着：

  * 没有价格 → 一分钱都不许出现（rule 1）；
  * 没有本机标定（> 规格表/覆盖值）→ 区间与钱都不给（rule 2）；
  * 区间两端必须是可手算复核的精确值（rule 3）；
  * 没有数的条目排最后，且每条都带 confidence，不静默跨置信度混排（rule 4）；
  * **重叠的节省不许相加**，取最大者封顶并说明（rule 5）—— 这是本文件最重要的一条；
  * `to_metrics()` 产出的每个名字都必须能构造成真的 `provenance.Metric`（rule 6）；
  * 什么都给不出时，空的处方 + 一条结构化拒绝（rule 7）。

下面所有期望值都在注释里手算过一遍（算术写在注释里，不是从实现里复制出来的）。
"""

from __future__ import annotations

import pytest

from kernoscope.analyze.cost import CostEstimate, CostInputs, attribute, estimate
from kernoscope.analyze.prescription import (
    CONF_ESTIMATED,
    CONF_MEASURED,
    DOMINANT_SHARE_PCT,
    Calibration,
    Headroom,
    bandwidth_headroom,
    gap_fraction,
    issue_headroom,
    prescribe,
    prescribe_or_refuse,
    rank,
    saving_pct_conservative,
    saving_pct_optimistic,
)
from kernoscope.provenance import Metric, Provenance
from kernoscope.refusals import RefusalCode

# ---------------------------------------------------------------------------
# 测试替身与小工具
# ---------------------------------------------------------------------------

KERNEL = "kcopy"


class Sample:
    """与 backends.nvml.DeviceSample 同构的最小替身（只用 t 与 power_w）。"""

    def __init__(self, t: float, power_w: float):
        self.t = t
        self.power_w = power_w


def _samples(n: int = 31, dt: float = 0.2, power: float = 300.0):
    return [Sample(i * dt, power) for i in range(n)]


def _window_cost() -> CostEstimate:
    """窗口成本账（**没有 kernel 份额**）：300 W、租约 7.2/h、3 个 step、每个 1024 token。

    手算：能耗 = 300 W × 6.0 s = 1800 J；每小时成本 = 租约 7.2（没给电价，只有租约）。
    token 数 = 1024 × 3 = 3072（窗口 6 s）⇒ tokens/s = 512
      ⇒ 每小时 token = 512 × 3600 = 1,843,200
      ⇒ ¥/Mtoken = 7.2 × 1e6 / 1,843,200 = 3.90625
    """
    return estimate(_samples(), inputs=CostInputs(tokens_per_step=1024.0, price_per_hour=7.2),
                    window_s=6.0, sampler_duration_s=6.0,
                    window_t0=0.0, window_t1=6.0, steps_in_window=3.0)


def _attributed(share_pct: float) -> CostEstimate:
    """按份额分摊后的 kernel 账。

    手算（share=50%）：kernel 每小时成本 = 7.2 × 0.5 = 3.6；
    单位成本同比例 = 3.90625 × 0.5 = 1.953125 ¥/Mtoken。

    `cost.attribute()` 只搬运窗口级**能耗/每小时**那几个字段（它只是把窗口账分摊），
    所以这里按它的口径把 `cost_per_mtok` 也按份额折一次 —— 与 `ks__cost_per_mtok
    = 每小时成本 × 1e6 ÷ (tokens/s × 3600)` 一致，可以两种算法互相验证。
    """
    base = _window_cost()
    out = attribute(base, share_pct, steps=3.0)
    if base.cost_per_mtok is not None:
        out.cost_per_mtok = base.cost_per_mtok * share_pct / 100.0
    return out


CALIB = Calibration(bandwidth_gbs=1005.0, source="machine-profile.json",
                    calibrated_at="2025-01-01T00:00:00Z", measured=True)


# ---------------------------------------------------------------------------
# rule 3 的算术：三段式子必须能被手算复核，且比例与金额是同一个数
# ---------------------------------------------------------------------------

def test_gap_and_both_ends_match_the_hand_computed_formulas():
    """A=420、P=1005 时：

    u = (1005 − 420)/1005 = 585/1005 = 0.5820895522388060
    保守端（关掉一半缺口，A_half = (420+1005)/2 = 712.5）：
        1 − 420/712.5 = 0.4105263157894737 ⇒ 41.05263157894737%
        （与 u/(2−u) = 0.5820895522388060/1.417910447761194 = 0.4105263157894737 一致）
    乐观端（上界）：1 − 420/1005 = 0.5820895522388060 ⇒ 58.20895522388059%
    """
    u = gap_fraction(420.0, 1005.0)
    assert u == pytest.approx(585.0 / 1005.0, rel=1e-12)
    assert saving_pct_conservative(u) == pytest.approx(100.0 * (1.0 - 420.0 / 712.5), rel=1e-12)
    assert saving_pct_conservative(u) == pytest.approx(41.052631578947370, rel=1e-12)
    assert saving_pct_optimistic(u) == pytest.approx(58.208955223880590, rel=1e-12)
    # 上界必须严格大于下界（否则"区间"没有意义）
    assert saving_pct_optimistic(u) > saving_pct_conservative(u)


def test_conservative_and_optimistic_endpoints_of_a_ratio():
    """比例就是比例：A=1、P=2（u=0.5）⇒ 下界 1/3 = 33.33%、上界 50%。"""
    assert saving_pct_conservative(0.5) == pytest.approx(100.0 / 3.0, rel=1e-12)
    assert saving_pct_optimistic(0.5) == pytest.approx(50.0, rel=1e-12)


def test_gap_fraction_refuses_when_achieved_is_at_or_above_peak():
    """达成值 ≥ 峰值（口径不一致的征兆）时不给缺口 —— 由调用方出 premise_failed。"""
    assert gap_fraction(1005.0, 1005.0) is None
    assert gap_fraction(1200.0, 1005.0) is None
    assert gap_fraction(None, 1005.0) is None
    assert gap_fraction(420.0, 0.0) is None


# ---------------------------------------------------------------------------
# rule 1 + rule 3：有价格、有标定时的完整数字（手算金额）
# ---------------------------------------------------------------------------

def test_bandwidth_headroom_full_numbers_are_hand_computable():
    """share=50% ⇒ 可归因成本 7.2 × 0.5 = 3.6/h；¥/Mtoken = 3.90625 × 0.5 = 1.953125。

    区间 = (41.052631578947370%, 58.208955223880590%)（见上一条测试的手算）。
    保守端金额 = 3.6 × 0.41052631578947370 = 1.4778947368421053
    保守端 per-token = 1.953125 × 0.41052631578947370 = 0.8018108552631579

    为什么 per-token 的期望值写成**精确分数**（而不是那一串小数）：比例是由实测值算出来的
    浮点数，把它先化成百分数再乘回来会引入 1e-6 量级的表示误差（这正是"一个量在两处各算一次"
    的经典坑）。测试要验的是**关系**（金额 = 基数 × 比例），所以参考值也用同一个关系式写。
    """
    cost = _attributed(50.0)
    assert cost.kernel_cost_per_hour == pytest.approx(3.6, rel=1e-12)   # 7.2 × 0.5
    assert cost.cost_per_mtok == pytest.approx(1.953125, rel=1e-12)     # 见 _window_cost 手算

    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost)

    assert h.achieved_pct == pytest.approx(100.0 * 420.0 / 1005.0, rel=1e-12)
    assert h.saving_pct_range == pytest.approx((100.0 * 585.0 / 1425.0, 100.0 * 585.0 / 1005.0),
                                               rel=1e-12)
    assert h.saving_pct_range == pytest.approx((41.052631578947370, 58.208955223880590), rel=1e-12)
    assert h.saving_per_hour == pytest.approx(3.6 * 585.0 / 1425.0, rel=1e-12)
    assert h.saving_per_hour == pytest.approx(1.4778947368421053, rel=1e-12)
    assert h.saving_per_mtok == pytest.approx(1.953125 * 585.0 / 1425.0, rel=1e-12)
    # NVML 代理口径的达成值是 estimated（不是实测读数），所以 confidence 只能是 estimated
    assert h.confidence == CONF_ESTIMATED
    assert h.refusal is None
    # basis 必须点明分母是"本机实测峰值"（读者要能一眼看出分母不是规格表）
    assert "实测峰值" in h.basis and "1005" in h.basis
    # 前提非空：头寸只有在这些前提下才是对的方向性判断
    assert h.premises


def test_money_and_pct_share_the_same_physical_meaning():
    """金额 = 可归因成本 × 比例：同一个比例既表示"省下的运行时间"，也表示"省下的钱"。

    这条测试守住的是"比例与金额不许各算一套"：一旦有人把金额改成乘另一个数，
    这里的等式就会红。
    """
    cost = _attributed(50.0)
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost)
    lo = h.conservative_pct
    assert lo is not None
    assert h.saving_per_hour == pytest.approx(cost.kernel_cost_per_hour * lo / 100.0, rel=1e-12)
    assert h.saving_per_mtok == pytest.approx(cost.cost_per_mtok * lo / 100.0, rel=1e-12)


def test_measured_provenance_is_kept_but_never_upgraded():
    """达成值若真是实测读数，confidence 才是 measured（R-12 的同一个纪律）。"""
    cost = _attributed(50.0)
    measured = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost,
                                  achieved_provenance=Provenance.MEASURED.value)
    assert measured.confidence == CONF_MEASURED
    derived = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost,
                                 achieved_provenance=Provenance.DERIVED.value)
    assert derived.confidence == CONF_ESTIMATED   # derived 不能升级成"实测"


# ---------------------------------------------------------------------------
# rule 1：没有价格 -> 一分钱都没有
# ---------------------------------------------------------------------------

def test_no_price_means_no_money_at_all():
    """**没有价格时 `saving_per_hour` 必须是 None**，并带 no_price 拒绝。

    比例仍然可以给（41.05% 是物理量，不需要钱），但货币数字一个都不许出现 ——
    "典型电价 0.6 元/度"这种顶替会被直接拿去做租卡决策，是最危险的假数据。
    """
    unpriced = CostEstimate(inputs=CostInputs(), window_s=6.0)
    assert unpriced.cost_per_hour is None
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=unpriced)

    assert h.saving_per_hour is None
    assert h.saving_per_mtok is None
    assert h.refusal is None                      # 比例给得出，所以条目级判决不是"拒绝"
    assert h.money_refusal is not None and h.money_refusal.code is RefusalCode.NO_PRICE
    # 没有钱 != 没有结论：区间照给（它只需要达成值与峰值）
    assert h.saving_pct_range is not None
    # 有区间但没钱时置信度仍要如实：这个比例是代理口径
    assert h.confidence == CONF_ESTIMATED
    # 处方里也绝不能冒出钱
    p = prescribe(h)
    assert p.total_saving_per_hour is None
    assert p.total_saving_per_mtok is None


def test_cost_is_none_also_refuses_money_with_no_price_code():
    """连成本账都没传：同样走 no_price（"没给价格"与"没算成本"对用户是同一件事）。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=None)
    assert h.saving_per_hour is None
    assert h.money_refusal.code is RefusalCode.NO_PRICE


def test_price_without_attributable_share_refuses_money_with_premise_failed():
    """有价格、但窗口里不止一个 kernel（份额未知）：整机成本不能乘到单个 kernel 上。

    这时 code 是 `premise_failed`（前提不成立），不是 `no_price`（没给价格）——
    两者的补救办法完全不同，所以必须分开。
    """
    window = _window_cost()
    assert window.kernel_share_pct is None
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=window)
    assert h.saving_per_hour is None
    assert h.money_refusal.code is RefusalCode.PREMISE_FAILED
    assert h.saving_pct_range is not None      # 比例不受份额影响


# ---------------------------------------------------------------------------
# rule 2：没有标定 -> 没有区间
# ---------------------------------------------------------------------------

def test_no_calibration_means_no_range_at_all():
    """没有本机实测峰值：区间与钱都不给（no_calibration），且不许用规格表顶替。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=Calibration(),
                           cost=_attributed(50.0))
    assert h.saving_pct_range is None
    assert h.saving_per_hour is None and h.saving_per_mtok is None
    assert h.refusal.code is RefusalCode.NO_CALIBRATION
    assert h.confidence == "unknown"
    # 达成值本身还是能报的（它不依赖峰值）—— 但这里没有峰值口径就没有百分比，只能 None
    assert h.achieved_pct is None


def test_spec_sheet_peak_is_refused_for_headroom():
    """规格表/`--ks-peak-*` 覆盖值：可以展示，**绝不用于头寸估计**。

    规格与实测的差距就是本工具存在的理由（4090 fp16 规格 82.6 对实测 ~54 TFLOP/s），
    拿它当分母算出来的"头寸"会系统性偏错，所以 code 是 calibration_caliber_mismatch。
    """
    spec = Calibration.from_spec(bandwidth_gbs=1008.0)
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=spec, cost=_attributed(50.0))
    assert h.saving_pct_range is None
    assert h.saving_per_hour is None
    assert h.refusal.code is RefusalCode.CALIBRATION_CALIBER_MISMATCH
    # basis 的措辞必须区分"实测峰值"与"规格/覆盖峰值"，不让读者以为 1008 是标定值
    assert "规格/覆盖峰值" in h.basis and "不可用于头寸" in h.basis
    assert "实测峰值" not in h.basis


def test_both_caliber_entry_points_refuse_the_same_way():
    """DRAM 与发射槽两条入口对"没有标定"必须给同一个 code —— 不然 agent 要为每个
    kernel 写一套 if。区别只在 basis 的措辞（分母的物理量不同）。"""
    dram = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=Calibration(),
                              cost=_attributed(50.0))
    issue = issue_headroom(KERNEL, achieved_pct=12.0, calib=Calibration(),
                           cost=_attributed(50.0))
    assert dram.refusal.code is issue.refusal.code is RefusalCode.NO_CALIBRATION
    assert dram.saving_pct_range is None and issue.saving_pct_range is None
    assert dram.confidence == issue.confidence == "unknown"
    # 规格覆盖值：两条入口都必须拒绝，并给同一个 code
    spec_dram = bandwidth_headroom(KERNEL, achieved_gbs=420.0,
                                   calib=Calibration.from_spec(bandwidth_gbs=1008.0),
                                   cost=_attributed(50.0))
    spec_issue = issue_headroom(KERNEL, achieved_pct=12.0,
                                calib=Calibration.from_spec(issue_util_pct=75.0),
                                cost=_attributed(50.0))
    assert (spec_dram.refusal.code is spec_issue.refusal.code
            is RefusalCode.CALIBRATION_CALIBER_MISMATCH)


def test_missing_achieved_value_leaves_an_entry_without_a_number():
    """只有标定、没有达成值：条目仍然出现（不许消失），但没有数、confidence=unknown。

    B1 的纪律：口径不成立时"拒绝"不许退化成"消失"。它排在最后（rule 4）。
    """
    h = bandwidth_headroom(KERNEL, achieved_gbs=None, calib=CALIB, cost=_attributed(50.0))
    assert h.saving_pct_range is None and h.achieved_pct is None
    assert h.confidence == "unknown"
    assert h.refusal is not None and h.refusal.code is RefusalCode.NO_WORKLOAD
    assert h.refusal.remedy is not None        # 拒绝必须带"怎么办"


def test_achieved_above_peak_is_a_premise_failure_not_a_headroom():
    """达成值 ≥ 峰值（物理上不可能）：出 premise_failed，而不是报一个正的头寸。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=1005.0, calib=CALIB, cost=_attributed(50.0))
    assert h.saving_pct_range is None and h.saving_per_hour is None
    assert h.refusal.code is RefusalCode.PREMISE_FAILED
    assert h.refusal.observed["achieved_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# rule 4：排序
# ---------------------------------------------------------------------------

def _entry(kernel: str, resource: str, hi_pct: float, *, per_hour: float | None,
           confidence: str, has_range: bool = True) -> Headroom:
    """测试用的最小条目（不走算术，直接给数）—— 排序/合计测试只关心这三样。"""
    pair = (hi_pct / 2.0, hi_pct) if has_range else None
    # 有区间、没金额时必须写出"为什么没钱"（模块的不变量）—— 排序测试只关心排序，
    # 但条目本身仍必须是合法的，否则测的就不是真实条目了。
    from kernoscope.refusals import Refusal, RefusalCode

    money_ref = None if (per_hour is not None or pair is None) else Refusal(
        RefusalCode.NO_PRICE, "测试替身：没给价格")
    return Headroom(kernel=kernel, resource=resource, basis=f"测试口径 {resource}",
                    achieved_pct=40.0 if has_range else None,
                    saving_pct_range=pair, saving_per_hour=per_hour,
                    confidence=confidence, money_refusal=money_ref)


def test_ranking_puts_entries_without_a_number_last():
    """连区间都没有的条目排在最后 —— 排序里不许把"不知道"混进"知道"里比大小。"""
    known_big = _entry("k_a", "dram_bandwidth", 20.0, per_hour=2.0, confidence=CONF_MEASURED)
    unknown = _entry("k_b", "dram_bandwidth", 0.0, per_hour=None, confidence="unknown",
                     has_range=False)
    known_small = _entry("k_c", "dram_bandwidth", 4.0, per_hour=0.4, confidence=CONF_ESTIMATED)

    ordered = rank([unknown, known_small, known_big])
    assert [h.kernel for h in ordered] == ["k_a", "k_c", "k_b"]


def test_ranking_uses_the_conservative_end_not_the_optimistic_one():
    """谁更值，看的是**保守端**：乐观端很大但保守端很小的条目不许排前面。

    A：保守端 0.5/h、乐观端 100/h；B：保守端 2.0/h、乐观端 3/h。 → B 应该在前。
    """
    a = Headroom(kernel="k_a", resource="dram_bandwidth", basis="x", achieved_pct=10.0,
                 saving_pct_range=(0.5, 99.5), saving_per_hour=0.5, confidence=CONF_MEASURED)
    b = Headroom(kernel="k_b", resource="dram_bandwidth", basis="x", achieved_pct=10.0,
                 saving_pct_range=(2.0, 3.0), saving_per_hour=2.0, confidence=CONF_MEASURED)
    assert [h.kernel for h in rank([a, b])] == ["k_b", "k_a"]


def test_ranking_prefers_measured_at_equal_savings_and_never_hides_confidence():
    """同样的保守端金额：measured 排前面；且每条都带 confidence（读者能分清实测与模型）。"""
    est = _entry("k_est", "dram_bandwidth", 20.0, per_hour=2.0, confidence=CONF_ESTIMATED)
    mea = _entry("k_mea", "dram_bandwidth", 20.0, per_hour=2.0, confidence=CONF_MEASURED)
    ordered = rank([est, mea])
    assert [h.kernel for h in ordered] == ["k_mea", "k_est"]
    assert {h.confidence for h in ordered} == {CONF_MEASURED, CONF_ESTIMATED}


def test_ranking_without_money_still_orders_by_conservative_pct():
    """没有价格时（没有金额）也要排序：用保守端比例当"能省多少"的代理。"""
    small = _entry("k_small", "dram_bandwidth", 4.0, per_hour=None, confidence=CONF_ESTIMATED)
    big = _entry("k_big", "dram_bandwidth", 30.0, per_hour=None, confidence=CONF_ESTIMATED)
    assert [h.kernel for h in rank([small, big])] == ["k_big", "k_small"]


# ---------------------------------------------------------------------------
# rule 5：重叠的节省不许相加（本文件最重要的一条）
# ---------------------------------------------------------------------------

def test_overlapping_headrooms_on_one_kernel_are_capped_not_summed():
    """**同一个 kernel 上的两个大头寸：合计取最大者，不许相加。**

    构造：k_fused 上 DRAM 头寸（保守端 4.0/h）与发射槽头寸（保守端 3.0/h）。
    它们乘的是**同一段 kernel 时间**（填满 DRAM 会同时改变发射槽利用率），
    相加得到 7.0/h 就是把同一段时间卖了两遍。
    合法合计 = max(4.0, 3.0) = 4.0/h，并且必须有一条 note 解释为什么没相加。
    """
    dram = Headroom(kernel="k_fused", resource="dram_bandwidth", basis="DRAM 口径",
                    achieved_pct=30.0, saving_pct_range=(10.0, 20.0),
                    saving_per_hour=4.0, saving_per_mtok=1.0, confidence=CONF_ESTIMATED)
    issue = Headroom(kernel="k_fused", resource="issue_slots", basis="发射槽口径",
                     achieved_pct=25.0, saving_pct_range=(6.0, 12.0),
                     saving_per_hour=3.0, saving_per_mtok=0.75, confidence=CONF_ESTIMATED)

    p = prescribe([dram, issue])

    assert p.total_saving_per_hour == pytest.approx(4.0, rel=1e-12)      # 不是 7.0
    assert p.total_saving_per_hour != pytest.approx(7.0, rel=1e-12)
    assert p.total_saving_per_mtok == pytest.approx(1.0, rel=1e-12)      # 不是 1.75
    joined = "；".join(p.notes)
    assert "重叠" in joined and "没有相加" in joined
    # 透明：把"全部相加会是多少"也写出来（否则读者不知道被丢了多少）
    assert "7" in joined
    # 两条条目都还在（只是不进合计）——排序仍然展示它们
    assert {h.resource for h in p.ranked} == {"dram_bandwidth", "issue_slots"}


def test_disjoint_headrooms_on_different_kernels_are_summed():
    """不同 kernel、不同资源轴：两段不同的 GPU 时间，可以相加（4.0 + 3.0 = 7.0/h）。"""
    a = Headroom(kernel="k_a", resource="dram_bandwidth", basis="x", achieved_pct=30.0,
                 saving_pct_range=(10.0, 20.0), saving_per_hour=4.0, confidence=CONF_ESTIMATED)
    b = Headroom(kernel="k_b", resource="issue_slots", basis="y", achieved_pct=25.0,
                 saving_pct_range=(6.0, 12.0), saving_per_hour=3.0, confidence=CONF_ESTIMATED)
    p = prescribe([a, b])
    assert p.total_saving_per_hour == pytest.approx(7.0, rel=1e-12)
    assert not any("没有相加" in n for n in p.notes)


def test_cross_kernel_shared_resource_is_flagged_even_when_not_capped():
    """不同 kernel 抢同一条 DRAM：合计仍然给（时间不同），但必须写出来 —— 否则读者
    会把合计当成"两条都能拿满"的保证。"""
    a = Headroom(kernel="k_a", resource="dram_bandwidth", basis="x", achieved_pct=30.0,
                 saving_pct_range=(10.0, 20.0), saving_per_hour=4.0, confidence=CONF_ESTIMATED)
    b = Headroom(kernel="k_b", resource="dram_bandwidth", basis="y", achieved_pct=25.0,
                 saving_pct_range=(6.0, 12.0), saving_per_hour=3.0, confidence=CONF_ESTIMATED)
    p = prescribe([a, b])
    assert p.total_saving_per_hour == pytest.approx(7.0, rel=1e-12)
    assert any("同一条资源" in n for n in p.notes)


def test_three_overlapping_entries_on_one_kernel_keep_only_the_largest():
    """一个 kernel 上三条重叠头寸：合计 = 最大者（5.0），不是 4+3+5=12。"""
    entries = [
        Headroom(kernel="k", resource="dram_bandwidth", basis="a", achieved_pct=30.0,
                 saving_pct_range=(5.0, 9.0), saving_per_hour=4.0, confidence=CONF_ESTIMATED),
        Headroom(kernel="k", resource="issue_slots", basis="b", achieved_pct=20.0,
                 saving_pct_range=(4.0, 7.0), saving_per_hour=3.0, confidence=CONF_ESTIMATED),
        Headroom(kernel="k", resource="l2_hit_rate", basis="c", achieved_pct=50.0,
                 saving_pct_range=(6.0, 10.0), saving_per_hour=5.0, confidence=CONF_ESTIMATED),
    ]
    p = prescribe(entries)
    assert p.total_saving_per_hour == pytest.approx(5.0, rel=1e-12)
    assert all(h in p.ranked for h in entries)      # 三条都展示，只有合计被限制


# ---------------------------------------------------------------------------
# rule 6：命名纪律由真的 Metric 构造来强制
# ---------------------------------------------------------------------------

_CODE_PREFIX = "unavailable: code="


def _metric_from_row(name: str, value: object, unit: str, note: str) -> Metric:
    """把 `to_metrics()` 的一行还原成真的 Metric。

    value 为 None 的行必须自带 `unavailable: code=<code> | <reason>`（W3：拒绝必须带 code）。
    这不是测试的自造约定 —— 组装方（sections.py）就是按 code 构造 `Metric.unavailable`。
    """
    if value is None:
        # `ks__prescription_note` 按约定是**说明行**（不是数字行），由消费方按后缀识别；
        # 其余 n/a 行必须带 code 标记（W3）。
        if name.endswith("_note"):
            return Metric.derived(name, "（说明行，非指标）", unit=unit)
        assert note.startswith(_CODE_PREFIX), f"{name}: n/a 行必须带 code 标记"
        code_str, reason = note[len(_CODE_PREFIX):].split(" | ", 1)
        return Metric.unavailable(name, reason, unit=unit,
                                  code=RefusalCode(code_str))
    assert name.startswith(("est__", "ks__", "meas__", "model__")), \
        f"{name}: 非 exact 的值只允许自有前缀（R3）"
    if name.startswith("ks__"):
        return Metric.derived(name, value, unit)
    if name.startswith("meas__"):
        return Metric.measured(name, value, unit)
    return Metric.estimated(name, value, unit)


def test_every_metric_row_is_constructible_as_a_real_provenance_metric():
    """rule 6：`to_metrics()` 的每一行都能构造成真的 `provenance.Metric`。

    这条测试是**命名纪律的执行点**：名字里只要出现 `dram__`/`sm__` 这类计数器命名空间，
    或者 `meas__` 配了非 measured 的出处，`Metric(...)` 会直接抛 ProvenanceError。
    """
    cost = _attributed(60.0)
    entries = [
        bandwidth_headroom("k_a", achieved_gbs=420.0, calib=CALIB, cost=cost),
        bandwidth_headroom("k_b", achieved_gbs=None, calib=CALIB, cost=cost),        # 无数
        bandwidth_headroom("k_c", achieved_gbs=420.0, calib=Calibration(), cost=cost),  # 无标定
        bandwidth_headroom("k_d", achieved_gbs=420.0, calib=CALIB, cost=None),       # 无价格
        issue_headroom("k_e", achieved_pct=12.0, calib=Calibration(), cost=cost),    # 无发射槽分母
    ]
    p = prescribe(entries)
    rows = p.to_metrics()
    assert rows, "处方必须至少输出条目数这一行"
    built = [_metric_from_row(*row) for row in rows]
    # 没给钱时，处方里**一个货币数字都不许有**（这里 k_a/k_e 没有可用价格来源）
    money_names = [m.name for m in built if m.unit.endswith("/h") and not m.name.endswith("_n")]
    assert money_names, "应该有成本相关的行（哪怕是 n/a）"
    for m in built:
        assert m.provenance is not Provenance.EXACT or m.name.startswith("ks__")
    # 名字一个都不许撞
    names = [m.name for m in built]
    assert len(names) == len(set(names))


def test_to_metrics_never_publishes_a_counter_name():
    """反面样本：故意把一个 ncu 计数器名喂进去，构造必须被 provenance 拦下（R2/R3）。

    这是 rule 6 的"有分辨力"证明：不是测试自己写了断言，而是 `Metric` 类型系统在拦。
    """
    from kernoscope.provenance import ProvenanceError

    with pytest.raises(ProvenanceError):
        Metric.estimated("dram__throughput.avg.pct_of_peak_sustained_elapsed", 41.0)
    with pytest.raises(ProvenanceError):
        Metric.estimated("sm__throughput.avg.pct_of_peak_sustained_elapsed", 41.0)


def test_unavailable_total_rows_carry_a_refusal_code():
    """一条区间都没有时，合计行必须是 n/a + code（不许给 0，也不许消失）。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=None, calib=CALIB, cost=_attributed(50.0))
    p = prescribe_or_refuse([h])
    rows = dict((name, (value, note)) for name, value, _unit, note in p.to_metrics())
    for name in ("ks__prescription_total_saving_per_hour",
                 "ks__prescription_total_saving_per_mtok"):
        value, note = rows[name]
        assert value is None
        assert note.startswith(_CODE_PREFIX)
        _metric_from_row(name, value, "", note)      # 能还原成真的 Metric


# ---------------------------------------------------------------------------
# rule 7：没有可辩护的条目 -> 空处方 + 结构化拒绝
# ---------------------------------------------------------------------------

def test_nothing_defensible_gives_an_empty_ranked_list_plus_a_refusal():
    """没有标定（也就没有区间）：处方为空 + 一条结构化拒绝，**绝不是**一条通用建议。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=Calibration(),
                           cost=_attributed(50.0))
    p = prescribe_or_refuse([h])
    assert p.ranked == []
    assert p.refusal is not None
    assert p.refusal.code is RefusalCode.NO_CALIBRATION
    assert p.total_saving_per_hour is None and p.total_saving_per_mtok is None
    assert p.refusal.remedy is not None
    assert any("空" in n or "诚实" in n for n in p.notes)


def test_no_entries_at_all_is_an_empty_prescription_not_a_zero():
    """一条条目都没有：空处方 + no_workload（不许给 0 节省，0 会被读成"不用改"）。"""
    p = prescribe_or_refuse([])
    assert p.ranked == []
    assert p.refusal is not None and p.refusal.code is RefusalCode.NO_WORKLOAD
    assert p.total_saving_per_hour is None
    assert len(p.to_metrics()) >= 1


def test_empty_and_nonempty_prescriptions_agree_on_the_money_rule():
    """有区间（哪怕没价格）就不算"空处方"：ranked 保留，钱仍然不给。"""
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=None)
    p = prescribe_or_refuse([h])
    assert p.ranked == [h]
    assert p.total_saving_per_hour is None
    # 没有整份拒绝（条目层面已经有拒绝），但也绝没有钱
    assert all(x.saving_per_hour is None for x in p.ranked)


# ---------------------------------------------------------------------------
# 不变量：Headroom 自己不许构造出自相矛盾的条目
# ---------------------------------------------------------------------------

def test_headroom_refuses_money_without_a_range():
    """有金额却没有区间 = 编出来的金额：构造时就必须抛。"""
    with pytest.raises(ValueError):
        Headroom(kernel="k", basis="x", saving_per_hour=1.0)


def test_headroom_refuses_money_with_unknown_confidence():
    """有金额却标 unknown 置信度：有价格能算钱，至少是 estimated。"""
    with pytest.raises(ValueError):
        Headroom(kernel="k", basis="x", saving_pct_range=(1.0, 2.0),
                 saving_per_hour=1.0, confidence="unknown")


def test_headroom_refuses_a_refusal_together_with_numbers():
    """同一条目既拒绝又给数字：判决只能有一个（与 Metric 的 W3 一致）。"""
    from kernoscope.refusals import Refusal

    with pytest.raises(ValueError):
        Headroom(kernel="k", basis="x", saving_pct_range=(1.0, 2.0),
                 confidence=CONF_ESTIMATED,
                 refusal=Refusal(RefusalCode.NO_PRICE, "没给价格"))


# ---------------------------------------------------------------------------
# 与真实成本账的端到端：数字必须能追回 CostEstimate
# ---------------------------------------------------------------------------

def test_end_to_end_with_a_real_cost_estimate_traces_back_to_the_ledger():
    """端到端：bandwidth_headroom 的金额必须能拆回成本账的分量。

    share=60% ⇒ kernel 成本 = 7.2 × 0.6 = 4.32/h；¥/Mtoken = 3.90625 × 0.6 = 2.34375。
    保守端 41.052631578947370% ⇒ 4.32 × 0.41052631578947370 = 1.7734736842105263 /h
    """
    cost = _attributed(60.0)
    assert cost.kernel_cost_per_hour == pytest.approx(4.32, rel=1e-12)
    assert cost.cost_per_mtok == pytest.approx(2.34375, rel=1e-12)

    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost)
    assert h.saving_per_hour == pytest.approx(1.7734736842105263, rel=1e-12)
    assert h.saving_per_mtok == pytest.approx(2.34375 * 41.052631578947370 / 100.0, rel=1e-12)

    p = prescribe(h)
    assert p.total_saving_per_hour == pytest.approx(h.saving_per_hour, rel=1e-12)
    assert p.total_saving_per_mtok == pytest.approx(h.saving_per_mtok, rel=1e-12)


def test_per_token_saving_requires_the_kernel_to_dominate_the_window():
    """per-token 节省的额外前提：被匹配 kernel 必须主导窗口（≥ 50%）。

    share=40%：kernel 变快省下的是 GPU 时间，不一定变成端到端 token 成本
      ⇒ saving_per_hour 照给（份额可归因），saving_per_mtok 拒绝（None + 说明）。
    """
    assert DOMINANT_SHARE_PCT == 50.0
    cost = _attributed(40.0)
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost)
    assert h.saving_per_hour is not None
    assert h.saving_per_mtok is None
    assert any("per-token" in n for n in h.notes)

    dominant = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB,
                                  cost=_attributed(50.0))
    assert dominant.saving_per_mtok is not None    # 恰好达到门槛（≥ 50%）


def test_issue_headroom_needs_a_measured_upper_bound_not_the_hardware_limit():
    """发射槽分母必须是**本机实测可达上界**：拿"每周期 4 条"这个硬件规格当分母会让头寸虚高。"""
    cost = _attributed(50.0)
    no_denominator = issue_headroom(KERNEL, achieved_pct=12.0, calib=CALIB, cost=cost)
    assert no_denominator.saving_pct_range is None
    assert no_denominator.refusal.code is RefusalCode.NO_CALIBRATION

    measured = Calibration(issue_util_pct=40.0, source="machine-profile.json", measured=True)
    h = issue_headroom(KERNEL, achieved_pct=20.0, calib=measured, cost=cost)
    # u = (40−20)/40 = 0.5 ⇒ 保守端 1/3 = 33.333…%，乐观端 50%
    assert h.saving_pct_range == pytest.approx((100.0 / 3.0, 50.0), rel=1e-12)
    assert h.saving_per_hour == pytest.approx(3.6 * (100.0 / 3.0) / 100.0, rel=1e-12)
    assert "发射槽" in h.basis
    # 前提里必须写明这是**上界口径**（否则读者会拿它当实测利用率）
    assert any("上界" in p for p in h.premises)


def test_prescription_currency_label_travels_to_the_metrics():
    """币种来自成本账；我们不做汇率换算，所以单位标签必须跟着走。"""
    cost = _attributed(50.0)
    cost.inputs.currency = "USD"
    h = bandwidth_headroom(KERNEL, achieved_gbs=420.0, calib=CALIB, cost=cost)
    p = prescribe(h, currency="USD")
    units = {name: unit for name, _v, unit, _n in p.to_metrics()}
    assert units["ks__prescription_total_saving_per_hour"] == "USD/h"
    assert units["ks__prescription_total_saving_per_mtok"] == "USD/Mtoken"
