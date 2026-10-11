"""成本账（C1）的契约测试。

这一类测试的重点不是"算得对"（那是最低要求），而是**给不出数时必须拒绝**：
成本账最容易出的错是"输入不全却硬凑一个数" —— 那比没有数字更危险，因为它会直接被
拿去做"改哪个 kernel 值不值"的决策。
"""

from __future__ import annotations

import math

import pytest

from kernoscope.analyze.cost import (CostInputs, attribute, estimate, integrate_power)
from kernoscope.refusals import RefusalCode


class Sample:
    """与 backends.nvml.DeviceSample 同构的最小替身（只用到 t 与 power_w）。"""

    def __init__(self, t: float, power_w: float):
        self.t = t
        self.power_w = power_w


def _samples(n: int = 6, t0: float = 0.0, dt: float = 0.2, power: float = 300.0):
    return [Sample(t0 + i * dt, power) for i in range(n)]


# --------------------------------------------------------------------------
# 能耗积分
# --------------------------------------------------------------------------
def test_power_integration_is_exact_for_constant_power():
    """恒定功率 300 W × 1.0 s = 300 J（梯形法与解析解一致）。"""
    got = integrate_power(_samples(), 0.0, 1.2)
    assert got is not None
    energy, span = got
    assert energy == pytest.approx(300.0, rel=1e-9)
    assert span == pytest.approx(1.0, rel=1e-9)


def test_power_integration_uses_trapezoid_not_step_sum():
    """变功率时梯形法与"取中值×时长"必须不同 —— 否则这条测试没有分辨力。"""
    samples = [Sample(0.0, 100.0), Sample(1.0, 300.0), Sample(2.0, 100.0)]
    energy, _ = integrate_power(samples, 0.0, 2.0)
    assert energy == pytest.approx(400.0, rel=1e-9)          # 100+300 两个梯形
    naive = 100.0 * 2.0                                       # 首点×时长 = 200
    assert energy != pytest.approx(naive)


def test_power_integration_excludes_points_outside_the_window():
    """窗口外的采样点不参与积分（窗口口径与 DRAM 账共用，绝不用整段 wall 顶替）。"""
    samples = _samples(n=10, power=300.0)                     # t = 0 .. 1.8
    energy, span = integrate_power(samples, 0.2, 0.8)
    assert energy == pytest.approx(300.0 * 0.6, rel=1e-9)
    assert span == pytest.approx(0.6, rel=1e-9)


# --------------------------------------------------------------------------
# 输入不全 -> 拒绝（这是本模块的核心纪律）
# --------------------------------------------------------------------------
def test_missing_tokens_refuses_unit_cost_but_still_prices_the_hour():
    """给价格不给 token 数：每小时成本照给，单位成本被拒（no_tokens），不猜典型值。"""
    est = estimate(_samples(), inputs=CostInputs(price_per_kwh=1.0),
                   window_s=1.0, sampler_duration_s=1.0,
                   window_t0=0.0, window_t1=1.0, steps_in_window=10.0)
    assert est.tokens_per_s is None and est.cost_per_mtok is None
    assert est.tokens_refusal is not None
    assert est.tokens_refusal.code is RefusalCode.NO_TOKENS
    # 每小时成本只需要功率与电价：它必须给出来（"给一半输入拿一半结论"）
    assert est.cost_per_hour == pytest.approx(300.0 * 3600 / 3.6e6 * 1.0, rel=1e-9)


def test_missing_price_refuses_money_but_keeps_energy_and_tokens():
    """给 token 不给价格：能耗与吞吐是实测/推导量，照给；钱被拒（no_price）。"""
    est = estimate(_samples(), inputs=CostInputs(tokens_per_step=1000.0),
                   window_s=1.0, sampler_duration_s=1.0,
                   window_t0=0.0, window_t1=1.0, steps_in_window=10.0)
    assert est.window_energy_j == pytest.approx(300.0, rel=1e-9)
    assert est.tokens_per_s == pytest.approx(10_000.0, rel=1e-9)
    assert est.j_per_token == pytest.approx(0.03, rel=1e-9)
    assert est.cost_per_mtok is None and est.cost_per_hour is None
    assert est.price_refusal is not None
    assert est.price_refusal.code is RefusalCode.NO_PRICE


def test_short_window_refuses_energy_instead_of_extrapolating():
    """窗口短于采样周期量级：拒绝给能耗（不外推、不按 TDP 顶替）。"""
    est = estimate(_samples(), inputs=CostInputs(tokens_per_step=100.0),
                   window_s=0.1, sampler_duration_s=1.0,
                   window_t0=0.0, window_t1=0.1, steps_in_window=1.0)
    assert est.window_energy_j is None
    assert est.energy_refusal is not None
    assert est.energy_refusal.code is RefusalCode.BELOW_RESOLUTION
    assert est.energy_refusal.threshold["window_s_min"] == 0.5
    # 能耗拿不到 => 依赖它的量与钱一并不能给
    assert est.tokens_per_s is None and est.cost_per_mtok is None
    assert est.tokens_refusal is est.energy_refusal


def test_no_power_samples_refuses_energy_with_no_energy_code():
    est = estimate([], inputs=CostInputs(tokens_per_step=100.0, price_per_kwh=1.0),
                   window_s=1.0, sampler_duration_s=1.0)
    assert est.window_energy_j is None
    assert est.energy_refusal.code is RefusalCode.NO_ENERGY


def test_unattributable_launch_count_refuses_unit_cost():
    """窗口内 launch 数对不上（a 与 b 都算不出）时，分母不成立 -> 拒绝单位成本。"""
    est = estimate(_samples(), inputs=CostInputs(tokens_per_step=1000.0, price_per_kwh=1.0),
                   window_s=1.0, sampler_duration_s=1.0,
                   window_t0=0.0, window_t1=1.0, steps_in_window=None)
    assert est.tokens_per_s is None
    assert est.tokens_refusal.code is RefusalCode.WINDOW_LAUNCH_UNALIGNED


# --------------------------------------------------------------------------
# 两个独立路径算同一个数（自校验）
# --------------------------------------------------------------------------
def test_cost_per_mtok_matches_its_two_definitions():
    """`¥/Mtoken` = 每小时成本 × 1e6 / (tokens/s × 3600) —— 用两条路径独立算，必须相等。"""
    est = estimate(_samples(power=200.0), inputs=CostInputs(
        tokens_per_step=4096.0, price_per_kwh=0.6, price_per_hour=3.0),
        window_s=2.0, sampler_duration_s=2.0,
        window_t0=0.0, window_t1=2.0, steps_in_window=5.0)
    assert est.tokens_per_s is not None and est.cost_per_mtok is not None
    via_hour = est.cost_per_hour * 1e6 / (est.tokens_per_s * 3600.0)
    assert est.cost_per_mtok == pytest.approx(via_hour, rel=1e-12)
    # 两种载体分开报，且分量之和 = 总量（否则"单位成本"无法拆回来源）
    assert est.cost_per_mtok == pytest.approx(
        est.cost_per_mtok_energy + est.cost_per_mtok_rental, rel=1e-9)
    assert est.cost_per_hour == pytest.approx(
        est.cost_per_hour_energy + est.cost_per_hour_rental, rel=1e-9)


def test_energy_accumulator_used_when_window_covers_the_whole_sample():
    """窗口与整段采样重合时用累加器（mJ 级），而不是对 200 ms 采样点做梯形积分。"""
    est = estimate(_samples(n=3, power=100.0), energy_j=1234.0,
                   inputs=CostInputs(), window_s=0.6, sampler_duration_s=0.6)
    assert est.window_energy_j == pytest.approx(1234.0)
    assert "累加器" in est.energy_source


# --------------------------------------------------------------------------
# 按 kernel 归因：代理口径必须有界
# --------------------------------------------------------------------------
def test_attribution_never_exceeds_the_window_energy():
    """份额是 0–100% 的占比：分摊结果不许超过窗口总量（越界的归因是 bug 信号）。"""
    base = estimate(_samples(power=300.0), inputs=CostInputs(
        tokens_per_step=1000.0, price_per_kwh=1.0),
        window_s=1.0, sampler_duration_s=1.0,
        window_t0=0.0, window_t1=1.0, steps_in_window=4.0)
    shares = [0.0, 12.5, 50.0, 99.9, 100.0, 250.0]
    got = [attribute(base, s, steps=4.0).kernel_energy_j for s in shares]
    assert got[0] == pytest.approx(0.0)
    assert got[3] == pytest.approx(base.window_energy_j * 0.999, rel=1e-9)
    assert got[4] == pytest.approx(base.window_energy_j, rel=1e-9)
    assert got[5] == pytest.approx(base.window_energy_j, rel=1e-9)   # 上限被夹住
    assert all(0.0 <= g <= base.window_energy_j + 1e-9 for g in got)


def test_idle_power_is_subtracted_before_attribution():
    """扣空闲功耗后，分摊基数变小；不扣就会把待机功耗算到 kernel 头上。"""
    inp = CostInputs(tokens_per_step=1000.0, idle_power_w=50.0)
    base = estimate(_samples(power=300.0), inputs=inp,
                    window_s=1.0, sampler_duration_s=1.0,
                    window_t0=0.0, window_t1=1.0, steps_in_window=4.0)
    attr = attribute(base, 50.0, steps=4.0)
    # 300 J 里 50 W × 1 s = 50 J 是空闲，剩 250 J，取一半
    assert attr.kernel_energy_j == pytest.approx(125.0, rel=1e-9)
    assert attr.kernel_energy_per_launch_j == pytest.approx(125.0 / 4.0, rel=1e-9)


def test_attribution_keeps_the_window_account_untouched():
    """分摊返回副本：窗口账被就地改写的话，第二个 kernel 会拿到被改过的基数。"""
    base = estimate(_samples(power=300.0), inputs=CostInputs(tokens_per_step=1000.0),
                    window_s=1.0, sampler_duration_s=1.0,
                    window_t0=0.0, window_t1=1.0, steps_in_window=4.0)
    before = (base.window_energy_j, base.steps_in_window, base.kernel_energy_j)
    attribute(base, 30.0, steps=2.0)
    assert (base.window_energy_j, base.steps_in_window, base.kernel_energy_j) == before
    assert math.isclose(base.kernel_energy_j or 0.0, 0.0)


# --------------------------------------------------------------------------
# CLI 端到端：开关接上了没、门禁能不能变红
# --------------------------------------------------------------------------
def _run(capsys, *argv):
    from kernoscope.cli import main

    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_cost_section_is_present_and_honest_without_inputs(capsys):
    """不给 token/价格：成本 section 出现（否则用户不知道有这功能），且报的是拒绝不是数字。"""
    rc, out, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic")
    assert rc == 0
    assert "Section: CostLedger" in out
    assert "ks__cost_per_mtok" in out
    assert "no_energy" in out or "below_resolution" in out


def test_cli_strict_turns_red_when_cost_inputs_cannot_be_priced(capsys):
    """--strict + 给了价格却算不出钱 => 退出码 2（门禁有分辨力，不是摆设）。"""
    rc, out, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                        "--ks-price-per-kwh", "0.6", "--strict")
    assert rc == 2, out + err
    assert "cost ledger" in err
