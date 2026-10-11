"""成本账（C1）—— 把"这个算子花了多少"从秒换算成钱与量。

为什么这是主线而不是附加功能：这个工具存在的理由是**降低训练/推理成本**。
"kernel 慢了 12%"只是中间量；要让人或 agent 真的去改代码，得看到
「每百万 token 多少钱、其中这个 kernel 吃掉多少、改完能省多少」。

口径的诚实边界（这是本模块唯一难的地方，也是它值得单独成文件的原因）：

  * **能量是实测的**：窗口内 NVML 功率读数积分（或整段能量累加器）。不按 TDP 顶替 ——
    TDP 是规格，不是这台机器此刻的功耗。
  * **token 数是调用方给的**：我们不知道你的 step 处理多少 token。给了才算 per-token，
    没给就拒绝给单位成本（`no_tokens`），而不是按"典型 8k"猜一个。
  * **价格是调用方给的**：电费与租约是两种**不同载体**的成本，分别给、分别报，
    都给才相加 —— 这样"¥/Mtoken"这个数永远能拆回它的组成部分。
  * **按 kernel 归因是代理**：能耗按 kernel 在窗口内的时间占比分摊（`est__` + estimated）。
    它假设窗口内功率大致均匀；`--ks-idle-power-w` 可以先把空闲功耗扣掉再分摊，
    扣除与否都会写进 note —— 归因是**方向性**的，不是计数器读数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..refusals import Refusal, RefusalCode, Remedy

# NVML 采样周期与功率读数分辨率的默认值（真机：200 ms 间隔、整数瓦）。
# 写在这里是为了让"能耗精度"这句话有具体数字，而不是一句"可能有误差"。
DEFAULT_SAMPLER_PERIOD_S = 0.2
DEFAULT_POWER_RESOLUTION_W = 1.0
# 窗口短于这么长时，积分只剩一两个采样点：拒绝给能耗（宁可不给，也不给一个数量级虚假的数）
MIN_WINDOW_S = 0.5
J_PER_KWH = 3.6e6


@dataclass
class CostInputs:
    """调用方给的**价格与工作量**。全是可选：没给的东西我们拒绝，而不是猜。"""

    tokens_per_step: float | None = None      # 一次迭代（step）处理的 token 数
    price_per_kwh: float | None = None        # 电费口径（每 kWh 多少钱）
    price_per_hour: float | None = None       # 租约口径（这台机器每小时多少钱）
    idle_power_w: float | None = None         # 空闲功耗（用于把非计算能耗先扣掉）
    currency: str = "CNY"                     # 只影响单位标签，我们不做汇率换算

    @property
    def any_given(self) -> bool:
        return any(v is not None for v in (self.tokens_per_step, self.price_per_kwh,
                                           self.price_per_hour))

    def to_dict(self) -> dict[str, Any]:
        return {"tokens_per_step": self.tokens_per_step,
                "price_per_kwh": self.price_per_kwh,
                "price_per_hour": self.price_per_hour,
                "idle_power_w": self.idle_power_w,
                "currency": self.currency}


@dataclass
class CostEstimate:
    """一次成本账的全部中间量与结论。**每个数都带它的前提**（此处只存事实，散文在 sections 侧）。"""

    inputs: CostInputs = field(default_factory=CostInputs)
    # --- 能耗（实测）---
    window_energy_j: float | None = None
    energy_source: str = ""
    energy_resolution: str = ""
    window_s: float = 0.0
    # --- 量（steps 由 launch 计数推出，token 数由调用方给）---
    steps_in_window: float | None = None
    tokens_total: float | None = None
    tokens_per_s: float | None = None
    j_per_token: float | None = None
    # --- 钱 ---
    cost_per_hour_energy: float | None = None
    cost_per_hour_rental: float | None = None
    cost_per_hour: float | None = None
    cost_per_mtok_energy: float | None = None
    cost_per_mtok_rental: float | None = None
    cost_per_mtok: float | None = None
    # --- 按 window 内时间占比分摊到被匹配 kernel（代理口径）---
    kernel_share_pct: float | None = None
    kernel_energy_j: float | None = None
    kernel_energy_per_launch_j: float | None = None
    kernel_cost_per_hour: float | None = None
    # --- 拒绝与说明 ---
    energy_refusal: Refusal | None = None    # 非空 -> 能耗不可用
    tokens_refusal: Refusal | None = None    # 非空 -> 单位量（tokens/s、J/token）不可用
    price_refusal: Refusal | None = None     # 非空 -> 钱不可用（没给价格）
    steps_refusal: Refusal | None = None     # 非空 -> 窗口内 steps 不可归因
    notes: list[str] = field(default_factory=list)

    @property
    def has_energy(self) -> bool:
        return self.window_energy_j is not None

    @property
    def energy_w(self) -> float | None:
        if self.window_energy_j is None or self.window_s <= 0:
            return None
        return self.window_energy_j / self.window_s


def integrate_power(samples: Sequence[Any], t0: float | None = None,
                    t1: float | None = None) -> tuple[float, float] | None:
    """把窗口内的功率采样按梯形法积分成焦耳。返回 (焦耳, 覆盖秒数)；点太少返回 None。

    只用调用方给的采样时钟与功率读数，不做任何外推 —— 采样点跨越窗口边界时被排除，
    所以返回的覆盖秒数可能小于窗口长度，这个差值由调用方如实报出来。
    """
    pts = [(float(getattr(s, "t", 0.0)), float(getattr(s, "power_w", 0.0)))
           for s in samples if getattr(s, "power_w", None) is not None]
    if t0 is not None:
        pts = [p for p in pts if p[0] >= t0]
    if t1 is not None:
        pts = [p for p in pts if p[0] <= t1]
    pts.sort()
    if len(pts) < 2:
        return None
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        dt = tb - ta
        if dt <= 0:
            continue
        energy += 0.5 * (pa + pb) * dt
    span = pts[-1][0] - pts[0][0]
    return (energy, span)


def estimate(samples: Sequence[Any], *,
             inputs: CostInputs | None = None,
             energy_j: float = 0.0,
             sampler_duration_s: float = 0.0,
             window_s: float = 0.0,
             window_t0: float | None = None,
             window_t1: float | None = None,
             steps_in_window: float | None = None,
             matched_kernel_share_pct: float | None = None,
             sampler_period_s: float = DEFAULT_SAMPLER_PERIOD_S) -> CostEstimate:
    """纯函数：采样点 + 窗口 + 调用方给的 token/价格 → 成本账。

    不读文件、不跑命令，所以能被测试完整覆盖（包括"给不出数"的那几条路径）。
    """
    inp = inputs or CostInputs()
    est = CostEstimate(inputs=inp, window_s=float(window_s or 0.0),
                       steps_in_window=steps_in_window,
                       kernel_share_pct=matched_kernel_share_pct)

    # ---- ① 能耗：窗口内功率积分优先，整段能量累加器次之 ----
    period = float(sampler_period_s or DEFAULT_SAMPLER_PERIOD_S)
    if est.window_s <= 0:
        est.energy_refusal = Refusal(
            RefusalCode.NO_ENERGY,
            "没有可用的采样窗口长度：能耗无法归到一段负载上",
            observed={"window_s": 0.0}, threshold={"window_s_min": MIN_WINDOW_S})
    elif est.window_s < MIN_WINDOW_S:
        est.energy_refusal = Refusal(
            RefusalCode.BELOW_RESOLUTION,
            f"窗口只有 {est.window_s:.2f} s（< {MIN_WINDOW_S:.1f} s）："
            f"按 {period:.1f} s 的采样周期积不出可信的能耗，宁可不给",
            observed={"window_s": round(est.window_s, 3)},
            threshold={"window_s_min": MIN_WINDOW_S, "sampler_period_s": period},
            remedy=Remedy(
                "把稳态循环跑长一点（至少几百毫秒到几秒），或去掉 --ks-window-s 之类"
                "把窗口裁到 0.5 s 以下的选项 —— 能耗是时间的积分，窗口太短就没有分辨率",
                command="kernoscope --set basic --ks-tokens 8192 --ks-price-per-kwh 0.6 "
                        "-o rep -- python bench_loop.py",
                cost_s=0.0, then_answers=("cost_per_token",)))
    else:
        full_window = (sampler_duration_s > 0
                       and est.window_s >= sampler_duration_s * 0.995)
        if full_window and energy_j > 0:
            est.window_energy_j = float(energy_j)
            est.energy_source = ("pynvml 能量累加器（整段采样，mJ 累加，"
                                 "窗口与采样段重合时可直接用）")
            est.energy_resolution = "0.001 J（累加器）"
        else:
            integ = integrate_power(samples, window_t0, window_t1)
            if integ is None:
                if energy_j > 0:
                    est.window_energy_j = float(energy_j)
                    est.energy_source = "pynvml 能量累加器（整段采样）"
                    est.energy_resolution = "0.001 J（累加器）"
                    est.notes.append(
                        "窗口被裁短（--ks-window/--ks-warmup/--ks-window-s），"
                        "而功率采样点不足以在窗口内积分：能耗用的是**整段**累加值，"
                        "比窗口略宽 —— 当作上界看")
                else:
                    est.energy_refusal = Refusal(
                        RefusalCode.NO_ENERGY,
                        "窗口内没有足够的功率采样点（NVML 被关掉/不可用，或稳态段太短）："
                        "我们不按 TDP 顶替一个能量数",
                        observed={"power_samples_in_window": 0})
            else:
                e_win, span = integ
                est.window_energy_j = e_win
                est.energy_source = (f"NVML 功率读数在窗口内梯形积分"
                                     f"（{span:.2f} s / 窗口 {est.window_s:.2f} s）")
                est.energy_resolution = (f"{period:.1f} s × "
                                         f"{DEFAULT_POWER_RESOLUTION_W:.0f} W（采样周期×读数分辨率）")
                if span + period < est.window_s:
                    est.notes.append(
                        f"功率采样只覆盖窗口的 {100.0 * span / est.window_s:.0f}%："
                        f"能耗按这 {span:.2f} s 积分，未覆盖部分不外推")

    # ---- ② 量：steps 由 launch 计数推出（窗口对齐），token 数由调用方给 ----
    if est.has_energy:
        if steps_in_window is None:
            est.steps_refusal = Refusal(
                RefusalCode.WINDOW_LAUNCH_UNALIGNED,
                "窗口内的 launch 数无法对齐到采样窗口：单位成本的分母不成立",
                observed={"launches_in_window": None})
        elif steps_in_window <= 0:
            est.steps_refusal = Refusal(
                RefusalCode.NO_LAUNCH_ACTIVITY,
                "窗口内没有完整的 launch：没有可分摊的迭代",
                observed={"launches_in_window": steps_in_window})
        if not inp.tokens_per_step or inp.tokens_per_step <= 0:
            est.tokens_refusal = Refusal(
                RefusalCode.NO_TOKENS,
                "没有给每次迭代的 token 数（--ks-tokens <N>）：unit cost 的分母我们不知道，"
                "不猜典型值",
                observed={"tokens_per_step": inp.tokens_per_step})
        elif est.steps_refusal is not None:
            est.tokens_refusal = est.steps_refusal
        else:
            est.tokens_total = inp.tokens_per_step * steps_in_window
            if est.window_s > 0:
                est.tokens_per_s = est.tokens_total / est.window_s
                est.j_per_token = est.window_energy_j / est.tokens_total
                est.notes.append(
                    f"token 数口径：{inp.tokens_per_step:g} token/step × "
                    f"{steps_in_window:g} step（窗口内等效 launch 数）")
    else:
        est.tokens_refusal = est.energy_refusal

    # ---- ③ 钱：两种载体分别算，都给才相加 ----
    e_w = est.energy_w
    if inp.price_per_kwh is not None and e_w is not None:
        est.cost_per_hour_energy = e_w * 3600.0 / J_PER_KWH * inp.price_per_kwh
    if inp.price_per_hour is not None:
        est.cost_per_hour_rental = float(inp.price_per_hour)
    parts = [c for c in (est.cost_per_hour_energy, est.cost_per_hour_rental) if c is not None]
    if parts:
        est.cost_per_hour = sum(parts)
    if inp.price_per_kwh is None and inp.price_per_hour is None:
        est.price_refusal = Refusal(
            RefusalCode.NO_PRICE,
            "没有给价格（--ks-price-per-kwh 电费口径 / --ks-price-per-hour 租约口径）："
            "钱换算不了，能耗数字仍然可用",
            observed={"price_per_kwh": None, "price_per_hour": None})
    if est.tokens_per_s:
        tph = est.tokens_per_s * 3600.0
        if est.cost_per_hour_energy is not None:
            est.cost_per_mtok_energy = est.cost_per_hour_energy * 1e6 / tph
        if est.cost_per_hour_rental is not None:
            est.cost_per_mtok_rental = est.cost_per_hour_rental * 1e6 / tph
        if est.cost_per_hour is not None:
            est.cost_per_mtok = est.cost_per_hour * 1e6 / tph

    return est


def attribute(est: CostEstimate, share_pct: float, *,
              steps: float | None = None) -> CostEstimate:
    """把一个窗口级成本账**分摊**到某个 kernel（代理口径：窗口内时间占比 × 能耗）。

    返回副本，不动原账 —— 因为要按每个 kernel 各分摊一次，而窗口账只有一份。
    分摊口径的说明（含是否扣了空闲功耗）由调用侧渲染；这里只做算术。
    """
    out = CostEstimate(
        inputs=est.inputs, window_energy_j=est.window_energy_j,
        energy_source=est.energy_source, energy_resolution=est.energy_resolution,
        window_s=est.window_s, steps_in_window=est.steps_in_window,
        cost_per_hour=est.cost_per_hour,
        cost_per_hour_energy=est.cost_per_hour_energy,
        cost_per_hour_rental=est.cost_per_hour_rental,
        notes=list(est.notes), kernel_share_pct=share_pct)
    if not est.has_energy or share_pct is None:
        return out
    share = max(0.0, min(100.0, share_pct)) / 100.0
    e_active = est.window_energy_j
    if est.inputs.idle_power_w is not None:
        e_active = max(0.0, est.window_energy_j - est.inputs.idle_power_w * est.window_s)
    out.kernel_energy_j = e_active * share
    n_steps = steps if steps else est.steps_in_window
    if n_steps:
        out.kernel_energy_per_launch_j = out.kernel_energy_j / n_steps
    if est.cost_per_hour is not None:
        out.kernel_cost_per_hour = est.cost_per_hour * share
    return out
