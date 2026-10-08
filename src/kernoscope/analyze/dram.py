"""DRAM 流量估算 —— 用 NVML 显存忙碌时间 × 本机实测峰值带宽重建 `dram__bytes`。

为什么这条链成立：NVML 的 `utilization.memory` 是**显存子系统忙碌时间占比**（实测确认），
而 DRAM 在忙碌期间就是按峰值速率搬运，于是

    bytes ≈ Σ_i ( mem_util_i / 100 × Δt_i ) × peak_bandwidth

真机验证过的性质（见 README 与测试）：
  * 饱和拷贝：mem_util 96.8% ↔ 解析值 920 GB/s（用 1008 GB/s 换算，误差 6%）
  * L2 驻留负载：mem_util 只有 45%（说明它正确区分了"不是 DRAM bound"）
  * **采样周期 200 ms，是区间平均** —— 若 GPU 在窗口内大量空闲，代理会被稀释
    （同一个 256MB 拷贝，带宿主间隙时只剩 6.3%），所以必须同时报"占空比"，
    并在占空比过低时**拒绝给出数字**（改为 unavailable + 原因），而不是给个偏小的值。

与 ncu 的对应关系（写在 note 里，但**不占用 ncu 的指标名**，遵守 provenance R3）：
    dram__bytes.sum                                   → est__dram_bytes_total
    dram__bytes.sum / launches                        → est__dram_bytes_per_launch
    dram__throughput.avg.pct_of_peak_sustained_elapsed → est__dram_throughput_pct_of_peak
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# 占空比阈值：低于 LOW 直接判为不可用；低于 HIGH 给出强提示
DUTY_UNUSABLE = 50.0
DUTY_WEAK = 80.0
# 判定"这一时刻 GPU 确实在干活"的阈值。
# 两个轴分开取阈值：SM 5% 说明有 kernel 在跑；显存要 20% 才算真在搬运
# （实测被宿主间隙稀释的拷贝只有 6.3% —— 那是空闲，不是"在干活"）。
# 只看显存会让"空闲窗口"被误判为忙碌，从而把偏小的估算当成可信值。
BUSY_SM_UTIL = 5.0
BUSY_MEM_UTIL = 20.0


@dataclass
class DramEstimate:
    bytes_total: float = 0.0
    bytes_per_launch: float | None = None
    throughput_pct_of_peak: float = 0.0
    busy_pct_mean: float = 0.0
    duty_pct: float = 0.0
    peak_gbs: float = 0.0
    window_s: float = 0.0
    sampler_period_s: float = 0.0
    launches: int = 0
    # 账本对照（可选）：账本通常预测的是**每次 launch** 的字节数（如 2RH），
    # 所以默认按 per-launch 对拍；要与整窗口总量比时用 ledger_total。
    ledger_per_launch: float | None = None
    ledger_total: float | None = None
    ledger_range: tuple[float, float] | None = None   # 区间同样是 per-launch 口径
    notes: list[str] = field(default_factory=list)

    @property
    def ledger_basis(self) -> str:
        if self.ledger_per_launch is not None:
            return "per-launch"
        if self.ledger_total is not None:
            return "total"
        return ""

    @property
    def ledger_ready(self) -> bool:
        """对拍所需的口径是否成立。

        per-launch 标量账本与**区间账本**都要求"可归因"（单一 kernel 主导 + 占空比达标）；
        只有总量账本才退而要求"总量可用"。
        """
        if self.ledger_per_launch is not None or self.ledger_range is not None:
            return self.bytes_per_launch is not None and self.attributable
        if self.ledger_total is not None:
            return self.reliable
        return False

    @property
    def ledger_target(self) -> float | None:
        if self.ledger_per_launch is not None:
            return self.bytes_per_launch
        if self.ledger_total is not None:
            return self.bytes_total
        return None

    @property
    def ledger_err_pct(self) -> float | None:
        """相对误差。**必须是同基比较**：per-launch 账本对 per-launch 估算。"""
        if not self.ledger_ready:
            return None
        target, ref = self.ledger_target, (self.ledger_per_launch
                                           if self.ledger_per_launch is not None else self.ledger_total)
        if not ref or target is None:
            return None
        return 100.0 * (target - ref) / ref

    @property
    def range_position_pct(self) -> float | None:
        """估算值在账本区间 [low, high] 里的相对位置（0=下界，100=上界），per-launch 口径。"""
        if not self.ledger_range or not self.ledger_ready:
            return None
        low, high = self.ledger_range
        if high <= low or self.bytes_per_launch is None:
            return None
        return 100.0 * (self.bytes_per_launch - low) / (high - low)

    @property
    def range_verdict(self) -> str:
        if not self.ledger_range:
            return ""
        low, high = self.ledger_range
        if not self.ledger_ready or self.bytes_per_launch is None:
            return "无法裁决（per-launch 口径不成立：占空比不足或未锁定单一 kernel）"
        if self.bytes_per_launch < low:
            return "低于下界（可能不是被测 kernel，或账本漏项反了）"
        if self.bytes_per_launch > high:
            return "高于上界（账本可能漏了一项流量，例如多读了一趟）"
        pos = self.range_position_pct or 0.0
        if pos <= 25:
            return "落在区间靠下界侧"
        if pos >= 75:
            return "落在区间靠上界侧"
        return "落在区间中段"

    @property
    def reliable(self) -> bool:
        """总量是否可用。

        估算 = Σ(mem_util×Δt)×峰值，对"窗口里有空闲"是**积分求和**式、不会被稀释；
        真正的假设是"忙碌时段显存处于饱和"。占空比太低时这个假设无法保证，故不给数字。
        """
        return self.peak_gbs > 0 and self.duty_pct >= DUTY_UNUSABLE and self.busy_pct_mean > 0

    @property
    def attributable(self) -> bool:
        """能否把总量归因到单个 kernel（per-launch 口径需要窗口被该 kernel 主导）。"""
        return self.reliable and self.duty_pct >= DUTY_WEAK and self.launches > 0


def estimate(samples: Sequence, peak_gbs: float, window_s: float,
             launches: int = 0, sampler_period_s: float = 0.0,
             ledger_per_launch: float | None = None,
             ledger_total: float | None = None,
             ledger_range: tuple[float, float] | None = None) -> DramEstimate:
    """从 NVML 采样序列估算 DRAM 流量。

    samples: 具 `.t`(秒) / `.mem_util_pct` / `.sm_util_pct` 属性的采样点列表。
    """
    est = DramEstimate(peak_gbs=peak_gbs, window_s=window_s, launches=launches,
                       ledger_per_launch=ledger_per_launch, ledger_total=ledger_total,
                       ledger_range=ledger_range, sampler_period_s=sampler_period_s)
    if not samples:
        est.notes.append("没有采样点：NVML 采样被关闭或不可用")
        return est

    # 采样间隔：用相邻时间戳差分；首点沿用中位间隔
    deltas: list[float] = []
    for a, b in zip(samples, samples[1:]):
        d = float(b.t) - float(a.t)
        if d > 0:
            deltas.append(d)
    deltas.sort()
    default_dt = deltas[len(deltas) // 2] if deltas else (sampler_period_s or 0.2)
    est.sampler_period_s = default_dt

    busy_time = 0.0          # 显存忙碌时间（秒）
    busy_pcts: list[float] = []
    busy_samples = 0
    total_time = 0.0
    for i, s in enumerate(samples):
        dt = (float(samples[i + 1].t) - float(s.t)) if i + 1 < len(samples) else default_dt
        if dt <= 0:
            dt = default_dt
        util = max(0.0, min(100.0, float(s.mem_util_pct)))
        busy_time += util / 100.0 * dt
        busy_pcts.append(util)
        total_time += dt
        if float(s.sm_util_pct) >= BUSY_SM_UTIL or util >= BUSY_MEM_UTIL:
            busy_samples += 1

    est.busy_pct_mean = sum(busy_pcts) / len(busy_pcts)
    est.duty_pct = 100.0 * busy_samples / len(samples)
    est.bytes_total = busy_time * peak_gbs * 1e9
    est.throughput_pct_of_peak = est.busy_pct_mean
    if launches > 0:
        est.bytes_per_launch = est.bytes_total / launches

    if peak_gbs <= 0:
        est.notes.append("缺少实测峰值带宽：加 --ks-calibration machine-profile.json 或 --ks-peak-gbps")
    if est.duty_pct < DUTY_UNUSABLE:
        est.notes.append(
            f"采样窗口内 GPU 只有 {est.duty_pct:.0f}% 时间在干活：无法确认忙碌时段显存处于饱和"
            f"（估算以此为假设），因此不给总量数字；请让被测 kernel 在紧循环里跑满窗口")
    elif est.duty_pct < DUTY_WEAK:
        est.notes.append(
            f"占空比 {est.duty_pct:.0f}%：忙碌时段是否饱和无法确认，总量可能偏大，且不能归因到单个 kernel")
    if total_time and window_s and total_time < 0.5 * window_s:
        est.notes.append("采样覆盖时间不足窗口一半：采样器可能启动过晚或提前退出")
    est.notes.append(
        "采样周期 %.0f ms、区间平均；字节 = Σ(mem_util×Δt)×实测峰值带宽" %
        (est.sampler_period_s * 1000))
    return est


def to_metrics(est: DramEstimate) -> list[tuple[str, float | None, str, str, str]]:
    """给 section 组装用的 (name, value, unit, source, note) 列表。

    返回时只做数值与文案的组装，provenance 由调用方决定（本模块永远产出 estimated）。
    """
    out: list[tuple[str, float | None, str, str, str]] = []
    note = "；".join(est.notes)
    out.append(("est__dram_bytes_total", round(est.bytes_total / 1e6, 1), "MB",
                "NVML memory-busy time × measured peak bandwidth（对应 ncu dram__bytes.sum）", note))
    if est.bytes_per_launch is not None:
        out.append(("est__dram_bytes_per_launch", round(est.bytes_per_launch / 1e6, 2), "MB",
                    f"total / {est.launches} launches（对应 ncu dram__bytes.sum 的按 launch 口径）",
                    "仅在『单一 kernel 主导采样窗口』时成立"))
    out.append(("est__dram_throughput_pct_of_peak", round(est.throughput_pct_of_peak, 1), "%",
                "NVML memory-busy 占比（对应 ncu dram__throughput.avg.pct_of_peak_sustained_elapsed）",
                "忙碌时间占比口径，与按字节/时间的口径略有差异"))
    out.append(("meas__sampler_duty_pct", round(est.duty_pct, 1), "%",
                "采样窗口中 SM≥5% 或显存≥20% 的采样点占比",
                "低于 80% 时不给 per-launch 归因；低于 50% 时连总量也不给"))
    if est.ledger_basis:
        ref = est.ledger_per_launch if est.ledger_per_launch is not None else est.ledger_total
        basis = "每次 launch" if est.ledger_basis == "per-launch" else "整窗口总量"
        out.append(("est__dram_vs_ledger_err_pct",
                    round(est.ledger_err_pct, 1) if est.ledger_err_pct is not None else None, "%",
                    f"（估算 − 账本）/ 账本，同为「{basis}」口径",
                    f"账本 {ref / 1e6:.1f} MB；对拍常用 |误差| < 10% 作为验收线"))
    if est.ledger_range:
        out.append(("est__dram_range_position_pct",
                    round(est.range_position_pct, 1) if est.range_position_pct is not None else None, "%",
                    f"在账本区间 [{est.ledger_range[0] / 1e6:.1f}, {est.ledger_range[1] / 1e6:.1f}] MB "
                    f"（per-launch）中的相对位置", est.range_verdict))
    return out
