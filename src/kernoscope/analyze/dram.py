"""DRAM 流量估算 —— 用 NVML 显存忙碌时间 × 本机实测峰值带宽重建 `dram__bytes`。

为什么这条链成立：NVML 的 `utilization.memory` 是**显存子系统忙碌时间占比**（实测确认），
而 DRAM 在忙碌期间就是按峰值速率搬运，于是

    bytes ≈ Σ_i ( mem_util_i / 100 × Δt_i ) × peak_bandwidth

真机验证过的性质（见 README 与 bench/artifacts）：
  * 饱和拷贝：mem_util 96.8% ↔ 解析值 920 GB/s（用 1008 GB/s 换算，误差 6%）
  * L2 驻留负载：mem_util 只有 45%（说明它正确区分了"不是 DRAM bound"）
  * **采样周期 200 ms、整数百分比、区间平均** —— 若 GPU 在窗口内大量空闲，代理会被稀释
    （同一个 256MB 拷贝，带宿主间隙时只剩 6.3%），所以必须同时报"窗口覆盖率"。

本模块的三条纪律（都是需求方实测踩出来的）：

1. **口径不成立就报 n/a + 原因，绝不静默消失**（R-02）。原因必须点明**哪个前提没成立**
   （R-32），而不是复述算法定义。
2. **一个量一个名字**（R-15/B5）：`窗口是否被负载主导`（覆盖率）与
   `忙碌时段显存是否饱和`（integrity of the integral）是两件事，分开命名、分开报。
3. **代理量必须带自身分辨率**（R-31）：`utilization.memory` 的读数是整数百分比，
   0.0% 与"0.3% 被截断成 0"不可区分 —— 这个不确定性必须与数字同时出现。

第四条（R-33）：账本/区间对拍**不只支持 NVML 代理口径**。当代理口径不成立
（占空比不足、显存忙碌读数为 0）而访存轨迹可用时，用轨迹口径的 per-launch 字节对拍 ——
否则 compute-bound（GEMM，mem busy 3.2%）与 L2-resident（FlashAttention，0.0%）这两类负载
在账本验收上永远关不掉。两条口径分开标注，不混用。

与 ncu 的对应关系（写在 note/source 里，但**不占用 ncu 的指标名**，遵守 provenance R3）：
    dram__bytes.sum                                   → est__dram_bytes_total
    dram__bytes.sum / launches                        → est__dram_bytes_per_launch[_trace]
    dram__throughput.avg.pct_of_peak_sustained_elapsed → est__dram_throughput_pct_of_peak
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# 窗口覆盖率阈值：低于 UNUSABLE 直接判为不可用；低于 WEAK 不给 per-launch 归因
DUTY_UNUSABLE = 50.0
DUTY_WEAK = 80.0
# 判定"这一时刻 GPU 确实在干活"的阈值。
# 两个轴分开取阈值：SM 5% 说明有 kernel 在跑；显存要 20% 才算真在搬运
# （实测被宿主间隙稀释的拷贝只有 6.3% —— 那是空闲，不是"在干活"）。
# 只看显存会让"空闲窗口"被误判为忙碌，从而把偏小的估算当成可信值。
BUSY_SM_UTIL = 5.0
BUSY_MEM_UTIL = 20.0
# R-31：NVML utilization.memory 是**整数百分比**，这就是分辨率下限。
MEM_UTIL_RESOLUTION_PCT = 1.0
# 自动裁剪窗口时在首个/末个忙碌采样点外保留的余量（秒）；用采样周期的倍数表示。
TRIM_MARGIN_PERIODS = 1.0

WINDOW_MODES = ("auto", "full")
LEDGER_BASES = ("auto", "nvml", "trace")


@dataclass
class DramEstimate:
    bytes_total: float = 0.0
    bytes_per_launch: float | None = None
    throughput_pct_of_peak: float = 0.0
    mem_util_pct_mean: float = 0.0          # 积分的被积函数均值（NVML 读数，整数%）
    window_coverage_pct: float = 0.0        # "窗口被负载主导"的观测（旧名 duty_pct）
    peak_gbs: float = 0.0
    peak_source: str = ""                   # R-06：峰值来源（文件 + 标定时间）
    window_s: float = 0.0                   # 采样窗（wall time）
    window_used_s: float = 0.0              # 实际参与积分的覆盖时间
    window_dropped_s: float = 0.0           # 被裁掉的（启动/JIT/空闲）时间
    window_mode: str = "full"
    trimmed: bool = False
    sampler_period_s: float = 0.0
    mem_util_resolution_pct: float = MEM_UTIL_RESOLUTION_PCT
    n_samples: int = 0
    launches: int = 0
    # 账本对照（可选）：账本通常预测的是**每次 launch** 的字节数（如 2RH），
    # 所以默认按 per-launch 对拍；要与整窗口总量比时用 ledger_total。
    ledger_per_launch: float | None = None
    ledger_total: float | None = None
    ledger_range: tuple[float, float] | None = None   # 区间同样是 per-launch 口径
    # R-33：第二条口径 —— 访存轨迹给出的 per-launch 字节（模型量，estimated）
    trace_bytes_per_launch: float | None = None
    trace_kernel: str = ""
    trace_unavailable_reason: str = ""
    notes: list[str] = field(default_factory=list)

    # ---- 兼容别名（旧字段名，语义不变）----
    @property
    def duty_pct(self) -> float:
        return self.window_coverage_pct

    @property
    def busy_pct_mean(self) -> float:
        return self.mem_util_pct_mean

    # ---- 前提（R-32）----
    @property
    def peak_missing(self) -> bool:
        return self.peak_gbs <= 0

    @property
    def below_resolution(self) -> bool:
        """读数落在了分辨率下限以下：无法区分"真的 0"与"被截断成 0"。"""
        return self.mem_util_pct_mean < self.mem_util_resolution_pct

    @property
    def reliable(self) -> bool:
        """总量是否可用。

        估算 = Σ(mem_util×Δt)×峰值，对"窗口里有空闲"是**积分求和**式、不会被稀释；
        真正的假设是"忙碌时段显存处于饱和"。以下任一前提不成立就不给总量数字：
          * 缺少实测峰值带宽（没有峰值就没有字节数）
          * 没有采样点
          * 显存忙碌读数低于分辨率下限（0.0% 时不能声称"真的没搬"）
          * 窗口覆盖率低于 50%（窗口没被负载主导，忙碌时段是否饱和无从确认）
        """
        return (not self.peak_missing and self.n_samples > 0
                and not self.below_resolution
                and self.window_coverage_pct >= DUTY_UNUSABLE)

    @property
    def attributable(self) -> bool:
        """能否把总量归因到单个 kernel（per-launch 口径需要窗口被该 kernel 主导）。"""
        return self.reliable and self.window_coverage_pct >= DUTY_WEAK and self.launches > 0

    @property
    def trace_available(self) -> bool:
        return self.trace_bytes_per_launch is not None and self.trace_bytes_per_launch > 0

    @property
    def trace_total(self) -> float | None:
        """轨迹口径的整窗口总量 = per-launch × 已匹配 launch 数（只在单 kernel 时成立）。"""
        if not self.trace_available or self.launches <= 0:
            return None
        return self.trace_bytes_per_launch * self.launches

    # ---- 拒绝给数的原因：点明哪一个前提没成立（R-32）----
    def _total_failures(self) -> list[str]:
        out: list[str] = []
        if self.n_samples <= 0:
            out.append("没有 NVML 采样点（--ks-no-nvml 打开，或采样器不可用）")
        if self.peak_missing:
            out.append("缺少本机实测峰值带宽：先跑 tools/machine-profile.py，"
                       "再加 --ks-calibration machine-profile.json（或 --ks-peak-gbps）")
        if self.n_samples > 0 and self.below_resolution:
            out.append(
                f"显存忙碌读数 {self.mem_util_pct_mean:.1f}% 低于 NVML utilization.memory 的"
                f"分辨率下限 {self.mem_util_resolution_pct:.0f} 个百分点："
                f"无法区分『真的没搬』与『{self.mem_util_resolution_pct:.1f}% 被截断成 0』")
        if (self.n_samples > 0 and not self.peak_missing and not self.below_resolution
                and self.window_coverage_pct < DUTY_UNUSABLE):
            need = _window_seconds_for(self.window_coverage_pct, DUTY_UNUSABLE, self.window_s)
            out.append(
                f"窗口覆盖率 {self.window_coverage_pct:.0f}% < {DUTY_UNUSABLE:.0f}%：窗口没被被测负载"
                f"主导，忙碌时段是否饱和无从确认"
                + (f"；把稳态循环延长到约 {need:.0f} 秒，或加 --ks-window auto 只统计稳态段"
                   if need else ""))
        return out

    def _attribution_failures(self) -> list[str]:
        out = self._total_failures()
        if self.launches <= 0:
            out.append("未锁定单一 kernel（匹配到多个 kernel）：per-launch 口径不适用，"
                       "请用 -k 只选一个 kernel")
        elif (self.reliable and self.window_coverage_pct < DUTY_WEAK and not out):
            need = _window_seconds_for(self.window_coverage_pct, DUTY_WEAK, self.window_s)
            out.append(
                f"窗口覆盖率 {self.window_coverage_pct:.0f}% < {DUTY_WEAK:.0f}%：无法把 "
                f"{self.bytes_total / 1e6:.0f} MB 总量归因到每个 launch"
                + (f"；把稳态循环延长到约 {need:.0f} 秒，或加 --ks-window auto" if need else ""))
        return out

    def reason_for(self, name: str, basis: str = "auto") -> str:
        """某个指标为什么是 n/a —— 具体到失败条件（R-32）。"""
        if name in ("est__dram_bytes_total",):
            fails = self._total_failures()
            return "；".join(fails) or "无法估算（未知原因）"
        if name in ("est__dram_bytes_per_launch",):
            fails = self._attribution_failures()
            return "；".join(fails) or "无法归因到单个 launch（未知原因）"
        if name == "est__dram_bytes_per_launch_trace":
            return self.trace_unavailable_reason or "轨迹未提供 per-launch 字节"
        if name == "est__dram_vs_ledger_err_pct":
            return self._ledger_failure(basis, need="误差")
        if name == "est__dram_range_position_pct":
            return self._ledger_failure(basis, need="区间位置", want_total=False)
        return "；".join(self.notes) or "无法估算"

    def _ledger_failure(self, basis: str, need: str, want_total: bool | None = None) -> str:
        why = "；".join(self._total_failures() if want_total else self._attribution_failures()) \
            or "对拍口径不成立"
        hint = ("；可以加 --ks-memtrace <file> 换用**轨迹口径**的 per-launch 字节"
                if not self.trace_available else "")
        return f"{need}行无法给出：{why}{hint}"

    # ---- 口径解析（R-33）----
    @property
    def ledger_basis(self) -> str:
        """账本请求的是哪种基（per-launch / total）。"""
        if self.ledger_per_launch is not None:
            return "per-launch"
        if self.ledger_total is not None:
            return "total"
        return ""

    @property
    def ledger_requested(self) -> bool:
        return self.ledger_basis != "" or bool(self.ledger_range)

    def resolve_basis(self, basis: str = "auto", want_total: bool | None = None) -> str:
        """返回实际可用的口径名：'' / 'nvml-per-launch' / 'trace-per-launch' / 'nvml-total'。

        选择规则（auto）：
          1. 请求了 per-launch 账本或区间 → 优先 NVML per-launch（可归因时）；
             否则退到轨迹 per-launch（单 kernel 时可用）—— 这就是 R-33 的落点。
          2. 请求了总量账本 → 优先 NVML 总量；否则退到轨迹总量（per-launch × launch 数）。
        显式指定 `nvml` / `trace` 时不做回退：拿不到就报 n/a + 原因（诚实优先）。

        `want_total` 单独可覆盖：**区间账本按定义是 per-launch 口径**，所以给区间算位置时
        必须传 want_total=False，否则会拿"整窗口总量"去比一个 per-launch 区间（荒谬位置）。
        """
        if not self.ledger_requested:
            return ""
        if want_total is None:
            wants_total = self.ledger_basis == "total"
        else:
            wants_total = want_total
        if basis == "nvml":
            return "nvml-total" if wants_total and self.reliable else (
                "nvml-per-launch" if not wants_total and self.attributable else "")
        if basis == "trace":
            if wants_total:
                return "trace-per-launch" if self.trace_total is not None else ""
            return "trace-per-launch" if self.trace_available and self.launches > 0 else ""
        # auto
        if wants_total:
            if self.reliable:
                return "nvml-total"
            return "trace-per-launch" if self.trace_total is not None else ""
        if self.attributable:
            return "nvml-per-launch"
        return "trace-per-launch" if self.trace_available and self.launches > 0 else ""

    def basis_label(self, basis: str = "auto") -> str:
        b = self.resolve_basis(basis)
        return {
            "nvml-per-launch": "NVML 代理 per-launch",
            "nvml-total": "NVML 代理总量",
            "trace-per-launch": "访存轨迹 per-launch（模型量）",
        }.get(b, "口径不成立")

    def target_bytes(self, basis: str = "auto", want_total: bool | None = None) -> float | None:
        b = self.resolve_basis(basis, want_total)
        total = self.ledger_basis == "total" if want_total is None else want_total
        if b == "nvml-per-launch":
            return self.bytes_per_launch
        if b == "nvml-total":
            return self.bytes_total
        if b == "trace-per-launch":
            return self.trace_total if total else self.trace_bytes_per_launch
        return None

    def reference_bytes(self, basis: str = "auto") -> float | None:
        if self.ledger_per_launch is not None:
            return self.ledger_per_launch
        return self.ledger_total

    # ---- 账本对拍（同基比较）----
    @property
    def ledger_ready(self) -> bool:
        """对拍所需的口径是否成立（auto 口径，含轨迹回退）。"""
        return self.resolve_basis("auto") != "" and self.reference_bytes() is not None

    @property
    def ledger_target(self) -> float | None:
        return self.target_bytes("auto")

    @property
    def ledger_err_pct(self) -> float | None:
        """相对误差。**必须是同基比较**：per-launch 账本对 per-launch 估算。"""
        return self.ledger_err_pct_for("auto")

    def ledger_err_pct_for(self, basis: str = "auto") -> float | None:
        target, ref = self.target_bytes(basis), self.reference_bytes(basis)
        if target is None or not ref:
            return None
        return 100.0 * (target - ref) / ref

    @property
    def range_position_pct(self) -> float | None:
        """估算值在账本区间 [low, high] 里的相对位置（0=下界，100=上界），per-launch 口径。"""
        return self.range_position_pct_for("auto")

    def range_position_pct_for(self, basis: str = "auto") -> float | None:
        """区间是 **per-launch** 口径，所以这里永远按 per-launch 取估算值。"""
        if not self.ledger_range:
            return None
        low, high = self.ledger_range
        target = self.target_bytes(basis, want_total=False)
        if high <= low or target is None:
            return None
        return 100.0 * (target - low) / (high - low)

    @property
    def range_verdict(self) -> str:
        return self.range_verdict_for("auto")

    def range_verdict_for(self, basis: str = "auto") -> str:
        """R-05：不仅给 %，还给判词与物理含义。"""
        if not self.ledger_range:
            return ""
        low, high = self.ledger_range
        target = self.target_bytes(basis, want_total=False)
        if target is None:
            return f"无法裁决（{self.basis_label(basis)}：口径不成立）"
        tag = "" if self.resolve_basis(basis, want_total=False) == "nvml-per-launch" \
            else "（轨迹口径）"
        if target < low:
            return f"低于下界{tag}（可能抓错了 kernel，或账本漏项方向反了）"
        if target > high:
            return f"高于上界{tag}（账本可能漏了一项流量，例如多读了一趟）"
        pos = self.range_position_pct_for(basis) or 0.0
        if pos <= 25:
            return f"落在区间靠下界侧{tag}"
        if pos >= 75:
            return f"落在区间靠上界侧{tag}"
        return f"落在区间中段{tag}"

    # 兼容旧调用：旧代码用 ledger_err_pct / range_position_pct 时的 detail 文本
    @property
    def ledger_detail(self) -> str:
        ref = self.reference_bytes()
        if ref is None:
            return ""
        basis = "每次 launch" if self.ledger_basis == "per-launch" else "整窗口总量"
        b = self.resolve_basis("auto")
        note = f"账本 {ref / 1e6:.1f} MB；对拍常用 |误差| < 10% 作为验收线"
        if b.startswith("trace"):
            note += f"；此次用**轨迹口径**（{self.trace_kernel or '轨迹'}），因为 NVML 代理口径不成立"
        return f"（估算 − 账本）/ 账本，同为「{basis}」口径；{note}"


def _window_seconds_for(coverage_pct: float, target_pct: float, window_s: float) -> float:
    """粗算"要把稳态循环延长到多少秒才能达到目标覆盖率"（给用户的预算提示）。

    模型：窗口 = 固定启动开销 O + 稳态段 S，覆盖率 = S/(O+S)。已知当前覆盖率 c 与窗口
    长度 W，则 O = W·(1−c/100)；要覆盖率达到 t 需要 S ≥ O·t/(100−t)，于是
    窗口总长 ≈ O + S。启动开销已由实测给出量级（`import torch` ≈ 9–10 s，见 README）。
    """
    if coverage_pct <= 0 or window_s <= 0 or target_pct >= 100:
        return 0.0
    overhead = window_s * (1.0 - coverage_pct / 100.0)
    steady = overhead * target_pct / (100.0 - target_pct)
    return overhead + steady


def _trim_window(samples: Sequence, mode: str, sampler_period_s: float) -> tuple[list, bool]:
    """按「首个/末个忙碌采样点」裁剪窗口（R-03）。

    为什么需要：教程的每个 bench 都是 Python + torch 脚本，`import torch` 等启动开销
    实测 ~9–10 秒。不去掉它，占空比会被启动开销稀释到 15%（实测），于是
    "能不能用代理"取决于"你要不要多租 37 秒的卡"，而不是取决于物理。
    """
    if mode != "auto" or not samples:
        return list(samples), False
    busy = [i for i, s in enumerate(samples)
            if float(getattr(s, "sm_util_pct", 0.0)) >= BUSY_SM_UTIL
            or float(getattr(s, "mem_util_pct", 0.0)) >= BUSY_MEM_UTIL]
    if not busy:
        return list(samples), False
    margin = max(1, int(round(TRIM_MARGIN_PERIODS))) if not sampler_period_s else 1
    lo = max(0, busy[0] - margin)
    hi = min(len(samples) - 1, busy[-1] + margin)
    if lo == 0 and hi == len(samples) - 1:
        return list(samples), False
    return list(samples[lo:hi + 1]), True


def estimate(samples: Sequence, peak_gbs: float, window_s: float,
             launches: int = 0, sampler_period_s: float = 0.0,
             ledger_per_launch: float | None = None,
             ledger_total: float | None = None,
             ledger_range: tuple[float, float] | None = None,
             peak_source: str = "",
             window_mode: str = "full",
             warmup_s: float = 0.0,
             window_limit_s: float | None = None,
             trace_bytes_per_launch: float | None = None,
             trace_kernel: str = "",
             trace_unavailable_reason: str = "") -> DramEstimate:
    """从 NVML 采样序列估算 DRAM 流量。

    samples: 具 `.t`(秒) / `.mem_util_pct` / `.sm_util_pct` 属性的采样点列表。
    window_mode: 'full' 用整段采样窗；'auto' 先按首个/末个忙碌采样点裁到稳态段。
    """
    est = DramEstimate(peak_gbs=peak_gbs, peak_source=peak_source,
                       window_s=window_s, launches=launches,
                       ledger_per_launch=ledger_per_launch, ledger_total=ledger_total,
                       ledger_range=ledger_range, sampler_period_s=sampler_period_s,
                       window_mode=window_mode,
                       trace_bytes_per_launch=trace_bytes_per_launch,
                       trace_kernel=trace_kernel,
                       trace_unavailable_reason=trace_unavailable_reason)
    samples = list(samples)
    if not samples:
        est.notes.append("没有采样点：NVML 采样被关闭或不可用")
        return est

    # ---- 采样间隔：相邻时间戳差分；首点沿用中位间隔 ----
    deltas: list[float] = []
    for a, b in zip(samples, samples[1:]):
        d = float(b.t) - float(a.t)
        if d > 0:
            deltas.append(d)
    deltas.sort()
    default_dt = deltas[len(deltas) // 2] if deltas else (sampler_period_s or 0.2)
    est.sampler_period_s = default_dt

    # ---- 窗口限定（R-03）：--ks-warmup / --ks-window-s / 自动裁剪 ----
    t_first = float(samples[0].t)
    if warmup_s > 0:
        samples = [s for s in samples if float(s.t) - t_first >= warmup_s] or [samples[-1]]
    if window_limit_s and window_limit_s > 0:
        t0 = float(samples[0].t)
        samples = [s for s in samples if float(s.t) - t0 <= window_limit_s] or [samples[0]]
    span_before_trim = float(samples[-1].t) - float(samples[0].t) if len(samples) > 1 else 0.0
    samples, trimmed = _trim_window(samples, window_mode, default_dt)
    est.trimmed = trimmed

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

    est.n_samples = len(samples)
    est.window_used_s = total_time
    est.window_dropped_s = max(0.0, (window_s or span_before_trim) - total_time)
    est.mem_util_pct_mean = sum(busy_pcts) / len(busy_pcts)
    est.window_coverage_pct = 100.0 * busy_samples / len(samples)
    est.bytes_total = busy_time * peak_gbs * 1e9
    est.throughput_pct_of_peak = est.mem_util_pct_mean
    if launches > 0:
        est.bytes_per_launch = est.bytes_total / launches

    # ---- 说明文字（原因文本由 reason_for 生成，这里只补口径与边界）----
    if peak_gbs <= 0:
        est.notes.append("缺少实测峰值带宽：加 --ks-calibration machine-profile.json 或 --ks-peak-gbps")
    if est.trimmed:
        dropped = max(0.0, window_s - total_time)
        est.notes.append(
            f"窗口已按首个/末个忙碌采样点裁到稳态段（--ks-window auto）："
            f"参与积分 {total_time:.1f} s，裁掉约 {dropped:.1f} s 启动/JIT/空闲"
            f"；用 --ks-window full 可关闭")
    if est.duty_pct < DUTY_WEAK:
        est.notes.append(
            f"窗口覆盖率 {est.window_coverage_pct:.0f}%（SM≥{BUSY_SM_UTIL:.0f}% 或显存≥"
            f"{BUSY_MEM_UTIL:.0f}% 的采样点占比）")
    if total_time and window_s and total_time < 0.5 * window_s:
        est.notes.append("采样覆盖时间不足窗口一半：采样器可能启动过晚或提前退出")
    est.notes.append(
        "采样周期 %.0f ms、区间平均（utilization.memory 分辨率 %.0f 个百分点）；"
        "字节 = Σ(mem_util×Δt)×实测峰值带宽" % (est.sampler_period_s * 1000,
                                                est.mem_util_resolution_pct))
    return est


def to_metrics(est: DramEstimate, basis: str = "auto") -> list[tuple[str, float | None, str, str, str, str]]:
    """给 section 组装用的 (name, value, unit, source, note, resolution) 列表。

    **value 为 None 的条目也必须被组装方输出**（作为 unavailable + 原因）——
    这正是 B1 的修法：口径不成立时那一行要"以 n/a 的形式存在"，而不是消失。

    provenance 由调用方决定（本模块永远产出 estimated）。
    """
    out: list[tuple[str, float | None, str, str, str, str]] = []
    note = "；".join(est.notes)
    out.append(("est__dram_bytes_total",
                round(est.bytes_total / 1e6, 1) if est.reliable else None, "MB",
                "NVML memory-busy time × measured peak bandwidth（对应 ncu dram__bytes.sum）",
                note, f"{est.mem_util_resolution_pct:.0f} %"))
    basis_used = est.resolve_basis(basis)
    out.append(("est__dram_bytes_per_launch",
                round(est.bytes_per_launch / 1e6, 2)
                if (est.bytes_per_launch is not None and est.attributable) else None, "MB",
                f"total / {est.launches} launches（对应 ncu dram__bytes.sum 的按 launch 口径）",
                "仅在『单一 kernel 主导采样窗口 + 覆盖率 ≥80%』时成立", ""))
    out.append(("est__dram_throughput_pct_of_peak",
                round(est.throughput_pct_of_peak, 1) if est.reliable else None, "%",
                "NVML memory-busy 占比（对应 ncu dram__throughput.avg.pct_of_peak_sustained_elapsed）",
                "忙碌时间占比口径，与按字节/时间的口径略有差异", f"{est.mem_util_resolution_pct:.0f} %"))
    # 前提指标：**永远出现**（R-15/B5：两个语义分开报）
    out.append(("ks__sampler_window_coverage_pct", round(est.window_coverage_pct, 1), "%",
                f"采样窗中 SM≥{BUSY_SM_UTIL:.0f}% 或显存≥{BUSY_MEM_UTIL:.0f}% 的采样点占比",
                f"前提①「窗口被被测负载主导」：<{DUTY_WEAK:.0f}% 不给 per-launch 归因，"
                f"<{DUTY_UNUSABLE:.0f}% 连总量也不给", "1 个采样点"))
    out.append(("est__dram_bytes_per_launch_trace",
                round(est.trace_bytes_per_launch / 1e6, 3) if est.trace_available else None, "MB",
                "访存轨迹模型（L2 miss sector × 32B × grid/blocks_traced / launches）",
                ("第二条口径：与 NVML 代理独立；NVML 口径不成立时用它做账本对拍"
                 if est.trace_available else est.trace_unavailable_reason), ""))
    if est.ledger_basis:
        ref = est.reference_bytes()
        b = "每次 launch" if est.ledger_basis == "per-launch" else "整窗口总量"
        out.append(("est__dram_vs_ledger_err_pct",
                    round(est.ledger_err_pct_for(basis), 1)
                    if est.ledger_err_pct_for(basis) is not None else None, "%",
                    f"（估算 − 账本）/ 账本，同为「{b}」口径；实际口径："
                    f"{est.basis_label(basis)}",
                    f"账本 {ref / 1e6:.1f} MB；对拍常用 |误差| < 10% 作为验收线"
                    + ("；此次用**轨迹口径**，因为 NVML 代理口径不成立"
                       if basis_used.startswith("trace") else ""), ""))
    if est.ledger_range:
        out.append(("est__dram_range_position_pct",
                    round(est.range_position_pct_for(basis), 1)
                    if est.range_position_pct_for(basis) is not None else None, "%",
                    f"在账本区间 [{est.ledger_range[0] / 1e6:.1f}, {est.ledger_range[1] / 1e6:.1f}] MB "
                    f"（per-launch）中的相对位置；实际口径：{est.basis_label(basis)}",
                    est.range_verdict_for(basis), ""))
        if est.range_verdict_for(basis):
            out.append(("ks__ledger_range_verdict", est.range_verdict_for(basis), "",
                        "由区间位置与两条口径推出（R-05）",
                        "靠下界 ⇒ 账本预测的流量偏大/O(S²) 口径成立；靠上界 ⇒ 账本可能漏项", ""))
    return out
