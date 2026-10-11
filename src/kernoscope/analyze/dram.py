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

from ..refusals import Refusal, RefusalCode, Remedy, combined_refusal

# 窗口覆盖率阈值：低于 UNUSABLE 直接判为不可用；低于 WEAK 不给 per-launch 归因
DUTY_UNUSABLE = 50.0
DUTY_WEAK = 80.0
# R-46：per-launch 归因需要**窗口被被匹配的那个 kernel 主导**（与覆盖率是两个前提）。
# 阈值由验收线反推，不是拍脑袋：把窗口总量除到某个 kernel 的 launch 数上，误差因子是
# 1/s（s = 该 kernel 占窗口的份额），于是归因本身引入的上偏 = 1/s − 1。
#   s=80% → +25%（远超 |误差|<10% 的验收线，所以 80 这个数守不住 10% 的线）
#   s=90% → +11.1%
#   s=95% →  +5.3%  ← 与本代理自身 −4%~+13% 的噪声同量级，取它作为门槛
#   s=99% →  +1.0%
SHARE_STRONG = 95.0
# 数「窗口里有几段连续忙碌期」时，连续多少个空闲采样点才算分段（避免把采样的偶发丢点
# 当成新阶段）：2 个 ≥ 400 ms。
SEGMENT_MIN_GAP = 2
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
    # ---- R-47：per-launch 的**分母**必须和分子覆盖同一个窗口 ----
    # 旧实现：分子 = 窗口内字节（bytes_total 只覆盖 window_used_s），分母 = **整段运行的**
    # launch 数（cli.py 传进来的 launches）。窗口一被裁，per-launch 就等比偏小：
    # 真机实测同一负载 auto/warmup40/window-s20 三种窗口给出 235.08 / 130.61 / 45.84 MB，
    # 而 share=100%、bias=+0%、无告警、rc=0 —— 账本是 per-launch 口径，于是使用者会得到
    # 「账本错了 44%/80%」这个完全错误的结论。所以：窗口被裁过时，分母必须按**窗口内**
    # 的等效 launch 数（按时间重叠加权）重算；算不出来就不给这个数。
    launches_total: int = 0                 # 整段运行里匹配到的 launch 数
    launches_in_window: float | None = None  # 窗口内的等效 launch 数（重叠加权）
    window_t0: float | None = None          # 积分窗口的绝对边界（采样时钟）
    window_t1: float | None = None
    sample_t0: float | None = None          # 采样覆盖的边界（裁窗口**之前**）
    sample_t1: float | None = None
    launch_clock_aligned: bool | None = None  # None=没给 launch 时间戳；False=两个时钟对不上
    launch_align_offset_s: float | None = None   # 推导出的偏移（采样时钟 − launch 时钟）
    launch_align_slope: float | None = None      # 两个时钟的速率比（自检：接近 1 才可信）
    launches_in_window_range: tuple[float, float] | None = None  # offset ±1 采样周期的计数区间
    launch_spans_mapped: list[tuple[float, float]] | None = None  # 对齐到采样时钟后的区间（内部用）
    window_bias_pct: float | None = None    # 100×(window_used_s/window_s − 1)，可算的偏差估计
    # ---- R-46：窗口主导性（per-launch 归因的第二个前提）----
    # 被匹配 kernel 的 GPU 时间（ns，来自 CUPTI 的逐 launch 时长求和）。
    # 分母是**参与积分的窗口长度** window_used_s。窗口被 --ks-window auto 裁短时，
    # 分子仍含窗口外的同名 launch —— 所以这是个**上界**（share_upper_bound=True），
    # 上界用在"放行"方向是不保守的，因此当窗口确实被裁剪时会额外提示。
    matched_kernel_ns: int = 0
    other_kernels_ns: int = 0
    other_kernel_count: int = 0
    window_busy_segments: int = 0
    share_upper_bound: bool = False
    # 轨迹口径自己的分辨率（采样比/全量），由调用方填
    trace_resolution: str = ""
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
    def window_used_ns(self) -> int:
        """参与积分的窗口长度（ns）。"""
        return int(round(self.window_used_s * 1e9))

    @property
    def share_ready(self) -> bool:
        """有没有判定窗口主导性所需的数据（被匹配 kernel 的 GPU 时间）。"""
        return self.matched_kernel_ns > 0 and self.window_used_s > 0

    @property
    def matched_kernel_window_share_pct(self) -> float | None:
        """被匹配 kernel 的 GPU 时间 ÷ 参与积分的窗口长度（R-46，与覆盖率同量纲）。

        注意它**不是**判据：空闲时间不产生字节，所以把分母取成窗口长度会冤枉
        「单 kernel + 窗口里有空闲采样」的正当场景。判据用的是下面那个内核时间份额。
        """
        if not self.share_ready:
            return None
        return min(100.0, 100.0 * self.matched_kernel_ns / self.window_used_ns)

    @property
    def matched_kernel_kernel_share_pct(self) -> float | None:
        """被匹配 kernel 的 GPU 时间 ÷ 窗口内**所有 kernel** 的 GPU 时间（R-46 的判据）。

        为什么用这个当判据：字节估算的积分里，空闲时段贡献 0 字节，真正会污染
        「总量 ÷ 某 kernel 的 launch 数」的是**其他 kernel 干的活**。所以主导性问的是
        「窗口里的 kernel 时间里，被匹配的那个占多少」，而不是「它占了多少墙钟」。
        单 kernel 的窗口无论有多少空闲，这个值都是 100%。

        它是**字节份额的代理**（假设各 kernel 的字节强度可比）：计算机密负载与访存密集
        负载混在一个窗口里时，它可能高估或低估真实的字节份额 —— 这也是为什么
        `--ks-memtrace` 的轨迹口径（只测被追踪的那个 kernel）才是混合窗口下唯一诚实的口径。
        """
        denom = self.matched_kernel_ns + self.other_kernels_ns
        if denom <= 0:
            return None
        return 100.0 * self.matched_kernel_ns / denom

    @property
    def other_kernel_window_share_pct(self) -> float | None:
        """窗口里其他 kernel 占的份额（原因文本要点名这个数）。"""
        if not self.share_ready:
            return None
        return min(100.0, 100.0 * self.other_kernels_ns / self.window_used_ns)

    @property
    def attribution_bias_estimate_pct(self) -> float | None:
        """归因本身带来的上偏估计 = 100×(1/s − 1)，s 取内核时间份额。

        它把「为什么这个数字可能是错的」变成可算的一步：s=44.9% → +123%。
        实测那次是 +146.9%：**略高于估计**，因为 k_fused 的字节密度低于窗口里其他
        kernel。所以这是估计/代理，不是严格上界 —— 混合强度的窗口必须换轨迹口径。
        """
        s = self.matched_kernel_kernel_share_pct
        if s is None or s <= 0:
            return None
        return 100.0 * (1.0 / (s / 100.0) - 1.0)

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
        """能否把总量归因到单个 kernel。

        per-launch 口径要求窗口**同时**满足三个前提：
          ① 窗口被被测负载主导（覆盖率 ≥ DUTY_WEAK）—— GPU 在不在干活；
          ② 窗口被**被匹配的那个** kernel 主导（份额 ≥ SHARE_STRONG）—— 它是不是主角；
          ③ **分母可信**：窗口没被裁过，或裁过但窗口内的等效 launch 数算得出来（R-47）。
        只有 ① 时，一个窗口里顺序跑了多个 kernel/变体的负载会被误判为可归因（R-46）。
        少了 ③ 时，分子是窗口内字节、分母是整段 launch 数，数字会等比偏小（D5）。
        """
        share = self.matched_kernel_kernel_share_pct
        return (self.reliable and self.window_coverage_pct >= DUTY_WEAK
                and self.launches > 0 and self.bytes_per_launch is not None
                and share is not None and share >= SHARE_STRONG)

    def per_launch_verdict(self) -> tuple[bool, str]:
        """(能否给 per-launch 数字, 原因) —— section、摘要、--strict 三处共用同一个判定。

        D3 的教训：判定被三处各自解释时，摘要会说「口径不成立」而同一行又打印出估算值。
        所以「原因文本」也只在这里生成一次。
        """
        if self.attributable:
            return True, ""
        return False, self.reason_for("est__dram_bytes_per_launch")

    @property
    def trace_available(self) -> bool:
        return self.trace_bytes_per_launch is not None and self.trace_bytes_per_launch > 0

    @property
    def trace_total(self) -> float | None:
        """轨迹口径的整窗口总量 = per-launch × 已匹配 launch 数（只在单 kernel 时成立）。"""
        if not self.trace_available or self.launches <= 0:
            return None
        return self.trace_bytes_per_launch * self.launches

    # ---- 拒绝给数的原因：点明哪一个前提没成立（R-32）+ 结构化拒绝（W3）----
    def _total_refusals(self) -> list[Refusal]:
        out: list[Refusal] = []
        if self.n_samples <= 0:
            out.append(Refusal(
                RefusalCode.NO_NVML_SAMPLES,
                "没有 NVML 采样点：采样被 --ks-no-nvml 关掉、采样器不可用，"
                "或这次运行的稳态段不足一个采样周期（200 ms）",
                observed={"nvml_samples": self.n_samples},
                remedy=Remedy("确认设备级采样开着（不要加 --ks-no-nvml），并把稳态段"
                              "跑长到至少几个采样周期（每个 200 ms）",
                              command="kernoscope --ks-window auto -- python bench_loop.py",
                              cost_s=20.0, then_answers=("dram_bytes", "device_busy"))))
        if self.peak_missing:
            out.append(Refusal(
                RefusalCode.NO_CALIBRATION,
                "缺少本机实测峰值带宽：先跑 tools/machine-profile.py，"
                "再加 --ks-calibration machine-profile.json（或 --ks-peak-gbps）",
                observed={"peak_gbs": self.peak_gbs}))
        if self.n_samples > 0 and self.below_resolution:
            out.append(Refusal(
                RefusalCode.BELOW_RESOLUTION,
                f"显存忙碌读数 {self.mem_util_pct_mean:.1f}% 低于 NVML utilization.memory 的"
                f"分辨率下限 {self.mem_util_resolution_pct:.0f} 个百分点："
                f"无法区分『真的没搬』与『{self.mem_util_resolution_pct:.1f}% 被截断成 0』",
                observed={"mem_busy_pct": round(self.mem_util_pct_mean, 3)},
                threshold={"resolution_pct": self.mem_util_resolution_pct}))
        if (self.n_samples > 0 and not self.peak_missing and not self.below_resolution
                and self.window_coverage_pct < DUTY_UNUSABLE):
            need = _window_seconds_for(self.window_coverage_pct, DUTY_UNUSABLE,
                                       self.window_used_s or self.window_s)
            out.append(Refusal(
                RefusalCode.COVERAGE_TOO_LOW,
                f"窗口覆盖率 {_fmt_pct_cmp(self.window_coverage_pct, DUTY_UNUSABLE)}："
                f"窗口没被被测负载主导，忙碌时段是否饱和无从确认"
                + (f"；把稳态循环延长到约 {need:.0f} 秒，或加 --ks-window auto 只统计稳态段"
                   if need else ""),
                observed={"coverage_pct": round(self.window_coverage_pct, 1)},
                threshold={"min_pct": DUTY_UNUSABLE},
                remedy=Remedy("把稳态循环延长到约 "
                              f"{need:.0f} 秒，或加 --ks-window auto 只统计稳态段",
                              command="kernoscope --ks-window auto -- python bench_loop.py",
                              cost_s=float(need or 0.0),
                              then_answers=("dram_bytes",)) if need else None))
        return out

    def _share_refusal(self) -> Refusal:
        """R-46：窗口主导性前提不成立时，点名份额与窗口里的其他 kernel。"""
        share = self.matched_kernel_kernel_share_pct
        if share is None:
            return Refusal(
                RefusalCode.SHARE_UNKNOWN,
                "无法判定窗口主导性：报告里没有被匹配 kernel 的 GPU 时间"
                "（逐 launch 时长为空），per-launch 归因不可验证",
                observed={"matched_kernel_ns": self.matched_kernel_ns})
        others = self.other_kernel_window_share_pct
        detail = ""
        if self.other_kernel_count:
            detail = (f"；窗口里还有 {self.other_kernel_count} 个 kernel"
                      + (f"，按墙钟占 {others:.0f}%" if others is not None else ""))
        bias = self.attribution_bias_estimate_pct
        bound = (f"归因本身的上偏估计 +{bias:.0f}%" if bias is not None else "归因偏差未知")
        return Refusal(
            RefusalCode.SHARE_TOO_LOW,
            f"窗口主导性不足：被匹配的 kernel 只占窗口 kernel 时间的 "
            f"{_fmt_pct_cmp(share, SHARE_STRONG, digits=1)}，{bound}{detail}；"
            f"份额是按**整段运行的 launch 表**算的，缩采样窗口改不动它 —— "
            f"要么让被测区间里只有这一个 kernel，要么改用 --ks-ledger-total-mb 走整窗口口径，"
            f"要么加 --ks-memtrace 换用**轨迹口径**（只测被追踪 kernel，不受窗口构成影响）",
            observed={"matched_kernel_kernel_share_pct": round(share, 1),
                      "other_kernel_count": self.other_kernel_count,
                      "matched_kernel_ns": self.matched_kernel_ns,
                      "other_kernels_ns": self.other_kernels_ns},
            threshold={"share_pct": SHARE_STRONG})

    def _total_failures(self) -> list[str]:
        return [r.message for r in self._total_refusals()]

    def _attribution_refusals(self) -> list[Refusal]:
        out = self._total_refusals()
        if self.launches <= 0:
            out.append(Refusal(
                RefusalCode.MULTI_KERNEL_WINDOW,
                "未锁定单一 kernel（匹配到多个 kernel）：per-launch 口径不适用，"
                "请用 -k 只选一个 kernel",
                observed={"matched_kernels_in_window": self.other_kernel_count}))
            return out
        if self.reliable and self.window_coverage_pct < DUTY_WEAK:
            need = _window_seconds_for(self.window_coverage_pct, DUTY_WEAK,
                                       self.window_used_s or self.window_s)
            out.append(Refusal(
                RefusalCode.COVERAGE_TOO_LOW,
                f"窗口覆盖率 {_fmt_pct_cmp(self.window_coverage_pct, DUTY_WEAK)}：无法把 "
                f"{self.bytes_total / 1e6:.0f} MB 总量归因到每个 launch"
                + (f"；把稳态循环延长到约 {need:.0f} 秒，或加 --ks-window auto" if need else ""),
                observed={"coverage_pct": round(self.window_coverage_pct, 1)},
                threshold={"min_pct": DUTY_WEAK}))
        share = self.matched_kernel_kernel_share_pct
        if self.reliable and (share is None or share < SHARE_STRONG):
            out.append(self._share_refusal())
            return out
        # 份额成立、但分母不可信：这才是"窗口被裁过"那种情况的真正原因。放在份额之后，
        # 是因为混合窗口下份额才是主因（D1 的验收要的正是 share_too_low），
        # 而"缩窗口改不动份额"这件事已写进 SHARE_TOO_LOW 的处方里。
        if self.reliable and self.launches > 0 and self.bytes_per_launch is None:
            if self.launch_clock_aligned is None:
                why = "没有 launch 时间戳（拿不到逐 launch 的起止时间）"
            elif self.launch_align_slope is not None and not 0.9 <= self.launch_align_slope <= 1.1:
                why = (f"launch 时间戳与采样时钟的**速率比 {self.launch_align_slope:.3f}**"
                       f"（应当≈1）：两者不同源或被缩放，不能相除")
            else:
                why = "launch 时间戳与采样时钟对不上，且无法用忙碌段推出偏移"
            bias = self.window_bias_pct
            est_bias = (f"按窗口长度粗算，这个数字本来会偏小约 {abs(bias):.0f}%；"
                        if bias not in (None, 0.0) else "")
            out.append(Refusal(
                RefusalCode.WINDOW_LAUNCH_UNALIGNED,
                f"窗口被裁过，而窗口内的 launch 数算不出来（{why}）：per-launch 的分子是"
                f"窗口内字节、分母必须是**同一个窗口内**的 launch 数，否则整个数字等比偏小。"
                f"{est_bias}改用 --ks-window full 让两者同覆盖，或走整窗口口径"
                f"（--ks-ledger-total-mb）/ 轨迹口径（--ks-memtrace）—— 后两条不需要这个分母",
                observed={"launches_total": self.launches_total,
                          "launches_in_window": None,
                          "window_used_s": round(self.window_used_s, 2),
                          "window_bias_pct": (round(bias, 1) if bias is not None else None),
                          "launch_align_slope": (round(self.launch_align_slope, 4)
                                                 if self.launch_align_slope is not None else None)},
                threshold={"launch_align_slope": 1.0}))
        return out

    def refusal_for(self, name: str, basis: str = "auto") -> Refusal:
        """某个指标为什么是 n/a —— 结构化版本（W3）：code + observed + threshold + remedy。"""
        if name in ("est__dram_bytes_total",):
            return self._total_refusal()
        if name in ("est__dram_bytes_per_launch",):
            return self._attribution_refusal()
        if name in ("est__dram_throughput_pct_of_peak",):
            return self._total_refusal("无法给出吞吐占比（未知原因）")
        if name == "est__dram_bytes_per_launch_trace":
            return self._trace_refusal()
        if name == "est__dram_vs_ledger_err_pct":
            return self._ledger_refusal(basis, need="误差")
        if name == "est__dram_range_position_pct":
            return self._ledger_refusal(basis, need="区间位置", want_total=False)
        if name in ("ks__matched_kernel_window_share_pct",
                    "ks__matched_kernel_kernel_share_pct"):
            return self._share_refusal()
        if name == "ks__attribution_bias_estimate_pct":
            inner = self._share_refusal()
            return Refusal(inner.code, "无法给出归因上偏估计：" + inner.message,
                           observed=dict(inner.observed), threshold=dict(inner.threshold),
                           remedy=inner.remedy, severity=inner.severity)
        if name == "ks__sampler_window_coverage_pct":
            if self.n_samples <= 0:
                return Refusal(
                    RefusalCode.NO_NVML_SAMPLES,
                    "没有 NVML 采样点（--ks-no-nvml 打开，或采样器不可用）：覆盖率无定义",
                    observed={"nvml_samples": self.n_samples})
            return Refusal(RefusalCode.INTERNAL_UNKNOWN,
                           "；".join(self.notes) or "覆盖率不可得")
        if name in ("ks__dram_window_mem_busy_mean_pct", "ks__window_busy_segments"):
            return Refusal(
                RefusalCode.NO_NVML_SAMPLES,
                "没有 NVML 采样点（采样被关掉／不可用／稳态段短于一个 200 ms 采样周期）："
                "窗口内统计量无定义",
                observed={"nvml_samples": self.n_samples})
        return Refusal(RefusalCode.INTERNAL_UNKNOWN, "；".join(self.notes) or "无法估算")

    def _total_refusal(self, fallback: str = "无法估算（未知原因）") -> Refusal:
        rs = self._total_refusals()
        return combined_refusal(rs) if rs else Refusal(RefusalCode.INTERNAL_UNKNOWN, fallback)

    def _attribution_refusal(self) -> Refusal:
        rs = self._attribution_refusals()
        return combined_refusal(rs) if rs else \
            Refusal(RefusalCode.INTERNAL_UNKNOWN, "无法归因到单个 launch（未知原因）")

    def _trace_refusal(self) -> Refusal:
        return Refusal(
            RefusalCode.TRACE_UNAVAILABLE,
            self.trace_unavailable_reason or "轨迹未提供 per-launch 字节",
            observed={"trace_bytes_per_launch": self.trace_bytes_per_launch})

    def _ledger_refusal(self, basis: str, need: str, want_total: bool | None = None) -> Refusal:
        inner = (self._total_refusal() if want_total else self._attribution_refusal())
        hint = ("；可以加 --ks-memtrace <trace> 换用**轨迹口径**的 per-launch 字节"
                if not self.trace_available else "")
        return Refusal(
            inner.code, f"{need}行无法给出：{inner.message or '对拍口径不成立'}{hint}",
            observed=dict(inner.observed), threshold=dict(inner.threshold),
            remedy=inner.remedy, severity=inner.severity, causes=inner.causes)

    def reason_for(self, name: str, basis: str = "auto") -> str:
        """散文版本 —— 与 `refusal_for` 同一来源（不允许两处各自解释，D3 的教训）。"""
        return self.refusal_for(name, basis).message

    def _ledger_failure(self, basis: str, need: str, want_total: bool | None = None) -> str:
        return self._ledger_refusal(basis, need, want_total).message

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


def _busy_segments(samples: Sequence) -> int:
    """窗口里有几段「连续忙碌期」（R-46 的辅助信号，不是硬门槛）。

    段与段之间要有 ≥ SEGMENT_MIN_GAP 个连续空闲采样点才算分开 —— 否则会把采样的
    偶发丢点当成新阶段。「依次跑 v1/v2/v3」这类 bench 在 --ks-window auto 下会表现
    为多段；这时 per-launch 归因要格外谨慎，所以把它报出来而不是替用户判断。
    """
    segs = 0
    gap = 0
    in_seg = False
    for s in samples:
        busy = (float(getattr(s, "sm_util_pct", 0.0)) >= BUSY_SM_UTIL
                or float(getattr(s, "mem_util_pct", 0.0)) >= BUSY_MEM_UTIL)
        if busy:
            if not in_seg:
                segs += 1
                in_seg = True
            gap = 0
        else:
            gap += 1
            if in_seg and gap >= SEGMENT_MIN_GAP:
                in_seg = False
    return segs


def _fmt_pct_cmp(value: float, threshold: float, digits: int = 0) -> str:
    """「X% < Y%」的文案（D2）。

    曾经写死 `:.0f`：覆盖率 79.9% 会被印成「窗口覆盖率 80% < 80%」，读者会以为工具坏了。
    四舍五入后两者相等就逐级加小数位，直到能区分开。
    """
    for d in (digits, digits + 1, digits + 2):
        v, t = f"{value:.{d}f}", f"{threshold:.{d}f}"
        if v != t:
            return f"{v}% < {t}%"
    return f"{value:.2f}% < {threshold:.2f}%"


def _fmt_launches(n: float | None) -> str:
    """等效 launch 数的人读形式：整数就不带小数点（跨窗口边界的会是小数）。"""
    if n is None:
        return "不可用"
    return f"{n:.0f}" if abs(n - round(n)) < 1e-6 else f"{n:.2f}"


def _count_in_window(spans: Sequence[tuple[float, float]], t0: float, t1: float) -> float:
    """窗口内的**等效 launch 数**：跨边界的 launch 只算落在窗口里的那一段。

    为什么按重叠加权而不是"数落在窗口里的 launch 个数"：分子的积分就是窗口那一段的字节，
    所以分母也应当是窗口那一段对应的 launch 数（跨边界的一半就是半个 launch）。
    """
    total = 0.0
    for a, b in spans:
        if b <= a:                       # 零长度（坏数据）：按起点判
            if t0 <= a <= t1:
                total += 1.0
            continue
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            total += (hi - lo) / (b - a)
    return total


def _launches_in_window(spans: Sequence[tuple[float, float]] | None,
                        t0: float | None, t1: float | None,
                        sample_t0: float | None, sample_t1: float | None,
                        busy_lo: float | None, busy_hi: float | None, tol: float
                        ) -> tuple[float | None, bool | None, float | None, float | None,
                                   list[tuple[float, float]] | None]:
    """窗口内的等效 launch 数 + 对齐的自检信息。返回 `(count, aligned, offset_s, slope)`。

    为什么不能假设"launch 时间戳与采样同域"：**真机实测**（4090 + nsys 2025.3）里
    `CUPTI_ACTIVITY_KIND_KERNEL.start` 是**相对 profiling 会话起点**的（第一次 launch ≈ 2.67 s），
    而 `DeviceSample.t` 是 `time.monotonic()`（开机秒）—— 两者相差一个未知常数。直接相减
    会把整段 launch 表判成"在采样窗外"，于是要么给错、要么全部拒绝。

    所以这里**推导**偏移，再**自检**：

    1. 若 launch 表本来就落在采样覆盖范围内 ⇒ 它已经在同一时钟域（有些 nsys 版本/导入路径
       给的是绝对时间），直接用；
    2. 否则用"窗口内的忙碌段"当锚点：忙碌采样点与这些 launch 覆盖的是**同一段物理时间**，
       于是 `offset = busy_lo − launch_lo`（误差 ≤ 1 个采样周期）；
    3. **速率必须一致**：`slope = (busy 长度) / (launch 表长度)` 必须落在 [0.9, 1.1]。
       两个时钟若不同源或有一方被缩放，这一条会失败 —— 那就拒绝，而不是把两个域相除。

    采样周期带来的 offset 不确定度由调用方按 `±1 周期` 额外算一次计数（见 `estimate`），
    作为分母的分辨率报出去。
    """
    if not spans:
        return None, None, None, None, None
    if t0 is None or t1 is None or sample_t0 is None or sample_t1 is None:
        return None, None, None, None
    lo_ok, hi_ok = sample_t0 - tol, sample_t1 + tol
    if all(lo_ok <= a and b <= hi_ok for a, b in spans):
        if not any(b >= sample_t0 and a <= sample_t1 for a, b in spans):
            return None, False, None, None, None
        return _count_in_window(spans, t0, t1), True, 0.0, 1.0, list(spans)
    if busy_lo is None or busy_hi is None or busy_hi - busy_lo <= 0:
        return None, False, None, None, None
    l_lo = min(a for a, _ in spans)
    l_hi = max(b for _, b in spans)
    if l_hi - l_lo <= 0:
        return None, False, None, None, None
    slope = (busy_hi - busy_lo) / (l_hi - l_lo)
    if not 0.9 <= slope <= 1.1:
        return None, False, None, slope, None
    offset = busy_lo - l_lo
    mapped = [(busy_lo + (a - l_lo) * slope, busy_lo + (b - l_lo) * slope) for a, b in spans]
    return _count_in_window(mapped, t0, t1), True, offset, slope, mapped


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
             trace_unavailable_reason: str = "",
             matched_kernel_ns: int = 0,
             other_kernels_ns: int = 0,
             other_kernel_count: int = 0,
             launch_spans: Sequence[tuple[float, float]] | None = None,
             trace_resolution: str = "") -> DramEstimate:
    """从 NVML 采样序列估算 DRAM 流量。

    samples: 具 `.t`(秒) / `.mem_util_pct` / `.sm_util_pct` 属性的采样点列表。
    window_mode: 'full' 用整段采样窗；'auto' 先按首个/末个忙碌采样点裁到稳态段。
    matched_kernel_ns: 被 `-k` 匹配的 kernel 的 GPU 时间（R-46 的分子）。
    launch_spans: 被匹配 launch 的 (start_s, end_s) 绝对时间（**与 samples 同一时钟**，
        即 `time.monotonic()` / CUPTI 的 CLOCK_MONOTONIC）。R-47 用它把 per-launch 的
        分母对齐到积分窗口内。给不出时，窗口一旦被裁就只能拒绝这个口径。
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
    # 对齐 launch 时间戳要用的锚点必须是**整段**采样里的忙碌段（不能取被裁后的窗口：
    # --ks-warmup 会把窗口裁短，而 launch 表仍是整段，两者长度不可比 —— R-47 真机教训）。
    raw_busy_ts = [float(s.t) for s in samples
                   if float(s.sm_util_pct) >= BUSY_SM_UTIL or float(s.mem_util_pct) >= BUSY_MEM_UTIL]

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
    # 采样器覆盖的整段（对上 launch 时间戳要用它判断两个时钟对不对得上，R-47）
    sample_t0 = float(samples[0].t)
    span_before_trim = float(samples[-1].t) - sample_t0 if len(samples) > 1 else 0.0
    sample_t1 = float(samples[-1].t) + default_dt
    t_first = float(samples[0].t)
    if warmup_s > 0:
        samples = [s for s in samples if float(s.t) - t_first >= warmup_s] or [samples[-1]]
    if window_limit_s and window_limit_s > 0:
        t0 = float(samples[0].t)
        samples = [s for s in samples if float(s.t) - t0 <= window_limit_s] or [samples[0]]
    span_before_trim = float(samples[-1].t) - float(samples[0].t) if len(samples) > 1 else 0.0
    samples, trimmed = _trim_window(samples, window_mode, default_dt)
    est.trimmed = trimmed
    est.launches_total = int(launches)
    est.sample_t0, est.sample_t1 = sample_t0, sample_t1
    est.window_t0 = float(samples[0].t)
    est.window_t1 = float(samples[-1].t) + default_dt
    window_cut = bool(trimmed or warmup_s > 0 or (window_limit_s or 0) > 0)
    est.matched_kernel_ns = int(matched_kernel_ns)
    est.other_kernels_ns = int(other_kernels_ns)
    est.other_kernel_count = int(other_kernel_count)
    est.trace_resolution = trace_resolution
    # 分子（被匹配 kernel 的 GPU 时间）是**整个运行**的；窗口被裁短时它含窗口外的同名
    # launch，于是份额偏大 —— 这在「放行」方向上不保守，所以标出来。
    est.share_upper_bound = bool(trimmed or warmup_s > 0 or (window_limit_s or 0) > 0)

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
    est.window_busy_segments = _busy_segments(samples)
    est.bytes_total = busy_time * peak_gbs * 1e9
    est.throughput_pct_of_peak = est.mem_util_pct_mean
    # 整段采样里"忙碌"的起止：把 launch 时间戳对齐到采样时钟的锚点（R-47）。
    # 忙碌 = 有 kernel 在跑，所以它与（整段运行的）launch 表覆盖同一段物理时间。
    busy_lo = raw_busy_ts[0] if raw_busy_ts else None
    busy_hi = (raw_busy_ts[-1] + default_dt) if raw_busy_ts else None
    # 可算的偏差估计：窗口被裁掉多少，per-launch 就会等比偏小多少（R-47 要报这个数）
    if window_s and window_s > 0:
        est.window_bias_pct = 100.0 * (total_time / window_s - 1.0)
    if window_cut:
        # 分母必须落在窗口内。窗口内的等效 launch 数按时间重叠加权（跨边界的 launch
        # 只算它落在窗口里的那一段），否则分子是窗口内字节、分母是整段 launch 数。
        tol_s = max(5.0 * default_dt, 1.0)
        (est.launches_in_window, est.launch_clock_aligned,
         est.launch_align_offset_s, est.launch_align_slope, mapped_spans) = _launches_in_window(
            launch_spans, est.window_t0, est.window_t1,
            sample_t0, sample_t1, busy_lo, busy_hi, tol=tol_s)
        est.launch_spans_mapped = mapped_spans
        if est.launches_in_window is not None and est.launches_in_window > 0:
            est.bytes_per_launch = est.bytes_total / est.launches_in_window
            # 分母的分辨率：锚点（首个忙碌采样点）的精度是 ±1 个采样周期，
            # 据此把整个对齐后的 launch 表左右各挪一周期，给出计数的区间。
            if mapped_spans:
                def _shifted(d: float) -> list[tuple[float, float]]:
                    return [(a + d, b + d) for a, b in mapped_spans]

                a = _count_in_window(_shifted(-default_dt), est.window_t0, est.window_t1)
                b = _count_in_window(_shifted(+default_dt), est.window_t0, est.window_t1)
                # 区间必须**包住**中点估计（偏移挪前/挪后各算一次，谁大谁小取决于窗口边界）
                est.launches_in_window_range = (round(min(a, b), 2), round(max(a, b), 2))
        if est.launches_in_window is not None and est.launches_in_window > 0:
            est.bytes_per_launch = est.bytes_total / est.launches_in_window
    elif launches > 0:
        # 窗口没被裁：分子与分母都覆盖整段运行，直接用整段 launch 数
        est.launches_in_window = float(launches)
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
    if est.window_busy_segments > 1:
        est.notes.append(
            f"窗口里有 {est.window_busy_segments} 段连续忙碌期（段间 ≥{SEGMENT_MIN_GAP} 个空闲"
            f"采样点）：可能是多个阶段/变体拼接，per-launch 归因需谨慎（见 ks__window_busy_segments）")
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
    share_w = est.matched_kernel_window_share_pct
    share_k = est.matched_kernel_kernel_share_pct
    bias = est.attribution_bias_estimate_pct
    # R-41：派生量也要带自己的分辨率/不确定度。per-launch 与由它推出的误差、区间位置的
    # 分辨率就是**归因上偏的上界**（这是它们真正的精度地板，不是 mem_util 的 1 个百分点）。
    attribution_res = (f"归因上偏估计 +{bias:.0f} %" if bias is not None
                       else "归因不确定度未知")
    out.append(("est__dram_bytes_total",
                round(est.bytes_total / 1e6, 1) if est.reliable else None, "MB",
                "NVML memory-busy time × measured peak bandwidth（对应 ncu dram__bytes.sum）",
                note, f"{est.mem_util_resolution_pct:.0f} %"))
    basis_used = est.resolve_basis(basis)
    # R-47：分母也要能被读者复核 —— 来自哪个窗口、偏移/速率推定、以及它的分辨率。
    denom_bits: list[str] = []
    if est.window_bias_pct not in (None, 0.0):
        denom_bits.append(f"分母 = 窗口内等效 launch 数 {_fmt_launches(est.launches_in_window)}"
                          f"（整段运行匹配到 {est.launches_total} 个；窗口 {est.window_used_s:.1f} s，"
                          f"比整段短 {abs(est.window_bias_pct):.0f}%）")
    if est.launches_in_window_range:
        denom_bits.append("offset ±1 采样周期下的计数区间 "
                          f"{est.launches_in_window_range[0]:g}…{est.launches_in_window_range[1]:g}")
    if est.launch_align_offset_s:
        denom_bits.append(f"launch 时间戳与采样时钟的推导偏移 {est.launch_align_offset_s:+.1f} s、"
                          f"速率比 {est.launch_align_slope:.3f}")
    denom_note = ("；" + "；".join(denom_bits)) if denom_bits else ""
    out.append(("est__dram_bytes_per_launch",
                round(est.bytes_per_launch / 1e6, 2)
                if (est.bytes_per_launch is not None and est.attributable) else None, "MB",
                f"窗口内字节 ÷ 窗口内 launch 数（{_fmt_launches(est.launches_in_window)}；"
                f"整段运行匹配到 {est.launches_total} 个）"
                f"（对应 ncu dram__bytes.sum 的按 launch 口径）",
                f"仅在『被匹配 kernel 占窗口 ≥{SHARE_STRONG:.0f}% + 窗口覆盖率 ≥{DUTY_WEAK:.0f}%"
                f" + 分母与分子覆盖同一窗口』时成立"
                f"（前提②见 ks__matched_kernel_kernel_share_pct）" + denom_note,
                attribution_res))
    out.append(("est__dram_throughput_pct_of_peak",
                round(est.throughput_pct_of_peak, 1) if est.reliable else None, "%",
                "NVML memory-busy 占比（对应 ncu dram__throughput.avg.pct_of_peak_sustained_elapsed）",
                "忙碌时间占比口径，与按字节/时间的口径略有差异", f"{est.mem_util_resolution_pct:.0f} %"))
    # ---- 前提指标：**永远出现**（R-15/B5：两个语义分开报）----
    # 前提①：窗口被被测负载主导（GPU 在不在干活）
    # N5：没有采样点时它必须是 n/a —— 报「0%」等于声称量过，而实际一次都没采到。
    out.append(("ks__sampler_window_coverage_pct",
                round(est.window_coverage_pct, 1) if est.n_samples > 0 else None, "%",
                f"采样窗中 SM≥{BUSY_SM_UTIL:.0f}% 或显存≥{BUSY_MEM_UTIL:.0f}% 的采样点占比",
                f"前提①「窗口被被测负载主导」：<{DUTY_WEAK:.0f}% 不给 per-launch 归因，"
                f"<{DUTY_UNUSABLE:.0f}% 连总量也不给", "1 个采样点"))
    # 前提②：窗口被**被匹配的那个** kernel 主导（R-46）。两个份额是两个量，分开报：
    #   · 内核时间份额 = 判据（空闲时段不产生字节，所以不能拿墙钟当分母）
    #   · 窗口份额     = 与覆盖率同量纲的对照量
    out.append(("ks__matched_kernel_kernel_share_pct",
                round(share_k, 1) if share_k is not None else None, "%",
                f"被匹配 kernel 的 gpu__time_duration.sum ÷ 窗口内**所有** kernel 的 GPU 时间"
                f"（{est.matched_kernel_ns / 1e9:.2f} s / "
                f"{(est.matched_kernel_ns + est.other_kernels_ns) / 1e9:.2f} s）",
                f"前提②「窗口被被匹配的 kernel 主导」：<{SHARE_STRONG:.0f}% 不给 per-launch 归因。"
                f"空闲时段贡献 0 字节，所以判据用内核时间份额而不是墙钟份额："
                f"单 kernel 的窗口无论有多少空闲都是 100%"
                + (f"；窗口里还有 {est.other_kernel_count} 个 kernel"
                   if est.other_kernel_count else ""),
                "0.1 %"))
    out.append(("ks__matched_kernel_window_share_pct",
                round(share_w, 1) if share_w is not None else None, "%",
                f"被匹配 kernel 的 GPU 时间 ÷ 参与积分的窗口长度"
                f"（{est.matched_kernel_ns / 1e9:.2f} s / {est.window_used_s:.1f} s）",
                f"与覆盖率同量纲的对照量（{SHARE_STRONG:.0f}% 门槛用的是内核时间份额）；"
                f"通常 ≤ 覆盖率，差值就是窗口里的空闲"
                + ("；本次分子含窗口外的同名 launch（窗口被裁过），所以它是**上界**，"
                   "可能高于覆盖率，不能当作保守估计" if est.share_upper_bound else ""),
                "0.1 %"))
    out.append(("ks__attribution_bias_estimate_pct",
                round(bias, 1) if bias is not None else None, "%",
                "100×(1/内核时间份额 − 1)：把窗口总量归因到单个 kernel 时**归因本身**"
                "引入的上偏估计",
                "实测校验：k_fused 内核时间份额 44.9% → 估计 +123%，而那次观测误差是 "
                "+146.9% —— 观测值比估计更高，因为该 kernel 的字节密度低于窗口里其他 kernel。"
                "所以这是**估计而非严格上界**：字节强度不一致的混合窗口必须换轨迹口径",
                "0.1 %"))
    out.append(("ks__window_busy_segments",
                float(est.window_busy_segments) if est.n_samples > 0 else None, "段",
                f"窗口里「连续忙碌期」的段数（段间 ≥{SEGMENT_MIN_GAP} 个空闲采样点才算分开）",
                ("多段 ⇒ 窗口可能是多个阶段/变体拼接（例如先跑正确性再跑性能、依次跑 v1/v2/v3），"
                 "此时 per-launch 归因需谨慎：这是启发式信号，不是判据"),
                "1 段"))
    # R-45：参与字节估算的那个「显存忙碌均值」必须有自己的名字。
    # 它与 meas__nvml_mem_busy_mean_pct（**整进程**均值）同名不同物，是两份读数。
    out.append(("ks__dram_window_mem_busy_mean_pct",
                round(est.mem_util_pct_mean, 1) if est.n_samples > 0 else None, "%",
                "参与积分窗口内 NVML utilization.memory 的区间平均（**被积函数的均值**）",
                "这就是摘要里那个「窗口显存忙碌」，也是字节估算真正用到的那个数；"
                "整进程均值另见 meas__nvml_mem_busy_mean_pct，两者不可互换",
                f"{est.mem_util_resolution_pct:.0f} %"))
    out.append(("est__dram_bytes_per_launch_trace",
                round(est.trace_bytes_per_launch / 1e6, 3) if est.trace_available else None, "MB",
                "访存轨迹模型（L2 miss sector × 32B × grid/blocks_traced / launches）",
                ("第二条口径：与 NVML 代理独立；NVML 口径不成立时用它做账本对拍。"
                 "它只测被追踪的那个 kernel，因此**不受窗口里还有别的 kernel 影响**——"
                 "多 kernel 窗口下这是唯一诚实的 per-launch 口径"
                 if est.trace_available else est.trace_unavailable_reason),
                est.trace_resolution or "模型量"))
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
                       if basis_used.startswith("trace") else ""),
                    attribution_res if basis_used.startswith("nvml-per-launch") else
                    (est.trace_resolution or "模型量")))
    if est.ledger_range:
        out.append(("est__dram_range_position_pct",
                    round(est.range_position_pct_for(basis), 1)
                    if est.range_position_pct_for(basis) is not None else None, "%",
                    f"在账本区间 [{est.ledger_range[0] / 1e6:.1f}, {est.ledger_range[1] / 1e6:.1f}] MB "
                    f"（per-launch）中的相对位置；实际口径：{est.basis_label(basis)}",
                    est.range_verdict_for(basis), attribution_res))
        if est.range_verdict_for(basis):
            out.append(("ks__ledger_range_verdict", est.range_verdict_for(basis), "",
                        "由区间位置与两条口径推出",
                        "靠下界 ⇒ 账本预测的流量偏大/O(S²) 口径成立；靠上界 ⇒ 账本可能漏项",
                        attribution_res))
    return out
