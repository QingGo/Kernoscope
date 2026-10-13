"""SpeedOfLight / roofline —— 用**本机实测峰值**替代规格表。

为什么必须标定：实测中 RTX 4090 的 fp32 matmul 常只有 ~54 TFLOP/s
（规格 82.6），如果拿规格表做 roofline 会得出"只发挥了 65%"的错误结论，
实际上那台机器的 fp32 上限就是 54，且它被 450W 功耗墙限制（满载 SM 2459–2495 MHz，
nvidia-smi 空闲时报 210 MHz）。

因此：
  * 峰值来自 machine-profile 实测（`dram.read_gbs` / `measured_matmul_tflops`）
  * 工作量（FLOPs / bytes）必须由用户提供（`--ks-flops` / `--ks-gemm`），因为我们无法
    从二进制推断语义
  * 出处一律 `estimated`（模型 + 用户输入），并使用 `est__` 前缀
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ITEMSIZE = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}
#: dtype → `measured_matmul_tflops` 里的条目键。
#:
#: **bf16 与 fp16 是两条独立条目，不互相顶替**（真机实测，2026-10-12）：RTX 2080 Ti 上
#: bf16 走 fp32 路径只有 6.8 TFLOP/s，而 fp16 张量核 49.4 —— 差 **7.3 倍**；RTX 4090 上
#: 两者只差 1%（169.0 vs 167.4），所以这条区别在主力机上一直没暴露。
#: 旧实现把 bf16 映射到 `torch.bfloat16` 后再和 `torch.float16` 取 `max()`，等于拿 fp16 的
#: 峰值当 bf16 的 roofline 分母（利用率低报 7.3 倍），同时把守卫上限抬高 7.3 倍。
DTYPE_KEY = {
    "fp32": "torch.float32", "float32": "torch.float32", "tf32": "torch.float32",
    "fp16": "torch.float16", "float16": "torch.float16", "half": "torch.float16",
    "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    "fp8": None, "e4m3": None, "e5m2": None,
}


@dataclass
class MachinePeaks:
    """本机实测可达峰值。"""

    compute_tflops: float
    bandwidth_gbs: float
    dtype_key: str
    #: 同一台机器上**任何 dtype** 都不可能超过的实测算力（只给"物理上不可能"守卫用）。
    #: 与 `compute_tflops`（本次 dtype 的实测峰值，roofline 的诚实分母）**故意分开**：
    #: 两者在 16 位条目相差悬殊的机器上（Turing）能差 7 倍，混用会同时错两个方向。
    ceiling_tflops: float = 0.0
    source: str = "machine-profile"
    raw: dict[str, Any] | None = None
    profile_path: str = ""
    calibrated_at: str = ""
    # R-09/R-10：标定口径。`caliber_ok=False` 时**拒绝用它算拐点**（compute roof 不可用），
    # 但带宽侧照常可用 —— 字节数与显存忙碌代理不依赖累加口径。
    caliber_ok: bool = True
    caliber_note: str = ""
    caliber_spread_pct: float | None = None
    caliber_distinguishable: bool | None = None
    acc_caliber: str = ""
    precision_settings: dict[str, Any] | None = None

    @staticmethod
    def from_profile(profile: dict[str, Any], dtype: str = "bf16",
                     profile_path: str = "") -> "MachinePeaks":
        mm = profile.get("measured_matmul_tflops", {}) or {}
        key = DTYPE_KEY.get(dtype.lower())
        entry = mm.get(key) if key else None
        chosen = entry if isinstance(entry, dict) and entry.get("tflops") else None
        tflops = float(chosen["tflops"]) if chosen else 0.0
        # 同一台机器上**任何** dtype 都不可能超过实测条目里的最大值。它只服务一件事：
        # 「达成值 > 峰值」的一致性守卫（R-38 要拦的正是"峰值被取小了"）。
        # 绝不能拿它当某个 dtype 的 roofline 分母 —— 那是本节开头那条 7.3 倍错的来源。
        ceiling = max((float(v["tflops"]) for v in mm.values()
                       if isinstance(v, dict) and v.get("tflops")), default=0.0)
        dram = profile.get("measured_dram", {})
        vals = [v for v in dram.values() if isinstance(v, (int, float))]
        # 默认用 copy 速率：DRAM bound 的 kernel 基本是读写混合，取 max（通常落在 write 峰值）
        # 会系统性高估流量（实测约 +4%）。read 峰值与 copy 峰值一般只差 ~1%。
        bw = dram.get("copy_gbs") if isinstance(dram.get("copy_gbs"), (int, float)) else (
            max(vals) if vals else 0.0)

        caliber_ok, note = True, ""
        if chosen is None:
            have = sorted(k for k, v in mm.items() if isinstance(v, dict) and v.get("tflops"))
            caliber_ok = False
            note = (f"标定文件里没有 dtype={dtype} 的实测条目（对应键 {key or '无'}）；"
                    f"现有条目：{', '.join(have) or '无'}。**不拿别的 dtype 的峰值顶替**"
                    f"（同一台机器上 bf16 与 fp16 实测可差数倍，真机实测 7.3 倍），"
                    f"所以拒绝用它算 compute roof；重新标定或改用 --ks-dtype 对应的 dtype")
        else:
            missing = [k for k in ("acc_caliber", "precision_settings", "method")
                       if k not in chosen]
            if missing:
                caliber_ok = False
                note = (f"标定条目 {key} 缺少口径字段 {', '.join(missing)}："
                        f"按 R-09 拒绝用它算 roofline 拐点（累加口径不同实测可差 5%，"
                        f"见 tools/machine-profile.py 的 caliber 探针）")
        peaks = MachinePeaks(
            compute_tflops=float(tflops),
            bandwidth_gbs=float(bw),
            dtype_key=dtype,
            ceiling_tflops=float(ceiling),
            source=str(profile.get("gpu", "machine-profile")),
            raw=profile,
            profile_path=profile_path,
            calibrated_at=str(profile.get("ts", "")),
            caliber_ok=caliber_ok,
            caliber_note=note,
        )
        if chosen is not None:
            peaks.acc_caliber = str(chosen.get("acc_caliber", ""))
            peaks.precision_settings = chosen.get("precision_settings") or None
            peaks.caliber_spread_pct = chosen.get("caliber_spread_pct")
            peaks.caliber_distinguishable = chosen.get("caliber_distinguishable")
            if not peaks.caliber_note and peaks.caliber_distinguishable is False:
                peaks.caliber_note = (
                    f"该机三种累加口径设置最大相对差 {peaks.caliber_spread_pct}% —— "
                    f"不可分辨，说明这一档的「精确单值」本身不可得")
        return peaks

    @staticmethod
    def from_spec(compute_tflops: float, bandwidth_gbs: float, dtype: str = "bf16") -> "MachinePeaks":
        return MachinePeaks(compute_tflops, bandwidth_gbs, dtype, source="spec/override",
                            ceiling_tflops=compute_tflops,
                            caliber_ok=False,
                            caliber_note="峰值来自 --ks-peak-* 覆盖或规格表，不是本机标定："
                                         "不用于拐点判定（R-09）")

    def source_label(self) -> str:
        """R-06：峰值必须带来源与时间戳。"""
        where = self.profile_path or self.source
        when = f" @ {self.calibrated_at}" if self.calibrated_at else ""
        cal = f", 累加口径={self.acc_caliber or '未记录'}"
        spread = (f", 口径离散={self.caliber_spread_pct}%"
                  if self.caliber_spread_pct is not None else "")
        # 两个分母不同时必须都写出来：否则读者以为 roofline 用的是"同机上限"，
        # 而守卫用的是另一个数（或反过来）—— 真机实测这两者能差 7.3 倍。
        ceil_txt = (f", 同机任何 dtype 的上限={self.ceiling_tflops:.1f} TFLOP/s（只给一致性守卫用）"
                    if self.ceiling_tflops
                    and abs(self.ceiling_tflops - self.compute_tflops) > 0.05 else "")
        return (f"{where}{when}, dtype={self.dtype_key}, "
                f"bw={self.bandwidth_gbs:.0f} GB/s, compute={self.compute_tflops:.1f} TFLOP/s"
                f"{cal}{spread}{ceil_txt}")


def gemm_workload(m: int, n: int, k: int, dtype: str = "bf16") -> tuple[float, float]:
    """返回 (flops, bytes)。bytes 按 A/B/C 各读写一次估算。"""
    isz = ITEMSIZE.get(dtype.lower(), 2)
    flops = 2.0 * m * n * k
    bytes_ = (m * k + k * n + m * n) * isz
    return flops, bytes_


def elementwise_workload(n: int, dtype: str = "fp32", inputs: int = 1) -> tuple[float, float]:
    isz = ITEMSIZE.get(dtype.lower(), 4)
    return float(n), float(n) * isz * (inputs + 1)


@dataclass
class SpeedOfLight:
    duration_ns: float
    flops: float
    bytes_: float
    peaks: MachinePeaks

    @property
    def achieved_tflops(self) -> float:
        return self.flops / (self.duration_ns / 1e9) / 1e12 if self.duration_ns else 0.0

    @property
    def achieved_gbs(self) -> float:
        return self.bytes_ / (self.duration_ns / 1e9) / 1e9 if self.duration_ns else 0.0

    @property
    def pct_compute(self) -> float:
        return 100.0 * self.achieved_tflops / self.peaks.compute_tflops if self.peaks.compute_tflops else 0.0

    @property
    def pct_compute_ceiling(self) -> float:
        """达成值相对**同机任何 dtype 的实测算力上限**的百分比。

        只用于「物理上不可能」守卫，与 `pct_compute` 的分母**故意不同**：
        `pct_compute` 回答"这次 dtype 用满了吗"（分母 = 本 dtype 实测峰值），
        守卫回答"这台机器有没有可能做到"（分母 = 所有 dtype 里的最大值）。
        两者在 16 位相差悬殊的机器上能差数倍（Turing bf16 vs fp16 = 7.3 倍）：
        拿本 dtype 峰值当守卫上限会**误报**，拿上限当 roofline 分母会**低报利用率**。
        """
        c = self.peaks.ceiling_tflops or self.peaks.compute_tflops
        return 100.0 * self.achieved_tflops / c if c else 0.0

    @property
    def dtype_peak_exceeded(self) -> bool:
        """达成值是否超过**本次 dtype 的实测峰值**（超过容差）。

        R-138：`over_peak` 用的是**同机任何 dtype 的上限**（避免峰值取小而误报），
        但被报给读者的 `pct_compute` 用的是**本 dtype 峰值**。两者在 16 位相差悬殊的机器上
        能差数倍，于是会出现"报 294%、守卫却放绿"的窗口：本 dtype 物理上不可能，而同机上限内可能。
        这一档不能静默 —— 它意味着**这个 kernel 走的不是本 dtype 的路径**
        （例如 bf16 请求在 Turing 上实际跑 fp16/fp32 路径），百分比因此不能当达成率读。
        """
        if not self.peaks.compute_tflops:
            return False
        return self.pct_compute > 100.0 + self.PEAK_EXCEED_TOLERANCE_PCT

    @property
    def pct_memory(self) -> float:
        return 100.0 * self.achieved_gbs / self.peaks.bandwidth_gbs if self.peaks.bandwidth_gbs else 0.0

    @property
    def ideal_ns_compute(self) -> float:
        return self.flops / (self.peaks.compute_tflops * 1e12) * 1e9 if self.peaks.compute_tflops else 0.0

    @property
    def ideal_ns_memory(self) -> float:
        return self.bytes_ / (self.peaks.bandwidth_gbs * 1e9) * 1e9 if self.peaks.bandwidth_gbs else 0.0

    @property
    def bound(self) -> str:
        return "compute" if self.ideal_ns_compute >= self.ideal_ns_memory else "memory"

    @property
    def efficiency_pct(self) -> float:
        ideal = max(self.ideal_ns_compute, self.ideal_ns_memory)
        return 100.0 * ideal / self.duration_ns if self.duration_ns else 0.0

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_ if self.bytes_ else 0.0

    @property
    def machine_balance(self) -> float:
        return (self.peaks.compute_tflops * 1e12) / (self.peaks.bandwidth_gbs * 1e9) \
            if self.peaks.bandwidth_gbs else 0.0

    # ---- R-38：达成值超过标定峰值必须是**红灯**，不能只印个除法结果 ----
    # 达成值 > 峰值在物理上不可能。出现它只有三种成因，而且都会让结论反过来：
    #   · 峰值取小了（标定口径/尺寸与本次负载不同，例如拿 fp32 峰值去比 fp16 的负载）
    #   · 工作量口径算错（FLOPS/字节数的解析式与实际 kernel 不符，常见于漏乘或重复计数）
    #   · 标定与本次运行不在同一状态（时钟/功耗受限、冷启动、MIG、别的进程抢卡）
    # 教程用「162.3 TFLOPS ≈ 98% fp16 峰值」做裁决；若峰值被取成 145.7，同一份数据
    # 就变成 114% —— 一个物理上不可能的结论被当成裁决依据，正是这里要拦住的东西。
    PEAK_EXCEED_TOLERANCE_PCT = 2.0     # 容差：标定本身有 3~4% 口径差，2% 内给**提示**（不报红灯）

    @property
    def near_peak(self) -> bool:
        """100% < 达成值 ≤ 100%+容差：超过标称峰值但还在标定自身口径差之内。

        这一段**不是**红灯（可能是口径差），但绝不能静默：101% 与 95% 的读数长得一样，
        而前者已经物理上说不通（D20）。所以它要出一句提示，让读者去核对标定与工作量。
        """
        if not self.peaks.compute_tflops and not self.peaks.bandwidth_gbs:
            return False
        return (not self.over_peak
                and (self.pct_compute_ceiling > 100.0 or self.pct_memory > 100.0))

    def near_peak_note(self) -> str:
        """容忍带内的提示（与红灯同一套措辞口径，但明确说"先别当达标证据"）。"""
        if not self.near_peak:
            return ""
        which = []
        if self.pct_compute_ceiling > 100.0:
            which.append(f"算力 {self.achieved_tflops:.1f} TFLOPS = 标定峰值的 "
                         f"{self.pct_compute_ceiling:.1f}%")
        if self.pct_memory > 100.0:
            which.append(f"带宽 {self.achieved_gbs:.1f} GB/s = 标定峰值的 {self.pct_memory:.1f}%")
        return ("达成值达到/略超过标定峰值（在标定自身 "
                f"{self.PEAK_EXCEED_TOLERANCE_PCT:.0f}% 口径差之内）：" + "；".join(which)
                + "。这不一定是「达标」：先核对峰值口径（dtype/尺寸）与工作量输入，"
                  "再拿它当结论。")

    def _dtype_peak_aside(self) -> str:
        """守卫用的是"同机上限"，而 roofline 用的是"本 dtype 峰值" —— 两者不同时点名，
        否则读者会把 7.3 倍的差当成测量噪声（真机实测：2080 Ti bf16 6.8 / fp16 49.4）。"""
        cap = self.peaks.ceiling_tflops or self.peaks.compute_tflops
        if not cap or abs(cap - self.peaks.compute_tflops) <= 0.05:
            return ""
        return (f"；本次 dtype={self.peaks.dtype_key} 的实测峰值是 "
                f"{self.peaks.compute_tflops:.1f} TFLOP/s")

    @property
    def over_peak(self) -> bool:
        """达成值是否超过标定峰值（超出容差）。"""
        if not self.peaks.compute_tflops and not self.peaks.bandwidth_gbs:
            return False
        return self.pct_compute_ceiling > 100.0 + self.PEAK_EXCEED_TOLERANCE_PCT \
            or self.pct_memory > 100.0 + self.PEAK_EXCEED_TOLERANCE_PCT

    def over_peak_note(self) -> str:
        """红灯说明：点明哪一个超了、超多少、以及最可能的三种成因。"""
        if not self.over_peak:
            return ""
        which = []
        if self.pct_compute_ceiling > 100.0 + self.PEAK_EXCEED_TOLERANCE_PCT:
            cap = self.peaks.ceiling_tflops or self.peaks.compute_tflops
            which.append(f"算力 {self.achieved_tflops:.1f} TFLOPS = 标定峰值的 "
                         f"{self.pct_compute_ceiling:.1f}%（同机任何 dtype 的上限 "
                         f"{cap:.1f} TFLOPS）" + self._dtype_peak_aside())
        if self.pct_memory > 100.0 + self.PEAK_EXCEED_TOLERANCE_PCT:
            which.append(f"带宽 {self.achieved_gbs:.1f} GB/s = 标定峰值的 "
                         f"{self.pct_memory:.1f}%（峰值 {self.peaks.bandwidth_gbs:.1f} GB/s）")
        return ("达成值超过本机标定峰值（物理上不可能）：" + "；".join(which)
                + "。最可能的成因：① 峰值取小了/口径不一致（例如拿 fp32 峰值比 fp16 负载、"
                  "标定尺寸与本次负载不同）；② 工作量口径算错（FLOPS/字节数的解析式与实际 "
                  "kernel 不符）；③ 标定与本次运行状态不同（时钟/功耗受限、冷启动、抢卡）。"
                  "先复核这三项，不要把它当成达标证据。")
