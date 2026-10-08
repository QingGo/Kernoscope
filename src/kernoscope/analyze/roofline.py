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
DTYPE_KEY = {
    "fp32": "fp32", "float32": "fp32", "tf32": "fp32",
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "bf16": "fp16", "bfloat16": "fp16",
    "fp8": "fp8", "e4m3": "fp8", "e5m2": "fp8",
}
# machine-profile.py 里 measured_matmul_tflops 的键
PROFILE_KEY = {"fp32": "torch.float32", "fp16": "torch.bfloat16", "fp8": None}


@dataclass
class MachinePeaks:
    """本机实测可达峰值。"""

    compute_tflops: float
    bandwidth_gbs: float
    dtype_key: str
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
        dt = DTYPE_KEY.get(dtype.lower(), "fp16")
        mm = profile.get("measured_matmul_tflops", {})
        # 用 .get()：标定文件里可能没有 matmul 数据（只标了带宽），不能直接下标
        cands = [mm.get(PROFILE_KEY[dt])] if PROFILE_KEY.get(dt) else []
        if dt == "fp16":  # bf16/fp16 都测过，取更好的那个
            cands = [mm.get("torch.bfloat16"), mm.get("torch.float16")]
        usable = [c for c in cands if isinstance(c, dict) and c.get("tflops")]
        tflops = max((c["tflops"] for c in usable), default=0.0)
        chosen = max(usable, key=lambda c: c["tflops"]) if usable else None
        dram = profile.get("measured_dram", {})
        vals = [v for v in dram.values() if isinstance(v, (int, float))]
        # 默认用 copy 速率：DRAM bound 的 kernel 基本是读写混合，取 max（通常落在 write 峰值）
        # 会系统性高估流量（实测约 +4%）。read 峰值与 copy 峰值一般只差 ~1%。
        bw = dram.get("copy_gbs") if isinstance(dram.get("copy_gbs"), (int, float)) else (
            max(vals) if vals else 0.0)

        caliber_ok, note = True, ""
        if chosen is None:
            caliber_ok, note = False, ("标定文件里没有可用的 measured_matmul_tflops 条目："
                                       "拒绝用它算 compute roof")
        else:
            missing = [k for k in ("acc_caliber", "precision_settings", "method")
                       if k not in chosen]
            if missing:
                caliber_ok = False
                note = (f"标定条目 torch.{dt} 缺少口径字段 {', '.join(missing)}："
                        f"按 R-09 拒绝用它算 roofline 拐点（累加口径不同实测可差 5%，"
                        f"见 tools/machine-profile.py 的 caliber 探针）")
        peaks = MachinePeaks(
            compute_tflops=float(tflops),
            bandwidth_gbs=float(bw),
            dtype_key=dtype,
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
        return (f"{where}{when}, dtype={self.dtype_key}, "
                f"bw={self.bandwidth_gbs:.0f} GB/s, compute={self.compute_tflops:.1f} TFLOP/s"
                f"{cal}{spread}")


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
