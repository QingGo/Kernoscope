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

    @staticmethod
    def from_profile(profile: dict[str, Any], dtype: str = "bf16") -> "MachinePeaks":
        dt = DTYPE_KEY.get(dtype.lower(), "fp16")
        mm = profile.get("measured_matmul_tflops", {})
        cands = [mm[PROFILE_KEY[dt]]] if PROFILE_KEY.get(dt) else []
        if dt == "fp16":  # bf16/fp16 都测过，取更好的那个
            cands = [mm.get("torch.bfloat16"), mm.get("torch.float16")]
        tflops = max((c["tflops"] for c in cands if isinstance(c, dict) and c.get("tflops")),
                     default=0.0)
        dram = profile.get("measured_dram", {})
        bw = max((v for v in dram.values() if isinstance(v, (int, float))), default=0.0)
        return MachinePeaks(
            compute_tflops=float(tflops),
            bandwidth_gbs=float(bw),
            dtype_key=dtype,
            source=str(profile.get("gpu", "machine-profile")),
            raw=profile,
        )

    @staticmethod
    def from_spec(compute_tflops: float, bandwidth_gbs: float, dtype: str = "bf16") -> "MachinePeaks":
        return MachinePeaks(compute_tflops, bandwidth_gbs, dtype, source="spec/override")


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
