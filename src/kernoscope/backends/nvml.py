"""NVML / nvidia-smi 后端：设备级计数器（免权限）。

实测可读（无需 admin）：
  * PCIe RX/TX 吞吐、replay 计数
  * 能量累加器（mJ）→ 每 kernel 能耗（ncu 没有这个指标）
  * 显存忙碌占比 utilization.memory —— 稳态下是 DRAM 带宽的可用代理
    （实测：饱和拷贝 96.8% ↔ 920 GB/s，误差 6%；但它是 200ms 区间平均，
     带宿主间隙时会被稀释成 6.3%，所以必须成对报告占空比/利用率）
  * SM 忙碌占比、时钟、功耗、降频原因位图

优先用 pynvml（`pip install kernoscope[nvml]`）；否则退回 `nvidia-smi` 子进程轮询。
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

SMi = "nvidia-smi"


@dataclass
class DeviceSample:
    t: float
    mem_util_pct: float
    sm_util_pct: float
    sm_clock_mhz: float
    power_w: float


@dataclass
class SamplerResult:
    samples: list[DeviceSample] = field(default_factory=list)
    energy_j: float = 0.0
    duration_s: float = 0.0
    source: str = "nvidia-smi"

    def _col(self, attr: str) -> list[float]:
        return [getattr(s, attr) for s in self.samples]

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        out: dict[str, float] = {}
        for attr, key in (("mem_util_pct", "mem_util"), ("sm_util_pct", "sm_util"),
                          ("sm_clock_mhz", "sm_clock"), ("power_w", "power")):
            vals = self._col(attr)
            out[f"{key}_mean"] = statistics.fmean(vals)
            out[f"{key}_max"] = max(vals)
            out[f"{key}_min"] = min(vals)
        out["samples"] = float(len(self.samples))
        if self.energy_j:
            out["energy_j"] = self.energy_j
            if self.duration_s:
                out["avg_power_w"] = self.energy_j / self.duration_s
        # DRAM 带宽代理：util% × 实测峰值带宽（调用方填 peak_gbs 后可用）
        out["mem_busy_pct_mean"] = out["mem_util_mean"]
        return out


class Sampler:
    """后台采样设备级指标（context manager）。"""

    def __init__(self, interval_s: float = 0.2, enabled: bool = True):
        self.interval_s = interval_s
        self.enabled = enabled
        self.result = SamplerResult()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None

    # ---- pynvml（可选，提供能量累加器）----
    def _try_pynvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.result.source = "pynvml"
        except Exception:
            self._pynvml = None

    def _energy_j(self) -> float:
        if not self._pynvml:
            return 0.0
        try:
            return self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) / 1000.0
        except Exception:
            return 0.0

    def __enter__(self) -> "Sampler":
        if not self.enabled or not shutil.which(SMi):
            self.enabled = False
            return self
        self._try_pynvml()
        self._e0 = self._energy_j()
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.result.duration_s = time.monotonic() - self._t0
        if self._pynvml:
            self.result.energy_j = max(0.0, self._energy_j() - self._e0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            s = self._read()
            if s:
                self.result.samples.append(s)

    def _read(self) -> DeviceSample | None:
        if self._pynvml:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                clk = self._pynvml.nvmlDeviceGetClockInfo(self._handle, self._pynvml.NVML_CLOCK_SM)
                pwr = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                return DeviceSample(time.monotonic(), float(util.memory), float(util.gpu),
                                    float(clk), float(pwr))
            except Exception:
                return None
        q = "utilization.memory,utilization.gpu,clocks.sm,power.draw"
        proc = subprocess.run([SMi, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            mem, sm, clk, pwr = [float(x) for x in proc.stdout.strip().split(",")]
        except Exception:
            return None
        return DeviceSample(time.monotonic(), mem, sm, clk, pwr)
