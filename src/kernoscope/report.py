"""报告数据模型：Device → Kernel → Section → Metric。

刻意保持扁平、可 JSON 序列化，便于：
  * `-o rep.json` 落盘、`-i rep.json` 回读（对标 ncu 的 `--export` / `--import`）
  * Perfetto / Chrome Trace 导出
  * CI 回归对比（同一 schema + `schema_version`）
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import __version__
from .provenance import Metric, Provenance, Section

SCHEMA_VERSION = 1


@dataclass
class DeviceInfo:
    name: str = ""
    driver: str = ""
    cuda_runtime: str = ""
    sm_count: int = 0
    compute_capability: str = ""
    l2_mib: float = 0.0
    memory_total_mib: float = 0.0
    max_sm_clock_mhz: float = 0.0
    power_limit_w: float = 0.0
    # 本机实测可达峰值（来自 machine-profile，用于 roofline）
    measured_peaks: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelLaunch:
    """单次 launch 的原始记录（来自 CUPTI activity，exact）。"""

    name: str
    demangled: str
    start_ns: int
    end_ns: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers_per_thread: int
    static_smem_bytes: int
    dynamic_smem_bytes: int
    local_mem_per_thread: int
    stream_id: int
    device_id: int
    grid_id: int
    correlation_id: int
    short_name: str = ""
    mangled: str = ""

    def display_name(self, base: str = "demangled") -> str:
        """按 --kernel-name-base 选择展示名。

        实测坑：cutlass/cuBLAS 的 nsys shortName 就是模板类名 `Kernel2`，
        真实信息在 demangledName 里（`void cutlass::Kernel2<cutlass_80_tensorop_bf16_...>`），
        所以默认用 demangled，`function` 才回退到短名。
        """
        if base == "function":
            return self.short_name or self.demangled or self.name
        if base == "mangled":
            return self.mangled or self.demangled or self.name
        return self.demangled or self.name or self.short_name

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def threads_per_block(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def total_threads(self) -> int:
        return self.threads_per_block * self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def is_cuda_graph(self) -> bool:
        return self.grid_id != 0


@dataclass
class KernelReport:
    """一个 kernel（按名字聚合）的报告。"""

    name: str
    launch_count: int = 0
    alias: str = ""     # nsys short name，与 name 不同时展示，便于对照 nsys/torch profiler
    sections: list[Section] = field(default_factory=list)
    launches: list[KernelLaunch] = field(default_factory=list)

    def section(self, name: str) -> Section:
        for s in self.sections:
            if s.name == name:
                return s
        s = Section(name)
        self.sections.append(s)
        return s

    # ---- 汇总（全部来自 exact 的 CUPTI activity 时间戳）----
    @property
    def total_ns(self) -> int:
        return sum(l.duration_ns for l in self.launches)

    @property
    def mean_ns(self) -> float:
        return self.total_ns / self.launch_count if self.launch_count else 0.0

    @property
    def min_ns(self) -> int:
        return min((l.duration_ns for l in self.launches), default=0)

    @property
    def max_ns(self) -> int:
        return max((l.duration_ns for l in self.launches), default=0)


@dataclass
class Report:
    device: DeviceInfo = field(default_factory=DeviceInfo)
    kernels: list[KernelReport] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    wall_time_ns: int = 0
    tool_version: str = __version__
    schema_version: int = SCHEMA_VERSION
    generated_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def kernel(self, name: str) -> KernelReport:
        for k in self.kernels:
            if k.name == name:
                return k
        k = KernelReport(name)
        self.kernels.append(k)
        return k

    def sort_kernels(self, key: str = "total") -> None:
        if key == "total":
            self.kernels.sort(key=lambda k: k.total_ns, reverse=True)
        elif key == "name":
            self.kernels.sort(key=lambda k: k.name)

    @property
    def total_kernel_ns(self) -> int:
        return sum(k.total_ns for k in self.kernels)

    def unavailable_metrics(self) -> list[Metric]:
        return [m for s in self.iter_sections() for m in s.metrics
                if m.provenance is Provenance.UNAVAILABLE]

    def iter_sections(self):
        for k in self.kernels:
            for s in k.sections:
                yield s

    def provenance_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {p.value: 0 for p in Provenance}
        for s in self.iter_sections():
            for m in s.metrics:
                hist[m.provenance.value] += 1
        return hist

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": "kernoscope",
            "tool_version": self.tool_version,
            "generated_at": self.generated_at or _now(),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "device": asdict(self.device),
            "wall_time_ns": self.wall_time_ns,
            "sources": self.sources,
            "meta": self.meta,
            "kernels": [
                {
                    "name": k.name,
                    "launch_count": k.launch_count,
                    "total_ns": k.total_ns,
                    "mean_ns": k.mean_ns,
                    "min_ns": k.min_ns,
                    "max_ns": k.max_ns,
                    "sections": [
                        {
                            "name": s.name,
                            "provenance": s.provenance.value,
                            "note": s.note,
                            "metrics": [
                                {
                                    "name": m.name,
                                    "value": m.value,
                                    "unit": m.unit,
                                    "provenance": m.provenance.value,
                                    "source": m.source,
                                    "reason": m.reason,
                                    "note": m.note,
                                }
                                for m in s.metrics
                            ],
                        }
                        for s in k.sections
                    ],
                    "launches": [
                        {
                            "start_ns": l.start_ns,
                            "end_ns": l.end_ns,
                            "grid": list(l.grid),
                            "block": list(l.block),
                            "registers_per_thread": l.registers_per_thread,
                            "static_smem_bytes": l.static_smem_bytes,
                            "dynamic_smem_bytes": l.dynamic_smem_bytes,
                            "stream_id": l.stream_id,
                            "device_id": l.device_id,
                            "grid_id": l.grid_id,
                        }
                        for l in k.launches
                    ],
                }
                for k in self.kernels
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Report":
        dev = DeviceInfo(**d.get("device", {}))
        rep = Report(
            device=dev,
            sources=d.get("sources", {}),
            wall_time_ns=d.get("wall_time_ns", 0),
            tool_version=d.get("tool_version", __version__),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            generated_at=d.get("generated_at", ""),
            meta=d.get("meta", {}),
        )
        for kd in d.get("kernels", []):
            k = KernelReport(kd["name"], kd.get("launch_count", 0))
            for sd in kd.get("sections", []):
                s = Section(sd["name"], note=sd.get("note", ""))
                for md in sd.get("metrics", []):
                    s.add(
                        Metric(
                            name=md["name"],
                            value=md.get("value"),
                            provenance=Provenance(md["provenance"]),
                            unit=md.get("unit", ""),
                            source=md.get("source", ""),
                            reason=md.get("reason", ""),
                            note=md.get("note", ""),
                        )
                    )
                k.sections.append(s)
            rep.kernels.append(k)
        return rep


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
