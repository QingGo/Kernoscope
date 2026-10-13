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
from .refusals import Refusal, legacy_refusal

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
class LaunchConfig:
    """一组 launch 配置（grid/block/资源全同）的汇总。

    为什么必须分开：第 4 章的边界矩阵是「8 形状 × 3 版本 = 24 个对拍点」，靠的是同一个
    kernel 换 launch 配置；第 6 章的 feasible_shapes 同理。把 grid=64/256/1024 三次扫描
    聚合成一行均值，等于把 24 个对拍点变成 1 个 —— 而且那一行还带着 `[exact]` 标签，
    陈述了"这个 kernel 的 grid 是 64"，而 2/3 的 launch 不是。
    """

    key: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    count: int = 0
    total_ns: int = 0
    registers_per_thread: int = 0
    static_smem_bytes: int = 0
    dynamic_smem_bytes: int = 0

    @property
    def mean_ns(self) -> float:
        return self.total_ns / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"config": self.key, "grid": list(self.grid), "block": list(self.block),
                "launch_count": self.count, "total_ns": self.total_ns,
                "mean_ns": self.mean_ns, "registers_per_thread": self.registers_per_thread,
                "static_smem_bytes": self.static_smem_bytes,
                "dynamic_smem_bytes": self.dynamic_smem_bytes}


def config_key(l: "KernelLaunch") -> str:
    """一组 launch 配置的稳定标识（用于 CSV/JSON/指标名）。"""
    return (f"grid{l.grid[0]}x{l.grid[1]}x{l.grid[2]}_block{l.block[0]}x{l.block[1]}x{l.block[2]}"
            f"_smem{l.static_smem_bytes + l.dynamic_smem_bytes}")


@dataclass
class MemcpyRecord:
    """一次拷贝 activity（CUPTI `CUPTI_ACTIVITY_KIND_MEMCPY`）。

    **字段的可信度不一样**（真机实测，2×2080 Ti / 2026-10-12，无 P2P 的主机中转路径）：
    `bytes` 可信；`duration_ns` **不可信**（约一半的行无论多大都报 80–95 µs）。判定见
    `analyze/comm.py` —— 那里拒绝"对时长求和"，只按 `(copyKind, bytes)` 组取 MAX。
    """

    start_ns: int
    end_ns: int
    duration_ns: int
    copy_kind: int
    #: NULL 表示 **activity 里这个字段是空的**，不是 0（R-140）。合计时不许当 0 折算：
    #: 那会把"没读到"变成"搬了 0 字节"，而合计标的是 `exact`。
    bytes: int | None
    #: NULL 表示**不知道是哪张卡**，不许并进 0 号卡（R-140）。
    device_id: int | None
    stream_id: int
    correlation_id: int = 0


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

    # ---- launch 配置分组（R-04/B4）----
    @property
    def configs(self) -> list["LaunchConfig"]:
        """按 (grid, block, 资源) 分组的 launch 汇总，按 launch 数降序。"""
        groups: dict[str, LaunchConfig] = {}
        for l in self.launches:
            k = config_key(l)
            c = groups.get(k)
            if c is None:
                c = groups[k] = LaunchConfig(
                    key=k, grid=l.grid, block=l.block,
                    registers_per_thread=l.registers_per_thread,
                    static_smem_bytes=l.static_smem_bytes,
                    dynamic_smem_bytes=l.dynamic_smem_bytes)
            c.count += 1
            c.total_ns += l.duration_ns
        return sorted(groups.values(), key=lambda c: (-c.count, c.key))

    @property
    def is_mixed_config(self) -> bool:
        return len(self.configs) > 1

    @property
    def dominant_config(self) -> "LaunchConfig | None":
        """launch 数最多的那组配置（occupancy 等单一取值的兜底口径）。"""
        cfgs = self.configs
        return cfgs[0] if cfgs else None

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
    #: 本次采集里出现过的 deviceId（CUPTI 的 `deviceId` 列）。
    #:
    #: **>1 表示跨卡采集**：所有按 kernel 名聚合的量（launch 数、时长合计、均值）都是**跨卡加总**，
    #: 不是单卡口径；而 NVML 读数（时钟/功耗/显存忙碌）只描述**一张卡**（`nvidia-smi` 的第一行）。
    #: 真机实测（2×2080 Ti / 2026-10-12，合成双设备轨迹）：不加 `--devices` 时同名 kernel 的
    #: 时长与 launch 数会**静默翻倍**（31.9 ms/21 次 → 63.7 ms/42 次），既无告警也无拒绝 ——
    #: 那正是本项目平时会拒绝的那种口径混用，所以它必须被显式说出来，而不是靠读者猜。
    #: 存成字段（而不是只做属性）是为了让 `-i` 回读也能保住这个事实。
    device_ids: list[int] = field(default_factory=list)
    #: 本次采集里的拷贝 activity（通信/传输账本的输入）。空表示**没采到**，不是"传输为 0"。
    memcpys: list["MemcpyRecord"] = field(default_factory=list)

    @property
    def is_multi_device(self) -> bool:
        """R-133：必须与 `observed_device_ids()` 同源。

        以前直接读 `self.device_ids`：当字段为空（老产物、或 `-i` 回读一份没有该字段的报告）
        而 launch 里明明跨了两张卡时，这里会答 `False`，与 `observed_device_ids()` 给的
        `[0, 1]` 自相矛盾 —— 于是"跨卡"这件事又被静默掉了。
        """
        return len(self.observed_device_ids()) > 1

    def observed_device_ids(self) -> list[int]:
        """deviceId 集合：优先用记录下来的；老产物没有这个字段时从 launch 现算。"""
        if self.device_ids:
            return list(self.device_ids)
        return sorted({l.device_id for k in self.kernels for l in k.launches})

    def device_breakdown(self) -> list[dict[str, Any]]:
        """按设备分段：每张卡各自的 launch 数、kernel 时间、出现过的 kernel 数。

        `--devices` 是**筛选**（只留一张卡），这里是**分段**（每张卡是多少）——
        跨卡运行里读者最需要的是后者：知道两张卡各干了多少活，才知道有没有负载不均。
        """
        out: dict[int, dict[str, Any]] = {}
        for k in self.kernels:
            for l in k.launches:
                row = out.setdefault(l.device_id, {"device_id": l.device_id,
                                                   "launch_count": 0, "kernel_ns": 0,
                                                   "kernels": set()})
                row["launch_count"] += 1
                row["kernel_ns"] += l.duration_ns
                row["kernels"].add(k.name)
        return [dict(r, kernels=len(r["kernels"]))
                for r in sorted(out.values(), key=lambda r: r["device_id"])]

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
    def to_dict(self, include_launches: bool = True) -> dict[str, Any]:
        """`include_launches=False` 让报告体积与 iteration 数解耦（R-22）。

        2000 次 launch 的 JSON 实测 901 KB，主体就是 `launches[]`；只关心聚合读数时
        没必要把它带上。需要逐 launch 数据时用 `--launch-csv` 单独出表。
        """
        return {
            "schema_version": self.schema_version,
            "tool": "kernoscope",
            "tool_version": self.tool_version,
            "generated_at": self.generated_at or _now(),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "device": asdict(self.device),
            # 跨卡事实必须进 JSON：否则 `-i` 回读后这个口径就丢了（见 device_ids 的注释）
            "device_ids": list(self.observed_device_ids()),
            "multi_device": self.is_multi_device,
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
                    "mixed_config": k.is_mixed_config,
                    "launch_configs": [c.as_dict() for c in k.configs],
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
                                    "resolution": m.resolution,
                                    **({"refusal": m.refusal.to_dict()} if m.refusal else {}),
                                }
                                for m in s.metrics
                            ],
                        }
                        for s in k.sections
                    ],
                    "launches": [] if not include_launches else [
                        {
                            "start_ns": l.start_ns,
                            "end_ns": l.end_ns,
                            "duration_ns": l.duration_ns,
                            "grid": list(l.grid),
                            "block": list(l.block),
                            "registers_per_thread": l.registers_per_thread,
                            "static_smem_bytes": l.static_smem_bytes,
                            "dynamic_smem_bytes": l.dynamic_smem_bytes,
                            "stream_id": l.stream_id,
                            "device_id": l.device_id,
                            "grid_id": l.grid_id,
                            "config": config_key(l),
                        }
                        for l in k.launches
                    ],
                }
                for k in self.kernels
            ],
        }

    def to_json(self, indent: int = 2, include_launches: bool = True) -> str:
        return json.dumps(self.to_dict(include_launches=include_launches), indent=indent,
                          ensure_ascii=False)

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
            device_ids=[int(x) for x in (d.get("device_ids") or [])],
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
                            resolution=md.get("resolution", ""),
                            refusal=(Refusal.from_dict(md["refusal"]) if md.get("refusal")
                                     else (legacy_refusal(md.get("reason", ""))
                                           if Provenance(md["provenance"])
                                           is Provenance.UNAVAILABLE else None)),
                        )
                    )
                k.sections.append(s)
            for ld in kd.get("launches", []):
                k.launches.append(KernelLaunch(
                    name=kd["name"], demangled=kd["name"],
                    start_ns=int(ld["start_ns"]), end_ns=int(ld["end_ns"]),
                    grid=tuple(ld.get("grid", (1, 1, 1))),      # type: ignore[arg-type]
                    block=tuple(ld.get("block", (1, 1, 1))),    # type: ignore[arg-type]
                    registers_per_thread=int(ld.get("registers_per_thread", 0)),
                    static_smem_bytes=int(ld.get("static_smem_bytes", 0)),
                    dynamic_smem_bytes=int(ld.get("dynamic_smem_bytes", 0)),
                    local_mem_per_thread=int(ld.get("local_mem_per_thread", 0)),
                    stream_id=int(ld.get("stream_id", 0)),
                    device_id=int(ld.get("device_id", 0)),
                    grid_id=int(ld.get("grid_id", 0)),
                    correlation_id=int(ld.get("correlation_id", 0)),
                ))
            rep.kernels.append(k)
        return rep


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
