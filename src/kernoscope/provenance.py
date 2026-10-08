"""数据出处（provenance）体系 —— 本项目的硬规则，由测试强制。

设计动机（见 docs/provenance.md 与 ADR）：
    在受限环境里，最危险的不是"没有数据"，而是**看起来合理的假数据**。
    我们在调研中亲身踩过三次：L2 命中率被窗口偏差算成 75%、NVML 内存忙碌占比被
    宿主间隙稀释成 6.3%、静态共享内存被 ptxas 当死代码删掉导致 occupancy 失真。
    因此把"出处"变成类型系统的一部分，而不是文档里的口头约定。

四条规则（`validate()` 会强制，tests/test_provenance.py 覆盖）：
    R1  UNAVAILABLE 的 metric 必须 value is None 且必须有 reason。
    R2  计数器命名空间（sm__/smsp__/l1tex__/lts__/dram__/...）下的 metric 只允许 EXACT。
        —— 我们拿不到计数器，所以这些名字只能出现在 unavailable 列表里。
        **加固（R-11）**：EXACT 也不够 —— 计数器名还必须登记在 `EXACT_COUNTER_ALLOWLIST`
        里（当前为空）。这样"在 ncu 指标名下面填一个数"在类型层面就是不可能的，
        哪怕调用方写的是 `Metric.exact`。
    R3  非 EXACT 的 metric 必须使用我们自有的前缀（est__/meas__/model__/ks__）。
        —— 这一条彻底堵死"在 ncu 指标名下表一个模型算出来的数"。
    R4  ncu 的 `launch__*` / `gpu__time_duration*` 允许 EXACT（我们确实同源可得）。

另有一条**命名纪律**（R-12，`tests/test_provenance.py` 守）：
    `meas__` 前缀的 metric，provenance 必须是 measured —— 名字承诺什么，出处就得是什么。
    由实测样本**再计算**出来的统计量（占比、均值）不是"实测读数"，用 `ks__` + derived。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class Provenance(str, Enum):
    """数据出处。继承 str 便于 JSON/CSV 直出。"""

    EXACT = "exact"
    DERIVED = "derived"
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"

    @property
    def tag(self) -> str:
        return f"[{self.value}]"

    @property
    def is_numeric_trustworthy(self) -> bool:
        """是否可直接用于跨工具/跨机器比较。

        只有与 ncu 同源的 exact 才允许；其余只能用于本机本版本做决策。
        """
        return self is Provenance.EXACT


# ncu 的硬件计数器命名空间：这些名字意味着"硬件计数器读出来的值"
COUNTER_NAMESPACES: tuple[str, ...] = (
    "sm__", "smsp__", "l1tex__", "lts__", "dram__", "l1__", "l2__",
    "sys__", "nvlink__", "membar__", "tpc__", "fbpa__", "idc__",
)

# 我们允许以 EXACT 声明的 ncu 名字（与 ncu 同源：CUPTI activity / cubin ELF）
EXACT_NCU_NAMESPACES: tuple[str, ...] = ("launch__", "gpu__time_duration")

# R2 加固：计数器命名空间下的**有值**指标，必须登记在这里。
# 当前为空 —— 这台机器上任何一个计数器读数都拿不到，所以任何有值的计数器指标都是 bug。
# 将来若真的接上计数器（例如回读 ncu 的 .ncu-rep），把允许的名字加进来即可。
EXACT_COUNTER_ALLOWLIST: frozenset[str] = frozenset()

# 自有指标前缀：任何非 EXACT 的数据都必须落在这里
OUR_PREFIXES: tuple[str, ...] = ("est__", "meas__", "model__", "ks__")

UNAVAILABLE_REASONS: dict[str, str] = {
    "counters": (
        "requires GPU hardware performance counters, which are restricted here "
        "(RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); "
        "CUPTI reports ERR_NVGPUCTRPERM"
    ),
    "instrumentation": (
        "requires in-kernel instrumentation (NVBit / IKP trace) which is not enabled; "
        "install NVBit and pass an existing trace with --ks-memtrace <file> "
        "(see tools/probes/nvbit_memtrace/collect.sh)"
    ),
    "sampling": "requires PC sampling, which needs the profiling permission here",
}


class ProvenanceError(ValueError):
    """违反 provenance 规则时抛出（CI 会因此失败）。"""


@dataclass(frozen=True)
class Metric:
    """一个带出处的指标。

    name        ncu 兼容名或自有名（自有名必须带 OUR_PREFIXES 前缀）
    value       数值或字符串；UNAVAILABLE 时必须为 None
    provenance  出处
    source      人可读的数据来源（例如 "nsys sqlite CUPTI_ACTIVITY_KIND_KERNEL"）
    reason      UNAVAILABLE 时的原因（规则 R1）
    note        误差/限制说明（例如代理的采样周期）
    resolution  量化的**分辨率下限**（R-31）：例如 NVML utilization.memory 是 1 个百分点、
                采样周期 200 ms。读数为 0 时"真的是 0"与"0.3% 被截断成 0"不可区分，
                所以分辨率必须与数字同时出现。
    """

    name: str
    value: float | int | str | None
    provenance: Provenance
    unit: str = ""
    source: str = ""
    reason: str = ""
    note: str = ""
    resolution: str = ""

    def __post_init__(self) -> None:
        self.validate()

    # ---- 规则 ----
    def validate(self) -> None:
        if self.provenance is Provenance.UNAVAILABLE:
            if self.value is not None:
                raise ProvenanceError(
                    f"R1: unavailable metric must have value=None, got {self.value!r} ({self.name})"
                )
            if not self.reason:
                raise ProvenanceError(f"R1: unavailable metric needs a reason ({self.name})")
            return

        if _in_namespace(self.name, COUNTER_NAMESPACES) and not _in_namespace(
            self.name, EXACT_NCU_NAMESPACES
        ):
            if self.provenance is not Provenance.EXACT:
                raise ProvenanceError(
                    f"R2: counter-namespace metric {self.name!r} may only be EXACT "
                    f"(got {self.provenance.value}); counters are unavailable here"
                )
            # R2 加固（R-11）：连 EXACT 也不行 —— 除非这个计数器名被显式允许。
            # 当前允许集为空，所以"在 ncu 计数器名下填一个数"在类型层面不可能发生。
            if self.name not in EXACT_COUNTER_ALLOWLIST:
                raise ProvenanceError(
                    f"R2: counter-namespace metric {self.name!r} carries a value, but no counter "
                    f"backend can produce it on this machine; report it as "
                    f"Metric.unavailable(name, reason) instead. If a real counter source is added, "
                    f"register the name in EXACT_COUNTER_ALLOWLIST."
                )
        # R4 先于 R3：ncu 的 launch__ / gpu__time_duration 命名空间只允许 EXACT
        if _in_namespace(self.name, ("gpu__time_duration", "launch__")):
            if self.provenance is not Provenance.EXACT:
                raise ProvenanceError(
                    f"R4: {self.name!r} is only allowed with EXACT provenance "
                    f"(got {self.provenance.value})"
                )
        elif self.provenance is not Provenance.EXACT and not _in_namespace(self.name, OUR_PREFIXES):
            raise ProvenanceError(
                f"R3: non-exact metric {self.name!r} must use one of {OUR_PREFIXES} "
                f"(got {self.provenance.value}) - never publish derived values under ncu names"
            )
        # R-12 命名纪律：meas__ 承诺"这是实测读数"，出处就必须是 measured。
        # 由实测样本再算出来的统计量（占比/均值/离散度）请用 ks__ + derived。
        if self.name.startswith("meas__") and self.provenance is not Provenance.MEASURED:
            raise ProvenanceError(
                f"R-12: {self.name!r} is named meas__ but has provenance "
                f"{self.provenance.value}; a meas__ metric must be an actual measurement "
                f"(use ks__ + derived for statistics computed from measurements)"
            )

    # ---- 便捷构造 ----
    @staticmethod
    def exact(name: str, value: float | int | str, unit: str = "", source: str = "",
              note: str = "", resolution: str = "") -> "Metric":
        return Metric(name, value, Provenance.EXACT, unit=unit, source=source, note=note,
                      resolution=resolution)

    @staticmethod
    def derived(name: str, value: float | int, unit: str = "", source: str = "",
                note: str = "", resolution: str = "") -> "Metric":
        return Metric(name, value, Provenance.DERIVED, unit=unit, source=source, note=note,
                      resolution=resolution)

    @staticmethod
    def measured(name: str, value: float | int, unit: str = "", source: str = "",
                 note: str = "", resolution: str = "") -> "Metric":
        return Metric(name, value, Provenance.MEASURED, unit=unit, source=source, note=note,
                      resolution=resolution)

    @staticmethod
    def estimated(name: str, value: float | int, unit: str = "", source: str = "",
                  note: str = "", resolution: str = "") -> "Metric":
        return Metric(name, value, Provenance.ESTIMATED, unit=unit, source=source, note=note,
                      resolution=resolution)

    @staticmethod
    def unavailable(name: str, reason: str, source: str = "",
                    resolution: str = "") -> "Metric":
        return Metric(name, None, Provenance.UNAVAILABLE, source=source, reason=reason,
                      resolution=resolution)

    # ---- 展示 ----
    def render_value(self) -> str:
        if self.provenance is Provenance.UNAVAILABLE:
            return "n/a"
        if isinstance(self.value, float):
            return f"{self.value:.4g}"
        return str(self.value)

    def render_line(self, width: int = 46) -> str:
        left = f"{self.name:<{width}} {self.render_value():>14} {self.unit:<8} {self.provenance.tag}"
        if self.resolution:
            left += f"  [resolution {self.resolution}]"
        if self.note:
            left += f"  {self.note}"
        return left.rstrip()


def _in_namespace(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


@dataclass
class Section:
    """一组同源指标（对标 ncu 的 section）。"""

    name: str
    metrics: list[Metric] = field(default_factory=list)
    note: str = ""

    def add(self, metric: Metric) -> "Section":
        self.metrics.append(metric)
        return self

    def extend(self, metrics: Iterable[Metric]) -> "Section":
        self.metrics.extend(metrics)
        return self

    def validate(self) -> None:
        for m in self.metrics:
            m.validate()

    @property
    def has_values(self) -> bool:
        return any(m.provenance is not Provenance.UNAVAILABLE for m in self.metrics)

    @property
    def provenance(self) -> Provenance:
        """section 头部展示的出处：只统计**有值**的指标，取最保守的那个。

        若把 n/a 条目也算进来，一个含 exact 数据 + 一条 unavailable 的 section
        会整体显示成 [unavailable]，掩盖了里面真实可用的数据。n/a 条目自身的
        `[unavailable]` 标签已经足够显眼，另外用 `n_unavailable` 单独计数。
        全部不可得时返回 UNAVAILABLE。
        """
        order = [Provenance.ESTIMATED, Provenance.MEASURED, Provenance.DERIVED, Provenance.EXACT]
        available = [m for m in self.metrics if m.provenance is not Provenance.UNAVAILABLE]
        if not available:
            return Provenance.UNAVAILABLE
        worst = Provenance.EXACT
        for m in available:
            if order.index(m.provenance) < order.index(worst):
                worst = m.provenance
        return worst

    @property
    def n_unavailable(self) -> int:
        return sum(1 for m in self.metrics if m.provenance is Provenance.UNAVAILABLE)
