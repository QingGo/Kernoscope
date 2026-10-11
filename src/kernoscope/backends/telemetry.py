"""消费平台**已经授权**的遥测端点（DCGM / Prometheus 文本格式）—— NON-GOALS.md 的 N7 择中形态。

为什么是"消费"而不是"绕过"（N7 的原话）：受限集群里管理员**可能已经在提供**这些数 ——
`dcgm-exporter`（K8s GPU operator 常装）、Prometheus 端点、云厂商 GPU 监控 API。读一个平台
自愿开给你的只读端点不是绕过 `RmProfilingAdminOnly`：**端点里有 `DCGM_FI_PROF_*`，只可能是
因为管理员授权了这些字段**；端点里没有，就是没有 —— 不许从 `DCGM_FI_DEV_*` 推 `PROF_*`。
"N7 前提"里查清的边界也要照抄进行为：`DEV_*`（利用率/时钟/功率/能耗/显存）不需要 profiling
权限但是没有 DRAM 字节；`PROF_*` 需要授权。

命名纪律（N3/R3 + R-12）：外部量有它**自己的真名**，我们发布的是**我们自己的名字**
（`meas__telemetry_dcgm_*`），永远不挂 ncu 计数器名，也不做同名冒充。

**两个陷阱在本模块里是构造期不可能事件**（第六轮调研的核心结论）：

  ① `DCGM_FI_PROF_DRAM_ACTIVE`（旧名 `DCGM_FI_PROF_DRAM_UTIL_RATIO`）是**带宽/周期占比**
     （0–1 的 ratio）；`DCGM_FI_DEV_MEM_COPY_UTIL` 与 NVML 的 `utilization.memory` 是
     **时间占比**。**都不是字节，也不是字节/秒。**
  ② 所以：这两类量不许起像字节数的名字、不许带字节单位、不许 × 峰值变成字节。
     这里靠**声明式维度**（`Quantity`）在执行期拦住：`FieldSpec` / `TelemetryMetric` 一构造
     就校验"维度 ↔ 单位 ↔ 名字"三者相容，另有一条 `byte_claim_violations()` 供发布前复核。
     本模块**不提供**、也不调用任何"比例 × 峰值 ⇒ 字节"的换算入口。

单位一律**照抄端点的真单位**（W / mJ / % / MHz / MiB / B/s / ratio / code），不做换算 ——
换算是错误族的高发区（`provenance.py` 里那张单位换算表的注释写了为什么），而这里没有任何
换算的必要。

只依赖标准库：`urllib.request`（带超时）或本地文件读文本；解析 Prometheus 曝光格式。
"""

from __future__ import annotations

import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
from typing import Any, Iterable, Mapping

from ..provenance import (
    COUNTER_NAMESPACES,
    OUR_PREFIXES,
    Metric,
    Provenance,
    provenance_for_name,
)
from ..refusals import RefusalCode, Remedy

#: 一次最多读多少字节。超了只读前 N 字节并**如实记进 skipped**（不许静默截断）。
MAX_BYTES = 8 << 20

PROBE_STATES = ("ok", "not_run", "failed")


class TelemetryTrapError(ValueError):
    """N7 陷阱的构造期防线：比例/时间占比量被当字节发布、或单位与维度不符时抛这个。

    它是 `ValueError` 的子类，所以调用方按参数错误处理即可；消息里点名决策记录（N7），
    免得下一轮有人把这条校验当"多余的严格"删掉。
    """


# ---------------------------------------------------------------------------
# 维度（单位 ∧ 名字的相容性只在这里定义一次）
# ---------------------------------------------------------------------------
class Quantity(str, Enum):
    """一个量的**物理维度**。单位与名字都必须与它相容（`_assert_no_byte_claim`）。"""

    POWER = "power"
    ENERGY = "energy"
    RATIO = "ratio"                     # 0–1 的比例：带宽占比 / 时间占比 / occupancy 都落这里
    PERCENT = "percent"                 # 0–100 的百分比
    FREQUENCY = "frequency"
    MEMORY_SIZE = "memory_size"         # 容量（不是传输量）
    BYTES_PER_SECOND = "bytes_per_second"
    ERROR_CODE = "error_code"
    UNKNOWN = "unknown"                 # 映射表不允许用它（见 FieldSpec.__post_init__）

    @property
    def is_ratio_like(self) -> bool:
        """带宽百分比 / 时间占比 —— **N7 两个陷阱的共同维度**，字节声称在这里被拒。"""
        return self in (Quantity.RATIO, Quantity.PERCENT)

    @property
    def is_byte_dimension(self) -> bool:
        return self in (Quantity.MEMORY_SIZE, Quantity.BYTES_PER_SECOND)


#: 每种维度**唯一**允许的单位。单位是语义的一部分：写错单位比写错数字更坏（N3）。
_REQUIRED_UNIT: dict[Quantity, str] = {
    Quantity.POWER: "W",
    Quantity.ENERGY: "mJ",
    Quantity.RATIO: "ratio",
    Quantity.PERCENT: "%",
    Quantity.FREQUENCY: "MHz",
    Quantity.MEMORY_SIZE: "MiB",
    Quantity.BYTES_PER_SECOND: "B/s",
    Quantity.ERROR_CODE: "code",
}

_BYTE_BASES = frozenset({"B", "KB", "MB", "GB", "TB", "PB",
                         "KIB", "MIB", "GIB", "TIB", "PIB"})

#: 像"字节数"的名字（`..._bytes` / `..._b` / `byte`）。比例量不许起这种名字。
_BYTEISH_NAME_RE = re.compile(r"byte|(?:^|_)b(?:_|$)", re.IGNORECASE)


def _byte_base(unit: str) -> str | None:
    """`"B"` / `"MiB"` / `"B/s"` → 字节基单位；不是字节单位就返回 None。"""
    u = (unit or "").strip().upper()
    if u.endswith("/S"):
        u = u[:-2]
    return u if u in _BYTE_BASES else None


def _assert_name_discipline(name: str) -> None:
    """N3/R3：外部量用自己的前缀（`meas__`/`ks__`），**永远**不挂 ncu 计数器名。"""
    if name.startswith(COUNTER_NAMESPACES):
        raise TelemetryTrapError(
            f"N3/R2: {name!r} 是 ncu 计数器命名空间 —— 平台遥测读到的量不许冒充计数器读数，"
            f"请用自有前缀 {OUR_PREFIXES}")
    if not name.startswith(OUR_PREFIXES):
        raise TelemetryTrapError(
            f"N3/R3: 外部遥测量必须落在自有前缀 {OUR_PREFIXES} 下（得到 {name!r}）")
    for ns in COUNTER_NAMESPACES:
        if ns in name:
            raise TelemetryTrapError(
                f"N3/R2: {name!r} 里混进了计数器命名空间 {ns!r} —— 一个量只有一个身份")


def _assert_no_byte_claim(quantity: Quantity, name: str, unit: str) -> None:
    """**N7 的机器化执行点**：比例/时间占比量既不许带字节单位，也不许起字节名字。

    DRAM_ACTIVE 是带宽占比、`utilization.memory` 是时间占比 —— 把它们当 `dram__bytes` 写出去
    是本项目定义的**研究诚信事故**（NON-GOALS.md N7 的复审触发信号里点名的就是这一条）。
    """
    if quantity.is_ratio_like:
        base = _byte_base(unit)
        if base is not None:
            raise TelemetryTrapError(
                f"N7: {name!r} 的维度是 {quantity.value}（比例/占比），却被声明成字节单位 "
                f"{unit!r} —— 比例 × 峰值**不是**字节，这个乘法在本模块里不存在（研究诚信事故）")
        if _BYTEISH_NAME_RE.search(name):
            raise TelemetryTrapError(
                f"N7: {name!r} 的维度是 {quantity.value}（比例/占比），名字却像字节数 —— "
                f"`DRAM_ACTIVE` 是带宽占比、`utilization.memory` 是时间占比，都不是字节")
    required = _REQUIRED_UNIT.get(quantity)
    if required is None:                     # UNKNOWN：没法定单位，因此只做上面的名字检查
        return
    if (unit or "").strip() != required:
        raise TelemetryTrapError(
            f"{name!r}: 维度 {quantity.value} 的唯一合法单位是 {required!r}，得到 {unit!r} —— "
            f"单位是语义的一部分，本模块不做换算也不猜单位")


# ---------------------------------------------------------------------------
# Prometheus 曝光格式解析（只用标准库）
# ---------------------------------------------------------------------------
_METRIC_NAME_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_LABEL_PAIR_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"$')
_TIMESTAMP_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

_NON_FINITE = {
    "nan": "nan", "+nan": "nan", "-nan": "nan",
    "inf": "+inf", "+inf": "+inf", "infinity": "+inf", "+infinity": "+inf",
    "-inf": "-inf", "-infinity": "-inf",
}


@dataclass(frozen=True)
class PrometheusSeries:
    """一条序列：`metric_name{labels}` → 一个采样值。"""

    key: str
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float
    line: int
    timestamp_ms: float | None = None


@dataclass(frozen=True)
class PrometheusParse:
    """解析结果。`values` 是 `parse_prometheus_text` 的返回形状，其余字段保留过程事实。

    `skipped` 的值是**机器可读的原因**（`nan` / `+inf` / `-inf` / `bad_value` /
    `malformed_line` / ...），键是序列键（知道是哪个字段时）或 `line <n>`；
    `NaN`/`±Inf` **绝不**被折算成 0（那正是 `other_procs_probe` 那条纪律的同类错误）。
    """

    values: dict[str, float] = field(default_factory=dict)
    series: tuple[PrometheusSeries, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)
    duplicates: tuple[str, ...] = ()
    comments_ignored: int = 0
    newest_timestamp_ms: float | None = None

    @property
    def n_series(self) -> int:
        return len(self.series)


def _unescape(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append({"n": "\n", '"': '"', "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def series_key(name: str, labels: "Mapping[str, str] | Iterable[tuple[str, str]]" = ()) -> str:
    """`(name, labels)` → 规范序列键：标签**按键名排序**后重新转义。

    排序是为了让同一个序列在两次抓取里得到同一个键（端点给出的标签顺序不保证稳定）。
    """
    items = labels.items() if isinstance(labels, Mapping) else labels
    pairs = sorted((str(k), str(v)) for k, v in items)
    if not pairs:
        return name
    body = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    return f"{name}{{{body}}}"


def split_series_key(key: str) -> tuple[str, dict[str, str]]:
    """规范序列键 → `(字段名, 标签)`；没有标签时标签是空字典。"""
    pos = key.find("{")
    if pos < 0:
        return key, {}
    name, body = key[:pos], key[pos + 1:]
    if not body.endswith("}"):
        return name, {}
    labels: dict[str, str] = {}
    for piece in _split_label_body(body[:-1]):
        m = _LABEL_PAIR_RE.match(piece)
        if m:
            labels[m.group(1)] = _unescape(m.group(2))
    return name, labels


def _split_label_body(body: str) -> list[str]:
    """按引号外的逗号切分标签体（引号内的逗号不是分隔符）。"""
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    esc = False
    for c in body:
        if in_quote:
            buf.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_quote = False
            continue
        if c == '"':
            in_quote = True
            buf.append(c)
        elif c == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_labels(body: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for piece in _split_label_body(body):
        m = _LABEL_PAIR_RE.match(piece)
        if m is None:
            raise ValueError(f"bad label {piece!r}")
        out.append((m.group(1), _unescape(m.group(2))))
    return out


def _split_line(line: str) -> "tuple[str, list[tuple[str, str]], str] | str":
    """`metric{labels} value [ts]` → `(name, labels, rest)`；读不懂就返回原因字符串。"""
    m = _METRIC_NAME_RE.match(line)
    if m is None or m.start() != 0:
        return "malformed_line"
    name = m.group(0)
    i = m.end()
    labels: list[tuple[str, str]] = []
    if i < len(line) and line[i] == "{":
        j = i + 1
        in_quote = False
        esc = False
        while j < len(line):
            c = line[j]
            if in_quote:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_quote = False
            elif c == '"':
                in_quote = True
            elif c == "}":
                break
            j += 1
        if j >= len(line) or line[j] != "}" or in_quote:
            return "unbalanced_labels"
        try:
            labels = _parse_labels(line[i + 1:j])
        except ValueError:
            return "malformed_labels"
        i = j + 1
    return name, labels, line[i:].strip()


def parse_prometheus_text_detailed(text: str) -> PrometheusParse:
    """把 Prometheus 曝光文本解析成序列，并**如实记录**每一条被跳过的原因。

    认得的：`metric{label="v",...} value [timestamp]`、科学计数法、
    `# HELP` / `# TYPE` / `# EOF` 等注释行、OpenMetrics 的 exemplar（`# {..} 1.0`）尾巴。
    `NaN` / `+Inf` / `-Inf` 一律**跳过并记原因**（`skipped`），绝不折算成 0。
    同一序列重复出现时后者覆盖前者，并把键记进 `duplicates`。
    """
    values: dict[str, float] = {}
    series: list[PrometheusSeries] = []
    skipped: dict[str, str] = {}
    duplicates: list[str] = []
    comments = 0
    newest_ts: float | None = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments += 1                       # HELP / TYPE / EOF：注释不是数据
            continue
        split = _split_line(line)
        if isinstance(split, str):
            skipped[f"line {lineno}"] = split
            continue
        name, labels, rest = split
        key = series_key(name, labels)
        tokens = rest.split()
        if not tokens:
            skipped[key] = "missing_value"
            continue
        tail = tokens[1:]
        # OpenMetrics exemplar：`... value # {labels} exemplar_value` —— 值仍取第一个 token。
        if len(tail) >= 2 and tail[0] == "#":
            tail = tail[2:]
        if len(tail) > 1:
            skipped[key] = "extra_tokens"
            continue
        timestamp: float | None = None
        if tail:
            if not _TIMESTAMP_RE.match(tail[0]):
                skipped[key] = "bad_timestamp"
                continue
            timestamp = float(tail[0])
            if not math.isfinite(timestamp):
                skipped[key] = "bad_timestamp"
                continue
            newest_ts = timestamp if newest_ts is None else max(newest_ts, timestamp)
        reason = _NON_FINITE.get(tokens[0].lower())
        if reason is not None:
            skipped[key] = reason               # NaN/Inf ≠ 0：跳过并说明，绝不冒充读数
            continue
        try:
            value = float(tokens[0])
        except ValueError:
            skipped[key] = "bad_value"
            continue
        if not math.isfinite(value):
            skipped[key] = "nan" if math.isnan(value) else (
                "+inf" if value > 0 else "-inf")
            continue
        if key in values:
            duplicates.append(key)
        values[key] = value
        series.append(PrometheusSeries(key, name, tuple(sorted(labels)), value, lineno,
                                       timestamp))
    return PrometheusParse(values=values, series=tuple(series), skipped=skipped,
                           duplicates=tuple(duplicates), comments_ignored=comments,
                           newest_timestamp_ms=newest_ts)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """`metric{label="v",...} value` → `{规范序列键: 数值}`（`# HELP`/`# TYPE` 忽略）。

    键是**规范序列键**（`series_key`：标签按键名排序），因此同一字段的多卡序列不会互相
    覆盖；要按字段取全部序列用 `values_for_field`，要取单值用 `single_value_for_field`。
    `NaN`/`±Inf` 与坏行被跳过，原因见 `parse_prometheus_text_detailed(text).skipped`。
    """
    return parse_prometheus_text_detailed(text).values


def values_for_field(values: Mapping[str, float], fieldname: str) -> list[tuple[str, float]]:
    """某个 DCGM 字段的**全部**序列（按序列键排序），不做任何聚合。"""
    return sorted((k, v) for k, v in values.items() if split_series_key(k)[0] == fieldname)


def single_value_for_field(values: Mapping[str, float], fieldname: str) -> float | None:
    """只有当该字段**恰好一条**序列时才给值；多卡时不猜（不求和、不取最大）。"""
    hits = values_for_field(values, fieldname)
    return hits[0][1] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# DCGM 字段映射（外部真名 → 我们自己的名字 + 单位 + 一行语义）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """一条 DCGM 字段 → 我们的指标名的映射。

    `requires_profiling` 必须与字段名的 `DCGM_FI_PROF_` 前缀一致（构造期强制）：
    这个布尔值决定"缺了它意味着什么"，写错就等于把"平台没授权"讲成"平台没装"。
    `alias_of` 用于同一物理量的两个 DCGM 拼写（新老版本），避免同时报两条 missing。
    """

    field: str
    name: str
    unit: str
    quantity: Quantity
    semantics: str
    requires_profiling: bool = False
    alias_of: str = ""

    def __post_init__(self) -> None:
        if not self.field.startswith("DCGM_FI_"):
            raise TelemetryTrapError(
                f"{self.field!r} 不是 DCGM 字段名：这张表只登记端点的**外部真名**")
        if "\n" in self.semantics or not self.semantics.strip():
            raise TelemetryTrapError(f"{self.field}: 语义必须是一行非空说明")
        if self.quantity is Quantity.UNKNOWN:
            raise TelemetryTrapError(
                f"{self.field}: 映射表不许用 UNKNOWN 维度 —— 无法确定单位/语义的字段应当"
                f"留在 missing 里报 semantics_unknown，而不是进这张表")
        if self.requires_profiling is not self.field.startswith("DCGM_FI_PROF_"):
            raise TelemetryTrapError(
                f"{self.field}: requires_profiling 必须与 DCGM_FI_PROF_ 前缀一致 ——"
                f"它决定「端点里没有它」意味着什么，不许写错")
        _assert_name_discipline(self.name)
        _assert_no_byte_claim(self.quantity, self.name, self.unit)


#: 顺序即发布顺序。单位照抄端点真单位（不做换算）。
DCGM_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "DCGM_FI_DEV_POWER_USAGE", "meas__telemetry_dcgm_power_w", "W",
        Quantity.POWER,
        "整卡瞬时功耗；设备级管理读数，不需要 profiling 权限",
    ),
    FieldSpec(
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "meas__telemetry_dcgm_energy_total_mj", "mJ",
        Quantity.ENERGY,
        "设备累计能耗（单调累加器，毫焦）；窗口能耗要取两次读数的差值，不是瞬时功率",
    ),
    FieldSpec(
        "DCGM_FI_DEV_GPU_UTIL", "meas__telemetry_dcgm_gpu_util_pct", "%",
        Quantity.PERCENT,
        "至少有一个 kernel 在执行的采样周期占比（**时间占比**，不是算力利用率、不是字节）",
    ),
    FieldSpec(
        "DCGM_FI_DEV_MEM_COPY_UTIL", "meas__telemetry_dcgm_mem_copy_util_pct", "%",
        Quantity.PERCENT,
        "显存接口处于忙碌的采样周期占比（**时间占比**，同 NVML 的 utilization.memory；"
        "不是带宽、更不是字节）",
    ),
    FieldSpec(
        "DCGM_FI_DEV_SM_CLOCK", "meas__telemetry_dcgm_sm_clock_mhz", "MHz",
        Quantity.FREQUENCY,
        "当前 SM 时钟",
    ),
    FieldSpec(
        "DCGM_FI_DEV_MEM_CLOCK", "meas__telemetry_dcgm_mem_clock_mhz", "MHz",
        Quantity.FREQUENCY,
        "当前显存时钟",
    ),
    FieldSpec(
        "DCGM_FI_DEV_FB_USED", "meas__telemetry_dcgm_fb_used_mib", "MiB",
        Quantity.MEMORY_SIZE,
        "已用帧缓冲**容量**（容量不是传输量；不要和 PCIe/DRAM 的字节速率混为一谈）",
    ),
    FieldSpec(
        "DCGM_FI_DEV_FB_FREE", "meas__telemetry_dcgm_fb_free_mib", "MiB",
        Quantity.MEMORY_SIZE,
        "空闲帧缓冲**容量**（容量不是传输量）",
    ),
    FieldSpec(
        "DCGM_FI_DEV_XID_ERRORS", "meas__telemetry_dcgm_last_xid_error_code", "code",
        Quantity.ERROR_CODE,
        "最近一次 XID 错误的**编号**（不是错误次数；次数只能靠相邻两次读数是否变化来推）",
    ),
    FieldSpec(
        "DCGM_FI_PROF_SM_OCCUPANCY", "meas__telemetry_dcgm_sm_occupancy_ratio", "ratio",
        Quantity.RATIO,
        "驻留 warp 数占 SM 上限的**比例**（0–1；不是百分比整数、不是字节、不是吞吐）",
        requires_profiling=True,
    ),
    FieldSpec(
        "DCGM_FI_PROF_DRAM_ACTIVE", "meas__telemetry_dcgm_dram_active_ratio", "ratio",
        Quantity.RATIO,
        "DRAM 处于活跃的周期占比（0–1 的**带宽/时间比例**；绝不是字节数，也不是字节/秒）",
        requires_profiling=True,
    ),
    # 同一物理量的老拼写（DCGM 3.x 之前）。两条映射共用一个发布名：先到者发布，
    # 后到者记 duplicate_published_name，绝不并成两个数。
    FieldSpec(
        "DCGM_FI_PROF_DRAM_UTIL_RATIO", "meas__telemetry_dcgm_dram_active_ratio", "ratio",
        Quantity.RATIO,
        "DRAM 活跃周期占比的老拼写（与 DCGM_FI_PROF_DRAM_ACTIVE 同义；"
        "0–1 的比例，绝不是字节）",
        requires_profiling=True, alias_of="DCGM_FI_PROF_DRAM_ACTIVE",
    ),
    FieldSpec(
        "DCGM_FI_PROF_PCIE_TX_BYTES", "meas__telemetry_dcgm_pcie_tx_bytes_per_s", "B/s",
        Quantity.BYTES_PER_SECOND,
        "PCIe 发送方向吞吐（字节/秒的**速率**；不是窗口内总字节）",
        requires_profiling=True,
    ),
    FieldSpec(
        "DCGM_FI_PROF_PCIE_RX_BYTES", "meas__telemetry_dcgm_pcie_rx_bytes_per_s", "B/s",
        Quantity.BYTES_PER_SECOND,
        "PCIe 接收方向吞吐（字节/秒的**速率**；不是窗口内总字节）",
        requires_profiling=True,
    ),
)

MAPPING: dict[str, FieldSpec] = {spec.field: spec for spec in DCGM_FIELDS}
#: 发布名 → 字段名。别名与主字段同名时保留**主字段**（先出现的那条），
#: 于是用我们自己的名字点单时追讨的是主字段，而不是老拼写。
FIELD_BY_NAME: dict[str, str] = {}
for _spec in DCGM_FIELDS:
    FIELD_BY_NAME.setdefault(_spec.name, _spec.field)

#: 需要平台授权的字段（`DCGM_FI_PROF_*`）。**它们的缺席本身就是结论**：
#: 这次端点没有给出 profiling 遥测 —— 不等于管理员拒绝了（可能只是导出器没配这几项）。
PROFILING_FIELDS: tuple[str, ...] = tuple(
    spec.field for spec in DCGM_FIELDS if spec.requires_profiling)

#: 语义上互不相同的 profiling 量：老拼写（别名）由主字段代表，**一个量只报一条缺席**。
PROFILING_PRIMARY_FIELDS: tuple[str, ...] = tuple(
    spec.field for spec in DCGM_FIELDS if spec.requires_profiling and not spec.alias_of)


# ---------------------------------------------------------------------------
# 缺件原因 → 人话 / 结构化拒绝（复用闭集，不新增 code）
# ---------------------------------------------------------------------------
_REASON_TEXT: dict[str, str] = {
    "absent": "端点没有导出这个字段",
    "semantics_unknown": "端点报了这个字段，但它的单位/语义我们无法确认：按未知处理，不猜、不当 0",
    "unknown_field": "请求的字段不在我们的映射表里（既不是 DCGM 字段名也不是我们发布的名字）",
    "duplicate_published_name": "同一个发布名已经由另一个字段给出，不重复发布",
    "nan": "读数是 NaN：这是「读不到」，不是 0，所以不给数",
    "+inf": "读数是 +Inf：不是 0，也不是有限读数",
    "-inf": "读数是 -Inf：不是 0，也不是有限读数",
    "no_source": "没有给遥测端点",
    "io_error": "读不到这个文件",
    "network_error": "连不上这个端点",
    "timeout": "读这个端点超时",
    "http_error": "端点返回了错误状态码",
    "unsupported_scheme": "只支持 http(s):// URL 或本地文件路径",
    "bad_timeout": "超时必须是正数（秒）",
    "no_prometheus_series": "这个来源里一行 Prometheus 序列都读不出来 —— 这是「读不出数据」，"
                            "不是「没有数据」",
    "truncated": f"来源大于 {MAX_BYTES} 字节，只读了开头一段",
    "internal_error": "读取过程中出现了未预期的错误",
}

#: 原因 → **既有**拒绝码（闭集来自 refusals.py，本模块不新增 code）：
#:   * 字段缺席 / 语义未知 → `metric_absent`、`env_reading_unknown`（未知 ≠ 零）；
#:   * 端点读不到（IO/网络/超时/HTTP/内容读不出）→ `env_reading_unknown`
#:     （"读数读不到"与"读数等于 0"必须能分开，这是 R-96 那一族的纪律）。
_REASON_CODE: dict[str, RefusalCode] = {
    "absent": RefusalCode.METRIC_ABSENT,
    "semantics_unknown": RefusalCode.ENV_READING_UNKNOWN,
    "unknown_field": RefusalCode.METRIC_ABSENT,
    "duplicate_published_name": RefusalCode.METRIC_ABSENT,
    "nan": RefusalCode.ENV_READING_UNKNOWN,
    "+inf": RefusalCode.ENV_READING_UNKNOWN,
    "-inf": RefusalCode.ENV_READING_UNKNOWN,
    "no_source": RefusalCode.ENV_READING_UNKNOWN,
    "io_error": RefusalCode.ENV_READING_UNKNOWN,
    "network_error": RefusalCode.ENV_READING_UNKNOWN,
    "timeout": RefusalCode.ENV_READING_UNKNOWN,
    "http_error": RefusalCode.ENV_READING_UNKNOWN,
    "unsupported_scheme": RefusalCode.ENV_READING_UNKNOWN,
    "bad_timeout": RefusalCode.ENV_READING_UNKNOWN,
    "no_prometheus_series": RefusalCode.ENV_READING_UNKNOWN,
    "truncated": RefusalCode.ENV_READING_UNKNOWN,
    "internal_error": RefusalCode.INTERNAL_UNKNOWN,
}

#: 原因 → 处方。N7 明确要求把"向平台要一个只读遥测端点"写进拒绝处方。
_REASON_REMEDY: dict[str, Remedy] = {
    "absent": Remedy(
        "这个字段没在这个端点上导出：请平台管理员把它加进 dcgm-exporter 的字段列表"
        "（DEV_* 只要导出配置；PROF_* 还需要平台授权 profiling 计数器 —— 端点里有它，"
        "只可能是因为管理员已经授权）",
        cost_kind="manual"),
    "semantics_unknown": Remedy(
        "端点报了我们的映射表不认识的字段：先按 DCGM 文档确认它的单位与语义，"
        "再把一条带单位的映射加进 backends/telemetry.py 的 DCGM_FIELDS；"
        "在此之前按未知处理（不猜、也不当 0）",
        cost_kind="manual"),
    "no_source": Remedy(
        "给一个平台已经授权的只读端点：dcgm-exporter 的 /metrics（Prometheus 文本格式）"
        "或已抓取下来的文本文件；没有端点时请平台开一个只读的，这是最便宜的一条通道",
        cost_kind="manual"),
}


def _reason_key(reason: str) -> str:
    """`"http_error:503"` → `"http_error"`（带细节的原因按前缀归类）。"""
    return (reason or "").split(":", 1)[0].strip()


def reason_text_for(reason: str) -> str:
    """原因 → 一句中文人话（不编：查不到就把原因原样给出）。"""
    key = _reason_key(reason)
    text = _REASON_TEXT.get(key)
    if text is None:
        return reason or "未知原因"
    detail = reason[len(key) + 1:] if reason.startswith(key + ":") else ""
    return f"{text}（{detail}）" if detail else text


def reason_code_for(reason: str) -> RefusalCode:
    """原因 → **既有**拒绝码闭集里的一个（查不到就用 `internal_unknown`，出现即为 bug 信号）。"""
    return _REASON_CODE.get(_reason_key(reason), RefusalCode.INTERNAL_UNKNOWN)


def reason_remedy_for(reason: str) -> Remedy | None:
    """原因 → 本模块的处方（查不到就返回 None，交给 `default_remedy` 兜底）。"""
    return _REASON_REMEDY.get(_reason_key(reason))


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetryMetric:
    """一条已发布的遥测读数：我们自己的名字 + 真单位 + 语义 + 端点署名。"""

    name: str
    value: float
    unit: str
    semantics: str
    source: str
    quantity: Quantity
    field: str = ""

    def __post_init__(self) -> None:
        _assert_name_discipline(self.name)
        _assert_no_byte_claim(self.quantity, self.name, self.unit)
        if not self.semantics.strip() or "\n" in self.semantics:
            raise TelemetryTrapError(f"{self.name}: 语义必须是一行非空说明")
        if not self.source.strip():
            raise TelemetryTrapError(
                f"{self.name}: 必须带端点署名（N3：一个挂错名字/来源的数字比没有数字更坏）")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TelemetryTrapError(f"{self.name}: 值必须是数值，得到 {self.value!r}")
        if not math.isfinite(float(self.value)):
            raise TelemetryTrapError(
                f"{self.name}: NaN/Inf 不许进指标（它们在解析期就被跳过了，走 missing 的原因）")

    def to_metric_dict(self) -> dict[str, Any]:
        """报告/决策包形状的一行（`provenance` 由命名纪律推出，不手填）。"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "provenance": provenance_for_name(self.name, self.value),
            "source": self.source,
            "note": f"{self.semantics}；端点字段 {self.field}",
            "quantity": self.quantity.value,
        }

    def to_provenance_metric(self) -> Metric:
        """转成项目自己的 `provenance.Metric`（可直接进 Section）。"""
        return Metric(
            name=self.name,
            value=self.value,
            provenance=Provenance.MEASURED,
            unit=self.unit,
            source=self.source,
            note=f"{self.semantics}（端点字段 {self.field}）",
        )


def byte_claim_violations(metrics: Iterable[TelemetryMetric]) -> list[str]:
    """发布前复核：有没有比例/时间占比量被当字节发布（**正常情况必须返回空**）。

    构造期已经拦过一次；这条是给接线处（报告/决策包）的最后一道闸 —— N7 把"百分比当字节"
    定为研究诚信事故，所以它值得两条独立的防线。
    """
    bad: list[str] = []
    for m in metrics:
        try:
            _assert_no_byte_claim(m.quantity, m.name, m.unit)
        except TelemetryTrapError as exc:
            bad.append(str(exc))
    return bad


@dataclass
class TelemetryResult:
    """一次遥测消费的结果。

    `probe` 三态（与 `nvml.SamplerResult.other_procs_probe` 同一套纪律）：
        ok       端点读到了、解析出了序列（有没有某个字段是另一件事，看 `missing`）
        not_run  调用方根本没给端点
        failed   端点读不到/读不出（IO、网络、超时、HTTP 错误、一行序列都解析不出）
    **"failed" / "missing" 与"实测到 0"必须能分开**：读数 0 是一条真读数（在 `metrics` 里），
    缺席是一条带原因的 `missing`，读不到是 `probe="failed"`。三者永不互相冒充。
    """

    metrics: list[TelemetryMetric] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)
    probe: str = "not_run"
    source: str = ""
    reason: str = ""
    series_total: int = 0
    dcgm_fields_seen: tuple[str, ...] = ()
    profiling_present: bool = False
    skipped: dict[str, str] = field(default_factory=dict)
    duplicates: tuple[str, ...] = ()
    newest_timestamp_ms: float | None = None

    def __post_init__(self) -> None:
        if self.probe not in PROBE_STATES:
            raise ValueError(f"probe 必须是 {PROBE_STATES} 之一，得到 {self.probe!r}")
        if self.probe == "ok" and not self.source:
            raise ValueError("probe=ok 必须带 source（没有端点的读数不存在）")

    @property
    def ok(self) -> bool:
        return self.probe == "ok"

    @property
    def profiling_absent(self) -> tuple[str, ...]:
        """这次没有给出读数的 profiling 量（缺席是结论，不是异常）。

        按**发布名**判定：老拼写（别名）给了读数时，主字段也算覆盖到了。
        """
        present = {m.name for m in self.metrics}
        return tuple(f for f in PROFILING_PRIMARY_FIELDS if MAPPING[f].name not in present)

    def names(self) -> list[str]:
        return [m.name for m in self.metrics]

    def find(self, name: str) -> TelemetryMetric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_metrics(self) -> dict[str, dict[str, Any]]:
        """`{指标名: 报告形状的一行}`（发布前跑一遍字节声称复核）。"""
        violations = byte_claim_violations(self.metrics)
        if violations:
            raise TelemetryTrapError("；".join(violations))
        return {m.name: m.to_metric_dict() for m in self.metrics}

    def to_provenance_metrics(self) -> list[Metric]:
        """有读数的字段 → `provenance.Metric` 列表（可直接 `Section.extend`）。"""
        return [m.to_provenance_metric() for m in self.metrics]

    def unavailable_metrics(self) -> list[Metric]:
        """缺席字段 → 带**结构化拒绝**的 `Metric.unavailable`（复用既有 code 与处方）。

        没有我们的发布名的字段（例如语义未知的外部字段）不在这里出现：编不出名字的量
        不许出现在报告里（N3）。
        """
        out: list[Metric] = []
        seen: set[str] = set()
        for key, why in self.missing.items():
            spec = MAPPING.get(key)
            if spec is None:
                continue
            if spec.name in seen:       # 别名与主字段同名时只给一条 n/a
                continue
            seen.add(spec.name)
            out.append(Metric.unavailable(
                spec.name,
                reason_text_for(why),
                unit=spec.unit,
                source=self.source,
                note=f"端点字段 {spec.field}；原因码 {why}",
                code=reason_code_for(why),
                remedy=reason_remedy_for(why),
                severity="degraded",
            ))
        return out


# ---------------------------------------------------------------------------
# 消费端点
# ---------------------------------------------------------------------------
def _open_url(url: str, timeout_s: float):
    """`urllib.request` 的薄包装（测试可以替换它来稳定地制造网络失败）。"""
    return urllib.request.urlopen(url, timeout=timeout_s)  # noqa: S310 - 只读遥测端点


def _read_source(source: str, timeout_s: float) -> "tuple[str | None, str | None, bool]":
    """读来源 → `(text, 失败原因, 是否被截断)`。**不抛异常**（异常在边界转成原因）。"""
    if timeout_s <= 0:
        return None, "bad_timeout", False
    if source.startswith(("http://", "https://")):
        try:
            with _open_url(source, timeout_s) as resp:
                body = resp.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as exc:
            return None, f"http_error:{exc.code}", False
        except urllib.error.URLError as exc:
            return None, f"network_error:{exc.reason}", False
        except TimeoutError:
            return None, "timeout", False
        except OSError as exc:
            return None, f"network_error:{exc}", False
        except Exception as exc:                # 兜底：绝不让异常漏出这个函数
            return None, f"internal_error:{type(exc).__name__}", False
        truncated = len(body) > MAX_BYTES
        return body[:MAX_BYTES].decode("utf-8", "replace"), None, truncated
    if "://" in source:
        return None, "unsupported_scheme", False
    try:
        with open(source, "rb") as fh:
            body = fh.read(MAX_BYTES + 1)
    except OSError as exc:
        return None, f"io_error:{exc.__class__.__name__}", False
    truncated = len(body) > MAX_BYTES
    return body[:MAX_BYTES].decode("utf-8", "replace"), None, truncated


def _wanted_fields(wanted: Iterable[str]) -> "tuple[list[str], dict[str, str]]":
    """调用方要的字段 → `(命中的字段名, 未识别的请求 → 原因)`。两种名字都认。"""
    fields: list[str] = []
    bad: dict[str, str] = {}
    for raw in wanted:
        key = str(raw).strip()
        if not key:
            continue
        if key in MAPPING:
            fields.append(key)
        elif key in FIELD_BY_NAME:
            fields.append(FIELD_BY_NAME[key])
        else:
            bad[key] = "unknown_field"
    return fields, bad


def _published_name(spec: FieldSpec, key: str, n_series: int) -> str:
    """多卡时给发布名加一个稳定后缀（单序列保持朴素名，便于下游按名引用）。"""
    if n_series <= 1:
        return spec.name
    _name, labels = split_series_key(key)
    gpu = labels.get("gpu") or labels.get("GPU") or labels.get("device")
    slug = re.sub(r"[^a-z0-9]+", "_", str(gpu).lower()).strip("_") if gpu else ""
    suffix = f"gpu_{slug}" if slug else f"l{sha1(key.encode('utf-8')).hexdigest()[:8]}"
    return f"{spec.name}__{suffix}"


def _first_skip_reason(parsed: PrometheusParse, fieldname: str) -> str | None:
    """这个字段的序列是不是被解析期跳过了（NaN/Inf 等）？有就给出原因。"""
    for key, why in sorted(parsed.skipped.items()):
        if key.startswith("line "):
            continue
        if split_series_key(key)[0] == fieldname:
            return why
    return None


def collect_telemetry(source: str, *, timeout_s: float = 5.0,
                      wanted: "tuple[str, ...]" = ()) -> TelemetryResult:
    """消费一个**已授权**的遥测端点：本地文本文件路径，或 `http(s)://` URL。

    语义边界（照 NON-GOALS.md N7 写，不夸大）：

    * 只读；不发任何写请求；不碰驱动策略、不提权。
    * `wanted` 决定**追讨**哪些字段（缺席会进 `missing`）；它接受 DCGM 字段名或我们发布的
      名字。没有请求的字段只要端点里有、且映射表认得，也一样发布。
    * `DCGM_FI_PROF_*` **总是**参与追讨：它们的缺席本身就是"这次没有 profiling 遥测"的结论
      （`profiling_present=False`）。注意结论的方向：**没有 ≠ 管理员拒绝了** —— 也可能只是
      导出器没配这几项，所以我们只报"没有"。
    * 端点里有、映射表不认得的 `DCGM_FI_*` 字段 → `missing[field] = "semantics_unknown"`
      （单位/语义无法确认就不发布，绝不猜）。
    * 值 `NaN`/`±Inf` → 解析期跳过，字段进 `missing` 并带原因；**永远不折成 0**。
    * 网络/IO/超时/HTTP 错误、以及一行序列都解析不出 → `probe="failed"` + `reason`，
      异常不泄漏。
    """
    src = source if isinstance(source, str) else ""
    if not src.strip():
        return TelemetryResult(probe="not_run", source="", reason="no_source",
                               missing=_profiling_reasons("no_source"))
    src = src.strip()
    text, error, truncated = _read_source(src, timeout_s)
    if error is not None or text is None:
        reason = error or "internal_error"
        return TelemetryResult(probe="failed", source=src, reason=reason,
                               missing=_profiling_reasons(reason), skipped={src: reason})
    parsed = parse_prometheus_text_detailed(text)
    skipped = dict(parsed.skipped)
    if truncated:
        skipped[src] = "truncated"
    if parsed.n_series == 0:
        # 拿到了字节，但一行序列都读不出：这**不是**"端点里没有数据"，是"读不出数据"。
        # 报 ok 会把一次读失败讲成"字段缺席"，那是另一类假绿。
        return TelemetryResult(probe="failed", source=src, reason="no_prometheus_series",
                               missing=_profiling_reasons("no_prometheus_series"),
                               skipped=skipped, duplicates=parsed.duplicates,
                               series_total=0)
    return _build_result(src, parsed, skipped, wanted)


def _profiling_reasons(reason: str) -> dict[str, str]:
    """没读到端点时，profiling 量的缺席原因就是**读取失败的原因**（不是"端点里没有"）。"""
    return {f: reason for f in PROFILING_PRIMARY_FIELDS}


def _build_result(src: str, parsed: PrometheusParse, skipped: dict[str, str],
                  wanted: "tuple[str, ...]") -> TelemetryResult:
    requested, unknown_wanted = _wanted_fields(wanted)
    missing: dict[str, str] = {}

    # 端点里有、但映射表不认得的 DCGM 字段：语义未知，不猜、不发布。
    # **先于** unknown_wanted 处理：一个真的在端点里出现的字段，它的原因应当是
    # "语义未知"，而不是"你点了一个不存在的名字"。
    for key in list(parsed.values) + list(parsed.skipped):
        if key.startswith("line "):
            continue
        base = split_series_key(key)[0]
        if base.startswith("DCGM_FI_") and base not in MAPPING:
            missing.setdefault(base, "semantics_unknown")
    for key, why in unknown_wanted.items():
        missing.setdefault(key, why)

    metrics: list[TelemetryMetric] = []
    taken: dict[str, str] = {}          # 发布名 → 已经发布的字段（防两个字段撞名）
    published: set[str] = set()         # 这次真的给出读数的**字段名**
    published_names: set[str] = set()   # 它们的发布名（别名与主字段共用一个发布名）
    for spec in DCGM_FIELDS:
        hits = values_for_field(parsed.values, spec.field)
        for key, value in hits:
            name = _published_name(spec, key, len(hits))
            owner = taken.get(name)
            if owner is not None and owner != spec.field:
                missing.setdefault(spec.field, "duplicate_published_name")
                continue
            metrics.append(TelemetryMetric(
                name=name, value=float(value), unit=spec.unit, semantics=spec.semantics,
                source=f"{src} [{key}]", quantity=spec.quantity, field=spec.field))
            taken[name] = spec.field
            published.add(spec.field)
            published_names.add(spec.name)

    requested_set = set(requested)
    for spec in DCGM_FIELDS:
        if spec.name in published_names:
            continue        # 别名与主字段共用发布名：有一个给了读数就不算缺席
        if spec.alias_of and spec.field not in requested_set:
            continue        # 没被点名要的别名不单独报缺席：缺席由主字段那一条代表
        if spec.field not in requested_set and not spec.requires_profiling:
            continue        # 没要、又不是 profiling 家族：不往 missing 里塞噪声
        missing.setdefault(spec.field, _first_skip_reason(parsed, spec.field) or "absent")

    fields_seen = tuple(sorted({split_series_key(k)[0] for k in parsed.values
                               if split_series_key(k)[0].startswith("DCGM_FI_")}))
    return TelemetryResult(
        metrics=metrics,
        missing=missing,
        probe="ok",
        source=src,
        series_total=parsed.n_series,
        dcgm_fields_seen=fields_seen,
        profiling_present=any(MAPPING[f].requires_profiling for f in published),
        skipped=skipped,
        duplicates=parsed.duplicates,
        newest_timestamp_ms=parsed.newest_timestamp_ms,
    )


__all__ = [
    "DCGM_FIELDS",
    "FIELD_BY_NAME",
    "MAPPING",
    "MAX_BYTES",
    "PROBE_STATES",
    "PROFILING_FIELDS",
    "PROFILING_PRIMARY_FIELDS",
    "FieldSpec",
    "PrometheusParse",
    "PrometheusSeries",
    "Quantity",
    "TelemetryMetric",
    "TelemetryResult",
    "TelemetryTrapError",
    "byte_claim_violations",
    "collect_telemetry",
    "parse_prometheus_text",
    "parse_prometheus_text_detailed",
    "reason_code_for",
    "reason_remedy_for",
    "reason_text_for",
    "series_key",
    "single_value_for_field",
    "split_series_key",
    "values_for_field",
]
