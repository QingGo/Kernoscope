"""NVTX 阶段归因 —— 把 kernel 时间摊到「训练/推理的哪一段」上。

为什么需要它：kernel 时长只回答「哪个 kernel 慢」，回答不了「慢在哪一段流程里」
（dataloader / forward / backward / optimizer …）。nsys 采集时带上 `--trace=nvtx` 之后，
NVTX 区间就是「哪一段」的官方标注；把 kernel 时间按**区间重叠**摊进去，问题就从
「哪个 kernel」变成「哪一段」，而这一步只需要已经导出的那个 SQLite，不需要重跑负载。

三条口径纪律（这个模块存在的全部理由）：

1. **不假设 schema。** NVTX 数据在 `NVTX_EVENTS` 表里，但列名与列的存在性随 nsys 版本变
   （`end` 可能没有；区间文本可能内联在 `text`，也可能走 `StringIds.textId` 外键；
   也可能两者都有）。所以我们先 `PRAGMA table_info` 看**实际有哪些列**，再按结构
   （有 `start`/`end` 且 `end > start`）挑区间；`eventType` 的数值含义跨版本不稳定，
   我们**不解释它的数值**。表缺失 / 为空 / 只有 open range 时返回 reason + **零数字**：
   `0%` 会被读成「测过了，是 0」，所以我们宁可不给数。

2. **区间重叠，按重叠时长计入。** kernel 与区间在同一采集时钟（`start`/`end`，ns）上做
   区间运算：跨了两个区间的 kernel 会被**切开**，只把重叠的那一段计入各自区间。
   「kernel 属于它开始的那个区间」这种朴素答案是错的（有测试钉死）。
   同一 kernel 落在嵌套的外层与内层区间时**两边都算**（`depth` 区分层级）——
   代价是各阶段之和可以 >100%，所以 `to_metrics()` 绝不把这些阶段当成划分
   （`basis` 里写明；份额也不做上钳位）。

3. **未归因单独报。** `unattributed_ms` = 总 kernel 时间 − 落在**任一**区间内的时间
   （按每个 kernel 的重叠算，不是「总时间 − 各阶段之和」——有嵌套时后者会算成负数）。

**分组口径（一个必须说清的假设）**：归因只在「同一条时间线」上做。NVTX 表有 `globalTid`，
CUPTI kernel 表有 `deviceId`/`streamId`；两张表在导出库里**没有**可验证的映射
（用 `CUPTI_ACTIVITY_KIND_RUNTIME.correlationId` 去连 host 线程是一条可能的路径，但那要求
runtime activity 开着、且 CUDA graph 的 correlationId 为 0 —— 我们**没有**在真机上验证过，
所以不实现，也不假装它成立）。因此：仅当两张表恰好都暴露同一个 `globalTid` 列时才按线程
分组；否则退化成**整段采集上的时间重叠**，并把这次退化写进 `basis`。退化是声明的，
不是默认的 —— 多卡/多线程 trace 上它会让「别的时间线」的区间也参与归因，读的人必须知道。

读取风格与错误处理跟随 `backends/nsys.py`：只读连接、`StringIds` 解引用复用它的实现、
sqlite 错误一律包成 `PhaseError`（不让裸 `sqlite3` traceback 逃出去）。时间单位 ns，
对外一律 ms。
"""

from __future__ import annotations

import bisect
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..backends.nsys import _resolve, _string_ids   # StringIds 解引用的单一实现（含踩坑注释）
from ..refusals import RefusalCode

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
NVTX_TABLE = "NVTX_EVENTS"
NS_PER_MS = 1_000_000.0

# 敢于当 join key 的列名。**只有 globalTid**：它是 host 线程 id，NVTX 与（某些导出里的）
# CUPTI 表都可能带；deviceId/streamId 与 NVTX 没有对应关系，绝不能拿来当同一维度。
_THREAD_KEYS = ("globalTid",)

# reason 的前缀是**机器可读的闭集**（人话跟在冒号后面）。to_metrics() 用它映射到
# refusals 闭集里的 code —— 我们不给这个模块发明新的拒绝码（那要改 refusals.py）。
REASON_NO_NVTX = "no_nvtx_ranges"
REASON_NO_KERNELS = "no_kernel_activity"


class PhaseError(RuntimeError):
    """拿不到可归因的输入（文件不在 / 不是 sqlite / 读失败）。与 NsysError 同一性质。"""


@dataclass
class Phase:
    """一个 NVTX 区间上的统计（区间自身与落在它里面的 kernel 时间）。"""

    name: str
    depth: int
    total_ms: float           # 区间自身时长（end - start）
    kernel_ms: float          # 落在本区间内的 kernel 时间（按重叠算，含重叠 kernel 的重数）
    share_pct: float          # 占**全部** kernel 时间的百分比；嵌套时可 >100%，不做钳位
    kernels: int              # 与本区间有正重叠的 kernel 个数


@dataclass
class PhaseAttribution:
    """一次阶段归因的结果。

    `attributed_ms` / `omitted` 是契约之外**多给的两个字段**（都有默认值，构造仍可只用
    前四个参数）：attributed 要按「任一区间的并集」算，事后无法从 phases 反推
    （嵌套时各阶段之和 ≠ 并集），所以存下来而不是让调用方去猜。
    """

    phases: list[Phase]
    unattributed_ms: float
    basis: str
    reason: str = ""
    attributed_ms: float = 0.0    # 落在任一区间内的 kernel 时间（并集口径）
    omitted: int = 0              # 被 top 截掉的阶段个数（basis 里也写了）

    def to_metrics(self) -> list[tuple[str, float | None, str, str]]:
        """(name, value, unit, note) 供 section 组装。

        value=None 表示「这一行拿不到数」——组装的调用方必须把这一行按 unavailable 输出，
        而不是丢掉（B1：口径不成立时那一行要**以 n/a 的形式存在**）。
        reason 非空时**所有** value 都是 None：0 在这里只代表「没测」，不代表「测到 0」。
        """
        if self.reason:
            note = f"unavailable: code={_refusal_code(self.reason).value} | {self.reason}"
            return [
                ("ks__nvtx_attributed_kernel_ms", None, "ms", note),
                ("ks__nvtx_unattributed_kernel_ms", None, "ms", note),
                ("ks__nvtx_phase_count", None, "", note),
                ("ks__nvtx_phase_omitted", None, "", note),
            ]
        name_note = self.basis
        # ms 保留 6 位小数 = 1 ns 分辨率：nsys 的时间戳是整数 ns，再粗就会把
        # 真实的短区间/短 kernel 舍成 0.0 —— 而 0.0 会被读成「测到 0」（口径纪律 1）。
        out: list[tuple[str, float | None, str, str]] = [
            ("ks__nvtx_attributed_kernel_ms", _ms(self.attributed_ms), "ms", name_note),
            ("ks__nvtx_unattributed_kernel_ms", _ms(self.unattributed_ms), "ms", name_note),
            ("ks__nvtx_phase_count", float(len(self.phases)), "", name_note),
            ("ks__nvtx_phase_omitted", float(self.omitted), "", name_note),
        ]
        for i, p in enumerate(self.phases):
            base = f"ks__nvtx_phase_{i}__{_slug(p.name)}"
            # 每个阶段只带自己的事实；口径（basis）挂在上面那几行，避免同一段散文重复 N 遍
            note = (f"区间「{p.name}」深度 {p.depth}，区间自身 {_fmt_ms(p.total_ms)} ms、"
                    f"覆盖 {p.kernels} 个 kernel；share 的分母是**全部** kernel 时间（含未归因），"
                    f"嵌套时各阶段之和可 >100% —— 本表不是划分，口径见 "
                    f"ks__nvtx_attributed_kernel_ms 的 note")
            out.append((f"{base}__kernel_ms", _ms(p.kernel_ms), "ms", note))
            out.append((f"{base}__share_pct", round(p.share_pct, 3), "%", note))
            out.append((f"{base}__total_ms", _ms(p.total_ms), "ms", note))
            out.append((f"{base}__kernels", float(p.kernels), "个", note))
        return out


# ---------------------------------------------------------------------------
# sqlite 读取（只读连接 + 结构探测）
# ---------------------------------------------------------------------------
def _connect_ro(db_path: str) -> sqlite3.Connection:
    """只读打开。文件不在就是调用方的输入错误 → PhaseError（不是「拿不到数」）。"""
    if not os.path.exists(db_path):
        raise PhaseError(f"sqlite not found: {db_path}")
    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:                     # pragma: no cover - 平台相关
        raise PhaseError(f"cannot open sqlite read-only: {db_path} ({e})") from e


def _tables(con: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in con.execute("select name from sqlite_master where type='table'")}


def _columns(con: sqlite3.Connection, table: str) -> dict[str, str]:
    """列名 → 声明类型（实际的列集合，而不是我们记得的列集合）。"""
    return {str(r[1]): str(r[2] or "") for r in con.execute(f'pragma table_info("{table}")')}


@dataclass
class _Range:
    name: str
    start: int
    end: int
    key: object | None = None
    depth: int = 0


@dataclass
class _Kernel:
    start: int
    end: int
    key: object | None = None


@dataclass
class _Loaded:
    """一张表的读取结果：条目 + 事实（给 basis 用）+ 拿不到时的说明。"""

    items: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    problem: str = ""


def _load_ranges(con: sqlite3.Connection, tables: set[str], key_col: str | None) -> _Loaded:
    """读 NVTX 区间。只认**结构**：有 start/end、end 非 NULL、end > start。"""
    out = _Loaded()
    if NVTX_TABLE not in tables:
        out.problem = (f"{REASON_NO_NVTX}: 库里没有 {NVTX_TABLE} 表 —— 采集时没带 "
                       f"`--trace=nvtx`（或这个 nsys 版本不导出 NVTX）")
        return out
    cols = _columns(con, NVTX_TABLE)
    if not cols:
        out.problem = f"{REASON_NO_NVTX}: {NVTX_TABLE} 表没有可读的列"
        return out
    if "start" not in cols:
        out.problem = (f"{REASON_NO_NVTX}: {NVTX_TABLE} 没有 start 列（实际列："
                       f"{', '.join(sorted(cols)) or '无'}）")
        return out
    if "end" not in cols:
        # 有的版本/有的 eventType 只有起点：那只是「某个时刻发生了一件事」，不是区间，
        # 做不了区间运算 —— 必须说出来，不能拿 open range 硬当区间。
        out.problem = (f"{REASON_NO_NVTX}: {NVTX_TABLE} 没有 end 列（只有起点，全是 open "
                       f"range），做不了区间重叠")
        return out

    want = [c for c in ("start", "end", "text", "textId", *_THREAD_KEYS) if c in cols]
    idx = {c: i for i, c in enumerate(want)}
    has_text, has_textid = "text" in idx, "textId" in idx
    ids = _string_ids(con) if has_textid else {}

    items: list[_Range] = []
    open_n = degen_n = null_start = 0
    inline_n = stringid_n = unnamed_n = 0
    for row in con.execute(f'select {", ".join(want)} from {NVTX_TABLE}'):
        start, end = row[idx["start"]], row[idx["end"]]
        if start is None:
            null_start += 1
            continue
        if end is None:
            open_n += 1                     # open range：没有终点，绝不外推成无限长
            continue
        start, end = int(start), int(end)
        if end <= start:
            degen_n += 1                    # mark / 零长区间：不是一个时段，不参与重叠
            continue
        raw = row[idx["text"]] if has_text else None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raw = row[idx["textId"]] if has_textid else None
            if raw is None:
                unnamed_n += 1
            else:
                stringid_n += 1
        else:
            inline_n += 1
        items.append(_Range(name=_resolve(raw, ids).strip() or "<unnamed>",
                            start=start, end=end,
                            key=row[idx[key_col]] if key_col else None))
    out.items = items
    out.facts = {
        "closed": len(items), "open": open_n, "degenerate": degen_n,
        "null_start": null_start, "inline_text": inline_n,
        "stringid_text": stringid_n, "unnamed": unnamed_n,
        "text_source": ("inline `text` + `StringIds`(textId)" if has_text and has_textid
                        else "inline `text`" if has_text
                        else "`StringIds`(textId)" if has_textid
                        else "两列都没有"),
        "columns": ", ".join(sorted(cols)),
    }
    if not items:
        out.problem = (f"{REASON_NO_NVTX}: {NVTX_TABLE} 里没有可用区间"
                       f"（open range {open_n}、退化区间 {degen_n}、start 为 NULL {null_start}）——"
                       f" 只有终点非 NULL 且 end > start 的行才算区间")
    return out


def _load_kernels(con: sqlite3.Connection, tables: set[str], key_col: str | None) -> _Loaded:
    out = _Loaded()
    if KERNEL_TABLE not in tables:
        out.problem = (f"{REASON_NO_KERNELS}: 库里没有 {KERNEL_TABLE} 表 —— "
                       f"采集时没带 `--trace=cuda`")
        return out
    cols = _columns(con, KERNEL_TABLE)
    if "start" not in cols or "end" not in cols:
        out.problem = (f"{REASON_NO_KERNELS}: {KERNEL_TABLE} 缺 start/end 列（实际列："
                       f"{', '.join(sorted(cols)) or '无'}）")
        return out
    want = [c for c in ("start", "end", *_THREAD_KEYS) if c in cols]
    idx = {c: i for i, c in enumerate(want)}
    items: list[_Kernel] = []
    degen_n = 0
    for row in con.execute(f'select {", ".join(want)} from {KERNEL_TABLE}'):
        start, end = row[idx["start"]], row[idx["end"]]
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if end <= start:
            degen_n += 1
            continue
        items.append(_Kernel(start=start, end=end,
                             key=row[idx[key_col]] if key_col else None))
    out.items = items
    out.facts = {"kernels": len(items), "degenerate": degen_n,
                 "columns": ", ".join(sorted(cols))}
    if not items:
        out.problem = (f"{REASON_NO_KERNELS}: {KERNEL_TABLE} 里没有正时长的 kernel"
                       f"（退化/零长 {degen_n} 行）—— 没有 kernel 时间可归因")
    return out


def _thread_key(kernel_cols: set[str], nvtx_cols: set[str]) -> str | None:
    for c in _THREAD_KEYS:
        if c in kernel_cols and c in nvtx_cols:
            return c
    return None


# ---------------------------------------------------------------------------
# 区间算术
# ---------------------------------------------------------------------------
class _Density:
    """kernel 覆盖的**分段密度** + 前缀积分（任何区间上的 kernel 时间 = 两次查询相减）。

    为什么不是「逐个 kernel × 逐个区间」：真实 trace 上两者都是 10^5 量级，双重循环是
    10^10 次比较。密度用一次排序扫描建好（O(n log n)），之后每个区间 O(log n)。
    """

    def __init__(self, kernels: list[_Kernel]):
        events: list[tuple[int, int]] = []
        for k in kernels:
            events.append((k.start, 1))
            events.append((k.end, -1))
        events.sort()
        self.pieces: list[tuple[int, int, int]] = []
        live = 0
        prev: int | None = None
        for t, d in events:
            if live > 0 and prev is not None and t > prev:
                self.pieces.append((prev, t, live))
            live += d
            prev = t
        self.starts = [p[0] for p in self.pieces]
        self.cum = [0]
        for _t0, t1, cnt in self.pieces:
            self.cum.append(self.cum[-1] + (t1 - _t0) * cnt)

    def prefix(self, t: int) -> int:
        """密度在 (-inf, t] 上的积分（单位：kernel·ns）。"""
        i = bisect.bisect_right(self.starts, t) - 1
        if i < 0:
            return 0
        t0, t1, cnt = self.pieces[i]
        if t >= t1:
            return self.cum[i + 1]
        return self.cum[i] + (t - t0) * cnt

    def integrate(self, a: int, b: int) -> int:
        return 0 if b <= a else self.prefix(b) - self.prefix(a)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠/相接的区间（并集口径用）。"""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _mark_depth(ranges: list[_Range]) -> None:
    """按 (start 升序, end 降序) 扫一遍栈，得到嵌套深度（同一条时间线内）。"""
    order = sorted(range(len(ranges)), key=lambda i: (ranges[i].start, -ranges[i].end))
    stack: list[int] = []
    for i in order:
        r = ranges[i]
        while stack and ranges[stack[-1]].end <= r.start:
            stack.pop()
        r.depth = len(stack)
        stack.append(i)


def _overlap_count(starts: list[int], ends: list[int], a: int, b: int) -> int:
    """与 [a, b) 有正重叠的区间个数（starts/ends 已排序，O(log n)）。

    = (start < b 的个数) − (end <= a 的个数)：后者一定包含在前者里，相减即重叠。
    """
    return bisect.bisect_left(starts, b) - bisect.bisect_right(ends, a)


def _slug(name: str) -> str:
    """区间名 → 指标名里可用的一段（与 prescription 同一套：只留字母数字下划线）。"""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in (name or "phase"))
    return out.strip("_") or "phase"


def _ms(x: float) -> float:
    """ms 值 → 1 ns 分辨率（nsys 时间戳是整数 ns；再粗会把短区间舍成 0.0）。"""
    return round(float(x), 6)


def _fmt_ms(x: float) -> str:
    """给人看的 ms：最多 6 位小数、去掉尾零（0.0001 而不是 1e-04）。"""
    return f"{float(x):.6f}".rstrip("0").rstrip(".") or "0"


def _refusal_code(reason: str) -> RefusalCode:
    """reason 前缀 → refusals 闭集里的 code。

    这个模块**不新增**拒绝码（那要改 `refusals.py`，属于单一真源）。所以映射到语义最近的
    现成 code：没有 kernel 活动 = nsys.py 自己的 `no_launch_activity`；NVTX 缺失 = 请求的
    指标在这次报告里不存在（`metric_absent`）。
    """
    return (RefusalCode.NO_LAUNCH_ACTIVITY if reason.startswith(REASON_NO_KERNELS)
            else RefusalCode.METRIC_ABSENT)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def attribute(db_path: str, *, top: int = 8) -> PhaseAttribution:
    """把 kernel 时间按 NVTX 区间重叠归因。

    `top` 是保留的阶段个数（按 kernel_ms 降序）；被截掉的个数写进 `basis`。
    拿不到可归因输入时**返回**带 `reason` 的结果（不是抛异常）——调用方据此输出 n/a。
    只有「文件不存在 / 不是 sqlite / 读失败」才是异常（那是调用方的输入错误）。
    """
    top = max(0, int(top))
    con = _connect_ro(db_path)
    try:
        tables = _tables(con)
        k_cols = _columns(con, KERNEL_TABLE) if KERNEL_TABLE in tables else set()
        n_cols = _columns(con, NVTX_TABLE) if NVTX_TABLE in tables else set()
        key_col = _thread_key(set(k_cols), set(n_cols))
        kern = _load_kernels(con, tables, key_col)
        rng = _load_ranges(con, tables, key_col)
    except sqlite3.Error as e:
        raise PhaseError(f"sqlite read failed: {db_path} ({e})") from e
    finally:
        con.close()

    if kern.problem or rng.problem:
        reason = "；同时 ".join(p for p in (rng.problem, kern.problem) if p)
        return PhaseAttribution(
            phases=[], unattributed_ms=0.0, reason=reason,
            basis="未归因（见 reason）：这一行不带任何数字 —— 0 在这里只表示『没测』，"
                  "不表示『测到 0』。")

    kernels: list[_Kernel] = kern.items
    ranges: list[_Range] = rng.items
    total_ns = sum(k.end - k.start for k in kernels)

    # 分组：有 join key 就按 key 分（只在同一条时间线上做重叠）；没有就整段一起算。
    k_groups: dict[object | None, list[_Kernel]] = {}
    for k in kernels:
        k_groups.setdefault(k.key, []).append(k)
    r_groups: dict[object | None, list[_Range]] = {}
    for r in ranges:
        r_groups.setdefault(r.key, []).append(r)

    phase_ns: dict[int, int] = {}          # 用 id(range) 当下标，避免 _Range 不好哈希
    phase_cnt: dict[int, int] = {}
    attributed_ns = 0
    for key, rs in r_groups.items():
        ks = k_groups.get(key, [])
        _mark_depth(rs)
        if not ks:
            continue                       # 这条时间线上没有 kernel：区间全空，计 0
        dens = _Density(ks)
        starts = sorted(k.start for k in ks)
        ends = sorted(k.end for k in ks)
        # 并集口径：嵌套/重叠的区间只算一次 —— 这才是「落在任一区间内的时间」
        for s, e in _merge([(r.start, r.end) for r in rs]):
            attributed_ns += dens.integrate(s, e)
        for r in rs:
            phase_ns[id(r)] = dens.integrate(r.start, r.end)
            phase_cnt[id(r)] = _overlap_count(starts, ends, r.start, r.end)

    unattributed_ns = total_ns - attributed_ns
    phases_all: list[Phase] = []
    empty_ranges = 0
    for r in ranges:
        ns = phase_ns.get(id(r), 0)
        if ns <= 0:
            empty_ranges += 1              # 没覆盖任何 kernel 的区间不进表（basis 里报数）
            continue
        phases_all.append(Phase(
            name=r.name, depth=r.depth,
            total_ms=(r.end - r.start) / NS_PER_MS,
            kernel_ms=ns / NS_PER_MS,
            share_pct=(ns / total_ns * 100.0) if total_ns else 0.0,
            kernels=int(phase_cnt.get(id(r), 0)),
        ))
    phases_all.sort(key=lambda p: (-p.kernel_ms, p.name, p.depth))
    kept = phases_all[:top]
    omitted = len(phases_all) - len(kept)

    basis = _basis(rng.facts, kern.facts, key_col, total_ns, top=top, omitted=omitted,
                   empty_ranges=empty_ranges, phase_count=len(phases_all))
    return PhaseAttribution(
        phases=kept,
        unattributed_ms=unattributed_ns / NS_PER_MS,
        basis=basis,
        attributed_ms=attributed_ns / NS_PER_MS,
        omitted=omitted,
    )


def _basis(rfacts: dict, kfacts: dict, key_col: str | None, total_ns: int, *,
           top: int, omitted: int, empty_ranges: int, phase_count: int) -> str:
    """口径说明。它不是散文装饰：份额能不能 >100%、unattributed 怎么算，都在这里定死。"""
    if key_col:
        group = f"按两张表都有的 `{key_col}` 列分线程归因（只在同一条时间线上做重叠）"
    else:
        group = ("**整段采集的时间重叠**：NVTX_EVENTS 有 globalTid、CUPTI kernel 表有 "
                 "deviceId/streamId，两张表在导出库里没有可验证的映射，所以没有做线程/设备"
                 "过滤。多卡/多线程 trace 上，别的时间线的区间也会参与归因 —— 这是**声明的"
                 "退化**，不是默认口径")
    return "；".join([
        "NVTX 区间重叠法：kernel 的 [start, end] 与区间在同一采集时钟（ns）上做区间重叠，"
        "只计入**重叠时长** —— 跨两个区间的 kernel 会被切开",
        f"归因分组：{group}",
        "嵌套区间两边都计入（depth 区分层级），所以各阶段 kernel_ms 之和与 share_pct 之和"
        "都**可以 >100%**：本表不是 kernel 时间的划分（partition），只有 "
        "attributed/unattributed 与它互补",
        "unattributed_ms = 总 kernel 时间 − 落在**任一**区间内的时间（并集口径），"
        "不是『总时间 − 各阶段之和』",
        f"kernel {kfacts.get('kernels', 0)} 个、合计 {_fmt_ms(total_ns / NS_PER_MS)} ms；"
        f"NVTX 可用区间 {rfacts.get('closed', 0)} 个"
        f"（open range {rfacts.get('open', 0)} 个、退化区间 {rfacts.get('degenerate', 0)} 个、"
        f"start 为 NULL {rfacts.get('null_start', 0)} 个未计入），"
        f"其中覆盖到 kernel 的 {phase_count} 个、kernel_ms=0 的 {empty_ranges} 个未列出；"
        f"按 kernel_ms 降序取前 top={top} 个，省略 {omitted} 个",
        f"区间文本来源：{rfacts.get('text_source', '未知')}"
        f"（未命名 {rfacts.get('unnamed', 0)} 个）；NVTX 列：{rfacts.get('columns', '')}",
    ])


__all__ = ["Phase", "PhaseAttribution", "PhaseError", "attribute",
           "REASON_NO_NVTX", "REASON_NO_KERNELS"]
