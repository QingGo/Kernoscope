"""访存轨迹分析：从 NVBit 插桩记录重建 ncu 的访存类读数。

三类输出，出处不同，**必须分开标注**：

1. **精确算术**（`measured`，采样比 <100% 时降为 `estimated`）
   从实测地址直接算，不依赖任何缓存模型：
   * `sectors/request`  ← 对应 ncu `l1tex__average_t_sectors_per_request`
   * 内层合并效率（真正需要的字节 / 实际搬运的 sector 字节）
   * 触达工作集（去重 sector × 32B）

2. **模型量**（`estimated`，参数写在 note 里）
   * L1/L2 命中率：用集合相联 + LRU 模拟。真实替换策略不是 LRU，所以这是
     **乐观上界**；同时给"只过 L2"和"先过 per-SM L1 再过 L2"两种口径，
     因为前者会把 L1 命中误记成 L2 命中（**高估**）。
   * DRAM 字节数：模型算出的 L2 miss × 32B，再按采样比与 block 覆盖率放大。

3. **拒绝给数**：轨迹被截断（`truncated=1`）时不给 DRAM 字节 —— 截断的轨迹算出来的
   总量是**下界**，而"下界"最容易被误读成"实测值"。这条纪律与 ncu 计数器不可得时
   只报 `n/a` 是同一个原则。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from ..backends.memtrace import SECTOR_BYTES, MemTrace

L1_BYTES_DEFAULT = 128 * 1024
# 建模粒度 = 32B sector，而不是 128B cache line。
# 原因（真机+单测双重验证）：ncu 报的是 **sector 级**命中率；若按 128B 行建模，
# 同一条行内的第 2~4 个 sector 会被算成"命中"，纯流式负载的命中率会被虚高到 ~7.5%
# （实测：200 条互不重叠的 4-sector 访问 → 上界口径算出 75% 命中，错得离谱）。
LINE_BYTES_DEFAULT = 32
L1_WAYS_DEFAULT = 4
L2_WAYS_DEFAULT = 16
MAX_ACCESSES_DEFAULT = 4_000_000


class SetAssocLRU:
    """集合相联 + LRU 的缓存模型（理想化）。

    默认按 **32B sector** 粒度建模，与 ncu 的 sector 级命中率口径对齐；
    `--line-bytes 128` 可切换成行粒度做敏感性实验（会让命中率偏高）。
    真实硬件的替换策略不是 LRU，所以这个模型给的是**乐观上界**。
    """

    def __init__(self, capacity_bytes: int, line_bytes: int = LINE_BYTES_DEFAULT,
                 ways: int = 16):
        self.line_bytes = line_bytes
        self.ways = max(1, ways)
        self.n_sets = max(1, capacity_bytes // (line_bytes * self.ways))
        self._sets: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(self.n_sets)]
        self.hits = 0
        self.misses = 0

    def access(self, addr: int) -> bool:
        line = addr // self.line_bytes
        s = line % self.n_sets
        tag = line // self.n_sets
        d = self._sets[s]
        if tag in d:
            d.move_to_end(tag)
            self.hits += 1
            return True
        self.misses += 1
        d[tag] = None
        if len(d) > self.ways:
            d.popitem(last=False)
        return False

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate_pct(self) -> float:
        return 100.0 * self.hits / self.total if self.total else 0.0


@dataclass
class MemTraceStats:
    kernel: str = ""
    records: int = 0
    accesses: int = 0                 # 去重后的 (记录, sector) 访问次数
    analyzed_accesses: int = 0
    sectors_per_request: float = 0.0
    coalescing_pct: float = 0.0
    working_set_mb: float = 0.0
    sampling_ratio: float = 1.0
    truncated: bool = False
    l2_bytes: int = 0
    l2_hit_rate_l2only_pct: float = 0.0
    l2_hit_rate_with_l1_pct: float = 0.0
    dram_bytes_per_launch: float | None = None
    dram_bytes_lower: float | None = None       # 替换策略下界（只算强制缺失）
    dram_bytes_upper: float | None = None       # 上界（完全不复用）
    dram_bytes_unavailable_reason: str = ""
    l1_bytes: int = L1_BYTES_DEFAULT
    line_bytes: int = LINE_BYTES_DEFAULT
    l1_ways: int = L1_WAYS_DEFAULT
    l2_ways: int = L2_WAYS_DEFAULT
    smid_approx: bool = False
    op_counts: dict[str, int] = field(default_factory=dict)
    # R-73：全局口径只用 kind == "global" 的记录；其余按类型计数报出来（可复核）
    records_global: int = 0
    trace_complete: str = "unknown"      # yes / no / unknown（R-76）
    trace_sha256: str = ""               # 轨迹指纹（R-75：对拍时比对是不是同一份轨迹）
    kind_counts: dict[str, int] = field(default_factory=dict)
    coalescing_clamped: bool = False


def analyze_memtrace(trace: MemTrace, l2_bytes: int | None = None,
                     l1_bytes: int = L1_BYTES_DEFAULT,
                     line_bytes: int = LINE_BYTES_DEFAULT,
                     l1_ways: int = L1_WAYS_DEFAULT,
                     l2_ways: int = L2_WAYS_DEFAULT,
                     max_accesses: int = MAX_ACCESSES_DEFAULT) -> MemTraceStats:
    st = MemTraceStats(
        kernel=trace.kernel, records=len(trace.records),
        smid_approx=trace.header.get("smid_approx") == "1",
        sampling_ratio=trace.sampling_ratio, truncated=trace.truncated,
        trace_complete=trace.complete, trace_sha256=trace.header.get("trace_sha256", ""),
        l2_bytes=l2_bytes or trace.l2_bytes,
        l1_bytes=l1_bytes, line_bytes=line_bytes, l1_ways=l1_ways, l2_ways=l2_ways,
    )

    # ---- 1) 精确算术（**只用全局访存**）----
    # 共享/本地访存与 L2↔DRAM 无关：混进来会让 sectors/request 偏大、合并效率算出 >100%
    # 这种物理上不可能的数（R-73；Triton 的 tl.dot 会灌进海量 LDS/STS）。
    glob = [r for r in trace.records if r.kind == "global"]
    st.records_global = len(glob)
    for r in trace.records:
        st.kind_counts[r.kind] = st.kind_counts.get(r.kind, 0) + 1
        st.op_counts[r.op] = st.op_counts.get(r.op, 0) + 1
    n = len(glob)
    if n:
        st.sectors_per_request = sum(len(r.sectors) for r in glob) / n
        useful = sum(r.useful_bytes for r in glob)
        moved = sum(r.sector_bytes for r in glob)
        raw = 100.0 * useful / moved if moved else 0.0
        # 夹紧 + 留痕：有用字节不可能多于搬进来的 sector 字节。夹紧是**兜底**，
        # 正常路径上是 kind 分流把不可能的输入挡在门外。
        st.coalescing_pct = min(100.0, raw)
        st.coalescing_clamped = raw > 100.0 + 1e-9
        unique = {a for r in glob for a in r.sectors}
        st.working_set_mb = len(unique) * SECTOR_BYTES / 1e6
        st.accesses = sum(len(r.sectors) for r in glob)

    # ---- 2) 缓存模型 ----
    l2_only = SetAssocLRU(st.l2_bytes, line_bytes, l2_ways) if st.l2_bytes else None
    l1_misses: list[tuple[int, int]] = []      # (addr, smid) 进入 L2 的访问
    l1_cache: dict[int, SetAssocLRU] = {}
    budget = max_accesses
    analyzed = 0
    for r in glob:                     # R-73：L2 模型同样只看全局访存
        if analyzed >= budget:
            break
        for addr in r.sectors:
            if analyzed >= budget:
                break
            if l2_only is not None:
                l2_only.access(addr)
            if l1_bytes and st.l2_bytes:
                c = l1_cache.get(r.smid)
                if c is None:
                    c = l1_cache[r.smid] = SetAssocLRU(l1_bytes, line_bytes, l1_ways)
                if not c.access(addr):
                    l1_misses.append((addr, r.smid))
            analyzed += 1
    st.analyzed_accesses = analyzed
    if l2_only is not None:
        st.l2_hit_rate_l2only_pct = l2_only.hit_rate_pct

    l2_filtered = SetAssocLRU(st.l2_bytes, line_bytes, l2_ways) if st.l2_bytes else None
    if l2_filtered is not None:
        for addr, _smid in l1_misses:
            l2_filtered.access(addr)
        st.l2_hit_rate_with_l1_pct = l2_filtered.hit_rate_pct

    # ---- 3) DRAM 字节（模型 + 采样放大；截断时拒绝给数）----
    if st.truncated:
        st.dram_bytes_unavailable_reason = (
            "轨迹被截断（truncated=1）：算出来的总量只是**下界**，不是实测值 —— 按本项目的"
            "纪律不给出数字。请提高 --max-records 或减小采样范围后重跑")
    elif l2_filtered is None or l2_filtered.total == 0:
        st.dram_bytes_unavailable_reason = (
            "轨迹头部缺少 l2_bytes（缓存模型无法定容）；请用 --ks-l2-bytes 指定")
    elif analyzed < st.accesses:
        need = int(st.accesses)
        st.dram_bytes_unavailable_reason = (
            f"只分析了 {analyzed}/{need} 次访问（达到 max_accesses 上限 {budget}）："
            f"放大后的总量会系统性偏小，不给出数字。**把 --ks-max-accesses 提到 ≥ {need}** "
            f"即可（或先用 --kernel 只留被测 kernel / 减小采集范围，把访问数压到 {budget} 以内）")
    else:
        miss_sectors = l2_filtered.misses
        # 放大：采样 block -> 全部 block（同一 launch 内）
        traced, grid = trace.blocks_traced, trace.grid_blocks
        scale = 1.0
        if traced and grid and grid > traced:
            scale = grid / traced
        total = miss_sectors * SECTOR_BYTES * scale / max(1, trace.launches_traced)
        st.dram_bytes_per_launch = total
        # R-77：同一条轨迹的**替换策略上下界**（地址本身是算术，只有"谁命中 L2"是模型）：
        #   下界 = 只算强制缺失（整段轨迹里去重后的 sector 各搬一次）
        #   上界 = 完全不命中（每次访问都算一次 miss）
        per = SECTOR_BYTES * scale / max(1, trace.launches_traced)
        unique_sectors = {a for r in glob for a in r.sectors}
        st.dram_bytes_lower = len(unique_sectors) * per
        st.dram_bytes_upper = sum(len(r.sectors) for r in glob) * per
    return st


def to_metrics(st: MemTraceStats) -> list[tuple[str, float | None, str, str, str]]:
    """(name, value, unit, source, note, provenance) 供 section 组装。

    provenance 用字符串返回，由 sections 侧转成枚举，避免这里 import provenance 造成环。
    """
    # 采样比 <100% 时连"精确算术"也要降为 estimated：样本代表性是假设
    arith = "measured" if st.sampling_ratio >= 0.999 else "estimated"
    sample_note = (f"全量（{st.records} 条 warp 指令记录）" if arith == "measured" else
                   f"按 block 采样 {st.sampling_ratio * 100:.1f}%（{st.records} 条记录），"
                   f"按采样比放大")
    # R-73：全局口径只用 `kind == global` 的记录，其余按类型计数报出来 ——
    # 读者要能解释"为什么我的轨迹有 1.28M 条记录、DRAM 字节却按 8K 条算"。
    excluded = {k: v for k, v in sorted(st.kind_counts.items()) if k != "global"}
    comp_note = {"yes": "轨迹带结束标记（完整）",
                 "no": "轨迹被截断（truncated）",
                 "unknown": "轨迹**没有结束标记** ⇒ 无法确认没被 ulimit/磁盘写满截断"
                            "（自己采集请用 tools/probes/nvbit_memtrace/collect.sh，它会写标记）"}[st.trace_complete]
    identity = f"轨迹指纹 {st.trace_sha256}（{comp_note}）" if st.trace_sha256 else comp_note
    scope = (f"{identity}；全局口径只用 LDG/STG（{st.records_global}/{st.records} 条）；"
             + ("其余为 " + "、".join(f"{k} {v}" for k, v in excluded.items())
                if excluded else "没有被排除的记录")
             + "。共享/本地访存与 L2↔DRAM 无关，混进来会把合并效率算成 >100%")
    out: list[tuple[str, float | None, str, str, str]] = [
        ("meas__sectors_per_request", round(st.sectors_per_request, 3), "",
         "NVBit 地址插桩后直接计数（对应 ncu l1tex__average_t_sectors_per_request）",
         f"{sample_note}；每个 sector 32B"),
        ("meas__coalescing_efficiency_pct", round(st.coalescing_pct, 1), "%",
         "每 warp 指令：真正所需字节 / 搬运的 sector 字节",
         "100% 表示无浪费；<50% 通常是跨步/未对齐访问"),
        ("meas__trace_working_set_mb", round(st.working_set_mb, 3), "MB",
         "去重后的 32B sector 数 × 32B",
         f"{sample_note}；用于判断工作集是否超过 L2"),
        ("meas__trace_records", float(st.records), "",
         "轨迹里的 warp 级访存指令执行记录数（**全部**访存类型）", sample_note),
        ("meas__trace_records_global", float(st.records_global), "",
         "其中属于全局口径（LDG/STG）的记录数 —— DRAM/L2 统计的分母",
         scope + ("；⚠ 本次合并效率被夹到 100%（原始算式 >100%，说明输入仍不干净）"
                  if st.coalescing_clamped else "")),
    ]
    smid_note = ("；SM id 由 CTA 近似分配（官方 mem_trace 不输出 SM id），"
                 "故『先过 L1』口径仅供参考" if st.smid_approx else "")
    if st.l2_bytes:
        out.append((
            "est__l2_hit_rate_l2only_pct", round(st.l2_hit_rate_l2only_pct, 1), "%",
            f"集合相联 LRU 模型（L2={st.l2_bytes / 2**20:.0f} MiB, line={st.line_bytes}B, "
            f"{st.l2_ways}way）",
            "**上界口径**：所有访问都当到达 L2，会把 L1 命中误记为 L2 命中（高估）"))
        out.append((
            "est__l2_hit_rate_with_l1_pct", round(st.l2_hit_rate_with_l1_pct, 1), "%",
            f"先过 per-SM L1（{st.l1_bytes // 1024} KiB, {st.l1_ways}way, line={st.line_bytes}B）"
            f"再过 L2（{st.l2_bytes / 2**20:.0f} MiB, {st.l2_ways}way）",
            "比上界口径更接近真实；两种口径之差就是 L1 吸收了多少" + smid_note))
    if st.dram_bytes_per_launch is not None:
        out.append((
            "est__dram_bytes_per_launch_trace", round(st.dram_bytes_per_launch / 1e6, 3), "MB",
            "模型算出的 L2 miss sector × 32B × (grid/blocks_traced) / launches",
            # R-77：出处拆开说 —— 地址是算术（逐 sector 可复核），只有"谁命中 L2"是模型；
            # 模型值必须落在下面那两个界之间。
            "**地址部分是 measured 级算术**（sector 由地址现算，可逐条复核）；"
            "只有「哪些访问命中 L2」是模型（LRU + 相联度），所以整体是 estimated。"
            "与 NVML 代理是两条独立路径：一致说明结论稳，不一致要查采样代表性"))
        if st.dram_bytes_lower is not None and st.dram_bytes_upper is not None:
            out.append((
                "est__dram_bytes_per_launch_trace_bounds", round(st.dram_bytes_lower / 1e6, 3),
                "MB",
                f"替换策略下界（强制缺失 {st.dram_bytes_lower / 1e6:.3f} MB）"
                f"~ 上界（完全不复用 {st.dram_bytes_upper / 1e6:.3f} MB）",
                "同一份轨迹的**模型影响范围**：真实值一定落在两者之间；"
                "界很宽说明结论对缓存模型敏感，要换口径或加约束（例如把工作集压到 L2 以内）"))
    return out
