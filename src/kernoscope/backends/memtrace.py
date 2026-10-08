"""`ks-memtrace v1` 轨迹解析 —— NVBit 访存插桩层的数据契约。

轨迹由 `tools/probes/nvbit_memtrace/memtrace.cu`（用户自行用 NVBit 编译）产出。
NVBit 本体是 NVIDIA 专有许可、**不得随本项目分发**，所以本项目只带工具源码与解析器。

格式（文本，便于 grep/diff，也让分析器能在没有 GPU 的机器上跑）：

    # ks-memtrace v1
    # kernel addk
    # arch sm_89
    # grid 3072
    # block 256
    # blocks_traced 384
    # sampling_ratio 0.125
    # l2_bytes 75497472
    # sm_count 128
    # truncated 0
    insn,op,lanes,size,sectors,smid,addrs
    0x120,LDG.E,32,4,4,17,1400000000;1400000020;1400000040;1400000060

* 一行 = 一条 warp 级访存指令的**一次执行**（不是每 lane 一行，否则轨迹会大 32 倍）
* `addrs` 是这条指令触达的**去重后的 32B sector 地址**（分号分隔）
* `insn` 是主机侧指派的静态指令编号，对应头部 `# insn <id> <函数> <opcode> <偏移>` legend
* `sectors` 允许写 0，表示"由分析器按 addrs 计算"（工具侧不做去重，省设备时间）
* `smid` 用于 per-SM 的 L1 过滤建模（L1 是每 SM 私有的，缺它就只能给 L2 上界口径）
* `sectors` 由工具给出，解析器会用 `addrs` 重算并校验 —— 工具写错时能立刻发现，
  而不是把错误的 sectors/request 一路带到报告里
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FORMAT_TAG = "ks-memtrace v1"
OFFICIAL_TAG = "MEMTRACE:"
SECTOR_BYTES = 32

# opcode 后缀 -> 每 lane 访问字节数（与工具侧 size_from_opcode 保持同一套规则）
_SIZE_RULES = ((".128", 16), (".64", 8), (".32", 4), (".16", 2), (".8", 1))


def size_from_opcode(op: str) -> int:
    for suffix, size in _SIZE_RULES:
        if suffix in op:
            return size
    return 4


class MemTraceError(ValueError):
    """轨迹文件格式/内容问题。一律带可操作信息，不静默降级。"""


@dataclass
class MemRecord:
    """一条 warp 级访存指令执行记录。"""

    pc: int
    op: str
    lanes: int
    size: int            # 每 lane 字节数
    smid: int
    sectors: list[int]   # 去重、32B 对齐的地址

    @property
    def useful_bytes(self) -> int:
        return self.lanes * self.size

    @property
    def sector_bytes(self) -> int:
        return len(self.sectors) * SECTOR_BYTES

    @property
    def coalescing_pct(self) -> float:
        """内层合并效率：真正需要的字节 / 实际搬运的 sector 字节。"""
        if not self.sector_bytes:
            return 0.0
        return min(100.0, 100.0 * self.useful_bytes / self.sector_bytes)


@dataclass
class MemTrace:
    header: dict[str, str] = field(default_factory=dict)
    records: list[MemRecord] = field(default_factory=list)

    # ---- 头部便捷访问 ----
    def _get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.header.get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.header.get(key, default))
        except (TypeError, ValueError):
            return default

    @property
    def kernel(self) -> str:
        return self.header.get("kernel", "<unknown>")

    @property
    def arch(self) -> str:
        return self.header.get("arch", "sm_89")

    @property
    def grid_blocks(self) -> int:
        return self._get_int("grid")

    @property
    def block_threads(self) -> int:
        return self._get_int("block")

    @property
    def blocks_traced(self) -> int:
        return self._get_int("blocks_traced")

    @property
    def launches_traced(self) -> int:
        return max(1, self._get_int("launches_traced", 1))

    @property
    def sampling_ratio(self) -> float:
        """被插桩的 block 占全部 block 的比例（1.0 表示全量）。"""
        r = self._get_float("sampling_ratio", 0.0)
        if r:
            return min(1.0, max(1e-9, r))
        if self.grid_blocks and self.blocks_traced:
            return min(1.0, self.blocks_traced / self.grid_blocks)
        return 1.0

    @property
    def l2_bytes(self) -> int:
        return self._get_int("l2_bytes")

    @property
    def sm_count(self) -> int:
        return self._get_int("sm_count")

    @property
    def truncated(self) -> bool:
        return self._get_int("truncated") == 1


def _parse_header_line(line: str) -> tuple[str, str] | None:
    body = line[1:].strip()
    if body == FORMAT_TAG:
        return None
    if " " in body:
        k, v = body.split(" ", 1)
        return k.strip(), v.strip()
    return None


def load_memtrace(path: str) -> MemTrace:
    """解析轨迹文件。任何格式问题都抛 MemTraceError（附行号与期望格式）。"""
    header: dict[str, str] = {}
    records: list[MemRecord] = []
    saw_tag = False
    bad_sector = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line[1:].strip() == FORMAT_TAG:
                    saw_tag = True
                    continue
                kv = _parse_header_line(line)
                if kv:
                    header[kv[0]] = kv[1]
                continue
            # 列头行（insn,/pc,）跳过：首列不是数字就是表头
            _head = line.split(",", 1)[0].strip().lower()
            if not _head.lstrip("0x").isdigit():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 7:
                raise MemTraceError(
                    f"{path}:{lineno}: 期望 7 列 'pc,op,lanes,size,sectors,smid,addrs'，"
                    f"得到 {len(parts)} 列")
            try:
                pc = int(parts[0], 16) if parts[0].lower().startswith("0x") else int(parts[0])
                lanes = int(parts[2])
                size = int(parts[3])
                claimed = int(parts[4])
                smid = int(parts[5])
                addrs = [int(a, 16) if a.lower().startswith("0x") else int(a)
                         for a in parts[6].split(";") if a.strip()]
            except ValueError as e:
                raise MemTraceError(f"{path}:{lineno}: 数值解析失败: {line[:80]!r}") from e
            sectors = sorted({a & ~(SECTOR_BYTES - 1) for a in addrs})
            if claimed and claimed != len(sectors):
                bad_sector += 1
                if bad_sector <= 3:
                    raise MemTraceError(
                        f"{path}:{lineno}: 工具报 sectors={claimed}，但 addrs 去重后是 "
                        f"{len(sectors)} —— 工具与解析器的口径不一致，不能继续（请检查插桩代码）")
            records.append(MemRecord(pc=pc, op=parts[1], lanes=lanes, size=size,
                                     smid=smid, sectors=sectors))

    if not saw_tag:
        raise MemTraceError(f"{path}: 缺少首行 '# {FORMAT_TAG}'（轨迹格式标识）")
    if not records:
        raise MemTraceError(
            f"{path}: 没有任何访存记录 —— 要么 kernel 没有访存，要么插桩没生效。"
            f"常见原因：① 未设 NO_EAGER_LOAD=0（NVBit 1.8 默认懒加载会错过插桩时机）；"
            f"② 工具 architecture 与 GPU 不匹配（用 ARCH=native 重建）")
    if bad_sector:
        header["sector_mismatch_rows"] = str(bad_sector)
    return MemTrace(header=header, records=records)


# ---------------------------------------------------------------------------
# 官方 NVBit mem_trace 输出的转换器
# ---------------------------------------------------------------------------
def _official_addresses(field: str) -> list[int]:
    out: list[int] = []
    for tok in field.replace(",", " ").split():
        tok = tok.strip()
        if tok.startswith("0x"):
            try:
                out.append(int(tok, 16))
            except ValueError:
                pass
    return out


def load_official_memtrace(path: str, sm_count: int = 0, l2_bytes: int = 0,
                           kernel_filter: str = "") -> MemTrace:
    """把 NVBit 官方 `mem_trace` 工具的 stdout 转成 ks-memtrace v1。

    为什么需要它：官方工具产出的是"一行 = 一条 warp 访存指令 + 全部 lane 地址"，
    粒度恰好够用，但格式是它自己的打印格式。示例：

        MEMTRACE: CTX 0x.. - LAUNCH - Kernel pc 0x.. - Kernel name k(...) \
                  - grid launch id 0 - grid size 1,1,1 - block size 32,1,1 - ...
        MEMTRACE: CTX 0x.. - grid_launch_id 0 - CTA 0,0,0 - warp 1 - LDG.E - 0x.. 0x.. ...

    官方工具会打印**所有** kernel 的记录，所以按 LAUNCH 行把记录归属到各自的 kernel，
    再用 `kernel_filter`（子串）只保留其中一个 —— 否则多 kernel 应用的统计会被混在一起。

    注意两个**信息缺失**，都用显式假设补上（并写进头部，便于事后追溯）：
      * 没有 SM id → 用 `cta_id % sm_count` 近似（同一 CTA 的 warp 通常在同一个 SM 上）；
        因此"先过 per-SM L1"那种口径只是近似，`# smid_approx 1` 会标注出来。
      * 没有指令 PC → 用 opcode 建 id（legend 里写清楚）。
    """
    header: dict[str, str] = {}
    records: list[MemRecord] = []
    opcode_ids: dict[str, int] = {}
    ctas: set[int] = set()
    launches = 0
    kernel = ""
    grid = block = 0
    cur_kernel = ""
    cur_grid = cur_block = 0
    keep_current = True
    first_kernel = ""
    kernels_seen: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line.startswith(OFFICIAL_TAG):
                continue
            body = line[len(OFFICIAL_TAG):].strip()
            parts = [p.strip() for p in body.split(" - ")]
            if len(parts) < 2:
                continue

            if "LAUNCH" in parts[1]:
                cur_kernel = ""
                for part in parts:
                    if part.startswith("Kernel name "):
                        cur_kernel = part[len("Kernel name "):].strip()
                    elif part.startswith("grid size "):
                        dims = [int(x) for x in part[len("grid size "):].split(",") if x.strip().isdigit()]
                        cur_grid = 1
                        for d in dims:
                            cur_grid *= d
                    elif part.startswith("block size "):
                        dims = [int(x) for x in part[len("block size "):].split(",") if x.strip().isdigit()]
                        cur_block = 1
                        for d in dims:
                            cur_block *= d
                if cur_kernel and cur_kernel not in kernels_seen:
                    kernels_seen.append(cur_kernel)
                if not first_kernel:
                    first_kernel = cur_kernel
                # 归属判断：给了 --kernel 就按子串选；否则只取**第一个** kernel，
                # 绝不把多个 kernel 的记录混在一起算（那样统计没有意义）
                if kernel_filter:
                    keep_current = bool(cur_kernel) and kernel_filter in cur_kernel
                else:
                    keep_current = bool(cur_kernel) and cur_kernel == first_kernel
                if keep_current:
                    launches += 1
                    kernel, grid, block = cur_kernel, cur_grid, cur_block
                continue
            if not keep_current:
                continue

            # 数据行：... - CTA x,y,z - warp W - OPCODE - addr addr ...
            cta = 0
            op = ""
            addrs: list[int] = []
            for part in parts:
                if part.startswith("CTA "):
                    coords = [int(x) for x in part[4:].split(",") if x.strip().lstrip("-").isdigit()]
                    cta = 0
                    for i, c in enumerate(coords[:3]):
                        cta += c * (65536 ** i)
                elif part.startswith("warp "):
                    continue
                elif part.startswith("0x") and " " in part:
                    addrs = _official_addresses(part)
                elif part and not part.startswith(("CTX", "grid_launch_id")):
                    op = part
            if not op or not addrs:
                continue
            ctas.add(cta)
            if op not in opcode_ids:
                opcode_ids[op] = len(opcode_ids)
            sectors = sorted({a & ~(SECTOR_BYTES - 1) for a in addrs})
            smid = (cta % sm_count) if sm_count else 0
            records.append(MemRecord(pc=opcode_ids[op], op=op, lanes=len(addrs),
                                     size=size_from_opcode(op), smid=smid, sectors=sectors))

    if not records:
        hint = (f"（本次输出里的 kernel: {' | '.join(kernels_seen[:5]) or '无 LAUNCH 行'}）"
                if kernels_seen else "")
        raise MemTraceError(
            f"{path}: 没有解析到任何 MEMTRACE 数据行{hint}。若用官方 mem_trace 产出："
            f"① 本环境需要 NO_EAGER_LOAD=0；② 若过滤了 kernel，确认名字匹配")
    header["source"] = "official-nvbit-mem-trace"
    header["kernel"] = kernel or "<unknown>"
    header["grid"] = str(grid)
    header["block"] = str(block)
    header["blocks_traced"] = str(len(ctas))
    header["sampling_ratio"] = "1.0"
    header["launches_traced"] = str(max(1, launches))
    header["sm_count"] = str(sm_count)
    header["l2_bytes"] = str(l2_bytes)
    header["smid_approx"] = "1"
    header["kernels_seen"] = " | ".join(kernels_seen[:8])
    header["opcode_legend"] = ",".join(f"{i}={o}" for o, i in opcode_ids.items())
    return MemTrace(header=header, records=records)


def load_memtrace_auto(path: str, sm_count: int = 0, l2_bytes: int = 0,
                       kernel_filter: str = "") -> MemTrace:
    """自动判别格式：官方 mem_trace 输出 vs ks-memtrace v1。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            if line.lstrip().startswith(OFFICIAL_TAG):
                return load_official_memtrace(path, sm_count=sm_count, l2_bytes=l2_bytes,
                                              kernel_filter=kernel_filter)
            if line.strip().startswith("#") and FORMAT_TAG in line:
                break
    return load_memtrace(path)
