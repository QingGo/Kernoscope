"""Occupancy：理论值（解析算术，exact）与达到值（区间重叠，measured）。

理论 occupancy 与 ncu 的 Occupancy 面板同源（静态属性 + 架构上限），我们在
在受限的多租户容器实例上验证过：结果与 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 一致。

达到 occupancy 是 ncu 用硬件计数器算的量；这里用**区间重叠法**从
`(smid, 起止时间)` 重建 —— 真机测量（RTX 4090，128 线程/块，产物见 bench/artifacts）：

    无 smem        : CUDA API 理论 12 blocks/SM = 100% → 区间重叠实测 11.27 = 93.9%
    32 KB 静态 smem: CUDA API 理论  3 blocks/SM =  25% → 区间重叠实测  2.99 =  24.9%

差距来自 ramp-up 与 tail（block 陆续开工/收尾），所以报告里同时给出实测并发块数的
均值与峰值，而不是只给一个百分比。它需要 kernel 内有我们的探针
（见 tools/probes/occupancy_probe.cu），因此出处是 `measured` 而非 `exact`。
"""

from __future__ import annotations

from dataclasses import dataclass

# 架构上限（来源：CUDA C++ Programming Guide 的 Compute Capabilities 表）
ARCH: dict[str, dict[str, int]] = {
    "sm_80": dict(blocks=32, threads=2048, warps=64, regs=65536, smem=163840),
    "sm_86": dict(blocks=16, threads=1536, warps=48, regs=65536, smem=102400),
    "sm_89": dict(blocks=24, threads=1536, warps=48, regs=65536, smem=102400),
    "sm_90": dict(blocks=32, threads=2048, warps=64, regs=65536, smem=233472),
    "sm_100": dict(blocks=32, threads=2048, warps=64, regs=65536, smem=233472),
    "sm_120": dict(blocks=24, threads=1536, warps=48, regs=65536, smem=102400),
}

REG_GRANULARITY = 256   # 每 warp 的寄存器分配粒度（8 regs/thread × 32 lanes）
SMEM_OVERHEAD = 1024    # 每 block 的驱动保留量（经验值）


def arch_for(compute_capability: str) -> str:
    """'8.9' -> 'sm_89'；未知则回退 sm_89 并允许调用方覆盖。"""
    try:
        major, minor = compute_capability.split(".")[:2]
        key = f"sm_{major}{minor}"
    except Exception:
        return "sm_89"
    return key if key in ARCH else "sm_89"


@dataclass
class OccupancyResult:
    arch: str
    threads_per_block: int
    registers_per_thread: int
    smem_static: int
    smem_dynamic: int
    blocks_per_sm: int
    warps_per_block: int
    warps_per_sm: int
    occupancy: float            # 0..1
    limiter: str                # 并列的限制因素，用 '+' 连接
    limits: dict[str, int]

    @property
    def occupancy_pct(self) -> float:
        return self.occupancy * 100.0


def theoretical_occupancy(
    registers_per_thread: int,
    smem_static: int,
    smem_dynamic: int,
    threads_per_block: int,
    arch: str,
) -> OccupancyResult:
    """理论上界（与 ncu Occupancy 面板同源算法）。"""
    a = ARCH.get(arch) or ARCH["sm_89"]
    threads = max(1, min(threads_per_block, a["threads"]))
    warps_per_block = -(-threads // 32)  # ceil

    regs = min(max(registers_per_thread, 1), 255)
    regs_per_warp = -(-(regs * 32) // REG_GRANULARITY) * REG_GRANULARITY
    smem_total = smem_static + smem_dynamic + SMEM_OVERHEAD

    limits = {
        "registers": a["regs"] // regs_per_warp // warps_per_block,
        "threads": a["threads"] // threads,
        "warps": a["warps"] // warps_per_block,
        "shared_mem": (a["smem"] // smem_total) if smem_total else 999,
        "block_limit": a["blocks"],
    }
    blocks = max(0, min(limits.values()))
    warps = blocks * warps_per_block
    mn = min(limits.values())
    limiter = "+".join(sorted(k for k, v in limits.items() if v == mn))
    return OccupancyResult(
        arch=arch,
        threads_per_block=threads,
        registers_per_thread=regs,
        smem_static=smem_static,
        smem_dynamic=smem_dynamic,
        blocks_per_sm=blocks,
        warps_per_block=warps_per_block,
        warps_per_sm=warps,
        occupancy=warps / a["warps"] if a["warps"] else 0.0,
        limiter=limiter,
        limits=limits,
    )


def achieved_occupancy_from_intervals(
    records: list[tuple[int, int, int]],
    n_sm: int,
    threads_per_block: int,
    arch: str,
) -> dict[str, float]:
    """区间重叠法重建达到 occupancy。

    records: [(smid, t_start, t_end), ...]，时间单位任意但需一致（推荐 globaltimer ns）。
    返回 {'avg_concurrent_blocks': .., 'peak_concurrent_blocks': .., 'occupancy': 0..1, 'occupancy_pct': ..}

    注意：只有同一 SM 内的区间可直接比较（%clock64 跨 SM 原点不同；这是我们在真机上验证过的坑）。
    若时间戳来自 %clock64 而非 %globaltimer，请先按 SM 分组各自归一化。
    """
    a = ARCH.get(arch) or ARCH["sm_89"]
    events: list[list[tuple[int, int]]] = [[] for _ in range(max(n_sm, 1))]
    for smid, t0, t1 in records:
        if 0 <= smid < len(events) and t1 > t0:
            events[smid].append((t0, +1))
            events[smid].append((t1, -1))

    total_avg = 0.0
    used = 0
    peak = 0
    for ev in events:
        if len(ev) < 2:
            continue
        ev.sort()
        t_first, t_last = ev[0][0], ev[-1][0]
        if t_last <= t_first:
            continue
        area = 0.0
        cur = 0
        prev = t_first
        for t, delta in ev:
            area += cur * (t - prev)
            prev = t
            cur += delta
            peak = max(peak, cur)
        total_avg += area / (t_last - t_first)
        used += 1

    avg_conc = total_avg / used if used else 0.0
    warps = -(-threads_per_block // 32)
    occ = avg_conc * warps / a["warps"] if a["warps"] else 0.0
    return {
        "avg_concurrent_blocks": avg_conc,
        "peak_concurrent_blocks": float(peak),
        "occupancy": occ,
        "occupancy_pct": occ * 100.0,
        "samples": float(used),
    }


# ---------------------------------------------------------------------------
# 探针记录（tools/probes/occupancy_probe.cu 的输出格式 ks-occupancy-records v1）
# ---------------------------------------------------------------------------
PROBE_FORMAT = "ks-occupancy-records v1"


@dataclass
class ProbeRecords:
    """探针采集结果：头部元信息 + 每 block 的 (smid, t0, t1)。"""

    header: dict[str, str]
    records: list[tuple[int, int, int]]

    @property
    def kernel(self) -> str:
        return self.header.get("kernel", "<unknown>")

    @property
    def threads_per_block(self) -> int | None:
        try:
            return int(self.header["threads_per_block"])
        except (KeyError, ValueError):
            return None

    @property
    def sm_count(self) -> int | None:
        try:
            return int(self.header["sm_count"])
        except (KeyError, ValueError):
            return None

    @property
    def arch(self) -> str | None:
        return self.header.get("arch")


def load_probe_records(path: str) -> ProbeRecords:
    """解析 ks-occupancy-records v1。

    容忍 `# key value` 头部与 `#` 注释、空行；数据行必须是 `smid,g0,g1`。
    格式不符时抛 ValueError 并给出可操作的提示（而不是静默给 0）。
    """
    header: dict[str, str] = {}
    records: list[tuple[int, int, int]] = []
    saw_format = False
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if body == PROBE_FORMAT:
                    saw_format = True
                    continue
                if " " in body:
                    k, v = body.split(" ", 1)
                    header[k.strip()] = v.strip()
                continue
            if line.lower().startswith("smid"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: 期望 'smid,g0,g1'，得到 {line!r}")
            try:
                records.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: 数值解析失败: {line!r}") from e

    if not saw_format:
        raise ValueError(f"{path}: 缺少首行 '# {PROBE_FORMAT}'（这是探针输出格式标识）")
    if not records:
        raise ValueError(f"{path}: 没有任何 block 记录，探针可能没跑到 kernel")
    return ProbeRecords(header, records)


def analyze_probe_records(rec: ProbeRecords, threads_per_block: int | None = None,
                          arch: str | None = None, sm_count: int | None = None) -> dict:
    """把探针记录算成 occupancy（区间重叠法）。

    头部里的元信息作为默认值；命令行可覆盖（例如探针文件是从别处拿来的）。
    """
    threads = threads_per_block or rec.threads_per_block or 128
    arch_used = arch or rec.arch or "sm_89"
    n_sm = sm_count or rec.sm_count or 0
    if not n_sm:
        n_sm = max((r[0] for r in rec.records), default=0) + 1

    achieved = achieved_occupancy_from_intervals(rec.records, n_sm, threads, arch_used)
    out: dict = {
        "kernel": rec.kernel,
        "arch": arch_used,
        "threads_per_block": threads,
        "sm_count": n_sm,
        "grid_blocks": len(rec.records),
        "n_records": len(rec.records),
        **achieved,
    }
    # 探针会用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 打印理论值并写进头部，
    # 这里透传便于"实测 vs 理论"对照（理论值由 CUDA API 给出，权威）
    theo = rec.header.get("theoretical_occupancy_pct")
    if theo:
        try:
            out["theoretical_occupancy_pct"] = float(theo)
        except ValueError:
            pass
    return out


def to_injection_json(result: dict) -> dict:
    """转成 `--ks-achieved-occupancy` 接受的 JSON：{kernel: {occupancy_pct, ...}}。"""
    return {
        result["kernel"]: {
            "occupancy_pct": round(float(result["occupancy_pct"]), 2),
            "avg_concurrent_blocks": round(float(result["avg_concurrent_blocks"]), 4),
            "peak_concurrent_blocks": int(result["peak_concurrent_blocks"]),
            "samples": int(result.get("samples", 0)),
        }
    }
