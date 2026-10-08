"""`kernoscope memtrace` —— 分析 NVBit 访存轨迹（ks-memtrace v1）。

轨迹由 tools/probes/nvbit_memtrace（用户自行用 NVBit 编译）产出：

    make -C tools/probes/nvbit_memtrace NVBIT_PATH=~/nvbit_release_x86_64
    CUDA_INJECTION64_PATH=.../memtrace.so LD_PRELOAD=.../memtrace.so \
        ./your_app            # 产出 ks-memtrace.txt
    kernoscope memtrace --from ks-memtrace.txt
    kernoscope --ks-memtrace ks-memtrace.txt -k regex:<kernel> -o rep -- ./your_app
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyze.memtrace import analyze_memtrace, to_metrics
from .backends.memtrace import MemTraceError, load_memtrace


def memtrace_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="kernoscope memtrace",
        description="分析 NVBit 访存轨迹：sectors/request、合并效率、工作集、"
                    "L1/L2 命中率模型、DRAM 字节（均为 measured/estimated，非硬件计数器）",
    )
    ap.add_argument("--from", dest="src", required=True, help="ks-memtrace v1 轨迹文件")
    ap.add_argument("--l2-bytes", type=int, default=None, help="覆盖 L2 容量（默认取轨迹头部）")
    ap.add_argument("--l1-bytes", type=int, default=None, help="L1 每 SM 容量（默认 128 KiB）")
    ap.add_argument("--line-bytes", type=int, default=32,
                    help="建模粒度（默认 32 B = sector，与 ncu 的 sector 命中率口径对齐）")
    ap.add_argument("--l1-ways", type=int, default=4)
    ap.add_argument("--l2-ways", type=int, default=16)
    ap.add_argument("--max-accesses", type=int, default=4_000_000,
                    help="缓存模型最多分析多少次访问（超出则拒绝给 DRAM 字节，避免偏小的假值）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args(argv)

    try:
        trace = load_memtrace(args.src)
    except (OSError, MemTraceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    st = analyze_memtrace(
        trace, l2_bytes=args.l2_bytes, l1_bytes=args.l1_bytes or 128 * 1024,
        line_bytes=args.line_bytes, l1_ways=args.l1_ways, l2_ways=args.l2_ways,
        max_accesses=args.max_accesses)
    metrics = to_metrics(st)

    if args.json:
        print(json.dumps({
            "kernel": st.kernel, "records": st.records, "accesses": st.accesses,
            "sampling_ratio": st.sampling_ratio, "truncated": st.truncated,
            "sectors_per_request": st.sectors_per_request,
            "coalescing_pct": st.coalescing_pct,
            "working_set_mb": st.working_set_mb,
            "l2_bytes": st.l2_bytes,
            "l2_hit_rate_l2only_pct": st.l2_hit_rate_l2only_pct,
            "l2_hit_rate_with_l1_pct": st.l2_hit_rate_with_l1_pct,
            "dram_bytes_per_launch": st.dram_bytes_per_launch,
            "dram_bytes_unavailable_reason": st.dram_bytes_unavailable_reason,
            "op_counts": st.op_counts,
            "metrics": [{"name": m[0], "value": m[1], "unit": m[2], "source": m[3], "note": m[4]}
                        for m in metrics],
        }, indent=2, ensure_ascii=False))
        return 0

    sampled = st.sampling_ratio < 0.999
    tag = "[estimated]" if sampled else "[measured]"
    w = 26
    print("访存轨迹分析（NVBit 插桩，非硬件计数器）")
    print(f"  {'kernel':<{w}}: {st.kernel}")
    print(f"  {'warp 指令记录数':<{w}}: {st.records}  "
          f"({'全量' if not sampled else f'按 block 采样 {st.sampling_ratio * 100:.1f}%'}）")
    for name, value, unit, source, note in metrics:
        if value is None:
            continue
        prov = "[estimated]" if name.startswith("est__") or sampled else "[measured]"
        print(f"  {name:<{w}}: {value}{' ' + unit if unit else ''}  {prov}")
        print(f"      └ {source}")
    if st.dram_bytes_unavailable_reason:
        print(f"  {'DRAM 字节/launch':<{w}}: n/a  [unavailable]")
        print(f"      └ {st.dram_bytes_unavailable_reason}")
    print()
    print("  对照关系：sectors/request ↔ ncu l1tex__average_t_sectors_per_request；")
    print("            L2 命中率 ↔ ncu lts__t_sector_hit_rate（本工具是 LRU 模型的上界，不是计数器）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(memtrace_main(sys.argv[1:]))
