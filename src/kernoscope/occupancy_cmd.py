"""`kernoscope occupancy` —— 从探针记录计算「达到的 occupancy」（区间重叠法）。

配合 tools/probes/occupancy_probe.cu 使用，把 record 文件变成
`--ks-achieved-occupancy` 能吃的 JSON，形成闭环：

    ./occupancy_probe occupancy-records.txt
    kernoscope occupancy --from occupancy-records.txt --write-json occ.json
    kernoscope --ks-achieved-occupancy occ.json -k regex:<kernel> -- <cmd>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze.occupancy import (ARCH, analyze_probe_records, load_probe_records,
                                to_injection_json)


def build_parser() -> "argparse.ArgumentParser":
    """`kernoscope occupancy` 的解析器。**唯一真源**：`--help`、文档命令门禁与
    下面的 `*_main` 都从这里取（两处各写一份，门禁就会与真实 CLI 漂移）。
    """
    ap = argparse.ArgumentParser(
            prog="kernoscope occupancy",
            description="从探针记录（ks-occupancy-records v1）计算达到的 occupancy —— "
                        "ncu 用硬件计数器，这里用 (smid, 起止时间) 的区间重叠重建",
        )
    ap.add_argument("--from", dest="src", required=True, help="探针输出文件")
    ap.add_argument("--kernel", default=None, help="覆盖记录中的 kernel 名")
    ap.add_argument("--threads", type=int, default=None, help="每 block 线程数（默认取记录头部）")
    ap.add_argument("--arch", default=None, help=f"目标架构（默认取记录头部），可用: {', '.join(ARCH)}")
    ap.add_argument("--sm-count", type=int, default=None, help="SM 数量（默认取记录头部）")
    ap.add_argument("--write-json", default=None,
                        help="把结果写成 --ks-achieved-occupancy 可用的 JSON")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出分析结果")
    return ap


def occupancy_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)


    try:
        rec = load_probe_records(args.src)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    result = analyze_probe_records(rec, threads_per_block=args.threads, arch=args.arch,
                                   sm_count=args.sm_count)
    if args.kernel:
        result["kernel"] = args.kernel

    if args.write_json:
        Path(args.write_json).write_text(json.dumps(to_injection_json(result), indent=2))
        result["written"] = args.write_json

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    a = ARCH.get(result["arch"]) or ARCH["sm_89"]
    warps = -(-result["threads_per_block"] // 32)
    print("达到的 occupancy（区间重叠法，出处 measured）")
    print(f"  kernel            : {result['kernel']}")
    print(f"  arch              : {result['arch']}（每 SM {a['warps']} warps）")
    print(f"  threads/block     : {result['threads_per_block']}（{warps} warps/block）")
    print(f"  blocks / SM 数    : {result['grid_blocks']} / {result['sm_count']}")
    print(f"  平均并发 blocks/SM: {result['avg_concurrent_blocks']:.2f}"
          f"（峰值 {int(result['peak_concurrent_blocks'])}）")
    print(f"  达到 occupancy    : {result['occupancy_pct']:.1f}%   [measured]")
    print()
    print("  说明：区间重叠法用每个 block 的 (smid, 起止时间) 重建时间平均并发度，")
    print("        语义与 ncu 的 Achieved Occupancy 一致，但出处是 measured（我们自己的探针），")
    print("        因此不适合跨工具/跨版本直接比较。")
    if args.write_json:
        print(f"  已写出 {args.write_json}（可直接用于 --ks-achieved-occupancy）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(occupancy_main(sys.argv[1:]))
