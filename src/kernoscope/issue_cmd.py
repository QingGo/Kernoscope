"""`kernoscope issue` —— 从 NVBit 官方 `opcode_hist` 输出算发射槽上界与瓶颈方向。

    # 采集（NVBit 需自行安装；NO_EAGER_LOAD=0 是关键，否则一条记录都没有）
    NO_EAGER_LOAD=0 CUDA_INJECTION64_PATH=$NVBIT/tools/build/opcode_hist/opcode_hist.so \
        LD_PRELOAD=$NVBIT/tools/build/opcode_hist/opcode_hist.so NOBANNER=1 \
        ./your_app > hist.txt 2>&1

    kernoscope issue --from hist.txt --kernel copy_k \
        --sm-count 128 --clock-mhz 2450 --duration-ms 86.7

它给的是**上界**：动态 warp 指令数 ÷（SM × 4 调度器 × 时钟 × 时长）。
`opcode_hist` 默认按 warp 级计数，所以分子就是发射槽占用；分母是硬件上限。
ncu 的 `smsp__warp_issue_stalled_*` 细分需要硬件计数器，这里拿不到，也不会假装有。
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyze.issue import from_histogram, to_metrics
from .backends.opcode_hist import OpcodeHistError, load
from .provenance import provenance_for_name


def build_parser() -> "argparse.ArgumentParser":
    """`kernoscope issue` 的解析器。**唯一真源**：`--help`、文档命令门禁与
    下面的 `*_main` 都从这里取（两处各写一份，门禁就会与真实 CLI 漂移）。
    """
    ap = argparse.ArgumentParser(
            prog="kernoscope issue",
            description="从 NVBit opcode_hist 的动态指令数算发射槽利用率上界与瓶颈方向"
                        "（estimated；不是 ncu 的 stall 计数器细分）",
        )
    ap.add_argument("--from", dest="src", required=True, help="opcode_hist 的 stdout 文件")
    ap.add_argument("--kernel", default=None, help="只分析名字含该子串的 kernel")
    ap.add_argument("--sm-count", type=int, default=0, help="SM 数（必需）")
    ap.add_argument("--clock-mhz", type=float, default=0.0, help="测量期 SM 时钟（必需）")
    ap.add_argument("--duration-ms", type=float, default=0.0,
                        help="这些 launch 的总时长（ms，取自 gpu__time_duration.sum）")
    ap.add_argument("--dram-pct", type=float, default=None,
                        help="同一次运行的显存忙碌占比（用于判断「是在等数据还是发不出去」）")
    ap.add_argument("--schedulers", type=int, default=4, help="每 SM 的 warp 调度器数（默认 4）")
    ap.add_argument("--json", action="store_true")
    return ap


def issue_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)


    try:
        hist = load(args.src)
    except OpcodeHistError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.kernel:
        hist = hist.for_kernel(args.kernel)
        if not hist.launches:
            print(f"error: 没有匹配 {args.kernel!r} 的 kernel；可用：{hist.kernels}", file=sys.stderr)
            return 1

    st = from_histogram(hist.merged(), duration_ns=args.duration_ms * 1e6,
                        sm_count=args.sm_count, clock_mhz=args.clock_mhz,
                        dram_throughput_pct=args.dram_pct,
                        schedulers_per_sm=args.schedulers)
    metrics = to_metrics(st)

    if args.json:
        print(json.dumps({
            "kernel": st.kernel, "launches": st.launches, "warp_instrs": st.warp_instrs,
            "thread_blocks": st.thread_blocks, "duration_ns": st.duration_ns,
            "sm_count": st.sm_count, "clock_mhz": st.clock_mhz,
            # V-27：前提不成立时 `issue_slots` 的算术结果是 0（缺时钟/缺时长），而 0 是
            # 一个**看起来像测到 0** 的数。n/a 就是 n/a：给 null，不给 0。
            "issue_slots": st.issue_slots if st.available else None,
            "issue_slot_utilization_pct": st.issue_slot_utilization_pct,
            "issued_ipc_per_sm": st.issued_ipc_per_sm,
            "category_pct": st.category_pct,
            "stall_direction": st.stall_direction,
            "available": st.available,
            "unavailable_reason": "" if st.available else st.unavailable_reason(),
            "opcode_counts": st.opcode_counts,
            "metrics": [{"name": m[0], "value": m[1], "unit": m[2], "source": m[3], "note": m[4],
                         "provenance": provenance_for_name(m[0], m[1])}
                        for m in metrics],
        }, indent=2, ensure_ascii=False))
        return 0 if st.available else 2

    ref = None if st.available else st.refusal()
    print("发射槽利用率（NVBit 动态指令计数，非硬件计数器）")
    print(f"  kernel                : {st.kernel}")
    print(f"  warp 级指令数          : {st.warp_instrs}")
    if ref is None:
        print(f"  发射槽上限（本窗口）    : {st.issue_slots:.3g}")
    else:
        # V-27：缺前提（最常见的是**没有 SM 时钟读数**）时 `issue_slots` 算出来是 0，
        # 而「0 会被读成测到 0」是这个项目自己的纪律。所以这里一律打 n/a + 闭集 code +
        # 原因：code 是现成的（`no_clock_reading` 等，见 analyze/issue.py 的 refusal()），
        # 处置也随 code 走（`kernoscope capabilities --json` / refusals 表里能查到）。
        print(f"  发射槽上限（本窗口）    : n/a（code={ref.code.value}：{ref.message}）")
        print("  无法计算              : 缺前提，见上一行的 code 与原因；"
              "0 在这里只表示『没测』，不表示『测到 0』")
        return 2
    print(f"  发射槽利用率（上界口径）: {st.issue_slot_utilization_pct:.1f} %")
    print(f"  每 SM 每周期发出指令数  : {st.issued_ipc_per_sm:.3f}")
    for cat, pct in st.category_pct.items():
        print(f"    动态指令占比 {cat:<9}: {pct:.1f} %")
    print(f"  方向性判断            : {st.stall_direction}")
    print()
    print("  说明：ncu 的 stall 周期细分（smsp__warp_issue_stalled_*）需要硬件计数器，"
          "受限环境拿不到；这里给的是「发射槽用了多少」的上界与方向判断，出处 estimated。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(issue_main(sys.argv[1:]))
