"""`kernoscope capabilities` —— 机器可读的能力清单 + 三态覆盖矩阵（W5）。

用法：
    kernoscope capabilities                 # 一屏人视图
    kernoscope capabilities --json          # 给 agent 的结构化清单
    kernoscope capabilities --probe         # 额外做一次 ncu 计数器实测探测（慢几秒）

为什么它是一个命令而不是文档里的一张表：文档里的人话不能被执行。agent 需要的是
「哪些问题能问、哪些必然被拒、被拒的 code 与补救命令、代价多少」的可编程形式。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .capabilities import CAPABILITIES_SCHEMA_VERSION, probe
from .refusals import RefusalCode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernoscope capabilities",
        description="本机能力清单：能回答哪些问题、哪些必然被拒（含 code 与补救办法）、代价多少",
    )
    p.add_argument("--json", action="store_true", help="输出结构化 JSON（给 agent 用）")
    p.add_argument("--probe", action="store_true",
                   help="额外做一次计数器实测探测（默认只做静态推断，快）")
    p.add_argument("-o", "--out", default=None, help="把 JSON 写到文件")
    p.add_argument("--quiet", action="store_true")
    return p


def capabilities_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    report = probe(probe_counters=args.probe)
    if args.json or args.out:
        blob = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(blob + "\n")
            if not args.quiet:
                print(f"wrote {args.out}", file=sys.stderr)
        if args.json:
            print(blob)
    else:
        print(report.render())
    # 能力清单本身永远是成功：它回答的是「现在能做什么」，不是「有没有做成某件事」。
    # 想把它当门禁用，就检查 JSON 里的 matrix（或看 tier）。
    return 0


def coverage_matrix_text(report) -> str:
    """供文档生成用：三态覆盖矩阵（与 `capabilities --json` 同源）。"""
    lines = ["| 问题 id | 问的是什么 | 三态 | 需要的最低层 | 会产出的指标 |", "|---|---|---|---|---|"]
    mark = {"available": "能", "degraded": "弱", "refused": "不答"}
    for a in report.answers:
        lines.append(
            f"| `{a.question.id}` | {a.question.text} | {mark[a.state]} | "
            f"{a.question.tier} | `{'`, `'.join(a.question.outputs)}` |")
    return "\n".join(lines)


__all__ = ["build_parser", "capabilities_main", "coverage_matrix_text",
           "CAPABILITIES_SCHEMA_VERSION", "RefusalCode"]
