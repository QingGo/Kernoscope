#!/usr/bin/env python3
"""发现维度评测：agent 会不会在**正确的时刻**选择 Kernoscope（G4 的 trigger_rate）。

W8 的评测台量的是"给了材料会不会误用"；这个脚本量的是它前面那一步：
**给一个真实任务描述 + 一份工具目录，它会不会想起来用我们。**

为什么要单独量：skill 的 `description` 是唯一常驻上下文的触发词（约 120 个词），
它决定采纳率里的"可发现性"这一项。它属于产品表面，不是文案 —— 所以要有数。

契约：被测 agent 在任务目录里写 `choice.json`：

    {"use": ["kernoscope"], "why": "…", "first_command": "kernoscope capabilities --brief"}

  * `use` 只能取工具目录里的 id（可以是多个，也可以是空列表 = 不需要工具）；
  * `first_command` 可选，用来检查它是否走我们设计的入口层。

指标：
  trigger_rate         应触发时真的选了我们的比例
  false_positive_rate  不应触发时选了我们（或把性能工具当答案）的比例
  entry_layer_rate     触发时第一条命令走 `capabilities --brief` 的比例

用法：
    python eval/trigger_eval.py --prepare-only --out /tmp/tr   # 只落地材料
    python eval/trigger_eval.py --score-only  --out /tmp/tr    # 只给已有 choice.json 打分
    python eval/trigger_eval.py --agent-cmd "python my_agent.py"   # 自己起子进程
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "eval" / "discovery"
CHOICE = "choice.json"
DEFAULT_TIMEOUT_S = 300.0

ENTER = "Kernoscope：取证式 kernel 性能分析"

# 我们真实存在的子命令与"无子命令也算合法"的入口；出现别的 `kernoscope <x>` 就是编的。
# 为什么要量这个：只给触发词（description）时，agent 可能凭印象编出一条命令 ——
# 真事：一次评测里出现了不存在的 `kernoscope compare --baseline …`。
KNOWN_SUBCOMMANDS = {"capabilities", "selftest", "diff", "doctor", "occupancy",
                     "memtrace", "issue"}


def command_shape(cmd: str) -> dict:
    """给第一条命令分类：走入口层 / 合法但不走入口层 / **编的**。"""
    parts = (cmd or "").strip().split()
    if not parts or parts[0] != "kernoscope":
        return {"is_kernoscope": False, "entry_layer": False, "made_up": False}
    # `--` 之后是被 profile 的目标命令，不是我们的子命令
    tokens = parts[1:]
    if "--" in tokens:
        tokens = tokens[:tokens.index("--")]
    positional = [t for t in tokens if not t.startswith("-")]
    sub = positional[0] if positional else ""
    made_up = bool(sub) and sub not in KNOWN_SUBCOMMANDS
    entry_layer = (not sub) or tokens[:1] == ["capabilities"]
    return {"is_kernoscope": True, "entry_layer": entry_layer, "made_up": made_up}


def case_paths(names: list[str] | None = None) -> list[Path]:
    paths = sorted(CASES_DIR.glob("*.json"))
    if names:
        out = []
        for n in names:
            hit = [p for p in paths if p.stem == n or p.stem.startswith(n)]
            if not hit:
                raise SystemExit(f"error: 没有匹配的任务 {n!r}（在 {CASES_DIR}）")
            out.extend(hit)
        return out
    return paths


def load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def skill_description() -> str:
    """从 SKILL.md 的 frontmatter 现读触发词 —— 评测必须跟着真实描述走，不能抄一份。"""
    text = (ROOT / "skills" / "kernoscope" / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^description:\s*>?-?\s*\n?((?:.|\n)*?)(?=\n\w+:|\Z)",
                  text.split("---")[1], re.M)
    if not m:
        raise SystemExit("error: 读不到 SKILL.md 的 description")
    return " ".join(m.group(1).split())


def prepare(case: dict, dest: Path) -> None:
    """干净房间：只写 agent 该看的东西（任务描述 + 工具目录）。"""
    if dest.exists():
        for p in dest.rglob("*"):
            if p.is_file():
                p.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    catalog = {"kernoscope": skill_description(), **case["catalog"]}
    lines = ["# 可用工具（目录）", ""]
    for tid, desc in catalog.items():
        lines.append(f"- `{tid}`：{desc}")
    lines += ["", "（目录之外的工具不属于本次可选范围。）", ""]
    (dest / "CATALOG.md").write_text("\n".join(lines), encoding="utf-8")
    (dest / "TASK.md").write_text(
        "# 任务\n\n" + case["task"].strip() + "\n\n## 你要产出\n\n"
        "在当前目录写 `choice.json`：\n\n"
        "```json\n"
        '{"use": ["工具目录里的 id，可多个；不需要工具就给空列表"],'
        ' "why": "一句话理由", "first_command": "如果要用，第一条命令"}\n'
        "```\n", encoding="utf-8")


def score(case: dict, choice: dict | None, fail_state: str | None) -> dict:
    """`expect_any`：这些工具里**至少一个**要出现（空列表 = "不需要工具"也算对）；
    `forbid_use`：这些一个都不许出现。`should_trigger` 指"从我们视角应当被选中"。"""
    expect = list(case.get("expect_any", []))
    forbid = list(case.get("forbid_use", []))
    out = {"case_id": case["id"], "expect_any": expect, "forbid_use": forbid,
           "status": fail_state or "ok", "use": [], "why": "", "first_command": "",
           "should_trigger": "kernoscope" in expect, "did_trigger": False,
           "expected_ok": False, "false_positive": False, "entry_layer_ok": None}
    if choice is None:
        out["expected_ok"] = False
        out["false_positive"] = False
        return out
    use = [str(x) for x in (choice.get("use") or [])]
    out["use"] = use
    out["why"] = str(choice.get("why") or "")
    out["first_command"] = str(choice.get("first_command") or "")
    did = "kernoscope" in use
    out["did_trigger"] = did
    if did:
        shape = command_shape(out["first_command"])
        out["first_command_is_kernoscope"] = shape["is_kernoscope"]
        out["entry_layer_ok"] = shape["entry_layer"] or out["first_command"].strip().startswith(
            "kernoscope --packet")
        out["made_up_command"] = shape["made_up"]
    out["expected_ok"] = ((not expect or any(e in use for e in expect))
                          and not any(f in use for f in forbid))
    out["false_positive"] = did and "kernoscope" in forbid
    return out


def summarize(results: list[dict], out_dir: Path) -> dict:
    pos = [r for r in results if r["should_trigger"]]
    neg = [r for r in results if not r["should_trigger"]]
    fired = [r for r in pos if r["did_trigger"]]
    entry = [r for r in fired if r["entry_layer_ok"]]
    board = {
        "cases": len(results),
        "positive_cases": len(pos),
        "negative_cases": len(neg),
        "trigger_rate": round(len(fired) / len(pos), 3) if pos else None,
        "expected_ok_rate": round(sum(1 for r in results if r["expected_ok"]) / len(results), 3),
        "false_positive_rate": (round(sum(1 for r in neg if r["false_positive"]) / len(neg), 3)
                                if neg else None),
        "entry_layer_rate": (round(len(entry) / len(fired), 3) if fired else None),
        "made_up_command_rate": (round(
            sum(1 for r in fired if r.get("made_up_command")) / len(fired), 3)
            if fired else None),
        "out_dir": str(out_dir),
    }
    return board


def render(board: dict, results: list[dict]) -> str:
    out = [f"# 发现维度评测（trigger）· 用例 {board['cases']} "
           f"（正 {board['positive_cases']} / 负 {board['negative_cases']}）", ""]
    out.append(f"- trigger_rate = **{board['trigger_rate']}**（应触发时真的选了 kernoscope）")
    out.append(f"- false_positive_rate = **{board['false_positive_rate']}**"
               f"（不该用时选了我们）")
    out.append(f"- entry_layer_rate = **{board['entry_layer_rate']}**"
               f"（触发时第一条命令走入口层）")
    out.append(f"- made_up_command_rate = **{board.get('made_up_command_rate')}**"
               f"（触发时第一条命令是编的：写了不存在的子命令）")
    out.append(f"- expected_ok_rate = {board['expected_ok_rate']}")
    out.append("")
    out.append("| 用例 | 应触发 | 用了什么 | 是否符合预期 | 理由 |")
    out.append("|---|---|---|---|---|")
    for r in results:
        out.append(f"| `{r['case_id']}` | {'是' if r['should_trigger'] else '否'} | "
                   f"{', '.join(r['use']) or '（不用工具）'} | "
                   f"{'✓' if r['expected_ok'] else '✗'} | {r['why'][:60]} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="发现维度评测：agent 会不会在正确的时刻用 Kernoscope")
    ap.add_argument("--agent-cmd", default=os.environ.get("KS_EVAL_AGENT_CMD"))
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    cases = [load_case(p) for p in case_paths(args.case)]
    argv_agent: list[str] = []
    if args.agent_cmd:
        argv_agent = shlex.split(args.agent_cmd)
    elif not (args.prepare_only or args.score_only):
        print("error: 需要 --agent-cmd，或用 --prepare-only/--score-only", file=sys.stderr)
        return 2

    results: list[dict] = []
    for case in cases:
        dest = out_dir / case["id"]
        if not args.score_only:
            prepare(case, dest)
        choice = None
        fail = None
        if args.score_only or args.agent_cmd:
            path = dest / CHOICE
            if path.exists():
                try:
                    choice = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    fail = f"choice.json 不是合法 JSON：{exc}"
            else:
                fail = "没有写 choice.json"
        if argv_agent and not args.prepare_only and not args.score_only:
            t0 = time.monotonic()
            proc = subprocess.run(argv_agent, cwd=dest, capture_output=True, text=True,
                                  timeout=args.timeout)
            if proc.returncode != 0:
                fail = f"agent 退出码 {proc.returncode}"
            path = dest / CHOICE
            if path.exists():
                try:
                    choice = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    fail = f"choice.json 不是合法 JSON：{exc}"
            elif fail is None:
                fail = "没有写 choice.json"
            (dest / "agent.log").write_text(
                f"rc={proc.returncode} wall={time.monotonic() - t0:.2f}s\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
                encoding="utf-8")
        r = score(case, choice, fail)
        results.append(r)
        mark = "ok " if r["expected_ok"] else "BAD"
        print(f"[{mark}] {r['case_id']}　use={r['use'] or '-'}　{r['status']}",
              file=sys.stderr)

    board = summarize(results, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trigger-board.json").write_text(
        json.dumps({"board": board, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "trigger-summary.md").write_text(render(board, results), encoding="utf-8")
    if args.json:
        print(json.dumps(board, ensure_ascii=False, indent=2))
    print(f"board: {out_dir / 'trigger-board.json'}", file=sys.stderr)
    missing = [r["case_id"] for r in results if r["status"] != "ok"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
