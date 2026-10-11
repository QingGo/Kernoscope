#!/usr/bin/env python3
"""生成 `skills/kernoscope/references/diagnosis-playbook.md`（受限环境诊断手册）。

为什么必须是生成物：这份手册的三张表全都是**代码里已有的表**的视图 ——

  * 拒绝码 / 人话 / 补救动作 / 命令 / 代价  ← `refusals.CODE_TITLES` + `DEFAULT_REMEDIES`
  * 会被报出的路由（触发条件）与能力层      ← `route_cmd.FAILURE_HINTS` + `capabilities.QUESTIONS`
  * 任务 → 命令、先诊断还是直接跑            ← `capabilities.QUESTIONS` + `route_cmd.COMMANDS`
  * 永久拿不到的 ncu section                ← `sections.COUNTER_SECTIONS`

手抄一遍的结果必然是漂移：加一个拒绝码、改一条 remedy 命令，手册会继续发一份**过期的处置
建议** —— 而"会漂移的文档"比"没有文档"更坏（agent 会照着执行）。所以单一真源是代码，
本脚本只负责渲染与排版；`--check` 进 `tools/dev_index.py --gates`。

**不写源表里没有的事实**：散文只解释"这张表是怎么从代码算出来的"。凡代码里没有真源、
算不出来的东西（例如每个 code 的**数值阈值** —— threshold 由各 builder 现算，没有静态表），
就在文档里明说"代码未登记"，而不是编一段听起来对的数字。

用法：
    python tools/gen_playbook.py           # 写出 diagnosis-playbook.md
    python tools/gen_playbook.py --check   # 只校验（不一致 → exit 1 + diff 形状的信息）
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernoscope import refusals as R                      # noqa: E402
from kernoscope import route_cmd as RC                    # noqa: E402
from kernoscope.capabilities import QUESTIONS, TIERS      # noqa: E402
from kernoscope.sections import COUNTER_SECTIONS          # noqa: E402

OUT = ROOT / "skills" / "kernoscope" / "references" / "diagnosis-playbook.md"

HEADER = "<!-- 由 tools/gen_playbook.py 生成；改动请改代码后重跑，不要手改本文件 -->\n"

_QUESTION_BY_ID = {q.id: q for q in QUESTIONS}
_TIER_BY_LEVEL = {t.level: t for t in TIERS}


# ---------------------------------------------------------------------------
# 从代码算出来的小工具（每一条都能追到源表）
# ---------------------------------------------------------------------------
def _rem(code) -> "R.Remedy":
    """code → remedy（闭集里每个 code 都必须有 remedy，缺了就是缺陷，见 preflight）。"""
    return R.DEFAULT_REMEDIES.get(code) or R.default_remedy(code)


def _routes_for(code) -> list[str]:
    """哪些路由（问题 id）会报出这个 code —— 反向索引 `FAILURE_HINTS`。"""
    return [qid for qid, codes in RC.FAILURE_HINTS.items() if code.value in codes]


def _max_tier(code) -> int:
    """这个 code 关联到的最高能力层（没有登记路由时返回 -1）。"""
    return max((_QUESTION_BY_ID[q].tier for q in _routes_for(code)
                if q in _QUESTION_BY_ID), default=-1)


def _cost_text(rem) -> str:
    """代价口径与 `tools/gen_agent_refs.py` 一致（同一份 Remedy 字段，同一套读法）。"""
    if rem.cost_kind == "unavailable":
        return "本机不可行"
    if rem.cost_kind == "manual":
        return "需人工"
    if rem.cost_s <= 0:
        return "≈0"
    return f"≈{rem.cost_s:.0f}s" if rem.cost_s < 90 else f"≈{rem.cost_s / 60:.0f}min"


def _cell(text) -> str:
    """表格单元里不让 `|` 和换行把列切断。"""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _command_cell(rem) -> str:
    return f"；命令：`{_cell(rem.command)}`" if rem.command else ""


def _observed_cell(code) -> str:
    """「观测到什么（阈值/条件）」—— 只能给**触发路由**。

    数值阈值**不在**任何静态表里（`Refusal.threshold` 由各 builder 现算），所以这一列
    给不出数字；给不出条件的 code 写 `——`，不编。
    """
    qids = _routes_for(code)
    if not qids:
        return "——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由）"
    shown = "、".join(f"`{q}`" for q in qids[:3])
    more = f" 等 {len(qids)} 条" if len(qids) > 3 else ""
    return f"{len(qids)} 条路由会报它（{shown}{more}）"


def _collection_cell(code, rule: str) -> str:
    """这个 code 缺的是哪一趟采集：用关联路由的最高能力层来说（层 → TIERS 的 needs）。

    规则 A（`cost_kind=unavailable`）的 code **不是**「缺一趟采集」—— 那些是本机就没有
    这条合法路径（缺硬件 / 缺权限）。分开说，免得读者以为重跑一次就能补回来。
    """
    tier = _max_tier(code)
    tier_txt = ("——（代码里没有登记它属于哪一层）" if tier < 0 or tier not in _TIER_BY_LEVEL
                else f"层 {tier} `{_TIER_BY_LEVEL[tier].name}`：需要 {_TIER_BY_LEVEL[tier].needs}")
    if rule == "A":
        return f"——（不是缺一趟采集：`cost_kind=unavailable`，本机没有这条合法路径）；{tier_txt}"
    return tier_txt


def _irrecoverable() -> list[tuple[str, object]]:
    """「缺哪一趟采集，就永久补不了」的行 —— 只有两条**可从代码复算**的规则：

    A) `remedy.cost_kind == "unavailable"`：本机没有任何合法补救路径（永久）。
    B) 关联到层 ≥3 的路由 **且** remedy 声明"要再等 ≥60 秒补一趟测量"
       （`cost_kind == "wall_seconds"`、`cost_s >= 60`）：那趟采集必须在**运行当时**打开，
       已有的产物里不含这份数据，事后只能重跑。

    规则外的测量（例如 `--ks-clock-mhz` 这种事后补一个标量的）**不列进来** —— 宁缺勿编。
    """
    rows: list[tuple[str, object]] = []
    for c in R.RefusalCode:
        if _rem(c).cost_kind == "unavailable":
            rows.append(("A", c))
    for c in R.RefusalCode:
        rem = _rem(c)
        if (rem.cost_kind == "wall_seconds" and rem.cost_s >= 60 and _max_tier(c) >= 3):
            rows.append(("B", c))
    return rows


# ---------------------------------------------------------------------------
# preflight：源表自己不自洽时，宁可红，也不要生成一份看起来完整的错文档
# ---------------------------------------------------------------------------
def preflight() -> list[str]:
    problems = [f"refusals.py 里这些 code 没有 remedy：{R.missing_remedies()}"
                if R.missing_remedies() else ""]
    for qid, codes in RC.FAILURE_HINTS.items():
        if qid not in _QUESTION_BY_ID:
            problems.append(f"FAILURE_HINTS 指向不存在的问题：{qid}")
        for name in codes:
            try:
                R.RefusalCode(name)
            except ValueError:
                problems.append(f"FAILURE_HINTS 里有不存在的 code：{name}（路由 {qid}）")
    for q in QUESTIONS:
        if q.id not in RC.COMMANDS:
            problems.append(f"问题 {q.id} 没有命令模板（COMMANDS）")
        if q.tier not in _TIER_BY_LEVEL:
            problems.append(f"问题 {q.id} 的层 {q.tier} 不在 TIERS 里")
    for name, (code, _reason) in COUNTER_SECTIONS.items():
        if not isinstance(code, R.RefusalCode):
            problems.append(f"COUNTER_SECTIONS[{name}] 的 code 不是 RefusalCode：{code!r}")
    return [p for p in problems if p]


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _intro() -> list[str]:
    n = len(list(R.RefusalCode))
    return [
        "# 受限环境诊断手册（signal → cause → fix）",
        "",
        "这份手册给的是**受限机器**（拿不到硬件性能计数器、`ncu` 跑不起来）上的处置路径：",
        "看到哪个拒绝码、它是什么意思、下一步跑什么、要花多久。",
        "",
        "本文件由 `tools/gen_playbook.py` 从 `src/kernoscope/refusals.py`、`route_cmd.py`、",
        "`capabilities.py`、`sections.py` **现算生成** —— **散文只是视图，代码才是真源**；",
        "手改本文件会在 `tools/dev_index.py --gates` 的 `playbook` 门禁上红。",
        "",
        "三条读法：",
        "",
        f"* 拒绝码是**闭集**（{n} 个）：见到 `n/a` 先看 `code`，再照 remedy 做 —— 不要猜原因。",
        "* `cost_kind=unavailable` 的 code 单独成节：它们**不是**「环境缺件」，",
        "  在这台机器上没有任何合法补救路径（容器内 `sudo`/`setuid` 也突破不了）。",
        "* 每一列都注明来源；代码里算不出来的列写 `——`，不编（这一条比表格好看更重要）。",
        "",
    ]


def _signal_table() -> list[str]:
    fixable = [c for c in R.RefusalCode if _rem(c).cost_kind != "unavailable"]
    out = [
        "## 一、signal → cause → fix（按拒绝码）",
        "",
        f"覆盖面：闭集里的 {len(list(R.RefusalCode))} 个 code 全部登记，其中 "
        f"{len(fixable)} 个在本机有补救路径（下表），{len(list(R.RefusalCode)) - len(fixable)} 个"
        f"没有（见下一节）。**不做「受限机器上不可能发生」的人工筛选** —— 那需要判断，"
        f"而判断题不该藏在生成器里；唯一的筛选规则就是 `cost_kind=unavailable`。",
        "",
        "「观测到什么」这一列只给**触发路由**：数值阈值（`Refusal.threshold`）由各 builder "
        "现算，代码里**没有**按 code 索引的静态阈值表，所以这一列给不出数字 —— 这是缺口，"
        "不是省略。",
        "",
        "| code | 人话（CODE_TITLES） | 观测到什么（阈值/条件） | 怎么办（remedy.action + command） | 代价 |",
        "|---|---|---|---|---|",
    ]
    for c in fixable:
        rem = _rem(c)
        out.append(f"| `{c.value}` | {_cell(R.code_title(c))} | {_observed_cell(c)} | "
                   f"{_cell(rem.action)}{_command_cell(rem)} | {_cost_text(rem)} |")
    out.append("")
    return out


def _unavailable_section() -> list[str]:
    un = [c for c in R.RefusalCode if _rem(c).cost_kind == "unavailable"]
    out = [
        "## 永久拿不到（本例无法补救）",
        "",
        f"下面 {len(un)} 个 code 的 remedy 声明 `cost_kind=unavailable`：**这不是"
        f"「环境缺件」，而是本机没有这条合法路径**。它们与上面的表分开，是因为处置完全不同 ——",
        "缺件可以去装/去开；这里只能换成上表里的替代口径，或换一台有权限的机器。",
        "",
        "| code | 人话（CODE_TITLES） | 观测到什么（阈值/条件） | 怎么办（remedy.action + command） | 代价 |",
        "|---|---|---|---|---|",
    ]
    for c in un:
        rem = _rem(c)
        out.append(f"| `{c.value}` | {_cell(R.code_title(c))} | {_observed_cell(c)} | "
                   f"{_cell(rem.action)}{_command_cell(rem)} | {_cost_text(rem)} |")
    out += [
        "",
        "### 同一件事在 ncu 报告里的样子（`sections.COUNTER_SECTIONS`）",
        "",
        "从别人的 ncu 报告里读到时，这些 section 会以 `n/a` 出现，对应的 code 如下"
        f"（{len(COUNTER_SECTIONS)} 个 section，与 `sections.py` 同源）：",
        "",
        "| ncu section | code | 原因（原样引用代码里的字符串） |",
        "|---|---|---|",
    ]
    for name, (code, reason) in COUNTER_SECTIONS.items():
        out.append(f"| `{name}` | `{code.value}` | {_cell(reason)} |")
    out.append("")
    return out


def _task_table() -> list[str]:
    out = [
        "## 二、任务 → 该跑哪条检查",
        "",
        "**规则（先看这条，再看表）**：最简单的问题**直接跑命令**，不要先跑诊断 —— 命令自己会"
        "在缺件时报出拒绝码，多跑一次 `doctor` 只是白等。只有层 ≥3 的（要 NVBit 插桩或"
        "计数器权限，装齐/开权限不是一条命令能解决的事）才值得先花一次 "
        "`kernoscope capabilities --json` 确认前提。",
        "",
        f"表由 `capabilities.QUESTIONS`（问题与层）+ `route_cmd.COMMANDS`（命令模板）+ "
        f"`route_cmd.FAILURE_HINTS`（会被拒的 code）生成，{len(QUESTIONS)} 条问题全覆盖。",
        "",
        "| 任务（问题） | 问题 id | 命令（COMMANDS） | 先跑诊断还是直接跑 | 可能被拒的 code |",
        "|---|---|---|---|---|",
    ]
    for q in QUESTIONS:
        if q.tier >= 3:
            t = _TIER_BY_LEVEL.get(q.tier)
            first = (f"先 `kernoscope capabilities --json`（层 {q.tier} 需要 {t.needs}）"
                     if t else f"先确认层 {q.tier} 的前提")
        else:
            first = f"**直接跑**（层 {q.tier} 的数据源在；缺件时命令会报 code）"
        codes = RC.FAILURE_HINTS.get(q.id, ())
        code_cell = "、".join(f"`{c}`" for c in codes) if codes else "——"
        out.append(f"| {_cell(q.text)} | `{q.id}` | `{_cell(RC.COMMANDS.get(q.id, ''))}` | "
                   f"{first} | {code_cell} |")
    out.append("")
    return out


def _irrecoverable_section() -> list[str]:
    rows = _irrecoverable()
    out = [
        "## 三、缺哪一趟采集，就永久补不了",
        "",
        "有些测量**不是「分析得不够」，而是采集当时没打开采集通道**：已有的产物里根本不含这份"
        "数据，事后只能重跑负载（而且这台机器上多半跑不成）。下表由两条**可从代码复算**的"
        "规则挑出来，没有第三条：",
        "",
        "* 规则 A：`remedy.cost_kind == \"unavailable\"` —— 本机没有任何合法补救路径。",
        "* 规则 B：该 code 关联到**层 ≥3** 的路由（NVBit 插桩 / 计数器权限），且 remedy 声明"
        "要再等 ≥60 秒补一趟测量（`cost_kind=wall_seconds`、`cost_s >= 60`）。",
        "",
        "落在规则外的**不列进来**（例如事后补一个 `--ks-clock-mhz` 就能算的，不属于"
        "「缺一趟采集」）。所以这张表是**下界**，不是「补不回来的完整清单」 —— 清单的完整性"
        "受限于代码里有没有把「需要哪趟采集」写成数据。",
        "",
        "| code（谁在告诉你） | 缺的是哪一趟采集 | 事后为什么补不了（remedy.action） | 代价 |",
        "|---|---|---|---|",
    ]
    for rule, c in rows:
        rem = _rem(c)
        out.append(f"| `{c.value}`（{_cell(R.code_title(c))}） | {_collection_cell(c, rule)} | "
                   f"{_cell(rem.action)} | {_cost_text(rem)} |")
    out.append("")
    return out


def _howto() -> list[str]:
    return [
        "## 怎么用这份表",
        "",
        "* 本文件是**生成物**：真源是 `src/kernoscope/refusals.py`（code / 人话 / remedy / 代价）、"
        "`route_cmd.py`（路由、命令模板、会被拒的 code）、`capabilities.py`（问题与能力层）、"
        "`sections.py`（ncu section → code）。这里的每一句散文都只是**它们的视图**，"
        "不是另一个真源。",
        "* 改了上面任何一份代码，重跑 `uv run python tools/gen_playbook.py`；"
        "`uv run python tools/dev_index.py --gates` 里的 `playbook` 门禁会跑 "
        "`--check`，不一致就红（带 diff）。",
        "* 只想知道**一条 code** 怎么办：看第一节里那一行，或 `scripts/remedy.sh <code>`；"
        "不要因为一条 `n/a` 通读全文。",
        "* 本机实际能答什么、缺什么，现场跑 `kernoscope capabilities --json`；"
        "不知道怎么走时先 `kernoscope route \"<你的问题>\"`。",
        "",
    ]


def render() -> str:
    parts = [HEADER, *_intro(), *_signal_table(), *_unavailable_section(),
             *_task_table(), *_irrecoverable_section(), *_howto()]
    return "\n".join(parts)


def _rel(path: Path) -> str:
    """日志用的相对路径（OUT 被测试指到仓库外时退回绝对路径，不炸）。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _diff_report(old: str, new: str) -> str:
    """diff 形状的信息：让"不一致"这件事一眼能看出差在哪，而不是只说一句"不一致"。"""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                      fromfile="文件现状", tofile="按代码重新生成",
                                      lineterm="", n=1))
    if not lines:
        return ""
    head = lines[:24]
    tail = f"\n…（diff 共 {len(lines)} 行，只显示前 24 行）" if len(lines) > 24 else ""
    return "\n".join(head) + tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验受限环境诊断手册（references/diagnosis-playbook.md）")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args(argv)

    problems = preflight()
    if problems:
        for p in problems:
            print(f"源表不自洽：{p}", file=sys.stderr)
        return 1

    text = render()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old == text:
            print(f"ok  {_rel(OUT)}  {len(text.encode('utf-8'))} B")
            return 0
        print(f"{_rel(OUT)}: 与代码不一致"
              f"（跑 `uv run python tools/gen_playbook.py` 重新生成）")
        print(_diff_report(old, text))
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {_rel(OUT)}  {len(text.encode('utf-8'))} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
