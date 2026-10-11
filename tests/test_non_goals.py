"""「不做清单」必须是一份**可机检的决策记录**，不是散文。

为什么需要这条门禁（来自实际事故）：不做清单此前散落在四五份文档里，于是同一件事出现了互相
矛盾的说法 —— 两处写着「不写 `.ncu-rep`」，而代码已经在导出**带署名的替代口径**；也没有任何
东西记录"这条判断的前提是什么、什么情况下该推翻它"。收敛成 `NON-GOALS.md`（单一真源）之后，
还必须防止它再次漂移回散文。这里守四件事：

  1. **每条必须写全六件事**（声明/保护/代价/前提/择中/判定）。缺一件就是**假决策** ——
     只写"不做"而不写"为什么不做、代价是什么、前提变了怎么办"的条目，会在前提变化后继续生效。
  2. **判定必须取自闭集**（保留/改写法/部分解除/解除），否则回退成散文。
  3. **§3.1「政策已定、尚未实现」必须存在**，且它点名的条目 id 必须真实存在 ——
     这条防"决定了但没人知道做没做"，也防 README 把没实现的东西写成已实现的行为。
  4. **README 必须指向真源**，否则就会出现第二份口径。

变异测试保证这四条**能红**，而不是永远绿的摆设（同 `test_documented_commands` 的纪律）。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "NON-GOALS.md"
README = ROOT / "README.md"

FIELDS = ("声明", "保护", "代价", "前提", "择中", "判定")
VERDICTS = ("保留", "改写法", "部分解除", "解除")

_ITEM_RE = re.compile(r"^###\s+(N\d+)\s*·", re.M)
_FIELD_RE = re.compile(r"^-\s*\*\*(声明|保护|代价|前提|择中|判定)\*\*(?:（[^）]*）)?：", re.M)
_VERDICT_RE = re.compile(r"^-\s*\*\*判定\*\*：\s*\*\*(.+?)\*\*", re.M)
_STATUS_HEAD_RE = re.compile(r"^###\s+3\.1\b.*$", re.M)
_STATUS_ROW_RE = re.compile(r"^\|\s*\*\*(N\d+)\*\*", re.M)
NEXT_H2_RE = re.compile(r"^##\s", re.M)


def _items(text: str) -> dict[str, str]:
    """`{N1: 正文, …}` —— 从 `### N1 · …` 到下一个 `###`/`##` 之前。"""
    out: dict[str, str] = {}
    marks = [(m.start(), m.group(1)) for m in _ITEM_RE.finditer(text)]
    for i, (start, nid) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out[nid] = text[start:end]
    return out


def _status_section(text: str) -> str | None:
    """§3.1 的正文（到下一个 `##` 之前）；没有这一节返回 None。"""
    m = _STATUS_HEAD_RE.search(text)
    if not m:
        return None
    tail = text[m.end():]
    nxt = NEXT_H2_RE.search(tail)
    return tail[:nxt.start()] if nxt else tail


def validate(non_goals: str, readme: str) -> list[str]:
    """返回问题列表（空 = 通过）。纯函数，变异测试复用同一条判据。"""
    problems: list[str] = []
    items = _items(non_goals)
    if not items:
        problems.append("一条 `### N<n> · …` 条目都没解析到（标题格式退化了？）")

    for nid, body in sorted(items.items()):
        missing = [f for f in FIELDS if f not in set(_FIELD_RE.findall(body))]
        if missing:
            problems.append(f"{nid} 缺字段：{missing}（六件事缺一件就是假决策）")
        vm = _VERDICT_RE.search(body)
        if not vm:
            problems.append(f"{nid} 没有可解析的「判定」行（写成 `- **判定**：**<闭集词>…**`）")
        elif not vm.group(1).startswith(VERDICTS):
            problems.append(f"{nid} 的判定 {vm.group(1)!r} 不在闭集 {VERDICTS} 里")

    status = _status_section(non_goals)
    if status is None:
        problems.append("缺少 §3.1「政策已定、尚未实现」一节（决定了却没记录做没做）")
    else:
        ids = _STATUS_ROW_RE.findall(status)
        if not ids:
            problems.append("§3.1 一条实施状态都没列")
        for nid in ids:
            if nid not in items:
                problems.append(f"§3.1 的实施状态引用了不存在的条目 {nid}")

    if "它明确不做什么" not in readme:
        problems.append("README 缺少「它明确不做什么」一节")
    if "NON-GOALS.md" not in readme:
        problems.append("README 没有指向真源 NON-GOALS.md（会出现第二份口径）")
    return problems


# --------------------------------------------------------------------------
# 1) 真文件必须通过
# --------------------------------------------------------------------------
def _real() -> tuple[str, str]:
    return DOC.read_text(encoding="utf-8"), README.read_text(encoding="utf-8")


def test_real_files_pass():
    problems = validate(*_real())
    assert not problems, "\n".join(problems)


def test_every_item_is_parsed():
    """条目数是显式契约：删掉一条也要在这里交代（否则清单可以被悄悄缩短）。"""
    items = _items(DOC.read_text(encoding="utf-8"))
    assert len(items) == 13, f"不做清单现在有 {len(items)} 条：{sorted(items)}"


def test_doc_is_tracked_not_ignored():
    """它必须在版本库里 —— 这份记录曾经只活在几份不进库的工作文档里，那正是漂移的成因。"""
    out = subprocess.run(["git", "ls-files", "--error-unmatch", "NON-GOALS.md"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, "NON-GOALS.md 没有被 git 跟踪（决定记录必须可审计）"


def test_readme_does_not_claim_unimplemented_middle_grounds_as_done():
    """README 只许写已实现的形态：单文件 HTML 渲染器目前**没有**实现，必须带"尚未实现"。"""
    readme = README.read_text(encoding="utf-8")
    section = readme[readme.index("### 它明确不做什么"):]
    section = section[:section.index("\n---")]
    assert "尚未实现" in section, "README 的不做清单必须区分「已实现」与「已决定未实现」"


# --------------------------------------------------------------------------
# 2) 变异：同一条判据必须能红
# --------------------------------------------------------------------------
_MUTATIONS = [
    ("少一个字段",
     lambda ng, rd: (ng.replace("- **择中**：", "- 顺带一提：", 1), rd),
     "缺字段"),
    ("判定用散文词",
     lambda ng, rd: (ng.replace("**判定**：**保留**", "**判定**：**也许吧**", 1), rd),
     "不在闭集"),
    ("实施状态引用不存在的 id",
     lambda ng, rd: (ng.replace("| **N7** ", "| **N99** ", 1), rd),
     "不存在的条目"),
    ("删掉 §3.1",
     lambda ng, rd: (ng.replace("### 3.1 政策已定", "### 3.7 政策已定", 1), rd),
     "缺少 §3.1"),
    ("README 不再指向真源",
     lambda ng, rd: (ng, rd.replace("NON-GOALS.md", "NON-GOALS.txt")),
     "没有指向真源"),
]


@pytest.mark.parametrize("tag,mutate,needle", _MUTATIONS)
def test_validator_catches_regressions(tag, mutate, needle):
    non_goals, readme = _real()
    ng2, rd2 = mutate(non_goals, readme)
    assert (ng2, rd2) != (non_goals, readme), f"{tag}：变异没生效（夹具过期了）"
    problems = validate(ng2, rd2)
    assert any(needle in p for p in problems), f"{tag}：变异没被抓到（{problems}）"
