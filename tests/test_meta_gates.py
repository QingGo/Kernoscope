"""R-68 / N5 / N6：**元门禁** —— 门禁体系自己也要有人守。

为什么需要这一层（第四轮验收报告 N5/N6/N7 共同指出的形状）：

* 全仓**没有任何测试引用 `.github/workflows`** —— 删掉本轮唯一的新 CI 入口，
  8 条门禁 + 400 多个测试照样全绿，没人知道；
* `dev_index.py` 里每类改动声明的"重跑生成器"也没有测试引用 —— 把
  `eval/make_tasks.py` 从"加/改能力问题"的重跑列表里删掉，`--check` 仍 exit 0
  （反查规则被"`make_tasks` 已经在 GATES 里"这件事满足了），于是那句话又变成声明。

所以这一层守两件事：**CI 还在、且真的跑门禁**；**每类改动的重跑耦合是明确的表**
（改耦合必须改这张表，改表会红）。
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
from pathlib import Path

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_dev_index():
    spec = importlib.util.spec_from_file_location("ks_dev_index", ROOT / "tools" / "dev_index.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# 每类改动的"重跑"耦合：**有意写死**。它是声明的一部分，改了要在这里改，
# 于是"顺手删掉一个生成器"不再可能悄悄通过。
EXPECTED_REGEN = {
    "metric": set(),
    "backend": set(),
    "workload": set(),
    "question": {"tools/gen_coverage.py", "eval/make_tasks.py"},
    "refusal": {"tools/gen_agent_refs.py"},
    "artifact": {"tools/gen_calibration_excerpt.py"},
    "verdict": {"tools/gen_agent_refs.py", "eval/make_tasks.py"},
    "entry": set(),
    "nongoal": set(),
    "cli": set(),
    "index": set(),
}


def test_change_types_declare_the_expected_regenerators():
    """N6：每类改动的重跑列表 == 这张表（删掉一个生成器要红）。"""
    mod = _load_dev_index()
    got = {c.key: set(c.regen) for c in mod.CHANGES}
    assert set(got) == set(EXPECTED_REGEN), (
        f"改动类别变了：{sorted(set(got) ^ set(EXPECTED_REGEN))}（请同步 EXPECTED_REGEN）")
    diff = {k: (got[k], EXPECTED_REGEN[k]) for k in got if got[k] != EXPECTED_REGEN[k]}
    assert not diff, "重跑耦合与声明不一致（实得 vs 期望）：\n  " + "\n  ".join(
        f"{k}: {sorted(v[0])} vs {sorted(v[1])}" for k, v in diff.items())


def _workflow_steps(path: Path) -> list[str]:
    """把 CI 工作流里的 **run 步骤**逐条抠出来（不依赖 PyYAML）：

    第五轮 V-19：旧实现是"整文件子串匹配"，于是把真步骤换成 `python -c "print(1)"`、
    只在注释里留一句 `dev_index.py --gates`，测试照样全绿 —— 守卫被工作流自己的注释满足。
    这里按行解析 `run:` 块（含 `run: |` 的折叠块），注释行**先剔除**，只有真的命令行才算。
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    steps: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("run:"):
            rest = stripped[len("run:"):].strip()
            if rest and rest != "|" and rest != ">":
                steps.append(rest)
                i += 1
                continue
            indent = len(raw) - len(raw.lstrip())
            i += 1
            block: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                if nxt.strip() and not nxt.strip().startswith("#"):
                    block.append(nxt.strip())
                i += 1
            steps.append("\n".join(block))
            continue
        i += 1
    return steps


def _workflow_text_without_comments(path: Path) -> str:
    return "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                      if not l.strip().startswith("#"))


def test_ci_workflow_exists_and_runs_the_gates():
    """N5 + V-19：CI 入口必须存在，且**真的**跑 `dev_index.py --gates`。

    "真的"的定义：它出现在某个 `run:` 步骤里 —— 不是注释、不是别处的散文。
    只删文件、把步骤换成 no-op、或把命令挪进注释，这条都要红。
    """
    wf_dir = ROOT / ".github" / "workflows"
    files = sorted(wf_dir.glob("*.y*ml")) if wf_dir.exists() else []
    assert files, "没有任何 CI 工作流 —— 门禁只在维护者记得的时候手跑"
    steps = [st for f in files for st in _workflow_steps(f)]
    joined = "\n".join(steps)
    assert any("dev_index.py --gates" in st for st in steps), (
        "没有任何 `run:` 步骤真的跑 `dev_index.py --gates`（注释里出现不算）：" + repr(steps))

    blob = "\n".join(_workflow_text_without_comments(f) for f in files)
    assert re.search(r"on:\s*\n(?:.*\n)*?\s+push:", blob), "CI 没有在 push 上触发"
    assert re.search(r"\bpull_request:", blob), "CI 没有在 PR 上触发"
    # N8：锁文件与最小权限
    assert any("uv sync --locked" in st for st in steps), \
        "CI 的 `run:` 步骤里没有 `uv sync --locked`（装了别的版本没人知道）"
    assert re.search(r"^permissions:", blob, re.M), "CI 没有声明最小 `permissions:`"


#: 期望的门禁清单（顺序无关）。加/删门禁必须同时改这里 —— 这就是"有意变更"的凭据。
EXPECTED_GATES = frozenset({"pytest", "coverage", "agentrefs", "playbook", "index", "tasks",
                            "calib", "sanitize", "build"})


def test_gate_list_is_exactly_the_expected_one():
    """R-68：参数化清单此前手抄 8 条而 `GATES` 已有 9 条（漏了 `playbook`）——
    漏掉的那条门禁没有任何测试引用，删了也没人知道。现在显式钉住集合。"""
    mod = _load_dev_index()
    assert {g.key for g in mod.GATES} == set(EXPECTED_GATES), (
        f"门禁清单变了：{sorted({g.key for g in mod.GATES})} vs "
        f"{sorted(EXPECTED_GATES)}（改测试里的 EXPECTED_GATES 才算有意变更）")


@pytest.mark.parametrize("gate", sorted(EXPECTED_GATES))
def test_every_gate_is_known_to_the_index(gate: str):
    """每条门禁都要在索引里有命令与说明（清单从 EXPECTED_GATES 派生，不再手抄一份）。"""
    mod = _load_dev_index()
    assert gate in {g.key for g in mod.GATES}, f"门禁 {gate} 不在 GATES 里（清单变了？）"
    g = next(x for x in mod.GATES if x.key == gate)
    assert g.cmd and g.what, f"门禁 {gate} 缺命令或缺 what"
