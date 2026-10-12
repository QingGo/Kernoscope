"""R-68 / N5 / N6：**元门禁** —— 门禁体系自己也要有人守。

为什么需要这一层（第四轮验收报告 N5/N6/N7 共同指出的形状）：

* 全仓**没有任何测试引用 `.github/workflows`** —— 删掉本轮唯一的新 CI 入口，
  全部门禁 + 测试套件照样全绿，没人知道（这里**不写条数**：手写计数会漂 —— 第七轮 X-10 就是
  这么来的，那时工作流写"8 条门禁"而 `GATES` 已经是 9 条）；
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


def _workflow_commands(path: Path) -> list[str]:
    """工作流里每个 `run:` 步骤的**命令行**（多行块逐行拆开；注释已剔除）。

    返回的是"一条完整命令"的粒度 —— 调用方用 `==` 比对，**不是**子串：
    第六轮 V-19 的残留就是 `run: echo "… dev_index.py --gates"` 仍能靠子串匹配放绿。
    """
    out: list[str] = []
    for step in _workflow_steps(path):
        out += [ln.strip() for ln in step.splitlines() if ln.strip()]
    return out


#: CI 必须**逐字**跑的提交前门禁命令（整行相等；`echo` 包装、注释、子串都不算）。
EXPECTED_CI_COMMANDS = ("uv sync --locked", "uv run python tools/dev_index.py --gates")


def test_ci_workflow_exists_and_runs_the_gates():
    """N5 + V-19/R-105：CI 入口必须存在，且**真的**跑 `dev_index.py --gates`。

    "真的"的定义：某条 `run:` **命令整行等于**期望命令 —— 不是注释、不是别处的散文、
    也不是被 `echo`/`&&` 包住后靠子串匹配蒙混。只删文件、把步骤换成 no-op、把命令挪进
    注释、或 `run: echo "…dev_index.py --gates"`，这条都要红。
    """
    wf_dir = ROOT / ".github" / "workflows"
    files = sorted(wf_dir.glob("*.y*ml")) if wf_dir.exists() else []
    assert files, "没有任何 CI 工作流 —— 门禁只在维护者记得的时候手跑"
    commands = [c for f in files for c in _workflow_commands(f)]
    for want in EXPECTED_CI_COMMANDS:
        assert want in commands, (
            f"没有任何 `run:` 命令**整行等于** {want!r}（注释、`echo` 包装、子串都不算）：\n"
            f"  实际命令：{commands}")

    blob = "\n".join(_workflow_text_without_comments(f) for f in files)
    assert re.search(r"on:\s*\n(?:.*\n)*?\s+push:", blob), "CI 没有在 push 上触发"
    assert re.search(r"\bpull_request:", blob), "CI 没有在 PR 上触发"
    # N8：锁文件与最小权限
    assert re.search(r"^permissions:", blob, re.M), "CI 没有声明最小 `permissions:`"


def test_ci_gate_rejects_a_command_that_only_mentions_the_gates(tmp_path):
    """R-105/V-19 的负样本：`run: echo "…dev_index.py --gates"` 不许再靠子串放绿。

    第五轮把"整文件子串"改成了"`run:` 步骤里出现子串"，但 `echo "… --gates"` 仍然满足
    子串 —— 12 passed（报告实测）。现在比对的是**整行相等**，这条负样本钉住它。
    """
    body = ("on:\n  push:\n  pull_request:\npermissions:\n  contents: read\n"
            "jobs:\n  gates:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Install\n"
            "        run: uv sync --locked\n")
    wf = tmp_path / "gates.yml"
    # ① 只把门禁命令**放进 echo 的字符串里** ⇒ 不算（旧实现会放绿）
    wf.write_text(body + '      - name: Gates\n'
                         '        run: echo "uv run python tools/dev_index.py --gates"\n',
                  encoding="utf-8")
    commands = _workflow_commands(wf)
    assert "uv sync --locked" in commands
    assert "uv run python tools/dev_index.py --gates" not in commands, \
        "被 echo 包住的命令不该算成'跑了门禁'"
    # ② 真正写对时为真（负样本不许把门禁本身逼死）
    wf.write_text(body + "      - name: Gates\n"
                         "        run: uv run python tools/dev_index.py --gates\n",
                  encoding="utf-8")
    commands = _workflow_commands(wf)
    for want in EXPECTED_CI_COMMANDS:
        assert want in commands, want


#: 期望的门禁清单（顺序无关）。加/删门禁必须同时改这里 —— 这就是"有意变更"的凭据。
EXPECTED_GATES = frozenset({"pytest", "coverage", "agentrefs", "playbook", "index", "tasks",
                            "calib", "sanitize", "release", "build"})


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


def test_ci_workflow_prose_count_matches_the_gate_list():
    """第七轮 X-11：CI 工作流里的散文写"8 条门禁"，而 `GATES` 已经是 9 条。

    手写计数会漂（这个仓库已经因为手写计数红过好几次）。这里把"散文里的数字"与
    `dev_index.GATES` 的真实条数**对拍**：以后再加一条门禁，这句话不改就红。
    """
    import re

    wf = (ROOT / ".github" / "workflows" / "gates.yml").read_text(encoding="utf-8")
    m = re.search(r"(\d+)\s*条门禁", wf)
    if m is None:            # 改成"全部门禁一把跑"这种不带数字的写法也可以（那就没有漂移面）
        assert "全部门禁一把跑" in wf, wf
        return
    assert int(m.group(1)) == len(dev_index.GATES), (
        f"CI 工作流写 {m.group(1)} 条门禁，`dev_index.GATES` 实际 {len(dev_index.GATES)} 条")
