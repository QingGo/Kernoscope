"""开发侧入口层（A 轨）的契约测试。

AI-first 是两个受众，不是一件事：

* **A 轨——改这个仓库的 agent**：入口是 `AGENTS.md`，L1 是 `CONTRIBUTING.md` 首屏的索引表；
* **B 轨——用这个工具分析 kernel 的 agent**：入口是 `SKILL.md` 的 frontmatter，
  L1 是 `capabilities --brief`。

两轨的失败模式不同。B 轨的典型失败是"编造子命令"（有 `eval/TRIGGER-EVAL.md` 量过）；
A 轨的典型失败是**改了代码没重跑生成物**、**根本不知道有这条门禁**。所以这一层守的是：

  * 索引与仓库现状同源：列出的每个路径、测试、生成器、章节都真的存在；
  * 每个改动类都有测试守它（没测试的改动类只靠自觉）；
  * **凡支持 `--check` 的脚本都必须挂在某条门禁上** —— 没有这条，生成物会静默漂移：
    `eval/tasks/*.json` 就这样在仓库里红过（决策包升到 v1.1 后没人重跑它，也没有测试发现）；
  * 两个受众不再混在一份文件里：A 轨入口不带 B 轨的闭环命令。
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import dev_index  # noqa: E402

AGENTS = ROOT / "AGENTS.md"
SKILL = ROOT / "skills" / "kernoscope" / "SKILL.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


# --------------------------------------------------------------------------
# 索引与仓库现状同源
# --------------------------------------------------------------------------
def test_dev_index_check_is_clean():
    problems = dev_index.check()
    assert problems == [], "开发侧索引与仓库现状不一致：\n  - " + "\n  - ".join(problems)


def test_contributing_embeds_the_generated_index():
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert dev_index.BEGIN in text and dev_index.END in text, \
        "CONTRIBUTING.md 缺 dev-index 标记块（跑 tools/dev_index.py 写入）"
    have = text[text.index(dev_index.BEGIN):text.index(dev_index.END) + len(dev_index.END)]
    assert have == dev_index.render_block(), "索引块被手改了（它是生成物）"


def test_every_change_type_has_a_test_and_a_section():
    secs = dev_index._sections()
    for c in dev_index.CHANGES:
        assert c.tests, f"{c.key} 没有任何测试守它 —— 那它就只靠自觉"
        assert c.sec in secs, f"{c.key} 指向了不存在的章节 §{c.sec}"
        assert (ROOT / c.truth[0]).exists()


def test_index_prices_are_computed_not_typed():
    """章节价格必须随文件内容变 —— 手抄的价格和文档里的实测数字一样会漂。"""
    secs = dev_index._sections()
    assert len(secs) >= 8
    assert all(v > 200 for k, v in secs.items()), secs


def test_gates_are_a_closed_set_covering_every_generator():
    keys = {g.key for g in dev_index.GATES}
    assert {"pytest", "coverage", "agentrefs", "index", "tasks", "sanitize", "build"} <= keys
    for g in dev_index.GATES:
        prog = g.cmd.split()[-1] if g.cmd.split()[-1].endswith(".py") else None
        if prog:
            assert (ROOT / prog).exists(), f"门禁 {g.key} 指向不存在的脚本 {prog}"


def test_an_ungated_check_script_is_reported(tmp_path, monkeypatch):
    """这条规则必须**能红** —— 它就是为 `eval/make_tasks.py` 那次静默漂移加的。

    造一个"支持 --check 但没人跑"的脚本，检查器必须点它的名。
    """
    (tmp_path / "tools").mkdir()
    (tmp_path / "eval").mkdir()
    (tmp_path / "bench").mkdir()
    (tmp_path / "tools" / "orphan.py").write_text("x = '--check'\n", encoding="utf-8")
    monkeypatch.setattr(dev_index, "ROOT", tmp_path)
    problems = dev_index.check()
    assert any("orphan.py" in p and "支持 --check" in p for p in problems), problems


def test_brief_fits_its_budget_and_covers_every_change():
    text = dev_index.render_brief()
    assert len(text.encode("utf-8")) <= 1500, f"开发侧入口层已经 {len(text.encode('utf-8'))} B"
    for c in dev_index.CHANGES:
        assert c.ask in text, f"入口层少了 {c.key}"


# --------------------------------------------------------------------------
# 两个受众：A 轨入口不带 B 轨内容
# --------------------------------------------------------------------------
def test_agents_md_is_the_development_entry():
    text = AGENTS.read_text(encoding="utf-8")
    lines = len(text.splitlines())
    assert lines <= 40, f"AGENTS.md 已经 {lines} 行，入口层不许长成手册"
    assert len(text.encode("utf-8")) <= 2500
    for needle in ("dev_index.py --gates", "CONTRIBUTING.md", "skills/kernoscope/SKILL.md",
                   "生成物", "以测试为准"):
        assert needle in text, f"开发入口少了 {needle!r}"
    # 受众分流：A 轨入口不再承载 B 轨的闭环命令（那些在 SKILL.md 里，有测试守）
    assert "capabilities --brief" not in text, "开发入口里混进了使用面内容（那是 SKILL.md 的活）"


def test_skill_md_carries_the_consumer_entry():
    text = SKILL.read_text(encoding="utf-8")
    for needle in ("capabilities --brief", "--packet", "actionable", "indistinguishable"):
        assert needle in text, f"使用面入口少了 {needle!r}"


@pytest.mark.parametrize("doc", ["AGENTS.md", "llms.txt", "CONTRIBUTING.md",
                                 "skills/kernoscope/SKILL.md"])
def test_docs_reachable_from_the_entry_layer_exist(doc):
    assert (ROOT / doc).exists()


# --------------------------------------------------------------------------
# 第七轮 X-2：`--gates` 在**没有 uv** 的机器上也要能跑（租来的 GPU 机就是这样）
# --------------------------------------------------------------------------
def test_gate_commands_fall_back_to_a_given_interpreter(monkeypatch):
    mod = dev_index

    assert mod._gate_cmd("uv run pytest", python="/usr/bin/python3") == \
        "/usr/bin/python3 -m pytest"
    assert mod._gate_cmd("uv run python tools/gen_coverage.py --check",
                         python="/opt/py") == "/opt/py tools/gen_coverage.py --check"
    # uv build 没有解释器等价物：回退成 `python -m build`（缺 build 就如实失败）
    assert mod._gate_cmd("uv build", python="/opt/py") == "/opt/py -m build"
    # 不给解释器就原样（CI 走 `uv run`，一个字都不变）
    assert mod._gate_cmd("uv run pytest", python=None) == "uv run pytest"


def test_runner_python_prefers_uv_and_falls_back_when_missing(monkeypatch):
    import shutil
    import sys

    mod = dev_index
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv")
    assert mod._runner_python(None) is None, "有 uv 时必须走 uv，不改变 CI 行为"
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert mod._runner_python(None) == sys.executable
    assert mod._runner_python("/custom/python") == "/custom/python", "显式 --python 优先"


def test_run_gates_without_uv_never_invokes_uv(monkeypatch):
    """行为级断言：没有 uv 时，10 条门禁命令里**一条**都不该再提 uv。"""
    import shutil
    import sys

    mod = dev_index
    seen: list[str] = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **kw: (seen.append(cmd), _R())[1])
    assert mod.run_gates() == 0
    assert len(seen) == len(mod.GATES)
    offenders = [c for c in seen if c.startswith("uv ")]
    assert not offenders, f"没有 uv 时仍去调 uv：{offenders}"
    import shlex as _shlex
    assert all(c.startswith(_shlex.quote(sys.executable)) for c in seen), seen
