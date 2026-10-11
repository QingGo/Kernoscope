"""受限环境诊断手册（`references/diagnosis-playbook.md`）的契约测试。

这份文档的价值全在**与代码同源**上：它把拒绝码、remedy、路由、能力层、ncu section 渲成一张
人/agent 能直接照着做的表。一旦它开始漂移，agent 会拿着**过期的处置建议**去执行 —— 比没有
文档更坏。所以这里守四件事：

* `--check` 与磁盘字节一致（生成物不许手改）；
* 不一致时 `--check` 必须**红**，并给出 diff 形状的信息（不是只说一句"不一致"）；
* 表里的每一行都能追回代码：code 覆盖闭集、问题覆盖 QUESTIONS、命令逐字来自 COMMANDS/remedy；
* 改代码 → 文档跟着变（monkeypatch 源表后重新渲染，必须看见新值）—— 这是"无法漂移"的证明。
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernoscope import refusals as R                    # noqa: E402
from kernoscope import route_cmd as RC                  # noqa: E402
from kernoscope.capabilities import QUESTIONS, TIERS    # noqa: E402
from kernoscope.sections import COUNTER_SECTIONS        # noqa: E402

DOC = ROOT / "skills" / "kernoscope" / "references" / "diagnosis-playbook.md"


def _load_gen():
    spec = importlib.util.spec_from_file_location("ks_gen_playbook",
                                                  ROOT / "tools" / "gen_playbook.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
TEXT = DOC.read_text(encoding="utf-8")


def _text_of(path) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """从 `heading` 到下一个同级 `## ` 标题之间的正文（表就在里面）。"""
    i = text.index(heading)
    rest = text[i + len(heading):]
    j = rest.find("\n## ")
    return rest[:j] if j >= 0 else rest


# ---------------------------------------------------------------------------
# 生成物纪律
# ---------------------------------------------------------------------------
def test_playbook_is_generated_and_in_sync():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_playbook.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert TEXT.startswith("<!-- 由 tools/gen_playbook.py 生成"), "生成物必须带「不要手改」抬头"
    assert DOC.exists() and len(TEXT.encode("utf-8")) > 4000


def test_check_fails_on_corrupted_file_with_a_diff_shaped_message(tmp_path, monkeypatch, capsys):
    """`--check` 必须在字节不一致时**红**，并指出差在哪（这里是"真的改坏再修好"）。"""
    target = tmp_path / "diagnosis-playbook.md"
    target.write_text(GEN.render(), encoding="utf-8")
    monkeypatch.setattr(GEN, "OUT", target)

    assert GEN.main(["--check"]) == 0
    capsys.readouterr()

    bad = GEN.render().replace("## 一、signal", "## 一、被改坏的手改版", 1)
    target.write_text(bad, encoding="utf-8")
    assert GEN.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "不一致" in out and "gen_playbook.py" in out
    # diff 形状：有 ---/+++/@@ 的文件头与 hunk，而不是一句"不一致"了事
    assert "--- " in out and "+++ " in out and "@@" in out
    assert "-## 一、被改坏的手改版" in out and "+## 一、signal" in out


# ---------------------------------------------------------------------------
# 第一张表：拒绝码覆盖闭集 + 程序性不可补救的单独成节
# ---------------------------------------------------------------------------
def test_every_refusal_code_has_a_row_and_unavailable_ones_are_grouped():
    missing = [c.value for c in R.RefusalCode if f"`{c.value}`" not in TEXT]
    assert not missing, f"手册漏了这些 code：{missing}"

    head = "## 一、signal → cause → fix（按拒绝码）"
    main_table = _section(TEXT, head).split("## 永久拿不到")[0]
    un_section = _section(TEXT, "## 永久拿不到（本例无法补救）")

    for c in R.RefusalCode:
        if R.DEFAULT_REMEDIES[c].cost_kind == "unavailable":
            assert f"`{c.value}`" not in main_table, f"{c.value} 不该出现在可补救的表里"
            assert f"`{c.value}`" in un_section, f"{c.value} 必须出现在「永久拿不到」节"
        else:
            assert f"`{c.value}`" in main_table, f"{c.value} 缺一行"


def test_rows_carry_the_code_truth_verbatim():
    """每行的 remedy action、命令、代价、人话都必须逐字来自代码（不是重写的散文）。"""
    for c in R.RefusalCode:
        rem = R.DEFAULT_REMEDIES[c]
        assert rem.action.replace("|", "\\|") in TEXT, f"{c.value} 的 action 没有逐字出现"
        assert R.code_title(c) in TEXT, f"{c.value} 的人话没有出现"
        if rem.command:
            assert rem.command.replace("|", "\\|") in TEXT, f"{c.value} 的命令没有逐字出现"


def test_counter_sections_are_documented():
    sec = _section(TEXT, "### 同一件事在 ncu 报告里的样子")
    for name, (code, _reason) in COUNTER_SECTIONS.items():
        assert f"`{name}`" in sec and f"`{code.value}`" in sec, name


# ---------------------------------------------------------------------------
# 第二张表：任务 → 命令（并带"最简单就直接跑"的规则）
# ---------------------------------------------------------------------------
def test_task_table_covers_every_question_with_its_own_command():
    sec = _section(TEXT, "## 二、任务 → 该跑哪条检查")
    for q in QUESTIONS:
        assert f"`{q.id}`" in sec, f"任务表少了 {q.id}"
        cmd = RC.COMMANDS[q.id].replace("|", "\\|")
        assert cmd in sec, f"{q.id} 的命令不是逐字来自 route_cmd.COMMANDS"
    # "最简单就直接跑"必须写成规则，而不只是某一列的取值
    assert "直接跑" in sec and "不要先跑诊断" in sec
    for q in QUESTIONS:
        tier_needs = next(t.needs for t in TIERS if t.level == q.tier)
        if q.tier >= 3:
            assert tier_needs in sec, f"{q.id} 属于层 {q.tier}，应提示先确认前提"
        else:
            row = [ln for ln in sec.splitlines() if f"`{q.id}`" in ln][0]
            assert "直接跑" in row and "先 `kernoscope capabilities" not in row, row


# ---------------------------------------------------------------------------
# 第三张表：缺哪一趟采集就补不回来（规则可从代码复算）
# ---------------------------------------------------------------------------
def _expected_irrecoverable() -> set[str]:
    tier_of = {q.id: q.tier for q in QUESTIONS}
    routes = {c.value: [q for q, codes in RC.FAILURE_HINTS.items() if c.value in codes]
              for c in R.RefusalCode}
    max_tier = {v: max((tier_of[q] for q in qs if q in tier_of), default=-1)
                for v, qs in routes.items()}
    out = set()
    for c in R.RefusalCode:
        rem = R.DEFAULT_REMEDIES[c]
        if rem.cost_kind == "unavailable":
            out.add(c.value)
        if rem.cost_kind == "wall_seconds" and rem.cost_s >= 60 and max_tier[c.value] >= 3:
            out.add(c.value)
    return out


def test_irrecoverable_section_lists_exactly_the_rule_derived_codes():
    sec = _section(TEXT, "## 三、缺哪一趟采集，就永久补不了")
    listed = {c.value for c in R.RefusalCode if f"`{c.value}`" in sec}
    expected = _expected_irrecoverable()
    assert listed == expected, f"表里 vs 规则算出来的：{sorted(listed)} vs {sorted(expected)}"
    assert expected, "规则至少应挑出永久拿不到的那几个 code"
    # 规则本身要写在文档里（读者要能自己复算，而不是信这张表）
    assert 'cost_kind == "unavailable"' in sec and "cost_s >= 60" in sec
    # 每一行都要点名 code（第一列就是它）
    for line in sec.splitlines():
        if line.startswith("| `"):
            assert line.split("|")[1].strip().startswith("`"), line


# ---------------------------------------------------------------------------
# "无法漂移"的证明：改代码 → 文档跟着变
# ---------------------------------------------------------------------------
def test_rendering_reads_live_code_not_a_frozen_copy(monkeypatch):
    marker = "MARKER-9f3a-来自代码改动"
    monkeypatch.setitem(R.CODE_TITLES, R.RefusalCode.NO_GPU, marker)
    monkeypatch.setitem(RC.COMMANDS, "kernel_time", f"kernoscope {marker}")
    monkeypatch.setitem(R.DEFAULT_REMEDIES, R.RefusalCode.NO_GPU,
                        R.Remedy(marker, command="pip install torch", cost_kind="manual"))
    text = GEN.render()
    assert text.count(marker) >= 3, "改 CODE_TITLES/COMMANDS/DEFAULT_REMEDIES 后文档必须跟着变"
    assert DOC.read_text(encoding="utf-8") != text, "monkeypatch 下生成的字节应与磁盘不同"


def test_preflight_refuses_to_generate_from_an_inconsistent_source(monkeypatch):
    """源表不自洽（code 没有 remedy）时宁可红，也不要生成一份看起来完整的错文档。"""
    assert GEN.preflight() == []
    monkeypatch.delitem(R.DEFAULT_REMEDIES, R.RefusalCode.NO_GPU)
    problems = GEN.preflight()
    assert any("没有 remedy" in p for p in problems), problems
    assert GEN.main([]) == 1


def test_no_fenced_bash_blocks_so_commands_stay_single_sourced():
    """命令只在表格里以内联代码出现，不是另写一段 bash。

    理由：文档命令门禁只解析 ```bash 围栏，而 route_cmd 的命令模板带 `<占位符>`；在手册里
    另抄一段可执行的 bash，就是给"手写的第二份命令"开口子（也正是这份文档要消灭的东西）。
    """
    assert "```" not in TEXT
