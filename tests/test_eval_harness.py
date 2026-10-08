"""W8：外部 agent 冷启动评测台。

守五条不变量：
  1. **能跑**：两个内置 agent 在无 GPU、无网络的机器上端到端跑通（harness 真起子进程）；
  2. **检测器有效**：故意幼稚的 agent（忽略 status、挑最大数字）误用率 > 0；
  3. **规范做法是可达的**：只依据 status/premises 的 agent 误用率 0、准确率 1.0；
  4. **任务自洽**：每个任务都有真值与 forbidden_actions，且至少一个任务的正确答案是
     "不要动代码"（前提不成立 → 先改测量口径）；
  5. **材料与包合法**：任务里内联的决策包能被产品代码读（refused 必带 refusal code），
     且材料里不准混进真值。

外加：agent 崩溃/超时/不写 decision.json 必须**如实记为失败**（准确率算它错），
以及误用检测器对"被裁掉的判词"也生效 —— 这几条是评测台不许"看起来在评测"的地方。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
TASKS_DIR = EVAL / "tasks"
AGENTS = EVAL / "agents"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_harness():
    spec = importlib.util.spec_from_file_location("ks_eval_harness", EVAL / "harness.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ks_eval_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness()


def _task_files() -> list[Path]:
    return sorted(TASKS_DIR.glob("*.json"))


def _run_harness(argv: list[str], timeout: float = 300.0):
    proc = subprocess.run([sys.executable, str(EVAL / "harness.py"), *argv],
                          capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    board = None
    out = _arg(argv, "--out")
    if out and Path(out, "scoreboard.json").exists():
        board = json.loads(Path(out, "scoreboard.json").read_text(encoding="utf-8"))
    return proc.returncode, board, proc.stdout + proc.stderr


def _arg(argv: list[str], name: str) -> str | None:
    if name in argv:
        return argv[argv.index(name) + 1]
    return None


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """两个内置 agent 各跑一次完整评测台（子进程），后续断言都复用这两份结果。"""
    base = tmp_path_factory.mktemp("w8-eval")
    out: dict[str, dict] = {}
    for agent in ("rule_baseline.py", "packet_reader.py"):
        run_dir = base / agent.replace(".py", "")
        rc, board, log = _run_harness(
            ["--agent-cmd", f"{sys.executable} eval/agents/{agent}", "--out", str(run_dir)])
        out[agent] = {"rc": rc, "board": board, "log": log, "out": run_dir}
    return out


# --------------------------------------------------------------------------
# ① 端到端跑通
# --------------------------------------------------------------------------
@pytest.mark.parametrize("agent", ["rule_baseline.py", "packet_reader.py"])
def test_harness_runs_builtin_agent_end_to_end(agent, runs):
    r = runs[agent]
    assert r["rc"] == 0, f"harness 退出码 {r['rc']}：\n{r['log']}"
    board = r["board"]
    assert board is not None, "必须写出 scoreboard.json"
    totals = board["totals"]
    assert totals["tasks"] == len(_task_files()) >= 4
    assert totals["scored_tasks"] == totals["tasks"], \
        f"有任务没跑出 decision.json：{totals['failed_task_ids']}"
    assert totals["failed_tasks"] == 0

    # summary.md 是人可读的，且必须写出两个关键数字
    summary = (r["out"] / "summary.md").read_text(encoding="utf-8")
    assert "误用率" in summary and "decision_accuracy" in summary

    # 每个任务的材料目录都落地了 packet.json，且真值没有泄漏
    for task in _task_files():
        tid = task.stem
        material = r["out"] / tid / "material"
        assert (material / "packet.json").exists(), f"{tid}: 材料里必须有 packet.json"
        assert (material / "QUESTION.md").exists()
        blob = "\n".join(p.read_text(encoding="utf-8")
                         for p in material.iterdir() if p.is_file())
        for leak in ("safe_verdicts", "forbidden_verdicts", "expected_action_keywords"):
            assert leak not in blob, f"{tid}: 材料里混进了真值字段 {leak}"
        assert (material / "decision.json").exists(), f"{tid}: agent 应写出 decision.json"


def test_material_dir_starts_clean(tmp_path):
    """上次运行的 decision.json 不许被继承（否则"没写"会被误当成"写了"）。"""
    out = tmp_path / "rerun"
    rc, board, log = _run_harness(
        ["--agent-cmd", f"{sys.executable} eval/agents/packet_reader.py",
         "--task", "01", "--json", "--out", str(out)])
    assert rc == 0 and board["totals"]["scored_tasks"] == 1, log
    assert '"decision_accuracy"' in log, "--json 要把 scoreboard 打到 stdout"
    stale = out / "01_saturated_copy_dram_bound" / "material" / "decision.json"
    assert stale.exists()

    rc2, board2, _log2 = _run_harness(
        ["--agent-cmd", f"{sys.executable} -c pass", "--task", "01", "--out", str(out)])
    assert rc2 == 1
    assert board2["totals"]["failed_tasks"] == 1, "上一轮的 decision.json 被继承了"
    assert board2["totals"]["failure_kinds"] == {"failed_no_decision": 1}


def test_task5_material_includes_capabilities(runs):
    """材料支持多份可见内容（能力清单）—— 被测 agent 该看什么就只给什么。"""
    cap = runs["packet_reader.py"]["out"] / "05_dram_caliber_unusable" / "material"
    assert (cap / "capabilities.json").exists()
    doc = json.loads((cap / "capabilities.json").read_text(encoding="utf-8"))
    assert doc["tool"] == "kernoscope" and doc["tier"]["level"] == 2


# --------------------------------------------------------------------------
# ② 检测器有效：幼稚 agent 的误用率必须 > 0
# --------------------------------------------------------------------------
def test_naive_agent_misuse_is_detected(runs):
    totals = runs["rule_baseline.py"]["board"]["totals"]
    assert totals["misuse_events"] > 0, "幼稚 agent 必须被算出误用，否则检测器是坏的"
    assert totals["misuse_rate"] is not None and totals["misuse_rate"] > 0
    assert totals["tasks_with_misuse"] >= 1
    assert totals["decision_accuracy"] < 1.0, "'挑最大数字'不可能样样都对"


def test_naive_agent_is_flagged_for_forbidden_basis_and_actions(runs):
    """幼稚 agent（只看数字大小）必须被真值禁止依据与禁止动作两条路抓到。

    「refused / 前提不成立」这两类误用的检测由本文件里的合成负样本测试负责
    （`test_misuse_detector_catches_*`），不再依赖幼稚 agent —— 因为产品修掉了一个
    真实漏洞之后，refused 判词不再带 value，幼稚 agent 也就"挑不到"它了：
    那个漏洞是「前提不成立的判词 status 仍然是 soft」，现在一律 refused。
    """
    tasks = {r["task_id"]: r for r in runs["rule_baseline.py"]["board"]["tasks"]}
    reasons = [u["reason"] for r in tasks.values() for u in r["unsafe"]]
    assert any("被真值列为禁止依据" in x for x in reasons), reasons
    assert any(u["kind"] == "action" for r in tasks.values() for u in r["unsafe"]), \
        "行动层面的禁止动作（如'转向优化计算'）也必须被记下来"


# --------------------------------------------------------------------------
# ③ 规范做法：误用率 0、准确率 1.0
# --------------------------------------------------------------------------
def test_canonical_agent_is_clean(runs):
    board = runs["packet_reader.py"]["board"]
    totals = board["totals"]
    assert totals["misuse_events"] == 0, \
        [u for r in board["tasks"] for u in r["unsafe"]]
    assert totals["misuse_rate"] == 0.0
    assert totals["decision_accuracy"] == 1.0, \
        [(r["task_id"], r["classification"], r["expected_classification"])
         for r in board["tasks"] if not r["classification_correct"]]
    assert totals["unsafe_events"] == 0, \
        "规范 agent 的 next_action 也不该命中禁止动作"


def test_canonical_agent_never_acts_on_refused_or_failed(runs):
    """逐条检查：acted_on 里不许出现 refused / 前提 ok=false / 包外判词。"""
    board = runs["packet_reader.py"]["board"]
    for r in board["tasks"]:
        states = {v["id"]: v for v in r["verdict_states"]}
        for vid in r["acted_on"]:
            v = states.get(vid)
            assert v is not None, f"{r['task_id']}: acted_on 里有包外判词 {vid}"
            assert v["status"] != "refused"
            assert not v["premises_failed"], f"{r['task_id']}: {vid} 前提不成立却进了 acted_on"
        # 规范做法会把被拒的判词放进 blocked_by（说明"我没拿它当依据"），而不是丢掉
        refused = [v["id"] for v in r["verdict_states"] if v["status"] == "refused"]
        for vid in refused:
            assert vid in r["blocked_by"], \
                f"{r['task_id']}: 被拒的 {vid} 应在 blocked_by 里"
        for vid in r["blocked_by"]:
            assert vid in states, f"{r['task_id']}: blocked_by 里有包外判词 {vid}"
    ids = [r["task_id"] for r in board["tasks"]]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# ④ 任务自洽：真值 + forbidden_actions + 至少一个"不要动代码"
# --------------------------------------------------------------------------
def test_every_task_declares_truth_and_forbidden_actions():
    tasks = [harness.load_task(p) for p in _task_files()]
    assert len(tasks) >= 4
    for t in tasks:
        assert t.get("id") and t.get("scenario") and t.get("question_for_agent")
        assert t.get("material", {}).get("packet"), f"{t['id']}: 任务必须内联决策包"
        truth = t.get("truth") or {}
        for k in ("classification", "safe_verdicts", "forbidden_verdicts",
                  "expected_action_keywords"):
            assert k in truth, f"{t['id']}: truth 缺 {k}"
        assert truth["safe_verdicts"] and truth["expected_action_keywords"]
        assert isinstance(t.get("forbidden_actions"), list) and t["forbidden_actions"], \
            f"{t['id']}: 必须声明 forbidden_actions"


def test_at_least_one_task_requires_fixing_the_measurement_not_code():
    """三类高不确定性场景里，必须有一个的正确答案是"不要动代码"。"""
    tasks = [harness.load_task(p) for p in _task_files()]
    no_change = [t for t in tasks if not t["truth"].get("code_change_allowed", True)]
    assert no_change, "至少要有一个任务的正确答案是'前提不成立 → 先改测量口径'"
    kinds = {t["truth"]["classification"] for t in tasks}
    assert {"dram_bound", "window_unattributable", "not_dram_bound", "compute_bound"} <= kinds
    # 前提不成立的归因任务：禁止依据里必须有 per-launch 那条
    attrib = [t for t in tasks if t["truth"]["classification"] == "window_unattributable"]
    assert attrib and any("dram_bytes_per_launch" in t["truth"]["forbidden_verdicts"]
                          for t in attrib)


# --------------------------------------------------------------------------
# ⑤ 包与材料合法
# --------------------------------------------------------------------------
def test_task_packets_are_structurally_valid():
    from kernoscope.packet import refusal_index, render_onepage
    from kernoscope.refusals import RefusalCode

    tasks = [harness.load_task(p) for p in _task_files()]
    assert harness.validate_all(tasks) == []
    codes = {c.value for c in RefusalCode}
    for t in tasks:
        packet = t["material"]["packet"]
        render_onepage(packet)                       # 渲染不炸
        index = refusal_index(packet)
        assert packet["schema_version"] == 1 and packet["tool"] == "kernoscope"
        assert packet["cost"]["trimmed"] is not None
        for v in packet["verdicts"]:
            assert v["status"] in ("ok", "soft", "refused")
            if v["status"] == "refused":
                code = v["refusal"]["code"]
                assert code in codes, f"{t['id']}: {v['id']} 的 code 不在拒绝闭集里"
                assert code in index, f"{t['id']}: {v['id']} 的 code 不在 refusals 表里"
                assert "value" not in v, "refused 判词不许带 value"


def test_validator_rejects_packets_that_break_the_contract():
    task = harness.load_task(_task_files()[0])
    broken = json.loads(json.dumps(task))
    v = broken["material"]["packet"]["verdicts"][0]
    v["status"] = "refused"                          # 有 refused 却没带 refusal code
    v.pop("refusal", None)
    errs = harness.validate_task(broken)
    assert any("refusal code" in e for e in errs), errs

    leaked = json.loads(json.dumps(task))
    leaked["material"]["files"] = {"notes.md": "truth: safe_verdicts 是 dram_bytes"}
    assert any("真值" in e for e in harness.validate_task(leaked))

    bogus = json.loads(json.dumps(task))
    bogus["material"]["packet"]["verdicts"][0]["id"] = "not_in_truth"
    bogus["truth"]["safe_verdicts"] = ["ghost"]
    assert any("不在包里" in e for e in harness.validate_task(bogus))


# --------------------------------------------------------------------------
# 失败必须如实记录（不许静默跳过）
# --------------------------------------------------------------------------
def test_agent_that_writes_nothing_is_recorded_as_failure(tmp_path):
    out = tmp_path / "noop"
    rc, board, log = _run_harness(
        ["--agent-cmd", f"{sys.executable} -c pass", "--task", "01", "--out", str(out)])
    assert rc == 1, "有任务失败时 harness 必须给出非零退出码"
    totals = board["totals"]
    assert totals["failed_tasks"] == 1
    assert totals["failed_task_ids"] == ["01_saturated_copy_dram_bound"]
    assert totals["failure_kinds"] == {"failed_no_decision": 1}
    assert totals["decision_accuracy"] == 0.0
    assert totals["scored_tasks"] == 0
    assert board["tasks"][0]["scored"] is False
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "failed_no_decision" in summary


def test_agent_timeout_is_recorded_as_failure(tmp_path):
    out = tmp_path / "slow"
    code = "import time; time.sleep(30)"
    rc, board, log = _run_harness(
        ["--agent-cmd", f'{sys.executable} -c "{code}"', "--task", "01",
         "--timeout", "1", "--out", str(out)], timeout=120)
    assert rc == 1
    assert board["totals"]["failure_kinds"] == {"failed_timeout": 1}


def test_crashing_agent_is_recorded_as_failure(tmp_path):
    out = tmp_path / "boom"
    rc, board, _log = _run_harness(
        ["--agent-cmd", f'{sys.executable} -c "import sys; sys.exit(3)"',
         "--task", "01", "--out", str(out)])
    assert rc == 1
    assert board["totals"]["failure_kinds"] == {"failed_crash": 1}
    assert board["tasks"][0]["run"]["rc"] == 3


# --------------------------------------------------------------------------
# 误用检测器本身：refused / 前提不成立 / 被裁掉 / 真值禁止 都要能抓到
# --------------------------------------------------------------------------
def _fake_task() -> dict:
    packet = {
        "schema_version": 1, "tool": "kernoscope",
        "verdicts": [
            {"id": "dram_bytes", "status": "soft",
             "premises": [{"id": "window_coverage_pct", "observed": 97.1,
                           "threshold": 50.0, "ok": True}]},
            {"id": "kernel_time", "status": "ok"},
            {"id": "dram_bytes_per_launch", "status": "refused",
             "refusal": {"code": "below_resolution"}},
        ],
        "refusals": {"below_resolution": {"action": "延长窗口", "cost_kind": "wall_seconds"}},
        "cost": {"trimmed": ["verdicts"]},
    }
    return {
        "id": "fake", "scenario": "s", "question_for_agent": "q",
        "material": {"packet": packet},
        "truth": {"classification": "dram_bound", "safe_verdicts": ["dram_bytes"],
                  "forbidden_verdicts": ["kernel_time"],
                  "expected_action_keywords": ["访存"]},
        "forbidden_actions": ["优化计算"],
    }


def test_misuse_detector_catches_refused_failed_trimmed_and_forbidden():
    task = _fake_task()
    decision = {"classification": "dram_bound",
                "acted_on": ["dram_bytes", "kernel_time", "dram_bytes_per_launch", "ghost"],
                "next_action": "先优化计算，把这个数字降下来"}
    r = harness.score_task(task, decision, None, {"rc": 0, "wall_s": 0.01})
    assert r["misuse_events"] == 3, r["unsafe"]
    assert r["misuse_rate"] == 0.75
    joined = " | ".join(u["reason"] for u in r["unsafe"])
    assert "refused" in joined and "被真值列为禁止依据" in joined and "不在包里" in joined
    assert r["forbidden_action_hits"] == ["优化计算"]
    assert r["unsafe_events"] == 4
    assert r["classification_correct"] is True


def test_misuse_detector_catches_failed_premise_and_contradiction():
    task = _fake_task()
    task["truth"]["forbidden_verdicts"] = []
    task["material"]["packet"]["verdicts"][0]["premises"][0]["ok"] = False
    decision = {"classification": "dram_bound", "acted_on": ["dram_bytes"],
                "blocked_by": ["dram_bytes"], "next_action": "提高访存效率"}
    r = harness.score_task(task, decision, None, {"rc": 0, "wall_s": 0.01})
    reasons = " | ".join(u["reason"] for u in r["unsafe"])
    assert "前提不成立" in reasons and "自相矛盾" in reasons
    assert r["misuse_events"] == 1, "同一条判词不该被算两次误用"
    assert [u["kind"] for u in r["unsafe"]] == ["verdict", "decision"]


def test_score_task_marks_failure_without_decision():
    task = _fake_task()
    r = harness.score_task(task, None, harness.ST_NO_DECISION, {"rc": 0, "wall_s": 0.0})
    assert r["scored"] is False and r["classification_correct"] is False
    assert r["misuse_rate"] is None
    assert r["acted_on_total"] == 0


def test_decision_contract_violations_are_failures(tmp_path):
    """契约缺项/坏 JSON 都算失败，但保留读到的结构以便诊断。"""
    d = tmp_path / "decision.json"
    d.write_text("not json", encoding="utf-8")
    assert harness.load_decision(d) == (None, harness.ST_BAD_JSON)

    d.write_text('{"classification": "x", "acted_on": []}', encoding="utf-8")
    dec, state = harness.load_decision(d)
    assert state == harness.ST_BAD_DECISION and dec == {"classification": "x", "acted_on": []}
    assert harness.score_task(_fake_task(), dec, state,
                              {"rc": 0})["scored"] is False

    d.write_text('{"classification": "", "acted_on": [], "next_action": "y"}', encoding="utf-8")
    assert harness.load_decision(d)[1] == harness.ST_BAD_DECISION

    d.write_text('{"classification": "x", "acted_on": [], "next_action": "y"}', encoding="utf-8")
    dec, state = harness.load_decision(d)
    assert state is None and dec["classification"] == "x"


def test_docs_and_agents_are_runnable_offline():
    for p in (EVAL / "README.md", EVAL / "harness.py", AGENTS / "rule_baseline.py",
              AGENTS / "packet_reader.py", EVAL / "make_tasks.py"):
        assert p.exists(), f"缺交付物：{p}"
    text = (EVAL / "README.md").read_text(encoding="utf-8")
    assert "decision.json" in text and "误用率" in text


def test_forbidden_action_matching_ignores_negation():
    """说「不要据此优化计算」不该被算成命中了禁止动作「优化计算」（外部评测假阳性）。"""
    _mentions = harness._mentions

    assert _mentions("先修测量口径，不要据此去优化计算", "优化计算") is False
    assert _mentions("下一步：优化计算侧的指令混合", "优化计算") is True
    assert _mentions("avoid compute-bound tuning", "compute") is False
    assert _mentions("go compute side now", "compute") is True
