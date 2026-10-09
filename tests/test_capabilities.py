"""W1/W5：能力探测分层 + 机器可读能力清单 + 三态覆盖矩阵。

关键不变量：
  1. 每个问题都有判定函数（漏登记立刻炸，不允许"文档里写了但代码不判"）；
  2. 每个 refused 的问题都带结构化拒绝 + 补救办法（"必然被拒的问题 + 原因 + 替代"）；
  3. 人视图与 --json 同一模型（人视图里的问题 id、三态必须与 JSON 一致）；
  4. 层次由探测结果决定，不由用户选择 —— 缺件只会降级，不会假装可用。
"""

from __future__ import annotations

import json

import pytest

from kernoscope.capabilities import (CHECKS, QUESTIONS, TIERS, Env, evaluate, probe,
                                     render_matrix)
from kernoscope.refusals import RefusalCode


def _env(**kw) -> Env:
    base = dict(has_gpu=True, gpu_name="RTX 4090", has_nvml=True, has_nsys=True,
                has_cuobjdump=True, has_ncu=True, counter_state="denied",
                has_nvbit=True, has_calibration=True, has_workload=True)
    base.update(kw)
    return Env(**base)


def test_every_question_has_a_check():
    assert {q.id for q in QUESTIONS} == set(CHECKS)


def test_nvbit_installed_is_tier_3_even_before_collecting_a_trace():
    """层是"这台机器能做什么"：装了 NVBit 就该是 tier 3，"要先采一条轨迹"是代价问题。"""
    r = evaluate(_env(has_memtrace=False, has_instr_hist=False))
    assert r.tier == 3
    a = [x for x in r.answers if x.question.id == "dram_bytes_trace"][0]
    assert a.state == "refused"
    assert a.refusal.code is RefusalCode.NO_MEMTRACE
    assert a.refusal.remedy_or_default.cost_s > 0    # 代价必须写出来（采轨迹不免费）
    nxt = [d for d in r.next_actions() if d["code"] == "no_memtrace"]
    assert nxt and nxt[0]["cost_s"] > 0


def test_full_environment_reaches_tier_3_and_answers_core_questions():
    r = evaluate(_env(has_memtrace=True, has_instr_hist=True, has_occupancy_probe=True))
    assert r.tier == 3
    m = r.matrix()
    for qid in ("kernel_time", "launch_configs", "device_busy", "dram_bytes",
                "achieved_occupancy", "dram_bytes_trace", "issue_slot"):
        assert m[qid] == "available", qid
    # 计数器级问题在 tier 3 仍然被拒（这就是诚实：它们与 ncu 同源）
    assert m["stall_breakdown"] == "refused"
    assert m["exact_dram"] == "refused"


def test_counters_available_unlocks_tier_4():
    r = evaluate(_env(counter_state="available"))
    assert r.tier == 4
    assert r.next_tier is None
    assert r.matrix()["exact_dram"] == "available"


def test_no_gpu_is_tier_0_and_everything_refused_with_remedy():
    r = evaluate(Env(has_gpu=False))
    assert r.tier == 0
    for a in r.answers:
        if a.state == "refused":
            assert a.refusal is not None
            assert a.refusal.remedy_or_default.action
    codes = {a.refusal.code for a in r.answers if a.refusal is not None}
    assert RefusalCode.NO_GPU in codes
    assert RefusalCode.NO_NVLINK in codes


def test_gpu_without_nsys_degrades_to_device_level():
    """W1 的核心场景：有 GPU 没 nsys —— 设备级仍然可用，kernel 级被拒并给出补救。"""
    r = evaluate(_env(has_nsys=False, has_nvbit=False))
    assert r.tier == 1
    m = r.matrix()
    assert m["device_busy"] == "available"
    assert m["kernel_time"] == "refused"
    a = [x for x in r.answers if x.question.id == "kernel_time"][0]
    assert a.refusal is not None
    assert a.refusal.code is RefusalCode.NSYS_MISSING
    assert "nsys" in a.refusal.remedy_or_default.action.lower() or \
           "Nsight" in a.refusal.remedy_or_default.action


def test_nvml_off_by_user_has_its_own_code():
    r = evaluate(_env(nvml_disabled_by_user=True))
    a = [x for x in r.answers if x.question.id == "device_busy"][0]
    assert a.refusal is not None and a.refusal.code is RefusalCode.NO_NVML_SAMPLES


def test_dram_bytes_is_degraded_without_calibration():
    """能答但更弱 —— 三态的意义就在这里：不把「口径未就绪」说成「能答」。"""
    r = evaluate(_env(has_calibration=False))
    assert r.matrix()["dram_bytes"] == "degraded"


def test_missing_counters_distinguishes_not_installed_from_denied():
    a = evaluate(_env(counter_state="not-installed"))
    b = evaluate(_env(counter_state="denied"))
    ca = [x for x in a.answers if x.question.id == "exact_dram"][0].refusal.code
    cb = [x for x in b.answers if x.question.id == "exact_dram"][0].refusal.code
    assert ca is RefusalCode.COUNTERS_NOT_INSTALLED
    assert cb is RefusalCode.NO_COUNTERS


def test_next_actions_are_grouped_and_ranked():
    r = evaluate(Env(has_gpu=True, has_nvml=True, has_nsys=False))
    na = r.next_actions()
    assert na, "缺 nsys 时必须有下一步建议"
    # 同一动作不重复出现
    keys = [(d["code"], d["action"]) for d in na]
    assert len(keys) == len(set(keys))
    # 装 nsys 会一次解锁多个 kernel 级问题
    nsys = [d for d in na if d["code"] == RefusalCode.NSYS_MISSING.value][0]
    assert len(nsys["unlocks"]) >= 3


def test_json_and_human_view_share_one_model():
    r = evaluate(_env())
    d = r.to_dict()
    text = r.render()
    assert d["matrix"] == r.matrix()
    for a in r.answers:
        assert a.question.id in d["matrix"]
        assert a.question.id in text, f"{a.question.id} 没出现在人视图里"
    # 三态标记必须一致（能/弱/不答 对应 available/degraded/refused）
    marks = {"available": "[ yes]", "degraded": "[weak]", "refused": "[  no]"}
    for a in r.answers:
        assert f"{marks[a.state]} {a.question.id}" in text


def test_schema_and_version_are_pinned():
    d = evaluate(_env()).to_dict()
    assert d["schema_version"] == 1
    assert d["tool"] == "kernoscope"
    assert isinstance(d["tier"]["level"], int)
    assert d["tier"]["name"] in {t.name for t in TIERS}


def test_refusal_list_matches_refused_answers():
    r = evaluate(Env(has_gpu=False))
    assert len(r.refusals()) == sum(1 for a in r.answers if a.state == "refused")


def test_matrix_render_is_markdown_and_covers_all_questions():
    text = render_matrix(evaluate(_env()))
    for q in QUESTIONS:
        assert f"`{q.id}`" in text
    assert text.startswith("| 问题 |")


def test_capabilities_cli_json_end_to_end(capsys):
    from kernoscope.cli import main

    rc = main(["capabilities", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    d = json.loads(out)
    assert d["schema_version"] == 1
    assert set(d["matrix"]) == {q.id for q in QUESTIONS}
    assert isinstance(d["next_actions"], list)


def test_capabilities_cli_human_view(capsys):
    from kernoscope.cli import main

    rc = main(["capabilities"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "能力自检" in out
    assert "能用它回答的问题" in out


def test_probe_runs_on_this_machine(capsys):
    """真机（可能没有 GPU）冒烟：探测本身必须永远成功。"""
    r = probe(probe_counters=False)
    assert 0 <= r.tier <= 4
    assert r.to_dict()["env"]["has_gpu"] in (True, False)
    assert r.render()


def test_env_from_doctor_reads_facts(monkeypatch):
    from kernoscope.capabilities import env_from_doctor
    from kernoscope.doctor import DoctorReport

    doc = DoctorReport()
    doc.tools = {"nsys": {"found": True}, "nvidia-smi": {"found": True},
                 "cuobjdump": {"found": False}, "ncu": {"found": True}}
    doc.device = {"name": "RTX 4090"}
    doc.counter_state = "denied"
    doc.nvbit_path = "/opt/nvbit"
    env = env_from_doctor(doc)
    assert env.has_gpu and env.has_nvml and env.has_nsys and env.has_nvbit
    assert not env.has_cuobjdump and env.has_ncu
    assert env.counter_state == "denied"


def test_env_from_doctor_rejects_typo_fields():
    from kernoscope.capabilities import env_from_doctor
    from kernoscope.doctor import DoctorReport

    with pytest.raises(KeyError):
        env_from_doctor(DoctorReport(), has_nsyss=True)


def test_tiers_are_contiguous_and_described():
    for i, t in enumerate(TIERS):
        assert t.level == i
        assert t.name and t.needs and t.gives


# --------------------------------------------------------------------------
# 渐进披露的入口层：--brief
# --------------------------------------------------------------------------
def test_brief_is_an_entry_layer_not_a_manual():
    """入口层必须小：完整 --json 是参考手册（本机实测 ~19-26 KB），brief 要 < 1.2 KB。"""
    import json

    r = evaluate(_env(has_memtrace=True, has_instr_hist=True))
    brief = json.dumps(r.brief(), ensure_ascii=False, separators=(",", ":"))
    full = json.dumps(r.to_dict(), ensure_ascii=False, separators=(",", ":"))
    assert len(brief.encode("utf-8")) <= 1200, len(brief.encode("utf-8"))
    assert len(brief.encode("utf-8")) * 8 < len(full.encode("utf-8"))


def test_brief_partitions_every_question_exactly_once():
    """入口层不许漏问题，也不许一个问题上两个列表。"""
    r = evaluate(_env())
    b = r.brief()
    listed = b["can_answer"] + b["weak"] + b["cannot_answer"]
    assert sorted(listed) == sorted(q.id for q in QUESTIONS)
    assert len(listed) == len(set(listed))


def test_brief_agrees_with_full_json_states():
    r = evaluate(_env(has_nvbit=False))
    b = r.brief()
    full = r.matrix()
    for qid in b["can_answer"]:
        assert full[qid] == "available"
    for qid in b["weak"]:
        assert full[qid] == "degraded"
    for qid in b["cannot_answer"]:
        assert full[qid] == "refused"


def test_brief_has_no_prose_fields():
    """入口层只有 id / code / 计数：解释与 remedy 全文属于参考层（--json）。"""
    b = evaluate(Env(has_gpu=False)).brief()
    blob = json.dumps(b, ensure_ascii=False)
    for banned in ("message", "observed", "threshold", "action"):
        assert banned not in blob, banned
    assert b["next"], "缺件时必须给出下一步"
    for item in b["next"]:
        assert set(item) <= {"code", "unlocks", "cost_kind", "cost_s", "cmd"}


def test_brief_next_is_ranked_and_capped():
    r = evaluate(Env(has_gpu=True, has_nvml=True, has_nsys=False))
    b = r.brief()
    assert len(b["next"]) <= 4
    unlocks = [i["unlocks"] for i in b["next"]]
    assert unlocks == sorted(unlocks, reverse=True)


def test_capabilities_brief_cli(capsys):
    from kernoscope.cli import main

    rc = main(["capabilities", "--brief"])
    out = capsys.readouterr().out
    assert rc == 0
    d = json.loads(out)                      # stdout 必须是纯 JSON（一行）
    assert "can_answer" in d and "cannot_answer" in d
    assert len(out.encode("utf-8")) <= 1300


def test_capabilities_brief_can_still_write_the_full_json(capsys, tmp_path):
    from kernoscope.cli import main

    out_path = tmp_path / "cap.json"
    rc = main(["capabilities", "--brief", "--out", str(out_path), "--quiet"])
    assert rc == 0
    full = json.loads(out_path.read_text())
    assert "answers" in full and "refusals" in full     # 参考层照常落盘
    captured = capsys.readouterr()
    json.loads(captured.out)                            # stdout 仍是纯 JSON
