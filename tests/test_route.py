"""A1：`kernoscope route` —— 先路由再动手。

为什么要有它：agent 拿到的是一条**问题**（"为什么这个 kernel 慢"），而工具暴露的是**口径与命令**。
从问题到命令这一步以前只存在于文档散文里，agent 只能凭印象拼 —— 而"凭印象拼子命令"是触发
评测里最常见的失败模式。所以路由本身必须是**机器可读的真源**（代码），文档只是它的视图。
"""

from __future__ import annotations

import json

import pytest

from kernoscope.capabilities import QUESTIONS, Env
from kernoscope.refusals import DEFAULT_REMEDIES, RefusalCode
from kernoscope.route_cmd import (CALIBER_NOTE, COMMANDS, FAILURE_HINTS, ROUTES, build_parser,
                                 render_text, route)


def test_every_route_targets_a_real_question():
    """路由表不许指向不存在的问题（否则 agent 会照着一个幽灵走）。"""
    ids = {q.id for q in QUESTIONS}
    bad = [qid for _words, qid in ROUTES if qid not in ids]
    assert not bad, f"ROUTES 指向了不存在的问题：{bad}"


def test_every_failure_hint_is_a_real_refusal_code():
    """`FAILURE_HINTS` 里的每个 code 都必须存在（写错名字等于给了个假承诺）。"""
    valid = {c.value for c in RefusalCode}
    bad = sorted({c for codes in FAILURE_HINTS.values() for c in codes if c not in valid})
    assert not bad, f"FAILURE_HINTS 里有不存在的 code：{bad}"
    # 有 remedy 的 code 才能带处置；没有的至少要出现在表里（显式报错）
    for qid, codes in FAILURE_HINTS.items():
        for c in codes:
            assert c in valid and RefusalCode(c) in DEFAULT_REMEDIES or c in valid


def test_every_question_has_a_command_or_says_why_not():
    ids = {q.id for q in QUESTIONS}
    missing = sorted(ids - set(COMMANDS))
    assert not missing, f"这些问题没有命令模板：{missing}"


@pytest.mark.parametrize("question,expect", [
    ("为什么这个 kernel 慢", "kernel_time"),
    ("哪个 kernel 最耗时", "kernel_time"),
    ("DRAM 搬了多少字节", "dram_bytes"),
    ("是不是 memory bound", "dram_bytes"),
    ("occupancy 上界是多少", "theoretical_occupancy"),
    ("实际达到的占用率", "achieved_occupancy"),
    ("L2 命中率怎么样", "l2_locality"),
    ("为什么 ncu 拿不到计数器", "exact_dram"),
    ("发射槽用了多少", "issue_slot"),
    ("多卡通信瓶颈", "nvlink"),
])
def test_chinese_colloquial_questions_route(question, expect):
    r = route(question)
    assert r["matched"], question
    assert expect in [x["id"] for x in r["routes"]], (question, [x["id"] for x in r["routes"]])


def test_no_match_does_not_force_fit():
    """匹配不上就说不匹配 —— 这是路由纪律（不许硬套一个看起来像的）。"""
    r = route("帮我调一下学习率")
    assert r["matched"] is False and r["routes"] == []
    assert "不要硬套" in r["hint"]
    assert "route --list" in r["hint"]


def test_route_carries_command_caliber_and_failure_modes():
    r = route("DRAM 搬了多少字节")
    top = r["routes"][0]
    assert top["id"] == "dram_bytes"
    assert "--ks-calibration" in top["command"] and "--ks-window" in top["command"]
    assert "轨迹口径" in top["caliber"]            # 口径决策规则要跟着路由走
    codes = [f["code"] for f in top["failures"]]
    assert "share_too_low" in codes                # 会被拒的条件也要给出
    text = render_text(r)
    assert "命令：" in text and "可能被拒：" in text and "走哪条口径" in text


def test_route_json_is_machine_readable():
    r = route("为什么这个 kernel 慢")
    blob = json.dumps(r, ensure_ascii=False)
    back = json.loads(blob)
    assert back["routes"][0]["answers"] and back["matched"] is True


def test_route_reports_local_capability_when_asked():
    r = route("DRAM 搬了多少字节", env=Env(has_gpu=True, has_nvml=True, has_nsys=False))
    assert "这台机器" in r["env_note"]
    r2 = route("DRAM 搬了多少字节", env=Env(has_gpu=True, has_nvml=True, counter_state="available"))
    assert "有" in r2["env_note"] and "计数器权限" in r2["env_note"]


def test_route_list_covers_every_question(capsys):
    from kernoscope.route_cmd import route_main

    assert route_main(["--list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {x["id"] for x in data["routes"]} == {q.id for q in QUESTIONS}


def test_route_without_question_prints_help_and_exits_2():
    from kernoscope.route_cmd import route_main

    assert route_main([]) == 2


def test_route_never_measures():
    """`route` 只输出计划：它的实现里不许出现测量/采集调用（防止"路由时顺手跑一下"）。"""
    import inspect

    import kernoscope.route_cmd as m

    src = inspect.getsource(m)
    for forbidden in ("subprocess", "run_profile", "Sampler(", "sampler."):
        assert forbidden not in src, f"route 里出现了 {forbidden} —— 它必须只输出计划"
