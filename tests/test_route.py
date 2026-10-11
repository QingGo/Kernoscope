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
from kernoscope.route_cmd import (CALIBER_NOTE, COMMANDS, DRAM_BYTES_REFUSALS, FAILURE_HINTS,
                                 ROUTES, build_parser, dram_bytes_refusal_codes, render_text,
                                 route)


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


# --------------------------------------------------------------------------
# V-29：`dram_bytes` 的拒绝条件必须**从真源派生**，且**全部**渲染
# --------------------------------------------------------------------------
def test_dram_failure_list_is_derived_not_handwritten():
    """表里那条 `dram_bytes` 就是 `analyze/dram.py` 现算的结果，不是另一份手抄。

    修前它只有 4 条（`share_too_low/coverage_too_low/below_resolution/no_calibration`），
    而这条路实际会报出 9 条 —— 验收时按它给的命令跑，首先撞到的 `no_nvml_samples`
    恰好不在列表里。
    """
    derived = dram_bytes_refusal_codes()
    assert len(derived) == len(set(derived)) and derived, derived
    assert set(derived) == set(FAILURE_HINTS["dram_bytes"]) == set(DRAM_BYTES_REFUSALS)
    # 报告里 agent 首先撞到的那 5 个（修前全部缺失）必须都在
    for code in ("no_nvml_samples", "share_unknown", "multi_kernel_window",
                 "window_launch_unaligned", "trace_unavailable"):
        assert code in derived, f"{code} 没被派生出来：{derived}"


def test_derived_set_covers_every_code_dram_can_actually_emit():
    """派生集合 ⊇ 实际可达集合：用真的 `DramEstimate` 把拒绝构造点跑一遍逐个对。

    这是"同源"的判据：派生不是靠读文档，而是拿真对象把 `_total_refusals()` /
    `_attribution_refusals()` / `_trace_refusal()` 的输出收上来，看有没有漏。
    """
    from kernoscope.analyze.dram import estimate

    from test_dram import _samples, _uniform_run

    samples40, spans40 = _uniform_run(seconds=40.0, launches=200)
    scenarios = [
        estimate([], peak_gbs=0.0, window_s=2.0, launches=0),
        estimate(_samples(10, mem=0.0, sm=99.0), peak_gbs=900.0, window_s=2.0, launches=10),
        estimate(_samples(10, mem=6.3, sm=0.0), peak_gbs=900.0, window_s=2.0,
                 window_mode="full"),
        estimate(_samples(10), peak_gbs=900.0, window_s=2.0, launches=100),
        estimate(samples40, peak_gbs=900.0, window_s=40.0, launches=200, window_mode="full",
                 launch_spans=spans40, matched_kernel_ns=int(16.0e9),
                 other_kernels_ns=int(24.0e9), other_kernel_count=3),
        estimate(_samples(200, dt=0.2), peak_gbs=900.0, window_s=40.0, launches=200,
                 window_mode="full", window_limit_s=20.0, matched_kernel_ns=int(20.0e9)),
    ]
    reachable: set[str] = set()
    for est in scenarios:
        for r in (*est._total_refusals(), *est._attribution_refusals(), est._trace_refusal()):
            reachable.add(r.code.value)
    derived = set(dram_bytes_refusal_codes())
    assert reachable, "场景没造对：一个拒绝都没触发，这条测试就白测了"
    assert reachable <= derived, f"实际可达但没登记：{sorted(reachable - derived)}"
    # 反向也要有信息量：派生集合不许是"把所有 code 都抄一遍"
    assert derived - reachable == set(), f"派生里有没有可达来源的 code：{derived - reachable}"


def test_route_text_renders_every_possible_refusal_not_just_the_first_three():
    """文本渲染不许截断：`--json` 里有几条，人视图里就要有几条（修前只印前 3 条）。"""
    r = route("DRAM 搬了多少字节")
    top = r["routes"][0]
    codes = [f["code"] for f in top["failures"]]
    assert len(codes) > 3, codes
    text = render_text(r)
    rendered = [ln for ln in text.splitlines() if "可能被拒：" in ln]
    assert len(rendered) == len(codes), (len(rendered), len(codes))
    for code in codes:
        assert f"可能被拒：{code}" in text, code


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
