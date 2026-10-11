"""C5：三段式一屏 + 案件目录（跨会话继续、交接）。

目标改写成"让降本变成一次便宜的、agent 能自己完成的迭代"之后，**一次迭代不再是"跑一条命令"**
—— 它要能跨会话接着做。所以证据、状态、下一步必须落在一处（案件目录），而一屏要能被
人和 agent 直接读成「结论 → 发现 → 路径」。
"""

from __future__ import annotations

import json
import pathlib

from kernoscope.case import case_dir_for, journal_record, manifest, next_markdown, write_case
import kernoscope.packet as packet_mod
from kernoscope.packet import build, next_ranked, packet_bytes, render_onepage

from test_packet import _plain_report, _report_with_all_verdicts, _rich_report


def _packet():
    p = build(_rich_report())
    packet_bytes(p)                     # cost.packet_bytes 要真实（一屏会印）
    return p


# --------------------------------------------------------------------------
# 一屏：三段式
# --------------------------------------------------------------------------
def test_onepage_is_three_sections_with_a_one_line_conclusion():
    text = render_onepage(_packet())
    assert "结论：" in text
    assert "-- 发现（判词与前提）--" in text
    assert "-- 路径（下一步，按挡住决策的程度排序）--" in text
    # 段落顺序：结论 → 发现 → 路径 → 代价
    order = [text.index("结论："), text.index("-- 发现"), text.index("-- 路径"),
             text.index("本次代价")]
    assert order == sorted(order)


def test_path_section_ranks_refusals_before_advisories():
    """路径段不能是模板话术：**被拒的判词排在最前**（它们挡住的决策最狠）。"""
    p = build(_report_with_all_verdicts())
    packet_bytes(p)
    ranked = next_ranked(p)
    kinds = [it["kind"] for it in ranked]
    assert "refused" in kinds
    assert kinds.index("refused") < len(kinds)          # 拒绝项在序列里
    first_non_refused = next((i for i, k in enumerate(kinds) if k != "refused"), len(kinds))
    assert all(k == "refused" for k in kinds[:first_non_refused]), kinds
    # 去重：同一 code / 同一提示只出现一次
    keys = [it.get("code") or it.get("hint") for it in ranked]
    assert len(keys) == len(set(keys))


def test_one_line_summary_counts_refused_verdicts():
    p = build(_report_with_all_verdicts())
    packet_bytes(p)
    text = render_onepage(p)
    refused = [v for v in p["verdicts"] if v["status"] == "refused"]
    assert f"{len(refused)}/{len(p['verdicts'])} 条判词被拒" in text


# --------------------------------------------------------------------------
# 案件目录
# --------------------------------------------------------------------------
def test_case_dir_has_the_five_files_and_relative_outputs(tmp_path):
    p = _packet()
    out = tmp_path / "rep"
    outs = {"report": str(tmp_path / "rep.ksreport.json"),
            "packet": str(tmp_path / "pk.json"), "perfetto": None}
    root = write_case(p, str(out), ["--set", "basic", "-o", str(out)], 0, outs, quiet=True)
    assert root == case_dir_for(str(out))
    for name in ("manifest.json", "env.json", "journal.jsonl", "next.md", "README.md"):
        assert (root / name).exists(), name
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    # 输出一律相对路径（相对**案件目录**；交接/入库时不泄漏别人的目录结构）
    assert man["outputs"]["report"] == "../rep.ksreport.json"
    assert not any(str(v).startswith("/") for v in man["outputs"].values())
    assert man["protocol_hash"] and man["tool_version"]
    assert man["verdicts"] and "next_actions" in man


def test_journal_appends_and_keeps_history(tmp_path):
    p = _packet()
    out = tmp_path / "rep"
    write_case(p, str(out), ["first"], 0, {}, quiet=True)
    write_case(p, str(out), ["second"], 2, {}, quiet=True)
    lines = (case_dir_for(str(out)) / "journal.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "journal 必须追加，不许覆盖历史"
    first, second = (json.loads(x) for x in lines)
    assert first["command"] == ["first"] and first["rc"] == 0
    assert second["command"] == ["second"] and second["rc"] == 2


def test_next_md_is_resumable_and_says_how_to_compare():
    text = next_markdown(_packet())
    assert text.startswith("# 下一步")
    assert "kernoscope diff" in text          # 改完之后怎么判断"真的好了"
    assert "MDD" in text


def test_manifest_records_input_identities():
    """交接时必须能确认"用的是同一份输入"：标定/轨迹指纹在协议元里，直接带出来。"""
    p = _packet()
    p.setdefault("protocol", {}).update({
        "calibration": {"name": "machine-profile.json", "sha256": "abc123"},
        "memtrace": {"name": "trace.txt", "sha256": "def456"},
    })
    man = manifest(p, "rep", ["cmd"], {}, pathlib.Path("/tmp"))
    assert man["protocol_inputs"]["calibration"]["sha256"] == "abc123"
    assert man["protocol_inputs"]["memtrace"]["name"] == "trace.txt"
