"""C5：三段式一屏 + 案件目录（跨会话继续、交接）。

目标改写成"让降本变成一次便宜的、agent 能自己完成的迭代"之后，**一次迭代不再是"跑一条命令"**
—— 它要能跨会话接着做。所以证据、状态、下一步必须落在一处（案件目录），而一屏要能被
人和 agent 直接读成「结论 → 发现 → 路径」。
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from kernoscope.case import (case_dir_for, journal_record, manifest, next_markdown,
                            sanitize_argv, write_case)
import kernoscope.packet as packet_mod
from kernoscope.packet import build, next_ranked, packet_bytes, render_onepage

from test_packet import _plain_report, _report_with_all_verdicts, _rich_report

ARTIFACTS = pathlib.Path(__file__).resolve().parents[1] / "bench" / "artifacts"
CALIB_A = ARTIFACTS / "20261009-012257" / "machine-profile.json"


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


def test_case_readme_stays_within_its_documented_budget(tmp_path):
    """案件目录的 `README.md` 是**每个案件都写一份**的入口：docstring 承诺 ≤ 1 KB。

    它曾经 1245 B（承诺已经破了，且没人守）—— 这条测试把承诺变成可执行的预算。
    """
    from kernoscope.case import _CASE_README

    assert len(_CASE_README.encode("utf-8")) <= 1024, len(_CASE_README.encode("utf-8"))
    root = write_case(_packet(), str(tmp_path / "rep"), ["cmd"], 0, {}, quiet=True)
    assert (root / "README.md").read_text(encoding="utf-8") == _CASE_README


def test_manifest_next_actions_and_next_md_count_the_same_items(tmp_path):
    """V-26②：`next_actions` 与 `next.md` 必须**同源** —— 数几条就渲染几条。

    修前：manifest 数的是 `packet["next"]`（按尺寸预算裁剪过的指针数组，`_rich_report` 这种
    包里被整段裁掉 ⇒ 0），而 next.md 渲染的是 `next_ranked` 现算的清单（通常 7 条）。
    同一个案件目录里两个数字打架，交接的人不知道该信哪个。
    """
    p = _packet()
    assert not p.get("next"), "这个包的 next 指针数组本来就被预算裁掉了（正是 V-26② 的场景）"
    root = write_case(p, str(tmp_path / "rep"), ["cmd"], 0, {}, quiet=True)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    text = (root / "next.md").read_text(encoding="utf-8")
    rendered = [ln for ln in text.splitlines() if re.match(r"^\d+\. \*\*", ln)]
    assert rendered, "next.md 必须有条目（否则这条测试退化成 0 == 0）"
    assert isinstance(man["next_actions"], dict) and "count" in man["next_actions"], (
        f"next_actions 要能表达「几条 + 哪几条」，修前只有一个 int：{man['next_actions']!r}")
    assert man["next_actions"]["count"] == len(rendered), (man["next_actions"], rendered)
    # 内容同源：清单里每一条的 code / 提示都能在 next.md 里找到
    for it in man["next_actions"]["items"]:
        needle = it.get("code") or it.get("hint")
        assert needle and needle in text, (it, text)


def test_manifest_and_journal_commands_have_no_absolute_paths(tmp_path):
    """V-26④：argv 里的绝对路径入 `manifest.json`/`journal.jsonl` 前必须脱敏。

    承诺（模块 docstring）= 实际：这个目录要能交接、（将来）入库，不许泄漏本机目录结构。
    """
    p = _packet()
    argv = ["--set", "basic",
            "--ks-calibration", str(tmp_path / "machine-profile.json"),
            "-o", str(tmp_path / "rep.json"),
            "--ks-memtrace=" + str(tmp_path / "trace.txt"),
            "--ks-sqlite", "/somewhere/else/run.sqlite"]
    root = write_case(p, str(tmp_path / "rep"), argv, 0, {}, quiet=True)
    man_blob = (root / "manifest.json").read_text(encoding="utf-8")
    man = json.loads(man_blob)
    journal = json.loads((root / "journal.jsonl").read_text(encoding="utf-8").strip())
    for blob in (man_blob, json.dumps(journal, ensure_ascii=False)):
        assert str(tmp_path) not in blob, blob
        assert "/somewhere/else" not in blob, blob
    assert not [t for t in man["command"] if t.startswith("/")], man["command"]
    # 案件目录旁的路径保持相对（记下来的命令在案件目录里可重跑），别处的用占位符
    assert "../machine-profile.json" in man["command"], man["command"]
    assert "--ks-memtrace=../trace.txt" in man["command"], man["command"]
    assert "<abs>/run.sqlite" in man["command"], man["command"]
    assert journal["command"] == man["command"]


def test_sanitize_argv_is_idempotent(tmp_path):
    """脱敏不许"越洗越花"：洗过一遍的 argv 再洗一遍必须原样。"""
    root = tmp_path / "rep.case"
    once = sanitize_argv(["--ks-calibration", str(tmp_path / "c.json"), "--ks-sqlite=x/y.json"],
                         root)
    assert sanitize_argv(once, root) == once


def test_case_is_written_with_screen_off_csv_and_launch_csv(tmp_path, capsys):
    """V-26③：只有 `--no-case` 能关掉案件目录 —— 这三种模式不许**静默**不写。

    修前：`--screen off`/`--csv`/`--launch-csv` 不建决策包 ⇒ `_write_case_if_needed` 直接
    return，`-o` 旁边什么都没有，而 README/`--no-case` 的承诺是"默认写、显式才关"。
    """
    from kernoscope.cli import main

    cases = (("screen_off", ["--screen", "off"]),
             ("csv", ["--csv"]),
             ("launch_csv", ["--launch-csv"]))
    for name, extra in cases:
        out = tmp_path / name
        rc = main(["--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--quiet",
                   "-o", str(out), *extra])
        assert rc == 0, (name, extra)
        assert (case_dir_for(str(out)) / "manifest.json").is_file(), (
            f"{extra} 静默不写案件目录：语义是「只有 --no-case 能关」")
    # 显式关：才是真的不写
    off = tmp_path / "explicit_off"
    rc = main(["--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--quiet",
               "--no-case", "-o", str(off)])
    assert rc == 0
    assert not case_dir_for(str(off)).exists(), "--no-case 必须真的关掉"
    capsys.readouterr()                     # 吞掉 CSV/一屏输出，别污染别的测试


@pytest.mark.skipif(not CALIB_A.exists(), reason="需要入库的标定产物")
def test_manifest_input_fingerprints_come_from_the_pipeline(tmp_path):
    """V-26①：`protocol_inputs` 必须是**管线真写进去的**指纹，不是测试手塞的键。

    修前：`packet.build()` 只带 `protocol={hash,basis,window_mode}`，manifest 于是恒为
    `{"calibration": {}, "memtrace": {}}` —— 报告里 `meta.calibration` 明明有
    `machine-profile.json:<哈希>`，交接的人却查不到"用的是哪份标定"。
    """
    from kernoscope.cli import main

    out = tmp_path / "rep"
    rc = main(["--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--quiet",
               "--ks-calibration", str(CALIB_A), "-o", str(out)])
    assert rc == 0
    man = json.loads((case_dir_for(str(out)) / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads((tmp_path / "rep.ksreport.json").read_text(encoding="utf-8"))
    cal = man["protocol_inputs"]["calibration"]
    assert cal, "给了 --ks-calibration，输入指纹就不许是空的"
    assert cal == rep["meta"]["calibration"], (cal, rep["meta"]["calibration"])
    assert str(CALIB_A) not in json.dumps(man, ensure_ascii=False), "指纹里不许带绝对路径"
    # 没给的输入是 null（不是 `{}`：`{}` 正是"给了却写成空"这条缺陷原来的样子）
    assert man["protocol_inputs"]["memtrace"] is None


def test_manifest_records_input_identities():
    """包的 `protocol` 块里带了输入指纹时，manifest 要原样带出来（形状随管线）。"""
    p = _packet()
    p.setdefault("protocol", {}).update({
        "calibration": "machine-profile.json:abc123",
        "memtrace": {"name": "trace.txt", "sha256": "def456"},
    })
    man = manifest(p, "rep", ["cmd"], {}, pathlib.Path("/tmp"))
    assert man["protocol_inputs"]["calibration"] == "machine-profile.json:abc123"
    assert man["protocol_inputs"]["memtrace"]["name"] == "trace.txt"
