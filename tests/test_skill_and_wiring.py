"""`kernoscope skill`（I4）与 I3/I8 接线的最小契约。

skill 安装的失败模式是"看起来装好了，其实写坏了/写了一半/把用户的文件覆盖了"，所以测试
盯的是：只打印不写、幂等、备份只留第一份、回滚、以及**回读校验**真的会炸。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kernoscope.cli import main
from kernoscope.skill_cmd import (BACKUP_SUFFIX, apply_install, default_target, plan_install,
                                  render_plan, skill_source_root)

REPO = Path(__file__).resolve().parents[1]
SKILL_SRC = REPO / "skills" / "kernoscope"


def _run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


# --------------------------------------------------------------------------
# 计划：默认只打印
# --------------------------------------------------------------------------
def test_source_root_is_discovered_and_has_the_entry_file():
    src = skill_source_root()
    assert src is not None and (src / "SKILL.md").is_file()
    assert (src / "references").is_dir()


def test_default_target_follows_the_client_convention(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert default_target("claude", home=str(tmp_path)) == tmp_path / ".claude" / "skills" / "kernoscope"
    assert default_target("agents", home=str(tmp_path)) == tmp_path / ".agents" / "skills" / "kernoscope"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert default_target("claude", home=str(tmp_path)) == tmp_path / "cfg" / "skills" / "kernoscope"


def test_plan_lists_create_update_unchanged_and_writes_nothing(tmp_path):
    target = tmp_path / "skills" / "kernoscope"
    plan = plan_install(target=target)
    assert plan.changed and all(a.operation == "create" for a in plan.actions)
    assert plan.requires_approval
    assert not target.exists(), "只算计划时不许创建任何东西"
    text = render_plan(plan)
    assert "默认只打印" in text and "新建" in text and BACKUP_SUFFIX in text

    # 已经装好之后：内容相同 => unchanged，不再要求批准
    apply_install(plan)
    again = plan_install(target=target)
    assert all(a.operation == "unchanged" for a in again.actions)
    assert not again.requires_approval


# --------------------------------------------------------------------------
# 执行：写、幂等、备份、回滚
# --------------------------------------------------------------------------
def test_apply_writes_files_verifies_and_is_idempotent(tmp_path, capsys):
    target = tmp_path / "dest"
    rc, out, _ = _run(capsys, "skill", "--target", str(target))
    assert rc == 0 and not target.exists()
    assert "安装计划" in out

    rc, out, _ = _run(capsys, "skill", "--target", str(target), "--apply")
    assert rc == 0, out
    assert (target / "SKILL.md").is_file()
    assert (target / "references" / "refusal-codes.md").is_file()
    # 脚本必须保持可执行位（agent 会直接跑它们）
    assert (target / "scripts" / "remedy.sh").stat().st_mode & 0o111
    before = {p: p.stat().st_mtime_ns for p in target.rglob("*") if p.is_file()}

    rc, out, _ = _run(capsys, "skill", "--target", str(target), "--apply")
    assert rc == 0 and "已是最新" in out
    after = {p: p.stat().st_mtime_ns for p in target.rglob("*") if p.is_file()}
    assert after == before, "幂等要求：内容相同不许重写（mtime 也不能变）"


def test_update_backs_up_once_and_keeps_the_first_snapshot(tmp_path):
    target = tmp_path / "dest"
    apply_install(plan_install(target=target))
    victim = target / "SKILL.md"
    victim.write_text("本地被改坏的内容", encoding="utf-8")

    plan = plan_install(target=target)
    assert any(a.operation == "update" and a.rel == "SKILL.md" for a in plan.actions)
    res = apply_install(plan)
    assert res["status"] == "installed"
    backup = Path(f"{victim}{BACKUP_SUFFIX}")
    assert backup.read_text(encoding="utf-8") == "本地被改坏的内容"
    assert victim.read_text(encoding="utf-8") != "本地被改坏的内容"

    # 第二次被改坏：备份仍是**第一份**快照（不覆盖）
    victim.write_text("又改坏了", encoding="utf-8")
    apply_install(plan_install(target=target))
    assert backup.read_text(encoding="utf-8") == "本地被改坏的内容"


def test_readback_mismatch_rolls_back_everything(tmp_path, monkeypatch):
    """回读校验失败必须回滚到写之前的状态（半个 skill 比没装更糟）。"""
    import kernoscope.skill_cmd as sc

    target = tmp_path / "dest"
    apply_install(plan_install(target=target))
    (target / "SKILL.md").write_text("改动前", encoding="utf-8")
    real_write = sc._write_atomic

    def bad_write(dst, data):
        # 写进去的东西与源不一致 -> 回读校验必须炸（模拟磁盘/文件系统把它写坏了）
        real_write(dst, b"garbage-not-the-skill")

    monkeypatch.setattr(sc, "_write_atomic", bad_write)
    res = apply_install(plan_install(target=target))
    assert res["status"] == "failed" and "readback mismatch" in res["reason"]
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "改动前"


def test_json_output_is_machine_readable(tmp_path, capsys):
    target = tmp_path / "dest"
    rc, out, _ = _run(capsys, "skill", "--target", str(target), "--json")
    payload = json.loads(out)
    assert rc == 0 and payload["applied"] is False and payload["requires_approval"] is True
    assert payload["operation"] == "skill_install"
    rc, out, _ = _run(capsys, "skill", "--target", str(target), "--json", "--apply")
    payload = json.loads(out)
    assert payload["applied"] is True and payload["result"]["status"] == "installed"


def test_source_that_does_not_exist_fails_instead_of_claiming_up_to_date(tmp_path, capsys):
    """V-30：源目录不存在/不是目录 ⇒ 自家 `error:` + 非零退出，且不许说"已经是目标状态"。

    修前：`skill --apply --source /nonexistent/dir` → rc=0、打「结论：已经是目标状态，无需改动」
    「已是最新」，而目标目录**根本没建** —— 一个 agent 会以为装好了。
    """
    target = tmp_path / "dest"
    missing = tmp_path / "nope"
    rc, out, err = _run(capsys, "skill", "--apply", "--source", str(missing),
                        "--target", str(target))
    assert rc != 0, out
    assert "error:" in err, err
    assert "已经是目标状态" not in out and "已是最新" not in out, out
    assert not target.exists(), "源都没有，绝不该建目标"

    not_a_dir = tmp_path / "a-file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    rc2, out2, err2 = _run(capsys, "skill", "--apply", "--source", str(not_a_dir),
                           "--target", str(target))
    assert rc2 != 0 and "error:" in err2, (out2, err2)
    assert not target.exists()

    # 只算计划（不加 --apply）也必须失败：计划本身就算不出来
    rc3, out3, err3 = _run(capsys, "skill", "--source", str(missing), "--target", str(target))
    assert rc3 != 0 and "error:" in err3, (out3, err3)

    # 源目录存在但是空的：也是缺陷（不是"已经是最新"）
    empty = tmp_path / "empty-src"
    empty.mkdir()
    rc4, out4, err4 = _run(capsys, "skill", "--apply", "--source", str(empty),
                           "--target", str(target))
    assert rc4 != 0 and "error:" in err4, (out4, err4)
    assert not target.exists()


def test_empty_source_plan_never_reports_up_to_date(tmp_path):
    """`render_plan`/`apply_install` 层同一条语义：空计划 ≠ 已经是最新。"""
    from kernoscope.skill_cmd import FileAction, SkillPlan, render_plan

    empty = SkillPlan(source=str(tmp_path / "empty"), target=str(tmp_path / "dest"))
    text = render_plan(empty)
    assert "已经是目标状态" not in text and "已是最新" not in text
    assert "没有可安装的文件" in text
    assert apply_install(empty)["status"] == "failed"
    # 有文件、且与目标一致时才是真的"已经是目标状态"
    same = SkillPlan(source=str(tmp_path), target=str(tmp_path / "d"),
                     actions=[FileAction(rel="SKILL.md", operation="unchanged", bytes_=1)])
    assert "已经是目标状态" in render_plan(same)
    assert apply_install(same)["status"] == "unchanged"


# --------------------------------------------------------------------------
# I3：Markdown 报告接线
# --------------------------------------------------------------------------
def test_report_md_flag_writes_a_structured_markdown_report(tmp_path, capsys):
    out_path = tmp_path / "reports" / "kernel.md"       # 父目录不存在：必须自动创建
    rc, out, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                      "--screen", "off", "--quiet", "--report-md", str(out_path))
    assert rc == 0, out
    text = out_path.read_text(encoding="utf-8")
    for heading in ("## 0. Profiling setup", "## 1. Headline numbers",
                    "## 2. Per-dimension analysis", "## 3. Summary diagnosis",
                    "## 4. Optimization directions", "## 5. Confidence & caveats",
                    "## 6. Reproduction"):
        assert heading in text, f"缺小节 {heading}"
    assert "**One-line read:**" in text
    assert "| Source |" in text or "Source" in text.split("## 1.")[1].split("## 2.")[0]


def test_report_md_without_kernels_fails_loudly(tmp_path, capsys):
    """没有 kernel 级数据时不许写出一份空报告（那会让人以为"分析过了"）。"""
    from kernoscope import cli as cli_mod

    class DeviceOnly:
        kernels = [type("K", (), {"name": "(device)", "total_ns": 0})()]
        meta: dict = {}

    rc = cli_mod._write_report_md(DeviceOnly(), type("A", (), {"report_md": "x.md"})(),
                                  None, [])
    err = capsys.readouterr().err
    assert rc == 1 and "至少一个 kernel" in err
    assert not Path("x.md").exists()

    # CLI 侧的同一件事：-k 匹配不到 kernel 时在校验之前就失败了（rc=1，不写文件）
    rc2, _, err2 = _run(capsys, "--ks-synthetic", "-k", "regex:does_not_exist", "--set",
                        "basic", "--screen", "off", "--report-md", str(tmp_path / "x.md"))
    assert rc2 == 1 and "kernel" in err2
    assert not (tmp_path / "x.md").exists()


# --------------------------------------------------------------------------
# I8：NVTX 阶段归因接线
# --------------------------------------------------------------------------
_KERNEL_COLUMNS = ("start, end, deviceId, streamId, demangledName, shortName, mangledName, "
                   "registersPerThread, gridX, gridY, gridZ, blockX, blockY, blockZ, "
                   "staticSharedMemory, dynamicSharedMemory, localMemoryPerThread, "
                   "gridId, correlationId")


def _nvtx_db(path: Path, *, with_nvtx: bool = True) -> str:
    """一个 nsys 形状的库：kernel 表必须带 backends/nsys.py 解析用到的全部列。"""
    con = sqlite3.connect(path)
    con.execute("create table StringIds (id integer primary key, value text)")
    con.execute(f"create table CUPTI_ACTIVITY_KIND_KERNEL ({_KERNEL_COLUMNS})")
    # 名字列是 StringIds 的外键（nsys 就是这么存的）
    for i, v in enumerate(("k_prefill(float*)", "k_prefill", "_Z10k_prefillPf",
                           "k_decode(float*)", "k_decode", "_Z9k_decodePf"), 1):
        con.execute("insert into StringIds values (?, ?)", (i, v))
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values "
                "(1000, 3000, 0, 7, 1, 2, 3, 32, 256, 1, 1, 256, 1, 1, 0, 0, 0, 0, 1)")
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values "
                "(5000, 9000, 0, 7, 4, 5, 6, 32, 512, 1, 1, 256, 1, 1, 0, 0, 0, 0, 2)")
    if with_nvtx:
        con.execute("create table NVTX_EVENTS "
                    "(start integer, end integer, text text, textId integer, globalTid integer, "
                    "eventType integer)")
        con.execute("insert into NVTX_EVENTS values (0, 4000, 'prefill', NULL, 1, 59)")
        con.execute("insert into NVTX_EVENTS values (4500, 10000, 'decode', NULL, 1, 59)")
    con.commit()
    con.close()
    return str(path)


def test_nvtx_phases_section_and_meta(tmp_path, capsys):
    db = _nvtx_db(tmp_path / "run.sqlite")
    rc, out, err = _run(capsys, "--ks-sqlite", db, "--ks-nvtx-phases", "--set", "basic",
                        "--screen", "off", "-o", str(tmp_path / "rep"), "-f")
    assert rc == 0, err
    assert "nvtx: 2 个阶段" in err
    payload = json.loads((tmp_path / "rep.ksreport.json").read_text(encoding="utf-8"))
    dev = [k for k in payload["kernels"] if k["name"] == "(device)"]
    assert dev, "device 伪 kernel 应当带着 NvtxPhases section"
    sec = next(s for s in dev[0]["sections"] if s["name"] == "NvtxPhases")
    names = {m["name"] for m in sec["metrics"]}
    assert "ks__nvtx_attributed_kernel_ms" in names
    assert any("__prefill__kernel_ms" in n for n in names)
    assert any("__decode__kernel_ms" in n for n in names)
    summary = next(m for m in sec["metrics"] if m["name"] == "ks__nvtx_attributed_kernel_ms")
    assert "重叠" in (summary.get("source") or ""), "口径必须随数字一起出现"
    # "阶段之和不是划分"这句话必须真的在表里（嵌套区间各自计入）
    assert any("不是划分" in (m.get("note") or "") for m in sec["metrics"])


def test_nvtx_phases_without_nvtx_gives_na_rows_not_silence(tmp_path, capsys):
    """没有 --trace=nvtx 的库：那一节必须**存在**、带对的 code，并给对的处置（V-28）。

    修前这一行的 code 是 `metric_absent`，于是报告里的 `fix:` 是"用 --json 看有哪些指标"——
    读者会去查指标名，而真正该做的是**重采时带 `--trace=nvtx`**。
    """
    db = _nvtx_db(tmp_path / "nonvtx.sqlite", with_nvtx=False)
    rc, out, err = _run(capsys, "--ks-sqlite", db, "--ks-nvtx-phases", "--set", "basic",
                        "--screen", "off", "-o", str(tmp_path / "rep2"), "-f")
    assert rc == 0
    assert "不给阶段数字" in err
    payload = json.loads((tmp_path / "rep2.ksreport.json").read_text(encoding="utf-8"))
    dev = [k for k in payload["kernels"] if k["name"] == "(device)"][0]
    sec = next(s for s in dev["sections"] if s["name"] == "NvtxPhases")
    row = sec["metrics"][0]
    assert row["provenance"] == "unavailable" and row["refusal"]["code"]
    assert row["refusal"]["code"] == "no_nvtx", row["refusal"]
    # 人看的报告里，这一行的处置必须指向"重采时带 --trace=nvtx"
    assert "[no_nvtx]" in out and "--trace=nvtx" in out
    assert "[metric_absent]" not in out


def test_skill_and_report_flags_are_documented_in_changelog():
    """接口面变化必须写进 CHANGELOG（R-67 的判据：外部消费者要改代码就得知道）。"""
    ch = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    for needle in ("kernoscope skill", "--report-md", "--ks-nvtx-phases", "--export-ncu-rep",
                   "--ks-ncu-rep", "--plan"):
        assert needle in ch, f"CHANGELOG 缺 {needle}"
