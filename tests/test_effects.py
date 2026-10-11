"""副作用账（A2）的契约测试。

两件事必须被钉住：
  1. **声明完整性** —— 每个操作都有 effects，四个 annotation 纯推导（不许按名字猜）；
  2. **声明与实际一致** —— 用 PATH 上的记录桩把"真的启动了哪些子进程"记下来，
     与 `--plan` 说的对照。只测第 1 条的话，`--plan` 可以是一份漂亮的散文而完全不真。
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from kernoscope.cli import build_parser, main
from kernoscope.effects import (OPERATIONS, Plan, UndeclaredEffect, effects_for, plan_for,
                               render_plan)


def _args(*argv: str):
    return build_parser().parse_args(list(argv))


# --------------------------------------------------------------------------
# ① 声明完整性
# --------------------------------------------------------------------------
def test_every_operation_has_effects_and_derived_annotations():
    for key, op in OPERATIONS.items():
        assert op.id == key, f"{key} 的 id 与键不一致"
        eff = effects_for(key)
        assert eff is op.effects
        ann = eff.annotations()
        # 纯推导：read_only 等价于"七项全 false"，不允许各写一份
        assert ann["read_only_hint"] == (not any(
            [eff.mutates_target, eff.mutates_session, eff.writes_filesystem,
             eff.launches_process, eff.accesses_network, eff.changes_ui_state,
             eff.may_discard_data]))
        assert ann["destructive_hint"] == eff.may_discard_data
        assert ann["idempotent_hint"] == eff.idempotent
        assert ann["open_world_hint"] == (eff.launches_process or eff.accesses_network)


def test_unknown_operation_raises_instead_of_silently_doing_nothing():
    with pytest.raises(UndeclaredEffect):
        effects_for("rm_rf_everything")


def test_side_effecting_operations_are_not_advertised_as_read_only():
    """起进程/写文件的操作不许是 read_only —— 这是"别把写操作说成只读"的类型级执行点。"""
    for op in OPERATIONS.values():
        e = op.effects
        if e.launches_process or e.writes_filesystem or e.may_discard_data:
            assert not e.read_only, f"{op.id} 有副作用却声称只读"


def test_static_dump_says_it_does_not_execute_the_target():
    """静态分析不执行被测二进制：这句话必须写在操作的 detail 里（不是文档里的口头约定）。"""
    assert "不执行" in OPERATIONS["static_dump"].detail
    assert "不注入" in OPERATIONS["run_target"].detail


# --------------------------------------------------------------------------
# ② 计划由参数推出（纯函数）
# --------------------------------------------------------------------------
def test_full_invocation_declares_every_channel():
    args = _args("--set", "basic", "-o", "rep", "--ks-cubin", "k.cubin",
                 "--ks-calibration", "machine-profile.json", "--", "python", "serve.py")
    items = {i.op.id for i in plan_for(args).items}
    assert items == {"probe_env", "read_inputs", "device_sample", "run_target",
                     "nsys_collect", "static_dump", "write_outputs"}


def test_synthetic_run_declares_the_probe_but_nothing_that_writes():
    """合成模式**仍会**做只读环境探测（doctor 能力判定需要）—— 这是实测抓出来的。"""
    plan = plan_for(_args("--ks-synthetic", "--ks-no-nvml", "--set", "basic"))
    assert [i.op.id for i in plan.items] == ["probe_env"]
    assert plan.launches_process is True          # 会起 nvidia-smi（只读查询）
    assert plan.writes_filesystem is False
    assert plan.declared_writes() == []           # 不给任何输出开关 ⇒ 一个文件都不声明
    assert plan.requires_approval is False        # 只读查询不需要点头
    assert "零外部副作用" in plan.note
    # `--packet -` 是 stdout，不是文件：不许把它算成一个落盘点。
    assert plan_for(_args("--ks-synthetic", "--packet", "-")).declared_writes() == []


def test_existing_sqlite_skips_collection_and_target():
    """给了 --ks-sqlite：不跑目标、不采 nsys —— 计划里必须没有这两条（否则计划是错的）。"""
    plan = plan_for(_args("--ks-sqlite", "run.sqlite", "--set", "basic"))
    ids = {i.op.id for i in plan.items}
    assert "run_target" not in ids and "nsys_collect" not in ids
    assert "read_inputs" in ids


def test_write_channel_lists_the_actual_paths():
    """计划里的文件名必须是**磁盘上的形状**（`{base}.ksreport.json`，案件目录逐文件列出）。"""
    plan = plan_for(_args("-o", "out/rep.json", "--set", "basic", "--ks-synthetic"))
    item = next(i for i in plan.items if i.op.id == "write_outputs")
    assert "out/rep.ksreport.json" in item.writes
    assert "out/rep.perfetto.json" in item.writes
    assert "out/rep.case/manifest.json" in item.writes
    assert "out/rep.case/journal.jsonl" in item.writes
    no_case = plan_for(_args("-o", "out/rep.json", "--no-case", "--set", "basic",
                             "--ks-synthetic"))
    item2 = next(i for i in no_case.items if i.op.id == "write_outputs")
    assert not [w for w in item2.writes if ".case/" in w]


def test_plan_requires_approval_only_when_it_changes_the_world():
    assert plan_for(_args("--ks-synthetic", "--ks-no-nvml")).requires_approval is False
    assert plan_for(_args("--ks-synthetic", "-o", "rep")).requires_approval is True
    assert plan_for(_args("--ks-memtrace", "t.txt", "--ks-synthetic")).requires_approval is False


def test_render_plan_first_line_promises_no_execution():
    text = render_plan(plan_for(_args("--ks-synthetic", "--set", "basic")))
    assert "不执行" in text.splitlines()[0]
    assert "只读" in text


# --------------------------------------------------------------------------
# ③ 声明 vs 实际：用 PATH 记录桩对账
# --------------------------------------------------------------------------
@pytest.fixture
def stub_path(tmp_path, monkeypatch):
    """把 nsys/nvidia-smi 换成"记录 argv 然后退出"的桩，返回记录文件路径。"""
    log = tmp_path / "launched.log"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("nsys", "nvidia-smi", "cuobjdump"):
        p = bindir / name
        p.write_text(f'#!/bin/sh\necho "{name} $@" >> "{log}"\nexit 1\n', encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


def _launched(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


# 计划里的每个操作**可能**用到哪个外部可执行文件（用于"声明 vs 观察"对账）。
# 这张表是测试侧的期望：漏一个就说明 plan_for 漏声明了一个会起进程的通道。
# 它是**穷举**的（下面 test_op_binaries_table_covers_every_operation 会钉住）——
# 不给默认空集，新增一个会起进程的操作就没法靠"没登记=空集"溜过去。
_OP_BINARIES: dict[str, set[str]] = {
    "probe_env": {"nvidia-smi", "nsys", "cuobjdump", "ncu"},
    "device_sample": {"nvidia-smi"},
    "nsys_collect": {"nsys"},
    "nsys_export": {"nsys"},
    "static_dump": {"cuobjdump"},
    "run_target": {"true"},
    "read_inputs": set(),
    "write_outputs": set(),
    "write_report_md": set(),
    "write_html_view": set(),
    "write_ncu_rep": set(),
    "write_ncu_truth": set(),
    "write_name_map": set(),
    "write_log_file": set(),
    "skill_install": set(),
}


def test_op_binaries_table_covers_every_operation():
    """操作表与"会起什么进程"的对照表必须同构：新增操作必须显式登记（哪怕登记成空集）。

    否则 `_declared_binaries` 的 `.get(id, set())` 会把一个新操作当成"不会起进程"，
    对账测试于是对它失效 —— 计划可以新增一条起进程的通道而测试照样绿。
    """
    assert set(_OP_BINARIES) == set(OPERATIONS)


def _declared_binaries(plan) -> set[str]:
    return {b for item in plan.items for b in _OP_BINARIES.get(item.op.id, set())}


def test_plan_matches_observed_launches(stub_path, tmp_path, monkeypatch, capsys):
    """计划说会起什么 ⇒ 实际观察到的子进程必须是它的子集；说不会起 ⇒ 一个都不许起。

    这条测试是"计划是不是真的"唯一的硬判据。它第一次跑就抓到真缺陷：
    `--ks-synthetic --ks-no-nvml` 时 plan 说"没有任何外部副作用"，而 doctor 的能力判定
    已经悄悄调了 4 次 `nvidia-smi` —— 于是 probe_env 这条操作被补进计划。
    """
    monkeypatch.chdir(tmp_path)

    # (a) 真采集路径：计划声明 nsys，观察到的必须 ⊆ 声明
    planned = plan_for(_args("--set", "basic", "-o", "rep", "--", "true"))
    assert any("nsys" in c for c in planned.commands())
    main(["--set", "basic", "-o", str(tmp_path / "rep"), "--", "true"])
    capsys.readouterr()
    observed = _launched(stub_path)
    got = {line.split()[0] for line in observed}
    assert got, "计划说会起 nsys/nvidia-smi，实际一个子进程都没起（声明与实现不一致）"
    assert got <= _declared_binaries(planned), f"起了计划没声明的进程：{got}"
    assert any(line.startswith("nsys profile") for line in observed), \
        "计划声明了 nsys_collect，实际没有发起采集"

    # (b) 合成 + 关采样 + 不写文件：只允许 probe_env 的只读查询
    stub_path.unlink(missing_ok=True)
    plan2 = plan_for(_args("--ks-synthetic", "--ks-no-nvml", "--set", "basic"))
    assert plan2.writes_filesystem is False
    rc2 = main(["--ks-synthetic", "--ks-no-nvml", "--set", "basic"])
    capsys.readouterr()
    assert rc2 == 0
    observed2 = _launched(stub_path)
    assert {line.split()[0] for line in observed2} <= _declared_binaries(plan2)
    assert not any("profile" in line for line in observed2), \
        "计划说这条运行不采集，实际却发起了 nsys 采集"


def test_write_outputs_is_observed_not_just_declared(stub_path, tmp_path, monkeypatch, capsys):
    """计划说会写哪些文件 ⇒ 那些文件必须真的出现（列出来的路径不许是装饰）。"""
    monkeypatch.chdir(tmp_path)
    plan = plan_for(_args("--ks-synthetic", "--ks-no-nvml", "-o", "out/rep.json",
                          "--no-case", "--set", "basic"))
    item = next(i for i in plan.items if i.op.id == "write_outputs")
    assert plan.requires_approval is True
    rc = main(["--ks-synthetic", "--ks-no-nvml", "-o", "out/rep.json", "--no-case",
               "--set", "basic", "--screen", "off", "--quiet"])
    capsys.readouterr()
    assert rc == 0
    assert (tmp_path / "out" / "rep.ksreport.json").exists(), \
        f"计划写了 {item.writes}，实际没写"
    assert not (tmp_path / "out" / "rep.case").exists()


def test_plan_flag_is_side_effect_free_and_machine_readable(tmp_path, monkeypatch, capsys):
    """`--plan` 自己不许有任何副作用：不起进程、不写文件（连临时目录都不留）。"""
    monkeypatch.chdir(tmp_path)
    rc = main(["--plan", "-o", "out/rep", "--ks-synthetic", "--", "true"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "只打印" in out and "不执行" in out
    assert list(tmp_path.rglob("*")) == []

    rc2 = main(["--plan-json", "-o", "out/rep", "--set", "basic", "--", "true"])
    payload = json.loads(capsys.readouterr().out)
    assert rc2 == 0
    assert payload["requires_approval"] is True
    ids = [s["id"] for s in payload["steps"]]
    assert "write_outputs" in ids and "run_target" in ids
    for step in payload["steps"]:
        assert set(step["effects"]) == {
            "mutates_target", "mutates_session", "writes_filesystem", "launches_process",
            "accesses_network", "changes_ui_state", "may_discard_data", "idempotent"}
        assert set(step["annotations"]) == {
            "read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint"}
    assert list(tmp_path.rglob("*")) == []


def test_plan_reports_third_party_tools_and_network_honestly():
    plan = plan_for(_args("--set", "basic", "--ks-nsys-rep", "a.nsys-rep"))
    text = render_plan(plan)
    assert "第三方软件" in text
    payload = plan_for(_args("--set", "basic", "--", "python", "serve.py")).to_dict()
    assert all(s["effects"]["accesses_network"] is False for s in payload["steps"])
    assert isinstance(Plan().to_dict()["steps"], list)


# --------------------------------------------------------------------------
# ④ 声明 vs 磁盘：一条带齐开关的命令，逐文件对账（V-18）
# --------------------------------------------------------------------------
# 这条命令刻意把**所有会写文件的开关**都打开，并全部走离线合成路径（不碰 GPU）。
# V-18（第五轮验收 P2）的原话：`--plan` 只声明 4 项，而这条命令实跑会留下 13 个文件 ——
# 漏的是 `--report-md`、`--export-ncu-rep`(+`.ATTRIBUTION.txt`)、`--export-name-map`、
# `--log-file`。漏声明的代价不是"文档少一行"：agent 读到"只写 1 个文件"就会据此决定要不要跑。
_FULL_WRITE_ARGV = [
    "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
    "-o", "out/rep.json", "--launch-csv", "--packet", "out/rep.packet.json",
    "--report-md", "out/rep.md", "--html", "out/rep.html",
    "--export-ncu-rep", "out/rep.ncu-rep",
    "--export-name-map", "out/names.json", "--log-file", "out/run.log",
    "--screen", "off", "--quiet",
]

# V-18 漏声明、现在必须逐条对上的那 5 个落盘点（含 `.ncu-rep` 的旁路署名）。
_V18_WRITES = ("out/rep.md", "out/rep.ncu-rep", "out/rep.ncu-rep.ATTRIBUTION.txt",
               "out/names.json", "out/run.log", "out/rep.html")


def _files_under(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_declared_writes_equal_files_on_disk(stub_path, tmp_path, monkeypatch, capsys):
    """**一条命令真跑一遍**：`plan_for()` 声明的路径集合 == 磁盘上真实出现的文件集合。

    两个方向都必须成立（只查一边都会漏）：
      * 磁盘上有、计划没说 ⇒ 承诺不实（V-18 就是这一边：声明 4 项、实得 13 个文件）；
      * 计划说了、磁盘上没有 ⇒ 计划在吹牛（列出来的路径成了装饰）。

    声明里允许 `<…>` 占位符（系统临时目录这类**不在工作目录**的落点），带占位符的项按
    "不许落在工作目录"处理。由子进程（nsys / nsys export）写的中间产物不在这次合成路径里，
    所以这里对账的是**我们自己的写入点** —— V-18 漏掉的那 5 项全在其中；子进程那条另见
    `test_nsys_intermediates_are_declared_where_they_really_land`。
    """
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    plan = plan_for(_args(*_FULL_WRITE_ARGV), _FULL_WRITE_ARGV)
    declared = plan.declared_writes()
    concrete = [p for p in declared if "<" not in p]
    templated = [p for p in declared if "<" in p]
    # 合成模式下不采集 ⇒ 这次没有任何"落在工作目录之外"的声明。
    assert templated == []
    assert plan.requires_approval is True

    rc = main(list(_FULL_WRITE_ARGV))
    capsys.readouterr()
    assert rc == 0

    actual = _files_under(work)
    assert sorted(concrete) == actual, (
        f"声明与实得对不上。\n声明({len(concrete)})：{sorted(concrete)}\n"
        f"实得({len(actual)})：{actual}\n"
        f"磁盘有而没声明：{sorted(set(actual) - set(concrete))}\n"
        f"声明了却没有：{sorted(set(concrete) - set(actual))}")
    # 这条命令的规模本身也是承诺的一部分：4 项的时代必须回不去（N9 之后是 ≥14）。
    assert len(actual) >= 14
    # V-18 的那 5 项必须在声明里（上面已证明它们真的落盘）。
    for path in _V18_WRITES:
        assert path in concrete, f"V-18 漏声明的落盘点又漏了：{path}"
    # 声明里不许出现绝对路径 / 家目录展开（会连带把机器身份写进计划）。
    assert not [p for p in declared if p.startswith(("/", "~"))]
    assert not [p for p in declared if str(tmp_path) in p]


def test_render_plan_lists_each_declared_write_on_its_own_line():
    """13+ 个文件挤成一行就没人读了：多于两个时逐条列，且数量可见。"""
    plan = plan_for(_args(*_FULL_WRITE_ARGV))
    item = next(i for i in plan.items if i.op.id == "write_outputs")
    text = render_plan(plan)
    assert f"会写（{len(item.writes)} 个文件）：" in text
    assert all(f"\n       - {w}" in text for w in item.writes)
    # 声明的每一条都必须能在计划文本里看到（不能只在 JSON 里有）。
    assert all(w in text for w in plan.declared_writes())


def test_nsys_intermediates_are_declared_where_they_really_land():
    """nsys 中间产物的落点由 `-o` 决定 —— 这是"默认行为"里最容易被漏声明的一条。

    实测（`cli.collect`）：给了 `-o` 就写在 `<前缀>_nsys.nsys-rep`/`.sqlite` 并**默认保留**
    （`--ks-keep-intermediate` 管不着）；没给 `-o` 才进系统临时目录、结束时删掉。
    `--ks-nsys-rep` 的 sqlite 也进临时目录（`backends.nsys.sqlite_from_rep`），不写在你旁边。
    """
    with_o = plan_for(_args("-o", "out/rep.json", "--set", "basic", "--", "true"))
    item = next(i for i in with_o.items if i.op.id == "nsys_collect")
    assert "out/rep_nsys.nsys-rep" in item.writes
    assert "out/rep_nsys.sqlite" in item.writes
    assert all("<" not in w for w in item.writes), "落在工作目录里就必须是具体路径"

    plain = plan_for(_args("--set", "basic", "--", "true"))
    item2 = next(i for i in plain.items if i.op.id == "nsys_collect")
    assert item2.writes and all("<" in w for w in item2.writes), \
        "没给 -o ⇒ 中间产物只在系统临时目录（必须带占位符，别写成具体路径）"

    export = plan_for(_args("--set", "basic", "--ks-nsys-rep", "a.nsys-rep"))
    item3 = next(i for i in export.items if i.op.id == "nsys_export")
    assert item3.writes and all("<" in w for w in item3.writes)


def test_ncu_truth_is_declared_next_to_the_report_only_when_asked():
    """`.ncu-truth.json`（I1 真值产物）只在 `--ks-ncu-rep` **且**给了 `-o` 时才会落盘。"""
    both = plan_for(_args("-o", "out/rep.json", "--ks-ncu-rep", "td.ncu-rep",
                          "--ks-synthetic"))
    assert "out/rep.ncu-truth.json" in both.declared_writes()
    only_rep = plan_for(_args("--ks-ncu-rep", "td.ncu-rep", "--ks-synthetic"))
    assert not [p for p in only_rep.declared_writes() if p.endswith(".ncu-truth.json")]
    only_o = plan_for(_args("-o", "out/rep.json", "--ks-synthetic"))
    assert not [p for p in only_o.declared_writes() if p.endswith(".ncu-truth.json")]
