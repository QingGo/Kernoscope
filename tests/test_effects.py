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
    assert plan.requires_approval is False        # 只读查询不需要点头
    assert "零外部副作用" in plan.note


def test_existing_sqlite_skips_collection_and_target():
    """给了 --ks-sqlite：不跑目标、不采 nsys —— 计划里必须没有这两条（否则计划是错的）。"""
    plan = plan_for(_args("--ks-sqlite", "run.sqlite", "--set", "basic"))
    ids = {i.op.id for i in plan.items}
    assert "run_target" not in ids and "nsys_collect" not in ids
    assert "read_inputs" in ids


def test_write_channel_lists_the_actual_paths():
    """计划里的文件名必须与 `_emit` 真写的文件一致（`{base}.ksreport.json`）。"""
    plan = plan_for(_args("-o", "out/rep.json", "--set", "basic", "--ks-synthetic"))
    item = next(i for i in plan.items if i.op.id == "write_outputs")
    assert "out/rep.ksreport.json" in item.target
    assert "out/rep.case/" in item.target
    no_case = plan_for(_args("-o", "out/rep.json", "--no-case", "--set", "basic",
                             "--ks-synthetic"))
    item2 = next(i for i in no_case.items if i.op.id == "write_outputs")
    assert ".case/" not in item2.target


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
_OP_BINARIES: dict[str, set[str]] = {
    "probe_env": {"nvidia-smi", "nsys", "cuobjdump", "ncu"},
    "device_sample": {"nvidia-smi"},
    "nsys_collect": {"nsys"},
    "nsys_export": {"nsys"},
    "static_dump": {"cuobjdump"},
    "run_target": {"true"},
    "read_inputs": set(),
    "write_outputs": set(),
    "skill_install": set(),
}


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
