"""V-25 P2：用户会直接撞到的一类错误 —— 每条都必须是「自家 `error:` 行 + 退出码」，
**不许**裸 Python traceback（rc=1、无自家 error 行）。

覆盖面（逐条对应验收报告点名的位置）：
  * 只读目录/无权限写产物：`-o` 报告、`--packet`、`--report-md`、`--export-ncu-rep`
    （含旁路 `.ATTRIBUTION.txt`）、`--export-name-map`、`--log-file`；
  * `--ks-calibration` / `-i` / `--ks-achieved-occupancy` 指向不存在 / 坏 JSON；
  * 坏 sqlite（`--ks-sqlite`）；
  * `--from` 系列（issue / memtrace / occupancy）不存在文件；
  * `main()` 最外层窄兜底的边界：只翻译输入/环境异常，不吞程序 bug、不吞
    `KeyboardInterrupt`。

构造只用 `tmp_path`（不写机器身份/绝对路径）；只读目录用 `chmod 0o500`，跑在 root 下
（0500 也能写）时 `pytest.skip`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernoscope.cli import main

TRACEBACK = "Traceback (most recent call last)"
# 合成负载：不需要 GPU、不需要 nsys，几毫秒出报告
SYNTH = ("--ks-synthetic", "--ks-no-nvml", "--set", "basic")


def run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def assert_own_error(rc, out, err, *, expect_rc=1):
    """约定：rc 固定；stderr 有一行 `error:`；两边都不能出现裸 traceback。"""
    assert rc == expect_rc, f"rc={rc}（期望 {expect_rc}）；stderr={err!r}"
    assert "error:" in err, f"stderr 里没有自家 error 行：{err!r}"
    assert TRACEBACK not in err, f"stderr 出现裸 traceback：{err!r}"
    assert TRACEBACK not in out, f"stdout 出现裸 traceback：{out!r}"


@pytest.fixture
def readonly_dir(tmp_path: Path) -> Path:
    """真的写不进去的目录；跑在 root 下（chmod 挡不住）时 skip。"""
    d = tmp_path / "readonly"
    d.mkdir()
    d.chmod(0o500)
    probe = d / ".probe"
    try:
        probe.write_text("x")
    except OSError:
        pass
    else:
        probe.unlink()
        d.chmod(0o700)
        pytest.skip("当前用户能写 0500 目录（多半是 root）—— 只读目录这条路径测不了")
    try:
        yield d
    finally:
        d.chmod(0o700)  # 让 tmp_path 清理得掉


# --------------------------------------------------------------------------
# 只读目录写产物
# --------------------------------------------------------------------------
def test_readonly_dir_report_json(capsys, readonly_dir: Path):
    rc, out, err = run(capsys, *SYNTH, "-o", str(readonly_dir / "rep"))
    assert_own_error(rc, out, err)
    assert "报告" in err


def test_readonly_dir_packet(capsys, readonly_dir: Path):
    rc, out, err = run(capsys, *SYNTH, "--packet", str(readonly_dir / "packet.json"))
    assert_own_error(rc, out, err)
    assert "决策包" in err


def test_readonly_dir_report_md(capsys, readonly_dir: Path):
    rc, out, err = run(capsys, *SYNTH, "--report-md", str(readonly_dir / "report.md"))
    assert_own_error(rc, out, err)


def test_readonly_dir_export_ncu_rep_and_sidecar(capsys, readonly_dir: Path):
    """`.ncu-rep` 与其旁路 `.ATTRIBUTION.txt` 都不能把 PermissionError 甩出来。"""
    target = readonly_dir / "sub.ncu-rep"
    rc, out, err = run(capsys, *SYNTH, "--export-ncu-rep", str(target))
    assert_own_error(rc, out, err)
    assert "ncu-rep" in err
    assert not target.exists()
    assert not (readonly_dir / "sub.ncu-rep.ATTRIBUTION.txt").exists()


def test_readonly_dir_export_name_map(capsys, readonly_dir: Path):
    rc, out, err = run(capsys, *SYNTH, "--export-name-map", str(readonly_dir / "names.json"))
    assert_own_error(rc, out, err)


def test_readonly_dir_log_file(capsys, readonly_dir: Path):
    rc, out, err = run(capsys, *SYNTH, "--log-file", str(readonly_dir / "run.log"))
    assert_own_error(rc, out, err)
    assert "--log-file" in err


def test_readonly_dir_cannot_create_parent(capsys, readonly_dir: Path):
    """父目录建不出来（在只读目录里 mkdir）也要是自家错误，不是 FileNotFoundError。"""
    rc, out, err = run(capsys, *SYNTH, "-o", str(readonly_dir / "sub" / "rep"))
    assert_own_error(rc, out, err)


# --------------------------------------------------------------------------
# 输入文件：不存在 / 坏 JSON / 顶层形状不对
# --------------------------------------------------------------------------
def test_calibration_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, *SYNTH, "--ks-calibration", str(tmp_path / "nope.json"))
    assert_own_error(rc, out, err)
    assert "标定" in err


def test_calibration_bad_json(capsys, tmp_path: Path):
    bad = tmp_path / "cal.json"
    bad.write_text("{ not json", encoding="utf-8")
    rc, out, err = run(capsys, *SYNTH, "--ks-calibration", str(bad))
    assert_own_error(rc, out, err)
    assert "JSON" in err


def test_calibration_wrong_shape(capsys, tmp_path: Path):
    """合法 JSON 但不是对象（数组）：也要自家错误，不能 AttributeError。"""
    arr = tmp_path / "cal.json"
    arr.write_text("[]", encoding="utf-8")
    rc, out, err = run(capsys, *SYNTH, "--ks-calibration", str(arr))
    assert_own_error(rc, out, err)


def test_import_report_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, "-i", str(tmp_path / "nope.json"))
    assert_own_error(rc, out, err)
    assert "回读报告" in err


def test_import_report_bad_json(capsys, tmp_path: Path):
    bad = tmp_path / "rep.json"
    bad.write_text("not json at all", encoding="utf-8")
    rc, out, err = run(capsys, "-i", str(bad))
    assert_own_error(rc, out, err)


def test_achieved_occupancy_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, *SYNTH, "--ks-achieved-occupancy", str(tmp_path / "nope.json"))
    assert_own_error(rc, out, err)
    assert "occupancy" in err


# --------------------------------------------------------------------------
# 坏 sqlite
# --------------------------------------------------------------------------
def test_ks_sqlite_not_a_database(capsys, tmp_path: Path):
    bad = tmp_path / "garbage.sqlite"
    bad.write_bytes(b"this is not a sqlite database\n" * 4)
    rc, out, err = run(capsys, "--ks-sqlite", str(bad))
    assert_own_error(rc, out, err)
    assert "sqlite" in err.lower() or "SQLite" in err


def test_ks_sqlite_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, "--ks-sqlite", str(tmp_path / "nope.sqlite"))
    assert_own_error(rc, out, err)


def test_errors_survive_quiet(capsys, tmp_path: Path):
    """`--quiet` 关的是噪声，不是错误：写不出产物 / 坏输入时那行 error 必须还在。

    静默的 rc=1 比一行错误更难排查（agent 只看退出码，会以为"没数据"而不是"路径写错"）。
    """
    bad = tmp_path / "garbage.sqlite"
    bad.write_bytes(b"not a database at all\n" * 4)
    rc, out, err = run(capsys, "--quiet", "--ks-sqlite", str(bad))
    assert_own_error(rc, out, err)

    rc, out, err = run(capsys, "--quiet", "-i", str(tmp_path / "nope.json"))
    assert_own_error(rc, out, err)


# --------------------------------------------------------------------------
# `--from` 系列（本来就走自家约定 —— 加测试钉住，别再退化）
# --------------------------------------------------------------------------
def test_issue_from_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, "issue", "--from", str(tmp_path / "nope.hist"))
    assert_own_error(rc, out, err)


def test_memtrace_from_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, "memtrace", "--from", str(tmp_path / "nope.trace"))
    assert_own_error(rc, out, err)


def test_occupancy_from_missing(capsys, tmp_path: Path):
    rc, out, err = run(capsys, "occupancy", "--from", str(tmp_path / "nope.probe"))
    assert_own_error(rc, out, err)


# --------------------------------------------------------------------------
# main() 最外层窄兜底的边界
# --------------------------------------------------------------------------
def test_unexpected_oserror_is_translated(monkeypatch, capsys):
    """没被单条路径接住的 OSError 也要变成 error: 行（rc=1），而不是裸 traceback。"""
    import kernoscope.cli as cli

    def boom(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "_emit", boom)
    rc, out, err = run(capsys, *SYNTH, "--ks-gemm", "8,8,8")
    assert_own_error(rc, out, err)
    assert "PermissionError" in err  # 保留异常类型，别把环境问题伪装成别的东西


def test_program_bug_is_not_swallowed(monkeypatch, capsys):
    """ValueError 这类内部错误必须继续暴露 —— 把它说成"用户写错了"更贵。"""
    import kernoscope.cli as cli

    def boom(*a, **k):
        raise ValueError("internal invariant broken")

    monkeypatch.setattr(cli, "_emit", boom)
    with pytest.raises(ValueError, match="internal invariant broken"):
        main(list(SYNTH) + ["--ks-gemm", "8,8,8"])


def test_keyboard_interrupt_is_not_swallowed(monkeypatch):
    import kernoscope.cli as cli

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_emit", boom)
    with pytest.raises(KeyboardInterrupt):
        main(list(SYNTH) + ["--ks-gemm", "8,8,8"])


def test_usage_error_still_exits_two():
    """argparse 用法错误保持原约定（SystemExit(2)），不能被窄兜底改掉。"""
    with pytest.raises(SystemExit) as exc:
        main(["--definitely-not-a-flag"])
    assert exc.value.code == 2


def test_quiet_never_swallows_an_error_line(tmp_path, capsys):
    """`--quiet` 是"别啰嗦"，不是"别报错"（第五轮 V-25 的延伸）。

    脚本/agent 只读 stderr：rc≠0 却一行线索都没有，等于把排查成本推给调用方。
    """
    from kernoscope.cli import main

    rc = main(["--ks-synthetic", "--ks-no-nvml", "--quiet", "-f", "-o", str(tmp_path / "rep"),
               "--ks-calibration", str(tmp_path / "nope.json")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err, f"--quiet 把错误吞了：{err!r}"
    assert "Traceback (most recent call last)" not in err


def test_plan_with_a_subcommand_name_is_rejected_with_guidance(capsys):
    """`kernoscope --plan skill --apply`：`skill` 是位置参数（要运行的程序）而不是子命令。

    计划会说"运行目标 = skill --apply"，而把 `--plan` 去掉之后它真的会 exec 它 ——
    语义差这么多必须当场拒绝（V-18 报告第 1 点），而不是给一份指错东西的计划。
    """
    from kernoscope.cli import main

    rc = main(["--plan", "skill", "--apply"])
    err = capsys.readouterr().err
    assert rc == 2, rc
    assert "位置参数" in err and "skill --help" in err
    assert "Traceback (most recent call last)" not in err

    # 正常的 `--plan -- <程序>` 不受影响
    rc2 = main(["--plan", "--ks-synthetic", "--", "true"])
    assert rc2 == 0
