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


# --------------------------------------------------------------------------
# W-2（第六轮）＋ 一次易用性实测：**模块级**函数拿不到 `_main` 的 `log_error` 闭包
# --------------------------------------------------------------------------
# `collect()` / `_attach_truth()` / `_export_substitute()` 都是模块级函数。它们历史上直接调
# `log_error(...)`，于是「没给目标命令 / 忘记写 `--` / 拼错子命令」这三条**最高频**路径抛
# `NameError` —— 源码里已经写好的那句提示永远打不出来。这里把每条路径钉住：
# rc 固定 + 有自家 `error:` 行 + **没有** traceback（有 traceback 就说明又有人直接调闭包了）。
@pytest.mark.parametrize("tag,argv", [
    ("没有参数", ()),
    ("给了 -o 但没给目标命令", ("-o", "rep.json")),
    ("忘了写 --", ("--set", "basic")),
    ("拼错子命令（`doctr` 被当成要运行的程序）", ("doctr",)),
])
def test_module_level_paths_never_raise_nameerror(capsys, tag, argv):
    rc, out, err = run(capsys, *argv)
    assert_own_error(rc, out, err)
    assert "no target command" in err or "无法执行" in err, \
        f"{tag}: 打出来的不是准备好的文案：{err!r}"


def test_no_target_command_error_survives_quiet(capsys):
    """同一路径加 `--quiet`：错误行必须照样出现（靠 `_main` 把 log_error 挂在 log 上）。"""
    rc, out, err = run(capsys, "--quiet")
    assert_own_error(rc, out, err)
    assert "no target command" in err, err


def test_export_substitute_empty_launch_table_has_its_own_error(monkeypatch, capsys,
                                                                 tmp_path: Path):
    """W-2 的复现值：空 launch 表 + `--export-ncu-rep`（受限机主推路径）。

    **不带** `--ks-synthetic`：带了就永远有 kernel，那个分支就全仓零覆盖（第六轮报告原话）。

    ⚠️ 这里把 nsys 后端**打桩**成"装了 nsys，但这份采集里一个 kernel 都没有"（X-3）：
    旧版本直接跑 `-- true`，于是结果取决于**机器上有没有 nsys** —— 没有 nsys 时走到降级路径
    （表为空），有 nsys 时 nsys 会先报 sqlite 里没有 `CUPTI_ACTIVITY_KIND_KERNEL` 表 ⇒
    同一个用例在两种机器上测的是**两条不同的路径**（验收方在真机上就红在这条）。
    打桩后它测的仍然是"空 launch 表 + 导出"这条产品分支，且与机器无关。
    """
    from kernoscope.backends import nsys as N

    # 机器无关地拿到"零 kernel 但仍然继续跑"的状态：**没有 nsys** ⇒ 走设备级降级路径
    monkeypatch.setattr(N, "find_nsys", lambda: None)

    rc, out, err = run(capsys, "--export-ncu-rep", str(tmp_path / "x.ncu-rep"), "--", "true")
    assert_own_error(rc, out, err)
    assert "没有可导出的 kernel" in err, err


def test_nsys_present_but_zero_kernels_has_its_own_error(monkeypatch, capsys, tmp_path: Path):
    """X-3：**装了 nsys** 但这次采集里一个 kernel 都没有 —— 它有自己的错误行。

    为什么单列一条：旧用例把两种机器状态混在一起（见上一条的说明），于是"有 nsys"时它红在
    sqlite/CUPTI 那条分支上。两条路径的产品行为**都**要能红、都要机器无关地测：
    这一条打桩成"nsys 装了，`parse_kernels` 返回空"，断言它的自家错误 + rc=1 + 无 traceback。
    """
    from kernoscope.backends import nsys as N

    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile", lambda *a, **k: str(tmp_path / "fake.nsys-rep"))
    monkeypatch.setattr(N, "export_sqlite", lambda *a, **k: str(tmp_path / "fake.sqlite"))
    monkeypatch.setattr(N, "parse_kernels", lambda *a, **k: [])

    rc, out, err = run(capsys, "--export-ncu-rep", str(tmp_path / "x.ncu-rep"), "--", "true")
    assert_own_error(rc, out, err)
    assert "没有采到任何 kernel 活动" in err, err


def test_attach_truth_says_its_own_error_instead_of_nameerror(tmp_path, capsys):
    """`_attach_truth()` 的失败分支：读不了 `.ncu-rep` ⇒ 自家 error 行 + 结构化拒绝。

    直接调模块级函数（不走 CLI），因为这里要钉的是"**这个函数**不再依赖闭包"。
    """
    import json

    from kernoscope import cli

    class _Rep:
        meta: dict = {}
        kernels: list = []

    rep = _Rep()
    logged: list[str] = []
    cli._attach_truth(rep, str(tmp_path / "nope.ncu-rep"), logged.append)

    assert logged and "读不了" in logged[-1], logged
    assert "error:" in logged[-1], logged
    dumped = json.dumps(rep.meta, ensure_ascii=False)
    assert "ncu_report_unreadable" in dumped, dumped


def test_log_error_helper_prefers_the_real_error_sink():
    """`_log_error` 在 `log` 上挂了 `log_error` 时用它（否则 `--quiet` 会吞掉错误）。"""
    from kernoscope.cli import _log_error

    seen: list[tuple[str, str]] = []

    def log(msg):
        seen.append(("log", msg))

    log.log_error = lambda msg: seen.append(("error", msg))   # type: ignore[attr-defined]
    _log_error(log, "error: x")
    assert seen == [("error", "error: x")], seen

    plain: list[tuple[str, str]] = []
    _log_error(lambda m: plain.append(("log", m)), "error: y")   # 没挂上也不许抛
    assert plain == [("log", "error: y")], plain


# --------------------------------------------------------------------------
# 第七轮 X-1：**有 nsys 的机器上**，目标程序的问题被报成"nsys 的问题"
# --------------------------------------------------------------------------
# 机制：目标不存在/非零退出都从 `NsysError("nsys profile failed (rc=…)")` 出来，CLI 再挂上
# "nsys 随 CUDA Toolkit 提供（或单独安装）"的提示 —— 把人送去装一个已经装好的工具，
# 而真实原因（目标失败）一个字都没说。真机实测三例，macOS（无 nsys）全对、真机全错。
def test_typo_target_with_nsys_present_is_still_a_target_error(monkeypatch, capsys):
    from kernoscope.backends import nsys as N

    ran: list[str] = []
    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile", lambda *a, **k: ran.append("ran") or "x.nsys-rep")

    rc, out, err = run(capsys, "doctr")          # 拼错的子命令被当成要运行的程序
    assert rc == 1
    assert "目标命令无法执行" in err and "doctr" in err, err
    assert "nsys" not in err, f"把目标的问题讲成了 nsys 的问题：{err!r}"
    assert ran == [], "采集根本不该开始（目标已经解析失败）"


def test_missing_input_file_is_reported_before_collection(monkeypatch, capsys):
    """X-1④：输入文件的存在性要在采集**之前**判 —— 否则会被采集错误盖住。"""
    from kernoscope.backends import nsys as N

    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile",
                        lambda *a, **k: (_ for _ in ()).throw(N.NsysError("nsys died")))
    rc, out, err = run(capsys, "--ks-calibration", "/tmp/definitely-nope.json", "--", "true")
    assert rc == 1
    assert "标定文件不存在" in err and "definitely-nope.json" in err, err
    assert "nsys" not in err, f"输入错误被采集错误盖住了：{err!r}"


def test_target_nonzero_exit_is_reported_as_the_targets_code(monkeypatch, capsys):
    """X-1②：nsys 把目标的退出码透传回来 ⇒ 报"目标命令退出码 N"，不是"nsys 失败"。"""
    from kernoscope.backends import nsys as N

    def boom(*a, **k):
        raise N.TargetFailed(3, rep_exists=False, detail="boom")

    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile", boom)
    rc, out, err = run(capsys, "--", "sh", "-c", "exit 3")
    assert rc == 1
    assert "目标命令退出码 3" in err, err
    assert "单独安装" not in err and "CUDA Toolkit" not in err, \
        f"目标自己退出 3，却给了装 nsys 的提示：{err!r}"
    assert "nsys" in err, "没有产出 .nsys-rep 时，次要可能（nsys 自身失败）要写清"


def test_target_failure_with_a_report_says_collection_is_still_usable(monkeypatch, capsys):
    from kernoscope.backends import nsys as N

    def boom(*a, **k):
        raise N.TargetFailed(2, rep_exists=True)

    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile", boom)
    rc, out, err = run(capsys, "--", "true")
    assert rc == 1
    assert "目标命令退出码 2" in err and "可以继续分析" in err, err


def test_nsys_own_failure_still_says_nsys(monkeypatch, capsys):
    """③ 只有真的是 nsys 自身失败（返回 0 却没有产物）才报 nsys。"""
    from kernoscope.backends import nsys as N

    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")

    def fake_run(args, capture_output=True, text=True):
        class P:
            returncode = 0
            stdout = "nsys said nothing"
            stderr = ""
        return P()

    monkeypatch.setattr(N.subprocess, "run", fake_run)
    with pytest.raises(N.NsysError) as e:
        N.run_profile(["true"], "/tmp/ks-nope-prefix")
    assert "nsys 没有产出 .nsys-rep" in str(e.value)
