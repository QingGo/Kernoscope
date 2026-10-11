"""CLI 端到端测试（用 --ks-synthetic，无需 GPU）。"""

import io
import json
import sys
from pathlib import Path

import pytest

from kernoscope.cli import main


def run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def report_part(text: str) -> str:
    """取出完整报告部分（一屏在它前面）。

    一屏（W4）默认铺在报告前面，所以"按行解析报告"的地方要先按这个分隔横幅切开。
    横幅本身是稳定接口（README 里也写了），所以这里可以依赖它。
    """
    marker = "完整报告（上面的判词都出自这里的数据"
    if marker in text:
        return text.split(marker, 1)[1].split("\n", 2)[2]
    return text


def test_synthetic_text_output(capsys):
    rc, out, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic")
    assert rc == 0
    assert "Section: LaunchStats" in out
    assert "launch__registers_per_thread" in out
    assert "[exact]" in out
    # 计数器指标必须显示 n/a + 原因，且不能有数字
    assert "sm__maximum_warps_per_active_cycle_pct" in out
    assert "n/a" in out
    assert "Kernel: synthetic_gemm" in out


def test_unavailable_never_carries_a_number(capsys):
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "full", "--csv")
    assert rc == 0
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(out)))
    bad = [r for r in rows if r["provenance"] == "unavailable" and r["value"] != ""]
    assert not bad, f"unavailable rows must have empty value: {bad[:3]}"
    # CSV 必须有 provenance 列（ncu 没有，这是我们的诚实性附加项）
    assert "provenance" in rows[0]


def test_strict_fails_when_requesting_counter_metrics(capsys):
    rc, _, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--strict",
                     "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert rc == 2
    assert "strict" in err
    assert "counters" in err


def test_strict_fails_on_noop_flags(capsys):
    rc, _, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--strict",
                     "--replay-mode", "kernel")
    assert rc == 2
    assert "no-op" in err


def test_lenient_mode_reports_noop_as_warning(capsys):
    rc, _, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--replay-mode", "kernel")
    assert rc == 0
    assert "no effect" in err


def test_kernel_filter_and_launch_count(capsys):
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:elementwise",
                     "-c", "3", "--print-summary", "per-kernel")
    assert rc == 0
    assert "synthetic_elementwise" in out
    assert "synthetic_gemm" not in out
    # -c 3 只保留 3 次 launch
    line = [l for l in report_part(out).splitlines() if "synthetic_elementwise" in l][0]
    assert line.split()[1] == "3"  # launches 列


def test_kernel_filter_no_match_lists_candidates(capsys):
    rc, _, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:does_not_exist")
    assert rc == 1
    assert "no kernels matched" in err
    assert "synthetic_gemm" in err


def test_export_writes_json_and_perfetto(capsys, tmp_path: Path):
    base = tmp_path / "rep"
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f")
    assert rc == 0
    js = Path(str(base) + ".ksreport.json")
    pf = Path(str(base) + ".perfetto.json")
    assert js.exists() and pf.exists()

    doc = json.loads(js.read_text())
    assert doc["schema_version"] == 1
    assert doc["tool"] == "kernoscope"
    assert any(k["name"].startswith("synthetic_gemm") for k in doc["kernels"]), \
        "默认展示名是 demangled 签名（与 ncu 一致）"

    trace = json.loads(pf.read_text())
    xs = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    assert len(xs) == 25  # 5 + 20 次 launch
    assert all("dur" in e and "ts" in e for e in xs)


def test_import_roundtrip(capsys, tmp_path: Path):
    base = tmp_path / "rep"
    run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f")
    js = Path(str(base) + ".ksreport.json")
    rc, out, _ = run(capsys, "-i", str(js), "--print-summary", "per-kernel")
    assert rc == 0
    assert "synthetic_gemm" in out


def test_launch_csv_is_one_row_per_launch(capsys, tmp_path: Path):
    """R-04/B4 的另一种读法：每个 launch 一行，形状扫描直接落表。"""
    import csv
    import io

    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:elementwise",
                     "--launch-csv")
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 20, "synthetic_elementwise 有 20 次 launch"
    assert rows[0]["grid_x"] and rows[0]["duration_ns"]
    assert rows[0]["provenance"] == "exact"
    assert rows[0]["config"].startswith("grid")


def test_missing_nsys_degrades_to_device_only(capsys, monkeypatch):
    """W1：缺 nsys 不是错误，而是退到设备级模式。

    旧行为（R-19）是一行可读错误 + exit 1。新要求是：**必须仍然成功**，并说清
    「这次少了什么、怎么补上」。exit 1 会让 CI/agent 以为工具坏了。
    """
    from kernoscope.backends import nsys as nsys_backend

    monkeypatch.setattr(nsys_backend, "find_nsys", lambda: None)
    monkeypatch.setattr(nsys_backend, "device_info_from_smi", lambda: __import__(
        "kernoscope.report", fromlist=["DeviceInfo"]).DeviceInfo())
    rc, out, err = run(capsys, "--ks-no-nvml", "--", "python", "-c", "pass")
    assert rc == 0
    assert "Traceback" not in err
    assert "nsys" in err                     # 说清缺什么
    assert "设备级" in err                    # 说清退到了哪一层


def test_nsys_found_but_profile_fails_is_still_an_error(capsys, monkeypatch):
    """nsys 在场但采集失败 = 真错误：不能悄悄降级掩盖它。"""
    from kernoscope.backends import nsys as nsys_backend

    def boom(*a, **k):
        raise nsys_backend.NsysError("nsys profile failed (rc=1)")

    monkeypatch.setattr(nsys_backend, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(nsys_backend, "run_profile", boom)
    rc, out, err = run(capsys, "--", "python", "-c", "pass")
    assert rc == 1
    assert "Traceback" not in err
    assert "nsys" in err and "hint:" in err


def test_missing_sqlite_is_a_readable_error(capsys, tmp_path: Path):
    """--ks-sqlite 指向不存在的文件：一行错误 + 建议。"""
    rc, out, err = run(capsys, "--ks-sqlite", str(tmp_path / "nope.sqlite"))
    assert rc == 1
    assert "Traceback" not in err
    assert "error:" in err


def _calibration(tmp_path: Path, caliber: bool = True) -> Path:
    entry = {"tflops": 171.7, "size": 8192, "ms": 6.64}
    if caliber:
        entry.update({"acc_caliber": "bf16 (fp32 accumulate)",
                      "precision_settings": {"allow_fp16_reduced_precision_reduction": True,
                                             "fp32_precision": "none"},
                      "method": "torch.matmul, 3 iters best-of",
                      "caliber_spread_pct": 5.0, "caliber_distinguishable": True})
    cal = tmp_path / "machine-profile.json"
    cal.write_text(json.dumps({
        "gpu": "NVIDIA GeForce RTX 4090",
        "ts": "2026-10-08 22:00:00",
        "measured_dram": {"copy_gbs": 909.0, "read_gbs": 900.0, "write_gbs": 934.0},
        "measured_matmul_tflops": {"torch.bfloat16": entry,
                                   "torch.float16": dict(entry, tflops=168.7)},
    }))
    return cal


def test_roofline_with_gemm_and_calibration(capsys, tmp_path: Path):
    cal = _calibration(tmp_path, caliber=True)
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-calibration", str(cal),
                     "--ks-gemm", "4096,4096,4096", "--ks-dtype", "bf16", "-k", "regex:gemm",
                     "--page", "raw")
    assert rc == 0
    assert "est__achieved_tflops" in out
    assert "est__pct_of_measured_compute_peak" in out
    # R-06：峰值必须带来源文件与标定时间
    assert "machine-profile.json" in out and "2026-10-08" in out
    assert "171.7 TFLOP/s" in out


def test_roofline_refuses_to_compute_without_caliber(capsys, tmp_path: Path):
    """R-09 的负样本：标定缺 acc_caliber/precision_settings/method → 拒绝算拐点。

    真机证据（B9）：同一台机器相隔一天，两种脚本对"口径可区分性"给出相反判断
    （0.4% vs 5.0%），所以缺字段的峰值不能拿去判 roofline 拐点。带宽侧不受影响。
    """
    cal = _calibration(tmp_path, caliber=False)
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-calibration", str(cal),
                     "--ks-gemm", "4096,4096,4096", "--ks-dtype", "bf16", "-k", "regex:gemm")
    assert rc == 0
    assert "est__achieved_tflops" in out            # 行还在……
    assert "拒绝用该标定算 roofline 拐点" in out     # ……但是 n/a
    assert "est__achieved_gbs" in out               # 带宽侧照常给数
    assert "R-09" in out


def test_print_units_and_details_and_metric_name(capsys, monkeypatch):
    """B6：--print-units/--print-details/--print-metric-name 必须真的有效果。"""
    from kernoscope.backends.nvml import DeviceSample, SamplerResult

    class _Sat:
        def __init__(self, interval_s=0.2, enabled=True):
            self.result = SamplerResult(source="fake")
            self.result.samples = [DeviceSample(t=i * 0.2, mem_util_pct=96.8, sm_util_pct=99.0,
                                                sm_clock_mhz=2500.0, power_w=300.0)
                                   for i in range(12)]
            self.result.duration_s = 2.4

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("kernoscope.cli.Sampler", _Sat)
    rc, base, _ = run(capsys, "--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                      "--print-units", "base")
    assert rc == 0
    assert "MB" in base, "--print-units base 应保持原始单位（MB），不做 MB→GB 换挡"
    assert "2.11179e+06 MB" in base, "base 单位下应直接打原始数值 + 原始单位"
    rc_auto, auto, _ = run(capsys, "--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                           "--print-units", "auto")
    assert "TB" in auto, "--print-units auto 应自适应换挡"

    rc2, hdr, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                      "--print-details", "header")
    assert rc2 == 0
    assert "Section: LaunchStats" in hdr
    assert "launch__registers_per_thread" not in hdr, "--print-details header 不铺开指标"

    rc3, full, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                       "--print-metric-name", "full")
    assert rc3 == 0
    assert "LaunchStats.launch__registers_per_thread" in full


def test_metrics_flag_actually_filters(capsys):
    """R-21/B7-d 的负样本：--metrics 必须真的筛选输出，但仍要显示 n/a 行。"""
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                     "--metrics", "launch__grid_size")
    assert rc == 0
    lines = [l for l in out.splitlines() if "launch__registers_per_thread" in l]
    assert not lines, "--metrics 未生效时会把所有指标都打出来"
    assert "launch__grid_size" in out
    rc2, out2, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                       "--metrics", "dram__bytes.sum")
    assert "n/a" in out2, "被筛掉的 n/a 行也要保留（拒绝必须可见）"


def test_devices_flag_filters_by_device_id(capsys):
    """--devices 是真实现，不是"接受但忽略"。"""
    rc, _, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                     "--devices", "3")
    assert rc == 1
    assert "--devices" in err and "0" in err


def test_achieved_occupancy_injection(capsys, tmp_path: Path):
    probe = tmp_path / "occ.json"
    probe.write_text(json.dumps({
        "synthetic_gemm": {"occupancy_pct": 81.0, "avg_concurrent_blocks": 9.72,
                           "peak_concurrent_blocks": 12},
    }))
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-achieved-occupancy", str(probe),
                     "-k", "regex:gemm")
    assert rc == 0
    assert "meas__achieved_occupancy_pct" in out
    assert "[measured]" in out


def test_list_sets(capsys):
    rc, out, _ = run(capsys, "--list-sets")
    assert rc == 0
    assert "basic" in out and "LaunchStats" in out


# ---------------------------------------------------------------------------
# 回归：argparse 会对 help 文本做 % 插值，help 里出现裸 % 会让 --help 直接崩。
# 这类 bug 常规测试看不见（不会去调 --help），但用户第一条命令就是它。
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from kernoscope.cli import main  # noqa: E402


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["doctor", "--help"],
    ["occupancy", "--help"],
    ["--list-sets"],
])
def test_help_output_does_not_crash(argv, capsys):
    try:
        rc = main(argv)
    except SystemExit as e:          # argparse 正常退出
        rc = e.code or 0
    out = capsys.readouterr()
    assert rc in (0, None), f"{argv} 退出码异常: {rc}"
    assert (out.out + out.err).strip(), f"{argv} 没有输出"


def test_top_level_help_lists_every_subcommand(capsys):
    """顶层 `--help` 必须列出全部子命令 —— 一次易用性实测发现 12,993 B 的帮助里
    `route`/`capabilities`/`gate`/`skill` 各出现 **0 次**：四个核心入口在帮助里不存在，
    陌生人（和 agent）只能先读 README 才知道它们。

    这条同时钉住"加子命令必须写一句说明"：`SUBCOMMAND_HELP` 的键集合 == `SUBCOMMANDS`。
    """
    from kernoscope.cli import SUBCOMMAND_HELP, SUBCOMMANDS

    assert set(SUBCOMMAND_HELP) == set(SUBCOMMANDS), (
        f"子命令说明与子命令表不一致：只在表里 {sorted(set(SUBCOMMANDS) - set(SUBCOMMAND_HELP))}，"
        f"只在说明里 {sorted(set(SUBCOMMAND_HELP) - set(SUBCOMMANDS))}")

    try:
        main(["--help"])
    except SystemExit as e:          # argparse 正常退出
        assert (e.code or 0) == 0
    text = capsys.readouterr().out
    missing = [s for s in SUBCOMMANDS if f"kernoscope {s} " not in text]
    assert not missing, f"顶层 --help 里找不到这些子命令：{missing}"
    for name, desc in SUBCOMMAND_HELP.items():
        assert desc in text, f"--help 里 {name} 的一句说明丢了：{desc!r}"


HIST_FOR_STRICT = """kernel 0 - synthetic_gemm - #thread-blocks 1048576,  \
kernel instructions 2000000, total instructions 2000000
  LDG.E.128 = 500000
  STG.E.128 = 500000
  IMAD = 1000000
"""


def test_strict_section_gate_is_data_driven(capsys, tmp_path: Path):
    """R-43 的负样本：门禁要按「真的产出了数据」判定，而不是按 section 名字判定。

    从前 `--section SchedulerStats` 一旦出现就报「needs hardware counters」，
    即使 `--ks-instr-hist` 已经把 est__issue_slot_utilization_pct 算出来了 ——
    `--strict` 在拦自己的成果，新增的 R-08 能力在严格模式下根本不可用。
    """
    hist = tmp_path / "hist.txt"
    hist.write_text(HIST_FOR_STRICT, encoding="utf-8")
    common = ["--ks-synthetic", "-k", "regex:gemm", "--section", "SchedulerStats"]

    # 没有直方图：确实拿不到 → 必须红
    rc_none, out_none, err_none = run(capsys, *common, "--strict")
    assert rc_none == 2
    assert "SchedulerStats" in err_none

    # 有直方图：数据真的产出了 → 不能再红
    rc_ok, out_ok, err_ok = run(capsys, *common, "--ks-instr-hist", str(hist),
                                "--ks-clock-mhz", "2000", "--strict")
    assert rc_ok == 0, err_ok
    assert "est__issue_slot_utilization_pct" in out_ok
    assert "needs hardware counters" not in err_ok


# ---------------------------------------------------------------------------
# M14「首用 ≤1 屏」：默认在**终端**上只给一屏，管道/重定向仍是完整报告。
# 为什么：README 承诺"一屏"，而 auto 的语义是"一屏 + 完整报告"（实测 21.6 KB / 200+ 行）。
# ---------------------------------------------------------------------------
class _FakeStdout:
    def __init__(self, tty: bool):
        self._buf = io.StringIO()
        self._tty = tty

    def write(self, s):        # print() 只要求 write/flush
        self._buf.write(s)

    def flush(self):
        pass

    def isatty(self):
        return self._tty

    def getvalue(self):
        return self._buf.getvalue()


def _run_with_stdout(tty: bool, *argv):
    fake = _FakeStdout(tty)
    real = sys.stdout
    sys.stdout = fake                      # type: ignore[assignment]
    try:
        try:
            rc = main(list(argv))
        except SystemExit as e:            # argparse 正常退出
            rc = e.code or 0
    finally:
        sys.stdout = real
    return rc, fake.getvalue()


def test_default_stdout_on_a_terminal_is_one_screen():
    rc, out = _run_with_stdout(True, "--ks-synthetic", "--set", "basic")
    assert rc == 0
    assert len(out.encode("utf-8")) <= 8192, f"终端上默认 {len(out.encode('utf-8'))} B 不是一屏"
    assert "结论：" in out or "发现" in out


def test_piped_stdout_keeps_the_full_report():
    """机器格式与日志一个字都不该变：管道里仍然是"一屏 + 完整报告"。"""
    rc, out = _run_with_stdout(False, "--ks-synthetic", "--set", "basic")
    assert rc == 0
    assert len(out.encode("utf-8")) > 8192, "管道里被缩成一屏了（脚本/CI 会少东西）"


def test_explicit_screen_auto_keeps_the_old_behaviour_even_on_a_terminal():
    rc, out = _run_with_stdout(True, "--ks-synthetic", "--set", "basic", "--screen", "auto")
    assert rc == 0
    assert len(out.encode("utf-8")) > 8192, "显式 --screen auto 必须保持老行为"


def test_explicit_screen_off_prints_only_the_report():
    rc, out = _run_with_stdout(True, "--ks-synthetic", "--set", "basic", "--screen", "off")
    assert rc == 0
    assert len(out.encode("utf-8")) > 8192


# ---------------------------------------------------------------------------
# N9：单文件 HTML 证据单（`--html`）—— 自包含、离线、零 JS、出处角标与数字同行
# ---------------------------------------------------------------------------
def test_html_flag_writes_a_self_contained_evidence_sheet(tmp_path):
    out = tmp_path / "evidence.html"
    rc, _ = _run_with_stdout(False, "--ks-synthetic", "--set", "basic", "--screen", "off",
                            "--html", str(out))
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")
    assert "http://" not in text and "https://" not in text, "自包含：不许有外部 URL"
    assert "<script" not in text, "零 JS：打开的是一份证据，不是应用"
    assert "<style" in text
    assert "badge" in text, "每个数字要带出处角标"


def test_html_dash_writes_to_stdout_but_not_together_with_packet_dash():
    rc, out = _run_with_stdout(False, "--ks-synthetic", "--set", "basic", "--screen", "off",
                              "--html", "-")
    assert rc == 0 and out.startswith("<!DOCTYPE html>")

    rc2, _ = _run_with_stdout(False, "--ks-synthetic", "--set", "basic",
                              "--packet", "-", "--html", "-")
    assert rc2 == 2, "stdout 上一次只能要一种格式"


def test_plan_declares_the_html_file(tmp_path):
    from kernoscope.cli import build_parser
    from kernoscope.effects import plan_for

    out = tmp_path / "evidence.html"
    argv = ["--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--screen", "off",
            "--html", str(out)]
    plan = plan_for(build_parser().parse_args(argv), argv)
    assert str(out) in plan.declared_writes()
