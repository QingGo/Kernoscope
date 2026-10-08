"""CLI 端到端测试（用 --ks-synthetic，无需 GPU）。"""

import json
from pathlib import Path

import pytest

from kernoscope.cli import main


def run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


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
    line = [l for l in out.splitlines() if "synthetic_elementwise" in l][0]
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


def test_roofline_with_gemm_and_calibration(capsys, tmp_path: Path):
    cal = tmp_path / "machine-profile.json"
    cal.write_text(json.dumps({
        "gpu": "NVIDIA GeForce RTX 4090",
        "measured_dram": {"copy_gbs": 909.0, "read_gbs": 900.0, "write_gbs": 934.0},
        "measured_matmul_tflops": {"torch.bfloat16": {"tflops": 171.7},
                                   "torch.float16": {"tflops": 168.7}},
    }))
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-calibration", str(cal),
                     "--ks-gemm", "4096,4096,4096", "--ks-dtype", "bf16", "-k", "regex:gemm",
                     "--page", "raw")
    assert rc == 0
    assert "est__achieved_tflops" in out
    assert "est__pct_of_measured_compute_peak" in out
    assert "measured peak 171.7" in out


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
