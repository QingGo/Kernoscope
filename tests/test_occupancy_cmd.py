"""探针记录解析 + `kernoscope occupancy` 子命令 + 与 --ks-achieved-occupancy 的互操作。"""

import json

import pytest

from kernoscope.analyze.occupancy import (analyze_probe_records, load_probe_records,
                                          to_injection_json)
from kernoscope.cli import main

REC = """# ks-occupancy-records v1
# kernel k_occ
# arch sm_89
# threads_per_block 128
# sm_count 1
# grid_blocks 3
# theoretical_occupancy_pct 25.0
smid,g0,g1
0,1000,2000
0,1000,2000
0,1000,2000
"""


def _write(tmp_path, text=REC, name="occ.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_records_parses_header_and_rows(tmp_path):
    rec = load_probe_records(str(_write(tmp_path)))
    assert rec.kernel == "k_occ"
    assert rec.arch == "sm_89"
    assert rec.threads_per_block == 128
    assert rec.sm_count == 1
    assert len(rec.records) == 3


def test_load_records_rejects_missing_format_marker(tmp_path):
    with pytest.raises(ValueError, match="ks-occupancy-records v1"):
        load_probe_records(str(_write(tmp_path, "0,1,2\n")))


def test_load_records_rejects_empty_and_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="没有任何 block 记录"):
        load_probe_records(str(_write(tmp_path, "# ks-occupancy-records v1\nsmid,g0,g1\n")))
    with pytest.raises(ValueError, match="期望 'smid,g0,g1'"):
        load_probe_records(str(_write(
            tmp_path, "# ks-occupancy-records v1\nsmid,g0,g1\n0,1\n")))


def test_analyze_matches_interval_overlap(tmp_path):
    rec = load_probe_records(str(_write(tmp_path)))
    out = analyze_probe_records(rec, arch="sm_89", sm_count=1, threads_per_block=128)
    assert out["avg_concurrent_blocks"] == pytest.approx(3.0)
    assert out["occupancy_pct"] == pytest.approx(25.0)   # 3 blocks × 4 warps / 48
    assert out["theoretical_occupancy_pct"] == pytest.approx(25.0)


def test_cli_writes_injection_json(tmp_path, capsys):
    src = _write(tmp_path)
    dst = tmp_path / "occ.json"
    rc = main(["occupancy", "--from", str(src), "--write-json", str(dst)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "区间重叠法" in out and "[measured]" in out
    payload = json.loads(dst.read_text())
    assert payload["k_occ"]["occupancy_pct"] == pytest.approx(25.0)

    # JSON 形状必须能被 --ks-achieved-occupancy 消费
    assert set(payload["k_occ"]) >= {"occupancy_pct", "avg_concurrent_blocks",
                                     "peak_concurrent_blocks"}


def test_cli_json_mode_and_bad_input(tmp_path, capsys):
    src = _write(tmp_path)
    assert main(["occupancy", "--from", str(src), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kernel"] == "k_occ"

    bad = _write(tmp_path, "garbage\n", name="bad.txt")
    assert main(["occupancy", "--from", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_end_to_end_injection_into_report(tmp_path, capsys):
    """探针结果 → JSON → 报告里出现 measure 类指标（闭环）。"""
    src = _write(tmp_path)
    dst = tmp_path / "occ.json"
    # 探针测的是 k_occ，而报告里是被 profile 的 kernel：用 --kernel 显式声明对应关系
    main(["occupancy", "--from", str(src), "--kernel", "synthetic_gemm",
          "--write-json", str(dst)])
    capsys.readouterr()

    base = tmp_path / "rep"
    rc = main(["--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f",
               "--ks-achieved-occupancy", str(dst), "-k", "regex:gemm"])
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    metrics = {m["name"]: m for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    assert "meas__achieved_occupancy_pct" in metrics
    assert metrics["meas__achieved_occupancy_pct"]["provenance"] == "measured"


def test_substring_fallback_matches_simple_probe_name(tmp_path):
    """探针写简单名、报告是 demangled 签名时，子串匹配应当生效。"""
    from kernoscope.report import KernelReport
    from kernoscope.sections import _lookup_in

    kr = KernelReport("k_occ(float*, int)", alias="k_occ")
    assert _lookup_in({"k_occ": {"occupancy_pct": 25.0}}, kr) == {"occupancy_pct": 25.0}
