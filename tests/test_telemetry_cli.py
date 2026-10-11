"""`--ks-telemetry` 的 CLI 面（N7）：把平台**已授权**的遥测变成报告里的一节。

三个必须成立的点（也是这一层做它的全部理由）：
  ① 外部量写**它自己的名字**、照抄端点的真单位，带端点署名；
  ② "比例/时间占比"绝不当字节（`DCGM_FI_PROF_DRAM_ACTIVE` 是 0–1 的占比）；
  ③ 缺席/读不到印 `n/a` + code，**不是 0**，也不让整次运行崩掉。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernoscope.cli import main

ENDPOINT = """# HELP DCGM_FI_DEV_POWER_USAGE power
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-abc"} 250.5
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 123456789
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.42
DCGM_FI_PROF_SM_OCCUPANCY{gpu="0"} 0.31
DCGM_FI_DEV_XID_ERRORS{gpu="0"} NaN
"""

SYNTH = ("--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--screen", "off")


def _run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def _telemetry_section(report_path: Path) -> dict:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    for k in rep["kernels"]:
        if k["name"] == "(device)":
            for s in k["sections"]:
                if s["name"] == "Telemetry":
                    return {m["name"]: m for m in s["metrics"]}
    return {}


def test_endpoint_readings_land_in_the_report_with_their_own_names_and_units(
        tmp_path, capsys):
    endpoint = tmp_path / "metrics.txt"
    endpoint.write_text(ENDPOINT, encoding="utf-8")
    out = tmp_path / "rep.json"

    rc, stdout, stderr = _run(capsys, "--ks-telemetry", str(endpoint), *SYNTH, "-o", str(out))
    assert rc == 0, (rc, stdout, stderr)
    metrics = _telemetry_section(tmp_path / "rep.ksreport.json")
    assert metrics, "报告里没有 Telemetry 节"
    assert metrics["meas__telemetry_dcgm_power_w"]["value"] == pytest.approx(250.5)
    assert metrics["meas__telemetry_dcgm_power_w"]["unit"] == "W"
    # 外部量用自己的名字：不借 ncu 计数器名
    assert all(n.startswith("meas__telemetry_") for n in metrics)


def test_ratio_fields_never_become_bytes_and_absent_fields_are_not_zero(tmp_path, capsys):
    endpoint = tmp_path / "metrics.txt"
    endpoint.write_text(ENDPOINT, encoding="utf-8")
    out = tmp_path / "rep.json"
    _run(capsys, "--ks-telemetry", str(endpoint), *SYNTH, "-o", str(out))
    metrics = _telemetry_section(tmp_path / "rep.ksreport.json")

    dram = metrics["meas__telemetry_dcgm_dram_active_ratio"]
    assert dram["unit"] == "ratio" and "byte" not in dram["name"].lower()
    # 端点里**没有**的 profiling 字段：缺席（unavailable + code），**不是** 0。
    # （没被请求、又不是 profiling 家族的字段（例如这条端点的 XID）按设计不进 missing ——
    #   "不往报告里塞噪声"；它们的原因仍记在 `TelemetryResult.skipped` 里。）
    rx = metrics.get("meas__telemetry_dcgm_pcie_rx_bytes_per_s")
    assert rx is not None and rx["value"] is None, metrics.keys()
    assert rx["provenance"] == "unavailable"
    assert (rx.get("refusal") or {}).get("code"), rx


def test_unreadable_endpoint_is_a_row_not_a_crash(tmp_path, capsys):
    out = tmp_path / "rep.json"
    rc, stdout, stderr = _run(capsys, "--ks-telemetry", str(tmp_path / "nope.txt"),
                             *SYNTH, "-o", str(out))
    assert rc == 0, (rc, stdout, stderr)          # 端点读不到不该让整次运行失败
    metrics = _telemetry_section(tmp_path / "rep.ksreport.json")
    assert metrics, "读不到端点时也该有一节说明'没读到'（而不是消失）"
    assert any(m["provenance"] == "unavailable" for m in metrics.values()), metrics


def test_plan_declares_the_telemetry_read(tmp_path):
    from kernoscope.cli import build_parser
    from kernoscope.effects import plan_for

    argv = [*SYNTH, "--ks-telemetry", "http://dcgm.example:9400/metrics"]
    plan = plan_for(build_parser().parse_args(argv), argv)
    ids = [i.op.id for i in plan.items]
    assert "read_telemetry" in ids, ids
    item = next(i for i in plan.items if i.op.id == "read_telemetry")
    assert item.op.external is True, "联网读必须是 external（--plan 的读者要知道这一点）"
    assert "联网" in item.detail
