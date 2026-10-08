"""doctor 自检：能力判定、provenance 一致性、退出码。"""

import json

import pytest

from kernoscope import doctor as D
from kernoscope.cli import main
from kernoscope.provenance import UNAVAILABLE_REASONS


def test_decode_capabilities():
    docker_default = D.decode_capabilities("00000000a80425fb")
    assert "DAC_OVERRIDE" in docker_default
    assert "CHOWN" in docker_default
    assert "SYS_ADMIN" not in docker_default
    assert "PERFMON" not in docker_default

    full = D.decode_capabilities("000000c000200000")  # bit21 SYS_ADMIN + bit38 PERFMON
    assert {"SYS_ADMIN", "PERFMON"} <= full
    assert D.decode_capabilities("not-hex") == set()


def test_parse_nv_params():
    text = "ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\nDeviceFileMode: 0666\n"
    p = D.parse_nv_params(text)
    assert p["RmProfilingAdminOnly"] == "1"
    assert D.parse_proc_status("CapEff:\t00000000a80425fb\nCapBnd:\t00000000a80425fb\n")["CapEff"] \
        == "00000000a80425fb"


@pytest.fixture
def no_tools(monkeypatch):
    monkeypatch.setattr(D, "find_tool", lambda name: None)
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda: (None, "未安装 ncu，无法实测判定"))


def test_doctor_fails_without_nsys(no_tools, capsys):
    rep = D.run_doctor(probe_counters=False)
    assert not rep.ok
    assert any(c.name == "nsys" and c.status == D.FAIL for c in rep.checks)
    rc = D.doctor_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """构造一个"受限容器"环境：ncu 被计数器门挡住、nsys/cubin/nvidia-smi 都在。"""
    tools = {"nsys": "/usr/local/bin/nsys", "cuobjdump": "/usr/local/cuda/bin/cuobjdump",
             "nvidia-smi": "/usr/bin/nvidia-smi", "ncu": "/usr/local/cuda/bin/ncu"}
    monkeypatch.setattr(D, "find_tool", lambda name: tools.get(name))
    monkeypatch.setattr(D, "tool_version", lambda cmd, timeout=20.0: "2025.3.1")
    monkeypatch.setattr(D, "gpu_info", lambda: {
        "name": "NVIDIA GeForce RTX 4090", "driver": "580.105", "compute_capability": "8.9",
        "memory_mib": "24564", "max_sm_clock_mhz": "3105", "power_limit_w": "450.00",
        "count": 1, "sm_count": 128})
    monkeypatch.setattr(D, "counter_gate_probe",
                        lambda: (False, "ncu reports ERR_NVGPUCTRPERM (empirically confirmed)"))
    params = tmp_path / "params"
    params.write_text("ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\n")
    status = tmp_path / "status"
    status.write_text("CapEff:\t00000000a80425fb\nCapBnd:\t00000000a80425fb\n")
    monkeypatch.setenv("KS_NV_PARAMS", str(params))
    monkeypatch.setenv("KS_PROC_STATUS", str(status))
    monkeypatch.setenv("KS_PERF_PARANOID", str(tmp_path / "nope"))
    monkeypatch.delenv("NVBIT_PATH", raising=False)


def test_doctor_reports_restricted_counters_honestly(fake_env, capsys):
    rep = D.run_doctor()
    assert rep.ok, "核心数据源（nsys）在，就不应判定失败"

    # 驱动策略被识别，且与权限分开陈述
    assert any(c.name == "内核驱动策略" and "1" in c.detail for c in rep.checks)
    cap_checks = {c.name: c for c in rep.checks if c.name.startswith("CAP_")}
    assert cap_checks["CAP_SYS_ADMIN"].status == D.FAIL
    assert "边界集" in cap_checks["CAP_SYS_ADMIN"].detail

    # 计数器判定为不可用，且原因与报告侧完全一致（同一处措辞）
    counter_check = [c for c in rep.checks if c.name.startswith("硬件计数器")][0]
    assert counter_check.status == D.FAIL
    assert counter_check.impact == UNAVAILABLE_REASONS["counters"]

    caps = {c.provenance: c for c in rep.capabilities}
    assert set(caps) == {"exact", "measured", "derived", "unavailable"}
    assert any("LaunchStats" in i for i in caps["exact"].items)
    assert caps["unavailable"].reason == UNAVAILABLE_REASONS["counters"]
    assert any("nsys" in p for p in rep.backend_plan)

    rc = D.doctor_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["schema_version"] == 1
    assert payload["device"]["name"].startswith("NVIDIA")
    assert payload["capabilities"]


def test_doctor_flags_sysadmin_present_as_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "find_tool", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda: (True, "ok"))
    mask = (1 << 21) | (1 << 38)          # CAP_SYS_ADMIN + CAP_PERFMON
    st = tmp_path / "status"
    st.write_text(f"CapEff:\t{mask:016x}\nCapBnd:\t{mask:016x}\n")
    monkeypatch.setenv("KS_PROC_STATUS", str(st))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    rep = D.run_doctor()
    ok = {c.name for c in rep.checks if c.status == D.OK}
    assert "CAP_SYS_ADMIN" in ok and "CAP_PERFMON" in ok


def test_cli_dispatches_doctor_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(D, "find_tool", lambda name: None)
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda: (None, "n/a"))
    rc = main(["doctor", "--no-probe"])
    out = capsys.readouterr().out
    assert "环境自检" in out
    assert rc == 1  # 无 nsys → 核心路径不可用


def test_unknown_probe_does_not_claim_restriction(monkeypatch, tmp_path):
    """没装 ncu 时只能说"无法判定"，不得套用"受限"的样板原因（避免错误归因）。"""
    monkeypatch.setattr(D, "find_tool", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda: (None, "未安装 ncu，无法实测判定"))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    chk = [c for c in rep.checks if c.name.startswith("硬件计数器")][0]
    assert chk.status == D.WARN
    assert chk.impact != UNAVAILABLE_REASONS["counters"]
    assert "无法判定" in chk.impact


def test_available_counters_move_to_exact_not_unavailable(monkeypatch, tmp_path):
    """若计数器真的可用，能力矩阵不应把它列为 unavailable。"""
    monkeypatch.setattr(D, "find_tool", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda: (True, "ok"))
    params = tmp_path / "params"
    params.write_text("RmProfilingAdminOnly: 0\n")
    monkeypatch.setenv("KS_NV_PARAMS", str(params))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    conv = {c.provenance: c for c in rep.capabilities}
    assert "unavailable" not in conv, "计数器可用时不该出现 unavailable 能力"
    assert any("MemoryWorkloadAnalysis" in i for i in conv["exact"].items)
