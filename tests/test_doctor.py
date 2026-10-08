"""doctor 自检：能力判定、provenance 一致性、退出码、**工具多路径发现（R-16/R-17）**。"""

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


def _discovery(name, path, in_path=True, searched=None):
    return D.Discovery(name=name, path=path, in_path=in_path,
                       searched=list(searched or (["/fake/bin"] if path else ["/fake/bin",
                                                                               "/fake/cuda/bin"])))


@pytest.fixture
def no_tools(monkeypatch):
    monkeypatch.setattr(D, "discover_tool", lambda name: _discovery(name, None))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe",
                        lambda ncu_path=None: (None, "在所有候选路径里都没有找到 ncu"))


def test_doctor_fails_without_nsys(no_tools, capsys):
    rep = D.run_doctor(probe_counters=False)
    assert not rep.ok
    assert any(c.name == "nsys" and c.status == D.FAIL for c in rep.checks)
    rc = D.doctor_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    # R-17："未安装"必须可证伪 —— 找过哪些目录要留在产物里
    assert payload["counter_state"] == "not-installed"
    assert payload["tools"]["ncu"]["searched_dirs"]


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """构造一个"受限容器"环境：ncu 被计数器门挡住、nsys/cubin/nvidia-smi 都在。"""
    tools = {"nsys": "/usr/local/bin/nsys", "cuobjdump": "/usr/local/cuda/bin/cuobjdump",
             "nvidia-smi": "/usr/bin/nvidia-smi", "ncu": "/usr/local/cuda/bin/ncu"}
    monkeypatch.setattr(D, "discover_tool",
                        lambda name: _discovery(name, tools.get(name)))
    monkeypatch.setattr(D, "tool_version", lambda cmd, timeout=20.0: "2025.3.1")
    monkeypatch.setattr(D, "gpu_info", lambda: {
        "name": "NVIDIA GeForce RTX 4090", "driver": "580.105", "compute_capability": "8.9",
        "memory_mib": "24564", "max_sm_clock_mhz": "3105", "power_limit_w": "450.00",
        "count": 1, "sm_count": 128})
    monkeypatch.setattr(D, "counter_gate_probe",
                        lambda ncu_path=None: (False,
                                               "ncu reports ERR_NVGPUCTRPERM (empirically confirmed)"))
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
    # R-17：四态结论是"已判定被拒"，且给出了实际路径
    assert rep.counter_state == "denied"
    assert "/usr/local/cuda/bin/ncu" in rep.counter_state_note

    caps = {c.provenance: c for c in rep.capabilities}
    assert set(caps) == {"exact", "measured", "derived", "estimated", "unavailable"}
    assert any("LaunchStats" in i for i in caps["exact"].items)
    # R-13：doctor 的分级必须与报告里实际打出的标签一致
    assert any("DRAM" in i for i in caps["estimated"].items)
    assert not any("roofline" in i for i in caps["derived"].items), \
        "roofline 在报告里是 estimated，doctor 不得标成 derived"
    assert caps["unavailable"].reason == UNAVAILABLE_REASONS["counters"]
    assert any("nsys" in p for p in rep.backend_plan)

    rc = D.doctor_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["schema_version"] == 2
    assert payload["device"]["name"].startswith("NVIDIA")
    assert payload["capabilities"]
    assert payload["tools"]["cuobjdump"]["path"].endswith("cuobjdump")


def test_doctor_finds_cuda_bin_not_on_path(monkeypatch, tmp_path, capsys):
    """R-16 的负样本：CUDA bin 不在 PATH 时，doctor 仍必须发现 ncu/cuobjdump。"""
    found = {"nsys": "/usr/local/bin/nsys",
             "cuobjdump": "/usr/local/cuda-13.0/bin/cuobjdump",
             "nvidia-smi": "/usr/bin/nvidia-smi",
             "ncu": "/usr/local/cuda-13.0/bin/ncu"}
    monkeypatch.setattr(D, "discover_tool",
                        lambda name: _discovery(name, found.get(name), in_path=False))
    monkeypatch.setattr(D, "tool_version", lambda cmd, timeout=20.0: "13.0")
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe",
                        lambda ncu_path=None: (False, "ncu reports ERR_NVGPUCTRPERM "
                                                      "(empirically confirmed)"))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    cu = [c for c in rep.checks if c.name == "cuobjdump"][0]
    assert cu.status == D.OK, "装在 /usr/local/cuda-13.0/bin 的 cuobjdump 必须被发现"
    assert "13.0" in cu.detail
    assert rep.counter_state == "denied"          # 而不是"未安装"
    assert "未安装" not in rep.counter_state_note
    # 能力矩阵里必须已经包含"静态资源"（因为 cuobjdump 找到了）
    caps = {c.provenance: c for c in rep.capabilities}
    assert any("SASS" in i for i in caps["exact"].items)
    out = capsys.readouterr().out


def test_doctor_flags_sysadmin_present_as_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "discover_tool",
                        lambda name: _discovery(name, "/usr/bin/nsys" if name == "nsys" else None))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda ncu_path=None: (True, "ok"))
    mask = (1 << 21) | (1 << 38)          # CAP_SYS_ADMIN + CAP_PERFMON
    st = tmp_path / "status"
    st.write_text(f"CapEff:\t{mask:016x}\nCapBnd:\t{mask:016x}\n")
    monkeypatch.setenv("KS_PROC_STATUS", str(st))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    rep = D.run_doctor()
    ok = {c.name for c in rep.checks if c.status == D.OK}
    assert "CAP_SYS_ADMIN" in ok and "CAP_PERFMON" in ok


def test_cli_dispatches_doctor_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(D, "discover_tool", lambda name: _discovery(name, None))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda ncu_path=None: (None, "n/a"))
    rc = main(["doctor", "--no-probe"])
    out = capsys.readouterr().out
    assert "环境自检" in out
    assert rc == 1  # 无 nsys → 核心路径不可用


def test_unknown_probe_does_not_claim_restriction(monkeypatch, tmp_path):
    """没装 ncu 时只能说"无法判定"，不得套用"受限"的样板原因（避免错误归因）。"""
    monkeypatch.setattr(D, "discover_tool",
                        lambda name: _discovery(name, "/usr/bin/nsys" if name == "nsys" else None))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe",
                        lambda ncu_path=None: (None, "在所有候选路径里都没有找到 ncu"))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    chk = [c for c in rep.checks if c.name.startswith("硬件计数器")][0]
    assert chk.status == D.WARN
    assert chk.impact != UNAVAILABLE_REASONS["counters"]
    assert "无法判定" in chk.impact
    assert rep.counter_state == "not-installed"


def test_unknown_probe_with_ncu_present_is_not_in_path(monkeypatch, tmp_path):
    """已安装但不在 PATH、且探测无法判定 → 四态里的 not-in-path（不能报"未安装"）。"""
    monkeypatch.setattr(
        D, "discover_tool",
        lambda name: _discovery(name, "/usr/local/cuda-13.0/bin/ncu" if name == "ncu" else
                                ("/usr/bin/nsys" if name == "nsys" else None),
                                in_path=(name != "ncu")))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda ncu_path=None: (None, "探测超时"))
    monkeypatch.setenv("KS_NV_PARAMS", str(tmp_path / "none"))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    assert rep.counter_state == "not-in-path"
    assert "cuda-13.0" in rep.counter_state_note
    assert "已按多路径找到" in rep.counter_state_note


def test_available_counters_move_to_exact_not_unavailable(monkeypatch, tmp_path):
    """若计数器真的可用，能力矩阵不应把它列为 unavailable。"""
    paths = {"nsys": "/usr/bin/nsys", "ncu": "/usr/local/cuda/bin/ncu"}
    monkeypatch.setattr(D, "discover_tool",
                        lambda name: _discovery(name, paths.get(name)))
    monkeypatch.setattr(D, "gpu_info", lambda: {})
    monkeypatch.setattr(D, "counter_gate_probe", lambda ncu_path=None: (True, "ok"))
    params = tmp_path / "params"
    params.write_text("RmProfilingAdminOnly: 0\n")
    monkeypatch.setenv("KS_NV_PARAMS", str(params))
    monkeypatch.setenv("KS_PROC_STATUS", str(tmp_path / "none"))
    rep = D.run_doctor()
    conv = {c.provenance: c for c in rep.capabilities}
    assert "unavailable" not in conv, "计数器可用时不该出现 unavailable 能力"
    assert any("MemoryWorkloadAnalysis" in i for i in conv["exact"].items)
