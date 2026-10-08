"""section 组装：寄存器来源优先级与 occupancy 取值。"""

import json

from kernoscope.cli import main
from kernoscope.provenance import Provenance
from kernoscope.report import KernelLaunch, KernelReport
from kernoscope.sections import BuildOptions, build_kernel_sections


def _kr(regs: int) -> KernelReport:
    l = KernelLaunch(name="addk(float *, int)", demangled="addk(float *, int)",
                     start_ns=0, end_ns=1_000_000, grid=(256, 1, 1), block=(128, 1, 1),
                     registers_per_thread=regs, static_smem_bytes=0, dynamic_smem_bytes=0,
                     local_mem_per_thread=0, stream_id=0, device_id=0, grid_id=0,
                     correlation_id=0, short_name="addk", mangled="_Z4addkPfi")
    return KernelReport("addk(float *, int)", launch_count=1, alias="addk", launches=[l])


def _metrics(secs):
    return {m.name: m for s in secs for m in s.metrics}


def test_cubin_registers_override_occupancy():
    # CUPTI 报 16，cubin(ptxas) 报 96 -> occupancy 用 96（静态真值）
    kr = _kr(16)
    opt = BuildOptions(cubin_res={"_Z4addkPfi": {"REG": 96, "SHARED": 0}}, arch="sm_89")
    m = _metrics(build_kernel_sections(kr, opt))
    assert m["launch__registers_per_thread"].value == 16
    assert m["ks__cubin_registers_per_thread"].value == 96
    assert "不一致" in m["ks__cubin_registers_per_thread"].note
    assert m["ks__registers_used_for_occupancy"].value == 96
    assert m["ks__registers_used_for_occupancy"].source.startswith("cubin")
    # 96 regs + 128 线程 = 4 warps/block；96 regs/warp 粒度 3072 -> 65536/3072=21 warps -> 5 blocks
    assert m["ks__blocks_per_sm"].value == 5


def test_cubin_consistent_marks_agreement():
    kr = _kr(8)
    opt = BuildOptions(cubin_res={"_Z4addkPfi": {"REG": 8}}, arch="sm_89")
    m = _metrics(build_kernel_sections(kr, opt))
    assert "一致" in m["ks__cubin_registers_per_thread"].note
    assert m["ks__registers_used_for_occupancy"].source == "CUPTI activity"


def test_all_unavailable_section_has_no_values():
    kr = _kr(16)
    opt = BuildOptions(set_name="full", sections=["SchedulerStats"])
    secs = build_kernel_sections(kr, opt)
    sched = [s for s in secs if s.name == "SchedulerStats"][0]
    assert sched.provenance is Provenance.UNAVAILABLE
    assert not sched.has_values
    assert all(m.value is None for m in sched.metrics)


def test_cli_reports_registers_source(capsys, tmp_path):
    base = tmp_path / "r"
    assert main(["--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f"]) == 0
    capsys.readouterr()
    doc = json.loads((tmp_path / "r.ksreport.json").read_text())
    names = {m["name"] for k in doc["kernels"] for s in k["sections"] for m in s["metrics"]}
    assert "ks__registers_used_for_occupancy" in names
