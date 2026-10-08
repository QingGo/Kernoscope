"""B4/R-04 的负样本：同一 kernel 的多组 launch 配置不得被聚合成一行。"""

import json

import pytest

from kernoscope.provenance import Provenance
from kernoscope.render import render_csv, render_text
from kernoscope.report import KernelLaunch, KernelReport, Report
from kernoscope.sections import BuildOptions, build_kernel_sections


def _launch(grid: int, dur_ns: int, i: int) -> KernelLaunch:
    return KernelLaunch(name="add_k(float *, float *)", demangled="add_k(float *, float *)",
                        start_ns=i * 10_000_000, end_ns=i * 10_000_000 + dur_ns,
                        grid=(grid, 1, 1), block=(128, 1, 1), registers_per_thread=32,
                        static_smem_bytes=0, dynamic_smem_bytes=0, local_mem_per_thread=0,
                        stream_id=0, device_id=0, grid_id=0, correlation_id=i,
                        short_name="add_k", mangled="_Z5add_kPfS_")


def _mixed_kr(sections: bool = False) -> KernelReport:
    """grid=64/256/1024 各若干次（含预热），时长随 grid 变化 —— 形状扫描的真机形态。"""
    launches = []
    i = 0
    for grid, dur, n in ((64, 11_500, 40), (256, 10_800, 30), (1024, 12_600, 30)):
        for _ in range(n):
            launches.append(_launch(grid, dur, i))
            i += 1
    kr = KernelReport("add_k(float *, float *)", launch_count=len(launches), alias="add_k",
                      launches=launches)
    if sections:
        kr.sections = build_kernel_sections(kr, BuildOptions(set_name="basic", arch="sm_89"))
    return kr


def _single_kr() -> KernelReport:
    return KernelReport("add_k(float *, float *)", launch_count=10, alias="add_k",
                        launches=[_launch(64, 11_500, i) for i in range(10)])


def _metrics(secs):
    return {m.name: m for s in secs for m in s.metrics}


def test_configs_are_grouped_and_counted():
    kr = _mixed_kr()
    assert kr.is_mixed_config
    cfgs = kr.configs
    assert len(cfgs) == 3
    assert sorted(c.grid[0] for c in cfgs) == [64, 256, 1024]
    assert sorted(c.count for c in cfgs) == [30, 30, 40]
    assert cfgs[0].grid[0] == 64, "主导配置 = launch 数最多的那组"
    assert cfgs[0].mean_ns == pytest.approx(11_500)
    means = {c.grid[0]: c.mean_ns for c in cfgs}
    assert means[256] == pytest.approx(10_800) and means[1024] == pytest.approx(12_600)


def test_mixed_config_grid_size_is_not_a_lying_single_value():
    """旧行为：`launch__grid_size = 64` 而 2/3 的 launch 不是 —— 一份"看似 exact"的错读数。"""
    kr = _mixed_kr()
    m = _metrics(build_kernel_sections(kr, BuildOptions(set_name="basic", arch="sm_89")))
    assert m["launch__grid_size"].provenance is Provenance.UNAVAILABLE
    assert "3 组" in m["launch__grid_size"].reason
    assert m["ks__grid_sizes_observed"].value == "64|256|1024"
    assert m["ks__grid_sizes_observed"].provenance is Provenance.EXACT
    assert m["ks__launch_config_count"].value == 3
    assert "混合均值" in (m["ks__launch_time_mean"].note or "")


def test_each_config_gets_its_own_readings():
    """R-04 验收：三档各自的 launch 配置与时长都要能读出来。"""
    kr = _mixed_kr()
    secs = build_kernel_sections(kr, BuildOptions(set_name="basic", arch="sm_89"))
    cfg_sec = [s for s in secs if s.name == "LaunchConfigs"]
    assert cfg_sec, "混合配置时必须给出逐组的 LaunchConfigs section"
    m = _metrics(secs)
    for grid, dur_ms, n in ((64, 0.0115, 40), (256, 0.0108, 30), (1024, 0.0126, 30)):
        key = f"grid{grid}x1x1_block128x1x1_smem0"
        assert m[f"ks__cfg_{key}__launch_time_mean_ms"].value == pytest.approx(dur_ms, rel=1e-3)
        assert m[f"ks__cfg_{key}__launch_count"].value == n
        assert f"ks__cfg_{key}__theoretical_occupancy_pct" in m


def test_single_config_keeps_plain_exact_values():
    kr = _single_kr()
    assert not kr.is_mixed_config
    secs = build_kernel_sections(kr, BuildOptions(set_name="basic", arch="sm_89"))
    assert not [s for s in secs if s.name == "LaunchConfigs"], "单配置不需要逐组 section"
    m = _metrics(secs)
    assert m["launch__grid_size"].value == 64
    assert m["launch__grid_size"].provenance is Provenance.EXACT
    assert "ks__grid_sizes_observed" not in m


def test_report_and_csv_expose_configs():
    rep = Report()
    rep.kernels.append(_mixed_kr(sections=True))
    text = render_text(rep, quiet=True)
    assert "组 launch 配置" in text
    assert "LaunchConfigs" in text
    csv_text = render_csv(rep)
    assert "grid1024x1x1_block128x1x1_smem0__launch_time_mean_ms" in csv_text


def test_json_roundtrip_keeps_launches_and_configs():
    rep = Report()
    rep.kernels.append(_mixed_kr())
    doc = json.loads(rep.to_json())
    kd = doc["kernels"][0]
    assert kd["mixed_config"] is True
    assert len(kd["launch_configs"]) == 3
    assert len(kd["launches"]) == 100
    assert kd["launches"][0]["config"].startswith("grid")
    back = Report.from_dict(doc)
    assert back.kernels[0].is_mixed_config
    assert len(back.kernels[0].launches) == 100, "-i 回读不得丢掉 launch 级记录"
