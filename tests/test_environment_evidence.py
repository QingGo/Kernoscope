"""窗口的环境证据（R-61 延伸）：降频位图、时钟驻留、同卡其他进程。

为什么这一组是"不重放"路线的必要补偿：我们的耗时是**真实时间**，而真实时间只在
**环境可比**时才能互相比较。没有重放/锁频，就只能把环境本身测量出来并写进前提
（哪个更诚实：ncu 锁频 + 重放得到的是另一台"实验室机器"上的数字）。

RE 纪律：位图按 NVML 公开位名解码；**未知位照实写 `bitN`**，驱动只说"Active"而没给
位图时记 `unspecified` —— 三种情况都不许猜成"没降频"。
"""

from __future__ import annotations

import os

import pytest

from kernoscope.backends.nvml import (
    DeviceSample,
    SamplerResult,
    decode_throttle,
    parse_throttle_token,
)
from kernoscope.packet import _environment_premises
from kernoscope.sections import BuildOptions, build_device_kernel


def _result(**kw) -> SamplerResult:
    r = SamplerResult(samples=[], duration_s=1.0)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


# --------------------------------------------------------------------------- #
# 位图解码
# --------------------------------------------------------------------------- #
def test_decode_throttle_names_known_bits_and_keeps_unknown_bits_verbatim():
    assert decode_throttle(0) == []
    assert decode_throttle(0x1) == ["gpu_idle"]
    assert decode_throttle(0x4 | 0x40) == ["sw_power_cap", "hw_thermal_slowdown"]
    # 0x1000 落在这份位表之外：照实报，不猜语义
    assert decode_throttle(0x1000) == ["bit12"]
    assert decode_throttle(0x4 | 0x1000) == ["sw_power_cap", "bit12"]


@pytest.mark.parametrize("text,mask", [
    ("0x0000000000000001", 1),
    ("0x4", 4),
    ("Not Active", 0),
    ("0", 0),
    ("", 0),
    ("Active", -1),          # 有降频但没给位图 —— 不能当成 0
    ("n/a", 0),
])
def test_parse_throttle_token_handles_driver_variants(text, mask):
    assert parse_throttle_token(text) == mask


# --------------------------------------------------------------------------- #
# 时钟驻留与降频画像
# --------------------------------------------------------------------------- #
def test_clock_residency_uses_nominal_max_and_counts_shortfall():
    r = _result(max_sm_clock_mhz=3105.0)
    r.samples = [DeviceSample(0, 0, 50, 3105, 100, 0),
                 DeviceSample(1, 0, 50, 3000, 100, 0),
                 DeviceSample(2, 0, 50, 210, 40, 0),
                 DeviceSample(3, 0, 50, 3105, 100, 0)]
    res = r.clock_residency()
    assert res["ref_mhz"] == 3105.0
    assert res["active_samples"] == 4.0
    assert res["min_mhz"] == 210.0
    assert res["at_max_pct"] == pytest.approx(50.0)      # 2/4 在标称最高时钟附近


def test_clock_residency_falls_back_to_observed_max():
    """拿不到 clocks.max.sm 时用窗口内最高时钟，并如实报出来（ref_mhz == max）。"""
    r = _result()
    r.samples = [DeviceSample(0, 0, 50, 1000, 100, 0), DeviceSample(1, 0, 50, 2000, 100, 0)]
    res = r.clock_residency()
    assert res["ref_mhz"] == 2000.0 and res["at_max_pct"] == pytest.approx(50.0)


def test_clock_residency_ignores_idle_samples():
    """空闲时驱动降到 210 MHz 是正常行为，不能算成"一直在低频跑"（真机第一次跑就撞上）。

    闲的比例由 `gpu_idle_pct` 单独报，两件事不混。
    """
    r = _result(max_sm_clock_mhz=3105.0)
    r.samples = [DeviceSample(0, 0, 0, 210, 20, 0x1),          # 空闲：util 0 + idle 位
                 DeviceSample(1, 0, 0, 210, 20, 0),            # 没在跑（util 0）
                 DeviceSample(2, 0, 80, 3105, 300, 0),         # 活跃且在峰值
                 DeviceSample(3, 0, 80, 3105, 300, 0)]         # 活跃且在峰值
    res = r.clock_residency()
    assert res["active_samples"] == 2.0 and res["samples"] == 4.0
    assert res["p50_mhz"] == 3105.0
    assert res["p50_all_mhz"] == pytest.approx(1657.5)   # 含空闲点的中位数：同时给出便于对照
    assert res["at_max_pct"] == pytest.approx(100.0)


def test_other_process_detection_excludes_our_own_process_tree():
    """`kernoscope ... -- python bench.py` 不许把 bench.py 数成"另一个租户"（真机撞上过）。"""
    from kernoscope.backends.nvml import _is_descendant

    assert _is_descendant(1, {1})                       # 自己是"自己人"
    assert _is_descendant(os.getpid(), {os.getpid()})
    assert not _is_descendant(1, {os.getpid()})         # init 不是我们的后代
    assert not _is_descendant(1 << 30, {os.getpid()})   # 不存在的 pid：读不到 → 算别人


def test_throttle_profile_separates_idle_from_performance_limits():
    r = _result(max_sm_clock_mhz=3105.0)
    r.samples = [
        DeviceSample(0, 0, 50, 1000, 100, 0x1),            # 只是空闲
        DeviceSample(1, 0, 50, 1000, 100, 0x4),            # 功率封顶
        DeviceSample(2, 0, 50, 1000, 100, 0x4 | 0x1),      # 两者都有
        DeviceSample(3, 0, 50, 1000, 100, 0),              # 干净
    ]
    prof = r.throttle_profile()
    assert prof["idle_pct"] == pytest.approx(50.0)
    assert prof["limited_pct"] == pytest.approx(50.0)
    assert prof["reasons"]["sw_power_cap"] == pytest.approx(50.0)


def test_throttle_profile_counts_unspecified_and_never_as_clean():
    r = _result()
    r.samples = [DeviceSample(0, 0, 50, 1000, 100, -1), DeviceSample(1, 0, 50, 1000, 100, 0)]
    prof = r.throttle_profile()
    assert prof["limited_pct"] == pytest.approx(50.0)
    assert prof["reasons"] == {"unspecified": pytest.approx(50.0)}


def test_environment_reports_contention_only_when_the_probe_ran():
    r = _result()
    r.samples = [DeviceSample(0, 0, 50, 1000, 100, 0)]
    assert "other_procs" not in r.environment()          # 没探测 → 不写这一项
    r.other_procs_probed = True
    r.other_procs = 0
    assert r.environment()["other_procs"] == 0


# --------------------------------------------------------------------------- #
# section 与决策包接线
# --------------------------------------------------------------------------- #
ENV = {
    "clock": {"p50_mhz": 3010.0, "p50_all_mhz": 210.0, "min_mhz": 3010.0, "max_mhz": 3105.0,
              "ref_mhz": 3105.0, "at_max_pct": 42.0, "active_samples": 18.0, "samples": 30.0},
    "throttle": {"limited_pct": 12.5, "idle_pct": 60.0, "reasons": {"sw_power_cap": 12.5}},
    "other_procs": 2,
    "interval_s": 0.2,
}


def _device_section(env):
    opt = BuildOptions(nvml_summary={"mem_util_mean": 3.0}, nvml_env=env)
    return {m.name: m for m in build_device_kernel(opt).metrics}


def test_device_section_carries_environment_evidence():
    m = _device_section(ENV)
    assert m["meas__sm_clock_p50_mhz"].value == 3010.0
    assert m["meas__throttle_limited_pct"].value == 12.5
    assert m["meas__gpu_idle_pct"].value == 60.0
    assert m["meas__other_procs_on_gpu"].value == 2.0
    # 出处必须是"实测"而不是 estimated：这些是从管理计数器读出来的
    for name in ("meas__sm_clock_p50_mhz", "meas__throttle_limited_pct",
                 "meas__gpu_idle_pct", "meas__other_procs_on_gpu"):
        assert m[name].provenance.value == "measured", name
    assert "sw_power_cap" in (m["meas__throttle_limited_pct"].note or "")


def test_device_section_without_environment_still_refuses_for_a_reason():
    m = _device_section({})
    assert "meas__other_procs_on_gpu" not in m
    assert m["meas__nvml_mem_busy_mean_pct"].provenance.value == "measured"


class _Rep:
    def __init__(self, metrics):
        self.kernels = [_K(metrics)]

    def iter_sections(self):
        yield from self.kernels[0].sections


class _K:
    def __init__(self, metrics):
        self.sections = [_S(metrics)]


class _S:
    def __init__(self, metrics):
        self.metrics = metrics


class _M:
    def __init__(self, name, value):
        self.name = name
        self.value = value


def test_packet_premises_flag_multi_tenant_and_throttling():
    ok = _environment_premises(_Rep([_M("meas__other_procs_on_gpu", 0.0),
                                     _M("meas__gpu_idle_pct", 2.0),
                                     _M("meas__throttle_limited_pct", 0.0)]))
    assert [p.id for p in ok] == ["env_gpu_exclusive", "env_gpu_busy_pct", "env_no_throttle_pct"]
    assert all(p.ok for p in ok)

    dirty = _environment_premises(_Rep([_M("meas__other_procs_on_gpu", 3.0),
                                        _M("meas__gpu_idle_pct", 55.0),
                                        _M("meas__throttle_limited_pct", 40.0)]))
    assert [p.id for p in dirty if p.ok is False] == ["env_gpu_exclusive", "env_gpu_busy_pct",
                                                      "env_no_throttle_pct"]
    assert dirty[0].observed == 3.0 and dirty[0].threshold == 0.0


def test_packet_premises_are_absent_when_we_did_not_measure_the_environment():
    assert _environment_premises(_Rep([])) == []
