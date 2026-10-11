"""窗口的环境证据（R-61 延伸）：降频位图、时钟驻留、同卡其他进程。

为什么这一组是"不重放"路线的必要补偿：我们的耗时是**真实时间**，而真实时间只在
**环境可比**时才能互相比较。没有重放/锁频，就只能把环境本身测量出来并写进前提
（哪个更诚实：ncu 锁频 + 重放得到的是另一台"实验室机器"上的数字）。

RE 纪律：位图按 NVML 公开位名解码；**未知位照实写 `bitN`**，驱动只说"Active"而没给
位图时记 `unspecified` —— 三种情况都不许猜成"没降频"。
"""

from __future__ import annotations

import os
import types

import pytest

from kernoscope.backends import nvml as nvml_mod
from kernoscope.backends.nvml import (
    THROTTLE_UNKNOWN,
    DeviceSample,
    Sampler,
    SamplerResult,
    decode_throttle,
    parse_throttle_token,
)
from kernoscope.packet import _environment_premises, build
from kernoscope.provenance import Metric, Section
from kernoscope.report import KernelReport, Report
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


def test_decode_throttle_negative_mask_is_unspecified_not_a_hang():
    """R-96⑦（第六轮 W-13）：`decode_throttle(-1)` 旧实现死循环（`-1 >> 1` 永远是 `-1`）。

    负数＝未知通道：给 `unspecified`（与"干净"的空列表区分开），并保证一定返回。
    """
    assert decode_throttle(-1) == ["unspecified"]
    assert decode_throttle(THROTTLE_UNKNOWN) == ["unspecified"]


@pytest.mark.parametrize("text,mask", [
    ("0x0000000000000001", 1),
    ("0x4", 4),
    ("Not Active", 0),
    ("false", 0),
    ("0", 0),
    ("1", 1),
    ("Active", THROTTLE_UNKNOWN),          # 有降频但没给位图 —— 不能当成 0
])
def test_parse_throttle_token_handles_driver_variants(text, mask):
    assert parse_throttle_token(text) == mask


@pytest.mark.parametrize("text", ["", "n/a", "N/A", "unknown", "0xZZ", "??"])
def test_parse_throttle_token_unreadable_is_unknown_not_zero(text):
    """R-96②：`n/a` 旧实现映射成 `0`（＝"没有降频"）—— 读不到必须走未知通道，不是 0。"""
    assert parse_throttle_token(text) == THROTTLE_UNKNOWN
    assert parse_throttle_token(text) != 0


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


def test_gpu_idle_falls_back_to_sm_util_when_bitmap_unreadable():
    """R-96③：位图拿不到时，真空闲的窗口旧实现报 `idle_pct=0.0` ⇒ 另一处假绿。"""
    r = _result()
    r.samples = [DeviceSample(0, 0, 0, 210, 20, THROTTLE_UNKNOWN),
                 DeviceSample(1, 0, 0, 210, 20, THROTTLE_UNKNOWN)]
    prof = r.throttle_profile()
    assert prof["idle_pct"] == pytest.approx(100.0)
    # 位图未知仍计入受限（宁可保守，不许洗成"没降频"）：两条前提一起挡住结论
    assert prof["limited_pct"] == pytest.approx(100.0)


def test_gpu_idle_fallback_does_not_use_sm_util_when_bitmap_is_readable():
    """位图可读时以驱动位图为准：`sm_util==0` 不等于驱动说的 gpu_idle。"""
    r = _result()
    r.samples = [DeviceSample(0, 0, 0, 210, 20, 0x4)]     # 明确功率封顶，不是空闲位
    prof = r.throttle_profile()
    assert prof["idle_pct"] == pytest.approx(0.0)
    assert prof["limited_pct"] == pytest.approx(100.0)


def test_environment_reports_contention_only_when_the_probe_ran():
    r = _result()
    r.samples = [DeviceSample(0, 0, 50, 1000, 100, 0)]
    assert "other_procs" not in r.environment()          # 没探测 → 不写这一项
    assert r.environment()["other_procs_probe"] == "not_run"
    r.other_procs_probed = True
    r.other_procs = 0
    assert r.environment()["other_procs"] == 0
    assert r.environment()["other_procs_probe"] == "ok"


def test_other_procs_probe_failure_is_unknown_not_zero_tenants():
    """R-96④：探测失败 ≠"0 个别的租户"，也 ≠"从没测过"——三者必须可区分。"""
    r = _result()
    r.samples = [DeviceSample(0, 0, 50, 1000, 100, 0)]
    r.other_procs_probe = "failed"                       # 探测过、但读不到
    env = r.environment()
    assert "other_procs" not in env                      # 不许冒充 0 个租户
    assert env["other_procs_probe"] == "failed"

    never = _result()
    never.samples = r.samples
    assert never.environment()["other_procs_probe"] == "not_run"

    got_zero = _result()
    got_zero.samples = r.samples
    got_zero.other_procs_probed = True
    assert got_zero.environment()["other_procs"] == 0
    assert got_zero.environment()["other_procs_probe"] == "ok"


def test_sampler_loop_marks_failed_probe_as_unknown(monkeypatch):
    """`Sampler._loop()` 的接线：`_other_procs()` 返回 None ⇒ 探测状态 = failed（不是 0）。"""
    s = Sampler()
    monkeypatch.setattr(s, "_other_procs", lambda: None)
    monkeypatch.setattr(s, "_read", lambda: None)
    rounds = {"n": 0}

    def _wait(_t):
        rounds["n"] += 1
        return rounds["n"] > 1                    # 只跑一轮就停
    monkeypatch.setattr(s._stop, "wait", _wait)
    s._loop()
    assert rounds["n"] == 2
    assert s.result.other_procs_probe == "failed"
    assert s.result.other_procs_probed is False
    assert "other_procs" not in s.result.environment()


def test_compute_apps_query_is_scoped_to_this_device(monkeypatch):
    """R-96⑤：`--query-compute-apps` 必须带本卡 `--id`，否则别的卡的进程算成"同卡"。"""
    seen: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(nvml_mod, "subprocess",
                        types.SimpleNamespace(run=lambda argv, **kw: seen.append(argv) or _Proc()))
    assert Sampler(device_index=2)._other_procs() == 0
    argv = " ".join(seen[0])
    assert "--query-compute-apps=pid" in argv
    assert "--id=2" in argv


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


# --------------------------------------------------------------------------- #
# R-96：读不到 ≠ 干净 —— 两条采样路径纪律一致（第六轮 W-1 P0 的复现）
# --------------------------------------------------------------------------- #
class _FakeUtil:
    def __init__(self, mem: float, gpu: float):
        self.memory = mem
        self.gpu = gpu


def _fake_pynvml(*, throttle_raises: bool = True, sm_util: float = 80.0):
    """最小假 pynvml：**只**让降频查询抛异常（W-1 的注入点，其余读数正常）。"""
    m = types.SimpleNamespace()
    m.NVML_CLOCK_SM = 1
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: object()
    m.nvmlDeviceGetTotalEnergyConsumption = lambda h: 1_000_000
    m.nvmlDeviceGetMaxClockInfo = lambda h, c: 3105
    m.nvmlDeviceGetUtilizationRates = lambda h: _FakeUtil(3.0, sm_util)
    m.nvmlDeviceGetClockInfo = lambda h, c: 2100
    m.nvmlDeviceGetPowerUsage = lambda h: 200_000
    m.nvmlDeviceGetComputeRunningProcesses = lambda h: []

    def _boom(h):
        raise RuntimeError("nvmlDeviceGetCurrentClocksThrottleReasons: 读不到")

    m.nvmlDeviceGetCurrentClocksThrottleReasons = _boom if throttle_raises else (lambda h: 0)
    return m


def _sampled(fake, n: int = 5) -> SamplerResult:
    """绕过线程直接走 `Sampler._read()`（生产路径就是它）。"""
    s = Sampler()
    s._pynvml = fake
    s._handle = object()
    for _ in range(n):
        sample = s._read()
        assert sample is not None, "读不到降频位图不许把整条样本也丢了"
        s.result.samples.append(sample)
    s.result.duration_s = 1.0
    return s.result


def _report_with_cost(res: SamplerResult) -> Report:
    section = build_device_kernel(BuildOptions(nvml_summary=res.summary(),
                                              nvml_env=res.environment()))
    cost = Section("Cost")
    cost.add(Metric.derived("ks__cost_per_mtok", 3.5, unit="¥/Mtoken"))
    rep = Report()
    rep.kernels = [KernelReport("k", sections=[section, cost])]
    return rep


def test_unreadable_throttle_never_becomes_clean_or_actionable():
    """R-96①⑤（W-1 P0）：假 pynvml 只让降频查询抛异常 ⇒ `cost_per_token` 不得 actionable。

    旧行为：`except Exception: mask = 0` ⇒ `limited_pct = 0.0` ⇒ 新前提
    `env_no_throttle_pct observed=0.0 ok=True` ⇒ `cost_per_token` 保持 `actionable=True`。
    """
    res = _sampled(_fake_pynvml(throttle_raises=True))
    assert [s.throttle_mask for s in res.samples] == [THROTTLE_UNKNOWN] * 5
    env = res.environment()
    assert env["throttle"]["limited_pct"] == pytest.approx(100.0)
    assert env["throttle"]["reasons"] == {"unspecified": pytest.approx(100.0)}

    rep = _report_with_cost(res)
    prem = {p.id: p for p in _environment_premises(rep)}
    assert "env_no_throttle_pct" in prem and prem["env_no_throttle_pct"].ok is False

    packet = build(rep)
    cost = next(v for v in packet["verdicts"] if v["id"] == "cost_per_token")
    assert cost["actionable"] is False


def test_readable_clean_throttle_bitmap_keeps_cost_actionable():
    """对照：位图可读且干净时前提必须成立 —— 修法不能变成"一律拒绝"。"""
    res = _sampled(_fake_pynvml(throttle_raises=False))
    assert [s.throttle_mask for s in res.samples] == [0] * 5
    prem = {p.id: p for p in _environment_premises(_report_with_cost(res))}
    assert prem["env_no_throttle_pct"].ok is True
    assert prem["env_gpu_busy_pct"].ok is True


def test_nvidia_smi_unreadable_throttle_token_matches_pynvml_unknown_channel(monkeypatch):
    """R-96①：两条路纪律一致 —— `n/a` 与 pynvml 抛异常都进"未知"，都不是"没有降频"。"""
    seen: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "3, 80, 2100, 200.00, n/a"

    monkeypatch.setattr(nvml_mod, "subprocess",
                        types.SimpleNamespace(run=lambda argv, **kw: seen.append(argv) or _Proc()))
    sample = Sampler(device_index=1)._read()
    assert sample is not None and sample.throttle_mask == THROTTLE_UNKNOWN
    assert "--id=1" in " ".join(seen[0])          # 与 pynvml 的 handle 指同一张卡

