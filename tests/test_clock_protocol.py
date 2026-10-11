"""择中项 ⑥：**时钟状态进协议哈希 + 锁频处方**。

为什么这件事值得单开一层：没锁频（或锁频状态读不到）时，两次运行可能跑在不同频率上，
把它们的时长/能效相减就是拿两个口径做减法 —— 与"换了标定"同类。四件事必须成立：

  ① 只有**粗粒度状态**（locked/unlocked/unknown）进协议哈希：实测 p50 这种每次都变的数字
     不许进（否则每一次比较都会判"协议不同"，把噪声当口径）；
  ② 状态一变，`diff` 必须不可比，并且**点名**是锁频状态不同（不能只说"协议不同"）；
  ③ 处方要能给（怎么锁、锁定什么、跑完怎么复位），并且**我们不替你锁**（写操作不在授权里）；
  ④ "读不到"（unknown）不许冒充"没锁"（unlocked）——两者都进处方，但结论的确定性不同。
"""

from __future__ import annotations

from kernoscope.backends import nvml
from kernoscope.diff_cmd import diff_packets
from kernoscope.packet import build, protocol_hash
from kernoscope.refusals import DEFAULT_REMEDIES, RefusalCode
from kernoscope.report import Report
from kernoscope.sections import BuildOptions, build_device_kernel


def _rep(**meta):
    rep = Report(device=None)
    rep.meta.update(meta)
    return rep


# ---------------------------------------------------------------- ① 协议哈希
def test_only_the_coarse_state_enters_the_protocol_hash():
    locked = _rep(clock_lock={"state": "locked", "min_mhz": 2100.0, "max_mhz": 2100.0})
    unlocked = _rep(clock_lock={"state": "unlocked"})
    assert protocol_hash(locked) != protocol_hash(unlocked), "锁频状态不同必须换协议"


def test_the_measured_clock_does_not_enter_the_protocol_hash():
    """实测时钟（每次都变）不许进哈希：否则**每次**比较都会被判"协议不同"。"""
    a = _rep(clock_lock={"state": "locked", "min_mhz": 2100.0, "max_mhz": 2100.0},
             window="mode=window wall=1.0s", warmup_s=1.0)
    b = _rep(clock_lock={"state": "locked", "min_mhz": 2100.0, "max_mhz": 2100.0},
             window="mode=window wall=1.0s", warmup_s=1.0)
    a.meta["nvml_env"] = {"clock": {"p50_mhz": 2100.0}}
    b.meta["nvml_env"] = {"clock": {"p50_mhz": 1955.0}}     # 只差实测值
    assert protocol_hash(a) == protocol_hash(b), "把实测时钟放进了哈希：噪声会被当成口径不同"


def test_missing_clock_state_is_unknown_not_unlocked():
    assert protocol_hash(_rep()) == protocol_hash(_rep(clock_lock={"state": "unknown"}))
    assert protocol_hash(_rep()) != protocol_hash(_rep(clock_lock={"state": "unlocked"}))


# ---------------------------------------------------------------- ② diff 点名
def _pkt(state: str) -> dict:
    p = {
        "tool_version": "0.2.0", "schema_version": "1.2", "kernel": "k",
        "protocol": {"hash": f"sha256:{state}", "basis": "b"},
        "verdicts": [{"id": "kernel_time", "status": "soft", "value": 4.0, "unit": "ms",
                      "provenance": "exact", "facts": {}}],
        "clock_lock": state,
    }
    return p


def test_diff_names_the_clock_state_instead_of_just_saying_protocols_differ():
    res = diff_packets(_pkt("locked"), _pkt("unlocked"))
    assert not res.comparable
    assert "clock_lock" in res.protocol_diffs, res.protocol_diffs
    assert "锁频状态不同" in str(res.protocol_diffs["clock_lock"])
    assert res.comparisons and all(c.status == "incomparable" for c in res.comparisons)


# ---------------------------------------------------------------- ③④ 探测的三态
class _FakeNvml:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def nvmlDeviceGetGpuLockedClocks(self, handle):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def _sampler_with(behaviour):
    s = nvml.Sampler.__new__(nvml.Sampler)          # 不跑 __init__（不碰真设备/线程）
    s._pynvml = _FakeNvml(behaviour)
    s._handle = object()
    return s


def test_lock_probe_three_states_are_distinguished():
    assert _sampler_with((2100, 2100))._clock_lock_state()["state"] == "locked"
    assert _sampler_with((0, 0))._clock_lock_state()["state"] == "unlocked"

    class NVMLError_NotSupported(Exception):
        pass

    assert _sampler_with(NVMLError_NotSupported())._clock_lock_state()["state"] == "unlocked"

    class NVMLError_Unknown(Exception):
        pass

    assert _sampler_with(NVMLError_Unknown())._clock_lock_state()["state"] == "unknown"

    no_py = nvml.Sampler.__new__(nvml.Sampler)
    no_py._pynvml, no_py._handle = None, None
    assert no_py._clock_lock_state()["state"] == "unknown", "读不到不许冒充'没锁'"


# ---------------------------------------------------------------- 处方
def test_unlocked_clock_reaches_the_packet_with_a_prescription():
    rep = _rep(clock_lock={"state": "unlocked", "source": "pynvml"})
    rep.meta["environment_refusals"] = [{
        "code": RefusalCode.CLOCK_NOT_LOCKED.value,
        "message": "时钟没被锁住：两次运行可能跑在不同频率上，时长/能效不可直接相减"}]
    p = build(rep)
    assert p.get("clock_lock") == "unlocked"
    codes = {e.get("code") for e in p.get("environment_refusals") or []}
    assert RefusalCode.CLOCK_NOT_LOCKED.value in codes, codes
    # 处方必须能照着做（具体命令），并且说清"我们不替你锁"
    rem = DEFAULT_REMEDIES[RefusalCode.CLOCK_NOT_LOCKED]
    assert "nvidia-smi -lgc" in (rem.command or "") and "nvidia-smi -rgc" in (rem.command or "")
    assert "不会替你锁" in rem.action


def test_unknown_clock_does_not_nag_on_every_real_run():
    """读不到锁频状态时**不**挂"去锁频"的告示：那种情况下我们连它锁没锁都不知道。"""
    rep = _rep(clock_lock={"state": "unknown", "source": "no_pynvml"})
    p = build(rep)
    codes = {e.get("code") for e in p.get("environment_refusals") or []}
    assert RefusalCode.CLOCK_NOT_LOCKED.value not in codes, codes


def test_device_section_reports_the_state_and_the_lock_recipe():
    opt = BuildOptions(nvml_env={"clock_lock": {"state": "unlocked", "source": "pynvml"}})
    sec = build_device_kernel(opt)
    m = next((x for x in sec.metrics if x.name == "ks__clock_lock_state"), None)
    assert m is not None and m.value == "未锁频", [x.name for x in sec.metrics]
    assert "nvidia-smi -lgc" in m.note and "不会替你锁" in m.note

    opt2 = BuildOptions(nvml_env={"clock_lock": {
        "state": "locked", "min_mhz": 2100.0, "max_mhz": 2100.0, "source": "pynvml"}})
    m2 = next(x for x in build_device_kernel(opt2).metrics if x.name == "ks__clock_lock_state")
    assert m2.value == "已锁频" and "sudo" not in m2.note, m2.note
