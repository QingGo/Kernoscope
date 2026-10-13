"""跨卡（多设备）采集的口径契约。

真机实测背景（2×RTX 2080 Ti / 2026-10-12）：
  * 合成双设备轨迹：不加 `--devices` 时同名 kernel 的时长与 launch 数**静默翻倍**
    （31.9 ms / 21 次 → 63.7 ms / 42 次），既无告警也无拒绝；
  * 真实双卡 NCCL 轨迹（2 进程 all_reduce，990 次 launch）：一屏原来一个字都不提，
    读者会把 `3126.471 ms / 182 次` 当成单卡口径，而它是 `1569.794 + 1556.677`（两张卡之和）。

按本项目的纪律，跨卡加总被当成"进程内 kernel 时间合计"报出来，正是平时会拒绝的那种口径混用。
所以这里钉住三件事：**事实要进 JSON**（`-i` 回读不丢）、**结论句要明说**、**单卡口径的问题要拒绝**。
"""

from __future__ import annotations

import json

from kernoscope.packet import build as packet_build
from kernoscope.packet import phrase_verdict
from kernoscope.report import KernelLaunch, KernelReport, Report


def _launch(name: str, device_id: int, duration_ns: int, corr: int) -> KernelLaunch:
    return KernelLaunch(
        name=name, demangled=name, start_ns=0, end_ns=duration_ns,
        grid=(128, 1, 1), block=(256, 1, 1), registers_per_thread=32,
        static_smem_bytes=0, dynamic_smem_bytes=0, local_mem_per_thread=0,
        stream_id=7, device_id=device_id, grid_id=0, correlation_id=corr,
        short_name=name)


def _report(devices: list[int]) -> Report:
    """同名 kernel 在每张卡上各跑一次（duration 相同）。"""
    rep = Report()
    kr = KernelReport("gemm_kernel", launch_count=0)
    for i, d in enumerate(devices):
        kr.launches.append(_launch("gemm_kernel", d, 1_000_000, i))
    kr.launch_count = len(kr.launches)
    rep.kernels.append(kr)
    rep.device_ids = rep.observed_device_ids()
    return rep


def test_single_device_report_is_not_flagged():
    rep = _report([0])
    assert rep.observed_device_ids() == [0]
    assert not rep.is_multi_device
    assert rep.device_breakdown() == [
        {"device_id": 0, "launch_count": 1, "kernel_ns": 1_000_000, "kernels": 1}]


def test_multi_device_breakdown_is_per_device():
    """分段（每张卡各干了多少）而不是只给一个跨卡合计。"""
    rep = _report([0, 1])
    assert rep.observed_device_ids() == [0, 1]
    assert rep.is_multi_device
    rows = rep.device_breakdown()
    assert [r["device_id"] for r in rows] == [0, 1]
    assert all(r["launch_count"] == 1 and r["kernel_ns"] == 1_000_000 for r in rows)
    # 跨卡合计是两条之和（读者拿到这个数时必须知道它是两张卡）
    assert rep.total_kernel_ns == 2_000_000


def test_device_ids_survive_json_round_trip():
    """`-i` 回读后跨卡事实不能丢 —— 丢了就等于又变回静默合计。"""
    rep = _report([0, 1])
    d = rep.to_dict()
    assert d["device_ids"] == [0, 1] and d["multi_device"] is True
    back = Report.from_dict(json.loads(json.dumps(d)))
    assert back.observed_device_ids() == [0, 1]
    assert back.is_multi_device


def _phrases(rep) -> str:
    """人话视图的文本（一屏就是从这里来的；包本身不存散文）。"""
    pkt = packet_build(rep)
    return " ".join(phrase_verdict(v, {}) for v in pkt.get("verdicts", []))


def test_headline_finding_says_aggregates_span_devices():
    """人话视图（一屏）必须说出来：它是绝大多数人唯一会读的东西。

    注意它**不能**只写在 `Verdict.claim` 里 —— 包与报告都不存散文，人话只由
    `phrase_verdict` 从 `facts` 生成，所以这条事实必须进 `facts`。
    """
    rep = _report([0, 1])
    pkt = packet_build(rep)
    kt = [v for v in pkt["verdicts"] if v["id"] == "kernel_time"]
    assert kt and kt[0]["facts"].get("device_ids") == [0, 1], kt
    text = _phrases(rep)
    assert "最耗时的是" in text
    assert "跨了 2 张卡" in text and "跨卡加总" in text


def test_headline_finding_is_silent_for_a_single_device():
    rep = _report([0])
    pkt = packet_build(rep)
    kt = [v for v in pkt["verdicts"] if v["id"] == "kernel_time"]
    assert kt and "device_ids" not in kt[0]["facts"], "单卡不该多出一个设备号 fact"
    text = _phrases(rep)
    assert "最耗时的是" in text
    assert "跨卡加总" not in text


def test_import_path_honours_device_filter(tmp_path):
    """R-143：`-i` 回读时 `--devices` 必须**真的生效**。

    它正是我们自己在拒绝语里给出的补救命令（"按卡分开看：--devices <id>"）——
    指一条空操作比不给补救更糟。以前过滤只挂在采集路径上，回读路径静默忽略它。
    """
    import json

    from kernoscope import cli

    rep = _report([0, 1])
    src = tmp_path / "in.ksreport.json"
    src.write_text(json.dumps(rep.to_dict()), encoding="utf-8")
    out = tmp_path / "out"

    rc = cli.main(["-i", str(src), "--devices", "1", "--screen", "off",
                   "--set", "basic", "-o", str(out)])
    assert rc == 0
    got = json.loads((tmp_path / "out.ksreport.json").read_text(encoding="utf-8"))
    assert got["device_ids"] == [1], got["device_ids"]
    assert got["multi_device"] is False
    assert all(l["device_id"] == 1 for k in got["kernels"] for l in k["launches"])
    # 跨卡事实跟着重算：过滤完不该还报"跨了 2 张卡"
    assert got["kernels"][0]["launch_count"] == 1
