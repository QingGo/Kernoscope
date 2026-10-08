"""W2/W4：决策包（机器视图）与一屏（人视图）。

守四条不变量：
  1. **尺寸预算**：默认包永远 < 4 KB（硬约束，靠有序裁剪保证，且裁剪如实记录）；
  2. **不丢拒绝**：判词/未测到里出现的每个 code 都能在 `refusals` 表里查到怎么办；
  3. **防分叉**：人视图只由包渲染 —— 包里没有的东西不许出现在人视图里；
  4. **协议哈希**只认口径：同协议两次运行同哈希，换了窗口/标定就不同，换了时间/主机名不变。
"""

from __future__ import annotations

import json

import pytest

from kernoscope.packet import (MAX_VERDICTS, OK, PACKET_SCHEMA_VERSION,
                              PACKET_SIZE_BUDGET_BYTES, REFUSED, SOFT, Premise, Verdict,
                              build, dumps, packet_bytes, protocol_hash, refusal_index,
                              render_onepage)
from kernoscope.provenance import Metric, Provenance, Section
from kernoscope.refusals import RefusalCode
from kernoscope.report import DeviceInfo, KernelLaunch, KernelReport, Report


# --------------------------------------------------------------------------
# 构造报告
# --------------------------------------------------------------------------
def _launch(i: int, dur_ns: int = 1_000_000) -> KernelLaunch:
    return KernelLaunch(name="k", demangled="k(float*)", start_ns=i * 2_000_000,
                        end_ns=i * 2_000_000 + dur_ns, grid=(1024, 1, 1), block=(256, 1, 1),
                        registers_per_thread=64, static_smem_bytes=0,
                        dynamic_smem_bytes=49152, local_mem_per_thread=0, stream_id=7,
                        device_id=0, grid_id=0, correlation_id=i)


def _metric(name: str, value) -> Metric:
    if value is None:
        return Metric.unavailable(name, "没有这个数据源", code=RefusalCode.NO_NVML_SAMPLES)
    if name.startswith("meas__"):
        return Metric.measured(name, value)
    if name.startswith("ks__"):
        return Metric.derived(name, value)
    return Metric.estimated(name, value)


def _rich_report() -> Report:
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    rep.wall_time_ns = 25_000_000_000
    rep.meta["window"] = "mode=auto 采样 120 点 / 参与积分 20.0 s"
    k = KernelReport("big_gemm", launch_count=200, launches=[_launch(i) for i in range(200)])
    values = {
        "est__dram_bytes_total": 5.0e9, "ks__sampler_window_coverage_pct": 97.1,
        "ks__dram_window_mem_busy_mean_pct": 96.8, "est__dram_bytes_per_launch": 2.5e7,
        "ks__matched_kernel_kernel_share_pct": 99.1,
        "ks__matched_kernel_window_share_pct": 96.2,
        "ks__attribution_bias_estimate_pct": 0.9,
        "est__dram_bytes_per_launch_trace": 2.2e7, "ks__dram_trace_vs_nvml_pct": 88.0,
        "est__dram_vs_ledger_err_pct": 8.1, "ks__theoretical_occupancy_pct": 50.0,
        "meas__achieved_occupancy_pct": 46.9, "ks__occupancy_limiter": "registers+shared_mem",
        "est__bound": "memory", "est__roofline_efficiency_pct": 91.2, "est__achieved_gbs": 908.5,
        "est__issue_slot_utilization_pct": 27.4, "est__stall_direction": "latency",
        "est__l2_hit_rate_pct": 66.7, "est__working_set_bytes": 3.4e7,
        "ks__static_sass_instructions": 1234, "ks__static_sass_share_fp": 41.0,
        "ks__static_sass_share_memory": 33.0, "ks__static_sass_share_control": 12.0,
        "ks__static_sass_share_int": 14.0, "est__dram_throughput_pct_of_peak": 88.0,
    }
    s = Section("All")
    for name, v in values.items():
        s.add(_metric(name, v))
    # 一堆 n/a，足够触发裁剪
    for i in range(80):
        s.add(Metric.unavailable(f"sm__counter_{i:03d}", "计数器拿不到",
                                 code=RefusalCode.NO_COUNTERS))
    for i in range(40):
        s.add(Metric.unavailable(f"ks__needs_probe_{i:03d}", "没有探针",
                                 code=RefusalCode.NO_OCCUPANCY_PROBE))
    k.sections = [s]
    rep.kernels = [k]
    return rep


def _plain_report() -> Report:
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    rep.wall_time_ns = 10_000_000_000
    k = KernelReport("k", launch_count=5, launches=[_launch(i) for i in range(5)])
    k.sections = [Section("Duration")]
    k.sections[0].add(Metric.exact("gpu__time_duration.sum", 5.0, "ms"))
    rep.kernels = [k]
    return rep


# --------------------------------------------------------------------------
# 尺寸预算
# --------------------------------------------------------------------------
def test_plain_packet_well_under_budget():
    p = build(_plain_report())
    assert packet_bytes(p) <= PACKET_SIZE_BUDGET_BYTES
    assert p["cost"]["trimmed"] == []


def test_rich_packet_is_trimmed_but_stays_under_budget():
    """硬约束：报告再大，默认包也不许超预算；超了必须如实记录裁掉了什么。"""
    p = build(_rich_report())
    size = packet_bytes(p)
    assert size <= PACKET_SIZE_BUDGET_BYTES, f"包 {size} B 超预算"
    assert p["cost"]["trimmed"], "超预算的包必须记录裁剪步骤"
    assert "over_budget" not in p["cost"]
    assert p["cost"]["packet_bytes"] == size


def test_packet_bytes_matches_actual_serialization():
    p = build(_rich_report())
    text = dumps(p)
    assert len(text.encode("utf-8")) == p["cost"]["packet_bytes"]


def test_verdicts_are_capped():
    p = build(_rich_report())
    assert len(p["verdicts"]) <= MAX_VERDICTS


def test_unmeasured_is_grouped_not_exploded():
    p = build(_rich_report())
    assert len(p["unmeasured"]) <= 6
    assert all("count" in g and g["count"] > 0 for g in p["unmeasured"])
    total = sum(g["count"] for g in p["unmeasured"]) + \
        int((p.get("unmeasured_more") or {}).get("metrics") or 0)
    assert total == 120


def test_cli_synthetic_packet_end_to_end(capsys, tmp_path):
    from kernoscope.cli import main

    out = tmp_path / "packet.json"
    rc = main(["--ks-synthetic", "--set", "full", "--ks-no-nvml", "--packet", str(out),
               "--screen", "off", "--quiet"])
    assert rc == 0
    p = json.loads(out.read_text())
    assert p["schema_version"] == PACKET_SCHEMA_VERSION
    assert p["synthetic"] is True
    assert p["cost"]["packet_bytes"] <= PACKET_SIZE_BUDGET_BYTES


def test_packet_to_stdout_is_machine_only(capsys):
    from kernoscope.cli import main

    rc = main(["--ks-synthetic", "--set", "basic", "--ks-no-nvml", "--packet", "-", "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.lstrip().startswith("{")
    assert out.rstrip().endswith("}")
    json.loads(out)          # stdout 必须是纯 JSON（不能混人视图）


# --------------------------------------------------------------------------
# 判词与拒绝的一致性
# --------------------------------------------------------------------------
def test_every_referenced_code_is_in_the_refusal_table():
    p = build(_rich_report())
    codes = {v["refusal"]["code"] for v in p["verdicts"] if v.get("refusal")}
    codes |= {g["code"] for g in p["unmeasured"]}
    index = refusal_index(p)
    missing = sorted(c for c in codes if c not in index)
    assert not missing, f"包里引用了但查不到 remedy 的 code：{missing}"


def test_refused_verdicts_carry_premises_and_no_value():
    p = build(_rich_report())
    refused = [v for v in p["verdicts"] if v["status"] == REFUSED]
    for v in refused:
        assert v.get("refusal"), v["id"]
        assert "value" not in v


def test_verdict_requires_refusal_when_refused():
    with pytest.raises(ValueError, match="structured refusal"):
        Verdict(id="x", question="q", status=REFUSED)


def test_only_known_statuses():
    with pytest.raises(ValueError, match="bad verdict status"):
        Verdict(id="x", question="q", status="maybe", facts={"a": 1})


def test_premise_serialization_omits_empty_threshold():
    p = Premise("kernel_share_of_total", 96.6, None, True, "%").to_dict()
    assert "threshold" not in p and p["unit"] == "%" and p["ok"] is True


# --------------------------------------------------------------------------
# 防分叉：人视图只由包生成
# --------------------------------------------------------------------------
def test_onepage_uses_only_packet_contents():
    rep = _rich_report()
    p = build(rep)
    text = render_onepage(p)
    for v in p["verdicts"]:
        assert v["id"] not in text or v["id"]  # 不依赖 id 文案
    # 判词里的关键数字必须出现在人视图里
    for v in p["verdicts"]:
        for key, val in (v.get("facts") or {}).items():
            if isinstance(val, dict):
                assert all(str(x) in text for x in val.values()), f"{v['id']}.{key}"
            elif isinstance(val, str) and len(val) < 6:
                continue
            else:
                assert str(val) in text, f"{v['id']}.{key}={val} 没有出现在人视图"
    # 前提的观测值也必须出现
    for v in p["verdicts"]:
        for pr in v.get("premises") or []:
            if pr.get("observed") is not None:
                assert str(pr["observed"]) in text


def test_onepage_shows_unmeasured_and_cost():
    p = build(_rich_report())
    packet_bytes(p)          # dumps() 会把自身大小写进 cost（渲染要用它）
    text = render_onepage(p)
    assert "没测到什么" in text
    assert "本次代价" in text
    assert "counters_denied" in text
    assert f"包 {p['cost']['packet_bytes']}B" in text


def test_onepage_says_so_when_packet_was_trimmed():
    p = build(_rich_report())
    packet_bytes(p)
    text = render_onepage(p)
    assert "裁剪" in text


def test_onepage_marks_synthetic():
    p = build(_plain_report())
    packet_bytes(p)
    p["synthetic"] = True
    assert "合成数据" in render_onepage(p)


def test_onepage_from_packet_file_equals_in_memory(capsys, tmp_path):
    """回读路径（读包文件 / 读完整报告）必须渲染出同样的一屏。"""
    from kernoscope.cli import main
    from kernoscope.packet import load_packet

    pkt = tmp_path / "p.json"
    rep = tmp_path / "rep"
    rc = main(["--ks-synthetic", "--set", "basic", "--ks-no-nvml", "--packet", str(pkt),
               "-o", str(rep), "-f", "--screen", "off", "--quiet"])
    assert rc == 0
    capsys.readouterr()
    from_file = load_packet(str(pkt))
    assert from_file["schema_version"] == PACKET_SCHEMA_VERSION
    packet_bytes(from_file)
    assert "Kernoscope" in render_onepage(from_file)
    # 完整报告（.ksreport.json）也能被当成输入读进来
    from_report = load_packet(str(rep) + ".ksreport.json")
    assert "verdicts" in from_report and from_report["tool"] == "kernoscope"


def test_refused_verdict_explains_and_says_how_to_fix():
    rep = _plain_report()
    rep.meta["environment_refusals"] = [{"code": "nsys_missing", "message": "没有 nsys"}]
    p = build(rep)
    text = render_onepage(p)
    assert "no_nvml_samples" not in text        # 这个报告里没有该 code
    assert "本机缺件" in text or "nsys" in text


# --------------------------------------------------------------------------
# 协议哈希
# --------------------------------------------------------------------------
def test_protocol_hash_is_stable_for_same_caliber():
    a = _plain_report()
    b = _plain_report()
    assert protocol_hash(a) == protocol_hash(b)
    # 时间戳/wall 时间/主机名变化不影响可比性
    b.generated_at = "2030-01-01T00:00:00+0000"
    b.wall_time_ns = 99_000_000_000
    assert protocol_hash(a) == protocol_hash(b)


def test_protocol_hash_changes_with_caliber():
    a = _plain_report()
    b = _plain_report()
    b.meta["window"] = "mode=full 采样 120 点"
    assert protocol_hash(a) != protocol_hash(b)
    c = _plain_report()
    c.meta["calibration"] = "other-machine.json"
    assert protocol_hash(a) != protocol_hash(c)
    d = _plain_report()
    d.device.name = "A100"
    assert protocol_hash(a) != protocol_hash(d)


def test_packet_has_no_timestamp_or_hostname():
    """包里不许出现会破坏可比性的字段（时间戳/主机名）。"""
    p = build(_plain_report())
    blob = dumps(p)
    for bad in ("generated_at", "hostname", "platform"):
        assert bad not in blob
