"""W2/W4：决策包（机器视图）与一屏（人视图）。

守四条不变量：
  1. **尺寸预算**：默认包永远 < 4 KB（硬约束，靠有序裁剪保证，且裁剪如实记录）；
  2. **不丢拒绝**：判词/未测到里出现的每个 code 都能在 `refusals` 表里查到怎么办；
  3. **防分叉**：人视图只由包渲染 —— 包里没有的东西不许出现在人视图里；
  4. **协议哈希**只认口径：同协议两次运行同哈希，换了窗口/标定就不同，换了时间/主机名不变。

另外一条是从真机产物学来的：**量纲必须同源**。这里的 fixture 过去把
`est__dram_bytes_total` 写成 `5.0e9`（**字节**口径），而真机 metric 的 unit 是 **MB** ——
于是"7.27 TB 在包里写成 7.3 MB"在 384 条绿测试下活了很久。所以单位回归走**入库真机产物**
（`bench/artifacts/**/rep-*.csv` 里带 value+unit），不用手搓的 fixture。
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib

import pytest

from kernoscope.packet import (MAX_VERDICTS, OK, PACKET_SCHEMA_VERSION,
                              PACKET_SIZE_BUDGET_BYTES, REFUSED, SOFT, Premise, Verdict,
                              build, dumps, packet_bytes, protocol_hash, refusal_index,
                              render_onepage)
from kernoscope.diff_cmd import diff_packets
from kernoscope.provenance import Metric, Provenance, Section, fmt_bytes, to_bytes
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


@pytest.mark.parametrize("limit", [1200, 1800, 2600, 3400, 4096])
def test_budget_is_enforced_even_when_the_trim_log_itself_costs_bytes(limit):
    """回归：`cost.trimmed` 自己占字节，量尺寸必须把它算进去。

    （真事：给判词加 `actionable` 之后，包先被裁到刚好 4090 B，再写进裁剪步骤就变成
    4119 B —— 超了预算却没人发现。现在每应用一步就先记账再量。）
    """
    p = build(_rich_report())
    from kernoscope.packet import enforce_budget

    trimmed = enforce_budget(p, limit=limit)
    size = packet_bytes(p)
    # 要么真的压进预算，要么如实标记压不进去（默认 4096 是能达到的，见另一个用例）
    assert size <= limit or p["cost"].get("over_budget") is True, (limit, size, trimmed)
    assert p["cost"]["trimmed"] == trimmed
    assert size == p["cost"]["packet_bytes"]


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
            elif key == "mb" and v.get("value") is not None:
                # `mb` 不是独立信息，而是 `value`（字节）的另一种单位表示；一屏按人读单位
                # 印它（7.27 TB），所以这里断言**同源**而不是子串 —— 断言关系比断言子串更强。
                assert float(val) == pytest.approx(float(v["value"]) / 1e6, rel=1e-6), v["id"]
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


def test_verdict_with_failed_premise_is_downgraded_to_refused():
    """外部评测暴露的漏洞：0.9 s 窗口 4 个 kernel、被匹配 kernel 只占 41% 时，
    per-launch 判词的 status 仍是 soft —— agent 只看 status，于是把它当结论。
    规则：前提不成立 ⇒ 必须 refused（值降级为 facts.uncitable_value，历史仍可见）。"""
    from kernoscope.packet import _downgrade_broken_premises

    v = Verdict(id="dram_bytes_per_launch", question="q", status=SOFT,
                facts={"mb": 25.0}, value=2.5e7, unit="B", premises=[])
    v.premises.append(Premise("matched_kernel_kernel_share_pct", 41.0, 95.0, False, "%"))
    _downgrade_broken_premises([v])
    assert v.status == REFUSED
    assert v.value is None
    assert v.refusal is not None
    assert v.refusal.code.value == "share_too_low"
    assert v.facts["uncitable_value"] == 2.5e7
    d = v.to_dict()
    assert d["status"] == "refused" and "value" not in d
    assert d["facts"]["uncitable_value"] == 2.5e7


def test_serializing_a_soft_verdict_with_failed_premise_raises():
    """旁路保护：不一致的判词在序列化时必须炸，而不是悄悄发出去。"""
    v = Verdict(id="x", question="q", status=SOFT, facts={}, value=1.0,
                premises=[Premise("window_coverage_pct", 71.4, 80.0, False, "%")])
    with pytest.raises(ValueError, match="premise"):
        v.to_dict()


def test_no_packet_verdict_is_soft_with_a_failed_premise():
    """端到端守卫：任何一次组包都不许出现"前提不成立但 status 是 ok/soft"。"""
    p = build(_rich_report())
    for v in p["verdicts"]:
        if v["status"] != "refused":
            assert all(pr.get("ok") for pr in (v.get("premises") or [])), v["id"]


# --------------------------------------------------------------------------
# schema v1.1：actionable 是追加字段，v1.0 的包必须仍然能读
# --------------------------------------------------------------------------
def test_packet_carries_schema_revision_and_actionable():
    from kernoscope.packet import PACKET_SCHEMA_REVISION, PACKET_SCHEMA_VERSION

    p = build(_rich_report())
    assert p["schema_version"] == PACKET_SCHEMA_VERSION == 1     # 主版本不动
    assert p["schema_revision"] == PACKET_SCHEMA_REVISION == "1.1"
    ids = {v["id"] for v in p["verdicts"]}
    assert ids, "这个报告应当产出判词"
    for v in p["verdicts"]:
        assert isinstance(v["actionable"], bool)
        if v["status"] == "refused":
            assert v["actionable"] is False
        else:
            assert v["actionable"] is True
            assert all(pr.get("ok") for pr in (v.get("premises") or []))


def test_v10_packet_without_actionable_still_reads_and_renders():
    """向后兼容：v1.0 的包没有 actionable/schema_revision，读包的人要能回退推导。"""
    from kernoscope.packet import verdict_is_actionable

    p = build(_rich_report())
    p.pop("schema_revision", None)
    for v in p["verdicts"]:
        v.pop("actionable", None)
    # 回退规则：不是 refused 且前提全部成立
    for v in p["verdicts"]:
        expect = v["status"] != "refused" and all(pr.get("ok")
                                                  for pr in (v.get("premises") or []))
        assert verdict_is_actionable(v) is expect, v["id"]
    text = render_onepage(p)
    assert "Kernoscope" in text


def test_actionable_helper_rules_match_the_explicit_field():
    """v1.0 回退规则必须与 v1.1 显式字段**完全一致**（否则新旧包判词含义不同）。"""
    from kernoscope.packet import verdict_is_actionable

    # 没列前提的判词（例如"最耗时的 kernel 是 X"这种一条实测事实）仍可依据
    assert verdict_is_actionable({"id": "x", "status": "soft"}) is True
    assert verdict_is_actionable({"id": "x", "status": "ok", "premises": []}) is True
    assert verdict_is_actionable({"id": "x", "status": "ok",
                                  "premises": [{"id": "p", "ok": False}]}) is False
    # 显式字段优先于推导
    assert verdict_is_actionable({"id": "x", "status": "soft", "actionable": False}) is False


# --------------------------------------------------------------------------
# 量纲：用**入库真机产物**回归（R-51）
# --------------------------------------------------------------------------
ARTIFACTS = pathlib.Path(__file__).resolve().parents[1] / "bench" / "artifacts"


def _real_csv_metrics() -> dict[str, tuple[float, str]]:
    """从入库真机产物的 CSV 里读 (value, unit)。CSV 是报告自己的输出，单位是真的。"""
    out: dict[str, tuple[float, str]] = {}
    for path in sorted(ARTIFACTS.glob("*/rep-copy.csv")):
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name, raw, unit = row.get("metric"), row.get("value"), row.get("unit")
                if name and raw and name.startswith("est__dram_bytes"):
                    out.setdefault(name, (float(raw), unit or ""))
    return out


def _report_with_real_dram_units() -> tuple[Report, dict[str, tuple[float, str]]]:
    rep = _rich_report()
    real = _real_csv_metrics()
    assert "est__dram_bytes_total" in real, "入库产物里应当有真机 dram 指标"
    # 用真机的 value+unit 换掉 fixture 里的字节口径值；前提给足，保证判词是 soft 而不是 refused
    reps = {
        "est__dram_bytes_total": real["est__dram_bytes_total"],
        "est__dram_bytes_per_launch": real.get("est__dram_bytes_per_launch", (2.5e7, "MB")),
    }
    for sec in rep.kernels[0].sections:
        for i, m in enumerate(sec.metrics):
            if m.name in reps:
                val, unit = reps[m.name]
                sec.metrics[i] = dataclasses.replace(m, value=val, unit=unit)
        sec.add(Metric.derived("ks__sampler_window_coverage_pct", 99.5, "%"))
        sec.add(Metric.derived("ks__matched_kernel_kernel_share_pct", 100.0, "%"))
    return rep, real


def test_dram_verdict_units_are_bytes_on_a_real_run():
    """真机 metric 的单位是 MB，包里的契约是字节 —— 两者必须只差一次换算，不是 10⁶。

    这条**能红**：把 `metric_bytes()` 换回 `float(m.value)`（旧实现），
    `dram_bytes.value` 会变成 7273090.3 而不是 7.2730903e12。
    """
    rep, real = _report_with_real_dram_units()
    p = build(rep)
    by_id = {v["id"]: v for v in p["verdicts"]}
    for vid, name in (("dram_bytes", "est__dram_bytes_total"),
                      ("dram_bytes_per_launch", "est__dram_bytes_per_launch")):
        val_mb, unit = real[name]
        v = by_id[vid]
        assert v["unit"] == "B", f"{vid}: 包里 dram 字节一族的单位必须是 B"
        assert v["value"] == pytest.approx(to_bytes(val_mb, unit), rel=1e-9), \
            f"{vid}: 包里 {v['value']} 与产物里的 {val_mb} {unit} 不是同一个量"
        assert v["facts"]["mb"] == pytest.approx(v["value"] / 1e6, rel=1e-6), \
            f"{vid}: value 与 facts.mb 必须同源"
        assert v["facts"].get("source_unit") == (unit or "B"), f"{vid}: 要留下原单位"
    # 散文（一屏）里印的量级也必须与 value 同源：它由 phrase_verdict 从 facts/value 生成
    text = render_onepage(p)
    for vid in ("dram_bytes", "dram_bytes_per_launch"):
        assert fmt_bytes(by_id[vid]["value"]) in text, (vid, text)


def test_packet_dram_value_matches_the_documented_terabyte_scale():
    """入库产物的报告写 7.273 TB；包必须同量级（这条就是 7.3 MB 那个 bug 的回归）。"""
    rep, real = _report_with_real_dram_units()
    p = build(rep)
    v = next(x for x in p["verdicts"] if x["id"] == "dram_bytes")
    assert v["value"] > 1e12, v["value"]
    onepage = render_onepage(p)
    assert "TB" in onepage and "7.3 MB" not in onepage, onepage
    assert "MB" == real["est__dram_bytes_total"][1], \
        "这条回归的意义就在于：真机单位是 MB，而包里的契约是字节"


def test_unknown_unit_is_refused_not_guessed():
    """单位认不出来时必须抛错，不许当字节用 —— 猜错就是 10⁶ 级。"""
    from kernoscope.provenance import ProvenanceError, to_bytes

    assert to_bytes(1.0, "MB") == 1e6
    assert to_bytes(1.0, "") == 1.0
    assert to_bytes(1.0, "MiB") == 2.0 ** 20
    with pytest.raises(ProvenanceError):
        to_bytes(1.0, "MB/s")


# --------------------------------------------------------------------------
# 协议哈希的输入必须由**管线**写（R-49），不能靠测试手工塞 meta
# --------------------------------------------------------------------------
CALIB_A = ARTIFACTS / "20261009-012257" / "machine-profile.json"
CALIB_B = ARTIFACTS / "20261008-235142" / "machine-profile.json"


def _packet_from_cli(tmp_path, name: str, *extra: str) -> dict:
    """**经 CLI** 产出一个包（不是手搓 Report），路径：--ks-synthetic + 入库标定产物。"""
    from kernoscope.cli import main

    out = tmp_path / f"{name}.json"
    rc = main(["--ks-synthetic", "--ks-no-nvml", "--set", "basic",
               "--packet", str(out), "--screen", "off", "--quiet", *extra])
    assert rc == 0, f"CLI 失败：{extra}"
    return json.loads(out.read_text())


@pytest.mark.skipif(not CALIB_A.exists() or not CALIB_B.exists(),
                    reason="需要入库的标定产物")
def test_pipeline_writes_every_protocol_input(tmp_path):
    """换标定 / 换窗口必须改变协议哈希 —— 这条**只能经 CLI 测**。

    旧测试（`test_protocol_hash_changes_with_caliber`）是手工 `c.meta["calibration"]=...`
    再断言哈希变化：它证明的是**函数**支持这个键，不是**管线**会填它。而 CLI 从来不写这组键，
    于是生产里 `calibration/peaks/kernel/warmup_s/window_s` 全是 null ——
    "换了一份标定（roofline 唯一的分母）"照样算同协议，`diff` 照样报"真改进"。
    """
    base = _packet_from_cli(tmp_path, "base", "--ks-calibration", str(CALIB_A))
    a = base["protocol"]["hash"]
    # ① 同一份标定、同一条命令 ⇒ 同哈希（不含路径/时间戳，否则每次都不可比）
    assert _packet_from_cli(tmp_path, "base2", "--ks-calibration", str(CALIB_A))["protocol"]["hash"] == a
    # ② 换标定（峰值不同）⇒ 必不同
    other = _packet_from_cli(tmp_path, "calib_b", "--ks-calibration", str(CALIB_B))
    assert other["protocol"]["hash"] != a, "换标定必须改变协议哈希"
    # ③ 窗口参数 ⇒ 必不同（D5 的量就随它变，不能算同协议）
    for extra in (("--ks-warmup", "40"), ("--ks-window-s", "20"), ("--ks-window", "full")):
        got = _packet_from_cli(tmp_path, f"w_{extra[0]}", "--ks-calibration", str(CALIB_A), *extra)
        assert got["protocol"]["hash"] != a, f"{extra} 必须改变协议哈希"


@pytest.mark.skipif(not CALIB_A.exists() or not CALIB_B.exists(),
                    reason="需要入库的标定产物")
def test_calibration_change_makes_diff_incomparable(tmp_path):
    """R-49 的验收：换了标定（分母变了）之后，`diff` 必须拒绝给方向。"""
    a = _packet_from_cli(tmp_path, "a", "--ks-calibration", str(CALIB_A))
    b = _packet_from_cli(tmp_path, "b", "--ks-calibration", str(CALIB_B))
    res = diff_packets(a, b)
    assert not res.comparable, "协议不同必须不可比"
    assert "protocol_hash" in res.protocol_diffs
    # 没有任何一条判词可以带方向
    assert all(c.status == "incomparable" for c in res.comparisons if c.a is not None)


def test_protocol_payload_has_no_paths_or_timestamps(tmp_path):
    """协议输入里不许出现本机路径或标定时间戳：那会让"同口径"每次都变成不可比，
    也会把本机目录写进包（包是要发出去的）。"""
    p = _packet_from_cli(tmp_path, "clean", "--ks-calibration", str(CALIB_A))
    blob = json.dumps(p, ensure_ascii=False)
    assert str(ARTIFACTS) not in blob and "/Users/" not in blob and "C:\\" not in blob
    cal = p["protocol"].get("calibration")
    if cal is not None:                      # 暴露与否都可；暴露了就必须是"文件名+内容哈希"
        assert "/" not in cal and ":" in cal
