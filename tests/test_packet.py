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
from kernoscope.packet import BUILDERS
import kernoscope.packet as packet_mod
from kernoscope.refusals import CONDITIONAL_PROVENANCE, DEFAULT_REMEDIES
from kernoscope.refusals import RefusalCode, conditional_fact_keys
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


def _report_with_all_verdicts() -> Report:
    """让 11 条 builder 全部触发（instrument 齐全的机器就是这样）。"""
    rep = _rich_report()
    extra = {"est__bound": "memory", "ks__static_sass_instructions": 4321,
             "est__roofline_efficiency_pct": 91.2, "est__achieved_gbs": 908.5,
             "ks__over_peak_warning": 1.0, "ks__static_sass_share_fp": 41.0,
             "ks__static_sass_share_memory": 33.0, "ks__static_sass_share_control": 12.0,
             "ks__static_sass_share_int": 14.0}
    for sec in rep.kernels[0].sections:
        for i, m in enumerate(sec.metrics):
            if m.name in extra:
                sec.metrics[i] = dataclasses.replace(m, value=extra[m.name])
        if not any(m.name == "ks__over_peak_warning" for m in sec.metrics):
            sec.add(Metric.derived("ks__over_peak_warning", 1.0, note="achieved > measured peak"))
    return rep


def test_verdict_truncation_leaves_a_trace():
    """R-53/D10：`MAX_VERDICTS` 截断必须留痕。

    旧实现先 `[:10]` 再进预算裁剪，于是 `cost.trimmed` 从不列 `verdicts`、也没有
    `verdicts_more` —— 读者（人或 agent）**无从知道**包里少了一条判词。
    """
    rep = _report_with_all_verdicts()
    fired = [v.id for v in (fn(rep) for fn in BUILDERS) if v is not None]
    assert len(fired) > MAX_VERDICTS, f"这个 fixture 要让 11 条 builder 全触发，实得 {fired}"
    p = build(rep)
    assert len(p["verdicts"]) == MAX_VERDICTS
    more = p.get("verdicts_more")
    assert more and more["count"] == len(fired) - MAX_VERDICTS
    assert set(more["ids"]) == set(fired) - {v["id"] for v in p["verdicts"]}


def test_refusals_are_trimmable_so_the_budget_holds_in_a_locked_down_env():
    """R-52/D9：**受限环境天然产生十几个拒绝码**，包预算不能因此被破。

    旧实现只删 refusal 的 `cmd` 子字段，条目本身不参与裁剪：45 个 code 时包 8883 B
    （预算 4096），而"硬约束"只写在散文里。现在有序降级包含处置文本截短与条目上限，
    并把丢掉的 code 列进 `refusals_omitted`（列表过长时明确标出）。
    """
    codes = [c for c in RefusalCode if c in DEFAULT_REMEDIES][:45]
    assert len(codes) == 45
    rep = _rich_report()
    sec = Section("受限环境下的未测项")
    for i, c in enumerate(codes):
        sec.add(Metric.unavailable(f"est__blocked_{i}", "本机拿不到这个数据源", code=c))
    rep.kernels[0].sections.append(sec)
    p = build(rep)
    size = packet_bytes(p)
    assert size <= PACKET_SIZE_BUDGET_BYTES, f"{size} B 超预算（拒绝条目必须可裁剪）"
    assert "over_budget" not in p["cost"]
    # R-110：`cost.trimmed` 现在只记**真正改动了包**的步骤（旧实现连 no-op 也记，于是
    # "裁剪步骤"不等于"真的裁了什么"）。所以这里断言"拒绝侧确实被裁过"，而不是钉住
    # 某个在这个 fixture 里可能恰好是 no-op 的步骤名。
    assert {"refusal_entries", "refusal_action_text"} & set(p["cost"]["trimmed"]), \
        f"拒绝条目/处置文本必须被裁过：{p['cost']['trimmed']}"
    omitted = p.get("refusals_omitted") or {}
    assert omitted.get("count", 0) > 0, "丢掉的 code 必须计数"
    assert omitted.get("codes"), "至少要列出丢掉的 code（列表过长时标 codes_truncated）"
    assert len(p["refusals"]) <= 8


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
    assert p["schema_revision"] == PACKET_SCHEMA_REVISION == "1.2"
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


def test_over_peak_refusal_propagates_into_roofline():
    """R-59/D18：`over_peak` 成立时，由它算出的 roofline 判词不能被当成依据。

    真机包实测过：`roofline status=soft actionable=True value=24041.6 %` 与
    `over_peak status=refused` **并存**。按「只有 actionable=true 能当依据」这条规矩，
    那个物理不可能的达成率是"可依据"的，diff 还会拿它报 improved。
    """
    p = build(_report_with_all_verdicts())
    by_id = {v["id"]: v for v in p["verdicts"]}
    assert by_id["over_peak"]["status"] == "refused"
    roof = by_id["roofline"]
    assert roof["status"] == "refused", roof
    assert roof.get("actionable") is False
    assert roof.get("value") is None
    assert roof["facts"].get("uncitable_value") == 91.2, "原值要留在 facts 里可查"
    assert roof["refusal"]["code"] == "calibration_caliber_mismatch"
    from kernoscope.packet import verdict_is_actionable
    assert verdict_is_actionable(roof) is False
    text = render_onepage(p)
    # 一屏要把"怎么办"给出来（按 code 查 remedy），而不是留一句"已达 roofline"
    assert "不给数字" in text and "标定口径" in text
    assert "已达 roofline" not in text
    # 没有 over_peak 时 roofline 必须照旧可依据（别把正常情况也拒了）
    plain = build(_rich_report())
    r2 = next(v for v in plain["verdicts"] if v["id"] == "roofline")
    assert r2["status"] != "refused" and r2.get("actionable") is True


def test_written_packet_size_equals_packet_bytes(tmp_path):
    """D30②：落盘字节数必须**等于** `cost.packet_bytes`（它是读者拿来比 4 KB 预算的数）。

    旧实现写 `text + "\\n"` 却返回 `len(text)`，于是文件总比包声称的多 1 B —— 一个
    "数字与文件对不上"的小谎，而这类小谎正是这个仓库最不该有的东西。
    """
    from kernoscope.packet import write_packet

    p = build(_rich_report())
    path, n = write_packet(p, str(tmp_path / "pk.json"))
    on_disk = (tmp_path / "pk.json").stat().st_size
    assert n == on_disk == p["cost"]["packet_bytes"], (n, on_disk, p["cost"]["packet_bytes"])


# --------------------------------------------------------------------------
# R-65：包里**被引用到的每个 code** 都必须查得到处置（+ 真能走到 refusal_entries 的 fixture）
# --------------------------------------------------------------------------
def test_refusal_codes_cited_in_the_packet_are_always_resolvable():
    """R-65/N2：不变量 —— `cited ⊆ keys(refusals) ∪ refusals_omitted.codes`。

    第四轮验收报告按不变量扫了 47 个构造场景，**32 个**有悬空 code（例如
    `share_unknown`、`trace_unavailable`）：`_cap_refusal_entries` 按字母序留 8 条，
    而 `unmeasured` 另按组截断 ⇒ 被引用的 code 可能整条消失、又不在 `refusals_omitted`
    里；一屏会印"还有 N 个 `no_occupancy_probe`"却不给办法。受限环境正是命中它的场景。

    这里用三种场景覆盖（含报告没试过的 `environment_refusals`：它的 code 此前**从不**
    进拒绝表，是结构性的悬空）。
    """
    from kernoscope.provenance import Section
    from kernoscope.refusals import DEFAULT_REMEDIES, RefusalCode

    def check(p: dict, tag: str) -> None:
        cited = packet_mod._cited_codes(p)
        have = (set((p.get("refusals") or {}).keys())
                | set((p.get("refusals_omitted") or {}).get("codes") or []))
        assert not (cited - have), f"{tag}: 悬空 code {sorted(cited - have)}"
        assert len(dumps(p).encode("utf-8")) <= 4096, f"{tag}: 超预算"

    codes = [c for c in RefusalCode if c in DEFAULT_REMEDIES]

    # ① 47 个 code 全塞进拒绝表
    rep = _rich_report()
    sec = Section("受限环境")
    for i, c in enumerate(codes):
        sec.add(Metric.unavailable(f"est__blocked_{i}", "本机拿不到", code=c))
    rep.kernels[0].sections.append(sec)
    p1 = build(rep)
    assert "refusal_entries" in p1["cost"]["trimmed"], "这个 fixture 必须真的走到条目上限那一步"
    check(p1, "47 code")

    # ② environment_refusals 的 code（曾经结构性悬空）
    rep2 = _rich_report()
    rep2.meta["environment_refusals"] = [
        {"code": c, "message": "缺件降级"}
        for c in ("nsys_missing", "nvbit_missing", "cuobjdump_missing", "no_nvml",
                  "pytorch_missing", "no_gpu", "no_calibration", "no_memtrace",
                  "no_instr_hist", "no_sass_source", "no_static_input", "no_workload")]
    p2 = build(rep2)
    check(p2, "env×12")

    # ③ 两者叠加
    rep3 = _rich_report()
    sec3 = Section("受限环境")
    for i, c in enumerate(codes):
        sec3.add(Metric.unavailable(f"est__b{i}", "本机拿不到", code=c))
    rep3.kernels[0].sections.append(sec3)
    rep3.meta["environment_refusals"] = rep2.meta["environment_refusals"]
    p3 = build(rep3)
    check(p3, "叠加")


def test_onepage_says_what_was_left_out():
    """R-70/D10：一屏不能只说"裁了哪一步"，要说**少了哪几条**。

    全仓此前除 `packet.py` 外没有任何视图引用 `verdicts_more` / `refusals_omitted` /
    `unmeasured_more` —— 机器契约有了，人视图没有。
    """
    rep = _rich_report()
    rep.meta["environment_refusals"] = [
        {"code": c, "message": "缺件降级"} for c in
        ("nsys_missing", "nvbit_missing", "cuobjdump_missing", "no_nvml", "pytorch_missing",
         "no_gpu", "no_calibration", "no_memtrace", "no_instr_hist", "no_sass_source")]
    p = build(rep)
    assert p.get("refusals_omitted"), "这个 fixture 应当触发条目省略"
    text = render_onepage(p)
    assert "拒绝条目省了" in text and str(p["refusals_omitted"]["count"]) in text
    assert (p["refusals_omitted"]["codes"] or [""])[0] in text


def _budget_fixture() -> dict:
    """拒绝密集 + 判词多的包（V-1 与 facts 分级裁剪都用它）。"""
    from kernoscope.packet import PRIMARY_FACT_IDS

    return {
        "schema_version": 1, "schema_revision": "1.2", "tool": "kernoscope",
        "tool_version": "0.2.0",
        "verdicts": ([
            {"id": "kernel_time", "status": "ok", "facts": {"top": "k", "top_ms": 37.8},
             "metrics": ["gpu__time_duration.sum"], "premises": [{"id": "x", "observed": 1.0}]},
            {"id": "cost_per_token", "status": "ok",
             "facts": {"per_mtok": 0.0089, "tokens_s": 62972.0},
             "metrics": ["ks__cost_per_mtok"]},
            {"id": "saving_estimate", "status": "ok",
             "facts": {"n": 1, "top_pct_lo": 79.0, "top_pct_hi": 88.2},
             "metrics": ["ks__prescription_total_saving_per_hour"]},
            {"id": "truth_vs_substitute", "status": "ok", "facts": {"n": 3},
             "metrics": ["ncu__gpu__time_duration.sum"]},
        ] + [{"id": f"extra_{i}", "status": "ok",
              "facts": {"padding": "x" * 60}, "metrics": [f"ks__m{i}"],
              "premises": [{"id": "p", "observed": i, "ok": True}]} for i in range(12)]),
        "unmeasured": [{"code": "counters_denied", "count": 3,
                        "examples": ["sm__a", "sm__b"]} for _ in range(4)],
        "refusals": {f"code_{i}": {"action": "很长很长的处置文本" * 12, "cmd": "kernoscope x",
                                   "cost_kind": "wall_seconds", "cost_s": 20}
                     for i in range(8)},
        "next": [{"verdict": f"v{i}", "action": "下一步" * 20} for i in range(6)],
    }


def test_primary_verdict_facts_survive_the_first_facts_trim():
    """拒绝密集的机器上，facts 是**分两级**裁的：先裁次要判词，结论四条最后才动。

    真机教训：一处 GEMM 负载 + 8 个拒绝码时，包被裁到"全部 facts 丢失"，
    一屏上出现"最耗时的是 ?（None ms）"—— 结论本身被裁没了。
    """
    from kernoscope.packet import PRIMARY_FACT_IDS, enforce_budget

    # ① 宽松一点：次要 facts 已经裁掉，结论四条的 facts 必须还在
    packet = _budget_fixture()
    enforce_budget(packet, limit=2600)
    trimmed = packet["cost"]["trimmed"]
    assert "facts_secondary" in trimmed, f"没走到次要 facts 裁剪：{trimmed}"
    assert "facts_primary" not in trimmed, f"这个预算不该动结论 facts：{trimmed}"
    by_id = {v["id"]: v for v in packet["verdicts"]}
    for vid in PRIMARY_FACT_IDS:
        if vid in by_id:
            assert by_id[vid].get("facts"), f"{vid} 的 facts 是结论本身，不该先被裁"
    assert any(not v.get("facts") for k, v in by_id.items() if k.startswith("extra_"))

    # ② 极紧预算：才允许动结论 facts，而且必须是**最后**一步、留下痕迹
    tight = _budget_fixture()
    enforce_budget(tight, limit=1400)
    t2 = tight["cost"]["trimmed"]
    assert t2.index("facts_secondary") < t2.index("facts_primary"), t2
    assert t2[-1] == "facts_primary", f"结论 facts 必须是最后一级：{t2}"


def test_budget_drops_refusal_prose_before_it_drops_verdicts():
    """V-1（第五轮 P0）：结论比处置散文重要 —— 顺序必须是"先裁散文，再裁判词"。

    旧顺序把 `verdicts` 排在 `refusal_entries`/`refusal_action_text` 之前，与函数自己的
    docstring 相反：真机 9 条判词 → 包 6 条，被删的正是 `occupancy`/`roofline`/`over_peak`
    （含"物理不可能"守卫），而 8 条拒绝的处置散文一条不少。
    """
    from kernoscope.packet import enforce_budget

    packet = _budget_fixture()
    enforce_budget(packet, limit=3200)          # 只够裁到拒绝散文这一级
    trimmed = packet["cost"]["trimmed"]
    assert "refusal_action_text" in trimmed or "refusal_entries" in trimmed, trimmed
    if "verdicts" in trimmed:                   # 真裁到判词时，散文必须已经先被裁过
        assert trimmed.index("refusal_action_text") < trimmed.index("verdicts"), trimmed
    # 只裁散文的那一档：判词一条不许少
    assert len(packet["verdicts"]) == 16, [v["id"] for v in packet["verdicts"]]


def test_budget_never_drops_a_verdict_without_recording_it():
    """V-1 的另一半：预算裁掉的判词 id 必须进 `verdicts_more`，不许静默消失。"""
    from kernoscope.packet import enforce_budget

    packet = _budget_fixture()
    original = [v["id"] for v in packet["verdicts"]]
    enforce_budget(packet, limit=1400)
    kept = [v["id"] for v in packet["verdicts"]]
    more = packet.get("verdicts_more") or {}
    recorded = list(more.get("ids") or [])
    assert set(original) == set(kept) | set(recorded), (kept, recorded)
    assert more.get("count") == len(original) - len(kept)
    assert len(recorded) == more.get("count")


def test_verdict_caps_are_both_traced():
    """`MAX_VERDICTS` 截断与预算截断共用一个留痕字段（count 是两者之和）。"""
    from kernoscope.packet import enforce_budget

    packet = _budget_fixture()
    enforce_budget(packet, limit=1000)
    more = packet["verdicts_more"]
    assert more["count"] == 16 - len(packet["verdicts"]) == len(more["ids"])
    assert len(packet["verdicts"]) <= 4


# --------------------------------------------------------------------------- #
# 第五轮 V-6：跨 kernel 传染（一份报告里 kernel A 的超峰污染 kernel B 的 roofline）
# --------------------------------------------------------------------------- #
class _KM:
    def __init__(self, name, value, unit="", provenance="measured", source="s", note=""):
        from kernoscope.provenance import Provenance
        self.name, self.value, self.unit = name, value, unit
        self.provenance = Provenance(provenance)
        self.source, self.note = source, note


class _KSec:
    def __init__(self, name, metrics):
        self.name, self.metrics = name, metrics


class _KK:
    def __init__(self, name, total_ns, sections, launch_count=1):
        self.name, self.total_ns, self.sections = name, total_ns, sections
        self.launch_count = launch_count


class _KRep:
    def __init__(self, kernels):
        self.kernels = kernels
        self.meta = {}
        self.device = None
        self.wall_time_ns = sum(k.total_ns for k in kernels)
        self.tool_version = "test"
        self.schema_version = 1

    def iter_sections(self):
        for k in self.kernels:
            yield from k.sections


def _two_kernel_report(*, cheap_over_peak: bool, top_over_peak: bool,
                       same_prefix: bool = False):
    # 第六轮 W-4 的构造：两个**不同** kernel 的名字前 48 字相同（C++ 模板实例化名的常态）。
    # `_clip(name, 48)` 的显示串因此完全一样 —— 用显示串当身份就会互相污染。
    prefix = "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_"
    cheap_name = f"{prefix}cheap" if same_prefix else "cheap_kernel"
    top_name = f"{prefix}top" if same_prefix else "top_kernel"
    cheap = _KK(cheap_name, 1_000_000, [_KSec("SpeedOfLight", [
        _KM("gpu__time_duration.sum", 1.0, "ms", "exact"),
        _KM("est__roofline_efficiency_pct", 20.0, "%", "estimated"),
        *([_KM("ks__over_peak_warning", 1.0, "", "derived")] if cheap_over_peak else []),
    ])])
    top = _KK(top_name, 9_000_000, [_KSec("SpeedOfLight", [
        _KM("gpu__time_duration.sum", 9.0, "ms", "exact"),
        _KM("est__bound", "memory", "", "estimated"),
        _KM("est__roofline_efficiency_pct", 91.0, "%", "estimated"),
        *([_KM("ks__over_peak_warning", 1.0, "", "derived")] if top_over_peak else []),
    ])])
    return _KRep([cheap, top])


def test_over_peak_of_another_kernel_does_not_poison_a_legit_roofline():
    """真机可达：便宜 kernel 超峰 → 另一个 kernel 合法的 91% roofline 被判 refused。"""
    from kernoscope.packet import build

    p = build(_two_kernel_report(cheap_over_peak=True, top_over_peak=False))
    by_id = {v["id"]: v for v in p["verdicts"]}
    assert by_id["over_peak"]["facts"]["kernel"] == "cheap_kernel"
    assert by_id["roofline"]["status"] != "refused", by_id["roofline"]
    assert by_id["roofline"]["facts"]["efficiency_pct"] == 91.0
    assert by_id["roofline"]["facts"]["kernel"] == "top_kernel"


def test_same_kernel_over_peak_still_poisons_its_own_roofline():
    """同 kernel 的超峰仍必须把 roofline 打掉（R-59/D18 的既有语义不许丢）。"""
    from kernoscope.packet import build

    p = build(_two_kernel_report(cheap_over_peak=False, top_over_peak=True))
    by_id = {v["id"]: v for v in p["verdicts"]}
    assert by_id["over_peak"]["facts"]["kernel"] == "top_kernel"
    assert by_id["roofline"]["status"] == "refused"
    assert by_id["roofline"].get("value") is None
    assert by_id["roofline"]["facts"]["uncitable_value"] == 91.0


def test_over_peak_identity_is_not_the_truncated_display_string():
    """第六轮 W-4：判词身份不能用 `facts["kernel"]` —— 那是 `_clip(name, 48)` 的**显示串**。

    真机 A/B：两个不同 kernel 名前 48 字相同（`void cutlass::Kernel2<…>…` 这类模板实例化名
    极易撞车）⇒ 便宜 kernel 的超峰把另一个 kernel 合法的 91% roofline 打成
    `refused/actionable=False`；把前缀改成不同 ⇒ `soft/actionable=True/91.0`。
    这里先断言夹具真的让两个显示串相同（否则测试就是假的），再断言 roofline 没被否决。
    """
    from kernoscope.packet import build

    p = build(_two_kernel_report(cheap_over_peak=True, top_over_peak=False, same_prefix=True))
    by_id = {v["id"]: v for v in p["verdicts"]}
    assert by_id["over_peak"]["facts"]["kernel"] == by_id["roofline"]["facts"]["kernel"], \
        "夹具没生效：两个 kernel 的**显示串**必须相同（这正是被击穿的地方）"
    assert by_id["roofline"]["status"] != "refused", by_id["roofline"]
    assert by_id["roofline"]["facts"]["efficiency_pct"] == 91.0


def test_same_prefix_over_peak_still_poisons_its_own_kernel():
    """反面对照：同一个 kernel（同一前缀、同一个 owner_key）的超峰仍必须打掉它自己的 roofline。"""
    from kernoscope.packet import build

    p = build(_two_kernel_report(cheap_over_peak=False, top_over_peak=True, same_prefix=True))
    by_id = {v["id"]: v for v in p["verdicts"]}
    assert by_id["over_peak"]["facts"]["kernel"] == by_id["roofline"]["facts"]["kernel"]
    assert by_id["roofline"]["status"] == "refused", by_id["roofline"]


def test_hard_cap_keeps_the_four_conclusion_verdicts_even_when_they_come_last():
    """第六轮 R-106：`_cap_verdicts` 旧实现是 `vs[:n]` —— **靠位置**，只因为 `BUILDERS`
    的顺序恰好把结论排在前面才没丢；集群②改一下顺序就构造出"结论四条整条掉出包"的红。

    这里把结论四条**放在最后**，断言任何预算下它们都不消失，且 `verdicts_more.ids` 保持原序。
    """
    from kernoscope.packet import PRIMARY_FACT_IDS, enforce_budget

    ids = [f"extra_{i}" for i in range(6)] + list(PRIMARY_FACT_IDS)
    packet = {"verdicts": [{"id": i, "status": "ok"} for i in ids],
              "unmeasured": [], "refusals": {}, "cost": {}}
    enforce_budget(packet, limit=260)
    kept = [v["id"] for v in packet["verdicts"]]
    missing = [i for i in PRIMARY_FACT_IDS if i not in kept]
    assert not missing, f"结论判词被静默裁掉：{missing}（留下的：{kept}）"
    dropped = (packet.get("verdicts_more") or {}).get("ids") or []
    assert dropped == [i for i in ids if i not in kept], \
        f"verdicts_more.ids 必须是**原序**里被丢掉的那些：{dropped}"


def test_trimmed_facts_are_recorded_by_item_id():
    """第六轮 R-106：facts 被裁以前只有步骤名（`facts_primary`），没有条目级留痕 ——
    2400 个场景里 facts 被裁 663 次，全都不知道是谁的 facts 没了。"""
    from kernoscope.packet import enforce_budget

    packet = _budget_fixture()
    enforce_budget(packet, limit=2600)
    got = packet["cost"].get("trimmed_ids") or []
    assert any(s.startswith("facts_secondary:") for s in got), \
        f"被裁 facts 的判词 id 必须有留痕：{packet['cost'].get('trimmed')}"
    live_ids = {v["id"] for v in packet["verdicts"]}
    for s in got:
        step, _, vid = s.partition(":")
        assert step and vid, f"留痕格式应为 `<步骤>:<判词 id>`：{s!r}"
        assert vid in live_ids, f"记了一个包里不存在的判词 id：{s!r}"


def test_trimmed_only_records_steps_that_actually_changed_the_packet():
    """R-110：`cost.trimmed` 的语义是"真的改动了包的步骤"，不是"走到了哪一步"（旧实现连
    no-op 也记）。用一个人造包：所有裁剪步骤都无处可裁 ⇒ `trimmed` 必须是空的。"""
    from kernoscope.packet import enforce_budget

    packet = {"verdicts": [], "unmeasured": [], "refusals": {}, "cost": {},
              "refusals_omitted": {}}
    trimmed = enforce_budget(packet, limit=1)          # 逼它把所有步骤都走完
    assert trimmed == [], f"没有东西可裁时不该记任何步骤：{trimmed}"


def test_device_level_over_peak_verdict_does_not_silently_disappear():
    """第六轮 W-12：超峰警告只挂在 `(device)` 伪 kernel 上时（`-i` 回读老报告可达），
    `kernel_with` 会**跳过** `(device)` ⇒ 旧实现让 `over_peak` 判词静默消失
    （相对更早的 `find_metric` 行为是回归）。这里退回设备级读取。"""
    from kernoscope.packet import build

    dev = _KK("(device)", 1_000_000, [_KSec("SpeedOfLight", [
        _KM("ks__over_peak_warning", 1.0, "", "derived"),
    ])])
    p = build(_KRep([dev]))
    ids = [v["id"] for v in p["verdicts"]]
    assert "over_peak" in ids, f"设备级超峰警告必须变成判词，而不是消失：{ids}"
    op = next(v for v in p["verdicts"] if v["id"] == "over_peak")
    assert op["status"] == "refused" and op["actionable"] is False


def test_discovery_ranking_survives_without_calibration_or_price():
    """M13（目标第二次修订）：缺标定与价格时，「钱在哪、先改什么」不许消失。

    实测（收口这一轮）：只有 nsys 的机器上 `saving_estimate` **整条不见** ——
    而"按实测时间占比排序"本来就不需要标定与价格（时长是 exact 的）。这条钉两件事：
    ① 判词必须存在（遵守我们自己的"拒绝≠空"）；② 它必须**明说自己不是钱**。
    """
    from kernoscope.packet import build

    rep = _two_kernel_report(cheap_over_peak=False, top_over_peak=False)
    p = build(rep)
    sv = next((v for v in p["verdicts"] if v["id"] == "saving_estimate_share"), None)
    assert sv is not None, f"发现层不许缺席（拒绝≠空）：{[v['id'] for v in p['verdicts']]}"
    assert sv["facts"]["basis"] == "time_share", sv
    assert sv["facts"]["money"] == "unavailable", sv
    assert sv["facts"]["top_kernel"] == "top_kernel"          # 9 ms ≫ 1 ms
    assert sv["facts"]["top_pct"] == pytest.approx(90.0, abs=0.1)
    # `top` 压成一行人类可读、机器可 split 的字符串（一屏契约要求每个 fact 都能被渲染）
    top = sv["facts"]["top"]
    assert top.split("；")[0].startswith("top_kernel ") and "90.0%" in top, top
    assert "cheap_kernel" in top and "10.0%" in top, top
    assert sv["facts"]["needs"] == "calibration,price", sv["facts"]["needs"]
    assert sv["provenance"] == "exact" and sv["unit"] == "%"
    assert sv["actionable"] is True, "排序本身是实测的，可依据"


def test_discovery_ranking_never_pretends_to_be_money():
    """反面：第一级清单里的数字是**时间占比**，不许被表述成节省比例或钱。"""
    from kernoscope.packet import build

    p = build(_two_kernel_report(cheap_over_peak=False, top_over_peak=False))
    sv = next(v for v in p["verdicts"] if v["id"] == "saving_estimate_share")
    facts = sv["facts"]
    assert "total_per_hour" not in facts and "total_per_mtok" not in facts
    assert facts.get("basis") == "time_share"
    assert facts.get("money") == "unavailable"


# --------------------------------------------------------------------------
# 反事实视图（条件答案）：缺**用户必须给的输入**时，拒绝不再 dead-end
# --------------------------------------------------------------------------
def _cost_report(*, price: float | None, tokens: float | None = 1000.0) -> Report:
    """成本口径的实测量齐全、只缺调用方输入的夹具。

    数字刻意取整，条件答案可以手算对拍：
      能量 300 J / 平均功率 300 W ⇒ 窗口 1.0 s；窗口内 10 个 step；10k token/s。
      假设电价 0.8 元/kWh ⇒ 0.3 kWh/h × 0.8 = 0.24 元/h；
      有 token 数时 ⇒ 0.24 × 1e6 ÷ (1e4 × 3600) = 0.006667 元/Mtoken。
    """
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    rep.wall_time_ns = 1_000_000_000
    k = KernelReport("big_gemm", launch_count=10, launches=[_launch(i) for i in range(10)])
    s = Section("CostLedger")
    s.add(Metric.measured("meas__window_energy_j", 300.0, "J", source="NVML 功率积分"))
    s.add(Metric.measured("meas__window_power_mean_w", 300.0, "W", source="能量 ÷ 窗口"))
    s.add(Metric.derived("ks__steps_in_window", 10.0, "", source="窗口对齐的 launch 数"))
    if tokens is not None:
        s.add(Metric.derived("ks__tokens_per_s", 10_000.0, "token/s"))
        s.add(Metric.derived("ks__j_per_token", 0.03, "J/token"))
    if price is not None:
        s.add(Metric.derived("ks__cost_per_hour", 0.24, "CNY/h"))
    else:
        s.add(Metric.unavailable("ks__cost_per_hour", "没有给价格：钱换算不了", "CNY/h",
                                 code=RefusalCode.NO_PRICE))
    if price is not None and tokens is not None:
        s.add(Metric.derived("ks__cost_per_mtok", 0.006667, "CNY/Mtoken"))
    else:
        # 报告的 per_mtok 拒绝码与 cost.py 同源：先看价格、再看 token 数
        code = RefusalCode.NO_PRICE if price is None else RefusalCode.NO_TOKENS
        s.add(Metric.unavailable("ks__cost_per_mtok", "缺输入：单位成本给不出", "CNY/Mtoken",
                                 code=code))
    k.sections = [s]
    rep.kernels = [k]
    return rep


def _cost_verdict(packet: dict) -> dict:
    return next(v for v in packet["verdicts"] if v["id"] == "cost_per_token")


def test_missing_price_gets_a_conditional_cost_answer():
    """**反事实视图**：没有价格时，拒绝里必须带一个命名清楚的"如果……那么……"。

    这条**能红**：把 `_conditional_cost_facts()` 从 ③ 分支去掉，包里的 facts 就只剩
    拒绝码（旧行为），`assumed_price_per_kwh` / `if_cost_per_mtok` 全部消失。
    """
    p = build(_cost_report(price=None))
    v = _cost_verdict(p)
    assert v["status"] == "refused"
    assert v["refusal"]["code"] == "no_price"          # 原拒绝码还在：缺什么必须说清楚
    f = v["facts"]
    assert f["assumed_price_per_kwh"] == 0.8            # 假设**具名**（不是匿名魔法数）
    assert f["if_cost_per_hour"] == pytest.approx(0.24, rel=1e-3)
    assert f["if_cost_per_mtok"] == pytest.approx(0.006667, rel=1e-3)
    assert f["if_missing"] == "no_price"                # 条件句因缺哪个输入而生
    assert f["if_provenance"] == "estimated"
    assert f["if_basis"] == "conditional"
    # 条件答案是**追加**的：包仍然压得进预算（常见情形）
    assert packet_bytes(p) <= PACKET_SIZE_BUDGET_BYTES


def test_missing_tokens_gets_a_conditional_unit_cost():
    """缺 token 数（价格已给）：每小时成本照给，单位成本**给条件答案**（no_tokens 不 dead-end）。"""
    p = build(_cost_report(price=0.8, tokens=None))
    v = _cost_verdict(p)
    assert v["status"] == "soft"                        # 每小时成本是真的 ⇒ 判词形状不变
    assert v["facts"]["per_hour"] == pytest.approx(0.24, rel=1e-3)
    f = v["facts"]
    assert f["assumed_tokens_per_step"] == 1_200_000    # 假设具名
    assert f["if_tokens_s"] == pytest.approx(1.2e7, rel=1e-6)
    assert f["if_cost_per_mtok"] == pytest.approx(5.556e-6, rel=1e-3)
    assert f["if_missing"] == "no_tokens"
    assert f["if_provenance"] == "estimated" and f["if_basis"] == "conditional"
    assert "if_cost_per_hour" not in f                  # 真有每小时成本时不重复报（要求 4）


def test_both_inputs_missing_names_both_assumptions():
    """既没价格也没 token 数：条件答案把**两个**假设都具名（功能说明里的那一句）。

    "if price = 0.8 元/kWh 且 1.2M token/step，那么 ¥X/Mtoken" —— 两个假设都不是测量，
    所以两个名字都必须写出来，且都要能在人话里读到。
    """
    p = build(_cost_report(price=None, tokens=None))
    v = _cost_verdict(p)
    assert v["status"] == "refused" and v["refusal"]["code"] == "no_price"
    f = v["facts"]
    assert f["assumed_price_per_kwh"] == 0.8
    assert f["assumed_tokens_per_step"] == 1_200_000
    assert f["if_missing"] == "no_price,no_tokens"
    assert f["if_cost_per_hour"] == pytest.approx(0.24, rel=1e-3)
    assert f["if_cost_per_mtok"] == pytest.approx(5.556e-6, rel=1e-3)
    text = render_onepage(p)
    assert "假设价格 0.8 CNY/kWh" in text and "假设 1200000 token/step" in text
    assert "no_price,no_tokens" in text


def test_giving_the_price_removes_the_conditional_block():
    """真输入给了 ⇒ 行为与今天完全一样：ok + 单位成本，**没有**条件块（要求 4）。"""
    p = build(_cost_report(price=0.8))
    v = _cost_verdict(p)
    assert v["status"] == "ok"
    assert v["facts"]["per_mtok"] == pytest.approx(0.0067, rel=1e-3)
    assert v["facts"]["tokens_s"] == pytest.approx(1e4, rel=1e-6)
    assert conditional_fact_keys(v["facts"]) == []
    assert "refusal" not in v and v["provenance"] == "derived"


def test_conditional_answer_can_never_be_read_as_a_measurement():
    """命名/出处纪律：条件值是 `estimated` + `conditional`，**永不**能被读成实测/精确。

    条件值不占 `value`（refused 判词没有 value），不挂 `meas__*`/`ks__*`，也不落进
    计数器命名空间 —— 名字承诺什么，出处就得是什么（PROVENANCE.md 的 R2/R3/R12）。
    """
    from kernoscope.provenance import COUNTER_NAMESPACES

    v = _cost_verdict(build(_cost_report(price=None)))
    f = v["facts"]
    keys = conditional_fact_keys(f)
    assert keys, "这个夹具必须产出条件答案，否则本测试没有分辨力"
    for k in keys:
        assert not k.startswith("meas__"), k
        assert not k.startswith("ks__"), k
        assert not k.startswith(COUNTER_NAMESPACES), k
    assert v["status"] == "refused" and v["status"] not in ("measured", "exact")
    assert v["provenance"] == CONDITIONAL_PROVENANCE
    assert v["provenance"] not in ("measured", "exact")
    assert "value" not in v, "条件值不许占 value（否则 diff/消费者会把它当结论）"
    assert f["if_provenance"] == CONDITIONAL_PROVENANCE
    # 对照：同一份报告给了价格之后，那些键一个都不该在包里出现
    ok_facts = _cost_verdict(build(_cost_report(price=0.8)))["facts"]
    assert not (set(keys) & set(ok_facts))


def test_onepage_shows_the_conditional_cost_sentence():
    """一屏契约：条件答案的每个 fact（含假设本身）都要能在人视图里看到。"""
    p = build(_cost_report(price=None))
    packet_bytes(p)
    text = render_onepage(p)
    v = _cost_verdict(p)
    for key, val in (v["facts"] or {}).items():
        assert str(val) in text, f"{key}={val} 没有出现在人视图（两个视图分叉）"
    assert "条件估计" in text
    assert "假设价格 0.8 CNY/kWh" in text
    assert "if_basis=conditional" in text and "if_provenance=estimated" in text
    assert "条件值不是实测" in text
    assert "no_price" in text                     # 原始拒绝码仍可见（缺什么一目了然）

    # 缺 token 数（soft 那一半）同样要印出来，否则 agent 只看到"钱给不出"
    soft = render_onepage(build(_cost_report(price=0.8, tokens=None)))
    assert "条件估计" in soft and "1200000 token/step" in soft


def test_conditional_answer_survives_the_packet_budget():
    """尺寸预算（要求 3）：条件答案挂在结论判词上 ⇒ 先裁次要 facts，再裁它。

    `cost_per_token` 是 `PRIMARY_FACT_IDS` 之一，所以预算裁到"次要 facts"那一档时，
    条件答案必须还在；真被裁到最后一级时，`cost.trimmed_ids` 要如实记下是哪条判词的
    facts 没了（R-106：结论的数字不许静默消失）。
    """
    rep = _rich_report()
    for m in (Metric.measured("meas__window_energy_j", 300.0, "J"),
              Metric.measured("meas__window_power_mean_w", 300.0, "W"),
              Metric.derived("ks__tokens_per_s", 10_000.0, "token/s"),
              Metric.derived("ks__steps_in_window", 10.0),
              Metric.unavailable("ks__cost_per_hour", "没有给价格", "CNY/h",
                                 code=RefusalCode.NO_PRICE),
              Metric.unavailable("ks__cost_per_mtok", "没有给价格", "CNY/Mtoken",
                                 code=RefusalCode.NO_PRICE)):
        rep.kernels[0].sections[0].add(m)
    p = build(rep)
    assert packet_bytes(p) <= PACKET_SIZE_BUDGET_BYTES
    assert "facts_secondary" in p["cost"]["trimmed"], \
        f"这个夹具必须真的走到裁剪（否则本测试没有分辨力）：{p['cost']['trimmed']}"
    v = _cost_verdict(p)
    facts = v.get("facts") or {}
    if facts:
        assert "if_cost_per_mtok" in facts, "条件答案不该先于次要 facts 被裁"
        assert facts["if_basis"] == "conditional"
    else:
        trimmed = p["cost"].get("trimmed_ids") or []
        assert "facts_primary:cost_per_token" in trimmed, \
            f"facts 没了却查不到留痕：{p['cost'].get('trimmed')} / {trimmed}"
