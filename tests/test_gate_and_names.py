"""`kernoscope gate`（C4）与 ncu 名字发现键/兼容导出（D4）的契约。

门禁最容易出的两种错，正是这两个测试族要挡住的：
  * 把 `refused` 当成通过（泄露一个不成立的结论）；
  * 用绝对阈值卡一个带误差带的量，或把 `n/a` 当成"没超阈值"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernoscope.cli import main
from kernoscope.gate_cmd import run_gate
from kernoscope.ncu_names import NCU_DISCOVERY_MAP, explain, render_name_map, resolve


def _packet(**over):
    base = {
        "schema_revision": "1.2",
        "verdicts": [
            {"id": "kernel_time", "status": "ok", "actionable": True,
             "premises": [{"id": "kernel_share_of_total", "observed": 99.0, "ok": True}]},
            {"id": "dram_bytes", "status": "soft", "actionable": True, "premises": []},
            {"id": "roofline", "status": "refused", "actionable": False,
             "refusal": {"code": "no_calibration"},
             "premises": [{"id": "kernel_share_of_total", "observed": 99.0, "ok": True}]},
        ],
        "refusals": {"no_calibration": {"action": "跑标定"}},
        "cost": {"packet_bytes": 3000},
    }
    base.update(over)
    return base


def _report():
    return {"kernels": [{"name": "k", "sections": [{"name": "Duration", "metrics": [
        {"name": "gpu__time_duration.sum", "value": 12.5, "provenance": "exact"},
        {"name": "ks__cost_per_mtok", "value": 3.5, "provenance": "derived"},
        {"name": "est__dram_bytes_total", "value": None, "provenance": "unavailable",
         "refusal": {"code": "no_nvml_samples"}},
    ]}]}]}


def _multi_report():
    """两个真 kernel + 一个 `(device)` 伪 kernel（V-2 的复现形状）。

    真 kernel 的 `ks__launch_count` 分别是 5 / 20，`ks__theoretical_occupancy_pct` 是
    33.3 / 100.0；`meas__nvml_mem_busy_mean_pct` 只挂在设备级伪 kernel 上。
    """
    return {"kernels": [
        {"name": "synthetic_gemm", "sections": [{"name": "Launch", "metrics": [
            {"name": "ks__launch_count", "value": 5, "provenance": "exact"},
            {"name": "ks__theoretical_occupancy_pct", "value": 33.3, "provenance": "derived"},
        ]}]},
        {"name": "synthetic_elementwise", "sections": [{"name": "Launch", "metrics": [
            {"name": "ks__launch_count", "value": 20, "provenance": "exact"},
            {"name": "ks__theoretical_occupancy_pct", "value": 100.0, "provenance": "derived"},
        ]}]},
        {"name": "(device)", "sections": [{"name": "DeviceStatistics", "metrics": [
            {"name": "meas__nvml_mem_busy_mean_pct", "value": 96.8, "provenance": "measured"},
        ]}]},
    ]}


# --------------------------------------------------------------------------
# gate 规则
# --------------------------------------------------------------------------
def test_require_ok_passes_and_soft_is_not_ok():
    assert run_gate(_packet(), require=["kernel_time"]).passed
    res = run_gate(_packet(), require=["dram_bytes"])
    assert not res.passed and "soft" in res.failures[0].reason
    assert run_gate(_packet(), require=["dram_bytes=soft"]).passed


def test_require_refused_state_is_never_satisfied_by_absence():
    res = run_gate(_packet(), require=["occupancy"])
    assert not res.passed and "没有这条判词" in res.failures[0].reason


def test_actionable_check_catches_refused_and_failed_premises():
    p = _packet()
    p["verdicts"].append({"id": "ledger_verdict", "status": "ok",
                          "premises": [{"id": "x", "observed": 1, "ok": False}]})
    assert run_gate(p, require_actionable=["kernel_time"]).passed
    assert not run_gate(p, require_actionable=["roofline"]).passed
    res = run_gate(p, require_actionable=["ledger_verdict"])
    assert not res.passed and res.failures[0].observed["failed_premises"] == ["x"]


def test_v10_packet_without_actionable_field_falls_back_to_premises():
    """v1.0 的包没有 `actionable`：回退规则必须与文档一致（不是 refused 且前提全成立）。"""
    p = {"verdicts": [{"id": "kernel_time", "status": "ok",
                       "premises": [{"id": "a", "ok": True}]}]}
    assert run_gate(p, require_actionable=["kernel_time"]).passed
    p2 = {"verdicts": [{"id": "kernel_time", "status": "ok",
                        "premises": [{"id": "a", "ok": False}]}]}
    assert not run_gate(p2, require_actionable=["kernel_time"]).passed


def test_forbid_code_reads_verdicts_refusals_index_and_environment():
    assert run_gate(_packet(), forbid_codes=["counters_denied"]).passed
    assert not run_gate(_packet(), forbid_codes=["no_calibration"]).passed
    env = _packet(environment_refusals=[{"code": "nsys_missing", "message": "x"}])
    assert not run_gate(env, forbid_codes=["nsys_missing"]).passed


# W-3/R-98 的载体名 → 只往包里塞这一处载体，`refusals` 表**故意留空**。
# 留空是关键：旧实现靠 refusals 表兜底，所以凡是表里没有的载体它都会放绿；
# 这些用例在旧实现下全部 rc=0（会红）。`unmeasured_more` 今天只有计数、不带 code，
# 列在这里是守着"它带上 code 时也不许漏"（此刻它的 code 由 `unmeasured[].code` 承载）。
_ONLY_CARRIERS = {
    "unmeasured": {"unmeasured": [{"code": "counters_denied", "count": 4}]},
    "refusals_omitted": {"refusals_omitted": {"count": 43,
                                              "codes": ["counters_denied", "no_nvtx"]}},
    "environment_refusals_more": {"environment_refusals_more": {
        "count": 9, "codes": ["counters_denied"]}},
    "next": {"next": [{"for": "roofline", "code": "counters_denied"}]},
    "env_blocked_by": {"env": {"counter_state": "denied", "blocked_by": ["counters_denied"]}},
    "nested": {"cost": {"trimmed": [{"codes": [{"code": "counters_denied"}]}]}},
    # 今天 `unmeasured_more` 只有计数、不带 code；这条守着"它哪天带上 code 也不许漏"
    # （此刻它的 code 由 `unmeasured[].code` 承载）。
    "unmeasured_more": {"unmeasured_more": {"groups": 2, "metrics": 9},
                        "unmeasured": [{"code": "counters_denied", "count": 4}]},
}


def test_refusal_codes_covers_every_carrier_a_packet_can_use():
    """R-98/W-3：`--forbid-code` 认的是"包里任何位置出现过的 code"，不是今天的三处。

    旧实现只读 `verdicts[].refusal.code` / `refusals{}` 键 / `environment_refusals[].code`，
    于是 code 被裁进 `refusals_omitted` / `unmeasured` / `environment_refusals_more` /
    `next` / `env.blocked_by` 时拼对的 code 也放绿 —— 而"code 被挪进 *_omitted/*_more"
    正是受限机（十几个 code）的常态。所以这里逐个载体验证，**并留一个 refusals 表为空的
    陷阱**：一旦有人把实现改回"枚举三处 / 靠 refusals 兜底"，这些用例立刻变红。
    """
    for carrier, extra in _ONLY_CARRIERS.items():
        p = _packet(**extra)
        p["refusals"] = {}                      # 陷阱：表里没有这个 code
        res = run_gate(p, forbid_codes=["counters_denied"])
        assert not res.passed, f"code 只在 {carrier} 里也必须命中（旧实现会放绿）"
        assert res.failures[0].check == "forbid-code counters_denied"
        assert not res.usage_errors, "命中不是用法错误，rc 必须留在 1"
    # 全树扫描不许把非 code 的字符串/别的 code 当成命中
    clean = _packet(refusals={}, cost={"trimmed": ["next_pointers"], "counter_state": "denied"})
    assert run_gate(clean, forbid_codes=["counters_denied"]).passed


def test_refusal_codes_scan_does_not_widen_the_closed_set_rule():
    """闭集校验一分没松：未知 code 仍是用法错误（rc=2），与"命中/没命中"无关。"""
    p = _packet(unmeasured=[{"code": "counters_denied", "count": 4}])
    p["refusals"] = {}
    res = run_gate(p, forbid_codes=["counters_denyed"])
    assert res.usage_errors and not res.passed


def test_metric_thresholds_are_compared_and_na_never_passes():
    ok = run_gate(_packet(), report=_report(), max_metric=["gpu__time_duration.sum=20"])
    assert ok.passed
    bad = run_gate(_packet(), report=_report(), max_metric=["gpu__time_duration.sum=10"])
    assert not bad.passed and bad.failures[0].observed["value"] == 12.5
    assert run_gate(_packet(), report=_report(), min_metric=["ks__cost_per_mtok=3"]).passed
    # n/a 不算通过（"没有数"不是"没超阈值"）
    na = run_gate(_packet(), report=_report(), max_metric=["est__dram_bytes_total=1e12"])
    assert not na.passed and "n/a" in na.failures[0].reason


def test_missing_metric_and_bad_rule_are_failures_not_silent_passes():
    assert not run_gate(_packet(), report=_report(), max_metric=["nope=1"]).passed
    res = run_gate(_packet(), max_metric=["no-equals-sign"])
    assert not res.passed and "NAME=VALUE" in res.failures[0].reason
    assert not run_gate(_packet(), require=["kernel_time=bogus"]).passed


def test_metric_rule_syntax_error_is_a_usage_error_not_a_missing_rule():
    """R-103/W-7：`--max-metric X=abc` / `--max-metric foo` 是**用法错误**（rc=2）。

    旧实现只记一条 Failure（CLI 退 1），而且因为解析失败的规则没进 `checked`，还会顺手
    追加一句假的"一条规则都没给" —— 把"规则写错"伪装成"没给规则"。
    """
    for spec in ("X=abc", "foo"):
        res = run_gate(_packet(), max_metric=[spec])
        assert res.usage_errors, f"{spec!r} 必须是用法错误（CLI 才会退 2）"
        assert not res.passed
        assert not any("一条规则都没给" in f.reason for f in res.failures), \
            "规则写错不等于没给规则：不许吐那句假失败"
    # 真的一条都没给时，那句仍然要在（这条边界不能被顺手删掉）
    empty = run_gate(_packet())
    assert not empty.passed and "一条规则都没给" in empty.failures[0].reason


def test_actionable_string_is_a_usage_error_and_never_passes():
    """R-103/W-7：`actionable` 只接受真布尔。

    `bool("false") is True` —— 旧实现把字符串强转成 True，`--require-actionable` 于是
    rc=0 放绿。字符串是"规则没法判"，要退 rc=2，绝不能退化成"判词通过"。
    """
    for bad in ("false", "true", 0, None):
        p = _packet(verdicts=[{"id": "kernel_time", "status": "ok", "actionable": bad}])
        res = run_gate(p, require_actionable=["kernel_time"])
        assert res.usage_errors and not res.passed, f"actionable={bad!r} 必须 rc=2，不是放绿"
    # 真布尔照旧：true 过、false 红
    assert run_gate(_packet(), require_actionable=["kernel_time"]).passed
    p = _packet(verdicts=[{"id": "kernel_time", "status": "ok", "actionable": False}])
    res = run_gate(p, require_actionable=["kernel_time"])
    assert not res.passed and not res.usage_errors, "真 false 是判词不合格（rc=1），不是用法错"


def test_max_metric_evaluates_every_kernel_and_names_the_violator():
    """V-2：第二个 kernel 超限也必须不通过，且输出点名是哪个 kernel（旧实现只看第一个）。"""
    rep = _multi_report()
    res = run_gate(_packet(), report=rep, max_metric=["ks__launch_count=10"])
    assert not res.passed, "第二个 kernel 是 20，超了 10，绝不能放绿"
    assert len(res.failures) == 1
    assert res.failures[0].observed["kernel"] == "synthetic_elementwise"
    assert "synthetic_elementwise" in res.failures[0].reason
    assert "synthetic_gemm" not in res.failures[0].reason      # 只点名违反的那个
    # 两个 kernel 都满足时才通过
    assert run_gate(_packet(), report=rep, max_metric=["ks__launch_count=25"]).passed


def test_min_metric_evaluates_every_kernel_and_names_the_violator():
    """V-2：min 同理 —— 不能拿第一个 kernel 的 33.3 去判第二个 kernel 的 100.0。"""
    rep = _multi_report()
    res = run_gate(_packet(), report=rep, min_metric=["ks__theoretical_occupancy_pct=90"])
    assert not res.passed
    assert res.failures[0].observed["kernel"] == "synthetic_gemm"
    assert "synthetic_gemm" in res.failures[0].reason
    assert run_gate(_packet(), report=rep,
                    min_metric=["ks__theoretical_occupancy_pct=30"]).passed


def test_metric_absent_on_one_kernel_is_a_failure_not_a_free_pass():
    """真 kernel 缺这个指标 = 空白，不是豁免：没有数不能当作通过。"""
    rep = _multi_report()
    rep["kernels"][1]["sections"][0]["metrics"] = [
        m for m in rep["kernels"][1]["sections"][0]["metrics"]
        if m["name"] != "ks__launch_count"]
    res = run_gate(_packet(), report=rep, max_metric=["ks__launch_count=100"])
    assert not res.passed
    assert res.failures[0].observed["kernel"] == "synthetic_elementwise"


def test_metric_absent_everywhere_is_a_failure():
    assert not run_gate(_packet(), report=_multi_report(), max_metric=["nope=1"]).passed
    assert not run_gate(_packet(), report=_multi_report(), min_metric=["nope=1"]).passed


def test_device_only_metric_is_judged_on_the_device_pseudo_kernel():
    """只有 `(device)` 带的指标是设备级判据：真 kernel 缺项不算违规，但设备自己必须达标。"""
    rep = _multi_report()
    assert run_gate(_packet(), report=rep,
                    max_metric=["meas__nvml_mem_busy_mean_pct=99"]).passed
    bad = run_gate(_packet(), report=rep, max_metric=["meas__nvml_mem_busy_mean_pct=50"])
    assert not bad.passed and bad.failures[0].observed["kernel"] == "(device)"


def test_device_metric_also_compared_when_it_carries_a_kernel_scoped_metric():
    """`(device)` 带了 kernel 级指标时也一起比（缺项豁免，不是数值豁免）。"""
    rep = _multi_report()
    rep["kernels"][2]["sections"][0]["metrics"].append(
        {"name": "ks__launch_count", "value": 99, "provenance": "derived"})
    res = run_gate(_packet(), report=rep, max_metric=["ks__launch_count=25"])
    assert not res.passed
    assert [f.observed["kernel"] for f in res.failures] == ["(device)"]


def test_no_rules_at_all_is_a_failure():
    """"一条规则都没给"绝不能变成"通过" —— 那会让 CI 永远绿灯。"""
    res = run_gate(_packet())
    assert not res.passed and "一条规则都没给" in res.failures[0].reason


# --------------------------------------------------------------------------
# gate CLI
# --------------------------------------------------------------------------
def _run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def _pipeline(tmp_path, capsys, extra=()):
    """跑一次合成采集，产出包 + 报告，返回 (packet, report) 路径。"""
    base = tmp_path / "rep"
    rc, out, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                        "--screen", "off", "--quiet", "-o", str(base),
                        "--packet", str(base) + ".packet.json", *extra)
    assert rc == 0, err
    return str(base) + ".packet.json", str(base) + ".ksreport.json"


def test_gate_cli_exit_codes_and_json(tmp_path, capsys):
    packet, report = _pipeline(tmp_path, capsys)
    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--require", "kernel_time")
    assert rc == 0 and "门禁通过" in out

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--require", "roofline")
    assert rc == 1 and "门禁不通过" in out

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--require", "kernel_time", "--json")
    assert rc == 0 and json.loads(out)["passed"] is True

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--report", report,
                      "--max-metric", "gpu__time_duration.sum=1e9", "--json")
    assert rc == 0 and json.loads(out)["checked"]

    rc, _, err = _run(capsys, "gate", "--packet", str(tmp_path / "nope.json"),
                      "--require", "kernel_time")
    assert rc == 2 and "读不了" in err


def test_gate_cli_refuses_to_be_green_on_an_empty_rule_set(tmp_path, capsys):
    packet, _ = _pipeline(tmp_path, capsys)
    rc, out, _ = _run(capsys, "gate", "--packet", packet)
    assert rc == 1 and "一条规则都没给" in out


def test_gate_cli_max_metric_flags_the_second_kernel(tmp_path, capsys):
    """V-2 的 CLI 面：合成报告第二个 kernel 超限 → rc=1 且输出点名它。"""
    packet, _ = _pipeline(tmp_path, capsys)
    report = tmp_path / "multi.ksreport.json"
    report.write_text(json.dumps(_multi_report()), encoding="utf-8")

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--report", str(report),
                      "--max-metric", "ks__launch_count=10")
    assert rc == 1 and "synthetic_elementwise" in out

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--report", str(report),
                      "--min-metric", "ks__theoretical_occupancy_pct=90")
    assert rc == 1 and "synthetic_gemm" in out

    rc, out, _ = _run(capsys, "gate", "--packet", packet, "--report", str(report),
                      "--max-metric", "ks__launch_count=25")
    assert rc == 0 and "门禁通过" in out


def test_gate_cli_forbid_code_rejects_a_typo_as_a_usage_error(tmp_path, capsys):
    """V-3：拼错的 code 是用法错误（rc=2），不是"没命中所以通过"。"""
    packet, _ = _pipeline(tmp_path, capsys)
    rc, out, err = _run(capsys, "gate", "--packet", packet,
                        "--forbid-code", "counters_denyed")
    assert rc == 2, "未知 code 绝不能静默放绿"
    assert "门禁通过" not in out
    assert "闭集" in err and "counters_denied" in err           # 打印了可用 code 的引导


def test_gate_cli_valid_forbid_code_keeps_its_meaning(tmp_path, capsys):
    """合法 code 语义不变：命中 → rc=1；没命中 → rc=0。"""
    packet, _ = _pipeline(tmp_path, capsys)
    rc, out, _ = _run(capsys, "gate", "--packet", packet,
                      "--forbid-code", "counters_denied")
    assert rc == 1 and "出现了被禁止的拒绝码" in out

    rc, out, _ = _run(capsys, "gate", "--packet", packet,
                      "--forbid-code", "protocol_mismatch")
    assert rc == 0 and "门禁通过" in out


def test_gate_cli_rejects_an_unknown_require_status_as_a_usage_error(tmp_path, capsys):
    """`--require` 的 status 也是闭集：写错同样是用法错误，不是"弱于要求"。"""
    packet, _ = _pipeline(tmp_path, capsys)
    rc, out, err = _run(capsys, "gate", "--packet", packet,
                        "--require", "kernel_time=bogus")
    assert rc == 2 and "门禁通过" not in out
    assert "ok/soft/refused" in err


def test_gate_cli_forbid_code_sees_a_real_packet_that_omitted_refusal_entries(tmp_path, capsys):
    """W-3/R-98 的"真包"面：`build()` 真触发 `refusals_omitted` 时，**每个**被省掉的 code
    都必须还能被 `--forbid-code` 命中（rc=1）。

    受限机（45 个 code）正是这条路径的常态：`refusals` 表只留 8 条，其余列在
    `refusals_omitted.codes` 里。旧实现只读 `refusals` 的键 ⇒ 这些 code 全部放绿。
    """
    from kernoscope.packet import build
    from kernoscope.provenance import Metric, Section
    from kernoscope.refusals import DEFAULT_REMEDIES, RefusalCode
    from kernoscope.report import DeviceInfo, KernelReport, Report

    codes = [c for c in RefusalCode if c in DEFAULT_REMEDIES]
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    k = KernelReport("big_gemm", launch_count=1)
    sec = Section("受限环境下的未测项")
    for i, c in enumerate(codes):
        sec.add(Metric.unavailable(f"est__blocked_{i}", "本机拿不到这个数据源", code=c))
    k.sections = [sec]
    rep.kernels = [k]
    p = build(rep)
    omitted = p.get("refusals_omitted") or {}
    assert omitted.get("codes"), "这个 fixture 必须真触发条目省略（否则用例就没在测 W-3）"
    f = tmp_path / "locked-down.packet.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    for code in omitted["codes"]:
        rc, out, err = _run(capsys, "gate", "--packet", str(f), "--forbid-code", code)
        assert rc == 1, f"code {code!r} 被裁进 refusals_omitted，仍必须命中（旧实现 rc=0）"
        assert "出现了被禁止的拒绝码" in out


def test_gate_cli_forbid_code_sees_a_code_that_only_lives_in_a_trimmed_field(tmp_path, capsys):
    """W-3/R-98 的 CLI 面：真包 + 抑制预算保留的载体 → `--forbid-code` 必须 rc=1。

    `refusals` 表被清空、code 只留在 `unmeasured[].code` / `refusals_omitted.codes` /
    `environment_refusals_more.codes` 里 —— 受限机上"code 被挪进 *_omitted/*_more"是常态。
    旧实现在这三种形状下都是 rc=0 放绿（这条用例就是给那个回归准备的）。
    """
    packet, _ = _pipeline(tmp_path, capsys)
    base = json.loads(Path(packet).read_text(encoding="utf-8"))
    shapes = {
        "unmeasured": {"refusals": {}, "unmeasured": [{"code": "counters_denied",
                                                       "count": 4}]},
        "refusals_omitted": {"refusals": {}, "unmeasured": [],
                             "refusals_omitted": {"count": 43,
                                                  "codes": ["counters_denied"]}},
        "environment_refusals_more": {"refusals": {}, "unmeasured": [],
                                      "environment_refusals_more": {
                                          "count": 9, "codes": ["counters_denied"]}},
    }
    for name, over in shapes.items():
        p = dict(base, **over)
        f = tmp_path / f"{name}.packet.json"
        f.write_text(json.dumps(p), encoding="utf-8")
        rc, out, err = _run(capsys, "gate", "--packet", str(f),
                            "--forbid-code", "counters_denied")
        assert rc == 1, f"{name}：code 就在包里，绝不能放绿（旧实现 rc=0）；err={err}"
        assert "出现了被禁止的拒绝码" in out
    # 闭集校验没被放宽
    rc, _, err = _run(capsys, "gate", "--packet", str(tmp_path / "unmeasured.packet.json"),
                      "--forbid-code", "counters_denyed")
    assert rc == 2 and "闭集" in err


def test_gate_cli_metric_rule_syntax_error_is_rc2_without_the_bogus_failure(tmp_path, capsys):
    """R-103/W-7：指标规则语法/数值错 = 用法错误（rc=2），且不许吐假失败"一条规则都没给"。"""
    packet, _ = _pipeline(tmp_path, capsys)
    for spec in ("X=abc", "foo"):
        rc, out, err = _run(capsys, "gate", "--packet", packet, "--max-metric", spec)
        assert rc == 2, f"--max-metric {spec} 是用法错误（旧实现 rc=1）"
        assert "一条规则都没给" not in out and "一条规则都没给" not in err
        assert "error:" in err
    # 真的一条都没给时仍然 rc=1 + 那句提示（这条边界不能被顺手删掉）
    rc, out, _ = _run(capsys, "gate", "--packet", packet)
    assert rc == 1 and "一条规则都没给" in out


def test_gate_cli_json_usage_errors_are_still_emitted(tmp_path, capsys):
    """R-103/W-7：用法错误 + `--json` 必须有机器可读输出（`usage_errors` 才可达）。

    旧实现先 return 2（只往 stderr 打印），stdout 全空 —— agent 无从知道是用法错。
    rc 仍然是 2（CI 照样红）。
    """
    packet, _ = _pipeline(tmp_path, capsys)
    cases = [["--forbid-code", "counters_denyed"],
             ["--require", "kernel_time=bogus"],
             ["--max-metric", "foo"]]
    for argv in cases:
        rc, out, _ = _run(capsys, "gate", "--packet", packet, *argv, "--json")
        assert rc == 2, f"{argv} 用法错误必须 rc=2"
        doc = json.loads(out)                       # stdout 不许再是空的
        assert doc["usage_errors"], f"{argv} 的 usage_errors 必须可见"


def test_gate_cli_actionable_string_is_a_usage_error(tmp_path, capsys):
    """R-103/W-7 的 CLI 面：`actionable:"false"` 不得被 `bool()` 强转成通过。"""
    packet, _ = _pipeline(tmp_path, capsys)
    p = json.loads(Path(packet).read_text(encoding="utf-8"))
    vid = p["verdicts"][0]["id"]
    p["verdicts"][0]["actionable"] = "false"
    f = tmp_path / "str-bool.packet.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    rc, out, err = _run(capsys, "gate", "--packet", str(f), "--require-actionable", vid)
    assert rc == 2, '字符串 "false" 不是 false，必须 rc=2（旧实现 rc=0 放绿）'
    assert "门禁通过" not in out and "布尔" in err


# --------------------------------------------------------------------------
# D4：名字发现键 + 兼容导出
# --------------------------------------------------------------------------
def test_discovery_map_is_internally_consistent():
    from kernoscope.ncu_names import _ALIAS_CONFLICTS

    # 别名索引不许有"同一个规范化键指向两种不同的东西"（那会让 --metrics 得到随机的答案）
    assert _ALIAS_CONFLICTS == [], f"映射表规范化后有冲突：{_ALIAS_CONFLICTS}"
    for d in NCU_DISCOVERY_MAP:
        assert d.ncu_name and d.caliber
        if d.kind == "substitute":
            assert d.ours and d.ours != d.ncu_name
            assert d.ours.startswith(("est__", "meas__", "model__", "ks__")), \
                f"{d.ncu_name} 的替代名必须在自有命名空间里（不能挂在 ncu 名下）"
        if d.kind == "same_source":
            assert d.ours == d.ncu_name
        if d.kind == "none":
            assert d.ours == ""


def test_resolve_handles_variants_but_never_guesses():
    assert resolve("dram__bytes.sum").ours == "est__dram_bytes_total"
    assert resolve("dram__bytes").ours == "est__dram_bytes_total"          # 后缀变体
    assert resolve("gpu__time_duration.sum").kind == "same_source"
    assert resolve("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed").ours == ""
    assert resolve("totally_made_up__metric") is None                      # 查不到就查不到


def test_explain_says_what_we_give_or_why_we_cannot():
    text = explain("dram__bytes.sum")
    assert "est__dram_bytes_total" in text and "不会挂在 dram__bytes.sum 下面" in text
    none = explain("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    assert "给不出" in none


def test_name_map_export_has_both_formats(tmp_path, capsys):
    js = json.loads(render_name_map("json"))
    assert js["schema"] == "kernoscope-name-map-v1"
    assert any(r["ncu_name"] == "dram__bytes.sum" for r in js["map"])
    csv_text = render_name_map("csv")
    assert csv_text.splitlines()[0] == "ncu_name,ours,kind,caliber"

    out_json = tmp_path / "names.json"
    rc, _, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--screen",
                    "off", "--quiet", "--export-name-map", str(out_json))
    assert rc == 0 and json.loads(out_json.read_text(encoding="utf-8"))["map"]
    out_csv = tmp_path / "names.csv"
    rc, _, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic", "--screen",
                    "off", "--quiet", "--export-name-map", str(out_csv))
    assert rc == 0 and out_csv.read_text(encoding="utf-8").startswith("ncu_name,ours")


def test_metrics_flag_accepts_a_counter_name_as_a_discovery_key(capsys, tmp_path):
    """`--metrics dram__bytes.sum` 必须拿到替代口径的行，而不是一句"不存在"。"""
    base = tmp_path / "rep"
    rc, out, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                        "--screen", "off", "--metrics", "dram__bytes.sum",
                        "-o", str(base))
    assert rc == 0
    assert "替代口径 est__dram_bytes_total" in err
    assert "est__dram_bytes_total" in out
    # 原来的规则一秒没松：数值绝不挂在 ncu 计数器名下
    assert "dram__bytes.sum " not in out.replace("--metrics dram__bytes.sum", "")


def test_strict_names_the_substitute_instead_of_saying_not_found(capsys, tmp_path):
    rc, _, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                      "--screen", "off", "--strict", "--metrics", "dram__bytes.sum")
    assert rc == 2
    assert "est__dram_bytes_total" in err and "not found" not in err


def test_strict_refuses_a_name_we_genuinely_cannot_produce(capsys):
    rc, _, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--set", "basic",
                      "--screen", "off", "--strict", "--metrics",
                      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    assert rc == 2 and "给不出" in err
