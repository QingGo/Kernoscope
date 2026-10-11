"""W3：结构化拒绝 —— 「不存在散文式拒绝」这条要求由本文件的测试执行。

为什么值得单独一个文件：拒绝是这台机器上**最常见**的输出（计数器全线拿不到）。
如果拒绝只有中文散文，agent 就无法判断「该不该照着这条改代码」，人会把它当成
「工具坏了」。所以拒绝必须带：闭集 code、观测值、门槛、可执行的补救办法与代价。
"""

from __future__ import annotations

import json

import pytest

from kernoscope.provenance import Metric, Provenance, ProvenanceError, Section
from kernoscope.refusals import (DEFAULT_REMEDIES, Refusal, RefusalCode, Remedy,
                                 combined_refusal, legacy_refusal, missing_remedies)


def test_every_code_has_a_remedy():
    """闭集里不允许出现「没有补救办法」的拒绝类型。"""
    assert missing_remedies() == []
    for code, rem in DEFAULT_REMEDIES.items():
        assert rem.action, f"{code.value} 的 remedy 没有 action"
        assert rem.cost_kind in ("wall_seconds", "manual", "unavailable")


def test_codes_are_unique_and_machine_readable():
    values = [c.value for c in RefusalCode]
    assert len(values) == len(set(values))
    for v in values:
        assert v == v.lower() and " " not in v, f"{v} 不是可编程的标识符"
        assert v.isascii()


def test_unavailable_requires_code():
    """不给 code 的 n/a 行在类型层面就构造不出来。"""
    with pytest.raises(TypeError):
        Metric.unavailable("est__x", "原因")           # type: ignore[call-arg]
    m = Metric.unavailable("est__x", "原因", code=RefusalCode.NO_NVML_SAMPLES)
    assert m.refusal is not None and m.refusal.code is RefusalCode.NO_NVML_SAMPLES


def test_legacy_dict_metric_becomes_legacy_refusal():
    """旧报告（没有 refusal 字段）回读后必须有 code=legacy_report，且不猜原因。"""
    from kernoscope.report import Report

    d = {
        "schema_version": 1, "tool_version": "0.1.0",
        "kernels": [{"name": "k", "sections": [{"name": "S", "metrics": [
            {"name": "sm__throughput.avg.pct_of_peak_sustained_elapsed", "value": None,
             "provenance": "unavailable", "reason": "老的散文原因"}]}]}],
    }
    rep = Report.from_dict(d)
    m = rep.kernels[0].sections[0].metrics[0]
    assert m.refusal is not None
    assert m.refusal.code is RefusalCode.LEGACY_REPORT
    assert m.reason == "老的散文原因"
    assert m.provenance is Provenance.UNAVAILABLE


def test_reason_and_refusal_message_cannot_diverge():
    """同一事实不能有两个说法（反门槛第 2 条的机制保障）。"""
    ref = Refusal(RefusalCode.NO_GPU, "没有 GPU")
    with pytest.raises(ProvenanceError, match="disagree"):
        Metric("est__x", None, Provenance.UNAVAILABLE, reason="另一句话", refusal=ref)


def test_refusal_roundtrip_json():
    ref = Refusal(RefusalCode.COVERAGE_TOO_LOW, "覆盖率 71.4% < 80%",
                  observed={"coverage_pct": 71.4}, threshold={"min_pct": 80.0},
                  remedy=Remedy("跑久一点", command="kernoscope --ks-window auto -- x.py",
                                cost_s=20.0, then_answers=("dram_bytes",)))
    back = Refusal.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert back == ref
    assert back.remedy is not None and back.remedy.command.endswith("-- x.py")


def test_unknown_code_from_future_version_does_not_crash():
    """前向兼容：读到一个本版本不认识的 code 时，不抛异常、不假装认识。"""
    d = Refusal(RefusalCode.NO_GPU, "x").to_dict()
    d["code"] = "some_future_code"
    back = Refusal.from_dict(d)
    assert back.code is RefusalCode.INTERNAL_UNKNOWN


def test_combined_refusal_keeps_all_premises():
    a = Refusal(RefusalCode.NO_NVML_SAMPLES, "没有采样点")
    b = Refusal(RefusalCode.NO_CALIBRATION, "缺峰值")
    c = combined_refusal([a, b])
    assert c.code is RefusalCode.NO_NVML_SAMPLES
    assert "没有采样点" in c.message and "缺峰值" in c.message
    assert [x.code for x in c.causes] == [RefusalCode.NO_CALIBRATION]
    assert "causes" in c.to_dict()


def test_combined_refusal_single_passthrough():
    a = Refusal(RefusalCode.NO_GPU, "x")
    assert combined_refusal([a]) is a
    assert combined_refusal([]).code is RefusalCode.INTERNAL_UNKNOWN


@pytest.mark.parametrize("name,reason", [("est__x", ""), ])
def test_empty_message_rejected(name, reason):
    with pytest.raises(ValueError):
        Refusal(RefusalCode.NO_GPU, reason)


def test_every_unavailable_metric_in_a_real_report_has_a_code():
    """端到端：合成报告里所有 n/a 行都必须带闭集里的 code（不存在散文式拒绝）。"""
    from kernoscope.cli import main

    import contextlib
    import io
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rep"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = main(["--ks-synthetic", "--set", "full", "--ks-no-nvml", "-o", str(out)])
        assert rc == 0
        rep = json.loads((Path(td) / "rep.ksreport.json").read_text())
        seen = 0
        for k in rep["kernels"]:
            for s in k["sections"]:
                for m in s["metrics"]:
                    if m["provenance"] != "unavailable":
                        continue
                    seen += 1
                    assert m.get("refusal"), f"{m['name']} 是 n/a 但没有结构化拒绝"
                    code = m["refusal"]["code"]
                    assert code in {c.value for c in RefusalCode}, code
                    assert m["refusal"]["message"] == m["reason"]
                    assert m["refusal"]["remedy"]["action"], m["name"]
        assert seen > 10, "full set 下 n/a 行应当很多，测试没有真正覆盖"


def test_legacy_refusal_helper_marks_unknown_origin():
    r = legacy_refusal("")
    assert r.code is RefusalCode.LEGACY_REPORT
    assert r.message


def test_section_helper_covers_only_na_rows():
    from kernoscope.sections import counter_section_code, counter_section_reason

    assert counter_section_reason("ComputeWorkloadAnalysis")
    assert counter_section_code("ComputeWorkloadAnalysis") is RefusalCode.NO_COUNTERS
    assert counter_section_code("SourceCounters") is RefusalCode.SAMPLING_UNAVAILABLE
    assert counter_section_code("NotASection") is RefusalCode.INTERNAL_UNKNOWN
    assert counter_section_reason("NotASection") == ""


def test_section_histogram_still_works():
    s = Section("S")
    s.add(Metric.unavailable("ks__x", "r", code=RefusalCode.NOT_APPLICABLE,
                             severity="degraded"))
    assert s.provenance is Provenance.UNAVAILABLE
    assert s.n_unavailable == 1


def test_conditional_answer_labels_match_the_provenance_vocabulary():
    """反事实视图（条件答案）的命名纪律：「拒绝」的第二半只能是条件值，不许像测量。

    为什么这条放在拒绝的测试文件里：条件答案是「不给数」这种输出的第二半
    （为什么 + 怎么办 + **如果……那么……**），不是新判词、也不是新 code。行为级测试
    （缺价格 → 包里出现条件成本）在 `tests/test_packet.py`；这里守的是**词表** ——
    `refusals.py` 不能 import `provenance.py`（后者反过来 import 前者），所以
    "条件值的出处就是 `estimated`" 这层一致性只能由测试钉住。
    """
    from kernoscope.provenance import COUNTER_NAMESPACES
    from kernoscope.refusals import (CONDITIONAL_ASSUMED_PREFIX, CONDITIONAL_BASIS,
                                     CONDITIONAL_PROVENANCE, CONDITIONAL_VALUE_PREFIX,
                                     conditional_fact_keys)

    assert CONDITIONAL_PROVENANCE == Provenance.ESTIMATED.value
    assert CONDITIONAL_PROVENANCE not in (Provenance.MEASURED.value, Provenance.EXACT.value)
    assert CONDITIONAL_BASIS == "conditional"
    assert CONDITIONAL_ASSUMED_PREFIX == "assumed_" and CONDITIONAL_VALUE_PREFIX == "if_"

    facts = {"per_mtok": 0.0067, "assumed_price_per_kwh": 0.8, "if_cost_per_mtok": 0.0067,
             "if_basis": CONDITIONAL_BASIS, "if_provenance": CONDITIONAL_PROVENANCE}
    keys = conditional_fact_keys(facts)
    assert keys == ["assumed_price_per_kwh", "if_basis", "if_cost_per_mtok", "if_provenance"]
    for k in keys:
        assert not k.startswith("meas__") and not k.startswith("ks__"), k
        assert not k.startswith(COUNTER_NAMESPACES), k
    assert conditional_fact_keys({"per_mtok": 0.0067}) == []
