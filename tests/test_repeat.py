"""M5 / N1：进程级重复执行（`--repeat N`）的契约测试。

为什么值得单独一个文件：这一层的**唯一价值**是"给出跨运行分布，让 diff 分得清噪声与真变化"，
所以它最危险的失败模式和别的模块相反 —— 不是"算错一个数"，而是：

  * 把失败的那次**悄悄丢掉**（于是 N 看起来是成功的次数，用户以为跑够了）；
  * 把同名但**不同单位**的读数混进同一个分布（sd 里混着两种口径的差，MDD 变成垃圾）；
  * 只跑了 1 次就给出 sd（写成 0 会让下游以为"极其稳定"）；
  * 把 wall 时间当 GPU 秒（成本算多），或干脆编一个价格（本项目不编价格）。

这些都在这里逐条钉住。测试**完全离线**：`runner` 是注入的 callable，不 shell 到真 CLI。

对应的实现见 `src/kernoscope/repeat.py`；MDD 的公式与它的坑写在那个模块的 docstring 里。
"""

from __future__ import annotations

import json
import math

import pytest

from kernoscope import repeat
from kernoscope.refusals import RefusalCode
from kernoscope.repeat import (
    MDD_FLOOR_PCT,
    MDD_SD_FACTOR,
    MIN_RUNS_PER_METRIC,
    RepeatResult,
    extract_metrics,
    repeat_runs,
)


# --------------------------------------------------------------------------
# 夹具：最小报告 + 脚本化 runner
# --------------------------------------------------------------------------
def _report(metrics: dict[str, tuple[float, str]], gpu_s: float | None = None) -> dict:
    """形如真实 ksreport 的最小报告：`kernels[].sections[].metrics[]`。"""
    md = [{"name": name, "value": value, "unit": unit, "provenance": "measured",
           "source": "test"}
          for name, (value, unit) in metrics.items()]
    rep: dict = {"tool": "kernoscope",
                 "kernels": [{"name": "k", "sections": [{"name": "S", "metrics": md}]}]}
    if gpu_s is not None:
        rep["cost"] = {"gpu_s": gpu_s}
    return rep


def _runner(*results):
    """按顺序吐出脚本化的返回值；`BaseException` 会被 raise（模拟 runner 自己抛）。"""
    seq = list(results)

    def run():
        got = seq.pop(0)
        if isinstance(got, BaseException):
            raise got
        return got

    return run


# --------------------------------------------------------------------------
# 1) 已知spread：mean/sd/min/max 手算对拍
# --------------------------------------------------------------------------
def test_three_runs_known_spread_gives_hand_computed_mean_sd_min_max():
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}),
                              _report({"dram_bytes": (12.0, "MB")}),
                              _report({"dram_bytes": (14.0, "MB")})), 3)

    d = res.distributions["dram_bytes"]
    assert (d.n, d.not_produced) == (3, 0)
    assert d.values == (10.0, 12.0, 14.0)          # 原始样本留着：分布可复核
    assert d.mean == pytest.approx(12.0)
    # 样本 sd（n−1 分母）：((−2)²+0²+2²)/2 = 4 → sd = 2。总体 sd 会是 1.633，
    # 两者差在这里就是"有没有分辨力"。
    assert d.sd == pytest.approx(2.0)
    assert d.sd != pytest.approx(math.sqrt(8.0 / 3.0))
    assert (d.min, d.max) == (10.0, 14.0)
    assert d.unit == "MB"
    # MDD 下界 = max(3×sd/|mean|×100, 0.5) = max(3×2/12×100, 0.5) = 50.0
    assert res.mdd_lower_bound["dram_bytes"] == pytest.approx(50.0)
    assert res.refusals == {} and res.refusal is None
    # 缓存冷热没被控制 —— 这句话必须出现在结果里（不是只在文档里）
    assert any("缓存" in n for n in res.notes)


def test_mdd_floor_keeps_a_zero_sd_from_becoming_zero_mdd():
    """N 次完全相同（sd=0）时 MDD 不能是 0：那会把任何抖动都判成真变化。"""
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}),
                              _report({"dram_bytes": (10.0, "MB")}),
                              _report({"dram_bytes": (10.0, "MB")})), 3)
    d = res.distributions["dram_bytes"]
    assert d.sd == pytest.approx(0.0)
    assert res.mdd_lower_bound["dram_bytes"] == pytest.approx(MDD_FLOOR_PCT)
    assert res.mdd_lower_bound["dram_bytes"] > 0.0


# --------------------------------------------------------------------------
# 2) 1 成功 + 2 失败：按 code 拒绝，失败的那几次必须看得见
# --------------------------------------------------------------------------
def test_one_success_and_two_failures_refuse_with_code_and_failures_stay_visible():
    # 失败那两次**也**带着数和报告回来：它们仍然不许进分布
    # （失败的报告常常是截断的，塞进去只会把 sd 压小、把 MDD 压低）
    res = repeat_runs(_runner(_report({"dram_bytes": (100.0, "MB")}),
                              (1, _report({"dram_bytes": (100.0, "MB")})),
                              (1, _report({"dram_bytes": (100.0, "MB")}))), 3)

    assert [(r.index, r.ok, r.rc) for r in res.runs] == [(0, True, 0), (1, False, 1),
                                                         (2, False, 1)]
    assert "dram_bytes" in res.runs[1].metrics       # 报告里的数照样看得见
    assert res.runs[1].report is not None            # 失败 ≠ 记录被丢掉

    d = res.distributions["dram_bytes"]
    assert (d.n, d.not_produced) == (1, 0)           # 只有成功的那一次进分布
    assert d.sd is None                              # 一次算不出 sd —— 但绝不当成 0

    r = res.refusals["dram_bytes"]
    assert r.code is RefusalCode.MDD_UNKNOWN
    assert r.observed["n"] == 1 and r.observed["ok_runs"] == 1
    assert r.observed["failed_runs_with_metric"] == 2
    assert r.threshold["min_runs_per_metric"] == MIN_RUNS_PER_METRIC
    assert r.remedy_or_default.action                  # 拒绝必须带"怎么办"
    assert res.refusal is not None and res.refusal.code is RefusalCode.MDD_UNKNOWN
    assert res.mdd_lower_bound["dram_bytes"] is None
    assert any("不参与分布" in n for n in res.notes)


def test_all_runs_failing_yields_aggregate_refusal_and_no_distributions():
    res = repeat_runs(_runner((1, None), (1, None), (1, None)), 3)
    assert len(res.runs) == 3 and all(not r.ok for r in res.runs)
    assert res.distributions == {} and res.refusals == {} and res.mdd_lower_bound == {}
    assert res.refusal is not None
    assert res.refusal.code is RefusalCode.MDD_UNKNOWN
    assert res.refusal.observed["runs"] == 3
    assert res.cost["runs"] == 3                     # 失败的那几次照样计成本
    assert res.refusal.remedy_or_default.action


def test_runner_exception_is_recorded_as_a_failed_run_not_dropped():
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}),
                              RuntimeError("采集进程没了"),
                              _report({"dram_bytes": (14.0, "MB")})), 3)
    assert [(r.index, r.ok) for r in res.runs] == [(0, True), (1, False), (2, True)]
    assert res.runs[1].report is None
    assert "RuntimeError" in res.runs[1].error and "采集进程没了" in res.runs[1].error
    d = res.distributions["dram_bytes"]
    # 抛异常的那次计入 N（runs/cost），但不进分布、也不算"成功却没产出"
    assert (d.n, d.not_produced) == (2, 0)
    assert d.n + d.not_produced == res.ok_runs == 2
    assert d.sd == pytest.approx(math.sqrt(8.0))     # (10,14) → 样本 sd = √8
    assert res.refusal is None                       # 两次成功运行够了


# --------------------------------------------------------------------------
# 3) 同名不同单位：宁缺勿混
# --------------------------------------------------------------------------
def test_mixed_units_for_one_metric_name_refuse_instead_of_mixing():
    res = repeat_runs(_runner(
        _report({"dram_bytes": (100.0, "MB"), "kernel_time": (1.0, "ms")}),
        _report({"dram_bytes": (100.0, "ms"), "kernel_time": (1.2, "ms")}),
        _report({"dram_bytes": (100.0, "MB"), "kernel_time": (1.4, "ms")})), 3)

    assert "dram_bytes" not in res.distributions      # 不混单位：宁缺勿混
    r = res.refusals["dram_bytes"]
    assert r.code is RefusalCode.PROTOCOL_MISMATCH
    assert r.observed["units"] == ["MB", "ms"]
    assert r.threshold["units"] == 1
    assert "MB" in r.message and "ms" in r.message    # 报文点名是哪两种口径
    assert res.mdd_lower_bound.get("dram_bytes") is None
    # 口径一致的那个指标照常出分布 —— 一个指标被拒不等于整批被拒
    assert res.distributions["kernel_time"].n == 3
    assert res.mdd_lower_bound["kernel_time"] is not None
    assert res.refusal is None


# --------------------------------------------------------------------------
# 4) 只在 1 次运行里出现的指标：逐指标拒绝
# --------------------------------------------------------------------------
def test_metric_in_only_one_run_is_refused_while_other_metrics_are_reported():
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB"), "l2_hit_rate": (0.9, "%")}),
                              _report({"dram_bytes": (12.0, "MB")}),
                              _report({"dram_bytes": (14.0, "MB")})), 3)

    assert res.distributions["dram_bytes"].n == 3
    assert res.mdd_lower_bound["dram_bytes"] is not None
    assert "dram_bytes" not in res.refusals

    d = res.distributions["l2_hit_rate"]
    assert (d.n, d.not_produced, d.sd) == (1, 2, None)
    assert res.refusals["l2_hit_rate"].code is RefusalCode.MDD_UNKNOWN
    assert res.refusals["l2_hit_rate"].observed["n"] == 1
    assert res.mdd_lower_bound["l2_hit_rate"] is None
    assert res.refusal is None                        # 整体可用：diff 还能用别的指标
    # 计数不丢不重：n + 没产出的次数 == 成功运行次数
    assert all(x.n + x.not_produced == res.ok_runs == 3
               for x in res.distributions.values())


# --------------------------------------------------------------------------
# 5) 成本 = N 次运行；只出 GPU 秒，不编价格
# --------------------------------------------------------------------------
def test_cost_accounting_is_n_runs_and_only_measured_gpu_seconds():
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}, gpu_s=2.5),
                              _report({"dram_bytes": (12.0, "MB")}, gpu_s=2.5),
                              _report({"dram_bytes": (14.0, "MB")}, gpu_s=2.5)), 3)
    assert res.cost["runs"] == 3
    assert res.cost["runs_with_gpu_s"] == 3
    assert res.cost["gpu_s"] == pytest.approx(7.5)
    # 价格一个字都不许编：钱是 analyze.cost 按调用方给的价格算的
    assert not [k for k in res.cost if "price" in k or "money" in k or "currency" in k]
    assert "不顶替" in res.cost["basis"]

    # 失败的那次也花钱：照样进成本
    res2 = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}, gpu_s=2.0),
                               (1, None)), 2)
    assert res2.cost["runs"] == 2 and res2.cost["runs_with_gpu_s"] == 1
    assert res2.cost["gpu_s"] == pytest.approx(2.0)
    assert any("不是 N 次的总花费" in n for n in res2.notes)


def test_cost_leaves_gpu_seconds_empty_rather_than_substituting_wall_time():
    rep = _report({"dram_bytes": (10.0, "MB")})
    rep["wall_time_ns"] = 5_000_000_000              # 5 s wall：不是 GPU 秒
    res = repeat_runs(_runner(rep, _report({"dram_bytes": (12.0, "MB")})), 2)
    assert res.cost["gpu_s"] is None
    assert res.cost["runs_with_gpu_s"] == 0
    assert any("gpu_s" in n and "不猜" in n for n in res.notes)

    # 调用方要 wall 口径就得自己注入，并会体现在 basis 里
    res2 = repeat_runs(_runner(rep, _report({"dram_bytes": (12.0, "MB")})), 2,
                       gpu_s_of=lambda i, rc, r: (r or {}).get("wall_time_ns", 0) / 1e9)
    assert res2.cost["gpu_s"] == pytest.approx(5.0)
    assert "注入" in res2.cost["basis"]


# --------------------------------------------------------------------------
# 诚实规则：均值 0、报告内部自相矛盾、参数错误
# --------------------------------------------------------------------------
def test_zero_mean_refuses_relative_mdd_but_keeps_absolute_spread():
    res = repeat_runs(_runner(_report({"roofline": (-1.0, "%")}),
                              _report({"roofline": (0.0, "%")}),
                              _report({"roofline": (1.0, "%")})), 3)
    d = res.distributions["roofline"]
    assert (d.mean, d.sd, d.min, d.max) == (0.0, pytest.approx(1.0), -1.0, 1.0)
    assert res.mdd_lower_bound["roofline"] is None    # 键在，值为 None：有分布但算不出相对 MDD
    assert res.refusals["roofline"].code is RefusalCode.ZERO_BASELINE
    assert res.refusal is not None                    # 唯一指标不可用 ⇒ 整体拒绝
    assert res.refusal.code is RefusalCode.ZERO_BASELINE


def test_duplicate_metric_in_one_report_is_refused_not_silently_picked():
    def dup():
        return {"kernels": [{"name": "k", "sections": [{"name": "S", "metrics": [
            {"name": "dram_bytes", "value": 100.0, "unit": "MB"},
            {"name": "dram_bytes", "value": 999.0, "unit": "MB"}]}]}]}

    got = extract_metrics(dup())
    assert got["dram_bytes"].value == 100.0 and got["dram_bytes"].conflict is True
    res = repeat_runs(_runner(dup(), dup(), dup()), 3)
    assert "dram_bytes" not in res.distributions      # 不猜哪个对：不出分布
    r = res.refusals["dram_bytes"]
    assert r.code is RefusalCode.PROTOCOL_MISMATCH
    assert r.observed["conflicted_reports"] == [0, 1, 2]
    assert res.refusal is not None and res.refusal.code is RefusalCode.PROTOCOL_MISMATCH


def test_bad_arguments_raise_type_or_value_error():
    def boom():
        raise AssertionError("参数不合法时 runner 一次都不该被调用")

    with pytest.raises(TypeError):
        repeat_runs(42, 3)                            # type: ignore[arg-type]
    for bad in (0, -1, True, 1.5, "3"):
        with pytest.raises(ValueError):
            repeat_runs(boom, bad)                    # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 形态：结构化输出可序列化，逐次记录四个字段齐全
# --------------------------------------------------------------------------
def test_result_to_dict_is_json_serializable_and_run_records_are_complete():
    res = repeat_runs(_runner(_report({"dram_bytes": (10.0, "MB")}, gpu_s=1.0),
                              (2, _report({"dram_bytes": (11.0, "MB")}, gpu_s=1.0))), 2)
    assert isinstance(res, RepeatResult)
    d = res.to_dict()
    assert d["schema_version"] == repeat.REPEAT_SCHEMA_VERSION and d["tool"] == "kernoscope"
    assert set(d) == {"schema_version", "tool", "runs", "distributions", "mdd_lower_bound",
                      "refusal", "refusals", "cost", "notes"}
    for i, rec in enumerate(d["runs"]):
        assert {"index", "ok", "rc", "report"} <= set(rec)
        assert rec["index"] == i
    assert json.loads(json.dumps(d, ensure_ascii=False))["cost"]["runs"] == 2
    assert "进程级重复 2 次" in res.render()             # 人视图能出中文


def test_verdicts_layer_is_read_and_the_report_metric_table_wins():
    """决策包（diff 的输入）只有 verdicts：也要能读；两层都有时以指标表为准。"""
    packet = _runner(*[{"verdicts": [{"id": "dram_bytes", "value": v, "unit": "MB"},
                                     {"id": "kernel_time", "value": 1.0, "unit": "ms"}]}
                       for v in (10.0, 12.0, 14.0)])
    res = repeat_runs(packet, 3)
    assert res.distributions["dram_bytes"].mean == pytest.approx(12.0)
    assert res.distributions["kernel_time"].n == 3

    both = {"kernels": [{"name": "k", "sections": [{"name": "S", "metrics": [
        {"name": "dram_bytes", "value": 111.0, "unit": "MB"}]}]}],
        "verdicts": [{"id": "dram_bytes", "value": 999.0, "unit": "MB"}]}
    res2 = repeat_runs(_runner(both, both), 2)
    assert res2.distributions["dram_bytes"].mean == pytest.approx(111.0)


def test_mdd_sd_factor_matches_selftest():
    """同一个统计量不许分头漂移：2.8~3σ 的理由两处必须一致。"""
    from kernoscope.selftest import SD_FACTOR

    assert MDD_SD_FACTOR == SD_FACTOR == 3.0


def test_docstring_states_what_process_level_repetition_is_not():
    """要求写在文档里：这些"不是"决定了数字能用来做什么，必须能被读到。"""
    text = (repeat.__doc__ or "") + (repeat_runs.__doc__ or "")
    for phrase in ("不是 kernel 级重放", "不在一个进程内重跑 kernel", "不锁时钟",
                   "不碰其他租户", "warm/cold 缓存", "含缓存效应"):
        assert phrase in text, f"docstring 里缺少对「不是什么」的说明：{phrase!r}"
