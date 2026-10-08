"""bench/sweep_row.py：把一次代理精度运行压成 CSV 的一行（R-35）。

这张表是"误差带必须带口径与边界"的载体，所以它自己也要有负样本：
最要紧的一条是**拒绝时要留空、不能留一个看起来像结论的数字**。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "bench"


def _load():
    spec = importlib.util.spec_from_file_location("ks_sweep_row", TOOLS / "sweep_row.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _report(tmp_path: Path, metrics: list[tuple[str, object, str]]) -> Path:
    doc = {"kernels": [{"name": "copy_k", "sections": [
        {"name": "DramTraffic",
         "metrics": [{"name": n, "value": v, "unit": u, "source": s}
                     for n, v, u, s in metrics]}]}]}
    p = tmp_path / "rep.ksreport.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_row_carries_the_premises_not_just_the_error(tmp_path: Path):
    """一行里必须同时有前提（覆盖率、两个份额、上偏估计）与结论（估算、账本、误差）。"""
    mod = _load()
    p = _report(tmp_path, [
        ("ks__sampler_window_coverage_pct", 95.2, "%", "采样点占比"),
        ("ks__dram_window_mem_busy_mean_pct", 95.2, "%", "窗口均值"),
        ("ks__matched_kernel_window_share_pct", 94.7, "%", "窗口份额"),
        ("ks__matched_kernel_kernel_share_pct", 100.0, "%", "内核时间份额"),
        ("ks__attribution_bias_estimate_pct", 0.0, "%", "上偏估计"),
        ("est__dram_bytes_per_launch", 585.08, "MB", "total / N launches"),
        ("est__dram_vs_ledger_err_pct", 9.0, "%", "（估算 − 账本）/ 账本，同为「每次 launch」口径"),
    ])
    row = mod.row(str(p), "8", 536.8709)
    assert row[0] == "8"
    assert row[1] == "95.2" and row[4] == "100.0" and row[5] == "0.0"
    assert row[6] == "585.08" and row[8] == "9.0"
    assert row[9] == "nvml"
    assert len(row) == len(mod.COLUMNS)


def test_row_leaves_the_error_empty_when_the_tool_refused(tmp_path: Path):
    """R-35 的负样本：前提不成立时工具拒绝给数，表里必须是**空**，不是 0、也不是估算值。"""
    mod = _load()
    p = _report(tmp_path, [
        ("ks__sampler_window_coverage_pct", 71.4, "%", "采样点占比"),
        ("ks__dram_window_mem_busy_mean_pct", 71.4, "%", "窗口均值"),
        ("ks__matched_kernel_window_share_pct", 87.9, "%", "窗口份额"),
        ("ks__matched_kernel_kernel_share_pct", 100.0, "%", "内核时间份额"),
        ("ks__attribution_bias_estimate_pct", 0.0, "%", "上偏估计"),
        ("est__dram_bytes_per_launch", None, "MB", "只能在可归因时给"),
        ("est__dram_vs_ledger_err_pct", None, "%", "误差行无法给出：窗口覆盖率 71% < 80%"),
    ])
    row = mod.row(str(p), "1.2", 536.8709)
    assert row[6] == "", "拒绝时不能给出 per-launch 估算值"
    assert row[8] == "", "拒绝时不能给出误差"
    assert row[9] == "unavailable", "要能看出这一行是「口径不成立」而不是「跑失败了」"
    assert row[1] == "71.4", "前提仍然要记下来：读者才知道为什么被拒绝"


def test_row_prefers_nvml_then_falls_back_to_the_trace_basis(tmp_path: Path):
    """NVML 口径不成立时，per-launch 那一列要取轨迹口径的值（否则整行没有信息）。"""
    mod = _load()
    p = _report(tmp_path, [
        ("ks__sampler_window_coverage_pct", 12.0, "%", "采样点占比"),
        ("est__dram_bytes_per_launch", None, "MB", "不可用"),
        ("est__dram_bytes_per_launch_trace", 234.9, "MB", "访存轨迹模型"),
        ("est__dram_vs_ledger_err_pct", 1.7, "%",
         "（估算 − 账本）/ 账本，同为「每次 launch」口径；实际口径：访存轨迹 per-launch（模型量）"),
    ])
    row = mod.row(str(p), "20", 234.888)
    assert row[6] == "234.9"
    assert row[9] == "trace", "口径列必须写清这次用的是哪条口径"
