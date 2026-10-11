"""W6 selftest + W7 diff。

selftest 的真机行为在 GPU 机器上验证（bench/artifacts）；这里守的是：
  * 没有 GPU/torch/NVML 时必须**拒绝**给误差带（不许猜），且缺依赖是成功路径；
  * 误差带与 MDD 的算法：已知答案偏差并集 + 重复离散度 + 仪器分辨率下限；
  * 输出文件同时能当 --ks-selfcheck 与 --ks-calibration 用（同构字段）。

diff 守的是四种判词的定义：
  * 同协议 + 差 < MDD → indistinguishable（**噪声不许叫改进**）
  * 同协议 + 差 > MDD → improved/regressed（按方向）
  * 协议不同 / 缺 MDD / 一侧被拒 → incomparable + 闭集 code
  * 方向不明确的判词进 not_judged，不替用户决定价值方向
"""

from __future__ import annotations

import json

import pytest

from kernoscope.diff_cmd import (IMPROVED, INCOMPARABLE, INDISTINGUISHABLE, NOT_JUDGED, REGRESSED,
                                 diff_packets, diff_main)
from kernoscope.refusals import RefusalCode
from kernoscope.report import DeviceInfo, KernelReport, Report
from kernoscope.packet import build
from kernoscope.selftest import SELFTEST_SCHEMA_VERSION, SelftestResult, WorkloadRun, run_selftest


# --------------------------------------------------------------------------
# W6
# --------------------------------------------------------------------------
def _run(wid: str, err: float, util: float = 96.8, ms: float = 1.0) -> WorkloadRun:
    return WorkloadRun(id=wid, analytic_bytes_per_iter=1e8,
                       measured_bytes_per_iter=1e8 * (1 + err / 100),
                       err_pct=err, mem_util_pct=util, duration_s=2.0, iters=20,
                       kernel_ms=ms)


def _filled(workloads, mdd_note=True) -> SelftestResult:
    from kernoscope.selftest import _fill_band_and_mdd

    res = SelftestResult(ok=True, device="RTX 4090", compute_capability="8.9")
    res.workloads = workloads
    _fill_band_and_mdd(res)
    return res


def test_band_is_union_of_known_answer_errors():
    """matmul 是 compute-bound：**不进**字节误差带（口径不适用，不是测量误差）。"""
    res = _filled([_run("copy", 9.1), _run("read", -4.1), _run("write", 12.5),
                   _run("matmul", 900.0)])
    b = res.band["dram_bytes_pct"]
    assert b["low"] == -4.1 and b["high"] == 12.5
    assert "已知答案" in b["basis"] and "DRAM-bound" in b["basis"]
    assert "matmul" in res.band["excluded"]


def test_mdd_uses_run_to_run_spread_and_resolution_floor():
    # 同负载重复：偏差 20.0 / 21.0 → SD≈0.7 → 2SD≈1.4；分辨率下限 100*1/50=2.0；
    # 误差带半宽 = (21-20)/2 = 0.5 → 取最大 2.0
    res = _filled([_run("copy", 20.0, util=50.0), _run("copy", 21.0, util=50.0)])
    assert res.mdd["resolution_floor_pct"] == 2.0
    assert res.mdd["band_half_width_pct"] == 0.5
    assert "max(" in res.mdd["basis"]
    # 实测漂移下限 5%：真机同协议两次运行差过 4.7%，只按 SD 算会把它报成「improved」
    assert res.mdd["dram_bytes_pct"] == 5.0
    assert res.mdd["floor_pct"] == 5.0 and res.mdd["floor_evidence"]


def test_mdd_never_smaller_than_half_the_band():
    """不同访问模式的代理偏差本身就能差几个百分点 —— MDD 必须把偏差差也算进去。

    （真机反例：copy +9.6% / read +7.8% / write +5.6%，同进程重复的 SD 只有 ~0.01%，
    如果 MDD 只看 SD 就会报 1.0%，于是"换了个 kernel 导致偏差变了 3 个点"会被判成真退化。）
    """
    res = _filled([_run("copy", 9.6), _run("read", 7.8), _run("write", 5.6)])
    assert res.mdd["band_half_width_pct"] == 2.0
    assert res.mdd["dram_bytes_pct"] >= 2.0
    assert res.mdd["dram_bytes_pct"] >= res.mdd["floor_pct"]


def test_cross_process_children_do_not_recurse(monkeypatch):
    """真机踩过的坑：子进程若继承 --cross-process，就会指数级起进程（永不结束）。"""
    import kernoscope.selftest as st

    seen: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = '{"workloads": [{"id": "copy", "err_pct": 9.0}]}'

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return _Proc()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = st.SelftestResult(ok=True, size_mb=64, duration_s=0.5, dtype="bf16")
    st._cross_process_samples(res, 2, ("copy", "read", "write"))
    assert len(seen) == 2
    for cmd in seen:
        assert "--cross-process" in cmd and cmd[cmd.index("--cross-process") + 1] == "0"


def test_cross_process_samples_widen_sd():
    from kernoscope.selftest import _fill_band_and_mdd

    res = SelftestResult(ok=True)
    res.workloads = [_run("copy", 9.0), _run("copy", 9.1)]     # 同进程：几乎不抖
    _fill_band_and_mdd(res, extra_runs={"copy": [4.0, 14.0]})  # 跨进程：抖得多
    assert res.mdd["run_to_run_sd_pct"] > 3.0
    assert res.mdd["cross_process_samples"] == {"copy": [4.0, 14.0]}


def test_mdd_never_below_half_a_point():
    res = _filled([_run("copy", 1.0, util=99.0), _run("copy", 1.0, util=99.0)])
    assert res.mdd["dram_bytes_pct"] >= 0.5


def test_low_saturation_is_flagged_as_optimistic():
    res = _filled([_run("copy", 5.0, util=12.0), _run("copy", 5.0, util=12.0)])
    assert any("偏乐观" in n for n in res.notes)


def test_selftest_refuses_without_torch_or_gpu():
    """缺依赖必须**拒绝**：给不出本机误差带时不许编一个。"""
    res = run_selftest(size_mb=8, duration_s=0.1, runs=1)
    if res.ok:                     # 在 GPU 机器上跑这个测试时会真的成功
        pytest.skip("这台机器能真跑 selftest（GPU 环境）")
    assert res.refusal is not None
    assert res.refusal.code in (RefusalCode.PYTORCH_MISSING, RefusalCode.NO_GPU,
                                RefusalCode.NO_NVML_SAMPLES)
    assert res.refusal.remedy_or_default.action
    assert "没能完成自检" in res.render()


def test_selftest_main_exit_codes(capsys):
    from kernoscope.selftest import selftest_main

    rc = selftest_main(["--quick", "--quiet"])
    out = capsys.readouterr().out
    if rc == 0 and "没能完成" not in out:
        pytest.skip("GPU 环境：缺依赖路径不适用")
    assert rc == 0                      # 缺依赖是成功路径（W1）
    rc2 = selftest_main(["--quick", "--strict", "--quiet"])
    assert rc2 == 2                     # --strict 才把它变成失败


def test_selftest_json_is_also_a_calibration_file():
    res = _filled([_run("copy", 9.1), _run("read", -4.1)])
    res.peaks = {"copy_gbs": 908.5, "read_gbs": 904.5, "matmul_tflops": 150.0}
    d = res.to_dict()
    assert d["measured_dram"]["copy_gbs"] == 908.5
    assert d["measured_dram"]["copy_method"]
    mm = d["measured_matmul_tflops"]["torch.bfloat16"]
    for k in ("acc_caliber", "precision_settings", "method"):
        assert mm[k], f"标定口径字段 {k} 缺失（roofline 会拒绝用它算拐点）"
    assert d["band"]["dram_bytes_pct"] and d["mdd"]["dram_bytes_pct"]


def test_torch_dtype_names_are_mapped_correctly():
    """真机踩过的坑：torch.bf16 不存在（是 torch.bfloat16），写错会静默变成"分配失败"。"""
    from kernoscope.selftest import TORCH_DTYPE_ATTR

    assert TORCH_DTYPE_ATTR == {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}

    class FakeTorch:
        float32 = "f32"
        bfloat16 = "bf16"
        float16 = "f16"

    for short, attr in TORCH_DTYPE_ATTR.items():
        assert getattr(FakeTorch, attr)


def test_selftest_schema_fields():
    res = _filled([_run("copy", 1.0)])
    d = res.to_dict()
    assert d["schema_version"] == SELFTEST_SCHEMA_VERSION and d["tool"] == "kernoscope"
    assert d["protocol"]["workloads"] == ["copy", "read", "write", "matmul"]


# --------------------------------------------------------------------------
# W7：构造两个包
# --------------------------------------------------------------------------
def _packet(*, kernel_ms: float = 4.75, dram_mb: float | None = 500.0,
            hash_: str = "sha256:aaaa", tier: int = 2, mdd_dram: float | None = 5.0,
            mdd_kernel: float | None = 3.0, refused_code: str | None = None,
            tool_version: str = "0.2.0") -> dict:
    """手搓一个最小决策包（比真跑一次快，且能精确控制每个字段）。"""
    def verdict(vid, value, unit="", mdd=None, direction_bearing=True):
        v = {"id": vid, "status": "soft", "value": value, "unit": unit,
             "provenance": "estimated", "facts": {}}
        if mdd is not None:
            v["mdd"] = mdd
        return v

    vs = [verdict("kernel_time", kernel_ms, "ms", mdd_kernel)]
    if dram_mb is None:
        vs.append({"id": "dram_bytes", "status": "refused", "facts": {},
                   "refusal": {"code": refused_code or "no_calibration"}})
    else:
        vs.append(verdict("dram_bytes", dram_mb * 1e6, "B", mdd_dram))
    vs.append({"id": "ledger_verdict", "status": "soft", "value": 3.0, "unit": "%",
               "provenance": "derived", "facts": {}})
    return {"schema_version": 1, "tool": "kernoscope", "tool_version": tool_version,
            "protocol": {"hash": hash_, "basis": "", "window_mode": "auto"},
            "env": {"tier": tier, "tier_name": "trace", "counter_state": "denied"},
            "verdicts": vs, "unmeasured": [], "refusals": {}, "next": [],
            "cost": {"wall_s": 10.0, "packet_bytes": 1000}}


def test_same_protocol_same_numbers_is_indistinguishable():
    r = diff_packets(_packet(), _packet())
    assert r.comparable
    assert r.counts()[INDISTINGUISHABLE] == 2      # kernel_time + dram_bytes
    assert r.counts()[IMPROVED] == 0 and r.counts()[REGRESSED] == 0
    assert all("噪声" in c.note for c in r.comparisons if c.status == INDISTINGUISHABLE)


def test_delta_below_mdd_is_noise_even_if_it_looks_better():
    """关键反例：少了 3% 看起来是"改进"，但 MDD 是 5% ⇒ 必须是 indistinguishable。"""
    r = diff_packets(_packet(dram_mb=500.0), _packet(dram_mb=485.0))
    c = [x for x in r.comparisons if x.id == "dram_bytes"][0]
    assert c.status == INDISTINGUISHABLE
    assert c.delta_pct == pytest.approx(-3.0, abs=0.05)


def test_real_regression_is_detected():
    r = diff_packets(_packet(dram_mb=500.0), _packet(dram_mb=560.0))
    c = [x for x in r.comparisons if x.id == "dram_bytes"][0]
    assert c.status == REGRESSED                # DRAM 字节变多 = 变坏
    assert c.delta_pct == pytest.approx(12.0, abs=0.05)


def test_real_improvement_is_detected():
    r = diff_packets(_packet(kernel_ms=5.0), _packet(kernel_ms=4.0))
    c = [x for x in r.comparisons if x.id == "kernel_time"][0]
    assert c.status == IMPROVED                 # 时长变短 = 变好


def test_percentage_verdicts_compare_in_points_not_ratio():
    a = _packet()
    b = _packet()
    b["verdicts"][1]["value"] = a["verdicts"][1]["value"] * 1.05   # +5% 相对
    r = diff_packets(a, b)
    c = [x for x in r.comparisons if x.id == "dram_bytes"][0]
    # MDD 是百分比 ⇒ 用相对差比较；5% == MDD ⇒ 仍算噪声（<=）
    assert c.status == INDISTINGUISHABLE


def test_without_mdd_everything_is_incomparable_with_code():
    a = _packet(mdd_dram=None, mdd_kernel=None)
    r = diff_packets(a, _packet(mdd_dram=None, mdd_kernel=None))
    codes = {c.refusal.code for c in r.comparisons if c.refusal}
    assert RefusalCode.MDD_UNKNOWN in codes
    assert r.counts()[INCOMPARABLE] >= 2
    assert "先跑" in r.render()


def test_mdd_from_selfcheck_file_is_used_and_conservative():
    a = _packet(mdd_dram=None, mdd_kernel=None)
    r = diff_packets(a, _packet(mdd_dram=None, mdd_kernel=None),
                     selfcheck={"mdd": {"dram_bytes_pct": 7.5, "kernel_time_pct": 2.0}})
    assert r.mdd["dram_bytes_pct"] == 7.5
    assert r.mdd_source.endswith("--selfcheck")
    assert r.counts()[INDISTINGUISHABLE] == 2


def test_mdd_takes_the_larger_of_packet_and_selfcheck():
    r = diff_packets(_packet(mdd_dram=3.0), _packet(mdd_dram=3.0),
                     selfcheck={"mdd": {"dram_bytes_pct": 9.0}})
    assert r.mdd["dram_bytes_pct"] == 9.0


def test_protocol_mismatch_blocks_direction_verdicts():
    a = _packet(hash_="sha256:aaaa")
    b = _packet(hash_="sha256:bbbb")
    r = diff_packets(a, b)
    assert not r.comparable
    assert "protocol_hash" in r.protocol_diffs
    assert all(c.status == INCOMPARABLE for c in r.comparisons)
    assert all(c.refusal.code is RefusalCode.PROTOCOL_MISMATCH for c in r.comparisons)
    # 差值仍然给出（信息不丢），只是不给方向判词
    assert all(c.delta is not None for c in r.comparisons)
    assert "不可比" in r.render()


def test_capability_tier_change_is_incomparable():
    r = diff_packets(_packet(tier=2), _packet(tier=3))
    assert "capability_tier" in r.protocol_diffs


def test_tool_version_change_is_incomparable():
    r = diff_packets(_packet(tool_version="0.2.0"), _packet(tool_version="0.3.0"))
    assert "tool_version" in r.protocol_diffs


def test_both_sides_refused_is_not_a_regression():
    r = diff_packets(_packet(dram_mb=None), _packet(dram_mb=None))
    c = [x for x in r.comparisons if x.id == "dram_bytes"][0]
    assert c.status == INCOMPARABLE
    assert c.refusal.code is RefusalCode.BOTH_REFUSED
    assert "不是" in c.refusal.message


def test_one_side_refused_is_metric_absent():
    r = diff_packets(_packet(dram_mb=None), _packet(dram_mb=500.0))
    c = [x for x in r.comparisons if x.id == "dram_bytes"][0]
    assert c.refusal.code is RefusalCode.METRIC_ABSENT


def test_missing_verdict_on_one_side_is_incomparable():
    a = _packet()
    b = _packet()
    b["verdicts"] = [v for v in b["verdicts"] if v["id"] != "kernel_time"]
    r = diff_packets(a, b)
    c = [x for x in r.comparisons if x.id == "kernel_time"][0]
    assert c.status == INCOMPARABLE
    assert c.refusal.code is RefusalCode.METRIC_ABSENT


def test_directionless_verdicts_go_to_not_judged():
    """ledger_verdict 没有价值方向：只给差值，不判 improved/regressed。"""
    r = diff_packets(_packet(), _packet())
    assert [c.id for c in r.not_judged] == ["ledger_verdict"]
    assert "不判方向" in r.render()


def test_json_output_is_machine_readable():
    d = diff_packets(_packet(), _packet(dram_mb=600.0)).to_dict()
    assert d["schema_version"] == 2
    assert set(d["summary"]) >= {IMPROVED, REGRESSED, INDISTINGUISHABLE, INCOMPARABLE}
    assert all("status" in c for c in d["comparisons"])


# --------------------------------------------------------------------------
# W7 CLI
# --------------------------------------------------------------------------
def _write_packet(path, **kw):
    p = _packet(**kw)
    import kernoscope.packet as pk

    pk.write_packet(p, str(path))
    return path


def test_diff_cli_end_to_end(capsys, tmp_path):
    a = _write_packet(tmp_path / "a.json")
    b = _write_packet(tmp_path / "b.json", dram_mb=700.0)
    rc = diff_main([str(a), str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "regressed" in out
    assert "Kernoscope diff" in out


def test_diff_cli_fail_on_regressed(capsys, tmp_path):
    a = _write_packet(tmp_path / "a.json")
    b = _write_packet(tmp_path / "b.json", dram_mb=700.0)
    rc = diff_main([str(a), str(b), "--fail-on", "regressed", "--quiet"])
    assert rc == 1
    rc2 = diff_main([str(a), str(_write_packet(tmp_path / "c.json")), "--fail-on",
                     "regressed", "--quiet"])
    assert rc2 == 0


def test_diff_cli_strict_on_incomparable(capsys, tmp_path):
    a = _write_packet(tmp_path / "a.json", hash_="sha256:aaaa")
    b = _write_packet(tmp_path / "b.json", hash_="sha256:bbbb")
    assert diff_main([str(a), str(b), "--strict", "--quiet"]) == 2


def test_diff_cli_json_mode(capsys, tmp_path):
    a = _write_packet(tmp_path / "a.json")
    rc = diff_main([str(a), str(a), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    d = json.loads(out)
    assert d["protocol"]["same"] is True


def test_diff_cli_missing_file(capsys, tmp_path):
    rc = diff_main([str(tmp_path / "nope.json"), str(tmp_path / "nope2.json")])
    assert rc == 1
    assert "找不到" in capsys.readouterr().err


def test_diff_accepts_full_reports_too(capsys, tmp_path):
    """回读路径：完整报告（.ksreport.json）也能直接 diff，不必先转成包。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    k = KernelReport("k", launch_count=1)
    k.sections = []
    rep.kernels = [k]
    path = tmp_path / "rep.ksreport.json"
    path.write_text(rep.to_json())
    rc = diff_main([str(path), str(path), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["protocol"]["same"] is True


# --------------------------------------------------------------------------
# R-58 / D16,D17：判据是"相对差 vs 相对误差带"；0 基线不能算成没变化
# --------------------------------------------------------------------------
def _with_verdict(p: dict, vid: str, value, unit: str, mdd: float | None = None) -> dict:
    p["verdicts"] = [v for v in p["verdicts"] if v["id"] != vid]
    v = {"id": vid, "status": "soft", "value": value, "unit": unit,
         "provenance": "estimated", "facts": {}}
    if mdd is not None:
        v["mdd"] = mdd
    p["verdicts"].append(v)
    return p


def test_percent_metric_uses_relative_band_not_percentage_points():
    """D16：roofline 20% → 15% 是**相对退化 25%**，不能被判成噪声。

    旧实现让 `%` 单位的判词拿**百分点**（5）去比**相对**误差带（14%）—— 于是真实的
    25% 退化被判成 indistinguishable。MDD 里的每个键量的都是相对离散度，所以判据必须
    一律是相对差。
    """
    a = _with_verdict(_packet(), "roofline", 20.0, "%", mdd=14.0)
    b = _with_verdict(_packet(), "roofline", 15.0, "%", mdd=14.0)
    r = diff_packets(a, b)
    c = next(x for x in r.comparisons if x.id == "roofline")
    assert c.delta_pct == pytest.approx(-25.0)
    assert c.status == REGRESSED, (c.status, c.note)
    assert "相对差" in c.note
    # 反向：相对差小于 MDD 才是噪声
    a2 = _with_verdict(_packet(), "roofline", 20.0, "%", mdd=14.0)
    b2 = _with_verdict(_packet(), "roofline", 19.0, "%", mdd=14.0)   # 相对 −5%
    c2 = next(x for x in diff_packets(a2, b2).comparisons if x.id == "roofline")
    assert c2.status == INDISTINGUISHABLE


def test_zero_baseline_is_not_reported_as_no_change():
    """D17：0 → 500 MB 是"相对差无定义"，不是"没变化"。

    旧实现 `delta_pct=None` 被折成 magnitude=0.0，于是判 indistinguishable 并印出
    「差值 0.00 在 MDD 5.0% 之内」—— 自相矛盾（差值是 5e8）。
    """
    a = _with_verdict(_packet(dram_mb=None), "dram_bytes", 0.0, "B", mdd=5.0)
    b = _with_verdict(_packet(dram_mb=None), "dram_bytes", 500e6, "B", mdd=5.0)
    r = diff_packets(a, b)
    c = next(x for x in r.comparisons if x.id == "dram_bytes")
    assert c.status == INCOMPARABLE
    assert c.refusal is not None and c.refusal.code is RefusalCode.ZERO_BASELINE
    assert "绝对差值" in c.refusal.message and "5e+08" in c.refusal.message
    # 两侧都是 0 仍然算"没变化"
    z1 = _with_verdict(_packet(dram_mb=None), "dram_bytes", 0.0, "B", mdd=5.0)
    z2 = _with_verdict(_packet(dram_mb=None), "dram_bytes", 0.0, "B", mdd=5.0)
    cz = next(x for x in diff_packets(z1, z2).comparisons if x.id == "dram_bytes")
    assert cz.status == INDISTINGUISHABLE


def test_not_judged_has_its_own_status_and_is_counted():
    """D25：只给差值、不下方向的项必须是独立状态，且计入 summary。"""
    a, b = _packet(), _packet()
    r = diff_packets(a, b)
    assert r.not_judged, "ledger_verdict 没有价值方向，应当在 not_judged 里"
    assert all(c.status == NOT_JUDGED for c in r.not_judged)
    counts = r.counts()
    assert counts[NOT_JUDGED] == len(r.not_judged)
    # 机器按 status 数出来的总数必须等于 summary 各格之和
    by_status = [c["status"] for c in r.to_dict()["comparisons"]] + \
                [c["status"] for c in r.to_dict()["not_judged"]]
    assert len(by_status) == sum(counts.values())
    assert set(by_status) <= set(counts)


def test_same_verdict_id_with_different_units_is_incomparable_not_subtracted():
    """第七轮 X-19（P1，口径级）：同一个判词 id 的两侧必须是**同一个量**。

    M13 的第一级（按实测时长占比，单位 `%`）与第二级（省下的钱，单位 `CNY/h`）曾经共用
    `saving_estimate` 这一个 id。实测：`diff` 把 `96.6 %` 与 `1.9801 CNY/h` 直接相减、
    吐出 `delta_pct=-97.95%` 并把单位标成 `CNY/h` —— 也就是本项目最老的错误族
    （用单一表示承载多重语义）。判词 id 已拆开（`saving_estimate_share` vs `saving_estimate`），
    这里是**第二道防线**：单位或 `facts.basis` 不同就不给差值、不给百分比。
    """
    a = _packet()
    b = _packet()
    a["verdicts"].append({"id": "saving_estimate", "status": "soft", "value": 96.6,
                          "unit": "%", "provenance": "exact",
                          "facts": {"basis": "time_share"}})
    b["verdicts"].append({"id": "saving_estimate", "status": "soft", "value": 1.9801,
                          "unit": "CNY/h", "provenance": "estimated",
                          "facts": {"basis": "money"}})
    res = diff_packets(a, b)
    row = next(c for c in res.comparisons if c.id == "saving_estimate")
    assert row.status == "incomparable", row.to_dict()
    assert row.delta is None and row.delta_pct is None, \
        f"把百分比和钱相减了：{row.to_dict()}"
    assert row.refusal is not None and row.refusal.code.value == "protocol_mismatch", row.to_dict()
    # 单位相同、basis 也相同时照常给差值（不许把这条防线做成"一律不给"）
    a2, b2 = _packet(), _packet()
    a2["verdicts"].append({"id": "saving_estimate", "status": "soft", "value": 10.0, "unit": "CNY/h",
                           "provenance": "estimated", "facts": {"basis": "money"}})
    b2["verdicts"].append({"id": "saving_estimate", "status": "soft", "value": 8.0, "unit": "CNY/h",
                           "provenance": "estimated", "facts": {"basis": "money"}})
    # 没有方向共识的判词会落在 `not_judged`（只给差值、不下方向判词），所以两边都查
    res2 = diff_packets(a2, b2)
    ok = next(c for c in (res2.comparisons + res2.not_judged) if c.id == "saving_estimate")
    assert ok.delta == -2.0 and ok.delta_pct == -20.0, ok.to_dict()
