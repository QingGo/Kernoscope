"""R-08 的验收：发射槽利用率上界与瓶颈方向（NVBit 动态指令计数）。

已知答案由手算给出（单条 launch）：
    warp 指令数 2,000,000；时长 2 ms；SM 128；时钟 2000 MHz；每 SM 4 调度器
    周期数 = 2e-3 × 2e9 = 4e6
    发射槽上限 = 4e6 × 128 × 4 = 2.048e9
    利用率 = 2e6 / 2.048e9 = 0.0977 %
两条 copy_k launch 合并后是它的两倍：0.1953 %（"按 launch 求和"这件事本身也要守住）。
"""

import json
import re

import pytest

from kernoscope.analyze.issue import from_histogram
from kernoscope.backends.opcode_hist import OpcodeHistError, parse
from kernoscope.cli import main

HIST = """kernel 0 - copy_k(float const*, float*, long) - #thread-blocks 1024,  \
kernel instructions 2000000, total instructions 2000000
  LDG.E.128 = 500000
  STG.E.128 = 500000
  IMAD = 1000000
kernel 1 - copy_k(float const*, float*, long) - #thread-blocks 1024,  \
kernel instructions 2000000, total instructions 4000000
  LDG.E.128 = 500000
  STG.E.128 = 500000
  IMAD = 1000000
kernel 2 - gemm_k(float const*, float*, long) - #thread-blocks 3072,  \
kernel instructions 900000000, total instructions 904000000
  HMMA.16816 = 800000000
  LDG.E = 100000000
"""


def run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_parse_counts_and_filter():
    hist = parse(HIST)
    assert hist.kernels == ["copy_k(float const*, float*, long)", "gemm_k(float const*, float*, long)"]
    assert len(hist.launches) == 3
    one = hist.for_kernel("copy_k")
    assert len(one.launches) == 2
    merged = one.merged()
    assert merged.warp_instrs == 4_000_000
    assert merged.opcode_counts["IMAD"] == 2_000_000
    assert hist.for_kernel("gemm_k").merged().warp_instrs == 900_000_000


def test_parse_rejects_output_without_records():
    """空直方图必须是错误，不能是"0 条指令"——那会把假数据一路带下去。"""
    with pytest.raises(OpcodeHistError):
        parse("NVBit: hello\n----------------\n")
    with pytest.raises(OpcodeHistError):
        parse("")


def test_issue_slot_upper_bound_known_answer():
    one = parse(HIST).for_kernel("copy_k").launches[0]
    st = from_histogram(one, duration_ns=2e6, sm_count=128, clock_mhz=2000.0)
    assert st.warp_instrs == 2_000_000

    def _stats(h):
        return from_histogram(h, duration_ns=2e6, sm_count=128, clock_mhz=2000.0)

    assert _stats(one).issue_slots == pytest.approx(4e6 * 128 * 4)
    assert _stats(one).issue_slot_utilization_pct == pytest.approx(0.0977, rel=2e-3)
    assert _stats(one).issued_ipc_per_sm == pytest.approx(2e6 / 4e6 / 128, rel=1e-3)
    assert _stats(one).category_pct["memory"] == pytest.approx(50.0)
    assert _stats(one).category_pct["alu"] == pytest.approx(50.0)
    # 同一 kernel 的两次 launch：指令数与占满的槽都翻倍
    two = parse(HIST).for_kernel("copy_k").merged()
    assert _stats(two).issue_slot_utilization_pct == pytest.approx(0.1953, rel=2e-3)
    assert from_histogram(two, duration_ns=2e6, sm_count=128,
                          clock_mhz=2000.0).warp_instrs_per_launch == pytest.approx(2_000_000)


def test_stall_direction_depends_on_both_premises():
    # 发射槽很低 + 显存也低 ⇒ 延迟受限
    low = from_histogram(parse(HIST).for_kernel("copy_k").merged(), duration_ns=2e6,
                         sm_count=128, clock_mhz=2000.0, dram_throughput_pct=5.0)
    assert "延迟受限" in low.stall_direction
    # 发射槽很低 + 显存很忙 ⇒ 显存受限
    mem = from_histogram(parse(HIST).for_kernel("copy_k").merged(), duration_ns=2e6,
                         sm_count=128, clock_mhz=2000.0, dram_throughput_pct=80.0)
    assert "显存受限" in mem.stall_direction
    # 发射槽很高 ⇒ 发射受限
    hot = from_histogram(parse(HIST).for_kernel("gemm_k").merged(), duration_ns=2e6,
                         sm_count=8, clock_mhz=1000.0)
    assert hot.issue_slot_utilization_pct > 60
    assert "发射受限" in hot.stall_direction


def test_missing_inputs_give_a_reason_not_a_number():
    st = from_histogram(parse(HIST).for_kernel("copy_k").merged(), duration_ns=0.0,
                        sm_count=128, clock_mhz=0.0)
    assert not st.available
    assert st.issue_slot_utilization_pct is None
    reason = st.unavailable_reason()
    assert "时长" in reason and "时钟" in reason


def test_missing_clock_prints_na_with_a_code_not_zero(capsys, tmp_path):
    """V-27：缺 SM 时钟读数时「发射槽上限」必须打 `n/a` + code，不许打 0。

    `issue_slots` 的算术在缺前提时给出的就是 0，而 **0 会被读成「测到 0」**（本项目自己的
    纪律：`n/a` 不是失败，是"前提不成立所以不给数"）。code 用现成的闭集
    （缺时钟 = `no_clock_reading`），处置随 code 走。
    """
    p = tmp_path / "hist.txt"
    p.write_text(HIST)
    rc, out, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                     "--sm-count", "128", "--clock-mhz", "0", "--duration-ms", "2")
    assert rc == 2
    line = next(ln for ln in out.splitlines() if "发射槽上限" in ln)
    assert "n/a" in line, line
    assert "no_clock_reading" in line, line
    assert not re.search(r"发射槽上限（本窗口）\s*:\s*0", line), line

    # --json 侧同一个纪律：前提不成立时给 null，不给 0
    rc2, out2, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                       "--sm-count", "128", "--clock-mhz", "0", "--duration-ms", "2", "--json")
    assert rc2 == 2
    doc = json.loads(out2)
    assert doc["issue_slots"] is None, doc["issue_slots"]
    assert doc["available"] is False and doc["unavailable_reason"]
    # 前提齐了才是数字（这条守卫也防止把"永远打 n/a"当成修复）
    rc3, out3, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                       "--sm-count", "128", "--clock-mhz", "2000", "--duration-ms", "2")
    assert rc3 == 0
    ok = next(ln for ln in out3.splitlines() if "发射槽上限" in ln)
    assert "n/a" not in ok and "2.05e+09" in ok, ok


def test_issue_subcommand_known_answer(capsys, tmp_path):
    p = tmp_path / "hist.txt"
    p.write_text(HIST)
    rc, out, err = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                       "--sm-count", "128", "--clock-mhz", "2000", "--duration-ms", "2")
    assert rc == 0
    assert "发射槽利用率" in out
    assert "0.2 %" in out      # 两次 copy_k launch 合并
    assert "延迟受限" in out
    assert "stall" in out

    rc2, out2, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                       "--sm-count", "128", "--clock-mhz", "2000", "--duration-ms", "2",
                       "--dram-pct", "80", "--json")
    doc = json.loads(out2)
    assert doc["issue_slot_utilization_pct"] == pytest.approx(0.1953, rel=2e-3)
    assert doc["launches"] == 2
    assert doc["category_pct"]["memory"] == pytest.approx(50.0)


def test_issue_json_metrics_are_isomorphic_with_report(capsys, tmp_path):
    """R-42：--json 的指标对象必须与主报告同构 —— 带 provenance，且与前缀规则一致。"""
    p = tmp_path / "hist.txt"
    p.write_text(HIST)
    rc, out, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                     "--sm-count", "128", "--clock-mhz", "2000", "--duration-ms", "2", "--json")
    assert rc == 0
    doc = json.loads(out)
    assert doc["metrics"], "指标列表不能为空"
    for m in doc["metrics"]:
        assert "provenance" in m, m
        if m["value"] is None:
            assert m["provenance"] == "unavailable", m
        elif m["name"].startswith("est__"):
            assert m["provenance"] == "estimated", m
    tags = {m["provenance"] for m in doc["metrics"]}
    assert tags <= {"estimated", "unavailable"}
    assert "exact" not in tags, "本子命令没有与 ncu 同源的来源，不允许出现 exact"

    # 缺输入时指标是 n/a：出处必须跟着变成 unavailable（R1 口径）
    rc2, out2, _ = run(capsys, "issue", "--from", str(p), "--kernel", "copy_k",
                       "--sm-count", "0", "--clock-mhz", "0", "--duration-ms", "0", "--json")
    assert rc2 == 2
    doc2 = json.loads(out2)
    unavailable = {m["name"]: m for m in doc2["metrics"] if m["value"] is None}
    assert unavailable, "缺输入时必须有 value=None 的指标"
    assert all(m["provenance"] == "unavailable" for m in unavailable.values())


def test_issue_subcommand_readable_errors(capsys, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("nothing useful here\n")
    rc, _, err = run(capsys, "issue", "--from", str(bad), "--sm-count", "1", "--clock-mhz", "1",
                     "--duration-ms", "1")
    assert rc == 1
    assert "error:" in err and "Traceback" not in err

    good = tmp_path / "good.txt"
    good.write_text(HIST)
    rc2, _, err2 = run(capsys, "issue", "--from", str(good), "--kernel", "nope",
                       "--sm-count", "1", "--clock-mhz", "1", "--duration-ms", "1")
    assert rc2 == 1
    assert "没有匹配" in err2

    rc3, out3, _ = run(capsys, "issue", "--from", str(good), "--sm-count", "0",
                       "--clock-mhz", "0", "--duration-ms", "0")
    assert rc3 == 2, "缺输入时必须返回「无法计算」而不是 0"
    assert "无法计算" in out3


def test_report_scales_histogram_to_the_reported_launch_count(capsys, tmp_path):
    """直方图那次跑的 launch 数与本次报告不同：必须按 per-launch 平均对齐，否则利用率无意义。"""
    p = tmp_path / "hist.txt"
    # 直方图：synthetic_elementwise 2 次 launch，每次 1000 warp 指令
    p.write_text("kernel 0 - synthetic_elementwise(float*) - #thread-blocks 2,  "
                 "kernel instructions 1000, total instructions 1000\n"
                 "  LDG.E = 600\n  IADD3 = 400\n"
                 "kernel 1 - synthetic_elementwise(float*) - #thread-blocks 2,  "
                 "kernel instructions 1000, total instructions 2000\n"
                 "  LDG.E = 600\n  IADD3 = 400\n")
    rc, out, _ = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:elementwise",
                     "--section", "SchedulerStats", "--ks-instr-hist", str(p),
                     "--ks-sm-count", "128", "--ks-clock-mhz", "2000")
    assert rc == 0
    assert "per-launch 平均对齐" in out
    # 20 次 launch（synthetic_elementwise）× (2000/2) 条/launch = 20000 条
    assert "est__dynamic_warp_instructions" in out
    line = [l for l in out.splitlines() if "est__dynamic_warp_instructions" in l][0]
    assert "20000" in line


def test_report_integration_puts_issue_slots_in_scheduler_sections(capsys, tmp_path):
    p = tmp_path / "hist.txt"
    p.write_text(HIST.replace("copy_k", "synthetic_gemm").replace("gemm_k", "synthetic_gemm"))
    rc, out, err = run(capsys, "--ks-synthetic", "--ks-no-nvml", "-k", "regex:gemm",
                       "--section", "SchedulerStats", "--section", "WarpStateStats",
                       "--ks-instr-hist", str(p), "--ks-sm-count", "128",
                       "--ks-clock-mhz", "2000")
    assert rc == 0
    assert "Section: SchedulerStats" in out
    assert "est__issue_slot_utilization_pct" in out
    assert "[estimated]" in out
    # 计数器版本的 stall 细分仍然只报 n/a + 原因
    assert "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" in out
    assert "n/a" in out
    assert "Section: WarpStateStats" in out
    assert "est__stall_direction" in out
