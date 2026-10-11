"""`report_md.render_report_md` 的契约测试。

为什么这些断言值得写：报告一旦缺了 `## 0` 或缺件清单，读者就无法判断数字是在什么口径下
得到的；一旦 `unavailable` 被渲染成数字，"拒绝给数"就会静默变成"读数是 0"——这正是本项目
存在的理由。所以下面把模板的每条要求变成可执行的检查，而不是靠评审时肉眼比对。

覆盖的契约（每条对应"不能靠约定"的一条规则）：
  * 标题顺序固定（缺一节就要红，而不是靠人记得）；
  * 头部元数据块 + `Source` 列存在（口径与指标名可追溯）；
  * `**One-line read:**` 恰好在 `## 1` 里且是最后一行；
  * 每个 `## 2.x` 即使没有可用读数也出现，并给出拒绝码；
  * `## 4` 最多 5 条，且每条都有 `**Evidence:**`；没有证据时明说而不是给模板话术；
  * `## 5` 恰好三段式，且"不确定"的每一条都写了怎么解掉它；
  * `## 6` 有一条可运行命令；
  * 不可得的指标渲染成 `n/a` + code，任何一行都不许把它的值写成数字；
  * 输出是可提交的纯 Markdown（无 ANSI/框线）。
"""

from __future__ import annotations

import re

import pytest

from kernoscope import __version__

from kernoscope.packet import build
from kernoscope.provenance import Metric, Section
from kernoscope.refusals import RefusalCode
from kernoscope.report import DeviceInfo, KernelLaunch, KernelReport, Report
from kernoscope.report_md import MAX_PRIORITIES, render_report_md

# --------------------------------------------------------------------------
# 构造报告（不跑命令、不读文件：渲染层的输入就是这两个对象）
# --------------------------------------------------------------------------
_KERNEL = "big_gemm(float const*, float const*)"


def _launch(i: int, dur_ns: int = 1_000_000) -> KernelLaunch:
    return KernelLaunch(name=_KERNEL, demangled=_KERNEL, start_ns=i * 2_000_000,
                        end_ns=i * 2_000_000 + dur_ns, grid=(1024, 1, 1), block=(256, 1, 1),
                        registers_per_thread=64, static_smem_bytes=0, dynamic_smem_bytes=49152,
                        local_mem_per_thread=0, stream_id=0, device_id=0, grid_id=0,
                        correlation_id=i, short_name="big_gemm")


def _rich_report() -> Report:
    """一份"仪器齐全"的报告：每个维度都有读数，另加 3 条不可得指标。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128,
                                   l2_mib=72.0, driver="555.42"))
    rep.wall_time_ns = 25_000_000_000
    rep.meta["window"] = "mode=auto 采样 120 点 / 参与积分 20.0 s / 进程 wall 25.0 s"
    rep.meta["target"] = "python bench_loop.py"
    rep.meta["out_json"] = "rep.ksreport.json"
    rep.meta["kernel_filter"] = "regex:big_gemm"
    rep.sources = {"kernel activity": "nsys sqlite CUPTI_ACTIVITY_KIND_KERNEL",
                   "device counters": "nvidia-smi (management counters)"}

    kr = KernelReport(_KERNEL, launch_count=200, alias="big_gemm",
                      launches=[_launch(i) for i in range(200)])

    dur = Section("Duration")
    dur.add(Metric.exact("gpu__time_duration.sum", 200.0, "ms", source="CUPTI activity"))
    dur.add(Metric.exact("ks__launch_time_mean", 1.0, "ms"))
    dur.add(Metric.exact("ks__launch_time_min", 0.95, "ms"))
    dur.add(Metric.exact("ks__launch_time_max", 1.4, "ms"))
    dur.add(Metric.exact("ks__launch_count", 200))

    launch = Section("LaunchStats")
    launch.add(Metric.exact("launch__grid_size", 1024))
    launch.add(Metric.exact("launch__block_size", 256))
    launch.add(Metric.exact("launch__registers_per_thread", 64))
    launch.add(Metric.exact("launch__shared_mem_per_block_dynamic", 49152, "B"))
    launch.add(Metric.exact("ks__registers_used_for_occupancy", 64))

    occ = Section("Occupancy")
    occ.add(Metric.exact("ks__theoretical_occupancy_pct", 50.0, "%"))
    occ.add(Metric.exact("ks__blocks_per_sm", 5))
    occ.add(Metric.exact("ks__warps_per_sm", 32))
    occ.add(Metric.exact("ks__occupancy_limiter", "registers+shared_mem"))
    occ.add(Metric.measured("meas__achieved_occupancy_pct", 46.9, "%",
                            note="区间重叠法重建；需要 kernel 内探针"))
    occ.add(Metric.unavailable("sm__maximum_warps_per_active_cycle_pct", "needs counters",
                               source="ncu computes this from hardware counters",
                               code=RefusalCode.NO_COUNTERS, severity="degraded"))

    sol = Section("SpeedOfLight")
    sol.add(Metric.estimated("est__bound", "memory"))
    sol.add(Metric.estimated("est__roofline_efficiency_pct", 91.2, "%"))
    sol.add(Metric.estimated("est__achieved_gbs", 908.5, "GB/s"))

    dram = Section("DramTraffic")
    dram.add(Metric.estimated("est__dram_bytes_total", 5000.0, "MB"))
    dram.add(Metric.estimated("est__dram_bytes_per_launch", 25.0, "MB"))
    dram.add(Metric.derived("ks__sampler_window_coverage_pct", 97.1, "%"))

    mem = Section("MemoryTrace")
    mem.add(Metric.measured("meas__coalescing_efficiency_pct", 72.0, "%"))
    mem.add(Metric.measured("meas__sectors_per_request", 1.4))
    mem.add(Metric.estimated("est__l2_hit_rate_pct", 66.7, "%"))
    mem.add(Metric.estimated("est__working_set_bytes", 3.4e7, "B"))
    mem.add(Metric.estimated("est__dram_bytes_per_launch_trace", 22.0, "MB"))

    sched = Section("SchedulerStats")
    sched.add(Metric.estimated("est__issue_slot_utilization_pct", 27.4, "%"))
    sched.add(Metric.estimated("est__stall_direction", "latency"))
    sched.add(Metric.unavailable(
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "needs counters",
        code=RefusalCode.NO_COUNTERS, severity="degraded"))

    instr = Section("InstructionStats")
    instr.add(Metric.exact("ks__static_sass_instructions", 1234))
    instr.add(Metric.exact("ks__static_sass_share_tensor", 0.0, "%"))
    instr.add(Metric.exact("ks__static_sass_share_fp", 41.0, "%"))
    instr.add(Metric.unavailable("smsp__inst_executed.sum", "needs counters",
                                 code=RefusalCode.NO_COUNTERS, severity="degraded"))

    device = Section("DeviceStatistics")
    device.add(Metric.measured("meas__nvml_mem_busy_mean_pct", 96.8, "%",
                               resolution="1 %（整数百分比）"))
    device.add(Metric.derived("ks__window_busy_segments", 2.0, "段"))

    cost = Section("CostLedger")
    cost.add(Metric.unavailable("ks__cost_per_mtok", "没有给价格：算不出钱",
                                code=RefusalCode.NO_PRICE))
    cost.add(Metric.unavailable("ks__cost_per_hour", "没有给价格：算不出钱",
                                code=RefusalCode.NO_PRICE))

    kr.sections = [dur, launch, occ, sol, dram, mem, sched, instr, device, cost]
    rep.kernels = [kr]
    return rep


def _sparse_report() -> Report:
    """每个 `## 2.x` 只有一个**不可得**条目：用来验证"没有读数也必须占位给码"。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    kr = KernelReport(_KERNEL, launch_count=3, launches=[_launch(i) for i in range(3)])
    sec = Section("All")
    sec.add(Metric.unavailable("launch__grid_size", "多组配置：单值会误导",
                               code=RefusalCode.MIXED_LAUNCH_CONFIG))
    sec.add(Metric.unavailable("ks__launch_time_max", "逐 launch 时长不可得",
                               code=RefusalCode.SHARE_UNKNOWN))
    sec.add(Metric.unavailable("est__issue_slot_utilization_pct", "没有指令直方图",
                               code=RefusalCode.NO_INSTR_HIST))
    sec.add(Metric.unavailable("ks__static_sass_share_tensor", "没有动态计数器",
                               code=RefusalCode.NO_COUNTERS))
    sec.add(Metric.unavailable("ks__sampler_window_coverage_pct", "没有采样点",
                               code=RefusalCode.NO_NVML_SAMPLES))
    sec.add(Metric.unavailable("meas__coalescing_efficiency_pct", "没有访存轨迹",
                               code=RefusalCode.NO_MEMTRACE))
    kr.sections = [sec]
    rep.kernels = [kr]
    return rep


def _bare_report() -> Report:
    """只有一条 exact 时长：任何方向规则都不成立（用来验证"不产出模板建议"）。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    kr = KernelReport(_KERNEL, launch_count=1, launches=[_launch(0)])
    sec = Section("Duration")
    sec.add(Metric.exact("gpu__time_duration.sum", 1.0, "ms"))
    sec.add(Metric.unavailable("dram__bytes_read.sum", "计数器拿不到",
                               code=RefusalCode.NO_COUNTERS))
    kr.sections = [sec]
    rep.kernels = [kr]
    return rep


# --------------------------------------------------------------------------
# 解析辅助
# --------------------------------------------------------------------------
def _render(rep: Report, **kw) -> str:
    return render_report_md(rep, build(rep), **kw)


def _section(md: str, start: str, *ends: str) -> str:
    """取 [start, 第一个 end) 之间的文本；end 依次尝试（找不到就切到文末）。"""
    i = md.index(start)
    body = md[i:]
    for end in ends:
        j = body.find(end, len(start))
        if j != -1:
            return body[:j]
    return body


def _headings(md: str) -> list[str]:
    return [l for l in md.splitlines() if l.startswith("#")]


# --------------------------------------------------------------------------
# 结构
# --------------------------------------------------------------------------
_EXPECTED_HEADINGS = (
    "## 0. Profiling setup",
    "### Artifacts",
    "## 1. Headline numbers",
    "## 2. Per-dimension analysis",
    "### 2.1 SM occupancy & launch geometry",
    "### 2.2 Thread-block balance (tail effect)",
    "### 2.3 Instruction-level stall analysis",
    "### 2.4 Tensor Core utilization",
    "### 2.5 SM utilization timeline",
    "### 2.6 Memory access pattern",
    "### 2.7 Additional findings",
    "## 3. Summary diagnosis",
    "## 4. Optimization directions (ranked by impact)",
    "## 5. Confidence & caveats",
    "## 6. Reproduction",
)


def test_every_heading_appears_in_order():
    md = _render(_rich_report())
    positions = []
    for h in _EXPECTED_HEADINGS:
        assert h in md, f"缺少标题：{h}"
        positions.append(md.index(h))
    assert positions == sorted(positions), "标题顺序与模板不一致"
    # 一级标题 + Priority 小标题
    assert md.splitlines()[0].startswith("# `")
    assert md.splitlines()[0].endswith("Profiling Report")
    assert re.search(r"^### Priority 1 — \S", md, re.M), "`## 4` 必须给出至少一条 Priority"


def test_headings_have_no_decoration():
    md = _render(_rich_report())
    assert "\x1b" not in md, "输出里不许有 ANSI 转义"
    assert not any("\u2500" <= ch <= "\u257f" for ch in md), "输出里不许有框线字符"
    for line in _headings(md):
        assert line.startswith("#") and "=" not in line


# --------------------------------------------------------------------------
# 头部元数据块 + Source 列
# --------------------------------------------------------------------------
def test_header_block_exists_and_names_the_caliber():
    md = _render(_rich_report())
    head = _section(md, "**Kernel:**", "## 0.")
    for label in ("**Kernel:**", "**Target GPU:**", "**Tool:**",
                  "**Caliber/metric-name caveats:**"):
        assert label in head, f"头部块缺少 {label}"
    assert "RTX 4090" in head and "128 SM" in head and "sm_89" in head
    assert f"kernoscope {__version__}" in head
    # 口径说明必须点名自有前缀与 ncu 导入前缀 —— 否则读者会跨机器比较模型量
    for token in ("meas__", "est__", "model__", "ks__", "ncu__"):
        assert token in head, f"口径说明没有点明 {token}"


def test_headline_table_has_source_column_with_metric_names():
    md = _render(_rich_report())
    headline = _section(md, "## 1. Headline numbers", "## 2.")
    assert "| Metric | Value | Unit | Provenance | Source |" in headline
    row = next(l for l in headline.splitlines() if "gpu__time_duration.sum" in l)
    assert "| 200 |" in row and "ms" in row and "[exact]" in row
    # 每行 Source 里必须真的出现指标名
    for line in headline.splitlines():
        if line.startswith("|") and "---" not in line and "Metric | Value" not in line:
            assert "`" in line, f"Source 列没有指标名：{line}"


def test_one_line_read_is_present_and_inside_headline():
    md = _render(_rich_report())
    headline = _section(md, "## 1. Headline numbers", "## 2.")
    lines = [l for l in headline.splitlines() if l.strip()]
    assert lines[-1].startswith("**One-line read:** ")
    assert len(lines[-1]) > len("**One-line read:** ") + 5
    assert "\n" not in lines[-1]


# --------------------------------------------------------------------------
# 0. 设置 / Artifacts / 6. 复现
# --------------------------------------------------------------------------
def test_setup_comes_first_and_lists_command_window_sources_gaps_command():
    md = _render(_rich_report())
    assert md.index("## 0. Profiling setup") < md.index("## 1. Headline numbers")
    setup = _section(md, "## 0. Profiling setup", "## 1.")
    assert "python bench_loop.py" in setup, "必须写清是哪条命令产生的"
    assert "`auto`" in setup, "必须写窗口模式"
    assert "20.0 s" in setup, "必须写参与积分的窗口秒数"
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in setup, "必须列出数据源"
    assert "needs counters" in setup, "必须列出不可得指标的原因"
    assert "`counters_denied`" in setup, "不可得必须带拒绝码"
    assert "```bash" in setup, "设置节必须有一条可复制命令"
    cmd = setup.split("```bash", 1)[1].split("```", 1)[0].strip()
    assert cmd.startswith("kernoscope ") and cmd.endswith("bench_loop.py")


def test_artifacts_subsection_lists_outputs():
    md = _render(_rich_report())
    artifacts = _section(md, "### Artifacts", "## 1.")
    assert "rep.ksreport.json" in artifacts
    assert md.index("### Artifacts") < md.index("## 1. Headline numbers")


def test_reproduction_section_has_a_runnable_command_block():
    md = _render(_rich_report())
    repro = _section(md, "## 6. Reproduction")
    blocks = re.findall(r"```bash\n(.*?)```", repro, re.S)
    assert len(blocks) == 1, "`## 6` 恰好给一条可运行命令，避免读者挑错"
    assert blocks[0].strip().startswith("kernoscope ")


def test_reproduction_falls_back_when_target_is_unknown():
    rep = _rich_report()
    rep.meta.pop("target")
    rep.meta.pop("out_json")
    md = _render(rep)
    repro = _section(md, "## 6. Reproduction")
    block = re.search(r"```bash\n(.*?)```", repro, re.S)
    assert block and block.group(1).strip().startswith("kernoscope ")
    assert "不重现本次数字" in repro, "退回的命令必须明说它不重现本次数字"


# --------------------------------------------------------------------------
# 2.x：每个维度都在，没有读数也要给码
# --------------------------------------------------------------------------
def test_each_dimension_present_even_when_metric_unavailable():
    md = _render(_sparse_report())
    expected = {
        "### 2.1 SM occupancy & launch geometry": "mixed_launch_config",
        "### 2.2 Thread-block balance (tail effect)": "share_unknown",
        "### 2.3 Instruction-level stall analysis": "no_instr_hist",
        "### 2.4 Tensor Core utilization": "counters_denied",
        "### 2.5 SM utilization timeline": "no_nvml_samples",
        "### 2.6 Memory access pattern": "no_memtrace",
        "### 2.7 Additional findings": "metric_absent",
    }
    order = list(expected)
    for i, (heading, code) in enumerate(expected.items()):
        end = order[i + 1] if i + 1 < len(order) else "## 3."
        body = _section(md, heading, end)
        assert code in body, f"{heading} 没有给出拒绝码 {code}"
        if heading != "### 2.7 Additional findings":
            # 2.7 的兜底不是一个指标（没有 measurement 被拒），所以只要求 code + 原因
            assert "n/a" in body, f"{heading} 必须明确写 n/a"


def test_dimension_without_any_metric_says_unknown_not_zero():
    md = _render(_bare_report())
    for heading in ("### 2.3 Instruction-level stall analysis",
                    "### 2.4 Tensor Core utilization"):
        body = _section(md, heading, "### 2.5", "## 3.")
        assert "没有可用读数" in body
        assert "**结论：本维度未知。**" in body
        assert "拒绝码" in body


# --------------------------------------------------------------------------
# 4. 方向：硬上限、证据、无证据时不给模板话术
# --------------------------------------------------------------------------
def test_priorities_capped_and_every_one_has_evidence():
    md = _render(_rich_report())
    body = _section(md, "## 4. Optimization directions (ranked by impact)", "## 5.")
    numbers = re.findall(r"^### Priority (\d+) — (.+)$", body, re.M)
    assert 0 < len(numbers) <= MAX_PRIORITIES
    assert [n for n, _ in numbers] == [str(i) for i in range(1, len(numbers) + 1)]
    blocks = re.split(r"^### Priority \d+ — ", body, flags=re.M)[1:]
    assert len(blocks) == len(numbers)
    for block in blocks:
        assert "**Evidence:**" in block, "每条 Priority 必须有 Evidence 块"
        for bullet in block.split("**Evidence:**", 1)[1].split("**Expected impact:**")[0].splitlines():
            if bullet.startswith("- "):
                assert "`" in bullet, f"Evidence 行必须引用具体指标：{bullet}"
        assert "**Expected impact:**" in block
        assert re.search(r"\*\*Effort:\*\* (low|medium|high) — ", block)


def test_no_evidence_means_no_generic_advice():
    md = _render(_bare_report())
    body = _section(md, "## 4. Optimization directions (ranked by impact)", "## 5.")
    assert "### Priority" not in body, "没有证据时不许产出模板建议"
    assert "没有有证据支撑的优化方向" in body


def test_directions_are_traced_to_real_metrics():
    md = _render(_rich_report())
    body = _section(md, "## 4. Optimization directions (ranked by impact)", "## 5.")
    # 每条方向引用的指标名都必须真的在报告里出现过（不许凭空引用）
    names = set(re.findall(r"`([a-z0-9_]+__[A-Za-z0-9_.]+)`", body))
    index = {m.name for k in _rich_report().kernels for s in k.sections for m in s.metrics}
    for name in names:
        assert name in index, f"方向里引用了报告里不存在的指标 {name}"


# --------------------------------------------------------------------------
# 5. 三段式
# --------------------------------------------------------------------------
def test_confidence_has_exactly_three_parts():
    md = _render(_rich_report())
    body = _section(md, "## 5. Confidence & caveats", "## 6.")
    sure = "What I'm sure about:"
    unsure = "What I'm uncertain about:"
    cannot = "Anything the profile couldn't answer that the user should know:"
    for prefix in (sure, unsure, cannot):
        assert prefix in body, f"`## 5` 缺少 {prefix}"
    assert body.index(sure) < body.index(unsure) < body.index(cannot)
    # "不确定"的每一条都必须写清怎么才能变确定
    uncertain_block = body.split(unsure, 1)[1].split(cannot, 1)[0]
    bullets = [l for l in uncertain_block.splitlines()
               if l.startswith("- ") and not l.startswith("- …")]
    assert bullets, "rich 报告里有 estimated/measured/derived 读数，必须列出不确定项"
    for line in bullets:
        assert "要收紧它：" in line, f"不确定项没有写解法：{line}"


# --------------------------------------------------------------------------
# 数字纪律：不可得只渲染成 n/a + code
# --------------------------------------------------------------------------
_UNAVAILABLE_NAMES = (
    "sm__maximum_warps_per_active_cycle_pct",
    "smsp__inst_executed.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "ks__cost_per_mtok",
    "ks__cost_per_hour",
)


def test_unavailable_metrics_render_as_na_plus_code_never_as_a_number():
    md = _render(_rich_report())
    for name in _UNAVAILABLE_NAMES:
        assert re.search(rf"`{re.escape(name)}` = n/a（code=`", md), \
            f"{name} 必须渲染成 n/a + code"
        bad = re.search(rf"`{re.escape(name)}`\s*=\s*[-+]?[0-9]", md)
        assert bad is None, f"{name} 被渲染成了数字（拒绝给数不得变成 0）"
    # 表头里不可得的行：值列必须是 n/a
    headline = _section(md, "## 1. Headline numbers", "## 2.")
    row = next(l for l in headline.splitlines() if "ks__cost_per_mtok" in l)
    assert "| n/a" in row and "no_price" in row


def test_every_rendered_metric_line_has_a_provenance_tag():
    md = _render(_rich_report())
    tags = ("[exact]", "[derived]", "[measured]", "[estimated]", "[unavailable]")
    for line in md.splitlines():
        if line.startswith("- `") and "` = " in line:
            assert any(t in line for t in tags), f"指标行没有出处标签：{line}"


# --------------------------------------------------------------------------
# 选主体 / packet 可选 / title
# --------------------------------------------------------------------------
def _two_kernel_report() -> Report:
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    big = KernelReport("big_kernel", launch_count=10,
                       launches=[_launch(i, 2_000_000) for i in range(10)])
    small = KernelReport("small_kernel", launch_count=10,
                         launches=[_launch(i, 100_000) for i in range(10)])
    device = KernelReport("(device)", launch_count=0)
    for kr in (big, small, device):
        sec = Section("Duration")
        sec.add(Metric.exact("gpu__time_duration.sum", kr.total_ns / 1e6, "ms"))
        kr.sections = [sec]
    rep.kernels = [small, device, big]
    return rep


def test_default_subject_is_the_most_expensive_kernel():
    md = render_report_md(_two_kernel_report(), build(_two_kernel_report()))
    assert md.splitlines()[0] == "# `big_kernel` Profiling Report"


def test_substring_selection_and_unknown_name_raises():
    rep = _two_kernel_report()
    md = render_report_md(rep, build(rep), kernel="small")
    assert md.splitlines()[0] == "# `small_kernel` Profiling Report"
    with pytest.raises(ValueError, match="not found"):
        render_report_md(rep, None, kernel="does_not_exist")


def test_device_only_report_uses_device_pseudo_kernel():
    rep = Report(device=DeviceInfo(name="synthetic-device"))
    dev = KernelReport("(device)", launch_count=0)
    sec = Section("DeviceStatistics")
    sec.add(Metric.measured("meas__nvml_mem_busy_mean_pct", 12.0, "%"))
    dev.sections = [sec]
    rep.kernels = [dev]
    md = render_report_md(rep, None)
    assert md.splitlines()[0] == "# `(device)` Profiling Report"


def test_title_override_changes_only_the_h1():
    md = _render(_rich_report(), title="My Kernel Review")
    assert md.splitlines()[0] == "# My Kernel Review"
    assert "## 0. Profiling setup" in md


def test_packet_is_optional_and_absence_is_stated():
    md = render_report_md(_rich_report(), None)
    headline = _section(md, "## 1. Headline numbers", "## 2.")
    assert "**One-line read:**" in headline
    assert "没有提供决策包" in headline
    assert "## 0. Profiling setup" in md and "## 6. Reproduction" in md
    assert "没有决策包" in _section(md, "## 3. Summary diagnosis", "## 4.")


def test_bad_packet_type_fails_fast():
    with pytest.raises(TypeError, match="packet must be a dict"):
        render_report_md(_rich_report(), packet=["not", "a", "dict"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 第五轮 V-4：per-kernel 报告不许借用别的**真 kernel** 的读数
# --------------------------------------------------------------------------
def _over_peak_two_kernel_report() -> Report:
    """主体 kernel A 没有 `ks__over_peak_warning`，kernel B 有（且值很刺眼）。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    rep.wall_time_ns = 30_000_000_000
    rep.meta["kernel_filter"] = "regex:gemm"

    a = KernelReport("synthetic_gemm(float const*, float const*)", launch_count=5)
    a.sections = [Section("Duration", [
        Metric.exact("gpu__time_duration.sum", 4.0, "ms"),
        Metric.exact("ks__launch_count", 5.0, ""),
    ])]

    b = KernelReport("synthetic_elementwise(float*)", launch_count=20)
    b.sections = [Section("SpeedOfLight", [
        Metric.exact("gpu__time_duration.sum", 0.2, "ms"),
        # 这个数属于 B（例如 16361.8 TFLOPS = 峰值 9779.9%，是个"物理不可能"信号）
        Metric.derived("ks__over_peak_warning", 1.0, ""),
    ])]

    dev = KernelReport("(device)")
    dev.sections = [Section("DeviceStatistics", [
        # 设备级读数：**应当**能在 A 的报告里看到（它是这次测量的属性）
        Metric.measured("meas__nvml_mem_busy_mean_pct", 96.8, "%"),
    ])]
    rep.kernels = [a, b, dev]
    return rep


def test_per_kernel_report_never_borrows_another_kernels_metric():
    """V-4：主体 `synthetic_gemm` 的报告里出现 `synthetic_elementwise` 的 over_peak 值 ⇒
    读者会去查错 kernel。本 kernel 没有的名字必须是 n/a / 不出现，而不是借别人的。"""
    md = render_report_md(_over_peak_two_kernel_report(), kernel="synthetic_gemm")
    assert "ks__over_peak_warning" not in md, "借用了别的 kernel 的读数"
    # 而设备级读数（挂在 (device) 上）本来就该能看到
    assert "meas__nvml_mem_busy_mean_pct" in md
    assert "96.8" in md


def test_device_only_metrics_still_reach_a_per_kernel_report():
    """保留 `(device)` 兜底：NVML/窗口级读数是"这次测量"的属性，不兜底会凭空缺一半。"""
    from kernoscope.report_md import _metric_index

    rep = _over_peak_two_kernel_report()
    idx = _metric_index(rep, rep.kernels[0])
    assert "meas__nvml_mem_busy_mean_pct" in idx
    assert "ks__over_peak_warning" not in idx
