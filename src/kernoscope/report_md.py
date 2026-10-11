"""单 kernel 的 Markdown 报告渲染（``render_report_md``）。

为什么要有这一层：``render_text`` 面向"逐指标铺开"，``packet.render_onepage`` 面向
"一屏结论"，两者都不是**可以贴进 issue/PR/复盘文档、别人只看这一份就能复核**的报告。
真机使用里，一份 per-kernel 报告必须让人（或 agent）只读它就能回答三个问题：这次量了
什么、哪个数字不能信、下一步该动哪里。所以这里固定一份结构（沿用生态里唯一被验证过的
per-kernel 报告模板），并强制三件事：

1. **设置先于结论**（``## 0`` 排在最前，且含缺件清单与复现命令）。没有窗口模式、数据源、
   缺什么，读者无法判断数字是在什么口径下得到的，也无法自己再跑一遍 —— 一份不可复核的
   报告在决策里等于没有报告，只是更贵。
2. **缺失也必须占位**（每个 ``## 2.x`` 哪怕一个读数都没有也要出现）。"这个维度没条目"和
   "这个维度没问题"是完全不同的两件事；静默省略会把前者伪装成后者。所以缺读数时那一节
   必须写出拒绝码与原因：**缺失覆盖 = 未知，不是不存在**。
3. **数字必须带出处**：每个值都带单位与 provenance 标签；``unavailable`` 一律渲染成
   ``n/a`` + 结构化拒绝码。一个没有出处的数字在这份渲染里没有合法的输出路径 —— 这是
   ``provenance.py`` 那套硬规则在展示层的落点。

``## 4`` 的方向只能来自报告/决策包里**真实存在的读数**：没有证据时宁可明说"没有有证据
支撑的方向"，也不产出"提高并行度"这类模板话术（真机评测里，模板话术会把 agent 送去抠一个
不存在的空间）。优先级硬上限 5 条 —— 超过 5 条时排序本身就不再传递信息。

本模块只读 ``Report``/决策包，不跑命令、不做二次测量：渲染层不许产生新的数字，否则
"报告里的数字"就有了两个来源。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Sequence

from . import __version__
from .packet import next_ranked, phrase_verdict, premise_label, refusal_index, verdict_is_actionable
from .provenance import Metric, Provenance
from .refusals import RefusalCode, code_title
from .report import KernelReport, Report

#: 优先级硬上限。超过 5 条时排序失去意义（读者会当成清单逐条做，而不是按影响排序）。
MAX_PRIORITIES = 5
#: 单个维度最多铺开多少条指标（其余指向完整报告；报告可能很大，Markdown 是决策面）。
DIM_METRIC_CAP = 14
#: 设置节的缺件清单最多列多少组（按拒绝码归组；完整清单在 rep.json 里）。
SETUP_UNAVAILABLE_CAP = 10

#: 设备级伪 kernel 的名字（``sections.build_device_kernel``）。它承载 NVML/能耗等
#: 设备级读数，不是一个可优化的 kernel，所以选主体 kernel 时要排除它。
DEVICE_KERNEL = "(device)"


@dataclass(frozen=True)
class _Dimension:
    """``## 2.x`` 的一个维度：它认哪些指标名，以及没有读数时报哪个拒绝码。

    ``fallback_code`` 是"这个维度所依赖的动态口径在这台机器上会以什么原因被拒"。
    它必须来自 ``RefusalCode`` 闭集（而不是我们另编一句话）——这样"没数据"与
    "被什么挡住"是同一个可编程的判决，读者能直接按 code 去查 remedy。
    """

    number: str
    title: str
    patterns: tuple[str, ...]
    fallback_code: RefusalCode
    fallback_reason: str


#: 维度 → 指标名模式（``fnmatch`` 风格，支持 ``ks__cfg_*`` 这类动态名）。
#: 按**名字**而不是 section 名归属，是因为同一份读数在 `--set` 不同档位下会被放进
#: 不同 section（``SchedulerStats`` vs ``WarpStateStats``），而报告模板的维度是按
#: "要回答的问题"切的；按 section 归属会让同名读数在维度间闪烁。
_DIMENSIONS: tuple[_Dimension, ...] = (
    _Dimension(
        "2.1", "SM occupancy & launch geometry",
        (
            "ks__theoretical_occupancy_pct", "meas__achieved_occupancy_pct",
            "ks__occupancy_limiter", "ks__blocks_per_sm", "ks__warps_per_sm",
            "launch__grid_size", "launch__block_size", "launch__thread_count",
            "launch__registers_per_thread", "launch__shared_mem_per_block_static",
            "launch__shared_mem_per_block_dynamic", "launch__local_mem_per_thread",
            "ks__registers_used_for_occupancy", "ks__cubin_registers_per_thread",
            "ks__cubin_shared_bytes", "ks__dominant_config", "ks__launch_config_count",
            "ks__grid_sizes_observed", "ks__block_sizes_observed", "ks__cuda_graph_capture",
            "ks__cfg_*", "sm__maximum_warps_per_active_cycle_pct",
        ),
        RefusalCode.NO_OCCUPANCY_PROBE,
        "逐 SM 的实际驻留 warp 数需要 kernel 内探针（NVBit 插桩）；"
        "本节给出的 occupancy 是**算术上界**，不是达到值",
    ),
    _Dimension(
        "2.2", "Thread-block balance (tail effect)",
        (
            "ks__launch_time_min", "ks__launch_time_max", "ks__launch_time_mean",
            "ks__launch_count", "ks__launch_config_count", "ks__grid_sizes_observed",
            "launch__grid_size", "ks__dominant_config", "ks__cfg_*",
        ),
        RefusalCode.INSTRUMENTATION_MISSING,
        "tail effect 的直接证据是**逐 block 的起止时间线**（哪些 SM 先空下来），"
        "这需要多趟重放/插桩；本工具不重放，只能用 launch 时长与 grid 几何做间接推断",
    ),
    _Dimension(
        "2.3", "Instruction-level stall analysis",
        (
            "est__issue_slot_utilization_pct", "est__stall_direction",
            "est__issued_ipc_per_sm", "est__dynamic_warp_instructions",
            "est__warp_instructions_per_launch", "est__dyn_instr_share_*",
            "ks__static_sass_instructions", "ks__static_sass_share_fp",
            "ks__static_sass_share_alu", "ks__static_sass_share_memory",
            "ks__static_sass_share_sync", "ks__static_sass_share_control",
            "ks__static_sass_share_int", "ks__static_sass_share_other",
            "smsp__inst_executed.sum", "smsp__warp_issue_stalled_*",
            "ks__clock_assumption_mhz",
        ),
        RefusalCode.NO_INSTR_HIST,
        "stall 周期细分（long scoreboard / barrier / …）是硬件计数器或 PC 采样的产物；"
        "本机给的是发射槽利用率上界 + 方向性判断，不是 stall 分解",
    ),
    _Dimension(
        "2.4", "Tensor Core utilization",
        (
            "ks__static_sass_share_tensor", "sm__pipe_tensor*", "sm__inst_executed_pipe_tensor*",
            "sm__ops_path_tensor*",
        ),
        RefusalCode.NO_COUNTERS,
        "Tensor Core 的**动态**利用率是计数器指标（tensor pipe active cycles）；"
        "静态 SASS 里有没有 TC 指令只说明代码长相，不等于它在动",
    ),
    _Dimension(
        "2.5", "SM utilization timeline",
        (
            "ks__sampler_window_coverage_pct", "ks__window_busy_segments",
            "ks__dram_window_mem_busy_mean_pct", "meas__nvml_mem_busy_mean_pct",
            "est__dram_busy_equivalent_gbs", "meas__sm_clock_mean_mhz",
            "meas__sm_clock_max_mhz", "meas__power_max_w", "meas__device_energy_j",
            "meas__window_energy_j", "meas__window_power_mean_w",
        ),
        RefusalCode.SAMPLING_UNAVAILABLE,
        "逐时刻的 SM 利用率时间线需要 PC 采样或 SM 级计数器；"
        "本机只有 NVML 的区间平均（采样周期 200 ms），只能给窗口级忙碌占比",
    ),
    _Dimension(
        "2.6", "Memory access pattern",
        (
            "meas__sectors_per_request", "meas__coalescing_efficiency_pct",
            "meas__trace_working_set_mb", "meas__trace_records", "meas__trace_records_global",
            "meas__trace_sampled_pct", "est__l2_hit_rate_pct", "est__l2_hit_rate_l2only_pct",
            "est__l2_hit_rate_with_l1_pct", "est__working_set_bytes",
            "est__dram_bytes_total", "est__dram_bytes_per_launch",
            "est__dram_bytes_per_launch_trace", "est__dram_bytes_per_launch_trace_bounds",
            "est__dram_throughput_pct_of_peak", "est__dram_range_position_pct",
            "est__dram_vs_ledger_err_pct", "ks__dram_trace_vs_nvml_pct",
            "ks__ledger_range_verdict", "ks__attribution_bias_estimate_pct",
            "ks__matched_kernel_kernel_share_pct", "ks__matched_kernel_window_share_pct",
            "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
            "dram__bytes_read.sum", "dram__bytes_write.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        RefusalCode.NO_MEMTRACE,
        "访问模式（合并效率、请求粒度、L2 命中）需要访存轨迹插桩；"
        "没有轨迹时只能用 NVML 显存忙碌做字节代理，它不含访问模式信息",
    ),
)

#: 标题行。``## 2. Per-dimension analysis`` 是 2.x 的父标题。
_HEADLINE_TITLE = "## 1. Headline numbers"
_DIM_PARENT = "## 2. Per-dimension analysis"
_DIRECTIONS_TITLE = "## 4. Optimization directions (ranked by impact)"


# --------------------------------------------------------------------------
# 选择主体 kernel 与读指标
# --------------------------------------------------------------------------
def _select_kernel(rep: Report, kernel: str | None) -> KernelReport:
    """选报告主体：显式名字优先，否则取总时间最大的那个 kernel。

    为什么不默认取 ``rep.kernels[0]``：报告的 kernel 顺序取决于采集/过滤，不是"重要性"；
    一份报告贴出来时，读者默认关心的是最耗时的那个。``(device)`` 被排除，因为它是设备级
    读数的伪 kernel、没有可优化的代码。

    显式给了一个报告里不存在的名字时**报错**而不是退回默认：静默换主体会让读者以为
    在看 kernel A 的数字，实际是 kernel B —— 这是最坏的一类错误。
    """
    named = [k for k in rep.kernels if k.name != DEVICE_KERNEL]
    pool = named or list(rep.kernels)
    if not pool:
        raise ValueError("report has no kernels: nothing to render (check the采集/过滤条件)")
    if kernel:
        for k in pool:
            if k.name == kernel:
                return k
        for k in pool:
            if kernel in k.name:      # 允许用短的 nsys 名/子串
                return k
        known = ", ".join(f"`{k.name}`" for k in pool[:8]) or "（无）"
        raise ValueError(f"kernel {kernel!r} not found in report; available: {known}")
    return max(pool, key=lambda k: (k.total_ns, k.launch_count))


def _metric_index(rep: Report, kr: KernelReport) -> dict[str, Metric]:
    """``name -> Metric``。主体 kernel 的读数优先，其余 kernel（含设备级）兜底。

    设备级读数（NVML 忙碌/时钟/能耗）挂在 ``(device)`` 伪 kernel 上，但它们对 2.5 这类
    维度是唯一来源；兜底读取让"这个维度有没有数据"的答案不依赖于读数被放在了哪个 kernel
    对象里。同名冲突时主体 kernel 的读法优先 —— 报告要讲的是一个 kernel 的故事。
    """
    out: dict[str, Metric] = {}
    for s in kr.sections:
        for m in s.metrics:
            out.setdefault(m.name, m)
    for k in rep.kernels:
        if k is kr:
            continue
        for s in k.sections:
            for m in s.metrics:
                out.setdefault(m.name, m)
    return out


def _matched(index: dict[str, Metric], patterns: Sequence[str]) -> list[Metric]:
    """按模式取指标，保持报告里的原始顺序（读数顺序本身是"这条管线先算什么"的信息）。"""
    return [m for name, m in index.items() if any(fnmatch(name, p) for p in patterns)]


def _claimed_patterns() -> tuple[str, ...]:
    return tuple(p for d in _DIMENSIONS for p in d.patterns)


# --------------------------------------------------------------------------
# 渲染原语：数字只有这一条出口
# --------------------------------------------------------------------------
def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fmt_metric(m: Metric) -> str:
    """一个指标的**唯一**渲染出口：值 + 单位 + provenance 标签（不可得时是 n/a + code）。

    单位为空时显式写"（无单位）"而不是省略：省略会让 ``ks__static_sass_instructions = 1234``
    看起来像一个没有量纲问题的裸数，而"1234 条指令"与"1234"在读法上不是一回事。
    """
    tag = f"[{m.provenance.value}]"
    if m.provenance is Provenance.UNAVAILABLE:
        code = m.refusal.code.value if m.refusal is not None else RefusalCode.LEGACY_REPORT.value
        return f"`{m.name}` = n/a（code=`{code}`，{_one_line(m.reason) or '未记录原因'}） {tag}"
    unit = m.unit or "（无单位）"
    line = f"`{m.name}` = {_fmt_value(m.value)} {unit} {tag}"
    if m.resolution:
        line += f"（分辨率 {m.resolution}）"
    if m.note:
        line += " — " + _one_line(m.note)
    return line


def _metric_source(m: Metric) -> str:
    """``Source`` 列的单元格：指标名（不可得时附拒绝码）。"""
    if m.provenance is Provenance.UNAVAILABLE:
        code = m.refusal.code.value if m.refusal is not None else RefusalCode.LEGACY_REPORT.value
        return f"`{m.name}`（code=`{code}`）"
    return f"`{m.name}`"


def _prov_cell(m: Metric) -> str:
    if m.provenance is Provenance.UNAVAILABLE:
        return "[unavailable]"
    return f"[{m.provenance.value}]"


def _cell(text: Any) -> str:
    """表格单元格：Markdown 里 ``|`` 与换行会截断表格，必须先中和。"""
    return _one_line(text).replace("|", "\\|")


def _code_ref(code: RefusalCode | str) -> str:
    value = code.value if isinstance(code, RefusalCode) else str(code)
    return f"`{value}`（{code_title(value)}）"


def _group_unavailable(metrics: Sequence[Metric]) -> list[dict[str, Any]]:
    """把不可得指标按拒绝码归组：同一 code 的原因与 remedy 是一份，散文不必逐行重复。"""
    groups: dict[str, dict[str, Any]] = {}
    for m in metrics:
        if m.provenance is not Provenance.UNAVAILABLE:
            continue
        code = m.refusal.code.value if m.refusal is not None else RefusalCode.LEGACY_REPORT.value
        g = groups.setdefault(code, {"code": code, "count": 0, "examples": [],
                                     "reason": _one_line(m.reason), "severity": "blocking"})
        g["count"] += 1
        if len(g["examples"]) < 3 and m.name not in g["examples"]:
            g["examples"].append(m.name)
        if m.refusal is not None and m.refusal.severity == "degraded":
            g["severity"] = "degraded"
    return sorted(groups.values(), key=lambda g: (-g["count"], g["code"]))


# --------------------------------------------------------------------------
# 0. 设置与复现命令
# --------------------------------------------------------------------------
_WINDOW_MODE_RE = re.compile(r"mode=([A-Za-z_]+)")
_WINDOW_USED_RE = re.compile(r"参与积分\s*([0-9.]+)\s*s")


def _window_info(rep: Report, packet: dict[str, Any] | None) -> tuple[str, str, str]:
    """(模式, 参与积分的窗口秒数, 原始窗口描述)。

    秒数优先从 ``meta['window']`` 那句人话里解析（那是**实际参与积分**的窗口），
    而不是 ``meta['window_s']``（那是 ``--ks-window-s`` 的请求值，可能被自动裁剪改过）。
    两个值混淆会让复现命令与报告数字对不上：请求 20 s、实际只用 8 s 时，读者按 20 s
    去复现就会得到不同的覆盖率。
    """
    raw = _one_line((rep.meta or {}).get("window", ""))
    mode = ""
    if packet:
        mode = _one_line(((packet.get("protocol") or {}) or {}).get("window_mode", ""))
    if not mode:
        m = _WINDOW_MODE_RE.search(raw)
        mode = m.group(1) if m else ""
    used = ""
    m = _WINDOW_USED_RE.search(raw)
    if m:
        used = f"{m.group(1)} s"
    return mode or "未知", used or "未知", raw


def _target_command(rep: Report, packet: dict[str, Any] | None) -> tuple[str, str]:
    """本次测量的命令行 + 它的来源字段名（用于复现）。

    取值优先级：决策包/报告的 ``argv``（最完整）→ ``meta['target']``（CLI 实测会写）。
    两处都可能是空的（``-i`` 回读的报告、``--ks-synthetic``），那时**明说没有**，
    而不是编一条看起来能跑的命令。
    """
    for source, value in (
        ("packet['argv']", (packet or {}).get("argv")),
        ("rep.meta['argv']", (rep.meta or {}).get("argv")),
        ("rep.meta['target']", (rep.meta or {}).get("target")),
    ):
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value), source
        return _one_line(value), source
    return "", ""


def _repro_command(rep: Report, packet: dict[str, Any] | None) -> tuple[str, str]:
    """(一条可运行的复现命令, 这条命令的来源说明)。

    有目标命令行时重建一条采集命令：口径开关（窗口/暖机/kernel 过滤）从 ``meta`` 里读，
    因为它们决定数字的口径，不写进命令就不是"同一口径的复现"。没有目标命令行时退到
    ``-i <报告>`` 回读（那条路能重现这份报告的全部读数），再没有就明说不可复现。
    """
    target, source = _target_command(rep, packet)
    meta = rep.meta or {}
    if target:
        flags: list[str] = []
        mode = _one_line(meta.get("window") or "")
        m = _WINDOW_MODE_RE.search(mode)
        if m:
            flags.append(f"--ks-window {m.group(1)}")
        if meta.get("warmup_s"):
            flags.append(f"--ks-warmup {float(meta['warmup_s']):g}")
        if meta.get("window_s"):
            flags.append(f"--ks-window-s {float(meta['window_s']):g}")
        if meta.get("kernel_filter"):
            flags.append(f"-k {meta['kernel_filter']!r}")
        flags.append("-o rep")
        return f"kernoscope {' '.join(flags)} -- {target}", source
    out_json = _one_line(meta.get("out_json", ""))
    if out_json:
        return f"kernoscope -i {out_json} -o rep2", "rep.meta['out_json']"
    return ("kernoscope --ks-synthetic -o rep -f",
            "报告里没有目标命令行；上面这条只跑通采集/渲染流程，**不重现本次数字**")


def _render_setup(out: list[str], rep: Report, packet: dict[str, Any] | None) -> None:
    """``## 0. Profiling setup``：口径、数据源、缺件、复现命令。

    排序理由：先"这次怎么量出来的"（命令/窗口），再"数据从哪来"（sources），
    再"什么没拿到、为什么"（缺件），最后给可复制的命令。缺件清单排在命令之前，
    是因为读者在复制命令之前就该知道这次缺了什么，缺件会改变他对数字的信任程度。
    """
    meta = rep.meta or {}
    target, target_source = _target_command(rep, packet)
    mode, used, raw = _window_info(rep, packet)
    cmd, cmd_source = _repro_command(rep, packet)

    out.append("## 0. Profiling setup")
    out.append("")
    if target:
        out.append(f"- **测量的目标命令：** `{_cell(target)}`（来自 `{target_source}`）")
    else:
        out.append("- **测量的目标命令：** 未记录（报告里没有 `argv`/`target`；"
                   "本次可能是回读的旧报告或合成数据）")
    if packet and packet.get("protocol", {}).get("hash"):
        out.append(f"- **协议哈希：** `{packet['protocol']['hash']}`"
                   f"（口径相同才可比；不同协议的两次运行不能直接比大小）")
    out.append(f"- **窗口模式：** `{mode}`")
    out.append(f"- **窗口秒数（参与积分）：** {used}")
    if raw:
        out.append(f"- **窗口原始描述：** {_cell(raw)}")
    out.append("")
    out.append("**数据源（`rep.sources`）：**")
    out.append("")
    if rep.sources:
        for key, value in rep.sources.items():
            out.append(f"- `{_cell(key)}`：{_cell(value)}")
    else:
        out.append("- 未记录数据源：**这份报告没有 `sources`** —— 数字的来源因此不可追溯，"
                   "不要把其中的值当作可引用的测量结果")
    out.append("")

    groups = _group_unavailable(list(_all_metrics(rep)))
    out.append("**没有量到的（按拒绝码归组，含原因）：**")
    out.append("")
    if groups:
        for g in groups[:SETUP_UNAVAILABLE_CAP]:
            sev = "" if g["severity"] == "blocking" else "（degraded：口径更弱但仍可用）"
            out.append(f"- {_code_ref(g['code'])}{sev}，{g['count']} 个指标：{g['reason']}"
                       f"；例：{'、'.join('`' + e + '`' for e in g['examples'])}")
        if len(groups) > SETUP_UNAVAILABLE_CAP:
            out.append(f"- …还有 {len(groups) - SETUP_UNAVAILABLE_CAP} 组拒绝，"
                       f"完整清单见 `-o` 输出的 `*.ksreport.json`（这里不省略，只截断并留痕）")
    else:
        out.append("- 本报告没有 `unavailable` 条目：**不是「全都拿到了」，而是这一份里"
                   "没有记录拒绝**（旧格式报告就是这样，见 `legacy_report`）")
    out.append("")
    out.append("**复现命令（复制即可重跑同一口径）：**")
    out.append("")
    out.append("```bash")
    out.append(cmd)
    out.append("```")
    out.append(f"（命令来源：{cmd_source}）")
    out.append("")
    _render_artifacts(out, rep)
    out.append("")


def _render_artifacts(out: list[str], rep: Report) -> None:
    """``### Artifacts``：这次落盘了哪些产物。

    产物的价值在于"数字能被独立复核"：读者可以拿 CSV 自己算，或拿 Perfetto trace 看
    时间线。所以这里列的是**产物路径**（报告自己的读数在任何渲染里都不如原始产物权威）。
    """
    meta = rep.meta or {}
    out.append("### Artifacts")
    out.append("")
    rows = (
        ("JSON 报告", meta.get("out_json")),
        ("launch CSV", meta.get("out_launch_csv")),
        ("Perfetto trace", meta.get("out_perfetto")),
        ("nsys 报告", meta.get("nsys_rep")),
        ("nsys sqlite", meta.get("nsys_sqlite")),
    )
    any_row = False
    for label, value in rows:
        if value:
            out.append(f"- {label}：`{_cell(value)}`")
            any_row = True
    if not any_row:
        out.append("- 未记录产物路径：这份报告只存在于内存/标准输出里，"
                   "**没有可独立复核的原始产物**")
    out.append(f"- 报告 schema：`{rep.schema_version}`"
               f"（跨版本对比前先确认 schema 相同，否则字段语义可能已变）")


def _all_metrics(rep: Report) -> list[Metric]:
    return [m for s in rep.iter_sections() for m in s.metrics]


# --------------------------------------------------------------------------
# 1. Headline numbers
# --------------------------------------------------------------------------
#: 头条行：(人读标签, 候选指标名)。取第一个存在的候选；多个都想要时（min/max）全列。
_HEADLINE_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("总 GPU 时间", ("gpu__time_duration.sum",)),
    ("单次 launch 时长（均值）", ("ks__launch_time_mean",)),
    ("单次 launch 时长（最小 / 最大）", ("ks__launch_time_min", "ks__launch_time_max")),
    ("launch 次数", ("ks__launch_count",)),
    ("多组 launch 配置数", ("ks__launch_config_count",)),
    ("理论 occupancy 上界", ("ks__theoretical_occupancy_pct",)),
    ("实测 occupancy", ("meas__achieved_occupancy_pct",)),
    ("occupancy 限制因素", ("ks__occupancy_limiter",)),
    ("roofline 达成率", ("est__roofline_efficiency_pct",)),
    ("roofline 判定（compute/memory）", ("est__bound",)),
    ("DRAM 总量（代理口径）", ("est__dram_bytes_total",)),
    ("DRAM 每次 launch（代理口径）", ("est__dram_bytes_per_launch",)),
    ("发射槽利用率上界", ("est__issue_slot_utilization_pct",)),
    ("瓶颈方向", ("est__stall_direction",)),
    ("单位成本", ("ks__cost_per_mtok", "ks__cost_per_hour")),
    ("窗口覆盖率（估算前提）", ("ks__sampler_window_coverage_pct",)),
    ("动态 occupancy 计数器（ncu 口径）", ("sm__maximum_warps_per_active_cycle_pct",)),
)


def _render_headline(out: list[str], rep: Report, kr: KernelReport, index: dict[str, Metric],
                     packet: dict[str, Any] | None) -> None:
    out.append(_HEADLINE_TITLE)
    out.append("")
    out.append(f"主体 kernel：`{_cell(kr.name)}`（{kr.launch_count} 次 launch）")
    out.append("")
    out.append("| Metric | Value | Unit | Provenance | Source |")
    out.append("| --- | --- | --- | --- | --- |")
    for label, names in _HEADLINE_ROWS:
        present = [index[n] for n in names if n in index]
        if not present:
            continue
        values = " / ".join(_cell(_fmt_value(m.value) if m.provenance is not Provenance.UNAVAILABLE
                                 else "n/a") for m in present)
        units = " / ".join("—" if m.provenance is Provenance.UNAVAILABLE
                           else (_cell(m.unit) or "（无单位）") for m in present)
        provs = " / ".join(_prov_cell(m) for m in present)
        sources = " / ".join(_metric_source(m) for m in present)
        out.append(f"| {_cell(label)} | {values} | {units} | {provs} | {sources} |")
    out.append("")
    out.append(f"**One-line read:** {_one_line_read(rep, kr, index, packet)}")
    out.append("")


def _one_line_read(rep: Report, kr: KernelReport, index: dict[str, Metric],
                   packet: dict[str, Any] | None) -> str:
    """一句话结论。有决策包时**只**读包（人视图/机器视图不许各写一套散文）。

    ``phrase_verdict`` 是包侧唯一的散文生成器；这里复用它，再补一句"多少条被拒"，
    这样读者一眼能看出"这句话能不能当依据"。没有包时用报告里的 exact 读数拼一句，
    并明确写出"没有判词"是数据缺失而不是"没有发现问题"。
    """
    if packet:
        verdicts = packet.get("verdicts") or []
        if verdicts:
            idx = refusal_index(packet)
            ordered = sorted(verdicts, key=lambda v: _verdict_rank(str(v.get("id", ""))))
            first = next((v for v in ordered if verdict_is_actionable(v)), ordered[0])
            parts = [_one_line(phrase_verdict(first, idx))]
            refused = [v for v in verdicts if v.get("status") == "refused"]
            if refused:
                codes: list[str] = []
                for v in refused:
                    c = (v.get("refusal") or {}).get("code")
                    if c and c not in codes:
                        codes.append(str(c))
                parts.append(f"{len(refused)}/{len(verdicts)} 条判词被拒"
                             f"（{', '.join(codes[:3])}），它们不能当依据")
            else:
                parts.append(f"{len(verdicts)} 条判词全部成立（前提见 `## 3`）")
            return "；".join(p for p in parts if p)
        return ("报告里没有可下结论的判词：先看「没测到什么」（`## 0`），"
                "不要把一个空结论读成「没有问题」")
    total_ms = kr.total_ns / 1e6
    prov = index.get("gpu__time_duration.sum")
    tag = f"[{prov.provenance.value}]" if prov is not None else "[derived]"
    return (f"`{kr.name}`：{kr.launch_count} 次 launch，合计 {total_ms:.4g} ms {tag}；"
            f"没有提供决策包，因此本报告只有读数、没有判词（这是数据缺失，不是「没问题」）")


def _verdict_rank(vid: str) -> tuple[int, str]:
    """判词的阅读优先级：先"哪个 kernel 最耗时"，再瓶颈类，最后账本/成本。

    与 ``packet.BUILDERS`` 的顺序一致 —— 那个顺序本身就是"读者先看哪个"的意图；
    这里复制成常量而不是从 packet import，是为了让渲染层不对构建顺序产生隐式依赖。
    """
    order = ("kernel_time", "occupancy", "roofline", "issue_slot", "l2_locality",
             "dram_bytes", "dram_bytes_per_launch", "dram_bytes_trace", "ledger_verdict",
             "static_mix", "over_peak", "cost_per_token")
    try:
        return (order.index(vid), vid)
    except ValueError:
        return (len(order), vid)


# --------------------------------------------------------------------------
# 2. Per-dimension analysis
# --------------------------------------------------------------------------
def _render_dimensions(out: list[str], index: dict[str, Metric]) -> None:
    out.append(_DIM_PARENT)
    out.append("")
    for dim in _DIMENSIONS:
        _render_dimension(out, dim, index)
    _render_additional(out, index)


def _render_dimension(out: list[str], dim: _Dimension, index: dict[str, Metric]) -> None:
    """一个维度：有读数就铺开，没有也保留小节并写出拒绝码与原因。

    "没有读数"分两种，必须在同一节里说清，否则读者会把两者混淆：
      * 报告里**有** n/a 条目 —— 说明我们确实尝试过、被某个前提挡住（给出 code 与 observed）；
      * 报告里**连条目都没有** —— 说明这条管线本次没有覆盖该维度（缺失覆盖 = 未知）。
    """
    out.append(f"### {dim.number} {dim.title}")
    out.append("")
    present = _matched(index, dim.patterns)
    avail = [m for m in present if m.provenance is not Provenance.UNAVAILABLE]
    unav = [m for m in present if m.provenance is Provenance.UNAVAILABLE]
    if avail:
        out.append("可用读数：")
        out.append("")
        for m in avail[:DIM_METRIC_CAP]:
            out.append(f"- {_fmt_metric(m)}")
        if len(avail) > DIM_METRIC_CAP:
            out.append(f"- …还有 {len(avail) - DIM_METRIC_CAP} 条读数（完整列表见 `-o` 的 JSON 报告）")
    else:
        out.append("**本维度没有可用读数。**")
    out.append("")
    if unav:
        out.append("不可得（`n/a` = 拒绝给数，不是 0）：")
        out.append("")
        for m in unav[:DIM_METRIC_CAP]:
            out.append(f"- {_fmt_metric(m)}")
        if len(unav) > DIM_METRIC_CAP:
            out.append(f"- …还有 {len(unav) - DIM_METRIC_CAP} 条不可得指标")
        out.append("")
    if not unav:
        # 这一行不是客套：本维度的**动态口径**没有出现在报告里，与"读数是 0"是两件事。
        out.append(f"- {_code_ref(dim.fallback_code)}：{dim.fallback_reason}。"
                   f"本报告里没有这个口径的条目 —— 没有条目不是「读数为 0」。")
        out.append("")
    if not present:
        out.append(f"**结论：本维度未知。** 报告里没有任何属于它的读数，"
                   f"不要把「没条目」读成「没问题」（拒绝码：{_code_ref(dim.fallback_code)}）。")
        out.append("")


def _render_additional(out: list[str], index: dict[str, Metric]) -> None:
    """``2.7 Additional findings``：不属于上面任何维度的读数。

    为什么需要它：指标集是会长的（成本账、账本对拍、能耗…），而模板的维度是固定的；
    没有这一节，新指标就会在两个选择里二选一 —— 塞进一个语义不符的维度，或者消失。
    两者都会让读者基于错误的位置读数字。
    """
    claimed = _claimed_patterns()
    rest = [m for name, m in index.items() if not any(fnmatch(name, p) for p in claimed)]
    out.append("### 2.7 Additional findings")
    out.append("")
    if rest:
        for m in rest[:DIM_METRIC_CAP]:
            out.append(f"- {_fmt_metric(m)}")
        if len(rest) > DIM_METRIC_CAP:
            out.append(f"- …还有 {len(rest) - DIM_METRIC_CAP} 条读数（完整列表见 JSON 报告）")
    else:
        out.append(f"- {_code_ref(RefusalCode.METRIC_ABSENT)}："
                   f"本报告没有超出上述维度的附加读数 —— 不等于「没有别的问题」，"
                   f"只等于这份报告的指标集没有覆盖到别处。")
    out.append("")


# --------------------------------------------------------------------------
# 3. Summary diagnosis
# --------------------------------------------------------------------------
def _render_diagnosis(out: list[str], rep: Report, kr: KernelReport, index: dict[str, Metric],
                      packet: dict[str, Any] | None) -> None:
    out.append("## 3. Summary diagnosis")
    out.append("")
    facts: list[str] = []
    dur = index.get("gpu__time_duration.sum")
    mean_m = index.get("ks__launch_time_mean")
    means = (_fmt_metric(mean_m) if mean_m is not None else
             f"`ks__launch_time_mean` = {kr.mean_ns / 1e6:.4g} ms [derived]"
             f"（由总时长 ÷ launch 数算出）")
    if dur is not None:
        facts.append(f"- 主体 kernel `{kr.name}`：{_fmt_metric(dur)}，"
                     f"{kr.launch_count} 次 launch（均值 {means}）")
    elif kr.launch_count:
        facts.append(f"- 主体 kernel `{kr.name}`：{kr.launch_count} 次 launch，"
                     f"总时长 {kr.total_ns / 1e6:.4g} ms [derived]"
                     f"（`gpu__time_duration.sum` 不在本报告里）")
    if rep.wall_time_ns and kr.total_ns:
        share = 100.0 * kr.total_ns / rep.wall_time_ns
        facts.append(f"- 该 kernel 自身占进程 wall 时间 {share:.4g} % "
                     f"（分子只有这一个 kernel、分母是进程 wall；"
                     f"与包里 `kernel_time` 判词的「kernel 时间占 wall」分母相同、分子不同"
                     f"——后者是全部 kernel 之和。[derived]，两个输入都是 exact）")
    if facts:
        out.extend(facts)
    else:
        out.append("- 本报告没有时长读数：无法判断这个 kernel 占了多少时间")
    out.append("")
    if packet and packet.get("verdicts"):
        idx = refusal_index(packet)
        out.append("判词（来自决策包；状态与前提一起读）：")
        out.append("")
        for v in packet["verdicts"]:
            status = str(v.get("status", "?"))
            line = f"- **[{status}]** {_one_line(phrase_verdict(v, idx))}"
            if v.get("value") is not None and v.get("unit"):
                line += f"（{_fmt_value(v['value'])} {v['unit']}"
                if v.get("provenance"):
                    line += f"，[{v['provenance']}]"
                if v.get("mdd") is not None:
                    line += f"，MDD {v['mdd']}%"
                line += "）"
            out.append(line)
            for p in v.get("premises") or []:
                label = premise_label(str(p.get("id", "")))
                unit = p.get("unit", "")
                if p.get("observed") is None:
                    out.append(f"  - 前提「{label}」：观测值缺失 —— 无法判定"
                               + (f"（门槛 {p['threshold']}{unit}）" if p.get("threshold") is not None
                                  else ""))
                elif p.get("threshold") is None:
                    out.append(f"  - 前提「{label}」：{p['observed']}{unit}")
                else:
                    flag = "满足" if p.get("ok", True) else "**不满足**"
                    out.append(f"  - 前提「{label}」：{p['observed']}{unit}"
                               f"（门槛 {p['threshold']}{unit}）→ {flag}")
            if status == "refused":
                code = (v.get("refusal") or {}).get("code")
                if code:
                    out.append(f"  - 拒绝码：{_code_ref(code)}；这条判词的数字不可引用")
        for item in next_ranked(packet, limit=3):
            if item.get("code"):
                out.append(f"- 路径：先解 {_code_ref(item['code'])}"
                           + (f" —— {_cell(item['action'])}" if item.get("action") else ""))
            elif item.get("hint"):
                out.append(f"- 路径：{_cell(item['hint'])}")
    else:
        out.append("没有决策包/判词：本节只能给读数，不能给「哪个是瓶颈」的判决。"
                   "要判决请用 `kernoscope` 正常路径（它会生成决策包），"
                   "或把本报告回灌给 `packet.build()`。")
    out.append("")


# --------------------------------------------------------------------------
# 4. Optimization directions
# --------------------------------------------------------------------------
@dataclass
class _Direction:
    """一条**有证据**的方向。``score`` 只用于排序，不是一个"预期收益"的数字。"""

    name: str
    action: str
    evidence: list[str]
    impact: str
    effort: str
    score: int


def _val(index: dict[str, Metric], name: str) -> Any:
    m = index.get(name)
    if m is None or m.provenance is Provenance.UNAVAILABLE:
        return None
    return m.value


def _ev(index: dict[str, Metric], name: str) -> str | None:
    """一条证据行；指标不在报告里时返回 None（**不把缺席的指标当证据**）。

    证据的定义是"报告里真的读到了这个值"。把 `xxx = 不在本报告里` 铺进 Evidence 会让
    读者以为那条方向有更多支撑，实际只是清单更长 —— 所以缺席的候选直接不出现。
    """
    m = index.get(name)
    return _fmt_metric(m) if m is not None else None


def _evs(index: dict[str, Metric], *names: str) -> list[str]:
    return [e for e in (_ev(index, n) for n in names) if e]


def _directions(rep: Report, kr: KernelReport, index: dict[str, Metric],
                packet: dict[str, Any] | None) -> list[_Direction]:
    """从真实读数推方向。每条规则**必须有存在的指标**才成立，否则不产出。

    这是与"模板建议"的分界线：规则里的每个数都来自 index（即报告/包里真实存在的读数），
    并且只对**当前这份报告**下判断。做不到这一点时，宁可不给方向（``## 4`` 会明说），
    也不给一句放之四海皆准的话 —— 后者看起来专业，实际会把优化 effort 引到不存在空间。
    """
    out: list[_Direction] = []
    for rule in (_dir_tail, _dir_occupancy, _dir_roofline, _dir_coalescing, _dir_l2,
                 _dir_stall, _dir_tensor_static):
        d = rule(rep, kr, index, packet)
        if d is not None:
            out.append(d)
    out.sort(key=lambda d: -d.score)
    return out


def _dir_tail(rep: Report, kr: KernelReport, index: dict[str, Metric],
              packet: dict[str, Any] | None) -> _Direction | None:
    """tail effect：用 grid 几何 + 理论 blocks/SM 算"最后一波填了多少"。

    为什么可以算：这三个输入都是 exact/derived（CUPTI 的 grid、解析算术的 blocks/SM、
    设备属性 SM 数），且公式只用到它们本身。算出来的填充率是**算术推导**，会显式标
    ``[derived]``；它说明的是"几何上最后一波不可能满"，不是"实测到有空闲 SM"。
    """
    grid = _val(index, "launch__grid_size")
    bps = _val(index, "ks__blocks_per_sm")
    sm = rep.device.sm_count
    if not (isinstance(grid, (int, float)) and isinstance(bps, (int, float)) and sm):
        return None
    cap = float(bps) * float(sm)
    if cap <= 0 or grid <= 0:
        return None
    waves = math.ceil(float(grid) / cap)
    last_fill = float(grid) - (waves - 1) * cap
    fill_pct = 100.0 * last_fill / cap
    if fill_pct >= 90.0:
        return None       # 最后一波本来就基本满了：没有 tail 可修（不给模板建议）
    return _Direction(
        name="消除最后一波的填充空洞（tail effect）",
        action=(f"最后一波只覆盖了约 {fill_pct:.0f} % 的驻留容量：把 block 数改成 "
                f"`{cap:.0f}` blocks（= SM 数 × 每 SM 驻留 block 数）的整数倍，"
                f"或调整 block 大小让 grid 能整除波容量。改的是 launch 配置，不动 kernel 代码。"),
        evidence=_evs(index, "launch__grid_size", "ks__blocks_per_sm",
                      "ks__theoretical_occupancy_pct")
        + [f"波容量 = SM 数 {sm} × blocks/SM `ks__blocks_per_sm` = {cap:.0f} blocks，"
           f"wave 数 = {waves} 个，最后一波填充率 = {fill_pct:.0f} %"
           f"（由上面两个 exact/derived 读数算术推导，[derived]）"],
        impact=("预期影响：最后一波的空闲不会直接变成加速比 —— 它只说明**几何上有空间**。"
                "改完用同一条命令重跑，只有 `gpu__time_duration.sum` 的变化大于本机 MDD "
                "才算真改进（没有 MDD 就先跑 selftest）。"),
        effort="low — 只改 launch 配置（grid/block），不动 kernel 代码",
        score=72 if fill_pct < 60 else 58,
    )


def _dir_occupancy(rep: Report, kr: KernelReport, index: dict[str, Metric],
                   packet: dict[str, Any] | None) -> _Direction | None:
    """occupancy 限制因素：limiter 决定"改什么"，上界决定"还有多少空间"。"""
    theo = _val(index, "ks__theoretical_occupancy_pct")
    limiter = _val(index, "ks__occupancy_limiter")
    if theo is None or not limiter:
        return None
    limiters = [p for p in str(limiter).replace("+", " ").split() if p]
    hints = {
        "registers": "降低寄存器压力（缩短 live range、减少展开；`--maxrregcount` 是下策，"
                     "它以 spill 换 occupancy，可能更慢）",
        "shared_mem": "减小每 block 共享内存（分块/流水级数），让更多 block 能同时驻留",
        "blocks": "每 SM 的 block 数已达硬件上限：换更大的 block（更多 warp/block）才有空间",
        "warps": "warp 数已达上限：调 block 大小或降低每线程资源占用",
        "block": "block 维度受限：检查 block 大小是否是 warp 的整数倍",
    }
    actions = [hints.get(x, f"针对限制因素 `{x}` 降低每线程/每 block 的资源占用") for x in limiters]
    return _Direction(
        name="抬高 occupancy 上界（先解限制因素）",
        action=(f"理论 occupancy 上界只有 {_fmt_value(theo)} %，限制因素是 "
                f"`{limiter}`。按顺序处理：" + "；".join(actions)
                + "。上界是**算术值**，抬高它本身不是目的 —— 目的是让实测达到值/时间跟着动。"),
        evidence=_evs(index, "ks__theoretical_occupancy_pct", "ks__occupancy_limiter",
                      "meas__achieved_occupancy_pct", "launch__registers_per_thread",
                      "ks__registers_used_for_occupancy", "launch__shared_mem_per_block_static",
                      "launch__shared_mem_per_block_dynamic"),
        impact=("预期影响：`ks__theoretical_occupancy_pct` 会上升；它是否能转成时间收益，"
                "要看实测达成的 occupancy（`meas__achieved_occupancy_pct`，需要探针）或"
                "同命令重跑的时长差是否大于 MDD。"),
        effort=("medium — 要改 kernel 的寄存器/共享内存用量（源码级改动）"
                if any(x in ("registers", "shared_mem") for x in limiters)
                else "low — 先改 launch 配置（block 大小）"),
        score=68,
    )


def _dir_roofline(rep: Report, kr: KernelReport, index: dict[str, Metric],
                  packet: dict[str, Any] | None) -> _Direction | None:
    """roofline：先看判定与达成率，再看是否值得继续抠。

    达成率已经很高时**不能**再建议"去查访问模式"：那会把 effort 送到一个不存在的空间
    （决策包侧已经因为同一原因改过一次文案）。所以这里按达成率分叉。
    """
    bound = _val(index, "est__bound")
    eff = _val(index, "est__roofline_efficiency_pct")
    if eff is None:
        return None
    if str(bound) == "memory":
        if float(eff) >= 90.0:
            action = (f"已达 roofline {_fmt_value(eff)} %：不要再抠访存效率，改为**减少总字节**"
                      f"（算子融合、提高 L2 复用、减小中间张量精度）。先核对工作量输入"
                      f"（`--ks-gemm`/`--ks-flops`）与标定口径是否可比。")
            score = 50
        else:
            action = (f"memory bound 且达成率 {_fmt_value(eff)} %：先查访问模式 —— 合并效率、"
                      f"请求粒度、工作集是否驻留 L2。这是把达成率抬起来最直接的杠杆。")
            score = 74
    elif str(bound) == "compute":
        action = (f"compute bound（达成率 {_fmt_value(eff)} %）：先看静态指令混合与发射槽上界，"
                  f"删冗余指令/提高 ILP；不要再按「优化访存」的方向改。")
        score = 60
    else:
        return None
    return _Direction(
        name=f"按 roofline 判定处理（{bound} bound）",
        action=action,
        evidence=_evs(index, "est__bound", "est__roofline_efficiency_pct", "est__achieved_gbs",
                      "est__achieved_tflops", "ks__over_peak_warning"),
        impact=("预期影响：如果确实是 memory bound，`est__achieved_gbs` / "
                "`est__roofline_efficiency_pct` 会上升；这两个都是 estimated（依赖标定峰值），"
                "重跑前先确认标定口径与 workload 输入没变。"),
        effort="medium — 通常要改访问模式/数据布局（源码级），不是开关能解决的",
        score=score,
    )


def _dir_coalescing(rep: Report, kr: KernelReport, index: dict[str, Metric],
                    packet: dict[str, Any] | None) -> _Direction | None:
    """访存合并效率：只有轨迹口径才能给，有就直接是可操作信号。"""
    coal = _val(index, "meas__coalescing_efficiency_pct")
    if coal is None:
        coal = _val(index, "est__coalescing_efficiency_pct")
    if coal is None or float(coal) >= 90.0:
        return None
    return _Direction(
        name="修访存合并（相邻线程访问相邻地址）",
        action=(f"合并效率只有 {_fmt_value(coal)} %：检查线程索引与地址计算 —— 相邻线程是否"
                f"落在同一段连续地址、下标是否按 (blockIdx.x*blockDim.x + threadIdx.x) 展开。"
                f"合并差会让每个请求多取扇区，字节账和带宽都虚高。"),
        evidence=_evs(index, "meas__coalescing_efficiency_pct",
                      "est__coalescing_efficiency_pct", "meas__sectors_per_request"),
        impact=("预期影响：`meas__coalescing_efficiency_pct` 与 `est__dram_bytes_per_launch_trace` "
                "会一起动；真实收益要看重跑后的时长差。注意轨迹口径只测被追踪的 kernel。"),
        effort="medium — 要改数据布局或索引方式（源码级）",
        score=76,
    )


def _dir_l2(rep: Report, kr: KernelReport, index: dict[str, Metric],
            packet: dict[str, Any] | None) -> _Direction | None:
    """L2/工作集：命中率与工作集大小一起看才有意义（单独一个都不足以推方向）。"""
    hit = _val(index, "est__l2_hit_rate_pct")
    if hit is None:
        hit = _val(index, "est__l2_hit_rate_l2only_pct")
    ws = _val(index, "est__working_set_bytes")
    if hit is None or ws is None or float(hit) >= 80.0:
        return None
    l2 = rep.device.l2_mib
    ws_ev = _ev(index, "est__working_set_bytes")
    if ws_ev and l2:
        ws_ev += f"；设备 L2 = {_fmt_value(l2)} MiB（[exact]，来自报告的设备信息）"
    return _Direction(
        name="提高 L2 复用（工作集分块）",
        action=(f"模型给的 L2 命中率 {_fmt_value(hit)} %，工作集大小在报告里；"
                f"把工作集切到 L2 能装下的块、或提高 block 内的复用（寄存器/共享内存暂存），"
                f"让命中的部分不重复走到 DRAM。"),
        evidence=_evs(index, "est__l2_hit_rate_pct", "est__l2_hit_rate_l2only_pct",
                      "est__l2_hit_rate_with_l1_pct") + ([ws_ev] if ws_ev else []),
        impact=("预期影响：`est__l2_hit_rate_pct` 是**模型**（不是计数器），它上升只说明模型认为"
                "复用变好；要确认，走轨迹口径（`--ks-memtrace`）看 `meas__*` 字节是否下降。"),
        effort="medium — 分块/复用要改源码与数据流",
        score=62,
    )


def _dir_stall(rep: Report, kr: KernelReport, index: dict[str, Metric],
               packet: dict[str, Any] | None) -> _Direction | None:
    """发射槽方向：latency 与 throughput 的改法相反，方向错了 effort 全废。"""
    direction = _val(index, "est__stall_direction")
    if not direction:
        return None
    if str(direction) == "latency":
        action = ("方向是 latency：提高**并行度与独立性** —— 更多独立工作（展开、预取）、"
                  "缩短长依赖链；此时精简指令不会有用（发射槽本来就没喂满）。")
    else:
        action = ("方向是 throughput：**精简指令**（减少重算、用更宽的指令/向量化）；"
                  "此时增加并行度没有用（发射槽已接近上限）。")
    return _Direction(
        name=f"按发射槽方向处理（{direction}）",
        action=action,
        evidence=_evs(index, "est__stall_direction", "est__issue_slot_utilization_pct",
                      "est__issued_ipc_per_sm", "ks__static_sass_instructions"),
        impact=("预期影响：方向性判断，不是 stall 周期分解（那需要计数器）。"
                "改完看 `est__issue_slot_utilization_pct` 与时长差；差值要大于 MDD。"),
        effort="medium — 要改 kernel 的指令级结构（源码级）",
        score=55,
    )


def _dir_tensor_static(rep: Report, kr: KernelReport, index: dict[str, Metric],
                       packet: dict[str, Any] | None) -> _Direction | None:
    """Tensor Core：静态 SASS 里有/没有 TC 指令，能推出**不同**的下一步。

    这里刻意不给"提高 Tensor Core 利用率"这种方向：动态利用率是本机拿不到的计数器，
    建议它去提高一个我们没量过的量，就是把 agent 送去猜。
    """
    share = _val(index, "ks__static_sass_share_tensor")
    if share is None:
        return None
    return _Direction(
        name="核对 Tensor Core 是否被用上（静态口径）",
        action=(("静态 SASS 里有 Tensor Core 指令，但**动态**利用率需要计数器（本机拿不到）："
                 "先确认 dtype/layout 与 kernel 选择符合预期，别把「代码里有 TC 指令」"
                 "当成「TC 是瓶颈」。")
                if float(share) > 0 else
                ("静态 SASS 里一条 Tensor Core 指令都没有：如果这个算子本应走 TC"
                 "（bf16/fp16/tf32 的 matmul/卷积），先查 dtype 与算子选择；"
                 "如果它本来就该走 FP32/INT，这不是问题。")),
        evidence=_evs(index, "ks__static_sass_share_tensor", "ks__static_sass_instructions",
                      "ks__static_sass_share_fp"),
        impact=("预期影响：这是**静态代码长相**，改 dtype/算子后 `ks__static_sass_share_tensor` "
                "会变；它不等于运行时 TC 利用率（那个量本机拿不到）。"),
        effort="low — 先核对 dtype/算子选择，可能不需要改代码",
        score=45,
    )


def _render_directions(out: list[str], rep: Report, kr: KernelReport,
                       index: dict[str, Metric], packet: dict[str, Any] | None) -> None:
    out.append(_DIRECTIONS_TITLE)
    out.append("")
    dirs_all = _directions(rep, kr, index, packet)
    dirs = dirs_all[:MAX_PRIORITIES]
    if not dirs:
        out.append("**没有有证据支撑的优化方向。** 本报告里没有任何一组读数足以支撑一条方向："
                   "要么这一维度没有读数（见 `## 2` 的拒绝码），要么读数之间还构不成判断。"
                   "这里**不给通用建议**（「提高并行度」「优化访存」这类话在任何 kernel 上都成立，"
                   "因此不传递任何信息），先按 `## 0` 的缺件清单补齐口径再谈方向。")
        out.append("")
        return
    for i, d in enumerate(dirs, start=1):
        out.append(f"### Priority {i} — {d.name}")
        out.append("")
        out.append(d.action)
        out.append("")
        out.append("**Evidence:**")
        for ev in d.evidence:
            out.append(f"- {ev}")
        out.append("")
        out.append(f"**Expected impact:** {d.impact}")
        out.append(f"**Effort:** {d.effort}")
        out.append("")
    if len(dirs_all) > MAX_PRIORITIES:
        out.append(f"（硬上限 {MAX_PRIORITIES} 条：其余方向未展开 —— 超过 {MAX_PRIORITIES} 条时"
                   f"排序不再传递信息，按上面顺序做即可。）")
        out.append("")


# --------------------------------------------------------------------------
# 5. Confidence & caveats
# --------------------------------------------------------------------------
def _render_confidence(out: list[str], index: dict[str, Metric],
                       packet: dict[str, Any] | None) -> None:
    """三段式：确定的 / 不确定的（每条都要说怎么解掉）/ 答不了的。

    为什么"不确定"必须自带解法：只说"这个数不确定"对决策没有用 —— 读者既不知道该不该
    用它，也不知道要花什么代价才能变确定。所以每条不确定都必须回答"做什么能让它确定"。
    """
    out.append("## 5. Confidence & caveats")
    out.append("")
    exact = [m for m in index.values() if m.provenance is Provenance.EXACT]
    out.append("What I'm sure about:")
    out.append("")
    if exact:
        for m in exact[:10]:
            out.append(f"- {_fmt_metric(m)}")
        if len(exact) > 10:
            out.append(f"- …另有 {len(exact) - 10} 条 `[exact]` 读数（来源相同：CUPTI activity / "
                       f"cubin / 解析算术）")
        out.append("- 以上是与 ncu 同源的 `[exact]` 读数：同机器同版本内可直接用于回归对比。")
    else:
        out.append("- 没有 `[exact]` 读数：这一份里没有一个值与 ncu 同源，"
                   "所有数字都依赖模型/采样/口径假设。")
    out.append("")
    soft = [m for m in index.values()
            if m.provenance in (Provenance.ESTIMATED, Provenance.MEASURED, Provenance.DERIVED)]
    out.append("What I'm uncertain about:")
    out.append("")
    soft_sorted = sorted(soft, key=_uncertainty_rank)
    if soft_sorted:
        for m in soft_sorted[:8]:
            out.append(f"- {_fmt_metric(m)}；要收紧它：{_resolve_hint(m)}")
        if len(soft_sorted) > 8:
            out.append(f"- …另有 {len(soft_sorted) - 8} 条非 exact 读数；"
                       f"它们的 `note`/`resolution` 字段已给出各自的误差来源（见 JSON 报告）")
    else:
        out.append("- 无：本报告的非 exact 读数为零")
    out.append("")
    unav = [m for m in index.values() if m.provenance is Provenance.UNAVAILABLE]
    out.append("Anything the profile couldn't answer that the user should know:")
    out.append("")
    groups = _group_unavailable(unav)
    if groups:
        for g in groups[:SETUP_UNAVAILABLE_CAP]:
            out.append(f"- {_code_ref(g['code'])}：{g['reason']}"
                       f"（受影响指标 {g['count']} 个，例："
                       f"{'、'.join('`' + e + '`' for e in g['examples'])}）")
        if len(groups) > SETUP_UNAVAILABLE_CAP:
            out.append(f"- …还有 {len(groups) - SETUP_UNAVAILABLE_CAP} 组（完整清单见 JSON 报告）")
    else:
        out.append("- 本报告没有记录任何拒绝条目：这**不等于**「什么都答得了」，"
                   "只等于这份报告没有记录（旧格式报告就是这样）")
    if packet:
        for e in packet.get("environment_refusals") or []:
            code = e.get("code")
            if code:
                out.append(f"- 环境缺件 {_code_ref(code)}：{_cell(e.get('message', ''))}"
                           f"（这是「这台机器没有」，不是「这次没测」）")
        for v in packet.get("verdicts") or []:
            if v.get("status") == "refused" and (v.get("refusal") or {}).get("code"):
                out.append(f"- 判词 `{v.get('id')}` 被拒："
                           f"{_code_ref((v.get('refusal') or {})['code'])} —— 那个数字不能当依据，"
                           f"方向也可能因此反转")
    out.append("")


def _uncertainty_rank(m: Metric) -> tuple[int, str]:
    """不确定项的排序：estimated（模型）> measured（采样/分辨率）> derived（推导）。

    模型量最不可信（它可能整体偏），所以排最前；这样读者看到的顺序就是"最该怀疑的先看"。
    """
    order = {Provenance.ESTIMATED: 0, Provenance.MEASURED: 1, Provenance.DERIVED: 2}
    return (order.get(m.provenance, 3), m.name)


def _resolve_hint(m: Metric) -> str:
    """一条不确定读数"怎么才能变确定"的具体动作（不是"多测几次"这种空话）。"""
    if m.resolution:
        return (f"它的分辨率下限是 {m.resolution}，读数在该粒度内不可分辨；"
                f"要收紧就得换粒度更细的口径（见该指标的 source 与对应拒绝码的 remedy）")
    if m.provenance is Provenance.ESTIMATED:
        return ("用不依赖模型的口径复核（例如访存量走 `--ks-memtrace` 轨迹口径，"
                "roofline 先重跑本机标定并确认 dtype/尺寸口径一致）；模型量不能靠重复采样变准")
    if m.provenance is Provenance.MEASURED:
        return ("同口径重复采集，并把两次的差与采样周期/分辨率比较：差小于分辨率时"
                "这个数不可分辨（不要当成变化）")
    return ("复核它的输入指标（推导链写在 note/source 里）：输入口径一变，这个数就要重算")


# --------------------------------------------------------------------------
# 6. Reproduction
# --------------------------------------------------------------------------
def _render_reproduction(out: list[str], rep: Report, packet: dict[str, Any] | None) -> None:
    cmd, source = _repro_command(rep, packet)
    out.append("## 6. Reproduction")
    out.append("")
    out.append("同口径重跑（只改代码，不改测量口径，才能把差值归因到代码）：")
    out.append("")
    out.append("```bash")
    out.append(cmd)
    out.append("```")
    out.append("")
    out.append(f"- 命令来源：{source}")
    if packet:
        proto = packet.get("protocol") or {}
        out.append(f"- 协议哈希：`{proto.get('hash', '未记录')}`"
                   f"（两次运行的哈希不同就是口径变了，差值不能归因到代码）")
    if rep.sources:
        out.append("- 数据源见 `## 0`；缺件清单也在那里（缺件会改变哪些数字是可比的）")
    out.append("")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def render_report_md(rep: Report, packet: dict[str, Any] | None = None, *,
                     kernel: str | None = None, title: str | None = None) -> str:
    """把一份 ``Report``（可选加上决策包）渲染成 per-kernel 的 Markdown 报告。

    参数：
        rep        `kernoscope.report.Report`。渲染只读它，不产生任何新数字。
        packet     `packet.build(rep)` 的决策包（可为 None）。给了它，"One-line read"、
                   `## 3` 的判词与 `## 4` 的路径都从包里读 —— 保证人视图与机器视图
                   是同一份证据的两副面孔，而不是两套各自解释的数字。
        kernel     主体 kernel 名（精确名或子串）。默认取总 GPU 时间最大的那个；
                   给了一个不存在的名字会报错（静默换主体读者会看错对象）。
        title      覆盖一级标题。默认是 `# <kernel> Profiling Report`；模板名很长时
                    调用方可以用更短的标题（标题不承载数据，只影响可读性）。

    返回：Markdown 文本（无 ANSI/框线装饰，可直接提交进仓库）。
    """
    kr = _select_kernel(rep, kernel)
    index = _metric_index(rep, kr)
    if packet is not None and not isinstance(packet, dict):  # 防御：错误类型要立刻可见
        raise TypeError(f"packet must be a dict from packet.build(), got {type(packet).__name__}")

    heading = _one_line(title) if title else f"`{kr.name}` Profiling Report"
    if not heading.startswith("#"):
        heading = f"# {heading}"
    out: list[str] = [heading, ""]

    _render_header(out, rep, kr)
    _render_setup(out, rep, packet)
    _render_headline(out, rep, kr, index, packet)
    _render_dimensions(out, index)
    _render_diagnosis(out, rep, kr, index, packet)
    _render_directions(out, rep, kr, index, packet)
    _render_confidence(out, index, packet)
    _render_reproduction(out, rep, packet)

    # 末尾统一裁剪：模板要求"一行一个条目"，多余空行会让 diff 噪声变大。
    text = "\n".join(out).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)


def _render_header(out: list[str], rep: Report, kr: KernelReport) -> None:
    """头部元数据块：读者第一眼要看到"哪个 kernel、哪台机器、哪个工具、口径怎么读"。

    `**Caliber/metric-name caveats:**` 不是免责声明，而是**读法**：指标名是我们自己的
    命名空间，`est__`/`model__` 是模型推算、`meas__` 是实测、`ks__` 是我们自有的量，
    `ncu__*` 是从别处导入的 `.ncu-rep` 原值。不写清这一点，读者会把 `est__` 的数当成
    与 ncu 同源而直接跨机器比较 —— 那正是本项目存在的原因。
    """
    out.append(f"**Kernel:** `{kr.name}` — {kr.launch_count} 次 launch"
               + (f"；nsys 短名 `{kr.alias}`" if kr.alias else "")
               + ("；**有多组 launch 配置**（单一取值会误导，逐组见 `2.1`）"
                  if kr.is_mixed_config else ""))
    d = rep.device
    gpu_bits: list[str] = []
    if d.name:
        gpu_bits.append(f"`{d.name}`")
    if d.compute_capability:
        gpu_bits.append(f"compute capability {d.compute_capability}"
                        f"（`sm_{str(d.compute_capability).replace('.', '')}`）")
    if d.sm_count:
        gpu_bits.append(f"{d.sm_count} SM")
    if d.l2_mib:
        gpu_bits.append(f"L2 {d.l2_mib:g} MiB")
    if d.driver:
        gpu_bits.append(f"driver {d.driver}")
    out.append(f"**Target GPU:** {' / '.join(gpu_bits)}" if gpu_bits
               else "**Target GPU:** 未知（报告里没有设备信息）")
    out.append(f"**Tool:** kernoscope {rep.tool_version}"
               + (f"（报告由 {__version__} 渲染）" if rep.tool_version != __version__ else ""))
    out.append("**Caliber/metric-name caveats:** 指标名是**本工具自己的命名空间**，"
               "不是 ncu 指标名的逐字对应；`meas__` = 本机实测读数（NVML 管理计数器/插桩探针），"
               "`est__` / `model__` = 模型或采样推算，`ks__` = 由实测/精确量再算出的统计量与判词，"
               "`ncu__*` = 从别处采集的 `.ncu-rep` 导入的原值（与 ncu 同源）。"
               "任何非 `[exact]` 的值都不能当精确测量或跨机器比较；"
               "`n/a` 表示拒绝给数，括号里是结构化拒绝码（可编程、可查 remedy）。")
    out.append("")
