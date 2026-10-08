"""section 组装：把各后端数据变成 ncu 命名的 section + 出处标签。

这里是"接口兼容、语义自证"的落点：
  * 与 ncu 同源可得的数据（CUPTI activity / cubin / 解析算术）→ 用 ncu 的 section 名与指标名，标 exact
  * 我们推导或实测的数据 → 用 `est__` / `meas__` / 自有 `ks__` 前缀，标 derived/measured/estimated
  * 需要硬件计数器的 section → 生成 `unavailable` 条目，附原因字符串，**永不填数字**
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .analyze.occupancy import arch_for, theoretical_occupancy
from .analyze.roofline import MachinePeaks, SpeedOfLight
from .provenance import Metric, Provenance, Section, UNAVAILABLE_REASONS

# `--set` 的分级（对齐 ncu --list-sets）
SET_SECTIONS: dict[str, list[str]] = {
    "basic": ["Duration", "LaunchStats", "Occupancy", "SpeedOfLight", "DramTraffic",
              "WorkloadDistribution"],
    "detailed": ["Duration", "LaunchStats", "Occupancy", "SpeedOfLight", "DramTraffic",
                 "WorkloadDistribution",
                 "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis", "SourceCounters"],
    "full": ["Duration", "LaunchStats", "Occupancy", "SpeedOfLight", "DramTraffic",
             "WorkloadDistribution",
             "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables",
             "InstructionStats", "SchedulerStats", "WarpStateStats", "SourceCounters",
             "PmSampling", "Nvlink_Tables", "NumaAffinity"],
}

# 需要硬件计数器才能给出的 section 及原因
COUNTER_SECTIONS: dict[str, str] = {
    "ComputeWorkloadAnalysis": UNAVAILABLE_REASONS["counters"],
    "MemoryWorkloadAnalysis": UNAVAILABLE_REASONS["counters"],
    "MemoryWorkloadAnalysis_Tables": UNAVAILABLE_REASONS["counters"],
    "SchedulerStats": UNAVAILABLE_REASONS["counters"],
    "WarpStateStats": UNAVAILABLE_REASONS["counters"],
    "SourceCounters": UNAVAILABLE_REASONS["sampling"],
    "PmSampling": UNAVAILABLE_REASONS["counters"],
    "Nvlink_Tables": "requires NVLink hardware with NVML counter support (not present / not supported)",
    "NumaAffinity": "not applicable on this device",
    "WorkloadDistribution": UNAVAILABLE_REASONS["instrumentation"],
}

# 这些 ncu 指标名"我们知道自己给不出"，显式登记以便 --metrics/--strict 处理
KNOWN_COUNTER_METRICS: dict[str, str] = {
    "sm__maximum_warps_per_active_cycle_pct": UNAVAILABLE_REASONS["counters"],
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": UNAVAILABLE_REASONS["counters"],
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": UNAVAILABLE_REASONS["counters"],
    "l1tex__t_sector_hit_rate.pct": UNAVAILABLE_REASONS["counters"],
    "lts__t_sector_hit_rate.pct": UNAVAILABLE_REASONS["counters"],
    "dram__bytes_read.sum": UNAVAILABLE_REASONS["counters"],
    "dram__bytes_write.sum": UNAVAILABLE_REASONS["counters"],
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": UNAVAILABLE_REASONS["counters"],
    "smsp__inst_executed.sum": UNAVAILABLE_REASONS["counters"],
}


@dataclass
class BuildOptions:
    set_name: str = "basic"
    sections: list[str] = field(default_factory=list)
    peaks: MachinePeaks | None = None
    workload: tuple[float, float] | None = None          # (flops, bytes) 来自用户输入
    workload_note: str = ""
    achieved_occupancy: dict[str, dict[str, float]] = field(default_factory=dict)
    cubin_res: dict[str, dict[str, int]] = field(default_factory=dict)
    cubin_sass: dict[str, Any] | None = None
    nvml_summary: dict[str, float] = field(default_factory=dict)
    dram: Any = None      # analyze.dram.DramEstimate（避免循环 import）
    sm_count: int = 0
    device_name: str = ""
    arch: str = "sm_89"
    peak_gbs: float = 0.0
    schema_extra: dict[str, Any] = field(default_factory=dict)


def requested_sections(opt: BuildOptions) -> list[str]:
    if opt.sections:
        return list(opt.sections)
    return list(SET_SECTIONS.get(opt.set_name, SET_SECTIONS["basic"]))


def build_device_kernel(opt: BuildOptions) -> Section:
    """设备级 section（伪 kernel "(device)"）。"""
    s = Section("DeviceStatistics")
    n = opt.nvml_summary
    if n:
        if "mem_util_mean" in n:
            s.add(Metric.estimated(
                "meas__dram_busy_proxy_pct", round(n["mem_util_mean"], 1), "%",
                source="NVML utilization.memory (200ms interval average)",
                note="区间平均；必须让 GPU 保持饱和，否则会被宿主间隙稀释（实测 96.8% vs 6.3%）",
            ))
            if opt.peak_gbs:
                s.add(Metric.estimated(
                    "meas__dram_busy_proxy_gbs", round(n["mem_util_mean"] / 100.0 * opt.peak_gbs, 1),
                    "GB/s",
                    source=f"NVML mem_util x measured peak {opt.peak_gbs:.0f} GB/s",
                    note="稳态代理，实测误差约 6%",
                ))
        for key, name, unit in (("sm_clock_mean", "meas__sm_clock_mean_mhz", "MHz"),
                                ("sm_clock_max", "meas__sm_clock_max_mhz", "MHz"),
                                ("power_max", "meas__power_max_w", "W")):
            if key in n:
                s.add(Metric.measured(name, round(n[key], 1), unit, source="nvidia-smi / NVML"))
        if n.get("energy_j"):
            s.add(Metric.measured("meas__device_energy_j", round(n["energy_j"], 3), "J",
                                  source="NVML total energy accumulator",
                                  note="ncu 不提供能耗指标"))
    if not s.metrics:
        s.add(Metric.unavailable("meas__dram_busy_proxy_pct",
                                 "NVML sampling disabled or unavailable (--ks-no-nvml?)"))
    return s


def build_dram_section(opt: BuildOptions) -> Section | None:
    """DRAM 流量估算 section（对标 ncu 的 MemoryWorkloadAnalysis / dram__bytes）。

    诚实性处理：占空比过低或缺少实测峰值带宽时，**不给数字**，改为 unavailable + 原因 ——
    否则会给出一个被空闲稀释、系统性偏小的值（实测同一拷贝 96.8% vs 6.3%）。
    """
    est = opt.dram
    if est is None:
        return None
    s = Section("DramTraffic")
    if not est.reliable or est.bytes_total <= 0:
        reason = "；".join(est.notes) or "无法估算（缺少采样或峰值带宽）"
        s.add(Metric.unavailable("est__dram_bytes_total", reason))
        return s
    from .analyze.dram import to_metrics

    for name, value, unit, source, note in to_metrics(est):
        if value is None:
            continue
        # per-launch / 吞吐占比需要"窗口被该 kernel 主导"，否则只能给 n/a
        if name in ("est__dram_vs_ledger_err_pct", "est__dram_range_position_pct") and value is None:
            s.add(Metric.unavailable(
                name, "账本对拍的口径不成立：per-launch 需要「单一 kernel 主导窗口 + 占空比 ≥80%」，"
                      f"当前占空比 {est.duty_pct:.0f}%、kernel 数 {'1' if est.attributable else '不确定'}"
                      "；请用 -k 锁定单个 kernel 并让它跑满窗口"))
            continue
        if name == "est__dram_bytes_per_launch" and not est.attributable:
            s.add(Metric.unavailable(
                name,
                f"占空比 {est.duty_pct:.0f}% 或未锁定单一 kernel：无法把 {est.bytes_total / 1e6:.0f} MB "
                f"总量归因到每个 launch（请用 -k 选单个 kernel，并让它在紧循环里跑满）"))
            continue
        s.add(Metric.estimated(name, value, unit, source=source, note=note))
    return s


def build_kernel_sections(kr, opt: BuildOptions) -> list[Section]:
    """为一个聚合后的 kernel 生成 section 列表。"""
    out: list[Section] = []
    want = set(requested_sections(opt))
    launch = kr.launches[0] if kr.launches else None
    if launch is None:
        return out

    # ---------------- Duration ----------------
    if "Duration" in want:
        s = Section("Duration")
        s.add(Metric.exact("gpu__time_duration.sum", round(kr.total_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps (nsys sqlite)"))
        s.add(Metric.exact("ks__launch_time_mean", round(kr.mean_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps"))
        s.add(Metric.exact("ks__launch_time_min", round(kr.min_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps"))
        s.add(Metric.exact("ks__launch_time_max", round(kr.max_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps"))
        s.add(Metric.exact("ks__launch_count", kr.launch_count, "", source="CUPTI activity"))
        out.append(s)

    # ---------------- LaunchStats ----------------
    if "LaunchStats" in want:
        s = Section("LaunchStats")
        gx, gy, gz = launch.grid
        bx, by, bz = launch.block
        threads = bx * by * bz
        s.add(Metric.exact("launch__grid_size", gx * gy * gz, "", source="CUPTI activity"))
        s.add(Metric.exact("launch__block_size", threads, "", source="CUPTI activity"))
        s.add(Metric.exact("launch__thread_count", threads * gx * gy * gz, "",
                           source="CUPTI activity"))
        s.add(Metric.exact("launch__registers_per_thread", launch.registers_per_thread, "",
                           source="CUPTI activity (nsys)"))
        # 交叉校验 + 决定 occupancy 用哪个值
        res_for_regs = _lookup_in(opt.cubin_res, kr)
        regs_effective = launch.registers_per_thread
        if res_for_regs and res_for_regs.get("REG"):
            regs_effective = int(res_for_regs["REG"])
            agree = regs_effective == launch.registers_per_thread
            s.add(Metric.exact(
                "ks__cubin_registers_per_thread", regs_effective, "",
                source="cuobjdump -res-usage (ptxas static)",
                note=("与 CUPTI 一致" if agree else
                      f"与 CUPTI 报的 {launch.registers_per_thread} 不一致；"
                      f"occupancy 采用 cubin 值（ptxas 静态真值）"),
            ))
            if res_for_regs.get("SHARED"):
                s.add(Metric.exact("ks__cubin_shared_bytes", res_for_regs["SHARED"], "B",
                                   source="cuobjdump -res-usage"))
        s.add(Metric.exact("ks__registers_used_for_occupancy", regs_effective, "",
                           source=("cubin (ptxas static)" if regs_effective != launch.registers_per_thread
                                   else "CUPTI activity")))
        s.add(Metric.exact("launch__shared_mem_per_block_static", launch.static_smem_bytes, "B",
                           source="CUPTI activity"))
        s.add(Metric.exact("launch__shared_mem_per_block_dynamic", launch.dynamic_smem_bytes, "B",
                           source="CUPTI activity"))
        s.add(Metric.exact("launch__local_mem_per_thread", launch.local_mem_per_thread, "B",
                           source="CUPTI activity"))
        if launch.is_cuda_graph:
            s.add(Metric.exact("ks__cuda_graph_capture", "yes", "", source="CUPTI gridId != 0"))
        out.append(s)

    # ---------------- Occupancy ----------------
    if "Occupancy" in want:
        s = Section("Occupancy")
        res_occ = _lookup_in(opt.cubin_res, kr)
        regs_occ = int(res_occ["REG"]) if (res_occ and res_occ.get("REG")) \
            else launch.registers_per_thread
        smem_static_occ = launch.static_smem_bytes
        if res_occ and res_occ.get("SHARED"):
            smem_static_occ = max(smem_static_occ, int(res_occ["SHARED"]))
        occ = theoretical_occupancy(regs_occ, smem_static_occ,
                                    launch.dynamic_smem_bytes, launch.threads_per_block, opt.arch)
        s.add(Metric.exact("ks__theoretical_occupancy_pct", round(occ.occupancy_pct, 1), "%",
                           source=f"arithmetic from CUPTI stats + {opt.arch} limits"))
        s.add(Metric.exact("ks__blocks_per_sm", occ.blocks_per_sm, "",
                           source="arithmetic (theoretical)"))
        s.add(Metric.exact("ks__warps_per_sm", occ.warps_per_sm, "",
                           source="arithmetic (theoretical)"))
        s.add(Metric.exact("ks__occupancy_limiter", occ.limiter, "",
                           source="arithmetic (theoretical)",
                           note="并列的限制因素用 + 连接"))
        # ncu 的 achieved occupancy：计数器实现 → 我们只能标 unavailable 或给出 meas__
        ach = _lookup_in(opt.achieved_occupancy, kr)
        if ach:
            s.add(Metric.measured(
                "meas__achieved_occupancy_pct", round(ach.get("occupancy_pct", 0.0), 1), "%",
                source="interval-overlap reconstruction from (smid, globaltimer) probes",
                note=f"实测并发 {ach.get('avg_concurrent_blocks', 0):.2f} blocks/SM "
                     f"(峰值 {int(ach.get('peak_concurrent_blocks', 0))})；需要 kernel 内有探针",
            ))
        s.add(Metric.unavailable(
            "sm__maximum_warps_per_active_cycle_pct", UNAVAILABLE_REASONS["counters"],
            source="ncu computes this from hardware counters",
        ))
        out.append(s)

    # ---------------- SpeedOfLight ----------------
    if "SpeedOfLight" in want:
        s = Section("SpeedOfLight")
        if opt.workload and opt.peaks:
            flops, bytes_ = opt.workload
            sol = SpeedOfLight(float(kr.mean_ns), flops, bytes_, opt.peaks)
            s.add(Metric.estimated("est__achieved_tflops", round(sol.achieved_tflops, 2), "TFLOP/s",
                                   source=f"user-provided workload / CUPTI duration",
                                   note=opt.workload_note))
            s.add(Metric.estimated("est__achieved_gbs", round(sol.achieved_gbs, 1), "GB/s",
                                   source="user-provided bytes / CUPTI duration"))
            s.add(Metric.estimated("est__pct_of_measured_compute_peak", round(sol.pct_compute, 1), "%",
                                   source=f"measured peak {opt.peaks.compute_tflops:.1f} TFLOP/s "
                                          f"({opt.peaks.dtype_key})"))
            s.add(Metric.estimated("est__pct_of_measured_bw_peak", round(sol.pct_memory, 1), "%",
                                   source=f"measured peak {opt.peaks.bandwidth_gbs:.0f} GB/s"))
            s.add(Metric.estimated("est__roofline_ideal_ns", round(max(sol.ideal_ns_compute,
                                                                      sol.ideal_ns_memory), 1), "ns",
                                   source="roofline with measured peaks"))
            s.add(Metric.estimated("est__bound", sol.bound, "",
                                   source="compute vs memory roof"))
            s.add(Metric.estimated("est__roofline_efficiency_pct", round(sol.efficiency_pct, 1), "%",
                                   source="ideal / measured"))
            s.add(Metric.estimated("est__arithmetic_intensity", round(sol.arithmetic_intensity, 2),
                                   "FLOP/B", source="user-provided workload"))
        else:
            s.add(Metric.unavailable(
                "est__achieved_tflops",
                "workload not provided: pass --ks-gemm M,N,K or --ks-flops/--ks-bytes "
                "(plus --ks-calibration machine-profile.json for measured peaks)",
            ))
        s.add(Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed",
                                 UNAVAILABLE_REASONS["counters"]))
        out.append(s)

    # ---------------- InstructionStats（静态，来自 cubin）----------------
    if "InstructionStats" in want:
        s = Section("InstructionStats")
        if opt.cubin_sass:
            mix = opt.cubin_sass
            s.add(Metric.exact("ks__static_sass_instructions", int(mix["total"]), "",
                               source="cuobjdump -sass (static)"))
            for cat, pct in mix["categories_pct"].items():
                s.add(Metric.exact(f"ks__static_sass_share_{cat}", round(pct, 1), "%",
                                   source="cuobjdump -sass (static)"))
        else:
            s.add(Metric.unavailable("ks__static_sass_instructions",
                                     "no cubin provided: pass --ks-cubin <file.so|.cubin>"))
        s.add(Metric.unavailable("smsp__inst_executed.sum", UNAVAILABLE_REASONS["counters"],
                                 source="dynamic instruction count needs counters; "
                                        "static mix from cubin is provided instead"))
        out.append(s)

    # ---------------- 明确不可得的 section ----------------
    for name in requested_sections(opt):
        reason = COUNTER_SECTIONS.get(name)
        if reason:
            sec = Section(name)
            sec.add(Metric.unavailable(f"ks__{name.lower()}_status", reason))
            out.append(sec)

    return out


# token 匹配时忽略这些"到处都有"的词，避免把 Kernel2 匹配到 Kernel3
_STOP_TOKENS = {
    "void", "int", "float", "double", "const", "long", "short", "char", "bool",
    "unsigned", "signed", "Params", "Kernel", "cuda", "templates", "native",
}


def _tokens(s: str) -> set[str]:
    """生成可比较的标识 token 集合。

    mangled 符号里标识符与类型编码是连在一起的（`_Z4addkPfi` 里 `addk` 和 `Pfi` 之间
    没有分隔符），所以除了整段字母串，还要生成**长度 >= 4 的前缀**，才能与 demangled
    的 `addk` 相交。前缀若以停用词开头就丢弃 —— 这既挡掉 `void`/`float`，
    也让 `Kernel2` 与 `Kernel3` 不会因为共有 `Kernel` 前缀而误配。
    """
    out: set[str] = set()
    for run in re.findall(r"[A-Za-z]{3,}", s or ""):
        for i in range(4, len(run) + 1):
            tok = run[:i]
            # 双向前缀判断：把 "Kern"/"Kerne" 这类停用词的前缀也丢掉
            if any(tok.startswith(st) or st.startswith(tok) for st in _STOP_TOKENS):
                continue
            out.add(tok)
    return out


def _lookup_in(d: dict, kr) -> object | None:
    """在 cubin/探针输出的字典里找这个 kernel。

    键可能来自 cuobjdump（mangled，如 `_Z4addkPfi`）、探针（用户写简单名）或 nsys 短名，
    而 kr.name 默认是 demangled 全签名（如 `addk(float *, int)`）—— 精确匹配注定失败，
    所以退化为 **token 交集** 匹配，并取"最长公共 token"作为最优候选。
    """
    for k in (kr.name, getattr(kr, "alias", "")):
        if k and k in d:
            return d[k]
    kr_tokens = _tokens(kr.name) | _tokens(getattr(kr, "alias", ""))
    kr_flat = f"{kr.name} {getattr(kr, 'alias', '')}".lower()
    best: object | None = None
    best_score = 0
    for k, v in d.items():
        if not k:
            continue
        # 子串回退：简单名（如探针里的 "k_occ"）与 demangled 签名互相包含时可直接匹配
        kl = str(k).lower()
        if len(kl) >= 3 and (kl in kr_flat or kr_flat.strip() in kl):
            if len(kl) > best_score:
                best, best_score = v, len(kl)
            continue
        inter = _tokens(k) & kr_tokens
        if not inter:
            continue
        score = max(len(t) for t in inter)
        if score > best_score:
            best, best_score = v, score
    return best


def _strip(name: str) -> str:
    from .backends.cubin import strip_mangled

    return strip_mangled(name)


def provenance_summary(report) -> dict[str, int]:
    return report.provenance_histogram()
