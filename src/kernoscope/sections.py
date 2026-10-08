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

from .analyze.occupancy import theoretical_occupancy
from .analyze.roofline import MachinePeaks, SpeedOfLight
from .provenance import Metric, Section, UNAVAILABLE_REASONS

# `--set` 的分级（对齐 ncu --list-sets）
SET_SECTIONS: dict[str, list[str]] = {
    "basic": ["Duration", "LaunchStats", "LaunchConfigs", "Occupancy", "SpeedOfLight", "DramTraffic",
              "MemoryTrace"],
    "detailed": ["Duration", "LaunchStats", "LaunchConfigs", "Occupancy", "SpeedOfLight", "DramTraffic",
                 "MemoryTrace",
                 "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis", "SourceCounters"],
    "full": ["Duration", "LaunchStats", "LaunchConfigs", "Occupancy", "SpeedOfLight", "DramTraffic",
             "MemoryTrace", "WorkloadDistribution",
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
    memtrace: Any = None  # {kernel: analyze.memtrace.MemTraceStats}
    issue_hist: Any = None  # backends.opcode_hist.OpcodeHist（NVBit 官方 opcode_hist 输出）
    clock_mhz: float = 0.0  # 用于发射槽上限的 SM 时钟（NVML 实测或用户给定）
    sm_count: int = 0
    device_name: str = ""
    arch: str = "sm_89"
    peak_gbs: float = 0.0
    peak_source: str = ""          # R-06：峰值从哪个标定文件的哪次测量来
    ledger_basis: str = "auto"     # R-33：账本对拍口径 auto|nvml|trace
    schema_extra: dict[str, Any] = field(default_factory=dict)


def requested_sections(opt: BuildOptions) -> list[str]:
    if opt.sections:
        return list(opt.sections)
    return list(SET_SECTIONS.get(opt.set_name, SET_SECTIONS["basic"]))


def build_device_kernel(opt: BuildOptions) -> Section:
    """设备级 section（伪 kernel "(device)"）。

    命名纪律（R-12）：NVML 的原始读数（时钟/功耗/能量/显存忙碌）是**实测**，
    用 `meas__` + measured；由它们再算出来的量（等效带宽）是 estimated，用 `est__`。
    """
    s = Section("DeviceStatistics")
    n = opt.nvml_summary
    if n:
        if "mem_util_mean" in n:
            # R-31：把 NVML 自身的分辨率与采样周期随数字一起报出来。
            s.add(Metric.measured(
                "meas__nvml_mem_busy_mean_pct", round(n["mem_util_mean"], 1), "%",
                source="NVML utilization.memory（显存子系统忙碌时间占比，管理计数器，免 admin）",
                note=("区间平均；是「显存子系统忙碌占比」，不是 DRAM 吞吐本身。"
                      "读数 0.0% 只说明低于分辨率下限，不等于「DRAM 完全没动」；"
                      "必须让 GPU 保持饱和，否则会被宿主间隙稀释（实测 96.8% vs 6.3%）"),
                resolution="1 %（整数百分比）",
            ))
            if opt.peak_gbs:
                s.add(Metric.estimated(
                    "est__dram_busy_equivalent_gbs",
                    round(n["mem_util_mean"] / 100.0 * opt.peak_gbs, 1), "GB/s",
                    source=f"NVML mem_util × 本机实测峰值 {opt.peak_gbs:.0f} GB/s"
                           + (f"（{opt.peak_source}）" if opt.peak_source else ""),
                    note="等效带宽口径；与按字节/时间的口径略有差异", resolution="1 %",
                ))
        for key, name, unit in (("sm_clock_mean", "meas__sm_clock_mean_mhz", "MHz"),
                                ("sm_clock_max", "meas__sm_clock_max_mhz", "MHz"),
                                ("power_max", "meas__power_max_w", "W")):
            if key in n:
                s.add(Metric.measured(name, round(n[key], 1), unit,
                                      source="nvidia-smi / NVML（管理计数器，免 admin）",
                                      resolution="1 MHz / 1 W（驱动报数粒度）"))
        if n.get("energy_j"):
            s.add(Metric.measured("meas__device_energy_j", round(n["energy_j"], 3), "J",
                                  source="NVML total energy accumulator",
                                  note="ncu 不提供能耗指标", resolution="1 mJ"))
    if not s.metrics:
        s.add(Metric.unavailable("meas__nvml_mem_busy_mean_pct",
                                 "没有 NVML 采样点：采样被关闭（--ks-no-nvml）或 nvidia-smi/pynvml "
                                 "不可用；没有它就无法给出显存忙碌代理",
                                 source="NVML"))
    return s


def build_dram_section(opt: BuildOptions) -> Section | None:
    """DRAM 流量估算 section（对标 ncu 的 MemoryWorkloadAnalysis / dram__bytes）。

    诚实性处理：任一口径前提不成立就**不给数字**，改为 unavailable + 原因 ——
    但那一行必须**存在**（B1：口径不成立时"拒绝"退化成"消失"是不可接受的），
    且原因必须点明哪个前提没成立（R-32），不是复述算法定义。
    """
    est = opt.dram
    if est is None:
        return None
    s = Section("DramTraffic")
    from .analyze.dram import to_metrics

    for name, value, unit, source, note, resolution in to_metrics(est, opt.ledger_basis):
        if value is None:
            s.add(Metric.unavailable(name, est.reason_for(name, opt.ledger_basis),
                                     source=source, resolution=resolution))
        elif name == "ks__ledger_range_verdict":
            s.add(Metric.derived(name, value, unit, source=source, note=note,
                                 resolution=resolution))
        elif name.startswith("ks__"):
            # 由实测样本再算出来的统计量/判词：derived（不是"实测读数"，R-12）
            s.add(Metric.derived(name, value, unit, source=source, note=note,
                                 resolution=resolution))
        else:
            s.add(Metric.estimated(name, value, unit, source=source, note=note,
                                   resolution=resolution))
    return s


def build_memtrace_section(st, opt: BuildOptions) -> Section | None:
    """访存轨迹 section（NVBit 插桩口径）。

    provenance 规则：全量轨迹 → `meas__` + measured；**按采样比放大**的轨迹不是"实测读数"
    （样本代表性是假设），此时前缀改为 `est__` + estimated（R-12：名字承诺什么，
    出处就得是什么）。前缀本身由 R3 强制。
    """
    from .analyze.memtrace import to_metrics

    if st is None:
        return None
    s = Section("MemoryTrace")
    sampled = st.sampling_ratio < 0.999
    for name, value, unit, source, note in to_metrics(st):
        if value is None:
            continue
        if name.startswith("meas__") and sampled:
            name = "est__" + name[len("meas__"):]
            note = f"{note}；**按 {st.sampling_ratio * 100:.1f}% 采样放大 → estimated**"
        if name.startswith("meas__"):
            s.add(Metric.measured(name, value, unit, source=source, note=note))
        else:
            s.add(Metric.estimated(name, value, unit, source=source, note=note))
    if st.dram_bytes_per_launch is None:
        s.add(Metric.unavailable("est__dram_bytes_per_launch_trace",
                                 st.dram_bytes_unavailable_reason or "轨迹不足以给出 DRAM 字节"))
    # 两条独立路径的交叉校验（NVML 代理 vs 轨迹模型）
    if (st.dram_bytes_per_launch is not None and opt.dram is not None
            and getattr(opt.dram, "bytes_per_launch", None)):
        ratio = 100.0 * st.dram_bytes_per_launch / opt.dram.bytes_per_launch
        s.add(Metric.derived(
            "ks__dram_trace_vs_nvml_pct", round(ratio, 1), "%",
            "轨迹模型（L2 miss×32B）/ NVML 代理（显存忙碌×峰值带宽）",
            note=("两条独立路径的比值；接近 100% 说明结论稳，"
                  "偏差大要查采样代表性或 L2 模型参数")))
    return s


def build_kernel_sections(kr, opt: BuildOptions) -> list[Section]:
    """为一个聚合后的 kernel 生成 section 列表。"""
    out: list[Section] = []
    want = set(requested_sections(opt))
    # B4：同一 kernel 可能以多组 launch 配置运行（形状扫描）。聚合行的单一
    # launch__grid_size 会陈述一个"多数 launch 不成立"的事实，所以混合时改为 n/a + 取值集合，
    # 并在 LaunchConfigs section 里逐组给出时间与 occupancy。
    configs = kr.configs
    mixed = len(configs) > 1
    dominant = configs[0] if configs else None
    launch = _launch_for_config(kr, dominant.key) if dominant is not None else None
    if launch is None:
        launch = kr.launches[0] if kr.launches else None
    if launch is None:
        return out

    # ---------------- Duration ----------------
    if "Duration" in want:
        s = Section("Duration")
        s.add(Metric.exact("gpu__time_duration.sum", round(kr.total_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps (nsys sqlite)"))
        s.add(Metric.exact("ks__launch_time_mean", round(kr.mean_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps",
                           note=("**多组 launch 配置的混合均值**，不可当作某一组配置的读数"
                                 if mixed else "")))
        s.add(Metric.exact("ks__launch_time_min", round(kr.min_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps"))
        s.add(Metric.exact("ks__launch_time_max", round(kr.max_ns / 1e6, 4), "ms",
                           source="CUPTI activity timestamps"))
        s.add(Metric.exact("ks__launch_count", kr.launch_count, "", source="CUPTI activity"))
        if mixed:
            s.add(Metric.exact("ks__launch_config_count", len(configs), "",
                               source="distinct (grid, block, smem) tuples in CUPTI activity",
                               note="逐组读数见 LaunchConfigs section"))
        out.append(s)

    # ---------------- LaunchStats ----------------
    if "LaunchStats" in want:
        s = Section("LaunchStats")
        gx, gy, gz = launch.grid
        bx, by, bz = launch.block
        threads = bx * by * bz
        if mixed:
            grids = sorted({c.grid[0] * c.grid[1] * c.grid[2] for c in configs})
            blocks = sorted({c.block[0] * c.block[1] * c.block[2] for c in configs})
            s.add(Metric.unavailable(
                "launch__grid_size",
                f"该 kernel 有 {len(configs)} 组不同 launch 配置（grid="
                f"{'/'.join(str(g) for g in grids)}）：单一数值会误导 2/3 的 launch，"
                f"故报 n/a；逐组读数见 LaunchConfigs section"))
            s.add(Metric.unavailable(
                "launch__block_size",
                f"该 kernel 有 {len(configs)} 组不同 launch 配置（block="
                f"{'/'.join(str(b) for b in blocks)}）：单一数值会误导，故报 n/a"))
            s.add(Metric.exact("ks__grid_sizes_observed", "|".join(str(g) for g in grids), "",
                               source="distinct grid sizes in CUPTI activity",
                               note="取值集合口径；逐组时长与 occupancy 见 LaunchConfigs section"))
            s.add(Metric.exact("ks__block_sizes_observed", "|".join(str(b) for b in blocks), "",
                               source="distinct block sizes in CUPTI activity"))
        else:
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
        if mixed and dominant is not None:
            s.add(Metric.exact("ks__dominant_config", dominant.key, "",
                               source="launch 数最多的那组配置",
                               note=f"{dominant.count}/{kr.launch_count} 次 launch；"
                                    f"本节未标注逐组的数值均取这组配置"))
        out.append(s)

    # ---------------- LaunchConfigs（逐组配置，R-04/B4）----------------
    if "LaunchConfigs" in want and mixed:
        sec = Section("LaunchConfigs")
        for c in configs:
            occ_c = _occupancy_for(c, opt)
            sec.add(Metric.exact(f"ks__cfg_{c.key}__launch_time_mean_ms",
                                 round(c.mean_ns / 1e6, 4), "ms",
                                 source="CUPTI activity timestamps（按组筛选）"))
            sec.add(Metric.exact(f"ks__cfg_{c.key}__launch_count", c.count, "",
                                 source="CUPTI activity"))
            if occ_c is not None:
                sec.add(Metric.exact(f"ks__cfg_{c.key}__theoretical_occupancy_pct",
                                     round(occ_c.occupancy_pct, 1), "%",
                                     source=f"arithmetic from CUPTI stats + {opt.arch} limits "
                                            f"(limiter={occ_c.limiter})"))
        out.append(sec)

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
                           note=("并列的限制因素用 + 连接"
                                 + (f"；本节按主导配置 {dominant.key} 计算，逐组见 LaunchConfigs"
                                    if mixed and dominant is not None else ""))))
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
            peak_src = f"measured peak ({opt.peak_source})" if opt.peak_source else "measured peak"
            caliber_ok = getattr(opt.peaks, "caliber_ok", True)
            caliber_note = getattr(opt.peaks, "caliber_note", "")
            s.add(Metric.estimated("est__achieved_gbs", round(sol.achieved_gbs, 1), "GB/s",
                                   source="user-provided bytes / CUPTI duration"))
            s.add(Metric.estimated("est__pct_of_measured_bw_peak", round(sol.pct_memory, 1), "%",
                                   source=f"{peak_src} {opt.peaks.bandwidth_gbs:.0f} GB/s"))
            if caliber_ok and opt.peaks.compute_tflops > 0:
                s.add(Metric.estimated("est__achieved_tflops", round(sol.achieved_tflops, 2),
                                       "TFLOP/s", source=peak_src.replace("measured peak",
                                                                          "user-provided workload / CUPTI duration, measured peak"),
                                       note=opt.workload_note))
                s.add(Metric.estimated("est__pct_of_measured_compute_peak", round(sol.pct_compute, 1),
                                       "%", source=f"{peak_src} {opt.peaks.compute_tflops:.1f} TFLOP/s "
                                                   f"({opt.peaks.dtype_key})"))
                s.add(Metric.estimated("est__roofline_ideal_ns", round(max(sol.ideal_ns_compute,
                                                                           sol.ideal_ns_memory), 1),
                                       "ns", source="roofline with measured peaks"))
                s.add(Metric.estimated("est__bound", sol.bound, "",
                                       source="compute vs memory roof",
                                       note=caliber_note))
                s.add(Metric.estimated("est__roofline_efficiency_pct", round(sol.efficiency_pct, 1),
                                       "%", source="ideal / measured"))
                if getattr(opt.peaks, "caliber_spread_pct", None):
                    s.add(Metric.estimated(
                        "ks__calibration_caliber_spread_pct", opt.peaks.caliber_spread_pct, "%",
                        source="machine-profile.py 的三种累加口径设置之差",
                        note=("口径可分辨时，roofline 在拐点附近的判定会随设置翻转 —— "
                              "单值不可当作精确值" if getattr(opt.peaks, "caliber_distinguishable",
                                                              True) else
                              "三种设置不可分辨：该档口径不可得，单值只是同一台机器上的一个样本")))
            else:
                # R-09/R-10：标定文件没记累加口径 → **拒绝用它算拐点**（但带宽侧照常）
                s.add(Metric.unavailable(
                    "est__achieved_tflops",
                    "拒绝用该标定算 roofline 拐点：" + (caliber_note or
                    "machine-profile.json 的 measured_matmul_tflops 缺少 acc_caliber / "
                    "precision_settings / method 字段。请用新版 tools/machine-profile.py 重新标定"),
                    source="calibration caliber check"))
                s.add(Metric.unavailable(
                    "est__pct_of_measured_compute_peak",
                    "同上：缺累加口径字段的 matmul 峰值不能用于拐点判定"))
                s.add(Metric.unavailable(
                    "est__bound",
                    "同上：compute roof 不可用，无法判定 compute/memory bound"))
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

    # ---------------- 访存轨迹（NVBit 插桩）----------------
    if "MemoryTrace" in want and opt.memtrace:
        st = _lookup_in(opt.memtrace, kr)
        if st is not None:
            sec = build_memtrace_section(st, opt)
            if sec is not None:
                out.append(sec)

    # ---------------- SchedulerStats / WarpStateStats（发射槽上界，R-08）----------------
    issue_want = [n for n in ("SchedulerStats", "WarpStateStats") if n in want]
    if issue_want and opt.issue_hist is not None:
        from .analyze.issue import from_histogram, to_metrics as issue_metrics

        hist = opt.issue_hist
        sub = hist.for_kernel(_simple_name(kr.name))
        if not sub.launches:
            sub = hist.for_kernel(kr.alias or "")
        if not sub.launches:
            for cand in hist.kernels:
                if _lookup_in({cand: cand}, kr):
                    sub = hist.for_kernel(cand)
                    break
        if sub.launches:
            merged = sub.merged()
            # 指令直方图与本次报告的 launch 数可能不同（通常轨迹跑得短、报告跑得长）。
            # 两者都是"每次 launch 处理同样的工作量"，所以按 per-launch 平均对齐，
            # 并把这件事写进 note —— 不这么做的结果取决于两次运行各跑了多少次，没有意义。
            n_hist = max(1, getattr(merged, "n_launches", 1) or 1)
            n_report = max(1, kr.launch_count or 1)
            scaled = merged.warp_instrs
            ratio_note = ""
            if n_hist != n_report:
                scaled = merged.warp_instrs * n_report / n_hist
                ratio_note = (f"；指令数按 per-launch 平均对齐到本报告的 {n_report} 次 launch"
                              f"（直方图那次是 {n_hist} 次）")
            merged.warp_instrs = int(round(scaled))
            st = from_histogram(
                merged, duration_ns=float(kr.total_ns), sm_count=opt.sm_count,
                clock_mhz=opt.clock_mhz,
                dram_throughput_pct=getattr(opt.dram, "throughput_pct_of_peak", None))
            if ratio_note:
                st.notes.append(ratio_note)
            metrics = issue_metrics(st)
            for section_name in issue_want:
                sec = Section(section_name)
                keep = (metrics if section_name == "SchedulerStats" else
                        [m for m in metrics if m[0].startswith(("est__stall_direction",
                                                                "est__dyn_instr_share_"))])
                for name, value, unit, source, note in keep:
                    if value is None:
                        sec.add(Metric.unavailable(name, st.unavailable_reason(), source=source))
                    else:
                        sec.add(Metric.estimated(name, value, unit, source=source, note=note))
                # 计数器版本的 stall 细分仍然不可得，必须显式说明
                sec.add(Metric.unavailable(
                    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                    UNAVAILABLE_REASONS["counters"] +
                    "；本工具给的是发射槽利用率上界 + 方向性判断（见本节 est__ 指标），"
                    "不是 stall 周期细分"))
                out.append(sec)

    # ---------------- 明确不可得的 section ----------------
    for name in requested_sections(opt):
        reason = COUNTER_SECTIONS.get(name)
        if reason:
            sec = Section(name)
            sec.add(Metric.unavailable(f"ks__{name.lower()}_status", reason))
            out.append(sec)

    return out


def _launch_for_config(kr, key: str):
    """取某一组配置的代表 launch（该组第一次）。"""
    from .report import config_key

    for l in kr.launches:
        if config_key(l) == key:
            return l
    return None


def _occupancy_for(config, opt: BuildOptions):
    """按一组 launch 配置算理论 occupancy（B4：逐组配置的读数）。"""
    try:
        return theoretical_occupancy(config.registers_per_thread, config.static_smem_bytes,
                                     config.dynamic_smem_bytes,
                                     config.block[0] * config.block[1] * config.block[2],
                                     opt.arch)
    except Exception:  # noqa: BLE001 - occupancy 是加分项，不能让它挡住报告
        return None


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


def _simple_name(name: str) -> str:
    """把 demangled 全签名压成"第一个标识符"，用于在外部工具输出里找同名 kernel。"""
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", name or "")
    return m.group(0) if m else ""


def _strip(name: str) -> str:
    from .backends.cubin import strip_mangled

    return strip_mangled(name)


def provenance_summary(report) -> dict[str, int]:
    return report.provenance_histogram()
