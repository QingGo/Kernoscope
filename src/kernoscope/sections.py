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
from .refusals import RefusalCode

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

# 需要硬件计数器才能给出的 section 及原因。
# W3：值从「一句散文」升级成 (拒绝码, 散文) —— 判决在码里，散文只解释。这样
# `--strict` 的门禁、结构化拒绝、capabilities 的覆盖矩阵共用同一份判决。
COUNTER_SECTIONS: dict[str, tuple[RefusalCode, str]] = {
    "ComputeWorkloadAnalysis": (RefusalCode.NO_COUNTERS, UNAVAILABLE_REASONS["counters"]),
    "MemoryWorkloadAnalysis": (RefusalCode.NO_COUNTERS, UNAVAILABLE_REASONS["counters"]),
    "MemoryWorkloadAnalysis_Tables": (RefusalCode.NO_COUNTERS, UNAVAILABLE_REASONS["counters"]),
    "SchedulerStats": (RefusalCode.NO_COUNTERS, UNAVAILABLE_REASONS["counters"]),
    "WarpStateStats": (RefusalCode.NO_COUNTERS, UNAVAILABLE_REASONS["counters"]),
    "SourceCounters": (RefusalCode.SAMPLING_UNAVAILABLE, UNAVAILABLE_REASONS["sampling"]),
    "PmSampling": (RefusalCode.SAMPLING_UNAVAILABLE, UNAVAILABLE_REASONS["counters"]),
    "Nvlink_Tables": (RefusalCode.NO_NVLINK,
                      "requires NVLink hardware with NVML counter support "
                      "(not present / not supported)"),
    "NumaAffinity": (RefusalCode.NOT_APPLICABLE, "not applicable on this device"),
    "WorkloadDistribution": (RefusalCode.INSTRUMENTATION_MISSING,
                             UNAVAILABLE_REASONS["instrumentation"]),
}


#: copyKind → 指标名里用的短标签（与 backends.nsys.COPY_KIND_NAMES 同一套语义）
COPY_KIND_LABEL = {1: "htod", 2: "dtoh", 3: "htoa", 4: "atoh", 5: "atoa",
                   6: "atod", 7: "dtoa", 8: "dtod", 9: "htoh", 10: "ptop"}


def _human_bytes(n: int) -> str:
    for unit, div in (("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)):
        if n >= div:
            v = n / div
            return f"{v:.0f}{unit}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}{unit}".replace(".", "p")
    return f"{n}b"


def counter_section_reason(name: str) -> str:
    """这个 section 被拒的散文原因（门禁消息与渲染共用）。"""
    entry = COUNTER_SECTIONS.get(name)
    return entry[1] if entry else ""


def counter_section_code(name: str) -> RefusalCode:
    entry = COUNTER_SECTIONS.get(name)
    return entry[0] if entry else RefusalCode.INTERNAL_UNKNOWN

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
    #: 窗口的环境证据（时钟驻留 / 降频画像 / 同卡其他进程），来自 SamplerResult.environment()
    nvml_env: dict[str, Any] = field(default_factory=dict)
    #: 平台**已授权**的遥测（backends.telemetry.TelemetryResult）：外部量写它自己的名字
    telemetry: Any = None
    dram: Any = None      # analyze.dram.DramEstimate（避免循环 import）
    cost: Any = None      # analyze.cost.CostEstimate（成本账；None = 本次不给成本结论）
    memtrace: Any = None  # {kernel: analyze.memtrace.MemTraceStats}
    issue_hist: Any = None  # backends.opcode_hist.OpcodeHist（NVBit 官方 opcode_hist 输出）
    clock_mhz: float = 0.0  # 用于发射槽上限的 SM 时钟（NVML 实测或用户给定）
    sm_count: int = 0
    #: 本次采集出现过的 deviceId（来自 CUPTI 的 `deviceId` 列）。>1 ⇒ 跨卡，见 build_device_kernel。
    device_ids: list[int] = field(default_factory=list)
    #: 按卡分段的原始行（每个 {device_id, launch_count, kernel_ns, kernels}），来自 Report。
    device_rows: list[dict[str, Any]] = field(default_factory=list)
    #: 传输/通信账本（analyze.comm.CommLedger）。None = 这次没读到拷贝 activity。
    comm: Any = None
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
    # ---- 跨卡事实（多卡真机实测，2026-10-12）------------------------------------------
    # 采集跨了多张卡时，按 kernel 名聚合的量（launch 数、时长合计、均值）是**跨卡加总**，
    # 而 NVML 读数只描述一张卡（`nvidia-smi` 的第一行）。旧行为是**什么都不说**：
    # 合成双设备轨迹实测，同名 kernel 的时长与 launch 数静默翻倍（31.9 ms/21 次 → 63.7 ms/42 次），
    # 既无告警也无拒绝 —— 那正是本项目平时会拒绝的口径混用。所以这里显式说出来，
    # 并把单卡口径的问题**拒绝**掉（不是给个偏小的数）。
    dev_ids = list(opt.device_ids)
    if dev_ids:
        s.add(Metric.derived(
            "ks__device_count", len(dev_ids), "张",
            source="CUPTI activity 的 deviceId 列（exact）",
            note=("本次采集出现过的设备：" + ", ".join(str(d) for d in dev_ids)
                  + ("。**跨卡**：按 kernel 名聚合的 launch 数与时长是**跨卡加总**，"
                     "不是单卡口径；NVML 读数（时钟/功耗/显存忙碌/能耗）只描述其中**一张**卡"
                     "（`nvidia-smi` 的第一行）。要单卡口径用 `--devices <id>`。"
                     if len(dev_ids) > 1 else
                     "。单卡采集：按 kernel 名聚合的量就是这张卡的量。"))))
        for row in opt.device_rows:
            tag = f"ks__device_{row['device_id']}"
            s.add(Metric.derived(f"{tag}__launch_count", row["launch_count"], "次",
                                 source="CUPTI activity（按 deviceId 分段，exact）",
                                 note="该设备上的 launch 次数；与跨卡合计分开看才知道有没有负载不均"))
            s.add(Metric.derived(f"{tag}__kernel_ms", round(row["kernel_ns"] / 1e6, 4), "ms",
                                 source="CUPTI activity（按 deviceId 分段，exact）"))
            s.add(Metric.derived(f"{tag}__kernels", row["kernels"], "个",
                                 source="CUPTI activity（按 deviceId 分段，exact）",
                                 note="该设备上出现过的不同 kernel 数"))
        if len(dev_ids) > 1:
            s.add(Metric.unavailable(
                "est__single_device_kernel_share_pct",
                "本次采集跨了 " + str(len(dev_ids)) + " 张卡：单卡口径的占比/负载均衡问题"
                "要按设备分段看，不能对跨卡合计下结论",
                source="CUPTI activity 的 deviceId 列",
                code=RefusalCode.MULTI_DEVICE_AGGREGATE,
                observed={"device_ids": dev_ids}))
    # 锁频状态与**采样**无关（它是驱动查询，不是采样读数），所以放在 `if n:` **外面**：
    # 采样被关掉/一次都没采到时，这条证据照样成立。
    lock = (opt.nvml_env or {}).get("clock_lock") or {}
    if lock:
        state = str(lock.get("state") or "unknown")
        label = {"locked": "已锁频", "unlocked": "未锁频",
                 "unknown": "读不到锁频状态"}.get(state, state)
        if state == "locked":
            note = (f"跨运行可比性：已锁频（{lock.get('min_mhz', 0):.0f}–"
                    f"{lock.get('max_mhz', 0):.0f} MHz）—— 两次运行跑在同一个频率上")
        else:
            note = ("跨运行可比性：" + label
                    + "。没锁频时两次运行可能跑在不同频率上，时长/能效**不可直接相减**；"
                      "要做 A/B 先锁频：`sudo nvidia-smi -lgc <min>,<max>`（跑完 "
                      "`sudo nvidia-smi -rgc` 复位）。我们**不会替你锁** —— 改设备时钟是写操作，"
                      "不在只读工具的授权里（见 NON-GOALS）。"
                    + (f"（{lock.get('note')}）" if lock.get("note") else ""))
        s.add(Metric.derived("ks__clock_lock_state", label, "",
                             source=f"锁频状态查询（{lock.get('source', 'n/a')}）",
                             note=note))
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
        for key, name, unit, note in (
                ("sm_clock_mean", "meas__sm_clock_mean_mhz", "MHz",
                 "窗口内**全部采样点**的均值：**含空闲点**，空闲降频不是问题。"
                 "要与活跃口径比较时看 `meas__sm_clock_p50_mhz`（只在干活时统计，R-100②）"),
                ("sm_clock_max", "meas__sm_clock_max_mhz", "MHz",
                 "窗口内峰值：它是「曾经跑到过」，不代表一直在跑"),
                ("power_max", "meas__power_max_w", "W",
                 "窗口内峰值功率：与功耗墙一起读才知道有没有贴着上限")):
            if key in n:
                s.add(Metric.measured(name, round(n[key], 1), unit,
                                      source="nvidia-smi / NVML（管理计数器，免 admin）",
                                      note=note,
                                      resolution="1 MHz / 1 W（驱动报数粒度）"))
        if n.get("energy_j"):
            s.add(Metric.measured("meas__device_energy_j", round(n["energy_j"], 3), "J",
                                  source="NVML total energy accumulator",
                                  note="ncu 不提供能耗指标", resolution="1 mJ"))
        # ---- 窗口的环境证据（R-61 延伸：不重放 ⇒ 必须说清环境是否可比）----
        env = opt.nvml_env or {}
        clock = env.get("clock") or {}
        if clock:
            s.add(Metric.measured(
                "meas__sm_clock_p50_mhz", round(clock["p50_mhz"], 1), "MHz",
                source="NVML/nvidia-smi 采样中位数（管理计数器，免 admin）",
                note=(f"**只在干活的时候统计**（{clock['active_samples']:.0f}/"
                      f"{clock['samples']:.0f} 个采样点算活跃）：{clock['at_max_pct']:.1f}% 跑在标称"
                      f"最高时钟附近（参考 {clock['ref_mhz']:.0f} MHz，活跃期最低 "
                      f"{clock['min_mhz']:.0f} MHz）；含空闲点的话中位数是 "
                      f"{clock['p50_all_mhz']:.0f} MHz（空闲降频不算问题）"),
                resolution=f"{env.get('interval_s', 0.0)} s/采样点"))
        thr = env.get("throttle") or {}
        if thr:
            reasons_map = thr.get("reasons") or {}
            reasons = "、".join(f"{k} {v:.0f}%" for k, v in reasons_map.items())
            fully_unknown = set(reasons_map) == {"unspecified"}
            if fully_unknown:
                # 位图**读不到**：这不是"没有降频"，也不是"100% 被限频"（两种都是编）。
                # 报 unavailable + code，让决策包前提因此**不成立**（R-96①④）—— 而不是
                # 印一个 measured 的 100.0% 让读者以为这台机器一直在掉频。
                s.add(Metric.unavailable(
                    "meas__throttle_limited_pct",
                    "降频位图读不到（pynvml/nvidia-smi 都没给出可信读数）："
                    "窗口内**可能有也可能没有**降频，我们不当成\"没有降频\"",
                    source="clocks_throttle_reasons.active 位图（NVML 公开位名解码）",
                    code=RefusalCode.ENV_READING_UNKNOWN,
                    note="降频未知时，能效/时间类结论按未知降级；"
                         "\"读不到\"与\"读到 0\"是两件事"))
            else:
                if reasons:
                    note = f"窗口内存在**性能受限**降频原因（按采样占比）：{reasons}"
                else:
                    note = "窗口内没有观察到性能受限降频（空闲位不算受限）"
                s.add(Metric.measured(
                    "meas__throttle_limited_pct", round(thr["limited_pct"], 1), "%",
                    source="clocks_throttle_reasons.active 位图（NVML 公开位名解码）",
                    note=note + "。降频不改变「耗时是真实时间」这个事实，但会改变**它能说明什么**",
                    resolution="位图口径，按采样点占比"))
            s.add(Metric.measured(
                "meas__gpu_idle_pct", round(thr["idle_pct"], 1), "%",
                source=("clocks_throttle_reasons.active 的 gpu_idle 位"
                        if not fully_unknown else
                        "位图读不到，暂用 SM 利用率为 0 的采样点判空闲（兜底口径）"),
                note="窗口内 GPU 处于空闲的采样占比：越高说明窗口越被宿主间隙稀释，"
                     "除以它的比值（如带宽/利用率）越不可信"
                     + ("；**位图读不到**时这一行用的是 SM 利用率兜底，不是驱动位图"
                        if fully_unknown else ""),
                resolution="位图口径，按采样点占比"
                           if not fully_unknown else "兜底口径（SM 利用率 0）"))
        if env.get("other_procs") is not None:
            others = int(env["other_procs"])
            note = ("采样期间同卡上**没有**其他计算进程，数字是独占环境下的"
                    if others == 0 else
                    f"采样期间同卡上检测到最多 {others} 个其他计算进程：**数字可能被同租户稀释**，"
                    "按它做的成本/能效结论要打折")
            s.add(Metric.measured(
                "meas__other_procs_on_gpu", float(others), "个",
                source="NVML nvmlDeviceGetComputeRunningProcesses / nvidia-smi --query-compute-apps",
                note=note, resolution="采样点最大值（每 5 个采样点探测一次）"))
        elif env.get("other_procs_probe") == "failed":
            # R-96④：探测**失败**不是"0 个别的租户"。包内必须能看出"这条没测"，
            # 否则多租户这条前提会被静默当成成立。
            s.add(Metric.unavailable(
                "meas__other_procs_on_gpu",
                "同卡其他计算进程这次没测到（NVML/nvidia-smi 探测失败）："
                "多租户是**未知**，不等于独占",
                source="NVML nvmlDeviceGetComputeRunningProcesses / "
                       "nvidia-smi --query-compute-apps",
                code=RefusalCode.ENV_READING_UNKNOWN))
    if not s.metrics:
        s.add(Metric.unavailable("meas__nvml_mem_busy_mean_pct",
                                 "没有 NVML 采样点：采样被关闭（--ks-no-nvml）、"
                                 "nvidia-smi/pynvml 不可用，或稳态段短于一个采样周期（200 ms）；"
                                 "没有它就无法给出显存忙碌代理",
                                 source="NVML", code=RefusalCode.NO_NVML_SAMPLES))
    return s


def build_comm_section(opt: BuildOptions) -> Section | None:
    """通信/传输账本（O1 的核心，也是 O2 分层账本共用的那块硬骨头）。

    这个 section 存在的理由是**把三类量的可信度分开**，而不是给出一个"传输时间"：
      * 字节 = `exact`（CUPTI 逐行相加）；
      * 单次时长 = `estimated`，按 `(copyKind, bytes)` 取 MAX（真机上 MAX 恰好捞回诚实那一行）；
      * **逐行时长求和 = 拒绝**，并把反面证据（求和会得到多少 GB/s）一起报出来。
    真机实测（2×2080 Ti，无 P2P 的主机中转）：求和 3184 GB/s vs 真值 5.49 GB/s，错约 580 倍。
    """
    led = opt.comm
    if led is None or not getattr(led, "has_any", False):
        # 没采到拷贝 activity **不等于传输为 0**（只有 --trace=cuda 才采）—— 如实说清。
        return None
    s = Section("CommLedger")
    if led.copies:
        s.add(Metric.exact(
            "ks__comm_bytes_total", led.bytes_total, "B",
            source="CUPTI CUPTI_ACTIVITY_KIND_MEMCPY 的 bytes 逐行相加（exact）",
            note=(f"共 {led.copies} 条拷贝 activity。**字节可信、时长不可信**："
                  "同一份 activity 的逐行时长相加会把传输带宽算错数百倍（见本节的拒绝条目）")))
        for kind, nb in sorted(led.bytes_by_kind.items()):
            name = COPY_KIND_LABEL.get(int(kind), f"kind_{kind}")
            s.add(Metric.exact(f"ks__comm_bytes_{name}", int(nb), "B",
                               source=f"CUPTI copyKind={kind}（{name}）的 bytes 合计"))
        for dev, nb in sorted(led.bytes_by_device.items()):
            s.add(Metric.exact(f"ks__device_{dev}__comm_bytes", int(nb), "B",
                               source="CUPTI MEMCPY 的 deviceId 分段（exact）"))
        # MAX 口径：单次时长的最佳可得估计
        for g in led.groups[:4]:
            if g.bytes < (1 << 20):
                continue
            label = COPY_KIND_LABEL.get(int(g.copy_kind), f"kind_{g.copy_kind}")
            s.add(Metric.estimated(
                f"est__copy_{label}_{_human_bytes(g.bytes)}_gbs",
                round(g.achieved_gbs, 2), "GB/s",
                source=f"CUPTI bytes ÷ 该 (copyKind={g.copy_kind}, bytes={g.bytes}) 组的**最大**时长",
                note=(f"这一组有 {g.count} 行；取 MAX 而不是求和，是因为同组里既有诚实行也有"
                      f"报 80–95 µs 的坏行（1 GiB 报 80 µs 等于 13.3 TB/s，物理上不可能）。"
                      f"MAX = {g.max_duration_ns / 1e6:.3f} ms")))
        # 反面证据：把"求和"这条路显式拒绝掉，并把错多少一起报出来
        s.add(Metric.unavailable(
            "ks__comm_time_sum_ns",
            "拒绝把拷贝 activity 的逐行时长相加当传输时间：这台机器上会出现数百倍的错",
            source="CUPTI CUPTI_ACTIVITY_KIND_MEMCPY 的 end-start 逐行求和",
            code=RefusalCode.MEMCPY_DURATION_UNTRUSTED,
            observed={"sum_ns": led.duration_sum_ns,
                      "bytes": led.bytes_total,
                      "implied_gbs_if_summed": round(led.implied_gbs_if_summed, 1)}))
    if led.comm_kernel_ns:
        s.add(Metric.exact(
            "ks__comm_kernel_ms", round(led.comm_kernel_ns / 1e6, 4), "ms",
            source="通信 kernel（NCCL 等）的 CUPTI activity 时长（exact）",
            note=("传输时间在这里是**可靠的**：真机实测 NCCL 的传输全部发生在 kernel 内，"
                  "每 rank 只有 24 KB 的 bootstrap 拷贝 —— 所以「通信时间」要从 kernel 拿，"
                  "不要去 memcpy 表里找。识别到的 kernel：" + "、".join(led.comm_kernel_names[:3]))))
        s.add(Metric.exact(
            "ks__comm_kernel_share_pct", round(led.comm_kernel_share_pct, 1), "%",
            source="通信 kernel 时长 ÷ 全部 kernel 时长（exact）",
            note="占比的分母是**全部 kernel 时间**（跨卡时它是跨卡加总，见 Devices section）"))
        s.add(Metric.exact(
            "ks__comm_compute_overlap_pct", round(led.overlap_pct_of_comm, 1), "%",
            source="通信 kernel 区间 ∩ 计算 kernel 区间（CUPTI 时间戳，exact）",
            note=("只对**通信 kernel** 算重叠：拷贝那一侧的时长不可信，拿它算重叠等于把"
                  "数百倍的错引进第二个数。重叠 = 通信时间里同时有计算在跑的比例")))
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
            ref = est.refusal_for(name, opt.ledger_basis)
            s.add(Metric.unavailable(name, ref.message, source=source, resolution=resolution,
                                     code=ref.code, observed=ref.observed,
                                     threshold=ref.threshold, remedy=ref.remedy,
                                     severity=ref.severity))
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


def _cost_unavailable(s: Section, name: str, ref, unit: str = "", source: str = "",
                      note: str = "") -> None:
    """成本账里的 n/a 行：拒绝必须带 code（R1/W3），散文只解释。"""
    s.add(Metric.unavailable(name, ref.message, unit=unit, source=source, note=note,
                             code=ref.code, observed=ref.observed,
                             threshold=ref.threshold, remedy=ref.remedy,
                             severity=ref.severity))


def build_telemetry_section(opt: BuildOptions) -> Section | None:
    """平台**已授权**的遥测读数（N7）：外部量写在自己的名字下，单位照抄端点。

    三件事必须成立，否则这个 section 宁可不出现：
      * 每条读数带**端点署名**与真单位（`to_provenance_metrics()` 已经这么做）；
      * 缺席/读不到的字段以 `unavailable` + code 出现（**不是 0**，也不是消失）；
      * "百分比/比例"绝不当字节 —— 那由 `backends.telemetry` 在构造期与发布前两道闸拦住，
        这里再跑一次 `byte_claim_violations()` 复核（N7 把它定为研究诚信事故）。
    """
    tel = opt.telemetry
    if tel is None:
        return None
    sec = Section("Telemetry")
    for m in tel.to_provenance_metrics():
        sec.add(m)
    for m in tel.unavailable_metrics():
        sec.add(m)
    if not sec.metrics:
        return None
    from .backends import telemetry as _tel_backend

    bad = _tel_backend.byte_claim_violations(tel.metrics)
    if bad:
        # 发布前的最后一道闸：真出现了就是我们的 bug，必须炸在写报告之前，而不是发出去
        raise ValueError("遥测里的比例/百分比被当成了字节（N7 的研究诚信红线）：" + "；".join(bad))
    note = (f"端点：{tel.source}；probe={tel.probe}"
            + (f"（{tel.reason}）" if tel.reason else "")
            + ("；端点里**有** profiling 字段 ⇒ 说明平台管理员授权了这些字段"
               if tel.profiling_present else
               "；端点里**没有** profiling 字段 —— 这只说明「没导出」，**不推**「被拒绝」"))
    sec.metrics.insert(0, Metric.measured(
        "meas__telemetry_endpoint_readings", float(len(tel.metrics)), "个",
        source=f"DCGM/Prometheus 文本端点（{tel.source}）", note=note,
        resolution="按端点自己的导出周期（不由我们控制）"))
    return sec


def build_cost_section(opt: BuildOptions) -> Section | None:
    """成本账（C1）：窗口能耗 → tokens/s → 单位成本（钱）。

    出数规则（每一格都对应 cost.CostEstimate 里的一个事实，不在这里二次计算）：
      * 能耗实测（`meas__`）→ J/token、tokens/s（`ks__` derived，取自实测与调用方给的 token 数）；
      * 没给 token 数 → 单位成本的**分母**不存在 ⇒ 那几行是 n/a + `no_tokens`，
        而"每小时多少钱"仍然给（它只需要功率与电价）；
      * 没给价格 → 钱那几行 n/a + `no_price`，能耗数字照给。
    这样"给一半输入也能拿到一半结论"，且缺什么一望可知（remedy 里写明加哪个开关）。
    """
    est = opt.cost
    if est is None:
        return None
    s = Section("CostLedger")
    cur = est.inputs.currency or "CNY"
    src = f"NVML 能耗（{est.energy_source}）" if est.energy_source else "NVML 能耗"

    # ---- ① 能耗（实测）----
    if est.has_energy:
        s.add(Metric.measured("meas__window_energy_j", round(est.window_energy_j, 3), "J",
                              source=src, resolution=est.energy_resolution,
                              note="；".join(est.notes) if est.notes else ""))
        if est.energy_w is not None:
            s.add(Metric.measured("meas__window_power_mean_w", round(est.energy_w, 1), "W",
                                  source=src, resolution=est.energy_resolution,
                                  note="窗口内平均功耗（能量 ÷ 窗口长度）；"
                                       "不是瞬时峰值功耗"))
    else:
        _cost_unavailable(s, "meas__window_energy_j", est.energy_refusal, "J",
                          source="NVML 功率/能量读数")

    body_note = ("；".join(est.notes) if est.notes else "")

    # ---- ② 量：窗口内 steps 与 token 数 ----
    if est.steps_in_window is not None:
        s.add(Metric.derived("ks__steps_in_window", round(est.steps_in_window, 3), "",
                             source="窗口内等效 launch 数（按 launch 时间区间与采样窗口重叠加权）",
                             note="单位成本的分母：只有窗口对齐时才算得出来"))
    elif est.steps_refusal is not None:
        _cost_unavailable(s, "ks__steps_in_window", est.steps_refusal,
                          source="窗口/launch 时间区间对齐")
    if est.tokens_total is not None:
        s.add(Metric.derived("ks__tokens_in_window", round(est.tokens_total, 1), "token",
                             source=f"--ks-tokens {est.inputs.tokens_per_step:g} × 窗口内 "
                                    f"{est.steps_in_window:g} step",
                             note="token 数由调用方给出（我们不知道你的 step 装了多少 token）"))
    elif est.tokens_refusal is not None:
        _cost_unavailable(s, "ks__tokens_in_window", est.tokens_refusal, "token",
                          source="--ks-tokens")
    if est.tokens_per_s is not None:
        s.add(Metric.derived("ks__tokens_per_s", round(est.tokens_per_s, 1), "token/s",
                             source="窗口内 token 数 ÷ 窗口秒数", note=body_note))
    elif est.tokens_refusal is not None:
        _cost_unavailable(s, "ks__tokens_per_s", est.tokens_refusal, "token/s",
                          source="--ks-tokens + 窗口对齐的 launch 数")
    if est.j_per_token is not None:
        s.add(Metric.derived("ks__j_per_token", float(f"{est.j_per_token:.4g}"), "J/token",
                             source=f"窗口能耗 ÷ 窗口内 token 数（{src}）",
                             resolution=f"由 {est.energy_resolution} 与 token 数共同决定",
                             note=body_note))
    elif est.tokens_refusal is not None:
        _cost_unavailable(s, "ks__j_per_token", est.tokens_refusal, "J/token",
                          source="窗口能耗 ÷ token 数")

    # ---- ③ 钱：两种载体 ----
    if est.cost_per_hour is not None:
        comp = []
        if est.cost_per_hour_energy is not None:
            comp.append(f"电费 {est.cost_per_hour_energy:.3g}（{est.inputs.price_per_kwh:g}/{cur}/kWh）")
        if est.cost_per_hour_rental is not None:
            comp.append(f"租约 {est.cost_per_hour_rental:.3g}（{est.inputs.price_per_hour:g}/{cur}/h）")
        s.add(Metric.derived("ks__cost_per_hour", round(est.cost_per_hour, 4), f"{cur}/h",
                             source=" + ".join(comp),
                             note="两种载体是不同口径的成本：分别可查，各自给就各自算，"
                                  "都给才相加（不做汇率换算）"))
        if est.cost_per_hour_energy is not None:
            s.add(Metric.derived("ks__cost_per_hour_energy", round(est.cost_per_hour_energy, 4),
                                 f"{cur}/h", source=f"窗口平均功耗 × {est.inputs.price_per_kwh:g} {cur}/kWh"))
        if est.cost_per_hour_rental is not None:
            s.add(Metric.derived("ks__cost_per_hour_rental", round(est.cost_per_hour_rental, 4),
                                 f"{cur}/h", source=f"--ks-price-per-hour {est.inputs.price_per_hour:g}"))
    else:
        ref = est.price_refusal or est.energy_refusal
        if ref is not None:
            _cost_unavailable(s, "ks__cost_per_hour", ref, f"{cur}/h",
                              source="--ks-price-per-kwh / --ks-price-per-hour")

    if est.cost_per_mtok is not None:
        s.add(Metric.derived("ks__cost_per_mtok", float(f"{est.cost_per_mtok:.4g}"),
                             f"{cur}/Mtoken",
                             source=f"每小时成本 ÷ 每小时 token 数（{est.tokens_per_s:,.0f} token/s）",
                             note="= ks__cost_per_hour × 1e6 ÷ (tokens/s × 3600)；"
                                  "拆分见下面两行（电费/租约）"))
    else:
        ref = est.price_refusal or est.tokens_refusal
        if ref is not None:
            _cost_unavailable(s, "ks__cost_per_mtok", ref, f"{cur}/Mtoken",
                              source="每小时成本 ÷ tokens/s")
    for name, value, label in (("ks__cost_per_mtok_energy", est.cost_per_mtok_energy, "电费口径"),
                               ("ks__cost_per_mtok_rental", est.cost_per_mtok_rental, "租约口径")):
        if value is not None:
            s.add(Metric.derived(name, float(f"{value:.4g}"), f"{cur}/Mtoken",
                                 source=f"{label}；只含这一种载体"))
    return s


def build_cost_attribution_section(opt: BuildOptions, kr) -> Section | None:
    """按 kernel 归因的成本账（代理口径：窗口内时间占比 × 能耗）。

    `est__` + estimated 是刻意的：能耗计在**设备**上，按时间占比分摊是假设
    （窗口内功率大致均匀）。名字与出处一起把这件事说清楚，读者不会把它当计数器读数。
    """
    est = opt.cost
    if est is None or kr is None or not est.has_energy:
        return None
    if est.kernel_share_pct is None or est.kernel_energy_j is None:
        return None
    cur = est.inputs.currency or "CNY"
    s = Section("CostLedger")
    note = (f"分摊口径：该 kernel 占窗口时间 {est.kernel_share_pct:.1f}% × 窗口能耗 "
            f"{est.window_energy_j:.1f} J。**代理口径**：它假设窗口内功率大致均匀，"
            f"所以是方向性的，不是计数器读数"
            + ("；已按 --ks-idle-power-w 扣除空闲能耗"
               if est.inputs.idle_power_w is not None else
               "；未扣空闲功耗，空闲占比较高时这一项会偏大（可用 --ks-idle-power-w 扣）"))
    s.add(Metric.estimated("est__kernel_energy_share_pct", round(est.kernel_share_pct, 1), "%",
                           source="窗口内该 kernel 的时间占比（CUPTI activity）", note=note))
    s.add(Metric.estimated("est__kernel_energy_j",
                           float(f"{est.kernel_energy_j:.4g}"), "J",
                           source=f"窗口能耗 × 时间占比（窗口合计 {est.window_energy_j:.1f} J）",
                           note="整个 kernel（同名全部 launch）在窗口内的能耗分摊"))
    if est.kernel_energy_per_launch_j is not None:
        s.add(Metric.estimated("est__kernel_energy_per_launch_j",
                               float(f"{est.kernel_energy_per_launch_j:.4g}"), "J",
                               source="上者 ÷ 窗口内等效 launch 数",
                               note="单次 launch 的分摊能耗"))
    if est.kernel_cost_per_hour is not None:
        s.add(Metric.estimated("est__kernel_cost_per_hour",
                               float(f"{est.kernel_cost_per_hour:.4g}"), f"{cur}/h",
                               source="ks__cost_per_hour × 时间占比",
                               note="这个 kernel 按时间占比分摊到的每小时成本；"
                                    "优化它省下的钱不会超过这个数"))
    return s


def _split_unavailable(note: str) -> tuple[str, str]:
    """从 `unavailable: code=xxx | 原因` 里拆出 (code, 原因)。

    为什么用这个约定：`Prescription.to_metrics()` 的返回元组只有四格，装不下结构化拒绝，
    所以拒绝被编码进 note 的第一段。**拆不出来时退回 PREMISE_FAILED**（有 code 总比
    散文式拒绝强：R1/W3 要求 n/a 行必须带 code）。
    """
    if note.startswith("unavailable: code="):
        rest = note[len("unavailable: code="):]
        code, _, why = rest.partition(" | ")
        try:
            return RefusalCode(code.strip()).value, (why.strip() or "口径不成立")
        except ValueError:
            return RefusalCode.PREMISE_FAILED.value, (why.strip() or "口径不成立")
    return RefusalCode.PREMISE_FAILED.value, note or "口径不成立"


def build_prescription_section(prescription: Any) -> Section | None:
    """处方 section（C2）：排序后的头寸 + 每条的区间与金额，拒绝逐行带 code。

    命名纪律也在这里执行：`ks__` 行是 derived（由实测再算），`est__` 行是 estimated；
    文本行（kernel 名、口径注记）不进指标表，改放 `Section.note` —— 指标表里混字符串会让
    CSV/JSON 下游的解析长出一堆特例。
    """
    if prescription is None:
        return None
    s = Section("Prescription")
    notes: list[str] = []
    for name, value, unit, note in prescription.to_metrics():
        if value is None:
            code, why = _split_unavailable(note)
            s.add(Metric.unavailable(name, why, unit=unit, code=RefusalCode(code),
                                     observed={"note": note[:200]}, severity="degraded"))
        elif isinstance(value, str):
            notes.append(f"{name}: {value}" if value else name)
        elif name.startswith("est__"):
            s.add(Metric.estimated(name, value, unit, source=note))
        else:
            s.add(Metric.derived(name, value, unit, source=note))
    s.note = "；".join(notes)
    return s


def build_memtrace_section(st, opt: BuildOptions) -> Section | None:
    """访存轨迹 section（NVBit 插桩口径）。

    provenance 规则：全量轨迹 → `meas__` + measured；**按采样比放大**的轨迹不是"实测读数"
    （样本代表性是假设），此时前缀改为 `est__` + estimated（R-12：名字承诺什么，
    出处就得是什么）。前缀本身由 R3 强制。
    """
    from .analyze.memtrace import split_unavailable_note, to_metrics

    if st is None:
        return None
    s = Section("MemoryTrace")
    sampled = st.sampling_ratio < 0.999
    for name, value, unit, source, note in to_metrics(st):
        if value is None:
            # R-78：「0 不是测到 0」—— 全局口径 0 条记录时这些指标没有定义，
            # `to_metrics` 把它们标成拒绝（note 第一段带 code），这里还原成结构化 n/a 行，
            # 而不是 `continue` 掉（静默消失比错的 0 更让人无从查起）。
            code, why = split_unavailable_note(note)
            s.add(Metric.unavailable(
                name, why, unit=unit, source=source, code=code,
                observed={"records": st.records, "records_global": st.records_global,
                          "kind_counts": dict(st.kind_counts)},
                severity="blocking"))
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
                                 st.dram_bytes_unavailable_reason or "轨迹不足以给出 DRAM 字节",
                                 code=RefusalCode.TRACE_UNAVAILABLE,
                                 observed={"sampling_ratio": round(st.sampling_ratio, 4),
                                           "records": st.records,
                                           "records_global": st.records_global,
                                           "l2_bytes": st.l2_bytes}))
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
                f"故报 n/a；逐组读数见 LaunchConfigs section",
                code=RefusalCode.MIXED_LAUNCH_CONFIG,
                observed={"configs": len(configs), "grid_sizes": grids},
                severity="degraded"))
            s.add(Metric.unavailable(
                "launch__block_size",
                f"该 kernel 有 {len(configs)} 组不同 launch 配置（block="
                f"{'/'.join(str(b) for b in blocks)}）：单一数值会误导，故报 n/a",
                code=RefusalCode.MIXED_LAUNCH_CONFIG,
                observed={"configs": len(configs), "block_sizes": blocks},
                severity="degraded"))
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
            code=RefusalCode.NO_COUNTERS, severity="degraded",
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
                                                   f"({opt.peaks.dtype_key})",
                                       note=sol.near_peak_note()))
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
                    source="calibration caliber check",
                    code=RefusalCode.CALIBRATION_CALIBER_MISMATCH,
                    observed={"caliber_note": caliber_note} if caliber_note else {}))
                s.add(Metric.unavailable(
                    "est__pct_of_measured_compute_peak",
                    "同上：缺累加口径字段的 matmul 峰值不能用于拐点判定",
                    code=RefusalCode.CALIBRATION_CALIBER_MISMATCH))
                s.add(Metric.unavailable(
                    "est__bound",
                    "同上：compute roof 不可用，无法判定 compute/memory bound",
                    code=RefusalCode.CALIBRATION_CALIBER_MISMATCH))
            s.add(Metric.estimated("est__arithmetic_intensity", round(sol.arithmetic_intensity, 2),
                                   "FLOP/B", source="user-provided workload"))
            # R-38：达成值 > 标定峰值是物理上不可能的，必须显式报红灯，
            # 而不是让它安静地变成一个"看起来更漂亮"的百分比。
            if sol.over_peak:
                s.add(Metric.derived("ks__over_peak_warning", 1.0, "",
                                     source="achieved > measured peak（roofline 一致性守卫）",
                                     note=sol.over_peak_note()))
        else:
            s.add(Metric.unavailable(
                "est__achieved_tflops",
                "workload not provided: pass --ks-gemm M,N,K or --ks-flops/--ks-bytes "
                "(plus --ks-calibration machine-profile.json for measured peaks)",
                code=RefusalCode.NO_WORKLOAD,
                observed={"peaks_provided": bool(opt.peaks)}, severity="degraded",
            ))
        s.add(Metric.unavailable("sm__throughput.avg.pct_of_peak_sustained_elapsed",
                                 UNAVAILABLE_REASONS["counters"],
                                 code=RefusalCode.NO_COUNTERS, severity="degraded"))
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
                                     "no cubin provided: pass --ks-cubin <file.so|.cubin>",
                                     code=RefusalCode.NO_STATIC_INPUT))
        s.add(Metric.unavailable("smsp__inst_executed.sum", UNAVAILABLE_REASONS["counters"],
                                 source="dynamic instruction count needs counters; "
                                        "static mix from cubin is provided instead",
                                 code=RefusalCode.NO_COUNTERS, severity="degraded"))
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
                        iref = st.refusal()
                        sec.add(Metric.unavailable(name, iref.message, source=source,
                                                   code=iref.code, observed=iref.observed,
                                                   threshold=iref.threshold,
                                                   remedy=iref.remedy,
                                                   severity=iref.severity))
                    else:
                        sec.add(Metric.estimated(name, value, unit, source=source, note=note))
                # 计数器版本的 stall 细分仍然不可得，必须显式说明
                sec.add(Metric.unavailable(
                    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                    UNAVAILABLE_REASONS["counters"] +
                    "；本工具给的是发射槽利用率上界 + 方向性判断（见本节 est__ 指标），"
                    "不是 stall 周期细分",
                    code=RefusalCode.NO_COUNTERS, severity="degraded"))
                out.append(sec)

    # ---------------- 明确不可得的 section ----------------
    # 已经用别的数据源产出过的同名 section 不再补一个只有 n/a 的孪生体（R-43 的前置：
    # 同一个 section 名出现两次，会让"这个 section 到底有没有数据"变得不可判定）。
    already = {s.name for s in out}
    for name in requested_sections(opt):
        reason = counter_section_reason(name)
        if reason and name not in already:
            sec = Section(name)
            sec.add(Metric.unavailable(f"ks__{name.lower()}_status", reason,
                                       code=counter_section_code(name),
                                       observed={"section": name}))
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
