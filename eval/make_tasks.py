#!/usr/bin/env python3
"""生成 `eval/tasks/*.json` —— **任务 = 被测 agent 能看到的材料 + 真值**。

为什么用生成器而不是手写 JSON：任务里的决策包必须与 `kernoscope --packet` 的**真实
结构**一致（前提、拒绝 code、代价、尺寸都在里面）。手抄一份"看起来像包"的 JSON，
迟早会和 `packet.build()` 漂开 —— 那时评测台测的就是抄错的结构，而不是真的包。

所以这里的每个包都由 `kernoscope.packet.build(Report(...))` 真的组出来：判词、前提、
结构化拒绝、`refusals` 表、`cost` 都由产品代码生成，评测台只负责加一层真值。

用法：
    python eval/make_tasks.py            # 写 eval/tasks/*.json（会覆盖）
    python eval/make_tasks.py --check    # 只校验已入库的任务，不写文件

**入库的 `eval/tasks/*.json` 才是任务的真源**（评测台与测试只读它们）。工具版本升级后
重新生成可能让包里的 `tool_version`/`protocol.hash` 变化 —— 那是预期的：包是"某次测量"
的快照，不是永恒常量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kernoscope.capabilities import Env, evaluate                  # noqa: E402
from kernoscope.packet import build, dumps, render_onepage         # noqa: E402
from kernoscope.provenance import Metric, Section                  # noqa: E402
from kernoscope.refusals import RefusalCode                        # noqa: E402
from kernoscope.report import DeviceInfo, KernelLaunch, KernelReport, Report  # noqa: E402

TASKS_DIR = ROOT / "eval" / "tasks"


# --------------------------------------------------------------------------
# 组报告（与 tests/test_packet.py 同一条路：Report -> packet.build）
# --------------------------------------------------------------------------
def _launch(i: int, dur_ns: int, name: str) -> KernelLaunch:
    return KernelLaunch(name=name, demangled=name + "(float*)", start_ns=i * 2_000_000,
                        end_ns=i * 2_000_000 + dur_ns, grid=(1024, 1, 1), block=(256, 1, 1),
                        registers_per_thread=64, static_smem_bytes=0, dynamic_smem_bytes=49152,
                        local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0,
                        correlation_id=i)


def _value_metric(name: str, value, unit: str = ""):
    if name.startswith("meas__"):
        return Metric.measured(name, value, unit)
    if name.startswith("ks__"):
        return Metric.derived(name, value, unit)
    if name.startswith("gpu__"):
        return Metric.exact(name, value, unit)
    return Metric.estimated(name, value, unit)


def _na_metric(name: str, code: RefusalCode, reason: str, observed=None, threshold=None,
               resolution: str = ""):
    return Metric.unavailable(name, reason, code=code, observed=observed, threshold=threshold,
                              resolution=resolution)


def _report(kernels: list[tuple[str, int, int]], wall_ns: int, window: str) -> Report:
    """kernels: [(名字, launch 数, 每次时长 ns)]，时长决定 total_ns 与 kernel 份额。"""
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128,
                                   l2_mib=72.0, memory_total_mib=24564.0))
    rep.wall_time_ns = wall_ns
    rep.meta["window"] = window
    rep.meta["dtype"] = "float16"
    rep.meta["peak_source"] = "measured"
    rep.meta["calibration"] = "machine-profile.json"
    for name, launches, dur in kernels:
        k = KernelReport(name, launch_count=launches,
                         launches=[_launch(i, dur, name) for i in range(launches)])
        k.sections = [Section("All")]
        rep.kernels.append(k)
    return rep


def _put(rep: Report, metrics: list, kernel: int = 0) -> None:
    rep.kernels[kernel].sections[0].extend(metrics)


def _caps(tier: int) -> object:
    """按目标层造一份能力清单（离线，不探测）。"""
    common = dict(has_gpu=True, gpu_name="RTX 4090", has_nvml=True, has_nsys=True,
                  has_cuobjdump=True, has_ncu=True, counter_state="denied",
                  has_calibration=True, has_workload=True)
    env = Env(has_nvbit=tier >= 3, **common)
    return evaluate(env)


COUNTER_NA = [
    _na_metric("dram__bytes.sum", RefusalCode.NO_COUNTERS,
               "requires GPU hardware performance counters, which are restricted here",
               observed={"counter_state": "denied"}, threshold={"counters": "available"}),
    _na_metric("lts__t_sector_hit_rate.pct", RefusalCode.NO_COUNTERS,
               "requires GPU hardware performance counters, which are restricted here",
               observed={"counter_state": "denied"}, threshold={"counters": "available"}),
    _na_metric("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
               RefusalCode.NO_COUNTERS,
               "requires GPU hardware performance counters, which are restricted here",
               observed={"counter_state": "denied"}, threshold={"counters": "available"}),
]


# --------------------------------------------------------------------------
# 任务 1：饱和拷贝 —— 前提全部成立，可以按访存优化
# --------------------------------------------------------------------------
def task_01() -> dict:
    wall = 60_000_000
    rep = _report([("vectorized_copy_kernel", 2000, 27_200)], wall,
                  "mode=auto 采样 300 点 / 参与积分 60.0 ms")
    _put(rep, [
        _value_metric("meas__nvml_mem_busy_mean_pct", 96.8),
        _value_metric("ks__dram_window_mem_busy_mean_pct", 96.8),
        _value_metric("ks__sampler_window_coverage_pct", 97.1),
        _value_metric("est__dram_bytes_total", 5.0e4, "MB"),
        _value_metric("est__dram_bytes_per_launch", 25.0, "MB"),
        _value_metric("ks__matched_kernel_kernel_share_pct", 99.1),
        _value_metric("ks__attribution_bias_estimate_pct", 0.9),
        _value_metric("est__dram_vs_ledger_err_pct", -1.2),
        _value_metric("ks__theoretical_occupancy_pct", 50.0),
        _value_metric("meas__achieved_occupancy_pct", 46.9),
        _value_metric("ks__occupancy_limiter", "registers"),
        _value_metric("est__bound", "memory"),
        _value_metric("est__roofline_efficiency_pct", 96.7),
        _value_metric("est__achieved_gbs", 919.0),
    ] + COUNTER_NA)
    cap = _caps(3)
    packet = build(rep, capabilities=cap,
                   basis="est__dram_bytes_total = NVML 显存忙碌 × 本机实测峰值")
    return {
        "id": "01_saturated_copy_dram_bound",
        "scenario": (
            "饱和拷贝：kernel `vectorized_copy_kernel` 2000 次 launch，窗口覆盖率 97.1%、"
            "被匹配 kernel 占窗口 kernel 时间 99.1%、NVML 显存忙碌 96.8%，roofline 判词给出 "
            "bound=memory、达成带宽 919 GB/s（离本机实测峰值 3.3%）。用户问：访存还有多少空间？"
        ),
        "question_for_agent": (
            "你要决定这个 kernel 下一步往哪个方向优化。请给出 classification、你**据以行动**的"
            "判词 id 列表、以及一句可执行的 next_action，并写成 decision.json。"
        ),
        "material": {
            "packet": packet,
            "files": {"onepage.txt": render_onepage(packet) + "\n"},
        },
        "truth": {
            "classification": "dram_bound",
            "safe_verdicts": ["kernel_time", "dram_bytes", "dram_bytes_per_launch",
                              "ledger_verdict", "roofline", "occupancy"],
            "forbidden_verdicts": [],
            "expected_action_keywords": ["访存", "带宽", "DRAM", "合并"],
            "code_change_allowed": True,
            "why": (
                "三个归因前提（窗口覆盖率 ≥80%、被匹配 kernel 份额 ≥95%、窗口被负载主导）"
                "全部成立，且两条口径（NVML 代理、roofline）都指向 memory bound。"
                "这时按访存优化是有依据的 —— 反过来说，用'别急，先改测量口径'去回避它，"
                "就是把一个已经成立的结论当成没有结论。"
            ),
        },
        "forbidden_actions": ["优化计算", "转计算侧", "减少 FLOPs", "先改窗口口径",
                              "重跑标定", "没有证据"],
    }


# --------------------------------------------------------------------------
# 任务 2：窗口里跑了多个 kernel —— per-launch 不可归因（正确答案：不要动代码）
# --------------------------------------------------------------------------
def task_02() -> dict:
    wall = 900_000_000
    # 4 个 kernel 顺序跑在同一个窗口里，用户只关心 gemm（-k 过滤匹配到它，但窗口没缩小）
    rep = _report([("preprocess_kernel", 120, 3_000_000),
                   ("cutlass_gemm_f16", 200, 1_640_000),
                   ("epilogue_kernel", 60, 1_200_000),
                   ("scatter_kernel", 40, 1_000_000)], wall,
                  "mode=auto 采样 180 点 / 参与积分 0.9 s")
    _put(rep, [
        _value_metric("meas__nvml_mem_busy_mean_pct", 93.0),
        _value_metric("ks__dram_window_mem_busy_mean_pct", 93.0),
        _value_metric("ks__sampler_window_coverage_pct", 96.0),
        _value_metric("est__dram_bytes_total", 5.0e3, "MB"),
        _value_metric("est__dram_vs_ledger_err_pct", 146.9),
        _value_metric("ks__theoretical_occupancy_pct", 50.0),
    ] + COUNTER_NA)
    # per-launch 归因口径的三个量只挂在被匹配的 gemm 上
    _put(rep, [
        _value_metric("est__dram_bytes_per_launch", 25.0, "MB"),
        _value_metric("ks__matched_kernel_kernel_share_pct", 41.0),
        _value_metric("ks__matched_kernel_window_share_pct", 38.5),
        _value_metric("ks__attribution_bias_estimate_pct", 146.9),
    ], kernel=1)
    # 账本对拍走的就是 per-launch 口径 —— 必须与 CLI 一样把口径写进 meta/basis，
    # 否则"per-launch 不可归因 ⇒ 账本误差也不可引用"这条继承判定不会生效
    # （外部评测里 agent 正是拿 +146.9% 当真结论用的）。
    rep.meta["ledger_basis"] = "per-launch"
    cap = _caps(2)
    packet = build(rep, capabilities=cap, basis="per-launch")
    return {
        "id": "02_multi_kernel_window_unattributable",
        "scenario": (
            "一个 0.9 s 的窗口里顺序跑了 4 个 kernel（preprocess / cutlass_gemm_f16 / epilogue / "
            "scatter）。用户只关心 `cutlass_gemm_f16`，但窗口没缩小：它只占窗口内 kernel 时间的 "
            "41.0%（门槛 95%）。工具照样给出了「每次 launch 约 25 MB」与「与账本相差 +146.9%」。"
            "窗口 DRAM 代理（5.0 GB）是**整个窗口**的，不是这个 kernel 的。"
        ),
        "question_for_agent": (
            "你要决定这个 kernel 下一步往哪个方向优化。请给出 classification、你**据以行动**的"
            "判词 id 列表、以及一句可执行的 next_action，并写成 decision.json。"
        ),
        "material": {
            "packet": packet,
            "files": {
                "window_launches.csv": (
                    "order,kernel,launches,total_ms,share_of_kernel_time_pct\n"
                    "1,preprocess_kernel,120,360.0,45.0\n"
                    "2,cutlass_gemm_f16,200,328.0,41.0\n"
                    "3,epilogue_kernel,60,72.0,9.0\n"
                    "4,scatter_kernel,40,40.0,5.0\n"
                    "# 注意：窗口是整段 0.9 s；kernel 过滤只匹配到 cutlass_gemm_f16，"
                    "窗口没有被缩小\n"
                ),
            },
        },
        "truth": {
            "classification": "window_unattributable",
            "safe_verdicts": ["kernel_time"],
            "forbidden_verdicts": ["dram_bytes_per_launch", "dram_bytes", "ledger_verdict"],
            "expected_action_keywords": ["缩小窗口", "-k", "重跑", "测量口径", "换轨迹"],
            "code_change_allowed": False,
            "why": (
                "per-launch 归因的两个前提里，'被匹配 kernel 占窗口 kernel 时间 ≥95%'不成立"
                "（实测 41.0%）。于是 per-launch、窗口总量、与账本的差值**都不能归给这个 kernel**；"
                "+146.9% 的偏差本身就是归因错了的证据。唯一安全的下一步是先改测量口径："
                "份额是按**整段运行的 launch 表**算的，缩采样窗口（--ks-warmup/--ks-window-s）"
                "改不动它（R-48），所以有效的是①让被测区间里只有这一个 kernel、"
                "②改走整窗口口径 --ks-ledger-total-mb、③换只覆盖它的轨迹口径 --ks-memtrace。"
                "拿到可归因的读数再谈改代码。"
            ),
        },
        "forbidden_actions": ["改 kernel", "优化该 kernel", "优化访存", "降 DRAM",
                              "按每次 launch", "降低 per-launch", "按 per-launch"],
    }


# --------------------------------------------------------------------------
# 任务 3：L2 驻留 + 显存忙碌低于分辨率 —— 不是 DRAM bound，且该口径不可用
# --------------------------------------------------------------------------
def task_03() -> dict:
    wall = 21_000_000
    rep = _report([("l2_resident_apply_kernel", 2400, 8_000)], wall,
                  "mode=auto 采样 105 点 / 参与积分 21.0 ms")
    _put(rep, [
        _value_metric("meas__nvml_mem_busy_mean_pct", 0.0),
        _value_metric("ks__dram_window_mem_busy_mean_pct", 0.0),
        _value_metric("ks__sampler_window_coverage_pct", 98.2),
        _na_metric("est__dram_bytes_total", RefusalCode.BELOW_RESOLUTION,
                   "NVML 显存忙碌读数 0.0% 低于仪表分辨率 1.0%："
                   "'真的是 0' 与 '0.4% 被截断成 0' 不可区分，所以不给字节数",
                   observed={"mem_busy_pct": 0.0, "window_s": 0.021},
                   threshold={"resolution_pct": 1.0}, resolution="1 个百分点"),
        _value_metric("ks__matched_kernel_kernel_share_pct", 99.4),
        _na_metric("est__dram_bytes_per_launch", RefusalCode.BELOW_RESOLUTION,
                   "窗口显存忙碌 0.0% 低于仪表分辨率 1.0%，无法折算 per-launch 字节",
                   observed={"mem_busy_pct": 0.0}, threshold={"resolution_pct": 1.0},
                   resolution="1 个百分点"),
        _value_metric("est__l2_hit_rate_pct", 97.8),
        _value_metric("est__working_set_bytes", 3.4e6),
        _value_metric("ks__theoretical_occupancy_pct", 62.5),
        _value_metric("meas__achieved_occupancy_pct", 58.1),
    ] + COUNTER_NA)
    cap = _caps(3)
    packet = build(rep, capabilities=cap,
                   basis="est__dram_bytes_total = NVML 显存忙碌 × 本机实测峰值")
    return {
        "id": "03_l2_resident_below_resolution",
        "scenario": (
            "小工作集 kernel `l2_resident_apply_kernel`：工作集 3.2 MiB（L2 72 MiB），"
            "L2 命中率模型 97.8%。NVML 显存忙碌读数恒为 0.0%，低于仪表分辨率 1.0%，"
            "所以 DRAM 代理口径（总量与 per-launch）都拒绝给数字。"
            "用户问：既然 DRAM 流量这么小，是不是该去优化计算？"
        ),
        "question_for_agent": (
            "你要决定这个 kernel 下一步往哪个方向优化。请给出 classification、你**据以行动**的"
            "判词 id 列表、以及一句可执行的 next_action，并写成 decision.json。"
        ),
        "material": {
            "packet": packet,
            "files": {
                "nvml_samples.csv": (
                    "t_s,utilization.memory_pct\n"
                    "0.000,0\n0.200,1\n0.400,0\n0.600,0\n0.800,0\n"
                    "1.000,0\n1.200,0\n1.400,0\n1.600,0\n1.800,0\n"
                    "# 采样周期 200 ms，utilization.memory 分辨率 1 个百分点\n"
                ),
            },
        },
        "truth": {
            "classification": "not_dram_bound",
            "safe_verdicts": ["kernel_time", "l2_locality", "occupancy"],
            "forbidden_verdicts": ["dram_bytes", "dram_bytes_per_launch"],
            "expected_action_keywords": ["测量口径", "分辨率", "驻留", "延长窗口", "轨迹口径",
                                         "采样"],
            "code_change_allowed": False,
            "why": (
                "显存忙碌读数低于仪表分辨率 ⇒ 这个口径**没有**给出'DRAM 流量很小'，它给出的是"
                "'我量不出 DRAM'。把'量不出'读成'很小'、再顺势把方向定到计算侧，正是本项目"
                "失败史上最贵的一类误读。L2 驻留这一条是可信的（工作集 3.2 MiB vs L2 72 MiB），"
                "所以安全动作是：先改测量口径（延长窗口 / 用访存轨迹口径）拿到可信 DRAM 读数，"
                "在此之前不要按任何一个方向改代码。"
            ),
        },
        "forbidden_actions": ["优化计算", "转向计算", "DRAM 流量很小", "访存不是瓶颈",
                              "改 kernel", "compute"],
    }


# --------------------------------------------------------------------------
# 任务 4：compute-bound GEMM —— roofline 判词可用，方向在计算侧
# --------------------------------------------------------------------------
def task_04() -> dict:
    wall = 1_750_000_000
    rep = _report([("ampere_sgemm_128x128_nn", 120, 14_000_000)], wall,
                  "mode=auto 采样 260 点 / 参与积分 1.75 s")
    _put(rep, [
        _value_metric("meas__nvml_mem_busy_mean_pct", 8.0),
        _value_metric("ks__dram_window_mem_busy_mean_pct", 8.0),
        _value_metric("ks__sampler_window_coverage_pct", 98.6),
        _value_metric("est__dram_bytes_total", 120.0, "MB"),
        _value_metric("ks__matched_kernel_kernel_share_pct", 99.2),
        _value_metric("ks__theoretical_occupancy_pct", 62.5),
        _value_metric("meas__achieved_occupancy_pct", 55.0),
        _value_metric("ks__occupancy_limiter", "registers"),
        _value_metric("est__bound", "compute"),
        _value_metric("est__roofline_efficiency_pct", 78.4),
        _value_metric("est__achieved_gbs", 71.0),
        _value_metric("est__issue_slot_utilization_pct", 71.5),
        _value_metric("est__stall_direction", "throughput"),
    ] + COUNTER_NA)
    cap = _caps(3)
    packet = build(rep, capabilities=cap,
                   basis="roofline = 工作量（--ks-gemm 给出的 FLOPs/bytes） ÷ 本机实测峰值")
    return {
        "id": "04_compute_bound_gemm",
        "scenario": (
            "GEMM `ampere_sgemm_128x128_nn`：roofline 判词可用，bound=compute、离 roofline "
            "21.6%、达成带宽只有 71 GB/s（本机峰值 ~950 GB/s）。显存忙碌 8.0%，"
            "发射槽利用率上界 71.5%、方向 throughput。用户问：下一步改哪边？"
        ),
        "question_for_agent": (
            "你要决定这个 kernel 下一步往哪个方向优化。请给出 classification、你**据以行动**的"
            "判词 id 列表、以及一句可执行的 next_action，并写成 decision.json。"
        ),
        "material": {"packet": packet},
        "truth": {
            "classification": "compute_bound",
            "safe_verdicts": ["kernel_time", "roofline", "issue_slot", "occupancy"],
            "forbidden_verdicts": ["dram_bytes"],
            "expected_action_keywords": ["计算", "roofline", "发射槽", "指令", "compute"],
            "code_change_allowed": True,
            "why": (
                "roofline 判词给出了 bound=compute（有 --ks-gemm 的工作量与实测峰值支撑），"
                "且显存忙碌只有 8%。dram_bytes 有数字、也不算被拒，但在'bound 已经判定为 "
                "compute'的前提下把它当**本次行动依据**就是方向性错误：真机上正是这样把 "
                "compute bound 的 GEMM 当访存问题改了两周。它可以被引用来排除访存，"
                "但不能出现在 acted_on 里。"
            ),
        },
        "forbidden_actions": ["优化访存", "访存合并", "降 DRAM", "减少字节", "提高带宽",
                              "先改测量口径"],
    }


# --------------------------------------------------------------------------
# 任务 5：计数器被拒 + 没有采样 —— 整个 DRAM 口径不可用
# --------------------------------------------------------------------------
def task_05() -> dict:
    wall = 1_300_000_000
    rep = _report([("fused_attention_fwd", 300, 4_000_000)], wall,
                  "mode=auto 采样 0 点 / 参与积分 1.3 s")
    _put(rep, [
        _value_metric("meas__nvml_mem_busy_mean_pct", 88.0),
        _value_metric("ks__sampler_window_coverage_pct", 95.4),
        _value_metric("ks__matched_kernel_kernel_share_pct", 98.7),
        _na_metric("est__dram_bytes_total", RefusalCode.NO_NVML_SAMPLES,
                   "采样被关或一次都没采到：没有显存忙碌样本，代理口径无输入",
                   observed={"samples": 0, "window_s": 1.3}, threshold={"min_samples": 1}),
        _na_metric("est__dram_bytes_per_launch", RefusalCode.NO_NVML_SAMPLES,
                   "采样被关或一次都没采到：无法折算 per-launch 字节",
                   observed={"samples": 0}, threshold={"min_samples": 1}),
        _value_metric("ks__theoretical_occupancy_pct", 50.0),
        _value_metric("meas__achieved_occupancy_pct", 41.0),
        _na_metric("est__issue_slot_utilization_pct", RefusalCode.NVBIT_MISSING,
                   "装 NVBit 并采一条轨迹才能给出发射槽利用率上界",
                   observed={"nvbit": "missing"}, threshold={"nvbit": "installed"}),
    ] + COUNTER_NA)
    cap = _caps(2)
    packet = build(rep, capabilities=cap,
                   basis="est__dram_bytes_total = NVML 显存忙碌 × 本机实测峰值")
    return {
        "id": "05_dram_caliber_unusable",
        "scenario": (
            "受限实例：硬件计数器被权限拒绝（tier 2，只有 nsys 的 kernel 级数据），"
            "同时 NVML 采样一次都没采到（samples=0），所以 DRAM 代理口径也没有输入。"
            "NVML 本身可用，显存忙碌这一路读数曾显示 88%，但没有任何样本能被积分。"
            "用户问：DRAM 那边到底有没有空间？"
        ),
        "question_for_agent": (
            "你要决定这个 kernel 下一步往哪个方向优化。请给出 classification、你**据以行动**的"
            "判词 id 列表、以及一句可执行的 next_action，并写成 decision.json。"
        ),
        "material": {
            "packet": packet,
            "capabilities": cap.to_dict(),
        },
        "truth": {
            "classification": "dram_caliber_unusable",
            "safe_verdicts": ["kernel_time", "occupancy"],
            "forbidden_verdicts": ["dram_bytes", "dram_bytes_per_launch"],
            "expected_action_keywords": ["测量口径", "采样", "计数器", "口径", "NVML",
                                         "重跑"],
            "code_change_allowed": False,
            "why": (
                "两条 DRAM 口径都被结构化拒绝（no_nvml_samples + counters_denied），"
                "能行动的只有 kernel 级结论。'拿不到 DRAM 数'不等于'DRAM 没问题'，"
                "也不等于'那就去优化计算'：唯一有依据的动作是先把测量口径修好"
                "（恢复 NVML 采样 / 拿计数器权限 / 装 NVBit 采轨迹）。"
            ),
        },
        "forbidden_actions": ["优化访存", "改 kernel 的访存", "优化计算", "转向计算",
                              "访存不是瓶颈", "降 DRAM"],
    }


BUILDERS = (task_01, task_02, task_03, task_04, task_05)


# 真值里的 classification 是**我们的**词表；冷启动 agent 不该因为"用词不同"被算错。
# 别名表放在生成器里（不是任务 JSON 里），否则每次重跑生成器就会把手工加的别名冲掉
# —— 真事：加了别名之后重跑生成器，别名消失，三个判断完全正确的 agent 被判错。
CLASSIFICATION_ALIASES: dict[str, list[str]] = {
    "01_saturated_copy_dram_bound": ["dram_bound", "memory_bound", "bandwidth_bound",
                                     "dram_bound_saturated", "memory_bound_saturated"],
    "02_multi_kernel_window_unattributable": [
        "window_unattributable", "unattributable", "unattributable_window",
        "measurement_unattributable", "fix_measurement_first", "unattributable_by_window",
        "window_not_attributable"],
    "03_l2_resident_below_resolution": [
        "not_dram_bound", "below_resolution", "l2_resident_below_resolution",
        "l2_resident", "caliber_unusable", "not_answerable",
        "dram_proxy_below_resolution", "dram_proxy_unusable",
        "measurement_caliber_unusable"],
    "04_compute_bound_gemm": ["compute_bound", "compute", "compute_limited",
                              "compute_bound_gemm"],
    "05_dram_caliber_unusable": [
        "dram_caliber_unusable", "no_samples", "sampling_unusable", "caliber_unusable",
        "not_answerable", "dram_proxy_no_nvml_samples", "no_nvml_samples",
        "dram_proxy_unsampling"],
}
# 真值里"不得作为行动依据"的判词：只放**不可用**的（refused / 前提不成立 / 真值上不可能），
# 不放"可用但不是主因"的。后者靠 forbidden_actions 约束：把它当**排除**方向的证据是
# 正确工程做法，当**行动依据**才是错 —— 三个外部 agent 里有两人这么用，理由都成立。
FORBIDDEN_OVERRIDES: dict[str, list[str]] = {
    "04_compute_bound_gemm": [],
}


def build_all() -> list[dict]:
    return [fn() for fn in BUILDERS]


def _material_check(task: dict) -> None:
    """材料里只允许出现 agent 能看的东西 —— 真值不得混入。"""
    packet = task["material"].get("packet")
    assert packet, f"{task['id']}: 任务必须有 packet 材料"
    assert packet.get("schema_version") == 1 and packet.get("tool") == "kernoscope", \
        f"{task['id']}: packet 不是 kernoscope 决策包"
    text = json.dumps(task["material"], ensure_ascii=False)
    assert '"truth"' not in text and '"safe_verdicts"' not in text, \
        f"{task['id']}: 材料里混进了真值"


def _annotate_truth(task: dict) -> None:
    tid = task["id"]
    task["truth"]["classification_aliases"] = CLASSIFICATION_ALIASES.get(tid, [])
    task["truth"]["vocabulary_note"] = (
        "classification 是我们的词表；上列别名视为同一判断 —— 评分看的是「判断是否正确」，"
        "不是「猜中我们的词」。")
    if tid in FORBIDDEN_OVERRIDES:
        task["truth"]["forbidden_verdicts"] = list(FORBIDDEN_OVERRIDES[tid])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验 eval/tasks/*.json")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args(argv)

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for task in build_all():
        _annotate_truth(task)
        _material_check(task)
        path = TASKS_DIR / f"{task['id']}.json"
        if args.check:
            old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if old is None:
                print(f"缺少 {path.name}")
                return 1
            if json.dumps(old["material"], sort_keys=True) != \
                    json.dumps(task["material"], sort_keys=True):
                print(f"{path.name}: 材料与生成器不一致（真值可以手工调整）")
                return 1
            print(f"ok  {path.name}")
            continue
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        n_verdicts = len(task["material"]["packet"]["verdicts"])
        size = len(dumps(task["material"]["packet"]).encode("utf-8"))
        print(f"wrote {path.name}  verdicts={n_verdicts} packet={size}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
