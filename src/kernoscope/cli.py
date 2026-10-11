"""kernoscope CLI —— 与 ncu 用法兼容。

选项按四类处理（兼容面与自有扩展分开，见 README 的「与 ncu 的关系」）：
  ① 真实现：kernel 过滤 / launch 选择 / 输出与格式类
  ② 映射实现：--set / --section / --metrics → 我们的数据源，缺则标 unavailable
  ③ 接受但空转：--replay-mode / --cache-control / --clock-control / --pm-sampling-* 等
  ④ 自有扩展：--ks-*（数据源、标定、workload、synthetic）

`argv[0]` 为 `ncu` 时自动进入兼容模式（供 `ln -s kernoscope ncu` 使用）。

退出码：
  0 正常；2 --strict 下**请求面**拿不到数据（供 CI 卡门）；1 运行错误。

  `--strict` 覆盖的请求面：空转的 ncu 开关、`--metrics` 请求了不可得的指标、
  `--set`/`--section` 请求了计数器 section、`--ks-ledger-*` 请求了账本对拍、以及环境缺件（降级跑）。
  **不覆盖**：低覆盖率导致 DRAM 代理口径不可用（那是"测量条件"不是"请求面"）。
  要卡"DRAM 数字必须在"，请同时给 `--ks-ledger-mb`（或 --set full）。
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .analyze.roofline import MachinePeaks, gemm_workload
from .backends import cubin as cubin_backend
from .backends import nsys as nsys_backend
from .backends.nvml import Sampler
from .refusals import Refusal, RefusalCode
from .report import DeviceInfo, KernelLaunch, KernelReport, Report
from .render import render_csv, render_launch_csv, render_perfetto, render_text
from .sections import (COUNTER_SECTIONS, KNOWN_COUNTER_METRICS, BuildOptions,
                       build_cost_attribution_section, build_cost_section,
                       build_device_kernel, build_dram_section, build_kernel_sections,
                       build_prescription_section, counter_section_reason,
                       requested_sections)

PROG = "kernoscope"

NOOP_FLAGS = [
    "--replay-mode", "--app-replay-mode", "--app-replay-buffer", "--app-replay-match",
    "--cache-control", "--clock-control", "--pm-sampling-buffer-size", "--pm-sampling-interval",
    "--pm-sampling-max-passes", "--warp-sampling-buffer-size", "--warp-sampling-interval",
    "--warp-sampling-max-passes", "--mps", "--mps-num-clients", "--mps-timeout",
    "--pipeline-boost-state", "--preload-library", "--import-sass", "--import-source",
]

# 子命令的一句话说明 —— **顶层 `--help` 从这里生成**。
#
# 为什么需要它：一次易用性实测发现顶层 `--help` 有 12,993 B，而 `route`/`capabilities`/
# `gate`/`skill` 各出现 **0 次** —— 四个核心入口在帮助里根本不存在，陌生人（和 agent）
# 只能靠读 README 才知道它们。argparse 的 subparsers 是更彻底的修法，但那会改动
# `--` 之后的目标命令语义；在帮助里列出来是零风险的等价修复。
# `tests/test_cli.py` 断言这里的键集合 == `SUBCOMMANDS`（加子命令不写说明就红）。
SUBCOMMAND_HELP: dict[str, str] = {
    "doctor": "这台机器能拿到什么、拿不到什么、为什么、备选是什么",
    "capabilities": "三态能力矩阵 + 补救动作与代价（`--brief` 给 agent，约 1 KB）",
    "route": "先路由：这个问题该用哪个口径、要付什么代价（只出计划，不测量）",
    "occupancy": "从探针记录重建「实际达到的 occupancy」",
    "memtrace": "把 NVBit 访存轨迹折成字节口径（含截断/口径不成立的拒绝）",
    "selftest": "用已知答案负载算本机误差带与最小可分辨差异（MDD）",
    "diff": "两次运行对比：improved / regressed / indistinguishable / incomparable",
    "gate": "CI 门禁：对决策包下断言，不达标就红（拿不到数据也算不达标）",
    "mcp": "MCP（stdio）：只给没有 shell 的 agent runtime 用（可选、零常驻）",
    "skill": "把 skill 装进 agent 的上下文（默认只打印计划，`--apply` 才写盘）",
    "issue": "把一次采集打包成可提交的 issue 材料",
}


def _subcommand_help_block() -> str:
    """按可读顺序（本字典的插入顺序）列出子命令；集合与 `SUBCOMMANDS` 由测试对拍。"""
    parts = ["子命令（每个都有 --help；`--json` 给机器读）:"]
    parts += [f"  kernoscope {name:<13} # {SUBCOMMAND_HELP[name]}" for name in SUBCOMMAND_HELP]
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Kernoscope - provenance-honest CUDA kernel profiler for locked-down GPUs "
                    "(ncu-compatible CLI). Every number carries a provenance label; metrics that "
                    "need hardware performance counters are reported as n/a with a reason.",
        epilog="示例:\n"
               "  kernoscope doctor                                   # 先看这台机器能拿到什么\n"
               "  kernoscope --set basic -k regex:gemm -c 3 -o rep -- python train.py\n"
               "  kernoscope -i rep.ksreport.json --csv\n"
               "  kernoscope --ks-synthetic --set detailed | head -50\n"
               "\n" + _subcommand_help_block() + "\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("ncu-compatible: selection")
    g.add_argument("-k", "--kernel-name", default=None,
                   help="kernel 过滤：'regex:<pat>' 或精确名（ncu 兼容）")
    g.add_argument("--kernel-name-base", default="demangled",
                   choices=["demangled", "mangled", "function"], help="(accepted)")
    g.add_argument("-s", "--launch-skip", type=int, default=0, help="跳过每个 kernel 的前 N 次 launch")
    g.add_argument("--launch-skip-before-match", type=int, default=0,
                   help="匹配开始前先跳过的 launch 数（全局）")
    g.add_argument("-c", "--launch-count", type=int, default=0,
                   help="每个 kernel 最多报告的 launch 数（0=全部）")
    g.add_argument("--target-processes", default="all", choices=["all", "application-only"])
    g.add_argument("--devices", default=None,
                   help="只报告这些设备上的 launch（逗号分隔的 deviceId；默认全部）")

    o = p.add_argument_group("ncu-compatible: output")
    o.add_argument("-o", "--export", default=None, help="输出基名（写 <base>.ksreport.json）")
    o.add_argument("-f", "--force-overwrite", action="store_true")
    o.add_argument("--csv", action="store_true", help="以 CSV 输出（含 provenance 列）")
    o.add_argument("--launch-csv", action="store_true",
                   help="以 CSV 输出**每个 launch 一行**（grid/block/时长），形状扫描直接可用")
    o.add_argument("--print-summary", default="none", choices=["none", "per-kernel"])
    o.add_argument("--print-details", default="all", choices=["all", "header"],
                   help="all=铺开所有指标（默认）；header=只打印 kernel/section 头")
    o.add_argument("--print-metric-name", default="name", choices=["name", "full", "base"],
                   help="name=指标名（默认）；full=带 section 前缀；base=同 name"
                        "（去掉 est__/meas__ 前缀会隐藏出处，本工具不做）")
    o.add_argument("--print-units", default="auto", choices=["base", "auto"],
                   help="base=原始单位；auto=自适应换单位（MB→GB/TB，默认）")
    o.add_argument("--no-case", action="store_true",
                   help="不写案件目录（默认在 -o <前缀> 旁写 <前缀>.case/：manifest/env/"
                        "journal/next.md，供跨会话继续与交接）")
    o.add_argument("--page", default="details", choices=["details", "raw", "source"],
                   help="details=默认；raw=额外打印每个指标的数据来源；"
                        "source=同 raw（SASS/源码对照不可得，会给出说明）")
    o.add_argument("-i", "--import", dest="import_report", default=None,
                   help="回读我们自己的 .ksreport.json（对标 ncu -i）")
    o.add_argument("--list-sets", action="store_true", help="列出 --set 可选面板")
    o.add_argument("--quiet", action="store_true")
    o.add_argument("--log-file", default=None)
    o.add_argument("--check-exit-code", action="store_true")
    o.add_argument("-v", "--version", action="version", version=f"{PROG} {__version__}")

    m = p.add_argument_group("ncu-compatible: metrics（映射实现，缺则 n/a）")
    m.add_argument("--set", dest="set_name", default="basic",
                   choices=["basic", "detailed", "full"])
    m.add_argument("--section", action="append", default=[], help="显式指定 section（可重复）")
    m.add_argument("--metrics", action="append", default=[], help="指标名（逗号分隔，可重复）")

    n = p.add_argument_group("ncu-compatible: accepted but no-op（--strict 下报错）")
    for flag in NOOP_FLAGS:
        n.add_argument(flag, default=None, help=argparse.SUPPRESS)

    k = p.add_argument_group("kernoscope extensions")
    k.add_argument("--ks-sqlite", default=None, help="直接读 nsys 导出的 sqlite（跳过采集）")
    k.add_argument("--ks-nsys-rep", default=None, help="读已有 .nsys-rep（自动 export sqlite）")
    k.add_argument("--ks-cubin", action="append", default=[],
                   help="cubin/.so 路径：静态资源与 SASS 混合（可重复）")
    k.add_argument("--ks-calibration", default=None, help="machine-profile.json（本机实测峰值）")
    k.add_argument("--ks-peak-tflops", type=float, default=None, help="覆盖计算峰值")
    k.add_argument("--ks-peak-gbps", type=float, default=None, help="覆盖带宽峰值")
    k.add_argument("--ks-gemm", default=None, help="GEMM 形状 M,N,K（算 FLOPs/bytes）")
    k.add_argument("--ks-flops", type=float, default=None, help="每次 launch 的 FLOPs")
    k.add_argument("--ks-bytes", type=float, default=None, help="每次 launch 搬运字节数")
    k.add_argument("--ks-dtype", default="bf16", help="精度（影响峰值与 itemsize）")
    k.add_argument("--ks-achieved-occupancy", default=None,
                   help='探针输出的 JSON: {"<kernel>": {"occupancy_pct":.., ...}}')
    k.add_argument("--ks-synthetic", action="store_true",
                   help="用合成数据跑通全流程（无需 GPU，供 CI/演示；会明确标注 synthetic）")
    k.add_argument("--ks-no-nvml", action="store_true", help="不做设备级采样")
    k.add_argument("--ks-perfetto", default=None, help="Perfetto trace 输出路径（.json）")
    k.add_argument("--ks-sm-count", type=int, default=0, help="覆盖 SM 数量")
    k.add_argument("--ks-ledger-mb", type=float, default=None,
                   help="账本预测的 DRAM 字节数（MB，**每次 launch** 口径）：报告里按同基给出相对误差")
    k.add_argument("--ks-ledger-total-mb", type=float, default=None,
                   help="账本预测的整窗口 DRAM 字节数（MB，总量口径）：与整窗口估算对拍")
    k.add_argument("--ks-ledger-range-mb", default=None,
                   help="账本给出的区间 LOW,HIGH（MB）：报告里给出估算落在区间的哪一侧")
    k.add_argument("--ks-ledger-basis", default="auto", choices=["auto", "nvml", "trace"],
                   help="账本对拍用哪条口径：auto=能用 NVML 代理就用它，否则退回访存轨迹；"
                        "nvml=只用 NVML 代理；trace=只用轨迹（两条口径不混用）")
    k.add_argument("--ks-window", default="auto", choices=["auto", "full"],
                   help="采样窗口：auto=按首个/末个 kernel 活动自动裁到稳态段（默认，"
                        "避免被 import torch / JIT 的 9~10 秒稀释）；full=用整段采样窗")
    k.add_argument("--ks-warmup", type=float, default=0.0,
                   help="从首个采样点起再丢掉 N 秒（跳过启动/JIT；与 --ks-window 叠加）")
    k.add_argument("--ks-window-s", type=float, default=None,
                   help="只保留从采样起点算起的 N 秒窗口（成本可预算）")
    k.add_argument("--ks-json-no-launches", action="store_true",
                   help="JSON 里不写逐 launch 数组：报告体积与 iteration 数解耦；"
                        "需要逐 launch 数据时用 --launch-csv")
    k.add_argument("--ks-memtrace", default=None,
                   help="NVBit 访存轨迹：给出 sectors/request、合并效率、L1/L2 命中率模型"
                        "与 trace 口径的 per-launch DRAM 字节（支持 ks-memtrace v1 与 NVBit "
                        "官方 mem_trace 输出，自动判别）")
    k.add_argument("--ks-l2-bytes", type=int, default=0, help="覆盖 L2 容量（字节）")
    k.add_argument("--ks-max-accesses", type=int, default=0,
                   help="访存轨迹缓存模型最多分析多少次访问（0=默认 4e6）。"
                        "超过上限时会拒绝给 DRAM 字节而不是给一个偏小的值")
    k.add_argument("--ks-instr-hist", default=None,
                   help="NVBit 官方 opcode_hist 的 stdout：给出动态指令数、"
                        "发射槽利用率**上界**与瓶颈方向（SchedulerStats/WarpStateStats）")
    k.add_argument("--ks-clock-mhz", type=float, default=None,
                   help="发射槽上界用的 SM 时钟（默认取 NVML 实测均值）")
    k.add_argument("--ks-keep-intermediate", action="store_true",
                   help="保留 nsys 中间产物（.nsys-rep/.sqlite）；给了 -o 时默认保留在其旁边")
    k.add_argument("--packet", default=None,
                   help="写出决策包 JSON（判词/前提/未测到/代价，默认 <4 KB）；"
                        "`-` 表示写到 stdout。比完整报告小两个数量级，给 agent 用")
    k.add_argument("--screen", default=None, choices=["auto", "only", "off"],
                   help="结论优先的一屏。**默认（不给）**：在终端上只打一屏（人读），"
                        "管道/重定向/CSV 里仍然是一屏 + 完整报告（机器格式一字不变）。"
                        "显式 auto=总是铺在完整报告前面；only=只打一屏；off=只打完整报告。"
                        "与 --packet 同一证据模型（同一份判词）")
    k.add_argument("--ks-selfcheck", default=None,
                   help="本机自检文件（kernoscope selftest 产出）：给判词带上误差带与 MDD，"
                        "diff 才能区分真改进与噪声")
    # ---- 成本账（C1）：把秒换算成钱与量。没给的输入我们拒绝，绝不按"典型值"猜 ----
    k.add_argument("--ks-tokens", type=float, default=None,
                   help="每次迭代(step)处理的 token 数（推理=一次 forward 的 token，"
                        "训练=一步的 token）：给了才有 tokens/s 与 ¥/Mtoken")
    k.add_argument("--ks-price-per-kwh", type=float, default=None,
                   help="电费口径：每 kWh 多少钱（能量实测，乘价格得钱）")
    k.add_argument("--ks-price-per-hour", type=float, default=None,
                   help="租约口径：这台机器每小时多少钱（与电费是两种载体，都给才相加）")
    k.add_argument("--ks-idle-power-w", type=float, default=None,
                   help="空闲功耗(W)：分摊到 kernel 前先把非计算能耗扣掉（不给则不扣，"
                        "空闲占比高时归因会偏大）")
    k.add_argument("--export-name-map", default=None, metavar="PATH",
                   help="写一份 ncu 名 → 我们给什么（或为什么给不出）的对照表（.json/.csv），"
                        "供下游脚本翻译；它是**声明**，不是数据替换")
    k.add_argument("--ks-currency", default="CNY",
                   help="价格单位标签（默认 CNY；我们只做标签，不做汇率换算）")
    k.add_argument("--strict", action="store_true",
                   help="请求了不可得数据时 exit 2（CI 卡门）；空转选项也视为错误")
    k.add_argument("--plan", action="store_true",
                   help="只打印**执行计划**（会起哪些进程、跑哪条命令、写哪些文件、读哪些输入、"
                        "哪些是第三方软件），不执行任何东西、不写文件；配合 --plan-json 给 agent")
    k.add_argument("--plan-json", action="store_true",
                   help="计划以 JSON 输出（机器读；含每个操作的 8 项 effects 与 4 项 annotations）")
    k.add_argument("--export-ncu-rep", default=None, metavar="PATH",
                   help="把替代口径**写成一个合法的 .ncu-rep**（官方只读 API 能读、"
                        "`ncu --import` / ncu-ui 能打开），同时写 PATH.ATTRIBUTION.txt 署名。"
                        "文件内署名（provider/comment）为空则拒绝生成；前提不成立的指标不写；"
                        "一个值都不剩时拒绝生成（宁可不生成，也不生成假的 ncu 报告）")
    k.add_argument("--repeat", type=int, default=1, metavar="N",
                   help="把整条命令跑 **N 次（进程级）**，给出跨运行分布与 MDD 下界 "
                        "（M5/N1）。不是 kernel 级重放：不插桩、不锁时钟、不碰其他租户，"
                        "也不控制 warm/cold 缓存 —— 所以真实噪声只会比这条下界更大。"
                        "喂给 `diff` 时与误差带**取最大值**，只能抬高门槛")
    k.add_argument("--html", default=None, metavar="PATH",
                   help="写一个**自包含**的单文件 HTML 证据单（给人看/截图/贴 issue：内联 CSS、"
                        "零外部 URL、零 JS、无服务器；每个数字带出处角标，拿不到的数字印 n/a + "
                        "拒绝码而不是 0）；`-` = stdout")
    k.add_argument("--report-md", default=None, metavar="PATH",
                   help="写出**人读的 per-kernel 报告**（Markdown，结构对齐生态里通用的 "
                        "per-kernel 报告模板：0 采集设置 → 1 头部数字（带 Source 列）→ "
                        "2 分维度 → 3 诊断 → 4 按影响排序的方向 → 5 置信与限制 → 6 复现）。"
                        "多 kernel 时导出最耗时的那个（用 -k 选别的）")
    k.add_argument("--ks-nvtx-phases", action="store_true",
                   help="从 --ks-sqlite/--ks-nsys-rep 的 NVTX 区间归因 kernel 时间"
                        "（训练/推理阶段各占多少）；没有 NVTX 数据时如实给 n/a + code")
    k.add_argument("--ks-ncu-rep", "--from-ncu-report", default=None, metavar="PATH",
                   dest="ks_ncu_rep",
                   help="读一个已有的 .ncu-rep 作**真值**（官方只读 API）：ncu__* 前缀 + exact "
                        "出真值，并与我们的替代口径逐条对拍（真值 vs 替代的相对误差）。"
                        "本机没有 Nsight Compute 的 Python 模块时报 ncu_report_unreadable + 处置")

    p.add_argument("program", nargs=argparse.REMAINDER, help="-- 之后是要 profile 的命令")
    return p


# --------------------------------------------------------------------------
# 采集与解析
# --------------------------------------------------------------------------
def _kernel_matches(l: KernelLaunch, pattern: str | None) -> bool:
    """匹配 demangled / short / mangled 三列 —— 用户不知道 nsys 把名字放在哪一列。"""
    if not pattern:
        return True
    candidates = [c for c in (l.demangled, l.short_name, l.mangled, l.name) if c]
    if pattern.startswith("regex:"):
        rx = pattern[len("regex:"):]
        return any(re.search(rx, c) for c in candidates)
    return any(pattern in c for c in candidates) or any(pattern == c for c in candidates)


def select_launches(launches: list[KernelLaunch], pattern: str | None,
                    skip: int, count: int, skip_before_match: int) -> list[KernelLaunch]:
    out: list[KernelLaunch] = []
    for idx, l in enumerate(sorted(launches, key=lambda x: x.start_ns)):
        if idx < skip_before_match:
            continue
        if not _kernel_matches(l, pattern):
            continue
        out.append(l)
    if skip or count:
        keep: list[KernelLaunch] = []
        per_kernel: dict[str, int] = {}
        for l in out:
            seen = per_kernel.get(l.name, 0)
            per_kernel[l.name] = seen + 1
            if seen < skip:
                continue
            if count and seen >= skip + count:
                continue
            keep.append(l)
        out = keep
    return out


def load_synthetic() -> list[KernelLaunch]:
    """合成数据：让无 GPU 的环境也能跑通全流程（明确标注 synthetic）。"""
    launches: list[KernelLaunch] = []
    t = 1_000_000_000
    for i in range(5):
        launches.append(KernelLaunch(
            name="synthetic_gemm", demangled="synthetic_gemm(float const*, float const*, float*)",
            start_ns=t, end_ns=t + 951_000, grid=(1048576, 1, 1), block=(256, 1, 1),
            registers_per_thread=96, static_smem_bytes=0, dynamic_smem_bytes=49152,
            local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0, correlation_id=i,
        ))
        t += 1_200_000
    for i in range(20):
        launches.append(KernelLaunch(
            name="synthetic_elementwise", demangled="synthetic_elementwise(float*)",
            start_ns=t, end_ns=t + 8_400, grid=(16384, 1, 1), block=(256, 1, 1),
            registers_per_thread=16, static_smem_bytes=0, dynamic_smem_bytes=0,
            local_mem_per_thread=0, stream_id=7, device_id=0, grid_id=0, correlation_id=100 + i,
        ))
        t += 20_000
    return launches


#: 采集**之前**必须存在的输入文件（第七轮 X-1④）：否则"文件不存在"会被采集错误盖住
#: （真机实测：`--ks-calibration /tmp/nope.json` 在有 nsys 的机器上先撞 sqlite 错误，
#: 于是"标定文件不存在"这件事从头到尾没被报出来）。
_MUST_EXIST_INPUTS: tuple[tuple[str, str], ...] = (
    ("ks_calibration", "标定文件"),
    ("ks_memtrace", "访存轨迹"),
    ("ks_instr_hist", "指令直方图"),
    ("ks_achieved_occupancy", "occupancy 注入 JSON"),
    ("ks_selfcheck", "自检/误差带报告"),
)


def _preflight_inputs(args: argparse.Namespace) -> None:
    """采集前把"输入文件在不在"查掉（X-1④）：输入错误不该被采集错误盖住。"""
    for attr, what in _MUST_EXIST_INPUTS:
        value = getattr(args, attr, None)
        if not value:
            continue
        path = str(value)
        if attr in ("ks_calibration", "ks_memtrace", "ks_instr_hist", "ks_selfcheck"):
            # 这几个既收路径也可能收 URL/内联文本，只有"看起来像路径"时才查
            if "://" in path or "\n" in path:
                continue
        if not pathlib.Path(path).exists():
            raise CliInputError(
                f"{what}不存在：{path}", hint="确认路径拼写与文件已经生成（怎么生成见 README）")


def _resolve_target(cmd: list[str]) -> None:
    """交给 nsys **之前**解析目标可执行文件（第七轮 X-1①）。

    为什么必须在采集前做：有 nsys 的机器上，`kernoscope doctr`（拼错的子命令被当成程序）
    会走 `nsys profile … doctr`，nsys 返回非零，于是旧实现报"nsys profile failed" +
    "装 nsys"的提示 —— 真实原因（这个程序不存在）一个字都没说，还把人送去装已装好的工具。
    这里先判一次，报的是与**无 nsys 路径逐字相同**的那句话。
    """
    prog = cmd[0]
    if os.sep in prog or (os.altsep and os.altsep in prog):
        p = pathlib.Path(prog)
        ok = p.exists() and os.access(prog, os.X_OK)
    else:
        ok = shutil.which(prog) is not None
    if not ok:
        raise CliInputError(f"目标命令无法执行：{prog} 不存在",
                            hint="检查拼写；用 `--` 把要运行的程序与它的参数放在最后")


def _target_command(program: Sequence[str]) -> list[str]:
    """把 `-- <cmd>` 后面的内容还原成 argv。

    `argparse.REMAINDER` 会保留一个单独的 `--`，而 shell 传进来时也可能是**一个**字符串
    `"-- python train.py"`（测试与某些封装就是这样调的）。降级路径要直接 exec 它，
    所以这两种形状都必须还原成 argv，而不是交给 nsys 去猜。
    """
    raw = [a for a in program if a != "--"]
    if len(raw) == 1 and raw[0].strip().startswith("-- "):
        return shlex.split(raw[0].strip()[3:])
    return raw


def collect(args: argparse.Namespace, log) -> tuple[list[KernelLaunch], dict[str, Any], int]:
    """返回 (launches, extra_meta, target_exit_code)。

    R-19：缺依赖/文件不存在时给**自家风格的错误 + 安装建议**，不能抛裸 traceback。
    W1：缺 nsys **不是**错误 —— 它只让我们退到设备级（NVML），所以这里返回
    `degraded="device-only"` 并照常跑目标命令，让报告给出「能拿到什么 + 缺什么会多出什么」。
    """
    meta: dict[str, Any] = {}
    rc = 0
    if args.ks_synthetic:
        meta["synthetic"] = True
        return load_synthetic(), meta, 0
    if args.ks_sqlite:
        return nsys_backend.parse_kernels(args.ks_sqlite), meta, 0
    if args.ks_nsys_rep:
        db = nsys_backend.sqlite_from_rep(args.ks_nsys_rep)
        meta["nsys_sqlite"] = db
        return nsys_backend.parse_kernels(db), meta, 0

    cmd = _target_command(args.program)
    if not cmd:
        # 走 `CliInputError` 而不是 `log_error(...) + raise SystemExit(1)`：
        # 后者是**在进程内调用 `main()` 时抛异常**、只有 console-script 才变成 rc=1，
        # 与其余输入错误（同一条 `error:` + 退出码通路）不一致。这条路径正是
        # 「无参数 / 忘记 `--` / 拼错子命令」的落点，也是最该稳的一条。
        raise CliInputError(
            "no target command：把它放在 `--` 之后"
            "（例如 `kernoscope --set basic -- python train.py`）；"
            "或用 --ks-sqlite/--ks-nsys-rep 读已有采集；没有 GPU 时用 --ks-synthetic")
    meta["target"] = " ".join(cmd)
    # X-1④：输入文件的存在性在采集**之前**判（否则会被采集错误盖住）
    _preflight_inputs(args)
    # X-1①：目标可执行文件也在采集之前解析（否则"程序不存在"会被讲成"nsys 失败"）
    _resolve_target(cmd)

    if not nsys_backend.find_nsys():
        # ---- W1：优雅降级。没有 nsys ≠ 没有信息 ----
        # 仍然把目标命令跑起来，好让 NVML 在这段时间里采设备级读数；报告会明确写出
        # 「kernel 级数据缺失」是哪一个 code，以及装上 nsys 之后会多出什么。
        meta["degraded"] = "device-only"
        ref = Refusal(
            RefusalCode.NSYS_MISSING,
            "没有 nsys：kernel 级数据（时长/launch 配置/occupancy 上界）本次不可得；"
            "本次报告只有设备级实测（NVML）",
            observed={"nsys": "not found"})
        meta["environment_refusals"] = [ref.to_dict()]
        if args.ks_no_nvml:
            log("warn: 没有 nsys 且 --ks-no-nvml 打开：本次不会有任何测量值；"
                "去掉 --ks-no-nvml 可拿到设备级读数")
            return [], meta, 0
        log("note: 没有 nsys —— 退到设备级模式（只用 NVML 采样），继续运行目标命令："
            f"{' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            rc = proc.returncode
            if rc != 0:
                log(f"warn: 目标命令退出码 {rc}；报告仍然是这台机器在运行期间的设备级读数")
        except OSError as e:
            # 与「没有目标命令」同一条通路：`CliInputError` ⇒ 一行 `error:` + rc=1，
            # 而不是只在 console-script 里才成立的 `raise SystemExit`。
            raise CliInputError(f"目标命令无法执行：{e}", hint=_io_remedy(e)) from e
        return [], meta, rc

    keep = bool(getattr(args, "ks_keep_intermediate", False))
    if args.export:
        base = args.export[:-len(".json")] if args.export.endswith(".json") else args.export
        out_prefix = base + "_nsys"
    else:
        tmpdir = tempfile.mkdtemp(prefix="kernoscope-")
        out_prefix = os.path.join(tmpdir, "kernoscope_run")
        meta["tmpdir"] = tmpdir
    rep_path = nsys_backend.run_profile(cmd, out_prefix, target_processes=args.target_processes)
    db = nsys_backend.export_sqlite(rep_path)
    meta["nsys_rep"] = rep_path
    meta["nsys_sqlite"] = db
    launches = nsys_backend.parse_kernels(db)
    if not keep and not args.export:
        shutil.rmtree(os.path.dirname(out_prefix), ignore_errors=True)
        meta.pop("tmpdir", None)
    return launches, meta, rc


def _collect_or_report(args: argparse.Namespace, log) -> tuple[list[KernelLaunch], dict[str, Any], int] | None:
    """采集，并把后端异常翻译成一行可读错误（R-19）。

    V-25：这里的两条错误**不受 `--quiet` 影响** —— 采集失败就没有报告可看，此时
    静默的 rc=1 比一行错误更难排查（坏 sqlite 就走这条路）。
    """
    try:
        return collect(args, log)
    except nsys_backend.TargetFailed as e:
        # X-1②：nsys 把**目标程序**的退出码透传回来 —— 先说目标，再说 nsys（如果有嫌疑）。
        print(f"error: 目标命令退出码 {e.rc}", file=sys.stderr)
        if e.rep_exists:
            print("hint: 采集文件仍然产出了，可以继续分析；如果目标是你的负载，先看它自己为什么退出。",
                  file=sys.stderr)
        else:
            print("hint: 并且没有产出 .nsys-rep —— 也可能是 nsys 自身失败"
                  "（目标是被 nsys 启动后再失败的，所以我们无法完全区分）。"
                  + (f"\nstderr 尾部：{e.detail}" if e.detail else ""), file=sys.stderr)
        return None
    except nsys_backend.NsysError as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: nsys 随 CUDA Toolkit 提供（或单独安装 Nsight Systems）。"
              "若已有 .nsys-rep / sqlite，可用 --ks-nsys-rep <file> 或 --ks-sqlite <db> 跳过采集；"
              "没有 GPU 时用 --ks-synthetic 跑通流程。", file=sys.stderr)
        return None
    except OSError as e:
        print(f"error: 采集失败：{e}", file=sys.stderr)
        print("hint: 检查路径是否存在与权限；--ks-sqlite/--ks-nsys-rep 需要可读的文件。",
              file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
SUBCOMMANDS: tuple[str, ...] = (
    "doctor", "occupancy", "memtrace", "issue", "capabilities", "selftest", "diff", "route",
    "skill", "gate", "mcp")


def subcommand_parser(name: str) -> Any:
    """某个子命令的 argparse 解析器（**唯一真源**）。

    `--help`、文档命令门禁、以及测试里的"这条文档命令真能跑吗"都从这里取。
    以前门禁是从子命令 `--help` 的**散文**里正则刮开关名，于是它对子命令**从不真的
    parse**：`diff` 上挂主命令的 `--packet`、`selftest` 上挂标定路径、`capabilities`
    上一个不存在的短开关，三条都被放行（验收报告第四轮 P1）。
    """
    from .capabilities_cmd import build_parser as capabilities_parser
    from .diff_cmd import build_parser as diff_parser
    from .doctor import build_parser as doctor_parser
    from .gate_cmd import build_parser as gate_parser
    from .issue_cmd import build_parser as issue_parser
    from .memtrace_cmd import build_parser as memtrace_parser
    from .occupancy_cmd import build_parser as occupancy_parser
    from .route_cmd import build_parser as route_parser
    from .selftest import build_parser as selftest_parser
    from .skill_cmd import build_parser as skill_parser
    from .mcp import build_parser as mcp_parser

    return {"doctor": doctor_parser, "occupancy": occupancy_parser,
            "memtrace": memtrace_parser, "issue": issue_parser,
            "capabilities": capabilities_parser, "selftest": selftest_parser,
            "diff": diff_parser, "route": route_parser, "skill": skill_parser,
            "gate": gate_parser, "mcp": mcp_parser}[name]()


def _SUBCOMMAND_ENTRY() -> dict[str, Any]:
    """子命令名 → 入口函数（延迟导入：`kernoscope --help` 不该拉起 torch/nsys 那套）。"""
    from .capabilities_cmd import capabilities_main
    from .diff_cmd import diff_main
    from .doctor import doctor_main
    from .issue_cmd import issue_main
    from .memtrace_cmd import memtrace_main
    from .occupancy_cmd import occupancy_main
    from .gate_cmd import gate_main
    from .route_cmd import route_main
    from .selftest import selftest_main
    from .skill_cmd import skill_main
    from .mcp import main as mcp_main_entry

    return {"doctor": doctor_main, "occupancy": occupancy_main, "memtrace": memtrace_main,
            "issue": issue_main, "capabilities": capabilities_main,
            "selftest": selftest_main, "diff": diff_main, "route": route_main,
            "skill": skill_main, "gate": gate_main, "mcp": mcp_main_entry}


# --------------------------------------------------------------------------
# 自家错误行（V-25 P2）
#
# 约定：用户输入/环境 I/O 类失败一律变成 `error: <人话>（<怎么修>）` 打到 stderr +
# 非零退出码，**绝不**把 Python traceback 甩给用户。参照物是本来就这么做的路径：
# `--ks-memtrace`（cli.py 里 try/except MemTraceError）、`issue/memtrace/occupancy --from`
# （子命令里 print(f"error: {e}", file=sys.stderr)）。本轮把漏网的收进来：
# 只读目录写产物、`--ks-calibration`/`-i`/`--ks-achieved-occupancy` 的坏 JSON、
# `--ks-sqlite` 的坏 sqlite。
#
# 边界（重要）：只翻译**输入与环境**异常。`ValueError`/`TypeError`/`KeyError` 这类
# 程序 bug 继续裸抛 —— 把 bug 说成"用户写错了"会让下一次排查更贵。
# --------------------------------------------------------------------------
class CliInputError(Exception):
    """用户输入/环境 I/O 错误：由 `main()` 翻成一行 `error:` + 退出码。"""

    def __init__(self, message: str, *, hint: str = "", rc: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.rc = rc

    def render(self) -> str:
        return f"error: {self.message}" + (f"（{self.hint}）" if self.hint else "")


def _io_remedy(exc: OSError) -> str:
    """按 errno 给一句"怎么修"（不是复述"哪里错了"）。"""
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return "目标位置不可写：换到可写目录（如 /tmp 下）或修正权限后重试"
    if exc.errno == errno.ENOENT:
        return "路径不存在或拼错：确认文件与父目录都已建好"
    if exc.errno == errno.EISDIR:
        return "这是一个目录：产物要写的是文件路径，不是目录"
    if exc.errno == errno.ENOSPC:
        return "磁盘空间不足：清理后重试"
    return "检查路径与读写权限后重试"


def _read_json_or_fail(path: Any, *, what: str, hint: str = "") -> Any:
    """读 JSON；读不了/不是合法 JSON 就抛 `CliInputError`（rc=1），不裸 traceback。"""
    p = str(path)
    try:
        text = Path(p).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise CliInputError(f"{what}不存在：{p}", hint=hint or _io_remedy(e)) from e
    except OSError as e:
        raise CliInputError(f"读不了{what} {p}：{e.strerror or e}",
                            hint=hint or _io_remedy(e)) from e
    except UnicodeDecodeError as e:
        raise CliInputError(f"{what} {p} 不是文本文件（{e}）",
                            hint=hint or "它应该是 JSON 文本，不是二进制") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise CliInputError(
            f"{what} {p} 不是合法 JSON：{e.msg}（第 {e.lineno} 行第 {e.colno} 列）",
            hint=hint or "确认文件没被截断/手改坏；用生成它的命令重新生成一份") from e


def _require_object(value: Any, path: Any, *, what: str, hint: str = "") -> dict:
    """JSON 合法但顶层不是对象（数组/数字/字符串）时也给自家错误。"""
    if not isinstance(value, dict):
        raise CliInputError(
            f"{what} {path} 的顶层应该是 JSON 对象，实际是 {type(value).__name__}",
            hint=hint or "用生成它的命令重新生成一份，别手改结构")
    return value


def _write_text_or_fail(text: str, path: Any, *, what: str, logged: list[str] | None = None,
                        hint: str = "") -> None:
    """写文本产物；失败抛 `CliInputError`（I/O 错 rc=1），不裸 traceback。"""
    p = str(path)
    try:
        _ensure_parent(p, logged)
        Path(p).write_text(text, encoding="utf-8")
    except OSError as e:
        raise CliInputError(f"写不了{what}：{p}（{e.strerror or e}）",
                            hint=hint or _io_remedy(e)) from e


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：把输入/环境类异常翻译成一行 `error:`，其余照旧抛出。

    窄兜底只收三类：`OSError`（读/写文件）、`json.JSONDecodeError`（坏 JSON）、
    `sqlite3.DatabaseError`（坏 sqlite）。`KeyboardInterrupt`/`SystemExit`/`MemoryError`
    以及 `ValueError`/`TypeError`/`KeyError` 这类程序 bug **不吞** —— 报文里也保留异常
    类型名，这样"用户写错了"与"我们写错了"在读日志时仍然分得开。
    """
    try:
        return _main(argv)
    except CliInputError as e:
        print(e.render(), file=sys.stderr)
        return e.rc
    except (OSError, json.JSONDecodeError, sqlite3.DatabaseError) as e:
        print(f"error: {type(e).__name__}: {e}"
              f"（这是输入/环境问题（文件路径、权限、格式）；确认命令里给的文件存在、可读、"
              f"格式正确后重试。若文件都没问题，那就是我们漏了这条路径，请带上下文报 bug）",
              file=sys.stderr)
        return 1


def _main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 子命令（不属于 ncu 兼容面）：**名字只有这一处**。文档命令门禁、--help、
    # 以及"哪些子命令存在"的检查都从这里读 —— 两处各写一份，新加一个子命令就会漏检
    # （真事：校验器只认 doctor/occupancy/memtrace/issue，`capabilities --brief` 这类
    #  合法命令反而被它判成不存在）。
    if argv and argv[0] in SUBCOMMANDS:
        return _SUBCOMMAND_ENTRY()[argv[0]](argv[1:])

    args = build_parser().parse_args(argv)
    compat_mode = os.path.basename(sys.argv[0] or "") == "ncu"

    logged: list[str] = []

    def log(msg: str) -> None:
        logged.append(msg)
        if not args.quiet:
            print(msg, file=sys.stderr)

    def log_error(msg: str) -> None:
        """`error:` 行**永远**打出来（不受 --quiet 影响）。

        为什么：`--quiet` 是"别啰嗦"，不是"别报错"。被吞掉的错误会让脚本层拿到 rc≠0
        却没有任何线索 —— 而 agent 恰恰只读 stderr。
        """
        logged.append(msg)
        print(msg, file=sys.stderr)

    # 模块级函数（`collect()`/`_attach_truth()`/`_export_substitute()`）拿不到这个闭包，
    # 它们走 `_log_error(log, …)`；挂上去才能保证"错误不受 --quiet 影响"这条纪律。
    log.log_error = log_error          # type: ignore[attr-defined]

    # ---------- 执行计划（A2）：只打印，不执行 ----------
    # 放在**所有采集之前**：`--plan` 的承诺是"没有任何副作用"，所以它必须在第一行采集
    # 之前就返回 —— 写在这里而不是 main 的开头，是因为要等 parse_args 完成（计划由参数决定）。
    if args.plan or args.plan_json:
        # 形状陷阱（V-18 报告里的第 1 点）：`kernoscope --plan skill --apply` 里的 `skill`
        # 是**位置参数**（要被运行的程序），不是子命令 —— 计划会说"运行目标 = skill --apply"，
        # 而把 `--plan` 去掉之后它真的会去 exec 它。语义差这么多，必须当场拒绝并给出写法，
        # 而不是给一份看起来正常、其实指错东西的计划。
        prog = list(getattr(args, "program", []) or [])
        if prog and prog[0] in SUBCOMMANDS and args.plan:
            hint = (f"`--plan` 是主命令的开关；子命令的计划要写成 "
                    f"`kernoscope {prog[0]} --help`（子命令自己声明它会写什么），"
                    f"或者直接用 `kernoscope {prog[0]} ...` 跑一次看输出。"
                    f"`kernoscope --plan {prog[0]} ...` 会被当成『计划运行程序 "
                    f"{prog[0]}{" " + " ".join(prog[1:]) if len(prog) > 1 else ""}』——"
                    f"那不是你想要的。")
            raise CliInputError(
                f"--plan 后面跟了子命令名 `{prog[0]}`：这是位置参数（要被运行的命令），"
                f"不是子命令。", hint=hint, rc=2)
        from .effects import plan_for, render_plan

        plan = plan_for(args, argv)
        if args.plan_json:
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(render_plan(plan))
        return 0

    # ---------- 显示开关的诚实性（B6：不允许"接受但静默忽略"）----------
    if args.print_metric_name == "base":
        log("warn: --print-metric-name base 会去掉 est__/meas__/ks__ 前缀，"
            "而那正是出处标签的一部分；本工具按 name 处理并保留前缀。")
    if args.page == "source":
        log("warn: --page source 退化为 raw（打印每个指标的 source）；"
            "SASS/源码逐行对照需要 --import-source，本工具不实现，"
            "对应的 ncu section 会以 n/a + 原因出现。")

    # ---------- 回读模式 ----------
    # `--repeat` 对"回读一份已有报告"没有意义（重跑的是采集，不是解析）：显式拒绝，
    # 不许静默忽略（静默忽略 = 承诺了 N 次却只做了一次）。
    if int(getattr(args, "repeat", 1) or 1) > 1 and getattr(args, "import_report", None):
        log_error("error: `--repeat` 只用于采集（要重复跑的是目标命令）；"
                  "`-i` 是回读已有报告，两者不能一起用")
        return 2
    if args.import_report:
        # V-25：`-i /nonexistent.json` 或坏 JSON 曾经是裸 FileNotFoundError/JSONDecodeError
        # （rc=1、无 error: 行）。这里按自家约定翻译。
        rep = Report.from_dict(_require_object(
            _read_json_or_fail(
                args.import_report, what="回读报告",
                hint="它应该是 `-o <前缀>` 写出的 .ksreport.json；路径写对，或先跑一次 "
                     "`kernoscope --ks-synthetic -o /tmp/rep` 生成一份"),
            args.import_report, what="回读报告"))
        used_noop_import = [f for f in NOOP_FLAGS
                            if getattr(args, f.lstrip("-").replace("-", "_"), None) is not None]
        problems = _strict_problems(rep, args, used_noop_import)
        if args.strict and problems:
            _report_strict(problems)
            return 2
        return _emit(rep, args, compat_mode, logged, packet_ctx={})

    if args.list_sets:
        from .sections import SET_SECTIONS

        print("Available sets and sections:")
        for name, secs in SET_SECTIONS.items():
            print(f"  {name:<9} {', '.join(secs)}")
        print("\nSections that require GPU performance counters are always reported as "
              "unavailable with a reason (never as substituted numbers).")
        return 0

    # ---------- M5/N1：进程级重复执行（`--repeat N`）----------
    # 放在 `--plan` **之后**（`--plan` 只打印、不执行，连重复也不跑）
    if int(getattr(args, "repeat", 1) or 1) > 1:
        return _repeat_path(args, argv)

    # ---------- 能力探测（W1/W5）----------
    # 便宜（只读 /proc + 多路径查工具，不跑 ncu 探测）但必须先做：它决定这次能给出什么，
    # 也决定要不要降级。做完缓存起来，组包（packet）与摘要共用同一份结论。
    from .capabilities import env_from_doctor, evaluate
    from .doctor import run_doctor

    cap = evaluate(env_from_doctor(run_doctor(probe_counters=False)))

    selfcheck = None
    if args.ks_selfcheck:
        try:
            selfcheck = json.loads(Path(args.ks_selfcheck).read_text())
        except (OSError, json.JSONDecodeError) as e:
            log(f"warn: --ks-selfcheck 读取失败（{e}）：本次判词不带误差带/MDD")

    # ---------- 采集 ----------
    sampler_ctx = Sampler(enabled=not args.ks_no_nvml)
    t0 = time.monotonic()
    with sampler_ctx as sampler:
        collected = _collect_or_report(args, log)
        if collected is None:
            return 1
        launches, meta, target_rc = collected
    wall_ns = int((time.monotonic() - t0) * 1e9)

    # ---------- 设备信息 ----------
    device = nsys_backend.device_info_from_smi()
    if args.ks_synthetic and not device.name:
        device = DeviceInfo(name="synthetic-device", compute_capability="8.9", sm_count=128,
                            l2_mib=72.0, max_sm_clock_mhz=3105.0, power_limit_w=450.0)
        device.driver = "n/a (synthetic)"
    sm_count = args.ks_sm_count or device.sm_count
    if not sm_count:
        try:
            import torch  # type: ignore

            sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
            device.sm_count = sm_count
            device.cuda_runtime = torch.version.cuda or ""
        except Exception:
            sm_count = 0
    if not device.l2_mib:
        try:
            import torch  # type: ignore

            device.l2_mib = torch.cuda.get_device_properties(0).L2_cache_size / 2 ** 20
        except Exception:
            pass

    # ---------- 峰值与 workload ----------
    peaks: MachinePeaks | None = None
    if args.ks_calibration:
        # V-25：标定文件读不了曾经是裸 FileNotFoundError（rc=1、无 error: 行）。
        prof = _require_object(
            _read_json_or_fail(
                args.ks_calibration, what="标定文件",
                hint="它应由 tools/machine-profile.py 生成（nvidia-smi 跑一次标定）；"
                     "不想标定就去掉 --ks-calibration，或改用 --ks-peak-tflops/--ks-peak-gbps "
                     "手工给峰值"),
            args.ks_calibration, what="标定文件",
            hint="它应是标定脚本写出的 JSON 对象（含 measured_dram/measured_matmul_tflops）")
        peaks = MachinePeaks.from_profile(prof, args.ks_dtype, profile_path=args.ks_calibration)
        if not peaks.caliber_ok:
            log(f"warn: 标定口径不可用 —— {peaks.caliber_note}")
        else:
            log(f"calibration: {peaks.source_label()}")

    if args.ks_peak_tflops or args.ks_peak_gbps:
        peaks = MachinePeaks.from_spec(
            args.ks_peak_tflops or (peaks.compute_tflops if peaks else 0.0),
            args.ks_peak_gbps or (peaks.bandwidth_gbs if peaks else 0.0),
            args.ks_dtype,
        )
        peaks.profile_path = args.ks_calibration or ""
        log("warn: 峰值被 --ks-peak-* 覆盖：来源不是本机标定，因此不用于 roofline 拐点判定")
    workload = None
    workload_note = ""
    if args.ks_gemm:
        try:
            m, n, k = (int(float(x)) for x in args.ks_gemm.replace("x", ",").split(","))
            workload = gemm_workload(m, n, k, args.ks_dtype)
            workload_note = f"user-provided GEMM {m}x{n}x{k} ({args.ks_dtype})"
        except Exception:
            log_error(f"error: bad --ks-gemm {args.ks_gemm!r}; expected M,N,K")
            return 1
    elif args.ks_flops and args.ks_bytes:
        workload = (float(args.ks_flops), float(args.ks_bytes))
        workload_note = f"user-provided FLOPs={args.ks_flops:g} bytes={args.ks_bytes:g}"
    elif args.ks_flops or args.ks_bytes:
        log_error("error: --ks-flops and --ks-bytes must be provided together")
        return 1

    # ---------- 静态后端 ----------
    cubin_res: dict[str, dict[str, int]] = {}
    cubin_sass = None
    for path in args.ks_cubin:
        if not os.path.exists(path):
            log(f"warn: --ks-cubin {path} not found, ignored")
            continue
        try:
            cubin_res.update(cubin_backend.res_usage(path))
            if cubin_sass is None:
                cubin_sass = cubin_backend.sass_mix(path)
        except Exception as e:  # noqa: BLE001
            log(f"warn: cuobjdump failed for {path}: {e}")

    achieved: dict[str, dict[str, float]] = {}
    if args.ks_achieved_occupancy:
        # V-25：注入文件读不了也不能裸 traceback（该文件由 `occupancy --write-json` 生成）。
        achieved = _require_object(
            _read_json_or_fail(
                args.ks_achieved_occupancy, what="occupancy 注入文件",
                hint="它应由 `kernoscope occupancy --from <probe> --write-json <file>` 生成；"
                     "确认路径写对，或去掉 --ks-achieved-occupancy"),
            args.ks_achieved_occupancy, what="occupancy 注入文件")

    # ---------- 过滤 ----------
    degraded = str(meta.get("degraded") or "")
    if not launches:
        if degraded:
            # W1：设备级模式下没有 kernel 记录是**预期**的，不是错误。报告仍然给出
            # 设备级实测 + 结构化拒绝（缺 nsys 会多出什么）。退出码 0。
            log("note: 本次没有任何 kernel 记录（设备级模式）；报告只含设备级读数与"
                "「装上 nsys 会多出什么」")
        else:
            ref = Refusal(RefusalCode.NO_LAUNCH_ACTIVITY,
                          "没有采到任何 kernel 活动：确认目标命令真的在 GPU 上跑 kernel，"
                          "且采集时带上了 CUDA activity（--trace=cuda）")
            log_error(f"error: {ref.message}")
            log(f"hint: {ref.remedy_or_default.action}"
                + (f" [{ref.remedy_or_default.command}]" if ref.remedy_or_default.command else ""))
            return 1
    if args.devices:
        try:
            want_dev = {int(x) for x in str(args.devices).replace(" ", "").split(",") if x != ""}
        except ValueError:
            log_error(f"error: --devices 需要逗号分隔的整数设备号，收到 {args.devices!r}")
            return 1
        before = len(launches)
        launches = [l for l in launches if l.device_id in want_dev]
        log(f"--devices {sorted(want_dev)}: 保留 {len(launches)}/{before} 条 launch 记录")
        if not launches:
            log_error(f"error: 这些设备上没有 kernel activity（可用 deviceId="
                f"{sorted({l.device_id for l in launches}) or '无'}）。")
            return 1
    matched = select_launches(launches, args.kernel_name, args.launch_skip,
                              args.launch_count, args.launch_skip_before_match)
    kernelless = bool(degraded) and not launches
    if not matched and not kernelless:
        names = sorted({l.demangled for l in launches})[:8]
        ref = Refusal(RefusalCode.NO_KERNEL_MATCH,
                      f"no kernels matched {args.kernel_name!r}；可用名字示例：{names}",
                      observed={"pattern": args.kernel_name or "",
                                "available_names": names})
        log_error(f"error: {ref.message}")
        log(f"hint: {ref.remedy_or_default.action}")
        return 1

    # ---------- R-46：窗口主导性所需的「谁在窗口里」----------
    # 逐 launch 的 CUPTI 时长一直在报告里，只是从前没有参与判定：
    # 于是一个窗口里顺序跑了四个 kernel 的负载被判成「可归因到单个 kernel」。
    # 注意 -c/-s 只限制**展示**，不能改变 per-launch 的分母：分母若被截断，窗口总量
    # 会被除到一个偏小的 launch 数上，误差同样是倍数级（与 D1 同一族）。
    matched_all = select_launches(launches, args.kernel_name, 0, 0,
                                  args.launch_skip_before_match)
    matched_ns = sum(max(0, l.end_ns - l.start_ns) for l in matched_all)
    total_kernel_ns = sum(max(0, l.end_ns - l.start_ns) for l in launches)
    other_ns = max(0, total_kernel_ns - matched_ns)
    matched_names = {l.display_name(args.kernel_name_base) for l in matched_all}
    other_names = {l.display_name(args.kernel_name_base) for l in launches} - matched_names

    # ---------- 账本参数（MB -> bytes）----------
    ledger_per_launch = args.ks_ledger_mb * 1e6 if args.ks_ledger_mb else None
    ledger_total = args.ks_ledger_total_mb * 1e6 if args.ks_ledger_total_mb else None
    ledger_range = None
    if args.ks_ledger_range_mb:
        try:
            parts = [float(x) for x in args.ks_ledger_range_mb.replace("..", ",").split(",")]
            if len(parts) != 2:
                raise ValueError("需要 LOW,HIGH 两个数")
            ledger_range = (min(parts) * 1e6, max(parts) * 1e6)
        except ValueError as e:
            log_error(f"error: --ks-ledger-range-mb 解析失败（{e}）；应形如 LOW,HIGH")
            return 1

    # ---------- DRAM 流量估算（NVML 代理；失败时可用轨迹口径对拍）----------
    from .analyze.dram import SHARE_STRONG
    from .analyze.dram import estimate as estimate_dram

    memtrace_stats = {}
    trace_bytes_per_launch = None
    trace_kernel = ""
    trace_reason = ("未提供访存轨迹：加 --ks-memtrace <trace> 可换用轨迹口径，在 NVML 代理口径不成立时仍能完成账本对拍")
    if args.ks_memtrace:
        from .analyze.memtrace import analyze_memtrace
        from .backends.memtrace import MemTraceError, load_memtrace_auto

        try:
            l2_bytes = args.ks_l2_bytes or int(device.l2_mib * 2 ** 20)
            trace = load_memtrace_auto(args.ks_memtrace, sm_count=sm_count, l2_bytes=l2_bytes)
            st = analyze_memtrace(trace, l2_bytes=l2_bytes or None,
                                  max_accesses=(args.ks_max_accesses or 4_000_000))
            memtrace_stats[st.kernel] = st
            trace_kernel = st.kernel
            trace_bytes_per_launch = st.dram_bytes_per_launch
            trace_reason = st.dram_bytes_unavailable_reason
            log(f"memtrace: {st.kernel} 载入 {st.records} 条记录"
                f"（L2={l2_bytes / 2**20:.0f} MiB"
                + (f"，来源 {args.ks_memtrace}" if not args.ks_l2_bytes else
                   "，用本机 L2 容量定容；device 未报 L2 时用 --ks-l2-bytes 指定")
                + "）")
        except (OSError, MemTraceError) as e:
            log_error(f"error: 访存轨迹读取失败: {e}")
            return 1

    issue_hist = None
    if args.ks_instr_hist:
        from .backends.opcode_hist import OpcodeHistError, load as load_hist

        try:
            issue_hist = load_hist(args.ks_instr_hist)
            log(f"instr-hist: 载入 {len(issue_hist.launches)} 条 launch 记录，"
                f"{len(issue_hist.kernels)} 个 kernel")
        except OpcodeHistError as e:
            log_error(f"error: 指令直方图读取失败: {e}")
            return 1

    grouped_dram: dict[str, list[KernelLaunch]] = {}
    for l in matched:
        grouped_dram.setdefault(l.display_name(args.kernel_name_base), []).append(l)
    peak_gbs = float(args.ks_peak_gbps or (peaks.bandwidth_gbs if peaks else 0.0) or 0.0)
    peak_source = ""
    if args.ks_peak_gbps:
        peak_source = "--ks-peak-gbps 覆盖值"
    elif peaks is not None:
        peak_source = peaks.source_label()
    launches_for_dram = len(matched_all) if len(grouped_dram) == 1 else 0
    # R-47：per-launch 的分母要落在**积分窗口内**。窗口被 --ks-window auto/--ks-warmup/
    # --ks-window-s 裁过时，光给总数会让"窗口内字节 ÷ 整段 launch 数"——数字等比偏小。
    # 所以把每个 launch 的绝对时间区间（CUPTI CLOCK_MONOTONIC，与采样同一时钟）传下去，
    # 由 dram.py 按时间重叠加权算窗口内的等效 launch 数；对不上就拒绝这个口径。
    launch_spans = ([(l.start_ns / 1e9, l.end_ns / 1e9) for l in matched_all]
                    if len(grouped_dram) == 1 else None)
    trace_resolution = ""
    st0 = memtrace_stats.get(trace_kernel)
    if trace_bytes_per_launch is not None and st0 is not None:
        trace_resolution = (f"抽样 {st0.sampling_ratio * 100:.1f} %"
                            if st0.sampling_ratio < 0.999 else "全量轨迹")
    dram_est = estimate_dram(
        sampler.result.samples, peak_gbs=peak_gbs, window_s=wall_ns / 1e9,
        launches=launches_for_dram,
        launch_spans=launch_spans,
        ledger_per_launch=ledger_per_launch, ledger_total=ledger_total,
        ledger_range=ledger_range,
        peak_source=peak_source,
        window_mode=args.ks_window,
        warmup_s=args.ks_warmup or 0.0,
        window_limit_s=args.ks_window_s,
        trace_bytes_per_launch=(trace_bytes_per_launch if len(grouped_dram) == 1 else None),
        trace_kernel=trace_kernel,
        trace_unavailable_reason=trace_reason,
        matched_kernel_ns=matched_ns,
        other_kernels_ns=other_ns,
        other_kernel_count=len(other_names),
        trace_resolution=trace_resolution,
    )
    if len(grouped_dram) != 1:
        dram_est.notes.append(
            f"匹配到 {len(grouped_dram)} 个 kernel：per-launch 口径不适用，只给总量")
        dram_est.trace_bytes_per_launch = None
        dram_est.trace_unavailable_reason = (
            f"匹配到 {len(grouped_dram)} 个 kernel：轨迹口径的 per-launch 字节不适用，"
            f"请用 -k 只选一个 kernel")

    # ---------- 组装报告 ----------
    from .analyze.occupancy import arch_for

    arch = arch_for(device.compute_capability or "8.9")
    nvml_summary = sampler.result.summary()
    clock_mhz = float(args.ks_clock_mhz or nvml_summary.get("sm_clock_mean") or 0.0)
    if issue_hist is not None and not clock_mhz:
        log("warn: 没有 SM 时钟读数（NVML 被关掉？）：发射槽上界算不出来，"
            "加 --ks-clock-mhz <MHz> 或用 --ks-no-nvml 之外的方式采样")
    # ---------- 成本账（C1）：窗口能耗 → 量 → 钱 ----------
    # 口径与 DRAM 账共用同一个积分窗口（dram_est.window_t0/t1/window_used_s），
    # 否则"窗口内字节"和"窗口内能耗"会各自裁一次窗口，两个数说的不是同一段负载。
    from .analyze.cost import CostInputs, estimate as estimate_cost

    cost_inputs = CostInputs(tokens_per_step=args.ks_tokens,
                             price_per_kwh=args.ks_price_per_kwh,
                             price_per_hour=args.ks_price_per_hour,
                             idle_power_w=args.ks_idle_power_w,
                             currency=args.ks_currency)
    cost_est = estimate_cost(
        sampler.result.samples,
        inputs=cost_inputs,
        energy_j=sampler.result.energy_j,
        sampler_duration_s=sampler.result.duration_s,
        window_s=(dram_est.window_used_s or wall_ns / 1e9),
        window_t0=dram_est.window_t0,
        window_t1=dram_est.window_t1,
        steps_in_window=dram_est.launches_in_window,
    )
    if cost_est.window_energy_j is not None:
        log(f"cost: 窗口能耗 {cost_est.window_energy_j:.1f} J"
            f"（{cost_est.energy_source}）"
            + (f"；{cost_est.tokens_per_s:,.0f} token/s、"
               f"{cost_est.j_per_token:.3g} J/token" if cost_est.tokens_per_s else "")
            + (f"；{cost_inputs.currency} {cost_est.cost_per_mtok:.3g}/Mtoken"
               if cost_est.cost_per_mtok is not None else ""))
    elif not args.quiet:
        log(f"cost: 能耗不给数（{cost_est.energy_refusal.message if cost_est.energy_refusal else '未知原因'}）")

    opt = BuildOptions(
        set_name=args.set_name,
        sections=list(args.section),
        peaks=peaks,
        workload=workload,
        workload_note=workload_note,
        achieved_occupancy=achieved,
        cubin_res=cubin_res,
        cubin_sass=cubin_sass,
        nvml_summary=nvml_summary,
        nvml_env=sampler.result.environment(),
        sm_count=sm_count,
        device_name=device.name,
        arch=arch,
        peak_gbs=peak_gbs,
        peak_source=peak_source,
        ledger_basis=args.ks_ledger_basis,
        dram=dram_est,
        cost=cost_est,
        memtrace=memtrace_stats,
        issue_hist=issue_hist,
        clock_mhz=clock_mhz,
    )

    rep = Report(device=device)
    rep.meta.update(meta)
    # 设备名/CC 在容器里常常取不到（NVML 不可用），但标定文件是 nvidia-smi 跑出来的，
    # 里面记了 `gpu`/`cc`。用它补齐：导出的 `.ncu-rep` 靠这两个事实渲染设备行
    # （缺了 ncu 会读成 `Device 255, CC 0.0`），JSON 报告里也不再是空串。
    if args.ks_calibration:
        try:
            _prof = json.loads(Path(args.ks_calibration).read_text())
        except Exception:
            _prof = {}
        if not rep.device.name and _prof.get("gpu"):
            rep.device.name = str(_prof["gpu"])
        if not rep.device.compute_capability and _prof.get("cc"):
            rep.device.compute_capability = str(_prof["cc"])
    rep.sources = {
        "kernel activity": "nsys sqlite CUPTI_ACTIVITY_KIND_KERNEL (no counters needed)",
        "device counters": f"{sampler.result.source} (energy/PCIe/mem-busy are management counters, "
                           f"available without admin)",
        "static": "cuobjdump -res-usage / -sass" if cubin_res or cubin_sass else "not used",
        "peaks": f"measured calibration ({args.ks_calibration})" if args.ks_calibration else
                 ("override" if (args.ks_peak_tflops or args.ks_peak_gbps) else "not provided"),
    }
    if args.ks_synthetic:
        rep.sources["WARNING"] = ("SYNTHETIC DATA - numbers are fabricated by --ks-synthetic and "
                                  "must not be interpreted as measurements")

    for name, ls in grouped_dram.items():
        alias = ls[0].short_name if ls[0].short_name and ls[0].short_name != name else ""
        kr = KernelReport(name, launch_count=len(ls), alias=alias,
                          launches=sorted(ls, key=lambda x: x.start_ns))
        kr.sections = build_kernel_sections(kr, opt)
        # 成本归因（C1）：按该 kernel 在窗口内的时间占比分摊窗口能耗与钱。
        # 份额的分母是**参与积分的窗口**（与能耗同一个窗口），不是整段 wall。
        if cost_est.has_energy and cost_est.window_s > 0:
            from .analyze.cost import attribute as attribute_cost

            share = min(100.0, 100.0 * kr.total_ns / (cost_est.window_s * 1e9))
            if share > 0:
                opt.cost = attribute_cost(cost_est, share, steps=kr.launch_count or None)
                attr_sec = build_cost_attribution_section(opt, kr)
                if attr_sec is not None:
                    kr.sections.append(attr_sec)
        rep.kernels.append(kr)
    opt.cost = cost_est
    dev_kr = KernelReport("(device)", launch_count=0)
    dev_sections = [build_device_kernel(opt)]
    if "DramTraffic" in requested_sections(opt):
        dram_sec = build_dram_section(opt)
        if dram_sec is not None:
            dev_sections.append(dram_sec)
    cost_sec = build_cost_section(opt)
    if cost_sec is not None:
        dev_sections.append(cost_sec)
    dev_kr.sections = dev_sections
    rep.kernels.append(dev_kr)

    # ---------- I8：NVTX 阶段归因（训练/推理阶段各占多少 kernel 时间）----------
    _attach_phases(rep, args, dev_kr, log)

    # ---------- C2：把"改哪里、能省多少"算出来（头寸 + 排序）----------
    _attach_prescription(rep, args, cost_est, peaks, log)

    # ---------- I1：导入一份真值 .ncu-rep 并与替代口径对拍 ----------
    if getattr(args, "ks_ncu_rep", None):
        _attach_truth(rep, str(args.ks_ncu_rep), log)

    rep.sort_kernels("total")
    # synthetic 模式是瞬时生成的，wall time 无物理意义 -> 用 kernel 跨度代替，不报百分比
    if args.ks_synthetic:
        span = 0
        for k in rep.kernels:
            for l in k.launches:
                span = max(span, l.end_ns)
        starts = [l.start_ns for k in rep.kernels for l in k.launches]
        rep.wall_time_ns = (span - min(starts)) if starts else 0
    else:
        rep.wall_time_ns = wall_ns
        if wall_ns:
            rep.meta["kernel_time_pct"] = 100.0 * rep.total_kernel_ns / wall_ns
    # 决策包要用它判断"账本裁决是否继承了不成立的 per-launch 口径"
    rep.meta["ledger_basis"] = dram_est.ledger_basis
    rep.meta["dtype"] = args.ks_dtype
    # ---- 协议哈希的输入（R-49）：口径参数必须由**管线**写成，不能靠测试手工塞 meta ----
    # 旧实现里 protocol_hash 读 meta["calibration"]/["peak_source"]/["kernel_filter"]/
    # ["warmup_s"]/["window_s"]，而 CLI 从不写这些键 → 哈希里全是 null。后果是
    # "换一份标定（roofline 的分母）"、"把窗口调短"都不算协议变化，`diff` 照报方向。
    rep.meta.update(_protocol_meta(args, peaks, peak_gbs))
    rep.generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep.meta["window"] = (
        f"mode={dram_est.window_mode}{'（已裁剪）' if dram_est.trimmed else ''} "
        f"采样 {dram_est.n_samples} 点 / 参与积分 {dram_est.window_used_s:.1f} s / "
        f"裁掉 {dram_est.window_dropped_s:.1f} s / 进程 wall {wall_ns / 1e9:.1f} s"
        + (f" / --ks-warmup {args.ks_warmup:g}s" if args.ks_warmup else "")
        + (f" / --ks-window-s {args.ks_window_s:g}s" if args.ks_window_s else ""))
    if dram_est.reliable and dram_est.bytes_total > 0:
        # R-45：摘要里的「显存忙碌」是**参与积分窗口**的均值，与 meas__nvml_mem_busy_mean_pct
        # （整进程均值）是两个数。两个都印出来，并给出参与估算那个的名字，读者才能复核估算。
        proc_busy = nvml_summary.get("mem_util_mean")
        summary = (f"DRAM 估算 {dram_est.bytes_total / 1e6:.1f} MB"
                   f"（窗口显存忙碌 {dram_est.mem_util_pct_mean:.1f}%"
                   + (f"／整进程 {proc_busy:.1f}%" if proc_busy is not None else "")
                   + f" × 峰值 {dram_est.peak_gbs:.0f} GB/s，"
                   f"窗口覆盖率 {dram_est.window_coverage_pct:.1f}%"
                   f"；参与估算的是 ks__dram_window_mem_busy_mean_pct）")
        if dram_est.window_busy_segments > 1:
            summary += f"；窗口有 {dram_est.window_busy_segments} 段连续忙碌期"
        # D3：per-launch 能不能给、为什么不能给，只由 per_launch_verdict 一个地方决定。
        # 从前这里是 `bytes_per_launch is not None`，于是摘要会一边说「口径不成立」一边
        # 把那个本该作废的 per-launch 值打印出来。
        ok_pl, why_pl = dram_est.per_launch_verdict()
        share = dram_est.matched_kernel_kernel_share_pct
        if share is not None:
            summary += (f"；窗口主导性 {share:.1f}%"
                        f"（内核时间份额，门槛 {SHARE_STRONG:.0f}%）")
        if ok_pl:
            summary += f" → {dram_est.bytes_per_launch / 1e6:.2f} MB/launch"
            if dram_est.attribution_bias_estimate_pct:
                summary += f"（归因上偏估计 +{dram_est.attribution_bias_estimate_pct:.0f}%）"
        else:
            summary += f" → per-launch 不给：{why_pl}"
        if dram_est.ledger_requested:
            ref = dram_est.reference_bytes()
            basis = "per-launch" if dram_est.ledger_basis == "per-launch" else "总量"
            err = dram_est.ledger_err_pct_for(args.ks_ledger_basis)
            if err is None:
                summary += (f"；账本对拍（{basis}）口径不成立，未给误差"
                            f"（原因见 DramTraffic section 的 n/a 行）")
            else:
                used = dram_est.basis_label(args.ks_ledger_basis)
                summary += (f"；vs 账本 {(ref or 0) / 1e6:.1f} MB（{basis}，口径 {used}），"
                            f"误差 {err:+.1f}%")
        if dram_est.ledger_range:
            summary += f"；{dram_est.range_verdict_for(args.ks_ledger_basis)}"
        rep.meta["dram_summary"] = summary
    else:
        rep.meta["dram_summary_unavailable"] = "；".join(
            dram_est.reason_for("est__dram_bytes_total", args.ks_ledger_basis).split("；"))
    # 账本裁决即使总量不可得也要出现在摘要里（B1：拒绝必须可见）
    if not dram_est.reliable and dram_est.ledger_requested:
        err = dram_est.ledger_err_pct_for(args.ks_ledger_basis)
        if err is None:
            reason = dram_est.reason_for(
                "est__dram_range_position_pct" if dram_est.ledger_range
                else "est__dram_vs_ledger_err_pct", args.ks_ledger_basis)
            rep.meta["dram_summary_unavailable"] += f"；账本裁决未给出：{reason}"

    # ---------- 空转选项告警 ----------
    used_noop = [f for f in NOOP_FLAGS
                 if getattr(args, f.lstrip("-").replace("-", "_"), None) is not None]
    if used_noop:
        log(f"warn: these flags are accepted for ncu compatibility but have no effect here: "
            f"{', '.join(used_noop)} (our data source has no counter/replay passes)")

    # ---------- R-38：达成值超过标定峰值 → 显式红灯 ----------
    # D20：100%~102% 这一档**不是**红灯（标定自身有 3~4% 口径差），但也绝不能静默 ——
    # 101% 与 95% 在报告里长得一样，而前者已经物理上说不通。所以这里对容忍带也给 warn。
    for _k in rep.kernels:
        for _s in _k.sections:
            for _m in _s.metrics:
                if _m.name == "ks__over_peak_warning":
                    log(f"warn: [{_k.name}] {_m.note}")
                elif _m.name == "est__pct_of_measured_compute_peak" and _m.note:
                    log(f"warn: [{_k.name}] {_m.note}")

    # ---------- R-39：--metrics 一个都没匹配上时不能静默成功 ----------
    # D4①：能映射到替代口径的请求名**不算"没匹配上"**（值会以替代口径的名字出现，
    # 前述 note 已经说明了这一点）；只有真正查不到的名字才报警。
    from .ncu_names import resolve as resolve_requested_name

    requested_metrics = [n for v in (args.metrics or []) for n in v.split(",") if n.strip()]
    if requested_metrics:
        produced = _collect_metric_names(rep)

        def _has_substitute(n: str) -> bool:
            d = resolve_requested_name(n)
            return bool(d is not None and d.ours)

        missing = [n for n in requested_metrics
                   if n not in produced and not _has_substitute(n)]
        if missing and len(missing) == len(requested_metrics):
            log(f"warn: --metrics 请求的 {len(missing)} 个指标在本报告里一个都不存在"
                f"（{', '.join(missing)}）—— 因此报告里不会有任何指标行。"
                f"这可能是名字写错，也可能是需要硬件计数器（本机拿不到）；"
                f"用 --strict 可把它变成失败（exit 2），或换 --set basic 看有哪些指标可用")
        elif missing:
            log(f"warn: --metrics 里有 {len(missing)} 个指标不存在、已忽略：{', '.join(missing)}")

    # ---------- 输出 ----------
    rc = _emit(rep, args, compat_mode, logged,
               packet_ctx={"capabilities": cap, "selfcheck": selfcheck})

    # ---------- --strict ----------
    if args.strict:
        problems = _strict_problems(rep, args, used_noop)
        if problems:
            _report_strict(problems)
            return 2

    if args.check_exit_code and target_rc:
        rc = target_rc
    _write_case_if_needed(rep, args, rc, argv)
    return rc


def _log_error(log, msg: str) -> None:
    """打一条 `error:` 行 —— **不受 `--quiet` 影响**。

    为什么要有这个模块级小函数：`collect()` / `_attach_truth()` / `_export_substitute()`
    都是**模块级**函数，拿不到 `_main` 里的 `log_error` 闭包。历史上它们直接调用 `log_error`，
    于是「没给目标命令 / 忘记写 `--` / 拼错子命令」这三条**最高频**路径抛 `NameError` ——
    源码里已经写好的那句 `error: no target command. Put it after '--'…` 永远打不出来
    （第六轮 W-2 与一次易用性审计各命中一处，共 6 个调用点）。

    `_main()`/`_stderr_log()` 都会把 `log_error` 挂在 `log` 上；万一没挂上，
    退回普通 `log`（至少不会再 NameError，代价是 `--quiet` 会吞掉它）。
    """
    err = getattr(log, "log_error", None)
    if callable(err):
        err(msg)
    else:
        log(msg)


def _stderr_log(args: argparse.Namespace, logged: list[str]):
    """把"记进 logged + 按 --quiet 打到 stderr"包成一个函数（`_emit` 内部用）。"""
    def log(msg: str) -> None:
        logged.append(msg)
        if not getattr(args, "quiet", False):
            print(msg, file=sys.stderr)

    def log_error(msg: str) -> None:
        logged.append(msg)
        print(msg, file=sys.stderr)

    log.log_error = log_error          # type: ignore[attr-defined]
    return log


def _attach_phases(rep: Report, args: argparse.Namespace, dev_kr, log) -> None:
    """I8：从 nsys sqlite 的 NVTX 区间把 kernel 时间分成训练/推理阶段。

    两个诚实点：
      * **没有 NVTX 数据也要出一行**（n/a + code）："这个阶段没标"与"没测"是两件事，
        消失的行会让读者以为工具没这个能力（B1）。
      * 阶段之和**不是划分**（嵌套区间各自计入），这句话写在 basis 里随数字一起走。
    """
    from .analyze.phases import attribute as attribute_phases
    from .provenance import Metric, Section
    from .sections import _split_unavailable  # noqa: PLC2701 - 同一仓库内的共享约定

    db = (getattr(args, "ks_sqlite", None) or rep.meta.get("nsys_sqlite")
          or rep.meta.get("nsys_sqlite_path"))
    if not db:
        if getattr(args, "ks_nvtx_phases", False):
            log("warn: --ks-nvtx-phases 需要 --ks-sqlite/--ks-nsys-rep 指向的 nsys 数据库"
                "（本次没有 kernel 级数据源，阶段归因无从谈起）")
        return
    try:
        att = attribute_phases(str(db))
    except Exception as exc:      # 只有输入错误才会抛：文件不存在/不是 sqlite
        log(f"warn: NVTX 阶段归因失败（{type(exc).__name__}: {exc}）")
        return
    sec = Section("NvtxPhases")
    for name, value, unit, note in att.to_metrics():
        if value is None:
            code, why = _split_unavailable(note)
            sec.add(Metric.unavailable(name, why, unit=unit, code=RefusalCode(code),
                                       observed={"basis": att.basis[:200]}, severity="degraded"))
        else:
            # 每行的 note 与 section 级 basis 可能不同（逐阶段那几行带"嵌套时不是划分"这类
            # 只对那一行成立的前提）：相同就省掉重复，不同就必须留 —— 前置纪律是"前提随数字走"。
            extra = "" if note == att.basis else note
            sec.add(Metric.derived(name, value, unit, source=att.basis, note=extra))
    if sec.metrics:
        dev_kr.sections.append(sec)
        rep.meta["nvtx_phases"] = {
            "attributed_ms": att.attributed_ms, "unattributed_ms": att.unattributed_ms,
            "count": len(att.phases), "omitted": att.omitted, "basis": att.basis,
            "reason": att.reason,
            "phases": [{"name": p.name, "depth": p.depth, "kernel_ms": p.kernel_ms,
                        "share_pct": p.share_pct, "kernels": p.kernels} for p in att.phases],
        }
        if att.reason:
            log(f"nvtx: 不给阶段数字（{att.reason}）")
        else:
            top = att.phases[0] if att.phases else None
            log(f"nvtx: {len(att.phases)} 个阶段"
                + (f"，最重的是「{top.name}」占 kernel 时间 {top.share_pct:.1f}%" if top else ""))


def _kernel_metric(kr, name: str) -> float | None:
    """在**某个** kernel 的 section 里取指标值（`packet.metric_value` 取的是全报告第一个）。"""
    for sec in getattr(kr, "sections", []) or []:
        for m in sec.metrics:
            if m.name == name and isinstance(m.value, (int, float)) \
                    and not isinstance(m.value, bool):
                return float(m.value)
    return None


def _attach_prescription(rep: Report, args: argparse.Namespace, cost_est, peaks, log) -> None:
    """C2：把"哪个 kernel 还有多少头寸、能省多少钱"算成可排序的处方。

    三条诚实边界（都在 analyze/prescription.py 里执行，这里只负责接线）：
      * 头寸的分母**只能**是本机实测峰值：`--ks-peak-*` 覆盖/规格表一律算不出区间；
      * 没有价格就没有钱（比例仍然给）；
      * 重叠的头寸不相加（按最大者封顶）。
    没有任何达成值可算时**不产生 section**：那种情况下 roofline 判词的拒绝已经说清了缺什么。
    """
    from .analyze.cost import attribute as attribute_cost
    from .analyze.prescription import (Calibration, bandwidth_headroom, issue_headroom,
                                       prescribe_or_refuse)
    from .packet import metric_value

    if peaks is not None and getattr(args, "ks_calibration", None) and not (
            getattr(args, "ks_peak_gbps", None) or getattr(args, "ks_peak_tflops", None)):
        calib = Calibration(bandwidth_gbs=peaks.bandwidth_gbs,
                            issue_util_pct=getattr(peaks, "issue_util_pct", None),
                            source=peaks.source_label(), measured=True,
                            caliber_ok=bool(getattr(peaks, "caliber_ok", True)),
                            caliber_note=getattr(peaks, "caliber_note", ""))
    else:
        calib = Calibration.from_spec(
            bandwidth_gbs=(getattr(args, "ks_peak_gbps", None)
                           or (peaks.bandwidth_gbs if peaks else None)),
            source="--ks-peak-*/规格表（不用于头寸估计）")

    window_s = float(getattr(cost_est, "window_s", 0.0) or 0.0) or rep.wall_time_ns / 1e9
    entries = []
    for kr in rep.kernels:
        if kr.name == "(device)":
            continue
        share = (min(100.0, 100.0 * kr.total_ns / (window_s * 1e9))
                 if window_s and kr.total_ns else None)
        cost = attribute_cost(cost_est, share, steps=kr.launch_count or None) if share else None
        achieved_gbs = _kernel_metric(kr, "est__achieved_gbs")
        if achieved_gbs is None:
            achieved_gbs = _kernel_metric(kr, "meas__achieved_gbs")
        if achieved_gbs is not None:
            entries.append(bandwidth_headroom(
                kr.name, achieved_gbs=achieved_gbs, calib=calib, cost=cost,
                kernel_share_pct=share, cost_per_mtok=cost_est.cost_per_mtok,
                achieved_provenance="measured" if _kernel_metric(
                    kr, "meas__achieved_gbs") is not None else "estimated"))
        util = _kernel_metric(kr, "est__issue_slot_utilization_pct")
        if util is not None:
            entries.append(issue_headroom(
                kr.name, achieved_pct=util, calib=calib, cost=cost,
                kernel_share_pct=share, cost_per_mtok=cost_est.cost_per_mtok))
    if not entries and _kernel_metric(rep.kernels[0], "ks__over_peak_warning") is None:
        # 一个达成值都没有：不给处方（roofline 判词的拒绝已经指向 --ks-gemm/--ks-calibration）
        return
    pre = prescribe_or_refuse(entries, currency=cost_est.inputs.currency)
    sec = build_prescription_section(pre)
    dev = next((k for k in rep.kernels if k.name == "(device)"), None)
    if sec is not None and dev is not None:
        dev.sections.append(sec)
    rep.meta["prescription"] = {
        "currency": pre.currency,
        "ranked": [{"kernel": h.kernel, "resource": h.resource, "confidence": h.confidence,
                    "pct_lo": (h.saving_pct_range[0] if h.saving_pct_range else None),
                    "pct_hi": (h.saving_pct_range[1] if h.saving_pct_range else None),
                    "per_hour": h.saving_per_hour, "per_mtok": h.saving_per_mtok,
                    "basis": h.basis,
                    "refusal": h.refusal.code.value if h.refusal else None,
                    "money_refusal": h.money_refusal.code.value if h.money_refusal else None}
                   for h in pre.ranked],
        "total_per_hour": pre.total_saving_per_hour,
        "total_per_mtok": pre.total_saving_per_mtok,
        "refusal": pre.refusal.code.value if pre.refusal else None,
        "refusal_message": pre.refusal.message if pre.refusal else "",
        "money_refusal": pre.money_refusal.code.value if pre.money_refusal else None,
        "money_refusal_message": pre.money_refusal.message if pre.money_refusal else "",
        "notes": pre.notes,
    }
    if pre.ranked:
        hy = (f"，合计可省 {pre.total_saving_per_hour:.3g} {pre.currency}/h"
              if pre.total_saving_per_hour is not None else "（钱还给不出：见 money_refusal）")
        log(f"prescription: {len(pre.ranked)} 条头寸{hy}")
    else:
        log("prescription: 没有可辩护的头寸（" +
            (pre.refusal.message[:120] if pre.refusal else "缺少达成值/实测峰值") + "）")


def _attach_truth(rep: Report, path: str, log) -> None:
    """把 `.ncu-rep` 里的真值接进报告（I1），并逐条与替代口径对拍。

    两条纪律：
      * 真值用 `ncu__` 前缀 + `exact`（与 ncu 同源），**不是**计数器命名空间，
        所以它绕开 R2 的 allowlist —— 这个名字本身就说明"它来自导入的报告"；
      * 读不了就报结构化拒绝（`ncu_report_unreadable` + 处置），绝不静默地当没有这回事。
        缺的往往只是 `PYTHONPATH`：读报告不需要 GPU、也不需要计数器权限。
    """
    from .analyze.truth import (comparison_metrics, compare, match_action, truth_metrics)
    from .backends import ncu_rep as ncu_rep_backend
    from .provenance import Metric, Section
    from .refusals import Refusal, RefusalCode

    try:
        # 必须走子进程：主进程里已经 import 了 torch，与 Nsight Compute 的原生模块同进程
        # 会让 `get_version()` 段错误（真机 rc=139，见 backends/ncu_rep.load_isolated 的说明）。
        # segfault 抓不住，所以隔离是唯一可靠的做法。
        data = ncu_rep_backend.load_isolated(path)
    except ncu_rep_backend.NcuRepUnavailable as e:
        msg = (f"读不了 {path}：{e}；读 .ncu-rep 只要 Nsight Compute 的 Python 模块"
               f"（把它放进 PYTHONPATH），不需要 GPU 与计数器权限")
        rep.meta.setdefault("environment_refusals", []).append(
            Refusal(RefusalCode.NCU_REPORT_UNREADABLE, msg).to_dict())
        _log_error(log, f"error: {msg}")
        return
    except ncu_rep_backend.NcuRepError as e:
        msg = f"读不了 {path}：{e}"
        rep.meta.setdefault("environment_refusals", []).append(
            Refusal(RefusalCode.NCU_REPORT_UNREADABLE, msg).to_dict())
        _log_error(log, f"error: {msg}")
        return

    ours = {m.name: m.value for k in rep.kernels for s in k.sections for m in s.metrics
            if isinstance(m.value, (int, float)) and not isinstance(m.value, bool)}
    hit = 0
    total_rows = 0
    comparisons: list[dict[str, Any]] = []
    for action in data.actions:
        kr = next((k for k in rep.kernels if k.name != "(device)"
                   and k.name == action.kernel_name), None)
        if kr is None:
            kr = next((k for k in rep.kernels if k.name != "(device)"
                       and match_action(data, k.name) is action), None)
        if kr is None:
            kr = KernelReport(action.kernel_name, launch_count=0)
            rep.kernels.append(kr)
        sec = Section("NcuGroundTruth")
        for name, value, source in truth_metrics(data, action):
            sec.add(Metric.exact(name, float(value), source=source,
                                 note="导入的真值（与 ncu 同源），用于校准我们的替代口径"))
        cmp = compare(data, ours, action=action,
                      ours_by_name={m.name: m.value for s in kr.sections for m in s.metrics
                                    if isinstance(m.value, (int, float))})
        pair_by_ours = {r.ours_name: r for r in cmp.rows}
        for name, value, unit, note in comparison_metrics(cmp):
            ours_name = name.split("[", 1)[1].rstrip("]")
            row = pair_by_ours.get(ours_name)
            truth_name = row.truth_name if row is not None else ""
            if value is None:
                sec.add(Metric.unavailable(
                    name, (row.reason if row is not None else "不可比"),
                    unit=unit, source=f"对拍：{truth_name} vs {ours_name}",
                    code=RefusalCode.PREMISE_FAILED if row is not None and row.truth is not None
                    else RefusalCode.METRIC_ABSENT,
                    observed={"truth": row.truth if row is not None else None,
                              "ours": row.ours if row is not None else None},
                    severity="degraded"))
                continue
            sec.add(Metric.derived(name, value, unit, source=f"对拍：{truth_name} vs {ours_name}",
                                   note=note))
            hit += 1
        total_rows += len(cmp.rows)
        comparisons.append(cmp.to_dict())
        kr.sections.append(sec)
    rep.meta["ncu_truth"] = {
        "path": data.path, "ncu_version": data.tool_version,
        "actions": len(data.actions),
        "metrics_read": sum(len(a.metrics) for a in data.actions),
        "metrics_skipped": sum(a.skipped_metrics for a in data.actions),
        "pairs_compared": hit, "pairs_total": total_rows,
        "actions": [{"kernel": a.kernel_name, "grid": list(a.grid), "block": list(a.block),
                     # Python API 的 grid/block 只能从 launch__grid_dim_* 指标里读，**没有**独立
                     # accessor（2025.3.1 实测）。缺这几个指标时这里是 0 = 未知，不是"0 个 block"；
                     # 而 `ncu --import` 能看到 Geometry（它读 ProfileResult.GridSize）。
                     "geometry_known": any(d in a.metrics for d in
                                           ("launch__grid_dim_x", "launch__block_dim_x")),
                     "metrics": len(a.metrics), "rules": a.rules} for a in data.actions],
        "comparisons": comparisons,
        "max_abs_err_pct": (max((abs(r["rel_err_pct"]) for c in comparisons
                                 for r in c.get("pairs", []) if "rel_err_pct" in r),
                                default=None)),
    }
    log(f"ncu-rep: 导入 {data.path}（ncu {data.tool_version or '?'}）：{len(data.actions)} 个 "
        f"action、{rep.meta['ncu_truth']['metrics_read']} 个真值指标；对拍了 {hit}/{total_rows} 条")
    if not hit and total_rows:
        log("warn: 一条对拍都没成功（不是没数，是两边口径对不上）—— 用 --json 看 pairs[].reason")


def _ncu_rep_disclosure(rep: Report, packet: dict[str, Any] | None, *,
                        tool_version: str) -> str:
    """写给 `.ncu-rep` 读者（在 ncu-ui/details 里会看到）的署名与结论。

    为什么需要它：details 页的表格只认 ncu 自己的指标标签，我们的替代口径进不了那些表 ——
    唯一能渲染的是 RuleResult 的 Message。所以把"这是替代口径、哪些是同源、钱是多少"
    写在这里，读者在 ncu 的界面里也能看到出处而不是以为全是它自己测的。
    """
    dev = getattr(rep, "device", None)
    dev_bits = []
    if dev is not None:
        if getattr(dev, "name", ""):
            dev_bits.append(str(dev.name))
        if getattr(dev, "compute_capability", ""):
            dev_bits.append(f"compute capability {dev.compute_capability}")
    lines = [
        f"kernoscope {tool_version} wrote this report on a machine where NVIDIA performance "
        f"counters are not accessible (ERR_NVGPUCTRPERM).",
        ("device: " + ", ".join(dev_bits) + ". ncu reads the header device line from an opaque "
         "PerfWorks blob we do not have; the kernel row therefore shows "
         "`Device 255, CC 0.0`." ) if dev_bits else
        "device: unknown on this machine (NVML unavailable); the header device line stays unknown.",

        "Launch Statistics / GPU Speed Of Light rows marked same-source are CUPTI kernel "
        "activity and cubin facts: identical semantics and units to Nsight Compute. "
        "Substitute calibers (NVML sampling, address-arithmetic traces, models, cost) are NOT "
        "counters and live under ks__/est__/meas__ names -- see them with "
        "`ncu --import <file> --page raw` and in <file>.ATTRIBUTION.txt.",
    ]
    facts: list[str] = []
    if packet:
        for v in packet.get("verdicts", []) or []:
            if v.get("id") == "cost_per_token" and v.get("status") != "refused":
                f = v.get("facts") or {}
                if f.get("per_mtok") is not None:
                    facts.append(f"unit cost {f['per_mtok']} (per Mtoken)")
                if f.get("tokens_s"):
                    facts.append(f"{float(f['tokens_s']):,.0f} token/s")
                if f.get("j_per_token") is not None:
                    facts.append(f"{float(f['j_per_token']):.3g} J/token")
            if v.get("id") == "saving_estimate" and v.get("status") != "refused":
                f = v.get("facts") or {}
                if f.get("top_pct_lo") is not None:
                    facts.append(f"top headroom {f.get('top_kernel', '')}: "
                                 f"{f['top_pct_lo']}%-{f.get('top_pct_hi')}% "
                                 f"(conservative..upper bound)")
    if facts:
        lines.append("Substitute conclusions: " + "; ".join(facts) + ".")
    return "\n".join(lines)


def _export_substitute(rep: Report, args: argparse.Namespace, packet: dict[str, Any] | None,
                       log) -> int:
    """把替代口径写成 `.ncu-rep`（I2/M4）+ 旁路署名。"""
    from . import __version__
    from .ncrep.substitute import SubstituteRefused, write_substitute

    kernels = [k for k in rep.kernels
               if k.name != "(device)" and list(getattr(k, "launches", []) or [])]
    if not kernels:
        _log_error(log, "error: 没有可导出的 kernel（没有 launch 记录就写不出 kernel 身份，"
                  "编一组 grid/block 就是在造身份）")
        return 2 if args.strict else 1
    kernels.sort(key=lambda k: k.total_ns, reverse=True)
    kr = kernels[0]
    if packet is None:
        from .packet import build

        packet = build(rep, basis=str(rep.meta.get("ledger_basis") or ""))
    # 文内署名/结论（进 RuleResult 的 Message）：这是 details 页**会渲染**散文的唯一通道。
    # 里面必须写清"哪些表是同源真值、哪些数是替代口径、不要当计数器引用"。
    disclosure = _ncu_rep_disclosure(rep, packet, tool_version=__version__)
    dev = getattr(rep, "device", None)
    try:
        res = write_substitute(kr, str(args.export_ncu_rep), packet=packet,
                               tool_version=__version__, disclosure=disclosure,
                               device_name=str(getattr(dev, "name", "") or ""),
                               compute_capability=str(getattr(dev, "compute_capability", "")
                                                      or ""))
    except SubstituteRefused as e:
        _log_error(log, f"error: 拒绝生成 .ncu-rep：{e}")
        return 2 if args.strict else 1
    except OSError as e:
        # V-25：只读目录写 .ncu-rep（或旁路 `.ATTRIBUTION.txt`）曾经是裸 PermissionError。
        # 错误不受 --quiet 影响（写不出产物时静默的 rc=1 更难排查）。
        _log_error(log, f"error: 写不了 ncu-rep 导出（含 .ATTRIBUTION.txt 署名）："
                       f"{args.export_ncu_rep}（{e.strerror or e}）")
        print(f"hint: {_io_remedy(e)}", file=sys.stderr)
        return 1
    log(f"ncu-rep: 写出 {res.ncu_rep_path}（{res.bytes_written} B，"
        f"{res.metrics_written} 个值，全部带出处）+ 署名 {res.sidecar_path}")
    if len(kernels) > 1:
        log(f"note: 报告里有 {len(kernels)} 个 kernel，只导出了最耗时的 {kr.name}"
            f"（用 -k 选别的）")
    rep.meta["out_ncu_rep"] = res.ncu_rep_path
    rep.meta["out_ncu_rep_sidecar"] = res.sidecar_path
    return 0


def _write_html_view(packet: dict[str, Any] | None, args: argparse.Namespace,
                     logged: list[str]) -> int:
    """N9：把决策包渲染成一个**自包含**的 HTML 证据单（离线、零 JS、无服务器）。

    为什么不是"Web UI"：这里没有前端工程、没有后端、没有状态 —— 是"JSON → 一个文件"的渲染器
    （NON-GOALS N9 的择中形态）。所以它天然带得走：可截图、可当附件、可离线打开。
    """
    from .html_view import render_html

    if packet is None:
        print("error: --html 需要决策包（本次没有判词可渲染）；去掉 --screen off/--csv 或加 -o",
              file=sys.stderr)
        return 1
    text = render_html(packet)
    out = str(args.html)
    if out == "-":
        if getattr(args, "packet", None) == "-":
            print("error: `--html -` 与 `--packet -` 都要求 stdout：一次只能要一种格式",
                  file=sys.stderr)
            return 2
        sys.stdout.write(text)
        return 0
    _write_text_or_fail(text, out, what="HTML 证据单", logged=logged)
    print(f"==PROF== HTML 证据单: {out}（{len(text.encode('utf-8'))} B）", file=sys.stderr)
    return 0


def _write_report_md(rep: Report, args: argparse.Namespace, packet: dict[str, Any] | None,
                     logged: list[str]) -> int:
    """I3：把 per-kernel 报告写成 Markdown（人读；结构对齐生态里通用的报告模板）。"""
    from .report_md import render_report_md

    kernels = [k for k in rep.kernels if k.name != "(device)"]
    if not kernels:
        print("error: --report-md 需要至少一个 kernel（本次没有 kernel 级数据）", file=sys.stderr)
        return 1
    kernels.sort(key=lambda k: k.total_ns, reverse=True)
    kr = kernels[0]
    try:
        text = render_report_md(rep, packet, kernel=kr.name)
    except (ValueError, TypeError) as exc:
        print(f"error: 生成 Markdown 报告失败：{exc}", file=sys.stderr)
        return 1
    out = str(args.report_md)
    # V-25：只读目录写 Markdown 报告曾经是裸 PermissionError。
    _write_text_or_fail(text, out, what="人读报告（Markdown）", logged=logged)
    rep.meta["out_report_md"] = out
    print(f"==PROF== per-kernel 报告: {out}（{len(text.encode('utf-8'))} B）", file=sys.stderr)
    if len(kernels) > 1:
        print(f"==PROF== 注：报告里有 {len(kernels)} 个 kernel，只写了最耗时的 {kr.name}"
              f"（用 -k 选别的）", file=sys.stderr)
    return 0


def _write_case_if_needed(rep: Report, args: argparse.Namespace, rc: int,
                          argv: Sequence[str]) -> None:
    """C5：案件目录 —— 让一次优化任务可跨会话恢复、可交接。

    默认在 `-o <前缀>` 旁写 `<前缀>.case/`；`--no-case` 关掉。写失败**不影响主流程**
    （它是辅助产物，不该让一次成功的测量变成失败），但会把原因说清楚。
    """
    if getattr(args, "no_case", False) or not getattr(args, "export", None):
        return
    packet = rep.meta.get("_packet")
    if not packet:
        return
    try:
        from .case import write_case

        outputs = {"report": rep.meta.get("out_json"),
                   "perfetto": rep.meta.get("out_perfetto"),
                   "launch_csv": rep.meta.get("out_launch_csv"),
                   "packet": getattr(args, "packet", None),
                   "onepage": rep.meta.get("out_onepage")}
        write_case(packet, args.export, argv, rc, outputs, quiet=bool(args.quiet))
    except OSError as e:
        print(f"warn: 案件目录写入失败（{type(e).__name__}: {e}）—— 主产物不受影响",
              file=sys.stderr)


def _calibration_identity(path: str) -> str:
    """标定文件的身份：**文件名 + 内容哈希**。

    不含绝对路径（那会把本机目录写进包 —— 包里不该有本机路径）；也不含标定时间戳
    （重跑一次标定就会变，于是"同口径"被误判成不可比）。用内容哈希而不是只有文件名：
    同名文件内容不同（换了一台机器、改了 dtype）必须算不同协议。
    """
    p = Path(path)
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = "unreadable"
    return f"{p.name}:{digest}"


def _protocol_meta(args: argparse.Namespace, peaks: Any, peak_gbs: float) -> dict[str, Any]:
    """一次写齐协议哈希要读的键（R-49）。

    `protocol_hash()` 的 payload 早就读 `meta["calibration"]` / `["peak_source"]` /
    `["kernel_filter"]` / `["warmup_s"]` / `["window_s"]`，但 CLI **从不写这些键** —— 于是
    哈希里全是 null。后果：换一份标定（roofline 唯一的分母）、把采样窗口调短，都不算协议
    变化，`diff` 照报"真改进"。协议输入必须由**管线**写，而不是靠测试手工塞 meta。

    只放**稳定**的量：不含路径、不含时间戳（那些每次都变，会把同协议误判成不同）。
    """
    if args.ks_peak_tflops or args.ks_peak_gbps:
        kind, override = "override", {"gbps": args.ks_peak_gbps, "tflops": args.ks_peak_tflops}
    elif args.ks_calibration:
        kind, override = "calibration", None
    else:
        kind, override = "none", None
    out: dict[str, Any] = {
        "kernel_filter": args.kernel_name or "",
        "warmup_s": float(args.ks_warmup or 0.0),
        "window_s": float(args.ks_window_s or 0.0),
        "peak_source": kind,
        "peak_override": override,
        "peaks": {
            "bw_gbs": round(float(peak_gbs), 3),
            "compute_tflops": (round(float(peaks.compute_tflops), 3)
                               if peaks is not None and peaks.compute_tflops else None),
        },
        "calibration": _calibration_identity(args.ks_calibration) if args.ks_calibration else None,
        # R-75：轨迹口径的**同一份轨迹**要能比对 —— 指纹 + 采集方式（完整标记）。
        # 只放稳定量：文件名与内容指纹，不含绝对路径。
        "memtrace": _memtrace_identity(getattr(args, "memtrace", None)),
    }
    return out


def _memtrace_identity(path: str | None) -> dict[str, Any] | None:
    """轨迹身份：文件名 + 内容指纹 + 是否有结束标记（R-75/R-76）。

    没有它，两次 diff 用了不同轨迹（不同采样范围/不同 kernel 子集）也会被当成同协议。
    """
    if not path:
        return None
    import os

    from .backends.memtrace import trace_fingerprint

    out: dict[str, Any] = {"name": os.path.basename(path)}
    try:
        out["sha256"], _ends_nl = trace_fingerprint(path)
    except OSError:
        out["sha256"] = None
    return out


def _strict_problems(rep: Report, args: argparse.Namespace, used_noop: list[str]) -> list[str]:
    """--strict 下"不可得即失败"的判定（采集模式与 -i 回读模式共用）。

    覆盖面（R-20/B7-c）—— 这四类都必须能让门禁变红：
      1. 空转的 ncu 开关（接受但无效果）；
      2. `--metrics` 请求了不可得/不存在的指标；
      3. **`--set` / `--section` 请求了计数器 section**（旧实现只看 --section，`--set full` 漏过）；
      4. **请求了账本对拍但裁决行没能算出来**（旧实现：B1 让这一行静默消失，门禁全绿）。
    外加：`--page source` 请求了未实现的 SASS/源码逐行对照。
    """
    problems: list[str] = []
    if used_noop:
        problems.append(f"no-op ncu flags used: {', '.join(used_noop)}")
    # 0) 环境缺件（W1）：降级跑出来的报告在 --strict 下必须失败 —— 用户要的是 kernel 级
    #    数据，而这台机器给不出来。降级只在非 strict 时是"成功路径"。
    for d in (rep.meta.get("environment_refusals") or []):
        problems.append(f"environment cannot provide requested data: "
                        f"{d.get('code', '?')} - {d.get('message', '')}")
    requested = [n for v in (args.metrics or []) for n in v.split(",") if n.strip()]
    unavailable = rep.unavailable_metrics()
    unavailable_names = {m.name for m in unavailable}
    reason_by_name = {m.name: m.reason for m in unavailable}
    from .ncu_names import explain as explain_ncu_name
    from .ncu_names import resolve as resolve_ncu_name

    for name in requested:
        if name in unavailable_names:
            problems.append(f"requested metric is unavailable: {name} - "
                            f"{reason_by_name.get(name, 'no reason recorded')}")
        elif name in KNOWN_COUNTER_METRICS:
            problems.append(f"requested metric needs hardware counters: {name} - "
                            f"{KNOWN_COUNTER_METRICS[name]}")
        elif name not in _collect_metric_names(rep):
            # D4①：能映射到替代口径的名字，错误信息里必须写出**替代口径叫什么**；
            # 映射表里标了"给不出"的（kind=none）也要说清为什么。
            disc = resolve_ncu_name(name)
            if disc is not None:
                problems.append(f"requested metric cannot be produced under that name: "
                                f"{explain_ncu_name(name)}")
            else:
                problems.append(f"requested metric not found: {name}")
    # 3) 数据驱动的门禁（R-43）：一个 section 只有在**真的没产出任何数据**时才算问题。
    #    从前这里只看名字：即使 --ks-instr-hist 已经把 est__issue_slot_utilization_pct
    #    算出来了，`--section SchedulerStats --strict` 仍然返回 2 —— 门禁在拦自己的成果。
    for s in args.section or []:
        if s in COUNTER_SECTIONS and not _section_produced(rep, s):
            problems.append(f"requested section produced no data: {s} - "
                            f"{counter_section_reason(s)}")
    # 4) --set 也要覆盖：只把 section 名报出来，不把 10 个 section 全铺开
    set_name = getattr(args, "set_name", "basic")
    if not (args.section or []):
        from .sections import SET_SECTIONS

        gated = [s for s in SET_SECTIONS.get(set_name, [])
                 if s in COUNTER_SECTIONS and not _section_produced(rep, s)]
        if gated:
            problems.append(
                f"--set {set_name} requests {len(gated)} section(s) that need hardware counters "
                f"or instrumentation: {', '.join(gated)} - use --set basic for a producible set")
    # 4) 账本对拍：请求了就必须有裁决行（B1 的 CI 侧修法）
    ledger_requested = bool(args.ks_ledger_mb or args.ks_ledger_total_mb
                            or args.ks_ledger_range_mb)
    if ledger_requested:
        for name in ("est__dram_vs_ledger_err_pct", "est__dram_range_position_pct"):
            if name in unavailable_names:
                problems.append(
                    f"requested ledger comparison cannot be produced: {name} - "
                    f"{reason_by_name.get(name, 'no reason recorded')}")
        if not any(n in _collect_metric_names(rep) for n in
                   ("est__dram_vs_ledger_err_pct", "est__dram_range_position_pct")):
            problems.append(
                "requested ledger comparison produced no verdict row at all "
                "(neither a value nor an n/a row) - that is a bug, not a result")
    if getattr(args, "page", "details") == "source":
        problems.append("--page source asks for SASS/source line correlation, which this tool "
                        "does not implement (use --page raw to see each metric's source field)")
    # 5) 成本账（C1）：**给了输入就必须给出对应的数** —— 门禁要能变红，否则
    #    "成本账"就成了一个永远说"稍后再算"的装饰。两类输入各查各的输出。
    def _cost_missing(name: str) -> bool:
        return name not in _collect_metric_names(rep) or name in unavailable_names

    if getattr(args, "ks_price_per_kwh", None) or getattr(args, "ks_price_per_hour", None):
        if _cost_missing("ks__cost_per_hour"):
            problems.append(
                "cost ledger cannot price this run: "
                + (reason_by_name.get("ks__cost_per_hour") or "未产出 ks__cost_per_hour"))
    if getattr(args, "ks_tokens", None):
        for name in ("ks__tokens_per_s", "ks__cost_per_mtok"):
            if _cost_missing(name):
                problems.append(
                    f"cost ledger cannot produce {name} with --ks-tokens given: "
                    + (reason_by_name.get(name) or f"{name} 未产出"))
    return problems


def _report_strict(problems: list[str]) -> None:
    for p in problems:
        print(f"error(strict): {p}", file=sys.stderr)
    print("error(strict): refusing to continue - some requested data cannot be produced "
          "honestly on this machine. Re-run without --strict to get 'n/a' rows instead.",
          file=sys.stderr)


def _section_produced(rep: Report, name: str) -> bool:
    """这个 section 是否**真的产出了数据**（有任一非 unavailable 指标）。

    R-43：门禁要按产出判定，而不是按请求的名字判定。
    """
    from .provenance import Provenance

    for k in rep.kernels:
        for s in k.sections:
            if s.name == name and any(m.provenance is not Provenance.UNAVAILABLE
                                      for m in s.metrics):
                return True
    return False


def _collect_metric_names(rep: Report) -> set[str]:
    return {m.name for k in rep.kernels for s in k.sections for m in s.metrics}


def _ensure_parent(path: str, logged: list[str] | None = None) -> None:
    """输出文件的父目录不存在就建出来。

    A2 的"声明 vs 观察"对账测试抓到的：`-o out/rep.json`（out 不存在）会让
    `.write_text()` 抛 FileNotFoundError 裸 traceback，而计划里明明写着"会写这个文件"。
    一个 profiler 因为目录不存在就崩掉、且崩在写报告这一步（几分钟的采集白费），
    比它多建一个目录糟得多。
    """
    parent = Path(path).parent
    if str(parent) in ("", ".") or parent.exists():
        return
    parent.mkdir(parents=True, exist_ok=True)
    if logged is not None:
        logged.append(f"note: 输出目录不存在，已创建 {parent}/")


def _strip_repeat(argv: Sequence[str]) -> list[str]:
    """去掉 `--repeat N` / `--repeat=N`（子进程不该再进重复路径）。"""
    out: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "--repeat":
            skip_next = True
            continue
        if a.startswith("--repeat="):
            continue
        out.append(a)
    return out


def _repeat_path(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """`--repeat N`：把整条命令跑 N 次（进程级），给出跨运行分布与 MDD 下界。

    为什么用**自我调用**而不是进程内循环：每次运行都需要自己的采集、报告与案件身份，
    进程内循环会让它们互相污染，也很容易长成"第二条测量路径"。子进程走的是与单次运行
    **完全相同**的那段代码，所以重复执行不引入新的口径。

    它不是 kernel 级重放：不插桩、不锁时钟、不碰其他租户，也**不控制 warm/cold 缓存** ——
    所以真实噪声只会比这里给的下界更大（喂给 `diff` 时与误差带取最大值）。
    """
    from . import repeat as repeat_mod

    n = int(args.repeat)
    child_argv = _strip_repeat(argv)
    work = tempfile.mkdtemp(prefix="ks-repeat-")
    runs: list[tuple[int, str]] = []
    try:
        for i in range(n):
            packet_path = os.path.join(work, f"run{i}.packet.json")
            cmd = [sys.executable, "-m", "kernoscope", *child_argv,
                   "--packet", packet_path, "--screen", "off", "--no-case", "--quiet"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            runs.append((proc.returncode, packet_path))
        counter = {"i": 0}

        def runner():
            rc, path = runs[counter["i"]]
            counter["i"] += 1
            try:
                return rc, json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return rc, None

        result = repeat_mod.repeat_runs(runner, n)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    blob = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if getattr(args, "export", None):
        export = str(args.export)
        base = export[:-len(".json")] if export.endswith(".json") else export
        out = f"{base}.repeat.json"
        _write_text_or_fail(blob, out, what="重复执行分布", logged=[])
    if getattr(args, "packet", None) == "-":
        sys.stdout.write(blob + "\n")          # agent 要机器格式就只给机器格式
    elif not getattr(args, "quiet", False):
        print(result.render())
    return 0 if result.mdd_lower_bound else 1


def _emit(rep: Report, args: argparse.Namespace, compat_mode: bool, logged: list[str],
          packet_ctx: dict[str, Any] | None = None) -> int:
    if compat_mode and not args.quiet:
        print("==PROF== kernoscope running in ncu compatibility mode "
              "(symlinked as 'ncu'); data sources differ from NVIDIA Nsight Compute.",
              file=sys.stderr)

    # D4①：`--metrics dram__bytes.sum` 这类**计数器名**是发现键 —— 它不该得到一句
    # "指标不存在"，而应拿到替代口径。展开在这里做（渲染用），原始请求名保留给
    # `--strict` 报告"你请求的是计数器，我们给的是替代口径"。
    from .ncu_names import resolve as resolve_ncu_name

    metrics = [n for v in (args.metrics or []) for n in v.split(",") if n.strip()]
    expanded: list[str] = []
    for name in metrics:
        d = resolve_ncu_name(name)
        if d is not None and d.ours and d.ours not in expanded:
            expanded.append(d.ours)
            if d.ours != name:
                print(f"==PROF== note: --metrics {name} → 替代口径 {d.ours}"
                      f"（{d.caliber}）；数值不会挂在 {name} 名下", file=sys.stderr)
        elif d is None and name not in expanded:
            expanded.append(name)
    metrics = expanded

    # 落盘
    if args.export:
        base = args.export
        if base.endswith(".json"):
            base = base[:-len(".json")]
        json_path = base + ".ksreport.json"
        _ensure_parent(json_path, logged)
        p = Path(json_path)
        if p.exists() and not args.force_overwrite:
            print(f"error: {json_path} exists (pass -f to overwrite)", file=sys.stderr)
            return 1
        # V-25：只读目录/无权限写报告曾经是裸 PermissionError（rc=1、无 error: 行）。
        _write_text_or_fail(rep.to_json(include_launches=not args.ks_json_no_launches),
                            json_path, what="报告 JSON", logged=logged)
        rep.meta["out_json"] = json_path
        if getattr(args, "launch_csv", False):
            lp = base + ".launches.csv"
            _write_text_or_fail(render_launch_csv(rep), lp, what="逐 launch 表（CSV）", logged=logged)
            rep.meta["out_launch_csv"] = lp
        perfetto_path = args.ks_perfetto or (base + ".perfetto.json")
        _ensure_parent(perfetto_path, logged)
        q = Path(perfetto_path)
        if q.exists() and not args.force_overwrite:
            print(f"error: {perfetto_path} exists (pass -f to overwrite)", file=sys.stderr)
            return 1
        _write_text_or_fail(render_perfetto(rep), perfetto_path, what="时间线（Perfetto）JSON",
                            logged=logged)
        rep.meta["out_perfetto"] = perfetto_path
    elif args.ks_perfetto:
        _write_text_or_fail(render_perfetto(rep), args.ks_perfetto, what="时间线（Perfetto）JSON",
                            logged=logged)
        rep.meta["out_perfetto"] = args.ks_perfetto

    # ---------- W2/W4：决策包（机器视图）与一屏（人视图）----------
    # 两者都由同一份 packet 渲染：人视图只是 packet 的一种排版，不是另一条代码路径。
    screen = getattr(args, "screen", None) or "auto"
    # M14「首用 ≤1 屏」：README 承诺"一屏"，而 auto 的语义是"一屏 **+** 完整报告"——
    # 实测 `--set basic` 21.6 KB / 200+ 行，与承诺不符（易用性实测的 #3）。
    # 默认（没显式给 --screen）在**终端**上收敛成 only；管道/重定向/CSV 保持完整报告，
    # 所以脚本、agent、CI 看到的字节一个都不变。要老行为就显式写 `--screen auto`。
    if (getattr(args, "screen", None) is None and sys.stdout.isatty()
            and not (args.csv or getattr(args, "launch_csv", False))):
        screen = "only"
    if getattr(args, "packet", None) == "-" and screen == "auto":
        # stdout 上不能混着两种格式：要包就只给包
        screen = "off"
    if (args.csv or getattr(args, "launch_csv", False)) and screen == "auto":
        # CSV 是机器格式，前面铺一屏人话会破坏它；要一屏请显式 --screen only
        screen = "off"
    packet = None
    # V-26③：案件目录（`-o <前缀>` 默认写 `<前缀>.case/`）此前只在"要打一屏或要决策包"时
    # 才建 —— `--screen off`/`--csv`/`--launch-csv` 会**静默不写** case，而 README 只写
    # `--no-case` 能关。语义统一：只要没显式 `--no-case`，就把 case 写出来。
    if (getattr(args, "packet", None) or screen != "off" or getattr(args, "html", None)
            or (not getattr(args, "no_case", False) and getattr(args, "export", None))):
        from .packet import build, dumps, render_onepage, write_packet

        ctx = packet_ctx or {}
        packet = build(rep, capabilities=ctx.get("capabilities"),
                       selfcheck=ctx.get("selfcheck"),
                       basis=str(rep.meta.get("ledger_basis") or ""),
                       report_path=rep.meta.get("out_json", ""))
        blob = dumps(packet)
        if getattr(args, "packet", None) == "-":
            sys.stdout.write(blob + "\n")
        elif getattr(args, "packet", None):
            # V-25：只读目录写决策包曾经是裸 PermissionError/FileNotFoundError 且崩在
            # 写完报告之后。写不了就是一行自家错误 + rc=1。
            _ensure_parent(str(args.packet), logged)
            try:
                write_packet(packet, args.packet)
            except OSError as e:
                raise CliInputError(
                    f"写不了决策包：{args.packet}（{e.strerror or e}）",
                    hint=_io_remedy(e)) from e
            print(f"==PROF== 决策包: {args.packet} "
                  f"({len(blob.encode('utf-8'))} B)", file=sys.stderr)
        if screen in ("auto", "only"):
            print(render_onepage(packet))
            if screen == "auto" and not (args.csv or getattr(args, "launch_csv", False)):
                print("")
                print("=" * 100)
                print("完整报告（上面的判词都出自这里的数据；每个数字都带出处）")
                print("=" * 100)

    # ---------- I2/M4：把替代口径导出成 .ncu-rep（+ 旁路署名）----------
    if getattr(args, "export_ncu_rep", None):
        # `_emit` 手里只有 `logged`（一个 list），而 `_export_substitute` 要的是打印函数 ——
        # 真机验收时这里传错成 list，`log(...)` 抛 TypeError，让进程崩在**写完文件之后**。
        rc_export = _export_substitute(rep, args, packet, _stderr_log(args, logged))
        if rc_export:
            return rc_export

    # ---------- I1：真值对照产物（机器可读，可归档、可 diff）----------
    if rep.meta.get("ncu_truth") and rep.meta.get("out_json"):
        truth_path = re.sub(r"\.ksreport\.json$", "", str(rep.meta["out_json"])) + ".ncu-truth.json"
        _write_text_or_fail(json.dumps(rep.meta["ncu_truth"], ensure_ascii=False, indent=2),
                            truth_path, what="真值对拍 JSON", logged=logged)
        rep.meta["out_ncu_truth"] = truth_path
        print(f"==PROF== 真值对拍: {truth_path}", file=sys.stderr)

    # ---------- D4②：ncu 名字对照表（声明，不是数据替换）----------
    if getattr(args, "export_name_map", None):
        from .ncu_names import render_name_map

        fmt = "csv" if str(args.export_name_map).lower().endswith(".csv") else "json"
        # V-25：只读目录写名字对照表曾经是裸 PermissionError。
        _write_text_or_fail(render_name_map(fmt), args.export_name_map,
                            what="名字对照表", logged=logged)
        rep.meta["out_name_map"] = str(args.export_name_map)
        print(f"==PROF== ncu 名字对照表: {args.export_name_map}（{fmt}）", file=sys.stderr)

    # ---------- N9：单文件 HTML 证据单（给人看/截图）----------
    if getattr(args, "html", None):
        rc_html = _write_html_view(packet, args, logged)
        if rc_html:
            return rc_html

    # ---------- I3：人读的 per-kernel 报告（Markdown）----------
    if getattr(args, "report_md", None):
        rc_md = _write_report_md(rep, args, packet, logged)
        if rc_md:
            return rc_md

    # 标准输出
    if getattr(args, "packet", None) == "-":
        # 要包就只给包：stdout 上不混两种格式（谁要报告谁自己加 -o）
        pass
    elif screen == "only" and not (args.csv or getattr(args, "launch_csv", False)):
        pass
    elif getattr(args, "launch_csv", False):
        sys.stdout.write(render_launch_csv(rep))
    elif args.csv:
        sys.stdout.write(render_csv(rep, metrics))
    else:
        print(render_text(rep, quiet=args.quiet, summary=args.print_summary, page=args.page,
                          metrics=metrics, print_details=args.print_details,
                          print_units=args.print_units,
                          print_metric_name=args.print_metric_name))

    # 成本可预算（R-22）：报告体积与时长要能被看见、能被限流
    sizes = {k: os.path.getsize(v) for k, v in
             (("json", rep.meta.get("out_json")), ("perfetto", rep.meta.get("out_perfetto")),
              ("launch_csv", rep.meta.get("out_launch_csv"))) if v and os.path.exists(v)}
    if sizes:
        blob = " / ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items())
        print(f"==PROF== 输出体积: {blob}"
              f"（{sum(len(k.launches) for k in rep.kernels)} 次 launch）", file=sys.stderr)
        if sum(sizes.values()) > 2 * 1024 * 1024 or sum(len(k.launches) for k in rep.kernels) > 1000:
            print("==PROF== 提示：体积主要来自逐 launch 记录。要限流可用 -c <N>（每个 kernel 只报 "
                  "N 次）、--ks-json-no-launches（JSON 不带逐 launch 数组）、或 --launch-csv 单独出表",
                  file=sys.stderr)

    if args.log_file:
        # V-25：只读目录写日志曾经是裸 PermissionError（且崩在报告已经打完之后）。
        _write_text_or_fail("\n".join(logged) + "\n", args.log_file, what="日志文件（--log-file）")
    rep.meta["_packet"] = packet
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
