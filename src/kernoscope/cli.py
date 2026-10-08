"""kernoscope CLI —— 与 ncu 用法兼容。

选项按四类处理（见 docs/ncu-compat.md）：
  ① 真实现：kernel 过滤 / launch 选择 / 输出与格式类
  ② 映射实现：--set / --section / --metrics → 我们的数据源，缺则标 unavailable
  ③ 接受但空转：--replay-mode / --cache-control / --clock-control / --pm-sampling-* 等
  ④ 自有扩展：--ks-*（数据源、标定、workload、synthetic）

`argv[0]` 为 `ncu` 时自动进入兼容模式（供 `ln -s kernoscope ncu` 使用）。

退出码：
  0 正常；2 --strict 下请求了不可得的数据（供 CI 卡门）；1 运行错误。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
from .provenance import Metric, Provenance, UNAVAILABLE_REASONS
from .report import DeviceInfo, KernelLaunch, KernelReport, Report
from .render import render_csv, render_perfetto, render_text
from .sections import (COUNTER_SECTIONS, KNOWN_COUNTER_METRICS, BuildOptions,
                       build_device_kernel, build_dram_section, build_kernel_sections,
                       requested_sections)

PROG = "kernoscope"

NOOP_FLAGS = [
    "--replay-mode", "--app-replay-mode", "--app-replay-buffer", "--app-replay-match",
    "--cache-control", "--clock-control", "--pm-sampling-buffer-size", "--pm-sampling-interval",
    "--pm-sampling-max-passes", "--warp-sampling-buffer-size", "--warp-sampling-interval",
    "--warp-sampling-max-passes", "--mps", "--mps-num-clients", "--mps-timeout",
    "--pipeline-boost-state", "--preload-library", "--import-sass", "--import-source",
]


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
               "  kernoscope --ks-synthetic --set detailed | head -50\n",
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
    g.add_argument("--devices", default=None, help="(accepted; 单卡环境取第一个)")

    o = p.add_argument_group("ncu-compatible: output")
    o.add_argument("-o", "--export", default=None, help="输出基名（写 <base>.ksreport.json）")
    o.add_argument("-f", "--force-overwrite", action="store_true")
    o.add_argument("--csv", action="store_true", help="以 CSV 输出（含 provenance 列）")
    o.add_argument("--print-summary", default="none", choices=["none", "per-kernel"])
    o.add_argument("--print-details", default="header", help="(accepted)")
    o.add_argument("--print-metric-name", default="name", help="(accepted)")
    o.add_argument("--print-units", default="base", choices=["base", "auto"], help="(accepted)")
    o.add_argument("--page", default="details", choices=["details", "raw", "source"],
                   help="details=默认；raw=额外打印每个指标的数据来源")
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
    k.add_argument("--ks-memtrace", default=None,
                   help="NVBit 访存轨迹（ks-memtrace v1）：给出 sectors/request、合并效率、"
                        "L1/L2 命中率模型与 trace 口径的 DRAM 字节")
    k.add_argument("--ks-l2-bytes", type=int, default=0, help="覆盖 L2 容量（字节）")
    k.add_argument("--ks-keep-intermediate", action="store_true",
                   help="保留 nsys 中间产物（.nsys-rep/.sqlite）；给了 -o 时默认保留在其旁边")
    k.add_argument("--strict", action="store_true",
                   help="请求了不可得数据时 exit 2（CI 卡门）；空转选项也视为错误")

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


def collect(args: argparse.Namespace, log) -> tuple[list[KernelLaunch], dict[str, Any], int]:
    """返回 (launches, extra_meta, target_exit_code)。"""
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

    cmd = [a for a in args.program if a != "--"]
    if not cmd:
        log("error: no target command. Put it after '--', or use --ks-sqlite/--ks-synthetic.")
        raise SystemExit(1)
    meta["target"] = " ".join(cmd)
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


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 子命令（不属于 ncu 兼容面）
    if argv and argv[0] == "doctor":
        from .doctor import doctor_main

        return doctor_main(argv[1:])
    if argv and argv[0] == "occupancy":
        from .occupancy_cmd import occupancy_main

        return occupancy_main(argv[1:])
    if argv and argv[0] == "memtrace":
        from .memtrace_cmd import memtrace_main

        return memtrace_main(argv[1:])

    args = build_parser().parse_args(argv)
    compat_mode = os.path.basename(sys.argv[0] or "") == "ncu"

    logged: list[str] = []

    def log(msg: str) -> None:
        logged.append(msg)
        if not args.quiet:
            print(msg, file=sys.stderr)

    # ---------- 回读模式 ----------
    if args.import_report:
        rep = Report.from_dict(json.loads(Path(args.import_report).read_text()))
        used_noop_import = [f for f in NOOP_FLAGS
                            if getattr(args, f.lstrip("-").replace("-", "_"), None) is not None]
        problems = _strict_problems(rep, args, used_noop_import)
        if args.strict and problems:
            _report_strict(problems)
            return 2
        return _emit(rep, args, compat_mode, logged)

    if args.list_sets:
        from .sections import SET_SECTIONS

        print("Available sets and sections:")
        for name, secs in SET_SECTIONS.items():
            print(f"  {name:<9} {', '.join(secs)}")
        print("\nSections that require GPU performance counters are always reported as "
              "unavailable with a reason (never as substituted numbers).")
        return 0

    # ---------- 采集 ----------
    sampler_ctx = Sampler(enabled=not args.ks_no_nvml)
    t0 = time.monotonic()
    with sampler_ctx as sampler:
        launches, meta, target_rc = collect(args, log)
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
        prof = json.loads(Path(args.ks_calibration).read_text())
        peaks = MachinePeaks.from_profile(prof, args.ks_dtype)
    if args.ks_peak_tflops or args.ks_peak_gbps:
        peaks = MachinePeaks.from_spec(
            args.ks_peak_tflops or (peaks.compute_tflops if peaks else 0.0),
            args.ks_peak_gbps or (peaks.bandwidth_gbs if peaks else 0.0),
            args.ks_dtype,
        )
    workload = None
    workload_note = ""
    if args.ks_gemm:
        try:
            m, n, k = (int(float(x)) for x in args.ks_gemm.replace("x", ",").split(","))
            workload = gemm_workload(m, n, k, args.ks_dtype)
            workload_note = f"user-provided GEMM {m}x{n}x{k} ({args.ks_dtype})"
        except Exception:
            log(f"error: bad --ks-gemm {args.ks_gemm!r}; expected M,N,K")
            return 1
    elif args.ks_flops and args.ks_bytes:
        workload = (float(args.ks_flops), float(args.ks_bytes))
        workload_note = f"user-provided FLOPs={args.ks_flops:g} bytes={args.ks_bytes:g}"
    elif args.ks_flops or args.ks_bytes:
        log("error: --ks-flops and --ks-bytes must be provided together")
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
        achieved = json.loads(Path(args.ks_achieved_occupancy).read_text())

    # ---------- 过滤 ----------
    if not launches:
        log("error: no kernel activity found (was the run collected with --trace=cuda?)")
        return 1
    matched = select_launches(launches, args.kernel_name, args.launch_skip,
                              args.launch_count, args.launch_skip_before_match)
    if not matched:
        names = sorted({l.demangled for l in launches})[:8]
        log(f"error: no kernels matched {args.kernel_name!r}. Available examples: {names}")
        return 1

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
            log(f"error: --ks-ledger-range-mb 解析失败（{e}）；应形如 LOW,HIGH")
            return 1

    # ---------- DRAM 流量估算（NVML 代理）----------
    from .analyze.dram import estimate as estimate_dram

    memtrace_stats = {}
    if args.ks_memtrace:
        from .analyze.memtrace import analyze_memtrace
        from .backends.memtrace import MemTraceError, load_memtrace

        try:
            _tr = load_memtrace(args.ks_memtrace)
            _st = analyze_memtrace(_tr, l2_bytes=args.ks_l2_bytes or None)
            memtrace_stats[_st.kernel] = _st
            log(f"memtrace: {_st.kernel} 载入 {_st.records} 条记录")
        except (OSError, MemTraceError) as e:
            log(f"error: 访存轨迹读取失败: {e}")
            return 1

    grouped_dram: dict[str, list[KernelLaunch]] = {}
    for l in matched:
        grouped_dram.setdefault(l.display_name(args.kernel_name_base), []).append(l)
    peak_gbs = float(args.ks_peak_gbps or (peaks.bandwidth_gbs if peaks else 0.0) or 0.0)
    launches_for_dram = len(next(iter(grouped_dram.values()))) if len(grouped_dram) == 1 else 0
    dram_est = estimate_dram(
        sampler.result.samples, peak_gbs=peak_gbs, window_s=wall_ns / 1e9,
        launches=launches_for_dram,
        ledger_per_launch=ledger_per_launch, ledger_total=ledger_total,
        ledger_range=ledger_range,
    )
    if len(grouped_dram) != 1:
        dram_est.notes.append(
            f"匹配到 {len(grouped_dram)} 个 kernel：per-launch 口径不适用，只给总量")

    # ---------- 组装报告 ----------
    from .analyze.occupancy import arch_for

    arch = arch_for(device.compute_capability or "8.9")
    opt = BuildOptions(
        set_name=args.set_name,
        sections=list(args.section),
        peaks=peaks,
        workload=workload,
        workload_note=workload_note,
        achieved_occupancy=achieved,
        cubin_res=cubin_res,
        cubin_sass=cubin_sass,
        nvml_summary=sampler.result.summary(),
        sm_count=sm_count,
        device_name=device.name,
        arch=arch,
        peak_gbs=peak_gbs,
        dram=dram_est,
        memtrace=memtrace_stats,
    )

    rep = Report(device=device)
    rep.meta.update(meta)
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
        rep.kernels.append(kr)
    dev_kr = KernelReport("(device)", launch_count=0)
    dev_sections = [build_device_kernel(opt)]
    if "DramTraffic" in requested_sections(opt):
        dram_sec = build_dram_section(opt)
        if dram_sec is not None:
            dev_sections.append(dram_sec)
    dev_kr.sections = dev_sections
    rep.kernels.append(dev_kr)

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
    rep.generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if dram_est.reliable and dram_est.bytes_total > 0:
        summary = (f"DRAM 估算 {dram_est.bytes_total / 1e6:.1f} MB"
                   f"（忙碌 {dram_est.busy_pct_mean:.1f}% × 峰值 {dram_est.peak_gbs:.0f} GB/s，"
                   f"占空比 {dram_est.duty_pct:.0f}%）")
        if dram_est.bytes_per_launch is not None:
            summary += f" → {dram_est.bytes_per_launch / 1e6:.2f} MB/launch"
        if dram_est.ledger_basis:
            ref = dram_est.ledger_per_launch if dram_est.ledger_per_launch is not None else dram_est.ledger_total
            basis = "per-launch" if dram_est.ledger_basis == "per-launch" else "总量"
            if dram_est.ledger_err_pct is None:
                summary += f"；账本对拍（{basis}）口径不成立，未给误差"
            else:
                summary += (f"；vs 账本 {ref / 1e6:.1f} MB（{basis}），"
                            f"误差 {dram_est.ledger_err_pct:+.1f}%")
        if dram_est.ledger_range:
            summary += f"；{dram_est.range_verdict}"
        rep.meta["dram_summary"] = summary
    else:
        rep.meta["dram_summary_unavailable"] = "；".join(dram_est.notes)

    # ---------- 空转选项告警 ----------
    used_noop = [f for f in NOOP_FLAGS
                 if getattr(args, f.lstrip("-").replace("-", "_"), None) is not None]
    if used_noop:
        log(f"warn: these flags are accepted for ncu compatibility but have no effect here: "
            f"{', '.join(used_noop)} (our data source has no counter/replay passes)")

    # ---------- 输出 ----------
    rc = _emit(rep, args, compat_mode, logged)

    # ---------- --strict ----------
    if args.strict:
        problems = _strict_problems(rep, args, used_noop)
        if problems:
            _report_strict(problems)
            return 2

    if args.check_exit_code and target_rc:
        return target_rc
    return rc


def _strict_problems(rep: Report, args: argparse.Namespace, used_noop: list[str]) -> list[str]:
    """--strict 下"不可得即失败"的判定（采集模式与 -i 回读模式共用）。"""
    problems: list[str] = []
    if used_noop:
        problems.append(f"no-op ncu flags used: {', '.join(used_noop)}")
    requested = [n for v in (args.metrics or []) for n in v.split(",") if n.strip()]
    unavailable = rep.unavailable_metrics()
    unavailable_names = {m.name for m in unavailable}
    reason_by_name = {m.name: m.reason for m in unavailable}
    for name in requested:
        if name in unavailable_names:
            problems.append(f"requested metric is unavailable: {name} - "
                            f"{reason_by_name.get(name, 'no reason recorded')}")
        elif name in KNOWN_COUNTER_METRICS:
            problems.append(f"requested metric needs hardware counters: {name} - "
                            f"{KNOWN_COUNTER_METRICS[name]}")
        elif name not in _collect_metric_names(rep):
            problems.append(f"requested metric not found: {name}")
    for s in args.section or []:
        if s in COUNTER_SECTIONS:
            problems.append(f"requested section needs hardware counters: {s}")
    return problems


def _report_strict(problems: list[str]) -> None:
    for p in problems:
        print(f"error(strict): {p}", file=sys.stderr)
    print("error(strict): refusing to continue - some requested data cannot be produced "
          "honestly on this machine. Re-run without --strict to get 'n/a' rows instead.",
          file=sys.stderr)


def _collect_metric_names(rep: Report) -> set[str]:
    return {m.name for k in rep.kernels for s in k.sections for m in s.metrics}


def _emit(rep: Report, args: argparse.Namespace, compat_mode: bool, logged: list[str]) -> int:
    if compat_mode and not args.quiet:
        print("==PROF== kernoscope running in ncu compatibility mode "
              "(symlinked as 'ncu'); data sources differ from NVIDIA Nsight Compute.",
              file=sys.stderr)

    # 落盘
    if args.export:
        base = args.export
        if base.endswith(".json"):
            base = base[:-len(".json")]
        json_path = base + ".ksreport.json"
        p = Path(json_path)
        if p.exists() and not args.force_overwrite:
            print(f"error: {json_path} exists (pass -f to overwrite)", file=sys.stderr)
            return 1
        p.write_text(rep.to_json())
        rep.meta["out_json"] = json_path
        perfetto_path = args.ks_perfetto or (base + ".perfetto.json")
        q = Path(perfetto_path)
        if q.exists() and not args.force_overwrite:
            print(f"error: {perfetto_path} exists (pass -f to overwrite)", file=sys.stderr)
            return 1
        q.write_text(render_perfetto(rep))
        rep.meta["out_perfetto"] = perfetto_path
    elif args.ks_perfetto:
        Path(args.ks_perfetto).write_text(render_perfetto(rep))
        rep.meta["out_perfetto"] = args.ks_perfetto

    # 标准输出
    if args.csv:
        sys.stdout.write(render_csv(rep))
    else:
        print(render_text(rep, quiet=args.quiet, summary=args.print_summary, page=args.page))

    if args.log_file:
        Path(args.log_file).write_text("\n".join(logged) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
