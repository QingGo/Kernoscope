"""渲染器：ncu 风格文本 / CSV（含 provenance 列）/ Perfetto trace。

三个显示开关是真的（B6/R-20 的修法：不能让"接受但静默忽略"存在）：
  * `--print-units base|auto`  base = 原始单位与数值；auto = 自适应换单位（MB→GB/TB）
  * `--print-details all|header`  header = 只打印 kernel/section 头与计数，不铺开指标
  * `--print-metric-name name|full`  full = 带上 section 前缀（`DramTraffic.est__dram_bytes_total`）
另外 `--metrics <名>` 在宽松模式下**真的筛选输出**（R-21）：只打印匹配的指标，
但匹配到的 n/a 行照常打印（带原因）—— 筛选不得把"拒绝给数"筛掉。
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

from ..provenance import Metric, Provenance
from ..report import Report

WIDTH = 46


def _fmt_value(value, unit: str, print_units: str = "auto") -> tuple[str, str]:
    """文本展示用的单位：`auto` 时 MB 超 1000 就换 GB/TB（JSON/CSV 仍保持原始单位）。"""
    if not isinstance(value, (int, float)):
        return str(value), unit
    if print_units == "auto" and unit == "MB":
        if abs(value) >= 1e6:
            return f"{value / 1e6:.3f}", "TB"
        if abs(value) >= 1e3:
            return f"{value / 1e3:.2f}", "GB"
    return f"{value:.6g}" if isinstance(value, float) else str(value), unit


def _wanted(name: str, metrics: Iterable[str] | None) -> bool:
    return not metrics or name in set(metrics)


def render_text(rep: Report, *, quiet: bool = False, summary: str = "none",
                page: str = "details", metrics: Iterable[str] | None = None,
                print_details: str = "all", print_units: str = "auto",
                print_metric_name: str = "name") -> str:
    metrics = list(metrics or [])
    out: list[str] = []
    if not quiet:
        out.append(f"==PROF== Kernoscope {rep.tool_version} - provenance-honest profile "
                   f"(no performance counters required)")
        meta = rep.meta
        if meta.get("target"):
            out.append(f"==PROF== Target: {meta['target']}")
        d = rep.device
        if d.name:
            out.append(f"==PROF== Device: {d.name}"
                       + (f" (sm_{str(d.compute_capability).replace('.', '')})" if d.compute_capability else "")
                       + (f" driver {d.driver}" if d.driver else "")
                       + (f"  L2 {d.l2_mib:.0f} MiB" if d.l2_mib else ""))
        if rep.meta.get("dram_summary"):
            out.append(f"==PROF== {rep.meta['dram_summary']}")
        elif rep.meta.get("dram_summary_unavailable"):
            out.append(f"==PROF== DRAM 估算不可用：{rep.meta['dram_summary_unavailable'][:160]}")
        if rep.meta.get("kernel_time_pct") is not None:
            out.append(f"==PROF== Kernel time: {rep.total_kernel_ns / 1e6:.3f} ms of "
                       f"{rep.wall_time_ns / 1e6:.3f} ms wall ({rep.meta['kernel_time_pct']:.1f}%)")
        if rep.meta.get("window"):
            out.append(f"==PROF== Window: {rep.meta['window']}")
        for k, v in (rep.sources or {}).items():
            out.append(f"==PROF== source[{k}]: {v}")
        out.append("")

    if summary == "per-kernel":
        out.append(f"{'kernel':<52} {'launches':>8} {'total_ms':>10} {'mean_ms':>10}  configs")
        for k in rep.kernels:
            name = k.name if len(k.name) <= 51 else k.name[:48] + "..."
            cfg = f"{len(k.configs)} 组" + ("（混合！）" if k.is_mixed_config else "")
            out.append(f"{name:<52} {k.launch_count:>8} {k.total_ns / 1e6:>10.4f} "
                       f"{k.mean_ns / 1e6:>10.4f}  {cfg}")
        out.append("")
        return "\n".join(out)

    for k in rep.kernels:
        title = k.name if len(k.name) <= 70 else k.name[:67] + "..."
        out.append(f"Kernel: {title}")
        if getattr(k, "alias", ""):
            out.append(f"  nsys short name: {k.alias}")
        cfg_note = ""
        if k.is_mixed_config:
            keys = ", ".join(c.key.replace("_block", " block").replace("_smem", " smem")
                             for c in k.configs[:6])
            cfg_note = (f"  ⚠ {len(k.configs)} 组 launch 配置（{keys}"
                        + ("…" if len(k.configs) > 6 else "") + "）：逐组读数见 LaunchConfigs")
        out.append(f"  {k.launch_count} launch(es), {k.total_ns / 1e6:.4f} ms total, "
                   f"{k.mean_ns / 1e6:.4f} ms mean")
        if cfg_note:
            out.append(cfg_note)
        for s in k.sections:
            shown = [m for m in s.metrics if _wanted(m.name, metrics)]
            if metrics and not shown:
                continue
            tag = s.provenance.tag
            extra = f" (+{s.n_unavailable} n/a)" if s.n_unavailable else ""
            out.append("")
            out.append(f"  Section: {s.name:<34} {tag}{extra}")
            if s.note:
                out.append(f"    note: {s.note}")
            if print_details == "header":
                if shown:
                    out.append(f"    ({len(shown)} metric(s) hidden by --print-details header)")
                continue
            for m in shown:
                name = f"{s.name}.{m.name}" if print_metric_name == "full" else m.name
                if m.provenance is Provenance.UNAVAILABLE:
                    out.append(f"    {name:<{WIDTH}} {'n/a':>14} {'':<8} "
                               f"{m.provenance.tag}  {m.reason[:160]}")
                else:
                    value_str, unit = _fmt_value(m.value, m.unit or "", print_units)
                    line = f"    {name:<{WIDTH}} {value_str:>14} {unit:<8} {m.provenance.tag}"
                    if m.resolution:
                        line += f"  [resolution {m.resolution}]"
                    if m.note:
                        line += f"  ({m.note})"
                    out.append(line)
                if page in ("raw", "source") and m.source:
                    out.append(f"        source: {m.source}")
        out.append("")
        out.append("-" * 100)

    hist = rep.provenance_histogram()
    out.append("")
    out.append("Provenance summary: " + ", ".join(f"{k}={v}" for k, v in hist.items() if v))
    out.append("Legend: exact=与 ncu 同源 | derived=本机实测标定/模型 | measured=我们的探针实测 | "
               "estimated=采样或代理(附误差) | unavailable=需要硬件计数器，只报 n/a")
    if metrics:
        out.append("Filtered by --metrics: " + ", ".join(metrics))
    for key in ("json", "csv", "perfetto"):
        if rep.meta.get(f"out_{key}"):
            out.append(f"Output({key}): {rep.meta[f'out_{key}']}")
    if rep.meta.get("out_perfetto"):
        out.append("Open the timeline at https://ui.perfetto.dev -> Load trace -> the file above")
    return "\n".join(out)


CSV_COLUMNS = ["kernel", "section", "section_provenance", "metric", "value", "unit",
               "provenance", "resolution", "source", "reason", "note"]

LAUNCH_CSV_COLUMNS = ["kernel", "launch_index", "config", "grid_x", "grid_y", "grid_z",
                      "block_x", "block_y", "block_z", "start_ns", "end_ns", "duration_ns",
                      "registers_per_thread", "static_smem_bytes", "dynamic_smem_bytes",
                      "stream_id", "device_id", "provenance"]


def render_csv(rep: Report, metrics: Iterable[str] | None = None) -> str:
    wanted = set(metrics or [])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for k in rep.kernels:
        for s in k.sections:
            for m in s.metrics:
                if wanted and m.name not in wanted:
                    continue
                w.writerow({
                    "kernel": k.name,
                    "section": s.name,
                    "section_provenance": s.provenance.value,
                    "metric": m.name,
                    "value": "" if m.value is None else m.value,
                    "unit": m.unit,
                    "provenance": m.provenance.value,
                    "resolution": m.resolution,
                    "source": m.source,
                    "reason": m.reason,
                    "note": m.note,
                })
    return buf.getvalue()


def render_launch_csv(rep: Report) -> str:
    """**每个 launch 一行**（R-04/B4）：形状扫描直接落到表格里，不再被聚合成均值。"""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LAUNCH_CSV_COLUMNS)
    w.writeheader()
    from ..report import config_key

    for k in rep.kernels:
        for i, l in enumerate(sorted(k.launches, key=lambda x: x.start_ns)):
            w.writerow({
                "kernel": k.name, "launch_index": i, "config": config_key(l),
                "grid_x": l.grid[0], "grid_y": l.grid[1], "grid_z": l.grid[2],
                "block_x": l.block[0], "block_y": l.block[1], "block_z": l.block[2],
                "start_ns": l.start_ns, "end_ns": l.end_ns, "duration_ns": l.duration_ns,
                "registers_per_thread": l.registers_per_thread,
                "static_smem_bytes": l.static_smem_bytes,
                "dynamic_smem_bytes": l.dynamic_smem_bytes,
                "stream_id": l.stream_id, "device_id": l.device_id,
                "provenance": "exact",
            })
    return buf.getvalue()


def render_perfetto(rep: Report) -> str:
    """Chrome Trace Event Format（Perfetto / chrome://tracing 可直接打开）。"""
    events: list[dict[str, Any]] = []
    base_ns: int | None = None
    for k in rep.kernels:
        for l in k.launches:
            base_ns = l.start_ns if base_ns is None else min(base_ns, l.start_ns)
    base_ns = base_ns or 0

    # 元数据事件
    streams = sorted({l.stream_id for k in rep.kernels for l in k.launches})
    for sid in streams:
        events.append({"ph": "M", "name": "thread_name", "pid": 0, "tid": sid,
                       "args": {"name": f"stream {sid}"}})
    events.append({"ph": "M", "name": "process_name", "pid": 0, "tid": 0,
                   "args": {"name": rep.device.name or "GPU"}})

    from ..report import config_key

    for k in rep.kernels:
        for l in k.launches:
            events.append({
                "ph": "X",
                "name": k.name,
                "cat": "kernel",
                "ts": (l.start_ns - base_ns) / 1000.0,      # Perfetto 用微秒
                "dur": l.duration_ns / 1000.0,
                "pid": 0,
                "tid": l.stream_id,
                "args": {
                    "grid": list(l.grid),
                    "block": list(l.block),
                    "config": config_key(l),
                    "registers_per_thread": l.registers_per_thread,
                    "static_smem_bytes": l.static_smem_bytes,
                    "dynamic_smem_bytes": l.dynamic_smem_bytes,
                    "cuda_graph": l.is_cuda_graph,
                    "provenance": "exact",
                    "source": "CUPTI activity (nsys sqlite)",
                },
            })

    doc = {
        "displayTimeUnit": "ns",
        "traceEvents": events,
        "metadata": {
            "generator": f"kernoscope {rep.tool_version}",
            "time_base_offset_ns": base_ns,
            "provenance": "exact (CUPTI activity timestamps)",
        },
    }
    return json.dumps(doc, indent=1)


def metric_by_name(rep: Report) -> dict[str, Metric]:
    return {m.name: m for k in rep.kernels for s in k.sections for m in s.metrics}
