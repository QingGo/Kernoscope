"""渲染器：ncu 风格文本 / CSV（含 provenance 列）/ Perfetto trace。"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..provenance import Provenance
from ..report import Report

WIDTH = 46


def render_text(rep: Report, *, quiet: bool = False, summary: str = "none",
                page: str = "details") -> str:
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
        if rep.meta.get("kernel_time_pct") is not None:
            out.append(f"==PROF== Kernel time: {rep.total_kernel_ns / 1e6:.3f} ms of "
                       f"{rep.wall_time_ns / 1e6:.3f} ms wall ({rep.meta['kernel_time_pct']:.1f}%)")
        for k, v in (rep.sources or {}).items():
            out.append(f"==PROF== source[{k}]: {v}")
        out.append("")

    if summary == "per-kernel":
        out.append(f"{'kernel':<52} {'launches':>8} {'total_ms':>10} {'mean_ms':>10}")
        for k in rep.kernels:
            name = k.name if len(k.name) <= 51 else k.name[:48] + "..."
            out.append(f"{name:<52} {k.launch_count:>8} {k.total_ns / 1e6:>10.4f} "
                       f"{k.mean_ns / 1e6:>10.4f}")
        out.append("")
        return "\n".join(out)

    for k in rep.kernels:
        title = k.name if len(k.name) <= 70 else k.name[:67] + "..."
        out.append(f"Kernel: {title}")
        if getattr(k, "alias", ""):
            out.append(f"  nsys short name: {k.alias}")
        out.append(f"  {k.launch_count} launch(es), {k.total_ns / 1e6:.4f} ms total, "
                   f"{k.mean_ns / 1e6:.4f} ms mean")
        for s in k.sections:
            tag = s.provenance.tag
            extra = f" (+{s.n_unavailable} n/a)" if s.n_unavailable else ""
            out.append("")
            out.append(f"  Section: {s.name:<34} {tag}{extra}")
            if s.note:
                out.append(f"    note: {s.note}")
            for m in s.metrics:
                if m.provenance is Provenance.UNAVAILABLE:
                    out.append(f"    {m.name:<{WIDTH}} {'n/a':>14} {'':<8} "
                               f"{m.provenance.tag}  {m.reason[:100]}")
                else:
                    unit = m.unit or ""
                    line = f"    {m.name:<{WIDTH}} {m.render_value():>14} {unit:<8} {m.provenance.tag}"
                    if m.note:
                        line += f"  ({m.note})"
                    out.append(line)
                    if page == "raw" and m.source:
                        out.append(f"        source: {m.source}")
        out.append("")
        out.append("-" * 100)

    hist = rep.provenance_histogram()
    out.append("")
    out.append("Provenance summary: " + ", ".join(f"{k}={v}" for k, v in hist.items() if v))
    out.append("Legend: exact=与 ncu 同源 | derived=本机实测标定/模型 | measured=我们的探针实测 | "
               "estimated=采样或代理(附误差) | unavailable=需要硬件计数器，只报 n/a")
    for key in ("json", "csv", "perfetto"):
        if rep.meta.get(f"out_{key}"):
            out.append(f"Output({key}): {rep.meta[f'out_{key}']}")
    if rep.meta.get("out_perfetto"):
        out.append("Open the timeline at https://ui.perfetto.dev -> Load trace -> the file above")
    return "\n".join(out)


CSV_COLUMNS = ["kernel", "section", "section_provenance", "metric", "value", "unit",
               "provenance", "source", "reason", "note"]


def render_csv(rep: Report) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for k in rep.kernels:
        for s in k.sections:
            for m in s.metrics:
                w.writerow({
                    "kernel": k.name,
                    "section": s.name,
                    "section_provenance": s.provenance.value,
                    "metric": m.name,
                    "value": "" if m.value is None else m.value,
                    "unit": m.unit,
                    "provenance": m.provenance.value,
                    "source": m.source,
                    "reason": m.reason,
                    "note": m.note,
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
