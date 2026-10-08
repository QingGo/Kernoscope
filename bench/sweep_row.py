"""把一次「代理精度」运行压成 CSV 的一行（R-35）。

为什么需要它：误差带不能只给一个数，还要说明它是**在什么口径与什么边界下**测出来的。
需求方实测到覆盖率 88.2%（过了门槛）时误差 −9.8%，而 45 秒负载是 −0.6% —— 也就是说
「精度」本身随稳态段长度变化。所以要有「精度 vs 稳态段长度」的实测点，
而不是把一次运行的三个数当成上界。

用法（由 bench/run.sh 调用）：
    python3 bench/sweep_row.py <rep.ksreport.json> <target_s> <ledger_mb>
输出一行 CSV（无表头）。
"""

from __future__ import annotations

import json
import sys

COLUMNS = ("target_s", "coverage_pct", "mem_busy_pct", "window_share_pct",
           "kernel_share_pct", "bias_est_pct", "est_per_launch_mb",
           "ledger_per_launch_mb", "err_pct", "basis")

# 门槛（与 analyze/dram.py 保持一致，仅用于本表的可读性）
DUTY_WEAK = 80.0
SHARE_STRONG = 95.0


def _metrics(doc: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k in doc.get("kernels", []):
        for s in k.get("sections", []):
            for m in s.get("metrics", []):
                out.setdefault(m["name"], m)
    return out


def row(report_path: str, target_s: str, ledger_mb: float) -> list[str]:
    with open(report_path, encoding="utf-8") as f:
        doc = json.load(f)
    m = _metrics(doc)

    def val(name: str) -> object:
        x = m.get(name)
        if x is None or x.get("value") is None:
            return ""
        return x["value"]

    basis = ""
    err = m.get("est__dram_vs_ledger_err_pct")
    if err is not None and err.get("value") is not None:
        src = str(err.get("source", ""))
        basis = "trace" if "轨迹" in src else "nvml"
    return [
        str(target_s),
        str(val("ks__sampler_window_coverage_pct")),
        str(val("ks__dram_window_mem_busy_mean_pct")),
        str(val("ks__matched_kernel_window_share_pct")),
        str(val("ks__matched_kernel_kernel_share_pct")),
        str(val("ks__attribution_bias_estimate_pct")),
        str(val("est__dram_bytes_per_launch")) if val("est__dram_bytes_per_launch") != ""
        else str(val("est__dram_bytes_per_launch_trace")),
        f"{ledger_mb:.4f}",
        str(val("est__dram_vs_ledger_err_pct")),
        basis or ("none" if err is None else "unavailable"),
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    r = row(argv[1], argv[2], float(argv[3]))
    print(",".join(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
