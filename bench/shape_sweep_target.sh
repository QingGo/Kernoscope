#!/usr/bin/env bash
# 形状扫描的被测负载：同一个 kernel、三组 grid（R-04/B4 的复现负载）。
# 用法：bash bench/shape_sweep_target.sh <ledger_probe 路径>
set -euo pipefail
PROBE="$1"
for G in 64 256 1024; do
  "$PROBE" --kernel copy --mb "${MB:-16}" --iters "${ITERS:-300}" --grid "$G"
done
