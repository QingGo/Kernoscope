#!/usr/bin/env bash
# bench/issue_probe.sh —— 采集动态指令数并算发射槽上界（R-08 的实测产物）。
#
#   NVBIT_PATH=~/nvbit_release_x86_64 PROBE=bench/artifacts/.build/ledger_probe \
#       bash bench/issue_probe.sh <输出目录>
#
# 需要 NVBit 官方 opcode_hist（自行安装；本脚本不打包它）。
# 关键：NO_EAGER_LOAD=0 —— NVBit 1.8 默认懒加载，会让插桩错过时机，
# 表现为"程序正常跑完但一条记录都没有"。
set -uo pipefail

OUT="${OUTDIR:-${1:-bench/artifacts/issue}}"
mkdir -p "$OUT"
PROBE="${PROBE:-bench/artifacts/.build/ledger_probe}"
PY="${PY:-python3}"
KSC() { PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src" "$PY" -m kernoscope "$@"; }

if [[ -z "${NVBIT_PATH:-}" ]]; then
  echo "请设置 NVBIT_PATH 指向 NVBit 安装目录" >&2
  exit 2
fi
HIST_SO="$NVBIT_PATH/tools/build/opcode_hist/opcode_hist.so"
if [[ ! -f "$HIST_SO" ]]; then
  echo "！找不到 ${HIST_SO}；先在 NVBit 目录里构建：make -C tools/opcode_hist" >&2
  exit 2
fi
if [[ ! -x "$PROBE" ]]; then
  NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
  "$NVCC" -O3 -o "$PROBE" bench/ledger_probe.cu || exit 1
fi

SM="$("$PY" -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo 0)"
[[ -z "$SM" || "$SM" == "0" ]] && SM=128

collect() {   # collect <名字> <probe 参数...>
  local name="$1"; shift
  echo "== opcode_hist: $name =="
  NO_EAGER_LOAD=0 CUDA_INJECTION64_PATH="$HIST_SO" LD_PRELOAD="$HIST_SO" NOBANNER=1 \
      "$PROBE" "$@" > "$OUT/hist-$name.txt" 2>&1
  local rc=$?
  echo "  rc=$rc  记录行: $(grep -c '^kernel ' "$OUT/hist-$name.txt" || true)"
  return $rc
}

# 1) 显存受限：大工作集拷贝（DRAM bound）
collect dram-copy --kernel copy --mb 256 --target-s 3
# 2) 小工作集（常驻 L2）：指令在等数据，但等的不是 DRAM
collect l2-resident --kernel norm --mb 2 --target-s 3

for K in dram-copy l2-resident; do
  echo "== kernoscope issue: $K =="
  KSC issue --from "$OUT/hist-$K.txt" --sm-count "$SM" --clock-mhz "${CLOCK_MHZ:-2450}" \
      --duration-ms "${DURATION_MS:-3000}" --json > "$OUT/issue-$K.json" 2> "$OUT/issue-$K.err"
  echo "  rc=$? -> $OUT/issue-$K.json"
done

echo "== 产物 =="
ls -1 "$OUT"
