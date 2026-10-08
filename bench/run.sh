#!/usr/bin/env bash
# bench/run.sh —— 复现仓库里引用的每一个"实测"数字（R-25：实测必须有可复现产物）。
#
# 设计原则：
#   * 产物是**机器可读**的（JSON / 文本原样落盘），不是往 README 里手抄数字；
#   * 每个实验都写清"真值从哪来"（账本由 ledger_probe 的 ANALYTIC 行给出）；
#   * 失败不掩盖：每步的 rc 都记进 manifest.json。
#
# 用法（在 **有 GPU 的机器**上，仓库根目录）：
#   bash bench/run.sh                     # 全量（约 3–6 分钟 GPU 时间）
#   bash bench/run.sh --quick             # 快速版（更短的稳态循环 / 跳过标定）
#   OUTDIR=bench/artifacts bash bench/run.sh
#
# 需要的环境：nvcc（随 CUDA Toolkit）、nsys、nvidia-smi；标定那一步还需要 torch。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTDIR="${OUTDIR:-bench/artifacts}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HOSTDIR="$OUTDIR/$STAMP"
BUILD="$OUTDIR/.build"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

mkdir -p "$HOSTDIR" "$BUILD"

PY="${PY:-python3}"
KSC() { PYTHONPATH="$ROOT/src" "$PY" -m kernoscope "$@"; }
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda/bin}"
NVCC="$(command -v nvcc || echo "$CUDA_BIN/nvcc")"
NSYS="$(command -v nsys || true)"

# 稳态时长：DRAM 代理要求"窗口被负载主导"，所以迭代数由**目标秒数**反推
# （ledger_probe --target-s 会先量一次每-launch 耗时再定迭代数）。
TARGET_BIG="${TARGET_BIG:-8}"      # 代理精度 / 账本对拍
TARGET_MID="${TARGET_MID:-1.2}"    # 覆盖率落在 50%~80% 之间（裁决行必须是 n/a 而不是消失）
MB=$(( QUICK ? 64 : 256 ))

declare -a STEPS=()
record() { STEPS+=("$1 rc=$2"); echo "[bench] $1 -> rc=$2"; }

echo "== bench: 环境快照 =="
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "uname: $(uname -a)"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
  echo "python: $($PY -V 2>&1)"
  echo "nvcc: $($NVCC --version 2>/dev/null | tail -1)"
  echo "nsys: $($NSYS --version 2>/dev/null | tail -1)"
  echo "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null)"
} > "$HOSTDIR/environment.txt" 2>&1
cat "$HOSTDIR/environment.txt"

# ---------------------------------------------------------------- 0) doctor
echo "== 0) doctor（工具多路径发现 + 计数器四态）=="
KSC doctor --json > "$HOSTDIR/doctor.json" 2> "$HOSTDIR/doctor.err"
record doctor $?
KSC doctor > "$HOSTDIR/doctor.txt" 2>&1
# 只在 PATH 上找 vs 多路径：证明"未安装"不是一个可被一次 ls 推翻的结论
KSC doctor --json > "$HOSTDIR/doctor.json" 2>/dev/null
PATH="/usr/bin:/bin" KSC doctor --json > "$HOSTDIR/doctor-path-stripped.json" 2>/dev/null
record doctor-path-stripped $?

# ------------------------------------------------- 1) 本机峰值标定（含口径）
if [[ "${SKIP_CALIBRATION:-0}" != "1" ]]; then
  echo "== 1) machine-profile（累加口径 + 口径离散度）=="
  CAL_ARGS=(--out "$HOSTDIR/machine-profile.json")
  [[ $QUICK == 1 ]] && CAL_ARGS+=(--quick)
  "$PY" tools/machine-profile.py "${CAL_ARGS[@]}" > "$HOSTDIR/machine-profile.log" 2>&1
  record machine-profile $?
else
  cp "${CALIBRATION:-machine-profile.json}" "$HOSTDIR/machine-profile.json" 2>/dev/null
fi
CAL="$HOSTDIR/machine-profile.json"

# ------------------------------------------------- 2) 编译已知答案微基准
echo "== 2) 编译 bench/ledger_probe.cu =="
"$NVCC" -O3 -o "$BUILD/ledger_probe" bench/ledger_probe.cu > "$HOSTDIR/nvcc.log" 2>&1
record nvcc $?
PROBE="$BUILD/ledger_probe"

# ------------------------------------------------- 3) DRAM 代理精度（已知字节数）
echo "== 3) DRAM 代理精度：账本 = 解析真值 =="
if [[ -x "$PROBE" ]]; then
  for K in copy read write; do
    LOG="$HOSTDIR/probe-$K.log"
    # 真值不需要 GPU：--analyze-only 只打印解析字节数
    "$PROBE" --kernel "$K" --mb "$MB" --iters "$ITERS_BIG" --analyze-only > "$LOG" 2>&1
    record "probe-$K" $?
    ANALYTIC="$(grep -m1 '^ANALYTIC' "$LOG" || true)"
    MB_PER_LAUNCH="$(echo "$ANALYTIC" | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p')"
    if [[ -n "$MB_PER_LAUNCH" ]]; then
      LEDGER_MB="$(awk -v b="$MB_PER_LAUNCH" 'BEGIN{printf "%.4f", b/1e6}')"
      KSC --set basic --ks-calibration "$CAL" --ks-window auto \
          --ks-ledger-mb "$LEDGER_MB" -k "regex:${K}_k" -o "$HOSTDIR/rep-$K" -f \
          -- "$PROBE" --kernel "$K" --mb "$MB" --target-s "$TARGET_BIG" \
          > "$HOSTDIR/rep-$K.txt" 2> "$HOSTDIR/rep-$K.err"
      record "report-$K" $?
      KSC -i "$HOSTDIR/rep-$K.ksreport.json" --csv > "$HOSTDIR/rep-$K.csv" 2>/dev/null
      record "csv-$K" $?
    fi
  done
fi

# ------------------------------------------------- 4) B1：口径不成立时不许消失
echo "== 4) B1/R-02：per-launch 账本 + 覆盖率不足 ⇒ 行必须存在且 --strict 变红 =="
if [[ -x "$PROBE" ]]; then
  "$PROBE" --kernel copy --mb "$MB" --iters 400 --analyze-only \
      > "$HOSTDIR/probe-lowduty.log" 2>&1
  KSC --set basic --ks-calibration "$CAL" --ks-window full \
      --ks-ledger-mb "$(awk -v m="$MB" 'BEGIN{printf "%.4f", 2*m*1048576/1e6}')" \
      -k regex:copy_k --strict -o "$HOSTDIR/rep-lowduty" -f \
      -- "$PROBE" --kernel copy --mb "$MB" --target-s "$TARGET_MID" \
      > "$HOSTDIR/rep-lowduty.txt" 2> "$HOSTDIR/rep-lowduty.err"
  record strict-lowduty $?          # 期望 2：请求了账本裁决但口径不成立
  grep -c 'est__dram_vs_ledger_err_pct' "$HOSTDIR/rep-lowduty.txt" \
      > "$HOSTDIR/rep-lowduty.ledger-line-count" || true
fi

# ------------------------------------------------- 5) 形状扫描（R-04/B4）
echo "== 5) 形状扫描：同名 kernel、三组 launch 配置 =="
if [[ -x "$PROBE" ]]; then
  KSC --set basic --ks-no-nvml -k regex:copy_k --launch-csv \
      -- bash bench/shape_sweep_target.sh "$PROBE" \
      > "$HOSTDIR/shape-sweep.launches.csv" 2> "$HOSTDIR/shape-sweep.err"
  record shape-sweep-launch-csv $?
  KSC --set basic --ks-no-nvml -k regex:copy_k -o "$HOSTDIR/rep-shape" -f \
      -- bash bench/shape_sweep_target.sh "$PROBE" \
      > "$HOSTDIR/rep-shape.txt" 2> "$HOSTDIR/rep-shape.err"
  record shape-sweep-report $?
  grep -c '^\"' "$HOSTDIR/shape-sweep.launches.csv" \
      > "$HOSTDIR/shape-sweep.launch-rows" 2>/dev/null || true
fi

# ---------------------------------- 6) compute-bound 与 L2 驻留：两个前提分开看
echo "== 6) compute-bound / L2 驻留：SM 忙 ≠ 显存忙 =="
if [[ -x "$PROBE" ]]; then
  KSC --set basic --ks-calibration "$CAL" --ks-window auto -k regex:norm_k \
      -o "$HOSTDIR/rep-compute" -f \
      -- "$PROBE" --kernel norm --mb "$MB" --target-s "$TARGET_MID" \
      > "$HOSTDIR/rep-compute.txt" 2> "$HOSTDIR/rep-compute.err"
  record report-compute-bound $?
  # 小工作集（2 MB）常驻 L2：显存几乎不动 ⇒ 代理必须拒绝给数，并点明分辨率下限
  KSC --set basic --ks-calibration "$CAL" --ks-window auto -k regex:norm_k \
      -o "$HOSTDIR/rep-l2resident" -f \
      -- "$PROBE" --kernel norm --mb 2 --target-s "$TARGET_MID" \
      > "$HOSTDIR/rep-l2resident.txt" 2> "$HOSTDIR/rep-l2resident.err"
  record report-l2-resident $?
fi

# ------------------------------------------------- 7) 达到的 occupancy（measured）
echo "== 7) occupancy 探针 =="
if [[ -f tools/probes/Makefile ]]; then
  make -C tools/probes run > "$HOSTDIR/occupancy.log" 2>&1
  record occupancy-probe $?
  cp -f tools/probes/occupancy-records.txt "$HOSTDIR/" 2>/dev/null || true
  KSC occupancy --from "$HOSTDIR/occupancy-records.txt" --json \
      > "$HOSTDIR/occupancy.json" 2> "$HOSTDIR/occupancy-cmd.err"
  record occupancy-analyze $?
fi

# ------------------------------------------------- 8) NVBit 访存轨迹（可选外部依赖）
if [[ -n "${NVBIT_PATH:-}" && -f tools/probes/nvbit_memtrace/collect.sh ]]; then
  echo "== 8) NVBit 官方 mem_trace → 轨迹口径账本对拍 =="
  # L2 容量：优先 torch（nvidia-smi 的 l2CacheSize 字段并非所有驱动都有）
  L2="$("$PY" -c 'import torch;print(int(torch.cuda.get_device_properties(0).L2_cache_size))' 2>/dev/null || echo 0)"
  if [[ -z "$L2" || "$L2" == "0" ]]; then L2=75497472; fi
  echo "L2 bytes = $L2"
  # 用一个访存密集的小负载产轨迹（形状简单，便于人工核对已知答案）
  KS_MEMTRACE_OUT="$HOSTDIR/memtrace-official.txt" NVBIT_PATH="$NVBIT_PATH" \
      bash tools/probes/nvbit_memtrace/collect.sh -- \
      "$PROBE" --kernel copy --mb 8 --iters 60 \
      > "$HOSTDIR/nvbit-collect.log" 2>&1
  record nvbit-collect $?
  if [[ -s "$HOSTDIR/memtrace-official.txt" ]]; then
    KSC memtrace --from "$HOSTDIR/memtrace-official.txt" --l2-bytes "$L2" --json \
        > "$HOSTDIR/memtrace.json" 2> "$HOSTDIR/memtrace.err"
    record memtrace-analyze $?
    # 轨迹口径参与账本对拍：显存忙碌低到无法定标时，仍然要能给出误差
    "$PROBE" --kernel copy --mb 8 --iters 60 --analyze-only \
        > "$HOSTDIR/probe-tracebasis.log" 2>&1
    TB_MB="$(grep -m1 '^ANALYTIC' "$HOSTDIR/probe-tracebasis.log" \
             | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p' || true)"
    if [[ -n "$TB_MB" ]]; then
      TB_LEDGER="$(awk -v b="$TB_MB" 'BEGIN{printf "%.6f", b/1e6}')"
      KSC --set basic --ks-calibration "$CAL" --ks-window full \
          --ks-memtrace "$HOSTDIR/memtrace-official.txt" --ks-l2-bytes "$L2" \
          --ks-ledger-mb "$TB_LEDGER" -k regex:copy_k -o "$HOSTDIR/rep-tracebasis" -f \
          -- "$PROBE" --kernel copy --mb 8 --iters 60 \
          > "$HOSTDIR/rep-tracebasis.txt" 2> "$HOSTDIR/rep-tracebasis.err"
      record report-trace-basis $?
    fi
  fi
fi

# ------------------------------------------------- 9) manifest
{
  echo "{"
  echo "  \"stamp\": \"$STAMP\","
  echo "  \"quick\": $QUICK,"
  echo "  \"iters_big\": $ITERS_BIG, \"iters_small\": $ITERS_SMALL, \"mb\": $MB,"
  echo "  \"steps\": ["
  first=1
  for s in "${STEPS[@]}"; do
    name="${s%% rc=*}"; rc="${s##*rc=}"
    [[ $first == 0 ]] && echo ","
    first=0
    printf '    {"step": "%s", "rc": %s}' "$name" "$rc"
  done
  echo ""
  echo "  ]"
  echo "}"
} > "$HOSTDIR/manifest.json"

echo
echo "== bench 完成：产物在 $HOSTDIR =="
ls -1 "$HOSTDIR"
if grep -q '"rc": [^0]' "$HOSTDIR/manifest.json"; then
  echo "注意：有步骤非零退出，逐条见 manifest.json 与同名 .err/.log"
fi
