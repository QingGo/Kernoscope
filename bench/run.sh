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

# 工具"发现"了还不够 —— 子进程是按 PATH 找的：
#   * tools/probes 的 Makefile 直接调 `nvcc`
#   * NVBit 官方 mem_trace 需要 `cuobjdump`
# 在 CUDA bin 不在 PATH 的机器上（本机就是：ncu/nvcc/cuobjdump 都在 CUDA bin 里），
# 这两处会分别以 `make: nvcc: No such file or directory` 和
# `ERROR: cuobjdump not found on PATH!!!` 失败，而主流程照样 rc=0 ——
# 于是产物里留下几个 rc=1 的步骤，却没有人知道是"发现到了但没用上"。
# 所以把发现的目录真的放进 PATH（doctor 的多路径发现解决"找得到"，这里解决"用得上"）。
if [[ -d "$CUDA_BIN" ]]; then
  export PATH="$CUDA_BIN:$PATH"
fi
if [[ -n "$NSYS" ]]; then
  NSYS_DIR="$(dirname "$NSYS")"
  [[ -d "$NSYS_DIR" ]] && export PATH="$NSYS_DIR:$PATH"
fi
# NVBit 是可选外部依赖，但"没设 NVBIT_PATH"不等于"这台机器没有 NVBit"：
# doctor 现在会按深度 1..3 找它。harness 也必须用同一份发现逻辑，否则步骤 8/9 会被
# 静默跳过（真事：一次产物里完全没有 NVBit 证据，而机器上装着 NVBit）。
if [[ -z "${NVBIT_PATH:-}" ]]; then
  found=""
  for base in . "$HOME" /root /opt /usr/local /workspace /mnt; do
    # 深度 1..3（与 doctor 的发现逻辑一致）：真实布局常见是
    # ~/<项目数据目录>/lab/nvbit_release_x86_64 —— 只找顶层会静默跳过步骤 8/9
    for d in "$base"/nvbit_release* "$base"/*/nvbit_release* "$base"/*/*/nvbit_release*; do
      if [[ -f "$d/core/libnvbit.a" ]]; then found="$(cd "$d" && pwd)"; break 2; fi
    done
  done
  [[ -n "$found" ]] && NVBIT_PATH="$found"
fi
if [[ -n "${NVBIT_PATH:-}" ]]; then
  export NVBIT_PATH
  echo "[bench] NVBit: $NVBIT_PATH"
else
  echo "[bench] warn: 没找到 NVBit（可选依赖）—— 步骤 8/9 会跳过"
fi
command -v nvcc >/dev/null 2>&1 || echo "[bench] warn: PATH 里仍然没有 nvcc —— probes 会编译失败"
command -v cuobjdump >/dev/null 2>&1 || echo "[bench] warn: PATH 里没有 cuobjdump —— NVBit mem_trace 会失败"

# 稳态时长：DRAM 代理要求"窗口被负载主导"，所以迭代数由**目标秒数**反推
# （ledger_probe --target-s 会先量一次每-launch 耗时再定迭代数）。
TARGET_BIG="${TARGET_BIG:-8}"      # 代理精度 / 账本对拍
TARGET_MID="${TARGET_MID:-1.2}"    # 覆盖率落在 50%~80% 之间（裁决行必须是 n/a 而不是消失）
MB=$(( QUICK ? 64 : 256 ))
ITERS_HINT=$(( QUICK ? 1000 : 8000 ))   # 只用于 --analyze-only 打印 total_bytes 的占位

declare -a STEPS=()
record() { STEPS+=("$1 rc=$2"); echo "[bench] $1 -> rc=$2"; }

echo "== bench: 环境快照 =="
# R-40：产物必须钉住「自己是哪个修订产生的」。仓库常常以不带 .git 的方式同步到租用
# 机器上，那种情况下 `git rev-parse` 会失败、产物只能写 git: n/a，于是没人能判断它由
# 哪一版代码产生。所以按下面的顺序找修订，并且**把来源一起记下来**：
#   1) KS_REVISION 环境变量（CI 显式传入）
#   2) 仓库里的 .git
#   3) 仓库根的 REVISION 文件（同步脚本写进去的）
#   4) 都找不到 → 明确写 n/a，来源记 none，绝不冒充「已钉住」
REVISION="${KS_REVISION:-}"
REVISION_SOURCE="env:KS_REVISION"
if [[ -z "$REVISION" ]] && git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  REVISION="$(git -C "$ROOT" rev-parse HEAD)"
  REVISION_SOURCE="git"
fi
if [[ -z "$REVISION" ]] && [[ -f "$ROOT/REVISION" ]]; then
  REVISION="$(head -1 "$ROOT/REVISION" | tr -d '[:space:]')"
  REVISION_SOURCE="file:REVISION"
fi
if [[ -z "$REVISION" ]]; then
  REVISION="n/a"
  REVISION_SOURCE="none"
fi
# 修订号只说明"基于哪个提交"，不说明"有没有未提交改动" —— 后者同样致命，
# 所以一起记下来（没有 .git 时无法判断，如实写 unknown）。
REVISION_DIRTY="unknown"
if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then
    REVISION_DIRTY="yes"
  else
    REVISION_DIRTY="no"
  fi
fi
KS_VERSION="$(KSC --version 2>/dev/null || echo 'n/a')"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "uname: $(uname -a)"
  echo "revision: $REVISION"
  echo "revision_source: $REVISION_SOURCE"
  echo "revision_dirty: $REVISION_DIRTY"
  echo "kernoscope: $KS_VERSION"
  echo "python: $($PY -V 2>&1)"
  echo "nvcc: $($NVCC --version 2>/dev/null | tail -1)"
  echo "nsys: $($NSYS --version 2>/dev/null | tail -1)"
  echo "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null)"
  echo "nvbit: ${NVBIT_PATH:-n/a}"
} > "$HOSTDIR/environment.txt" 2>&1
cat "$HOSTDIR/environment.txt"
if [[ "$REVISION_SOURCE" == "none" ]]; then
  echo "[bench] warn: 找不到修订号（没有 .git 也没有 REVISION 文件）—— 产物无法钉住版本。" \
       "同步仓库时请执行：git rev-parse HEAD > REVISION" | tee -a "$HOSTDIR/environment.txt"
fi

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
SM_COUNT="$("$PY" -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo 0)"
[[ -z "$SM_COUNT" || "$SM_COUNT" == "0" ]] && SM_COUNT=128

# ------------------------------------------------- 3) DRAM 代理精度（已知字节数）
echo "== 3) DRAM 代理精度：账本 = 解析真值 =="
if [[ -x "$PROBE" ]]; then
  for K in copy read write; do
    LOG="$HOSTDIR/probe-$K.log"
    # 真值不需要 GPU：--analyze-only 只打印解析字节数
    "$PROBE" --kernel "$K" --mb "$MB" --iters "$ITERS_HINT" --analyze-only > "$LOG" 2>&1
    record "probe-$K" $?
    ANALYTIC="$(grep -m1 '^ANALYTIC' "$LOG" || true)"
    MB_PER_LAUNCH="$(echo "$ANALYTIC" | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p')"
    if [[ -n "$MB_PER_LAUNCH" ]]; then
      LEDGER_MB="$(awk -v b="$MB_PER_LAUNCH" 'BEGIN{printf "%.4f", b/1e6}')"
      KSC --set basic --ks-calibration "$CAL" --ks-window auto \
          --ks-ledger-mb "$LEDGER_MB" -k "regex:${K}_k" -o "$HOSTDIR/rep-$K" -f \
          --packet "$HOSTDIR/rep-$K.packet.json" \
          -- "$PROBE" --kernel "$K" --mb "$MB" --target-s "$TARGET_BIG" \
          > "$HOSTDIR/rep-$K.txt" 2> "$HOSTDIR/rep-$K.err"
      record "report-$K" $?
      KSC -i "$HOSTDIR/rep-$K.ksreport.json" --csv > "$HOSTDIR/rep-$K.csv" 2>/dev/null
      record "csv-$K" $?
    fi
  done
fi

# ------------------- 3b) 代理精度 vs 稳态段长度（R-35）
# 误差带不能只给一个数：它得说明「在什么口径、什么边界下」测出来的。需求方实测到
# 覆盖率 88.2%（过了 80% 门槛）时误差 −9.8%，而 45 秒负载只有 −0.6% —— 精度本身
# 随稳态段长度变。所以这里把同一个已知答案负载在 4 个时长上各测一遍，落成一张表。
echo "== 3b) 代理精度 vs 稳态段长度（同一个已知答案负载，4 个时长）=="
if [[ -x "$PROBE" ]]; then
  "$PROBE" --kernel copy --mb "$MB" --iters "$ITERS_HINT" --analyze-only \
      > "$HOSTDIR/probe-sweep.log" 2>&1
  SWEEP_ANALYTIC="$(grep -m1 '^ANALYTIC' "$HOSTDIR/probe-sweep.log" || true)"
  SWEEP_BPL="$(echo "$SWEEP_ANALYTIC" | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p')"
  if [[ -n "$SWEEP_BPL" ]]; then
    SWEEP_LEDGER="$(awk -v b="$SWEEP_BPL" 'BEGIN{printf "%.4f", b/1e6}')"
    echo "target_s,coverage_pct,mem_busy_pct,window_share_pct,kernel_share_pct,bias_est_pct,est_per_launch_mb,ledger_per_launch_mb,err_pct,basis" \
        > "$HOSTDIR/proxy-duration-sweep.csv"
    for T in 1.2 3 8 20; do
      SOUT="$HOSTDIR/sweep-$T"
      KSC --set basic --ks-calibration "$CAL" --ks-window auto \
          --ks-ledger-mb "$SWEEP_LEDGER" -k regex:copy_k -o "$SOUT" -f \
          -- "$PROBE" --kernel copy --mb "$MB" --target-s "$T" \
          > "$SOUT.txt" 2> "$SOUT.err"
      record "sweep-$T" $?
      if [[ -f "$SOUT.ksreport.json" ]]; then
        "$PY" bench/sweep_row.py "$SOUT.ksreport.json" "$T" "$SWEEP_LEDGER" \
            >> "$HOSTDIR/proxy-duration-sweep.csv" 2>> "$SOUT.err"
      fi
    done
    echo "  时长扫描（误差 vs 稳态段长度）："
    column -s, -t "$HOSTDIR/proxy-duration-sweep.csv" 2>/dev/null || cat "$HOSTDIR/proxy-duration-sweep.csv"
  fi
fi

# --------------------------- 4) 口径不成立时裁决行不许消失（且 --strict 要能红）
echo "== 4) per-launch 账本 + 覆盖率不足 ⇒ 裁决行必须存在，且 --strict 变红 =="
if [[ -x "$PROBE" ]]; then
  "$PROBE" --kernel copy --mb "$MB" --iters "$ITERS_HINT" --analyze-only \
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

# ------------------------------------------------- 5) 形状扫描（多组 launch 配置）
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
  # 小 kernel × 长循环 = 十万级 launch：报告体积会爆（这正是"成本可预算"要解决的问题），
  # 所以这里用 -c 限流 + JSON 不带逐 launch 数组。
  KSC --set basic --ks-calibration "$CAL" --ks-window auto -k regex:norm_k -c 50 \
      --ks-json-no-launches -o "$HOSTDIR/rep-l2resident" -f \
      -- "$PROBE" --kernel norm --mb 2 --target-s "$TARGET_MID" \
      > "$HOSTDIR/rep-l2resident.txt" 2> "$HOSTDIR/rep-l2resident.err"
  record report-l2-resident $?
  # 同一次运行的体积对比：逐 launch 数组 vs --ks-json-no-launches（要的是可预算，不是劝告）
  KSC --set basic --ks-calibration "$CAL" --ks-window auto -k regex:norm_k \
      -o "$HOSTDIR/rep-l2resident-full" -f \
      -- "$PROBE" --kernel norm --mb 2 --target-s "$TARGET_MID" \
      > "$HOSTDIR/rep-l2resident-full.txt" 2> "$HOSTDIR/rep-l2resident-full.err"
  record report-l2-resident-full $?
  {
    echo "输出方式,报告字节数"
    echo "默认 JSON（含逐 launch 数组）,$(stat -c%s "$HOSTDIR/rep-l2resident-full.ksreport.json" 2>/dev/null || echo 0)"
    echo "--ks-json-no-launches -c 50,$(stat -c%s "$HOSTDIR/rep-l2resident.ksreport.json" 2>/dev/null || echo 0)"
  } > "$HOSTDIR/size-comparison.csv"
  rm -f "$HOSTDIR"/rep-l2resident-full.ksreport.json "$HOSTDIR"/rep-l2resident-full.perfetto.json
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
  # 探针同时写了 .smem32k（32 KB 静态共享内存把 occupancy 压到 25% 那个读数）。
  # 从前只分析前一份，于是 README 里第二行数字**不在产物里** —— 文档引用的数字
  # 必须能追到产物，所以这一份也要走同一条分析路径。
  if [[ -f tools/probes/occupancy-records.txt.smem32k ]]; then
    cp -f tools/probes/occupancy-records.txt.smem32k "$HOSTDIR/occupancy-records.smem32k.txt"
    KSC occupancy --from "$HOSTDIR/occupancy-records.smem32k.txt" --json \
        > "$HOSTDIR/occupancy-smem32k.json" 2>> "$HOSTDIR/occupancy-cmd.err"
    record occupancy-analyze-smem32k $?
  fi
fi

# ------------------------------------------------- 8) NVBit 访存轨迹（可选外部依赖）
if [[ -n "${NVBIT_PATH:-}" && -f tools/probes/nvbit_memtrace/collect.sh ]]; then
  echo "== 8) NVBit 官方 mem_trace → 轨迹口径账本对拍 =="
  # L2 容量：优先 torch（nvidia-smi 的 l2CacheSize 字段并非所有驱动都有）
  L2="$("$PY" -c 'import torch;print(int(torch.cuda.get_device_properties(0).L2_cache_size))' 2>/dev/null || echo 0)"
  if [[ -z "$L2" || "$L2" == "0" ]]; then L2=75497472; fi
  echo "L2 bytes = $L2"
  # 轨迹体积 ∝ 流量 × launch 次数（插桩不区分预热），所以用小负载 + 小预热：
  # 一个 80 MB 工作集 + 默认 50 次预热曾写出 10 GB 文本（真事）。小工作集（1 MB）常驻 L2，
  # 这恰好演示"换口径不是为了凑数，而是换一个前提成立的问法"：轨迹口径会告诉你
  # "这份 DRAM 账本对不上，因为流量根本没出 L2"。
  KS_MEMTRACE_OUT="$HOSTDIR/memtrace-official.txt" NVBIT_PATH="$NVBIT_PATH" \
      bash tools/probes/nvbit_memtrace/collect.sh -- \
      "$PROBE" --kernel copy --mb 1 --iters 1 --warmup 2 --grid 16 --block 128 \
      > "$HOSTDIR/nvbit-collect.log" 2>&1
  record nvbit-collect $?
  if [[ -s "$HOSTDIR/memtrace-official.txt" ]]; then
    head -40 "$HOSTDIR/memtrace-official.txt" > "$HOSTDIR/memtrace-sample.txt"
    KSC memtrace --from "$HOSTDIR/memtrace-official.txt" --l2-bytes "$L2" \
        --max-accesses 8000000 --json \
        > "$HOSTDIR/memtrace.json" 2> "$HOSTDIR/memtrace.err"
    record memtrace-analyze $?
    # 轨迹口径参与账本对拍：显存忙碌低到无法定标时，仍然要能给出误差
    "$PROBE" --kernel copy --mb 1 --iters 1 --analyze-only \
        > "$HOSTDIR/probe-tracebasis.log" 2>&1
    TB_MB="$(grep -m1 '^ANALYTIC' "$HOSTDIR/probe-tracebasis.log" \
             | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p' || true)"
    if [[ -n "$TB_MB" ]]; then
      TB_LEDGER="$(awk -v b="$TB_MB" 'BEGIN{printf "%.6f", b/1e6}')"
      KSC --set basic --ks-calibration "$CAL" --ks-window full \
          --ks-memtrace "$HOSTDIR/memtrace-official.txt" --ks-l2-bytes "$L2" \
          --ks-ledger-mb "$TB_LEDGER" -k regex:copy_k -o "$HOSTDIR/rep-tracebasis" -f \
          --ks-json-no-launches \
          -- "$PROBE" --kernel copy --mb 1 --iters 1 --warmup 2 --grid 16 --block 128 \
          > "$HOSTDIR/rep-tracebasis.txt" 2> "$HOSTDIR/rep-tracebasis.err"
      record report-trace-basis $?
    fi
  fi
fi

# ------------------------------------------------- 9) 发射槽上界（可选外部依赖）
if [[ -n "${NVBIT_PATH:-}" && -x "$PROBE" ]]; then
  echo "== 9) 发射槽利用率上界（NVBit opcode_hist）=="
  PROBE="$PROBE" PY="$PY" NVBIT_PATH="$NVBIT_PATH" OUTDIR="$HOSTDIR/issue" \
      bash bench/issue_probe.sh "$HOSTDIR/issue" 2>&1 | tail -8
  record issue-probe $?
  # 分母必须来自与直方图**同等工作量**的运行，所以把直方图喂给报告路径
  #（时长取自 CUPTI gpu__time_duration.sum），而不是手填一个秒数。
  for K in dram-copy l2-resident; do
    case "$K" in
      dram-copy)   ARGS=(--kernel copy --mb "$MB" --target-s "$TARGET_MID"); RX=copy_k ;;
      *)           ARGS=(--kernel norm --mb 2 --target-s "$TARGET_MID"); RX=norm_k ;;
    esac
    if [[ -s "$HOSTDIR/issue/hist-$K.txt" ]]; then
      KSC --set full --ks-calibration "$CAL" --ks-json-no-launches \
          --ks-instr-hist "$HOSTDIR/issue/hist-$K.txt" --ks-sm-count "$SM_COUNT" \
          -k regex:"$RX" -o "$HOSTDIR/rep-issue-$K" -f \
          -- "$PROBE" "${ARGS[@]}" \
          > "$HOSTDIR/rep-issue-$K.txt" 2> "$HOSTDIR/rep-issue-$K.err"
      record "report-issue-$K" $?
    fi
  done
  rm -f "$HOSTDIR"/issue/hist-*.txt     # 原始直方图按需重采（体积随 launch 数增长）
fi


# ------------------------------------------- 9.6) 能力分层 / 决策包 / 自检 / 对拍（W1–W7）
echo "== 9.6) 能力清单 + 一屏 + 本机自检 + 误差带对拍 =="
KSC capabilities --json > "$HOSTDIR/capabilities.json" 2> "$HOSTDIR/capabilities.err"
record capabilities-json $?
KSC capabilities > "$HOSTDIR/capabilities.txt" 2>> "$HOSTDIR/capabilities.err"
record capabilities-text $?
# 计数器实测探测：静态推断 vs 实测是两种证据，结论必须分开写
KSC capabilities --probe --json > "$HOSTDIR/capabilities-probe.json" 2>/dev/null
record capabilities-probe $?

# 一屏（人视图）：与 --packet 同一份判词，只是排版不同
if [[ -f "$HOSTDIR/rep-copy.ksreport.json" ]]; then
  KSC -i "$HOSTDIR/rep-copy.ksreport.json" --screen only \
      > "$HOSTDIR/rep-copy.onepage.txt" 2> "$HOSTDIR/rep-copy.onepage.err"
  record onepage $?
  KSC -i "$HOSTDIR/rep-copy.ksreport.json" --packet - \
      > "$HOSTDIR/rep-copy.packet-only.json" 2>/dev/null
  record packet-from-report $?
  "$PY" - "$HOSTDIR/rep-copy.packet.json" "$HOSTDIR/rep-copy.packet-only.json" \
      > "$HOSTDIR/packet-size.txt" 2>&1 <<'EOF'
import json, sys
buf = []
for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
        buf.append(f"{path}: {d['cost'].get('packet_bytes')} B, "
                   f"{len(d.get('verdicts', []))} verdicts, "
                   f"trimmed={d['cost'].get('trimmed')}")
    except Exception as e:  # noqa: BLE001
        buf.append(f"{path}: 读不了（{e}）")
print("\n".join(buf))
EOF
  cat "$HOSTDIR/packet-size.txt"
fi

# 本机自检：已知答案 → 误差带 + MDD（纯 torch + NVML，不需要 nsys/nvcc/NVBit）
KSC selftest --out "$HOSTDIR/machine-selfcheck.json" --duration 2 --runs 3 \
    > "$HOSTDIR/selftest.txt" 2> "$HOSTDIR/selftest.err"
record selftest $?

# 误差带对拍：同协议两次运行必须报"分不出来"；换窗口模式（协议不同）必须报"不可比"
if [[ -f "$HOSTDIR/rep-copy.ksreport.json" && -x "$PROBE" ]]; then
  KSC --set basic --ks-calibration "$CAL" --ks-window auto \
      --ks-ledger-mb "$(grep -m1 '^ANALYTIC' "$HOSTDIR/probe-copy.log" 2>/dev/null | sed -n 's/.*bytes_per_launch=\([0-9.]*\).*/\1/p' | awk '{printf "%.4f", $1/1e6}')" \
      -k regex:copy_k -o "$HOSTDIR/rep-copy-again" -f \
      -- "$PROBE" --kernel copy --mb "$MB" --target-s "$TARGET_BIG" \
      > "$HOSTDIR/rep-copy-again.txt" 2> "$HOSTDIR/rep-copy-again.err"
  record report-copy-again $?
  if [[ -f "$HOSTDIR/rep-copy-again.ksreport.json" ]]; then
    KSC diff "$HOSTDIR/rep-copy.ksreport.json" "$HOSTDIR/rep-copy-again.ksreport.json" \
        --selfcheck "$HOSTDIR/machine-selfcheck.json" --json \
        > "$HOSTDIR/diff-same-protocol.json" 2> "$HOSTDIR/diff-same-protocol.err"
    record diff-same-protocol $?
    KSC diff "$HOSTDIR/rep-copy.ksreport.json" "$HOSTDIR/rep-copy-again.ksreport.json" \
        --selfcheck "$HOSTDIR/machine-selfcheck.json" \
        > "$HOSTDIR/diff-same-protocol.txt" 2>> "$HOSTDIR/diff-same-protocol.err"
    # 协议不同的负样本：把窗口模式换掉（口径不同 ⇒ 只能给差值，不给方向）
    KSC --set basic --ks-calibration "$CAL" --ks-window full \
        -k regex:copy_k -o "$HOSTDIR/rep-copy-fullwin" -f \
        -- "$PROBE" --kernel copy --mb "$MB" --target-s "$TARGET_BIG" \
        > "$HOSTDIR/rep-copy-fullwin.txt" 2> "$HOSTDIR/rep-copy-fullwin.err"
    KSC diff "$HOSTDIR/rep-copy.ksreport.json" "$HOSTDIR/rep-copy-fullwin.ksreport.json" \
        --json > "$HOSTDIR/diff-protocol-mismatch.json" 2>&1
    record diff-protocol-mismatch $?
    KSC diff "$HOSTDIR/rep-copy.ksreport.json" "$HOSTDIR/rep-copy-fullwin.ksreport.json" \
        > "$HOSTDIR/diff-protocol-mismatch.txt" 2>&1
  fi
fi

# ------------------------------------------------- 9.5) 脱敏（提交前就该是干净的）
# 路径规则按**本仓库的位置**生成，所以脚本里不出现任何平台路径。
"$PY" bench/sanitize_artifacts.py "$HOSTDIR" --repo-path "$ROOT" || true
record sanitize $?

# ------------------------------------------------- 10) manifest
{
  echo "{"
  echo "  \"stamp\": \"$STAMP\","
  echo "  \"revision\": \"$REVISION\","
  echo "  \"revision_source\": \"$REVISION_SOURCE\","
  echo "  \"revision_dirty\": \"$REVISION_DIRTY\","
  echo "  \"kernoscope\": \"$KS_VERSION\","
  echo "  \"quick\": $QUICK,"
  echo "  \"target_big_s\": \"$TARGET_BIG\", \"target_mid_s\": \"$TARGET_MID\", \"mb\": $MB,"
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
