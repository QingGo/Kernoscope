#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 一键采集「达到的 occupancy」（NVBit 版：被测 kernel **零源码改动**）。
#
#   NVBIT_OCCUPANCY_SO=/path/to/ks_occupancy.so KS_OCC_OUT=ks-occupancy.txt \
#       ./collect.sh -- ./your_app
#
#   ./collect.sh --resolve-only          # 只回答"这次会用哪份 .so、它是哪个版本"
#
# 版本纪律与 ../nvbit_memtrace/collect.sh 完全同一套（R-72/R-77/R-102）：
#   * 版本**从实物读**（复用 ../nvbit_memtrace/nvbit_version.sh：先看 core/nvbit.h，
#     退化时看 .so 里的整行版本串）；读不出就写 n/a，**绝不编一个版本**。
#   * `--resolve-only` 把"这次会用哪份 .so、它是哪个版本"以机器可读的两行给出来 ——
#     "用了哪份产物"只有这一处真源，别的脚本不许再抄一份候选路径。
#   * 显式指了 NVBIT_PATH / NVBIT_OCCUPANCY_SO 却找不到产物 ⇒ exit 2，并列出
#     **实际找过的路径**（而不是让人再指一遍同一个变量）。
#   * 版本号只是**指认构建**的线索，能不能用是**构建级**的结论：同一版本号的不同构建
#     可以一个能用一个不能用。本仓库真机唯一钉住可用的组合（官方 mem_trace 的结论）
#     见 ../nvbit_memtrace/README.md 的版本矩阵；**本工具在任何版本上都没在真机验证过**
#     —— 见同目录 README.md 的「未在真机验证」一节。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_SH="$HERE/../nvbit_memtrace/nvbit_version.sh"

# 本工具在 $NVBIT_PATH 那棵树下的两种产物布局（顺序：CMake 的 build 目录 → 工具目录里那份）。
# 两份都必须真的出现在这个列表里：删掉任一种，显式指了那棵树的人就会看到"静默找不到"。
CANDIDATES=()
if [ -n "${NVBIT_PATH:-}" ]; then
  CANDIDATES=("$NVBIT_PATH/tools/ks_occupancy/build/ks_occupancy.so" \
              "$NVBIT_PATH/tools/ks_occupancy/ks_occupancy.so")
fi

if [ -z "${NVBIT_OCCUPANCY_SO:-}" ]; then
  # ① 本脚本旁边：`make` 会把构建好的 .so 拷到这里，是最常见的位置
  if [ -f "$HERE/ks_occupancy.so" ]; then
    NVBIT_OCCUPANCY_SO="$HERE/ks_occupancy.so"
  fi
  # ② $NVBIT_PATH 那棵树里的两种布局
  if [ -z "${NVBIT_OCCUPANCY_SO:-}" ] && [ "${#CANDIDATES[@]}" -gt 0 ]; then
    for cand in "${CANDIDATES[@]}"; do
      if [ -f "$cand" ]; then NVBIT_OCCUPANCY_SO="$cand"; break; fi
    done
  fi
fi

if [ -z "${NVBIT_OCCUPANCY_SO:-}" ]; then
  if [ "${#CANDIDATES[@]}" -gt 0 ] || [ -n "${NVBIT_PATH:-}" ]; then
    {
      echo "！这棵树里没有 ks_occupancy.so 的构建产物（本工具要自己构建）。"
      echo "  实际找过的路径："
      echo "    - $HERE/ks_occupancy.so"
      for cand in "${CANDIDATES[@]}"; do echo "    - $cand"; done
      echo "  构建（用你自己的 NVBit，本项目不分发它）："
      echo "    make -C $HERE NVBIT_PATH=\$NVBIT_PATH ARCH=sm_89"
      echo "  手上已经有构建好的 .so 时，直接 NVBIT_OCCUPANCY_SO=/path/to/ks_occupancy.so 指过去。"
    } >&2
  else
    {
      echo "！没有可用的 ks_occupancy.so：请设置 NVBIT_OCCUPANCY_SO=/path/to/ks_occupancy.so，" \
           "或用 NVBIT_PATH 指向 NVBit 安装树（本工具需自行构建）。"
      echo "  构建：make -C $HERE NVBIT_PATH=<NVBit 解压目录> ARCH=sm_89"
    } >&2
  fi
  exit 2
fi
if [ ! -f "$NVBIT_OCCUPANCY_SO" ]; then
  echo "！NVBIT_OCCUPANCY_SO 指向的文件不存在: $NVBIT_OCCUPANCY_SO" >&2
  echo "  构建方式与两种布局见 $HERE/README.md" >&2
  exit 2
fi

# 用了哪份产物、它是哪个版本 —— 版本**从实物读**（不手抄），采集记录里能对得上。
NV_V=""
if [ -f "$VERSION_SH" ]; then
  NV_V="$(bash "$VERSION_SH" "$NVBIT_OCCUPANCY_SO" 2>/dev/null || true)"
fi
if [ -n "${NV_V:-}" ]; then
  echo "工具: ${NVBIT_OCCUPANCY_SO}（NVBit ${NV_V}，从实物读）"
else
  echo "工具: ${NVBIT_OCCUPANCY_SO}（NVBit 版本读不出：core/nvbit.h 与 .so 字符串里都没有）"
fi

# `--resolve-only`：只回答"这次会用哪份 .so、它是哪个版本"，不采集。
# 输出固定两行、机器可读；版本读不出就写 n/a（**绝不编一个版本**）。
if [ "${1:-}" = "--resolve-only" ] || [ "${KS_OCC_RESOLVE_ONLY:-0}" = "1" ]; then
  printf 'occupancy_so: %s\n' "$NVBIT_OCCUPANCY_SO"
  printf 'occupancy_so_version: %s\n' "${NV_V:-n/a}"
  exit 0
fi

# NVBit 的工具链要 cuobjdump 在 PATH 上（租用机上 CUDA 常常装了但不在 PATH）。
# 按 doctor 的同一套发现逻辑补 PATH；补不到就说清是缺它，而不是让人去查别处。
if ! command -v cuobjdump >/dev/null 2>&1; then
  for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" "${CUDA_BIN:-}" /usr/local/cuda /usr/local/cuda-*; do
    [ -n "$d" ] && [ -x "$d/bin/cuobjdump" ] && { PATH="$d/bin:$PATH"; break; }
  done
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "！找不到 cuobjdump：NVBit 需要它来 dump kernel 镜像。" >&2
  echo "  把它所在的 bin 目录加进 PATH（例如 CUDA_BIN=/usr/local/cuda-13.0）。" >&2
  exit 2
fi

OUT="${KS_OCC_OUT:-ks-occupancy.txt}"
LOG="${KS_OCC_LOG:-${OUT}.run.log}"

if [ "${1:-}" = "--" ]; then shift; fi
if [ "$#" -eq 0 ]; then
  echo "用法: NVBIT_OCCUPANCY_SO=... [KS_OCC_OUT=ks-occupancy.txt] [KS_OCC_KERNEL=<子串>] " \
       "$0 -- <命令> [参数...]" >&2
  exit 2
fi

# 先删掉旧文件：工具的纪律是"覆盖率不够就**不写**文件"，留着旧文件会让人把上一次的
# 结果当成本次的（而覆盖率/launch 数的判断在文件里，不在文件名里）。
rm -f "$OUT"

echo "采集 → ${OUT}（NO_EAGER_LOAD=0；被测程序的输出写到 ${LOG}）"
set +e
NO_EAGER_LOAD=0 \
CUDA_INJECTION64_PATH="$NVBIT_OCCUPANCY_SO" \
LD_PRELOAD="$NVBIT_OCCUPANCY_SO" \
NOBANNER=1 \
KS_OCC_OUT="$OUT" \
"$@" > "$LOG" 2>&1
RC=$?
set -e
echo "被测程序退出码: $RC"

if [ "$RC" -ne 0 ]; then
  echo "！被测程序没跑成功（退出码 ${RC}）—— 先看 ${LOG} 的末尾几行：" >&2
  tail -5 "$LOG" >&2 || true
  exit "$RC"
fi

# 覆盖率/多 launch 纪律没过时，工具**不写文件**（除非 KS_OCC_ALLOW_PARTIAL=1）。
if [ ! -f "$OUT" ]; then
  echo "！工具没有写出 ${OUT}（多半是覆盖率不足或同一个 kernel 被 launch 了多次）。" >&2
  grep -E '^KS-OCCUPANCY' "$LOG" >&2 || true
  echo "  详见 $HERE/README.md 的「覆盖率纪律」：部分数据算出来的分布只会偏小，" >&2
  echo "  所以这里**故意**不产出文件；不要用别的路径从部分数据里凑一个数。" >&2
  exit 3
fi
if grep -q 'KS-OCCUPANCY: 状态 拒绝' "$LOG"; then
  echo "！工具自己标了 拒绝（KS_OCC_ALLOW_PARTIAL=1 强行写出了文件）：" >&2
  grep -E '^KS-OCCUPANCY' "$LOG" >&2 || true
  echo "  这份文件**不能**用来算达到 occupancy，也不要喂给 " \
       "\`kernoscope occupancy --from\`。" >&2
  exit 3
fi

N=$(grep -cE '^[0-9]+,[0-9]+,[0-9]+$' "$OUT" || true)
N=${N:-0}
echo "记录行数: $N"
grep -E '^# (coverage|cta_rows|grid_blocks|launches_traced|theoretical_occupancy_pct) ' \
     "$OUT" | sed 's/^/  /' || true
if [ "$N" = "0" ]; then
  echo "！没有任何 CTA 记录：确认 NO_EAGER_LOAD=0（本脚本已设）、工具架构与 GPU 匹配" >&2
  echo "  （ARCH=native 重建）、KS_OCC_KERNEL 过滤的是真正要测的那个 kernel。" >&2
  echo "  排查：KS_OCC_DEBUG_CALLS=1 重跑，看注入调用执行了几次。" >&2
  exit 3
fi
echo "下一步: kernoscope occupancy --from ${OUT} --write-json occ.json"
