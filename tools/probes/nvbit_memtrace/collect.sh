#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 一键采集访存轨迹（用 NVBit 官方 mem_trace 工具）。
#
#   NVBIT_MEMTRACE_SO=/path/to/mem_trace.so KS_MEMTRACE_OUT=t.txt \
#       ./collect.sh -- ./your_app args
#
# 工具选择与版本（R-72：以前这里把"布局"和"版本号"写成了互相矛盾的一对）：
#   * 本仓库实测**可用**的组合：NVBit **1.8.1** + CMake 构建（`cmake --build … --target mem_trace`
#     或新版 `make --target mem_trace`）→ `tools/build/mem_trace/mem_trace.so`（本脚本的默认路径）。
#     注意 1.8.1 里**不要** `make` 整个 tools：`mem_trace_tma` 用 `%cluster_ctaid` 要求 sm_90+，
#     会把整棵树带崩；架构变量要**裸数字**（写 `89`，不是 `sm_89`）。
#   * 另一条实测可用的组合：NVBit **v1.7.7.3** + 老 Makefile 布局
#     （`make -C $NVBIT_PATH/tools/mem_trace ARCH=sm_89`，产出 tools/build/mem_trace/mem_trace.so）。
#     验收报告第四轮在那台上是这么跑通的（1.8.1 的整树 make 会失败）。
#   * 两条都要求 `NO_EAGER_LOAD=0`：1.8 默认懒加载会让插桩错过时机（程序跑完但零记录）。
# 官方工具产出的是"一行 = 一条 warp 访存指令 + 全部 lane 地址"，粒度刚好够用。
#
# ⚠️ 必须 NO_EAGER_LOAD=0：NVBit 1.8 默认懒加载模块，会让插桩错过时机，
#    表现为"程序正常跑完但一条记录都没有"。
set -euo pipefail

# 自动发现官方工具：$NVBIT_PATH/tools/build/mem_trace/mem_trace.so（NVBit 自带的示例）
if [ -z "${NVBIT_MEMTRACE_SO:-}" ] && [ -n "${NVBIT_PATH:-}" ]; then
  for cand in "$NVBIT_PATH/tools/build/mem_trace/mem_trace.so" \
              "$NVBIT_PATH/tools/build/mem_trace/mem_trace.so" \
              "$NVBIT_PATH/tools/build/mem_trace/*.so"; do
    if [ -f "$cand" ]; then NVBIT_MEMTRACE_SO="$cand"; break; fi
  done
fi
: "${NVBIT_MEMTRACE_SO:?请设置 NVBIT_MEMTRACE_SO=/path/to/mem_trace.so，或用 NVBIT_PATH 指向 NVBit 安装目录（工具需自行构建）}"
if [ ! -f "$NVBIT_MEMTRACE_SO" ]; then
  echo "！NVBIT_MEMTRACE_SO 指向的文件不存在: $NVBIT_MEMTRACE_SO" >&2
  echo "  构建官方工具（在 NVBit 目录下）：" >&2
  echo "    make -C tools/mem_trace    # 产出 tools/build/mem_trace/mem_trace.so" >&2
  exit 2
fi
# 工具"发现"了还不够：NVBit 的官方 mem_trace 在启动时直接 `exec` `cuobjdump`，
# 而它按 **PATH** 找 —— 租用机上 CUDA 常常装了但不在 PATH（本机就是：nvcc/cuobjdump 都在
# /usr/local/cuda-13.0/bin）。缺它时官方工具会以 `ERROR: cuobjdump not found on PATH!!!` 退出，
# 而"我明明装了 CUDA"会让人往别处查。所以这里按 doctor 的同一套发现逻辑补 PATH。
# （这条是 tests/test_nvbit_smoke.py 按文档冒烟时抓出来的 —— 文档说的路径当时跑不通。）
if ! command -v cuobjdump >/dev/null 2>&1; then
  for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" "${CUDA_BIN:-}" /usr/local/cuda /usr/local/cuda-*; do
    [ -n "$d" ] && [ -x "$d/bin/cuobjdump" ] && { PATH="$d/bin:$PATH"; break; }
  done
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "！找不到 cuobjdump：NVBit 官方 mem_trace 需要它来 dump kernel 镜像。" >&2
  echo "  把它所在的 bin 目录加进 PATH（例如 CUDA_BIN=/usr/local/cuda-13.0；" >&2
  echo "  本机 nvcc/cuobjdump 常在这个目录而 PATH 里没有）。" >&2
  exit 2
fi

OUT="${KS_MEMTRACE_OUT:-memtrace-official.txt}"

if [ "${1:-}" = "--" ]; then shift; fi
if [ "$#" -eq 0 ]; then
  echo "用法: NVBIT_MEMTRACE_SO=... [KS_MEMTRACE_OUT=out.txt] $0 -- <命令> [参数...]" >&2
  exit 2
fi

# 采集是**无界**的：一行 = 一条 warp 访存指令，而官方工具不会自己限流 ——
# 真事：一个紧循环 1 秒就把 6.6 GB 写满，`echo` 报 `No space left on device`（验收报告
# §10.7 也记过"两次把轨迹采到被截断 / 一次把盘吃满"）。所以这里做两件事：
#   ① 采集前检查剩余空间；
#   ② 用 `ulimit -f` 给输出文件设上限（默认 2 GiB，KS_MEMTRACE_MAX_MB 可调）。
# 被上限截断时**不会有结束标记**，解析器据此报"无法确认完整"（R-76），不会静默给偏小的数。
MAX_MB="${KS_MEMTRACE_MAX_MB:-2048}"
OUT_DIR="$(dirname "$OUT")"
FREE_MB=$(df -Pk "$OUT_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1024)}')
# 上限取 min(用户设的, 剩余空间-256MB)：**永远不把盘写满**，同时尽量不因为"盘小"就拒绝干活
if [ -n "${FREE_MB:-}" ]; then
  HEADROOM=$(( FREE_MB > 512 ? FREE_MB - 256 : FREE_MB ))
  if [ "$HEADROOM" -lt 64 ]; then
    echo "！剩余空间只有 ${FREE_MB} MB，不够采轨迹（至少 64 MB 余量）。先清空间。" >&2
    exit 2
  fi
  if [ "$HEADROOM" -lt "$MAX_MB" ]; then
    echo "注：剩余空间 ${FREE_MB} MB ⇒ 本次实际上限降到 ${HEADROOM} MiB（KS_MEMTRACE_MAX_MB=${MAX_MB}）" >&2
    MAX_MB="$HEADROOM"
  fi
fi

echo "采集 → $OUT（NO_EAGER_LOAD=0；上限 ${MAX_MB} MiB，超出即截断且**不会**写结束标记）"
set +e
ulimit -f $((MAX_MB * 1024)) 2>/dev/null || true     # 512B 块：MAX_MB MiB
NO_EAGER_LOAD=0 \
CUDA_INJECTION64_PATH="$NVBIT_MEMTRACE_SO" \
LD_PRELOAD="$NVBIT_MEMTRACE_SO" \
NOBANNER=1 \
"$@" > "$OUT" 2>&1
RC=$?
set -e
echo "被测程序退出码: $RC"
if [ "$RC" -ne 0 ] && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" -ge $((MAX_MB * 1024 * 1024 - 1048576)) ]; then
  echo "！轨迹达到了体积上限 ${MAX_MB} MiB（多半是被 ulimit 截断）：" >&2
  echo "  这条轨迹**不能**用来算 DRAM 字节（解析器会按'无法确认完整'处理）。" >&2
  echo "  请缩小负载/只让被测 kernel 进轨迹，或提高 KS_MEMTRACE_MAX_MB 后重采。" >&2
fi
if [ "$RC" -ne 0 ]; then
  echo "！被测程序没跑成功（退出码 $RC）—— 先看 $OUT 的末尾几行：" >&2
  tail -5 "$OUT" >&2
  exit "$RC"
fi

# R-76：官方格式**没有结束标记**，被 ulimit/磁盘写满截断时工具察觉不到（验收报告里
# 两次因此拿到低于物理下界的数）。我们自己采集的轨迹补一个哨兵，解析器据此判定完整性。
echo "MEMTRACE: KS-TRACE-COMPLETE" >> "$OUT"

N=$(grep -c 'MEMTRACE:.*warp ' "$OUT" 2>/dev/null || true)
N=${N:-0}
echo "记录行数: $N"
if [ "$N" = "0" ]; then
  echo "！没有任何记录：确认 NO_EAGER_LOAD=0（本脚本已设）、工具架构与 GPU 匹配、app 确实跑了 kernel" >&2
  exit 1
fi
grep 'MEMTRACE:.*LAUNCH' "$OUT" | sed 's/^/  kernel: /' | head -5
echo "下一步: kernoscope memtrace --from $OUT --sm-count <SM数> --l2-bytes <L2字节> [--kernel <子串>]"
