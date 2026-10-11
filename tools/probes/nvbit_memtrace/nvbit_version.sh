#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# nvbit_version.sh —— 从**实物**里读 NVBit 版本号（单一真源，别处不许手抄）。
#
#   tools/probes/nvbit_memtrace/nvbit_version.sh <NVBIT_PATH | mem_trace.so>
#
# 读的顺序：
#   ① `<tree>/core/nvbit.h` 里的 `#define NVBIT_VERSION "x.y.z"`（权威：NVBit 官方头文件）；
#   ② 该树的 `mem_trace.so`（两种布局都试）里的**整行**版本字符串（头文件被裁剪过时用）。
# 读不到就非 0 退出、且**不打印任何东西**（调用方自己决定写 n/a）—— 绝不编一个版本出来。
#
# 为什么要有它：验收报告 V-10 —— `bench/artifacts/*/environment.txt` 只记了 NVBit 的**路径**，
# 没记版本；而"哪个版本能用"正是那一整节结论的全部依据（1.8.1 在本环境实测失败）。
set -uo pipefail

arg="${1:-}"
if [ -z "$arg" ]; then
  echo "用法: $0 <NVBIT_PATH | mem_trace.so>" >&2
  exit 2
fi

# ① 头文件。给的是 .so 时先往上找出它所属的 NVBit 树（.../tools[/build]/mem_trace/mem_trace.so）。
hdr=""
if [ -d "$arg" ]; then
  hdr="$arg/core/nvbit.h"
elif [ -f "$arg" ]; then
  d="$(cd "$(dirname "$arg")" && pwd)"
  for _ in 1 2 3 4 5; do
    if [ -f "$d/core/nvbit.h" ]; then hdr="$d/core/nvbit.h"; break; fi
    [ "$d" = "/" ] && break
    d="$(dirname "$d")"
  done
fi
if [ -n "$hdr" ] && [ -f "$hdr" ]; then
  v="$(grep -oE 'NVBIT_VERSION[[:space:]]+"[^"]+"' "$hdr" 2>/dev/null | head -1 \
       | sed -E 's/.*"([^"]+)".*/\1/' || true)"
  if [ -n "${v:-}" ]; then printf '%s\n' "$v"; exit 0; fi
fi

# ② 退回到 .so 里的版本字符串（官方构建把它作为一个独立的字符串常量放在里面）。
so=""
if [ -f "$arg" ]; then
  so="$arg"
elif [ -d "$arg" ]; then
  for c in "$arg/tools/build/mem_trace/mem_trace.so" "$arg/tools/mem_trace/mem_trace.so"; do
    if [ -f "$c" ]; then so="$c"; break; fi
  done
fi
[ -n "$so" ] || exit 1
if command -v strings >/dev/null 2>&1; then
  blob="$(strings -a "$so" 2>/dev/null || true)"
else
  blob="$(tr -c '[:print:]' '\n' < "$so" 2>/dev/null || true)"
fi
# NVBit 的版本号形如 1.x[.y[.z]]；只要**整行就是版本号**的那些（避免把别的库版本串进来）。
v="$(printf '%s\n' "$blob" | grep -xE '1\.[0-9]+(\.[0-9]+)+' | head -1 || true)"
[ -n "${v:-}" ] && { printf '%s\n' "$v"; exit 0; }
exit 1
