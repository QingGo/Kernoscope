#!/usr/bin/env bash
# 陌生人首跑（G1）的可重跑检查 —— 每一步都必须 rc=0 且产出非空。
#
# 为什么要它：`docs/private-polish-plan.md` 的开放门槛 G1 要求「在一台**不是你的**机器上，
# 从零到第一个结论 < 5 分钟；三种环境都要验证」。此前这条只有人工记录，而 README 推荐的
# 第一条命令（`doctor`）在**没有 GPU 的机器上 rc=1 且 stderr 0 B** —— 陌生人与 CI 第一眼
# 看到的是"失败且没有解释"（易用性实测）。这个脚本把首跑路径变成**可重跑、可进 CI 的检查**。
#
# 它做什么：把仓库**打成的 wheel** 装进一个隔离的工具目录（不碰全局、不碰仓库），
# 然后按 README §2 的顺序跑三条命令，逐条断言 rc=0、产出非空，并打印字节数与耗时。
#
# 用法：bash bench/preflight-first-run.sh [--keep]
#   --keep  保留临时目录（排查用；默认删掉）
# 退出码：0 = 全部通过；1 = 有步骤失败；3 = 环境缺件（没有 uv / 没有可装的 wheel）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

say() { printf '%s\n' "$*"; }
die() { printf 'PREFLIGHT FAIL: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || { say "PREFLIGHT SKIP: 没有 uv（首跑路径依赖 uv；装上再跑）"; exit 3; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/ks-preflight.XXXXXX")"
# 自带一个可写的 uv 缓存目录：陌生机器/沙箱里 uv 的**默认**缓存位置不一定可写，
# 而"缓存目录不可写"会让 `uv build` 失败，看起来像产品坏了（实测过一次）。
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv-cache}"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

# 1) **现场构建** wheel —— 模拟"从 PyPI 装"的那一步。
#    为什么不用现成的 dist/*.whl：它可能是**上一个提交**构建的，于是这个脚本会验一个陈旧产物
#    而给出假绿（本脚本第一版就是这样：装了旧 wheel，doctor 仍然 rc=1）。
#    要复用现成产物（例如 CI 刚构建过）时显式给 KS_PREFLIGHT_USE_DIST=1。
if [ "${KS_PREFLIGHT_USE_DIST:-0}" = "1" ]; then
  WHEEL="$(ls -1t "$ROOT"/dist/kernoscope-*.whl 2>/dev/null | head -1)"
  [ -n "$WHEEL" ] || die "dist/ 里没有 wheel（先跑 uv build）"
else
  say "[0/3] 现场构建 wheel（uv build --out-dir <tmp>）"
  (cd "$ROOT" && uv build --wheel --out-dir "$WORK/dist" >/dev/null 2>&1) || die "uv build 失败"
  WHEEL="$(ls -1t "$WORK"/dist/kernoscope-*.whl 2>/dev/null | head -1)"
fi
[ -n "$WHEEL" ] || die "找不到 wheel（先跑 uv build）"
say "wheel: $(basename "$WHEEL")  ($(wc -c <"$WHEEL") B)"

# 2) 隔离安装（等价于 uv tool install kernoscope）
export UV_TOOL_DIR="$WORK/tools" UV_TOOL_BIN_DIR="$WORK/bin"
T0=$(date +%s)
if ! uv tool install --quiet "$WHEEL" >"$WORK/install.log" 2>&1; then
  say "--- install.log ---"; cat "$WORK/install.log"; die "uv tool install 失败"
fi
T1=$(date +%s)
say "[0/3] 装进隔离目录：$((T1 - T0)) s，可执行文件 $UV_TOOL_BIN_DIR/kernoscope"

KS="$UV_TOOL_BIN_DIR/kernoscope"
[ -x "$KS" ] || KS="$(command -v kernoscope)"      # 兜底：工具目录布局变化时
[ -x "$KS" ] || die "装完了却找不到 kernoscope 可执行文件"

FAIL=0
step() {  # step <序号> <说明> <期望 rc> <命令...>
  local n="$1" desc="$2" want="$3"; shift 3
  local out; out="$WORK/step-$n.out"
  local t0 t1 rc
  t0=$(date +%s%N)
  "$@" >"$out" 2>"$WORK/step-$n.err"; rc=$?
  t1=$(date +%s%N)
  local bytes ms
  bytes=$(wc -c <"$out" | tr -d ' ')
  ms=$(( (t1 - t0) / 1000000 ))
  if [ "$rc" != "$want" ]; then
    printf '  ✗ [%s] %s → rc=%s（期望 %s），stderr：%s\n' "$n" "$desc" "$rc" "$want" \
      "$(head -c 200 "$WORK/step-$n.err")"
    FAIL=1; return
  fi
  if [ "$bytes" -eq 0 ]; then
    printf '  ✗ [%s] %s → rc=0 但 stdout 是空的\n' "$n" "$desc"; FAIL=1; return
  fi
  printf '  ✓ [%s] %s → rc=%s，%s B，%s ms\n' "$n" "$desc" "$rc" "$bytes" "$ms"
}

say "[1/3] 这台机器能答什么（README 的第一条命令）"
step 1 "capabilities --brief" 0 "$KS" capabilities --brief
# 第一条命令就必须给出"下一步做什么"（否则"≤2 条命令到第一条可行动结论"是空话）
grep -q '"next"' "$WORK/step-1.out" || { say "  ✗ capabilities --brief 里没有 next[]"; FAIL=1; }

say "[2/3] 拿不到什么、为什么"
step 2 "doctor" 0 "$KS" doctor

say "[3/3] 没有 GPU 也能跑通全流程"
step 3 "doctor --no-probe" 0 "$KS" doctor --no-probe
step 4 "--ks-synthetic --set basic" 0 "$KS" --ks-synthetic --set basic
step 5 "--packet -（决策包）" 0 "$KS" --packet - --ks-synthetic

say "[3b/3] 一屏的人视图（README 的「先看这一段」）"
step 6 "--screen only（人读的一屏）" 0 "$KS" --ks-synthetic --set basic --screen only
if [ -f "$WORK/step-6.out" ]; then
  ONE_BYTES=$(wc -c <"$WORK/step-6.out" | tr -d ' ')
  ONE_LINES=$(wc -l <"$WORK/step-6.out" | tr -d ' ')
  # "≤1 屏"必须可判：预算 8 KB / 60 行（当前约 4.4 KB / 39 行）。超了说明又长成报告了。
  if [ "$ONE_BYTES" -gt 8192 ] || [ "$ONE_LINES" -gt 60 ]; then
    printf '  ✗ 一屏 %s B / %s 行 超预算（≤8192 B / ≤60 行）—— README 承诺的是"一屏"\n' \
      "$ONE_BYTES" "$ONE_LINES"; FAIL=1
  else
    printf '  ✓ [6] 一屏 %s B / %s 行（预算 ≤8192 B / ≤60 行）\n' "$ONE_BYTES" "$ONE_LINES"
  fi
fi

if [ "$FAIL" != 0 ]; then
  say "PREFLIGHT FAIL（临时目录保留在 ${WORK}，或用 --keep 重跑）"
  exit 1
fi
say "PREFLIGHT OK 6/6（陌生首跑路径：每一步 rc=0、产出非空；第一步就带 next[]；一屏在预算内）"
