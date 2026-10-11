#!/usr/bin/env bash
# 发布预演 / 发布后验证（**同一条命令**，两种来源）：
#
#   bash bench/release-dry-run.sh                 # 预演：现场构建 sdist+wheel，从**本地 wheel** 隔离安装
#   bash bench/release-dry-run.sh --from-pypi     # 发布后：从 **PyPI** 装（`kernoscope==<版本>`）再验一遍
#   bash bench/release-dry-run.sh --metadata-only # 只构建 + 检查元数据（不装，最快）
#
# 为什么要有它：README §2 的三条安装路径里，`uv tool install kernoscope` 只有在包**真的发上去**
# 之后才可用（第七轮 X-21：发布前它是 404，而文档已经宣布了同日发布）。这条脚本把"发布后要验什么"
# 提前写成命令，于是 ①发布前能预演隔离安装，②发布后一条命令就能给出通过/不通过，
# ③两者检查的是同一组断言（不是两份会漂的清单）。
#
# 它验的是**用户第一分钟真的会做的事**：
#   ① 产物元数据：名字/版本/许可/Alpha 分类器/Python 下限/入口点；测试目录不许进包；
#   ② 从**隔离目录**安装（不碰你现有的 tool 安装），装出来 2 个可执行文件；
#   ③ 中性工作目录下跑首用三连（`capabilities --brief` / `doctor --no-probe` / 一屏），
#      每条 rc=0，且两个预算成立：`capabilities --brief` ≤1.5 KB、一屏 ≤8192 B / ≤60 行。
#
# 退出码：0 通过；1 有断言不通过；3 环境缺件（没有 uv）。
set -uo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
MODE="local"
case "${1:-}" in
  --from-pypi) MODE="pypi" ;;
  --metadata-only) MODE="meta" ;;
  "") ;;
  *) echo "用法：bash bench/release-dry-run.sh [--from-pypi|--metadata-only]"; exit 2 ;;
esac

command -v uv >/dev/null 2>&1 || { echo "RELEASE SKIP: 没有 uv（发布要用 uv；装上再跑）"; exit 3; }

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || { echo "RELEASE FAIL: 读不到 pyproject 的 version"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/ks-release.XXXXXX")"
CACHE="$(mktemp -d "${TMPDIR:-/tmp}/ks-release-cache.XXXXXX")"
trap 'rm -rf "$WORK" "$CACHE"' EXIT
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE}"
export UV_TOOL_DIR="$WORK/tools" UV_TOOL_BIN_DIR="$WORK/bin"     # 隔离：不碰你现有的安装

fail=0
say() { printf '%s\n' "$*"; }
ok()  { say "  ✓ $*"; }
bad() { say "  ✗ $*"; fail=1; }

say "==> 版本：${VERSION}（模式：${MODE}）"

# ---------- ① 构建 + 元数据 ----------
SRC=""
if [ "$MODE" = "pypi" ]; then
  SRC="kernoscope==$VERSION"
  say "==> 从 PyPI 装（${SRC}）"
else
  say "==> 现场构建 sdist + wheel"
  (cd "$ROOT" && uv build --out-dir "$WORK/dist" >/dev/null 2>&1) \
    || { bad "uv build 失败"; say "RELEASE FAIL"; exit 1; }
  WHEEL="$(ls "$WORK"/dist/*.whl 2>/dev/null | head -1)"
  [ -n "$WHEEL" ] || { bad "构建目录里没有 wheel"; say "RELEASE FAIL"; exit 1; }
  ok "$(basename "$WHEEL")（$(wc -c < "$WHEEL" | tr -d ' ') B）"
  SRC="$WHEEL"

  # 元数据断言是**一份共用脚本**（`publish.yml` 在 CI 里调用的也是它）：
  # 发布前预演、CI 发布、发布后验证三处只有一组判据，不会再漂。
  if python3 "$ROOT/bench/check_dist_metadata.py" "$WHEEL" --expect-version "$VERSION"; then
    :
  else
    fail=1
  fi

fi

if [ "$MODE" = "meta" ]; then
  [ "$fail" -eq 0 ] && say "RELEASE OK（元数据；未做隔离安装）" || say "RELEASE FAIL"
  exit "$fail"
fi

# ---------- ② 隔离安装 ----------
say "==> 隔离安装（UV_TOOL_DIR=${UV_TOOL_DIR}）"
if ! uv tool install --force "$SRC" >/dev/null 2>&1; then
  bad "隔离安装失败（${SRC}）"
  [ "$MODE" = "pypi" ] && say "  提示：PyPI 上还没有这个版本？先按 RELEASING.md §1 发布。"
  say "RELEASE FAIL"; exit 1
fi
K="$UV_TOOL_BIN_DIR/kernoscope"
[ -x "$K" ] || { bad "装完没有 kernoscope 可执行文件"; say "RELEASE FAIL"; exit 1; }
ok "装出：$(ls "$UV_TOOL_BIN_DIR" | tr '\n' ' ')"

# ---------- ③ 中性工作目录下的首用三连 ----------
cd "$WORK"          # 刻意离开仓库：验证装出来的包**自己**能跑
run() { "$K" "$@" >"$WORK/out.txt" 2>"$WORK/err.txt"; }
bytes_of() { wc -c < "$WORK/out.txt" | tr -d ' '; }

say "==> 首用三连（工作目录 ${WORK}，与仓库无关）"
if run capabilities --brief; then
  n="$(bytes_of)"
  [ "$n" -le 1536 ] && ok "capabilities --brief：rc=0，${n} B（预算 ≤1.5 KB）" \
                    || bad "capabilities --brief 有 ${n} B（超 1.5 KB 预算）"
  grep -q '"next"' "$WORK/out.txt" && ok "第一步就带 next[]（告诉人下一步做什么）" \
                                   || bad "capabilities --brief 里没有 next[]"
else
  bad "capabilities --brief rc≠0：$(tail -1 "$WORK/err.txt")"
fi

if run doctor --no-probe; then
  ok "doctor --no-probe：rc=0，$(bytes_of) B"
else
  bad "doctor --no-probe rc≠0：$(tail -1 "$WORK/err.txt")"
fi

if run --ks-synthetic --set basic --screen only; then
  n="$(bytes_of)"; lines="$(wc -l < "$WORK/out.txt" | tr -d ' ')"
  if [ "$n" -le 8192 ] && [ "$lines" -le 60 ]; then
    ok "一屏：rc=0，${n} B / ${lines} 行（预算 ≤8192 B / ≤60 行）"
  else
    bad "一屏超预算：${n} B / ${lines} 行"
  fi
else
  bad "一屏路径 rc≠0：$(tail -1 "$WORK/err.txt")"
fi

if [ "$MODE" = "pypi" ]; then
  say "==> 发布后验证通过：PyPI 上的 kernoscope==$VERSION 能装、能跑、首屏在预算内"
else
  say "==> 预演通过：本地 wheel 能装、能跑、首屏在预算内"
  say "    发布命令（Trusted Publisher / OIDC：推 tag → CI 发，无需任何密钥）："
  say "      git tag -a v${VERSION} -m \"v${VERSION}\" && git push origin v${VERSION}"
  say "      （tag 指向的提交里要已经有 .github/workflows/publish.yml）"
  say "    应急（CI 不可用时才用项目级 token；见 RELEASING §1.1）："
  say "      uv build && UV_PUBLISH_TOKEN=pypi-… uv publish"
  say "    发布后跑同一条命令的另一半：bash bench/release-dry-run.sh --from-pypi"
fi
[ "$fail" -eq 0 ] && say "RELEASE OK" || say "RELEASE FAIL"
exit "$fail"
