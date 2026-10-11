#!/usr/bin/env bash
# CI 对等检查：把 **HEAD 的干净快照** 导出到临时目录、装好依赖、跑与 CI **完全相同**的那条命令。
#
# 为什么需要它（两次真实的红）：
#   ① 一条测试依赖"仓库里有没有 `.venv`" —— 本机有，CI 由 `uv sync` 创建 ⇒ 本机绿、CI 红；
#   ② 另一条把 `PATH` 清空，脚本里的 `dirname` 因此失效、根目录解析错 ⇒ **本机假绿**。
#   两次的共同点：**本地跑的是"我的工作树 + 我的环境"，CI 跑的是"刚 clone 的某次提交"**。
#   这条脚本把后者搬到本地，代价约 1 分钟。
#
# 用法：bash bench/ci-parity.sh [<commit-ish>]     （默认 HEAD）
# 退出码：0 = 干净快照上 9/9；1 = 门禁红（就是 CI 会红的那一条）；3 = 环境缺件（没有 uv）。
set -uo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
REF="${1:-HEAD}"

command -v uv >/dev/null 2>&1 || { echo "CI-PARITY SKIP: 没有 uv（CI 用 uv；装上再跑）"; exit 3; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/ks-ci-parity.XXXXXX")"
# uv 缓存必须放在快照**外面**：放在里面时 `uv build` 会把 uv-cache/ 一起打进 sdist，
# 而缓存里有指向 uv 托管解释器的**符号链接** ⇒ sdist 直接报 "symlink destination … is outside
# of the target directory"（本脚本第一版就踩了这个，`build` 门禁因此假红）。
CACHE="$(mktemp -d "${TMPDIR:-/tmp}/ks-ci-cache.XXXXXX")"
trap 'rm -rf "$WORK" "$CACHE"' EXIT
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE}"

echo "[1/3] 导出 $REF 的干净快照（只有被跟踪的文件）→ $WORK"
git -C "$ROOT" archive "$REF" | tar -x -C "$WORK" || { echo "CI-PARITY FAIL: git archive 失败"; exit 1; }

# CI 的 checkout 里**有** `.git`（文档命令门禁靠 `git ls-files` 取被跟踪的 *.md）。
# 所以这里也造一个等价的仓库，而不是留一个没有版本控制的目录 —— 否则门禁会因为
# "空输入集"直接红，看到的就不是 CI 的那个红了。
echo "[2/3] 造一个等价的工作树（git init + 全部文件入库）"
(cd "$WORK" && git init -q && git add -A \
  && git -c user.name=ci -c user.email=ci@example.invalid commit -qm "snapshot of $REF") \
  || { echo "CI-PARITY FAIL: 造快照仓库失败"; exit 1; }

echo "[3/3] uv sync && uv run python tools/dev_index.py --gates"
(cd "$WORK" && uv sync --quiet) || { echo "CI-PARITY FAIL: uv sync 失败"; exit 1; }
if (cd "$WORK" && uv run python tools/dev_index.py --gates); then
  echo "CI-PARITY OK：干净快照上 9/9（CI 不会再因为"只在你的工作树里成立"的东西变红）"
  exit 0
fi
echo "CI-PARITY FAIL：干净快照上有门禁红 —— 这就是 CI 会看到的那一条。"
echo "提示：把红的那条测试改成**自带夹具**（自己造树/自己设 PATH），别依赖仓库的 .venv 或环境。"
exit 1
