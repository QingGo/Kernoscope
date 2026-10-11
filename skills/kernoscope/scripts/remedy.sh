#!/usr/bin/env bash
# remedy.sh <code> —— 只查一个拒绝码：什么意思、怎么办、要多少代价。
# 不需要 GPU，也不需要读整张参考表（它有多大由 gen_agent_refs.py 现算）。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_ks.sh"
REFS="$KS_SKILL_ROOT/skills/kernoscope/references/refusal-codes.md"

case "${1:-}" in
  --help|-h|"") ks_help "${BASH_SOURCE[0]}"
               echo; echo "例子：bash remedy.sh share_too_low"; exit 0;;
esac
code="$1"
[[ -f "$REFS" ]] || { echo "error: 缺少参考表 ${REFS}（跑 tools/gen_agent_refs.py 生成）" >&2; exit 1; }

row="$(grep -F "| \`$code\` |" "$REFS" | head -1)"
if [[ -z "$row" ]]; then
  echo "没有这个 code：${code}（用 \`kernoscope capabilities --json\` 看本次实际出现的 code）" >&2
  exit 1
fi
IFS='|' read -r _ c meaning action cmd cost _ <<<"$row"
printf 'code    :%s\n' "$(echo "$c" | xargs)"
printf '什么意思:%s\n' "$(echo "$meaning" | xargs)"
printf '怎么办  :%s\n' "$(echo "$action" | xargs)"
printf '命令    :%s\n' "$(echo "$cmd" | xargs)"
printf '代价    :%s\n' "$(echo "$cost" | xargs)"
