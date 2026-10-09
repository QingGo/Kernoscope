#!/usr/bin/env bash
# 找 kernoscope：优先 PATH 里的可执行文件，否则用源码树（PYTHONPATH=src python -m kernoscope）。
# 被 scripts/ 下的其它脚本 source。
set -uo pipefail

KS_SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KS_SKILL_ROOT="$(cd "$KS_SKILL_DIR/../../.." && pwd)"

ks() {
  if command -v kernoscope >/dev/null 2>&1; then
    kernoscope "$@"
  elif [[ -d "$KS_SKILL_ROOT/src/kernoscope" ]]; then
    PYTHONPATH="$KS_SKILL_ROOT/src" "${KS_PY:-python3}" -m kernoscope "$@"
  else
    echo "error: 找不到 kernoscope（既不在 PATH，也不在当前仓库的 src/ 下）" >&2
    return 127
  fi
}

ks_help() {  # 打印本脚本开头连续的注释行作为 --help
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$1"
}

# 本机自检文件：默认放在仓库根（可用 KS_SELFCHECK 覆盖）
ks_selfcheck_path() { echo "${KS_SELFCHECK:-$KS_SKILL_ROOT/machine-selfcheck.json}"; }
