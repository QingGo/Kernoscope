#!/usr/bin/env bash
# 找 kernoscope：优先 PATH 里的可执行文件，否则用源码树（PYTHONPATH=src python -m kernoscope）。
# 被 scripts/ 下的其它脚本 source。
set -uo pipefail

KS_SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KS_SKILL_ROOT="$(cd "$KS_SKILL_DIR/../../.." && pwd)"

# 找一个能用的 Python 解释器（第五轮真机教训：容器里 `python3` 常常不在 PATH，
# 只有 `/root/miniconda3/bin/python3` 这种绝对路径 —— 脚本写死 `python3` 就会 127）。
#: 这个工具需要 Python ≥3.10（`X | None` 那类语法）。**每个候选都要过这一关**：
#: macOS 的 `/usr/bin/python3` 可能是 3.9（甚至是只提示装 Xcode 的桩），
#: 排在前面就会把真能用的解释器挡住 —— 本条与"找过 conda 路径"是同一次真机教训的两半。
_ks_py_ok() {
  [[ -n "${1:-}" ]] || return 1
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    >/dev/null 2>&1
}

ks_python() {
  local cand shebang
  if [[ -n "${KS_PY:-}" ]]; then echo "$KS_PY"; return 0; fi
  # 1) kernoscope 这个 console script 的 shebang（只有它确实是某个 python 时才信）
  if command -v kernoscope >/dev/null 2>&1; then
    shebang="$(head -1 "$(command -v kernoscope)" 2>/dev/null || true)"
    case "$shebang" in
      '#!'*)
        shebang="${shebang#\#!}"; shebang="${shebang%% *}"
        if [[ "$shebang" == *python* && -x "$shebang" ]] && _ks_py_ok "$shebang"; then
          echo "$shebang"; return 0
        fi;;
    esac
  fi
  # 2) 仓库或相邻的 .venv（开发树上最可能是**对的那个**解释器，所以排在系统候选之前）
  for cand in "$KS_SKILL_ROOT/.venv/bin/python" "$KS_SKILL_ROOT/../.venv/bin/python"; do
    if [[ -x "$cand" ]] && _ks_py_ok "$cand"; then echo "$cand"; return 0; fi
  done
  # 3) 常见 conda / 发行版绝对路径（真机教训：容器里 python 只在 /root/miniconda3/bin/python3，
  #    PATH 上既没有 python3 也没有 /usr/bin/python3 ⇒ 旧实现走到兜底，调用方拿到 127）
  for cand in "$HOME/miniconda3/bin/python3" "/root/miniconda3/bin/python3" \
              "$HOME/anaconda3/bin/python3" "/opt/conda/bin/python3" \
              "/usr/local/miniconda3/bin/python3" "/usr/local/bin/python3" "/usr/bin/python3"; do
    if [[ -x "$cand" ]] && _ks_py_ok "$cand"; then echo "$cand"; return 0; fi
  done
  # 4) PATH 上的 python3 / python（同样要过版本关：PATH 上的可能太旧）
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && _ks_py_ok "$(command -v "$cand")"; then
      echo "$cand"; return 0
    fi
  done
  # 5) 兜底：让报错出现在调用方，而不是这里静默吞掉。**这一步之前必须先找过 conda 与
  #    /usr/bin 候选** —— 旧顺序让真机上"PATH 没有 python3"直接落到这里（rc=127）。
  echo "python3"
}

ks() {
  if command -v kernoscope >/dev/null 2>&1; then
    kernoscope "$@"
  elif [[ -d "$KS_SKILL_ROOT/src/kernoscope" ]]; then
    PYTHONPATH="$KS_SKILL_ROOT/src" "$(ks_python)" -m kernoscope "$@"
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
