#!/usr/bin/env bash
# evidence.sh —— 跑一次目标，给出「结论优先」的证据，并把完整报告写到文件。
#
#   bash evidence.sh [kernoscope 选项…] -- <你的命令…>
#   例：bash evidence.sh -k regex:gemm --ks-calibration machine-profile.json -- python train.py
#
# stdout：一屏（结论 + 前提 + 怎么办 + 本次代价）；完整报告路径与决策包路径写到磁盘。
# 只看结论时用 stdout；要喂给程序时用 --packet 产出的包。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_ks.sh"

case "${1:-}" in
  --help|-h) ks_help "${BASH_SOURCE[0]}"; exit 0;;
esac

OUT="${KS_EVIDENCE_OUT:-$PWD/ks-evidence}"
mkdir -p "$OUT"
base="$OUT/rep-$(date +%Y%m%d-%H%M%S)"
packet="$base.packet.json"
sc="$(ks_selfcheck_path)"
args=(--packet "$packet" -o "$base" -f --screen only)
[[ -f "$sc" ]] && args+=(--ks-selfcheck "$sc")

ks "${args[@]}" "$@"
rc=$?
if [[ $rc -ne 0 ]]; then
  # 缺依赖/没匹配到 kernel 之类：工具已经把原因与下一步打出来了，如实透传退出码
  echo "== 失败（rc=${rc}）：上面的原因与 remedy 就是下一步；产物目录 ${OUT}" >&2
  exit $rc
fi
{
  echo "== 决策包（<4 KB，给 agent 读）：$packet"
  echo "== 完整报告（体积大，只在被明确要求时读）：$base.ksreport.json"
  [[ -f "$sc" ]] || echo "== 提示：还没有本机误差带，diff 前先跑 kernoscope selftest"
} >&2
