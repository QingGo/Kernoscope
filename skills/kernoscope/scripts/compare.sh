#!/usr/bin/env bash
# compare.sh —— 比较两次运行，并且**强制执行 MDD 纪律**：
#   协议不同 → exit 2（不下结论）；没有 MDD → exit 2（不下结论）；
#   有真退化（超过 MDD 且方向是变坏）→ exit 1；否则 exit 0。
#
#   bash compare.sh before.json after.json
#
# 两次输入可以是决策包，也可以是完整报告（*.ksreport.json）。
# 注意：这里不把「某些判词不可比」（例如两侧都被拒的轨迹口径）当成整轮失败 ——
# 那些判词本来就没有数字可比，结论由其余判词给出。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_ks.sh"

case "${1:-}" in
  --help|-h) ks_help "${BASH_SOURCE[0]}"; exit 0;;
esac
[[ $# -ge 2 ]] || { echo "用法: compare.sh <before.json> <after.json>" >&2; exit 2; }

sc="$(ks_selfcheck_path)"
[[ -f "$sc" ]] || {
  echo "error: 没有本机误差带（${sc}）—— 先跑 \`kernoscope selftest --out ${sc}\`；没有 MDD 时不下结论" >&2
  exit 2; }

json="$(ks diff "$1" "$2" --selfcheck "$sc" --json)" || {
  echo "error: diff 没能给出结果" >&2; exit 2; }

printf '%s' "$json" | "${KS_PY:-python3}" -c '
import json, sys
d = json.load(sys.stdin)
same = d["protocol"]["same"]
mdd = d["mdd"]["values"]
t = d["summary"]
print("协议相同=%s · MDD=%s · improved=%s regressed=%s indistinguishable=%s incomparable=%s"
      % (same, mdd or "无", t["improved"], t["regressed"],
         t["indistinguishable"], t["incomparable"]))
if not same:
    print("不可比：" + ", ".join("%s=%s" % kv for kv in d["protocol"]["diffs"].items()))
    sys.exit(2)
if not mdd:
    print("没有 MDD：先跑 kernoscope selftest（工具拒绝在没有 MDD 时下结论）")
    sys.exit(2)
for c in d["comparisons"]:
    if c["status"] in ("improved", "regressed"):
        tag = "REGRESSED" if c["status"] == "regressed" else "improved"
        print("  [%s] %s delta %s%% vs MDD %s%%"
              % (tag, c["id"], c.get("delta_pct"), c.get("mdd")))
    elif c["status"] == "indistinguishable":
        print("  [noise] %s delta %s%% <= MDD %s%% —— 不要当改进"
              % (c["id"], c.get("delta_pct"), c.get("mdd")))
sys.exit(1 if t["regressed"] else 0)
'
