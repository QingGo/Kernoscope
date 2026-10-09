#!/usr/bin/env bash
# setup-check.sh —— 开工前的一次性检查：这台机器能答什么 + 有没有本机误差带。
#
# 输出（stdout）刻意很小：它是**入口层**，只告诉 agent「能答什么、缺什么、下一步」。
#   { "tier": …, "blocked_by": [...], "next": [...], "selfcheck": "…|null", "mdd": …|null }
#
# 用法：
#   bash setup-check.sh            # 没有自检文件时会**提示**（不自动跑，跑一次要 ~1 分钟）
#   bash setup-check.sh --run      # 没有就真的跑一遍 selftest（需要 GPU + torch）
#   bash setup-check.sh --help
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_ks.sh"

case "${1:-}" in
  --help|-h)
    ks_help "${BASH_SOURCE[0]}"
    exit 0;;
esac

brief="$(ks capabilities --brief 2>/dev/null)" || {
  echo "error: capabilities --brief 失败（kernoscope 装好了吗？）" >&2; exit 1; }

selfcheck="$(ks_selfcheck_path)"
if [[ ! -f "$selfcheck" && "${1:-}" == "--run" ]]; then
  echo "note: 没有自检文件，正在跑 selftest（约 1 分钟；只需要跑一次）…" >&2
  ks selftest --out "$selfcheck" >/dev/null 2>&1 || echo "warn: selftest 没成功（缺 GPU/torch？）" >&2
fi

KS_BRIEF="$brief" KS_SELFCHECK_RESOLVED="$selfcheck" \
"${KS_PY:-python3}" - <<'PY'
import json, os, pathlib
b = json.loads(os.environ["KS_BRIEF"])
sc = pathlib.Path(os.environ["KS_SELFCHECK_RESOLVED"])
out = {
    "tier": b["tier"],
    "counter_state": b.get("counter_state"),
    "can_answer": len(b["can_answer"]),
    "cannot_answer": len(b["cannot_answer"]),
    "blocked_by": b.get("blocked_by", []),
    "next": b.get("next", [])[:2],
    "selfcheck": str(sc) if sc.exists() else None,
}
if sc.exists():
    try:
        out["mdd"] = json.loads(sc.read_text()).get("mdd", {})
    except Exception as exc:                     # noqa: BLE001
        out["selfcheck_note"] = f"读不了：{exc}"
else:
    out["selfcheck_note"] = ("还没有本机误差带：跑 `kernoscope selftest --out %s`"
                             "（约 1 分钟，之后所有 diff 都复用）" % sc)
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY
