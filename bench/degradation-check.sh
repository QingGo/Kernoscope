#!/usr/bin/env bash
# bench/degradation-check.sh —— W1 的"三种环境"验收：缺 nsys 时必须**成功降级**，不是报错退出。
#
# 为什么需要它：`bench/run.sh` 跑的是"这台机器一切正常"的路径。而 W1 的验收条件里有一条
# 只有在"缺件"时才成立：没有 nsys 时仍然要给出设备级结论 + 明确说明缺什么会多出什么，
# 并且退出码 0（缺依赖是成功路径，不是失败）。
#
# 怎么模拟"缺 nsys"：把 nsys 从 PATH 与候选目录里**临时**移开。这会改动系统状态，
# 所以默认不跑；要跑必须显式同意：
#
#   KS_ALLOW_NSYS_MOVE=1 bash bench/degradation-check.sh [输出目录]
#
# 脚本用 trap 保证无论怎么退出都把 nsys 移回原位；移动前会先记录原路径。
# 它不会碰 ncu / CUDA / NVBit，也不会改任何仓库文件。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-bench/artifacts/degradation-$(date +%Y%m%d-%H%M%S)}"
PY="${PY:-python3}"
KSC() { PYTHONPATH="$ROOT/src" "$PY" -m kernoscope "$@"; }

if [[ "${KS_ALLOW_NSYS_MOVE:-0}" != "1" ]]; then
  echo "跳过：这个脚本会临时把 nsys 移开，需要 KS_ALLOW_NSYS_MOVE=1 显式同意" >&2
  exit 0
fi

NSYS="$(command -v nsys || true)"
if [[ -z "$NSYS" ]]; then
  echo "这台机器本来就没有 nsys —— 那正好是环境 B，直接跑即可" >&2
  MOVE=0
else
  MOVE=1
  OFF="${NSYS}.off"
  restore() {
    if [[ -e "$OFF" ]]; then mv "$OFF" "$NSYS"; echo "[deg] nsys 已恢复：$NSYS"; fi
  }
  trap restore EXIT INT TERM
fi

mkdir -p "$OUT"
TARGET=("$PY" -c "import time; time.sleep(1.5)")   # 纯 CPU 目标：不占 GPU，但仍让采样器有机会取点

echo "== 环境 A：当前状态（应有 nsys）=="
KSC capabilities --json > "$OUT/capabilities-full.json" 2>/dev/null
"$PY" - "$OUT/capabilities-full.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  tier={d['tier']['level']} {d['tier']['name']} · nsys={d['env']['has_nsys']} · "
      f"nvbit={d['env']['has_nvbit']} · counters={d['env']['counter_state']}")
EOF

if [[ "$MOVE" == "1" ]]; then
  mv "$NSYS" "$OFF"
  echo "== 环境 B：有 GPU、没有 nsys（临时移开 $NSYS）=="
else
  echo "== 环境 B：这台机器本来没有 nsys =="
fi
KSC capabilities --json > "$OUT/capabilities-no-nsys.json" 2>/dev/null
"$PY" - "$OUT/capabilities-no-nsys.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  tier={d['tier']['level']} {d['tier']['name']} · nsys={d['env']['has_nsys']} · "
      f"kernel_time={d['matrix']['kernel_time']} · device_busy={d['matrix']['device_busy']}")
EOF

KSC --set basic --packet "$OUT/degraded.packet.json" --screen only -- "${TARGET[@]}" \
    > "$OUT/degraded.onepage.txt" 2> "$OUT/degraded.err"
rc=$?
echo "  降级运行退出码 = ${rc}（验收要求 0）"
"$PY" - "$OUT/degraded.packet.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  决策包 {d['cost']['packet_bytes']} B · verdicts={[v['id'] for v in d['verdicts']]}")
print(f"  environment_refusals={[r['code'] for r in d.get('environment_refusals', [])]}")
print(f"  blocked_by={d['env'].get('blocked_by')}")
EOF
echo "  一屏前 3 行："
head -3 "$OUT/degraded.onepage.txt" | sed 's/^/    /'

echo
echo "== 环境 C：没有 GPU =="
echo "  在一台没有 NVIDIA 设备的机器上跑同样的命令："
echo "    kernoscope capabilities        # 期望 tier 0 (demo)，exit 0"
echo "    kernoscope --set basic -- python -c pass   # 期望 exit 0 + 一屏说明缺什么"
echo "  （本脚本不做 GPU 掩蔽：伪装「没有 GPU」比伪装「没有 nsys」更容易给出错的证据）"
echo
echo "产物：$OUT"
