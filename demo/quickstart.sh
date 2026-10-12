#!/usr/bin/env bash
# Kernoscope demo —— 一条命令看清：ncu 在这台机器上能做什么、我们在这台机器上能测到什么、
# 拿不到的那些指标我们给什么替代口径。
#
#   bash demo/quickstart.sh                    # 默认：两个场景 + 本机标定 + 示例价格
#   bash demo/quickstart.sh --scene gemm       # 只跑 GEMM（单 kernel，结论最干净）
#   bash demo/quickstart.sh --scene train_step # 只跑一步训练（多 kernel + 阶段归因 + 每 token 成本）
#   bash demo/quickstart.sh --price-per-hour 2.5   # 用你自己的租用价（元/小时）
#   bash demo/quickstart.sh --kwh 0.8              # 或者用你自己的电价（元/kWh）
#   bash demo/quickstart.sh --no-ncu           # 跳过第一幕（不想在这里跑 ncu 时）
#
# 产物落在 demo/out/：一屏文本（截图用）、单文件 HTML（截图/发帖用）、决策包、机器标定。
# 设计原则：**能测的都测** —— 本机峰值在 demo 里现测（约 3 秒），价格用示例值（可覆盖），
# token 数由负载自身给出，于是"能省多少 / 每 token 多少钱 / roofline"这些结论都能真的出数，
# 而不是一路拒绝。
#
# 退出码：0 跑完；1 有一步失败；3 环境缺件（没有 GPU / 没有 torch / 没有 kernoscope）。
set -uo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
OUT="${ROOT}/demo/out"
SCENES=()
PRICE_PER_HOUR="${DEMO_PRICE_PER_HOUR:-2.5}"     # 示例租用价：元/小时（AutoDL 4090 量级）
KWH="${DEMO_KWH:-}"                              # 只给其中一个；给了 kwh 就用它
WITH_NCU=1

while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENES+=("$2"); shift 2 ;;
    --price-per-hour) PRICE_PER_HOUR="$2"; shift 2 ;;
    --kwh) KWH="$2"; shift 2 ;;
    --no-ncu) WITH_NCU=0; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "用法见 python3 demo/quickstart.sh --help"; exit 2 ;;
  esac
done
[ "${#SCENES[@]}" -eq 0 ] && SCENES=(gemm train_step)

say()  { printf '%s\n' "$*"; }
head1() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail=0
bad() { say "  ✗ $*"; fail=1; }

# ---- 解释器：**必须能 import torch**（demo 的负载是 torch 写的）；复用 _ks.sh 的发现顺序 ----
KS_SH="${ROOT}/skills/kernoscope/scripts/_ks.sh"
[ -f "$KS_SH" ] || { say "demo 需要仓库里的 skills/kernoscope/scripts/_ks.sh"; exit 3; }
# shellcheck source=/dev/null
source "$KS_SH"
PY="$(ks_python)"
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  for cand in "$HOME/miniconda3/bin/python3" /root/miniconda3/bin/python3 \
              "$HOME/anaconda3/bin/python3" /opt/conda/bin/python3 python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  say "demo 需要一台有 GPU 且装了 torch 的机器（当前解释器 ${PY} 里没有 torch）"; exit 3
fi
if ! "$PY" -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  say "demo 需要可用的 CUDA 设备（当前解释器能 import torch，但 torch.cuda.is_available() 为假）"
  exit 3
fi

# ---- 用哪个 kernoscope：**永远是 demo 所在的这棵源码树** ----
# 真机教训（AutoDL 容器实测）：`command -v kernoscope` 可能指向**别人留下的旧版本**
# （那台机上 /usr/local/bin/kernoscope 是个包装脚本，指向验收方冻结的 /root/ksfix/ks11/src，
# 连 `--html` 都没有）。demo 的职责是展示**这份仓库**的能力，所以源码树优先，并且
# 把"PATH 上另有 X"这件事显式说出来 —— 否则用户看到的结论与被测的代码对不上。
KS_CMD=()
if [ -d "${ROOT}/src/kernoscope" ]; then
  KS_CMD=("$PY" -m kernoscope)
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  ON_PATH="$(command -v kernoscope 2>/dev/null || true)"
  if [ -n "$ON_PATH" ]; then
    say "注意：PATH 上还有一个 kernoscope（${ON_PATH}）—— 它可能是旧版本；"
    say "      本 demo 用的是**这份仓库的源码树**：${ROOT}/src（一次只用一个，避免结论与代码对不上）"
  fi
elif command -v kernoscope >/dev/null 2>&1; then
  KS_CMD=(kernoscope)
  say "本 demo 不在源码树里，用 PATH 上的 kernoscope：$(command -v kernoscope)"
else
  say "找不到 kernoscope：既不在源码树，也不在 PATH（试试 uv tool install kernoscope）"; exit 3
fi
KS_RUN() { "${KS_CMD[@]}" "$@"; }

mkdir -p "$OUT"
DEV="$("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
say "Kernoscope demo · 设备：${DEV} · 解释器：${PY}"
say "产物目录：${OUT}"

# =====================================================================
# 第一幕：ncu 在这台机器上到底能不能用（**真的跑一次**，把它的报错原样放出来）
# =====================================================================
if [ "$WITH_NCU" = "1" ]; then
  head1 "第一幕：ncu 在这台机器上能不能用（我们真的跑一次，报错也照登）"
  NCU="$("$PY" -c "
import sys
try:
    from kernoscope.toolchain import find
except Exception:
    sys.exit(1)
print(find('ncu') or '')
" 2>/dev/null)"
  if [ -z "$NCU" ]; then
    say "  这台机器上没有 ncu（Kernoscope 的 doctor 会把它记成 counters_not_installed）"
  else
    say "  找到 ncu：${NCU}"
    say "  跑一次最小采集（--set basic，1 次 launch）——这是 ncu 自己的输出："
    # 只为**拿到 ncu 的报错**，所以用 0.5 秒的小负载就够（默认 3 秒是给分析的，不必在这里付）
    DEMO_SECONDS=0.5 "$NCU" --set basic --launch-count 1 \
      -- "$PY" "${ROOT}/demo/workloads/${SCENES[0]}.py" >"$OUT/ncu.out" 2>"$OUT/ncu.err"
    ncu_rc=$?
    say "  ncu rc=${ncu_rc}"
    grep -E "ERR_|ERROR|Error" "$OUT/ncu.err" "$OUT/ncu.out" 2>/dev/null | head -3 \
      | sed 's/^/    /' || say "    （没有 ERR_ 行）"
    say "  ↑ 这不是我们的报错，是 ncu 在**这台机器**上取不到硬件计数器的结果；"
    say "    下面是我们对同一件事的结论（含原因与替代口径）。"
  fi
fi

# =====================================================================
# 第二幕：这台机器上我们能答什么（1 KB 的入口层 + 计数器状态的一句人话）
# =====================================================================
head1 "第二幕：这台机器上我们**能**回答什么"
KS_RUN capabilities --brief >"$OUT/capabilities.json" 2>"$OUT/capabilities.err"
say "  capabilities --brief rc=$? （$(wc -c <"$OUT/capabilities.json" | tr -d ' ') B）"
"$PY" - "$OUT/capabilities.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  计数器状态：{d.get('counter_state')}")
print(f"  现在能答：{'、'.join(d.get('can_answer') or []) or '（无）'}")
if d.get("blocked_by"):
    print(f"  缺的依赖：{'、'.join(d['blocked_by'])}")
PY
KS_RUN doctor --json >"$OUT/doctor.json" 2>/dev/null
"$PY" - "$OUT/doctor.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
note = d.get("counter_state_note")
if note:
    print(f"  计数器状态说明：{note}")
PY

# =====================================================================
# 第三幕：把能测的都测上（本机标定 + 示例价格 + token 数）→ 一屏 + HTML
# =====================================================================
head1 "第三幕：一次把能测的都测上（本机峰值现测，价格用示例值）"
if [ -s "$OUT/machine-profile.json" ]; then
  say "  复用已有标定：$OUT/machine-profile.json（想重测就删掉它）"
else
  say "  先做一次本机标定（--quick，约 3 秒）——roofline 与“能省多少”的分母必须是**这台机器**实测的"
  if "$PY" "${ROOT}/tools/machine-profile.py" --quick --out "$OUT/machine-profile.json" \
       >"$OUT/calibration.log" 2>&1; then
    say "  标定完成：$OUT/machine-profile.json（$(wc -c <"$OUT/machine-profile.json" | tr -d ' ') B；本机实测峰值，roofline 的分母）"
  else
    bad "标定失败：见 ${OUT}/calibration.log"
  fi
fi

for scene in "${SCENES[@]}"; do
  wf="${ROOT}/demo/workloads/${scene}.py"
  [ -f "$wf" ] || { bad "没有这个场景：${scene}（可选 gemm / train_step）"; continue; }
  case "$scene" in
    gemm)
                # GEMM 没有 token 概念：**不喂** --ks-tokens（喂了会让"每 token 成本"变成一个
                # 没有意义的数）；它的看点是 roofline 与带宽，token 口径留给 train_step。
                TOKENS=""
                WS_ARGS=(--ks-gemm "${DEMO_M:-8192},${DEMO_N:-8192},${DEMO_K:-8192}" --ks-dtype bf16) ;;
    train_step) TOKENS="$("$PY" -c 'import os;print(int(os.environ.get("DEMO_BATCH","8"))*int(os.environ.get("DEMO_SEQ","512")))')"
                WS_ARGS=() ;;
    *)          TOKENS=0; WS_ARGS=() ;;
  esac
  price_args=()
  if [ -n "$KWH" ]; then price_args=(--ks-price-per-kwh "$KWH")
  else price_args=(--ks-price-per-hour "$PRICE_PER_HOUR"); fi

  say ""
  say "  --- 场景 ${scene}（$([ -n "$TOKENS" ] && echo "token 数 ${TOKENS}；" || echo "")价格 $([ -n "$KWH" ] && echo "${KWH} 元/kWh" || echo "${PRICE_PER_HOUR} 元/小时（示例，可改）")）---"
  token_args=()
  [ -n "$TOKENS" ] && token_args=(--ks-tokens "$TOKENS")
  KS_RUN --ks-calibration "$OUT/machine-profile.json" "${price_args[@]}" \
         "${token_args[@]}" "${WS_ARGS[@]}" \
         --screen only -o "$OUT/rep-${scene}.json" \
         --html "$OUT/rep-${scene}.html" \
         --packet "$OUT/packet-${scene}.json" \
         -f -- "$PY" "$wf" >"$OUT/onepage-${scene}.txt" 2>"$OUT/run-${scene}.err"
  rc=$?
  if [ "$rc" != "0" ]; then
    bad "场景 ${scene} 失败（rc=${rc}）：见 ${OUT}/run-${scene}.err"
    tail -2 "$OUT/run-${scene}.err" | sed 's/^/    /'
    continue
  fi
  say "  一屏：$(wc -c <"$OUT/onepage-${scene}.txt" | tr -d ' ') B / $(wc -l <"$OUT/onepage-${scene}.txt" | tr -d ' ') 行"
  say "  单文件 HTML：$OUT/rep-${scene}.html（自包含，可直接截图/发帖）"
  if grep -q "coverage_too_low" "$OUT/onepage-${scene}.txt"; then
    say "  提示：有结论被 coverage_too_low 挡住（窗口里 GPU 忙占比 <80% ⇒ 比值类口径不成立）。"
    say "        共享容器上这很常见（宿主间隙/采样粒度都会稀释它）。想让它成立：把负载跑长跑大"
    say "        （DEMO_SECONDS=10 DEMO_BATCH=32），或换独占机器。"
  fi
  cat "$OUT/onepage-${scene}.txt" | sed 's/^/    /'
done

# =====================================================================
# 第四幕：ncu 能给、这里拿不到的，我们给什么（**从包里生成**，不手抄）
# =====================================================================
head1 "第四幕：计数器拿不到的那部分，我们给什么"
for scene in "${SCENES[@]}"; do
  pkt="$OUT/packet-${scene}.json"
  [ -s "$pkt" ] || continue
  say ""
  say "  --- 场景 ${scene} ---"
  "$PY" - "$pkt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ref = d.get("refusals") or {}
for v in d.get("verdicts") or []:
    if v.get("status") != "refused":
        continue
    code = (v.get("refusal") or {}).get("code", "?")
    act = (ref.get(code) or {}).get("action") or ""
    print(f"  · {v['id']}：不给数字（{code}）→ {act[:110]}")
um = d.get("unmeasured") or []
if um:
    print("  · 没测到的分组（按原因）：")
    for g in um[:4]:
        print(f"      {g['count']} 个 {g['code']}（{(ref.get(g['code']) or {}).get('title') or ''}）")
cost = d.get("cost") or {}
print(f"  · 本次测到 {cost.get('measured_metrics', '?')} 项指标；"
      f"wall {cost.get('wall_s')}s、包 {cost.get('packet_bytes')}B")
PY
done

# =====================================================================
# 第五幕：与 ncu 的兼容性 —— CLI 与产物
# =====================================================================
head1 "第五幕：与 ncu 的兼容性（CLI 与产物）"
mkdir -p "$OUT/bin"
# ⚠️ heredoc 的分隔符**必须带引号**：不带引号时 bash 会对正文做命令替换与变量展开 ——
# 正文里写 `` `argv[0]` `` 这种说明文字会被真的当命令执行（真机实测：`argv[0]: command not found`）。
# 所以正文里不做任何展开，源码树路径改成**运行时**从环境变量读。
export KS_DEMO_SRC="${ROOT}/src"
# shebang 必须写**当前这个解释器的绝对路径**：真机实测 `#!/usr/bin/env python` 会 rc=127
# （那台容器的 PATH 里没有 `python`，只有 /root/miniconda3/bin/python3）。
printf '#!%s\n' "$PY" > "$OUT/bin/ncu"
cat >> "$OUT/bin/ncu" <<'PYEOF'
"""把 Kernoscope 当成 ncu 用（等价于 README §4.1 的 `ln -s $(which kernoscope) ~/bin/ncu`）。

兼容模式按 argv[0] 的基名判断是不是 ncu —— 所以 shim 里显式把它设成 "ncu"。
源码树路径由 KS_DEMO_SRC 传入（避免在 heredoc 里做展开）。
"""
import os
import sys

sys.path.insert(0, os.environ["KS_DEMO_SRC"])
sys.argv[0] = "ncu"
from kernoscope.cli import main

raise SystemExit(main())
PYEOF
chmod +x "$OUT/bin/ncu"
say "  ① CLI 兼容：ncu 风格调用（argv[0]=ncu 自动进兼容模式；未知开关会明确报错）"
# 用 `--ks-synthetic` 演示**开关兼容**（1 秒内），不重复跑真负载 —— 真负载已经在第三幕跑过了
"$OUT/bin/ncu" --set basic --ks-synthetic --screen off -- true \
  >"$OUT/ncu-compat.out" 2>"$OUT/ncu-compat.err"
say "     rc=$? ；报告字节=$(wc -c <"$OUT/ncu-compat.out" | tr -d ' ')"
grep -iE "兼容|ncu" "$OUT/ncu-compat.err" | head -2 | sed 's/^/     /'
say "  ② 产物兼容：导出合法 .ncu-rep + 署名（哪些表是同源真值、哪些是替代口径）"
# 复用第三幕已经产出的报告（回读模式），而不是把负载再跑一遍：demo 的每一秒都该花在
# "展示能力"上，不是花在重复采集上（实测：跑一遍真负载 ≈ 9 秒 + nsys ≈ 2 秒）。
KS_RUN -i "$OUT/rep-${SCENES[0]}.ksreport.json" --export-ncu-rep "$OUT/rep.ncu-rep" \
       >/dev/null 2>"$OUT/export.err"
say "     rc=$? ；.ncu-rep=$(wc -c <"$OUT/rep.ncu-rep" 2>/dev/null | tr -d ' ') B"
head -8 "$OUT/rep.ncu-rep.ATTRIBUTION.txt" 2>/dev/null | sed 's/^/     /'
say "     提醒：我们写的是**合法** .ncu-rep，但每个量的出处写在署名里 —— 同源真值走 ncu 名下，"
say "           替代口径走 ks__/est__/meas__，绝不占用 ncu 的指标名。"

head1 "怎么改"
say "  · 换价格：--price-per-hour 2.5（元/小时）或 --kwh 0.8（元/kWh）"
say "  · 换规模/时长：DEMO_SECONDS=5 DEMO_BATCH=16 DEMO_SEQ=1024 bash demo/quickstart.sh"
say "  · 换场景：--scene gemm | --scene train_step"
say "  · 截图：demo/out/onepage-*.txt（纯文本一屏）或 demo/out/rep-*.html（单文件，离线可开）"
say ""
if [ "$fail" = "0" ]; then say "DEMO OK"; else say "DEMO 有失败步骤（见上）"; fi
exit "$fail"
