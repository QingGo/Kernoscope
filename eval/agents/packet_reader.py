#!/usr/bin/env python3
"""样例 agent（**规范做法**）：只依据 status 与 premises 下结论。

它回答的不是"哪个数字最大"，而是"哪条判词的前提成立到可以支撑一个行动"：

* `status == refused` 的判词 → 进 `blocked_by`（不是证据，是"这条路暂时不可用"）；
* 任一前提 `ok == false` 的判词 → 进 `blocked_by`（数字在，但口径不成立）；
* 只有 ok/soft 且前提全成立、并且在包里的判词 → 才进 `acted_on`；
* 前提不成立/口径不可用时，**下一步是改测量口径、不是改代码**。

它的误用率必须是 0，准确率必须是 100% —— 否则要么真值写错了，要么评测台的判据坏了。

用法（harness 会把 cwd 设成材料目录）：
    python eval/agents/packet_reader.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 口径"缺件"类拒绝：数据源不在 ⇒ 这个方向根本没有读数
CALIBER_MISSING_CODES = {
    "counters_denied", "counters_not_installed", "counters_unknown",
    "no_nvml", "no_nvml_samples", "sampling_unavailable",
    "no_calibration", "calibration_caliber_mismatch",
    "nvbit_missing", "no_memtrace", "no_instr_hist", "instrumentation_missing",
}
# 读数"低于分辨率"类拒绝：口径在，但这次读不出来 ⇒ 不是 DRAM bound 的证据
BELOW_RESOLUTION_CODES = {"below_resolution", "mem_busy_zero", "not_applicable"}
# 归因前提：这些前提不成立时，per-launch/窗口数字不能归给某个 kernel
ATTRIBUTION_PREMISES = {"matched_kernel_kernel_share_pct", "share_unknown",
                        "window_coverage_pct"}
ATTRIBUTION_REFUSALS = {"multi_kernel_window", "share_too_low", "share_unknown",
                        "coverage_too_low"}

DRAM_IDS = ("dram_bytes", "dram_bytes_per_launch", "dram_bytes_trace")
L2_RESIDENT_MIB = 32.0            # 工作集远小于 L2 时，DRAM 口径本来就量不到


def find_packet(cwd: Path) -> dict:
    env_path = os.environ.get("KS_EVAL_PACKET")
    candidates = [Path(env_path)] if env_path else []
    candidates.append(cwd / "packet.json")
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(cwd.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("tool") == "kernoscope" and "verdicts" in d:
            return d
    raise SystemExit("packet_reader: 找不到决策包（packet.json）")


# --------------------------------------------------------------------------
# 判词可用性：唯一判据是 status 与 premises
# --------------------------------------------------------------------------
def refused_code(v: dict) -> str | None:
    return (v.get("refusal") or {}).get("code")


def failed_premises(v: dict) -> list[dict]:
    return [p for p in (v.get("premises") or []) if not p.get("ok", True)]


def usable(v: dict) -> bool:
    return v.get("status") in ("ok", "soft") and not failed_premises(v)


def premise_of(v: dict, pid: str):
    for p in (v.get("premises") or []):
        if p.get("id") == pid:
            return p.get("observed")
    return None


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.1f}"
    return str(x)


# --------------------------------------------------------------------------
# 决策
# --------------------------------------------------------------------------
def decide(packet: dict) -> dict:
    verdicts = [v for v in (packet.get("verdicts") or []) if v.get("id")]
    by_id = {v["id"]: v for v in verdicts}
    good = [v["id"] for v in verdicts if usable(v)]
    refused = [v["id"] for v in verdicts if v.get("status") == "refused"]
    broken = [v["id"] for v in verdicts if v.get("status") != "refused" and failed_premises(v)]
    blocked_by = [vid for vid in (broken + refused) if vid not in good]

    dram_good = [vid for vid in DRAM_IDS if vid in good]
    dram_refused_codes = {refused_code(by_id[vid]) for vid in DRAM_IDS
                          if vid in by_id and vid not in good and refused_code(by_id[vid])}

    roofline = by_id.get("roofline")
    l2 = by_id.get("l2_locality")
    attribution_broken = any(
        p.get("id") in ATTRIBUTION_PREMISES
        for vid in broken for p in failed_premises(by_id[vid])
    ) or bool(dram_refused_codes & ATTRIBUTION_REFUSALS) or any(
        refused_code(by_id[vid]) in ATTRIBUTION_REFUSALS
        for vid in (refused + broken)
    )

    # 判据顺序即优先级：先看能不能定向，再看能不能归因，最后才看流量
    if roofline is not None and "roofline" in good \
            and str((roofline.get("facts") or {}).get("bound", "")).lower() == "compute":
        eff = (roofline.get("facts") or {}).get("efficiency_pct")
        gbs = (roofline.get("facts") or {}).get("gbs")
        acted = [vid for vid in ("kernel_time", "roofline", "issue_slot") if vid in good]
        text = ("roofline 判定 compute bound"
                + (f"（离 roofline {100.0 - float(eff):.1f}%，达成带宽 {fmt(gbs)} GB/s）"
                   if eff is not None and gbs is not None else "")
                + "：方向在计算侧 —— 查指令混合与发射槽"
                + ("（方向 " + str((by_id.get('issue_slot') or {}).get('facts', {}).get('direction'))
                   + "）" if (by_id.get("issue_slot") or {}).get("facts", {}).get("direction") else "")
                + "以及精度选择；访存不是瓶颈。")
        return _decision("compute_bound", acted, text, blocked_by)

    if attribution_broken:
        share = next((p for vid in broken for p in failed_premises(by_id[vid])
                      if p.get("id") in ATTRIBUTION_PREMISES), None)
        head = "per-launch 归因前提不成立"
        if share is not None:
            head += (f"（{share.get('id')} = {fmt(share.get('observed'))}"
                     + (f"，门槛 {fmt(share.get('threshold'))}" if share.get("threshold") is not None
                        else "") + "）")
        acted = [vid for vid in ("kernel_time",) if vid in good]
        text = (head + "：先缩小窗口到只含目标 kernel（或 -k 过滤后重跑）拿到可归因读数，"
                       "在此之前不要改代码。")
        return _decision("window_unattributable", acted, text, blocked_by)

    if not dram_good and dram_refused_codes & CALIBER_MISSING_CODES:
        acted = [vid for vid in ("kernel_time", "occupancy") if vid in good]
        codes = ", ".join(sorted(dram_refused_codes & CALIBER_MISSING_CODES))
        text = (f"DRAM 口径整体不可用（{codes}）：先修测量口径 —— 恢复 NVML 采样或拿到"
                "计数器权限（或装 NVBit 用访存轨迹口径），在此之前不下任何方向的结论。")
        return _decision("dram_caliber_unusable", acted, text, blocked_by)

    if not dram_good and (dram_refused_codes & BELOW_RESOLUTION_CODES
                          or _l2_resident(by_id.get("l2_locality"))):
        acted = [vid for vid in ("kernel_time", "l2_locality") if vid in good]
        bits = []
        for vid in DRAM_IDS:
            v = by_id.get(vid)
            if v is None or vid in good or bits:
                continue
            busy = premise_of(v, "window_mem_busy_pct")
            if busy is not None:
                bits.append(f"窗口显存忙碌 {fmt(busy)}%")
        if _l2_resident(by_id.get("l2_locality")):
            ws = (by_id["l2_locality"].get("facts") or {}).get("working_set_mib")
            bits.append(f"工作集 {fmt(ws)} MiB 能驻留 L2")
        why = "低于仪表分辨率" if "below_resolution" in dram_refused_codes else "给不出可信读数"
        head = f"DRAM 读数{why}" + (f"（{'；'.join(bits)}）" if bits else "")
        text = (head + "：这条测量口径给不出字节数。先延长窗口或换访存轨迹口径拿到可信读数，"
                       "在此之前不下任何方向的结论。")
        return _decision("not_dram_bound", acted, text, blocked_by)

    if dram_good:
        v = by_id["dram_bytes"] if "dram_bytes" in good else by_id[dram_good[0]]
        busy = premise_of(v, "window_mem_busy_pct")
        cov = premise_of(v, "window_coverage_pct")
        bits = []
        if cov is not None:
            bits.append(f"窗口覆盖率 {fmt(cov)}%")
        if busy is not None:
            bits.append(f"显存忙碌 {fmt(busy)}%")
        acted = [vid for vid in ("kernel_time", "dram_bytes", "dram_bytes_per_launch")
                 if vid in good]
        text = ("访存是瓶颈"
                + (f"（{'、'.join(bits)}）" if bits else "")
                + "：先做合并访存、提高 L2 复用；改完用 kernoscope diff 确认差值超过 MDD "
                  "再说改进。")
        return _decision("dram_bound", acted, text, blocked_by)

    acted = [vid for vid in ("kernel_time",) if vid in good]
    return _decision("insufficient_evidence", acted,
                     "包里没有前提成立到足以定向的判词：先按 cost.trimmed 与 refusals 把"
                     "测量口径补齐（或读完整报告），不要先改代码。", blocked_by)


def _l2_resident(l2: dict | None) -> bool:
    if l2 is None or not usable(l2):
        return False
    ws = (l2.get("facts") or {}).get("working_set_mib")
    return isinstance(ws, (int, float)) and float(ws) <= L2_RESIDENT_MIB


def _decision(classification: str, acted_on: list[str], text: str,
              blocked_by: list[str]) -> dict:
    acted = [vid for vid in acted_on if vid not in blocked_by]
    return {
        "classification": classification,
        "acted_on": acted,
        "next_action": text,
        "blocked_by": [vid for vid in blocked_by if vid not in acted],
        "tokens_used": 0,          # 纯脚本：不编造 token 数
        "gpu_minutes": 0.0,
    }


def main() -> int:
    cwd = Path.cwd()
    packet = find_packet(cwd)
    decision = decide(packet)
    out = Path(os.environ.get("KS_EVAL_DECISION") or (cwd / "decision.json"))
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"packet_reader: {decision['classification']}；acted_on={decision['acted_on']}；"
          f"blocked_by={decision['blocked_by']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
