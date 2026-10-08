#!/usr/bin/env python3
"""样例 agent（**故意幼稚**）：忽略 status 与前提，挑最大的数字就行动。

它的存在是为了证明评测台的检测器**真的有效**：一个只看数字大小的 agent，会把
refused 判词、前提 `ok=false` 的判词、以及被真值禁止的判词写进 acted_on，
于是 misuse_rate 必然 > 0。如果连它都拿 0 误用率，那说明评测台的判据是坏的。

真实世界里这种 agent 很常见：它不做错"读 JSON"这件事，它只是把
「数字存在」当成「数字可用」。

用法（harness 会把 cwd 设成材料目录）：
    python eval/agents/rule_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ACTION_BY_FAMILY = {
    "dram": "DRAM 数字最大，说明访存是瓶颈：把 per-launch DRAM 流量降下来，"
            "先改 kernel 的访存合并与共享内存复用。",
    "l2_locality": "工作集小、命中率高：DRAM 流量很小，转向优化计算。",
    "roofline": "roofline 有数字：按 bound 字段指的那一侧改对应代码。",
    "issue_slot": "发射槽利用率有数字：按这个比例精细调发射槽与指令混合。",
    "occupancy": "occupancy 有数字：先把这个数字顶到理论上界再谈别的。",
    "kernel_time": "kernel 时间有数字：按这个耗时挑最贵的 kernel，直接改它。",
}
DEFAULT_ACTION = "按这个最大的数字改代码，把数字降下来。"


def find_packet(cwd: Path) -> dict:
    for name in ("packet.json", os.environ.get("KS_EVAL_PACKET") or ""):
        if not name:
            continue
        p = cwd / name if not Path(name).is_absolute() else Path(name)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(cwd.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("tool") == "kernoscope" and "verdicts" in d:
            return d
    raise SystemExit("rule_baseline: 找不到决策包（packet.json）")


def headline(v: dict) -> float:
    """判词的"头条数字"：只看 value，不看它可不可用（这正是幼稚之处）。"""
    val = v.get("value")
    if isinstance(val, bool) or val is None:
        return float("-inf")
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val))
    except ValueError:
        return float("-inf")


def family(vid: str) -> str:
    if vid.startswith("dram"):
        return "dram"
    return vid


def classify(top: dict) -> str:
    fam = family(str(top.get("id", "")))
    if fam == "dram":
        return "dram_bound"
    if fam == "roofline":
        bound = (top.get("facts") or {}).get("bound") or "memory"
        return f"{bound}_bound"
    if fam == "l2_locality":
        return "not_dram_bound"
    if fam in ("issue_slot", "occupancy"):
        return "compute_bound"
    return "dram_bound"          # 看不出来时也当成访存问题（幼稚的默认）


def decide(packet: dict) -> dict:
    verdicts = packet.get("verdicts") or []
    # 忽略 status：refused 的判词没有 value，自然排不上；但前提不成立的 soft 判词有
    ranked = sorted(verdicts, key=headline, reverse=True)
    picked = [v for v in ranked if headline(v) > float("-inf")][:2]
    acted_on = [str(v.get("id")) for v in picked]
    top = picked[0] if picked else {}
    parts: list[str] = []
    for v in picked:
        text = ACTION_BY_FAMILY.get(family(str(v.get("id", ""))), DEFAULT_ACTION)
        if text not in parts:
            parts.append(text)
    return {
        "classification": classify(top) if picked else "dram_bound",
        "acted_on": acted_on,
        "next_action": "；".join(parts) if parts else DEFAULT_ACTION,
        "tokens_used": 0,          # 纯脚本：不编造 token 数
        "gpu_minutes": 0.0,
    }


def main() -> int:
    cwd = Path.cwd()
    packet = find_packet(cwd)
    decision = decide(packet)
    out = Path(os.environ.get("KS_EVAL_DECISION") or (cwd / "decision.json"))
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rule_baseline: 挑了最大的 {len(decision['acted_on'])} 个数字 → {decision['acted_on']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
