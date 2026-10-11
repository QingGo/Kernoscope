#!/usr/bin/env python3
"""生成给 agent 的参考层文档（L3）：`refusal-codes.md` 与 `verdicts.md`。

为什么要生成而不是手写：这两份是**静态知识**（拒绝码与 remedy、判词目录与方向），
它们已经是代码里的表（`refusals.py` / `packet.py` / `diff_cmd.py`）。手抄一遍的唯一
结果是——总有一天会和代码不一致，而"会漂移的文档"比"没有文档"更坏：
agent 会照着一份过期的 remedy 去执行。

所以：单一来源是代码，本脚本只负责渲染；`--check` 进测试（与 COVERAGE.md 同一套纪律）。

用法：
    python tools/gen_agent_refs.py            # 写出两份文档 + 刷新 SKILL.md 的参考层表格
    python tools/gen_agent_refs.py --check    # 只校验（不一致则 exit 1）

`SKILL.md` 里那张"参考层"表格的字节数也是这里**现算**的：手抄的价格一定会漂
（加一个拒绝码会让 `refusal-codes.md` 变大，而价格标签不会自己跟上）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernoscope.diff_cmd import DIRECTION, MDD_KEY              # noqa: E402
from kernoscope.packet import (PACKET_SCHEMA_REVISION, PREMISE_CODES,  # noqa: E402
                               PREMISE_LABELS, VERDICT_IDS)
from kernoscope.refusals import (CODE_TITLES, DEFAULT_REMEDIES, RefusalCode,  # noqa: E402
                                 code_title, default_remedy)

REFS = ROOT / "skills" / "kernoscope" / "references"
SKILL = ROOT / "skills" / "kernoscope" / "SKILL.md"

HEADER = ("<!-- 由 tools/gen_agent_refs.py 生成；改动请改代码后重跑，不要手改本文件 -->\n")

REFS_BEGIN = "<!-- refs-index:begin -->"
REFS_END = "<!-- refs-index:end -->"

# 参考层清单：文件名、一句话说明、是否由本脚本生成。字节数现算。
REF_DOCS = [
    ("refusal-codes.md", "每个拒绝码什么意思、怎么办、代价", True),
    ("verdicts.md", "判词目录、方向、前提、MDD 键", True),
    # 受限环境的 signal→cause→fix 手册：由 tools/gen_playbook.py 从同一批代码（refusals /
    # route_cmd / capabilities / sections）生成，所以它和上面两份一样属于**生成物**。
    ("diagnosis-playbook.md", "受限环境诊断手册：signal→cause→fix、任务路由、补不回来的采集", True),
    ("provenance-rules.md", "出处纪律的短版", False),
    ("recipes.md", "四个闭环配方", False),
]

# 判词的中文问法（与各 builder 里的 question 保持一致；这里只用于文档）
VERDICT_QUESTION = {
    "kernel_time": "哪个 kernel 最耗时、每次跑多久",
    "dram_bytes": "DRAM 搬了多少字节 / 是不是 DRAM bound",
    "dram_bytes_per_launch": "每次 launch 搬多少字节",
    "dram_bytes_trace": "DRAM 字节的第二条口径（访存轨迹 + 缓存模型）",
    "ledger_verdict": "与你的账本差多少（同口径对拍）",
    "occupancy": "occupancy 上界与实际达到值",
    "roofline": "离 roofline 多远、compute 还是 memory bound",
    "issue_slot": "发射槽用了多少、瓶颈方向",
    "l2_locality": "L1/L2 命中率与工作集",
    "over_peak": "达成值是否超过标定峰值（物理上不可能）",
    "static_mix": "静态 SASS 指令混合",
}

_PREMISES_BY_VERDICT = {
    "kernel_time": ["kernel_share_of_total", "kernel_time_pct_of_wall"],
    "dram_bytes": ["window_coverage_pct", "window_mem_busy_pct"],
    "dram_bytes_per_launch": ["window_coverage_pct", "matched_kernel_kernel_share_pct"],
    "dram_bytes_trace": ["trace_vs_nvml_pct"],
    "ledger_verdict": ["继承 dram_bytes_per_launch 的前提"],
    "occupancy": [],
    "roofline": [],
    "issue_slot": [],
    "l2_locality": [],
    "over_peak": ["over_peak"],
    "static_mix": [],
}

_DIR_LABEL = {"lower": "越低越好", "higher": "越高越好", "": "不判方向（只给差值）"}


def _cost_text(rem) -> str:
    if rem.cost_kind == "unavailable":
        return "本机不可行"
    if rem.cost_kind == "manual":
        return "需人工"
    if rem.cost_s <= 0:
        return "≈0"
    return f"≈{rem.cost_s:.0f}s" if rem.cost_s < 90 else f"≈{rem.cost_s / 60:.0f}min"


def render_refusal_codes() -> str:
    with_cmd = sum(1 for c in RefusalCode
                   if DEFAULT_REMEDIES.get(c) and DEFAULT_REMEDIES[c].command)
    no_cmd = len(list(RefusalCode)) - with_cmd
    out = [HEADER, "# 拒绝码（`refusal.code`）参考\n",
           # 这句以前写成"每条 n/a 都带观测值 + 门槛 + 可执行 remedy"—— 实测 41 行 n/a 里
           # threshold 非空 0/41、16 个 code 没有默认命令。散文按实测（且现算）写，不夸大。
           f"每条 `n/a` 都带一个**闭集 code**（判决在 code 里，散文只解释）；"
           f"{len(RefusalCode)} 个 code 里 {with_cmd} 个带可直接执行的命令、"
           f"{no_cmd} 个只给动作（多为本机走不通的路，故意不给命令免得让人白试）。"
           "观测值与门槛只有在**该前提本身可观测**时才给。",
           "**硬规则：`status=refused` 的判词不得作为行动依据**；先按 remedy 解除前提，",
           "或换一条口径，再下结论。改 code / 加 code 请改 `src/kernoscope/refusals.py`。\n",
           f"共 {len(RefusalCode)} 个 code（本文件由代码生成，与 `DEFAULT_REMEDIES` 同源）。\n",
           "| code | 什么意思 | 怎么办 | 命令 | 代价 |",
           "|---|---|---|---|---|"]
    for code in RefusalCode:
        rem = DEFAULT_REMEDIES.get(code, default_remedy(code))
        cmd = f"`{rem.command}`" if rem.command else ""
        out.append(f"| `{code.value}` | {code_title(code)} | {rem.action} | {cmd} | "
                   f"{_cost_text(rem)} |")
    missing = [c.value for c in RefusalCode if c not in DEFAULT_REMEDIES]
    if missing:  # pragma: no cover - 由测试保证为空
        out.append(f"\n> 注意：以下 code 没有登记 remedy（属于缺陷）：{', '.join(missing)}")
    out.append("")
    out.append("## 常见组合\n")
    out.append("- 拿到 `no_nvml_samples` / `no_calibration`：**先补测量前提**，不要改代码。")
    out.append("- 拿到 `share_too_low` / `coverage_too_low` / `multi_kernel_window`：")
    out.append("  窗口不成立 —— 缩小窗口、换轨迹口径，或走整窗口口径。")
    out.append("- 拿到 `counters_denied`：与 ncu 同源的那类精确量在这台机器上不可得；")
    out.append("  容器内没有合法绕过，换物理观测量口径。")
    out.append("")
    return "\n".join(out)


def render_verdicts() -> str:
    out = [HEADER, f"# 判词（`verdicts[]`）参考 · 决策包 schema rev {PACKET_SCHEMA_REVISION}\n",
           "决策包里每条判词有 `id` / `status` / `actionable` / `facts` / `premises`"
           "（可选 `refusal` / `band` / `mdd` / `next`）。",
           "**硬规则：只有 `actionable=true` 的判词能当行动依据**（v1.0 的包没有这个字段，",
           "回退规则是「不是 refused 且列出的前提全部成立」）。\n",
           "| 判词 id | 问的是什么 | 方向（diff 用） | 需要的前提 | 被拒时的 code 来源 |",
           "|---|---|---|---|---|"]
    for vid in VERDICT_IDS:
        premise_ids = _PREMISES_BY_VERDICT.get(vid, [])
        premises = "、".join(PREMISE_LABELS.get(p, p) for p in premise_ids) or "（无强制前提）"
        codes = sorted({PREMISE_CODES[p].value for p in premise_ids if p in PREMISE_CODES})
        code_txt = f"`{'`, `'.join(codes)}`" if codes else "见 packet 的 `refusals` 表"
        out.append(f"| `{vid}` | {VERDICT_QUESTION.get(vid, vid)} | "
                   f"{_DIR_LABEL[DIRECTION.get(vid, '')]} | {premises} | {code_txt} |")
    out.append("")
    out.append("## 状态与可依据性\n")
    out.append("| status | 含义 | actionable |")
    out.append("|---|---|---|")
    out.append("| `ok` | 与 ncu 同源或本机实测（`exact` / `measured`） | true |")
    out.append("| `soft` | 估计/代理口径（`estimated` / `derived`），引用时必须带前提与误差带 | true |")
    out.append("| `refused` | 不给数字，只给 code + remedy | false |")
    out.append("")
    out.append("## 比较两次运行时的 MDD\n")
    out.append("`diff` 用 MDD 判定「真差别 vs 噪声」；MDD 来自 `kernoscope selftest` 的产物"
               "（`--ks-selfcheck`）。")
    out.append("两次运行还必须**同协议**（`protocol.hash` 相同），否则一律 `incomparable`。\n")
    out.append("| 判词 | 用哪个 MDD 键 |")
    out.append("|---|---|")
    for vid in VERDICT_IDS:
        if vid in MDD_KEY:
            out.append(f"| `{vid}` | `{MDD_KEY[vid]}` |")
    out.append("")
    return "\n".join(out)


FILES = {"refusal-codes.md": render_refusal_codes, "verdicts.md": render_verdicts}


def _size_text(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


def render_skill_refs_index() -> str:
    """SKILL.md 里的参考层表格（字节数现算，不手抄）。"""
    out = [REFS_BEGIN,
           "<!-- 由 tools/gen_agent_refs.py 生成（字节数是现算的）；改动请改那里的 REF_DOCS -->",
           "",
           "| 文件 | 字节 | 里面有什么 |",
           "|---|---|---|"]
    for name, what, generated in REF_DOCS:
        p = REFS / name
        size = _size_text(p.stat().st_size) if p.exists() else "?"
        tag = "（生成）" if generated else ""
        out.append(f"| `references/{name}` | ~{size}{tag} | {what} |")
    out += ["", REFS_END]
    return "\n".join(out)


def sync_skill_index(check: bool) -> int:
    text = SKILL.read_text(encoding="utf-8")
    if REFS_BEGIN not in text or REFS_END not in text:
        print(f"{SKILL.relative_to(ROOT)}: 缺少参考层标记块（{REFS_BEGIN}）")
        return 1
    new = (text[:text.index(REFS_BEGIN)] + render_skill_refs_index()
           + text[text.index(REFS_END) + len(REFS_END):])
    if check:
        if new != text:
            print(f"{SKILL.relative_to(ROOT)}: 参考层表格与文件实际大小不一致（重跑生成器）")
            return 1
        print("ok  SKILL.md 参考层表格")
        return 0
    SKILL.write_text(new, encoding="utf-8")
    print(f"wrote {SKILL.relative_to(ROOT)} 的参考层表格")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验给 agent 的参考层文档")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args(argv)
    REFS.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, fn in FILES.items():
        text = fn()
        path = REFS / name
        if args.check:
            old = path.read_text(encoding="utf-8") if path.exists() else None
            if old != text:
                print(f"{path.relative_to(ROOT)}: 与代码不一致（跑 tools/gen_agent_refs.py 重新生成）")
                bad += 1
            else:
                print(f"ok  {name}  {len(text.encode('utf-8'))} B")
            continue
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}  {len(text.encode('utf-8'))} B")
    bad += sync_skill_index(args.check)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
