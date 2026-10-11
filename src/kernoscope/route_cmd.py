"""`kernoscope route` —— 先路由，再动手（C1/A1 的入口）。

为什么需要它：我们的能力清单（`capabilities --brief`）回答的是"这台机器**能答什么**"，
而 agent 实际带着的是一条**问题**（"为什么这个 kernel 慢"、"DRAM 搬了多少"、"下一步优化哪里"）。
从问题到"走哪条口径、跑哪条命令、会看到什么、什么条件下会被拒"这一步，以前只存在于文档散文里，
agent 只能凭印象拼命令 —— 而"凭印象拼子命令"正是触发评测里最常见的失败模式。

三条纪律（与 reverse-skill 的路由纪律同构，但我们用机器可读的形式表达）：

1. **先路由再执行**：`route` 只输出计划，不跑任何测量（要跑请照它给的命令）。
2. **匹配不上就说不匹配**，不许硬套（"If route not matched → do NOT force-fit"）。
3. **每条路由都带拒绝条件**：告诉你什么情况下这条路会给出 `n/a` 以及怎么办 ——
   避免 agent 拿到 `n/a` 之后开始瞎猜。

路由表是**代码**（`ROUTES` + `capabilities.QUESTIONS`），`--json` 是给 agent 的唯一真源；
文档只是它的视图。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .capabilities import QUESTIONS, Env, env_from_doctor, evaluate
from .refusals import DEFAULT_REMEDIES, RefusalCode

# 每条路由：匹配词 → 问题 id（问题本身在 capabilities.QUESTIONS 里，避免两处写"能答什么"）。
# 词表要覆盖**人怎么说**（中文口语 + ncu 术语 + 英文），而不只是我们的指标名。
ROUTES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("慢", "耗时", "时间", "duration", "latency", "哪个 kernel", "瓶颈在哪", "hot"), "kernel_time"),
    (("launch", "配置", "grid", "block", "几种", "变体"), "launch_configs"),
    (("寄存器", "registers", "共享内存", "shared", "smem", "限制因素", "limiter"), "static_resources"),
    (("占用率", "occupancy", "并行度"), "theoretical_occupancy"),
    (("显存忙", "device busy", "利用率", "utilization", "功耗", "能耗", "energy", "joule"),
     "device_busy"),
    (("成本", "花多少", "多少钱", "降本", "省钱", "每百万 token", "token 成本", "cost",
      "price", "token/s", "tokens per second", "单位成本"), "cost_per_token"),
    (("dram", "带宽", "字节", "bytes", "搬了多少", "流量", "memory bound", "访存"),
     "dram_bytes"),
    (("roofline", "多远", "上限", "算力", "tflops", "compute bound"), "roofline"),
    (("达到的占用率", "实际 occupancy", "achieved"), "achieved_occupancy"),
    (("轨迹", "trace", "sectors", "合并效率", "coalesc", "L2 命中", "命中率", "工作集"),
     "dram_bytes_trace"),
    (("发射槽", "issue slot", "指令混合", "latency bound", "throughput bound", "方向"),
     "issue_slot"),
    (("L2", "L1", "缓存", "cache", "locality", "复用"), "l2_locality"),
    (("sass", "指令", "instruction", "静态"), "static_mix"),
    (("ncu", "计数器", "counter", "权限", "permission", "拿不到", "为什么没有"), "exact_dram"),
    (("stall", "等待", "卡住", "warp 状态", "停顿"), "stall_breakdown"),
    (("pipe", "管线", "fp32", "整数单元", "tensor core"), "pipe_utilization"),
    (("nvlink", "多卡", "通信", "nccl"), "nvlink"),
)

# 每条路由**可能拿不到**的 code（显式列出：比按名字猜准，也便于测试校验 code 真的存在）。
FAILURE_HINTS: dict[str, tuple[str, ...]] = {
    "kernel_time": ("no_kernels", "no_launch_activity", "nsys_missing"),
    "launch_configs": ("mixed_launch_config", "no_kernels"),
    "static_resources": ("no_static_input", "cuobjdump_missing"),
    "theoretical_occupancy": ("unknown_arch", "no_static_input"),
    "device_busy": ("no_nvml", "no_nvml_samples", "no_gpu"),
    "cost_per_token": ("no_energy", "no_tokens", "no_price", "below_resolution"),
    "dram_bytes": ("share_too_low", "coverage_too_low", "below_resolution", "no_calibration"),
    "roofline": ("no_workload", "no_calibration", "calibration_caliber_mismatch"),
    "achieved_occupancy": ("no_occupancy_probe", "unknown_arch"),
    "dram_bytes_trace": ("no_memtrace", "nvbit_missing", "trace_unavailable",
                          "trace_too_many_accesses"),
    "issue_slot": ("no_instr_hist", "nvbit_missing", "no_clock_reading"),
    "l2_locality": ("no_memtrace", "sampling_unavailable"),
    "static_mix": ("no_sass_source", "no_static_input"),
    "exact_dram": ("counters_denied", "counters_not_installed", "counters_unknown"),
    "stall_breakdown": ("counters_denied", "instrumentation_missing", "sampling_unavailable"),
    "pipe_utilization": ("counters_denied", "instrumentation_missing"),
    "nvlink": ("no_nvlink", "counters_denied"),
}

# 每条路由的"命令模板"：用占位符，agent 自己替换。口径选择写在 note 里（哪条优先、为什么）。
COMMANDS: dict[str, str] = {
    "kernel_time": "kernoscope --set basic -k regex:<kernel> -c 20 -o rep -- <你的命令>",
    "launch_configs": "kernoscope --set basic -c 50 --launch-csv -o rep -- <你的命令>",
    "static_resources": "kernoscope --set basic --ks-cubin <kernel.cubin> -o rep（不需要 GPU）",
    "theoretical_occupancy": "kernoscope occupancy --from <occupancy-records.txt> --json",
    "device_busy": "kernoscope --set basic -o rep -- <你的命令>（只要有 NVML）",
    "cost_per_token": "kernoscope --set basic --ks-tokens <每步 token 数> "
                      "--ks-price-per-kwh <元/kWh> --ks-price-per-hour <元/小时> "
                      "-o rep -- <你的命令>（价格给哪个就算哪个载体）",
    "dram_bytes": "kernoscope --set basic --ks-calibration machine-profile.json "
                  "--ks-ledger-mb <每次 launch 的 MB> --ks-window auto -o rep -- <你的命令>",
    "roofline": "kernoscope --set basic --ks-gemm 4096,4096,4096 --ks-dtype bf16 "
                "--ks-calibration machine-profile.json -o rep -- <你的命令>",
    "achieved_occupancy": "kernoscope --set basic --ks-achieved-occupancy <occ.json> -o rep -- <你的命令>",
    "dram_bytes_trace": "kernoscope memtrace --from <trace.txt> --l2-bytes <L2字节> --json",
    "issue_slot": "kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> "
                  "--duration-ms <ms>",
    "l2_locality": "kernoscope --set basic --ks-memtrace <trace.txt> --l2-bytes <L2字节> -o rep -- <你的命令>",
    "static_mix": "kernoscope --set basic --ks-cubin <kernel.cubin> -o rep（不需要 GPU）",
    # 拿不到的那几个：命令的用途是**确认拿不到的原因**，不是硬凑一个替代
    "exact_dram": "kernoscope doctor --json（先确认是「没装 ncu」还是「被权限拒绝」——两者处置不同）",
    "stall_breakdown": "kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> "
                       "--duration-ms <ms>（只有上界+方向，stall 周期细分原理上拿不到）",
    "pipe_utilization": "kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> "
                        "--duration-ms <ms>（每 pipe 的**可计算下界**，不是计数器）",
    "nvlink": "kernoscope doctor --json（NVLink 计数器的可用性随驱动/权限变化）",
}

# 口径选择的**决策规则**（与 README 的口径选择表、SKILL frontmatter 同源，写在一处）
CALIBER_NOTE = {
    "cost_per_token": "能耗是**实测**（NVML 功率窗口积分）；token/价格由你给 —— 没给就拒绝给"
                      "那一半，不按典型值猜。电费与租约是两种载体，分别报、都给才相加。"
                      "按 kernel 的归因是时间占比代理（est__），窗口功率越均匀越准",
    "dram_bytes": "绝对字节：**有 NVBit 就用轨迹口径**（地址算术 ≈ −0.0001%）；"
                  "没有就走 NVML 代理（−4%~+13% 带内，小 kernel/常驻 L2 会被明确拒绝）",
    "dram_bytes_trace": "轨迹口径是绝对字节的**首选**；它要求 NVBit 外挂与足够磁盘（采集体积 ∝ 流量）",
    "achieved_occupancy": "区间重叠法只要 nvcc 编译的探针；没有探针时只能给理论上界",
    "issue_slot": "发射槽是**上界**口径 + 方向性判断，不是 ncu 的 stall 周期细分",
    "l2_locality": "L1/L2 命中率是**模型上界**，不是计数器读数",
}


def _match(question: str) -> list[tuple[str, int]]:
    """返回 [(问题 id, 命中词数)]，按命中数降序；不命中就是空列表（不许硬套）。"""
    q = question.lower()
    scored: list[tuple[str, int]] = []
    for words, qid in ROUTES:
        hit = sum(1 for w in words if w.lower() in q)
        if hit:
            scored.append((qid, hit))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def _refusals_for(qid: str) -> list[dict[str, Any]]:
    """这条路由可能拿不到的 code（显式表，带处置与代价）。

    "可能被拒"必须写进路由：否则 agent 拿到 `n/a` 之后会开始猜 —— 那正是我们要消灭的行为。
    """
    out: list[dict[str, Any]] = []
    for name in FAILURE_HINTS.get(qid, ()):
        try:
            code = RefusalCode(name)
        except ValueError:                       # 表里写错名字：宁可显式报出来
            out.append({"code": name, "action": "（route 表里这个 code 不存在，请修正）"})
            continue
        rem = DEFAULT_REMEDIES.get(code)
        out.append({"code": name, "action": rem.action if rem else "",
                    "cmd": rem.command if rem else "", "cost_kind": rem.cost_kind if rem else "",
                    "cost_s": rem.cost_s if rem else None})
    return out


def _env_note(env: Env) -> str:
    """这台机器上"首选口径"是否可用（只看能力，不测量）。"""
    if env.counter_state == "available":
        return "这台机器**有**计数器权限：优先直接读真值（本工具也能读 ncu 报告做对拍）"
    return ("这台机器**没有**计数器权限：走替代口径；"
            "证据都会带出处（exact/measured/estimated/unavailable）")


def route(question: str, *, env: Env | None = None, limit: int = 3) -> dict[str, Any]:
    """把一条问题路由成"口径 + 命令 + 期望 + 拒绝条件"。纯函数，便于测试。"""
    matches = _match(question)
    out: dict[str, Any] = {"question": question, "matched": False, "routes": []}
    if not matches:
        out["hint"] = ("没有匹配到已知问题 —— **不要硬套**。用 `kernoscope route --list` 看"
                       "全部可路由的问题，或 `kernoscope capabilities --json` 看这台机器能答什么。")
        return out
    out["matched"] = True
    for qid, score in matches[:limit]:
        q = next((x for x in QUESTIONS if x.id == qid), None)
        if q is None:
            continue
        out["routes"].append({
            "id": qid,
            "question": q.text,
            "tier": q.tier,
            "answers": list(q.outputs),
            "alt": q.alt,
            "command": COMMANDS.get(qid, ""),
            "caliber": CALIBER_NOTE.get(qid, ""),
            "failures": _refusals_for(qid),
            "score": score,
        })
    if env is not None:
        out["env_note"] = _env_note(env)
    return out


def render_text(r: dict[str, Any]) -> str:
    lines: list[str] = []
    if not r.get("matched"):
        lines.append(f"没有匹配到：{r['question']}")
        lines.append("  " + (r.get("hint") or ""))
        return "\n".join(lines)
    for i, route_item in enumerate(r["routes"], 1):
        lines.append(f"{i}. {route_item['question']}  （层级 {route_item['tier']}）")
        if route_item.get("caliber"):
            lines.append(f"   走哪条口径：{route_item['caliber']}")
        if route_item.get("command"):
            lines.append(f"   命令：{route_item['command']}")
        lines.append(f"   期望看到：{', '.join(route_item['answers'])}")
        if route_item.get("alt"):
            lines.append(f"   拿不到时改问：{route_item['alt']}")
        for f in route_item["failures"][:3]:
            lines.append(f"   可能被拒：{f['code']} → {f.get('action') or ''}"
                         + (f"  [{f['cmd']}]" if f.get("cmd") else ""))
    if r.get("env_note"):
        lines.append("")
        lines.append("这台机器：" + r["env_note"])
    return "\n".join(lines)


def build_parser() -> "argparse.ArgumentParser":
    ap = argparse.ArgumentParser(
        prog="kernoscope route",
        description="先路由再动手：把一条问题转成「走哪条口径 + 命令 + 期望输出 + 拒绝条件」。"
                    "只输出计划，不跑测量。")
    ap.add_argument("question", nargs="*", help="你的问题（中文口语也行）")
    ap.add_argument("--list", action="store_true", help="列出全部可路由的问题")
    ap.add_argument("--json", action="store_true", help="机器可读（agent 用这个）")
    ap.add_argument("--probe", action="store_true",
                    help="顺带探测本机能力（默认只做静态路由）")
    return ap


def route_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        rows = [{"id": q.id, "question": q.text, "tier": q.tier, "answers": list(q.outputs),
            "alt": q.alt,
                 "command": COMMANDS.get(q.id, "")} for q in QUESTIONS]
        if args.json:
            print(json.dumps({"routes": rows}, ensure_ascii=False, indent=2))
        else:
            for r in rows:
                print(f"  [{r['tier']}] {r['id']:24s} {r['question']}")
        return 0
    question = " ".join(args.question).strip()
    if not question:
        build_parser().print_help()
        return 2
    env = None
    if args.probe:
        from .doctor import run_doctor

        env = env_from_doctor(run_doctor(probe_counters=False))
    result = route(question, env=env)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    # 路由本身永远是成功：它回答"该怎么走"，不是"有没有做成某件事"
    return 0
