#!/usr/bin/env python3
"""从代码生成 COVERAGE.md（仓库根）—— 三态覆盖矩阵 + 四层能力分层 + 必然被拒清单。

单一真源是 `src/kernoscope/capabilities.py`：问题清单（QUESTIONS）、能力层（TIERS）、
判定函数（CHECKS）与拒绝码都在那里。本脚本**不手抄**任何问题 id 或指标名——
改了代码就必须重跑，否则 `tests/test_opening_package.py` 会红：

    uv run python tools/gen_coverage.py          # 重写 COVERAGE.md
    uv run python tools/gen_coverage.py --check   # 只校验是否与代码同源（CI 可用）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kernoscope.capabilities import (  # noqa: E402
    QUESTIONS,
    TIERS,
    Env,
    evaluate,
    render_matrix,
)

OUT = ROOT / "COVERAGE.md"

# 三态 → 人话（与 capabilities 的 yes/weak/no 同一套语义）
MARK = {"available": "能答", "degraded": "更弱", "refused": "不答"}


def _envs() -> list[tuple[int, str, str, Env]]:
    """五个代表环境：每层一个，只给"刚好够到该层"的事实（层由探测结果决定）。

    返回 (层号, 这一层的样子, 还缺什么, Env)。这些环境是**代表**，不是某台具体机器：
    真机上的三态请用 `kernoscope capabilities --json` 现场探测。
    """
    gpu = {"has_gpu": True, "gpu_name": "NVIDIA GPU（示例）"}
    base = dict(gpu, has_nvml=True)
    # 层 2 起假设 CUDA Toolkit 与 --ks-cubin 都在（静态资源/SASS 混合是纯静态的，不需要 GPU）
    static = dict(has_cuobjdump=True, has_cubin=True)
    return [
        (0, "无 GPU：只能跑合成数据", "任何 NVIDIA 设备", Env()),
        (1, "GPU + NVML（设备级读数）", "nsys（kernel 级）、本机标定", Env(**base)),
        (2, "再加 nsys + 本机标定 + 工作量 + cuobjdump/cubin", "NVBit（指令级/访存轨迹）",
         Env(**base, **static, has_nsys=True, has_calibration=True, has_workload=True)),
        (3, "再加 NVBit（轨迹/直方图/探针），计数器仍被拒", "计数器权限（层 4）",
         Env(**base, **static, has_nsys=True, has_calibration=True, has_workload=True,
             has_nvbit=True, has_memtrace=True, has_instr_hist=True,
             has_occupancy_probe=True, counter_state="denied")),
        (4, "计数器可用（与 ncu 同源）", "——（已到最高层）",
         Env(**base, **static, has_nsys=True, has_calibration=True, has_workload=True,
             has_nvbit=True, has_memtrace=True, has_instr_hist=True,
             has_occupancy_probe=True, counter_state="available")),
    ]


def render() -> str:
    envs = _envs()
    reports = [(lvl, looks, missing, evaluate(env)) for lvl, looks, missing, env in envs]

    out: list[str] = []
    out.append("# 能力覆盖（三态）与能力分层")
    out.append("")
    out.append("本文件由 `kernoscope capabilities` 生成（脚本 `tools/gen_coverage.py` 调用")
    out.append("`kernoscope.capabilities`）；**改动请改 "
               "`src/kernoscope/capabilities.py` 后重跑，不要手改本文件**。")
    out.append("")
    out.append("要看你**这台机器**的实际状态，请现场跑 `kernoscope capabilities --json`"
               "（机器可读）或 `kernoscope capabilities`（一屏）。")
    out.append("")
    out.append(f"<!-- question ids（与 kernoscope.capabilities.QUESTIONS 同源）: "
               f"{', '.join(q.id for q in QUESTIONS)} -->")
    out.append("")
    out.append("## 三态怎么读")
    out.append("")
    out.append("| 标记 | 含义 |")
    out.append("|---|---|")
    out.append("| 能答 | 口径就绪：这一层的数据源与前提都在，数字可以引用 |")
    out.append("| 更弱 | 能答，但口径更弱或有前提（例如缺本机标定、缺工作量），"
               "引用时必须同时引用前提 |")
    out.append("| 不答 | **不给数字**，只报 `n/a` + 结构化拒绝（code + 原因 + 替代口径） |")
    out.append("")
    out.append("二态会逼我们在「能答但弱」时撒谎，所以这里是三态。")
    out.append("")

    out.append("## 四层能力分层")
    out.append("")
    out.append("| 层 | 名字 | 需要的依赖 | 这一层能给 |")
    out.append("|---|---|---|---|")
    for t in TIERS:
        out.append(f"| {t.level} | `{t.name}` | {t.needs} | {t.gives} |")
    out.append("")
    out.append("层是可以叠加的：层 N 可用意味着层 < N 也可用。最高可用层由**探测结果**决定，"
               "不由用户选择；缺件只会降级，不会假装可用。")
    out.append("")

    out.append("## 三态覆盖矩阵（问题 × 能力层）")
    out.append("")
    out.append("| " + " | ".join(["问题 id", "问的是什么"]
                                + [f"层 {lvl}" for lvl, *_ in reports]) + " |")
    out.append("|" + "---|" * (2 + len(reports)))
    for q in QUESTIONS:
        cells = []
        for _lvl, _looks, _missing, rep in reports:
            by_id = {a.question.id: a.state for a in rep.answers}
            cells.append(MARK[by_id[q.id]])
        out.append(f"| `{q.id}` | {q.text} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("五个代表环境（只给「刚好够到该层」的事实，不是某台具体机器）：")
    out.append("")
    out.append("| 层 | 这一层的样子 | 还缺什么 |")
    out.append("|---|---|---|")
    for lvl, looks, missing, _rep in reports:
        out.append(f"| {lvl} | {looks} | {missing} |")
    out.append("")

    tier3 = reports[3][3]
    out.append("## 必然被拒的问题（层 3：非计数器手段都装齐、但没有计数器权限）")
    out.append("")
    out.append("这一层是**受限多租户实例的典型形态**：`nsys` + NVBit 都可用，"
               "但宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON`。"
               "下面这些问题在这一层**仍然必然被拒** —— 与 ncu 同源的硬件计数器我们不给替代数字。")
    out.append("")
    out.append("| 问题 id | 问的是什么 | 拒绝码 | 原因 | 替代口径 |")
    out.append("|---|---|---|---|---|")
    for a in tier3.answers:
        if a.state != "refused" or a.refusal is None:
            continue
        alt = a.question.alt or a.refusal.remedy_or_default.action
        why = a.refusal.message.replace("|", "\\|")
        out.append(f"| `{a.question.id}` | {a.question.text} | `{a.refusal.code.value}` | "
                   f"{why} | {alt} |")
    out.append("")
    out.append("层 0–2 被拒的更多 —— 那些是**环境缺件**（没有 GPU / 没有 nsys / 没给输入），"
               "每一行都带可执行的补救动作与代价，见 `kernoscope capabilities` 的"
               "「下一步」清单。层 4（计数器可用）会解锁上表里的计数器问题。")
    out.append("")
    out.append("## 层 3 的逐问读数（含会产出的指标）")
    out.append("")
    out.append(render_matrix(tier3))
    out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur == text:
            print("COVERAGE.md 与代码同源 ✓")
            return 0
        print("COVERAGE.md 与代码不一致：请重跑 `uv run python tools/gen_coverage.py`")
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
