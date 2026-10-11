"""真值对拍（I1）：拿一份真的 `.ncu-rep` 来校准我们的替代口径。

为什么这是可信度的来源：我们的每个替代值都带出处，但"代理口径准不准"这件事**只能**靠
跟真值比。所以当用户手上真有一份 `.ncu-rep`（在别的机器上采的、或这台机器其实有权限）时，
我们要做的不只是把真值读出来，而是**逐条对拍**：

    真值（ncu__dram__bytes.sum，exact）  vs  我们的替代值（est__dram_bytes_per_launch）
    → ks__truth_vs_substitute_pct（相对偏差，derived）

对拍表是**代码**（`PAIRS`），每条都写明口径为什么可比、以及在什么前提下可比。
不可比的一律不给数：宁可说"这条对不上"，也不给一个把两种口径混在一起的百分比。

一条重要的纪律：`ncu__*` 前缀 + `exact` 表示"这个数与 ncu 同源"，而不是"这个数由我们测得"。
所以它**不是**计数器命名空间（不触发 R2 的 allowlist），但读的人一眼能看出它来自导入的报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..backends.ncu_rep import NCU_METRIC_PREFIX, NcuRepAction, NcuRepData

# 对拍表：(真值原名, 我们的指标名, 换算系数, 口径说明)
#   * 换算系数把真值换成我们的单位（真值 × 系数 = 我们的单位）；
#   * 口径说明里必须写清"为什么可比"以及"什么情况下不可比"。
# 只有**同一台机器、同一次负载**下的对拍才有意义 —— 这一点写在报告里（前提），
# 因为把 A 机器的真值和 B 机器的替代值相比，得到的百分比什么也不说明。
PAIRS: tuple[tuple[str, str, float, str], ...] = (
    ("gpu__time_duration.sum", "ks__launch_time_mean", 1e-6,
     "每次 launch 时长：ncu 的单位是 ns，我们的是 ms 均值；只在 ncu 采集了 1 次 launch"
     "（默认 -c 1）且我们匹配到单一 kernel 配置时可比"),
    ("launch__registers_per_thread", "launch__registers_per_thread", 1.0,
     "每线程寄存器数：两边同源（cubin/LaunchStats），应当逐位相同 —— 这一行是**解析路径的"
     "自检**，对不上说明我们的解析或导入有问题，而不是机器有问题"),
    ("launch__shared_mem_per_block_static", "launch__shared_mem_per_block_static", 1.0,
     "静态共享内存：同上，自检行"),
    ("dram__bytes.sum", "est__dram_bytes_per_launch", 1.0,
     "每次 launch 的 DRAM 字节：真值 vs 我们的代理口径（NVML 显存忙碌 × 实测峰值）。"
     "这是**代理精度**的证据；ncu 采了多次 launch 时真值是总量，需自行折算"),
    ("lts__t_sector_hit_rate.pct", "est__l2_hit_rate_pct", 1.0,
     "L2 命中率：真值是计数器，我们是模型上界 —— 只应作为上界比较，我们给的值不该高于真值"),
    ("dram__bytes_read.sum", "est__dram_bytes_read_per_launch", 1.0,
     "读字节：同上，真值 vs 代理"),
)

# 进「真值 section」的头部指标（不是对拍，是给人看的 headline）
HEADLINE_TRUTH_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)


@dataclass
class TruthRow:
    """一条对拍结果。`rel_err_pct` 为 None 表示前提不成立（不给数）。"""

    truth_name: str
    ours_name: str
    truth: float | None
    ours: float | None
    rel_err_pct: float | None
    basis: str
    reason: str = ""            # 非空 -> 不给数（宁可说对不上）

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"truth": self.truth_name, "ours": self.ours_name,
                             "basis": self.basis}
        if self.rel_err_pct is None:
            d["reason"] = self.reason or "不可比"
        else:
            d["truth_value"] = self.truth
            d["ours_value"] = self.ours
            d["rel_err_pct"] = round(self.rel_err_pct, 4)
        return d


@dataclass
class TruthComparison:
    path: str
    tool_version: str
    action: NcuRepAction | None = None
    rows: list[TruthRow] = field(default_factory=list)
    matched: dict[str, float] = field(default_factory=dict)   # 头部真值
    rules: list[tuple[str, float]] = field(default_factory=list)
    actions_total: int = 0
    metrics_total: int = 0
    skipped_total: int = 0

    @property
    def compared(self) -> list[TruthRow]:
        return [r for r in self.rows if r.rel_err_pct is not None]

    @property
    def max_abs_err_pct(self) -> float | None:
        vals = [abs(r.rel_err_pct) for r in self.compared if r.rel_err_pct is not None]
        return max(vals) if vals else None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "ncu_version": self.tool_version,
                "actions": self.actions_total, "metrics": self.metrics_total,
                "skipped_metrics": self.skipped_total,
                "kernel": self.action.kernel_name if self.action else "",
                "grid": list(self.action.grid) if self.action else [],
                "block": list(self.action.block) if self.action else [],
                "headline": self.matched, "rules": self.rules,
                "pairs": [r.to_dict() for r in self.rows]}


def compare(data: NcuRepData, metrics: dict[str, float], *, action: NcuRepAction | None = None,
            ours_by_name: dict[str, float] | None = None) -> TruthComparison:
    """把一份真值与我们的指标值逐条对拍。

    `ours_by_name` 是"我们报告里有的值"（名字 → 值）。**不猜、不换算我们不知道的口径**：
    任何一边缺值就写进 `reason`，不给百分比。
    """
    ours = dict(ours_by_name or {})
    act = action
    if act is None and data.actions:
        act = data.actions[0]
    cmp = TruthComparison(path=data.path, tool_version=data.tool_version, action=act,
                          rules=list(act.rules) if act else [],
                          actions_total=len(data.actions),
                          metrics_total=len(act.metrics) if act else 0,
                          skipped_total=act.skipped_metrics if act else 0)
    if act is None:
        return cmp
    for raw in HEADLINE_TRUTH_METRICS:
        if raw in act.metrics:
            cmp.matched[NCU_METRIC_PREFIX + raw] = act.metrics[raw]
    for truth_name, ours_name, factor, basis in PAIRS:
        truth = act.metrics.get(truth_name)
        ours_val = ours.get(ours_name)
        if truth is None:
            cmp.rows.append(TruthRow(truth_name, ours_name, None, ours_val, None, basis,
                                     reason="这份 ncu 报告里没有这个指标（不同架构/采集配置"
                                            "采集到的指标集不同）"))
            continue
        if ours_val is None:
            cmp.rows.append(TruthRow(truth_name, ours_name, truth, None, None, basis,
                                     reason=f"我们的报告里没有 {ours_name}"
                                            "（缺输入或前提不成立）"))
            continue
        truth_conv = truth * factor
        if truth_conv == 0:
            cmp.rows.append(TruthRow(truth_name, ours_name, truth_conv, ours_val, None, basis,
                                     reason="真值为 0：相对偏差无定义（除以 0），不给百分比"))
            continue
        rel = 100.0 * (ours_val - truth_conv) / truth_conv
        cmp.rows.append(TruthRow(truth_name, ours_name, truth_conv, ours_val, rel, basis))
    return cmp


def comparison_metrics(cmp: TruthComparison) -> list[tuple[str, float | None, str, str]]:
    """把对拍结果变成指标行：`(name, value, unit, source/note)`，None = 不给数。

    命名纪律：真值用 `ncu__`（exact，与 ncu 同源），我们的值沿用原名，
    偏差用 `ks__truth_vs_substitute_pct`（derived）。
    """
    out: list[tuple[str, float | None, str, str]] = []
    for row in cmp.rows:
        out.append((f"ks__truth_vs_substitute_pct[{row.ours_name}]",
                    round(row.rel_err_pct, 4) if row.rel_err_pct is not None else None,
                    "%", f"({row.truth_name} × 系数) vs {row.ours_name}：{row.basis}"
                    + (f"；不给数：{row.reason}" if row.reason else "")))
    return out


def truth_metrics(data: NcuRepData, action: NcuRepAction | None = None, *,
                  names: Sequence[str] = HEADLINE_TRUTH_METRICS
                  ) -> list[tuple[str, float, str]]:
    """头部真值指标 → `(ncu__name, value, source)`。只在报告里出现过的才出。"""
    act = action or (data.actions[0] if data.actions else None)
    if act is None:
        return []
    source = f"ncu-rep {data.path}（导入的真值，与 ncu 同源）"
    if data.tool_version:
        source += f" ncu {data.tool_version}"
    return [(NCU_METRIC_PREFIX + n, act.metrics[n], source) for n in names if n in act.metrics]


def match_action(data: NcuRepData, kernel_name: str) -> NcuRepAction | None:
    """在我们的 kernel 名与报告里的 action 之间做**保守**匹配（对不上就不匹配）。

    只做两条判定：完全相等，或"我们的 demangled 名以 action 的 function 名开头/相反包含"。
    不用 token 交集 —— 那会把 `addConstDouble` 和 `addConstDouble3` 混起来，
    而对拍表里出现另一个 kernel 的数就意味着那个百分比是假的。
    """
    if not kernel_name:
        return None
    for a in data.actions:
        if a.kernel_name == kernel_name or a.demangled_name == kernel_name:
            return a
    simple = kernel_name.split("(")[0].strip()
    for a in data.actions:
        if a.function_name and a.function_name == simple:
            return a
    for a in data.actions:
        if a.demangled_name and (a.demangled_name == kernel_name
                                 or a.kernel_name == kernel_name):
            return a
    return None
