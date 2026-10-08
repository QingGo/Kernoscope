"""发射槽利用率的**上界**与瓶颈方向判断（对标 ncu 的 SchedulerStats / WarpStateStats）。

ncu 的 stall 细分来自硬件计数器（受限环境拿不到）。这里走另一条路：

    动态 warp 指令数（NVBit 插桩实测）÷ 这段时间里 SM 能提供的发射槽总数

* 分子：官方 `opcode_hist` 在 `count_warp_level=1` 下统计的就是**每 warp 每条指令执行一次**，
  正好等于消耗掉的发射槽数（读源码确认，见 backends/opcode_hist.py 的说明）。
* 分母：`SM 数 × 每 SM 的 warp 调度器数 × 时钟频率 × kernel 时长`。
  每 SM 有 4 个 warp 调度器、每周期各可发 1 条指令（Volta 及以后），所以这是**上限**：
  真实可获得的发射槽只会更少（还有 dual-issue 的历史包袱、指令类型限制等）。

由此得到的 `est__issue_slot_utilization_pct` 是一个**方向性**指标：
* 很高（≳ 60%）⇒ 发射受限，指令太碎/依赖链太短，优化方向是减少指令数（向量化、合并访存）
* 很低（≲ 20%）且显存吞吐高 ⇒ 显存受限（latency/bandwidth）
* 很低且显存吞吐也低 ⇒ 延迟受限（长依赖、访存延迟、同步），也就是"有 stall 但不在发射上"

**它不能替代 ncu 的 stall 计数器**：它不给 `smsp__warp_issue_stalled_*` 的细分，
只给"发射槽用了多少"这个上界 + 结合访存强度的方向判断。报告里一律标 `estimated`，
并把三个输入（指令数 / 时长 / 时钟）都写进 source，便于复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.opcode_hist import LaunchHistogram
from ..backends.cubin import CATEGORY
from ..refusals import Refusal, RefusalCode, combined_refusal

# 每 SM 的 warp 调度器数（Volta 及以后为 4；这是"每周期最多发几条 warp 指令"的上限）
SCHEDULERS_PER_SM = 4


@dataclass
class IssueStats:
    kernel: str = ""
    warp_instrs: int = 0
    launches: int = 0
    thread_blocks: int = 0
    duration_ns: float = 0.0
    sm_count: int = 0
    clock_mhz: float = 0.0
    schedulers_per_sm: int = SCHEDULERS_PER_SM
    opcode_counts: dict[str, int] = field(default_factory=dict)
    dram_throughput_pct: float | None = None
    notes: list[str] = field(default_factory=list)

    # ---- 可用性前提 ----
    @property
    def available(self) -> bool:
        return (self.warp_instrs > 0 and self.duration_ns > 0
                and self.sm_count > 0 and self.clock_mhz > 0)

    def unavailable_reason(self) -> str:
        return self.refusal().message

    def refusal(self) -> Refusal:
        """结构化拒绝（W3）：哪一条前提没成立 + 观测值 + 补救办法。

        发射槽上界需要四个量同时成立。缺时钟是最常见的一种（`--ks-no-nvml` 或
        NVML 不可用），它和"缺 SM 数"要分开报 —— 补救办法不同。
        """
        facts = {"warp_instrs": self.warp_instrs, "duration_ns": self.duration_ns,
                 "sm_count": self.sm_count, "clock_mhz": self.clock_mhz}
        out: list[Refusal] = []
        if self.warp_instrs <= 0:
            out.append(Refusal(RefusalCode.NO_INSTR_HIST,
                               "没有解析到动态指令数（opcode_hist 没产出记录？）",
                               observed=facts))
        if self.duration_ns <= 0:
            out.append(Refusal(RefusalCode.SHARE_UNKNOWN,
                               "缺少 kernel 时长：需要同一次运行的 gpu__time_duration.sum",
                               observed=facts, severity="degraded"))
        if self.sm_count <= 0:
            out.append(Refusal(RefusalCode.UNKNOWN_ARCH,
                               "缺少 SM 数：加 --ks-sm-count <N>", observed=facts))
        if self.clock_mhz <= 0:
            out.append(Refusal(RefusalCode.NO_CLOCK_READING,
                               "缺少 SM 时钟：加 --ks-clock-mhz <MHz>（或让 NVML 采样开着）",
                               observed=facts))
        if not out:
            raise ValueError("refusal() called on an available IssueStats")
        return combined_refusal(out)

    @property
    def issue_slots(self) -> float:
        """这段时间里全卡最多能提供的发射槽数（上限）。"""
        cycles = self.duration_ns / 1e9 * self.clock_mhz * 1e6
        return cycles * self.sm_count * self.schedulers_per_sm

    @property
    def issue_slot_utilization_pct(self) -> float | None:
        if not self.available:
            return None
        slots = self.issue_slots
        return 100.0 * self.warp_instrs / slots if slots else None

    @property
    def issued_ipc_per_sm(self) -> float | None:
        """每 SM 每周期实际发出的 warp 指令数（上界口径下的实测值）。"""
        if not self.available:
            return None
        cycles = self.duration_ns / 1e9 * self.clock_mhz * 1e6
        return self.warp_instrs / cycles / self.sm_count if cycles else None

    @property
    def warp_instrs_per_launch(self) -> float | None:
        if self.launches <= 0 or self.warp_instrs <= 0:
            return None
        return self.warp_instrs / self.launches

    # ---- 指令构成 ----
    @property
    def category_pct(self) -> dict[str, float]:
        if not self.opcode_counts:
            return {}
        total = sum(self.opcode_counts.values()) or 1
        cats: dict[str, int] = {}
        for op, n in self.opcode_counts.items():
            base = op.split(".")[0]
            for name, rx in CATEGORY:
                if rx.match(base):
                    cats[name] = cats.get(name, 0) + n
                    break
        return {k: 100.0 * v / total for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}

    @property
    def memory_instr_pct(self) -> float:
        return self.category_pct.get("memory", 0.0)

    # ---- 方向性判断 ----
    @property
    def stall_direction(self) -> str:
        """主要瓶颈方向（方向性，不是 ncu 的 stall 细分）。"""
        util = self.issue_slot_utilization_pct
        if util is None:
            return "无法判断（缺少动态指令数或时长/时钟）"
        mem = self.dram_throughput_pct
        if util >= 60.0:
            return ("偏发射受限：发射槽已用掉约 %.0f%%，优化方向是减少指令数"
                    "（向量化、减少地址运算、合并访存）" % util)
        if mem is not None and mem >= 50.0:
            return ("偏显存受限：发射槽只用掉约 %.0f%%，而显存忙碌已达 %.0f%% —— "
                    "指令在等数据，不是发不出去" % (util, mem))
        if util <= 20.0:
            return ("偏延迟受限：发射槽只用掉约 %.0f%%、显存也不忙 ⇒ 大部分时间在等"
                    "（长依赖链 / 访存延迟 / 同步），需要看 ILP、占用率与访存模式" % util)
        return "发射与等待混合：发射槽用掉约 %.0f%%，建议结合占用率与访存效率一起看" % util


def from_histogram(hist: LaunchHistogram, *, duration_ns: float, sm_count: int,
                   clock_mhz: float, dram_throughput_pct: float | None = None,
                   schedulers_per_sm: int = SCHEDULERS_PER_SM) -> IssueStats:
    st = IssueStats(
        kernel=hist.kernel, warp_instrs=hist.warp_instrs,
        launches=getattr(hist, "n_launches", 1) or 1,
        thread_blocks=hist.thread_blocks, duration_ns=duration_ns,
        sm_count=sm_count, clock_mhz=clock_mhz, opcode_counts=dict(hist.opcode_counts),
        dram_throughput_pct=dram_throughput_pct, schedulers_per_sm=schedulers_per_sm)
    if st.category_pct:
        mix = "、".join(f"{k} {v:.1f}%" for k, v in list(st.category_pct.items())[:4])
        st.notes.append(f"动态指令构成：{mix}")
    st.notes.append(
        "发射槽上限 = SM 数 × %d 调度器 × 时钟 × 时长；动态指令数按 **warp 级**计"
        "（NVBit count_warp_level=1），因此这是发射槽占用的直接计数" % schedulers_per_sm)
    return st


def to_metrics(st: IssueStats) -> list[tuple[str, float | None, str, str, str]]:
    """(name, value, unit, source, note) 供 section 组装（全部 estimated/derived）。"""
    src = (f"NVBit opcode_hist 动态指令数 × 发射槽上限（{st.sm_count} SM × "
           f"{st.schedulers_per_sm} 调度器 × {st.clock_mhz:.0f} MHz × "
           f"{st.duration_ns / 1e6:.3f} ms）")
    note = "；".join(st.notes)
    out: list[tuple[str, float | None, str, str, str]] = [
        ("est__dynamic_warp_instructions", float(st.warp_instrs) if st.warp_instrs else None,
         "", src, note),
        ("est__issue_slot_utilization_pct",
         round(st.issue_slot_utilization_pct, 1) if st.issue_slot_utilization_pct is not None
         else None, "%", src,
         "**上界口径**：每 SM 4 个调度器、每周期各发 1 条；真实可用发射槽只会更少"),
        ("est__issued_ipc_per_sm",
         round(st.issued_ipc_per_sm, 3) if st.issued_ipc_per_sm is not None else None,
         "warp-instr/SM/cycle", src, "同为上界口径"),
        ("est__stall_direction", st.stall_direction if st.available else None, "",
         "由发射槽利用率 + 显存忙碌 + 指令构成推出的方向性判断",
         "**不是** ncu 的 stall 计数器细分（smsp__warp_issue_stalled_*）："
         "这里只回答「到底发不出去还是在等数据」，且标 estimated"),
    ]
    if st.warp_instrs_per_launch is not None:
        out.append(("est__warp_instructions_per_launch",
                    round(st.warp_instrs_per_launch, 0), "", src,
                    f"共 {st.launches} 次 launch"))
    for cat, pct in st.category_pct.items():
        out.append((f"est__dyn_instr_share_{cat}", round(pct, 1), "%",
                    "同一份动态计数的指令类别构成（类别规则与 SASS 静态混合一致）",
                    "按 warp 级指令数加权"))
    return out
