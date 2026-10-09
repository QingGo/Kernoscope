"""能力探测与分层（W1）+ 机器可读能力清单（W5）。

这个模块回答的问题是：**这台机器上，我们能回答哪些问题、哪些问题必然被拒、代价多少。**

为什么值得单独一层：
  * 用户（尤其是 agent）在开始之前需要一个**正交的**判断："我该不该用这个工具、
    能用它问什么"。原先把这件事写在 README 的表格里，等于要求人手读文档。
  * 缺件必须优雅降级而不是退出：没有 nsys 不等于没有信息 —— NVML 设备级读数仍然可用。
    所以「探测 → 选最高可用层 → 给出下一步」是主流程的一部分，不是一个可选命令。

三态（不是二态）：`available`（能答，口径就绪）/ `degraded`（能答，但口径更弱或有前提）/
`refused`（不答，附 code + remedy + 代价）。二态会逼我们在"能答但弱"时撒谎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import __version__
from .refusals import Refusal, RefusalCode, Remedy

CAPABILITIES_SCHEMA_VERSION = 1
STATE_AVAILABLE = "available"
STATE_DEGRADED = "degraded"
STATE_REFUSED = "refused"
STATES = (STATE_AVAILABLE, STATE_DEGRADED, STATE_REFUSED)
_MARK = {STATE_AVAILABLE: "yes", STATE_DEGRADED: "weak", STATE_REFUSED: "no"}


@dataclass(frozen=True)
class Tier:
    """一个数据源层。层是可以叠加的：tier N 可用意味着 tier < N 也可用。"""

    level: int
    name: str
    needs: str
    gives: str


TIERS: tuple[Tier, ...] = (
    Tier(0, "demo", "无（不需要 GPU）",
         "用合成数据看输出形状与拒绝格式；任何数字都不是测量"),
    Tier(1, "device", "NVIDIA GPU + NVML（nvidia-smi / pynvml）",
         "设备级实测：显存忙碌占比、SM 时钟、功耗、能量；给出 DRAM 字节代理（需标定峰值）"),
    Tier(2, "trace", "nsys（CUPTI activity）",
         "kernel 级精确量：时长、launch 配置、寄存器/共享内存、occupancy 上界"),
    Tier(3, "instrumented", "NVBit 自行编译（+ nvcc）",
         "指令级：访存轨迹（sectors/request、L2 命中模型）、动态指令数、发射槽利用率上界"),
    Tier(4, "counters", "ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0）",
         "与 ncu 同源的硬件计数器：精确 DRAM 字节、各 pipe 利用率、stall 周期细分"),
)


@dataclass
class Env:
    """探测结果（全部是可被一次 `ls`/`nvidia-smi` 推翻的事实，不是猜测）。"""

    has_gpu: bool = False
    gpu_name: str = ""
    has_nvml: bool = False
    has_nsys: bool = False
    has_cuobjdump: bool = False
    has_ncu: bool = False
    counter_state: str = "unknown"          # available / denied / not-installed / not-in-path / unknown
    has_nvbit: bool = False
    has_calibration: bool = False
    has_workload: bool = False
    has_memtrace: bool = False
    has_instr_hist: bool = False
    has_occupancy_probe: bool = False
    has_cubin: bool = False
    nvml_disabled_by_user: bool = False

    @property
    def tier(self) -> int:
        """最高可用层 —— 由实际探测结果决定，不由用户选择。

        第 3 层的判据是**NVBit 装了**（工具链具备指令级能力），而不是"这次带了轨迹文件"：
        层回答的是"这台机器能做什么、要花多少代价"，所以「能，但要先花 10 分钟采一条轨迹」
        应该表现为 tier 3 + 对应问题 refused（带 NO_MEMTRACE 与代价），而不是 tier 2。
        """
        if self.counter_state == "available":
            return 4
        if self.has_gpu and self.has_nvbit:
            return 3
        if self.has_gpu and self.has_nsys:
            return 2
        if self.has_gpu and self.has_nvml and not self.nvml_disabled_by_user:
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class Question:
    """一个「我们要回答的问题」——能力清单的单位是问题，不是指标名。

    指标名是**我们**的内部词，问题是**用户**的词。对 agent 来说，可规划的单位也是问题
    （"我想知道是不是 DRAM bound"），所以清单按问题组织，并给出会产出的指标名。
    """

    id: str
    text: str
    tier: int
    outputs: tuple[str, ...]
    alt: str = ""            # 被拒时改问什么


QUESTIONS: tuple[Question, ...] = (
    Question("kernel_time", "哪个 kernel 最耗时、每次跑多久", 2,
             ("gpu__time_duration.sum", "ks__launch_time_mean")),
    Question("launch_configs", "同名 kernel 有几组 launch 配置，各自多快", 2,
             ("ks__launch_time_mean", "launch__grid_size", "launch__block_size")),
    Question("static_resources", "静态寄存器/共享内存，以及谁是 occupancy 限制因素", 2,
             ("ks__registers_used_for_occupancy", "ks__occupancy_limiter"),
             alt="用 --ks-cubin 做纯静态分析（不需要 GPU）"),
    Question("theoretical_occupancy", "occupancy 理论上界是多少", 2,
             ("ks__theoretical_occupancy_pct", "ks__blocks_per_sm")),
    Question("device_busy", "这段时间 GPU 显存子系统有多忙（实测）", 1,
             ("meas__nvml_mem_busy_mean_pct", "meas__device_energy_j")),
    Question("dram_bytes", "DRAM 搬了多少字节 / 是不是 DRAM bound（NVML 代理口径）", 1,
             ("est__dram_bytes_total", "ks__sampler_window_coverage_pct"),
             alt="轨迹口径 dram_bytes_trace（只测被追踪的 kernel）"),
    Question("roofline", "离 roofline 多远、compute 还是 memory bound", 1,
             ("est__achieved_gbs", "est__bound", "est__roofline_efficiency_pct"),
             alt="给 --ks-gemm 或 --ks-flops/--ks-bytes 才能算"),
    Question("achieved_occupancy", "实际达到的 occupancy（不是上界）", 3,
             ("meas__achieved_occupancy_pct",),
             alt="用探针区间重叠重建；没有探针时只给理论上界"),
    Question("dram_bytes_trace", "DRAM 字节的第二条口径（访存轨迹 + 缓存模型）", 3,
             ("est__dram_bytes_per_launch_trace", "meas__sectors_per_request"),
             alt="NVML 代理口径 dram_bytes"),
    Question("issue_slot", "发射槽用了多少、瓶颈方向是 latency 还是 throughput", 3,
             ("est__issue_slot_utilization_pct", "est__stall_direction"),
             alt="上界口径 + 方向性判断，不是 stall 周期细分"),
    Question("l2_locality", "L1/L2 命中率与工作集是否驻留", 3,
             ("est__l2_hit_rate_pct", "est__working_set_bytes")),
    Question("static_mix", "静态 SASS 指令混合（哪些指令占多数）", 2,
             ("ks__static_sass_instructions", "ks__static_sass_share_*"),
             alt="需要 --ks-cubin <file.so|.cubin>；不需要 GPU"),
    Question("stall_breakdown", "warp 到底在等什么（stall 周期细分）", 4,
             ("smsp__warp_issue_stalled_*_per_warp_active.pct",),
             alt="issue_slot 的方向性判断（latency vs throughput）"),
    Question("exact_dram", "DRAM 字节/L2 命中率的精确值（与 ncu 同源）", 4,
             ("dram__bytes.sum", "lts__t_sector_hit_rate.pct"),
             alt="dram_bytes（代理，带误差带）或 dram_bytes_trace（模型）"),
    Question("pipe_utilization", "各 pipe（FP32/INT/Tensor）的动态利用率", 4,
             ("sm__pipe_*.avg.pct_of_peak_sustained_elapsed",),
             alt="issue_slot 的上界口径 + roofline 的达成百分比"),
    Question("nvlink", "NVLink 的流量与利用率", 4,
             ("nvlink__*",), alt="单卡/无 NVLink 实例上这个问题不适用"),
)


@dataclass
class Answer:
    """一个问题的判定结果。"""

    question: Question
    state: str
    why: str = ""
    refusal: Refusal | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.question.id,
            "question": self.question.text,
            "state": self.state,
            "tier_required": self.question.tier,
            "outputs": list(self.question.outputs),
            "how": self.why,
        }
        if self.question.alt:
            d["alternative"] = self.question.alt
        if self.refusal is not None:
            d["refusal"] = self.refusal.to_dict()
        return d


@dataclass
class CapabilityReport:
    env: Env
    answers: list[Answer] = field(default_factory=list)
    probe_note: str = ""

    # ---- 汇总 ----
    @property
    def tier(self) -> int:
        return self.env.tier

    @property
    def tier_obj(self) -> Tier:
        return TIERS[self.tier]

    @property
    def next_tier(self) -> Tier | None:
        return TIERS[self.tier + 1] if self.tier + 1 < len(TIERS) else None

    def matrix(self) -> dict[str, str]:
        """三态覆盖矩阵：问题 id → available/degraded/refused。"""
        return {a.question.id: a.state for a in self.answers}

    def refusals(self) -> list[Refusal]:
        out: list[Refusal] = []
        for a in self.answers:
            if a.refusal is not None:
                out.append(a.refusal)
        return out

    def next_actions(self) -> list[dict[str, Any]]:
        """按「解锁的问题数 / 代价」排序的下一步建议（agent 可直接执行的清单）。

        同一个补救动作会解锁多个问题，所以按 (code, action) 合并 —— 否则清单里会出现
        五行一模一样的「装 nsys」，读者会以为要做五件事。
        """
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for a in self.answers:
            if a.state != STATE_REFUSED or a.refusal is None:
                continue
            rem = a.refusal.remedy_or_default
            if rem.cost_kind == "unavailable":
                continue
            key = (a.refusal.code.value, rem.action)
            g = groups.setdefault(key, {
                "code": a.refusal.code.value, "action": rem.action,
                "command": rem.command, "cost_kind": rem.cost_kind,
                "cost_s": rem.cost_s, "unlocks": []})
            g["unlocks"].append(a.question.id)
        def rank(g: dict[str, Any]) -> tuple[float, int, str]:
            cost = 1.0 if g["cost_kind"] == "manual" else max(float(g["cost_s"] or 0.0), 1.0)
            return (cost / max(len(g["unlocks"]), 1), -len(g["unlocks"]), g["code"])
        return sorted(groups.values(), key=rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITIES_SCHEMA_VERSION,
            "tool": "kernoscope",
            "tool_version": __version__,
            "tier": {"level": self.tier, "name": self.tier_obj.name,
                     "gives": self.tier_obj.gives},
            "next_tier": (None if self.next_tier is None else
                          {"level": self.next_tier.level, "name": self.next_tier.name,
                           "needs": self.next_tier.needs, "gives": self.next_tier.gives}),
            "env": self.env.to_dict(),
            "matrix": self.matrix(),
            "answers": [a.to_dict() for a in self.answers],
            "next_actions": self.next_actions(),
            "refusals": [r.to_dict() for r in self.refusals()],
            "probe_note": self.probe_note,
        }

    # ---- 摘要（给 agent 的**入口层**：只有 id 与计数，没有散文）----
    def brief(self, max_next: int = 4) -> dict[str, Any]:
        """渐进披露的第一层：**能答什么、不能答什么、下一步做什么**，不含任何解释。

        为什么需要它：完整的 `capabilities --json` 是一份**参考手册**（本机实测 25.8 KB，
        约 8k token）。把它当"入口"发给 agent，等于每轮都付一次全量上下文的钱。
        入口层只回答三个问题，且全部用 id / code / 计数表达：

            can_answer / weak / cannot_answer   问题 id 列表
            blocked_by                          缺哪些依赖（code）
            next                                按"解锁数/代价"排序的动作（code + 命令）

        要解释、要 remedy 全文、要 observed/threshold，都去 `--json`（参考层）取。
        """
        matrix = self.matrix()
        summary: dict[str, Any] = {
            "schema_version": CAPABILITIES_SCHEMA_VERSION,
            "tool": "kernoscope",
            "tool_version": __version__,
            "tier": {"level": self.tier, "name": self.tier_obj.name},
            "counter_state": self.env.counter_state,
            "can_answer": [a.question.id for a in self.answers
                           if a.state == STATE_AVAILABLE],
            "weak": [a.question.id for a in self.answers if a.state == STATE_DEGRADED],
            "cannot_answer": [a.question.id for a in self.answers
                              if a.state == STATE_REFUSED],
            "blocked_by": sorted({d["code"] for d in self.next_actions()}),
            "next": [{"code": d["code"], "unlocks": len(d["unlocks"]),
                      "cost_kind": d["cost_kind"], "cost_s": d["cost_s"],
                      "cmd": d["command"]}
                     for d in self.next_actions()[:max_next]],
        }
        if len(matrix) != len(self.answers):  # pragma: no cover - 防御性
            raise ValueError("matrix 与 answers 不一致")
        return summary

    # ---- 一屏（人视图；与 --json 同一模型）----
    def render(self) -> str:
        out: list[str] = []
        t = self.tier_obj
        out.append(f"Kernoscope {__version__} 能力自检：本机最高可用层 = {t.level} ({t.name})")
        out.append(f"  这一层能给：{t.gives}")
        if self.next_tier is not None:
            n = self.next_tier
            out.append(f"  再加 {n.level} 层（{n.name}）能多给：{n.gives}")
            out.append(f"    需要的依赖：{n.needs}")
        else:
            out.append("  已到最高层（与 ncu 同源的计数器可用）")
        out.append("")
        out.append("能用它回答的问题（yes=能答 / weak=能答但口径更弱 / no=不答）：")
        for a in self.answers:
            mark = _MARK[a.state]
            line = f"  [{mark:>4}] {a.question.id:<22} {a.question.text}"
            out.append(line)
            if a.state == STATE_REFUSED and a.refusal is not None:
                out.append(f"         no 的原因：{a.refusal.code.value} — "
                           f"{a.refusal.message[:110]}")
                rem = a.refusal.remedy_or_default
                if rem.cost_kind != "unavailable" and rem.action:
                    cost = ("需人工" if rem.cost_kind == "manual" else
                            (f"≈{rem.cost_s:.0f}s" if rem.cost_s else "≈0s"))
                    out.append(f"         怎么办：{rem.action}（代价 {cost}）")
        na = self.next_actions()
        if na:
            out.append("")
            out.append("下一步（按解锁的价值/代价排序）：")
            for d in na[:5]:
                cost = ("需人工" if d["cost_kind"] == "manual" else
                        (f"≈{d['cost_s']:.0f}s" if d["cost_s"] else "≈0s"))
                out.append(f"  - 解锁 {len(d['unlocks'])} 个问题"
                           f"（{', '.join(d['unlocks'][:4])}"
                           + ("…" if len(d["unlocks"]) > 4 else "") + f"），代价 {cost}：{d['action']}"
                           + (f"\n      {d['command']}" if d["command"] else ""))
        if self.probe_note:
            out.append("")
            out.append(f"注：{self.probe_note}")
        return "\n".join(out)


# --------------------------------------------------------------------------
# 判定：每个问题一个纯函数（env → (state, why, refusal)）
# --------------------------------------------------------------------------
def _need(env: Env, *, gpu: bool = False, nvml: bool = False, nsys: bool = False,
          nvbit: bool = False, counters: bool = False, calib: bool = False,
          workload: bool = False, cubin: bool = False, probe: bool = False,
          memtrace: bool = False, instr_hist: bool = False,
          question: str = "") -> tuple[str, str, Refusal | None]:
    """缺件 → refused（带 code + remedy）。`calib/workload/probe/...` 缺时为 degraded。"""
    if gpu and not env.has_gpu:
        return STATE_REFUSED, "本机没有 NVIDIA GPU", Refusal(
            RefusalCode.NO_GPU, "本机没有 NVIDIA GPU：只能跑合成数据看输出形状",
            observed={"gpu": env.gpu_name})
    if nvml and (not env.has_nvml or env.nvml_disabled_by_user):
        code = RefusalCode.NO_NVML_SAMPLES if env.nvml_disabled_by_user else RefusalCode.NO_NVML
        msg = ("采样被 --ks-no-nvml 关掉了" if env.nvml_disabled_by_user
               else "nvidia-smi / pynvml 不可用")
        return STATE_REFUSED, msg, Refusal(code, msg)
    if nsys and not env.has_nsys:
        return STATE_REFUSED, "缺少 nsys（CUPTI activity 的唯一来源）", Refusal(
            RefusalCode.NSYS_MISSING,
            "缺少 nsys：kernel 级精确量（时长/launch 配置）拿不到；"
            "设备级读数仍然可用")
    if nvbit and not env.has_nvbit:
        return STATE_REFUSED, "未配置 NVBit（指令级/访存轨迹需要自行编译）", Refusal(
            RefusalCode.NVBIT_MISSING,
            "未配置 NVBit：指令级与访存轨迹口径不可用（NVBit 许可证禁止随本项目分发）")
    if memtrace and not env.has_memtrace:
        return STATE_REFUSED, "没有提供 --ks-memtrace 轨迹文件", Refusal(
            RefusalCode.NO_MEMTRACE, "没有提供轨迹文件：加 --ks-memtrace <trace> 换轨迹口径")
    if instr_hist and not env.has_instr_hist:
        return STATE_REFUSED, "没有提供 --ks-instr-hist 直方图", Refusal(
            RefusalCode.NO_INSTR_HIST, "没有提供指令直方图：加 --ks-instr-hist <file>")
    if probe and not env.has_occupancy_probe:
        return STATE_REFUSED, "没跑 occupancy 探针", Refusal(
            RefusalCode.NO_OCCUPANCY_PROBE,
            "没跑 occupancy 探针：实际达到的 occupancy 需要 (smid, globaltimer) 探针")
    if counters and env.counter_state != "available":
        code = {"denied": RefusalCode.NO_COUNTERS,
                "not-installed": RefusalCode.COUNTERS_NOT_INSTALLED}.get(
                    env.counter_state, RefusalCode.COUNTERS_UNKNOWN)
        return STATE_REFUSED, f"硬件计数器状态：{env.counter_state}", Refusal(
            code, f"硬件计数器不可用（状态 {env.counter_state}）："
                  f"这一项与 ncu 同源，本工具不给替代数字")
    if cubin and not env.has_cubin:
        return STATE_REFUSED, "没有提供 cubin/.so", Refusal(
            RefusalCode.NO_STATIC_INPUT, "没有提供 cubin/.so：加 --ks-cubin <file>")
    # 以下不是"缺件"，而是"能答但更弱"
    if calib and not env.has_calibration:
        return STATE_DEGRADED, "没有本机标定（峰值来自规格或用户覆盖值时口径更弱）", None
    if workload and not env.has_workload:
        return STATE_DEGRADED, "没有给工作量（FLOPs/bytes）：只能给带宽与时长侧", None
    return STATE_AVAILABLE, "", None


Checks = dict[str, Callable[[Env], tuple[str, str, Refusal | None]]]


def _q_kernel_time(env: Env):
    return _need(env, gpu=True, nsys=True)


def _q_launch_configs(env: Env):
    return _need(env, gpu=True, nsys=True)


def _q_static_resources(env: Env):
    # 纯静态也能给（有 cubin 时不需要 nsys）；但限制因素判定需要架构表 + 实际配置
    if env.has_cuobjdump and env.has_cubin:
        return STATE_AVAILABLE, "用 cuobjdump 静态读出（不需要 GPU）", None
    return _need(env, gpu=True, nsys=True)


def _q_theoretical_occupancy(env: Env):
    return _need(env, gpu=True, nsys=True)


def _q_device_busy(env: Env):
    return _need(env, gpu=True, nvml=True)


def _q_dram_bytes(env: Env):
    state, why, ref = _need(env, gpu=True, nvml=True)
    if state != STATE_AVAILABLE:
        return state, why, ref
    if not env.has_calibration:
        return STATE_DEGRADED, ("没有本机标定：代理口径缺少实测峰值，"
                                "roofline/DRAM 字节会拒绝给数而不是用规格表顶替"), None
    return STATE_AVAILABLE, "", None


def _q_roofline(env: Env):
    state, why, ref = _need(env, gpu=True, nvml=True, workload=True)
    return state, why, ref


def _q_achieved_occupancy(env: Env):
    return _need(env, gpu=True, nvbit=True, probe=True)


def _q_dram_trace(env: Env):
    return _need(env, gpu=True, nvbit=True, memtrace=True)


def _q_issue_slot(env: Env):
    return _need(env, gpu=True, nvbit=True, instr_hist=True)


def _q_l2_locality(env: Env):
    return _need(env, gpu=True, nvbit=True, memtrace=True)


def _q_static_mix(env: Env):
    if env.has_cuobjdump and env.has_cubin:
        return STATE_AVAILABLE, "cuobjdump -sass（静态，不需要 GPU）", None
    return STATE_REFUSED, "没有 cubin/.so（SASS 混合的来源）", Refusal(
        RefusalCode.NO_STATIC_INPUT, "没有 cubin/.so：加 --ks-cubin <file> 才能给静态指令混合")


def _q_stall_breakdown(env: Env):
    return _need(env, counters=True)


def _q_exact_dram(env: Env):
    return _need(env, counters=True)


def _q_pipe_utilization(env: Env):
    return _need(env, counters=True)


def _q_nvlink(env: Env):
    return STATE_REFUSED, "需要 NVLink 硬件与 NVML 支持", Refusal(
        RefusalCode.NO_NVLINK, "NVLink 计数不可得（单卡实例上这个问题本来就不适用）")


CHECKS: Checks = {
    "kernel_time": _q_kernel_time,
    "launch_configs": _q_launch_configs,
    "static_resources": _q_static_resources,
    "theoretical_occupancy": _q_theoretical_occupancy,
    "device_busy": _q_device_busy,
    "dram_bytes": _q_dram_bytes,
    "roofline": _q_roofline,
    "achieved_occupancy": _q_achieved_occupancy,
    "dram_bytes_trace": _q_dram_trace,
    "issue_slot": _q_issue_slot,
    "l2_locality": _q_l2_locality,
    "static_mix": _q_static_mix,
    "stall_breakdown": _q_stall_breakdown,
    "exact_dram": _q_exact_dram,
    "pipe_utilization": _q_pipe_utilization,
    "nvlink": _q_nvlink,
}


def evaluate(env: Env, note: str = "") -> CapabilityReport:
    """纯函数：由环境事实得到能力清单（可测试，不需要真的 GPU）。"""
    answers: list[Answer] = []
    for q in QUESTIONS:
        fn = CHECKS.get(q.id)
        if fn is None:  # pragma: no cover - 新问题忘了登记判定函数时立刻炸
            raise KeyError(f"question {q.id!r} has no check function")
        state, why, ref = fn(env)
        if state not in STATES:  # pragma: no cover
            raise ValueError(f"bad state {state!r} for {q.id}")
        answers.append(Answer(q, state, why, ref))
    return CapabilityReport(env=env, answers=answers, probe_note=note)


# --------------------------------------------------------------------------
# 从 doctor 的自检结果构造 Env（复用同一份探测，不重复造轮子）
# --------------------------------------------------------------------------
def env_from_doctor(report: Any, **overrides: Any) -> Env:
    """把 DoctorReport 翻译成能力判定需要的环境事实。

    只读 doctor 已经探明的事实（tools / device / counter_state），不再调用任何外部命令。
    """
    tools = getattr(report, "tools", {}) or {}

    def found(name: str) -> bool:
        return bool((tools.get(name) or {}).get("found"))

    device = getattr(report, "device", {}) or {}
    env = Env(
        has_gpu=bool(device.get("name")),
        gpu_name=str(device.get("name") or ""),
        has_nvml=found("nvidia-smi"),
        has_nsys=found("nsys"),
        has_cuobjdump=found("cuobjdump"),
        has_ncu=found("ncu"),
        counter_state=str(getattr(report, "counter_state", "unknown")),
        has_nvbit=bool(getattr(report, "nvbit_path", "")),
    )
    for k, v in overrides.items():
        if not hasattr(env, k):  # pragma: no cover
            raise KeyError(f"unknown Env field {k!r}")
        setattr(env, k, v)
    return env


def probe(probe_counters: bool = False, **overrides: Any) -> CapabilityReport:
    """跑一次自检并给出能力清单（`kernoscope capabilities` 与主流程降级共用）。"""
    from .doctor import run_doctor

    doc = run_doctor(probe_counters=probe_counters)
    note = ("本次未做计数器实测探测（静态推断）：要实测结论请加 --probe"
            if not probe_counters else "")
    return evaluate(env_from_doctor(doc, **overrides), note=note)


def render_matrix(report: CapabilityReport) -> str:
    """覆盖矩阵的表格形式（文档与首屏共用）。"""
    lines = ["| 问题 | 三态 | 需要的最低层 | 会产出的指标 |", "|---|---|---|---|"]
    for a in report.answers:
        lines.append(f"| `{a.question.id}` — {a.question.text} | {_MARK[a.state]} | "
                     f"{a.question.tier} | `{'`, `'.join(a.question.outputs)}` |")
    return "\n".join(lines)


__all__ = ["Answer", "CAPABILITIES_SCHEMA_VERSION", "CapabilityReport", "Env",
           "Question", "QUESTIONS", "TIERS", "Tier", "env_from_doctor", "evaluate",
           "probe", "render_matrix", "STATES"]
