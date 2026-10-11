"""副作用账（A2）—— 动手之前，先说清"会发生什么"。

问题：`kernoscope ... -- python serve.py` 这类跑一次 profiler 的命令，副作用并不平凡 ——
它会**跑你的命令**、可能启动 `nsys`/`cuobjdump`、在临时目录写几百 MB 中间产物、
在你的工作目录写报告与案件目录、读你的标定/轨迹文件。对人和 agent 都一样：
"先给我看会发生什么，再决定让不让它发生"。

做法（借用 reverse-skill 的 effects 审计模式，但落成我们自己的形状）：

  1. **每个操作（operation）必须有一条 effects 声明**，缺了就直接抛 —— 不是"文档里提一句"。
  2. MCP 风格的四个 annotation（`read_only_hint` / `destructive_hint` / `idempotent_hint` /
     `open_world_hint`）**从 effects 纯推导**，禁止按命令名猜。
  3. `--plan` 只**渲染计划、不执行任何东西**，打印每条操作的 target / 命令 / 会写的文件 /
     网络源 / 是否第三方软件，最后才是"要不要跑由你决定"。
  4. 声明与观察必须能对账：测试用 PATH 上的记录桩把**实际启动的子进程**记下来，
     与计划声明的对照（见 `tests/test_effects.py::test_plan_matches_observed_launches`）。

两条纪律写在类型里而不是散文里：
  * **静态分析不执行目标**（`static_dump` 的 detail 明写），也不把它当成"观察到了运行行为"；
  * 我们**不注入**任何东西到被测进程：NVBit 插桩由 `tools/probes/nvbit_memtrace/collect.sh`
    在你调用它时执行，本工具只读它产出的轨迹文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class ToolEffects:
    """一个操作对世界的八项影响。默认全 false / 幂等 —— 少写一项比多写一项危险。"""

    mutates_target: bool = False        # 改变了被测对象（文件/进程状态）
    mutates_session: bool = False       # 改变了本次会话状态（临时产物、后台线程）
    writes_filesystem: bool = False
    launches_process: bool = False
    accesses_network: bool = False
    changes_ui_state: bool = False
    may_discard_data: bool = False
    idempotent: bool = True

    @property
    def read_only(self) -> bool:
        """REA 的判据：只要有一项为真就不是 read-only（"只记了一条证据"也算）。"""
        return not any((self.mutates_target, self.mutates_session, self.writes_filesystem,
                        self.launches_process, self.accesses_network, self.changes_ui_state,
                        self.may_discard_data))

    def annotations(self) -> dict[str, bool]:
        """四个 MCP annotation，**纯推导**（不许按 tool 名/命令族硬编码）。"""
        return {
            "read_only_hint": self.read_only,
            "destructive_hint": self.may_discard_data,
            "idempotent_hint": self.idempotent,
            "open_world_hint": self.launches_process or self.accesses_network,
        }

    def to_dict(self) -> dict[str, bool]:
        return {"mutates_target": self.mutates_target,
                "mutates_session": self.mutates_session,
                "writes_filesystem": self.writes_filesystem,
                "launches_process": self.launches_process,
                "accesses_network": self.accesses_network,
                "changes_ui_state": self.changes_ui_state,
                "may_discard_data": self.may_discard_data,
                "idempotent": self.idempotent}


@dataclass(frozen=True)
class Operation:
    """一个可执行的操作：静态说明 + 副作用。`external` 标"这是第三方软件"。"""

    id: str
    label: str                     # 人话：这一步在干什么
    detail: str                    # 更具体的边界说明（含"不做什么"）
    effects: ToolEffects
    external: bool = False         # 第三方软件（nsys/cuobjdump/NVBit）
    network_origins: tuple[str, ...] = ()


# **唯一**的操作表。新增一个会启动子进程/写文件/联网的动作，必须在这里登记 ——
# `plan_for()` 会在这里查，`effects_for()` 查不到就抛（见 tests）。
OPERATIONS: dict[str, Operation] = {
    "read_inputs": Operation(
        "read_inputs", "读你给的数据文件",
        "只读：标定文件、访存轨迹、指令直方图、cubin、自检/回显报告。不修改它们。",
        ToolEffects()),
    "device_sample": Operation(
        "device_sample", "后台采样设备级读数（NVML：能耗/功率/显存忙碌/时钟）",
        "只读设备管理计数器（免 admin）；会开一个采样线程。不改 GPU 状态、不碰你的进程。",
        ToolEffects(mutates_session=True)),
    "probe_env": Operation(
        "probe_env", "探测本机环境（设备型号、驱动、工具链在哪）",
        "只读探测：设备查询（`nvidia-smi --query-gpu=…`；装了 pynvml 时走库）"
        "＋对找到的每个工具跑一次 `<tool> --version`（nsys/ncu/cuobjdump/nvidia-smi，判断版本与可用性）。它决定这次报告"
        "能给出什么、要降级到哪一层；不写文件、不改设备状态。**注意：即使在 --ks-synthetic 下"
        "也会跑**（要判断本机能给出什么），所以「合成模式完全没有外部副作用」是不成立的 —— "
        "写一份不完整的计划比不写计划更糟。",
        ToolEffects(launches_process=True),
        external=True),
    "run_target": Operation(
        "run_target", "把你的被测命令跑起来",
        "用**你自己的权限**执行 `--` 之后的命令（这是分析的前提）；它会写自己的输出、"
        "可能联网 —— 那都是你那条命令的行为，不是本工具替你决定的。我们不注入任何代码到该进程。",
        ToolEffects(mutates_session=True, writes_filesystem=True, launches_process=True,
                    idempotent=False)),
    "nsys_collect": Operation(
        "nsys_collect", "用 nsys 采集 kernel 级活动（CUPTI，不需要计数器权限）",
        "启动第三方 `nsys` 子进程，向临时目录写 `.nsys-rep`/`.sqlite`（体积与 launch 数成正比）。",
        ToolEffects(writes_filesystem=True, launches_process=True, idempotent=False),
        external=True),
    "nsys_export": Operation(
        "nsys_export", "用 nsys 把已有 `.nsys-rep` 导出成 sqlite",
        "启动第三方 `nsys export`；默认把 sqlite 写在 `.nsys-rep` 旁边（`--ks-keep-intermediate` 控制）。",
        ToolEffects(writes_filesystem=True, launches_process=True),
        external=True),
    "static_dump": Operation(
        "static_dump", "用 cuobjdump 读 cubin/.so 的静态资源与 SASS",
        "启动第三方 `cuobjdump` 读文件。**静态分析不执行被测二进制**，也不证明任何运行行为。",
        ToolEffects(launches_process=True),
        external=True),
    "write_outputs": Operation(
        "write_outputs", "把报告/包/表格写到你要的路径",
        "写 `-o <前缀>` 一族产物与案件目录 `<前缀>.case/`（`--no-case` 关）；已存在的同名文件会被覆盖。",
        ToolEffects(writes_filesystem=True, may_discard_data=True)),
    "skill_install": Operation(
        "skill_install", "把 skill 安装到 agent 的 skill 目录",
        "把 `skills/kernoscope/` 复制到目标目录（默认 `~/.claude/skills/`）：先备份 `.kernoscope.backup`"
        "（已存在则保留第一份）、原子写、写后回读校验；不动你已有的其它 skill 与配置。",
        ToolEffects(writes_filesystem=True)),
}


class UndeclaredEffect(RuntimeError):
    """操作没有 effects 声明 —— 属于本工具缺陷（宁可抛，也不静默发生副作用）。"""


def effects_for(op_id: str) -> ToolEffects:
    op = OPERATIONS.get(op_id)
    if op is None:
        raise UndeclaredEffect(
            f"操作 {op_id!r} 没有 effects 声明：任何会起进程/写文件/联网的动作都必须先登记到 "
            f"effects.OPERATIONS（这是纪律，不是文档）")
    return op.effects


@dataclass
class PlanItem:
    """计划里的一条：op + 这次运行的具体形状（target/命令/会写的文件）。"""

    op: Operation
    target: str = ""
    detail: str = ""
    commands: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    cost_s_est: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.op.id, "label": self.op.label, "target": self.target,
            "detail": self.detail or self.op.detail,
            "external": self.op.external,
            "effects": self.op.effects.to_dict(),
            "annotations": self.op.effects.annotations(),
        }
        if self.commands:
            d["commands"] = list(self.commands)
        if self.writes:
            d["writes"] = list(self.writes)
        if self.op.network_origins:
            d["network_origins"] = list(self.op.network_origins)
        if self.cost_s_est is not None:
            d["cost_s_est"] = round(self.cost_s_est, 1)
        return d


@dataclass
class Plan:
    items: list[PlanItem] = field(default_factory=list)
    note: str = ""
    argv: list[str] = field(default_factory=list)

    @property
    def launches_process(self) -> bool:
        return any(i.op.effects.launches_process for i in self.items)

    @property
    def writes_filesystem(self) -> bool:
        return any(i.op.effects.writes_filesystem for i in self.items)

    @property
    def requires_approval(self) -> bool:
        """什么算"需要你点头"：写文件 / 可能覆盖 / 改变被测对象。

        只读的外部查询工具（`nvidia-smi --query-gpu`）**不算** —— 否则"什么都要求批准"
        等于没有批准粒度。计划里仍然会如实标出"它会起进程"。
        """
        return any(i.op.effects.writes_filesystem or i.op.effects.may_discard_data
                   or i.op.effects.mutates_target for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "requires_approval": self.requires_approval,
                "steps": [i.to_dict() for i in self.items],
                "note": self.note}

    def commands(self) -> list[str]:
        return [c for i in self.items for c in i.commands]


# --------------------------------------------------------------------------
# 由本次参数推出计划（纯函数：不执行任何东西，便于测试）
# --------------------------------------------------------------------------
def plan_for(args: Any, argv: Sequence[str] | None = None) -> Plan:
    """把一次调用的副作用算出来。**只读 args，不碰系统**。"""
    plan = Plan(argv=list(argv or []))
    argv = list(argv or [])

    # ---- ① 读入的输入文件（只读）----
    read_targets = [("--ks-calibration", getattr(args, "ks_calibration", None)),
                    ("--ks-memtrace", getattr(args, "ks_memtrace", None)),
                    ("--ks-instr-hist", getattr(args, "ks_instr_hist", None)),
                    ("--ks-achieved-occupancy", getattr(args, "ks_achieved_occupancy", None)),
                    ("--ks-selfcheck", getattr(args, "ks_selfcheck", None)),
                    ("-k/--import", getattr(args, "import_report", None)),
                    ("--ks-sqlite", getattr(args, "ks_sqlite", None)),
                    ("--ks-nsys-rep", getattr(args, "ks_nsys_rep", None))]
    cubins = list(getattr(args, "ks_cubin", []) or [])
    named = [f"{flag} {path}" for flag, path in read_targets if path]
    if cubins:
        named.append(f"--ks-cubin {' '.join(cubins)}")
    if named:
        plan.items.append(PlanItem(OPERATIONS["read_inputs"], target="；".join(named)))

    synthetic = bool(getattr(args, "ks_synthetic", False))
    importing = bool(getattr(args, "import_report", None))
    sqlite = bool(getattr(args, "ks_sqlite", None))
    nsys_rep = bool(getattr(args, "ks_nsys_rep", None))

    # ---- ② 环境探测（只读设备查询；**任何模式都会发生**，包括 --ks-synthetic）----
    # 这一条是本模块第一次跑测试时抓出来的：`--ks-synthetic --ks-no-nvml` 明明声明
    # "没有任何外部副作用"，PATH 桩却记到 4 次 nvidia-smi 调用 —— doctor 的能力判定
    # 需要先知道本机有什么。计划漏掉它，就是一份"看起来严谨但不真"的计划。
    plan.items.append(PlanItem(
        OPERATIONS["probe_env"], target="nvidia-smi --query-gpu=name,driver_version,…",
        detail="决定这次报告能给出什么、要降级到哪一层；只读、不写文件"))

    # ---- ③ 设备采样（NVML，只读）----
    if not getattr(args, "ks_no_nvml", False) and not importing:
        plan.items.append(PlanItem(
            OPERATIONS["device_sample"],
            target="NVML（pynvml 优先，退回 nvidia-smi 子进程轮询）",
            detail="只读设备管理计数器（能耗/功率/显存忙碌/时钟）；200 ms 周期。"
                   "加 --ks-no-nvml 可以完全关掉（代价：没有能耗与显存忙碌代理）"))

    # ---- ④ 跑被测命令 / 采集 ----
    if not synthetic and not importing and not sqlite and not nsys_rep:
        program = " ".join(getattr(args, "program", []) or [])
        plan.items.append(PlanItem(
            OPERATIONS["run_target"], target=program or "（未给出命令）",
            detail="用你自己的权限执行；它写什么文件、连不连网都由那条命令决定",
            cost_s_est=float(getattr(args, "ks_warmup", 0.0) or 0.0) or None))
        plan.items.append(PlanItem(
            OPERATIONS["nsys_collect"],
            target="临时目录（--ks-keep-intermediate 时留在 -o 旁边）",
            detail="采集体积与 launch 数成正比：十万级 launch 会写出上百 MB 中间产物",
            commands=["nsys profile --stats=false -o <tmp>/run -- <你的命令>"]))
    if nsys_rep:
        plan.items.append(PlanItem(
            OPERATIONS["nsys_export"], target=str(args.ks_nsys_rep),
            commands=[f"nsys export --type sqlite --output <旁路> {args.ks_nsys_rep}"]))
    if cubins:
        plan.items.append(PlanItem(
            OPERATIONS["static_dump"], target=" ".join(cubins),
            commands=[f"cuobjdump -res-usage/-sass {cubins[0]}"]))
    if synthetic:
        plan.note = ("--ks-synthetic：**不执行你的命令、不采集**，数字是编造的，只用来验形状。"
                     "但它仍然会做一次只读的环境探测（probe_env），所以不是「零外部副作用」。")

    # ---- ⑤ 写出 ----
    # 这里**必须**和 `cli._emit` 算出同一批路径：计划写"out/rep.json"而实际写
    # "out/rep.ksreport.json" 就是一封撒谎的计划（A2 的对账测试正是这么抓到的）。
    writes: list[str] = []
    export = getattr(args, "export", None)
    if export:
        base = str(export)[:-len(".json")] if str(export).endswith(".json") else str(export)
        writes.append(f"{base}.ksreport.json")
        if getattr(args, "launch_csv", False):
            writes.append(f"{base}.launches.csv")
        if not getattr(args, "ks_perfetto", None):
            writes.append(f"{base}.perfetto.json")
        if not getattr(args, "no_case", False):
            writes.append(f"{base}.case/（manifest.json / env.json / journal.jsonl / "
                          f"next.md / README.md）")
    if getattr(args, "packet", None) and getattr(args, "packet") != "-":
        writes.append(str(args.packet))
    if getattr(args, "ks_perfetto", None):
        writes.append(str(args.ks_perfetto))
    if writes:
        plan.items.append(PlanItem(
            OPERATIONS["write_outputs"], target="；".join(writes), writes=writes,
            detail="同名文件会被覆盖（含案件目录里的 journal 追加）"))
    return plan


REQUIRES_APPROVAL_TEXT = ("需要你点头的：写文件（可能覆盖同名文件）、改变被测对象。"
                          "只读的外部查询（`nvidia-smi --query-gpu`、`<tool> --version`）"
                          "不算 —— 计划里仍然会如实标出它会起进程。")


def render_plan(plan: Plan, width: int = 96) -> str:
    """人读的计划（`--plan`）。第一行就说清"这一步会不会动东西"。"""
    lines: list[str] = []
    lines.append("执行计划（`--plan`：只打印，不执行；没有跑你的命令、没有采集、没有写文件）")
    lines.append("-" * width)
    if not plan.items:
        lines.append("  这次调用没有任何副作用。")
    for i, item in enumerate(plan.items, 1):
        e = item.op.effects
        flags = []
        if e.launches_process:
            flags.append("起进程")
        if e.writes_filesystem:
            flags.append("写文件")
        if e.may_discard_data:
            flags.append("可能覆盖")
        if e.mutates_session:
            flags.append("会话内状态")
        tag = "、".join(flags) if flags else "只读"
        ext = "（第三方软件）" if item.op.external else ""
        lines.append(f"{i}. {item.op.label}{ext}  [{tag}]")
        if item.target:
            lines.append(f"     目标：{item.target}")
        if item.detail or item.op.detail:
            lines.append(f"     说明：{item.detail or item.op.detail}")
        if item.commands:
            for c in item.commands:
                lines.append(f"     命令：{c}")
        if item.writes:
            lines.append(f"     会写：{'；'.join(item.writes)}")
        if item.cost_s_est:
            lines.append(f"     预计额外等待：≈{item.cost_s_est:.0f}s")
    if plan.note:
        lines.append("")
        lines.append("注：" + plan.note)
    lines.append("")
    lines.append(REQUIRES_APPROVAL_TEXT)
    lines.append("结论：" + ("这次调用会改变世界 —— 确认后去掉 --plan 再跑。"
                             if plan.requires_approval else
                             "这次调用是只读的（除了会话内的采样线程）。"))
    return "\n".join(lines)
