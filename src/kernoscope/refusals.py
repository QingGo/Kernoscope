"""结构化拒绝（W3）—— 拒绝给数字时，除了「为什么」，还要给「怎么办」。

设计动机（来自本项目的失败史）：`n/a` 行曾经只带一句中文散文原因。对**人**来说那还算
够用；对**agent**来说不够 —— agent 需要的是可编程的东西：

  code        闭集里的一个标识符（可以写进 if、可以统计、可以翻译）
  observed    实测到的值（例如覆盖率 71.4%）
  threshold   门槛（例如 80%），与 observed 同单位 —— 两者相减才知道差多少
  remedy      可执行的补救动作 + 代价估计（几秒？几分钟？要人工吗？）
  severity    blocking=这条结论不能用；degraded=能用但口径更弱

**散文不是被禁止，而是不再承载判决。** 判决在 code/threshold 里，散文只解释。

`RefusalCode` 是闭集：新增一个拒绝类型就要在这里加一个 code，并给出 remedy。
这样「拒绝」这件事就有了单一来源，可以被测试、被统计、被 capabilities 复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RefusalCode(str, Enum):
    """闭集：每一个都对应一种「我们不给数」的真实原因。

    命名规则：`<被拒绝的东西>_<为什么>`。带 `_MISSING` 的是环境缺件（可安装解决），
    带 `_LOW`/`_UNKNOWN`/`_MISMATCH` 的是前提不成立（可改测量方式解决）。
    """

    # ---- 环境缺件：这台机器上没有这个数据源 ----
    NO_GPU = "no_gpu"                         # 没有任何 NVIDIA 设备
    PYTORCH_MISSING = "pytorch_missing"       # 没有 PyTorch（selftest 的已知答案负载需要它）
    NO_NVML = "no_nvml"                       # nvidia-smi/pynvml 不可用
    NO_NVML_SAMPLES = "no_nvml_samples"       # 采样被关掉或一次都没采到
    NSYS_MISSING = "nsys_missing"             # 没有 nsys -> 没有 kernel 级数据
    CUOBJDUMP_MISSING = "cuobjdump_missing"   # 没有 cuobjdump -> 没有静态资源/SASS
    NO_STATIC_INPUT = "no_static_input"       # 没给 --ks-cubin，静态 section 无从下手
    NVBIT_MISSING = "nvbit_missing"           # 没有 NVBit -> 没有指令级/访存轨迹
    NO_MEMTRACE = "no_memtrace"               # 没给 --ks-memtrace
    NO_INSTR_HIST = "no_instr_hist"           # 没给 --ks-instr-hist
    NO_OCCUPANCY_PROBE = "no_occupancy_probe"  # 没跑 occupancy 探针
    NO_SASS_SOURCE = "no_sass_source"         # SASS/源码逐行对照不可得
    NO_COUNTERS = "counters_denied"           # 硬件计数器被权限/驱动策略拒绝
    COUNTERS_UNKNOWN = "counters_unknown"     # 无法判定计数器是否可用（不是"被拒"）
    COUNTERS_NOT_INSTALLED = "counters_not_installed"  # 连 ncu 都没有，无法判定
    SAMPLING_UNAVAILABLE = "sampling_unavailable"  # PC 采样（PmSampling/SourceCounters）不可得
    INSTRUMENTATION_MISSING = "instrumentation_missing"  # 需要插桩/额外采集通道
    NO_NVLINK = "no_nvlink"                   # 没有 NVLink / NVML 不支持
    NOT_APPLICABLE = "not_applicable"         # 这个指标在本设备上本来就无定义
    MIXED_LAUNCH_CONFIG = "mixed_launch_config"  # 同名 kernel 有多组配置，单值会误导
    NO_CLOCK_READING = "no_clock_reading"     # 没有 SM 时钟读数（NVML 被关/不可用）

    # ---- 前提不成立：数据源在，但口径不成立 ----
    NO_LAUNCH_ACTIVITY = "no_launch_activity"  # 采到了但没有任何 kernel 活动
    NO_KERNEL_MATCH = "no_kernel_match"        # -k 过滤后为空
    NO_KERNELS = "no_kernels"                  # 报告里没有 kernel
    MULTI_KERNEL_WINDOW = "multi_kernel_window"  # 窗口里不止一个 kernel，不可归因
    COVERAGE_TOO_LOW = "coverage_too_low"      # 窗口没被负载主导
    SHARE_TOO_LOW = "share_too_low"            # 被匹配 kernel 不是窗口主角
    SHARE_UNKNOWN = "share_unknown"            # 逐 launch 时长为空，份额无法判定
    BELOW_RESOLUTION = "below_resolution"      # 读数低于仪器分辨率下限
    MEM_BUSY_ZERO = "mem_busy_zero"            # 显存忙碌恒为 0
    TRACE_UNAVAILABLE = "trace_unavailable"    # 轨迹口径给不出字节
    TRACE_TOO_MANY_ACCESSES = "trace_too_many_accesses"  # 访问数超上限，拒绝给偏小值
    LEDGER_BASIS_UNAVAILABLE = "ledger_basis_unavailable"  # 对拍口径不成立
    NO_CALIBRATION = "no_calibration"          # 缺本机实测峰值
    CALIBRATION_CALIBER_MISMATCH = "calibration_caliber_mismatch"  # 标定口径不可比
    NO_WORKLOAD = "no_workload"                # 没给 FLOPs/bytes，roofline 无输入
    UNKNOWN_ARCH = "unknown_arch"              # 架构不在表里，算不出上界
    NO_SELFTEST_BAND = "no_selftest_band"      # 没跑 selftest，没有本机误差带/MDD
    MDD_UNKNOWN = "mdd_unknown"                # 误差带里没有这个指标的 MDD
    PROTOCOL_MISMATCH = "protocol_mismatch"    # 两次运行协议不同，不可比
    BOTH_REFUSED = "both_refused"              # 两次运行都被同一个前提挡住：没有数字可比
    PREMISE_FAILED = "premise_failed"          # 判词自己的前提之一不成立（值因此不能引用）
    METRIC_ABSENT = "metric_absent"            # 请求的指标不在报告/包里
    LEGACY_REPORT = "legacy_report"            # 旧报告没有结构化拒绝信息
    INTERNAL_UNKNOWN = "internal_unknown"      # 兜底：未分类的拒绝（出现即为 bug 信号）


# 每个 code 的**静态**一句话说明。它不进决策包（每次都一样，而包里的字节要花在每次都变
# 的东西上），但人视图要用它把「code」翻成人话 —— 所以它是 code 的一部分，不是文档。
CODE_TITLES: dict["RefusalCode", str] = {
    RefusalCode.NO_GPU: "本机没有 NVIDIA GPU",
    RefusalCode.PYTORCH_MISSING: "没有安装 PyTorch",
    RefusalCode.NO_NVML: "nvidia-smi / pynvml 不可用",
    RefusalCode.NO_NVML_SAMPLES: "没有 NVML 采样点（采样被关掉或不可用）",
    RefusalCode.NSYS_MISSING: "没有 nsys：kernel 级数据不可得",
    RefusalCode.CUOBJDUMP_MISSING: "没有 cuobjdump：静态资源/SASS 不可得",
    RefusalCode.NO_STATIC_INPUT: "没有提供 cubin/.so",
    RefusalCode.NVBIT_MISSING: "未配置 NVBit：指令级/访存轨迹不可得",
    RefusalCode.NO_MEMTRACE: "没有提供访存轨迹文件",
    RefusalCode.NO_INSTR_HIST: "没有提供指令直方图",
    RefusalCode.NO_OCCUPANCY_PROBE: "没跑 occupancy 探针",
    RefusalCode.NO_SASS_SOURCE: "SASS/源码逐行对照未实现",
    RefusalCode.NO_COUNTERS: "硬件计数器被权限/驱动策略拒绝",
    RefusalCode.COUNTERS_UNKNOWN: "无法判定计数器是否可用（不等于被拒绝）",
    RefusalCode.COUNTERS_NOT_INSTALLED: "没有 ncu，无法判定计数器是否可用",
    RefusalCode.SAMPLING_UNAVAILABLE: "PC 采样（PmSampling/SourceCounters）不可得",
    RefusalCode.INSTRUMENTATION_MISSING: "需要插桩/多趟重放的通道不可得",
    RefusalCode.NO_NVLINK: "没有 NVLink 硬件/NVML 支持",
    RefusalCode.NOT_APPLICABLE: "该指标在本设备上无定义",
    RefusalCode.MIXED_LAUNCH_CONFIG: "同名 kernel 有多组 launch 配置，单值会误导",
    RefusalCode.NO_CLOCK_READING: "没有 SM 时钟读数",
    RefusalCode.NO_LAUNCH_ACTIVITY: "没有采到任何 kernel 活动",
    RefusalCode.NO_KERNEL_MATCH: "过滤后没有匹配的 kernel",
    RefusalCode.NO_KERNELS: "报告里没有 kernel",
    RefusalCode.MULTI_KERNEL_WINDOW: "窗口里不止一个 kernel，per-launch 不可归因",
    RefusalCode.COVERAGE_TOO_LOW: "采样窗口没被被测负载主导",
    RefusalCode.SHARE_TOO_LOW: "被匹配的 kernel 不是窗口主角",
    RefusalCode.SHARE_UNKNOWN: "逐 launch 时长为空，份额无法判定",
    RefusalCode.BELOW_RESOLUTION: "读数低于仪器分辨率下限",
    RefusalCode.MEM_BUSY_ZERO: "显存忙碌恒为 0",
    RefusalCode.TRACE_UNAVAILABLE: "访存轨迹给不出字节数",
    RefusalCode.TRACE_TOO_MANY_ACCESSES: "访问数超过分析上限",
    RefusalCode.LEDGER_BASIS_UNAVAILABLE: "账本对拍口径不成立",
    RefusalCode.NO_CALIBRATION: "缺少本机实测峰值（未标定）",
    RefusalCode.CALIBRATION_CALIBER_MISMATCH: "标定口径与本次测量不可比",
    RefusalCode.NO_WORKLOAD: "没有给工作量（FLOPs/bytes）",
    RefusalCode.UNKNOWN_ARCH: "未知架构，算不出上界",
    RefusalCode.NO_SELFTEST_BAND: "没有本机误差带与 MDD",
    RefusalCode.MDD_UNKNOWN: "误差带里没有这个指标的 MDD",
    RefusalCode.PROTOCOL_MISMATCH: "两次运行的协议不同，不可比",
    RefusalCode.BOTH_REFUSED: "两侧都被同一个前提挡住，没有数字可比",
    RefusalCode.PREMISE_FAILED: "判词的前提之一不成立（数字不可引用）",
    RefusalCode.METRIC_ABSENT: "请求的指标不存在",
    RefusalCode.LEGACY_REPORT: "旧报告没有结构化拒绝信息",
    RefusalCode.INTERNAL_UNKNOWN: "未分类的拒绝（属于本工具缺陷）",
}


def code_title(code: "RefusalCode | str") -> str:
    """code → 一句话人话（静态；查不到时退回 code 本身，绝不编造）。"""
    if isinstance(code, str):
        try:
            code = RefusalCode(code)
        except ValueError:
            return code
    return CODE_TITLES.get(code, code.value)


def missing_titles() -> list[str]:
    return [c.value for c in RefusalCode if c not in CODE_TITLES]


@dataclass(frozen=True)
class Remedy:
    """补救动作 + 代价估计。`cost_s` 是**用户侧额外等待**的秒数（0=几乎免费）。

    `then_answers` 列出补救之后会多出哪些能力（能力 id），让 agent 能估「值不值」。
    """

    action: str
    command: str = ""
    cost_s: float = 0.0
    cost_kind: str = "wall_seconds"      # wall_seconds | manual | unavailable
    then_answers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["then_answers"] = list(self.then_answers)
        return d


# 每个 code 的默认补救办法。具体场景可在构造 Refusal 时覆盖（例如把"延长到约 N 秒"
# 里的 N 用实测覆盖率算出来）。
DEFAULT_REMEDIES: dict[RefusalCode, Remedy] = {
    RefusalCode.PYTORCH_MISSING: Remedy(
        "装 PyTorch（自检的已知答案负载只需要 torch）",
        command="pip install torch", cost_kind="manual",
        then_answers=("selftest_band", "mdd")),
    RefusalCode.NO_GPU: Remedy(
        "换一台有 NVIDIA GPU 的机器；或在没有 GPU 时用 --ks-synthetic 跑通流程看输出形状",
        command="kernoscope --ks-synthetic --set basic", cost_kind="manual"),
    RefusalCode.NO_NVML: Remedy(
        "装 pynvml（pip install 'kernoscope[nvml]'）或确认 nvidia-smi 可用",
        command="pip install 'kernoscope[nvml]'", cost_kind="manual",
        then_answers=("dram_bytes", "device_busy")),
    RefusalCode.NO_NVML_SAMPLES: Remedy(
        "去掉 --ks-no-nvml 让设备级采样生效（代价接近 0：采样在后台线程里做）",
        cost_s=0.0, then_answers=("dram_bytes", "device_busy")),
    RefusalCode.NSYS_MISSING: Remedy(
        "安装 Nsight Systems（随 CUDA Toolkit 提供），或提供已有的 .nsys-rep/.sqlite",
        command="kernoscope --ks-sqlite <kernels.sqlite>", cost_kind="manual",
        then_answers=("kernel_times", "launch_configs")),
    RefusalCode.CUOBJDUMP_MISSING: Remedy(
        "安装 CUDA Toolkit（cuobjdump 随 toolkit 提供），或用已有报告里的静态字段",
        cost_kind="manual", then_answers=("static_resources", "sass_mix")),
    RefusalCode.NO_STATIC_INPUT: Remedy(
        "用 --ks-cubin <cubin/.so> 指向被测二进制；没有它就没有寄存器/共享内存/SASS",
        command="kernoscope --ks-cubin build/app.so ...", cost_kind="manual",
        then_answers=("static_resources", "sass_mix")),
    RefusalCode.NVBIT_MISSING: Remedy(
        "自行安装 NVBit（许可证禁止本项目随包分发）并设 NVBIT_PATH",
        cost_kind="manual", then_answers=("memtrace", "instruction_mix")),
    RefusalCode.NO_MEMTRACE: Remedy(
        "加 --ks-memtrace <trace>：轨迹口径只测被追踪的 kernel，不受窗口构成影响",
        command="kernoscope --ks-memtrace ks-memtrace.txt ...", cost_s=600.0,
        then_answers=("dram_bytes_trace", "l2_hit_rate")),
    RefusalCode.NO_INSTR_HIST: Remedy(
        "加 --ks-instr-hist <opcode_hist.txt>：给出动态指令数与发射槽利用率上界",
        command="kernoscope --ks-instr-hist opcode_hist.txt ...", cost_s=600.0,
        then_answers=("issue_slot_upper_bound",)),
    RefusalCode.NO_OCCUPANCY_PROBE: Remedy(
        "跑 occupancy 探针并把 JSON 用 --ks-achieved-occupancy 传进来（需要 NVBit）",
        command="kernoscope occupancy --records <probe-records> -o occ.json",
        cost_s=600.0, then_answers=("achieved_occupancy",)),
    RefusalCode.NO_SASS_SOURCE: Remedy(
        "SASS/源码逐行对照需要 --import-source，本项目不实现：改用 cuobjdump -sass 自行对照",
        cost_kind="manual"),
    RefusalCode.NO_COUNTERS: Remedy(
        "计数器是 host-wide 的，容器内无法突破（不要用 sudo/setuid 试）；"
        "换用本工具的非计数器口径，或在有权限的机器上跑 ncu",
        cost_kind="unavailable"),
    RefusalCode.COUNTERS_UNKNOWN: Remedy(
        "装 ncu 后重跑 `kernoscope doctor --json` 才能判定计数器是否可用",
        command="kernoscope doctor --json", cost_kind="manual"),
    RefusalCode.COUNTERS_NOT_INSTALLED: Remedy(
        "装 ncu（CUDA Toolkit 的一部分）后重跑 doctor："
        "「没装」不等于「被权限限制」，两者不能混",
        command="kernoscope doctor --json", cost_kind="manual"),
    RefusalCode.SAMPLING_UNAVAILABLE: Remedy(
        "PC 采样（PmSampling/SourceCounters）与计数器同源、同样被权限锁死；"
        "需要 PC 级归因时用 --ks-instr-hist 的发射槽上界替代（方向性结论，非周期细分）",
        command="kernoscope --ks-instr-hist opcode_hist.txt ...", cost_kind="unavailable"),
    RefusalCode.INSTRUMENTATION_MISSING: Remedy(
        "这一类需要额外的采集通道（插桩/多趟重放）：本工具不重放，"
        "用 NVBit 轨迹（--ks-memtrace）或指令直方图（--ks-instr-hist）替代",
        cost_s=600.0),
    RefusalCode.NO_NVLINK: Remedy(
        "需要 NVLink 硬件与 NVML 支持；单卡实例上这个 section 本来就不适用",
        cost_kind="unavailable"),
    RefusalCode.NOT_APPLICABLE: Remedy(
        "该指标在本设备/本平台上无定义（不是数据缺失）：忽略这一行即可",
        cost_s=0.0),
    RefusalCode.MIXED_LAUNCH_CONFIG: Remedy(
        "同名 kernel 有多组 launch 配置时我们不报单一数值（会误导多数 launch）："
        "逐组读数见 LaunchConfigs section；只关心一组时用 -c/-k 固定它",
        cost_s=0.0, then_answers=("launch_configs",)),
    RefusalCode.NO_CLOCK_READING: Remedy(
        "加 --ks-clock-mhz <MHz>（或让 NVML 采样开着）：没有 SM 时钟就算不出发射槽上限",
        command="kernoscope --ks-clock-mhz 2520 ...", cost_s=0.0,
        then_answers=("issue_slot_upper_bound",)),
    RefusalCode.NO_LAUNCH_ACTIVITY: Remedy(
        "确认被测命令真的在 GPU 上跑 kernel，且采集时带上了 CUDA activity"
        "（--ks-sqlite 指向的库要有 CUPTI_ACTIVITY_KIND_KERNEL 表）",
        command="nsys profile --trace=cuda -o run -- python app.py", cost_s=30.0,
        then_answers=("kernel_times",)),
    RefusalCode.NO_KERNEL_MATCH: Remedy(
        "放宽 -k 过滤（用更短的正则片段），或先不加 -k 看有哪些 kernel 名字",
        command="kernoscope --ks-sqlite kernels.sqlite --launch-csv", cost_s=5.0,
        then_answers=("kernel_times",)),
    RefusalCode.NO_KERNELS: Remedy(
        "先确认采集到了 kernel 活动（见 no_launch_activity 的补救办法）",
        cost_s=30.0),
    RefusalCode.MULTI_KERNEL_WINDOW: Remedy(
        "用 -k 只选一个 kernel，让窗口里只有它；否则只能给整窗口总量",
        command="kernoscope -k regex:<kernel> --set detailed -- python app.py", cost_s=0.0,
        then_answers=("dram_bytes_per_launch",)),
    RefusalCode.COVERAGE_TOO_LOW: Remedy(
        "把稳态循环跑长（建议 ≥20 秒），或加 --ks-window auto 只统计稳态段；"
        "覆盖率不足时窗口里大部分时间是空闲，忙碌时段是否饱和无从确认",
        command="kernoscope --ks-window auto -- python bench_loop.py", cost_s=20.0,
        then_answers=("dram_bytes",)),
    RefusalCode.SHARE_TOO_LOW: Remedy(
        "把窗口缩到只含被匹配的 kernel（--ks-warmup/--ks-window-s 只跑一段），"
        "或走整窗口口径（--ks-ledger-total-mb），或加 --ks-memtrace 换轨迹口径",
        command="kernoscope --ks-window-s 10 -k regex:<kernel> -- python bench_loop.py",
        cost_s=20.0, then_answers=("dram_bytes_per_launch",)),
    RefusalCode.SHARE_UNKNOWN: Remedy(
        "逐 launch 时长为空：确认 nsys 的 CUPTI kernel 表可用（kernel 时长是 exact 的，"
        "通常不缺）；缺时用 --ks-sqlite 直接指定库",
        command="kernoscope --ks-sqlite kernels.sqlite", cost_s=5.0),
    RefusalCode.BELOW_RESOLUTION: Remedy(
        "让 GPU 保持饱和：NVML 显存忙碌是整数百分比，低于 1 个百分点的读数会被截断成 0；"
        "把测量窗口对准真正在搬数据的时段（--ks-window auto / --ks-warmup）",
        command="kernoscope --ks-window auto -- python bench_loop.py", cost_s=20.0,
        then_answers=("dram_bytes",)),
    RefusalCode.MEM_BUSY_ZERO: Remedy(
        "显存忙碌恒为 0：确认被测负载真的在搬显存（而不是 L2 命中），且采样周期"
        "（200 ms）没有整段错过忙碌期",
        cost_s=20.0),
    RefusalCode.TRACE_UNAVAILABLE: Remedy(
        "轨迹口径给不出字节时，改用 NVML 代理口径（--ks-calibration + --ks-window auto）"
        "或检查轨迹是否覆盖了被匹配的 kernel",
        cost_s=20.0, then_answers=("dram_bytes",)),
    RefusalCode.TRACE_TOO_MANY_ACCESSES: Remedy(
        "提高 --ks-max-accesses（内存换精度），或缩小被追踪 kernel 的迭代数："
        "超上限时我们拒绝给一个偏小的字节数",
        command="kernoscope --ks-max-accesses 20000000 --ks-memtrace t.txt ...",
        cost_kind="manual", then_answers=("dram_bytes_trace",)),
    RefusalCode.LEDGER_BASIS_UNAVAILABLE: Remedy(
        "换对拍口径：--ks-ledger-total-mb 走整窗口，--ks-ledger-mb 走 per-launch"
        "（后者要求窗口主导性达标）",
        command="kernoscope --ks-ledger-total-mb 512 -- python bench_loop.py", cost_s=0.0,
        then_answers=("ledger_verdict",)),
    RefusalCode.NO_CALIBRATION: Remedy(
        "跑一次本机标定并把结果传进来（roofline 与 DRAM 代理都依赖它）",
        command="kernoscope selftest --out machine-profile.json", cost_s=60.0,
        then_answers=("dram_bytes", "roofline")),
    RefusalCode.CALIBRATION_CALIBER_MISMATCH: Remedy(
        "标定口径与本次测量不可比（例如 fp32 用 reduced-precision reduction 跑出来的峰值）："
        "重跑标定并让口径一致",
        command="python tools/machine-profile.py --quick", cost_s=300.0,
        then_answers=("roofline",)),
    RefusalCode.NO_WORKLOAD: Remedy(
        "给出每次 launch 的工作量：--ks-gemm M,N,K 或 --ks-flops/--ks-bytes",
        command="kernoscope --ks-gemm 4096,4096,4096 -- python app.py", cost_s=0.0,
        then_answers=("roofline",)),
    RefusalCode.UNKNOWN_ARCH: Remedy(
        "用 --ks-sm-count 指定 SM 数，或报告里带上 compute capability："
        "未知架构算不出 occupancy 上界，我们不猜",
        cost_kind="manual"),
    RefusalCode.NO_SELFTEST_BAND: Remedy(
        "跑一次 selftest 得到本机误差带与最小可分辨差异（MDD）；"
        "没有它就无法判断两次运行的差别是真差别还是噪声",
        command="kernoscope selftest --out machine-selfcheck.json", cost_s=120.0,
        then_answers=("run_to_run_diff", "mdd")),
    RefusalCode.MDD_UNKNOWN: Remedy(
        "误差带里没有这个指标（MDD 按指标族给）：用 --metric 指定有 MDD 的指标，"
        "或重跑 selftest 覆盖该负载族",
        command="kernoscope selftest --out machine-selfcheck.json", cost_s=120.0),
    RefusalCode.PREMISE_FAILED: Remedy(
        "先让前提成立（按该前提对应的 code 处理）：前提不成立时那个数字不能用，"
        "把它当结论会得到一个方向性错误",
        command="kernoscope capabilities --json", cost_kind="manual"),
    RefusalCode.BOTH_REFUSED: Remedy(
        "两侧都被同一种原因拒绝（例如都没有标定）：先按该原因的补救办法解除前提，"
        "这次比较才有意义 —— 没有数字时 diff 不下任何结论",
        command="kernoscope capabilities --json", cost_kind="manual"),
    RefusalCode.PROTOCOL_MISMATCH: Remedy(
        "两次比较必须同协议：同标定文件、同窗口模式、同工具版本、同 kernel 选择；"
        "协议不同时我们拒绝给方向判词（宁可说不可比）",
        cost_s=0.0),
    RefusalCode.METRIC_ABSENT: Remedy(
        "用 --json 看本次报告里实际存在哪些指标，或用 `kernoscope capabilities --json` "
        "看哪些问题能答（不存在 ≠ 被拒）",
        command="kernoscope capabilities --json", cost_s=0.0),
    RefusalCode.LEGACY_REPORT: Remedy(
        "报告是结构化拒绝之前生成的（没有 code 字段）：重跑一次采集即可带上；"
        "本工具按 legacy 标注，不猜原因",
        cost_s=5.0),
    RefusalCode.INTERNAL_UNKNOWN: Remedy(
        "未分类的拒绝 —— 这属于本工具的缺陷，请按 issue 模板报告",
        cost_kind="manual"),
}


@dataclass(frozen=True)
class Refusal:
    """一条结构化的「我们不给这个数」。"""

    code: RefusalCode
    message: str
    observed: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    remedy: Remedy | None = None
    severity: str = "blocking"          # blocking | degraded
    # 复合拒绝：一个 n/a 行可能同时踩中多个前提（例如"没有采样点"且"缺峰值"）。
    # 主 code/观测值是第一条，其余放进 causes —— 不让读者以为只有一个原因。
    causes: tuple["Refusal", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, RefusalCode):
            raise ValueError(f"refusal code must be a RefusalCode, got {self.code!r}")
        if not self.message:
            raise ValueError(f"refusal {self.code.value} needs a message")
        if self.remedy is None:
            object.__setattr__(self, "remedy", default_remedy(self.code))

    @property
    def remedy_or_default(self) -> Remedy:
        return self.remedy if self.remedy is not None else default_remedy(self.code)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity,
            "observed": dict(self.observed),
            "threshold": dict(self.threshold),
            "remedy": self.remedy_or_default.to_dict(),
        }
        if self.causes:
            d["causes"] = [c.to_dict() for c in self.causes]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Refusal":
        code = d.get("code") or RefusalCode.LEGACY_REPORT.value
        try:
            code_enum = RefusalCode(code)
        except ValueError:
            code_enum = RefusalCode.INTERNAL_UNKNOWN
        rem = d.get("remedy") or {}
        remedy = None
        if rem:
            remedy = Remedy(
                action=str(rem.get("action", "")),
                command=str(rem.get("command", "")),
                cost_s=float(rem.get("cost_s") or 0.0),
                cost_kind=str(rem.get("cost_kind") or "wall_seconds"),
                then_answers=tuple(rem.get("then_answers") or ()),
            )
        return Refusal(
            code=code_enum,
            message=str(d.get("message") or ""),
            observed=dict(d.get("observed") or {}),
            threshold=dict(d.get("threshold") or {}),
            remedy=remedy,
            severity=str(d.get("severity") or "blocking"),
            causes=tuple(Refusal.from_dict(c) for c in (d.get("causes") or [])),
        )


def combined_refusal(refusals: "list[Refusal] | tuple[Refusal, ...]") -> Refusal:
    """把同一行上的多个失败前提合成一条拒绝（主 code = 第一条）。

    `n/a` 行只有一个位置放 refusal，但失败前提可能不止一个。合成而不是取第一条，
    是为了让 message 仍然点明**全部**没成立的前提（R-32 的要求）。
    """
    if not refusals:
        return Refusal(RefusalCode.INTERNAL_UNKNOWN, "无法估算（未知原因）")
    first = refusals[0]
    if len(refusals) == 1:
        return first
    return Refusal(
        code=first.code,
        message="；".join(r.message for r in refusals),
        observed=dict(first.observed),
        threshold=dict(first.threshold),
        remedy=first.remedy,
        severity=first.severity,
        causes=tuple(refusals[1:]),
    )


def default_remedy(code: RefusalCode) -> Remedy:
    """闭集里的每个 code 都必须有补救办法 —— 没有补救办法的 code 不允许存在。"""
    return DEFAULT_REMEDIES.get(
        code, Remedy("未知拒绝类型：请按 issue 模板报告", cost_kind="manual"))


def missing_remedies() -> list[str]:
    """自检用：闭集里有 code 没写 remedy 时列出来（测试断言为空）。"""
    return [c.value for c in RefusalCode if c not in DEFAULT_REMEDIES]


def legacy_refusal(message: str) -> Refusal:
    """旧报告的 `n/a` 行没有 code：标注 legacy，不猜原因。"""
    return Refusal(RefusalCode.LEGACY_REPORT, message or "旧报告未记录结构化拒绝信息",
                   severity="blocking")


__all__ = ["CODE_TITLES", "Refusal", "RefusalCode", "Remedy", "code_title",
           "combined_refusal", "default_remedy", "legacy_refusal", "missing_remedies",
           "missing_titles"]
