"""`kernoscope doctor` —— 环境自检、能力矩阵与后端选择。

为什么需要它：在受限（共享租用容器）环境里，用户第一件想知道的事是
**"这台机器上我到底能拿到什么、拿不到什么、为什么"**。逐个手工检查
`/proc/driver/nvidia/params`、容器 capability、nsys/cuobjdump/NVML 是否存在很烦，
而且很容易得出错误结论（例如"我是 root 所以应该能用 ncu"）。

doctor 把这些检查收敛成一条命令，并且把结论直接映射到 provenance 分类：
    exact / measured / estimated / unavailable
—— 与报告里的标签是同一套语义，避免"文档说 A、工具给 B"。

所有路径都可通过环境变量覆盖，便于测试与在非 Linux 上运行：
    KS_NV_PARAMS       默认 /proc/driver/nvidia/params
    KS_PROC_STATUS     默认 /proc/self/status
    KS_PERF_PARANOID   默认 /proc/sys/kernel/perf_event_paranoid
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from . import __version__
from .provenance import UNAVAILABLE_REASONS
from .toolchain import Discovery, discover

DOCTOR_SCHEMA_VERSION = 2

# Linux capability 位号（与内核 include/uapi/linux/capability.h 一致）
CAP_NAMES: dict[int, str] = {
    0: "CHOWN", 1: "DAC_OVERRIDE", 2: "DAC_READ_SEARCH", 3: "FOWNER", 4: "FSETID",
    5: "KILL", 6: "SETGID", 7: "SETUID", 8: "SETPCAP", 9: "LINUX_IMMUTABLE",
    10: "NET_BIND_SERVICE", 11: "NET_BROADCAST", 12: "NET_ADMIN", 13: "NET_RAW",
    14: "IPC_LOCK", 15: "IPC_OWNER", 16: "SYS_MODULE", 17: "SYS_RAWIO", 18: "SYS_CHROOT",
    19: "SYS_PTRACE", 20: "SYS_PACCT", 21: "SYS_ADMIN", 22: "SYS_BOOT", 23: "SYS_NICE",
    24: "SYS_RESOURCE", 25: "SYS_TIME", 26: "SYS_TTY_CONFIG", 27: "MKNOD", 28: "LEASE",
    29: "AUDIT_WRITE", 30: "AUDIT_CONTROL", 31: "SETFCAP", 32: "MAC_OVERRIDE",
    33: "MAC_ADMIN", 34: "SYSLOG", 35: "WAKE_ALARM", 36: "BLOCK_SUSPEND",
    37: "AUDIT_READ", 38: "PERFMON", 39: "BPF", 40: "CHECKPOINT_RESTORE",
}

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "✓", WARN: "!", FAIL: "✗"}


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str = ""
    impact: str = ""


@dataclass
class Capability:
    """一项可交付的度量能力及其出处等级。"""

    provenance: str          # exact / derived / measured / estimated / unavailable
    items: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class DoctorReport:
    ok: bool = True
    checks: list[Check] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    backend_plan: list[str] = field(default_factory=list)
    device: dict[str, Any] = field(default_factory=dict)
    advice: list[str] = field(default_factory=list)
    # R-16/R-17：每个工具的发现结果（含"找过哪些路径"），让"未安装"可证伪
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 计数器四态：not-installed / not-in-path / denied / available / unknown
    counter_state: str = "unknown"
    counter_state_note: str = ""
    # NVBit 的发现结果（能力分层要用：有它才是 tier 3 指令级）
    nvbit_path: str = ""

    def add(self, group: str, name: str, status: str, detail: str = "", impact: str = "") -> None:
        self.checks.append(Check(group, name, status, detail, impact))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "tool": "kernoscope",
            "tool_version": __version__,
            "ok": self.ok,
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "device": self.device,
            "tools": self.tools,
            "counter_state": self.counter_state,
            "counter_state_note": self.counter_state_note,
            "nvbit_path": self.nvbit_path,
            "checks": [asdict(c) for c in self.checks],
            "capabilities": [asdict(c) for c in self.capabilities],
            "backend_plan": self.backend_plan,
            "advice": self.advice,
        }


# --------------------------------------------------------------------------
# 小工具（全部可测试：路径/命令查找都走这里）
# --------------------------------------------------------------------------
def _read(path_env: str, default: str) -> str:
    path = os.environ.get(path_env, default)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def discover_tool(name: str) -> Discovery:
    """单点发现入口（测试与 `KS_TOOL_SEARCH_DIRS` 都从这里注入）。"""
    return discover(name)


def find_tool(name: str) -> str | None:
    """多路径发现（PATH → CUDA_HOME → /usr/local/cuda* → 常见目录）。

    历史坑（B3）：只查 PATH 会在这类实例上给出"未安装 ncu"，而 ncu 就躺在
    `/usr/local/cuda-13.0/bin/ncu` —— 与后端 `find_cuobjdump()` 的结论互相矛盾。
    """
    return discover_tool(name).path


def tool_version(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    text = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"(\d{4}\.\d+(?:\.\d+)?|\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else (text.strip().splitlines() or [""])[0][:60]


def decode_capabilities(mask_hex: str) -> set[str]:
    """把 CapEff/CapBnd 的十六进制掩码解成 capability 名字集合。"""
    try:
        mask = int(mask_hex.strip(), 16)
    except (ValueError, AttributeError):
        return set()
    return {name for bit, name in CAP_NAMES.items() if mask >> bit & 1}


def parse_proc_status(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_nv_params(text: str) -> dict[str, str]:
    """`/proc/driver/nvidia/params` 是 `Key: Value` 行式文本。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def gpu_info() -> dict[str, Any]:
    if not find_tool("nvidia-smi"):
        return {}
    q = "name,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit"
    try:
        p = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if p.returncode != 0 or not p.stdout.strip():
        return {}
    rows = [r for r in p.stdout.strip().splitlines() if r.strip()]
    first = [x.strip() for x in rows[0].split(",")]
    while len(first) < 6:
        first.append("")
    info = {
        "name": first[0], "driver": first[1], "compute_capability": first[2],
        "memory_mib": first[3], "max_sm_clock_mhz": first[4], "power_limit_w": first[5],
        "count": len(rows),
    }
    # SM 数量：优先 torch（若装了），否则留空由报告侧处理
    try:
        import torch  # type: ignore

        info["sm_count"] = int(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception:
        info["sm_count"] = None
    return info


def counter_gate_probe(ncu_path: str | None = None) -> tuple[bool | None, str]:
    """用 ncu 自身快速探测"硬件计数器是否可用"。

    返回 (可用?, 说明)，其中 **None = 无法判定**（例如没装 ncu）——
    这一点很重要：不能把"没装工具"误报成"被权限限制"，否则会给出错误的归因。

    `ncu --query-metrics` 在无权限时会打印 ERR_NVGPUCTRPERM（实测），
    因此可作为 1 秒级的经验判定。
    """
    ncu = ncu_path or find_tool("ncu")
    if not ncu:
        return None, "在所有候选路径里都没有找到 ncu"
    try:
        p = subprocess.run([ncu, "--query-metrics"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None, f"{ncu} 探测超时/无法执行，无法判定"
    blob = (p.stdout or "") + (p.stderr or "")
    if "ERR_NVGPUCTRPERM" in blob:
        return False, "ncu reports ERR_NVGPUCTRPERM (empirically confirmed)"
    if "No devices" in blob or "no devices" in blob:
        return None, "ncu 找不到设备，无法判定"
    return True, "ncu probe did not report a permission error"


def classify_counter_state(found: Discovery | None, allowed: bool | None,
                           probe: bool | None, static: bool = False) -> tuple[str, str]:
    """计数器四态（R-17）：未安装 / 已安装但不在 PATH / 被拒 / 实测可用 / 无法判定。

    关键是**把「未安装」与「无法判定」分开**：前者可被一次 `ls` 推翻，后者才是诚实结论。

    `static=True` 表示调用方一次实测都没做（`doctor --no-probe`）：结论只能来自驱动策略
    与容器 capability。此时措辞必须写明「静态推断」，绝不能冒充实测；同时静态否定
    （RmProfilingAdminOnly=1）与实测否定一样足以定论，因此同样归入 denied。
    """
    if found is None or not found.found:
        return "not-installed", (f"在 {len(found.searched) if found else 0} 个候选目录里都没找到 ncu "
                                 f"（不是「没试」，是真的找遍了；--json 里有目录清单）")
    where = found.path or ""
    not_in_path = "（不在 PATH 上，已按多路径找到）" if found.found_but_not_in_path else ""
    if probe is False:
        who = "policy" if allowed is False else "runtime"
        return "denied", (f"ncu 存在于 {where}{not_in_path}，实测被拒（ERR_NVGPUCTRPERM，{who}）")
    if probe is True:
        return "available", f"ncu 存在于 {where}{not_in_path}，实测可用"
    if static and allowed is False:
        # 静态否定：策略只允许 admin 用计数器，容器 capability 又不足以突破。
        # 这已经足够定论，不需要（也不允许）说成实测。
        return "denied", (f"ncu 存在于 {where}{not_in_path}，静态推断被拒：驱动策略 "
                          f"RmProfilingAdminOnly=1（仅 admin 可用性能计数器），"
                          f"容器 capability 也不足以突破；本次未做实测探测（--no-probe）")
    if found.found_but_not_in_path:
        tail = "静态信息不足以定论（--no-probe，未做实测探测）" if static else "实测未完成判定"
        return "not-in-path", (f"ncu 存在于 {where}{not_in_path}；"
                               f"若只看 PATH 会误报「未安装」。{tail}")
    tail = "静态信息不足以定论（--no-probe，未做实测探测）" if static else "但实测无法判定"
    return "unknown", f"ncu 存在于 {where}，{tail}"


# --------------------------------------------------------------------------
# 自检主体
# --------------------------------------------------------------------------
def run_doctor(probe_counters: bool = True) -> DoctorReport:
    r = DoctorReport()

    # ---- 主机 ----
    r.add("host", "Python", OK,
          f"{platform.python_version()} on {platform.system()} {platform.release()}")

    # ---- GPU / 驱动 ----
    gpu = gpu_info()
    r.device = gpu
    if gpu:
        r.add("hardware", "GPU", OK,
              f"{gpu['name']} · driver {gpu['driver']} · sm_{str(gpu['compute_capability']).replace('.', '')}"
              f" · {gpu['memory_mib']} MiB"
              + (f" · {gpu['count']} devices" if gpu.get("count", 1) > 1 else ""))
    else:
        r.add("hardware", "GPU", FAIL, "nvidia-smi 不可用或未检测到 GPU",
              impact="无法采集任何 CUDA 数据；仅 --ks-synthetic 可用")

    # ---- 内核驱动策略 ----
    params = parse_nv_params(_read("KS_NV_PARAMS", "/proc/driver/nvidia/params"))
    regkey = params.get("RmProfilingAdminOnly")
    if regkey is None:
        r.add("policy", "内核驱动策略", WARN,
              "读不到 RmProfilingAdminOnly（非 NVIDIA 机器或未暴露 /proc）",
              impact="无法从驱动侧解释计数器是否受限")
        counters_allowed = None
    elif regkey.strip() == "0":
        r.add("policy", "内核驱动策略", OK,
              "RmProfilingAdminOnly = 0（所有用户可用性能计数器）")
        counters_allowed = True
    else:
        r.add("policy", "内核驱动策略", WARN,
              f"RmProfilingAdminOnly = {regkey}（仅 admin 可用性能计数器）",
              impact="ncu/DCGM PROF 等计数器类工具会报 ERR_NVGPUCTRPERM")
        counters_allowed = False

    # ---- 容器权限 ----
    status = parse_proc_status(_read("KS_PROC_STATUS", "/proc/self/status"))
    caps_eff = decode_capabilities(status.get("CapEff", ""))
    caps_bnd = decode_capabilities(status.get("CapBnd", ""))
    is_root = os.geteuid() == 0
    if status:
        r.add("permissions", "进程身份", OK if is_root else WARN,
              f"uid={os.geteuid()}" + ("（root）" if is_root else "（非 root）"),
              impact="" if is_root else "非 root 且无 capability 时无法 profile")
        for cap in ("SYS_ADMIN", "PERFMON"):
            have = cap in caps_eff
            in_bnd = cap in caps_bnd
            if have:
                r.add("permissions", f"CAP_{cap}", OK, "生效集中存在")
            else:
                r.add("permissions", f"CAP_{cap}", FAIL,
                      f"生效集无，边界集{'有' if in_bnd else '也没有'}",
                      impact="容器内无法获得硬件计数器权限（su/setuid/file-cap 都无法突破 CapBnd）")
    else:
        r.add("permissions", "容器 capability", WARN, "读不到 /proc/self/status")

    # ---- 工具链（多路径发现，R-16）----
    disc: dict[str, Discovery] = {name: discover_tool(name) for name in
                                 ("nsys", "ncu", "cuobjdump", "nvidia-smi")}
    r.tools = {k: v.as_dict() for k, v in disc.items()}
    nsys = disc["nsys"].path
    if nsys:
        r.add("toolchain", "nsys", OK,
              (tool_version([nsys, "--version"]) or "present")
              + (f"  @ {nsys}" if disc["nsys"].found_but_not_in_path else ""),
              impact="提供 exact 的 LaunchStats / 时长（CUPTI activity，不需要计数器）")
    else:
        r.add("toolchain", "nsys", FAIL,
              f"未找到（找过 {len(disc['nsys'].searched)} 个候选目录，见 --json 的 tools.nsys）",
              impact="缺少核心数据源：无法得到 kernel 时长与 LaunchStats")
        r.ok = False

    cubin = disc["cuobjdump"].path
    r.add("toolchain", "cuobjdump", OK if cubin else WARN,
          (tool_version([cubin, "--version"]) if cubin else
           f"未找到（找过 {len(disc['cuobjdump'].searched)} 个候选目录）")
          + (f"  @ {cubin}" if cubin and disc["cuobjdump"].found_but_not_in_path else ""),
          impact="提供静态寄存器/共享内存与 SASS 指令混合（exact，且不需要 GPU）")

    smi = disc["nvidia-smi"].path
    r.add("toolchain", "nvidia-smi", OK if smi else WARN,
          "present" if smi else "未找到",
          impact="提供设备级计数器：能耗/显存忙碌/时钟功耗（measured，无需 admin）")

    try:
        import pynvml  # type: ignore  # noqa: F401

        r.add("toolchain", "pynvml", OK, "已安装",
              impact="可读能量累加器 → 每 kernel 能耗")
    except Exception:
        r.add("toolchain", "pynvml", WARN, "未安装（可选）",
              impact="装 `kernoscope[nvml]` 可解锁能耗指标")

    paranoid = _read("KS_PERF_PARANOID", "/proc/sys/kernel/perf_event_paranoid").strip()
    if paranoid:
        try:
            val = int(paranoid)
            r.add("toolchain", "CPU perf_event_paranoid", OK if val <= 1 else WARN, f"= {val}",
                  impact="" if val <= 1 else "CPU 侧硬件计数器（perf）对非特权用户不可用")
        except ValueError:
            pass

    # ---- 可选的 instrumentation 后端 ----
    nvbit_hint = os.environ.get("NVBIT_PATH") or _find_nvbit()
    r.nvbit_path = nvbit_hint or ""
    r.add("optional", "NVBit", OK if nvbit_hint else WARN,
          nvbit_hint or "未配置（可选外部依赖，需自行安装，许可证禁止随本项目分发）",
          impact="指令级归因 / 访存轨迹 / sectors-per-request / locality")

    # ---- 计数器门：经验判定 + 四态结论（R-17）----
    ncu_disc = disc["ncu"]
    if probe_counters:
        available, why = counter_gate_probe(ncu_disc.path)
    else:
        # --no-probe：一次实测都不做。probe 传 None 只表示「没有实测证据」，
        # 不等于「没有证据」——驱动策略这条静态证据仍要交给分类器定夺（R-44）。
        available, why = counters_allowed, "静态信息不足（--no-probe，未做实测探测）"
    state, state_note = classify_counter_state(ncu_disc, counters_allowed, available
                                               if probe_counters else None,
                                               static=not probe_counters)
    r.counter_state, r.counter_state_note = state, state_note
    mark = {"available": OK, "denied": FAIL, "not-installed": WARN,
            "not-in-path": WARN, "unknown": WARN}[state]
    impacts = {
        "denied": UNAVAILABLE_REASONS["counters"],
        "not-installed": ("计数器是否可用**无法判定**：所有候选目录里都没有 ncu。"
                          "这不等于「被权限限制」——装上 ncu（或用平台文档确认策略）后重跑 "
                          "doctor 才能定性"),
        "not-in-path": ("ncu 装在非 PATH 目录里；只看 PATH 的工具会误报「未安装」。"
                        "本工具已按多路径找到它"),
        "available": "",
        "unknown": "无法判定：ncu 存在但探测既没报权限错、也没给出可用证据",
    }
    details = {
        "available": f"实测可用 @ {ncu_disc.path}",
        "denied": (f"{ncu_disc.path} 实测被拒：ERR_NVGPUCTRPERM" if probe_counters else
                   f"{ncu_disc.path} 静态推断被拒：驱动策略 RmProfilingAdminOnly=1"
                   f"（--no-probe，未做实测探测）"),
        "not-installed": f"未找到 ncu（已在 {len(ncu_disc.searched)} 个候选目录里确认）",
        "not-in-path": f"已安装但不在 PATH：{ncu_disc.path}",
        "unknown": f"ncu 在 {ncu_disc.path or '?'}，但无法判定：{why}",
    }
    impact = impacts[state]
    if state == "denied" and not probe_counters:
        # 原因文案与报告侧保持一致，但要显式标注这是静态推断，避免被当成实测结论
        impact = (impact + "；本次结论来自静态推断（驱动策略 + 容器 capability），"
                  "未做实测探测；要看实测证据请去掉 --no-probe")
    r.add("counters", "硬件计数器（ncu）", mark, details[state], impact=impact)
    if state == "unknown" and not ncu_disc.found:
        # 没装就当"无法判定"，但把原因说清（不是"被权限限制"）
        available = None

    # ---- 能力矩阵 ----
    caps: list[Capability] = []
    exact = ["kernel 时长与 launch 配置 (LaunchStats)",
             "同一 kernel 的多组 launch 配置（LaunchConfigs，形状扫描用）",
             "理论 occupancy 上界与限制因素"]
    if cubin:
        exact.append("静态寄存器/共享内存、SASS 指令混合")
    caps.append(Capability("exact", exact,
                           "" if nsys else "缺少 nsys：时长与 LaunchStats 不可得"))
    measured = ["NVML 原始读数：显存忙碌占比 / 时钟 / 功耗 / 能量（管理计数器）"]
    if nvbit_hint:
        measured.append("访存轨迹入口（NVBit 外部工具产出的 trace）：sectors/request、"
                        "合并效率、工作集")
    caps.append(Capability("measured", measured))
    # R-13：doctor 的分级必须与报告里实际打出的标签一致
    caps.append(Capability("estimated", [
        "DRAM 字节/吞吐代理（NVML 显存忙碌 × 本机实测峰值，误差带见 bench/artifacts）",
        "roofline 与 SpeedOfLight（需 --ks-calibration 本机标定 + 你提供 FLOPs/bytes）",
        "访存轨迹的 L1/L2 命中率与 DRAM 字节（缓存模型，上界口径）",
    ]))
    caps.append(Capability("derived", [
        "采样窗口覆盖率（由 NVML 采样点统计）",
        "账本对拍误差与区间判词（由估算与你的账本推出）",
    ]))
    counter_items = [
        "MemoryWorkloadAnalysis（精确字节/命中率/sector 效率）",
        "ComputeWorkloadAnalysis（各 pipe 动态利用率）",
        "SchedulerStats / WarpStateStats（stall 周期归因）",
        "SourceCounters（PC 采样）与 PmSampling",
    ]
    if state == "available":
        # 计数器可用时这些指标确实拿得到（来源：ncu / CUPTI 计数器）
        caps.append(Capability("exact", [f"ncu：{i}" for i in counter_items]))
    else:
        reason = UNAVAILABLE_REASONS["counters"] if state in ("denied", "not-in-path") else (
            "ncu 未安装，无法实测判定计数器是否可用（这只是「无法判定」，"
            "不等于「被权限限制」；装 ncu 后重跑 doctor --json 可定性）")
        caps.append(Capability("unavailable", counter_items, reason))
    r.capabilities = caps

    # ---- 后端计划与建议 ----
    plan = []
    if nsys:
        plan.append("nsys → LaunchStats / 时长 / 时间线 (exact)")
    if smi:
        plan.append("NVML/nvidia-smi → 设备级计数器 (measured)")
    if cubin:
        plan.append("cuobjdump → 静态资源与 SASS 混合 (exact)")
    if not plan:
        plan.append("无可用后端：请安装 Nsight Systems（nsys）后重试")
    r.backend_plan = plan

    r.advice = [
        "首次使用先做一次机器标定：`uv run python tools/machine-profile.py --quick`，"
        "然后加 `--ks-calibration machine-profile.json` 得到可信 roofline"
        "（标定会同时记录累加口径与口径离散度；缺字段时 compute roof 会被拒绝，见 R-09）",
        "报告里出现 `n/a` 属预期：那是需要硬件计数器的指标，"
        "用 `--strict` 可让 CI 在请求这类数据时直接失败",
        "想拿账本对拍：`--ks-ledger-mb <每次 launch 的 MB>`（或 --ks-ledger-total-mb 总量口径），"
        "并让被测 kernel 在紧循环里跑满；默认 `--ks-window auto` 会自动裁掉 import/JIT 的启动开销",
        "代理口径不成立（显存忙碌读数为 0、窗口覆盖率不足）时，"
        "用 `--ks-memtrace <trace>` 换用访存轨迹口径做账本对拍（R-33）",
    ]
    if cubin and disc["cuobjdump"].found_but_not_in_path:
        r.advice.insert(0, f"cuobjdump 不在 PATH 上（{cubin}）：本工具已自动找到它，"
                           f"但 `python -m kerneloscope` 之外的工具可能需要 export PATH")
    if state == "not-in-path":
        r.advice.insert(0, f"ncu 已安装但不在 PATH（{ncu_disc.path}）：如果只想手工复核权限，"
                           f"可直接用绝对路径执行 `{ncu_disc.path} --query-metrics`")
    if not nsys:
        r.advice.insert(0, "安装 Nsight Systems（随 CUDA Toolkit 提供，或单独下载）后重试")
    return r


def _find_nvbit() -> str | None:
    """找 NVBit 解包目录（含 core/libnvbit.a）。

    R-47：只在几个**顶层**目录里看一眼会漏掉真实布局。租用实例上常见的是
    `~/<project-data>/lab/nvbit_release_x86_64` 这种"往下两层"的路径，于是能力分层会把
    tier 3（指令级）误判成不可用 —— 而"我找过哪些目录"正是 doctor 要能证明的事。
    这里按深度 1..3 找，且只在这几个基准目录下找（不做全盘遍历）。
    """
    bases = (".", os.path.expanduser("~"), "/root", "/opt", "/usr/local",
             "/workspace", "/mnt")
    patterns = ("nvbit_release*", "*/nvbit_release*", "*/*/nvbit_release*")
    for base in bases:
        for pat in patterns:
            for p in sorted(glob.glob(os.path.join(base, pat))):
                if os.path.exists(os.path.join(p, "core", "libnvbit.a")):
                    return os.path.abspath(p)
    return None


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def render(report: DoctorReport, color: bool = False) -> str:
    out: list[str] = [f"Kernoscope {__version__} 环境自检 (doctor)", ""]
    groups = ["host", "hardware", "policy", "permissions", "toolchain", "optional", "counters"]
    titles = {
        "host": "主机", "hardware": "硬件与驱动", "policy": "内核驱动策略",
        "permissions": "容器权限", "toolchain": "工具链", "optional": "可选后端",
        "counters": "硬件计数器判定",
    }
    for g in groups:
        items = [c for c in report.checks if c.group == g]
        if not items:
            continue
        out.append(f"[{titles.get(g, g)}]")
        for c in items:
            line = f"  {_MARK.get(c.status, '?')} {c.name:<26} {c.detail}"
            out.append(line.rstrip())
            if c.impact:
                out.append(f"      → {c.impact}")
        out.append("")

    out.append("[可交付的度量能力]")
    order = ["exact", "measured", "derived", "estimated", "unavailable"]
    caps = sorted(report.capabilities, key=lambda c: order.index(c.provenance)
                  if c.provenance in order else 99)
    for cap in caps:
        out.append(f"  {cap.provenance:<12}")
        for item in cap.items:
            out.append(f"      · {item}")
        if cap.reason:
            out.append(f"      原因: {cap.reason[:140]}")
    out.append("")

    if report.tools:
        out.append("[工具发现（多路径，不只看 PATH）]")
        for name, t in report.tools.items():
            if t.get("found"):
                loc = t["path"] + ("" if t.get("in_path") else "   (不在 PATH 上)")
            else:
                loc = f"未找到（找过 {len(t.get('searched_dirs') or [])} 个目录）"
            out.append(f"  · {name:<12} {loc}")
        out.append(f"  · 计数器结论: {report.counter_state} —— {report.counter_state_note}")
        out.append("")

    out.append("[后端计划]")
    for p in report.backend_plan:
        out.append(f"  · {p}")
    out.append("")
    out.append("[建议]")
    for a in report.advice:
        out.append(f"  · {a}")
    out.append("")
    out.append(f"结论: {'核心路径可用' if report.ok else '核心路径不可用（见上方 ✗）'}"
               "   （用 --json 获取机器可读结果）")
    return "\n".join(out)


def doctor_main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="kernoscope doctor",
                                 description="环境自检：这台机器上能拿到什么、拿不到什么、为什么")
    ap.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    ap.add_argument("--no-probe", action="store_true",
                    help="不做 ncu 实测探测，只用静态信息推断计数器是否可用")
    args = ap.parse_args(argv)

    rep = run_doctor(probe_counters=not args.no_probe)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render(rep))
    return 0 if rep.ok else 1
