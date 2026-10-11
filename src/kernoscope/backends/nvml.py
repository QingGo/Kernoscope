"""NVML / nvidia-smi 后端：设备级计数器（免权限）。

实测可读（无需 admin）：
  * PCIe RX/TX 吞吐、replay 计数
  * 能量累加器（mJ）→ 每 kernel 能耗（ncu 没有这个指标）
  * 显存忙碌占比 utilization.memory —— 稳态下是 DRAM 带宽的可用代理
    （实测：饱和拷贝 96.8% ↔ 920 GB/s，误差 6%；但它是 200ms 区间平均，
     带宿主间隙时会被稀释成 6.3%，所以必须成对报告占空比/利用率）
  * SM 忙碌占比、时钟、功耗、降频原因位图
  * **窗口的环境证据**（R-61 的延伸，也是"不重放"路线的必要补偿）：
    - 降频原因位图 → "这个数是不是被功率封顶/热降频拖了"
    - GPU 空转占比 → "窗口里 GPU 有多少时间是闲的"（数字被稀释的直接证据）
    - 同卡其他计算进程 → "采样期间有没有别的租户在同卡上跑"
    这三条让"真实时间"的数字带上**可比性前提**：没重放也能说清环境干不干净。

RE 说明：这里的读数全部来自可读的 NVML / nvidia-smi 面（管理计数器，无需 admin）。
`clocks_throttle_reasons.active` 是驱动给的位图，位名按 NVML 公开常量解码；
位图未知位一律如实报 `bitN`，不猜语义。

优先用 pynvml（`pip install kernoscope[nvml]`）；否则退回 `nvidia-smi` 子进程轮询。
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

SMi = "nvidia-smi"

#: 降频原因位图 —— 名字取自 NVML 公开常量（nvidia-smi 的 `clocks_throttle_reasons.active`
#: 是同一个位图）。位 0 是 "GPU 空闲"，不是性能受限，单独计。
THROTTLE_BITS: tuple[tuple[int, str], ...] = (
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks_setting"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
)
IDLE_BIT = 0x1

#: 「位图未知」通道（R-96①②）：驱动只说了 Active 而没给位图，或者这次查询**根本读不到**
#: （pynvml 抛异常 / nvidia-smi 给 `n/a`）。**绝不允许**把它当成 0（＝"没有降频"）——
#: 那正是第六轮 W-1 的假绿：`env_no_throttle_pct observed=0.0 ok=True` ⇒
#: `cost_per_token` 保持 `actionable=True`。未知一律走这个负数，且要**计入受限**。
THROTTLE_UNKNOWN = -1


def decode_throttle(mask: int) -> list[str]:
    """位图 → 原因名（**已知位**给名字；未知位照实给 `bitN`，不猜）。

    负数是「未知」通道（`THROTTLE_UNKNOWN`）：没有位图可解，如实给 `unspecified`，
    既不是"干净"（空列表）也不猜位名。**负数必须先挡住**：旧实现直接进位移循环，
    `-1 >> 1` 永远是 `-1` ⇒ 死循环（第六轮 W-13，真机上够不到、但一次调用就挂死）。
    """
    if mask < 0:
        return ["unspecified"]
    out = [name for bit, name in THROTTLE_BITS if mask & bit]
    known = 0
    for bit, _ in THROTTLE_BITS:
        known |= bit
    unknown = mask & ~known
    n = 0
    while unknown:
        if unknown & 1:
            out.append(f"bit{n}")
        unknown >>= 1
        n += 1
    return out


def _is_descendant(pid: int, roots: set[int], max_depth: int = 16) -> bool:
    """`pid` 是不是 `roots` 里某个进程的后代（用来把**我们自己的负载**算成"自己人"）。

    只读 `/proc/<pid>/stat` 的 ppid 链；读不到就返回 False（宁可用保守的"算作别人"，
    也不要静默把真实的多租户吃掉）。
    """
    cur = pid
    seen: set[int] = set()
    for _ in range(max_depth):
        if cur in roots:
            return True
        if cur in (0, 1) or cur in seen:
            return False
        seen.add(cur)
        try:
            with open(f"/proc/{cur}/stat", "rb") as f:
                data = f.read().decode("utf-8", "replace")
        except Exception:
            return False
        try:
            rest = data[data.rfind(")") + 2:].split()
            cur = int(rest[1])                      # state 之后是 ppid
        except Exception:
            return False
    return False


def parse_throttle_token(text: str) -> int:
    """``"0x0000000000000001"`` / ``"Not Active"`` / ``"Active"`` → 位图。

    驱动器版本不同，`--query-gpu` 对同一字段可能给十六进制位图也可能给 Active/Not Active；
    两种都认。**读不到的一律返回 `THROTTLE_UNKNOWN`（-1）**，不返回 0：
    R-96② —— `n/a` 过去被映射成 `0`（＝"没有降频"），于是"读不到"混进了"干净"。
    只有驱动**明确说没有**（`Not Active`/`false`/`no`/`0`）才是 0。
    """
    t = (text or "").strip()
    if not t:
        return THROTTLE_UNKNOWN           # 空 = 没读到，不是"没降频"
    low = t.lower()
    if low.startswith("0x"):
        try:
            return int(low, 16)
        except ValueError:
            return THROTTLE_UNKNOWN
    if low in ("active", "true", "yes"):
        return THROTTLE_UNKNOWN           # 有降频，但驱动没给位图
    if low in ("not active", "false", "no", "0"):
        return 0
    try:
        return int(low)
    except ValueError:
        return THROTTLE_UNKNOWN           # 认不出来的 token：未知，不许降级成 0


@dataclass
class DeviceSample:
    t: float
    mem_util_pct: float
    sm_util_pct: float
    sm_clock_mhz: float
    power_w: float
    #: 降频原因位图（`THROTTLE_UNKNOWN`(-1) = 位图未知：驱动只说"有降频"或这次读不到；
    #: 0 = 驱动明确说没降频）
    throttle_mask: int = 0


@dataclass
class SamplerResult:
    samples: list[DeviceSample] = field(default_factory=list)
    energy_j: float = 0.0
    duration_s: float = 0.0
    source: str = "nvidia-smi"
    #: 采样期间该卡上观测到的其他计算进程数（最大值）＋是否探测成功
    other_procs: int = 0
    other_procs_probed: bool = False
    #: 同卡其他进程探测的**结果状态**（R-96④）：`not_run` / `ok` / `failed`。
    #: 为什么不能只看 `other_procs_probed`：探测失败与"根本没测过"过去都是同一个
    #: `False`，包里看不出"多租户这条没测"；而"没测"≠"0 个别的租户"。
    other_procs_probe: str = "not_run"
    #: 设备标称最高 SM 时钟（nvidia-smi clocks.max.sm；拿不到就是 0）
    max_sm_clock_mhz: float = 0.0

    def _col(self, attr: str) -> list[float]:
        return [getattr(s, attr) for s in self.samples]

    def _active(self) -> list[DeviceSample]:
        """"有活在干"的采样点：SM 利用率 > 0 且驱动没有标 gpu_idle。

        为什么要筛：空闲时驱动把 SM 时钟降到 210 MHz 是正常行为，把它算进"驻留"会得出
        "一直在低频跑"这种没有信息量的结论（真机第一次跑就撞上：p50 = 210 MHz）。
        闲的比例由 `gpu_idle_pct` 单独报，两件事不混。
        """
        return [s for s in self.samples
                if s.sm_util_pct > 0 and not (0 <= s.throttle_mask and s.throttle_mask & IDLE_BIT)]

    def clock_residency(self) -> dict[str, float]:
        """时钟驻留：干活的时候有没有在峰值频率上跑，还是被压下来了。

        统计只在**活跃采样点**上做（见 `_active`）。`ref_mhz` 优先用设备标称最高时钟
        （`clocks.max.sm`），拿不到就用窗口内观测最高时钟。`p50_all_mhz` 同时给出，
        便于和空闲时钟对照。
        """
        clocks = self._col("sm_clock_mhz")
        if not clocks:
            return {}
        act = self._active()
        ref = self.max_sm_clock_mhz or max(clocks)
        base = [s.sm_clock_mhz for s in act] or clocks
        at_max = sum(1 for c in base if ref and c >= 0.99 * ref)
        return {
            "min_mhz": min(base),
            "p50_mhz": statistics.median(base),
            "p50_all_mhz": statistics.median(clocks),
            "max_mhz": max(clocks),
            "ref_mhz": ref,
            "at_max_pct": 100.0 * at_max / len(base),
            "active_samples": float(len(act)),
            "samples": float(len(clocks)),
        }

    def throttle_profile(self) -> dict[str, object]:
        """降频画像：把"位图"变成"窗口里有多少时间被什么限制"。

        `gpu_idle` 单独算（它是"没活干"，不是"被限制"）。
        """
        if not self.samples:
            return {}
        total = len(self.samples)
        reasons: dict[str, int] = {}
        idle = 0
        limited = 0
        for s in self.samples:
            mask = s.throttle_mask
            if mask < 0:                       # 位图未知（读不到 / 驱动只说 Active）
                # **不许**把它当成"没有降频"（R-96①）：算进受限是保守方向 —— 前提
                # `env_no_throttle_pct` 必须因此不成立，而不是 observed=0.0 ok=True。
                reasons["unspecified"] = reasons.get("unspecified", 0) + 1
                limited += 1
                # 兜底（R-96③）：位图拿不到时用真实利用率判空闲。否则"真空闲"会被算成
                # idle_pct=0.0 ⇒ `env_gpu_busy_pct observed=100.0 ok=True`（另一处假绿）。
                if s.sm_util_pct <= 0.0:
                    idle += 1
                continue
            if mask & IDLE_BIT:
                idle += 1
            non_idle = mask & ~IDLE_BIT
            if non_idle:
                limited += 1
                for name in decode_throttle(non_idle):
                    reasons[name] = reasons.get(name, 0) + 1
        return {
            "samples": total,
            "idle_pct": 100.0 * idle / total,
            "limited_pct": 100.0 * limited / total,
            "reasons": {k: 100.0 * v / total for k, v in sorted(reasons.items())},
        }

    def environment(self) -> dict[str, object]:
        """窗口的环境证据（进 section 与决策包前提）。

        "这条没测"与"这条测到了 0"必须是两个样子（R-96④）：探测失败写
        `other_procs_probe="failed"` 且**不写** `other_procs`，让下游能报"未知"而不是 0。
        """
        out: dict[str, object] = {}
        res = self.clock_residency()
        thr = self.throttle_profile()
        if res:
            out["clock"] = res
        if thr:
            out["throttle"] = thr
        if self.other_procs_probed:
            out["other_procs"] = self.other_procs
            out["other_procs_probe"] = "ok"
        else:
            # not_run / failed：两者都能与"0 个别的租户"区分开
            out["other_procs_probe"] = self.other_procs_probe
        out["sampler"] = self.source
        out["interval_s"] = round(self.duration_s / max(1, len(self.samples)), 3) \
            if self.samples else 0.0
        return out

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        out: dict[str, float] = {}
        for attr, key in (("mem_util_pct", "mem_util"), ("sm_util_pct", "sm_util"),
                          ("sm_clock_mhz", "sm_clock"), ("power_w", "power")):
            vals = self._col(attr)
            out[f"{key}_mean"] = statistics.fmean(vals)
            out[f"{key}_max"] = max(vals)
            out[f"{key}_min"] = min(vals)
        out["samples"] = float(len(self.samples))
        res = self.clock_residency()
        if res:
            out["sm_clock_p50"] = res["p50_mhz"]
            out["clock_at_max_pct"] = res["at_max_pct"]
        thr = self.throttle_profile()
        if thr:
            out["throttle_limited_pct"] = thr["limited_pct"]
            out["gpu_idle_pct"] = thr["idle_pct"]
        if self.other_procs_probed:
            out["other_procs"] = float(self.other_procs)
        if self.energy_j:
            out["energy_j"] = self.energy_j
            if self.duration_s:
                out["avg_power_w"] = self.energy_j / self.duration_s
        # DRAM 带宽代理：util% × 实测峰值带宽（调用方填 peak_gbs 后可用）
        out["mem_busy_pct_mean"] = out["mem_util_mean"]
        return out


class Sampler:
    """后台采样设备级指标（context manager）。"""

    def __init__(self, interval_s: float = 0.2, enabled: bool = True,
                 exclude_pids: set[int] | None = None, device_index: int = 0):
        self.interval_s = interval_s
        self.enabled = enabled
        #: 采样哪张卡（pynvml 的 handle 与 nvidia-smi 的 `--id` 必须指同一个，R-96⑤）
        self.device_index = device_index
        #: 额外要排除的进程（例如我们自己拉起的负载进程树）
        self.exclude_pids: set[int] = set(exclude_pids or ())
        self.result = SamplerResult()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None

    # ---- pynvml（可选，提供能量累加器）----
    def _try_pynvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.result.source = "pynvml"
        except Exception:
            self._pynvml = None

    def _energy_j(self) -> float:
        if not self._pynvml:
            return 0.0
        try:
            return self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) / 1000.0
        except Exception:
            return 0.0

    def _max_sm_clock(self) -> float:
        if self._pynvml:
            try:
                return float(self._pynvml.nvmlDeviceGetMaxClockInfo(
                    self._handle, self._pynvml.NVML_CLOCK_SM))
            except Exception:
                pass
        proc = subprocess.run([SMi, "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits",
                               f"--id={self.device_index}"],
                              capture_output=True, text=True)
        try:
            return float(proc.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    def _other_procs(self) -> int | None:
        """同卡**其他**（别人的）计算进程数；探测失败返回 None（不当作 0）。

        注意要把**我们自己拉起的负载进程树**排除掉：否则 `kernoscope ... -- python bench.py`
        会把 bench.py 数成"另一个租户"（真机第一次跑就撞上：报了 1 个）。

        R-96⑤：退回 `nvidia-smi` 时**必须带 `--id=<本卡>`**。不带 `--id` 的
        `--query-compute-apps` 会把**别的卡**上的进程也算进来 —— note 却写"同卡"，
        于是多卡机器上结论被静默误降级（方向保守，但不是我们声称的口径）。
        pynvml 那条路本来就是 handle 级（`_handle`），天然同卡。
        """
        mine = {os.getpid(), os.getppid()} | set(self.exclude_pids)
        if self._pynvml:
            try:
                procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                pids = [int(getattr(p, "pid", 0)) for p in procs]
                return sum(1 for p in pids if p and not _is_descendant(p, mine))
            except Exception:
                pass
        proc = subprocess.run([SMi, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
                               f"--id={self.device_index}"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        try:
            pids = [int(x.strip()) for x in proc.stdout.replace("\n", ",").split(",") if x.strip()]
        except ValueError:
            return None
        return sum(1 for p in pids if not _is_descendant(p, mine))

    def __enter__(self) -> "Sampler":
        if not self.enabled or not shutil.which(SMi):
            self.enabled = False
            return self
        self._try_pynvml()
        self.result.max_sm_clock_mhz = self._max_sm_clock()
        self._e0 = self._energy_j()
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.result.duration_s = time.monotonic() - self._t0
        if self._pynvml:
            self.result.energy_j = max(0.0, self._energy_j() - self._e0)

    def _loop(self) -> None:
        n = 0
        while not self._stop.wait(self.interval_s):
            s = self._read()
            if s:
                self.result.samples.append(s)
            # 其他租户探测：每 5 个采样点一次（一次探测要 ~50 ms 子进程），取窗口内的最大值
            if n % 5 == 0:
                others = self._other_procs()
                if others is None:
                    # 探测失败 = **未知**（R-96④）：与"从没测过"分开记，也绝不冒充 0 个租户。
                    # 之前成功过就保留成功状态（我们手里确实有数，没必要倒回去说"未知"）。
                    if not self.result.other_procs_probed:
                        self.result.other_procs_probe = "failed"
                else:
                    self.result.other_procs_probed = True
                    self.result.other_procs_probe = "ok"
                    self.result.other_procs = max(self.result.other_procs, others)
            n += 1

    def _read(self) -> DeviceSample | None:
        if self._pynvml:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                clk = self._pynvml.nvmlDeviceGetClockInfo(self._handle, self._pynvml.NVML_CLOCK_SM)
                pwr = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                try:
                    mask = int(self._pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle))
                except Exception:
                    # R-96①（第六轮 W-1，P0）：读不到 ≠ 没有降频。旧写法 `mask = 0` 把
                    # "查询抛异常"写成"窗口内没有降频"⇒ `env_no_throttle_pct observed=0.0
                    # ok=True` ⇒ `cost_per_token` 保持 `actionable=True`。**这里不能丢样本**：
                    # 丢掉会把本可读的利用率/时钟/功耗一起丢掉，环境前提整条消失，反而更绿。
                    # 保留样本 + 位图记未知（-1），由 throttle_profile 计入"受限"。
                    # nvidia-smi 分支对 `n/a` 走同一条未知通道（见 parse_throttle_token）。
                    mask = THROTTLE_UNKNOWN
                return DeviceSample(time.monotonic(), float(util.memory), float(util.gpu),
                                    float(clk), float(pwr), mask)
            except Exception:
                return None
        q = ("utilization.memory,utilization.gpu,clocks.sm,power.draw,"
             "clocks_throttle_reasons.active")
        # `--id`：不带它时多卡机器会吐多行 → 解析失败 → **静默丢光所有采样**（与 pynvml 的
        # `nvmlDeviceGetHandleByIndex(self.device_index)` 指同一张卡）。
        proc = subprocess.run([SMi, f"--query-gpu={q}", "--format=csv,noheader,nounits",
                               f"--id={self.device_index}"],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            mem, sm, clk, pwr, thr = [x.strip() for x in proc.stdout.strip().split(",")]
            return DeviceSample(time.monotonic(), float(mem), float(sm), float(clk), float(pwr),
                                parse_throttle_token(thr))
        except Exception:
            return None
