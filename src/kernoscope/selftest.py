"""`kernoscope selftest` —— 纯 torch 的已知答案负载，量出**本机**误差带与 MDD（W6）。

为什么需要它：整个工具的诚实性建立在"我们知道自己的误差有多大"之上。而误差是
**逐机**的（同一份代码在 L2 更大的卡上代理偏差不同），所以不能写死在文档里。

这个命令做三件事，只依赖 torch（不需要 nsys / nvcc / NVBit）：
  1. **已知答案**：四类负载的搬运字节数/bound 由张量形状解析可得（copy=2×T×4、
     read=T×4、write=T×4、matmul=(MK+KN+MN)×itemsize），于是"我们测出来的"可以
     与"物理上应该是的"直接对拍 —— 这是唯一不依赖外部真值的自证方式。
  2. **误差带**：把每类负载的偏差并成一个区间（low…high），随包一起走。
  3. **MDD（最小可分辨差异）**：同一负载重复 N 次，取重复间的离散度 +
     仪器分辨率下限（NVML 显存忙碌是整数百分比），得到"两次运行差多少才算真差别"。
     `kernoscope diff` 没有它就只能报"变了"，而不是"真的变了 / 还是噪声"。

输出文件同时是**两种输入**：
  * `--ks-selfcheck machine-selfcheck.json` → 判词带误差带与 MDD；
  * `--ks-calibration machine-selfcheck.json` → 至少带宽侧的 roofline 可用
    （matmul 峰值也测，并如实标注口径是 torch.matmul）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .backends.nvml import Sampler
from .refusals import Refusal, RefusalCode

SELFTEST_SCHEMA_VERSION = 1
DEFAULT_SIZE_MB = 256
DEFAULT_DURATION_S = 2.0
DEFAULT_RUNS = 3
MIN_SATURATION_MEM_UTIL_PCT = 50.0     # 低于它说明负载没把显存打满，误差带不可信
WORKLOAD_IDS = ("copy", "read", "write", "matmul")
# 只有 DRAM-bound 的负载能做"已知答案"对拍：matmul 是 compute-bound，DRAM 忙碌只有
# 20%~40%，用"忙碌时间×峰值"去折算必然高估 —— 那不是误差，是口径不适用（真机实测
# 偏差 +800%…+1600%，正是这个原因）。所以它只作为**负样本**报告，不进误差带。
DRAM_BOUND_IDS = ("copy", "read", "write")
# 采样前先跑一段：NVML 的 utilization.memory 是窗口平均，刚开始的 1~4 个采样点会
# 读到上一段空闲的 0%（真机实测：不预热会让 copy 的偏差从 +9% 变成 +72%）。
WARMUP_S = 1.0
ITEM_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}
# torch 的 dtype 属性名和我们的短名不是一回事：`torch.bf16` 不存在，`torch.bfloat16` 才存在。
# 这个映射被真机踩过一次（当时错误信息还说成"显存不足"，更难查）。
TORCH_DTYPE_ATTR = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


@dataclass
class WorkloadRun:
    id: str
    analytic_bytes_per_iter: float
    measured_bytes_per_iter: float
    err_pct: float
    mem_util_pct: float
    duration_s: float
    iters: int
    kernel_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelftestResult:
    ok: bool = False
    refusal: Refusal | None = None
    device: str = ""
    compute_capability: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    dtype: str = "bf16"
    size_mb: int = DEFAULT_SIZE_MB
    duration_s: float = DEFAULT_DURATION_S
    runs: int = DEFAULT_RUNS
    workloads: list[WorkloadRun] = field(default_factory=list)
    peaks: dict[str, Any] = field(default_factory=dict)
    band: dict[str, Any] = field(default_factory=dict)
    mdd: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ---- 派生 ----
    def by_id(self, wid: str) -> list[WorkloadRun]:
        return [w for w in self.workloads if w.id == wid]

    def render(self) -> str:
        out: list[str] = [f"Kernoscope {__version__} 本机自检（selftest）"]
        if self.refusal is not None:
            out.append("")
            out.append(f"没能完成自检：{self.refusal.code.value} — {self.refusal.message}")
            rem = self.refusal.remedy_or_default
            if rem.action:
                out.append(f"怎么办：{rem.action}"
                           + (f"  [{rem.command}]" if rem.command else ""))
            out.append("")
            out.append("自检需要：一块 NVIDIA GPU + NVML（nvidia-smi/pynvml）+ PyTorch。"
                       "缺任何一项都无法给出本机误差带 —— 我们不猜。")
            return "\n".join(out)
        out.append(f"  设备 {self.device}（sm_{str(self.compute_capability).replace('.', '')}）"
                   f" · torch {self.torch_version} / CUDA {self.cuda_version}")
        out.append(f"  口径：{self.size_mb} MB 张量 · 每类负载 {self.runs} 次 × "
                   f"{self.duration_s:g} s 饱和循环 · {self.dtype}")
        out.append("")
        out.append("已知答案对拍（物理上应该是多少 vs 我们测出来多少）：")
        out.append(f"  {'负载':<8}{'解析字节/次':>14}{'实测字节/次':>14}{'偏差':>9}"
                   f"{'显存忙碌':>10}")
        for w in self.workloads:
            out.append(f"  {w.id:<8}{w.analytic_bytes_per_iter / 1e6:>11.1f} MB"
                       f"{w.measured_bytes_per_iter / 1e6:>11.1f} MB"
                       f"{w.err_pct:>+8.1f}%{w.mem_util_pct:>9.1f}%")
        b = self.band.get("dram_bytes_pct") or {}
        if b:
            out.append("")
            out.append(f"本机误差带（DRAM 字节代理，只用 {b.get('basis', '')}）："
                       f"{b.get('low'):+.1f}% … {b.get('high'):+.1f}%   "
                       f"← 单边偏高是代理口径的已知性质，不是 bug")
            ex = (self.band.get("excluded") or {}).get("matmul")
            if ex:
                out.append(f"  （matmul 不进误差带：{ex}）")
        m = self.mdd or {}
        if m:
            out.append(f"MDD（两次运行差多少才算真差别）：DRAM 字节 "
                       f"{m.get('dram_bytes_pct')}% · kernel 时长 "
                       f"{m.get('kernel_time_pct')}%")
            out.append(f"  口径：{m.get('basis')}"
                       f"（本次 SD={m.get('run_to_run_sd_pct')}% · "
                       f"分辨率下限={m.get('resolution_floor_pct')}% · "
                       f"误差带半宽={m.get('band_half_width_pct')}%）")
            if m.get("caveat"):
                out.append(f"  适用范围：{m['caveat']}")
        if self.peaks:
            out.append("")
            out.append("顺带量到的本机峰值（可直接当 --ks-calibration 用）："
                       f"copy {self.peaks.get('copy_gbs', 0):.0f} GB/s"
                       + (f" · read {self.peaks.get('read_gbs', 0):.0f} GB/s"
                          if self.peaks.get("read_gbs") else "")
                       + (f" · matmul {self.peaks.get('matmul_tflops', 0):.1f} TFLOP/s"
                          if self.peaks.get("matmul_tflops") else ""))
        for n in self.notes:
            out.append(f"注：{n}")
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": SELFTEST_SCHEMA_VERSION,
            "tool": "kernoscope",
            "tool_version": __version__,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": self.device,
            "compute_capability": self.compute_capability,
            "torch": self.torch_version,
            "cuda": self.cuda_version,
            "protocol": {"dtype": self.dtype, "size_mb": self.size_mb,
                         "duration_s": self.duration_s, "runs": self.runs,
                         "workloads": list(WORKLOAD_IDS)},
            "ok": self.ok,
            "workloads": [w.to_dict() for w in self.workloads],
            "band": self.band,
            "mdd": self.mdd,
            "notes": self.notes,
            # 与 tools/machine-profile.py 同构的字段，让这个文件也能当标定文件用
            "measured_dram": {"copy_gbs": self.peaks.get("copy_gbs"),
                              "read_gbs": self.peaks.get("read_gbs"),
                              "write_gbs": self.peaks.get("write_gbs"),
                              "copy_method": "kernoscope selftest（饱和循环 + NVML 显存忙碌）"},
        }
        if self.peaks.get("matmul_tflops"):
            d["measured_matmul_tflops"] = {
                "torch.bfloat16" if self.dtype == "bf16" else f"torch.{self.dtype}": {
                    "tflops": self.peaks["matmul_tflops"],
                    "acc_caliber": "default",
                    "precision_settings": "torch.matmul 默认（selftest 未做累加口径扫描）",
                    "method": "torch.matmul（selftest）",
                }
            }
        if self.refusal is not None:
            d["refusal"] = self.refusal.to_dict()
        return d


# --------------------------------------------------------------------------
# 负载：解析字节数与执行
# --------------------------------------------------------------------------
def _allocate(torch, size_mb: int, dtype: str):
    """按 MB 分配三个张量（拷贝类负载需要 2 个输入 1 个输出）。"""
    n = max(1, int(size_mb * 2 ** 20 // ITEM_BYTES[dtype]))
    mk = getattr(torch, TORCH_DTYPE_ATTR[dtype])
    a = torch.empty(n, dtype=mk, device="cuda").normal_()
    b = torch.empty(n, dtype=mk, device="cuda").normal_()
    c = torch.empty(n, dtype=mk, device="cuda")
    return a, b, c, n * ITEM_BYTES[dtype]


def _analytic_bytes(wid: str, nbytes: int, mm: tuple[int, int, int, int]) -> float:
    if wid in ("copy",):
        return 2.0 * nbytes          # 读一份 + 写一份
    if wid in ("read", "write"):
        return 1.0 * nbytes
    m, k, n, itemsize = mm
    return float((m * k + k * n + m * n) * itemsize)


def _one_iter(wid: str, t, a, b, c, mm: tuple[int, int, int, int]):
    if wid == "copy":
        c.copy_(a)
    elif wid == "read":
        a.sum()
    elif wid == "write":
        b.fill_(1.0)
    else:
        m, k, n, _ = mm
        am = a[: m * k].view(m, k)
        bm = b[: k * n].view(k, n)
        cm = c[: m * n].view(m, n)
        t.matmul(am, bm, out=cm)


def _spin(t, wid: str, seconds: float, a, b, c, mm) -> tuple[int, float]:
    """跑一段饱和循环，返回 (迭代数, 实测秒数)。迭代数按 20 个一批计数。"""
    t0 = time.monotonic()
    iters = 0
    while time.monotonic() - t0 < seconds:
        for _ in range(20):
            _one_iter(wid, t, a, b, c, mm)
        t.cuda.synchronize()
        iters += 20
    return iters, time.monotonic() - t0


def _window(sampler: Sampler, mark: float | None) -> list:
    return [s for s in sampler.result.samples if mark is None or s.t >= mark]


def _measure_one(t, sampler: Sampler, wid: str, args, a, b, c, nbytes,
                 mm: tuple[int, int, int, int], peak_gbs: float) -> WorkloadRun:
    """跑一段饱和循环，用 NVML 采样估字节，并与解析值对拍。"""
    from .analyze.dram import estimate as estimate_dram

    _spin(t, wid, min(WARMUP_S, max(args.duration_s, 0.5)), a, b, c, mm)  # 预热（不计数）
    mark = sampler.result.samples[-1].t if sampler.result.samples else None
    iters, elapsed = _spin(t, wid, args.duration_s, a, b, c, mm)
    samples = _window(sampler, mark)
    # 与主流程**同一套算法**（同样的窗口自动裁剪、同样的末点外推），因为我们量的是
    # "本工具自己的误差"，不是"一个理想算法的误差"。
    est = estimate_dram(samples, peak_gbs=peak_gbs, window_s=elapsed, launches=iters,
                        window_mode="full")
    measured_per_iter = (est.bytes_total / iters) if iters else 0.0
    analytic = _analytic_bytes(wid, nbytes, mm)
    err = 100.0 * (measured_per_iter / analytic - 1.0) if analytic else 0.0
    return WorkloadRun(
        id=wid, analytic_bytes_per_iter=analytic,
        measured_bytes_per_iter=measured_per_iter, err_pct=round(err, 2),
        mem_util_pct=round(est.mem_util_pct_mean, 2), duration_s=round(elapsed, 3),
        iters=iters, kernel_ms=round(1000.0 * elapsed / max(iters, 1), 4),
    )


def _measure_matmul_tflops(t, a, b, c, mm, seconds: float = 1.5) -> float:
    m, k, n, _ = mm
    flops = 2.0 * m * k * n
    for _ in range(3):
        _one_iter("matmul", t, a, b, c, mm)
    t.cuda.synchronize()
    start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
    start.record()
    iters = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for _ in range(10):
            _one_iter("matmul", t, a, b, c, mm)
        iters += 10
    end.record()
    t.cuda.synchronize()
    ms = start.elapsed_time(end)
    return flops * iters / (ms / 1000.0) / 1e12 if ms > 0 else 0.0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_selftest(size_mb: int = DEFAULT_SIZE_MB, duration_s: float = DEFAULT_DURATION_S,
                 runs: int = DEFAULT_RUNS, dtype: str = "bf16",
                 only: tuple[str, ...] = WORKLOAD_IDS,
                 cross_process: int = 0) -> SelftestResult:
    """跑已知答案自检。

    `only` 限定负载（跨进程子测量只跑 DRAM-bound 的三个，省时间）；
    `cross_process`>0 时另起 N 个**新进程**重测一遍，用来把"跨进程漂移"计进 MDD ——
    只测同一进程内的重复会系统性低估噪声（真机实测：同进程 1.0%，换进程可达数个百分点）。
    """
    res = SelftestResult(
        dtype=dtype, size_mb=size_mb, duration_s=duration_s, runs=runs)
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        res.refusal = Refusal(RefusalCode.PYTORCH_MISSING,
                              f"没有 PyTorch（{e}）：自检的已知答案负载只需要 torch，"
                              f"装了就能跑")
        return res
    res.torch_version = getattr(torch, "__version__", "")
    res.cuda_version = getattr(getattr(torch, "version", None), "cuda", "") or ""
    if not torch.cuda.is_available():
        res.refusal = Refusal(RefusalCode.NO_GPU, "torch 说没有可用的 CUDA 设备")
        return res
    try:
        props = torch.cuda.get_device_properties(0)
        res.device = props.name
        res.compute_capability = f"{props.major}.{props.minor}"
    except Exception:  # noqa: BLE001
        res.device = "unknown"

    class _Args:
        pass

    args = _Args()
    args.duration_s = duration_s
    args.runs = runs

    sampler = Sampler(enabled=True)
    with sampler:
        time.sleep(0.5)                    # 给采样线程一点时间；没有 NVML 时它是空的
        if not sampler.result.samples:
            res.refusal = Refusal(
                RefusalCode.NO_NVML_SAMPLES,
                "没有 NVML 采样点：自检靠「显存忙碌时间」把读数折算成字节，"
                "没有它就只能报 0 字节 —— 与其给一个假误差带，不如不给")
            return res
        try:
            a, b, c, nbytes = _allocate(torch, size_mb, dtype)
        except Exception as e:  # noqa: BLE001
            # 不说成"显存不足"：真实原因可能是 dtype 名字、驱动、被别的进程占满 ——
            # 归因错了比不归因更糟（这是本项目反复踩到的坑）。
            res.refusal = Refusal(RefusalCode.NO_GPU,
                                  f"无法分配 {size_mb} MB × 3 张量（dtype={dtype}）："
                                  f"{type(e).__name__}: {e}")
            return res
        mm = _matmul_shape(nbytes, dtype)
        # 自标定峰值带宽（copy），再用它把 NVML 忙碌时间折算成字节
        peak_gbs = _measure_peak_gbs_live(torch, sampler, args, a, b, c, nbytes, mm)
        res.peaks["copy_gbs"] = round(peak_gbs, 1)
        res.peaks["peak_method"] = "饱和 copy 循环 + NVML 显存忙碌折算"
        for wid in [w for w in WORKLOAD_IDS if w in only]:
            for _ in range(runs):
                run = _measure_one(torch, sampler, wid, args, a, b, c, nbytes, mm, peak_gbs)
                res.workloads.append(run)
        for wid, key in (("copy", "copy_gbs"), ("read", "read_gbs"),
                         ("write", "write_gbs")):
            got = [w for w in res.workloads if w.id == wid]
            if got:
                # 每个负载的**峰值** = 解析字节÷（忙碌时间）：必须乘上迭代数，
                # 否则算出来的是"每次搬运的字节 ÷ 秒"（真机第一次跑就错在这里，
                # 报出 0 GB/s）。
                res.peaks[key] = round(statistics.fmean(
                    w.analytic_bytes_per_iter * w.iters / max(w.duration_s, 1e-9)
                    / max(w.mem_util_pct / 100.0, 1e-6) for w in got) / 1e9, 1)
        res.peaks["matmul_tflops"] = round(_measure_matmul_tflops(torch, a, b, c, mm), 2)

    _fill_band_and_mdd(res, extra_runs=_cross_process_samples(res, cross_process, only))
    res.ok = True
    return res


def _cross_process_samples(res: SelftestResult, n: int,
                           only: tuple[str, ...]) -> dict[str, list[float]]:
    """另起 n 个进程重测同一批负载，返回 {workload_id: [err_pct, ...]}（跨进程样本）。

    子进程失败不影响主结果：如实记进 notes（"跨进程样本拿不到"比"假装没有漂移"诚实）。
    """
    out: dict[str, list[float]] = {}
    if n <= 0:
        return out
    import subprocess

    wl = ",".join(w for w in DRAM_BOUND_IDS if w in only)
    for i in range(n):
        # `--cross-process 0` 是关键：子进程若也默认起子进程，就会指数爆炸
        # （真机第一次跑时 7 个进程同时跑、永不结束）。
        cmd = [sys.executable, "-m", "kernoscope", "selftest", "--json", "--quiet",
               "--cross-process", "0", "--only", wl, "--size-mb", str(res.size_mb),
               "--duration", str(res.duration_s), "--runs", "2", "--dtype", res.dtype]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                res.notes.append(f"跨进程第 {i + 1} 次测量失败（rc={proc.returncode}）："
                                 f"MDD 只含同进程重复的离散度，偏乐观")
                continue
            d = json.loads(proc.stdout)
            for w in d.get("workloads", []):
                out.setdefault(w["id"], []).append(float(w["err_pct"]))
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            res.notes.append(f"跨进程第 {i + 1} 次测量失败（{type(e).__name__}）："
                             f"MDD 只含同进程重复的离散度，偏乐观")
    return out


def _matmul_shape(nbytes: int, dtype: str) -> tuple[int, int, int, int]:
    """把一维缓冲区折成一个方阵乘法形状（M=K=N，尽量用满显存）。"""
    import math

    itemsize = ITEM_BYTES[dtype]
    dim = max(256, int(math.isqrt(nbytes // itemsize)))
    return dim, dim, dim, itemsize


def _measure_peak_gbs_live(torch, sampler, args, a, b, c, nbytes, mm) -> float:
    """自标定：饱和 copy 的解析字节 ÷（NVML 忙碌时间）＝ 峰值带宽。

    注意这里**假设 copy 打满了 DRAM**（4090 上实测 mem_util=100%、达成 ≈920 GB/s）。
    假设不成立时自检会在 note 里报"显存忙碌低于 50%"，误差带随即不可信。
    """
    from .analyze.dram import estimate as estimate_dram

    _spin(torch, "copy", max(WARMUP_S, 1.0), a, b, c, mm)
    mark = sampler.result.samples[-1].t if sampler.result.samples else None
    iters, elapsed = _spin(torch, "copy", max(args.duration_s, 2.0), a, b, c, mm)
    samples = _window(sampler, mark)
    est = estimate_dram(samples, peak_gbs=1.0, window_s=elapsed, launches=iters,
                        window_mode="full")
    busy_frac = max(est.mem_util_pct_mean / 100.0, 1e-6)
    return _analytic_bytes("copy", nbytes, mm) * iters / (busy_frac * elapsed) / 1e9


def _fill_band_and_mdd(res: SelftestResult,
                       extra_runs: dict[str, list[float]] | None = None) -> None:
    """误差带 = DRAM-bound 负载偏差的并集；MDD = 三者取大：

      ① 同负载重复（含跨进程）的 2×SD —— 噪声
      ② NVML 1 个百分点的分辨率下限 —— 仪器
      ③ 本机误差带的半宽 —— 不同访问模式下代理**偏差**的差别。
         diff 比较的两次运行不一定跑的是同一个 kernel，所以偏差差也必须算进 MDD，
         否则会把"换了个 kernel 导致偏差变了 3 个百分点"判成"真退化"。

    `extra_runs` 是跨进程样本（{id: [err_pct,...]}）。
    """
    if not res.workloads:
        return
    extra = extra_runs or {}
    bound = [w for w in res.workloads if w.id in DRAM_BOUND_IDS]
    errs = [w.err_pct for w in bound]
    res.band = {"dram_bytes_pct": {
        "low": round(min(errs), 1), "high": round(max(errs), 1),
        "basis": f"{'/'.join(DRAM_BOUND_IDS)} 的已知答案偏差并集（DRAM-bound 负载）",
        "n": len(errs)},
        "excluded": {"matmul": "compute-bound：显存忙碌只有 20%~40%，"
                               "「忙碌时间×峰值」口径不适用（偏差可达 +1000%，"
                               "这不是测量误差而是口径错用）"}}
    # DRAM 字节的 MDD：同负载重复间的 SD ×2（近似 95%），再加分辨率下限
    sds: list[float] = []
    floors: list[float] = []
    for wid in DRAM_BOUND_IDS:
        runs = res.by_id(wid)
        vals = [r.err_pct for r in runs] + list(extra.get(wid, []))
        if len(vals) >= 2:
            sds.append(statistics.stdev(vals))
        for r in runs:
            if r.mem_util_pct > 0:
                floors.append(100.0 * 1.0 / r.mem_util_pct)   # ±1 个百分点的忙碌读数
    sd = max(sds) if sds else 0.0
    floor = max(floors) if floors else 0.0
    half_band = (max(errs) - min(errs)) / 2.0 if len(errs) > 1 else 0.0
    res.mdd["dram_bytes_pct"] = round(max(2.0 * sd, floor, half_band, 0.5), 1)
    res.mdd["basis"] = ("max(同负载重复的 2×SD（含跨进程，若有）, NVML 1 个百分点分辨率下限, "
                        "本机误差带半宽)")
    res.mdd["run_to_run_sd_pct"] = round(sd, 2)
    res.mdd["resolution_floor_pct"] = round(floor, 2)
    res.mdd["band_half_width_pct"] = round(half_band, 2)
    # 原始跨进程样本要留下来：MDD 是被 diff 用来下判词的门槛，它必须可复核
    res.mdd["cross_process_samples"] = {k: [round(x, 2) for x in v]
                                        for k, v in sorted(extra.items())}
    # MDD 是"这段机器状态下"的噪声估计，不是机器的固有常数：真机实测同一台卡在两次
    # selftest 里给出过 SD=0.13% 与 SD=5.38% 两种结果（前者是热机稳定态，后者紧跟在
    # 别的负载之后）。所以必须把适用范围一起写出来，否则 diff 会给出过于自信的判词。
    res.mdd["caveat"] = ("MDD 只对测出它的那段机器状态成立：换机器、时钟/温度状态或"
                         "同机其他租户负载变化后都要重跑 selftest；比较两次运行前"
                         "先让 GPU 进入稳态")
    # kernel 时长的 MDD：用同一负载单次迭代时长的相对离散度近似（CUPTI 时长口径同源）
    ms_errs: list[float] = []
    for wid in DRAM_BOUND_IDS:
        runs = res.by_id(wid)
        if len(runs) >= 2:
            mean = statistics.fmean(r.kernel_ms for r in runs)
            if mean > 0:
                ms_errs.append(100.0 * statistics.stdev([r.kernel_ms for r in runs]) / mean)
    res.mdd["kernel_time_pct"] = round(max(2.0 * (max(ms_errs) if ms_errs else 0.0), 0.5), 1)
    sat = [w for w in bound if w.mem_util_pct < MIN_SATURATION_MEM_UTIL_PCT]
    if sat:
        res.notes.append(
            f"{len(sat)} 次测量的显存忙碌低于 {MIN_SATURATION_MEM_UTIL_PCT:.0f}%"
            f"（最低 {min(w.mem_util_pct for w in sat):.1f}%）：本次误差带偏乐观，"
            f"建议关掉同机其他负载或用更大的张量重跑")
    if len(res.workloads) >= 2:
        res.notes.append("误差带是**本机、本次口径**的测量结果：换机器或换 dtype 都要重跑")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernoscope selftest",
        description="纯 torch 的已知答案自检：量出本机误差带与 MDD（不需要 nsys/nvcc/NVBit）",
    )
    p.add_argument("-o", "--out", default=None,
                   help="写机器可读自检文件（同时可当 --ks-selfcheck 与 --ks-calibration 用）")
    p.add_argument("--json", action="store_true", help="把自检结果打到 stdout（JSON）")
    p.add_argument("--size-mb", type=int, default=DEFAULT_SIZE_MB, help="张量大小（MB）")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                   help="每类负载每次测量的饱和循环时长（秒）")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="每类负载重复次数")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--only", default=",".join(WORKLOAD_IDS),
                   help=f"只跑这些负载（逗号分隔，默认全部；可选 {','.join(WORKLOAD_IDS)}）")
    p.add_argument("--cross-process", type=int, default=2,
                   help="另起 N 个新进程重测，把跨进程漂移计进 MDD（0=不做）")
    p.add_argument("--quick", action="store_true", help="1 秒 × 2 次的快速版（误差带更粗）")
    p.add_argument("--strict", action="store_true", help="没能完成自检时 exit 2（CI 用）")
    p.add_argument("--quiet", action="store_true")
    return p


def selftest_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.duration = min(args.duration, 1.0)
        args.runs = min(args.runs, 2)
    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    bad = [x for x in only if x not in WORKLOAD_IDS]
    if bad or not only:
        print(f"error: --only 只接受 {','.join(WORKLOAD_IDS)}，收到 {args.only!r}",
              file=sys.stderr)
        return 1
    res = run_selftest(size_mb=args.size_mb, duration_s=args.duration, runs=args.runs,
                       dtype=args.dtype, only=only,
                       cross_process=args.cross_process)
    if args.out:
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(res.render())
    if not res.ok:
        # 缺依赖是"成功路径"（W1）：自检说的是"这台机器现在能给什么"，不是"命令失败"。
        # 需要把它当门禁时用 --strict。
        return 2 if args.strict else 0
    return 0


__all__ = ["SELFTEST_SCHEMA_VERSION", "SelftestResult", "WorkloadRun", "build_parser",
           "run_selftest", "selftest_main"]
