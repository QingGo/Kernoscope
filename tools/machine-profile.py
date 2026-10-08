#!/usr/bin/env python3
"""
machine-profile.py —— 在这台机器上实测"可达峰值"，替代 ncu 的 SpeedOfLight 基准

为什么必须先做这一步：
  ncu 的 SOL 面板用「实测值 / 理论峰值」。理论峰值来自规格表，但真实机器会因为
  功耗墙、时钟、驱动、cuBLAS 版本而明显低于规格。把它换成本机实测的"可达峰值"，
  手工 roofline 的结论才可靠 —— 这叫 machine calibration。

测量内容：
  1. D2D 拷贝带宽（读+写）、只读带宽（reduce）、只写带宽（fill）
  2. L2 容量拐点（工作集扫描：什么时候从 L2 带宽掉到 DRAM 带宽）
  3. matmul 可达 TFLOP/s（bf16 / fp16 / fp32，扫描尺寸取最好值）
  4. 测量期间的 SM 时钟采样（确认是在 boost 时钟下测的，不是 idle 210MHz）
  5. PCIe link 与功耗墙（nvidia-smi，免权限）

输出：./machine-profile.json（可给 `kernoscope --ks-calibration` 使用）

用法：
  python tools/machine-profile.py           # 全量
  python machine-profile.py --quick         # 快速版（尺寸减小，约 15s）
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time

try:                       # 让 `--help` 与单元测试在没有 torch 的环境里也能用
    import torch
except ImportError:        # pragma: no cover - 取决于环境
    torch = None


def _require_torch():
    if torch is None:
        raise SystemExit("需要 PyTorch（这个脚本要跑真实 matmul/带宽微基准）："
                         "pip install torch；无 GPU 环境请用 `kernoscope --ks-synthetic`")
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA 设备（标定必须有 GPU；"
                         "无 GPU 环境请用 `kernoscope --ks-synthetic` 跑通流程）")


def write_profile(profile: dict, out: str) -> str:
    """落盘（R-18）：默认 --out 是**裸文件名**，`os.path.dirname` 会返回 ''。

    旧实现直接 `os.makedirs(os.path.dirname(args.out))` → 真机 `FileNotFoundError: ''`，
    而这一步是 README 与 doctor 共同推荐的第一步。用 abspath 兜住。
    """
    path = os.path.abspath(out)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    return path


# --------------------------------------------------------------------------
# 采样 SM 时钟：确认测量发生在 boost 时钟下
# --------------------------------------------------------------------------
class ClockSampler:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.wait(self.interval):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip()
            try:
                sm, pw, ut = [float(x) for x in out.split(",")]
                self.samples.append((sm, pw, ut))
            except Exception:
                pass

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def summary(self):
        if not self.samples:
            return None
        sm = [s[0] for s in self.samples]
        pw = [s[1] for s in self.samples]
        ut = [s[2] for s in self.samples]
        return {"sm_clock_min": min(sm), "sm_clock_max": max(sm),
                "sm_clock_mean": round(statistics.mean(sm), 1),
                "power_max": max(pw), "util_mean": round(statistics.mean(ut), 1),
                "n_samples": len(sm)}


def timeit(fn, reps=5, warmup=3):
    """返回 (秒/次, 本次会话内最好的一次)。best-of 能规避别抖动。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    t0 = time.perf_counter()
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - s)
    return best, (time.perf_counter() - t0) / reps


# --------------------------------------------------------------------------
# 1) DRAM 带宽
# --------------------------------------------------------------------------
def bandwidth_sweep(sizes_mb, dtype_bytes=4, reps=5):
    print("\n===== 1) 带宽扫描（找 L2 拐点 + 测 DRAM 可达带宽） =====")
    print(f"{'size_MB':>9} {'copy_GB/s':>11} {'read_GB/s':>11} {'write_GB/s':>11}")
    rows = []
    for mb in sizes_mb:
        n = mb * 1024 * 1024 // dtype_bytes
        a = torch.empty(n, dtype=torch.float32, device="cuda")
        b = torch.empty(n, dtype=torch.float32, device="cuda")
        tc, _ = timeit(lambda: b.copy_(a), reps=reps)
        tr, _ = timeit(lambda: a.sum(), reps=reps)
        tw, _ = timeit(lambda: b.fill_(0.0), reps=reps)
        nbytes = n * dtype_bytes
        row = {"size_mb": mb,
               "copy_gbs": 2 * nbytes / tc / 1e9,          # 读 + 写
               "read_gbs": nbytes / tr / 1e9,
               "write_gbs": nbytes / tw / 1e9}
        rows.append(row)
        print(f"{mb:>9} {row['copy_gbs']:>11.1f} {row['read_gbs']:>11.1f} {row['write_gbs']:>11.1f}")
        del a, b
        torch.cuda.empty_cache()
    # 大工作集 = DRAM
    big = [r for r in rows if r["size_mb"] >= 128] or rows[-1:]
    dram = {"copy_gbs": max(r["copy_gbs"] for r in big),
            "read_gbs": max(r["read_gbs"] for r in big),
            "write_gbs": max(r["write_gbs"] for r in big)}
    # L2 判定：read 带宽显著高于 DRAM（>1.5×）的最大工作集；再往上一档就是掉出 L2 的位置。
    # 注意别看小尺寸 —— 2/4/8MB 的带宽低是启动开销/延迟造成的，不是容量效应。
    l2_max = max((r["size_mb"] for r in rows if r["read_gbs"] > 1.5 * dram["read_gbs"]),
                 default=None)
    after = [r["size_mb"] for r in rows
             if l2_max is not None and r["size_mb"] > l2_max
             and r["read_gbs"] <= 1.5 * dram["read_gbs"]]
    knee = min(after) if after else None
    l2_peak = max((r["read_gbs"] for r in rows), default=0.0)
    print(f"  -> 可达 DRAM: copy {dram['copy_gbs']:.0f} GB/s, read {dram['read_gbs']:.0f} GB/s, "
          f"write {dram['write_gbs']:.0f} GB/s")
    print(f"  -> L2 驻留上限: {l2_max} MB（read {l2_peak:.0f} GB/s, "
          f"{l2_peak / max(dram['read_gbs'], 1):.1f}× DRAM）; 掉出 L2: {knee} MB")
    return rows, dram, {"l2_resident_max_mb": l2_max, "l2_drop_mb": knee,
                        "l2_read_peak_gbs": round(l2_peak, 1)}


# --------------------------------------------------------------------------
# 2) matmul 可达算力（并测量"累加口径可区分性"，R-09/R-10）
# --------------------------------------------------------------------------
CALIBER_VARIANTS = [
    # (名字, 设置函数, 记录进 profile 的 precision_settings)
    ("default", lambda: None, {"allow_fp16_reduced_precision_reduction": None,
                               "fp32_precision": "none"}),
]


def _precision_settings_snapshot():
    """读回当前的 torch 精度设置，作为标定产物的一部分（R-09）。"""
    snap = {}
    try:
        snap["allow_fp16_reduced_precision_reduction"] = bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    except Exception:
        snap["allow_fp16_reduced_precision_reduction"] = None
    for attr in ("fp32_precision", "float32_matmul_precision"):
        try:
            snap[attr] = str(getattr(torch.backends.cuda.matmul, attr))
        except Exception:
            pass
    try:
        snap["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    return snap


def _apply_precision(settings: dict):
    """把一份精度设置快照写回 torch（写不动的项静默跳过 —— torch 版本差异很大）。"""
    if settings.get("allow_fp16_reduced_precision_reduction") is not None:
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = \
                settings["allow_fp16_reduced_precision_reduction"]
        except Exception:
            pass
    if "fp32_precision" in settings:
        try:
            torch.backends.cuda.matmul.fp32_precision = settings["fp32_precision"]
        except Exception:
            pass
    if settings.get("allow_tf32") is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        except Exception:
            pass


def _set_precision(allow_reduced: bool | None = None, fp32_precision: str | None = None,
                   baseline: dict | None = None):
    """设置一项精度开关，并返回"恢复到 baseline"的可调用对象。

    baseline 必须**在外面算一次**再传进来：如果每个变体各自快照，链式设置会让后面的
    快照记下前面变体改过的值，于是"恢复现场"实际恢复成了中间状态（这个坑有测试守）。
    """
    prev = dict(baseline) if baseline is not None else _precision_settings_snapshot()
    if allow_reduced is not None:
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_reduced
        except Exception:
            pass
    if fp32_precision is not None:
        try:
            torch.backends.cuda.matmul.fp32_precision = fp32_precision
        except Exception:
            try:
                torch.backends.cuda.matmul.allow_tf32 = (fp32_precision != "ieee")
            except Exception:
                pass

    def restore():
        _apply_precision(prev)

    return restore


def _variants():
    """三种累加口径设置（default / fp16 reduced-reduction off / fp32 ieee）。"""
    baseline = _precision_settings_snapshot()
    out = [("default", _set_precision(baseline=baseline),
            {"allow_fp16_reduced_precision_reduction": True, "fp32_precision": "none"})]
    out.append(("reduced_precision_reduction_off",
                _set_precision(allow_reduced=False, baseline=baseline),
                {"allow_fp16_reduced_precision_reduction": False, "fp32_precision": "none"}))
    out.append(("fp32_ieee", _set_precision(fp32_precision="ieee", baseline=baseline),
                {"allow_fp16_reduced_precision_reduction": True, "fp32_precision": "ieee"}))
    return out


def matmul_peak(sizes, dtypes, caliber_probe=True, reps=3):
    """测每个 dtype 的可达 TFLOP/s，并**测量**累加口径的可区分性。

    为什么口径可区分性必须被测而不是被假设（B9 的真机证据）：
      10-07 的三设置最大相对差 0.4%（不可分辨），10-08 是 5.0%（可分辨）——
      同一台机器、相隔一天、两个脚本给出相反判断。所以标定产物必须带离散度。
    """
    print("\n===== 2) matmul 可达 TFLOP/s（本机真实上限，替代规格表） =====")
    out = {}
    for dt in dtypes:
        best = 0.0
        best_cfg = None
        best_variant = "default"
        variant_tflops: dict[str, float] = {}
        for s in sizes:
            a = torch.randn(s, s, device="cuda", dtype=dt)
            b = torch.randn(s, s, device="cuda", dtype=dt)
            try:
                t, _ = timeit(lambda: a @ b, reps=reps, warmup=2)
            except Exception as e:
                print(f"  {dt} {s}: 失败 {e}")
                del a, b
                torch.cuda.empty_cache()
                continue
            tflops = 2.0 * s ** 3 / t / 1e12
            print(f"  {str(dt):>14} {s:>6}x{s:<6} {t*1e3:>8.3f} ms  {tflops:>8.1f} TFLOP/s")
            if tflops > best:
                best, best_cfg = tflops, {"size": s, "ms": t * 1e3}
            del a, b
            torch.cuda.empty_cache()

        # ---- 口径探针：同一尺寸、三种设置各测一遍 ----
        spread = None
        distinguishable = None
        if caliber_probe and best_cfg:
            size = best_cfg["size"]
            a = torch.randn(size, size, device="cuda", dtype=dt)
            b = torch.randn(size, size, device="cuda", dtype=dt)
            try:
                for name, restore, _settings in _variants():
                    try:
                        t, _ = timeit(lambda: a @ b, reps=reps, warmup=2)
                        variant_tflops[name] = round(2.0 * size ** 3 / t / 1e12, 1)
                    except Exception as e:
                        print(f"  {dt} 口径探针 {name} 失败: {e}")
                    finally:
                        restore()
            finally:
                del a, b
                torch.cuda.empty_cache()
            vals = [v for v in variant_tflops.values() if v]
            if len(vals) >= 2:
                spread = round(100.0 * (max(vals) - min(vals)) / max(vals), 1)
                distinguishable = spread >= 2.0
                print(f"  -> {dt} 口径探针: {variant_tflops}  最大相对差 {spread}% "
                      f"=> {'可分辨' if distinguishable else '不可分辨'}")
            # 用探针中最好的一次作为该 dtype 的读数（与 default 一致时数值不变）
            if vals:
                best = max(best, max(vals))

        entry = {"tflops": round(best, 1), **(best_cfg or {}),
                 "acc_caliber": _caliber_name(dt),
                 "precision_settings": _precision_settings_snapshot(),
                 "method": f"torch.matmul, {reps} iters best-of, size sweep {list(sizes)}",
                 "caliber_variants_tflops": variant_tflops,
                 "caliber_spread_pct": spread,
                 "caliber_distinguishable": distinguishable}
        out[str(dt)] = entry
        print(f"  -> {dt} 最好: {best:.1f} TFLOP/s  @ {best_cfg}  "
              f"(口径离散 {spread}% / 可分辨={distinguishable})")
    return out


def _caliber_name(dt) -> str:
    if dt == torch.float32:
        return "fp32 (fp32 accumulate)"
    if dt == torch.float16:
        return "fp16 (fp32 accumulate; 允许 reduced-precision reduction)"
    return "bf16 (fp32 accumulate)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速版（小尺寸，约 15s）")
    ap.add_argument("--out", default="machine-profile.json",
                    help="输出 JSON 路径（默认 ./machine-profile.json；含目录时自动创建）")
    ap.add_argument("--no-caliber-probe", action="store_true",
                    help="跳过累加口径探针（省时间；产物将缺 caliber_spread_pct）")
    args = ap.parse_args()

    _require_torch()
    props = torch.cuda.get_device_properties(0)
    name = props.name
    print(f"GPU: {name}  SM {props.multi_processor_count}  CC {props.major}.{props.minor}  "
          f"显存 {props.total_memory/2**30:.1f} GiB  L2 {getattr(props, 'L2_cache_size', 0)/2**20:.0f} MiB")

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit,"
         "pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    print(f"nvidia-smi: driver, maxSM_MHz, power_W, pcie_gen, pcie_width = {smi}")

    sizes = [2, 4, 8, 16, 32, 64, 128] if args.quick else \
            [2, 4, 8, 16, 32, 48, 64, 96, 128, 256]
    msizes = [4096, 8192] if args.quick else [4096, 8192, 16384]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]

    with ClockSampler() as cs:
        bw_rows, dram, l2_knee = bandwidth_sweep(sizes)
        mm = matmul_peak(msizes, dtypes, caliber_probe=not args.no_caliber_probe,
                         reps=2 if args.quick else 3)
        clocks = cs.summary()

    profile = {
        "schema_version": 2,
        "gpu": name,
        "cc": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_mib": round(getattr(props, "L2_cache_size", 0) / 2**20, 1),
        "nvidia_smi": smi,
        "measured_dram": {k: round(v, 1) for k, v in dram.items()},
        "l2_info": l2_knee,
        "measured_matmul_tflops": mm,
        "precision_settings_at_end": _precision_settings_snapshot(),
        "clocks_during_measurement": clocks,
        "bandwidth_sweep": bw_rows,
        "torch": torch.__version__,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "tools/machine-profile.py",
    }
    written = write_profile(profile, args.out)

    print("\n===== 测量期间的时钟/功耗 =====")
    print(" ", clocks)
    if clocks and clocks["sm_clock_max"] < 0.7 * float(smi.split(",")[1] or 0 or 1):
        print("  ⚠ SM 时钟明显低于最大值：测量可能在非 boost 状态，结论偏保守。")
        print("    建议先跑 10 秒满载预热，或长循环取 best-of。")
    print(f"\n已写入 {written}")
    print("roofline 用法（同仓库自带，不需要额外脚本）：")
    print("  kernoscope --ks-calibration %s --ks-gemm 4096,4096,4096 --ks-dtype bf16 "
          "-k regex:<kernel> -- <bench>" % args.out)
    print("口径提示：被测 matmul 的累加口径（fp32/fp16 acc）必须与产物里的 acc_caliber 一致；"
          "标定缺少该字段时 kernoscope 会拒绝用它算 roofline 拐点（R-09）。")


if __name__ == "__main__":
    main()
