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

import torch


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
# 2) matmul 可达算力
# --------------------------------------------------------------------------
def matmul_peak(sizes, dtypes):
    print("\n===== 2) matmul 可达 TFLOP/s（本机真实上限，替代规格表） =====")
    out = {}
    for dt in dtypes:
        best = 0.0
        best_cfg = None
        for s in sizes:
            a = torch.randn(s, s, device="cuda", dtype=dt)
            b = torch.randn(s, s, device="cuda", dtype=dt)
            try:
                t, _ = timeit(lambda: a @ b, reps=3, warmup=2)
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
        out[str(dt)] = {"tflops": round(best, 1), **(best_cfg or {})}
        print(f"  -> {dt} 最好: {best:.1f} TFLOP/s  @ {best_cfg}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速版（小尺寸，约 15s）")
    ap.add_argument("--out", default="machine-profile.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA 设备")
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
        mm = matmul_peak(msizes, dtypes)
        clocks = cs.summary()

    profile = {
        "gpu": name,
        "cc": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_mib": round(getattr(props, "L2_cache_size", 0) / 2**20, 1),
        "nvidia_smi": smi,
        "measured_dram": {k: round(v, 1) for k, v in dram.items()},
        "l2_info": l2_knee,
        "measured_matmul_tflops": mm,
        "clocks_during_measurement": clocks,
        "bandwidth_sweep": bw_rows,
        "torch": torch.__version__,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2, default=str)

    print("\n===== 测量期间的时钟/功耗 =====")
    print(" ", clocks)
    if clocks and clocks["sm_clock_max"] < 0.7 * float(smi.split(",")[1] or 0 or 1):
        print("  ⚠ SM 时钟明显低于最大值：测量可能在非 boost 状态，结论偏保守。")
        print("    建议先跑 10 秒满载预热，或长循环取 best-of。")
    print(f"\n已写入 {args.out}")
    print("roofline 用法：python roofline-calc.py --profile %s ..." % args.out)


if __name__ == "__main__":
    main()
