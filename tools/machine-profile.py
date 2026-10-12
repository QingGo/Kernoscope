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
    # (名字, 要应用的精度设置)
    # 只描述"要改什么"，记录的是**测量那一刻**从 torch 后端读回的实况（R-50/D7）。
    # 空 dict = 不改动现场（各 torch 版本的"默认"本来就不一样，写死字面量就是编造）。
    ("default", {}),
    ("reduced_precision_reduction_off", {"allow_fp16_reduced_precision_reduction": False}),
    ("fp32_ieee", {"fp32_precision": "ieee"}),
]

# 产物里对 null 的说明（R-50/D7）：读不到就留空，不编造
PRECISION_SETTINGS_NOTE = (
    "precision_settings 与 caliber_variant_settings 都是**测量那一刻**从 torch 后端读回的"
    "实况（不是字面量）；某项为 null 表示这个 torch 版本读不到它 —— 如实留空，未编造。")


def _precision_settings_snapshot():
    """读回当前的 torch 精度设置，作为标定产物的一部分（R-09/R-50）。

    读不到的项**如实写 None**（这个 torch 版本没有/读不了这一项），不填猜测值：
    产物里的 `precision_settings_note` 说明了 null 的含义。
    """
    snap = {}
    try:
        snap["allow_fp16_reduced_precision_reduction"] = bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    except Exception:
        snap["allow_fp16_reduced_precision_reduction"] = None
    try:
        snap["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        snap["allow_tf32"] = None
    try:
        snap["fp32_precision"] = str(torch.backends.cuda.matmul.fp32_precision)
    except Exception:
        snap["fp32_precision"] = None
    try:  # 老 torch 没有 fp32_precision 时，这个字符串档位是能读到的最接近的东西
        snap["float32_matmul_precision"] = str(
            torch.backends.cuda.matmul.float32_matmul_precision)
    except Exception:
        snap["float32_matmul_precision"] = None
    return snap


def _apply_precision(settings: dict):
    """把一份精度设置写回 torch（写不动的项静默跳过 —— torch 版本差异很大）。

    `None` 表示"这一项当时读不到"，跳过而不是编一个值。
    `fp32_precision` 最后写：它是新 torch 的权威字段，`allow_tf32` 是它的旧布尔别名，
    这样恢复现场时以权威字段为准（老 torch 没有前者，才退回写别名）。
    """
    if settings.get("allow_fp16_reduced_precision_reduction") is not None:
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = \
                settings["allow_fp16_reduced_precision_reduction"]
        except Exception:
            pass
    if settings.get("allow_tf32") is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        except Exception:
            pass
    if settings.get("fp32_precision") is not None:
        try:
            torch.backends.cuda.matmul.fp32_precision = settings["fp32_precision"]
        except Exception:
            if settings.get("allow_tf32") is None:
                try:  # 老 torch 只有 allow_tf32 这个布尔别名
                    torch.backends.cuda.matmul.allow_tf32 = (
                        settings["fp32_precision"] != "ieee")
                except Exception:
                    pass


def _set_precision(settings: dict | None = None, baseline: dict | None = None):
    """应用一份精度设置，并返回"恢复到 baseline"的可调用对象。

    调用方必须按「应用 → 测量 → 恢复」用：先 `restore = _set_precision(settings, baseline)`，
    再测量，测量完 `restore()`。只有这个顺序才能保证"每个变体都在自己的设置下被测"，
    并且让测量时刻读回的快照与产物记录一致（R-50/D7）。

    baseline 必须**在外面算一次**再传进来：如果每个变体各自快照，链式设置会让后面的
    快照记下前面变体改过的值，于是"恢复现场"实际恢复成了中间状态（这个坑有测试守）。
    """
    prev = dict(baseline) if baseline is not None else _precision_settings_snapshot()
    _apply_precision(settings or {})

    def restore():
        _apply_precision(prev)

    return restore


def _variants():
    """三种累加口径设置（default / fp16 reduced-reduction off / fp32 ieee）。

    只返回**设置描述** `(名字, 要应用的设置)`，构造期不做任何写入；应用与恢复由调用方
    在每次测量前后成对完成（应用 → 测量 → 恢复）。

    为什么不能在这里应用：旧实现把三次 `_set_precision` 都放在构造列表时调用，三个副作用
    在第一次测量之前全部落地，于是"default"与"reduced_precision_reduction_off"实际是在
    fp32_ieee 的设置下测的，而产物里记的却是写死的字面量（R-50 / 验收报告 D7）。
    """
    return [(name, dict(settings)) for name, settings in CALIBER_VARIANTS]


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
        # R-37：头条读数必须**能对应到一次具体的测量**。所以这里不只记离散度，
        # 还记下每个变体自己的 precision_settings；头条则取「变体里最快的那一个」，
        # 并明确写出它是哪个协议 —— 否则产物会出现「headline 不等于任何变体」
        # （真机产物实测：fp16 headline 167.4，三个变体是 145.5/163.2/163.3）。
        spread = None
        distinguishable = None
        variant_ms: dict[str, float] = {}
        variant_settings: dict[str, dict] = {}
        probe_size = None
        if caliber_probe and best_cfg:
            probe_size = best_cfg["size"]
            a = torch.randn(probe_size, probe_size, device="cuda", dtype=dt)
            b = torch.randn(probe_size, probe_size, device="cuda", dtype=dt)
            try:
                baseline = _precision_settings_snapshot()
                for name, settings in _variants():
                    # 顺序：应用 → 测量 → 恢复。快照在**应用之后、测量之时**读，
                    # 记录的才是这一次测量的实况，而不是写死的字面量（R-50/D7）。
                    restore = _set_precision(settings, baseline)
                    try:
                        measured_settings = _precision_settings_snapshot()
                        t, _ = timeit(lambda: a @ b, reps=reps, warmup=2)
                        variant_tflops[name] = round(2.0 * probe_size ** 3 / t / 1e12, 1)
                        variant_ms[name] = round(t * 1e3, 4)
                        variant_settings[name] = measured_settings
                    except Exception as e:
                        print(f"  {dt} 口径探针 {name} 失败: {e}")
                    finally:
                        restore()
            finally:
                del a, b
                torch.cuda.empty_cache()
            vals = [v for v in variant_tflops.values() if v]
            if len(vals) >= 2:
                # 三个变体是**同一尺寸、同一方法、同一 reps** 下测的，所以离散度只在
                # 同协议之间比较（R-37）。
                spread = round(100.0 * (max(vals) - min(vals)) / max(vals), 1)
                distinguishable = spread >= 2.0
                print(f"  -> {dt} 口径探针: {variant_tflops}  最大相对差 {spread}% "
                      f"=> {'可分辨' if distinguishable else '不可分辨'}")

        # ---- 头条读数：口径探针可用时由变体里最快的那次给出（可复现），否则用扫频值 ----
        if variant_tflops:
            protocol = max(variant_tflops, key=lambda k: variant_tflops[k])
            headline_tflops = variant_tflops[protocol]
            headline_size = probe_size
            headline_ms = variant_ms.get(protocol)
            headline_settings = variant_settings.get(protocol, {})
        else:
            protocol = "sweep-default"
            headline_tflops = round(best, 1)
            headline_size = (best_cfg or {}).get("size")
            headline_ms = (best_cfg or {}).get("ms")
            headline_settings = _precision_settings_snapshot()

        entry = {"tflops": headline_tflops,
                 "size": headline_size,
                 "ms": headline_ms,
                 "acc_caliber": _caliber_name(dt),
                 "precision_settings": headline_settings,
                 "precision_settings_note": PRECISION_SETTINGS_NOTE,
                 "tflops_protocol": protocol,
                 "tflops_is_variant_reading": bool(variant_tflops),
                 "method": f"torch.matmul, {reps} iters best-of, size sweep {list(sizes)}",
                 "caliber_variants_tflops": variant_tflops,
                 "caliber_variant_settings": variant_settings,
                 "caliber_probe_size": probe_size,
                 "caliber_probe_method": (f"torch.matmul, {reps} iters best-of, 同一尺寸 "
                                          f"{probe_size}" if probe_size else ""),
                 "caliber_spread_pct": spread,
                 "caliber_distinguishable": distinguishable}
        out[str(dt)] = entry
        print(f"  -> {dt} 最好: {headline_tflops:.1f} TFLOP/s  @ size={headline_size} "
              f"协议={protocol} (口径离散 {spread}% / 可分辨={distinguishable})")
    return out


def _caliber_name(dt) -> str:
    if dt == torch.float32:
        return "fp32 (fp32 accumulate)"
    if dt == torch.float16:
        return "fp16 (fp32 accumulate; 允许 reduced-precision reduction)"
    return "bf16 (fp32 accumulate)"


def build_parser() -> argparse.ArgumentParser:
    """标定的开关表（**唯一真源**）。

    抽出来是为了让装出来的包也能按同一套开关调用：`kernoscope calibrate` 借这个解析器
    暴露 `--help` 与开关校验（见 `src/kernoscope/calibrate_cmd.py`），而不是另抄一份开关表
    —— 两份开关表一定会漂。
    """
    ap = argparse.ArgumentParser(
        prog="kernoscope calibrate",
        description="测出这台机器的可达峰值（带宽 + 各 dtype 的 matmul 峰值 + 累加口径），"
                    "供 `--ks-calibration` 当作 roofline 的分母。")
    ap.add_argument("--quick", action="store_true", help="快速版（小尺寸，约 15s）")
    ap.add_argument("--out", default="machine-profile.json",
                    help="输出 JSON 路径（默认 ./machine-profile.json；含目录时自动创建）")
    ap.add_argument("--no-caliber-probe", action="store_true",
                    help="跳过累加口径探针（省时间；产物将缺 caliber_spread_pct）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
