"""demo 可选负载：**单个大 GEMM**（3 秒内跑很多次）。

用途与 `train_step.py` 互补：这里几乎只有一个 kernel（cuBLAS/cutlass GEMM）在跑，
所以一屏上"最耗时的是谁 + 占比"是一条**干净、无争议**的结论 —— 适合做封面截图；
而 `train_step.py`（默认）用来展示多 kernel 排序、阶段归因与成本。

    DEMO_SECONDS=3 DEMO_M=8192 DEMO_N=8192 DEMO_K=8192
"""

from __future__ import annotations

import os
import time

import torch

SECONDS = float(os.environ.get("DEMO_SECONDS", "3"))
M = int(os.environ.get("DEMO_M", "8192"))
N = int(os.environ.get("DEMO_N", "8192"))
K = int(os.environ.get("DEMO_K", "8192"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("DEMO_DTYPE", "bf16")]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("demo：这台机器没有可用的 CUDA 设备（demo 需要一张 GPU）")
    torch.manual_seed(0)
    print(f"[demo] GEMM {M}x{N}x{K} · {DTYPE} · "
          f"{torch.cuda.get_device_name(0)}")
    a = torch.randn(M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=DTYPE)

    def step() -> None:
        torch.cuda.nvtx.range_push("gemm")
        c = a @ b
        torch.cuda.nvtx.range_pop()
        return c

    for _ in range(2):
        step()
    torch.cuda.synchronize()

    # 计时循环里**必须分批同步**（真机实测，RTX 2080 Ti / 2026-10-12）：`time.time()` 量的是
    # CPU 入队时间，不是 GPU 时间。原实现只在循环外同步一次，于是 3 秒里入队了 1042 次 8192³
    # bf16 GEMM（该卡上 bf16 走 fp32 路径，一次 150 ms ⇒ 166.7 s 的 GPU 工作），
    # `synchronize()` 把这堆积压等完，标称 3 s 的场景实跑 166.7 s —— **慢 55 倍**。
    # 分批（而不是每次同步）是为了少切碎窗口：一次同步只留一个入队气泡。
    SYNC_EVERY = 4

    t0 = time.time()
    n = 0
    while True:
        for _ in range(SYNC_EVERY):
            step()
            n += 1
        torch.cuda.synchronize()
        if time.time() - t0 >= SECONDS:
            break
    elapsed = time.time() - t0
    flops = 2.0 * M * N * K * n
    print(f"[demo] iters={n} elapsed={elapsed:.2f}s 实测 {flops / elapsed / 1e12:.1f} TFLOP/s")
    print(f"[demo] ks-tokens={M * N}")


if __name__ == "__main__":
    main()
