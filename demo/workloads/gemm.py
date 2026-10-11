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

    t0 = time.time()
    n = 0
    while time.time() - t0 < SECONDS:
        step()
        n += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    flops = 2.0 * M * N * K * n
    print(f"[demo] iters={n} elapsed={elapsed:.2f}s 实测 {flops / elapsed / 1e12:.1f} TFLOP/s")
    print(f"[demo] ks-tokens={M * N}")


if __name__ == "__main__":
    main()
