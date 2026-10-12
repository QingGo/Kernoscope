"""demo 默认负载：**一步 mini 训练**（attention + MLP + LayerNorm + AdamW），带 NVTX 标注。

为什么用它当默认（而不是单个 GEMM）：
  * 一次运行里就有**几十个不同 kernel** ⇒ 能展示"谁最耗时"的排序与归属；
  * 外层套 NVTX `train_step` 区间 ⇒ 能展示阶段归因（每步花在哪）；
  * 有真实的 backward/优化器 ⇒ 能展示 `--ks-tokens` 之后的**每 token 成本**；
  * 3 秒钟跑完，足够截图，也不需要 GPU 之外的任何依赖。

规模可以用环境变量调（demo/README.md 里写了怎么调）：
    DEMO_SECONDS=3 DEMO_BATCH=8 DEMO_SEQ=512 DEMO_DIM=512
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

SECONDS = float(os.environ.get("DEMO_SECONDS", "3"))
BATCH = int(os.environ.get("DEMO_BATCH", "8"))
SEQ = int(os.environ.get("DEMO_SEQ", "512"))
DIM = int(os.environ.get("DEMO_DIM", "512"))
HEADS = int(os.environ.get("DEMO_HEADS", "8"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("DEMO_DTYPE", "bf16")]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("demo：这台机器没有可用的 CUDA 设备（demo 需要一张 GPU）")
    torch.manual_seed(0)
    dev = "cuda"
    print(f"[demo] {torch.cuda.get_device_name(0)} · {DTYPE} · "
          f"batch={BATCH} seq={SEQ} dim={DIM} heads={HEADS}")
    layer = nn.TransformerEncoderLayer(d_model=DIM, nhead=HEADS, dim_feedforward=4 * DIM,
                                       batch_first=True, dtype=DTYPE).to(dev)
    x = torch.randn(BATCH, SEQ, DIM, device=dev, dtype=DTYPE)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    def step() -> None:
        torch.cuda.nvtx.range_push("train_step")
        y = layer(x)
        loss = y.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.nvtx.range_pop()

    for _ in range(3):                       # warmup：别让第一次的编译/分配进窗口
        step()
    torch.cuda.synchronize()

    # 计时循环里**必须分批同步**：`time.time()` 量的是 CPU 入队时间，不是 GPU 时间。
    # 只在循环外同步时，慢卡上入队的积压会把标称 3 s 的场景拉长数倍
    # （同目录 gemm.py 在 RTX 2080 Ti 上实测 3 s → 166.7 s；本文件约 3 s → 11.2 s）。
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
    print(f"[demo] steps={n} tokens={n * BATCH * SEQ} elapsed={elapsed:.2f}s "
          f"({n / elapsed:.1f} step/s)")
    # token 数打在这里，是为了让 demo 脚本能用同一个数喂给 --ks-tokens（不手抄）
    print(f"[demo] ks-tokens={BATCH * SEQ}")


if __name__ == "__main__":
    main()
