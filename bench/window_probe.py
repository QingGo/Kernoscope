#!/usr/bin/env python3
"""R-47 的验收工具：per-launch 的**分母**必须落在参与积分的那个窗口内。

两种用法（在**有 GPU 的机器**上）：

    # 1) 跑单 kernel 紧循环（让窗口里只有它），供 bench 的三种窗口各跑一次
    python bench/window_probe.py --run --elems 134217728 --target-s 30

    # 2) 拿三次运行的决策包判验收（与验收报告的 R-47 判据同款）
    python bench/window_probe.py --check --analytic-mb 1073.741824 \\
        pk-auto.json pk-warmup10.json pk-warmup20.json

验收判据（**会红**）：

  * 同一负载换窗口后，per-launch 的相对离散必须 ≤ 5%（`MAX_SPREAD_PCT`）；
  * 或者该口径被**明确拒绝**（`status=refused` + 结构化 code）—— 那也是合格答案；
  * **"差异 > 5% 却照样给了数字"是失败** —— 这正是旧实现的行为：分子按窗口裁、
    分母不裁，真机实测同一负载 `--ks-window auto` 1.06 GB / `--ks-warmup 10` 0.86 GB /
    `--ks-warmup 20` 0.52 GB，而 share=100%、bias=+0%、无告警、rc=0。

为什么要单独跑一个单 kernel 负载：多 kernel 窗口在份额前提上就会被拒（那是 D1 的修法），
本文件要验的是**份额成立时**分母有没有跟着窗口走。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_SPREAD_PCT = 5.0


def run_loop(elems: int, target_s: float) -> None:
    """单 kernel 紧循环：读一遍写一遍，字节数解析可得（2×N×itemsize）。"""
    import time

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def ks_window_probe_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 1.000001, mask=m)

    x = torch.randn(elems, device="cuda")
    y = torch.empty_like(x)
    block = 1024
    grid = (triton.cdiv(elems, block),)

    def launch() -> None:
        ks_window_probe_kernel[grid](x, y, elems, BLOCK=block)

    for _ in range(20):
        launch()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < target_s:
        launch()
        n += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    analytic_mb = 2.0 * elems * 4 / 1e6          # fp32：读 + 写
    print(f"LOOP END: {n} launches in {dt:.2f}s ({dt / max(1, n) * 1e6:.1f} us/launch); "
          f"analytic {analytic_mb:.3f} MB/launch", flush=True)


def _per_launch(packet: dict) -> tuple[float | None, str]:
    for v in packet.get("verdicts") or []:
        if v.get("id") == "dram_bytes_per_launch":
            code = ((v.get("refusal") or {}).get("code") or "")
            return (v.get("value"), v.get("status", "") + (f"/{code}" if code else ""))
    return None, "missing"


def check(paths: list[Path], analytic_mb: float | None) -> int:
    rows = []
    for p in paths:
        val, note = _per_launch(json.loads(p.read_text(encoding="utf-8")))
        rows.append((p.name, val, note))
    nums = [v for _, v, _ in rows if v is not None]
    print("窗口            per-launch            相对解析真值      状态")
    for name, val, note in rows:
        if val is None:
            print(f"  {name:14s} {'n/a':>12s}                          {note}")
        else:
            dev = "" if analytic_mb is None else f"{(val / 1e6 / analytic_mb - 1) * 100:+.1f}%"
            print(f"  {name:14s} {val / 1e6:>12.2f} MB {dev:>16s}      {note}")
    if not nums:
        print("全部被拒 —— 只要每条都带结构化 code，这就是合格答案")
        return 0 if all("refused" in note for _, _, note in rows) else 1
    spread = 100.0 * (max(nums) - min(nums)) / (sum(nums) / len(nums))
    print(f"\n相对离散 {spread:.2f}%（判据 ≤ {MAX_SPREAD_PCT:g}%）")
    if analytic_mb is not None:
        worst = max(abs(v / 1e6 / analytic_mb - 1) for v in nums)
        print(f"与解析真值的最大偏差 {worst * 100:.2f}%")
    if spread > MAX_SPREAD_PCT:
        print(f"FAIL: 换窗口改变了 per-launch（{spread:.2f}% > {MAX_SPREAD_PCT:g}%）"
              f" —— 说明分母没有跟着窗口走（R-47）", file=sys.stderr)
        return 1
    if analytic_mb is not None and max(abs(v / 1e6 / analytic_mb - 1) for v in nums) > 0.10:
        print("FAIL: per-launch 与解析真值差 >10%（代理口径不该差这么多）", file=sys.stderr)
        return 1
    print("OK: per-launch 与窗口无关（分母落在同一个窗口里）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-47 验收：per-launch 的分母必须跟着窗口")
    ap.add_argument("--run", action="store_true", help="跑单 kernel 紧循环")
    ap.add_argument("--check", action="store_true", help="对若干决策包判验收")
    ap.add_argument("--elems", type=int, default=1 << 27)
    ap.add_argument("--target-s", type=float, default=30.0)
    ap.add_argument("--analytic-mb", type=float, default=None,
                    help="每次 launch 的解析字节数（MB）；给了就同时比绝对值")
    ap.add_argument("packets", nargs="*", type=Path)
    args = ap.parse_args(argv)
    if args.run:
        run_loop(args.elems, args.target_s)
        return 0
    if args.check:
        if not args.packets:
            ap.error("--check 需要若干决策包路径")
        return check(args.packets, args.analytic_mb)
    ap.error("要么 --run，要么 --check")


if __name__ == "__main__":
    raise SystemExit(main())
