#!/usr/bin/env python3
"""把 `CALIBRATION.md` 里的**产物节选**改成生成物（R-56 / D13）。

为什么必须生成：那段节选原本是手抄的，于是它慢慢变成了"两轮混合体" ——
`read/write_gbs`、`bf16.tflops` 来自最新产物，`fp16/fp32.tflops`、`l2_read_peak`、
`clocks` 来自上一次产物，`ts` 与 `fp32.ms` **两份都对不上**。而数字门禁只对拍了其中 3 个
（而且恰好是两份相同的字段），所以混合体能一路绿灯。

手抄的节选会漂；由产物渲染、并在门禁里 `--check` 的节选不会。取值口径：
**最新的一份 `bench/artifacts/<stamp>/machine-profile.json`**（登记表里标"现行"的那个）。

用法：
    python tools/gen_calibration_excerpt.py           # 写入标记块
    python tools/gen_calibration_excerpt.py --check   # 只校验（不一致则 exit 1）
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "CALIBRATION.md"
ARTIFACTS = ROOT / "bench" / "artifacts"

BEGIN = "<!-- calib-excerpt:begin -->"
END = "<!-- calib-excerpt:end -->"

# 节选哪些字段：只挑**下面正文真的在讨论**的那些，别把整份产物抄进来。
# （节选越全，越容易在下次产物更新时把无关字段也拖进来。）
SCALARS = ("schema_version", "gpu", "driver", "torch", "ts", "tool")
DTYPES = ("torch.bfloat16", "torch.float16", "torch.float32")


def newest_profile() -> tuple[pathlib.Path, dict]:
    dirs = sorted(d for d in ARTIFACTS.glob("*") if d.is_dir() and d.name[:1].isdigit())
    for d in reversed(dirs):
        p = d / "machine-profile.json"
        if p.exists():
            return d, json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit("bench/artifacts 下没有可用的 machine-profile.json")


def render() -> str:
    art, prof = newest_profile()
    out: dict = {}
    for k in SCALARS:
        if prof.get(k) is not None:
            out[k] = prof[k]
    if prof.get("measured_dram"):
        out["measured_dram"] = {k: v for k, v in prof["measured_dram"].items()
                                if not k.endswith("_method")}
    if prof.get("l2_info"):
        out["l2_info"] = prof["l2_info"]
    mm = prof.get("measured_matmul_tflops") or {}
    if mm:
        out["measured_matmul_tflops"] = {
            k: {kk: vv for kk, vv in v.items()
                if kk in ("tflops", "size", "ms", "acc_caliber", "tflops_protocol",
                          "tflops_is_variant_reading", "precision_settings",
                          "caliber_variants_tflops", "caliber_probe_size",
                          "caliber_spread_pct", "caliber_distinguishable")}
            for k, v in mm.items() if k in DTYPES}
    if prof.get("clocks_during_measurement"):
        out["clocks_during_measurement"] = prof["clocks_during_measurement"]
    if prof.get("bandwidth_sweep"):
        sweep = prof["bandwidth_sweep"]
        out["bandwidth_sweep"] = ([*sweep[:2], {"…": f"共 {len(sweep)} 个尺寸，明细见产物"}]
                                  if len(sweep) > 3 else sweep)
    body = json.dumps(out, ensure_ascii=False, indent=2)
    return "\n".join([
        BEGIN,
        "<!-- 由 tools/gen_calibration_excerpt.py 生成（取自最新产物）；不要手改本块 -->",
        "",
        "```jsonc",
        body,
        "```",
        "",
        f"（以上是 [`bench/artifacts/{art.name}/machine-profile.json`](bench/artifacts/) 的节选，"
        f"**由该产物渲染**（`tools/gen_calibration_excerpt.py`）；原始产物与 `bandwidth_sweep` "
        f"明细见同一目录。新的 bench 运行会让这一段与门禁一起更新。）",
        END,
    ])


def sync(check: bool) -> int:
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{DOC.name}: 缺少产物节选标记块（{BEGIN}）", file=sys.stderr)
        return 1
    new = text[:text.index(BEGIN)] + render() + text[text.index(END) + len(END):]
    if check:
        if new != text:
            print(f"{DOC.name}: 产物节选与最新产物不一致（重跑 tools/gen_calibration_excerpt.py）",
                  file=sys.stderr)
            return 1
        print(f"ok  {DOC.name} 产物节选")
        return 0
    DOC.write_text(new, encoding="utf-8")
    print(f"wrote {DOC.name} 产物节选（{len(render().encode('utf-8'))} B）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验 CALIBRATION.md 的产物节选")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args(argv)
    return sync(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
