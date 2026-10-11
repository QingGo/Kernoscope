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
# 第二块：**正文**里那节"口径能不能被分辨"的数字。手抄的正文曾经与紧邻上方的节选
# 直接矛盾（验收报告第四轮 P3：正文 bf16 3.8%/fp16 13.2%/fp32 0.7%/54.5/166.9 全部
# 来自上一次产物），而覆盖它的 needle 门禁在改成生成节的同一轮被删掉了 —— 所以正文
# 数字现在是生成物，不再靠"记得更新"。
CALIB_BEGIN = "<!-- calib-caliber:begin -->"
CALIB_END = "<!-- calib-caliber:end -->"

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


def render_caliber() -> str:
    """口径可分辨性表 + 跨产物的一句话（都现算）。"""
    art, prof = newest_profile()
    mm = prof.get("measured_matmul_tflops") or {}
    rows = ["| dtype | 头条 TFLOP/s | 头条来自哪个变体 | 三个口径变体的读数 | 最大相对差 | 可分辨（≥2%） |",
            "|---|---|---|---|---|---|"]
    for dtype in DTYPES:
        v = mm.get(dtype)
        if not v:
            continue
        variants = v.get("caliber_variants_tflops") or {}
        vr = " / ".join(f"{k} {val}" for k, val in variants.items()) or "—"
        dist = v.get("caliber_distinguishable")
        rows.append(f"| `{dtype}` | **{v.get('tflops')}** | `{v.get('tflops_protocol')}` | {vr} | "
                    f"{v.get('caliber_spread_pct')}% | {'**是**' if dist else '**否**'} |")
    # 跨产物：同一 dtype 的 spread 在不同产物里可以跨过 2% 门槛（结论相反）
    per: dict[str, list[float]] = {}
    for d in sorted(x for x in ARTIFACTS.glob("*") if x.is_dir()):
        p = d / "machine-profile.json"
        if not p.exists():
            continue
        try:
            prof2 = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for dtype, v in (prof2.get("measured_matmul_tflops") or {}).items():
            if v.get("caliber_spread_pct") is not None:
                per.setdefault(dtype, []).append(float(v["caliber_spread_pct"]))
    spans = "；".join(f"`{k}` {min(v):.1f}%~{max(v):.1f}%" for k, v in sorted(per.items()) if v)
    # 头条必须等于某个变体读数（这曾经是个真缺陷：头条与三个变体都不相等，读者无法复现）
    checked = [(d, v) for d, v in mm.items() if v.get("caliber_variants_tflops")]
    bad = [d for d, v in checked
           if v.get("tflops") not in (v["caliber_variants_tflops"] or {}).values()]
    anchor = ("本产物的三个 dtype 头条**都等于**某个变体读数 ✓"
              if checked and not bad else
              f"⚠ 这些 dtype 的头条不等于任何变体读数：{bad}（正是 `tflops_protocol` 要钉住的东西）")
    return "\n".join([
        CALIB_BEGIN,
        "<!-- 由 tools/gen_calibration_excerpt.py 生成（取自最新产物）；不要手改本块 -->",
        "",
        *rows,
        "",
        f"（本表由 `bench/artifacts/{art.name}/machine-profile.json` 渲染。同一 dtype 的"
        f"三口径最大相对差在不同产物里可以跨过 2% 门槛 —— 跨已登记产物看：{spans}。"
        f"所以「口径能不能被分辨」本身是**测量结果**，不是常量。{anchor}。）",
        CALIB_END,
    ])


def sync(check: bool) -> int:
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{DOC.name}: 缺少产物节选标记块（{BEGIN}）", file=sys.stderr)
        return 1
    new = text[:text.index(BEGIN)] + render() + text[text.index(END) + len(END):]
    if CALIB_BEGIN in new:
        new = (new[:new.index(CALIB_BEGIN)] + render_caliber()
               + new[new.index(CALIB_END) + len(CALIB_END):])
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
