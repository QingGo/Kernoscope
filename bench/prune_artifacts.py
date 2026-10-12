"""把一次 `bench/run.sh` 的产物裁成**可入库的那几份**（KB–MB 级）。

为什么需要它（真机实测）：`bench/run.sh --quick` 在 4090 上跑完会写出 **1.3 GB** ——
里面绝大部分是**中间文件**（`.nsys-rep`、`.sqlite`、`.perfetto.json`、几个几十 MB 的完整报告），
而仓库里**已登记**的 5 份产物都只有 968 K–2.0 M，且只有文本/JSON/日志。
不裁的话：① 1.3 GB 进不了仓库；② 15 条产物门禁会红（`test_doc_artifact_consistency` 逐条对拍
文档里引用的数字，混进未登记的大文件会把它搅乱）。

保留规则（从"已登记产物长什么样"反推，**不是拍脑袋**）：
  * 丢弃：`DROP_SUFFIXES` 里那些（`.nsys-rep`/`.sqlite`/完整报告/…）—— 名单从**已登记产物的
    实际内容**反推，不是拍脑袋；
  * 兜底：**单个超过 `--max-mb`（默认 8 MB）**的文件也丢（并打印），因为登记惯例里的文件
    都在百 KB 级，"某份文件大得离谱"本身值得知道；
  * 永不删除 `manifest.json`（它是这次运行的登记表，`run.sh` 的每步 rc 都在里面）。

安全性：对**已经裁剪过**的目录再跑一次**必须什么都不做**（幂等）—— 仓库里那 5 份就是标准答案，
`--check` 会对它们验这一点。

用法：
    python bench/prune_artifacts.py bench/artifacts/<stamp>          # 真删
    python bench/prune_artifacts.py bench/artifacts/<stamp> --dry-run # 只看会删什么
    python bench/prune_artifacts.py bench/artifacts/*/ --check        # 幂等自检（已登记产物必须干净）
"""

from __future__ import annotations

import argparse
import pathlib
import sys

#: **丢**这些（体积大、且仓库的登记惯例里从来不放它们）。
#: 名单是从"已登记的 5 份产物里到底有什么"反推的：只有
#: `.txt/.err/.json/.log/.csv` 与两个无后缀文件，**没有**任何 `.nsys-rep`/`.sqlite`/
#: 完整报告（`*.ksreport.json`）。注意 `*.packet.json`（决策包，几 KB）**要留**：
#: 已登记的 3 份产物里就有它，它是判词形状的证据。
DROP_SUFFIXES = (".nsys-rep", ".nsys-rep.sqlite", ".sqlite", ".sqlite3", ".ncu-rep",
                 ".ksreport.json", ".perfetto.json",
                 ".cubin", ".ptx", ".fatbin", ".so", ".o", ".a")
#: 永不丢：这次运行的登记表（每步 rc 都在里面）
NEVER_DROP = ("manifest.json",)
DEFAULT_MAX_MB = 8.0


def plan(root: pathlib.Path, *, max_mb: float = DEFAULT_MAX_MB) -> tuple[list[pathlib.Path],
                                                                       list[pathlib.Path]]:
    """返回 `(要删的文件, 因体积过大而删的文件)`。**只读**，不碰磁盘。"""
    drop: list[pathlib.Path] = []
    too_big: list[pathlib.Path] = []
    cap = max_mb * 1024 * 1024
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in NEVER_DROP:
            continue
        name = p.name.lower()
        if any(name.endswith(suf) for suf in DROP_SUFFIXES):
            drop.append(p)
            continue
        # 兜底：登记惯例里的文件都在 ~百 KB；单个超过上限的多半是漏网的中间件
        if p.stat().st_size > cap:
            too_big.append(p)
    return drop, too_big


def prune(root: pathlib.Path, *, max_mb: float = DEFAULT_MAX_MB, dry_run: bool = False) -> dict:
    drop, too_big = plan(root, max_mb=max_mb)
    before = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    for p in drop + too_big:
        if not dry_run:
            p.unlink()
    # 删空目录（`.build/` 这类中间目录会留下空壳）
    if not dry_run:
        for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
    after = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return {"drop": drop, "too_big": too_big, "before": before, "after": after,
            "kept": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="裁剪 bench 产物到可入库的那几份")
    ap.add_argument("roots", nargs="+", help="产物目录（`bench/artifacts/<stamp>`）")
    ap.add_argument("--dry-run", action="store_true", help="只打印会删什么")
    ap.add_argument("--check", action="store_true",
                    help="幂等自检：已裁剪的目录必须**什么都不用删**（非零退出=它其实脏）")
    ap.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help="单文件上限（MB）")
    args = ap.parse_args(argv)

    bad = 0
    for raw in args.roots:
        root = pathlib.Path(raw)
        if not root.is_dir():
            print(f"FAIL 不是目录：{root}")
            bad += 1
            continue
        res = prune(root, max_mb=args.max_mb, dry_run=args.dry_run or args.check)
        n = len(res["drop"]) + len(res["too_big"])
        mb = res["before"] / 1e6
        msg = (f"{root.name}: 保留 {len(res['kept'])} 个文件、"
               f"{res['after'] / 1e6:.2f} MB（原 {mb:.2f} MB），"
               f"{'将删' if (args.dry_run or args.check) else '已删'} {n} 个中间文件")
        if res["too_big"]:
            msg += f"；因 >{args.max_mb} MB 而删：{[p.name for p in res['too_big']][:3]}"
        if args.check and n:
            bad += 1
            print(f"FAIL {msg} —— 已登记的产物不该有中间文件（要么先裁剪，要么别提交它）")
            for p in (res["drop"] + res["too_big"])[:5]:
                print(f"       {p.relative_to(root)}")
        else:
            print(f"ok   {msg}")
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
