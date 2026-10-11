#!/usr/bin/env python3
"""算出"这次跑的是哪份代码"的指纹，供 bench 产物钉住（R-40 / D26）。

为什么需要它：产物按约定要钉住修订号与"有没有未提交改动"，但仓库常常是**不带 `.git`**
同步到租用机器上的（那时 `git rev-parse` 失败，只能回落到 `REVISION` 文件），而
`git status` 在有 `.git` 时又会因为"产物目录本身就是新建的"而永远报 dirty。
所以除了修订号，再记一个**只覆盖真正跑起来的源码**的指纹：

    python bench/source_hash.py          # 16 位 sha256 前缀

覆盖面：`src/**/*.py`、`tools/**/*.py`、`bench/*.py`、`bench/*.sh`、`bench/**/*.cu|h`、
`tools/probes/**/*.cu|h|sh|txt`、`pyproject.toml`。
明确的**不覆盖**：产物目录、测试、文档 —— 它们不影响本次测量的行为。
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

PATTERNS = ("src/**/*.py", "tools/**/*.py", "bench/*.py", "bench/*.sh",
            # R-69/N4：真正产出被引用数字的**微基准与探针源码**也要覆盖 ——
            # 账本真值（−4.2% / +12.8%）来自 `bench/ledger_probe.cu`，改它指纹不变就等于没钉。
            "bench/**/*.cu", "bench/**/*.h", "tools/probes/**/*.cu", "tools/probes/**/*.h",
            "tools/probes/**/*.sh", "tools/probes/**/*.txt")
EXTRA = ("pyproject.toml", "bench/run.sh")


def source_files(root: pathlib.Path) -> list[pathlib.Path]:
    files = {p for pat in PATTERNS for p in root.glob(pat) if p.is_file()}
    files |= {root / e for e in EXTRA if (root / e).is_file()}
    return sorted(files)


def source_hash(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in source_files(root):
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(source_hash(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
