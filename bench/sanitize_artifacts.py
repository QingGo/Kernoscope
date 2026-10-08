#!/usr/bin/env python3
"""bench/sanitize_artifacts.py —— 把实测产物里的机器标识替换成占位符（提交前跑一次）。

为什么需要：产物是**原样落盘**的（这正是它们的价值：可复核），但里面会带租用实例的
主机名与绝对路径。入库前统一脱敏，规则写在 `bench/artifacts/README.md` 里，谁都能复核。

替换规则（只动文本文件，且规则**不含任何具体平台字符串**）：
    * 主机名：从 `environment.txt` 的 `host:` 行读出来 → `<host>`
    * 仓库根目录的绝对路径（`--repo-path`，默认当前目录）→ `<repo>`
    * 仓库根目录的父目录 → `<workdir>`
    * `.../bin/python*` 之类的解释器绝对路径 → `<python>`
CUDA 的绝对路径、IPv4/GPU 型号/驱动版本/日期**一律保留**（它们是实验条件，不是身份）。

用法：
    python bench/sanitize_artifacts.py bench/artifacts
    python bench/sanitize_artifacts.py bench/artifacts --check     # 只检查，不改
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".json", ".csv", ".log", ".md", ".err", ".count", ".jsonl"}



def path_rules(repo_path: Path | None) -> list[tuple[re.Pattern[str], str]]:
    """按**本次运行的位置**生成路径规则（所以脚本里不出现任何具体平台路径）。

    同时覆盖"原样写的路径"与"完全解析后的路径"：/tmp 在 macOS 上是 /private/tmp 的
    符号链接，两种写法都会出现在产物里。
    """
    rules: list[tuple[re.Pattern[str], str]] = []
    if repo_path:
        raw = str(Path(repo_path))
        resolved = str(Path(repo_path).resolve())
        for cand, label in ((resolved, "<repo>"), (raw, "<repo>"),
                            (str(Path(resolved).parent), "<workdir>"),
                            (str(Path(raw).parent), "<workdir>")):
            if cand and cand not in ("/", "."):
                rules.append((re.compile(re.escape(cand) + r"(?=[/\s\"')\]]|$)"), label))
    rules.append((re.compile(r"/[^\s\"')]*?/bin/python[0-9.]*"), "<python>"))
    return rules


def hostnames(root: Path) -> list[str]:
    """从 environment.txt 的 `host:` 行取真实主机名（连同短名一起替换）。"""
    out: list[str] = []
    envs = [root / "environment.txt", *sorted(root.glob("*/environment.txt"))]
    for env in envs:
        if not env.exists():
            continue
        for line in env.read_text(errors="replace").splitlines():
            if line.startswith("host:"):
                h = line.split(":", 1)[1].strip()
                if h and h not in out:
                    out.append(h)
                    if h.split(".")[0] not in out:
                        out.append(h.split(".")[0])
    return out


def sanitize_text(text: str, hosts: list[str], fake_host: str,
                  rules: list[tuple[re.Pattern[str], str]]) -> tuple[str, int]:
    n = 0
    for pat, repl in rules:
        text, k = pat.subn(repl, text)
        n += k
    for h in hosts:
        if len(h) >= 6:
            text, k = re.subn(re.escape(h), fake_host, text)
            n += k
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="bench/artifacts")
    ap.add_argument("--check", action="store_true", help="只报告需要替换的地方，不改文件")
    ap.add_argument("--fake-host", default="<host>")
    ap.add_argument("--repo-path", default=".", help="仓库根目录（用于生成路径脱敏规则）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"error: {root} 不存在", file=sys.stderr)
        return 1
    hosts = hostnames(root)
    rules = path_rules(Path(args.repo_path))
    changed = total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        new, n = sanitize_text(text, hosts, args.fake_host, rules)
        if n:
            total += n
            changed += 1
            if not args.check:
                p.write_text(new, encoding="utf-8")
    verb = "需要脱敏" if args.check else "已脱敏"
    print(f"{verb}: {changed} 个文件 / {total} 处替换（主机名 {hosts[:1]} → {args.fake_host}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
