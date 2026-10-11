"""从 `CHANGELOG.md` 取某个版本的发布说明 —— **Release 正文的唯一真源**。

为什么要它：tag message、GitHub Release 正文、以及"这次发布了什么"的口径必须是同一份文本。
手抄一份到 Release 页，就一定会与 CHANGELOG 漂（这个仓库已经因为"同一件事两个真源"红过多次）。

用法：
    python bench/release_notes.py 0.2.0            # 打印该版本那一节的正文（不含 `## 0.2.0` 标题行）
    python bench/release_notes.py 0.2.0 --title    # 只打印标题行文本（给 Release 的 title / tag message 首行）
    python bench/release_notes.py 0.2.0 --tag      # 直接产出可 `git tag -a -F` 的完整文本（含标题）

退出码：0 找到；1 找不到该版本那一节（**不许静默给空**：tag 了却没有说明，是发布流程的漏洞）。
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"


def _section(version: str, changelog: pathlib.Path = CHANGELOG) -> tuple[str, str] | None:
    """返回 `(标题行文本, 正文)`；找不到返回 None。"""
    text = changelog.read_text(encoding="utf-8")
    lines = text.splitlines()
    hit = None
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        heading = line[3:].strip()
        # 允许 `## 0.2.0` 与 `## 0.2.0 — 2026-10-11` 两种写法；也允许 `v0.2.0`
        if re.match(rf"^v?{re.escape(version)}(?:\s|$|[—\-–(（])", heading):
            hit = (heading, i)
            break
    if hit is None:
        return None
    heading, start = hit
    body: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return heading, "\n".join(body).strip("\n")


def render(version: str, *, mode: str = "body", changelog: pathlib.Path = CHANGELOG) -> str | None:
    got = _section(version, changelog)
    if got is None:
        return None
    heading, body = got
    if mode == "title":
        return heading if heading.startswith("v") else f"v{heading}"
    if mode == "tag":
        return f"{heading if heading.startswith('v') else 'v' + heading}\n\n{body}\n"
    return body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 CHANGELOG 取某版本的发布说明（Release 正文的唯一真源）")
    ap.add_argument("version", help="版本号，例如 0.2.0")
    ap.add_argument("--title", action="store_true", help="只打印标题（给 Release title / tag 首行）")
    ap.add_argument("--tag", action="store_true", help="打印可直接喂给 `git tag -a -F` 的完整文本")
    args = ap.parse_args(argv)

    mode = "title" if args.title else ("tag" if args.tag else "body")
    out = render(args.version, mode=mode)
    if out is None:
        print(f"error: CHANGELOG 里找不到 `## {args.version}` 那一节 —— "
              f"发布前必须先写它（Release 正文以它为准，不许手抄）", file=sys.stderr)
        return 1
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
