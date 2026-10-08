"""仓库卫生：把几条"只靠约定"的规则变成可执行的测试。

* `docs/` 是本地工作目录，**不进仓库**（`.gitignore` 整目录忽略，且不允许出现取反行）——
  这条曾经是口头约定，结果很容易被一次 `git add -f` 破坏。
* 被测产物入库前必须脱敏：不允许出现平台主机名/绝对路径这类身份信息。

测试在没有 git（或不在 git 仓库里）时自动跳过，不影响正常开发。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


needs_git = pytest.mark.skipif(_git("rev-parse", "--is-inside-work-tree") is None,
                               reason="不在 git 仓库里（例如导出的源码包）")


@needs_git
def test_docs_directory_is_not_tracked():
    tracked = _git("ls-files", "docs/") or ""
    assert tracked.strip() == "", f"docs/ 不应进入仓库，但被跟踪了：{tracked.split()}"


@needs_git
def test_gitignore_ignores_docs_without_negation():
    text = (ROOT / ".gitignore").read_text()
    lines = [l.strip() for l in text.splitlines()]
    assert "docs/" in lines, ".gitignore 必须整目录忽略 docs/"
    assert not [l for l in lines if l.startswith("!") and "docs" in l], \
        "不允许出现 docs 的取反行（那本身就是线索）"


@needs_git
def test_tracked_files_have_no_platform_identity():
    """入库文件里不得出现租用平台/实例身份字符串（产物走 bench/sanitize_artifacts.py）。"""
    banned = [
        re.compile(r"\bautodl-[a-z0-9-]+", re.I),          # 平台容器名
        re.compile(r"\bssh\s+-p\s+\d+\s+\w+@"),            # 带端口与用户的 ssh 命令
        re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),        # 裸 IPv4
    ]
    files = (_git("ls-files") or "").split()
    offenders: list[str] = []
    for f in files:
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        for rx in banned:
            if rx.search(text):
                offenders.append(f"{f}: {rx.pattern}")
    assert not offenders, "以下入库文件里出现了平台身份信息：" + "; ".join(offenders)
