"""产物元数据自检：**发布前预演、CI 发布、发布后验证共用同一份断言**。

为什么要单独一个脚本（而不是在 `release-dry-run.sh` 与 `publish.yml` 里各写一份）：
"这个包能不能当 alpha 发出去"这件事只有一组判据（名字/版本/许可/Alpha 分类器/Python 下限/
console script 入口点/测试目录没被打进包）。写两份就一定会漂 —— 而漂的那一份正好是发布
那一刻在跑的那份。这个仓库已经因为"同一件事两个真源"红过好几次了。

用法：
    python bench/check_dist_metadata.py dist/kernoscope-0.2.0-py3-none-any.whl
    python bench/check_dist_metadata.py dist/*.whl --expect-version 0.2.0

退出码：0 全过；1 有判据不过（每条都点名）；2 用法/文件问题。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 每条判据：(METADATA 里的字面前缀, 人话)
REQUIRED_METADATA = (
    ("License-Expression: Apache-2.0", "许可是 Apache-2.0"),
    ("Classifier: Development Status :: 3 - Alpha", "Alpha 分类器（README 也声明了接口会变）"),
    ("Requires-Python: >=3.10", "Python 下限与代码里用的语法一致"),
)

#: console script 必须逐字包含这一条（`ksc` 是短别名，不在这里钉）
REQUIRED_ENTRY_POINT = "kernoscope = kernoscope.cli:main"


def project_version(root: pathlib.Path = ROOT) -> str:
    """`pyproject.toml` 里的 version（**唯一真源**：tag 也要与它一致）。"""
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            _, _, value = line.partition("=")
            return value.strip().strip('"')
    raise SystemExit("check_dist_metadata: pyproject.toml 里找不到 version")


def check(wheel: pathlib.Path, expect_version: str | None = None) -> list[str]:
    """返回问题清单（空 = 干净）。**只读**这一个 wheel，不碰网络。"""
    problems: list[str] = []
    if not wheel.exists():
        return [f"找不到产物：{wheel}"]
    try:
        z = zipfile.ZipFile(wheel)
    except zipfile.BadZipFile as exc:
        return [f"不是一个合法 wheel：{exc}"]
    names = z.namelist()

    metas = [n for n in names if n.endswith(".dist-info/METADATA")]
    if not metas:
        return ["wheel 里没有 .dist-info/METADATA"]
    meta = z.read(metas[0]).decode("utf-8", "replace")
    lines = meta.splitlines()

    if "Name: kernoscope" not in lines:
        problems.append("包名不是 kernoscope")
    want = expect_version or project_version()
    if f"Version: {want}" not in lines:
        got = next((l for l in lines if l.startswith("Version:")), "（没有 Version 行）")
        problems.append(f"版本不是 {want}（实际 {got}）—— tag 与 pyproject 不一致？")
    for prefix, what in REQUIRED_METADATA:
        if prefix not in lines:
            problems.append(f"{what}没写（期望 METADATA 里有 `{prefix}`）")

    eps = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
    if not eps:
        problems.append("没有 entry_points.txt（装完就没有 kernoscope 可执行文件）")
    elif REQUIRED_ENTRY_POINT not in z.read(eps[0]).decode("utf-8", "replace"):
        problems.append(f"console script 入口点不对（期望 `{REQUIRED_ENTRY_POINT}`）")

    leaked = [n for n in names if "/tests/" in n or n.startswith("tests/")]
    if leaked:
        problems.append(f"测试被打进包了：{leaked[:3]}")

    readme = [n for n in names if n.endswith(".dist-info/METADATA") and "Description-Content-Type" in meta]
    if not readme:
        problems.append("METADATA 里没有 Description-Content-Type（PyPI 页面的 README 渲染可能丢）")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查构建产物能不能当 alpha 发出去")
    ap.add_argument("wheel", nargs="+", help="一个或多个 .whl 路径")
    ap.add_argument("--expect-version", default=None,
                    help="期望版本（默认取 pyproject.toml；CI 里应传 tag 去掉 v 前缀的那个）")
    args = ap.parse_args(argv)

    bad = 0
    for raw in args.wheel:
        wheel = pathlib.Path(raw)
        problems = check(wheel, args.expect_version)
        if problems:
            bad += 1
            print(f"FAIL {wheel.name}（{wheel.stat().st_size if wheel.exists() else 0} B）")
            for p in problems:
                print(f"  ✗ {p}")
        else:
            print(f"ok   {wheel.name}  {wheel.stat().st_size} B —— 名字/版本/许可/Alpha/"
                  f"Python 下限/入口点/不含测试")
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
