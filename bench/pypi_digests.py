"""把 PyPI 上某个版本的产物 sha256 打成一节 Markdown（Release 页用）。

为什么要单独一个脚本：`publish.yml`（发布后自动建 Release）与 `release.yml`（给**已有** tag
补建 Release）都要这一节 —— 两份内联脚本一定会漂，而漂的那份正好是给人对拍用的数字。

用法：
    python bench/pypi_digests.py 0.2.0                 # 打印一节 Markdown
    python bench/pypi_digests.py 0.2.0 --from-json f.json   # 离线（测试用）：从本地 JSON 读

读不到就**如实说明**并且仍然 exit 0：Release 页不该因为一次网络抖动而建不出来。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

PROJECT = "kernoscope"


def _load(version: str, from_json: pathlib.Path | None) -> dict | None:
    if from_json is not None:
        return json.loads(from_json.read_text(encoding="utf-8"))
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:      # noqa: S310 —— 固定 https
            return json.load(r)
    except Exception:                                          # noqa: BLE001
        return None


def render(version: str, from_json: pathlib.Path | None = None) -> str:
    data = _load(version, from_json)
    lines = ["", "### 产物（PyPI）", ""]
    if data is None:
        lines.append(f"（读不到 PyPI 元数据；请到 https://pypi.org/project/{PROJECT}/{version}/ 核对）")
        return "\n".join(lines)
    for u in data.get("urls") or []:
        lines.append(f"- `{u['filename']}`（{u['size']} B）—— sha256 `{u['digests']['sha256']}`")
    lines.append("")
    lines.append(f"安装：`uv tool install {PROJECT}`（PyPI 那份才带 attestation）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PyPI 产物的 sha256 → Release 页的一节 Markdown")
    ap.add_argument("version")
    ap.add_argument("--from-json", default=None, help="离线：从本地 /pypi/<name>/<ver>/json 读")
    args = ap.parse_args(argv)
    path = pathlib.Path(args.from_json) if args.from_json else None
    sys.stdout.write(render(args.version, path) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
