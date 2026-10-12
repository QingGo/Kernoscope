"""一处升版本号：`pyproject.toml` → `__init__.py` → `README.md` → `CHANGELOG.md`。

为什么需要它（0.2.1 发布时真踩到的）：版本号在仓库里有**四处**要同步，
漏一处就有门禁变红，而这条门禁是**对**的（`test_packaging_metadata_is_consistent_with_reality`
钉住 pyproject 与 `__version__` 必须一致）。我在一次发布里为此来回改了两次。
机械的事就该交给脚本，人只决定"这次升多少"。

用法：
    python tools/release_bump.py 0.2.1                # 升到 0.2.1，并把 CHANGELOG 的 Unreleased 落成该版本
    python tools/release_bump.py 0.2.1 --title "补丁级：真机验证与修复"
    python tools/release_bump.py --check              # 只校验四处是否自洽（CI 可跑，不写文件）
    python tools/release_bump.py 0.2.1 --dry-run      # 打印将要改什么，不落盘

约定（见 RELEASING.md）：
  * 小改动只升**末位**（0.2.0 → 0.2.1 → 0.2.2）；只有 `packet.schema_version` /
    `schema_revision` 变化或破坏性接口变化才动中位/首位 —— 本脚本据此**拒绝**末位以外的升法，
    除非显式 `--allow-minor` / `--allow-major`；
  * `CHANGELOG.md` 的 `## Unreleased` 段落变成 `## <版本> —— <日期>（<标题>）`，
    并在前面新开一个空的 `## Unreleased`。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def current_version(root: pathlib.Path = ROOT) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', _read(root / "pyproject.toml"), re.M)
    if not m:
        raise SystemExit("pyproject.toml 里找不到 version")
    return m.group(1)


def survey(root: pathlib.Path = ROOT) -> dict[str, str]:
    """把四处版本号捞出来（不做判断，交给 `check` / `bump`）。"""
    out = {"pyproject": current_version(root)}
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', _read(root / "src/kernoscope/__init__.py"), re.M)
    out["__init__"] = m.group(1) if m else ""
    m = re.search(r"\*\*已发布到 PyPI\*\*（`([^`]+)`", _read(root / "README.md"))
    out["README"] = m.group(1) if m else ""
    m = re.search(r"^## (\d+\.\d+\.\d+)", _read(root / "CHANGELOG.md"), re.M)
    out["CHANGELOG_latest"] = m.group(1) if m else ""
    return out


def check(root: pathlib.Path = ROOT) -> tuple[bool, str]:
    s = survey(root)
    same = len(set(s.values())) == 1 and all(s.values())
    detail = "、".join(f"{k}={v or '(缺)'}" for k, v in s.items())
    return same, detail


def _bump_kind(old: str, new: str) -> str:
    a = [int(x) for x in old.split(".")]
    b = [int(x) for x in new.split(".")]
    if b[0] != a[0]:
        return "major"
    if b[1] != a[1]:
        return "minor"
    return "patch"


def bump(new: str, *, title: str, date: str, root: pathlib.Path = ROOT,
         dry_run: bool = False, allow_minor: bool = False, allow_major: bool = False) -> list[str]:
    old = current_version(root)
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        raise SystemExit(f"版本号要写成 X.Y.Z：{new!r}")
    if new == old:
        return [f"已经是 {new}：无需改动"]
    kind = _bump_kind(old, new)
    if (kind == "minor" and not allow_minor) or (kind == "major" and not allow_major):
        raise SystemExit(
            f"{old} → {new} 是 {kind} 升法；仓库约定默认只升末位（patch）。\n"
            f"确有意为之就加 --allow-{kind}，并在 CHANGELOG 里写清为什么。")

    changes: list[str] = []
    edits = [
        (root / "pyproject.toml", f'version = "{old}"', f'version = "{new}"'),
        (root / "src/kernoscope/__init__.py", f'__version__ = "{old}"', f'__version__ = "{new}"'),
        (root / "README.md", f"**已发布到 PyPI**（`{old}`", f"**已发布到 PyPI**（`{new}`"),
    ]
    for path, old_s, new_s in edits:
        text = _read(path)
        if old_s not in text:
            raise SystemExit(f"{path.relative_to(root)} 里找不到 {old_s!r}（版本号是不是已经不一致了？"
                             f"先跑 --check）")
        if not dry_run:
            path.write_text(text.replace(old_s, new_s, 1), encoding="utf-8")
        changes.append(f"{path.relative_to(root)}：{old} → {new}")

    # CHANGELOG：Unreleased 落成这一版，并新开一个 Unreleased
    cl = root / "CHANGELOG.md"
    text = _read(cl)
    if "## Unreleased" not in text:
        raise SystemExit("CHANGELOG.md 里没有 `## Unreleased` 段（版本段落应该都从这里落下来）")
    head = f"## {new} —— {date}（{title}）" if title else f"## {new} —— {date}"
    fresh = "## Unreleased\n\n（下一版的改动写在这里。小改动只升末位。）\n\n"
    if not dry_run:
        cl.write_text(text.replace("## Unreleased", fresh + head, 1), encoding="utf-8")
    changes.append(f"CHANGELOG.md：Unreleased → {head}")
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="一处升版本号（四处同步 + CHANGELOG 落段）")
    ap.add_argument("version", nargs="?", help="目标版本 X.Y.Z")
    ap.add_argument("--check", action="store_true", help="只校验四处是否自洽")
    ap.add_argument("--dry-run", action="store_true", help="打印将改什么，不落盘")
    ap.add_argument("--title", default="", help="CHANGELOG 版本段的小标题")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--allow-minor", action="store_true")
    ap.add_argument("--allow-major", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        ok, detail = check()
        print(("ok   " if ok else "FAIL ") + "四处版本号自洽：" + detail)
        return 0 if ok else 1
    if not args.version:
        ap.error("要么给版本号，要么用 --check")
    for line in bump(args.version, title=args.title, date=args.date, dry_run=args.dry_run,
                     allow_minor=args.allow_minor, allow_major=args.allow_major):
        print(("（dry-run）" if args.dry_run else "") + line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
