"""`tools/release_bump.py`：升版本号必须**一处命令、四处同步**，且守住"默认只升末位"。

真机发布的教训（0.2.1）：`pyproject.toml`、`__init__.py` 的 `__version__`、README 的已发布版本、
`CHANGELOG.md` 的版本段 —— 四处漏一处就有门禁变红（那条门禁是对的），而我在一次发布里为此
来回改了两次。机械的事不该靠人记。
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("release_bump", ROOT / "tools" / "release_bump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load()


def _mini(tmp_path: pathlib.Path, version: str = "0.2.0") -> pathlib.Path:
    (tmp_path / "src/kernoscope").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "kernoscope"\nversion = "{version}"\n', encoding="utf-8")
    (tmp_path / "src/kernoscope/__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(
        f"**已发布到 PyPI**（`{version}`，alpha）：`uv tool install kernoscope`。\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# 变更记录\n\n## Unreleased\n\n- 一些改动\n\n## 0.1.0 —— 2026-01-01\n\n- 首发\n",
        encoding="utf-8")
    return tmp_path


def test_repo_version_is_self_consistent():
    """真仓库四处版本号必须自洽（这条在两个分支上都成立：0.2.0 / 0.2.1）。"""
    ok, detail = rb.check(ROOT)
    assert ok, detail


def test_patch_bump_syncs_all_four_places_and_is_idempotent(tmp_path):
    root = _mini(tmp_path)
    rb.bump("0.2.1", title="补丁级", date="2026-10-12", root=root)
    assert 'version = "0.2.1"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "0.2.1"' in (root / "src/kernoscope/__init__.py").read_text(encoding="utf-8")
    assert "`0.2.1`" in (root / "README.md").read_text(encoding="utf-8")
    cl = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.2.1 —— 2026-10-12（补丁级）" in cl
    assert cl.index("## Unreleased") < cl.index("## 0.2.1 ——"), "必须新开一个空的 Unreleased"
    assert "## 0.1.0" in cl, "旧版本段不能被弄丢"
    ok, detail = rb.check(root)
    assert ok, detail
    # 幂等：同一版本再来一次不算改动
    assert rb.bump("0.2.1", title="x", date="2026-10-12", root=root) == ["已经是 0.2.1：无需改动"]


def test_minor_bump_is_refused_by_default(tmp_path):
    root = _mini(tmp_path)
    with pytest.raises(SystemExit, match="只升末位"):
        rb.bump("0.3.0", title="", date="2026-10-12", root=root)
    rb.bump("0.3.0", title="", date="2026-10-12", root=root, allow_minor=True)
    assert rb.current_version(root) == "0.3.0"


def test_dry_run_changes_nothing(tmp_path):
    root = _mini(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rb.bump("0.2.1", title="", date="2026-10-12", root=root, dry_run=True)
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
