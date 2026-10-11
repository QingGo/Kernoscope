"""R-16/R-17 的负样本：工具发现不得只看 PATH。

真机证据（B3）：同一台机器，默认 PATH 下 doctor 说"未安装 ncu"、`cuobjdump 未找到（可选）`，
而后端 `find_cuobjdump()` 却返回 `/usr/local/cuda-13.0/bin/cuobjdump`，`os.path.exists(ncu)`
为 True —— 两个结论互相矛盾，而且"未安装"这个词会被复制进别处指标的 n/a 原因里。

**第五轮 §4.6 的教训**：这些用例只有在"工具不在 PATH"的机器上才绿 —— 验收方那台机器上
`cuobjdump`/`nvcc` 都在 PATH 里，`discover()` 按文档（PATH 优先）直接返回 PATH 命中，
于是断言全红。测试必须**自己构造前提**（把 PATH 命中屏蔽掉），否则测的是"这台机器装了什么"。
"""

import os
import stat

from kernoscope import toolchain as T


def _off_path(monkeypatch):
    """让 `shutil.which` 返回 None：模拟"这个工具不在 PATH 上"（与文档的 PATH 优先不冲突）。"""
    monkeypatch.setattr(T.shutil, "which", lambda name: None)


def _mkexe(path, name):
    p = path / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


def test_finds_tool_in_explicit_search_dirs(tmp_path, monkeypatch):
    d = tmp_path / "cuda-13.0" / "bin"
    d.mkdir(parents=True)
    exe = _mkexe(d, "cuobjdump")
    monkeypatch.setenv("KS_TOOL_SEARCH_DIRS", str(d))
    _off_path(monkeypatch)          # 真机装了 cuobjdump 也要走"非 PATH"这条路径
    disc = T.discover("cuobjdump")
    assert disc.found and str(exe) == disc.path
    assert disc.found_but_not_in_path, "它确实不在 PATH 上（这正是旧实现的盲区）"
    assert str(d) in disc.searched


def test_missing_tool_reports_what_was_searched(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setenv("KS_TOOL_SEARCH_DIRS", str(d))
    disc = T.discover("definitely-not-a-real-tool-xyz")
    assert not disc.found
    assert disc.searched, "必须留下'找过哪里'，否则'未安装'不可证伪"
    assert str(d) in disc.searched


def test_cuda_home_and_versioned_dirs_are_searched(tmp_path, monkeypatch):
    home = tmp_path / "cuda-home"
    (home / "bin").mkdir(parents=True)
    _mkexe(home / "bin", "nvcc")
    monkeypatch.setenv("KS_TOOL_SEARCH_DIRS", "")
    monkeypatch.setenv("CUDA_HOME", str(home))
    dirs = T.cuda_bin_dirs()
    assert str(home / "bin") in dirs
    assert "/usr/local/cuda/bin" in dirs
    _off_path(monkeypatch)
    disc = T.discover("nvcc")
    # 只有当我们注入的目录在搜索列表里、且真实存在该文件时才会命中
    assert disc.found
    assert disc.path == str(home / "bin" / "nvcc")


def test_versioned_dirs_sorted_newest_first():
    keyed = sorted(["/usr/local/cuda-12.4/bin", "/usr/local/cuda-13.0/bin",
                    "/usr/local/cuda-9.0/bin"], key=T._version_key)
    assert keyed[0] == "/usr/local/cuda-13.0/bin"


def test_existing_executables_are_not_shadowed_by_missing_ones(tmp_path, monkeypatch):
    """候选目录里没有该文件时不得返回一个不存在的路径（旧实现用 exists 兜住，这里守回归）。"""
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    (d2 / "bin").mkdir(parents=True)
    _mkexe(d2 / "bin", "nsys")
    monkeypatch.setenv("KS_TOOL_SEARCH_DIRS", f"{d1}{os.pathsep}{d2 / 'bin'}")
    _off_path(monkeypatch)
    disc = T.discover("nsys")
    assert disc.path == str(d2 / "bin" / "nsys")


def test_path_hit_wins_over_search_dirs_because_docs_say_so(monkeypatch, tmp_path):
    """文档契约：PATH 命中优先于搜索目录（`discover()` 的 docstring 第 1 条）。

    第五轮那三条用例在真机上变红，正是因为这条契约没被显式钉住：它们默认"工具不在 PATH"。
    """
    d = tmp_path / "elsewhere" / "bin"
    d.mkdir(parents=True)
    _mkexe(d, "cuobjdump")
    monkeypatch.setenv("KS_TOOL_SEARCH_DIRS", str(d))
    monkeypatch.setattr(T.shutil, "which", lambda name: "/usr/bin/cuobjdump")
    disc = T.discover("cuobjdump")
    assert disc.path == "/usr/bin/cuobjdump" and disc.in_path
