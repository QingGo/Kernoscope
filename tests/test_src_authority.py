"""R-109：测试 import 的必须是**本仓库**的 `src/`，否则一切变异都是假绿。

真事（第六轮验收 W-13）：editable 安装的 `.pth` 钉住的是"安装那一刻"的源码树，而 pytest
从前没有 `pythonpath` 设置 ⇒ 在同步出来的副本上跑单个测试文件时，`import kernoscope`
会解析到**另一棵树**的 `src/`。于是一个复核集群把 `src/` 改坏的变异仍然全绿 ——
它验的不是被验对象。这条防线分两层：

  ① `pyproject.toml` 的 `[tool.pytest.ini_options] pythonpath = ["src"]`：pytest 在收集任何
     测试之前把本仓库的 `src` 插到 `sys.path[0]`，优先于 `.pth`（site-packages 在最后）
     与 `PYTHONPATH`；
  ② 本文件的断言：`kernoscope.__file__` 必须落在本仓库的 `src/` 下。

第 ② 条是"能红"的那条。可复现的证明（把 ① 用 `-o pythonpath=` 关掉、用一份副本冒充
editable 树，从别的目录跑）：

    cd /tmp && PYTHONPATH=/tmp/ks-fakesrc .venv/bin/python -m pytest \\
        <repo>/tests/test_src_authority.py -o pythonpath= -q
    # -> test_imported_kernoscope_lives_in_this_repo_src 失败，报文里点名 /tmp/ks-fakesrc

开着 ① 时同一条命令通过，且 `kernoscope.__file__` 就是本仓库那一份。
"""

from __future__ import annotations

import pathlib
import sys

try:                                    # Python 3.10 没有 tomllib（打包元数据断言会跳过）
    import tomllib
except ModuleNotFoundError:             # pragma: no cover
    tomllib = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _pytest_ini() -> dict:
    if tomllib is None:                 # pragma: no cover - 3.10 上退化成只跑行为断言
        return {}
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("pytest", {}).get("ini_options", {})


def test_pytest_config_puts_this_repo_src_first():
    """配置必须显式声明 `pythonpath = ["src"]`（去掉它，下面那条行为断言就该红）。"""
    ini = _pytest_ini()
    if not ini:                         # pragma: no cover
        return
    assert "src" in ini.get("pythonpath", []), (
        "pyproject 的 [tool.pytest.ini_options] 必须写 pythonpath = [\"src\"] —— "
        "editable 的 .pth 可以指向别的源码树，没有这一条时单跑测试文件会给出假绿（R-109）")


def test_imported_kernoscope_lives_in_this_repo_src():
    """本体：被 import 的 `kernoscope` 必须在本仓库 `src/` 下 —— 路径不对就点名报出来。"""
    import kernoscope

    got = pathlib.Path(kernoscope.__file__).resolve()
    want = (SRC / "kernoscope" / "__init__.py").resolve()
    assert got == want, (
        f"import 到的 kernoscope 是 {got}，而本仓库的是 {want} —— 测试跑的不是本次改动的代码"
        f"（sys.path 前几项：{sys.path[:4]}）。"
        " 处理：从仓库根跑 `uv run python -m pytest`，或确认 .venv 的 editable .pth 没有指着"
        " 另一份源码树；pytest 侧由 pyproject 的 pythonpath = [\"src\"] 兜底（R-109）")

    loaded = {m.__file__ for name, m in sys.modules.items()
              if name == "kernoscope" or name.startswith("kernoscope.")}
    outside = sorted(str(p) for p in loaded
                     if p and not pathlib.Path(p).resolve().is_relative_to(SRC.resolve()))
    assert not outside, (
        "这些 kernoscope 子模块来自本仓库 src/ 之外的地方（同一进程里混入了两棵树，"
        f"任何关于它们的断言都不算数）：{outside}")
