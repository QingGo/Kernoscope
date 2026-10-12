"""`kernoscope calibrate` —— 让**装了包**的用户也能做本机标定（A-2′/R-127）。

为什么需要它（全新环境真机实测，2026-10-12）：标定的实现一直在 `tools/machine-profile.py`，
而 `tools/` **不在 wheel 里**（wheel 只含 `kernoscope/` 下的文件，实测 129 个）。于是 pip 装出来
的用户被 `doctor` 与拒绝语支到一条**不存在的命令**上：

    $ python tools/machine-profile.py --quick
    can't open file '.../tools/machine-profile.py': [Errno 2] No such file or directory

而同一台机器上源码 checkout 里跑 demo 的标定一切正常（实测产出 7655 B 的 profile）。所以这个洞
**只在"陌生人的机器"上出现** —— 正是本项目最该管的那一格。

实现上**不做搬运**：`tools/machine-profile.py` 仍是唯一实现（它的口径探针、`--quick` 语义、
以及已登记产物里的 `"tool": "tools/machine-profile.py"` 全都依赖它），由 `pyproject.toml` 的
force-include 随 wheel 发到 `kernoscope/_machine_profile.py`（与 `skills/` 同一套机制）。
本模块只做两件事：**找到它**、**按当前解释器调用它**。这样源码树与装出来的包跑的是同一份代码，
不会出现"仓库里修好了、装出来的包里还是旧的"。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: wheel 里那份的名字（由 pyproject 的 force-include 放进来）
BUNDLED_NAME = "_machine_profile.py"


def bundled_script() -> Path | None:
    """标定脚本的路径：优先源码树里的 `tools/machine-profile.py`，否则 wheel 里那份。

    源码树优先是**刻意**的：在这个仓库里开发时，被测对象必须是工作树里的那份代码
    （`src/` 是权威，同理 `tools/` 也是），而不是某次 `pip install` 留下的快照。
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent / BUNDLED_NAME,                                  # wheel：kernoscope/_machine_profile.py
        here.parent.parent.parent / "tools" / "machine-profile.py",   # checkout：<repo>/tools/...
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_module() -> ModuleType:
    path = bundled_script()
    if path is None:
        raise FileNotFoundError("找不到标定脚本")
    spec = importlib.util.spec_from_file_location("kernoscope._machine_profile", path)
    if spec is None or spec.loader is None:      # pragma: no cover - 只在文件坏掉时发生
        raise FileNotFoundError(f"标定脚本无法加载：{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    """标定自己的开关表（**从脚本里取**，不在这里重抄一遍）。

    文档命令门禁要用真实解析器校验 `kernoscope calibrate --quick` 这类命令；
    如果这里自己写一份开关表，两边就会漂 —— 所以直接把脚本的解析器借过来。
    """
    return _load_module().build_parser()


def calibrate_main(argv: list[str] | None = None) -> int:
    """跑一次本机标定。返回值是标定脚本的退出码。"""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        mod = _load_module()
    except FileNotFoundError as exc:
        print(f"error: {exc}（这个安装里没有随包发布的标定脚本；"
              f"请从源码树运行 `python tools/machine-profile.py`）", file=sys.stderr)
        return 1
    # 真机实测（全新环境，装出来的包）：脚本成功路径以前**不返回**（`main()` 落到 None），
    # `int(None)` 会在**标定已经成功、文件已经写好之后**抛 TypeError —— 用户看到 traceback，
    # 于是以为标定失败了。所以这里容忍 None（= 成功），并已在脚本侧补上 `return 0`。
    rc = mod.main(args)
    return 0 if rc is None else int(rc)
