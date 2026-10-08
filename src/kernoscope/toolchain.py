"""统一的工具发现（R-16/R-17）—— **不得只看 PATH**。

为什么需要它（真机实测）：某共享租用的 GPU 容器里 `ncu` / `cuobjdump` / `nvcc` 都在
`/usr/local/cuda-<版本>/bin/`，而默认 `PATH` 里没有这个目录。于是：

    doctor（只看 PATH）      → "未安装 ncu，无法实测判定"
    后端 find_cuobjdump()    → /usr/local/cuda-13.0/bin/cuobjdump   ← 同一台机器，两个答案
    os.path.exists(ncu)      → True

"未安装"是一个**可被一次 `ls` 推翻**的结论，而它还会被复制进别处指标的 `unavailable`
原因里，让一个错误的诊断传播到整份报告。所以发现逻辑只有一份，doctor 与后端共用。

搜索顺序（先命中先用，全部真实存在才返回）：
  1. `shutil.which`（尊重用户的 PATH 与 shell 别名目录）
  2. `$CUDA_HOME/bin` / `$CUDA_PATH/bin`（CUDA 官方约定）
  3. `/usr/local/cuda/bin`、`/usr/local/cuda-*/bin`（多版本共存，版本号大的优先）
  4. `/opt/cuda/bin`、`/usr/local/bin`、`/usr/bin`、`/bin`
  5. `KS_TOOL_SEARCH_DIRS`（冒号分隔，测试与特殊环境用；排在最前）

返回的 `Discovery` 同时给出**找过哪些路径**，这样"未安装"这句结论是可证伪的。
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import dataclass, field

# 需要"多路径发现"的工具：都是 CUDA Toolkit 的成员，装在 CUDA_HOME 里但常不在 PATH 上
CUDA_TOOLS = ("ncu", "cuobjdump", "nvcc", "nvdisasm", "nsys")

_ALWAYS_DIRS = (
    "/usr/local/cuda/bin",
    "/opt/cuda/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
)


@dataclass
class Discovery:
    """一次工具发现的完整结果（可证伪：searched 里没有的路径不会被声称找过）。"""

    name: str
    path: str | None = None
    searched: list[str] = field(default_factory=list)
    in_path: bool = False

    @property
    def found(self) -> bool:
        return bool(self.path)

    @property
    def found_but_not_in_path(self) -> bool:
        return self.found and not self.in_path

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": self.path, "in_path": self.in_path,
                "found": self.found, "searched_dirs": list(self.searched)}


def _version_key(path: str) -> tuple:
    """让 `/usr/local/cuda-13.0/bin` 排在 `/usr/local/cuda-12.4/bin` 前面（版本号大的优先）。"""
    nums = [int(t) for t in re.findall(r"\d+", path)]
    return tuple(-n for n in nums) if nums else ()


def cuda_bin_dirs() -> list[str]:
    """CUDA 相关的候选 bin 目录（顺序即优先级）。"""
    out: list[str] = []
    extra = os.environ.get("KS_TOOL_SEARCH_DIRS", "")
    for d in extra.split(os.pathsep):
        if d.strip():
            out.append(d.strip())
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        base = os.environ.get(var)
        if base:
            out.append(os.path.join(base, "bin"))
    out.append("/usr/local/cuda/bin")
    # /usr/local/cuda-13.0/bin 这类版本化目录：版本高的优先
    out.extend(sorted(glob.glob("/usr/local/cuda-*/bin"), key=_version_key, reverse=True))
    out.extend(_ALWAYS_DIRS)
    seen: list[str] = []
    for d in out:
        if d not in seen:
            seen.append(d)
    return seen


def search_paths(name: str) -> list[str]:
    """这个工具会被在哪些目录里找（doctor 用它证明"我真的找遍了"）。"""
    dirs: list[str] = []
    which = shutil.which(name)
    if which:
        dirs.append(os.path.dirname(which))
    for d in cuda_bin_dirs():
        if d not in dirs:
            dirs.append(d)
    return dirs


def discover(name: str) -> Discovery:
    which = shutil.which(name)
    searched: list[str] = []
    if which:
        return Discovery(name=name, path=which, searched=[os.path.dirname(which)], in_path=True)
    for d in cuda_bin_dirs():
        searched.append(d)
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return Discovery(name=name, path=cand, searched=searched, in_path=False)
    return Discovery(name=name, path=None, searched=searched, in_path=False)


def find(name: str) -> str | None:
    """找到一个可执行文件（PATH → CUDA_HOME → /usr/local/cuda* → 常见目录）。"""
    return discover(name).path
