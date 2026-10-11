"""`demo/` 的契约：它必须**跑得起来、且不假装**。

demo 是陌生人第一眼看到的东西，所以它的形状也要有测试钉住：
  * 脚本可执行、语法合法（含 `LC_ALL=C`），并且不会把 `PATH` 上的旧版本当成被测对象；
  * 场景文件存在、能 `py_compile`，且**不 import 本仓库**（它要能脱离 kernoscope 单独跑）；
  * `demo/out/` 不入库（产物是每次运行的结果，不是源码）。
"""

from __future__ import annotations

import os
import pathlib
import py_compile
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def test_demo_script_is_executable_and_parses_in_a_c_locale():
    script = DEMO / "quickstart.sh"
    assert script.exists(), "demo/quickstart.sh 不见了"
    assert script.stat().st_mode & 0o111, "quickstart.sh 应当可执行"
    for env in ({}, {"LC_ALL": "C", "LANG": "C"}):
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True,
                           env={**os.environ, **env})
        assert r.returncode == 0, r.stderr


def test_demo_prefers_the_checkout_it_lives_in_and_warns_about_path():
    """真机踩过：`PATH` 上的 `kernoscope` 可能是**别人留下的旧版本**（包装脚本指向冻结的旧树）。

    demo 必须用它自己所在的源码树，并把"PATH 上另有 X"显式说出来 —— 否则用户看到的结论
    与被测代码对不上，而这类错位在截图上完全看不出来。
    """
    text = (DEMO / "quickstart.sh").read_text(encoding="utf-8")
    assert "src/kernoscope" in text
    assert re.search(r"PATH 上还有一个 kernoscope", text), text[-2000:]
    # 源码树优先：判断顺序必须是"先看 src/，再看 PATH"
    assert text.index("src/kernoscope") < text.index("command -v kernoscope >/dev/null")


def test_demo_workloads_exist_and_are_self_contained():
    names = ["gemm", "train_step"]
    for name in names:
        wf = DEMO / "workloads" / f"{name}.py"
        assert wf.exists(), f"缺少场景 {name}"
        py_compile.compile(str(wf), doraise=True)
        src = wf.read_text(encoding="utf-8")
        assert "import kernoscope" not in src and "from kernoscope" not in src, \
            "负载必须能脱离本仓库单独跑（它只是被测程序，不是工具的一部分）"
        assert "nvtx" in src, f"{name} 应当带 NVTX 标注（阶段归因要用）"
    text = (DEMO / "quickstart.sh").read_text(encoding="utf-8")
    for name in names:
        assert name in text, f"脚本里没有 {name} 这个场景"


def test_demo_outputs_are_gitignored():
    r = subprocess.run(["git", "check-ignore", "-q", "demo/out/x.txt"], cwd=ROOT)
    assert r.returncode == 0, "demo/out/ 必须在 .gitignore 里（产物是运行结果，不是源码）"


def test_demo_readme_says_where_the_numbers_come_from():
    text = (DEMO / "README.md").read_text(encoding="utf-8")
    # 三类"拿不到"必须讲清区别（这是这个工具最容易被误解的地方）
    for needle in ("平台挡住", "补一条命令就能答", "需要额外采集"):
        assert needle in text, needle
    assert "本机标定" in text, "要说清标定的作用（roofline/能省多少的分母）"
    assert "示例值" in text, "价格是示例值这件事必须写出来"
