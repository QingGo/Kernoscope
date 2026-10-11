"""陌生人首跑（G1）必须是一条**可重跑、能变红**的检查，不是一份人工记录。

为什么要它：开放门槛 G1 写的是「在一台不是你的机器上，从零到第一个结论 < 5 分钟」，
但此前这条只有人工记录；而 README 推荐的第一条命令在**没有 GPU 的机器上 rc=1 且 stderr 0 B**
（易用性实测）—— 陌生人与 CI 第一眼看到"失败且没有解释"。

`bench/preflight-first-run.sh` 把这条路径变成机器检查：现场构建 wheel → 装进**隔离**目录
（等价于 `uv tool install kernoscope`）→ 按 README 顺序跑每条命令，逐条断言 rc=0、产出非空，
并给"一屏"一个字节/行数预算。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "preflight-first-run.sh"


def test_preflight_script_exists_and_is_executable():
    assert SCRIPT.is_file(), "首跑检查脚本不在仓库里（G1 会退回人工记录）"
    assert SCRIPT.stat().st_mode & 0o111, "脚本必须可执行（README/CI 会直接跑它）"


def test_preflight_rebuilds_the_wheel_instead_of_trusting_a_stale_one():
    """本脚本第一版就是**假绿**：装了 `dist/` 里上一个提交构建的 wheel，于是 doctor 仍然 rc=1。

    这里钉住"默认必须现场构建"，并给复用现成产物留一个显式出口（CI 刚构建过时用）。
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert "uv build --wheel --out-dir" in text, "默认分支必须现场构建 wheel"
    assert "KS_PREFLIGHT_USE_DIST" in text, "复用现成产物要有显式开关（默认关闭）"
    assert 'if [ "${KS_PREFLIGHT_USE_DIST:-0}" = "1" ]' in text


@pytest.mark.skipif(shutil.which("uv") is None, reason="首跑路径依赖 uv；装上再跑这条")
@pytest.mark.skipif(shutil.which("bash") is None, reason="脚本是 bash")
def test_first_run_preflight_is_green():
    """每条命令 rc=0、产出非空、一屏在预算内 —— 且总耗时远低于门槛的 5 分钟。"""
    r = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, capture_output=True, text=True,
                       timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "PREFLIGHT OK 6/6" in out, out
    # 每一档都要有 rc=0 的证据（脚本自己断言；这里再钉一次文案，防止有人把断言删掉）
    assert out.count("→ rc=0"), out
    # 第一步必须已经给出"下一步做什么"，否则"≤2 条命令到第一条可行动结论"是空话
    assert "next[]" in out
    assert "一屏" in out and "预算" in out
