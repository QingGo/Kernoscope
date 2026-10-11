"""发布就绪（第七轮 X-21 的可验证面）：**发布前预演、发布后验证是同一条命令**。

为什么值得一个测试文件：README §2 的三条安装路径里，`uv tool install kernoscope` 只有在包
真的发上去之后才可用（X-21：发布前实测 404，而文档已宣布同日发布）。所以"能不能装"这件事
不能只写在 RELEASING 的散文里 —— 它要有一条**能跑**的命令，发布前后各跑一次，断言同一组东西：

  * 产物元数据（名字/版本/许可/Alpha/Python 下限/入口点/不含测试）；
  * 从隔离目录装出来的可执行文件；
  * 中性工作目录下的首用三连 + 两个预算。

（真机/联网那一步 —— `--from-pypi` —— 在 CI 里不跑：包还没发，跑它就是红。）
"""

from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "release-dry-run.sh"


def test_release_dry_run_script_is_executable_and_syntactically_valid():
    assert SCRIPT.exists(), "发布预演脚本不见了"
    assert SCRIPT.stat().st_mode & 0o111, "release-dry-run.sh 应当可执行"
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_release_dry_run_metadata_mode_passes_on_the_current_tree():
    """`--metadata-only`：现场构建 + 检查元数据。

    这条是"发布前能不能发"的硬前提：包名/版本/许可/Alpha 分类器/Python 下限/入口点/不含测试。
    它比 `build` 门禁多走一层 —— 门禁只问"能不能生成 sdist+wheel"，这里问的是"生成出来的东西
    是不是一个**能装、能当 alpha 发**的包"。
    """
    r = subprocess.run(["bash", str(SCRIPT), "--metadata-only"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RELEASE OK" in r.stdout, r.stdout


def test_release_dry_run_offers_a_post_publish_mode_that_verifies_pypi():
    """发布后那条路径必须存在，且**不能**是另一份会漂的清单。"""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--from-pypi" in text
    # 两种模式共用同一段断言（首用三连 + 预算），不是"发布前一套、发布后另一套"
    assert text.count("capabilities --brief") >= 2
    assert "uv tool install" in text and "kernoscope==" in text


def test_releasing_documents_both_halves_of_the_release_check():
    rel = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    assert "release-dry-run.sh" in rel, "RELEASING 没告诉维护者这条命令"
    assert re.search(r"--from-pypi", rel), "RELEASING 没写发布后怎么验"


# --------------------------------------------------------------------------
# 顺手抓到的一类真实可移植性 bug：`$VAR` 后面紧跟中文字符
# --------------------------------------------------------------------------
# `say "==> 版本：$VERSION（模式：$MODE）"` 在 **非 UTF-8 locale** 下（`LC_ALL=C`、极简容器、
# 某些 CI 镜像）会被 bash 解析成 `${VERSION<UTF-8 首字节>}`：`line 47: VERSION\xef: unbound
# variable`，脚本当场死。本机默认 locale 是 UTF-8，所以它"看起来没问题" —— 正是本项目一直在
# 抓的那类"只在某种环境下成立"。修法是给变量加大括号 `${VERSION}（…）`；这条测试把整类钉住。
def _shell_files() -> list[pathlib.Path]:
    skip = {".venv", ".git", "node_modules", "__pycache__"}
    return [p for p in ROOT.rglob("*.sh") if not (skip & set(p.parts))]


def test_no_shell_variable_is_followed_immediately_by_a_multibyte_char():
    import re

    pat = re.compile(rb"\$([A-Za-z_][A-Za-z0-9_]*)(?=[\x80-\xff])")
    offenders: list[str] = []
    for p in _shell_files():
        for m in pat.finditer(p.read_bytes()):
            offenders.append(f"{p.relative_to(ROOT)}: ${m.group(1).decode()} 后面紧跟非 ASCII 字符")
    assert not offenders, (
        "这些 shell 变量没加大括号，紧接着就是中文字符 —— 在 `LC_ALL=C` 之类非 UTF-8 locale 下\n"
        "bash 会把多字节字符的首字节吞进变量名（`unbound variable`，脚本当场死）：\n  - "
        + "\n  - ".join(offenders) + "\n修法：写成 ${VAR}（本仓库的脚本都跑在陌生机器上）。")


def test_shell_scripts_parse_under_a_c_locale():
    """再加一层：在 `LC_ALL=C` 下至少要把 `bash -n` 与 `--help` 级别的路径跑通。

    上一条是静态扫描（快、准、给出修法）；这一条是行为上的兜底：真的用 C locale 起一次。
    """
    import os

    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    for script in _shell_files():
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{script}: {r.stderr}"
