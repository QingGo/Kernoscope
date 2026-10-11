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
import sys

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


def _shell_code_lines(path: pathlib.Path) -> list[tuple[int, str]]:
    """`(行号, 行)`：跳过注释行与 **heredoc 正文**。

    heredoc（`<<'PY'` / `<<EOF`）里的内容 bash 一个字都不展开，所以里面的反引号、
    `$VAR（` 都是普通文本 —— 我的第一版门禁把 `setup-check.sh` 里嵌的 Python 当成 bash 判了，
    那正是"门禁自己的假阳性"。这里按起止标记跳过正文。
    """
    import re as _re

    out: list[tuple[int, str]] = []
    heredoc: str | None = None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        m = _re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if m:
            heredoc = m.group(1)
        if not line.lstrip().startswith("#"):
            out.append((n, line))
    return out


def test_no_shell_variable_is_followed_immediately_by_a_multibyte_char():
    import re

    pat = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(?=[\u0080-\uffff])")
    offenders: list[str] = []
    for p in _shell_files():
        for n, line in _shell_code_lines(p):
            for m in pat.finditer(line):
                offenders.append(f"{p.relative_to(ROOT)}:{n}: ${m.group(1)} 后面紧跟非 ASCII 字符")
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


# --------------------------------------------------------------------------
# 发布工作流（Trusted Publisher / OIDC）：形状本身也要有测试
# --------------------------------------------------------------------------
WF = ROOT / ".github" / "workflows" / "publish.yml"


def _workflow() -> str:
    assert WF.exists(), "发布工作流不见了（PyPI 上登记的就是这个文件名）"
    return WF.read_text(encoding="utf-8")


def test_publish_workflow_triggers_only_on_version_tags():
    wf = _workflow()
    assert re.search(r"tags:\s*\[?\"?v\*", wf), "只允许 tag 触发（推 main 不该发版）"
    # 注意：要查的是**触发器**（行首的 YAML 键），不是正文里有没有出现这个词 ——
    # 注释里写着"不用 pull_request_target"是完全正常的（第一版测试就栽在这个区别上）。
    assert not re.search(r"^\s*pull_request_target\s*:", wf, re.M), \
        "pull_request_target + OIDC = 给别人发版的机会"
    assert not re.search(r"^\s*workflow_dispatch\s*:", wf, re.M), \
        "手动触发会让'发版只由 tag 决定'这句话变假"


def test_oidc_permission_is_only_on_the_publish_job():
    wf = _workflow()
    build, _, publish = wf.partition("  publish:")
    assert "id-token" not in build, "build job 不该有 id-token（最小权限）"
    assert "id-token: write" in publish, "publish job 必须有 id-token: write 才能走 OIDC"
    assert "contents: read" in wf, "顶层权限应当只有 contents: read"


def test_publish_workflow_environment_matches_what_pypi_registered():
    """`environment:` 必须与 PyPI 上登记的一致，否则 OIDC 交换失败。

    这里只钉"仓库这一侧写了 pypi"以及"文档里写了同一件事"——PyPI 侧的值我读不到，
    所以两处必须同源：workflow 里的 `environment:` 与 RELEASING 里登记说明。
    """
    wf = _workflow()
    m = re.search(r"^\s*environment:\s*(\S+)", wf, re.M)     # 允许行尾注释
    assert m, "publish job 没写 environment（PyPI 那边登记了 environment 就必须对上）"
    env_name = m.group(1)
    assert env_name == "pypi", env_name
    rel = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    assert f"environment={env_name}" in rel or f"environment `{env_name}`" in rel, \
        "RELEASING 必须把登记用的 environment 名写出来（照抄才不会错）"


def test_publish_workflow_uses_the_attesting_action_without_any_token():
    """上传必须用会**附 attestation** 的那条路，且不给任何 token。

    为什么不能用 `uv publish`（第七轮发布实测）：它的 `--help` 里没有任何 attestation 开关，
    0.2.0 就是这样发出去的 —— PyPI Integrity API 对它返回 404「No provenance available」。
    改用 `pypa/gh-action-pypi-publish` 并在 0.3.0 验证 provenance 是否真的出现。
    """
    wf = _workflow()
    assert "pypa/gh-action-pypi-publish" in wf, "上传步骤必须用官方动作（它才会附 attestation）"
    assert re.search(r"attestations:\s*true", wf), "要显式打开 attestation"
    assert "UV_PUBLISH_TOKEN" not in wf, "工作流里不该出现任何 token 兜底"
    assert not re.search(r"^\s*run:\s*uv publish", wf, re.M), \
        "别再用 uv publish 上传（它不生成 attestation）"


def test_publish_workflow_guards_tag_version_and_reuses_the_metadata_checker():
    wf = _workflow()
    assert "pyproject version" in wf, "没有 tag↔版本护栏：v0.3.0 的 tag 可能发出 0.2.0 的包"
    assert "check_dist_metadata.py" in wf, \
        "CI 必须调用与本地预演**同一份**元数据断言脚本（各写一份就一定会漂）"
    assert "dev_index.py --gates" in wf, "要发布的那个提交自己必须先全绿"


def test_releasing_documents_the_oidc_path_and_the_emergency_token_path():
    rel = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    assert "Trusted Publisher" in rel and "pending publisher" in rel
    assert "publish.yml" in rel, "文档要写清 PyPI 那边登记的 workflow 文件名"
    assert "UV_PUBLISH_TOKEN" in rel, "应急 token 路径也要留着（CI 挂了还能发版）"
    assert "账号级" in rel and "project-scoped" in rel, \
        "首次发布只能用账号级 token、随后必须换项目级 —— 这条坑要写在文档里"


# --------------------------------------------------------------------------
# 元数据断言脚本：能红才有意义
# --------------------------------------------------------------------------
def test_metadata_checker_goes_red_on_a_bad_wheel(tmp_path):
    """喂一个"元数据不对"的假 wheel ⇒ 必须 rc=1 并逐条点名。

    真 wheel 由 `--metadata-only` 那条用例覆盖（它每次都现场构建）；这条钉的是**判据本身**
    会不会失效 —— 没有它，判据被删空也照样绿。
    """
    import zipfile

    wheel = tmp_path / "kernoscope-9.9.9-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("kernoscope-9.9.9.dist-info/METADATA",
                   "Name: kernoscope\nVersion: 9.9.9\n")          # 缺许可/Alpha/Python 下限
        z.writestr("kernoscope/cli.py", "")
        z.writestr("tests/test_leak.py", "")                        # 测试被打进包
    r = subprocess.run([sys.executable, str(ROOT / "bench" / "check_dist_metadata.py"),
                        str(wheel), "--expect-version", "0.2.0"],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    for needle in ("版本不是 0.2.0", "Alpha", "Python 下限", "entry_points", "测试被打进包"):
        assert needle in r.stdout, (needle, r.stdout)


def test_metadata_checker_passes_on_a_minimal_correct_wheel(tmp_path):
    import zipfile

    wheel = tmp_path / "kernoscope-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("kernoscope-0.2.0.dist-info/METADATA",
                   "Name: kernoscope\nVersion: 0.2.0\n"
                   "License-Expression: Apache-2.0\n"
                   "Classifier: Development Status :: 3 - Alpha\n"
                   "Requires-Python: >=3.10\n"
                   "Description-Content-Type: text/markdown\n")
        z.writestr("kernoscope-0.2.0.dist-info/entry_points.txt",
                   "[console_scripts]\nkernoscope = kernoscope.cli:main\nksc = kernoscope.cli:main\n")
        z.writestr("kernoscope/cli.py", "")
    r = subprocess.run([sys.executable, str(ROOT / "bench" / "check_dist_metadata.py"),
                        str(wheel), "--expect-version", "0.2.0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_unescaped_backticks_in_shell_code():
    """`.sh` 里**非注释行**不许出现没转义的反引号。

    为什么值得一条门禁：反引号在双引号里是**命令替换** —— 于是"打印一句带命令名的说明"会变成
    "真的执行那个命令"。本轮实测踩到两次：`say "…由 `uv publish` 发出…"` 让 `uv publish` 真的跑了一次
    （无凭据、安全失败，但那纯属运气）。要打印反引号就转义成 `\``（仓库里 5 处都这么写），
    要命令替换就用 `$( )`。
    """
    offenders: list[str] = []
    for path in _shell_files():
        for n, line in _shell_code_lines(path):
            for i, ch in enumerate(line):
                if ch == "`" and (i == 0 or line[i - 1] != "\\"):
                    offenders.append(f"{path.relative_to(ROOT)}:{n}")
                    break
    assert not offenders, (
        "这些行里有没转义的反引号（双引号里会被 bash 当命令替换执行）：\n  - "
        + "\n  - ".join(offenders) + "\n修法：打印用 \\`，命令替换用 $( )")
