"""W9「开窗包」的门禁：首屏契约、覆盖矩阵同源、根文档措辞、issue 模板。

这里守的不是功能，而是**公开形态**：
  * README 第一个 `---` 之前 ≤10 行，且必须给出"不能回答什么"与"30 秒怎么用"；
  * COVERAGE.md 的问题 id 集合必须与 `kernoscope.capabilities.QUESTIONS` 一致
    （手改生成物会在下一次生成时被覆盖，所以这里对拍的是代码）；
  * 仓库根文档里不得出现会暴露维护者工作方式的措辞；
  * issue 模板必须存在且是合法 YAML（没有 PyYAML 时退回结构断言，不引入新依赖）。

为什么首屏要 ≤10 行：公开的实质是**让别人准确地知道我们是什么**。首屏是绝大多数人
唯一会读的部分；它一旦需要"再往下读第 N 节"才能回答"这工具不能做什么"，
那就等于没有回答。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRST_SCREEN_MAX_LINES = 10

# 这些词会暴露"另有一套只在维护者机器上存在的文档"，本身即线索。
BANNED_IN_ROOT_DOCS = ("内部文档", "本地文档", "不同步")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1) README 首屏
# --------------------------------------------------------------------------
def _first_screen_lines(text: str) -> list[str]:
    """第一个 `---` 之前的行（不含 `---` 本身）。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "---":
            return lines[:i]
    return lines


def test_readme_first_screen_is_short():
    screen = _first_screen_lines(_read("README.md"))
    assert len(screen) <= FIRST_SCREEN_MAX_LINES, (
        f"README 首屏 {len(screen)} 行，超过 {FIRST_SCREEN_MAX_LINES} 行。"
        f"公开的实质是让别人 30 秒内知道我们是什么。")
    # `---` 紧贴正文会被 Markdown 当成 setext 标题（H2），首屏就不再是首屏了
    lines = _read("README.md").splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip() == "---")
    assert lines[idx - 1].strip() == "", "第一个 `---` 前必须有空行，否则它会被渲染成标题"


def test_readme_first_screen_says_what_it_cannot_do():
    text = "\n".join(_first_screen_lines(_read("README.md")))
    assert "不能回答" in text or "不回答" in text, \
        "首屏必须直接说清「它不能回答什么」（这是诚实文档的最低要求）"


def test_readme_first_screen_has_a_30_second_path():
    text = "\n".join(_first_screen_lines(_read("README.md")))
    assert "30 秒" in text, "首屏必须给出 30 秒可用的入口"
    assert "kernoscope doctor" in text, "30 秒路径要是一条真命令"


def test_readme_first_screen_positions_against_neighbours():
    text = "\n".join(_first_screen_lines(_read("README.md")))
    for tool in ("ncu", "nsys", "torch.profiler"):
        assert tool in text, f"首屏必须说清与 {tool} 的关系"


def test_readme_has_a_standalone_limitations_section():
    """「局限与已知失败」必须是独立一节，且点到几条硬事实。"""
    text = _read("README.md")
    assert re.search(r"^##\s+局限与已知失败", text, re.M), "缺少独立的「局限与已知失败」一节"
    for needle in ("计数器", "单边偏高", "1 个百分点", "MDD", "拒绝给数"):
        assert needle in text, f"局限一节缺少「{needle}」"


# --------------------------------------------------------------------------
# 2) COVERAGE.md 与代码同源
# --------------------------------------------------------------------------
_QUESTION_IDS_RE = re.compile(r"<!--\s*question ids[^:]*:\s*(.*?)-->", re.S)


def test_coverage_lists_exactly_the_code_questions():
    from kernoscope.capabilities import QUESTIONS

    text = _read("COVERAGE.md")
    m = _QUESTION_IDS_RE.search(text)
    assert m, "COVERAGE.md 缺少由代码生成的 question ids 注释（请重跑 tools/gen_coverage.py）"
    listed = {x.strip() for x in m.group(1).replace("\n", " ").split(",") if x.strip()}
    code = {q.id for q in QUESTIONS}
    assert listed == code, (
        f"COVERAGE.md 的问题集合与代码不一致：\n"
        f"  只在文档里: {sorted(listed - code)}\n"
        f"  只在代码里: {sorted(code - listed)}")
    for qid in code:
        assert f"`{qid}`" in text, f"COVERAGE.md 的矩阵里没有 {qid}"


def test_coverage_says_it_is_generated_from_code():
    text = _read("COVERAGE.md")
    assert "kernoscope capabilities" in text
    assert "capabilities.py" in text, "必须写清改动请改哪里"


def test_coverage_covers_three_states_and_all_tiers():
    from kernoscope.capabilities import TIERS

    text = _read("COVERAGE.md")
    for state in ("能答", "更弱", "不答"):
        assert state in text, f"三态覆盖矩阵缺少「{state}」"
    for t in TIERS:
        assert f"`{t.name}`" in text, f"能力分层缺少层 {t.level}（{t.name}）"


# --------------------------------------------------------------------------
# 3) 根文档措辞
# --------------------------------------------------------------------------
@pytest.mark.parametrize("doc", ["README.md", "COVERAGE.md", "CONTRIBUTING.md"])
def test_root_docs_avoid_leaky_wording(doc):
    text = _read(doc)
    for bad in BANNED_IN_ROOT_DOCS:
        assert bad not in text, f"{doc} 出现了会暴露维护者工作方式的措辞：{bad!r}"


def test_root_docs_do_not_enumerate_private_doc_names():
    """根文档只描述仓库里存在的东西：不得列出 `docs/` 下的具体文件名。"""
    pattern = re.compile(r"docs/[A-Za-z0-9._-]+")
    for doc in ("README.md", "COVERAGE.md", "CONTRIBUTING.md"):
        hits = pattern.findall(_read(doc))
        assert not hits, f"{doc} 枚举了工作目录里的文件名：{hits}"


# --------------------------------------------------------------------------
# 4) issue 模板
# --------------------------------------------------------------------------
TEMPLATES = ("bug_report.yml", "capability_request.yml", "metric_mapping.yml", "config.yml")
TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"
_ALLOWED_TYPES = {"markdown", "input", "textarea", "dropdown", "checkboxes"}


def _load_yaml(text: str):
    """能 import yaml 就用它；不能就退回 None（用结构断言，不引入新依赖）。"""
    try:
        import yaml            # type: ignore
    except Exception:          # pragma: no cover - 环境无 PyYAML 时走结构断言
        return None
    return yaml.safe_load(text)


@pytest.mark.parametrize("name", TEMPLATES)
def test_issue_template_exists_and_parses(name):
    path = TEMPLATE_DIR / name
    assert path.is_file(), f"缺少 issue 模板 {name}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{name} 是空文件"
    assert "\t" not in text, f"{name} 含制表符（YAML 缩进不允许 tab）"

    data = _load_yaml(text)
    if data is None:
        # 无 PyYAML：只断言结构骨架（够抓住"写成散文"这类错误）
        if name == "config.yml":
            assert "blank_issues_enabled" in text
            assert "contact_links" in text
        else:
            for key in ("name:", "description:", "body:"):
                assert re.search(rf"^{key}", text, re.M), f"{name} 缺少顶层键 {key}"
            types = set(re.findall(r"^\s*-\s*type:\s*([a-z]+)", text, re.M))
            assert types, f"{name} 的 body 里没有任何 element"
            assert types <= _ALLOWED_TYPES, f"{name} 有非法 element 类型：{types - _ALLOWED_TYPES}"
        return

    assert isinstance(data, dict), f"{name} 顶层必须是映射"
    if name == "config.yml":
        assert data.get("blank_issues_enabled") is False
        links = data.get("contact_links") or []
        assert links and all({"name", "url", "about"} <= set(link) for link in links)
        return
    for key in ("name", "description", "body"):
        assert data.get(key), f"{name} 缺少顶层键 {key}"
    body = data["body"]
    assert isinstance(body, list) and body, f"{name} 的 body 不能为空"
    for element in body:
        assert element.get("type") in _ALLOWED_TYPES, f"{name} 有非法 element：{element}"
        if element["type"] != "markdown":
            assert element.get("id"), f"{name} 的非 markdown element 缺少 id"


def test_issue_templates_ask_for_the_self_check_outputs():
    """每个模板都要引导提交者先跑两条自检并贴输出（否则维护者无法分诊）。"""
    for name in ("bug_report.yml", "capability_request.yml", "metric_mapping.yml", "config.yml"):
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert "capabilities --json" in text, f"{name} 没要求贴 capabilities --json"
        assert "doctor --json" in text, f"{name} 没要求贴 doctor --json"


# --------------------------------------------------------------------------
# 5) 打包元数据（PyPI 就绪，但不发布）
# --------------------------------------------------------------------------
def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 没有 tomllib
        pytest.skip("tomllib 需要 Python ≥3.11（3.10 上跳过打包元数据断言）")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_packaging_metadata_points_at_the_real_repository():
    data = _load_toml(ROOT / "pyproject.toml")
    urls = data["project"].get("urls", {})
    for key in ("Homepage", "Issues", "Documentation", "Repository"):
        assert key in urls, f"pyproject 缺少 {key} URL"
        assert urls[key].startswith("https://github.com/QingGo/Kernoscope"), \
            f"{key} 指向了不存在的地址：{urls[key]}"
    # `docs/` 不进仓库，所以 Documentation 不能指向它
    assert "/tree/main/docs" not in urls["Documentation"]


def test_packaging_metadata_is_consistent_with_reality():
    from kernoscope import __version__

    data = _load_toml(ROOT / "pyproject.toml")
    proj = data["project"]
    assert proj["version"] == __version__
    assert proj["dependencies"] == [], "本项目运行时零第三方依赖"
    assert proj["readme"] == "README.md"
    assert proj["requires-python"] == ">=3.10"
    assert proj["license"] == "Apache-2.0"
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "Version 2.0" in license_text, \
        "pyproject 声明的 license 必须与 LICENSE 文件一致"
    scripts = proj.get("scripts", {})
    assert scripts.get("kernoscope") and scripts.get("ksc")
    extras = proj.get("optional-dependencies", {})
    assert "nvml" in extras
    # 声明了 extra 却没有任何代码用到的依赖就不该出现（诚实性：不承诺未实现的东西）
    assert "analyze" not in extras
