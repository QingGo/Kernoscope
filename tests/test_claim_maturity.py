"""R-107/R-90：**"真值对拍"这句话只能按"包装逻辑已验；真报告口径未验证"读**。

第六轮验收 §3 的原话：仓库里没有真 `.ncu-rep`（唯一那份的头是 `kernoscope-substitute-v1`）、
`test_ncu_rep.py` 往 `sys.modules` 里塞假模块、`test_truth_and_substitute.py` 手搓 dataclass ——
而 README/CHANGELOG 写着"真值对拍"。选定的收口方式是 R-90 的**选项②**：把声明降级
（不是入库真报告：采一份真报告需要一台 `ncu` 能收集计数器的机器，那正是本项目要绕开的前提）。

这条测试是"不许被静默改回去"的那道闸：

* 降级句必须出现在 **README §4.3** 与 **CHANGELOG** 里；
* 降级的**理由**（仓库里没有真报告 + 真报告需要能采的机器）必须写在 §4.3；
* 仓库里**确实**还没有真报告 —— 一旦有真报告入库（情况①），这条会红，提醒作者把措辞
  升级回"已验证"并补端到端对拍测试；
* 少数几种"直接改口成已验证"的写法被显式列为禁止（黑名单只是兜底，真正的防线是上面三条：
  删掉降级句就红）。

改这段措辞时，请连着 README §5 的能力表、§8 输出清单、§11 的 v1.9 那段一起改 ——
它们是同一句话的四处视图。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"

#: R-90 选项② 的**规范措辞**（README 与 CHANGELOG 必须逐字包含它）。
DOWNGRADE = "包装逻辑已验；真报告口径未验证"

#: 降级的两个事实依据（README §4.3 要说清"为什么"）。
REASONS = ("没有一份真 ncu 报告", "能收集计数器的机器")

#: `ncu` 采出来的报告里不会出现本工具的**署名标记**；我们的写入器（含最小报告）都会留痕。
#: R-135：以前是裸词 `b"kernoscope"` —— 它可能出现在任何无辜的字串里（路径、注释、别人报告里的
#: 文本），于是"这是不是我们写的"会误判。改用写入器自己的常量（唯一真源，见 ncrep/substitute.py）。
from kernoscope.ncrep.substitute import ATTRIBUTION_JSON_MARK, ATTRIBUTION_MARK  # noqa: E402

_OURS = (ATTRIBUTION_MARK.encode(), ATTRIBUTION_JSON_MARK.encode())


def _section(text: str, heading: str) -> str:
    """取 `### <heading>` 到下一个同级标题之间的正文。"""
    m = re.search(rf"^### {re.escape(heading)}.*?(?=^### |\Z)", text, re.S | re.M)
    return m.group(0) if m else ""


def _find_real_ncu_reports() -> list[Path]:
    """仓库里**不是**本工具写的 `.ncu-rep`（= 别人用 ncu 采的真报告）。"""
    out: list[Path] = []
    for p in ROOT.rglob("*.ncu-rep*"):
        if not p.is_file() or ".venv" in p.parts:
            continue
        try:
            blob = p.read_bytes()
        except OSError:                                     # pragma: no cover
            continue
        if not any(mark in blob for mark in _OURS):
            out.append(p)
    return sorted(out)


def test_truth_comparison_claim_is_downgraded_in_readme_and_changelog():
    readme = README.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    assert DOWNGRADE in readme, (
        f"README 缺少降级措辞「{DOWNGRADE}」—— R-107/R-90 的选项②要求把"
        "「真值对拍（I1）」从『已验证』降级成这句话（理由：仓库里没有真报告，"
        "采一份需要一台 ncu 能收集计数器的机器）。四处视图都要改：§4.3 正文、"
        f"§5 能力表那一行、§8 输出清单、§11 的 v1.9 那段")
    assert DOWNGRADE in changelog, (
        f"CHANGELOG 缺少降级措辞「{DOWNGRADE}」—— 同一句话在 CHANGELOG 的"
        "「真值对拍与写 `.ncu-rep`（I1/I2）」一节里也必须降级")
    # §4.3 必段出现一次（不能只在边角里提一句就算数）
    sec = _section(readme, "4.3 导入真值")
    assert sec, "README 里找不到 §4.3 小节（标题被改过？）"
    assert DOWNGRADE in sec, f"§4.3 正文里没有降级措辞：\n{sec[:600]}"


def test_the_downgrade_says_why():
    """降级必须给出理由，否则读者无法判断什么时候能升级回去。"""
    sec = _section(README.read_text(encoding="utf-8"), "4.3 导入真值")
    for why in REASONS:
        assert why in sec, (f"§4.3 没写清降级的理由（缺 {why!r}）：采一份真报告需要一台"
                            f"`ncu` 能收集计数器的机器，而我们手上没有真报告")


def test_repo_really_has_no_real_ncu_report_yet():
    """降级的**事实依据**：仓库里没有真报告。

    一旦有人入库一份真报告（R-90 的选项①），这条会红 —— 那是好事：把措辞升级回
    "真值对拍已验证"并补一条端到端测试（读它 → 出真值行 → 对拍）之后再删掉这条。
    """
    real = _find_real_ncu_reports()
    assert not real, (
        "仓库里出现了不是本工具写的 `.ncu-rep`（可能是别人用 ncu 采的真报告）："
        f"{[str(p.relative_to(ROOT)) for p in real]} —— 请按 R-90 选项①收口："
        "确认它已脱敏、补一条端到端对拍测试，然后把 README/CHANGELOG 的"
        f"「{DOWNGRADE}」升级回「真值对拍已验证」，并删掉这条断言")


def test_no_plain_re_upgrade_phrasing_is_left_behind():
    """兜底黑名单：几种"直接改口成已验证"的写法不许出现。"""
    text = README.read_text(encoding="utf-8") + CHANGELOG.read_text(encoding="utf-8")
    banned = ("真值对拍已验证", "真报告口径已验证", "已在真报告上验证",
              "真 ncu 报告已入库", "真报告对拍已验")
    hit = [b for b in banned if b in text]
    assert not hit, (f"这些说法把未验证的东西写成了已验证：{hit} —— "
                     f"只能用「{DOWNGRADE}」加理由的写法")


def test_readme_discloses_that_the_docs_evidence_is_not_in_the_repo():
    """R-108/R-91：(c) 的收口方式是**披露**，不是把 `docs/` 纳入版本控制。

    `docs/` 整目录被 `.gitignore` 忽略（`tests/test_repo_hygiene.py` 守着"不许有取反行"），
    所以 `docs/acceptance/` 里的 m6 真机证据克隆者与 CI 都看不到。README 必须明说这一点 ——
    否则"证据在 docs/ 里"会被读成"证据可复核"。
    """
    readme = README.read_text(encoding="utf-8")
    assert ".gitignore" in readme and "acceptance/" in readme, (
        "README 必须写清：`acceptance/` 那套 m6 真机证据所在的本地工作目录被 `.gitignore` "
        "整目录忽略（R-91/R-108）")
    assert "不构成可复核证据" in readme, (
        "披露的结论句不能省：不入库的证据就是不构成可复核证据（要复核得在自己的机器上重采）")
    # 披露必须与 `.gitignore` 的实际状态一致（把 docs/ 拉回版本控制会让上面那句话变假）
    ignored = [l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "docs/" in ignored, "`.gitignore` 必须继续整目录忽略 docs/（否则披露失效）"
    assert not [l for l in ignored if l.startswith("!") and "docs" in l], \
        "不许出现 docs 的取反行（那会让'不入库'的披露变成假话）"
