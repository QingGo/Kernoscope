"""AI-first 入口层的契约测试（L0/L1 + 生成的 L3 参考）。

这一层要守的不是功能，而是**预算与同源**：入口层一旦长大，渐进披露就退化回"每次都付
全量上下文的钱"；参考层一旦手抄，就会和代码不一致。

  * 预算：AGENTS.md ≤ 40 行、SKILL.md ≤ 150 行、L0 三层合计 ≤ 4 KB；
  * 触发：SKILL.md 的 frontmatter 必须有 name/description，且 description 长度在
    60–800 字符之间（太短说不清"什么时候用"，太长就不是"常驻元数据"了）；
  * 同源：references/ 下的生成文件必须能 `--check` 通过；每个拒绝码、每个判词 id
    都要在文件里出现（否则 agent 查不到）；
  * 可用：四个脚本 `bash -n` 通过、`--help` exit 0、`remedy.sh <code>` 能查到具体 code；
  * 干净：面向 agent 的文件里不得出现本地文档目录的引用。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernoscope.packet import BUILDERS, VERDICT_IDS  # noqa: E402
from kernoscope.provenance import Provenance  # noqa: E402
from kernoscope.refusals import RefusalCode  # noqa: E402

AGENTS = ROOT / "AGENTS.md"
LLMS = ROOT / "llms.txt"
SKILL_DIR = ROOT / "skills" / "kernoscope"
SKILL = SKILL_DIR / "SKILL.md"
REFS = SKILL_DIR / "references"
SCRIPTS = SKILL_DIR / "scripts"


def _lines(p: pathlib.Path) -> int:
    return len(p.read_text(encoding="utf-8").splitlines())


# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------
def test_agents_md_is_an_entry_point_not_a_manual():
    """`AGENTS.md` 是**开发侧**入口（A 轨）：这里只守预算与体量。

    受众分流（开发入口不带使用面的闭环命令、使用面入口必须带）由
    `tests/test_dev_index.py` 守。
    """
    assert AGENTS.exists(), "仓库根应当有 AGENTS.md（给改这个仓库的 agent 的入口）"
    text = AGENTS.read_text(encoding="utf-8")
    assert _lines(AGENTS) <= 40, f"AGENTS.md 已经 {_lines(AGENTS)} 行，入口层不许长成手册"
    assert len(text.encode("utf-8")) <= 2500


def test_skill_md_has_a_triggerable_description():
    assert SKILL.exists()
    text = SKILL.read_text(encoding="utf-8")
    assert _lines(SKILL) <= 150, f"SKILL.md 已经 {_lines(SKILL)} 行"
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md 必须以 YAML frontmatter 开头"
    fm = m.group(1)
    assert re.search(r"^name:\s*kernoscope\s*$", fm, re.M), "frontmatter 缺 name"
    desc = re.search(r"^description:\s*>?-?\s*\n?((?:.|\n)*?)(?=\n\w+:|\Z)", fm, re.M)
    assert desc, "frontmatter 缺 description（它是唯一常驻上下文的触发词）"
    words = " ".join(desc.group(1).split())
    assert 60 <= len(words) <= 800, f"description 长度 {len(words)} 不在 60–800 之间"
    # 触发词要同时说清"什么时候用"与"什么时候不要用"
    assert "不要用" in words, "description 必须写明什么时候不要用它"
    # 使用面（B 轨）的闭环必须在 SKILL.md 里 —— 它是 B 轨唯一的常驻入口
    body = text
    for needle in ("capabilities --brief", "--packet", "actionable", "indistinguishable"):
        assert needle in body, f"使用面入口少了 {needle!r}"


def test_entry_layer_stays_small():
    """L0 三层合计 ≤ 4 KB（常驻成本）。"""
    total = 0
    for p in (AGENTS, LLMS):
        total += len(p.read_text(encoding="utf-8").encode("utf-8"))
    fm = re.match(r"^---\n(.*?)\n---\n", SKILL.read_text(encoding="utf-8"), re.S)
    total += len(fm.group(0).encode("utf-8"))
    assert total <= 4096, f"入口层常驻部分已经 {total} B"


def test_llms_txt_follows_the_convention():
    assert LLMS.exists()
    text = LLMS.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    assert lines[0].startswith("# "), "llms.txt 第一行应当是 H1 名字"
    assert any(l.lstrip().startswith("> ") for l in lines[:5]), "开头要有一句 blockquote 概述"
    assert "## Optional" in text, "llms.txt 约定里 Optional 段用来放可跳过的东西"


# --------------------------------------------------------------------------
# 链接与引用
# --------------------------------------------------------------------------
@pytest.mark.parametrize("doc", ["AGENTS.md", "llms.txt", "skills/kernoscope/SKILL.md",
                                 "skills/kernoscope/references/provenance-rules.md",
                                 "skills/kernoscope/references/recipes.md"])
def test_relative_links_resolve(doc):
    path = ROOT / doc
    text = path.read_text(encoding="utf-8")
    bad: list[str] = []
    for target in re.findall(r"\]\(([^)#\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).resolve().exists():
            bad.append(target)
    assert not bad, f"{doc} 里有失效的相对链接：{bad}"


@pytest.mark.parametrize("doc", ["AGENTS.md", "llms.txt", "skills/kernoscope/SKILL.md",
                                 "skills/kernoscope/references/provenance-rules.md",
                                 "skills/kernoscope/references/recipes.md",
                                 "skills/kernoscope/references/refusal-codes.md",
                                 "skills/kernoscope/references/verdicts.md"])
def test_agent_facing_files_do_not_reference_local_doc_dir(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "docs/" not in text, f"{doc} 里出现了本地文档目录的引用"


# --------------------------------------------------------------------------
# 生成的参考层：与代码同源
# --------------------------------------------------------------------------
def test_generated_refs_are_in_sync():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_agent_refs.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_generated_refs_carry_the_do_not_edit_header():
    for name in ("refusal-codes.md", "verdicts.md"):
        text = (REFS / name).read_text(encoding="utf-8")
        assert text.startswith("<!-- 由 tools/gen_agent_refs.py 生成"), name


def test_every_refusal_code_is_documented():
    text = (REFS / "refusal-codes.md").read_text(encoding="utf-8")
    missing = [c.value for c in RefusalCode if f"`{c.value}`" not in text]
    assert not missing, f"参考层漏了这些拒绝码：{missing}"


def test_every_verdict_id_is_documented():
    text = (REFS / "verdicts.md").read_text(encoding="utf-8")
    missing = [v for v in VERDICT_IDS if f"`{v}`" not in text]
    assert not missing, f"参考层漏了这些判词：{missing}"


def test_verdict_catalog_is_exactly_what_builders_can_emit(monkeypatch):
    """判词目录必须与 BUILDERS 实际能产出的 id 完全一致（否则生成文档会漏判词）。"""
    from kernoscope import packet as pk

    values = {
        "est__dram_bytes_total": 5.0e9, "ks__sampler_window_coverage_pct": 97.1,
        "ks__dram_window_mem_busy_mean_pct": 96.8, "est__dram_bytes_per_launch": 2.5e7,
        "ks__matched_kernel_kernel_share_pct": 99.1,
        "ks__attribution_bias_estimate_pct": 0.9,
        "est__dram_bytes_per_launch_trace": 2.2e7, "ks__dram_trace_vs_nvml_pct": 88.0,
        "est__dram_vs_ledger_err_pct": 8.1, "ks__theoretical_occupancy_pct": 50.0,
        "meas__achieved_occupancy_pct": 46.9, "ks__occupancy_limiter": "registers",
        "est__bound": "memory", "est__roofline_efficiency_pct": 91.2,
        "est__achieved_gbs": 908.5, "est__issue_slot_utilization_pct": 27.4,
        "est__stall_direction": "latency", "est__l2_hit_rate_pct": 66.7,
        "est__working_set_bytes": 3.4e7, "ks__over_peak_warning": 1.0,
        "ks__static_sass_instructions": 1234,
        # 成本账（C1）：单位成本是主线判词，替身必须能产出它，否则"判词目录与实际一致"
        # 这条测试会因为替身缺字段而**假绿**（漏掉 cost_per_token）。
        "ks__cost_per_mtok": 12.5, "ks__cost_per_hour": 3.25,
        "ks__tokens_per_s": 1.2e4, "ks__j_per_token": 0.35,
        "meas__window_energy_j": 4200.0, "est__kernel_cost_per_hour": 2.1,
    }

    class FakeMetric:
        def __init__(self, name, value):
            self.name, self.value, self.note = name, value, "note"
            self.provenance = Provenance.ESTIMATED
            # 与真 Metric 同构的字段：R-51 之后 packet 会读 unit 做字节换算，
            # 这个替身少了它就会炸 —— 替身与真类型不一致时，测的是替身不是产品
            self.unit = "B"
            self.refusal = None

    monkeypatch.setattr(pk, "find_metric", lambda rep, name: FakeMetric(name, values.get(name)))
    monkeypatch.setattr(pk, "metric_value", lambda rep, name: values.get(name))

    class FakeSection:
        def __init__(self, metrics):
            self.name, self.metrics = "SpeedOfLight", metrics

    class FakeKernel:
        name, total_ns, launch_count = "k", 1_000_000_000, 10
        # 判词是**绑在 kernel 上**的（第五轮 V-6：跨 kernel 传染），所以替身必须有
        # 与真 Report 同构的 sections —— 否则测的是替身，不是产品。
        sections = [FakeSection([FakeMetric(n, v) for n, v in values.items()])]

    class FakeRep:
        kernels = [FakeKernel()]
        wall_time_ns = 2_000_000_000
        # 真值对拍判词（I1）只有在导入了 .ncu-rep 时才存在；替身必须能表达这个状态，
        # 否则"判词目录与实际一致"这条测试会因为替身缺字段而漏掉 truth_vs_substitute。
        meta = {
            "ncu_truth": {"ncu_version": "2025.3.1", "actions": 1, "pairs_compared": 2,
                          "pairs_total": 3, "max_abs_err_pct": 4.5},
            # 处方判词（C2）同理：替身必须能表达"有头寸"的状态
            "prescription": {
                "currency": "CNY",
                "ranked": [{"kernel": "k", "resource": "dram_bandwidth",
                            "confidence": "measured", "pct_lo": 10.0, "pct_hi": 20.0,
                            "per_hour": 1.5, "per_mtok": None, "basis": "b",
                            "refusal": None, "money_refusal": None}],
                "total_per_hour": 1.5, "total_per_mtok": None, "refusal": None,
                "money_refusal": None, "notes": []},
        }

    ids = set()
    for fn in BUILDERS:
        v = fn(FakeRep())
        if v is not None:
            ids.add(v.id)
    assert ids == set(VERDICT_IDS), (sorted(ids), sorted(VERDICT_IDS))


# --------------------------------------------------------------------------
# 脚本可用
# --------------------------------------------------------------------------
def test_scripts_exist_executable_and_syntactically_valid():
    expected = {"_ks.sh", "setup-check.sh", "evidence.sh", "compare.sh", "remedy.sh"}
    got = {p.name for p in SCRIPTS.iterdir() if p.suffix == ".sh"}
    assert expected <= got, expected - got
    for name in expected:
        p = SCRIPTS / name
        assert p.stat().st_mode & 0o111, f"{name} 应当可执行"
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


@pytest.mark.parametrize("script", ["setup-check.sh", "evidence.sh", "compare.sh", "remedy.sh"])
def test_script_help_works_without_gpu(script):
    r = subprocess.run(["bash", str(SCRIPTS / script), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert len(r.stdout.strip()) > 20


def test_remedy_script_answers_one_code_without_reading_the_whole_table():
    r = subprocess.run(["bash", str(SCRIPTS / "remedy.sh"), "share_too_low"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "share_too_low" in r.stdout
    assert "怎么办" in r.stdout
    assert len(r.stdout.encode("utf-8")) < 800      # 一条 code 的处置必须很小


def test_remedy_script_rejects_unknown_code():
    r = subprocess.run(["bash", str(SCRIPTS / "remedy.sh"), "not_a_code"],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "没有这个 code" in r.stderr


def test_compare_script_refuses_without_mdd(tmp_path):
    """没有本机误差带时 compare.sh 必须 exit 2（不下结论），而不是给出一个方向。"""
    env_off = tmp_path / "missing.json"
    r = subprocess.run(["bash", str(SCRIPTS / "compare.sh"), str(tmp_path / "a.json"),
                        str(tmp_path / "b.json")],
                       capture_output=True, text=True,
                       env={"KS_SELFCHECK": str(env_off), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 2
    assert "没有本机误差带" in r.stderr


def test_setup_check_script_prints_one_small_json_object():
    r = subprocess.run(["bash", str(SCRIPTS / "setup-check.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)                     # stdout 必须是纯 JSON
    assert "blocked_by" in d and "next" in d
    assert len(r.stdout.encode("utf-8")) < 1500, "入口层不许变大"


def test_nothing_promises_selftest_output_as_a_calibration():
    """R-66 / D14 残留：**任何地方**都不许说 selftest 的产物能当 `--ks-calibration`。

    第四轮验收报告的原形：`PEAK_CAVEAT` 写"不要当 --ks-calibration"，而同一个文件的模块
    docstring、`--out` 帮助文本、以及 `no_calibration` 的处方命令三处都在许诺它能当 ——
    读者照着做会拿到 >100% 的物理不可能值。所以这里扫**源码 + 生成物 + 文档**。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in [*root.glob("*.md"), *root.glob("src/**/*.py"), *root.glob("skills/**/*.md")]:
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            # 只查"把 selftest/自检 与 --ks-calibration 连起来说"的行 ——
            # 普通的 `--ks-calibration machine-profile.json` 命令当然允许
            if "ks-calibration" not in low or not ("selftest" in low or "自检" in low
                                                   or "machine-selfcheck" in low):
                continue
            # 只有"否定式"的说法才允许（不要/不能/别/不是/参考）
            if re.search(r"(不要|不能|别|不是|仅供参考|不可)", line):
                continue
            offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()[:100]}")
    assert not offenders, ("这些地方还在暗示 selftest 的产物能当 --ks-calibration：\n  - "
                           + "\n  - ".join(offenders))


_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
# 「命令 | 开关 | 作用」表：表头里要同时有**开关列**与**说明列**，同一行的说明列非空
# 才算一次登记（开关名 + 一句话说明）。
_TABLE_FLAG_HEADER = "开关"
_TABLE_PURPOSE_HEADER = ("作用", "说明", "意义")

# 与 `tests/test_documented_commands.py` 同一个出口（V-24）：导出树里的显式降级开关。
EMPTY_DOC_SCAN_ENV = "KS_ALLOW_EMPTY_DOC_SCAN"

# argparse 在每个解析器上**自动**加的开关：不是我们声明的接口，因此不要求逐条登记。
# （旧实现里 capabilities/selftest/diff 的 `--help` 是靠 CHANGELOG **散文**里两处
# `` `--help` `` 才"满足"的 —— 那正是 V-22 说的自我满足。）
PARSER_BUILTIN_FLAGS = frozenset({"--help"})


def _fenced_texts(text: str) -> list[str]:
    """``` 围栏里的内容（只看围栏，不看散文）。"""
    out: list[str] = []
    inside = False
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if inside:
                out.append("\n".join(buf))
                inside, buf = False, []
            else:
                inside = True
            continue
        if inside:
            buf.append(line)
    return out


def _table_documented_flags(text: str) -> dict[str, str]:
    """「命令 | 开关 | 作用」表里登记过的开关 → 那一行的说明。

    只有"开关列里有名字 **且** 说明列非空"才算登记 —— 空说明的表格行不构成声明。
    """
    out: dict[str, str] = {}
    flag_col: int | None = None
    purpose_col: int | None = None
    for row in text.splitlines():
        s = row.strip()
        if not s.startswith("|"):
            flag_col = purpose_col = None            # 表格结束，重新找表头
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if flag_col is None:
            if any(_TABLE_FLAG_HEADER in c for c in cells):
                flag_col = next(k for k, c in enumerate(cells) if _TABLE_FLAG_HEADER in c)
                purpose_col = next((k for k, c in enumerate(cells)
                                    if any(w in c for w in _TABLE_PURPOSE_HEADER)), None)
            continue
        if set(s) <= set("|-: "):                    # 对齐行
            continue
        purpose = cells[purpose_col] if (purpose_col is not None
                                         and purpose_col < len(cells)) else ""
        if not purpose or flag_col >= len(cells):
            continue
        for f in _FLAG_RE.findall(cells[flag_col]):
            out.setdefault(f, purpose)
    return out


def _structurally_documented_flags(text: str) -> dict[str, str]:
    """**结构化**登记过的开关 → 出处形状：围栏代码块里的命令，或「命令|开关|作用」表。

    散文里的 `` `--fail-on` `` **不进来** —— V-22 的原形就是"散文里留一句就永远满足"。
    """
    found = {f: "围栏代码块里的命令" for f in _FLAG_RE.findall("\n".join(_fenced_texts(text)))}
    for f, purpose in _table_documented_flags(text).items():
        found.setdefault(f, f"「命令|开关|作用」表：{purpose[:40]}")
    return found


def test_every_subcommand_flag_is_documented():
    """R-67/P5 + **V-22**：新子命令（`capabilities`/`selftest`/`diff`）的每个长开关都要
    **结构化登记**在 CHANGELOG 里；其余子命令的开关至少要出现在某份被跟踪文档的
    围栏/表格里。

    旧实现是 `flag not in changelog_text` —— 整文件子串匹配，于是把真说明删掉、只在散文里
    留一句 `` `--fail-on` `` 就永远满足（报告 V-22）。这里只认**解析出来的结构**：
    围栏代码块里的命令行，或「命令|开关|作用」表里"开关名 + 一句话说明"的那一行。
    """
    from kernoscope import cli

    root = Path(__file__).resolve().parents[1]
    ch_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    ch_flags = _structurally_documented_flags(ch_text)
    assert ch_flags, "CHANGELOG 里一个结构性登记都没解析出来 —— 解析逻辑退化了"

    tracked = subprocess.run(["git", "ls-files", "*.md", "*.txt"], cwd=root,
                             capture_output=True, text=True).stdout.split()
    if not tracked:
        # V-24 的同一原则：空输入集**失败**并说清出口，不静默通过、也不静默 skip
        msg = ("`git ls-files` 没列出任何文档 —— 这层门禁需要在 git 仓库里跑"
               "（`git archive` 导出的树拿不到清单）。若确实要在无 .git 的树里跑，"
               f"显式设置 {EMPTY_DOC_SCAN_ENV}=1 表示接受这次降级。")
        if os.environ.get(EMPTY_DOC_SCAN_ENV, "").strip() == "1":
            pytest.skip(msg)
        pytest.fail(msg)
    repo_flags: dict[str, str] = {}
    for d in tracked:
        if d.startswith("bench/artifacts/"):
            continue          # 入库产物是证据不是文档（与 test_documented_commands 同口径）
        p = root / d
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for flag, why in _structurally_documented_flags(text).items():
            repo_flags.setdefault(flag, f"{d}：{why}")

    missing: list[str] = []
    for sub in cli.SUBCOMMANDS:
        for a in cli.subcommand_parser(sub)._actions:
            for flag in a.option_strings:
                if not flag.startswith("--") or flag in PARSER_BUILTIN_FLAGS:
                    continue
                if sub in ("capabilities", "selftest", "diff"):
                    # 新子命令的开关是**接口**：必须写进 CHANGELOG（本文件的判据）
                    if flag not in ch_flags:
                        missing.append(f"{sub} {flag}（CHANGELOG 里没有结构化登记：既不在"
                                       f"围栏代码块，也不在「命令|开关|作用」表里）")
                elif flag not in repo_flags:
                    missing.append(f"{sub} {flag}（任何被跟踪文档的围栏/表格里都没有）")
    assert not missing, "这些子命令开关没有任何文档：\n  - " + "\n  - ".join(missing)


def test_the_changelog_flag_guard_is_structural():
    """V-22 的负样本：散文里提一句**不**算登记；围栏/表格里的才算。

    旧守卫（整文件子串）在这段负样本上会全绿 —— 这就是"自我满足"的形状。
    """
    from kernoscope import cli

    prose = "- 顺手记一句：`--fail-on` 是唯一能卡 CI 退出码的开关（这只是散文）。\n"
    assert "--fail-on" not in _structurally_documented_flags(prose)

    fenced = "```console\nkernoscope diff a.json b.json --fail-on regressed\n```\n"
    assert "--fail-on" in _structurally_documented_flags(fenced)

    table = ("| 命令 | 开关 | 作用 |\n|---|---|---|\n"
             "| `diff` | `--selfcheck <f>` / `--fail-on {none,regressed,change}` | 误差带对拍 |\n")
    assert "--fail-on" in _structurally_documented_flags(table)

    # 开关列有名字、说明列是空的 ⇒ 不构成"开关名 + 一句话说明"，不算登记
    silent = "| 命令 | 开关 | 作用 |\n|---|---|---|\n| `diff` | `--fail-on` |  |\n"
    assert "--fail-on" not in _structurally_documented_flags(silent)
    # 没有说明列的表也不算（避免"把开关塞进任意表格"就成了登记）
    no_purpose = "| 命令 | 开关 |\n|---|---|\n| `diff` | `--fail-on` |\n"
    assert "--fail-on" not in _structurally_documented_flags(no_purpose)

    # `--help` 被排除是有依据的：argparse 在每个子命令解析器上都自动加了它
    for sub in cli.SUBCOMMANDS:
        flags = {s for a in cli.subcommand_parser(sub)._actions for s in a.option_strings}
        assert "--help" in flags, f"{sub} 的解析器上没有 argparse 自动加的 --help？"
        assert "-h" in flags, sub


def test_skill_scripts_resolve_python_without_relying_on_path(tmp_path):
    """第五轮真机教训：容器里 `python3` 常常不在 PATH（只有绝对路径的解释器）。

    脚本写死 `python3` ⇒ 真机上 `setup-check.sh` 直接 127 退出（验收报告 §4.6 的那条失败）。
    现在 `_ks.sh` 提供 `ks_python`：`KS_PY` → `kernoscope` shebang（仅当它是 python）
    → PATH 上的 python3/python → 仓库 .venv → 兜底。
    """
    import subprocess

    ks = ROOT / "skills" / "kernoscope" / "scripts" / "_ks.sh"
    assert "ks_python()" in ks.read_text(encoding="utf-8"), "_ks.sh 必须提供 ks_python"
    setup = (ROOT / "skills" / "kernoscope" / "scripts" / "setup-check.sh").read_text(
        encoding="utf-8")
    assert '"$(ks_python)"' in setup, "setup-check.sh 不许把解释器写死成 python3"

    # KS_PY 显式给定时必须优先（这是脚本的可替换契约）
    r = subprocess.run(["bash", "-c", f'source "{ks}" && KS_PY=/bin/echo ks_python'],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "/bin/echo", (r.stdout, r.stderr)
