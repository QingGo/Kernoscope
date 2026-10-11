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


def test_every_subcommand_flag_is_documented():
    """R-67/P5：`capabilities`/`selftest`/`diff` 的每个长开关都必须写进 CHANGELOG。

    本文件开头自己定的判据是"外部消费者要改代码或改解析逻辑，就必须写在这里" ——
    而 `diff --fail-on`（唯一能卡 CI 退出码的开关）在包里出现 0 次、任何文档里 0 次。
    """
    from kernoscope import cli

    import subprocess

    root = Path(__file__).resolve().parents[1]
    ch = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    docs = subprocess.run(["git", "ls-files", "*.md", "*.txt"], cwd=root,
                          capture_output=True, text=True).stdout.split()
    all_text = "\n".join((root / d).read_text(encoding="utf-8", errors="replace")
                         for d in docs)
    missing: list[str] = []
    for sub in cli.SUBCOMMANDS:
        for a in cli.subcommand_parser(sub)._actions:
            for flag in a.option_strings:
                if not flag.startswith("--"):
                    continue
                if sub in ("capabilities", "selftest", "diff"):
                    # 新子命令的开关是**接口**：必须写进 CHANGELOG（本文件的判据）
                    if flag not in ch:
                        missing.append(f"{sub} {flag}（不在 CHANGELOG）")
                elif flag not in all_text:
                    missing.append(f"{sub} {flag}（任何 tracked 文档里都没有）")
    assert not missing, "这些子命令开关没有任何文档：\n  - " + "\n  - ".join(missing)
