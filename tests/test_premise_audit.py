"""第七轮 X-22：前提检查必须是**逐判词**的，而且要落在 `agentrefs` 门禁里。

旧状况：`gen_agent_refs.py --check` 只做"生成物 == 生成器输出"，另有一条 pytest 检查前提，
但它只做**全局并集** —— 于是"给 `dram_bytes` 挂一整套环境前提"（验收方的变异 A4）在
`agentrefs` 门禁 rc=0、相关测试 116 条全绿：那些前提 id 当然都在标签表里（别的判词在用）。
现在 `tools/premise_audit.py` 逐 builder 展开（含它调用的 helper），要求
"这个判词能发出的前提 ⊆ 文档给它列的那套"，并挂进 `gen_agent_refs.py --check`。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from premise_audit import audit, audit_source  # noqa: E402

from kernoscope.packet import BUILDERS, VERDICT_PREMISES  # noqa: E402

PACKET_PY = ROOT / "src" / "kernoscope" / "packet.py"
BUILDER_NAMES = [b.__name__ for b in BUILDERS]


def test_the_real_source_passes_the_per_verdict_audit():
    assert audit() == []


def test_audit_goes_red_when_a_verdict_gets_undeclared_premises():
    """验收方变异 A4 的复刻：给 `_verdict_dram_total` 加一个 `_environment_premises(...)` 调用。

    静态审计看的是"这个 builder（含它调用的 helper）能发出什么"，所以只要**存在**这个调用，
    它经由 helper 发出的三个环境前提就都算在 `dram_bytes` 头上 —— 而文档没给它列这些。
    """
    src = PACKET_PY.read_text(encoding="utf-8")
    anchor = "def _verdict_dram_total(rep: Any) -> Verdict | None:\n"
    assert src.count(anchor) == 1
    mutated = src.replace(
        anchor,
        anchor + "    _environment_premises(rep)          # ← 变异 A4（静态审计不进仓库）\n",
        1)
    assert mutated != src

    problems = audit_source(mutated, VERDICT_PREMISES, BUILDER_NAMES)
    assert problems, "给判词挂上文档没列的前提，审计居然没红"
    joined = " ".join(problems)
    assert "dram_bytes" in joined and "env_gpu_exclusive" in joined, joined


def test_audit_reports_a_builder_whose_verdict_is_not_in_the_table():
    src = ("def _verdict_new(rep):\n"
           "    return Verdict(id='brand_new', premises=[Premise('some_premise')])\n")
    problems = audit_source(src, {}, ["_verdict_new"])
    assert problems and "brand_new" in problems[0], problems


def test_the_gate_actually_calls_the_audit():
    """`gen_agent_refs.py --check` 必须真的跑这条审计（不是只活在 pytest 里）。"""
    gen = (ROOT / "tools" / "gen_agent_refs.py").read_text(encoding="utf-8")
    assert "premise_audit" in gen and "audit()" in gen, gen[-800:]
