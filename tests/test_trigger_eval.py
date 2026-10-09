"""发现维度评测（trigger_rate）的契约测试。

W8 的评测台量"给了材料会不会误用"；这里量的是它**前面**那一步：agent 会不会在正确的
时刻选择 Kernoscope。这个指标的关键在于材料不能泄题、且评测用的触发词必须与
`SKILL.md` 的 frontmatter 同源（否则我们调的是文案，量的却是别的东西）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"


def _load():
    spec = importlib.util.spec_from_file_location("ks_trigger_eval", EVAL / "trigger_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    sys.modules["ks_trigger_eval"] = mod
    return mod


trigger = _load()


def test_cases_are_well_formed():
    paths = trigger.case_paths()
    assert len(paths) >= 4, "至少要有 4 个用例（含正负样本）"
    cases = [trigger.load_case(p) for p in paths]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        for key in ("id", "task", "catalog", "expect_any", "forbid_use", "why"):
            assert key in c, (c["id"], key)
        assert c["task"].strip()
        assert isinstance(c["catalog"], dict) and c["catalog"]
        assert "kernoscope" not in c["catalog"], \
            f"{c['id']}: 目录里不要手写 kernoscope（它的描述从 SKILL.md 现读）"
    pos = [c for c in cases if "kernoscope" in c["expect_any"]]
    neg = [c for c in cases if "kernoscope" in c["forbid_use"]]
    assert pos, "要有应当触发我们的用例"
    assert neg, "要有不应当触发我们的用例"


def test_material_does_not_leak_the_expectation(tmp_path):
    """干净房间：只给任务与目录，truth（expect_any/forbid_use/why）绝不许落地。"""
    case = trigger.load_case(trigger.case_paths()[0])
    dest = tmp_path / case["id"]
    trigger.prepare(case, dest)
    files = sorted(p.name for p in dest.iterdir())
    assert files == ["CATALOG.md", "TASK.md"], files
    blob = (dest / "CATALOG.md").read_text(encoding="utf-8") + \
        (dest / "TASK.md").read_text(encoding="utf-8")
    for banned in ("expect_any", "forbid_use", case["why"]):
        assert banned not in blob, banned


def test_catalog_description_comes_from_the_skill_file(tmp_path):
    """评测里的触发词必须与 SKILL.md 同源（调文案才等于调产品）。"""
    case = trigger.load_case(trigger.case_paths()[0])
    dest = tmp_path / case["id"]
    trigger.prepare(case, dest)
    catalog = (dest / "CATALOG.md").read_text(encoding="utf-8")
    desc = trigger.skill_description()
    assert desc[:40] in catalog
    assert "不要用它" in catalog, "目录里必须保留「什么时候不要用」这一句"


def test_scoring_semantics():
    case = {"id": "c", "expect_any": ["kernoscope"], "forbid_use": []}
    ok = trigger.score(case, {"use": ["kernoscope"], "first_command":
                              "kernoscope capabilities --brief"}, None)
    assert ok["expected_ok"] and ok["did_trigger"] and ok["entry_layer_ok"]
    bad = trigger.score(case, {"use": ["nsys"]}, None)
    assert not bad["expected_ok"] and not bad["did_trigger"]
    missing = trigger.score(case, None, "没有写 choice.json")
    assert not missing["expected_ok"] and missing["status"] == "没有写 choice.json"

    neg = {"id": "n", "expect_any": ["ncu"], "forbid_use": ["kernoscope"]}
    good = trigger.score(neg, {"use": ["ncu"]}, None)
    assert good["expected_ok"] and not good["false_positive"]
    fp = trigger.score(neg, {"use": ["kernoscope"]}, None)
    assert not fp["expected_ok"] and fp["false_positive"]

    none_needed = {"id": "z", "expect_any": [], "forbid_use": ["kernoscope"]}
    assert trigger.score(none_needed, {"use": []}, None)["expected_ok"]


def test_invented_command_is_detected():
    """只给触发词时 agent 可能编命令（真事：出现过不存在的 `kernoscope compare`）。"""
    case = {"id": "c", "expect_any": ["kernoscope"], "forbid_use": []}
    good = trigger.score(case, {"use": ["kernoscope"],
                                "first_command": "kernoscope capabilities --brief"}, None)
    assert good["entry_layer_ok"] and not good["made_up_command"]
    good2 = trigger.score(case, {"use": ["kernoscope"],
                                 "first_command": "kernoscope --packet - -- python t.py"}, None)
    assert good2["entry_layer_ok"] and not good2["made_up_command"]
    bad = trigger.score(case, {"use": ["kernoscope"],
                               "first_command": "kernoscope compare --baseline a b"}, None)
    assert bad["made_up_command"] and not bad["entry_layer_ok"]
    weird = trigger.score(case, {"use": ["kernoscope"], "first_command": "ksc doctor"}, None)
    assert not weird["made_up_command"] and not weird["entry_layer_ok"]


def test_summary_computes_rates():
    results = [
        trigger.score({"id": "p1", "expect_any": ["kernoscope"], "forbid_use": []},
                      {"use": ["kernoscope"]}, None),
        trigger.score({"id": "p2", "expect_any": ["kernoscope"], "forbid_use": []},
                      {"use": ["nsys"]}, None),
        trigger.score({"id": "n1", "expect_any": ["ncu"], "forbid_use": ["kernoscope"]},
                      {"use": ["kernoscope"]}, None),
        trigger.score({"id": "n2", "expect_any": ["py-spy"], "forbid_use": ["kernoscope"]},
                      {"use": ["py-spy"]}, None),
    ]
    b = trigger.summarize(results, Path("/tmp/x"))
    assert b["trigger_rate"] == 0.5
    assert b["false_positive_rate"] == 0.5
    assert b["expected_ok_rate"] == 0.5


def test_prepare_only_cli_writes_material(capsys, tmp_path):
    rc = trigger.main(["--prepare-only", "--out", str(tmp_path)])
    assert rc in (0, 1)          # 没有 choice.json 时返回 1，但材料必须落地
    cases = list((tmp_path).glob("*/TASK.md"))
    assert len(cases) == len(trigger.case_paths())
    assert (tmp_path / "trigger-board.json").exists()
    assert (tmp_path / "trigger-summary.md").exists()
    capsys.readouterr()
