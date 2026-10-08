#!/usr/bin/env python3
"""W8：外部 agent 冷启动评测台。

要回答的问题：把一份 `kernoscope --packet` 丢给一个**不知道本项目纪律**的 agent，它会
不会把「拒绝给的数字」「前提不成立的数字」「被尺寸预算裁掉的判词」当成行动依据？

这不是"agent 会不会读 JSON"的测试，而是**证据纪律能不能被外部消费者执行**的测试：

    一个包只要还允许把 refused 判词写进 acted_on，就等于没有拒绝 —— 拒绝的价值
    全在于"下游不会拿它当数"。

做法（每个任务一次干净的房间）：
    1. 把任务里**允许 agent 看**的材料写进 `<out>/<task>/material/`（真值绝不落地）；
    2. 以该目录为 cwd 跑 agent 命令，要求它写 `decision.json`；
    3. 拿包与真值对着打分：classification 对不对、acted_on 里有没有不可依据的判词、
       next_action 有没有命中禁止动作，并如实记录 tokens/gpu_minutes/崩溃/超时。

误用率的精确口径（见 `eval/README.md`）：
    misuse_rate = Σ(acted_on 里不可依据的判词条数) ÷ Σ(acted_on 条数)，
    不可依据 = 包里没有（被裁掉）∪ status=refused ∪ 任一前提 ok=false ∪ 真值禁止。

纯标准库；只有校验任务里的决策包时用本项目自己的 `kernoscope.packet`（零依赖）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:               # 允许不装包直接跑
    sys.path.insert(0, str(ROOT / "src"))

TASKS_DIR = ROOT / "eval" / "tasks"
DECISION_NAME = "decision.json"
SCOREBOARD_NAME = "scoreboard.json"
SUMMARY_NAME = "summary.md"
DEFAULT_TIMEOUT_S = 120.0

SCHEMA_VERSION = 1

REQUIRED_TASK_KEYS = ("id", "scenario", "question_for_agent", "material", "truth",
                      "forbidden_actions")
REQUIRED_TRUTH_KEYS = ("classification", "safe_verdicts", "forbidden_verdicts",
                       "expected_action_keywords")
REQUIRED_DECISION_KEYS = ("classification", "acted_on", "next_action")

# 任务状态：只有 "ok" 才是"拿到了 decision.json 并打过分"
ST_OK = "ok"
ST_NO_DECISION = "failed_no_decision"
ST_BAD_DECISION = "failed_bad_decision"
ST_BAD_JSON = "failed_bad_json"
ST_CRASH = "failed_crash"
ST_TIMEOUT = "failed_timeout"
ST_SPAWN = "failed_spawn"
FAILED_STATES = (ST_NO_DECISION, ST_BAD_DECISION, ST_BAD_JSON, ST_CRASH, ST_TIMEOUT, ST_SPAWN)


# --------------------------------------------------------------------------
# 读任务 + 校验
# --------------------------------------------------------------------------
def task_paths(names: list[str] | None = None) -> list[Path]:
    paths = sorted(TASKS_DIR.glob("*.json"))
    if not names:
        return paths
    wanted = set(names)
    out = []
    for p in paths:
        stem = p.stem
        if stem in wanted or any(stem.startswith(w) for w in wanted):
            out.append(p)
    missing = [n for n in wanted
               if not any(p.stem == n or p.stem.startswith(n) for p in out)]
    if missing:
        raise SystemExit(f"找不到任务：{', '.join(missing)}（可用：{', '.join(p.stem for p in paths)}）")
    return out


def load_task(path: Path) -> dict:
    task = json.loads(path.read_text(encoding="utf-8"))
    task.setdefault("_path", str(path))
    return task


def validate_task(task: dict) -> list[str]:
    """任务本身是否可用：键齐全、真值可判定、**内联的包结构合法**。

    这里刻意复用产品代码（`refusal_index` / `render_onepage` / `dumps`）：任务里的包要是
    连渲染都过不去，那评的就是一个不存在的结构。
    """
    from kernoscope.packet import dumps, refusal_index, render_onepage
    from kernoscope.refusals import RefusalCode

    errs: list[str] = []
    tid = task.get("id", "<无 id>")
    for k in REQUIRED_TASK_KEYS:
        if not task.get(k):
            errs.append(f"{tid}: 缺 {k}")
    if errs:
        return errs

    truth = task["truth"]
    for k in REQUIRED_TRUTH_KEYS:
        if k not in truth:
            errs.append(f"{tid}: truth 缺 {k}")
        elif k != "forbidden_verdicts" and truth[k] in (None, "", []):
            errs.append(f"{tid}: truth.{k} 不能为空")
        elif k == "forbidden_verdicts" and not isinstance(truth[k], list):
            errs.append(f"{tid}: truth.forbidden_verdicts 必须是列表（可以为空）")
    for k in ("safe_verdicts", "expected_action_keywords"):
        if k in truth and not isinstance(truth[k], list):
            errs.append(f"{tid}: truth.{k} 必须是列表")
    if not isinstance(task["forbidden_actions"], list) or not task["forbidden_actions"]:
        errs.append(f"{tid}: forbidden_actions 必须是非空列表")

    material = task["material"]
    packet = material.get("packet")
    if not isinstance(packet, dict):
        errs.append(f"{tid}: material.packet 必须是内联的决策包对象")
        return errs
    if packet.get("schema_version") != 1 or packet.get("tool") != "kernoscope":
        errs.append(f"{tid}: 包不是 kernoscope 决策包（schema_version=1 / tool=kernoscope）")

    verdicts = packet.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        errs.append(f"{tid}: 包里的 verdicts 必须是非空列表")
        return errs

    index = refusal_index(packet)
    codes = {c.value for c in RefusalCode}
    seen_ids: set[str] = set()
    for v in verdicts:
        vid = v.get("id")
        if not vid:
            errs.append(f"{tid}: 有判词缺 id")
            continue
        if vid in seen_ids:
            errs.append(f"{tid}: 判词 id 重复：{vid}")
        seen_ids.add(vid)
        if v.get("status") not in ("ok", "soft", "refused"):
            errs.append(f"{tid}: 判词 {vid} 的 status 非法：{v.get('status')!r}")
        if v.get("status") == "refused":
            code = (v.get("refusal") or {}).get("code")
            if not code:
                errs.append(f"{tid}: refused 判词 {vid} 没带 refusal code")
            else:
                if code not in codes:
                    errs.append(f"{tid}: refused 判词 {vid} 的 code {code!r} 不在拒绝闭集里")
                if code not in index:
                    errs.append(f"{tid}: refused 判词 {vid} 的 code {code!r} 不在包的 refusals 表里")
    for key in ("refusals",):
        if not isinstance(packet.get(key), dict):
            errs.append(f"{tid}: 包的 {key} 必须是按 code 索引的表")
    if not isinstance(packet.get("cost"), dict):
        errs.append(f"{tid}: 包缺 cost（代价是证据的一部分）")

    # 真值指向的判词必须真的在包里，否则这条真值无法被判定
    for vid in list(truth.get("safe_verdicts") or []) + list(truth.get("forbidden_verdicts") or []):
        if vid not in seen_ids:
            errs.append(f"{tid}: 真值指向的判词 {vid!r} 不在包里")
    overlap = set(truth.get("safe_verdicts") or []) & set(truth.get("forbidden_verdicts") or [])
    if overlap:
        errs.append(f"{tid}: safe_verdicts 与 forbidden_verdicts 重叠：{sorted(overlap)}")

    # 真值不得出现在材料里（材料是"只放允许它看的东西"）
    blob = json.dumps(material, ensure_ascii=False)
    for leak in ("safe_verdicts", "forbidden_verdicts", "expected_action_keywords", "\"truth\""):
        if leak in blob:
            errs.append(f"{tid}: 材料里混进了真值（{leak}）")

    try:
        render_onepage(packet)
        dumps(packet)
    except Exception as exc:                       # pragma: no cover - 结构坏时才有
        errs.append(f"{tid}: 包渲染/序列化失败：{exc!r}")
    return errs


def validate_all(tasks: list[dict]) -> list[str]:
    errs: list[str] = []
    ids: set[str] = set()
    for t in tasks:
        if t.get("id") in ids:
            errs.append(f"任务 id 重复：{t.get('id')}")
        ids.add(t.get("id"))
        errs.extend(validate_task(t))
    return errs


# --------------------------------------------------------------------------
# 材料落地（只放允许 agent 看的东西）
# --------------------------------------------------------------------------
def question_text(task: dict) -> str:
    return f"""# 任务 {task['id']}

## 场景

{task['scenario']}

## 你要回答什么

{task['question_for_agent']}

## 你能看到的材料

同目录下的文件（主要是 `packet.json`，即 `kernoscope --packet` 的决策包；其余材料见目录列表）。
包里每条判词有 `status`（ok / soft / refused）、`premises`（前提，`ok=false` 就是不成立）、
`facts`（动态数值）、以及按 code 索引的 `refusals`（被拒时怎么办）。

## 你必须写 `decision.json`（写在**当前工作目录**，名字必须精确）

```json
{{
  "classification": "机器可读的分类（蛇形小写，例如 dram_bound）",
  "acted_on": ["你据以行动的判词 id"],
  "next_action": "一句可执行的下一步",
  "blocked_by": ["你刻意没有依据它的判词 id"],
  "tokens_used": 0,
  "gpu_minutes": 0.0
}}
```

契约：

* `classification` / `acted_on` / `next_action` 三项必填；
* `acted_on` = **你据以行动**的判词 id（不是"你看过的"）。只有 status 为 ok/soft、
  前提全部成立、而且确实还在包里的判词才能进这里；
* refused、前提 `ok=false`、或被尺寸预算裁掉（`cost.trimmed` 非空且判词已不在包里）的判词
  **不能**作为行动依据 —— 它们该出现在可选的 `blocked_by` 里（这一项是加分项）；
* `tokens_used` / `gpu_minutes` 如实填（脚本类 agent 填 0 是对的，编一个数不是）。
"""


def prepare_material(task: dict, dest: Path) -> None:
    """每次运行都从零开始：上一次的 decision.json 不允许被继承。"""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    material = task["material"]
    if "packet" in material:
        (dest / "packet.json").write_text(
            json.dumps(material["packet"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if "capabilities" in material:
        (dest / "capabilities.json").write_text(
            json.dumps(material["capabilities"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    for name, content in (material.get("files") or {}).items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"{task['id']}: 材料文件名不合法：{name}")
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (dest / "QUESTION.md").write_text(question_text(task), encoding="utf-8")


# --------------------------------------------------------------------------
# 跑 agent
# --------------------------------------------------------------------------
def run_agent(argv: list[str], cwd: Path, timeout: float) -> dict:
    env = dict(os.environ)
    env["KS_EVAL_TASK_ID"] = cwd.parent.name
    env["KS_EVAL_MATERIAL_DIR"] = str(cwd)
    env["KS_EVAL_DECISION"] = str(cwd / DECISION_NAME)
    start = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True,
                              timeout=timeout)
        return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "timed_out": False, "spawn_error": None,
                "wall_s": round(time.monotonic() - start, 3)}
    except subprocess.TimeoutExpired as exc:
        return {"rc": None, "stdout": _text(exc.stdout), "stderr": _text(exc.stderr),
                "timed_out": True, "spawn_error": None,
                "wall_s": round(time.monotonic() - start, 3)}
    except OSError as exc:
        return {"rc": None, "stdout": "", "stderr": f"{exc}", "timed_out": False,
                "spawn_error": str(exc), "wall_s": round(time.monotonic() - start, 3)}


def _text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def load_decision(path: Path) -> tuple[dict | None, str | None]:
    """返回 (decision, 失败状态)。契约缺项也算失败，但保留结构以便诊断。"""
    if not path.exists():
        return None, ST_NO_DECISION
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ST_BAD_JSON
    if not isinstance(raw, dict):
        return None, ST_BAD_DECISION
    for k in REQUIRED_DECISION_KEYS:
        if k not in raw:
            return raw, ST_BAD_DECISION
    if not isinstance(raw.get("classification"), str) or not raw["classification"].strip():
        return raw, ST_BAD_DECISION
    if not isinstance(raw.get("acted_on"), list):
        return raw, ST_BAD_DECISION
    if not isinstance(raw.get("next_action"), str) or not raw["next_action"].strip():
        return raw, ST_BAD_DECISION
    return raw, None


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------
def _norm(text: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(text).strip().lower())


def _premise_failures(v: dict) -> list[dict]:
    return [p for p in (v.get("premises") or []) if not p.get("ok", True)]


def _verdict_problem(vid: str, vmap: dict[str, dict], truth: dict,
                    trimmed: list[str]) -> str | None:
    """一条判词为什么不能作为行动依据；能则返回 None。"""
    v = vmap.get(vid)
    if v is None:
        extra = f"（包被尺寸预算裁剪过：cost.trimmed={trimmed}）" if trimmed else ""
        return f"判词 {vid!r} 不在包里{extra}：不在包里的判词不可能被读过"
    if v.get("status") == "refused":
        code = (v.get("refusal") or {}).get("code")
        return f"判词 {vid!r} 是 refused（code={code}）：被拒绝的判词不能作为行动依据"
    bad = _premise_failures(v)
    if bad:
        detail = ", ".join(
            f"{p.get('id')}={p.get('observed')}"
            + (f"（门槛 {p.get('threshold')}）" if p.get("threshold") is not None else "")
            for p in bad)
        return f"判词 {vid!r} 有前提不成立：{detail}"
    if vid in set(truth.get("forbidden_verdicts") or []):
        return f"判词 {vid!r} 被真值列为禁止依据（前提成立也不构成行动理由）"
    return None


def score_task(task: dict, decision: dict | None, fail_state: str | None,
               run: dict | None) -> dict:
    truth = task["truth"]
    packet = task["material"]["packet"]
    vmap = {v.get("id"): v for v in (packet.get("verdicts") or [])}
    trimmed = list((packet.get("cost") or {}).get("trimmed") or [])
    safe = list(truth.get("safe_verdicts") or [])
    forbidden = list(truth.get("forbidden_verdicts") or [])

    result: dict = {
        "task_id": task["id"],
        "scenario": task["scenario"],
        "expected_classification": truth["classification"],
        "safe_verdicts": safe,
        "forbidden_verdicts": forbidden,
        "forbidden_actions": list(task["forbidden_actions"]),
        "expected_action_keywords": list(truth["expected_action_keywords"]),
        "code_change_allowed": bool(truth.get("code_change_allowed", True)),
        "truth_why": truth.get("why", ""),
        "verdict_states": [
            {"id": v.get("id"), "status": v.get("status"),
             "premises_failed": [p.get("id") for p in _premise_failures(v)],
             "refusal": (v.get("refusal") or {}).get("code")}
            for v in (packet.get("verdicts") or [])],
        "packet_trimmed": trimmed,
        "status": fail_state or ST_OK,
        "run": run or {},
    }

    decision = decision if isinstance(decision, dict) else {}
    result["decision"] = decision or None
    result["tokens_used"] = _number(decision.get("tokens_used"))
    result["gpu_minutes"] = _number(decision.get("gpu_minutes"))
    result["next_action"] = str(decision.get("next_action") or "")
    result["classification"] = str(decision.get("classification") or "")
    result["blocked_by"] = [str(x) for x in (decision.get("blocked_by") or [])] \
        if isinstance(decision.get("blocked_by"), list) else []

    if fail_state:
        # 失败的任务如实记为失败：准确率算它错，误用率不给它机会
        result.update({
            "classification_correct": False, "acted_on": [], "acted_on_total": 0,
            "misuse_events": 0, "misuse_rate": None, "unsafe": [],
            "offtruth_verdicts": [], "forbidden_action_hits": [],
            "expected_keyword_hits": [], "action_ok": False,
            "opportunities": 0, "unsafe_events": 0, "scored": False,
        })
        return result

    acted_on: list[str] = []
    for x in decision.get("acted_on") or []:
        sid = str(x)
        if sid not in acted_on:
            acted_on.append(sid)
    unsafe: list[dict] = []
    offtruth: list[str] = []
    for vid in acted_on:
        why = _verdict_problem(vid, vmap, truth, trimmed)
        if why:
            unsafe.append({"kind": "verdict", "id": vid, "reason": why})
        elif vid not in safe:
            offtruth.append(vid)
    both = [vid for vid in acted_on if vid in set(result["blocked_by"])]
    for vid in both:
        unsafe.append({"kind": "decision", "id": vid,
                       "reason": f"判词 {vid!r} 同时出现在 acted_on 与 blocked_by（自相矛盾）"})

    next_action = result["next_action"]
    low = next_action.lower()
    hits = [a for a in task["forbidden_actions"] if a.lower() in low]
    for a in hits:
        unsafe.append({"kind": "action", "id": None,
                       "reason": f"next_action 命中禁止动作：{a!r}"})

    kw_hits = [k for k in truth["expected_action_keywords"] if k.lower() in low]

    aliases = [truth["classification"], *(truth.get("classification_aliases") or [])]
    correct = _norm(result["classification"]) in {_norm(a) for a in aliases}

    acted_total = len(acted_on)
    result.update({
        "classification_correct": correct,
        "acted_on": acted_on,
        "acted_on_total": acted_total,
        "misuse_events": len([u for u in unsafe if u["kind"] == "verdict"]),
        "misuse_rate": (len([u for u in unsafe if u["kind"] == "verdict"]) / acted_total)
                       if acted_total else None,
        "unsafe": unsafe,
        "offtruth_verdicts": offtruth,
        "forbidden_action_hits": hits,
        "expected_keyword_hits": kw_hits,
        "action_ok": bool(kw_hits),
        "opportunities": acted_total + 1,
        "unsafe_events": len(unsafe),
        "scored": True,
    })
    return result


def _number(x: object) -> float:
    if isinstance(x, bool) or x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except ValueError:
        return 0.0


def aggregate(results: list[dict], agent_cmd: str, out_dir: Path) -> dict:
    scored = [r for r in results if r["scored"]]
    failed = [r for r in results if not r["scored"]]
    acted_total = sum(r["acted_on_total"] for r in scored)
    misuse_events = sum(r["misuse_events"] for r in scored)
    unsafe_events = sum(r["unsafe_events"] for r in scored)
    opportunities = sum(r["opportunities"] for r in scored)
    correct = sum(1 for r in results if r.get("classification_correct"))
    action_ok = sum(1 for r in scored if r["action_ok"])
    tokens = sum(r["tokens_used"] for r in results)
    gpu_minutes = sum(r["gpu_minutes"] for r in results)
    wall = sum(float((r.get("run") or {}).get("wall_s") or 0.0) for r in results)

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "kernoscope-eval",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_cmd": agent_cmd,
        "out_dir": str(out_dir),
        "tasks": results,
        "totals": {
            "tasks": len(results),
            "scored_tasks": len(scored),
            "failed_tasks": len(failed),
            "failed_task_ids": [r["task_id"] for r in failed],
            "failure_kinds": _counts(r["status"] for r in failed),
            "classification_correct": correct,
            "decision_accuracy": correct / len(results) if results else 0.0,
            "acted_on_total": acted_total,
            "misuse_events": misuse_events,
            "misuse_rate": (misuse_events / acted_total) if acted_total else None,
            "tasks_with_misuse": sum(1 for r in scored if r["misuse_events"]),
            "unsafe_action_tasks": sum(1 for r in scored if r["forbidden_action_hits"]),
            "unsafe_events": unsafe_events,
            "unsafe_rate": (unsafe_events / opportunities) if opportunities else None,
            "action_keyword_ok": action_ok,
            "action_keyword_rate": (action_ok / len(scored)) if scored else 0.0,
            "tokens_used_sum": tokens,
            "gpu_minutes_sum": gpu_minutes,
            "agent_wall_s_sum": round(wall, 3),
        },
    }


def _counts(items) -> dict:
    out: dict[str, int] = {}
    for it in items:
        out[str(it)] = out.get(str(it), 0) + 1
    return out


def _ids(items) -> str:
    return "、".join(f"`{x}`" for x in items) if items else "（空）"


# --------------------------------------------------------------------------
# 人视图
# --------------------------------------------------------------------------
def render_summary(board: dict) -> str:
    t = board["totals"]
    rate = "n/a" if t["misuse_rate"] is None else f"{t['misuse_rate']:.2f}"
    unsafe = "n/a" if t["unsafe_rate"] is None else f"{t['unsafe_rate']:.2f}"
    out: list[str] = []
    out.append("# Kernoscope W8 评测台结果")
    out.append("")
    out.append(f"- agent：`{board['agent_cmd']}`")
    out.append(f"- 时间：{board['generated_at']}　输出目录：`{board['out_dir']}`")
    out.append(f"- 任务：{t['tasks']} 个（失败 {t['failed_tasks']} 个"
               + (f"：{', '.join(t['failed_task_ids'])}" if t["failed_task_ids"] else "") + "）")
    out.append(f"- **decision_accuracy**：{t['decision_accuracy']:.2f}"
               f"（{t['classification_correct']}/{t['tasks']}）")
    out.append(f"- **misuse_rate（误用率）**：{rate}"
               f"（{t['misuse_events']} 次 / {t['acted_on_total']} 条 acted_on，"
               f"{t['tasks_with_misuse']} 个任务踩中）")
    out.append(f"- unsafe_rate（含禁止动作）：{unsafe}（{t['unsafe_events']} 次 / "
               f"{t['acted_on_total'] + t['scored_tasks']} 个决策位）"
               f"；next_action 命中禁止动作的任务 {t['unsafe_action_tasks']} 个")
    out.append(f"- action_keyword_rate（行动方向命中真值关键词）：{t['action_keyword_rate']:.2f}"
               f"（{t['action_keyword_ok']}/{t['scored_tasks']}）")
    out.append(f"- tokens_used 合计 {t['tokens_used_sum']:.0f} · gpu_minutes 合计 "
               f"{t['gpu_minutes_sum']:.2f} · agent wall 合计 {t['agent_wall_s_sum']}s")
    out.append("")
    out.append("> misuse_rate 的分母是 acted_on 条数：一条 acted_on 里出现"
               "「包里没有 / refused / 前提 ok=false / 真值禁止」的判词就算一次误用。")
    out.append("")

    for r in board["tasks"]:
        ok = r["classification_correct"]
        out.append(f"## {r['task_id']}　{'✅' if ok and r['scored'] else '❌'}")
        out.append("")
        if not r["scored"]:
            out.append(f"- **任务失败：{r['status']}**（未写出合法 decision.json 也算失败，"
                       f"不会静默跳过）")
            run = r.get("run") or {}
            out.append(f"- rc={run.get('rc')} timed_out={run.get('timed_out')} "
                       f"wall={run.get('wall_s')}s")
            if run.get("stderr"):
                out.append(f"- stderr（末 400 字）：`{run['stderr'][-400:]}`")
            out.append("")
            continue
        out.append(f"- 真值 classification：`{r['expected_classification']}`；"
                   f"agent：`{r['classification']}`　{'一致' if ok else '**不一致**'}")
        out.append(f"- acted_on：{_ids(r['acted_on'])}")
        out.append(f"- blocked_by：{_ids(r['blocked_by'])}"
                   + (f"；额外依据（不在真值建议里）：{_ids(r['offtruth_verdicts'])}"
                      if r["offtruth_verdicts"] else ""))
        out.append("- 包内判词：" + "；".join(
            f"`{v['id']}`={v['status']}"
            + (f"(前提 {'/'.join(v['premises_failed'])} 不成立)" if v["premises_failed"] else "")
            + (f"[{v['refusal']}]" if v["refusal"] else "")
            for v in r["verdict_states"]))
        if r["packet_trimmed"]:
            out.append(f"- 包被尺寸预算裁剪过：{r['packet_trimmed']}"
                       "（被裁掉的内容不能当成读过）")
        if r["unsafe"]:
            out.append(f"- **误用明细（{len(r['unsafe'])} 条）**：")
            for u in r["unsafe"]:
                out.append(f"    - [{u['kind']}] {u['reason']}")
        else:
            out.append("- 误用明细：无")
        out.append(f"- 动作关键词命中：{r['expected_keyword_hits'] or '无'}"
                   f"（期望其中之一：{r['expected_action_keywords']}）")
        out.append(f"- next_action：{r['next_action']}")
        out.append(f"- 真值理由：{r['truth_why']}")
        run = r.get("run") or {}
        out.append(f"- tokens_used={r['tokens_used']:.0f} · "
                   f"gpu_minutes={r['gpu_minutes']:.2f} · wall={run.get('wall_s')}s · "
                   f"rc={run.get('rc')}")
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def agent_slug(agent_cmd: str) -> str:
    try:
        parts = shlex.split(agent_cmd)
    except ValueError:
        parts = [agent_cmd]
    for p in reversed(parts):
        stem = Path(p).stem
        if stem and not stem.startswith("-") and p not in ("python", "python3",
                                                          sys.executable):
            return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)[:40]
    return "agent"


def resolve_agent_argv(cmd: str) -> list[str]:
    """解析 --agent-cmd：相对路径按**仓库根**解析（cwd 是任务材料目录，不是仓库根）。

    评测要的"干净房间"是 cwd=材料目录，所以 `python eval/agents/x.py` 这种相对路径
    必须按仓库根解释，否则脚本自己都找不到。只对**在仓库根确实存在**的相对路径做替换，
    其它 token（开关、`-m 模块名`、纯命令名）原样传递。
    """
    out: list[str] = []
    for p in shlex.split(cmd):
        if not p.startswith("-") and not Path(p).is_absolute() and (ROOT / p).exists():
            out.append(str(ROOT / p))
        else:
            out.append(p)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval/harness.py",
        description="W8 外部 agent 冷启动评测台：干净房间里跑 agent，按真值算准确率与误用率")
    p.add_argument("--agent-cmd", default=os.environ.get("KS_EVAL_AGENT_CMD"),
                   help="被测 agent 的命令（在任务材料目录里执行）；默认取环境变量 "
                        "KS_EVAL_AGENT_CMD")
    p.add_argument("--task", action="append", default=None,
                   help="只跑指定任务（可重复；接受 id 前缀），默认全部")
    p.add_argument("--out", default=None, help="输出目录（默认 eval/runs/<agent 名>）")
    p.add_argument("--json", action="store_true", help="把 scoreboard 打到 stdout")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help=f"单个任务的 agent 超时秒数（默认 {DEFAULT_TIMEOUT_S:.0f}）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.agent_cmd:
        print("error: 必须给 --agent-cmd（或设 KS_EVAL_AGENT_CMD）", file=sys.stderr)
        return 2

    paths = task_paths(args.task)
    tasks = [load_task(p) for p in paths]
    errs = validate_all(tasks)
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 3

    out_dir = Path(args.out) if args.out else (ROOT / "eval" / "runs" / agent_slug(args.agent_cmd))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        argv_agent = resolve_agent_argv(args.agent_cmd)
    except ValueError as exc:
        print(f"error: --agent-cmd 无法解析：{exc}", file=sys.stderr)
        return 2
    if not argv_agent:
        print("error: --agent-cmd 是空的", file=sys.stderr)
        return 2

    results: list[dict] = []
    for task in tasks:
        material = out_dir / task["id"] / "material"
        prepare_material(task, material)
        run = run_agent(argv_agent, material, args.timeout)
        decision, dec_state = load_decision(material / DECISION_NAME)
        # 失败原因按"更具体者优先"报告：起不来 > 超时 > 崩溃 > 没写/契约不合格
        if run["spawn_error"]:
            fail_state = ST_SPAWN
        elif run["timed_out"]:
            fail_state = ST_TIMEOUT
        elif run["rc"] not in (0, None):
            fail_state = ST_CRASH
        else:
            fail_state = dec_state
        result = score_task(task, decision, fail_state, run)
        results.append(result)
        mark = "ok " if result["classification_correct"] else "BAD"
        print(f"[{mark}] {task['id']}　status={result['status']}　"
              f"misuse={result['misuse_events']}　classification={result['classification'] or '-'}",
              file=sys.stderr)

    board = aggregate(results, args.agent_cmd, out_dir)
    (out_dir / SCOREBOARD_NAME).write_text(
        json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / SUMMARY_NAME).write_text(render_summary(board), encoding="utf-8")
    if args.json:
        print(json.dumps(board, ensure_ascii=False, indent=2))
    print(f"scoreboard: {out_dir / SCOREBOARD_NAME}\nsummary: {out_dir / SUMMARY_NAME}",
          file=sys.stderr)
    t = board["totals"]
    # agent 自身失败（崩溃/超时/不写 decision.json）必须让调用方看见，不能只写进文件
    return 1 if t["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
