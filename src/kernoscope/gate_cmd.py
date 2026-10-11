"""`kernoscope gate` —— 把决策包变成**便宜、可编程的 CI 门禁**（C4）。

为什么需要它：`--packet` 已经给出判词、前提、结构化拒绝与代价，但把它接进 CI 还得让每个
团队自己写一段"读 JSON、判断词、比对指标"的胶水 —— 而那段胶水最容易出两种错：
**把 refused 当成 ok**（泄露一个不成立的结论）与**用绝对阈值卡一个带误差带的量**
（把噪声当回归）。所以门禁本身做成子命令，规则是确定的、退出码是明确的：

    0  全部通过
    1  有规则不通过（逐条打印原因；`--json` 是给 agent 的机器视图）
    2  用法错误（文件读不了、规则写错）

四条判据（都按**可编程的东西**判，不按散文）：

  `--require <id>[=<status>]`   判词必须存在，且状态至少是 `status`（默认 `ok`；
                                 顺序 ok > soft > refused）
  `--require-actionable <id>`   判词必须 `actionable=true`（v1.0 包没有这个字段时，
                                 回退到"不是 refused 且列出的前提全部成立"）
  `--forbid-code <code>`        出现该拒绝码即失败（含判词的 refusal 与 refusals 索引）
  `--max-metric <name>=<v>` /   报告里的指标必须 ≤ / ≥ 阈值（**只有存在且可引用的指标**
  `--min-metric <name>=<v>`     才会被比较；unavailable 的指标一律算"不满足"，
                                 因为"没数"不能当作"通过了"）

**不读也不跑测量**：门禁只消费已经产出的包与报告。要新数据就先跑一次采集。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 状态强弱顺序：`--require x=soft` 意思是"至少 soft"，因此 ok 也算过。
_STATUS_RANK = {"ok": 2, "soft": 1, "refused": 0}


@dataclass
class Failure:
    check: str
    reason: str
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = {"check": self.check, "reason": self.reason}
        if self.observed is not None:
            d["observed"] = self.observed
        return d


@dataclass
class GateResult:
    passed: bool = True
    failures: list[Failure] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    packet_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "packet": self.packet_path,
                "checked": self.checked, "failures": [f.to_dict() for f in self.failures]}


def _verdicts(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return list(packet.get("verdicts") or [])


def _find_verdict(packet: dict[str, Any], vid: str) -> dict[str, Any] | None:
    return next((v for v in _verdicts(packet) if v.get("id") == vid), None)


def _refusal_codes(packet: dict[str, Any]) -> set[str]:
    out = {str((v.get("refusal") or {}).get("code")) for v in _verdicts(packet)
           if v.get("refusal")}
    out |= {str(k) for k in (packet.get("refusals") or {}).keys()}
    for d in packet.get("environment_refusals") or []:
        if d.get("code"):
            out.add(str(d["code"]))
    out.discard("None")
    return out


def _metric_lookup(report: dict[str, Any]) -> dict[str, Any]:
    """报告 JSON 里的指标名 → metric 对象（跨 kernel/section 全收）。"""
    out: dict[str, Any] = {}
    for k in report.get("kernels") or []:
        for s in k.get("sections") or []:
            for m in s.get("metrics") or []:
                out.setdefault(str(m.get("name")), m)
    return out


def _is_actionable(v: dict[str, Any]) -> bool:
    """v1.1 起包里有 `actionable`；更早的包按"不是 refused 且前提全成立"回退。"""
    if "actionable" in v:
        return bool(v["actionable"])
    if v.get("status") == "refused":
        return False
    return all(p.get("ok", True) for p in v.get("premises") or [])


def _parse_assign(text: str) -> tuple[str, float]:
    name, _, raw = text.partition("=")
    if not name or not raw:
        raise ValueError(f"规则应形如 NAME=VALUE，收到 {text!r}")
    return name.strip(), float(raw)


def run_gate(packet: dict[str, Any], *, report: dict[str, Any] | None = None,
             require: list[str] | None = None, require_actionable: list[str] | None = None,
             forbid_codes: list[str] | None = None, max_metric: list[str] | None = None,
             min_metric: list[str] | None = None, packet_path: str = "") -> GateResult:
    """纯函数：跑一遍门禁规则。便于测试，也让「门禁判什么」只有一处实现。"""
    res = GateResult(packet_path=packet_path)
    metrics = _metric_lookup(report or {})

    for spec in require or []:
        vid, _, want = spec.partition("=")
        want = (want or "ok").strip()
        if want not in _STATUS_RANK:
            res.failures.append(Failure(f"require {vid}",
                                        f"状态名非法：{want}（可选 ok/soft/refused）"))
            res.passed = False
            continue
        v = _find_verdict(packet, vid.strip())
        res.checked.append(f"require {vid.strip()}")
        if v is None:
            res.failures.append(Failure(f"require {vid.strip()}",
                                        "包里没有这条判词（可能被尺寸预算裁掉了 —— 见 "
                                        "cost.trimmed / verdicts_more）"))
            res.passed = False
            continue
        got = str(v.get("status") or "")
        if _STATUS_RANK.get(got, -1) < _STATUS_RANK[want]:
            res.failures.append(Failure(
                f"require {vid.strip()}", f"状态 {got} 弱于要求的 {want}",
                {"status": got, "refusal": (v.get("refusal") or {}).get("code")}))
            res.passed = False

    for vid in require_actionable or []:
        res.checked.append(f"require-actionable {vid}")
        v = _find_verdict(packet, vid)
        if v is None:
            res.failures.append(Failure(f"require-actionable {vid}", "包里没有这条判词"))
            res.passed = False
        elif not _is_actionable(v):
            res.failures.append(Failure(
                f"require-actionable {vid}", "判词不可作为依据（false/前提不成立/被拒）",
                {"status": v.get("status"),
                 "failed_premises": [p.get("id") for p in v.get("premises") or []
                                     if p.get("ok") is False]}))
            res.passed = False

    codes = _refusal_codes(packet)
    for code in forbid_codes or []:
        res.checked.append(f"forbid-code {code}")
        if code in codes:
            res.failures.append(Failure(f"forbid-code {code}", "出现了被禁止的拒绝码",
                                        {"all_codes": sorted(codes)}))
            res.passed = False

    for spec, op, better in [(s, "max", "≤") for s in (max_metric or [])] + \
                            [(s, "min", "≥") for s in (min_metric or [])]:
        try:
            name, threshold = _parse_assign(spec)
        except ValueError as exc:
            res.failures.append(Failure(f"{op}-metric {spec}", str(exc)))
            res.passed = False
            continue
        res.checked.append(f"{op}-metric {name}")
        m = metrics.get(name)
        if m is None:
            res.failures.append(Failure(f"{op}-metric {name}", "报告里没有这个指标"))
            res.passed = False
            continue
        if m.get("value") is None or m.get("provenance") == "unavailable":
            res.failures.append(Failure(
                f"{op}-metric {name}", "指标不可用（n/a）：没有数不等于通过",
                {"refusal": (m.get("refusal") or {}).get("code")}))
            res.passed = False
            continue
        value = float(m["value"])
        ok = value <= threshold if op == "max" else value >= threshold
        if not ok:
            res.failures.append(Failure(
                f"{op}-metric {name}", f"实测 {value:g} 不满足 {better} {threshold:g}",
                {"value": value, "threshold": threshold, "provenance": m.get("provenance")}))
            res.passed = False

    if not res.checked:
        res.failures.append(Failure("rules", "一条规则都没给：门禁不会凭空通过（加 --require/"
                                             "--forbid-code/--max-metric）"))
        res.passed = False
    return res


def render_text(res: GateResult) -> str:
    lines = []
    if res.passed:
        lines.append(f"门禁通过（检查 {len(res.checked)} 条：{', '.join(res.checked[:6])}"
                     + ("…" if len(res.checked) > 6 else "") + "）")
        return "\n".join(lines)
    lines.append(f"门禁不通过（{len(res.failures)}/{len(res.checked)} 条不合格）")
    for f in res.failures:
        lines.append(f"  [失败] {f.check}：{f.reason}")
        if f.observed is not None:
            lines.append(f"         观测：{json.dumps(f.observed, ensure_ascii=False)}")
    lines.append("")
    lines.append("提示：门禁判的是**已经产出的**包与报告 —— 要新数据先跑一次采集，"
                 "再对新的包重跑门禁。")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kernoscope gate",
        description="把决策包变成便宜、可编程的 CI 门禁：判词状态 / 可依据性 / 禁止的拒绝码 / "
                    "指标阈值。只读已产出的包与报告，不跑测量。")
    ap.add_argument("--packet", required=True, help="决策包 JSON（kernoscope --packet 产出）")
    ap.add_argument("--report", default=None, help="完整报告 JSON（--metrics 阈值需要它）")
    ap.add_argument("--require", action="append", default=[], metavar="ID[=STATUS]",
                    help="判词必须存在且状态至少 STATUS（默认 ok；可选 soft/refused）")
    ap.add_argument("--require-actionable", action="append", default=[], metavar="ID",
                    help="判词必须可作为行动依据（actionable=true / 前提全部成立）")
    ap.add_argument("--forbid-code", action="append", default=[], metavar="CODE",
                    help="出现该拒绝码即失败（可重复）")
    ap.add_argument("--max-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="报告里的指标必须 ≤ 阈值（不可用的指标算不通过）")
    ap.add_argument("--min-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="报告里的指标必须 ≥ 阈值")
    ap.add_argument("--json", action="store_true", help="机器可读（agent 用这个）")
    return ap


def gate_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        packet = json.loads(Path(args.packet).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: 决策包读不了：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report = None
    if args.report:
        try:
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: 报告读不了：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    res = run_gate(packet, report=report, require=list(args.require),
                   require_actionable=list(args.require_actionable),
                   forbid_codes=list(args.forbid_code),
                   max_metric=list(args.max_metric), min_metric=list(args.min_metric),
                   packet_path=str(args.packet))
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text(res))
    return 0 if res.passed else 1
