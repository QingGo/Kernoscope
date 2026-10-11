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
                                 顺序 ok > soft > refused）。`status` 是闭集，写错即用法错误
  `--require-actionable <id>`   判词必须 `actionable=true`（v1.0 包没有这个字段时，
                                 回退到"不是 refused 且列出的前提全部成立"）
  `--forbid-code <code>`        出现该拒绝码即失败（**包里任何位置**：判词的 refusal、
                                 refusals 索引与 refusals_omitted / unmeasured /
                                 environment_refusals(_more) / next / env.blocked_by …，
                                 见 `_refusal_codes`）。
                                 `code` 必须是 `refusals.py` 闭集成员：拼错的 code 是
                                 **用法错误（rc=2）**，绝不静默放绿
  `--max-metric <name>=<v>` /   指标阈值，**逐 kernel** 判（任一个违反即失败）：只要至少一个真
  `--min-metric <name>=<v>`     kernel 带这个指标，它就是 kernel 级判据 —— 每个真 kernel 都必须
                                 带、可引用、且达标（缺项 / n/a / 超限都算不合格，并在输出里点名
                                 是哪个 kernel）。只有 `(device)` 带的指标（如 `ks__cost_per_mtok`）
                                 按设备级判 `(device)`。**没有数不能当作"通过了"**

判据是"任一个违反即失败"，不是"任一个通过即通过"：门禁被拿去卡 CI，松一格就是绿灯骗人。


**不读也不跑测量**：门禁只消费已经产出的包与报告。要新数据就先跑一次采集。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .refusals import RefusalCode
from .report_md import DEVICE_KERNEL

# 状态强弱顺序：`--require x=soft` 意思是"至少 soft"，因此 ok 也算过。
_STATUS_RANK = {"ok": 2, "soft": 1, "refused": 0}

# `--forbid-code` 的合法取值 = refusals.py 的 code 闭集（`.value` 是包里出现的字符串）。
# 拼错的 code 永远不会出现在任何包里 —— 如果当成"没命中"，`--forbid-code counters_denyed`
# 就会放绿，把一个拼错的规则伪装成"规则通过"。所以未知 code 是用法错误（rc=2）。
_VALID_CODES: frozenset[str] = frozenset(c.value for c in RefusalCode)

# 包里"一串拒绝码"用的键名（`unmeasured` 是 `{code,count}`，不是这里）。
# packet.py 的 `_cited_codes()` 用的也是这两个名字。
_CODE_LIST_KEYS: frozenset[str] = frozenset({"codes", "blocked_by"})


def valid_codes() -> list[str]:
    """闭集里全部合法 code（排序，供用法错误的引导文本使用）。"""
    return sorted(_VALID_CODES)


def _codes_hint() -> str:
    return "合法 code（refusals.py 闭集）：" + "、".join(valid_codes())


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
    # 规则本身写错（未知 code / 未知 status）→ CLI 按"用法错误"退出（rc=2），
    # 而不是像"规则不满足"那样退 1：这两种失败对 CI 的含义不同。
    usage_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "packet": self.packet_path,
                "checked": self.checked, "usage_errors": self.usage_errors,
                "failures": [f.to_dict() for f in self.failures]}


def _verdicts(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return list(packet.get("verdicts") or [])


def _find_verdict(packet: dict[str, Any], vid: str) -> dict[str, Any] | None:
    return next((v for v in _verdicts(packet) if v.get("id") == vid), None)


def _omitted_codes_truncated(packet: dict[str, Any]) -> tuple[bool, int]:
    """包里"丢掉的拒绝码"列表有没有被自己的上限截断，以及截掉了多少。

    为什么这决定了门禁能不能判通过（第七轮 X-16）：`refusals_omitted.codes` 最多列 4 个
    （再多只给 `count`），受限期机器上十几个 code 时它是**常态**。于是"我在可见的那部分里
    没找到 → 通过"这条推理是**错的**：13 个真实 code 查不到就被放绿了。截断时唯一诚实的结论是
    "无法判定"，而不是"通过"。
    """
    omitted = packet.get("refusals_omitted") or {}
    env_more = packet.get("environment_refusals_more") or {}
    truncated = bool(omitted.get("codes_truncated")) or bool(env_more.get("codes_truncated"))
    hidden = max(0, int(omitted.get("count") or 0) - len(omitted.get("codes") or []))
    hidden += max(0, int(env_more.get("count") or 0) - len(env_more.get("codes") or []))
    return truncated, hidden


def _refusal_codes(packet: dict[str, Any]) -> set[str]:
    """包里**所有**位置出现过的拒绝码 —— `--forbid-code` 只认这个集合。

    R-98/W-3：旧实现只认三处（`verdicts[].refusal.code` / `refusals{}` 的键 /
    `environment_refusals[].code`），于是 code 被预算裁进 `refusals_omitted.codes`、
    `unmeasured[].code` 或 `environment_refusals_more.codes` 时，拼对的 `--forbid-code`
    也 **rc=0 放绿** —— 而"code 被挪进 *_omitted/*_more"正是受限机（十几个 code）的常态，
    拿 gate 卡 CI 就会被骗。

    所以不再逐个枚举今天的载体，而是**扫全树**：每个 `code` 键、每个 code 列表
    （`codes` / `blocked_by`），外加 `refusals` 表的键（那是 code 做键的地方）。将来
    packet.py 再添一个携带 code 的字段，只要它沿用这两个名字就自动跟上；用新名字
    （例如 `foo_codes`）就要在 `_CODE_LIST_KEYS` 里加一条 —— 这是刻意的：宁可漏掉
    一个全新的形状（测试里逐个载体点名守着），也不把任意字符串列表都当成拒绝码。
    非 code 的附带值（`for`/`action`/remedy 文本、环境消息里的散文）即使被扫到，
    也不在 `RefusalCode` 闭集里，不会被误判命中。
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                # `code` 是单个 code；`codes` / `blocked_by` 是 code 列表
                # （refusals_omitted.codes / *_more.codes / env.blocked_by，
                #  packet.py 的 `_cited_codes()` 也是这两个名字）。别的字段一律往下钻。
                if k == "code" and isinstance(v, str):
                    out.add(v)
                elif k in _CODE_LIST_KEYS and isinstance(v, list):
                    # 列表元素通常是字符串；万一是 `{code: ...}` 形状也一并钻下去。
                    out.update(x for x in v if isinstance(x, str))
                    walk([x for x in v if not isinstance(x, str)])
                else:
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(packet)
    out |= {str(k) for k in (packet.get("refusals") or {})}
    return out


def _kernel_metrics(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """报告 JSON → [(kernel 名, {指标名: metric 对象}), ...]，**按 kernel 分开**。

    V-2（验收报告第五轮）：旧实现用 `setdefault` 把跨 kernel 的同名指标压成一个映射，
    于是 `--max-metric`/`--min-metric` 只看**第一个** kernel。报告里只要有第二个 kernel
    超限，门禁照样放绿 —— 对"拿去卡 CI"的子命令来说这是最坏的失败模式。指标规则因此
    逐个 kernel 评估，失败的 Failure 里带 `kernel` 字段，输出点名是哪一个。
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for k in report.get("kernels") or []:
        metrics: dict[str, Any] = {}
        for s in k.get("sections") or []:
            for m in s.get("metrics") or []:
                metrics.setdefault(str(m.get("name")), m)
        out.append((str(k.get("name") or "?"), metrics))
    return out


def _is_actionable(v: dict[str, Any]) -> bool:
    """v1.1 起包里有 `actionable`；更早的包按"不是 refused 且前提全成立"回退。

    R-103/W-7：显式字段**只接受真布尔** —— 旧实现 `bool(v["actionable"])` 会把字符串
    `"false"`（JSON 手写包的常见形态）强转成 True，`--require-actionable` 于是 rc=0 放绿。
    类型不对是**规则没法判**（用法错误 rc=2），不是"判词不合格"；调用方负责报这个 usage
    error（`_actionable_type_error`），不会静默把它当通过。
    """
    if "actionable" in v:
        return v["actionable"] is True
    if v.get("status") == "refused":
        return False
    return all(p.get("ok", True) for p in v.get("premises") or [])


def _actionable_type_error(v: dict[str, Any]) -> str | None:
    """`actionable` 不是真布尔就返回一句用法错误（rc=2），否则 None。

    `bool("false") is True`：字符串冒充布尔是"规则没法判"，绝不能退化成判词通过。
    """
    if "actionable" not in v:
        return None
    got = v["actionable"]
    if isinstance(got, bool):
        return None
    return (f"判词 `{v.get('id')}` 的 `actionable` 必须是布尔值，收到 "
            f"{type(got).__name__} {got!r}（字符串 `\"false\"` 不是 false —— "
            f"不接受会强转的写法，否则门禁会放绿）")


def _parse_assign(text: str) -> tuple[str, float]:
    name, _, raw = text.partition("=")
    if not name or not raw:
        raise ValueError(f"规则应形如 NAME=VALUE，收到 {text!r}")
    return name.strip(), float(raw)


def _check_metric_rule(res: GateResult, *, op: str, better: str, name: str,
                       threshold: float, report: dict[str, Any]) -> None:
    """一条 max/min 指标规则：对报告里的 kernel **逐个**判定（V-2 的修复）。

    语义（**任一个违反即失败**，绝不是"任一个通过即通过"）：

      * **kernel 级判据**：只要报告里**至少一个真 kernel** 带了这个指标，它就是 kernel 级的
        —— 那么**每一个**真 kernel 都必须带它、可引用、且达标。缺项、n/a、超限都算不合格：
        "没数"不是"通过"（沿用旧纪律）。这样第二个 kernel 超限不会再被第一个盖住。
      * **设备级判据**：如果只有 `(device)` 伪 kernel 带这个指标（例如 `ks__cost_per_mtok`
        与 NVML 的设备读数），它就是设备级判据，只判 `(device)` —— 真 kernel 天然不带它，
        不能因此把规则判红（否则 README 里的例子会无脑变红）。
      * 报告里**任何 kernel 都没有**这个名字 → 不合格（同旧的"报告里没有这个指标"）。
      * `(device)` 带了 kernel 级指标时也一起比（多一个点名的地方，不是豁免）。

    每条失败都带 `kernel` 字段并在 reason 里点名，用户才知道该去改哪个 kernel。
    """
    check = f"{op}-metric {name}"
    kernels = _kernel_metrics(report)
    # 有真 kernel 带这个指标 → kernel 级判据：每个真 kernel 都必须带。
    kernel_scoped = any(name in km for kname, km in kernels if kname != DEVICE_KERNEL)
    present = False
    for kname, kmetrics in kernels:
        m = kmetrics.get(name)
        if m is None:
            if kname == DEVICE_KERNEL or not kernel_scoped:
                continue
            res.failures.append(Failure(
                check, f"kernel `{kname}` 没有这个指标：没有数不能当作通过",
                {"kernel": kname}))
            res.passed = False
            continue
        present = True
        if m.get("value") is None or m.get("provenance") == "unavailable":
            res.failures.append(Failure(
                check, f"kernel `{kname}` 的指标不可用（n/a）：没有数不能当作通过",
                {"kernel": kname, "refusal": (m.get("refusal") or {}).get("code")}))
            res.passed = False
            continue
        raw_value = m["value"]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            # X-13（第七轮）：数值阈值打到**字符串取值**的指标上（`est__bound="memory"`、
            # `ks__occupancy_limiter="registers"`）曾经是裸 `ValueError` traceback。
            # 语义上这是**用法错误**（规则对这类指标无定义），与 `--max-metric X=abc` 同类 ⇒ rc=2；
            # 同时记一条 failure，让 `--json` 里也能看到是哪一条指标、值是什么。
            msg = (f"规则对指标 `{name}` 无定义：`{kname}` 的值是字符串 {raw_value!r}，"
                   f"数值阈值只适用于数值型指标")
            res.usage_errors.append(f"{op}-metric {name}: {msg}")
            res.failures.append(Failure(check, msg, {"kernel": kname, "value": raw_value}))
            res.passed = False
            continue
        ok = value <= threshold if op == "max" else value >= threshold
        if not ok:
            res.failures.append(Failure(
                check, f"kernel `{kname}` 实测 {value:g} 不满足 {better} {threshold:g}",
                {"kernel": kname, "value": value, "threshold": threshold,
                 "provenance": m.get("provenance")}))
            res.passed = False
    if not present:
        res.failures.append(Failure(check, "报告里没有这个指标"))
        res.passed = False


def run_gate(packet: dict[str, Any], *, report: dict[str, Any] | None = None,
             require: list[str] | None = None, require_actionable: list[str] | None = None,
             forbid_codes: list[str] | None = None, max_metric: list[str] | None = None,
             min_metric: list[str] | None = None, packet_path: str = "") -> GateResult:
    """纯函数：跑一遍门禁规则。便于测试，也让「门禁判什么」只有一处实现。"""
    res = GateResult(packet_path=packet_path)

    for spec in require or []:
        vid, _, want = spec.partition("=")
        want = (want or "ok").strip()
        if want not in _STATUS_RANK:
            # status 是闭集：写错是**规则本身**写错（用法错误），不是"规则不满足"。
            msg = (f"状态名 {want!r} 非法：status 是闭集，可选 "
                   f"{'/'.join(_STATUS_RANK)}")
            res.usage_errors.append(f"require {vid.strip()}: {msg}")
            res.failures.append(Failure(f"require {vid}", msg))
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
            continue
        # R-103：先判类型（用法错误），再判值。顺序不能反 —— 字符串 `"false"` 若先走
        # `_is_actionable` 就会因为"不是真布尔"而报成"判词不可作为依据"（rc=1），
        # 掩盖了其实是包/规则写错（rc=2）。
        type_err = _actionable_type_error(v)
        if type_err is not None:
            res.usage_errors.append(f"require-actionable {vid}: {type_err}")
            res.failures.append(Failure(f"require-actionable {vid}", type_err))
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
        if code not in _VALID_CODES:
            # 未知 code 永远不会命中任何包：当成"没命中"就会静默放绿一个拼错的规则。
            msg = (f"{code!r} 不是 refusals.py 闭集里的拒绝码（用法错误：拼错的 code "
                   f"永远不会命中，静默放绿等于骗人）。{_codes_hint()}")
            res.usage_errors.append(f"forbid-code {code}: {msg}")
            res.failures.append(Failure(f"forbid-code {code}", msg))
            res.passed = False
            continue
        if code not in codes and _omitted_codes_truncated(packet)[0]:
            # 找不到 + 列表被截断 ⇒ **不能判通过**（X-16）。报文要把"为什么判不了"说清，
            # 并给出最便宜的补法（把预算放宽或直接看完整报告）。
            truncated, hidden = _omitted_codes_truncated(packet)
            msg = (f"无法判定：包里丢掉的拒绝码列表被截断（还有约 {hidden} 个 code 未列出），"
                   f"所以 `{code}` 没找到**不能**当作没出现。"
                   f"补法：放宽决策包预算重跑采集，或直接对完整报告 `-o rep` 里的 refusals 判")
            res.failures.append(Failure(f"forbid-code {code}", msg,
                                        {"codes_truncated": truncated, "hidden_codes": hidden}))
            res.passed = False
            continue
        if code in codes:
            res.failures.append(Failure(f"forbid-code {code}", "出现了被禁止的拒绝码",
                                        {"all_codes": sorted(codes)}))
            res.passed = False

    for spec, op, better in [(s, "max", "≤") for s in (max_metric or [])] + \
                            [(s, "min", "≥") for s in (min_metric or [])]:
        # R-103/W-7：规则**语法/数值**写错（`foo`、`X=abc`）是用法错误（rc=2），与未知
        # code/status 同类；旧实现只记一条 Failure（rc=1），而且因为没进 `checked`，
        # 后面还会追加一句假的"一条规则都没给"——把"规则写错"伪装成"没给规则"。
        res.checked.append(f"{op}-metric {spec}")
        try:
            name, threshold = _parse_assign(spec)
        except ValueError as exc:
            # 规则写错也是"给了一条规则"：进 checked，才不会再吐那句假的"一条规则都没给"。
            msg = str(exc)
            res.usage_errors.append(f"{op}-metric {spec}: {msg}")
            res.failures.append(Failure(f"{op}-metric {spec}", msg))
            res.passed = False
            continue
        _check_metric_rule(res, op=op, better=better, name=name, threshold=threshold,
                           report=report or {})

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
                    help="判词必须存在且状态至少 STATUS（默认 ok；闭集：ok/soft/refused。"
                         "写错即用法错误 rc=2）")
    ap.add_argument("--require-actionable", action="append", default=[], metavar="ID",
                    help="判词必须可作为行动依据（actionable=true / 前提全部成立）")
    ap.add_argument("--forbid-code", action="append", default=[], metavar="CODE",
                    help="出现该拒绝码即失败（可重复）。CODE 必须是 refusals.py 闭集成员；"
                         "未知 code 是用法错误（rc=2），绝不静默放绿")
    ap.add_argument("--max-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="指标必须 ≤ 阈值，**逐 kernel 判**：只要一个真 kernel 带这个指标，"
                         "每个真 kernel 都必须带且达标（缺项 / n/a / 超限 → 不通过并点名 "
                         "kernel）；只有 (device) 带的指标按设备级判 (device)")
    ap.add_argument("--min-metric", action="append", default=[], metavar="NAME=VALUE",
                    help="指标必须 ≥ 阈值（语义同 --max-metric：逐 kernel 判，任一个违反即"
                         "不通过并点名 kernel）")
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
    if res.usage_errors:
        # 规则本身写错（未知 code / 未知 status / 指标规则语法错 / actionable 不是布尔）：
        # 与"文件读不了"同一类用法错误，按项目惯例 rc=2 并把可用取值打出来。**绝不退 0**。
        if args.json:
            # R-103/W-7：用法错误 + `--json` 也必须吐机器可读的包 —— 旧实现只在
            # stderr 打印，stdout 全空，agent 拿到的 `usage_errors` 字段永远不可达
            # （它只能看到"命令没输出"，分不清是用法错还是工具崩了）。rc=2 保持不变，
            # 所以 CI 依然会失败。
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
            return 2
        for msg in res.usage_errors:
            print(f"error: {msg}", file=sys.stderr)
        print("hint: 拒绝码是 `refusals.py` 里 RefusalCode 的闭集（上面已列出全部合法取值）；"
              "指标规则要写 `NAME=VALUE`（VALUE 是数）；别忘了 `--forbid-code` 只吃 `.value`"
              "（小写下划线形式）。", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text(res))
    return 0 if res.passed else 1
