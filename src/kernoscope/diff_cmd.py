"""`kernoscope diff` —— 两次运行比出「真改进 / 真退化 / 分不出来 / 不可比」（W7）。

为什么它不是"两个数字相减"：本机实测的跨运行噪声可以达到 **5 个百分点**（见
bench/artifacts 的时长扫描）。没有 MDD（最小可分辨差异）时，相减出来的每一格都是
"变了"，agent 会照着噪声改代码 —— 那正是"会骗人的工具"最容易犯的错。

于是只有四种判词，且都有明确定义：
    improved          同协议、差值超过 MDD、方向是变好
    regressed         同协议、差值超过 MDD、方向是变坏
    indistinguishable 同协议、差值在 MDD 之内（**这就是"看起来变了但其实是噪声"**）
    incomparable      协议不同 / 缺 MDD / 一侧没有这个判词 —— 附 code 与补救办法

方向不明确的判词（例如发射槽利用率：高不一定好）不参与 improved/regressed，
单独列在 `not_judged` 里给出差值 —— 我们不替用户决定"高是好还是坏"。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .packet import load_packet
from .refusals import Refusal, RefusalCode

DIFF_SCHEMA_VERSION = 2

IMPROVED, REGRESSED, INDISTINGUISHABLE, INCOMPARABLE, NOT_JUDGED = (
    "improved", "regressed", "indistinguishable", "incomparable", "not_judged")
# v2：`not_judged`（只给差值、不下方向判词）从前被序列化成 `incomparable`，与"协议不同/
# 一侧缺失"混在同一个 token 里，机器消费者按 status 计数必然对不上（D25）。现在它是独立状态。

# 方向：只对"高好还是低好"有共识的判词下 improved/regressed 判词。
# 没有共识的一律进 not_judged —— 不替用户决定价值方向。
DIRECTION: dict[str, str] = {
    "kernel_time": "lower",
    "dram_bytes": "lower",
    "dram_bytes_per_launch": "lower",
    "dram_bytes_trace": "lower",
    "occupancy": "higher",
    "roofline": "higher",
    "l2_locality": "higher",
}

# 判词 → 用哪个指标的 MDD（误差带随包走；缺了就找 selfcheck 文件）
MDD_KEY: dict[str, str] = {
    "kernel_time": "kernel_time_pct",
    "dram_bytes": "dram_bytes_pct",
    "dram_bytes_per_launch": "dram_bytes_pct",
    "dram_bytes_trace": "dram_bytes_pct",
    "occupancy": "kernel_time_pct",
    "roofline": "dram_bytes_pct",
    "l2_locality": "dram_bytes_pct",
}


@dataclass
class Comparison:
    id: str
    status: str
    a: float | None = None
    b: float | None = None
    unit: str = ""
    delta: float | None = None
    delta_pct: float | None = None
    mdd: float | None = None
    direction: str = ""
    refusal: Refusal | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "status": self.status}
        for k in ("a", "b", "unit", "delta", "delta_pct", "mdd", "direction"):
            v = getattr(self, k)
            if v not in (None, ""):
                d[k] = v
        if self.note:
            d["note"] = self.note
        if self.refusal is not None:
            d["refusal"] = self.refusal.to_dict()
        return d


@dataclass
class DiffResult:
    a_path: str = ""
    b_path: str = ""
    protocol_a: str = ""
    protocol_b: str = ""
    protocol_diffs: dict[str, Any] = field(default_factory=dict)
    mdd: dict[str, float] = field(default_factory=dict)
    mdd_source: str = ""
    mdd_caveat: str = ""
    comparisons: list[Comparison] = field(default_factory=list)
    not_judged: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return not self.protocol_diffs

    def counts(self) -> dict[str, int]:
        """按 status 计数 —— 与机器看到的一致：`not_judged` 也是独立一格，不再并入 incomparable。"""
        c = {IMPROVED: 0, REGRESSED: 0, INDISTINGUISHABLE: 0, INCOMPARABLE: 0, NOT_JUDGED: 0}
        for x in self.comparisons:
            c[x.status] = c.get(x.status, 0) + 1
        for x in self.not_judged:
            c[NOT_JUDGED] = c.get(NOT_JUDGED, 0) + 1
        return c

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIFF_SCHEMA_VERSION,
            "tool": "kernoscope",
            "a": self.a_path,
            "b": self.b_path,
            "protocol": {"same": self.comparable, "a": self.protocol_a,
                         "b": self.protocol_b, "diffs": self.protocol_diffs},
            "mdd": {"source": self.mdd_source, "values": self.mdd},
            "comparisons": [c.to_dict() for c in self.comparisons],
            "not_judged": [c.to_dict() for c in self.not_judged],
            "summary": self.counts(),
            "notes": self.notes,
        }

    def render(self) -> str:
        out = [f"Kernoscope diff {DIFF_SCHEMA_VERSION}"]
        if not self.comparable:
            out.append(f"  两次运行**不可比**（{', '.join(sorted(self.protocol_diffs))}）："
                       f"只给差值，不给方向判词")
            for k, v in sorted(self.protocol_diffs.items()):
                out.append(f"    {k}: {v}")
        if self.mdd:
            out.append("  MDD（来自 " + (self.mdd_source or "?") + "）："
                       + " · ".join(f"{k}={v}%" for k, v in sorted(self.mdd.items())))
            if self.mdd_caveat:
                out.append(f"    {self.mdd_caveat}")
        else:
            out.append("  没有 MDD：分不出噪声与真差别 —— 跑一次 `kernoscope selftest` "
                       "并把结果用 --selfcheck 传进来")
        out.append("")
        mark = {IMPROVED: "better", REGRESSED: "worse ", INDISTINGUISHABLE: "same  ",
                INCOMPARABLE: "n/a   "}
        for c in self.comparisons:
            line = f"  [{mark[c.status]:>7}] {c.id:<22}"
            if c.a is not None and c.b is not None:
                line += f" {c.a:>10.2f} → {c.b:>10.2f} {c.unit:<4}"
            if c.delta is not None:
                line += f" Δ{c.delta:+.2f}"
                if c.delta_pct is not None:
                    line += f" ({c.delta_pct:+.1f}%)"
            if c.mdd is not None:
                line += f"  MDD {c.mdd}%"
            out.append(line)
            if c.note:
                out.append(f"            {c.note}")
            if c.refusal is not None:
                rem = c.refusal.remedy_or_default
                out.append(f"            {c.refusal.code.value}：{rem.action}"
                           + (f"  [{rem.command}]" if rem.command else ""))
        if self.not_judged:
            out.append("")
            out.append("  不判方向（高不一定好，或缺少可比数值）：")
            for c in self.not_judged:
                line = f"    {c.id:<24}"
                if c.a is not None and c.b is not None:
                    line += f" {c.a:>10.2f} → {c.b:>10.2f} {c.unit:<4}"
                if c.delta is not None:
                    line += f" Δ{c.delta:+.2f}"
                out.append(line)
        counts = self.counts()
        out.append("")
        out.append("  汇总：" + " · ".join(f"{k}={v}" for k, v in counts.items()
                                          if k != INCOMPARABLE or v)
                   + (f" · 不判方向={len(self.not_judged)}" if self.not_judged else ""))
        for n in self.notes:
            out.append(f"  注：{n}")
        return "\n".join(out)


def _verdicts(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {v["id"]: v for v in (packet.get("verdicts") or []) if v.get("id")}


def _protocol_diffs(pa: dict[str, Any], pb: dict[str, Any]) -> dict[str, Any]:
    """哪些协议差异会让两次运行不可比（并区分"只是不同"与"一强一弱"）。"""
    diffs: dict[str, Any] = {}
    if pa.get("tool_version") != pb.get("tool_version"):
        diffs["tool_version"] = f"{pa.get('tool_version')} → {pb.get('tool_version')}"
    if pa.get("schema_version") != pb.get("schema_version"):
        diffs["packet_schema"] = f"{pa.get('schema_version')} → {pb.get('schema_version')}"
    # 锁频状态不同：只说"协议不同"太含糊，读者要的是"怎么才可比"（择中项 ⑥ 的处方面）
    la = (pa.get("environment") or {}).get("clock_lock") or pa.get("clock_lock")
    lb = (pb.get("environment") or {}).get("clock_lock") or pb.get("clock_lock")
    if la and lb and la != lb:
        diffs["clock_lock"] = f"{la} → {lb}（锁频状态不同 ⇒ 两次运行的时长/能效不可相减）"
    ha = (pa.get("protocol") or {}).get("hash")
    hb = (pb.get("protocol") or {}).get("hash")
    if ha != hb:
        diffs["protocol_hash"] = (f"{ha} → {hb}"
                                  f"（标定 / 窗口与 warmup / 精度 / 内核选择任一项不同都会这样）")
        for k in ("window_mode", "basis"):
            va = (pa.get("protocol") or {}).get(k)
            vb = (pb.get("protocol") or {}).get(k)
            if va != vb:
                diffs[k] = f"{va} → {vb}"
    ea = (pa.get("env") or {}).get("tier")
    eb = (pb.get("env") or {}).get("tier")
    if ea is not None and eb is not None and ea != eb:
        diffs["capability_tier"] = f"{ea} → {eb}（能力层不同 ⇒ 数字口径不同）"
    if bool(pa.get("synthetic")) != bool(pb.get("synthetic")):
        diffs["synthetic"] = "一侧是合成数据"
    return diffs


def diff_packets(pa: dict[str, Any], pb: dict[str, Any], *,
                 selfcheck: dict[str, Any] | None = None,
                 a_path: str = "", b_path: str = "") -> DiffResult:
    res = DiffResult(a_path=a_path, b_path=b_path,
                     protocol_a=str((pa.get("protocol") or {}).get("hash", "")),
                     protocol_b=str((pb.get("protocol") or {}).get("hash", "")))
    res.protocol_diffs = _protocol_diffs(pa, pb)

    # MDD：优先用包里的（随判词走），其次 selfcheck 文件；取较大者（保守）
    mdd: dict[str, float] = {}
    for key in ("dram_bytes_pct", "kernel_time_pct"):
        vals = []
        for p in (pa, pb):
            for v in p.get("verdicts") or []:
                if v.get("mdd") is not None and MDD_KEY.get(v.get("id", "")) == key:
                    vals.append(float(v["mdd"]))
        if selfcheck and (selfcheck.get("mdd") or {}).get(key) is not None:
            vals.append(float(selfcheck["mdd"][key]))
        if vals:
            mdd[key] = max(vals)
    res.mdd = mdd
    res.mdd_caveat = ((selfcheck or {}).get("mdd") or {}).get("caveat", "")
    res.mdd_source = ("包内误差带 + --selfcheck" if (mdd and selfcheck) else
                      ("包内误差带（--ks-selfcheck 生成）" if mdd else ""))

    va, vb = _verdicts(pa), _verdicts(pb)
    for vid in sorted(set(va) | set(vb)):
        A, B = va.get(vid), vb.get(vid)
        if A is None or B is None:
            res.comparisons.append(Comparison(
                id=vid, status=INCOMPARABLE,
                refusal=Refusal(RefusalCode.METRIC_ABSENT,
                                f"只有一侧有这个判词（{'b' if A is None else 'a'} 缺）："
                                f"不能当成 0 来比"),
                note="一侧缺失" + ("（通常是那次运行该口径不成立）")))
            continue
        direction = DIRECTION.get(vid)
        c = Comparison(id=vid, status=INCOMPARABLE, unit=str(B.get("unit") or A.get("unit") or ""),
                       direction=direction or "")
        c.a = A.get("value")
        c.b = B.get("value")
        key = MDD_KEY.get(vid, "dram_bytes_pct")
        c.mdd = mdd.get(key)
        # 差值先算出来：不可比不等于"信息不给" —— 读者仍然能看到两个数差多少，
        # 只是我们不下方向判词。
        if c.a is not None and c.b is not None:
            # X-19（第七轮 P1，口径级）：同一个判词 id 的**两侧必须是同一个量**。
            # M13 的第一级（按实测时长占比，单位 %）与第二级（省下的钱，单位 CNY/h）曾经
            # 共用 `saving_estimate` 这一个 id，于是这里把 96.6 % 与 1.9801 CNY/h 直接相减、
            # 吐出 `delta_pct=-97.95%` 并把单位标成 CNY/h —— 正是本项目最老的错误族
            # （用单一表示承载多重语义）。判词 id 已经拆分（见 packet），这里是第二道防线：
            # 只要单位或 `facts.basis` 不同，就不给差值、不给百分比。
            unit_a = str(A.get("unit") or "")
            unit_b = str(B.get("unit") or "")
            basis_a = str((A.get("facts") or {}).get("basis") or "")
            basis_b = str((B.get("facts") or {}).get("basis") or "")
            if unit_a != unit_b or basis_a != basis_b:
                why = [f"单位 {unit_a!r} vs {unit_b!r}"]
                if basis_a != basis_b:
                    why.append(f"basis {basis_a!r} vs {basis_b!r}")
                c.refusal = Refusal(
                    RefusalCode.PROTOCOL_MISMATCH,
                    f"同一个判词 id 的两次运行不是同一个量（{'、'.join(why)}）："
                    f"不给差值、也不给百分比")
                res.comparisons.append(c)
                continue
            c.delta = round(float(c.b) - float(c.a), 4)
            c.delta_pct = (round(100.0 * c.delta / abs(float(c.a)), 3)
                           if float(c.a) != 0 else None)
        if not res.comparable:
            c.refusal = Refusal(RefusalCode.PROTOCOL_MISMATCH,
                                "协议不同：两次运行的口径不可比（同协议才谈方向）")
            res.comparisons.append(c)
            continue
        if c.a is None or c.b is None:
            ra = (A.get("refusal") or {}).get("code")
            rb = (B.get("refusal") or {}).get("code")
            if c.a is None and c.b is None and ra == rb:
                c.refusal = Refusal(RefusalCode.BOTH_REFUSED,
                                    f"两侧都被 {ra} 挡住：这次比较没有数字可比"
                                    f"（不是「没变化」，也不是「退化」）")
            else:
                c.refusal = Refusal(RefusalCode.METRIC_ABSENT,
                                    "一侧有数值、另一侧被拒 ⇒ 不可比（不能拿 0 顶替）")
            res.comparisons.append(c)
            continue
        if direction is None:
            c.status = NOT_JUDGED
            c.note = "这个量没有公认的价值方向（高好还是低好），所以只给差值、不下判词"
            res.not_judged.append(c)
            continue
        if c.mdd is None:
            c.refusal = Refusal(RefusalCode.MDD_UNKNOWN,
                                "没有这个指标的 MDD：分不出噪声与真差别")
            res.comparisons.append(c)
            continue
        # 判据一律是**相对差 vs 相对误差带**：MDD 里的每个键（dram_bytes_pct /
        # kernel_time_pct）量的都是相对离散度。旧实现让 `%` 单位的判词拿**百分点**去比
        # 相对带 —— roofline 20%→15%（真实退化 25%）被判成噪声（D16）。
        if c.a == 0 and c.b != 0:
            # 相对差无定义：基线是 0 时不能用相对误差带。旧实现把 magnitude 当成 0，
            # 于是「0 → 500 MB」被判成"没变化"（D17）。
            c.refusal = Refusal(RefusalCode.ZERO_BASELINE,
                                f"基线为 0，相对误差带不适用：只能报绝对差值 "
                                f"（{c.delta:+.6g}{(' ' + c.unit) if c.unit else ''}）")
            c.status = INCOMPARABLE
            res.comparisons.append(c)
            continue
        magnitude = abs(c.delta_pct if c.delta_pct is not None else 0.0)
        if magnitude <= c.mdd:
            c.status = INDISTINGUISHABLE
            c.note = (f"相对差 {magnitude:.2f}% 在 MDD {c.mdd}% 之内 ⇒ 这是噪声，"
                      f"不要当成改进")
        else:
            better = (c.delta < 0) if direction == "lower" else (c.delta > 0)
            c.status = IMPROVED if better else REGRESSED
            c.note = (f"相对差超过 MDD {c.mdd}%（{magnitude:.2f}%）⇒ 方向明确"
                      f"（{direction} is better）")
        res.comparisons.append(c)
    if not res.comparable:
        res.notes.append("协议不同时我们拒绝给方向判词：宁可说不可比，也不说一个会误导的「变好了」")
    if not mdd:
        res.notes.append("没有 MDD ⇒ 所有判词都是 incomparable(mdd_unknown)："
                         "先跑 `kernoscope selftest --out machine-selfcheck.json`")
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernoscope diff",
        description="比较两次运行（决策包或完整报告）：真改进 / 真退化 / 分不出来 / 不可比",
    )
    p.add_argument("a", help="基线：*.ksreport.json 或 --packet 产出的包")
    p.add_argument("b", help="新结果：同上")
    p.add_argument("--selfcheck", default=None,
                   help="kernoscope selftest 产出的自检文件（提供 MDD）")
    p.add_argument("--json", action="store_true", help="结构化输出（给 agent 用）")
    p.add_argument("--fail-on", default="none", choices=["none", "regressed", "change"],
                   help="none=永远 exit 0（默认）；regressed=有真退化时 exit 1；"
                        "change=有 improved/regressed 时 exit 1（CI 卡门用）")
    p.add_argument("--strict", action="store_true",
                   help="不可比时 exit 2（协议不同或缺 MDD 都算没给出结论）")
    p.add_argument("--quiet", action="store_true")
    return p


def diff_main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.a, args.b):
        if not Path(path).exists():
            print(f"error: 找不到 {path}", file=sys.stderr)
            return 1
    try:
        pa = load_packet(args.a)
        pb = load_packet(args.b)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: 读取失败：{e}", file=sys.stderr)
        return 1
    selfcheck = None
    if args.selfcheck:
        try:
            selfcheck = json.loads(Path(args.selfcheck).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"warn: --selfcheck 读取失败（{e}）：本次只能用包内误差带", file=sys.stderr)

    res = diff_packets(pa, pb, selfcheck=selfcheck, a_path=args.a, b_path=args.b)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(res.render())

    counts = res.counts()
    if args.strict and (counts[INCOMPARABLE] or not res.comparable):
        print("error(strict): 没有给出可比结论（协议不同或缺少 MDD）", file=sys.stderr)
        return 2
    if args.fail_on == "regressed" and counts[REGRESSED]:
        print(f"error: {counts[REGRESSED]} 项真退化（超过 MDD）", file=sys.stderr)
        return 1
    if args.fail_on == "change" and (counts[REGRESSED] or counts[IMPROVED]):
        print(f"error: {counts[IMPROVED]} improved / {counts[REGRESSED]} regressed",
              file=sys.stderr)
        return 1
    return 0


__all__ = ["DIFF_SCHEMA_VERSION", "IMPROVED", "INCOMPARABLE", "INDISTINGUISHABLE",
           "REGRESSED", "Comparison", "DiffResult", "build_parser", "diff_main",
           "diff_packets"]
