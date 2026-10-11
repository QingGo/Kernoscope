"""单文件 HTML 渲染器（N9）—— 把决策包变成一张可截图、可附进 issue 的证据单。

为什么是**渲染器**而不是 Web UI：这个工具的人视图是**传播引擎**（一张截图比一屏终端
文字走得远，教程也需要图），但一个 Web UI 会带来服务器、依赖、版本漂移与"离线打不开"。
所以形态定成：输入决策包（JSON）→ 输出**一个自包含 `.html`**：

  * 零依赖（只用标准库 `html`/`json`/`argparse`），无服务器、无 JS 框架、无 CDN；
  * 打开离线可用，也能直接当附件贴进 issue / 截图进教程；
  * 渲染是**纯函数**：同一个包渲染两次字节完全相同（不写时间戳、不读环境）。

**出处必须和数字印在一起**（这是本工具存在的理由）：每条判词/数值旁边都有可见角标
（`exact`/`measured`/`estimated`/`derived`，四种颜色 + 图例）；拿不到数的位置印
`n/a` + 拒绝码，**绝不印 `0`** —— 0 会被读成一个读数，n/a 才是事实。渲染器也不渲染
包里的 `facts.uncitable_value`（那是"曾经算出过、但已不可引用"的历史值，给 agent 用；
印在给人看的图上，一个划掉的 0 仍然会被读成 0）。

尺寸预算：正常包（≤4 KB）渲染出来必须 ≤ `SIZE_BUDGET_BYTES`（64 KB），由测试守着 ——
预算不是为了省字节，而是防止"顺手加一张图/一个统计"把单文件拖成几 MB。

用法（手动检查）：

    python -m kernoscope.html_view packet.json > out.html
    python -m kernoscope.html_view rep.ksreport.json --title "A/B · 旧版" -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from typing import Any, Sequence

from .packet import (OK, REFUSED, SOFT, load_packet, phrase_verdict, premise_label,
                     refusal_index, render_onepage)
from .refusals import code_title

SIZE_BUDGET_BYTES = 64 * 1024        # 正常包（≤4 KB）渲染的硬上限（测试守着）
FACT_PAIRS_MAX = 8                   # 判词的 facts 最多铺开几对（防止未来的包拖大页面）
FACT_VALUE_LIMIT = 72                # 每个 fact 值的显示上限

# 出处角标的四种颜色**就是**这个工具的语义层：同一个数字，颜色不同，可引用程度不同。
PROVENANCE_ORDER: tuple[str, ...] = ("exact", "measured", "estimated", "derived")

PROVENANCE_LABELS: dict[str, str] = {
    "exact": "与 ncu 同源（CUPTI activity / cubin ELF / 解析算术）",
    "measured": "我们真的读到的量（NVML 管理计数器 / 探针 / 轨迹计数）",
    "estimated": "模型或代理（roofline / DRAM 字节代理 / 缓存模型）",
    "derived": "由实测或精确量再算出来的统计量与判词",
    "unavailable": "拿不到这个数（缺数据源或前提不成立）—— 只报 n/a 与拒绝码，绝不编数字",
}

_STATUS_LABELS: dict[str, str] = {OK: "成立", SOFT: "成立（口径较弱）", REFUSED: "不给数字"}
_STATUS_CLASSES: dict[str, str] = {OK: "st-ok", SOFT: "st-soft", REFUSED: "st-refused"}

_COST_LABELS: dict[str, str] = {
    "wall_s": "wall 时间（s）",
    "packet_bytes": "决策包大小（B）",
    "report_bytes": "完整报告大小（B）",
    "launches": "launch 次数",
    "unmeasured_metrics": "没测到的指标数",
    "selftest_band": "本机误差带（selftest）",
    "trimmed": "按尺寸预算裁剪的步骤",
    "trimmed_ids": "被裁掉细节的条目",
    "over_budget": "包超出尺寸预算",
}
_COST_ORDER: tuple[str, ...] = ("wall_s", "packet_bytes", "report_bytes", "launches",
                               "unmeasured_metrics", "selftest_band", "trimmed",
                               "trimmed_ids", "over_budget")

# 内联 CSS：没有 <link>、没有 url()、没有外部字体 —— 单文件离线可开是一等约束。
_CSS = """
:root{--ink:#1c1c1c;--mut:#5d5d5d;--line:#d8d8d4;--bg:#f7f7f4;--panel:#fff}
*{box-sizing:border-box}
body{margin:0;padding:26px 20px 44px;background:var(--bg);color:var(--ink);
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",
 "PingFang SC","Microsoft YaHei",sans-serif}
main{display:block;max-width:1080px;margin:0 auto}
h1{font-size:21px;line-height:1.35;margin:0 0 4px}
h2{font-size:16px;margin:30px 0 8px;padding-left:8px;border-left:4px solid #444}
p{margin:6px 0}
.sub{color:var(--mut);font-size:13px;margin:0}
.warn{background:#fff3cd;border:1px solid #e0b667;color:#7a4a00;padding:6px 10px;
 border-radius:5px;font-weight:600;margin-top:10px}
.conclusion{font-size:17px;font-weight:600;background:var(--panel);border:1px solid var(--line);
 border-left:6px solid #333;border-radius:5px;padding:10px 14px;margin-top:8px}
table{border-collapse:collapse;width:100%;background:var(--panel);margin-top:4px}
caption{caption-side:top;text-align:left;color:var(--mut);font-size:12.5px;padding:2px 0 6px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
thead th{background:#efefeb;font-size:13px;white-space:nowrap}
tr.premise td,tr.vnext td,tr.vrefusal td{font-size:13px;padding:4px 9px 4px 26px}
tr.premise-ok td{color:#333}
tr.premise-bad td{background:#fdf0ee;color:#8a1616;font-weight:600}
tr.vrefusal td{background:#fbf1ef;color:#5b1a1a}
tr.vnext td{color:var(--mut)}
td.value{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:600}
td.unit,td.prov{color:var(--mut);white-space:nowrap}
td.status,th.status{white-space:nowrap;font-size:13px}
.st-ok{color:#0b6b3a;font-weight:600}
.st-soft{color:#8a5a00;font-weight:600}
.st-refused{color:#8a1616;font-weight:700}
.st-unknown{color:#666}
.badge{display:inline-block;padding:0 7px;border-radius:9px;border:1px solid #bbb;
 font-size:12px;font-weight:700;letter-spacing:.02em;white-space:nowrap;color:#333}
.badge-exact{background:#d9f2e2;color:#0b6b3a;border-color:#8fd0aa}
.badge-measured{background:#dbeafe;color:#124a86;border-color:#93b8ef}
.badge-estimated{background:#fdecc8;color:#7a4a00;border-color:#e0b667}
.badge-derived{background:#eee0fb;color:#5b2a86;border-color:#c1a1e6}
.badge-unavailable{background:#fadbd8;color:#8a1616;border-color:#e6a29b}
.badge-other{background:#ececec;color:#444;border-color:#bbb}
.na{color:#8a1616;font-weight:700}
.muted{color:var(--mut);font-weight:400}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;
 background:#f0f0ec;border:1px solid #e2e2de;border-radius:4px;padding:0 4px}
.facts{color:var(--mut);font-size:12.5px;margin-top:3px}
pre{background:#1f1f1f;color:#f2f2f2;padding:12px 14px;border-radius:6px;overflow-x:auto;
 font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
details{margin-top:8px}
summary{cursor:pointer;color:var(--mut);font-size:13px}
footer{color:#7a7a7a;font-size:12px;margin-top:34px;border-top:1px solid var(--line);
 padding-top:10px}
@media print{body{background:#fff;padding:0}}
"""


# --------------------------------------------------------------------------
# 小工具：转义与取值
# --------------------------------------------------------------------------
def _e(value: Any) -> str:
    """任何来自包/报告的文本都必须走这里，包括看起来"只能是数字"的字段。

    渲染层不接受"这个字段大概安全"的假设：kernel 名、note、拒绝消息、remedy 命令里
    都可能出现 `<`/`&`（真机上 C++ 模板名就带 `<`），不转义就等于把包里的字符串当标记执行。
    """
    return html.escape("" if value is None else str(value), quote=True)


def _scalar(value: Any, limit: int = FACT_VALUE_LIMIT) -> str:
    """把包里的任意标量/容器压成一行可读文本（**未转义**，调用方负责 `_e`）。

    `None` 明确写成"读不到（未知，不是 0）"：这个仓库里最容易犯的错就是把"没读到"
    当成 0（或反过来）。渲染层也不接受这个歧义。
    """
    if value is None:
        return "读不到（未知，不是 0）"
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (list, tuple)):
        text = json.dumps(list(value), ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _badge(provenance: str) -> str:
    """出处角标：四种等级各有颜色，未知等级走中性灰（不猜、不冒充 exact）。"""
    p = str(provenance or "").strip().lower()
    if p in PROVENANCE_LABELS:
        return f'<span class="badge badge-{p}">{_e(p)}</span>'
    return f'<span class="badge badge-other">{_e("未标注（旧包）")}</span>'


def _remedy_cost(entry: Any) -> str:
    """拒绝的代价：与 `packet._cost_text` 同一套口径（几秒 / 几分钟 / 需人工 / 本机不可行）。"""
    if not isinstance(entry, dict):
        return ""
    kind = str(entry.get("cost_kind") or "")
    if kind == "unavailable":
        return "本机不可行"
    if kind == "manual":
        return "需人工"
    try:
        seconds = float(entry.get("cost_s") or 0.0)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return "≈0"
    return f"≈{seconds:.0f}s" if seconds < 90 else f"≈{seconds / 60:.0f}min"


def _table(headers: Sequence[str], rows: Sequence[str], caption: str = "") -> str:
    head = "".join(f'<th scope="col">{_e(h)}</th>' for h in headers)
    cap = f"<caption>{_e(caption)}</caption>" if caption else ""
    return (f"<table>{cap}\n<thead><tr>{head}</tr></thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>")


# --------------------------------------------------------------------------
# 各段
# --------------------------------------------------------------------------
def _meta_line(packet: dict[str, Any]) -> str:
    protocol = packet.get("protocol") or {}
    env = packet.get("env") or {}
    bits = [f"工具 kernoscope {packet.get('tool_version') or '?'}"]
    digest = str(protocol.get("hash") or "")
    if digest:
        bits.append(f"协议 {digest[7:19] or digest}")
    if protocol.get("basis"):
        bits.append(f"口径 {protocol['basis']}")
    if protocol.get("window_mode"):
        bits.append(f"窗口 {protocol['window_mode']}")
    for key in ("calibration", "memtrace"):
        value = protocol.get(key)
        if value:
            bits.append(f"{key}={_scalar(value, 48)}")
    if env:
        tier = f"层 {env.get('tier')}"
        if env.get("tier_name"):
            tier += f"（{env['tier_name']}）"
        bits.append(tier)
        if env.get("counter_state"):
            bits.append(f"计数器 {env['counter_state']}")
    return " · ".join(bits)


def _header_html(packet: dict[str, Any], doc_title: str) -> str:
    out = ["<header>", f"<h1>{_e(doc_title)}</h1>",
           f'<p class="sub">{_e(_meta_line(packet))}</p>']
    if packet.get("synthetic"):
        out.append('<p class="warn">合成数据：不是真实测量，只用于检查渲染形状。</p>')
    out.append("</header>")
    return "\n".join(out)


def _onepage_conclusion(onepage_text: str) -> str:
    """一屏文本里的结论行（优先 `结论：` 那一行，否则第一行非空）。"""
    lines = [line.strip() for line in onepage_text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("结论："):
            return line[len("结论："):].strip()
    return lines[0] if lines else ""


def _derive_conclusion(packet: dict[str, Any]) -> str:
    """没给一屏文本时，从包派生一行结论。

    复用**公开的**人视图 `render_onepage`（只取它的「结论：」行）而不是另写一套规则：
    两个视图的散文必须只有一个来源，否则迟早分叉。
    """
    for line in render_onepage(packet).splitlines():
        if line.startswith("结论："):
            return line[len("结论："):].strip()
    return ""


def _conclusion_html(packet: dict[str, Any], onepage_text: str) -> str:
    text = onepage_text.rstrip()
    conclusion = _onepage_conclusion(text) if text.strip() else ""
    if not conclusion:
        conclusion = _derive_conclusion(packet) or "见下方判词、拒绝与代价。"
    out = ['<section id="conclusion">', "<h2>结论</h2>",
           f'<p class="conclusion">{_e(conclusion)}</p>']
    if text.strip():
        out.append("<details open><summary>一屏文本（原样嵌入，不重新排版）</summary>")
        out.append(f'<pre class="onepage">{_e(text)}</pre>')
        out.append("</details>")
    out.append("</section>")
    return "\n".join(out)


def _legend_html() -> str:
    rows = [f'<tr><td>{_badge(p)}</td><td>{_e(PROVENANCE_LABELS[p])}</td></tr>'
            for p in PROVENANCE_ORDER]
    rows.append(f'<tr><td>{_badge("unavailable")}</td>'
                f'<td>{_e(PROVENANCE_LABELS["unavailable"])}</td></tr>')
    return ('<section id="provenance-legend">\n<h2>出处图例（数字旁边的角标）</h2>\n'
            + _table(("角标", "含义"), rows,
                     "角标颜色与数字同时出现：同一个数，出处不同，能引用的程度就不同。")
            + "\n</section>")


def _facts_html(facts: Any) -> str:
    if not isinstance(facts, dict) or not facts:
        return ""
    # `uncitable_value` 是"曾经算出过、但已不可引用"的历史值（前提被推翻时由 build() 放进
    # facts）。它是给机器对比用的：印在给人看的图上，一个划掉的 0 仍然会被读成 0 ——
    # 所以人视图只印 n/a + code，历史值留在 --json 里。
    items = [(k, v) for k, v in facts.items() if k != "uncitable_value"]
    if not items:
        return ""
    pairs = [f"{_e(key)}={_e(_scalar(value))}" for key, value in items[:FACT_PAIRS_MAX]]
    if len(items) > FACT_PAIRS_MAX:
        pairs.append(f"…另 {len(items) - FACT_PAIRS_MAX} 项见 --json")
    return f'<div class="facts">{" · ".join(pairs)}</div>'


def _premise_html(premise: Any) -> str:
    p = premise if isinstance(premise, dict) else {}
    pid = str(p.get("id") or "")
    unit = str(p.get("unit") or "")
    ok = bool(p.get("ok"))
    body = _e(_scalar(p.get("observed")))
    if unit:
        body += _e(unit)
    if p.get("threshold") is not None:
        body += f'（门槛 {_e(_scalar(p["threshold"]))}{_e(unit)}）'
    cls = "premise-ok" if ok else "premise-bad"
    flag = "成立" if ok else "不成立"
    return (f'<tr class="premise {cls}"><td colspan="5">前提 <code>{_e(pid)}</code> '
            f'{_e(premise_label(pid))}：{body} → <b>{flag}</b></td></tr>')


def _verdict_refusal_html(refusal: Any, code: str, index: dict[str, dict[str, Any]]) -> str:
    ref = refusal if isinstance(refusal, dict) else {}
    entry = index.get(code) or {}
    bits = [f'拒绝码 <code>{_e(code or "?")}</code>（{_e(code_title(code))}）']
    if entry.get("action"):
        bits.append(f'怎么办：{_e(entry["action"])}')
    cost = _remedy_cost(entry)
    if cost:
        bits.append(f'代价 {_e(cost)}')
    if entry.get("cmd"):
        bits.append(f'<code>{_e(entry["cmd"])}</code>')
    if ref.get("observed") or ref.get("threshold"):
        bits.append(f'实测 {_e(_scalar(ref.get("observed") or {}, 120))}'
                    f' · 门槛 {_e(_scalar(ref.get("threshold") or {}, 120))}')
    return f'<tr class="vrefusal"><td colspan="5">{" · ".join(bits)}</td></tr>'


def _verdict_html(verdict: Any, index: dict[str, dict[str, Any]]) -> str:
    v = verdict if isinstance(verdict, dict) else {}
    vid = str(v.get("id") or "")
    status = str(v.get("status") or "")
    code = str(((v.get("refusal") or {}) or {}).get("code") or "")
    out = [f'<tr class="verdict" id="verdict-{_e(vid)}">',
           f'<td class="status {_STATUS_CLASSES.get(status, "st-unknown")}">'
           f'{_e(_STATUS_LABELS.get(status, status or "?"))}</td>',
           f'<td class="claim">{_e(phrase_verdict(v, index))}{_facts_html(v.get("facts"))}</td>']
    if status == REFUSED:
        # n/a 与 code **同一格**：读者看到"没有数"的下一秒就看到"为什么、去哪查"。
        out.append(f'<td class="value"><span class="na">n/a</span> '
                   f'<code>{_e(code or "?")}</code></td>')
        out.append('<td class="unit"></td>')
        out.append(f'<td class="prov">{_badge("unavailable")}</td>')
    else:
        value = v.get("value")
        if value is None:
            out.append('<td class="value"><span class="muted">—</span></td>')
        else:
            out.append(f'<td class="value">{_e(_scalar(value, 40))}</td>')
        out.append(f'<td class="unit">{_e(v.get("unit") or "")}</td>')
        out.append(f'<td class="prov">{_badge(str(v.get("provenance") or ""))}</td>')
    out.append("</tr>")
    for premise in v.get("premises") or []:
        out.append(_premise_html(premise))
    if status == REFUSED:
        out.append(_verdict_refusal_html(v.get("refusal"), code, index))
    elif v.get("next"):
        out.append(f'<tr class="vnext"><td colspan="5">下一步：{_e(v["next"])}</td></tr>')
    return "\n".join(out)


def _verdicts_html(packet: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    verdicts = packet.get("verdicts") or []
    more = packet.get("verdicts_more") or {}
    rows = [_verdict_html(v, index) for v in verdicts]
    if not rows:
        rows.append('<tr><td colspan="5">没有任何可下结论的判词：见下面「没测到什么」。</td></tr>')
    body = _table(("状态", "判词与前提", "值", "单位", "出处"), rows,
                  "每条判词都带出处角标；前提不成立的判词只能是 refused（数字不可引用）。")
    tail = ""
    if more:
        tail = (f'<p class="sub">另有 {_e(more.get("count"))} 条判词超出上限被裁：'
                f'{_e("、".join(str(i) for i in (more.get("ids") or [])))}'
                f'（完整内容见完整报告 / --json）。</p>')
    return ('<section id="verdicts">\n<h2>判词：结论与它的前提</h2>\n'
            + body + ("\n" + tail if tail else "") + "\n</section>")


def _refusal_row(code: str, entry: Any, source: str, message: str = "") -> str:
    r = entry if isinstance(entry, dict) else {}
    title = code_title(code)
    note = f'<div class="muted">{_e(_scalar(message, 160))}</div>' if message else ""
    cmd = r.get("cmd")
    return ("<tr>"
            f'<td><code>{_e(code)}</code><div class="muted">来源：{_e(source)}</div></td>'
            f'<td>{_e(title)}{note}</td>'
            f'<td>{_e(r.get("action") or "—")}</td>'
            f'<td>{_e(_remedy_cost(r) or "—")}</td>'
            f'<td>{f"<code>{_e(cmd)}</code>" if cmd else "—"}</td>'
            "</tr>")


def _refusals_html(packet: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    env_ref = [d for d in (packet.get("environment_refusals") or []) if isinstance(d, dict)]
    env_by_code = {str(d.get("code")): d for d in env_ref if d.get("code")}
    rows = []
    for code, entry in index.items():
        code = str(code)
        source = "本机缺件" if code in env_by_code else "判词 / 未测项"
        message = str((env_by_code.get(code) or {}).get("message") or "")
        rows.append(_refusal_row(code, entry, source, message))
    if not rows:
        rows.append('<tr><td colspan="5">没有拒绝条目：这次没有"不给数字"的结论。</td></tr>')
    omitted = packet.get("refusals_omitted") or {}
    tail = ""
    if omitted:
        codes = "、".join(str(c) for c in (omitted.get("codes") or []))
        tail = (f'<p class="sub">还有 {_e(omitted.get("count"))} 条拒绝条目按尺寸预算省略'
                + (f'（{_e(codes)}{"…" if omitted.get("codes_truncated") else ""}）' if codes else "")
                + "；它们的 remedy 仍可用 <code>kernoscope remedy &lt;code&gt;</code> 查到。</p>")
    body = _table(("拒绝码", "说明", "怎么办", "代价", "命令"), rows,
                  "拒绝本身是输出：每个 code 都给「为什么」与「怎么办」（代价单独一列）。")
    return ('<section id="refusals">\n<h2>拒绝与补救</h2>\n'
            + body + ("\n" + tail if tail else "") + "\n</section>")


def _unmeasured_row(code: str, group: Any, index: dict[str, dict[str, Any]]) -> str:
    g = group if isinstance(group, dict) else {}
    entry = index.get(code) or {}
    examples = "、".join(str(x) for x in (g.get("examples") or []))
    return ("<tr>"
            f'<td><code>{_e(code)}</code></td>'
            f'<td>{_e(code_title(code))}'
            + (f'<div class="muted">怎么办：{_e(entry.get("action"))}</div>'
               if entry.get("action") else "")
            + "</td>"
            f'<td class="value">{_e(g.get("count") if g.get("count") is not None else "?")}</td>'
            f'<td>{_e(examples or "—")}</td>'
            "</tr>")


def _unmeasured_html(packet: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    groups = packet.get("unmeasured") or []
    seen = {str(g.get("code")) for g in groups if isinstance(g, dict)}
    rows = [_unmeasured_row(str(g.get("code") or "?"), g, index) for g in groups]
    blocked = (packet.get("env") or {}).get("blocked_by") or []
    for code in blocked:
        code = str(code)
        if code in seen:
            continue
        seen.add(code)
        entry = index.get(code) or {}
        rows.append("<tr>"
                    f'<td><code>{_e(code)}</code></td>'
                    f'<td>{_e(code_title(code))}'
                    + (f'<div class="muted">怎么办：{_e(entry.get("action"))}</div>'
                       if entry.get("action") else "")
                    + "</td>"
                    '<td class="value"><span class="muted">本机缺件</span></td>'
                    "<td>—</td></tr>")
    if not rows:
        rows.append('<tr><td colspan="4">没有"没测到"的组：这次该测的都测到了。</td></tr>')
    more = packet.get("unmeasured_more") or {}
    tail = ""
    if more:
        tail = (f'<p class="sub">另有 {_e(more.get("groups"))} 组 / '
                f'{_e(more.get("metrics"))} 个指标按尺寸预算合并（完整清单见完整报告）。</p>')
    body = _table(("拒绝码", "说明", "指标数", "例子"), rows,
                  "没测到的东西按原因归组（不是按指标名铺开）：先看哪一类挡住了多少。")
    return ('<section id="unmeasured">\n<h2>没测到什么</h2>\n'
            + body + ("\n" + tail if tail else "") + "\n</section>")


def _cost_value(value: Any) -> str:
    if isinstance(value, bool):
        return "有" if value else "无"
    if isinstance(value, (list, tuple)):
        items = [_scalar(v, 40) for v in value]
        return "、".join(items) if items else "—"
    return _scalar(value, 160)


def _cost_html(packet: dict[str, Any]) -> str:
    cost = packet.get("cost") if isinstance(packet.get("cost"), dict) else {}
    rows = []
    ordered = [k for k in _COST_ORDER if k in cost]
    ordered += [k for k in cost if k not in _COST_ORDER]
    for key in ordered:
        rows.append(f'<tr><td>{_e(_COST_LABELS.get(key, key))}</td>'
                    f'<td>{_e(_cost_value(cost[key]))}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="2">包里没有代价块。</td></tr>')
    # 包外的东西：裁剪留下的痕迹也算代价（"少了什么"必须可查）。
    more = packet.get("verdicts_more") or {}
    if more:
        rows.append(f'<tr><td>判词超上限被裁</td><td>{_e(more.get("count"))} 条'
                    f'（{_e("、".join(str(i) for i in (more.get("ids") or [])))}）</td></tr>')
    omitted = packet.get("refusals_omitted") or {}
    if omitted:
        rows.append(f'<tr><td>拒绝条目被省</td><td>{_e(omitted.get("count"))} 条'
                    f'（{_e("、".join(str(c) for c in (omitted.get("codes") or [])))}）</td></tr>')
    un_more = packet.get("unmeasured_more") or {}
    if un_more:
        rows.append(f'<tr><td>未测组被合并</td><td>{_e(un_more.get("groups"))} 组 / '
                    f'{_e(un_more.get("metrics"))} 个指标</td></tr>')
    body = _table(("项目", "值"), rows,
                  "代价也是结论的一部分：这次花了多少 wall / 多少字节 / 少了什么。")
    return '<section id="cost">\n<h2>本次代价</h2>\n' + body + "\n</section>"


def _footer_html(packet: dict[str, Any]) -> str:
    version = packet.get("tool_version") or "?"
    return (f'<footer>由 kernoscope {_e(version)} 渲染 · 单文件 / 离线 / 无 JavaScript / '
            '无外部资源 · 渲染是纯函数：同一份包两次输出字节相同（不写入时间戳）。</footer>')


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def render_html(packet: dict[str, Any], *, title: str = "",
                onepage_text: str = "") -> str:
    """把决策包渲染成一个自包含 HTML 文档（纯函数：不改 packet、不读环境、不写时间）。

    `title` 为空时用包里的工具版本生成标题；`onepage_text` 非空时**原样**嵌进
    「结论」段（一行结论取它的 `结论：` 行），否则由包派生。
    """
    if not isinstance(packet, dict):
        raise TypeError(f"packet 必须是 dict，得到 {type(packet).__name__}")
    index = refusal_index(packet)
    doc_title = title.strip() or f"Kernoscope 证据单 {packet.get('tool_version') or ''}".strip()
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(doc_title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        _header_html(packet, doc_title),
        _conclusion_html(packet, onepage_text),
        _legend_html(),
        _verdicts_html(packet, index),
        _refusals_html(packet, index),
        _unmeasured_html(packet, index),
        _cost_html(packet),
        _footer_html(packet),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m kernoscope.html_view <packet.json> > out.html` 的入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m kernoscope.html_view",
        description="把决策包（或完整报告）渲染成一个自包含 HTML 文件。")
    parser.add_argument("packet", help="决策包 JSON；也接受完整报告（*.ksreport.json）；"
                                       "用 - 从 stdin 读")
    parser.add_argument("-o", "--out", help="输出文件（默认写 stdout）")
    parser.add_argument("--title", default="", help="页面标题（默认由包生成）")
    parser.add_argument("--onepage", default="",
                        help="把一屏文本原样嵌进「结论」段（文件路径）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.packet == "-":
        data = json.loads(sys.stdin.read() or "{}")
        packet = data
    else:
        packet = load_packet(args.packet)
    onepage = ""
    if args.onepage:
        with open(args.onepage, "r", encoding="utf-8") as handle:
            onepage = handle.read()
    text = render_html(packet, title=args.title, onepage_text=onepage)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROVENANCE_LABELS", "PROVENANCE_ORDER", "SIZE_BUDGET_BYTES", "main", "render_html"]
