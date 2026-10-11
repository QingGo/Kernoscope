"""`html_view.render_html` 的契约测试（N9：单文件 HTML 渲染器）。

为什么这些断言值得写：HTML 是**给人看的传播形态**（截图会走进 issue 与教程），所以它比
终端更容易"看起来对"。四条最容易悄悄坏掉的契约在这里变成可执行的检查：

  1. **自包含**：没有 `http://` / `https://` / `<link>` / `<script src>`，也没有任何 JS ——
     一个需要联网或需要脚本才能读的文件，离线打开就是一张白纸；
  2. **出处和数字同时出现**：四种出处各有角标与颜色 + 图例；`unavailable` 印 `n/a` + code，
     **绝不印 0**（0 会被读成一个读数，这正是本工具存在的理由）；
  3. **转义**：包里的字符串（kernel 名、remedy、例子）都当文本，不当标记；
  4. **尺寸预算**：正常包（≤4 KB）渲染 ≤ 64 KB，且结尾就是 `</html>`（没有尾巴上的外链）。

fixture 分两种：一种用**真实的 `packet.build`**（形状由代码定义，不靠手搓），一种手写
最小包（精确构造边界：refused 无 value、前提 ok=False、文本里带 `<script>`）。
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import sys
from html.parser import HTMLParser

import pytest

from kernoscope.html_view import (PROVENANCE_LABELS, PROVENANCE_ORDER, SIZE_BUDGET_BYTES,
                                  render_html)
from kernoscope.packet import build
from kernoscope.provenance import Metric, Section
from kernoscope.refusals import RefusalCode
from kernoscope.report import DeviceInfo, KernelLaunch, KernelReport, Report

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_PACKETS = sorted((ROOT / "bench" / "artifacts").glob("*/*.packet.json"))


# --------------------------------------------------------------------------
# fixture ①：真实的 build() 产物（四个出处齐全 + 一个 unavailable）
# --------------------------------------------------------------------------
def _launch(i: int, dur_ns: int = 1_000_000) -> KernelLaunch:
    return KernelLaunch(name="big_gemm", demangled="big_gemm(float const*)",
                        start_ns=i * 2_000_000, end_ns=i * 2_000_000 + dur_ns,
                        grid=(1024, 1, 1), block=(256, 1, 1), registers_per_thread=64,
                        static_smem_bytes=0, dynamic_smem_bytes=49152,
                        local_mem_per_thread=0, stream_id=0, device_id=0, grid_id=0,
                        correlation_id=i)


def _report() -> Report:
    """一份小报告，但四个出处都真的会被判词用到。

    exact    `gpu__time_duration.sum` → kernel_time
    measured `meas__achieved_occupancy_pct` → occupancy
    estimated `est__dram_bytes_total` → dram_bytes
    derived  `ks__cost_per_mtok` → cost_per_token
    """
    rep = Report(device=DeviceInfo(name="RTX 4090", compute_capability="8.9", sm_count=128))
    rep.wall_time_ns = 20_000_000_000
    rep.meta["window"] = "mode=auto 采样 120 点"
    kernel = KernelReport("big_gemm", launch_count=50,
                          launches=[_launch(i) for i in range(50)])
    section = Section("All")
    section.add(Metric.exact("gpu__time_duration.sum", 150.0, "ms"))
    section.add(Metric.derived("ks__theoretical_occupancy_pct", 50.0, "%"))
    section.add(Metric.measured("meas__achieved_occupancy_pct", 46.9, "%",
                                note="区间重叠法重建"))
    section.add(Metric.estimated("est__dram_bytes_total", 5.0e9, "B"))
    section.add(Metric.derived("ks__sampler_window_coverage_pct", 97.1, "%"))
    section.add(Metric.derived("ks__dram_window_mem_busy_mean_pct", 96.8, "%"))
    section.add(Metric.derived("ks__cost_per_mtok", 3.2, "CNY/Mtoken"))
    section.add(Metric.measured("meas__window_energy_j", 1200.0, "J"))
    # unavailable：一个进「没测到什么」（计数器），一个让判词变成 refused（没有轨迹）
    section.add(Metric.unavailable("sm__dram_bytes_read.sum", "计数器拿不到",
                                   code=RefusalCode.NO_COUNTERS))
    section.add(Metric.unavailable("est__dram_bytes_per_launch", "没有访存轨迹",
                                   code=RefusalCode.INSTRUMENTATION_MISSING))
    kernel.sections = [section]
    rep.kernels = [kernel]
    return rep


def _built_packet() -> dict:
    return build(_report())


# --------------------------------------------------------------------------
# fixture ②：手写最小包（精确构造边界）
# --------------------------------------------------------------------------
def _minimal_packet(**overrides) -> dict:
    packet = {
        "schema_version": 1,
        "schema_revision": "1.2",
        "tool": "kernoscope",
        "tool_version": "0.2.0",
        "protocol": {"hash": "sha256:0123456789abcdef", "basis": "per-launch"},
        "env": {"tier": 3, "tier_name": "instrumented", "counter_state": "denied"},
        "verdicts": [],
        "unmeasured": [],
        "refusals": {},
        "cost": {"wall_s": 12.0, "packet_bytes": 900, "launches": 50},
    }
    packet.update(overrides)
    return packet


def _row(html: str, verdict_id: str) -> str:
    """取某条判词那一行（`<tr id="verdict-…">` 到 `</tr>`），供逐格断言。"""
    start = html.index(f'id="verdict-{verdict_id}"')
    return html[start:html.index("</tr>", start)]


def _section(html: str, section_id: str) -> str:
    start = html.index(f'id="{section_id}"')
    return html[start:html.index("</section>", start)]


# --------------------------------------------------------------------------
# 1) 自包含：零外链、零 JS
# --------------------------------------------------------------------------
def test_output_is_self_contained_and_has_no_javascript():
    html = render_html(_built_packet())
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html
    assert "<script src" not in html
    assert "<script" not in html, "人视图要能在关掉 JS 的浏览器里读"
    assert "<style>" in html, "CSS 必须内联（单文件离线可开）"


# --------------------------------------------------------------------------
# 2) 出处角标：四种颜色 + 图例 + unavailable 的 n/a 与 code
# --------------------------------------------------------------------------
def test_all_four_provenances_have_badges_and_a_legend():
    packet = _built_packet()
    provenances = {str(v.get("provenance") or "") for v in packet["verdicts"]}
    # fixture 自检：四种出处必须真的都出现在包里（否则下面的断言是空转）
    assert set(PROVENANCE_ORDER) <= provenances, f"fixture 缺出处：{sorted(provenances)}"
    html = render_html(packet)
    legend = _section(html, "provenance-legend")
    for provenance in PROVENANCE_ORDER:
        verdict = next(v for v in packet["verdicts"] if v.get("provenance") == provenance)
        row = _row(html, str(verdict["id"]))
        assert f'class="badge badge-{provenance}">{provenance}<' in row, \
            f"{verdict['id']} 的数字旁边没有 {provenance} 角标"
        assert f'class="badge badge-{provenance}"' in legend, "图例里缺这个出处"
        assert PROVENANCE_LABELS[provenance] in legend
    # 同一个 fixture 里的 unavailable：n/a + 拒绝码（角标是第五种颜色）
    refused = [v for v in packet["verdicts"] if v.get("status") == "refused"]
    assert refused, "fixture 必须包含一条 unavailable 的判词"
    code = str(refused[0]["refusal"]["code"])
    row = _row(html, str(refused[0]["id"]))
    assert '<span class="na">n/a</span>' in row
    assert code in row
    assert '<span class="badge badge-unavailable">unavailable</span>' in row
    # 未测到的 unavailable 指标（计数器）也带 code，且也在图例覆盖的语汇里
    assert any(g["code"] == "counters_denied" for g in packet["unmeasured"])


def test_unavailable_value_is_na_with_code_never_zero():
    packet = _minimal_packet(
        verdicts=[{
            "id": "dram_bytes", "status": "refused", "actionable": False,
            # build() 在前提出问题时会把历史值放进 facts —— 人视图不许把它印成一个读数
            "facts": {"uncitable_value": 0},
            "refusal": {"code": "counters_denied"},
        }],
        refusals={"counters_denied": {"action": "计数器在容器内拿不到，换一台机器",
                                      "cost_kind": "unavailable"}},
    )
    html = render_html(packet)
    row = _row(html, "dram_bytes")
    assert '<span class="na">n/a</span>' in row
    assert "counters_denied" in row, "n/a 旁边必须给拒绝码"
    assert '<span class="badge badge-unavailable">unavailable</span>' in row
    assert "0" not in row, "unavailable 的值不许渲染成 0"
    assert 'class="value">0' not in html
    assert "uncitable_value" not in html, "历史值（不可引用）不进人视图"


# --------------------------------------------------------------------------
# 3) 判词 / 前提 / 拒绝 / 代价都要在
# --------------------------------------------------------------------------
def test_premises_distinguish_ok_from_not_ok():
    packet = _minimal_packet(
        verdicts=[{
            "id": "dram_bytes_per_launch", "status": "refused", "actionable": False,
            "premises": [
                {"id": "window_coverage_pct", "observed": 94.6, "ok": True,
                 "threshold": 80.0, "unit": "%"},
                {"id": "matched_kernel_kernel_share_pct", "observed": 41.0, "ok": False,
                 "threshold": 95.0, "unit": "%"},
            ],
            "refusal": {"code": "share_too_low"},
        }],
        refusals={"share_too_low": {"action": "缩小窗口让被匹配 kernel 成为主角",
                                    "cost_kind": "wall_seconds", "cost_s": 20.0,
                                    "cmd": "kernoscope -k regex:big_gemm"}},
    )
    html = render_html(packet)
    section = _section(html, "verdicts")
    assert 'class="premise premise-ok"' in section
    assert 'class="premise premise-bad"' in section
    assert "→ <b>成立</b>" in section
    assert "→ <b>不成立</b>" in section
    assert "被匹配 kernel 是否占窗口 kernel 时间" in section, "前提必须用人话标签"


def test_refusals_render_code_remedy_and_cost():
    packet = _minimal_packet(
        verdicts=[{
            "id": "occupancy", "status": "refused", "actionable": False,
            "refusal": {"code": "no_occupancy_probe", "observed": {"probe": None},
                        "threshold": {"probe": 1}},
        }],
        refusals={"no_occupancy_probe": {
            "action": "跑 occupancy 探针并把 JSON 用 --ks-achieved-occupancy 传进来",
            "cost_kind": "wall_seconds", "cost_s": 600.0,
            "cmd": "kernoscope occupancy --from probe.json --write-json",
        }},
    )
    html = render_html(packet)
    section = _section(html, "refusals")
    assert "<code>no_occupancy_probe</code>" in section
    assert "没跑 occupancy 探针" in section, "code 要翻成人话"
    assert "跑 occupancy 探针" in section, "remedy 动作必须出现"
    assert "≈10min" in section, "代价要与 remedy 一起出现"
    assert "kernoscope occupancy" in section, "可执行的命令要出现"
    # 判词行里也要能看到 code（n/a 与 code 同格）
    assert "no_occupancy_probe" in _row(html, "occupancy")


def test_unmeasured_groups_and_cost_block_are_rendered():
    packet = _built_packet()
    html = render_html(packet)
    unmeasured = _section(html, "unmeasured")
    for group in packet["unmeasured"]:
        assert str(group["code"]) in unmeasured
    cost = _section(html, "cost")
    assert "wall 时间（s）" in cost
    assert "决策包大小（B）" in cost
    assert str(packet["cost"]["wall_s"]) in cost
    assert str(packet["cost"]["packet_bytes"]) in cost


# --------------------------------------------------------------------------
# 4) 转义
# --------------------------------------------------------------------------
def test_packet_text_is_html_escaped():
    hostile = '<script>alert(1)&</script>'
    packet = _minimal_packet(
        verdicts=[{
            "id": "kernel_time", "status": "ok", "actionable": True,
            "facts": {"top": hostile}, "value": 1.5, "unit": "ms", "provenance": "exact",
            "next": 'x < y & "z"',
        }],
        refusals={"counters_denied": {"action": "a < b & c",
                                      "cost_kind": "unavailable"}},
        unmeasured=[{"code": "counters_denied", "count": 2, "examples": ["<b>&x</b>"]}],
    )
    html = render_html(packet)
    assert hostile not in html
    assert "&lt;script&gt;alert(1)&amp;&lt;/script&gt;" in html
    assert "a &lt; b &amp; c" in html
    assert "&lt;b&gt;&amp;x&lt;/b&gt;" in html
    assert "<script" not in html


# --------------------------------------------------------------------------
# 5) 尺寸预算 + 结尾没有外链
# --------------------------------------------------------------------------
def test_normal_packet_is_within_size_budget():
    for packet in (_built_packet(), _minimal_packet()):
        html = render_html(packet)
        size = len(html.encode("utf-8"))
        assert size <= SIZE_BUDGET_BYTES, f"渲染出了 {size} B，超 {SIZE_BUDGET_BYTES} B 预算"
        assert html.rstrip().endswith("</html>")
        tail = html.rstrip()[-400:]
        for bad in ("http", "://", "<link", "src="):
            assert bad not in tail, f"文件尾部出现了外部引用痕迹：{bad}"


# --------------------------------------------------------------------------
# 6) 一屏文本透传 / 结论派生
# --------------------------------------------------------------------------
def test_onepage_text_is_passed_through_and_conclusion_is_derived():
    packet = _built_packet()
    # 不传：结论从包里派生（复用 render_onepage 的「结论：」行）
    derived = render_html(packet)
    assert "条判词被拒" in derived, "包里有 refused 时结论必须点出来"

    onepage = "Kernoscope 0.2.0 · 协议 0123\n结论：先解 instrumentation_missing\n尾部 & <b>"
    html = render_html(packet, onepage_text=onepage)
    assert "先解 instrumentation_missing" in html
    assert "尾部 &amp; &lt;b&gt;" in html, "一屏文本原样嵌入但不能当标记"
    assert '<pre class="onepage">' in html


def test_title_is_used_and_escaped():
    packet = _built_packet()
    html = render_html(packet, title='A/B <旧版> & "new"')
    assert "<title>A/B &lt;旧版&gt; &amp; &quot;new&quot;</title>" in html
    assert "<h1>A/B &lt;旧版&gt; &amp; &quot;new&quot;</h1>" in html


# --------------------------------------------------------------------------
# 7) 语义标记 / 纯函数
# --------------------------------------------------------------------------
def test_semantic_markup_for_no_js_readers():
    html = render_html(_built_packet())
    assert "<h2>结论</h2>" in html
    assert "<table>" in html
    assert "<caption>" in html
    assert '<th scope="col">' in html
    assert '<span class="badge' in html
    assert '<html lang="zh-CN">' in html


def test_render_is_pure_and_deterministic():
    packet = _built_packet()
    before = copy.deepcopy(packet)
    first = render_html(packet)
    second = render_html(packet)
    assert first == second, "纯函数：同一个包两次渲染必须字节相同"
    assert packet == before, "渲染不许改动调用方的包"


def test_output_is_well_formed_html():
    """结构必须真的能打开：标签配对、无游离闭合（浏览器容错太好了，靠肉眼看不出来）。"""
    void = {"meta", "br", "hr", "img", "input", "link"}

    class _Checker(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.stack: list[str] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if not self.stack:
                self.errors.append(f"多余的 </{tag}>")
            elif self.stack[-1] != tag:
                self.errors.append(f"</{tag}> 关掉的是 <{self.stack[-1]}>")
            else:
                self.stack.pop()

    for packet in (_built_packet(), _minimal_packet()):
        checker = _Checker()
        checker.feed(render_html(packet))
        assert not checker.errors, checker.errors
        assert not checker.stack, f"标签没有闭合：{checker.stack}"


# --------------------------------------------------------------------------
# 8) `python -m kernoscope.html_view` 入口
# --------------------------------------------------------------------------
def test_module_main_renders_a_packet_file(tmp_path):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_minimal_packet(
        verdicts=[{"id": "kernel_time", "status": "ok", "actionable": True,
                   "value": 1.5, "unit": "ms", "provenance": "exact"}]),
        ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "kernoscope.html_view", str(packet_path)],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith("</html>")
    assert "Kernoscope 证据单" in proc.stdout

    out_path = tmp_path / "out.html"
    proc = subprocess.run([sys.executable, "-m", "kernoscope.html_view", str(packet_path),
                           "-o", str(out_path)], capture_output=True, text=True, env=env,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert out_path.read_text(encoding="utf-8").rstrip().endswith("</html>")


# --------------------------------------------------------------------------
# 9) 真机产物：真实形状（拒绝密集、facts 被裁）也要能渲染
# --------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT_PACKETS, reason="bench/artifacts 里没有入库的决策包")
def test_real_repo_packet_renders_offline():
    html = render_html(json.loads(ARTIFACT_PACKETS[0].read_text(encoding="utf-8")))
    assert html.rstrip().endswith("</html>")
    assert len(html.encode("utf-8")) <= SIZE_BUDGET_BYTES
    assert "http://" not in html and "https://" not in html
    assert '<span class="na">n/a</span>' in html, "真机包里有 refused，必须印 n/a"
    assert 'class="badge badge-unavailable"' in html
