#!/usr/bin/env python3
"""开发侧索引的**单一真源**：改哪类东西 → 改哪个文件 → 要重跑什么 → 跑哪条门禁。

## 为什么这份索引是生成物

它第一次是手写的（6 类适配器）。之后仓库多出三类单一真源（判词/决策包、评测材料、
agent 入口层），索引没跟上；"提交前跑"里也漏了 `gen_agent_refs.py --check` 与
`sanitize_artifacts.py --check`。后果是可复现的：决策包升到 schema v1.1 之后，
`eval/tasks/*.json` 这份生成物没人重跑，`make_tasks.py --check` **红在仓库里**，
而没有任何测试会发现它 —— 因为新加的那类改动不在索引里，也就没人知道要跑那条门禁。

手抄的索引会漂。所以这里的每一条都是数据，表格由数据渲染：

* 列出的每个路径、测试、生成器、章节都会被 `--check` 校验**真的存在**；
* 章节的字节数**现算**（手抄的价格和文档里的实测数字一样，一定会漂）；
* 逆向校验：凡是支持 `--check` 的脚本，必须出现在某类改动的「重跑」里或 `GATES` 里 ——
  这条正是上面那次漂移的对应门禁（没有它，"有生成器但没人跑"会再次发生）。

用法：
    python tools/dev_index.py             # 把索引表写进 CONTRIBUTING.md 的标记块
    python tools/dev_index.py --check     # 只校验（不一致则 exit 1）
    python tools/dev_index.py --brief     # 给 agent 的入口层（≤1.5 KB，带价格）
    python tools/dev_index.py --gates     # 提交前：全部门禁一把跑完
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

BEGIN = "<!-- dev-index:begin -->"
END = "<!-- dev-index:end -->"


# --------------------------------------------------------------------------
# 门禁：闭集。cost_s 是在开发机上实测的墙钟秒数（和 remedies 里的 cost_s 一个性质：
# 声明的实测值，用来让 agent 能规划，不是精确基准）。
# --------------------------------------------------------------------------
class Gate:
    def __init__(self, key: str, cmd: str, cost_s: float, what: str):
        self.key, self.cmd, self.cost_s, self.what = key, cmd, cost_s, what


GATES = [
    Gate("pytest", "uv run pytest", 12.7, "全量测试"),
    Gate("coverage", "uv run python tools/gen_coverage.py --check", 0.2,
         "COVERAGE.md 与 capabilities 同源"),
    Gate("agentrefs", "uv run python tools/gen_agent_refs.py --check", 0.2,
         "references/*.md 与拒绝码/判词同源"),
    Gate("playbook", "uv run python tools/gen_playbook.py --check", 0.2,
         "references/diagnosis-playbook.md 与拒绝码/路由/能力层同源"),
    Gate("index", "uv run python tools/dev_index.py --check", 0.1,
         "本索引与仓库现状同源"),
    Gate("tasks", "uv run python eval/make_tasks.py --check", 0.1,
         "评测任务材料与 packet.build() 同源"),
    Gate("calib", "uv run python tools/gen_calibration_excerpt.py --check", 0.1,
         "CALIBRATION.md 的产物节选与最新产物同源"),
    Gate("sanitize", "uv run python bench/sanitize_artifacts.py bench/artifacts --check", 0.2,
         "入库产物已脱敏"),
    Gate("release", "uv run python tools/release_bump.py --check", 1.0,
         "四处版本号自洽（pyproject / __version__ / README / CHANGELOG）"),
    Gate("build", "uv build", 1.1, "sdist + wheel 能生成"),
]
GATE_BY_KEY = {g.key: g for g in GATES}


# --------------------------------------------------------------------------
# 索引：数据。字段含义见表格表头。
#   truth  单一真源（先改这里；改别处会绕过校验）
#   touch  连带要动的文件
#   regen  生成物 → 跑这个脚本重跑
#   gates  除 pytest 外还要跑的门禁（key）
#   tests  守它的测试
#   sec    CONTRIBUTING 里的章节号（"" = 无专章，按真源与测试来）
# --------------------------------------------------------------------------
class Change:
    def __init__(self, key, ask, truth, touch=(), regen=(), gates=(), tests=(), sec=""):
        self.key, self.ask, self.truth = key, ask, list(truth)
        self.touch, self.regen, self.gates, self.tests = list(touch), list(regen), list(gates), list(tests)
        self.sec = sec


CHANGES = [
    Change("metric", "加/改一个指标（数字 + 出处）",
           truth=["src/kernoscope/provenance.py"],
           touch=["src/kernoscope/sections.py"],
           tests=["tests/test_provenance.py", "tests/test_output_consistency.py"], sec="①"),
    Change("backend", "加一族数据源（nsys / NVML / cuobjdump / NVBit）",
           truth=["src/kernoscope/backends"],
           touch=["src/kernoscope/doctor.py", "src/kernoscope/capabilities.py"],
           tests=["tests/test_doctor.py", "tests/test_toolchain.py",
                  # 第五轮：窗口的环境证据（降频位图/时钟驻留/同卡其他租户）也属于数据源面
                  "tests/test_environment_evidence.py"], sec="②"),
    Change("workload", "加一个已知答案负载（selftest 那一族）",
           truth=["src/kernoscope/selftest.py"],
           tests=["tests/test_diff_selftest.py"], sec="③"),
    Change("question", "加/改一个能力问题",
           truth=["src/kernoscope/capabilities.py"],
           # D30④：评测任务材料里嵌了 `capabilities`（`build(rep, capabilities=cap, ...)`），
           # 所以改能力问题也必须重跑 make_tasks，否则材料与 capabilities 不同源。
           regen=["tools/gen_coverage.py", "eval/make_tasks.py"],
           tests=["tests/test_capabilities.py", "tests/test_opening_package.py"], sec="④"),
    Change("refusal", "加一种拒绝理由（我们不给数）",
           truth=["src/kernoscope/refusals.py"],
           regen=["tools/gen_agent_refs.py"],
           touch=["CHANGELOG.md"],
           tests=["tests/test_refusals.py", "tests/test_agent_surface.py",
                  # 第五轮 V-13：处方命令必须交给真解析器（28 条 kernoscope 命令 + AST 超集扫）
                  "tests/test_remedy_commands.py"], sec="⑤"),
    Change("artifact", "加一次 bench 产物 / 改公开数字",
           truth=["bench/run.sh"],
           touch=["bench/artifacts/README.md"],
           regen=["tools/gen_calibration_excerpt.py"],
           gates=["sanitize"],
           tests=["tests/test_doc_artifact_consistency.py", "tests/test_repo_hygiene.py"], sec="⑥"),
    Change("verdict", "加/改一个判词或决策包字段",
           truth=["src/kernoscope/packet.py"],
           regen=["tools/gen_agent_refs.py", "eval/make_tasks.py"],
           touch=["CHANGELOG.md"],
           tests=["tests/test_packet.py", "tests/test_agent_surface.py",
                  "tests/test_eval_harness.py"], sec="⑧"),
    Change("entry", "改 agent 入口层（常驻那部分）",
           truth=["AGENTS.md"],
           touch=["llms.txt", "skills/kernoscope/SKILL.md"],
           tests=["tests/test_agent_surface.py"], sec="⑧"),
    Change("nongoal", "加/改一条「不做」或它的前提",
           truth=["NON-GOALS.md"],
           touch=["README.md"],
           tests=["tests/test_non_goals.py"], sec="①"),
    Change("cli", "加/改一个子命令或开关",
           truth=["src/kernoscope/cli.py"],
           touch=["README.md", "CHANGELOG.md"],
           tests=["tests/test_cli.py", "tests/test_flag_suggestions.py",
                  "tests/test_documented_commands.py"], sec="⑧"),
    Change("index", "加一条门禁 / 改这份索引本身",
           truth=["tools/dev_index.py"],
           touch=["tests/test_dev_index.py"],
           tests=["tests/test_dev_index.py"], sec="⑧"),
]


# --------------------------------------------------------------------------
# 章节字节数：现算，不手抄
# --------------------------------------------------------------------------
def _sections() -> dict[str, int]:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    lines = text.splitlines()
    heads = [(i, m.group(1)) for i, l in enumerate(lines)
             if (m := re.match(r"^##\s+(.)", l))]
    out: dict[str, int] = {}
    for n, (i, mark) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        out[mark] = len("\n".join(lines[i:end]).encode("utf-8"))
    return out


def _fmt_bytes(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def render_block() -> str:
    secs = _sections()
    out = [BEGIN,
           "<!-- 由 tools/dev_index.py 生成；改动请改那里的 CHANGES 后重跑，不要手改本表 -->",
           "",
           "| 你要改什么 | 单一真源（先改这里） | 连带 | 生成物（重跑）/ 额外门禁 | 守它的测试 | 先读 |",
           "|---|---|---|---|---|---|"]
    for c in CHANGES:
        truth = " / ".join(f"`{p}`" for p in c.truth)
        touch = " / ".join(f"`{p}`" for p in c.touch) or "—"
        extra = " / ".join(f"`{p}`" for p in c.regen)
        if c.gates:
            extra = f"{extra}；" if extra else ""
            extra += "门禁 " + "、".join(f"`{g}`" for g in c.gates)
        extra = extra or "—"
        tests = " / ".join(f"`{p}`" for p in c.tests)
        if c.sec and c.sec in secs:
            read = f"§{c.sec}（{_fmt_bytes(secs[c.sec])}）"
        else:
            read = "—"
        out.append(f"| {c.ask} | {truth} | {touch} | {extra} | {tests} | {read} |")
    out += ["", "提交前跑一条命令即可（下面的表就是它依次执行的东西）：", "",
            "```bash", "uv run python tools/dev_index.py --gates", "```", "",
            "| 门禁 | 命令 | 代价 | 守什么 |", "|---|---|---|---|"]
    for g in GATES:
        out.append(f"| `{g.key}` | `{g.cmd}` | ≈{g.cost_s:g} s | {g.what} |")
    out += ["", END]
    return "\n".join(out)


def render_brief() -> str:
    secs = _sections()
    total = sum(g.cost_s for g in GATES)
    out = [f"Kernoscope 开发索引：{len(CHANGES)} 类改动 · 全文见 CONTRIBUTING.md（首屏是同一张表）",
           "改什么 → 单一真源 [→ 重跑] [→ 门禁] | 章节"]
    for c in CHANGES:
        regen = " →" + ",".join(Path(p).name for p in c.regen) if c.regen else ""
        gate = " !" + ",".join(c.gates) if c.gates else ""
        sec = f" | §{c.sec} {_fmt_bytes(secs[c.sec])}" if c.sec in secs else ""
        out.append(f"- {c.ask} → {', '.join(c.truth)}{regen}{gate}{sec}")
    out.append(f"提交前：uv run python tools/dev_index.py --gates"
               f"（{len(GATES)} 条门禁一把跑，≈{total:g} s，全绿才算完）")
    out.append("生成物不许手改；测试比文档权威，冲突以测试为准。")
    return "\n".join(out)


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def check() -> list[str]:
    problems: list[str] = []
    secs = _sections()

    for c in CHANGES:
        for p in dict.fromkeys(c.truth + c.touch + c.regen + c.tests):
            if not (ROOT / p).exists():
                problems.append(f"{c.key}: 索引里的路径不存在：{p}")
        if c.sec and c.sec not in secs:
            problems.append(f"{c.key}: CONTRIBUTING 里没有章节 §{c.sec}")
        if not c.tests:
            problems.append(f"{c.key}: 没有任何测试守它（那它就只靠自觉）")
        for k in c.gates:
            if k not in GATE_BY_KEY:
                problems.append(f"{c.key}: 引用了未登记的门禁 {k}")
    if not any("pytest" == g.key for g in GATES):
        problems.append("门禁表里必须有全量测试")

    # 逆向：支持 --check 的脚本必须被**三种接线之一**用到 —— 否则它就会像
    # eval/make_tasks.py 那样"带 --check 但从来没人在 CI/提交前跑"，静默漂移：
    #   ① 某类改动的「重跑」生成器；② 某条门禁的命令；③ bench 采集脚本调用的验收工具
    #   （bench/run.sh 本身是产生产物的那一步，它的验收判据进 manifest 的 rc）。
    referenced = {p for c in CHANGES for p in c.regen}
    referenced |= {re.sub(r"^uv run python ", "", g.cmd).split()[0] for g in GATES}
    bench_sh = (ROOT / "bench" / "run.sh")
    bench_text = bench_sh.read_text(encoding="utf-8") if bench_sh.exists() else ""
    for pat in ("tools/*.py", "eval/*.py", "bench/*.py"):
        for p in sorted(ROOT.glob(pat)):
            if p.name == "__init__.py" or "--check" not in p.read_text(encoding="utf-8"):
                continue
            rel = str(p.relative_to(ROOT))
            if rel not in referenced and rel not in bench_text:
                problems.append(
                    f"{rel}: 支持 --check 但没有任何门禁/改动类/bench 步骤会跑它")

    text = CONTRIBUTING.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        problems.append("CONTRIBUTING.md 里没有 dev-index 标记块（跑 tools/dev_index.py 写入）")
    else:
        have = text[text.index(BEGIN):text.index(END) + len(END)]
        if have != render_block():
            problems.append("CONTRIBUTING.md 的索引块与生成器不一致（重跑 tools/dev_index.py）")
    return problems


def write_block() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    new = text[:text.index(BEGIN)] + render_block() + text[text.index(END) + len(END):]
    CONTRIBUTING.write_text(new, encoding="utf-8")
    print(f"已写入 {CONTRIBUTING.relative_to(ROOT)} 的索引块（{len(render_block().encode())} B）")


def _gate_cmd(cmd: str, *, python: str | None = None) -> str:
    """把门禁命令里的 `uv run` 换成给定解释器（**只在没有 uv 时**用）。

    为什么（第七轮 X-2）：10 条门禁全部硬编码 `uv run …`，而**租来的 GPU 机器**（正是 NVBit 冒烟、
    真 nsys/ncu 用例唯一有分辨力的环境）通常没有 `uv` —— 那时 `--gates` 是 0/9，等于"提交前跑一遍
    门禁"这条工作流在目标机器上不可用。CI 仍然用 `uv run`（不变），这里的回退只影响本机调用。
    `uv build` 没有解释器等价物，回退时改成 `python -m build`（缺 build 就如实失败）。
    """
    if python is None or not cmd.startswith("uv "):
        return cmd
    rest = cmd[len("uv "):]
    if rest.startswith("run "):
        rest = rest[len("run "):]
        if rest.startswith("python "):
            # `uv run python tools/x.py --check` → `<py> tools/x.py --check`
            return f"{shlex_quote(python)} {rest[len('python '):]}"
        # `uv run pytest` → `<py> -m pytest`（pytest 是个模块，不是解释器参数）
        return f"{shlex_quote(python)} -m {rest}"
    if rest.startswith("build"):
        return f"{shlex_quote(python)} -m build" + rest[len("build"):]
    return cmd


def _runner_python(explicit: str | None) -> str | None:
    """显式 `--python` > 有 `uv` 时用 uv 原样 > 回退当前解释器。"""
    if explicit:
        return explicit
    from shutil import which

    return None if which("uv") else sys.executable


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def run_gates(python: str | None = None) -> int:
    failed: list[str] = []
    total = 0.0
    interp = _runner_python(python)
    if interp is not None:
        print(f"note: 没有 uv（或用 --python 指定）⇒ 门禁改用 {interp}"
              f"（CI 仍用 `uv run`；8 条纯 Python 门禁等价，`build` 走 `python -m build`"
              f"—— 缺该模块时先 `pip install build` 或装 uv）")
    for g in GATES:
        import time
        t0 = time.monotonic()
        cmd = _gate_cmd(g.cmd, python=interp)
        # **不注入 PYTHONPATH**：各生成脚本自己会 `sys.path.insert(0, ROOT/src)`，
        # pytest 走 pyproject 的 `pythonpath=["src"]`；而外面多一层 PYTHONPATH=<repo>/src
        # 会让 `uv build`（preflight 里那一步）与 setup-check 的行为变样 —— 实测这两条会红。
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
        dt = time.monotonic() - t0
        total += dt
        mark = "ok  " if r.returncode == 0 else "FAIL"
        print(f"[{mark}] {g.key:<10} {dt:5.1f}s  {g.what}")
        if r.returncode != 0:
            failed.append(g.key)
            print((r.stdout + r.stderr).strip()[-1200:])
    print(f"\n{len(GATES) - len(failed)}/{len(GATES)} 通过，合计 {total:.1f} s")
    if failed:
        print("失败：" + ", ".join(failed))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="开发侧索引的单一真源（改哪类东西 → 改哪 → 跑什么）")
    ap.add_argument("--check", action="store_true", help="只校验索引与仓库现状是否一致")
    ap.add_argument("--brief", action="store_true", help="打印给 agent 的入口层（≤1.5 KB）")
    ap.add_argument("--gates", action="store_true", help="提交前：全部门禁一把跑完")
    ap.add_argument("--python", default=None, metavar="EXE",
                    help="跑门禁用的解释器（默认：有 uv 用 `uv run`；没有 uv 时回退到当前解释器。"
                         "租来的 GPU 机器通常没有 uv，用这个或直接不给）")
    args = ap.parse_args()

    if args.brief:
        text = render_brief()
        sys.stdout.write(text + "\n")
        if len(text.encode("utf-8")) > 1500:
            print(f"[dev_index] 入口层已 {len(text.encode('utf-8'))} B，超过 1500 B 预算",
                  file=sys.stderr)
            return 1
        return 0
    if args.gates:
        return run_gates(args.python)
    if args.check:
        problems = check()
        for p in problems:
            print(p, file=sys.stderr)
        if problems:
            print(f"索引与仓库现状不一致（{len(problems)} 处）", file=sys.stderr)
            return 1
        print(f"索引与仓库现状一致 ✓（{len(CHANGES)} 类改动、{len(GATES)} 条门禁）")
        return 0
    write_block()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
