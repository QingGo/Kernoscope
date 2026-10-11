"""V-13：`refusals.py` 里每条 `remedy.command` 都必须能被**真实 CLI 解析器**接受。

第五轮验收报告 V-13（P1）：`no_occupancy_probe` 的处方命令写成了
`kernoscope occupancy --records <probe-records> -o occ.json` —— 真实的 `occupancy`
子命令只认 `--from` / `--write-json`，这条命令照抄一定失败。它从 `refusals.py`
渗进了生成文档（`refusal-codes.md` / `diagnosis-playbook.md`）、评测材料
（`eval/tasks/*.json`）与入库产物（`capabilities.txt`），而**全仓没有任何测试**
把 `remedy.command` 交给真解析器。这条测试补上那个缺口。

纪律（防止"跳过一片"式的自我满足）：

* 所有命令都被分类并**计数**（kernoscope / 非 kernoscope），`ks + other == total`；
* 提到 `kernoscope` 却没被认成 kernoscope 命令的（包装前缀没认全）算**问题**，不许静默跳过；
* kernoscope 命令数低于下限 ⇒ 测试红（扫不到东西不等于全绿）；
* 只 `parse_args`，绝不 `main()`：这些是副作用命令（会真的采集/跑负载）。

登记新 remedy 命令时不需要改这里：扫描是**全量**的，从 `DEFAULT_REMEDIES` 直接取。
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import shlex
from dataclasses import replace
from pathlib import Path

from kernoscope import refusals as R
import kernoscope.cli as cli

# 文档/处方里的 `<真值>` 占位符归一成它（读者要替换成自己的路径，不是开关写错）。
PLACEHOLDER = "PLACEHOLDER"
ANGLE = re.compile(r"<[^<>]*>")
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# 验收报告独立扫出 28 条 kernoscope 命令；下限留出余量，但必须能抓住"跳过一片"。
MIN_KERNSCOPE_COMMANDS = 25

# 这些 code 的处方命令必须真的在扫描集合里（它们是读者最常照抄的几条）。
MUST_BE_SCANNED = ("no_occupancy_probe", "no_memtrace", "no_instr_hist", "nsys_missing")


# --------------------------------------------------------------------------
# 取命令 + 分类
# --------------------------------------------------------------------------
def remedy_commands(remedies=None) -> list[tuple[str, str]]:
    """(code, command) —— 全量取 `DEFAULT_REMEDIES` 里所有非空 command。"""
    rems = R.DEFAULT_REMEDIES if remedies is None else remedies
    out: list[tuple[str, str]] = []
    for code, rem in rems.items():
        cmd = (getattr(rem, "command", "") or "").strip()
        if cmd:
            out.append((getattr(code, "value", str(code)), cmd))
    return out


def kernoscope_argv(command: str, *, allow_wrappers: bool = True) -> list[str] | None:
    """命令如果是在调用 kernoscope，返回它的 argv（不含程序名）；否则 None。

    处理（与 `test_documented_commands` 的取词口径一致，但按 token 走）：
    `VAR=value` 前缀、`sudo`/`time`/`nohup`/`command`/`exec`、`uv run`、`python -m`。
    真事：`PYTHONPATH=<nsight-compute>/extras/python kernoscope --ks-ncu-rep …`
    这条命令如果只认"首词是 kernoscope"，就会被静默跳过 —— 正是要防的那种跳过。
    """
    try:
        toks = shlex.split(command, comments=True)
    except ValueError:
        return None
    while toks:
        if ENV_ASSIGN.match(toks[0]):
            toks.pop(0)
            continue
        if allow_wrappers and toks[0] in ("sudo", "time", "nohup", "command", "exec"):
            toks.pop(0)
            continue
        if allow_wrappers and toks[0] == "uv" and len(toks) > 1 and toks[1] == "run":
            toks = toks[2:]
            continue
        if allow_wrappers and toks[0] in ("python", "python3") and len(toks) > 1 \
                and toks[1] == "-m":
            toks = toks[2:]
            continue
        break
    if not toks or toks[0] != "kernoscope":
        return None
    argv = [ANGLE.sub(PLACEHOLDER, t) for t in toks[1:]]
    if "--" in argv:                       # `-- <被测命令>` 是别人在跑，不属于我们解析
        argv = argv[:argv.index("--")]
    return argv


def mentions_kernoscope(command: str) -> bool:
    try:
        return "kernoscope" in shlex.split(command, comments=True)
    except ValueError:
        return False


def parse_problem(argv: list[str]) -> str | None:
    """把 argv 交给**真实解析器**：None = 合法。子命令走它自己的解析器。

    只解析、不执行：`--help` 之外的 SystemExit 非 0 就是被拒。
    """
    if not argv:
        return "空命令"
    sub = argv[0] if not argv[0].startswith("-") else ""
    if sub and sub not in cli.SUBCOMMANDS:
        return f"不认识子命令 `{sub}`（只有 {', '.join(cli.SUBCOMMANDS)}）"
    parser = cli.subcommand_parser(sub) if sub else cli.build_parser()
    rest = argv[1:] if sub else argv
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parser.parse_args(rest)
    except SystemExit as e:
        if e.code in (0, None):
            return None
        tail = err.getvalue().strip().splitlines()
        detail = tail[-1] if tail else f"exit {e.code}"
        return f"{'子命令' if sub else '主'}解析器拒绝（exit {e.code}）：{detail}"
    return None


def scan_commands(pairs) -> tuple[int, int, list[str]]:
    """(标签, 命令) 列表 → (kernoscope 命令数, 非 kernoscope 命令数, 问题列表)。"""
    ks = other = 0
    problems: list[str] = []
    for label, cmd in pairs:
        argv = kernoscope_argv(cmd)
        if argv is None:
            if mentions_kernoscope(cmd):
                problems.append(f"[{label}] 提到了 kernoscope 却没被认成 kernoscope 命令"
                                f"（包装前缀没认全，命令因此永远不会被解析）：{cmd}")
            else:
                other += 1
            continue
        ks += 1
        why = parse_problem(argv)
        if why:
            problems.append(f"[{label}] {why}\n      命令：{cmd}")
    return ks, other, problems


def scan(remedies=None) -> tuple[int, int, list[str]]:
    """`DEFAULT_REMEDIES` 的全量扫描。"""
    return scan_commands(remedy_commands(remedies))


def kernoscope_remedy_codes(remedies=None) -> set[str]:
    """被**真的当成 kernoscope 命令**扫描到的 code 集合（不是"命令非空"的 code 集合）。"""
    return {code for code, cmd in remedy_commands(remedies)
            if kernoscope_argv(cmd) is not None}


def guard_problems(ks: int, other: int, problems: list[str], total: int) -> list[str]:
    """主测试的守卫：问题 + 扫描退化 + 计不上账，三类都算红。"""
    out = list(problems)
    if ks < MIN_KERNSCOPE_COMMANDS:
        out.append(f"只扫到 {ks} 条 kernoscope remedy 命令（非 kernoscope {other} 条，"
                   f"共 {total} 条）：扫描范围退化 —— 扫不到东西不等于全绿")
    if ks + other != total:
        out.append(f"分类没计上账：kernoscope {ks} + 非 kernoscope {other} != 总数 {total}"
                   f"（有命令被静默丢掉了）")
    return out


# --------------------------------------------------------------------------
# 1) 静态扫描工具：`src/` 里**所有** `Remedy(command="…")` 字面量
#    （不只是 refusals.py：`analyze/dram.py`、`analyze/cost.py`、
#    `analyze/prescription.py` 里还有内联处方 —— 同一类缺陷在那边同样没人守）。
# --------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[1] / "src" / "kernoscope"

# 现在 refusals.py 的 28 条 kernoscope 命令 + 其他模块内联的几条；留出余量。
MIN_SRC_KERNSCOPE_COMMANDS = 30


def static_remedy_commands() -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """从 `src/` 的 AST 取所有 `Remedy(command=…)`（含函数内部的）。

    返回 (命令字面量, 透传项, 不可静态校验项)：

    * **字面量**逐个交给真解析器；
    * **透传**= 从别处读回一条已有命令（`str(rem.get("command", ""))`，即
      `Refusal.from_dict` 那条路）—— 命令的来源已在别处被守，这里只计数；
    * **不可静态校验**= f-string / 变量拼接出来的命令：静态门禁看不见，必须为空
      —— 那正是"跳过一片"的藏身处。
    """
    literals: list[tuple[str, str]] = []
    passthrough: list[str] = []
    unverifiable: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname != "Remedy":
                continue
            for kw in node.keywords:
                if kw.arg != "command":
                    continue
                where = f"{rel}:{kw.value.lineno}"
                try:
                    val = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    src = ast.unparse(kw.value)
                    (passthrough if "command" in src else unverifiable).append(
                        f"{where} = {src}")
                    continue
                if isinstance(val, str) and val.strip():
                    literals.append((where, val.strip()))
    return literals, passthrough, unverifiable


# --------------------------------------------------------------------------
# 2) 全量扫：`refusals.py` 的每条 kernoscope remedy 命令都必须过真解析器
# --------------------------------------------------------------------------
def test_every_kernoscope_remedy_command_is_parseable():
    """V-13 的主门禁：`refusals.py` 的 `remedy.command` 逐条交给真解析器。"""
    all_cmds = remedy_commands()
    ks, other, problems = scan()
    bad = guard_problems(ks, other, problems, len(all_cmds))
    assert not bad, ("refusals.py 里的 remedy 命令过不了真实 CLI 解析器"
                     "（读者会照抄失败）：\n  - " + "\n  - ".join(bad))
    codes = kernoscope_remedy_codes()
    missing = [c for c in MUST_BE_SCANNED if c not in codes]
    assert not missing, f"这些 code 的命令没被当成 kernoscope 命令扫到：{missing}"


def test_the_wrong_occupancy_command_is_gone():
    """点名那条：`occupancy` 子命令只认 `--from` / `--write-json`（不是 `--records` / `-o`）。"""
    cmd = R.DEFAULT_REMEDIES[R.RefusalCode.NO_OCCUPANCY_PROBE].command
    argv = kernoscope_argv(cmd)
    assert argv is not None and argv[0] == "occupancy", cmd
    assert parse_problem(argv) is None, cmd
    flags = {s for a in cli.subcommand_parser("occupancy")._actions
             for s in a.option_strings}
    used = {t.split("=")[0] for t in argv if t.startswith("-")}
    assert used <= flags, f"{cmd} 用了 occupancy 不认识的开关：{sorted(used - flags)}"
    assert "--records" not in used and "-o" not in used, \
        f"旧错命令又回来了（occupancy 没有这两个开关）：{cmd}"


def test_every_static_remedy_command_in_src_parses():
    """`src/` 里**所有**静态 `Remedy(command="…")` 都要过真解析器（含其他模块的内联处方）。

    这条是主门禁的超集：错处方不只藏在 `DEFAULT_REMEDIES` 里。透传的命令（从别处读回）
    单独计数；不可静态校验的（f-string / 变量拼接）必须为空，而不是被"跳过"。
    """
    pairs, passthrough, unverifiable = static_remedy_commands()
    ks, other, problems = scan_commands(pairs)
    total = len(pairs)
    bad = list(problems)
    if ks < MIN_SRC_KERNSCOPE_COMMANDS:
        bad.append(f"src/ 里只扫到 {ks} 条 kernoscope remedy 命令（非 kernoscope {other} 条，"
                   f"共 {total} 条，透传 {len(passthrough)} 条）：扫描退化")
    if ks + other != total:
        bad.append(f"分类没计上账：{ks} + {other} != {total}（有命令被静默丢掉了）")
    assert not bad, ("src/ 里的 remedy 命令过不了真实 CLI 解析器：\n  - " + "\n  - ".join(bad))
    assert not unverifiable, ("这些 remedy.command 静态门禁看不见（f-string/拼接 = 没被守）：\n  - "
                              + "\n  - ".join(unverifiable))


# --------------------------------------------------------------------------
# 3) 自我满足反例：检查器必须能红
# --------------------------------------------------------------------------
def _with_command(code: R.RefusalCode, command: str) -> dict:
    rems = dict(R.DEFAULT_REMEDIES)
    rems[code] = replace(rems[code], command=command)
    return rems


def test_checker_catches_the_old_bad_command_by_code_and_command():
    """把那条命令改错 → 必须红，并且报文里点名 code + 命令（不是只有一句"不一致"）。"""
    bad_cmd = "kernoscope occupancy --records <probe-records> -o occ.json"
    rems = _with_command(R.RefusalCode.NO_OCCUPANCY_PROBE, bad_cmd)
    ks, other, problems = scan(rems)
    bad = guard_problems(ks, other, problems, len(remedy_commands(rems)))
    assert bad, "已知的错命令被放行了 —— 这个检查器没有价值"
    joined = "\n".join(bad)
    assert "no_occupancy_probe" in joined, joined
    assert "--records" in joined, joined


def test_checker_does_not_pass_silently_when_kernoscope_commands_vanish():
    """把全部 remedy 换成非 kernoscope 命令 → 计数守卫必须红（不许静默全绿）。"""
    rems = {code: replace(rem, command="pip install something")
            for code, rem in R.DEFAULT_REMEDIES.items()}
    ks, other, problems = scan(rems)
    assert ks == 0 and other == len(rems) and not problems
    bad = guard_problems(ks, other, problems, len(remedy_commands(rems)))
    assert bad and "只扫到 0 条" in "\n".join(bad), bad


def test_checker_notices_a_kernoscope_command_it_cannot_classify():
    """提到 kernoscope 却带着不认识的包装前缀 → 必须算问题（不许当成"非 kernoscope"跳过）。"""
    rems = _with_command(R.RefusalCode.NO_GPU,
                         "docker run --rm kernoscope --ks-synthetic --set basic")
    ks, other, problems = scan(rems)
    total = len(remedy_commands(rems))
    bad = guard_problems(ks, other, problems, total)
    assert any("docker" in p for p in bad), bad
    assert ks + other == total - 1, "不可分类的 kernoscope 命令不许被算进「非 kernoscope」"


def test_checker_catches_a_bad_flag_in_an_otherwise_known_command():
    """开关写错（子命令本身对）也必须红：`--ks-achieved-occupancy` 挂在 occupancy 上不合法。"""
    rems = _with_command(R.RefusalCode.NO_OCCUPANCY_PROBE,
                         "kernoscope occupancy --from rec.txt --ks-achieved-occupancy occ.json")
    _ks, _other, problems = scan(rems)
    assert problems and "no_occupancy_probe" in "\n".join(problems), problems
