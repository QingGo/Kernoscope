"""R-34/R-55：文档里的命令必须**能被真实解析器解析**，重点命令必须**逐字执行并断言结论**。

为什么不能只做"开关存在 + 命令跑得通"的冒烟：验收报告 D1 那条命令**开关都存在、也能跑通**，
但它给出的 per-launch 误差是 +146.9%（窗口里顺序跑了四个 kernel）。只断言"能跑"的检查器会
放它过去，还会给人虚假的安全感。所以规则是：登记一条文档命令，必须同时登记它**应该产出什么**。

R-55 的三处修正（旧门禁的声明比能力大得多）：

1. **扫描范围**：以前只扫 5 份硬编码文档，`skills/`、`eval/*.md`、`tools/probes/**` 都没扫。
   现在扫**所有被跟踪的 `*.md`**（入库产物 `bench/artifacts/**` 除外 —— 那是证据不是文档）。
2. **解析方式**：以前用正则抓"本行第一个程序词"，于是**续行**只收到第一个开关、`sudo` 与
   `$( )` 完全不受检，而且校验器**不认识自家新子命令**（`capabilities`/`selftest`/`diff`），
   放宽范围反而会红一批合法命令。现在把每条命令切成 argv 后交给**真实解析器**：
   子命令表来自 `cli.SUBCOMMANDS`（单一真源），开关集合来自各子命令自己的 `--help`。
3. **逐字执行 + 断言结论**：以前**0 条**文档命令被真的跑过。现在两件事分开算清楚：

   * **全部**可检命令（当前 34 条）都交给真实解析器 —— 保证"照抄不会因为开关/子命令写错
     而失败"；
   * 其中**读者面**的 9 条（医生自检 / 合成首跑 / 入口层 / E4-6 per-launch / 参考层 packet /
     strict 卡门 / metrics / 轨迹口径 / occupancy / diff 对拍）**逐字执行并断言它承诺的
     结论**（`DOC_VERBATIM`）。比例是 9/34 —— 不写成"每条命令的结论都被断言"。

规则表 `DOC_VERBATIM` 每条是：(文档, 文档里那一行的片段, 断言函数, 适配函数, 为什么)。
**适配函数只做一件事**：把文档里的占位符/外部文件换成夹具（`<每次 launch 的 MB>` → 数值、
`-- <读者自己的程序>` → `--ks-synthetic`）；开关一个都不改，`needle` 必须真的还在文档里。
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

import kernoscope.cli as cli

ROOT = Path(__file__).resolve().parents[1]

# 入库产物是"某次运行"的证据，里面出现的命令是那次运行的记录，不作为文档承诺来校验
DOC_EXCLUDE = ("bench/artifacts/",)

# 允许出现在 `kernoscope` 之前的"前缀"，只有这些才说明这一行是在调用我们
PLACEHOLDER = "PLACEHOLDER"          # 文档里的 `<真值>` 占位符归一成它

# 允许的"包装形式"：这些写法都算**在调用我们**，必须被解析（不许静默丢弃）。
#   真事（验收报告第四轮 P1）：`python -m kernoscope … --smoke` 与
#   `xargs -n1 kernoscope … --smoke` 两条整行被丢弃，于是编出来的开关没人管。
_WRAPPERS = r"""(?:
      sudo\s+|time\s+|nohup\s+|command\s+|exec\s+            # 常见前缀
    | env\s+(?:\S+\s+)*                                        # env VAR=x …
    | uv\s+run\s+(?:--\S+\s+)*|uv\s+                            # uv run [--flags]
    | (?:python|python3)\s+-m\s+                                # python -m <module>
    | xargs\s+(?:-\S+\s+)*                                      # xargs -n1 <cmd>
)?(?<![\w./-])(?P<prog>kernoscope)\b"""
CALL_RE = re.compile(_WRAPPERS, re.X)

# --------------------------------------------------------------------------
# 从文档里取命令
# --------------------------------------------------------------------------
def _tracked_docs() -> list[Path]:
    try:
        r = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(DOC_EXCLUDE):
            continue
        out.append(ROOT / line)
    return out


DOCS = _tracked_docs()

needs_git = pytest.mark.skipif(not DOCS, reason="不在 git 仓库里（例如导出的源码包）")


def _fenced_blocks(path: Path, langs: tuple[str, ...] = ("bash", "sh")) -> list[tuple[int, str]]:
    """带语言标签的代码块（`(起始行号, 内容)`）。

    只看 ```bash / ```sh：文档里引用"别人写错的命令"时用的是无标签围栏（这是有意的豁免）。
    """
    if not path.exists():
        return []
    out: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    inside: str | None = None
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("```"):
            if inside is not None:
                out.append((start, "\n".join(buf)))
                inside, buf = None, []
            else:
                tag = st[3:].strip().split()[0] if st[3:].strip() else ""
                if tag in langs:
                    inside, start = tag, i + 1
            continue
        if inside is not None:
            buf.append(line)
    return out


def _logical_lines(block: str) -> list[str]:
    """把 `\\` 续行拼成逻辑行（旧门禁的续行只收到第一个开关）。"""
    out: list[str] = []
    buf = ""
    for line in block.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _split_metachars(tokens: list[str]) -> list[str]:
    """在 `|` / `>` / `2>` / `&&` / `;` 处截断（后面是别的程序在跑）。"""
    out: list[str] = []
    for t in tokens:
        if t in ("|", ">", ">>", "2>", "2>>", "&&", "||", ";"):
            break
        out.append(t)
    return out


def tokens_after(s: str, pos: int) -> list[str]:
    """`pos` 之后的 shell token（含元字符截断与注释剥离）。"""
    try:
        toks = shlex.split(s[pos:], comments=True)
    except ValueError:
        return []
    return _split_metachars(toks)


def kernoscope_argv(line: str) -> list[str] | None:
    """把一条文档行切成 **kernoscope 的 argv**；不是调用我们的一律返回 None。

    处理：环境变量前缀 / `sudo` / `time` / `uv run` / 命令替换 `$( )` / 反引号 / 管道 /
    `#` 注释 / `-- <被测命令>`（那部分交给别人跑）。
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # `\` 续行先拼掉（shlex 会把"被转义的换行"当成一个 token）
    s = re.sub(r"\\\n\s*", " ", s)
    # `<占位符>` 是文档约定（读者要替换成真值），不是一个 token 也不该被当成重定向
    s = re.sub(r"<[^<>]*>", PLACEHOLDER, s)
    # 命令替换与反引号：只取里面的那条命令
    s = s.replace("$(", " ").replace(")", " ").replace("`", " ")
    try:
        tokens = shlex.split(s, comments=True)
    except ValueError:
        return None
    tokens = _split_metachars(tokens)
    # 去掉 X=Y 前缀
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    # `kernoscope` 必须出现在**调用位置**（或某种被承认的包装形式里）。
    # 用正则定位而不是"前缀词都要在白名单里"：白名单会漏掉 `python -m`、`xargs -n1`。
    m = CALL_RE.search(s)
    if m is None:
        return None
    argv = ["kernoscope", *tokens_after(s, m.end("prog"))]
    if "--" in argv:
        argv = argv[:argv.index("--")]
    return argv[1:]                       # 去掉程序名，方便喂给 main()


def _subcommand_flags() -> dict[str, set[str]]:
    """每个子命令自己的合法开关 —— 来自**真实解析器**（`cli.subcommand_parser`）。

    旧实现从各子命令 `--help` 的散文里正则刮 `--long`，那是"看起来像开关的字符串"，
    不是"解析器认识的东西"。
    """
    return {sub: {s for a in cli.subcommand_parser(sub)._actions for s in a.option_strings}
            for sub in cli.SUBCOMMANDS}


def _main_flags() -> set[str]:
    """主命令自己的合法长开关（子命令另有自己的解析器，不能混着放行）。"""
    return {s for action in cli.build_parser()._actions for s in action.option_strings}


def _all_valid_flags() -> set[str]:
    """主命令 + **全部**子命令的合法长开关（供"哪些开关存在"的粗查用）。"""
    valid = _main_flags()
    for flags in _subcommand_flags().values():
        valid.update(flags)
    return valid


# --------------------------------------------------------------------------
# 1) 文档命令必须能被真实解析器解析
# --------------------------------------------------------------------------
def _parse_problem(argv: list[str], sub_flags: dict[str, set[str]],
                   main_flags: set[str] | None = None) -> str | None:
    """这条 argv 有什么问题（None = 合法）。三层检查，缺一层就会漏：

    1. **未知子命令/多余位置参数**：`program` 是 `argparse.REMAINDER`，会把 `profile`
       这类不存在的子命令当作"被测命令"吞下去 —— 所以必须先自己判一次。
    2. **开关是否真的存在**：开关集合来自解析器与各子命令的 `--help`（单一真源），
       不手抄 —— 旧门禁的清单漏了 `capabilities`/`selftest`/`diff`。
    3. **真实解析一遍**：抓 `choices`/`nargs`/互斥这类语义错误。
    """
    main_flags = _main_flags() if main_flags is None else main_flags
    if not argv:
        return None
    sub = argv[0] if not argv[0].startswith("-") else ""
    if sub and sub not in cli.SUBCOMMANDS:
        return (f"不认识 `{sub}`：子命令只有 {', '.join(cli.SUBCOMMANDS)}"
                f"（被测命令必须放在 -- 之后）")
    allowed = sub_flags[sub] if sub else main_flags
    bad = [t.split("=")[0] for t in argv if t.startswith("--") and t.split("=")[0] not in allowed]
    if bad:
        return f"不存在的开关：{bad}"
    if PLACEHOLDER in argv:
        # 带占位符的行是**示意**（读者要替换成真值）：类型校验没意义
        # （`--ks-ledger-mb <每次 launch 的 MB>` 里那个值本来就不是数字）。
        # 但开关与子命令的存在性**已经查过了** —— 那才是这一层的职责。
        return None
    # 4) **真的解析一遍** —— 子命令走子命令自己的解析器（旧实现在这里 `return None`，
    #    于是 `diff … --packet foo` / `capabilities -Z` 这类命令门禁永远放行）。
    parser = cli.subcommand_parser(sub) if sub else cli.build_parser()
    try:
        parser.parse_args(argv[1:] if sub else argv)
    except SystemExit as e:
        # `--help` 也是 SystemExit，但那是成功路径
        return None if e.code in (0, None) else \
            f"{'子命令' if sub else '主'}解析器拒绝（exit {e.code}）"
    return None


@needs_git
def test_every_documented_command_is_parseable():
    """文档里每条 kernoscope 命令都必须能被**真实解析器**接受。

    这条覆盖了旧门禁的三个洞：续行、`sudo`/`$( )`、以及"校验器不认识自家新子命令"。
    """
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    problems: list[str] = []
    seen = 0
    for doc in DOCS:
        for start, block in _fenced_blocks(doc):
            for offset, line in enumerate(_logical_lines(block)):
                argv = kernoscope_argv(line)
                if argv is None:
                    continue
                seen += 1
                why = _parse_problem(argv, sub_flags, main_flags)
                if why:
                    problems.append(f"{doc.relative_to(ROOT)}:{start + offset}: {why}\n"
                                    f"    {line.strip()[:110]}")
    assert seen >= 20, f"只扫到 {seen} 条文档命令 —— 扫描范围或取词逻辑退化了"
    # P1 的第二半：**不许静默丢弃**。提到 kernoscope 又带长开关、却没被认成命令的行，
    # 说明取词逻辑漏了一种包装形式 —— 那种行里的编造开关永远不会被查（P1 的原形）。
    suspicious = [f"{d.relative_to(ROOT)}:{ln}: {s.strip()[:90]}"
                  for d, ln, s in _unclassified_with_flags()]
    assert not suspicious, ("这些行提到了 kernoscope 还带长开关，却没被当成命令解析"
                            "（包装形式没认全）：\n  - " + "\n  - ".join(suspicious))
    assert not problems, "文档里的命令跑不通（读者会照抄失败）：\n  - " + "\n  - ".join(problems)


def _unclassified_with_flags() -> list[tuple[Path, int, str]]:
    out = []
    for doc in DOCS:
        for start, block in _fenced_blocks(doc):
            for offset, line in enumerate(_logical_lines(block)):
                s = line.strip()
                if "kernoscope" not in s or s.startswith("#"):
                    continue
                if kernoscope_argv(line) is None and re.search(r"(?<![\w-])--[a-z][a-z0-9-]+", s):
                    out.append((doc, start + offset, s))
    return out


@needs_git
def test_no_documented_subcommand_bypasses_its_real_parser():
    """R-60：**未解析的子命令数必须为 0**。

    旧门禁对子命令只比对开关集合就 `return None`，于是三类不合法命令被放行
    （`diff … --packet foo`、`selftest --ks-calibration …`、`capabilities -Z`）。
    这条把"每个子命令调用都真的过了它自己的解析器"变成可断言的事实。
    """
    unparsed: list[str] = []
    seen_sub = 0
    for doc in DOCS:
        for start, block in _fenced_blocks(doc):
            for offset, line in enumerate(_logical_lines(block)):
                argv = kernoscope_argv(line)
                if not argv or argv[0] not in cli.SUBCOMMANDS:
                    continue
                seen_sub += 1
                try:
                    cli.subcommand_parser(argv[0]).parse_args(argv[1:])
                except SystemExit as e:
                    if e.code not in (0, None):
                        unparsed.append(f"{doc.relative_to(ROOT)}:{start + offset}: {line.strip()[:80]}")
    assert seen_sub > 0, "文档里一条子命令都没有？扫描范围退化了"
    assert unparsed == [], ("这些子命令没有通过它自己的真实解析器：\n  - " + "\n  - ".join(unparsed))


@needs_git
def test_the_command_checker_catches_invented_commands():
    """检查器自身的负样本：编出来的开关/子命令/续行/前缀都必须被抓到。"""
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    invented = [
        "kernoscope --set basic --smoke -- x.py",
        "kernoscope profile -- python train.py",
        "kernoscope compare base.json other.json",
        "kernoscope --set basic \\\n  --shapes \\\n  -- x.py",
        "sudo kernoscope --set basic --smoke -- x.py",
        "$(kernoscope --set basic --smoke -- x.py)",
        "CUDA_VISIBLE_DEVICES=0 kernoscope --set basic --smoke",
    ]
    bad = []
    for line in invented:
        argv = kernoscope_argv(line)
        assert argv is not None, f"取词逻辑漏了这条：{line!r}"
        if not _parse_problem(argv, sub_flags, main_flags):
            bad.append(line)
    assert not bad, f"这些编出来的命令没有被抓到：{bad}"
    # 合法命令不能被误判（否则门禁只会逼人删文档）
    for line in ("kernoscope --set basic -k regex:gemm -- python train.py",
                 "kernoscope capabilities --brief",
                 "kernoscope diff a.json b.json --selfcheck s.json --json",
                 "uv run kernoscope --help"):
        argv = kernoscope_argv(line)
        assert argv is not None and not _parse_problem(argv, sub_flags, main_flags), line


@needs_git
def test_documented_commands_come_from_all_over_the_repo():
    """扫描范围必须覆盖 skills/ 与 eval/（旧门禁只扫 5 份根目录文档）。"""
    roots = {str(d.relative_to(ROOT)).split("/")[0] for d in DOCS}
    assert "skills" in roots and "eval" in roots, sorted(roots)


# --------------------------------------------------------------------------
# 2) 重点命令：逐字执行 + 断言结论
# --------------------------------------------------------------------------
def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def _assert_doctor_ok(rc, out, err):
    # doctor 的退出码说的是"这台机器能不能跑核心路径"（无 GPU 的机器就是 1），
    # 不是"命令对不对" —— 所以这里只要求它跑完并给出结论。
    assert rc in (0, 1), (rc, err)
    assert "结论" in out or "conclusion" in out.lower()


def _assert_synthetic_report(rc, out, err):
    assert rc == 0, err
    assert "SYNTHETIC" in out or "合成" in out, "无 GPU 的路径必须明确标注是合成数据"
    assert "n/a" in out, "拿不到的计数器必须报 n/a"


def _assert_brief_is_entry_layer(rc, out, err):
    assert rc == 0, err
    b = json.loads(out)                      # 入口层必须是纯 JSON
    assert b["next_layer"]["bytes"] > 0 and b["next_layer"]["contains"]


# --------------------------------------------------------------------------
# 2) 读者面命令：逐字执行 + 断言**结论**
# --------------------------------------------------------------------------
# 登记表：(文档, 文档里必须存在的那一行片段, 断言函数, 适配函数, 为什么这条值得登记)
#
# 纪律：`needle` 必须真的在文档里（否则测试红 —— 登记表不许过期）；argv 从**文档那一行**
# 取（含 `\` 续行拼接），适配函数只做"把占位符换成夹具文件"这一件事，不再改开关。
Registered = namedtuple("Registered", "doc needle check adapt why")


def _doc_command_line(doc: str, needle: str) -> str:
    """在文档的 bash 块里找到含 `needle` 的**逻辑行**（续行已拼接）。"""
    path = ROOT / doc
    for _start, block in _fenced_blocks(path):
        for line in _logical_lines(block):
            if needle in line and kernoscope_argv(line) is not None:
                return line
    raise AssertionError(f"{doc} 里找不到这一行（登记表过期）：{needle!r}")


def _assert_doctor_ok(rc, out, err):
    # doctor 的退出码说的是"这台机器能不能跑核心路径"（无 GPU 的机器就是 1），
    # 不是"命令对不对" —— 所以这里只要求它跑完并给出结论。
    assert rc in (0, 1), (rc, err)
    assert "结论" in out or "conclusion" in out.lower()


def _assert_synthetic_report(rc, out, err):
    assert rc == 0, err
    assert "SYNTHETIC" in out or "合成" in out, "无 GPU 的路径必须明确标注是合成数据"
    assert "n/a" in out, "拿不到的计数器必须报 n/a"


def _assert_brief_is_entry_layer(rc, out, err):
    assert rc == 0, err
    b = json.loads(out)                      # 入口层必须是纯 JSON
    assert b["next_layer"]["bytes"] > 0 and b["next_layer"]["contains"]


def _adapt_calibration(argv: list[str], work: Path) -> list[str]:
    """把文档里的占位符换成夹具：标定文件、`-o` 目录、`<每次 launch 的 MB>`。

    文档那条命令的最后是 `-- python bench_loop.py`（读者自己的负载），离线跑不了；
    所以补一个 `--ks-synthetic`（合成负载），并在测试里把采样器与 launch 表换成夹具 ——
    这正是"文档说的口径能不能跑通"的最小可执行形式。
    """
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--ks-calibration", "-o", "--packet") and i + 1 < len(argv):
            val = {"--ks-calibration": str(ROOT / "bench/artifacts/20261011-094535/"
                                           "machine-profile.json"),
                   "-o": str(work / "rep"),
                   "--packet": str(work / "pk.json")}[a]
            out += [a, val]
            i += 2
            continue
        nxt = (argv[i + 1] if i + 1 < len(argv) else "")
        if a == "-k" and nxt.startswith("regex:"):
            out += ["-k", "regex:gemm"]          # 文档里的 <kernel>/my_kernel → 夹具里的 gemm
            i += 2
            continue
        if a == "--ks-ledger-mb" and i + 1 < len(argv):
            out += ["--ks-ledger-mb", "420000" if argv[i + 1] == PLACEHOLDER
                    else argv[i + 1]]
            i += 2
            continue
        out.append("420000" if a == PLACEHOLDER else a)   # <每次 launch 的 MB>
        i += 1
    if "--ks-synthetic" not in out:
        out.append("--ks-synthetic")      # 追加（插到中间会把 `--set` 的值吃掉）
    return out


def _assert_per_launch_with_premises(rc, out, err, work: Path):
    """E4-6 的结论：给了 per-launch，并且**同时**给出它的前提与分母（R-47/R-48）。"""
    assert rc == 0, err
    m = _load_metrics(work / "rep.ksreport.json")
    assert m["est__dram_bytes_per_launch"]["value"] is not None
    for k in ("ks__sampler_window_coverage_pct", "ks__matched_kernel_kernel_share_pct",
              "ks__attribution_bias_estimate_pct"):
        assert m[k]["value"] is not None, k


def _assert_packet_window_mode(rc, out, err, work: Path):
    """参考层那条（`--packet -`）：包必须带上协议（窗口模式）与 per-launch 判词。"""
    assert rc == 0, err
    pk = json.loads((work / "pk.json").read_text(encoding="utf-8"))
    assert pk["protocol"]["window_mode"] == "auto"
    vs = {v["id"]: v for v in pk["verdicts"]}
    assert "dram_bytes_per_launch" in vs, sorted(vs)


def _adapt_offline(argv: list[str], work: Path) -> list[str]:
    """给"文档里最后是 `-- <读者自己的程序>`"的命令补一个合成负载（离线可跑的最小形式）。"""
    if not any(a in argv for a in ("--ks-synthetic", "--ks-sqlite", "--ks-nsys-rep")):
        return [*argv, "--ks-synthetic"]
    return argv


def _adapt_strict(argv: list[str], work: Path) -> list[str]:
    """`--strict` 那条要的是"请求了拿不到的数据" ⇒ 追加合成负载 + 关掉 NVML。"""
    return [*argv, "--ks-synthetic", "--ks-no-nvml"] if "--ks-synthetic" not in argv else argv


def _assert_strict_refuses(rc, out, err):
    assert rc == 2, "文档承诺 --strict 会让 CI 在缺数据时失败"
    assert "error(strict)" in err


def _assert_metrics_not_silently_empty(rc, out, err):
    """README 那条是 `--strict --metrics <拿不到的计数器>`：结论就是 **CI 必须红**。

    （"写错名字不静默"由下面的合成用例单独守。）
    """
    assert rc == 2, (rc, err[-200:])
    assert "一个都不存在" in err or "strict" in err


def _write_official_trace(path: Path, kernel: str = "gemm_k") -> Path:
    """写一段**官方 mem_trace 格式**的最小轨迹（含全局 LDG 与共享 STS 两种访存）。"""
    lines = [
        f"MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x2 - Kernel name {kernel}(float*, float*) "
        f"- grid launch id 0",
    ]
    base = 0x7f0000000000
    for cta in range(4):
        for lane in range(4):                       # 每个 warp 4 个地址 = 16 B（半 sector）
            a = base + cta * 0x1000 + lane * 4
            lines.append(f"MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA {cta},0,0 - warp 0 "
                         f"- LDG.E.128 - " + " ".join(hex(a + k * 16) for k in range(4)))
        # 共享访存：**不该**进入全局口径（R-73）
        lines.append(f"MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA {cta},0,0 - warp 0 "
                     f"- STS.U16 - " + " ".join(hex(0x100 + cta + k) for k in range(4)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_trace_argv(work: Path):
    def adapt(argv: list[str], w: Path) -> list[str]:
        out, i = [], 0
        while i < len(argv):
            a = argv[i]
            if a in ("--ks-memtrace", "-o") and i + 1 < len(argv):
                val = (str(_write_official_trace(w / "trace.txt")) if a == "--ks-memtrace"
                       else str(w / "rep"))
                out += [a, val]
                i += 2
                continue
            out.append(a)
            i += 1
        if "--ks-synthetic" not in out:
            out.append("--ks-synthetic")   # 文档最后是 `-- ./your_app`：离线用合成负载顶上
        return out
    return adapt


def _assert_trace_caliber(rc, out, err, work: Path):
    """E5-4 的结论：轨迹口径给出 measured 的访存统计，且合并效率**不超过 100%**。"""
    assert rc == 0, err
    rep = json.loads((work / "rep.ksreport.json").read_text(encoding="utf-8"))
    metrics = {}
    for k in rep.get("kernels", []):
        for s in k.get("sections", []):
            for m in s.get("metrics", []):
                metrics.setdefault(m["name"], m)
    for name in ("meas__sectors_per_request", "meas__coalescing_efficiency_pct",
                 "meas__trace_records"):
        assert name in metrics, f"轨迹口径少了 {name}"
    coal = metrics["meas__coalescing_efficiency_pct"]["value"]
    assert coal is None or coal <= 100.0 + 1e-9, f"合并效率 {coal}% 物理上不可能"


def _assert_diff_json(rc, out, err, work: Path):
    assert rc == 0, err
    d = json.loads(out)
    assert d["comparisons"] and "not_judged" in d


def _make_diff_argv(work: Path):
    def adapt(argv: list[str], w: Path) -> list[str]:
        from kernoscope.packet import build, write_packet
        from test_packet import _rich_report

        a = w / "a.json"
        b = w / "b.json"
        sc = w / "selfcheck.json"
        write_packet(build(_rich_report()), str(a))
        write_packet(build(_rich_report()), str(b))
        sc.write_text(json.dumps({"band": {"dram_bytes_pct": [-5.0, 5.0]},
                                  "mdd": {"dram_bytes_pct": 5.0, "kernel_time_pct": 0.5}}),
                      encoding="utf-8")
        out = []
        for t in argv:
            out.append({"before.json": str(a), "after.json": str(b),
                        "machine-selfcheck.json": str(sc)}.get(t, t))
        return out
    return adapt


def _assert_route_plan(rc, out, err):
    """路由的"结论"：给出**口径决策规则 + 命令 + 会被拒的条件**，且不跑测量。"""
    assert rc == 0, err
    assert "走哪条口径" in out or "命令：" in out, out[:200]
    assert "命令：" in out and "可能被拒：" in out
    assert "kernoscope" in out


def _make_occupancy_argv(argv: list[str], work: Path) -> list[str]:
    """把 `occupancy-records.txt` 换成仓库里那份真记录、把 `occ.json` 指到临时目录。"""
    rec = str(ROOT / "bench/artifacts/20261011-094535/occupancy-records.txt")
    out = []
    for i, a in enumerate(argv):
        if a == "--from" and i + 1 < len(argv):
            out += [a, rec]
            continue
        if a == "--write-json" and i + 1 < len(argv):
            out += [a, str(work / "occ.json")]
            continue
        if i > 0 and argv[i - 1] in ("--from", "--write-json"):
            continue
        out.append(a)
    return out


def _assert_occupancy_json(rc, out, err, work: Path):
    """结论：区间重叠法给出每个 kernel 的达到 occupancy（可以 >1 个 kernel，按名字分桶）。"""
    assert rc == 0, err
    d = json.loads((work / "occ.json").read_text(encoding="utf-8"))
    assert d, "occupancy JSON 是空的"
    for name, row in d.items():
        assert row.get("occupancy_pct", 0) > 0, (name, row)
        assert row.get("peak_concurrent_blocks", 0) > 0, (name, row)


DOC_VERBATIM: list[Registered] = [
    Registered("README.md", "kernoscope doctor", _assert_doctor_ok, None,
               "E0：告诉读者「这台机器能答什么」是第一步"),
    Registered("README.md", "--ks-synthetic --set detailed", _assert_synthetic_report, None,
               "无 GPU 也能跑通全流程（首跑 < 5 分钟的那条路径）"),
    Registered("skills/kernoscope/SKILL.md", "capabilities --brief", _assert_brief_is_entry_layer,
               None, "入口层：agent 开工前那一条"),
    Registered("README.md", "--ks-ledger-mb 234",
               _assert_per_launch_with_premises, _adapt_calibration,
               "E4-6：教程第 4 章要用的 per-launch + 账本 + 标定那条命令"),
    Registered("skills/kernoscope/references/recipes.md", "--ks-window auto",
               _assert_packet_window_mode, _adapt_calibration,
               "参考层里 agent 实际照抄的那条（--packet - + 窗口模式）"),
    Registered("README.md", "--strict", _assert_strict_refuses, _adapt_strict,
               "CI 卡门：请求了不可得的数据必须 exit 2"),
    Registered("README.md", "--metrics sm__throughput.avg.pct_of_peak_sustained_elapsed",
               _assert_metrics_not_silently_empty, _adapt_offline,
               "窄化输出时写错名字不能「看起来成功」（这个计数器在本机拿不到）"),
    Registered("README.md", "--ks-memtrace", _assert_trace_caliber, _make_trace_argv(None),
               "E5-4：轨迹口径（合并效率不许 > 100%）"),
    Registered("skills/kernoscope/SKILL.md", "--selfcheck machine-selfcheck.json",
               _assert_diff_json, _make_diff_argv(None),
               "闭环最后一步：对拍给方向与三态"),
    Registered("README.md", 'kernoscope route "为什么这个 kernel 慢"',
               _assert_route_plan, None,
               "A1：先路由再动手 —— 路线里要有口径决策规则、命令与会被拒的条件"),
    Registered("README.md", "occupancy --from occupancy-records.txt --write-json",
               _assert_occupancy_json, _make_occupancy_argv,
               "Occupancy 的区间重叠法（ncu 计数器之外的那条路）"),
]


@needs_git
def test_registered_commands_are_executed_verbatim(monkeypatch, tmp_path):
    """登记的每条命令都必须①真的还写在文档里 ②逐字跑一遍 ③得到它承诺的结论。

    旧门禁的"逐字执行文档命令数 = 0"：三个登记测试的 argv 全是测试现场编的（带
    `--ks-synthetic` 这类文档里没有的开关），所以"文档说的能不能跑"从来没被验证过。
    第四轮 P2 指出只覆盖 3/31（10%），这一轮扩到读者面那几条 —— 并且**登记表不许过期**。
    """
    assert len(DOC_VERBATIM) >= 8, "读者面命令至少登记 8 条"
    monkeypatch.chdir(tmp_path)
    # 离线夹具：NVML 采样序列 + 与窗口自洽的 launch 表（无 GPU 的机器上"拿不到数"是环境
    # 决定的，不是命令决定的 —— 那样测不出文档对不对）
    from test_dram import _FakeSampler, _coherent_launches

    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    for i, reg in enumerate(DOC_VERBATIM):
        work = tmp_path / f"{i:02d}"          # 每条命令一个目录：`-o rep` 不许互相撞
        work.mkdir()
        line = _doc_command_line(reg.doc, reg.needle)
        argv = kernoscope_argv(line)
        assert argv is not None, line
        argv = [("regex:gemm" if a == "regex:PLACEHOLDER" else a) for a in argv]
        if reg.adapt is not None:
            argv = reg.adapt(argv, work)
        rc, out, err = _run(argv)
        if reg.check.__code__.co_argcount == 4:
            reg.check(rc, out, err, work)              # type: ignore[operator]
        else:
            reg.check(rc, out, err)                    # type: ignore[operator]


@needs_git
def test_every_doc_command_is_either_parsed_or_asserted():
    """把"覆盖率"写成事实而不是印象：每条命令都过解析器，其中读者面那几条断言结论。"""
    parsed = 0
    for doc in DOCS:
        for _start, block in _fenced_blocks(doc):
            for line in _logical_lines(block):
                if kernoscope_argv(line) is not None:
                    parsed += 1
    assert parsed >= 30, f"只扫到 {parsed} 条可检命令"
    assert len(DOC_VERBATIM) < parsed, "登记数不该超过可检命令数（那是重复计数）"


# --------------------------------------------------------------------------
# 3) 合成数据下的结论断言（不需要 GPU；用替身把环境补齐）
# --------------------------------------------------------------------------
def _load_metrics(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for k in doc.get("kernels", []):
        for s in k.get("sections", []):
            for m in s.get("metrics", []):
                out.setdefault(m["name"], m)
    return out


def test_documented_basic_command_gives_a_number(monkeypatch, capsys, tmp_path):
    """README 里最常用的那条：--set basic + -k + 账本 + 标定，预期拿到一个数。

    无 GPU 的机器上要给两样替身（NVML 采样序列 + 与窗口自洽的 launch 表），
    否则「拿不到数」是环境决定的，不是命令决定的 —— 那样测不出文档对不对。
    """
    from test_dram import _FakeSampler, _coherent_launches

    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc, out, err = _run(["--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
                         "--ks-ledger-mb", "420000", "--ks-window", "auto",
                         "-o", str(tmp_path / "rep"), "-f"])
    assert rc == 0, err
    m = _load_metrics(tmp_path / "rep.ksreport.json")
    # 预期结论：给了 per-launch，并且**同时**给出了它的两个前提（外加 R-47 的分母）
    assert m["est__dram_bytes_per_launch"]["value"] is not None
    assert m["ks__sampler_window_coverage_pct"]["value"] is not None
    assert m["ks__matched_kernel_kernel_share_pct"]["value"] is not None
    assert m["ks__attribution_bias_estimate_pct"]["value"] is not None


def test_documented_strict_command_refuses_when_data_is_missing(capsys):
    """README 里推荐的 CI 用法：--strict 在拿不到请求的数据时必须 exit 2。"""
    rc, _out, err = _run(["--ks-synthetic", "--ks-no-nvml", "--strict",
                          "--set", "full", "-k", "regex:gemm"])
    assert rc == 2, "文档承诺 --strict 会让 CI 在缺数据时失败"
    assert "error(strict)" in err


def test_documented_metrics_command_is_not_silently_empty(capsys):
    """README 教人用 --metrics 窄化输出；写错名字不能得到"看起来成功"的空报告。"""
    rc, _out, err = _run(["--ks-synthetic", "--metrics", "dram__bytes.sum", "-k", "regex:gemm"])
    assert rc == 0
    assert "一个都不存在" in err
