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

   * **全部**可检命令（当前 68 条）都交给真实解析器 —— 保证"照抄不会因为开关/子命令写错
     而失败"；
   * 其中读者面那 **11 条**（医生自检 / 合成首跑 / 入口层 / E4-6 per-launch / 参考层 packet /
     strict 卡门 / metrics / 轨迹口径 / occupancy / diff 对拍）**逐字执行并断言它承诺的
     结论**（`DOC_VERBATIM`）。比例是 **11/68** —— 不写成"每条命令的结论都被断言"。

   这两个数字**由测试现算并断言**（`test_prose_counts_match_the_scan` /
   `test_changelog_command_counts_match_the_scan`）：写死在散文里的旧版本漂到过
   "docstring 34/9、CHANGELOG 59"，而没有任何断言管它们（W-9/R-104）。

规则表 `DOC_VERBATIM` 每条是：(文档, 文档里那一行的片段, 断言函数, 适配函数, 为什么)。
**适配函数只做一件事**：把文档里的占位符/外部文件换成夹具（`<每次 launch 的 MB>` → 数值、
`-- <读者自己的程序>` → `--ks-synthetic`）；开关一个都不改，`needle` 必须真的还在文档里。

第五轮（V-23/V-24）的两处修正：

* **扫描范围（V-23）**：旧实现只认 ```bash / ```sh 围栏 —— 实测往 ```console / ```text /
  **无标签**围栏里注入一条编造的开关，门禁全绿。现在把命令围栏扩到读者真会照抄的那几种，
  并把"围栏里明显不是命令的东西"（JSON/别的语言/整块输出）**逐条写清**地排除掉，
  而不是静默丢弃。取词仍然**只在围栏内 + 行首**：报告里验收方自己写散文扫描时产生过
  10 条假阳性（把 `入口命令：kernoscope …` 这种半截命令当成命令）。
* **标签穷尽（W-11/R-105）**：白名单之外的标签此前被**静默**当成"非命令围栏"，于是
  ```output / ```shellsession / ```terminal 里的编造开关全绿。现在 `output` / `shellsession` /
  `terminal` 进了命令白名单（里面的坏命令必须被抓到），而**任何没被显式分类**的标签都会被
  `_fence_tag_problem()` 报出来要求二选一（扫 or 写清理由豁免）—— 静默跳过这条路没有了。
* **空输入集（V-24）**：旧实现在没有 `.git` 时把整族测试 `skipif` 掉，于是导出的源码树里
  「文档命令门禁」静默失效（23 passed / 6 skipped，看起来全绿）。现在空输入集**失败**并
  说清原因，唯一出口是显式 `KS_ALLOW_EMPTY_DOC_SCAN=1`（接受这次降级 ⇒ skip，不是假绿）。
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
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

# 允许的**行首**包装形式：这些写法都算在调用我们，必须被解析（不许静默丢弃）。
#   真事（验收报告第四轮 P1）：`python -m kernoscope … --smoke` 与
#   `xargs -n1 kernoscope … --smoke` 两条整行被丢弃，于是编出来的开关没人管。
#   真事（第五轮 V-23）：散文里的半截命令被当成命令（10 条假阳性），所以这里要求
#   调用出现在**行首**，并且用 `^\s*` 锚定 —— 散文行不再参与取词。
_PREFIX_ATOMS = (
    r"[A-Za-z_][A-Za-z0-9_]*=\S*\s+"                    # VAR=value
    r"|(?:\$|>|❯|%)\s+"                                  # console 提示符（`$ `）
    r"|sudo\s+|time\s+|nohup\s+|command\s+|exec\s+"      # 常见前缀
    r"|env\s+(?:\S+\s+)*"                                # env VAR=x …
    r"|uv\s+run\s+(?:--\S+\s+)*|uv\s+"                   # uv run [--flags]
    r"|(?:python|python3)\s+-m\s+"                       # python -m <module>
    r"|xargs\s+(?:-\S+\s+)*"                             # xargs -n1 <cmd>
)
INVOCATION_RE = re.compile(r"^\s*(?:" + _PREFIX_ATOMS + r")*(?P<prog>kernoscope)\b(?![\w./-])")

# 「行首像命令」的**弱**判据：kernoscope 出现在行首前 4 个 ASCII 词里。
#   只用来抓"包装形式没认全"（例如 `xvfb-run kernoscope … --smoke`）。中文散文行
#   （`入口命令：kernoscope …`）在这里不会匹配 —— 那正是 10 条假阳性的来源。
NEAR_START_RE = re.compile(r"^\s*(?:[A-Za-z_][\w./-]*\s+){0,4}kernoscope\b(?![\w./-])")
LONG_FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")
# **散文**的判据：中文标点。参数可以是中文（`route "为什么这个 kernel 慢"`），
# 但一行里出现中文**标点**说明它是说明文字 —— 例如
# `kernoscope selftest / kernoscope diff。不要凭印象拼子命令（先看 kernoscope --help）。`
# 这种"命令列表写成散文"的行，正是报告里 10 条假阳性的形状。
PROSE_PUNCT_RE = re.compile(r"[。，、；：！？（）【】「」『』〈〉《》→←…—]")

# **命令围栏**的标签白名单（V-23 + W-11）：读者真会照抄的那几种 + 无标签 + 会贴命令的
#   "会话/输出" 类标签。旧实现只认 ```bash / ```sh，于是往 ```console / ```text / 无标签
#   围栏里注入一条编造的开关，门禁全绿；W-11 又把 ```output / ```shellsession / ```terminal
#   这几个**白名单外**的标签静默当成"非命令围栏" —— 编造的开关同样全绿。
COMMAND_FENCE_TAGS = frozenset({"", "bash", "sh", "shell", "zsh", "console", "text", "txt",
                                "shell-session", "shellsession", "terminal", "output"})
# **明确排除**的围栏标签 → 理由（逐条写清，不静默丢）。这张表是**穷尽**的：
#   没被登记的标签会被 `_fence_tag_problem()` 报出来，而不是被静默跳过（W-11 的修法）。
#   json/jsonc/yaml/toml/csv… 是**数据**；python/c/cuda 是**别的语言的代码**；
#   powershell 是**别的 shell**（这里只保证 POSIX shell 的取词规则）。
NON_COMMAND_FENCE_REASONS: dict[str, str] = {
    "json": "数据（机器可读 JSON 输出）",
    "jsonc": "数据（带注释的 JSON 输出）",
    "json5": "数据（JSON5 输出）",
    "yaml": "数据（YAML 配置/输出）",
    "yml": "数据（YAML 配置/输出）",
    "toml": "数据（TOML 配置）",
    "ini": "数据（INI 配置）",
    "csv": "数据（CSV 表格）",
    "tsv": "数据（TSV 表格）",
    "xml": "数据（XML 输出）",
    "html": "数据/标记（HTML）",
    "python": "别的语言的代码（Python）",
    "py": "别的语言的代码（Python）",
    "c": "别的语言的代码（C）",
    "cpp": "别的语言的代码（C++）",
    "cc": "别的语言的代码（C++）",
    "cuda": "别的语言的代码（CUDA C++）",
    "powershell": "别的 shell（PowerShell，这里只保证 POSIX shell 的取词规则）",
    "ps1": "别的 shell（PowerShell）",
    "diff": "差异输出（不是可执行命令）",
}

# 有意展示的**反面教材**：文档里那些"agent 凭印象编出来的命令"是**证据**，不是承诺。
# 逐条登记（文件、那些行的特征、为什么）。豁免是**逐行**的：同一个围栏里再多出别的坏
# 命令照样要被查（不然登记表就成了"豁免整块"的橡皮擦）。登记项必须仍然存在、确实不合法，
# 且**文档自己说它是错的**（见 `test_registered_negative_examples_are_really_unparseable`）。
NEGATIVE_EXAMPLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("eval/TRIGGER-EVAL.md",
     ("kernoscope profile -- python train.py",
      "kernoscope compare base_run vec_run",
      "kernoscope compare --baseline a.nsys-rep --candidate b.nsys-rep --report out.json"),
     "A/B 轮 agent 凭印象编的子命令；同段正文紧接着说「`profile` 与 `compare` 都不存在」"),
)
# 反面教材还要有**文档自己的证据**：块附近 ±20 行内的散文必须说它不对。没有这句话的块
# 不许进登记表（否则这张表就成了"随手豁免任何坏命令"的橡皮擦）。
# 判据是**具体措辞**（不存在/编了/凭印象/反例…），不是单个"错/编/误"字 —— 后者太容易
# 被无关句子满足。
NEGATIVE_EVIDENCE_RE = re.compile(
    r"不存在|不成立|编出来|编造|编了|凭印象|错例|反例|反面教材|不要照抄|别照抄|不许照抄"
    r"|跑不通|不合法|不许静默")
NEGATIVE_EVIDENCE_WINDOW = 20

# --------------------------------------------------------------------------
# 扫描输入集：空输入必须**失败**，不许静默 skip（V-24）
# --------------------------------------------------------------------------
ALLOW_EMPTY_DOC_SCAN = "KS_ALLOW_EMPTY_DOC_SCAN"


def _git_ls_files(*patterns: str) -> tuple[bool, list[str]]:
    """`git ls-files`：(git 可用吗, 被跟踪的路径)。不存在 `.git` 时第一项是 False。"""
    try:
        r = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False, []
    if r.returncode != 0:
        return False, []
    return True, [line.strip() for line in r.stdout.splitlines() if line.strip()]


def _tracked_docs() -> list[Path]:
    ok, files = _git_ls_files("*.md")
    if not ok:
        return []
    return [ROOT / f for f in files if not f.startswith(DOC_EXCLUDE)]


DOCS = _tracked_docs()


def _empty_scan_input(why: str) -> None:
    """空输入集的唯一出口：显式 `KS_ALLOW_EMPTY_DOC_SCAN=1` ⇒ skip；否则 fail。"""
    msg = (
        f"文档命令门禁的输入集是空的：{why}\n"
        "  这层门禁不能在没有输入时静默通过 —— 那正是 V-24 的原形（导出的源码树里\n"
        "  23 passed / 6 skipped，看起来全绿，其实一条命令都没查）。两种正经出路：\n"
        "  ① 在 **git 仓库**里跑：`git ls-files '*.md'` 必须能列出被跟踪的文档；\n"
        f"  ② 显式接受这次降级：设置 {ALLOW_EMPTY_DOC_SCAN}=1"
        "（此时本族测试 skip，而不是假绿）。"
    )
    if os.environ.get(ALLOW_EMPTY_DOC_SCAN, "").strip() == "1":
        pytest.skip(msg)
    pytest.fail(msg)


def _scan_inputs_problem() -> str | None:
    """扫不出输入集的**原因**（None = 有输入）：三种空输入都要能被说清。"""
    ok, files = _git_ls_files("*.md")
    if not ok:
        return "这里不是 git 工作树（例如 `git archive` 导出的源码包），拿不到被跟踪的文档清单"
    if not [f for f in files if not f.startswith(DOC_EXCLUDE)]:
        return "git 工作树里没有 DOC_EXCLUDE 之外被跟踪的 *.md"
    if not _total_command_blocks():
        return f"被跟踪的 {len(DOCS)} 份 *.md 里一个命令围栏都扫不到（扫描/取词逻辑退化了？）"
    return None


def needs_doc_scan_inputs(fn):
    """替换旧的 `skipif(not DOCS)`：空输入集 fail 而不是 skip（V-24）。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        why = _scan_inputs_problem()
        if why is not None:
            _empty_scan_input(why)
        return fn(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------
# 从文档里取命令
# --------------------------------------------------------------------------
def _registered_negative_line(path: Path, line: str) -> bool:
    """这一**行**是不是已登记的**反面教材**（见 `NEGATIVE_EXAMPLES`）。

    逐行豁免：同围栏里别的坏命令照样要被查。
    """
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    return any(rel == doc and any(snippet in line for snippet in snippets)
               for doc, snippets, _why in NEGATIVE_EXAMPLES)


def _scan_lines(path: Path):
    """要校验的 `(行号, 逻辑行)` —— 命令围栏内、去掉已登记的反面教材。"""
    for start, block in _fenced_blocks(path):
        for offset, line in enumerate(_logical_lines(block)):
            if _registered_negative_line(path, line):
                continue          # 文档**有意**展示的错例（逐行登记，见上）
            yield start + offset, line


def _is_data_block(content: str) -> bool:
    """整块像**数据/输出**而不是命令：首个非空行以 `{`/`[` 开头（JSON/数组输出）。

    `console`/`text` 围栏里贴机器输出很常见；这类块整块排除（空块也没有命令）。
    """
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        return s.startswith("{") or s.startswith("[")
    return True


def _iter_fences(path: Path):
    """文件里**每一个**围栏：`(标签, 起始行号, 内容)`（标签已小写；无标签是 `""`）。

    唯一一处解析 ```` ``` ```` 的逻辑 —— `_fenced_blocks()`（取命令）与
    `_all_fence_tags()`（查"标签有没有被显式分类"，W-11）都从这里派生，
    于是不会出现"一个函数认这个标签、另一个函数不认"的分叉。
    """
    if not path.exists():
        return
    inside = False
    buf: list[str] = []
    tag = ""
    start = 0
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        st = line.strip()
        if st.startswith("```"):
            if inside:
                yield tag, start, "\n".join(buf)
                inside, buf = False, []
            else:
                inside = True
                info = st[3:].strip()
                tag = info.split()[0].lower() if info else ""
                start = i + 1
                buf = []
            continue
        if inside:
            buf.append(line)


def _fenced_blocks(path: Path, langs: frozenset[str] | None = None) -> list[tuple[int, str]]:
    """**命令围栏**（`(起始行号, 内容)`；V-23 起不再只认 ```bash/```sh）。

    扫描范围 = COMMAND_FENCE_TAGS 里的围栏，再减掉两类**明确**排除的内容：

      * 标签排除：`NON_COMMAND_FENCE_REASONS`（JSON/YAML/Python/CUDA/PowerShell… 是数据或
        别的语言，逐条写了理由）；
      * 整块形状排除：`_is_data_block`（整块以 `{`/`[` 开头的当输出/数据）。

    **白名单外的标签不是"排除"，是"没分类"**：`_fence_tag_problem()` 会把它们报出来
    （W-11 之前是静默跳过 —— 往 ```output 里塞编造的开关，门禁全绿）。

    （第三类排除 —— 文档有意展示的反面教材 —— 是**逐行**做的，在 `_scan_lines` 里；
    这样同一个围栏里再多出别的坏命令照样会被查到。）

    行首取词（命令必须出现在行首）不在这里做，而在 `kernoscope_argv` 里 —— 于是
    "围栏里的散文行"自然不产生命令，而"围栏里的坏命令"一条都跑不掉。
    """
    langs = COMMAND_FENCE_TAGS if langs is None else langs
    return [(start, content) for tag, start, content in _iter_fences(path)
            if tag in langs and not _is_data_block(content)]


def _all_fence_tags(path: Path) -> list[str]:
    """文档里用过的**全部**围栏标签（按出现顺序，含重复；用于穷尽性检查）。"""
    return [tag for tag, _start, _content in _iter_fences(path)]


def _fence_tag_problem(tag: str) -> str | None:
    """这个围栏标签有没有被**显式**分类（None = 有）。

    被分类 = 在 `COMMAND_FENCE_TAGS`（会被当命令扫）或 `NON_COMMAND_FENCE_REASONS`
    （有理由的豁免）。**没分类的标签不再是"静默跳过"**：`unknown` 会被报出来，
    要求作者二选一（W-11/R-105）。
    """
    if tag in COMMAND_FENCE_TAGS:
        return None
    if NON_COMMAND_FENCE_REASONS.get(tag, "").strip():
        return None
    return (f"未分类的围栏标签 ```{tag}：它既不在 COMMAND_FENCE_TAGS（会被当命令扫），"
            f"也不在 NON_COMMAND_FENCE_REASONS（有理由的豁免）里。白名单外的标签不许"
            f"被静默跳过 —— 请二选一：① 里面可能贴命令 ⇒ 加进 COMMAND_FENCE_TAGS；"
            f"② 它是数据/别的语言/纯输出 ⇒ 加进 NON_COMMAND_FENCE_REASONS 并写清理由。")


def _total_command_blocks() -> int:
    """全部被跟踪文档里的命令围栏总数（V-24 的"扫不到任何围栏"判据）。"""
    return sum(len(_fenced_blocks(doc)) for doc in DOCS)


def _scanned_command_count() -> int:
    """真正被送进解析器的文档命令条数 —— **计数类断言的唯一真源**（R-104/W-9）。

    任何写在散文里的数字（本文件模块 docstring、`CHANGELOG.md`）都要跟这个函数现算的值
    对得上：手写的 34/9 与 59 都漂过，而没有任何断言管它们。
    """
    return sum(1 for doc in DOCS for _lineno, line in _scan_lines(doc)
               if kernoscope_argv(line) is not None)


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


def strip_comment(s: str) -> str:
    """剥掉 shell 注释（引号里的 `#` 不算）。

    必须先剥注释再判"散文"：`kernoscope doctor   # 计数器门（1 秒）` 是**命令**，
    中文标点只出现在注释里。
    """
    out: list[str] = []
    quote = ""
    for i, ch in enumerate(s):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or s[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out)


def kernoscope_argv(line: str) -> list[str] | None:
    """把一条文档行切成 **kernoscope 的 argv**；不是"在调用我们"的一律返回 None。

    处理：行首包装（环境变量 / 提示符 / `sudo` / `time` / `uv run` / `python -m` /
    `xargs`）/ 命令替换 `$( )` / 反引号 / 管道 / `#` 注释 / `-- <被测命令>`（那部分交给
    别人跑）。

    关键：调用必须落在**行首**（`INVOCATION_RE` 是 `^` 锚定的）。散文里的半截命令
    （`入口命令：kernoscope … → kernoscope diff`）不产生 argv —— 第五轮 V-23 的假阳性
    就是这么来的。
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
    # 注释先剥掉再判散文（`kernoscope doctor  # 门（1 秒）` 里的标点在注释里）
    s = strip_comment(s)
    # 散文行不是命令（报告里 10 条假阳性的形状）：参数可以是中文，但中文**标点**说明
    # 这行是把命令写进了说明文字里（`… selftest / kernoscope diff。不要凭印象拼…`）。
    if PROSE_PUNCT_RE.search(s):
        return None
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
    # `kernoscope` 必须出现在**行首的调用位置**（或某种被承认的包装形式里）。
    # 用正则定位而不是"前缀词都要在白名单里"：白名单会漏掉 `python -m`、`xargs -n1`。
    m = INVOCATION_RE.match(s)
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
        # 带占位符的行是**示意**（读者要替换成真值）：先把占位符换成"形状合法"的夹具值，
        # 再交给真解析器 —— R-60 的第三个旁路就是在这里 `return None`，于是
        # `diff --packet <a.json> --packet <b.json>` 这种**值级**错误一律不查。
        # （`--ks-ledger-mb <每次 launch 的 MB>` 里那个值本来不是数字，换掉才谈得上校验。）
        parser0 = cli.subcommand_parser(sub) if sub else cli.build_parser()
        argv = _placeholder_values(argv, _choices_map(parser0))
        if any(PLACEHOLDER in a for a in argv):      # 还有换不掉的：位置未知，只能按示意放过
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


def _choices_map(parser) -> dict[str, list[str]]:
    """`--flag -> [合法取值]`（从真解析器的 action 里读，测试专用的小反射）。"""
    out: dict[str, list[str]] = {}
    for act in getattr(parser, "_actions", []):
        choices = getattr(act, "choices", None)
        if choices:
            for opt in getattr(act, "option_strings", []):
                out[opt] = list(choices)
    return out


def _placeholder_values(argv: list[str], choices: dict[str, list[str]] | None = None) -> list[str]:
    """把 `<占位符>` 换成形状合法的夹具值（只为让真解析器能走到**值级**校验）。

    取值规则按**上一个开关**判断：路径类开关给一个存在的临时风格路径、数值类开关给数字、
    `-k` 给 `regex:gemm`、其余给 `x`。换不掉的（例如 `--` 之后是读者自己的负载名）保持原样，
    由调用方按示意处理。
    """
    path_flags = {"--packet", "-o", "--report-md", "--export-ncu-rep", "--export-name-map",
                  "--ks-calibration", "--ks-sqlite", "--ks-ncu-rep", "--ks-memtrace",
                  "--ks-instr-hist", "--ks-cubin", "--ks-achieved-occupancy", "--log-file",
                  "--from", "--write-json"}
    numeric_flags = {"--ks-ledger-mb", "--ks-ledger-total-mb", "--ks-window-s", "--ks-warmup",
                     "--ks-tokens", "--ks-price-per-kwh", "--ks-price-per-hour",
                     "--ks-peak-gbps", "--ks-peak-tflops", "--ks-l2-bytes", "--ks-clock-mhz",
                     "--ks-max-accesses", "--ks-sm-count", "--max-records"}
    def repl(prev: str) -> str:
        if choices and prev in choices and choices[prev]:
            return str(choices[prev][0])          # 枚举开关：取真解析器的第一个合法值
        if prev in path_flags:
            return "/tmp/ks-fixture-placeholder.json"
        if prev in numeric_flags:
            return "1"
        if prev == "-k":
            return "regex:gemm"
        return "x"

    out: list[str] = []
    prev = ""
    for a in argv:
        if PLACEHOLDER in a:
            # `kernoscope_argv()` 已经把 `<...>` 归一成 PLACEHOLDER，所以这里只需处理它
            a = a.replace(PLACEHOLDER, repl(prev))
        out.append(a)
        prev = a
    return out


@needs_doc_scan_inputs
def test_every_documented_command_is_parseable():
    """文档里每条 kernoscope 命令都必须能被**真实解析器**接受。

    这条覆盖了旧门禁的三个洞：续行、`sudo`/`$( )`、以及"校验器不认识自家新子命令"。
    """
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    problems: list[str] = []
    seen = 0
    for doc in DOCS:
        for lineno, line in _scan_lines(doc):
            argv = kernoscope_argv(line)
            if argv is None:
                continue
            seen += 1
            why = _parse_problem(argv, sub_flags, main_flags)
            if why:
                problems.append(f"{doc.relative_to(ROOT)}:{lineno}: {why}\n"
                                f"    {line.strip()[:110]}")
    assert seen >= 20, f"只扫到 {seen} 条文档命令 —— 扫描范围或取词逻辑退化了"
    # P1 的第二半：**不许静默丢弃**。行首像命令、带长开关、却没被认成命令的行，说明
    # 包装形式没认全 —— 那种行里的编造开关永远不会被查（P1 的原形）。散文行不算
    # （V-23 的 10 条假阳性就是这么来的，所以这里先用 NEAR_START_RE 收窄）。
    suspicious = [f"{d.relative_to(ROOT)}:{ln}: {s.strip()[:90]}"
                  for d, ln, s in _unclassified_with_flags()]
    assert not suspicious, ("这些行行首像命令、带长开关，却没被当成命令解析"
                            "（包装形式没认全）：\n  - " + "\n  - ".join(suspicious))
    assert not problems, "文档里的命令跑不通（读者会照抄失败）：\n  - " + "\n  - ".join(problems)


def _unclassified_with_flags() -> list[tuple[Path, int, str]]:
    """行首像在调用我们（`NEAR_START_RE`）、带长开关、却没被 `kernoscope_argv` 认出来。

    只认**行首**的命令形状：散文里的 `入口命令：kernoscope … --brief` 不算 —— 报告里
    验收方写全仓散文扫描时产生的 10 条假阳性正是这一形状。
    """
    out = []
    for doc in DOCS:
        for lineno, line in _scan_lines(doc):
            s = strip_comment(line).strip()
            if not s or PROSE_PUNCT_RE.search(s) or not NEAR_START_RE.match(s):
                continue
            if kernoscope_argv(line) is None and LONG_FLAG_RE.search(s):
                out.append((doc, lineno, s))
    return out


@needs_doc_scan_inputs
def test_no_documented_subcommand_bypasses_its_real_parser():
    """R-60：**未解析的子命令数必须为 0**。

    旧门禁对子命令只比对开关集合就 `return None`，于是三类不合法命令被放行
    （`diff … --packet foo`、`selftest --ks-calibration …`、`capabilities -Z`）。
    这条把"每个子命令调用都真的过了它自己的解析器"变成可断言的事实。
    """
    unparsed: list[str] = []
    seen_sub = 0
    for doc in DOCS:
        for lineno, line in _scan_lines(doc):
            argv = kernoscope_argv(line)
            if not argv or argv[0] not in cli.SUBCOMMANDS:
                continue
            seen_sub += 1
            try:
                cli.subcommand_parser(argv[0]).parse_args(argv[1:])
            except SystemExit as e:
                if e.code not in (0, None):
                    unparsed.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()[:80]}")
    assert seen_sub > 0, "文档里一条子命令都没有？扫描范围退化了"
    assert unparsed == [], ("这些子命令没有通过它自己的真实解析器：\n  - " + "\n  - ".join(unparsed))


@needs_doc_scan_inputs
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


@needs_doc_scan_inputs
def test_documented_commands_come_from_all_over_the_repo():
    """扫描范围必须覆盖 skills/ 与 eval/（旧门禁只扫 5 份根目录文档）。"""
    roots = {str(d.relative_to(ROOT)).split("/")[0] for d in DOCS}
    assert "skills" in roots and "eval" in roots, sorted(roots)


def test_fence_scanner_takes_console_text_and_untagged_blocks(tmp_path):
    """V-23 的负样本：坏命令塞进 ```console/```text/**无标签**围栏里必须被抓到。

    旧实现只认 ```bash/```sh —— 报告实测：同一条 `--smoke` 注入，bash 红、其余全绿。
    同时钉住三条**明确写清**的排除规则（JSON/别的语言、整块输出、行首不是命令），
    它们必须是在"排除"而不是在"静默丢"。
    """
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    bad = "kernoscope --set basic --smoke -- x.py"
    for tag in ("bash", "sh", "console", "text", ""):
        doc = tmp_path / f"doc-{tag or 'untagged'}.md"
        doc.write_text(f"```{tag}\n{bad}\n```\n", encoding="utf-8")
        blocks = _fenced_blocks(doc)
        assert len(blocks) == 1, (tag, blocks)
        argv = kernoscope_argv(_logical_lines(blocks[0][1])[0])
        assert argv is not None, tag
        assert _parse_problem(argv, sub_flags, main_flags), f"{tag} 围栏里的坏命令被放行了"
    # 排除①：数据/别的语言的围栏不是命令（标签排除，逐条有理由）
    for tag in ("json", "jsonc", "yaml", "python", "cuda", "powershell"):
        doc = tmp_path / f"data-{tag}.md"
        doc.write_text(f'```{tag}\n{{"cmd": "{bad}"}}\n```\n', encoding="utf-8")
        assert _fenced_blocks(doc) == [], tag
    # 排除②：整块像 JSON/数组输出（`console`/`text` 里贴机器输出很常见）
    out = tmp_path / "output.md"
    out.write_text(f'```console\n{{"argv": "{bad}"}}\n```\n', encoding="utf-8")
    assert _fenced_blocks(out) == []
    # 排除③：**只在围栏内**取词 —— 围栏外的散文行（半截命令）不算命令
    prose = tmp_path / "prose.md"
    prose.write_text(f"就这样写：{bad}（这行在围栏外）\n", encoding="utf-8")
    assert _fenced_blocks(prose) == []
    # 围栏内、但行首是散文的行也不算命令（`入口命令：kernoscope …`）
    mixed = tmp_path / "mixed.md"
    mixed.write_text("```text\n入口命令：kernoscope capabilities --brief → kernoscope diff\n```\n",
                     encoding="utf-8")
    blocks = _fenced_blocks(mixed)
    assert len(blocks) == 1
    assert kernoscope_argv(_logical_lines(blocks[0][1])[0]) is None
    # 反面：`$ ` 提示符之后**才是**命令（console 围栏的标准写法，必须被取到）
    prompt = tmp_path / "prompt.md"
    prompt.write_text("```console\n$ kernoscope capabilities --brief\n```\n", encoding="utf-8")
    blocks = _fenced_blocks(prompt)
    assert kernoscope_argv(_logical_lines(blocks[0][1])[0]) == ["capabilities", "--brief"]


def test_unknown_fence_tags_are_not_silently_skipped(tmp_path):
    """W-11/R-105 的负样本：白名单外的标签不许被静默当成"非命令围栏"。

    第五轮 V-23 把命令围栏扩到 `console`/`text`/无标签，但 ```output / ```shellsession /
    ```terminal 这些**白名单外**的标签仍然被静默跳过 —— 报告实测：同一条编造的 `--smoke`
    塞进去，门禁全绿。这里把两半都钉死：

      ① `output`/`shellsession`/`terminal` 进了命令白名单 ⇒ 里面的坏命令必须被抓到；
      ② 真没分类的标签（```rust/```mermaid…）不许静默跳过，而是**报出来**要求二选一。
    """
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    bad = "kernoscope --set basic --smoke -- x.py"
    for tag in ("output", "shellsession", "terminal", "shell-session"):
        doc = tmp_path / f"fence-{tag}.md"
        doc.write_text(f"```{tag}\n{bad}\n```\n", encoding="utf-8")
        assert _all_fence_tags(doc) == [tag]
        blocks = _fenced_blocks(doc)
        assert len(blocks) == 1, (tag, blocks)
        argv = kernoscope_argv(_logical_lines(blocks[0][1])[0])
        assert argv is not None, tag
        assert _parse_problem(argv, sub_flags, main_flags), f"{tag} 围栏里的坏命令被放行了"
    # ② 没分类的标签：不是"跳过"，是"报错"。旧实现的 `_fenced_blocks` 会安静地返回 []。
    for tag in ("rust", "mermaid", "output-text"):
        problem = _fence_tag_problem(tag)
        assert problem, f"```{tag} 没被分类，却没有任何报错（静默跳过还在）"
        assert tag in problem and "静默" in problem
        doc = tmp_path / f"unknown-{tag}.md"
        doc.write_text(f"```{tag}\n{bad}\n```\n", encoding="utf-8")
        assert _all_fence_tags(doc) == [tag]      # 标签被看见了，只是没被分类


def test_fence_exemption_registry_is_explicit(tmp_path):
    """豁免表本身也要能被查：两张表不许重叠、每条豁免必须写理由、且真的被排除。"""
    assert not (COMMAND_FENCE_TAGS & set(NON_COMMAND_FENCE_REASONS)), \
        "同一个标签不许既当命令围栏又被豁免"
    for tag, why in NON_COMMAND_FENCE_REASONS.items():
        assert why.strip(), f"```{tag} 的豁免没写理由"
        assert _fence_tag_problem(tag) is None
    for tag in ("json", "jsonc", "yaml", "python", "cuda", "powershell"):
        assert tag in NON_COMMAND_FENCE_REASONS, f"```{tag} 是数据/别的语言，必须有理由地豁免"
        doc = tmp_path / f"data-{tag}.md"
        doc.write_text('```%s\n{"cmd": "kernoscope --smoke"}\n```\n' % tag, encoding="utf-8")
        assert _fenced_blocks(doc) == [], tag


@needs_doc_scan_inputs
def test_every_fence_tag_in_the_docs_is_explicitly_classified():
    """W-11 的正面门禁：被跟踪文档里**每一个**围栏标签都必须被显式分类。

    这条是"穷尽性"本身：新增一个没分类的标签（无论有意还是无意）都会红，
    而不是像旧实现那样安静地不扫它。
    """
    bad: list[str] = []
    for doc in DOCS:
        for tag in dict.fromkeys(_all_fence_tags(doc)):     # 去重、保序
            problem = _fence_tag_problem(tag)
            if problem:
                bad.append(f"{doc.relative_to(ROOT)}: {problem}")
    assert not bad, "\n  - ".join(bad)


def test_registered_negative_examples_are_really_unparseable():
    """`NEGATIVE_EXAMPLES` 不许长成万能橡皮擦：登记项必须仍然存在**且确实不合法**，
    而且**文档自己要说它是错的**（±20 行内有证据）。豁免是逐行的 —— 同围栏里多出来的
    坏命令照样要被 `test_every_documented_command_is_parseable` 查到。"""
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    assert NEGATIVE_EXAMPLES, "反面教材登记表为空 —— 那条无标签围栏里的错例不该被静默跳过"
    for doc, snippets, why in NEGATIVE_EXAMPLES:
        path = ROOT / doc
        assert path.exists(), f"登记的反面教材所在文档不存在：{doc}"
        assert why.strip(), f"{doc} 的登记项没写理由"
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = _fenced_blocks(path)
        for snippet in snippets:
            assert snippet in text, f"{doc} 里找不到登记的片段（登记表过期）：{snippet!r}"
            hit = next(((s, line) for s, b in blocks for line in _logical_lines(b)
                        if snippet in line), None)
            assert hit, f"{doc} 里那一段没有落在命令围栏里（登记的排除项过期了）：{snippet!r}"
            _start, line = hit
            assert _registered_negative_line(path, line), "登记行没有被 `_scan_lines` 认出来"
            argv = kernoscope_argv(line)
            assert argv is not None and _parse_problem(argv, sub_flags, main_flags), \
                f"{doc} 里登记的片段其实能通过解析器 —— 它不该出现在反面教材表里：{line!r}"
        # 文档自己的证据：那些行附近必须有一句"这是错的"
        start, content = next(((s, b) for s, b in blocks
                               if any(x in b for x in snippets)), (0, ""))
        lines = text.splitlines()
        lo = max(0, start - 1 - NEGATIVE_EVIDENCE_WINDOW)
        hi = min(len(lines), start - 1 + len(content.splitlines()) + NEGATIVE_EVIDENCE_WINDOW)
        window = "\n".join(lines[lo:hi])
        assert NEGATIVE_EVIDENCE_RE.search(window), (
            f"{doc} 里登记的排除项附近没有「这是错例」的证据"
            f"（±{NEGATIVE_EVIDENCE_WINDOW} 行）—— 排除项不许是无依据的橡皮擦")


def test_empty_scan_input_fails_unless_explicitly_opted_out(monkeypatch):
    """V-24：空输入集必须 **fail**；唯一出口是显式 `KS_ALLOW_EMPTY_DOC_SCAN=1`（skip）。"""
    monkeypatch.delenv(ALLOW_EMPTY_DOC_SCAN, raising=False)
    with pytest.raises(pytest.fail.Exception) as e:
        _empty_scan_input("模拟：这里没有 .git")
    # 失败信息必须把出口写清楚（否则读者只会看到"红了"，不知道还能怎么办）
    assert ALLOW_EMPTY_DOC_SCAN in str(e.value)

    monkeypatch.setenv(ALLOW_EMPTY_DOC_SCAN, "1")
    with pytest.raises(pytest.skip.Exception):
        _empty_scan_input("模拟：这里没有 .git")

    # 只有 "1" 算接受降级：`=0` / 空串 / 别的值都还是 fail（不许用"设了就算同意"蒙混）
    for value in ("0", "", "yes", "true"):
        monkeypatch.setenv(ALLOW_EMPTY_DOC_SCAN, value)
        with pytest.raises(pytest.fail.Exception):
            _empty_scan_input("模拟：这里没有 .git")


def test_scan_inputs_are_not_empty_here():
    """在这个仓库里跑时，输入集必须真的非空（空的那个分支由上面两条守）。"""
    why = _scan_inputs_problem()
    if why is not None:
        _empty_scan_input(why)
    assert DOCS and _total_command_blocks() > 0


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
    """在文档的命令围栏里找到含 `needle` 的**逻辑行**（续行已拼接）。"""
    path = ROOT / doc
    for _lineno, line in _scan_lines(path):
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


@needs_doc_scan_inputs
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


@needs_doc_scan_inputs
def test_every_doc_command_is_either_parsed_or_asserted():
    """把"覆盖率"写成事实而不是印象：每条命令都过解析器，其中读者面那几条断言结论。"""
    parsed = _scanned_command_count()
    assert parsed >= 30, f"只扫到 {parsed} 条可检命令"
    assert len(DOC_VERBATIM) < parsed, "登记数不该超过可检命令数（那是重复计数）"


@needs_doc_scan_inputs
def test_prose_counts_match_the_scan():
    """R-104/W-9：本文件 docstring 里的计数必须是**现算**的，不是手写的。

    旧 docstring 写 "34 条 / 9 条 / 比例 11/57"，而扫描器实测 56/11 —— 两处都手写、
    无断言绑定，于是各自漂。这条把散文里的数字绑到 `_scanned_command_count()` 与
    `len(DOC_VERBATIM)` 上：写错了会在这里红，并且直接告诉你活值。
    """
    parsed = _scanned_command_count()
    verbatim = len(DOC_VERBATIM)
    doc = __doc__ or ""
    want = {
        f"当前 {parsed} 条": "可检命令数",
        f"读者面那 **{verbatim} 条**": "读者面逐字执行条数",
        f"比例是 **{verbatim}/{parsed}**": "逐字执行比例",
    }
    missing = [f"{what}：docstring 里找不到 {needle!r}" for needle, what in want.items()
               if needle not in doc]
    assert not missing, (
        "文档命令门禁的计数是手写的、已经漂了（W-9/R-104）：\n  - " + "\n  - ".join(missing)
        + f"\n现算值：可检命令 {parsed} 条、读者面逐字执行 {verbatim} 条、比例 {verbatim}/{parsed}。"
        + "\n请按现算值改 tests/test_documented_commands.py 的模块 docstring —— "
          "数字由这条测试守，不许手写。")


@needs_doc_scan_inputs
def test_changelog_command_counts_match_the_scan():
    """R-104/W-9：`CHANGELOG.md` 里的命令计数必须与扫描器现算的值一致。

    旧 CHANGELOG 写"（当前 68 条）… 比例 11/59"，实测是 **56**（读者面 11 条没漂，
    分母漂了）。CHANGELOG 不许手改数字 —— 这条给出**可以直接粘贴**的正确句子。
    """
    parsed = _scanned_command_count()
    verbatim = len(DOC_VERBATIM)
    subs = len(cli.SUBCOMMANDS)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # 只在**「文档命令门禁」那一段**里找这些数字：R-88④ 的教训是"整文件子串匹配"会被
    # 文件里任何一处（甚至注释）满足 —— 数字写在该在的那一段才算数。
    lines = changelog.splitlines()
    idx = next((i for i, l in enumerate(lines) if "文档命令门禁" in l), None)
    assert idx is not None, "CHANGELOG 里找不到「文档命令门禁」那一段（登记表过期？）"
    block: list[str] = []
    for l in lines[idx:]:
        if block and (l.startswith("- ") or l.startswith("#") or not l.strip()):
            break
        block.append(l)
    para = "\n".join(block)
    needles = {
        f"（当前 {parsed} 条）": "可检命令总数",
        f"读者面 {verbatim} 条": "读者面逐字执行条数",
        f"比例 {verbatim}/{parsed}": "逐字执行比例",
        f"{subs} 个子命令": "子命令数量",
    }
    missing = [f"{what}：CHANGELOG「文档命令门禁」段里找不到 {needle!r}"
               for needle, what in needles.items() if needle not in para]
    assert not missing, (
        "CHANGELOG 的文档命令计数与扫描器现算值不一致（W-9/R-104）：\n  - "
        + "\n  - ".join(missing)
        + f"\n现算值：可检命令 {parsed} 条、读者面逐字执行 {verbatim} 条、比例 {verbatim}/{parsed}、"
          f"子命令 {subs} 个（子命令数来自 `cli.SUBCOMMANDS`）。"
        + "\n请把 CHANGELOG「文档命令门禁」那段的前两行**整体替换**为：\n"
        + "- 文档命令门禁：**全部**可检命令（当前 %d 条）都交给**真实解析器**（%d 个子命令各有自己的\n"
          "  `build_parser()`，是 `--help`／门禁／测试共用的唯一真源；续行、`sudo`、`$( )`、\n"
          "  `python -m kernoscope`、`xargs … kernoscope` 都覆盖，且**不许静默丢弃**）；\n"
          "  其中**读者面 %d 条**逐字执行并断言结论（比例 %d/%d，不是\"每条都断言\"）。"
        % (parsed, subs, verbatim, verbatim, parsed))


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
    """README 教人用 --metrics 窄化输出；写错名字不能得到"看起来成功"的空报告。

    `dram__bytes.sum` 是**已知的 ncu 名字**：D4① 之后它会解析成替代口径并给出说明；
    随机名字仍然走"一个都不存在"。两种都不允许静默。"""
    rc, out, err = _run(["--ks-synthetic", "--metrics", "dram__bytes.sum",
                         "-k", "regex:gemm"])
    assert rc == 0
    assert "替代口径 est__dram_bytes_total" in err and "est__dram_bytes_total" in out
    rc2, _out2, err2 = _run(["--ks-synthetic", "--metrics", "totally_made_up__metric",
                             "-k", "regex:gemm"])
    assert rc2 == 0 and "一个都不存在" in err2


def test_placeholder_lines_are_still_parsed_for_value_level_errors():
    """R-60 的第三个旁路：带 `<占位符>` 的命令此前在真解析**之前**就 `return None`，
    于是同一行里的**值级**错误一律放行。现在先把占位符换成形状合法的夹具值（枚举取真解析器
    的第一个合法值），再交给真解析器。
    """
    sub_flags = _subcommand_flags()
    main_flags = _main_flags()
    # 占位符换成夹具值之后是合法的（示意行不该变红）
    assert _parse_problem(["--ks-synthetic", "--set", PLACEHOLDER], sub_flags, main_flags) is None
    assert _parse_problem(["--ks-synthetic", "--ks-ledger-mb", PLACEHOLDER],
                          sub_flags, main_flags) is None
    # 但同一行的结构性/值级错误必须被抓到（旧实现整行放行）
    assert _parse_problem(["diff", "--packet", PLACEHOLDER], sub_flags, main_flags) is not None
    assert _parse_problem(["--ks-synthetic", "--plan", "--bogus-flag", "1"],
                          sub_flags, main_flags) is not None
