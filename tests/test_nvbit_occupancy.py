"""NVBit 达到-occupancy 工具（被测 kernel 零源码改动）的契约测试。

这些测试**不需要 GPU、也不需要 NVBit**（本机就没有）：它们守的是"别的部分靠什么才能
继续工作"的那几条硬契约 ——

1. **格式契约**：`tools/probes/nvbit_occupancy/` 写出来的记录必须与
   `src/kernoscope/analyze/occupancy.py`（既有分析路径，本工具不许改它）逐字兼容：
   首行 `# ks-occupancy-records v1`、列头 `smid,g0,g1`。这里**从真源取字面量**
   （`occupancy.PROBE_FORMAT`），不手抄。
2. **脚本契约**：`collect.sh` 能过 `bash -n`，带着与 `nvbit_memtrace/collect.sh`
   同一套版本纪律（`--resolve-only`、版本**从实物读**、脚本里不写死版本号），
   并且显式指了树却没有产物时 **exit 2 并列出找过的路径**。
3. **黄金端到端**：一份合成记录 → 真的 `kernoscope occupancy --from … --write-json …`
   代码路径 → 断言算出来的数。这条证明"格式契约"在没有 GPU 的情况下也成立。
4. **纪律契约**：工具在覆盖率不足时打的 `状态 拒绝` 标记，与 `collect.sh` 里 grep 的
   标记必须是同一个字符串（两边漂了就红）。

需要 NVBit 才能做的检查一律 `pytest.skip` 并写清原因 —— **绝不假装成功**。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from kernoscope.analyze.occupancy import PROBE_FORMAT, load_probe_records
from kernoscope.cli import main

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "tools" / "probes" / "nvbit_occupancy"
HOST_SRC = PROBE_DIR / "occupancy.cu"
DEVICE_SRC = PROBE_DIR / "inject_funcs.cu"
COMMON_H = PROBE_DIR / "ks_occ_common.h"
COLLECT = PROBE_DIR / "collect.sh"
MAKEFILE = PROBE_DIR / "Makefile"
README = PROBE_DIR / "README.md"
ANALYZER = ROOT / "src" / "kernoscope" / "analyze" / "occupancy.py"

# 工具与脚本之间的状态标记（两处必须是同一个字符串，见 `test_..._markers_agree`）
STATUS_REFUSED = "KS-OCCUPANCY: 状态 拒绝"
STATUS_USABLE = "KS-OCCUPANCY: 状态 可用"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1) 格式契约：工具写的东西必须与分析器认的东西是同一套字面量
# ---------------------------------------------------------------------------
def test_records_header_literal_matches_the_analyzer_constant():
    """首行格式标识从 `occupancy.PROBE_FORMAT` 取 —— 两边漂开就红。"""
    assert f"# {PROBE_FORMAT}" in _read(HOST_SRC), (
        f"occupancy.cu 必须写出 `# {PROBE_FORMAT}`（与 analyze/occupancy.py 的 "
        "PROBE_FORMAT 同一个字面量），否则解析器会以「缺少首行」拒绝整份记录")


def test_column_header_literal_matches_the_analyzer_message():
    """列头 `smid,g0,g1` 必须是分析器解析/报错里用的那一串。"""
    col = "smid,g0,g1"
    assert col in _read(ANALYZER), f"分析器里应当出现列头字面量 {col!r}"
    assert col in _read(HOST_SRC), f"工具必须写出列头 {col!r}"
    # 老源码探针是同一个格式的真源之一：三边一致才算契约没漂
    assert col in _read(ROOT / "tools" / "probes" / "occupancy_probe.cu")


def _strip_cpp_comments(src: str) -> str:
    """去掉注释后的源码（注释里**提到** %clock64 是说明，不是使用它）。

    这两份源码里没有含 `//` 的字符串字面量，所以这个朴素的剥离够用。
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//.*", "", src)


def test_tool_reads_globaltimer_and_never_clock64():
    """跨 SM 比较只能用 %globaltimer；%clock64 是每 SM 各自原点（真机实测 2e12 极差）。"""
    src = _strip_cpp_comments(_read(DEVICE_SRC))
    assert "%globaltimer" in src
    assert "%clock64" not in src, (
        "inject_funcs.cu 里不许**使用** %clock64：它每 SM 各自原点，跨 SM 比出来的区间是错的")
    assert "%smid" in src


def test_injected_function_names_are_defined_device_side():
    """主机侧插入的调用名 + 形参表，必须与设备侧定义严格对上（NVBit 按名字找函数）。

    名字/形参写错时现象是运行时 "instrumentation function ... not found in binary!" 或
    参数错位 —— 这条在没有 NVBit 的机器上就能拦住它。
    """
    host = _read(HOST_SRC)
    device = _read(DEVICE_SRC)
    inserted = set(re.findall(r'nvbit_insert_call\(\s*\w+\s*,\s*"([^"]+)"', host))
    assert inserted, "occupancy.cu 里没找到 nvbit_insert_call 的调用名"
    assert inserted == {"ks_occ_enter", "ks_occ_exit"}, inserted

    # 每次插入一个调用 + 恰好一个 const_val64 实参（设备侧就是一个 uint64_t）
    assert len(re.findall(r"nvbit_insert_call\(", host)) == len(inserted)
    assert len(re.findall(r"nvbit_add_call_arg_const_val64\(", host)) == len(inserted)

    for name in sorted(inserted):
        sig = f'extern "C" __device__ __noinline__ void {name}(uint64_t cfg_ptr)'
        assert sig in device, f"inject_funcs.cu 里没有对得上的设备侧定义：{sig}"


def test_makefile_dry_run_expands_to_the_nvbit_cmake_helper(tmp_path):
    """`make -n` 能过（Makefile 语法/制表符没坏），且用的是 NVBit 的 CMake 助手。"""
    make = shutil.which("make")
    if make is None:
        pytest.skip("本机没有 make —— 只校验 Makefile 的文本")
    r = subprocess.run([make, "-n", "NVBIT_PATH=" + str(tmp_path), "all"],
                       capture_output=True, text=True, timeout=60, cwd=str(PROBE_DIR))
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "cmake -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES=native" in r.stdout
    assert "ks_occupancy.so" in r.stdout


def test_coverage_discipline_markers_agree_between_tool_and_script():
    """工具打的标记与 collect.sh grep 的标记必须是同一个字符串（否则拒绝会静默漏过）。"""
    assert STATUS_REFUSED in _read(HOST_SRC)
    assert STATUS_USABLE in _read(HOST_SRC)
    assert STATUS_REFUSED in _read(COLLECT), (
        "collect.sh 必须 grep 工具打的『状态 拒绝』标记，否则覆盖率不足时会照样往下走")
    # 覆盖率阈值与分子/分母都必须写进文件头（否则事后无法复核这份数据能不能用）
    host = _read(HOST_SRC)
    for key in ("# grid_blocks ", "# cta_rows ", "# coverage ", "# coverage_ok ", "# min_coverage "):
        assert key in host, f"记录头部少了 {key.strip()!r}（覆盖率纪律要求可复核）"


# ---------------------------------------------------------------------------
# 2) collect.sh：语法 + 版本纪律 + 显式缺失时的行为
# ---------------------------------------------------------------------------
def test_collect_script_syntax_and_version_discipline():
    r = subprocess.run(["bash", "-n", str(COLLECT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"bash -n 失败：{r.stderr}"

    text = _read(COLLECT)
    assert "--resolve-only" in text, "collect.sh 必须支持 --resolve-only（机器可读两行）"
    assert "从实物读" in text, "collect.sh 必须写明版本是『从实物读』的"
    assert "n/a" in text, "版本读不出时必须写 n/a（绝不编一个版本）"
    # 版本读取只有一处真源：复用 memtrace 的 nvbit_version.sh，不另抄一份
    assert "nvbit_version.sh" in text
    assert 'occupancy_so_version: %s' in text, (
        "resolve-only 必须打出机器可读的 occupancy_so_version 行")
    assert "${NV_V:-n/a}" in text, (
        "resolve-only 的版本必须来自实物（${NV_V:-n/a}），不能是常量")
    # 采集必须带上 NO_EAGER_LOAD=0 / NOBANNER=1（1.8 起默认懒加载会让插桩错过时机）
    for marker in ("NO_EAGER_LOAD=0", "NOBANNER=1", "CUDA_INJECTION64_PATH", "LD_PRELOAD"):
        assert marker in text, f"collect.sh 少了 {marker}"
    assert '"${1:-}" = "--"' in text, "被测命令必须走 `--` 分隔符"
    assert "KS_OCC_OUT" in text, "输出路径要能从环境变量给"


def test_collect_script_contains_no_hard_coded_version_string():
    """脚本里不许把 NVBit 版本号写死 —— 硬编码的版本正是这份纪律要防的东西。"""
    text = _read(COLLECT)
    found = re.findall(r"\d+\.\d+\.\d+", text)
    assert not found, (
        f"collect.sh 里出现了写死的版本号形状 {found} —— 版本只许从实物读"
        "（nvbit_version.sh），脚本里不留常量")


def test_collect_script_is_executable_and_git_mode_is_755_if_tracked():
    mode = COLLECT.stat().st_mode
    assert mode & stat.S_IXUSR, "collect.sh 必须带可执行位（它要被直接 ./collect.sh 调用）"
    assert os.access(COLLECT, os.X_OK)

    # 在 git 仓库里、且文件已被跟踪时，额外核对 git 记的权限位（"它 is in git" 的那条）
    shown = subprocess.run(["git", "ls-files", "-s", "--", str(COLLECT.relative_to(ROOT))],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
    if shown.returncode != 0:
        pytest.skip("不在 git 仓库里（导出的源码包）—— 只校验工作区的可执行位")
    if not shown.stdout.strip():
        pytest.skip("collect.sh 还没被 git 跟踪（本轮刚创建）—— 提交后这条会校验 100755")
    assert shown.stdout.split()[0] == "100755", (
        f"git 记的模式不是 100755：{shown.stdout.strip()}（chmod +x 丢了）")


def _run_collect(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("NVBIT_PATH", "NVBIT_OCCUPANCY_SO", "KS_OCC_OUT",
                        "KS_OCC_RESOLVE_ONLY", "LD_PRELOAD")}
    env.update(env_extra or {})
    return subprocess.run(["bash", str(COLLECT), *args], capture_output=True, text=True,
                          env=env, timeout=120, cwd=str(PROBE_DIR))


def test_collect_resolve_only_never_invents_a_version(tmp_path):
    """`--resolve-only` 报的是**实物**：路径 + 版本（读不出就 n/a，绝不编）。"""
    so = tmp_path / "ks_occupancy.so"
    so.write_bytes(b"fake nvbit tool, no version string\n")
    r = _run_collect("--resolve-only", env_extra={"NVBIT_OCCUPANCY_SO": str(so)})
    assert r.returncode == 0, (r.stdout, r.stderr)
    fields = dict(re.findall(r"^(occupancy_so(?:_version)?): (.*)$", r.stdout, re.M))
    assert fields.get("occupancy_so") == str(so)
    assert fields.get("occupancy_so_version") == "n/a", (
        "读不出实物版本时必须写 n/a，不许编一个：\n" + r.stdout)


def test_collect_reports_explicit_missing_tree_with_the_searched_paths(tmp_path):
    """显式指了 NVBIT_PATH 却没有产物 ⇒ exit 2，且把**实际找过的路径**列出来。"""
    tree = tmp_path / "empty-tree"
    (tree / "tools").mkdir(parents=True)
    r = _run_collect("--resolve-only", env_extra={"NVBIT_PATH": str(tree)})
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert not re.search(r"^occupancy_so: ", r.stdout, re.M), (
        "找不到产物时 stdout 里不许有能被误当成答案的行：\n" + r.stdout)
    for layout in ("tools/ks_occupancy/build/ks_occupancy.so",
                   "tools/ks_occupancy/ks_occupancy.so"):
        assert str(tmp_path / "empty-tree" / layout) in r.stderr, (
            f"报错必须列出实际找过的路径（缺 {layout}）：\n" + r.stderr)
    assert "实际找过的路径" in r.stderr


def test_collect_exits_3_when_the_tool_produced_no_records(tmp_path):
    """被测程序跑成功、但工具没写出文件（覆盖率不足就会这样）⇒ exit 3，不是 0。

    这里用一个**假的目标程序**：它在退出码上成功，但不写记录文件 —— 正是"部分数据
    不许静默变成分布"那条纪律在脚本层的表现。
    """
    fake_so = tmp_path / "ks_occupancy.so"
    fake_so.write_bytes(b"not a real nvbit tool\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cuobjdump = bindir / "cuobjdump"
    cuobjdump.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cuobjdump.chmod(0o755)
    target = tmp_path / "target.sh"
    target.write_text("#!/bin/sh\necho 'KS-OCCUPANCY: 状态 拒绝'\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)

    env = {"NVBIT_OCCUPANCY_SO": str(fake_so), "KS_OCC_OUT": str(tmp_path / "rec.txt"),
           "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    r = _run_collect("--", str(target), env_extra=env)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert not (tmp_path / "rec.txt").exists()
    assert "不能" in r.stderr or "拒绝" in r.stderr


# ---------------------------------------------------------------------------
# 3) 黄金端到端：合成记录 → 真的 `kernoscope occupancy` 代码路径 → 断言数字
# ---------------------------------------------------------------------------
# 两个 SM 上的四条区间（手算见断言）：
#   SM0: [1000,2000] 与 [1500,2500] → 重叠段并发 2
#   SM1: [1000,2000] 与 [3000,4000] → 全程并发 1
#   平均并发 = (2000/1500 + 2000/3000) / 2 = 1.0；峰值 = 2
#   occupancy = 1.0 × 8 warps/block ÷ 64 warps/SM（sm_100） = 12.5%
GOLDEN_RECORDS = f"""# {PROBE_FORMAT}
# kernel golden_kernel
# arch sm_100
# threads_per_block 256
# sm_count 4
# grid_blocks 4
# theoretical_occupancy_pct 50.0
# cta_rows 4
# coverage 1.0000
# coverage_ok 1
# min_coverage 0.9500
# launches_traced 1
# config_mismatch 0
# instrumented_instrs 2
# allow_partial 0
# source nvbit-sass-injection
smid,g0,g1
0,1000,2000
0,1500,2500
1,1000,2000
1,3000,4000
"""


def test_golden_records_file_is_parsed_by_the_real_analyzer(tmp_path):
    """工具多写的那些头部键（coverage/coverage_ok/…）不许把解析器弄坏。"""
    src = tmp_path / "golden.txt"
    src.write_text(GOLDEN_RECORDS, encoding="utf-8")
    rec = load_probe_records(str(src))
    assert rec.kernel == "golden_kernel"
    assert rec.arch == "sm_100"
    assert rec.threads_per_block == 256
    assert rec.sm_count == 4
    assert len(rec.records) == 4


def test_golden_end_to_end_cli_numbers(tmp_path, capsys):
    """真的走 `kernoscope occupancy --from … --write-json …`，断言它算出来的数。

    这条是"格式契约"在没有 GPU 时的证明：头部里的 arch / threads_per_block 只要有一个
    没被解析到，12.5% 这个数就变了。
    """
    src = tmp_path / "golden.txt"
    src.write_text(GOLDEN_RECORDS, encoding="utf-8")
    dst = tmp_path / "occ.json"
    rc = main(["occupancy", "--from", str(src), "--write-json", str(dst), "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json.loads(out)
    assert result["kernel"] == "golden_kernel"
    assert result["provenance"] == "measured"
    assert result["arch"] == "sm_100"
    assert result["threads_per_block"] == 256
    assert result["sm_count"] == 4
    assert result["grid_blocks"] == 4
    assert result["avg_concurrent_blocks"] == pytest.approx(1.0)
    assert result["peak_concurrent_blocks"] == 2.0
    assert result["occupancy_pct"] == pytest.approx(12.5)   # 1.0 × 8 / 64
    assert result["theoretical_occupancy_pct"] == pytest.approx(50.0)

    payload = json.loads(dst.read_text(encoding="utf-8"))
    assert payload == {"golden_kernel": {"occupancy_pct": 12.5, "avg_concurrent_blocks": 1.0,
                                         "peak_concurrent_blocks": 2, "samples": 2}}


def test_golden_low_coverage_file_is_still_accepted_by_the_analyzer(tmp_path):
    """**分析侧看不见覆盖率**（既有路径不许改）—— 所以拒绝只能发生在产出文件之前。

    这条把那个事实钉住：带 `# coverage_ok 0` 的文件照样会被算出一个数。工具因此默认
    **不写**这种文件（见 occupancy.cu 的纪律与 README 的「覆盖率纪律」）。
    """
    src = tmp_path / "partial.txt"
    src.write_text(GOLDEN_RECORDS.replace("# coverage_ok 1", "# coverage_ok 0")
                   .replace("# coverage 1.0000", "# coverage 0.2500"),
                   encoding="utf-8")
    rec = load_probe_records(str(src))
    assert rec.header["coverage_ok"] == "0"       # 标记在，但分析器不据此拒绝
    assert len(rec.records) == 4
    assert "coverage_ok" not in _read(ANALYZER), (
        "既有分析路径不校验覆盖率（本工具不许改它）—— 若哪天校验了，这条测试该更新，"
        "并且工具侧的『默认不写文件』策略可以放宽")


# ---------------------------------------------------------------------------
# 4) 构建与文档：Makefile 认 NVBIT_PATH/ARCH；README 必须自曝"未在真机验证"
# ---------------------------------------------------------------------------
def test_makefile_respects_nvbit_path_and_arch():
    text = _read(MAKEFILE)
    assert "NVBIT_PATH ?=" in text and "ARCH ?=" in text
    assert "TOOL := ks_occupancy" in text and "SO := $(TOOL).so" in text
    # 依赖列表必须包含全部源码（少一个就会出现"改了源码却用了旧产物"的假绿）
    for src in ("CMakeLists.txt", "inject_funcs.cu", "occupancy.cu", "ks_occ_common.h"):
        assert src in text, f"Makefile 的依赖/拷贝列表里少了 {src}"
    # 架构必须按 ARCH 构建，而且**必须是裸数字**：真机实测（AutoDL/CUDA 13/NVBit 1.8.1）
    # 传 `sm_89` 会被助手拼成 `compute_sm_89`，nvcc 直接 `Unsupported gpu architecture`。
    assert "NVCC_ARCH := $(patsubst sm_%,%,$(ARCH))" in text, \
        "Makefile 必须把 `sm_89` 这种写法剥成裸数字（否则 nvcc 报 compute_sm_89）"
    assert "-DNVBIT_TOOL_CUDA_ARCHITECTURES=$(NVCC_ARCH)" in text, (
        "注入代码必须按 ARCH 构建（架构不匹配是「编好了但跑不出记录」的常见原因）")


def test_readme_documents_the_two_hooks_and_the_coverage_refusal():
    text = _read(README)
    assert "未在真机验证" in text, "README 必须自曝『未在真机验证』"
    assert "early exit" in text or "early-exit" in text, "README 必须说清出口钩子对 early exit 的低估"
    assert "KS_OCC_MIN_COVERAGE" in text and "95" in text, "README 要写明覆盖率阈值可配（默认 95%）"
    assert "拒绝" in text, "README 必须写明覆盖率不足时该拒绝给数（部分数据不许变成分布）"
    assert "occupancy_probe.cu" in text, "README 要指路到可对拍的老探针"
    assert "collect.sh" in text


def test_readme_commands_are_parseable_by_the_real_parsers():
    """README 里的 `kernoscope …` 命令要能被真解析器接受（照抄不会因为开关写错而失败）。

    只做**解析**，不执行（这些命令要真的采集）。
    """
    import kernoscope.cli as cli

    sub_flags = {sub: {s for a in cli.subcommand_parser(sub)._actions
                       for s in a.option_strings} for sub in cli.SUBCOMMANDS}
    main_flags = {s for a in cli.build_parser()._actions for s in a.option_strings}

    seen = 0
    for line in _read(README).splitlines():
        line = line.strip()
        if not line.startswith("kernoscope "):
            continue
        argv = line.replace("kernoscope", "", 1).split()
        if "--" in argv:
            argv = argv[:argv.index("--")]
        sub = argv[0] if argv and not argv[0].startswith("-") else ""
        assert not sub or sub in cli.SUBCOMMANDS, f"README 里出现了不存在的子命令：{line!r}"
        allowed = sub_flags[sub] if sub else main_flags
        bad = [t for t in argv if t.startswith("--") and t.split("=")[0] not in allowed]
        assert not bad, f"README 里的开关不存在：{bad}（{line!r}）"
        parser = cli.subcommand_parser(sub) if sub else cli.build_parser()
        parser.parse_args(argv[1:] if sub else argv)
        seen += 1
    assert seen >= 2, f"README 里至少要有采集后的分析与注入两条命令（只扫到 {seen} 条）"


# ---------------------------------------------------------------------------
# 5) 需要 NVBit 的检查：没有就 skip，绝不假装成功
# ---------------------------------------------------------------------------
def test_resolved_artifact_version_is_read_from_the_artifact_when_present(tmp_path):
    """本机真的有一份 ks_occupancy.so 时，`--resolve-only` 报的版本必须来自实物。

    没有 NVBit / 没有构建产物时 **skip**（这条本来就依赖外部可选件）。
    """
    local = PROBE_DIR / "ks_occupancy.so"
    env_so = os.environ.get("NVBIT_OCCUPANCY_SO", "").strip()
    so = Path(env_so) if env_so else local
    if not so.is_file():
        pytest.skip(
            "本机没有 ks_occupancy.so（也没有 NVBIT_OCCUPANCY_SO）：这条检查需要一份"
            "真的 NVBit 构建产物，无法在无 GPU/无 NVBit 的机器上验证——见 "
            "tools/probes/nvbit_occupancy/README.md 的「未在真机验证」一节")
    r = _run_collect("--resolve-only", env_extra={"NVBIT_OCCUPANCY_SO": str(so)})
    assert r.returncode == 0, (r.stdout, r.stderr)
    fields = dict(re.findall(r"^(occupancy_so(?:_version)?): (.*)$", r.stdout, re.M))
    assert fields.get("occupancy_so") == str(so)
    ver = fields.get("occupancy_so_version", "")
    assert ver == "n/a" or re.fullmatch(r"\d+\.\d+(\.\d+)*", ver), (
        f"版本只能来自实物或 n/a，得到 {ver!r}：\n{r.stdout}")
