"""R-72/R-77：NVBit 依赖说明必须**与实物一致**，而且"版本×布局配错"必须红。

第五轮验收报告 V-10 记下了三条旧版全部放过的教训：

1. **版本反了**：文档说"v1.8.1 是本仓库真机跑通的"，真机 A/B 却是 v1.8.1 失败
   （`CUDA_ERROR_INVALID_SOURCE`）、v1.7.7.3 成功（6 条 `MEMTRACE` 行）。
2. **找不到就不查**：`collect.sh` 的候选路径把同一行写了两遍，v1.7.7.3 的布局
   （`tools/mem_trace/`）永远没人找；`NVBIT_PATH` 指到那棵树时旧门禁**静默 skip**
   —— 一条不会红的门禁等于没有门禁。
3. **只查版本号字符串**：把文档里 v1.8.1 那行的构建命令换成实测失败的老 Makefile 写法，
   旧断言仍然放行（变异证明）。

所以这里分四层，前两层**不需要 NVBit**（任何机器都跑，包括本机 Mac）：

* **A. 文档矩阵结构**：把 `README.md` 的「版本 × 构建命令 × 产物布局 × 本环境结论」表
  **解析成结构化行**，与测试内写死的实测结论（`VERIFIED`）逐格比对 ——
  改坏任何一格（含把 v1.8.1 的命令换回老 Makefile 写法、把两行结论对调）都会红。
* **B. `collect.sh` 的候选路径**：在临时目录里搭两种布局的假树**真跑** `collect.sh`，
  断言它找到的是哪一种；并断言"显式给了 `NVBIT_PATH` 却没有产物"时 `exit 2` 且列出
  **实际找过的路径**（不重复）、不再让人重复指同一个变量。
* **C. 实物核对**（有 NVBit 时）：解析出的 `.so` 的**版本**（`core/nvbit.h` 或 `.so` 字符串）
  与它所在的**布局**必须与文档那一行一致 —— 版本×布局配错就红。
* **D. 按文档真采一段轨迹**（有 NVBit + nvcc 时）：整体跑通并断言格式/完整性/可分析性。
  若用的是本环境已知失败的版本，失败信息要**指名道姓**（版本 + 文档记的现象）。

skip 的语义（写清楚，别让它变成静默放行）：**只有在 `NVBIT_PATH`/`NVBIT_MEMTRACE_SO`
都没给、本机也确实没有 NVBit 时**才 skip（它是可选外部依赖）；
**显式指了路径却找不到产物 ⇒ fail**（C/D 直接红，报文给出找过哪些路径）。
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "tools" / "probes" / "nvbit_memtrace"
COLLECT = PROBE_DIR / "collect.sh"
VERSION_SH = PROBE_DIR / "nvbit_version.sh"
README = PROBE_DIR / "README.md"
RUN_SH = ROOT / "bench" / "run.sh"
ARTIFACTS = ROOT / "bench" / "artifacts"
PROBE_SRC = ROOT / "bench" / "ledger_probe.cu"

# NVBit 示例工具在 NVBit 树里的两种产物布局（版本不同，布局不同 —— 这就是 R-72 的要害）
LAYOUT_CMAKE = "tools/build/mem_trace/mem_trace.so"      # CMake / 新版
LAYOUT_MAKEFILE = "tools/mem_trace/mem_trace.so"          # 老 Makefile

# **实测结论**（第五轮验收报告 V-10：同一份 app、同一 NO_EAGER_LOAD=0、只换 mem_trace.so）。
# 这张表**独立于 README 写死**：README 与它不一致就红。键带 `v` 前缀（版本号 4 段形状像 IPv4，
# 仓库卫生门禁不允许裸写 —— 见 tests/test_repo_hygiene.py 的说明）。
VERIFIED: dict[str, dict] = {
    "v1.7.7.3": dict(layout=LAYOUT_MAKEFILE,
                     build_tool="make",
                     usable=True,
                     evidence="LDG.E.CONSTANT",
                     where="老 Makefile：`make -C $NVBIT_PATH/tools/mem_trace ARCH=sm_89`"),
    "v1.8.1": dict(layout=LAYOUT_CMAKE,
                   build_tool="cmake",
                   usable=False,
                   evidence="CUDA_ERROR_INVALID_SOURCE",
                   where="CMake：`cmake -S tools -B build && cmake --build build --target mem_trace`"),
}


def _norm(version: str) -> str:
    """版本号归一成带 `v` 前缀的键（裸版本号 → `v<版本号>`；四段版本号的形状与裸 IPv4 相同，
    仓库卫生门禁要求文档/源码里带 `v` 写 —— 见 tests/test_repo_hygiene.py）。"""
    v = (version or "").strip()
    return v if v.startswith("v") else f"v{v}"


# --------------------------------------------------------------------------
# A) 文档矩阵：结构化解析 + 与实测结论逐格比对（不依赖 NVBit）
# --------------------------------------------------------------------------
def _readme_rows() -> dict[str, list[str]]:
    """把 README 的版本矩阵解析成 `{vX.Y.Z: [单元格...]}`。

    只认第一格写成 `**v<版本>**` 的表格行 —— 这是文档侧的约定，也是"结构化"与
    "整文件子串匹配"的分界线：解析不出行就没有可比对的东西。
    """
    rows: dict[str, list[str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        m = re.fullmatch(r"\*\*v(\d+\.\d+(?:\.\d+)*)\*\*", cells[0])
        if m:
            rows["v" + m.group(1)] = cells
    return rows


def test_readme_matrix_matches_the_verified_version_layout_combinations():
    """版本 ↔ 构建命令 ↔ 产物布局 ↔ 本环境结论，四格必须与实测结论一致。

    这条是"文档对不对"的门禁本体：旧断言只查"版本号字符串出现过"，
    所以把 v1.8.1 那行的构建命令换成**实测失败**的老 Makefile 写法也能放行（V-10 的变异）。
    """
    rows = _readme_rows()
    assert rows, ("README 的 NVBit 版本矩阵没被解析出来 —— 表格第一格要写成 `**vX.Y.Z**`"
                  "（见 tools/probes/nvbit_memtrace/README.md）")
    missing = sorted(set(VERIFIED) - set(rows))
    assert not missing, (f"README 的版本矩阵缺了这些行：{missing}；"
                         f"实测结论里的版本是 {sorted(VERIFIED)}")

    for ver, spec in sorted(VERIFIED.items()):
        cells = rows[ver]
        joined = " | ".join(cells)
        assert len(cells) >= 4, (
            f"{ver} 那一行应当是「版本 | 构建方式 | 产物布局 | 本环境结论」四格：{joined}")
        cmd, layout, verdict = cells[1], cells[2], " | ".join(cells[3:])

        # ① 产物布局 = 被验证过的那个（版本×布局配对的真源）
        assert spec["layout"] in layout, (
            f"{ver} 那行写的产物布局是 {layout!r}，实测是 {spec['layout']!r}")
        other = LAYOUT_MAKEFILE if spec["layout"] == LAYOUT_CMAKE else LAYOUT_CMAKE
        assert other not in layout, f"{ver} 那行的布局格同时写了另一种布局：{layout!r}"

        # ② 构建命令必须与「被验证过的版本 + 布局」一致
        if spec["layout"] == LAYOUT_MAKEFILE:
            assert "make -C" in cmd and "tools/mem_trace" in cmd and "ARCH=sm_89" in cmd, (
                f"{ver} 那行没写它真正被验证过的构建命令（{spec['where']}）：{cmd!r}")
            assert LAYOUT_CMAKE not in cmd, (
                f"{ver} 那行的构建命令写成了 CMake 布局（{LAYOUT_CMAKE}）：{cmd!r}")
        else:
            assert spec["build_tool"] in cmd.lower(), (
                f"{ver} 那行没写它真正被验证过的构建方式（{spec['where']}）：{cmd!r}")
            assert "make -C" not in cmd, (
                f"{ver} 那行把**实测失败**的老 Makefile 命令行当成了它的构建方式：{cmd!r}")

        # ③ 结论必须与实测一致；失败的那个要把真机现象写出来
        if spec["usable"]:
            assert "✅" in verdict and "失败" not in verdict, (
                f"{ver} 在本环境实测**可用**，文档却写成：{verdict!r}")
        else:
            assert "失败" in verdict and "✅" not in verdict, (
                f"{ver} 在本环境实测**失败**，文档却写成：{verdict!r}")
            assert spec["evidence"] in verdict, (
                f"{ver} 那行没写实测失败的现象（{spec['evidence']}）"
                f"—— 只说「不可用」没法复核：{verdict!r}")

    usable = {v for v, cells in rows.items() if any("✅" in c for c in cells)}
    expect = {v for v, s in VERIFIED.items() if s["usable"]}
    assert usable == expect, (
        f"矩阵里标 ✅ 的版本是 {sorted(usable)}，而真机实测可用的只有 {sorted(expect)}"
        "（同一 app、同一 NO_EAGER_LOAD=0、只换 .so；V-10）")

    # 版本是**从实物读**的，不是手抄进文档的：生成器得用同一个真源
    assert "nvbit_version.sh" in RUN_SH.read_text(encoding="utf-8"), (
        "bench/run.sh 必须用 tools/probes/nvbit_memtrace/nvbit_version.sh 从实物读版本")


# --------------------------------------------------------------------------
# B) collect.sh 的候选路径：真跑一遍（不依赖 NVBit）
# --------------------------------------------------------------------------
def _run_collect(tmp_path: pathlib.Path, *, nvbit_path=None, so=None):
    """在临时目录里跑 collect.sh（假 cuobjdump，避免"本机没 CUDA 工具链"干扰发现逻辑）。"""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "cuobjdump"
    if not fake.exists():
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("NVBIT_PATH", "NVBIT_MEMTRACE_SO", "KS_MEMTRACE_OUT", "LD_PRELOAD")}
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    env["KS_MEMTRACE_OUT"] = str(tmp_path / "trace.txt")
    if nvbit_path is not None:
        env["NVBIT_PATH"] = str(nvbit_path)
    if so is not None:
        env["NVBIT_MEMTRACE_SO"] = str(so)
    return subprocess.run(["bash", str(COLLECT), "--", "/bin/true"],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)


def test_collect_sh_finds_both_documented_layouts(tmp_path):
    """两种布局都要找（旧版把同一行写了两遍 ⇒ v1.7.7.3 的布局永远找不到、然后静默跳过）。"""
    for ver, layout in (("v1.7.7.3", LAYOUT_MAKEFILE), ("v1.8.1", LAYOUT_CMAKE)):
        tree = tmp_path / ver.replace(".", "_")
        so = tree / layout
        so.parent.mkdir(parents=True, exist_ok=True)
        so.write_bytes(b"NVBit mem_trace (fake)\n")
        r = _run_collect(tmp_path, nvbit_path=tree)
        assert f"工具: {so}" in r.stdout, (
            f"NVBIT_PATH 指到 {ver} 的布局（{layout}）时 collect.sh 没找到产物：\n"
            f"stdout={r.stdout[:500]}\nstderr={r.stderr[:500]}")


def test_collect_sh_refuses_loudly_when_the_tree_has_no_build_product(tmp_path):
    """显式指了 NVBIT_PATH 却没有产物 ⇒ exit 2，并列出**实际找过的路径**（不重复、不再让人重指）。"""
    tree = tmp_path / "nvbit_release_x86_64"
    (tree / "tools").mkdir(parents=True)
    r = _run_collect(tmp_path, nvbit_path=tree)
    assert r.returncode == 2, (r.returncode, r.stdout[-300:], r.stderr[-300:])
    assert "这棵树里没有构建产物" in r.stderr, r.stderr[-500:]
    for cand in (tree / LAYOUT_CMAKE, tree / LAYOUT_MAKEFILE):
        assert r.stderr.count(str(cand)) == 1, (
            f"报错必须列出实际找过的路径，且每个只出现一次：{cand} 出现 "
            f"{r.stderr.count(str(cand))} 次\n{r.stderr[-800:]}")
    assert "请设置 NVBIT_MEMTRACE_SO" not in r.stderr, (
        "已经指了 NVBIT_PATH，报错不该再让人指一遍同一个变量：\n" + r.stderr[-500:])


def test_collect_sh_refuses_an_explicit_so_that_does_not_exist(tmp_path):
    missing = tmp_path / "nope" / "mem_trace.so"
    r = _run_collect(tmp_path, so=missing)
    assert r.returncode == 2 and str(missing) in r.stderr and "不存在" in r.stderr, \
        (r.returncode, r.stderr[-500:])


# --------------------------------------------------------------------------
# C/D) 实物核对 + 按文档真采
# --------------------------------------------------------------------------
class NvbitLookupMiss(Exception):
    """显式指了路径却没有产物 —— 这是配置错，必须 fail，**不许 skip**。"""


def _tree_of(so: pathlib.Path) -> pathlib.Path | None:
    for p in so.parents:
        if (p / "core" / "nvbit.h").is_file():
            return p
    return None


def _layout_of(so: pathlib.Path, tree: pathlib.Path | None) -> str:
    if tree:
        try:
            return so.resolve().relative_to(pathlib.Path(tree).resolve()).as_posix()
        except ValueError:
            pass
    s = so.resolve().as_posix()          # 认不出 NVBit 树时，按路径后缀判布局
    for layout in (LAYOUT_CMAKE, LAYOUT_MAKEFILE):
        if s.endswith("/" + layout):
            return layout
    return "/".join(so.parts[-3:])


def _lookup() -> tuple[pathlib.Path, str, pathlib.Path | None] | None:
    """定位 mem_trace.so：`NVBIT_MEMTRACE_SO` → `NVBIT_PATH`（两种布局）→ 探针目录。

    `None` = 什么都没指且本机确实没有（调用方 skip）；显式指了却找不到 ⇒ 抛 `NvbitLookupMiss`。
    """
    env_so = (os.environ.get("NVBIT_MEMTRACE_SO") or "").strip()
    base = (os.environ.get("NVBIT_PATH") or "").strip()
    if env_so:
        p = pathlib.Path(env_so)
        if p.is_file():
            return p, "NVBIT_MEMTRACE_SO", _tree_of(p)
        raise NvbitLookupMiss(
            f"NVBIT_MEMTRACE_SO={env_so} 指向的文件不存在。\n"
            f"  v1.7.7.3 的产物在 {LAYOUT_MAKEFILE}，v1.8.1 的在 {LAYOUT_CMAKE}"
            f"（见 tools/probes/nvbit_memtrace/README.md 的版本矩阵）")
    if base:
        tried = [pathlib.Path(base) / LAYOUT_CMAKE, pathlib.Path(base) / LAYOUT_MAKEFILE]
        for cand in tried:
            if cand.is_file():
                return cand, "NVBIT_PATH", pathlib.Path(base)
        raise NvbitLookupMiss(
            f"NVBIT_PATH={base} 这棵树里没有构建产物（NVBit 示例工具要自己构建）。\n"
            "  实际找过的路径：\n  - " + "\n  - ".join(str(t) for t in tried) +
            "\n  （v1.7.7.3 的产物在 tools/mem_trace/，v1.8.1 在 tools/build/mem_trace/；"
            "本环境实测可用的是 v1.7.7.3 —— 见 tools/probes/nvbit_memtrace/README.md）")
    local = sorted(PROBE_DIR.glob("*.so"))
    if local:
        return local[0], "探针目录", _tree_of(local[0])
    return None


def _require_so() -> tuple[pathlib.Path, str, pathlib.Path | None]:
    try:
        found = _lookup()
    except NvbitLookupMiss as exc:
        pytest.fail(str(exc))
    if found is None:
        pytest.skip("本机没有 NVBit（也没显式给 NVBIT_PATH/NVBIT_MEMTRACE_SO）—— "
                    "可选外部依赖，见 tools/probes/nvbit_memtrace/README.md")
    return found


def _version_from_h(tree: pathlib.Path | None) -> str | None:
    if tree is None:
        return None
    hdr = pathlib.Path(tree) / "core" / "nvbit.h"
    if not hdr.is_file():
        return None
    m = re.search(r'NVBIT_VERSION\s+"([^"]+)"', hdr.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def _version_from_so(so: pathlib.Path) -> str | None:
    """从 .so 里读版本：只认**已知版本号**作为独立字符串出现（别把别的库版本串进来）。"""
    try:
        blob = so.read_bytes()
    except OSError:
        return None
    for key in VERIFIED:
        bare = key.lstrip("v").encode()
        if re.search(rb"(?<![\d.])" + re.escape(bare) + rb"(?![\d.])", blob):
            return key.lstrip("v")
    return None


def _nvbit_version(so: pathlib.Path, tree: pathlib.Path | None) -> str | None:
    """先看 `core/nvbit.h`（权威），退化到 `.so` 里的版本字符串。"""
    return _version_from_h(tree) or _version_from_so(so)


def _nvcc() -> str | None:
    """找 nvcc：PATH → `$CUDA_HOME/bin` → `$CUDA_BIN` → `/usr/local/cuda*/bin`。

    租用机上常见"CUDA 装了但不在 PATH"（本仓库 `bench/run.sh` 为此显式加过 PATH）——
    门禁不能因为 PATH 而误判成"这台机器没有 NVBit"。
    """
    found = shutil.which("nvcc")
    if found:
        return found
    cands: list[pathlib.Path] = []
    for env in ("CUDA_HOME", "CUDA_PATH", "CUDA_BIN"):
        if os.environ.get(env):
            cands.append(pathlib.Path(os.environ[env]) / "nvcc")
    cands += [pathlib.Path("/usr/local/cuda/bin/nvcc")]
    cands += sorted(pathlib.Path("/usr/local").glob("cuda*/bin/nvcc"))
    return next((str(c) for c in cands if c.is_file()), None)


def test_resolved_memtrace_so_matches_the_documented_version_and_layout():
    """实物核对：用哪份 `.so`，它的版本与布局就必须与文档那一行对得上。

    旧门禁在这里只对"版本号字符串在文档里出现过" —— 于是把 v1.8.1 的布局/命令配错也放行。
    """
    so, source, tree = _require_so()
    rel = _layout_of(so, tree)
    assert rel in (LAYOUT_CMAKE, LAYOUT_MAKEFILE), (
        f"{source}={so} 不在文档的两种布局里（解析出的相对路径 {rel!r}）")
    ver = _nvbit_version(so, tree)
    assert ver, (f"读不出 {so} 的 NVBit 版本：core/nvbit.h 与 .so 字符串里都没有 —— "
                 f"版本读不出来就无法判断这套组合能不能用（README 的版本矩阵就是按版本写的）")
    key = _norm(ver)
    spec = VERIFIED.get(key)
    assert spec, (f"实物版本 {key} 不在被验证过的矩阵里（现在只钉住了 {sorted(VERIFIED)}）："
                  f"要么补一次真机实测把结论写进矩阵，要么别用它当证据")
    assert spec["layout"] == rel, (
        f"{key} 的产物在 {rel}，而实测结论说它在 {spec['layout']} —— **版本 × 目录布局配错了**")
    hdr_ver = _version_from_h(tree)
    if hdr_ver:
        assert _norm(hdr_ver) == key, f"core/nvbit.h 写 {hdr_ver}，.so 里却是 {ver}"
    row = " | ".join(_readme_rows().get(key, []))
    assert spec["layout"] in row, (
        f"README 里 {key} 那一行没写它真实的产物布局 {spec['layout']}")


def test_documented_collect_command_produces_a_complete_parseable_trace(tmp_path):
    """按文档真采一段轨迹：有记录、带哨兵、能解析、能分析出全局口径。

    失败信息必须**指名道姓**：如果是本环境已知失败的版本，直接说清是哪个版本、
    文档记的现象是什么、该换成哪一份 —— 而不是一句"采集失败"。
    """
    so, source, tree = _require_so()
    nvcc = _nvcc()
    if nvcc is None:
        pytest.skip("本机没有 nvcc —— 有 NVBit 也编译不了文档里的探针（CUDA 工具链不完整）")
    ver = _nvbit_version(so, tree)
    spec = VERIFIED.get(_norm(ver)) if ver else None

    probe = tmp_path / "ledger_probe"
    r = subprocess.run([nvcc, "-O3", "-o", str(probe), str(PROBE_SRC)],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    if r.returncode != 0:
        pytest.skip(f"nvcc 编译探针失败（本机 CUDA 工具链不完整）：{r.stderr.strip()[:200]}")

    trace = tmp_path / "trace.txt"
    env = dict(os.environ, KS_MEMTRACE_OUT=str(trace), NVBIT_MEMTRACE_SO=str(so))
    # 负载必须**有界**：轨迹一行 = 一条 warp 访存指令，紧循环 1 秒 ≈ 6.6 GB（真事，
    # 会把磁盘写满）。这里只跑 3 次 launch，够断言格式与完整性即可。
    free_mb = shutil.disk_usage(tmp_path).free // (1 << 20)
    if free_mb < 1024:
        pytest.skip(f"剩余空间只有 {free_mb} MB，采集需要至少 1 GB")
    r = subprocess.run(["bash", str(COLLECT), "--", str(probe), "--kernel", "copy",
                        "--mb", "4", "--iters", "3", "--warmup", "1"],
                       capture_output=True, text=True, cwd=ROOT, timeout=900, env=env)
    if r.returncode != 0 and spec is not None and not spec["usable"]:
        pytest.fail(
            f"{source} 解析出的 NVBit {ver}（{so}）在本环境（CUDA 13 + sm_89 + 驱动 580）"
            f"**实测失败** —— README 的版本矩阵已标注这个现象：{spec['evidence']}。\n"
            f"  请改用实测可用的 v1.7.7.3（产物在 {LAYOUT_MAKEFILE}）：\n"
            f"    NVBIT_PATH=<那棵树>  # 或 NVBIT_MEMTRACE_SO=<那棵树>/{LAYOUT_MAKEFILE}\n"
            f"  采集输出尾部：{r.stdout[-400:]}\n{r.stderr[-400:]}")
    assert r.returncode == 0, f"按文档采集失败：{r.stdout[-400:]}\n{r.stderr[-400:]}"
    assert trace.exists() and trace.stat().st_size > 0, "轨迹是空的"
    assert trace.stat().st_size < 200 * 1024 * 1024, "有界负载不该产出这么大的轨迹"
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "KS-TRACE-COMPLETE" in text, "collect.sh 必须写结束标记（R-76）"
    assert "MEMTRACE:" in text and "warp" in text, "轨迹里没有 warp 访存记录"
    if spec is not None and not spec["usable"] and source == "NVBIT_PATH":
        # **反方向**的文档/实物冲突：`NVBIT_PATH` 的语义是"按文档的配方用这棵树里的产物"，
        # 而这棵树的产物在本机跑通了、文档却把它标成不可用 —— 那说明矩阵需要更新
        # （真事：验收方机器上确实存在两份结果不同的 1.8.1 构建）。
        # 显式给 `NVBIT_MEMTRACE_SO` 时不算冲突：那是使用者**点名**要用这一份产物。
        pytest.fail(
            f"{ver} 在本环境按文档（{spec['where']}）属于**不可用**组合"
            f"（现象：{spec['evidence']}），但 NVBIT_PATH={os.environ.get('NVBIT_PATH')} 里的这份产物"
            f"（{so}）真的采出了轨迹 —— 同一个版本号、不同构建结果不同（验收方机器上就有这样的"
            f"两份 1.8.1），矩阵需要按实物更新。请二选一：\n"
            f"  * 用文档钉住的可用组合（v1.7.7.3 + 老 Makefile，产物在 {LAYOUT_MAKEFILE}）；或\n"
            f"  * 把这份产物的证据（构建时间/命令、E2E 结果）补进 README 的矩阵，"
            f"并同步更新 tests/test_nvbit_smoke.py 的 VERIFIED 表。")

    from kernoscope.analyze.memtrace import analyze_memtrace, to_metrics
    from kernoscope.backends.memtrace import load_memtrace_auto

    tr = load_memtrace_auto(str(trace), sm_count=128)
    assert tr.complete == "yes", f"完整性标记应当是 yes，实得 {tr.complete}"
    assert len(tr.header.get("trace_sha256", "")) == 16, "轨迹要带指纹（R-75）"
    st = analyze_memtrace(tr)
    assert st.records > 0 and st.records_global > 0, st.kind_counts
    assert st.coalescing_pct <= 100.0 + 1e-9, f"合并效率 {st.coalescing_pct}% 物理上不可能（R-73）"
    names = {m[0] for m in to_metrics(st)}
    assert "meas__sectors_per_request" in names


# --------------------------------------------------------------------------
# E) 版本从实物读 + 产物环境指纹必须记版本（V-10：只记路径等于没记）
# --------------------------------------------------------------------------
def test_version_helper_reads_the_version_from_the_header_and_from_the_so(tmp_path):
    tree = tmp_path / "nvbit_release_x86_64"
    (tree / "core").mkdir(parents=True)
    (tree / "core" / "nvbit.h").write_text('#define NVBIT_VERSION "9.9.9"\n', encoding="utf-8")
    r = subprocess.run(["bash", str(VERSION_SH), str(tree)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "9.9.9", (r.returncode, r.stdout, r.stderr)

    # 头文件被裁剪过 → 退回 .so 里的**整行**版本字符串（版本号本身不写死在脚本里）
    bare = next(k for k, s in VERIFIED.items() if not s["usable"]).lstrip("v")
    tree2 = tmp_path / "soonly"
    so = tree2 / LAYOUT_MAKEFILE
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(b"NVBit mem_trace\n" + bare.encode() + b"\n")
    r2 = subprocess.run(["bash", str(VERSION_SH), str(so)], capture_output=True, text=True)
    assert r2.returncode == 0 and r2.stdout.strip() == bare, (r2.returncode, r2.stdout, r2.stderr)

    # 读不到就**不许编**：非 0 且没有输出（调用方自己决定写 n/a）
    r3 = subprocess.run(["bash", str(VERSION_SH), str(tmp_path / "nope")],
                        capture_output=True, text=True)
    assert r3.returncode != 0 and r3.stdout.strip() == "", (r3.returncode, r3.stdout)


def test_environment_snapshot_records_the_nvbit_version():
    """`environment.txt` 必须同时记 NVBit 的**路径**与**版本**。

    V-10 的原话：这一整节结论（哪个版本可用）建立在版本上，而产物只记了路径。
    """
    run = RUN_SH.read_text(encoding="utf-8")
    assert re.search(r'echo "nvbit_version: ', run), \
        "bench/run.sh 必须在 environment.txt 里写一行 nvbit_version:"
    if not ARTIFACTS.is_dir():
        pytest.skip("bench/artifacts 下还没有产物目录")
    dirs = sorted(p for p in ARTIFACTS.iterdir() if p.is_dir() and p.name[:1].isdigit())
    if not dirs:
        pytest.skip("bench/artifacts 下还没有产物目录")
    env = (dirs[-1] / "environment.txt").read_text(encoding="utf-8", errors="replace")
    m_ver = re.search(r"^nvbit_version: (.*)$", env, re.M)
    assert m_ver, (f"{dirs[-1].name}/environment.txt 没记 NVBit 版本 —— "
                   f"只记路径等于没记（V-10）")
    m_path = re.search(r"^nvbit: (.*)$", env, re.M)
    if m_path and m_path.group(1).strip() not in ("n/a", ""):
        assert m_ver.group(1).strip() not in ("n/a", "unknown", "", "?"), (
            f"记了 NVBit 路径 {m_path.group(1).strip()}，却把版本记成 "
            f"{m_ver.group(1).strip()!r} —— 版本必须从实物读出来")
