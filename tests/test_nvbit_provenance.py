"""R-102：`environment.txt` 必须记下**实际被注入的那份 `mem_trace.so`** 及其版本。

第六轮验收 §2.3 的残留：`bench/run.sh` 的 `nvbit_version` 取自 `NVBIT_PATH` 那棵树，而真正
采集的 `collect.sh` 优先用环境里预置的 `NVBIT_MEMTRACE_SO` —— **两者可以不是同一棵树**。
真机上 `NVBIT_PATH` 那棵树里 3 份 v1.8.1 全挂，实际采到轨迹的是 `lab/` 里的另一份构建：
于是产物里的"NVBit 版本"解释不了它自己的轨迹。所以：

* `collect.sh --resolve-only` 只回答"这次会用哪份 .so、它是哪个版本"（机器可读两行）——
  发现顺序与版本读取都只有这一处真源，`bench/run.sh` 不许再抄一份候选路径；
* 版本**从实物读**（`nvbit_version.sh`）：读不出就写 `n/a`，**绝不编一个**；
* 口径是"**构建级**"：同一个版本号的不同构建结果相反，所以这两行只用来**指认**用的是哪份
  产物，任何"某版本可用/不可用"的结论都不能只看版本号（README 的矩阵与冒烟门禁负责那一层）。
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECT = ROOT / "tools" / "probes" / "nvbit_memtrace" / "collect.sh"
RUN_SH = ROOT / "bench" / "run.sh"

LAYOUT_CMAKE = "tools/build/mem_trace/mem_trace.so"
LAYOUT_MAKEFILE = "tools/mem_trace/mem_trace.so"


def _tree(base: pathlib.Path, header_version: str | None = None,
          layout: str = LAYOUT_MAKEFILE, so_version: str | None = None) -> pathlib.Path:
    """造一棵**假的** NVBit 树（只放发现逻辑要看的东西：core/nvbit.h 与 mem_trace.so）。"""
    if header_version:
        (base / "core").mkdir(parents=True, exist_ok=True)
        (base / "core" / "nvbit.h").write_text(
            f'#define NVBIT_VERSION "{header_version}"\n', encoding="utf-8")
    so = base / layout
    so.parent.mkdir(parents=True, exist_ok=True)
    # nvbit_version.sh 的退化路径只认"整行就是一个版本号"的字符串（形状 1.x[.y[.z]]）
    so.write_bytes(b"NVBit mem_trace (fake)\n"
                   + ((so_version + "\n").encode() if so_version else b""))
    return base


def _resolve(*, nvbit_path: pathlib.Path | None = None, so: pathlib.Path | None = None
             ) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("NVBIT_PATH", "NVBIT_MEMTRACE_SO", "KS_MEMTRACE_RESOLVE_ONLY")}
    if nvbit_path is not None:
        env["NVBIT_PATH"] = str(nvbit_path)
    if so is not None:
        env["NVBIT_MEMTRACE_SO"] = str(so)
    return subprocess.run(["bash", str(COLLECT), "--resolve-only"],
                          capture_output=True, text=True, env=env, timeout=60)


def _fields(stdout: str) -> dict[str, str]:
    """按 `bench/run.sh` 用的那套解析取字段（键名/形状变了这里就红）。"""
    out = {}
    for key in ("memtrace_so", "memtrace_so_version"):
        m = re.search(rf"^{key}: (.*)$", stdout, re.M)
        out[key] = m.group(1).strip() if m else ""
    return out


def test_resolve_only_reports_the_so_from_nvbit_path(tmp_path):
    tree = _tree(tmp_path / "treeA", header_version="9.9.9")
    r = _resolve(nvbit_path=tree)
    assert r.returncode == 0, (r.stdout, r.stderr)
    got = _fields(r.stdout)
    assert got["memtrace_so"] == str(tree / LAYOUT_MAKEFILE)
    assert got["memtrace_so_version"] == "9.9.9", (
        "版本必须从实物（core/nvbit.h 或 .so 字符串）读出来：\n" + r.stdout)


def test_resolve_only_prefers_the_preset_so_and_reads_its_own_version(tmp_path):
    """R-102 的要害：预置 `NVBIT_MEMTRACE_SO` 时，记的必须是**它**的路径与**它**的版本 ——
    哪怕树里另外还有一个头文件版本（两棵树的情形）。"""
    tree = _tree(tmp_path / "treeA", header_version="9.9.9")
    other = _tree(tmp_path / "treeB", layout=LAYOUT_CMAKE, so_version="1.8.1")
    r = _resolve(nvbit_path=tree, so=other / LAYOUT_CMAKE)
    assert r.returncode == 0, (r.stdout, r.stderr)
    got = _fields(r.stdout)
    assert got["memtrace_so"] == str(other / LAYOUT_CMAKE)
    assert got["memtrace_so_version"] == "1.8.1", (
        "记的是 NVBIT_PATH 那棵树的版本，而不是实际注入的那份 .so 的版本：\n" + r.stdout)


def test_resolve_only_never_invents_a_version(tmp_path):
    """读不出就 `n/a`（`nvbit_version.sh` 读不到时不打印任何东西）。"""
    so = tmp_path / "noversion" / LAYOUT_MAKEFILE
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(b"NVBit mem_trace with no version string\n")
    r = _resolve(so=so)
    assert r.returncode == 0, (r.stdout, r.stderr)
    got = _fields(r.stdout)
    assert got["memtrace_so"] == str(so)
    assert got["memtrace_so_version"] == "n/a", r.stdout


def test_resolve_only_is_silent_and_rc2_when_the_tree_has_no_product(tmp_path):
    """指了一棵没有构建产物的树 ⇒ rc=2、stdout 里**没有**可被误当答案的行。"""
    tree = tmp_path / "empty"
    (tree / "tools").mkdir(parents=True)
    r = _resolve(nvbit_path=tree)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert not re.search(r"^memtrace_so: ", r.stdout, re.M), r.stdout


def test_run_sh_records_both_trees_and_routes_them_from_collect_sh():
    """`bench/run.sh` 侧：environment.txt 同时写两处，且"这次用哪份"只问 collect.sh。"""
    run = RUN_SH.read_text(encoding="utf-8")
    for line in ('echo "nvbit: ', 'echo "nvbit_version: ',
                 'echo "memtrace_so: ', 'echo "memtrace_so_version: '):
        assert line in run, f"bench/run.sh 的 environment.txt 少了 {line!r} 这一行（R-102）"

    assert "--resolve-only" in run, (
        "bench/run.sh 必须向 collect.sh 问'这次用哪份 .so'（--resolve-only）—— "
        "发现顺序只能有一处真源，不能在这里再抄一份候选路径")
    # 版本读取只允许两处：run.sh 对 NVBIT_PATH、collect.sh 对它自己选中的 .so。
    # run.sh 里再出现一次就是"自己编版本"的入口。
    assert run.count("nvbit_version.sh") == 1, (
        "bench/run.sh 只能有一次读版本的调用（对 NVBIT_PATH）；"
        "实际注入的 .so 的版本由 collect.sh 的 --resolve-only 回给 run.sh")

    # 只给了 NVBIT_MEMTRACE_SO（没给树）时不许静默跳过整步 —— collect.sh 明明能采
    m = re.search(r"if (?P<cond>.*?); then\s*\n\s*echo \"== 8\)", run, re.S)
    assert m, "找不到步骤 8 的条件块（`if … ; then` + `echo \"== 8)\"`）"
    assert "NVBIT_MEMTRACE_SO" in m.group("cond"), (
        "步骤 8 的条件必须认 NVBIT_MEMTRACE_SO：旧版只认 NVBIT_PATH，"
        "于是'显式给了能用的 .so'也会被静默跳过（产物里既没有轨迹也没有它的证据）：\n"
        + m.group("cond"))


def test_stored_artifacts_record_the_two_lines_together():
    """已入库产物的不变量：`memtrace_so:` 与 `memtrace_so_version:` **要么都有、要么都没有**。

    本改动之前生成的产物两行都没有（那不是错：当时没有这条证据，且产物不能手改）。
    管的是今后：只记路径不记版本（或反过来）的产物在这里红。

    版本**读不出**时写 `n/a` 是允许的（`nvbit_version.sh` 读不到就不打印，编一个版本才是错），
    所以这里只查"成对出现"，不要求版本非 n/a —— 那会把诚实的 `n/a` 判成红。
    """
    art = ROOT / "bench" / "artifacts"
    if not art.is_dir():
        pytest.skip("bench/artifacts 下还没有产物目录")
    checked = 0
    for env in sorted(art.glob("*/environment.txt")):
        text = env.read_text(encoding="utf-8", errors="replace")
        so = re.search(r"^memtrace_so: (.*)$", text, re.M)
        ver = re.search(r"^memtrace_so_version: (.*)$", text, re.M)
        if so is None and ver is None:
            continue
        checked += 1
        assert so and ver, (
            f"{env.parent.name}/environment.txt 只记了 "
            f"{'memtrace_so' if so else 'memtrace_so_version'} 一行 —— "
            "路径与版本必须成对（只记路径等于没记：R-102/V-10）")
    if not checked:
        pytest.skip("还没有记 memtrace_so 的产物 —— 本改动之后第一次 bench/run.sh 才会写这两行")
