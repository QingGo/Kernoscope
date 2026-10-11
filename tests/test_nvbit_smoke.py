"""R-72：**按文档冒烟采一段真轨迹** —— NVBit 依赖说明必须与可用版本一致。

为什么需要一条独立门禁（第四轮的教训）：依赖说明里写的版本与构建布局**互相矛盾**
（文档说"NVBit 1.8.1"，却让人跑 1.7.x 的 `make -C tools/mem_trace`），照抄的人会失败；
而"文档说的能不能跑"在仓库里没有任何一条检查 —— 直到有人真的去装一遍。

这条测试做的事就是照抄文档：
  1. 编译 `bench/ledger_probe.cu`（`nvcc`，仓库自己的探针）；
  2. 用 `tools/probes/nvbit_memtrace/collect.sh` 采一段小轨迹（脚本自己发现 `mem_trace.so`）；
  3. 断言：有记录、**带结束标记**（R-76 的哨兵）、能解析、能分析出全局口径统计。

本机没有 NVBit（要自己下载/许可/构建）时**跳过** —— 它是可选外部依赖，不是 CI 的前置条件；
但只要有，就必须能按文档跑通。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECT = ROOT / "tools" / "probes" / "nvbit_memtrace" / "collect.sh"
PROBE_SRC = ROOT / "bench" / "ledger_probe.cu"


def _memtrace_so() -> pathlib.Path | None:
    env = os.environ.get("NVBIT_MEMTRACE_SO")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    base = os.environ.get("NVBIT_PATH")
    cands = []
    if base:
        cands += [pathlib.Path(base) / "tools/build/mem_trace/mem_trace.so",
                  pathlib.Path(base) / "tools/build/mem_trace/mem_trace.so"]
    cands += list((ROOT / "tools" / "probes" / "nvbit_memtrace").glob("*.so"))
    return next((c for c in cands if c.is_file()), None)


needs_nvbit = pytest.mark.skipif(
    _memtrace_so() is None or shutil.which("nvcc") is None,
    reason="本机没有 NVBit 的 mem_trace.so 或 nvcc（可选外部依赖，见 tools/probes/nvbit_memtrace/README.md）")


@needs_nvbit
def test_documented_collect_command_produces_a_complete_parseable_trace(tmp_path):
    probe = tmp_path / "ledger_probe"
    r = subprocess.run(["nvcc", "-O3", "-o", str(probe), str(PROBE_SRC)],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    if r.returncode != 0:
        pytest.skip(f"nvcc 编译探针失败（本机 CUDA 工具链不完整）：{r.stderr.strip()[:200]}")

    trace = tmp_path / "trace.txt"
    env = dict(os.environ, KS_MEMTRACE_OUT=str(trace))
    r = subprocess.run(["bash", str(COLLECT), "--", str(probe), "--kernel", "copy",
                        "--mb", "16", "--target-s", "1"],
                       capture_output=True, text=True, cwd=ROOT, timeout=900, env=env)
    assert r.returncode == 0, f"按文档采集失败：{r.stdout[-400:]}\n{r.stderr[-400:]}"
    assert trace.exists() and trace.stat().st_size > 0, "轨迹是空的"
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "KS-TRACE-COMPLETE" in text, "collect.sh 必须写结束标记（R-76）"
    assert "MEMTRACE:" in text and "warp" in text, "轨迹里没有 warp 访存记录"

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


@needs_nvbit
def test_memtrace_so_version_matches_the_documented_matrix():
    """依赖说明里的版本必须和**本机实际那份**对得上（文档说 1.8.1，就得真的是 1.8.1）。"""
    so = _memtrace_so()
    base = pathlib.Path(os.environ.get("NVBIT_PATH", so.parents[3] if so else ""))
    hdr = base / "core" / "nvbit.h"
    if not hdr.is_file():
        pytest.skip("找不到 NVBit 的 core/nvbit.h，无法核对版本")
    import re

    m = re.search(r'NVBIT_VERSION\s+"([^"]+)"', hdr.read_text(encoding="utf-8", errors="replace"))
    if not m:
        pytest.skip("nvbit.h 里没有版本宏")
    version = m.group(1)
    readme = (ROOT / "tools" / "probes" / "nvbit_memtrace" / "README.md").read_text(encoding="utf-8")
    collect = COLLECT.read_text(encoding="utf-8")
    assert version in readme or f"v{version}" in readme, \
        f"README 没写本机在用的 NVBit 版本 {version}"
    assert version in collect or f"v{version}" in collect, \
        f"collect.sh 没写本机在用的 NVBit 版本 {version}"
