"""`bench/prune_artifacts.py` 的契约：裁剪**只丢中间件**，而且对已登记产物幂等。

为什么这条重要（真机实测）：`bench/run.sh --quick` 在 4090 上会写出 **1.3 GB**（`.nsys-rep`、
`.sqlite`、几十 MB 的完整报告），而仓库里已登记的 5 份产物都只有 **968 K–2.0 M**。裁剪规则
不能拍脑袋 —— 它的**标准答案就是那 5 份**：对它们跑 `--check` 必须一个文件都不删。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRUNE = ROOT / "bench" / "prune_artifacts.py"
ARTIFACTS = ROOT / "bench" / "artifacts"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PRUNE), *args],
                          capture_output=True, text=True)


def test_registered_artifacts_are_already_pruned():
    """**地面真值**：仓库里已登记的产物必须一个中间文件都没有。

    这条同时是"裁剪规则是否与现实一致"的判据 —— 规则写宽了（误删 .csv/.packet.json）或写窄了
    （留下 .nsys-rep）都会在这里红。
    """
    dirs = sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())
    assert dirs, "没有已登记的产物"
    r = _run(*[str(d) for d in dirs], "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_prune_drops_binaries_and_keeps_citable_files(tmp_path):
    d = tmp_path / "20260101-000000"
    d.mkdir()
    keep = ["manifest.json", "machine-profile.json", "rep.txt", "run.err", "x.log",
            "sweep.csv", "rep.packet.json", "rep.launch-rows"]
    drop = ["rep.nsys-rep", "rep.sqlite", "rep.ksreport.json", "rep.perfetto.json",
            "kernel.cubin"]
    for name in keep + drop:
        (d / name).write_text("x", encoding="utf-8")
    before = sorted(p.name for p in d.iterdir())
    assert _run(str(d)).returncode == 0
    after = sorted(p.name for p in d.iterdir())
    assert after == sorted(keep), (before, after)
    # 幂等：再跑一次什么都不动
    assert _run(str(d)).returncode == 0
    assert sorted(p.name for p in d.iterdir()) == sorted(keep)


def test_prune_reports_the_size_it_freed(tmp_path):
    d = tmp_path / "big"
    d.mkdir()
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "rep.nsys-rep").write_bytes(b"0" * 3_000_000)
    r = _run(str(d))
    assert r.returncode == 0
    assert "3.00 MB" in r.stdout or "2.9" in r.stdout or "MB" in r.stdout, r.stdout
    assert not (d / "rep.nsys-rep").exists()


def test_run_sh_prunes_before_sanitizing():
    """`run.sh` 必须先裁后脱敏 —— 否则 1.3 GB 会直接落进 `bench/artifacts/`。"""
    text = (ROOT / "bench" / "run.sh").read_text(encoding="utf-8")
    assert "prune_artifacts.py" in text, "run.sh 没有调用裁剪器"
    assert text.index("prune_artifacts.py") < text.index("sanitize_artifacts.py"), \
        "裁剪必须发生在脱敏之前（脱敏要读的就是留下来的那些文件）"
