"""--strict 在 -i 回读模式下也必须生效（曾经被 early-return 跳过）。"""

import json
from pathlib import Path

from kernoscope.cli import main


def _make_report(tmp_path: Path, capsys) -> Path:
    base = tmp_path / "rep"
    assert main(["--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f"]) == 0
    capsys.readouterr()
    return Path(str(base) + ".ksreport.json")


def test_strict_in_import_mode_returns_2(tmp_path, capsys):
    js = _make_report(tmp_path, capsys)
    rc = main(["-i", str(js), "--strict", "--metrics", "l1tex__t_sector_hit_rate.pct"])
    err = capsys.readouterr().err
    assert rc == 2, "import 模式下 --strict 必须仍然卡门"
    assert "hardware counters" in err


def test_lenient_import_mode_is_fine(tmp_path, capsys):
    js = _make_report(tmp_path, capsys)
    rc = main(["-i", str(js), "--metrics", "l1tex__t_sector_hit_rate.pct"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out  # 默认模式照常输出 n/a 行


def test_cubin_token_matching(tmp_path):
    """cuobjdump 的 mangled 键要能匹配 demangled 的 kernel 名。"""
    from kernoscope.report import KernelReport
    from kernoscope.sections import _lookup_in

    cubin_res = {"_Z4addkPfi": {"REG": 16, "SHARED": 0}}
    kr = KernelReport("addk(float *, int)", alias="addk")
    assert _lookup_in(cubin_res, kr) == {"REG": 16, "SHARED": 0}

    # 不能把 Kernel2 误配到 Kernel3
    kr2 = KernelReport("void cutlass::Kernel3<cutlass_80_tensorop_bf16_s16816gemm>(T1::Params)",
                       alias="Kernel3")
    assert _lookup_in({"_Z8Kernel2v": {"REG": 8}}, kr2) is None
