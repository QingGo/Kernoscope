"""四个输出口的一致性（R-13）与成本可预算（R-22）、合成数据标注（R-23）。

R-13 的验收标准是"同一指标在四个输出中的 provenance 相同"：
文本 / CSV / JSON / `-i` 回读 —— 旧实现里 doctor 的能力矩阵与报告实际标签就互相矛盾
（doctor 把 roofline 归 derived、报告实际是 estimated），所以这里把它变成机器可查的断言。
"""

import csv
import io
import json
from pathlib import Path

from kernoscope.cli import main
from kernoscope.provenance import Provenance


def _run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def _text_provenance(text: str) -> dict[str, str]:
    """从文本行里抠出 `name value unit [prov]`（含 n/a 行）。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("=", "Kernel:", "Section:", "note:", "Legend", "Provenance",
                                        "Output", "Open", "Filtered", "-", "⚠")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        tag = [p for p in parts if p.startswith("[") and p.endswith("]")]
        if not tag:
            continue
        prov = tag[-1][1:-1]
        if prov in {p.value for p in Provenance}:
            out[parts[0]] = prov
    return out


def test_provenance_is_identical_across_all_four_outputs(capsys, tmp_path: Path):
    base = tmp_path / "rep"
    rc, text, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-peak-gbps", "909",
                       "-k", "regex:gemm", "-o", str(base), "-f")
    assert rc == 0
    text_map = _text_provenance(text)
    assert text_map, "文本里应当能抠出指标行"

    rc2, csv_text, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "--ks-peak-gbps", "909",
                            "-k", "regex:gemm", "--csv")
    assert rc2 == 0
    csv_map = {r["metric"]: r["provenance"]
               for r in csv.DictReader(io.StringIO(csv_text))}

    js = json.loads((tmp_path / "rep.ksreport.json").read_text())
    json_map = {m["name"]: m["provenance"]
                for k in js["kernels"] for s in k["sections"] for m in s["metrics"]}

    rc3, back_text, _ = _run(capsys, "-i", str(base) + ".ksreport.json")
    assert rc3 == 0
    back_map = _text_provenance(back_text)

    for name, prov in json_map.items():
        if name in csv_map:
            assert csv_map[name] == prov, f"CSV 与 JSON 不一致: {name}"
        if name in text_map:
            assert text_map[name] == prov, f"文本与 JSON 不一致: {name}"
        if name in back_map:
            assert back_map[name] == prov, f"-i 回读与 JSON 不一致: {name}"
    # 至少验证过若干关键指标
    for key in ("est__dram_bytes_total", "ks__sampler_window_coverage_pct"):
        if key in json_map:
            assert json_map[key] in {"estimated", "derived", "unavailable"}
    assert json_map  # 不是空跑


def test_thin_json_decouples_size_from_iterations(capsys, tmp_path: Path):
    """R-22：`--ks-json-no-launches` 让报告体积与 iteration 数解耦。"""
    base_full = tmp_path / "full"
    base_thin = tmp_path / "thin"
    _run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base_full), "-f")
    _run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base_thin), "-f",
         "--ks-json-no-launches")
    full = json.loads(Path(str(base_full) + ".ksreport.json").read_text())
    thin = json.loads(Path(str(base_thin) + ".ksreport.json").read_text())
    assert len(full["kernels"][0]["launches"]) == 5
    assert thin["kernels"][0]["launches"] == []
    # 聚合读数必须一模一样（瘦身只掉逐 launch 明细）
    assert full["kernels"][0]["total_ns"] == thin["kernels"][0]["total_ns"]
    assert len(Path(str(base_thin) + ".ksreport.json").read_bytes()) < \
        len(Path(str(base_full) + ".ksreport.json").read_bytes())


def test_size_advisory_is_printed(capsys, tmp_path: Path):
    base = tmp_path / "rep"
    rc, _, err = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f")
    assert rc == 0
    assert "输出体积" in err
    assert "KB" in err


def test_synthetic_data_is_labelled_everywhere(capsys, tmp_path: Path):
    """R-23：合成数据必须在文本与 JSON 里都明确标注，不得冒充实测。"""
    base = tmp_path / "rep"
    rc, out, _ = _run(capsys, "--ks-synthetic", "--ks-no-nvml", "-o", str(base), "-f")
    assert rc == 0
    assert "SYNTHETIC" in out.upper()
    doc = json.loads(Path(str(base) + ".ksreport.json").read_text())
    warn = " ".join(str(v) for v in doc["sources"].values()).upper()
    assert "SYNTHETIC" in warn
    assert doc["meta"].get("synthetic") is True
