"""NVBit 访存轨迹层：解析、精确算术、缓存模型、CLI 与报告集成。

关键是用**已知答案**的合成轨迹验证算术，而不是"跑通就行"：
    stride-1 fp32（32 lane × 4B 连续）      → 4 个 32B sector，合并效率 100%
    stride-32 fp32（lane 间隔 32B）          → 32 个 sector，合并效率 12.5%
    未对齐 stride-1（首地址 +4）             → 5 个 sector，合并效率 80%
"""

import json

import pytest

import kernoscope.cli as cli
from kernoscope.analyze.memtrace import analyze_memtrace
from kernoscope.backends.memtrace import MemTraceError, load_memtrace

L2_72MB = 72 * 1024 * 1024


def _mk_trace(path, rows, *, grid=8, blocks_traced=8, ratio=1.0, l2=L2_72MB,
              truncated=0, launches=1, kernel="addk", claim=None):
    """rows: (pc, op, lanes, size, smid, [addrs])；claim 可覆盖 sectors 列以测一致性校验。"""
    head = [f"# ks-memtrace v1", f"# kernel {kernel}", "# arch sm_89",
            f"# grid {grid}", "# block 128", f"# blocks_traced {blocks_traced}",
            f"# sampling_ratio {ratio}", f"# l2_bytes {l2}", "# sm_count 2",
            f"# launches_traced {launches}", f"# truncated {truncated}",
            "pc,op,lanes,size,sectors,smid,addrs"]
    lines = list(head)
    for pc, op, lanes, size, smid, addrs in rows:
        secs = sorted({a & ~31 for a in addrs})
        claimed = len(secs) if claim is None else claim(len(secs))
        lines.append(f"{pc},{op},{lanes},{size},{claimed},{smid}," + ";".join(str(s) for s in secs))
    path.write_text("\n".join(lines) + "\n")
    return path


def _coalesced(base, lanes=32, size=4, smid=0):
    return (0x100, "LDG.E", lanes, size, smid, [base + size * i for i in range(lanes)])


def _strided(base, stride, lanes=32, size=4, smid=0):
    return (0x100, "LDG.E", lanes, size, smid, [base + stride * i for i in range(lanes)])


# ---------------------------------------------------------------- 解析
def test_parse_and_integrity(tmp_path):
    p = _mk_trace(tmp_path / "t.txt", [_coalesced(0x1000)])
    tr = load_memtrace(str(p))
    assert tr.kernel == "addk" and len(tr.records) == 1
    assert tr.records[0].smid == 0
    assert len(tr.records[0].sectors) == 4


def test_parser_rejects_sector_mismatch(tmp_path):
    """工具报的 sectors 与 addrs 不符时必须报错 —— 否则错误会一路带进报告。"""
    p = _mk_trace(tmp_path / "bad.txt", [_coalesced(0x1000)], claim=lambda n: n + 1)
    with pytest.raises(MemTraceError, match="口径不一致"):
        load_memtrace(str(p))


def test_parser_rejects_missing_tag_and_empty(tmp_path):
    p = tmp_path / "notag.txt"
    p.write_text("pc,op,lanes,size,sectors,smid,addrs\n0x1,LDG.E,32,4,4,0,0;32;64;96\n")
    with pytest.raises(MemTraceError, match="ks-memtrace v1"):
        load_memtrace(str(p))

    p2 = tmp_path / "empty.txt"
    p2.write_text("# ks-memtrace v1\n# kernel k\npc,op,lanes,size,sectors,smid,addrs\n")
    with pytest.raises(MemTraceError, match="没有任何访存记录"):
        load_memtrace(str(p2))


def test_parser_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "cols.txt"
    p.write_text("# ks-memtrace v1\n0x1,LDG.E,32,4,4,0\n")
    with pytest.raises(MemTraceError, match="期望 7 列"):
        load_memtrace(str(p))


# ---------------------------------------------------------------- 精确算术
def test_sectors_per_request_and_coalescing_known_answers(tmp_path):
    # 三种已知访存形态放在同一条轨迹里
    rows = [_coalesced(0x1000),                       # 4 sectors, 100%
            _strided(0x2000, 32),                     # 32 sectors, 12.5%
            (0x300, "LDG.E", 32, 4, 0, [0x3004 + 4 * i for i in range(32)])]  # 未对齐 -> 5 sectors, 80%
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))), l2_bytes=L2_72MB)
    assert st.sectors_per_request == pytest.approx((4 + 32 + 5) / 3)
    assert st.coalescing_pct == pytest.approx(100 * (128 + 128 + 128) / (4 * 32 + 32 * 32 + 5 * 32))
    # 单条也可逐条验证
    tr = load_memtrace(str(_mk_trace(tmp_path / "one.txt", [_strided(0x2000, 32)])))
    assert tr.records[0].coalescing_pct == pytest.approx(12.5)
    tr2 = load_memtrace(str(_mk_trace(tmp_path / "u.txt",
                                      [(0x300, "LDG.E", 32, 4, 0, [0x3004 + 4 * i for i in range(32)])])))
    assert len(tr2.records[0].sectors) == 5
    assert tr2.records[0].coalescing_pct == pytest.approx(80.0)


def test_working_set_counts_unique_sectors(tmp_path):
    rows = [_coalesced(0x1000) for _ in range(10)]     # 同一 128B 区重复 10 次
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))), l2_bytes=L2_72MB)
    assert st.working_set_mb == pytest.approx(4 * 32 / 1e6)   # 只有 4 个 sector


# ---------------------------------------------------------------- 缓存模型
def test_l2_model_repeated_working_set_hits(tmp_path):
    """2KB 工作集反复访问：72MB L2 下命中率应接近 100%。"""
    rows = []
    for _ in range(50):
        rows.append(_coalesced(0x1000))     # 每轮 4 sector，共 64 sector = 2KB
        rows.append(_coalesced(0x2000))
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))), l2_bytes=L2_72MB)
    assert st.l2_hit_rate_l2only_pct > 95.0


def test_l2_model_streaming_has_no_hits(tmp_path):
    """纯流式（每个 sector 只碰一次）：命中率应为 0。

    这条是防"建模粒度选错"的哨兵：按 128B 行建模会把同一行内的后几个 sector
    算成命中，从而给出 ~7.5% 的假命中率。
    """
    rows = [_coalesced(0x100000 + 0x1000 * i) for i in range(200)]
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))), l2_bytes=L2_72MB)
    assert st.l2_hit_rate_l2only_pct == pytest.approx(0.0)


def test_l1_filter_lowers_l2_hit_rate(tmp_path):
    """L1 过滤口径必须低于"全部到达 L2"的上界口径 —— 这是这个 caveat 的意义。

    小工作集在 L1(128KB) 里全命中，于是真正到达 L2 的访问极少，L2 命中率应接近 0；
    而上界口径会把 L1 命中误算成 L2 命中（高）。
    """
    rows = []
    for _ in range(50):
        rows.append(_coalesced(0x1000, smid=0))
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))),
                          l2_bytes=L2_72MB, l1_bytes=128 * 1024)
    assert st.l2_hit_rate_l2only_pct > 95.0
    assert st.l2_hit_rate_with_l1_pct < 20.0
    assert st.l2_hit_rate_l2only_pct > st.l2_hit_rate_with_l1_pct


# ---------------------------------------------------------------- DRAM 字节与拒绝给数
def test_dram_bytes_scaled_by_sampling(tmp_path):
    """采样 2/8 block → 模型 miss 要按 4 倍放大。"""
    rows = [_coalesced(0x100000 + 0x1000 * i) for i in range(10)]   # 40 sector，全 miss
    full = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "f.txt", rows))), l2_bytes=L2_72MB)
    samp = analyze_memtrace(load_memtrace(str(_mk_trace(
        tmp_path / "s.txt", rows, grid=8, blocks_traced=2, ratio=0.25))), l2_bytes=L2_72MB)
    assert full.dram_bytes_per_launch == pytest.approx(40 * 32)
    assert samp.dram_bytes_per_launch == pytest.approx(40 * 32 * 4)


def test_truncated_trace_refuses_dram_bytes(tmp_path):
    """截断轨迹只能给下界 → 按纪律不给数字（下界最容易被误读成实测）。"""
    rows = [_coalesced(0x100000 + 0x1000 * i) for i in range(10)]
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows, truncated=1))),
                          l2_bytes=L2_72MB)
    assert st.dram_bytes_per_launch is None
    assert "下界" in st.dram_bytes_unavailable_reason


def test_max_accesses_refuses_dram_bytes(tmp_path):
    rows = [_coalesced(0x100000 + 0x1000 * i) for i in range(50)]
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows))),
                          l2_bytes=L2_72MB, max_accesses=10)
    assert st.dram_bytes_per_launch is None
    assert "上限" in st.dram_bytes_unavailable_reason


def test_missing_l2_size_refuses_dram_bytes(tmp_path):
    rows = [_coalesced(0x1000)]
    st = analyze_memtrace(load_memtrace(str(_mk_trace(tmp_path / "t.txt", rows, l2=0))))
    assert st.dram_bytes_per_launch is None
    assert "l2_bytes" in st.dram_bytes_unavailable_reason


# ---------------------------------------------------------------- CLI 与报告集成
def test_cli_memtrace_text_and_json(tmp_path, capsys):
    p = _mk_trace(tmp_path / "t.txt", [_coalesced(0x1000), _strided(0x2000, 32)])
    assert cli.main(["memtrace", "--from", str(p)]) == 0
    out = capsys.readouterr().out
    assert "sectors/request" in out and "l1tex__average_t_sectors_per_request" in out

    assert cli.main(["memtrace", "--from", str(p), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["kernel"] == "addk"
    assert doc["sectors_per_request"] == pytest.approx(18.0)
    assert set(doc["op_counts"]) == {"LDG.E"}


def test_cli_memtrace_bad_file_returns_1(tmp_path, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("garbage\n")
    assert cli.main(["memtrace", "--from", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_report_integration_adds_section_with_right_provenance(monkeypatch, capsys, tmp_path):
    # 采样轨迹 -> 连精确算术也降为 estimated（样本代表性是假设）
    p = _mk_trace(tmp_path / "t.txt", [_coalesced(0x1000)], grid=8, blocks_traced=1,
                  ratio=0.125, kernel="synthetic_gemm")
    base = tmp_path / "rep"
    rc = cli.main(["--ks-synthetic", "--ks-no-nvml", "--ks-memtrace", str(p),
                   "-k", "regex:gemm", "-o", str(base), "-f"])
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    secs = {s["name"]: s for k in doc["kernels"] for s in k["sections"]}
    mt = secs["MemoryTrace"]
    metrics = {m["name"]: m for m in mt["metrics"]}
    assert metrics["meas__sectors_per_request"]["provenance"] == "estimated"
    assert metrics["est__l2_hit_rate_l2only_pct"]["provenance"] == "estimated"
    assert "lts__t_sector_hit_rate" not in metrics      # 不占用 ncu 指标名
