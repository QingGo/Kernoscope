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


def test_memtrace_json_metrics_carry_provenance(tmp_path, capsys):
    """R-42：--json 的指标对象与主报告同构（名字前缀 ↔ provenance 一一对应）。"""
    p = _mk_trace(tmp_path / "full.txt", [_coalesced(0x1000), _strided(0x2000, 32)])
    assert cli.main(["memtrace", "--from", str(p), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    metrics = {m["name"]: m for m in doc["metrics"]}
    assert metrics
    assert metrics["meas__sectors_per_request"]["provenance"] == "measured"
    assert metrics["meas__coalescing_efficiency_pct"]["provenance"] == "measured"
    assert metrics["est__l2_hit_rate_l2only_pct"]["provenance"] == "estimated"
    for m in metrics.values():
        assert "provenance" in m, m
        if m["value"] is None:
            assert m["provenance"] == "unavailable", m
        elif m["name"].startswith("meas__"):
            assert m["provenance"] == "measured", m
        else:
            assert m["provenance"] == "estimated", m
    assert "exact" not in {m["provenance"] for m in metrics.values()}

    # 采样轨迹：按采样比放大不是「实测读数」，与主报告一致地降级为 est__ + estimated
    q = _mk_trace(tmp_path / "sampled.txt", [_coalesced(0x1000)], grid=8, blocks_traced=1,
                  ratio=0.125)
    assert cli.main(["memtrace", "--from", str(q), "--json"]) == 0
    sampled = {m["name"]: m for m in json.loads(capsys.readouterr().out)["metrics"]}
    assert sampled
    assert not [n for n in sampled if n.startswith("meas__")], \
        "采样放大的量不得用 meas__ 前缀（R-12）"
    assert sampled["est__sectors_per_request"]["provenance"] == "estimated"
    assert all(m["provenance"] in {"estimated", "unavailable"} for m in sampled.values())


def test_cli_memtrace_bad_file_returns_1(tmp_path, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("garbage\n")
    assert cli.main(["memtrace", "--from", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_report_integration_adds_section_with_right_provenance(monkeypatch, capsys, tmp_path):
    # 采样轨迹 -> 连精确算术也降为 estimated（样本代表性是假设），
    # 且前缀必须跟着出处改（R-12）：meas__ 只能是全量实测，采样放大的用 est__。
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
    assert metrics["est__sectors_per_request"]["provenance"] == "estimated"
    assert metrics["est__l2_hit_rate_l2only_pct"]["provenance"] == "estimated"
    assert "meas__sectors_per_request" not in metrics, "采样放大的量不得用 meas__ 前缀"
    assert "lts__t_sector_hit_rate" not in metrics      # 不占用 ncu 指标名


# ---------------------------------------------------------------- 官方输出转换器
OFFICIAL = """MEMTRACE: CTX 0x562f - LAUNCH - Kernel pc 0x7fb0 - Kernel name k_coalesced(float const*, float*) \
- grid launch id 0 - grid size 1,1,1 - block size 32,1,1 - nregs 0 - shmem 0 - cuda stream id 0
MEMTRACE: CTX 0x562f - grid_launch_id 0 - CTA 0,0,0 - warp 1 - LDG.E - {ldg}
MEMTRACE: CTX 0x562f - grid_launch_id 0 - CTA 0,0,0 - warp 1 - STG.E - {stg}
"""


def _addr_list(base, lanes=32, size=4, stride=None):
    st = size if stride is None else stride
    return " ".join(hex(base + st * i) for i in range(lanes))


def _official(tmp_path, ldg, stg=None, name="official.txt"):
    p = tmp_path / name
    p.write_text(OFFICIAL.format(ldg=ldg, stg=stg or ldg))
    return p


def test_official_converter_known_answers(tmp_path):
    from kernoscope.backends.memtrace import load_official_memtrace

    tr = load_official_memtrace(str(_official(tmp_path, _addr_list(0x1000))), sm_count=4)
    assert tr.kernel.startswith("k_coalesced")
    assert tr.grid_blocks == 1 and tr.block_threads == 32
    assert tr.header["source"] == "official-nvbit-mem-trace"
    assert [len(r.sectors) for r in tr.records] == [4, 4]      # 连续 32×4B = 128B = 4 sector
    assert tr.records[0].size == 4 and tr.records[0].lanes == 32

    # strided32：lane 间隔 32 个 float
    tr2 = load_official_memtrace(str(_official(tmp_path, _addr_list(0x2000, stride=128))), sm_count=4)
    assert len(tr2.records[0].sectors) == 32

    # unaligned：首地址 +4
    tr3 = load_official_memtrace(str(_official(tmp_path, _addr_list(0x3004))), sm_count=4)
    assert len(tr3.records[0].sectors) == 5


def test_official_converter_known_sector_counts_and_smid_approx(tmp_path):
    from kernoscope.backends.memtrace import load_official_memtrace
    from kernoscope.analyze.memtrace import analyze_memtrace

    tr = load_official_memtrace(str(_official(tmp_path, _addr_list(0x1000))), sm_count=4)
    st = analyze_memtrace(tr, l2_bytes=L2_72MB, l1_bytes=128 * 1024)
    assert st.sectors_per_request == pytest.approx(4.0)
    assert st.coalescing_pct == pytest.approx(100.0)
    assert st.smid_approx, "官方输出没有 SM id，必须标注为近似"
    assert tr.records[0].smid == 0                              # cta 0 % 4


def test_official_converter_rejects_empty(tmp_path):
    from kernoscope.backends.memtrace import MemTraceError, load_official_memtrace

    p = tmp_path / "empty.txt"
    p.write_text("MEMTRACE: CTX 0x1 - LAUNCH - Kernel name k() - grid size 1,1,1 - block size 32,1,1\n")
    with pytest.raises(MemTraceError, match="NO_EAGER_LOAD"):
        load_official_memtrace(str(p))


def test_auto_detect_both_formats(tmp_path):
    from kernoscope.backends.memtrace import load_memtrace_auto

    off = _official(tmp_path, _addr_list(0x1000))
    assert load_memtrace_auto(str(off), sm_count=4).header["source"] == "official-nvbit-mem-trace"

    ours = _mk_trace(tmp_path / "ours.txt", [_coalesced(0x1000)])
    tr = load_memtrace_auto(str(ours))
    assert tr.header.get("source") != "official-nvbit-mem-trace"
    assert len(tr.records) == 1


def test_cli_accepts_official_format(tmp_path, capsys):
    off = _official(tmp_path, _addr_list(0x1000))
    assert cli.main(["memtrace", "--from", str(off), "--sm-count", "4",
                     "--l2-bytes", str(L2_72MB), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sectors_per_request"] == pytest.approx(4.0)
    assert doc["coalescing_pct"] == pytest.approx(100.0)
    assert doc["l2_bytes"] == L2_72MB


MULTI = """MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x10 - Kernel name k_A(float*) \
- grid launch id 0 - grid size 2,1,1 - block size 32,1,1
MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 - warp 1 - LDG.E - {a}
MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 1,0,0 - warp 1 - LDG.E - {a}
MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x20 - Kernel name k_B(float*) \
- grid launch id 1 - grid size 1,1,1 - block size 32,1,1
MEMTRACE: CTX 0x1 - grid_launch_id 1 - CTA 0,0,0 - warp 1 - LDG.E - {b}
"""


def test_official_converter_attributes_and_filters_kernels(tmp_path):
    from kernoscope.backends.memtrace import load_official_memtrace

    p = tmp_path / "multi.txt"
    p.write_text(MULTI.format(a=_addr_list(0x1000), b=_addr_list(0x2000, stride=128)))

    # 无过滤：只取第一个 kernel（绝不把两个 kernel 的记录混算）
    tr = load_official_memtrace(str(p), sm_count=4)
    assert tr.kernel.startswith("k_A")
    assert len(tr.records) == 2 and tr.launches_traced == 1
    assert tr.grid_blocks == 2 and tr.blocks_traced == 2
    assert "k_B" in tr.header["kernels_seen"]

    # 过滤到第二个 kernel：拿到 strided 的 32 sectors
    tr2 = load_official_memtrace(str(p), sm_count=4, kernel_filter="k_B")
    assert tr2.kernel.startswith("k_B")
    assert len(tr2.records) == 1
    assert len(tr2.records[0].sectors) == 32
    assert tr2.launches_traced == 1

    # 过滤不中：报错并列出实际看到的 kernel（可操作）
    with pytest.raises(MemTraceError, match="k_A"):
        load_official_memtrace(str(p), sm_count=4, kernel_filter="k_ZZZ")


# --------------------------------------------------------------------------
# R-73：轨迹口径必须按访存类型分流（共享/本地不许进入全局口径）
# --------------------------------------------------------------------------
def test_size_from_opcode_knows_sass_widths_and_ptx_vectors():
    """SASS 的 `STS.U16` / `LDG.E.U8` 与 PTX 的 `v4.f32` 都要算对。

    旧实现只认 PTX 风格的 `.<宽度>`：`STS.U16` 匹配不上 → 回落默认 4 B（应为 2 B），
    而 `ld.global.v4.f32` 被当成 4 B（应为 16 B）。两者都会把合并效率算歪。
    """
    from kernoscope.backends.memtrace import size_from_opcode

    cases = {"LDG.E": 4, "LDG.E.128": 16, "STG.E.64": 8, "STS.U16": 2, "STS.U8": 1,
             "LDG.E.U8": 1, "LDS.64": 8, "ld.global.f32": 4, "ld.global.v4.f32": 16,
             "ld.shared.v2.b32": 8, "ld.global.u16": 2}
    for op, want in cases.items():
        assert size_from_opcode(op) == want, (op, size_from_opcode(op), want)


def test_access_kind_splits_global_from_shared_and_local():
    """`kind` 分类是 R-73 的第一道门：LDS/STS/LDL/STL 不许进全局口径。"""
    from kernoscope.backends.memtrace import access_kind

    for op, want in (("LDG.E.128", "global"), ("STG.E", "global"),
                     ("LDS", "shared"), ("STS.U16", "shared"), ("LDSM.16.M88.4", "shared"),
                     ("LDL", "local"), ("STL", "local"),
                     ("ld.global.v4.f32", "global"), ("st.shared.f32", "shared"),
                     ("ld.local.f32", "local"), ("LD.E", "unknown"), ("BAR.SYNC", "other")):
        assert access_kind(op) == want, (op, access_kind(op), want)


def test_shared_memory_records_do_not_inflate_coalescing(tmp_path):
    """真机原形（验收报告 §10.3）：`STS.U16` 让合并效率变成 **104.4%**（物理上不可能）。

    一个 FlashAttention 轨迹里 1,282,048 条记录只有 8,192 条是全局的。修法是把共享/本地
    访存剔出全局口径（`LDG/STG` 之外不算），并保留 `min(100, …)` 兜底。
    这里按报告的最小复现构造：32 lane × 4 B 连续（4 sector，100%）+
    32 lane × 2 B 共享写（1 sector）—— 旧代码把两者混在一起就会 >100%。
    """
    path = tmp_path / "mixed.txt"
    glob = (0x100, "LDG.E.128", 32, 16, 0, [0x1000 + 16 * i for i in range(32)])
    shm = (0x200, "STS.U16", 32, 2, 0, [0x100 + i for i in range(32)])
    _mk_trace(path, [glob, shm])
    st = analyze_memtrace(load_memtrace(str(path)))
    assert st.kind_counts == {"global": 1, "shared": 1}
    assert st.records_global == 1 and st.records == 2
    assert st.coalescing_pct <= 100.0 + 1e-9, st.coalescing_pct
    assert not st.coalescing_clamped, "正常路径应当是分流把输入挡掉，而不是靠夹紧"
    # 全局那一条：32 lane × 16 B = 512 B 有用字节 / 16 个 sector × 32 B = 512 B → 100%
    assert st.sectors_per_request == pytest.approx(16.0)
    assert st.coalescing_pct == pytest.approx(100.0)


def test_l2_model_ignores_shared_memory_traffic(tmp_path):
    """共享访存的地址与 L2 无关：混进来会污染命中率与 DRAM 字节。"""
    path = tmp_path / "shm.txt"
    rows = [(0x100, "LDG.E.128", 32, 4, 0, [0x2000 + 16 * i for i in range(32)]),
            (0x200, "STS.U16", 32, 2, 0, [0x10 + i for i in range(32)])]
    _mk_trace(path, rows)
    st = analyze_memtrace(load_memtrace(str(path)))
    assert st.working_set_mb == pytest.approx(16 * 32 / 1e6)   # 只算全局那 16 个 sector
    assert st.accesses == 16


def test_max_accesses_refusal_names_the_concrete_number(tmp_path):
    """R-74：拒绝是对的，但文案必须给出**具体要设多大**。

    真机原形（验收报告 §10.7）：R=16384 那次需要 11.17M 次访问，默认 4M ⇒ 报告给 n/a。
    工具当时说"只分析了 4M/11.17M 次访问"，但没说"把 --ks-max-accesses 提到 ≥ 11.17M"。
    """
    path = tmp_path / "many.txt"
    rows = [(0x100, "LDG.E", 32, 4, 0, [0x1000 + 4 * i for i in range(32)])] * 50
    _mk_trace(path, rows)
    st = analyze_memtrace(load_memtrace(str(path)), max_accesses=10)
    reason = st.dram_bytes_unavailable_reason or ""
    assert st.dram_bytes_per_launch is None and reason
    need = int(st.accesses)
    assert f"--ks-max-accesses` 提到 ≥ {need}" in reason or f"≥ {need}" in reason, reason


def test_trace_bounds_bracket_the_cache_model_value(tmp_path):
    """R-77：同一份轨迹要给出替换策略的**上下界**，模型值必须落在中间。

    地址部分本来就是算术（可逐条复核），只有"谁命中 L2"是模型；界的存在让读者能判断
    结论对模型有多敏感（报告 §10.5 手算过这两个界）。
    """
    path = tmp_path / "reuse.txt"
    base = 0x10000
    # 同一个 sector 被访问 4 次（有复用）+ 另外 4 个只访问一次
    rows = [(0x100, "LDG.E", 32, 4, 0, [base]) for _ in range(4)]
    rows += [(0x100, "LDG.E", 32, 4, 0, [base + 32 * (i + 1)]) for i in range(4)]
    _mk_trace(path, rows)
    st = analyze_memtrace(load_memtrace(str(path)))
    assert st.dram_bytes_lower is not None and st.dram_bytes_upper is not None
    assert st.dram_bytes_lower <= st.dram_bytes_per_launch <= st.dram_bytes_upper
    from kernoscope.analyze.memtrace import to_metrics

    names = {m[0] for m in to_metrics(st)}
    assert "est__dram_bytes_per_launch_trace_bounds" in names
    note = next(m[4] for m in to_metrics(st) if m[0] == "est__dram_bytes_per_launch_trace")
    assert "measured 级算术" in note and "模型" in note


def test_official_trace_completeness_sentinel_is_detected(tmp_path):
    """R-76：哨兵行只有一段（`MEMTRACE: KS-TRACE-COMPLETE`），必须在切分前判定。

    真事：第一版把检查放在 `len(parts) < 2` 之后，于是哨兵永远被当成"格式不对的行"跳过，
    `complete` 一直是 `unknown` —— 冒烟门禁（`test_nvbit_smoke.py`）在真机上把它抓了出来。
    """
    from kernoscope.backends.memtrace import COMPLETE_SENTINEL, load_official_memtrace

    body = ["MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x2 - Kernel name k(float*) "
            "- grid launch id 0 - grid size 1,1,1 - block size 32,1,1",
            "MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 - warp 0 - LDG.E - 0x1000 0x1010"]
    without = tmp_path / "without.txt"
    without.write_text("\n".join(body) + "\n", encoding="utf-8")
    with_sentinel = tmp_path / "with.txt"
    with_sentinel.write_text("\n".join(body) + f"\nMEMTRACE: {COMPLETE_SENTINEL}\n",
                             encoding="utf-8")
    a = load_official_memtrace(str(without), sm_count=1)
    b = load_official_memtrace(str(with_sentinel), sm_count=1)
    assert a.complete == "unknown" and b.complete == "yes"
    assert a.header["trace_sha256"] != b.header["trace_sha256"], "两份轨迹的指纹必须不同"


# --------------------------------------------------------------------------
# R-78（第五轮验收 V-5）：全局原子/归约**也是全局访存**；0 条记录 ≠ 测到 0
#
# 验收方在 RTX 4090（sm_89）上用 NVBit 1.7.x 采的 `atomicAdd` 轨迹里，opcode 全是
# `RED.E.ADD.F32.FTZ.RN.STRONG.GPU`。旧实现只认 LDG/STG（与含 GLOBAL 的字样），于是
# 256 条记录全被剔出全局口径：sectors/request 报 `0.0 [measured]`（看起来像"测到 0"），
# 且 DRAM 字节拒绝时还说"轨迹头部缺少 l2_bytes"—— 而用户明明给了 `--l2-bytes`。
#
# 下面的合成数据照这个形态造（**不含任何机器身份**：没有主机名/IP/绝对路径）。
# --------------------------------------------------------------------------
# 各架构实测形态（`cuobjdump -sass`；sm_75/80/86/89 与 sm_90a/sm_100 各自一组）
GLOBAL_ATOMIC_OPCODES = (
    "RED.E.ADD.F32.FTZ.RN.STRONG.GPU",      # sm_75..sm_89 的 atomicAdd（真机采到）
    "RED.E.ADD.STRONG.GPU",                 # 泛型指针归约（真机采到）
    "RED.E.ADD.F32.FTZ.RN",                 # 不带作用域词的裸归约
    "REDG.E.ADD.F32.FTZ.RN.STRONG.GPU",     # sm_90a/sm_100 带 G 后缀
    "REDG.E.ADD.STRONG.GPU",
    "ATOMG.E.CAS.STRONG.GPU",               # 真机采到的 atomicCAS
    "ATOMG.E.ADD", "ATOM.E.CAS", "ATOM.E.ADD.STRONG.SYS", "CAS",
    "red.global.add.f32", "atom.global.cas.b32", "atom.acq_rel.gpu.global.add.u32",
)
NON_GLOBAL_ATOMIC_OPCODES = {
    "ATOMS.ADD": "shared",                  # SASS 的共享原子（S 后缀）
    "ATOMS.POPC.INC.32": "shared",          # 真机采到的共享 atomicAdd
    "ATOMS.CAS": "shared",
    "REDS.ADD": "shared",
    "RED.SHARED.ADD.F32": "shared",
    "ATOM.SHARED.ADD": "shared",
    "RED.LOCAL.ADD": "local",
    "atom.shared.cas.b32": "shared",
    "atom.local.add.u32": "local",
}


def _acceptance_atomic_rows(n_warps=256, lanes=32, stride_elems=37):
    """验收报告里那个 kernel 的地址形态：`atomicAdd(&y[(i*37) & (2^20-1)], 1.0f)`。

    lane 之间间隔 37 个 float（148 B）⇒ 每个 warp 触达 32 个互不重叠的 32B sector。
    """
    mask = (1 << 20) - 1
    rows = []
    for w in range(n_warps):
        addrs = [((w * lanes + lane) * stride_elems & mask) * 4 for lane in range(lanes)]
        rows.append((0x30, "RED.E.ADD.F32.FTZ.RN.STRONG.GPU", lanes, 4, w % 2, addrs))
    return rows


def test_access_kind_counts_global_atomics_as_global():
    """全局原子/归约 → global；共享/本地原子**不进**全局口径；判不出的仍归 unknown。"""
    from kernoscope.backends.memtrace import access_kind

    for op in GLOBAL_ATOMIC_OPCODES:
        assert access_kind(op) == "global", (op, access_kind(op))
    for op, want in NON_GLOBAL_ATOMIC_OPCODES.items():
        assert access_kind(op) == want, (op, access_kind(op), want)
    # 现有行为不能被这次改动带走
    for op, want in (("LDG.E.128", "global"), ("STG.E", "global"),
                     ("LDS", "shared"), ("STS.U16", "shared"), ("LDSM.16.M88.4", "shared"),
                     ("LDL", "local"), ("STL", "local"),
                     ("ld.global.v4.f32", "global"), ("st.shared.f32", "shared"),
                     ("ld.local.f32", "local"), ("LD.E", "unknown"), ("ST.E", "unknown"),
                     ("MULTIMEM.RED", "unknown"), ("BAR.SYNC", "other")):
        assert access_kind(op) == want, (op, access_kind(op), want)


def test_global_atomic_trace_matches_acceptance_numbers(tmp_path):
    """真机那种 opcode 形态要能算出验收方独立重算的那组数（R-78 的核心回归）。

    8×1024 线程 = 256 个 warp；每个 warp 触达 32 个散射 sector：
        sectors/request 32.0、合并效率 12.5%、工作集 0.262 MB、DRAM 262,144 B（全 miss）。
    """
    path = _mk_trace(tmp_path / "atom.txt", _acceptance_atomic_rows(), grid=8, blocks_traced=8)
    st = analyze_memtrace(load_memtrace(str(path)), l2_bytes=75497472)
    assert st.kind_counts == {"global": 256}, st.kind_counts
    assert st.records_global == 256 and st.records == 256
    assert st.sectors_per_request == pytest.approx(32.0)
    assert st.coalescing_pct == pytest.approx(12.5)
    assert st.working_set_mb == pytest.approx(256 * 32 * 32 / 1e6)      # 0.262144 MB
    assert st.dram_bytes_per_launch == pytest.approx(256 * 32 * 32)     # 262,144 B
    assert st.dram_bytes_unavailable_reason == ""


def test_mixed_atomic_and_shared_traffic_keeps_global_caliber_clean(tmp_path):
    """一条轨迹里混着全局原子、共享原子与共享访存时，只有全局那条进全局口径。"""
    path = tmp_path / "mixed.txt"
    rows = [(0x30, "RED.E.ADD.F32.FTZ.RN.STRONG.GPU", 32, 4, 0,
             [0x1000 + 4 * i for i in range(32)]),                    # 4 sector，100%
            (0x40, "ATOMS.POPC.INC.32", 32, 4, 0, [0x10 + i for i in range(32)]),
            (0x50, "LDS", 32, 4, 0, [0x20 + 4 * i for i in range(32)]),
            (0x60, "STG.E", 32, 4, 0, [0x5000 + 4 * i for i in range(32)])]
    _mk_trace(path, rows)
    st = analyze_memtrace(load_memtrace(str(path)), l2_bytes=L2_72MB)
    assert st.kind_counts == {"global": 2, "shared": 2}, st.kind_counts
    assert st.records_global == 2
    # 全局口径只算那两条：各 4 个 sector、各 128 B 有用字节
    assert st.sectors_per_request == pytest.approx(4.0)
    assert st.coalescing_pct == pytest.approx(100.0)
    assert st.working_set_mb == pytest.approx(8 * 32 / 1e6)
    assert not st.coalescing_clamped


def test_zero_global_records_is_unavailable_not_measured_zero(tmp_path, capsys):
    """全局口径 0 条记录 ⇒ 那一串指标是 `unavailable` + code，绝不是 `0.0 [measured]`。"""
    from kernoscope.analyze.memtrace import split_unavailable_note, to_metrics
    from kernoscope.refusals import RefusalCode

    path = tmp_path / "shared-only.txt"
    _mk_trace(path, [(0x10, "LDS", 32, 4, 0, [0x100 + 4 * i for i in range(32)]),
                     (0x20, "STS.U16", 32, 2, 0, [0x200 + 2 * i for i in range(32)]),
                     (0x30, "ATOMS.POPC.INC.32", 32, 4, 0, [0x10 + i for i in range(32)])])
    st = analyze_memtrace(load_memtrace(str(path)), l2_bytes=L2_72MB)
    assert st.records_global == 0 and st.kind_counts == {"shared": 3}

    metrics = {name: (value, note) for name, value, _u, _s, note in to_metrics(st)}
    for name in ("meas__sectors_per_request", "meas__coalescing_efficiency_pct",
                 "meas__trace_working_set_mb", "meas__trace_records_global",
                 "est__l2_hit_rate_l2only_pct", "est__l2_hit_rate_with_l1_pct"):
        value, note = metrics[name]
        assert value is None, (name, value)
        code, why = split_unavailable_note(note)
        assert code is RefusalCode.TRACE_UNAVAILABLE, (name, code)
        assert "而不是 0" in why
    # 全部记录数仍然是实测值：拒绝只针对"全局口径"那一族
    assert metrics["meas__trace_records"][0] == pytest.approx(3.0)

    assert cli.main(["memtrace", "--from", str(path), "--l2-bytes", str(L2_72MB)]) == 0
    out = capsys.readouterr().out
    assert "meas__sectors_per_request" in out
    assert "n/a" in out and "code=trace_unavailable" in out
    assert "meas__sectors_per_request : 0.0" not in out
    assert "meas__trace_records_global: 0.0" not in out

    assert cli.main(["memtrace", "--from", str(path), "--l2-bytes", str(L2_72MB),
                     "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    m = {x["name"]: x for x in doc["metrics"]}
    assert m["meas__sectors_per_request"]["value"] is None
    assert m["meas__sectors_per_request"]["provenance"] == "unavailable"
    assert m["meas__sectors_per_request"]["refusal"]["code"] == "trace_unavailable"
    assert m["meas__trace_records"]["value"] == pytest.approx(3.0)
    assert doc["dram_bytes_per_launch"] is None
    assert "全局口径 0 条记录" in doc["dram_bytes_unavailable_reason"]


def test_unrecognized_opcodes_are_listed_so_the_gap_is_visible(tmp_path):
    """分类缺口要一眼可见：拒绝理由里列出实际出现的 opcode 形态（不猜、但要说清）。"""
    path = tmp_path / "unknown-ops.txt"
    _mk_trace(path, [(0x1, "LD.E", 32, 4, 0, [0x1000 + 4 * i for i in range(32)]),
                     (0x2, "MULTIMEM.RED", 32, 4, 0, [0x2000 + 4 * i for i in range(32)])])
    st = analyze_memtrace(load_memtrace(str(path)), l2_bytes=L2_72MB)
    assert st.kind_counts == {"unknown": 2} and st.records_global == 0
    why = st.dram_bytes_unavailable_reason
    assert "LD.E×1" in why and "MULTIMEM.RED×1" in why
    assert "这不是" in why and "0 字节" in why


def test_dram_refusal_reason_tracks_whether_l2_was_given(tmp_path, capsys, monkeypatch):
    """拒绝理由不许张冠李戴：给了 `--l2-bytes` 就不能说"缺 l2_bytes"（验收 V-5）。"""
    # ① 只有共享访存（全局口径 0 条）+ **给了** --l2-bytes
    given = tmp_path / "shared-only.txt"
    _mk_trace(given, [(0x10, "LDS", 32, 4, 0, [0x100 + 4 * i for i in range(32)])])
    assert cli.main(["memtrace", "--from", str(given), "--l2-bytes", str(L2_72MB)]) == 0
    out = capsys.readouterr().out
    assert "全局口径 0 条记录" in out
    assert "缺少 l2_bytes" not in out and "没有 l2_bytes" not in out

    # ② 真的没给（轨迹头也没有）⇒ 理由必须点明 l2_bytes 与具体参数名
    nol2 = _mk_trace(tmp_path / "nol2.txt", [_coalesced(0x1000)], l2=0)
    st = analyze_memtrace(load_memtrace(str(nol2)))
    assert st.dram_bytes_per_launch is None
    why = st.dram_bytes_unavailable_reason
    assert "l2_bytes" in why and "--l2-bytes" in why and "--ks-l2-bytes" in why

    # ②b 同一条前提在 CLI 上也要成立（--l2-bytes 不给、本机 nvidia-smi 也拿不到）
    def _no_smi():
        raise RuntimeError("本机没有 nvidia-smi")

    monkeypatch.setattr("kernoscope.backends.nsys.device_info_from_smi", _no_smi)
    assert cli.main(["memtrace", "--from", str(nol2)]) == 0
    out2 = capsys.readouterr().out
    assert "--l2-bytes" in out2 and "l2_bytes" in out2

    # ③ 给了 l2 但访问数超上限：理由只讲 max_accesses，不许把锅甩给 l2_bytes
    many = _mk_trace(tmp_path / "many.txt", [_coalesced(0x100000 + 0x1000 * i) for i in range(50)])
    st3 = analyze_memtrace(load_memtrace(str(many)), l2_bytes=L2_72MB, max_accesses=10)
    assert st3.dram_bytes_per_launch is None
    assert "上限" in st3.dram_bytes_unavailable_reason
    assert "l2_bytes" not in st3.dram_bytes_unavailable_reason


def test_report_section_marks_zero_global_metrics_unavailable(monkeypatch, tmp_path):
    """主报告同构：MemoryTrace section 里这些行是 n/a + code，而不是 0.0 [measured]。"""
    path = tmp_path / "shared-only.txt"
    _mk_trace(path, [(0x10, "LDS", 32, 4, 0, [0x100 + 4 * i for i in range(32)])],
              kernel="synthetic_gemm")
    base = tmp_path / "rep"
    rc = cli.main(["--ks-synthetic", "--ks-no-nvml", "--ks-memtrace", str(path),
                   "-k", "regex:gemm", "-o", str(base), "-f"])
    assert rc == 0
    doc = json.loads((tmp_path / "rep.ksreport.json").read_text())
    secs = {s["name"]: s for k in doc["kernels"] for s in k["sections"]}
    metrics = {m["name"]: m for m in secs["MemoryTrace"]["metrics"]}
    for name in ("meas__sectors_per_request", "meas__coalescing_efficiency_pct",
                 "meas__trace_working_set_mb", "meas__trace_records_global"):
        assert metrics[name]["value"] is None, (name, metrics[name])
        assert metrics[name]["provenance"] == "unavailable"
        assert metrics[name]["refusal"]["code"] == "trace_unavailable"
    assert metrics["meas__trace_records"]["value"] == pytest.approx(1.0)
    dram = metrics["est__dram_bytes_per_launch_trace"]
    assert dram["provenance"] == "unavailable"
    assert dram["refusal"]["code"] == "trace_unavailable"
    assert "全局口径 0 条记录" in dram["reason"]


def test_official_atomic_lines_reach_global_caliber(tmp_path, capsys):
    """官方 `mem_trace` 输出里的原子数据行（验收方用的那条路径）同样算全局访存。"""
    from kernoscope.backends.memtrace import load_official_memtrace

    body = ["MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x10 - Kernel name k_atomic(float*) "
            "- grid launch id 0 - grid size 8,1,1 - block size 1024,1,1"]
    for w in range(4):
        addrs = " ".join(hex(0x100000 + (w * 32 + lane) * 148) for lane in range(32))
        body.append(f"MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 - warp {w} - "
                    f"RED.E.ADD.F32.FTZ.RN.STRONG.GPU - {addrs}")
    p = tmp_path / "official-atomic.txt"
    p.write_text("\n".join(body) + "\n", encoding="utf-8")

    tr = load_official_memtrace(str(p), sm_count=8)
    assert [r.kind for r in tr.records] == ["global"] * 4
    st = analyze_memtrace(tr, l2_bytes=L2_72MB)
    assert st.records_global == 4
    assert st.sectors_per_request == pytest.approx(32.0)
    assert st.dram_bytes_per_launch is not None

    assert cli.main(["memtrace", "--from", str(p), "--sm-count", "8",
                     "--l2-bytes", str(L2_72MB), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["records_global"] == 4
    assert doc["sectors_per_request"] == pytest.approx(32.0)
    assert doc["dram_bytes_per_launch"] is not None


# --------------------------------------------------------------------------- #
# 第五轮 R-75/R-76：v1 路径也要有指纹与"结尾换行"的截断判定
# --------------------------------------------------------------------------- #
def test_v1_trace_carries_a_fingerprint_and_truncation_flag(tmp_path):
    """v1 路径此前只认工具自己写的 `# truncated 1`：

    手动重定向（工具来不及写那一行）/ 被 ulimit 截断的文件，照样给数。
    现在与官方路径同一纪律：指纹 + 结尾换行 + 显式结束标记。
    """
    p = tmp_path / "v1.txt"
    body = ("# ks-memtrace v1\n# kernel k(float*)\n# l2_bytes 75497472\n"
            "# sm_count 128\n# launches_traced 1\npc,op,lanes,size,sectors,smid,addrs\n"
            "0x1,LDG.E,32,4,4,0,0;32;64;96\n")
    p.write_text(body, encoding="utf-8")          # 结尾**有**换行
    tr = load_memtrace(str(p))
    assert tr.header.get("trace_sha256"), "v1 也要写内容指纹（R-75）"
    assert not tr.truncated and tr.complete == "unknown"   # 没结束标记 ⇒ 不假装完整

    p2 = tmp_path / "v1_cut.txt"
    p2.write_text(body.rstrip("\n"), encoding="utf-8")     # 结尾**没有**换行 ⇒ 截断
    tr2 = load_memtrace(str(p2))
    assert tr2.truncated, "结尾不是换行 ⇒ 必须判截断（旧实现只看 # truncated）"
    assert tr2.complete == "no"
    # 同一内容两次读的指纹稳定；换一个字节就变
    assert tr2.header["trace_sha256"] == load_memtrace(str(p2)).header["trace_sha256"]
    p3 = tmp_path / "v1_other.txt"
    p3.write_text(body.replace("0x1,", "0x2,"), encoding="utf-8")
    assert load_memtrace(str(p3)).header["trace_sha256"] != tr.header["trace_sha256"]


def test_v1_trace_with_complete_sentinel_is_marked_complete(tmp_path):
    """显式结束标记（`KS-TRACE-COMPLETE`）才算完整 —— 与官方路径同一判定。"""
    p = tmp_path / "v1_done.txt"
    p.write_text("# ks-memtrace v1\n# kernel k(float*)\npc,op,lanes,size,sectors,smid,addrs\n"
                 "0x1,LDG.E,32,4,4,0,0;32;64;96\n# KS-TRACE-COMPLETE\n", encoding="utf-8")
    tr = load_memtrace(str(p))
    assert tr.complete == "yes" and not tr.truncated


def test_truncated_v1_trace_refuses_per_launch_bytes(tmp_path):
    """截断 ⇒ 拒绝给 DRAM 字节（总量只是下界，不是实测值）—— 两条路径同一个后果。"""
    from kernoscope.analyze.memtrace import analyze_memtrace

    p = tmp_path / "v1_cut2.txt"
    p.write_text("# ks-memtrace v1\n# kernel k(float*)\n# l2_bytes 75497472\n# sm_count 128\n"
                 "# launches_traced 1\npc,op,lanes,size,sectors,smid,addrs\n"
                 "0x1,LDG.E,32,4,4,0,0;32;64;96", encoding="utf-8")   # 无结尾换行
    tr = load_memtrace(str(p))
    st = analyze_memtrace(tr, l2_bytes=75497472, max_accesses=1_000_000)
    assert st.truncated
    assert st.dram_bytes_per_launch is None, "截断的轨迹不许给 per-launch 字节"
    assert "截断" in (st.dram_bytes_unavailable_reason or "")
