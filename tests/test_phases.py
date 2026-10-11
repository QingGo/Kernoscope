"""NVTX 阶段归因（`analyze/phases.py`）的契约测试。

全部用手搭的区间在临时 SQLite 里跑：不联网、不需要 GPU、不需要 nsys。重点不在"能跑通"，
而在**归因的算术与口径**是否诚实：

* 跨两个区间的 kernel 必须被**切开**（"kernel 属于它开始的那个区间"是错的答案）；
* 嵌套区间两边都算 ⇒ 份额之和可以 >100%，而且不许被悄悄钳到 100%；
* `unattributed_ms` 是「总 kernel 时间 − 落在任一区间内的时间」，不是「总时间 − 各阶段之和」；
* NVTX 拿不到时给 reason + **零数字**，绝不把 0% 当成"测过了"；
* schema 是**探测**出来的：`end` 列缺失、只有 open range、文本走 `StringIds`，都要各自走对。
"""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

from kernoscope.analyze.phases import (KERNEL_TABLE, NVTX_TABLE, REASON_NO_KERNELS,
                                       REASON_NO_NVTX, PhaseError, _refusal_code, attribute)
from kernoscope.refusals import DEFAULT_REMEDIES, RefusalCode

KERNEL_DDL = (f"create table {KERNEL_TABLE} "
              "(start integer, end integer, deviceId integer, streamId integer)")
# 故意带上 eventType：真实 nsys 有这列，但它的数值含义跨版本不稳定 —— 我们不该解释它。
NVTX_DDL = (f"create table {NVTX_TABLE} "
            "(start integer, end integer, text text, textId integer, globalTid integer, "
            "eventType integer)")
STRINGIDS_DDL = "create table StringIds (id integer primary key, value text)"
NS = 1_000_000.0        # 1 ms = 1e6 ns


def _write(tmp_path, name: str, *, kernels=(), ranges=None,
           kernel_ddl: str = KERNEL_DDL, nvtx_ddl: str = NVTX_DDL, strings=()) -> str:
    """搭一个 nsys 形状的 SQLite。

    kernels 行 = (start, end, deviceId, streamId)；ranges 行 = (start, end, text, textId,
    globalTid, eventType)。`ranges=None` 表示**没有** NVTX 表（没带 --trace=nvtx）。
    """
    db = tmp_path / name
    con = sqlite3.connect(db)
    con.execute(STRINGIDS_DDL)
    con.execute(kernel_ddl)
    if ranges is not None:
        con.execute(nvtx_ddl)
    for i, v in enumerate(strings, 1):
        con.execute("insert into StringIds values (?, ?)", (i, v))
    if kernels:
        con.executemany(
            f"insert into {KERNEL_TABLE} values ({','.join('?' * len(kernels[0]))})", kernels)
    if ranges:
        con.executemany(
            f"insert into {NVTX_TABLE} values ({','.join('?' * len(ranges[0]))})", ranges)
    con.commit()
    con.close()
    return str(db)


def _range(start, end, name, *, tid=1, textid=None, event=59):
    return (start, end, name, textid, tid, event)


def _by_name(att) -> dict:
    return {p.name: p for p in att.phases}


# ---------------------------------------------------------------------------
# 口径 1：schema 靠探测，拿不到就给 reason + 零数字
# ---------------------------------------------------------------------------
def test_no_nvtx_table_is_refused_and_never_reports_zero_as_measured(tmp_path):
    """没有 NVTX 表 = 没做归因：给 reason，且 to_metrics 的每个值都是 None。

    "0.0 ms 未归因" 与 "没有数据可归因" 是两件事；把它们混起来就是把没测过的当测过。
    """
    db = _write(tmp_path, "no_nvtx.sqlite", kernels=[(0, 1000, 0, 7)], ranges=None)
    att = attribute(db)
    assert att.reason.startswith(REASON_NO_NVTX)
    assert att.phases == []
    assert "0 在这里只表示" in att.basis
    metrics = att.to_metrics()
    assert metrics, "拒绝也不能让那一行消失（B1）"
    assert all(m[1] is None for m in metrics), metrics
    assert all("code=no_nvtx" in m[3] for m in metrics), metrics


def test_missing_nvtx_maps_to_its_own_code_with_a_recollect_remedy(tmp_path):
    """V-28：没有 NVTX 区间不是 `metric_absent`（"指标不存在"），而是 `no_nvtx`。

    修前它映射到 `metric_absent`，处置于是变成"用 --json 看有哪些指标"—— 而真正该做的是
    **重采时带 `--trace=nvtx`**。三个断言把这条钉死：
      ① 没有 NVTX ⇒ 自己的 code（不是 metric_absent）；
      ② 这个 code 的 remedy 必须指向 `--trace=nvtx`；
      ③ 它与"没有 kernel 活动"必须是**两个** code（否则读者不知道该补哪趟采集）。
    """
    db = _write(tmp_path, "no_nvtx.sqlite", kernels=[(0, 1000, 0, 7)], ranges=None)
    att = attribute(db)
    metrics = att.to_metrics()
    assert all("code=no_nvtx" in m[3] for m in metrics), metrics
    assert all("code=metric_absent" not in m[3] for m in metrics), metrics

    rem = DEFAULT_REMEDIES[RefusalCode.NO_NVTX]
    assert "--trace=nvtx" in rem.action, rem.action
    # 可执行的命令要把 nvtx 通道真的打开（`--trace=cuda,nvtx`：kernel 与区间一起采）
    assert "nsys" in rem.command and "nvtx" in rem.command, rem.command
    assert rem.cost_kind == "wall_seconds" and rem.cost_s, "重采是要花时间的一件事"

    assert _refusal_code(REASON_NO_KERNELS) is RefusalCode.NO_LAUNCH_ACTIVITY
    assert _refusal_code(REASON_NO_NVTX) is RefusalCode.NO_NVTX
    assert RefusalCode.NO_NVTX is not RefusalCode.NO_LAUNCH_ACTIVITY


def test_nvtx_with_only_open_ranges_is_refused(tmp_path):
    """只有 open range（end IS NULL）：没有终点就没法做区间运算，不给数。"""
    import sqlite3 as _sq

    db = tmp_path / "open.sqlite"
    con = _sq.connect(db)
    con.execute(KERNEL_DDL)
    # 只有 start 的版本：连 end 列都没有（另一种真实存在的形状）
    con.execute(f"create table {NVTX_TABLE} (start integer, text text)")
    con.execute(f"insert into {KERNEL_TABLE} values (0, 1000, 0, 7)")
    con.execute(f"insert into {NVTX_TABLE} values (10, 'push')")
    con.commit()
    con.close()
    att = attribute(str(db))
    assert att.reason.startswith(REASON_NO_NVTX)
    assert "没有 end 列" in att.reason
    assert att.phases == []


def test_open_range_rows_are_excluded_but_counted(tmp_path):
    """end 为 NULL 的行（open range）不能外推成无限长：排除，并在 basis 里报个数。"""
    db = _write(tmp_path, "open_rows.sqlite",
                kernels=[(50, 250, 0, 7)],
                ranges=[_range(0, 100, "closed"), (200, None, "still-open", None, 1, 59)])
    att = attribute(db)
    assert att.reason == ""
    assert [p.name for p in att.phases] == ["closed"]
    assert _by_name(att)["closed"].kernel_ms == pytest.approx(50 / NS)
    # 200ms 之后那段没有区间：open range 不算数
    assert att.unattributed_ms == pytest.approx(150 / NS)
    assert "open range 1 个" in att.basis
    assert "still-open" not in att.basis


def test_no_kernel_activity_is_reported_not_attributed(tmp_path):
    db_empty = _write(tmp_path, "empty_kernels.sqlite", kernels=(), ranges=[_range(0, 100, "f")])
    att = attribute(db_empty)
    assert att.reason.startswith(REASON_NO_KERNELS)
    assert all(m[1] is None for m in att.to_metrics())

    # 两张表都没有时，reason 要把两个原因都说出来（不能只报一个）
    db_none = tmp_path / "nothing.sqlite"
    con = sqlite3.connect(db_none)
    con.execute(STRINGIDS_DDL)
    con.commit()
    con.close()
    both = attribute(str(db_none))
    assert REASON_NO_NVTX in both.reason and REASON_NO_KERNELS in both.reason


# ---------------------------------------------------------------------------
# 口径 2：区间重叠（最强的一条测试 —— 朴素"归属起始区间"会给出不同答案）
# ---------------------------------------------------------------------------
def test_straddling_kernel_is_split_at_the_range_boundary(tmp_path):
    """kernel [50,200] 跨 A[0,100] 与 B[150,300]：两边各只拿**重叠**的 50ns。

    朴素的「kernel 属于它开始的区间」会把 150ns 全给 A、给 B 0ns —— 本测试就是钉死这一点：
    A 必须是 50ns 而不是 150ns，B 必须不是 0。
    """
    db = _write(tmp_path, "straddle.sqlite",
                kernels=[(50, 200, 0, 7), (1000, 1100, 0, 7)],
                ranges=[_range(0, 100, "A"), _range(150, 300, "B")])
    att = attribute(db)
    a, b = _by_name(att)["A"], _by_name(att)["B"]
    assert a.kernel_ms == pytest.approx(50 / NS)
    assert b.kernel_ms == pytest.approx(50 / NS)
    # 朴素答案（150ns 全给 A、B 拿 0）必须被证伪
    assert a.kernel_ms != pytest.approx(150 / NS)
    assert b.kernel_ms != pytest.approx(0.0)
    # 被切开的那 100ns 之外的 150ns（50ns 之前 + 100ns 之后）没有区间
    assert att.attributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms == pytest.approx(150 / NS)
    assert a.kernels == 1 and b.kernels == 1
    assert "跨两个区间的 kernel 会被切开" in att.basis


def test_kernel_covering_a_whole_range_contributes_only_the_range(tmp_path):
    """反方向：kernel [0,1000] 完全包住区间 [400,500] —— 区间只拿 100ns，不是 1000ns。"""
    db = _write(tmp_path, "contain.sqlite",
                kernels=[(0, 1000, 0, 7)],
                ranges=[_range(400, 500, "inner-of-kernel")])
    att = attribute(db)
    p = _by_name(att)["inner-of-kernel"]
    assert p.kernel_ms == pytest.approx(100 / NS)
    assert p.total_ms == pytest.approx(100 / NS)
    assert att.attributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms == pytest.approx(900 / NS)


# ---------------------------------------------------------------------------
# 口径 3：嵌套两边都算，份额可以 >100%，且不做钳位、不当作划分
# ---------------------------------------------------------------------------
def test_nested_ranges_count_in_both_and_shares_can_exceed_100(tmp_path):
    """kernel 同时落在外层与内层区间：两边都计入 ⇒ 份额之和 = 200%。

    这条既是算术测试，也是**文档测试**：>100% 必须被 basis 写明（不是悄悄钳成 100%，
    也不是让读者以为这些阶段合起来就是 100%）。
    """
    db = _write(tmp_path, "nested.sqlite",
                kernels=[(400, 500, 0, 7)],
                ranges=[_range(0, 1000, "outer"), _range(400, 500, "inner")])
    att = attribute(db)
    by = _by_name(att)
    assert by["outer"].kernel_ms == pytest.approx(100 / NS)
    assert by["inner"].kernel_ms == pytest.approx(100 / NS)
    assert by["outer"].depth == 0 and by["inner"].depth == 1
    assert by["outer"].total_ms == pytest.approx(1000 / NS)
    # 没有钳位：两个都还是 100%
    assert by["outer"].share_pct == pytest.approx(100.0)
    assert by["inner"].share_pct == pytest.approx(100.0)
    assert sum(p.share_pct for p in att.phases) == pytest.approx(200.0)
    # 并集口径：attributed 只能算一次
    assert att.attributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms == pytest.approx(0.0)
    for needle in ("不是", "划分", ">100%"):
        assert needle in att.basis, (needle, att.basis)
    # 阶段自己的 note 也必须说它不是划分
    phase_rows = [m for m in att.to_metrics() if "__share_pct" in m[0]]
    assert phase_rows and all("不是划分" in m[3] for m in phase_rows)
    assert max(m[1] for m in phase_rows) == pytest.approx(100.0)   # 没被钳到 50/50


# ---------------------------------------------------------------------------
# 口径 4：unattributed = 总 kernel 时间 − 落在任一区间内的时间
# ---------------------------------------------------------------------------
def test_unattributed_uses_per_kernel_overlap_not_wall_clock(tmp_path):
    """两个 kernel 在 [0,100] 上并发（总 kernel 时间 200ns，墙钟只有 100ns）。

    区间 [0,50] 覆盖每个 kernel 各 50ns ⇒ 归因 100ns、未归因 100ns。
    按墙钟算的人会得到"未归因 50ns"—— 那也是错的：kernel 时间的口径是**逐 kernel 求和**。
    """
    db = _write(tmp_path, "concurrent.sqlite",
                kernels=[(0, 100, 0, 1), (0, 100, 0, 2)],
                ranges=[_range(0, 50, "covered")])
    att = attribute(db)
    p = _by_name(att)["covered"]
    assert p.kernel_ms == pytest.approx(100 / NS)     # 50 + 50
    assert p.kernels == 2
    assert att.attributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms + att.attributed_ms == pytest.approx(200 / NS)


def test_phases_sorted_by_kernel_ms_and_share_uses_all_kernel_time(tmp_path):
    db = _write(tmp_path, "sorted.sqlite",
                kernels=[(0, 100, 0, 7), (100, 200, 0, 7)],
                ranges=[_range(0, 100, "second"), _range(0, 200, "first")])
    att = attribute(db)
    assert [p.name for p in att.phases] == ["first", "second"]
    assert att.phases[0].kernel_ms == pytest.approx(200 / NS)
    assert att.phases[0].share_pct == pytest.approx(100.0)
    assert att.phases[1].share_pct == pytest.approx(50.0)
    assert att.unattributed_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 口径 5：top 截断，省略个数写进 basis
# ---------------------------------------------------------------------------
def test_top_limits_phases_and_reports_omitted_in_basis(tmp_path):
    db = _write(tmp_path, "top.sqlite",
                kernels=[(0, 100, 0, 7), (200, 500, 0, 7), (600, 700, 0, 7)],
                ranges=[_range(0, 100, "P"), _range(200, 500, "Q"), _range(600, 700, "R")])
    att = attribute(db, top=2)
    assert [p.name for p in att.phases] == ["Q", "P"]
    assert att.omitted == 1
    assert "top=2" in att.basis and "省略 1 个" in att.basis
    assert ("ks__nvtx_phase_omitted", 1.0, "", att.basis) in att.to_metrics()

    # top=0：一个阶段都不留，但省略个数仍然要说清楚
    zero = attribute(db, top=0)
    assert zero.phases == [] and zero.omitted == 3 and "省略 3 个" in zero.basis


# ---------------------------------------------------------------------------
# schema 变体：文本来源
# ---------------------------------------------------------------------------
def test_range_text_comes_from_inline_or_stringids(tmp_path):
    db = _write(tmp_path, "text.sqlite",
                kernels=[(0, 10, 0, 7), (20, 30, 0, 7), (40, 50, 0, 7)],
                ranges=[(0, 10, "inline-wins", 1, 1, 59),      # 两列都在 → 内联优先
                        (20, 30, None, 1, 1, 59),              # 只有 textId → 查 StringIds
                        (40, 50, None, None, 1, 59)],          # 都没有 → 未命名
                strings=["from_stringids", "unused"])
    att = attribute(db)
    names = {p.name for p in att.phases}
    assert names == {"inline-wins", "from_stringids", "<unnamed>"}
    assert "inline `text` + `StringIds`(textId)" in att.basis
    assert "未命名 1 个" in att.basis


def test_eventtype_values_are_not_interpreted(tmp_path):
    """eventType 的数值含义跨版本不稳定：换个数值，归因结果必须完全一样（我们不解释它）。"""
    def run(tag, event):
        db = _write(tmp_path, f"ev{tag}.sqlite",
                    kernels=[(0, 100, 0, 7)],
                    ranges=[_range(0, 100, "X", event=event)])
        return attribute(db)
    a, b = run("a", 59), run("b", 34)
    assert [p.name for p in a.phases] == [p.name for p in b.phases] == ["X"]
    assert a.phases[0].kernel_ms == b.phases[0].kernel_ms


def test_nvtx_table_without_text_columns_still_attributes(tmp_path):
    """最小 schema（只有 start/end）也要能归因：区间名退化为 <unnamed>，不是崩掉。"""
    db = _write(tmp_path, "minimal.sqlite",
                kernels=[(0, 100, 0, 7)],
                ranges=[(0, 100)],
                nvtx_ddl=f"create table {NVTX_TABLE} (start integer, end integer)")
    att = attribute(db)
    assert att.reason == ""
    assert [p.name for p in att.phases] == ["<unnamed>"]
    assert att.phases[0].kernel_ms == pytest.approx(100 / NS)
    assert "未命名 1 个" in att.basis


def test_input_database_is_opened_read_only(tmp_path):
    """只读连接：分析不该改动被分析的库（不留 journal/WAL，字节数不变）。"""
    db = _write(tmp_path, "ro.sqlite", kernels=[(0, 100, 0, 7)], ranges=[_range(0, 100, "R")])
    before = (pathlib.Path(db).read_bytes(),
              sorted(p.name for p in pathlib.Path(db).parent.iterdir()))
    attribute(db)
    after = (pathlib.Path(db).read_bytes(),
             sorted(p.name for p in pathlib.Path(db).parent.iterdir()))
    assert before == after



# ---------------------------------------------------------------------------
# 分组口径：两张表都有 globalTid 才分线程；否则声明退化
# ---------------------------------------------------------------------------
def test_thread_join_used_only_when_both_tables_expose_globaltid(tmp_path):
    k_ddl_tid = (f"create table {KERNEL_TABLE} "
                 "(start integer, end integer, deviceId integer, streamId integer, "
                 "globalTid integer)")
    # 有 join key：只有 tid=1 的 kernel 进入 tid=1 的区间
    db_join = _write(tmp_path, "join.sqlite", kernel_ddl=k_ddl_tid,
                     kernels=[(0, 100, 0, 1, 1), (0, 100, 0, 2, 2)],
                     ranges=[_range(0, 100, "t1", tid=1)])
    att = attribute(db_join)
    assert att.attributed_ms == pytest.approx(100 / NS)
    assert att.unattributed_ms == pytest.approx(100 / NS)
    assert "分线程归因" in att.basis
    assert _by_name(att)["t1"].kernels == 1

    # 没有 join key（真实 nsys 的 CUPTI kernel 表就是这样）：退化为整段重叠 —— 两个都进
    db_time = _write(tmp_path, "time.sqlite",
                     kernels=[(0, 100, 0, 1), (0, 100, 0, 2)],
                     ranges=[_range(0, 100, "t1", tid=1)])
    att2 = attribute(db_time)
    assert att2.attributed_ms == pytest.approx(200 / NS)
    assert att2.unattributed_ms == pytest.approx(0.0)
    assert _by_name(att2)["t1"].kernels == 2
    assert "没有做线程/设备过滤" in att2.basis
    assert "声明的" in att2.basis


# ---------------------------------------------------------------------------
# 错误处理：输入错误是异常，口径不成立是 reason
# ---------------------------------------------------------------------------
def test_missing_or_garbage_file_raises_phase_error(tmp_path):
    with pytest.raises(PhaseError, match="sqlite not found"):
        attribute(str(tmp_path / "nope.sqlite"))
    junk = tmp_path / "junk.sqlite"
    junk.write_bytes(b"this is not a sqlite database at all" * 10)
    with pytest.raises(PhaseError):
        attribute(str(junk))


def test_empty_kernel_durations_are_not_silently_kernel_activity(tmp_path):
    """零长 kernel（end == start）不是 kernel 活动：不能拿它凑出"有 kernel"的结论。"""
    db = _write(tmp_path, "zero.sqlite",
                kernels=[(100, 100, 0, 7)],
                ranges=[_range(0, 200, "R")])
    att = attribute(db)
    assert att.reason.startswith(REASON_NO_KERNELS)
    assert all(m[1] is None for m in att.to_metrics())


# ---------------------------------------------------------------------------
# 真机实测的回归（AutoDL 4090 容器）：不带 `-o` 时 NVTX 归因曾经 100% 失效
# ---------------------------------------------------------------------------
def test_phase_attribution_survives_the_temp_sqlite_cleanup(tmp_path, monkeypatch):
    """`--export` 缺席时临时目录会被删掉，而归因发生在**更晚**的报告装配阶段。

    真机证据（AutoDL 4090 容器，不带 `-o`）：
        warn: NVTX 阶段归因失败（PhaseError: sqlite not found: /tmp/kernoscope-…/kernoscope_run.sqlite）
    —— 默认路径（也就是首用路径）上这条功能整个失效，只留一行 warn。现在 `collect()` 在删目录
    **之前**先把归因算好留在 meta 里，`_attach_phases` 优先用它；渲染逻辑两条路径共用一份。
    """
    from kernoscope import cli
    from kernoscope.backends import nsys as N
    from kernoscope.report import Report

    # 用与上面成功用例相同的形状（deviceId 0 / stream 7，区间与 kernel 有重叠）
    db = _write(tmp_path, "run.sqlite",
                kernels=[(50, 250, 0, 7)],
                ranges=[_range(0, 100, "closed")])
    monkeypatch.setattr(N, "find_nsys", lambda: "/usr/local/bin/nsys")
    monkeypatch.setattr(N, "run_profile", lambda *a, **k: str(tmp_path / "run.nsys-rep"))
    monkeypatch.setattr(N, "export_sqlite", lambda *a, **k: db)
    monkeypatch.setattr(N, "parse_kernels", lambda *a, **k: [])

    class _Args:
        ks_synthetic = False
        ks_sqlite = None
        ks_nsys_rep = None
        export = None                     # ← 关键：没有 -o ⇒ 临时目录会被删
        ks_keep_intermediate = False
        target_processes = "all"
        program = ["true"]

    logged: list[str] = []
    _, meta, _ = cli.collect(_Args(), logged.append)

    # ① 归因在删目录之前就算好了（否则真机那行 warn 会回来）
    assert meta.get("_nvtx_attribution") is not None, "没能在清理临时目录之前算出阶段归因"
    # ② 临时目录确实被清掉了（清理纪律没被破坏）
    assert "tmpdir" not in meta

    # ③ 装配阶段用它渲染出 NvtxPhases，且**不许**再出现"归因失败"
    rep = Report()
    rep.meta.update(meta)

    class _Dev:
        sections: list = []

    dev = _Dev()
    cli._attach_phases(rep, _Args(), dev, logged.append)
    assert [s.name for s in dev.sections] == ["NvtxPhases"], [s.name for s in dev.sections]
    assert rep.meta["nvtx_phases"]["count"] == 1
    assert not [m for m in logged if "阶段归因失败" in m], logged
    # 预计算的结果不许再留在 meta 里（否则会被别的消费者看到一份内部对象）
    assert "_nvtx_attribution" not in rep.meta
