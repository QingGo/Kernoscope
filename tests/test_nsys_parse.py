"""nsys SQLite 解析测试（含 StringIds 间接引用这个真实踩坑）。"""

import sqlite3

from kernoscope.backends.nsys import parse_kernels

SCHEMA = """
create table StringIds (id integer primary key, value text);
create table CUPTI_ACTIVITY_KIND_KERNEL (
    start integer, end integer, deviceId integer, streamId integer,
    demangledName, shortName, mangledName,
    registersPerThread integer, gridX integer, gridY integer, gridZ integer,
    blockX integer, blockY integer, blockZ integer,
    staticSharedMemory integer, dynamicSharedMemory integer, localMemoryPerThread integer,
    gridId integer, correlationId integer
);
"""


def _make_db(path, rows):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("insert into StringIds (id, value) values (?, ?)",
                    [(1, "_Z4addkPfi"), (2, "addk"), (3, "_Z4addkPfi"),
                     (4, "_Z6matmulPfS_S_"), (5, "matmul")])
    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_resolves_string_ids(tmp_path):
    """nsys 存的是整数 id：必须解引用，否则拿到的名字是数字。"""
    db = tmp_path / "t.sqlite"
    _make_db(db, [
        (100, 600, 0, 7, 1, 2, 3, 40, 4096, 1, 1, 128, 1, 1, 0, 49152, 0, 0, 11),
    ])
    ks = parse_kernels(str(db))
    assert len(ks) == 1
    k = ks[0]
    # 默认展示 demangled（cutlass 的 shortName 只是模板类名，如 Kernel2）
    assert k.name == "_Z4addkPfi"
    assert k.demangled == "_Z4addkPfi"
    assert k.short_name == "addk"
    assert k.display_name("demangled") == "_Z4addkPfi"
    assert k.display_name("function") == "addk"
    assert k.duration_ns == 500
    assert k.grid == (4096, 1, 1) and k.block == (128, 1, 1)
    assert k.registers_per_thread == 40
    assert k.dynamic_smem_bytes == 49152
    assert k.threads_per_block == 128
    assert k.total_threads == 4096 * 128


def test_inline_string_names_still_work(tmp_path):
    """少数版本可能内联字符串：同样必须支持。"""
    db = tmp_path / "t2.sqlite"
    _make_db(db, [
        (10, 20, 0, 3, "inline_demangled", "inline_short", "inline_mangled",
         8, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0, 1),
    ])
    ks = parse_kernels(str(db))
    assert ks[0].name == "inline_demangled"
    assert ks[0].demangled == "inline_demangled"
    assert ks[0].display_name("function") == "inline_short"


def test_missing_kernel_table_raises(tmp_path):
    import pytest

    from kernoscope.backends.nsys import NsysError

    db = tmp_path / "empty.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table StringIds (id integer, value text)")
    con.commit()
    con.close()
    with pytest.raises(NsysError, match="no CUPTI_ACTIVITY_KIND_KERNEL"):
        parse_kernels(str(db))
