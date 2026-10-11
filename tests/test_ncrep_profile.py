"""``kernoscope.ncrep`` 的 M2 契约：非空报告 + 归因硬门 + 读端能回读的字节形状。

本地只用标准库：本文件**不**导入 protobuf / ncu_report / 任何第三方包。这里钉住的是
"我们写出来的字节就是官方读端要的那些字节"，以及"没有出处就不许写文件"这条红线。

官方读端（``ncu_report.load_report()`` / ``ncu --import``）的验收在远端单独跑，
本文件把它固化成可本地回归的结构断言：

* 容器：``NVR\\0`` + ``uint32 LE len(FileHeader)`` + FileHeader，随后逐块
  ``uint32 LE len(BlockHeader)`` + BlockHeader + ``PayloadSize`` 字节负载，
  负载本身是 ``[uint32 LE size][message]``；``PayloadSize`` **含** 这 4 字节前缀。
* ``ProfileResult`` 的 12 个 proto2 required 字段一个都不能少（少一个官方读端会把整块
  丢掉、``num_ranges()`` 归零，而不是报错），所以这里逐个钉住字段号与是否发出。
* 指标名走 ``BlockHeader.StringTable``，``ProfileMetricResult.NameId`` 是它的下标；
  指标值只发 ``ProfileMetricValue.DoubleValue``（float/uint32 也能编码，但双精度是
  唯一被验证过稳定回读的形状）。
"""

import struct

import pytest

from kernoscope.ncrep import (
    NcuRepAttributionError,
    RepMetric,
    RepSource,
    build_report,
    write_report,
)
from kernoscope.ncrep.profile import (
    BLOCK_HEADER,
    PROFILE_METRIC_RESULT,
    PROFILE_METRIC_VALUE,
    PROFILE_RESULT,
    PROFILE_SECTION,
    STRING_TABLE,
    UINT64X3,
    SOURCE_DATA,
)
from kernoscope.ncrep.writer import (
    BLOCK_HEADER as WIRE_BLOCK_HEADER,
    FILE_HEADER,
    FILE_VERSION,
    MAGIC,
    SESSION_DETAILS,
)

# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #

SOURCE = RepSource(
    kernel_name="synthetic_gemm(float const*, float const*, float*)",
    grid=(4, 1, 1),
    block=(32, 1, 1),
)

METRICS = (
    RepMetric("ks__cost_per_mtok", 42.5, "usd/1e6tok"),
    RepMetric("est__kernel_energy_per_launch_j", 1.25e-4, "J"),
)

PROVIDER = "kernoscope"
COMMENT = "synthetic cost model"

#: The twelve fields ``ProfileResult`` declares ``required``. The official reader
#: enforces proto2 required-field semantics on read, so all twelve must be on the
#: wire -- dropping any one costs the whole block.
PROFILE_RESULT_REQUIRED = (
    "ThreadID", "APICallID", "ProgramHandle", "KernelID", "KernelMangledName",
    "KernelFunctionName", "KernelDemangledName", "WorkDimensions",
    "GlobalWorkOffset", "GridSize", "BlockSize", "Source",
)


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode one varint; returns ``(value, next_offset)``."""
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _header_field(header: bytes, field_number: int) -> int | None:
    """Independent scan for a varint field inside a raw BlockHeader.

    Deliberately does not use :mod:`kernoscope.ncrep.schema`: the container
    framing has to be checkable without the codec that produced it.
    """
    offset = 0
    while offset < len(header):
        tag, offset = _varint(header, offset)
        number, wire_type = tag >> 3, tag & 0x07
        if wire_type == 0:
            value, offset = _varint(header, offset)
            if number == field_number:
                return value
        elif wire_type == 2:
            length, offset = _varint(header, offset)
            offset += length
        elif wire_type == 5:
            offset += 4
        elif wire_type == 1:
            offset += 8
        else:  # pragma: no cover - our writer never emits groups
            raise AssertionError(f"unexpected wire type {wire_type}")
    return None


def _walk_container(data: bytes) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Split a report into ``(file_header, [(block_header, payload), ...])``.

    Returns the raw header bytes untouched so tests can assert on them directly.
    Raises if the file is not exactly consumed, which is the property that makes
    a truncated report detectable.
    """
    assert data[:4] == MAGIC, "container must start with the NVR\\0 magic"
    offset = 4
    (header_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    file_header = data[offset:offset + header_len]
    offset += header_len

    blocks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        (header_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        block_header = data[offset:offset + header_len]
        offset += header_len
        payload_size = _header_field(block_header, 5)
        assert payload_size is not None, "every block must carry PayloadSize"
        payload = data[offset:offset + payload_size]
        assert len(payload) == payload_size, "payload shorter than PayloadSize"
        offset += payload_size
        blocks.append((block_header, payload))
    assert offset == len(data), "file must be consumed exactly (no trailing slack)"
    return file_header, blocks


def _profile_result(data: bytes) -> tuple[bytes, dict, list[str]]:
    """Return ``(block_header, profile_result, string_table)`` for the report."""
    _file_header, blocks = _walk_container(data)
    assert len(blocks) == 2, "one session block + one result block"
    block_header, payload = blocks[1]
    (frame_len,) = struct.unpack_from("<I", payload, 0)
    assert frame_len == len(payload) - 4, "PayloadSize counts the uint32 prefix"
    decoded_header = BLOCK_HEADER.decode(block_header)
    return (
        block_header,
        PROFILE_RESULT.decode(payload[4:]),
        decoded_header["StringTable"]["Strings"],
    )


def _build(**overrides):
    kwargs = {
        "source": SOURCE,
        "metrics": METRICS,
        "provider": PROVIDER,
        "comment": COMMENT,
    }
    kwargs.update(overrides)
    return build_report(**kwargs)


# --------------------------------------------------------------------------- #
# the red gate: no attribution, no file
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": ""},
        {"comment": ""},
        {"provider": "", "comment": ""},
        {"provider": "   "},
        {"comment": "\t\n "},
        {"provider": None},
        {"comment": None},
    ],
)
def test_build_report_refuses_missing_attribution(kwargs):
    with pytest.raises(NcuRepAttributionError):
        _build(**kwargs)


def test_attribution_must_be_passed_at_all():
    """Keyword-only and without defaults: an unattributed report is unspellable."""
    with pytest.raises(TypeError):
        build_report(SOURCE, METRICS)
    with pytest.raises(TypeError):
        build_report(SOURCE, METRICS, provider=PROVIDER)
    with pytest.raises(TypeError):
        build_report(SOURCE, METRICS, comment=COMMENT)


def test_attribution_must_be_text():
    with pytest.raises(TypeError):
        _build(provider=7)
    with pytest.raises(TypeError):
        _build(comment=b"bytes")


def test_write_report_refuses_and_writes_nothing(tmp_path):
    target = tmp_path / "unattributed.ncu-rep"
    with pytest.raises(NcuRepAttributionError):
        write_report(target, SOURCE, METRICS, provider="", comment=COMMENT)
    with pytest.raises(NcuRepAttributionError):
        write_report(target, SOURCE, METRICS, provider=PROVIDER, comment="")
    assert not target.exists(), "a refused report must not leave a file behind"


def test_gate_does_not_reject_a_complete_attribution():
    """The red gate must stay usable: a truthful provider/comment pair passes."""
    data = _build(provider="kernoscope ncrep", comment="synthetic cost model, no hardware")
    assert len(data) > 10


# --------------------------------------------------------------------------- #
# container framing
# --------------------------------------------------------------------------- #


def test_report_is_non_empty_and_starts_with_the_m1_prologue():
    data = _build()
    assert len(data) > 10, "M2 output must be a real report, not the 10-byte stub"
    assert data[:10] == MAGIC + b"\x02\x00\x00\x00\x08\x07"
    file_header, blocks = _walk_container(data)
    assert FILE_HEADER.decode(file_header) == {"Version": FILE_VERSION}
    assert len(blocks) == 2


def test_session_block_carries_attribution():
    _file_header, blocks = _walk_container(_build())
    header, payload = blocks[0]
    decoded = BLOCK_HEADER.decode(header)
    assert payload == b"", "the session block carries no payload"
    assert decoded["NumSources"] == 0
    assert decoded["NumResults"] == 0
    assert decoded["PayloadSize"] == 0
    assert decoded["SessionDetails"] == {
        "Comments": COMMENT,
        "VersionInfo": [{"Provider": PROVIDER, "Version": "unknown"}],
    }


def test_result_block_declares_one_result_and_counts_the_length_prefix():
    header, payload = _walk_container(_build())[1][1]
    decoded = BLOCK_HEADER.decode(header)
    (frame_len,) = struct.unpack_from("<I", payload, 0)
    assert decoded["NumSources"] == 0
    assert decoded["NumResults"] == 1
    assert decoded["PayloadSize"] == len(payload) == 4 + frame_len
    assert decoded["StringTable"]["Strings"] == [m.name for m in METRICS]


def test_string_table_is_deduplicated_and_ordered_first_seen():
    metrics = (
        RepMetric("b", 1.0),
        RepMetric("a", 2.0),
        RepMetric("b", 3.0),
    )
    _header, _result, strings = _profile_result(_build(metrics=metrics))
    assert strings == ["b", "a"]
    _header, result, strings = _profile_result(_build(metrics=metrics))
    assert [m["NameId"] for m in result["MetricResults"]] == [0, 1, 0]
    assert result["MetricResults"][2]["MetricValue"] == {"DoubleValue": 3.0}


# --------------------------------------------------------------------------- #
# ProfileResult: the twelve required fields and the identity payload
# --------------------------------------------------------------------------- #


def test_all_required_profile_result_fields_are_emitted():
    _header, result, _strings = _profile_result(_build())
    missing = [name for name in PROFILE_RESULT_REQUIRED if name not in result]
    assert not missing, f"reader drops the block without these: {missing}"
    assert set(result) == set(PROFILE_RESULT_REQUIRED) | {"MetricResults", "Sections"}


def test_declared_required_fields_match_the_reader_contract():
    declared = {f.name for f in PROFILE_RESULT.fields}
    assert set(PROFILE_RESULT_REQUIRED) <= declared


def test_kernel_identity_round_trips_through_our_own_decoder():
    _header, result, _strings = _profile_result(_build())
    assert result["KernelFunctionName"] == "synthetic_gemm"
    assert result["KernelDemangledName"] == SOURCE.kernel_name
    # The schema requires a mangled name; we cannot mangle, so it stays empty
    # rather than duplicating the demangled string.
    assert result["KernelMangledName"] == ""
    assert "KernelMangledName" in result


@pytest.mark.parametrize(
    "kernel_name,expected",
    [
        ("addk(float *, int)", "addk"),
        ("addk", "addk"),
        ("void cutlass::Kernel3<cutlass_80_tensorop>(T1::Params)",
         "void cutlass::Kernel3<cutlass_80_tensorop>"),
        ("  spaced (int)  ", "spaced"),
    ],
)
def test_function_name_is_the_text_before_the_first_paren(kernel_name, expected):
    source = RepSource(kernel_name, (1, 1, 1), (1, 1, 1))
    _header, result, _strings = _profile_result(_build(source=source))
    assert result["KernelFunctionName"] == expected
    assert result["KernelDemangledName"] == kernel_name


def test_launch_geometry_is_stored_axis_by_axis():
    source = RepSource("gemm(float*)", (7, 8, 9), (10, 11, 12))
    _header, result, _strings = _profile_result(_build(source=source))
    assert result["GridSize"] == {"X": 7, "Y": 8, "Z": 9}
    assert result["BlockSize"] == {"X": 10, "Y": 11, "Z": 12}
    assert result["GlobalWorkOffset"] == {"X": 0, "Y": 0, "Z": 0}
    assert result["WorkDimensions"] == 3


def test_handles_are_zero_rather_than_invented():
    """We never observed a launch, so no host-side handle is fabricated."""
    _header, result, _strings = _profile_result(_build())
    assert result["ThreadID"] == 0
    assert result["APICallID"] == 0
    assert result["ProgramHandle"] == 0
    assert result["KernelID"] == 0


def test_source_is_present_but_empty_without_source_text():
    _header, result, _strings = _profile_result(_build())
    assert result["Source"] == {}


def test_source_text_lands_in_source_code():
    _header, result, _strings = _profile_result(
        _build(source=RepSource(SOURCE.kernel_name, SOURCE.grid, SOURCE.block,
                                source_text="__global__ void synthetic_gemm(){}")))
    assert result["Source"] == {"Code": "__global__ void synthetic_gemm(){}"}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_metric_names_and_values_round_trip_through_our_own_decoder():
    _header, result, strings = _profile_result(_build())
    read = {}
    for entry in result["MetricResults"]:
        read[strings[entry["NameId"]]] = entry["MetricValue"]["DoubleValue"]
    assert read == {m.name: m.value for m in METRICS}


def test_metric_values_use_double_only():
    """One oneof member per metric; the double shape is the verified one."""
    _header, result, _strings = _profile_result(_build())
    for entry in result["MetricResults"]:
        assert set(entry["MetricValue"]) == {"DoubleValue"}
    assert [f.name for f in PROFILE_METRIC_VALUE.fields] == [
        "StringValue", "FloatValue", "DoubleValue", "Uint32Value", "Uint64Value",
    ]


def test_metric_category_is_left_to_its_proto2_default():
    """Forcing MetricCategorySampler makes the official reader report 0.0."""
    _header, result, _strings = _profile_result(_build())
    assert all("Category" not in entry for entry in result["MetricResults"])
    assert [f.name for f in PROFILE_METRIC_RESULT.fields] == ["NameId", "MetricValue"]


def test_units_ride_along_in_a_section():
    _header, result, _strings = _profile_result(_build())
    (section,) = result["Sections"]
    assert section["Identifier"] == "KernoscopeMetrics"
    assert section["DisplayName"] == "Kernoscope cost model"
    assert section["Header"]["Rows"] == len(METRICS)
    assert [(c["Name"], c["Label"], c["Unit"]) for c in section["Header"]["Metrics"]] == [
        (m.name, m.name, m.unit) for m in METRICS
    ]


def test_empty_unit_is_omitted_not_written_as_empty_string():
    _header, result, _strings = _profile_result(
        _build(metrics=(RepMetric("unitless", 1.0),)))
    cell = result["Sections"][0]["Header"]["Metrics"][0]
    assert "Unit" not in cell


def test_reports_without_metrics_still_carry_the_kernel_identity():
    data = _build(metrics=())
    header, payload = _walk_container(data)[1][1]
    decoded = BLOCK_HEADER.decode(header)
    result = PROFILE_RESULT.decode(payload[4:])
    assert decoded["NumResults"] == 1
    assert "StringTable" not in decoded
    assert "MetricResults" not in result
    assert "Sections" not in result
    assert result["KernelFunctionName"] == "synthetic_gemm"


def test_build_is_deterministic():
    assert _build() == _build()


def test_integer_and_negative_values_are_stored_as_doubles():
    metrics = (RepMetric("count", 3), RepMetric("delta", -0.5))
    _header, result, strings = _profile_result(_build(metrics=metrics))
    read = {strings[e["NameId"]]: e["MetricValue"]["DoubleValue"] for e in result["MetricResults"]}
    assert read == {"count": 3.0, "delta": -0.5}
    assert isinstance(read["count"], float)


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("grid", [(1, 1), (1, 1, 1, 1), (1, 1.0, 1), (1, "1", 1), (1, True, 1), (1, 1, -1)])
def test_bad_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        _build(source=RepSource("gemm(float*)", grid, (1, 1, 1)))


@pytest.mark.parametrize("block", [(1, 1), (1, 1, 1, 1), (1, None, 1), (0, 0, -2)])
def test_bad_block_is_rejected(block):
    with pytest.raises(ValueError):
        _build(source=RepSource("gemm(float*)", (1, 1, 1), block))


def test_non_tuple_dimensions_are_rejected():
    with pytest.raises(ValueError):
        _build(source=RepSource("gemm(float*)", [1, 1, 1], (1, 1, 1)))


def test_empty_kernel_name_is_rejected():
    with pytest.raises(ValueError):
        _build(source=RepSource("   ", (1, 1, 1), (1, 1, 1)))


def test_wrong_input_types_are_rejected():
    with pytest.raises(TypeError):
        _build(source="synthetic_gemm(float*)")
    with pytest.raises(TypeError):
        _build(metrics=[("ks__a", 1.0)])
    with pytest.raises(TypeError):
        _build(source=RepSource("g()", (1, 1, 1), (1, 1, 1), source_text=b"bytes"))
    with pytest.raises(TypeError):
        _build(metrics=(RepMetric("ks__a", 1.0, unit=None),))


@pytest.mark.parametrize(
    "metric",
    [
        RepMetric("", 1.0),
        RepMetric("   ", 1.0),
        RepMetric("ks__a", "1.0"),
        RepMetric("ks__a", None),
        RepMetric("ks__a", True),
    ],
)
def test_bad_metrics_are_rejected(metric):
    with pytest.raises((ValueError, TypeError)):
        _build(metrics=(metric,))


# --------------------------------------------------------------------------- #
# write_report
# --------------------------------------------------------------------------- #


def test_write_report_writes_exactly_the_built_bytes(tmp_path):
    target = tmp_path / "out2.ncu-rep"
    written = write_report(target, SOURCE, METRICS, provider=PROVIDER, comment=COMMENT)
    assert written == len(target.read_bytes()) == target.stat().st_size
    assert target.read_bytes() == _build()


def test_write_report_accepts_str_paths(tmp_path):
    target = tmp_path / "as-string.ncu-rep"
    assert write_report(str(target), SOURCE, METRICS,
                        provider=PROVIDER, comment=COMMENT) == len(target.read_bytes())


# --------------------------------------------------------------------------- #
# declarations: the contract is the field number
# --------------------------------------------------------------------------- #


def test_block_header_field_numbers_match_the_proto():
    assert [(f.name, f.number) for f in BLOCK_HEADER.fields] == [
        ("NumSources", 1), ("NumResults", 2), ("SessionDetails", 3),
        ("StringTable", 4), ("PayloadSize", 5),
    ]


def test_block_header_agrees_with_the_minimal_writer_declaration():
    """Two declarations of one message must not drift on the shared fields."""
    shared = {f.number for f in WIRE_BLOCK_HEADER.fields}
    minimal = {f.number: (f.name, f.kind) for f in WIRE_BLOCK_HEADER.fields}
    for field in BLOCK_HEADER.fields:
        if field.number in shared:
            assert (field.name, field.kind) == minimal[field.number]
    assert shared < {f.number for f in BLOCK_HEADER.fields}


def test_profile_result_field_numbers_match_the_proto():
    assert [(f.name, f.number) for f in PROFILE_RESULT.fields] == [
        ("ThreadID", 1), ("APICallID", 2), ("ProgramHandle", 3), ("KernelID", 4),
        ("KernelMangledName", 5), ("KernelFunctionName", 6),
        ("KernelDemangledName", 7), ("WorkDimensions", 8),
        ("GlobalWorkOffset", 9), ("GridSize", 10), ("BlockSize", 11),
        ("MetricResults", 13), ("Source", 15), ("Sections", 17), ("RuleResults", 19),
    ]


def test_supporting_message_field_numbers_match_the_proto():
    assert [(f.name, f.number, f.kind) for f in UINT64X3.fields] == [
        ("X", 1, "uint64"), ("Y", 2, "uint64"), ("Z", 3, "uint64"),
    ]
    # M6：CudaSmVersion(6) 是 details 页 CC 的来源（真报告里它就在 SourceData 块里）。
    assert [(f.name, f.number) for f in SOURCE_DATA.fields] == [("Code", 2), ("CudaSmVersion", 6)]
    assert [(f.name, f.number) for f in STRING_TABLE.fields] == [("Strings", 1)]
    assert [(f.name, f.number) for f in PROFILE_SECTION.fields] == [
        ("Identifier", 1), ("DisplayName", 2), ("Order", 3), ("Header", 4),
        ("Description", 7), ("Sets", 8),
    ]


def test_session_details_declaration_is_reused_not_redeclared():
    assert SESSION_DETAILS is BLOCK_HEADER.fields[2].message


# --------------------------------------------------------------------------- #
# M6：details 页渲染（真机实验结果固化成测试）
# --------------------------------------------------------------------------- #
def test_ncu_rows_table_is_self_consistent():
    """节表/行表必须自洽：一个名字只属于一个节、每个节引用的名字都有行定义。"""
    from kernoscope.ncrep.ncu_rows import NCU_ROWS, NCU_SECTIONS, consensus_check, row_for

    assert consensus_check() == []
    for row in NCU_ROWS:
        assert row_for(row.name) is row
        assert row.label and row.scale > 0
    assert {s.identifier for s in NCU_SECTIONS} == {"LaunchStats", "SpeedOfLight"}


def test_report_carries_ncu_sections_with_ncu_labels_and_units():
    """details 页的表靠 Header.Metrics 的 **Label** 认行；Label 必须是 ncu 的标签。

    真机二分实验（改真报告的 Label → 行消失；改回 → 行回来）是这条测试的来源。
    """
    from kernoscope.ncrep.profile import PROFILE_RESULT

    data = build_report(
        RepSource("k(float *)", (1024, 1, 1), (256, 1, 1)),
        [RepMetric("gpu__time_duration.sum", 95.1), RepMetric("launch__grid_size", 1024.0),
         RepMetric("launch__registers_per_thread", 96.0), RepMetric("ks__cost_per_mtok", 12.5)],
        provider="kernoscope", comment="not counters", disclosure="substitute calibers",
        device_name="RTX 4090", compute_capability="8.9")
    blocks = _split_blocks(data)
    # 署名块、SourceData 块（CC）、结果块
    assert len(blocks) == 3, [len(b) for b in blocks]
    assert blocks[0][1] == 0 and blocks[1][1] > 0
    session = blocks[0][0]["SessionDetails"]      # decode 已递归解出嵌套 message
    attrs = session["DeviceAttributes"][0]
    assert attrs["ID"] == 0 and attrs["Name"] == "RTX 4090"
    assert attrs["Attributes"][0]["Value"]["Uint64Value"] == (8 << 16) | 9
    result = PROFILE_RESULT.decode(_payload_message(data, num_results=1))
    sections = {s["Identifier"]: s for s in result["Sections"]}
    assert set(sections) == {"SpeedOfLight", "LaunchStats", "KernoscopeMetrics"}
    soff = sections["SpeedOfLight"]["Header"]["Metrics"][0]
    assert (soff["Name"], soff["Label"], soff["Unit"]) == ("gpu__time_duration.sum", "Duration", "ns")
    launch = {m["Name"]: m for m in sections["LaunchStats"]["Header"]["Metrics"]}
    assert launch["launch__grid_size"]["Label"] == "Grid Size"
    assert launch["launch__registers_per_thread"]["Unit"] == "register/thread"
    # 每个节的 Rows 必须等于它声明的行数（不一致时 details 页不渲染）
    for s in result["Sections"]:
        assert s["Header"]["Rows"] == len(s["Header"]["Metrics"])
    # 同源值按 ncu 单位换算：95.1 ms → 95100000 ns
    values = {m["NameId"]: m["MetricValue"]["DoubleValue"] for m in result["MetricResults"]}
    strings = _strings_of(data)
    by_name = {strings[i]: v for i, v in values.items()}
    assert by_name["gpu__time_duration.sum"] == pytest.approx(95.1 * 1e6)
    # 替代口径留在自有命名空间的节里（details 表认不出它，但 raw/Python API 能读）
    own = sections["KernoscopeMetrics"]["Header"]["Metrics"]
    assert [m["Name"] for m in own] == ["ks__cost_per_mtok"]
    # RuleResult 是 details 页唯一会渲染散文的通道
    rule = result["RuleResults"][0]
    assert rule["Identifier"] == "KernoscopeSubstituteCalibers"
    assert rule["Body"]["Items"][0]["Message"]["Type"] == 2      # Warning
    assert "substitute calibers" in rule["Body"]["Items"][0]["Message"]["Message"]


def test_cuda_sm_version_parsing_is_conservative():
    from kernoscope.ncrep.profile import cuda_sm_version

    assert cuda_sm_version("8.9") == (8 << 16) | 9
    assert cuda_sm_version("10.0") == (10 << 16)
    assert cuda_sm_version("") is None and cuda_sm_version(None) is None
    assert cuda_sm_version("sm_89") is None and cuda_sm_version("bogus") is None


def _split_blocks(data: bytes):
    """→ [[session_blockheader_dict, payload_len], ...]（只解析 BlockHeader）。"""
    from kernoscope.ncrep.profile import BLOCK_HEADER

    flen = int.from_bytes(data[4:8], "little")
    off = 8 + flen
    out = []
    while off < len(data):
        blen = int.from_bytes(data[off:off + 4], "little")
        bh = BLOCK_HEADER.decode(data[off + 4:off + 4 + blen])
        payload = data[off + 4 + blen: off + 4 + blen + bh["PayloadSize"]]
        out.append((bh, len(payload), payload))
        off += 4 + blen + bh["PayloadSize"]
    return out


def _strings_of(data: bytes) -> list[str]:
    for bh, _n, _payload in _split_blocks(data):
        table = bh.get("StringTable")
        if table:
            return table.get("Strings", [])
    return []


def _payload_message(data: bytes, *, num_results: int = 1) -> bytes:
    for bh, _n, payload in _split_blocks(data):
        if bh.get("NumResults") == num_results:
            size = int.from_bytes(payload[0:4], "little")
            return payload[4:4 + size]
    raise AssertionError("没找到结果块")
