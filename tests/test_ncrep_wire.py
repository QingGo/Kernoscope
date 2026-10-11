"""``kernoscope.ncrep`` 的纯标准库自证：wire 原语、声明式编解码、以及容器框头字节。

这里**不**导入 protobuf、ncu_report 或任何第三方包：本模块的卖点就是手写 wire 格式。
官方读端（``ncu_report.load_report`` / ``ncu --import``）的验收在远端单独跑，
本地只钉住"我们自己写出来的字节就是我们要的那些字节"。
"""

import struct

import pytest

from kernoscope.ncrep import wire
from kernoscope.ncrep.schema import Field, Message
from kernoscope.ncrep.writer import (
    BLOCK_HEADER,
    FILE_HEADER,
    FILE_VERSION,
    MAGIC,
    SESSION_DETAILS,
    report_bytes,
    write_minimal_report,
)

# --------------------------------------------------------------------------- #
# varint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, b"\x00"),
        (1, b"\x01"),
        (2, b"\x02"),
        (127, b"\x7f"),
        (128, b"\x80\x01"),
        (300, b"\xac\x02"),
        (2**32 - 1, b"\xff\xff\xff\xff\x0f"),
        (2**63 - 1, b"\xff\xff\xff\xff\xff\xff\xff\xff\x7f"),
        (2**64 - 1, b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"),
    ],
)
def test_varint_known_encodings(value, expected):
    assert wire.encode_varint(value) == expected
    reader = wire.Reader(expected)
    assert reader.read_varint() == value
    assert reader.eof


def test_varint_roundtrip_edges():
    for value in (0, 1, 2, 127, 128, 129, 16383, 16384, 2**31, 2**63 - 1, 2**64 - 1):
        assert wire.Reader(wire.encode_varint(value)).read_varint() == value


def test_varint_rejects_out_of_range():
    with pytest.raises(wire.WireError):
        wire.encode_varint(-1)
    with pytest.raises(wire.WireError):
        wire.encode_varint(2**64)


def test_varint_rejects_truncated_and_overlong():
    with pytest.raises(wire.WireError):
        wire.Reader(b"\x80").read_varint()  # continuation bit with no next byte
    with pytest.raises(wire.WireError):
        wire.Reader(b"\xff" * 11).read_varint()
    with pytest.raises(wire.WireError):
        # 10 digits but the value overflows 64 bits
        wire.Reader(b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x7f").read_varint()


# --------------------------------------------------------------------------- #
# tags and unknown-field skipping
# --------------------------------------------------------------------------- #


def test_tag_packs_field_and_wire_type():
    assert wire.encode_tag(1, wire.WIRETYPE_VARINT) == b"\x08"
    assert wire.encode_tag(2, wire.WIRETYPE_LENGTH_DELIMITED) == b"\x12"
    assert wire.encode_tag(5, wire.WIRETYPE_FIXED32) == b"\x2d"
    reader = wire.Reader(wire.encode_tag(1000, wire.WIRETYPE_FIXED64))
    assert reader.read_tag() == (1000, wire.WIRETYPE_FIXED64)


def test_tag_rejects_field_zero():
    with pytest.raises(wire.WireError):
        wire.encode_tag(0, wire.WIRETYPE_VARINT)
    with pytest.raises(wire.WireError):
        wire.Reader(b"\x00").read_tag()


def test_skip_every_supported_wire_type():
    reader = wire.Reader(
        b"\x08\x2a"                      # varint
        + b"\x11" + b"\x01" * 8          # fixed64
        + b"\x1a\x03abc"                 # length-delimited
        + b"\x25" + b"\x02" * 4          # fixed32
    )
    for expected_type in (
        wire.WIRETYPE_VARINT,
        wire.WIRETYPE_FIXED64,
        wire.WIRETYPE_LENGTH_DELIMITED,
        wire.WIRETYPE_FIXED32,
    ):
        _, wire_type = reader.read_tag()
        assert wire_type == expected_type
        reader.skip(wire_type)
    assert reader.eof


def test_skip_rejects_groups():
    with pytest.raises(wire.WireError):
        wire.Reader(b"").skip(wire.WIRETYPE_START_GROUP)


# --------------------------------------------------------------------------- #
# declarative codec
# --------------------------------------------------------------------------- #

NESTED = Message("Nested", [
    Field("Id", 1, "uint32"),
    Field("Label", 2, "string"),
])

INT64 = Message("I64", [Field("A", 1, "int64")])

EVERYTHING = Message("Everything", [
    Field("U32", 1, "uint32"),
    Field("U64", 2, "uint64"),
    Field("I32", 3, "int32"),
    Field("Flag", 4, "bool"),
    Field("Kind", 5, "enum"),
    Field("Fixed32", 6, "fixed32"),
    Field("Fixed64", 7, "fixed64"),
    Field("Ratio", 8, "float"),
    Field("Exact", 9, "double"),
    Field("Text", 10, "string"),
    Field("Blob", 11, "bytes"),
    Field("Child", 12, "message", message=NESTED),
    Field("Many", 13, "uint32", repeated=True),
    Field("Kids", 14, "message", repeated=True, message=NESTED),
])


def test_roundtrip_all_kinds():
    values = {
        "U32": 4294967295,
        "U64": 2**64 - 1,
        "I32": -1,
        "Flag": True,
        "Kind": 2,
        "Fixed32": 0xDEADBEEF,
        "Fixed64": 0x0123456789ABCDEF,
        "Ratio": 0.5,
        "Exact": -1.25,
        "Text": "héllo",
        "Blob": b"\x00\xff",
        "Child": {"Id": 7, "Label": "n"},
        "Many": [1, 2, 300],
        "Kids": [{"Id": 1}, {"Label": "two"}],
    }
    assert EVERYTHING.decode(EVERYTHING.encode(values)) == values


def test_encode_omits_none_and_empty_repeated():
    assert EVERYTHING.encode({}) == b""
    assert EVERYTHING.encode({"U32": None, "Many": []}) == b""
    assert Message("M", [Field("A", 1, "uint32")]).encode({"A": 0}) == b"\x08\x00"


def test_encode_rejects_unknown_field_and_bad_value():
    with pytest.raises(ValueError):
        NESTED.encode({"Nope": 1})
    with pytest.raises(TypeError):
        NESTED.encode({"Label": 5})
    with pytest.raises(TypeError):
        INT64.encode({"A": "not an int"})


def test_decode_skips_unknown_fields_and_wrong_wire_type():
    # Field 99 is not declared; field 1 arrives as length-delimited instead of
    # varint. Both must be skipped without disturbing the fields that follow.
    payload = (
        wire.encode_tag(99, wire.WIRETYPE_VARINT) + wire.encode_varint(5)
        + wire.encode_tag(99, wire.WIRETYPE_FIXED64) + b"\x00" * 8
        + wire.encode_tag(99, wire.WIRETYPE_LENGTH_DELIMITED) + b"\x02hi"
        + wire.encode_tag(1, wire.WIRETYPE_LENGTH_DELIMITED) + b"\x02hi"
        + NESTED.encode({"Id": 3, "Label": "keep"})
    )
    assert NESTED.decode(payload) == {"Id": 3, "Label": "keep"}


def test_decode_singular_field_last_wins():
    assert NESTED.decode(b"\x08\x01\x08\x02") == {"Id": 2}


def test_decode_repeated_field_accumulates():
    one = Message("R", [Field("A", 1, "uint32", repeated=True)])
    assert one.decode(b"\x08\x01\x08\x02") == {"A": [1, 2]}


def test_declaration_validation():
    with pytest.raises(ValueError):
        Field("Bad", 1, "sint32")
    with pytest.raises(ValueError):
        Field("Bad", 1, "message")
    with pytest.raises(ValueError):
        Field("Bad", 1, "uint32", message=NESTED)
    with pytest.raises(ValueError):
        Field("Bad", 0, "uint32")
    with pytest.raises(ValueError):
        Message("Dup", [Field("A", 1, "uint32"), Field("B", 1, "uint32")])


def test_int_range_checking():
    with pytest.raises(wire.WireError):
        Message("M", [Field("A", 1, "uint32")]).encode({"A": 2**32})
    with pytest.raises(wire.WireError):
        Message("M", [Field("A", 1, "uint32")]).encode({"A": -1})
    # proto2 stores negative int64 as its 64-bit two's complement.
    assert INT64.encode({"A": -(2**31)}) == b"\x08" + wire.encode_varint(-(2**31) & (2**64 - 1))
    assert INT64.decode(INT64.encode({"A": -(2**31)})) == {"A": -(2**31)}


# --------------------------------------------------------------------------- #
# container framing -- exact bytes
# --------------------------------------------------------------------------- #


def test_small_message_framing_bytes():
    """``[uint32 size][message]`` framing, checked byte for byte."""
    body = FILE_HEADER.encode({"Version": 7})
    assert body == b"\x08\x07"
    framed = struct.pack("<I", len(body)) + body
    assert framed == b"\x02\x00\x00\x00\x08\x07"
    assert MAGIC + framed == b"NVR\x00\x02\x00\x00\x00\x08\x07"


def test_report_bytes_minimum_is_ten_bytes():
    data = report_bytes()
    assert data == b"NVR\x00\x02\x00\x00\x00\x08\x07"
    assert len(data) == 10
    assert FILE_VERSION == 7


def test_report_bytes_with_metadata_framing():
    data = report_bytes(comment="hi", provider="kernoscope")
    assert data[:10] == b"NVR\x00\x02\x00\x00\x00\x08\x07"

    offset = 10
    header_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    header = data[offset:offset + header_size]
    offset += header_size
    assert offset == len(data), "PayloadSize is 0, so the block must end the file"

    decoded = BLOCK_HEADER.decode(header)
    assert decoded["NumSources"] == 0
    assert decoded["NumResults"] == 0
    assert decoded["PayloadSize"] == 0
    assert decoded["SessionDetails"] == {
        "Comments": "hi",
        "VersionInfo": [{"Provider": "kernoscope", "Version": "unknown"}],
    }


def test_report_bytes_rejects_version_below_minimum():
    with pytest.raises(ValueError):
        report_bytes(version=6)
    body = FILE_HEADER.encode({"Version": 9})
    assert report_bytes(version=9) == MAGIC + struct.pack("<I", len(body)) + body


def test_write_minimal_report_writes_the_bytes(tmp_path):
    target = tmp_path / "out.ncu-rep"
    written = write_minimal_report(target, comment="c", provider="p")
    assert written == len(target.read_bytes()) == target.stat().st_size
    assert target.read_bytes() == report_bytes(comment="c", provider="p")


def test_session_details_field_numbers_match_the_proto():
    """Field numbers are the contract; a silent renumber breaks real readers."""
    assert [(f.name, f.number, f.kind) for f in FILE_HEADER.fields] == [("Version", 1, "uint32")]
    # M6：多了 DeviceAttributes(6) —— details 页首页的设备行（`Device 0, CC 8.6`）由它渲染，
    # 缺了 ncu 会把设备读成未知（实测 `Device 255, CC 0.0`）。字段号按随包 proto。
    assert [(f.name, f.number) for f in SESSION_DETAILS.fields] == [
        ("DeviceAttributes", 6), ("Comments", 7), ("VersionInfo", 8)]
    assert [(f.name, f.number) for f in BLOCK_HEADER.fields] == [
        ("NumSources", 1), ("NumResults", 2), ("SessionDetails", 3), ("PayloadSize", 5),
    ]
