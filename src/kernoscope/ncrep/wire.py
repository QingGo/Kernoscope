"""Protobuf wire-format primitives, standard library only.

Wire format (proto2/proto3 share it):

    tag = (field_number << 3) | wire_type

    wire_type 0  varint            int32/int64/uint32/uint64/sint*/bool/enum
    wire_type 1  fixed64           64-bit little endian
    wire_type 2  length-delimited  string/bytes/embedded message/packed repeated
    wire_type 3  start group       deprecated (rejected here, see ``Reader.skip``)
    wire_type 4  end group         deprecated
    wire_type 5  fixed32           32-bit little endian

``Reader`` is a cursor that walks a serialized message and can skip fields it does
not care about, which is what makes a hand-written decoder forward-compatible.
"""

from __future__ import annotations

import struct

WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5

#: varints are at most 64 bits, i.e. at most 10 base-128 digits.
VARINT_MAX = (1 << 64) - 1
MAX_VARINT_BYTES = 10


class WireError(ValueError):
    """Raised when a byte string is not valid protobuf wire format."""


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #

def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a base-128 varint."""
    if value < 0 or value > VARINT_MAX:
        raise WireError(f"varint out of range: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encode ``(field_number, wire_type)`` as a varint tag."""
    if field_number <= 0:
        raise WireError(f"field number must be positive, got {field_number}")
    return encode_varint((field_number << 3) | wire_type)


def encode_fixed32(value: int) -> bytes:
    return struct.pack("<I", value)


def encode_fixed64(value: int) -> bytes:
    return struct.pack("<Q", value)


def encode_float(value: float) -> bytes:
    return struct.pack("<f", value)


def encode_double(value: float) -> bytes:
    return struct.pack("<d", value)


def encode_length_delimited(payload: bytes) -> bytes:
    return encode_varint(len(payload)) + payload


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #

class Reader:
    """Cursor over a serialized protobuf message."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    @property
    def eof(self) -> bool:
        return self.pos >= len(self.buf)

    def _take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.buf):
            raise WireError(
                f"truncated message: want {count} byte(s) at {self.pos}, "
                f"only {len(self.buf) - self.pos} left"
            )
        chunk = self.buf[self.pos:end]
        self.pos = end
        return chunk

    def read_varint(self) -> int:
        result = 0
        for shift in range(0, MAX_VARINT_BYTES * 7, 7):
            byte = self._take(1)[0]
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                if result > VARINT_MAX:
                    raise WireError(f"varint overflow: {result}")
                return result
        raise WireError("varint longer than 10 bytes")

    def read_tag(self) -> tuple[int, int]:
        """Return ``(field_number, wire_type)``."""
        tag = self.read_varint()
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 0:
            raise WireError("field number 0 is not valid")
        return field_number, wire_type

    def read_fixed32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def read_fixed64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def read_float(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def read_double(self) -> float:
        return struct.unpack("<d", self._take(8))[0]

    def read_length_delimited(self) -> bytes:
        return self._take(self.read_varint())

    def skip(self, wire_type: int) -> None:
        """Advance past one field of the given wire type.

        Groups (wire types 3/4) are deprecated and no longer emitted by protoc;
        supporting them here would be dead weight, so they are rejected.
        """
        if wire_type == WIRETYPE_VARINT:
            self.read_varint()
        elif wire_type == WIRETYPE_FIXED64:
            self._take(8)
        elif wire_type == WIRETYPE_LENGTH_DELIMITED:
            self.read_length_delimited()
        elif wire_type == WIRETYPE_FIXED32:
            self._take(4)
        else:
            raise WireError(f"cannot skip wire type {wire_type} (groups are not supported)")
