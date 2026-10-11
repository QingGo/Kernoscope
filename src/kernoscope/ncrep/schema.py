"""Declarative protobuf message codec, standard library only.

A message is declared once as a :class:`Message` holding :class:`Field`
declarations; the same declaration drives both :meth:`Message.encode` (dict ->
bytes) and :meth:`Message.decode` (bytes -> dict), so the two can never drift.

Only the pieces this package actually needs are implemented: singular and
repeated fields, embedded messages, and the scalar kinds used by the report
format. Declarations may cover a subset of a real message's fields -- unknown
fields are skipped on decode and simply never emitted, which is exactly the
forward-compatibility rule protobuf itself uses.

Zigzag (`sint32`/`sint64`) is deliberately absent: nothing in the report format
this package writes uses it, and unimplemented kinds raise at declaration time
rather than at encode time.
"""

from __future__ import annotations

from .wire import (
    WIRETYPE_FIXED32,
    WIRETYPE_FIXED64,
    WIRETYPE_LENGTH_DELIMITED,
    WIRETYPE_VARINT,
    VARINT_MAX,
    Reader,
    WireError,
    encode_fixed32,
    encode_fixed64,
    encode_double,
    encode_float,
    encode_length_delimited,
    encode_tag,
    encode_varint,
)

#: kind -> (wire type, python type produced by decode)
_SCALAR_KINDS = {
    "uint32": (WIRETYPE_VARINT, int),
    "uint64": (WIRETYPE_VARINT, int),
    "int32": (WIRETYPE_VARINT, int),
    "int64": (WIRETYPE_VARINT, int),
    "bool": (WIRETYPE_VARINT, bool),
    "enum": (WIRETYPE_VARINT, int),
    "fixed32": (WIRETYPE_FIXED32, int),
    "fixed64": (WIRETYPE_FIXED64, int),
    "float": (WIRETYPE_FIXED32, float),
    "double": (WIRETYPE_FIXED64, float),
    "string": (WIRETYPE_LENGTH_DELIMITED, str),
    "bytes": (WIRETYPE_LENGTH_DELIMITED, bytes),
}
_MAX = {
    "uint32": (1 << 32) - 1,
    "uint64": VARINT_MAX,
    "int32": (1 << 31) - 1,
    "int64": (1 << 63) - 1,
}
_MIN = {kind: -(limit + 1) for kind, limit in _MAX.items()}


def _encode_varint_kind(kind: str, value) -> bytes:
    if kind == "bool":
        return encode_varint(1 if value else 0)
    if not isinstance(value, int):
        raise TypeError(f"{kind} field needs an int, got {type(value).__name__}")
    if kind in _MAX:
        # proto2 encodes negative int32/int64 as their 64-bit two's complement.
        if value < 0:
            if kind in ("int32", "int64") and value >= _MIN[kind]:
                return encode_varint(value & VARINT_MAX)
            raise WireError(f"{kind} out of range: {value}")
        if value > _MAX[kind]:
            raise WireError(f"{kind} out of range: {value}")
    return encode_varint(value)


def _decode_varint_kind(kind: str, raw: int):
    if kind in ("int32", "int64") and raw > _MAX[kind]:
        return raw - (1 << 64)
    return bool(raw) if kind == "bool" else raw


class Field:
    """One field of a message.

    Args:
        name: key used in the dicts passed to :meth:`Message.encode` / returned
            by :meth:`Message.decode`.
        number: protobuf field number.
        kind: one of the scalar kinds, or ``"message"``.
        repeated: whether the field may appear more than once.
        message: the :class:`Message` describing the payload, required when
            ``kind == "message"``.
    """

    __slots__ = ("name", "number", "kind", "repeated", "message", "_wire_type")

    def __init__(self, name, number, kind, repeated=False, message=None):
        if kind == "message":
            if not isinstance(message, Message):
                raise ValueError(f"field {name!r}: kind 'message' needs a Message")
            wire_type = WIRETYPE_LENGTH_DELIMITED
        elif kind in _SCALAR_KINDS:
            wire_type = _SCALAR_KINDS[kind][0]
            if message is not None:
                raise ValueError(f"field {name!r}: 'message' only applies to kind 'message'")
        else:
            raise ValueError(f"field {name!r}: unsupported kind {kind!r}")
        if number <= 0:
            raise ValueError(f"field {name!r}: field number must be positive")
        self.name = name
        self.number = number
        self.kind = kind
        self.repeated = repeated
        self.message = message
        self._wire_type = wire_type

    def _encode_one(self, value) -> bytes:
        if self.kind == "message":
            return encode_length_delimited(self.message.encode(value))
        if self.kind in _MAX or self.kind in ("bool", "enum"):
            return _encode_varint_kind(self.kind, value)
        if self.kind == "fixed32":
            return encode_fixed32(value)
        if self.kind == "fixed64":
            return encode_fixed64(value)
        if self.kind == "float":
            return encode_float(value)
        if self.kind == "double":
            return encode_double(value)
        if self.kind == "string":
            if not isinstance(value, str):
                raise TypeError(f"field {self.name!r} needs a str, got {type(value).__name__}")
            return encode_length_delimited(value.encode("utf-8"))
        if self.kind == "bytes":
            return encode_length_delimited(bytes(value))
        raise AssertionError(f"unhandled kind {self.kind!r}")  # pragma: no cover

    def _decode_one(self, reader: Reader):
        wire_type = self._wire_type
        if self.kind == "message":
            return self.message.decode(reader.read_length_delimited())
        if wire_type == WIRETYPE_VARINT:
            return _decode_varint_kind(self.kind, reader.read_varint())
        if wire_type == WIRETYPE_FIXED32:
            return reader.read_float() if self.kind == "float" else reader.read_fixed32()
        if wire_type == WIRETYPE_FIXED64:
            return reader.read_double() if self.kind == "double" else reader.read_fixed64()
        payload = reader.read_length_delimited()
        return payload.decode("utf-8") if self.kind == "string" else payload


class Message:
    """A declarative message: an ordered list of :class:`Field`."""

    __slots__ = ("name", "fields", "_by_number")

    def __init__(self, name: str, fields):
        self.name = name
        self.fields = tuple(fields)
        self._by_number = {f.number: f for f in self.fields}
        if len(self._by_number) != len(self.fields):
            raise ValueError(f"message {name!r}: duplicate field numbers")

    def encode(self, values: dict) -> bytes:
        """Serialize ``values`` (keyed by field name). Missing/``None`` fields are omitted."""
        unknown = set(values) - {f.name for f in self.fields}
        if unknown:
            raise ValueError(f"message {self.name!r}: unknown field(s) {sorted(unknown)}")
        out = bytearray()
        for field in self.fields:
            value = values.get(field.name)
            if value is None:
                continue
            items = value if field.repeated else (value,)
            for item in items:
                out += encode_tag(field.number, field._wire_type)
                out += field._encode_one(item)
        return bytes(out)

    def decode(self, data: bytes) -> dict:
        """Parse ``data`` into a dict, skipping fields this declaration omits."""
        reader = Reader(data)
        out: dict = {}
        while not reader.eof:
            number, wire_type = reader.read_tag()
            field = self._by_number.get(number)
            if field is None or field._wire_type != wire_type:
                reader.skip(wire_type)
                continue
            value = field._decode_one(reader)
            if field.repeated:
                out.setdefault(field.name, []).append(value)
            else:
                out[field.name] = value
        return out
