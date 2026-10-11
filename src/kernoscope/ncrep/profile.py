"""Emit a **non-empty**, reader-verifiable Nsight Compute ``.ncu-rep`` report.

:mod:`writer` answers "what is the smallest file the reader still accepts?".
This module answers the next question: "what is the smallest file whose kernel
identity *and* metric name/value pairs the official reader can read back out?"

Layout produced by :func:`build_report` (byte-for-byte grammar in :mod:`writer`)::

    "NVR\\0" + uint32 LE len(FileHeader) + FileHeader{Version: 7}
    block 0:  BlockHeader{NumSources:0, NumResults:0, SessionDetails}, PayloadSize 0
    block 1:  BlockHeader{NumResults:1, StringTable, PayloadSize}, payload =
              [uint32 LE len(ProfileResult)][ProfileResult]

Block 0 carries the attribution (``comment`` + ``provider``). Block 1 carries one
``ProfileResult`` plus the master ``ProfilerStringTable`` that resolves each
``ProfileMetricResult.NameId`` back to a metric name.
``BlockHeader.PayloadSize`` counts the leading ``uint32`` length prefix, so it
equals ``4 + len(ProfileResult)`` -- M1's rule, re-confirmed here on our own
bytes.

What the official reader actually needs (Nsight Compute 2025.3.1, measured)
--------------------------------------------------------------------------
``ProfileResult`` is proto2 with **twelve required fields**, and the reader runs
with ordinary proto2 required-field semantics: dropping any one of them makes
the whole block unparseable, and the context silently falls back to
``num_ranges() == 0`` (an *empty* context, not an exception).  The required set,
all of which this module fills, is::

    ThreadID, APICallID, ProgramHandle, KernelID, KernelMangledName,
    KernelFunctionName, KernelDemangledName, WorkDimensions, GlobalWorkOffset,
    GridSize, BlockSize, Source

* ``ThreadID``/``APICallID``/``ProgramHandle``/``KernelID`` are written as ``0``:
  we never observed a launch, so inventing a nonzero handle would be a lie.
* ``KernelMangledName`` is written as the **empty string**. The field is required
  by the schema but we cannot mangle C++ here, so it stays empty rather than
  duplicating the demangled name; ``action.name(IAction.NameBase_MANGLED)``
  consequently returns ``""``.
* ``KernelFunctionName`` is derived from ``RepSource.kernel_name`` by cutting at
  the first ``(``; ``KernelDemangledName`` is the string we were handed. The
  reader resolves ``action.name()`` (whose default base is *function*) to
  ``KernelFunctionName``, so ``"synthetic_gemm(float const*, ...)"`` reads back
  as ``"synthetic_gemm"``.
* ``Source`` is emitted as a present-but-empty ``SourceData`` when
  ``RepSource.source_text`` is ``None``; when it is given the text lands in
  ``SourceData.Code``.
* Every metric value is written as ``ProfileMetricValue.DoubleValue`` (field 3).
  Measured ``IMetric.kind()`` per shape: double -> 4, float -> 3, uint64 -> 6.
  ``ProfileMetricResult.Category`` is deliberately **not** emitted: its proto2
  default is ``MetricCategoryPm`` (0) and forcing ``MetricCategorySampler`` (1)
  made the reader report every value as ``0.0``.
* ``RepMetric.unit`` has no slot in ``ProfileMetricResult`` at all, so it is
  stored in the same ``ProfileResult`` under ``Sections``, in
  ``ProfilerSection.Header.Metrics[].Unit`` (field 5). Verified caveat: this
  build's ``IMetric.unit()`` returns ``""`` anyway -- the unit is in the file,
  but the official reader does not surface it for metrics it does not know.
  ``unit`` is therefore *recorded*, not *round-tripped*.

Everything here is standard library only: :mod:`wire` and :mod:`schema` are the
hand-rolled protobuf codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schema import Field, Message
from .writer import SESSION_DETAILS, block_bytes, file_header_bytes

# --------------------------------------------------------------------------- #
# errors and input contract
# --------------------------------------------------------------------------- #


class NcuRepAttributionError(ValueError):
    """Raised when a report would be written without saying who produced it.

    A ``.ncu-rep`` whose numbers cannot be traced back to a producer is worse
    than no report at all: the official reader presents ``ProfileMetricResult``
    values exactly as if Nsight Compute had measured them. :func:`build_report`
    therefore refuses an empty ``provider`` or ``comment``.
    """


@dataclass(frozen=True)
class RepSource:
    """The kernel launch a report describes.

    Args:
        kernel_name: kernel name, ideally demangled with its signature, e.g.
            ``"synthetic_gemm(float const*, float const*, float*)"``. Stored as
            ``ProfileResult.KernelDemangledName``; the part before the first
            ``(`` becomes ``KernelFunctionName``.
        grid: CUDA grid dimensions, ``(x, y, z)``.
        block: CUDA block dimensions, ``(x, y, z)``.
        source_text: optional source, stored in ``SourceData.Code``. The reader
            does not require it.
    """

    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    source_text: str | None = None


@dataclass(frozen=True)
class RepMetric:
    """One metric to store in the report.

    Args:
        name: metric name as the reader should see it, e.g.
            ``"ks__cost_per_mtok"``. Round-trips exactly.
        value: numeric value; written as a double.
        unit: base unit, e.g. ``"usd/1e6tok"``. Recorded in the report's
            ``ProfilerSection``; see the module docstring for why the official
            reader does not return it from ``IMetric.unit()``.
    """

    name: str
    value: float
    unit: str = ""


def _require_attribution(value, what: str) -> str:
    if value is None:
        # The common real-world shape of "no attribution": an optional config
        # value that ended up unset. Report it as the attribution failure it is.
        raise NcuRepAttributionError(
            f"{what} is required: writing a .ncu-rep without telling the reader "
            f"who produced it is not allowed"
        )
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise NcuRepAttributionError(
            f"{what} must be a non-empty string: writing a .ncu-rep without "
            f"telling the reader who produced it is not allowed"
        )
    return value


# --------------------------------------------------------------------------- #
# message declarations
#
# Field numbers and kinds come from the ``.proto`` files NVIDIA ships under
# ``extras/FileFormat/``. Only the fields this writer emits are declared;
# ``Message.decode`` skips the rest.
# --------------------------------------------------------------------------- #

#: ``ProfilerReportCommon.proto`` -- work offset / grid / block sizes.
UINT64X3 = Message("Uint64x3", [
    Field("X", 1, "uint64"),
    Field("Y", 2, "uint64"),
    Field("Z", 3, "uint64"),
])

#: ``ProfilerResultsCommon.proto`` -- the embedded cubin/source carrier.
#: ``Binary`` (4) and ``Intermediate`` (3) are deliberately absent: this module
#: never has a cubin to embed, and an absent field is honest where a fabricated
#: one would not be.
SOURCE_DATA = Message("SourceData", [
    Field("Code", 2, "string"),
])

#: ``ProfilerResultsCommon.proto`` -- the ``MetricValue`` oneof. Exactly one
#: member is written per metric; ``StringValue``/``Uint32Value`` are declared so
#: a decoded file is not silently misread, but this writer always uses
#: ``DoubleValue``.
PROFILE_METRIC_VALUE = Message("ProfileMetricValue", [
    Field("StringValue", 1, "string"),
    Field("FloatValue", 2, "float"),
    Field("DoubleValue", 3, "double"),
    Field("Uint32Value", 4, "uint32"),
    Field("Uint64Value", 5, "uint64"),
])

#: ``ProfilerResultsCommon.proto``. ``Category`` (4) is left to its proto2
#: default ``MetricCategoryPm``; see the module docstring.
PROFILE_METRIC_RESULT = Message("ProfileMetricResult", [
    Field("NameId", 1, "uint32"),
    Field("MetricValue", 2, "message", message=PROFILE_METRIC_VALUE),
])

#: ``ProfilerStringTable.proto`` -- ``NameId`` is an index into ``Strings``.
STRING_TABLE = Message("ProfilerStringTable", [
    Field("Strings", 1, "string", repeated=True),
])

#: ``ProfilerSection.proto`` -- only enough to carry per-metric units.
PROFILE_SECTION_METRIC = Message("ProfilerSectionMetric", [
    Field("Name", 1, "string"),
    Field("Label", 2, "string"),
    Field("Unit", 5, "string"),
])

PROFILE_SECTION_HEADER = Message("ProfilerSectionHeader", [
    Field("Rows", 1, "uint32"),
    Field("Metrics", 2, "message", repeated=True, message=PROFILE_SECTION_METRIC),
])

PROFILE_SECTION = Message("ProfilerSection", [
    Field("Identifier", 1, "string"),
    Field("DisplayName", 2, "string"),
    Field("Order", 3, "uint32"),
    Field("Header", 4, "message", message=PROFILE_SECTION_HEADER),
])

#: ``ProfilerReport.proto`` -- one kernel launch. All twelve required fields are
#: declared *and* always emitted by :func:`build_report`.
PROFILE_RESULT = Message("ProfileResult", [
    Field("ThreadID", 1, "uint32"),
    Field("APICallID", 2, "uint64"),
    Field("ProgramHandle", 3, "uint64"),
    Field("KernelID", 4, "uint64"),
    Field("KernelMangledName", 5, "string"),
    Field("KernelFunctionName", 6, "string"),
    Field("KernelDemangledName", 7, "string"),
    Field("WorkDimensions", 8, "uint32"),
    Field("GlobalWorkOffset", 9, "message", message=UINT64X3),
    Field("GridSize", 10, "message", message=UINT64X3),
    Field("BlockSize", 11, "message", message=UINT64X3),
    Field("MetricResults", 13, "message", repeated=True, message=PROFILE_METRIC_RESULT),
    Field("Source", 15, "message", message=SOURCE_DATA),
    Field("Sections", 17, "message", repeated=True, message=PROFILE_SECTION),
])

#: ``ProfilerReport.proto`` ``BlockHeader``, plus the string table (4) that
#: :mod:`writer`'s minimal declaration omits. The shared fields (1, 2, 3, 5)
#: are pinned to agree with ``writer.BLOCK_HEADER`` by a test.
BLOCK_HEADER = Message("BlockHeader", [
    Field("NumSources", 1, "uint32"),
    Field("NumResults", 2, "uint32"),
    Field("SessionDetails", 3, "message", message=SESSION_DETAILS),
    Field("StringTable", 4, "message", message=STRING_TABLE),
    Field("PayloadSize", 5, "uint32"),
])

#: ``SessionDetails.VersionInfo[0].Version``; the reader requires the sibling
#: ``Provider`` and ``Version`` to be present together.
PROVIDER_VERSION = "unknown"

#: Identifier/display name of the section that carries per-metric units.
SECTION_ID = "KernoscopeMetrics"
SECTION_DISPLAY_NAME = "Kernoscope cost model"


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #

def _dims(value, what: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{what} must be a 3-tuple, got {value!r}")
    for axis in value:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
            raise ValueError(f"{what} must hold non-negative ints, got {value!r}")
    return value


def _function_name(kernel_name: str) -> str:
    """``"addk(float *, int)"`` -> ``"addk"``; a bare name is returned as-is."""
    head, sep, _ = kernel_name.partition("(")
    return head.strip() if sep else kernel_name.strip()


def _string_table(metrics: Sequence[RepMetric]) -> tuple[list[str], list[int]]:
    """Return ``(strings, name_ids)``: the table and each metric's index into it.

    Names are deduplicated first-seen-first so ``NameId`` is stable and the table
    stays tight. If a caller passes the same name twice the reader keeps the last
    value, which mirrors "later wins" for a repeated map entry.
    """
    strings: list[str] = []
    index: dict[str, int] = {}
    ids: list[int] = []
    for metric in metrics:
        if metric.name not in index:
            index[metric.name] = len(strings)
            strings.append(metric.name)
        ids.append(index[metric.name])
    return strings, ids


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #

def _session_block(comment: str, provider: str) -> bytes:
    header = BLOCK_HEADER.encode({
        "NumSources": 0,
        "NumResults": 0,
        "SessionDetails": {
            "Comments": comment,
            "VersionInfo": [{"Provider": provider, "Version": PROVIDER_VERSION}],
        },
        "PayloadSize": 0,
    })
    return block_bytes(header)


def _profile_result(source: RepSource, metrics: Sequence[RepMetric],
                    name_ids: Sequence[int]) -> bytes:
    cells = [
        {
            "Name": metric.name,
            "Label": metric.name,
            "Unit": metric.unit or None,
        }
        for metric in metrics
    ]
    values = {
        "ThreadID": 0,
        "APICallID": 0,
        "ProgramHandle": 0,
        "KernelID": 0,
        "KernelMangledName": "",
        "KernelFunctionName": _function_name(source.kernel_name),
        "KernelDemangledName": source.kernel_name,
        "WorkDimensions": 3,
        "GlobalWorkOffset": {"X": 0, "Y": 0, "Z": 0},
        "GridSize": dict(zip("XYZ", source.grid)),
        "BlockSize": dict(zip("XYZ", source.block)),
        "MetricResults": [
            {"NameId": name_id, "MetricValue": {"DoubleValue": float(metric.value)}}
            for name_id, metric in zip(name_ids, metrics)
        ],
        "Source": (
            {"Code": source.source_text} if source.source_text is not None else {}
        ),
        # Per-metric units have no slot in ProfileMetricResult; a section is the
        # only schema-sanctioned carrier. Emitted only when there are metrics.
        "Sections": [{
            "Identifier": SECTION_ID,
            "DisplayName": SECTION_DISPLAY_NAME,
            "Order": 0,
            "Header": {"Rows": len(cells), "Metrics": cells},
        }] if cells else None,
    }
    return PROFILE_RESULT.encode(values)


def build_report(
    source: RepSource,
    metrics: Sequence[RepMetric],
    *,
    provider: str,
    comment: str,
) -> bytes:
    """Build a non-empty, reader-verifiable report and return it as bytes.

    The result always contains a session block with the attribution and one
    profile result carrying ``source``'s kernel identity plus every metric in
    ``metrics``. Reading it back with the official bindings yields
    ``num_ranges() == 1``, one range with ``num_actions() == 1``, an action whose
    ``name()`` is the function part of ``source.kernel_name``, and
    ``action.metric_by_name(m.name).value() == m.value`` for every metric.

    Args:
        source: kernel identity and launch geometry.
        metrics: metrics to store; may be empty, in which case the report still
            carries the kernel identity.
        provider: who produced the numbers. Required; must be non-empty.
        comment: free-form note shown to whoever opens the report. Required;
            must be non-empty.

    Raises:
        NcuRepAttributionError: if ``provider`` or ``comment`` is missing
            (``None``), empty, or whitespace-only.
        TypeError: if ``provider``/``comment`` is not a ``str``.
        ValueError: if ``source``'s geometry or a metric is malformed.
    """
    provider = _require_attribution(provider, "provider")
    comment = _require_attribution(comment, "comment")
    if not isinstance(source, RepSource):
        raise TypeError(f"source must be a RepSource, got {type(source).__name__}")
    if not isinstance(source.kernel_name, str) or not source.kernel_name.strip():
        raise ValueError("source.kernel_name must be a non-empty string")
    grid = _dims(source.grid, "source.grid")
    block = _dims(source.block, "source.block")
    if source.source_text is not None and not isinstance(source.source_text, str):
        raise TypeError("source.source_text must be a str or None")

    metrics = tuple(metrics)
    for metric in metrics:
        if not isinstance(metric, RepMetric):
            raise TypeError(f"metrics must hold RepMetric, got {type(metric).__name__}")
        if not isinstance(metric.name, str) or not metric.name.strip():
            raise ValueError("RepMetric.name must be a non-empty string")
        if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            raise ValueError(f"RepMetric {metric.name!r}: value must be a number")
        if not isinstance(metric.unit, str):
            raise TypeError(f"RepMetric {metric.name!r}: unit must be a str")

    # Rebuild from the validated fields so what gets encoded is exactly what was
    # checked (and so a caller's subclass/alias cannot smuggle anything else in).
    source = RepSource(source.kernel_name, grid, block, source.source_text)
    strings, name_ids = _string_table(metrics)
    result = _profile_result(source, metrics, name_ids)
    payload = len(result).to_bytes(4, "little") + result

    header = BLOCK_HEADER.encode({
        "NumSources": 0,
        "NumResults": 1,
        "StringTable": {"Strings": strings} if strings else None,
        "PayloadSize": len(payload),
    })
    return (
        file_header_bytes()
        + _session_block(comment, provider)
        + block_bytes(header, payload)
    )


def write_report(
    path: str,
    source: RepSource,
    metrics: Sequence[RepMetric],
    *,
    provider: str,
    comment: str,
) -> int:
    """Write :func:`build_report`'s bytes to ``path``; return the byte count."""
    data = build_report(source, metrics, provider=provider, comment=comment)
    Path(path).write_bytes(data)
    return len(data)
