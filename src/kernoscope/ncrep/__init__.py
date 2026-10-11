"""Minimal, stdlib-only writer for Nsight Compute ``.ncu-rep`` report files.

    from kernoscope.ncrep import RepMetric, RepSource, write_report
    write_report(
        "out.ncu-rep",
        RepSource("synthetic_gemm(float const*, float const*, float*)", (4, 1, 1), (32, 1, 1)),
        [RepMetric("ks__cost_per_mtok", 42.5, "usd/1e6tok")],
        provider="kernoscope",
        comment="synthetic cost model",
    )

The emitted file is accepted by ``ncu_report.load_report()`` from the official
Nsight Compute Python bindings and by ``ncu --import``. Nothing here imports a
third-party package: :mod:`wire` and :mod:`schema` are a hand-rolled protobuf
wire-format codec.

* :mod:`writer` -- the container grammar and how *little* the reader requires
  (:func:`write_minimal_report`).
* :mod:`profile` -- the same container carrying a kernel identity and metrics
  that round-trip through the official reader (:func:`build_report`).
"""

from .profile import (
    NcuRepAttributionError,
    RepMetric,
    RepSource,
    build_report,
    write_report,
)
from .schema import Field, Message
from .wire import WireError
from .writer import (
    BLOCK_HEADER,
    FILE_HEADER,
    FILE_VERSION,
    MAGIC,
    MIN_SUPPORTED_VERSION,
    SESSION_DETAILS,
    block_bytes,
    file_header_bytes,
    report_bytes,
    write_minimal_report,
)

__all__ = [
    "BLOCK_HEADER",
    "FILE_HEADER",
    "FILE_VERSION",
    "MAGIC",
    "MIN_SUPPORTED_VERSION",
    "Message",
    "Field",
    "NcuRepAttributionError",
    "RepMetric",
    "RepSource",
    "SESSION_DETAILS",
    "WireError",
    "block_bytes",
    "build_report",
    "file_header_bytes",
    "report_bytes",
    "write_minimal_report",
    "write_report",
]
