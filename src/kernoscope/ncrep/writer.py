"""Write a minimal, valid Nsight Compute ``.ncu-rep`` report.

Container layout (verified byte-by-byte against reports produced by Nsight
Compute 2025.3.1; three shipped sample reports of 0.28-0.57 MB each walk to
exactly EOF with this grammar and no slack):

    "NVR\\0"                          magic, 4 bytes
    uint32 LE  sizeof(FileHeader)     length of the serialized FileHeader
    FileHeader                        protobuf
    then, repeated until EOF, one entry per block:
        uint32 LE  sizeof(BlockHeader)
        BlockHeader                   protobuf
        payload                       exactly BlockHeader.PayloadSize bytes,
                                      itself a sequence of
                                      [uint32 LE size][protobuf message]

There are no checksums. The ``[]`` around ``size`` in the payload is a length
prefix, not an optional field.

What the reader actually requires
---------------------------------
``ncu_report.load_report()`` is far more permissive than its docstring implies:
it raises only :exc:`FileNotFoundError`. A zero-byte file, a file with a bad
magic number, and a truncated report all "load" and yield an *empty*
:class:`IContext` rather than an exception. The red-capable discriminators are:

* ``len(ctx) == 0`` (``ctx.num_ranges() == 0``) for a rejected file versus
  ``>= 1`` for an accepted one -- but the converse is weak: the 10-byte
  header-only file below already reports ``num_ranges() == 1`` with a single
  *empty* range (``num_actions() == 0``). Anything that wants to claim "this
  report carries results" has to look at ``range.num_actions() >= 1``; and
* the ``ncu --import`` CLI, which exits non-zero and prints
  ``==ERROR== The file is not an Nsight Compute profiler report.``

Consequently the minimum accepted report is the 10-byte header-only file
emitted by :func:`report_bytes` when neither ``comment`` nor ``provider`` is
given. Blocks are optional; one block carrying ``SessionDetails`` is added only
when there is metadata to store.

FileHeader.Version is a *minimum*, not an equality check: 7 is the lowest value
this reader accepts and 2**31-1 is accepted too, so 7 is the natural choice.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .schema import Field, Message

#: Container magic.
MAGIC = b"NVR\x00"

#: Lowest ``FileHeader.Version`` the Nsight Compute 2025.3.1 reader accepts.
MIN_SUPPORTED_VERSION = 7

#: Version this writer emits.
FILE_VERSION = MIN_SUPPORTED_VERSION

# --------------------------------------------------------------------------- #
# message declarations
#
# Field numbers and kinds come from the ``.proto`` files NVIDIA ships in the
# Nsight Compute install under ``extras/FileFormat/``. Declarations below cover
# only the fields this writer emits; ``Message.decode`` skips the rest.
# --------------------------------------------------------------------------- #

FILE_HEADER = Message("FileHeader", [
    Field("Version", 1, "uint32"),
])

VERSION_INFO = Message("VersionInfo", [
    Field("Provider", 1, "string"),
    Field("Version", 2, "string"),
])

#: A subset of ``ReportSessionDetails``. Fields omitted here include
#: ``ProcessID`` (1), ``CreationTime`` (2), the system/device info blocks (3-6,
#: 9, 10) and ``ProfilerCmdline`` (11).
SESSION_DETAILS = Message("ReportSessionDetails", [
    Field("Comments", 7, "string"),
    Field("VersionInfo", 8, "message", repeated=True, message=VERSION_INFO),
])

#: A subset of ``BlockHeader``. Fields 7-9 (``NumRangeResults``,
#: ``NumCmdlistResults``, ``NumGraphResults``) are not emitted by this writer.
#: Fields 4 (``StringTable``) and 6 (``Process``) exist in real reports but are
#: not needed for the reader to accept a block. ``PayloadSize`` is the byte
#: length of the payload that follows the block header.
BLOCK_HEADER = Message("BlockHeader", [
    Field("NumSources", 1, "uint32"),
    Field("NumResults", 2, "uint32"),
    Field("SessionDetails", 3, "message", message=SESSION_DETAILS),
    Field("PayloadSize", 5, "uint32"),
])


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #

def file_header_bytes(version: int = FILE_VERSION) -> bytes:
    """Return ``magic + uint32 LE len(FileHeader) + FileHeader`` for ``version``.

    This is the fixed prologue every report starts with; :mod:`profile` reuses it
    rather than re-deriving the framing.
    """
    body = FILE_HEADER.encode({"Version": version})
    return MAGIC + struct.pack("<I", len(body)) + body


def block_bytes(block_header: bytes, payload: bytes = b"") -> bytes:
    """Frame one block: ``uint32 LE len(BlockHeader) + BlockHeader + payload``.

    ``payload`` must be exactly ``BlockHeader.PayloadSize`` bytes long; this
    function does not check that, because the size lives *inside* the serialized
    header it is handed.
    """
    return struct.pack("<I", len(block_header)) + block_header + payload


def report_bytes(
    *,
    comment: str | None = None,
    provider: str | None = None,
    provider_version: str = "unknown",
    version: int = FILE_VERSION,
) -> bytes:
    """Build the report in memory and return it.

    See :func:`write_minimal_report` for the meaning of the arguments. The
    returned bytes are the smallest valid report when no metadata is supplied.
    """
    if version < MIN_SUPPORTED_VERSION:
        raise ValueError(
            f"FileHeader.Version must be >= {MIN_SUPPORTED_VERSION}, got {version}"
        )
    out = bytearray(file_header_bytes(version))
    if comment is None and provider is None:
        return bytes(out)

    details: dict = {}
    if comment is not None:
        details["Comments"] = comment
    if provider is not None:
        # VersionInfo.Provider and .Version are both required by the schema, so
        # provider_version always has to be emitted alongside provider.
        details["VersionInfo"] = [{"Provider": provider, "Version": provider_version}]

    # NumSources/NumResults/PayloadSize are written as explicit zeros: real
    # reports carry them explicitly, and readers that trust NumSources prior to
    # PayloadSize would otherwise see a nonzero default.
    header = BLOCK_HEADER.encode({
        "NumSources": 0,
        "NumResults": 0,
        "SessionDetails": details,
        "PayloadSize": 0,
    })
    out += block_bytes(header)
    return bytes(out)


def write_minimal_report(
    path,
    *,
    comment: str | None = None,
    provider: str | None = None,
) -> int:
    """Write a minimal ``.ncu-rep`` report to ``path`` and return its size in bytes.

    With no metadata the file is the 10-byte header-only form: magic, the
    serialized ``FileHeader{Version: 7}``, and no blocks. Passing ``comment``
    and/or ``provider`` appends a single block whose ``SessionDetails`` carries
    them, which is still under 100 bytes.

    Args:
        path: destination file path.
        comment: free-form text stored in ``SessionDetails.Comments``.
        provider: value stored in ``SessionDetails.VersionInfo[0].Provider``.

    Returns:
        The number of bytes written.
    """
    data = report_bytes(comment=comment, provider=provider)
    Path(path).write_bytes(data)
    return len(data)
