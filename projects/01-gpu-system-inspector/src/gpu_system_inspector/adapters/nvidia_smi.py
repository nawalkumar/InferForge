"""Production `NvidiaSmiAdapter` (architecture §4.5.5, IMP-A02), built on a `CommandRunner`.

- `locate()` looks for `nvidia-smi` on `PATH` and, on WSL2, in `/usr/lib/wsl/lib` (EV-09).
  macOS has no NVIDIA driver, so it's `NOT_SUPPORTED` there without any lookup.
- `version()` runs plain `nvidia-smi` and reads the driver and CUDA versions from its header,
  accepting the old (`Driver Version`, `CUDA Version`) and new (`KMD Version`,
  `CUDA UMD Version`) labels (EV-03).
- `query_gpus()` runs the documented scripted query
  (`--query-gpu=… --format=csv,noheader,nounits`, EV-02) and parses it with
  `parse_query_output`: CSV, never table columns. A row with the wrong number of columns, a
  bad index or truncated output makes the whole query `COLLECTION_FAILED`/`PARSE_ERROR` (a
  half-trusted inventory would miscount devices); one unparsable value makes only that field
  `PARSE_ERROR`. `[N/A]`, `[Not Supported]` and `[Insufficient Permissions]` follow data
  model §3.3. A non-zero exit is `COLLECTION_FAILED`/`EXIT_<n>`, except an unknown query field
  (older drivers), which is `NOT_SUPPORTED`/`NVIDIA_SMI_NOT_SUPPORTED`.

Nothing runs until a method is called. The executable is located, and each command run, at
most once per adapter (one adapter set per inspection), so the GPU hardware, driver and
runtime collectors share one `query_gpus()` result and one `version()` header.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable

from gpu_system_inspector.adapters.outcomes import (
    PARSE_ERROR,
    PLATFORM_NOT_SUPPORTED,
    excerpt,
    require_success,
)
from gpu_system_inspector.adapters.protocols import (
    NVIDIA_SMI_EXECUTABLE,
    NVIDIA_SMI_QUERY_FIELDS,
    NVIDIA_SMI_WSL_PATH,
    CommandRunner,
    PlatformAdapter,
    SmiGpuRow,
    SmiVersion,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
)
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    DataOrigin,
    Observed,
    PlatformProfile,
    Source,
)

MIB = 2**20
MIB_NOTE = "Converted from MiB; precision 1 MiB"
HEADER_METHOD = "nvidia-smi header"
QUERY_METHOD = "nvidia-smi --query-gpu --format=csv,noheader,nounits"
NVIDIA_SMI_WSL_DIRECTORY = os.path.dirname(NVIDIA_SMI_WSL_PATH)
NVIDIA_SMI_NOT_SUPPORTED = "NVIDIA_SMI_NOT_SUPPORTED"
NOT_ON_MACOS = "nvidia-smi isn't available on macOS: NVIDIA CUDA GPUs are not supported there."
DEFAULT_TIMEOUT_S = 10.0

# Placeholders nvidia-smi prints instead of a value (data model §3.3).
VALUE_MARKERS: dict[str, tuple[CollectionStatus, str]] = {
    "[N/A]": (CollectionStatus.NOT_SUPPORTED, "NVIDIA_SMI_NA"),
    "N/A": (CollectionStatus.NOT_SUPPORTED, "NVIDIA_SMI_NA"),
    "[Not Supported]": (CollectionStatus.NOT_SUPPORTED, NVIDIA_SMI_NOT_SUPPORTED),
    "[Insufficient Permissions]": (
        CollectionStatus.PERMISSION_DENIED,
        "NVIDIA_SMI_INSUFFICIENT_PERMISSIONS",
    ),
    "[Unknown Error]": (CollectionStatus.COLLECTION_FAILED, "NVIDIA_SMI_UNKNOWN_ERROR"),
    "[GPU is lost]": (CollectionStatus.COLLECTION_FAILED, "NVIDIA_SMI_GPU_IS_LOST"),
}
_INT = "int"
_MIB = "mib"
_FLOAT = "float"
_TEXT = "text"
# Query field → (kind, unit). `index` is the row key.
FIELD_TYPES: dict[str, tuple[str, str | None]] = {
    "uuid": (_TEXT, None),
    "name": (_TEXT, None),
    "pci.bus_id": (_TEXT, None),
    "memory.total": (_MIB, "bytes"),
    "memory.used": (_MIB, "bytes"),
    "memory.free": (_MIB, "bytes"),
    "temperature.gpu": (_INT, "celsius"),
    "utilization.gpu": (_INT, "percent"),
    "utilization.memory": (_INT, "percent"),
    "power.draw": (_FLOAT, "watts"),
    "power.limit": (_FLOAT, "watts"),
    "driver_version": (_TEXT, None),
}
assert tuple(FIELD_TYPES) == NVIDIA_SMI_QUERY_FIELDS[1:]
DRIVER_LABEL = re.compile(r"(?:KMD Version|Driver Version)\s*:\s*([0-9][0-9.]*)")
CUDA_LABEL = re.compile(r"(?:CUDA UMD Version|CUDA Version)\s*:\s*([0-9][0-9.]*)")
UNKNOWN_FIELD = re.compile(r"is not a valid field to query", re.IGNORECASE)


class NvidiaSmiCli:
    def __init__(
        self,
        commands: CommandRunner,
        platform: PlatformAdapter | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._commands = commands
        self._platform = platform
        self._timeout_s = timeout_s
        self._location: Observed[str] | None = None
        self._version: SmiVersion | None = None
        self._rows: Observed[tuple[SmiGpuRow, ...]] | None = None
        self.provenance: DataOrigin = getattr(commands, "provenance", DataOrigin.LIVE)
        self._prefix = FIXTURE_METHOD_PREFIX if self.provenance is DataOrigin.FIXTURE else ""

    def locate(self) -> Observed[str]:
        if self._location is None:
            profile = self._platform.profile if self._platform is not None else None
            if profile is PlatformProfile.MACOS:
                self._location = Observed.failed(
                    CollectionStatus.NOT_SUPPORTED,
                    code=PLATFORM_NOT_SUPPORTED,
                    message=NOT_ON_MACOS,
                    source=Source.NVIDIA_SMI,
                    method=None,
                )
            else:
                extra = (NVIDIA_SMI_WSL_DIRECTORY,) if profile is PlatformProfile.WSL2 else ()
                self._location = self._commands.which(NVIDIA_SMI_EXECUTABLE, extra_paths=extra)
        return self._location

    def version(self) -> SmiVersion:
        if self._version is None:
            self._version = self._read_version()
        return self._version

    def query_gpus(self) -> Observed[tuple[SmiGpuRow, ...]]:
        if self._rows is None:
            self._rows = self._query_gpus()
        return self._rows

    # --- helpers --------------------------------------------------------------------------

    def _read_version(self) -> SmiVersion:
        output = self._run(nvidia_smi_version_argv)
        if not output.is_collected:
            return SmiVersion(driver_version=output, cuda_version=output)
        text = output.value or ""
        return SmiVersion(
            driver_version=self._header_value(DRIVER_LABEL, text, "driver version"),
            cuda_version=self._header_value(CUDA_LABEL, text, "CUDA version"),
        )

    def _query_gpus(self) -> Observed[tuple[SmiGpuRow, ...]]:
        location = self.locate()
        if not location.is_collected:
            return location  # type: ignore[return-value]
        assert location.value is not None
        outcome = self._commands.run(
            nvidia_smi_query_argv(location.value),
            timeout_s=self._timeout_s,
            source=Source.NVIDIA_SMI,
        )
        method = self._prefix + QUERY_METHOD
        if not outcome.is_collected:
            return outcome  # type: ignore[return-value]
        result = outcome.value
        assert result is not None
        if result.returncode != 0:
            text = f"{result.stderr}\n{result.stdout}"
            if UNKNOWN_FIELD.search(text):
                return Observed.failed(
                    CollectionStatus.NOT_SUPPORTED,
                    code=NVIDIA_SMI_NOT_SUPPORTED,
                    message=excerpt(text),
                    source=Source.NVIDIA_SMI,
                    method=method,
                )
            return require_success(outcome)  # type: ignore[return-value]
        if result.truncated:
            return _parse_failure("nvidia-smi output was truncated", method)
        return parse_query_output(result.stdout, prefix=self._prefix)

    def _run(self, argv: Callable[[str], tuple[str, ...]]) -> Observed[str]:
        location = self.locate()
        if not location.is_collected:
            return location
        assert location.value is not None
        outcome = require_success(
            self._commands.run(
                argv(location.value), timeout_s=self._timeout_s, source=Source.NVIDIA_SMI
            )
        )
        if not outcome.is_collected:
            return outcome  # type: ignore[return-value]
        assert outcome.value is not None
        return Observed.collected(
            outcome.value.stdout, source=Source.NVIDIA_SMI, method=self._prefix + HEADER_METHOD
        )

    def _header_value(self, label: re.Pattern[str], text: str, what: str) -> Observed[str]:
        method = self._prefix + HEADER_METHOD
        match = label.search(text)
        if match is None:
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code="FIELD_NOT_FOUND",
                message=f"no {what} in the nvidia-smi header",
                source=Source.NVIDIA_SMI,
                method=method,
            )
        return Observed.collected(
            match.group(1).rstrip("."), source=Source.NVIDIA_SMI, method=method
        )


# --- the query parser ---------------------------------------------------------------------


def parse_query_output(stdout: str, *, prefix: str = "") -> Observed[tuple[SmiGpuRow, ...]]:
    """`--query-gpu=<NVIDIA_SMI_QUERY_FIELDS> --format=csv,noheader,nounits` output → rows.

    Empty output is zero GPUs. Rows come back sorted by index."""
    method = prefix + QUERY_METHOD
    lines = [line for line in stdout.splitlines() if line.strip()]
    rows: dict[int, SmiGpuRow] = {}
    for number, cells in enumerate(csv.reader(lines, skipinitialspace=True), start=1):
        cells = [cell.strip() for cell in cells]
        if len(cells) != len(NVIDIA_SMI_QUERY_FIELDS):
            return _parse_failure(
                f"row {number} has {len(cells)} columns, expected {len(NVIDIA_SMI_QUERY_FIELDS)}",
                method,
            )
        index_text, values = cells[0], cells[1:]
        if not index_text.isdigit():
            return _parse_failure(f"row {number} has no valid index: {index_text[:32]!r}", method)
        index = int(index_text)
        if index in rows:
            return _parse_failure(f"index {index} appears twice", method)
        rows[index] = SmiGpuRow(
            index=index,
            values={
                field: _value(field, text, prefix)
                for field, text in zip(FIELD_TYPES, values, strict=True)
            },
        )
    return Observed.collected(
        tuple(rows[i] for i in sorted(rows)), source=Source.NVIDIA_SMI, method=method
    )


def _value(field: str, text: str, prefix: str) -> Observed:
    kind, unit = FIELD_TYPES[field]
    method = f"{prefix}nvidia-smi --query-gpu={field}"
    marker = VALUE_MARKERS.get(text)
    if marker is not None:
        status, code = marker
        return Observed.failed(
            status, code=code, message=text, source=Source.NVIDIA_SMI, method=method, unit=unit
        )
    if not text:
        return Observed.failed(
            CollectionStatus.NOT_DETECTED,
            code="FIELD_NOT_FOUND",
            message=f"nvidia-smi reported no {field}",
            source=Source.NVIDIA_SMI,
            method=method,
            unit=unit,
        )
    try:
        value: object = _convert(kind, text)
    except ValueError:
        return Observed.failed(
            CollectionStatus.COLLECTION_FAILED,
            code=PARSE_ERROR,
            message=f"{field} is not a valid {kind} value: {text[:32]!r}",
            source=Source.NVIDIA_SMI,
            method=method,
            unit=unit,
        )
    note = MIB_NOTE if kind == _MIB else None
    return Observed.collected(value, source=Source.NVIDIA_SMI, method=method, unit=unit, note=note)


def _convert(kind: str, text: str) -> object:
    if kind == _TEXT:
        return text
    if kind in (_INT, _MIB):
        if not text.isdigit():
            raise ValueError(text)
        number = int(text)
        return number * MIB if kind == _MIB else number
    number = float(text)
    if not number >= 0 or number == float("inf"):
        raise ValueError(text)
    return number


def _parse_failure(message: str, method: str) -> Observed:
    return Observed.failed(
        CollectionStatus.COLLECTION_FAILED,
        code=PARSE_ERROR,
        message=message,
        source=Source.NVIDIA_SMI,
        method=method,
    )
