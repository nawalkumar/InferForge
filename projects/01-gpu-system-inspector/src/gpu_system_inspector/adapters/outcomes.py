"""Normalised adapter outcomes (docs/architecture.md §4.5.2, data model §3.3).

Adapters report every *environmental* condition as an `Observed` value: success, an
unsupported capability, a missing dependency, denied permission, a timeout, a failure or
malformed output. Only a broken caller contract raises (`AdapterUsageError`).
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_system_inspector.models import CollectionStatus, Observed, Source

TIMEOUT = "TIMEOUT"
PARSE_ERROR = "PARSE_ERROR"
FIELD_NOT_FOUND = "FIELD_NOT_FOUND"
NOT_SET = "NOT_SET"
NO_DRIVER_BOUND = "NO_DRIVER_BOUND"
# The host is UNKNOWN Linux (data model §4.1), so native-Linux-only checks can't safely run.
LINUX_ENVIRONMENT_UNKNOWN = "LINUX_ENVIRONMENT_UNKNOWN"
PLATFORM_NOT_SUPPORTED = "PLATFORM_NOT_SUPPORTED"
VALUE_NOT_AVAILABLE = "VALUE_NOT_AVAILABLE"
IMPORT_ERROR = "IMPORT_ERROR"
PROBE_PROTOCOL_ERROR = "PROBE_PROTOCOL_ERROR"
UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"
# The mechanism or collector isn't part of this version yet (DEC-44); always NOT_SUPPORTED.
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

ERROR_EXCERPT_CHARS = 500
MAX_COMMAND_OUTPUT_CHARS = 2**20

NVML_ERROR_PREFIX = "NVML_ERROR_"
NVML_ERROR_STATUS: dict[str, CollectionStatus] = {
    "NVML_ERROR_LIBRARY_NOT_FOUND": CollectionStatus.NOT_DETECTED,
    "NVML_ERROR_DRIVER_NOT_LOADED": CollectionStatus.NOT_DETECTED,
    "NVML_ERROR_NO_PERMISSION": CollectionStatus.PERMISSION_DENIED,
    "NVML_ERROR_NOT_SUPPORTED": CollectionStatus.NOT_SUPPORTED,
    "NVML_ERROR_FUNCTION_NOT_FOUND": CollectionStatus.NOT_SUPPORTED,
}


class AdapterUsageError(RuntimeError):
    """The caller broke an adapter contract. Always a bug, never an environment condition."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandResult:
    """A process that ran to completion, whatever its exit status."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool = False


def exit_code(returncode: int) -> str:
    return f"EXIT_{returncode}"


def probe_exit_code(returncode: int) -> str:
    return f"PROBE_EXIT_{returncode}"


def excerpt(text: str, limit: int = ERROR_EXCERPT_CHARS) -> str:
    """The last `limit` characters of `text`, stripped; error messages quote only a tail."""
    return text.strip()[-limit:]


def require_success(outcome: Observed[CommandResult]) -> Observed[CommandResult]:
    """Turn a non-zero exit into `COLLECTION_FAILED`/`EXIT_<n>` with a stderr excerpt."""
    result = outcome.value
    if result is None or result.returncode == 0:
        return outcome
    return Observed.failed(
        CollectionStatus.COLLECTION_FAILED,
        code=exit_code(result.returncode),
        message=excerpt(result.stderr) or f"exited with status {result.returncode}",
        source=outcome.source,
        method=outcome.method,
    )


def nvml_error_status(code: str) -> CollectionStatus:
    """Status for an NVML error name; unlisted NVML errors are collection failures."""
    if not code.startswith(NVML_ERROR_PREFIX):
        raise AdapterUsageError(f"{code!r} is not an NVML error name")
    return NVML_ERROR_STATUS.get(code, CollectionStatus.COLLECTION_FAILED)


def exception_outcome(
    exc: Exception, *, source: Source | None, method: str | None, unit: str | None = None
) -> Observed:
    """Map an exception raised while collecting to an `Observed` failure (data model §3.3)."""
    # Imported here: importlib.metadata pulls in email/socket, which plain imports must not.
    from importlib.metadata import PackageNotFoundError

    name = type(exc).__name__
    if isinstance(exc, PackageNotFoundError):
        status, code = CollectionStatus.NOT_INSTALLED, "PackageNotFoundError"
    elif isinstance(exc, ImportError):
        status, code = CollectionStatus.NOT_INSTALLED, "ImportError"
    elif isinstance(exc, FileNotFoundError):
        status, code = CollectionStatus.NOT_DETECTED, "FileNotFoundError"
    elif isinstance(exc, PermissionError):
        status, code = CollectionStatus.PERMISSION_DENIED, "PermissionError"
    elif isinstance(exc, TimeoutError):
        status, code = CollectionStatus.COLLECTION_FAILED, TIMEOUT
    elif isinstance(exc, OSError):
        status, code = CollectionStatus.COLLECTION_FAILED, name
    else:
        status, code = CollectionStatus.UNKNOWN, name
    return Observed.failed(
        status, code=code, message=str(exc) or name, source=source, method=method, unit=unit
    )
