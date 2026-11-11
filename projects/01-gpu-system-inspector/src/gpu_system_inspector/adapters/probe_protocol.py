"""The PyTorch probe protocol (docs/data_flow.md §5) as typed values.

The probe child prints one JSON document; `decode_probe_output` turns the child's command
outcome into a `TorchProbeReport` or a `COLLECTION_FAILED` outcome, and
`decode_probe_payload` validates the same document produced in-process. Normalising the
report into `PyTorchInfo` is the framework collector's job. Protocol 2 (DEC-48) adds
`device_error`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gpu_system_inspector.adapters.outcomes import (
    IMPORT_ERROR,
    PROBE_PROTOCOL_ERROR,
    CommandResult,
    excerpt,
    probe_exit_code,
)
from gpu_system_inspector.models import PROBE_PROTOCOL_VERSION, CollectionStatus, Observed, Source

TORCH_PROBE_MODULE = "gpu_system_inspector.adapters._torch_probe_main"


def torch_probe_argv(python_executable: str) -> tuple[str, ...]:
    return (python_executable, "-m", TORCH_PROBE_MODULE)


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchProbeDevice:
    """`None` means the attribute isn't exposed by this PyTorch version."""

    index: int
    name: str | None
    capability: str | None
    total_memory: int | None
    multiprocessor_count: int | None
    uuid: str | None
    pci_bus_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchAllocatorSample:
    """In-process only (FR-10, DEC-08): the calling process's caching-allocator counters."""

    index: int
    allocated: int
    reserved: int
    peak_allocated: int
    peak_reserved: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchProbeReport:
    torch_version: str
    cuda_build: bool
    build_cuda_version: str | None
    hip_version: str | None
    cudnn_version: str | None
    cuda_available: bool
    cuda_init_warnings: tuple[str, ...]
    device_count: int | None
    devices: tuple[TorchProbeDevice, ...]
    device_error: str | None = None
    # Not part of the JSON protocol: set by the probe that produced the report.
    allocator: tuple[TorchAllocatorSample, ...] | None = None  # in-process only, or None
    duration_ms: int | None = None


_REPORT_FIELDS: dict[str, tuple[type, ...]] = {
    "torch_version": (str,),
    "cuda_build": (bool,),
    "build_cuda_version": (str, type(None)),
    "hip_version": (str, type(None)),
    "cudnn_version": (str, type(None)),
    "cuda_available": (bool,),
    "cuda_init_warnings": (list,),
    "device_count": (int, type(None)),
    "devices": (list,),
    "device_error": (str, type(None)),
}
_DEVICE_FIELDS: dict[str, tuple[type, ...]] = {
    "index": (int,),
    "name": (str, type(None)),
    "capability": (str, type(None)),
    "total_memory": (int, type(None)),
    "multiprocessor_count": (int, type(None)),
    "uuid": (str, type(None)),
    "pci_bus_id": (str, type(None)),
}
_HEADER = {"probe_protocol", "import_ok", "import_error"}


class _ProtocolError(ValueError):
    pass


def _checked(data: Any, spec: dict[str, tuple[type, ...]], where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise _ProtocolError(f"{where} must be a JSON object")
    if set(data) != set(spec):
        missing, extra = sorted(set(spec) - set(data)), sorted(set(data) - set(spec))
        raise _ProtocolError(f"{where} keys differ (missing {missing}, unexpected {extra})")
    for key, types in spec.items():
        value = data[key]
        # bool is an int subclass; JSON booleans must never pass for numbers.
        if not isinstance(value, types) or (isinstance(value, bool) and bool not in types):
            raise _ProtocolError(f"{where}.{key} has the wrong type ({type(value).__name__})")
    return data


def _report(payload: dict[str, Any]) -> TorchProbeReport:
    body = _checked({k: v for k, v in payload.items() if k not in _HEADER}, _REPORT_FIELDS, "probe")
    warnings = body["cuda_init_warnings"]
    if not all(isinstance(w, str) for w in warnings):
        raise _ProtocolError("probe.cuda_init_warnings must contain strings")
    devices = tuple(
        TorchProbeDevice(**_checked(d, _DEVICE_FIELDS, f"probe.devices[{i}]"))
        for i, d in enumerate(body["devices"])
    )
    return TorchProbeReport(**{**body, "cuda_init_warnings": tuple(warnings), "devices": devices})


def decode_probe_output(outcome: Observed[CommandResult]) -> Observed[TorchProbeReport]:
    """Validate the probe child's output against protocol version `PROBE_PROTOCOL_VERSION`.

    Launch failures (timeout, missing interpreter, denied execution) pass through unchanged.
    """
    result = outcome.value
    if result is None:
        return outcome  # type: ignore[return-value]  # a failure carries no value

    def failed(code: str, message: str) -> Observed[TorchProbeReport]:
        return Observed.failed(
            CollectionStatus.COLLECTION_FAILED,
            code=code,
            message=message,
            source=outcome.source,
            method=outcome.method,
        )

    if result.returncode != 0:
        tail = excerpt(result.stderr)
        return failed(
            probe_exit_code(result.returncode),
            f"probe exited with status {result.returncode}" + (f": {tail}" if tail else ""),
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return failed(PROBE_PROTOCOL_ERROR, str(exc))
    return decode_probe_payload(payload, source=outcome.source, method=outcome.method)


def decode_probe_payload(
    payload: Any, *, source: Source | None, method: str | None
) -> Observed[TorchProbeReport]:
    """Validate one probe document (decoded JSON, or the in-process probe's dict)."""

    def failed(code: str, message: str) -> Observed[TorchProbeReport]:
        return Observed.failed(
            CollectionStatus.COLLECTION_FAILED,
            code=code,
            message=message,
            source=source,
            method=method,
        )

    try:
        if not isinstance(payload, dict):
            raise _ProtocolError("probe output must be a JSON object")
        if not _HEADER <= set(payload):
            raise _ProtocolError(f"probe output lacks {sorted(_HEADER - set(payload))}")
        version = payload.get("probe_protocol")
        if type(version) is not int or version != PROBE_PROTOCOL_VERSION:
            raise _ProtocolError(f"unsupported probe protocol {version!r}")
        if not isinstance(payload.get("import_ok"), bool):
            raise _ProtocolError("probe.import_ok must be a boolean")
        if not payload["import_ok"]:
            reason = payload.get("import_error")
            return failed(IMPORT_ERROR, reason if isinstance(reason, str) else "import failed")
        if payload.get("import_error") is not None:
            raise _ProtocolError("probe.import_error must be null when import_ok is true")
        report = _report(payload)
    except _ProtocolError as exc:
        return failed(PROBE_PROTOCOL_ERROR, str(exc))
    return Observed(value=report, status=CollectionStatus.COLLECTED, source=source, method=method)
