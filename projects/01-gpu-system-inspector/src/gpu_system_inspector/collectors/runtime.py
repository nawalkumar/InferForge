"""Runtime metrics collector (architecture §4.4, DEC-46): one `RuntimeMetrics` per device.

It samples the devices the GPU hardware collector inventoried, from the same source:

- **NVML inventory.** The core's shared session (never opened here): memory, utilization,
  temperature and slowdown threshold, power draw and limit, and processes (`SKIPPED`, with
  no process calls, when `include_processes` is false).
- **`nvidia-smi` inventory.** The adapter's cached `--query-gpu` rows (no second run). The
  fields that query doesn't carry are `NOT_SUPPORTED`/`FALLBACK_FIELD_UNAVAILABLE`.
- **No inventory.** No records.

**A lost device.** A device that can't be reached again (handle failure) or reports
`GPU is lost` on any read gets an unavailable record and one error at `/runtime/<i>`, and
no more calls (KI-34). Values read before the loss aren't mixed with a lost device. Other
read failures stay field-level: the run is `PARTIAL` for a failure status.

**Capability records** (NVML devices; this run owns `RUNTIME.*`). Per device:
`RUNTIME.GPU_UTILIZATION` when utilization is `NOT_SUPPORTED`; `RUNTIME.PROCESSES` when
process enumeration is `NOT_SUPPORTED` or `PERMISSION_DENIED` (with that status);
`RUNTIME.PROCESS_MEMORY` when a listed process's memory is `NOT_SUPPORTED`. The reason
names the documented cause where it's known, and the fields get it as their note: WSL2
(EV-09), MIG enabled, WDDM on Windows (EV-08). Other unsupported metrics (temperature,
power) are field-level only. The `nvidia-smi` fallback's missing fields aren't records.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import ClassVar

from gpu_system_inspector.adapters import (
    Clock,
    NvidiaSmiAdapter,
    NvmlDevice,
    NvmlSession,
    SmiGpuRow,
)
from gpu_system_inspector.collectors.gpu_hardware import (
    FALLBACK_FIELD_MESSAGE,
    FALLBACK_FIELD_UNAVAILABLE,
)
from gpu_system_inspector.collectors.system import FAILURE_STATUSES
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    GPUDeviceInfo,
    GPUInventory,
    GpuProcess,
    Observed,
    PlatformProfile,
    RuntimeMetrics,
    Source,
    UnsupportedCapability,
)

RUNTIME_REPORT_PATH = "/runtime"
LOST_CODES = frozenset({"NVML_ERROR_GPU_IS_LOST", "NVIDIA_SMI_GPU_IS_LOST"})
PROCESSES_SKIPPED_NOTE = "Process enumeration was skipped (include_processes is false)."
WSL2_RUNTIME_NOTE = "Documented as not yet supported on WSL2 (CUDA on WSL User Guide)"
WDDM_PROCESS_MEMORY_NOTE = (
    "Under WDDM, per-process GPU memory is not available (NVML documentation)"
)
MIG_ENABLED = "ENABLED"
MIG_UTILIZATION_NOTE = "GPU-level utilization is not reported while MIG is enabled"
MIG_UTILIZATION_REASON = "Not reported while MIG is enabled."
UTILIZATION_REASON = "NVML reports utilization as not supported."
PROCESSES_REASON = "NVML reports process enumeration as not supported."
PROCESSES_DENIED_REASON = "Process enumeration requires more privileges."
PROCESS_MEMORY_REASON = "NVML doesn't report per-process GPU memory on this device."
RUNTIME_GPU_UTILIZATION = "RUNTIME.GPU_UTILIZATION"
RUNTIME_PROCESSES = "RUNTIME.PROCESSES"
RUNTIME_PROCESS_MEMORY = "RUNTIME.PROCESS_MEMORY"
SMI_RUNTIME_METHOD = "nvidia-smi"
# RuntimeMetrics field → nvidia-smi query field.
SMI_RUNTIME_FIELDS = {
    "gpu_utilization_percent": "utilization.gpu",
    "memory_bus_utilization_percent": "utilization.memory",
    "memory_total_bytes": "memory.total",
    "memory_used_bytes": "memory.used",
    "memory_free_bytes": "memory.free",
    "temperature_gpu_celsius": "temperature.gpu",
    "power_draw_watts": "power.draw",
    "power_limit_watts": "power.limit",
}
# RuntimeMetrics fields the nvidia-smi query doesn't carry, with their units.
SMI_RUNTIME_MISSING = {
    "memory_reserved_bytes": "bytes",
    "temperature_slowdown_threshold_celsius": "celsius",
    "processes": None,
}
_METRIC_FIELDS = tuple(
    f.name for f in fields(RuntimeMetrics) if f.name not in ("device_ref", "sampled_at")
)


class _DeviceLost(Exception):
    def __init__(self, outcome: Observed) -> None:
        self.outcome = outcome


class RuntimeCollector:
    name: ClassVar[str] = "runtime"

    def __init__(
        self,
        session: NvmlSession,
        nvidia_smi: NvidiaSmiAdapter,
        clock: Clock,
        *,
        inventory: GPUInventory,
        profile: PlatformProfile,
        include_processes: bool = True,
    ) -> None:
        self._session = session
        self._nvidia_smi = nvidia_smi
        self._clock = clock
        self._inventory = inventory
        self._profile = profile
        self._include_processes = include_processes
        synthetic = getattr(nvidia_smi, "provenance", DataOrigin.LIVE) is DataOrigin.FIXTURE
        self._prefix = FIXTURE_METHOD_PREFIX if synthetic else ""

    def collect(self) -> CollectorOutcome[tuple[RuntimeMetrics, ...]]:
        started = self._clock.monotonic_ms()
        errors: list[CollectionError] = []
        unsupported: list[UnsupportedCapability] = []
        source = self._inventory.inventory_source
        devices = self._inventory.devices
        if source is Source.NVML:
            metrics = tuple(
                self._nvml_metrics(position, device, errors, unsupported)
                for position, device in enumerate(devices)
            )
        elif source is Source.NVIDIA_SMI:
            metrics = self._smi_metrics(devices, errors)
        else:
            metrics = ()
        partial = bool(errors) or any(
            getattr(m, field).status in FAILURE_STATUSES
            for m in metrics
            for field in _METRIC_FIELDS
        )
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(
            section=metrics, run=run, errors=tuple(errors), unsupported=tuple(unsupported)
        )

    def unavailable(self, error: ErrorDetail) -> tuple[RuntimeMetrics, ...]:
        sampled_at = self._clock.now_utc()
        return tuple(
            RuntimeMetrics.unavailable(
                device.device_ref, sampled_at, CollectionStatus.COLLECTION_FAILED, error
            )
            for device in self._inventory.devices
        )

    # --- NVML -----------------------------------------------------------------------------

    def _nvml_metrics(
        self,
        position: int,
        device: GPUDeviceInfo,
        errors: list[CollectionError],
        unsupported: list[UnsupportedCapability],
    ) -> RuntimeMetrics:
        ref = device.device_ref
        sampled_at = self._clock.now_utc()
        handle = self._session.device(device.nvml_index)
        try:
            if not handle.is_collected:
                raise _DeviceLost(handle)
            assert handle.value is not None
            values = self._read(handle.value)
        except _DeviceLost as lost:
            outcome = lost.outcome
            errors.append(_error(outcome, f"{RUNTIME_REPORT_PATH}/{position}", ref))
            return RuntimeMetrics.unavailable(ref, sampled_at, outcome.status, outcome.error)
        values = self._capabilities(device, values, unsupported)
        return RuntimeMetrics(device_ref=ref, sampled_at=sampled_at, **values)

    def _read(self, device: NvmlDevice) -> dict[str, Observed]:
        """Reads every metric, stopping at the first sign the device is gone."""

        def check(*outcomes: Observed) -> None:
            for outcome in outcomes:
                if outcome.error is not None and outcome.error.code in LOST_CODES:
                    raise _DeviceLost(outcome)

        memory = device.memory_info()
        check(memory.total, memory.used, memory.free, memory.reserved)
        utilization = device.utilization()
        check(utilization.gpu, utilization.memory_bus)
        values: dict[str, Observed] = {
            "gpu_utilization_percent": utilization.gpu,
            "memory_bus_utilization_percent": utilization.memory_bus,
            "memory_total_bytes": memory.total,
            "memory_used_bytes": memory.used,
            "memory_free_bytes": memory.free,
            "memory_reserved_bytes": memory.reserved,
        }
        reads = (
            ("temperature_gpu_celsius", device.temperature),
            ("temperature_slowdown_threshold_celsius", device.slowdown_threshold),
            ("power_draw_watts", device.power_draw),
            ("power_limit_watts", device.power_limit),
        )
        for field, read in reads:
            values[field] = read()
            check(values[field])
        if self._include_processes:
            values["processes"] = device.processes()
            check(values["processes"])
        else:
            values["processes"] = Observed.skipped(note=PROCESSES_SKIPPED_NOTE)
        return values

    def _capabilities(
        self,
        device: GPUDeviceInfo,
        values: dict[str, Observed],
        unsupported: list[UnsupportedCapability],
    ) -> dict[str, Observed]:
        """The runtime capability records, and the platform notes on the fields."""
        ref, wsl2 = device.device_ref, self._profile is PlatformProfile.WSL2
        utilization = ("gpu_utilization_percent", "memory_bus_utilization_percent")
        if any(values[f].status is CollectionStatus.NOT_SUPPORTED for f in utilization):
            if wsl2:
                note, reason = WSL2_RUNTIME_NOTE, f"{WSL2_RUNTIME_NOTE}."
            elif device.mig_mode.value == MIG_ENABLED:
                note, reason = MIG_UTILIZATION_NOTE, MIG_UTILIZATION_REASON
            else:
                note, reason = None, UTILIZATION_REASON
            for field in utilization:
                values[field] = _noted(values[field], note)
            unsupported.append(_unsupported(RUNTIME_GPU_UTILIZATION, reason, ref))
        processes = values["processes"]
        if processes.status is CollectionStatus.NOT_SUPPORTED:
            note = WSL2_RUNTIME_NOTE if wsl2 else None
            values["processes"] = _noted(processes, note)
            reason = f"{WSL2_RUNTIME_NOTE}." if wsl2 else PROCESSES_REASON
            unsupported.append(_unsupported(RUNTIME_PROCESSES, reason, ref))
        elif processes.status is CollectionStatus.PERMISSION_DENIED:
            unsupported.append(
                _unsupported(
                    RUNTIME_PROCESSES,
                    PROCESSES_DENIED_REASON,
                    ref,
                    status=CollectionStatus.PERMISSION_DENIED,
                )
            )
        elif processes.is_collected:
            found: tuple[GpuProcess, ...] = processes.value or ()
            if any(p.used_memory_bytes.status is CollectionStatus.NOT_SUPPORTED for p in found):
                windows = self._profile is PlatformProfile.WINDOWS
                note = WDDM_PROCESS_MEMORY_NOTE if windows else None
                noted = tuple(
                    replace(p, used_memory_bytes=_noted(p.used_memory_bytes, note)) for p in found
                )
                values["processes"] = replace(processes, value=noted)
                reason = f"{WDDM_PROCESS_MEMORY_NOTE}." if windows else PROCESS_MEMORY_REASON
                unsupported.append(_unsupported(RUNTIME_PROCESS_MEMORY, reason, ref))
        return values

    # --- nvidia-smi fallback --------------------------------------------------------------

    def _smi_metrics(
        self, devices: tuple[GPUDeviceInfo, ...], errors: list[CollectionError]
    ) -> tuple[RuntimeMetrics, ...]:
        rows = self._nvidia_smi.query_gpus()
        by_index: dict[int, SmiGpuRow] = {row.index: row for row in rows.value or ()}
        metrics = []
        for position, device in enumerate(devices):
            ref = device.device_ref
            sampled_at = self._clock.now_utc()
            row = by_index.get(device.nvml_index)
            if row is None:  # the cached query is the inventory's own source
                failure = rows if not rows.is_collected else _missing_row(device.nvml_index)
                errors.append(_error(failure, f"{RUNTIME_REPORT_PATH}/{position}", ref))
                metrics.append(
                    RuntimeMetrics.unavailable(ref, sampled_at, failure.status, failure.error)
                )
                continue
            values = {field: row.values[smi] for field, smi in SMI_RUNTIME_FIELDS.items()}
            lost = next(
                (v for v in values.values() if v.error is not None and v.error.code in LOST_CODES),
                None,
            )
            if lost is not None:
                errors.append(_error(lost, f"{RUNTIME_REPORT_PATH}/{position}", ref))
                metrics.append(RuntimeMetrics.unavailable(ref, sampled_at, lost.status, lost.error))
                continue
            missing = {
                field: Observed.failed(
                    CollectionStatus.NOT_SUPPORTED,
                    code=FALLBACK_FIELD_UNAVAILABLE,
                    message=FALLBACK_FIELD_MESSAGE,
                    source=Source.NVIDIA_SMI,
                    method=self._prefix + SMI_RUNTIME_METHOD,
                    unit=unit,
                )
                for field, unit in SMI_RUNTIME_MISSING.items()
            }
            metrics.append(
                RuntimeMetrics(device_ref=ref, sampled_at=sampled_at, **values, **missing)
            )
        return tuple(metrics)


def _missing_row(index: int) -> Observed:
    return Observed.failed(
        CollectionStatus.COLLECTION_FAILED,
        code="DEVICE_NOT_IN_QUERY",
        message=f"nvidia-smi returned no row for GPU index {index}",
        source=Source.NVIDIA_SMI,
        method=SMI_RUNTIME_METHOD,
    )


def _noted(outcome: Observed, note: str | None) -> Observed:
    if note is not None and outcome.status is CollectionStatus.NOT_SUPPORTED and not outcome.note:
        return replace(outcome, note=note)
    return outcome


def _unsupported(
    capability: str,
    reason: str,
    ref: str,
    *,
    status: CollectionStatus = CollectionStatus.NOT_SUPPORTED,
) -> UnsupportedCapability:
    return UnsupportedCapability(
        capability=capability, status=status, device_ref=ref, reason=reason
    )


def _error(outcome: Observed, report_path: str, device_ref: str) -> CollectionError:
    assert outcome.error is not None
    return CollectionError(
        collector=RuntimeCollector.name,
        device_ref=device_ref,
        status=outcome.status,
        error_code=outcome.error.code,
        message=outcome.error.message,
        report_path=report_path,
    )
