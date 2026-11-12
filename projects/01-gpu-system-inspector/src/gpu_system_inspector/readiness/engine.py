"""Mode B readiness (architecture §4.9, IMP-R01, DEC-50).

The only criterion is `free memory ≥ required` per inventoried device, measured once. A
missing value is never a verdict: free memory that wasn't collected makes the device
`UNKNOWN`, and "no NVIDIA device" is only concluded from the observations that prove it
(NVML counted zero devices, or a complete PCI scan found none while NVML is absent — KI-32,
KI-33).
"""

from __future__ import annotations

from gpu_system_inspector.diagnostics import context as ctx
from gpu_system_inspector.models import (
    READINESS_CRITERION,
    READINESS_DISCLAIMER,
    CollectionStatus,
    DeviceReadiness,
    GPUDeviceInfo,
    InspectionSnapshot,
    Observed,
    ReadinessReason,
    ReadinessResult,
    ReadinessStatus,
    RuntimeMetrics,
    Source,
    required_bytes_for,
)

R = ReadinessReason
_GIB = 1 << 30
SMI_PRECISION_LIMITATION = "Memory values from the nvidia-smi fallback have 1 MiB precision."
MIG_LIMITATION = (
    "MIG-enabled devices are not assessed: device-level free memory does not describe what a "
    "MIG instance can allocate."
)
NO_DEVICE_LIMITATION = "No inventoried NVIDIA device was available to measure."


def _gib(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


def _unavailable(reason: str) -> Observed[int]:
    return Observed.not_applicable(note=reason, unit="bytes")


class ReadinessEngine:
    def assess(self, snapshot: InspectionSnapshot, required_vram_gib: float) -> ReadinessResult:
        required = required_bytes_for(required_vram_gib)
        runtime = {m.device_ref: m for m in snapshot.runtime}
        devices = tuple(
            _device(d, runtime.get(d.device_ref), required) for d in snapshot.gpu.devices
        )
        status, reason = _overall(snapshot, devices)
        limitations = []
        if snapshot.runtime:
            sampled = min(m.sampled_at for m in snapshot.runtime).isoformat()
            limitations.append(f"Single snapshot at {sampled}.")
        else:
            limitations.append(NO_DEVICE_LIMITATION)
        if snapshot.gpu.inventory_source is Source.NVIDIA_SMI or any(
            m.memory_free_bytes.source is Source.NVIDIA_SMI for m in snapshot.runtime
        ):
            limitations.append(SMI_PRECISION_LIMITATION)
        if any(d.reason is R.MIG_ENABLED_UNSUPPORTED for d in devices):
            limitations.append(MIG_LIMITATION)
        return ReadinessResult(
            required_vram_gib=required_vram_gib,
            required_bytes=required,
            criterion=READINESS_CRITERION,
            overall_status=status,
            overall_reason=reason,
            devices=devices,
            disclaimer=READINESS_DISCLAIMER,
            limitations=tuple(limitations),
        )


def _device(
    device: GPUDeviceInfo, metrics: RuntimeMetrics | None, required: int
) -> DeviceReadiness:
    if metrics is not None:
        free, total = metrics.memory_free_bytes, metrics.memory_total_bytes
    else:
        free = _unavailable("No runtime sample for this device")
        total = device.memory_total_bytes
    if not total.is_collected and device.memory_total_bytes.is_collected:
        total = device.memory_total_bytes
    margin = free.value - required if free.value is not None else None
    need = _gib(required)
    if device.mig_mode.value == "ENABLED":
        status, reason = ReadinessStatus.UNKNOWN, R.MIG_ENABLED_UNSUPPORTED
        text = "MIG mode is enabled; device-level free memory is not assessed."
    elif free.value is None:
        status, reason = ReadinessStatus.UNKNOWN, R.FREE_MEMORY_UNAVAILABLE
        text = f"Free memory is {free.status}, so it can't be compared with {need}."
    elif total.value is not None and total.value < required:
        status, reason = ReadinessStatus.NOT_READY, R.INSUFFICIENT_TOTAL_MEMORY
        text = f"Total memory {_gib(total.value)} is below the required {need}."
    elif free.value >= required:
        status, reason = ReadinessStatus.READY, R.SUFFICIENT_FREE_MEMORY
        text = f"Observed {_gib(free.value)} free ({free.source}), at least the required {need}."
    else:
        status, reason = ReadinessStatus.NOT_READY, R.INSUFFICIENT_FREE_MEMORY
        text = f"Observed {_gib(free.value)} free ({free.source}), below the required {need}."
    return DeviceReadiness(
        device_ref=device.device_ref,
        status=status,
        reason=reason,
        free_bytes=free,
        total_bytes=total,
        margin_bytes=margin,
        explanation=text,
    )


def _overall(
    s: InspectionSnapshot, devices: tuple[DeviceReadiness, ...]
) -> tuple[ReadinessStatus, ReadinessReason]:
    if any(d.status is ReadinessStatus.READY for d in devices):
        return ReadinessStatus.READY, R.AT_LEAST_ONE_DEVICE_READY
    if any(d.status is ReadinessStatus.UNKNOWN for d in devices):
        return ReadinessStatus.UNKNOWN, R.SOME_DEVICES_UNKNOWN
    if devices:
        return ReadinessStatus.NOT_READY, R.NO_DEVICE_READY
    if ctx.is_macos(s):
        return ReadinessStatus.NOT_READY, R.PLATFORM_NOT_SUPPORTED
    if ctx.nvml_device_count(s) == 0 or (
        s.driver.nvml_version.status is CollectionStatus.NOT_DETECTED
        and ctx.pci_scan_complete(s)
        and not ctx.nvidia_pci(s)
    ):
        return ReadinessStatus.NOT_READY, R.NO_NVIDIA_DEVICE
    return ReadinessStatus.UNKNOWN, R.DEVICE_PRESENCE_UNKNOWN
