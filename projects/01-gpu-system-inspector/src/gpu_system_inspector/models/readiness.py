"""Readiness result models (docs/data_model_specification.md §6.2).

These are data structures only. The readiness engine that produces them lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    constrained,
    device_ref_key,
    observed,
    set_field,
    validate_model,
)
from gpu_system_inspector.models.enums import ReadinessReason, ReadinessStatus
from gpu_system_inspector.models.observed import InvalidModelError, Observed

BYTES_PER_GIB = 2**30
MAX_REQUIRED_VRAM_GIB = 1_048_576
READINESS_CRITERION = "DEVICE_FREE_MEMORY_GTE_REQUIRED"
READINESS_DISCLAIMER = (
    "Free memory at least equal to the requested amount was observed at the time of "
    "inspection. This does not guarantee that a model will fit or run: actual usage depends "
    "on weights, activations, KV cache, framework/allocator overhead, CUDA context size, "
    "fragmentation and other processes that may allocate memory after this snapshot."
)

DEVICE_REASONS: dict[ReadinessStatus, frozenset[ReadinessReason]] = {
    ReadinessStatus.READY: frozenset({ReadinessReason.SUFFICIENT_FREE_MEMORY}),
    ReadinessStatus.NOT_READY: frozenset(
        {ReadinessReason.INSUFFICIENT_FREE_MEMORY, ReadinessReason.INSUFFICIENT_TOTAL_MEMORY}
    ),
    ReadinessStatus.UNKNOWN: frozenset(
        {ReadinessReason.FREE_MEMORY_UNAVAILABLE, ReadinessReason.MIG_ENABLED_UNSUPPORTED}
    ),
}
OVERALL_REASONS: dict[ReadinessStatus, frozenset[ReadinessReason]] = {
    ReadinessStatus.READY: frozenset({ReadinessReason.AT_LEAST_ONE_DEVICE_READY}),
    ReadinessStatus.NOT_READY: frozenset(
        {
            ReadinessReason.NO_DEVICE_READY,
            ReadinessReason.NO_NVIDIA_DEVICE,
            ReadinessReason.PLATFORM_NOT_SUPPORTED,
        }
    ),
    ReadinessStatus.UNKNOWN: frozenset(
        {ReadinessReason.SOME_DEVICES_UNKNOWN, ReadinessReason.DEVICE_PRESENCE_UNKNOWN}
    ),
}


def required_bytes_for(required_vram_gib: float) -> int:
    return round(required_vram_gib * BYTES_PER_GIB)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceReadiness:
    device_ref: str = constrained(pattern=DEVICE_REF_REGEX)
    status: ReadinessStatus
    reason: ReadinessReason
    free_bytes: Observed[int] = observed("bytes")
    total_bytes: Observed[int] = observed("bytes")
    margin_bytes: int | None
    explanation: str = constrained(minLength=1)

    def __post_init__(self) -> None:
        validate_model(self)
        if self.reason not in DEVICE_REASONS[self.status]:
            raise InvalidModelError(f"reason {self.reason} doesn't fit device status {self.status}")
        if self.free_bytes.is_collected != (self.margin_bytes is not None):
            raise InvalidModelError("margin_bytes is set if and only if free_bytes is COLLECTED")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadinessResult:
    required_vram_gib: float = constrained(exclusiveMinimum=0, maximum=MAX_REQUIRED_VRAM_GIB)
    required_bytes: int = constrained(minimum=1)
    criterion: str = constrained(const=READINESS_CRITERION)
    overall_status: ReadinessStatus
    overall_reason: ReadinessReason
    devices: tuple[DeviceReadiness, ...]
    disclaimer: str = constrained(const=READINESS_DISCLAIMER)
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_model(self)
        if self.required_bytes != required_bytes_for(self.required_vram_gib):
            raise InvalidModelError("required_bytes must equal round(required_vram_gib * 2**30)")
        if self.overall_reason not in OVERALL_REASONS[self.overall_status]:
            raise InvalidModelError(
                f"reason {self.overall_reason} doesn't fit overall status {self.overall_status}"
            )
        devices = tuple(sorted(self.devices, key=lambda d: device_ref_key(d.device_ref)))
        set_field(self, "devices", devices)
        if len({d.device_ref for d in devices}) != len(devices):
            raise InvalidModelError("ReadinessResult.devices must have unique device_ref values")
        for device in devices:
            free = device.free_bytes
            if free.is_collected and device.margin_bytes != free.value - self.required_bytes:
                raise InvalidModelError(
                    f"{device.device_ref}: margin_bytes must equal free_bytes - required_bytes"
                )
