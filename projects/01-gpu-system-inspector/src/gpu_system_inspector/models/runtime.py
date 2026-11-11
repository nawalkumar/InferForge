"""Runtime snapshot models (docs/data_model_specification.md §6.1)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    constrained,
    observed,
    set_field,
    unavailable,
    validate_model,
)
from gpu_system_inspector.models.enums import CollectionStatus, ProcessKind
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuProcess:
    pid: int = constrained(minimum=0)
    kind: ProcessKind
    used_memory_bytes: Observed[int] = observed("bytes")
    name: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeMetrics:
    device_ref: str = constrained(pattern=DEVICE_REF_REGEX)
    sampled_at: datetime
    gpu_utilization_percent: Observed[int] = observed("percent")
    memory_bus_utilization_percent: Observed[int] = observed("percent")
    memory_total_bytes: Observed[int] = observed("bytes")
    memory_used_bytes: Observed[int] = observed("bytes")
    memory_free_bytes: Observed[int] = observed("bytes")
    memory_reserved_bytes: Observed[int] = observed("bytes")
    temperature_gpu_celsius: Observed[int] = observed("celsius")
    temperature_slowdown_threshold_celsius: Observed[int] = observed("celsius")
    power_draw_watts: Observed[float] = observed("watts")
    power_limit_watts: Observed[float] = observed("watts")
    processes: Observed[tuple[GpuProcess, ...]] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        if self.processes.is_collected:
            ordered = tuple(sorted(self.processes.value or (), key=lambda p: p.pid))
            if len({p.pid for p in ordered}) != len(ordered):
                raise InvalidModelError("processes must have unique pid values")
            set_field(self, "processes", replace(self.processes, value=ordered))

    @classmethod
    def unavailable(
        cls,
        device_ref: str,
        sampled_at: datetime,
        status: CollectionStatus,
        error: ErrorDetail | None,
    ) -> RuntimeMetrics:
        return unavailable(cls, status, error, device_ref=device_ref, sampled_at=sampled_at)
