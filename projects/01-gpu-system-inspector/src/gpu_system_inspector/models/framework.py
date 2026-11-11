"""Framework (PyTorch) models (docs/data_model_specification.md §5.3, §5.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    constrained,
    observed,
    set_field,
    unavailable,
    validate_model,
)
from gpu_system_inspector.models.enums import CollectionStatus, ProbeMode
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

BUILD_CUDA_NOTE = "CUDA version PyTorch was built with; not the toolkit and not the driver."
ALLOCATOR_NOT_APPLICABLE_NOTE = (
    "Allocator statistics describe only the calling process; the inspector's probe process "
    "has made no allocations."
)
ALLOCATOR_SCOPE_NOTE = "Calling process only"
# 2 (DEC-48): adds `device_error`, a device-enumeration failure that keeps the build facts.
PROBE_PROTOCOL_VERSION = 2

PYTORCH_SECTION_STATUSES = frozenset(
    {
        CollectionStatus.COLLECTED,
        CollectionStatus.NOT_INSTALLED,
        CollectionStatus.SKIPPED,
        CollectionStatus.COLLECTION_FAILED,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeInfo:
    mode: ProbeMode
    protocol_version: int = constrained(minimum=1)
    duration_ms: int | None = constrained(minimum=0)

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchDeviceInfo:
    torch_index: int = constrained(minimum=0)
    name: Observed[str] = observed()
    compute_capability: Observed[str] = observed()
    total_memory_bytes: Observed[int] = observed("bytes")
    multiprocessor_count: Observed[int] = observed()
    uuid: Observed[str] = observed()
    pci_bus_id: Observed[str] = observed()
    correlated_device_ref: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        ref = self.correlated_device_ref
        if ref.is_collected and not _is_device_ref(ref.value):
            raise InvalidModelError(f"correlated_device_ref {ref.value!r} is not a device_ref")


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocatorStats:
    torch_index: int = constrained(minimum=0)
    allocated_bytes: Observed[int] = observed("bytes")
    reserved_bytes: Observed[int] = observed("bytes")
    peak_allocated_bytes: Observed[int] = observed("bytes")
    peak_reserved_bytes: Observed[int] = observed("bytes")

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PyTorchInfo:
    status: CollectionStatus
    distribution_version: Observed[str] = observed()
    version: Observed[str] = observed()
    cuda_build: Observed[bool] = observed()
    build_cuda_version: Observed[str] = observed()
    rocm_build: Observed[bool] = observed()
    cudnn_version: Observed[str] = observed()
    cuda_available: Observed[bool] = observed()
    cuda_init_warnings: tuple[str, ...]
    device_count: Observed[int] = observed()
    devices: tuple[TorchDeviceInfo, ...]
    allocator: Observed[tuple[AllocatorStats, ...]] = observed()
    probe: ProbeInfo

    def __post_init__(self) -> None:
        validate_model(self)
        if self.status not in PYTORCH_SECTION_STATUSES:
            raise InvalidModelError(
                f"PyTorchInfo.status must be one of {sorted(PYTORCH_SECTION_STATUSES)}"
            )
        devices = tuple(sorted(self.devices, key=lambda d: d.torch_index))
        set_field(self, "devices", devices)
        if len({d.torch_index for d in devices}) != len(devices):
            raise InvalidModelError("PyTorchInfo.devices must have unique torch_index values")
        if self.status is not CollectionStatus.COLLECTED and devices:
            raise InvalidModelError("devices can only be listed when the section is COLLECTED")
        if self.device_count.is_collected and self.device_count.value != len(devices):
            raise InvalidModelError(
                f"device_count {self.device_count.value} doesn't match {len(devices)} devices"
            )
        if self.cuda_build.is_collected and self.cuda_build.value is False:
            for name in ("build_cuda_version", "cudnn_version"):
                if getattr(self, name).status is not CollectionStatus.NOT_APPLICABLE:
                    raise InvalidModelError(f"{name} must be NOT_APPLICABLE for a non-CUDA build")
        if self.build_cuda_version.is_collected:
            if not (self.cuda_build.is_collected and self.cuda_build.value is True):
                raise InvalidModelError("build_cuda_version requires cuda_build to be true")
            if self.build_cuda_version.note != BUILD_CUDA_NOTE:
                raise InvalidModelError(
                    "build_cuda_version must carry the note that it is neither the toolkit "
                    "nor the driver version"
                )
        if self.allocator.is_collected:
            stats = tuple(sorted(self.allocator.value or (), key=lambda s: s.torch_index))
            if len({s.torch_index for s in stats}) != len(stats):
                raise InvalidModelError("allocator stats must have unique torch_index values")
            set_field(self, "allocator", replace(self.allocator, value=stats))

    @classmethod
    def unavailable(
        cls, status: CollectionStatus, error: ErrorDetail | None, *, probe: ProbeInfo
    ) -> PyTorchInfo:
        return unavailable(
            cls, status, error, status=status, cuda_init_warnings=(), devices=(), probe=probe
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameworkInfo:
    pytorch: PyTorchInfo

    def __post_init__(self) -> None:
        validate_model(self)


def _is_device_ref(value: object) -> bool:
    return isinstance(value, str) and re.match(DEVICE_REF_REGEX, value) is not None
