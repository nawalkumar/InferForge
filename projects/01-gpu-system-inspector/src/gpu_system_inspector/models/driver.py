"""`DriverInfo` (docs/data_model_specification.md §4.3)."""

from __future__ import annotations

from dataclasses import dataclass

from gpu_system_inspector.models._validation import observed, unavailable, validate_model
from gpu_system_inspector.models.enums import CollectionStatus
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

DRIVER_SUPPORTED_CUDA_NOTE = (
    "Newest CUDA version supported by the installed driver. Not the installed CUDA Toolkit version."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DriverInfo:
    nvml_binding: Observed[str] = observed()
    nvml_binding_distribution: Observed[str] = observed()
    nvml_version: Observed[str] = observed()
    driver_version: Observed[str] = observed()
    driver_supported_cuda_version: Observed[str] = observed()
    cuda_driver_library: Observed[str] = observed()
    nvidia_smi_path: Observed[str] = observed()
    nvidia_smi_driver_version: Observed[str] = observed()
    nvidia_smi_cuda_version: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        supported = self.driver_supported_cuda_version
        if supported.is_collected and supported.note != DRIVER_SUPPORTED_CUDA_NOTE:
            raise InvalidModelError(
                "driver_supported_cuda_version must carry the note that it is not the "
                "installed CUDA Toolkit version"
            )

    @classmethod
    def unavailable(cls, status: CollectionStatus, error: ErrorDetail | None) -> DriverInfo:
        return unavailable(cls, status, error)
