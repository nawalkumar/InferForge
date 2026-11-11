"""CUDA Toolkit and environment models (docs/data_model_specification.md §5.1, §5.2, §5.5)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from gpu_system_inspector.models._validation import (
    constrained,
    observed,
    set_field,
    unavailable,
    validate_model,
)
from gpu_system_inspector.models.enums import CollectionStatus, ToolkitDiscovery
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

ENVIRONMENT_ALLOWLIST: tuple[str, ...] = (
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
)
CUDA_PATH_VERSIONED_REGEX = r"^CUDA_PATH_V[0-9A-Z_]+$"
ENVIRONMENT_NAME_REGEX = "^(" + "|".join(ENVIRONMENT_ALLOWLIST) + ")$|" + CUDA_PATH_VERSIONED_REGEX
_ENVIRONMENT_NAME = re.compile(ENVIRONMENT_NAME_REGEX)
_DISCOVERY_ORDER = {member: i for i, member in enumerate(ToolkitDiscovery)}


@dataclass(frozen=True, slots=True, kw_only=True)
class CudaToolkitCandidate:
    root_path: str = constrained(minLength=1)
    discovered_via: tuple[ToolkitDiscovery, ...]
    version: Observed[str] = observed()
    nvcc_path: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        if not self.discovered_via:
            raise InvalidModelError("CudaToolkitCandidate.discovered_via must not be empty")
        ordered = tuple(sorted(set(self.discovered_via), key=_DISCOVERY_ORDER.__getitem__))
        set_field(self, "discovered_via", ordered)


@dataclass(frozen=True, slots=True, kw_only=True)
class CUDAEnvironmentInfo:
    toolkits: Observed[tuple[CudaToolkitCandidate, ...]] = observed()
    active_nvcc_path: Observed[str] = observed()
    active_nvcc_version: Observed[str] = observed()
    environment: Mapping[str, Observed[str]] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        for name in self.environment:
            if _ENVIRONMENT_NAME.match(name) is None:
                raise InvalidModelError(f"environment variable {name!r} is not on the allowlist")
        if self.toolkits.is_collected:
            candidates = tuple(sorted(self.toolkits.value or (), key=lambda c: c.root_path))
            if not candidates:
                raise InvalidModelError("toolkits must be NOT_DETECTED, not COLLECTED and empty")
            if len({c.root_path for c in candidates}) != len(candidates):
                raise InvalidModelError("toolkit candidates must have unique root_path values")
            set_field(self, "toolkits", replace(self.toolkits, value=candidates))

    @classmethod
    def unavailable(
        cls, status: CollectionStatus, error: ErrorDetail | None
    ) -> CUDAEnvironmentInfo:
        return unavailable(cls, status, error, environment={})
