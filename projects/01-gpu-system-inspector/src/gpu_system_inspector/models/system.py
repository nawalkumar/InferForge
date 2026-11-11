"""`SystemInfo` (docs/data_model_specification.md §4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from gpu_system_inspector.models._validation import observed, unavailable, validate_model
from gpu_system_inspector.models.enums import CollectionStatus, LinuxEnvironment, PlatformProfile
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

LINUX_PROFILES = (PlatformProfile.LINUX, PlatformProfile.WSL2)


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemInfo:
    platform_profile: PlatformProfile
    os_name: Observed[str] = observed()
    os_release: Observed[str] = observed()
    os_version: Observed[str] = observed()
    os_distribution: Observed[str] = observed()
    architecture: Observed[str] = observed()
    processor_model: Observed[str] = observed()
    logical_cpu_count: Observed[int] = observed()
    memory_total_bytes: Observed[int] = observed("bytes")
    memory_available_bytes: Observed[int] = observed("bytes")
    python_version: Observed[str] = observed()
    python_implementation: Observed[str] = observed()
    python_executable: Observed[str] = observed()
    linux_environment: Observed[LinuxEnvironment] = observed()

    # A classified environment must agree with the profile: WSL2 ⇔ profile WSL2, any other
    # classification ⇒ profile LINUX, and non-Linux profiles carry no classification.
    _SCHEMA_EXTRA: ClassVar[dict[str, Any]] = {
        "allOf": [
            {
                "if": {
                    "properties": {
                        "linux_environment": {"properties": {"value": {"const": "WSL2"}}}
                    }
                },
                "then": {"properties": {"platform_profile": {"const": "WSL2"}}},
            },
            {
                "if": {
                    "properties": {
                        "linux_environment": {
                            "properties": {"value": {"enum": ["NATIVE_LINUX", "WSL1", "UNKNOWN"]}}
                        }
                    }
                },
                "then": {"properties": {"platform_profile": {"const": "LINUX"}}},
            },
            {
                "if": {"properties": {"platform_profile": {"enum": ["WINDOWS", "MACOS", "OTHER"]}}},
                "then": {
                    "properties": {"linux_environment": {"properties": {"value": {"type": "null"}}}}
                },
            },
        ]
    }

    def __post_init__(self) -> None:
        validate_model(self)
        environment = self.linux_environment.value
        if environment is None:
            return
        if self.platform_profile not in LINUX_PROFILES:
            raise InvalidModelError(
                f"linux_environment must not be classified on {self.platform_profile}"
            )
        expected = (
            PlatformProfile.WSL2 if environment is LinuxEnvironment.WSL2 else PlatformProfile.LINUX
        )
        if self.platform_profile is not expected:
            raise InvalidModelError(
                f"linux_environment {environment} requires platform_profile {expected}"
            )

    @classmethod
    def unavailable(
        cls,
        status: CollectionStatus,
        error: ErrorDetail | None,
        *,
        platform_profile: PlatformProfile = PlatformProfile.OTHER,
    ) -> SystemInfo:
        return unavailable(cls, status, error, platform_profile=platform_profile)
