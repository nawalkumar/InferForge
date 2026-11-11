"""Shared enumerations (docs/data_model_specification.md §2, §3.1).

Every member's value equals its name, so enums serialise as their upper-case name.
"""

from enum import StrEnum


class CollectionStatus(StrEnum):
    COLLECTED = "COLLECTED"
    NOT_INSTALLED = "NOT_INSTALLED"
    NOT_DETECTED = "NOT_DETECTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    COLLECTION_FAILED = "COLLECTION_FAILED"
    SKIPPED = "SKIPPED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class Source(StrEnum):
    PLATFORM = "PLATFORM"
    OS = "OS"
    ENVIRONMENT = "ENVIRONMENT"
    NVML = "NVML"
    NVIDIA_SMI = "NVIDIA_SMI"
    CUDA_TOOLKIT = "CUDA_TOOLKIT"
    PYTORCH = "PYTORCH"
    INSPECTOR = "INSPECTOR"


class DataOrigin(StrEnum):
    LIVE = "LIVE"
    FIXTURE = "FIXTURE"


class InspectionMode(StrEnum):
    GENERAL = "GENERAL"
    READINESS = "READINESS"


class PlatformProfile(StrEnum):
    LINUX = "LINUX"
    WSL2 = "WSL2"
    WINDOWS = "WINDOWS"
    MACOS = "MACOS"
    OTHER = "OTHER"


class LinuxEnvironment(StrEnum):
    """What kind of Linux kernel environment the inspector runs in (data model §4.1).

    `UNKNOWN` means the kernel evidence is ambiguous, conflicting, malformed or unreadable;
    it is a classification result, not the `CollectionStatus.UNKNOWN` defect signal.
    Non-Linux platforms carry no value (`NOT_APPLICABLE` status).
    """

    NATIVE_LINUX = "NATIVE_LINUX"
    WSL1 = "WSL1"
    WSL2 = "WSL2"
    UNKNOWN = "UNKNOWN"


class DiagnosticStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    INFO = "INFO"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceKind(StrEnum):
    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"


class ReadinessStatus(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    UNKNOWN = "UNKNOWN"


class ReadinessReason(StrEnum):
    # Device-level reasons.
    SUFFICIENT_FREE_MEMORY = "SUFFICIENT_FREE_MEMORY"
    INSUFFICIENT_FREE_MEMORY = "INSUFFICIENT_FREE_MEMORY"
    INSUFFICIENT_TOTAL_MEMORY = "INSUFFICIENT_TOTAL_MEMORY"
    FREE_MEMORY_UNAVAILABLE = "FREE_MEMORY_UNAVAILABLE"
    MIG_ENABLED_UNSUPPORTED = "MIG_ENABLED_UNSUPPORTED"
    # Overall reasons.
    AT_LEAST_ONE_DEVICE_READY = "AT_LEAST_ONE_DEVICE_READY"
    NO_DEVICE_READY = "NO_DEVICE_READY"
    SOME_DEVICES_UNKNOWN = "SOME_DEVICES_UNKNOWN"
    NO_NVIDIA_DEVICE = "NO_NVIDIA_DEVICE"
    PLATFORM_NOT_SUPPORTED = "PLATFORM_NOT_SUPPORTED"
    DEVICE_PRESENCE_UNKNOWN = "DEVICE_PRESENCE_UNKNOWN"


class CollectorRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ProbeMode(StrEnum):
    SUBPROCESS = "SUBPROCESS"
    IN_PROCESS = "IN_PROCESS"


class ProcessKind(StrEnum):
    COMPUTE = "COMPUTE"
    GRAPHICS = "GRAPHICS"
    COMPUTE_AND_GRAPHICS = "COMPUTE_AND_GRAPHICS"


class ToolkitDiscovery(StrEnum):
    CUDA_HOME = "CUDA_HOME"
    CUDA_PATH = "CUDA_PATH"
    CUDA_PATH_VERSIONED = "CUDA_PATH_VERSIONED"
    NVCC_ON_PATH = "NVCC_ON_PATH"
    DEFAULT_LOCATION = "DEFAULT_LOCATION"


class RedactionCategory(StrEnum):
    PATH = "PATH"
    PROCESS_NAME = "PROCESS_NAME"
    DEVICE_ID = "DEVICE_ID"
    USER = "USER"
