"""Normalised, immutable data models for GPU System Inspector.

Pure data: nothing here touches hardware, NVML, PyTorch, subprocesses or the OS.
"""

from gpu_system_inspector.models.collection import (
    CollectionError,
    CollectorOutcome,
    CollectorRun,
    UnsupportedCapability,
)
from gpu_system_inspector.models.cuda import (
    ENVIRONMENT_ALLOWLIST,
    CUDAEnvironmentInfo,
    CudaToolkitCandidate,
)
from gpu_system_inspector.models.diagnostics import (
    VALID_SEVERITIES,
    DiagnosticFinding,
    Evidence,
    Recommendation,
)
from gpu_system_inspector.models.driver import DRIVER_SUPPORTED_CUDA_NOTE, DriverInfo
from gpu_system_inspector.models.enums import (
    CollectionStatus,
    CollectorRunStatus,
    Confidence,
    DataOrigin,
    DiagnosticStatus,
    EvidenceKind,
    InspectionMode,
    LinuxEnvironment,
    PlatformProfile,
    ProbeMode,
    ProcessKind,
    ReadinessReason,
    ReadinessStatus,
    RedactionCategory,
    Severity,
    Source,
    ToolkitDiscovery,
)
from gpu_system_inspector.models.framework import (
    ALLOCATOR_NOT_APPLICABLE_NOTE,
    ALLOCATOR_SCOPE_NOTE,
    BUILD_CUDA_NOTE,
    PROBE_PROTOCOL_VERSION,
    AllocatorStats,
    FrameworkInfo,
    ProbeInfo,
    PyTorchInfo,
    TorchDeviceInfo,
)
from gpu_system_inspector.models.gpu import GPUDeviceInfo, GPUInventory, PciObservation
from gpu_system_inspector.models.observed import (
    FIXTURE_METHOD_PREFIX,
    UNITS,
    ErrorDetail,
    InvalidModelError,
    Observed,
)
from gpu_system_inspector.models.readiness import (
    READINESS_CRITERION,
    READINESS_DISCLAIMER,
    DeviceReadiness,
    ReadinessResult,
    required_bytes_for,
)
from gpu_system_inspector.models.report import (
    SNAPSHOT_LIMITATION,
    ApplicationInfo,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
    RedactionInfo,
)
from gpu_system_inspector.models.runtime import GpuProcess, RuntimeMetrics
from gpu_system_inspector.models.system import SystemInfo

__all__ = [
    "ALLOCATOR_NOT_APPLICABLE_NOTE",
    "ALLOCATOR_SCOPE_NOTE",
    "BUILD_CUDA_NOTE",
    "DRIVER_SUPPORTED_CUDA_NOTE",
    "ENVIRONMENT_ALLOWLIST",
    "FIXTURE_METHOD_PREFIX",
    "PROBE_PROTOCOL_VERSION",
    "READINESS_CRITERION",
    "READINESS_DISCLAIMER",
    "SNAPSHOT_LIMITATION",
    "UNITS",
    "VALID_SEVERITIES",
    "AllocatorStats",
    "ApplicationInfo",
    "CollectionError",
    "CollectionStatus",
    "CollectorOutcome",
    "CollectorRun",
    "CollectorRunStatus",
    "Confidence",
    "CudaToolkitCandidate",
    "CUDAEnvironmentInfo",
    "DataOrigin",
    "DeviceReadiness",
    "DiagnosticFinding",
    "DiagnosticStatus",
    "DriverInfo",
    "ErrorDetail",
    "Evidence",
    "EvidenceKind",
    "FrameworkInfo",
    "GPUDeviceInfo",
    "GPUInventory",
    "GpuProcess",
    "InspectionMode",
    "LinuxEnvironment",
    "InspectionOptions",
    "InspectionReport",
    "InspectionSnapshot",
    "InvalidModelError",
    "Observed",
    "PciObservation",
    "PlatformProfile",
    "ProbeInfo",
    "ProbeMode",
    "ProcessKind",
    "PyTorchInfo",
    "ReadinessReason",
    "ReadinessResult",
    "ReadinessStatus",
    "Recommendation",
    "RedactionCategory",
    "RedactionInfo",
    "RuntimeMetrics",
    "Severity",
    "Source",
    "SystemInfo",
    "ToolkitDiscovery",
    "TorchDeviceInfo",
    "UnsupportedCapability",
    "required_bytes_for",
]
