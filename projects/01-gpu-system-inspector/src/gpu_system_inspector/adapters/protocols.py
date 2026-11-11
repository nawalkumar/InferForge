"""Adapter interfaces (docs/architecture.md §4.5).

Typed contracts between the inspector and the outside world. Implementations wrap NVML,
processes, the OS or PyTorch; nothing in this module does. Every operation is read-only
and returns `Observed` outcomes: environmental conditions never raise, and only a broken
caller contract raises `AdapterUsageError` (see `adapters.outcomes`). Third-party objects
(`pynvml` handles and structs, `subprocess` results, `torch` objects) never cross these
interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from gpu_system_inspector.adapters.outcomes import AdapterUsageError, CommandResult
from gpu_system_inspector.adapters.probe_protocol import TorchProbeReport
from gpu_system_inspector.models import (
    CollectionStatus,
    CudaToolkitCandidate,
    ErrorDetail,
    GpuProcess,
    Observed,
    PciObservation,
    PlatformProfile,
    Source,
    SystemInfo,
    ToolkitDiscovery,
)

# --- NVML -------------------------------------------------------------------------------

NVML_SESSION_FUNCTIONS: dict[str, str] = {
    "open": "nvmlInit",
    "close": "nvmlShutdown",
    "init_result": "nvmlSystemGetNVMLVersion",
    "driver_version": "nvmlSystemGetDriverVersion",
    "cuda_driver_version": "nvmlSystemGetCudaDriverVersion",
    "cuda_driver_library": "nvmlSystemGetCudaDriverVersion_v2",
    "device_count": "nvmlDeviceGetCount_v2",
    "device": "nvmlDeviceGetHandleByIndex_v2",
}
NVML_DEVICE_FUNCTIONS: dict[str, str] = {
    "name": "nvmlDeviceGetName",
    "uuid": "nvmlDeviceGetUUID",
    "pci_info": "nvmlDeviceGetPciInfo_v3",
    "memory_info": "nvmlDeviceGetMemoryInfo_v2",
    "compute_capability": "nvmlDeviceGetCudaComputeCapability",
    "mig_mode": "nvmlDeviceGetMigMode",
    "utilization": "nvmlDeviceGetUtilizationRates",
    "temperature": "nvmlDeviceGetTemperatureV",
    "slowdown_threshold": "nvmlDeviceGetTemperatureThreshold",
    "power_draw": "nvmlDeviceGetPowerUsage",
    "power_limit": "nvmlDeviceGetEnforcedPowerLimit",
    "processes": "nvmlDeviceGetComputeRunningProcesses_v3",
}
# Older variants an implementation may fall back to (architecture §4.5), plus process names.
NVML_FALLBACK_FUNCTIONS = frozenset(
    {
        "nvmlInit_v2",
        "nvmlDeviceGetCount",
        "nvmlDeviceGetHandleByIndex",
        "nvmlDeviceGetPciInfo",
        "nvmlDeviceGetMemoryInfo",
        "nvmlDeviceGetTemperature",
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlSystemGetProcessName",
    }
)
NVML_READ_ONLY_FUNCTIONS = frozenset(
    {*NVML_SESSION_FUNCTIONS.values(), *NVML_DEVICE_FUNCTIONS.values(), *NVML_FALLBACK_FUNCTIONS}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NvmlPciInfo:
    """One `nvmlDeviceGetPciInfo` read."""

    bus_id: Observed[str]
    device_id: Observed[str]
    subsystem_id: Observed[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class NvmlMemoryInfo:
    """One `nvmlDeviceGetMemoryInfo` read, so total, used and free are mutually consistent."""

    total: Observed[int]
    used: Observed[int]
    free: Observed[int]
    reserved: Observed[int]


@dataclass(frozen=True, slots=True, kw_only=True)
class NvmlUtilization:
    """One `nvmlDeviceGetUtilizationRates` read."""

    gpu: Observed[int]
    memory_bus: Observed[int]


@runtime_checkable
class NvmlDevice(Protocol):
    """A device handle. Valid only while its session is open."""

    @property
    def index(self) -> int: ...
    def name(self) -> Observed[str]: ...
    def uuid(self) -> Observed[str]: ...
    def pci_info(self) -> NvmlPciInfo: ...
    def memory_info(self) -> NvmlMemoryInfo: ...
    def compute_capability(self) -> Observed[str]: ...
    def mig_mode(self) -> Observed[str]: ...
    def utilization(self) -> NvmlUtilization: ...
    def temperature(self) -> Observed[int]: ...
    def slowdown_threshold(self) -> Observed[int]: ...
    def power_draw(self) -> Observed[float]: ...
    def power_limit(self) -> Observed[float]: ...
    def processes(self) -> Observed[tuple[GpuProcess, ...]]: ...


@runtime_checkable
class NvmlSession(Protocol):
    """One initialised NVML session, shared by every NVML-backed collector.

    `init_result` is the NVML accessibility outcome (`COLLECTED` with the NVML version, or
    the initialisation failure). If initialisation failed, every query returns that failure
    with the queried field's unit. `close()` is idempotent and calls `nvmlShutdown` at most
    once, and only after a successful initialisation.
    """

    @property
    def init_result(self) -> Observed[str]: ...
    @property
    def is_open(self) -> bool: ...
    def driver_version(self) -> Observed[str]: ...
    def cuda_driver_version(self) -> Observed[str]: ...
    def cuda_driver_library(self) -> Observed[str]: ...
    def device_count(self) -> Observed[int]: ...
    def device(self, index: int) -> Observed[NvmlDevice]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class NvmlAdapter(Protocol):
    """Mandatory. Binding facts without initialising NVML, and session creation."""

    def binding_version(self) -> Observed[str]: ...
    def binding_distribution(self) -> Observed[str]: ...
    def open(self) -> NvmlSession: ...


# --- nvidia-smi ---------------------------------------------------------------------------

NVIDIA_SMI_EXECUTABLE = "nvidia-smi"
NVIDIA_SMI_WSL_PATH = "/usr/lib/wsl/lib/nvidia-smi"
NVIDIA_SMI_QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "driver_version",
)


def nvidia_smi_version_argv(executable: str) -> tuple[str, ...]:
    """Plain `nvidia-smi`: its header carries the driver and CUDA versions (EV-03)."""
    return (executable,)


def nvidia_smi_query_argv(executable: str) -> tuple[str, ...]:
    return (
        executable,
        "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SmiVersion:
    driver_version: Observed[str]
    cuda_version: Observed[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class SmiGpuRow:
    """One `--query-gpu` row. Every requested field except `index` is present; a value the
    GPU can't report is a non-collected `Observed`, never omitted."""

    index: int
    values: Mapping[str, Observed] = field(default_factory=dict)


@runtime_checkable
class NvidiaSmiAdapter(Protocol):
    """Optional tool. When `locate()` fails, `version()` and `query_gpus()` return that
    failure without running anything."""

    def locate(self) -> Observed[str]: ...
    def version(self) -> SmiVersion: ...
    def query_gpus(self) -> Observed[tuple[SmiGpuRow, ...]]: ...


# --- PCI ----------------------------------------------------------------------------------


class PciFailureKind(StrEnum):
    """Why one PCI device couldn't be (fully) observed."""

    INVALID_ADDRESS = "INVALID_ADDRESS"  # entry isn't a canonical PCI address; device omitted
    ATTRIBUTE_UNREADABLE = "ATTRIBUTE_UNREADABLE"  # vendor/device/class read failed; omitted
    ATTRIBUTE_MALFORMED = "ATTRIBUTE_MALFORMED"  # attribute isn't canonical hex; omitted
    DRIVER_LINK_UNREADABLE = "DRIVER_LINK_UNREADABLE"  # observation kept, kernel_driver failed


@dataclass(frozen=True, slots=True, kw_only=True)
class PciDeviceFailure:
    """One device the scan couldn't fully observe. `address` is the sysfs entry as listed
    (it may be malformed); `attribute` names the failed attribute (`None` for the address)."""

    address: str
    kind: PciFailureKind
    attribute: str | None
    status: CollectionStatus
    error: ErrorDetail

    def __post_init__(self) -> None:
        if self.status is CollectionStatus.COLLECTED:
            raise AdapterUsageError("a PCI device failure can't have status COLLECTED")


@dataclass(frozen=True, slots=True, kw_only=True)
class PciScan:
    """A PCI scan: the observations (or the whole-scan outcome) plus per-device failures.

    Failures exist only for a scan that ran (`observations` `COLLECTED`); they are sorted by
    `(address, attribute)` so the result is deterministic.
    """

    observations: Observed[tuple[PciObservation, ...]]
    failures: tuple[PciDeviceFailure, ...] = ()

    def __post_init__(self) -> None:
        failures = tuple(sorted(self.failures, key=lambda f: (f.address, f.attribute or "")))
        object.__setattr__(self, "failures", failures)
        if failures and not self.observations.is_collected:
            raise AdapterUsageError("per-device failures require a scan that ran (COLLECTED)")


@runtime_checkable
class PciScanner(Protocol):
    """Driver-independent PCI display-controller scan (FR-04). Only native Linux scans;
    every other environment is `NOT_SUPPORTED` without touching sysfs."""

    def scan(self) -> PciScan: ...


# --- PyTorch ------------------------------------------------------------------------------


@runtime_checkable
class TorchProbe(Protocol):
    """Optional framework. `distribution_version()` never imports torch. Callers must not
    call `probe()` when the distribution is `NOT_INSTALLED` or PyTorch is skipped."""

    def distribution_version(self) -> Observed[str]: ...
    def probe(self, timeout_s: float) -> Observed[TorchProbeReport]: ...


# --- Platform -----------------------------------------------------------------------------


@runtime_checkable
class PlatformAdapter(Protocol):
    """Mandatory. The single seam for OS facts and environment reads. `getenv` accepts only
    allowlisted names (data model §5.5); an unset variable is `NOT_DETECTED`."""

    @property
    def profile(self) -> PlatformProfile: ...
    def system_info(self) -> SystemInfo: ...
    def getenv(self, name: str) -> Observed[str]: ...
    def versioned_cuda_path_names(self) -> tuple[str, ...]:
        """Set `CUDA_PATH_V*` names, sorted; Windows only, else `()` (KI-28, DEC-47).
        Names are filtered by `CUDA_PATH_VERSIONED_REGEX` before any value is read."""
        ...


# --- CUDA Toolkit -------------------------------------------------------------------------

# The signals a caller supplies (environment values); the locator adds the others.
TOOLKIT_HINT_KINDS = frozenset(
    {ToolkitDiscovery.CUDA_HOME, ToolkitDiscovery.CUDA_PATH, ToolkitDiscovery.CUDA_PATH_VERSIONED}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolkitProblem:
    """A discovery failure no field shows, e.g. an unreadable default location next to
    candidates found through other signals."""

    path: str
    status: CollectionStatus
    error: ErrorDetail


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolkitDiscoveryResult:
    toolkits: Observed[tuple[CudaToolkitCandidate, ...]]
    active_nvcc_path: Observed[str]
    active_nvcc_version: Observed[str]
    problems: tuple[ToolkitProblem, ...] = ()


@runtime_checkable
class ToolkitLocator(Protocol):
    """CUDA Toolkit discovery (architecture §4.6, DEC-47). `hints` maps the environment
    signals (`TOOLKIT_HINT_KINDS`) to the paths they name; `NVCC_ON_PATH` and
    `DEFAULT_LOCATION` are the locator's own. Unsupported platforms answer
    `NOT_SUPPORTED` without reading or running anything."""

    def discover(
        self, hints: Mapping[ToolkitDiscovery, Sequence[str]]
    ) -> ToolkitDiscoveryResult: ...


# --- Commands -----------------------------------------------------------------------------


@runtime_checkable
class CommandRunner(Protocol):
    """Mandatory. Runs fixed argument vectors without a shell. A process that ran is
    `COLLECTED` even with a non-zero exit (callers decide, see `require_success`); a missing
    executable is `NOT_DETECTED`, denied execution `PERMISSION_DENIED`, a timeout
    `COLLECTION_FAILED`/`TIMEOUT`. Output beyond `MAX_COMMAND_OUTPUT_CHARS` is truncated."""

    def which(self, name: str, *, extra_paths: Sequence[str] = ()) -> Observed[str]: ...
    def run(
        self, argv: Sequence[str], *, timeout_s: float, source: Source
    ) -> Observed[CommandResult]: ...


# --- Clock --------------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    """`now_utc()` is timezone-aware UTC; `monotonic_ms()` never decreases."""

    def now_utc(self) -> datetime: ...
    def monotonic_ms(self) -> int: ...
