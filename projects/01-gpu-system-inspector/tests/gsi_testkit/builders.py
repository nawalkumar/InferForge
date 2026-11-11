"""Synthetic model builders shared by the FX fixtures and the general model tests.

Every value is synthetic and deterministic. Every `Observed.method` carries the
`fixture:` prefix.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from gpu_system_inspector.adapters.pci_sysfs import (
    PCI_SCAN_UNSUPPORTED_CODES,
    PCI_SCAN_UNSUPPORTED_REASONS,
)
from gpu_system_inspector.adapters.platform_os import (
    CPU_COUNT_NOTE,
    ENVIRONMENT_METHOD,
    LINUX_AVAILABLE_NOTE,
    LINUX_TOTAL_NOTE,
    MAC_CPU_SYSCTL,
    MAC_MEMORY_SYSCTL,
    MACOS_AVAILABLE_REASON,
    MACOS_TOTAL_NOTE,
    NOT_LINUX_NOTE,
    WINDOWS_AVAILABLE_NOTE,
    WINDOWS_CPU_KEY,
    WINDOWS_TOTAL_NOTE,
    WSL2_MEMORY_NOTE,
    classify_linux_environment,
)
from gpu_system_inspector.collectors.capabilities import unsupported_capabilities_for
from gpu_system_inspector.models import (
    ALLOCATOR_NOT_APPLICABLE_NOTE,
    BUILD_CUDA_NOTE,
    DRIVER_SUPPORTED_CUDA_NOTE,
    ENVIRONMENT_ALLOWLIST,
    FIXTURE_METHOD_PREFIX,
    PROBE_PROTOCOL_VERSION,
    READINESS_CRITERION,
    READINESS_DISCLAIMER,
    SNAPSHOT_LIMITATION,
    CollectionError,
    CollectionStatus,
    CollectorRun,
    CollectorRunStatus,
    Confidence,
    CUDAEnvironmentInfo,
    CudaToolkitCandidate,
    DataOrigin,
    DeviceReadiness,
    DiagnosticFinding,
    DiagnosticStatus,
    DriverInfo,
    ErrorDetail,
    Evidence,
    EvidenceKind,
    FrameworkInfo,
    GPUDeviceInfo,
    GPUInventory,
    GpuProcess,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
    LinuxEnvironment,
    Observed,
    PciObservation,
    PlatformProfile,
    ProbeInfo,
    ProbeMode,
    ProcessKind,
    PyTorchInfo,
    ReadinessReason,
    ReadinessResult,
    ReadinessStatus,
    Recommendation,
    RuntimeMetrics,
    Severity,
    Source,
    SystemInfo,
    ToolkitDiscovery,
    TorchDeviceInfo,
    UnsupportedCapability,
    required_bytes_for,
)

GIB = 2**30
MIB = 2**20
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)
NVML_BINDING_VERSION = "13.615.71"
COLLECTOR_ORDER = (
    "system",
    "pci",
    "nvml",
    "driver",
    "gpu_hardware",
    "runtime",
    "nvidia_smi",
    "cuda_environment",
    "pytorch",
    "correlation",
)


def fx(method: str) -> str:
    return FIXTURE_METHOD_PREFIX + method


# --- Observed primitives ----------------------------------------------------------------


def ok(
    value: Any,
    source: Source = Source.NVML,
    unit: str | None = None,
    *,
    method: str = "synthetic",
    note: str | None = None,
) -> Observed:
    return Observed.collected(value, source=source, method=fx(method), unit=unit, note=note)


def failed(
    status: CollectionStatus,
    code: str,
    message: str = "synthetic failure",
    *,
    source: Source | None = Source.NVML,
    method: str = "synthetic",
    unit: str | None = None,
    note: str | None = None,
) -> Observed:
    return Observed.failed(
        status, code=code, message=message, source=source, method=fx(method), unit=unit, note=note
    )


def skipped(note: str, unit: str | None = None) -> Observed:
    return Observed.skipped(note=note, unit=unit)


def not_applicable(
    note: str, unit: str | None = None, *, source: Source | None = None, method: str | None = None
) -> Observed:
    return Observed.not_applicable(
        note=note, unit=unit, source=source, method=fx(method) if method else None
    )


def recast(obs: Observed, *, unit: str | None) -> Observed:
    """The same outcome for a field with a different unit (used for shared failures)."""
    return replace(obs, unit=unit)


# --- system -------------------------------------------------------------------------------

_LINUX_SYSTEM = dict(
    os_name="Linux",
    version="#1 SMP synthetic",
    arch="x86_64",
    processor="Synthetic CPU (fixture)",
    python_exe="/opt/synthetic/venv/bin/python",
)
SYSTEM_PROFILES: dict[PlatformProfile, dict[str, Any]] = {
    PlatformProfile.LINUX: dict(
        _LINUX_SYSTEM, release="6.8.0-synthetic", distro="Synthetic Linux 24.04 (fixture)"
    ),
    PlatformProfile.WSL2: dict(
        _LINUX_SYSTEM,
        release="5.15.167.4-microsoft-standard-WSL2",
        distro="Synthetic Linux 22.04 (fixture)",
    ),
    PlatformProfile.WINDOWS: dict(
        os_name="Windows",
        release="11",
        version="10.0.26100",
        edition="Professional",
        distro="Windows 11 Professional",
        arch="AMD64",
        processor="Synthetic CPU (fixture)",
        python_exe="C:/Synthetic/venv/Scripts/python.exe",
    ),
    PlatformProfile.MACOS: dict(
        os_name="Darwin",
        release="24.0.0",
        version="Darwin Kernel Version 24.0.0 (synthetic)",
        mac_release="15.0",
        distro="macOS 15.0",
        arch="arm64",
        processor="Synthetic Apple Silicon (fixture)",
        python_exe="/opt/synthetic/venv/bin/python",
    ),
}
SYSTEM_TOTAL_BYTES = 64 * GIB
SYSTEM_AVAILABLE_BYTES = 48 * GIB
SYSTEM_CPUS = 16


def synthetic_proc_version(release: str) -> str:
    return f"Linux version {release} (fixture@synthetic) (gcc 13.2.0) #1 SMP synthetic\n"


def _linux_environment(profile: PlatformProfile) -> Observed:
    if profile not in (PlatformProfile.LINUX, PlatformProfile.WSL2):
        return Observed.not_applicable(note=NOT_LINUX_NOTE, source=Source.INSPECTOR)
    release = SYSTEM_PROFILES[profile]["release"]
    environment, rationale = classify_linux_environment(
        release + "\n", synthetic_proc_version(release)
    )
    expected = (
        LinuxEnvironment.WSL2 if profile is PlatformProfile.WSL2 else LinuxEnvironment.NATIVE_LINUX
    )
    assert environment is expected, rationale
    return ok(environment, Source.INSPECTOR, method=ENVIRONMENT_METHOD, note=rationale)


def _system_platform_fields(profile: PlatformProfile, processor_detected: bool) -> dict:
    """Distribution, processor and memory exactly as the production adapter reports them."""
    p = SYSTEM_PROFILES[profile]
    if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
        vm = WSL2_MEMORY_NOTE if profile is PlatformProfile.WSL2 else ""
        processor = (
            ok(p["processor"], Source.OS, method="/proc/cpuinfo:model name")
            if processor_detected
            else failed(
                CollectionStatus.NOT_DETECTED,
                "FIELD_NOT_FOUND",
                "no 'model name' entry",
                source=Source.OS,
                method="/proc/cpuinfo:model name",
            )
        )
        return dict(
            os_distribution=ok(p["distro"], Source.OS, method="/etc/os-release:PRETTY_NAME"),
            processor_model=processor,
            memory_total_bytes=ok(
                SYSTEM_TOTAL_BYTES,
                Source.OS,
                "bytes",
                method="/proc/meminfo:MemTotal",
                note=LINUX_TOTAL_NOTE + vm,
            ),
            memory_available_bytes=ok(
                SYSTEM_AVAILABLE_BYTES,
                Source.OS,
                "bytes",
                method="/proc/meminfo:MemAvailable",
                note=LINUX_AVAILABLE_NOTE + vm,
            ),
        )
    if profile is PlatformProfile.WINDOWS:
        return dict(
            os_distribution=ok(
                p["distro"],
                Source.PLATFORM,
                method="platform.win32_ver(), platform.win32_edition()",
            ),
            processor_model=ok(
                p["processor"],
                Source.OS,
                method=f"winreg:HKLM\\{WINDOWS_CPU_KEY}:ProcessorNameString",
            ),
            memory_total_bytes=ok(
                SYSTEM_TOTAL_BYTES,
                Source.OS,
                "bytes",
                method="GlobalMemoryStatusEx.ullTotalPhys",
                note=WINDOWS_TOTAL_NOTE,
            ),
            memory_available_bytes=ok(
                SYSTEM_AVAILABLE_BYTES,
                Source.OS,
                "bytes",
                method="GlobalMemoryStatusEx.ullAvailPhys",
                note=WINDOWS_AVAILABLE_NOTE,
            ),
        )
    if profile is PlatformProfile.MACOS:
        return dict(
            os_distribution=ok(p["distro"], Source.PLATFORM, method="platform.mac_ver()"),
            processor_model=ok(p["processor"], Source.OS, method=f"sysctlbyname({MAC_CPU_SYSCTL})"),
            memory_total_bytes=ok(
                SYSTEM_TOTAL_BYTES,
                Source.OS,
                "bytes",
                method=f"sysctlbyname({MAC_MEMORY_SYSCTL})",
                note=MACOS_TOTAL_NOTE,
            ),
            memory_available_bytes=Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code="PLATFORM_NOT_SUPPORTED",
                message=MACOS_AVAILABLE_REASON,
                source=Source.OS,
                unit="bytes",
            ),
        )
    raise ValueError(f"no synthetic system profile for {profile}")


def system_info(
    profile: PlatformProfile = PlatformProfile.LINUX, *, processor_detected: bool = True
) -> SystemInfo:
    """The production adapter's output for the synthetic host `FakeHostApi.for_profile()`."""
    p = SYSTEM_PROFILES[profile]
    return SystemInfo(
        platform_profile=profile,
        os_name=ok(p["os_name"], Source.PLATFORM, method="platform.system()"),
        os_release=ok(p["release"], Source.PLATFORM, method="platform.release()"),
        os_version=ok(p["version"], Source.PLATFORM, method="platform.version()"),
        architecture=ok(p["arch"], Source.PLATFORM, method="platform.machine()"),
        logical_cpu_count=ok(
            SYSTEM_CPUS, Source.PLATFORM, method="os.cpu_count()", note=CPU_COUNT_NOTE
        ),
        python_version=ok("3.12.12", Source.PLATFORM, method="platform.python_version()"),
        python_implementation=ok(
            "CPython", Source.PLATFORM, method="platform.python_implementation()"
        ),
        python_executable=ok(p["python_exe"], Source.PLATFORM, method="sys.executable"),
        linux_environment=_linux_environment(profile),
        **_system_platform_fields(profile, processor_detected),
    )


# --- GPU inventory ------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSpec:
    """A synthetic GPU model. Names say `fixture` so they can't pass for real products."""

    name: str
    total_bytes: int
    uuid: str
    compute_capability: str = "8.9"
    pci_device_id: str = "0x268410de"


GPU_A = GpuSpec("Synthetic GPU A (fixture)", 24 * GIB, "GPU-aaaaaaaa-0000-4000-8000-00000000000a")
GPU_B = GpuSpec("Synthetic GPU B (fixture)", 12 * GIB, "GPU-bbbbbbbb-0000-4000-8000-00000000000b")
GPU_M = GpuSpec(
    "Synthetic GPU M (fixture, MIG-capable)",
    80 * GIB,
    "GPU-cccccccc-0000-4000-8000-00000000000c",
    compute_capability="9.0",
)

_SMI_MISSING = "not part of the nvidia-smi fallback query"


def bus_id(bus: int) -> str:
    return f"00000000:{bus:02x}:00.0"


def gpu_device(
    index: int,
    spec: GpuSpec,
    *,
    bus: int | None = None,
    source: Source = Source.NVML,
    mig_mode: str = "DISABLED",
    memory_total: Observed | None = None,
) -> GPUDeviceInfo:
    bus = index + 1 if bus is None else bus
    if source is Source.NVIDIA_SMI:

        def smi(
            field: str, value: Any, unit: str | None = None, note: str | None = None
        ) -> Observed:
            return ok(
                value, Source.NVIDIA_SMI, unit, method=f"nvidia-smi --query-gpu={field}", note=note
            )

        def missing(method: str) -> Observed:
            return failed(
                CollectionStatus.NOT_SUPPORTED,
                "FALLBACK_FIELD_UNAVAILABLE",
                _SMI_MISSING,
                source=Source.NVIDIA_SMI,
                method=method,
            )

        return GPUDeviceInfo(
            device_ref=f"smi:{index}",
            nvml_index=index,
            vendor="NVIDIA",
            name=smi("name", spec.name),
            uuid=smi("uuid", spec.uuid),
            pci_bus_id=smi("pci.bus_id", bus_id(bus)),
            pci_device_id=missing("pci.device_id"),
            pci_subsystem_id=missing("pci.sub_device_id"),
            memory_total_bytes=memory_total
            or smi(
                "memory.total", spec.total_bytes, "bytes", "Converted from MiB; precision 1 MiB"
            ),
            compute_capability=missing("compute_cap"),
            mig_mode=missing("mig.mode.current"),
        )
    return GPUDeviceInfo(
        device_ref=f"nvml:{index}",
        nvml_index=index,
        vendor="NVIDIA",
        name=ok(spec.name, method="nvmlDeviceGetName"),
        uuid=ok(spec.uuid, method="nvmlDeviceGetUUID"),
        pci_bus_id=ok(bus_id(bus), method="nvmlDeviceGetPciInfo_v3"),
        pci_device_id=ok(spec.pci_device_id, method="nvmlDeviceGetPciInfo_v3"),
        pci_subsystem_id=ok("0x00000000", method="nvmlDeviceGetPciInfo_v3"),
        memory_total_bytes=memory_total
        or ok(spec.total_bytes, unit="bytes", method="nvmlDeviceGetMemoryInfo_v2"),
        compute_capability=ok(spec.compute_capability, method="nvmlDeviceGetCudaComputeCapability"),
        mig_mode=ok(mig_mode, method="nvmlDeviceGetMigMode"),
    )


def pci_device(
    bus: int,
    *,
    vendor_id: str = "0x10de",
    device_id: str = "0x2684",
    class_code: str = "0x030000",
    driver: str | None = "nvidia",
) -> PciObservation:
    address = f"0000:{bus:02x}:00.0"
    kernel_driver = (
        ok(driver, Source.OS, method="/sys/bus/pci/devices/*/driver")
        if driver
        else failed(
            CollectionStatus.NOT_DETECTED,
            "NO_DRIVER_BOUND",
            f"no kernel driver is bound to {address}",
            source=Source.OS,
            method="/sys/bus/pci/devices/*/driver",
        )
    )
    return PciObservation(
        address=address,
        vendor_id=vendor_id,
        device_id=device_id,
        class_code=class_code,
        is_nvidia=vendor_id == "0x10de",
        kernel_driver=kernel_driver,
    )


def pci_scan(*observations: PciObservation) -> Observed:
    return ok(tuple(observations), Source.OS, method="/sys/bus/pci/devices")


def pci_not_supported(environment: LinuxEnvironment | None) -> Observed:
    """The scan outcome the production adapter reports for an ineligible environment."""
    return failed(
        CollectionStatus.NOT_SUPPORTED,
        PCI_SCAN_UNSUPPORTED_CODES[environment],
        PCI_SCAN_UNSUPPORTED_REASONS[environment],
        source=Source.OS,
        method="/sys/bus/pci/devices",
    )


def nvml_inventory(devices: Sequence[GPUDeviceInfo], pci: Observed) -> GPUInventory:
    return GPUInventory(
        device_count=ok(len(devices), method="nvmlDeviceGetCount_v2"),
        inventory_source=Source.NVML,
        devices=tuple(devices),
        pci_observations=pci,
    )


def smi_inventory(devices: Sequence[GPUDeviceInfo], pci: Observed) -> GPUInventory:
    return GPUInventory(
        device_count=ok(len(devices), Source.NVIDIA_SMI, method="nvidia-smi --query-gpu=index"),
        inventory_source=Source.NVIDIA_SMI,
        devices=tuple(devices),
        pci_observations=pci,
    )


def no_inventory(device_count: Observed, pci: Observed) -> GPUInventory:
    return GPUInventory(
        device_count=device_count, inventory_source=None, devices=(), pci_observations=pci
    )


# --- driver -------------------------------------------------------------------------------


def nvml_failure(status: CollectionStatus, code: str, message: str) -> Observed:
    return failed(status, code, message, method="nvmlInit")


def driver_nvml(
    *,
    supported_cuda: str = "12.4",
    driver_version: str = "550.54.15",
    binding_distribution: str = "nvidia-ml-py",
    smi_path: str | None = "/usr/bin/nvidia-smi",
) -> DriverInfo:
    smi_found = smi_path is not None
    smi_missing = failed(
        CollectionStatus.NOT_DETECTED,
        "FileNotFoundError",
        "nvidia-smi not found",
        source=Source.NVIDIA_SMI,
        method="shutil.which",
    )
    return DriverInfo(
        nvml_binding=ok(NVML_BINDING_VERSION, Source.PLATFORM, method="importlib.metadata.version"),
        nvml_binding_distribution=ok(
            binding_distribution,
            Source.PLATFORM,
            method="importlib.metadata.packages_distributions",
        ),
        nvml_version=ok("12.550.54.15", method="nvmlSystemGetNVMLVersion"),
        driver_version=ok(driver_version, method="nvmlSystemGetDriverVersion"),
        driver_supported_cuda_version=ok(
            supported_cuda, method="nvmlSystemGetCudaDriverVersion", note=DRIVER_SUPPORTED_CUDA_NOTE
        ),
        cuda_driver_library=ok(supported_cuda, method="nvmlSystemGetCudaDriverVersion_v2"),
        nvidia_smi_path=ok(smi_path, Source.NVIDIA_SMI, method="shutil.which")
        if smi_found
        else smi_missing,
        nvidia_smi_driver_version=ok(driver_version, Source.NVIDIA_SMI, method="nvidia-smi header")
        if smi_found
        else smi_missing,
        nvidia_smi_cuda_version=ok(supported_cuda, Source.NVIDIA_SMI, method="nvidia-smi header")
        if smi_found
        else smi_missing,
    )


def driver_without_nvml(
    failure: Observed,
    *,
    binding_installed: bool = True,
    smi_path: str | None = None,
    smi_status: CollectionStatus = CollectionStatus.NOT_DETECTED,
    supported_cuda: str = "12.4",
    driver_version: str = "550.54.15",
) -> DriverInfo:
    """NVML unusable (no driver, missing binding, unsupported platform); `nvidia-smi` optional."""
    if binding_installed:
        binding = ok(NVML_BINDING_VERSION, Source.PLATFORM, method="importlib.metadata.version")
        distribution = ok(
            "nvidia-ml-py", Source.PLATFORM, method="importlib.metadata.packages_distributions"
        )
    else:
        binding = failed(
            CollectionStatus.NOT_INSTALLED,
            "ImportError",
            "No module named 'pynvml'",
            source=Source.PLATFORM,
            method="import pynvml",
        )
        distribution = binding
    if smi_path is not None:
        smi = ok(smi_path, Source.NVIDIA_SMI, method="shutil.which")
        driver = ok(
            driver_version, Source.NVIDIA_SMI, method="nvidia-smi --query-gpu=driver_version"
        )
        cuda = ok(
            supported_cuda,
            Source.NVIDIA_SMI,
            method="nvidia-smi header",
            note=DRIVER_SUPPORTED_CUDA_NOTE,
        )
        smi_driver = ok(driver_version, Source.NVIDIA_SMI, method="nvidia-smi header")
        smi_cuda = ok(supported_cuda, Source.NVIDIA_SMI, method="nvidia-smi header")
    else:
        code = (
            "FileNotFoundError"
            if smi_status is CollectionStatus.NOT_DETECTED
            else ("PLATFORM_NOT_SUPPORTED")
        )
        smi = failed(
            smi_status,
            code,
            "nvidia-smi not available",
            source=Source.NVIDIA_SMI,
            method="shutil.which",
        )
        driver = cuda = failure
        smi_driver = smi_cuda = smi
    return DriverInfo(
        nvml_binding=binding,
        nvml_binding_distribution=distribution,
        nvml_version=failure,
        driver_version=driver,
        driver_supported_cuda_version=cuda,
        cuda_driver_library=failure,
        nvidia_smi_path=smi,
        nvidia_smi_driver_version=smi_driver,
        nvidia_smi_cuda_version=smi_cuda,
    )


# --- CUDA environment -----------------------------------------------------------------------


def environment(values: Mapping[str, str | None]) -> dict[str, Observed]:
    return {
        name: ok(value, Source.ENVIRONMENT, method="os.environ")
        if value is not None
        else failed(
            CollectionStatus.NOT_DETECTED,
            "NOT_SET",
            f"{name} is not set",
            source=Source.ENVIRONMENT,
            method="os.environ",
        )
        for name, value in values.items()
    }


DEFAULT_ENV: dict[str, str | None] = {
    "CUDA_DEVICE_ORDER": None,
    "CUDA_VISIBLE_DEVICES": None,
    "PYTORCH_NVML_BASED_CUDA_CHECK": None,
}


def cuda_environment(
    *,
    toolkit_version: str | None = "12.4.131",
    root: str = "/usr/local/cuda-12.4",
    discovered_via: Sequence[ToolkitDiscovery] = (
        ToolkitDiscovery.CUDA_HOME,
        ToolkitDiscovery.NVCC_ON_PATH,
    ),
    env: Mapping[str, str | None] | None = None,
    platform_supported: bool = True,
    nvcc_name: str = "nvcc",
) -> CUDAEnvironmentInfo:
    """DEC-47 corrections: `environment` lists the whole allowlist (data model §5.1, as the
    environment collector reads it), and a root under `/usr/local` named `cuda[-X.Y]` is also
    `DEFAULT_LOCATION` (§4.6 consults the default locations and records every signal)."""
    values: dict[str, str | None] = dict.fromkeys(ENVIRONMENT_ALLOWLIST)
    values.update(DEFAULT_ENV)
    if toolkit_version is not None and ToolkitDiscovery.CUDA_HOME in discovered_via:
        values["CUDA_HOME"] = root
    values.update(env or {})
    if posixpath.dirname(root) == "/usr/local" and re.match(
        r"^cuda(-[0-9][0-9.]*)?$", posixpath.basename(root)
    ):
        discovered_via = (*discovered_via, ToolkitDiscovery.DEFAULT_LOCATION)
    nvcc = f"{root}/bin/{nvcc_name}"
    if not platform_supported:
        unsupported = failed(
            CollectionStatus.NOT_SUPPORTED,
            "PLATFORM_NOT_SUPPORTED",
            "CUDA Toolkit is not available on this platform",
            source=Source.CUDA_TOOLKIT,
            method="toolkit discovery",
        )
        return CUDAEnvironmentInfo(
            toolkits=unsupported,
            active_nvcc_path=unsupported,
            active_nvcc_version=unsupported,
            environment=environment(values),
        )
    if toolkit_version is None:
        none_found = failed(
            CollectionStatus.NOT_DETECTED,
            "NO_TOOLKIT_FOUND",
            "no CUDA Toolkit candidates",
            source=Source.CUDA_TOOLKIT,
            method="toolkit discovery",
        )
        nvcc_missing = failed(
            CollectionStatus.NOT_DETECTED,
            "FileNotFoundError",
            "nvcc not found",
            source=Source.CUDA_TOOLKIT,
            method="shutil.which",
        )
        return CUDAEnvironmentInfo(
            toolkits=none_found,
            active_nvcc_path=nvcc_missing,
            active_nvcc_version=nvcc_missing,
            environment=environment(values),
        )
    candidate = CudaToolkitCandidate(
        root_path=root,
        discovered_via=tuple(discovered_via),
        version=ok(toolkit_version, Source.CUDA_TOOLKIT, method="version.json:cuda.version"),
        nvcc_path=ok(nvcc, Source.CUDA_TOOLKIT, method="toolkit layout"),
    )
    return CUDAEnvironmentInfo(
        toolkits=ok((candidate,), Source.CUDA_TOOLKIT, method="toolkit discovery"),
        active_nvcc_path=ok(nvcc, Source.CUDA_TOOLKIT, method="shutil.which"),
        active_nvcc_version=ok(toolkit_version, Source.CUDA_TOOLKIT, method="nvcc --version"),
        environment=environment(values),
    )


# --- PyTorch --------------------------------------------------------------------------------


def probe_info(duration_ms: int | None = 1500) -> ProbeInfo:
    return ProbeInfo(
        mode=ProbeMode.SUBPROCESS, protocol_version=PROBE_PROTOCOL_VERSION, duration_ms=duration_ms
    )


def torch_device(
    torch_index: int,
    spec: GpuSpec,
    *,
    correlated_ref: str | None,
    correlated_by: str = "uuid",
    bus: int | None = None,
    uuid_supported: bool = True,
) -> TorchDeviceInfo:
    bus = torch_index + 1 if bus is None else bus
    uuid = (
        ok(spec.uuid.removeprefix("GPU-"), Source.PYTORCH, method="get_device_properties().uuid")
        if uuid_supported
        else failed(
            CollectionStatus.NOT_SUPPORTED,
            "ATTRIBUTE_MISSING",
            "uuid is not exposed by this PyTorch version",
            source=Source.PYTORCH,
            method="get_device_properties().uuid",
        )
    )
    correlated = (
        ok(correlated_ref, Source.INSPECTOR, method=correlated_by)
        if correlated_ref is not None
        else not_applicable("No NVML inventory to correlate with", source=Source.INSPECTOR)
    )
    return TorchDeviceInfo(
        torch_index=torch_index,
        name=ok(spec.name, Source.PYTORCH, method="get_device_name"),
        compute_capability=ok(
            spec.compute_capability, Source.PYTORCH, method="get_device_capability"
        ),
        total_memory_bytes=ok(
            spec.total_bytes, Source.PYTORCH, "bytes", method="get_device_properties().total_memory"
        ),
        multiprocessor_count=ok(
            128, Source.PYTORCH, method="get_device_properties().multi_processor_count"
        ),
        uuid=uuid,
        pci_bus_id=ok(f"0000:{bus:02x}:00.0", Source.PYTORCH, method="get_device_properties().pci"),
        correlated_device_ref=correlated,
    )


def _allocator_not_applicable() -> Observed:
    return not_applicable(ALLOCATOR_NOT_APPLICABLE_NOTE, source=Source.PYTORCH)


def pytorch_cuda(
    devices: Sequence[TorchDeviceInfo],
    *,
    version: str = "2.4.1+cu124",
    build_cuda: str = "12.4",
    cuda_available: bool = True,
    warnings: Sequence[str] = (),
) -> PyTorchInfo:
    return PyTorchInfo(
        status=CollectionStatus.COLLECTED,
        distribution_version=ok(
            version.split("+")[0], Source.PYTORCH, method="importlib.metadata.version('torch')"
        ),
        version=ok(version, Source.PYTORCH, method="torch.__version__"),
        cuda_build=ok(True, Source.PYTORCH, method="torch.version.cuda"),
        build_cuda_version=ok(
            build_cuda, Source.PYTORCH, method="torch.version.cuda", note=BUILD_CUDA_NOTE
        ),
        rocm_build=ok(False, Source.PYTORCH, method="torch.version.hip"),
        cudnn_version=ok("90100", Source.PYTORCH, method="torch.backends.cudnn.version()"),
        cuda_available=ok(cuda_available, Source.PYTORCH, method="torch.cuda.is_available()"),
        cuda_init_warnings=tuple(warnings),
        device_count=ok(len(devices), Source.PYTORCH, method="torch.cuda.device_count()"),
        devices=tuple(devices),
        allocator=_allocator_not_applicable(),
        probe=probe_info(),
    )


def pytorch_cpu_only(version: str = "2.4.1+cpu") -> PyTorchInfo:
    cpu = "CPU-only PyTorch build"
    return PyTorchInfo(
        status=CollectionStatus.COLLECTED,
        distribution_version=ok(
            version.split("+")[0], Source.PYTORCH, method="importlib.metadata.version('torch')"
        ),
        version=ok(version, Source.PYTORCH, method="torch.__version__"),
        cuda_build=ok(False, Source.PYTORCH, method="torch.version.cuda"),
        build_cuda_version=not_applicable(cpu, source=Source.PYTORCH),
        rocm_build=ok(False, Source.PYTORCH, method="torch.version.hip"),
        cudnn_version=not_applicable(cpu, source=Source.PYTORCH),
        cuda_available=ok(False, Source.PYTORCH, method="torch.cuda.is_available()"),
        cuda_init_warnings=(),
        device_count=ok(0, Source.PYTORCH, method="torch.cuda.device_count()"),
        devices=(),
        allocator=_allocator_not_applicable(),
        probe=probe_info(),
    )


def pytorch_not_installed() -> PyTorchInfo:
    error = ErrorDetail(code="PackageNotFoundError", message="No package metadata for 'torch'")
    return PyTorchInfo.unavailable(CollectionStatus.NOT_INSTALLED, error, probe=probe_info(None))


def pytorch_probe_failed(code: str, message: str, *, version: str = "2.4.1") -> PyTorchInfo:
    """Distribution present, probe failed (import error, timeout, crash)."""
    info = PyTorchInfo.unavailable(
        CollectionStatus.COLLECTION_FAILED,
        ErrorDetail(code=code, message=message),
        probe=probe_info(None),
    )
    return replace(
        info,
        distribution_version=ok(
            version, Source.PYTORCH, method="importlib.metadata.version('torch')"
        ),
    )


# --- runtime --------------------------------------------------------------------------------


def one_process(source: Source = Source.NVML) -> tuple[GpuProcess, ...]:
    return (
        GpuProcess(
            pid=4242,
            kind=ProcessKind.COMPUTE,
            used_memory_bytes=ok(
                2 * GIB, source, "bytes", method="nvmlDeviceGetComputeRunningProcesses_v3"
            ),
            name=ok("python", Source.OS, method="nvmlSystemGetProcessName"),
        ),
    )


def runtime_metrics(
    ref: str,
    total: int,
    free: int,
    *,
    source: Source = Source.NVML,
    sampled_at: datetime = FIXED_TIME,
    **overrides: Observed,
) -> RuntimeMetrics:
    """All metrics collected unless overridden (NVML or the `nvidia-smi` fallback query)."""
    if source is Source.NVIDIA_SMI:

        def m(field: str, value: Any, unit: str | None, note: str | None = None) -> Observed:
            return ok(
                value, Source.NVIDIA_SMI, unit, method=f"nvidia-smi --query-gpu={field}", note=note
            )

        def missing(unit: str | None) -> Observed:
            return failed(
                CollectionStatus.NOT_SUPPORTED,
                "FALLBACK_FIELD_UNAVAILABLE",
                _SMI_MISSING,
                source=Source.NVIDIA_SMI,
                method="nvidia-smi",
                unit=unit,
            )

        mib = "Converted from MiB; precision 1 MiB"
        fields: dict[str, Observed] = dict(
            gpu_utilization_percent=m("utilization.gpu", 3, "percent"),
            memory_bus_utilization_percent=m("utilization.memory", 1, "percent"),
            memory_total_bytes=m("memory.total", total, "bytes", mib),
            memory_used_bytes=m("memory.used", total - free, "bytes", mib),
            memory_free_bytes=m("memory.free", free, "bytes", mib),
            memory_reserved_bytes=missing("bytes"),
            temperature_gpu_celsius=m("temperature.gpu", 45, "celsius"),
            temperature_slowdown_threshold_celsius=missing("celsius"),
            power_draw_watts=m("power.draw", 31.5, "watts"),
            power_limit_watts=m("power.limit", 450.0, "watts"),
            processes=missing(None),
        )
    else:
        mem = "nvmlDeviceGetMemoryInfo_v2"
        fields = dict(
            gpu_utilization_percent=ok(3, unit="percent", method="nvmlDeviceGetUtilizationRates"),
            memory_bus_utilization_percent=ok(
                1, unit="percent", method="nvmlDeviceGetUtilizationRates"
            ),
            memory_total_bytes=ok(total, unit="bytes", method=mem),
            memory_used_bytes=ok(total - free, unit="bytes", method=mem),
            memory_free_bytes=ok(free, unit="bytes", method=mem),
            memory_reserved_bytes=ok(512 * MIB, unit="bytes", method=mem),
            temperature_gpu_celsius=ok(45, unit="celsius", method="nvmlDeviceGetTemperatureV"),
            temperature_slowdown_threshold_celsius=ok(
                90, unit="celsius", method="nvmlDeviceGetTemperatureThreshold"
            ),
            power_draw_watts=ok(31.5, unit="watts", method="nvmlDeviceGetPowerUsage"),
            power_limit_watts=ok(450.0, unit="watts", method="nvmlDeviceGetEnforcedPowerLimit"),
            processes=ok(one_process(), method="nvmlDeviceGetComputeRunningProcesses_v3"),
        )
    fields.update(overrides)
    return RuntimeMetrics(device_ref=ref, sampled_at=sampled_at, **fields)


# --- bookkeeping --------------------------------------------------------------------------


def collector_runs(**statuses: CollectorRunStatus) -> tuple[CollectorRun, ...]:
    """Every collector in execution order; COMPLETED unless overridden. Durations are fixed."""
    unknown = set(statuses) - set(COLLECTOR_ORDER)
    if unknown:
        raise ValueError(f"unknown collectors {sorted(unknown)}")
    return tuple(
        CollectorRun(
            collector=name,
            status=statuses.get(name, CollectorRunStatus.COMPLETED),
            duration_ms=5 * (i + 1),
        )
        for i, name in enumerate(COLLECTOR_ORDER)
    )


def unsupported(
    capability: str,
    reason: str,
    *,
    device_ref: str | None = None,
    status: CollectionStatus = CollectionStatus.NOT_SUPPORTED,
) -> UnsupportedCapability:
    return UnsupportedCapability(
        capability=capability, status=status, device_ref=device_ref, reason=reason
    )


def collection_error(
    collector: str,
    code: str,
    message: str,
    *,
    device_ref: str | None = None,
    status: CollectionStatus = CollectionStatus.COLLECTION_FAILED,
    report_path: str | None = None,
) -> CollectionError:
    return CollectionError(
        collector=collector,
        device_ref=device_ref,
        status=status,
        error_code=code,
        message=message,
        report_path=report_path,
    )


# --- general-purpose builders (exposed through the `build` pytest fixture) -----------------


class Builders:
    """Free-form model builders for model, serialisation and schema tests."""

    @staticmethod
    def system(profile: PlatformProfile = PlatformProfile.LINUX) -> SystemInfo:
        return system_info(profile, processor_detected=False)

    @staticmethod
    def device(
        index: int, name: str = "Synthetic GPU A (fixture)", total: int = 24 * GIB
    ) -> GPUDeviceInfo:
        spec = GpuSpec(name, total, f"GPU-0000000{index}-0000-0000-0000-000000000000")
        return gpu_device(index, spec)

    @staticmethod
    def inventory(devices: tuple[GPUDeviceInfo, ...]) -> GPUInventory:
        return nvml_inventory(devices, pci_scan(*(pci_device(d.nvml_index + 1) for d in devices)))

    @staticmethod
    def driver() -> DriverInfo:
        return driver_nvml(supported_cuda="13.1", driver_version="615.71")

    @staticmethod
    def cuda_environment() -> CUDAEnvironmentInfo:
        return cuda_environment(
            env={
                "CUDA_VISIBLE_DEVICES": "",
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "CUDA_DEVICE_ORDER": None,
            }
        )

    @staticmethod
    def pytorch(device_count: int = 2) -> PyTorchInfo:
        devices = [
            torch_device(
                i,
                GPU_A,
                correlated_ref=f"nvml:{device_count - 1 - i}",
                correlated_by="pci_bus_id",
                uuid_supported=False,
            )
            for i in range(device_count)
        ]
        return pytorch_cuda(devices, version="2.15.0+cu130", build_cuda="13.0")

    @staticmethod
    def runtime(ref: str, free: int = 20 * GIB, *, with_processes: bool = False) -> RuntimeMetrics:
        processes = (
            ok(
                (
                    GpuProcess(
                        pid=4242,
                        kind=ProcessKind.COMPUTE,
                        used_memory_bytes=ok(2 * GIB, unit="bytes"),
                        name=ok("python", Source.OS),
                    ),
                    GpuProcess(
                        pid=17,
                        kind=ProcessKind.GRAPHICS,
                        used_memory_bytes=failed(
                            CollectionStatus.NOT_SUPPORTED, "VALUE_NOT_AVAILABLE", unit="bytes"
                        ),
                        name=failed(CollectionStatus.PERMISSION_DENIED, "NoPermission"),
                    ),
                )
            )
            if with_processes
            else skipped("--no-processes")
        )
        return runtime_metrics(
            ref,
            24 * GIB,
            free,
            gpu_utilization_percent=ok(0, unit="percent"),
            memory_bus_utilization_percent=failed(
                CollectionStatus.NOT_SUPPORTED, "NVML_ERROR_NOT_SUPPORTED", unit="percent"
            ),
            memory_reserved_bytes=ok(0, unit="bytes"),
            processes=processes,
        )

    @staticmethod
    def snapshot(device_count: int = 2) -> InspectionSnapshot:
        devices = tuple(Builders.device(i) for i in range(device_count))
        return InspectionSnapshot(
            system=Builders.system(),
            gpu=Builders.inventory(devices),
            driver=Builders.driver(),
            cuda_environment=Builders.cuda_environment(),
            frameworks=FrameworkInfo(pytorch=Builders.pytorch(device_count)),
            runtime=tuple(
                Builders.runtime(d.device_ref, with_processes=(i == 0))
                for i, d in enumerate(devices)
            ),
            collection_errors=(),
            unsupported_capabilities=(),
            collector_runs=(
                CollectorRun(
                    collector="system", status=CollectorRunStatus.COMPLETED, duration_ms=3
                ),
                CollectorRun(
                    collector="runtime", status=CollectorRunStatus.PARTIAL, duration_ms=40
                ),
            ),
        )

    @staticmethod
    def finding(check_id: str = "D04", device_ref: str | None = None) -> DiagnosticFinding:
        return DiagnosticFinding(
            check_id=check_id,
            device_ref=device_ref,
            status=DiagnosticStatus.PASS,
            severity=Severity.NONE,
            title="NVIDIA GPU detected",
            explanation="NVML reported devices.",
            evidence=(
                Evidence(
                    kind=EvidenceKind.OBSERVATION,
                    statement="device_count is 2",
                    source=Source.NVML,
                    report_path="/gpu/device_count/value",
                    value=2,
                ),
            ),
            detection_sources=(Source.NVML,),
            recommendations=(),
            confidence=Confidence.HIGH,
            limitations=(),
        )

    @staticmethod
    def warning_finding() -> DiagnosticFinding:
        return DiagnosticFinding(
            check_id="D07",
            device_ref="nvml:1",
            status=DiagnosticStatus.WARNING,
            severity=Severity.MEDIUM,
            title="Temperature at or above the slowdown threshold",
            explanation="Measured 92 °C at sampling time.",
            evidence=(
                Evidence(
                    kind=EvidenceKind.OBSERVATION,
                    statement="temperature 92 °C",
                    source=Source.NVML,
                    report_path="/runtime/1/temperature_gpu_celsius",
                    value={"celsius": 92, "samples": [92]},
                ),
                Evidence(
                    kind=EvidenceKind.INFERENCE,
                    statement="cooling may be insufficient",
                    source=None,
                    report_path=None,
                    value=None,
                ),
            ),
            detection_sources=(Source.NVML, Source.INSPECTOR, Source.NVML),
            recommendations=(
                Recommendation(
                    action="Check cooling.",
                    rationale="Sustained heat matters.",
                    reference_url="https://docs.nvidia.com/deploy/nvml-api/",
                ),
            ),
            confidence=Confidence.MEDIUM,
            limitations=("A single sample cannot establish throttling.",),
        )

    @staticmethod
    def readiness(required_vram_gib: float, free_by_ref: dict[str, int]) -> ReadinessResult:
        required = required_bytes_for(required_vram_gib)
        devices = []
        for ref, free in free_by_ref.items():
            ready = free >= required
            devices.append(
                DeviceReadiness(
                    device_ref=ref,
                    status=ReadinessStatus.READY if ready else ReadinessStatus.NOT_READY,
                    reason=ReadinessReason.SUFFICIENT_FREE_MEMORY
                    if ready
                    else ReadinessReason.INSUFFICIENT_FREE_MEMORY,
                    free_bytes=ok(free, unit="bytes"),
                    total_bytes=ok(24 * GIB, unit="bytes"),
                    margin_bytes=free - required,
                    explanation="synthetic",
                )
            )
        any_ready = any(d.status is ReadinessStatus.READY for d in devices)
        return ReadinessResult(
            required_vram_gib=required_vram_gib,
            required_bytes=required,
            criterion=READINESS_CRITERION,
            overall_status=ReadinessStatus.READY if any_ready else ReadinessStatus.NOT_READY,
            overall_reason=ReadinessReason.AT_LEAST_ONE_DEVICE_READY
            if any_ready
            else ReadinessReason.NO_DEVICE_READY,
            devices=tuple(devices),
            disclaimer=READINESS_DISCLAIMER,
            limitations=("Snapshot taken at 2026-01-01T00:00:00Z",),
        )

    @staticmethod
    def report(
        *, required_vram_gib: float | None = None, device_count: int = 2, **overrides: Any
    ) -> InspectionReport:
        snapshot = Builders.snapshot(device_count)
        readiness = (
            Builders.readiness(required_vram_gib, {d: 20 * GIB for d in snapshot.gpu.device_refs})
            if required_vram_gib is not None
            else None
        )
        kwargs: dict[str, Any] = dict(
            generated_at=FIXED_TIME,
            duration_ms=412,
            data_origin=DataOrigin.FIXTURE,
            options=InspectionOptions(required_vram_gib=required_vram_gib),
            diagnostics=(Builders.finding(),),
            readiness=readiness,
        )
        kwargs.update(overrides)
        return InspectionReport.from_snapshot(snapshot, **kwargs)

    @staticmethod
    def no_gpu_report() -> InspectionReport:
        """Shape of a WSL2 machine without an NVIDIA driver (similar to OBS-01)."""
        missing = nvml_failure(
            CollectionStatus.NOT_DETECTED,
            "NVML_ERROR_LIBRARY_NOT_FOUND",
            "NVML Shared Library Not Found",
        )
        snapshot = InspectionSnapshot(
            system=system_info(PlatformProfile.WSL2, processor_detected=False),
            gpu=no_inventory(missing, pci_not_supported(LinuxEnvironment.WSL2)),
            driver=DriverInfo.unavailable(missing.status, missing.error),
            cuda_environment=CUDAEnvironmentInfo.unavailable(
                CollectionStatus.NOT_DETECTED, missing.error
            ),
            frameworks=FrameworkInfo(pytorch=pytorch_not_installed()),
            runtime=(),
            collection_errors=(),
            unsupported_capabilities=unsupported_capabilities_for(
                LinuxEnvironment.WSL2, PlatformProfile.WSL2
            ),
            collector_runs=(
                CollectorRun(
                    collector="system", status=CollectorRunStatus.COMPLETED, duration_ms=12
                ),
            ),
        )
        return InspectionReport.from_snapshot(
            snapshot,
            generated_at=FIXED_TIME,
            duration_ms=412,
            data_origin=DataOrigin.FIXTURE,
            options=InspectionOptions(),
            limitations=(SNAPSHOT_LIMITATION,),
        )

    @staticmethod
    def partial_failure_report() -> InspectionReport:
        """FX10 shape: CUDA environment collector crashed; device nvml:1 lost."""
        snapshot = Builders.snapshot(2)
        lost = ErrorDetail(code="NVML_ERROR_GPU_IS_LOST", message="GPU is lost")
        crashed = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: synthetic")
        partial = replace(
            snapshot,
            cuda_environment=CUDAEnvironmentInfo.unavailable(
                CollectionStatus.COLLECTION_FAILED, crashed
            ),
            runtime=(
                snapshot.runtime[0],
                RuntimeMetrics.unavailable(
                    "nvml:1", FIXED_TIME, CollectionStatus.COLLECTION_FAILED, lost
                ),
            ),
            collection_errors=(
                collection_error(
                    "runtime",
                    lost.code,
                    lost.message,
                    device_ref="nvml:1",
                    report_path="/runtime/1",
                ),
                collection_error(
                    "cuda_environment",
                    crashed.code,
                    crashed.message,
                    report_path="/cuda_environment",
                ),
            ),
        )
        return InspectionReport.from_snapshot(
            partial,
            generated_at=FIXED_TIME,
            duration_ms=500,
            data_origin=DataOrigin.FIXTURE,
            options=InspectionOptions(),
        )
