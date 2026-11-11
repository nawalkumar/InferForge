"""Adapter interfaces, normalised outcomes and the adapter set (docs/architecture.md §4.5).

Production adapters so far: `SystemClock` (Stage 4), `OsPlatformAdapter` (Stage 5,
read-only OS facts through the standard library) and `PciSysfsAdapter` (Stage 6.1, read-only
PCI sysfs, native Linux only), `SubprocessCommandRunner` (IMP-A05), `PynvmlAdapter`
(IMP-A01), `NvidiaSmiCli` (IMP-A02), `CudaToolkitLocator` (IMP-A06, DEC-47) and the PyTorch
probes `SubprocessTorchProbe` / `InProcessTorchProbe` (IMP-A07, DEC-48). The not-implemented
stand-ins (`unimplemented.py`, DEC-44) are no longer used in production. Importing this
package loads neither NVML nor PyTorch, imports no `subprocess` and touches no file: those
happen on first use.
"""

from gpu_system_inspector.adapters.adapter_set import AdapterSet
from gpu_system_inspector.adapters.clock import SystemClock
from gpu_system_inspector.adapters.commands import SubprocessCommandRunner
from gpu_system_inspector.adapters.nvidia_smi import NvidiaSmiCli, parse_query_output
from gpu_system_inspector.adapters.nvml import PynvmlAdapter, PynvmlDevice, PynvmlSession
from gpu_system_inspector.adapters.outcomes import (
    MAX_COMMAND_OUTPUT_CHARS,
    AdapterUsageError,
    CommandResult,
    exception_outcome,
    nvml_error_status,
    require_success,
)
from gpu_system_inspector.adapters.pci_sysfs import (
    PCI_DEVICES_ROOT,
    PciSysfsAdapter,
    StdlibSysfsReader,
    SysfsReader,
)
from gpu_system_inspector.adapters.platform_os import HostApi, OsPlatformAdapter, StdlibHostApi
from gpu_system_inspector.adapters.probe_protocol import (
    TORCH_PROBE_MODULE,
    TorchAllocatorSample,
    TorchProbeDevice,
    TorchProbeReport,
    decode_probe_output,
    decode_probe_payload,
    torch_probe_argv,
)
from gpu_system_inspector.adapters.protocols import (
    NVIDIA_SMI_EXECUTABLE,
    NVIDIA_SMI_QUERY_FIELDS,
    NVIDIA_SMI_WSL_PATH,
    NVML_DEVICE_FUNCTIONS,
    NVML_READ_ONLY_FUNCTIONS,
    NVML_SESSION_FUNCTIONS,
    TOOLKIT_HINT_KINDS,
    Clock,
    CommandRunner,
    NvidiaSmiAdapter,
    NvmlAdapter,
    NvmlDevice,
    NvmlMemoryInfo,
    NvmlPciInfo,
    NvmlSession,
    NvmlUtilization,
    PciDeviceFailure,
    PciFailureKind,
    PciScan,
    PciScanner,
    PlatformAdapter,
    SmiGpuRow,
    SmiVersion,
    ToolkitDiscoveryResult,
    ToolkitLocator,
    ToolkitProblem,
    TorchProbe,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
)
from gpu_system_inspector.adapters.toolkit import (
    CudaToolkitLocator,
    StdlibToolkitFiles,
    ToolkitFiles,
)
from gpu_system_inspector.adapters.torch_probe import InProcessTorchProbe, SubprocessTorchProbe
from gpu_system_inspector.adapters.unimplemented import (
    UnimplementedCommandRunner,
    UnimplementedNvidiaSmi,
    UnimplementedNvmlAdapter,
    UnimplementedNvmlSession,
    UnimplementedTorchProbe,
)

__all__ = [
    "MAX_COMMAND_OUTPUT_CHARS",
    "NVIDIA_SMI_EXECUTABLE",
    "NVIDIA_SMI_QUERY_FIELDS",
    "NVIDIA_SMI_WSL_PATH",
    "NVML_DEVICE_FUNCTIONS",
    "NVML_READ_ONLY_FUNCTIONS",
    "NVML_SESSION_FUNCTIONS",
    "TOOLKIT_HINT_KINDS",
    "TORCH_PROBE_MODULE",
    "AdapterSet",
    "AdapterUsageError",
    "Clock",
    "CommandResult",
    "CommandRunner",
    "CudaToolkitLocator",
    "HostApi",
    "InProcessTorchProbe",
    "NvidiaSmiAdapter",
    "NvidiaSmiCli",
    "NvmlAdapter",
    "NvmlDevice",
    "NvmlMemoryInfo",
    "NvmlPciInfo",
    "NvmlSession",
    "NvmlUtilization",
    "OsPlatformAdapter",
    "PCI_DEVICES_ROOT",
    "PciDeviceFailure",
    "PciFailureKind",
    "PciScan",
    "PciScanner",
    "PciSysfsAdapter",
    "PlatformAdapter",
    "PynvmlAdapter",
    "PynvmlDevice",
    "PynvmlSession",
    "SmiGpuRow",
    "SmiVersion",
    "StdlibHostApi",
    "StdlibSysfsReader",
    "StdlibToolkitFiles",
    "SubprocessCommandRunner",
    "SubprocessTorchProbe",
    "SysfsReader",
    "SystemClock",
    "ToolkitDiscoveryResult",
    "ToolkitFiles",
    "ToolkitLocator",
    "ToolkitProblem",
    "TorchAllocatorSample",
    "TorchProbe",
    "TorchProbeDevice",
    "TorchProbeReport",
    "UnimplementedCommandRunner",
    "UnimplementedNvidiaSmi",
    "UnimplementedNvmlAdapter",
    "UnimplementedNvmlSession",
    "UnimplementedTorchProbe",
    "decode_probe_output",
    "decode_probe_payload",
    "exception_outcome",
    "nvidia_smi_query_argv",
    "nvidia_smi_version_argv",
    "nvml_error_status",
    "parse_query_output",
    "require_success",
    "torch_probe_argv",
]
