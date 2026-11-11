"""Collectors turn adapter outcomes into report sections (docs/architecture.md §4.2–§4.4).

Implemented so far: the system collector and allowlisted environment collection (Stage 5),
the PCI collector (Stage 6.3, DEC-43), the GPU hardware collector (IMP-K02, DEC-44), and
the driver, `nvidia-smi` and runtime metrics collectors (IMP-K03/K04, DEC-46) and the CUDA
environment collector (IMP-K05, DEC-47), the PyTorch collector (IMP-K06, DEC-48) and device
correlation (IMP-K07, DEC-49). All of them
satisfy `Collector` (`base.py`). Collectors never touch the OS, NVML, processes or PyTorch
directly, and never open or close the NVML session.
"""

from gpu_system_inspector.collectors.base import Collector
from gpu_system_inspector.collectors.capabilities import (
    GPU_PCI_SCAN,
    NVIDIA_ALL,
    unsupported_capabilities_for,
)
from gpu_system_inspector.collectors.correlation import (
    CORRELATION_REPORT_PATH,
    CorrelationCollector,
    normalise_pci_bus_id,
    normalise_uuid,
)
from gpu_system_inspector.collectors.cuda_env import (
    CUDA_ENVIRONMENT_REPORT_PATH,
    CUDA_TOOLKIT,
    CudaEnvironmentCollector,
)
from gpu_system_inspector.collectors.driver import (
    DRIVER_REPORT_PATH,
    NVIDIA_SMI_REPORT_PATH,
    DriverCollector,
    NvidiaSmiCollector,
)
from gpu_system_inspector.collectors.environment import collect_environment
from gpu_system_inspector.collectors.framework import PYTORCH_REPORT_PATH, FrameworkCollector
from gpu_system_inspector.collectors.gpu_hardware import (
    FALLBACK_FIELD_UNAVAILABLE,
    GPU_REPORT_PATH,
    GpuHardwareCollector,
)
from gpu_system_inspector.collectors.pci import PCI_REPORT_PATH, PciCollector
from gpu_system_inspector.collectors.runtime import (
    RUNTIME_GPU_UTILIZATION,
    RUNTIME_PROCESS_MEMORY,
    RUNTIME_PROCESSES,
    RUNTIME_REPORT_PATH,
    RuntimeCollector,
)
from gpu_system_inspector.collectors.system import SYSTEM_CAPABILITIES, SystemCollector

__all__ = [
    "CORRELATION_REPORT_PATH",
    "CUDA_ENVIRONMENT_REPORT_PATH",
    "CUDA_TOOLKIT",
    "DRIVER_REPORT_PATH",
    "FALLBACK_FIELD_UNAVAILABLE",
    "GPU_PCI_SCAN",
    "GPU_REPORT_PATH",
    "NVIDIA_ALL",
    "NVIDIA_SMI_REPORT_PATH",
    "PCI_REPORT_PATH",
    "PYTORCH_REPORT_PATH",
    "RUNTIME_GPU_UTILIZATION",
    "RUNTIME_PROCESSES",
    "RUNTIME_PROCESS_MEMORY",
    "RUNTIME_REPORT_PATH",
    "SYSTEM_CAPABILITIES",
    "Collector",
    "CorrelationCollector",
    "CudaEnvironmentCollector",
    "DriverCollector",
    "FrameworkCollector",
    "GpuHardwareCollector",
    "NvidiaSmiCollector",
    "PciCollector",
    "RuntimeCollector",
    "SystemCollector",
    "collect_environment",
    "normalise_pci_bus_id",
    "normalise_uuid",
    "unsupported_capabilities_for",
]
