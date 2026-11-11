"""Unsupported-capability policy for PCI discovery and NVIDIA availability (data model §3.4).

Pure policy, not a collector: it says which of `GPU.PCI_SCAN` and `NVIDIA.ALL` an environment
can't provide, and why. Emitting these records into a report is the job of the PCI collector
(Stage 6.3) and report assembly (IMP-P01); nothing emits them at runtime yet.
"""

from __future__ import annotations

from gpu_system_inspector.adapters import AdapterUsageError
from gpu_system_inspector.adapters.pci_sysfs import PCI_SCAN_UNSUPPORTED_REASONS, pci_scan_eligible
from gpu_system_inspector.models import (
    CollectionStatus,
    LinuxEnvironment,
    PlatformProfile,
    UnsupportedCapability,
)

GPU_PCI_SCAN = "GPU.PCI_SCAN"
NVIDIA_ALL = "NVIDIA.ALL"
NVIDIA_ALL_MACOS_REASON = (
    "NVIDIA CUDA GPUs are not supported on macOS (product_specification.md §4)."
)


def _check_consistent(environment: LinuxEnvironment | None, profile: PlatformProfile) -> None:
    """Same agreement rule as `SystemInfo` (data model §4.1)."""
    if environment is None:
        if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
            raise AdapterUsageError(f"a {profile} host needs a LinuxEnvironment")
        return
    if not isinstance(environment, LinuxEnvironment):
        raise AdapterUsageError(
            f"environment must be a LinuxEnvironment or None, got {environment!r}"
        )
    expected = (
        PlatformProfile.WSL2 if environment is LinuxEnvironment.WSL2 else PlatformProfile.LINUX
    )
    if profile is not expected:
        raise AdapterUsageError(
            f"linux_environment {environment} requires platform_profile {expected}"
        )


def unsupported_capabilities_for(
    environment: LinuxEnvironment | None, profile: PlatformProfile
) -> tuple[UnsupportedCapability, ...]:
    """`GPU.PCI_SCAN` is supported only for `NATIVE_LINUX`; the profile never changes that and
    only adds `NVIDIA.ALL` on macOS. Records are `NOT_SUPPORTED` and host-wide."""
    _check_consistent(environment, profile)
    records = []
    if not pci_scan_eligible(environment):
        records.append(
            UnsupportedCapability(
                capability=GPU_PCI_SCAN,
                status=CollectionStatus.NOT_SUPPORTED,
                device_ref=None,
                reason=PCI_SCAN_UNSUPPORTED_REASONS[environment],
            )
        )
    if profile is PlatformProfile.MACOS:
        records.append(
            UnsupportedCapability(
                capability=NVIDIA_ALL,
                status=CollectionStatus.NOT_SUPPORTED,
                device_ref=None,
                reason=NVIDIA_ALL_MACOS_REASON,
            )
        )
    return tuple(records)
