"""Snapshot facts shared by several rules (diagnostics spec §3, KI-32, KI-33).

Absence is only concluded from a *complete* PCI scan (KI-32: a `PARTIAL` scan omitted
devices) that was `COLLECTED` (KI-33: `NOT_DETECTED`/`NOT_SUPPORTED` scans prove nothing).
"""

from __future__ import annotations

from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    GPUDeviceInfo,
    InspectionSnapshot,
    Observed,
    PciObservation,
    PlatformProfile,
    Source,
)

PCI_COLLECTOR = "pci"
GPU_IS_LOST = "NVML_ERROR_GPU_IS_LOST"
LIBRARY_NOT_FOUND = "NVML_ERROR_LIBRARY_NOT_FOUND"
NVIDIA_KERNEL_DRIVER = "nvidia"
_ABSENT = (CollectionStatus.NOT_DETECTED, CollectionStatus.NOT_INSTALLED)


def is_macos(s: InspectionSnapshot) -> bool:
    return s.system.platform_profile is PlatformProfile.MACOS


def nvml_device_count(s: InspectionSnapshot) -> int | None:
    """NVML's device count, when NVML initialised and counted devices."""
    if s.driver.nvml_version.is_collected and s.gpu.device_count.is_collected:
        if s.gpu.inventory_source in (Source.NVML, None):
            return s.gpu.device_count.value
    return None


def fallback_inventory(s: InspectionSnapshot) -> bool:
    return s.gpu.inventory_source is Source.NVIDIA_SMI and bool(s.gpu.devices)


def nvidia_pci(s: InspectionSnapshot) -> tuple[PciObservation, ...]:
    pci = s.gpu.pci_observations
    return tuple(o for o in pci.value if o.is_nvidia) if pci.value is not None else ()


def pci_scan_complete(s: InspectionSnapshot) -> bool:
    if not s.gpu.pci_observations.is_collected:
        return False
    runs = [r for r in s.collector_runs if r.collector == PCI_COLLECTOR]
    return all(r.status is CollectorRunStatus.COMPLETED for r in runs)


def nvidia_observed(s: InspectionSnapshot) -> bool:
    count = nvml_device_count(s)
    return (count is not None and count >= 1) or fallback_inventory(s) or bool(nvidia_pci(s))


def nvidia_observed_absent(s: InspectionSnapshot) -> bool:
    if is_macos(s) or nvml_device_count(s) == 0:
        return True
    return (
        pci_scan_complete(s)
        and not nvidia_pci(s)
        and s.driver.nvml_version.status is CollectionStatus.NOT_DETECTED
    )


def nvml_absent(s: InspectionSnapshot) -> bool:
    return s.driver.nvml_version.status in _ABSENT


def smi_absent(s: InspectionSnapshot) -> bool:
    return s.driver.nvidia_smi_path.status in (*_ABSENT, CollectionStatus.NOT_SUPPORTED)


def error_code(observed: Observed[object]) -> str | None:
    return observed.error.code if observed.error is not None else None


def env(s: InspectionSnapshot, name: str) -> Observed[str] | None:
    return s.cuda_environment.environment.get(name)


def env_path(name: str) -> str:
    return f"/cuda_environment/environment/{name}"


def device_path(s: InspectionSnapshot, ref: str, field: str) -> str:
    index = next(i for i, d in enumerate(s.gpu.devices) if d.device_ref == ref)
    return f"/gpu/devices/{index}/{field}"


def visible_entry_matches(entry: str, devices: tuple[GPUDeviceInfo, ...]) -> bool:
    """Best-effort match of one `CUDA_VISIBLE_DEVICES` entry (index, UUID prefix, MIG ID)."""
    entry = entry.strip()
    if entry.isdigit():
        return int(entry) < len(devices)
    if entry.upper().startswith("MIG-"):
        return any(d.mig_mode.value == "ENABLED" for d in devices)
    if entry.upper().startswith("GPU-"):
        return any(
            d.uuid.value is not None and d.uuid.value.lower().startswith(entry.lower())
            for d in devices
        )
    return False
