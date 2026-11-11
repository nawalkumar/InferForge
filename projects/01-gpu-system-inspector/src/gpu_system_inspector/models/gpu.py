"""GPU inventory models (docs/data_model_specification.md §4.2)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    constrained,
    device_ref_index,
    device_ref_key,
    observed,
    set_field,
    unavailable,
    validate_model,
)
from gpu_system_inspector.models.enums import CollectionStatus, Source
from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

NVIDIA_VENDOR = "NVIDIA"
NVIDIA_PCI_VENDOR_ID = "0x10de"
MIG_MODES = ("ENABLED", "DISABLED")

_INVENTORY_PREFIX = {Source.NVML: "nvml", Source.NVIDIA_SMI: "smi"}


@dataclass(frozen=True, slots=True, kw_only=True)
class PciObservation:
    address: str = constrained(minLength=1)
    vendor_id: str = constrained(pattern=r"^0x[0-9a-f]{4}$")
    device_id: str = constrained(pattern=r"^0x[0-9a-f]{4}$")
    class_code: str = constrained(pattern=r"^0x[0-9a-f]{6}$")
    is_nvidia: bool
    kernel_driver: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        if self.is_nvidia != (self.vendor_id == NVIDIA_PCI_VENDOR_ID):
            raise InvalidModelError(
                f"PciObservation.is_nvidia must equal vendor_id == {NVIDIA_PCI_VENDOR_ID!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class GPUDeviceInfo:
    device_ref: str = constrained(pattern=DEVICE_REF_REGEX)
    nvml_index: int = constrained(minimum=0)
    vendor: str = constrained(const=NVIDIA_VENDOR)
    name: Observed[str] = observed()
    uuid: Observed[str] = observed()
    pci_bus_id: Observed[str] = observed()
    pci_device_id: Observed[str] = observed()
    pci_subsystem_id: Observed[str] = observed()
    memory_total_bytes: Observed[int] = observed("bytes")
    compute_capability: Observed[str] = observed()
    mig_mode: Observed[str] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        if device_ref_index(self.device_ref) != self.nvml_index:
            raise InvalidModelError(
                f"GPUDeviceInfo.device_ref {self.device_ref!r} doesn't match "
                f"nvml_index {self.nvml_index}"
            )
        if self.mig_mode.is_collected and self.mig_mode.value not in MIG_MODES:
            raise InvalidModelError(f"GPUDeviceInfo.mig_mode must be one of {MIG_MODES}")

    @classmethod
    def unavailable(
        cls, device_ref: str, status: CollectionStatus, error: ErrorDetail | None
    ) -> GPUDeviceInfo:
        return unavailable(
            cls,
            status,
            error,
            device_ref=device_ref,
            nvml_index=device_ref_index(device_ref),
            vendor=NVIDIA_VENDOR,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GPUInventory:
    device_count: Observed[int] = observed()
    inventory_source: Source | None
    devices: tuple[GPUDeviceInfo, ...]
    pci_observations: Observed[tuple[PciObservation, ...]] = observed()

    def __post_init__(self) -> None:
        validate_model(self)
        devices = tuple(sorted(self.devices, key=lambda d: device_ref_key(d.device_ref)))
        set_field(self, "devices", devices)
        refs = [d.device_ref for d in devices]
        if len(set(refs)) != len(refs):
            raise InvalidModelError("GPUInventory.devices must have unique device_ref values")
        if self.inventory_source is not None and self.inventory_source not in _INVENTORY_PREFIX:
            raise InvalidModelError(
                "GPUInventory.inventory_source must be NVML, NVIDIA_SMI or None"
            )
        if devices:
            if self.inventory_source is None:
                raise InvalidModelError("GPUInventory with devices must name its inventory_source")
            prefix = _INVENTORY_PREFIX[self.inventory_source]
            if any(device_ref_key(ref)[0] != prefix for ref in refs):
                raise InvalidModelError(
                    f"device_ref values must use the {prefix!r} prefix for {self.inventory_source}"
                )
        if self.device_count.is_collected:
            if self.inventory_source is None:
                raise InvalidModelError("a collected device_count requires an inventory_source")
            if self.device_count.value != len(devices):
                raise InvalidModelError(
                    f"device_count {self.device_count.value} doesn't match {len(devices)} devices"
                )
        elif devices:
            raise InvalidModelError("devices can only be listed when device_count is collected")
        if self.pci_observations.is_collected:
            observations = self.pci_observations.value or ()
            ordered = tuple(sorted(observations, key=lambda o: o.address))
            if len({o.address for o in ordered}) != len(ordered):
                raise InvalidModelError("pci_observations must have unique addresses")
            set_field(self, "pci_observations", replace(self.pci_observations, value=ordered))

    @property
    def device_refs(self) -> tuple[str, ...]:
        return tuple(d.device_ref for d in self.devices)

    @classmethod
    def unavailable(cls, status: CollectionStatus, error: ErrorDetail | None) -> GPUInventory:
        return unavailable(cls, status, error, inventory_source=None, devices=())
