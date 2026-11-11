"""Fake `nvidia-smi` adapter implementing `NvidiaSmiAdapter`. Never runs a process."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from gpu_system_inspector.adapters import NVIDIA_SMI_QUERY_FIELDS, SmiGpuRow, SmiVersion
from gpu_system_inspector.models import (
    DataOrigin,
    DriverInfo,
    GPUInventory,
    InspectionSnapshot,
    Observed,
    RuntimeMetrics,
    Source,
)
from gsi_testkit.builders import MIB, fx

# Query field → (model attribute, section). `index` is the row's own key.
_ROW_FIELDS = {
    "uuid": ("uuid", "device"),
    "name": ("name", "device"),
    "pci.bus_id": ("pci_bus_id", "device"),
    "memory.total": ("memory_total_bytes", "runtime"),
    "memory.used": ("memory_used_bytes", "runtime"),
    "memory.free": ("memory_free_bytes", "runtime"),
    "temperature.gpu": ("temperature_gpu_celsius", "runtime"),
    "utilization.gpu": ("gpu_utilization_percent", "runtime"),
    "utilization.memory": ("memory_bus_utilization_percent", "runtime"),
    "power.draw": ("power_draw_watts", "runtime"),
    "power.limit": ("power_limit_watts", "runtime"),
    "driver_version": ("driver_version", "driver"),
}
assert tuple(_ROW_FIELDS) == NVIDIA_SMI_QUERY_FIELDS[1:]


def _as_smi(obs: Observed, query_field: str) -> Observed:
    """Re-express a value as the fallback query reports it (MiB precision for memory)."""
    method = fx(f"nvidia-smi --query-gpu={query_field}")
    if not obs.is_collected:
        return Observed(
            value=None,
            status=obs.status,
            source=Source.NVIDIA_SMI,
            method=method,
            unit=obs.unit,
            error=obs.error,
            note=obs.note,
        )
    value = obs.value
    note = None
    if obs.unit == "bytes":
        value = (value // MIB) * MIB
        note = "Converted from MiB; precision 1 MiB"
    return Observed.collected(
        value, source=Source.NVIDIA_SMI, method=method, unit=obs.unit, note=note
    )


@dataclass
class FakeNvidiaSmi:
    driver: DriverInfo
    inventory: GPUInventory
    runtime: tuple[RuntimeMetrics, ...]
    calls: list[str] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakeNvidiaSmi:
        return cls(driver=snapshot.driver, inventory=snapshot.gpu, runtime=snapshot.runtime)

    def locate(self) -> Observed[str]:
        self.calls.append("locate")
        return self.driver.nvidia_smi_path

    def version(self) -> SmiVersion:
        self.calls.append("nvidia-smi")
        return SmiVersion(
            driver_version=self.driver.nvidia_smi_driver_version,
            cuda_version=self.driver.nvidia_smi_cuda_version,
        )

    def query_gpus(self) -> Observed[tuple[SmiGpuRow, ...]]:
        self.calls.append("nvidia-smi --query-gpu")
        if not self.driver.nvidia_smi_path.is_collected:
            return self.driver.nvidia_smi_path
        header = self.driver.nvidia_smi_driver_version
        if not header.is_collected:
            return replace(header, method=fx("nvidia-smi --query-gpu"))
        runtime = {r.device_ref: r for r in self.runtime}
        rows = []
        for device in self.inventory.devices:
            holders = {
                "device": device,
                "runtime": runtime[device.device_ref],
                "driver": self.driver,
            }
            values = {
                query_field: _as_smi(getattr(holders[section], attribute), query_field)
                for query_field, (attribute, section) in _ROW_FIELDS.items()
            }
            rows.append(SmiGpuRow(index=device.nvml_index, values=values))
        return Observed.collected(
            tuple(rows),
            source=Source.NVIDIA_SMI,
            method=fx("nvidia-smi --query-gpu --format=csv,noheader,nounits"),
        )
