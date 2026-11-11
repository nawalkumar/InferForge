"""GPU hardware collector (architecture §4.4, DEC-44): the `GPUInventory` section.

The device inventory comes from the shared NVML session (opened and closed by the core,
never here), with the `nvidia-smi` query as the fallback when NVML can't count devices. The
PCI observations are the `PciCollector`'s section, passed in unchanged: this collector never
scans PCI and never emits `GPU.PCI_SCAN` (DEC-43). It owns `NVIDIA.ALL` (macOS).
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from gpu_system_inspector.adapters import Clock, NvidiaSmiAdapter, NvmlSession, SmiGpuRow
from gpu_system_inspector.collectors.capabilities import NVIDIA_ALL, unsupported_capabilities_for
from gpu_system_inspector.collectors.system import FAILURE_STATUSES
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    GPUDeviceInfo,
    GPUInventory,
    LinuxEnvironment,
    Observed,
    PciObservation,
    PlatformProfile,
    Source,
)
from gpu_system_inspector.models.gpu import NVIDIA_VENDOR

GPU_REPORT_PATH = "/gpu"
FALLBACK_FIELD_UNAVAILABLE = "FALLBACK_FIELD_UNAVAILABLE"
FALLBACK_FIELD_MESSAGE = "not part of the nvidia-smi fallback query"
SMI_COUNT_METHOD = "nvidia-smi --query-gpu=index"
# GPUDeviceInfo fields the nvidia-smi fallback query doesn't carry, and their smi field names.
SMI_MISSING_FIELDS = {
    "pci_device_id": "pci.device_id",
    "pci_subsystem_id": "pci.sub_device_id",
    "compute_capability": "compute_cap",
    "mig_mode": "mig.mode.current",
}
_DEVICE_FIELDS = (
    "name",
    "uuid",
    "pci_bus_id",
    "pci_device_id",
    "pci_subsystem_id",
    "memory_total_bytes",
    "compute_capability",
    "mig_mode",
)


class GpuHardwareCollector:
    name: ClassVar[str] = "gpu_hardware"

    def __init__(
        self,
        session: NvmlSession,
        nvidia_smi: NvidiaSmiAdapter,
        clock: Clock,
        *,
        pci: Observed[tuple[PciObservation, ...]],
        environment: LinuxEnvironment | None,
        profile: PlatformProfile,
    ) -> None:
        # Raises AdapterUsageError for an inconsistent environment/profile pair.
        policy = unsupported_capabilities_for(environment, profile)
        self._session = session
        self._nvidia_smi = nvidia_smi
        self._clock = clock
        self._pci = pci
        self._unsupported = tuple(u for u in policy if u.capability == NVIDIA_ALL)
        synthetic = getattr(nvidia_smi, "provenance", DataOrigin.LIVE) is DataOrigin.FIXTURE
        self._prefix = FIXTURE_METHOD_PREFIX if synthetic else ""

    def collect(self) -> CollectorOutcome[GPUInventory]:
        started = self._clock.monotonic_ms()
        errors: list[CollectionError] = []
        count = self._session.device_count()
        if count.is_collected:
            devices = tuple(self._nvml_device(index, errors) for index in range(count.value))
            inventory = GPUInventory(
                device_count=count,
                inventory_source=Source.NVML,
                devices=devices,
                pci_observations=self._pci,
            )
        else:
            if self._session.init_result.is_collected and count.status in FAILURE_STATUSES:
                errors.append(_error(count, f"{GPU_REPORT_PATH}/device_count"))
            inventory = self._fallback(count, errors)
        partial = bool(errors) or any(
            getattr(device, field).status in FAILURE_STATUSES
            for device in inventory.devices
            for field in _DEVICE_FIELDS
        )
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(
            section=inventory, run=run, errors=tuple(errors), unsupported=self._unsupported
        )

    def unavailable(self, error: ErrorDetail) -> GPUInventory:
        """The guard's fallback. The PCI section came from another collector, so it's kept."""
        failed = GPUInventory.unavailable(CollectionStatus.COLLECTION_FAILED, error)
        return replace(failed, pci_observations=self._pci)

    # --- NVML -----------------------------------------------------------------------------

    def _nvml_device(self, index: int, errors: list[CollectionError]) -> GPUDeviceInfo:
        ref = f"nvml:{index}"
        handle = self._session.device(index)
        if not handle.is_collected:
            errors.append(_error(handle, f"{GPU_REPORT_PATH}/devices/{index}", device_ref=ref))
            return GPUDeviceInfo.unavailable(ref, handle.status, handle.error)
        device = handle.value
        assert device is not None
        pci = device.pci_info()
        return GPUDeviceInfo(
            device_ref=ref,
            nvml_index=index,
            vendor=NVIDIA_VENDOR,
            name=device.name(),
            uuid=device.uuid(),
            pci_bus_id=pci.bus_id,
            pci_device_id=pci.device_id,
            pci_subsystem_id=pci.subsystem_id,
            memory_total_bytes=device.memory_info().total,
            compute_capability=device.compute_capability(),
            mig_mode=device.mig_mode(),
        )

    # --- nvidia-smi fallback --------------------------------------------------------------

    def _fallback(self, nvml_count: Observed[int], errors: list[CollectionError]) -> GPUInventory:
        rows = self._nvidia_smi.query_gpus()
        if rows.is_collected:
            assert rows.value is not None
            devices = tuple(self._smi_device(row) for row in rows.value)
            count = Observed.collected(
                len(devices), source=Source.NVIDIA_SMI, method=self._prefix + SMI_COUNT_METHOD
            )
            return GPUInventory(
                device_count=count,
                inventory_source=Source.NVIDIA_SMI,
                devices=devices,
                pci_observations=self._pci,
            )
        if rows.status in FAILURE_STATUSES:
            errors.append(_error(rows, f"{GPU_REPORT_PATH}/devices"))
        return GPUInventory(
            device_count=nvml_count,
            inventory_source=None,
            devices=(),
            pci_observations=self._pci,
        )

    def _smi_device(self, row: SmiGpuRow) -> GPUDeviceInfo:
        values = row.values
        missing = {
            field: Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code=FALLBACK_FIELD_UNAVAILABLE,
                message=FALLBACK_FIELD_MESSAGE,
                source=Source.NVIDIA_SMI,
                method=self._prefix + smi_field,
            )
            for field, smi_field in SMI_MISSING_FIELDS.items()
        }
        return GPUDeviceInfo(
            device_ref=f"smi:{row.index}",
            nvml_index=row.index,
            vendor=NVIDIA_VENDOR,
            name=values["name"],
            uuid=values["uuid"],
            pci_bus_id=values["pci.bus_id"],
            memory_total_bytes=values["memory.total"],
            **missing,
        )


def _error(
    outcome: Observed, report_path: str, *, device_ref: str | None = None
) -> CollectionError:
    assert outcome.error is not None
    return CollectionError(
        collector=GpuHardwareCollector.name,
        device_ref=device_ref,
        status=outcome.status,
        error_code=outcome.error.code,
        message=outcome.error.message,
        report_path=report_path,
    )
