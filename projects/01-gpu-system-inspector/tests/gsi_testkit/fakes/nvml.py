"""Fake NVML adapter implementing `NvmlAdapter` / `NvmlSession` / `NvmlDevice`.

Answers read-only NVML queries from a fixture snapshot, records every NVML function it
would call, and turns configured NVML errors per (device index, function) into the mapped
outcomes (data model §3.3), exactly as a real adapter must. It never loads NVML.

**Device loss** (KI-34). `faults` are static: they hold for the whole session. A GPU that
falls off the bus is a change *during* the session, so it has its own lifecycle:
`lost_after_handles={index: n}` loses the device once `n` handles were acquired for it
(FX10: `{1: 1}`, lost after enumeration), `lost_on={index: function}` at the first call
of that NVML function (a device or a system call, e.g. while the driver collector reads
`nvmlSystemGetDriverVersion`). From then on its handle and every query on it report
`NVML_ERROR_GPU_IS_LOST`, as NVML does; system queries keep working.

**Memory.** `memory_info().total` is the inventory's total (what enumeration saw); used,
free and reserved are the runtime record's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, Self

from gpu_system_inspector.adapters import (
    NVML_DEVICE_FUNCTIONS,
    NVML_READ_ONLY_FUNCTIONS,
    NVML_SESSION_FUNCTIONS,
    AdapterUsageError,
    NvmlMemoryInfo,
    NvmlPciInfo,
    NvmlUtilization,
    nvml_error_status,
)
from gpu_system_inspector.models import (
    DataOrigin,
    DriverInfo,
    GPUDeviceInfo,
    GPUInventory,
    GpuProcess,
    InspectionSnapshot,
    Observed,
    RuntimeMetrics,
    Source,
)
from gsi_testkit.builders import fx

READ_ONLY_FUNCTIONS = NVML_READ_ONLY_FUNCTIONS
GPU_IS_LOST = "NVML_ERROR_GPU_IS_LOST"

# Messages as `str(pynvml.NVMLError_*)` reports them; unlisted codes use the code itself.
NVML_ERROR_MESSAGES = {
    "NVML_ERROR_GPU_IS_LOST": "GPU is lost",
    "NVML_ERROR_NOT_SUPPORTED": "Not Supported",
    "NVML_ERROR_NO_PERMISSION": "Insufficient Permissions",
    "NVML_ERROR_TIMEOUT": "Timeout",
    "NVML_ERROR_UNKNOWN": "Unknown Error",
}


@dataclass
class FakeNvml:
    driver: DriverInfo
    inventory: GPUInventory
    runtime: tuple[RuntimeMetrics, ...]
    faults: Mapping[tuple[int, str], str] = field(default_factory=dict)
    lost_after_handles: Mapping[int, int] = field(default_factory=dict)
    lost_on: Mapping[int, str] = field(default_factory=dict)
    shutdown_fault: str | None = None
    calls: list[str] = field(default_factory=list)
    metadata_calls: list[str] = field(default_factory=list)
    shutdown_errors: list[str] = field(default_factory=list)
    inits: int = 0
    shutdowns: int = 0
    lost: set[int] = field(default_factory=set)
    handles: dict[int, int] = field(default_factory=dict)
    _session: FakeNvmlSession | None = field(default=None, init=False, repr=False)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(
        cls,
        snapshot: InspectionSnapshot,
        faults: Mapping[tuple[int, str], str] | None = None,
        *,
        lost_after_handles: Mapping[int, int] | None = None,
        lost_on: Mapping[int, str] | None = None,
    ) -> FakeNvml:
        return cls(
            driver=snapshot.driver,
            inventory=snapshot.gpu,
            runtime=snapshot.runtime,
            faults=dict(faults or {}),
            lost_after_handles=dict(lost_after_handles or {}),
            lost_on=dict(lost_on or {}),
        )

    @classmethod
    def from_scenario(cls, scenario: Any) -> FakeNvml:
        return cls.from_snapshot(
            scenario.snapshot,
            scenario.nvml_faults,
            lost_after_handles=scenario.nvml_lost_after_handles,
        )

    @property
    def available(self) -> bool:
        return self.driver.nvml_version.is_collected

    def is_lost(self, index: int, function: str) -> bool:
        """Whether the device is gone at this call (and from now on)."""
        if index not in self.lost and self.lost_on.get(index) == function:
            self.lost.add(index)
        return index in self.lost

    def acquire_handle(self, index: int, function: str) -> bool:
        """Counts a handle acquisition; False once the device is (or just became) lost."""
        if self.is_lost(index, function):
            return False
        self.handles[index] = self.handles.get(index, 0) + 1
        limit = self.lost_after_handles.get(index)
        if limit is not None and self.handles[index] > limit:
            self.lost.add(index)
            return False
        return True

    def binding_version(self) -> Observed[str]:
        self.metadata_calls.append("importlib.metadata.version")
        return self.driver.nvml_binding

    def binding_distribution(self) -> Observed[str]:
        self.metadata_calls.append("importlib.metadata.packages_distributions")
        return self.driver.nvml_binding_distribution

    def open(self) -> FakeNvmlSession:
        if self._session is not None and self._session.is_open:
            raise AdapterUsageError("an NVML session is already open; share it")
        self.calls.append(NVML_SESSION_FUNCTIONS["open"])
        if self.available:
            self.inits += 1
        self._session = FakeNvmlSession(self, initialised=self.available)
        return self._session


class FakeNvmlSession:
    def __init__(self, nvml: FakeNvml, *, initialised: bool) -> None:
        self.nvml = nvml
        self._initialised = initialised
        self._closed = False

    @property
    def init_result(self) -> Observed[str]:
        return self.nvml.driver.nvml_version

    @property
    def is_open(self) -> bool:
        return not self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._initialised:
            return
        self.nvml.calls.append(NVML_SESSION_FUNCTIONS["close"])
        self.nvml.shutdowns += 1
        if self.nvml.shutdown_fault is not None:
            self.nvml.shutdown_errors.append(self.nvml.shutdown_fault)

    def require_open(self) -> None:
        if self._closed:
            raise AdapterUsageError("the NVML session is closed")

    def _system(self, method: str, value: Observed) -> Observed:
        self.require_open()
        function = NVML_SESSION_FUNCTIONS[method]
        self.nvml.calls.append(function)
        for index in self.nvml.lost_on:  # a device may fall off during any NVML call
            self.nvml.is_lost(index, function)
        return value if self._initialised else replace(self.init_result, unit=value.unit)

    def driver_version(self) -> Observed[str]:
        return self._system("driver_version", self.nvml.driver.driver_version)

    def cuda_driver_version(self) -> Observed[str]:
        return self._system("cuda_driver_version", self.nvml.driver.driver_supported_cuda_version)

    def cuda_driver_library(self) -> Observed[str]:
        return self._system("cuda_driver_library", self.nvml.driver.cuda_driver_library)

    def device_count(self) -> Observed[int]:
        return self._system("device_count", self.nvml.inventory.device_count)

    def device(self, index: int) -> Observed[FakeNvmlDevice]:
        self.require_open()
        function = NVML_SESSION_FUNCTIONS["device"]
        self.nvml.calls.append(function)
        if not self._initialised:
            return self.init_result  # type: ignore[return-value]
        inventory = self.nvml.inventory
        assert inventory.inventory_source is Source.NVML, "fixture NVML has no NVML inventory"
        devices = {d.nvml_index: d for d in inventory.devices}
        if index not in devices:
            raise AdapterUsageError(f"device index {index} is outside 0..{len(devices) - 1}")
        code = self.nvml.faults.get((index, function))
        if code is not None:
            return nvml_failure(code, function)
        if not self.nvml.acquire_handle(index, function):
            return nvml_failure(GPU_IS_LOST, function)
        device = devices[index]
        runtime = next(r for r in self.nvml.runtime if r.device_ref == device.device_ref)
        return Observed.collected(
            FakeNvmlDevice(self, device, runtime), source=Source.NVML, method=fx(function)
        )


def nvml_failure(code: str, function: str, unit: str | None = None) -> Observed:
    return Observed.failed(
        nvml_error_status(code),
        code=code,
        message=NVML_ERROR_MESSAGES.get(code, code),
        source=Source.NVML,
        method=fx(function),
        unit=unit,
    )


class FakeNvmlDevice:
    def __init__(
        self, session: FakeNvmlSession, device: GPUDeviceInfo, runtime: RuntimeMetrics
    ) -> None:
        self._session = session
        self._device = device
        self._runtime = runtime

    @property
    def index(self) -> int:
        return self._device.nvml_index

    def _fault(self, method: str) -> str | None:
        """Records the call and returns the injected NVML error code, if any."""
        self._session.require_open()
        function = NVML_DEVICE_FUNCTIONS[method]
        nvml = self._session.nvml
        nvml.calls.append(function)
        if nvml.is_lost(self.index, function):
            return GPU_IS_LOST
        return nvml.faults.get((self.index, function))

    def _read(self, method: str, value: Observed) -> Observed:
        code = self._fault(method)
        return (
            value if code is None else nvml_failure(code, NVML_DEVICE_FUNCTIONS[method], value.unit)
        )

    def name(self) -> Observed[str]:
        return self._read("name", self._device.name)

    def uuid(self) -> Observed[str]:
        return self._read("uuid", self._device.uuid)

    def pci_info(self) -> NvmlPciInfo:
        code = self._fault("pci_info")
        d = self._device
        if code is not None:
            lost = nvml_failure(code, NVML_DEVICE_FUNCTIONS["pci_info"])
            return NvmlPciInfo(bus_id=lost, device_id=lost, subsystem_id=lost)
        return NvmlPciInfo(
            bus_id=d.pci_bus_id, device_id=d.pci_device_id, subsystem_id=d.pci_subsystem_id
        )

    def memory_info(self) -> NvmlMemoryInfo:
        code = self._fault("memory_info")
        r = self._runtime
        if code is not None:
            lost = nvml_failure(code, NVML_DEVICE_FUNCTIONS["memory_info"], "bytes")
            return NvmlMemoryInfo(total=lost, used=lost, free=lost, reserved=lost)
        return NvmlMemoryInfo(
            total=self._device.memory_total_bytes,
            used=r.memory_used_bytes,
            free=r.memory_free_bytes,
            reserved=r.memory_reserved_bytes,
        )

    def compute_capability(self) -> Observed[str]:
        return self._read("compute_capability", self._device.compute_capability)

    def mig_mode(self) -> Observed[str]:
        return self._read("mig_mode", self._device.mig_mode)

    def utilization(self) -> NvmlUtilization:
        code = self._fault("utilization")
        r = self._runtime
        if code is not None:
            lost = nvml_failure(code, NVML_DEVICE_FUNCTIONS["utilization"], "percent")
            return NvmlUtilization(gpu=lost, memory_bus=lost)
        return NvmlUtilization(
            gpu=r.gpu_utilization_percent, memory_bus=r.memory_bus_utilization_percent
        )

    def temperature(self) -> Observed[int]:
        return self._read("temperature", self._runtime.temperature_gpu_celsius)

    def slowdown_threshold(self) -> Observed[int]:
        return self._read(
            "slowdown_threshold", self._runtime.temperature_slowdown_threshold_celsius
        )

    def power_draw(self) -> Observed[float]:
        return self._read("power_draw", self._runtime.power_draw_watts)

    def power_limit(self) -> Observed[float]:
        return self._read("power_limit", self._runtime.power_limit_watts)

    def processes(self) -> Observed[tuple[GpuProcess, ...]]:
        return self._read("processes", self._runtime.processes)
