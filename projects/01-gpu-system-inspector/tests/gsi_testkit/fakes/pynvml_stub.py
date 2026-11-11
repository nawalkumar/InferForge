"""A stub `pynvml` module for the production `PynvmlAdapter` (injected through its loader).

It mirrors the binding's surface the adapter uses: `NVMLError` with a numeric `value` and
the binding's message, the `NVML_ERROR_*` constants (a test checks them against the real
binding), the version and sensor constants, and the read-only functions. It records every
NVML C function it stands for, counts init/shutdown, raises configured errors per function
or per (function, device index), can hide functions (an older binding) and never loads a
library. `from_snapshot()` reproduces a fixture's NVML facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from gpu_system_inspector.models import CollectionStatus, InspectionSnapshot, Source

GIB = 2**30
MIB = 2**20
NVML_ERRORS = {
    "NVML_ERROR_UNINITIALIZED": 1,
    "NVML_ERROR_INVALID_ARGUMENT": 2,
    "NVML_ERROR_NOT_SUPPORTED": 3,
    "NVML_ERROR_NO_PERMISSION": 4,
    "NVML_ERROR_NOT_FOUND": 6,
    "NVML_ERROR_INSUFFICIENT_POWER": 8,
    "NVML_ERROR_DRIVER_NOT_LOADED": 9,
    "NVML_ERROR_TIMEOUT": 10,
    "NVML_ERROR_LIBRARY_NOT_FOUND": 12,
    "NVML_ERROR_FUNCTION_NOT_FOUND": 13,
    "NVML_ERROR_GPU_IS_LOST": 15,
    "NVML_ERROR_ARGUMENT_VERSION_MISMATCH": 25,
    "NVML_ERROR_UNKNOWN": 999,
}
# `str(pynvml.NVMLError(value))`, as the real binding prints it.
NVML_MESSAGES = {
    1: "Uninitialized",
    2: "Invalid Argument",
    3: "Not Supported",
    4: "Insufficient Permissions",
    6: "Not Found",
    9: "Driver Not Loaded",
    10: "Timeout",
    12: "NVML Shared Library Not Found",
    13: "Function Not Found",
    15: "GPU is lost",
    25: "NVML Error with code 25",
    999: "Unknown Error",
}
MEMORY_V2 = 33554472


class StubNVMLError(Exception):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.value = value

    def __str__(self) -> str:
        return NVML_MESSAGES.get(self.value, f"NVML error {self.value}")


@dataclass
class StubGpu:
    """One synthetic device. `None` in a field is NVML_VALUE_NOT_AVAILABLE."""

    name: str = "Synthetic GPU A (fixture)"
    uuid: str = "GPU-aaaaaaaa-0000-4000-8000-00000000000a"
    bus_id: str = "00000000:01:00.0"
    pci_device_id: int = 0x268410DE
    pci_subsystem_id: int = 0
    memory: tuple[int | None, int | None, int | None, int | None] = (
        24 * GIB,
        4 * GIB,
        20 * GIB,
        300 * MIB,
    )
    compute: tuple[int, int] = (8, 9)
    mig: tuple[int, int] = (0, 0)
    utilization: tuple[int | None, int | None] = (35, 20)
    temperature: int | None = 55
    slowdown: int | None = 90
    power_mw: int | None = 150_000
    power_limit_mw: int | None = 300_000
    processes: tuple[tuple[int, int | None, str], ...] = ()


@dataclass(frozen=True)
class _Handle:
    index: int


class StubPynvml:
    NVMLError = StubNVMLError
    nvmlMemory_v2 = MEMORY_V2
    NVML_TEMPERATURE_GPU = 0
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1
    NVML_DEVICE_MIG_ENABLE = 1
    locals().update(NVML_ERRORS)

    def __init__(
        self,
        gpus: Iterable[StubGpu] = (),
        *,
        faults: Mapping[Any, int] | None = None,
        lost_after_handles: Mapping[int, int] | None = None,
        missing: Iterable[str] = (),
        driver: str = "550.54.15",
        nvml_version: str = "12.550.54.15",
        cuda: int = 12040,
        cuda_library: int = 12040,
        shutdown_fault: int | None = None,
        import_error: str | None = None,
        forbid_load: bool = False,
    ) -> None:
        self.forbid_load = forbid_load
        self.gpus = list(gpus)
        self.faults = dict(faults or {})
        # Device index → handles acquired before the GPU falls off the bus (KI-34); from
        # then on its handle and every call on it raise NVML_ERROR_GPU_IS_LOST.
        self.lost_after_handles = dict(lost_after_handles or {})
        self.handles: dict[int, int] = {}
        self.lost: set[int] = set()
        self.missing = frozenset(missing)
        self.driver = driver
        self.nvml_version = nvml_version
        self.cuda = cuda
        self.cuda_library = cuda_library
        self.shutdown_fault = shutdown_fault
        self.import_error = import_error
        self.calls: list[str] = []
        self.inits = 0
        self.shutdowns = 0
        self.loads = 0

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_") and name in object.__getattribute__(self, "missing"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def loader(self) -> StubPynvml:
        """The adapter's `loader`: the module, or the import failure."""
        self.loads += 1
        if self.forbid_load:
            raise AssertionError("the binding must not be loaded on this platform")
        if self.import_error is not None:
            raise ModuleNotFoundError(self.import_error)
        return self

    # --- system ---------------------------------------------------------------------------

    def _call(self, function: str, index: int | None = None) -> None:
        self.calls.append(function)
        code = self.faults.get((function, index), self.faults.get(function))
        if code is not None:
            raise StubNVMLError(code)

    def nvmlInit(self) -> None:
        self._call("nvmlInit")
        self.inits += 1

    def nvmlShutdown(self) -> None:
        self.calls.append("nvmlShutdown")
        self.shutdowns += 1
        if self.shutdown_fault is not None:
            raise StubNVMLError(self.shutdown_fault)

    def nvmlSystemGetNVMLVersion(self) -> str:
        self._call("nvmlSystemGetNVMLVersion")
        return self.nvml_version

    def nvmlSystemGetDriverVersion(self) -> str:
        self._call("nvmlSystemGetDriverVersion")
        return self.driver

    def nvmlSystemGetCudaDriverVersion(self) -> int:
        self._call("nvmlSystemGetCudaDriverVersion")
        return self.cuda

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        self._call("nvmlSystemGetCudaDriverVersion_v2")
        return self.cuda_library

    def nvmlDeviceGetCount(self) -> int:
        self._call("nvmlDeviceGetCount_v2")
        return len(self.gpus)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> _Handle:
        self._call("nvmlDeviceGetHandleByIndex_v2", index)
        if not 0 <= index < len(self.gpus):
            raise StubNVMLError(NVML_ERRORS["NVML_ERROR_INVALID_ARGUMENT"])
        self.handles[index] = self.handles.get(index, 0) + 1
        limit = self.lost_after_handles.get(index)
        if limit is not None and self.handles[index] > limit:
            self.lost.add(index)
        if index in self.lost:
            raise StubNVMLError(NVML_ERRORS["NVML_ERROR_GPU_IS_LOST"])
        return _Handle(index)

    def nvmlSystemGetProcessName(self, pid: int) -> bytes:
        self._call("nvmlSystemGetProcessName")
        for gpu in self.gpus:
            for process_pid, _, name in gpu.processes:
                if process_pid == pid:
                    return name.encode()
        raise StubNVMLError(NVML_ERRORS["NVML_ERROR_NOT_FOUND"])

    # --- device ---------------------------------------------------------------------------

    def _gpu(self, function: str, handle: _Handle) -> StubGpu:
        self._call(function, handle.index)
        if handle.index in self.lost:
            raise StubNVMLError(NVML_ERRORS["NVML_ERROR_GPU_IS_LOST"])
        return self.gpus[handle.index]

    def nvmlDeviceGetName(self, handle: _Handle) -> str:
        return self._gpu("nvmlDeviceGetName", handle).name

    def nvmlDeviceGetUUID(self, handle: _Handle) -> str:
        return self._gpu("nvmlDeviceGetUUID", handle).uuid

    def nvmlDeviceGetPciInfo_v3(self, handle: _Handle) -> SimpleNamespace:
        gpu = self._gpu("nvmlDeviceGetPciInfo_v3", handle)
        return SimpleNamespace(
            busId=gpu.bus_id.encode(),
            pciDeviceId=gpu.pci_device_id,
            pciSubSystemId=gpu.pci_subsystem_id,
        )

    def nvmlDeviceGetPciInfo(self, handle: _Handle) -> SimpleNamespace:
        gpu = self._gpu("nvmlDeviceGetPciInfo", handle)
        return SimpleNamespace(
            busId=gpu.bus_id, pciDeviceId=gpu.pci_device_id, pciSubSystemId=gpu.pci_subsystem_id
        )

    def nvmlDeviceGetMemoryInfo(
        self, handle: _Handle, version: int | None = None
    ) -> SimpleNamespace:
        if version is not None and version != MEMORY_V2:
            raise StubNVMLError(NVML_ERRORS["NVML_ERROR_ARGUMENT_VERSION_MISMATCH"])
        function = "nvmlDeviceGetMemoryInfo_v2" if version else "nvmlDeviceGetMemoryInfo"
        total, used, free, reserved = self._gpu(function, handle).memory
        if version:
            return SimpleNamespace(total=total, used=used, free=free, reserved=reserved)
        return SimpleNamespace(total=total, used=used, free=free)

    def nvmlDeviceGetCudaComputeCapability(self, handle: _Handle) -> tuple[int, int]:
        return self._gpu("nvmlDeviceGetCudaComputeCapability", handle).compute

    def nvmlDeviceGetMigMode(self, handle: _Handle) -> tuple[int, int]:
        return self._gpu("nvmlDeviceGetMigMode", handle).mig

    def nvmlDeviceGetUtilizationRates(self, handle: _Handle) -> SimpleNamespace:
        gpu, memory = self._gpu("nvmlDeviceGetUtilizationRates", handle).utilization
        return SimpleNamespace(gpu=gpu, memory=memory)

    def nvmlDeviceGetTemperatureV(self, handle: _Handle, sensor: int) -> int | None:
        assert sensor == self.NVML_TEMPERATURE_GPU
        return self._gpu("nvmlDeviceGetTemperatureV", handle).temperature

    def nvmlDeviceGetTemperature(self, handle: _Handle, sensor: int) -> int | None:
        assert sensor == self.NVML_TEMPERATURE_GPU
        return self._gpu("nvmlDeviceGetTemperature", handle).temperature

    def nvmlDeviceGetTemperatureThreshold(self, handle: _Handle, threshold: int) -> int | None:
        assert threshold == self.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
        return self._gpu("nvmlDeviceGetTemperatureThreshold", handle).slowdown

    def nvmlDeviceGetPowerUsage(self, handle: _Handle) -> int | None:
        return self._gpu("nvmlDeviceGetPowerUsage", handle).power_mw

    def nvmlDeviceGetEnforcedPowerLimit(self, handle: _Handle) -> int | None:
        return self._gpu("nvmlDeviceGetEnforcedPowerLimit", handle).power_limit_mw

    def _processes(self, function: str, handle: _Handle) -> list[SimpleNamespace]:
        gpu = self._gpu(function, handle)
        return [SimpleNamespace(pid=pid, usedGpuMemory=used) for pid, used, _ in gpu.processes]

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: _Handle) -> list[SimpleNamespace]:
        return self._processes("nvmlDeviceGetComputeRunningProcesses_v3", handle)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: _Handle) -> list[SimpleNamespace]:
        return self._processes("nvmlDeviceGetComputeRunningProcesses", handle)

    # --- from a fixture -------------------------------------------------------------------

    @classmethod
    def from_snapshot(
        cls, snapshot: InspectionSnapshot, *, lost_after_handles: Mapping[int, int] | None = None
    ) -> StubPynvml:
        """The fixture's NVML facts: its init outcome, versions and NVML inventory."""
        driver, inventory = snapshot.driver, snapshot.gpu
        init = driver.nvml_version
        if init.status is CollectionStatus.NOT_INSTALLED:
            return cls(
                import_error=init.error.message if init.error else "No module named 'pynvml'"
            )
        faults: dict[Any, int] = {}
        if not init.is_collected:
            assert init.error is not None
            if init.error.code not in NVML_ERRORS:  # a platform gate: never loaded at all
                return cls(import_error=None, forbid_load=True)
            faults["nvmlInit"] = NVML_ERRORS[init.error.code]
            return cls(faults=faults)
        runtime = {r.device_ref: r for r in snapshot.runtime}
        gpus = []
        if inventory.inventory_source is Source.NVML:
            for index, device in enumerate(inventory.devices):
                metrics = runtime[device.device_ref]
                memory = []
                # Enumeration's total is the inventory's; the rest is runtime state (KI-34).
                for value in (
                    device.memory_total_bytes,
                    metrics.memory_used_bytes,
                    metrics.memory_free_bytes,
                    metrics.memory_reserved_bytes,
                ):
                    memory.append(value.value)
                if not device.memory_total_bytes.is_collected:
                    error = device.memory_total_bytes.error
                    if error is not None and error.code in NVML_ERRORS:
                        faults[("nvmlDeviceGetMemoryInfo_v2", index)] = NVML_ERRORS[error.code]
                major, minor = (int(part) for part in device.compute_capability.value.split("."))
                gpus.append(
                    StubGpu(
                        name=device.name.value,
                        uuid=device.uuid.value,
                        bus_id=device.pci_bus_id.value,
                        pci_device_id=int(device.pci_device_id.value, 16),
                        pci_subsystem_id=int(device.pci_subsystem_id.value, 16),
                        memory=tuple(memory),  # type: ignore[arg-type]
                        compute=(major, minor),
                        mig=(1, 1) if device.mig_mode.value == "ENABLED" else (0, 0),
                    )
                )
        return cls(
            gpus,
            faults=faults,
            lost_after_handles=lost_after_handles,
            driver=driver.driver_version.value or "550.54.15",
            nvml_version=init.value,
            cuda=_encode(driver.driver_supported_cuda_version.value),
            cuda_library=_encode(driver.cuda_driver_library.value),
        )


def _encode(version: str | None) -> int:
    if not version:
        return 12040
    major, minor = (int(part) for part in version.split("."))
    return major * 1000 + minor * 10
