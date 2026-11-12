"""Production `NvmlAdapter` / `NvmlSession` / `NvmlDevice` over `pynvml` (`nvidia-ml-py`,
DEC-05, EV-11; architecture §4.5.3, §4.5.5; IMP-A01).

- **Lazy, injectable binding.** `pynvml` is imported by `open()` through `loader` (a test
  passes a stub module); importing this package loads no library. A missing binding is
  `NOT_INSTALLED`/`ImportError` (a broken installation, not "no GPU").
- **Lifecycle (§4.5.3).** One `open()` = one `nvmlInit()`. `close()` is idempotent and calls
  `nvmlShutdown()` once, and only after a successful init; a shutdown error is swallowed,
  logged at DEBUG and kept in `shutdown_errors`, so it never hides the inspection's outcome.
  A second `open()` while a session is open is an `AdapterUsageError`; so is any query on a
  closed session or a handle from it.
- **Outcomes (data model §3.3).** An `NVMLError` becomes the outcome for its NVML error name
  (`nvml_error_status`: library not found / driver not loaded → `NOT_DETECTED`, no
  permission → `PERMISSION_DENIED`, not supported / function not found → `NOT_SUPPORTED`,
  anything else, e.g. GPU lost → `COLLECTION_FAILED`). `None` from the binding is
  `NOT_SUPPORTED`/`VALUE_NOT_AVAILABLE`. If init failed, every query returns the init
  outcome. Each query is independent, so one failing field or device leaves the rest intact.
- **Versions.** `_v2`/`_v3`/`V` variants first; on `FUNCTION_NOT_FOUND` (older driver) or a
  version mismatch, the older call (`nvmlDeviceGetMemoryInfo` v1, `nvmlDeviceGetTemperature`,
  `nvmlDeviceGetComputeRunningProcesses`, `nvmlDeviceGetPciInfo`).
- **macOS.** NVIDIA CUDA GPUs aren't supported there (product spec §4): with a macOS
  platform the session is `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED` without loading anything.

Only read-only functions are called (`NVML_READ_ONLY_FUNCTIONS`, NFR-12).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from types import TracebackType
from typing import Any, Self

from gpu_system_inspector.adapters.outcomes import (
    PARSE_ERROR,
    PLATFORM_NOT_SUPPORTED,
    VALUE_NOT_AVAILABLE,
    AdapterUsageError,
    exception_outcome,
    nvml_error_status,
)
from gpu_system_inspector.adapters.protocols import (
    NvmlMemoryInfo,
    NvmlPciInfo,
    NvmlUtilization,
    PlatformAdapter,
)
from gpu_system_inspector.models import (
    DRIVER_SUPPORTED_CUDA_NOTE,
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    DataOrigin,
    GpuProcess,
    Observed,
    PlatformProfile,
    ProcessKind,
    Source,
)

LOG = logging.getLogger(__name__)
PYNVML = "pynvml"
PREFERRED_DISTRIBUTION = "nvidia-ml-py"
FUNCTION_NOT_FOUND = "NVML_ERROR_FUNCTION_NOT_FOUND"
VERSION_MISMATCH = "NVML_ERROR_ARGUMENT_VERSION_MISMATCH"
UNKNOWN_ERROR = "NVML_ERROR_UNKNOWN"
NOT_ON_MACOS = "NVIDIA CUDA GPUs are not supported on macOS"
_OLDER_VARIANT = frozenset({FUNCTION_NOT_FOUND, VERSION_MISMATCH})
_MISSING = object()

PynvmlLoader = Callable[[], Any]


def import_pynvml() -> Any:
    import importlib

    return importlib.import_module(PYNVML)


class PynvmlAdapter:
    def __init__(
        self,
        loader: PynvmlLoader = import_pynvml,
        *,
        platform: PlatformAdapter | None = None,
        provenance: DataOrigin = DataOrigin.LIVE,
        packages_distributions: Callable[[], Any] | None = None,
        version: Callable[[str], str] | None = None,
    ) -> None:
        self._loader = loader
        self._platform = platform
        self.provenance = provenance
        self._prefix = FIXTURE_METHOD_PREFIX if provenance is DataOrigin.FIXTURE else ""
        self._packages_distributions = packages_distributions
        self._version = version
        self._session: PynvmlSession | None = None
        # One distribution lookup per adapter. A returned outcome, including a failure,
        # is reused. An exception from the lookup is not stored.
        self._binding_distribution: Observed[str] | None = None

    # --- binding facts, without importing pynvml ------------------------------------------

    def binding_distribution(self) -> Observed[str]:
        cached = self._binding_distribution
        if cached is None:
            cached = self._read_binding_distribution()
            self._binding_distribution = cached
        return cached

    def _read_binding_distribution(self) -> Observed[str]:
        method = self._prefix + "importlib.metadata.packages_distributions"
        lookup = self._packages_distributions
        if lookup is None:
            from importlib.metadata import packages_distributions as lookup
        distributions = sorted(set(lookup().get(PYNVML, ())))
        if not distributions:
            return Observed.failed(
                CollectionStatus.NOT_INSTALLED,
                code="ImportError",
                message=f"No module named {PYNVML!r}",
                source=Source.PLATFORM,
                method=method,
            )
        chosen = (
            PREFERRED_DISTRIBUTION if PREFERRED_DISTRIBUTION in distributions else distributions[0]
        )
        return Observed.collected(chosen, source=Source.PLATFORM, method=method)

    def binding_version(self) -> Observed[str]:
        method = self._prefix + "importlib.metadata.version"
        distribution = self.binding_distribution()
        if not distribution.is_collected:
            return replace(distribution, method=method)
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as installed_version

        version = self._version or installed_version
        assert distribution.value is not None
        try:
            value = version(distribution.value)
        except PackageNotFoundError as exc:
            return exception_outcome(exc, source=Source.PLATFORM, method=method)
        return Observed.collected(value, source=Source.PLATFORM, method=method)

    # --- the session ----------------------------------------------------------------------

    def open(self) -> PynvmlSession:
        if self._session is not None and self._session.is_open:
            raise AdapterUsageError("an NVML session is already open; share it")
        macos = self._platform is not None and self._platform.profile is PlatformProfile.MACOS
        self._session = PynvmlSession(None if macos else self._loader, prefix=self._prefix)
        return self._session


class PynvmlSession:
    def __init__(self, loader: PynvmlLoader | None, *, prefix: str = "") -> None:
        self._prefix = prefix
        self._nvml: Any = None
        self._codes: dict[int, str] = {}
        self._initialised = False
        self._closed = False
        self._count: int | None = None
        self.shutdown_errors: list[str] = []
        if loader is None:
            self._init = Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code=PLATFORM_NOT_SUPPORTED,
                message=NOT_ON_MACOS,
                source=Source.NVML,
                method=prefix + "platform gate",
            )
            return
        try:
            module = loader()
        except ImportError as exc:
            self._init = Observed.failed(
                CollectionStatus.NOT_INSTALLED,
                code="ImportError",
                message=str(exc) or f"No module named {PYNVML!r}",
                source=Source.NVML,
                method=prefix + "import pynvml",
            )
            return
        self._nvml = module
        constants = {
            name: getattr(module, name) for name in dir(module) if name.startswith("NVML_ERROR_")
        }
        self._codes = {
            value: name
            for name, value in sorted(constants.items())
            if isinstance(value, int) and not isinstance(value, bool)
        }
        try:
            module.nvmlInit()
        except module.NVMLError as exc:
            self._init = self._failure(exc, "nvmlInit")
            return
        self._initialised = True
        self._init = self._query("nvmlSystemGetNVMLVersion", module.nvmlSystemGetNVMLVersion, _text)

    # --- NvmlSession ----------------------------------------------------------------------

    @property
    def init_result(self) -> Observed[str]:
        return self._init

    @property
    def is_open(self) -> bool:
        return not self._closed

    def driver_version(self) -> Observed[str]:
        return self._read(
            "nvmlSystemGetDriverVersion", lambda m: m.nvmlSystemGetDriverVersion(), _text
        )

    def cuda_driver_version(self) -> Observed[str]:
        return self._read(
            "nvmlSystemGetCudaDriverVersion",
            lambda m: m.nvmlSystemGetCudaDriverVersion(),
            _cuda_version,
            note=DRIVER_SUPPORTED_CUDA_NOTE,
        )

    def cuda_driver_library(self) -> Observed[str]:
        return self._read(
            "nvmlSystemGetCudaDriverVersion_v2",
            lambda m: m.nvmlSystemGetCudaDriverVersion_v2(),
            _cuda_version,
        )

    def device_count(self) -> Observed[int]:
        count = self._read("nvmlDeviceGetCount_v2", lambda m: m.nvmlDeviceGetCount(), _count)
        if count.is_collected:
            self._count = count.value
        return count

    def device(self, index: int) -> Observed[PynvmlDevice]:
        self.require_open()
        if not self._initialised:
            return self._init  # type: ignore[return-value]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise AdapterUsageError(f"device index must be a non-negative int, got {index!r}")
        if self._count is None:
            self.device_count()
        if self._count is not None and index >= self._count:
            raise AdapterUsageError(f"device index {index} is outside 0..{self._count - 1}")
        handle = self._read(
            "nvmlDeviceGetHandleByIndex_v2", lambda m: m.nvmlDeviceGetHandleByIndex(index)
        )
        if not handle.is_collected:
            return handle  # type: ignore[return-value]
        return Observed.collected(
            PynvmlDevice(self, index, handle.value),
            source=Source.NVML,
            method=handle.method or "",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._initialised:
            return
        try:
            self._nvml.nvmlShutdown()
        except self._nvml.NVMLError as exc:
            code = self._code(exc)
            self.shutdown_errors.append(code)
            LOG.debug("nvmlShutdown failed (%s); ignored", code)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- shared helpers (also used by PynvmlDevice) ----------------------------------------

    def require_open(self) -> None:
        if self._closed:
            raise AdapterUsageError("the NVML session is closed")

    def _read(
        self,
        function: str,
        call: Callable[[Any], Any],
        convert: Callable[[Any], Any] = lambda value: value,
        *,
        unit: str | None = None,
        note: str | None = None,
    ) -> Observed:
        self.require_open()
        if not self._initialised:
            return replace(self._init, unit=unit)
        return self._query(function, lambda: call(self._nvml), convert, unit=unit, note=note)

    def _query(
        self,
        function: str,
        call: Callable[[], Any],
        convert: Callable[[Any], Any] = lambda value: value,
        *,
        unit: str | None = None,
        note: str | None = None,
    ) -> Observed:
        method = self._prefix + function
        try:
            raw = call()
        except self._nvml.NVMLError as exc:
            return self._failure(exc, function, unit)
        if raw is None:
            return Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code=VALUE_NOT_AVAILABLE,
                message=f"{function} reported no value",
                source=Source.NVML,
                method=method,
                unit=unit,
            )
        try:
            value = convert(raw)
        except (TypeError, ValueError, AttributeError) as exc:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=PARSE_ERROR,
                message=f"{function} returned an unusable value: {exc}",
                source=Source.NVML,
                method=method,
                unit=unit,
            )
        return Observed.collected(value, source=Source.NVML, method=method, unit=unit, note=note)

    def _failure(self, exc: BaseException, function: str, unit: str | None = None) -> Observed:
        code = self._code(exc)
        return Observed.failed(
            nvml_error_status(code),
            code=code,
            message=str(exc) or code,
            source=Source.NVML,
            method=self._prefix + function,
            unit=unit,
        )

    def _code(self, exc: BaseException) -> str:
        return self._codes.get(getattr(exc, "value", None), UNKNOWN_ERROR)  # type: ignore[arg-type]


class PynvmlDevice:
    """A device handle; valid only while its session is open."""

    def __init__(self, session: PynvmlSession, index: int, handle: Any) -> None:
        self._session = session
        self._index = index
        self._handle = handle

    @property
    def index(self) -> int:
        return self._index

    def name(self) -> Observed[str]:
        return self._read("nvmlDeviceGetName", lambda m: m.nvmlDeviceGetName(self._handle), _text)

    def uuid(self) -> Observed[str]:
        return self._read("nvmlDeviceGetUUID", lambda m: m.nvmlDeviceGetUUID(self._handle), _text)

    def pci_info(self) -> NvmlPciInfo:
        info = self._with_fallback(
            ("nvmlDeviceGetPciInfo_v3", lambda m: m.nvmlDeviceGetPciInfo_v3(self._handle)),
            ("nvmlDeviceGetPciInfo", lambda m: m.nvmlDeviceGetPciInfo(self._handle)),
        )
        if not info.is_collected:
            return NvmlPciInfo(bus_id=info, device_id=info, subsystem_id=info)
        return NvmlPciInfo(
            bus_id=self._field(info, "busId", _text),
            device_id=self._field(info, "pciDeviceId", _hex32),
            subsystem_id=self._field(info, "pciSubSystemId", _hex32),
        )

    def memory_info(self) -> NvmlMemoryInfo:
        version = getattr(self._session._nvml, "nvmlMemory_v2", None)
        candidates = []
        if version is not None:
            candidates.append(
                (
                    "nvmlDeviceGetMemoryInfo_v2",
                    lambda m: m.nvmlDeviceGetMemoryInfo(self._handle, version=version),
                )
            )
        candidates.append(
            ("nvmlDeviceGetMemoryInfo", lambda m: m.nvmlDeviceGetMemoryInfo(self._handle))
        )
        info = self._with_fallback(*candidates, unit="bytes")
        if not info.is_collected:
            return NvmlMemoryInfo(total=info, used=info, free=info, reserved=info)
        has_reserved = info.method is not None and info.method.endswith("_v2")
        reserved = (
            self._field(info, "reserved", _bytes, unit="bytes")
            if has_reserved
            else Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code=VALUE_NOT_AVAILABLE,
                message="nvmlDeviceGetMemoryInfo (v1) reports no reserved memory",
                source=Source.NVML,
                method=info.method,
                unit="bytes",
            )
        )
        return NvmlMemoryInfo(
            total=self._field(info, "total", _bytes, unit="bytes"),
            used=self._field(info, "used", _bytes, unit="bytes"),
            free=self._field(info, "free", _bytes, unit="bytes"),
            reserved=reserved,
        )

    def compute_capability(self) -> Observed[str]:
        return self._read(
            "nvmlDeviceGetCudaComputeCapability",
            lambda m: m.nvmlDeviceGetCudaComputeCapability(self._handle),
            lambda pair: f"{int(pair[0])}.{int(pair[1])}",
        )

    def mig_mode(self) -> Observed[str]:
        enabled = getattr(self._session._nvml, "NVML_DEVICE_MIG_ENABLE", 1)
        return self._read(
            "nvmlDeviceGetMigMode",
            lambda m: m.nvmlDeviceGetMigMode(self._handle),
            lambda modes: "ENABLED" if int(modes[0]) == enabled else "DISABLED",
        )

    def utilization(self) -> NvmlUtilization:
        rates = self._read(
            "nvmlDeviceGetUtilizationRates",
            lambda m: m.nvmlDeviceGetUtilizationRates(self._handle),
            unit="percent",
        )
        if not rates.is_collected:
            return NvmlUtilization(gpu=rates, memory_bus=rates)
        return NvmlUtilization(
            gpu=self._field(rates, "gpu", _percent, unit="percent"),
            memory_bus=self._field(rates, "memory", _percent, unit="percent"),
        )

    def temperature(self) -> Observed[int]:
        def sensor(m: Any) -> int:
            return getattr(m, "NVML_TEMPERATURE_GPU", 0)

        return self._with_fallback(
            (
                "nvmlDeviceGetTemperatureV",
                lambda m: m.nvmlDeviceGetTemperatureV(self._handle, sensor(m)),
            ),
            (
                "nvmlDeviceGetTemperature",
                lambda m: m.nvmlDeviceGetTemperature(self._handle, sensor(m)),
            ),
            convert=int,
            unit="celsius",
        )

    def slowdown_threshold(self) -> Observed[int]:
        return self._read(
            "nvmlDeviceGetTemperatureThreshold",
            lambda m: m.nvmlDeviceGetTemperatureThreshold(
                self._handle, getattr(m, "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN", 1)
            ),
            int,
            unit="celsius",
        )

    def power_draw(self) -> Observed[float]:
        return self._read(
            "nvmlDeviceGetPowerUsage",
            lambda m: m.nvmlDeviceGetPowerUsage(self._handle),
            _watts,
            unit="watts",
        )

    def power_limit(self) -> Observed[float]:
        return self._read(
            "nvmlDeviceGetEnforcedPowerLimit",
            lambda m: m.nvmlDeviceGetEnforcedPowerLimit(self._handle),
            _watts,
            unit="watts",
        )

    def processes(self) -> Observed[tuple[GpuProcess, ...]]:
        listed = self._with_fallback(
            (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                lambda m: m.nvmlDeviceGetComputeRunningProcesses_v3(self._handle),
            ),
            (
                "nvmlDeviceGetComputeRunningProcesses",
                lambda m: m.nvmlDeviceGetComputeRunningProcesses(self._handle),
            ),
        )
        if not listed.is_collected:
            return listed
        processes = []
        for entry in listed.value or ():
            pid = int(entry.pid)
            used = entry.usedGpuMemory
            processes.append(
                GpuProcess(
                    pid=pid,
                    kind=ProcessKind.COMPUTE,
                    used_memory_bytes=(
                        Observed.collected(
                            _bytes(used),
                            source=Source.NVML,
                            method=listed.method or "",
                            unit="bytes",
                        )
                        if used is not None
                        else Observed.failed(
                            CollectionStatus.NOT_SUPPORTED,
                            code=VALUE_NOT_AVAILABLE,
                            message="NVML reports no per-process memory (e.g. under WDDM)",
                            source=Source.NVML,
                            method=listed.method,
                            unit="bytes",
                        )
                    ),
                    name=self._read(
                        "nvmlSystemGetProcessName",
                        lambda m, pid=pid: m.nvmlSystemGetProcessName(pid),
                        _basename,
                    ),
                )
            )
        return Observed.collected(
            tuple(sorted(processes, key=lambda p: p.pid)),
            source=Source.NVML,
            method=listed.method or "",
        )

    # --- helpers --------------------------------------------------------------------------

    def _read(
        self, function: str, call: Callable[[Any], Any], convert=lambda v: v, **kw
    ) -> Observed:
        return self._session._read(function, call, convert, **kw)

    def _with_fallback(
        self, *candidates: tuple[str, Callable[[Any], Any]], convert=lambda v: v, unit=None
    ) -> Observed:
        """The first variant the binding and driver support; older ones on
        `FUNCTION_NOT_FOUND` / version mismatch or a binding without the newer call."""
        outcome: Observed | None = None
        for function, call in candidates:
            try:
                outcome = self._read(function, call, convert, unit=unit)
            except AttributeError:  # binding too old to have this call
                continue
            if (
                outcome.is_collected
                or outcome.error is None
                or outcome.error.code not in _OLDER_VARIANT
            ):
                return outcome
        assert outcome is not None
        return outcome

    def _field(
        self, outcome: Observed, attribute: str, convert: Callable[[Any], Any], *, unit=None
    ) -> Observed:
        """One field of a struct read: `None` is NVML_VALUE_NOT_AVAILABLE, a missing or
        unusable field a parse error."""
        raw = getattr(outcome.value, attribute, _MISSING)
        if raw is None:
            return Observed.failed(
                CollectionStatus.NOT_SUPPORTED,
                code=VALUE_NOT_AVAILABLE,
                message=f"NVML reported no {attribute}",
                source=Source.NVML,
                method=outcome.method,
                unit=unit,
            )
        try:
            if raw is _MISSING:
                raise ValueError(f"the NVML struct has no {attribute}")
            value = convert(raw)
        except (TypeError, ValueError) as exc:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=PARSE_ERROR,
                message=f"unusable {attribute}: {exc}",
                source=Source.NVML,
                method=outcome.method,
                unit=unit,
            )
        return Observed.collected(value, source=Source.NVML, method=outcome.method or "", unit=unit)


# --- conversions --------------------------------------------------------------------------


def _text(value: Any) -> str:
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"expected text, got {value!r}")
    return text.strip()


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"expected a device count, got {value!r}")
    return value


def _bytes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"expected a byte count, got {value!r}")
    return value


def _percent(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"expected a percentage, got {value!r}")
    return value


def _watts(milliwatts: Any) -> float:
    if isinstance(milliwatts, bool) or not isinstance(milliwatts, int) or milliwatts < 0:
        raise ValueError(f"expected milliwatts, got {milliwatts!r}")
    return milliwatts / 1000


def _hex32(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError(f"expected a 32-bit id, got {value!r}")
    return f"0x{value:08x}"


def _cuda_version(value: Any) -> str:
    """NVML encodes CUDA versions as 1000 × major + 10 × minor (12040 → "12.4")."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"expected an encoded CUDA version, got {value!r}")
    return f"{value // 1000}.{value % 1000 // 10}"


def _basename(value: Any) -> str:
    """Process names are basenames only (NFR-08): a path never enters the report."""
    text = _text(value)
    return text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or text
