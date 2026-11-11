"""Driver collectors (architecture §4.4, DEC-46): the `DriverInfo` section, in two runs.

- **`driver`** (after `nvml`) reads the NVML facts from the core's shared session: binding
  metadata, `init_result` (`nvml_version`), the driver version, the newest CUDA version the
  driver supports, and the `libcuda` version (`cuda_driver_library`). It never opens NVML.
- **`nvidia_smi`** (after `runtime`) adds the `nvidia-smi` evidence: its location and the
  header's driver and CUDA versions. It reuses the adapter's cached query and header, so
  nothing runs twice in one inspection.

**Precedence.** `driver_version`: NVML, else the `nvidia-smi` query's `driver_version`, else
the `nvidia-smi` header. `driver_supported_cuda_version`: NVML, else the header (with the
fixed note). The `nvidia_smi_*` fields always hold what `nvidia-smi` itself said, so a
disagreement between the tools stays visible, and is noted on the `nvidia-smi` field.

A field NVML or `nvidia-smi` couldn't read is field-level evidence (the run is `PARTIAL` for
a failure status); a host without them (`NOT_DETECTED`, `NOT_SUPPORTED`) is a fact, not a
failure. Neither run emits capability records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from gpu_system_inspector.adapters import Clock, NvidiaSmiAdapter, NvmlAdapter, NvmlSession
from gpu_system_inspector.collectors.system import FAILURE_STATUSES
from gpu_system_inspector.models import (
    DRIVER_SUPPORTED_CUDA_NOTE,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DriverInfo,
    ErrorDetail,
    Observed,
)

DRIVER_REPORT_PATH = "/driver"
NVIDIA_SMI_REPORT_PATH = "/driver/nvidia_smi_path"
NVIDIA_SMI_VERSION_REPORT_PATH = "/driver/nvidia_smi_driver_version"
SMI_PENDING_NOTE = "Filled by the nvidia_smi collector run."
_NVML_FIELDS = ("driver_version", "driver_supported_cuda_version", "cuda_driver_library")


class DriverCollector:
    name: ClassVar[str] = "driver"

    def __init__(self, nvml: NvmlAdapter, session: NvmlSession, clock: Clock) -> None:
        self._nvml = nvml
        self._session = session
        self._clock = clock

    def collect(self) -> CollectorOutcome[DriverInfo]:
        started = self._clock.monotonic_ms()
        session = self._session
        pending: Observed[str] = Observed.skipped(note=SMI_PENDING_NOTE)
        section = DriverInfo(
            nvml_binding=self._nvml.binding_version(),
            nvml_binding_distribution=self._nvml.binding_distribution(),
            nvml_version=session.init_result,
            driver_version=session.driver_version(),
            driver_supported_cuda_version=session.cuda_driver_version(),
            cuda_driver_library=session.cuda_driver_library(),
            nvidia_smi_path=pending,
            nvidia_smi_driver_version=pending,
            nvidia_smi_cuda_version=pending,
        )
        # Without a session every read repeats the `nvml` run's outcome, already reported.
        partial = session.init_result.is_collected and any(
            getattr(section, field).status in FAILURE_STATUSES for field in _NVML_FIELDS
        )
        return CollectorOutcome(section=section, run=_run(self.name, started, self._clock, partial))

    def unavailable(self, error: ErrorDetail) -> DriverInfo:
        return DriverInfo.unavailable(CollectionStatus.COLLECTION_FAILED, error)


class NvidiaSmiCollector:
    name: ClassVar[str] = "nvidia_smi"

    def __init__(self, nvidia_smi: NvidiaSmiAdapter, clock: Clock, *, driver: DriverInfo) -> None:
        self._nvidia_smi = nvidia_smi
        self._clock = clock
        self._driver = driver

    def collect(self) -> CollectorOutcome[DriverInfo]:
        started = self._clock.monotonic_ms()
        smi = self._nvidia_smi
        path = smi.locate()
        header = smi.version()
        errors: list[CollectionError] = []
        if path.status in FAILURE_STATUSES:
            errors.append(_smi_error(path, NVIDIA_SMI_REPORT_PATH))
        elif path.is_collected and header.driver_version.status in FAILURE_STATUSES:
            errors.append(_smi_error(header.driver_version, NVIDIA_SMI_VERSION_REPORT_PATH))
        nvml = self._driver
        driver_version = nvml.driver_version
        if not driver_version.is_collected and path.is_collected:
            driver_version = self._queried_driver_version() or header.driver_version
            if not driver_version.is_collected:
                driver_version = nvml.driver_version
        cuda = nvml.driver_supported_cuda_version
        if not cuda.is_collected and header.cuda_version.is_collected:
            cuda = replace(header.cuda_version, note=DRIVER_SUPPORTED_CUDA_NOTE)
        section = replace(
            nvml,
            driver_version=driver_version,
            driver_supported_cuda_version=cuda,
            nvidia_smi_path=path,
            nvidia_smi_driver_version=_compared(header.driver_version, nvml.driver_version),
            nvidia_smi_cuda_version=_compared(
                header.cuda_version, nvml.driver_supported_cuda_version
            ),
        )
        return CollectorOutcome(
            section=section,
            run=_run(self.name, started, self._clock, bool(errors)),
            errors=tuple(errors),
        )

    def unavailable(self, error: ErrorDetail) -> DriverInfo:
        """The guard's fallback: the NVML facts came from the `driver` run, so they're kept."""
        failed: Observed[str] = Observed(
            value=None,
            status=CollectionStatus.COLLECTION_FAILED,
            source=None,
            method=None,
            error=error,
        )
        return replace(
            self._driver,
            nvidia_smi_path=failed,
            nvidia_smi_driver_version=failed,
            nvidia_smi_cuda_version=failed,
        )

    def _queried_driver_version(self) -> Observed[str] | None:
        """The first collected `driver_version` of the cached `--query-gpu` rows."""
        rows = self._nvidia_smi.query_gpus()
        for row in rows.value or ():
            value = row.values.get("driver_version")
            if value is not None and value.is_collected:
                return value
        return None


def _compared(smi: Observed[str], nvml: Observed[str]) -> Observed[str]:
    if smi.is_collected and nvml.is_collected and smi.value != nvml.value and smi.note is None:
        return replace(smi, note=f"Differs from NVML, which reports {nvml.value}.")
    return smi


def _run(name: str, started: int, clock: Clock, partial: bool) -> CollectorRun:
    return CollectorRun(
        collector=name,
        status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
        duration_ms=max(0, clock.monotonic_ms() - started),
    )


def _smi_error(outcome: Observed[str], report_path: str) -> CollectionError:
    assert outcome.error is not None
    return CollectionError(
        collector=NvidiaSmiCollector.name,
        device_ref=None,
        status=outcome.status,
        error_code=outcome.error.code,
        message=outcome.error.message,
        report_path=report_path,
    )
