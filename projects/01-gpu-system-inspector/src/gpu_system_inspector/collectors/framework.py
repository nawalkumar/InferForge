"""PyTorch collector (architecture §4.7, IMP-K06, DEC-48): `FrameworkInfo.pytorch`.

It turns the `TorchProbe`'s answers into `PyTorchInfo`, keeping apart what PyTorch's
facts are: installed (`distribution_version`), importable and which build (`version`,
`cuda_build`, `build_cuda_version`, `rocm_build`, `cudnn_version`), what PyTorch says about
CUDA at runtime (`cuda_available`, its warnings) and which devices it enumerates. None of
them shows that CUDA code runs: the probe executes nothing on a GPU. The build's CUDA
version is neither the toolkit (`cuda_environment`) nor the driver's (`driver`).

- `include_pytorch` false → `SKIPPED`, nothing read or run; run `COMPLETED`.
- No `torch` distribution → `NOT_INSTALLED`, no probe; run `COMPLETED`.
- Metadata unreadable → `COLLECTION_FAILED`; run `PARTIAL`, error at
  `/frameworks/pytorch/distribution_version`.
- The probe failed (import, timeout, exit, protocol) → `COLLECTION_FAILED` with the
  distribution version kept; run `PARTIAL`, error at `/frameworks/pytorch`.
- Devices couldn't be enumerated → `COLLECTED` with `device_count` failed; run `PARTIAL`,
  error at `/frameworks/pytorch/devices`.

Device correlation with NVML is the correlation run's (IMP-K07, DEC-49): this collector
leaves `correlated_device_ref` `NOT_APPLICABLE` for it to set, and matches nothing. The
collector owns no capability records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from gpu_system_inspector.adapters import (
    Clock,
    TorchProbe,
    TorchProbeDevice,
    TorchProbeReport,
)
from gpu_system_inspector.adapters.outcomes import TIMEOUT
from gpu_system_inspector.models import (
    ALLOCATOR_NOT_APPLICABLE_NOTE,
    ALLOCATOR_SCOPE_NOTE,
    BUILD_CUDA_NOTE,
    FIXTURE_METHOD_PREFIX,
    PROBE_PROTOCOL_VERSION,
    AllocatorStats,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    FrameworkInfo,
    InspectionOptions,
    Observed,
    ProbeInfo,
    ProbeMode,
    PyTorchInfo,
    Source,
    TorchDeviceInfo,
)

PYTORCH_REPORT_PATH = "/frameworks/pytorch"
CPU_ONLY_NOTE = "CPU-only PyTorch build"
NON_CUDA_NOTE = "Non-CUDA (ROCm) PyTorch build"
ALLOCATOR_NOT_INITIALISED_NOTE = (
    "CUDA isn't initialised in the calling process, so it has no allocator statistics."
)
# The interim value: the correlation run (IMP-K07) always replaces it (DEC-49 (5)).
CORRELATION_PENDING_NOTE = "Set by the correlation run (DEC-09)"
ATTRIBUTE_MISSING = "ATTRIBUTE_MISSING"
DEVICE_ENUMERATION_FAILED = "DEVICE_ENUMERATION_FAILED"
CUDNN_NOT_AVAILABLE = "CUDNN_NOT_AVAILABLE"
# TorchProbeDevice attribute → (TorchDeviceInfo field, method, unit).
DEVICE_FIELDS = {
    "name": ("name", "get_device_name", None),
    "capability": ("compute_capability", "get_device_capability", None),
    "total_memory": ("total_memory_bytes", "get_device_properties().total_memory", "bytes"),
    "multiprocessor_count": (
        "multiprocessor_count",
        "get_device_properties().multi_processor_count",
        None,
    ),
    "uuid": ("uuid", "get_device_properties().uuid", None),
    "pci_bus_id": ("pci_bus_id", "get_device_properties().pci", None),
}
ALLOCATOR_FIELDS = {
    "allocated": ("allocated_bytes", "torch.cuda.memory_allocated"),
    "reserved": ("reserved_bytes", "torch.cuda.memory_reserved"),
    "peak_allocated": ("peak_allocated_bytes", "torch.cuda.max_memory_allocated"),
    "peak_reserved": ("peak_reserved_bytes", "torch.cuda.max_memory_reserved"),
}


class FrameworkCollector:
    name: ClassVar[str] = "pytorch"

    def __init__(self, probe: TorchProbe, clock: Clock, *, options: InspectionOptions) -> None:
        self._probe = probe
        self._clock = clock
        self._options = options
        synthetic = getattr(probe, "provenance", DataOrigin.LIVE) is DataOrigin.FIXTURE
        self._prefix = FIXTURE_METHOD_PREFIX if synthetic else ""

    def collect(self) -> CollectorOutcome[FrameworkInfo]:
        started = self._clock.monotonic_ms()
        pytorch, errors = self._pytorch()
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if errors else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=FrameworkInfo(pytorch=pytorch), run=run, errors=errors)

    def unavailable(self, error: ErrorDetail) -> FrameworkInfo:
        return FrameworkInfo(
            pytorch=PyTorchInfo.unavailable(
                CollectionStatus.COLLECTION_FAILED, error, probe=self._probe_info(None)
            )
        )

    # --- the section ----------------------------------------------------------------------

    def _pytorch(self) -> tuple[PyTorchInfo, tuple[CollectionError, ...]]:
        if not self._options.include_pytorch:
            return PyTorchInfo.unavailable(
                CollectionStatus.SKIPPED, None, probe=self._probe_info(None)
            ), ()
        distribution = self._probe.distribution_version()
        if not distribution.is_collected:
            assert distribution.error is not None
            if distribution.status is CollectionStatus.NOT_INSTALLED:
                return PyTorchInfo.unavailable(
                    CollectionStatus.NOT_INSTALLED, distribution.error, probe=self._probe_info(None)
                ), ()
            section = replace(
                PyTorchInfo.unavailable(
                    CollectionStatus.COLLECTION_FAILED,
                    distribution.error,
                    probe=self._probe_info(None),
                ),
                distribution_version=distribution,
            )
            return section, (
                _error(
                    distribution, distribution.error, f"{PYTORCH_REPORT_PATH}/distribution_version"
                ),
            )
        timeout = self._options.pytorch_timeout_s
        outcome = self._probe.probe(timeout)
        if not outcome.is_collected:
            assert outcome.error is not None
            error = outcome.error
            if error.code == TIMEOUT:
                error = ErrorDetail(
                    code=TIMEOUT, message=f"PyTorch probe exceeded pytorch_timeout_s ({timeout} s)"
                )
            section = replace(
                PyTorchInfo.unavailable(
                    CollectionStatus.COLLECTION_FAILED, error, probe=self._probe_info(None)
                ),
                distribution_version=distribution,
            )
            return section, (_error(outcome, error, PYTORCH_REPORT_PATH),)
        assert outcome.value is not None
        return self._normalise(outcome.value, distribution)

    def _normalise(
        self, report: TorchProbeReport, distribution: Observed[str]
    ) -> tuple[PyTorchInfo, tuple[CollectionError, ...]]:
        ok = self._ok
        non_cuda_note = CPU_ONLY_NOTE if report.hip_version is None else NON_CUDA_NOTE
        if report.cuda_build:
            build = ok(report.build_cuda_version, "torch.version.cuda", note=BUILD_CUDA_NOTE)
            cudnn = (
                ok(report.cudnn_version, "torch.backends.cudnn.version()")
                if report.cudnn_version is not None
                else self._failed(
                    CollectionStatus.NOT_DETECTED,
                    CUDNN_NOT_AVAILABLE,
                    "torch.backends.cudnn.version() returned None",
                    "torch.backends.cudnn.version()",
                )
            )
        else:
            build = cudnn = Observed.not_applicable(note=non_cuda_note, source=Source.PYTORCH)
        errors: tuple[CollectionError, ...] = ()
        devices: tuple[TorchDeviceInfo, ...] = ()
        if report.device_error is not None:
            count = self._failed(
                CollectionStatus.COLLECTION_FAILED,
                DEVICE_ENUMERATION_FAILED,
                report.device_error,
                "torch.cuda.get_device_properties",
            )
            assert count.error is not None
            errors = (_error(count, count.error, f"{PYTORCH_REPORT_PATH}/devices"),)
        elif report.device_count is None or report.device_count != len(report.devices):
            count = self._failed(
                CollectionStatus.COLLECTION_FAILED,
                DEVICE_ENUMERATION_FAILED,
                f"the probe counted {report.device_count} devices but listed {len(report.devices)}",
                "torch.cuda.device_count()",
            )
            assert count.error is not None
            errors = (_error(count, count.error, f"{PYTORCH_REPORT_PATH}/devices"),)
        else:
            count = ok(report.device_count, "torch.cuda.device_count()")
            devices = tuple(self._device(d) for d in report.devices)
        section = PyTorchInfo(
            status=CollectionStatus.COLLECTED,
            distribution_version=distribution,
            version=ok(report.torch_version, "torch.__version__"),
            cuda_build=ok(report.cuda_build, "torch.version.cuda"),
            build_cuda_version=build,
            rocm_build=ok(report.hip_version is not None, "torch.version.hip"),
            cudnn_version=cudnn,
            cuda_available=ok(report.cuda_available, "torch.cuda.is_available()"),
            cuda_init_warnings=report.cuda_init_warnings,
            device_count=count,
            devices=devices,
            allocator=self._allocator(report),
            probe=self._probe_info(report.duration_ms),
        )
        return section, errors

    def _device(self, device: TorchProbeDevice) -> TorchDeviceInfo:
        fields: dict[str, Observed] = {}
        for attribute, (field, method, unit) in DEVICE_FIELDS.items():
            value = getattr(device, attribute)
            fields[field] = (
                self._ok(value, method, unit=unit)
                if value is not None
                else self._failed(
                    CollectionStatus.NOT_SUPPORTED,
                    ATTRIBUTE_MISSING,
                    f"{attribute} is not exposed by this PyTorch version",
                    method,
                    unit=unit,
                )
            )
        correlation: Observed[str] = Observed.not_applicable(
            note=CORRELATION_PENDING_NOTE, source=Source.INSPECTOR
        )
        return TorchDeviceInfo(
            torch_index=device.index, correlated_device_ref=correlation, **fields
        )

    def _allocator(self, report: TorchProbeReport) -> Observed:
        if self._options.pytorch_probe_mode is ProbeMode.SUBPROCESS:
            return Observed.not_applicable(
                note=ALLOCATOR_NOT_APPLICABLE_NOTE, source=Source.PYTORCH
            )
        if report.allocator is None:
            return Observed.not_applicable(
                note=ALLOCATOR_NOT_INITIALISED_NOTE, source=Source.PYTORCH
            )
        stats = tuple(
            AllocatorStats(
                torch_index=sample.index,
                **{
                    field: self._ok(
                        getattr(sample, attribute), method, unit="bytes", note=ALLOCATOR_SCOPE_NOTE
                    )
                    for attribute, (field, method) in ALLOCATOR_FIELDS.items()
                },
            )
            for sample in report.allocator
        )
        return self._ok(stats, "torch.cuda memory statistics", note=ALLOCATOR_SCOPE_NOTE)

    # --- helpers --------------------------------------------------------------------------

    def _probe_info(self, duration_ms: int | None) -> ProbeInfo:
        return ProbeInfo(
            mode=self._options.pytorch_probe_mode,
            protocol_version=PROBE_PROTOCOL_VERSION,
            duration_ms=duration_ms,
        )

    def _ok(
        self, value: object, method: str, *, unit: str | None = None, note: str | None = None
    ) -> Observed:
        return Observed.collected(
            value, source=Source.PYTORCH, method=self._prefix + method, unit=unit, note=note
        )

    def _failed(
        self,
        status: CollectionStatus,
        code: str,
        message: str,
        method: str,
        *,
        unit: str | None = None,
    ) -> Observed:
        return Observed.failed(
            status,
            code=code,
            message=message,
            source=Source.PYTORCH,
            method=self._prefix + method,
            unit=unit,
        )


def _error(outcome: Observed, error: ErrorDetail, report_path: str) -> CollectionError:
    return CollectionError(
        collector=FrameworkCollector.name,
        device_ref=None,
        status=outcome.status,
        error_code=error.code,
        message=error.message,
        report_path=report_path,
    )
