"""The inspection core (architecture §4.2, §4.11, DEC-44): ordering, guard, aggregation.

`Inspector(adapters).run(options)` runs the implemented collectors in a fixed order
(`system`, `pci`, `nvml`, `driver`, `gpu_hardware`, `runtime`, `nvidia_smi`,
`cuda_environment`, `pytorch`, `correlation`), each under the guard, owns the single NVML
session, assembles the `InspectionSnapshot` and hands it to report assembly.

- **Environment.** The PCI scanner and both environment-aware collectors get the
  environment resolved from the system section, fail-closed (`resolve_linux_environment`).
  A `pci_scanner_factory` builds the scanner only after that; without one, the adapter set's
  pre-built scanner is used, and `PciCollector` rejects it if it was built for another
  environment.
- **One PCI scan.** `PciCollector` scans; `GpuHardwareCollector` only receives its section.
- **NVML session.** The `nvml` step opens it and reports `init_result`; `driver`,
  `gpu_hardware` and `runtime` share it; the core closes it after them, on every path.
- **One `nvidia-smi`.** The adapter caches its query and header, so the GPU hardware
  fallback, the runtime fallback and the `nvidia_smi` run never execute it twice.
- **Driver section.** `driver` reads the NVML facts; `nvidia_smi` completes the same
  section with the `nvidia-smi` evidence (DEC-46), and its result is the report's.
- **CUDA environment.** `cuda_environment` runs after the session is closed: the toolkit
  locator (DEC-47) uses files and `nvcc`, never NVML.
- **PyTorch.** `pytorch` runs last (DEC-48): the probe is a separate process by default, so
  the inspector's own process never creates a CUDA context, and nothing touches CUDA before
  this step.
- **Correlation.** `correlation` runs last (DEC-49): it relates the PyTorch devices to the
  NVML inventory from the two sections already collected (DEC-09), probing nothing again;
  its result is the report's PyTorch section.
- **Readiness and diagnostics.** After collection, Mode B runs the readiness engine
  (IMP-R01), then the diagnostics engine evaluates D01–D08 over the snapshot (IMP-D00); a
  rule that raises adds a `diagnostics.Dxx` collection error, never a crash (DEC-50).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import TracebackType
from typing import ClassVar, Self

from gpu_system_inspector.adapters import (
    AdapterSet,
    Clock,
    NvmlAdapter,
    NvmlDevice,
    NvmlSession,
    PciScanner,
)
from gpu_system_inspector.collectors import (
    CORRELATION_REPORT_PATH,
    CUDA_ENVIRONMENT_REPORT_PATH,
    DRIVER_REPORT_PATH,
    GPU_REPORT_PATH,
    NVIDIA_SMI_REPORT_PATH,
    PCI_REPORT_PATH,
    PYTORCH_REPORT_PATH,
    RUNTIME_REPORT_PATH,
    CorrelationCollector,
    CudaEnvironmentCollector,
    DriverCollector,
    FrameworkCollector,
    GpuHardwareCollector,
    NvidiaSmiCollector,
    PciCollector,
    RuntimeCollector,
    SystemCollector,
)
from gpu_system_inspector.core.environment import resolve_linux_environment
from gpu_system_inspector.core.guard import run_guarded
from gpu_system_inspector.diagnostics import DiagnosticInput, DiagnosticsEngine
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
    LinuxEnvironment,
    Observed,
)
from gpu_system_inspector.readiness import ReadinessEngine
from gpu_system_inspector.reporting import assemble_report

PciScannerFactory = Callable[[LinuxEnvironment | None], PciScanner]

SYSTEM_REPORT_PATH = "/system"
NVML_REPORT_PATH = "/driver/nvml_version"
NVML_BINDING_REPORT_PATH = "/driver/nvml_binding"
# NVML outcomes that are a collection problem rather than a host fact (NOT_DETECTED: no
# driver; NOT_SUPPORTED: e.g. macOS). A missing binding is a broken installation (FX04a).
NVML_PROBLEM_STATUSES = frozenset(
    {
        CollectionStatus.NOT_INSTALLED,
        CollectionStatus.PERMISSION_DENIED,
        CollectionStatus.COLLECTION_FAILED,
        CollectionStatus.UNKNOWN,
    }
)
SCOPE_LIMITATION = (
    "Findings and readiness are derived from this single snapshot only. CUDA execution isn't "
    "checked: frameworks.pytorch.cuda_available is PyTorch's own answer, the probe runs no "
    "CUDA code, a CUDA Toolkit or nvcc in cuda_environment is an installation fact, and "
    "driver.cuda_driver_library is the libcuda version NVML reports; none of them shows that "
    "code runs on a GPU."
)


class Inspector:
    def __init__(
        self,
        adapters: AdapterSet,
        *,
        pci_scanner_factory: PciScannerFactory | None = None,
        limitations: Iterable[str] = (),
    ) -> None:
        if not isinstance(adapters, AdapterSet):
            raise TypeError(f"Inspector needs an AdapterSet, got {type(adapters).__name__}")
        self._adapters = adapters
        self._pci_scanner_factory = pci_scanner_factory
        self._limitations = tuple(limitations)

    def run(self, options: InspectionOptions | None = None) -> InspectionReport:
        options = options or InspectionOptions()
        clock = self._adapters.clock
        started = clock.monotonic_ms()
        generated_at = clock.now_utc()
        snapshot = self.collect(options)
        readiness = (
            ReadinessEngine().assess(snapshot, options.required_vram_gib)
            if options.required_vram_gib is not None
            else None
        )
        diagnostics = DiagnosticsEngine().evaluate(DiagnosticInput(snapshot, readiness, options))
        return assemble_report(
            snapshot,
            options=options,
            generated_at=generated_at,
            duration_ms=max(0, clock.monotonic_ms() - started),
            data_origin=self._adapters.provenance,
            limitations=(SCOPE_LIMITATION, *self._limitations),
            diagnostics=diagnostics.findings,
            readiness=readiness,
            errors=diagnostics.errors,
        )

    def collect(self, options: InspectionOptions | None = None) -> InspectionSnapshot:
        """Runs the collectors in order and aggregates their outcomes (no report metadata)."""
        options = options or InspectionOptions()
        adapters, clock = self._adapters, self._adapters.clock
        system = run_guarded(
            SystemCollector(adapters.platform, clock), clock, report_path=SYSTEM_REPORT_PATH
        )
        environment = resolve_linux_environment(system.section)
        profile = system.section.platform_profile
        scanner = (
            self._pci_scanner_factory(environment)
            if self._pci_scanner_factory is not None
            else adapters.pci
        )
        pci = run_guarded(
            PciCollector(scanner, clock, environment=environment, profile=profile),
            clock,
            report_path=PCI_REPORT_PATH,
        )
        step = NvmlSessionStep(adapters.nvml, clock)
        nvml = run_guarded(step, clock, report_path=NVML_REPORT_PATH)
        session = step.session or FailedNvmlSession(nvml.section)
        try:
            driver = run_guarded(
                DriverCollector(adapters.nvml, session, clock),
                clock,
                report_path=DRIVER_REPORT_PATH,
            )
            gpu = run_guarded(
                GpuHardwareCollector(
                    session,
                    adapters.nvidia_smi,
                    clock,
                    pci=pci.section,
                    environment=environment,
                    profile=profile,
                ),
                clock,
                report_path=GPU_REPORT_PATH,
            )
            runtime = run_guarded(
                RuntimeCollector(
                    session,
                    adapters.nvidia_smi,
                    clock,
                    inventory=gpu.section,
                    profile=profile,
                    include_processes=options.include_processes,
                ),
                clock,
                report_path=RUNTIME_REPORT_PATH,
            )
        finally:
            session.close()
        smi = run_guarded(
            NvidiaSmiCollector(adapters.nvidia_smi, clock, driver=driver.section),
            clock,
            report_path=NVIDIA_SMI_REPORT_PATH,
        )
        cuda = run_guarded(
            CudaEnvironmentCollector(adapters.toolkit, adapters.platform, clock),
            clock,
            report_path=CUDA_ENVIRONMENT_REPORT_PATH,
        )
        pytorch = run_guarded(
            FrameworkCollector(adapters.torch_probe, clock, options=options),
            clock,
            report_path=PYTORCH_REPORT_PATH,
        )
        correlation = run_guarded(
            CorrelationCollector(
                clock,
                inventory=gpu.section,
                frameworks=pytorch.section,
                synthetic=adapters.provenance is DataOrigin.FIXTURE,
            ),
            clock,
            report_path=CORRELATION_REPORT_PATH,
        )
        outcomes: tuple[CollectorOutcome[object], ...] = (
            system,
            pci,
            nvml,
            driver,
            gpu,
            runtime,
            smi,
            cuda,
            pytorch,
            correlation,
        )
        return InspectionSnapshot(
            system=system.section,
            gpu=gpu.section,
            driver=smi.section,
            cuda_environment=cuda.section,
            frameworks=correlation.section,
            runtime=runtime.section,
            collection_errors=tuple(e for o in outcomes for e in o.errors),
            unsupported_capabilities=tuple(u for o in outcomes for u in o.unsupported),
            collector_runs=tuple(o.run for o in outcomes),
        )


# --- the NVML session step ----------------------------------------------------------------


class NvmlSessionStep:
    """The `nvml` run: the core opens the one NVML session here (collectors never do) and
    reports its accessibility. The section is `init_result`; the session stays open for the
    NVML-backed collectors until the core closes it."""

    name: ClassVar[str] = "nvml"

    def __init__(self, nvml: NvmlAdapter, clock: Clock) -> None:
        self._nvml = nvml
        self._clock = clock
        self.session: NvmlSession | None = None

    def collect(self) -> CollectorOutcome[Observed[str]]:
        started = self._clock.monotonic_ms()
        self.session = self._nvml.open()
        init = self.session.init_result
        errors: tuple[CollectionError, ...] = ()
        if init.status in NVML_PROBLEM_STATUSES:
            assert init.error is not None
            binding = init.status is CollectionStatus.NOT_INSTALLED
            errors = (
                CollectionError(
                    collector=self.name,
                    device_ref=None,
                    status=init.status,
                    error_code=init.error.code,
                    message=init.error.message,
                    report_path=NVML_BINDING_REPORT_PATH if binding else NVML_REPORT_PATH,
                ),
            )
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if errors else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=init, run=run, errors=errors)

    def unavailable(self, error: ErrorDetail) -> Observed[str]:
        return Observed(
            value=None,
            status=CollectionStatus.COLLECTION_FAILED,
            source=None,
            method=None,
            error=error,
        )


class FailedNvmlSession:
    """Stands in for the session when opening it raised: every query reports that failure,
    so the GPU hardware collector still runs (and can use the `nvidia-smi` fallback)."""

    def __init__(self, failure: Observed[str]) -> None:
        self._failure = failure
        self._open = True

    @property
    def init_result(self) -> Observed[str]:
        return self._failure

    @property
    def is_open(self) -> bool:
        return self._open

    def driver_version(self) -> Observed[str]:
        return self._failure

    def cuda_driver_version(self) -> Observed[str]:
        return self._failure

    def cuda_driver_library(self) -> Observed[str]:
        return self._failure

    def device_count(self) -> Observed[int]:
        return self._failure  # type: ignore[return-value]

    def device(self, index: int) -> Observed[NvmlDevice]:
        return self._failure  # type: ignore[return-value]

    def close(self) -> None:
        self._open = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
