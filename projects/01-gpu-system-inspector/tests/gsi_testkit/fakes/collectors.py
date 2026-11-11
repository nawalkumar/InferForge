"""Fake collectors implementing the `Collector` contract (`collectors.base`, DEC-44): each
returns its fixture section as a `CollectorOutcome` (the fixture's run, errors and owned
capability records) or raises the fixture's configured exception, so the guard can be tested
over every fixture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    CUDAEnvironmentInfo,
    DriverInfo,
    ErrorDetail,
    GPUInventory,
    Observed,
    PyTorchInfo,
    RuntimeMetrics,
    SystemInfo,
    UnsupportedCapability,
)

SectionT = TypeVar("SectionT")

# Which collector owns an unsupported-capability record (architecture §4.11).
CAPABILITY_OWNERS = {
    "GPU.PCI_SCAN": "pci",
    "NVIDIA.ALL": "gpu_hardware",
    "SYSTEM": "system",
    "RUNTIME": "runtime",
    "CUDA": "cuda_environment",
}


def capability_owner(record: UnsupportedCapability) -> str | None:
    return CAPABILITY_OWNERS.get(
        record.capability, CAPABILITY_OWNERS.get(record.capability.split(".")[0])
    )


@dataclass
class FakeCollector(Generic[SectionT]):
    name: str
    section: SectionT | None
    unavailable_factory: Callable[[ErrorDetail], SectionT]
    failure: BaseException | None = None
    run: CollectorRun | None = None
    errors: tuple[Any, ...] = ()
    unsupported: tuple[UnsupportedCapability, ...] = ()
    calls: int = field(default=0)

    def collect(self) -> CollectorOutcome[SectionT]:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        assert self.section is not None and self.run is not None
        return CollectorOutcome(
            section=self.section, run=self.run, errors=self.errors, unsupported=self.unsupported
        )

    def unavailable(self, error: ErrorDetail) -> SectionT:
        return self.unavailable_factory(error)


def fake_collectors(scenario: Any) -> dict[str, FakeCollector[Any]]:
    """Collectors for every snapshot section; a collector in `collector_failures` raises."""
    snapshot = scenario.snapshot
    failures = scenario.collector_failures
    failed = CollectionStatus.COLLECTION_FAILED
    probe = snapshot.frameworks.pytorch.probe
    sampled_at = {r.device_ref: r.sampled_at for r in snapshot.runtime}
    runs = {r.collector: r for r in snapshot.collector_runs}
    specs: dict[str, tuple[Any, Callable[[ErrorDetail], Any]]] = {
        "system": (
            snapshot.system,
            lambda e: SystemInfo.unavailable(
                failed, e, platform_profile=snapshot.system.platform_profile
            ),
        ),
        # The separate PCI collector (DEC-43). `gpu_hardware` returns the whole
        # `GPUInventory`, including `pci_observations`, as the production collector does.
        "pci": (
            snapshot.gpu.pci_observations,
            lambda e: Observed(value=None, status=failed, source=None, method=None, error=e),
        ),
        "gpu_hardware": (snapshot.gpu, lambda e: GPUInventory.unavailable(failed, e)),
        "driver": (snapshot.driver, lambda e: DriverInfo.unavailable(failed, e)),
        "cuda_environment": (
            snapshot.cuda_environment,
            lambda e: CUDAEnvironmentInfo.unavailable(failed, e),
        ),
        "pytorch": (
            snapshot.frameworks.pytorch,
            lambda e: PyTorchInfo.unavailable(failed, e, probe=probe),
        ),
        "runtime": (
            snapshot.runtime,
            lambda e: tuple(
                RuntimeMetrics.unavailable(ref, at, failed, e) for ref, at in sampled_at.items()
            ),
        ),
    }
    collectors = {}
    for name, (section, factory) in specs.items():
        run = runs[name]
        if run.status is CollectorRunStatus.FAILED:
            # The guard reports FAILED, never the collector; a FAILED fixture run is the
            # outcome of the configured failure.
            run = None
        collectors[name] = FakeCollector(
            name=name,
            section=None if name in failures else section,
            unavailable_factory=factory,
            failure=failures.get(name),
            run=run,
            errors=tuple(e for e in snapshot.collection_errors if e.collector == name),
            unsupported=tuple(
                u for u in snapshot.unsupported_capabilities if capability_owner(u) == name
            ),
        )
    return collectors
