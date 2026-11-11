"""PCI collector (architecture §4.4, DEC-43): the `pci_observations` section and its run record.

The `PciScanner` does all sysfs access, eligibility gating and parsing; this collector only
maps its `PciScan` to the report's bookkeeping: run status, duration (from the injected
`Clock`), host-wide `CollectionError` records and the `GPU.PCI_SCAN` unsupported capability.
The scan's `Observed` value is passed through unchanged. Merging it into `GPUInventory` is
IMP-K02; `NVIDIA.ALL` is emitted by later stages, not here.
"""

from __future__ import annotations

from typing import ClassVar

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    Clock,
    PciDeviceFailure,
    PciFailureKind,
    PciScanner,
)
from gpu_system_inspector.adapters.pci_sysfs import (
    PCI_SCAN_UNSUPPORTED_CODES,
    PCI_SCAN_UNSUPPORTED_REASONS,
    pci_scan_eligible,
)
from gpu_system_inspector.collectors.capabilities import (
    GPU_PCI_SCAN,
    unsupported_capabilities_for,
)
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    ErrorDetail,
    LinuxEnvironment,
    Observed,
    PciObservation,
    PlatformProfile,
)

PCI_REPORT_PATH = "/gpu/pci_observations"
# Whole-scan outcomes that are collection problems. NOT_DETECTED (no PCI sysfs root) and
# NOT_SUPPORTED (ineligible environment) describe the host, so the run stays COMPLETED.
SCAN_FAILURE_STATUSES = frozenset(
    {CollectionStatus.PERMISSION_DENIED, CollectionStatus.COLLECTION_FAILED}
)
HOST_STATUSES = frozenset({CollectionStatus.NOT_DETECTED, CollectionStatus.NOT_SUPPORTED})
# The device was listed but couldn't be observed, so it's missing from the observations.
OMITTED_DEVICE_KINDS = frozenset(
    {
        PciFailureKind.INVALID_ADDRESS,
        PciFailureKind.ATTRIBUTE_UNREADABLE,
        PciFailureKind.ATTRIBUTE_MALFORMED,
    }
)

PciSection = Observed[tuple[PciObservation, ...]]


class PciCollector:
    name: ClassVar[str] = "pci"

    def __init__(
        self,
        scanner: PciScanner,
        clock: Clock,
        *,
        environment: LinuxEnvironment | None,
        profile: PlatformProfile,
    ) -> None:
        # Raises AdapterUsageError for an inconsistent environment/profile pair.
        policy = unsupported_capabilities_for(environment, profile)
        self._scanner = scanner
        self._clock = clock
        self._environment = environment
        self._unsupported = tuple(u for u in policy if u.capability == GPU_PCI_SCAN)

    def collect(self) -> CollectorOutcome[PciSection]:
        started = self._clock.monotonic_ms()
        scan = self._scanner.scan()
        section = scan.observations
        self._check_matches_environment(section)
        errors: tuple[CollectionError, ...] = ()
        unsupported = ()
        partial = False
        if section.is_collected:
            errors = tuple(
                _device_error(f) for f in scan.failures if f.kind in OMITTED_DEVICE_KINDS
            )
            partial = bool(scan.failures)
        elif section.status is CollectionStatus.NOT_SUPPORTED:
            unsupported = self._unsupported
        elif section.status in SCAN_FAILURE_STATUSES:
            assert section.error is not None
            errors = (_error(section.status, section.error.code, section.error.message),)
            partial = True
        elif section.status not in HOST_STATUSES:
            raise AdapterUsageError(f"a PCI scan can't report {section.status}")
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=section, run=run, errors=errors, unsupported=unsupported)

    def unavailable(self, error: ErrorDetail) -> PciSection:
        """The section the collector guard substitutes when `collect()` raised."""
        return Observed(
            value=None,
            status=CollectionStatus.COLLECTION_FAILED,
            source=None,
            method=None,
            error=error,
        )

    def _check_matches_environment(self, section: PciSection) -> None:
        """The scanner must have been built for the same environment (DEC-41, DEC-42)."""
        environment = self._environment
        if pci_scan_eligible(environment):
            if section.status is CollectionStatus.NOT_SUPPORTED:
                raise AdapterUsageError(
                    f"the PCI scanner reports NOT_SUPPORTED, but {environment} is eligible"
                )
            return
        expected = (
            PCI_SCAN_UNSUPPORTED_CODES[environment],
            PCI_SCAN_UNSUPPORTED_REASONS[environment],
        )
        actual = (section.error.code, section.error.message) if section.error else None
        if section.status is not CollectionStatus.NOT_SUPPORTED or actual != expected:
            raise AdapterUsageError(
                f"the PCI scanner reports {section.status}, but {environment or 'non-Linux'} "
                "must be NOT_SUPPORTED with the policy code and reason"
            )


def _device_error(failure: PciDeviceFailure) -> CollectionError:
    target = failure.attribute or "address"
    message = f"{failure.address} {target}: {failure.error.message}"
    return _error(failure.status, failure.error.code, message)


def _error(status: CollectionStatus, code: str, message: str) -> CollectionError:
    return CollectionError(
        collector=PciCollector.name,
        device_ref=None,
        status=status,
        error_code=code,
        message=message,
        report_path=PCI_REPORT_PATH,
    )
