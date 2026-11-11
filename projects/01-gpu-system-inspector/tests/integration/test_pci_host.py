"""Host integration (Stage 6.5): the production PCI adapter and collector on the machine
running the tests, with the environment classified by the production platform adapter.

On every non-native environment (WSL1, WSL2, UNKNOWN Linux, Windows, macOS, other) the scan
must be gated: `NOT_SUPPORTED`, the policy reason, the PCI-only capability record, and no
access to `/sys` at all (a guard fails the test on any `/sys` path). A native Linux host is
skipped: scanning its real sysfs is hardware validation (HW-02), not part of this suite.
"""

import builtins
import os

import jsonschema
import pytest

from gpu_system_inspector.adapters import OsPlatformAdapter, PciSysfsAdapter, SystemClock
from gpu_system_inspector.adapters.pci_sysfs import (
    PCI_SCAN_UNSUPPORTED_CODES,
    PCI_SCAN_UNSUPPORTED_REASONS,
    SCAN_METHOD,
    pci_scan_eligible,
)
from gpu_system_inspector.collectors import (
    GPU_PCI_SCAN,
    NVIDIA_ALL,
    PciCollector,
    unsupported_capabilities_for,
)
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    LinuxEnvironment,
    PlatformProfile,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

pytestmark = pytest.mark.host


@pytest.fixture
def sysfs_attempts(monkeypatch) -> list[str]:
    """Records, and refuses, any attempt to open, list or resolve a path under `/sys`."""
    attempts: list[str] = []

    def guard(function):
        def guarded(path, *args, **kwargs):
            if os.fspath(path).startswith("/sys"):
                attempts.append(os.fspath(path))
                raise AssertionError(f"the PCI host test touched {path}")
            return function(path, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(os, "listdir", guard(os.listdir))
    monkeypatch.setattr(os, "readlink", guard(os.readlink))
    return attempts


@pytest.fixture
def host_pci(sysfs_attempts):
    """The host's classification and the production PCI collector's outcome on it."""
    system = OsPlatformAdapter().system_info()
    environment = system.linux_environment.value
    if pci_scan_eligible(environment):
        pytest.skip("native Linux host: a real PCI scan is hardware validation (HW-02)")
    scanner = PciSysfsAdapter(environment=environment)  # default: the real sysfs reader
    collector = PciCollector(
        scanner, SystemClock(), environment=environment, profile=system.platform_profile
    )
    return system, scanner, collector.collect()


def test_scan_is_gated_without_touching_sysfs(host_pci, sysfs_attempts):
    system, scanner, outcome = host_pci
    environment = system.linux_environment.value
    section = outcome.section
    assert sysfs_attempts == []
    assert scanner.provenance is DataOrigin.LIVE
    assert section.status is CollectionStatus.NOT_SUPPORTED and section.value is None
    assert section.error.code == PCI_SCAN_UNSUPPORTED_CODES[environment]
    assert section.error.message == PCI_SCAN_UNSUPPORTED_REASONS[environment]
    assert section.method == SCAN_METHOD  # live: no fixture prefix


def test_collector_completes_with_only_the_pci_capability(host_pci):
    system, _, outcome = host_pci
    assert outcome.run.collector == "pci"
    assert outcome.run.status is CollectorRunStatus.COMPLETED and outcome.run.duration_ms >= 0
    assert outcome.errors == ()
    policy = unsupported_capabilities_for(system.linux_environment.value, system.platform_profile)
    assert outcome.unsupported == tuple(u for u in policy if u.capability == GPU_PCI_SCAN)
    assert len(outcome.unsupported) == 1
    assert all(u.capability != NVIDIA_ALL for u in outcome.unsupported)


def test_wsl2_is_never_scanned_as_native_linux(host_pci):
    system, _, outcome = host_pci
    if system.linux_environment.value is not LinuxEnvironment.WSL2:
        pytest.skip("this host isn't classified as WSL2")
    assert system.platform_profile is PlatformProfile.WSL2
    assert outcome.section.error.code == "PLATFORM_NOT_SUPPORTED"
    assert "WSL2" in outcome.unsupported[0].reason


def test_host_pci_outcome_serialises_against_the_schema(host_pci):
    _, _, outcome = host_pci
    defs = load_report_schema()["$defs"]

    def check(value, name):
        jsonschema.validate(to_json_dict(value), {"$ref": f"#/$defs/{name}", "$defs": defs})

    check(outcome.section, "ObservedPciObservationList")
    check(outcome.run, "CollectorRun")
    for record in outcome.unsupported:
        check(record, "UnsupportedCapability")
