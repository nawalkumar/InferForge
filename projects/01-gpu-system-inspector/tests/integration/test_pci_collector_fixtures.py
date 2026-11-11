"""Stage 6.4: the PCI collector reproduces every fixture's PCI outcome (section, `pci` run,
`pci` collection errors and the `GPU.PCI_SCAN` record), over the production adapter with a
synthetic sysfs and over the test kit's fakes. No real sysfs is touched."""

import dataclasses

import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeSysfsReader, FixedClock, sysfs_reader_for
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import PciSysfsAdapter
from gpu_system_inspector.adapters.pci_sysfs import PCI_SCAN_UNSUPPORTED_REASONS, pci_scan_eligible
from gpu_system_inspector.collectors import GPU_PCI_SCAN, NVIDIA_ALL, PciCollector
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    ErrorDetail,
    LinuxEnvironment,
)
from gpu_system_inspector.reporting import to_json_dict

S = CollectionStatus
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
# Fixture runs have fixed durations (5 ms × position in COLLECTOR_ORDER); `pci` is second.
PCI_RUN_STEP_MS = 10


def _expected(snapshot):
    """The fixture's own PCI outcome: section, run, errors and the PCI capability record."""
    (run,) = (r for r in snapshot.collector_runs if r.collector == "pci")
    errors = tuple(e for e in snapshot.collection_errors if e.collector == "pci")
    unsupported = tuple(
        u for u in snapshot.unsupported_capabilities if u.capability == GPU_PCI_SCAN
    )
    return snapshot.gpu.pci_observations, run, errors, unsupported


def _sysfs_for(snapshot) -> FakeSysfsReader:
    pci = snapshot.gpu.pci_observations
    if pci.is_collected:
        return sysfs_reader_for(pci.value)
    return FakeSysfsReader(failures={"list": AssertionError("an ineligible scan touched sysfs")})


def _collect(snapshot, scanner):
    system = snapshot.system
    collector = PciCollector(
        scanner,
        FixedClock(step_ms=PCI_RUN_STEP_MS),
        environment=system.linux_environment.value,
        profile=system.platform_profile,
    )
    return collector.collect()


def _production(snapshot):
    reader = _sysfs_for(snapshot)
    environment = snapshot.system.linux_environment.value
    return _collect(snapshot, PciSysfsAdapter(environment=environment, reader=reader)), reader


@ids
def test_collector_reproduces_every_fixture_pci_outcome(fixture):
    snapshot = fixture.snapshot
    outcome, reader = _production(snapshot)
    section, run, errors, unsupported = _expected(snapshot)
    assert outcome.section == section
    assert outcome.run == run
    assert outcome.errors == errors
    assert outcome.unsupported == unsupported
    assert all(u.capability != NVIDIA_ALL for u in outcome.unsupported)
    if not pci_scan_eligible(snapshot.system.linux_environment.value):
        assert reader.calls == []


@ids
def test_collector_output_rebuilds_the_fixture_snapshot(fixture):
    """Swapping the fixture's PCI parts for the collector's output changes nothing."""
    snapshot = fixture.snapshot
    outcome, _ = _production(snapshot)
    rebuilt = dataclasses.replace(
        snapshot,
        gpu=dataclasses.replace(snapshot.gpu, pci_observations=outcome.section),
        collection_errors=tuple(e for e in snapshot.collection_errors if e.collector != "pci")
        + outcome.errors,
        unsupported_capabilities=tuple(
            u for u in snapshot.unsupported_capabilities if u.capability != GPU_PCI_SCAN
        )
        + outcome.unsupported,
        collector_runs=tuple(
            outcome.run if r.collector == "pci" else r for r in snapshot.collector_runs
        ),
    )
    assert rebuilt == snapshot
    assert to_json_dict(rebuilt) == to_json_dict(snapshot)


@ids
def test_fake_scanner_and_production_adapter_agree(fixture):
    snapshot = fixture.snapshot
    production, _ = _production(snapshot)
    fake = _collect(snapshot, FakeAdapterSet.from_scenario(fixture).pci)
    assert fake == production


@ids
def test_fake_pci_collector_matches_the_real_collector(fixture):
    snapshot = fixture.snapshot
    fake = FakeAdapterSet.from_scenario(fixture).collectors["pci"]
    production, _ = _production(snapshot)
    assert fake.collect() == production  # section, run, errors and capability records
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    real = PciCollector(
        FakeAdapterSet.from_scenario(fixture).pci,
        FixedClock(),
        environment=snapshot.system.linux_environment.value,
        profile=snapshot.system.platform_profile,
    )
    assert fake.unavailable(error) == real.unavailable(error)


def test_named_fixture_outcomes():
    def outcome(fixture_id):
        return _production(FIXTURES_BY_ID[fixture_id].snapshot)[0]

    # FX03: native Linux, the scan ran and found only an Intel display controller.
    fx03 = outcome("FX03")
    (intel,) = fx03.section.value
    assert intel.vendor_id == "0x8086" and not intel.is_nvidia
    assert fx03.run.status is CollectorRunStatus.COMPLETED
    assert fx03.errors == () and fx03.unsupported == ()
    # FX17: an NVIDIA device bound to nouveau is observed, not absent and not unsupported.
    fx17 = outcome("FX17")
    (nvidia,) = fx17.section.value
    assert nvidia.is_nvidia and nvidia.kernel_driver.value == "nouveau"
    assert fx17.run.status is CollectorRunStatus.COMPLETED
    assert fx17.errors == () and fx17.unsupported == ()
    # WSL2 (FX04b, FX11), Windows (FX12), macOS (FX13): NOT_SUPPORTED with the PCI record only.
    for fixture_id, environment in (
        ("FX04b", LinuxEnvironment.WSL2),
        ("FX11", LinuxEnvironment.WSL2),
        ("FX12", None),
        ("FX13", None),
    ):
        result = outcome(fixture_id)
        assert result.section.status is S.NOT_SUPPORTED
        assert result.run.status is CollectorRunStatus.COMPLETED and result.errors == ()
        (record,) = result.unsupported
        assert record.capability == GPU_PCI_SCAN
        assert record.reason == PCI_SCAN_UNSUPPORTED_REASONS[environment]
    # FX13 lists NVIDIA.ALL too, but that record belongs to a later stage, not this collector.
    fx13 = FIXTURES_BY_ID["FX13"].snapshot
    assert any(u.capability == NVIDIA_ALL for u in fx13.unsupported_capabilities)


def test_fixture_catalogue_pci_coverage():
    """Which PCI outcomes the catalogue pins. Missing-root, empty and partial scans are pinned
    by no fixture; `unit/test_pci_collector.py` covers them synthetically."""
    kinds = set()
    for fixture in ALL_FIXTURES:
        section = fixture.snapshot.gpu.pci_observations
        if not section.is_collected:
            kinds.add(section.status.value)
        elif not section.value:
            kinds.add("COLLECTED_EMPTY")
        elif any(o.is_nvidia for o in section.value):
            kinds.add("COLLECTED_NVIDIA")
        else:
            kinds.add("COLLECTED_NON_NVIDIA")
    assert kinds == {"COLLECTED_NVIDIA", "COLLECTED_NON_NVIDIA", "NOT_SUPPORTED"}
    assert all(e.collector != "pci" for f in ALL_FIXTURES for e in f.snapshot.collection_errors)
    runs = {
        r.status for f in ALL_FIXTURES for r in f.snapshot.collector_runs if r.collector == "pci"
    }
    assert runs == {CollectorRunStatus.COMPLETED}
