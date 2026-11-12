"""IMP-K02 GPU hardware collector (DEC-44) over the fake NVML session and `nvidia-smi`:
fixture parity, per-device isolation, the fallback, the PCI hand-over and `NVIDIA.ALL`.
No NVML library, process or sysfs access."""

import dataclasses
import inspect

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeNvidiaSmi, FakeNvml, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import UnimplementedNvmlSession
from gpu_system_inspector.collectors import (
    FALLBACK_FIELD_UNAVAILABLE,
    GPU_PCI_SCAN,
    NVIDIA_ALL,
    GpuHardwareCollector,
)
from gpu_system_inspector.collectors.gpu_hardware import READINESS_MIG
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    GPUInventory,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    Source,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

S = CollectionStatus
R = CollectorRunStatus
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
# Fixture runs have fixed durations (5 ms × position in COLLECTOR_ORDER); gpu_hardware is 5th.
GPU_RUN_STEP_MS = 25
FX10 = FIXTURES_BY_ID["FX10"]


def _collector(snapshot, nvml=None, smi=None, *, clock=None, pci=None):
    nvml = nvml or FakeNvml.from_snapshot(snapshot)
    session = nvml.open()
    collector = GpuHardwareCollector(
        session,
        smi or FakeNvidiaSmi.from_snapshot(snapshot),
        clock or FixedClock(step_ms=GPU_RUN_STEP_MS),
        pci=pci if pci is not None else snapshot.gpu.pci_observations,
        environment=snapshot.system.linux_environment.value,
        profile=snapshot.system.platform_profile,
    )
    return collector, session


def _expected(snapshot):
    (run,) = (r for r in snapshot.collector_runs if r.collector == "gpu_hardware")
    errors = tuple(e for e in snapshot.collection_errors if e.collector == "gpu_hardware")
    unsupported = tuple(
        u for u in snapshot.unsupported_capabilities if u.capability in (NVIDIA_ALL, READINESS_MIG)
    )
    return snapshot.gpu, run, errors, unsupported


class _Smi:
    """A `NvidiaSmiAdapter` with a fixed query result."""

    def __init__(self, rows, provenance=DataOrigin.FIXTURE):
        self._rows = rows
        self.provenance = provenance

    def locate(self):
        return self._rows

    def version(self):
        raise AssertionError("the hardware collector doesn't read the nvidia-smi version")

    def query_gpus(self):
        return self._rows


# --- parity -------------------------------------------------------------------------------


@ids
def test_collector_reproduces_every_fixture_inventory(fixture):
    """FX10 included: its `nvml:1` is lost only after enumeration (KI-34, resolved)."""
    snapshot = fixture.snapshot
    collector, session = _collector(snapshot, FakeNvml.from_scenario(fixture))
    outcome = collector.collect()
    section, run, errors, unsupported = _expected(snapshot)
    assert outcome.section == section
    assert outcome.run == run
    assert outcome.errors == errors
    assert outcome.unsupported == unsupported
    assert all(u.capability != GPU_PCI_SCAN for u in outcome.unsupported)
    session.close()


def test_fx10_enumerates_the_gpu_before_it_is_lost():
    nvml = FakeNvml.from_scenario(FX10)
    collector, session = _collector(FX10.snapshot, nvml)
    outcome = collector.collect()
    assert outcome.section.devices[1] == FX10.snapshot.gpu.devices[1]
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()
    assert nvml.handles == {0: 1, 1: 1} and nvml.lost == set()  # the next handle loses it
    session.close()


def test_an_unreachable_device_handle_is_isolated_and_reported():
    fixture = FIXTURES_BY_ID["FX06"]
    nvml = FakeNvml.from_snapshot(
        fixture.snapshot, {(1, "nvmlDeviceGetHandleByIndex_v2"): "NVML_ERROR_GPU_IS_LOST"}
    )
    collector, session = _collector(fixture.snapshot, nvml)
    outcome = collector.collect()
    nvml0, nvml1 = outcome.section.devices
    assert nvml0 == fixture.snapshot.gpu.devices[0]
    assert nvml1.device_ref == "nvml:1" and nvml1.name.status is S.COLLECTION_FAILED
    assert outcome.section.device_count == fixture.snapshot.gpu.device_count
    assert outcome.errors == (
        CollectionError(
            collector="gpu_hardware",
            device_ref="nvml:1",
            status=S.COLLECTION_FAILED,
            error_code="NVML_ERROR_GPU_IS_LOST",
            message="GPU is lost",
            report_path="/gpu/devices/1",
        ),
    )
    assert outcome.run.status is R.PARTIAL


# --- the nvidia-smi fallback --------------------------------------------------------------


def test_fallback_devices_mark_fields_the_query_lacks():
    fixture = FIXTURES_BY_ID["FX04a"]
    collector, _ = _collector(fixture.snapshot)
    outcome = collector.collect()
    (device,) = outcome.section.devices
    assert outcome.section.inventory_source is Source.NVIDIA_SMI and device.device_ref == "smi:0"
    for field in ("pci_device_id", "pci_subsystem_id", "compute_capability", "mig_mode"):
        value = getattr(device, field)
        assert value.status is S.NOT_SUPPORTED and value.error.code == FALLBACK_FIELD_UNAVAILABLE
    assert outcome.section.device_count.method == "fixture:nvidia-smi --query-gpu=index"
    assert outcome.run.status is R.COMPLETED


def test_live_fallback_methods_carry_no_fixture_prefix():
    snapshot = FIXTURES_BY_ID["FX04a"].snapshot
    rows = FakeNvidiaSmi.from_snapshot(snapshot).query_gpus()
    collector = GpuHardwareCollector(
        UnimplementedNvmlSession(),
        _Smi(rows, DataOrigin.LIVE),
        FixedClock(),
        pci=snapshot.gpu.pci_observations,
        environment=LinuxEnvironment.NATIVE_LINUX,
        profile=PlatformProfile.LINUX,
    )
    section = collector.collect().section
    assert section.device_count.method == "nvidia-smi --query-gpu=index"
    assert section.devices[0].mig_mode.method == "mig.mode.current"


@pytest.mark.parametrize(
    ("status", "reported"),
    [
        (S.NOT_DETECTED, False),
        (S.NOT_SUPPORTED, False),
        (S.PERMISSION_DENIED, True),
        (S.COLLECTION_FAILED, True),
    ],
)
def test_fallback_query_failures_are_reported_only_when_they_are_problems(status, reported):
    snapshot = FIXTURES_BY_ID["FX03"].snapshot  # NVML not detected
    rows = Observed.failed(status, code="X", message="smi says no", source=Source.NVIDIA_SMI)
    collector, _ = _collector(snapshot, smi=_Smi(rows))
    outcome = collector.collect()
    assert outcome.section.device_count == snapshot.gpu.device_count  # NVML's own outcome
    assert outcome.section.devices == () and outcome.section.inventory_source is None
    if reported:
        (error,) = outcome.errors
        assert error.report_path == "/gpu/devices" and error.status is status
        assert outcome.run.status is R.PARTIAL
    else:
        assert outcome.errors == () and outcome.run.status is R.COMPLETED


def test_a_failed_count_after_a_good_init_is_a_collection_error():
    snapshot = FIXTURES_BY_ID["FX01"].snapshot
    nvml = FakeNvml.from_snapshot(snapshot)
    lost = Observed.failed(
        S.COLLECTION_FAILED, code="NVML_ERROR_UNKNOWN", message="Unknown Error", source=Source.NVML
    )
    nvml.inventory = dataclasses.replace(
        nvml.inventory, device_count=lost, devices=(), inventory_source=None
    )
    collector, _ = _collector(snapshot, nvml, smi=_Smi(lost))
    outcome = collector.collect()
    assert [e.report_path for e in outcome.errors] == ["/gpu/device_count", "/gpu/devices"]
    assert outcome.run.status is R.PARTIAL


# --- PCI hand-over, fallback section, capabilities ----------------------------------------


def test_the_collector_has_no_way_to_scan_pci():
    assert "scanner" not in inspect.signature(GpuHardwareCollector).parameters
    adapters = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX17"])
    collector, _ = _collector(FIXTURES_BY_ID["FX17"].snapshot)
    collector.collect()
    assert adapters.pci.calls == []


@pytest.mark.parametrize("fixture_id", ["FX03", "FX17", "FX01", "FX04b"])
def test_the_fallback_section_keeps_the_pci_collectors_result(fixture_id):
    snapshot = FIXTURES_BY_ID[fixture_id].snapshot
    collector, _ = _collector(snapshot)
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    section = collector.unavailable(error)
    assert section.pci_observations == snapshot.gpu.pci_observations
    assert (
        section.device_count.status is S.COLLECTION_FAILED and section.device_count.error == error
    )
    assert section.devices == () and section.inventory_source is None
    assert section == dataclasses.replace(
        GPUInventory.unavailable(S.COLLECTION_FAILED, error),
        pci_observations=snapshot.gpu.pci_observations,
    )


def test_nvidia_all_only_on_macos_readiness_mig_only_with_mig_and_never_gpu_pci_scan():
    for fixture in ALL_FIXTURES:
        collector, _ = _collector(fixture.snapshot)
        records = collector.collect().unsupported
        macos = fixture.snapshot.system.platform_profile is PlatformProfile.MACOS
        mig = [READINESS_MIG for d in fixture.snapshot.gpu.devices if d.mig_mode.value == "ENABLED"]
        assert [u.capability for u in records] == ([NVIDIA_ALL] if macos else []) + mig


def test_unexpected_exceptions_propagate():
    class Exploding(UnimplementedNvmlSession):
        def device_count(self):
            raise RuntimeError("binding bug")

    snapshot = FIXTURES_BY_ID["FX01"].snapshot
    collector = GpuHardwareCollector(
        Exploding(),
        FakeNvidiaSmi.from_snapshot(snapshot),
        FixedClock(),
        pci=snapshot.gpu.pci_observations,
        environment=LinuxEnvironment.NATIVE_LINUX,
        profile=PlatformProfile.LINUX,
    )
    with pytest.raises(RuntimeError, match="binding bug"):
        collector.collect()


@ids
def test_sections_validate_against_the_schema(fixture):
    defs = load_report_schema()["$defs"]
    collector, _ = _collector(fixture.snapshot)
    outcome = collector.collect()
    jsonschema.validate(
        to_json_dict(outcome.section), {"$ref": "#/$defs/GPUInventory", "$defs": defs}
    )
