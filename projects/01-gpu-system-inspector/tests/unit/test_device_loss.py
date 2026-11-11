"""KI-34: a GPU that is enumerated and then falls off the bus, end to end through the
production `Inspector`, over `FakeNvml`'s device-loss lifecycle and the production NVML
adapter over the stub binding.

`LEGACY_FX10_FAULTS` is FX10's previous fault model, kept as evidence of why it changed:
static, session-wide faults on the runtime functions, `nvmlDeviceGetMemoryInfo_v2` included.
They also fail the enumeration's memory read, which FX10's own snapshot records as
collected, so no collector could reproduce that snapshot from them."""

import dataclasses
import json

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeNvidiaSmi, FakeNvml, FixedClock, StubPynvml
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import AdapterSet, PynvmlAdapter
from gpu_system_inspector.collectors import GpuHardwareCollector
from gpu_system_inspector.core import Inspector
from gpu_system_inspector.models import CollectionStatus, CollectorRunStatus, DataOrigin
from gpu_system_inspector.reporting import load_report_schema, render_json

S = CollectionStatus
R = CollectorRunStatus
FX01, FX06, FX10 = (FIXTURES_BY_ID[i] for i in ("FX01", "FX06", "FX10"))
LOST = "NVML_ERROR_GPU_IS_LOST"
LEGACY_FX10_FAULTS = {
    (1, name): LOST
    for name in (
        "nvmlDeviceGetUtilizationRates",
        "nvmlDeviceGetMemoryInfo_v2",
        "nvmlDeviceGetTemperatureV",
        "nvmlDeviceGetTemperatureThreshold",
        "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetEnforcedPowerLimit",
        "nvmlDeviceGetComputeRunningProcesses_v3",
    )
}


def _inspect(fixture, **loss):
    adapters = FakeAdapterSet.from_scenario(fixture)
    if loss:
        adapters.nvml = FakeNvml.from_snapshot(fixture.snapshot, **loss)
    report = Inspector(adapters.adapter_set()).run()
    return report, adapters


def _runtime_errors(report):
    return [
        (e.device_ref, e.error_code, e.report_path)
        for e in report.collection_errors
        if e.collector == "runtime"
    ]


def _runs(report):
    return {r.collector: r.status for r in report.collector_runs}


def test_the_legacy_fault_model_contradicts_fx10s_own_inventory():
    """Evidence for the FX10 correction: same snapshot, old faults, the enumeration fails."""
    nvml = FakeNvml.from_snapshot(FX10.snapshot, LEGACY_FX10_FAULTS)
    session = nvml.open()
    inventory = GpuHardwareCollector(
        session,
        FakeNvidiaSmi.from_snapshot(FX10.snapshot),
        FixedClock(),
        pci=FX10.snapshot.gpu.pci_observations,
        environment=FX10.snapshot.system.linux_environment.value,
        profile=FX10.snapshot.system.platform_profile,
    ).collect()
    session.close()
    assert inventory.section.devices[1].memory_total_bytes.error.code == LOST
    assert FX10.snapshot.gpu.devices[1].memory_total_bytes.is_collected  # the contradiction


def test_one_gpu_lost_after_enumeration_reproduces_fx10():
    report, adapters = _inspect(FX10)
    snapshot = FX10.snapshot
    assert (report.gpu, report.driver, report.runtime) == (
        snapshot.gpu,
        snapshot.driver,
        snapshot.runtime,
    )
    assert _runtime_errors(report) == [("nvml:1", LOST, "/runtime/1")]
    assert _runs(report)["runtime"] is R.PARTIAL
    assert [c for c, s in _runs(report).items() if s is R.FAILED] == ["cuda_environment"]
    assert adapters.nvml.lost == {1} and adapters.nvml.inits == adapters.nvml.shutdowns


@pytest.mark.parametrize(
    ("fixture", "lost"),
    [(FX06, {0: 1, 1: 1}), (FX01, {0: 1})],
    ids=["several-gpus", "all-gpus"],
)
def test_several_or_all_gpus_lost_keep_the_inventory_and_one_error_each(fixture, lost):
    report, adapters = _inspect(fixture, lost_after_handles=lost)
    assert report.gpu == fixture.snapshot.gpu  # enumeration saw every device
    assert [m.memory_free_bytes.error.code for m in report.runtime] == [LOST] * len(lost)
    assert _runtime_errors(report) == [(f"nvml:{i}", LOST, f"/runtime/{i}") for i in sorted(lost)]
    assert report.driver == fixture.snapshot.driver  # system-level NVML still answers
    assert _runs(report)["gpu_hardware"] is R.COMPLETED and _runs(report)["runtime"] is R.PARTIAL


def test_a_gpu_lost_before_enumeration_is_reported_by_both_runs():
    report, _ = _inspect(FX06, lost_after_handles={1: 0})
    assert report.gpu.devices[0] == FX06.snapshot.gpu.devices[0]
    assert report.gpu.devices[1].name.error.code == LOST
    assert report.gpu.device_count == FX06.snapshot.gpu.device_count  # still counted
    assert [(e.collector, e.device_ref, e.report_path) for e in report.collection_errors] == [
        ("gpu_hardware", "nvml:1", "/gpu/devices/1"),
        ("runtime", "nvml:1", "/runtime/1"),
    ]


def test_a_gpu_lost_during_the_driver_observations():
    """Lost while the driver run reads the driver version: the driver facts are
    system-level and complete; the device is lost to enumeration and runtime."""
    report, _ = _inspect(FX06, lost_on={1: "nvmlSystemGetDriverVersion"})
    assert report.driver == FX06.snapshot.driver
    assert report.gpu.devices[1].uuid.error.code == LOST
    assert report.runtime[1].memory_free_bytes.error.code == LOST
    assert report.runtime[0] == FX06.snapshot.runtime[0]


def test_a_gpu_lost_during_the_runtime_observations():
    report, _ = _inspect(FX06, lost_on={1: "nvmlDeviceGetPowerUsage"})
    assert report.gpu == FX06.snapshot.gpu
    assert report.runtime[0] == FX06.snapshot.runtime[0]
    assert report.runtime[1].temperature_gpu_celsius.error.code == LOST  # no partial values
    assert _runtime_errors(report) == [("nvml:1", LOST, "/runtime/1")]


@pytest.mark.parametrize(
    "loss", [{"lost_after_handles": {1: 1}}, {"lost_after_handles": {0: 1, 1: 1}}]
)
def test_loss_keeps_pci_unique_capabilities_and_a_deterministic_valid_report(loss):
    report, adapters = _inspect(FX10, **loss)
    assert report.gpu.pci_observations == FX10.snapshot.gpu.pci_observations
    assert adapters.pci.calls == ["/sys/bus/pci/devices"]
    records = [(u.capability, u.device_ref) for u in report.unsupported_capabilities]
    assert len(records) == len(set(records))
    text = render_json(report)
    jsonschema.validate(json.loads(text), load_report_schema())
    assert render_json(_inspect(FX10, **loss)[0]) == text


def test_the_production_adapter_reproduces_fx10_through_the_inspector():
    fakes = FakeAdapterSet.from_scenario(FX10)
    stub = StubPynvml.from_snapshot(FX10.snapshot, lost_after_handles={1: 1})
    adapters = dataclasses.replace(
        fakes.adapter_set(),
        nvml=PynvmlAdapter(stub.loader, platform=fakes.platform, provenance=DataOrigin.FIXTURE),
    )
    assert isinstance(adapters, AdapterSet)
    report = Inspector(adapters).run()
    assert report.gpu == FX10.snapshot.gpu
    healthy, lost = report.runtime
    assert lost.device_ref == "nvml:1" and lost.memory_free_bytes.error.code == LOST
    assert healthy.memory_free_bytes.is_collected
    assert _runtime_errors(report) == [("nvml:1", LOST, "/runtime/1")]
    assert stub.lost == {1} and stub.inits == stub.shutdowns == 1
