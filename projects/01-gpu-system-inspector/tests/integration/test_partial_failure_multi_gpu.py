"""Partial failures stay contained and multiple GPUs keep stable identities (FX06, FX07, FX10)."""

import json

import pytest
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    DiagnosticStatus,
    ErrorDetail,
    ReadinessStatus,
)
from gpu_system_inspector.reporting import render_json, to_json_dict

FX06, FX07, FX10 = (FIXTURES_BY_ID[i] for i in ("FX06", "FX07", "FX10"))


def test_failed_collector_keeps_other_collectors_results():
    report = FX10.report()
    assert report.cuda_environment.toolkits.status is CollectionStatus.COLLECTION_FAILED
    assert report.system.os_name.is_collected
    assert report.driver.driver_version.is_collected
    assert report.frameworks.pytorch.cuda_available.value is True
    assert report.gpu.device_count.value == 2
    runs = {r.collector: r.status for r in report.collector_runs}
    assert runs["cuda_environment"] is CollectorRunStatus.FAILED
    assert runs["driver"] is CollectorRunStatus.COMPLETED


def test_guarded_collector_failure_reproduces_the_fixture_section():
    """Simulates the collector guard (data flow §3) with the fake collector."""
    collector = FakeAdapterSet.from_scenario(FX10).collectors["cuda_environment"]
    with pytest.raises(RuntimeError) as exc:
        collector.collect()
    error = ErrorDetail(
        code="UNHANDLED_EXCEPTION", message=f"{type(exc.value).__name__}: {exc.value}"
    )
    assert collector.unavailable(error) == FX10.snapshot.cuda_environment


def test_inaccessible_gpu_does_not_erase_the_other_gpu():
    healthy, lost = FX10.report().runtime
    assert healthy.device_ref == "nvml:0" and lost.device_ref == "nvml:1"
    assert healthy.memory_free_bytes.is_collected and healthy.processes.is_collected
    for name in (
        "memory_free_bytes",
        "gpu_utilization_percent",
        "temperature_gpu_celsius",
        "power_draw_watts",
    ):
        value = getattr(lost, name)
        assert value.status is CollectionStatus.COLLECTION_FAILED
        assert value.value is None, f"{name} must not become a value"
        assert value.error.code == "NVML_ERROR_GPU_IS_LOST"
    assert FX10.report().gpu.devices[1].name.is_collected


def test_missing_runtime_metrics_serialise_as_null_not_zero():
    lost = to_json_dict(FX10.report())["runtime"][1]
    observed = [v for v in lost.values() if isinstance(v, dict) and "status" in v]
    assert observed and all(v["value"] is None for v in observed)


def test_one_failing_check_does_not_suppress_other_findings():
    findings = {(d.check_id, d.device_ref): d.status for d in FX10.report().diagnostics}
    assert findings[("D06", "nvml:1")] is DiagnosticStatus.FAIL
    assert findings[("D06", "nvml:0")] is DiagnosticStatus.PASS
    assert findings[("D07", "nvml:0")] is DiagnosticStatus.INFO
    assert findings[("D01", None)] is DiagnosticStatus.PASS
    assert len(findings) == 10  # 6 system-wide + D06 and D07 for two devices


def test_fake_nvml_loses_the_gpu_after_enumeration_only():
    """KI-34: nvml:1 answers its first handle (enumeration) and is lost from then on."""
    adapters = FakeAdapterSet.from_scenario(FX10)
    lost_in_fixture = FX10.snapshot.runtime[1]
    with adapters.nvml.open() as session:
        assert session.device(0).value.memory_info().free.is_collected
        enumerated = session.device(1).value
        assert enumerated.name().is_collected
        assert enumerated.memory_info().total == FX10.snapshot.gpu.devices[1].memory_total_bytes
        again = session.device(1)
        assert again.status is CollectionStatus.COLLECTION_FAILED
        assert again.error == lost_in_fixture.memory_free_bytes.error
        memory, power = enumerated.memory_info(), enumerated.power_draw()
        for value in (memory.free, memory.total, power):
            assert value.status is CollectionStatus.COLLECTION_FAILED and value.value is None
            assert value.error == lost_in_fixture.memory_free_bytes.error
        assert power.unit == "watts" and memory.free.unit == "bytes"
        assert session.device(0).value.name().is_collected  # nvml:0 unaffected


def test_multiple_devices_keep_stable_identities_and_order():
    report = FX06.report()
    assert report.gpu.device_refs == ("nvml:0", "nvml:1")
    names = [d.name.value for d in report.gpu.devices]
    assert names == ["Synthetic GPU B (fixture)", "Synthetic GPU A (fixture)"]
    assert [d.uuid.value for d in report.gpu.devices] == [
        "GPU-bbbbbbbb-0000-4000-8000-00000000000b",
        "GPU-aaaaaaaa-0000-4000-8000-00000000000a",
    ]
    assert render_json(FX06.report()) == render_json(report)


def test_nvml_and_pytorch_devices_stay_correctly_associated():
    report = FX06.report()
    nvml_by_ref = {d.device_ref: d for d in report.gpu.devices}
    for torch_device in report.frameworks.pytorch.devices:
        nvml = nvml_by_ref[torch_device.correlated_device_ref.value]
        assert nvml.uuid.value.removeprefix("GPU-") == torch_device.uuid.value
        assert nvml.name.value == torch_device.name.value
    assert [
        (d.torch_index, d.correlated_device_ref.value) for d in report.frameworks.pytorch.devices
    ] == [(0, "nvml:1"), (1, "nvml:0")]


def test_readiness_is_evaluated_per_device():
    b_run = FX06.readiness_runs[0]
    readiness = FX06.report(b_run).readiness
    per_device = {
        d.device_ref: (d.status, d.reason.value, d.margin_bytes) for d in readiness.devices
    }
    assert per_device == {
        "nvml:0": (ReadinessStatus.NOT_READY, "INSUFFICIENT_TOTAL_MEMORY", -6 * 2**30),
        "nvml:1": (ReadinessStatus.READY, "SUFFICIENT_FREE_MEMORY", 4 * 2**30),
    }
    assert readiness.overall_status is ReadinessStatus.READY
    fx07 = FX07.report(FX07.readiness_runs[0]).readiness
    assert {d.status for d in fx07.devices} == {ReadinessStatus.NOT_READY}
    assert fx07.overall_status is ReadinessStatus.NOT_READY


def test_device_order_survives_json_round_trip():
    data = json.loads(render_json(FX10.report()))
    assert [d["device_ref"] for d in data["gpu"]["devices"]] == ["nvml:0", "nvml:1"]
    assert [e["device_ref"] for e in data["collection_errors"]] == [None, "nvml:1"]
