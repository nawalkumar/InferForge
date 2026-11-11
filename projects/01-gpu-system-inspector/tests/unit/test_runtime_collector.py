"""IMP-K04 runtime metrics collector (DEC-46) over the fake NVML session, the `nvidia-smi`
fallback and the production adapter over a stub binding: fixture parity, the process
opt-out, per-device isolation, device loss, capability records and the guard fallback. No
NVML library, process or sysfs access."""

import dataclasses

import pytest
from gsi_testkit.builders import COLLECTOR_ORDER, failed
from gsi_testkit.fakes import FakeNvidiaSmi, FakeNvml, FakePlatform, FixedClock, StubGpu, StubPynvml
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import PynvmlAdapter, SmiGpuRow
from gpu_system_inspector.collectors import (
    RUNTIME_PROCESS_MEMORY,
    RUNTIME_PROCESSES,
    GpuHardwareCollector,
    RuntimeCollector,
)
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    GpuProcess,
    Observed,
    PlatformProfile,
    ProcessKind,
    Source,
)

S = CollectionStatus
R = CollectorRunStatus
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
RUNTIME_STEP_MS = 5 * (COLLECTOR_ORDER.index("runtime") + 1)
FX01, FX04A, FX06, FX09 = (FIXTURES_BY_ID[i] for i in ("FX01", "FX04a", "FX06", "FX09"))
PROCESS_FUNCTIONS = {"nvmlDeviceGetComputeRunningProcesses_v3", "nvmlSystemGetProcessName"}


def _collect(
    snapshot, nvml=None, smi=None, *, profile=None, include_processes=True, inventory=None
):
    nvml = nvml or FakeNvml.from_snapshot(snapshot)
    session = nvml.open()
    outcome = RuntimeCollector(
        session,
        smi or FakeNvidiaSmi.from_snapshot(snapshot),
        FixedClock(step_ms=RUNTIME_STEP_MS),
        inventory=inventory or snapshot.gpu,
        profile=profile or snapshot.system.platform_profile,
        include_processes=include_processes,
    ).collect()
    session.close()
    return outcome, nvml


def _expected(snapshot):
    (run,) = (r for r in snapshot.collector_runs if r.collector == "runtime")
    errors = tuple(e for e in snapshot.collection_errors if e.collector == "runtime")
    unsupported = tuple(
        u for u in snapshot.unsupported_capabilities if u.capability.startswith("RUNTIME.")
    )
    return snapshot.runtime, run, errors, unsupported


def _with_runtime(snapshot, index=0, **fields):
    runtime = list(snapshot.runtime)
    runtime[index] = dataclasses.replace(runtime[index], **fields)
    return dataclasses.replace(snapshot, runtime=tuple(runtime))


# --- parity -------------------------------------------------------------------------------


@ids
def test_collector_reproduces_every_fixture_runtime_section(fixture):
    """Every field, status, source, method, unit, error and note; runs, errors, records."""
    nvml = FakeNvml.from_scenario(fixture)
    snapshot = fixture.snapshot
    if snapshot.gpu.inventory_source is Source.NVML:  # the GPU hardware run enumerates first
        session = nvml.open()
        for device in snapshot.gpu.devices:
            session.device(device.nvml_index)
        session.close()
    outcome, _ = _collect(snapshot, nvml)
    runtime, run, errors, unsupported = _expected(snapshot)
    assert outcome.section == runtime
    assert outcome.run == run
    assert outcome.errors == errors
    assert outcome.unsupported == unsupported


def test_processes_can_be_skipped_without_any_process_call():
    """AC59: `include_processes=False` → `SKIPPED`, no process NVML calls."""
    outcome, nvml = _collect(FX01.snapshot, include_processes=False)
    (metrics,) = outcome.section
    assert metrics.processes.status is S.SKIPPED and metrics.processes.error is None
    assert metrics.memory_free_bytes.is_collected
    assert PROCESS_FUNCTIONS.isdisjoint(nvml.calls)
    assert outcome.run.status is R.COMPLETED


def test_no_inventory_means_no_records_and_no_calls():
    outcome, nvml = _collect(FIXTURES_BY_ID["FX03"].snapshot)
    assert outcome.section == () and outcome.run.status is R.COMPLETED
    assert nvml.calls == ["nvmlInit"]


def test_the_nvidia_smi_fallback_reuses_the_inventory_query():
    smi = FakeNvidiaSmi.from_snapshot(FX04A.snapshot)
    outcome, _ = _collect(FX04A.snapshot, smi=smi)
    assert outcome.section == FX04A.snapshot.runtime
    assert smi.calls == ["nvidia-smi --query-gpu"]  # the adapter caches it in production


def test_a_device_missing_from_the_query_is_reported_not_invented():
    class Smi:
        provenance = DataOrigin.FIXTURE

        def query_gpus(self):
            return Observed.collected((), source=Source.NVIDIA_SMI, method="nvidia-smi")

    outcome, _ = _collect(FX04A.snapshot, smi=Smi())
    (metrics,) = outcome.section
    assert metrics.memory_free_bytes.status is S.COLLECTION_FAILED
    (error,) = outcome.errors
    assert (error.error_code, error.device_ref, error.report_path) == (
        "DEVICE_NOT_IN_QUERY",
        "smi:0",
        "/runtime/0",
    )


def test_a_lost_gpu_in_the_nvidia_smi_query_is_a_lost_device():
    rows = FakeNvidiaSmi.from_snapshot(FX04A.snapshot).query_gpus()
    lost = failed(
        S.COLLECTION_FAILED, "NVIDIA_SMI_GPU_IS_LOST", "[GPU is lost]", source=Source.NVIDIA_SMI
    )
    (row,) = rows.value
    values = dict(row.values, **{"memory.free": lost})

    class Smi:
        provenance = DataOrigin.FIXTURE

        def query_gpus(self):
            return dataclasses.replace(rows, value=(SmiGpuRow(index=row.index, values=values),))

    outcome, _ = _collect(FX04A.snapshot, smi=Smi())
    (metrics,) = outcome.section
    assert metrics.gpu_utilization_percent.error.code == "NVIDIA_SMI_GPU_IS_LOST"
    assert [e.report_path for e in outcome.errors] == ["/runtime/0"]


# --- isolation and device loss ------------------------------------------------------------


def test_an_unreachable_device_is_isolated_and_reported():
    nvml = FakeNvml.from_snapshot(
        FX06.snapshot, {(1, "nvmlDeviceGetHandleByIndex_v2"): "NVML_ERROR_UNKNOWN"}
    )
    outcome, _ = _collect(FX06.snapshot, nvml)
    healthy, unreachable = outcome.section
    assert healthy == FX06.snapshot.runtime[0]
    assert unreachable.memory_free_bytes.status is S.COLLECTION_FAILED
    assert outcome.errors == (
        CollectionError(
            collector="runtime",
            device_ref="nvml:1",
            status=S.COLLECTION_FAILED,
            error_code="NVML_ERROR_UNKNOWN",
            message="Unknown Error",
            report_path="/runtime/1",
        ),
    )
    assert outcome.run.status is R.PARTIAL


def test_a_gpu_lost_mid_read_stops_its_reads_and_keeps_no_partial_values():
    nvml = FakeNvml.from_snapshot(FX06.snapshot, lost_on={1: "nvmlDeviceGetTemperatureV"})
    outcome, _ = _collect(FX06.snapshot, nvml)
    healthy, lost = outcome.section
    assert healthy == FX06.snapshot.runtime[0]
    for field in ("memory_free_bytes", "gpu_utilization_percent", "power_draw_watts"):
        assert getattr(lost, field).error == ErrorDetail(
            code="NVML_ERROR_GPU_IS_LOST", message="GPU is lost"
        )
    assert nvml.calls.count("nvmlDeviceGetTemperatureV") == 2  # nvml:0, then nvml:1 is lost
    for after_the_loss in ("nvmlDeviceGetPowerUsage", "nvmlDeviceGetComputeRunningProcesses_v3"):
        assert nvml.calls.count(after_the_loss) == 1  # nvml:0 only
    assert [e.device_ref for e in outcome.errors] == ["nvml:1"]


def test_field_failures_stay_field_level():
    outcome, _ = _collect(FX09.snapshot)
    (metrics,) = outcome.section
    assert metrics.memory_free_bytes.status is S.PERMISSION_DENIED
    assert outcome.errors == () and outcome.run.status is R.PARTIAL


def test_the_guard_fallback_gives_every_device_an_unavailable_record():
    nvml = FakeNvml.from_snapshot(FX06.snapshot)
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    collector = RuntimeCollector(
        nvml.open(),
        FakeNvidiaSmi.from_snapshot(FX06.snapshot),
        FixedClock(),
        inventory=FX06.snapshot.gpu,
        profile=PlatformProfile.LINUX,
    )
    records = collector.unavailable(error)
    assert [r.device_ref for r in records] == ["nvml:0", "nvml:1"]
    assert all(r.memory_free_bytes.error == error for r in records)


# --- capability records -------------------------------------------------------------------


def test_unsupported_process_enumeration_on_linux_is_recorded_without_a_platform_note():
    unsupported = failed(
        S.NOT_SUPPORTED,
        "NVML_ERROR_NOT_SUPPORTED",
        "Not Supported",
        method="nvmlDeviceGetComputeRunningProcesses_v3",
    )
    outcome, _ = _collect(_with_runtime(FX01.snapshot, processes=unsupported))
    (record,) = outcome.unsupported
    assert (record.capability, record.device_ref, record.reason) == (
        RUNTIME_PROCESSES,
        "nvml:0",
        "NVML reports process enumeration as not supported.",
    )
    assert outcome.section[0].processes.note is None


def test_unreported_process_memory_off_windows_has_a_generic_reason():
    process = GpuProcess(
        pid=7,
        kind=ProcessKind.COMPUTE,
        used_memory_bytes=failed(
            S.NOT_SUPPORTED, "VALUE_NOT_AVAILABLE", "n/a", unit="bytes", method="nvml"
        ),
        name=failed(S.PERMISSION_DENIED, "NVML_ERROR_NO_PERMISSION", "denied", method="nvml"),
    )
    processes = Observed.collected((process,), source=Source.NVML, method="nvml")
    outcome, _ = _collect(_with_runtime(FX01.snapshot, processes=processes))
    (record,) = outcome.unsupported
    assert record.capability == RUNTIME_PROCESS_MEMORY
    assert record.reason == "NVML doesn't report per-process GPU memory on this device."


@pytest.mark.parametrize(
    ("fixture_id", "fields"),
    [
        ("FX11", ("gpu_utilization_percent", "memory_bus_utilization_percent", "processes")),
        ("FX14", ("gpu_utilization_percent", "memory_bus_utilization_percent")),
    ],
)
def test_the_collector_adds_the_documented_notes_the_adapter_cannot_know(fixture_id, fields):
    """The production adapter reports a bare `NOT_SUPPORTED`; the platform (WSL2) or the
    device's MIG mode is what the collector knows, so the result equals the fixture."""
    fixture = FIXTURES_BY_ID[fixture_id]
    bare = {
        f: dataclasses.replace(getattr(fixture.snapshot.runtime[0], f), note=None) for f in fields
    }
    outcome, _ = _collect(_with_runtime(fixture.snapshot, **bare))
    assert outcome.section == fixture.snapshot.runtime
    assert outcome.unsupported == _expected(fixture.snapshot)[3]


def test_the_wddm_note_is_added_to_unreported_process_memory():
    fixture = FIXTURES_BY_ID["FX12"]
    processes = fixture.snapshot.runtime[0].processes
    bare = tuple(
        dataclasses.replace(
            p, used_memory_bytes=dataclasses.replace(p.used_memory_bytes, note=None)
        )
        for p in processes.value
    )
    snapshot = _with_runtime(fixture.snapshot, processes=dataclasses.replace(processes, value=bare))
    outcome, _ = _collect(snapshot)
    assert outcome.section == fixture.snapshot.runtime
    assert outcome.unsupported == _expected(fixture.snapshot)[3]


def test_each_device_gets_its_own_records():
    unsupported = failed(
        S.NOT_SUPPORTED, "NVML_ERROR_NOT_SUPPORTED", "Not Supported", method="nvml"
    )
    snapshot = _with_runtime(FX06.snapshot, 0, processes=unsupported)
    snapshot = _with_runtime(snapshot, 1, processes=unsupported)
    outcome, _ = _collect(snapshot)
    assert [(u.capability, u.device_ref) for u in outcome.unsupported] == [
        (RUNTIME_PROCESSES, "nvml:0"),
        (RUNTIME_PROCESSES, "nvml:1"),
    ]


# --- the production adapter ---------------------------------------------------------------


def test_production_adapter_metrics_are_converted_and_independent():
    gpu = StubGpu(power_mw=31_500, power_limit_mw=450_000, temperature=None)
    stub = StubPynvml([gpu])
    adapter = PynvmlAdapter(stub.loader, platform=FakePlatform.from_snapshot(FX01.snapshot))
    smi = FakeNvidiaSmi.from_snapshot(FX01.snapshot)
    with adapter.open() as session:
        inventory = (
            GpuHardwareCollector(
                session,
                smi,
                FixedClock(),
                pci=FX01.snapshot.gpu.pci_observations,
                environment=FX01.snapshot.system.linux_environment.value,
                profile=PlatformProfile.LINUX,
            )
            .collect()
            .section
        )
        outcome = RuntimeCollector(
            session, smi, FixedClock(), inventory=inventory, profile=PlatformProfile.LINUX
        ).collect()
    (metrics,) = outcome.section
    assert (metrics.power_draw_watts.value, metrics.power_limit_watts.value) == (31.5, 450.0)
    assert metrics.temperature_gpu_celsius.error.code == "VALUE_NOT_AVAILABLE"
    assert metrics.memory_total_bytes.value == gpu.memory[0]
    assert metrics.processes.is_collected and outcome.errors == ()
    assert metrics.sampled_at.tzinfo is not None
