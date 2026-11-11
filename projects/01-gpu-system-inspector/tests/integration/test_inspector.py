"""IMP-K08 + IMP-K02 + IMP-P01 (DEC-44): the production core and collectors over fake
adapters, end to end to a validated report. Covers every fixture plus the failure
scenarios. No real `/sys`, NVML, process or hardware access: PCI scanners are fakes or the
production adapter over `FakeSysfsReader`."""

import dataclasses
import errno
import json

import jsonschema
import pytest
from gsi_testkit.builders import COLLECTOR_ORDER
from gsi_testkit.fakes import (
    FakeAdapterSet,
    FakeNvidiaSmi,
    FakeNvmlDevice,
    FakeNvmlSession,
    FakePlatform,
    FakeSysfsReader,
    sysfs_device,
    sysfs_reader_for,
)
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import PciSysfsAdapter
from gpu_system_inspector.adapters.outcomes import (
    LINUX_ENVIRONMENT_UNKNOWN,
    NOT_IMPLEMENTED,
    UNHANDLED_EXCEPTION,
)
from gpu_system_inspector.collectors import GPU_PCI_SCAN, NVIDIA_ALL
from gpu_system_inspector.core import NOT_IMPLEMENTED_LIMITATION, Inspector
from gpu_system_inspector.models import (
    SNAPSHOT_LIMITATION,
    CollectionError,
    CollectionStatus,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    InspectionMode,
    InspectionOptions,
    InvalidModelError,
    LinuxEnvironment,
    UnsupportedCapability,
)
from gpu_system_inspector.reporting import (
    assemble_report,
    load_report_schema,
    parse_report_json,
    render_json,
)

S = CollectionStatus
R = CollectorRunStatus
E = LinuxEnvironment
IMPLEMENTED = (
    "system",
    "pci",
    "nvml",
    "driver",
    "gpu_hardware",
    "runtime",
    "nvidia_smi",
    "cuda_environment",
    "pytorch",
    "correlation",
)
FX10 = FIXTURES_BY_ID["FX10"]
all_fixtures = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)


def _owned(record: UnsupportedCapability) -> bool:
    return record.capability in (GPU_PCI_SCAN, NVIDIA_ALL) or record.capability.startswith(
        ("SYSTEM.", "RUNTIME.", "CUDA.")
    )


def _inspect(fixture, *, adapters=None, factory=None):
    adapters = adapters or FakeAdapterSet.from_scenario(fixture)
    inspector = Inspector(adapters.adapter_set(), pci_scanner_factory=factory)
    return inspector.run(), adapters


def _inspect_with(fixture, options):
    adapters = FakeAdapterSet.from_scenario(fixture)
    return Inspector(adapters.adapter_set()).run(options), adapters


class _ScannerFactory:
    """Records the environment it was given and builds the production adapter over a
    synthetic sysfs."""

    def __init__(self, reader: FakeSysfsReader):
        self.reader = reader
        self.environments: list[E | None] = []

    def __call__(self, environment):
        self.environments.append(environment)
        return PciSysfsAdapter(environment=environment, reader=self.reader)


class _ExplodingPlatform(FakePlatform):
    def system_info(self):
        raise RuntimeError("synthetic system crash")


def _runs(report):
    return {r.collector: r.status for r in report.collector_runs}


def _errors(report, collector):
    return [e for e in report.collection_errors if e.collector == collector]


# --- every fixture ------------------------------------------------------------------------


@all_fixtures
def test_inspection_reproduces_the_implemented_parts_of_every_fixture(fixture):
    """Complete sections, FX10 included (KI-34 resolved): every field, status, source,
    method, unit, error and note of the system, GPU, driver, runtime, CUDA environment and
    PyTorch sections, correlation included (FX10's CUDA crash is injected into the locator
    and contained by the guard)."""
    report, _ = _inspect(fixture)
    snapshot = fixture.snapshot
    assert report.system == snapshot.system
    assert report.gpu == snapshot.gpu
    assert report.driver == snapshot.driver
    assert report.runtime == snapshot.runtime
    assert report.cuda_environment == snapshot.cuda_environment
    assert report.frameworks.pytorch == snapshot.frameworks.pytorch  # correlation included
    assert [(r.collector, r.status) for r in report.collector_runs] == [
        (r.collector, r.status) for r in snapshot.collector_runs if r.collector in IMPLEMENTED
    ]
    assert report.collection_errors == tuple(
        e for e in snapshot.collection_errors if e.collector in IMPLEMENTED
    )
    assert report.unsupported_capabilities == tuple(
        u for u in snapshot.unsupported_capabilities if _owned(u)
    )


def test_fx10_keeps_the_enumerated_gpu_and_reports_its_loss_once():
    report, adapters = _inspect(FX10)
    assert report.gpu == FX10.snapshot.gpu  # enumeration saw nvml:1 whole
    lost = report.runtime[1]
    assert lost.device_ref == "nvml:1" and lost.memory_free_bytes.error.code == (
        "NVML_ERROR_GPU_IS_LOST"
    )
    assert report.runtime[0] == FX10.snapshot.runtime[0]
    assert [(e.collector, e.device_ref, e.report_path) for e in report.collection_errors] == [
        ("cuda_environment", None, "/cuda_environment"),  # the injected crash, guard-owned
        ("runtime", "nvml:1", "/runtime/1"),
    ]
    assert _runs(report)["gpu_hardware"] is R.COMPLETED and _runs(report)["runtime"] is R.PARTIAL
    assert adapters.nvml.lost == {1}


@all_fixtures
def test_every_report_is_valid_complete_and_deterministic(fixture):
    report, _ = _inspect(fixture)
    text = render_json(report)
    jsonschema.validate(json.loads(text), load_report_schema())
    assert parse_report_json(text) == report
    assert render_json(_inspect(fixture)[0]) == text
    assert report.data_origin is DataOrigin.FIXTURE and report.mode is InspectionMode.GENERAL
    assert report.limitations == (SNAPSHOT_LIMITATION, NOT_IMPLEMENTED_LIMITATION)
    assert [r.collector for r in report.collector_runs] == [
        c for c in COLLECTOR_ORDER if c in IMPLEMENTED
    ]
    assert report.driver.nvml_version.error is None or (
        report.driver.nvml_version.error.code != NOT_IMPLEMENTED
    )
    assert report.cuda_environment.toolkits.error is None or (
        report.cuda_environment.toolkits.error.code != NOT_IMPLEMENTED
    )
    assert report.frameworks.pytorch.status is not S.SKIPPED  # include_pytorch defaults on
    assert report.diagnostics == () and report.readiness is None
    failed = {c for c, status in _runs(report).items() if status is R.FAILED}
    assert failed == {  # only the fixture's own failures (FX10's injected CUDA crash)
        r.collector
        for r in fixture.snapshot.collector_runs
        if r.status is R.FAILED and r.collector in IMPLEMENTED
    }


@all_fixtures
def test_the_nvml_session_is_opened_once_and_always_closed(fixture):
    _, adapters = _inspect(fixture)
    nvml = adapters.nvml
    assert nvml.calls.count("nvmlInit") == 1
    assert nvml.inits == nvml.shutdowns
    assert adapters.pci.calls == ["/sys/bus/pci/devices"]  # exactly one PCI scan


# --- scenarios ----------------------------------------------------------------------------


def test_a_normal_inspection():
    report, _ = _inspect(FIXTURES_BY_ID["FX01"])
    assert set(_runs(report).values()) == {R.COMPLETED}
    (device,) = report.gpu.devices
    assert device.device_ref == "nvml:0" and device.name.is_collected
    assert report.gpu.pci_observations.value[0].is_nvidia
    assert report.collection_errors == () and report.unsupported_capabilities == ()


def test_no_nvidia_gpu():
    report, _ = _inspect(FIXTURES_BY_ID["FX03"])
    assert report.gpu.device_count.status is S.NOT_DETECTED and report.gpu.devices == ()
    assert [o.is_nvidia for o in report.gpu.pci_observations.value] == [False]
    assert set(_runs(report).values()) == {R.COMPLETED}


@pytest.mark.parametrize("fixture_id", ["FX04b", "FX11"])
def test_wsl2_is_gated_and_reported_once(fixture_id):
    factory = _ScannerFactory(FakeSysfsReader(failures={"list": AssertionError("touched")}))
    report, _ = _inspect(FIXTURES_BY_ID[fixture_id], factory=factory)
    assert factory.environments == [E.WSL2] and factory.reader.calls == []
    assert report.gpu.pci_observations.status is S.NOT_SUPPORTED
    assert [u.capability for u in report.unsupported_capabilities].count(GPU_PCI_SCAN) == 1
    assert _runs(report)["pci"] is R.COMPLETED


def test_a_missing_pci_sysfs_root_completes():
    reader = FakeSysfsReader(
        failures={"list": FileNotFoundError(errno.ENOENT, "No such file", "/sys/bus/pci")}
    )
    report, _ = _inspect(FIXTURES_BY_ID["FX03"], factory=_ScannerFactory(reader))
    assert report.gpu.pci_observations.status is S.NOT_DETECTED
    assert _runs(report)["pci"] is R.COMPLETED and _errors(report, "pci") == []
    assert R.FAILED not in _runs(report).values()


def test_a_partial_pci_scan_survives_into_the_report():
    fixture = FIXTURES_BY_ID["FX17"]
    reader = sysfs_reader_for(fixture.snapshot.gpu.pci_observations.value)
    reader.devices["junk"] = sysfs_device()
    report, _ = _inspect(fixture, factory=_ScannerFactory(reader))
    assert report.gpu.pci_observations == fixture.snapshot.gpu.pci_observations
    assert _runs(report)["pci"] is R.PARTIAL
    (error,) = _errors(report, "pci")
    assert (
        error.message.startswith("junk address:") and error.report_path == "/gpu/pci_observations"
    )
    jsonschema.validate(json.loads(render_json(report)), load_report_schema())


@pytest.mark.parametrize(
    ("fixture_id", "environment", "pci_code"),
    [("FX03", E.UNKNOWN, LINUX_ENVIRONMENT_UNKNOWN), ("FX04b", E.WSL2, "PLATFORM_NOT_SUPPORTED")],
    ids=["linux-fails-closed", "wsl2-stays-wsl2"],
)
def test_a_system_failure_fails_closed(fixture_id, environment, pci_code):
    fixture = FIXTURES_BY_ID[fixture_id]
    adapters = FakeAdapterSet.from_scenario(fixture)
    adapters.platform = _ExplodingPlatform.from_snapshot(fixture.snapshot)
    factory = _ScannerFactory(FakeSysfsReader(failures={"list": AssertionError("touched")}))
    report, _ = _inspect(fixture, adapters=adapters, factory=factory)
    assert _runs(report)["system"] is R.FAILED
    assert _errors(report, "system") == [
        CollectionError(
            collector="system",
            device_ref=None,
            status=S.COLLECTION_FAILED,
            error_code=UNHANDLED_EXCEPTION,
            message="RuntimeError: synthetic system crash",
            report_path="/system",
        )
    ]
    assert report.system.platform_profile is fixture.snapshot.system.platform_profile
    assert report.system.linux_environment.status is S.COLLECTION_FAILED
    assert factory.environments == [environment] and factory.reader.calls == []
    assert report.gpu.pci_observations.error.code == pci_code
    assert _runs(report)["pci"] is R.COMPLETED
    jsonschema.validate(json.loads(render_json(report)), load_report_schema())


def test_a_prebuilt_native_scanner_is_refused_after_a_system_failure():
    """Without a factory, the adapter set's scanner was built for native Linux; once the
    environment falls back to UNKNOWN the PCI collector refuses its result."""
    fixture = FIXTURES_BY_ID["FX03"]
    adapters = FakeAdapterSet.from_scenario(fixture)
    adapters.platform = _ExplodingPlatform.from_snapshot(fixture.snapshot)
    report, _ = _inspect(fixture, adapters=adapters)
    assert _runs(report)["pci"] is R.FAILED
    (error,) = _errors(report, "pci")
    assert error.message.startswith("AdapterUsageError:")
    assert report.gpu.pci_observations.status is S.COLLECTION_FAILED


def test_a_pci_collector_exception_is_contained():
    class Exploding:
        provenance = DataOrigin.FIXTURE

        def scan(self):
            raise RuntimeError("synthetic scanner crash")

    report, _ = _inspect(FIXTURES_BY_ID["FX01"], factory=lambda environment: Exploding())
    assert _runs(report) == {
        "system": R.COMPLETED,
        "pci": R.FAILED,
        "nvml": R.COMPLETED,
        "driver": R.COMPLETED,
        "gpu_hardware": R.COMPLETED,
        "runtime": R.COMPLETED,
        "nvidia_smi": R.COMPLETED,
        "cuda_environment": R.COMPLETED,
        "pytorch": R.COMPLETED,
        "correlation": R.COMPLETED,
    }
    (error,) = report.collection_errors
    assert (error.collector, error.report_path) == ("pci", "/gpu/pci_observations")
    assert report.gpu.pci_observations.status is S.COLLECTION_FAILED
    assert report.gpu.devices == FIXTURES_BY_ID["FX01"].snapshot.gpu.devices


def test_a_gpu_hardware_failure_keeps_the_pci_section(monkeypatch):
    def explode(self):
        raise RuntimeError("synthetic NVML binding bug")

    monkeypatch.setattr(FakeNvmlSession, "device_count", explode)
    fixture = FIXTURES_BY_ID["FX17"]
    report, adapters = _inspect(fixture)
    assert _runs(report)["gpu_hardware"] is R.FAILED
    (error,) = _errors(report, "gpu_hardware")
    assert error.report_path == "/gpu" and error.device_ref is None
    assert report.gpu.pci_observations == fixture.snapshot.gpu.pci_observations
    assert report.gpu.device_count.status is S.COLLECTION_FAILED
    assert adapters.nvml.inits == adapters.nvml.shutdowns  # closed despite the failure
    jsonschema.validate(json.loads(render_json(report)), load_report_schema())


def test_multiple_failures_stay_separate(monkeypatch):
    def explode(self):
        raise RuntimeError("synthetic NVML open crash")

    fixture = FIXTURES_BY_ID["FX01"]
    adapters = FakeAdapterSet.from_scenario(fixture)
    adapters.platform = _ExplodingPlatform.from_snapshot(fixture.snapshot)
    monkeypatch.setattr(type(adapters.nvml), "open", explode)
    factory = _ScannerFactory(FakeSysfsReader(failures={"list": AssertionError("touched")}))
    report, _ = _inspect(fixture, adapters=adapters, factory=factory)
    assert _runs(report) == {
        "system": R.FAILED,
        "pci": R.COMPLETED,  # fail-closed: UNKNOWN, not scanned
        "nvml": R.FAILED,
        "driver": R.COMPLETED,  # every NVML read repeats the reported open failure
        "gpu_hardware": R.COMPLETED,  # NVML unavailable → nvidia-smi fallback
        "runtime": R.COMPLETED,  # from the same cached nvidia-smi rows
        "nvidia_smi": R.COMPLETED,
        "cuda_environment": R.COMPLETED,
        "pytorch": R.COMPLETED,
        "correlation": R.COMPLETED,
    }
    assert [(e.collector, e.report_path) for e in report.collection_errors] == [
        ("nvml", "/driver/nvml_version"),
        ("system", "/system"),
    ]
    assert report.gpu.inventory_source.value == "NVIDIA_SMI"
    assert report.gpu.device_count.is_collected
    assert report.driver.driver_version.method == "fixture:nvidia-smi --query-gpu=driver_version"
    assert [r.device_ref for r in report.runtime] == ["smi:0"]
    jsonschema.validate(json.loads(render_json(report)), load_report_schema())


def test_the_process_opt_out_reaches_the_runtime_run():
    report, adapters = _inspect_with(
        FIXTURES_BY_ID["FX01"], InspectionOptions(include_processes=False)
    )
    (metrics,) = report.runtime
    assert metrics.processes.status is S.SKIPPED
    assert "nvmlDeviceGetComputeRunningProcesses_v3" not in adapters.nvml.calls
    assert report.options.include_processes is False


def test_a_runtime_crash_is_contained_and_the_session_still_closes(monkeypatch):
    def explode(self):
        raise RuntimeError("synthetic telemetry bug")

    monkeypatch.setattr(FakeNvmlDevice, "utilization", explode)
    fixture = FIXTURES_BY_ID["FX06"]
    report, adapters = _inspect(fixture)
    assert _runs(report)["runtime"] is R.FAILED
    (error,) = _errors(report, "runtime")
    assert (error.report_path, error.device_ref) == ("/runtime", None)
    assert [m.device_ref for m in report.runtime] == ["nvml:0", "nvml:1"]
    assert all(m.memory_free_bytes.status is S.COLLECTION_FAILED for m in report.runtime)
    assert report.gpu == fixture.snapshot.gpu and report.driver == fixture.snapshot.driver
    assert adapters.nvml.inits == adapters.nvml.shutdowns


def test_an_nvidia_smi_crash_keeps_the_nvml_driver_facts(monkeypatch):
    def explode(self):
        raise RuntimeError("synthetic nvidia-smi bug")

    monkeypatch.setattr(FakeNvidiaSmi, "version", explode)
    fixture = FIXTURES_BY_ID["FX01"]
    report, _ = _inspect(fixture)
    assert _runs(report)["nvidia_smi"] is R.FAILED
    (error,) = _errors(report, "nvidia_smi")
    assert error.report_path == "/driver/nvidia_smi_path"
    driver = report.driver
    assert driver.driver_version == fixture.snapshot.driver.driver_version
    assert driver.nvidia_smi_path.status is S.COLLECTION_FAILED
    jsonschema.validate(json.loads(render_json(report)), load_report_schema())


def test_capabilities_have_one_owner_each():
    report, _ = _inspect(FIXTURES_BY_ID["FX13"])
    capabilities = [u.capability for u in report.unsupported_capabilities]
    assert capabilities == [
        "CUDA.TOOLKIT",
        "GPU.PCI_SCAN",
        "NVIDIA.ALL",
        "SYSTEM.MEMORY_AVAILABLE",
    ]
    assert len(set(capabilities)) == len(capabilities)


# --- report assembly ----------------------------------------------------------------------


def _assemble(snapshot):
    return assemble_report(
        snapshot,
        options=InspectionOptions(),
        generated_at=FIXTURES_BY_ID["FX01"].report().generated_at,
        duration_ms=1,
        data_origin=DataOrigin.FIXTURE,
        limitations=("extra", SNAPSHOT_LIMITATION, "extra"),
    )


def test_assembly_rejects_duplicate_capability_records_and_runs():
    snapshot = FIXTURES_BY_ID["FX13"].snapshot
    record = snapshot.unsupported_capabilities[0]
    with pytest.raises(InvalidModelError, match="more than one owner"):
        _assemble(
            dataclasses.replace(
                snapshot, unsupported_capabilities=(*snapshot.unsupported_capabilities, record)
            )
        )
    run = CollectorRun(collector="system", status=R.COMPLETED, duration_ms=1)
    with pytest.raises(InvalidModelError, match="more than once"):
        _assemble(dataclasses.replace(snapshot, collector_runs=(*snapshot.collector_runs, run)))


def test_assembly_states_the_snapshot_limitation_first_and_once():
    report = _assemble(FIXTURES_BY_ID["FX01"].snapshot)
    assert report.limitations == (SNAPSHOT_LIMITATION, "extra")


def test_mode_b_is_refused_until_readiness_exists():
    inspector = Inspector(FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"]).adapter_set())
    with pytest.raises(NotImplementedError, match="IMP-R01"):
        inspector.run(InspectionOptions(required_vram_gib=8))


def test_the_inspector_needs_an_adapter_set():
    with pytest.raises(TypeError, match="AdapterSet"):
        Inspector(FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"]))
