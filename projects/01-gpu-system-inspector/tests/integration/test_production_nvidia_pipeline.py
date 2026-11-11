"""IMP-A01/A02/A05 integration: the production NVML and `nvidia-smi` adapters behind the
production `Inspector`, over a stub `pynvml` and the fake command runner. Covers the NVML /
`nvidia-smi` fallback matrix (DEC-45), session cleanup, the single PCI scan, capability
ownership, the driver and runtime runs (IMP-K03/K04, DEC-46) and the report contract. No
library is loaded and no process is started."""

import json

import jsonschema
import pytest
from gsi_testkit.fakes import (
    FakeAdapterSet,
    FakeCommandRunner,
    FakePlatform,
    StubGpu,
    StubPynvml,
)
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    NVML_READ_ONLY_FUNCTIONS,
    AdapterSet,
    NvidiaSmiCli,
    PynvmlAdapter,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
)
from gpu_system_inspector.core import Inspector
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    Source,
)
from gpu_system_inspector.reporting import load_report_schema, parse_report_json, render_json

S = CollectionStatus
R = CollectorRunStatus
SMI = "/usr/bin/nvidia-smi"
GPU_LOST = 15
HEADER = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |"
QUERY, VERSION = nvidia_smi_query_argv(SMI), nvidia_smi_version_argv(SMI)
ROW = (
    "0, GPU-cccc, NVIDIA RTX A6000, 00000000:41:00.0, 49140, 1024, 48116, 40, 3, 1, 70.5, "
    "300.00, 550.54.15"
)


def _inspect(fixture_id, *, stub=None, runner=None, platform=None):
    fixture = FIXTURES_BY_ID[fixture_id]
    fakes = FakeAdapterSet.from_scenario(fixture)
    platform = platform or fakes.platform
    stub = stub or StubPynvml.from_snapshot(fixture.snapshot)
    runner = runner if runner is not None else FakeCommandRunner()
    adapters = AdapterSet(
        platform=platform,
        pci=fakes.pci,
        nvml=PynvmlAdapter(stub.loader, platform=platform, provenance=DataOrigin.FIXTURE),
        nvidia_smi=NvidiaSmiCli(runner, platform),
        commands=runner,
        toolkit=fakes.toolkit,
        torch_probe=fakes.torch_probe,
        clock=fakes.clock,
        provenance=DataOrigin.FIXTURE,
    )
    return Inspector(adapters).run(), stub, runner, fakes


def _smi(header=None, **response):
    """`nvidia-smi` on PATH with a canned query response and a canned plain run (the
    header; by default a working one)."""
    runner = FakeCommandRunner(executables={"nvidia-smi": SMI}).on(*QUERY, **response)
    return runner.on(*VERSION, **(header or {"stdout": HEADER}))


def _runs(report):
    return {r.collector: r.status for r in report.collector_runs}


def _valid(report):
    text = render_json(report)
    jsonschema.validate(json.loads(text), load_report_schema())
    assert parse_report_json(text) == report


# --- the fallback matrix ------------------------------------------------------------------


FAILING = {"returncode": 9, "stderr": "driver gone"}


@pytest.mark.parametrize(
    ("smi", "header_runs", "smi_run"),
    [
        (FakeCommandRunner(), [], R.COMPLETED),  # not installed is a fact, not a failure
        (_smi(stdout=ROW), [VERSION], R.COMPLETED),
        (_smi(header=FAILING, **FAILING), [VERSION], R.PARTIAL),
    ],
    ids=["smi-missing", "smi-works", "smi-fails"],
)
def test_working_nvml_takes_precedence_and_the_inventory_query_is_never_run(
    smi, header_runs, smi_run
):
    """The `nvidia_smi` run still reads the header once for `DriverInfo` (§4.11)."""
    report, stub, runner, _ = _inspect("FX01", runner=smi)
    assert report.gpu == FIXTURES_BY_ID["FX01"].snapshot.gpu
    assert report.gpu.inventory_source is Source.NVML
    assert [c.argv for c in runner.calls] == header_runs and runner.lookups == ["nvidia-smi"]
    assert report.driver.driver_version.source is Source.NVML  # NVML is authoritative
    runs = _runs(report)
    assert runs.pop("nvidia_smi") is smi_run
    assert set(runs.values()) == {R.COMPLETED}
    _valid(report)


def test_nvml_failure_with_a_working_nvidia_smi_keeps_the_cli_inventory():
    report, stub, runner, _ = _inspect("FX03", runner=_smi(stdout=ROW))
    gpu = report.gpu
    assert gpu.inventory_source is Source.NVIDIA_SMI and gpu.device_count.value == 1
    (device,) = gpu.devices
    assert device.device_ref == "smi:0" and device.name.value == "NVIDIA RTX A6000"
    assert device.memory_total_bytes.value == 49140 * 2**20
    assert device.compute_capability.error.code == "FALLBACK_FIELD_UNAVAILABLE"
    assert report.collection_errors == () and _runs(report)["gpu_hardware"] is R.COMPLETED
    assert [c.argv for c in runner.calls] == [QUERY, VERSION]  # each once, shared
    driver = report.driver
    assert (driver.driver_version.value, driver.driver_version.source) == (
        "550.54.15",
        Source.NVIDIA_SMI,
    )
    assert driver.driver_supported_cuda_version.value == "12.4"
    assert driver.nvml_version.error.code == "NVML_ERROR_LIBRARY_NOT_FOUND"
    (metrics,) = report.runtime
    assert metrics.device_ref == "smi:0" and metrics.memory_free_bytes.value == 48116 * 2**20
    assert metrics.processes.error.code == "FALLBACK_FIELD_UNAVAILABLE"
    _valid(report)


def test_partial_nvml_evidence_is_kept_and_not_merged_with_nvidia_smi():
    stub = StubPynvml.from_snapshot(FIXTURES_BY_ID["FX06"].snapshot)
    stub.faults[("nvmlDeviceGetHandleByIndex_v2", 1)] = GPU_LOST
    report, _, runner, _ = _inspect("FX06", stub=stub, runner=_smi(stdout=ROW))
    nvml0, nvml1 = report.gpu.devices
    assert nvml0 == FIXTURES_BY_ID["FX06"].snapshot.gpu.devices[0]
    assert nvml1.name.error.code == "NVML_ERROR_GPU_IS_LOST"
    assert [(e.collector, e.device_ref, e.report_path) for e in report.collection_errors] == [
        ("gpu_hardware", "nvml:1", "/gpu/devices/1"),
        ("runtime", "nvml:1", "/runtime/1"),  # each run saw the lost handle itself
    ]
    assert report.runtime[0].memory_free_bytes.is_collected
    assert report.runtime[1].memory_free_bytes.error.code == "NVML_ERROR_GPU_IS_LOST"
    assert _runs(report)["gpu_hardware"] is R.PARTIAL
    assert [c.argv for c in runner.calls] == [VERSION]  # header only, never the query
    _valid(report)


def test_both_paths_failing_is_an_honest_failure_not_an_empty_inventory():
    failing = {"returncode": 9, "stderr": "NVIDIA-SMI has failed: couldn't communicate"}
    report, _, _, _ = _inspect("FX17", runner=_smi(header=failing, **failing))
    gpu = report.gpu
    assert gpu.devices == () and gpu.inventory_source is None and gpu.device_count.value is None
    assert gpu.device_count.error.code == "NVML_ERROR_DRIVER_NOT_LOADED"
    assert [(e.collector, e.error_code, e.report_path) for e in report.collection_errors] == [
        ("gpu_hardware", "EXIT_9", "/gpu/devices"),
        ("nvidia_smi", "EXIT_9", "/driver/nvidia_smi_driver_version"),
    ]
    assert _runs(report)["gpu_hardware"] is R.PARTIAL
    assert _runs(report)["nvidia_smi"] is R.PARTIAL and report.runtime == ()
    assert report.driver.driver_version.error.code == "NVML_ERROR_DRIVER_NOT_LOADED"
    assert gpu.pci_observations == FIXTURES_BY_ID["FX17"].snapshot.gpu.pci_observations
    _valid(report)


def test_zero_devices_from_nvml_is_a_collected_zero():
    report, stub, runner, _ = _inspect("FX01", stub=StubPynvml([]), runner=_smi(stdout=""))
    assert report.gpu.device_count.is_collected and report.gpu.device_count.value == 0
    assert report.gpu.inventory_source is Source.NVML and report.runtime == ()
    assert [c.argv for c in runner.calls] == [VERSION]


def test_zero_devices_from_nvidia_smi_after_an_nvml_failure_is_the_tools_zero():
    report, _, _, _ = _inspect("FX03", runner=_smi(stdout=""))
    assert report.gpu.device_count.value == 0 and report.gpu.inventory_source is Source.NVIDIA_SMI


def test_both_tools_missing_is_not_reported_as_no_gpu():
    report, stub, runner, _ = _inspect("FX03", runner=FakeCommandRunner())
    count = report.gpu.device_count
    assert count.value is None and count.status is S.NOT_DETECTED
    assert count.error.code == "NVML_ERROR_LIBRARY_NOT_FOUND"  # about the library, not the GPU
    assert report.collection_errors == () and runner.calls == []
    assert runner.lookups == ["nvidia-smi"]
    _valid(report)


def test_a_missing_binding_falls_back_to_nvidia_smi_and_reports_the_broken_install():
    stub = StubPynvml(import_error="No module named 'pynvml'")
    report, _, _, _ = _inspect("FX01", stub=stub, runner=_smi(stdout=ROW))
    assert report.gpu.inventory_source is Source.NVIDIA_SMI
    (error,) = report.collection_errors
    assert (error.collector, error.status, error.report_path) == (
        "nvml",
        S.NOT_INSTALLED,
        "/driver/nvml_binding",
    )
    assert _runs(report)["nvml"] is R.PARTIAL


# --- lifecycle, single PCI scan, ownership ------------------------------------------------


@pytest.mark.parametrize("fixture_id", ["FX01", "FX03", "FX06", "FX13", "FX17"])
def test_one_nvml_session_one_pci_scan_and_read_only_calls(fixture_id):
    report, stub, _, fakes = _inspect(fixture_id)
    assert stub.loads == (0 if fixture_id == "FX13" else 1)
    assert stub.inits == stub.shutdowns <= 1
    assert set(stub.calls) <= NVML_READ_ONLY_FUNCTIONS
    assert fakes.pci.calls == ["/sys/bus/pci/devices"]
    assert render_json(_inspect(fixture_id)[0]) == render_json(report)  # deterministic


def test_macos_gates_both_tools_and_owns_each_capability_once():
    report, stub, runner, _ = _inspect("FX13", runner=FakeCommandRunner())
    assert report.gpu == FIXTURES_BY_ID["FX13"].snapshot.gpu
    assert [u.capability for u in report.unsupported_capabilities] == [
        "CUDA.TOOLKIT",
        "GPU.PCI_SCAN",
        "NVIDIA.ALL",
        "SYSTEM.MEMORY_AVAILABLE",
    ]
    assert stub.loads == 0 and runner.lookups == [] and runner.calls == []
    _valid(report)


def test_a_gpu_hardware_exception_keeps_pci_and_still_shuts_nvml_down():
    stub = StubPynvml.from_snapshot(FIXTURES_BY_ID["FX01"].snapshot)

    def explode(handle):
        raise RuntimeError("binding bug")

    stub.nvmlDeviceGetName = explode
    report, _, _, _ = _inspect("FX01", stub=stub)
    assert _runs(report)["gpu_hardware"] is R.FAILED
    (error,) = report.collection_errors
    assert error.collector == "gpu_hardware" and error.message == "RuntimeError: binding bug"
    assert report.gpu.pci_observations == FIXTURES_BY_ID["FX01"].snapshot.gpu.pci_observations
    assert stub.inits == stub.shutdowns == 1
    _valid(report)


def test_independent_failures_stay_separately_visible():
    class Exploding(FakePlatform):
        def system_info(self):
            raise RuntimeError("system crash")

    platform = Exploding.from_snapshot(FIXTURES_BY_ID["FX01"].snapshot)
    stub = StubPynvml(import_error="No module named 'pynvml'")
    failing = {"returncode": 12, "stderr": "NVML library not found"}
    runner = _smi(header=failing, **failing)
    report, _, _, _ = _inspect("FX01", stub=stub, runner=runner, platform=platform)
    assert [(e.collector, e.error_code) for e in report.collection_errors] == [
        ("gpu_hardware", "EXIT_12"),
        ("nvidia_smi", "EXIT_12"),
        ("nvml", "ImportError"),
        ("pci", "UNHANDLED_EXCEPTION"),
        ("system", "UNHANDLED_EXCEPTION"),
    ]
    assert _runs(report) == {
        "system": R.FAILED,
        "pci": R.FAILED,  # the pre-built native scanner is refused after the fallback
        "nvml": R.PARTIAL,
        "driver": R.COMPLETED,  # its reads repeat the reported binding failure
        "gpu_hardware": R.PARTIAL,
        "runtime": R.COMPLETED,  # no inventory, no records
        "nvidia_smi": R.PARTIAL,
        "cuda_environment": R.COMPLETED,
        "pytorch": R.COMPLETED,
        "correlation": R.COMPLETED,
    }
    _valid(report)


def test_a_shutdown_failure_does_not_hide_the_inventory():
    stub = StubPynvml.from_snapshot(FIXTURES_BY_ID["FX01"].snapshot)
    stub.shutdown_fault = 1
    report, _, _, _ = _inspect("FX01", stub=stub)
    assert report.gpu == FIXTURES_BY_ID["FX01"].snapshot.gpu and stub.shutdowns == 1
    assert report.collection_errors == ()


def test_telemetry_is_read_by_the_runtime_run_after_the_inventory():
    gpu = StubGpu(temperature=None)
    stub = StubPynvml([gpu])
    report, _, _, _ = _inspect("FX01", stub=stub)
    handles = [i for i, c in enumerate(stub.calls) if c == "nvmlDeviceGetHandleByIndex_v2"]
    assert len(handles) == 2  # one per run: gpu_hardware, then runtime
    assert stub.calls.index("nvmlDeviceGetTemperatureV") > handles[1]
    assert "nvmlDeviceGetTemperatureV" not in stub.calls[: handles[1]]
    (metrics,) = report.runtime
    assert metrics.temperature_gpu_celsius.error.code == "VALUE_NOT_AVAILABLE"
    assert metrics.memory_free_bytes.is_collected
    assert stub.inits == stub.shutdowns == 1
