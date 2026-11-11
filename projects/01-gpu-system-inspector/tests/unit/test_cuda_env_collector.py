"""IMP-K05 `CudaEnvironmentCollector` (DEC-47) over each fixture's production locator
(`CudaToolkitLocator` on `FakeToolkitFiles` and `FakeCommandRunner`), plus the platform
adapter's `CUDA_PATH_V*` listing (KI-28). No real filesystem, process or hardware."""

import dataclasses
import errno

import pytest
from gsi_testkit.builders import COLLECTOR_ORDER
from gsi_testkit.fakes import (
    FakeAdapterSet,
    FakeCommandRunner,
    FakeHostApi,
    FakeToolkitFiles,
    FixedClock,
    version_json,
)
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import CudaToolkitLocator, OsPlatformAdapter, StdlibHostApi
from gpu_system_inspector.collectors import CUDA_TOOLKIT, CudaEnvironmentCollector
from gpu_system_inspector.core.guard import run_guarded
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    CollectionStatus,
    CollectorRunStatus,
    ErrorDetail,
    Observed,
    PlatformProfile,
    Source,
    ToolkitDiscovery,
)

S = CollectionStatus
R = CollectorRunStatus
STEP_MS = 5 * (COLLECTOR_ORDER.index("cuda_environment") + 1)
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
# FX10's CUDA collector crashes by design; its parity is checked through the guard above.
running = pytest.mark.parametrize(
    "fixture", [f for f in ALL_FIXTURES if not f.collector_failures], ids=lambda f: f.fixture_id
)
FX01, FX08, FX12, FX13 = (FIXTURES_BY_ID[i] for i in ("FX01", "FX08", "FX12", "FX13"))


def _collector(adapters, clock=None):
    return CudaEnvironmentCollector(
        adapters.toolkit, adapters.platform, clock or FixedClock(step_ms=STEP_MS)
    )


class _RecordingLocator:
    def __init__(self, result):
        self.result, self.hints = result, []
        self.provenance = result

    def discover(self, hints):
        self.hints.append(dict(hints))
        return self.result


# --- parity -------------------------------------------------------------------------------


@ids
def test_collector_reproduces_every_fixture_cuda_environment(fixture):
    """Whole section, run, errors and `CUDA.*` records; FX10's injected crash through the
    guard, as the inspector runs it."""
    snapshot = fixture.snapshot
    adapters = FakeAdapterSet.from_scenario(fixture)
    clock = FixedClock(step_ms=STEP_MS)
    outcome = run_guarded(_collector(adapters, clock), clock, report_path="/cuda_environment")
    assert outcome.section == snapshot.cuda_environment
    (run,) = (r for r in snapshot.collector_runs if r.collector == "cuda_environment")
    if run.status is R.FAILED:  # guard-timed: the fixed-step clock ticks once more
        assert (outcome.run.collector, outcome.run.status) == (run.collector, run.status)
    else:
        assert outcome.run == run
    assert outcome.errors == tuple(
        e for e in snapshot.collection_errors if e.collector == "cuda_environment"
    )
    assert outcome.unsupported == tuple(
        u for u in snapshot.unsupported_capabilities if u.capability.startswith("CUDA.")
    )


@running
def test_no_driver_value_and_no_unsupported_signal_reaches_the_section(fixture):
    """FR-07: toolkit versions come from version.json or nvcc only, never NVML/nvidia-smi."""
    outcome = _collector(FakeAdapterSet.from_scenario(fixture)).collect()
    if not outcome.section.toolkits.is_collected:
        assert outcome.section.toolkits.value is None
        return
    for candidate in outcome.section.toolkits.value:
        assert candidate.version.method.removeprefix("fixture:") in (
            "version.json:cuda.version",
            "nvcc --version",
        )
        assert candidate.version.source is Source.CUDA_TOOLKIT


@running
def test_discovery_is_deterministic(fixture):
    first = _collector(FakeAdapterSet.from_scenario(fixture)).collect()
    assert _collector(FakeAdapterSet.from_scenario(fixture)).collect() == first


# --- environment and hints ----------------------------------------------------------------


def test_the_whole_allowlist_is_read_and_only_toolkit_variables_become_hints():
    adapters = FakeAdapterSet.from_scenario(FX01)
    locator = _RecordingLocator(adapters.toolkit.discover({}))
    outcome = CudaEnvironmentCollector(locator, adapters.platform, FixedClock()).collect()
    assert sorted(outcome.section.environment) == sorted(ENVIRONMENT_ALLOWLIST)
    assert locator.hints == [
        {
            ToolkitDiscovery.CUDA_HOME: ("/usr/local/cuda-12.4",),
            ToolkitDiscovery.CUDA_PATH: (),
            ToolkitDiscovery.CUDA_PATH_VERSIONED: (),
        }
    ]


def test_windows_versioned_paths_are_read_and_passed_on():
    adapters = FakeAdapterSet.from_scenario(FX12)
    outcome = _collector(adapters).collect()
    assert "CUDA_PATH_V12_4" in outcome.section.environment
    assert "versioned_cuda_path_names" in adapters.platform.calls


def test_an_empty_variable_is_recorded_but_is_no_hint():
    adapters = FakeAdapterSet.from_scenario(FX08)
    env = dict(adapters.platform.environment)
    env["CUDA_HOME"] = Observed.collected(
        "", source=Source.ENVIRONMENT, method="fixture:os.environ"
    )
    adapters.platform.environment = env
    locator = _RecordingLocator(adapters.toolkit.discover({}))
    outcome = CudaEnvironmentCollector(locator, adapters.platform, FixedClock()).collect()
    assert outcome.section.environment["CUDA_HOME"].value == ""
    assert locator.hints[0][ToolkitDiscovery.CUDA_HOME] == ()


# --- statuses, errors, capability ----------------------------------------------------------


def test_no_toolkit_is_informational_and_completed():
    outcome = _collector(FakeAdapterSet.from_scenario(FX08)).collect()
    assert outcome.section.toolkits.status is S.NOT_DETECTED
    assert outcome.run.status is R.COMPLETED and outcome.errors == () and outcome.unsupported == ()


def test_a_failed_default_listing_is_one_error_and_partial():
    adapters = FakeAdapterSet.from_scenario(FX01)
    adapters.toolkit_files.fail("list_dir", "/usr/local", PermissionError(errno.EACCES, "denied"))
    outcome = _collector(adapters).collect()
    (candidate,) = outcome.section.toolkits.value  # CUDA_HOME and nvcc still found it
    assert ToolkitDiscovery.DEFAULT_LOCATION not in candidate.discovered_via
    (error,) = outcome.errors
    assert (error.collector, error.status, error.report_path) == (
        "cuda_environment",
        S.PERMISSION_DENIED,
        "/cuda_environment/toolkits",
    )
    assert error.message.startswith("/usr/local: ")
    assert outcome.run.status is R.PARTIAL


def test_a_failing_compiler_is_field_level_and_partial():
    adapters = FakeAdapterSet.from_scenario(FX01)
    nvcc = "/usr/local/cuda-12.4/bin/nvcc"
    adapters.commands.responses.clear()
    adapters.commands.on(nvcc, "--version", timeout=True)
    outcome = _collector(adapters).collect()
    assert outcome.section.active_nvcc_version.error.code == "TIMEOUT"
    assert outcome.section.toolkits.value[0].version.value == "12.4.131"
    assert outcome.errors == () and outcome.run.status is R.PARTIAL


@pytest.mark.parametrize(
    ("profile", "reason"),
    [
        (PlatformProfile.MACOS, "The CUDA Toolkit is not available on macOS."),
        (PlatformProfile.OTHER, "The CUDA Toolkit is not supported on this platform."),
    ],
)
def test_unsupported_platforms_own_one_cuda_toolkit_record(profile, reason):
    adapters = FakeAdapterSet.from_scenario(FX13)
    adapters.platform.system = dataclasses.replace(
        adapters.platform.system, platform_profile=profile
    )
    outcome = _collector(adapters).collect()
    (record,) = outcome.unsupported
    assert (record.capability, record.reason, record.device_ref) == (CUDA_TOOLKIT, reason, None)
    assert outcome.run.status is R.COMPLETED
    assert adapters.toolkit_files.calls == [] and adapters.commands.calls == []


def test_the_guard_fallback_is_the_unavailable_section():
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: x")
    section = _collector(FakeAdapterSet.from_scenario(FX01)).unavailable(error)
    assert section.toolkits.error == error and section.environment == {}


def test_the_collector_touches_no_file_or_process_itself():
    adapters = FakeAdapterSet.from_scenario(FX01)
    locator = _RecordingLocator(adapters.toolkit.discover({}))
    files_before, calls_before = len(adapters.toolkit_files.calls), len(adapters.commands.calls)
    CudaEnvironmentCollector(locator, adapters.platform, FixedClock()).collect()
    assert len(adapters.toolkit_files.calls) == files_before
    assert len(adapters.commands.calls) == calls_before


# --- the platform adapter's CUDA_PATH_V* listing (KI-28) ----------------------------------


def test_versioned_cuda_path_names_are_listed_on_windows_only_and_filtered():
    environment = {
        "CUDA_PATH_V12_4": "C:/CUDA/v12.4",
        "CUDA_PATH_V11_8": "C:/CUDA/v11.8",
        "CUDA_PATH": "C:/CUDA/v12.4",
        "AWS_SECRET_ACCESS_KEY": "never read",
        "cuda_path_v1": "lowercase",
    }
    windows = FakeHostApi.windows_host(environment=environment)
    assert OsPlatformAdapter(windows).versioned_cuda_path_names() == (
        "CUDA_PATH_V11_8",
        "CUDA_PATH_V12_4",
    )
    assert windows.env_reads == []  # names only; no value read here
    linux = FakeHostApi.linux(environment=environment)
    assert OsPlatformAdapter(linux).versioned_cuda_path_names() == ()
    assert "environ_names" not in linux.calls


def test_the_production_locator_and_collector_on_a_synthetic_windows_host():
    root = "C:/CUDA/v12.4"
    host = FakeHostApi.windows_host(environment={"CUDA_PATH": root, "CUDA_PATH_V12_4": root})
    platform = OsPlatformAdapter(host)
    files = FakeToolkitFiles().add_directory(root).add_file(f"{root}/bin/nvcc.exe")
    files.add_file(f"{root}/version.json", version_json("12.4.131"))
    locator = CudaToolkitLocator(FakeCommandRunner(), platform, files)
    outcome = CudaEnvironmentCollector(locator, platform, FixedClock()).collect()
    (candidate,) = outcome.section.toolkits.value
    assert candidate.discovered_via == (
        ToolkitDiscovery.CUDA_PATH,
        ToolkitDiscovery.CUDA_PATH_VERSIONED,
    )
    assert candidate.version.value == "12.4.131"
    assert candidate.nvcc_path.value == f"{root}/bin/nvcc.exe"
    assert outcome.section.environment["CUDA_PATH_V12_4"].value == root


def test_the_real_host_api_lists_names_without_values(monkeypatch):
    monkeypatch.setenv("CUDA_PATH_V9_9", "C:/CUDA/v9.9")
    names = StdlibHostApi().environ_names()
    assert "CUDA_PATH_V9_9" in names and all(isinstance(n, str) for n in names)
