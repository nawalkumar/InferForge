"""IMP-K03 driver collectors (DEC-46): the `driver` run (NVML facts from the shared session)
and the `nvidia_smi` run (location, header, fallbacks). Fixture parity, version precedence,
disagreement, failure handling, no re-execution and the guard fallbacks. No NVML library,
process or sysfs access."""

import dataclasses

import pytest
from gsi_testkit.builders import COLLECTOR_ORDER, failed, ok
from gsi_testkit.fakes import FakeCommandRunner, FakeNvidiaSmi, FakeNvml, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    NvidiaSmiCli,
    SmiGpuRow,
    SmiVersion,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
)
from gpu_system_inspector.collectors import DriverCollector, NvidiaSmiCollector
from gpu_system_inspector.core.guard import run_guarded
from gpu_system_inspector.models import (
    DRIVER_SUPPORTED_CUDA_NOTE,
    CollectionError,
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    Observed,
    Source,
)

S = CollectionStatus
R = CollectorRunStatus
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
# Fixture runs last 5 ms × their position in COLLECTOR_ORDER.
STEP = {name: 5 * (i + 1) for i, name in enumerate(COLLECTOR_ORDER)}
FX01, FX03, FX04A, FX13 = (FIXTURES_BY_ID[i] for i in ("FX01", "FX03", "FX04a", "FX13"))
SMI = "/usr/bin/nvidia-smi"
HEADER = "| NVIDIA-SMI 545.29.06   Driver Version: 545.29.06   CUDA Version: 12.3 |"


def _driver(snapshot, nvml=None):
    nvml = nvml or FakeNvml.from_snapshot(snapshot)
    session = nvml.open()
    outcome = DriverCollector(nvml, session, FixedClock(step_ms=STEP["driver"])).collect()
    session.close()
    return outcome


def _smi_run(driver, smi):
    return NvidiaSmiCollector(smi, FixedClock(step_ms=STEP["nvidia_smi"]), driver=driver).collect()


class _Smi:
    """A `NvidiaSmiAdapter` with fixed answers that records what was asked."""

    provenance = DataOrigin.FIXTURE

    def __init__(self, path, header, rows=None):
        self._path, self._header, self._rows = path, header, rows
        self.calls: list[str] = []

    def locate(self):
        self.calls.append("locate")
        return self._path

    def version(self):
        self.calls.append("version")
        return self._header

    def query_gpus(self):
        self.calls.append("query")
        if self._rows is None:
            raise AssertionError("the query must not be needed")
        return self._rows


def _path():
    return ok(SMI, Source.NVIDIA_SMI, method="shutil.which")


def _header(driver="550.54.15", cuda="12.4"):
    return SmiVersion(
        driver_version=ok(driver, Source.NVIDIA_SMI, method="nvidia-smi header"),
        cuda_version=ok(cuda, Source.NVIDIA_SMI, method="nvidia-smi header"),
    )


def _rows(driver_version):
    row = SmiGpuRow(index=0, values={"driver_version": driver_version})
    return Observed.collected((row,), source=Source.NVIDIA_SMI, method="nvidia-smi --query-gpu")


def _nvml_failure():
    return failed(S.NOT_DETECTED, "NVML_ERROR_LIBRARY_NOT_FOUND", "NVML Shared Library Not Found")


def _without_nvml(snapshot=FX01.snapshot):
    """The driver run's section when NVML has no driver (FX03's NVML facts)."""
    return _driver(FX03.snapshot).section if snapshot is FX01.snapshot else snapshot


# --- parity -------------------------------------------------------------------------------


@ids
def test_both_runs_reproduce_every_fixture_driver_section(fixture):
    snapshot = fixture.snapshot
    driver = _driver(snapshot, FakeNvml.from_scenario(fixture))
    smi = FakeNvidiaSmi.from_snapshot(snapshot)
    outcome = _smi_run(driver.section, smi)
    assert outcome.section == snapshot.driver
    runs = {r.collector: r for r in snapshot.collector_runs}
    assert driver.run == runs["driver"] and outcome.run == runs["nvidia_smi"]
    expected = {c: tuple(e for e in snapshot.collection_errors if e.collector == c) for c in runs}
    assert driver.errors == expected["driver"] and outcome.errors == expected["nvidia_smi"]
    assert driver.unsupported == () and outcome.unsupported == ()


@ids
def test_nvidia_smi_is_never_queried_when_nvml_reports_the_driver(fixture):
    snapshot = fixture.snapshot
    smi = FakeNvidiaSmi.from_snapshot(snapshot)
    _smi_run(_driver(snapshot).section, smi)
    queried = "nvidia-smi --query-gpu" in smi.calls
    assert queried is (
        not snapshot.driver.nvml_version.is_collected
        and snapshot.driver.nvidia_smi_path.is_collected
    )
    assert smi.calls[:2] == ["locate", "nvidia-smi"]  # located once, header read once


def test_the_driver_run_never_opens_nvml_and_reads_only_system_functions():
    nvml = FakeNvml.from_snapshot(FX01.snapshot)
    session = nvml.open()
    DriverCollector(nvml, session, FixedClock()).collect()
    assert nvml.calls == [
        "nvmlInit",
        "nvmlSystemGetDriverVersion",
        "nvmlSystemGetCudaDriverVersion",
        "nvmlSystemGetCudaDriverVersion_v2",
    ]
    assert nvml.metadata_calls == [
        "importlib.metadata.version",
        "importlib.metadata.packages_distributions",
    ]
    session.close()


# --- precedence and disagreement ----------------------------------------------------------


def test_nvml_is_authoritative_and_a_disagreeing_nvidia_smi_stays_visible():
    driver = _driver(FX01.snapshot).section
    smi = _Smi(_path(), _header(driver="545.29.06", cuda="12.3"))
    section = _smi_run(driver, smi).section
    assert section.driver_version == driver.driver_version  # NVML's 550.54.15
    assert section.driver_supported_cuda_version == driver.driver_supported_cuda_version
    assert section.nvidia_smi_driver_version.value == "545.29.06"
    assert section.nvidia_smi_driver_version.note == "Differs from NVML, which reports 550.54.15."
    assert section.nvidia_smi_cuda_version.note == "Differs from NVML, which reports 12.4."
    assert smi.calls == ["locate", "version"]


def test_without_nvml_the_query_then_the_header_provide_the_driver_version():
    nvml = _without_nvml()
    queried = ok("550.54.15", Source.NVIDIA_SMI, method="nvidia-smi --query-gpu=driver_version")
    from_query = _smi_run(nvml, _Smi(_path(), _header(), _rows(queried))).section
    assert from_query.driver_version == queried
    no_value = failed(S.NOT_SUPPORTED, "NVIDIA_SMI_NA", "[N/A]", source=Source.NVIDIA_SMI)
    from_header = _smi_run(nvml, _Smi(_path(), _header(), _rows(no_value))).section
    assert from_header.driver_version == _header().driver_version
    cuda = from_header.driver_supported_cuda_version
    assert (cuda.value, cuda.source, cuda.note) == (
        "12.4",
        Source.NVIDIA_SMI,
        DRIVER_SUPPORTED_CUDA_NOTE,
    )


def test_without_any_source_the_nvml_reason_is_kept():
    nvml = _without_nvml()
    failure = failed(S.COLLECTION_FAILED, "EXIT_9", "driver gone", source=Source.NVIDIA_SMI)
    header = SmiVersion(driver_version=failure, cuda_version=failure)
    outcome = _smi_run(nvml, _Smi(_path(), header, failure))
    assert outcome.section.driver_version == nvml.driver_version  # NVML's own reason
    assert outcome.section.driver_supported_cuda_version == nvml.driver_supported_cuda_version
    assert outcome.run.status is R.PARTIAL
    assert outcome.errors == (
        CollectionError(
            collector="nvidia_smi",
            device_ref=None,
            status=S.COLLECTION_FAILED,
            error_code="EXIT_9",
            message="driver gone",
            report_path="/driver/nvidia_smi_driver_version",
        ),
    )


def test_a_cuda_toolkit_version_never_enters_the_driver_section():
    """AC04 with distinct values: the driver supports 12.4; no toolkit field is read."""
    section = _smi_run(_driver(FX01.snapshot).section, _Smi(_path(), _header())).section
    texts = [getattr(section, f.name).note or "" for f in dataclasses.fields(section)]
    assert section.driver_supported_cuda_version.note == DRIVER_SUPPORTED_CUDA_NOTE
    assert not any("Toolkit version" in t and "Not the" not in t for t in texts)


# --- failures -----------------------------------------------------------------------------


def test_nvidia_smi_not_installed_is_a_fact_not_a_failure():
    missing = failed(S.NOT_DETECTED, "FileNotFoundError", "nvidia-smi not found")
    header = SmiVersion(driver_version=missing, cuda_version=missing)
    smi = _Smi(missing, header)
    outcome = _smi_run(_driver(FX01.snapshot).section, smi)
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()
    assert outcome.section.nvidia_smi_path == missing and smi.calls == ["locate", "version"]


def test_a_failed_lookup_is_reported_at_the_path():
    denied = failed(S.PERMISSION_DENIED, "PermissionError", "denied", source=Source.OS)
    smi = _Smi(denied, SmiVersion(driver_version=denied, cuda_version=denied))
    outcome = _smi_run(_driver(FX01.snapshot).section, smi)
    (error,) = outcome.errors
    assert (error.status, error.report_path) == (S.PERMISSION_DENIED, "/driver/nvidia_smi_path")
    assert outcome.run.status is R.PARTIAL


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("NVML_ERROR_UNKNOWN", S.COLLECTION_FAILED),
        ("NVML_ERROR_NO_PERMISSION", S.PERMISSION_DENIED),
    ],
)
def test_an_nvml_read_failure_after_init_is_field_level_and_partial(code, status):
    nvml = FakeNvml.from_snapshot(FX01.snapshot)
    nvml.driver = dataclasses.replace(
        nvml.driver, driver_version=failed(status, code, code, method="nvmlSystemGetDriverVersion")
    )
    outcome = _driver(FX01.snapshot, nvml)
    assert outcome.section.driver_version.status is status
    assert outcome.section.cuda_driver_library.is_collected
    assert outcome.run.status is R.PARTIAL and outcome.errors == ()


@pytest.mark.parametrize("fixture", [FX03, FX04A, FX13], ids=lambda f: f.fixture_id)
def test_without_an_nvml_session_the_driver_run_completes(fixture):
    """No driver, a missing binding, macOS: the `nvml` run already reported the reason."""
    outcome = _driver(fixture.snapshot)
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()
    assert outcome.section.nvml_version == fixture.snapshot.driver.nvml_version


def test_the_guard_fallbacks_keep_what_the_other_run_observed():
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    driver = _driver(FX01.snapshot).section
    collector = NvidiaSmiCollector(_Smi(_path(), _header()), FixedClock(), driver=driver)
    kept = collector.unavailable(error)
    assert kept.driver_version == driver.driver_version and kept.nvml_version.is_collected
    for name in ("nvidia_smi_path", "nvidia_smi_driver_version", "nvidia_smi_cuda_version"):
        value = getattr(kept, name)
        assert (value.status, value.error) == (S.COLLECTION_FAILED, error)
    nvml = FakeNvml.from_snapshot(FX01.snapshot)
    lost = DriverCollector(nvml, nvml.open(), FixedClock()).unavailable(error)
    assert lost.driver_version.error == error and lost.nvml_binding.error == error


def test_a_driver_crash_is_contained_and_the_smi_run_still_completes():
    class Exploding:
        provenance = DataOrigin.FIXTURE

        def binding_version(self):
            raise RuntimeError("binding metadata bug")

    nvml = FakeNvml.from_snapshot(FX01.snapshot)
    session = nvml.open()
    clock = FixedClock()
    driver = run_guarded(DriverCollector(Exploding(), session, clock), clock, report_path="/driver")
    assert driver.run.status is R.FAILED
    assert driver.errors[0].message == "RuntimeError: binding metadata bug"
    smi = _smi_run(driver.section, FakeNvidiaSmi.from_snapshot(FX01.snapshot))
    assert smi.section.nvidia_smi_path.is_collected and smi.run.status is R.COMPLETED
    assert smi.section.nvml_version.status is S.COLLECTION_FAILED  # the NVML part failed
    assert smi.section.driver_version.source is Source.NVIDIA_SMI  # so precedence falls back
    session.close()


# --- the production adapter: no re-execution ----------------------------------------------


def test_the_production_adapter_runs_each_command_once_per_inspection():
    runner = (
        FakeCommandRunner(executables={"nvidia-smi": SMI})
        .on(*nvidia_smi_query_argv(SMI), stdout="")
        .on(*nvidia_smi_version_argv(SMI), stdout=HEADER)
    )
    smi = NvidiaSmiCli(runner)
    assert smi.query_gpus() is smi.query_gpus()  # the GPU hardware fallback, then the driver's
    first = smi.version()
    assert smi.version() is first
    outcome = _smi_run(_without_nvml(), smi)
    assert [c.argv for c in runner.calls] == [
        nvidia_smi_query_argv(SMI),
        nvidia_smi_version_argv(SMI),
    ]
    assert runner.lookups == ["nvidia-smi"]
    assert outcome.section.nvidia_smi_driver_version.value == "545.29.06"
    assert outcome.section.driver_version == outcome.section.nvidia_smi_driver_version
