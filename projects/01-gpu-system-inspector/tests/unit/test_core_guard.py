"""IMP-K08 core pieces (DEC-44): the collector contract, the guard, fail-closed environment
resolution, the NVML session step, and the production adapter set and its stand-ins.
No hardware, NVML, process or sysfs access."""

import dataclasses

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeHostApi, FakeSysfsReader, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    AdapterSet,
    AdapterUsageError,
    InProcessTorchProbe,
    NvidiaSmiCli,
    OsPlatformAdapter,
    PciSysfsAdapter,
    PynvmlAdapter,
    SubprocessCommandRunner,
    SubprocessTorchProbe,
    SystemClock,
    UnimplementedCommandRunner,
    UnimplementedNvidiaSmi,
    UnimplementedNvmlAdapter,
    UnimplementedTorchProbe,
)
from gpu_system_inspector.adapters.outcomes import NOT_IMPLEMENTED, UNHANDLED_EXCEPTION
from gpu_system_inspector.collectors import (
    Collector,
    GpuHardwareCollector,
    PciCollector,
    SystemCollector,
)
from gpu_system_inspector.core import (
    DeferredPciScanner,
    FailedNvmlSession,
    NvmlSessionStep,
    build_production_adapters,
    describe,
    production_pci_scanner,
    resolve_linux_environment,
    run_guarded,
)
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    DataOrigin,
    ErrorDetail,
    InspectionOptions,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    ProbeMode,
    Source,
    SystemInfo,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

S = CollectionStatus
R = CollectorRunStatus
E = LinuxEnvironment
P = PlatformProfile


class _Collector:
    """A minimal collector whose `collect()` is configurable."""

    name = "probe"

    def __init__(self, result):
        self._result = result
        self.fallbacks: list[ErrorDetail] = []

    def collect(self):
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    def unavailable(self, error: ErrorDetail):
        self.fallbacks.append(error)
        return Observed(
            value=None, status=S.COLLECTION_FAILED, source=None, method=None, error=error
        )


class _SteppingClock(FixedClock):
    def __init__(self, *ticks: int) -> None:
        super().__init__()
        self._values = iter(ticks)

    def monotonic_ms(self) -> int:
        return next(self._values)


def _ok(status=R.COMPLETED, errors=(), name="probe"):
    section = Observed.collected("x", source=Source.INSPECTOR, method="m")
    run = CollectorRun(collector=name, status=status, duration_ms=3)
    return CollectorOutcome(section=section, run=run, errors=errors)


# --- the collector contract ---------------------------------------------------------------


def test_every_collector_satisfies_the_one_contract():
    fx = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"])
    session = fx.nvml.open()
    production = [
        SystemCollector(fx.platform, fx.clock),
        PciCollector(fx.pci, fx.clock, environment=E.NATIVE_LINUX, profile=P.LINUX),
        GpuHardwareCollector(
            session,
            fx.nvidia_smi,
            fx.clock,
            pci=fx.pci.result,
            environment=E.NATIVE_LINUX,
            profile=P.LINUX,
        ),
        NvmlSessionStep(fx.nvml, fx.clock),
    ]
    for collector in [*production, *fx.collectors.values()]:
        assert isinstance(collector, Collector), collector
        assert isinstance(collector.name, str) and collector.name
    assert [c.name for c in production] == ["system", "pci", "gpu_hardware", "nvml"]
    session.close()


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_fake_collectors_return_outcomes_the_guard_accepts(fixture):
    collectors = FakeAdapterSet.from_scenario(fixture).collectors
    for name, collector in collectors.items():
        outcome = run_guarded(collector, FixedClock(), report_path=f"/{name}")
        expected = next(r for r in fixture.snapshot.collector_runs if r.collector == name)
        assert outcome.run.status is expected.status
        if name in fixture.collector_failures:
            (error,) = outcome.errors
            assert error.error_code == UNHANDLED_EXCEPTION and error.collector == name
        else:
            assert outcome.run == expected


# --- the guard ----------------------------------------------------------------------------


@pytest.mark.parametrize("status", [R.COMPLETED, R.PARTIAL])
def test_valid_outcomes_pass_through_unchanged(status):
    outcome = _ok(status)
    collector = _Collector(outcome)
    assert run_guarded(collector, FixedClock(), report_path="/probe") is outcome
    assert collector.fallbacks == []


@pytest.mark.parametrize("status", [S.NOT_DETECTED, S.NOT_SUPPORTED])
def test_expected_host_states_are_not_failures(status):
    section = Observed.failed(status, code="X", message="host fact", source=Source.OS)
    run = CollectorRun(collector="probe", status=R.COMPLETED, duration_ms=0)
    outcome = run_guarded(
        _Collector(CollectorOutcome(section=section, run=run)), FixedClock(), report_path="/p"
    )
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()


def test_an_exception_becomes_one_failed_run_with_the_fallback_section():
    collector = _Collector(RuntimeError("synthetic crash"))
    outcome = run_guarded(collector, _SteppingClock(100, 142), report_path="/probe")
    assert outcome.run == CollectorRun(collector="probe", status=R.FAILED, duration_ms=42)
    assert outcome.errors == (
        CollectionError(
            collector="probe",
            device_ref=None,
            status=S.COLLECTION_FAILED,
            error_code=UNHANDLED_EXCEPTION,
            message="RuntimeError: synthetic crash",
            report_path="/probe",
        ),
    )
    (detail,) = collector.fallbacks
    assert detail == ErrorDetail(code=UNHANDLED_EXCEPTION, message="RuntimeError: synthetic crash")
    assert outcome.section.status is S.COLLECTION_FAILED and outcome.section.error == detail


def test_adapter_usage_errors_are_contained_and_named():
    outcome = run_guarded(
        _Collector(AdapterUsageError("caller bug")), FixedClock(), report_path="/p"
    )
    assert outcome.run.status is R.FAILED
    assert outcome.errors[0].message == "AdapterUsageError: caller bug"


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        ("a bare section", "not a CollectorOutcome"),
        (_ok(name="someone_else"), "reported a run for 'someone_else'"),
        (
            _ok(
                R.FAILED,
                errors=(
                    CollectionError(
                        collector="probe",
                        device_ref=None,
                        status=S.COLLECTION_FAILED,
                        error_code="X",
                        message="x",
                        report_path=None,
                    ),
                ),
            ),
            "only the guard may",
        ),
    ],
    ids=["bare-section", "foreign-run", "self-reported-failed"],
)
def test_contract_violations_become_guard_failures(result, fragment):
    outcome = run_guarded(_Collector(result), FixedClock(), report_path="/p")
    assert outcome.run.status is R.FAILED and outcome.run.collector == "probe"
    (error,) = outcome.errors
    assert error.collector == "probe" and fragment in error.message


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
def test_non_exception_base_exceptions_propagate(exc):
    with pytest.raises(type(exc)):
        run_guarded(_Collector(exc), FixedClock(), report_path="/p")


def test_describe_uses_the_type_when_there_is_no_message():
    assert describe(ValueError()) == "ValueError"
    assert describe(ValueError("bad")) == "ValueError: bad"


# --- fail-closed environment --------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (P.LINUX, E.UNKNOWN),
        (P.WSL2, E.WSL2),
        (P.WINDOWS, None),
        (P.MACOS, None),
        (P.OTHER, None),
    ],
)
def test_unavailable_classification_fails_closed(profile, expected):
    fallback = SystemInfo.unavailable(
        S.COLLECTION_FAILED,
        ErrorDetail(code=UNHANDLED_EXCEPTION, message="RuntimeError: x"),
        platform_profile=profile,
    )
    assert resolve_linux_environment(fallback) is expected
    assert expected is not E.NATIVE_LINUX


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_a_collected_classification_is_used_as_is(fixture):
    system = fixture.snapshot.system
    assert resolve_linux_environment(system) is system.linux_environment.value


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (FakeHostApi.linux(), E.NATIVE_LINUX),
        (FakeHostApi.wsl2(), E.WSL2),
        (FakeHostApi.windows_host(), None),
        (FakeHostApi.macos(), None),
        (FakeHostApi.other(), None),
    ],
    ids=["native", "wsl2", "windows", "macos", "other"],
)
def test_production_platform_classification_resolves(host, expected):
    assert resolve_linux_environment(OsPlatformAdapter(host).system_info()) is expected


# --- the NVML session step ----------------------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_nvml_step_reproduces_every_fixture_nvml_run(fixture):
    snapshot = fixture.snapshot
    step = NvmlSessionStep(FakeAdapterSet.from_scenario(fixture).nvml, FixedClock(step_ms=15))
    outcome = run_guarded(step, FixedClock(), report_path="/driver/nvml_version")
    (expected,) = (r for r in snapshot.collector_runs if r.collector == "nvml")
    assert outcome.run == expected  # including the fixture's fixed duration
    assert outcome.errors == tuple(e for e in snapshot.collection_errors if e.collector == "nvml")
    assert outcome.section == snapshot.driver.nvml_version
    assert step.session is not None and step.session.is_open
    step.session.close()


def test_a_failing_open_leaves_a_failed_session_stand_in():
    class Exploding:
        def open(self):
            raise OSError("dlopen crashed")

    step = NvmlSessionStep(Exploding(), FixedClock())
    outcome = run_guarded(step, FixedClock(), report_path="/driver/nvml_version")
    assert outcome.run.status is R.FAILED and step.session is None
    session = FailedNvmlSession(outcome.section)
    assert session.device_count() == outcome.section == session.device(0)
    assert session.driver_version() == session.cuda_driver_version() == outcome.section
    assert session.cuda_driver_library() == outcome.section
    assert session.init_result.error.message == "OSError: dlopen crashed"
    with session:
        assert session.is_open
    assert not session.is_open


# --- the production adapter set -----------------------------------------------------------


def test_production_adapter_set_is_live_and_complete():
    adapters = build_production_adapters()
    assert isinstance(adapters, AdapterSet) and adapters.provenance is DataOrigin.LIVE
    assert isinstance(adapters.platform, OsPlatformAdapter)
    assert isinstance(adapters.clock, SystemClock)
    assert isinstance(adapters.pci, DeferredPciScanner)
    assert isinstance(adapters.nvml, PynvmlAdapter)
    assert isinstance(adapters.commands, SubprocessCommandRunner)
    assert isinstance(adapters.nvidia_smi, NvidiaSmiCli)
    assert isinstance(adapters.torch_probe, SubprocessTorchProbe)  # IMP-A07 (DEC-07)
    in_process = build_production_adapters(
        InspectionOptions(pytorch_probe_mode=ProbeMode.IN_PROCESS)
    )
    assert isinstance(in_process.torch_probe, InProcessTorchProbe)  # Python API opt-in (DEC-48)


def test_the_production_set_takes_the_subprocess_timeout_from_the_options():
    adapters = build_production_adapters(InspectionOptions(subprocess_timeout_s=3.5))
    assert adapters.nvidia_smi._timeout_s == 3.5
    assert build_production_adapters().nvidia_smi._timeout_s == 10.0


def test_the_deferred_scanner_refuses_to_scan():
    with pytest.raises(AdapterUsageError, match="classified Linux environment"):
        DeferredPciScanner().scan()


@pytest.mark.parametrize("environment", [*E, None], ids=lambda e: str(e))
def test_production_pci_scanner_keeps_the_fail_closed_gate(environment):
    scanner = production_pci_scanner(environment)
    assert isinstance(scanner, PciSysfsAdapter)
    assert scanner.eligible is (environment is E.NATIVE_LINUX)
    if environment is not E.NATIVE_LINUX:
        assert scanner.scan().observations.status is S.NOT_SUPPORTED  # no sysfs access


def test_stand_ins_report_not_implemented_and_never_raise():
    nvml = UnimplementedNvmlAdapter()
    smi = UnimplementedNvidiaSmi()
    runner = UnimplementedCommandRunner()
    probe = UnimplementedTorchProbe()
    with nvml.open() as session:
        outcomes = [
            nvml.binding_version(),
            nvml.binding_distribution(),
            session.init_result,
            session.driver_version(),
            session.cuda_driver_version(),
            session.cuda_driver_library(),
            session.device_count(),
            session.device(0),
            smi.locate(),
            smi.version().driver_version,
            smi.version().cuda_version,
            smi.query_gpus(),
            runner.which("nvidia-smi"),
            runner.run(("nvidia-smi",), timeout_s=1.0, source=Source.NVIDIA_SMI),
            probe.distribution_version(),
            probe.probe(1.0),
        ]
    assert not session.is_open
    defs = load_report_schema()["$defs"]
    for outcome in outcomes:
        assert outcome.status is S.NOT_SUPPORTED and outcome.error.code == NOT_IMPLEMENTED
        assert (
            outcome.method is None and "isn't implemented in this version" in outcome.error.message
        )
        jsonschema.validate(
            to_json_dict(outcome), {"$ref": "#/$defs/ObservedString", "$defs": defs}
        )


def test_stand_ins_and_a_scanner_cannot_enter_a_fixture_set():
    fakes = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"]).adapter_set()
    with pytest.raises(TypeError, match="provenance"):
        dataclasses.replace(fakes, nvml=UnimplementedNvmlAdapter())
    live = build_production_adapters()
    with pytest.raises(TypeError, match="provenance"):
        dataclasses.replace(
            live,
            pci=PciSysfsAdapter(environment=E.NATIVE_LINUX, reader=FakeSysfsReader()),
        )
