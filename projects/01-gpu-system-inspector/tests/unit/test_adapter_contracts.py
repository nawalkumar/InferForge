"""Fakes conform to the adapter protocols: same members and signatures, and the documented
behaviour on every fixture (architecture §4.5)."""

import dataclasses
import inspect
from datetime import timedelta

import pytest
from gsi_testkit.builders import FIXTURE_METHOD_PREFIX
from gsi_testkit.fakes import (
    FailingToolkitLocator,
    FakeAdapterSet,
    FakeCommandRunner,
    FakeHostApi,
    FakeSysfsReader,
    FakeToolkitFiles,
    FixedClock,
    StubGpu,
    StubPynvml,
)
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    NVIDIA_SMI_QUERY_FIELDS,
    AdapterSet,
    Clock,
    CommandRunner,
    CudaToolkitLocator,
    HostApi,
    NvidiaSmiAdapter,
    NvidiaSmiCli,
    NvmlAdapter,
    NvmlDevice,
    NvmlMemoryInfo,
    NvmlPciInfo,
    NvmlSession,
    NvmlUtilization,
    OsPlatformAdapter,
    PciScan,
    PciScanner,
    PciSysfsAdapter,
    PlatformAdapter,
    PynvmlAdapter,
    SmiVersion,
    StdlibHostApi,
    StdlibSysfsReader,
    StdlibToolkitFiles,
    SubprocessCommandRunner,
    SysfsReader,
    SystemClock,
    ToolkitFiles,
    ToolkitLocator,
    TorchProbe,
    UnimplementedCommandRunner,
    UnimplementedNvidiaSmi,
    UnimplementedNvmlAdapter,
    UnimplementedNvmlSession,
    UnimplementedTorchProbe,
)
from gpu_system_inspector.core import DeferredPciScanner, FailedNvmlSession
from gpu_system_inspector.models import (
    CollectionStatus,
    DataOrigin,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    Source,
    SystemInfo,
)

ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
FX01 = FIXTURES_BY_ID["FX01"]


def protocol_members(protocol: type) -> list[str]:
    return [
        name
        for name in vars(protocol)
        if not name.startswith("_") or name in ("__enter__", "__exit__")
        if callable(vars(protocol)[name]) or isinstance(vars(protocol)[name], property)
    ]


def _parameters(function) -> list[tuple[str, inspect._ParameterKind, bool]]:
    params = list(inspect.signature(function).parameters.values())[1:]  # drop self
    return [(p.name, p.kind, p.default is not p.empty) for p in params]


def signature_mismatches(protocol: type, implementation: object) -> list[str]:
    problems = []
    for name in protocol_members(protocol):
        expected = vars(protocol)[name]
        try:
            actual = inspect.getattr_static(implementation, name)
        except AttributeError:
            problems.append(f"{name}: missing")
            continue
        if isinstance(expected, property):
            if not isinstance(actual, property):
                problems.append(f"{name}: must be a property")
        elif _parameters(expected) != _parameters(actual):
            problems.append(f"{name}: {_parameters(actual)} != {_parameters(expected)}")
    return problems


def _implementations():
    adapters = FakeAdapterSet.from_scenario(FX01)
    session = adapters.nvml.open()
    yield NvmlAdapter, adapters.nvml
    yield NvmlSession, session
    yield NvmlDevice, session.device(0).value
    yield NvidiaSmiAdapter, adapters.nvidia_smi
    yield PciScanner, adapters.pci
    yield TorchProbe, adapters.torch_probe
    yield PlatformAdapter, adapters.platform
    yield CommandRunner, adapters.commands
    yield Clock, adapters.clock
    yield Clock, SystemClock()
    yield PlatformAdapter, OsPlatformAdapter(FakeHostApi.linux())
    yield HostApi, FakeHostApi.linux()
    yield HostApi, StdlibHostApi()
    yield (
        PciScanner,
        PciSysfsAdapter(environment=LinuxEnvironment.NATIVE_LINUX, reader=FakeSysfsReader()),
    )
    yield SysfsReader, FakeSysfsReader()
    yield SysfsReader, StdlibSysfsReader()
    # Production stand-ins and the core's session/scanner placeholders (DEC-44).
    yield NvmlAdapter, UnimplementedNvmlAdapter()
    yield NvmlSession, UnimplementedNvmlSession()
    yield NvidiaSmiAdapter, UnimplementedNvidiaSmi()
    yield CommandRunner, UnimplementedCommandRunner()
    yield TorchProbe, UnimplementedTorchProbe()
    yield NvmlSession, FailedNvmlSession(session.init_result)
    yield PciScanner, DeferredPciScanner()
    session.close()
    # Production NVIDIA adapters (IMP-A01, A02, A05) over a stub binding / fake runner.
    nvml = PynvmlAdapter(StubPynvml([StubGpu()]).loader)
    live = nvml.open()
    yield NvmlAdapter, nvml
    yield NvmlSession, live
    yield NvmlDevice, live.device(0).value
    yield CommandRunner, SubprocessCommandRunner()
    yield NvidiaSmiAdapter, NvidiaSmiCli(FakeCommandRunner())
    live.close()
    # CUDA Toolkit locator (IMP-A06) over the fake tree and runner, and its file seam.
    yield ToolkitLocator, adapters.toolkit
    yield ToolkitLocator, CudaToolkitLocator(SubprocessCommandRunner(), OsPlatformAdapter())
    yield ToolkitLocator, FailingToolkitLocator(RuntimeError("x"))
    yield ToolkitFiles, FakeToolkitFiles()
    yield ToolkitFiles, StdlibToolkitFiles()


IMPLEMENTATIONS = list(_implementations())


@pytest.mark.parametrize(
    ("protocol", "implementation"),
    IMPLEMENTATIONS,
    ids=[f"{p.__name__}-{type(i).__name__}" for p, i in IMPLEMENTATIONS],
)
def test_implementation_matches_the_protocol(protocol, implementation):
    assert isinstance(implementation, protocol)
    assert signature_mismatches(protocol, type(implementation)) == []


def test_signature_check_detects_drift():
    class Drifted:
        def which(self, name, extra_paths=()):  # keyword-only marker missing
            ...

        def run(self, command, *, timeout_s, source):  # renamed parameter
            ...

    problems = signature_mismatches(CommandRunner, Drifted)
    assert [p.split(":")[0] for p in problems] == ["which", "run"]
    assert signature_mismatches(PciScanner, Drifted) == ["scan: missing"]


def test_every_protocol_has_a_conforming_fake():
    covered = {protocol for protocol, _ in IMPLEMENTATIONS}
    assert covered == {
        NvmlAdapter,
        NvmlSession,
        NvmlDevice,
        NvidiaSmiAdapter,
        PciScanner,
        TorchProbe,
        PlatformAdapter,
        CommandRunner,
        Clock,
        HostApi,
        SysfsReader,
        ToolkitLocator,
        ToolkitFiles,
    }


# --- behaviour on every fixture -----------------------------------------------------------


def _observed_parts(value) -> list[Observed]:
    if isinstance(value, Observed):
        return [value]
    if isinstance(value, NvmlPciInfo | NvmlMemoryInfo | NvmlUtilization | SmiVersion):
        return [getattr(value, f.name) for f in dataclasses.fields(value)]
    if isinstance(value, PciScan):
        return [value.observations]
    raise AssertionError(f"unexpected adapter output {value!r}")


def _nvml_outputs(adapters) -> list:
    nvml = adapters.nvml
    outputs = [nvml.binding_version(), nvml.binding_distribution()]
    with nvml.open() as session:
        outputs += [
            session.init_result,
            session.driver_version(),
            session.cuda_driver_version(),
            session.cuda_driver_library(),
        ]
        count = session.device_count()
        outputs.append(count)
        for index in range(count.value if count.is_collected else 1):
            handle = session.device(index)
            outputs.append(handle)
            if handle.is_collected:
                device = handle.value
                outputs += [getattr(device, name)() for name in _DEVICE_QUERIES]
    return outputs


_DEVICE_QUERIES = [m for m in protocol_members(NvmlDevice) if m != "index"]


@ids
def test_adapters_report_conditions_as_outcomes_never_exceptions(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    outputs = _nvml_outputs(adapters)
    outputs += [
        adapters.nvidia_smi.locate(),
        adapters.nvidia_smi.version(),
        adapters.nvidia_smi.query_gpus(),
        adapters.pci.scan(),
        adapters.torch_probe.distribution_version(),
        adapters.commands.which("nvidia-smi"),
    ]
    outputs += [
        adapters.platform.getenv(name) for name in fixture.snapshot.cuda_environment.environment
    ]
    pytorch = fixture.snapshot.frameworks.pytorch
    if pytorch.status not in (CollectionStatus.NOT_INSTALLED, CollectionStatus.SKIPPED):
        outputs.append(adapters.torch_probe.probe(60.0))
    for output in outputs:
        for observed in _observed_parts(output):
            method = observed.method
            assert method is None or method.startswith(FIXTURE_METHOD_PREFIX), observed
    assert isinstance(adapters.platform.system_info(), SystemInfo)


@ids
def test_nvml_init_failure_answers_every_query_with_that_failure(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    with adapters.nvml.open() as session:
        if session.init_result.is_collected:
            return
        for outcome in (session.driver_version(), session.device_count(), session.device(0)):
            assert outcome.status is session.init_result.status
            assert outcome.error == session.init_result.error


@ids
def test_nvidia_smi_failure_propagates_and_rows_are_complete(fixture):
    smi = FakeAdapterSet.from_scenario(fixture).nvidia_smi
    located, version, rows = smi.locate(), smi.version(), smi.query_gpus()
    if not located.is_collected:
        assert rows.status is located.status
        assert not version.driver_version.is_collected and not version.cuda_version.is_collected
    if rows.is_collected:
        for row in rows.value:
            assert tuple(row.values) == NVIDIA_SMI_QUERY_FIELDS[1:]
            assert all(v.source is Source.NVIDIA_SMI for v in row.values.values())


@ids
def test_pci_scan_is_gated_to_native_linux(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    scan = adapters.pci.scan()
    assert isinstance(scan, PciScan) and scan.failures == ()
    environment = adapters.platform.system_info().linux_environment.value
    if environment is not LinuxEnvironment.NATIVE_LINUX:
        assert scan.observations.status is CollectionStatus.NOT_SUPPORTED
        assert scan.observations.error.code == "PLATFORM_NOT_SUPPORTED"
    if adapters.platform.profile is not PlatformProfile.LINUX:
        assert environment is not LinuxEnvironment.NATIVE_LINUX


@ids
def test_platform_profile_and_environment_contract(fixture):
    platform = FakeAdapterSet.from_scenario(fixture).platform
    assert platform.profile is platform.system_info().platform_profile
    for name in fixture.snapshot.cuda_environment.environment:
        value = platform.getenv(name)
        assert value.source is Source.ENVIRONMENT
        assert value.is_collected or value.status is CollectionStatus.NOT_DETECTED


@ids
def test_torch_probe_agrees_with_the_fixture(fixture):
    probe = FakeAdapterSet.from_scenario(fixture).torch_fake
    info = fixture.snapshot.frameworks.pytorch
    assert probe.distribution_version() == info.distribution_version
    if info.status in (CollectionStatus.NOT_INSTALLED, CollectionStatus.SKIPPED):
        with pytest.raises(AssertionError, match="must not run"):
            probe.probe(60.0)
        return
    report = probe.probe(60.0)
    if info.status is CollectionStatus.COLLECTION_FAILED:
        assert report.status is CollectionStatus.COLLECTION_FAILED
        assert report.error.code == info.version.error.code
        return
    assert report.value.torch_version == info.version.value
    assert report.value.cuda_available == info.cuda_available.value
    assert [(d.index, d.uuid) for d in report.value.devices] == [
        (d.torch_index, d.uuid.value) for d in info.devices
    ]


@ids
def test_command_runner_and_nvidia_smi_agree_on_availability(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    located = adapters.nvidia_smi.locate()
    found = adapters.commands.which("nvidia-smi")
    assert found.is_collected == located.is_collected
    assert found.value == located.value


@pytest.mark.parametrize("clock", [FixedClock(), SystemClock()], ids=["fixed", "system"])
def test_clock_contract(clock):
    now = clock.now_utc()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
    ticks = [clock.monotonic_ms() for _ in range(50)]
    assert ticks == sorted(ticks) and all(isinstance(t, int) for t in ticks)


# --- adapter set --------------------------------------------------------------------------


@ids
def test_fake_adapter_set_builds_a_valid_adapter_set(fixture):
    adapter_set = FakeAdapterSet.from_scenario(fixture).adapter_set()
    assert isinstance(adapter_set, AdapterSet)
    assert adapter_set.provenance is DataOrigin.FIXTURE


def test_adapter_set_rejects_non_conforming_adapters():
    valid = FakeAdapterSet.from_scenario(FX01).adapter_set()
    with pytest.raises(TypeError, match="doesn't implement PciScanner"):
        dataclasses.replace(valid, pci=object())
    with pytest.raises(TypeError, match="provenance"):
        dataclasses.replace(valid, provenance="FIXTURE")
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid.clock = SystemClock()  # type: ignore[misc]


def test_adapter_set_requires_every_adapter_explicitly():
    fields = dataclasses.fields(AdapterSet)
    assert [f.name for f in fields if f.default is not dataclasses.MISSING] == []
    adapters = FakeAdapterSet.from_scenario(FX01)
    given = {f.name: getattr(adapters, f.name) for f in fields if f.name != "torch_probe"}
    with pytest.raises(TypeError, match="torch_probe"):
        AdapterSet(**given)
