"""IMP-A01 production `PynvmlAdapter` over a stub `pynvml` (injected through the loader):
lifecycle, error mapping, per-device isolation, version fallbacks, the macOS gate and parity
with every fixture. No NVML library is ever loaded."""

import inspect
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pynvml
import pytest
from gsi_testkit.fakes import FakeNvidiaSmi, FakePlatform, FixedClock, StubGpu, StubPynvml
from gsi_testkit.fakes.pynvml_stub import NVML_ERRORS, NVML_MESSAGES
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    NVML_READ_ONLY_FUNCTIONS,
    AdapterUsageError,
    NvmlAdapter,
    NvmlDevice,
    NvmlSession,
    PynvmlAdapter,
)
from gpu_system_inspector.adapters.nvml import import_pynvml
from gpu_system_inspector.collectors import GpuHardwareCollector
from gpu_system_inspector.models import (
    DRIVER_SUPPORTED_CUDA_NOTE,
    CollectionStatus,
    DataOrigin,
    PlatformProfile,
    ProcessKind,
    Source,
)

S = CollectionStatus
E = NVML_ERRORS
GIB = 2**30
FX10 = FIXTURES_BY_ID["FX10"]


class _Platform:
    def __init__(self, profile):
        self.profile = profile


def _adapter(stub, **kwargs):
    kwargs.setdefault("provenance", DataOrigin.LIVE)
    return PynvmlAdapter(stub.loader, **kwargs)


def _session(gpus=None, **stub_kwargs):
    stub = StubPynvml((StubGpu(),) if gpus is None else gpus, **stub_kwargs)
    return stub, _adapter(stub).open()


# --- the stub is faithful to the real binding ---------------------------------------------


def test_stub_error_codes_and_messages_match_the_real_binding():
    for name, value in NVML_ERRORS.items():
        assert getattr(pynvml, name) == value, name
    for value, message in NVML_MESSAGES.items():
        assert str(pynvml.NVMLError(value)) == message
    assert StubPynvml.nvmlMemory_v2 == pynvml.nvmlMemory_v2
    for name in ("NVML_TEMPERATURE_GPU", "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN"):
        assert getattr(StubPynvml, name) == getattr(pynvml, name)
    assert StubPynvml.NVML_DEVICE_MIG_ENABLE == pynvml.NVML_DEVICE_MIG_ENABLE


# What the adapter calls on the binding. KI-16: all present in the declared floor
# (nvidia-ml-py 12.535.108) and in 13.615.71; of the preferred calls, the floor lacks only
# nvmlDeviceGetTemperatureV, which falls back to nvmlDeviceGetTemperature.
BINDING_REQUIRED = (
    "NVMLError",
    "nvmlInit",
    "nvmlShutdown",
    "nvmlSystemGetNVMLVersion",
    "nvmlSystemGetDriverVersion",
    "nvmlSystemGetCudaDriverVersion",
    "nvmlDeviceGetCount",
    "nvmlDeviceGetHandleByIndex",
    "nvmlDeviceGetName",
    "nvmlDeviceGetUUID",
    "nvmlDeviceGetPciInfo",
    "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetCudaComputeCapability",
    "nvmlDeviceGetMigMode",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetTemperatureThreshold",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetEnforcedPowerLimit",
    "nvmlDeviceGetComputeRunningProcesses",
    "nvmlSystemGetProcessName",
)
BINDING_PREFERRED = (
    "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetPciInfo_v3",
    "nvmlMemory_v2",
    "nvmlDeviceGetTemperatureV",
    "nvmlDeviceGetComputeRunningProcesses_v3",
)


def test_the_installed_binding_has_every_call_the_adapter_needs():
    assert [name for name in BINDING_REQUIRED if not hasattr(pynvml, name)] == []
    assert "version" in inspect.signature(pynvml.nvmlDeviceGetMemoryInfo).parameters
    missing = {name for name in BINDING_PREFERRED if not hasattr(pynvml, name)}
    assert missing <= {"nvmlDeviceGetTemperatureV"}  # the floor's only gap (KI-16)
    _, session = _session(missing=missing)
    assert session.device(0).value.temperature().is_collected  # served by the fallback


# --- binding facts (no import of pynvml) --------------------------------------------------


@pytest.mark.parametrize(
    ("distributions", "expected"),
    [
        (["nvidia-ml-py"], "nvidia-ml-py"),
        (["pynvml"], "pynvml"),
        (["pynvml", "nvidia-ml-py"], "nvidia-ml-py"),
    ],
)
def test_binding_distribution_prefers_nvidia_ml_py(distributions, expected):
    stub = StubPynvml()
    adapter = _adapter(
        stub,
        packages_distributions=lambda: {"pynvml": distributions},
        version=lambda d: "13.615.71",
    )
    assert adapter.binding_distribution().value == expected
    version = adapter.binding_version()
    assert version.value == "13.615.71" and version.source is Source.PLATFORM
    assert stub.loads == 0  # facts come from metadata, not from importing the binding


def test_a_missing_binding_distribution_is_not_installed():
    adapter = _adapter(StubPynvml(), packages_distributions=dict)
    for outcome in (adapter.binding_distribution(), adapter.binding_version()):
        assert outcome.status is S.NOT_INSTALLED and outcome.error.code == "ImportError"


def test_a_vanished_distribution_version_is_not_installed():
    def version(name):
        raise PackageNotFoundError(name)

    adapter = _adapter(
        StubPynvml(), packages_distributions=lambda: {"pynvml": ["nvidia-ml-py"]}, version=version
    )
    assert adapter.binding_version().status is S.NOT_INSTALLED


# --- initialisation outcomes --------------------------------------------------------------


def test_a_missing_binding_module_is_not_installed_and_never_shut_down():
    stub = StubPynvml(import_error="No module named 'pynvml'")
    with _adapter(stub).open() as session:
        init = session.init_result
        assert init.status is S.NOT_INSTALLED and init.error.code == "ImportError"
        assert init.method == "import pynvml" and init.error.message == "No module named 'pynvml'"
        assert session.device_count() == init and session.device(0) == init
    assert stub.calls == [] and stub.shutdowns == 0


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("NVML_ERROR_LIBRARY_NOT_FOUND", S.NOT_DETECTED),
        ("NVML_ERROR_DRIVER_NOT_LOADED", S.NOT_DETECTED),
        ("NVML_ERROR_NO_PERMISSION", S.PERMISSION_DENIED),
        ("NVML_ERROR_UNKNOWN", S.COLLECTION_FAILED),
        ("NVML_ERROR_TIMEOUT", S.COLLECTION_FAILED),
    ],
)
def test_init_failures_map_to_their_status_and_skip_shutdown(code, status):
    stub, session = _session(faults={"nvmlInit": E[code]})
    init = session.init_result
    assert (init.status, init.error.code, init.method) == (status, code, "nvmlInit")
    assert init.error.message == NVML_MESSAGES[E[code]]
    for outcome in (
        session.driver_version(),
        session.device_count(),
        session.cuda_driver_version(),
    ):
        assert outcome.error == init.error
    session.close()
    assert stub.inits == 0 and stub.shutdowns == 0


def test_an_unlisted_error_value_is_an_unknown_collection_failure():
    _, session = _session(faults={"nvmlInit": 77})
    assert session.init_result.error.code == "NVML_ERROR_UNKNOWN"
    assert session.init_result.status is S.COLLECTION_FAILED


def test_macos_is_gated_without_loading_the_binding():
    stub = StubPynvml(forbid_load=True)
    with _adapter(stub, platform=_Platform(PlatformProfile.MACOS)).open() as session:
        init = session.init_result
    assert init.status is S.NOT_SUPPORTED and init.error.code == "PLATFORM_NOT_SUPPORTED"
    assert stub.loads == 0


# --- a working session --------------------------------------------------------------------


def test_a_session_initialises_once_reports_versions_and_shuts_down_once():
    stub, session = _session(cuda=11080, cuda_library=12020)
    assert session.init_result.value == "12.550.54.15"
    assert session.init_result.method == "nvmlSystemGetNVMLVersion"
    assert session.driver_version().value == "550.54.15"
    cuda = session.cuda_driver_version()
    assert cuda.value == "11.8" and cuda.note == DRIVER_SUPPORTED_CUDA_NOTE
    assert session.cuda_driver_library().value == "12.2"
    session.close()
    session.close()
    assert stub.inits == 1 and stub.shutdowns == 1 and not session.is_open


def test_zero_devices_is_a_collected_zero():
    _, session = _session(gpus=())
    count = session.device_count()
    assert count.is_collected and count.value == 0 and count.method == "nvmlDeviceGetCount_v2"
    with pytest.raises(AdapterUsageError, match="outside"):
        session.device(0)


def test_device_properties_are_read_and_converted():
    gpu = StubGpu(
        mig=(1, 1),
        processes=((4242, 512 * 2**20, "/usr/bin/python3"), (17, None, "worker")),
    )
    _, session = _session(gpus=(gpu,))
    device = session.device(0).value
    assert device.index == 0
    assert device.name().value == "Synthetic GPU A (fixture)" and device.uuid().value == gpu.uuid
    pci = device.pci_info()
    assert (pci.bus_id.value, pci.device_id.value, pci.subsystem_id.value) == (
        "00000000:01:00.0",
        "0x268410de",
        "0x00000000",
    )
    memory = device.memory_info()
    assert (
        memory.total.value,
        memory.used.value,
        memory.free.value,
        memory.reserved.value,
    ) == gpu.memory
    assert memory.total.unit == "bytes" and memory.total.method == "nvmlDeviceGetMemoryInfo_v2"
    assert device.compute_capability().value == "8.9" and device.mig_mode().value == "ENABLED"
    rates = device.utilization()
    assert (rates.gpu.value, rates.memory_bus.value, rates.gpu.unit) == (35, 20, "percent")
    assert (device.temperature().value, device.temperature().unit) == (55, "celsius")
    assert device.slowdown_threshold().value == 90
    assert (device.power_draw().value, device.power_limit().value) == (150.0, 300.0)
    assert device.power_draw().unit == "watts"
    first, second = device.processes().value
    assert (first.pid, first.kind, first.used_memory_bytes.status) == (
        17,
        ProcessKind.COMPUTE,
        S.NOT_SUPPORTED,
    )
    assert first.used_memory_bytes.error.code == "VALUE_NOT_AVAILABLE"
    assert (second.pid, second.used_memory_bytes.value, second.name.value) == (
        4242,
        512 * 2**20,
        "python3",
    )


def test_multiple_devices_are_independent_and_a_lost_handle_is_isolated():
    gpus = (StubGpu(), StubGpu(name="Synthetic GPU B (fixture)", bus_id="00000000:02:00.0"))
    _, session = _session(
        gpus=gpus, faults={("nvmlDeviceGetHandleByIndex_v2", 0): E["NVML_ERROR_GPU_IS_LOST"]}
    )
    assert session.device_count().value == 2
    lost = session.device(0)
    assert lost.status is S.COLLECTION_FAILED and lost.error.code == "NVML_ERROR_GPU_IS_LOST"
    assert session.device(1).value.name().value == "Synthetic GPU B (fixture)"


def test_one_failing_or_missing_value_leaves_the_rest_intact():
    gpu = StubGpu(temperature=None)
    faults = {
        ("nvmlDeviceGetUtilizationRates", 0): E["NVML_ERROR_NOT_SUPPORTED"],
        ("nvmlDeviceGetPowerUsage", 0): E["NVML_ERROR_NO_PERMISSION"],
        ("nvmlDeviceGetMigMode", 0): E["NVML_ERROR_NOT_SUPPORTED"],
    }
    _, session = _session(gpus=(gpu,), faults=faults)
    device = session.device(0).value
    rates = device.utilization()
    assert rates.gpu.status is rates.memory_bus.status is S.NOT_SUPPORTED
    assert rates.memory_bus.unit == "percent"
    assert device.power_draw().status is S.PERMISSION_DENIED
    assert device.mig_mode().status is S.NOT_SUPPORTED
    assert device.temperature().error.code == "VALUE_NOT_AVAILABLE"
    assert device.name().is_collected and device.memory_info().total.is_collected


def test_a_gpu_lost_mid_device_fails_only_that_read():
    _, session = _session(faults={("nvmlDeviceGetMemoryInfo_v2", 0): E["NVML_ERROR_GPU_IS_LOST"]})
    device = session.device(0).value
    memory = device.memory_info()
    assert all(v.error.code == "NVML_ERROR_GPU_IS_LOST" for v in (memory.total, memory.free))
    assert memory.total.unit == "bytes" and device.uuid().is_collected


def test_an_enumeration_failure_is_an_outcome_and_the_session_still_closes():
    stub, session = _session(faults={"nvmlDeviceGetCount_v2": E["NVML_ERROR_UNKNOWN"]})
    count = session.device_count()
    assert count.status is S.COLLECTION_FAILED and count.error.code == "NVML_ERROR_UNKNOWN"
    session.close()
    assert stub.shutdowns == 1


@pytest.mark.parametrize(
    ("gpu", "read", "fragment"),
    [
        (StubGpu(pci_device_id=-1), lambda d: d.pci_info().device_id, "32-bit"),
        (StubGpu(name=""), lambda d: d.name(), "expected text"),
        (StubGpu(utilization=(250, 1)), lambda d: d.utilization().gpu, "percentage"),
        (StubGpu(power_mw=-5), lambda d: d.power_draw(), "milliwatts"),
    ],
    ids=["pci-id", "empty-name", "percent", "power"],
)
def test_unusable_values_are_parse_errors_not_values(gpu, read, fragment):
    _, session = _session(gpus=(gpu,))
    outcome = read(session.device(0).value)
    assert outcome.status is S.COLLECTION_FAILED and outcome.error.code == "PARSE_ERROR"
    assert fragment in outcome.error.message


def test_an_unusable_cuda_version_is_a_parse_error():
    _, session = _session(cuda=0)
    assert session.cuda_driver_version().error.code == "PARSE_ERROR"


def test_a_missing_struct_value_is_not_available_and_a_bad_one_a_parse_error():
    gpu = StubGpu(memory=(-1, 4 * GIB, 20 * GIB, None), utilization=(None, 3))
    _, session = _session(gpus=(gpu,))
    device = session.device(0).value
    memory = device.memory_info()
    assert (
        memory.reserved.status is S.NOT_SUPPORTED
        and memory.reserved.error.code == "VALUE_NOT_AVAILABLE"
    )
    assert memory.total.error.code == "PARSE_ERROR" and memory.free.is_collected
    rates = device.utilization()
    assert rates.gpu.error.code == "VALUE_NOT_AVAILABLE" and rates.memory_bus.value == 3


@pytest.mark.parametrize("count", [-1, "2", True])
def test_an_unusable_device_count_is_a_parse_error(count):
    stub, session = _session()
    stub.nvmlDeviceGetCount = lambda: count
    assert session.device_count().error.code == "PARSE_ERROR"


def test_a_failed_struct_read_fails_each_of_its_fields():
    faults = {
        ("nvmlDeviceGetPciInfo_v3", 0): E["NVML_ERROR_NO_PERMISSION"],
        ("nvmlDeviceGetComputeRunningProcesses_v3", 0): E["NVML_ERROR_NOT_SUPPORTED"],
    }
    _, session = _session(faults=faults)
    device = session.device(0).value
    pci = device.pci_info()
    assert {pci.bus_id.status, pci.device_id.status, pci.subsystem_id.status} == {
        S.PERMISSION_DENIED
    }
    assert device.processes().status is S.NOT_SUPPORTED


def test_when_every_variant_is_too_new_the_last_outcome_is_reported():
    faults = {
        ("nvmlDeviceGetTemperatureV", 0): E["NVML_ERROR_FUNCTION_NOT_FOUND"],
        ("nvmlDeviceGetTemperature", 0): E["NVML_ERROR_FUNCTION_NOT_FOUND"],
    }
    _, session = _session(faults=faults)
    temperature = session.device(0).value.temperature()
    assert (
        temperature.status is S.NOT_SUPPORTED and temperature.method == "nvmlDeviceGetTemperature"
    )


def test_a_struct_without_an_expected_field_is_a_parse_error():
    stub, session = _session()
    stub.nvmlDeviceGetUtilizationRates = lambda handle: SimpleNamespace(gpu=5)
    rates = session.device(0).value.utilization()
    assert rates.gpu.value == 5 and rates.memory_bus.error.code == "PARSE_ERROR"


def test_the_default_loader_imports_the_real_binding_without_initialising_it():
    assert import_pynvml() is pynvml  # importing loads no library; only nvmlInit() does


def test_binding_facts_default_to_this_environments_metadata():
    """No injection: real `importlib.metadata` (the required binding is installed here)."""
    adapter = PynvmlAdapter(StubPynvml().loader)
    assert adapter.binding_distribution().value == "nvidia-ml-py"
    assert adapter.binding_version().is_collected


# --- older drivers and bindings -----------------------------------------------------------


def test_older_drivers_fall_back_to_the_previous_calls():
    faults = {
        ("nvmlDeviceGetMemoryInfo_v2", 0): E["NVML_ERROR_FUNCTION_NOT_FOUND"],
        ("nvmlDeviceGetTemperatureV", 0): E["NVML_ERROR_FUNCTION_NOT_FOUND"],
        ("nvmlDeviceGetComputeRunningProcesses_v3", 0): E["NVML_ERROR_FUNCTION_NOT_FOUND"],
    }
    stub, session = _session(faults=faults)
    device = session.device(0).value
    memory = device.memory_info()
    assert memory.total.method == "nvmlDeviceGetMemoryInfo" and memory.total.is_collected
    assert memory.reserved.status is S.NOT_SUPPORTED
    assert device.temperature().method == "nvmlDeviceGetTemperature"
    assert device.processes().method == "nvmlDeviceGetComputeRunningProcesses"
    assert {"nvmlDeviceGetMemoryInfo", "nvmlDeviceGetTemperature"} <= set(stub.calls)


def test_an_older_binding_without_the_newer_calls_uses_the_older_ones():
    stub, session = _session(
        missing={"nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetTemperatureV", "nvmlMemory_v2"}
    )
    device = session.device(0).value
    assert device.pci_info().bus_id.method == "nvmlDeviceGetPciInfo"
    assert device.pci_info().bus_id.value == "00000000:01:00.0"
    assert device.temperature().method == "nvmlDeviceGetTemperature"
    assert device.memory_info().total.method == "nvmlDeviceGetMemoryInfo"


def test_a_failure_that_is_not_about_versions_does_not_fall_back():
    stub, session = _session(
        faults={("nvmlDeviceGetMemoryInfo_v2", 0): E["NVML_ERROR_NO_PERMISSION"]}
    )
    assert session.device(0).value.memory_info().total.status is S.PERMISSION_DENIED
    assert "nvmlDeviceGetMemoryInfo" not in stub.calls


# --- lifecycle ----------------------------------------------------------------------------


def test_an_exception_closes_the_session_and_propagates():
    stub = StubPynvml([StubGpu()])
    with pytest.raises(RuntimeError, match="collector bug"), _adapter(stub).open() as session:
        session.device_count()
        raise RuntimeError("collector bug")
    assert not session.is_open and stub.shutdowns == 1


def test_a_shutdown_failure_is_swallowed_and_kept_as_evidence():
    stub, session = _session(shutdown_fault=E["NVML_ERROR_UNINITIALIZED"])
    session.close()
    assert stub.shutdowns == 1 and session.shutdown_errors == ["NVML_ERROR_UNINITIALIZED"]


def test_one_session_at_a_time_and_sequential_sessions_balance():
    stub = StubPynvml([StubGpu()])
    adapter = _adapter(stub)
    with adapter.open():
        with pytest.raises(AdapterUsageError, match="already open"):
            adapter.open()
    with adapter.open():
        pass
    assert stub.inits == stub.shutdowns == 2


def test_queries_after_close_are_usage_errors():
    _, session = _session()
    device = session.device(0).value
    session.close()
    for call in (session.driver_version, session.device_count, device.name, device.memory_info):
        with pytest.raises(AdapterUsageError, match="closed"):
            call()


@pytest.mark.parametrize("index", [-1, True, "0"])
def test_bad_device_indices_are_usage_errors(index):
    _, session = _session()
    with pytest.raises(AdapterUsageError):
        session.device(index)


def test_only_read_only_nvml_functions_are_called():
    gpu = StubGpu(processes=((1, 1, "a"),))
    stub, session = _session(gpus=(gpu,))
    device = session.device(0).value
    for read in (
        device.name,
        device.uuid,
        device.pci_info,
        device.memory_info,
        device.compute_capability,
        device.mig_mode,
        device.utilization,
        device.temperature,
        device.slowdown_threshold,
        device.power_draw,
        device.power_limit,
        device.processes,
    ):
        read()
    session.driver_version(), session.cuda_driver_version(), session.cuda_driver_library()
    session.close()
    assert set(stub.calls) <= NVML_READ_ONLY_FUNCTIONS
    assert stub.calls.count("nvmlInit") == stub.calls.count("nvmlShutdown") == 1


def test_the_production_adapter_conforms_to_the_protocols():
    adapter = _adapter(StubPynvml([StubGpu()]))
    session = adapter.open()
    assert isinstance(adapter, NvmlAdapter) and isinstance(session, NvmlSession)
    assert isinstance(session.device(0).value, NvmlDevice)
    session.close()


def test_fixture_provenance_prefixes_every_method():
    _, session = _session()
    session.close()
    adapter = _adapter(StubPynvml([StubGpu()]), provenance=DataOrigin.FIXTURE)
    with adapter.open() as session:
        assert session.init_result.method == "fixture:nvmlSystemGetNVMLVersion"
        assert session.device(0).value.name().method == "fixture:nvmlDeviceGetName"


# --- parity with every fixture ------------------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_the_production_adapter_reproduces_every_fixtures_nvml_facts(fixture):
    snapshot = fixture.snapshot
    stub = StubPynvml.from_snapshot(snapshot, lost_after_handles=fixture.nvml_lost_after_handles)
    adapter = PynvmlAdapter(
        stub.loader, platform=FakePlatform.from_snapshot(snapshot), provenance=DataOrigin.FIXTURE
    )
    with adapter.open() as session:
        assert session.init_result == snapshot.driver.nvml_version
        if snapshot.driver.nvml_version.is_collected:
            assert session.driver_version() == snapshot.driver.driver_version
            assert session.cuda_driver_version() == snapshot.driver.driver_supported_cuda_version
            assert session.cuda_driver_library() == snapshot.driver.cuda_driver_library
        outcome = GpuHardwareCollector(
            session,
            FakeNvidiaSmi.from_snapshot(snapshot),
            FixedClock(step_ms=25),
            pci=snapshot.gpu.pci_observations,
            environment=snapshot.system.linux_environment.value,
            profile=snapshot.system.platform_profile,
        ).collect()
    assert outcome.section == snapshot.gpu  # FX10 too: nvml:1 is lost after enumeration
    (run,) = (r for r in snapshot.collector_runs if r.collector == "gpu_hardware")
    assert outcome.run == run
    assert stub.inits == stub.shutdowns <= 1
    assert set(stub.calls) <= NVML_READ_ONLY_FUNCTIONS
