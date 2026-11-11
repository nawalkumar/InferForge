"""NVML session lifecycle contract (architecture §4.5.3, data flow §4), checked on `FakeNvml`.

No real NVML session is opened: the fake implements the same contract a production adapter
must, and these tests pin that contract down.
"""

import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeNvml
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import AdapterUsageError
from gpu_system_inspector.models import CollectionStatus

HANDLE = "nvmlDeviceGetHandleByIndex_v2"


def _nvml(fixture_id: str) -> FakeNvml:
    return FakeAdapterSet.from_scenario(FIXTURES_BY_ID[fixture_id]).nvml


def test_successful_session_initialises_once_and_shuts_down_once():
    nvml = _nvml("FX01")
    session = nvml.open()
    assert session.is_open and session.init_result.is_collected
    assert nvml.calls == ["nvmlInit"] and nvml.inits == 1
    session.close()
    session.close()
    assert not session.is_open
    assert nvml.calls.count("nvmlShutdown") == 1 and nvml.shutdowns == 1


def test_session_is_closed_after_an_exception_and_the_exception_propagates():
    nvml = _nvml("FX01")
    with pytest.raises(RuntimeError, match="collector bug"), nvml.open() as session:
        session.device_count()
        raise RuntimeError("collector bug")
    assert not session.is_open and nvml.shutdowns == 1


def test_failed_initialisation_returns_a_session_and_never_shuts_down():
    nvml = _nvml("FX03")
    with nvml.open() as session:
        assert session.init_result.status is CollectionStatus.NOT_DETECTED
        for outcome in (session.driver_version(), session.device_count(), session.device(0)):
            assert outcome.status is CollectionStatus.NOT_DETECTED
            assert outcome.error == session.init_result.error
    assert nvml.inits == nvml.shutdowns == 0
    assert "nvmlShutdown" not in nvml.calls


def test_all_queries_share_one_session():
    nvml = _nvml("FX06")
    with nvml.open() as session:
        for index in range(session.device_count().value):
            device = session.device(index).value
            device.memory_info()
            device.utilization()
            device.processes()
        session.driver_version()
    assert nvml.calls.count("nvmlInit") == 1 and nvml.calls.count("nvmlShutdown") == 1


def test_a_second_concurrent_session_is_refused_but_sequential_sessions_balance():
    nvml = _nvml("FX01")
    with nvml.open():
        with pytest.raises(AdapterUsageError, match="already open"):
            nvml.open()
    with nvml.open():
        pass
    assert nvml.inits == nvml.shutdowns == 2


def test_queries_after_close_are_usage_errors():
    nvml = _nvml("FX01")
    with nvml.open() as session:
        device = session.device(0).value
    with pytest.raises(AdapterUsageError, match="closed"):
        session.driver_version()
    with pytest.raises(AdapterUsageError, match="closed"):
        device.memory_info()


def test_shutdown_errors_are_swallowed():
    nvml = _nvml("FX01")
    nvml.shutdown_fault = "NVML_ERROR_UNINITIALIZED"
    with nvml.open():
        pass
    assert nvml.shutdowns == 1 and nvml.shutdown_errors == ["NVML_ERROR_UNINITIALIZED"]


def test_out_of_range_device_index_is_a_usage_error():
    with _nvml("FX01").open() as session, pytest.raises(AdapterUsageError, match="outside"):
        session.device(1)


def test_handle_failure_is_an_outcome_and_isolated_to_its_device():
    nvml = _nvml("FX06")
    nvml.faults = {(0, HANDLE): "NVML_ERROR_NO_PERMISSION"}
    with nvml.open() as session:
        denied = session.device(0)
        assert denied.status is CollectionStatus.PERMISSION_DENIED
        assert denied.error.code == "NVML_ERROR_NO_PERMISSION" and denied.method.endswith(HANDLE)
        assert session.device(1).value.name().is_collected


def test_query_faults_map_to_status_and_keep_the_field_unit():
    nvml = _nvml("FX01")
    nvml.faults = {(0, "nvmlDeviceGetUtilizationRates"): "NVML_ERROR_NOT_SUPPORTED"}
    with nvml.open() as session:
        utilization = session.device(0).value.utilization()
    for value in (utilization.gpu, utilization.memory_bus):
        assert value.status is CollectionStatus.NOT_SUPPORTED and value.unit == "percent"
        assert value.error.message == "Not Supported"
