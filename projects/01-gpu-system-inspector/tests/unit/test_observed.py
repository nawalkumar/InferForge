import dataclasses
import math

import pytest

from gpu_system_inspector.models import (
    UNITS,
    CollectionStatus,
    ErrorDetail,
    InvalidModelError,
    Observed,
    Source,
)

FAILURE_STATUSES = [
    CollectionStatus.NOT_INSTALLED,
    CollectionStatus.NOT_DETECTED,
    CollectionStatus.NOT_SUPPORTED,
    CollectionStatus.PERMISSION_DENIED,
    CollectionStatus.COLLECTION_FAILED,
    CollectionStatus.UNKNOWN,
]


def test_successful_measurement_keeps_value_and_provenance():
    obs = Observed.collected(
        25757220864, source=Source.NVML, method="nvmlDeviceGetMemoryInfo_v2", unit="bytes"
    )
    assert obs.value == 25757220864
    assert obs.status is CollectionStatus.COLLECTED
    assert (obs.source, obs.method, obs.unit, obs.error) == (
        Source.NVML,
        "nvmlDeviceGetMemoryInfo_v2",
        "bytes",
        None,
    )
    assert obs.is_collected


def test_a_real_zero_is_a_valid_collected_value():
    assert Observed.collected(0, source=Source.NVML, method="x", unit="percent").value == 0


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_each_failure_status_is_preserved_with_its_error(status):
    obs = Observed.failed(
        status, code=f"CODE_{status}", message="why", source=Source.NVML, method="api"
    )
    assert obs.status is status
    assert obs.value is None
    assert obs.error == ErrorDetail(code=f"CODE_{status}", message="why")
    assert not obs.is_collected


@pytest.mark.parametrize("status", FAILURE_STATUSES)
@pytest.mark.parametrize("placeholder", [0, -1, 0.0, "", "N/A", False])
def test_missing_values_can_never_be_zero_or_a_placeholder(status, placeholder):
    with pytest.raises(InvalidModelError, match="missing data is never a value"):
        Observed(
            value=placeholder,
            status=status,
            source=Source.NVML,
            method="api",
            error=ErrorDetail(code="X", message="y"),
        )


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_failure_statuses_require_an_error(status):
    with pytest.raises(InvalidModelError, match="requires an error"):
        Observed(value=None, status=status, source=Source.NVML, method="api")


def test_missing_dependency_is_not_installed_not_a_generic_failure():
    obs = Observed.failed(
        CollectionStatus.NOT_INSTALLED,
        code="ImportError",
        message="No module named 'pynvml'",
        source=Source.NVML,
        method="import pynvml",
    )
    assert obs.status is CollectionStatus.NOT_INSTALLED
    assert obs.status is not CollectionStatus.COLLECTION_FAILED


def test_skipped_needs_no_source_and_carries_no_error():
    obs = Observed.skipped(note="--no-processes")
    assert (obs.status, obs.value, obs.source, obs.error) == (
        CollectionStatus.SKIPPED,
        None,
        None,
        None,
    )
    with pytest.raises(InvalidModelError, match="SKIPPED"):
        Observed(
            value=None,
            status=CollectionStatus.SKIPPED,
            source=None,
            method=None,
            error=ErrorDetail(code="X", message="y"),
        )


def test_not_applicable_must_explain_why():
    obs = Observed.not_applicable(note="CPU-only build", source=Source.PYTORCH)
    assert obs.status is CollectionStatus.NOT_APPLICABLE
    assert obs.value is None and obs.error is None
    with pytest.raises(InvalidModelError, match="explain why"):
        Observed(value=None, status=CollectionStatus.NOT_APPLICABLE, source=None, method=None)


def test_unknown_state_is_distinct_and_requires_an_error():
    obs = Observed.failed(CollectionStatus.UNKNOWN, code="WeirdError", message="unclassified")
    assert obs.status is CollectionStatus.UNKNOWN


def test_collected_values_must_name_source_and_method():
    with pytest.raises(InvalidModelError, match="source and method"):
        Observed(value=1, status=CollectionStatus.COLLECTED, source=None, method="x")
    with pytest.raises(InvalidModelError, match="source and method"):
        Observed(value=1, status=CollectionStatus.COLLECTED, source=Source.NVML, method="")


def test_collected_values_reject_none_and_errors():
    with pytest.raises(InvalidModelError, match="must not be None"):
        Observed(value=None, status=CollectionStatus.COLLECTED, source=Source.NVML, method="x")
    with pytest.raises(InvalidModelError, match="must not carry an error"):
        Observed(
            value=1,
            status=CollectionStatus.COLLECTED,
            source=Source.NVML,
            method="x",
            error=ErrorDetail(code="X", message="y"),
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(bad):
    with pytest.raises(InvalidModelError, match="finite"):
        Observed.collected(bad, source=Source.NVML, method="x", unit="watts")


def test_unknown_units_and_string_statuses_are_rejected():
    with pytest.raises(InvalidModelError, match="unit"):
        Observed.collected(1, source=Source.NVML, method="x", unit="MiB")
    with pytest.raises(InvalidModelError, match="CollectionStatus"):
        Observed(value=1, status="COLLECTED", source=Source.NVML, method="x")
    assert UNITS == ("bytes", "celsius", "watts", "percent", "ms")


def test_error_detail_requires_a_code():
    with pytest.raises(InvalidModelError, match="code"):
        ErrorDetail(code="", message="m")


def test_observed_is_immutable_and_lists_become_tuples():
    obs = Observed.collected([1, 2], source=Source.NVML, method="x")
    assert obs.value == (1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.value = (3,)  # type: ignore[misc]
