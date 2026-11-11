import copy
import json
from datetime import UTC, datetime

import pytest

from gpu_system_inspector.models import (
    CollectionStatus,
    InspectionReport,
    Observed,
    RuntimeMetrics,
)
from gpu_system_inspector.reporting import (
    SerializationError,
    format_timestamp,
    from_json_dict,
    parse_report_json,
    parse_timestamp,
    render_json,
    to_json_dict,
)

TOP_LEVEL_KEYS = [
    "schema_name",
    "schema_version",
    "application",
    "generated_at",
    "duration_ms",
    "data_origin",
    "mode",
    "options",
    "system",
    "gpu",
    "driver",
    "cuda_environment",
    "frameworks",
    "runtime",
    "diagnostics",
    "readiness",
    "collection_errors",
    "unsupported_capabilities",
    "limitations",
    "collector_runs",
    "redaction",
]
OBSERVED_KEYS = ["value", "status", "source", "method", "unit", "error", "note"]


def _all_observed(node):
    """Yield every Observed-shaped object in a JSON tree."""
    if isinstance(node, dict):
        if list(node) == OBSERVED_KEYS:
            yield node
        for value in node.values():
            yield from _all_observed(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_observed(item)


def test_render_produces_valid_deterministic_json(build):
    text = render_json(build.report())
    assert text.endswith("}\n")
    assert json.loads(text)["schema_version"] == "1.0"
    assert render_json(build.report()) == text


def test_top_level_keys_are_stable_and_ordered(build):
    assert list(to_json_dict(build.report())) == TOP_LEVEL_KEYS


def test_every_observed_has_all_keys_and_the_value_invariant(build):
    for report in (build.report(), build.no_gpu_report(), build.partial_failure_report()):
        observed = list(_all_observed(to_json_dict(report)))
        assert observed
        for item in observed:
            assert (item["value"] is not None) == (item["status"] == "COLLECTED")


def test_unavailable_values_are_null_never_zero(build):
    data = to_json_dict(build.partial_failure_report())
    lost = data["runtime"][1]
    for key in ("memory_free_bytes", "gpu_utilization_percent", "power_draw_watts"):
        assert lost[key]["value"] is None
        assert lost[key]["status"] == "COLLECTION_FAILED"
    assert data["runtime"][0]["gpu_utilization_percent"]["value"] == 0


def test_enums_and_timestamps_serialise_as_strings(build):
    data = to_json_dict(build.report())
    assert data["data_origin"] == "FIXTURE"
    assert data["mode"] == "GENERAL"
    assert data["system"]["platform_profile"] == "LINUX"
    assert data["gpu"]["inventory_source"] == "NVML"
    assert data["generated_at"] == "2026-01-01T00:00:00Z"
    assert data["runtime"][0]["sampled_at"] == "2026-01-01T00:00:00Z"
    assert data["options"]["pytorch_probe_mode"] == "SUBPROCESS"


def test_optional_sections_are_null_and_lists_are_present(build):
    data = to_json_dict(build.no_gpu_report())
    assert data["readiness"] is None
    assert data["gpu"]["devices"] == []
    assert data["runtime"] == []
    assert data["gpu"]["inventory_source"] is None
    assert data["frameworks"]["pytorch"]["probe"]["duration_ms"] is None
    assert data["cuda_environment"]["environment"] == {}


def test_multiple_devices_serialise_independently(build):
    data = to_json_dict(build.report(device_count=3))
    assert [d["device_ref"] for d in data["gpu"]["devices"]] == ["nvml:0", "nvml:1", "nvml:2"]
    assert [r["device_ref"] for r in data["runtime"]] == ["nvml:0", "nvml:1", "nvml:2"]
    assert data["gpu"]["device_count"]["value"] == 3


@pytest.mark.parametrize(
    "factory",
    ["report", "no_gpu_report", "partial_failure_report", "readiness_report"],
)
def test_round_trip_is_lossless(build, factory):
    report = (
        build.report(required_vram_gib=7.5)
        if factory == "readiness_report"
        else getattr(build, factory)()
    )
    text = render_json(report)
    restored = parse_report_json(text)
    assert restored == report
    assert render_json(restored) == text


def test_round_trip_preserves_float_and_int_distinction(build):
    restored = parse_report_json(render_json(build.report()))
    assert type(restored.runtime[0].power_limit_watts.value) is float
    assert type(restored.runtime[0].memory_free_bytes.value) is int


def test_timestamp_format_and_parsing():
    whole = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    fractional = whole.replace(microsecond=120000)
    assert format_timestamp(whole) == "2026-01-01T09:00:00Z"
    assert format_timestamp(fractional) == "2026-01-01T09:00:00.120000Z"
    assert parse_timestamp("2026-01-01T09:00:00.12Z") == fractional
    for bad in ("2026-01-01T09:00:00", "2026-01-01T09:00:00+00:00", "2026-01-01 09:00:00Z", 5):
        with pytest.raises(SerializationError):
            parse_timestamp(bad)
    with pytest.raises(SerializationError, match="timezone-aware"):
        format_timestamp(datetime(2026, 1, 1))


def test_non_report_models_serialise_and_decode():
    metrics_json = {
        "value": None,
        "status": "SKIPPED",
        "source": None,
        "method": None,
        "unit": None,
        "error": None,
        "note": "--no-processes",
    }
    restored = from_json_dict(metrics_json, Observed)
    assert restored.status is CollectionStatus.SKIPPED
    with pytest.raises(SerializationError, match="model instance"):
        to_json_dict({"not": "a model"})


# --- invalid input -----------------------------------------------------------------------


@pytest.fixture
def report_data(build):
    return to_json_dict(build.report())


def _assert_rejected(data, match):
    with pytest.raises(SerializationError, match=match):
        from_json_dict(data, InspectionReport)


def test_invalid_json_text_is_rejected():
    with pytest.raises(SerializationError, match="invalid JSON"):
        parse_report_json("{not json")
    with pytest.raises(SerializationError, match="NaN is not allowed"):
        parse_report_json('{"a": NaN}')


def test_wrong_schema_name_or_version_is_rejected(report_data):
    _assert_rejected({**report_data, "schema_version": "2.0"}, "unsupported version '2.0'")
    _assert_rejected({**report_data, "schema_name": "other"}, "/schema_name")
    _assert_rejected([], "must be a JSON object")


def test_missing_and_unknown_keys_are_rejected(report_data):
    missing = copy.deepcopy(report_data)
    del missing["runtime"][0]["memory_free_bytes"]
    _assert_rejected(missing, r"/runtime/0: missing key\(s\) memory_free_bytes")
    unknown = copy.deepcopy(report_data)
    unknown["gpu"]["devices"][1]["serial"] = "123"
    _assert_rejected(unknown, r"/gpu/devices/1: unknown key\(s\) serial")


def test_zero_for_a_missing_value_is_rejected_on_input(report_data):
    data = copy.deepcopy(report_data)
    data["runtime"][0]["memory_bus_utilization_percent"]["value"] = 0
    _assert_rejected(data, "missing data is never a value")


def test_wrong_value_types_are_rejected_with_a_path(report_data):
    data = copy.deepcopy(report_data)
    data["runtime"][0]["memory_free_bytes"]["value"] = "20 GiB"
    _assert_rejected(data, "/runtime/0/memory_free_bytes/value: expected an integer")
    data = copy.deepcopy(report_data)
    data["system"]["linux_environment"]["value"] = "WSL3"
    _assert_rejected(
        data, "/system/linux_environment/value: 'WSL3' is not a valid LinuxEnvironment"
    )
    data = copy.deepcopy(report_data)
    data["gpu"]["device_count"]["value"] = True
    _assert_rejected(data, "expected an integer")


def test_invalid_enum_and_timestamp_are_rejected(report_data):
    data = copy.deepcopy(report_data)
    data["gpu"]["devices"][0]["name"]["status"] = "OK"
    _assert_rejected(data, "'OK' is not a valid CollectionStatus")
    data = copy.deepcopy(report_data)
    data["generated_at"] = "2026-01-01T00:00:00+05:30"
    _assert_rejected(data, "/generated_at")


def test_model_invariants_apply_on_decode(report_data):
    data = copy.deepcopy(report_data)
    data["gpu"]["device_count"]["value"] = 5
    _assert_rejected(data, r"/gpu: invalid GPUInventory: device_count 5")


def test_non_finite_values_cannot_be_serialised(build):
    metrics = build.runtime("nvml:0")
    object.__setattr__(metrics.power_draw_watts, "value", float("nan"))
    with pytest.raises(SerializationError, match="NaN"):
        to_json_dict(metrics)


def test_report_serialisation_is_independent_of_the_cli():
    import gpu_system_inspector.reporting.json_report as module

    assert "gpu_system_inspector.cli" not in module.__dict__


def test_runtime_unavailable_round_trip():
    metrics = RuntimeMetrics.unavailable(
        "smi:0",
        datetime(2026, 1, 1, tzinfo=UTC),
        CollectionStatus.PERMISSION_DENIED,
        Observed.failed(CollectionStatus.PERMISSION_DENIED, code="X", message="y").error,
    )
    assert from_json_dict(to_json_dict(metrics), RuntimeMetrics) == metrics
    assert to_json_dict(metrics)["memory_free_bytes"]["value"] is None
