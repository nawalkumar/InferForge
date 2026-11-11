import copy

import jsonschema
import pytest

from gpu_system_inspector import REPORT_SCHEMA_VERSION
from gpu_system_inspector.reporting import (
    build_report_schema,
    load_report_schema,
    render_json,
    to_json_dict,
)


@pytest.fixture(scope="module")
def validator():
    schema = load_report_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _errors(validator, data):
    return sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))


def test_shipped_schema_matches_the_models():
    assert load_report_schema() == build_report_schema(), (
        "Shipped schema is out of date; regenerate with "
        "`uv run python -m gpu_system_inspector.reporting > "
        "src/gpu_system_inspector/reporting/schema/inspection_report.schema.json`"
    )


def test_schema_declares_its_version():
    schema = load_report_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(f":report:{REPORT_SCHEMA_VERSION}")
    assert schema["properties"]["schema_version"]["const"] == REPORT_SCHEMA_VERSION
    assert schema["properties"]["schema_name"]["const"] == "inferforge.gpu_system_inspector.report"


@pytest.mark.parametrize(
    "factory", ["report", "no_gpu_report", "partial_failure_report", "readiness_report", "big"]
)
def test_generated_reports_validate(build, validator, factory):
    if factory == "readiness_report":
        report = build.report(required_vram_gib=8)
    elif factory == "big":
        report = build.report(device_count=4, diagnostics=(build.warning_finding(),))
    else:
        report = getattr(build, factory)()
    assert _errors(validator, to_json_dict(report)) == []


def test_missing_optional_information_validates(build, validator):
    data = to_json_dict(build.no_gpu_report())
    assert data["readiness"] is None and data["gpu"]["devices"] == []
    assert _errors(validator, data) == []


def _mutations():
    def drop_key(d):
        del d["runtime"][0]["memory_free_bytes"]

    def extra_key(d):
        d["gpu"]["devices"][0]["serial"] = "123"

    def zero_for_missing(d):
        d["runtime"][0]["memory_bus_utilization_percent"]["value"] = 0

    def collected_without_value(d):
        d["runtime"][0]["memory_free_bytes"]["value"] = None

    def failure_without_error(d):
        d["runtime"][0]["memory_bus_utilization_percent"]["error"] = None

    def bad_status(d):
        d["system"]["os_name"]["status"] = "OK"

    def wrong_unit(d):
        d["runtime"][0]["memory_free_bytes"]["unit"] = "percent"

    def bad_device_ref(d):
        d["runtime"][0]["device_ref"] = "gpu0"

    def bad_timestamp(d):
        d["generated_at"] = "yesterday"

    def wrong_version(d):
        d["schema_version"] = "2.0"

    def readiness_in_general_mode(d):
        d["options"]["required_vram_gib"] = 8.0

    def string_for_integer(d):
        d["gpu"]["device_count"]["value"] = "2"

    def bad_environment_name(d):
        d["cuda_environment"]["environment"]["HOME"] = d["cuda_environment"]["environment"][
            "CUDA_HOME"
        ]

    return [
        drop_key,
        extra_key,
        zero_for_missing,
        collected_without_value,
        failure_without_error,
        bad_status,
        wrong_unit,
        bad_device_ref,
        bad_timestamp,
        wrong_version,
        readiness_in_general_mode,
        string_for_integer,
        bad_environment_name,
    ]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_invalid_reports_fail_validation(build, validator, mutate):
    data = copy.deepcopy(to_json_dict(build.report()))
    mutate(data)
    assert _errors(validator, data), f"{mutate.__name__} should be rejected"


def test_multiple_devices_are_validated_individually(build, validator):
    data = to_json_dict(build.report(device_count=3))
    data["runtime"][2]["memory_free_bytes"]["value"] = -1.5
    paths = [list(e.absolute_path)[:2] for e in _errors(validator, data)]
    assert ["runtime", 2] in paths
    assert ["runtime", 0] not in paths


def test_round_tripped_json_still_validates(build, validator):
    import json

    data = json.loads(render_json(build.report(required_vram_gib=7.5)))
    assert _errors(validator, data) == []
