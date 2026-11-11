"""Every fixture report case passes the full data contract: models, JSON, schema, round trip,
and carries exactly its expected diagnostic and readiness outcomes."""

import json

import jsonschema
import pytest
from gsi_testkit.fixtures import report_cases

from gpu_system_inspector.models import DataOrigin, InspectionMode
from gpu_system_inspector.reporting import load_report_schema, parse_report_json, render_json

CASES = report_cases()
case_ids = pytest.mark.parametrize(
    ("fixture", "label", "run"), CASES, ids=[label for _, label, _ in CASES]
)


@pytest.fixture(scope="module")
def validator():
    schema = load_report_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@case_ids
def test_report_validates_against_the_shipped_schema(validator, fixture, label, run):
    data = json.loads(render_json(fixture.report(run)))
    errors = [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(data)]
    assert errors == []


@case_ids
def test_report_round_trips_losslessly(fixture, label, run):
    report = fixture.report(run)
    text = render_json(report)
    restored = parse_report_json(text)
    assert restored == report
    assert render_json(restored) == text
    assert render_json(fixture.report(run)) == text  # deterministic across rebuilds


@case_ids
def test_report_is_marked_as_fixture_data(fixture, label, run):
    report = fixture.report(run)
    assert report.data_origin is DataOrigin.FIXTURE
    assert any(f"Synthetic fixture {fixture.fixture_id}" in text for text in report.limitations)


@case_ids
def test_report_carries_exactly_the_expected_findings(fixture, label, run):
    report = fixture.report(run)
    actual = {
        (d.check_id, d.device_ref): (d.status, d.severity, d.confidence) for d in report.diagnostics
    }
    expected = {
        (e.check_id, e.device_ref): (e.status, e.severity, e.confidence)
        for e in fixture.expected_for(run)
    }
    assert actual == expected


@case_ids
def test_report_mode_and_readiness_match_the_run(fixture, label, run):
    report = fixture.report(run)
    if run is None:
        assert report.mode is InspectionMode.GENERAL and report.readiness is None
        return
    assert report.mode is InspectionMode.READINESS
    readiness = report.readiness
    assert (readiness.overall_status, readiness.overall_reason) == (
        run.overall_status,
        run.overall_reason,
    )
    assert {(d.device_ref, d.status, d.reason) for d in readiness.devices} == {
        (d.device_ref, d.status, d.reason) for d in run.devices
    }
    for device in readiness.devices:
        runtime = next(r for r in report.runtime if r.device_ref == device.device_ref)
        assert device.free_bytes == runtime.memory_free_bytes  # copied with its provenance
    d02 = next(d for d in report.diagnostics if d.check_id == "D02")
    assert d02.status is run.d02.status
