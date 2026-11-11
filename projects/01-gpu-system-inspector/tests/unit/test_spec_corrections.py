"""Regression tests for the Stage 3 specification corrections (data model §3.2, §1;
diagnostics spec §2)."""

import dataclasses
import random

import jsonschema
import pytest
from gsi_testkit.builders import FIXED_TIME, GIB, Builders, not_applicable

from gpu_system_inspector.models import (
    VALID_SEVERITIES,
    CollectionStatus,
    Confidence,
    DiagnosticFinding,
    DiagnosticStatus,
    ErrorDetail,
    InspectionReport,
    InvalidModelError,
    Observed,
    Severity,
    Source,
)
from gpu_system_inspector.reporting import (
    load_report_schema,
    parse_report_json,
    render_json,
    to_json_dict,
)

ALL_COMBINATIONS = [(status, severity) for status in DiagnosticStatus for severity in Severity]


@pytest.fixture(scope="module")
def validator():
    return jsonschema.Draft202012Validator(load_report_schema())


def _finding(status: DiagnosticStatus, severity: Severity, **kw) -> DiagnosticFinding:
    base = Builders.warning_finding()
    confidence = (
        Confidence.NOT_APPLICABLE
        if status is DiagnosticStatus.NOT_APPLICABLE
        else (Confidence.MEDIUM)
    )
    return dataclasses.replace(base, status=status, severity=severity, confidence=confidence, **kw)


# --- B. diagnostic severity -------------------------------------------------------------


def test_severity_table_matches_the_corrected_specification():
    assert VALID_SEVERITIES == {
        DiagnosticStatus.PASS: {Severity.NONE},
        DiagnosticStatus.NOT_APPLICABLE: {Severity.NONE},
        DiagnosticStatus.INFO: {Severity.NONE, Severity.LOW},
        DiagnosticStatus.WARNING: {Severity.LOW, Severity.MEDIUM},
        DiagnosticStatus.UNKNOWN: {Severity.LOW, Severity.MEDIUM},
        DiagnosticStatus.FAIL: {Severity.HIGH},
    }


@pytest.mark.parametrize(("status", "severity"), ALL_COMBINATIONS)
def test_every_status_severity_combination_is_validated(status, severity):
    if severity in VALID_SEVERITIES[status]:
        assert _finding(status, severity).severity is severity
    else:
        with pytest.raises(InvalidModelError, match="isn't valid for a"):
            _finding(status, severity)


def test_d08_info_with_low_severity_is_valid():
    finding = _finding(DiagnosticStatus.INFO, Severity.LOW, check_id="D08", device_ref=None)
    assert (finding.status, finding.severity) == (DiagnosticStatus.INFO, Severity.LOW)


@pytest.mark.parametrize(
    ("status", "severity"),
    [(st, sv) for st, sv in ALL_COMBINATIONS if sv not in VALID_SEVERITIES[st]],
)
def test_schema_rejects_invalid_severity(validator, status, severity):
    data = to_json_dict(Builders.report(diagnostics=(Builders.warning_finding(),)))
    finding = next(d for d in data["diagnostics"] if d["check_id"] == "D07")
    finding["status"], finding["severity"] = status.value, severity.value
    assert any("severity" in list(e.absolute_path) for e in validator.iter_errors(data))


# --- A. NOT_APPLICABLE ----------------------------------------------------------------------


def test_not_applicable_may_carry_an_error_but_needs_a_note(validator):
    inapplicable = Observed.not_applicable(
        note="No NVML inventory to correlate with",
        error=ErrorDetail(code="NVML_ERROR_LIBRARY_NOT_FOUND", message="NVML missing"),
    )
    assert inapplicable.error is not None
    pytorch = Builders.pytorch(2)
    device = dataclasses.replace(pytorch.devices[0], correlated_device_ref=inapplicable)
    report = Builders.report()
    report = dataclasses.replace(
        report,
        frameworks=dataclasses.replace(
            report.frameworks,
            pytorch=dataclasses.replace(pytorch, devices=(device, pytorch.devices[1])),
        ),
    )
    data = to_json_dict(report)
    assert list(validator.iter_errors(data)) == []
    assert parse_report_json(render_json(report)) == report
    data["frameworks"]["pytorch"]["devices"][0]["correlated_device_ref"]["note"] = None
    assert list(validator.iter_errors(data))
    with pytest.raises(InvalidModelError, match="explain why"):
        Observed(
            value=None,
            status=CollectionStatus.NOT_APPLICABLE,
            source=None,
            method=None,
            error=inapplicable.error,
        )


def test_not_applicable_is_not_reported_as_a_collection_failure():
    obs = not_applicable("CPU-only PyTorch build")
    assert obs.status is CollectionStatus.NOT_APPLICABLE
    assert obs.error is None


# --- C. device ordering ---------------------------------------------------------------------


def _device_report(indices, *, seed: int) -> InspectionReport:
    devices = [Builders.device(i) for i in indices]
    runtime = [Builders.runtime(d.device_ref) for d in devices]
    rng = random.Random(seed)
    rng.shuffle(devices)
    rng.shuffle(runtime)
    base = Builders.snapshot(1)
    snapshot = dataclasses.replace(
        base,
        gpu=Builders.inventory(tuple(devices)),
        frameworks=dataclasses.replace(
            base.frameworks,
            pytorch=dataclasses.replace(
                base.frameworks.pytorch,
                devices=(),
                device_count=Observed.collected(0, source=Source.PYTORCH, method="fixture:x"),
            ),
        ),
        runtime=tuple(runtime),
    )
    return InspectionReport.from_snapshot(
        snapshot,
        generated_at=FIXED_TIME,
        duration_ms=1,
        data_origin=Builders.report().data_origin,
        options=Builders.report().options,
    )


def test_device_ordering_is_natural_numeric():
    report = _device_report([10, 2, 0, 1], seed=0)
    assert report.gpu.device_refs == ("nvml:0", "nvml:1", "nvml:2", "nvml:10")
    assert [r.device_ref for r in report.runtime] == ["nvml:0", "nvml:1", "nvml:2", "nvml:10"]


@pytest.mark.parametrize("seed", range(5))
def test_device_ordering_is_stable_regardless_of_input_order(seed):
    assert render_json(_device_report([10, 2, 0, 1], seed=seed)) == render_json(
        _device_report([10, 2, 0, 1], seed=99)
    )


def test_readiness_and_findings_use_natural_device_order():
    readiness = Builders.readiness(8, {"nvml:10": 20 * GIB, "nvml:2": 20 * GIB})
    assert [d.device_ref for d in readiness.devices] == ["nvml:2", "nvml:10"]
