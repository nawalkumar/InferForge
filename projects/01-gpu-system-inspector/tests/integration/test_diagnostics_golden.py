"""IMP-D00–D08 + IMP-R01 (DEC-50): the diagnostics and readiness engines, run by the real
`Inspector` over every fixture's fake adapters, reproduce the fixture catalogue's expected
outcomes — status, severity, confidence and device for every check, and the readiness
status, reasons and margins for every Mode B run. Also enforces the wording rules of
diagnostics spec §5 over every produced finding."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import ALL_FIXTURES

from gpu_system_inspector.core import Inspector
from gpu_system_inspector.diagnostics.evidence import ALLOWED_REFERENCE_DOMAINS
from gpu_system_inspector.models import (
    READINESS_DISCLAIMER,
    DiagnosticStatus,
    EvidenceKind,
    InspectionMode,
)
from gpu_system_inspector.reporting import load_report_schema, parse_report_json, render_json

S = DiagnosticStatus
CASES = [
    pytest.param(fixture, run, id=label)
    for fixture in ALL_FIXTURES
    for label, run in fixture.report_cases()
]
CHECK_IDS = tuple(f"D0{i}" for i in range(1, 9))


def _run(fixture, run):
    adapters = FakeAdapterSet.from_scenario(fixture)
    return Inspector(adapters.adapter_set()).run(fixture.options(run))


def _key(item):
    return (item.check_id, item.device_ref or "")


@pytest.mark.parametrize(("fixture", "run"), CASES)
def test_findings_match_the_fixture_expectations(fixture, run):
    report = _run(fixture, run)
    actual = {_key(f): (f.status, f.severity, f.confidence) for f in report.diagnostics}
    expected = {_key(e): (e.status, e.severity, e.confidence) for e in fixture.expected_for(run)}
    assert actual == expected
    assert [_key(f) for f in report.diagnostics] == sorted(actual)
    assert {f.check_id for f in report.diagnostics} == set(CHECK_IDS)
    assert not [e for e in report.collection_errors if e.collector.startswith("diagnostics.")]


@pytest.mark.parametrize(("fixture", "run"), [c for c in CASES if c.values[1] is not None])
def test_readiness_matches_every_mode_b_run(fixture, run):
    report = _run(fixture, run)
    result = report.readiness
    assert report.mode is InspectionMode.READINESS
    assert (result.overall_status, result.overall_reason) == (
        run.overall_status,
        run.overall_reason,
    )
    assert [(d.device_ref, d.status, d.reason) for d in result.devices] == [
        (d.device_ref, d.status, d.reason) for d in run.devices
    ]
    margins = {d.device_ref: d.margin_bytes for d in result.devices}
    for ref, margin in run.margins:
        assert margins[ref] == margin
    assert result.disclaimer == READINESS_DISCLAIMER


@pytest.mark.parametrize(("fixture", "run"), CASES)
def test_reports_with_findings_are_schema_valid_and_round_trip(fixture, run):
    report = _run(fixture, run)
    text = render_json(report)
    jsonschema.validate(json.loads(text), load_report_schema())
    assert parse_report_json(text) == report
    assert render_json(_run(fixture, run)) == text


_CAUSAL = re.compile(r"\b(cause is|caused by)\b", re.IGNORECASE)
_THERMAL = re.compile(r"throttling is occurring|is throttling|throttled|overheating", re.I)
_FIT = re.compile(r"will fit|\bfits\b|guarantee", re.IGNORECASE)
_INSTALL_TORCH = re.compile(r"(pip|conda|uv)\s+(install|add)\s+[^\n]*\btorch", re.IGNORECASE)


def _texts(finding, *, limitations=True):
    yield finding.title
    yield finding.explanation
    for e in finding.evidence:
        yield e.statement
    for r in finding.recommendations:
        yield r.action
        yield r.rationale
    if limitations:
        yield from finding.limitations


@pytest.mark.parametrize(("fixture", "run"), CASES)
def test_wording_rules(fixture, run):
    report = _run(fixture, run)
    for f in report.diagnostics:
        if f.status is not S.NOT_APPLICABLE:
            assert f.evidence, f  # AC10
        if f.status in (S.WARNING, S.FAIL):
            assert f.recommendations and f.limitations, f
        for e in f.evidence:
            if e.kind is EvidenceKind.INFERENCE:
                assert e.value is None
        for text in _texts(f):
            if f.confidence.value != "HIGH":
                assert not _CAUSAL.search(text), (f.check_id, text)
            assert not _INSTALL_TORCH.search(text), (f.check_id, text)
        for r in f.recommendations:
            if r.reference_url is not None:
                assert urlparse(r.reference_url).hostname in ALLOWED_REFERENCE_DOMAINS
        if f.check_id == "D07":
            for text in _texts(f, limitations=False):
                assert not _THERMAL.search(text), text
        if f.check_id == "D02":
            for text in _texts(f):
                assert not _FIT.search(text.replace(READINESS_DISCLAIMER, "")), text
    if report.readiness is not None:
        for d in report.readiness.devices:
            assert not _FIT.search(d.explanation)
        for text in report.readiness.limitations:
            assert not _FIT.search(text)


def _resolve(document, pointer):
    node = document
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@pytest.mark.parametrize(("fixture", "run"), CASES)
def test_observation_pointers_resolve_in_the_report(fixture, run):
    """AC28: observations reference a `report_path` that resolves in the report."""
    report = _run(fixture, run)
    document = json.loads(render_json(report))
    for f in report.diagnostics:
        for e in f.evidence:
            if e.kind is EvidenceKind.OBSERVATION:
                assert e.report_path is not None, (f.check_id, e.statement)
                _resolve(document, e.report_path)


_INSTALLERS = {"pip", "pip3", "uv", "conda", "mamba", "apt", "apt-get", "dnf", "yum", "winget"}


@pytest.mark.parametrize(("fixture", "run"), CASES)
def test_no_installer_is_ever_invoked(fixture, run):
    """AC32: the inspection runs no package installer (command log of the fake runner)."""
    adapters = FakeAdapterSet.from_scenario(fixture)
    Inspector(adapters.adapter_set()).run(fixture.options(run))
    for call in adapters.commands.calls:
        names = {Path(arg).name for arg in call.argv[:3]}
        assert not names & _INSTALLERS, call.argv
        assert "install" not in call.argv, call.argv
