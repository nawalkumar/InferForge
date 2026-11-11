"""The FX fixture catalog matches the approved specification (testing strategy §4,
diagnostics spec §6, testing strategy §4.1)."""

from collections import Counter
from dataclasses import fields, is_dataclass

import pytest
from gsi_testkit import spec_tables
from gsi_testkit.builders import FIXTURE_METHOD_PREFIX
from gsi_testkit.expectations import CHECK_IDS, PER_DEVICE_CHECKS
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID, report_cases

from gpu_system_inspector.models import (
    VALID_SEVERITIES,
    Confidence,
    DataOrigin,
    DiagnosticStatus,
    Observed,
    ReadinessReason,
    ReadinessStatus,
    required_bytes_for,
)

SPEC_FIXTURE_IDS = (
    "FX01",
    "FX02a",
    "FX02b",
    "FX02c",
    "FX02d",
    "FX03",
    "FX04a",
    "FX04b",
    "FX05",
    "FX06",
    "FX07",
    "FX08",
    "FX09",
    "FX10",
    "FX11",
    "FX12",
    "FX13",
    "FX14",
    "FX15",
    "FX16",
    "FX17",
    "FX18",
    "FX19",
    "FX20",
)
MODE_B_ONLY = {"FX07", "FX14", "FX19"}
EXTRA_MODE_B_RUNS = {"FX01 (B)", "FX03 (B)", "FX04b (B)", "FX06 (B)", "FX09 (B)", "FX13 (B)"}
DOCS_MISSING = pytest.mark.skipif(
    not spec_tables.DOCS_DIR.is_dir(),
    reason="specification documents are not available (e.g. isolated wheel test run)",
)

ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)


def test_catalog_has_exactly_the_20_fixtures_and_24_variants():
    assert tuple(f.fixture_id for f in ALL_FIXTURES) == SPEC_FIXTURE_IDS
    assert len({f.fixture_id[:4] for f in ALL_FIXTURES}) == 20
    assert {f.fixture_id for f in ALL_FIXTURES if f.mode_b_only} == MODE_B_ONLY


def test_report_cases_cover_mode_a_and_every_mode_b_run():
    labels = [label for _, label, _ in report_cases()]
    assert len(labels) == len(set(labels)) == 30
    runs = {label for _, label, run in report_cases() if run is not None}
    assert runs == MODE_B_ONLY | EXTRA_MODE_B_RUNS


@ids
def test_fixture_is_fully_described(fixture):
    assert fixture.name and fixture.description
    assert fixture.data_origin is DataOrigin.FIXTURE
    assert fixture.premises, "every fixture states the input facts its outcomes depend on"


@ids
def test_premises_hold(fixture):
    failing = [p.description for p in fixture.premises if not p.holds(fixture.snapshot)]
    assert failing == []


@ids
def test_every_check_is_expected_once_per_scope(fixture):
    expected = fixture.expected_diagnostics
    keys = [(e.check_id, e.device_ref) for e in expected]
    assert len(keys) == len(set(keys))
    assert set(e.check_id for e in expected) == set(CHECK_IDS)
    refs = fixture.device_refs
    for check in PER_DEVICE_CHECKS:
        scoped = sorted(e.device_ref or "" for e in expected if e.check_id == check)
        assert scoped == (sorted(refs) if refs else [""]), check
    for check in set(CHECK_IDS) - set(PER_DEVICE_CHECKS):
        assert [e.device_ref for e in expected if e.check_id == check] == [None]


@ids
def test_expected_severity_and_confidence_are_valid(fixture):
    for e in fixture.expected_diagnostics:
        assert e.severity in VALID_SEVERITIES[e.status], e
        if e.status is DiagnosticStatus.NOT_APPLICABLE:
            assert e.confidence is Confidence.NOT_APPLICABLE, e
        assert e.basis, e


@DOCS_MISSING
@ids
def test_expected_statuses_match_the_diagnostics_matrix(fixture):
    matrix = spec_tables.diagnostics_matrix()[fixture.fixture_id]
    for check in CHECK_IDS:
        cell = matrix[check]
        for e in (e for e in fixture.expected_diagnostics if e.check_id == check):
            spec_status = cell.get(e.device_ref, cell.get(None))
            assert e.status.value == spec_status, (fixture.fixture_id, check, e.device_ref)


@DOCS_MISSING
def test_readiness_runs_match_the_readiness_table():
    table = spec_tables.readiness_table()
    runs = {run.label: run for f in ALL_FIXTURES for run in f.readiness_runs}
    assert set(runs) == set(table)
    for label, row in table.items():
        run = runs[label]
        assert run.required_vram_gib == row.required_vram_gib, label
        assert (run.overall_status.value, run.overall_reason.value) == (
            row.overall_status,
            row.overall_reason,
        ), label
        assert run.d02.status.value == row.d02, label
        assert int(run.exit_code) == row.exit_code, label
        devices = spec_tables.device_results(row.device_text)
        actual = {d.device_ref: d.reason.value for d in run.devices}
        if isinstance(devices, str):
            assert set(actual.values()) == {devices} and len(actual) >= 2, label
        else:
            assert actual == devices, label


@pytest.mark.parametrize(
    ("fixture", "label", "run"),
    [case for case in report_cases() if case[2] is not None],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_expected_readiness_agrees_with_fixture_memory(fixture, label, run):
    """The expectations obey the readiness rules (architecture §4.9) for this fixture's data."""
    required = required_bytes_for(run.required_vram_gib)
    runtime = {r.device_ref: r for r in fixture.snapshot.runtime}
    devices = {d.device_ref: d for d in fixture.snapshot.gpu.devices}
    assert {d.device_ref for d in run.devices} == set(devices)
    for expected in run.devices:
        free, total = (
            runtime[expected.device_ref].memory_free_bytes,
            runtime[expected.device_ref].memory_total_bytes,
        )
        mig = devices[expected.device_ref].mig_mode
        if expected.reason is ReadinessReason.MIG_ENABLED_UNSUPPORTED:
            assert mig.value == "ENABLED"
        elif expected.reason is ReadinessReason.FREE_MEMORY_UNAVAILABLE:
            assert not free.is_collected
        elif expected.reason is ReadinessReason.INSUFFICIENT_TOTAL_MEMORY:
            assert total.value < required
        elif expected.reason is ReadinessReason.INSUFFICIENT_FREE_MEMORY:
            assert free.value < required <= total.value
        else:
            assert free.value >= required
    for ref, margin in run.margins:
        assert runtime[ref].memory_free_bytes.value - required == margin, (label, ref)
    statuses = {d.status for d in run.devices}
    if ReadinessStatus.READY in statuses:
        assert run.overall_status is ReadinessStatus.READY


def _observed_values(node):
    if isinstance(node, Observed):
        yield node
        node = node.value
    if is_dataclass(node) and not isinstance(node, type):
        for f in fields(node):
            yield from _observed_values(getattr(node, f.name))
    elif isinstance(node, (tuple, list)):
        for item in node:
            yield from _observed_values(item)
    elif hasattr(node, "values") and not isinstance(node, str):
        for item in node.values():
            yield from _observed_values(item)


@ids
def test_every_synthetic_value_carries_fixture_provenance(fixture):
    values = list(_observed_values(fixture.snapshot))
    assert values
    bad = [
        o for o in values if o.method is not None and not o.method.startswith(FIXTURE_METHOD_PREFIX)
    ]
    assert bad == []
    assert all(o.method is not None for o in values if o.is_collected)


def test_fixture_ids_are_unique_and_indexed():
    assert Counter(f.fixture_id for f in ALL_FIXTURES).most_common(1)[0][1] == 1
    assert FIXTURES_BY_ID["FX10"].fixture_id == "FX10"
