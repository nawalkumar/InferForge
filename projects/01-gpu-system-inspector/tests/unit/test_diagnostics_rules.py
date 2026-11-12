"""IMP-D00–D08 + IMP-R01 (DEC-50): the engine's guard and the decision-table rows no fixture
reaches, over fixture snapshots modified in memory. Pure: no adapters, no host access."""

import dataclasses

import pytest
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.diagnostics import (
    UNEVALUATED_TITLE,
    DiagnosticInput,
    DiagnosticsEngine,
    RegisteredRule,
)
from gpu_system_inspector.diagnostics.evidence import describe, major
from gpu_system_inspector.diagnostics.rules import DEFAULT_RULES, d01, d02, d03, d04, d05, d06, d08
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRun,
    CollectorRunStatus,
    Confidence,
    DiagnosticStatus,
    InspectionOptions,
    Observed,
    PlatformProfile,
    PyTorchInfo,
    ReadinessReason,
    ReadinessStatus,
    Severity,
    Source,
)
from gpu_system_inspector.readiness import ReadinessEngine

S = DiagnosticStatus
C = CollectionStatus
OPTIONS = InspectionOptions()


def _snap(fixture_id):
    return FIXTURES_BY_ID[fixture_id].snapshot


def _one(rule, snapshot, readiness=None):
    (finding,) = rule(DiagnosticInput(snapshot, readiness, OPTIONS))
    return finding.status, finding.severity, finding.confidence, finding


def _env(snapshot, **values):
    env = dict(snapshot.cuda_environment.environment)
    for name, value in values.items():
        env[name] = Observed.collected(value, source=Source.ENVIRONMENT, method="test")
    return dataclasses.replace(
        snapshot, cuda_environment=dataclasses.replace(snapshot.cuda_environment, environment=env)
    )


def _pytorch(snapshot, **fields):
    return dataclasses.replace(
        snapshot,
        frameworks=dataclasses.replace(
            snapshot.frameworks,
            pytorch=dataclasses.replace(snapshot.frameworks.pytorch, **fields),
        ),
    )


def _driver(snapshot, **fields):
    return dataclasses.replace(snapshot, driver=dataclasses.replace(snapshot.driver, **fields))


def _denied(unit=None):
    return Observed.failed(
        C.PERMISSION_DENIED, code="NVML_ERROR_NO_PERMISSION", message="no", unit=unit
    )


def _cuda_unavailable(snapshot):
    false = Observed.collected(False, source=Source.PYTORCH, method="test")
    return _pytorch(snapshot, cuda_available=false)


# --- engine ---------------------------------------------------------------------------


def _boom(data):
    raise RuntimeError("synthetic rule failure")


def test_a_rule_that_raises_yields_unknown_and_a_collection_error():
    rules = (RegisteredRule("D01", "x", _boom), *DEFAULT_RULES[1:])
    result = DiagnosticsEngine(rules).evaluate(DiagnosticInput(_snap("FX01"), None, OPTIONS))
    first = result.findings[0]
    assert (first.check_id, first.status, first.title) == ("D01", S.UNKNOWN, UNEVALUATED_TITLE)
    (error,) = result.errors
    assert error.collector == "diagnostics.D01" and "synthetic rule failure" in error.message
    assert {f.check_id for f in result.findings} == {f"D0{i}" for i in range(1, 9)}


@pytest.mark.parametrize(
    "fn",
    [
        lambda data: [],
        lambda data: list(d05(data)) * 2,
        lambda data: d04(data),
    ],
    ids=["empty", "duplicate", "foreign-check"],
)
def test_malformed_rule_output_is_contained(fn):
    result = DiagnosticsEngine([RegisteredRule("D05", "x", fn)]).evaluate(
        DiagnosticInput(_snap("FX01"), None, OPTIONS)
    )
    assert result.findings[0].title == UNEVALUATED_TITLE
    assert result.errors[0].collector == "diagnostics.D05"


def test_findings_are_sorted_by_check_and_device():
    result = DiagnosticsEngine().evaluate(DiagnosticInput(_snap("FX10"), None, OPTIONS))
    keys = [(f.check_id, f.device_ref or "") for f in result.findings]
    assert keys == sorted(keys)


def test_helpers():
    assert major("12.4.131") == 12 and major("x") is None and major(None) is None
    assert describe(Observed.not_applicable(note="why")) == "NOT_APPLICABLE (why)"


# --- D01 ------------------------------------------------------------------------------


def test_d01_rocm_build_is_not_applicable():
    snap = _pytorch(
        _snap("FX01"), rocm_build=Observed.collected(True, source=Source.PYTORCH, method="t")
    )
    assert _one(d01, snap)[0] is S.NOT_APPLICABLE


def test_d01_unknown_when_cuda_available_was_not_collected():
    snap = _pytorch(
        _snap("FX01"), cuda_available=Observed.failed(C.COLLECTION_FAILED, code="X", message="y")
    )
    status, severity, confidence, _ = _one(d01, snap)
    assert (status, severity, confidence) == (S.UNKNOWN, Severity.MEDIUM, Confidence.NOT_APPLICABLE)


def test_d01_hidden_devices_is_a_high_confidence_cause():
    snap = _env(_cuda_unavailable(_snap("FX01")), CUDA_VISIBLE_DEVICES="")
    status, _, confidence, finding = _one(d01, snap)
    assert (status, confidence) == (S.FAIL, Confidence.HIGH)
    assert "D01.2" in finding.explanation


def test_d01_an_unknown_first_visible_entry_is_a_cause():
    snap = _env(_cuda_unavailable(_snap("FX01")), CUDA_VISIBLE_DEVICES="GPU-ffff")
    assert "D01.2" in _one(d01, snap)[3].explanation


def test_d01_unbound_driver_and_nvml_permission_causes():
    fx17 = _snap("FX17")
    snap = dataclasses.replace(fx17, frameworks=_snap("FX02b").frameworks)
    finding = _one(d01, snap)[3]
    assert "D01.3" in finding.explanation and "D01.4" in finding.explanation
    denied = _driver(_cuda_unavailable(_snap("FX01")), nvml_version=_denied())
    assert "D01.6" in _one(d01, denied)[3].explanation


def test_d01_warning_when_gpu_presence_is_undeterminable():
    snap = dataclasses.replace(_snap("FX04b"), frameworks=_snap("FX02d").frameworks)
    status, severity, confidence, _ = _one(d01, snap)
    assert (status, severity) == (S.WARNING, Severity.MEDIUM)
    assert confidence in (Confidence.MEDIUM, Confidence.LOW)


def test_d01_nvml_based_check_is_context_evidence():
    snap = _env(_cuda_unavailable(_snap("FX01")), PYTORCH_NVML_BASED_CUDA_CHECK="1")
    statements = [e.statement for e in _one(d01, snap)[3].evidence]
    assert any("PYTORCH_NVML_BASED_CUDA_CHECK" in s for s in statements)


# --- D02 / readiness ----------------------------------------------------------------


def test_d02_mode_a_is_not_applicable():
    assert _one(d02, _snap("FX01"))[0] is S.NOT_APPLICABLE


def test_readiness_mig_and_missing_runtime():
    fx = _snap("FX01")
    device = dataclasses.replace(
        fx.gpu.devices[0], mig_mode=Observed.collected("ENABLED", source=Source.NVML, method="t")
    )
    mig = dataclasses.replace(fx, gpu=dataclasses.replace(fx.gpu, devices=(device,)))
    result = ReadinessEngine().assess(mig, 8)
    assert result.devices[0].reason is ReadinessReason.MIG_ENABLED_UNSUPPORTED
    assert any("MIG" in text for text in result.limitations)
    no_runtime = dataclasses.replace(fx, runtime=())
    result = ReadinessEngine().assess(no_runtime, 8)
    assert result.devices[0].reason is ReadinessReason.FREE_MEMORY_UNAVAILABLE
    assert result.overall_status is ReadinessStatus.UNKNOWN


def test_readiness_never_concludes_absence_from_a_partial_pci_scan():
    """KI-32: a PARTIAL scan omitted devices, so it can't prove there is no NVIDIA GPU."""
    fx = _snap("FX03")
    runs = tuple(
        CollectorRun(collector="pci", status=CollectorRunStatus.PARTIAL, duration_ms=r.duration_ms)
        if r.collector == "pci"
        else r
        for r in fx.collector_runs
    )
    partial = dataclasses.replace(fx, collector_runs=runs)
    assert ReadinessEngine().assess(fx, 8).overall_reason is ReadinessReason.NO_NVIDIA_DEVICE
    result = ReadinessEngine().assess(partial, 8)
    assert result.overall_reason is ReadinessReason.DEVICE_PRESENCE_UNKNOWN
    assert _one(d04, partial)[:3] == (S.INFO, Severity.NONE, Confidence.MEDIUM)


def test_readiness_nvml_with_zero_devices_is_no_nvidia_device():
    fx = _snap("FX03")
    zero = dataclasses.replace(
        _driver(fx, nvml_version=Observed.collected("12.550", source=Source.NVML, method="t")),
        gpu=dataclasses.replace(
            fx.gpu,
            device_count=Observed.collected(0, source=Source.NVML, method="t"),
            inventory_source=Source.NVML,
        ),
    )
    assert ReadinessEngine().assess(zero, 8).overall_reason is ReadinessReason.NO_NVIDIA_DEVICE
    assert _one(d04, zero)[:2] == (S.WARNING, Severity.MEDIUM)


# --- D03 ------------------------------------------------------------------------------


def test_d03_toolkit_newer_than_driver_and_conflicting_toolkits():
    fx = _snap("FX01")
    env = fx.cuda_environment
    newer = Observed.collected("13.0.1", source=Source.CUDA_TOOLKIT, method="t")
    snap = dataclasses.replace(
        fx, cuda_environment=dataclasses.replace(env, active_nvcc_version=newer)
    )
    status, severity, confidence, _ = _one(d03, snap)
    assert (status, severity, confidence) == (S.WARNING, Severity.LOW, Confidence.MEDIUM)
    same_major = Observed.collected("12.1.105", source=Source.CUDA_TOOLKIT, method="t")
    snap = dataclasses.replace(
        fx, cuda_environment=dataclasses.replace(env, active_nvcc_version=same_major)
    )
    assert _one(d03, snap)[:3] == (S.WARNING, Severity.LOW, Confidence.HIGH)


# --- D04 / D05 / D06 ------------------------------------------------------------------


def test_d04_nvml_permission_denied_and_unknown():
    denied = _driver(_snap("FX04b"), nvml_version=_denied())
    assert _one(d04, denied)[:3] == (S.WARNING, Severity.MEDIUM, Confidence.MEDIUM)
    failed = _driver(
        _snap("FX04b"), nvml_version=Observed.failed(C.COLLECTION_FAILED, code="X", message="y")
    )
    assert _one(d04, failed)[0] is S.UNKNOWN


def test_d05_missing_binding_without_nvidia_evidence_and_skipped_pytorch():
    fx = _snap("FX04b")
    missing = _driver(fx, nvml_binding=Observed.failed(C.NOT_INSTALLED, code="X", message="y"))
    status, severity, _, finding = _one(d05, missing)
    assert (status, severity) == (S.WARNING, Severity.LOW)
    assert "unaffected" in finding.explanation
    fx01 = _snap("FX01")
    skipped = dataclasses.replace(
        fx01,
        frameworks=dataclasses.replace(
            fx01.frameworks,
            pytorch=PyTorchInfo.unavailable(C.SKIPPED, None, probe=fx01.frameworks.pytorch.probe),
        ),
    )
    assert "--no-pytorch" in _one(d05, skipped)[3].explanation


def test_d06_permission_denied_is_a_warning():
    fx = _snap("FX01")
    metrics = dataclasses.replace(fx.runtime[0], power_draw_watts=_denied("watts"))
    snap = dataclasses.replace(fx, runtime=(metrics,))
    findings = d06(DiagnosticInput(snap, None, OPTIONS))
    assert (findings[0].status, findings[0].severity) == (S.WARNING, Severity.MEDIUM)


def test_d06_and_d07_without_devices_are_single_not_applicable_findings():
    data = DiagnosticInput(_snap("FX03"), None, OPTIONS)
    for rule in (DEFAULT_RULES[5].fn, DEFAULT_RULES[6].fn):
        (finding,) = rule(data)
        assert finding.status is S.NOT_APPLICABLE and finding.device_ref is None


# --- D08 ------------------------------------------------------------------------------


def test_d08_rows():
    fx = _snap("FX01")
    assert _one(d08, _env(fx, CUDA_VISIBLE_DEVICES="7"))[:3] == (
        S.WARNING,
        Severity.MEDIUM,
        Confidence.MEDIUM,
    )
    status, severity, _, finding = _one(d08, _env(fx, CUDA_VISIBLE_DEVICES="0"))
    assert (status, severity) == (S.INFO, Severity.NONE)
    assert "see 1 device(s)" in finding.explanation
    nvd = _env(fx, NVIDIA_VISIBLE_DEVICES="all")
    assert any("container" in e.statement for e in _one(d08, nvd)[3].evidence)
    no_devices = _env(_snap("FX03"), CUDA_VISIBLE_DEVICES="0")
    assert _one(d08, no_devices)[0] is S.NOT_APPLICABLE


def test_d08_different_names_without_correlation_suggest_pci_bus_order():
    fx = _snap("FX07")
    uncorrelated = tuple(
        dataclasses.replace(d, correlated_device_ref=Observed.not_applicable(note="test"))
        for d in fx.frameworks.pytorch.devices
    )
    snap = _pytorch(fx, devices=uncorrelated)
    status, severity, confidence, finding = _one(d08, snap)
    assert (status, severity, confidence) == (S.INFO, Severity.LOW, Confidence.MEDIUM)
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in finding.explanation
    aligned = _env(snap, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    assert _one(d08, aligned)[0] is S.PASS


def test_d04_macos_and_d01_expected_absence():
    fx = _snap("FX13")
    assert fx.system.platform_profile is PlatformProfile.MACOS
    assert _one(d04, fx)[0] is S.INFO
