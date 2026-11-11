"""Expected diagnostic and readiness outcomes for the FX fixtures.

These are *expectations* transcribed from the specification (diagnostics spec §3/§6,
testing strategy §4.1), not the output of a diagnostics or readiness engine. They are
materialised into real `DiagnosticFinding` / `ReadinessResult` models so the data contract
(model invariants, JSON Schema, round trip) can be checked for every expected outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gpu_system_inspector.cli.exit_codes import ExitCode
from gpu_system_inspector.models import (
    READINESS_CRITERION,
    READINESS_DISCLAIMER,
    Confidence,
    DeviceReadiness,
    DiagnosticFinding,
    DiagnosticStatus,
    Evidence,
    EvidenceKind,
    InspectionSnapshot,
    ReadinessReason,
    ReadinessResult,
    ReadinessStatus,
    Recommendation,
    Severity,
    Source,
    required_bytes_for,
)

CHECK_IDS = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
PER_DEVICE_CHECKS = ("D06", "D07")

S = DiagnosticStatus
_DEFAULT_SEVERITY = {
    S.PASS: Severity.NONE,
    S.INFO: Severity.NONE,
    S.NOT_APPLICABLE: Severity.NONE,
    S.WARNING: Severity.MEDIUM,
    S.UNKNOWN: Severity.MEDIUM,
    S.FAIL: Severity.HIGH,
}

READINESS_EXIT = {
    ReadinessStatus.READY: ExitCode.SUCCESS,
    ReadinessStatus.NOT_READY: ExitCode.READINESS_NOT_MET,
    ReadinessStatus.UNKNOWN: ExitCode.READINESS_UNKNOWN,
}


@dataclass(frozen=True)
class ExpectedFinding:
    check_id: str
    status: DiagnosticStatus
    severity: Severity
    confidence: Confidence
    device_ref: str | None
    basis: str


def expect(
    check_id: str,
    status: DiagnosticStatus,
    basis: str,
    *,
    severity: Severity | None = None,
    confidence: Confidence | None = None,
    device_ref: str | None = None,
) -> ExpectedFinding:
    if confidence is None:
        confidence = Confidence.NOT_APPLICABLE if status is S.NOT_APPLICABLE else Confidence.HIGH
    return ExpectedFinding(
        check_id, status, severity or _DEFAULT_SEVERITY[status], confidence, device_ref, basis
    )


def per_device(
    check_id: str, refs: Sequence[str], status: DiagnosticStatus, basis: str, **kw: object
) -> tuple[ExpectedFinding, ...]:
    """One finding per device, or a single system-wide NOT_APPLICABLE when there are none."""
    if not refs:
        return (expect(check_id, S.NOT_APPLICABLE, f"{check_id} row 1 (no inventoried devices)"),)
    return tuple(expect(check_id, status, basis, device_ref=ref, **kw) for ref in refs)  # type: ignore[arg-type]


def healthy(refs: Sequence[str]) -> tuple[ExpectedFinding, ...]:
    """Outcomes of a fully working FX01-style machine (diagnostics spec §6, row FX01)."""
    return (
        expect("D01", S.PASS, "D01 row 5 (cuda_available = true)"),
        expect("D02", S.NOT_APPLICABLE, "D02 Mode A (no --required-vram)"),
        expect("D03", S.INFO, "D03 row 5"),
        expect("D04", S.PASS, "D04 row 1 (NVML device_count >= 1)"),
        expect("D05", S.PASS, "D05 (all dependencies present)"),
        *per_device("D06", refs, S.PASS, "D06 row 5 (all fields COLLECTED)"),
        *per_device("D07", refs, S.INFO, "D07 row 3"),
        expect("D08", S.PASS, "D08 row 7"),
    )


def no_gpu(
    d01: ExpectedFinding, d04: ExpectedFinding, d05: ExpectedFinding
) -> tuple[ExpectedFinding, ...]:
    """Machines without an accessible NVIDIA GPU: D03, D06–D08 are NOT_APPLICABLE."""
    return (
        d01,
        expect("D02", S.NOT_APPLICABLE, "D02 Mode A (no --required-vram)"),
        expect("D03", S.NOT_APPLICABLE, "D03 row 1 (no CUDA version collected)"),
        d04,
        d05,
        *per_device("D06", (), S.NOT_APPLICABLE, ""),
        *per_device("D07", (), S.NOT_APPLICABLE, ""),
        expect("D08", S.NOT_APPLICABLE, "D08 row 1 (no devices, CUDA_VISIBLE_DEVICES unset)"),
    )


def replace_checks(
    base: Sequence[ExpectedFinding], *new: ExpectedFinding
) -> tuple[ExpectedFinding, ...]:
    """Replace every expectation of the checks in `new` (all devices) with `new`."""
    replaced = {e.check_id for e in new}
    return tuple(e for e in base if e.check_id not in replaced) + tuple(new)


@dataclass(frozen=True)
class ExpectedDeviceReadiness:
    device_ref: str
    status: ReadinessStatus
    reason: ReadinessReason


@dataclass(frozen=True)
class ReadinessRun:
    """One Mode B run from testing strategy §4.1."""

    label: str
    required_vram_gib: float
    overall_status: ReadinessStatus
    overall_reason: ReadinessReason
    devices: tuple[ExpectedDeviceReadiness, ...]
    d02: ExpectedFinding
    exit_code: ExitCode
    margins: tuple[tuple[str, int], ...] = ()


def device_expectation(ref: str, reason: ReadinessReason) -> ExpectedDeviceReadiness:
    status = {
        ReadinessReason.SUFFICIENT_FREE_MEMORY: ReadinessStatus.READY,
        ReadinessReason.INSUFFICIENT_FREE_MEMORY: ReadinessStatus.NOT_READY,
        ReadinessReason.INSUFFICIENT_TOTAL_MEMORY: ReadinessStatus.NOT_READY,
        ReadinessReason.FREE_MEMORY_UNAVAILABLE: ReadinessStatus.UNKNOWN,
        ReadinessReason.MIG_ENABLED_UNSUPPORTED: ReadinessStatus.UNKNOWN,
    }[reason]
    return ExpectedDeviceReadiness(ref, status, reason)


def readiness_run(
    label: str,
    required_vram_gib: float,
    overall_reason: ReadinessReason,
    devices: Sequence[ExpectedDeviceReadiness] = (),
    *,
    d02_confidence: Confidence | None = None,
    margins: Sequence[tuple[str, int]] = (),
) -> ReadinessRun:
    overall = {
        ReadinessReason.AT_LEAST_ONE_DEVICE_READY: ReadinessStatus.READY,
        ReadinessReason.NO_DEVICE_READY: ReadinessStatus.NOT_READY,
        ReadinessReason.NO_NVIDIA_DEVICE: ReadinessStatus.NOT_READY,
        ReadinessReason.PLATFORM_NOT_SUPPORTED: ReadinessStatus.NOT_READY,
        ReadinessReason.SOME_DEVICES_UNKNOWN: ReadinessStatus.UNKNOWN,
        ReadinessReason.DEVICE_PRESENCE_UNKNOWN: ReadinessStatus.UNKNOWN,
    }[overall_reason]
    d02_status = {
        ReadinessStatus.READY: S.PASS,
        ReadinessStatus.NOT_READY: S.FAIL,
        ReadinessStatus.UNKNOWN: S.UNKNOWN,
    }[overall]
    d02 = expect(
        "D02", d02_status, f"D02 mapping from readiness {overall}", confidence=d02_confidence
    )
    return ReadinessRun(
        label,
        required_vram_gib,
        overall,
        overall_reason,
        tuple(devices),
        d02,
        READINESS_EXIT[overall],
        tuple(margins),
    )


# --- materialisation into models ------------------------------------------------------------

_SYNTHETIC_LIMITATION = "Synthetic fixture expectation; not produced by a diagnostics engine."


def materialize_finding(expected: ExpectedFinding, fixture_id: str) -> DiagnosticFinding:
    needs_evidence = expected.status is not S.NOT_APPLICABLE
    needs_action = expected.status in (S.WARNING, S.FAIL)
    needs_limitation = expected.status in (S.WARNING, S.FAIL, S.UNKNOWN)
    return DiagnosticFinding(
        check_id=expected.check_id,
        device_ref=expected.device_ref,
        status=expected.status,
        severity=expected.severity,
        title=f"{expected.check_id} expected {expected.status} ({fixture_id})",
        explanation=f"Expected outcome for fixture {fixture_id}: {expected.basis}.",
        evidence=(
            Evidence(
                kind=EvidenceKind.INFERENCE,
                statement=f"Fixture {fixture_id} expectation ({expected.basis})",
                source=Source.INSPECTOR,
                report_path=None,
                value=None,
            ),
        )
        if needs_evidence
        else (),
        detection_sources=(Source.INSPECTOR,) if needs_evidence else (),
        recommendations=(
            Recommendation(
                action="None: synthetic fixture expectation.",
                rationale="Placeholder required by the finding contract.",
                reference_url=None,
            ),
        )
        if needs_action
        else (),
        confidence=expected.confidence,
        limitations=(_SYNTHETIC_LIMITATION,) if needs_limitation else (),
    )


def materialize_readiness(run: ReadinessRun, snapshot: InspectionSnapshot) -> ReadinessResult:
    """Build the expected `ReadinessResult`, copying free/total memory from the snapshot."""
    required = required_bytes_for(run.required_vram_gib)
    runtime = {r.device_ref: r for r in snapshot.runtime}
    devices = []
    for expected in run.devices:
        metrics = runtime[expected.device_ref]
        free = metrics.memory_free_bytes
        devices.append(
            DeviceReadiness(
                device_ref=expected.device_ref,
                status=expected.status,
                reason=expected.reason,
                free_bytes=free,
                total_bytes=metrics.memory_total_bytes,
                margin_bytes=free.value - required if free.is_collected else None,
                explanation=f"Expected {expected.reason} ({run.label}).",
            )
        )
    return ReadinessResult(
        required_vram_gib=run.required_vram_gib,
        required_bytes=required,
        criterion=READINESS_CRITERION,
        overall_status=run.overall_status,
        overall_reason=run.overall_reason,
        devices=tuple(devices),
        disclaimer=READINESS_DISCLAIMER,
        limitations=("Synthetic fixture snapshot; not a real measurement.",),
    )
