"""The diagnostics engine (architecture §4.8, IMP-D00, DEC-50).

Rules are pure functions of `DiagnosticInput` (the snapshot, the readiness result and the
options): no I/O, no clock, no hardware access. They run in check-ID order; each is guarded,
so a rule that raises yields one `UNKNOWN` finding ("Check could not be evaluated") and a
`CollectionError` (`collector = "diagnostics.Dxx"`), and the other rules still run. A rule
must emit at least one finding and at most one per `device_ref`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gpu_system_inspector.diagnostics.evidence import finding, inference
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    Confidence,
    DiagnosticFinding,
    DiagnosticStatus,
    InspectionOptions,
    InspectionSnapshot,
    ReadinessResult,
    Severity,
)

UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"
UNEVALUATED_TITLE = "Check could not be evaluated"


@dataclass(frozen=True, slots=True)
class DiagnosticInput:
    snapshot: InspectionSnapshot
    readiness: ReadinessResult | None
    options: InspectionOptions


Rule = Callable[[DiagnosticInput], Sequence[DiagnosticFinding]]


@dataclass(frozen=True, slots=True)
class RegisteredRule:
    check_id: str
    title: str
    fn: Rule


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    findings: tuple[DiagnosticFinding, ...]
    errors: tuple[CollectionError, ...]


class DiagnosticsEngine:
    def __init__(self, rules: Sequence[RegisteredRule] | None = None) -> None:
        if rules is None:
            from gpu_system_inspector.diagnostics.rules import DEFAULT_RULES

            rules = DEFAULT_RULES
        self._rules = tuple(sorted(rules, key=lambda r: r.check_id))

    def evaluate(self, data: DiagnosticInput) -> DiagnosticsResult:
        findings: list[DiagnosticFinding] = []
        errors: list[CollectionError] = []
        for rule in self._rules:
            try:
                produced = tuple(rule.fn(data))
                _check(rule, produced)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                findings.append(_unevaluated(rule, message))
                errors.append(
                    CollectionError(
                        collector=f"diagnostics.{rule.check_id}",
                        device_ref=None,
                        status=CollectionStatus.COLLECTION_FAILED,
                        error_code=UNHANDLED_EXCEPTION,
                        message=message,
                        report_path="/diagnostics",
                    )
                )
                continue
            findings.extend(produced)
        ordered = sorted(findings, key=lambda f: (f.check_id, f.device_ref or ""))
        return DiagnosticsResult(findings=tuple(ordered), errors=tuple(errors))


def _check(rule: RegisteredRule, produced: tuple[DiagnosticFinding, ...]) -> None:
    if not produced:
        raise ValueError(f"rule {rule.check_id} emitted no finding")
    if any(f.check_id != rule.check_id for f in produced):
        raise ValueError(f"rule {rule.check_id} emitted another check's finding")
    refs = [f.device_ref for f in produced]
    if len(set(refs)) != len(refs):
        raise ValueError(f"rule {rule.check_id} emitted two findings for one device")


def _unevaluated(rule: RegisteredRule, message: str) -> DiagnosticFinding:
    return finding(
        rule.check_id,
        DiagnosticStatus.UNKNOWN,
        Severity.MEDIUM,
        Confidence.NOT_APPLICABLE,
        UNEVALUATED_TITLE,
        f"The {rule.check_id} check ({rule.title}) raised an error, so no conclusion was drawn.",
        (inference(f"The check failed with {message}."),),
        limitations=("This check's result is unknown; the other checks are unaffected.",),
    )
