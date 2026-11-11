"""Diagnostic finding models (docs/data_model_specification.md §6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    JSON_POINTER_REGEX,
    constrained,
    set_field,
    validate_model,
)
from gpu_system_inspector.models.enums import (
    Confidence,
    DiagnosticStatus,
    EvidenceKind,
    Severity,
    Source,
)
from gpu_system_inspector.models.observed import InvalidModelError

CHECK_ID_REGEX = r"^D[0-9]{2}$"

VALID_SEVERITIES: dict[DiagnosticStatus, frozenset[Severity]] = {
    DiagnosticStatus.PASS: frozenset({Severity.NONE}),
    DiagnosticStatus.NOT_APPLICABLE: frozenset({Severity.NONE}),
    DiagnosticStatus.INFO: frozenset({Severity.NONE, Severity.LOW}),
    DiagnosticStatus.WARNING: frozenset({Severity.LOW, Severity.MEDIUM}),
    DiagnosticStatus.UNKNOWN: frozenset({Severity.LOW, Severity.MEDIUM}),
    DiagnosticStatus.FAIL: frozenset({Severity.HIGH}),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Evidence:
    kind: EvidenceKind
    statement: str = constrained(minLength=1)
    source: Source | None
    report_path: str | None = constrained(pattern=JSON_POINTER_REGEX)
    value: Any

    def __post_init__(self) -> None:
        validate_model(self)
        if self.kind is EvidenceKind.INFERENCE and self.value is not None:
            raise InvalidModelError("INFERENCE evidence must not carry a value")


@dataclass(frozen=True, slots=True, kw_only=True)
class Recommendation:
    action: str = constrained(minLength=1)
    rationale: str = constrained(minLength=1)
    reference_url: str | None = constrained(pattern=r"^https://")

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticFinding:
    _SCHEMA_EXTRA: ClassVar[dict[str, Any]] = {
        "allOf": [
            {
                "if": {"properties": {"status": {"const": status.value}}},
                "then": {"properties": {"severity": {"enum": sorted(s.value for s in severities)}}},
            }
            for status, severities in VALID_SEVERITIES.items()
        ]
    }

    check_id: str = constrained(pattern=CHECK_ID_REGEX)
    device_ref: str | None = constrained(pattern=DEVICE_REF_REGEX)
    status: DiagnosticStatus
    severity: Severity
    title: str = constrained(minLength=1)
    explanation: str = constrained(minLength=1)
    evidence: tuple[Evidence, ...]
    detection_sources: tuple[Source, ...]
    recommendations: tuple[Recommendation, ...]
    confidence: Confidence
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_model(self)
        set_field(self, "detection_sources", tuple(sorted(set(self.detection_sources))))
        status = self.status
        if self.severity not in VALID_SEVERITIES[status]:
            allowed = ", ".join(sorted(VALID_SEVERITIES[status]))
            raise InvalidModelError(
                f"severity {self.severity} isn't valid for a {status} finding (allowed: {allowed})"
            )
        if status is not DiagnosticStatus.NOT_APPLICABLE and not self.evidence:
            raise InvalidModelError(f"a {status} finding needs at least one evidence item")
        if status in (DiagnosticStatus.WARNING, DiagnosticStatus.FAIL) and not self.recommendations:
            raise InvalidModelError(f"a {status} finding needs at least one recommendation")
        if (
            status in (DiagnosticStatus.WARNING, DiagnosticStatus.FAIL, DiagnosticStatus.UNKNOWN)
            and not self.limitations
        ):
            raise InvalidModelError(f"a {status} finding needs at least one limitation")
        if status is DiagnosticStatus.NOT_APPLICABLE and self.confidence is not (
            Confidence.NOT_APPLICABLE
        ):
            raise InvalidModelError("a NOT_APPLICABLE finding must have NOT_APPLICABLE confidence")
        if any(not text for text in self.limitations):
            raise InvalidModelError("limitations must not contain empty strings")
