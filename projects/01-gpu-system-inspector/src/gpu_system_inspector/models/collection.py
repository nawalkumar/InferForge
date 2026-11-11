"""Collection bookkeeping models (docs/data_model_specification.md §3.4, §7.1).

`CollectorOutcome` is internal (never serialised): it carries what one guarded collector
run produced (docs/data_flow.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from gpu_system_inspector.models._validation import (
    DEVICE_REF_REGEX,
    JSON_POINTER_REGEX,
    constrained,
    validate_model,
)
from gpu_system_inspector.models.enums import CollectionStatus, CollectorRunStatus
from gpu_system_inspector.models.observed import InvalidModelError

SectionT = TypeVar("SectionT")

CAPABILITY_REGEX = r"^[A-Z0-9_]+(\.[A-Z0-9_]+)+$"


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionError:
    collector: str = constrained(minLength=1)
    device_ref: str | None = constrained(pattern=DEVICE_REF_REGEX)
    status: CollectionStatus
    error_code: str = constrained(minLength=1)
    message: str
    report_path: str | None = constrained(pattern=JSON_POINTER_REGEX)

    def __post_init__(self) -> None:
        validate_model(self)
        if self.status is CollectionStatus.COLLECTED:
            raise InvalidModelError("a CollectionError can't have status COLLECTED")


@dataclass(frozen=True, slots=True, kw_only=True)
class UnsupportedCapability:
    capability: str = constrained(pattern=CAPABILITY_REGEX)
    status: CollectionStatus
    device_ref: str | None = constrained(pattern=DEVICE_REF_REGEX)
    reason: str = constrained(minLength=1)

    def __post_init__(self) -> None:
        validate_model(self)
        if self.status is CollectionStatus.COLLECTED:
            raise InvalidModelError("an UnsupportedCapability can't have status COLLECTED")


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectorRun:
    collector: str = constrained(minLength=1)
    status: CollectorRunStatus
    duration_ms: int = constrained(minimum=0)

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectorOutcome(Generic[SectionT]):
    """One collector's section model, run record, the errors it reported and the
    capabilities it found unsupported."""

    section: SectionT
    run: CollectorRun
    errors: tuple[CollectionError, ...] = ()
    unsupported: tuple[UnsupportedCapability, ...] = ()

    def __post_init__(self) -> None:
        for name in ("errors", "unsupported"):
            if isinstance(getattr(self, name), list):
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.run, CollectorRun):
            raise InvalidModelError("CollectorOutcome.run must be a CollectorRun")
        if not all(isinstance(e, CollectionError) for e in self.errors):
            raise InvalidModelError("CollectorOutcome.errors must contain CollectionError items")
        if not all(isinstance(u, UnsupportedCapability) for u in self.unsupported):
            raise InvalidModelError(
                "CollectorOutcome.unsupported must contain UnsupportedCapability items"
            )
        if self.run.status is CollectorRunStatus.COMPLETED and self.errors:
            raise InvalidModelError("a COMPLETED run can't report collection errors")
        if self.run.status is CollectorRunStatus.FAILED and not self.errors:
            raise InvalidModelError("a FAILED run must report at least one collection error")
        if any(e.collector != self.run.collector for e in self.errors):
            raise InvalidModelError("collection errors must belong to the outcome's collector")
