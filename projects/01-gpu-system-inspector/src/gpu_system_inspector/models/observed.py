"""The value/status envelope for collected values (docs/data_model_specification.md §3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from gpu_system_inspector.models.enums import CollectionStatus, Source

T = TypeVar("T")

UNITS: tuple[str, ...] = ("bytes", "celsius", "watts", "percent", "ms")

# Every synthetic (DataOrigin.FIXTURE) value's `method` starts with this prefix, so
# synthetic values can never pass for live measurements.
FIXTURE_METHOD_PREFIX = "fixture:"

ERROR_REQUIRED_STATUSES: frozenset[CollectionStatus] = frozenset(
    {
        CollectionStatus.NOT_INSTALLED,
        CollectionStatus.NOT_DETECTED,
        CollectionStatus.NOT_SUPPORTED,
        CollectionStatus.PERMISSION_DENIED,
        CollectionStatus.COLLECTION_FAILED,
        CollectionStatus.UNKNOWN,
    }
)


class InvalidModelError(ValueError):
    """A model was given values that violate the data model specification."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorDetail:
    code: str = field(metadata={"schema": {"minLength": 1}})
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise InvalidModelError("ErrorDetail.code must be a non-empty string")
        if not isinstance(self.message, str):
            raise InvalidModelError("ErrorDetail.message must be a string")


@dataclass(frozen=True, slots=True, kw_only=True)
class Observed(Generic[T]):
    """A collected value with its provenance.

    `value` is non-null if and only if `status` is COLLECTED, so a missing measurement can
    never masquerade as a real zero.
    """

    value: T | None
    status: CollectionStatus
    source: Source | None
    method: str | None
    unit: str | None = None
    error: ErrorDetail | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CollectionStatus):
            raise InvalidModelError(
                f"Observed.status must be a CollectionStatus, got {self.status!r}"
            )
        if self.source is not None and not isinstance(self.source, Source):
            raise InvalidModelError(
                f"Observed.source must be a Source or None, got {self.source!r}"
            )
        for name in ("method", "note"):
            text = getattr(self, name)
            if text is not None and not isinstance(text, str):
                raise InvalidModelError(f"Observed.{name} must be a string or None")
        if self.unit is not None and self.unit not in UNITS:
            raise InvalidModelError(
                f"Observed.unit must be one of {UNITS} or None, got {self.unit!r}"
            )
        if self.error is not None and not isinstance(self.error, ErrorDetail):
            raise InvalidModelError("Observed.error must be an ErrorDetail or None")
        if isinstance(self.value, list):
            object.__setattr__(self, "value", tuple(self.value))
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise InvalidModelError("Observed.value must be a finite number")

        if self.status is CollectionStatus.COLLECTED:
            if self.value is None:
                raise InvalidModelError("a COLLECTED value must not be None")
            if self.error is not None:
                raise InvalidModelError("a COLLECTED value must not carry an error")
            if self.source is None or not self.method:
                raise InvalidModelError("a COLLECTED value must name its source and method")
            return
        if self.value is not None:
            raise InvalidModelError(
                f"value must be None when status is {self.status}; missing data is never a value"
            )
        if self.status in ERROR_REQUIRED_STATUSES and self.error is None:
            raise InvalidModelError(f"status {self.status} requires an error")
        if self.status is CollectionStatus.SKIPPED and self.error is not None:
            raise InvalidModelError("a SKIPPED value must not carry an error")
        if self.status is CollectionStatus.NOT_APPLICABLE and not self.note:
            raise InvalidModelError("a NOT_APPLICABLE value must explain why in note")

    @property
    def is_collected(self) -> bool:
        return self.status is CollectionStatus.COLLECTED

    @classmethod
    def collected(
        cls,
        value: T,
        *,
        source: Source,
        method: str,
        unit: str | None = None,
        note: str | None = None,
    ) -> Observed[T]:
        return cls(
            value=value,
            status=CollectionStatus.COLLECTED,
            source=source,
            method=method,
            unit=unit,
            note=note,
        )

    @classmethod
    def failed(
        cls,
        status: CollectionStatus,
        *,
        code: str,
        message: str,
        source: Source | None = None,
        method: str | None = None,
        unit: str | None = None,
        note: str | None = None,
    ) -> Observed[T]:
        return cls(
            value=None,
            status=status,
            source=source,
            method=method,
            unit=unit,
            error=ErrorDetail(code=code, message=message),
            note=note,
        )

    @classmethod
    def skipped(
        cls,
        *,
        unit: str | None = None,
        note: str | None = None,
        source: Source | None = None,
        method: str | None = None,
    ) -> Observed[T]:
        return cls(
            value=None,
            status=CollectionStatus.SKIPPED,
            source=source,
            method=method,
            unit=unit,
            note=note,
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        note: str,
        unit: str | None = None,
        source: Source | None = None,
        method: str | None = None,
        error: ErrorDetail | None = None,
    ) -> Observed[T]:
        return cls(
            value=None,
            status=CollectionStatus.NOT_APPLICABLE,
            source=source,
            method=method,
            unit=unit,
            error=error,
            note=note,
        )
