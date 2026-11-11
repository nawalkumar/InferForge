"""The collector guard (data flow §3, DEC-44): the only place that produces `FAILED` runs.

`run_guarded` calls `collector.collect()` and returns its outcome unchanged when it honours
the collector contract (`collectors.base`). Any exception, including `AdapterUsageError`
(a caller bug) and a collector that breaks the contract (wrong outcome type, another
collector's name, a self-reported `FAILED`), is contained: the section becomes
`collector.unavailable(...)`, the run `FAILED` with the guard's own timing, and exactly one
host-wide `UNHANDLED_EXCEPTION` `CollectionError` names the collector. `BaseException`s that
aren't `Exception` (`KeyboardInterrupt`, `SystemExit`) propagate.
"""

from __future__ import annotations

from typing import TypeVar

from gpu_system_inspector.adapters import AdapterUsageError, Clock
from gpu_system_inspector.adapters.outcomes import UNHANDLED_EXCEPTION
from gpu_system_inspector.collectors import Collector
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    ErrorDetail,
)

SectionT = TypeVar("SectionT")


def run_guarded(
    collector: Collector[SectionT], clock: Clock, *, report_path: str
) -> CollectorOutcome[SectionT]:
    started = clock.monotonic_ms()
    try:
        outcome = collector.collect()
        _check_contract(collector, outcome)
        return outcome
    except Exception as exc:
        detail = ErrorDetail(code=UNHANDLED_EXCEPTION, message=describe(exc))
        error = CollectionError(
            collector=collector.name,
            device_ref=None,
            status=CollectionStatus.COLLECTION_FAILED,
            error_code=detail.code,
            message=detail.message,
            report_path=report_path,
        )
        run = CollectorRun(
            collector=collector.name,
            status=CollectorRunStatus.FAILED,
            duration_ms=max(0, clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=collector.unavailable(detail), run=run, errors=(error,))


def describe(exc: BaseException) -> str:
    """`Type: message`, or just the type when the exception has no message."""
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _check_contract(collector: Collector[SectionT], outcome: object) -> None:
    if not isinstance(outcome, CollectorOutcome):
        raise AdapterUsageError(
            f"collector {collector.name!r} returned {type(outcome).__name__}, "
            "not a CollectorOutcome"
        )
    if outcome.run.collector != collector.name:
        raise AdapterUsageError(
            f"collector {collector.name!r} reported a run for {outcome.run.collector!r}"
        )
    if outcome.run.status is CollectorRunStatus.FAILED:
        raise AdapterUsageError(f"collector {collector.name!r} reported FAILED; only the guard may")
