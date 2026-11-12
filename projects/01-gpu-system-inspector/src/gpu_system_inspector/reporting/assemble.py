"""Report assembly (architecture §4.10, IMP-P01): snapshot + metadata → `InspectionReport`.

The core aggregates every collector's section, runs, errors and unsupported capabilities
into an `InspectionSnapshot`; this module adds the report metadata (time, duration,
provenance, options, limitations), the diagnostics findings and the readiness result, and
enforces the aggregation rules the models can't see on their own: each collector runs at
most once, and each capability record has one owner, so a `(capability, device_ref)` pair
appears at most once. `errors` are the diagnostics engine's `diagnostics.Dxx` errors; they
join the snapshot's collection errors.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from datetime import datetime

from gpu_system_inspector.models import (
    SNAPSHOT_LIMITATION,
    CollectionError,
    DataOrigin,
    DiagnosticFinding,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
    InvalidModelError,
    ReadinessResult,
)


def assemble_report(
    snapshot: InspectionSnapshot,
    *,
    options: InspectionOptions,
    generated_at: datetime,
    duration_ms: int,
    data_origin: DataOrigin,
    limitations: Iterable[str] = (),
    diagnostics: Iterable[DiagnosticFinding] = (),
    readiness: ReadinessResult | None = None,
    errors: Iterable[CollectionError] = (),
) -> InspectionReport:
    runs = [r.collector for r in snapshot.collector_runs]
    if len(set(runs)) != len(runs):
        raise InvalidModelError(f"a collector ran more than once: {runs}")
    keys = [(u.capability, u.device_ref) for u in snapshot.unsupported_capabilities]
    if len(set(keys)) != len(keys):
        raise InvalidModelError(f"an unsupported capability has more than one owner: {keys}")
    extra = tuple(errors)
    if extra:
        snapshot = dataclasses.replace(
            snapshot, collection_errors=(*snapshot.collection_errors, *extra)
        )
    stated = [SNAPSHOT_LIMITATION]
    for text in limitations:
        if text not in stated:
            stated.append(text)
    return InspectionReport.from_snapshot(
        snapshot,
        generated_at=generated_at,
        duration_ms=duration_ms,
        data_origin=data_origin,
        options=options,
        diagnostics=tuple(diagnostics),
        readiness=readiness,
        limitations=tuple(stated),
    )
