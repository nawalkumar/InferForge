"""Report assembly (architecture §4.10, IMP-P01): snapshot + metadata → `InspectionReport`.

The core aggregates every collector's section, runs, errors and unsupported capabilities
into an `InspectionSnapshot`; this module adds the report metadata (time, duration,
provenance, options, limitations) and enforces the aggregation rules the models can't see
on their own: each collector runs at most once, and each capability record has one owner,
so a `(capability, device_ref)` pair appears at most once. Diagnostics (IMP-D*) and readiness
(IMP-R01) aren't implemented yet, so a report carries no findings and Mode A only.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from gpu_system_inspector.models import (
    SNAPSHOT_LIMITATION,
    DataOrigin,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
    InvalidModelError,
)


def assemble_report(
    snapshot: InspectionSnapshot,
    *,
    options: InspectionOptions,
    generated_at: datetime,
    duration_ms: int,
    data_origin: DataOrigin,
    limitations: Iterable[str] = (),
) -> InspectionReport:
    runs = [r.collector for r in snapshot.collector_runs]
    if len(set(runs)) != len(runs):
        raise InvalidModelError(f"a collector ran more than once: {runs}")
    keys = [(u.capability, u.device_ref) for u in snapshot.unsupported_capabilities]
    if len(set(keys)) != len(keys):
        raise InvalidModelError(f"an unsupported capability has more than one owner: {keys}")
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
        limitations=tuple(stated),
    )
