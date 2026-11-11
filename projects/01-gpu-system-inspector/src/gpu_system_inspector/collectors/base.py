"""The collector contract (architecture §4.2, DEC-44).

Every collector, production or test double, has a `name` (its `collector_runs` and
`CollectionError.collector` key), gets its adapters and `Clock` in the constructor, and
offers:

- `collect()`: the section plus its run record, errors and unsupported capabilities, as a
  `CollectorOutcome`. A collector reports `COMPLETED` or `PARTIAL`, never `FAILED`, and lets
  every unexpected exception (including `AdapterUsageError`) propagate.
- `unavailable(error)`: the section the core's guard substitutes when `collect()` raised.
  It takes an `ErrorDetail`, because the fallback section's `Observed.error` can only hold
  one.

Only the core's guard (`core.guard.run_guarded`) produces `FAILED` runs.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from gpu_system_inspector.models import CollectorOutcome, ErrorDetail

SectionT = TypeVar("SectionT")


@runtime_checkable
class Collector(Protocol[SectionT]):
    name: str

    def collect(self) -> CollectorOutcome[SectionT]: ...
    def unavailable(self, error: ErrorDetail) -> SectionT: ...
