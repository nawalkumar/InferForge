"""System collector (architecture §4.3): the `SystemInfo` section and its run record.

The `PlatformAdapter` does all OS access and per-field status mapping; this collector adds
the bookkeeping the report needs: run status, duration (from the injected `Clock`) and the
capabilities that are unsupported on this platform. Field-level failures stay inside their
`Observed` values and don't create `CollectionError` records (data model §3.4).
"""

from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

from gpu_system_inspector.adapters import Clock, PlatformAdapter
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    ErrorDetail,
    Observed,
    SystemInfo,
    UnsupportedCapability,
)

SYSTEM_CAPABILITIES: dict[str, str] = {
    "os_distribution": "SYSTEM.OS_DISTRIBUTION",
    "processor_model": "SYSTEM.PROCESSOR_MODEL",
    "logical_cpu_count": "SYSTEM.LOGICAL_CPU_COUNT",
    "memory_total_bytes": "SYSTEM.MEMORY_TOTAL",
    "memory_available_bytes": "SYSTEM.MEMORY_AVAILABLE",
}
# Statuses that make the run PARTIAL. NOT_DETECTED / NOT_SUPPORTED describe the host, not a
# collection problem.
FAILURE_STATUSES = frozenset(
    {
        CollectionStatus.COLLECTION_FAILED,
        CollectionStatus.PERMISSION_DENIED,
        CollectionStatus.UNKNOWN,
    }
)


class SystemCollector:
    name: ClassVar[str] = "system"

    def __init__(self, platform: PlatformAdapter, clock: Clock) -> None:
        self._platform = platform
        self._clock = clock

    def collect(self) -> CollectorOutcome[SystemInfo]:
        started = self._clock.monotonic_ms()
        info = self._platform.system_info()
        values = {
            f.name: getattr(info, f.name)
            for f in fields(info)
            if isinstance(getattr(info, f.name), Observed)
        }
        unsupported = tuple(
            UnsupportedCapability(
                capability=SYSTEM_CAPABILITIES.get(name, f"SYSTEM.{name.upper()}"),
                status=value.status,
                device_ref=None,
                reason=value.error.message or value.error.code,
            )
            for name, value in values.items()
            if value.status is CollectionStatus.NOT_SUPPORTED and value.error is not None
        )
        partial = any(value.status in FAILURE_STATUSES for value in values.values())
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=info, run=run, unsupported=unsupported)

    def unavailable(self, error: ErrorDetail) -> SystemInfo:
        """The section the collector guard substitutes when `collect()` raised."""
        return SystemInfo.unavailable(
            CollectionStatus.COLLECTION_FAILED, error, platform_profile=self._platform.profile
        )
