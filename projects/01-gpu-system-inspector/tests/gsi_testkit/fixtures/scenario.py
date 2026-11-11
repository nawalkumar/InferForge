"""The FX fixture record (testing strategy §4)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from gpu_system_inspector.models import (
    SNAPSHOT_LIMITATION,
    DataOrigin,
    InspectionOptions,
    InspectionReport,
    InspectionSnapshot,
)
from gsi_testkit.builders import FIXED_TIME
from gsi_testkit.expectations import (
    ExpectedFinding,
    ReadinessRun,
    materialize_finding,
    materialize_readiness,
    replace_checks,
)


@dataclass(frozen=True)
class Premise:
    """A fact about the fixture inputs that the expected outcome depends on."""

    description: str
    holds: Callable[[InspectionSnapshot], bool]


@dataclass(frozen=True)
class FixtureScenario:
    fixture_id: str
    name: str
    description: str
    snapshot: InspectionSnapshot
    expected_diagnostics: tuple[ExpectedFinding, ...]
    readiness_runs: tuple[ReadinessRun, ...] = ()
    mode_b_only: bool = False
    premises: tuple[Premise, ...] = ()
    limitations: tuple[str, ...] = ()
    collector_failures: Mapping[str, BaseException] = field(default_factory=dict)
    nvml_faults: Mapping[tuple[int, str], str] = field(default_factory=dict)
    # Device index → handles acquired before the GPU is lost (`FakeNvml`, KI-34).
    nvml_lost_after_handles: Mapping[int, int] = field(default_factory=dict)

    data_origin = DataOrigin.FIXTURE

    def __post_init__(self) -> None:
        if self.mode_b_only and not self.readiness_runs:
            raise ValueError(f"{self.fixture_id}: a Mode B fixture needs a readiness run")

    @property
    def device_refs(self) -> tuple[str, ...]:
        return self.snapshot.gpu.device_refs

    def report_cases(self) -> tuple[tuple[str, ReadinessRun | None], ...]:
        """(label, run) for every report this fixture defines: Mode A and/or Mode B runs."""
        mode_a = () if self.mode_b_only else ((self.fixture_id, None),)
        return mode_a + tuple((run.label, run) for run in self.readiness_runs)

    def options(self, run: ReadinessRun | None = None) -> InspectionOptions:
        return InspectionOptions(required_vram_gib=run.required_vram_gib if run else None)

    def expected_for(self, run: ReadinessRun | None = None) -> tuple[ExpectedFinding, ...]:
        """Mode A outcomes; in Mode B runs D02 comes from the run (the rest is unchanged)."""
        if run is None:
            if self.mode_b_only:
                raise ValueError(f"{self.fixture_id} is defined only in Mode B")
            return self.expected_diagnostics
        return replace_checks(self.expected_diagnostics, run.d02)

    def report(self, run: ReadinessRun | None = None) -> InspectionReport:
        return InspectionReport.from_snapshot(
            self.snapshot,
            generated_at=FIXED_TIME,
            duration_ms=500,
            data_origin=DataOrigin.FIXTURE,
            options=self.options(run),
            diagnostics=tuple(
                materialize_finding(e, self.fixture_id) for e in self.expected_for(run)
            ),
            readiness=materialize_readiness(run, self.snapshot) if run else None,
            limitations=(
                SNAPSHOT_LIMITATION,
                f"Synthetic fixture {self.fixture_id}; not a real measurement.",
                *self.limitations,
            ),
        )
