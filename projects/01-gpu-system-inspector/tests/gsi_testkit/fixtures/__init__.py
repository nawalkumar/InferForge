"""The 20 FX fixtures (24 variants) from testing strategy §4."""

from __future__ import annotations

from gsi_testkit.fixtures.catalog_core import CORE_FIXTURES
from gsi_testkit.fixtures.catalog_platform import PLATFORM_FIXTURES
from gsi_testkit.fixtures.scenario import FixtureScenario, Premise

ALL_FIXTURES: tuple[FixtureScenario, ...] = tuple(
    factory() for factory in (*CORE_FIXTURES, *PLATFORM_FIXTURES)
)
FIXTURES_BY_ID: dict[str, FixtureScenario] = {f.fixture_id: f for f in ALL_FIXTURES}


def report_cases() -> list[tuple[FixtureScenario, str, object]]:
    """Every (fixture, label, run) report case: Mode A reports plus every Mode B run."""
    return [(f, label, run) for f in ALL_FIXTURES for label, run in f.report_cases()]


__all__ = ["ALL_FIXTURES", "FIXTURES_BY_ID", "FixtureScenario", "Premise", "report_cases"]
