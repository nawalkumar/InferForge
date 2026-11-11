"""Shared pytest fixtures. Builders and FX fixtures live in the `gsi_testkit` package."""

from __future__ import annotations

import pytest
from gsi_testkit.builders import Builders


@pytest.fixture
def build() -> type[Builders]:
    return Builders
