"""The example reports in `examples/reports/` stay valid and honestly labelled.

`live-*` files are real runs on the development machine (redacted); `fixture-*` files come
from `examples/synthetic_showcase.py` and must say so. The examples aren't in the wheel, so
these tests skip outside the repository checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from gpu_system_inspector.models import DataOrigin
from gpu_system_inspector.reporting import load_report_schema, parse_report_json
from gpu_system_inspector.reporting.text_report import FIXTURE_BANNER

REPORTS = Path(__file__).resolve().parents[2] / "examples" / "reports"
if not REPORTS.is_dir():
    pytest.skip("examples/ isn't in the wheel", allow_module_level=True)

JSON_FILES = sorted(REPORTS.glob("*.json"))
TEXT_FILES = sorted(REPORTS.glob("*.txt"))


def test_there_are_live_and_fixture_examples():
    names = [p.name for p in JSON_FILES + TEXT_FILES]
    assert any(n.startswith("live-") for n in names)
    assert any(n.startswith("fixture-") for n in names)
    assert all(n.startswith(("live-", "fixture-")) for n in names)


@pytest.mark.parametrize("path", JSON_FILES, ids=lambda p: p.name)
def test_json_examples_validate_and_declare_their_origin(path):
    text = path.read_text(encoding="utf-8")
    jsonschema.validate(json.loads(text), load_report_schema())
    report = parse_report_json(text)
    expected = DataOrigin.LIVE if path.name.startswith("live-") else DataOrigin.FIXTURE
    assert report.data_origin is expected
    if path.name.startswith("live-"):
        assert report.redaction.applied, "live examples are published redacted"


@pytest.mark.parametrize("path", TEXT_FILES, ids=lambda p: p.name)
def test_text_examples_are_labelled(path):
    lines = path.read_text(encoding="utf-8").strip("\n").splitlines()
    if path.name.startswith("fixture-"):
        assert lines[0] == FIXTURE_BANNER and lines[-1] == FIXTURE_BANNER
    else:
        assert FIXTURE_BANNER not in lines
        assert "data origin LIVE" in lines[1]
        assert lines[2].startswith("Redacted: ")


@pytest.mark.parametrize("path", JSON_FILES + TEXT_FILES, ids=lambda p: p.name)
def test_live_examples_carry_no_home_directory(path):
    assert "/home/" not in path.read_text(encoding="utf-8")
