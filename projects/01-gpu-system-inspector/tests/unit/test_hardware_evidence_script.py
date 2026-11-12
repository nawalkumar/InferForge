"""The hardware evidence script (`validation/hardware_evidence.py`, a repository script).

Its classification and verdict are checked on fixture reports: a FIXTURE report is never
evidence, but it pins how each collection status is reported. The end-to-end run on this
host is a `host` test. The script isn't in the wheel, so these tests skip outside the
repository checkout (like the spec-table tests, KI-23).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.core import Inspector
from gpu_system_inspector.models import InspectionOptions
from gpu_system_inspector.reporting import render_json

SCRIPT = Path(__file__).resolve().parents[2] / "validation" / "hardware_evidence.py"
if not SCRIPT.exists():
    pytest.skip("validation/ is a repository script, not in the wheel", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("hardware_evidence", SCRIPT)
evidence = importlib.util.module_from_spec(_spec)
sys.modules["hardware_evidence"] = evidence
_spec.loader.exec_module(evidence)


def fixture_report(fixture_id: str, required_vram_gib: float | None = None) -> dict:
    adapters = FakeAdapterSet.from_scenario(FIXTURES_BY_ID[fixture_id]).adapter_set()
    report = Inspector(adapters).run(InspectionOptions(required_vram_gib=required_vram_gib))
    return json.loads(render_json(report))


def features(report: dict) -> dict[str, dict]:
    return {row["feature"]: row for row in evidence.feature_table(report)}


@pytest.mark.parametrize(
    ("status", "code", "category"),
    [
        ("COLLECTED", None, evidence.OBSERVED),
        ("NOT_DETECTED", "NVML_ERROR_LIBRARY_NOT_FOUND", evidence.UNAVAILABLE),
        ("NOT_INSTALLED", "PackageNotFoundError", evidence.UNAVAILABLE),
        ("PERMISSION_DENIED", "NVML_ERROR_NO_PERMISSION", evidence.UNAVAILABLE),
        ("NOT_SUPPORTED", "PLATFORM_NOT_SUPPORTED", evidence.NOT_SUPPORTED),
        ("NOT_SUPPORTED", "NOT_IMPLEMENTED", evidence.NOT_IMPLEMENTED),
        ("COLLECTION_FAILED", "IMPORT_ERROR", evidence.FAILED),
        ("UNKNOWN", None, evidence.FAILED),
        ("SKIPPED", None, evidence.SKIPPED),
        ("NOT_APPLICABLE", None, evidence.NOT_APPLICABLE),
    ],
)
def test_every_collection_status_has_one_category(status, code, category):
    error = {"code": code, "message": "m"} if code else None
    assert evidence.classify({"status": status, "error": error}) == category


def test_a_working_gpu_is_observed_everywhere_it_matters():
    rows = features(fixture_report("FX01"))
    for name in (
        "GPU model and count",
        "NVML initialisation",
        "GPU UUID",
        "Driver version",
        "PyTorch CUDA availability",
        "Device correlation",
    ):
        assert rows[name]["categories"] == [evidence.OBSERVED], name


def test_no_gpu_is_unavailable_never_observed_and_blocks_the_verdict():
    report = fixture_report("FX03")
    rows = features(report)
    assert rows["GPU model and count"]["categories"] == [evidence.UNAVAILABLE]
    assert rows["GPU UUID"]["categories"] == [evidence.NO_DATA]
    assert evidence.verdict(report).startswith("BLOCKED")


def test_macos_is_not_supported_rather_than_unavailable():
    rows = features(fixture_report("FX13"))
    assert rows["NVML initialisation"]["categories"] == [evidence.NOT_SUPPORTED]


def test_a_broken_pytorch_is_a_failure():
    rows = features(fixture_report("FX15"))
    assert evidence.FAILED in rows["PyTorch installation and build"]["categories"]


def test_the_verdict_never_claims_hardware_validation():
    for fixture_id in ("FX01", "FX06", "FX11"):
        text = evidence.verdict(fixture_report(fixture_id))
        assert text.startswith("EVIDENCE_COLLECTED") and "owner review" in text
    assert evidence.verdict(None).startswith("BLOCKED")


def test_resolve_expands_list_indices():
    data = {"a": [{"b": 1}, {"b": 2}], "c": {"d": 3}}
    assert evidence.resolve(data, "/a/*/b") == [("/a/0/b", 1), ("/a/1/b", 2)]
    assert evidence.resolve(data, "/c/d") == [("/c/d", 3)]
    assert evidence.resolve(data, "/missing/*") == []


def test_junit_outcomes_and_properties_are_read(tmp_path):
    junit = tmp_path / "r.xml"
    junit.write_text(
        '<testsuites><testsuite name="s">'
        '<testcase name="test_hw01_x"><properties><property name="rows" value="[]"/>'
        "</properties></testcase>"
        '<testcase name="test_hw02_x"><skipped message="NVML unavailable"/></testcase>'
        '<testcase name="test_hw03_x"><failure message="assert 1 == 2"/></testcase>'
        "</testsuite></testsuites>"
    )
    results = evidence.parse_junit(junit)
    assert [(r["test"], r["outcome"], r["message"]) for r in results] == [
        ("test_hw01_x", "passed", None),
        ("test_hw02_x", "skipped", "NVML unavailable"),
        ("test_hw03_x", "failed", "assert 1 == 2"),
    ]
    assert results[0]["rows"] == "[]"


def test_the_summary_lists_unimplemented_features_and_the_sharing_note():
    report = fixture_report("FX01", 8)
    summary = {
        "environment": {
            "label": "fixture",
            "collected_at": "t",
            "platform": "p",
            "machine": "m",
            "google_colab": False,
            "python": "3",
            "implementation": "CPython",
            "distributions": {"torch": None},
        },
        "verdict": evidence.verdict(report),
        "inspector_runs": {},
        "features": evidence.feature_table(report),
        "readiness": report["readiness"],
        "findings": [
            {k: f[k] for k in ("check_id", "device_ref", "status", "confidence")}
            for f in report["diagnostics"]
        ],
        "hardware_tests": {"results": []},
        "reference_tools": {},
        "cuda_smoke_test": {"outcome": "not run", "reason": "opt-in"},
    }
    text = evidence.summary_markdown(summary)
    assert "Graphics process listing" in text and "KI-36" in text
    assert "A skipped hardware test is not a pass." in text
    assert "not part of the inspection" in text
    assert "Required 8.0 GiB: **READY**" in text


@pytest.mark.host
def test_the_script_writes_a_bundle_on_this_host(tmp_path):
    out = tmp_path / "bundle"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--skip-hardware-tests"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text())
    mode_a = json.loads((out / "inspect-modeA.json").read_text())
    assert mode_a["data_origin"] == "LIVE"
    assert summary["verdict"] == evidence.verdict(mode_a)
    assert summary["cuda_smoke_test"]["outcome"] == "not run"
    assert all(r["schema"] == "valid" for r in summary["inspector_runs"].values())
    redacted = json.loads((out / "inspect-modeA-redacted.json").read_text())
    assert redacted["redaction"]["applied"] is True
    assert str(Path.home()) not in (out / "inspect-modeA-redacted.json").read_text()
    assert (out / "SUMMARY.md").read_text().startswith("# GPU System Inspector")
