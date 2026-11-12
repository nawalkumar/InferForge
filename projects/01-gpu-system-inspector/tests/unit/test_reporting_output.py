"""IMP-P03 text renderer, IMP-P04 redaction, IMP-P05 atomic writer (DEC-50). Reports come
from fixture snapshots; the writer runs in `tmp_path` only."""

import dataclasses
import json
import os

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.core import Inspector
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    DataOrigin,
    InspectionOptions,
    Observed,
    RedactionCategory,
    Source,
)
from gpu_system_inspector.reporting import (
    FIXTURE_BANNER,
    OutputError,
    RedactionPolicy,
    Redactor,
    load_report_schema,
    redact,
    render_json,
    render_text,
    write_atomic,
)
from gpu_system_inspector.reporting import writer as writer_module
from gpu_system_inspector.reporting.text_report import ordered_findings, show

all_fixtures = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)


def _report(fixture_id, gib=None):
    fixture = FIXTURES_BY_ID[fixture_id]
    adapters = FakeAdapterSet.from_scenario(fixture)
    return Inspector(adapters.adapter_set()).run(InspectionOptions(required_vram_gib=gib))


# --- text ------------------------------------------------------------------------------


@all_fixtures
def test_text_is_plain_bounded_and_bannered(fixture):
    text = render_text(_report(fixture.fixture_id))
    lines = text.splitlines()
    assert lines[0] == lines[-1] == FIXTURE_BANNER
    assert "\x1b" not in text
    assert max(len(line) for line in lines) <= 100
    assert text.endswith("\n")


def test_a_live_report_has_no_banner():
    report = dataclasses.replace(_report("FX01"), data_origin=DataOrigin.LIVE)
    assert FIXTURE_BANNER not in render_text(report)


def test_missing_values_show_status_and_reason_never_a_number():
    missing = Observed.failed(
        CollectionStatus.NOT_SUPPORTED,
        code="NVML_ERROR_NOT_SUPPORTED",
        message="not on WSL2",
        unit="percent",
    )
    assert show(missing) == "—  NOT_SUPPORTED (NVML_ERROR_NOT_SUPPORTED: not on WSL2)"
    assert show(None) == "—  NOT_INSPECTED"
    free = Observed.collected(24 * 2**30, source=Source.NVML, method="t", unit="bytes")
    assert show(free, " free") == "24.0 GiB free (NVML)"


def test_cuda_versions_are_labelled_distinctly_and_memory_bus_named():
    text = render_text(_report("FX01"))
    assert "Driver supports CUDA up to 12.4 (NVML)" in text
    assert "CUDA Toolkit 12.4.131 (nvcc)" in text
    assert "PyTorch built with CUDA 12.4 (PYTORCH)" in text
    assert "Memory bus activity" in text


def test_mode_b_text_ends_readiness_with_the_disclaimer_and_orders_findings():
    report = _report("FX07", 8)
    text = render_text(report)
    block = text[text.index("READINESS  required free memory: 8.0 GiB") : text.index("FINDINGS")]
    assert "short by" in block and "overall: NOT_READY (NO_DEVICE_READY)" in block
    assert block.strip().splitlines()[-1].endswith("after this snapshot.")
    ranks = [f.status for f in ordered_findings(report.diagnostics)]
    order = ["FAIL", "WARNING", "UNKNOWN", "INFO", "PASS", "NOT_APPLICABLE"]
    assert [order.index(s) for s in ranks] == sorted(order.index(s) for s in ranks)
    assert text.index("[FAIL] D02") < text.index("[PASS] D01")


def test_collection_errors_and_unsupported_capabilities_are_listed():
    text = render_text(_report("FX10"))
    errors = text[text.index("COLLECTION ERRORS") : text.index("LIMITATIONS")]
    assert "cuda_environment" in errors
    wsl = render_text(_report("FX04b"))
    unsupported = wsl[wsl.index("UNSUPPORTED") : wsl.index("COLLECTION ERRORS")]
    assert "GPU.PCI_SCAN" in unsupported


# --- redaction ------------------------------------------------------------------------


POLICY = RedactionPolicy(user="alice", home="/home/alice")


@all_fixtures
def test_redacted_reports_stay_valid_and_keep_structure(fixture):
    report = _report(fixture.fixture_id)
    redacted = redact(report, POLICY)
    jsonschema.validate(json.loads(render_json(redacted)), load_report_schema())
    assert redacted.redaction.applied and redacted.options.redact
    assert set(redacted.redaction.categories) == set(RedactionCategory)
    assert [f.status for f in redacted.diagnostics] == [f.status for f in report.diagnostics]
    assert [r.status for r in redacted.collector_runs] == [r.status for r in report.collector_runs]
    assert redacted.gpu.device_count == report.gpu.device_count
    for d in redacted.gpu.devices:
        assert d.uuid.value is None or d.uuid.value.startswith("<redacted:uuid:")
        assert d.pci_bus_id.value is None or d.pci_bus_id.value.startswith("<redacted:pci:")


def test_one_device_keeps_one_ordinal_across_sources():
    redacted = redact(_report("FX06"), POLICY)
    by_ref = {d.device_ref: d for d in redacted.gpu.devices}
    for dev in redacted.frameworks.pytorch.devices:
        nvml = by_ref[dev.correlated_device_ref.value]
        assert dev.uuid.value == nvml.uuid.value
        assert dev.pci_bus_id.value == nvml.pci_bus_id.value
    addresses = {o.address for o in redacted.gpu.pci_observations.value}
    assert addresses == {d.pci_bus_id.value for d in redacted.gpu.devices}


def test_user_home_and_library_paths_are_redacted_in_free_text():
    report = _report("FX01")
    env = dict(report.cuda_environment.environment)
    env["LD_LIBRARY_PATH"] = Observed.collected(
        "/home/alice/lib:/opt/x/lib", source=Source.ENVIRONMENT, method="t"
    )
    report = dataclasses.replace(
        report, cuda_environment=dataclasses.replace(report.cuda_environment, environment=env)
    )
    redactor = Redactor.for_report(report, POLICY)
    assert redactor.text("/opt/x/lib/libfoo.so").startswith("<redacted:path:")
    assert redactor.text("see /home/alice/notes by alice") == "see ~/notes by <redacted:user>"
    assert redactor.text("alice-box and malice stay") == "alice-box and malice stay"
    redacted = redact(report, POLICY)
    text = render_text(redacted) + render_json(redacted)
    for secret in ("/home/alice", "/opt/x/lib", "/usr/local/cuda-12.4", "python ("):
        assert secret not in text


def test_categories_can_be_restricted():
    only_paths = RedactionPolicy(categories=(RedactionCategory.PATH,))
    redacted = redact(_report("FX01"), only_paths)
    assert redacted.gpu.devices[0].uuid.value.startswith("GPU-")
    assert redacted.system.python_executable.value.startswith("<redacted:path:")


# --- writer ---------------------------------------------------------------------------


def test_write_atomic_replaces_an_existing_file(tmp_path):
    target = tmp_path / "r.json"
    target.write_text("old", encoding="utf-8")
    write_atomic("new\n", target)
    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.iterdir()) == [target]
    assert oct(target.stat().st_mode & 0o777) != oct(0o600)


@pytest.mark.parametrize("exc", [OSError(28, "No space left on device"), KeyboardInterrupt()])
def test_a_failed_rename_removes_the_temporary_file(tmp_path, monkeypatch, exc):
    target = tmp_path / "r.json"
    target.write_text("old", encoding="utf-8")

    def fail(src, dst):
        raise exc

    monkeypatch.setattr(writer_module.os, "replace", fail)
    expected = OutputError if isinstance(exc, OSError) else KeyboardInterrupt
    with pytest.raises(expected) as info:
        write_atomic("new", target)
    if expected is OutputError:
        assert info.value.reason == "No space left on device" and info.value.path == target
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_unwritable_targets_raise_output_error(tmp_path):
    with pytest.raises(OutputError, match="Is a directory"):
        write_atomic("x", tmp_path)
    with pytest.raises(OutputError):
        write_atomic("x", tmp_path / "missing" / "r.json")
    assert os.listdir(tmp_path) == []


def test_sensitive_literals_inside_warnings_and_errors_are_redacted():
    """AC40: values inside free text (PyTorch warnings, error messages) are redacted too, with
    the same placeholder as the field they come from."""
    report = _report("FX06")
    uuid = report.gpu.devices[0].uuid.value
    pytorch = dataclasses.replace(
        report.frameworks.pytorch,
        cuda_init_warnings=(f"device {uuid} at /home/alice/.cache failed for alice",),
    )
    error = CollectionError(
        collector="runtime",
        device_ref=None,
        status=CollectionStatus.COLLECTION_FAILED,
        error_code="X",
        message=f"cannot open /usr/bin/nvidia-smi for {uuid}",
        report_path=None,
    )
    report = dataclasses.replace(
        report,
        frameworks=dataclasses.replace(report.frameworks, pytorch=pytorch),
        collection_errors=(error,),
    )
    redacted = redact(report, POLICY)
    placeholder = redacted.gpu.devices[0].uuid.value
    (warning,) = redacted.frameworks.pytorch.cuda_init_warnings
    assert warning == f"device {placeholder} at ~/.cache failed for <redacted:user>"
    message = redacted.collection_errors[0].message
    assert uuid not in message and "/usr/bin/nvidia-smi" not in message
    assert placeholder in message
    text = render_text(redacted) + render_json(redacted)
    assert uuid not in text and "/home/alice" not in text


def test_without_redaction_values_are_unchanged():
    report = _report("FX06")
    assert not report.redaction.applied
    assert report.gpu.devices[0].uuid.value.startswith("GPU-")
