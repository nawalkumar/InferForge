"""Normalised adapter outcomes follow data model §3.3, and the probe decoder enforces the
probe protocol (data flow §5)."""

import json
from importlib.metadata import PackageNotFoundError

import pytest
from gsi_testkit import spec_tables
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    CommandResult,
    decode_probe_output,
    exception_outcome,
    nvml_error_status,
    require_success,
)
from gpu_system_inspector.adapters.outcomes import ERROR_EXCERPT_CHARS, excerpt
from gpu_system_inspector.models import (
    PROBE_PROTOCOL_VERSION,
    CollectionStatus,
    Observed,
    Source,
)

DOCS_MISSING = pytest.mark.skipif(
    not spec_tables.DOCS_DIR.is_dir(),
    reason="specification documents are not available (e.g. isolated wheel test run)",
)
S = CollectionStatus
METHOD = "fixture:test"


@DOCS_MISSING
def test_nvml_error_mapping_matches_the_normative_table():
    rows = [r for r in spec_tables.error_mapping_table() if r.origin == "NVML"]
    listed = [r for r in rows if r.code.startswith("NVML_ERROR_")]
    assert len(listed) == 5
    for row in listed:
        assert nvml_error_status(row.code) is S[row.status], row
    (other,) = [r for r in rows if r.code == "NVML error name"]
    assert other.status == "COLLECTION_FAILED"
    for unlisted in ("NVML_ERROR_GPU_IS_LOST", "NVML_ERROR_TIMEOUT", "NVML_ERROR_UNKNOWN"):
        assert nvml_error_status(unlisted) is S.COLLECTION_FAILED


EXCEPTIONS = {
    "ImportError": ImportError("No module named 'pynvml'"),
    "PackageNotFoundError": PackageNotFoundError("torch"),
    "FileNotFoundError": FileNotFoundError("nvidia-smi"),
    "PermissionError": PermissionError("denied"),
}


@DOCS_MISSING
def test_exception_mapping_matches_the_normative_table():
    rows = spec_tables.error_mapping_table()
    checked = 0
    for row in rows:
        if row.code in EXCEPTIONS:
            outcome = exception_outcome(EXCEPTIONS[row.code], source=Source.OS, method=METHOD)
            assert (outcome.status, outcome.error.code) == (S[row.status], row.code), row
            checked += 1
        elif "OSError" in row.condition:
            outcome = exception_outcome(OSError(8, "Exec format error"), source=None, method=None)
            assert (outcome.status, outcome.error.code) == (S[row.status], "OSError")
            checked += 1
        elif row.origin == "Anything else":
            outcome = exception_outcome(ValueError("odd"), source=None, method=None)
            assert (outcome.status, outcome.error.code) == (S[row.status], "ValueError")
            checked += 1
    assert checked == 8  # 2 import rows, 2 subprocess, 2 files, OSError, anything else


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (ModuleNotFoundError("No module named 'pynvml'"), S.NOT_INSTALLED, "ImportError"),
        (PackageNotFoundError("torch"), S.NOT_INSTALLED, "PackageNotFoundError"),
        (FileNotFoundError("missing"), S.NOT_DETECTED, "FileNotFoundError"),
        (PermissionError("denied"), S.PERMISSION_DENIED, "PermissionError"),
        (TimeoutError("slow"), S.COLLECTION_FAILED, "TIMEOUT"),
        (IsADirectoryError("dir"), S.COLLECTION_FAILED, "IsADirectoryError"),
        (KeyError("x"), S.UNKNOWN, "KeyError"),
    ],
)
def test_exception_outcome_keeps_the_four_failure_kinds_apart(exc, status, code):
    """Unsupported ≠ not installed ≠ installed but unusable ≠ failed while running."""
    outcome = exception_outcome(exc, source=Source.OS, method=METHOD, unit="bytes")
    assert (outcome.status, outcome.error.code) == (status, code)
    assert outcome.value is None and outcome.unit == "bytes" and outcome.method == METHOD


def test_exception_outcome_message_falls_back_to_the_type_name():
    assert exception_outcome(OSError(), source=None, method=None).error.message == "OSError"


def test_nvml_error_status_rejects_non_nvml_codes():
    with pytest.raises(AdapterUsageError):
        nvml_error_status("FileNotFoundError")


def _result(returncode=0, stdout="", stderr="") -> Observed[CommandResult]:
    result = CommandResult(
        argv=("tool",), returncode=returncode, stdout=stdout, stderr=stderr, duration_ms=3
    )
    return Observed.collected(result, source=Source.NVIDIA_SMI, method=METHOD)


def test_require_success_maps_non_zero_exit_to_exit_code():
    failed = require_success(_result(9, stderr="NVIDIA-SMI has failed\n"))
    assert failed.status is S.COLLECTION_FAILED and failed.error.code == "EXIT_9"
    assert failed.error.message == "NVIDIA-SMI has failed"
    assert failed.source is Source.NVIDIA_SMI and failed.method == METHOD
    assert require_success(_result(4)).error.message == "exited with status 4"


def test_require_success_passes_success_and_launch_failures_through():
    ok = _result(0, stdout="x")
    assert require_success(ok) is ok
    missing = exception_outcome(FileNotFoundError("tool"), source=Source.OS, method=METHOD)
    assert require_success(missing) is missing


def test_excerpt_keeps_only_the_tail():
    assert excerpt("a" * 10 + "b" * ERROR_EXCERPT_CHARS + "\n") == "b" * ERROR_EXCERPT_CHARS


# --- probe protocol ---------------------------------------------------------------------


def _payload() -> dict:
    probe = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX06"]).torch_fake
    return json.loads(probe.subprocess_output(60.0).value.stdout)


def _decode(stdout: str, returncode: int = 0, stderr: str = "") -> Observed:
    return decode_probe_output(_result(returncode, stdout, stderr))


def test_decoder_accepts_the_documented_payload():
    report = _decode(json.dumps(_payload()))
    assert report.is_collected and report.method == METHOD
    assert report.value.torch_version == "2.4.1+cu124"
    assert report.value.cuda_available is True and report.value.device_count == 2
    assert report.value.devices[0].uuid == "aaaaaaaa-0000-4000-8000-00000000000a"
    assert isinstance(report.value.cuda_init_warnings, tuple)


def test_decoder_reports_import_failure_as_installed_but_unusable():
    outcome = _decode(
        json.dumps(
            {"probe_protocol": PROBE_PROTOCOL_VERSION, "import_ok": False, "import_error": "boom"}
        )
    )
    assert outcome.status is S.COLLECTION_FAILED
    assert (outcome.error.code, outcome.error.message) == ("IMPORT_ERROR", "boom")


def test_decoder_maps_a_crash_to_probe_exit_with_stderr_tail():
    outcome = _decode("", returncode=-11, stderr="Segmentation fault\n")
    assert outcome.error.code == "PROBE_EXIT_-11"
    assert outcome.error.message == "probe exited with status -11: Segmentation fault"
    assert _decode("", returncode=2).error.message == "probe exited with status 2"


def _with(**changes) -> str:
    payload = _payload()
    for key, value in changes.items():
        if value is KeyError:
            del payload[key]
        else:
            payload[key] = value
    return json.dumps(payload)


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("not json", id="invalid-json"),
        pytest.param("[1, 2]", id="not-an-object"),
        pytest.param(_with(probe_protocol=PROBE_PROTOCOL_VERSION + 1), id="newer-protocol"),
        pytest.param(_with(probe_protocol=1), id="protocol-1-without-device-error"),
        pytest.param(_with(probe_protocol=True), id="protocol-as-bool"),
        pytest.param(_with(import_error=KeyError), id="missing-header-key"),
        pytest.param(_with(import_ok="yes"), id="import-ok-not-bool"),
        pytest.param(_with(import_error="stale"), id="error-with-successful-import"),
        pytest.param(_with(cuda_available=KeyError), id="missing-field"),
        pytest.param(_with(extra_field=1), id="unexpected-field"),
        pytest.param(_with(torch_version=2.4), id="wrong-type"),
        pytest.param(_with(device_count=True), id="bool-for-int"),
        pytest.param(_with(cuda_init_warnings=[1]), id="non-string-warning"),
        pytest.param(_with(devices=[{"index": 0}]), id="incomplete-device"),
        pytest.param(_with(devices=["gpu"]), id="device-not-an-object"),
        pytest.param(_with(device_error=3), id="device-error-not-a-string"),
        pytest.param(_with(device_error=KeyError), id="protocol-2-missing-device-error"),
    ],
)
def test_decoder_rejects_malformed_output(stdout):
    outcome = _decode(stdout)
    assert outcome.status is S.COLLECTION_FAILED
    assert outcome.error.code == "PROBE_PROTOCOL_ERROR" and outcome.value is None


def test_decoder_passes_launch_failures_through():
    timeout = Observed.failed(
        S.COLLECTION_FAILED, code="TIMEOUT", message="slow", source=Source.PYTORCH, method=METHOD
    )
    assert decode_probe_output(timeout) is timeout
