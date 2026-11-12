import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import jsonschema
import pytest
from gsi_testkit.fakes import FakeAdapterSet
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector import __version__
from gpu_system_inspector.cli import main as cli_main
from gpu_system_inspector.cli.exit_codes import ExitCode
from gpu_system_inspector.cli.main import main
from gpu_system_inspector.core import Inspector
from gpu_system_inspector.models import InspectionOptions
from gpu_system_inspector.reporting import (
    FIXTURE_BANNER,
    RedactionPolicy,
    load_report_schema,
    parse_report_json,
)

INSPECT_OPTIONS = [
    "--json",
    "--required-vram",
    "--no-pytorch",
    "--no-processes",
    "--redact",
    "--timeout",
    "--pytorch-timeout",
    "--verbose",
]


def _usage_error(capsys, argv: list[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == ExitCode.USAGE_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


# --- help -------------------------------------------------------------------------------


def test_top_level_help_lists_both_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: inferforge-gpu")
    assert "inspect" in out and "version" in out


def test_inspect_help_documents_the_interface(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["inspect", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in INSPECT_OPTIONS:
        assert option in out
    assert "does not guarantee a model will fit or run" in out
    for code in ("0  report produced", "2  invalid usage", "4  the JSON report", "5  internal"):
        assert code in out


def test_version_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["version", "--help"])
    assert exc.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_missing_command_is_a_usage_error(capsys):
    assert "error: a command is required" in _usage_error(capsys, [])


def test_unknown_command_is_a_usage_error(capsys):
    _usage_error(capsys, ["frobnicate"])


# --- version ----------------------------------------------------------------------------


def test_version_text(capsys):
    assert main(["version"]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"GPU System Inspector {__version__} (distribution 01-gpu-system-inspector)",
        "report schema inferforge.gpu_system_inspector.report 1.0",
        f"python {platform.python_version()} ({platform.python_implementation()})",
        f"nvidia-ml-py {metadata.version('nvidia-ml-py')}",
    ]
    assert captured.err == ""


def test_version_json_on_stdout_replaces_text(capsys):
    assert main(["version", "--json", "-"]) == ExitCode.SUCCESS
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "application": {
            "name": "gpu-system-inspector",
            "display_name": "GPU System Inspector",
            "distribution": "01-gpu-system-inspector",
            "version": __version__,
        },
        "schema_name": "inferforge.gpu_system_inspector.report",
        "schema_version": "1.0",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "nvml_binding": {
            "distribution": "nvidia-ml-py",
            "version": metadata.version("nvidia-ml-py"),
        },
    }


def test_version_json_file_is_written_atomically(tmp_path, capsys):
    target = tmp_path / "version.json"
    assert main(["version", "--json", str(target)]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert captured.out.startswith("GPU System Inspector")
    assert f"written to {target}" in captured.err
    assert list(tmp_path.iterdir()) == [target]


def test_version_json_with_missing_parent_is_an_output_error(tmp_path, capsys):
    target = tmp_path / "missing-dir" / "version.json"
    assert main(["version", "--json", str(target)]) == ExitCode.OUTPUT_ERROR
    captured = capsys.readouterr()
    assert "error: cannot write JSON to" in captured.err
    assert captured.out.startswith("GPU System Inspector")
    assert list(tmp_path.iterdir()) == []


def test_version_json_onto_a_directory_leaves_no_temporary_file(tmp_path, capsys):
    target = tmp_path / "existing-dir"
    target.mkdir()
    assert main(["version", "--json", str(target)]) == ExitCode.OUTPUT_ERROR
    assert "error: cannot write JSON to" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [target]
    assert list(target.iterdir()) == []


def test_version_reports_a_missing_required_binding(monkeypatch, capsys):
    real_version = metadata.version

    def fake_version(name: str) -> str:
        if name == "nvidia-ml-py":
            raise metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(metadata, "version", fake_version)
    assert main(["version"]) == ExitCode.SUCCESS
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "nvidia-ml-py NOT INSTALLED (required dependency missing)"
    assert main(["version", "--json", "-"]) == ExitCode.SUCCESS
    assert json.loads(capsys.readouterr().out)["nvml_binding"] is None


# --- inspect (IMP-C01–C03) over fixture adapters -------------------------------------------


def _use_fixture(monkeypatch, fixture, seen=None):
    def run(options):
        if seen is not None:
            seen.append(options)
        return Inspector(FakeAdapterSet.from_scenario(fixture).adapter_set()).run(options)

    monkeypatch.setattr(cli_main, "run_inspection", run)


def _raise(exc):
    def run(options):
        raise exc

    return run


def test_inspect_prints_the_text_report_and_exits_0(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])
    assert main(["inspect"]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == lines[-1] == FIXTURE_BANNER
    assert lines[1].startswith(f"GPU System Inspector {__version__} · InferForge")
    order = [
        "SYSTEM",
        "DEVICES",
        "DRIVER & CUDA",
        "CUDA TOOLKIT",
        "PYTORCH",
        "RUNTIME SNAPSHOT",
        "FINDINGS",
        "UNSUPPORTED",
        "COLLECTION ERRORS",
        "LIMITATIONS",
    ]
    heads = [line.split("  ")[0] for line in lines if line and not line.startswith(" ")]
    assert [h for h in heads if h in order] == order
    assert "READINESS" not in heads
    assert captured.err == ""
    assert "\x1b" not in captured.out


def test_inspect_options_reach_the_inspection(monkeypatch, capsys):
    seen = []
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"], seen)
    argv = [
        "inspect",
        "--required-vram",
        "7.5",
        "--no-pytorch",
        "--no-processes",
        "--timeout",
        "3",
        "--pytorch-timeout",
        "30",
    ]
    assert main(argv) == ExitCode.SUCCESS
    (options,) = seen
    assert options.required_vram_gib == 7.5
    assert not options.include_pytorch and not options.include_processes
    assert options.subprocess_timeout_s == 3 and options.pytorch_timeout_s == 30
    assert not options.redact
    assert "READINESS" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("fixture", "run"),
    [pytest.param(f, r, id=r.label) for f in ALL_FIXTURES for r in f.readiness_runs],
)
def test_mode_b_exit_codes_follow_readiness(monkeypatch, capsys, fixture, run):
    _use_fixture(monkeypatch, fixture)
    code = main(["inspect", "--required-vram", str(run.required_vram_gib)])
    assert code == run.exit_code
    assert f"overall: {run.overall_status} ({run.overall_reason})" in capsys.readouterr().out


def test_json_to_stdout_replaces_the_text_report(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX06"])
    assert main(["inspect", "--json", "-"]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    jsonschema.validate(data, load_report_schema())
    assert captured.out.endswith("}\n") and captured.err == ""
    assert {f["check_id"] for f in data["diagnostics"]} == {f"D0{i}" for i in range(1, 9)}


def test_json_file_is_written_atomically_and_text_still_printed(monkeypatch, tmp_path, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])
    target = tmp_path / "report.json"
    target.write_text("old", encoding="utf-8")
    assert main(["inspect", "--json", str(target)]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    report = parse_report_json(target.read_text(encoding="utf-8"))
    assert report.data_origin.value == "FIXTURE"
    assert captured.err.strip() == f"JSON report written to {target}"
    assert captured.out.startswith(FIXTURE_BANNER)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kind", ["missing-parent", "directory"])
def test_json_write_failures_exit_4_and_leave_no_partial_file(monkeypatch, tmp_path, capsys, kind):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])
    target = tmp_path / "missing" / "r.json" if kind == "missing-parent" else tmp_path / "d"
    if kind == "directory":
        target.mkdir()
    assert main(["inspect", "--json", str(target)]) == ExitCode.OUTPUT_ERROR
    captured = capsys.readouterr()
    assert f"error: cannot write JSON report to {target}: " in captured.err
    assert captured.out.startswith(FIXTURE_BANNER)
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_the_json_file_equals_the_stdout_json(monkeypatch, tmp_path, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX07"])
    target = tmp_path / "r.json"
    assert main(["inspect", "--required-vram", "8", "--json", str(target)]) == 1
    capsys.readouterr()
    assert main(["inspect", "--required-vram", "8", "--json", "-"]) == 1
    assert target.read_text(encoding="utf-8") == capsys.readouterr().out


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_read_only_directory_exits_4_and_leaves_nothing(monkeypatch, tmp_path, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert main(["inspect", "--json", str(locked / "r.json")]) == ExitCode.OUTPUT_ERROR
        assert "Permission denied" in capsys.readouterr().err
        assert list(locked.iterdir()) == []
    finally:
        locked.chmod(0o700)


def test_an_output_error_takes_precedence_over_readiness(monkeypatch, tmp_path, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX07"])
    target = tmp_path / "missing" / "r.json"
    assert main(["inspect", "--required-vram", "8", "--json", str(target)]) == 4


def test_an_internal_error_exits_5_with_nothing_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "run_inspection", _raise(RuntimeError("boom")))
    assert main(["inspect"]) == ExitCode.INTERNAL_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "error: internal error: RuntimeError: boom (run with -vv for details)"
    )


def test_an_internal_error_with_vv_prints_the_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "run_inspection", _raise(RuntimeError("boom")))
    assert main(["inspect", "-vv"]) == ExitCode.INTERNAL_ERROR
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_ctrl_c_exits_130(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "run_inspection", _raise(KeyboardInterrupt()))
    assert main(["inspect"]) == ExitCode.INTERRUPTED
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip() == "interrupted"


def test_a_collector_that_raised_gets_a_warning_line(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX10"])
    assert main(["inspect"]) == ExitCode.SUCCESS
    err = capsys.readouterr().err
    assert "WARNING: collector 'cuda_environment' raised an error" in err


def test_a_missing_required_binding_gets_a_warning_line(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX04a"])
    assert main(["inspect"]) == ExitCode.SUCCESS
    err = capsys.readouterr().err
    assert "WARNING: the required dependency nvidia-ml-py is not installed" in err


def test_redact_removes_identifiers_from_text_json_and_messages(monkeypatch, tmp_path, capsys):
    seen = []
    fixture = FIXTURES_BY_ID["FX06"]
    _use_fixture(monkeypatch, fixture, seen)
    home = tmp_path / "home" / "alice"
    home.mkdir(parents=True)
    monkeypatch.setattr(
        cli_main,
        "local_redaction_policy",
        lambda: RedactionPolicy(user="alice", home=str(home)),
    )
    target = home / "shareable.json"
    assert main(["inspect", "--redact", "--json", str(target)]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    assert seen[0].redact
    data = target.read_text(encoding="utf-8")
    report = parse_report_json(data)
    assert report.redaction.applied and report.options.redact
    secrets = [d.uuid.value for d in fixture.snapshot.gpu.devices]
    secrets += [d.pci_bus_id.value for d in fixture.snapshot.gpu.devices]
    secrets += ["/opt/synthetic/venv/bin/python", "/usr/local/cuda-12.4", "/usr/bin/nvidia-smi"]
    for secret in secrets:
        assert secret not in captured.out and secret not in data, secret
    assert "<redacted:uuid:1>" in captured.out and "<redacted:process>" in captured.out
    assert captured.err.strip() == "JSON report written to ~/shareable.json"
    assert "alice" not in captured.err


def test_redacted_internal_errors_hide_the_home_directory(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main,
        "local_redaction_policy",
        lambda: RedactionPolicy(user="alice", home="/home/alice"),
    )
    monkeypatch.setattr(
        cli_main, "run_inspection", _raise(OSError("cannot read /home/alice/x for alice"))
    )
    assert main(["inspect", "--redact"]) == ExitCode.INTERNAL_ERROR
    err = capsys.readouterr().err
    assert "/home/alice" not in err and "alice" not in err
    assert "~/x for <redacted:user>" in err


@pytest.mark.parametrize(
    "fixture", [f for f in ALL_FIXTURES if not f.mode_b_only], ids=lambda f: f.fixture_id
)
def test_mode_a_always_exits_0(monkeypatch, capsys, fixture):
    """CLI §6: Mode A never fails because of findings (AC01, AC03, AC08)."""
    _use_fixture(monkeypatch, fixture)
    assert main(["inspect"]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


def test_json_on_stdout_stays_parseable_at_debug_verbosity(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX10"])
    assert main(["inspect", "--json", "-", "-vv"]) == ExitCode.SUCCESS
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "WARNING: collector 'cuda_environment'" in captured.err


def test_report_content_is_identical_across_verbosity(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])
    outputs = []
    for flags in ([], ["-v"], ["-vv"]):
        main(["inspect", "--json", "-", *flags])
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] == outputs[2]


def test_an_exception_inside_the_core_exits_5(monkeypatch, capsys):
    _use_fixture(monkeypatch, FIXTURES_BY_ID["FX01"])

    def explode(self, options=None):
        raise ValueError("core failure")

    monkeypatch.setattr(Inspector, "run", explode)
    assert main(["inspect"]) == ExitCode.INTERNAL_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "error: internal error: ValueError: core failure (run with -vv for details)"
    ]


def test_the_api_report_equals_the_cli_json(monkeypatch, capsys):
    """AC58, over a fixture: the library and the CLI produce the same report."""
    fixture = FIXTURES_BY_ID["FX06"]
    _use_fixture(monkeypatch, fixture)
    assert main(["inspect", "--json", "-", "--required-vram", "16"]) == ExitCode.SUCCESS
    cli_report = parse_report_json(capsys.readouterr().out)
    api_report = Inspector(FakeAdapterSet.from_scenario(fixture).adapter_set()).run(
        InspectionOptions(required_vram_gib=16)
    )
    assert cli_report == api_report


GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize(
    ("fixture_id", "argv"),
    [("FX01", []), ("FX03", []), ("FX07", ["--required-vram", "8"]), ("FX13", [])],
)
def test_text_report_matches_its_golden_file(monkeypatch, capsys, fixture_id, argv):
    """AC26 golden text files. Regenerate deliberately with GSI_UPDATE_GOLDEN=1."""
    _use_fixture(monkeypatch, FIXTURES_BY_ID[fixture_id])
    main(["inspect", *argv])
    text = capsys.readouterr().out
    path = GOLDEN / f"{fixture_id}.txt"
    if os.environ.get("GSI_UPDATE_GOLDEN") == "1":
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


def test_version_and_help_load_no_gpu_libraries_or_the_core():
    code = (
        "import sys\n"
        "from gpu_system_inspector.cli.main import main\n"
        "status = main(['version'])\n"
        "print(int(status), sorted(m for m in ('pynvml', 'torch', 'gpu_system_inspector.core')"
        " if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip().splitlines()[-1] == "0 []"


def test_the_inspect_help_lists_the_exit_codes_without_a_placeholder(capsys):
    with pytest.raises(SystemExit):
        main(["inspect", "--help"])
    out = capsys.readouterr().out
    assert "NOT IMPLEMENTED" not in out and "6  " not in out
    assert "(not implemented yet)" not in out


@pytest.mark.parametrize(
    "value", ["0", "-1", "nan", "inf", "abc", "1e9", "1048577", "1e-12", "4.6e-10"]
)
def test_invalid_required_vram_is_rejected(capsys, value):
    err = _usage_error(capsys, ["inspect", "--required-vram", value])
    assert "--required-vram must be a positive number of GiB" in err
    assert "do not include a unit" not in err


@pytest.mark.parametrize("value", ["8GB", "8GiB", "8G", "7.5 GiB"])
def test_required_vram_with_unit_is_rejected_with_hint(capsys, value):
    assert "do not include a unit" in _usage_error(capsys, ["inspect", "--required-vram", value])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--timeout", "0"),
        ("--timeout", "301"),
        ("--timeout", "nan"),
        ("--pytorch-timeout", "601"),
        ("--pytorch-timeout", "soon"),
    ],
)
def test_invalid_timeouts_are_rejected(capsys, option, value):
    assert f"{option} must be between 0 and" in _usage_error(capsys, ["inspect", option, value])


def test_empty_json_path_is_rejected(capsys):
    assert "--json requires a file path or '-'" in _usage_error(capsys, ["inspect", "--json", ""])


# --- entry points -----------------------------------------------------------------------


def test_console_script_is_installed_and_runs():
    executable = shutil.which("inferforge-gpu", path=str(Path(sys.executable).parent))
    assert executable is not None
    help_result = subprocess.run([executable, "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert help_result.stdout.startswith("usage: inferforge-gpu")
    version_result = subprocess.run([executable, "version"], capture_output=True, text=True)
    assert version_result.returncode == 0
    assert version_result.stdout.startswith(f"GPU System Inspector {__version__}")


@pytest.mark.host
def test_console_script_inspects_this_host_live():
    """Runs the real inspection on this machine (host test: reads the real system)."""
    executable = shutil.which("inferforge-gpu", path=str(Path(sys.executable).parent))
    result = subprocess.run(
        [executable, "inspect", "--json", "-"], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == ExitCode.SUCCESS, result.stderr
    data = json.loads(result.stdout)
    jsonschema.validate(data, load_report_schema())
    assert data["data_origin"] == "LIVE" and data["mode"] == "GENERAL"
    assert {f["check_id"] for f in data["diagnostics"]} == {f"D0{i}" for i in range(1, 9)}
    text = subprocess.run([executable, "inspect"], capture_output=True, text=True, timeout=300)
    assert text.returncode == ExitCode.SUCCESS and FIXTURE_BANNER not in text.stdout


def test_module_invocation_matches_console_script():
    result = subprocess.run(
        [sys.executable, "-m", "gpu_system_inspector", "version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.startswith(f"GPU System Inspector {__version__}")
