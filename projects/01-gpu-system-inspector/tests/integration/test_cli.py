import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from gpu_system_inspector import __version__
from gpu_system_inspector.cli.exit_codes import ExitCode
from gpu_system_inspector.cli.main import main

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


def test_inspect_help_documents_interface_and_placeholder_status(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["inspect", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in INSPECT_OPTIONS:
        assert option in out
    assert "NOT IMPLEMENTED YET" in out
    assert "does not guarantee a model will fit or run" in out
    for code in ("0  report produced", "2  invalid usage", "6  inspect is not implemented"):
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


# --- inspect placeholder ----------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect"],
        ["inspect", "--required-vram", "8"],
        ["inspect", "--required-vram", "7.5", "--no-pytorch", "--no-processes", "--redact"],
        ["inspect", "--timeout", "1", "--pytorch-timeout", "600", "-vv"],
        ["inspect", "--json", "-"],
    ],
)
def test_inspect_is_an_explicit_placeholder(capsys, argv):
    assert main(argv) == ExitCode.NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "'inspect' is not implemented" in captured.err
    assert "No inspection was performed" in captured.err


def test_inspect_placeholder_writes_no_report(tmp_path, capsys):
    target = tmp_path / "report.json"
    assert main(["inspect", "--json", str(target)]) == ExitCode.NOT_IMPLEMENTED
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_inspect_placeholder_loads_no_gpu_libraries():
    code = (
        "import sys\n"
        "from gpu_system_inspector.cli.main import main\n"
        "status = main(['inspect'])\n"
        "print(int(status), sorted(m for m in ('pynvml', 'torch') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "6 []"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc", "1e9", "1048577"])
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
    inspect_result = subprocess.run([executable, "inspect"], capture_output=True, text=True)
    assert inspect_result.returncode == ExitCode.NOT_IMPLEMENTED
    assert inspect_result.stdout == ""


def test_module_invocation_matches_console_script():
    result = subprocess.run(
        [sys.executable, "-m", "gpu_system_inspector", "version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.startswith(f"GPU System Inspector {__version__}")
