"""IMP-A05 production `SubprocessCommandRunner`: real child processes from tiny fake
executables in `tmp_path` (POSIX), plus a mocked `subprocess.run` for the boundary checks.
No NVIDIA tool or driver is involved."""

import errno
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from gsi_testkit.fakes import FixedClock

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    CommandRunner,
    SubprocessCommandRunner,
    require_success,
)
from gpu_system_inspector.adapters.outcomes import TIMEOUT
from gpu_system_inspector.models import CollectionStatus, DataOrigin, Source

S = CollectionStatus
posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake executables")


class _SteppingClock(FixedClock):
    def __init__(self, *ticks: int) -> None:
        super().__init__()
        self._values = iter(ticks)

    def monotonic_ms(self) -> int:
        return next(self._values)


@pytest.fixture
def tool(tmp_path):
    """Writes an executable Python script `name` into `tmp_path` and returns its path."""

    def make(name: str, body: str, *, mode: int = 0o755) -> str:
        path = tmp_path / name
        path.write_text(f"#!{sys.executable}\nimport sys, json, time\n{body}\n", encoding="utf-8")
        path.chmod(mode)
        return str(path)

    return make


def _run(argv, *, timeout_s=10.0, runner=None, source=Source.NVIDIA_SMI):
    return (runner or SubprocessCommandRunner()).run(argv, timeout_s=timeout_s, source=source)


# --- discovery ----------------------------------------------------------------------------


@posix
def test_which_finds_an_executable_on_path(tool, tmp_path, monkeypatch):
    path = tool("fake-tool", "print('hi')")
    monkeypatch.setenv("PATH", str(tmp_path))
    found = SubprocessCommandRunner().which("fake-tool")
    assert found.is_collected and found.value == path
    assert found.source is Source.OS and found.method == "shutil.which(fake-tool)"


@posix
def test_which_searches_extra_directories_after_path(tool, tmp_path, monkeypatch):
    path = tool("fake-tool", "print('hi')")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert SubprocessCommandRunner().which("fake-tool", extra_paths=(str(tmp_path),)).value == path


@posix
def test_which_ignores_files_that_are_not_executable(tool, tmp_path, monkeypatch):
    tool("fake-tool", "print('hi')", mode=0o644)
    monkeypatch.setenv("PATH", "/nonexistent")
    missing = SubprocessCommandRunner().which("fake-tool", extra_paths=(str(tmp_path),))
    assert missing.status is S.NOT_DETECTED and missing.error.code == "FileNotFoundError"


def test_a_missing_executable_is_not_detected(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    missing = SubprocessCommandRunner().which("nvidia-smi", extra_paths=("/also/missing",))
    assert missing.status is S.NOT_DETECTED and missing.value is None
    assert missing.error.message == "nvidia-smi not found on PATH or /also/missing"


@pytest.mark.parametrize("name", ["", "bin/nvidia-smi", "a\0b", 7])
def test_which_rejects_anything_but_a_name(name):
    with pytest.raises(AdapterUsageError):
        SubprocessCommandRunner().which(name)


# --- execution ----------------------------------------------------------------------------


@posix
def test_a_successful_command_is_collected_with_its_output_and_duration(tool):
    path = tool("ok", "print('out'); print('warn', file=sys.stderr)")
    runner = SubprocessCommandRunner(_SteppingClock(10, 25))
    outcome = _run((path, "--flag"), runner=runner)
    assert outcome.is_collected and outcome.source is Source.NVIDIA_SMI
    assert outcome.method == f"{path} --flag"
    result = outcome.value
    assert (result.argv, result.returncode, result.stdout, result.stderr) == (
        (path, "--flag"),
        0,
        "out\n",
        "warn\n",
    )
    assert result.duration_ms == 15 and result.truncated is False


@posix
def test_arguments_reach_the_program_verbatim_without_a_shell(tool, tmp_path):
    path = tool("echo-args", "print(json.dumps(sys.argv[1:]))")
    tricky = ["a b", "quote'\"", "; rm -rf /", "$HOME", "$(id)", "`id`", "*", "new\nline", "é"]
    outcome = _run((path, *tricky))
    assert json.loads(outcome.value.stdout) == tricky
    assert list(tmp_path.iterdir()) == [tmp_path / "echo-args"]  # nothing was expanded or run


@posix
def test_a_non_zero_exit_is_collected_and_require_success_reports_it(tool):
    path = tool("fails", "print('partial'); print('boom: bad GPU', file=sys.stderr); sys.exit(3)")
    outcome = _run((path,))
    assert outcome.is_collected and outcome.value.returncode == 3
    assert outcome.value.stdout == "partial\n" and "boom" in outcome.value.stderr
    failed = require_success(outcome)
    assert failed.status is S.COLLECTION_FAILED and failed.error.code == "EXIT_3"
    assert failed.error.message == "boom: bad GPU"


@posix
def test_empty_output_and_stderr_are_kept_as_they_are(tool):
    path = tool("quiet", "print('only stderr', file=sys.stderr)")
    result = _run((path,)).value
    assert result.stdout == "" and result.stderr == "only stderr\n" and result.returncode == 0


@posix
def test_the_child_gets_no_input(tool):
    path = tool("reads", "print(repr(sys.stdin.read()))")
    assert _run((path,), timeout_s=5).value.stdout == "''\n"


@posix
def test_a_timeout_is_a_failure_and_the_child_is_killed(tool):
    path = tool("hangs", "time.sleep(30)")
    started = time.monotonic()
    outcome = _run((path,), timeout_s=0.3)
    assert time.monotonic() - started < 10
    assert outcome.status is S.COLLECTION_FAILED and outcome.value is None
    assert outcome.error.code == TIMEOUT and "0.3 s" in outcome.error.message


def test_a_missing_program_is_not_detected(tmp_path):
    outcome = _run((str(tmp_path / "absent"),))
    assert outcome.status is S.NOT_DETECTED and outcome.error.code == "FileNotFoundError"


@posix
def test_a_program_without_execute_permission_is_permission_denied(tool):
    path = tool("noexec", "print('never')", mode=0o644)
    outcome = _run((path,))
    assert outcome.status is S.PERMISSION_DENIED and outcome.error.code == "PermissionError"


@posix
def test_output_beyond_the_cap_is_truncated_and_flagged(tool):
    path = tool("loud", "print('x' * 100)")
    result = _run((path,), runner=SubprocessCommandRunner(max_output_chars=10)).value
    assert result.stdout == "x" * 10 and result.truncated is True


# --- the subprocess boundary (mocked) -----------------------------------------------------


def _mock_run(monkeypatch, result=None, raises=None):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_subprocess_is_called_with_a_list_no_shell_no_stdin_and_the_timeout(monkeypatch):
    calls = _mock_run(monkeypatch, SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    _run(("tool", "--x"), timeout_s=2.5)
    ((args, kwargs),) = calls
    assert args == (["tool", "--x"],)
    assert kwargs["shell"] is False and kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True and kwargs["timeout"] == 2.5
    assert kwargs["text"] is True and kwargs["errors"] == "replace" and "env" not in kwargs


def test_other_os_errors_are_collection_failures(monkeypatch):
    _mock_run(monkeypatch, raises=OSError(errno.ENOEXEC, "Exec format error"))
    outcome = _run(("tool",))
    assert outcome.status is S.COLLECTION_FAILED and outcome.error.code == "OSError"


def test_none_output_is_normalised_to_empty_text(monkeypatch):
    _mock_run(monkeypatch, SimpleNamespace(returncode=0, stdout=None, stderr=None))
    result = _run(("tool",)).value
    assert result.stdout == "" and result.stderr == ""


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode="0", stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout=b"bytes", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=3),
    ],
    ids=["returncode-not-int", "stdout-bytes", "stderr-not-text"],
)
def test_a_malformed_subprocess_result_is_a_bug_that_propagates(monkeypatch, result):
    _mock_run(monkeypatch, result)
    with pytest.raises(TypeError, match="unexpected result"):
        _run(("tool",))


def test_unexpected_exceptions_propagate_to_the_guard(monkeypatch):
    _mock_run(monkeypatch, raises=RuntimeError("interpreter trouble"))
    with pytest.raises(RuntimeError, match="interpreter trouble"):
        _run(("tool",))


@pytest.mark.parametrize(
    ("argv", "timeout_s"),
    [
        ("nvidia-smi -q", 1.0),
        (b"nvidia-smi", 1.0),
        ((), 1.0),
        (("",), 1.0),
        (("tool", 3), 1.0),
        (("tool", "a\0b"), 1.0),
        (("tool",), 0),
        (("tool",), -1),
        (("tool",), True),
        (("tool",), "5"),
    ],
    ids=[
        "shell-string",
        "bytes",
        "empty",
        "empty-program",
        "non-str",
        "nul",
        "zero",
        "negative",
        "bool",
        "str-timeout",
    ],
)
def test_caller_bugs_are_usage_errors_and_start_nothing(monkeypatch, argv, timeout_s):
    calls = _mock_run(monkeypatch, SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(AdapterUsageError):
        _run(argv, timeout_s=timeout_s)
    assert calls == []


def test_the_runner_is_a_live_command_runner():
    runner = SubprocessCommandRunner()
    assert isinstance(runner, CommandRunner) and runner.provenance is DataOrigin.LIVE


def test_importing_the_runner_does_not_import_subprocess():
    code = (
        "import sys\n"
        "import gpu_system_inspector.adapters.commands\n"
        "print('subprocess' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ}
    )
    assert result.stdout.strip() == "False", result.stderr
