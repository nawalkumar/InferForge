"""`FakeCommandRunner` honours the `CommandRunner` contract and serves the future
`nvidia-smi` and probe command interfaces without starting any process."""

import json
import subprocess

import pytest
from gsi_testkit.fakes import CommandCall, FakeAdapterSet, FakeCommandRunner
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    MAX_COMMAND_OUTPUT_CHARS,
    NVIDIA_SMI_QUERY_FIELDS,
    NVIDIA_SMI_WSL_PATH,
    AdapterUsageError,
    decode_probe_output,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
    require_success,
    torch_probe_argv,
)
from gpu_system_inspector.models import CollectionStatus, Source

SMI = "/usr/bin/nvidia-smi"
QUERY = nvidia_smi_query_argv(SMI)
# Canned `--query-gpu` output (architecture §4.4): one healthy GPU, one with the markers
# nvidia-smi prints for unsupported, unavailable and denied fields.
CANNED_CSV = (
    "0, GPU-aaaaaaaa-0000-4000-8000-00000000000a, Synthetic GPU A (fixture), 00000000:01:00.0, "
    "24564, 1024, 23540, 41, 3, 1, 61.25, 450.00, 550.54.15\n"
    "1, GPU-bbbbbbbb-0000-4000-8000-00000000000b, Synthetic GPU B (fixture), 00000000:02:00.0, "
    "12282, [Insufficient Permissions], [Insufficient Permissions], [N/A], [Not Supported], "
    "[Not Supported], [N/A], 200.00, 550.54.15\n"
)


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a real process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)


def _runner(**kwargs) -> FakeCommandRunner:
    return FakeCommandRunner(executables={"nvidia-smi": SMI}, **kwargs)


def test_query_command_is_a_fixed_argument_vector():
    assert QUERY == (
        SMI,
        "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    )
    assert nvidia_smi_version_argv(SMI) == (SMI,)
    assert not any(c in arg for arg in QUERY for c in ";|&$`<>")


def test_canned_query_output_is_returned_verbatim():
    runner = _runner().on(*QUERY, stdout=CANNED_CSV)
    outcome = runner.run(QUERY, timeout_s=10.0, source=Source.NVIDIA_SMI)
    assert outcome.is_collected and outcome.source is Source.NVIDIA_SMI
    assert outcome.method.startswith("fixture:")
    rows = [line.split(", ") for line in outcome.value.stdout.splitlines()]
    assert [len(row) for row in rows] == [len(NVIDIA_SMI_QUERY_FIELDS)] * 2
    assert {"[N/A]", "[Not Supported]", "[Insufficient Permissions]"} <= set(rows[1])
    assert outcome.value.returncode == 0 and outcome.value.truncated is False


def test_non_zero_exit_is_collected_and_require_success_maps_it():
    runner = _runner().on(
        *QUERY,
        returncode=9,
        stderr="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
    )
    outcome = runner.run(QUERY, timeout_s=10.0, source=Source.NVIDIA_SMI)
    assert outcome.is_collected and outcome.value.returncode == 9
    failed = require_success(outcome)
    assert failed.error.code == "EXIT_9" and "NVIDIA driver" in failed.error.message


def test_missing_executable_is_not_detected_whatever_is_configured():
    runner = FakeCommandRunner().on(*QUERY, stdout=CANNED_CSV)
    outcome = runner.run(QUERY, timeout_s=10.0, source=Source.NVIDIA_SMI)
    assert outcome.status is CollectionStatus.NOT_DETECTED
    assert outcome.error.code == "FileNotFoundError"
    assert runner.calls == [CommandCall(QUERY, 10.0, Source.NVIDIA_SMI)]


def test_timeout_is_a_collection_failure():
    outcome = _runner().on(*QUERY, timeout=True).run(QUERY, timeout_s=2.5, source=Source.NVIDIA_SMI)
    assert outcome.status is CollectionStatus.COLLECTION_FAILED
    assert outcome.error.code == "TIMEOUT" and "2.5 s" in outcome.error.message


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (PermissionError("denied"), CollectionStatus.PERMISSION_DENIED, "PermissionError"),
        (OSError(8, "Exec format error"), CollectionStatus.COLLECTION_FAILED, "OSError"),
    ],
)
def test_execution_failures_are_mapped(exc, status, code):
    outcome = _runner().on(*QUERY, raises=exc).run(QUERY, timeout_s=10.0, source=Source.NVIDIA_SMI)
    assert (outcome.status, outcome.error.code) == (status, code)


def test_which_searches_path_then_extra_directories():
    runner = FakeCommandRunner(files=frozenset({NVIDIA_SMI_WSL_PATH}))
    assert runner.which("nvidia-smi").status is CollectionStatus.NOT_DETECTED
    found = runner.which("nvidia-smi", extra_paths=("/usr/lib/wsl/lib/",))
    assert found.value == NVIDIA_SMI_WSL_PATH
    assert _runner().which("nvidia-smi").value == SMI
    assert runner.lookups == ["nvidia-smi", "nvidia-smi"]
    wsl = FakeCommandRunner(files=frozenset({NVIDIA_SMI_WSL_PATH})).on(
        NVIDIA_SMI_WSL_PATH, stdout="x"
    )
    assert wsl.run((NVIDIA_SMI_WSL_PATH,), timeout_s=1.0, source=Source.NVIDIA_SMI).is_collected


def test_exact_match_beats_prefix_and_longest_prefix_wins():
    runner = (
        _runner()
        .on(SMI, prefix=True, stdout="any")
        .on(SMI, "--query-gpu=uuid", prefix=True, stdout="longer")
        .on(*QUERY, stdout="exact")
    )

    def run(argv):
        return runner.run(argv, timeout_s=1.0, source=Source.NVIDIA_SMI).value.stdout

    assert run(QUERY) == "exact"
    assert run((SMI, "--query-gpu=uuid", "--format=csv")) == "longer"
    assert run((SMI, "-L")) == "any"


def test_unconfigured_commands_fail_the_test():
    with pytest.raises(AssertionError, match="no canned response"):
        _runner().run((SMI, "-r"), timeout_s=1.0, source=Source.NVIDIA_SMI)


@pytest.mark.parametrize(
    ("argv", "timeout_s"),
    [
        pytest.param(f"{SMI} -L", 1.0, id="shell-string"),
        pytest.param((), 1.0, id="empty"),
        pytest.param((SMI, ""), 1.0, id="empty-argument"),
        pytest.param(QUERY, 0, id="zero-timeout"),
    ],
)
def test_contract_violations_are_usage_errors(argv, timeout_s):
    runner = _runner().on(*QUERY)
    with pytest.raises(AdapterUsageError):
        runner.run(argv, timeout_s=timeout_s, source=Source.NVIDIA_SMI)
    assert runner.calls == []


def test_output_is_capped():
    runner = _runner().on(*QUERY, stdout="x" * (MAX_COMMAND_OUTPUT_CHARS + 10))
    result = runner.run(QUERY, timeout_s=1.0, source=Source.NVIDIA_SMI).value
    assert result.truncated and len(result.stdout) == MAX_COMMAND_OUTPUT_CHARS


def test_responses_are_deterministic():
    def outcomes():
        runner = _runner().on(*QUERY, stdout=CANNED_CSV).on(SMI, timeout=True)
        return [
            runner.run(argv, timeout_s=1.0, source=Source.NVIDIA_SMI) for argv in (QUERY, (SMI,))
        ]

    assert outcomes() == outcomes()


def test_runner_from_a_fixture_knows_only_the_fixture_executables():
    healthy = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"]).commands
    assert healthy.which("nvidia-smi").value == "/usr/bin/nvidia-smi"
    absent = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX03"]).commands
    assert absent.run(QUERY, timeout_s=1.0, source=Source.NVIDIA_SMI).status is (
        CollectionStatus.NOT_DETECTED
    )


def test_probe_command_interface_through_the_runner():
    python = "/opt/synthetic/venv/bin/python"
    argv = torch_probe_argv(python)
    payload = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"]).torch_fake
    stdout = payload.subprocess_output(60.0).value.stdout
    runner = FakeCommandRunner(executables={"python": python}).on(*argv, stdout=stdout)
    report = decode_probe_output(runner.run(argv, timeout_s=60.0, source=Source.PYTORCH))
    assert report.is_collected and report.value.torch_version == json.loads(stdout)["torch_version"]
    crashed = FakeCommandRunner(executables={"python": python}).on(
        *argv, returncode=1, stderr="Fatal Python error"
    )
    outcome = decode_probe_output(crashed.run(argv, timeout_s=60.0, source=Source.PYTORCH))
    assert outcome.error.code == "PROBE_EXIT_1" and "Fatal Python error" in outcome.error.message
