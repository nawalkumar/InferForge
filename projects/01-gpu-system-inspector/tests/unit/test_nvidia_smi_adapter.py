"""IMP-A02 production `NvidiaSmiCli` and its CSV parser: the parser on its own, the adapter
over `FakeCommandRunner`, a real child process (a fake `nvidia-smi` script) through the
production runner, and FX04a fallback parity. No NVIDIA tool is ever run."""

import sys

import pytest
from gsi_testkit.fakes import FakeCommandRunner, FakeNvidiaSmi, FixedClock, StubPynvml
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    NVIDIA_SMI_QUERY_FIELDS,
    NvidiaSmiAdapter,
    NvidiaSmiCli,
    PynvmlAdapter,
    SubprocessCommandRunner,
    nvidia_smi_query_argv,
    nvidia_smi_version_argv,
    parse_query_output,
)
from gpu_system_inspector.collectors import GpuHardwareCollector
from gpu_system_inspector.models import CollectionStatus, DataOrigin, PlatformProfile, Source

S = CollectionStatus
MIB = 2**20
SMI = "/usr/bin/nvidia-smi"
GPU0 = (
    "0, GPU-aaaa, NVIDIA A100-SXM4-40GB, 00000000:01:00.0, 40960, 1024, 39936, 33, 0, 0, "
    "52.10, 400.00, 550.54.15"
)
GPU1 = (
    '1, GPU-bbbb, "NVIDIA GeForce RTX 4090, Founders", 00000000:02:00.0, 24564, 0, 24564, '
    "41, 7, 3, 23.5, 450, 550.54.15"
)


class _Platform:
    def __init__(self, profile):
        self.profile = profile


def _cli(runner, profile=PlatformProfile.LINUX):
    return NvidiaSmiCli(runner, _Platform(profile), timeout_s=7.5)


def _query_runner(stdout="", **kwargs):
    return FakeCommandRunner(executables={"nvidia-smi": SMI}).on(
        *nvidia_smi_query_argv(SMI), stdout=stdout, **kwargs
    )


# --- the parser ---------------------------------------------------------------------------


def test_one_gpu_is_parsed_into_typed_values():
    rows = parse_query_output(GPU0 + "\n")
    assert rows.is_collected and rows.source is Source.NVIDIA_SMI
    (row,) = rows.value
    values = row.values
    assert row.index == 0 and set(values) == set(NVIDIA_SMI_QUERY_FIELDS[1:])
    assert values["name"].value == "NVIDIA A100-SXM4-40GB"
    assert values["memory.total"].value == 40960 * MIB and values["memory.total"].unit == "bytes"
    assert values["memory.total"].note == "Converted from MiB; precision 1 MiB"
    assert values["temperature.gpu"].value == 33 and values["temperature.gpu"].unit == "celsius"
    assert values["utilization.gpu"].value == 0 and values["utilization.gpu"].unit == "percent"
    assert values["power.draw"].value == 52.1 and values["power.limit"].unit == "watts"
    assert values["driver_version"].value == "550.54.15"
    assert values["uuid"].method == "nvidia-smi --query-gpu=uuid"


def test_multiple_gpus_come_back_sorted_by_index_with_quoted_fields():
    (first, second) = parse_query_output(f"{GPU1}\n\n{GPU0}\n").value
    assert (first.index, second.index) == (0, 1)
    assert second.values["name"].value == "NVIDIA GeForce RTX 4090, Founders"
    assert second.values["power.limit"].value == 450.0


@pytest.mark.parametrize("stdout", ["", "\n", "  \n\n"])
def test_empty_output_is_zero_gpus(stdout):
    rows = parse_query_output(stdout)
    assert rows.is_collected and rows.value == ()


@pytest.mark.parametrize(
    ("marker", "status", "code"),
    [
        ("[N/A]", S.NOT_SUPPORTED, "NVIDIA_SMI_NA"),
        ("[Not Supported]", S.NOT_SUPPORTED, "NVIDIA_SMI_NOT_SUPPORTED"),
        ("[Insufficient Permissions]", S.PERMISSION_DENIED, "NVIDIA_SMI_INSUFFICIENT_PERMISSIONS"),
        ("[Unknown Error]", S.COLLECTION_FAILED, "NVIDIA_SMI_UNKNOWN_ERROR"),
        ("[GPU is lost]", S.COLLECTION_FAILED, "NVIDIA_SMI_GPU_IS_LOST"),
    ],
)
def test_placeholders_are_outcomes_not_values(marker, status, code):
    line = GPU0.replace("52.10", marker).replace(", 33,", f", {marker},")
    (row,) = parse_query_output(line).value
    for field in ("power.draw", "temperature.gpu"):
        value = row.values[field]
        assert (value.status, value.error.code, value.value) == (status, code, None)
    assert row.values["power.draw"].unit == "watts" and row.values["name"].is_collected


@pytest.mark.parametrize("bad", ["12a", "-5", "1.5", "nan"])
def test_an_unparsable_number_fails_only_that_field(bad):
    (row,) = parse_query_output(GPU0.replace(", 33,", f", {bad},")).value
    value = row.values["temperature.gpu"]
    assert value.status is S.COLLECTION_FAILED and value.error.code == "PARSE_ERROR"
    assert row.values["memory.total"].is_collected


@pytest.mark.parametrize("bad", ["-1.0", "inf", "abc"])
def test_an_unparsable_power_value_fails_only_that_field(bad):
    (row,) = parse_query_output(GPU0.replace("52.10", bad)).value
    assert row.values["power.draw"].error.code == "PARSE_ERROR"


def test_an_empty_value_is_not_detected():
    (row,) = parse_query_output(GPU0.replace("GPU-aaaa", "")).value
    assert (
        row.values["uuid"].status is S.NOT_DETECTED
        and row.values["uuid"].error.code == "FIELD_NOT_FOUND"
    )


@pytest.mark.parametrize(
    ("stdout", "fragment"),
    [
        ("0, GPU-aaaa, name\n", "3 columns"),
        (GPU0 + ", extra\n", "14 columns"),
        (GPU0.replace("0, GPU-aaaa", "x, GPU-aaaa", 1), "no valid index"),
        (f"{GPU0}\n{GPU0}\n", "appears twice"),
        (f"{GPU0}\n0, GPU-bbbb", "columns"),
    ],
    ids=["too-few", "too-many", "bad-index", "duplicate", "partial-second-row"],
)
def test_a_malformed_row_fails_the_whole_query(stdout, fragment):
    rows = parse_query_output(stdout)
    assert rows.status is S.COLLECTION_FAILED and rows.error.code == "PARSE_ERROR"
    assert fragment in rows.error.message


def test_a_prefix_labels_synthetic_output():
    rows = parse_query_output(GPU0, prefix="fixture:")
    assert rows.method.startswith("fixture:") and rows.value[0].values["name"].method.startswith(
        "fixture:"
    )


# --- the adapter over a fake runner -------------------------------------------------------


def test_locate_uses_path_and_is_cached():
    runner = FakeCommandRunner(executables={"nvidia-smi": SMI})
    cli = _cli(runner)
    assert cli.locate().value == SMI and cli.locate().value == SMI
    assert runner.lookups == ["nvidia-smi"]


def test_wsl2_also_searches_the_wsl_driver_directory():
    runner = FakeCommandRunner(files=frozenset({"/usr/lib/wsl/lib/nvidia-smi"}))
    assert _cli(runner, PlatformProfile.WSL2).locate().value == "/usr/lib/wsl/lib/nvidia-smi"
    assert _cli(runner, PlatformProfile.LINUX).locate().status is S.NOT_DETECTED


def test_macos_is_not_supported_without_any_lookup():
    runner = FakeCommandRunner()
    cli = _cli(runner, PlatformProfile.MACOS)
    for outcome in (cli.locate(), cli.query_gpus(), cli.version().driver_version):
        assert outcome.status is S.NOT_SUPPORTED and outcome.error.code == "PLATFORM_NOT_SUPPORTED"
    assert runner.lookups == [] and runner.calls == []


def test_a_missing_executable_is_reported_and_nothing_runs():
    runner = FakeCommandRunner()
    cli = _cli(runner)
    query = cli.query_gpus()
    assert query.status is S.NOT_DETECTED and query.error.code == "FileNotFoundError"
    assert cli.version().cuda_version.status is S.NOT_DETECTED and runner.calls == []


def test_a_successful_query_runs_exactly_once_with_explicit_arguments():
    runner = _query_runner(f"{GPU0}\n{GPU1}\n")
    rows = _cli(runner).query_gpus()
    assert [r.index for r in rows.value] == [0, 1]
    (call,) = runner.calls
    assert call.argv == nvidia_smi_query_argv(SMI) and call.timeout_s == 7.5
    assert call.source is Source.NVIDIA_SMI
    assert rows.method == "fixture:nvidia-smi --query-gpu --format=csv,noheader,nounits"


def test_a_non_zero_exit_is_a_failure_with_the_tools_message():
    runner = _query_runner(
        returncode=9,
        stderr="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
    )
    rows = _cli(runner).query_gpus()
    assert rows.status is S.COLLECTION_FAILED and rows.error.code == "EXIT_9"
    assert "couldn't communicate" in rows.error.message


def test_an_unknown_query_field_on_an_older_driver_is_not_supported():
    runner = _query_runner(
        returncode=2, stdout='Field "power.limit" is not a valid field to query.'
    )
    rows = _cli(runner).query_gpus()
    assert rows.status is S.NOT_SUPPORTED and rows.error.code == "NVIDIA_SMI_NOT_SUPPORTED"


def test_a_timeout_is_a_failure_not_an_empty_inventory():
    rows = _cli(_query_runner(timeout=True)).query_gpus()
    assert (
        rows.status is S.COLLECTION_FAILED and rows.error.code == "TIMEOUT" and rows.value is None
    )


def test_truncated_output_is_never_trusted():
    huge = (GPU0 + "\n") * 30_000  # beyond MAX_COMMAND_OUTPUT_CHARS
    rows = _cli(_query_runner(huge)).query_gpus()
    assert rows.status is S.COLLECTION_FAILED and "truncated" in rows.error.message


@pytest.mark.parametrize(
    ("header", "driver", "cuda"),
    [
        (
            "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |",
            "550.54.15",
            "12.4",
        ),
        (
            "| NVIDIA-SMI 575.51   KMD Version: 575.51.02   CUDA UMD Version: 12.9 |",
            "575.51.02",
            "12.9",
        ),
    ],
    ids=["old-labels", "new-labels"],
)
def test_version_reads_old_and_new_header_labels(header, driver, cuda):
    runner = FakeCommandRunner(executables={"nvidia-smi": SMI}).on(
        *nvidia_smi_version_argv(SMI), stdout=header
    )
    version = _cli(runner).version()
    assert (version.driver_version.value, version.cuda_version.value) == (driver, cuda)
    assert version.driver_version.method == "fixture:nvidia-smi header"


def test_version_without_labels_is_field_not_found_and_a_failed_run_is_exit():
    runner = FakeCommandRunner(executables={"nvidia-smi": SMI}).on(
        *nvidia_smi_version_argv(SMI), stdout="garbage"
    )
    assert _cli(runner).version().driver_version.error.code == "FIELD_NOT_FOUND"
    failing = FakeCommandRunner(executables={"nvidia-smi": SMI}).on(
        *nvidia_smi_version_argv(SMI), returncode=15, stderr="GPU has fallen off the bus"
    )
    assert _cli(failing).version().cuda_version.error.code == "EXIT_15"


def test_the_adapter_conforms_and_inherits_its_runners_provenance():
    assert isinstance(_cli(FakeCommandRunner()), NvidiaSmiAdapter)
    assert _cli(FakeCommandRunner()).provenance is DataOrigin.FIXTURE
    assert NvidiaSmiCli(SubprocessCommandRunner()).provenance is DataOrigin.LIVE


# --- a real child process through the production runner -----------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake executable")
def test_a_fake_nvidia_smi_through_the_production_runner(tmp_path, monkeypatch):
    script = tmp_path / "nvidia-smi"
    script.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if any(a.startswith('--query-gpu=') for a in sys.argv[1:]):\n"
        f"    print({GPU0!r}); print({GPU1!r})\n"
        "else:\n"
        "    print('| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    cli = NvidiaSmiCli(SubprocessCommandRunner(FixedClock()), _Platform(PlatformProfile.LINUX))
    assert cli.locate().value == str(script)
    rows = cli.query_gpus()
    assert [r.values["uuid"].value for r in rows.value] == ["GPU-aaaa", "GPU-bbbb"]
    assert rows.method == "nvidia-smi --query-gpu --format=csv,noheader,nounits"
    assert cli.version().cuda_version.value == "12.4"


# --- FX04a: the fallback inventory through both production adapters ------------------------


def _csv_from_fixture(snapshot) -> str:
    lines = []
    for row in FakeNvidiaSmi.from_snapshot(snapshot).query_gpus().value:
        cells = [str(row.index)]
        for field in NVIDIA_SMI_QUERY_FIELDS[1:]:
            value = row.values[field]
            if not value.is_collected:
                cells.append("[N/A]")
            elif value.unit == "bytes":
                cells.append(str(value.value // MIB))
            else:
                cells.append(str(value.value))
        lines.append(", ".join(cells))
    return "\n".join(lines) + "\n"


def test_fx04a_fallback_inventory_is_reproduced_by_the_production_adapters():
    snapshot = FIXTURES_BY_ID["FX04a"].snapshot
    stub = StubPynvml.from_snapshot(snapshot)  # the binding is missing
    runner = FakeCommandRunner.from_snapshot(snapshot)
    path = snapshot.driver.nvidia_smi_path.value
    runner.on(*nvidia_smi_query_argv(path), stdout=_csv_from_fixture(snapshot))
    with PynvmlAdapter(stub.loader, provenance=DataOrigin.FIXTURE).open() as session:
        outcome = GpuHardwareCollector(
            session,
            NvidiaSmiCli(runner),
            FixedClock(step_ms=25),
            pci=snapshot.gpu.pci_observations,
            environment=snapshot.system.linux_environment.value,
            profile=snapshot.system.platform_profile,
        ).collect()
    assert outcome.section == snapshot.gpu
    assert [c.argv for c in runner.calls] == [nvidia_smi_query_argv(path)]
