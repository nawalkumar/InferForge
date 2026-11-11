"""IMP-A06 `CudaToolkitLocator` (DEC-47) over `FakeToolkitFiles` and `FakeCommandRunner`,
and `StdlibToolkitFiles` over a `tmp_path` tree. Every DEC-47 signal, multiple and aliased
installations, version evidence, failure statuses and platform gates. No real toolkit,
`/usr/local` or process is used."""

import errno
import os

import pytest
from gsi_testkit.fakes import (
    FakeCommandRunner,
    FakeToolkitFiles,
    nvcc_version_output,
    version_json,
)

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    CudaToolkitLocator,
    StdlibToolkitFiles,
)
from gpu_system_inspector.models import (
    CollectionStatus,
    Observed,
    PlatformProfile,
    Source,
    ToolkitDiscovery,
)

S = CollectionStatus
D = ToolkitDiscovery
ROOT = "/usr/local/cuda-12.4"
OLD = "/usr/local/cuda-11.8"
NVCC = f"{ROOT}/bin/nvcc"


class _Platform:
    def __init__(self, profile=PlatformProfile.LINUX):
        self.profile = profile


def _toolkit(files, root=ROOT, version="12.4.131", *, nvcc=True, metadata=True):
    files.add_directory(root)
    if metadata:
        files.add_file(f"{root}/version.json", version_json(version))
    if nvcc:
        files.add_file(f"{root}/bin/nvcc")
    return files


def _locate(files, runner=None, hints=None, profile=PlatformProfile.LINUX):
    runner = runner if runner is not None else FakeCommandRunner()
    locator = CudaToolkitLocator(runner, _Platform(profile), files)
    return locator.discover(hints or {}), runner


def _on_path(runner, path=NVCC, version="12.4.131", **response):
    runner.executables = {**runner.executables, "nvcc": path}
    response = response or {"stdout": nvcc_version_output(version)}
    return runner.on(path, "--version", **response)


def _candidates(result):
    return {c.root_path: c for c in result.toolkits.value}


# --- nothing, one, several ----------------------------------------------------------------


def test_no_toolkit_is_not_detected_and_nothing_is_guessed():
    result, runner = _locate(FakeToolkitFiles().add_directory("/usr/local/bin"))
    assert result.toolkits.status is S.NOT_DETECTED
    assert result.toolkits.error.code == "NO_TOOLKIT_FOUND"
    assert result.active_nvcc_path.error.code == "FileNotFoundError"
    assert result.active_nvcc_version == result.active_nvcc_path
    assert result.problems == () and runner.calls == []


def test_one_installation_from_cuda_home_with_version_json():
    files = _toolkit(FakeToolkitFiles())
    result, runner = _locate(files, hints={D.CUDA_HOME: (ROOT,)})
    (candidate,) = result.toolkits.value
    assert candidate.root_path == ROOT
    assert candidate.discovered_via == (D.CUDA_HOME, D.DEFAULT_LOCATION)
    assert (candidate.version.value, candidate.version.method) == (
        "12.4.131",
        "fixture:version.json:cuda.version",
    )
    assert candidate.version.source is Source.CUDA_TOOLKIT
    assert (
        candidate.nvcc_path.value == NVCC and candidate.nvcc_path.method == "fixture:toolkit layout"
    )
    assert runner.calls == []  # version.json is enough; no nvcc on PATH to run


def test_multiple_installations_and_a_symlink_alias_are_all_reported_in_order():
    files = _toolkit(_toolkit(FakeToolkitFiles(), OLD, "11.8.89"))
    files.add_link("/usr/local/cuda", ROOT)
    runner = _on_path(FakeCommandRunner(), "/usr/local/cuda/bin/nvcc")
    result, _ = _locate(files, runner, hints={D.CUDA_HOME: (OLD,)})
    old, new = result.toolkits.value
    assert [old.root_path, new.root_path] == [OLD, ROOT]
    assert old.discovered_via == (D.CUDA_HOME, D.DEFAULT_LOCATION)
    assert new.discovered_via == (D.NVCC_ON_PATH, D.DEFAULT_LOCATION)  # alias merged
    assert (old.version.value, new.version.value) == ("11.8.89", "12.4.131")  # both kept
    assert result.active_nvcc_path.value == "/usr/local/cuda/bin/nvcc"  # as PATH found it
    assert result.active_nvcc_version.value == "12.4.131"


def test_duplicate_signals_for_one_root_become_one_candidate():
    files = _toolkit(FakeToolkitFiles())
    result, _ = _locate(files, hints={D.CUDA_HOME: (ROOT,), D.CUDA_PATH: (ROOT + "/",)})
    (candidate,) = result.toolkits.value
    assert candidate.discovered_via == (D.CUDA_HOME, D.CUDA_PATH, D.DEFAULT_LOCATION)


def test_an_nvcc_on_path_through_a_symlink_points_at_its_real_toolkit():
    files = _toolkit(FakeToolkitFiles(), "/opt/cuda-12.4").add_link(
        "/usr/bin/nvcc", "/opt/cuda-12.4/bin/nvcc"
    )
    runner = _on_path(FakeCommandRunner(), "/usr/bin/nvcc")
    result, _ = _locate(files, runner)
    (candidate,) = result.toolkits.value
    assert candidate.root_path == "/opt/cuda-12.4" and candidate.discovered_via == (D.NVCC_ON_PATH,)
    assert result.active_nvcc_path.value == "/usr/bin/nvcc"


def test_an_nvcc_outside_a_bin_directory_names_no_toolkit():
    files = FakeToolkitFiles().add_file("/opt/tools/nvcc")
    result, _ = _locate(files, _on_path(FakeCommandRunner(), "/opt/tools/nvcc"))
    assert result.toolkits.status is S.NOT_DETECTED
    assert result.active_nvcc_version.value == "12.4.131"  # the compiler is still evidence


@pytest.mark.parametrize("path", ["/nowhere/cuda", "/usr/local/cuda-12.4/version.json"])
def test_a_signal_that_names_no_directory_is_not_a_candidate(path):
    result, _ = _locate(_toolkit(FakeToolkitFiles(), "/srv/other"), hints={D.CUDA_HOME: (path,)})
    assert result.toolkits.status is S.NOT_DETECTED


def test_only_cuda_named_default_entries_are_considered():
    files = FakeToolkitFiles()
    for name in ("cudnn", "cuda-x", "cuda_12", "bin"):
        _toolkit(files, f"/usr/local/{name}")
    files.add_file("/usr/local/cuda-13.0")  # a file, not a directory
    result, _ = _locate(files)
    assert result.toolkits.status is S.NOT_DETECTED


# --- version and compiler evidence --------------------------------------------------------


def test_a_missing_compiler_is_separate_evidence():
    files = _toolkit(FakeToolkitFiles(), nvcc=False)
    (candidate,) = _locate(files)[0].toolkits.value
    assert candidate.version.value == "12.4.131"
    assert candidate.nvcc_path.status is S.NOT_DETECTED
    assert candidate.nvcc_path.error.code == "NVCC_NOT_IN_TOOLKIT"


def test_without_version_json_the_toolkits_nvcc_is_run_once():
    files = _toolkit(FakeToolkitFiles(), metadata=False)
    runner = _on_path(FakeCommandRunner(), NVCC, "12.4.99")
    result, _ = _locate(files, runner)
    (candidate,) = result.toolkits.value
    assert (candidate.version.value, candidate.version.method) == (
        "12.4.99",
        "fixture:nvcc --version",
    )
    assert [c.argv for c in runner.calls] == [(NVCC, "--version")]  # shared with the active


def test_without_version_json_or_nvcc_the_version_is_not_detected():
    files = _toolkit(FakeToolkitFiles(), metadata=False, nvcc=False)
    (candidate,) = _locate(files)[0].toolkits.value
    assert candidate.version.error.code == "NO_VERSION_EVIDENCE"
    assert candidate.version.status is S.NOT_DETECTED


def test_a_directory_name_is_never_a_version():
    files = _toolkit(FakeToolkitFiles(), "/usr/local/cuda-12.4", metadata=False, nvcc=False)
    (candidate,) = _locate(files)[0].toolkits.value
    assert candidate.version.value is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        '{"cuda": {}}',
        '{"cuda": {"version": 12.4}}',
        '{"cuda": {"version": "12"}}',
        '{"cuda": {"version": "twelve"}}',
        '{"cuda": {"version": "12.4.131 "}}',
        " " * (64 * 1024 + 1),
    ],
    ids=["invalid", "array", "no-version", "number", "major-only", "text", "space", "too-large"],
)
def test_malformed_or_unsupported_metadata_is_a_parse_error_without_fallback(content):
    files = _toolkit(FakeToolkitFiles(), metadata=False)
    files.add_file(f"{ROOT}/version.json", content)
    (candidate,) = _locate(files)[0].toolkits.value
    assert candidate.version.status is S.COLLECTION_FAILED
    assert candidate.version.error.code == "PARSE_ERROR"
    assert candidate.version.method == "fixture:version.json:cuda.version"  # no nvcc fallback


@pytest.mark.parametrize(
    ("operation", "path", "error", "status"),
    [
        (
            "read_text",
            f"{ROOT}/version.json",
            PermissionError(errno.EACCES, "denied"),
            S.PERMISSION_DENIED,
        ),
        ("read_text", f"{ROOT}/version.json", OSError(errno.EIO, "I/O error"), S.COLLECTION_FAILED),
        ("kind", ROOT, PermissionError(errno.EACCES, "denied"), S.PERMISSION_DENIED),
    ],
    ids=["unreadable-metadata", "io-error", "unstatable-root"],
)
def test_inaccessible_evidence_keeps_its_status_and_is_never_not_detected(
    operation, path, error, status
):
    files = _toolkit(FakeToolkitFiles()).fail(operation, path, error)
    result, _ = _locate(files, hints={D.CUDA_HOME: (ROOT,)})
    (candidate,) = result.toolkits.value
    assert candidate.version.status is status


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PermissionError(errno.EACCES, "denied"), S.PERMISSION_DENIED),
        (OSError(errno.EIO, "io"), S.COLLECTION_FAILED),
    ],
)
def test_an_unreadable_default_location_is_reported_not_hidden(error, status):
    files = FakeToolkitFiles().add_directory("/usr/local").fail("list_dir", "/usr/local", error)
    result, _ = _locate(files)
    assert result.toolkits.status is status  # never NOT_DETECTED
    (problem,) = result.problems
    assert (problem.path, problem.status) == ("/usr/local", status)
    found, _ = _locate(
        _toolkit(FakeToolkitFiles(), "/opt/cuda").fail("list_dir", "/usr/local", error),
        hints={D.CUDA_HOME: ("/opt/cuda",)},
    )
    assert found.toolkits.is_collected and len(found.problems) == 1


def test_a_missing_default_location_is_simply_absent():
    result, _ = _locate(FakeToolkitFiles())
    assert result.toolkits.status is S.NOT_DETECTED and result.problems == ()


# --- the active nvcc ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "status", "code"),
    [
        ({"timeout": True}, S.COLLECTION_FAILED, "TIMEOUT"),
        ({"returncode": 2, "stderr": "broken"}, S.COLLECTION_FAILED, "EXIT_2"),
        ({"stdout": "nvcc: something else entirely"}, S.COLLECTION_FAILED, "PARSE_ERROR"),
        ({"stdout": ""}, S.COLLECTION_FAILED, "PARSE_ERROR"),
        (
            {"raises": PermissionError(errno.EACCES, "denied")},
            S.PERMISSION_DENIED,
            "PermissionError",
        ),
    ],
    ids=["timeout", "non-zero", "unexpected", "empty", "denied"],
)
def test_nvcc_failures_are_reported_with_their_status(response, status, code):
    files = _toolkit(FakeToolkitFiles())
    result, _ = _locate(files, _on_path(FakeCommandRunner(), NVCC, **response))
    assert (result.active_nvcc_version.status, result.active_nvcc_version.error.code) == (
        status,
        code,
    )
    assert result.active_nvcc_version.method == "fixture:nvcc --version"
    assert result.active_nvcc_path.value == NVCC  # found, just not usable
    assert result.toolkits.value[0].version.value == "12.4.131"  # version.json unaffected


def test_an_executable_that_vanished_before_running_is_not_detected():
    runner = _on_path(FakeCommandRunner(), raises=FileNotFoundError(errno.ENOENT, "gone"))
    result, _ = _locate(FakeToolkitFiles(), runner)
    assert result.active_nvcc_version.status is S.NOT_DETECTED


def test_release_only_output_gives_major_minor():
    result, _ = _locate(
        FakeToolkitFiles(), _on_path(FakeCommandRunner(), "/opt/x/nvcc", stdout="release 11.0,")
    )
    assert result.active_nvcc_version.value == "11.0"


def test_commands_use_the_configured_timeout_and_no_shell():
    files = _toolkit(FakeToolkitFiles())
    runner = _on_path(FakeCommandRunner(), NVCC)
    CudaToolkitLocator(runner, _Platform(), files, timeout_s=3.5).discover({})
    (call,) = runner.calls
    assert call.argv == (NVCC, "--version") and call.timeout_s == 3.5
    assert call.source is Source.CUDA_TOOLKIT


# --- platforms, determinism, import -------------------------------------------------------


@pytest.mark.parametrize("profile", [PlatformProfile.MACOS, PlatformProfile.OTHER])
def test_unsupported_platforms_read_and_run_nothing(profile):
    files = _toolkit(FakeToolkitFiles())
    runner = _on_path(FakeCommandRunner(), NVCC)
    result, _ = _locate(files, runner, hints={D.CUDA_HOME: (ROOT,)}, profile=profile)
    for field in (result.toolkits, result.active_nvcc_path, result.active_nvcc_version):
        assert (field.status, field.error.code) == (S.NOT_SUPPORTED, "PLATFORM_NOT_SUPPORTED")
    assert files.calls == [] and runner.calls == [] and runner.lookups == []


@pytest.mark.parametrize("profile", [PlatformProfile.LINUX, PlatformProfile.WSL2])
def test_linux_and_wsl2_use_the_same_linux_signals(profile):
    files = _toolkit(FakeToolkitFiles())
    (candidate,) = _locate(files, profile=profile)[0].toolkits.value
    assert candidate.discovered_via == (D.DEFAULT_LOCATION,)


def test_windows_uses_nvcc_exe_versioned_paths_and_no_default_location():
    root = "C:/CUDA/v12.4"
    files = FakeToolkitFiles().add_directory(root).add_file(f"{root}/bin/nvcc.exe")
    files.add_file(f"{root}/version.json", version_json("12.4.131"))
    _toolkit(files)  # a Linux-style tree that Windows must not scan for
    runner = _on_path(FakeCommandRunner(), f"{root}/bin/nvcc.exe")
    result, _ = _locate(
        files,
        runner,
        hints={D.CUDA_PATH: (root,), D.CUDA_PATH_VERSIONED: (root,)},
        profile=PlatformProfile.WINDOWS,
    )
    (candidate,) = result.toolkits.value
    assert candidate.discovered_via == (D.CUDA_PATH, D.CUDA_PATH_VERSIONED, D.NVCC_ON_PATH)
    assert candidate.nvcc_path.value == f"{root}/bin/nvcc.exe"
    assert ("list_dir", "/usr/local") not in files.calls


def test_only_environment_signals_can_be_passed_as_hints():
    with pytest.raises(AdapterUsageError, match="NVCC_ON_PATH"):
        _locate(FakeToolkitFiles(), hints={D.NVCC_ON_PATH: (ROOT,)})


def test_discovery_is_deterministic():
    def run():
        files = _toolkit(_toolkit(FakeToolkitFiles(), OLD, "11.8.89"))
        return _locate(files, _on_path(FakeCommandRunner(), NVCC), {D.CUDA_HOME: (OLD,)})[0]

    assert run() == run()


def test_construction_reads_and_runs_nothing():
    files, runner = FakeToolkitFiles(), FakeCommandRunner()
    CudaToolkitLocator(runner, _Platform(), files)
    assert files.calls == [] and runner.calls == [] and runner.lookups == []


# --- the production file seam on a temporary tree -----------------------------------------


def test_stdlib_files_report_kinds_links_and_errors(tmp_path):
    files = StdlibToolkitFiles()
    root = tmp_path / "cuda-12.4"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nvcc").write_text("")
    (root / "version.json").write_text(version_json("12.4.131"))
    alias = tmp_path / "cuda"
    alias.symlink_to(root)
    assert files.kind(str(root)) == "directory" and files.kind(str(alias)) == "directory"
    assert files.kind(str(root / "bin" / "nvcc")) == "file"
    assert files.kind(str(tmp_path / "missing")) is None
    assert files.kind(str(root / "version.json" / "x")) is None  # NotADirectoryError
    assert files.realpath(str(alias)) == str(root)
    assert sorted(files.list_dir(str(tmp_path))) == ["cuda", "cuda-12.4"]
    assert files.read_text(str(root / "version.json"), 5) == version_json("12.4.131")[:6]
    assert files.join("a", "b") == os.path.join("a", "b")
    assert (files.dirname("/a/b"), files.basename("/a/b")) == ("/a", "b")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert files.kind(str(fifo)) == "other"


def test_the_production_seam_through_the_locator(tmp_path):
    root = tmp_path / "toolkit"
    (root / "bin").mkdir(parents=True)
    (root / "version.json").write_text(version_json("12.6.20"))
    runner = FakeCommandRunner()
    result = CudaToolkitLocator(runner, _Platform(), StdlibToolkitFiles()).discover(
        {D.CUDA_HOME: (str(root),)}
    )
    candidates = _candidates(result)
    assert candidates[str(root)].version.value == "12.6.20"
    assert candidates[str(root)].nvcc_path.error.code == "NVCC_NOT_IN_TOOLKIT"


def test_a_lookup_that_fails_is_reported_with_its_own_status():
    class DeniedLookup(FakeCommandRunner):
        def which(self, name, *, extra_paths=()):
            return Observed.failed(
                S.PERMISSION_DENIED,
                code="PermissionError",
                message="PATH entry not searchable",
                source=Source.OS,
                method="shutil.which(nvcc)",
            )

    result, _ = _locate(_toolkit(FakeToolkitFiles()), DeniedLookup())
    for field in (result.active_nvcc_path, result.active_nvcc_version):
        assert (field.status, field.error.code) == (S.PERMISSION_DENIED, "PermissionError")
        assert field.source is Source.CUDA_TOOLKIT and field.method == "fixture:shutil.which"
    assert result.toolkits.is_collected  # the default location still found the toolkit


def test_an_uncheckable_compiler_is_reported_and_carried_into_the_version():
    files = _toolkit(FakeToolkitFiles(), metadata=False)
    files.fail("kind", f"{ROOT}/bin/nvcc", PermissionError(errno.EACCES, "denied"))
    (candidate,) = _locate(files)[0].toolkits.value
    assert candidate.nvcc_path.status is S.PERMISSION_DENIED
    assert candidate.version == candidate.nvcc_path  # no evidence either way: not NOT_DETECTED
