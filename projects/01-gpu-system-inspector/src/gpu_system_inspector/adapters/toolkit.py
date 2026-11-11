"""Production `ToolkitLocator` (architecture §4.6, IMP-A06, DEC-47).

`CudaToolkitLocator` finds CUDA Toolkit installations from the signals DEC-47 allows:
the environment hints the caller passes (`CUDA_HOME`, `CUDA_PATH`, Windows `CUDA_PATH_V*`),
`nvcc` on `PATH` (through the shared `CommandRunner`) and, on Linux and WSL2, one
non-recursive listing of `/usr/local`. Each candidate's version comes from `version.json`
or, without it, `<root>/bin/nvcc --version`; never from a directory name or the driver.

File access goes through `ToolkitFiles`, a seam only this locator uses: type checks that
report errors instead of hiding them (`os.path.isdir` would turn a permission error into
"absent"), directory listing, bounded reads and symlink resolution. Commands go through the
runner, so no process is started here directly. Nothing is read or run until `discover()`.

Toolkit discovery is not CUDA execution validation: a toolkit directory or `nvcc` says
nothing about whether CUDA works on this machine.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from gpu_system_inspector.adapters.outcomes import (
    PARSE_ERROR,
    PLATFORM_NOT_SUPPORTED,
    AdapterUsageError,
    exception_outcome,
    excerpt,
    require_success,
)
from gpu_system_inspector.adapters.protocols import (
    TOOLKIT_HINT_KINDS,
    CommandRunner,
    PlatformAdapter,
    ToolkitDiscoveryResult,
    ToolkitProblem,
)
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    CudaToolkitCandidate,
    DataOrigin,
    ErrorDetail,
    Observed,
    PlatformProfile,
    Source,
    ToolkitDiscovery,
)

NVCC = "nvcc"
DEFAULT_TOOLKIT_PARENT = "/usr/local"
DEFAULT_TOOLKIT_NAME = re.compile(r"^cuda(-[0-9][0-9.]*)?$")
VERSION_FILE = "version.json"
VERSION_FILE_LIMIT = 64 * 1024
VERSION_VALUE = re.compile(r"^[0-9]+(\.[0-9]+)+$")
NVCC_FULL_RELEASE = re.compile(r"release ([0-9]+\.[0-9]+), V([0-9]+(?:\.[0-9]+)+)")
NVCC_RELEASE = re.compile(r"release ([0-9]+\.[0-9]+)")
DEFAULT_TIMEOUT_S = 10.0
DISCOVERY_METHOD = "toolkit discovery"
VERSION_FILE_METHOD = "version.json:cuda.version"
NVCC_VERSION_METHOD = "nvcc --version"
LAYOUT_METHOD = "toolkit layout"
WHICH_METHOD = "shutil.which"
NO_TOOLKIT_FOUND = "NO_TOOLKIT_FOUND"
NO_VERSION_EVIDENCE = "NO_VERSION_EVIDENCE"
NVCC_NOT_IN_TOOLKIT = "NVCC_NOT_IN_TOOLKIT"
NOT_AVAILABLE = "CUDA Toolkit is not available on this platform"
UNSUPPORTED_PROFILES = frozenset({PlatformProfile.MACOS, PlatformProfile.OTHER})
DIRECTORY, FILE, OTHER = "directory", "file", "other"


@runtime_checkable
class ToolkitFiles(Protocol):
    """Read-only file access for toolkit discovery. `kind` follows symlinks and returns
    `None` for a missing path; every other `OSError` propagates, so a permission problem is
    never mistaken for absence."""

    def kind(self, path: str) -> str | None: ...
    def list_dir(self, path: str) -> list[str]: ...
    def read_text(self, path: str, limit: int) -> str: ...
    def realpath(self, path: str) -> str: ...
    def join(self, *parts: str) -> str: ...
    def dirname(self, path: str) -> str: ...
    def basename(self, path: str) -> str: ...


class StdlibToolkitFiles:
    """The real filesystem, through `os` only."""

    def kind(self, path: str) -> str | None:
        try:
            mode = os.stat(path).st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISDIR(mode):
            return DIRECTORY
        return FILE if stat.S_ISREG(mode) else OTHER

    def list_dir(self, path: str) -> list[str]:
        return os.listdir(path)

    def read_text(self, path: str, limit: int) -> str:
        """At most `limit + 1` characters, so the caller can tell the file was too long."""
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit + 1)

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def join(self, *parts: str) -> str:
        return os.path.join(*parts)

    def dirname(self, path: str) -> str:
        return os.path.dirname(path)

    def basename(self, path: str) -> str:
        return os.path.basename(path)


class CudaToolkitLocator:
    def __init__(
        self,
        commands: CommandRunner,
        platform: PlatformAdapter,
        files: ToolkitFiles | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._commands = commands
        self._platform = platform
        self._files = files if files is not None else StdlibToolkitFiles()
        self._timeout_s = timeout_s
        self.provenance: DataOrigin = getattr(commands, "provenance", DataOrigin.LIVE)
        self._prefix = FIXTURE_METHOD_PREFIX if self.provenance is DataOrigin.FIXTURE else ""

    def discover(self, hints: Mapping[ToolkitDiscovery, Sequence[str]]) -> ToolkitDiscoveryResult:
        unknown = set(hints) - TOOLKIT_HINT_KINDS
        if unknown:
            raise AdapterUsageError(f"not environment hints: {sorted(unknown)}")
        profile = self._platform.profile
        if profile in UNSUPPORTED_PROFILES:
            unsupported = self._failed(
                CollectionStatus.NOT_SUPPORTED, PLATFORM_NOT_SUPPORTED, NOT_AVAILABLE
            )
            return ToolkitDiscoveryResult(
                toolkits=unsupported, active_nvcc_path=unsupported, active_nvcc_version=unsupported
            )
        return _Discovery(self, windows=profile is PlatformProfile.WINDOWS).run(hints)

    # --- helpers shared with the per-call state ------------------------------------------

    def _failed(
        self, status: CollectionStatus, code: str, message: str, method: str = DISCOVERY_METHOD
    ) -> Observed:
        return Observed.failed(
            status,
            code=code,
            message=message,
            source=Source.CUDA_TOOLKIT,
            method=self._method(method),
        )

    def _method(self, method: str) -> str:
        return self._prefix + method


class _Discovery:
    """One `discover()` call: the roots found so far and the `nvcc` runs already made."""

    def __init__(self, locator: CudaToolkitLocator, *, windows: bool) -> None:
        self._locator = locator
        self._files = locator._files
        self._windows = windows
        self._nvcc = NVCC + ".exe" if windows else NVCC
        self._roots: dict[str, set[ToolkitDiscovery]] = {}
        self._root_errors: dict[str, Observed] = {}
        self._runs: dict[str, Observed[str]] = {}
        self._problems: list[ToolkitProblem] = []

    def run(self, hints: Mapping[ToolkitDiscovery, Sequence[str]]) -> ToolkitDiscoveryResult:
        for kind in ToolkitDiscovery:
            for path in hints.get(kind, ()):
                if path:
                    self._add(path, kind)
        active_path, active_version = self._active_nvcc()
        if not self._windows:
            self._defaults()
        candidates = tuple(self._candidate(root) for root in sorted(self._roots))
        if candidates:
            toolkits = Observed.collected(
                candidates,
                source=Source.CUDA_TOOLKIT,
                method=self._locator._method(DISCOVERY_METHOD),
            )
        elif self._problems:
            problem = self._problems[0]
            toolkits = self._locator._failed(
                problem.status, problem.error.code, problem.error.message
            )
        else:
            toolkits = self._locator._failed(
                CollectionStatus.NOT_DETECTED, NO_TOOLKIT_FOUND, "no CUDA Toolkit candidates"
            )
        return ToolkitDiscoveryResult(
            toolkits=toolkits,
            active_nvcc_path=active_path,
            active_nvcc_version=active_version,
            problems=tuple(self._problems),
        )

    # --- signals --------------------------------------------------------------------------

    def _add(self, path: str, signal: ToolkitDiscovery) -> None:
        """A signalled directory becomes a candidate under its resolved path."""
        root = self._files.realpath(path)
        try:
            kind = self._files.kind(root)
        except OSError as exc:
            self._root_errors.setdefault(root, self._error(exc, DISCOVERY_METHOD))
            kind = DIRECTORY
        if kind == DIRECTORY:
            self._roots.setdefault(root, set()).add(signal)

    def _active_nvcc(self) -> tuple[Observed[str], Observed[str]]:
        found = self._locator._commands.which(NVCC)
        if not found.is_collected:
            if found.status is CollectionStatus.NOT_DETECTED:
                missing = self._locator._failed(
                    CollectionStatus.NOT_DETECTED,
                    "FileNotFoundError",
                    "nvcc not found",
                    WHICH_METHOD,
                )
                return missing, missing
            assert found.error is not None
            failure = self._locator._failed(
                found.status, found.error.code, found.error.message, WHICH_METHOD
            )
            return failure, failure
        assert found.value is not None
        path = Observed.collected(
            found.value, source=Source.CUDA_TOOLKIT, method=self._locator._method(WHICH_METHOD)
        )
        resolved = self._files.realpath(found.value)
        bin_dir = self._files.dirname(resolved)
        if self._files.basename(bin_dir) == "bin":
            self._add(self._files.dirname(bin_dir), ToolkitDiscovery.NVCC_ON_PATH)
        return path, self._nvcc_version(found.value)

    def _defaults(self) -> None:
        try:
            names = self._files.list_dir(DEFAULT_TOOLKIT_PARENT)
        except (FileNotFoundError, NotADirectoryError):
            return
        except OSError as exc:
            failure = self._error(exc, DISCOVERY_METHOD)
            assert failure.error is not None
            self._problems.append(
                ToolkitProblem(
                    path=DEFAULT_TOOLKIT_PARENT, status=failure.status, error=failure.error
                )
            )
            return
        for name in sorted(names):
            if DEFAULT_TOOLKIT_NAME.match(name):
                self._add(
                    self._files.join(DEFAULT_TOOLKIT_PARENT, name),
                    ToolkitDiscovery.DEFAULT_LOCATION,
                )

    # --- one candidate --------------------------------------------------------------------

    def _candidate(self, root: str) -> CudaToolkitCandidate:
        signals = tuple(self._roots[root])
        failure = self._root_errors.get(root)
        if failure is not None:  # the root's type couldn't be checked
            return CudaToolkitCandidate(
                root_path=root, discovered_via=signals, version=failure, nvcc_path=failure
            )
        nvcc_path = self._files.join(root, "bin", self._nvcc)
        nvcc = self._nvcc_path(nvcc_path)
        return CudaToolkitCandidate(
            root_path=root,
            discovered_via=signals,
            version=self._version(root, nvcc_path, nvcc),
            nvcc_path=nvcc,
        )

    def _nvcc_path(self, path: str) -> Observed[str]:
        try:
            kind = self._files.kind(path)
        except OSError as exc:
            return self._error(exc, LAYOUT_METHOD)
        if kind == FILE:
            return Observed.collected(
                path, source=Source.CUDA_TOOLKIT, method=self._locator._method(LAYOUT_METHOD)
            )
        return self._locator._failed(
            CollectionStatus.NOT_DETECTED,
            NVCC_NOT_IN_TOOLKIT,
            f"no {self._nvcc} in the toolkit's bin directory",
            LAYOUT_METHOD,
        )

    def _version(self, root: str, nvcc_path: str, nvcc: Observed[str]) -> Observed[str]:
        path = self._files.join(root, VERSION_FILE)
        try:
            text = self._files.read_text(path, VERSION_FILE_LIMIT)
        except (FileNotFoundError, NotADirectoryError):
            if nvcc.is_collected:
                return self._nvcc_version(nvcc_path)
            if nvcc.status is CollectionStatus.NOT_DETECTED:
                return self._locator._failed(
                    CollectionStatus.NOT_DETECTED,
                    NO_VERSION_EVIDENCE,
                    f"no {VERSION_FILE} and no {self._nvcc} in the toolkit",
                )
            return nvcc
        except OSError as exc:
            return self._error(exc, VERSION_FILE_METHOD)
        return self._parse_version_file(text)

    def _parse_version_file(self, text: str) -> Observed[str]:
        def malformed(why: str) -> Observed[str]:
            return self._locator._failed(
                CollectionStatus.COLLECTION_FAILED, PARSE_ERROR, why, VERSION_FILE_METHOD
            )

        if len(text) > VERSION_FILE_LIMIT:
            return malformed(f"{VERSION_FILE} is larger than {VERSION_FILE_LIMIT} bytes")
        try:
            data = json.loads(text)
        except ValueError as exc:
            return malformed(f"{VERSION_FILE} isn't valid JSON: {exc}")
        cuda = data.get("cuda") if isinstance(data, dict) else None
        version = cuda.get("version") if isinstance(cuda, dict) else None
        if not isinstance(version, str) or VERSION_VALUE.match(version) is None:
            return malformed(f"{VERSION_FILE} has no valid .cuda.version: {version!r}")
        return Observed.collected(
            version,
            source=Source.CUDA_TOOLKIT,
            method=self._locator._method(VERSION_FILE_METHOD),
        )

    # --- nvcc -----------------------------------------------------------------------------

    def _nvcc_version(self, executable: str) -> Observed[str]:
        """`<executable> --version`, run at most once per resolved executable."""
        key = self._files.realpath(executable)
        if key not in self._runs:
            self._runs[key] = self._run_nvcc(executable)
        return self._runs[key]

    def _run_nvcc(self, executable: str) -> Observed[str]:
        locator = self._locator
        outcome = require_success(
            locator._commands.run(
                (executable, "--version"), timeout_s=locator._timeout_s, source=Source.CUDA_TOOLKIT
            )
        )
        if not outcome.is_collected:
            assert outcome.error is not None
            return locator._failed(
                outcome.status, outcome.error.code, outcome.error.message, NVCC_VERSION_METHOD
            )
        assert outcome.value is not None
        text = outcome.value.stdout
        full = NVCC_FULL_RELEASE.search(text)
        release = full.group(2) if full else None
        if release is None:
            short = NVCC_RELEASE.search(text)
            release = short.group(1) if short else None
        if release is None:
            return locator._failed(
                CollectionStatus.COLLECTION_FAILED,
                PARSE_ERROR,
                f"unrecognised nvcc --version output: {excerpt(text) or '(empty)'}",
                NVCC_VERSION_METHOD,
            )
        return Observed.collected(
            release, source=Source.CUDA_TOOLKIT, method=locator._method(NVCC_VERSION_METHOD)
        )

    def _error(self, exc: OSError, method: str) -> Observed:
        failure = exception_outcome(
            exc, source=Source.CUDA_TOOLKIT, method=self._locator._method(method)
        )
        assert isinstance(failure.error, ErrorDetail)
        return failure
