"""Fake `CommandRunner`: deterministic canned responses, never a real process.

Executables are available only if configured. Each response matches an exact argument
vector or, with `prefix=True`, any vector starting with it; the most specific match wins.
A command without a response fails the test, so unexpected (possibly mutating) commands
can't slip through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gpu_system_inspector.adapters import (
    MAX_COMMAND_OUTPUT_CHARS,
    NVIDIA_SMI_EXECUTABLE,
    AdapterUsageError,
    CommandResult,
    exception_outcome,
)
from gpu_system_inspector.adapters.outcomes import TIMEOUT
from gpu_system_inspector.models import (
    CollectionStatus,
    DataOrigin,
    InspectionSnapshot,
    Observed,
    Source,
)
from gsi_testkit.builders import fx
from gsi_testkit.fakes.toolkit import nvcc_version_output


@dataclass(frozen=True)
class CannedResponse:
    argv: tuple[str, ...]
    prefix: bool = False
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    raises: OSError | None = None
    duration_ms: int = 1

    def matches(self, argv: tuple[str, ...]) -> bool:
        if self.prefix:
            return argv[: len(self.argv)] == self.argv
        return argv == self.argv


@dataclass(frozen=True)
class CommandCall:
    argv: tuple[str, ...]
    timeout_s: float
    source: Source


@dataclass
class FakeCommandRunner:
    executables: Mapping[str, str] = field(default_factory=dict)
    files: frozenset[str] = frozenset()
    responses: list[CannedResponse] = field(default_factory=list)
    calls: list[CommandCall] = field(default_factory=list)
    lookups: list[str] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakeCommandRunner:
        """Executables as the fixture found them; canned only for `nvcc --version` (the
        `nvidia-smi` runs are canned by the tests that need them)."""
        smi = snapshot.driver.nvidia_smi_path
        runner = cls(executables={NVIDIA_SMI_EXECUTABLE: smi.value} if smi.is_collected else {})
        cuda = snapshot.cuda_environment
        if cuda.active_nvcc_path.is_collected and cuda.active_nvcc_path.value:
            nvcc = cuda.active_nvcc_path.value
            runner.executables = {**runner.executables, "nvcc": nvcc}
            if cuda.active_nvcc_version.is_collected and cuda.active_nvcc_version.value:
                runner.on(
                    nvcc, "--version", stdout=nvcc_version_output(cuda.active_nvcc_version.value)
                )
        for candidate in cuda.toolkits.value or () if cuda.toolkits.is_collected else ():
            version, nvcc_path = candidate.version, candidate.nvcc_path
            if (
                version.is_collected
                and (version.method or "").endswith("nvcc --version")
                and nvcc_path.is_collected
                and nvcc_path.value != cuda.active_nvcc_path.value
            ):
                runner.files = runner.files | {nvcc_path.value or ""}
                runner.on(
                    nvcc_path.value or "",
                    "--version",
                    stdout=nvcc_version_output(version.value or ""),
                )
        return runner

    def on(
        self,
        *argv: str,
        prefix: bool = False,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        timeout: bool = False,
        raises: OSError | None = None,
        duration_ms: int = 1,
    ) -> FakeCommandRunner:
        self.responses.append(
            CannedResponse(
                argv=tuple(argv),
                prefix=prefix,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                raises=raises,
                duration_ms=duration_ms,
            )
        )
        return self

    def which(self, name: str, *, extra_paths: Sequence[str] = ()) -> Observed[str]:
        self.lookups.append(name)
        method = fx(f"shutil.which({name})")
        if name in self.executables:
            return Observed.collected(self.executables[name], source=Source.OS, method=method)
        for directory in extra_paths:
            candidate = f"{directory.rstrip('/')}/{name}"
            if candidate in self.files:
                return Observed.collected(candidate, source=Source.OS, method=method)
        return Observed.failed(
            CollectionStatus.NOT_DETECTED,
            code="FileNotFoundError",
            message=f"{name} not found",
            source=Source.OS,
            method=method,
        )

    def _available(self, executable: str) -> bool:
        return (
            executable in self.executables
            or executable in self.executables.values()
            or executable in self.files
        )

    def run(
        self, argv: Sequence[str], *, timeout_s: float, source: Source
    ) -> Observed[CommandResult]:
        if isinstance(argv, str):
            raise AdapterUsageError("argv must be a sequence of arguments; a shell is never used")
        args = tuple(argv)
        if not args or not all(isinstance(a, str) and a for a in args):
            raise AdapterUsageError(f"argv must be non-empty strings, got {args!r}")
        if not timeout_s > 0:
            raise AdapterUsageError(f"timeout_s must be positive, got {timeout_s!r}")
        self.calls.append(CommandCall(args, timeout_s, source))
        method = fx(" ".join(args))
        if not self._available(args[0]):
            return exception_outcome(
                FileNotFoundError(f"{args[0]} not found"), source=source, method=method
            )
        response = self._response(args)
        if response.raises is not None:
            return exception_outcome(response.raises, source=source, method=method)
        if response.timeout:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=TIMEOUT,
                message=f"{args[0]} did not finish within {timeout_s} s",
                source=source,
                method=method,
            )
        stdout, stderr = response.stdout, response.stderr
        truncated = max(len(stdout), len(stderr)) > MAX_COMMAND_OUTPUT_CHARS
        result = CommandResult(
            argv=args,
            returncode=response.returncode,
            stdout=stdout[:MAX_COMMAND_OUTPUT_CHARS],
            stderr=stderr[:MAX_COMMAND_OUTPUT_CHARS],
            duration_ms=response.duration_ms,
            truncated=truncated,
        )
        return Observed.collected(result, source=source, method=method)

    def _response(self, argv: tuple[str, ...]) -> CannedResponse:
        matches = [r for r in self.responses if r.matches(argv)]
        if not matches:
            raise AssertionError(f"no canned response for {argv!r}")
        return max(matches, key=lambda r: (not r.prefix, len(r.argv)))
