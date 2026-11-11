"""Production `CommandRunner` (architecture §4.5.5, IMP-A05): fixed argument vectors, no shell.

`run()` executes `subprocess.run(list(argv), shell=False, stdin=DEVNULL, capture_output=True,
timeout=…)`: the child gets no input, stdout and stderr are captured (decoded as UTF-8 with
replacement), and on timeout `subprocess.run` kills the child before raising, so no process
outlives the call. Any process that ran is `COLLECTED`, whatever its exit status (callers
decide, e.g. with `require_success`); a missing executable is `NOT_DETECTED`, denied
execution `PERMISSION_DENIED`, a timeout `COLLECTION_FAILED`/`TIMEOUT`, any other `OSError`
`COLLECTION_FAILED`. A malformed argument vector or timeout is the caller's bug
(`AdapterUsageError`); any other exception propagates to the collector guard.

Only the argument vector and its outputs are recorded, never the environment. `subprocess`
and `shutil` are imported on first use, so importing the package starts nothing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from gpu_system_inspector.adapters.clock import SystemClock
from gpu_system_inspector.adapters.outcomes import (
    MAX_COMMAND_OUTPUT_CHARS,
    TIMEOUT,
    AdapterUsageError,
    CommandResult,
    exception_outcome,
)
from gpu_system_inspector.adapters.protocols import Clock
from gpu_system_inspector.models import CollectionStatus, DataOrigin, Observed, Source


def check_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise AdapterUsageError("argv must be a sequence of arguments; a shell is never used")
    args = tuple(argv)
    if not args or not all(isinstance(a, str) for a in args) or not args[0]:
        raise AdapterUsageError(f"argv must be non-empty strings, got {args!r}")
    if any("\0" in a for a in args):
        raise AdapterUsageError("argv must not contain NUL characters")
    return args


def check_timeout(timeout_s: float) -> None:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not timeout_s > 0:
        raise AdapterUsageError(f"timeout_s must be a positive number, got {timeout_s!r}")


class SubprocessCommandRunner:
    provenance = DataOrigin.LIVE

    def __init__(
        self, clock: Clock | None = None, *, max_output_chars: int = MAX_COMMAND_OUTPUT_CHARS
    ) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._max_output = max_output_chars

    def which(self, name: str, *, extra_paths: Sequence[str] = ()) -> Observed[str]:
        """`PATH` first (`shutil.which`), then each directory in `extra_paths`."""
        import shutil

        if not isinstance(name, str) or not name or os.sep in name or "\0" in name:
            raise AdapterUsageError(f"{name!r} is not an executable name")
        method = f"shutil.which({name})"
        found = shutil.which(name)
        if found is None:
            for directory in extra_paths:
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    found = candidate
                    break
        if found is None:
            where = " or ".join(["PATH", *extra_paths])
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code="FileNotFoundError",
                message=f"{name} not found on {where}",
                source=Source.OS,
                method=method,
            )
        return Observed.collected(found, source=Source.OS, method=method)

    def run(
        self, argv: Sequence[str], *, timeout_s: float, source: Source
    ) -> Observed[CommandResult]:
        import subprocess

        args = check_argv(argv)
        check_timeout(timeout_s)
        method = " ".join(args)
        started = self._clock.monotonic_ms()
        try:
            completed = subprocess.run(
                list(args),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=TIMEOUT,
                message=f"{args[0]} did not finish within {timeout_s} s",
                source=source,
                method=method,
            )
        except OSError as exc:
            return exception_outcome(exc, source=source, method=method)
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        if not isinstance(returncode, int) or not isinstance(stdout, str | None):
            raise TypeError(f"subprocess.run returned an unexpected result: {completed!r}")
        if not isinstance(stderr, str | None):
            raise TypeError(f"subprocess.run returned an unexpected result: {completed!r}")
        stdout, stderr = stdout or "", stderr or ""
        result = CommandResult(
            argv=args,
            returncode=returncode,
            stdout=stdout[: self._max_output],
            stderr=stderr[: self._max_output],
            duration_ms=max(0, self._clock.monotonic_ms() - started),
            truncated=max(len(stdout), len(stderr)) > self._max_output,
        )
        return Observed.collected(result, source=source, method=method)
