"""Production `TorchProbe`s (architecture §4.7, IMP-A07, DEC-48).

Both first read `importlib.metadata.version("torch")` (no import); `PackageNotFoundError` is
`NOT_INSTALLED`, and the caller then runs nothing. PyTorch is never a dependency (DEC-26).

- `SubprocessTorchProbe` (default; the CLI always): runs
  `[python, "-m", "gpu_system_inspector.adapters._torch_probe_main"]` through the shared
  `CommandRunner` with the given timeout, and decodes its JSON (`decode_probe_output`).
  The inspector's own process never imports torch or creates a CUDA context (DEC-07).
- `InProcessTorchProbe` (Python API opt-in): imports torch in the calling process and runs
  the same probe function. It can read the caller's allocator counters, which exist only
  if CUDA was initialised before the probe (FR-10, DEC-08). No timeout can be enforced, and
  reading device properties may initialise CUDA in the caller.

Nothing is imported, read or run until a method is called.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from gpu_system_inspector.adapters.outcomes import exception_outcome
from gpu_system_inspector.adapters.probe_protocol import (
    TorchAllocatorSample,
    TorchProbeReport,
    decode_probe_output,
    decode_probe_payload,
    torch_probe_argv,
)
from gpu_system_inspector.adapters.protocols import CommandRunner
from gpu_system_inspector.adapters.torch_probe_payload import import_failure, probe_payload
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    DataOrigin,
    Observed,
    Source,
)

TORCH_DISTRIBUTION = "torch"
DISTRIBUTION_METHOD = "importlib.metadata.version('torch')"
IN_PROCESS_METHOD = "in-process probe"
# The in-process counterpart of a crashing child (`PROBE_EXIT_<n>`).
PROBE_EXCEPTION = "PROBE_EXCEPTION"


def distribution_version_of(name: str) -> str:
    # Imported here: importlib.metadata pulls in email/socket, which plain imports must not.
    from importlib.metadata import version

    return version(name)


def import_torch() -> Any:
    return importlib.import_module("torch")


class _DistributionVersion:
    def __init__(self, lookup: Callable[[str], str], provenance: DataOrigin) -> None:
        self._lookup = lookup
        self._method = (FIXTURE_METHOD_PREFIX if provenance is DataOrigin.FIXTURE else "") + (
            DISTRIBUTION_METHOD
        )

    def __call__(self) -> Observed[str]:
        try:
            value = self._lookup(TORCH_DISTRIBUTION)
        except Exception as exc:  # PackageNotFoundError → NOT_INSTALLED; else its own status
            return exception_outcome(exc, source=Source.PYTORCH, method=self._method)
        return Observed.collected(value, source=Source.PYTORCH, method=self._method)


class SubprocessTorchProbe:
    def __init__(
        self,
        commands: CommandRunner,
        python_executable: str | None = None,
        *,
        distribution: Callable[[str], str] = distribution_version_of,
    ) -> None:
        self._commands = commands
        self._python = python_executable or sys.executable
        self.provenance: DataOrigin = getattr(commands, "provenance", DataOrigin.LIVE)
        self._distribution = _DistributionVersion(distribution, self.provenance)

    def distribution_version(self) -> Observed[str]:
        return self._distribution()

    def probe(self, timeout_s: float) -> Observed[TorchProbeReport]:
        outcome = self._commands.run(
            torch_probe_argv(self._python), timeout_s=timeout_s, source=Source.PYTORCH
        )
        decoded = decode_probe_output(outcome)
        if not decoded.is_collected:
            return decoded
        assert decoded.value is not None and outcome.value is not None
        return replace(decoded, value=replace(decoded.value, duration_ms=outcome.value.duration_ms))


class InProcessTorchProbe:
    def __init__(
        self,
        loader: Callable[[], Any] = import_torch,
        *,
        distribution: Callable[[str], str] = distribution_version_of,
        provenance: DataOrigin = DataOrigin.LIVE,
    ) -> None:
        self._loader = loader
        self.provenance = provenance
        self._distribution = _DistributionVersion(distribution, provenance)
        self._method = (FIXTURE_METHOD_PREFIX if provenance is DataOrigin.FIXTURE else "") + (
            IN_PROCESS_METHOD
        )

    def distribution_version(self) -> Observed[str]:
        return self._distribution()

    def probe(self, timeout_s: float) -> Observed[TorchProbeReport]:
        """`timeout_s` can't be enforced in-process; it is accepted for the protocol."""
        started = time.monotonic()
        try:
            torch = self._loader()
        except Exception as exc:  # installed but not importable
            payload = import_failure(exc)
            return decode_probe_payload(payload, source=Source.PYTORCH, method=self._method)
        try:
            initialised = bool(torch.cuda.is_initialized())  # before the probe may initialise it
            payload = probe_payload(torch)
        except Exception as exc:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=PROBE_EXCEPTION,
                message=f"{type(exc).__name__}: {exc}",
                source=Source.PYTORCH,
                method=self._method,
            )
        decoded = decode_probe_payload(payload, source=Source.PYTORCH, method=self._method)
        if not decoded.is_collected:
            return decoded
        report = decoded.value
        assert report is not None
        samples = (
            tuple(_allocator(torch, device.index) for device in report.devices)
            if initialised
            else None
        )
        elapsed = max(0, round((time.monotonic() - started) * 1000))
        return replace(decoded, value=replace(report, allocator=samples, duration_ms=elapsed))


def _allocator(torch: Any, index: int) -> TorchAllocatorSample:
    cuda = torch.cuda
    return TorchAllocatorSample(
        index=index,
        allocated=int(cuda.memory_allocated(index)),
        reserved=int(cuda.memory_reserved(index)),
        peak_allocated=int(cuda.max_memory_allocated(index)),
        peak_reserved=int(cuda.max_memory_reserved(index)),
    )
