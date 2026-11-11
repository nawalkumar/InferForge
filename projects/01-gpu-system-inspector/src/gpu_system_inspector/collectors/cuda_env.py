"""CUDA environment collector (architecture §4.6, IMP-K05, DEC-47): `CUDAEnvironmentInfo`.

It reads the allowlisted variables through the `PlatformAdapter` (plus, on Windows, the set
`CUDA_PATH_V*` names), hands the toolkit-related values to the `ToolkitLocator` as hints,
and reports what the locator found. Discovery itself (files, `nvcc`) is the locator's; this
collector never touches the filesystem or runs anything, and never uses the driver's CUDA
version (FR-07). It owns `CUDA.TOOLKIT`, recorded where toolkit discovery isn't supported.

A toolkit found here says nothing about whether CUDA works: driver compatibility, the CUDA
runtime, PyTorch and GPU execution are other sections' facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from gpu_system_inspector.adapters import Clock, PlatformAdapter, ToolkitLocator
from gpu_system_inspector.adapters.outcomes import PLATFORM_NOT_SUPPORTED
from gpu_system_inspector.collectors.environment import collect_environment
from gpu_system_inspector.collectors.system import FAILURE_STATUSES
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    CUDAEnvironmentInfo,
    ErrorDetail,
    Observed,
    PlatformProfile,
    ToolkitDiscovery,
    UnsupportedCapability,
)

CUDA_ENVIRONMENT_REPORT_PATH = "/cuda_environment"
CUDA_TOOLKITS_REPORT_PATH = "/cuda_environment/toolkits"
CUDA_TOOLKIT = "CUDA.TOOLKIT"
CUDA_TOOLKIT_REASONS = {
    PlatformProfile.MACOS: "The CUDA Toolkit is not available on macOS.",
    PlatformProfile.OTHER: "The CUDA Toolkit is not supported on this platform.",
}


class CudaEnvironmentCollector:
    name: ClassVar[str] = "cuda_environment"

    def __init__(self, locator: ToolkitLocator, platform: PlatformAdapter, clock: Clock) -> None:
        self._locator = locator
        self._platform = platform
        self._clock = clock

    def collect(self) -> CollectorOutcome[CUDAEnvironmentInfo]:
        started = self._clock.monotonic_ms()
        versioned = self._platform.versioned_cuda_path_names()
        environment = collect_environment(self._platform, (*ENVIRONMENT_ALLOWLIST, *versioned))
        hints = {
            ToolkitDiscovery.CUDA_HOME: _values(environment, ("CUDA_HOME",)),
            ToolkitDiscovery.CUDA_PATH: _values(environment, ("CUDA_PATH",)),
            ToolkitDiscovery.CUDA_PATH_VERSIONED: _values(environment, versioned),
        }
        result = self._locator.discover(hints)
        section = CUDAEnvironmentInfo(
            toolkits=result.toolkits,
            active_nvcc_path=result.active_nvcc_path,
            active_nvcc_version=result.active_nvcc_version,
            environment=environment,
        )
        errors = tuple(
            CollectionError(
                collector=self.name,
                device_ref=None,
                status=problem.status,
                error_code=problem.error.code,
                message=f"{problem.path}: {problem.error.message}",
                report_path=CUDA_TOOLKITS_REPORT_PATH,
            )
            for problem in result.problems
        )
        partial = bool(errors) or any(o.status in FAILURE_STATUSES for o in _observations(section))
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.PARTIAL if partial else CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(
            section=section, run=run, errors=errors, unsupported=self._unsupported(section)
        )

    def unavailable(self, error: ErrorDetail) -> CUDAEnvironmentInfo:
        return CUDAEnvironmentInfo.unavailable(CollectionStatus.COLLECTION_FAILED, error)

    def _unsupported(self, section: CUDAEnvironmentInfo) -> tuple[UnsupportedCapability, ...]:
        toolkits = section.toolkits
        if toolkits.status is not CollectionStatus.NOT_SUPPORTED or (
            toolkits.error is None or toolkits.error.code != PLATFORM_NOT_SUPPORTED
        ):
            return ()
        reason = CUDA_TOOLKIT_REASONS.get(
            self._platform.profile, CUDA_TOOLKIT_REASONS[PlatformProfile.OTHER]
        )
        return (
            UnsupportedCapability(
                capability=CUDA_TOOLKIT,
                status=CollectionStatus.NOT_SUPPORTED,
                device_ref=None,
                reason=reason,
            ),
        )


def _values(environment: Mapping[str, Observed[str]], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        value.value
        for name in names
        if (value := environment.get(name)) is not None and value.is_collected and value.value
    )


def _observations(section: CUDAEnvironmentInfo) -> list[Observed]:
    found = [section.toolkits, section.active_nvcc_path, section.active_nvcc_version]
    for candidate in section.toolkits.value or ():
        found += [candidate.version, candidate.nvcc_path]
    return found
