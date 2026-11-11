"""Production stand-ins for adapters that aren't implemented yet (DEC-44).

The production `AdapterSet` must hold an implementation of every protocol (architecture
§4.5.4), but the NVML (IMP-A01), `nvidia-smi` (IMP-A02), command-runner (IMP-A05) and
PyTorch-probe (IMP-A07) adapters don't exist yet. These null objects fill those slots
honestly: every operation is `NOT_SUPPORTED` with code `NOT_IMPLEMENTED` and no method
(nothing is attempted). They load no library, start no process and never raise. Each is
replaced by its real adapter when that milestone lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

from gpu_system_inspector.adapters.outcomes import NOT_IMPLEMENTED, CommandResult
from gpu_system_inspector.adapters.probe_protocol import TorchProbeReport
from gpu_system_inspector.adapters.protocols import NvmlDevice, SmiGpuRow, SmiVersion
from gpu_system_inspector.models import CollectionStatus, DataOrigin, Observed, Source

NVML_NOT_IMPLEMENTED = "The NVML adapter isn't implemented in this version (IMP-A01)."
NVIDIA_SMI_NOT_IMPLEMENTED = "The nvidia-smi adapter isn't implemented in this version (IMP-A02)."
COMMANDS_NOT_IMPLEMENTED = "The command runner isn't implemented in this version (IMP-A05)."
TORCH_PROBE_NOT_IMPLEMENTED = "The PyTorch probe isn't implemented in this version (IMP-A07)."


def not_implemented(message: str, source: Source, unit: str | None = None) -> Observed:
    return Observed.failed(
        CollectionStatus.NOT_SUPPORTED,
        code=NOT_IMPLEMENTED,
        message=message,
        source=source,
        method=None,
        unit=unit,
    )


def _nvml() -> Observed:
    return not_implemented(NVML_NOT_IMPLEMENTED, Source.NVML)


class UnimplementedNvmlSession:
    """A session that never initialised NVML: every query reports the stand-in outcome."""

    def __init__(self) -> None:
        self._open = True

    @property
    def init_result(self) -> Observed[str]:
        return _nvml()

    @property
    def is_open(self) -> bool:
        return self._open

    def driver_version(self) -> Observed[str]:
        return _nvml()

    def cuda_driver_version(self) -> Observed[str]:
        return _nvml()

    def cuda_driver_library(self) -> Observed[str]:
        return _nvml()

    def device_count(self) -> Observed[int]:
        return _nvml()

    def device(self, index: int) -> Observed[NvmlDevice]:
        return _nvml()

    def close(self) -> None:
        self._open = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class UnimplementedNvmlAdapter:
    provenance = DataOrigin.LIVE

    def binding_version(self) -> Observed[str]:
        return not_implemented(NVML_NOT_IMPLEMENTED, Source.PLATFORM)

    def binding_distribution(self) -> Observed[str]:
        return not_implemented(NVML_NOT_IMPLEMENTED, Source.PLATFORM)

    def open(self) -> UnimplementedNvmlSession:
        return UnimplementedNvmlSession()


class UnimplementedNvidiaSmi:
    provenance = DataOrigin.LIVE

    def locate(self) -> Observed[str]:
        return not_implemented(NVIDIA_SMI_NOT_IMPLEMENTED, Source.NVIDIA_SMI)

    def version(self) -> SmiVersion:
        missing = not_implemented(NVIDIA_SMI_NOT_IMPLEMENTED, Source.NVIDIA_SMI)
        return SmiVersion(driver_version=missing, cuda_version=missing)

    def query_gpus(self) -> Observed[tuple[SmiGpuRow, ...]]:
        return not_implemented(NVIDIA_SMI_NOT_IMPLEMENTED, Source.NVIDIA_SMI)


class UnimplementedCommandRunner:
    provenance = DataOrigin.LIVE

    def which(self, name: str, *, extra_paths: Sequence[str] = ()) -> Observed[str]:
        return not_implemented(COMMANDS_NOT_IMPLEMENTED, Source.PLATFORM)

    def run(
        self, argv: Sequence[str], *, timeout_s: float, source: Source
    ) -> Observed[CommandResult]:
        return not_implemented(COMMANDS_NOT_IMPLEMENTED, source)


class UnimplementedTorchProbe:
    provenance = DataOrigin.LIVE

    def distribution_version(self) -> Observed[str]:
        return not_implemented(TORCH_PROBE_NOT_IMPLEMENTED, Source.PYTORCH)

    def probe(self, timeout_s: float) -> Observed[TorchProbeReport]:
        return not_implemented(TORCH_PROBE_NOT_IMPLEMENTED, Source.PYTORCH)
