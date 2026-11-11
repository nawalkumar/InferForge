"""The adapters one inspection uses (docs/architecture.md §4.5.4).

Built once per inspection and passed explicitly to the inspection core; there is no
registry or module-level instance. Every adapter is always present: an unavailable
capability is reported through the adapter's outcomes, never by a missing adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from gpu_system_inspector.adapters.protocols import (
    Clock,
    CommandRunner,
    NvidiaSmiAdapter,
    NvmlAdapter,
    PciScanner,
    PlatformAdapter,
    ToolkitLocator,
    TorchProbe,
)
from gpu_system_inspector.models import DataOrigin


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterSet:
    """`provenance` becomes the report's `data_origin`: `LIVE` for production adapters,
    `FIXTURE` for test fakes (DEC-17). An adapter that declares its own `provenance` must
    agree with the set, so synthetic and live adapters can't be mixed."""

    platform: PlatformAdapter
    pci: PciScanner
    nvml: NvmlAdapter
    nvidia_smi: NvidiaSmiAdapter
    commands: CommandRunner
    toolkit: ToolkitLocator
    torch_probe: TorchProbe
    clock: Clock
    provenance: DataOrigin

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, DataOrigin):
            raise TypeError(f"AdapterSet.provenance must be a DataOrigin, got {self.provenance!r}")
        for f in fields(self):
            if f.name == "provenance":
                continue
            adapter, protocol = getattr(self, f.name), _PROTOCOLS[f.name]
            if not isinstance(adapter, protocol):
                raise TypeError(f"AdapterSet.{f.name} doesn't implement {protocol.__name__}")
            declared = getattr(adapter, "provenance", self.provenance)
            if declared is not self.provenance:
                raise TypeError(
                    f"AdapterSet.{f.name} has provenance {declared}, "
                    f"but the set is {self.provenance}"
                )


_PROTOCOLS: dict[str, type] = {
    "platform": PlatformAdapter,
    "pci": PciScanner,
    "nvml": NvmlAdapter,
    "nvidia_smi": NvidiaSmiAdapter,
    "commands": CommandRunner,
    "toolkit": ToolkitLocator,
    "torch_probe": TorchProbe,
    "clock": Clock,
}
