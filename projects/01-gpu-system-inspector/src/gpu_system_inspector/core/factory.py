"""The production adapter set (architecture §4.5.4, DEC-44, DEC-45): the only place that
names production adapter classes.

The PCI scanner depends on the Linux environment, which is known only after the system
collector has run, so the core builds it with `production_pci_scanner(environment)`. Until
then the set's PCI slot holds `DeferredPciScanner`, which refuses to scan. NVML
(`PynvmlAdapter`), `nvidia-smi` (`NvidiaSmiCli` over the `SubprocessCommandRunner`) and the
command runner are the real adapters, and so is the CUDA Toolkit locator
(`CudaToolkitLocator` over the same runner and the real filesystem, DEC-47); nothing is
loaded, read or run until the core uses them. The PyTorch probe is `SubprocessTorchProbe`
(DEC-07), or `InProcessTorchProbe` when the options ask for `IN_PROCESS` (Python API only,
DEC-48); PyTorch itself is never imported here.
"""

from __future__ import annotations

from gpu_system_inspector.adapters import (
    AdapterSet,
    AdapterUsageError,
    CudaToolkitLocator,
    InProcessTorchProbe,
    NvidiaSmiCli,
    OsPlatformAdapter,
    PciScan,
    PciScanner,
    PciSysfsAdapter,
    PynvmlAdapter,
    SubprocessCommandRunner,
    SubprocessTorchProbe,
    SystemClock,
    TorchProbe,
)
from gpu_system_inspector.models import (
    DataOrigin,
    InspectionOptions,
    LinuxEnvironment,
    ProbeMode,
)

PRODUCTION_LIMITATIONS = (
    "PyTorch is inspected only in the Python interpreter running the inspector (KI-11).",
)


class DeferredPciScanner:
    """The production set's PCI slot before the environment is classified. Scanning it is a
    caller bug: the core must use `production_pci_scanner` instead."""

    provenance = DataOrigin.LIVE

    def scan(self) -> PciScan:
        raise AdapterUsageError(
            "the production PCI scanner is built from the classified Linux environment "
            "(production_pci_scanner); the deferred placeholder can't scan"
        )


def production_pci_scanner(environment: LinuxEnvironment | None) -> PciScanner:
    return PciSysfsAdapter(environment=environment)


def production_torch_probe(
    options: InspectionOptions, commands: SubprocessCommandRunner
) -> TorchProbe:
    if options.pytorch_probe_mode is ProbeMode.IN_PROCESS:
        return InProcessTorchProbe()
    return SubprocessTorchProbe(commands)


def build_production_adapters(options: InspectionOptions | None = None) -> AdapterSet:
    options = options or InspectionOptions()
    clock = SystemClock()
    platform = OsPlatformAdapter()
    commands = SubprocessCommandRunner(clock)
    return AdapterSet(
        platform=platform,
        pci=DeferredPciScanner(),
        nvml=PynvmlAdapter(platform=platform),
        nvidia_smi=NvidiaSmiCli(commands, platform, timeout_s=options.subprocess_timeout_s),
        commands=commands,
        toolkit=CudaToolkitLocator(commands, platform, timeout_s=options.subprocess_timeout_s),
        torch_probe=production_torch_probe(options, commands),
        clock=clock,
        provenance=DataOrigin.LIVE,
    )
