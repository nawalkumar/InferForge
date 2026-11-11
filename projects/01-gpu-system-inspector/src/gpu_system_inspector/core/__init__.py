"""The inspection core (architecture §4.2, §4.11, DEC-44): collector order, the guard, the
NVML session, fail-closed environment resolution, snapshot aggregation and the production
adapter set. The public entry point is `gpu_system_inspector.api.inspect()`."""

from gpu_system_inspector.core.environment import resolve_linux_environment
from gpu_system_inspector.core.factory import (
    PRODUCTION_LIMITATIONS,
    DeferredPciScanner,
    build_production_adapters,
    production_pci_scanner,
)
from gpu_system_inspector.core.guard import describe, run_guarded
from gpu_system_inspector.core.inspector import (
    NOT_IMPLEMENTED_LIMITATION,
    FailedNvmlSession,
    Inspector,
    NvmlSessionStep,
    PciScannerFactory,
)

__all__ = [
    "NOT_IMPLEMENTED_LIMITATION",
    "PRODUCTION_LIMITATIONS",
    "DeferredPciScanner",
    "FailedNvmlSession",
    "Inspector",
    "NvmlSessionStep",
    "PciScannerFactory",
    "build_production_adapters",
    "describe",
    "production_pci_scanner",
    "resolve_linux_environment",
    "run_guarded",
]
