"""Public Python API (architecture §4.2, DEC-44).

`inspect()` runs one live, read-only inspection with the production adapters and returns a
model-validated `InspectionReport` (Mode A). Serialise it with `reporting.render_json`.
Nothing is installed or written, and no device setting is changed. What it may touch:

- **NVML**, in this process: `pynvml` is imported and `nvmlInit` called once (not on macOS),
  read-only queries only, and, if the init succeeded, `nvmlShutdown` once after the
  NVML-backed steps.
- **`nvidia-smi`**, if found on `PATH` (or `/usr/lib/wsl/lib` on WSL2): its plain header is
  run once for the driver section; its `--query-gpu` query runs at most once, only when
  NVML can't count the devices or report the driver version.
- **CUDA Toolkit discovery**: allowlisted environment variables, `nvcc` on `PATH`, one
  listing of `/usr/local` (Linux, WSL2) and `version.json` reads; `nvcc --version` may run
  for the active `nvcc` and for a toolkit without `version.json`.
- **PyTorch**, only if the `torch` distribution is installed and `include_pytorch` is set:
  by default a child process of this interpreter (`python -m
  gpu_system_inspector.adapters._torch_probe_main`, bounded by `pytorch_timeout_s`) imports
  torch, reads its build facts, calls `torch.cuda.is_available()` and reads device
  properties, which may initialise CUDA in that child; no tensor is allocated and no CUDA
  code is run. With `pytorch_probe_mode = IN_PROCESS` the same probe runs in this process
  instead and may initialise CUDA here.
- **PCI**: read-only sysfs, on native Linux only.

Every command runs through one runner: argument lists, no shell, bounded timeouts.
"""

from __future__ import annotations

from gpu_system_inspector.core import (
    PRODUCTION_LIMITATIONS,
    Inspector,
    build_production_adapters,
    production_pci_scanner,
)
from gpu_system_inspector.models import InspectionOptions, InspectionReport

__all__ = ["InspectionOptions", "InspectionReport", "inspect"]


def inspect(options: InspectionOptions | None = None) -> InspectionReport:
    inspector = Inspector(
        build_production_adapters(options),
        pci_scanner_factory=production_pci_scanner,
        limitations=PRODUCTION_LIMITATIONS,
    )
    return inspector.run(options)
