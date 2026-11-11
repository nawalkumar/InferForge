"""The PyTorch probe function (architecture §4.7, data flow §5, DEC-48), shared by the probe
child (`_torch_probe_main`) and `InProcessTorchProbe`.

It reads versions and build flags, calls `torch.cuda.is_available()` with its warnings
captured, counts devices and reads their properties. It allocates no tensor, runs no CUDA
kernel and never calls `mem_get_info`: `cuda_available` is PyTorch's own answer, not a proof
that CUDA code runs. A failure while reading device properties is `device_error`, so the
version and build facts survive it; anything else unexpected propagates.

Kept apart from `_torch_probe_main` so importing the adapters package never imports the
child's entry module (which `python -m` would then execute a second time).
"""

from __future__ import annotations

import warnings
from typing import Any

from gpu_system_inspector.models import PROBE_PROTOCOL_VERSION


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def import_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "probe_protocol": PROBE_PROTOCOL_VERSION,
        "import_ok": False,
        "import_error": _failure(exc),
    }


def probe_payload(torch: Any) -> dict[str, Any]:
    """The protocol document for an imported `torch` module."""
    build_cuda = getattr(torch.version, "cuda", None)
    hip = getattr(torch.version, "hip", None)
    cudnn = torch.backends.cudnn.version() if build_cuda is not None else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    devices: list[dict[str, Any]] = []
    device_error = None
    if count:  # also when unavailable: then reading a device fails, and says why
        try:
            devices = [_device(torch, index) for index in range(count)]
        except Exception as exc:  # a CUDA error enumerating devices keeps the build facts
            devices, device_error = [], _failure(exc)
    return {
        "probe_protocol": PROBE_PROTOCOL_VERSION,
        "import_ok": True,
        "import_error": None,
        "torch_version": str(torch.__version__),
        "cuda_build": build_cuda is not None,
        "build_cuda_version": None if build_cuda is None else str(build_cuda),
        "hip_version": None if hip is None else str(hip),
        "cudnn_version": None if cudnn is None else str(cudnn),
        "cuda_available": available,
        "cuda_init_warnings": [str(w.message) for w in caught],
        "device_count": count,
        "devices": devices,
        "device_error": device_error,
    }


def _device(torch: Any, index: int) -> dict[str, Any]:
    cuda = torch.cuda
    properties = cuda.get_device_properties(index)
    major, minor = cuda.get_device_capability(index)
    uuid = getattr(properties, "uuid", None)
    pci = [getattr(properties, f"pci_{part}_id", None) for part in ("domain", "bus", "device")]
    return {
        "index": index,
        "name": str(cuda.get_device_name(index)),
        "capability": f"{major}.{minor}",
        "total_memory": _int(getattr(properties, "total_memory", None)),
        "multiprocessor_count": _int(getattr(properties, "multi_processor_count", None)),
        "uuid": None if uuid is None else str(uuid),
        "pci_bus_id": None if None in pci else f"{pci[0]:04x}:{pci[1]:02x}:{pci[2]:02x}.0",
    }


def _int(value: Any) -> int | None:
    return None if value is None else int(value)
