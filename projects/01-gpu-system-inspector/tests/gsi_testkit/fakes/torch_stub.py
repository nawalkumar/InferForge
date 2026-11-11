"""A stub `torch` for the PyTorch probe tests: never the real package, never CUDA.

`StubTorch` answers exactly the calls the probe makes (versions, `cuda.is_available`,
`device_count`, device properties, `is_initialized` and the allocator counters), with
configurable failures. `write_stub_torch_package` writes the same stub as an importable
`torch` package into a directory, for the probe child run in a real subprocess.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass
class StubDevice:
    name: str = "Synthetic GPU A (stub)"
    capability: tuple[int, int] = (8, 9)
    total_memory: int = 24 * 2**30
    multi_processor_count: int = 128
    uuid: str | None = "aaaaaaaa-0000-4000-8000-00000000000a"
    pci: tuple[int, int, int] | None = (0, 1, 0)
    allocated: int = 0


@dataclass
class _Cuda:
    torch: StubTorch

    def is_available(self) -> bool:
        self.torch.calls.append("cuda.is_available")
        for message in self.torch.init_warnings:
            warnings.warn(message, UserWarning, stacklevel=2)
        return self.torch.available

    def device_count(self) -> int:
        self.torch.calls.append("cuda.device_count")
        return len(self.torch.devices)

    def is_initialized(self) -> bool:
        return self.torch.initialised

    def _device(self, index: int, call: str) -> StubDevice:
        self.torch.calls.append(f"cuda.{call}({index})")
        if self.torch.device_failure is not None:
            raise self.torch.device_failure
        return self.torch.devices[index]

    def get_device_properties(self, index: int) -> SimpleNamespace:
        device = self._device(index, "get_device_properties")
        props: dict[str, Any] = {
            "total_memory": device.total_memory,
            "multi_processor_count": device.multi_processor_count,
        }
        if device.uuid is not None:
            props["uuid"] = device.uuid
        if device.pci is not None:
            props.update(
                pci_domain_id=device.pci[0], pci_bus_id=device.pci[1], pci_device_id=device.pci[2]
            )
        return SimpleNamespace(**props)

    def get_device_name(self, index: int) -> str:
        return self._device(index, "get_device_name").name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self._device(index, "get_device_capability").capability

    def memory_allocated(self, index: int) -> int:
        return self.torch.devices[index].allocated

    def memory_reserved(self, index: int) -> int:
        return 2 * self.torch.devices[index].allocated

    def max_memory_allocated(self, index: int) -> int:
        return 3 * self.torch.devices[index].allocated

    def max_memory_reserved(self, index: int) -> int:
        return 4 * self.torch.devices[index].allocated


@dataclass
class StubTorch:
    __version__: str = "2.4.1+cu124"
    cuda_version: str | None = "12.4"
    hip_version: str | None = None
    cudnn: int | None = 90100
    available: bool = True
    devices: list[StubDevice] = field(default_factory=lambda: [StubDevice()])
    init_warnings: tuple[str, ...] = ()
    device_failure: BaseException | None = None
    initialised: bool = False
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.version = SimpleNamespace(cuda=self.cuda_version, hip=self.hip_version)
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: self.cudnn))
        self.cuda = _Cuda(self)

    @classmethod
    def cpu_only(cls) -> StubTorch:
        return cls(
            __version__="2.4.1+cpu", cuda_version=None, cudnn=None, available=False, devices=[]
        )


def write_stub_torch_package(
    directory: Path, stub: StubTorch | None = None, *, import_error: str | None = None
) -> Path:
    """An importable `torch` package mirroring `stub` (or failing to import), in `directory`."""
    package = directory / "torch"
    package.mkdir(parents=True, exist_ok=True)
    if import_error is not None:
        (package / "__init__.py").write_text(f"raise OSError({import_error!r})\n")
        return directory
    stub = stub or StubTorch()
    spec = {
        "version": stub.__version__,
        "cuda": stub.cuda_version,
        "hip": stub.hip_version,
        "cudnn": stub.cudnn,
        "available": stub.available,
        "warnings": list(stub.init_warnings),
        "devices": [vars(d) for d in stub.devices],
    }
    source = _PACKAGE_TEMPLATE.replace("SPEC_JSON", repr(json.dumps(spec)))
    (package / "__init__.py").write_text(source)
    return directory


_PACKAGE_TEMPLATE = """\
import json
import types
import warnings

_spec = json.loads(SPEC_JSON)
__version__ = _spec["version"]
version = types.SimpleNamespace(cuda=_spec["cuda"], hip=_spec["hip"])
_cudnn = types.SimpleNamespace(version=lambda: _spec["cudnn"])
backends = types.SimpleNamespace(cudnn=_cudnn)


def _is_available():
    for message in _spec["warnings"]:
        warnings.warn(message, UserWarning)
    return _spec["available"]


def _props(i):
    d = _spec["devices"][i]
    p = {"total_memory": d["total_memory"]}
    p["multi_processor_count"] = d["multi_processor_count"]
    if d["uuid"] is not None:
        p["uuid"] = d["uuid"]
    if d["pci"] is not None:
        p["pci_domain_id"], p["pci_bus_id"], p["pci_device_id"] = d["pci"]
    return types.SimpleNamespace(**p)


cuda = types.SimpleNamespace(
    is_available=_is_available,
    device_count=lambda: len(_spec["devices"]),
    is_initialized=lambda: False,
    get_device_properties=_props,
    get_device_name=lambda i: _spec["devices"][i]["name"],
    get_device_capability=lambda i: tuple(_spec["devices"][i]["capability"]),
)
"""
