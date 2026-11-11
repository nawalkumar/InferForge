"""Fake `SysfsReader`: a dict-based, synthetic `/sys/bus/pci/devices`. No filesystem access.

`devices` maps an entry name to its attribute files (`vendor`, `device`, `class`, with the
trailing newline sysfs prints) and optionally a `driver` symlink target. `failures` makes one
path raise: `"list"` for the directory itself, `"<address>/<attribute>"` or
`"<address>/driver"`. Every call is logged. Provenance is FIXTURE, so the production adapter
emits `fixture:` methods.
"""

from __future__ import annotations

import errno
from collections.abc import Iterable
from dataclasses import dataclass, field

from gpu_system_inspector.adapters.pci_sysfs import check_attribute_request, check_entry_name
from gpu_system_inspector.models import DataOrigin, PciObservation

DRIVERS = "../../../bus/pci/drivers"


def sysfs_device(
    vendor: str = "0x10de",
    device: str = "0x2684",
    class_code: str = "0x030000",
    driver: str | None = "nvidia",
) -> dict[str, str]:
    """Attribute files exactly as sysfs prints them, plus the driver link target."""
    files = {"vendor": f"{vendor}\n", "device": f"{device}\n", "class": f"{class_code}\n"}
    if driver is not None:
        files["driver"] = f"{DRIVERS}/{driver}"
    return files


def sysfs_reader_for(observations: Iterable[PciObservation]) -> FakeSysfsReader:
    """A synthetic sysfs holding exactly these observations (one entry per address)."""
    return FakeSysfsReader(
        devices={
            o.address: sysfs_device(o.vendor_id, o.device_id, o.class_code, o.kernel_driver.value)
            for o in observations
        }
    )


@dataclass
class FakeSysfsReader:
    devices: dict[str, dict[str, str]] = field(default_factory=dict)
    failures: dict[str, Exception] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> DataOrigin:
        return DataOrigin.FIXTURE

    def _call(self, path: str) -> None:
        self.calls.append(path)
        failure = self.failures.get(path)
        if failure is not None:
            raise failure

    def list_devices(self) -> tuple[str, ...]:
        self._call("list")
        return tuple(self.devices)

    def read_attribute(self, address: str, name: str) -> str:
        check_attribute_request(address, name)
        path = f"{address}/{name}"
        self._call(path)
        files = self.devices.get(address, {})
        if name not in files:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        return files[name]

    def read_driver_link(self, address: str) -> str:
        check_entry_name(address)
        path = f"{address}/driver"
        self._call(path)
        files = self.devices.get(address, {})
        if "driver" not in files:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        return files["driver"]
