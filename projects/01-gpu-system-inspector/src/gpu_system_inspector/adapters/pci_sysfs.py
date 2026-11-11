"""Production `PciScanner` (architecture §4.5, FR-04, DEC-22): driver-independent PCI
display-controller observations from Linux sysfs.

Safety boundary (fail-closed): the adapter is built with an explicit `LinuxEnvironment`
value, and only `NATIVE_LINUX` may call the `SysfsReader`. WSL1, WSL2, `UNKNOWN` and
non-Linux platforms get `NOT_SUPPORTED` without any sysfs access: `LINUX_ENVIRONMENT_UNKNOWN`
for `UNKNOWN`, `PLATFORM_NOT_SUPPORTED` for the known unsupported environments. The reader
only lists `/sys/bus/pci/devices`, reads the `vendor`, `device` and `class` attributes and
resolves the `driver` symlink: no writes, no PCI config space, no subprocess, no `lspci`.
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from gpu_system_inspector.adapters.outcomes import (
    LINUX_ENVIRONMENT_UNKNOWN,
    NO_DRIVER_BOUND,
    PARSE_ERROR,
    PLATFORM_NOT_SUPPORTED,
    AdapterUsageError,
    exception_outcome,
)
from gpu_system_inspector.adapters.protocols import PciDeviceFailure, PciFailureKind, PciScan
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    DataOrigin,
    ErrorDetail,
    LinuxEnvironment,
    Observed,
    PciObservation,
    Source,
)

PCI_DEVICES_ROOT = "/sys/bus/pci/devices"
SCAN_METHOD = PCI_DEVICES_ROOT
DRIVER_METHOD = f"{PCI_DEVICES_ROOT}/*/driver"
PCI_ATTRIBUTES = frozenset({"vendor", "device", "class"})
DISPLAY_BASE_CLASS = "0x03"  # display controllers, including 3D controllers (0x0302xx)

# Always applied with `fullmatch`: `$` would also accept a trailing newline.
# Canonical sysfs PCI address: domain (4+ hex digits), bus, device, function.
PCI_ADDRESS = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
PCI_ID = re.compile(r"0x[0-9a-f]{4}")
PCI_CLASS = re.compile(r"0x[0-9a-f]{6}")
_ATTRIBUTE_FORMAT = {"vendor": PCI_ID, "device": PCI_ID, "class": PCI_CLASS}

# Why the PCI presence check doesn't run (data model §3.4): the scan's error message and the
# `GPU.PCI_SCAN` unsupported-capability reason use the same string. `None` is non-Linux.
PCI_SCAN_UNSUPPORTED_REASONS: dict[LinuxEnvironment | None, str] = {
    LinuxEnvironment.WSL2: (
        "GPUs are not enumerated on PCI under WSL2 (FR-04; product_specification.md §4, OBS-01)."
    ),
    LinuxEnvironment.WSL1: (
        "The PCI presence check runs only on native Linux, and this host is WSL1 "
        "(FR-04; DEC-41; data_model_specification.md §4.1)."
    ),
    LinuxEnvironment.UNKNOWN: (
        "The Linux environment couldn't be classified as native Linux, so the PCI presence "
        "check doesn't run (DEC-41; data_model_specification.md §4.1)."
    ),
    None: "PCI presence check is Linux-only in V1 (FR-04; DEC-22; product_specification.md §4).",
}
PCI_SCAN_UNSUPPORTED_CODES: dict[LinuxEnvironment | None, str] = {
    LinuxEnvironment.WSL2: PLATFORM_NOT_SUPPORTED,
    LinuxEnvironment.WSL1: PLATFORM_NOT_SUPPORTED,
    LinuxEnvironment.UNKNOWN: LINUX_ENVIRONMENT_UNKNOWN,
    None: PLATFORM_NOT_SUPPORTED,
}


def pci_scan_eligible(environment: LinuxEnvironment | None) -> bool:
    """The only eligibility rule (DEC-41): native Linux. The platform profile is never used."""
    return environment is LinuxEnvironment.NATIVE_LINUX


@runtime_checkable
class SysfsReader(Protocol):
    """Read-only access to `/sys/bus/pci/devices`. Methods may raise `OSError`."""

    @property
    def provenance(self) -> DataOrigin: ...
    def list_devices(self) -> tuple[str, ...]: ...
    def read_attribute(self, address: str, name: str) -> str: ...
    def read_driver_link(self, address: str) -> str: ...


def check_attribute_request(address: str, name: str) -> None:
    """Only `vendor`, `device` and `class` of a plain entry name may ever be read."""
    if name not in PCI_ATTRIBUTES:
        raise AdapterUsageError(
            f"{name!r} is not a readable PCI attribute (only vendor/device/class)"
        )
    check_entry_name(address)


def check_entry_name(address: str) -> None:
    if not address or address in (".", "..") or "/" in address or "\0" in address:
        raise AdapterUsageError(f"{address!r} is not a sysfs entry name")


class StdlibSysfsReader:
    """The real sysfs, through `os.listdir`, read-only `open` and `os.readlink`."""

    def __init__(self, root: str = PCI_DEVICES_ROOT) -> None:
        self._root = root

    @property
    def provenance(self) -> DataOrigin:
        return DataOrigin.LIVE

    def list_devices(self) -> tuple[str, ...]:
        return tuple(os.listdir(self._root))

    def read_attribute(self, address: str, name: str) -> str:
        check_attribute_request(address, name)
        path = os.path.join(self._root, address, name)
        with open(path, encoding="ascii", errors="replace") as handle:
            return handle.read(64)

    def read_driver_link(self, address: str) -> str:
        check_entry_name(address)
        return os.readlink(os.path.join(self._root, address, "driver"))


class PciSysfsAdapter:
    """`PciScanner` over a `SysfsReader` (the real sysfs unless a test injects one).

    `environment` is the Stage 5.1 `linux_environment` value (`None` off Linux). Anything but
    `LinuxEnvironment.NATIVE_LINUX` returns `NOT_SUPPORTED` and never calls the reader.
    """

    def __init__(
        self, *, environment: LinuxEnvironment | None, reader: SysfsReader | None = None
    ) -> None:
        if environment is not None and not isinstance(environment, LinuxEnvironment):
            raise AdapterUsageError(
                f"environment must be a LinuxEnvironment or None, got {environment!r}"
            )
        self._environment = environment
        self._reader: SysfsReader = reader if reader is not None else StdlibSysfsReader()

    @property
    def provenance(self) -> DataOrigin:
        return self._reader.provenance

    @property
    def eligible(self) -> bool:
        return pci_scan_eligible(self._environment)

    def scan(self) -> PciScan:
        if not self.eligible:
            return PciScan(
                observations=Observed.failed(
                    CollectionStatus.NOT_SUPPORTED,
                    code=PCI_SCAN_UNSUPPORTED_CODES[self._environment],
                    message=PCI_SCAN_UNSUPPORTED_REASONS[self._environment],
                    source=Source.OS,
                    method=self._method(SCAN_METHOD),
                )
            )
        try:
            entries = self._reader.list_devices()
        except OSError as exc:
            return PciScan(
                observations=exception_outcome(
                    exc, source=Source.OS, method=self._method(SCAN_METHOD)
                )
            )
        observations: list[PciObservation] = []
        failures: list[PciDeviceFailure] = []
        for address in sorted(entries):
            observation = self._observe(address, failures)
            if observation is not None:
                observations.append(observation)
        return PciScan(
            observations=Observed.collected(
                tuple(observations), source=Source.OS, method=self._method(SCAN_METHOD)
            ),
            failures=tuple(failures),
        )

    # --- helpers ------------------------------------------------------------------------

    def _method(self, method: str) -> str:
        if self.provenance is DataOrigin.FIXTURE:
            return FIXTURE_METHOD_PREFIX + method
        return method

    def _observe(self, address: str, failures: list[PciDeviceFailure]) -> PciObservation | None:
        """One device, or `None` if it's excluded (not a display controller) or failed."""
        if PCI_ADDRESS.fullmatch(address) is None:
            failures.append(
                _failure(
                    address,
                    PciFailureKind.INVALID_ADDRESS,
                    None,
                    f"{address!r} is not a canonical PCI address",
                )
            )
            return None
        class_code = self._attribute(address, "class", failures)
        if class_code is None or not class_code.startswith(DISPLAY_BASE_CLASS):
            return None
        vendor_id = self._attribute(address, "vendor", failures)
        device_id = self._attribute(address, "device", failures)
        if vendor_id is None or device_id is None:
            return None
        return PciObservation(
            address=address,
            vendor_id=vendor_id,
            device_id=device_id,
            class_code=class_code,
            is_nvidia=vendor_id == "0x10de",
            kernel_driver=self._kernel_driver(address, failures),
        )

    def _attribute(self, address: str, name: str, failures: list[PciDeviceFailure]) -> str | None:
        try:
            text = self._reader.read_attribute(address, name)
        except OSError as exc:
            outcome = exception_outcome(exc, source=Source.OS, method=None)
            assert outcome.error is not None
            failures.append(
                PciDeviceFailure(
                    address=address,
                    kind=PciFailureKind.ATTRIBUTE_UNREADABLE,
                    attribute=name,
                    status=outcome.status,
                    error=outcome.error,
                )
            )
            return None
        value = text[:-1] if text.endswith("\n") else text
        if _ATTRIBUTE_FORMAT[name].fullmatch(value) is None:
            failures.append(
                _failure(
                    address,
                    PciFailureKind.ATTRIBUTE_MALFORMED,
                    name,
                    f"{name} of {address} is not canonical lowercase hex: {value[:32]!r}",
                )
            )
            return None
        return value

    def _kernel_driver(self, address: str, failures: list[PciDeviceFailure]) -> Observed[str]:
        method = self._method(DRIVER_METHOD)
        try:
            target = self._reader.read_driver_link(address)
        except FileNotFoundError:
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code=NO_DRIVER_BOUND,
                message=f"no kernel driver is bound to {address}",
                source=Source.OS,
                method=method,
            )
        except OSError as exc:
            outcome = exception_outcome(exc, source=Source.OS, method=method)
            assert outcome.error is not None
            failures.append(
                PciDeviceFailure(
                    address=address,
                    kind=PciFailureKind.DRIVER_LINK_UNREADABLE,
                    attribute="driver",
                    status=outcome.status,
                    error=outcome.error,
                )
            )
            return outcome
        name = target.rstrip("/").rsplit("/", 1)[-1]
        if not name or any(char.isspace() for char in name):
            message = f"driver link of {address} has no usable target: {target[:64]!r}"
            failures.append(
                _failure(address, PciFailureKind.DRIVER_LINK_UNREADABLE, "driver", message)
            )
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=PARSE_ERROR,
                message=message,
                source=Source.OS,
                method=method,
            )
        return Observed.collected(name, source=Source.OS, method=method)


def _failure(
    address: str, kind: PciFailureKind, attribute: str | None, message: str
) -> PciDeviceFailure:
    return PciDeviceFailure(
        address=address,
        kind=kind,
        attribute=attribute,
        status=CollectionStatus.COLLECTION_FAILED,
        error=ErrorDetail(code=PARSE_ERROR, message=message),
    )
