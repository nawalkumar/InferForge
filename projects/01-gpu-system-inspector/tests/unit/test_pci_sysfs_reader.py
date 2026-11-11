"""`StdlibSysfsReader` primitives on temporary trees only. A guard fails the test if anything
under `/sys` is opened, listed or resolved, so the real sysfs is never touched."""

import builtins
import errno
import os

import pytest

from gpu_system_inspector.adapters import (
    PCI_DEVICES_ROOT,
    AdapterUsageError,
    PciSysfsAdapter,
    StdlibSysfsReader,
    SysfsReader,
)
from gpu_system_inspector.models import CollectionStatus, DataOrigin, LinuxEnvironment

ADDRESS = "0000:01:00.0"


@pytest.fixture(autouse=True)
def no_real_sysfs(monkeypatch):
    real_open, real_listdir, real_readlink = builtins.open, os.listdir, os.readlink
    touched: list[str] = []

    def guard(function):
        def wrapper(path, *args, **kwargs):
            if os.fspath(path).startswith("/sys"):
                raise AssertionError(f"real sysfs touched: {path}")
            touched.append(os.fspath(path))
            return function(path, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(builtins, "open", guard(real_open))
    monkeypatch.setattr(os, "listdir", guard(real_listdir))
    monkeypatch.setattr(os, "readlink", guard(real_readlink))
    return touched


@pytest.fixture
def tree(tmp_path):
    """A small synthetic PCI tree: one GPU bound to a driver, one device without a driver."""
    root = tmp_path / "devices"
    drivers = tmp_path / "drivers" / "nvidia"
    drivers.mkdir(parents=True)
    gpu = root / ADDRESS
    gpu.mkdir(parents=True)
    (gpu / "vendor").write_text("0x10de\n")
    (gpu / "device").write_text("0x2684\n")
    (gpu / "class").write_text("0x030000\n")
    (gpu / "config").write_bytes(b"\x00" * 256)
    os.symlink(os.path.relpath(drivers, gpu), gpu / "driver")
    bare = root / "0000:02:00.0"
    bare.mkdir()
    (bare / "class").write_text("0x030200\n")
    return root


def test_default_root_is_the_sysfs_pci_directory_and_construction_touches_nothing(
    no_real_sysfs,
):
    reader = StdlibSysfsReader()
    assert PCI_DEVICES_ROOT == "/sys/bus/pci/devices"
    assert reader.provenance is DataOrigin.LIVE
    assert isinstance(reader, SysfsReader)
    assert no_real_sysfs == []


@pytest.mark.parametrize(
    "environment",
    [LinuxEnvironment.WSL1, LinuxEnvironment.WSL2, LinuxEnvironment.UNKNOWN, None],
    ids=["wsl1", "wsl2", "unknown", "non-linux"],
)
def test_production_adapter_on_the_real_reader_never_touches_sysfs_when_ineligible(
    no_real_sysfs, environment
):
    adapter = PciSysfsAdapter(environment=environment)  # default: the real /sys reader
    result = adapter.scan()
    assert no_real_sysfs == []
    assert adapter.provenance is DataOrigin.LIVE
    assert result.observations.status is CollectionStatus.NOT_SUPPORTED
    assert result.observations.method == "/sys/bus/pci/devices"  # live: no fixture prefix


def test_lists_device_entries(tree):
    assert sorted(StdlibSysfsReader(str(tree)).list_devices()) == [ADDRESS, "0000:02:00.0"]


def test_reads_attributes_exactly_as_stored(tree):
    reader = StdlibSysfsReader(str(tree))
    assert reader.read_attribute(ADDRESS, "vendor") == "0x10de\n"
    assert reader.read_attribute(ADDRESS, "device") == "0x2684\n"
    assert reader.read_attribute(ADDRESS, "class") == "0x030000\n"


def test_resolves_the_driver_symlink_target(tree):
    target = StdlibSysfsReader(str(tree)).read_driver_link(ADDRESS)
    assert target.endswith("drivers/nvidia") and not os.path.isabs(target)


@pytest.mark.parametrize("name", ["config", "resource0", "rom", "driver", "uevent"])
def test_other_attributes_are_refused_before_anything_is_opened(tree, no_real_sysfs, name):
    with pytest.raises(AdapterUsageError, match="vendor/device/class"):
        StdlibSysfsReader(str(tree)).read_attribute(ADDRESS, name)
    assert no_real_sysfs == []


@pytest.mark.parametrize("address", ["..", ".", "", "../devices", "0000:01:00.0/../x"])
def test_path_traversal_is_refused(tree, no_real_sysfs, address):
    reader = StdlibSysfsReader(str(tree))
    with pytest.raises(AdapterUsageError):
        reader.read_attribute(address, "vendor")
    with pytest.raises(AdapterUsageError):
        reader.read_driver_link(address)
    assert no_real_sysfs == []


def test_missing_driver_link_raises_file_not_found(tree):
    with pytest.raises(FileNotFoundError):
        StdlibSysfsReader(str(tree)).read_driver_link("0000:02:00.0")


def test_driver_that_is_not_a_symlink_raises_os_error(tree):
    (tree / "0000:02:00.0" / "driver").write_text("not a link")
    with pytest.raises(OSError) as exc:
        StdlibSysfsReader(str(tree)).read_driver_link("0000:02:00.0")
    assert exc.value.errno == errno.EINVAL


def test_missing_attribute_raises_file_not_found(tree):
    with pytest.raises(FileNotFoundError):
        StdlibSysfsReader(str(tree)).read_attribute("0000:02:00.0", "vendor")


def test_missing_root_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        StdlibSysfsReader(str(tmp_path / "absent")).list_devices()


def test_root_that_is_a_file_raises_not_a_directory(tmp_path):
    root = tmp_path / "file"
    root.write_text("")
    with pytest.raises(NotADirectoryError):
        StdlibSysfsReader(str(root)).list_devices()


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_unreadable_root_raises_permission_error(tree):
    tree.chmod(0)
    try:
        with pytest.raises(PermissionError):
            StdlibSysfsReader(str(tree)).list_devices()
    finally:
        tree.chmod(0o755)


def test_non_ascii_bytes_are_replaced_not_raised(tree):
    (tree / ADDRESS / "vendor").write_bytes(b"0x10\xffe\n")
    value = StdlibSysfsReader(str(tree)).read_attribute(ADDRESS, "vendor")
    assert value == "0x10\ufffde\n"


def test_attribute_reads_are_bounded(tree):
    (tree / ADDRESS / "class").write_text("0x" + "f" * 10_000)
    assert len(StdlibSysfsReader(str(tree)).read_attribute(ADDRESS, "class")) == 64


def test_reader_never_writes(tree, no_real_sysfs, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("write attempted")

    real_open = builtins.open

    def read_only_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            refuse()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only_open)
    for name in ("unlink", "remove", "rename", "replace", "symlink", "chmod", "mkdir"):
        monkeypatch.setattr(os, name, refuse)
    reader = StdlibSysfsReader(str(tree))
    reader.list_devices()
    for name in ("vendor", "device", "class"):
        reader.read_attribute(ADDRESS, name)
    reader.read_driver_link(ADDRESS)
