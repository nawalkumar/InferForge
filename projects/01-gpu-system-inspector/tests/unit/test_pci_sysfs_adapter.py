"""`PciSysfsAdapter` on a synthetic `FakeSysfsReader`: eligibility, parsing, filtering,
failure reporting, ordering and provenance. No real sysfs is touched."""

import errno

import pytest
from gsi_testkit.fakes import FakeSysfsReader, sysfs_device

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    PciDeviceFailure,
    PciFailureKind,
    PciScan,
    PciSysfsAdapter,
)
from gpu_system_inspector.adapters.pci_sysfs import PCI_SCAN_UNSUPPORTED_REASONS
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    ErrorDetail,
    LinuxEnvironment,
    Observed,
    Source,
)

S = CollectionStatus
NATIVE = LinuxEnvironment.NATIVE_LINUX
NV, INTEL, AMD = "0x10de", "0x8086", "0x1002"


def scan(devices=None, failures=None, environment=NATIVE) -> tuple[PciScan, FakeSysfsReader]:
    reader = FakeSysfsReader(devices=dict(devices or {}), failures=dict(failures or {}))
    return PciSysfsAdapter(environment=environment, reader=reader).scan(), reader


def by_address(result: PciScan) -> dict:
    return {o.address: o for o in result.observations.value}


# --- eligibility (fail-closed) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "code"),
    [
        (LinuxEnvironment.WSL1, "PLATFORM_NOT_SUPPORTED"),
        (LinuxEnvironment.WSL2, "PLATFORM_NOT_SUPPORTED"),
        (LinuxEnvironment.UNKNOWN, "LINUX_ENVIRONMENT_UNKNOWN"),
        (None, "PLATFORM_NOT_SUPPORTED"),
    ],
    ids=["wsl1", "wsl2", "unknown", "non-linux"],
)
def test_ineligible_environments_never_touch_the_reader(environment, code):
    reader = FakeSysfsReader(
        devices={"0000:01:00.0": sysfs_device()}, failures={"list": AssertionError("touched")}
    )
    result = PciSysfsAdapter(environment=environment, reader=reader).scan()
    assert reader.calls == []
    observations = result.observations
    assert observations.status is S.NOT_SUPPORTED and observations.value is None
    assert observations.error.code == code
    assert observations.error.message == PCI_SCAN_UNSUPPORTED_REASONS[environment]
    assert observations.source is Source.OS
    assert result.failures == ()


@pytest.mark.parametrize("environment", ["NATIVE_LINUX", "LINUX", True, 0], ids=repr)
def test_eligibility_must_be_an_explicit_linux_environment(environment):
    with pytest.raises(AdapterUsageError, match="LinuxEnvironment"):
        PciSysfsAdapter(environment=environment, reader=FakeSysfsReader())


def test_eligibility_argument_is_required():
    with pytest.raises(TypeError):
        PciSysfsAdapter(reader=FakeSysfsReader())  # type: ignore[call-arg]


def test_only_native_linux_is_eligible():
    eligible = {
        env: PciSysfsAdapter(environment=env, reader=FakeSysfsReader()).eligible
        for env in [*LinuxEnvironment, None]
    }
    assert eligible == {env: env is NATIVE for env in eligible}


# --- observations -------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["nvidia", "nouveau", "vfio-pci"])
def test_nvidia_device_with_each_driver(driver):
    result, _ = scan({"0000:01:00.0": sysfs_device(driver=driver)})
    (observation,) = result.observations.value
    assert (observation.vendor_id, observation.device_id) == (NV, "0x2684")
    assert observation.class_code == "0x030000" and observation.is_nvidia is True
    assert observation.kernel_driver.value == driver
    assert observation.kernel_driver.source is Source.OS
    assert result.failures == ()


def test_nvidia_without_a_driver_is_not_detected_with_no_driver_bound():
    result, _ = scan({"0000:01:00.0": sysfs_device(driver=None)})
    driver = result.observations.value[0].kernel_driver
    assert driver.status is S.NOT_DETECTED and driver.value is None
    assert driver.error.code == "NO_DRIVER_BOUND"
    assert result.failures == ()  # no driver bound is an observation, not a failure


def test_intel_and_amd_are_non_nvidia_observations():
    result, _ = scan(
        {
            "0000:00:02.0": sysfs_device(INTEL, "0x46a6", driver="i915"),
            "0000:03:00.0": sysfs_device(AMD, "0x73bf", driver="amdgpu"),
        }
    )
    observations = by_address(result)
    assert {a: (o.vendor_id, o.is_nvidia) for a, o in observations.items()} == {
        "0000:00:02.0": (INTEL, False),
        "0000:03:00.0": (AMD, False),
    }


@pytest.mark.parametrize(
    ("class_code", "included"),
    [
        ("0x030000", True),  # VGA-compatible
        ("0x030001", True),
        ("0x030100", True),  # XGA
        ("0x030200", True),  # 3D controller (data-centre GPUs)
        ("0x038000", True),  # other display controller
        ("0x040300", False),  # audio function of a GPU
        ("0x0c0330", False),  # USB controller on a GPU
        ("0x020000", False),  # network controller
        ("0x060400", False),  # PCI bridge
    ],
)
def test_only_display_base_class_is_included(class_code, included):
    result, reader = scan({"0000:01:00.0": sysfs_device(class_code=class_code)})
    assert bool(result.observations.value) is included
    if not included:
        assert reader.calls == ["list", "0000:01:00.0/class"]  # nothing else is read


def test_empty_root_is_collected_and_empty():
    result, reader = scan({})
    assert result.observations.status is S.COLLECTED and result.observations.value == ()
    assert reader.calls == ["list"]


def test_observations_are_sorted_by_address_whatever_the_listing_order():
    listed = ["0000:81:00.0", "0000:01:00.0", "0001:00:00.0", "0000:01:00.1"]
    devices = {address: sysfs_device(class_code="0x030200") for address in listed}
    result, _ = scan(devices)
    assert [o.address for o in result.observations.value] == sorted(listed)
    again, _ = scan(dict(reversed(list(devices.items()))))
    assert again == result


def test_provenance_and_methods_are_synthetic_for_the_fake():
    result, _ = scan({"0000:01:00.0": sysfs_device()})
    adapter = PciSysfsAdapter(environment=NATIVE, reader=FakeSysfsReader())
    assert adapter.provenance.value == "FIXTURE"
    assert result.observations.method == FIXTURE_METHOD_PREFIX + "/sys/bus/pci/devices"
    driver = result.observations.value[0].kernel_driver
    assert driver.method == FIXTURE_METHOD_PREFIX + "/sys/bus/pci/devices/*/driver"


def test_reader_is_only_asked_for_allowed_attributes():
    result, reader = scan(
        {"0000:01:00.0": sysfs_device(), "0000:00:1f.3": sysfs_device(INTEL, class_code="0x040300")}
    )
    assert len(result.observations.value) == 1
    attributes = {call.rsplit("/", 1)[-1] for call in reader.calls if call != "list"}
    assert attributes <= {"vendor", "device", "class", "driver"}
    assert reader.calls.count("list") == 1


@pytest.mark.parametrize("name", ["config", "resource0", "rom", "../vendor", ""])
def test_any_other_attribute_is_a_usage_error(name):
    with pytest.raises(AdapterUsageError):
        FakeSysfsReader(devices={"0000:01:00.0": sysfs_device()}).read_attribute(
            "0000:01:00.0", name
        )


@pytest.mark.parametrize("address", ["../0000:01:00.0", "a/b", "..", "."])
def test_entry_names_with_paths_are_usage_errors(address):
    with pytest.raises(AdapterUsageError):
        FakeSysfsReader().read_attribute(address, "vendor")
    with pytest.raises(AdapterUsageError):
        FakeSysfsReader().read_driver_link(address)


# --- whole-scan failures ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (FileNotFoundError(errno.ENOENT, "missing"), S.NOT_DETECTED, "FileNotFoundError"),
        (PermissionError(errno.EACCES, "denied"), S.PERMISSION_DENIED, "PermissionError"),
        (OSError(errno.EIO, "I/O error"), S.COLLECTION_FAILED, "OSError"),
        (NotADirectoryError(errno.ENOTDIR, "not a dir"), S.COLLECTION_FAILED, "NotADirectoryError"),
    ],
    ids=["missing-root", "denied-root", "io-error", "not-a-directory"],
)
def test_root_failures_become_the_whole_scan_outcome(failure, status, code):
    result, reader = scan({"0000:01:00.0": sysfs_device()}, {"list": failure})
    assert result.observations.status is status and result.observations.value is None
    assert result.observations.error.code == code and result.failures == ()
    assert reader.calls == ["list"]


def test_programming_errors_are_not_swallowed():
    with pytest.raises(RuntimeError, match="bug"):
        scan({"0000:01:00.0": sysfs_device()}, {"0000:01:00.0/vendor": RuntimeError("bug")})


# --- per-device failures ------------------------------------------------------------------

GOOD = "0000:01:00.0"


@pytest.mark.parametrize(
    "address",
    [
        "0000:1:00.0",  # bus not two digits
        "0000:01:00.8",  # function out of range
        "0000:01:00",  # no function
        "0000:01:0G.0",  # not hex
        "0000:01:00.0 ",  # trailing space
        "0000:01:00.0\n",  # trailing newline
        "0000:AB:00.0",  # uppercase
        "000:01:00.0",  # short domain
        "pci0000:00",  # a bus, not a device
    ],
)
def test_malformed_addresses_are_reported_not_discarded(address):
    result, reader = scan({GOOD: sysfs_device(), address: sysfs_device()})
    assert [o.address for o in result.observations.value] == [GOOD]
    (failure,) = result.failures
    assert failure.address == address and failure.kind is PciFailureKind.INVALID_ADDRESS
    assert failure.attribute is None and failure.status is S.COLLECTION_FAILED
    assert failure.error.code == "PARSE_ERROR"
    assert not any(call.startswith(address + "/") for call in reader.calls)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("vendor", "0x10DE\n"),  # uppercase
        ("vendor", "10de\n"),  # no prefix
        ("vendor", "0x010de\n"),  # too long
        ("vendor", " 0x10de\n"),  # leading space
        ("vendor", "0x10de\n\n"),  # extra newline
        ("vendor", ""),  # empty
        ("device", "0x26\n"),  # too short
        ("device", "0X2684\n"),  # uppercase prefix
        ("class", "0x0300\n"),  # class too short
        ("class", "0x03000G\n"),  # not hex
        ("class", "0x030000 \n"),  # trailing space
    ],
)
def test_non_canonical_attributes_are_malformed_failures(attribute, value):
    bad = "0000:02:00.0"
    files = sysfs_device()
    files[attribute] = value
    result, _ = scan({GOOD: sysfs_device(), bad: files})
    assert [o.address for o in result.observations.value] == [GOOD]
    (failure,) = result.failures
    assert (failure.address, failure.kind, failure.attribute) == (
        bad,
        PciFailureKind.ATTRIBUTE_MALFORMED,
        attribute,
    )
    assert failure.error.code == "PARSE_ERROR" and attribute in failure.error.message


def test_value_without_trailing_newline_is_accepted():
    files = {"vendor": "0x10de", "device": "0x2684", "class": "0x030000"}
    result, _ = scan({GOOD: files})
    assert result.observations.value[0].vendor_id == "0x10de"


@pytest.mark.parametrize(
    ("attribute", "failure", "status", "code"),
    [
        ("class", PermissionError(errno.EACCES, "denied"), S.PERMISSION_DENIED, "PermissionError"),
        ("vendor", OSError(errno.EIO, "I/O error"), S.COLLECTION_FAILED, "OSError"),
        ("device", FileNotFoundError(errno.ENOENT, "gone"), S.NOT_DETECTED, "FileNotFoundError"),
    ],
)
def test_unreadable_attributes_are_typed_failures_and_others_are_kept(
    attribute, failure, status, code
):
    bad = "0000:02:00.0"
    result, _ = scan(
        {GOOD: sysfs_device(), bad: sysfs_device(), "0000:03:00.0": sysfs_device(INTEL)},
        {f"{bad}/{attribute}": failure},
    )
    assert [o.address for o in result.observations.value] == [GOOD, "0000:03:00.0"]
    assert result.observations.status is S.COLLECTED
    assert result.failures == (
        PciDeviceFailure(
            address=bad,
            kind=PciFailureKind.ATTRIBUTE_UNREADABLE,
            attribute=attribute,
            status=status,
            error=ErrorDetail(code=code, message=str(failure)),
        ),
    )


def test_missing_attribute_file_is_unreadable():
    files = sysfs_device()
    del files["device"]
    result, _ = scan({GOOD: files})
    (failure,) = result.failures
    assert failure.kind is PciFailureKind.ATTRIBUTE_UNREADABLE and failure.attribute == "device"
    assert failure.status is S.NOT_DETECTED and result.observations.value == ()


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (PermissionError(errno.EACCES, "denied"), S.PERMISSION_DENIED),
        (OSError(errno.EINVAL, "not a symlink"), S.COLLECTION_FAILED),
    ],
    ids=["denied", "not-a-symlink"],
)
def test_driver_link_errors_keep_the_observation_and_are_reported_distinctly(failure, status):
    result, _ = scan({GOOD: sysfs_device()}, {f"{GOOD}/driver": failure})
    (observation,) = result.observations.value
    assert observation.is_nvidia and observation.kernel_driver.status is status
    assert observation.kernel_driver.value is None
    (reported,) = result.failures
    assert (reported.kind, reported.attribute, reported.status) == (
        PciFailureKind.DRIVER_LINK_UNREADABLE,
        "driver",
        status,
    )


@pytest.mark.parametrize("target", ["", "/", "../drivers/has space"], ids=repr)
def test_unusable_driver_link_target_is_a_parse_failure(target):
    files = sysfs_device()
    files["driver"] = target
    result, _ = scan({GOOD: files})
    driver = result.observations.value[0].kernel_driver
    assert driver.status is S.COLLECTION_FAILED and driver.error.code == "PARSE_ERROR"
    assert result.failures[0].kind is PciFailureKind.DRIVER_LINK_UNREADABLE


@pytest.mark.parametrize(
    ("target", "name"),
    [
        ("../../../bus/pci/drivers/nvidia", "nvidia"),
        ("/sys/bus/pci/drivers/vfio-pci/", "vfio-pci"),
        ("nouveau", "nouveau"),
    ],
)
def test_driver_name_is_the_link_basename(target, name):
    files = sysfs_device()
    files["driver"] = target
    result, _ = scan({GOOD: files})
    assert result.observations.value[0].kernel_driver.value == name


def test_failures_are_sorted_and_carry_enough_for_a_collection_error():
    result, _ = scan(
        {
            "0000:05:00.0": sysfs_device(),
            "bogus": sysfs_device(),
            "0000:04:00.0": sysfs_device(),
            GOOD: sysfs_device(),
        },
        {"0000:05:00.0/class": OSError(errno.EIO, "x"), "0000:04:00.0/driver": OSError("y")},
    )
    assert [(f.address, f.attribute) for f in result.failures] == [
        ("0000:04:00.0", "driver"),
        ("0000:05:00.0", "class"),
        ("bogus", None),
    ]
    for failure in result.failures:
        assert failure.status is not S.COLLECTED and failure.error.code and failure.error.message
    assert [o.address for o in result.observations.value] == [GOOD, "0000:04:00.0"]


# --- result type --------------------------------------------------------------------------


def test_scan_result_rejects_failures_without_a_completed_scan():
    failure = PciDeviceFailure(
        address=GOOD,
        kind=PciFailureKind.ATTRIBUTE_UNREADABLE,
        attribute="vendor",
        status=S.PERMISSION_DENIED,
        error=ErrorDetail(code="PermissionError", message="denied"),
    )
    unsupported = Observed.failed(
        S.NOT_SUPPORTED, code="PLATFORM_NOT_SUPPORTED", message="x", source=Source.OS, method=None
    )
    with pytest.raises(AdapterUsageError, match="COLLECTED"):
        PciScan(observations=unsupported, failures=(failure,))
    with pytest.raises(AdapterUsageError, match="COLLECTED"):
        PciDeviceFailure(
            address=GOOD,
            kind=PciFailureKind.ATTRIBUTE_UNREADABLE,
            attribute="vendor",
            status=S.COLLECTED,
            error=ErrorDetail(code="x", message="y"),
        )
