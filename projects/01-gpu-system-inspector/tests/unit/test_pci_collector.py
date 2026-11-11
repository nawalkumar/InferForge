"""Stage 6.3 PCI collector (DEC-43): run status, host-wide collection errors and the
`GPU.PCI_SCAN` record, over the production adapter with a dict-based `FakeSysfsReader`.
Synthetic only: no real sysfs is touched."""

import dataclasses
import errno

import jsonschema
import pytest
from gsi_testkit.fakes import FakePciScanner, FakeSysfsReader, FixedClock, sysfs_device
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.adapters import AdapterUsageError, PciScan, PciSysfsAdapter
from gpu_system_inspector.adapters.pci_sysfs import PCI_SCAN_UNSUPPORTED_REASONS
from gpu_system_inspector.collectors import (
    GPU_PCI_SCAN,
    NVIDIA_ALL,
    PCI_REPORT_PATH,
    PciCollector,
    unsupported_capabilities_for,
)
from gpu_system_inspector.models import (
    CollectionError,
    CollectionStatus,
    CollectorRunStatus,
    ErrorDetail,
    InspectionSnapshot,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    Source,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

S = CollectionStatus
E = LinuxEnvironment
P = PlatformProfile
R = CollectorRunStatus

NVIDIA = "0000:01:00.0"
INTEL = "0000:00:02.0"
OTHER_GPU = "0000:02:00.0"

INELIGIBLE = {
    "wsl1": (E.WSL1, P.LINUX),
    "wsl2": (E.WSL2, P.WSL2),
    "unknown-linux": (E.UNKNOWN, P.LINUX),
    "windows": (None, P.WINDOWS),
    "macos": (None, P.MACOS),
    "other": (None, P.OTHER),
}


def _reader(devices=None, failures=None) -> FakeSysfsReader:
    return FakeSysfsReader(devices=devices or {}, failures=failures or {})


def _collect(reader, environment=E.NATIVE_LINUX, profile=P.LINUX, clock=None):
    scanner = PciSysfsAdapter(environment=environment, reader=reader)
    collector = PciCollector(
        scanner, clock or FixedClock(), environment=environment, profile=profile
    )
    return collector.collect()


def _healthy() -> dict[str, dict[str, str]]:
    return {
        NVIDIA: sysfs_device(),
        INTEL: sysfs_device("0x8086", "0x46a6", "0x030000", "i915"),
    }


def _enoent(path: str) -> FileNotFoundError:
    return FileNotFoundError(errno.ENOENT, "No such file or directory", path)


def _eacces(path: str) -> PermissionError:
    return PermissionError(errno.EACCES, "Permission denied", path)


def _eio(path: str) -> OSError:
    return OSError(errno.EIO, "Input/output error", path)


class _StubScanner:
    """A `PciScanner` returning a fixed result, for contract-violation cases."""

    def __init__(self, scan: PciScan | Exception) -> None:
        self._scan = scan

    def scan(self) -> PciScan:
        if isinstance(self._scan, Exception):
            raise self._scan
        return self._scan


class _SteppingClock(FixedClock):
    def __init__(self, *ticks: int) -> None:
        super().__init__()
        self._values = iter(ticks)

    def monotonic_ms(self) -> int:
        return next(self._values)


# --- successful scans ---------------------------------------------------------------------


def test_collected_devices_complete_without_errors():
    outcome = _collect(_reader(_healthy()))
    expected = PciSysfsAdapter(environment=E.NATIVE_LINUX, reader=_reader(_healthy())).scan()
    assert outcome.section == expected.observations
    assert [o.address for o in outcome.section.value] == [INTEL, NVIDIA]
    assert outcome.run.collector == "pci" and outcome.run.status is R.COMPLETED
    assert outcome.errors == () and outcome.unsupported == ()


def test_empty_collected_scan_completes():
    outcome = _collect(_reader())
    assert outcome.section.status is S.COLLECTED and outcome.section.value == ()
    assert outcome.run.status is R.COMPLETED
    assert outcome.errors == () and outcome.unsupported == ()


def test_no_driver_bound_is_an_observation_not_a_failure():
    outcome = _collect(_reader({NVIDIA: sysfs_device(driver=None)}))
    (device,) = outcome.section.value
    assert device.kernel_driver.status is S.NOT_DETECTED
    assert device.kernel_driver.error.code == "NO_DRIVER_BOUND"
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()


def test_section_is_the_scanners_observed_value_unchanged():
    result = Observed.collected((), source=Source.OS, method="fixture:/sys/bus/pci/devices")
    scanner = FakePciScanner(result=result)
    outcome = PciCollector(
        scanner, FixedClock(), environment=E.NATIVE_LINUX, profile=P.LINUX
    ).collect()
    assert outcome.section is result and scanner.calls == ["/sys/bus/pci/devices"]


# --- field-level (driver link) failures ---------------------------------------------------


@pytest.mark.parametrize(
    ("devices", "failures", "status", "code"),
    [
        (
            {NVIDIA: sysfs_device()},
            {f"{NVIDIA}/driver": _eacces("driver")},
            S.PERMISSION_DENIED,
            "PermissionError",
        ),
        (
            {NVIDIA: sysfs_device()},
            {f"{NVIDIA}/driver": _eio("driver")},
            S.COLLECTION_FAILED,
            "OSError",
        ),
        (
            {NVIDIA: {**sysfs_device(), "driver": "../drivers/bad name"}},
            {},
            S.COLLECTION_FAILED,
            "PARSE_ERROR",
        ),
    ],
    ids=["denied", "io-error", "unusable-target"],
)
def test_unreadable_driver_link_is_partial_without_collection_error(
    devices, failures, status, code
):
    outcome = _collect(_reader(devices, failures))
    (device,) = outcome.section.value
    assert device.address == NVIDIA and device.is_nvidia
    assert device.kernel_driver.status is status and device.kernel_driver.error.code == code
    assert device.kernel_driver.method == "fixture:/sys/bus/pci/devices/*/driver"
    assert outcome.run.status is R.PARTIAL
    assert outcome.errors == ()


# --- omitted-device failures --------------------------------------------------------------


OMITTED = {
    "invalid-address": (
        {"junk": sysfs_device()},
        {},
        S.COLLECTION_FAILED,
        "PARSE_ERROR",
        "junk address: 'junk' is not a canonical PCI address",
    ),
    "attribute-malformed": (
        {OTHER_GPU: sysfs_device(vendor="0X10DE")},
        {},
        S.COLLECTION_FAILED,
        "PARSE_ERROR",
        f"{OTHER_GPU} vendor: vendor of {OTHER_GPU} is not canonical lowercase hex: '0X10DE'",
    ),
    "unreadable-not-detected": (
        {OTHER_GPU: {k: v for k, v in sysfs_device().items() if k != "vendor"}},
        {},
        S.NOT_DETECTED,
        "FileNotFoundError",
        f"{OTHER_GPU} vendor: {_enoent(f'{OTHER_GPU}/vendor')}",
    ),
    "unreadable-permission-denied": (
        {OTHER_GPU: sysfs_device()},
        {f"{OTHER_GPU}/class": _eacces(f"{OTHER_GPU}/class")},
        S.PERMISSION_DENIED,
        "PermissionError",
        f"{OTHER_GPU} class: {_eacces(f'{OTHER_GPU}/class')}",
    ),
    "unreadable-collection-failed": (
        {OTHER_GPU: sysfs_device()},
        {f"{OTHER_GPU}/device": _eio(f"{OTHER_GPU}/device")},
        S.COLLECTION_FAILED,
        "OSError",
        f"{OTHER_GPU} device: {_eio(f'{OTHER_GPU}/device')}",
    ),
}


@pytest.mark.parametrize(
    ("bad", "failures", "status", "code", "message"), OMITTED.values(), ids=OMITTED.keys()
)
def test_omitted_device_gives_one_host_wide_error_and_a_partial_run(
    bad, failures, status, code, message
):
    outcome = _collect(_reader({**_healthy(), **bad}, failures))
    assert [o.address for o in outcome.section.value] == [INTEL, NVIDIA]
    assert outcome.run.status is R.PARTIAL
    assert outcome.errors == (
        CollectionError(
            collector="pci",
            device_ref=None,
            status=status,
            error_code=code,
            message=message,
            report_path="/gpu/pci_observations",
        ),
    )
    assert outcome.unsupported == ()


def test_errors_follow_the_scans_failure_order_and_skip_driver_link_failures():
    devices = {
        **_healthy(),
        "0000:05:00.0": sysfs_device(vendor="bad"),
        "0000:03:00.0": sysfs_device(),
        "0000:04:00.0": sysfs_device(),
        "zz-entry": sysfs_device(),
    }
    failures = {
        "0000:03:00.0/class": _eacces("0000:03:00.0/class"),
        "0000:04:00.0/driver": _eio("0000:04:00.0/driver"),
        f"{NVIDIA}/device": _eio(f"{NVIDIA}/device"),
    }
    first = _collect(_reader(devices, failures))
    again = _collect(_reader(devices, failures))
    assert first == again
    assert [e.message.split(" ")[0] for e in first.errors] == [
        NVIDIA,
        "0000:03:00.0",
        "0000:05:00.0",
        "zz-entry",
    ]
    assert all(e.device_ref is None and e.report_path == PCI_REPORT_PATH for e in first.errors)
    # The driver-link failure keeps its device and adds no CollectionError.
    assert [o.address for o in first.section.value] == [INTEL, "0000:04:00.0"]
    assert first.run.status is R.PARTIAL


# --- whole-scan outcomes ------------------------------------------------------------------


def test_missing_sysfs_root_is_not_detected_and_completes():
    outcome = _collect(_reader(failures={"list": _enoent("/sys/bus/pci/devices")}))
    assert outcome.section.status is S.NOT_DETECTED
    assert outcome.section.error.code == "FileNotFoundError"
    assert outcome.run.status is R.COMPLETED
    assert outcome.errors == () and outcome.unsupported == ()


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (_eacces("/sys/bus/pci/devices"), S.PERMISSION_DENIED, "PermissionError"),
        (_eio("/sys/bus/pci/devices"), S.COLLECTION_FAILED, "OSError"),
    ],
    ids=["permission-denied", "collection-failed"],
)
def test_root_failure_is_partial_with_one_host_wide_error(exc, status, code):
    outcome = _collect(_reader(failures={"list": exc}))
    assert outcome.section.status is status and outcome.section.value is None
    assert outcome.run.status is R.PARTIAL
    assert outcome.errors == (
        CollectionError(
            collector="pci",
            device_ref=None,
            status=status,
            error_code=code,
            message=str(exc),
            report_path=PCI_REPORT_PATH,
        ),
    )
    assert outcome.unsupported == ()


@pytest.mark.parametrize(("environment", "profile"), INELIGIBLE.values(), ids=INELIGIBLE.keys())
def test_unsupported_environment_completes_with_only_the_pci_capability(environment, profile):
    reader = _reader(_healthy())
    outcome = _collect(reader, environment, profile)
    assert reader.calls == []
    assert outcome.section.status is S.NOT_SUPPORTED
    assert outcome.run.status is R.COMPLETED and outcome.errors == ()
    (record,) = outcome.unsupported
    assert record.capability == GPU_PCI_SCAN and record.device_ref is None
    assert record.status is S.NOT_SUPPORTED
    assert record.reason == PCI_SCAN_UNSUPPORTED_REASONS[environment]
    assert record == unsupported_capabilities_for(environment, profile)[0]
    assert all(u.capability != NVIDIA_ALL for u in outcome.unsupported)


# --- fallback, clock and consistency ------------------------------------------------------


def test_unavailable_section_matches_the_shared_convention():
    collector = PciCollector(
        _StubScanner(RuntimeError("bug")), FixedClock(), environment=None, profile=P.WINDOWS
    )
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    section = collector.unavailable(error)
    assert section == Observed(
        value=None, status=S.COLLECTION_FAILED, source=None, method=None, error=error
    )
    assert section.error is error


def test_duration_comes_from_the_injected_clock():
    outcome = _collect(_reader(_healthy()), clock=_SteppingClock(100, 137))
    assert outcome.run.duration_ms == 37
    assert _collect(_reader(), clock=FixedClock(step_ms=7)).run.duration_ms == 7


@pytest.mark.parametrize(
    ("scanner_environment", "collector_environment", "profile"),
    [
        (E.WSL2, E.NATIVE_LINUX, P.LINUX),  # scanner NOT_SUPPORTED, environment eligible
        (E.NATIVE_LINUX, E.WSL2, P.WSL2),  # scanner ran, environment ineligible
        (E.WSL2, E.UNKNOWN, P.LINUX),  # both NOT_SUPPORTED, different code and reason
        (None, E.WSL1, P.LINUX),  # both PLATFORM_NOT_SUPPORTED, different reason
    ],
    ids=["eligible-but-unsupported", "ineligible-but-scanned", "wsl2-vs-unknown", "os-vs-wsl1"],
)
def test_scanner_and_environment_mismatch_is_a_usage_error(
    scanner_environment, collector_environment, profile
):
    scanner = PciSysfsAdapter(environment=scanner_environment, reader=_reader(_healthy()))
    collector = PciCollector(
        scanner, FixedClock(), environment=collector_environment, profile=profile
    )
    with pytest.raises(AdapterUsageError):
        collector.collect()


@pytest.mark.parametrize(
    ("environment", "profile"),
    [(E.NATIVE_LINUX, P.WSL2), (E.WSL2, P.LINUX), (None, P.LINUX), (E.UNKNOWN, P.MACOS)],
)
def test_inconsistent_environment_and_profile_are_rejected(environment, profile):
    with pytest.raises(AdapterUsageError):
        PciCollector(
            _StubScanner(RuntimeError("unused")),
            FixedClock(),
            environment=environment,
            profile=profile,
        )


@pytest.mark.parametrize("status", [S.UNKNOWN, S.NOT_INSTALLED, S.SKIPPED, S.NOT_APPLICABLE])
def test_scan_statuses_outside_the_contract_are_usage_errors(status):
    if status is S.SKIPPED:
        observations = Observed.skipped(note="skipped")
    elif status is S.NOT_APPLICABLE:
        observations = Observed.not_applicable(note="n/a")
    else:
        observations = Observed.failed(status, code="X", message="x", source=Source.OS)
    collector = PciCollector(
        _StubScanner(PciScan(observations=observations)),
        FixedClock(),
        environment=E.NATIVE_LINUX,
        profile=P.LINUX,
    )
    with pytest.raises(AdapterUsageError, match=str(status)):
        collector.collect()


@pytest.mark.parametrize(
    "scanner",
    [
        _StubScanner(RuntimeError("scanner bug")),
        PciSysfsAdapter(
            environment=E.NATIVE_LINUX, reader=_reader(failures={"list": RuntimeError("bug")})
        ),
    ],
    ids=["scanner-raises", "reader-raises-non-oserror"],
)
def test_unexpected_exceptions_propagate_to_the_guard(scanner):
    collector = PciCollector(scanner, FixedClock(), environment=E.NATIVE_LINUX, profile=P.LINUX)
    with pytest.raises(RuntimeError, match="bug"):
        collector.collect()


# --- model and schema contracts -----------------------------------------------------------


def _all_outcomes():
    yield _collect(_reader(_healthy()))
    yield _collect(_reader())
    yield _collect(
        _reader(
            {**_healthy(), "junk": sysfs_device()},
            {
                f"{INTEL}/driver": _eio("driver"),
            },
        )
    )
    yield _collect(_reader(failures={"list": _enoent("root")}))
    yield _collect(_reader(failures={"list": _eacces("root")}))
    for environment, profile in INELIGIBLE.values():
        yield _collect(_reader(), environment, profile)


def test_the_collector_never_owns_a_failed_run():
    assert {o.run.status for o in _all_outcomes()} == {R.COMPLETED, R.PARTIAL}


def test_outcomes_validate_against_the_shipped_schema():
    defs = load_report_schema()["$defs"]

    def check(value, name):
        jsonschema.validate(to_json_dict(value), {"$ref": f"#/$defs/{name}", "$defs": defs})

    for outcome in _all_outcomes():
        check(outcome.section, "ObservedPciObservationList")
        check(outcome.run, "CollectorRun")
        for error in outcome.errors:
            check(error, "CollectionError")
        for record in outcome.unsupported:
            check(record, "UnsupportedCapability")


def test_outcome_fits_an_inspection_snapshot():
    """The section, errors and capability records are accepted by the snapshot models."""
    base = FIXTURES_BY_ID["FX03"].snapshot
    outcome = _collect(_reader({**_healthy(), "junk": sysfs_device()}))
    snapshot = dataclasses.replace(
        base,
        gpu=dataclasses.replace(base.gpu, pci_observations=outcome.section),
        collection_errors=outcome.errors,
    )
    assert isinstance(snapshot, InspectionSnapshot)
    assert snapshot.gpu.pci_observations == outcome.section
    assert snapshot.collection_errors == outcome.errors
