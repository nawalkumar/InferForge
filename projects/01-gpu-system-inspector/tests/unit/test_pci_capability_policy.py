"""Stage 6.2 PCI eligibility policy: error codes, stable reasons, `GPU.PCI_SCAN` / `NVIDIA.ALL`
capability semantics and fixture parity. Synthetic only: no real sysfs is touched."""

import inspect

import jsonschema
import pytest
from gsi_testkit.fakes import FakeSysfsReader, sysfs_device, sysfs_reader_for
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import AdapterUsageError, PciSysfsAdapter
from gpu_system_inspector.adapters.outcomes import LINUX_ENVIRONMENT_UNKNOWN
from gpu_system_inspector.adapters.pci_sysfs import (
    PCI_SCAN_UNSUPPORTED_CODES,
    PCI_SCAN_UNSUPPORTED_REASONS,
    pci_scan_eligible,
)
from gpu_system_inspector.collectors import GPU_PCI_SCAN, NVIDIA_ALL, unsupported_capabilities_for
from gpu_system_inspector.models import (
    CollectionStatus,
    LinuxEnvironment,
    PlatformProfile,
    UnsupportedCapability,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

S = CollectionStatus
E = LinuxEnvironment
P = PlatformProfile

# The six approved environments: (linux_environment, platform_profile).
ENVIRONMENTS = {
    "native-linux": (E.NATIVE_LINUX, P.LINUX),
    "wsl1": (E.WSL1, P.LINUX),
    "wsl2": (E.WSL2, P.WSL2),
    "unknown-linux": (E.UNKNOWN, P.LINUX),
    "windows": (None, P.WINDOWS),
    "macos": (None, P.MACOS),
}
environments = pytest.mark.parametrize(
    ("environment", "profile"), ENVIRONMENTS.values(), ids=ENVIRONMENTS.keys()
)

# Pinned wording: changing a reason must be a deliberate, reviewed change.
REASONS = {
    E.WSL2: "GPUs are not enumerated on PCI under WSL2 (FR-04; product_specification.md §4, "
    "OBS-01).",
    E.WSL1: "The PCI presence check runs only on native Linux, and this host is WSL1 "
    "(FR-04; DEC-41; data_model_specification.md §4.1).",
    E.UNKNOWN: "The Linux environment couldn't be classified as native Linux, so the PCI "
    "presence check doesn't run (DEC-41; data_model_specification.md §4.1).",
    None: "PCI presence check is Linux-only in V1 (FR-04; DEC-22; product_specification.md §4).",
}
NVIDIA_MACOS = "NVIDIA CUDA GPUs are not supported on macOS (product_specification.md §4)."


# --- error codes and reasons --------------------------------------------------------------


def test_reasons_are_pinned_and_cite_documented_evidence():
    assert PCI_SCAN_UNSUPPORTED_REASONS == REASONS
    for reason in [*REASONS.values(), NVIDIA_MACOS]:
        assert reason.endswith(").") and "http" not in reason
        assert any(ref in reason for ref in ("FR-04", "DEC-22", "DEC-41", "product_specification"))


def test_error_codes_per_environment():
    assert LINUX_ENVIRONMENT_UNKNOWN == "LINUX_ENVIRONMENT_UNKNOWN"
    assert PCI_SCAN_UNSUPPORTED_CODES == {
        E.WSL1: "PLATFORM_NOT_SUPPORTED",
        E.WSL2: "PLATFORM_NOT_SUPPORTED",
        E.UNKNOWN: "LINUX_ENVIRONMENT_UNKNOWN",
        None: "PLATFORM_NOT_SUPPORTED",
    }


def test_only_native_linux_is_eligible():
    assert {env: pci_scan_eligible(env) for env in [*E, None]} == {
        E.NATIVE_LINUX: True,
        E.WSL1: False,
        E.WSL2: False,
        E.UNKNOWN: False,
        None: False,
    }


@environments
def test_adapter_outcome_per_environment(environment, profile):
    reader = FakeSysfsReader(devices={"0000:01:00.0": sysfs_device()})
    result = PciSysfsAdapter(environment=environment, reader=reader).scan()
    if environment is E.NATIVE_LINUX:
        assert result.observations.status is S.COLLECTED and len(result.observations.value) == 1
        assert reader.calls
        return
    assert reader.calls == []  # no sysfs access for any unsupported environment
    observations = result.observations
    assert observations.status is S.NOT_SUPPORTED and observations.value is None
    assert observations.error.code == PCI_SCAN_UNSUPPORTED_CODES[environment]
    assert observations.error.message == REASONS[environment]


def test_not_supported_is_not_an_empty_collected_scan():
    unsupported = PciSysfsAdapter(environment=E.UNKNOWN, reader=FakeSysfsReader()).scan()
    empty = PciSysfsAdapter(environment=E.NATIVE_LINUX, reader=FakeSysfsReader()).scan()
    assert empty.observations.status is S.COLLECTED and empty.observations.value == ()
    assert unsupported.observations.status is S.NOT_SUPPORTED
    assert unsupported.observations.value is None
    assert unsupported.observations != empty.observations


def test_the_adapter_has_no_platform_profile_input():
    """The gate can't be overridden by a profile: the adapter doesn't accept one."""
    parameters = inspect.signature(PciSysfsAdapter).parameters
    assert list(parameters) == ["environment", "reader"]
    assert parameters["environment"].default is inspect.Parameter.empty


def test_unknown_error_code_is_schema_valid():
    result = PciSysfsAdapter(environment=E.UNKNOWN, reader=FakeSysfsReader()).scan()
    schema = load_report_schema()
    jsonschema.validate(
        to_json_dict(result.observations),
        {"$ref": "#/$defs/ObservedPciObservationList", "$defs": schema["$defs"]},
    )


# --- GPU.PCI_SCAN / NVIDIA.ALL capability policy ------------------------------------------


@environments
def test_capability_policy_for_each_environment(environment, profile):
    records = unsupported_capabilities_for(environment, profile)
    by_id = {r.capability: r for r in records}
    if environment is E.NATIVE_LINUX:
        assert records == ()  # GPU.PCI_SCAN is supported
        return
    pci = by_id[GPU_PCI_SCAN]
    assert pci.status is S.NOT_SUPPORTED and pci.device_ref is None
    assert pci.reason == REASONS[environment]
    expected_ids = {GPU_PCI_SCAN, NVIDIA_ALL} if profile is P.MACOS else {GPU_PCI_SCAN}
    assert set(by_id) == expected_ids


def test_macos_lists_both_pci_scan_and_nvidia_all():
    records = unsupported_capabilities_for(None, P.MACOS)
    assert [(r.capability, r.reason) for r in records] == [
        (GPU_PCI_SCAN, REASONS[None]),
        (NVIDIA_ALL, NVIDIA_MACOS),
    ]


def test_capability_reason_equals_the_scan_message():
    for environment, profile in ENVIRONMENTS.values():
        if environment is E.NATIVE_LINUX:
            continue
        scan = PciSysfsAdapter(environment=environment, reader=FakeSysfsReader()).scan()
        (pci,) = [
            r
            for r in unsupported_capabilities_for(environment, profile)
            if r.capability == GPU_PCI_SCAN
        ]
        assert pci.reason == scan.observations.error.message


@pytest.mark.parametrize("environment", [E.UNKNOWN, E.WSL1], ids=["unknown", "wsl1"])
def test_a_linux_profile_never_makes_a_non_native_environment_eligible(environment):
    records = unsupported_capabilities_for(environment, P.LINUX)
    assert [r.capability for r in records] == [GPU_PCI_SCAN]


@pytest.mark.parametrize(
    ("environment", "profile"),
    [
        (None, P.LINUX),
        (None, P.WSL2),
        (E.NATIVE_LINUX, P.WSL2),
        (E.WSL2, P.LINUX),
        (E.UNKNOWN, P.WINDOWS),
        (E.NATIVE_LINUX, P.MACOS),
        ("NATIVE_LINUX", P.LINUX),
    ],
    ids=repr,
)
def test_inconsistent_environment_and_profile_are_usage_errors(environment, profile):
    with pytest.raises(AdapterUsageError):
        unsupported_capabilities_for(environment, profile)


def test_other_non_linux_platforms_get_only_the_pci_record():
    assert [r.capability for r in unsupported_capabilities_for(None, P.OTHER)] == [GPU_PCI_SCAN]


def test_capability_records_are_valid_models():
    for environment, profile in ENVIRONMENTS.values():
        for record in unsupported_capabilities_for(environment, profile):
            assert isinstance(record, UnsupportedCapability)


# --- fixture parity -----------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_adapter_reproduces_every_fixture_pci_section(fixture):
    snapshot = fixture.snapshot
    environment = snapshot.system.linux_environment.value
    expected = snapshot.gpu.pci_observations
    if expected.is_collected:
        reader = sysfs_reader_for(expected.value)
    else:
        reader = FakeSysfsReader(failures={"list": AssertionError("touched")})
    result = PciSysfsAdapter(environment=environment, reader=reader).scan()
    assert result.observations == expected and result.failures == ()
    if not pci_scan_eligible(environment):
        assert reader.calls == []


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_fixture_pci_and_nvidia_capabilities_follow_the_policy(fixture):
    snapshot = fixture.snapshot
    policy = unsupported_capabilities_for(
        snapshot.system.linux_environment.value, snapshot.system.platform_profile
    )
    listed = tuple(
        u for u in snapshot.unsupported_capabilities if u.capability in (GPU_PCI_SCAN, NVIDIA_ALL)
    )
    assert listed == policy


def test_named_fixture_expectations():
    fx03, fx04b, fx11, fx12, fx13, fx17 = (
        FIXTURES_BY_ID[i].snapshot for i in ("FX03", "FX04b", "FX11", "FX12", "FX13", "FX17")
    )
    # FX03: native Linux, the scan ran and found only a non-NVIDIA (Intel) display controller.
    (intel,) = fx03.gpu.pci_observations.value
    assert fx03.gpu.pci_observations.status is S.COLLECTED
    assert intel.vendor_id == "0x8086" and intel.is_nvidia is False
    assert not any(u.capability == GPU_PCI_SCAN for u in fx03.unsupported_capabilities)
    # FX17: NVIDIA on PCI bound to nouveau, observed (not absent, not unsupported).
    (nvidia,) = fx17.gpu.pci_observations.value
    assert nvidia.is_nvidia and nvidia.kernel_driver.value == "nouveau"
    # FX04b/FX11 (WSL2), FX12 (Windows), FX13 (macOS): NOT_SUPPORTED with the policy reason.
    for snapshot, environment in ((fx04b, E.WSL2), (fx11, E.WSL2), (fx12, None), (fx13, None)):
        pci = snapshot.gpu.pci_observations
        assert pci.status is S.NOT_SUPPORTED and pci.error.code == "PLATFORM_NOT_SUPPORTED"
        assert pci.error.message == REASONS[environment]
    assert {u.capability for u in fx13.unsupported_capabilities} >= {GPU_PCI_SCAN, NVIDIA_ALL}
    # FX13's NVIDIA fields keep their existing platform-gate message.
    assert fx13.gpu.device_count.error.message == "NVIDIA CUDA GPUs are not supported on macOS"
