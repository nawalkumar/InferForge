"""Linux environment classification (data model §4.1): the documented conservative rule,
checked directly and through the production adapter on synthetic kernel evidence."""

import dataclasses
import errno

import jsonschema
import pytest
from gsi_testkit.builders import system_info
from gsi_testkit.fakes import FakeHostApi
from gsi_testkit.fixtures import ALL_FIXTURES

from gpu_system_inspector.adapters import OsPlatformAdapter
from gpu_system_inspector.adapters.platform_os import classify_linux_environment
from gpu_system_inspector.collectors import SystemCollector
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    InvalidModelError,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
)
from gpu_system_inspector.reporting import load_report_schema, to_json_dict

E = LinuxEnvironment
OSRELEASE = "/proc/sys/kernel/osrelease"
WSL1_BUILD = "(Microsoft@Microsoft.com) (gcc version 5.4.0 (GCC) ) #1237-Microsoft Sat Sep 11 2021"


def version(release: str, rest: str = "(fixture@synthetic) (gcc 13.2.0) #1 SMP synthetic") -> str:
    return f"Linux version {release} {rest}\n"


def consistent(release: str, rest: str | None = None) -> tuple[str, str]:
    return release + "\n", version(release, rest) if rest else version(release)


MATRIX = [
    # (id, osrelease, /proc/version, expected, rationale keyword)
    ("native", *consistent("6.8.0-synthetic"), E.NATIVE_LINUX, "No Microsoft or WSL marker"),
    ("native-azure-vm", *consistent("5.15.0-1052-azure"), E.NATIVE_LINUX, "virtual machine"),
    ("native-cloud-kernel", *consistent("6.1.0-18-cloud-amd64"), E.NATIVE_LINUX, "not WSL"),
    ("native-kvm-suffix", *consistent("6.5.0-kvm"), E.NATIVE_LINUX, "not WSL"),
    ("wsl1", *consistent("4.4.0-19041-Microsoft", WSL1_BUILD), E.WSL1, "WSL1 format"),
    ("wsl2", *consistent("5.15.167.4-microsoft-standard-WSL2"), E.WSL2, "WSL2 kernel marker"),
    ("wsl2-custom-build", *consistent("6.6.36.3-microsoft-standard-WSL2+"), E.WSL2, "WSL2"),
    ("case-wsl2", *consistent("5.15.167.4-MICROSOFT-STANDARD-wsl2"), E.WSL2, "WSL2"),
    ("case-wsl1", *consistent("4.4.0-19041-microsoft", WSL1_BUILD), E.WSL1, "WSL1"),
    ("case-native", *consistent("6.8.0-SYNTHETIC"), E.NATIVE_LINUX, "not WSL"),
    ("old-wsl2-name", *consistent("4.19.128-microsoft-standard"), E.UNKNOWN, "can't be determined"),
    ("microsoft-custom", *consistent("5.15.0-microsoft-custom"), E.UNKNOWN, "can't be determined"),
    (
        "marker-only-in-version",
        *consistent("6.8.0-synthetic", "(builder@microsoft.com) (gcc) #1 SMP"),
        E.UNKNOWN,
        "Microsoft/WSL marker",
    ),
    (
        "wsl2-substring-not-suffix",
        *consistent("5.15.0-microsoft-standard-WSL2x"),
        E.UNKNOWN,
        "can't be determined",
    ),
    (
        "conflicting-releases",
        "5.15.167.4-microsoft-standard-WSL2\n",
        version("6.8.0-synthetic"),
        E.UNKNOWN,
        "Conflicting",
    ),
    (
        "wsl1-release-but-wsl2-text",
        *consistent("4.4.0-19041-Microsoft", "(Microsoft@Microsoft.com) WSL2 #1"),
        E.UNKNOWN,
        "conflicting",
    ),
    ("empty-osrelease", "", version("6.8.0-synthetic"), E.UNKNOWN, "empty or malformed"),
    ("blank-osrelease", "  \n", version("6.8.0-synthetic"), E.UNKNOWN, "empty or malformed"),
    ("spaced-osrelease", "6.8.0 synthetic\n", version("6.8.0"), E.UNKNOWN, "empty or malformed"),
    ("empty-version", "6.8.0-synthetic\n", "", E.UNKNOWN, "empty or malformed"),
    ("malformed-version", "6.8.0-synthetic\n", "garbage\n", E.UNKNOWN, "empty or malformed"),
]


@pytest.mark.parametrize(
    ("osrelease", "proc_version", "expected", "keyword"),
    [row[1:] for row in MATRIX],
    ids=[row[0] for row in MATRIX],
)
def test_rule(osrelease, proc_version, expected, keyword):
    environment, rationale = classify_linux_environment(osrelease, proc_version)
    assert environment is expected
    assert keyword in rationale


@pytest.mark.parametrize(
    ("osrelease", "proc_version", "expected", "keyword"),
    [row[1:] for row in MATRIX],
    ids=[row[0] for row in MATRIX],
)
def test_adapter_applies_the_rule_and_derives_the_profile(
    osrelease, proc_version, expected, keyword
):
    info = OsPlatformAdapter(FakeHostApi.linux().kernel(osrelease, proc_version)).system_info()
    value = info.linux_environment
    assert value.status is CollectionStatus.COLLECTED and value.value is expected
    assert keyword in value.note and value.method.startswith("fixture:")
    expected_profile = PlatformProfile.WSL2 if expected is E.WSL2 else PlatformProfile.LINUX
    assert info.platform_profile is expected_profile


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(errno.ENOENT, "missing"),
        PermissionError(errno.EACCES, "denied"),
        OSError(errno.EIO, "I/O error"),
    ],
    ids=["missing", "permission", "io-error"],
)
@pytest.mark.parametrize("path", [OSRELEASE, "/proc/version"], ids=["osrelease", "version"])
def test_unreadable_evidence_is_unknown_even_on_a_wsl2_host(path, failure):
    host = FakeHostApi.wsl2(failures={path: failure})
    info = OsPlatformAdapter(host).system_info()
    assert info.linux_environment.value is E.UNKNOWN
    assert type(failure).__name__ in info.linux_environment.note
    assert info.platform_profile is PlatformProfile.LINUX


def test_missing_file_is_unknown():
    host = FakeHostApi.wsl2().kernel(None, version("5.15.167.4-microsoft-standard-WSL2"))
    assert OsPlatformAdapter(host).system_info().linux_environment.value is E.UNKNOWN


@pytest.mark.parametrize(
    "make", [FakeHostApi.windows_host, FakeHostApi.macos, FakeHostApi.other], ids=str
)
def test_non_linux_platforms_are_not_applicable(make):
    host = make()
    value = OsPlatformAdapter(host).system_info().linux_environment
    assert value.status is CollectionStatus.NOT_APPLICABLE and value.value is None
    assert value.note and value.error is None
    assert not any(c.startswith("/proc") for c in host.calls)


def test_other_fields_are_collected_when_classification_fails():
    host = FakeHostApi.wsl2(
        failures={OSRELEASE: PermissionError("denied"), "/proc/version": PermissionError("x")}
    )
    outcome = SystemCollector(OsPlatformAdapter(host), clock=_Clock()).collect()
    info = outcome.section
    assert info.linux_environment.value is E.UNKNOWN
    others = [
        getattr(info, f.name)
        for f in dataclasses.fields(info)
        if f.name not in ("platform_profile", "linux_environment")
    ]
    assert others and all(isinstance(v, Observed) and v.is_collected for v in others)
    assert outcome.run.status is CollectorRunStatus.COMPLETED


def test_classification_reads_each_kernel_source_once():
    host = FakeHostApi.wsl2()
    adapter = OsPlatformAdapter(host)
    adapter.system_info()
    adapter.system_info()
    assert host.calls.count(OSRELEASE) == host.calls.count("/proc/version") == 1


class _Clock:
    def now_utc(self):  # pragma: no cover - unused by the collector
        raise AssertionError

    def monotonic_ms(self) -> int:
        return 0


# --- model and schema consistency ---------------------------------------------------------


def _classified(profile: PlatformProfile, value: LinuxEnvironment):
    base = system_info(PlatformProfile.LINUX)
    environment = dataclasses.replace(base.linux_environment, value=value)
    return dict(platform_profile=profile, linux_environment=environment)


@pytest.mark.parametrize(
    ("profile", "value", "message"),
    [
        (PlatformProfile.WINDOWS, E.NATIVE_LINUX, "must not be classified on WINDOWS"),
        (PlatformProfile.MACOS, E.UNKNOWN, "must not be classified on MACOS"),
        (PlatformProfile.LINUX, E.WSL2, "requires platform_profile WSL2"),
        (PlatformProfile.WSL2, E.WSL1, "requires platform_profile LINUX"),
        (PlatformProfile.WSL2, E.UNKNOWN, "requires platform_profile LINUX"),
    ],
    ids=["windows", "macos", "wsl2-on-linux", "wsl1-on-wsl2", "unknown-on-wsl2"],
)
def test_classification_and_profile_must_agree_in_model_and_schema(profile, value, message):
    base = system_info(PlatformProfile.LINUX)
    with pytest.raises(InvalidModelError, match=message):
        dataclasses.replace(base, **_classified(profile, value))
    data = to_json_dict(base)
    data["platform_profile"] = profile.value
    data["linux_environment"]["value"] = value.value
    schema = load_report_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, {"$ref": "#/$defs/SystemInfo", "$defs": schema["$defs"]})


# --- fixtures agree with production -------------------------------------------------------

PROFILES = [
    PlatformProfile.LINUX,
    PlatformProfile.WSL2,
    PlatformProfile.WINDOWS,
    PlatformProfile.MACOS,
]


@pytest.mark.parametrize("profile", PROFILES, ids=str)
def test_production_on_the_synthetic_host_reproduces_the_fixture_section(profile):
    assert OsPlatformAdapter(FakeHostApi.for_profile(profile)).system_info() == system_info(profile)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
def test_no_fixture_claims_a_capability_production_lacks(fixture):
    """Per field: same status, source, method and unit as production on that platform."""
    expected = fixture.snapshot.system
    actual = OsPlatformAdapter(FakeHostApi.for_profile(expected.platform_profile)).system_info()
    assert actual.platform_profile is expected.platform_profile
    for name in (f.name for f in dataclasses.fields(expected)):
        want, got = getattr(expected, name), getattr(actual, name)
        if not isinstance(want, Observed):
            continue
        if want.status is CollectionStatus.NOT_DETECTED and name == "processor_model":
            continue  # FX variants whose synthetic CPU lacks a model name entry
        assert (got.status, got.source, got.method, got.unit) == (
            want.status,
            want.source,
            want.method,
            want.unit,
        ), name
