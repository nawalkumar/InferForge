"""Host integration: the production system adapter and collector on the machine running the
tests. Read-only, no NVIDIA hardware needed.

Nothing here assumes a machine model, OS version, RAM size or architecture. Values are
checked for structure and against independent standard-library signals, and never compared
with fixture data.
"""

import dataclasses
import os
import platform
import subprocess
import sys

import jsonschema
import pytest

from gpu_system_inspector.adapters import OsPlatformAdapter, SystemClock
from gpu_system_inspector.collectors import SystemCollector, collect_environment
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    SystemInfo,
)
from gpu_system_inspector.reporting import from_json_dict, load_report_schema, to_json_dict

pytestmark = pytest.mark.host

EXPECTED_PROFILES = {
    "linux": {PlatformProfile.LINUX, PlatformProfile.WSL2},
    "win32": {PlatformProfile.WINDOWS},
    "darwin": {PlatformProfile.MACOS},
}


@pytest.fixture
def no_processes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("system inspection started a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    monkeypatch.setattr(os, "popen", refuse)
    if hasattr(os, "posix_spawn"):
        monkeypatch.setattr(os, "posix_spawn", refuse)


@pytest.fixture
def host_info(no_processes) -> SystemInfo:
    return OsPlatformAdapter().system_info()


def _observed(info: SystemInfo) -> dict[str, Observed]:
    values = {f.name: getattr(info, f.name) for f in dataclasses.fields(info)}
    return {name: value for name, value in values.items() if isinstance(value, Observed)}


def test_host_observations_are_live_and_never_labelled_as_fixtures(host_info):
    assert OsPlatformAdapter().provenance is DataOrigin.LIVE
    for name, value in _observed(host_info).items():
        assert value.method is None or not value.method.startswith(FIXTURE_METHOD_PREFIX), name
        if value.is_collected:
            assert value.source is not None and value.method, name


def test_profile_matches_the_interpreter_platform(host_info):
    expected = EXPECTED_PROFILES.get(sys.platform, {PlatformProfile.OTHER})
    assert host_info.platform_profile in expected


def test_os_python_and_cpu_facts_agree_with_the_standard_library(host_info):
    pairs = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "logical_cpu_count": os.cpu_count(),
    }
    for name, expected in pairs.items():
        value = getattr(host_info, name)
        if expected:
            assert value.is_collected and value.value == expected, name
        else:
            assert value.status is CollectionStatus.NOT_DETECTED, name


def test_memory_is_structurally_valid(host_info):
    total, available = host_info.memory_total_bytes, host_info.memory_available_bytes
    if host_info.platform_profile in (PlatformProfile.MACOS, PlatformProfile.OTHER):
        assert total.status is CollectionStatus.NOT_SUPPORTED
        return
    assert total.is_collected and isinstance(total.value, int) and total.value > 0
    assert total.unit == "bytes" and total.note
    if available.is_collected:
        assert 0 <= available.value <= total.value


@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="os.sysconf is POSIX-only")
def test_linux_total_memory_agrees_with_sysconf(host_info):
    if host_info.platform_profile not in (PlatformProfile.LINUX, PlatformProfile.WSL2):
        pytest.skip("MemTotal is the Linux source")
    page = os.sysconf("SC_PAGE_SIZE")
    sysconf_total = os.sysconf("SC_PHYS_PAGES") * page
    # MemTotal is reported in whole KiB; sysconf rounds down to whole pages.
    assert abs(host_info.memory_total_bytes.value - sysconf_total) < page + 1024


def test_linux_environment_agrees_with_the_kernel_release(host_info):
    """`platform.release()` (uname) is an independent view of the same kernel release."""
    environment = host_info.linux_environment
    if sys.platform != "linux":
        assert environment.status is CollectionStatus.NOT_APPLICABLE
        return
    assert environment.is_collected and environment.note
    release = platform.release().lower()
    if "-microsoft-standard-wsl2" in release:
        assert environment.value in (LinuxEnvironment.WSL2, LinuxEnvironment.UNKNOWN)
    elif "microsoft" not in release and "wsl" not in release:
        assert environment.value in (LinuxEnvironment.NATIVE_LINUX, LinuxEnvironment.UNKNOWN)
    assert (host_info.platform_profile is PlatformProfile.WSL2) == (
        environment.value is LinuxEnvironment.WSL2
    )


def test_host_system_info_serialises_against_the_schema(host_info):
    schema = load_report_schema()
    data = to_json_dict(host_info)
    jsonschema.validate(data, {"$ref": "#/$defs/SystemInfo", "$defs": schema["$defs"]})
    assert from_json_dict(data, SystemInfo) == host_info


def test_collector_on_the_host(no_processes):
    outcome = SystemCollector(OsPlatformAdapter(), SystemClock()).collect()
    assert outcome.run.status in (CollectorRunStatus.COMPLETED, CollectorRunStatus.PARTIAL)
    assert outcome.run.duration_ms >= 0 and outcome.errors == ()
    problems = {
        CollectionStatus.COLLECTION_FAILED,
        CollectionStatus.PERMISSION_DENIED,
        CollectionStatus.UNKNOWN,
    }
    has_problem = any(v.status in problems for v in _observed(outcome.section).values())
    assert (outcome.run.status is CollectorRunStatus.PARTIAL) == has_problem


def test_environment_matches_the_process_environment_for_allowlisted_names(no_processes):
    environment = collect_environment(OsPlatformAdapter())
    assert list(environment) == sorted(ENVIRONMENT_ALLOWLIST)
    for name, value in environment.items():
        actual = os.environ.get(name)
        if actual is None:
            assert value.status is CollectionStatus.NOT_DETECTED
        else:
            assert value.is_collected and value.value == actual
