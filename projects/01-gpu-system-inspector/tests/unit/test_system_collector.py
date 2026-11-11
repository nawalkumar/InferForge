"""`SystemCollector` and `collect_environment` on fake adapters and a fixed clock."""

import dataclasses

import pytest
from gsi_testkit.fakes import FakeAdapterSet, FakeHostApi, FakePlatform, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import (
    AdapterUsageError,
    OsPlatformAdapter,
    StdlibHostApi,
)
from gpu_system_inspector.collectors import SystemCollector, collect_environment
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    CollectorRunStatus,
    ErrorDetail,
    PlatformProfile,
)

ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)


def _collect(platform):
    return SystemCollector(platform, FixedClock()).collect()


@ids
def test_collector_reproduces_each_fixture_system_section(fixture):
    outcome = _collect(FakePlatform.from_snapshot(fixture.snapshot))
    snapshot = fixture.snapshot
    assert outcome.section == snapshot.system
    (expected_run,) = [r for r in snapshot.collector_runs if r.collector == "system"]
    assert outcome.run.status is expected_run.status
    assert outcome.errors == ()
    assert outcome.unsupported == tuple(
        u for u in snapshot.unsupported_capabilities if u.capability.startswith("SYSTEM.")
    )


def test_duration_comes_from_the_injected_clock():
    clock = FixedClock(step_ms=7)
    outcome = SystemCollector(OsPlatformAdapter(FakeHostApi.linux()), clock).collect()
    assert outcome.run.duration_ms == 7
    again = SystemCollector(OsPlatformAdapter(FakeHostApi.linux()), FixedClock(step_ms=7)).collect()
    assert again == outcome


@pytest.mark.parametrize(
    ("host", "status"),
    [
        (FakeHostApi.linux(), CollectorRunStatus.COMPLETED),
        (FakeHostApi.windows_host(), CollectorRunStatus.COMPLETED),
        (FakeHostApi.linux(cpus=None), CollectorRunStatus.COMPLETED),  # absent, not failed
        (
            FakeHostApi.linux(failures={"/proc/meminfo": PermissionError("denied")}),
            CollectorRunStatus.PARTIAL,
        ),
        (
            FakeHostApi.windows_host(failures={"windows_memory": OSError("api")}),
            CollectorRunStatus.PARTIAL,
        ),
    ],
    ids=["linux", "windows", "no-cpu-count", "permission-denied", "api-failure"],
)
def test_run_status_is_partial_only_for_collection_problems(host, status):
    outcome = _collect(OsPlatformAdapter(host))
    assert outcome.run.status is status and outcome.errors == ()


@pytest.mark.parametrize(
    ("host", "capabilities"),
    [
        (FakeHostApi.macos(), ["SYSTEM.MEMORY_AVAILABLE"]),
        (
            FakeHostApi.other(),
            [
                "SYSTEM.OS_DISTRIBUTION",
                "SYSTEM.PROCESSOR_MODEL",
                "SYSTEM.MEMORY_TOTAL",
                "SYSTEM.MEMORY_AVAILABLE",
            ],
        ),
    ],
    ids=["macos", "other"],
)
def test_unsupported_capabilities_are_listed_per_platform(host, capabilities):
    outcome = _collect(OsPlatformAdapter(host))
    assert outcome.run.status is CollectorRunStatus.COMPLETED
    assert [u.capability for u in outcome.unsupported] == capabilities
    assert all(u.status is CollectionStatus.NOT_SUPPORTED and u.reason for u in outcome.unsupported)


def test_unavailable_section_keeps_the_platform_profile():
    collector = SystemCollector(OsPlatformAdapter(FakeHostApi.windows_host()), FixedClock())
    error = ErrorDetail(code="UNHANDLED_EXCEPTION", message="RuntimeError: bug")
    section = collector.unavailable(error)
    assert section.platform_profile is PlatformProfile.WINDOWS
    assert section.os_name.status is CollectionStatus.COLLECTION_FAILED
    assert section.os_name.error == error


def test_adapter_bugs_propagate_to_the_guard():
    host = FakeHostApi.linux(failures={"release": RuntimeError("bug")})
    with pytest.raises(RuntimeError, match="bug"):
        _collect(OsPlatformAdapter(host))


def test_fake_backed_sections_stay_synthetic():
    outcome = _collect(OsPlatformAdapter(FakeHostApi.wsl2()))
    methods = [
        getattr(outcome.section, f.name).method
        for f in dataclasses.fields(outcome.section)
        if f.name != "platform_profile"
    ]
    assert all(m is None or m.startswith(FIXTURE_METHOD_PREFIX) for m in methods)


def test_adapter_set_refuses_to_mix_live_and_synthetic_adapters():
    fakes = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX01"])
    synthetic = fakes.adapter_set()
    accepted = dataclasses.replace(synthetic, platform=OsPlatformAdapter(FakeHostApi.linux()))
    assert accepted.platform.provenance is synthetic.provenance
    with pytest.raises(TypeError, match="provenance LIVE"):
        dataclasses.replace(synthetic, platform=OsPlatformAdapter(StdlibHostApi()))


# --- environment --------------------------------------------------------------------------


def _full_environment_host(**values: str) -> FakeHostApi:
    return FakeHostApi.linux(environment={"AWS_SECRET_ACCESS_KEY": "secret", **values})


def test_environment_reads_exactly_the_allowlist():
    host = _full_environment_host(CUDA_HOME="/opt/cuda", CUDA_VISIBLE_DEVICES="0,1")
    environment = collect_environment(OsPlatformAdapter(host))
    assert list(environment) == sorted(ENVIRONMENT_ALLOWLIST)
    assert sorted(host.env_reads) == sorted(ENVIRONMENT_ALLOWLIST)
    assert "AWS_SECRET_ACCESS_KEY" not in host.env_reads
    assert environment["CUDA_HOME"].value == "/opt/cuda"
    assert environment["CUDA_VISIBLE_DEVICES"].value == "0,1"
    missing = [n for n, v in environment.items() if not v.is_collected]
    assert len(missing) == len(ENVIRONMENT_ALLOWLIST) - 2
    assert all(environment[n].status is CollectionStatus.NOT_DETECTED for n in missing)


def test_environment_rejects_non_allowlisted_names_before_reading_anything():
    host = _full_environment_host()
    with pytest.raises(AdapterUsageError, match="AWS_SECRET_ACCESS_KEY"):
        collect_environment(OsPlatformAdapter(host), ["CUDA_HOME", "AWS_SECRET_ACCESS_KEY"])
    assert host.env_reads == []


@ids
def test_environment_from_fixture_platform_matches_the_fixture(fixture):
    expected = fixture.snapshot.cuda_environment.environment
    platform = FakePlatform.from_snapshot(fixture.snapshot)
    assert collect_environment(platform, expected) == dict(expected)
