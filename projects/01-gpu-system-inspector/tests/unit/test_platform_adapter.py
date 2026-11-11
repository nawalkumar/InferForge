"""`OsPlatformAdapter` on a synthetic `FakeHostApi`: per-platform mechanisms, per-field
degradation, provenance and environment access. No real host is read."""

import ctypes
import ctypes.util
import dataclasses
import errno

import jsonschema
import pytest
from gsi_testkit.fakes import FakeHostApi

from gpu_system_inspector.adapters import AdapterUsageError, OsPlatformAdapter, StdlibHostApi
from gpu_system_inspector.adapters.platform_os import (
    ENVIRONMENT_METHOD,
    LINUX_AVAILABLE_NOTE,
    LINUX_TOTAL_NOTE,
    MACOS_AVAILABLE_REASON,
    MACOS_TOTAL_NOTE,
    WINDOWS_AVAILABLE_NOTE,
    WSL2_MEMORY_NOTE,
)
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    Source,
    SystemInfo,
)
from gpu_system_inspector.reporting import from_json_dict, load_report_schema, to_json_dict

S = CollectionStatus
GIB = 2**30


def _info(host: FakeHostApi) -> SystemInfo:
    return OsPlatformAdapter(host).system_info()


def _observed(info: SystemInfo) -> dict[str, Observed]:
    values = {f.name: getattr(info, f.name) for f in dataclasses.fields(info)}
    return {name: value for name, value in values.items() if isinstance(value, Observed)}


PROFILES = {
    "linux": FakeHostApi.linux,
    "wsl2": FakeHostApi.wsl2,
    "windows": FakeHostApi.windows_host,
    "macos": FakeHostApi.macos,
    "other": FakeHostApi.other,
}
profiles = pytest.mark.parametrize("make", PROFILES.values(), ids=PROFILES.keys())


def test_linux_host_collects_every_field_with_its_source_method_and_unit():
    info = _info(FakeHostApi.linux())
    assert info.platform_profile is PlatformProfile.LINUX
    assert all(v.is_collected for v in _observed(info).values())
    assert info.os_name.value == "Linux" and info.os_name.source is Source.PLATFORM
    assert info.os_name.method == "fixture:platform.system()"
    assert info.os_distribution.value == "Synthetic Linux 24.04 (fixture)"
    assert info.os_distribution.method == "fixture:/etc/os-release:PRETTY_NAME"
    assert info.processor_model.value == "Synthetic CPU (fixture)"
    assert info.processor_model.method == "fixture:/proc/cpuinfo:model name"
    assert info.processor_model.source is Source.OS
    assert info.memory_total_bytes.value == 64 * GIB and info.memory_total_bytes.unit == "bytes"
    assert info.memory_total_bytes.note == LINUX_TOTAL_NOTE
    assert info.memory_available_bytes.value == 48 * GIB
    assert info.memory_available_bytes.note == LINUX_AVAILABLE_NOTE
    assert info.logical_cpu_count.value == 16 and info.logical_cpu_count.unit is None
    environment = info.linux_environment
    assert environment.value is LinuxEnvironment.NATIVE_LINUX
    assert environment.source is Source.INSPECTOR
    assert environment.method == "fixture:" + ENVIRONMENT_METHOD
    assert info.python_executable.value == "/opt/synthetic/venv/bin/python"


def test_wsl2_is_classified_from_kernel_evidence_and_memory_is_labelled_as_the_vm():
    info = _info(FakeHostApi.wsl2())
    assert info.platform_profile is PlatformProfile.WSL2
    assert info.linux_environment.value is LinuxEnvironment.WSL2
    assert info.memory_total_bytes.note.endswith(WSL2_MEMORY_NOTE)


def test_windows_host_uses_registry_and_global_memory_status_never_linux_paths():
    host = FakeHostApi.windows_host()
    info = _info(host)
    assert info.platform_profile is PlatformProfile.WINDOWS
    linux_only = {"linux_environment"}
    assert all(v.is_collected for k, v in _observed(info).items() if k not in linux_only)
    assert info.os_distribution.value == "Windows 11 Professional"
    assert info.processor_model.value == "Synthetic CPU (fixture)"
    assert "ProcessorNameString" in info.processor_model.method
    assert info.memory_total_bytes.value == 64 * GIB
    assert info.memory_available_bytes.value == 48 * GIB
    assert info.memory_available_bytes.note == WINDOWS_AVAILABLE_NOTE
    assert info.linux_environment.status is S.NOT_APPLICABLE
    assert host.calls.count("windows_memory") == 1  # one call feeds both memory fields
    assert not any(c.startswith("/") for c in host.calls)


def test_macos_uses_read_only_sysctlbyname_and_keeps_available_memory_unsupported():
    host = FakeHostApi.macos()
    info = _info(host)
    assert info.platform_profile is PlatformProfile.MACOS
    assert info.os_distribution.value == "macOS 15.0"
    processor, total = info.processor_model, info.memory_total_bytes
    assert processor.value == "Synthetic Apple Silicon (fixture)"
    assert processor.method == "fixture:sysctlbyname(machdep.cpu.brand_string)"
    assert total.value == 64 * GIB and total.unit == "bytes" and total.note == MACOS_TOTAL_NOTE
    assert total.method == "fixture:sysctlbyname(hw.memsize)"
    available = info.memory_available_bytes
    assert available.status is S.NOT_SUPPORTED and available.value is None
    assert available.error.message == MACOS_AVAILABLE_REASON and available.unit == "bytes"
    assert info.linux_environment.status is S.NOT_APPLICABLE
    assert sorted(c for c in host.calls if c.startswith("sysctl:")) == [
        "sysctl:hw.memsize",
        "sysctl:machdep.cpu.brand_string",
    ]
    assert not any(c.startswith("/") or c.startswith("windows") for c in host.calls)


@pytest.mark.parametrize(
    ("sysctls", "field", "status", "code"),
    [
        ({"hw.memsize": b"\x00" * 8}, "processor_model", S.NOT_DETECTED, "FileNotFoundError"),
        ({"machdep.cpu.brand_string": b"\0"}, "processor_model", S.NOT_DETECTED, "FIELD_NOT_FOUND"),
        (
            {"machdep.cpu.brand_string": b"\xff\xfe\0"},
            "processor_model",
            S.COLLECTION_FAILED,
            "PARSE_ERROR",
        ),
        (
            {"hw.memsize": b"\x01\x02\x03\x04"},
            "memory_total_bytes",
            S.COLLECTION_FAILED,
            "PARSE_ERROR",
        ),
        ({"hw.memsize": b"\x00" * 8}, "memory_total_bytes", S.COLLECTION_FAILED, "PARSE_ERROR"),
    ],
    ids=["unknown-name", "empty-string", "invalid-utf8", "short-int64", "zero-memory"],
)
def test_macos_sysctl_values_are_decoded_strictly(sysctls, field, status, code):
    host = FakeHostApi.macos()
    host.sysctls = {**host.sysctls, **sysctls}
    if field == "processor_model" and "machdep.cpu.brand_string" not in sysctls:
        del host.sysctls["machdep.cpu.brand_string"]
    value = getattr(_info(host), field)
    assert (value.status, value.error.code) == (status, code) and value.value is None


def test_macos_sysctl_permission_error_affects_only_its_field():
    host = FakeHostApi.macos(failures={"sysctl:hw.memsize": PermissionError(errno.EPERM, "no")})
    info = _info(host)
    assert info.memory_total_bytes.status is S.PERMISSION_DENIED
    assert info.processor_model.is_collected and info.os_name.is_collected


class _SimulatedSysctl:
    """Stands in for libc `sysctlbyname` with the man-page semantics; records every call."""

    def __init__(self, values: dict[bytes, bytes]):
        self.values, self.calls, self.errno = values, [], 0
        self.argtypes = self.restype = None

    def __call__(self, name, oldp, oldlenp, newp, newlen):
        self.calls.append((name, oldp is None, newp, newlen))
        size = oldlenp._obj
        if name not in self.values:
            self.errno = errno.ENOENT
            return -1
        data = self.values[name]
        if oldp is not None:
            ctypes.memmove(oldp, data, len(data))
        size.value = len(data)
        return 0


def _patch_libc(monkeypatch, sysctl, library="/usr/lib/libc.dylib"):
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: library)
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path, use_errno: type("Libc", (), {"sysctlbyname": sysctl})()
    )
    monkeypatch.setattr(ctypes, "get_errno", lambda: sysctl.errno)


def test_stdlib_mac_sysctl_follows_the_documented_two_call_protocol(monkeypatch):
    brand = b"Synthetic Apple Silicon (fixture)\0"
    sysctl = _SimulatedSysctl({b"machdep.cpu.brand_string": brand})
    _patch_libc(monkeypatch, sysctl)
    assert StdlibHostApi().mac_sysctl("machdep.cpu.brand_string") == brand
    assert sysctl.argtypes == (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    assert sysctl.restype is ctypes.c_int
    # Sizing call (oldp NULL), then the read; never a write (newp NULL, newlen 0).
    assert sysctl.calls == [
        (b"machdep.cpu.brand_string", True, None, 0),
        (b"machdep.cpu.brand_string", False, None, 0),
    ]


def test_stdlib_mac_sysctl_maps_errno_and_a_missing_libc(monkeypatch):
    _patch_libc(monkeypatch, _SimulatedSysctl({}))
    with pytest.raises(FileNotFoundError):  # ENOENT: unknown sysctl name
        StdlibHostApi().mac_sysctl("hw.memsize")
    _patch_libc(monkeypatch, _SimulatedSysctl({}), library=None)
    with pytest.raises(FileNotFoundError, match="C library"):
        StdlibHostApi().mac_sysctl("hw.memsize")


class _GrowingSysctl(_SimulatedSysctl):
    """The value grows between the sizing call and the read, so the read fails (ENOMEM)."""

    def __call__(self, name, oldp, oldlenp, newp, newlen):
        if oldp is None:
            return super().__call__(name, oldp, oldlenp, newp, newlen)
        self.errno = errno.ENOMEM
        return -1


def test_stdlib_mac_sysctl_reports_a_failing_read_after_sizing(monkeypatch):
    _patch_libc(monkeypatch, _GrowingSysctl({b"hw.memsize": b"\x00" * 8}))
    with pytest.raises(OSError) as exc:
        StdlibHostApi().mac_sysctl("hw.memsize")
    assert exc.value.errno == errno.ENOMEM


def test_other_platforms_report_not_supported_without_guessing():
    info = _info(FakeHostApi.other())
    assert info.platform_profile is PlatformProfile.OTHER
    for name in ("os_distribution", "processor_model", "memory_total_bytes"):
        value = getattr(info, name)
        assert value.status is S.NOT_SUPPORTED and value.value is None
        assert value.error.code == "PLATFORM_NOT_SUPPORTED" and value.method is None
    assert info.memory_available_bytes.status is S.NOT_SUPPORTED
    assert info.memory_total_bytes.unit == "bytes"
    assert info.linux_environment.status is S.NOT_APPLICABLE
    for name in ("os_name", "os_release", "architecture", "python_version"):
        assert getattr(info, name).is_collected


@pytest.mark.parametrize(
    ("cpuinfo", "message"),
    [
        ("processor\t: 0\nHardware\t: Synthetic SoC\n", "no 'model name' entry"),
        ("processor\t: 0\nmodel name\t:   \n", "empty 'model name' entry"),
    ],
    ids=["no-entry", "empty-entry"],
)
def test_missing_processor_description_is_not_detected(cpuinfo, message):
    host = FakeHostApi.linux()
    host.files["/proc/cpuinfo"] = cpuinfo
    processor = _info(host).processor_model
    assert processor.status is S.NOT_DETECTED and processor.value is None
    assert (processor.error.code, processor.error.message) == ("FIELD_NOT_FOUND", message)


def test_missing_windows_processor_key_is_not_detected():
    host = FakeHostApi.windows_host(failures={"windows_processor_name": FileNotFoundError("key")})
    assert _info(host).processor_model.status is S.NOT_DETECTED


def test_missing_logical_cpu_count_is_not_detected_not_zero():
    count = _info(FakeHostApi.linux(cpus=None)).logical_cpu_count
    assert count.status is S.NOT_DETECTED and count.value is None


def test_missing_total_memory_is_not_detected_and_available_is_unaffected():
    host = FakeHostApi.linux()
    host.files["/proc/meminfo"] = "MemFree: 1024 kB\nMemAvailable: 2048 kB\n"
    info = _info(host)
    assert info.memory_total_bytes.status is S.NOT_DETECTED
    assert info.memory_available_bytes.value == 2048 * 1024


def test_missing_available_memory_is_never_replaced_by_free_memory():
    host = FakeHostApi.linux()
    host.files["/proc/meminfo"] = "MemTotal: 4096 kB\nMemFree: 1024 kB\n"  # pre-3.14 kernel
    info = _info(host)
    assert info.memory_total_bytes.value == 4096 * 1024
    available = info.memory_available_bytes
    assert available.status is S.NOT_DETECTED and available.error.message == "no MemAvailable entry"


@pytest.mark.parametrize(
    ("host", "fields"),
    [
        (
            FakeHostApi.linux(failures={"/proc/meminfo": PermissionError(errno.EACCES, "denied")}),
            ("memory_total_bytes", "memory_available_bytes"),
        ),
        (
            FakeHostApi.windows_host(failures={"windows_processor_name": PermissionError("no")}),
            ("processor_model",),
        ),
    ],
    ids=["linux-meminfo", "windows-registry"],
)
def test_permission_denied_affects_only_its_own_fields(host, fields):
    info = _info(host)
    for name, value in _observed(info).items():
        if name in fields:
            assert value.status is S.PERMISSION_DENIED and value.error.code == "PermissionError"
        elif name == "linux_environment" and info.platform_profile is PlatformProfile.WINDOWS:
            assert value.status is S.NOT_APPLICABLE
        else:
            assert value.is_collected, name


@pytest.mark.parametrize(
    ("host", "field", "code"),
    [
        (
            FakeHostApi.linux(failures={"/proc/cpuinfo": OSError(errno.EIO, "I/O error")}),
            "processor_model",
            "OSError",
        ),
        (
            FakeHostApi.windows_host(failures={"windows_memory": OSError("GlobalMemoryStatusEx")}),
            "memory_available_bytes",
            "OSError",
        ),
    ],
    ids=["linux-io-error", "windows-api-error"],
)
def test_individual_collection_failures_are_contained(host, field, code):
    info = _info(host)
    value = getattr(info, field)
    assert value.status is S.COLLECTION_FAILED and value.error.code == code
    assert info.os_name.is_collected and info.python_version.is_collected


def test_malformed_meminfo_is_a_parse_error():
    host = FakeHostApi.linux()
    host.files["/proc/meminfo"] = "MemTotal: lots\nMemAvailable: 12 MB\n"
    info = _info(host)
    for value in (info.memory_total_bytes, info.memory_available_bytes):
        assert value.status is S.COLLECTION_FAILED and value.error.code == "PARSE_ERROR"


def test_meminfo_is_read_once_per_call():
    host = FakeHostApi.linux()
    _info(host)
    assert host.calls.count("/proc/meminfo") == 1


def test_unreadable_proc_version_leaves_the_environment_unknown_and_profile_linux():
    host = FakeHostApi.wsl2(failures={"/proc/version": PermissionError("denied")})
    info = _info(host)
    assert info.platform_profile is PlatformProfile.LINUX
    assert info.linux_environment.value is LinuxEnvironment.UNKNOWN
    assert "PermissionError" in info.linux_environment.note


def test_os_release_falls_back_to_usr_lib_and_unquotes():
    host = FakeHostApi.linux()
    del host.files["/etc/os-release"]
    host.files["/usr/lib/os-release"] = 'PRETTY_NAME="Synthetic \\"Edge\\" (fixture)"\n'
    distribution = _info(host).os_distribution
    assert distribution.value == 'Synthetic "Edge" (fixture)'
    assert distribution.method == "fixture:/usr/lib/os-release:PRETTY_NAME"
    del host.files["/usr/lib/os-release"]
    assert _info(host).os_distribution.status is S.NOT_DETECTED


@pytest.mark.parametrize(
    ("content", "status", "value"),
    [
        (
            "PRETTY_NAME='Synthetic \\Single (fixture)'\n",
            S.COLLECTED,
            "Synthetic \\Single (fixture)",
        ),
        ('PRETTY_NAME=""\n', S.NOT_DETECTED, None),
        ('NAME="Synthetic"\n', S.NOT_DETECTED, None),
    ],
    ids=["single-quoted-verbatim", "empty", "absent"],
)
def test_pretty_name_parsing(content, status, value):
    host = FakeHostApi.linux()
    host.files["/etc/os-release"] = content
    distribution = _info(host).os_distribution
    assert (distribution.status, distribution.value) == (status, value)


def test_os_release_permission_error_is_reported():
    host = FakeHostApi.linux(failures={"/etc/os-release": PermissionError("denied")})
    assert _info(host).os_distribution.status is S.PERMISSION_DENIED


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"machine_name": ""}, "architecture"),
        ({"executable_path": ""}, "python_executable"),
        ({"windows": ("", "")}, "os_distribution"),
    ],
    ids=["architecture", "executable", "windows-release"],
)
def test_empty_values_are_not_detected_never_fabricated(changes, field):
    make = FakeHostApi.windows_host if "windows" in changes else FakeHostApi.linux
    value = getattr(_info(make(**changes)), field)
    assert value.status is S.NOT_DETECTED and value.value is None


def test_programming_errors_are_not_swallowed():
    host = FakeHostApi.linux(failures={"python_version": RuntimeError("bug")})
    with pytest.raises(RuntimeError, match="bug"):
        _info(host)


@profiles
def test_every_synthetic_observation_carries_the_fixture_prefix(make):
    adapter = OsPlatformAdapter(make())
    assert adapter.provenance.value == "FIXTURE"
    for value in _observed(adapter.system_info()).values():
        assert value.method is None or value.method.startswith(FIXTURE_METHOD_PREFIX), value


@profiles
def test_system_info_reads_no_environment_variable(make):
    host = make()
    OsPlatformAdapter(host).system_info()
    assert host.env_reads == []


@profiles
def test_synthetic_system_info_serialises_against_the_schema(make):
    info = _info(make())
    schema = load_report_schema()
    data = to_json_dict(info)
    jsonschema.validate(data, {"$ref": "#/$defs/SystemInfo", "$defs": schema["$defs"]})
    assert from_json_dict(data, SystemInfo) == info


def test_getenv_reads_only_allowlisted_variables():
    host = FakeHostApi.linux(
        environment={"CUDA_VISIBLE_DEVICES": "", "CUDA_HOME": "/opt/cuda", "API_TOKEN": "secret"}
    )
    adapter = OsPlatformAdapter(host)
    assert adapter.getenv("CUDA_VISIBLE_DEVICES") == Observed.collected(
        "", source=Source.ENVIRONMENT, method="fixture:os.environ"
    )
    assert adapter.getenv("CUDA_HOME").value == "/opt/cuda"
    unset = adapter.getenv("CUDA_DEVICE_ORDER")
    assert unset.status is S.NOT_DETECTED and unset.error.code == "NOT_SET"
    for name in ("API_TOKEN", "HOME", "PATH", "WSL_DISTRO_NAME"):
        with pytest.raises(AdapterUsageError, match="allowlisted"):
            adapter.getenv(name)
    assert host.env_reads == ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_DEVICE_ORDER"]
    assert set(host.env_reads) <= set(ENVIRONMENT_ALLOWLIST)


def test_profile_is_detected_once():
    host = FakeHostApi.wsl2()
    adapter = OsPlatformAdapter(host)
    adapter.system_info()
    adapter.system_info()
    assert adapter.profile is PlatformProfile.WSL2
    assert host.calls.count("/proc/version") == 1
