"""Fake `HostApi`: synthetic, deterministic host primitives for `OsPlatformAdapter` tests.

Profiles are generated from `builders.SYSTEM_PROFILES`, so the production adapter turns
`FakeHostApi.for_profile(p)` into exactly `builders.system_info(p)`. Each profile only
answers its own platform's primitives: a Linux host fails the test if the adapter calls a
Windows or macOS API, and a Windows host fails it if the adapter reads a Linux path.
`failures` makes one primitive (by name, or a file by path) raise. Provenance is FIXTURE,
so the production adapter prefixes every method with `fixture:`.
"""

from __future__ import annotations

import errno
import struct
from dataclasses import dataclass, field

from gpu_system_inspector.models import DataOrigin, PlatformProfile
from gsi_testkit.builders import (
    SYSTEM_AVAILABLE_BYTES,
    SYSTEM_CPUS,
    SYSTEM_PROFILES,
    SYSTEM_TOTAL_BYTES,
    synthetic_proc_version,
)

KIB = 1024
OSRELEASE = "/proc/sys/kernel/osrelease"
PROC_VERSION = "/proc/version"


def linux_files(profile: PlatformProfile) -> dict[str, str]:
    p = SYSTEM_PROFILES[profile]
    return {
        "/proc/meminfo": (
            f"MemTotal:       {SYSTEM_TOTAL_BYTES // KIB} kB\n"
            f"MemFree:        {4 * 2**20} kB\n"
            f"MemAvailable:   {SYSTEM_AVAILABLE_BYTES // KIB} kB\n"
            "SwapTotal:      2097148 kB\n"
        ),
        "/proc/cpuinfo": (
            f"processor\t: 0\nvendor_id\t: SyntheticVendor\nmodel name\t: {p['processor']}\n\n"
            f"processor\t: 1\nmodel name\t: {p['processor']}\n"
        ),
        PROC_VERSION: synthetic_proc_version(p["release"]),
        OSRELEASE: p["release"] + "\n",
        "/etc/os-release": f'NAME="Synthetic Linux"\nPRETTY_NAME="{p["distro"]}"\n',
    }


@dataclass
class FakeHostApi:
    system_name: str
    release_text: str
    version_text: str
    machine_name: str
    files: dict[str, str] | None = None
    environment: dict[str, str] = field(default_factory=dict)
    cpus: int | None = SYSTEM_CPUS
    executable_path: str = "/opt/synthetic/venv/bin/python"
    windows: tuple[str, str] | None = None
    windows_cpu: str | None = None
    windows_mem: tuple[int, int] | None = None
    mac_release: str | None = None
    sysctls: dict[str, bytes] | None = None
    failures: dict[str, Exception] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    env_reads: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> DataOrigin:
        return DataOrigin.FIXTURE

    # --- profiles -----------------------------------------------------------------------

    @classmethod
    def for_profile(cls, profile: PlatformProfile, **changes: object) -> FakeHostApi:
        p = SYSTEM_PROFILES[profile]
        host = cls(p["os_name"], p["release"], p["version"], p["arch"])
        host.executable_path = p["python_exe"]
        if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
            host.files = linux_files(profile)
        elif profile is PlatformProfile.WINDOWS:
            host.windows = (p["release"], p["edition"])
            host.windows_cpu = f"  {p['processor']}  "
            host.windows_mem = (SYSTEM_TOTAL_BYTES, SYSTEM_AVAILABLE_BYTES)
        elif profile is PlatformProfile.MACOS:
            host.mac_release = p["mac_release"]
            host.sysctls = {
                "machdep.cpu.brand_string": p["processor"].encode() + b"\0",
                "hw.memsize": struct.pack("=q", SYSTEM_TOTAL_BYTES),
            }
        return host._with(changes)

    @classmethod
    def linux(cls, **changes: object) -> FakeHostApi:
        return cls.for_profile(PlatformProfile.LINUX, **changes)

    @classmethod
    def wsl2(cls, **changes: object) -> FakeHostApi:
        return cls.for_profile(PlatformProfile.WSL2, **changes)

    @classmethod
    def windows_host(cls, **changes: object) -> FakeHostApi:
        return cls.for_profile(PlatformProfile.WINDOWS, **changes)

    @classmethod
    def macos(cls, **changes: object) -> FakeHostApi:
        return cls.for_profile(PlatformProfile.MACOS, **changes)

    @classmethod
    def other(cls, **changes: object) -> FakeHostApi:
        return cls("FreeBSD", "14.1-RELEASE", "FreeBSD 14.1", "amd64")._with(changes)

    def kernel(self, osrelease: str | None, proc_version: str | None) -> FakeHostApi:
        """Replace the kernel evidence of a Linux host; `None` removes the file."""
        assert self.files is not None, "kernel evidence only exists on Linux hosts"
        for path, text in ((OSRELEASE, osrelease), (PROC_VERSION, proc_version)):
            if text is None:
                self.files.pop(path, None)
            else:
                self.files[path] = text
        return self

    def _with(self, changes: dict[str, object]) -> FakeHostApi:
        for name, value in changes.items():
            if not hasattr(self, name):
                raise AttributeError(name)
            setattr(self, name, value)
        return self

    # --- HostApi ------------------------------------------------------------------------

    def _call(self, name: str) -> None:
        self.calls.append(name)
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def system(self) -> str:
        self._call("system")
        return self.system_name

    def release(self) -> str:
        self._call("release")
        return self.release_text

    def version(self) -> str:
        self._call("version")
        return self.version_text

    def machine(self) -> str:
        self._call("machine")
        return self.machine_name

    def python_version(self) -> str:
        self._call("python_version")
        return "3.12.12"

    def python_implementation(self) -> str:
        self._call("python_implementation")
        return "CPython"

    def executable(self) -> str:
        self._call("executable")
        return self.executable_path

    def cpu_count(self) -> int | None:
        self._call("cpu_count")
        return self.cpus

    def read_text(self, path: str) -> str:
        if self.files is None:
            raise AssertionError(f"{path} read on a {self.system_name} host")
        self._call(path)
        if path not in self.files:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        return self.files[path]

    def getenv(self, name: str) -> str | None:
        self.env_reads.append(name)
        return self.environment.get(name)

    def environ_names(self) -> tuple[str, ...]:
        self._call("environ_names")
        return tuple(self.environment)

    def windows_version(self) -> tuple[str, str]:
        self._call("windows_version")
        return self._only("Windows", self.windows)

    def windows_processor_name(self) -> str:
        self._call("windows_processor_name")
        return self._only("Windows", self.windows_cpu)

    def windows_memory(self) -> tuple[int, int]:
        self._call("windows_memory")
        return self._only("Windows", self.windows_mem)

    def mac_version(self) -> str:
        self._call("mac_version")
        return self._only("macOS", self.mac_release)

    def mac_sysctl(self, name: str) -> bytes:
        self._call(f"sysctl:{name}")
        sysctls = self._only("macOS", self.sysctls)
        if name not in sysctls:
            raise OSError(errno.ENOENT, "No such file or directory", name)
        return sysctls[name]

    def _only(self, platform_name: str, value):
        if value is None:
            raise AssertionError(f"{platform_name} API called on a {self.system_name} host")
        return value
