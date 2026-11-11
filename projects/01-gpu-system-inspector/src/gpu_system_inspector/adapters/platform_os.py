"""Production `PlatformAdapter` (architecture §4.3, §4.5): OS, CPU, memory and Python facts
from the standard library, plus allowlisted environment reads.

Linux and WSL use `platform`, `os.cpu_count()`, `/proc` and `os-release`. Windows uses
`platform`, the registry (`winreg`, read-only) and `GlobalMemoryStatusEx` (`ctypes`). macOS
uses `platform` and read-only `sysctlbyname(3)` (`ctypes`) for the processor model and total
memory; available memory is `NOT_SUPPORTED` there in V1. On other systems the processor
model and memory are `NOT_SUPPORTED`. Nothing here starts a process, writes anything or
reads an environment variable outside the allowlist; `platform.processor()` is deliberately
avoided because it runs `uname -p` on POSIX and reads `PROCESSOR_IDENTIFIER` on Windows.
"""

from __future__ import annotations

import os
import platform
import re
import struct
import sys
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from gpu_system_inspector.adapters.outcomes import (
    FIELD_NOT_FOUND,
    NOT_SET,
    PARSE_ERROR,
    PLATFORM_NOT_SUPPORTED,
    AdapterUsageError,
    exception_outcome,
)
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    DataOrigin,
    LinuxEnvironment,
    Observed,
    PlatformProfile,
    Source,
    SystemInfo,
)
from gpu_system_inspector.models.cuda import CUDA_PATH_VERSIONED_REGEX, ENVIRONMENT_NAME_REGEX

PROC_MEMINFO = "/proc/meminfo"
PROC_CPUINFO = "/proc/cpuinfo"
PROC_VERSION = "/proc/version"
PROC_OSRELEASE = "/proc/sys/kernel/osrelease"
ENVIRONMENT_METHOD = f"classify({PROC_OSRELEASE}, {PROC_VERSION})"
MAC_CPU_SYSCTL = "machdep.cpu.brand_string"  # char[] (sysctl(3))
MAC_MEMORY_SYSCTL = "hw.memsize"  # int64_t (sysctl(3))
OS_RELEASE_FILES = ("/etc/os-release", "/usr/lib/os-release")
WINDOWS_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"

CPU_COUNT_NOTE = "Logical CPUs in the system; this process may be restricted to fewer."
LINUX_TOTAL_NOTE = (
    "MemTotal: physical RAM usable by the kernel (installed RAM minus firmware and kernel "
    "reservations); excludes swap."
)
WSL2_MEMORY_NOTE = " Inside WSL2 this is the WSL2 virtual machine's memory, not the Windows host's."
LINUX_AVAILABLE_NOTE = (
    "MemAvailable: kernel estimate of memory available to start new applications without "
    "swapping; not MemFree and not this process's memory."
)
WINDOWS_TOTAL_NOTE = "ullTotalPhys: physical memory usable by Windows; excludes the page file."
WINDOWS_AVAILABLE_NOTE = (
    "ullAvailPhys: physical memory that can be reused without writing to disk (free plus "
    "standby); not this process's memory."
)
MACOS_TOTAL_NOTE = "hw.memsize: installed physical memory; excludes swap."
MACOS_AVAILABLE_REASON = (
    "Available memory isn't collected on macOS in V1: there is no sysctl equivalent of "
    "MemAvailable, and the Mach VM statistics needed to estimate it are out of scope."
)
NOT_LINUX_NOTE = "Not a Linux platform, so there is no Linux environment to classify."

# WSL classification evidence (data model §4.1). The kernel release names the running
# kernel, so it can't be a leftover of a removed WSL installation.
WSL2_RELEASE = re.compile(r"-microsoft-standard-wsl2(?![0-9a-z])", re.IGNORECASE)
WSL1_RELEASE = re.compile(r"^\d+\.\d+\.\d+-\d+-microsoft$", re.IGNORECASE)
WSL_MARKER = re.compile(r"microsoft|wsl", re.IGNORECASE)
WSL2_TOKEN = re.compile(r"wsl2", re.IGNORECASE)
PROC_VERSION_RELEASE = re.compile(r"^Linux version (\S+) ")

_ALLOWLISTED = re.compile(ENVIRONMENT_NAME_REGEX)
_CUDA_PATH_VERSIONED = re.compile(CUDA_PATH_VERSIONED_REGEX)
_PROFILES = {
    "Linux": PlatformProfile.LINUX,
    "Windows": PlatformProfile.WINDOWS,
    "Darwin": PlatformProfile.MACOS,
}


@runtime_checkable
class HostApi(Protocol):
    """Primitive read-only host reads. File and OS calls may raise `OSError`."""

    @property
    def provenance(self) -> DataOrigin: ...
    def system(self) -> str: ...
    def release(self) -> str: ...
    def version(self) -> str: ...
    def machine(self) -> str: ...
    def python_version(self) -> str: ...
    def python_implementation(self) -> str: ...
    def executable(self) -> str: ...
    def cpu_count(self) -> int | None: ...
    def read_text(self, path: str) -> str: ...
    def getenv(self, name: str) -> str | None: ...
    def environ_names(self) -> tuple[str, ...]: ...
    def windows_version(self) -> tuple[str, str]: ...
    def windows_processor_name(self) -> str: ...
    def windows_memory(self) -> tuple[int, int]: ...
    def mac_version(self) -> str: ...
    def mac_sysctl(self, name: str) -> bytes: ...


class StdlibHostApi:
    """The real host, through the standard library only."""

    @property
    def provenance(self) -> DataOrigin:
        return DataOrigin.LIVE

    def system(self) -> str:
        return platform.system()

    def release(self) -> str:
        return platform.release()

    def version(self) -> str:
        return platform.version()

    def machine(self) -> str:
        return platform.machine()

    def python_version(self) -> str:
        return platform.python_version()

    def python_implementation(self) -> str:
        return platform.python_implementation()

    def executable(self) -> str:
        return sys.executable or ""

    def cpu_count(self) -> int | None:
        return os.cpu_count()

    def read_text(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def getenv(self, name: str) -> str | None:
        return os.environ.get(name)

    def environ_names(self) -> tuple[str, ...]:
        """Names only; the platform adapter keeps just the allowlisted pattern."""
        return tuple(os.environ)

    def windows_version(self) -> tuple[str, str]:
        release = platform.win32_ver()[0]
        return release, platform.win32_edition() or ""

    def windows_processor_name(self) -> str:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, WINDOWS_CPU_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value)

    def windows_memory(self) -> tuple[int, int]:
        """`(ullTotalPhys, ullAvailPhys)` in bytes from `GlobalMemoryStatusEx`."""
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            raise ctypes.WinError()  # type: ignore[attr-defined]
        return int(status.ullTotalPhys), int(status.ullAvailPhys)

    def mac_version(self) -> str:
        return platform.mac_ver()[0]

    def mac_sysctl(self, name: str) -> bytes:
        """Raw value of `sysctlbyname(3)`, read-only (`newp` NULL, `newlen` 0).

        `int sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp,
        size_t newlen)` returns 0, or -1 with `errno` set (e.g. `ENOENT` for an unknown
        name). The first call sizes the buffer, the second fills it. macOS only:
        `ctypes.util.find_library` resolves libc through dyld there, without a subprocess.
        """
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("c")
        if path is None:
            raise FileNotFoundError("the C library wasn't found")
        sysctlbyname = ctypes.CDLL(path, use_errno=True).sysctlbyname
        sysctlbyname.argtypes = (
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        sysctlbyname.restype = ctypes.c_int
        key = name.encode("ascii")
        size = ctypes.c_size_t(0)
        if sysctlbyname(key, None, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), name)
        buffer = ctypes.create_string_buffer(size.value)
        if sysctlbyname(key, buffer, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), name)
        return buffer.raw[: size.value]


class _FieldNotFound(LookupError):
    """A well-formed source without the expected entry (data model §3.3 `FIELD_NOT_FOUND`)."""


class _Files:
    """Reads each file at most once per `system_info()` call; failures are replayed."""

    def __init__(self, host: HostApi) -> None:
        self._host = host
        self._results: dict[str, str | OSError] = {}

    def read(self, path: str) -> str:
        if path not in self._results:
            try:
                self._results[path] = self._host.read_text(path)
            except OSError as exc:
                self._results[path] = exc
        result = self._results[path]
        if isinstance(result, OSError):
            raise result
        return result


class OsPlatformAdapter:
    """`PlatformAdapter` over a `HostApi` (the real host unless a test injects one).

    Each field is read independently: one failing source never affects another. Expected
    conditions become `Observed` outcomes; anything else (a bug) propagates.
    """

    def __init__(self, host: HostApi | None = None) -> None:
        self._host: HostApi = host if host is not None else StdlibHostApi()
        self._profile: PlatformProfile | None = None
        self._environment: Observed[LinuxEnvironment] | None = None

    @property
    def provenance(self) -> DataOrigin:
        return self._host.provenance

    @property
    def profile(self) -> PlatformProfile:
        if self._profile is None:
            self._profile = self._detect_profile()
        return self._profile

    def system_info(self) -> SystemInfo:
        host, files, profile = self._host, _Files(self._host), self.profile
        total, available = self._memory(profile, files)
        return SystemInfo(
            platform_profile=profile,
            os_name=self._read(host.system, Source.PLATFORM, "platform.system()"),
            os_release=self._read(host.release, Source.PLATFORM, "platform.release()"),
            os_version=self._read(host.version, Source.PLATFORM, "platform.version()"),
            os_distribution=self._distribution(profile, files),
            architecture=self._read(host.machine, Source.PLATFORM, "platform.machine()"),
            processor_model=self._processor(profile, files),
            logical_cpu_count=self._read(
                host.cpu_count, Source.PLATFORM, "os.cpu_count()", note=CPU_COUNT_NOTE
            ),
            memory_total_bytes=total,
            memory_available_bytes=available,
            python_version=self._read(
                host.python_version, Source.PLATFORM, "platform.python_version()"
            ),
            python_implementation=self._read(
                host.python_implementation, Source.PLATFORM, "platform.python_implementation()"
            ),
            python_executable=self._read(host.executable, Source.PLATFORM, "sys.executable"),
            linux_environment=self._linux_environment(),
        )

    def getenv(self, name: str) -> Observed[str]:
        if _ALLOWLISTED.match(name) is None:
            raise AdapterUsageError(f"{name} is not an allowlisted environment variable")
        value = self._host.getenv(name)
        method = self._method("os.environ")
        if value is None:
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code=NOT_SET,
                message=f"{name} is not set",
                source=Source.ENVIRONMENT,
                method=method,
            )
        return Observed.collected(value, source=Source.ENVIRONMENT, method=method)

    def versioned_cuda_path_names(self) -> tuple[str, ...]:
        """Windows only (data model §5.5, DEC-47); no other name is kept."""
        if self.profile is not PlatformProfile.WINDOWS:
            return ()
        return tuple(sorted(n for n in self._host.environ_names() if _CUDA_PATH_VERSIONED.match(n)))

    # --- helpers ------------------------------------------------------------------------

    def _method(self, method: str) -> str:
        if self.provenance is DataOrigin.FIXTURE:
            return FIXTURE_METHOD_PREFIX + method
        return method

    def _read(
        self,
        read: Callable[[], Any],
        source: Source,
        method: str,
        unit: str | None = None,
        *,
        note: str | None = None,
    ) -> Observed:
        method = self._method(method)
        try:
            value = read()
        except OSError as exc:
            return exception_outcome(exc, source=source, method=method, unit=unit)
        except _FieldNotFound as exc:
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code=FIELD_NOT_FOUND,
                message=str(exc),
                source=source,
                method=method,
                unit=unit,
            )
        except ValueError as exc:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=PARSE_ERROR,
                message=str(exc),
                source=source,
                method=method,
                unit=unit,
            )
        if value is None or value == "":
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code=FIELD_NOT_FOUND,
                message=f"{method} returned no value",
                source=source,
                method=method,
                unit=unit,
            )
        return Observed.collected(value, source=source, method=method, unit=unit, note=note)

    def _unsupported(
        self,
        what: str,
        profile: PlatformProfile,
        source: Source,
        unit: str | None = None,
        *,
        reason: str | None = None,
    ) -> Observed:
        """Nothing is attempted, so there is no method."""
        return Observed.failed(
            CollectionStatus.NOT_SUPPORTED,
            code=PLATFORM_NOT_SUPPORTED,
            message=reason or f"{what} isn't collected on {profile} in this version",
            source=source,
            method=None,
            unit=unit,
        )

    def _linux_environment(self) -> Observed[LinuxEnvironment]:
        """Classified once per adapter: the running kernel doesn't change."""
        if self._environment is None:
            if self._host.system() != "Linux":
                self._environment = Observed.not_applicable(
                    note=NOT_LINUX_NOTE, source=Source.INSPECTOR
                )
            else:
                evidence = [self._evidence(path) for path in (PROC_OSRELEASE, PROC_VERSION)]
                environment, rationale = classify_linux_environment(*evidence)
                self._environment = Observed.collected(
                    environment,
                    source=Source.INSPECTOR,
                    method=self._method(ENVIRONMENT_METHOD),
                    note=rationale,
                )
        return self._environment

    def _evidence(self, path: str) -> str | OSError:
        try:
            return self._host.read_text(path)
        except OSError as exc:
            return exc

    def _detect_profile(self) -> PlatformProfile:
        profile = _PROFILES.get(self._host.system(), PlatformProfile.OTHER)
        if profile is PlatformProfile.LINUX:
            if self._linux_environment().value is LinuxEnvironment.WSL2:
                return PlatformProfile.WSL2
        return profile

    def _distribution(self, profile: PlatformProfile, files: _Files) -> Observed[str]:
        if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
            for path in OS_RELEASE_FILES:
                try:
                    files.read(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    pass  # replayed and reported by _read below
                return self._read(
                    lambda path=path: _pretty_name(files.read(path)),
                    Source.OS,
                    f"{path}:PRETTY_NAME",
                )
            return Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code="FileNotFoundError",
                message="neither /etc/os-release nor /usr/lib/os-release exists",
                source=Source.OS,
                method=self._method("os-release:PRETTY_NAME"),
            )
        if profile is PlatformProfile.WINDOWS:
            return self._read(
                self._windows_distribution,
                Source.PLATFORM,
                "platform.win32_ver(), platform.win32_edition()",
            )
        if profile is PlatformProfile.MACOS:
            return self._read(
                lambda: _prefixed("macOS", self._host.mac_version()),
                Source.PLATFORM,
                "platform.mac_ver()",
            )
        return self._unsupported("The OS distribution", profile, Source.PLATFORM)

    def _windows_distribution(self) -> str:
        release, edition = self._host.windows_version()
        return f"{_prefixed('Windows', release)} {edition}".strip()

    def _processor(self, profile: PlatformProfile, files: _Files) -> Observed[str]:
        if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
            return self._read(
                lambda: _cpuinfo_model(files.read(PROC_CPUINFO)),
                Source.OS,
                f"{PROC_CPUINFO}:model name",
            )
        if profile is PlatformProfile.WINDOWS:
            return self._read(
                lambda: self._host.windows_processor_name().strip(),
                Source.OS,
                f"winreg:HKLM\\{WINDOWS_CPU_KEY}:ProcessorNameString",
            )
        if profile is PlatformProfile.MACOS:
            return self._read(
                lambda: _sysctl_string(self._host.mac_sysctl(MAC_CPU_SYSCTL)),
                Source.OS,
                f"sysctlbyname({MAC_CPU_SYSCTL})",
            )
        return self._unsupported("The processor model", profile, Source.OS)

    def _memory(
        self, profile: PlatformProfile, files: _Files
    ) -> tuple[Observed[int], Observed[int]]:
        if profile in (PlatformProfile.LINUX, PlatformProfile.WSL2):
            vm_note = WSL2_MEMORY_NOTE if profile is PlatformProfile.WSL2 else ""
            total = self._read(
                lambda: _meminfo_bytes(files.read(PROC_MEMINFO), "MemTotal"),
                Source.OS,
                f"{PROC_MEMINFO}:MemTotal",
                "bytes",
                note=LINUX_TOTAL_NOTE + vm_note,
            )
            available = self._read(
                lambda: _meminfo_bytes(files.read(PROC_MEMINFO), "MemAvailable"),
                Source.OS,
                f"{PROC_MEMINFO}:MemAvailable",
                "bytes",
                note=LINUX_AVAILABLE_NOTE + vm_note,
            )
            return total, available
        if profile is PlatformProfile.WINDOWS:
            try:
                reading: tuple[int, int] | OSError = self._host.windows_memory()
            except OSError as exc:
                reading = exc

            def part(index: int) -> int:
                if isinstance(reading, OSError):
                    raise reading
                return reading[index]

            return (
                self._read(
                    lambda: part(0),
                    Source.OS,
                    "GlobalMemoryStatusEx.ullTotalPhys",
                    "bytes",
                    note=WINDOWS_TOTAL_NOTE,
                ),
                self._read(
                    lambda: part(1),
                    Source.OS,
                    "GlobalMemoryStatusEx.ullAvailPhys",
                    "bytes",
                    note=WINDOWS_AVAILABLE_NOTE,
                ),
            )
        if profile is PlatformProfile.MACOS:
            return (
                self._read(
                    lambda: _sysctl_int64(self._host.mac_sysctl(MAC_MEMORY_SYSCTL)),
                    Source.OS,
                    f"sysctlbyname({MAC_MEMORY_SYSCTL})",
                    "bytes",
                    note=MACOS_TOTAL_NOTE,
                ),
                self._unsupported(
                    "Available physical memory",
                    profile,
                    Source.OS,
                    "bytes",
                    reason=MACOS_AVAILABLE_REASON,
                ),
            )
        return (
            self._unsupported("Total physical memory", profile, Source.OS, "bytes"),
            self._unsupported("Available physical memory", profile, Source.OS, "bytes"),
        )


def classify_linux_environment(
    osrelease: str | OSError, proc_version: str | OSError
) -> tuple[LinuxEnvironment, str]:
    """Classify the Linux environment from kernel evidence; returns `(value, rationale)`.

    Policy (data model §4.1), applied in order:
    1. Either source unreadable, empty or malformed, or the two name different kernel
       releases → `UNKNOWN`.
    2. Release carries Microsoft's WSL2 kernel marker `-microsoft-standard-WSL2` → `WSL2`.
    3. Release has the WSL1 format `<x.y.z>-<Windows build>-Microsoft` and nothing mentions
       WSL2 → `WSL1` (WSL1 has no Linux kernel; this release is synthesised by Windows).
    4. Any other `microsoft`/`wsl` marker in either source → `UNKNOWN` (ambiguous).
    5. Otherwise → `NATIVE_LINUX`: not WSL. Generic virtualisation (KVM, cloud kernels) is
       still `NATIVE_LINUX`; the value doesn't claim bare metal.
    Matching ignores case.
    """
    for path, evidence in ((PROC_OSRELEASE, osrelease), (PROC_VERSION, proc_version)):
        if isinstance(evidence, OSError):
            return LinuxEnvironment.UNKNOWN, (
                f"{path} couldn't be read ({type(evidence).__name__}), so the Linux "
                "environment can't be established."
            )
    assert isinstance(osrelease, str) and isinstance(proc_version, str)
    release = osrelease.strip()
    if not release or any(char.isspace() for char in release):
        return LinuxEnvironment.UNKNOWN, f"{PROC_OSRELEASE} is empty or malformed."
    match = PROC_VERSION_RELEASE.match(proc_version)
    if match is None:
        return LinuxEnvironment.UNKNOWN, f"{PROC_VERSION} is empty or malformed."
    if match.group(1) != release:
        return LinuxEnvironment.UNKNOWN, (
            f"Conflicting kernel releases: {PROC_OSRELEASE} says {release!r}, "
            f"{PROC_VERSION} says {match.group(1)!r}."
        )
    if WSL2_RELEASE.search(release):
        return LinuxEnvironment.WSL2, (
            f"Kernel release {release!r} carries the WSL2 kernel marker "
            "'-microsoft-standard-WSL2' (both sources agree)."
        )
    if WSL1_RELEASE.match(release):
        if WSL2_TOKEN.search(proc_version):
            return LinuxEnvironment.UNKNOWN, (
                f"Kernel release {release!r} has the WSL1 format but {PROC_VERSION} mentions "
                "WSL2: conflicting evidence."
            )
        return LinuxEnvironment.WSL1, (
            f"Kernel release {release!r} has the WSL1 format "
            "'<version>-<Windows build>-Microsoft' (both sources agree)."
        )
    if WSL_MARKER.search(proc_version):
        return LinuxEnvironment.UNKNOWN, (
            f"A Microsoft/WSL marker is present (kernel release {release!r}) but neither the "
            "WSL1 nor the WSL2 kernel format matches, so the WSL version can't be determined."
        )
    return LinuxEnvironment.NATIVE_LINUX, (
        f"No Microsoft or WSL marker in kernel release {release!r} or {PROC_VERSION}: not WSL "
        "(this doesn't distinguish bare metal from a virtual machine)."
    )


def _sysctl_string(raw: bytes) -> str:
    """`char[]` sysctl values are NUL-terminated."""
    text = raw.split(b"\0", 1)[0].decode("utf-8").strip()
    if not text:
        raise _FieldNotFound("empty sysctl string")
    return text


def _sysctl_int64(raw: bytes) -> int:
    """`int64_t` in native byte order."""
    if len(raw) != 8:
        raise ValueError(f"expected 8 bytes for an int64_t, got {len(raw)}")
    (value,) = struct.unpack("=q", raw)
    if value <= 0:
        raise ValueError(f"implausible memory size {value}")
    return value


def _prefixed(prefix: str, value: str) -> str:
    if not value:
        raise _FieldNotFound(f"no {prefix} release reported")
    return f"{prefix} {value}"


def _entries(text: str, separator: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        key, found, value = line.partition(separator)
        if found:
            pairs.append((key.strip(), value.strip()))
    return pairs


def _meminfo_bytes(text: str, key: str) -> int:
    """`/proc/meminfo` values are in KiB despite the `kB` label."""
    for name, value in _entries(text, ":"):
        if name == key:
            amount, _, unit = value.partition(" ")
            if not amount.isdigit() or unit.strip() != "kB":
                raise ValueError(f"unexpected {key} value {value!r}")
            return int(amount) * 1024
    raise _FieldNotFound(f"no {key} entry")


def _cpuinfo_model(text: str) -> str:
    for name, value in _entries(text, ":"):
        if name == "model name":
            if not value:
                raise _FieldNotFound("empty 'model name' entry")
            return value
    raise _FieldNotFound("no 'model name' entry")


def _pretty_name(text: str) -> str:
    for name, value in _entries(text, "="):
        if name == "PRETTY_NAME":
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                quote, value = value[0], value[1:-1]
                if quote == '"':
                    value = re.sub(r"\\(.)", r"\1", value)
            if not value:
                raise _FieldNotFound("empty PRETTY_NAME")
            return value
    raise _FieldNotFound("no PRETTY_NAME entry")
