"""Fake platform, PCI scanner and clock implementing `PlatformAdapter`, `PciScanner` and
`Clock`. No OS, filesystem or real clock access."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

from gpu_system_inspector.adapters import AdapterUsageError, PciScan
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    CollectionStatus,
    DataOrigin,
    InspectionSnapshot,
    Observed,
    PciObservation,
    PlatformProfile,
    Source,
    SystemInfo,
)
from gpu_system_inspector.models.cuda import CUDA_PATH_VERSIONED_REGEX, ENVIRONMENT_NAME_REGEX
from gsi_testkit.builders import FIXED_TIME, fx

_ALLOWLISTED = re.compile(ENVIRONMENT_NAME_REGEX)
_CUDA_PATH_VERSIONED = re.compile(CUDA_PATH_VERSIONED_REGEX)


@dataclass
class FakePlatform:
    system: SystemInfo
    environment: Mapping[str, Observed]
    calls: list[str] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakePlatform:
        """A crashed CUDA section records no variables (FX10); the fake then has them all
        unset, so the run reaches the injected failure."""
        environment = snapshot.cuda_environment.environment or {
            name: Observed.failed(
                CollectionStatus.NOT_DETECTED,
                code="NOT_SET",
                message=f"{name} is not set",
                source=Source.ENVIRONMENT,
                method=fx("os.environ"),
            )
            for name in ENVIRONMENT_ALLOWLIST
        }
        return cls(system=snapshot.system, environment=environment)

    @property
    def profile(self) -> PlatformProfile:
        return self.system.platform_profile

    def system_info(self) -> SystemInfo:
        self.calls.append("system_info")
        return self.system

    def getenv(self, name: str) -> Observed[str]:
        """Allowlisted variables only (data model §5.5); an unset variable is NOT_DETECTED."""
        if _ALLOWLISTED.match(name) is None:
            raise AdapterUsageError(f"{name} is not an allowlisted environment variable")
        self.calls.append(f"getenv({name})")
        if name not in self.environment:
            raise AssertionError(f"the fixture doesn't declare the allowlisted variable {name}")
        return self.environment[name]

    def versioned_cuda_path_names(self) -> tuple[str, ...]:
        """The fixture's set `CUDA_PATH_V*` names, on Windows only."""
        self.calls.append("versioned_cuda_path_names")
        if self.profile is not PlatformProfile.WINDOWS:
            return ()
        return tuple(
            sorted(
                name
                for name, value in self.environment.items()
                if _CUDA_PATH_VERSIONED.match(name) and value.is_collected
            )
        )


@dataclass
class FakePciScanner:
    result: Observed[tuple[PciObservation, ...]]
    calls: list[str] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakePciScanner:
        return cls(result=snapshot.gpu.pci_observations)

    def scan(self) -> PciScan:
        """The fixture's scan; fixtures pin no per-device failures."""
        self.calls.append("/sys/bus/pci/devices")
        return PciScan(observations=self.result)


@dataclass
class FixedClock:
    """Fixed wall clock; monotonic milliseconds advance by a fixed step per call."""

    now: datetime = FIXED_TIME
    step_ms: int = 5
    _ticks: Iterator[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ticks = count(0, self.step_ms)

    def now_utc(self) -> datetime:
        return self.now

    def monotonic_ms(self) -> int:
        return next(self._ticks)
