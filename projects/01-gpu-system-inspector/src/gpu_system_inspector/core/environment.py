"""Fail-closed Linux-environment resolution (architecture §4.11, DEC-41, DEC-44).

PCI eligibility depends on `linux_environment`, which the system collector classifies. When
that section is the guard's fallback (the system collector raised), the classification
isn't available, and the profile alone must never make a host look like native Linux.
"""

from __future__ import annotations

from gpu_system_inspector.models import LinuxEnvironment, PlatformProfile, SystemInfo


def resolve_linux_environment(system: SystemInfo) -> LinuxEnvironment | None:
    """The collected classification, or the fail-closed value for the profile:
    `LINUX` → `UNKNOWN`, `WSL2` → `WSL2` (only a WSL2 classification gives that profile),
    anything else → `None` (non-Linux)."""
    if system.linux_environment.is_collected:
        return system.linux_environment.value
    if system.platform_profile is PlatformProfile.LINUX:
        return LinuxEnvironment.UNKNOWN
    if system.platform_profile is PlatformProfile.WSL2:
        return LinuxEnvironment.WSL2
    return None
