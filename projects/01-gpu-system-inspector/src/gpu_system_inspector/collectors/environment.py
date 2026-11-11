"""Allowlisted environment variables (data model §5.5) for `CUDAEnvironmentInfo.environment`.

Every name is checked against the allowlist before anything is read, so a non-allowlisted
variable is never accessed. Values are reported as-is (redaction happens at report time);
an unset variable is `NOT_DETECTED`. Toolkit discovery, the rest of the CUDA environment
section, is a later stage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from gpu_system_inspector.adapters import AdapterUsageError, PlatformAdapter
from gpu_system_inspector.models import ENVIRONMENT_ALLOWLIST, Observed
from gpu_system_inspector.models.cuda import ENVIRONMENT_NAME_REGEX

_ALLOWLISTED = re.compile(ENVIRONMENT_NAME_REGEX)


def collect_environment(
    platform: PlatformAdapter, names: Iterable[str] = ENVIRONMENT_ALLOWLIST
) -> dict[str, Observed[str]]:
    wanted = sorted(set(names))
    rejected = [name for name in wanted if _ALLOWLISTED.match(name) is None]
    if rejected:
        raise AdapterUsageError(f"not allowlisted environment variables: {rejected}")
    return {name: platform.getenv(name) for name in wanted}
