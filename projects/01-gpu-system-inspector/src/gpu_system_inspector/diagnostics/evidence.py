"""Evidence and finding helpers shared by the diagnostic rules (diagnostics spec §1, §2).

An *observation* quotes a collected value with its source and JSON pointer; an *inference*
states a conclusion and never carries a value. `finding()` fills the boilerplate and lets
the `DiagnosticFinding` model enforce the status/severity/evidence/limitation rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from gpu_system_inspector.models import (
    CollectionStatus,
    Confidence,
    DiagnosticFinding,
    DiagnosticStatus,
    Evidence,
    EvidenceKind,
    Observed,
    Recommendation,
    Severity,
    Source,
)

# Official documentation only (diagnostics spec §5, AC28).
ALLOWED_REFERENCE_DOMAINS = (
    "docs.nvidia.com",
    "developer.nvidia.com",
    "docs.pytorch.org",
    "pytorch.org",
    "docs.python.org",
    "pypi.org",
)
PYTORCH_SELECTOR_URL = "https://pytorch.org/get-started/locally/"
WSL_GUIDE_URL = "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
LINUX_INSTALL_URL = "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html"
CUDA_COMPAT_URL = "https://docs.nvidia.com/deploy/cuda-compatibility/index.html"
CUDA_ENV_VARS_URL = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html"
)
CONTAINER_TOOLKIT_URL = (
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html"
)
DCGM_URL = "https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html"
NVML_BINDING_URL = "https://pypi.org/project/nvidia-ml-py/"
NVIDIA_SMI_URL = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"

S = DiagnosticStatus
_FAILURE_STATUSES = frozenset(
    {
        CollectionStatus.COLLECTION_FAILED,
        CollectionStatus.PERMISSION_DENIED,
        CollectionStatus.UNKNOWN,
    }
)


def json_value(value: Any) -> Any:
    """A copy of a simple collected value (`None` for anything that isn't scalar JSON)."""
    return value if isinstance(value, str | int | float | bool) or value is None else None


def describe(observed: Observed[Any]) -> str:
    """`12.4` for a collected value, `NOT_DETECTED (CODE: message)` otherwise."""
    if observed.is_collected:
        return repr(observed.value) if isinstance(observed.value, str) else str(observed.value)
    if observed.error is not None:
        return f"{observed.status} ({observed.error.code}: {observed.error.message})"
    return f"{observed.status}" + (f" ({observed.note})" if observed.note else "")


def observation(statement: str, observed: Observed[Any] | None, path: str) -> Evidence:
    value = observed.value if observed is not None and observed.is_collected else None
    return Evidence(
        kind=EvidenceKind.OBSERVATION,
        statement=statement,
        source=observed.source if observed is not None else None,
        report_path=path,
        value=json_value(value),
    )


def observed_field(label: str, observed: Observed[Any], path: str) -> Evidence:
    return observation(f"{label} is {describe(observed)}", observed, path)


def inference(statement: str) -> Evidence:
    return Evidence(
        kind=EvidenceKind.INFERENCE, statement=statement, source=None, report_path=None, value=None
    )


def advice(action: str, rationale: str, url: str | None = None) -> Recommendation:
    return Recommendation(action=action, rationale=rationale, reference_url=url)


def is_failure(observed: Observed[Any]) -> bool:
    return observed.status in _FAILURE_STATUSES


def finding(
    check_id: str,
    status: DiagnosticStatus,
    severity: Severity,
    confidence: Confidence,
    title: str,
    explanation: str,
    evidence: Iterable[Evidence] = (),
    *,
    device_ref: str | None = None,
    recommendations: Iterable[Recommendation] = (),
    limitations: Iterable[str] = (),
) -> DiagnosticFinding:
    items = tuple(evidence)
    sources = tuple(e.source for e in items if e.source is not None)
    if not sources and status is not S.NOT_APPLICABLE:
        sources = (Source.INSPECTOR,)
    return DiagnosticFinding(
        check_id=check_id,
        device_ref=device_ref,
        status=status,
        severity=severity,
        title=title,
        explanation=explanation,
        evidence=items,
        detection_sources=sources,
        recommendations=tuple(recommendations),
        confidence=confidence,
        limitations=tuple(limitations),
    )


def not_applicable(
    check_id: str, title: str, explanation: str, evidence: Iterable[Evidence] = (), **kw: Any
) -> DiagnosticFinding:
    return finding(
        check_id,
        S.NOT_APPLICABLE,
        Severity.NONE,
        Confidence.NOT_APPLICABLE,
        title,
        explanation,
        evidence,
        **kw,
    )


def major(version: str | None) -> int | None:
    """The CUDA major version of `12.4` / `12.4.131`; `None` if it doesn't parse."""
    if not version:
        return None
    head = version.split(".")[0]
    return int(head) if head.isdigit() else None
