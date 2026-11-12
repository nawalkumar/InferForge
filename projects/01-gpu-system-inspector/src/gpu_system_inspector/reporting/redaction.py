"""Report redaction (data model §8, architecture §4.10, IMP-P04, DEC-50).

`redact(report, policy)` returns a new, re-validated report in which every string — field
values *and* free text (explanations, evidence, error messages, CUDA init warnings) — has
the sensitive values replaced; structure, statuses and non-sensitive values are unchanged.
JSON and text are rendered from the same redacted report, so both outputs agree.

- PATH → `<redacted:path:N>`: the Python executable, `nvidia-smi`, toolkit roots and `nvcc`,
  the active `nvcc`, and `CUDA_HOME`/`CUDA_PATH*`/`LD_LIBRARY_PATH` values.
- PROCESS_NAME → `<redacted:process>`: GPU process names.
- DEVICE_ID → `<redacted:uuid:N>`, `<redacted:pci:N>`: GPU UUIDs and PCI bus IDs (NVML,
  PCI scan, PyTorch).
- USER → `<redacted:user>`, `~`: the local user name and home directory, in any string.

Ordinals are stable within one report: the same device (UUID or PCI address, however it is
spelled by NVML, sysfs or PyTorch) gets the same number, so correlation stays visible, and
distinct paths stay distinct (the models require unique toolkit roots and PCI addresses).
The user name and home directory are inputs only (the caller supplies them, see
`RedactionPolicy`); they are never stored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gpu_system_inspector.models import (
    InspectionReport,
    RedactionCategory,
    RedactionInfo,
)
from gpu_system_inspector.reporting.json_report import from_json_dict, to_json_dict

PROCESS_PLACEHOLDER = "<redacted:process>"
USER_PLACEHOLDER = "<redacted:user>"
PATH_ENVIRONMENT = ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH")
# Keys whose values are enums, identifiers or timestamps: never free text.
_STRUCTURAL = frozenset(
    {
        "schema_name",
        "schema_version",
        "status",
        "source",
        "unit",
        "kind",
        "check_id",
        "device_ref",
        "correlated_device_ref",
        "collector",
        "capability",
        "error_code",
        "code",
        "report_path",
        "severity",
        "confidence",
        "mode",
        "data_origin",
        "overall_status",
        "overall_reason",
        "criterion",
        "disclaimer",
        "platform_profile",
        "discovered_via",
        "categories",
        "generated_at",
        "sampled_at",
        "vendor",
        "vendor_id",
        "device_id",
        "class_code",
        "inventory_source",
        "detection_sources",
        "reference_url",
        "pytorch_probe_mode",
    }
)


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    categories: tuple[RedactionCategory, ...] = tuple(RedactionCategory)
    user: str | None = None
    home: str | None = None


DEFAULT_REDACTION = RedactionPolicy()


def _uuid_key(value: str) -> str:
    v = value.strip().lower()
    return v[4:] if v.startswith("gpu-") else v


def _pci_key(value: str) -> str:
    parts = value.strip().lower().split(":")
    if len(parts) == 3 and all(parts):
        try:
            return f"{int(parts[0], 16):x}:{parts[1]}:{parts[2]}"
        except ValueError:
            return value.lower()
    return f"0:{':'.join(parts)}" if len(parts) == 2 else value.lower()


@dataclass
class Redactor:
    """The replacement table built from one report; also redacts stray text (CLI errors)."""

    policy: RedactionPolicy = DEFAULT_REDACTION
    literals: dict[str, str] = field(default_factory=dict)
    whole: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_report(cls, report: InspectionReport, policy: RedactionPolicy) -> Redactor:
        r = cls(policy)
        cats = set(policy.categories)
        if RedactionCategory.DEVICE_ID in cats:
            r._devices(report)
        if RedactionCategory.PATH in cats:
            r._paths(report)
        if RedactionCategory.PROCESS_NAME in cats:
            for m in report.runtime:
                for p in m.processes.value or ():
                    if p.name.value:
                        r.whole[p.name.value] = PROCESS_PLACEHOLDER
                        r.literals[repr(p.name.value)] = repr(PROCESS_PLACEHOLDER)
        return r

    def _ordinal(self, values: Iterable[str], key: Any, kind: str) -> None:
        numbers: dict[str, int] = {}
        for value in values:
            if not value:
                continue
            k = key(value)
            numbers.setdefault(k, len(numbers) + 1)
            placeholder = f"<redacted:{kind}:{numbers[k]}>"
            self.whole[value] = placeholder
            self.literals[value] = placeholder

    def _devices(self, report: InspectionReport) -> None:
        gpu, torch = report.gpu, report.frameworks.pytorch
        self._ordinal(
            [d.uuid.value for d in gpu.devices if d.uuid.value]
            + [d.uuid.value for d in torch.devices if d.uuid.value],
            _uuid_key,
            "uuid",
        )
        self._ordinal(
            [d.pci_bus_id.value for d in gpu.devices if d.pci_bus_id.value]
            + [o.address for o in gpu.pci_observations.value or ()]
            + [d.pci_bus_id.value for d in torch.devices if d.pci_bus_id.value],
            _pci_key,
            "pci",
        )

    def _paths(self, report: InspectionReport) -> None:
        env = report.cuda_environment
        paths = [report.system.python_executable.value, report.driver.nvidia_smi_path.value]
        for t in env.toolkits.value or ():
            paths += [t.root_path, t.nvcc_path.value]
        paths.append(env.active_nvcc_path.value)
        for name, value in sorted(env.environment.items()):
            if name in PATH_ENVIRONMENT or name.startswith("CUDA_PATH_V"):
                if value.value:
                    paths.append(value.value)
                    paths += [p for p in value.value.split(":") if len(p) > 1]
        self._ordinal([p for p in paths if p], lambda p: p, "path")

    def text(self, value: str) -> str:
        if value in self.whole:
            return self.whole[value]
        for literal in sorted(self.literals, key=len, reverse=True):
            if literal in value:
                value = value.replace(literal, self.literals[literal])
        if RedactionCategory.USER in self.policy.categories:
            home, user = self.policy.home, self.policy.user
            if home and len(home) > 1:
                value = re.sub(re.escape(home.rstrip("/")) + r"(?=/|\b|$)", "~", value)
            if user:
                value = re.sub(rf"(?<![\w-]){re.escape(user)}(?![\w-])", USER_PLACEHOLDER, value)
        return value

    def tree(self, node: Any, key: str | None = None) -> Any:
        if isinstance(node, dict):
            return {k: self.tree(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [self.tree(v, key) for v in node]
        if isinstance(node, str) and key not in _STRUCTURAL:
            return self.text(node)
        return node


def redact(
    report: InspectionReport, policy: RedactionPolicy = DEFAULT_REDACTION
) -> InspectionReport:
    redactor = Redactor.for_report(report, policy)
    data = to_json_dict(report)
    application = data.pop("application")
    redacted = redactor.tree(data)
    redacted["application"] = application
    redacted["options"]["redact"] = True
    redacted["redaction"] = to_json_dict(
        RedactionInfo(applied=True, categories=tuple(policy.categories))
    )
    return from_json_dict(redacted, InspectionReport)
