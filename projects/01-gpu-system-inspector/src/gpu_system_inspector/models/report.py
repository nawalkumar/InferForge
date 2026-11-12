"""Inspection snapshot and report models (docs/data_model_specification.md §7.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from gpu_system_inspector import (
    APPLICATION_NAME,
    DISPLAY_NAME,
    DISTRIBUTION_NAME,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    __version__,
)
from gpu_system_inspector.models._validation import (
    constrained,
    device_ref_key,
    optional_ref_key,
    set_field,
    validate_model,
)
from gpu_system_inspector.models.collection import (
    CollectionError,
    CollectorRun,
    UnsupportedCapability,
)
from gpu_system_inspector.models.cuda import CUDAEnvironmentInfo
from gpu_system_inspector.models.diagnostics import DiagnosticFinding
from gpu_system_inspector.models.driver import DriverInfo
from gpu_system_inspector.models.enums import (
    DataOrigin,
    InspectionMode,
    ProbeMode,
    RedactionCategory,
)
from gpu_system_inspector.models.framework import FrameworkInfo
from gpu_system_inspector.models.gpu import GPUInventory
from gpu_system_inspector.models.observed import InvalidModelError
from gpu_system_inspector.models.readiness import (
    MAX_REQUIRED_VRAM_GIB,
    ReadinessResult,
    required_bytes_for,
)
from gpu_system_inspector.models.runtime import RuntimeMetrics
from gpu_system_inspector.models.system import SystemInfo

SNAPSHOT_LIMITATION = "This report is a point-in-time snapshot."
MAX_SUBPROCESS_TIMEOUT_S = 300
MAX_PYTORCH_TIMEOUT_S = 600


@dataclass(frozen=True, slots=True, kw_only=True)
class ApplicationInfo:
    name: str = constrained(minLength=1, default=APPLICATION_NAME)
    display_name: str = constrained(minLength=1, default=DISPLAY_NAME)
    distribution: str = constrained(minLength=1, default=DISTRIBUTION_NAME)
    version: str = constrained(minLength=1, default=__version__)

    def __post_init__(self) -> None:
        validate_model(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class InspectionOptions:
    """Effective inspection options; echoed into the report as `options`."""

    required_vram_gib: float | None = constrained(
        exclusiveMinimum=0, maximum=MAX_REQUIRED_VRAM_GIB, default=None
    )
    include_pytorch: bool = True
    include_processes: bool = True
    redact: bool = False
    subprocess_timeout_s: float = constrained(
        exclusiveMinimum=0, maximum=MAX_SUBPROCESS_TIMEOUT_S, default=10.0
    )
    pytorch_timeout_s: float = constrained(
        exclusiveMinimum=0, maximum=MAX_PYTORCH_TIMEOUT_S, default=60.0
    )
    pytorch_probe_mode: ProbeMode = ProbeMode.SUBPROCESS

    def __post_init__(self) -> None:
        validate_model(self)
        if self.required_vram_gib is not None and required_bytes_for(self.required_vram_gib) < 1:
            raise InvalidModelError(
                "required_vram_gib must amount to at least 1 byte (round(GiB * 2**30) >= 1)"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RedactionInfo:
    applied: bool = False
    categories: tuple[RedactionCategory, ...] = ()

    def __post_init__(self) -> None:
        validate_model(self)
        set_field(self, "categories", tuple(sorted(set(self.categories))))
        if not self.applied and self.categories:
            raise InvalidModelError("redaction categories require applied = true")


@dataclass(frozen=True, slots=True, kw_only=True)
class InspectionSnapshot:
    """Normalised collector output, before readiness, diagnostics and report assembly."""

    system: SystemInfo
    gpu: GPUInventory
    driver: DriverInfo
    cuda_environment: CUDAEnvironmentInfo
    frameworks: FrameworkInfo
    runtime: tuple[RuntimeMetrics, ...]
    collection_errors: tuple[CollectionError, ...]
    unsupported_capabilities: tuple[UnsupportedCapability, ...]
    collector_runs: tuple[CollectorRun, ...]

    def __post_init__(self) -> None:
        validate_model(self)
        _normalise_snapshot_parts(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class InspectionReport:
    _SCHEMA_EXTRA: ClassVar[dict[str, Any]] = {
        "if": {"properties": {"mode": {"const": InspectionMode.READINESS.value}}},
        "then": {
            "properties": {
                "readiness": {"type": "object"},
                "options": {"properties": {"required_vram_gib": {"type": "number"}}},
            }
        },
        "else": {
            "properties": {
                "readiness": {"type": "null"},
                "options": {"properties": {"required_vram_gib": {"type": "null"}}},
            }
        },
    }

    schema_name: str = constrained(const=REPORT_SCHEMA_NAME, default=REPORT_SCHEMA_NAME)
    schema_version: str = constrained(const=REPORT_SCHEMA_VERSION, default=REPORT_SCHEMA_VERSION)
    application: ApplicationInfo = field(default_factory=ApplicationInfo)
    generated_at: datetime
    duration_ms: int = constrained(minimum=0)
    data_origin: DataOrigin
    mode: InspectionMode
    options: InspectionOptions
    system: SystemInfo
    gpu: GPUInventory
    driver: DriverInfo
    cuda_environment: CUDAEnvironmentInfo
    frameworks: FrameworkInfo
    runtime: tuple[RuntimeMetrics, ...]
    diagnostics: tuple[DiagnosticFinding, ...]
    readiness: ReadinessResult | None
    collection_errors: tuple[CollectionError, ...]
    unsupported_capabilities: tuple[UnsupportedCapability, ...]
    limitations: tuple[str, ...]
    collector_runs: tuple[CollectorRun, ...]
    redaction: RedactionInfo = field(default_factory=RedactionInfo)

    def __post_init__(self) -> None:
        validate_model(self)
        _normalise_snapshot_parts(self)
        known = set(self.gpu.device_refs)

        diagnostics = tuple(
            sorted(self.diagnostics, key=lambda d: (d.check_id, optional_ref_key(d.device_ref)))
        )
        set_field(self, "diagnostics", diagnostics)
        keys = [(d.check_id, d.device_ref) for d in diagnostics]
        if len(set(keys)) != len(keys):
            raise InvalidModelError("each (check_id, device_ref) pair may appear only once")
        for finding in diagnostics:
            if finding.device_ref is not None and finding.device_ref not in known:
                raise InvalidModelError(f"finding refers to unknown device {finding.device_ref}")

        readiness_requested = self.options.required_vram_gib is not None
        if (self.mode is InspectionMode.READINESS) != readiness_requested:
            raise InvalidModelError("mode is READINESS exactly when required_vram_gib is set")
        if readiness_requested != (self.readiness is not None):
            raise InvalidModelError("readiness is present exactly in READINESS mode")
        if self.readiness is not None:
            if self.readiness.required_vram_gib != self.options.required_vram_gib:
                raise InvalidModelError("readiness.required_vram_gib must match the options")
            readiness_refs = {d.device_ref for d in self.readiness.devices}
            if readiness_refs != known:
                raise InvalidModelError("readiness must assess exactly the inventoried devices")

        if SNAPSHOT_LIMITATION not in self.limitations:
            raise InvalidModelError("limitations must include the snapshot statement")
        if any(not text for text in self.limitations):
            raise InvalidModelError("limitations must not contain empty strings")

    @property
    def snapshot(self) -> InspectionSnapshot:
        return InspectionSnapshot(
            system=self.system,
            gpu=self.gpu,
            driver=self.driver,
            cuda_environment=self.cuda_environment,
            frameworks=self.frameworks,
            runtime=self.runtime,
            collection_errors=self.collection_errors,
            unsupported_capabilities=self.unsupported_capabilities,
            collector_runs=self.collector_runs,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: InspectionSnapshot,
        *,
        generated_at: datetime,
        duration_ms: int,
        data_origin: DataOrigin,
        options: InspectionOptions,
        diagnostics: tuple[DiagnosticFinding, ...] = (),
        readiness: ReadinessResult | None = None,
        limitations: tuple[str, ...] = (SNAPSHOT_LIMITATION,),
        redaction: RedactionInfo | None = None,
        application: ApplicationInfo | None = None,
    ) -> InspectionReport:
        mode = (
            InspectionMode.READINESS
            if options.required_vram_gib is not None
            else InspectionMode.GENERAL
        )
        return cls(
            application=application or ApplicationInfo(),
            generated_at=generated_at,
            duration_ms=duration_ms,
            data_origin=data_origin,
            mode=mode,
            options=options,
            system=snapshot.system,
            gpu=snapshot.gpu,
            driver=snapshot.driver,
            cuda_environment=snapshot.cuda_environment,
            frameworks=snapshot.frameworks,
            runtime=snapshot.runtime,
            diagnostics=diagnostics,
            readiness=readiness,
            collection_errors=snapshot.collection_errors,
            unsupported_capabilities=snapshot.unsupported_capabilities,
            limitations=limitations,
            collector_runs=snapshot.collector_runs,
            redaction=redaction or RedactionInfo(),
        )


def _normalise_snapshot_parts(obj: InspectionSnapshot | InspectionReport) -> None:
    """Shared ordering and cross-reference rules for snapshots and reports."""
    known = set(obj.gpu.device_refs)
    runtime = tuple(sorted(obj.runtime, key=lambda r: device_ref_key(r.device_ref)))
    refs = [r.device_ref for r in runtime]
    if len(set(refs)) != len(refs):
        raise InvalidModelError("runtime must have at most one entry per device")
    unknown = set(refs) - known
    if unknown:
        raise InvalidModelError(f"runtime refers to unknown devices {sorted(unknown)}")
    set_field(obj, "runtime", runtime)
    set_field(
        obj,
        "collection_errors",
        tuple(
            sorted(
                obj.collection_errors,
                key=lambda e: (e.collector, optional_ref_key(e.device_ref), e.error_code),
            )
        ),
    )
    set_field(
        obj,
        "unsupported_capabilities",
        tuple(
            sorted(
                obj.unsupported_capabilities,
                key=lambda u: (u.capability, optional_ref_key(u.device_ref)),
            )
        ),
    )
