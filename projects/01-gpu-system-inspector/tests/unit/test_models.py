import dataclasses
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gpu_system_inspector.models import (
    ALLOCATOR_NOT_APPLICABLE_NOTE,
    BUILD_CUDA_NOTE,
    SNAPSHOT_LIMITATION,
    CollectionError,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    Confidence,
    CUDAEnvironmentInfo,
    CudaToolkitCandidate,
    DeviceReadiness,
    DiagnosticFinding,
    DiagnosticStatus,
    DriverInfo,
    ErrorDetail,
    Evidence,
    EvidenceKind,
    GPUDeviceInfo,
    GPUInventory,
    InspectionMode,
    InspectionOptions,
    InspectionReport,
    InvalidModelError,
    Observed,
    PciObservation,
    PyTorchInfo,
    ReadinessReason,
    ReadinessStatus,
    RedactionCategory,
    RedactionInfo,
    RuntimeMetrics,
    Severity,
    Source,
    SystemInfo,
    ToolkitDiscovery,
    UnsupportedCapability,
    required_bytes_for,
)

GIB = 2**30
ERROR = ErrorDetail(code="UNHANDLED_EXCEPTION", message="boom")


def ok(value, source=Source.NVML, unit=None, **kw):
    return Observed.collected(value, source=source, method="fixture", unit=unit, **kw)


# --- construction and type checking ------------------------------------------------------


def test_valid_full_report_builds(build):
    report = build.report()
    assert report.mode is InspectionMode.GENERAL
    assert report.readiness is None
    assert report.gpu.device_refs == ("nvml:0", "nvml:1")
    assert report.limitations == (SNAPSHOT_LIMITATION,)


def test_wrong_field_types_are_rejected(build):
    system = build.system()
    with pytest.raises(InvalidModelError, match="SystemInfo.os_name: expected Observed"):
        dataclasses.replace(system, os_name="Linux")
    with pytest.raises(InvalidModelError, match="expected int"):
        dataclasses.replace(system, logical_cpu_count=ok("16", Source.PLATFORM))
    with pytest.raises(InvalidModelError, match="expected int"):
        dataclasses.replace(system, logical_cpu_count=ok(True, Source.PLATFORM))
    with pytest.raises(InvalidModelError, match="expected LinuxEnvironment"):
        dataclasses.replace(system, linux_environment=ok("WSL2", Source.INSPECTOR))


def test_observed_fields_must_carry_their_declared_unit(build):
    with pytest.raises(InvalidModelError, match="unit must be 'bytes'"):
        dataclasses.replace(build.system(), memory_total_bytes=ok(1, Source.OS, "ms"))
    with pytest.raises(InvalidModelError, match="unit must be None"):
        dataclasses.replace(build.system(), os_name=ok("Linux", Source.PLATFORM, "bytes"))


def test_unit_is_kept_even_for_failed_values(build):
    failed = Observed.failed(CollectionStatus.NOT_SUPPORTED, code="X", message="y", unit="percent")
    metrics = dataclasses.replace(build.runtime("nvml:0"), gpu_utilization_percent=failed)
    assert metrics.gpu_utilization_percent.unit == "percent"


def test_models_are_immutable(build):
    report = build.report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.duration_ms = 1  # type: ignore[misc]
    assert isinstance(report.runtime, tuple)
    with pytest.raises(TypeError):
        report.cuda_environment.environment["CUDA_HOME"] = ok("/x", Source.ENVIRONMENT)


def test_integer_given_for_float_field_is_normalised(build):
    metrics = dataclasses.replace(build.runtime("nvml:0"), power_limit_watts=ok(450, unit="watts"))
    assert metrics.power_limit_watts.value == 450.0
    assert type(metrics.power_limit_watts.value) is float


# --- unavailable() builders -------------------------------------------------------------


@pytest.mark.parametrize(
    "build_unavailable",
    [
        lambda: SystemInfo.unavailable(CollectionStatus.COLLECTION_FAILED, ERROR),
        lambda: DriverInfo.unavailable(CollectionStatus.COLLECTION_FAILED, ERROR),
        lambda: CUDAEnvironmentInfo.unavailable(CollectionStatus.COLLECTION_FAILED, ERROR),
        lambda: GPUInventory.unavailable(CollectionStatus.COLLECTION_FAILED, ERROR),
        lambda: GPUDeviceInfo.unavailable("nvml:3", CollectionStatus.COLLECTION_FAILED, ERROR),
    ],
)
def test_unavailable_sections_mark_every_value_failed(build_unavailable):
    section = build_unavailable()
    observed = [
        getattr(section, f.name)
        for f in dataclasses.fields(section)
        if isinstance(getattr(section, f.name), Observed)
    ]
    assert observed
    assert all(o.status is CollectionStatus.COLLECTION_FAILED and o.value is None for o in observed)
    assert all(o.error == ERROR for o in observed)


def test_unavailable_runtime_keeps_units(build):
    metrics = RuntimeMetrics.unavailable(
        "nvml:1",
        datetime(2026, 1, 1, tzinfo=UTC),
        CollectionStatus.COLLECTION_FAILED,
        ERROR,
    )
    assert metrics.memory_free_bytes.unit == "bytes"
    assert metrics.power_draw_watts.unit == "watts"
    assert metrics.memory_free_bytes.value is None


def test_pytorch_not_installed_section(build):
    report = build.no_gpu_report()
    pytorch = report.frameworks.pytorch
    assert pytorch.status is CollectionStatus.NOT_INSTALLED
    assert pytorch.version.status is CollectionStatus.NOT_INSTALLED
    assert pytorch.devices == ()


def test_skipped_section_has_no_errors(build):
    probe = build.pytorch().probe
    skipped = PyTorchInfo.unavailable(CollectionStatus.SKIPPED, None, probe=probe)
    assert skipped.cuda_available.status is CollectionStatus.SKIPPED
    assert skipped.cuda_available.error is None


# --- GPU inventory and multi-GPU ---------------------------------------------------------


def test_devices_are_sorted_naturally_and_must_be_unique(build):
    devices = (build.device(10), build.device(2), build.device(0))
    inventory = build.inventory(devices)
    assert inventory.device_refs == ("nvml:0", "nvml:2", "nvml:10")
    with pytest.raises(InvalidModelError, match="unique device_ref"):
        build.inventory((build.device(1), build.device(1)))


def test_device_count_must_match_listed_devices(build):
    inventory = build.inventory((build.device(0),))
    with pytest.raises(InvalidModelError, match="doesn't match"):
        dataclasses.replace(inventory, device_count=ok(2))
    with pytest.raises(InvalidModelError, match="only be listed"):
        dataclasses.replace(
            inventory,
            device_count=Observed.failed(CollectionStatus.COLLECTION_FAILED, code="X", message="y"),
        )


def test_zero_devices_reported_by_nvml_is_a_collected_zero():
    inventory = GPUInventory(
        device_count=ok(0),
        inventory_source=Source.NVML,
        devices=(),
        pci_observations=ok((), Source.OS),
    )
    assert inventory.device_count.value == 0


def test_device_ref_prefix_must_match_inventory_source(build):
    with pytest.raises(InvalidModelError, match="'smi' prefix"):
        dataclasses.replace(build.inventory((build.device(0),)), inventory_source=Source.NVIDIA_SMI)


@pytest.mark.parametrize("ref", ["gpu0", "nvml:-1", "nvml:01", "NVML:0", "smi:"])
def test_invalid_device_refs_are_rejected(build, ref):
    with pytest.raises(InvalidModelError):
        dataclasses.replace(build.device(0), device_ref=ref)


def test_device_ref_must_match_nvml_index(build):
    with pytest.raises(InvalidModelError, match="doesn't match nvml_index"):
        dataclasses.replace(build.device(0), nvml_index=1)


def test_vendor_and_mig_mode_are_constrained(build):
    with pytest.raises(InvalidModelError, match="must be 'NVIDIA'"):
        dataclasses.replace(build.device(0), vendor="AMD")
    with pytest.raises(InvalidModelError, match="mig_mode"):
        dataclasses.replace(build.device(0), mig_mode=ok("ON"))


def test_pci_observation_vendor_consistency():
    common = dict(
        address="0000:01:00.0",
        device_id="0x2684",
        class_code="0x030000",
        kernel_driver=ok("nouveau", Source.OS),
    )
    assert PciObservation(vendor_id="0x10de", is_nvidia=True, **common).is_nvidia
    with pytest.raises(InvalidModelError, match="is_nvidia"):
        PciObservation(vendor_id="0x8086", is_nvidia=True, **common)
    with pytest.raises(InvalidModelError, match="doesn't match"):
        PciObservation(vendor_id="0x10DE", is_nvidia=True, **common)


def test_runtime_must_refer_to_inventoried_devices(build):
    snapshot = build.snapshot(1)
    with pytest.raises(InvalidModelError, match="unknown devices"):
        dataclasses.replace(snapshot, runtime=(build.runtime("nvml:5"),))
    with pytest.raises(InvalidModelError, match="at most one entry"):
        dataclasses.replace(snapshot, runtime=(build.runtime("nvml:0"), build.runtime("nvml:0")))


def test_multi_gpu_values_stay_per_device(build):
    snapshot = build.snapshot(2)
    first, second = snapshot.runtime
    assert first.device_ref == "nvml:0" and second.device_ref == "nvml:1"
    assert first.processes.is_collected and second.processes.status is CollectionStatus.SKIPPED
    assert [p.pid for p in first.processes.value] == [17, 4242]


# --- partial failures -------------------------------------------------------------------


def test_partial_collector_failure_keeps_unrelated_results(build):
    report = build.partial_failure_report()
    healthy, lost = report.runtime
    assert healthy.memory_free_bytes.is_collected
    assert lost.memory_free_bytes.status is CollectionStatus.COLLECTION_FAILED
    assert lost.memory_free_bytes.value is None
    assert report.cuda_environment.toolkits.status is CollectionStatus.COLLECTION_FAILED
    assert report.system.os_name.is_collected and report.driver.driver_version.is_collected
    assert [(e.collector, e.device_ref) for e in report.collection_errors] == [
        ("cuda_environment", None),
        ("runtime", "nvml:1"),
    ]


def test_collection_error_and_unsupported_capability_rules():
    with pytest.raises(InvalidModelError, match="COLLECTED"):
        CollectionError(
            collector="nvml",
            device_ref=None,
            status=CollectionStatus.COLLECTED,
            error_code="X",
            message="m",
            report_path=None,
        )
    with pytest.raises(InvalidModelError, match="doesn't match"):
        CollectionError(
            collector="nvml",
            device_ref=None,
            status=CollectionStatus.COLLECTION_FAILED,
            error_code="X",
            message="m",
            report_path="runtime/0",
        )
    with pytest.raises(InvalidModelError, match="doesn't match"):
        UnsupportedCapability(
            capability="gpu scan",
            status=CollectionStatus.NOT_SUPPORTED,
            device_ref=None,
            reason="r",
        )


def test_collector_outcome_consistency(build):
    run = CollectorRun(collector="runtime", status=CollectorRunStatus.FAILED, duration_ms=5)
    error = CollectionError(
        collector="runtime",
        device_ref="nvml:0",
        status=CollectionStatus.COLLECTION_FAILED,
        error_code="X",
        message="m",
        report_path=None,
    )
    outcome = CollectorOutcome(section=build.runtime("nvml:0"), run=run, errors=[error])
    assert outcome.errors == (error,)
    with pytest.raises(InvalidModelError, match="FAILED run"):
        CollectorOutcome(section=None, run=run)
    completed = dataclasses.replace(run, status=CollectorRunStatus.COMPLETED)
    with pytest.raises(InvalidModelError, match="COMPLETED run"):
        CollectorOutcome(section=None, run=completed, errors=(error,))
    with pytest.raises(InvalidModelError, match="belong"):
        CollectorOutcome(
            section=None, run=dataclasses.replace(run, collector="system"), errors=(error,)
        )


# --- CUDA versions stay separate --------------------------------------------------------


def test_driver_supported_cuda_version_requires_its_label(build):
    with pytest.raises(InvalidModelError, match="not the installed CUDA Toolkit"):
        dataclasses.replace(build.driver(), driver_supported_cuda_version=ok("13.1"))


def test_build_cuda_version_rules(build):
    pytorch = build.pytorch()
    with pytest.raises(InvalidModelError, match="neither the toolkit"):
        dataclasses.replace(pytorch, build_cuda_version=ok("13.0", Source.PYTORCH))
    cpu_only = dataclasses.replace(
        pytorch,
        cuda_build=ok(False, Source.PYTORCH),
        build_cuda_version=Observed.not_applicable(note="CPU-only build"),
        cudnn_version=Observed.not_applicable(note="CPU-only build"),
    )
    assert cpu_only.build_cuda_version.value is None
    with pytest.raises(InvalidModelError, match="NOT_APPLICABLE for a non-CUDA build"):
        dataclasses.replace(cpu_only, cudnn_version=ok("91000", Source.PYTORCH))
    assert BUILD_CUDA_NOTE in pytorch.build_cuda_version.note


def test_allocator_stats_not_applicable_in_cli(build):
    allocator = build.pytorch().allocator
    assert allocator.status is CollectionStatus.NOT_APPLICABLE
    assert allocator.value is None
    assert allocator.note == ALLOCATOR_NOT_APPLICABLE_NOTE


def test_toolkit_candidates_and_environment_allowlist(build):
    candidate = CudaToolkitCandidate(
        root_path="/opt/cuda",
        discovered_via=[
            ToolkitDiscovery.DEFAULT_LOCATION,
            ToolkitDiscovery.CUDA_HOME,
            ToolkitDiscovery.CUDA_HOME,
        ],
        version=ok("12.4.1", Source.CUDA_TOOLKIT),
        nvcc_path=Observed.failed(
            CollectionStatus.NOT_DETECTED, code="FileNotFoundError", message="no nvcc"
        ),
    )
    assert candidate.discovered_via == (
        ToolkitDiscovery.CUDA_HOME,
        ToolkitDiscovery.DEFAULT_LOCATION,
    )
    env = build.cuda_environment()
    assert list(env.environment) == sorted(env.environment)
    assert env.environment["CUDA_VISIBLE_DEVICES"].value == ""
    with pytest.raises(InvalidModelError, match="allowlist"):
        dataclasses.replace(env, environment={"HOME": ok("/home/x", Source.ENVIRONMENT)})
    windows = dataclasses.replace(
        env, environment={"CUDA_PATH_V12_4": ok("C:/c", Source.ENVIRONMENT)}
    )
    assert "CUDA_PATH_V12_4" in windows.environment
    with pytest.raises(InvalidModelError, match="NOT_DETECTED"):
        dataclasses.replace(env, toolkits=ok((), Source.CUDA_TOOLKIT))


# --- timestamps ---------------------------------------------------------------------------


def test_timestamps_must_be_timezone_aware_and_are_normalised_to_utc(build):
    with pytest.raises(InvalidModelError, match="timezone-aware"):
        dataclasses.replace(build.runtime("nvml:0"), sampled_at=datetime(2026, 1, 1))
    ist = timezone(timedelta(hours=5, minutes=30))
    metrics = dataclasses.replace(
        build.runtime("nvml:0"), sampled_at=datetime(2026, 1, 1, 5, 30, tzinfo=ist)
    )
    assert metrics.sampled_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert metrics.sampled_at.tzinfo is UTC


# --- diagnostics ------------------------------------------------------------------------


def test_finding_rules(build):
    finding = build.warning_finding()
    assert finding.detection_sources == (Source.INSPECTOR, Source.NVML)
    with pytest.raises(InvalidModelError, match="recommendation"):
        dataclasses.replace(finding, recommendations=())
    with pytest.raises(InvalidModelError, match="limitation"):
        dataclasses.replace(finding, limitations=())
    with pytest.raises(InvalidModelError, match="evidence"):
        dataclasses.replace(finding, evidence=())
    with pytest.raises(InvalidModelError, match="doesn't match"):
        dataclasses.replace(finding, check_id="DX1")
    not_applicable = DiagnosticFinding(
        check_id="D02",
        device_ref=None,
        status=DiagnosticStatus.NOT_APPLICABLE,
        severity=Severity.NONE,
        title="Mode A",
        explanation="no --required-vram",
        evidence=(),
        detection_sources=(),
        recommendations=(),
        confidence=Confidence.NOT_APPLICABLE,
        limitations=(),
    )
    with pytest.raises(InvalidModelError, match="NOT_APPLICABLE confidence"):
        dataclasses.replace(not_applicable, confidence=Confidence.HIGH)


def test_evidence_rules():
    with pytest.raises(InvalidModelError, match="INFERENCE"):
        Evidence(kind=EvidenceKind.INFERENCE, statement="s", source=None, report_path=None, value=1)
    with pytest.raises(InvalidModelError, match="not a JSON value"):
        Evidence(
            kind=EvidenceKind.OBSERVATION,
            statement="s",
            source=None,
            report_path=None,
            value=object(),
        )
    frozen = Evidence(
        kind=EvidenceKind.OBSERVATION,
        statement="s",
        source=Source.NVML,
        report_path="/gpu/devices/0/name",
        value={"b": [1, 2], "a": None},
    )
    assert frozen.value["b"] == (1, 2)
    assert list(frozen.value) == ["a", "b"]


def test_report_rejects_findings_for_unknown_devices_and_duplicates(build):
    with pytest.raises(InvalidModelError, match="unknown device"):
        build.report(diagnostics=(build.finding("D06", "nvml:9"),))
    with pytest.raises(InvalidModelError, match="only once"):
        build.report(diagnostics=(build.finding(), build.finding()))
    report = build.report(diagnostics=(build.warning_finding(), build.finding("D01")))
    assert [d.check_id for d in report.diagnostics] == ["D01", "D07"]


# --- readiness ----------------------------------------------------------------------------


def test_readiness_model_consistency(build):
    readiness = build.readiness(8, {"nvml:1": 5 * GIB, "nvml:0": 20 * GIB})
    assert readiness.required_bytes == 8589934592
    assert [d.device_ref for d in readiness.devices] == ["nvml:0", "nvml:1"]
    assert readiness.devices[1].margin_bytes == -3 * GIB
    with pytest.raises(InvalidModelError, match="required_bytes"):
        dataclasses.replace(readiness, required_bytes=8 * 10**9)
    with pytest.raises(InvalidModelError, match="doesn't fit overall status"):
        dataclasses.replace(readiness, overall_reason=ReadinessReason.NO_DEVICE_READY)
    with pytest.raises(InvalidModelError, match="must be 'Free memory"):
        dataclasses.replace(readiness, disclaimer="It fits.")
    bad_margin = dataclasses.replace(readiness.devices[0], margin_bytes=1)
    with pytest.raises(InvalidModelError, match="margin_bytes must equal"):
        dataclasses.replace(readiness, devices=(bad_margin, readiness.devices[1]))


def test_device_readiness_status_reason_pairing():
    unknown_free = Observed.failed(
        CollectionStatus.PERMISSION_DENIED, code="X", message="y", unit="bytes"
    )
    device = DeviceReadiness(
        device_ref="nvml:0",
        status=ReadinessStatus.UNKNOWN,
        reason=ReadinessReason.FREE_MEMORY_UNAVAILABLE,
        free_bytes=unknown_free,
        total_bytes=ok(24 * GIB, unit="bytes"),
        margin_bytes=None,
        explanation="no free memory",
    )
    assert device.margin_bytes is None
    with pytest.raises(InvalidModelError, match="doesn't fit device status"):
        dataclasses.replace(device, status=ReadinessStatus.READY)
    with pytest.raises(InvalidModelError, match="margin_bytes is set"):
        dataclasses.replace(device, margin_bytes=0)


def test_required_bytes_conversion():
    assert required_bytes_for(8) == 8589934592
    assert required_bytes_for(7.5) == 8053063680


def test_mode_options_and_readiness_must_agree(build):
    report = build.report(required_vram_gib=8)
    assert report.mode is InspectionMode.READINESS
    with pytest.raises(InvalidModelError, match="mode is READINESS"):
        dataclasses.replace(report, mode=InspectionMode.GENERAL)
    with pytest.raises(InvalidModelError, match="readiness is present"):
        dataclasses.replace(report, readiness=None)
    with pytest.raises(InvalidModelError, match="exactly the inventoried devices"):
        build.report(
            required_vram_gib=8,
            readiness=build.readiness(8, {"nvml:0": 20 * GIB}),
        )


def test_report_requires_snapshot_limitation(build):
    with pytest.raises(InvalidModelError, match="snapshot statement"):
        build.report(limitations=("Something else.",))


def test_options_validation_matches_cli_ranges():
    assert InspectionOptions().subprocess_timeout_s == 10.0
    for bad in (
        dict(required_vram_gib=0.0),
        dict(required_vram_gib=1e-12),
        dict(required_vram_gib=1048577.0),
        dict(subprocess_timeout_s=0.0),
        dict(pytorch_timeout_s=601.0),
    ):
        with pytest.raises(InvalidModelError):
            InspectionOptions(**bad)


def test_redaction_info_rules():
    info = RedactionInfo(
        applied=True,
        categories=[RedactionCategory.USER, RedactionCategory.PATH, RedactionCategory.USER],
    )
    assert info.categories == (RedactionCategory.PATH, RedactionCategory.USER)
    with pytest.raises(InvalidModelError, match="applied"):
        RedactionInfo(applied=False, categories=(RedactionCategory.PATH,))


def test_snapshot_round_trips_through_report(build):
    snapshot = build.snapshot(2)
    report = InspectionReport.from_snapshot(
        snapshot,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_ms=1,
        data_origin=build.report().data_origin,
        options=InspectionOptions(),
    )
    assert report.snapshot == snapshot
