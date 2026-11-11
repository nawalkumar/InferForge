"""FX01–FX10 (testing strategy §4; expected outcomes from diagnostics spec §6 and §4.1)."""

from __future__ import annotations

from dataclasses import replace

from gpu_system_inspector.collectors.capabilities import unsupported_capabilities_for
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    Confidence,
    CUDAEnvironmentInfo,
    DiagnosticStatus,
    ErrorDetail,
    FrameworkInfo,
    InspectionSnapshot,
    LinuxEnvironment,
    PlatformProfile,
    PyTorchInfo,
    ReadinessReason,
    RuntimeMetrics,
    Severity,
    Source,
)
from gsi_testkit.builders import (
    FIXED_TIME,
    GIB,
    GPU_A,
    GPU_B,
    collection_error,
    collector_runs,
    cuda_environment,
    driver_nvml,
    driver_without_nvml,
    failed,
    gpu_device,
    no_inventory,
    nvml_failure,
    nvml_inventory,
    pci_device,
    pci_not_supported,
    pci_scan,
    pytorch_cpu_only,
    pytorch_cuda,
    pytorch_not_installed,
    runtime_metrics,
    smi_inventory,
    system_info,
    torch_device,
    unsupported,
)
from gsi_testkit.expectations import (
    device_expectation,
    expect,
    healthy,
    no_gpu,
    per_device,
    readiness_run,
    replace_checks,
)
from gsi_testkit.fixtures.scenario import FixtureScenario, Premise

S = DiagnosticStatus
R = ReadinessReason


def _torch(s: InspectionSnapshot) -> PyTorchInfo:
    return s.frameworks.pytorch


def single_gpu_snapshot(
    *,
    free: int = 20 * GIB,
    pytorch: PyTorchInfo | None = None,
    supported_cuda: str = "12.4",
    toolkit_version: str | None = "12.4.131",
    env: dict[str, str | None] | None = None,
    binding_distribution: str = "nvidia-ml-py",
    **runtime_overrides,
) -> InspectionSnapshot:
    """The FX01 machine: Linux, NVML OK, one GPU A, toolkit 12.4, PyTorch CUDA 12.4."""
    device = gpu_device(0, GPU_A)
    return InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=nvml_inventory((device,), pci_scan(pci_device(1))),
        driver=driver_nvml(
            supported_cuda=supported_cuda, binding_distribution=binding_distribution
        ),
        cuda_environment=cuda_environment(toolkit_version=toolkit_version, env=env),
        frameworks=FrameworkInfo(
            pytorch=pytorch or pytorch_cuda([torch_device(0, GPU_A, correlated_ref="nvml:0")])
        ),
        runtime=(runtime_metrics("nvml:0", GPU_A.total_bytes, free, **runtime_overrides),),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )


def _fx01_b_run():
    return readiness_run(
        "FX01 (B)",
        8,
        R.AT_LEAST_ONE_DEVICE_READY,
        [device_expectation("nvml:0", R.SUFFICIENT_FREE_MEMORY)],
        margins=[("nvml:0", 12 * GIB)],
    )


def fx01() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX01",
        name="NVIDIA GPU with working CUDA",
        description="Linux, NVML OK, one GPU A with 20 GiB free; driver supports CUDA 12.4; "
        "toolkit 12.4 via version.json; PyTorch built with CUDA 12.4, CUDA available, "
        "correlated by UUID; all runtime metrics collected.",
        snapshot=single_gpu_snapshot(),
        expected_diagnostics=healthy(("nvml:0",)),
        readiness_runs=(_fx01_b_run(),),
        premises=(
            Premise(
                "PyTorch reports CUDA available", lambda s: _torch(s).cuda_available.value is True
            ),
            Premise(
                "PyTorch device correlated by UUID",
                lambda s: _torch(s).devices[0].correlated_device_ref.method == "fixture:uuid",
            ),
            Premise(
                "every runtime metric collected",
                lambda s: all(
                    getattr(s.runtime[0], n).is_collected
                    for n in (
                        "gpu_utilization_percent",
                        "memory_free_bytes",
                        "temperature_gpu_celsius",
                        "processes",
                    )
                ),
            ),
        ),
    )


def _cuda_unavailable(**kw) -> PyTorchInfo:
    return pytorch_cuda([], cuda_available=False, **kw)


def fx02a() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX02a",
        name="NVIDIA GPU, PyTorch CUDA unavailable: CPU-only build",
        description="As FX01, but PyTorch is a CPU-only build (cuda_build = false).",
        snapshot=single_gpu_snapshot(pytorch=pytorch_cpu_only()),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect("D01", S.FAIL, "D01 row 6, cause D01.1 (CPU-only build)"),
        ),
        premises=(
            Premise("cuda_build is false", lambda s: _torch(s).cuda_build.value is False),
            Premise("NVML sees a GPU", lambda s: s.gpu.device_count.value == 1),
        ),
    )


def fx02b() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX02b",
        name="NVIDIA GPU, PyTorch CUDA unavailable: driver older than build",
        description="Driver supports CUDA 11.8; PyTorch built with CUDA 12.4; CUDA unavailable.",
        snapshot=single_gpu_snapshot(supported_cuda="11.8", pytorch=_cuda_unavailable()),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect(
                "D01",
                S.FAIL,
                "D01 row 6, cause D01.5 (driver older than build)",
                confidence=Confidence.MEDIUM,
            ),
            expect(
                "D03",
                S.WARNING,
                "D03 row 2 (build major > driver major)",
                confidence=Confidence.MEDIUM,
            ),
        ),
        premises=(
            Premise(
                "driver CUDA major < build CUDA major",
                lambda s: (
                    int(s.driver.driver_supported_cuda_version.value.split(".")[0])
                    < int(_torch(s).build_cuda_version.value.split(".")[0])
                ),
            ),
        ),
        limitations=("Forward-compatibility packages are not modelled.",),
    )


def fx02c() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX02c",
        name="NVIDIA GPU, PyTorch CUDA unavailable: devices hidden",
        description='CUDA_VISIBLE_DEVICES="" hides every device; PyTorch sees 0 devices.',
        snapshot=single_gpu_snapshot(env={"CUDA_VISIBLE_DEVICES": ""}, pytorch=_cuda_unavailable()),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect("D01", S.FAIL, "D01 row 6, cause D01.2 (devices hidden)"),
            expect("D08", S.WARNING, 'D08 row 2 (CUDA_VISIBLE_DEVICES = "")'),
        ),
        premises=(
            Premise(
                "CUDA_VISIBLE_DEVICES collected as empty string",
                lambda s: s.cuda_environment.environment["CUDA_VISIBLE_DEVICES"].value == "",
            ),
            Premise("PyTorch device_count is 0", lambda s: _torch(s).device_count.value == 0),
        ),
    )


def fx02d() -> FixtureScenario:
    warning = "CUDA initialization: Unexpected error from cudaGetDeviceCount() (synthetic)"
    return FixtureScenario(
        fixture_id="FX02d",
        name="NVIDIA GPU, PyTorch CUDA unavailable: unexplained",
        description="All layers healthy, but CUDA is unavailable and PyTorch emitted an "
        "initialisation warning.",
        snapshot=single_gpu_snapshot(pytorch=_cuda_unavailable(warnings=[warning])),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect(
                "D01", S.FAIL, "D01 row 6, no cause D01.1–D01.6 matched", confidence=Confidence.LOW
            ),
        ),
        premises=(
            Premise(
                "PyTorch warning captured", lambda s: _torch(s).cuda_init_warnings == (warning,)
            ),
            Premise(
                "CUDA build and driver are compatible",
                lambda s: (
                    _torch(s).cuda_build.value is True
                    and s.driver.driver_supported_cuda_version.value == "12.4"
                ),
            ),
        ),
    )


def fx03() -> FixtureScenario:
    missing = nvml_failure(
        CollectionStatus.NOT_DETECTED,
        "NVML_ERROR_LIBRARY_NOT_FOUND",
        "NVML Shared Library Not Found",
    )
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=no_inventory(
            missing, pci_scan(pci_device(0, vendor_id="0x8086", device_id="0x46a6", driver="i915"))
        ),
        driver=driver_without_nvml(missing),
        cuda_environment=cuda_environment(toolkit_version=None),
        frameworks=FrameworkInfo(pytorch=pytorch_cpu_only()),
        runtime=(),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )
    return FixtureScenario(
        fixture_id="FX03",
        name="No NVIDIA GPU",
        description="Linux bare metal; NVML library not found; no nvidia-smi; PCI shows one "
        "Intel display controller; PyTorch CPU-only installed.",
        snapshot=snapshot,
        expected_diagnostics=no_gpu(
            expect("D01", S.INFO, "D01 row 7 (NVIDIA GPU observed absent)"),
            expect("D04", S.INFO, "D04 row 7 (PCI collected, no NVIDIA device)"),
            expect(
                "D05", S.PASS, "D05 (nvidia-smi missing with NVML unavailable: no contribution)"
            ),
        ),
        readiness_runs=(
            readiness_run("FX03 (B)", 8, R.NO_NVIDIA_DEVICE, d02_confidence=Confidence.MEDIUM),
        ),
        premises=(
            Premise(
                "PCI scan collected with no NVIDIA device",
                lambda s: (
                    s.gpu.pci_observations.is_collected
                    and not any(o.is_nvidia for o in s.gpu.pci_observations.value)
                ),
            ),
            Premise(
                "NVML not detected",
                lambda s: s.driver.nvml_version.status is CollectionStatus.NOT_DETECTED,
            ),
        ),
    )


def fx04a() -> FixtureScenario:
    device = gpu_device(0, GPU_A, source=Source.NVIDIA_SMI)
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=smi_inventory((device,), pci_scan(pci_device(1))),
        driver=driver_without_nvml(
            failed(
                CollectionStatus.NOT_INSTALLED,
                "ImportError",
                "No module named 'pynvml'",
                source=Source.NVML,
                method="import pynvml",
            ),
            binding_installed=False,
            smi_path="/usr/bin/nvidia-smi",
        ),
        cuda_environment=cuda_environment(),
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda([torch_device(0, GPU_A, correlated_ref=None)])
        ),
        runtime=(runtime_metrics("smi:0", GPU_A.total_bytes, 20 * GIB, source=Source.NVIDIA_SMI),),
        collection_errors=(
            collection_error(
                "nvml",
                "ImportError",
                "No module named 'pynvml'",
                status=CollectionStatus.NOT_INSTALLED,
                report_path="/driver/nvml_binding",
            ),
        ),
        unsupported_capabilities=(),
        collector_runs=collector_runs(nvml=CollectorRunStatus.PARTIAL),
    )
    return FixtureScenario(
        fixture_id="FX04a",
        name="NVML missing (binding; broken installation of a required dependency)",
        description="pynvml import fails; nvidia-smi provides 1 × GPU A via CSV; toolkit 12.4; "
        "PyTorch CUDA available.",
        snapshot=snapshot,
        expected_diagnostics=(
            expect("D01", S.PASS, "D01 row 5"),
            expect("D02", S.NOT_APPLICABLE, "D02 Mode A (no --required-vram)"),
            expect("D03", S.INFO, "D03 row 5"),
            expect(
                "D04",
                S.PASS,
                "D04 row 2 (nvidia-smi fallback inventory)",
                confidence=Confidence.MEDIUM,
            ),
            expect("D05", S.WARNING, "D05 required binding missing with NVIDIA evidence"),
            *per_device("D06", ("smi:0",), S.INFO, "D06 row 4 (fallback fields NOT_SUPPORTED)"),
            *per_device("D07", ("smi:0",), S.INFO, "D07 row 3 (no threshold in fallback)"),
            expect("D08", S.PASS, "D08 row 7"),
        ),
        premises=(
            Premise(
                "inventory comes from nvidia-smi",
                lambda s: s.gpu.inventory_source is Source.NVIDIA_SMI,
            ),
            Premise(
                "required binding not installed",
                lambda s: s.driver.nvml_binding.status is CollectionStatus.NOT_INSTALLED,
            ),
        ),
        limitations=("nvidia-smi values have 1 MiB precision.",),
    )


def fx04b() -> FixtureScenario:
    missing = nvml_failure(
        CollectionStatus.NOT_DETECTED,
        "NVML_ERROR_LIBRARY_NOT_FOUND",
        "NVML Shared Library Not Found",
    )
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.WSL2),
        gpu=no_inventory(missing, pci_not_supported(LinuxEnvironment.WSL2)),
        driver=driver_without_nvml(missing),
        cuda_environment=cuda_environment(toolkit_version=None),
        frameworks=FrameworkInfo(pytorch=pytorch_not_installed()),
        runtime=(),
        collection_errors=(),
        unsupported_capabilities=unsupported_capabilities_for(
            LinuxEnvironment.WSL2, PlatformProfile.WSL2
        ),
        collector_runs=collector_runs(pytorch=CollectorRunStatus.COMPLETED),
    )
    return FixtureScenario(
        fixture_id="FX04b",
        name="NVML missing (library)",
        description="WSL2; binding installed; NVML library not found; no nvidia-smi; PCI not "
        "supported; PyTorch not installed (mirrors OBS-01).",
        snapshot=snapshot,
        expected_diagnostics=no_gpu(
            expect("D01", S.NOT_APPLICABLE, "D01 row 1 (PyTorch not installed)"),
            expect(
                "D04",
                S.INFO,
                "D04 row 8 (PCI not supported, NVML not detected)",
                confidence=Confidence.MEDIUM,
            ),
            expect("D05", S.INFO, "D05 PyTorch not installed"),
        ),
        readiness_runs=(readiness_run("FX04b (B)", 8, R.DEVICE_PRESENCE_UNKNOWN),),
        premises=(
            Premise("WSL2 profile", lambda s: s.system.platform_profile is PlatformProfile.WSL2),
            Premise(
                "PCI scan not supported",
                lambda s: s.gpu.pci_observations.status is CollectionStatus.NOT_SUPPORTED,
            ),
        ),
        limitations=("Absence of a GPU can't be confirmed under WSL2.",),
    )


def fx05() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX05",
        name="PyTorch missing",
        description="As FX01, but the torch distribution is not installed.",
        snapshot=single_gpu_snapshot(pytorch=pytorch_not_installed()),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect("D01", S.NOT_APPLICABLE, "D01 row 1 (PyTorch not installed)"),
            expect("D05", S.INFO, "D05 PyTorch not installed"),
        ),
        premises=(
            Premise(
                "PyTorch NOT_INSTALLED",
                lambda s: _torch(s).status is CollectionStatus.NOT_INSTALLED,
            ),
        ),
    )


def fx06() -> FixtureScenario:
    devices = (gpu_device(0, GPU_B, bus=1), gpu_device(1, GPU_A, bus=2))
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=nvml_inventory(devices, pci_scan(pci_device(1), pci_device(2))),
        driver=driver_nvml(),
        cuda_environment=cuda_environment(),
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda(
                [
                    torch_device(0, GPU_A, correlated_ref="nvml:1", bus=2),
                    torch_device(1, GPU_B, correlated_ref="nvml:0", bus=1),
                ]
            )
        ),
        runtime=(
            runtime_metrics("nvml:0", GPU_B.total_bytes, 10 * GIB),
            runtime_metrics("nvml:1", GPU_A.total_bytes, 20 * GIB),
        ),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )
    refs = ("nvml:0", "nvml:1")
    return FixtureScenario(
        fixture_id="FX06",
        name="Multiple GPUs",
        description="NVML 0 = GPU B (bus 01), NVML 1 = GPU A (bus 02); PyTorch cuda:0 = GPU A "
        "(UUID match to nvml:1), cuda:1 = GPU B; CUDA_DEVICE_ORDER unset.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(refs),
            expect("D08", S.INFO, "D08 row 5 (PyTorch index != NVML index)", severity=Severity.LOW),
        ),
        readiness_runs=(
            readiness_run(
                "FX06 (B)",
                16,
                R.AT_LEAST_ONE_DEVICE_READY,
                [
                    device_expectation("nvml:0", R.INSUFFICIENT_TOTAL_MEMORY),
                    device_expectation("nvml:1", R.SUFFICIENT_FREE_MEMORY),
                ],
                margins=[("nvml:1", 4 * GIB), ("nvml:0", -6 * GIB)],
            ),
        ),
        premises=(
            Premise(
                "cuda:0 correlates to nvml:1",
                lambda s: _torch(s).devices[0].correlated_device_ref.value == "nvml:1",
            ),
            Premise(
                "CUDA_DEVICE_ORDER unset",
                lambda s: not s.cuda_environment.environment["CUDA_DEVICE_ORDER"].is_collected,
            ),
        ),
    )


def fx07() -> FixtureScenario:
    devices = (gpu_device(0, GPU_A), gpu_device(1, GPU_B))
    free_b = round(7.99 * GIB)
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=nvml_inventory(devices, pci_scan(pci_device(1), pci_device(2))),
        driver=driver_nvml(),
        cuda_environment=cuda_environment(),
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda(
                [
                    torch_device(0, GPU_A, correlated_ref="nvml:0"),
                    torch_device(1, GPU_B, correlated_ref="nvml:1"),
                ]
            )
        ),
        runtime=(
            runtime_metrics("nvml:0", GPU_A.total_bytes, 5 * GIB),
            runtime_metrics("nvml:1", GPU_B.total_bytes, free_b),
        ),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )
    run = readiness_run(
        "FX07",
        8,
        R.NO_DEVICE_READY,
        [
            device_expectation("nvml:0", R.INSUFFICIENT_FREE_MEMORY),
            device_expectation("nvml:1", R.INSUFFICIENT_FREE_MEMORY),
        ],
        margins=[("nvml:0", -3 * GIB), ("nvml:1", free_b - 8 * GIB)],
    )
    return FixtureScenario(
        fixture_id="FX07",
        name="Insufficient free memory",
        description="Mode B, --required-vram 8; two devices with 5 GiB and 7.99 GiB free.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(healthy(("nvml:0", "nvml:1")), run.d02),
        readiness_runs=(run,),
        mode_b_only=True,
        premises=(
            Premise(
                "both devices have less than 8 GiB free",
                lambda s: all(r.memory_free_bytes.value < 8 * GIB for r in s.runtime),
            ),
            Premise(
                "PyTorch indices equal NVML indices",
                lambda s: all(
                    d.correlated_device_ref.value == f"nvml:{d.torch_index}"
                    for d in _torch(s).devices
                ),
            ),
        ),
    )


def fx08() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX08",
        name="Driver present, toolkit unavailable",
        description="As FX01, but no CUDA Toolkit candidates and no nvcc.",
        snapshot=single_gpu_snapshot(toolkit_version=None),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect("D03", S.INFO, "D03 row 5 with the toolkit-absent note"),
        ),
        premises=(
            Premise(
                "no toolkit detected",
                lambda s: s.cuda_environment.toolkits.status is CollectionStatus.NOT_DETECTED,
            ),
        ),
    )


def fx09() -> FixtureScenario:
    denied = failed(
        CollectionStatus.PERMISSION_DENIED,
        "NVML_ERROR_NO_PERMISSION",
        "Insufficient Permissions",
        method="nvmlDeviceGetMemoryInfo_v2",
        unit="bytes",
    )
    unsupported_util = failed(
        CollectionStatus.NOT_SUPPORTED,
        "NVML_ERROR_NOT_SUPPORTED",
        "Not Supported",
        method="nvmlDeviceGetUtilizationRates",
        unit="percent",
    )
    temp_unsupported = failed(
        CollectionStatus.NOT_SUPPORTED,
        "NVML_ERROR_NOT_SUPPORTED",
        "Not Supported",
        method="nvmlDeviceGetTemperatureV",
        unit="celsius",
    )
    power_denied = failed(
        CollectionStatus.PERMISSION_DENIED,
        "NVML_ERROR_NO_PERMISSION",
        "Insufficient Permissions",
        method="nvmlDeviceGetPowerUsage",
        unit="watts",
    )
    base = single_gpu_snapshot(
        gpu_utilization_percent=unsupported_util,
        memory_bus_utilization_percent=unsupported_util,
        memory_total_bytes=denied,
        memory_used_bytes=denied,
        memory_free_bytes=denied,
        memory_reserved_bytes=denied,
        temperature_gpu_celsius=temp_unsupported,
        temperature_slowdown_threshold_celsius=temp_unsupported,
        power_draw_watts=power_denied,
        power_limit_watts=power_denied,
        processes=failed(
            CollectionStatus.PERMISSION_DENIED,
            "NVML_ERROR_NO_PERMISSION",
            "Insufficient Permissions",
            method="nvmlDeviceGetComputeRunningProcesses_v3",
        ),
    )
    device = replace(base.gpu.devices[0], memory_total_bytes=denied)
    snapshot = replace(
        base,
        gpu=replace(base.gpu, devices=(device,)),
        unsupported_capabilities=(
            unsupported(
                "RUNTIME.GPU_UTILIZATION",
                "NVML reports utilization as not supported.",
                device_ref="nvml:0",
            ),
            unsupported(
                "RUNTIME.PROCESSES",
                "Process enumeration requires more privileges.",
                device_ref="nvml:0",
                status=CollectionStatus.PERMISSION_DENIED,
            ),
        ),
        collector_runs=collector_runs(
            gpu_hardware=CollectorRunStatus.PARTIAL, runtime=CollectorRunStatus.PARTIAL
        ),
    )
    return FixtureScenario(
        fixture_id="FX09",
        name="Runtime metrics inaccessible",
        description="As FX01, but memory info and processes are PERMISSION_DENIED, utilization "
        "and temperature NOT_SUPPORTED, power PERMISSION_DENIED.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            *per_device("D06", ("nvml:0",), S.WARNING, "D06 row 3 (PERMISSION_DENIED fields)"),
            *per_device(
                "D07", ("nvml:0",), S.NOT_APPLICABLE, "D07 row 1 (temperature not collected)"
            ),
        ),
        readiness_runs=(
            readiness_run(
                "FX09 (B)",
                8,
                R.SOME_DEVICES_UNKNOWN,
                [device_expectation("nvml:0", R.FREE_MEMORY_UNAVAILABLE)],
            ),
        ),
        premises=(
            Premise(
                "free memory PERMISSION_DENIED",
                lambda s: (
                    s.runtime[0].memory_free_bytes.status is CollectionStatus.PERMISSION_DENIED
                ),
            ),
            Premise(
                "no runtime metric fabricated as zero",
                lambda s: s.runtime[0].memory_free_bytes.value is None,
            ),
        ),
    )


def fx10() -> FixtureScenario:
    devices = (gpu_device(0, GPU_A), gpu_device(1, GPU_B))
    crash = RuntimeError("synthetic collector crash (fixture)")
    crashed = ErrorDetail(code="UNHANDLED_EXCEPTION", message=f"RuntimeError: {crash}")
    lost = ErrorDetail(code="NVML_ERROR_GPU_IS_LOST", message="GPU is lost")
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=nvml_inventory(devices, pci_scan(pci_device(1), pci_device(2))),
        driver=driver_nvml(),
        cuda_environment=CUDAEnvironmentInfo.unavailable(
            CollectionStatus.COLLECTION_FAILED, crashed
        ),
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda(
                [
                    torch_device(0, GPU_A, correlated_ref="nvml:0"),
                    torch_device(1, GPU_B, correlated_ref="nvml:1"),
                ]
            )
        ),
        runtime=(
            runtime_metrics("nvml:0", GPU_A.total_bytes, 20 * GIB),
            RuntimeMetrics.unavailable(
                "nvml:1", FIXED_TIME, CollectionStatus.COLLECTION_FAILED, lost
            ),
        ),
        collection_errors=(
            collection_error(
                "cuda_environment", crashed.code, crashed.message, report_path="/cuda_environment"
            ),
            collection_error(
                "runtime", lost.code, lost.message, device_ref="nvml:1", report_path="/runtime/1"
            ),
        ),
        unsupported_capabilities=(),
        collector_runs=collector_runs(
            cuda_environment=CollectorRunStatus.FAILED, runtime=CollectorRunStatus.PARTIAL
        ),
    )
    return FixtureScenario(
        fixture_id="FX10",
        name="Partial collector failure",
        description="Two devices; the CUDA-environment collector raises RuntimeError; nvml:1 "
        "is lost after enumeration, so its runtime queries raise NVMLError_GpuIsLost; "
        "everything else healthy.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0", "nvml:1")),
            expect("D06", S.PASS, "D06 row 5", device_ref="nvml:0"),
            expect("D06", S.FAIL, "D06 row 2 (GPU lost)", device_ref="nvml:1"),
            expect("D07", S.INFO, "D07 row 3", device_ref="nvml:0"),
            expect(
                "D07",
                S.NOT_APPLICABLE,
                "D07 row 1 (temperature not collected)",
                device_ref="nvml:1",
            ),
        ),
        premises=(
            Premise("nvml:0 runtime intact", lambda s: s.runtime[0].memory_free_bytes.is_collected),
            Premise("nvml:1 runtime lost", lambda s: s.runtime[1].memory_free_bytes.error == lost),
            Premise(
                "CUDA environment section failed but driver/system intact",
                lambda s: (
                    s.cuda_environment.toolkits.status is CollectionStatus.COLLECTION_FAILED
                    and s.driver.driver_version.is_collected
                ),
            ),
        ),
        collector_failures={"cuda_environment": crash},
        # KI-34: nvml:1 is enumerated (one handle), then lost. The fault model used to be
        # session-wide faults on the runtime functions, `nvmlDeviceGetMemoryInfo_v2`
        # included, which also failed the enumeration's memory read that this snapshot
        # records as collected (LEGACY_FX10_FAULTS in tests/unit/test_device_loss.py).
        nvml_lost_after_handles={1: 1},
    )


CORE_FIXTURES = (
    fx01,
    fx02a,
    fx02b,
    fx02c,
    fx02d,
    fx03,
    fx04a,
    fx04b,
    fx05,
    fx06,
    fx07,
    fx08,
    fx09,
    fx10,
)
