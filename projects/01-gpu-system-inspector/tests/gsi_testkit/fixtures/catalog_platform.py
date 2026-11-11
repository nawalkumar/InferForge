"""FX11–FX20 (testing strategy §4; expected outcomes from diagnostics spec §6 and §4.1)."""

from __future__ import annotations

from dataclasses import replace

from gpu_system_inspector.collectors.capabilities import unsupported_capabilities_for
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    Confidence,
    DiagnosticStatus,
    FrameworkInfo,
    GpuProcess,
    InspectionSnapshot,
    LinuxEnvironment,
    PlatformProfile,
    ProcessKind,
    ReadinessReason,
    Severity,
    Source,
    ToolkitDiscovery,
)
from gsi_testkit.builders import (
    GIB,
    GPU_A,
    GPU_M,
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
    ok,
    pci_device,
    pci_not_supported,
    pci_scan,
    pytorch_cpu_only,
    pytorch_cuda,
    pytorch_not_installed,
    pytorch_probe_failed,
    runtime_metrics,
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
from gsi_testkit.fixtures.catalog_core import single_gpu_snapshot
from gsi_testkit.fixtures.scenario import FixtureScenario, Premise

S = DiagnosticStatus
R = ReadinessReason


def _nvml_platform_snapshot(
    profile: PlatformProfile, *, pci, cuda_env, smi_path: str, **runtime_overrides
) -> InspectionSnapshot:
    device = gpu_device(0, GPU_A)
    return InspectionSnapshot(
        system=system_info(profile),
        gpu=nvml_inventory((device,), pci),
        driver=driver_nvml(smi_path=smi_path),
        cuda_environment=cuda_env,
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda([torch_device(0, GPU_A, correlated_ref="nvml:0")])
        ),
        runtime=(runtime_metrics("nvml:0", GPU_A.total_bytes, 20 * GIB, **runtime_overrides),),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )


def fx11() -> FixtureScenario:
    wsl_note = "Documented as not yet supported on WSL2 (CUDA on WSL User Guide)"
    util = failed(
        CollectionStatus.NOT_SUPPORTED,
        "NVML_ERROR_NOT_SUPPORTED",
        "Not Supported",
        method="nvmlDeviceGetUtilizationRates",
        unit="percent",
        note=wsl_note,
    )
    snapshot = _nvml_platform_snapshot(
        PlatformProfile.WSL2,
        pci=pci_not_supported(LinuxEnvironment.WSL2),
        cuda_env=cuda_environment(),
        smi_path="/usr/lib/wsl/lib/nvidia-smi",
        gpu_utilization_percent=util,
        memory_bus_utilization_percent=util,
        processes=failed(
            CollectionStatus.NOT_SUPPORTED,
            "NVML_ERROR_NOT_SUPPORTED",
            "Not Supported",
            method="nvmlDeviceGetComputeRunningProcesses_v3",
            note=wsl_note,
        ),
    )
    snapshot = replace(
        snapshot,
        unsupported_capabilities=(
            *unsupported_capabilities_for(LinuxEnvironment.WSL2, PlatformProfile.WSL2),
            unsupported("RUNTIME.GPU_UTILIZATION", f"{wsl_note}.", device_ref="nvml:0"),
            unsupported("RUNTIME.PROCESSES", f"{wsl_note}.", device_ref="nvml:0"),
        ),
    )
    return FixtureScenario(
        fixture_id="FX11",
        name="WSL2 with NVIDIA",
        description="WSL2; NVML OK via /usr/lib/wsl/lib; utilization and processes not "
        "supported; toolkit 12.4; PyTorch CUDA available.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            *per_device("D06", ("nvml:0",), S.INFO, "D06 row 4 (WSL2 documented limitation)"),
        ),
        premises=(
            Premise(
                "nvidia-smi found in /usr/lib/wsl/lib",
                lambda s: s.driver.nvidia_smi_path.value == "/usr/lib/wsl/lib/nvidia-smi",
            ),
            Premise(
                "utilization NOT_SUPPORTED, not zero",
                lambda s: s.runtime[0].gpu_utilization_percent.value is None,
            ),
        ),
    )


def fx12() -> FixtureScenario:
    root = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    wddm = "Under WDDM, per-process GPU memory is not available (NVML documentation)"
    process = GpuProcess(
        pid=4242,
        kind=ProcessKind.COMPUTE,
        used_memory_bytes=failed(
            CollectionStatus.NOT_SUPPORTED,
            "VALUE_NOT_AVAILABLE",
            "NVML_VALUE_NOT_AVAILABLE",
            method="nvmlDeviceGetComputeRunningProcesses_v3",
            unit="bytes",
            note=wddm,
        ),
        name=ok("python.exe", Source.OS, method="nvmlSystemGetProcessName"),
    )
    snapshot = _nvml_platform_snapshot(
        PlatformProfile.WINDOWS,
        pci=pci_not_supported(None),
        # DEC-47: `CUDA_PATH_V12_4` and the active nvcc also point at this root, and a
        # Windows nvcc is nvcc.exe; the fixture omitted both signals and the suffix.
        cuda_env=cuda_environment(
            root=root,
            discovered_via=(
                ToolkitDiscovery.CUDA_PATH,
                ToolkitDiscovery.CUDA_PATH_VERSIONED,
                ToolkitDiscovery.NVCC_ON_PATH,
            ),
            env={"CUDA_PATH": root, "CUDA_PATH_V12_4": root},
            nvcc_name="nvcc.exe",
        ),
        smi_path="C:/Windows/System32/nvidia-smi.exe",
        processes=ok((process,), method="nvmlDeviceGetComputeRunningProcesses_v3"),
    )
    snapshot = replace(
        snapshot,
        unsupported_capabilities=(
            *unsupported_capabilities_for(None, PlatformProfile.WINDOWS),
            unsupported("RUNTIME.PROCESS_MEMORY", f"{wddm}.", device_ref="nvml:0"),
        ),
    )
    return FixtureScenario(
        fixture_id="FX12",
        name="Windows WDDM",
        description="Windows; NVML OK; processes listed with used memory NOT_SUPPORTED; PCI not "
        "supported; toolkit 12.4 via CUDA_PATH; PyTorch CUDA available.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            *per_device("D06", ("nvml:0",), S.INFO, "D06 row 4 (WDDM process memory)"),
        ),
        premises=(
            Premise(
                "process listed, used memory NOT_SUPPORTED",
                lambda s: (
                    s.runtime[0].processes.value[0].used_memory_bytes.status
                    is CollectionStatus.NOT_SUPPORTED
                ),
            ),
            Premise(
                "toolkit discovered via CUDA_PATH, CUDA_PATH_V12_4 and nvcc on PATH (DEC-47)",
                lambda s: (
                    s.cuda_environment.toolkits.value[0].discovered_via
                    == (
                        ToolkitDiscovery.CUDA_PATH,
                        ToolkitDiscovery.CUDA_PATH_VERSIONED,
                        ToolkitDiscovery.NVCC_ON_PATH,
                    )
                ),
            ),
        ),
    )


def fx13() -> FixtureScenario:
    unsupported_nvidia = failed(
        CollectionStatus.NOT_SUPPORTED,
        "PLATFORM_NOT_SUPPORTED",
        "NVIDIA CUDA GPUs are not supported on macOS",
        method="platform gate",
    )
    system = system_info(PlatformProfile.MACOS)
    snapshot = InspectionSnapshot(
        system=system,
        gpu=no_inventory(unsupported_nvidia, pci_not_supported(None)),
        driver=driver_without_nvml(unsupported_nvidia, smi_status=CollectionStatus.NOT_SUPPORTED),
        cuda_environment=cuda_environment(platform_supported=False),
        frameworks=FrameworkInfo(pytorch=pytorch_cpu_only("2.4.1")),
        runtime=(),
        collection_errors=(),
        unsupported_capabilities=(
            unsupported("CUDA.TOOLKIT", "The CUDA Toolkit is not available on macOS."),
            *unsupported_capabilities_for(None, PlatformProfile.MACOS),  # GPU.PCI_SCAN, NVIDIA.ALL
            # The system collector's record: the field's own reason (architecture §4.3).
            unsupported("SYSTEM.MEMORY_AVAILABLE", system.memory_available_bytes.error.message),
        ),
        collector_runs=collector_runs(),
    )
    return FixtureScenario(
        fixture_id="FX13",
        name="macOS",
        description="Darwin arm64; NVIDIA functionality not supported; PyTorch installed as a "
        "CPU/MPS build with CUDA unavailable.",
        snapshot=snapshot,
        expected_diagnostics=no_gpu(
            expect("D01", S.INFO, "D01 row 7 (macOS: NVIDIA GPU observed absent)"),
            expect("D04", S.INFO, "D04 row 3 (macOS)"),
            expect(
                "D05", S.PASS, "D05 (binding present; nvidia-smi unsupported, NVML unavailable)"
            ),
        ),
        readiness_runs=(
            readiness_run(
                "FX13 (B)", 8, R.PLATFORM_NOT_SUPPORTED, d02_confidence=Confidence.MEDIUM
            ),
        ),
        premises=(
            Premise("macOS profile", lambda s: s.system.platform_profile is PlatformProfile.MACOS),
            Premise(
                "available memory NOT_SUPPORTED, not zero",
                lambda s: s.system.memory_available_bytes.status is CollectionStatus.NOT_SUPPORTED,
            ),
        ),
    )


def fx14() -> FixtureScenario:
    util = failed(
        CollectionStatus.NOT_SUPPORTED,
        "NVML_ERROR_NOT_SUPPORTED",
        "Not Supported",
        method="nvmlDeviceGetUtilizationRates",
        unit="percent",
        note="GPU-level utilization is not reported while MIG is enabled",
    )
    device = gpu_device(0, GPU_M, mig_mode="ENABLED")
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=nvml_inventory((device,), pci_scan(pci_device(1))),
        driver=driver_nvml(),
        cuda_environment=cuda_environment(),
        frameworks=FrameworkInfo(
            pytorch=pytorch_cuda([torch_device(0, GPU_M, correlated_ref="nvml:0")])
        ),
        runtime=(
            runtime_metrics(
                "nvml:0",
                GPU_M.total_bytes,
                40 * GIB,
                gpu_utilization_percent=util,
                memory_bus_utilization_percent=util,
            ),
        ),
        collection_errors=(),
        unsupported_capabilities=(
            unsupported(
                "READINESS.MIG", "MIG instances are not enumerated in V1.", device_ref="nvml:0"
            ),
            unsupported(
                "RUNTIME.GPU_UTILIZATION", "Not reported while MIG is enabled.", device_ref="nvml:0"
            ),
        ),
        collector_runs=collector_runs(),
    )
    run = readiness_run(
        "FX14", 8, R.SOME_DEVICES_UNKNOWN, [device_expectation("nvml:0", R.MIG_ENABLED_UNSUPPORTED)]
    )
    return FixtureScenario(
        fixture_id="FX14",
        name="MIG enabled",
        description="Mode B, --required-vram 8; one 80 GiB device with MIG enabled, 40 GiB free, "
        "aggregate memory collected.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            run.d02,
            *per_device("D06", ("nvml:0",), S.INFO, "D06 row 4 (utilization not supported)"),
        ),
        readiness_runs=(run,),
        mode_b_only=True,
        premises=(
            Premise("MIG enabled", lambda s: s.gpu.devices[0].mig_mode.value == "ENABLED"),
            Premise(
                "aggregate free memory collected",
                lambda s: s.runtime[0].memory_free_bytes.value == 40 * GIB,
            ),
        ),
        limitations=(
            "Assumption: utilization is the NOT_SUPPORTED field behind D06 = INFO; the pinned "
            "inputs only require aggregate memory to be collected.",
        ),
    )


def fx15() -> FixtureScenario:
    message = "OSError: libcudart.so.12: cannot open shared object file"
    base = single_gpu_snapshot(pytorch=pytorch_probe_failed("IMPORT_ERROR", message))
    snapshot = replace(
        base,
        collection_errors=(
            collection_error("pytorch", "IMPORT_ERROR", message, report_path="/frameworks/pytorch"),
        ),
        collector_runs=collector_runs(pytorch=CollectorRunStatus.PARTIAL),
    )
    return FixtureScenario(
        fixture_id="FX15",
        name="PyTorch import fails",
        description="As FX01, but the torch distribution is present and importing it fails.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect("D01", S.FAIL, "D01 row 2 (installed but not importable)"),
        ),
        premises=(
            Premise(
                "distribution present",
                lambda s: s.frameworks.pytorch.distribution_version.is_collected,
            ),
            Premise(
                "import error recorded",
                lambda s: s.frameworks.pytorch.version.error.code == "IMPORT_ERROR",
            ),
        ),
    )


def fx16() -> FixtureScenario:
    message = "PyTorch probe exceeded pytorch_timeout_s (60.0 s)"
    base = single_gpu_snapshot(pytorch=pytorch_probe_failed("TIMEOUT", message))
    snapshot = replace(
        base,
        collection_errors=(
            collection_error("pytorch", "TIMEOUT", message, report_path="/frameworks/pytorch"),
        ),
        collector_runs=collector_runs(pytorch=CollectorRunStatus.PARTIAL),
    )
    return FixtureScenario(
        fixture_id="FX16",
        name="PyTorch probe timeout",
        description="As FX01, but the PyTorch probe exceeds pytorch_timeout_s.",
        snapshot=snapshot,
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect(
                "D01", S.UNKNOWN, "D01 row 3 (probe timeout)", confidence=Confidence.NOT_APPLICABLE
            ),
        ),
        premises=(
            Premise(
                "timeout recorded",
                lambda s: s.frameworks.pytorch.cuda_available.error.code == "TIMEOUT",
            ),
        ),
    )


def fx17() -> FixtureScenario:
    not_loaded = nvml_failure(
        CollectionStatus.NOT_DETECTED, "NVML_ERROR_DRIVER_NOT_LOADED", "Driver Not Loaded"
    )
    snapshot = InspectionSnapshot(
        system=system_info(PlatformProfile.LINUX),
        gpu=no_inventory(not_loaded, pci_scan(pci_device(1, driver="nouveau"))),
        driver=driver_without_nvml(not_loaded),
        cuda_environment=cuda_environment(toolkit_version=None),
        frameworks=FrameworkInfo(pytorch=pytorch_not_installed()),
        runtime=(),
        collection_errors=(),
        unsupported_capabilities=(),
        collector_runs=collector_runs(),
    )
    return FixtureScenario(
        fixture_id="FX17",
        name="NVIDIA on PCI, driver not loaded",
        description="Linux; PCI shows an NVIDIA device bound to nouveau; NVML driver not "
        "loaded; no nvidia-smi; PyTorch not installed.",
        snapshot=snapshot,
        expected_diagnostics=no_gpu(
            expect("D01", S.NOT_APPLICABLE, "D01 row 1 (PyTorch not installed)"),
            expect("D04", S.FAIL, "D04 row 4 (NVIDIA on PCI, no driver interface)"),
            expect("D05", S.INFO, "D05 PyTorch not installed"),
        ),
        premises=(
            Premise(
                "NVIDIA PCI device bound to nouveau",
                lambda s: (
                    s.gpu.pci_observations.value[0].is_nvidia
                    and s.gpu.pci_observations.value[0].kernel_driver.value == "nouveau"
                ),
            ),
        ),
    )


def fx18() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX18",
        name="Hot GPU",
        description="As FX01, temperature 92 °C with a device-reported slowdown threshold of "
        "90 °C.",
        snapshot=single_gpu_snapshot(
            temperature_gpu_celsius=ok(92, unit="celsius", method="nvmlDeviceGetTemperatureV"),
        ),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            *per_device(
                "D07",
                ("nvml:0",),
                S.WARNING,
                "D07 row 2 (at or above slowdown threshold)",
                confidence=Confidence.MEDIUM,
            ),
        ),
        premises=(
            Premise(
                "temperature >= threshold",
                lambda s: (
                    s.runtime[0].temperature_gpu_celsius.value
                    >= s.runtime[0].temperature_slowdown_threshold_celsius.value
                ),
            ),
        ),
        limitations=("A single temperature sample can't establish throttling.",),
    )


def fx19() -> FixtureScenario:
    run = readiness_run(
        "FX19",
        8,
        R.AT_LEAST_ONE_DEVICE_READY,
        [device_expectation("nvml:0", R.SUFFICIENT_FREE_MEMORY)],
        margins=[("nvml:0", 0)],
    )
    return FixtureScenario(
        fixture_id="FX19",
        name="Readiness boundary",
        description="Mode B, --required-vram 8; free memory exactly 8589934592 bytes.",
        snapshot=single_gpu_snapshot(free=8 * GIB),
        expected_diagnostics=replace_checks(healthy(("nvml:0",)), run.d02),
        readiness_runs=(run,),
        mode_b_only=True,
        premises=(
            Premise(
                "free == required", lambda s: s.runtime[0].memory_free_bytes.value == 8589934592
            ),
        ),
    )


def fx20() -> FixtureScenario:
    return FixtureScenario(
        fixture_id="FX20",
        name="pynvml distribution",
        description="As FX01, but module pynvml is provided by the `pynvml` distribution.",
        snapshot=single_gpu_snapshot(binding_distribution="pynvml"),
        expected_diagnostics=replace_checks(
            healthy(("nvml:0",)),
            expect(
                "D05", S.INFO, "D05 non-recommended binding distribution", severity=Severity.LOW
            ),
        ),
        premises=(
            Premise(
                "binding distribution is pynvml",
                lambda s: s.driver.nvml_binding_distribution.value == "pynvml",
            ),
        ),
    )


PLATFORM_FIXTURES = (fx11, fx12, fx13, fx14, fx15, fx16, fx17, fx18, fx19, fx20)
