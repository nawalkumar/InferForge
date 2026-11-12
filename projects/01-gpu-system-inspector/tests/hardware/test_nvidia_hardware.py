"""Hardware tests HW-01 to HW-13 (testing strategy §6). Excluded from the ordinary suite.

    uv run pytest -m hardware tests/hardware -rA          # in the workspace
    python -m pytest -m hardware tests/hardware -rA       # any environment with the package

Each test checks its preconditions on the machine it runs on and skips with the reason
when they don't hold. **A skipped hardware test is not a pass**, and nothing here marks a
tracker item `HARDWARE_VALIDATED`: results are recorded in `docs/app_tracker.md` §6 by the
owner, from the evidence bundle (`validation/hardware_evidence.py`, see
`docs/hardware_validation.md`).

The inspection runs through the real CLI in a child process, so this pytest process never
imports PyTorch or initialises CUDA. Reference values come from `nvidia-smi` and `nvcc`,
run directly. Only HW-12 allocates GPU memory, in a separate child process, and only when
`GSI_HW_ALLOW_CUDA_ALLOCATION=1` is set; it is never part of an inspection.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest

from gpu_system_inspector.models import (
    CollectionStatus,
    DiagnosticStatus,
    InspectionReport,
    LinuxEnvironment,
    PlatformProfile,
    ReadinessStatus,
    Source,
)
from gpu_system_inspector.reporting import parse_report_json

pytestmark = pytest.mark.hardware

CLI = (sys.executable, "-m", "gpu_system_inspector")
CLI_TIMEOUT_S = 300
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
SMI_QUERY = "index,name,uuid,pci.bus_id,driver_version,memory.total"
MIB = 1 << 20
GIB = 1 << 30
ALLOW_ALLOCATION = "GSI_HW_ALLOW_CUDA_ALLOCATION"
NFR11_NO_DRIVER_S = 2.0
NFR11_NVML_S = 3.0
NVCC_RELEASE = re.compile(r"release ([0-9]+\.[0-9]+), V([0-9]+(?:\.[0-9]+)+)")


@dataclass(frozen=True)
class CliRun:
    exit_code: int
    report: InspectionReport
    stderr: str
    seconds: float


def run_cli(*args: str, env: dict[str, str] | None = None) -> CliRun:
    started = time.monotonic()
    result = subprocess.run(
        [*CLI, "inspect", "--json", "-", *args],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
        env={**os.environ, **(env or {})},
    )
    seconds = time.monotonic() - started
    assert "Traceback" not in result.stderr, result.stderr
    assert result.stdout, f"no JSON on stdout (exit {result.returncode}): {result.stderr}"
    return CliRun(result.returncode, parse_report_json(result.stdout), result.stderr, seconds)


@pytest.fixture(scope="module")
def live() -> CliRun:
    return run_cli()


def nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    return WSL_NVIDIA_SMI if os.path.exists(WSL_NVIDIA_SMI) else None


def require_nvml_devices(report: InspectionReport, minimum: int = 1) -> None:
    nvml, count = report.driver.nvml_version, report.gpu.device_count
    if not nvml.is_collected:
        code = nvml.error.code if nvml.error else nvml.status
        pytest.skip(f"NVML unavailable: {code}")
    if report.gpu.inventory_source is not Source.NVML:
        pytest.skip(f"inventory source is {report.gpu.inventory_source}, not NVML")
    if not count.is_collected or count.value < minimum:
        pytest.skip(f"needs at least {minimum} NVML device(s); found {count.value}")


def require_pytorch_cuda(report: InspectionReport) -> None:
    pytorch = report.frameworks.pytorch
    if pytorch.status is not CollectionStatus.COLLECTED:
        pytest.skip(f"PyTorch section is {pytorch.status} in this Python environment")
    if pytorch.cuda_available.value is not True:
        pytest.skip(f"PyTorch reports cuda_available = {pytorch.cuda_available.value}")


def smi_rows(executable: str) -> list[list[str]]:
    result = subprocess.run(
        [executable, f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return [[c.strip() for c in line.split(",")] for line in result.stdout.splitlines() if line]


def finding(report: InspectionReport, check_id: str):
    return [f for f in report.diagnostics if f.check_id == check_id]


def observed_fields(record: object) -> dict[str, object]:
    return {
        name: getattr(record, name)
        for name in record.__dataclass_fields__
        if hasattr(getattr(record, name), "status")
    }


# --- HW-01 NVML vs nvidia-smi --------------------------------------------------------------


def test_hw01_nvml_inventory_matches_nvidia_smi(live, record_property):
    report = live.report
    require_nvml_devices(report)
    executable = nvidia_smi()
    if executable is None:
        pytest.skip("nvidia-smi not found on PATH or /usr/lib/wsl/lib")
    rows = smi_rows(executable)
    record_property("nvidia_smi_rows", json.dumps(rows))
    assert len(rows) == report.gpu.device_count.value
    for row, device in zip(rows, report.gpu.devices, strict=True):
        index, name, uuid, bus_id, driver, memory_mib = row
        assert int(index) == device.nvml_index
        assert device.name.value == name
        assert device.uuid.value == uuid
        assert device.pci_bus_id.value.lower() == bus_id.lower()
        assert report.driver.driver_version.value == driver
        assert abs(device.memory_total_bytes.value - int(memory_mib) * MIB) <= MIB


# --- HW-02 PCI scan on native Linux ------------------------------------------------------


def test_hw02_pci_scan_sees_the_nvidia_device(live, record_property):
    report = live.report
    if report.system.linux_environment.value is not LinuxEnvironment.NATIVE_LINUX:
        pytest.skip(f"linux_environment is {report.system.linux_environment.value}")
    require_nvml_devices(report)
    pci = report.gpu.pci_observations
    record_property("pci_status", str(pci.status))
    assert pci.is_collected, f"PCI scan {pci.status}: {pci.error}"
    nvidia = [o for o in pci.value if o.is_nvidia]
    record_property("nvidia_pci", json.dumps([[o.address, o.kernel_driver.value] for o in nvidia]))
    assert any(o.kernel_driver.value == "nvidia" for o in nvidia)


# --- HW-03 Runtime snapshot sanity -------------------------------------------------------


def test_hw03_runtime_snapshot_is_plausible(live, record_property):
    report = live.report
    require_nvml_devices(report)
    for m in report.runtime:
        statuses = {name: str(o.status) for name, o in observed_fields(m).items()}
        record_property(f"runtime_{m.device_ref}", json.dumps(statuses))
        assert "UNKNOWN" not in statuses.values(), statuses
        for pct in (m.gpu_utilization_percent, m.memory_bus_utilization_percent):
            assert not pct.is_collected or 0 <= pct.value <= 100
        total = m.memory_total_bytes.value
        if total is not None:
            for part in (m.memory_used_bytes, m.memory_free_bytes):
                assert not part.is_collected or part.value <= total
        temp = m.temperature_gpu_celsius
        assert not temp.is_collected or 0 <= temp.value <= 120
        power = m.power_draw_watts
        assert not power.is_collected or power.value >= 0


# --- HW-04 PyTorch probe and isolation ----------------------------------------------------

_ISOLATION_SCRIPT = """
import json, sys
from gpu_system_inspector import api
from gpu_system_inspector.reporting import render_json
report = api.inspect()
print(json.dumps({"torch_imported": "torch" in sys.modules, "report": render_json(report)}))
"""


def test_hw04_pytorch_probe_runs_outside_the_inspector(live, record_property):
    require_pytorch_cuda(live.report)
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    report = parse_report_json(payload["report"])
    assert payload["torch_imported"] is False
    pytorch = report.frameworks.pytorch
    assert pytorch.probe.mode.value == "SUBPROCESS"
    assert pytorch.cuda_available.value is True
    assert pytorch.devices, "PyTorch reports CUDA available but lists no device"
    record_property(
        "correlation",
        json.dumps({d.torch_index: d.correlated_device_ref.value for d in pytorch.devices}),
    )
    if report.gpu.inventory_source is Source.NVML:
        for device in pytorch.devices:
            assert device.correlated_device_ref.is_collected, device


# --- HW-05 Toolkit detection --------------------------------------------------------------


def _nvcc_release(nvcc: str) -> tuple[str, str]:
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60)
    match = NVCC_RELEASE.search(result.stdout)
    assert match, result.stdout
    return match.group(1), match.group(2)


def test_hw05_toolkit_version_matches_nvcc(live, record_property):
    cuda = live.report.cuda_environment
    if not cuda.active_nvcc_path.is_collected and not cuda.toolkits.is_collected:
        pytest.skip("no CUDA Toolkit or nvcc detected")
    if cuda.active_nvcc_path.is_collected:
        release, full = _nvcc_release(cuda.active_nvcc_path.value)
        record_property("active_nvcc", json.dumps([release, full]))
        assert cuda.active_nvcc_version.value in (full, release)
    for toolkit in cuda.toolkits.value or ():
        if not (toolkit.version.is_collected and toolkit.nvcc_path.is_collected):
            continue
        release, full = _nvcc_release(toolkit.nvcc_path.value)
        detected = toolkit.version.value
        # C12: whether version.json's full string equals nvcc's is the assumption under test;
        # both are recorded, and the release (major.minor) D03 relies on must agree.
        record_property(
            f"toolkit_{toolkit.root_path}",
            json.dumps({"detected": detected, "method": toolkit.version.method, "nvcc": full}),
        )
        assert ".".join(detected.split(".")[:2]) == release


# --- HW-06 PyTorch without a local toolkit ------------------------------------------------


def test_hw06_pytorch_works_without_a_local_toolkit(live):
    report = live.report
    require_pytorch_cuda(report)
    cuda = report.cuda_environment
    if cuda.toolkits.is_collected or cuda.active_nvcc_path.is_collected:
        pytest.skip("a CUDA Toolkit is installed on this machine")
    [d03] = finding(report, "D03")
    text = " ".join([d03.explanation, *(e.statement for e in d03.evidence)])
    assert "not to run PyTorch binary packages" in text


# --- HW-07 WSL2 with the Windows NVIDIA driver --------------------------------------------


def test_hw07_wsl2_reports_unsupported_fields_as_unsupported(live, record_property):
    report = live.report
    if report.system.platform_profile is not PlatformProfile.WSL2:
        pytest.skip(f"platform is {report.system.platform_profile}, not WSL2")
    require_nvml_devices(report)
    assert report.driver.nvidia_smi_path.is_collected
    record_property("nvidia_smi_path", report.driver.nvidia_smi_path.value)
    for m in report.runtime:
        for name, o in observed_fields(m).items():
            assert o.status not in (CollectionStatus.COLLECTION_FAILED, CollectionStatus.UNKNOWN), (
                m.device_ref,
                name,
                o,
            )


# --- HW-08 Windows WDDM -------------------------------------------------------------------


def test_hw08_windows_wddm_process_memory_is_not_supported(live):
    report = live.report
    if report.system.platform_profile is not PlatformProfile.WINDOWS:
        pytest.skip(f"platform is {report.system.platform_profile}, not Windows")
    require_nvml_devices(report)
    processes = [p for m in report.runtime for p in (m.processes.value or ())]
    if not processes:
        pytest.skip("no GPU process listed, so per-process memory can't be checked")
    for p in processes:
        assert p.used_memory_bytes.status is CollectionStatus.NOT_SUPPORTED, p


# --- HW-09 Multi-GPU ordering -------------------------------------------------------------


def _assert_correlated(report: InspectionReport) -> dict[int, str]:
    nvml = {d.device_ref: d for d in report.gpu.devices}
    mapping = {}
    for device in report.frameworks.pytorch.devices:
        assert device.correlated_device_ref.is_collected, device
        ref = device.correlated_device_ref.value
        if device.uuid.is_collected and nvml[ref].uuid.is_collected:
            assert device.uuid.value.lower().removeprefix("gpu-") == (
                nvml[ref].uuid.value.lower().removeprefix("gpu-")
            )
        mapping[device.torch_index] = ref
    return mapping


def test_hw09_correlation_holds_under_every_device_order(live, record_property):
    report = live.report
    require_nvml_devices(report, minimum=2)
    require_pytorch_cuda(report)
    record_property("default_order", json.dumps(_assert_correlated(report)))
    pci_order = run_cli(env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}).report
    record_property("pci_bus_id_order", json.dumps(_assert_correlated(pci_order)))
    last = report.gpu.devices[-1]
    subset = run_cli(env={"CUDA_VISIBLE_DEVICES": last.uuid.value}).report
    assert _assert_correlated(subset) == {0: last.device_ref}


# --- HW-10 Performance (NFR-11) -----------------------------------------------------------


def test_hw10_general_inspection_wall_time(live, record_property):
    report = live.report
    if report.driver.nvml_version.is_collected:
        if report.gpu.device_count.value is None or report.gpu.device_count.value > 8:
            pytest.skip("NFR-11 has a target only for up to 8 GPUs")
        target = NFR11_NVML_S
    elif report.driver.nvml_version.status is CollectionStatus.NOT_DETECTED:
        target = NFR11_NO_DRIVER_S
    else:
        pytest.skip(f"NVML is {report.driver.nvml_version.status}; no NFR-11 target applies")
    times = [run_cli("--no-pytorch").seconds for _ in range(3)]
    record_property("wall_times_s", json.dumps([round(t, 3) for t in times]))
    record_property("target_s", target)
    assert statistics.median(times) < target


# --- HW-11 macOS --------------------------------------------------------------------------


def test_hw11_macos_reports_general_information_only(live):
    report = live.report
    if report.system.platform_profile is not PlatformProfile.MACOS:
        pytest.skip(f"platform is {report.system.platform_profile}, not macOS")
    assert live.exit_code == 0
    assert report.gpu.device_count.status is CollectionStatus.NOT_SUPPORTED
    assert report.system.os_name.is_collected
    [d04] = finding(report, "D04")
    assert d04.status is DiagnosticStatus.INFO


# --- HW-12 Readiness under real memory pressure (opt-in CUDA allocation) -------------------

_ALLOCATE = """
import sys, torch
x = torch.empty(int(sys.argv[1]), dtype=torch.uint8, device="cuda:0")
torch.cuda.synchronize()
print("ALLOCATED", flush=True)
sys.stdin.readline()
"""


def test_hw12_readiness_follows_real_free_memory(live, record_property):
    if os.environ.get(ALLOW_ALLOCATION) != "1":
        pytest.skip(f"allocates GPU memory; set {ALLOW_ALLOCATION}=1 to run it")
    report = live.report
    require_nvml_devices(report, minimum=1)
    require_pytorch_cuda(report)
    if report.gpu.device_count.value != 1:
        pytest.skip("needs exactly one visible GPU so the allocation targets the assessed device")
    free = report.runtime[0].memory_free_bytes.value
    if free is None or free < 2 * GIB:
        pytest.skip(f"needs at least 2 GiB free on nvml:0; free is {free}")
    required_gib = round(free * 0.5 / GIB, 3)
    before = run_cli("--required-vram", str(required_gib))
    record_property("required_gib", required_gib)
    record_property("before", str(before.report.readiness.overall_status))
    assert before.report.readiness.overall_status is ReadinessStatus.READY
    holder = subprocess.Popen(
        [sys.executable, "-c", _ALLOCATE, str(int(free * 0.75))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ALLOCATED"
        during = run_cli("--required-vram", str(required_gib))
        record_property("during", str(during.report.readiness.overall_status))
        assert during.report.readiness.overall_status is ReadinessStatus.NOT_READY
        assert during.exit_code == 1
    finally:
        holder.communicate("\n", timeout=60)


# --- HW-13 Machine without an NVIDIA driver -----------------------------------------------


def test_hw13_no_driver_machine_degrades_gracefully(live, record_property):
    report = live.report
    if report.driver.nvml_version.is_collected or report.driver.nvidia_smi_path.is_collected:
        pytest.skip("an NVIDIA driver interface is accessible on this machine")
    record_property("platform", str(report.system.platform_profile))
    assert live.exit_code == 0
    [d04] = finding(report, "D04")
    assert d04.status is DiagnosticStatus.INFO
    mode_b = run_cli("--required-vram", "8")
    assert mode_b.exit_code in (1, 3)
    assert mode_b.report.readiness.overall_status is not ReadinessStatus.READY


def test_the_harness_runs_the_cli_of_this_interpreter(record_property):
    result = subprocess.run([*CLI, "version"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.startswith("GPU System Inspector")
    record_property("version", result.stdout.strip().replace("\n", " | "))
