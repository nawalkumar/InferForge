"""Host integration (IMP-K08/K02/P01, IMP-A01/A02/A05): the public `api.inspect()` with the
production adapter set on the machine running the tests. Read-only, no NVIDIA hardware
needed: on a machine without NVIDIA, the NVIDIA sections must say so honestly.

A guard fails the test if anything under `/sys` is opened, listed or resolved, and the only
processes the inspection may start are `nvidia-smi`, `nvcc` (when installed) and the PyTorch
probe child (when PyTorch is installed). NVML is initialised
for real if its library is present. A native Linux host is skipped: its PCI scan reads the
real sysfs, which is hardware validation (HW-02), not part of this suite.
"""

import builtins
import json
import os
import subprocess
import sys

import jsonschema
import pytest

from gpu_system_inspector import api
from gpu_system_inspector.adapters import OsPlatformAdapter, torch_probe_argv
from gpu_system_inspector.adapters.outcomes import NOT_IMPLEMENTED
from gpu_system_inspector.adapters.pci_sysfs import PCI_SCAN_UNSUPPORTED_REASONS, pci_scan_eligible
from gpu_system_inspector.collectors import GPU_PCI_SCAN
from gpu_system_inspector.core import NOT_IMPLEMENTED_LIMITATION, PRODUCTION_LIMITATIONS
from gpu_system_inspector.models import (
    ENVIRONMENT_ALLOWLIST,
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    CollectorRunStatus,
    DataOrigin,
    InspectionMode,
    LinuxEnvironment,
    PlatformProfile,
    Source,
)
from gpu_system_inspector.reporting import load_report_schema, parse_report_json, render_json

pytestmark = pytest.mark.host


@pytest.fixture
def guarded(monkeypatch) -> list[str]:
    attempts: list[str] = []

    def guard(function):
        def guarded_call(path, *args, **kwargs):
            if os.fspath(path).startswith("/sys"):
                attempts.append(os.fspath(path))
                raise AssertionError(f"the inspection touched {path}")
            return function(path, *args, **kwargs)

        return guarded_call

    real_popen = subprocess.Popen

    def only_nvidia_smi(argv, *args, **kwargs):
        program = os.path.basename(argv[0]) if isinstance(argv, (list, tuple)) and argv else argv
        probe = isinstance(argv, (list, tuple)) and tuple(argv) == torch_probe_argv(sys.executable)
        if (program not in ("nvidia-smi", "nvcc") and not probe) or kwargs.get("shell"):
            raise AssertionError(f"the inspection started {argv!r}")
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(os, "listdir", guard(os.listdir))
    monkeypatch.setattr(os, "readlink", guard(os.readlink))
    monkeypatch.setattr(subprocess, "Popen", only_nvidia_smi)
    return attempts


@pytest.fixture
def host_report(guarded):
    environment = OsPlatformAdapter().system_info().linux_environment.value
    if pci_scan_eligible(environment):
        pytest.skip("native Linux host: the real PCI scan is hardware validation (HW-02)")
    return api.inspect(), guarded


def test_the_public_api_produces_a_valid_live_report(host_report):
    report, attempts = host_report
    assert attempts == []
    text = render_json(report)
    jsonschema.validate(json.loads(text), load_report_schema())
    assert parse_report_json(text) == report
    assert report.data_origin is DataOrigin.LIVE and report.mode is InspectionMode.GENERAL
    assert FIXTURE_METHOD_PREFIX not in text
    assert NOT_IMPLEMENTED_LIMITATION in report.limitations
    assert set(PRODUCTION_LIMITATIONS) <= set(report.limitations)


def test_collectors_run_in_order_without_failures(host_report):
    report, _ = host_report
    assert [r.collector for r in report.collector_runs] == [
        "system",
        "pci",
        "nvml",
        "driver",
        "gpu_hardware",
        "runtime",
        "nvidia_smi",
        "cuda_environment",
        "pytorch",
        "correlation",
    ]
    statuses = {r.collector: r.status for r in report.collector_runs}
    assert CollectorRunStatus.FAILED not in statuses.values()
    assert statuses["pci"] is CollectorRunStatus.COMPLETED


def test_pci_is_gated_and_reported_once(host_report):
    report, _ = host_report
    environment = report.system.linux_environment.value
    pci = report.gpu.pci_observations
    assert pci.status is CollectionStatus.NOT_SUPPORTED
    assert pci.error.message == PCI_SCAN_UNSUPPORTED_REASONS[environment]
    records = [u for u in report.unsupported_capabilities if u.capability == GPU_PCI_SCAN]
    assert len(records) == 1 and records[0].reason == pci.error.message
    if environment is LinuxEnvironment.WSL2:
        assert report.system.platform_profile is PlatformProfile.WSL2


def test_the_gpu_inventory_is_real_or_honestly_unavailable(host_report):
    """Whatever this machine has, the inventory comes from NVML or nvidia-smi, or says why
    neither could provide it; it's never a stand-in and never a fabricated zero."""
    report, _ = host_report
    gpu = report.gpu
    count = gpu.device_count
    assert count.error is None or count.error.code != NOT_IMPLEMENTED
    if count.is_collected:
        assert gpu.inventory_source in (Source.NVML, Source.NVIDIA_SMI)
        assert count.value == len(gpu.devices)
    else:
        assert gpu.devices == () and gpu.inventory_source is None and count.value is None
        assert count.source is Source.NVML  # the NVML outcome, e.g. NVML_ERROR_LIBRARY_NOT_FOUND
        assert count.error.code.startswith("NVML_ERROR_") or count.error.code == "ImportError"
    nvml_errors = [e for e in report.collection_errors if e.collector == "nvml"]
    assert all(e.status is not CollectionStatus.COLLECTED for e in nvml_errors)


def test_driver_and_runtime_sections_are_real_or_honestly_unavailable(host_report):
    """IMP-K03/K04: real evidence or the tool's own reason; never a stand-in, never a
    toolkit version in the driver section, one runtime record per inventoried device."""
    report, _ = host_report
    driver = report.driver
    for name in ("nvml_version", "driver_version", "nvidia_smi_path"):
        value = getattr(driver, name)
        assert value.is_collected or value.error.code != NOT_IMPLEMENTED
    assert driver.nvml_binding.is_collected  # the required nvidia-ml-py is installed
    supported = driver.driver_supported_cuda_version
    assert not supported.is_collected or "Not the installed CUDA Toolkit" in supported.note
    if not driver.nvml_version.is_collected and not driver.nvidia_smi_path.is_collected:
        assert not driver.driver_version.is_collected  # nothing could have reported it
    assert [m.device_ref for m in report.runtime] == list(report.gpu.device_refs)


def test_the_cuda_environment_is_real_or_honestly_absent(host_report):
    """IMP-K05/A06: toolkit evidence from the real filesystem and PATH, or NOT_DETECTED;
    never a version taken from the driver, never a stand-in."""
    report, _ = host_report
    cuda = report.cuda_environment
    assert cuda.toolkits.error is None or cuda.toolkits.error.code != NOT_IMPLEMENTED
    if cuda.toolkits.is_collected:
        for candidate in cuda.toolkits.value:
            assert candidate.discovered_via and candidate.root_path.startswith("/")
            method = candidate.version.method
            assert not candidate.version.is_collected or method in (
                "version.json:cuda.version",
                "nvcc --version",
            )
    else:
        assert cuda.toolkits.status is CollectionStatus.NOT_DETECTED
        assert cuda.toolkits.error.code == "NO_TOOLKIT_FOUND"
    assert cuda.active_nvcc_path.is_collected or cuda.active_nvcc_path.error.code in (
        "FileNotFoundError",
        "PermissionError",
    )
    assert sorted(cuda.environment) == sorted(ENVIRONMENT_ALLOWLIST)
    assert "CUDA.TOOLKIT" not in [u.capability for u in report.unsupported_capabilities]


def test_the_pytorch_section_is_real_or_honestly_not_installed(host_report):
    """IMP-A07/K06: the inspector's own interpreter, probed in a subprocess; the main
    process never imports torch; no allocator numbers from the CLI-style probe."""
    report, _ = host_report
    pytorch = report.frameworks.pytorch
    assert "torch" not in sys.modules
    assert pytorch.status in (CollectionStatus.NOT_INSTALLED, CollectionStatus.COLLECTED)
    if pytorch.status is CollectionStatus.NOT_INSTALLED:
        assert pytorch.distribution_version.error.code == "PackageNotFoundError"
        assert pytorch.probe.duration_ms is None  # no probe process was started
    else:
        assert pytorch.allocator.status is CollectionStatus.NOT_APPLICABLE
        assert all(
            d.correlated_device_ref.status is not CollectionStatus.NOT_APPLICABLE
            or (report.gpu.inventory_source is not Source.NVML)
            for d in pytorch.devices
        )
    assert {r.collector: r.status for r in report.collector_runs}["pytorch"] in (
        CollectorRunStatus.COMPLETED,
        CollectorRunStatus.PARTIAL,
    )
