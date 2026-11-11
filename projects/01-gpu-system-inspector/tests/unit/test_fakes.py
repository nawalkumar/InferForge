"""The fake adapters reproduce their fixture exactly, are deterministic, and touch nothing real."""

import ast
import json
from pathlib import Path

import gsi_testkit
import pytest
from gsi_testkit.builders import FIXTURE_METHOD_PREFIX, MIB
from gsi_testkit.fakes import READ_ONLY_FUNCTIONS, FakeAdapterSet, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import AdapterUsageError
from gpu_system_inspector.models import (
    PROBE_PROTOCOL_VERSION,
    CollectionStatus,
    DataOrigin,
    Observed,
    Source,
)

ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)


def _device_reads(device) -> dict[str, Observed]:
    pci, memory, utilization = device.pci_info(), device.memory_info(), device.utilization()
    return {
        "name": device.name(),
        "uuid": device.uuid(),
        "pci_bus_id": pci.bus_id,
        "memory_total": memory.total,
        "memory_free": memory.free,
        "compute_capability": device.compute_capability(),
        "mig_mode": device.mig_mode(),
        "gpu_utilization": utilization.gpu,
        "memory_bus_utilization": utilization.memory_bus,
        "temperature": device.temperature(),
        "slowdown_threshold": device.slowdown_threshold(),
        "power_draw": device.power_draw(),
        "power_limit": device.power_limit(),
        "processes": device.processes(),
    }


@ids
def test_fake_nvml_reproduces_the_fixture(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    snapshot = fixture.snapshot
    with adapters.nvml.open() as session:
        assert session.driver_version() == (
            snapshot.driver.driver_version if adapters.nvml.available else session.init_result
        )
        count = session.device_count()
        if snapshot.gpu.inventory_source is Source.NVML:
            assert count == snapshot.gpu.device_count
            for device in snapshot.gpu.devices:
                handle = session.device(device.nvml_index).value
                assert handle.name() == device.name
                assert handle.uuid() == device.uuid
        else:
            assert not count.is_collected
            assert session.device(0) == session.init_result
    expected_inits = 1 if adapters.nvml.available else 0
    assert adapters.nvml.inits == adapters.nvml.shutdowns == expected_inits
    assert set(adapters.nvml.calls) <= READ_ONLY_FUNCTIONS


@ids
def test_every_fake_reports_fixture_provenance(fixture):
    adapters = FakeAdapterSet.from_scenario(fixture)
    assert adapters.provenance is DataOrigin.FIXTURE
    for fake in (
        adapters.nvml,
        adapters.nvidia_smi,
        adapters.pci,
        adapters.torch_probe,
        adapters.platform,
        adapters.commands,
    ):
        assert fake.provenance is DataOrigin.FIXTURE
    version = adapters.nvidia_smi.version()
    outputs: list[Observed] = [
        adapters.pci.scan().observations,
        adapters.nvidia_smi.locate(),
        version.driver_version,
        version.cuda_version,
        adapters.nvidia_smi.query_gpus(),
        adapters.torch_probe.distribution_version(),
        adapters.commands.which("nvidia-smi"),
        adapters.nvml.binding_version(),
    ]
    for obs in outputs:
        assert obs.method is None or obs.method.startswith(FIXTURE_METHOD_PREFIX), obs


def test_nvml_queries_on_a_healthy_device_cover_every_field():
    fixture = FIXTURES_BY_ID["FX01"]
    runtime = fixture.snapshot.runtime[0]
    with FakeAdapterSet.from_scenario(fixture).nvml.open() as session:
        results = _device_reads(session.device(0).value)
    assert results["memory_free"] == runtime.memory_free_bytes
    assert results["power_draw"] == runtime.power_draw_watts
    assert all(r.is_collected for r in results.values())


def test_nvml_unavailable_returns_the_init_failure_with_each_unit():
    fixture = FIXTURES_BY_ID["FX03"]
    with FakeAdapterSet.from_scenario(fixture).nvml.open() as session:
        count = session.device_count()
        assert count.status is CollectionStatus.NOT_DETECTED
        assert count.error.code == "NVML_ERROR_LIBRARY_NOT_FOUND"
        assert session.cuda_driver_version().value is None


def test_missing_binding_reports_not_installed():
    nvml = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX04a"]).nvml
    assert nvml.binding_version().status is CollectionStatus.NOT_INSTALLED
    with nvml.open() as session:
        assert session.init_result.status is CollectionStatus.NOT_INSTALLED


def test_nvidia_smi_fallback_rows_have_mib_precision():
    fixture = FIXTURES_BY_ID["FX01"]
    rows = FakeAdapterSet.from_scenario(fixture).nvidia_smi.query_gpus()
    assert rows.is_collected and rows.source is Source.NVIDIA_SMI
    (row,) = rows.value
    free = row.values["memory.free"]
    assert free.value % MIB == 0 and free.note == "Converted from MiB; precision 1 MiB"
    assert free.method == "fixture:nvidia-smi --query-gpu=memory.free"


def test_nvidia_smi_missing_returns_not_detected():
    result = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX03"]).nvidia_smi.query_gpus()
    assert result.status is CollectionStatus.NOT_DETECTED


def test_torch_probe_emits_the_documented_protocol():
    probe = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX06"]).torch_fake
    output = probe.subprocess_output(60.0)
    assert output.is_collected and output.value.returncode == 0
    payload = json.loads(output.value.stdout)
    assert payload["probe_protocol"] == PROBE_PROTOCOL_VERSION
    assert payload["import_ok"] is True and payload["cuda_available"] is True
    assert [d["index"] for d in payload["devices"]] == [0, 1]
    assert set(payload) == {
        "probe_protocol",
        "import_ok",
        "import_error",
        "torch_version",
        "cuda_build",
        "build_cuda_version",
        "hip_version",
        "cudnn_version",
        "cuda_available",
        "cuda_init_warnings",
        "device_count",
        "devices",
        "device_error",
    }
    report = probe.probe(60.0)
    assert report.is_collected and [d.index for d in report.value.devices] == [0, 1]


def test_torch_probe_failure_shapes():
    fx16 = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX16"]).torch_fake
    timeout = fx16.subprocess_output(60.0)
    assert timeout.status is CollectionStatus.COLLECTION_FAILED
    assert timeout.value is None and timeout.error.code == "TIMEOUT"
    assert fx16.probe(60.0).error.code == "TIMEOUT"
    fx15 = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX15"]).torch_fake
    payload = json.loads(fx15.subprocess_output(60.0).value.stdout)
    assert payload["import_ok"] is False and "libcudart" in payload["import_error"]
    import_error = fx15.probe(60.0)
    assert import_error.error.code == "IMPORT_ERROR" and "libcudart" in import_error.error.message
    with pytest.raises(AssertionError, match="must not run"):
        FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX05"]).torch_fake.probe(60.0)


def test_platform_serves_only_allowlisted_environment():
    platform = FakeAdapterSet.from_scenario(FIXTURES_BY_ID["FX02c"]).platform
    assert platform.getenv("CUDA_VISIBLE_DEVICES").value == ""
    assert platform.getenv("CUDA_DEVICE_ORDER").status is CollectionStatus.NOT_DETECTED
    with pytest.raises(AdapterUsageError, match="allowlisted"):
        platform.getenv("HOME")
    assert "getenv(HOME)" not in platform.calls


def test_fixed_clock_is_deterministic():
    first, second = FixedClock(), FixedClock()
    assert [first.monotonic_ms() for _ in range(3)] == [second.monotonic_ms() for _ in range(3)]
    assert first.now_utc() == second.now_utc()


@ids
def test_fake_collectors_return_the_fixture_sections(fixture):
    collectors = FakeAdapterSet.from_scenario(fixture).collectors
    for name, collector in collectors.items():
        if name in fixture.collector_failures:
            with pytest.raises(type(fixture.collector_failures[name])):
                collector.collect()
        else:
            assert collector.collect() is not None
        assert collector.calls == 1


FORBIDDEN_IMPORTS = {
    "pynvml",
    "torch",
    "subprocess",
    "socket",
    "ctypes",
    "os",
    "platform",
    "shutil",
    "urllib",
    "http",
}


@pytest.mark.parametrize(
    "path",
    sorted(Path(gsi_testkit.__file__).parent.rglob("*.py")),
    ids=lambda p: str(p.relative_to(Path(gsi_testkit.__file__).parent)),
)
def test_test_kit_never_touches_hardware_processes_or_network(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & FORBIDDEN_IMPORTS
