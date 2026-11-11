"""IMP-A07 PyTorch probes (DEC-48): the probe function over `StubTorch`, the probe child in
a real subprocess with a stub `torch` package on its path, `SubprocessTorchProbe` over the
fake runner and `InProcessTorchProbe` over the stub. The real PyTorch is never imported and
no GPU is touched."""

import errno
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from gsi_testkit.fakes import (
    FakeCommandRunner,
    StubDevice,
    StubTorch,
    write_stub_torch_package,
)

import gpu_system_inspector
from gpu_system_inspector.adapters import (
    TORCH_PROBE_MODULE,
    InProcessTorchProbe,
    SubprocessCommandRunner,
    SubprocessTorchProbe,
    decode_probe_payload,
    torch_probe_argv,
)
from gpu_system_inspector.adapters.torch_probe_payload import import_failure, probe_payload
from gpu_system_inspector.models import (
    PROBE_PROTOCOL_VERSION,
    CollectionStatus,
    DataOrigin,
    Source,
)

S = CollectionStatus
PYTHON = "/opt/synthetic/venv/bin/python"
ARGV = torch_probe_argv(PYTHON)
SRC = str(Path(gpu_system_inspector.__file__).parent.parent)


def _decoded(payload):
    return decode_probe_payload(payload, source=Source.PYTORCH, method="test")


def _runner(**response):
    return FakeCommandRunner(files=frozenset({PYTHON})).on(*ARGV, **response)


def _version(name):
    return "2.4.1"


# --- the probe function -------------------------------------------------------------------


def test_a_cuda_build_with_devices_reports_build_runtime_and_devices_separately():
    torch = StubTorch(devices=[StubDevice(), StubDevice(name="GPU B", pci=(0, 2, 0))])
    report = _decoded(probe_payload(torch)).value
    assert (report.torch_version, report.cuda_build, report.build_cuda_version) == (
        "2.4.1+cu124",
        True,
        "12.4",
    )
    assert (report.cudnn_version, report.hip_version) == ("90100", None)
    assert report.cuda_available is True and report.device_count == 2
    first, second = report.devices
    assert (first.name, first.capability, first.total_memory) == (
        "Synthetic GPU A (stub)",
        "8.9",
        24 * 2**30,
    )
    assert first.uuid == "aaaaaaaa-0000-4000-8000-00000000000a"
    assert (first.pci_bus_id, second.pci_bus_id) == ("0000:01:00.0", "0000:02:00.0")
    assert report.device_error is None


def test_a_cpu_only_build_is_not_a_cuda_build_and_reads_no_device():
    torch = StubTorch.cpu_only()
    report = _decoded(probe_payload(torch)).value
    assert report.cuda_build is False and report.build_cuda_version is None
    assert report.cudnn_version is None and report.cuda_available is False
    assert report.device_count == 0 and report.devices == ()
    assert not [c for c in torch.calls if "get_device" in c]


def test_a_rocm_build_is_flagged_and_not_a_cuda_build():
    report = _decoded(
        probe_payload(StubTorch(cuda_version=None, hip_version="6.1", cudnn=None, devices=[]))
    ).value
    assert (report.cuda_build, report.hip_version) == (False, "6.1")


def test_warnings_during_is_available_are_captured_not_raised():
    message = "CUDA initialization: Unexpected error from cudaGetDeviceCount()"
    torch = StubTorch(available=False, devices=[], init_warnings=(message,))
    report = _decoded(probe_payload(torch)).value
    assert report.cuda_available is False and report.cuda_init_warnings == (message,)


def test_attributes_an_older_pytorch_lacks_are_null():
    torch = StubTorch(devices=[StubDevice(uuid=None, pci=None)])
    (device,) = _decoded(probe_payload(torch)).value.devices
    assert (device.uuid, device.pci_bus_id) == (None, None)
    assert device.name is not None


@pytest.mark.parametrize("available", [True, False], ids=["available", "unavailable"])
def test_a_device_enumeration_failure_keeps_the_build_facts(available):
    torch = StubTorch(available=available, device_failure=RuntimeError("CUDA error: no device"))
    report = _decoded(probe_payload(torch)).value
    assert report.devices == () and report.device_error == "RuntimeError: CUDA error: no device"
    assert report.torch_version == "2.4.1+cu124" and report.cuda_build is True
    assert report.device_count == 1  # what device_count() said


def test_an_import_failure_is_a_protocol_document():
    payload = import_failure(OSError("libcudart.so.12: cannot open shared object file"))
    outcome = _decoded(payload)
    assert outcome.error.code == "IMPORT_ERROR"
    assert outcome.error.message == "OSError: libcudart.so.12: cannot open shared object file"
    assert payload["probe_protocol"] == PROBE_PROTOCOL_VERSION


def test_the_probe_allocates_nothing_and_never_asks_for_free_memory():
    torch = StubTorch()
    probe_payload(torch)
    assert not [c for c in torch.calls if "mem_get_info" in c or "empty" in c or "tensor" in c]


# --- the probe child, as a real process ---------------------------------------------------


def _child(tmp_path, monkeypatch, **stub):
    write_stub_torch_package(tmp_path, **stub)
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{SRC}")
    runner = SubprocessCommandRunner()
    return SubprocessTorchProbe(runner, sys.executable, distribution=_version).probe(60.0)


def test_the_child_reports_a_stub_torch_through_the_production_runner(tmp_path, monkeypatch):
    outcome = _child(tmp_path, monkeypatch, stub=StubTorch(init_warnings=("careful",)))
    report = outcome.value
    assert outcome.is_collected and report.torch_version == "2.4.1+cu124"
    assert report.cuda_init_warnings == ("careful",)
    assert [d.pci_bus_id for d in report.devices] == ["0000:01:00.0"]
    assert report.duration_ms is not None and report.duration_ms >= 0


def test_the_child_reports_a_failing_import(tmp_path, monkeypatch):
    outcome = _child(tmp_path, monkeypatch, import_error="libcudart.so.12: cannot open")
    assert outcome.error.code == "IMPORT_ERROR"
    assert outcome.error.message == "OSError: libcudart.so.12: cannot open"


def test_the_child_without_torch_reports_it_isnt_importable(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {SRC!r}); "
            f"import runpy; runpy.run_module({TORCH_PROBE_MODULE!r}, run_name='__main__')",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    assert payload["import_ok"] is False and "ModuleNotFoundError" in payload["import_error"]


def test_importing_the_probe_modules_imports_no_torch_and_starts_nothing():
    code = (
        "import sys; import gpu_system_inspector.adapters.torch_probe, "
        "gpu_system_inspector.adapters._torch_probe_main, "
        "gpu_system_inspector.collectors.framework; "
        "print('torch' in sys.modules, 'subprocess' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.split() == ["False", "False"]


# --- SubprocessTorchProbe -----------------------------------------------------------------


def test_the_subprocess_probe_runs_the_documented_command_with_the_timeout():
    payload = probe_payload(StubTorch())
    runner = _runner(stdout=json.dumps(payload), duration_ms=1234)
    report = SubprocessTorchProbe(runner, PYTHON, distribution=_version).probe(42.0)
    (call,) = runner.calls
    assert call.argv == (PYTHON, "-m", TORCH_PROBE_MODULE) and call.timeout_s == 42.0
    assert call.source is Source.PYTORCH
    assert report.value.duration_ms == 1234 and report.value.allocator is None


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ({"timeout": True}, "TIMEOUT"),
        ({"returncode": -11, "stderr": "Segmentation fault"}, "PROBE_EXIT_-11"),
        ({"stdout": "not json"}, "PROBE_PROTOCOL_ERROR"),
        ({"raises": PermissionError(errno.EACCES, "denied")}, "PermissionError"),
    ],
    ids=["timeout", "crash", "garbage", "denied"],
)
def test_subprocess_failures_keep_their_codes(response, code):
    outcome = SubprocessTorchProbe(_runner(**response), PYTHON, distribution=_version).probe(60)
    assert outcome.value is None and outcome.error.code == code


def test_the_default_interpreter_is_the_inspectors_own():
    runner = FakeCommandRunner(files=frozenset({sys.executable})).on(
        *torch_probe_argv(sys.executable), stdout=json.dumps(probe_payload(StubTorch()))
    )
    SubprocessTorchProbe(runner).probe(60)
    assert runner.calls[0].argv[0] == sys.executable


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PackageNotFoundError("torch"), S.NOT_INSTALLED),
        (PermissionError(errno.EACCES, "metadata denied"), S.PERMISSION_DENIED),
    ],
)
def test_the_distribution_version_maps_failures_honestly(error, status):
    def lookup(name):
        raise error

    probe = SubprocessTorchProbe(FakeCommandRunner(), PYTHON, distribution=lookup)
    outcome = probe.distribution_version()
    assert outcome.status is status and outcome.source is Source.PYTORCH
    assert outcome.method == "fixture:importlib.metadata.version('torch')"


def test_construction_runs_and_reads_nothing():
    runner = FakeCommandRunner()

    def lookup(name):
        raise AssertionError("read at construction")

    SubprocessTorchProbe(runner, PYTHON, distribution=lookup)
    assert runner.calls == [] and runner.lookups == []


# --- InProcessTorchProbe ------------------------------------------------------------------


def _in_process(torch, **kw):
    return InProcessTorchProbe(lambda: torch, distribution=_version, **kw)


def test_in_process_without_cuda_initialised_reads_no_allocator():
    report = _in_process(StubTorch()).probe(60).value
    assert report.allocator is None and report.torch_version == "2.4.1+cu124"
    assert report.duration_ms is not None


def test_in_process_with_cuda_initialised_reads_the_callers_allocator():
    torch = StubTorch(devices=[StubDevice(allocated=2**20)], initialised=True)
    (sample,) = _in_process(torch).probe(60).value.allocator
    assert (sample.index, sample.allocated, sample.reserved) == (0, 2**20, 2 * 2**20)
    assert (sample.peak_allocated, sample.peak_reserved) == (3 * 2**20, 4 * 2**20)


def test_in_process_import_and_probe_failures_are_outcomes():
    def broken():
        raise OSError("libcudart.so.12: cannot open")

    failed = InProcessTorchProbe(broken, distribution=_version).probe(60)
    assert failed.error.code == "IMPORT_ERROR"

    class Odd(StubTorch):
        def __post_init__(self):
            super().__post_init__()
            self.cuda = None  # unexpected API shape: no torch.cuda

    odd = _in_process(Odd()).probe(60)
    assert odd.status is S.COLLECTION_FAILED and odd.error.code == "PROBE_EXCEPTION"
    assert odd.error.message.startswith("AttributeError:")


def test_in_process_provenance_labels_its_methods():
    probe = _in_process(StubTorch(), provenance=DataOrigin.FIXTURE)
    assert probe.probe(60).method == "fixture:in-process probe"
    assert probe.distribution_version().method == "fixture:importlib.metadata.version('torch')"


def test_the_child_runs_as_a_module_with_a_clean_stderr(tmp_path):
    """`-m` must not find the entry module already imported (no runpy RuntimeWarning)."""
    write_stub_torch_package(tmp_path)
    result = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", TORCH_PROBE_MODULE],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={"PYTHONPATH": f"{tmp_path}:{SRC}", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout)["import_ok"] is True
