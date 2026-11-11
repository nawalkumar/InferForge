"""IMP-K06 `FrameworkCollector` (DEC-48) over the production probes: each fixture's
`SubprocessTorchProbe` on the fake runner, and `InProcessTorchProbe` on `StubTorch`.
Fixture parity, the opt-out, not-installed and failure paths, separate build / runtime /
device facts, the correlation boundary and the allocator rules. No real PyTorch or GPU."""

import dataclasses
import errno
import json

import pytest
from gsi_testkit.builders import COLLECTOR_ORDER
from gsi_testkit.fakes import (
    FakeAdapterSet,
    FakeCommandRunner,
    FixedClock,
    StubDevice,
    StubTorch,
)
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.adapters import InProcessTorchProbe, SubprocessTorchProbe
from gpu_system_inspector.adapters.torch_probe_payload import probe_payload
from gpu_system_inspector.collectors import FrameworkCollector
from gpu_system_inspector.collectors.framework import CORRELATION_PENDING_NOTE
from gpu_system_inspector.core import Inspector
from gpu_system_inspector.core.guard import run_guarded
from gpu_system_inspector.models import (
    ALLOCATOR_NOT_APPLICABLE_NOTE,
    ALLOCATOR_SCOPE_NOTE,
    BUILD_CUDA_NOTE,
    CollectionStatus,
    CollectorRunStatus,
    InspectionOptions,
    Observed,
    ProbeMode,
    Source,
)

S = CollectionStatus
R = CollectorRunStatus
STEP_MS = 5 * (COLLECTOR_ORDER.index("pytorch") + 1)
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
FX01, FX05, FX15, FX16 = (FIXTURES_BY_ID[i] for i in ("FX01", "FX05", "FX15", "FX16"))
PYTHON = "/opt/synthetic/venv/bin/python"


def _collect(probe, options=None):
    collector = FrameworkCollector(
        probe, FixedClock(step_ms=STEP_MS), options=options or InspectionOptions()
    )
    return collector.collect()


def _expected(fixture):
    pytorch = fixture.snapshot.frameworks.pytorch
    placeholder = Observed.not_applicable(note=CORRELATION_PENDING_NOTE, source=Source.INSPECTOR)
    devices = tuple(
        dataclasses.replace(d, correlated_device_ref=placeholder) for d in pytorch.devices
    )
    return dataclasses.replace(pytorch, devices=devices)


def _subprocess(torch, **response):
    runner = FakeCommandRunner(files=frozenset({PYTHON})).on(
        PYTHON,
        "-m",
        "gpu_system_inspector.adapters._torch_probe_main",
        **{"stdout": json.dumps(probe_payload(torch)), **response},
    )
    return SubprocessTorchProbe(runner, PYTHON, distribution=lambda name: "2.4.1"), runner


# --- parity -------------------------------------------------------------------------------


@ids
def test_collector_reproduces_every_fixture_pytorch_section(fixture):
    """Every field, status, source, method, note and error; `correlated_device_ref` is the
    interim value the correlation run replaces (DEC-49 (5))."""
    adapters = FakeAdapterSet.from_scenario(fixture)
    outcome = _collect(adapters.torch_probe)
    assert outcome.section.pytorch == _expected(fixture)
    (run,) = (r for r in fixture.snapshot.collector_runs if r.collector == "pytorch")
    assert outcome.run == run
    assert outcome.errors == tuple(
        e for e in fixture.snapshot.collection_errors if e.collector == "pytorch"
    )
    assert outcome.unsupported == ()


@ids
def test_no_probe_runs_unless_pytorch_is_installed(fixture):
    """AC03: without a torch distribution no process is started."""
    adapters = FakeAdapterSet.from_scenario(fixture)
    _collect(adapters.torch_probe)
    probe_runs = [c for c in adapters.commands.calls if "_torch_probe_main" in " ".join(c.argv)]
    installed = fixture.snapshot.frameworks.pytorch.status is not S.NOT_INSTALLED
    assert len(probe_runs) == (1 if installed else 0)


def test_the_opt_out_reads_and_runs_nothing():
    """AC59: `include_pytorch=False` → `SKIPPED`."""
    adapters = FakeAdapterSet.from_scenario(FX01)
    outcome = _collect(adapters.torch_probe, InspectionOptions(include_pytorch=False))
    assert outcome.section.pytorch.status is S.SKIPPED
    assert outcome.section.pytorch.distribution_version.status is S.SKIPPED
    assert adapters.commands.calls == [] and outcome.run.status is R.COMPLETED


def test_installed_but_broken_is_not_not_installed():
    """AC15: FX15 (import error) vs FX05 (not installed)."""
    broken = _collect(FakeAdapterSet.from_scenario(FX15).torch_probe).section.pytorch
    missing = _collect(FakeAdapterSet.from_scenario(FX05).torch_probe).section.pytorch
    assert broken.status is S.COLLECTION_FAILED and broken.version.error.code == "IMPORT_ERROR"
    assert broken.distribution_version.is_collected  # installed
    assert missing.status is S.NOT_INSTALLED


def test_a_timeout_is_reported_with_the_configured_limit():
    outcome = _collect(
        FakeAdapterSet.from_scenario(FX16).torch_probe, InspectionOptions(pytorch_timeout_s=60.0)
    )
    (error,) = outcome.errors
    assert (error.error_code, error.report_path) == ("TIMEOUT", "/frameworks/pytorch")
    assert error.message == "PyTorch probe exceeded pytorch_timeout_s (60.0 s)"
    assert outcome.run.status is R.PARTIAL


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ({"returncode": 3, "stderr": "boom"}, "PROBE_EXIT_3"),
        ({"stdout": "{}"}, "PROBE_PROTOCOL_ERROR"),
    ],
)
def test_other_probe_failures_keep_their_code_and_the_distribution(response, code):
    probe, _ = _subprocess(StubTorch(), **response)
    outcome = _collect(probe)
    pytorch = outcome.section.pytorch
    assert pytorch.status is S.COLLECTION_FAILED and pytorch.version.error.code == code
    assert pytorch.distribution_version.value == "2.4.1"
    assert [e.error_code for e in outcome.errors] == [code]


def test_unreadable_metadata_is_a_failure_not_absence():
    def lookup(name):
        raise PermissionError(errno.EACCES, "metadata denied")

    runner = FakeCommandRunner()
    outcome = _collect(SubprocessTorchProbe(runner, PYTHON, distribution=lookup))
    assert outcome.section.pytorch.status is S.COLLECTION_FAILED
    (error,) = outcome.errors
    assert (error.status, error.report_path) == (
        S.PERMISSION_DENIED,
        "/frameworks/pytorch/distribution_version",
    )
    assert runner.calls == []


# --- separate facts -----------------------------------------------------------------------


def test_build_runtime_and_devices_are_separate_facts():
    probe, _ = _subprocess(StubTorch(available=True))
    pytorch = _collect(probe).section.pytorch
    assert pytorch.cuda_build.value is True and pytorch.cuda_available.value is True
    assert pytorch.build_cuda_version.note == BUILD_CUDA_NOTE  # neither toolkit nor driver
    assert pytorch.build_cuda_version.method == "fixture:torch.version.cuda"
    assert pytorch.device_count.value == 1 and len(pytorch.devices) == 1


def test_a_cpu_only_build_never_looks_cuda_capable():
    probe, _ = _subprocess(StubTorch.cpu_only())
    pytorch = _collect(probe).section.pytorch
    assert pytorch.cuda_build.value is False and pytorch.cuda_available.value is False
    assert pytorch.build_cuda_version.status is S.NOT_APPLICABLE
    assert pytorch.build_cuda_version.note == "CPU-only PyTorch build"
    assert pytorch.devices == ()


def test_a_rocm_build_is_not_called_cpu_only():
    probe, _ = _subprocess(StubTorch(cuda_version=None, hip_version="6.1", cudnn=None, devices=[]))
    pytorch = _collect(probe).section.pytorch
    assert pytorch.rocm_build.value is True and pytorch.cuda_build.value is False
    assert pytorch.build_cuda_version.note == "Non-CUDA (ROCm) PyTorch build"


def test_a_cuda_build_without_cudnn_says_so():
    probe, _ = _subprocess(StubTorch(cudnn=None))
    cudnn = _collect(probe).section.pytorch.cudnn_version
    assert (cudnn.status, cudnn.error.code) == (S.NOT_DETECTED, "CUDNN_NOT_AVAILABLE")


def test_cuda_unavailable_with_a_warning_is_preserved():
    message = "CUDA initialization: Found no NVIDIA driver on your system."
    probe, _ = _subprocess(StubTorch(available=False, devices=[], init_warnings=(message,)))
    pytorch = _collect(probe).section.pytorch
    assert pytorch.cuda_available.value is False and pytorch.cuda_init_warnings == (message,)
    assert pytorch.device_count.value == 0


def test_a_device_enumeration_failure_keeps_everything_else():
    probe, _ = _subprocess(StubTorch(device_failure=RuntimeError("CUDA error: unknown")))
    outcome = _collect(probe)
    pytorch = outcome.section.pytorch
    assert pytorch.status is S.COLLECTED and pytorch.version.value == "2.4.1+cu124"
    assert pytorch.cuda_available.value is True and pytorch.devices == ()
    assert pytorch.device_count.error.code == "DEVICE_ENUMERATION_FAILED"
    (error,) = outcome.errors
    assert error.report_path == "/frameworks/pytorch/devices" and outcome.run.status is R.PARTIAL


def test_device_attributes_an_older_pytorch_lacks_are_not_supported():
    probe, _ = _subprocess(StubTorch(devices=[StubDevice(uuid=None, pci=None)]))
    (device,) = _collect(probe).section.pytorch.devices
    for field in (device.uuid, device.pci_bus_id):
        assert (field.status, field.error.code) == (S.NOT_SUPPORTED, "ATTRIBUTE_MISSING")
    assert device.name.is_collected


def test_devices_are_not_correlated_here():
    """The correlation run (IMP-K07) owns it: no UUID, PCI or index matching here."""
    probe, _ = _subprocess(StubTorch(devices=[StubDevice(), StubDevice(pci=(0, 2, 0))]))
    for device in _collect(probe).section.pytorch.devices:
        ref = device.correlated_device_ref
        assert (ref.status, ref.note) == (S.NOT_APPLICABLE, CORRELATION_PENDING_NOTE)
        assert device.uuid.is_collected  # the facts K07 will need are kept


# --- allocator (FR-10, AC16) --------------------------------------------------------------


def test_the_subprocess_probe_never_reports_allocator_numbers():
    probe, _ = _subprocess(StubTorch())
    allocator = _collect(probe).section.pytorch.allocator
    assert (allocator.status, allocator.note) == (S.NOT_APPLICABLE, ALLOCATOR_NOT_APPLICABLE_NOTE)


def test_in_process_allocator_statistics_need_cuda_initialised_in_the_caller():
    options = InspectionOptions(pytorch_probe_mode=ProbeMode.IN_PROCESS)

    def probe(torch):
        return InProcessTorchProbe(lambda: torch, distribution=lambda name: "2.4.1")

    cold = _collect(probe(StubTorch()), options).section.pytorch
    assert cold.allocator.status is S.NOT_APPLICABLE and cold.probe.mode is ProbeMode.IN_PROCESS
    warm = StubTorch(devices=[StubDevice(allocated=5 * 2**20)], initialised=True)
    allocator = _collect(probe(warm), options).section.pytorch.allocator
    (stats,) = allocator.value
    assert allocator.note == ALLOCATOR_SCOPE_NOTE
    assert stats.allocated_bytes.value == 5 * 2**20
    assert (
        stats.allocated_bytes.note == ALLOCATOR_SCOPE_NOTE and stats.allocated_bytes.unit == "bytes"
    )


# --- the guard and the pipeline -----------------------------------------------------------


def test_a_crashing_probe_is_contained_by_the_guard_and_spares_other_sections():
    class Exploding:
        provenance = FakeAdapterSet.from_scenario(FX01).torch_probe.provenance

        def distribution_version(self):
            raise RuntimeError("synthetic probe bug")

        def probe(self, timeout_s):
            raise AssertionError("not reached")

    adapters = FakeAdapterSet.from_scenario(FX01)
    adapters.torch_probe = Exploding()
    report = Inspector(adapters.adapter_set()).run()
    runs = {r.collector: r.status for r in report.collector_runs}
    assert runs["pytorch"] is R.FAILED
    (error,) = [e for e in report.collection_errors if e.collector == "pytorch"]
    assert (error.report_path, error.message) == (
        "/frameworks/pytorch",
        "RuntimeError: synthetic probe bug",
    )
    assert report.frameworks.pytorch.status is S.COLLECTION_FAILED
    snapshot = FX01.snapshot
    assert (report.cuda_environment, report.gpu, report.driver) == (
        snapshot.cuda_environment,
        snapshot.gpu,
        snapshot.driver,
    )


def test_the_guard_fallback_section():
    from gpu_system_inspector.models import ErrorDetail

    adapters = FakeAdapterSet.from_scenario(FX01)
    clock = FixedClock()

    class Broken:
        provenance = adapters.torch_probe.provenance

        def distribution_version(self):
            raise ValueError("bad")

        def probe(self, timeout_s):
            raise AssertionError("not reached")

    outcome = run_guarded(
        FrameworkCollector(Broken(), clock, options=InspectionOptions()),
        clock,
        report_path="/frameworks/pytorch",
    )
    assert outcome.run.status is R.FAILED
    assert outcome.section.pytorch.version.error == ErrorDetail(
        code="UNHANDLED_EXCEPTION", message="ValueError: bad"
    )
