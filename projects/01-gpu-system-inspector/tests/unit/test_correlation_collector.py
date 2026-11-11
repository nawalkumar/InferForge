"""IMP-K07 `CorrelationCollector` (DEC-09, DEC-49): PyTorch devices related to the NVML
inventory from the two sections already collected. Fixture parity, UUID first, PCI bus ID
only where a UUID can't decide, ambiguity left unresolved, never by index, name or memory,
missing inventories. Synthetic observations only; nothing is probed."""

import dataclasses
import inspect
import random

import pytest
from gsi_testkit.builders import (
    GPU_A,
    GPU_B,
    failed,
    gpu_device,
    nvml_inventory,
    ok,
    pci_scan,
    pytorch_cuda,
    smi_inventory,
    torch_device,
)
from gsi_testkit.fakes import FakeAdapterSet, FixedClock
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID

from gpu_system_inspector.collectors import (
    CorrelationCollector,
    normalise_pci_bus_id,
    normalise_uuid,
)
from gpu_system_inspector.collectors.framework import CORRELATION_PENDING_NOTE
from gpu_system_inspector.core import Inspector
from gpu_system_inspector.core.guard import run_guarded
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    ErrorDetail,
    FrameworkInfo,
    GPUDeviceInfo,
    GPUInventory,
    Observed,
    Source,
)

S = CollectionStatus
R = CollectorRunStatus
ids = pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
FX04A, FX06 = FIXTURES_BY_ID["FX04a"], FIXTURES_BY_ID["FX06"]
PENDING = Observed.not_applicable(note=CORRELATION_PENDING_NOTE, source=Source.INSPECTOR)


def _pending(pytorch):
    """The PyTorch section as the `pytorch` run hands it over (DEC-49 (5))."""
    devices = tuple(dataclasses.replace(d, correlated_device_ref=PENDING) for d in pytorch.devices)
    return dataclasses.replace(pytorch, devices=devices)


def _correlate(inventory, torch_devices, *, synthetic=True):
    pytorch = _pending(pytorch_cuda(list(torch_devices)))
    collector = CorrelationCollector(
        FixedClock(),
        inventory=inventory,
        frameworks=FrameworkInfo(pytorch=pytorch),
        synthetic=synthetic,
    )
    return collector.collect()


def _refs(outcome):
    return [d.correlated_device_ref for d in outcome.section.pytorch.devices]


def _nvml(*devices):
    return nvml_inventory(devices, pci_scan())


def _torch(index, spec, **changes):
    device = torch_device(index, spec, correlated_ref=None)
    return dataclasses.replace(device, **changes)


def _nvml_device(index, spec, *, bus=None, **changes):
    return dataclasses.replace(gpu_device(index, spec, bus=bus), **changes)


def _value(text, method="get_device_properties().pci"):
    return ok(text, Source.PYTORCH, method=method)


UNSUPPORTED_UUID = failed(
    S.NOT_SUPPORTED, "ATTRIBUTE_MISSING", "uuid is not exposed by this PyTorch version"
)
MISSING = failed(S.NOT_SUPPORTED, "ATTRIBUTE_MISSING", "not exposed")
LOST = ErrorDetail(code="NVML_ERROR_GPU_IS_LOST", message="GPU is lost")


# --- fixture parity -----------------------------------------------------------------------


@ids
def test_collector_reproduces_every_fixture_correlation(fixture):
    """Whole PyTorch section, run and errors, from the fixture's own inventories."""
    snapshot = fixture.snapshot
    frameworks = FrameworkInfo(pytorch=_pending(snapshot.frameworks.pytorch))
    outcome = CorrelationCollector(
        FixedClock(step_ms=50), inventory=snapshot.gpu, frameworks=frameworks, synthetic=True
    ).collect()
    assert outcome.section == snapshot.frameworks
    (run,) = (r for r in snapshot.collector_runs if r.collector == "correlation")
    assert outcome.run == run
    assert outcome.errors == () and outcome.unsupported == ()


def test_fx06_cross_ordered_devices_match_by_uuid():
    """AC19: `cuda:0` → `nvml:1`, `cuda:1` → `nvml:0`, never by index."""
    report = Inspector(FakeAdapterSet.from_scenario(FX06).adapter_set()).run()
    refs = {d.torch_index: d.correlated_device_ref for d in report.frameworks.pytorch.devices}
    assert (refs[0].value, refs[0].method) == ("nvml:1", "fixture:uuid")
    assert (refs[1].value, refs[1].method) == ("nvml:0", "fixture:uuid")
    assert refs[0].source is Source.INSPECTOR


def test_fx04a_without_an_nvml_inventory_is_not_applicable_not_unmatched():
    report = Inspector(FakeAdapterSet.from_scenario(FX04A).adapter_set()).run()
    assert report.gpu.inventory_source is Source.NVIDIA_SMI
    (device,) = report.frameworks.pytorch.devices
    ref = device.correlated_device_ref
    assert (ref.status, ref.note, ref.error) == (
        S.NOT_APPLICABLE,
        "No NVML inventory to correlate with",
        None,
    )
    assert [e for e in report.collection_errors if e.collector == "correlation"] == []


# --- UUID, then PCI bus ID ----------------------------------------------------------------


def test_uuids_are_normalised_as_the_data_model_states():
    nvml = _nvml(_nvml_device(0, GPU_A, uuid=ok("GPU-AAAAAAAA-0000-4000-8000-00000000000A")))
    (ref,) = _refs(_correlate(nvml, [_torch(0, GPU_A)]))
    assert (ref.value, ref.method) == ("nvml:0", "fixture:uuid")


def test_pci_bus_id_is_the_fallback_when_pytorch_exposes_no_uuid():
    nvml = _nvml(_nvml_device(0, GPU_B, bus=1), _nvml_device(1, GPU_A, bus=2))
    torch = _torch(0, GPU_A, uuid=UNSUPPORTED_UUID, pci_bus_id=_value("0000:02:00.0"))
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert (ref.value, ref.method) == ("nvml:1", "fixture:pci_bus_id")  # 00000000:02 = 0000:02


def test_pci_bus_id_is_the_fallback_when_nvml_has_no_uuid():
    nvml = _nvml(_nvml_device(0, GPU_A, uuid=MISSING))
    (ref,) = _refs(_correlate(nvml, [_torch(0, GPU_A)]))
    assert (ref.value, ref.method) == ("nvml:0", "fixture:pci_bus_id")


def test_uuid_takes_precedence_and_a_contradicting_pci_bus_id_is_noted():
    nvml = _nvml(_nvml_device(0, GPU_A, bus=1), _nvml_device(1, GPU_B, bus=2))
    torch = _torch(0, GPU_A, pci_bus_id=_value("0000:02:00.0"))  # GPU B's slot
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert (ref.value, ref.method) == ("nvml:0", "fixture:uuid")
    assert "PCI bus IDs differ" in ref.note and "matched by UUID" in ref.note


def test_a_pci_bus_id_never_overrides_two_different_uuids():
    nvml = _nvml(_nvml_device(0, GPU_B, bus=1))
    torch = _torch(0, GPU_A, pci_bus_id=_value("0000:01:00.0"))  # same slot, other UUID
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert (ref.status, ref.error.code) == (S.NOT_DETECTED, "NO_MATCH")


@pytest.mark.parametrize(
    "uuid",
    [
        "not-a-uuid",
        "MIG-aaaaaaaa-0000-4000-8000-00000000000a",
        "",
        "aaaaaaaa00004000800000000000000a",
    ],
)
def test_a_malformed_uuid_is_unusable_and_the_pci_bus_id_decides(uuid):
    nvml = _nvml(_nvml_device(0, GPU_A))
    (ref,) = _refs(_correlate(nvml, [_torch(0, GPU_A, uuid=_value(uuid, "uuid"))]))
    assert ref.method == "fixture:pci_bus_id"


@pytest.mark.parametrize(
    "pci", ["01:00.0", "0000:1:00.0", "0000:01:00", "zzzz:01:00.0", "0000:01:20.0"]
)
def test_a_malformed_pci_bus_id_is_unusable_and_never_guessed(pci):
    nvml = _nvml(_nvml_device(0, GPU_A, uuid=MISSING))
    torch = _torch(0, GPU_A, uuid=UNSUPPORTED_UUID, pci_bus_id=_value(pci))
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert (ref.status, ref.error.code) == (S.NOT_DETECTED, "NO_IDENTIFIER")  # not guessed


def test_a_device_with_no_usable_identifier_is_never_matched_by_index_name_or_memory():
    nvml = _nvml(_nvml_device(0, GPU_A))
    torch = _torch(0, GPU_A, uuid=UNSUPPORTED_UUID, pci_bus_id=MISSING)  # same index, name, memory
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert (ref.status, ref.error.code) == (S.NOT_DETECTED, "NO_IDENTIFIER")


def test_the_same_index_with_different_identifiers_is_no_match():
    nvml = _nvml(_nvml_device(0, GPU_B, bus=5))
    (ref,) = _refs(_correlate(nvml, [_torch(0, GPU_A)]))
    assert ref.error.code == "NO_MATCH"


# --- ambiguity ----------------------------------------------------------------------------


def test_duplicate_nvml_uuids_are_ambiguous_and_nothing_is_chosen():
    nvml = _nvml(_nvml_device(0, GPU_A, bus=1), _nvml_device(1, GPU_A, bus=2))
    (ref,) = _refs(_correlate(nvml, [_torch(0, GPU_A)]))
    assert (ref.status, ref.error.code, ref.value) == (S.NOT_DETECTED, "AMBIGUOUS_MATCH", None)
    assert "nvml:0, nvml:1" in ref.error.message


def test_duplicate_nvml_pci_bus_ids_are_ambiguous():
    nvml = _nvml(
        _nvml_device(0, GPU_A, bus=1, uuid=MISSING), _nvml_device(1, GPU_B, bus=1, uuid=MISSING)
    )
    torch = _torch(0, GPU_A, uuid=UNSUPPORTED_UUID)
    (ref,) = _refs(_correlate(nvml, [torch]))
    assert ref.error.code == "AMBIGUOUS_MATCH" and "pci_bus_id" in ref.error.message


def test_one_nvml_device_claimed_by_two_pytorch_devices_is_neither_ones():
    nvml = _nvml(_nvml_device(0, GPU_A), _nvml_device(1, GPU_B))
    first, second = _refs(
        _correlate(nvml, [_torch(0, GPU_A), _torch(1, GPU_A, pci_bus_id=MISSING)])
    )
    assert first.error.code == second.error.code == "AMBIGUOUS_MATCH"
    assert "cuda:1" in first.error.message and "cuda:0" in second.error.message


# --- partial and missing inventories ------------------------------------------------------


def test_nvml_only_and_pytorch_only_devices_are_simply_unmatched():
    nvml = _nvml(_nvml_device(0, GPU_A), _nvml_device(1, GPU_B))
    only_torch = _torch(
        1,
        GPU_A,
        uuid=_value("dddddddd-0000-4000-8000-00000000000d", "uuid"),
        pci_bus_id=_value("0000:09:00.0"),
    )
    outcome = _correlate(nvml, [_torch(0, GPU_B, pci_bus_id=_value("0000:02:00.0")), only_torch])
    matched, unmatched = _refs(outcome)
    assert matched.value == "nvml:1"  # nvml:0 has no PyTorch counterpart: not an error
    assert unmatched.error.code == "NO_MATCH"
    assert outcome.errors == () and outcome.run.status is R.COMPLETED


def test_an_unidentifiable_nvml_device_is_not_ruled_out_and_spares_the_others():
    lost = GPUDeviceInfo.unavailable("nvml:1", S.COLLECTION_FAILED, LOST)
    nvml = _nvml(_nvml_device(0, GPU_A), dataclasses.replace(lost, nvml_index=1))
    other = _torch(1, GPU_B, pci_bus_id=_value("0000:07:00.0"))
    matched, unresolved = _refs(_correlate(nvml, [_torch(0, GPU_A), other]))
    assert matched.value == "nvml:0"
    assert (
        unresolved.error.code == "UNIDENTIFIED_CANDIDATES" and "nvml:1" in unresolved.error.message
    )


def test_an_empty_nvml_inventory_is_no_match():
    (ref,) = _refs(_correlate(_nvml(), [_torch(0, GPU_A)]))
    assert ref.error.code == "NO_MATCH"


@pytest.mark.parametrize(
    "inventory",
    [
        smi_inventory([gpu_device(0, GPU_A, source=Source.NVIDIA_SMI)], pci_scan()),
        GPUInventory.unavailable(S.COLLECTION_FAILED, LOST),
    ],
    ids=["nvidia-smi-inventory", "no-inventory"],
)
def test_without_an_nvml_inventory_every_device_is_not_applicable(inventory):
    refs = _refs(_correlate(inventory, [_torch(0, GPU_A), _torch(1, GPU_B)]))
    assert [(r.status, r.note) for r in refs] == [
        (S.NOT_APPLICABLE, "No NVML inventory to correlate with")
    ] * 2


def test_no_pytorch_devices_leaves_the_section_unchanged():
    frameworks = FIXTURES_BY_ID["FX05"].snapshot.frameworks
    outcome = CorrelationCollector(FixedClock(), inventory=_nvml(), frameworks=frameworks).collect()
    assert outcome.section == frameworks


# --- determinism, provenance, boundaries --------------------------------------------------


def test_results_follow_identifiers_whatever_the_enumeration_order():
    nvml_devices = [_nvml_device(0, GPU_A, bus=1), _nvml_device(1, GPU_B, bus=2)]
    torch_devices = [_torch(0, GPU_B, pci_bus_id=_value("0000:02:00.0")), _torch(1, GPU_A)]
    expected = [r.value for r in _refs(_correlate(_nvml(*nvml_devices), torch_devices))]
    for seed in range(5):
        shuffled = nvml_devices[:]
        random.Random(seed).shuffle(shuffled)
        inventory = dataclasses.replace(_nvml(*nvml_devices), devices=tuple(shuffled))
        assert [r.value for r in _refs(_correlate(inventory, torch_devices))] == expected
    assert expected == ["nvml:1", "nvml:0"]


def test_live_results_carry_no_fixture_prefix():
    (ref,) = _refs(_correlate(_nvml(_nvml_device(0, GPU_A)), [_torch(0, GPU_A)], synthetic=False))
    assert ref.method == "uuid"


def test_the_collector_takes_observations_only():
    parameters = set(inspect.signature(CorrelationCollector).parameters)
    assert parameters == {"clock", "inventory", "frameworks", "synthetic"}  # no adapter


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00000000:01:00.0", (0, 1, 0, 0)),
        ("0000:01:00.0", (0, 1, 0, 0)),
        ("0001:AF:1f.7", (1, 0xAF, 0x1F, 7)),
        ("0000:01:00.8", None),
        (" 0000:01:00.0 ", (0, 1, 0, 0)),
    ],
)
def test_pci_bus_id_normalisation(text, expected):
    assert normalise_pci_bus_id(ok(text)) == expected


def test_uuid_normalisation_rejects_what_it_cannot_prove():
    assert normalise_uuid(ok("GPU-AAAAAAAA-0000-4000-8000-00000000000A")) == (
        "aaaaaaaa-0000-4000-8000-00000000000a"
    )
    assert normalise_uuid(MISSING) is None and normalise_uuid(ok("GPU-")) is None


# --- the guard ----------------------------------------------------------------------------


def test_an_exception_is_contained_and_keeps_the_pytorch_facts():
    class Broken(GPUInventory):
        @property
        def inventory_source(self):  # type: ignore[override]
            raise RuntimeError("synthetic correlation bug")

    snapshot = FX06.snapshot
    frameworks = FrameworkInfo(pytorch=_pending(snapshot.frameworks.pytorch))
    broken = object.__new__(Broken)
    clock = FixedClock()
    outcome = run_guarded(
        CorrelationCollector(clock, inventory=broken, frameworks=frameworks),
        clock,
        report_path="/frameworks/pytorch/devices",
    )
    assert outcome.run.status is R.FAILED
    (error,) = outcome.errors
    assert error.message == "RuntimeError: synthetic correlation bug"
    for device, original in zip(
        outcome.section.pytorch.devices, snapshot.frameworks.pytorch.devices, strict=True
    ):
        assert device.correlated_device_ref.status is S.COLLECTION_FAILED
        assert (
            dataclasses.replace(device, correlated_device_ref=original.correlated_device_ref)
            == original
        )


def test_an_inspection_with_a_failing_correlation_keeps_every_other_section(monkeypatch):
    def explode(self):
        raise RuntimeError("synthetic correlation crash")

    monkeypatch.setattr(CorrelationCollector, "collect", explode)
    report = Inspector(FakeAdapterSet.from_scenario(FX06).adapter_set()).run()
    runs = {r.collector: r.status for r in report.collector_runs}
    assert runs["correlation"] is R.FAILED and runs["pytorch"] is R.COMPLETED
    (error,) = [e for e in report.collection_errors if e.collector == "correlation"]
    assert error.report_path == "/frameworks/pytorch/devices"
    assert report.gpu == FX06.snapshot.gpu
    assert report.frameworks.pytorch.version == FX06.snapshot.frameworks.pytorch.version
