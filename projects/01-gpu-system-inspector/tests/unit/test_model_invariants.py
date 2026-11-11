"""Critical model invariants and strict-decoder branches identified by the coverage baseline."""

import copy
import dataclasses

import pytest
from gsi_testkit.builders import GIB, GPU_A, ok, pci_device, pci_scan, torch_device
from gsi_testkit.expectations import materialize_readiness
from gsi_testkit.fixtures import FIXTURES_BY_ID

from gpu_system_inspector.models import (
    ALLOCATOR_SCOPE_NOTE,
    AllocatorStats,
    CollectionStatus,
    InspectionReport,
    InvalidModelError,
    Observed,
    Source,
)
from gpu_system_inspector.reporting import SerializationError, from_json_dict, to_json_dict

FX01 = FIXTURES_BY_ID["FX01"]
PYTORCH = FX01.snapshot.frameworks.pytorch
INVENTORY = FX01.snapshot.gpu


def _stats(index: int) -> AllocatorStats:
    def b(v: int) -> Observed:
        return ok(v, Source.PYTORCH, "bytes", note=ALLOCATOR_SCOPE_NOTE)

    return AllocatorStats(
        torch_index=index,
        allocated_bytes=b(1),
        reserved_bytes=b(2),
        peak_allocated_bytes=b(3),
        peak_reserved_bytes=b(4),
    )


# --- PyTorch section (drives D01) -----------------------------------------------------------


def test_pytorch_section_status_vocabulary_is_restricted():
    with pytest.raises(InvalidModelError, match="PyTorchInfo.status"):
        dataclasses.replace(PYTORCH, status=CollectionStatus.PERMISSION_DENIED)


def test_pytorch_devices_are_unique_and_only_listed_when_collected():
    device = PYTORCH.devices[0]
    with pytest.raises(InvalidModelError, match="unique torch_index"):
        dataclasses.replace(PYTORCH, devices=(device, device), device_count=ok(2, Source.PYTORCH))
    with pytest.raises(InvalidModelError, match="only be listed"):
        dataclasses.replace(PYTORCH, status=CollectionStatus.COLLECTION_FAILED)
    with pytest.raises(InvalidModelError, match="doesn't match"):
        dataclasses.replace(PYTORCH, device_count=ok(3, Source.PYTORCH))


def test_build_cuda_version_needs_a_cuda_build():
    unknown_build = Observed.failed(CollectionStatus.COLLECTION_FAILED, code="X", message="y")
    with pytest.raises(InvalidModelError, match="requires cuda_build to be true"):
        dataclasses.replace(PYTORCH, cuda_build=unknown_build)


def test_in_process_allocator_stats_are_ordered_and_unique():
    collected = dataclasses.replace(PYTORCH, allocator=ok((_stats(1), _stats(0)), Source.PYTORCH))
    assert [s.torch_index for s in collected.allocator.value] == [0, 1]
    with pytest.raises(InvalidModelError, match="allocator stats must have unique"):
        dataclasses.replace(PYTORCH, allocator=ok((_stats(0), _stats(0)), Source.PYTORCH))


def test_correlated_device_ref_must_be_a_device_ref():
    with pytest.raises(InvalidModelError, match="is not a device_ref"):
        dataclasses.replace(
            PYTORCH.devices[0], correlated_device_ref=ok("cuda:0", Source.INSPECTOR)
        )
    assert torch_device(0, GPU_A, correlated_ref=None).correlated_device_ref.status is (
        CollectionStatus.NOT_APPLICABLE
    )


# --- GPU inventory consistency ----------------------------------------------------------------


def test_inventory_source_must_be_an_inventory_source():
    with pytest.raises(InvalidModelError, match="NVML, NVIDIA_SMI or None"):
        dataclasses.replace(INVENTORY, inventory_source=Source.PYTORCH)


def test_devices_and_collected_count_need_an_inventory_source():
    with pytest.raises(InvalidModelError, match="must name its inventory_source"):
        dataclasses.replace(INVENTORY, inventory_source=None)
    with pytest.raises(InvalidModelError, match="requires an inventory_source"):
        dataclasses.replace(INVENTORY, inventory_source=None, devices=(), device_count=ok(0))


def test_pci_observations_need_unique_addresses():
    with pytest.raises(InvalidModelError, match="unique addresses"):
        dataclasses.replace(INVENTORY, pci_observations=pci_scan(pci_device(1), pci_device(1)))


# --- report consistency -----------------------------------------------------------------------


def test_readiness_must_use_the_requested_threshold():
    run = FX01.readiness_runs[0]  # 8 GiB
    sixteen = materialize_readiness(dataclasses.replace(run, required_vram_gib=16), FX01.snapshot)
    assert sixteen.required_bytes == 16 * GIB
    with pytest.raises(InvalidModelError, match="must match the options"):
        dataclasses.replace(FX01.report(run), readiness=sixteen)


def test_limitations_must_not_be_empty_strings():
    with pytest.raises(InvalidModelError, match="empty strings"):
        dataclasses.replace(FX01.report(), limitations=(*FX01.report().limitations, ""))


# --- strict decoder branches ------------------------------------------------------------------


@pytest.fixture
def data():
    return copy.deepcopy(to_json_dict(FX01.report()))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.__setitem__("system", None), "/system: null is not allowed"),
        (lambda d: d.__setitem__("runtime", {}), "/runtime: expected an array"),
        (
            lambda d: d["cuda_environment"].__setitem__("environment", []),
            "/cuda_environment/environment: expected an object",
        ),
        (
            lambda d: d["runtime"][0]["power_draw_watts"].__setitem__("value", "31.5 W"),
            "power_draw_watts/value: expected a number",
        ),
        (
            lambda d: d["system"]["os_name"].__setitem__("value", 7),
            "/system/os_name/value: expected a string",
        ),
        (
            lambda d: d.__setitem__("application", "gpu-system-inspector"),
            "/application: expected an object for ApplicationInfo",
        ),
    ],
    ids=["null", "array", "mapping", "number", "string", "object"],
)
def test_decoder_reports_the_path_of_each_type_error(data, mutate, message):
    mutate(data)
    with pytest.raises(SerializationError, match=message):
        from_json_dict(data, InspectionReport)
