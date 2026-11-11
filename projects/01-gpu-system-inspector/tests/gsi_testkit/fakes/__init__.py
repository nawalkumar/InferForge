"""Deterministic fake adapters and collectors built from an FX fixture.

Each fake implements one adapter protocol from `gpu_system_inspector.adapters`; the
conformance tests in `tests/unit/test_adapter_contracts.py` check signatures and behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpu_system_inspector.adapters import (
    AdapterSet,
    CudaToolkitLocator,
    SubprocessTorchProbe,
    ToolkitLocator,
    TorchProbe,
)
from gpu_system_inspector.models import DataOrigin
from gsi_testkit.fakes.collectors import FakeCollector, fake_collectors
from gsi_testkit.fakes.commands import CannedResponse, CommandCall, FakeCommandRunner
from gsi_testkit.fakes.host import FakeHostApi
from gsi_testkit.fakes.nvidia_smi import FakeNvidiaSmi
from gsi_testkit.fakes.nvml import READ_ONLY_FUNCTIONS, FakeNvml, FakeNvmlDevice, FakeNvmlSession
from gsi_testkit.fakes.platform import FakePciScanner, FakePlatform, FixedClock
from gsi_testkit.fakes.pynvml_stub import StubGpu, StubNVMLError, StubPynvml
from gsi_testkit.fakes.sysfs import FakeSysfsReader, sysfs_device, sysfs_reader_for
from gsi_testkit.fakes.toolkit import FakeToolkitFiles, nvcc_version_output, version_json
from gsi_testkit.fakes.torch_probe import PROBE_ARGV, FakeTorchProbe
from gsi_testkit.fakes.torch_stub import StubDevice, StubTorch, write_stub_torch_package


class FailingToolkitLocator:
    """Raises the fixture's injected collector failure (FX10), through the real guard."""

    provenance = DataOrigin.FIXTURE

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def discover(self, hints: Any) -> Any:
        raise self.error


@dataclass
class FakeAdapterSet:
    nvml: FakeNvml
    nvidia_smi: FakeNvidiaSmi
    pci: FakePciScanner
    torch_fake: FakeTorchProbe  # the fixture's probe facts; `torch_probe` is the production one
    torch_probe: TorchProbe
    platform: FakePlatform
    commands: FakeCommandRunner
    toolkit_files: FakeToolkitFiles
    toolkit: ToolkitLocator
    clock: FixedClock
    collectors: dict[str, FakeCollector[Any]]

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_scenario(cls, scenario: Any) -> FakeAdapterSet:
        snapshot = scenario.snapshot
        platform = FakePlatform.from_snapshot(snapshot)
        commands = FakeCommandRunner.from_snapshot(snapshot)
        files = FakeToolkitFiles.from_snapshot(snapshot)
        torch_fake = FakeTorchProbe.from_snapshot(snapshot)
        torch_fake.canned(commands)
        crash = scenario.collector_failures.get("cuda_environment")
        toolkit: ToolkitLocator = (
            FailingToolkitLocator(crash)
            if crash is not None
            else CudaToolkitLocator(commands, platform, files)
        )
        return cls(
            nvml=FakeNvml.from_scenario(scenario),
            nvidia_smi=FakeNvidiaSmi.from_snapshot(snapshot),
            pci=FakePciScanner.from_snapshot(snapshot),
            torch_fake=torch_fake,
            torch_probe=SubprocessTorchProbe(
                commands, PROBE_ARGV[0], distribution=torch_fake.distribution_lookup
            ),
            platform=platform,
            commands=commands,
            toolkit_files=files,
            toolkit=toolkit,
            clock=FixedClock(),
            collectors=fake_collectors(scenario),
        )

    def adapter_set(self) -> AdapterSet:
        """The fakes as the production-shaped, validated `AdapterSet`."""
        return AdapterSet(
            platform=self.platform,
            pci=self.pci,
            nvml=self.nvml,
            nvidia_smi=self.nvidia_smi,
            commands=self.commands,
            toolkit=self.toolkit,
            torch_probe=self.torch_probe,
            clock=self.clock,
            provenance=self.provenance,
        )


__all__ = [
    "READ_ONLY_FUNCTIONS",
    "CannedResponse",
    "CommandCall",
    "FakeAdapterSet",
    "FakeCollector",
    "FakeCommandRunner",
    "FakeHostApi",
    "FakeNvidiaSmi",
    "FakeNvml",
    "FakeNvmlDevice",
    "FakeNvmlSession",
    "FakePciScanner",
    "FakePlatform",
    "FakeSysfsReader",
    "FakeTorchProbe",
    "StubDevice",
    "StubTorch",
    "write_stub_torch_package",
    "FailingToolkitLocator",
    "FakeToolkitFiles",
    "nvcc_version_output",
    "version_json",
    "FixedClock",
    "StubGpu",
    "StubNVMLError",
    "StubPynvml",
    "fake_collectors",
    "sysfs_device",
    "sysfs_reader_for",
]
