"""Fake PyTorch probe implementing `TorchProbe` (protocol in data flow §5).

`subprocess_output()` produces what the probe child would leave behind (JSON on stdout, an
import-error payload, or a timeout) from a fixture's `PyTorchInfo`; `probe()` decodes it
with the production protocol decoder. Never imports torch or starts a process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError

from gpu_system_inspector.adapters import (
    CommandResult,
    TorchProbeReport,
    decode_probe_output,
)
from gpu_system_inspector.adapters.outcomes import TIMEOUT
from gpu_system_inspector.models import (
    PROBE_PROTOCOL_VERSION,
    CollectionStatus,
    DataOrigin,
    InspectionSnapshot,
    Observed,
    PyTorchInfo,
    Source,
)
from gsi_testkit.builders import fx
from gsi_testkit.fakes.commands import FakeCommandRunner

PROBE_METHOD = fx("python -m gpu_system_inspector.adapters._torch_probe_main")
PROBE_ARGV = (
    "/opt/synthetic/venv/bin/python",
    "-m",
    "gpu_system_inspector.adapters._torch_probe_main",
)


def _v(obs: Observed) -> object:
    return obs.value


@dataclass
class FakeTorchProbe:
    pytorch: PyTorchInfo
    calls: list[str] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakeTorchProbe:
        return cls(pytorch=snapshot.frameworks.pytorch)

    def distribution_version(self) -> Observed[str]:
        self.calls.append("importlib.metadata.version('torch')")
        return self.pytorch.distribution_version

    def subprocess_output(self, timeout_s: float) -> Observed[CommandResult]:
        self.calls.append(f"probe(timeout_s={timeout_s})")
        info = self.pytorch
        if info.status in (CollectionStatus.NOT_INSTALLED, CollectionStatus.SKIPPED):
            raise AssertionError(f"the probe must not run when PyTorch is {info.status}")
        code = info.version.error.code if info.version.error else None
        if code == TIMEOUT:
            return Observed.failed(
                CollectionStatus.COLLECTION_FAILED,
                code=TIMEOUT,
                message=f"probe did not finish within {timeout_s} s",
                source=Source.PYTORCH,
                method=PROBE_METHOD,
            )
        payload: dict[str, object] = {"probe_protocol": PROBE_PROTOCOL_VERSION}
        if code == "IMPORT_ERROR":
            payload.update(import_ok=False, import_error=info.version.error.message)
        else:
            payload.update(
                import_ok=True,
                import_error=None,
                torch_version=_v(info.version),
                cuda_build=_v(info.cuda_build),
                build_cuda_version=_v(info.build_cuda_version),
                hip_version=None,
                cudnn_version=_v(info.cudnn_version),
                cuda_available=_v(info.cuda_available),
                cuda_init_warnings=list(info.cuda_init_warnings),
                device_count=_v(info.device_count),
                devices=[
                    {
                        "index": d.torch_index,
                        "name": _v(d.name),
                        "capability": _v(d.compute_capability),
                        "total_memory": _v(d.total_memory_bytes),
                        "multiprocessor_count": _v(d.multiprocessor_count),
                        "uuid": _v(d.uuid),
                        "pci_bus_id": _v(d.pci_bus_id),
                    }
                    for d in info.devices
                ],
                device_error=None,
            )
        result = CommandResult(
            argv=PROBE_ARGV,
            returncode=0,
            stdout=json.dumps(payload, sort_keys=True),
            stderr="",
            duration_ms=info.probe.duration_ms or 0,
        )
        return Observed.collected(result, source=Source.PYTORCH, method=PROBE_METHOD)

    def probe(self, timeout_s: float) -> Observed[TorchProbeReport]:
        return decode_probe_output(self.subprocess_output(timeout_s))

    def distribution_lookup(self, name: str) -> str:
        """`importlib.metadata.version` as the fixture saw it, for the production probes."""
        found = self.pytorch.distribution_version
        if found.is_collected and found.value is not None:
            return found.value
        raise FixturePackageNotFound(found.error.message if found.error else name)

    def canned(self, runner: FakeCommandRunner, timeout_s: float = 60.0) -> FakeCommandRunner:
        """Teach `runner` the probe child's run for this fixture (nothing if it never runs)."""
        if self.pytorch.status in (CollectionStatus.NOT_INSTALLED, CollectionStatus.SKIPPED):
            return runner
        runner.files = runner.files | {PROBE_ARGV[0]}
        output = self.subprocess_output(timeout_s)
        self.calls.clear()
        if not output.is_collected:  # the fixture's timeout
            return runner.on(*PROBE_ARGV, timeout=True)
        assert output.value is not None
        return runner.on(
            *PROBE_ARGV, stdout=output.value.stdout, duration_ms=output.value.duration_ms
        )


class FixturePackageNotFound(PackageNotFoundError):
    """`PackageNotFoundError` with the fixture's message."""

    def __init__(self, message: str) -> None:
        super().__init__("torch")
        self.message = message

    def __str__(self) -> str:
        return self.message
