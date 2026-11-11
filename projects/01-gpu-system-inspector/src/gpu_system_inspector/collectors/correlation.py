"""Device correlation (architecture §4.7, IMP-K07, DEC-09, DEC-49).

It relates each PyTorch device to an NVML device from what the two runs already observed:
the `gpu_hardware` run's `GPUInventory` and the `pytorch` run's `FrameworkInfo`. Nothing is
probed again: no NVML, no PyTorch, no process. The result is the PyTorch section with every
device's `correlated_device_ref` set; nothing else changes.

**Order (DEC-09).** By normalised UUID among the NVML devices with a usable UUID; only if
that finds nothing, by PCI bus ID among the NVML devices a UUID can't rule out. Never by
index, name or memory. An identifier that is missing or doesn't parse is unusable, never a
wildcard. When the evidence doesn't single out one device, none is chosen.

- One match → `COLLECTED` = the NVML `device_ref`, method `uuid` or `pci_bus_id`.
- No NVML inventory → `NOT_APPLICABLE` (FX04a).
- Otherwise `NOT_DETECTED`: `NO_MATCH`, `NO_IDENTIFIER`, `AMBIGUOUS_MATCH` or
  `UNIDENTIFIED_CANDIDATES`.

These are facts about the devices, not collection failures: the run is `COMPLETED` with no
`CollectionError`. The collector owns no capability records.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import ClassVar

from gpu_system_inspector.adapters import Clock
from gpu_system_inspector.models import (
    FIXTURE_METHOD_PREFIX,
    CollectionStatus,
    CollectorOutcome,
    CollectorRun,
    CollectorRunStatus,
    FrameworkInfo,
    GPUDeviceInfo,
    GPUInventory,
    Observed,
    Source,
    TorchDeviceInfo,
)
from gpu_system_inspector.models.observed import ErrorDetail

CORRELATION_REPORT_PATH = "/frameworks/pytorch/devices"
NO_NVML_INVENTORY_NOTE = "No NVML inventory to correlate with"
UUID_METHOD = "uuid"
PCI_METHOD = "pci_bus_id"
NO_MATCH = "NO_MATCH"
NO_IDENTIFIER = "NO_IDENTIFIER"
AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
UNIDENTIFIED_CANDIDATES = "UNIDENTIFIED_CANDIDATES"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PCI = re.compile(
    r"^(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})"
    r"\.(?P<function>[0-7])$"
)


def normalise_uuid(observed: Observed[str]) -> str | None:
    """Data model §5.4: lower-case, strip a leading `GPU-`, 8-4-4-4-12 hex; else unusable."""
    if not observed.is_collected or not isinstance(observed.value, str):
        return None
    text = observed.value.strip().lower().removeprefix("gpu-")
    return text if _UUID.match(text) else None


def normalise_pci_bus_id(observed: Observed[str]) -> tuple[int, int, int, int] | None:
    """`domain:bus:device.function` by value (NVML prints an 8-digit domain, PyTorch 4)."""
    if not observed.is_collected or not isinstance(observed.value, str):
        return None
    match = _PCI.match(observed.value.strip().lower())
    if match is None:
        return None
    parts = (int(match[name], 16) for name in ("domain", "bus", "device", "function"))
    domain, bus, device, function = parts
    return (domain, bus, device, function) if device <= 0x1F else None


@dataclass(frozen=True, slots=True)
class _Nvml:
    ref: str
    uuid: str | None
    pci: tuple[int, int, int, int] | None
    pci_text: str | None


@dataclass(frozen=True, slots=True)
class _Match:
    ref: str | None
    method: str | None
    code: str | None = None
    message: str | None = None
    note: str | None = None


class CorrelationCollector:
    name: ClassVar[str] = "correlation"

    def __init__(
        self,
        clock: Clock,
        *,
        inventory: GPUInventory,
        frameworks: FrameworkInfo,
        synthetic: bool = False,
    ) -> None:
        self._clock = clock
        self._inventory = inventory
        self._frameworks = frameworks
        self._prefix = FIXTURE_METHOD_PREFIX if synthetic else ""

    def collect(self) -> CollectorOutcome[FrameworkInfo]:
        started = self._clock.monotonic_ms()
        devices = self._frameworks.pytorch.devices
        if self._inventory.inventory_source is not Source.NVML:
            refs = [
                Observed.not_applicable(note=NO_NVML_INVENTORY_NOTE, source=Source.INSPECTOR)
                for _ in devices
            ]
        else:
            nvml = [_identified(d) for d in self._inventory.devices]
            matches = _unique([_match(device, nvml) for device in devices], devices)
            refs = [self._observed(m) for m in matches]
        section = self._with(refs)
        run = CollectorRun(
            collector=self.name,
            status=CollectorRunStatus.COMPLETED,
            duration_ms=max(0, self._clock.monotonic_ms() - started),
        )
        return CollectorOutcome(section=section, run=run)

    def unavailable(self, error: ErrorDetail) -> FrameworkInfo:
        """The guard's fallback: the PyTorch facts are another run's, so they're kept."""
        failed: Observed[str] = Observed(
            value=None,
            status=CollectionStatus.COLLECTION_FAILED,
            source=Source.INSPECTOR,
            method=None,
            error=error,
        )
        return self._with([failed for _ in self._frameworks.pytorch.devices])

    def _with(self, refs: list[Observed[str]]) -> FrameworkInfo:
        pytorch = self._frameworks.pytorch
        devices = tuple(
            replace(device, correlated_device_ref=ref)
            for device, ref in zip(pytorch.devices, refs, strict=True)
        )
        return FrameworkInfo(pytorch=replace(pytorch, devices=devices))

    def _observed(self, match: _Match) -> Observed[str]:
        if match.ref is not None:
            assert match.method is not None
            return Observed.collected(
                match.ref,
                source=Source.INSPECTOR,
                method=self._prefix + match.method,
                note=match.note,
            )
        assert match.code is not None and match.message is not None
        return Observed.failed(
            CollectionStatus.NOT_DETECTED,
            code=match.code,
            message=match.message,
            source=Source.INSPECTOR,
            method=None,
        )


def _identified(device: GPUDeviceInfo) -> _Nvml:
    pci = device.pci_bus_id
    return _Nvml(
        ref=device.device_ref,
        uuid=normalise_uuid(device.uuid),
        pci=normalise_pci_bus_id(pci),
        pci_text=pci.value if pci.is_collected else None,
    )


def _match(device: TorchDeviceInfo, nvml: list[_Nvml]) -> _Match:
    uuid = normalise_uuid(device.uuid)
    pci = normalise_pci_bus_id(device.pci_bus_id)
    if uuid is None and pci is None:
        return _Match(
            None,
            None,
            NO_IDENTIFIER,
            "PyTorch exposes no usable UUID or PCI bus ID for this device",
        )
    if uuid is not None:
        by_uuid = [n for n in nvml if n.uuid == uuid]
        if len(by_uuid) > 1:
            return _ambiguous(by_uuid, UUID_METHOD)
        if by_uuid:
            (found,) = by_uuid
            note = None
            if pci is not None and found.pci is not None and found.pci != pci:
                note = (
                    f"PCI bus IDs differ (NVML {found.pci_text}, PyTorch "
                    f"{device.pci_bus_id.value}); matched by UUID (DEC-09)"
                )
            return _Match(found.ref, UUID_METHOD, note=note)
    # A UUID rules a device out only when both sides have one and they differ.
    possible = [n for n in nvml if uuid is None or n.uuid is None]
    if pci is not None:
        by_pci = [n for n in possible if n.pci == pci]
        if len(by_pci) > 1:
            return _ambiguous(by_pci, PCI_METHOD)
        if by_pci:
            return _Match(by_pci[0].ref, PCI_METHOD)
    unidentified = [n.ref for n in possible if n.uuid is None and n.pci is None]
    if unidentified:
        return _Match(
            None,
            None,
            UNIDENTIFIED_CANDIDATES,
            "no identified NVML device matches; "
            f"{', '.join(unidentified)} exposes no usable identifier, so it can't be ruled out",
        )
    return _Match(None, None, NO_MATCH, "no NVML device has this device's UUID or PCI bus ID")


def _ambiguous(candidates: list[_Nvml], method: str) -> _Match:
    refs = ", ".join(n.ref for n in candidates)
    return _Match(
        None,
        None,
        AMBIGUOUS_MATCH,
        f"several NVML devices match by {method} ({refs}); none is chosen",
    )


def _unique(matches: list[_Match], devices: tuple[TorchDeviceInfo, ...]) -> list[_Match]:
    """An NVML device claimed by several PyTorch devices is none of theirs."""
    claims = Counter(m.ref for m in matches if m.ref is not None)
    result = []
    for match, device in zip(matches, devices, strict=True):
        if match.ref is not None and claims[match.ref] > 1:
            others = [
                f"cuda:{d.torch_index}"
                for m, d in zip(matches, devices, strict=True)
                if m.ref == match.ref and d is not device
            ]
            match = _Match(
                None,
                None,
                AMBIGUOUS_MATCH,
                f"{match.ref} is also matched by {', '.join(others)}; none is chosen",
            )
        result.append(match)
    return result
