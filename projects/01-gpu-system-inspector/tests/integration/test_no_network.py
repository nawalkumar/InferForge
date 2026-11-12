"""AC42 (NFR-15): a full live inspection makes no network access.

Every socket entry point is patched to raise and record the attempt, then the production
pipeline runs end to end on this host: collection, Mode B readiness, diagnostics, redaction
and both renderings. The patch covers this process only; the child processes an inspection
may start are listed in AC43 (`nvidia-smi`, `nvcc`, the PyTorch probe).
"""

import socket

import pytest

from gpu_system_inspector import api
from gpu_system_inspector.models import InspectionOptions
from gpu_system_inspector.reporting import redact, render_json, render_text

pytestmark = pytest.mark.host

SOCKET_ENTRY_POINTS = (
    "socket",
    "create_connection",
    "create_server",
    "getaddrinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
)


def test_a_full_inspection_never_attempts_network_access(monkeypatch):
    attempts: list[str] = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise OSError(f"network access attempted: socket.{name}")

        return call

    for name in SOCKET_ENTRY_POINTS:
        monkeypatch.setattr(socket, name, refuse(name))

    report = api.inspect(InspectionOptions(required_vram_gib=8))
    render_text(redact(report))
    render_json(report)

    assert attempts == []
    assert report.readiness is not None and report.diagnostics
    assert not any("network access attempted" in e.message for e in report.collection_errors)
