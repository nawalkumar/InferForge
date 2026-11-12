"""The human-readable report (CLI spec §5.2, architecture §4.10, IMP-P03).

`render_text(report)` is a pure function of a validated report — the same report the JSON
form serialises, so a redacted report renders redacted text. Fixed section order; sections
that weren't collected still print with their status. Collected values show a human unit
and their source; missing values show status and reason, never a number. No ANSI. Non-LIVE
reports open and close with the fixture banner.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from typing import Any

from gpu_system_inspector.models import (
    DataOrigin,
    DiagnosticFinding,
    DiagnosticStatus,
    EvidenceKind,
    InspectionReport,
    Observed,
)

FIXTURE_BANNER = "*** FIXTURE DATA — NOT A REAL MEASUREMENT ***"
WIDTH = 100
LABEL = 24
MISSING = "—"
STATUS_RANK = {
    DiagnosticStatus.FAIL: 0,
    DiagnosticStatus.WARNING: 1,
    DiagnosticStatus.UNKNOWN: 2,
    DiagnosticStatus.INFO: 3,
    DiagnosticStatus.PASS: 4,
    DiagnosticStatus.NOT_APPLICABLE: 5,
}
_GIB = 1 << 30


def _bytes(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


def _value(o: Observed[Any]) -> str:
    v = o.value
    if isinstance(v, bool):
        text = "yes" if v else "no"
    elif o.unit == "bytes" and isinstance(v, int):
        text = _bytes(v)
    elif o.unit == "percent":
        text = f"{v} %"
    elif o.unit == "celsius":
        text = f"{v} °C"
    elif o.unit == "watts":
        text = f"{v:.1f} W" if isinstance(v, float) else f"{v} W"
    else:
        text = str(v)
    return f"{text} ({o.source})" if o.source is not None else text


def show(o: Observed[Any] | None, suffix: str = "") -> str:
    """`24.0 GiB free (NVML)` when collected; `— NOT_SUPPORTED (reason)` otherwise."""
    if o is None:
        return f"{MISSING}  NOT_INSPECTED"
    if o.is_collected:
        if o.source is not None and suffix:
            return _value(o).replace(f" ({o.source})", f"{suffix} ({o.source})")
        return _value(o) + suffix
    reason = ""
    if o.error is not None:
        reason = f" ({o.error.code}: {o.error.message})"
    elif o.note:
        reason = f" ({o.note})"
    return f"{MISSING}  {o.status}{reason}"


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(text.rstrip())

    def field(self, label: str, text: str, indent: int = 2) -> None:
        head = " " * indent + f"{label:<{LABEL - 1}} "
        self.wrap(text, head, " " * len(head))

    def wrap(self, text: str, first: str, rest: str) -> None:
        lines = textwrap.wrap(
            text,
            WIDTH,
            initial_indent=first,
            subsequent_indent=rest,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [first]
        self.lines.extend(line.rstrip() for line in lines)

    def section(self, title: str) -> None:
        if self.lines and self.lines[-1]:
            self.line()
        self.line(title)


def _system(out: _Out, r: InspectionReport) -> None:
    s = r.system
    out.section("SYSTEM")
    out.field("Platform", str(s.platform_profile))
    out.field("OS", show(s.os_name))
    out.field("OS release", show(s.os_release))
    out.field("OS version", show(s.os_version))
    out.field("Distribution", show(s.os_distribution))
    out.field("Linux environment", show(s.linux_environment))
    out.field("Architecture", show(s.architecture))
    out.field("Processor", show(s.processor_model))
    out.field("Logical CPUs", show(s.logical_cpu_count))
    out.field("Memory total", show(s.memory_total_bytes))
    out.field("Memory available", show(s.memory_available_bytes))
    out.field("Python", f"{show(s.python_version)} · {show(s.python_implementation)}")
    out.field("Python executable", show(s.python_executable))


def _devices(out: _Out, r: InspectionReport) -> None:
    g = r.gpu
    out.section("DEVICES")
    source = f" via {g.inventory_source}" if g.inventory_source is not None else ""
    out.field("NVIDIA devices", show(g.device_count) + source)
    for d in g.devices:
        out.line(f"  {d.device_ref}  {show(d.name)}")
        out.field("UUID", show(d.uuid), 4)
        out.field("PCI bus ID", show(d.pci_bus_id), 4)
        out.field("Total memory", show(d.memory_total_bytes), 4)
        out.field("Compute capability", show(d.compute_capability), 4)
        out.field("MIG mode", show(d.mig_mode), 4)
    pci = g.pci_observations
    if pci.is_collected and pci.value is not None:
        nvidia = [o for o in pci.value if o.is_nvidia]
        out.field("PCI scan", f"{len(pci.value)} display device(s), {len(nvidia)} NVIDIA (OS)")
        for o in pci.value:
            vendor = "NVIDIA" if o.is_nvidia else o.vendor_id
            out.field(
                f"PCI {o.address}", f"{vendor} {o.device_id} · driver {show(o.kernel_driver)}", 4
            )
    else:
        out.field("PCI scan", show(pci))


def _driver(out: _Out, r: InspectionReport) -> None:
    d = r.driver
    out.section("DRIVER & CUDA")
    out.field("Driver version", show(d.driver_version))
    out.field("NVML version", show(d.nvml_version))
    binding = show(d.nvml_binding)
    if d.nvml_binding_distribution.is_collected:
        binding += f" · distribution {d.nvml_binding_distribution.value}"
    out.field("NVML binding", binding)
    cuda = d.driver_supported_cuda_version
    out.field(
        "Driver CUDA",
        f"Driver supports CUDA up to {_value(cuda)}" if cuda.is_collected else show(cuda),
    )
    out.field("CUDA driver library", show(d.cuda_driver_library))
    out.field("nvidia-smi", show(d.nvidia_smi_path))
    out.field("nvidia-smi driver", show(d.nvidia_smi_driver_version))
    out.field("nvidia-smi CUDA", show(d.nvidia_smi_cuda_version))


def _toolkit(out: _Out, r: InspectionReport) -> None:
    c = r.cuda_environment
    out.section("CUDA TOOLKIT")
    toolkits = c.toolkits
    if toolkits.is_collected and toolkits.value is not None:
        if not toolkits.value:
            out.field("Toolkits", "none found (CUDA_TOOLKIT)")
        for t in toolkits.value:
            version = (
                f"CUDA Toolkit {_value(t.version)}"
                if t.version.is_collected
                else f"CUDA Toolkit {show(t.version)}"
            )
            via = ", ".join(str(v) for v in t.discovered_via)
            out.field("Toolkit", f"{version} at {t.root_path} (found via {via})")
            out.field("nvcc", show(t.nvcc_path), 4)
    else:
        out.field("Toolkits", show(toolkits))
    nvcc = c.active_nvcc_version
    out.field(
        "Active nvcc", f"CUDA Toolkit {nvcc.value} (nvcc)" if nvcc.is_collected else show(nvcc)
    )
    out.field("Active nvcc path", show(c.active_nvcc_path))
    for name, value in sorted(c.environment.items()):
        out.field(name, show(value))


def _pytorch(out: _Out, r: InspectionReport) -> None:
    p = r.frameworks.pytorch
    out.section("PYTORCH  (in this Python environment)")
    out.field("Status", str(p.status))
    out.field("Version", show(p.version))
    out.field("Distribution version", show(p.distribution_version))
    out.field("CUDA build", show(p.cuda_build))
    build = p.build_cuda_version
    out.field(
        "Build CUDA",
        f"PyTorch built with CUDA {_value(build)}" if build.is_collected else show(build),
    )
    out.field("ROCm build", show(p.rocm_build))
    out.field("cuDNN", show(p.cudnn_version))
    out.field("CUDA available", show(p.cuda_available))
    out.field("Device count", show(p.device_count))
    for dev in p.devices:
        out.line(f"  cuda:{dev.torch_index}  {show(dev.name)}")
        out.field("NVML device", show(dev.correlated_device_ref), 4)
        out.field("Total memory", show(dev.total_memory_bytes), 4)
        out.field("Compute capability", show(dev.compute_capability), 4)
    for message in p.cuda_init_warnings:
        out.field("CUDA init warning", message)
    out.field("Probe", f"{p.probe.mode} · protocol {p.probe.protocol_version}")


def _runtime(out: _Out, r: InspectionReport) -> None:
    out.section("RUNTIME SNAPSHOT")
    if not r.runtime:
        out.field("Devices", "none inventoried")
    for m in r.runtime:
        out.line(f"  {m.device_ref}  sampled at {m.sampled_at.isoformat()}")
        out.field("GPU utilization", show(m.gpu_utilization_percent), 4)
        out.field("Memory bus activity", show(m.memory_bus_utilization_percent), 4)
        out.field("Memory total", show(m.memory_total_bytes), 4)
        out.field("Memory used", show(m.memory_used_bytes, " used"), 4)
        out.field("Memory free", show(m.memory_free_bytes, " free"), 4)
        out.field("Memory reserved", show(m.memory_reserved_bytes), 4)
        out.field("Temperature", show(m.temperature_gpu_celsius), 4)
        out.field("Slowdown threshold", show(m.temperature_slowdown_threshold_celsius), 4)
        out.field("Power draw", show(m.power_draw_watts), 4)
        out.field("Power limit", show(m.power_limit_watts), 4)
        procs = m.processes
        if procs.is_collected and procs.value is not None:
            out.field("Processes", f"{len(procs.value)} ({procs.source})", 4)
            for proc in procs.value:
                out.field(
                    f"pid {proc.pid}",
                    f"{proc.kind} · {show(proc.name)} · {show(proc.used_memory_bytes)}",
                    6,
                )
        else:
            out.field("Processes", show(procs), 4)


def _readiness(out: _Out, r: InspectionReport) -> None:
    res = r.readiness
    if res is None:
        return
    out.section(
        f"READINESS  required free memory: {_bytes(res.required_bytes)} "
        f"({res.required_bytes} bytes)"
    )
    names = {d.device_ref: d.name.value for d in r.gpu.devices}
    for d in res.devices:
        detail = f"free {show(d.free_bytes)}"
        if d.margin_bytes is not None:
            sign = "margin +" if d.margin_bytes >= 0 else "short by "
            detail += f", {sign}{_bytes(abs(d.margin_bytes))}"
        out.wrap(
            f"{d.device_ref}  {names.get(d.device_ref) or ''}  {d.status}  {detail} — "
            f"{d.explanation}",
            "  ",
            "      ",
        )
    out.line(f"  overall: {res.overall_status} ({res.overall_reason})")
    for text in res.limitations:
        out.wrap(text, "  limitation: ", "      ")
    out.wrap(f"Note: {res.disclaimer}", "  ", "  ")


def _finding(out: _Out, f: DiagnosticFinding) -> None:
    device = f" ({f.device_ref})" if f.device_ref else ""
    out.line(f"  [{f.status}] {f.check_id} {f.title}{device}")
    out.wrap(f.explanation, "      ", "      ")
    for e in f.evidence:
        tag = "observed:" if e.kind is EvidenceKind.OBSERVATION else "inferred:"
        source = (
            f" ({e.source})" if e.source is not None and e.kind is EvidenceKind.OBSERVATION else ""
        )
        out.wrap(f"{e.statement}{source}", f"      {tag} ", "                ")
    for rec in f.recommendations:
        out.wrap(rec.action, "      suggestion: ", "                  ")
        if rec.reference_url:
            out.line(f"      {rec.reference_url}")
    out.line(f"      confidence: {f.confidence} · severity: {f.severity}")
    for text in f.limitations:
        out.wrap(text, "      limitation: ", "                  ")


def ordered_findings(findings: Iterable[DiagnosticFinding]) -> list[DiagnosticFinding]:
    return sorted(findings, key=lambda f: (STATUS_RANK[f.status], f.check_id, f.device_ref or ""))


def render_text(report: InspectionReport) -> str:
    out = _Out()
    fixture = report.data_origin is not DataOrigin.LIVE
    if fixture:
        out.line(FIXTURE_BANNER)
    app = report.application
    out.line(
        f"{app.display_name} {app.version} · InferForge · report schema {report.schema_version}"
    )
    generated = report.generated_at.isoformat().replace("+00:00", "Z")
    out.line(f"Generated {generated} · mode {report.mode} · data origin {report.data_origin}")
    if report.redaction.applied:
        cats = ", ".join(str(c) for c in report.redaction.categories)
        out.line(f"Redacted: {cats}")
    _system(out, report)
    _devices(out, report)
    _driver(out, report)
    _toolkit(out, report)
    _pytorch(out, report)
    _runtime(out, report)
    _readiness(out, report)
    out.section("FINDINGS")
    for f in ordered_findings(report.diagnostics):
        _finding(out, f)
    out.section("UNSUPPORTED")
    if not report.unsupported_capabilities:
        out.line("  none")
    for u in report.unsupported_capabilities:
        device = f" ({u.device_ref})" if u.device_ref else ""
        out.wrap(f"{u.capability}{device}: {u.status} — {u.reason}", "  ", "      ")
    out.section("COLLECTION ERRORS")
    if not report.collection_errors:
        out.line("  none")
    for e in report.collection_errors:
        device = f" ({e.device_ref})" if e.device_ref else ""
        out.wrap(f"{e.collector}{device}: {e.status} {e.error_code} — {e.message}", "  ", "      ")
    out.section("LIMITATIONS")
    for text in report.limitations:
        out.wrap(text, "  - ", "    ")
    if fixture:
        out.line()
        out.line(FIXTURE_BANNER)
    return "\n".join(out.lines) + "\n"
