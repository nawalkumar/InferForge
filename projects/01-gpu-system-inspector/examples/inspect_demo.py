"""End-to-end demonstration of the public inspection API (IMP-K08, IMP-K02, IMP-P01).

Runs one live, read-only inspection with the production adapters, validates the report
against the shipped JSON Schema, checks that it round-trips through JSON, and prints a
summary of collector runs, errors and unsupported capabilities. It needs no NVIDIA GPU.

    uv run python examples/inspect_demo.py            # summary only
    uv run python examples/inspect_demo.py --json -   # also print the JSON report
    uv run python examples/inspect_demo.py --json report.json

Nothing is written unless `--json PATH` is given. Not part of the shipped package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpu_system_inspector import api
from gpu_system_inspector.models import InspectionReport, Observed
from gpu_system_inspector.reporting import load_report_schema, parse_report_json, render_json


def _status(value: Observed) -> str:
    if value.is_collected:
        shown = getattr(value.value, "value", value.value)  # enums by their value
        return f"COLLECTED ({shown!r})"
    code = f" / {value.error.code}" if value.error else ""
    return f"{value.status.value}{code}"


def _validate(report: InspectionReport, text: str) -> str:
    if parse_report_json(text) != report:
        raise SystemExit("report did not round-trip through JSON")
    try:
        import jsonschema
    except ImportError:
        return "round trip OK; JSON Schema check skipped (jsonschema not installed)"
    jsonschema.validate(json.loads(text), load_report_schema())
    return "valid against the shipped JSON Schema; round trip OK"


def summarise(report: InspectionReport, validation: str) -> str:
    system, gpu = report.system, report.gpu
    lines = [
        f"GPU System Inspector {report.application.version}, "
        f"data origin {report.data_origin.value}",
        f"Report: schema {report.schema_version}, {validation}",
        f"Platform: {system.platform_profile.value}; "
        f"Linux environment: {_status(system.linux_environment)}",
        f"OS: {_status(system.os_name)} {_status(system.os_release)}",
        f"PCI scan: {_status(gpu.pci_observations)}",
        f"GPU inventory: device count {_status(gpu.device_count)}, {len(gpu.devices)} device(s)",
        f"Driver: {_status(report.driver.driver_version)}; "
        f"supports CUDA up to {_status(report.driver.driver_supported_cuda_version)}; "
        f"nvidia-smi {_status(report.driver.nvidia_smi_path)}",
        f"Runtime metrics: {len(report.runtime)} device record(s)",
        f"CUDA Toolkit: {_status(report.cuda_environment.toolkits)}; "
        f"active nvcc {_status(report.cuda_environment.active_nvcc_path)} "
        "(an installation fact, not a CUDA execution check)",
        f"PyTorch: {report.frameworks.pytorch.status.value}; "
        f"CUDA build {_status(report.frameworks.pytorch.cuda_build)}; "
        f"cuda_available {_status(report.frameworks.pytorch.cuda_available)} "
        "(PyTorch's answer; no CUDA code was run)",
        "Collector runs:",
        *(
            f"  {r.collector:<16} {r.status.value:<10} {r.duration_ms} ms"
            for r in report.collector_runs
        ),
        f"Collection errors: {len(report.collection_errors)}",
        *(
            f"  {e.collector}: {e.status.value} {e.error_code} ({e.report_path}) {e.message}"
            for e in report.collection_errors
        ),
        f"Unsupported capabilities: {len(report.unsupported_capabilities)}",
        *(f"  {u.capability}: {u.reason}" for u in report.unsupported_capabilities),
        "Limitations:",
        *(f"  - {text}" for text in report.limitations),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", metavar="PATH", help="write the JSON report to PATH ('-' for stdout)"
    )
    args = parser.parse_args(argv)
    report = api.inspect()
    text = render_json(report)
    print(
        summarise(report, _validate(report, text)),
        file=sys.stderr if args.json == "-" else sys.stdout,
    )
    if args.json == "-":
        sys.stdout.write(text)
    elif args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
