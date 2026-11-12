"""Collect a hardware validation evidence bundle for GPU System Inspector (Project 01).

Run it with the Python environment that has the inspector installed, on the machine under
test (Google Colab with a GPU runtime, or native Linux with an NVIDIA GPU):

    python validation/hardware_evidence.py --label "Colab T4" --out gsi-evidence

It writes one directory with:

- the inspector's own reports (real CLI, `data_origin` LIVE): Mode A text and JSON, the same
  redacted, and Mode B (`--required-vram`), each with exit code, stderr and wall time;
- reference outputs taken **independently** of the inspector: `nvidia-smi` (header, `-L`,
  `--query-gpu`), `nvcc --version`, `/proc/driver/nvidia/version`, and PyTorch's own answers
  from a separate child process (no tensor is allocated);
- the hardware tests (`pytest -m hardware tests/hardware`, junit XML and log);
- `summary.json` and `SUMMARY.md`: every observed field classified as supported and
  observed, supported but unavailable, not supported on this environment, not implemented,
  or failed; readiness; findings; hardware test outcomes; the verdict.

The verdict is never `HARDWARE_VALIDATED`: it is `BLOCKED` when no NVIDIA GPU is accessible,
otherwise `EVIDENCE_COLLECTED`, pending the owner's review (docs/hardware_validation.md).

`--cuda-smoke-test` adds an **optional CUDA execution check** after everything else: a
separate child process runs a tiny PyTorch computation on `cuda:0`. It is recorded in its
own section, never feeds the inspector's report, and is off by default.
`--allow-cuda-allocation` lets hardware test HW-12 allocate GPU memory in a child process.

The bundle contains GPU UUIDs, PCI bus IDs and paths in the unredacted files; share
`SUMMARY.md` and the `*-redacted.*` files publicly, and the rest only privately.
Exit status: 0 bundle written, 1 bundle written but a check failed, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "gpu_system_inspector"]
TIMEOUT_S = 300
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
DISTRIBUTIONS = (
    "01-gpu-system-inspector",
    "nvidia-ml-py",
    "pynvml",
    "torch",
    "pytest",
    "jsonschema",
)

OBSERVED = "SUPPORTED_AND_OBSERVED"
UNAVAILABLE = "SUPPORTED_BUT_UNAVAILABLE"
NOT_SUPPORTED = "NOT_SUPPORTED_ON_THIS_ENVIRONMENT"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
FAILED = "FAILED"
SKIPPED = "SKIPPED_BY_OPTION"
NOT_APPLICABLE = "NOT_APPLICABLE"
NO_DATA = "NO_DATA"
CATEGORY_BY_STATUS = {
    "COLLECTED": OBSERVED,
    "NOT_DETECTED": UNAVAILABLE,
    "NOT_INSTALLED": UNAVAILABLE,
    "PERMISSION_DENIED": UNAVAILABLE,
    "NOT_SUPPORTED": NOT_SUPPORTED,
    "COLLECTION_FAILED": FAILED,
    "UNKNOWN": FAILED,
    "SKIPPED": SKIPPED,
    "NOT_APPLICABLE": NOT_APPLICABLE,
}

FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GPU model and count", ("/gpu/device_count", "/gpu/devices/*/name")),
    ("NVML initialisation", ("/driver/nvml_version",)),
    (
        "Device inventory",
        (
            "/gpu/devices/*/memory_total_bytes",
            "/gpu/devices/*/compute_capability",
            "/gpu/devices/*/mig_mode",
        ),
    ),
    ("GPU UUID", ("/gpu/devices/*/uuid",)),
    ("PCI bus identity", ("/gpu/devices/*/pci_bus_id", "/gpu/pci_observations")),
    ("Driver version", ("/driver/driver_version", "/driver/nvidia_smi_driver_version")),
    (
        "Driver-side CUDA (supported version, libcuda)",
        (
            "/driver/driver_supported_cuda_version",
            "/driver/cuda_driver_library",
            "/driver/nvidia_smi_cuda_version",
        ),
    ),
    ("CUDA Toolkit", ("/cuda_environment/toolkits", "/cuda_environment/active_nvcc_version")),
    (
        "PyTorch installation and build",
        (
            "/frameworks/pytorch/version",
            "/frameworks/pytorch/cuda_build",
            "/frameworks/pytorch/build_cuda_version",
            "/frameworks/pytorch/cudnn_version",
        ),
    ),
    (
        "PyTorch CUDA availability",
        ("/frameworks/pytorch/cuda_available", "/frameworks/pytorch/device_count"),
    ),
    ("Device correlation", ("/frameworks/pytorch/devices/*/correlated_device_ref",)),
    (
        "Runtime observations",
        (
            "/runtime/*/gpu_utilization_percent",
            "/runtime/*/memory_bus_utilization_percent",
            "/runtime/*/memory_used_bytes",
            "/runtime/*/memory_free_bytes",
            "/runtime/*/temperature_gpu_celsius",
            "/runtime/*/power_draw_watts",
            "/runtime/*/processes",
        ),
    ),
    (
        "System and software versions",
        (
            "/system/os_name",
            "/system/os_release",
            "/system/linux_environment",
            "/system/python_version",
            "/driver/nvml_binding",
        ),
    ),
)

LEGEND = (
    "Classification: SUPPORTED_AND_OBSERVED = collected (COLLECTED); SUPPORTED_BUT_UNAVAILABLE"
    " = the inspector supports it but it wasn't there or wasn't accessible (NOT_DETECTED,"
    " NOT_INSTALLED, PERMISSION_DENIED); NOT_SUPPORTED_ON_THIS_ENVIRONMENT = NOT_SUPPORTED by"
    " the platform or driver; NOT_IMPLEMENTED = NOT_SUPPORTED with code NOT_IMPLEMENTED;"
    " FAILED = COLLECTION_FAILED or UNKNOWN; SKIPPED_BY_OPTION and NOT_APPLICABLE are neither;"
    " NO_DATA = no field to classify (for example, no device was inventoried).",
)

NOT_IMPLEMENTED_FEATURES = (
    ("Graphics process listing (only compute processes are listed)", "KI-36, FR-11"),
    ("CUDA runtime library (libcudart) version", "not collected; see the driver-side fields"),
    ("MIG instance enumeration", "deferred, product specification §7"),
    ("NVML clock event (throttle) reasons", "deferred, product specification §7"),
    ("Windows default CUDA Toolkit location", "KI-37, owner decision"),
    ("Container detection", "deferred, product specification §7"),
)

TORCH_REFERENCE = """
import json, torch
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print(json.dumps({
    "version": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": available,
    "device_count": count,
    "device_names": [torch.cuda.get_device_name(i) for i in range(count)],
}))
"""
CUDA_SMOKE = """
import json, torch
x = torch.arange(1024, dtype=torch.float32, device="cuda:0")
result = float((x * 2).sum())
torch.cuda.synchronize()
print(json.dumps({"device": torch.cuda.get_device_name(0), "result": result}))
"""
CUDA_SMOKE_EXPECTED = 1047552.0


# --- classification ------------------------------------------------------------------------


def classify(observed: dict[str, Any]) -> str:
    code = (observed.get("error") or {}).get("code")
    if observed["status"] == "NOT_SUPPORTED" and code == "NOT_IMPLEMENTED":
        return NOT_IMPLEMENTED
    return CATEGORY_BY_STATUS.get(observed["status"], FAILED)


def resolve(data: Any, pattern: str) -> list[tuple[str, Any]]:
    """Expand a JSON pointer whose `*` segments match every list index."""
    found = [("", data)]
    for part in pattern.strip("/").split("/"):
        step = []
        for pointer, node in found:
            if part == "*" and isinstance(node, list):
                step += [(f"{pointer}/{i}", item) for i, item in enumerate(node)]
            elif isinstance(node, dict) and part in node:
                step.append((f"{pointer}/{part}", node[part]))
        found = step
    return found


def feature_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature, patterns in FEATURES:
        fields = [
            {
                "pointer": pointer,
                "status": node["status"],
                "category": classify(node),
                "error_code": (node.get("error") or {}).get("code"),
            }
            for pattern in patterns
            for pointer, node in resolve(report, pattern)
            if isinstance(node, dict) and "status" in node
        ]
        categories = sorted({f["category"] for f in fields})
        rows.append(
            {
                "feature": feature,
                "category": categories[0] if len(categories) == 1 else "MIXED",
                "categories": categories or [NO_DATA],
                "fields": fields,
            }
        )
    return rows


def nvidia_accessible(report: dict[str, Any]) -> bool:
    gpu = report["gpu"]
    return gpu["device_count"]["status"] == "COLLECTED" and (gpu["device_count"]["value"] or 0) > 0


def verdict(report: dict[str, Any] | None) -> str:
    if report is None:
        return "BLOCKED: the inspector produced no report"
    if not nvidia_accessible(report):
        return "BLOCKED: no NVIDIA GPU is accessible in this environment"
    return "EVIDENCE_COLLECTED: owner review required before any HW item is HARDWARE_VALIDATED"


def parse_junit(path: Path) -> list[dict[str, Any]]:
    results = []
    for case in ET.parse(path).getroot().iter("testcase"):
        outcome, message = "passed", None
        for tag in ("failure", "error", "skipped"):
            element = case.find(tag)
            if element is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = element.get("message")
                break
        properties = {p.get("name"): p.get("value") for p in case.iter("property")}
        results.append(
            {"test": case.get("name"), "outcome": outcome, "message": message, **properties}
        )
    return results


# --- execution -----------------------------------------------------------------------------


def run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=cwd, env=env
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}", "returncode": None}
    return {
        "argv": argv,
        "returncode": result.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in DISTRIBUTIONS:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def environment(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "collected_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "google_colab": has_module("google.colab"),
        "distributions": versions(),
    }


def reference_tools(out: Path, include_torch: bool) -> dict[str, Any]:
    smi = shutil.which("nvidia-smi") or (WSL_NVIDIA_SMI if Path(WSL_NVIDIA_SMI).exists() else None)
    commands: dict[str, list[str]] = {}
    if smi:
        commands["nvidia-smi"] = [smi]
        commands["nvidia-smi-L"] = [smi, "-L"]
        commands["nvidia-smi-query"] = [
            smi,
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,"
            "memory.free,temperature.gpu,utilization.gpu",
            "--format=csv",
        ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        commands["nvcc-version"] = [nvcc, "--version"]
    if include_torch:
        commands["pytorch-reference"] = [sys.executable, "-c", TORCH_REFERENCE]
    results: dict[str, Any] = {name: run(argv) for name, argv in commands.items()}
    if not smi:
        results["nvidia-smi"] = {"error": "nvidia-smi not found on PATH or /usr/lib/wsl/lib"}
    if not nvcc:
        results["nvcc-version"] = {"error": "nvcc not found on PATH"}
    proc_version = Path("/proc/driver/nvidia/version")
    results["proc-driver-nvidia-version"] = (
        {"stdout": proc_version.read_text(errors="replace")}
        if proc_version.exists()
        else {"error": f"{proc_version} does not exist"}
    )
    (out / "reference").mkdir(exist_ok=True)
    for name, result in results.items():
        (out / "reference" / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return {name: r.get("returncode", r.get("error")) for name, r in results.items()}


def inspector_run(out: Path, name: str, args: list[str]) -> dict[str, Any]:
    json_path = out / f"{name}.json"
    result = run([*CLI, "inspect", *args, "--json", str(json_path)])
    (out / f"{name}.txt").write_text(result.get("stdout", ""))
    (out / f"{name}.stderr.txt").write_text(result.get("stderr", result.get("error", "")))
    report = json.loads(json_path.read_text()) if json_path.exists() else None
    return {
        "args": args,
        "exit_code": result.get("returncode"),
        "seconds": result.get("seconds"),
        "traceback_on_stderr": "Traceback" in result.get("stderr", ""),
        "report": report,
    }


def schema_check(report: dict[str, Any] | None) -> str:
    if report is None:
        return "no report"
    try:
        import jsonschema

        from gpu_system_inspector.reporting import load_report_schema
    except ImportError as exc:
        return f"not checked ({exc})"
    try:
        jsonschema.validate(report, load_report_schema())
    except jsonschema.ValidationError as exc:
        return f"INVALID: {exc.message}"
    return "valid"


def hardware_tests(out: Path, allow_allocation: bool) -> dict[str, Any]:
    if importlib.util.find_spec("pytest") is None:
        return {"error": "pytest is not installed in this environment", "results": []}
    if not (PROJECT / "tests" / "hardware").is_dir():
        return {"error": f"{PROJECT / 'tests' / 'hardware'} not found", "results": []}
    junit = out / "hardware-tests.xml"
    env = None
    if allow_allocation:
        env = {**os.environ, "GSI_HW_ALLOW_CUDA_ALLOCATION": "1"}
    result = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "hardware",
            "tests/hardware",
            "-rA",
            "-p",
            "no:cacheprovider",
            "-o",
            "junit_family=xunit1",
            f"--junitxml={junit}",
        ],
        cwd=PROJECT,
        env=env,
    )
    (out / "hardware-tests.log").write_text(result.get("stdout", "") + result.get("stderr", ""))
    results = parse_junit(junit) if junit.exists() else []
    return {"exit_code": result.get("returncode"), "results": results}


def cuda_smoke_test(include_torch: bool) -> dict[str, Any]:
    if not include_torch:
        return {"outcome": "not run", "reason": "PyTorch is not installed"}
    result = run([sys.executable, "-c", CUDA_SMOKE])
    try:
        payload = json.loads(result["stdout"].strip().splitlines()[-1])
    except (KeyError, IndexError, ValueError):
        return {"outcome": "failed", "returncode": result.get("returncode"), **result}
    ok = payload.get("result") == CUDA_SMOKE_EXPECTED
    return {"outcome": "passed" if ok else "failed", "expected": CUDA_SMOKE_EXPECTED, **payload}


# --- summary -------------------------------------------------------------------------------


def summary_markdown(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    lines = [
        "# GPU System Inspector — hardware validation evidence",
        "",
        f"- Label: {env['label']}",
        f"- Collected: {env['collected_at']}",
        f"- Platform: {env['platform']} ({env['machine']}); Google Colab: {env['google_colab']}",
        f"- Python: {env['python']} ({env['implementation']})",
        "- Distributions: "
        + ", ".join(f"{k} {v or 'not installed'}" for k, v in env["distributions"].items()),
        "",
        f"**Verdict: {summary['verdict']}**",
        "",
        "## Inspector runs (real CLI, data origin LIVE)",
        "",
        "| Run | Arguments | Exit | Seconds | Schema | Traceback |",
        "| --- | --------- | ---- | ------- | ------ | --------- |",
    ]
    for name, r in summary["inspector_runs"].items():
        lines.append(
            f"| {name} | `{' '.join(r['args']) or '(none)'}` | {r['exit_code']} | {r['seconds']} "
            f"| {r['schema']} | {'yes' if r['traceback_on_stderr'] else 'no'} |"
        )
    lines += [
        "",
        "## Features (Mode A report)",
        "",
        "| Feature | Classification | Fields |",
        "| ------- | -------------- | ------ |",
    ]
    for row in summary["features"]:
        detail = "; ".join(
            f"{f['pointer']} {f['status']}" + (f" ({f['error_code']})" if f["error_code"] else "")
            for f in row["fields"]
        )
        lines.append(f"| {row['feature']} | {', '.join(row['categories'])} | {detail or '—'} |")
    lines += ["", *LEGEND, "", "Not implemented by the inspector (never reported as observed):", ""]
    lines += [f"- {name} ({ref})" for name, ref in NOT_IMPLEMENTED_FEATURES]
    readiness = summary.get("readiness")
    lines += ["", "## Readiness (Mode B)", ""]
    if readiness:
        lines.append(
            f"Required {readiness['required_vram_gib']} GiB: **{readiness['overall_status']}** "
            f"({readiness['overall_reason']})"
        )
        lines += [
            f"- {d['device_ref']}: {d['status']} ({d['reason']})" for d in readiness["devices"]
        ]
    else:
        lines.append("No Mode B report.")
    lines += ["", "## Findings (Mode A)", "", "| Check | Device | Status | Confidence |"]
    lines.append("| ----- | ------ | ------ | ---------- |")
    for f in summary["findings"]:
        lines.append(
            f"| {f['check_id']} | {f['device_ref'] or '—'} | {f['status']} | {f['confidence']} |"
        )
    tests = summary["hardware_tests"]
    lines += ["", "## Hardware tests (`pytest -m hardware`)", ""]
    if tests.get("error"):
        lines.append(f"Not run: {tests['error']}")
    lines += ["| Test | Outcome | Message |", "| ---- | ------- | ------- |"]
    for t in tests.get("results", []):
        message = (t["message"] or "").replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {t['test']} | {t['outcome']} | {message} |")
    lines += ["", "A skipped hardware test is not a pass.", "", "## Reference tools", ""]
    lines += [f"- {name}: {status}" for name, status in summary["reference_tools"].items()]
    smoke = summary["cuda_smoke_test"]
    lines += [
        "",
        "## Optional CUDA execution check (not part of the inspection)",
        "",
        f"Outcome: {smoke['outcome']}" + (f" — {smoke['reason']}" if smoke.get("reason") else ""),
        "",
        "## Sharing",
        "",
        "Unredacted files (`inspect-modeA.*`, `inspect-modeB.*`, `reference/`, `hardware-tests.*`)"
        " contain GPU UUIDs, PCI bus IDs and paths. Share them privately. Failure messages above"
        " are copied verbatim; review them before posting this file publicly.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output directory (default: gsi-evidence-*)")
    parser.add_argument("--label", default="unlabelled", help="machine label, e.g. 'Colab T4'")
    parser.add_argument("--required-vram", default="1", help="Mode B requirement in GiB")
    parser.add_argument("--skip-hardware-tests", action="store_true")
    parser.add_argument("--cuda-smoke-test", action="store_true", help="opt-in CUDA execution")
    parser.add_argument("--allow-cuda-allocation", action="store_true", help="enable HW-12")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("gpu_system_inspector") is None:
        print(
            "error: gpu_system_inspector is not installed in this Python environment; "
            "install it first (docs/hardware_validation.md)",
            file=sys.stderr,
        )
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"gsi-evidence-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    env = environment(args.label)
    include_torch = env["distributions"]["torch"] is not None
    runs = {
        "inspect-modeA": inspector_run(out, "inspect-modeA", []),
        "inspect-modeA-redacted": inspector_run(out, "inspect-modeA-redacted", ["--redact"]),
        "inspect-modeB": inspector_run(
            out, "inspect-modeB", ["--required-vram", args.required_vram]
        ),
    }
    for r in runs.values():
        r["schema"] = schema_check(r["report"])
    mode_a, mode_b = runs["inspect-modeA"]["report"], runs["inspect-modeB"]["report"]
    reference = reference_tools(out, include_torch)
    tests = (
        {"error": "skipped by --skip-hardware-tests", "results": []}
        if args.skip_hardware_tests
        else hardware_tests(out, args.allow_cuda_allocation)
    )
    smoke = (
        cuda_smoke_test(include_torch)
        if args.cuda_smoke_test
        else {"outcome": "not run", "reason": "opt-in; use --cuda-smoke-test"}
    )
    summary = {
        "environment": env,
        "verdict": verdict(mode_a),
        "inspector_runs": {
            name: {k: v for k, v in r.items() if k != "report"} for name, r in runs.items()
        },
        "features": feature_table(mode_a) if mode_a else [],
        "not_implemented": [
            {"feature": n, "reference": ref} for n, ref in NOT_IMPLEMENTED_FEATURES
        ],
        "readiness": mode_b["readiness"] if mode_b else None,
        "findings": [
            {k: f[k] for k in ("check_id", "device_ref", "status", "severity", "confidence")}
            for f in (mode_a["diagnostics"] if mode_a else [])
        ],
        "hardware_tests": tests,
        "reference_tools": reference,
        "cuda_smoke_test": smoke,
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out / "cuda-smoke-test.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(summary_markdown(summary))

    failed = (
        any(
            r["exit_code"] not in ({0} if n.startswith("inspect-modeA") else {0, 1, 3})
            for n, r in runs.items()
        )
        or any(
            r["traceback_on_stderr"] or not r["schema"].startswith(("valid", "not checked"))
            for r in runs.values()
        )
        or any(t["outcome"] in ("failed", "error") for t in tests.get("results", []))
        or smoke["outcome"] == "failed"
    )
    print(f"{summary['verdict']}\nEvidence bundle: {out.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
