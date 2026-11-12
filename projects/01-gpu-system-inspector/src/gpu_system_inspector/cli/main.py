"""Entry point for the `inferforge-gpu` command (docs/cli_specification.md, IMP-C01–C03).

`inspect` validates its options (exit 2), configures logging on stderr, runs one live
inspection through `api.inspect`, redacts the report when `--redact` is set (before anything
leaves the process), writes the JSON report atomically if asked, prints the text report (or
the JSON with `--json -`) and exits with the code of §6: `5` → `4` → readiness `1`/`3` → `0`.
`version` reads distribution metadata only.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import platform
import re
import sys
import traceback
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from gpu_system_inspector import (
    APPLICATION_NAME,
    DISPLAY_NAME,
    DISTRIBUTION_NAME,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    __version__,
)
from gpu_system_inspector.cli.exit_codes import ExitCode
from gpu_system_inspector.models import (
    CollectionStatus,
    CollectorRunStatus,
    InspectionOptions,
    InspectionReport,
    ReadinessStatus,
    required_bytes_for,
)
from gpu_system_inspector.reporting import (
    OutputError,
    RedactionPolicy,
    Redactor,
    redact,
    render_json,
    render_text,
    write_atomic,
)

PROG = "inferforge-gpu"
NVML_BINDING_DISTRIBUTION = "nvidia-ml-py"

MAX_REQUIRED_VRAM_GIB = 1_048_576
MAX_TIMEOUT_S = 300
MAX_PYTORCH_TIMEOUT_S = 600

REQUIRED_VRAM_ERROR = "--required-vram must be a positive number of GiB (for example 8 or 7.5)"
_NUMBER_WITH_UNIT = re.compile(r"^\s*[0-9]*\.?[0-9]+\s*[A-Za-z]+\s*$")

INSPECT_EPILOG = """\
PyTorch is inspected in the Python environment that runs this command.

exit status:
  0  report produced (and, with --required-vram, at least one GPU is ready)
  1  --required-vram given and no GPU has enough free memory
  2  invalid usage
  3  --required-vram given and readiness could not be determined
  4  the JSON report could not be written to PATH
  5  internal error
"""
READINESS_EXIT = {
    ReadinessStatus.READY: ExitCode.SUCCESS,
    ReadinessStatus.NOT_READY: ExitCode.READINESS_NOT_MET,
    ReadinessStatus.UNKNOWN: ExitCode.READINESS_UNKNOWN,
}
LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)
log = logging.getLogger("gpu_system_inspector")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            f"{DISPLAY_NAME}: inspect the local GPU environment across system, hardware, "
            "driver, CUDA, PyTorch and runtime layers, and explain any problems found. "
            "Read-only; installs nothing; makes no network connections."
        ),
    )
    commands = parser.add_subparsers(dest="command", metavar="{inspect,version}")

    inspect = commands.add_parser(
        "inspect",
        help="inspect the GPU environment",
        description=(
            f"{DISPLAY_NAME}: inspect the local GPU environment across system, hardware, "
            "driver, CUDA, PyTorch and runtime layers, and explain any problems found. "
            "Read-only; installs nothing; makes no network connections."
        ),
        epilog=INSPECT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inspect.add_argument(
        "--json",
        metavar="PATH",
        help="also write the JSON report to PATH ('-' = JSON to standard output instead of "
        "the text report)",
    )
    inspect.add_argument(
        "--required-vram",
        metavar="GIB",
        help="check that each GPU has at least GIB gibibytes (2^30 bytes) of FREE memory "
        "right now. Passing this check does not guarantee a model will fit or run.",
    )
    inspect.add_argument("--no-pytorch", action="store_true", help="do not inspect PyTorch")
    inspect.add_argument(
        "--no-processes", action="store_true", help="do not list processes using the GPU"
    )
    inspect.add_argument(
        "--redact",
        action="store_true",
        help="replace paths, process names, device IDs and user names",
    )
    inspect.add_argument(
        "--timeout",
        metavar="SECONDS",
        default="10",
        help="timeout for external commands (default: 10)",
    )
    inspect.add_argument(
        "--pytorch-timeout",
        metavar="SECONDS",
        default="60",
        help="timeout for the PyTorch probe (default: 60)",
    )
    inspect.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="more logging on stderr (-vv for debug)",
    )
    inspect.set_defaults(handler=_run_inspect, subparser=inspect)

    version = commands.add_parser(
        "version",
        help="show application, report schema, Python and nvidia-ml-py versions",
        description="Show application, report schema, Python and nvidia-ml-py versions.",
    )
    version.add_argument(
        "--json",
        metavar="PATH",
        help="also write the version information as JSON to PATH ('-' = JSON to standard "
        "output instead of text)",
    )
    version.set_defaults(handler=_run_version, subparser=version, verbose=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return ExitCode.INTERRUPTED
    except Exception as exc:
        print(
            f"error: internal error: {type(exc).__name__}: {exc} (run with -vv for details)",
            file=sys.stderr,
        )
        if args.verbose >= 2:
            traceback.print_exc()
        return ExitCode.INTERNAL_ERROR


def _run_inspect(args: argparse.Namespace) -> int:
    parser: argparse.ArgumentParser = args.subparser
    _validate_required_vram(parser, args.required_vram)
    _validate_timeout(parser, "--timeout", args.timeout, MAX_TIMEOUT_S)
    _validate_timeout(parser, "--pytorch-timeout", args.pytorch_timeout, MAX_PYTORCH_TIMEOUT_S)
    if args.json is not None and not args.json:
        parser.error("--json requires a file path or '-'")
    _configure_logging(args.verbose)
    options = InspectionOptions(
        required_vram_gib=float(args.required_vram) if args.required_vram is not None else None,
        include_pytorch=not args.no_pytorch,
        include_processes=not args.no_processes,
        redact=args.redact,
        subprocess_timeout_s=float(args.timeout),
        pytorch_timeout_s=float(args.pytorch_timeout),
    )
    policy = local_redaction_policy() if args.redact else None
    scrub = Redactor(policy).text if policy is not None else _identity
    try:
        report = run_inspection(options)
        if policy is not None:
            scrub = Redactor.for_report(report, policy).text
            report = redact(report, policy)
        json_text = render_json(report)
        text = render_text(report) if args.json != "-" else None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        print(
            scrub(f"error: internal error: {message} (run with -vv for details)"),
            file=sys.stderr,
        )
        if args.verbose >= 2:
            sys.stderr.write(scrub(traceback.format_exc()))
        return ExitCode.INTERNAL_ERROR
    _warn(report)
    exit_code = (
        READINESS_EXIT[report.readiness.overall_status] if report.readiness else (ExitCode.SUCCESS)
    )
    if args.json is not None and args.json != "-":
        target = Path(args.json)
        try:
            write_atomic(json_text, target)
        except OutputError as exc:
            print(
                scrub(f"error: cannot write JSON report to {target}: {exc.reason}"),
                file=sys.stderr,
            )
            exit_code = ExitCode.OUTPUT_ERROR
        else:
            print(scrub(f"JSON report written to {target}"), file=sys.stderr)
    _write_stdout(json_text if text is None else text)
    return exit_code


def run_inspection(options: InspectionOptions) -> InspectionReport:
    """One live inspection (imported lazily: `version` and `--help` never load the core)."""
    from gpu_system_inspector.api import inspect

    return inspect(options)


def local_redaction_policy() -> RedactionPolicy:
    """The local user name and home directory, used only as redaction inputs (data model §8)."""
    try:
        user: str | None = getpass.getuser()
    except Exception:
        user = None
    try:
        home: str | None = str(Path.home())
    except Exception:
        home = None
    return RedactionPolicy(user=user, home=home)


def _identity(text: str) -> str:
    return text


def _configure_logging(verbose: int) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(LOG_LEVELS[min(verbose, 2)])
    log.propagate = False


def _warn(report: InspectionReport) -> None:
    """CLI §7: one WARNING line per collector that raised, and for a missing binding."""
    for run in report.collector_runs:
        if run.status is CollectorRunStatus.FAILED:
            log.warning(
                "collector %r raised an error; see COLLECTION ERRORS in the report", run.collector
            )
    for error in report.collection_errors:
        if error.collector.startswith("diagnostics."):
            log.warning("%s could not be evaluated; see COLLECTION ERRORS", error.collector)
    if report.driver.nvml_binding.status is CollectionStatus.NOT_INSTALLED:
        log.warning(
            "the required dependency %s is not installed; NVML-based fields are unavailable",
            NVML_BINDING_DISTRIBUTION,
        )


def _write_stdout(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
    sys.stdout.flush()


def _validate_required_vram(parser: argparse.ArgumentParser, raw: str | None) -> None:
    if raw is None:
        return
    try:
        value = float(raw)
    except ValueError:
        suffix = "; do not include a unit" if _NUMBER_WITH_UNIT.match(raw) else ""
        parser.error(REQUIRED_VRAM_ERROR + suffix)
    if (
        not math.isfinite(value)
        or not 0 < value <= MAX_REQUIRED_VRAM_GIB
        or required_bytes_for(value) < 1
    ):
        parser.error(REQUIRED_VRAM_ERROR)


def _validate_timeout(
    parser: argparse.ArgumentParser, option: str, raw: str, maximum: float
) -> None:
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or not 0 < value <= maximum:
        parser.error(f"{option} must be between 0 and {maximum} seconds")


def _run_version(args: argparse.Namespace) -> int:
    info = version_info()
    if args.json is not None and not args.json:
        args.subparser.error("--json requires a file path or '-'")
    if args.json == "-":
        sys.stdout.write(_to_json(info))
        return ExitCode.SUCCESS

    exit_code = ExitCode.SUCCESS
    if args.json is not None:
        target = Path(args.json)
        try:
            write_atomic(_to_json(info), target)
        except OutputError as exc:
            print(f"error: cannot write JSON to {target}: {exc.reason}", file=sys.stderr)
            exit_code = ExitCode.OUTPUT_ERROR
        else:
            print(f"Version information written to {target}", file=sys.stderr)
    sys.stdout.write(_version_text(info))
    return exit_code


def version_info() -> dict[str, Any]:
    """Version facts. Reads distribution metadata only; never imports `pynvml` or `torch`."""
    try:
        binding: dict[str, str] | None = {
            "distribution": NVML_BINDING_DISTRIBUTION,
            "version": metadata.version(NVML_BINDING_DISTRIBUTION),
        }
    except metadata.PackageNotFoundError:
        binding = None
    return {
        "application": {
            "name": APPLICATION_NAME,
            "display_name": DISPLAY_NAME,
            "distribution": DISTRIBUTION_NAME,
            "version": __version__,
        },
        "schema_name": REPORT_SCHEMA_NAME,
        "schema_version": REPORT_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "nvml_binding": binding,
    }


def _version_text(info: dict[str, Any]) -> str:
    app = info["application"]
    binding = info["nvml_binding"]
    binding_line = (
        f"{binding['distribution']} {binding['version']}"
        if binding
        else f"{NVML_BINDING_DISTRIBUTION} NOT INSTALLED (required dependency missing)"
    )
    return (
        f"{app['display_name']} {app['version']} (distribution {app['distribution']})\n"
        f"report schema {info['schema_name']} {info['schema_version']}\n"
        f"python {info['python']['version']} ({info['python']['implementation']})\n"
        f"{binding_line}\n"
    )


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
