"""Entry point for the `inferforge-gpu` command (docs/cli_specification.md).

Only `--help`, `version` and the argument interface of `inspect` exist so far. `inspect`
validates its arguments and then reports that inspection is not implemented; it collects
nothing and writes no report.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import re
import sys
import tempfile
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

PROG = "inferforge-gpu"
NVML_BINDING_DISTRIBUTION = "nvidia-ml-py"

MAX_REQUIRED_VRAM_GIB = 1_048_576
MAX_TIMEOUT_S = 300
MAX_PYTORCH_TIMEOUT_S = 600

REQUIRED_VRAM_ERROR = "--required-vram must be a positive number of GiB (for example 8 or 7.5)"
_NUMBER_WITH_UNIT = re.compile(r"^\s*[0-9]*\.?[0-9]+\s*[A-Za-z]+\s*$")

INSPECT_EPILOG = """\
NOT IMPLEMENTED YET: this development build accepts and validates these options, then
exits with status 6 without inspecting anything.

PyTorch is inspected in the Python environment that runs this command.

exit status:
  0  report produced (and, with --required-vram, at least one GPU is ready)
  1  --required-vram given and no GPU has enough free memory
  2  invalid usage
  3  --required-vram given and readiness could not be determined
  4  the JSON report could not be written to PATH
  5  internal error
  6  inspect is not implemented in this development build
"""

INSPECT_NOT_IMPLEMENTED = (
    f"error: 'inspect' is not implemented in this development build of {DISPLAY_NAME} "
    f"{__version__}.\n"
    "No inspection was performed: no system, GPU, driver, CUDA or PyTorch information was "
    "collected, and no report was written."
)


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
        help="inspect the GPU environment (not implemented yet)",
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
    print(INSPECT_NOT_IMPLEMENTED, file=sys.stderr)
    return ExitCode.NOT_IMPLEMENTED


def _validate_required_vram(parser: argparse.ArgumentParser, raw: str | None) -> None:
    if raw is None:
        return
    try:
        value = float(raw)
    except ValueError:
        suffix = "; do not include a unit" if _NUMBER_WITH_UNIT.match(raw) else ""
        parser.error(REQUIRED_VRAM_ERROR + suffix)
    if not math.isfinite(value) or not 0 < value <= MAX_REQUIRED_VRAM_GIB:
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
        except OSError as exc:
            print(f"error: cannot write JSON to {target}: {exc.strerror or exc}", file=sys.stderr)
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


def write_atomic(text: str, path: Path) -> None:
    """Write `text` to `path` via a temporary file in the same directory and a rename."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
