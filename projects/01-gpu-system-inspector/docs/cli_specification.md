# CLI Specification — Project 01: GPU System Inspector

Status: **Specification approved 2026-11-04.** Command name and JSON option approved
(DEC-03, DEC-25). Implementation status: `--help`, `version` and `inspect` are implemented
(IMP-C01–C03, DEC-50); the temporary exit code 6 was removed as §6 required. See
[app_tracker.md](app_tracker.md).

## 1. Command tree

```
inferforge-gpu [-h]
├── inspect [options]        # Mode A; Mode B with --required-vram
└── version [--json PATH]
```

The console script is `inferforge-gpu` (DEC-03). There is no `inferforge` umbrella
command in V1. The module form is equivalent: `python -m gpu_system_inspector inspect …`.

| Invocation | Behaviour | Exit |
| ---------- | --------- | ---- |
| `inferforge-gpu --help`, `inferforge-gpu inspect --help`, `inferforge-gpu version --help` | Help for that level on stdout | 0 |
| `inferforge-gpu` (no command) | Usage + `error: a command is required` on stderr | 2 |
| Unknown command or option | Usage + error on stderr | 2 |

## 2. `inferforge-gpu inspect`

### 2.1 Options

| Option | Argument | Default | Description |
| ------ | -------- | ------- | ----------- |
| `--json` | `PATH` or `-` | none | Also write the versioned JSON report to `PATH` (atomic; overwrites an existing file). `-` writes the JSON report to stdout **instead of** the text report. To write to a file literally named `-`, use `./-`. |
| `--required-vram` | `GIB` (number) | none | Enable Mode B. Required **free** device memory in GiB (1 GiB = 2³⁰ bytes). |
| `--no-pytorch` | – | off | Don't probe PyTorch (`frameworks.pytorch.status = SKIPPED`). |
| `--no-processes` | – | off | Don't enumerate GPU processes (`processes.status = SKIPPED`). |
| `--redact` | – | off | Redact paths, process names, device identifiers and user identity in all outputs (data model §8). |
| `--timeout` | `SECONDS` | `10` | Timeout for each external command (`nvidia-smi`, `nvcc`, `sysctl`). |
| `--pytorch-timeout` | `SECONDS` | `60` | Timeout for the PyTorch probe subprocess. |
| `-v`, `--verbose` | repeatable | WARNING | `-v` = INFO, `-vv` = DEBUG (and tracebacks for internal errors). Logs go to stderr. |
| `-h`, `--help` | – | – | Show help. |

The text report always goes to stdout, unless `--json -` is used.

### 2.2 Input validation (all failures → exit 2, message on stderr, nothing on stdout, no inspection)

| Option | Rule | Error message (exact prefix) |
| ------ | ---- | ---------------------------- |
| `--required-vram` | Parses with `float()`; finite; `> 0`; `≤ 1048576` | `error: --required-vram must be a positive number of GiB (for example 8 or 7.5)` |
| `--required-vram` with a unit suffix (`8GB`, `8GiB`, `8G`) | Rejected (the value is always GiB) | same message plus `; do not include a unit` |
| `--timeout` | finite; `0 < x ≤ 300` | `error: --timeout must be between 0 and 300 seconds` |
| `--pytorch-timeout` | finite; `0 < x ≤ 600` | `error: --pytorch-timeout must be between 0 and 600 seconds` |
| `--json` | A non-empty argument is required. The path itself isn't checked before inspection; write errors are reported afterwards with exit 4. | `error: --json requires a file path or '-'` |

`--required-vram 0` is rejected: a zero requirement makes readiness meaningless.

Messages use the standard argparse format: the sub-command usage line, then
`inferforge-gpu inspect: error: <message>`, where `<message>` is the text in the table
without its leading `error: `.

### 2.3 Behaviour

1. Validate options, then configure logging (stderr).
2. Run the inspection ([data_flow.md](data_flow.md)).
3. If `--json PATH` (not `-`): write the JSON report atomically to `PATH`. On success,
   print `JSON report written to <PATH>` on stderr. On failure, print
   `error: cannot write JSON report to <PATH>: <reason>` on stderr and remember exit 4.
4. Print the text report on stdout, or the JSON report if `--json -`.
5. Exit with the code in §6.

Mode A never fails because of findings: a machine with no GPU produces exit `0` and a
report that says so.

## 3. `inferforge-gpu version`

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--json PATH` or `--json -` | none | Also write the version information as JSON to `PATH` (atomic; the text form is still printed, and stderr gets `Version information written to <PATH>`), or write JSON to stdout with `-` instead of the text form. A write failure prints `error: cannot write JSON to <PATH>: <reason>` and exits 4. |

Text output (values illustrative):

```
GPU System Inspector 0.1.0 (distribution 01-gpu-system-inspector)
report schema inferforge.gpu_system_inspector.report 1.0
python 3.11.14 (CPython)
nvidia-ml-py 13.615.71
```

`nvidia-ml-py` is a required dependency (DEC-05). If it is missing because of a broken
installation, the last line reads `nvidia-ml-py NOT INSTALLED (required dependency missing)`.
The command never initialises NVML or imports PyTorch. It exits `0`, or `4` if `--json PATH`
can't be written.

JSON output:

```json
{"application": {"name": "gpu-system-inspector", "display_name": "GPU System Inspector",
                 "distribution": "01-gpu-system-inspector", "version": "0.1.0"},
 "schema_name": "inferforge.gpu_system_inspector.report", "schema_version": "1.0",
 "python": {"version": "3.11.14", "implementation": "CPython"},
 "nvml_binding": {"distribution": "nvidia-ml-py", "version": "13.615.71"}}
```

(`nvml_binding` is `null` when the binding can't be found.)

## 4. Help text (normative content; wording may be polished)

```
usage: inferforge-gpu inspect [-h] [--json PATH] [--required-vram GIB]
                              [--no-pytorch] [--no-processes] [--redact]
                              [--timeout SECONDS] [--pytorch-timeout SECONDS] [-v]

GPU System Inspector: inspect the local GPU environment across system, hardware,
driver, CUDA, PyTorch and runtime layers, and explain any problems found.
Read-only; installs nothing; makes no network connections.

options:
  --json PATH            also write the JSON report to PATH ('-' = JSON to standard
                         output instead of the text report)
  --required-vram GIB    check that each GPU has at least GIB gibibytes (2^30 bytes) of
                         FREE memory right now. Passing this check does not guarantee a
                         model will fit or run.
  --no-pytorch           do not inspect PyTorch
  --no-processes         do not list processes using the GPU
  --redact               replace paths, process names, device IDs and user names
  --timeout SECONDS      timeout for external commands (default: 10)
  --pytorch-timeout SECONDS
                         timeout for the PyTorch probe (default: 60)
  -v, --verbose          more logging on stderr (-vv for debug)

PyTorch is inspected in the Python environment that runs this command.

exit status:
  0  report produced (and, with --required-vram, at least one GPU is ready)
  1  --required-vram given and no GPU has enough free memory
  2  invalid usage
  3  --required-vram given and readiness could not be determined
  4  the JSON report could not be written to PATH
  5  internal error
```

## 5. Output formats

### 5.1 JSON (`--json PATH` or `--json -`)

Exactly one JSON document following the schema in
[data_model_specification.md](data_model_specification.md) §7, UTF-8, 2-space indent,
trailing newline. With `--json -`, nothing else is written to stdout.

### 5.2 Text (default, stdout)

Fixed section order. Sections that can't be collected are still printed, with their status.

```
GPU System Inspector 0.1.0 · InferForge · report schema 1.0
Generated 2026-11-04T09:00:00Z · mode GENERAL · data origin LIVE

SYSTEM
DEVICES            (one block per device, by NVML index)
DRIVER & CUDA      (driver, NVML, driver-supported CUDA, libcuda, nvidia-smi)
CUDA TOOLKIT       (candidates, active nvcc, environment variables)
PYTORCH            (in this Python environment)
RUNTIME SNAPSHOT   (per device)
READINESS          (only in Mode B; always ends with the disclaimer)
FINDINGS           (FAIL, WARNING, UNKNOWN, INFO, PASS, NOT_APPLICABLE; then check ID)
UNSUPPORTED        (capabilities with reasons)
COLLECTION ERRORS
LIMITATIONS
```

Display rules:

1. Collected values show a human unit and their source: `24.0 GiB free (NVML)`.
   Exact bytes appear only in JSON.
2. Missing values show the status and reason, never a number:
   `GPU utilization   —  NOT_SUPPORTED (NVML: not supported on WSL2)`.
3. The three CUDA versions are always labelled distinctly:
   `Driver supports CUDA up to 12.4`, `CUDA Toolkit 12.1.105 (nvcc)`,
   `PyTorch built with CUDA 12.1`.
4. `Memory bus activity` is the label for `memory_bus_utilization_percent`.
5. If `data_origin` isn't `LIVE` (only possible for test fixtures, DEC-17), the first and
   last lines are `*** FIXTURE DATA — NOT A REAL MEASUREMENT ***`.
6. No colour or ANSI sequences in V1 (DEC-24). Lines ≤ 100 characters where practical.
7. Findings print `[STATUS] Dxx title`, then explanation, evidence (`observed:` /
   `inferred:`), recommendations (`suggestion:`), confidence and limitations.

Illustrative fragment (**not a real measurement**):

```
READINESS  required free memory: 8.0 GiB (8589934592 bytes)
  nvml:0  NVIDIA GeForce RTX 4090   READY      free 21.3 GiB (NVML), margin +13.3 GiB
  nvml:1  NVIDIA GeForce RTX 3060   NOT_READY  free 5.1 GiB (NVML), short by 2.9 GiB
  overall: READY (at least one device ready)
  Note: Free memory at least equal to the requested amount was observed at the time of
  inspection. This does not guarantee that a model will fit or run …
```

## 6. Exit codes

| Code | Name | When |
| ---- | ---- | ---- |
| 0 | `SUCCESS` | Report produced. Mode A always; Mode B with `overall_status = READY`. `version` and `--help`. |
| 1 | `READINESS_NOT_MET` | Mode B, report produced, `overall_status = NOT_READY`. |
| 2 | `USAGE_ERROR` | Invalid arguments (argparse or §2.2). No inspection performed. |
| 3 | `READINESS_UNKNOWN` | Mode B, report produced, `overall_status = UNKNOWN`. |
| 4 | `OUTPUT_ERROR` | `--json PATH` couldn't be written (no partial file left). The text report is still printed. |
| 5 | `INTERNAL_ERROR` | Unexpected exception outside the collector/rule guards. |
| ~~6~~ | ~~`NOT_IMPLEMENTED`~~ | **Removed** (DEC-50): it existed only while `inspect` was a Stage 1 placeholder and is never returned. |
| 130 | `INTERRUPTED` | `KeyboardInterrupt` (Ctrl-C). |

Precedence when several apply: `2` (before inspection) → `5` → `4` → readiness code
(`1`/`3`) → `0`.

## 7. Error behaviour

| Situation | stdout | stderr | Exit |
| --------- | ------ | ------ | ---- |
| No NVIDIA GPU / no driver / PyTorch missing | report | warnings only if `-v` | 0 (Mode A) |
| Required `nvidia-ml-py` missing (broken installation) | report (NVML fields `NOT_INSTALLED`, D05 WARNING) | WARNING line | 0 (Mode A) |
| A collector raised | report (with `collection_errors`) | WARNING line naming the collector | 0 (Mode A) |
| `--json PATH` not writable / directory / missing parent | text report | `error: cannot write JSON report to …` | 4 |
| Internal error | nothing | `error: internal error: <ExceptionType>: <message> (run with -vv for details)` | 5 |
| Ctrl-C | nothing | `interrupted` | 130 |

Error messages are redacted when `--redact` is set.

## 8. Usage examples

```bash
inferforge-gpu inspect                                   # general inspection, text
inferforge-gpu inspect --required-vram 8                 # readiness; exit 0/1/3
inferforge-gpu inspect --json report.json                # text on screen + JSON file
inferforge-gpu inspect --json -                          # JSON on stdout only
inferforge-gpu inspect --redact --json shareable.json    # report safe to share
inferforge-gpu inspect --no-pytorch --no-processes       # skip PyTorch probe and process list
inferforge-gpu version
inferforge-gpu --help

# CI gate: keep the JSON evidence and fail the job unless a GPU has 16 GiB free
inferforge-gpu inspect --required-vram 16 --json gpu.json
```
