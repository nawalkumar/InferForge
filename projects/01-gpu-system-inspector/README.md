# GPU System Inspector — InferForge Project 01

> **Status: Stage 5.1. GPU inspection is not implemented yet.** The specification is
> approved. The report models, JSON schema, adapter contracts and the first production
> collector exist; the collector reads system and platform facts (OS, CPU, physical memory,
> Python, Linux environment, allowlisted environment variables) on Linux, Windows and macOS.
> Only Linux (WSL2) has been run on a real host so far. The `inferforge-gpu`
> command still offers only `--help` and `version`: `inspect` is a placeholder that collects
> nothing, prints that it isn't implemented, and exits with status 6. Progress is tracked
> in [docs/app_tracker.md](docs/app_tracker.md).

## Current development usage

From the repository root (requires uv and Python 3.11+):

```bash
uv sync                                          # installs the workspace, including this package (editable)
uv run inferforge-gpu --help
uv run inferforge-gpu version                    # application, schema, Python and nvidia-ml-py versions
uv run inferforge-gpu version --json -           # the same as JSON
cd projects/01-gpu-system-inspector && uv run pytest
```

## What it will be

**GPU System Inspector** (`inferforge-gpu`) is a lightweight, read-only command-line tool
and Python library. It inspects a machine's GPU environment **across layers** and explains
what it finds:

- **System:** OS, CPU architecture, memory, Python, Linux environment (native Linux, WSL1,
  WSL2, or unknown when the kernel evidence is inconclusive).
- **GPU hardware:** NVIDIA devices, UUIDs, memory, compute capability, PCI, MIG mode.
- **Driver & CUDA:** driver version, NVML, *driver-supported* CUDA version, local CUDA
  Toolkit versions, and the CUDA version PyTorch was built with, each labelled for what it is.
- **Framework:** whether PyTorch is installed in the same Python environment, importable
  and able to use CUDA, with device correlation to NVML.
- **Runtime snapshot:** utilization, memory, temperature, power and processes, where the
  platform allows.

Every value carries its **source, method, unit and collection status**. A missing value is
reported as missing, with a reason, never as `0`.

On top of the inventory it provides:

- **Evidence-based diagnostics** (D01–D08): PyTorch can't use CUDA, free memory vs
  requirement, CUDA version layers, no NVIDIA GPU, dependency availability, unavailable
  metrics, temperature observations, device visibility. Findings keep observations,
  inferences and recommendations separate, and state their confidence and limitations.
- **Basic free-VRAM readiness**: `--required-vram 8` checks each GPU for at least 8 GiB of
  *free* memory right now. This is not a guarantee that a model will fit.
- **Reports**: a human-readable terminal report and a versioned JSON report, with optional
  redaction for sharing.

It never installs software, never changes GPU state, and works without an NVIDIA GPU or
driver (it then reports general system information and explains what is missing). It
complements, and does not replace, `nvidia-smi`, NVML, DCGM, `gpustat`, PyTorch's CUDA APIs
and Nsight ([docs/existing_tools_analysis.md](docs/existing_tools_analysis.md)).

## Planned usage (V1; `inspect` not available yet)

```bash
inferforge-gpu inspect                          # general inspection (text report)
inferforge-gpu inspect --required-vram 8        # readiness: exit 0 ready / 1 not ready / 3 unknown
inferforge-gpu inspect --json report.json       # text report + versioned JSON report file
inferforge-gpu inspect --json -                 # JSON report on stdout instead of text
inferforge-gpu version
```

There is no demo or simulation mode. Every report describes the machine it ran on.

## Dependencies

| Kind | Component |
| ---- | --------- |
| Required (installed with the tool) | `nvidia-ml-py` (NVIDIA's NVML bindings; works without a GPU) |
| Optional (detected, never installed) | PyTorch, inspected in the same Python environment as the tool |
| External (detected, never installed or modified) | NVIDIA driver/NVML, `nvidia-smi`, CUDA Toolkit |

Python: `>=3.11`, the InferForge workspace floor
([../../docs/python-version-review.md](../../docs/python-version-review.md), WSD-01).
Direct testing on 3.11 is still pending; development and tests currently run on 3.12.

## Planned platform support

Linux and Windows (full NVIDIA inspection where the driver allows), WSL2 (with its
documented NVML limitations), and macOS (general system information only, with NVIDIA
features reported as unsupported). Details:
[docs/product_specification.md](docs/product_specification.md) §4.

## Specification documents

| Document | Contents |
| -------- | -------- |
| [problem_statement.md](docs/problem_statement.md) | Problem, users, product gap |
| [product_specification.md](docs/product_specification.md) | Product definition, dependencies, modes, V1 scope, non-goals |
| [requirements.md](docs/requirements.md) | Functional and non-functional requirements |
| [architecture.md](docs/architecture.md) | Components, interfaces, decisions, project configuration |
| [data_flow.md](docs/data_flow.md) | Execution order, error flow, output routing |
| [data_model_specification.md](docs/data_model_specification.md) | Models, statuses, units, JSON contract, privacy |
| [cli_specification.md](docs/cli_specification.md) | `inferforge-gpu` commands, options, validation, exit codes |
| [diagnostics_specification.md](docs/diagnostics_specification.md) | Diagnostic rules D01–D08 |
| [testing_strategy.md](docs/testing_strategy.md) | Test levels, 20 fixtures, hardware tests |
| [acceptance_criteria.md](docs/acceptance_criteria.md) | 60 acceptance criteria with traceability |
| [existing_tools_analysis.md](docs/existing_tools_analysis.md) | nvidia-smi, NVML, DCGM, gpustat, torch.cuda, Nsight |
| [technical_evidence.md](docs/technical_evidence.md) | Sources and claim classification |
| [app_tracker.md](docs/app_tracker.md) | **Single source of truth for status** |

## Workspace note

This directory is a member of the InferForge uv workspace. Distribution
`01-gpu-system-inspector`, import package `gpu_system_inspector` (src layout), console
script `inferforge-gpu` ([docs/architecture.md](docs/architecture.md) §9). It is marked
`Private :: Do Not Upload` and is not published.

## License

Apache-2.0, as for the InferForge repository.
