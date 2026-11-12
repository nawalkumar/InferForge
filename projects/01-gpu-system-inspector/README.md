# GPU System Inspector — InferForge Project 01

> **Status: functional MVP (development build 0.1.0), not published.** `inferforge-gpu inspect`
> runs the full pipeline: collection, diagnostics D01–D08, Mode B readiness, text and JSON
> reports, redaction and atomic JSON writing. It's tested against 24 synthetic fixture
> scenarios, installed from its wheel and tested on CPython 3.11 and 3.12, and run live on
> one WSL2 machine **without an NVIDIA GPU**.
>
> **It has never run on NVIDIA hardware.** Hardware validation (HW-01 to HW-12) is blocked
> until someone runs [the prepared procedure](docs/hardware_validation.md) on a GPU machine.
> Native Linux, Windows and macOS hosts are also untested. Status:
> [docs/app_tracker.md](docs/app_tracker.md).

**GPU System Inspector** (`inferforge-gpu`) is a read-only command-line tool and Python
library that answers "why can't this machine use its NVIDIA GPU the way I expect?". It
inspects the GPU environment **layer by layer**: system, GPU hardware, driver, CUDA Toolkit,
PyTorch, live runtime state. It then explains what it found, with the evidence behind each
conclusion, and says plainly what it couldn't see.

It's for ML engineers setting up or debugging a training or inference machine, people who
support them ("send me your report"), and CI jobs that need a GPU memory gate with evidence.

## Quickstart (about five minutes)

You need Python 3.11 or newer. No GPU is needed to try it.

**Option A: uv** (recommended for development), from the repository root:

```bash
git clone https://github.com/nawalkumar/InferForge.git && cd InferForge
uv sync                                    # installs the workspace, including this project
uv run inferforge-gpu version
uv run inferforge-gpu inspect              # text report on stdout, exit 0
```

**Option B: pip into an existing environment**, for example the one that has your
PyTorch. PyTorch is inspected only in the environment that runs the inspector:

```bash
python -m pip install ./projects/01-gpu-system-inspector   # from the repository root
inferforge-gpu inspect                                     # or: python -m gpu_system_inspector inspect
```

The package isn't on PyPI (see [Distribution and license](#distribution-and-license)).
Installation problems: [docs/troubleshooting.md](docs/troubleshooting.md).

## Supported Python versions and platforms

| | Status |
| - | ------ |
| Python | `>=3.11`. Full suite passes on CPython 3.12.12 (workspace and installed wheel) and 3.11.15 (installed wheel) |
| Linux (native) with NVIDIA | Implemented; **not run on hardware** (HW-01, HW-02 blocked) |
| WSL2 without NVIDIA | **Run live** (the development machine) |
| WSL2 with NVIDIA | Implemented and fixture-tested (FX11); not run |
| Windows (WDDM) | Implemented and fixture-tested (FX12); never run on Windows (KI-27) |
| macOS | General information only, by design; fixture-tested (FX13); never run on a Mac (KI-30) |

The tool never promises that a machine exposes NVIDIA information: when it can't, the report
says which layer was missing and why.

## What it reports

| Layer | What is reported |
| ----- | ---------------- |
| System | OS, distribution, Linux environment (native, WSL1, WSL2), CPU, memory, Python |
| GPU hardware | NVIDIA devices from NVML (or the `nvidia-smi` fallback): name, UUID, PCI bus ID, memory, compute capability, MIG mode; Linux PCI presence |
| Driver & CUDA | Driver and NVML versions, the **driver-supported** CUDA version, libcuda, `nvidia-smi` |
| CUDA Toolkit | Toolkit candidates (`CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH`, `/usr/local/cuda-*`), the active `nvcc`, allowlisted environment variables |
| PyTorch | In **this** Python environment: installed or not, version, CUDA build, PyTorch's own `cuda.is_available()`, devices correlated to NVML by UUID, then PCI bus ID |
| Runtime snapshot | Per device: utilization, memory bus activity, memory, temperature, power, compute processes |

Every value carries its **source, method, unit and collection status**. A missing value is
reported as missing, with a reason, never as `0`.

On top of the inventory:

- **Diagnostics D01–D08**, evidence-based. Every finding separates `observed:` from
  `inferred:`, states its confidence and limitations, and links only to official NVIDIA,
  PyTorch or Python documentation.

  | Check | Topic | Check | Topic |
  | ----- | ----- | ----- | ----- |
  | D01 | PyTorch can't use CUDA, with ranked causes | D05 | Dependencies |
  | D02 | Free memory vs requirement | D06 | Runtime metric availability |
  | D03 | CUDA version layers | D07 | Temperature |
  | D04 | NVIDIA GPU availability | D08 | Device visibility and ordering |

- **Mode B readiness** (`--required-vram GIB`): checks each GPU for at least that much
  **free** memory at the moment of sampling. This **doesn't** mean a model will fit or run;
  the report always ends this section with that disclaimer.
- **Reports**: plain text on stdout; a versioned JSON report (`--json PATH`, written
  atomically, or `--json -`); `--redact` replaces paths, process names, GPU UUIDs and PCI
  IDs, the user name and the home directory before anything leaves the process.

It never installs software, never changes GPU state and makes no network connections. It
complements, and doesn't replace, `nvidia-smi`, NVML, DCGM, `gpustat`, PyTorch's CUDA APIs
and Nsight ([docs/existing_tools_analysis.md](docs/existing_tools_analysis.md)).

## What it can and cannot establish

| It can establish | It cannot establish |
| ---------------- | ------------------- |
| Which layers are present, missing or inaccessible, each with its source and reason | That CUDA code will run: no CUDA code is executed; `cuda_available` is PyTorch's own answer |
| The driver's **supported** CUDA version, the toolkit's version and PyTorch's **build** CUDA version, kept apart | That a toolkit or driver version makes PyTorch work: versions are installation facts |
| Likely causes when PyTorch reports CUDA unavailable, ranked, with confidence | A definitive root cause: causes are inferred from one snapshot |
| Free device memory at one moment, compared with a number you give | That a model fits: free memory changes immediately, and model memory isn't estimated |
| That no NVIDIA GPU is present, **only** from NVML counting zero, macOS, or a complete native-Linux PCI scan | GPU absence from a missing driver: "not accessible" is reported as such |
| PyTorch facts for the Python environment running the inspector | PyTorch in any other environment (KI-11) |
| Compute processes using the GPU | Graphics processes (not collected yet, KI-36) |

## CLI

```bash
inferforge-gpu inspect                                  # general inspection (Mode A), always exit 0
inferforge-gpu inspect --required-vram 8                # readiness (Mode B): exit 0 / 1 / 3
inferforge-gpu inspect --json report.json               # text on screen + JSON file (atomic)
inferforge-gpu inspect --json -                         # JSON on stdout only
inferforge-gpu inspect --redact --json shareable.json   # report safe to share
inferforge-gpu inspect --no-pytorch --no-processes      # skip the PyTorch probe and process list
inferforge-gpu inspect -vv                              # debug logging on stderr
inferforge-gpu version [--json -]
inferforge-gpu inspect --help                           # every option and the exit codes
```

| Exit | Meaning |
| ---- | ------- |
| 0 | Report produced (Mode B: at least one GPU has enough free memory) |
| 1 | Mode B: no GPU has enough free memory |
| 2 | Invalid usage (nothing inspected) |
| 3 | Mode B: readiness couldn't be determined |
| 4 | The JSON report couldn't be written (the text report is still printed; no partial file) |
| 5 | Internal error |
| 130 | Interrupted |

Full specification: [docs/cli_specification.md](docs/cli_specification.md). Python API:
`from gpu_system_inspector import api; report = api.inspect()` (provisional, FR-23).

## Example output

### Live: this WSL2 machine, no NVIDIA GPU

```bash
uv run inferforge-gpu inspect                       # exit 0
uv run inferforge-gpu inspect --required-vram 8     # exit 3: readiness UNKNOWN
uv run inferforge-gpu inspect --redact --json /tmp/live-redacted.json
```

Excerpt of a real run (2026-11-11, `data origin LIVE`):

```
DRIVER & CUDA
  Driver version          —  NOT_DETECTED (NVML_ERROR_LIBRARY_NOT_FOUND: NVML Shared Library Not
                          Found)
  nvidia-smi              —  NOT_DETECTED (FileNotFoundError: nvidia-smi not found on PATH or
                          /usr/lib/wsl/lib)
FINDINGS
  [INFO] D04 NVIDIA GPU availability
      No NVIDIA GPU is accessible: NVML and nvidia-smi are unavailable and the PCI scan couldn't be
      used to confirm hardware presence. The rest of the report is still valid.
      suggestion: Install or update the NVIDIA driver on the Windows host (never inside WSL2).
      confidence: MEDIUM · severity: NONE
      limitation: Hardware absence can't be confirmed from this snapshot; a GPU may exist without an
                  accessible driver interface.
```

The report doesn't claim that no GPU exists, doesn't report NVIDIA readiness, and doesn't
infer CUDA usability from any version number. With `--required-vram 8`, readiness is
`UNKNOWN` (`DEVICE_PRESENCE_UNKNOWN`), exit 3.

Complete reports: [`examples/reports/`](examples/reports/): the live run above (redacted text
and JSON) and three **synthetic** reports, each named `fixture-*` and marked as fixture data.

### Synthetic: what an NVIDIA machine's report looks like (not a real measurement)

The showcase runs the production pipeline over the project's **synthetic test fixtures**
(fake NVML, fake `nvidia-smi`, fake PyTorch); only the adapters are fakes. The report says
so: `data_origin` is `FIXTURE`, the text opens and closes with
`*** FIXTURE DATA — NOT A REAL MEASUREMENT ***`, device names read "Synthetic GPU … (fixture)",
and stderr names the fixture. It's a repository script, not a CLI option: the
`inferforge-gpu` command has no demo mode (DEC-17), and live reports are always `LIVE`.

```bash
cd projects/01-gpu-system-inspector
uv run python examples/synthetic_showcase.py --list
uv run python examples/synthetic_showcase.py FX06 --required-vram 16   # two GPUs, one READY; exit 0
uv run python examples/synthetic_showcase.py FX07 --required-vram 8    # not enough free memory; exit 1
uv run python examples/synthetic_showcase.py FX09 --required-vram 8    # metrics inaccessible; exit 3
uv run python examples/synthetic_showcase.py FX02b                     # driver older than the PyTorch build
```

FX02b (fixture data) shows how D01 ranks a cause and keeps versions apart:

```
  [FAIL] D01 PyTorch can't use CUDA
      PyTorch reports CUDA unavailable. Likely causes, strongest first: D01.5.
      observed: torch.cuda.is_available() is False (PYTORCH)
      observed: Driver-supported CUDA version is '11.8' (NVML)
      observed: PyTorch build CUDA version is '12.4' (PYTORCH)
      inferred: D01.5: the driver likely doesn't support the CUDA major version PyTorch was built
                with.
      suggestion: Update the NVIDIA driver, or install a PyTorch build for an older CUDA major
                  version.
      confidence: MEDIUM · severity: HIGH
```

A walkthrough of every demonstration journey and a recording plan:
[docs/demonstration.md](docs/demonstration.md).

## Hardware validation

The tool has never run on NVIDIA hardware. Hardware validation stays **BLOCKED** until someone
produces evidence on a GPU machine and the owner records it in
[docs/app_tracker.md](docs/app_tracker.md) §6.

**Contributor path (GitHub → Colab → evidence):**
[`notebooks/inferforge_nvidia_gpu_validation.ipynb`](notebooks/inferforge_nvidia_gpu_validation.ipynb)

1. Open the notebook in Colab (badge below, once published).
2. Select a **GPU** runtime.
3. Keep `SOURCE_MODE = "git"` and set `GIT_BRANCH` to the public branch that contains Project 01.
   The notebook downloads that branch's source archive itself — **no manual `.tar.gz` upload**.
4. It installs from that local tree only (not PyPI), runs live `inferforge-gpu inspect`,
   `pytest -m hardware`, and [`validation/hardware_evidence.py`](validation/hardware_evidence.py).
5. Download the evidence bundle and attach it to a GitHub issue.

Full instructions: [docs/hardware_validation.md](docs/hardware_validation.md) §0.

`SOURCE_MODE = "archive"` remains available for unpublished, private, or offline trees (build a
project tarball and upload it in Colab Files).

Colab results validate **only the recorded Colab environment** (GPU model, driver, software
versions, source repository/branch/commit when known). They are not a blanket claim about
every platform.

**Open in Colab.** Enable the badge only after this notebook exists on a **published** GitHub
branch. A remote check of `origin/main` found no Project 01 validation assets yet, so there
is no live badge. Until then: **File → Upload notebook** in Colab. After publish, use:

```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nawalkumar/InferForge/blob/<branch>/projects/01-gpu-system-inspector/notebooks/inferforge_nvidia_gpu_validation.ipynb)
```

Replace `<branch>` with the actual published branch name. Do not treat a placeholder as live.

## Architecture

```
cli ──► api / core.Inspector ──► collectors ──► adapters ──► host (NVML, nvidia-smi, files, PyTorch child)
                 │
                 ├─► readiness (Mode B)   ─┐  pure functions of the snapshot
                 ├─► diagnostics D01–D08 ─┘  (models only; no host access)
                 └─► reporting: assemble → redact → render_text / render_json → write_atomic
```

The collectors run in a fixed order, each under one guard, so a failing collector becomes a
`collection_errors` entry, never a crash: system, PCI, NVML session (driver, GPU hardware,
runtime), `nvidia-smi`, CUDA environment, PyTorch (a child process of this interpreter, so
the inspector itself never creates a CUDA context), correlation. Details:
[docs/architecture.md](docs/architecture.md) (DEC-44 to DEC-50),
[docs/data_flow.md](docs/data_flow.md), and the module guide in
[CONTRIBUTING.md](CONTRIBUTING.md#where-things-live).

## Limitations

- A report is a **single snapshot**. Free memory, temperature and utilization can change
  immediately afterwards.
- Readiness compares free memory with a number. It doesn't estimate model memory and isn't
  a guarantee.
- No CUDA code is run: `cuda_available` is PyTorch's own answer, and a toolkit or driver
  version is an installation fact, not proof that CUDA works.
- PyTorch is inspected only in the Python environment that runs the inspector (KI-11).
- Only **compute** processes are listed; graphics processes aren't collected yet (KI-36,
  FR-11 incomplete).
- Owner decisions still open (none is resolved by the code): KI-37 (Windows default toolkit
  location), KI-38 (PCI domains wider than 16 bits), OQ-11 (process-name fallback), the
  UUID/PCI one-to-one correlation policy, approval of DEC-04/06/07/08 and DEC-44 to DEC-50,
  review of AC43 and AC09 (schema `1.0` freeze). The full list: [docs/roadmap.md](docs/roadmap.md).

## Troubleshooting

| Symptom | Where to look |
| ------- | ------------- |
| `requires a different Python` | Python 3.11+ is required ([troubleshooting](docs/troubleshooting.md#installation)) |
| Everything NVIDIA says `NOT_DETECTED` | No accessible driver library; on WSL2 the driver belongs on the Windows host |
| PyTorch `NOT_INSTALLED` although you use it | Run the inspector from the environment that has PyTorch |
| Exit 4 | The `--json` path's directory is missing or not writable |

More: [docs/troubleshooting.md](docs/troubleshooting.md).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers setup, tests, the module guide, and how to add a
diagnostic rule, fixture, collector or platform adapter. Planned work and open decisions:
[docs/roadmap.md](docs/roadmap.md). Small, well-defined starting points:
[docs/good_first_issues.md](docs/good_first_issues.md).

```bash
cd projects/01-gpu-system-inspector
uv run pytest                                  # unit + integration + host tests (no hardware)
uv run ruff check . && uv run ruff format --check .
```

## Dependencies

| Kind | Component |
| ---- | --------- |
| Required (installed with the tool) | `nvidia-ml-py` (NVIDIA's NVML bindings; works without a GPU) |
| Optional (detected, never installed) | PyTorch, inspected in the same Python environment as the tool |
| External (detected, never installed or modified) | NVIDIA driver/NVML, `nvidia-smi`, CUDA Toolkit |

## Distribution and license

- **Not published.** The package is marked `Private :: Do Not Upload`; install it from a
  checkout. The distribution name `01-gpu-system-inspector` is provisional (OQ-03); import
  package `gpu_system_inspector`; console script `inferforge-gpu`. Publication needs the
  owner decisions in [docs/roadmap.md](docs/roadmap.md#owner-decision-required).
- **License:** the InferForge repository is Apache-2.0 ([LICENSE](../../LICENSE)), and this
  project declares `Apache-2.0` in its metadata. The built wheel doesn't yet include the
  license text; that has to be fixed before any distribution of built artifacts.

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
| [hardware_validation.md](docs/hardware_validation.md) | How to produce and record NVIDIA hardware evidence (incl. Colab notebook) |
| [inferforge_nvidia_gpu_validation.ipynb](notebooks/inferforge_nvidia_gpu_validation.ipynb) | Reusable Google Colab notebook for NVIDIA validation |
| [demonstration.md](docs/demonstration.md) | Demonstration journeys A–G and recording plan |
| [troubleshooting.md](docs/troubleshooting.md) | Installation, platform and status reference |
| [roadmap.md](docs/roadmap.md) | Implemented, in progress, hardware-blocked, owner decisions, future |
| [app_tracker.md](docs/app_tracker.md) | **Single source of truth for status** |
