# Troubleshooting — GPU System Inspector

The inspector reports what it can see and says why the rest is missing. Most "problems" are
the report doing its job: read the status and reason next to each value first. This page
covers installation and platform issues, and how to read the commonest statuses.

## Installation

| Symptom | Cause | What to do |
| ------- | ----- | ---------- |
| `requires a different Python: … not in '>=3.11'` | Python older than 3.11 (for example Ubuntu 22.04's system `python3` is 3.10) | Use Python 3.11 or newer: `uv python install 3.11`, the deadsnakes PPA, or a conda/pyenv environment. |
| `uv: command not found` | uv isn't installed | Install it ([uv docs](https://docs.astral.sh/uv/getting-started/installation/)), or use the `pip` route in the README. |
| `inferforge-gpu: command not found` after `uv sync` | The command lives in the workspace environment | Run it as `uv run inferforge-gpu …`, or activate `.venv`. |
| `inferforge-gpu: command not found` after `pip install` | The environment's `bin/` (or `Scripts\`) isn't on `PATH` | Activate the virtual environment, or run `python -m gpu_system_inspector …`. |
| `pip install 01-gpu-system-inspector` finds nothing | The package isn't published (classifier `Private :: Do Not Upload`) | Install from a checkout: `python -m pip install ./projects/01-gpu-system-inspector`. |
| `version` prints `nvidia-ml-py NOT INSTALLED (required dependency missing)` | A broken installation removed the required NVML binding | Reinstall the inspector; the report then shows NVML fields `NOT_INSTALLED` and D05 `WARNING`. |
| D05 mentions a `pynvml` distribution | A third-party `pynvml` package is installed next to, or instead of, NVIDIA's `nvidia-ml-py` | Prefer `nvidia-ml-py`; the report records which binding was imported (`driver.nvml_binding_distribution`). |

## Platform

| Situation | What the report shows | Meaning |
| --------- | --------------------- | ------- |
| No NVIDIA driver (common on laptops, CI, WSL2 without GPU support) | NVML `NOT_DETECTED (NVML_ERROR_LIBRARY_NOT_FOUND)`, `nvidia-smi` `NOT_DETECTED`, D04 `INFO` | The driver's user-mode library isn't loadable. The inspector does **not** conclude that no GPU exists unless a complete PCI scan (native Linux) shows none. |
| WSL2 | PCI scan `NOT_SUPPORTED (PLATFORM_NOT_SUPPORTED)`; `nvidia-smi` looked up in `/usr/lib/wsl/lib` | GPUs aren't enumerated on PCI inside WSL2. Install or update the NVIDIA driver on the **Windows** host, never inside WSL2 ([NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)). Some runtime fields are `NOT_SUPPORTED` there by design. |
| macOS | NVIDIA sections `NOT_SUPPORTED`; D04 `INFO` | General system information only, by design. |
| Windows | Per-process GPU memory `NOT_SUPPORTED` under WDDM | A driver limitation, not a failure. Windows has only been tested with fakes (KI-27). |
| Containers | Whatever the container exposes | NVML works when the container runtime exposes the GPU and driver libraries; `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` are listed in the CUDA TOOLKIT section. Container detection isn't implemented. |

## Reading statuses

| Status | Meaning |
| ------ | ------- |
| `COLLECTED` | The value was read; its source (NVML, `NVIDIA_SMI`, OS, …) is shown in brackets. |
| `NOT_DETECTED` | Looked for and not found (no library, no executable, variable unset). |
| `NOT_INSTALLED` | A Python distribution (PyTorch, the NVML binding) isn't installed in **this** environment. |
| `NOT_SUPPORTED` | The platform or driver doesn't offer it; or, with code `NOT_IMPLEMENTED`, the inspector doesn't. |
| `PERMISSION_DENIED` | Exists but the current user can't access it. |
| `COLLECTION_FAILED` | Collection was attempted and failed; the error says how. A collector crash also adds a `COLLECTION ERRORS` entry and a `WARNING` line on stderr. |
| `SKIPPED` | Turned off by an option (`--no-pytorch`, `--no-processes`). |
| `NOT_APPLICABLE` | Meaningless here (for example a CUDA version for a CPU-only PyTorch build). |

A missing value is never shown as `0`.

## PyTorch

- PyTorch is inspected **only in the Python environment that runs the inspector** (KI-11).
  If the report says `NOT_INSTALLED` but you use PyTorch elsewhere, install the inspector into
  that environment and run it from there.
- `CUDA available` is PyTorch's own `torch.cuda.is_available()` answer from a child process.
  The inspector runs no CUDA code, so a `True` there isn't proof that a kernel will run.
- A slow first import of PyTorch can hit the probe timeout: raise `--pytorch-timeout`
  (default 60 s).

## Exit codes in scripts

| Exit | When |
| ---- | ---- |
| 0 | Report produced (Mode B: at least one GPU has enough free memory) |
| 1 | Mode B: no GPU has enough free memory |
| 2 | Invalid usage; nothing was inspected |
| 3 | Mode B: readiness couldn't be determined (for example no accessible NVIDIA GPU) |
| 4 | The JSON report couldn't be written (missing directory, read-only location); the text report is still printed and no partial file is left |
| 5 | Internal error: please report it (see [CONTRIBUTING.md](../CONTRIBUTING.md#reporting-a-bug)) |
| 130 | Interrupted (Ctrl-C) |

## Reporting a problem

Attach a **redacted** report, which replaces paths, process names, GPU UUIDs, PCI bus IDs,
your user name and home directory:

```bash
inferforge-gpu version
inferforge-gpu inspect --redact --json report.json
```

Then follow [CONTRIBUTING.md](../CONTRIBUTING.md#reporting-a-bug).
