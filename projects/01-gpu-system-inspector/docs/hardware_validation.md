# Hardware Validation Procedure — Project 01: GPU System Inspector

Status (2026-11-11): **BLOCKED.** The development machine (WSL2, no NVIDIA GPU exposed) can't
validate NVIDIA behaviour. Everything below has been run there, and it correctly reports
`BLOCKED`. No NVIDIA hardware result exists yet. The tracker's hardware table is the record:
[app_tracker.md](app_tracker.md) §6.

This document says how to produce hardware evidence on Google Colab or a native Linux machine
with an NVIDIA GPU, what the evidence means, and what to send back.

The preferred Colab path is the project notebook
[`notebooks/inferforge_nvidia_gpu_validation.ipynb`](../notebooks/inferforge_nvidia_gpu_validation.ipynb).
It drives the same CLI, hardware tests and evidence script documented here; it does **not**
reimplement diagnostics. A successful notebook run on Colab validates **only the recorded
Colab environment** (GPU model, driver, software versions, source revision). It does not by
itself change tracker statuses to `HARDWARE_VALIDATED`.

## 0. Google Colab notebook (recommended)

End-to-end contributor path once Project 01 is on a **public** branch:

**GitHub → open notebook in Colab → GPU runtime → notebook downloads source → install from
that tree → inspect / hardware tests / evidence → download evidence → attach to a GitHub
issue.**

In `SOURCE_MODE = "git"` (the notebook default) contributors do **not** manually download or
upload a Project 01 `.tar.gz`. The notebook fetches the public GitHub source archive itself.

### 0.1 Opening the notebook

**Publication check.** An "Open in Colab" badge is valid only after this notebook exists on a
**published** GitHub branch. As of the last remote check, `origin/main` does **not** contain
Project 01's validation assets, so there is **no** live badge URL yet. Do not invent one.

| Situation | How to open |
| --------- | ----------- |
| Notebook pushed to public branch `<branch>` | Open<br>`https://colab.research.google.com/github/nawalkumar/InferForge/blob/<branch>/projects/01-gpu-system-inspector/notebooks/inferforge_nvidia_gpu_validation.ipynb`<br>and use the badge in the [project README](../README.md) (enabled only after that push). |
| Work not pushed yet | In Colab: **File → Upload notebook** and choose `inferforge_nvidia_gpu_validation.ipynb` from a local checkout, **or** upload it via the Files pane and open it. Use `SOURCE_MODE = "archive"` for the source tree (below). |

No GitHub personal access token belongs in any notebook cell. Public downloads need no
credentials.

### 0.2 Select a Colab GPU runtime

1. **Runtime → Change runtime type → GPU** (for example T4).
2. Confirm with `nvidia-smi` (Section 2 of the notebook does this and **stops with `BLOCKED`**
   if no GPU is visible).
3. GPU availability depends on Colab account limits and capacity. A CPU runtime must never be
   treated as NVIDIA hardware validation.

### 0.3 Run cells in order

1. Edit the **Configuration** cell if needed (`GIT_BRANCH`, optional opt-ins). Default
   `SOURCE_MODE = "git"`.
2. Run **Runtime verification** — must show an NVIDIA GPU.
3. **Obtain source (GitHub-first):**
   - `SOURCE_MODE = "git"` (**recommended**): the notebook resolves the commit (when the
     public API allows), downloads
     `https://github.com/nawalkumar/InferForge/archive/refs/heads/<GIT_BRANCH>.tar.gz`
     (or `/archive/<GIT_REF>.tar.gz`), extracts only under `/content/inferforge-validation/`,
     and uses `projects/01-gpu-system-inspector/`. No manual archive upload. Fails clearly if
     the ref is missing or does not contain Project 01 — it does **not** fall back to PyPI
     or a stale tree.
   - `SOURCE_MODE = "archive"` (**fallback** for unpublished, private, or offline work): build
     and upload a project tarball (commands in the notebook Section 3 / §2 below), then run
     the cell.
4. **Install** from that local tree only: `pip install . pytest jsonschema` (never PyPI).
5. **Live inspection**, **hardware tests**, **evidence bundle**, **download**.

Optional CUDA execution (`OPT_IN_CUDA_SMOKE`) and HW-12 allocation (`OPT_IN_HW12_ALLOCATION`)
stay **off** unless you enable them deliberately. They run outside ordinary inspection.

### 0.4 Download and share results

The notebook writes `/content/gsi-evidence/` and `/content/gsi-evidence.tar.gz`. Provenance is
recorded in `PROVENANCE.md` and `notebook-run-meta.json` (repository, branch, commit SHA or
`unknown`, source mode, timestamp, GPU model, driver, CUDA-related env, test outcomes).
Attach the archive (or at least `SUMMARY.md`, `summary.json`, `hardware-tests.xml`,
`PROVENANCE.md`) to a GitHub issue or send it to the owner (§5). Unredacted files contain GPU
UUIDs, PCI bus IDs and paths — share those privately.

### 0.5 Known Colab limitations

- One GPU is typical → HW-09 (multi-GPU) skips.
- PCI (HW-02) is "native Linux in a VM/container", not bare metal.
- A CUDA Toolkit is often present → HW-06 (PyTorch without a toolkit) usually skips.
- Sessions can disconnect; re-run from a clean GPU runtime if that happens.
- An old `pynvml` package may conflict with `nvidia-ml-py`; note it — the report records
  which binding was imported (D05 / FX20).

### 0.6 Reproducing on another NVIDIA machine

Use the same notebook on Colab with a different GPU runtime, or run the CLI procedure in §3
on native Linux / WSL2-with-NVIDIA / Windows. Keep the evidence label, source revision and
`SUMMARY.md` environment block so runs stay comparable. Tracker updates still require owner
review of the evidence.

## 1. What is run

| Part | What it is | Touches the GPU? |
| ---- | ---------- | ---------------- |
| Inspector runs | The real `inferforge-gpu inspect` CLI: Mode A, Mode A `--redact`, Mode B `--required-vram` | Read-only NVML and `nvidia-smi` queries; the PyTorch probe child may initialise CUDA (no tensor, no kernel) |
| Reference tools | `nvidia-smi` (header, `-L`, `--query-gpu`), `nvcc --version`, `/proc/driver/nvidia/version`, PyTorch's own API in a separate child | Same as above; nothing allocated |
| Hardware tests | `pytest -m hardware tests/hardware` (HW-01 to HW-13, [testing_strategy.md](testing_strategy.md) §6) | Read-only, except HW-12 (opt-in, below) |
| Optional CUDA execution check | `--cuda-smoke-test`: a child process computes `sum(2·x)` for 1024 floats on `cuda:0` | **Yes**, in its own child, after everything else |
| Optional HW-12 | `--allow-cuda-allocation`: a child allocates 75 % of free memory while readiness is re-checked | **Yes**, in its own child |

The two optional parts are off by default. They're recorded in their own sections and never
feed the inspector's report: an inspection never runs CUDA code, and a successful CUDA
computation never turns a finding or readiness result into `PASS`/`READY`.

The script is [`validation/hardware_evidence.py`](../validation/hardware_evidence.py).

## 2. Google Colab (manual cells, same commands as the notebook)

Prefer §0. For a manual Colab session without the notebook UI:

1. **Runtime → Change runtime type → a GPU** (for example T4). Check it: `!nvidia-smi`.
2. **Get the project** (prefer the notebook's GitHub archive download; manual equivalent):
   - After Project 01 is on a published branch `<branch>` (no auth for a public repo):
     ```
     !wget -q -O /tmp/inferforge.tar.gz https://github.com/nawalkumar/InferForge/archive/refs/heads/<branch>.tar.gz
     !mkdir -p /content/inferforge-validation && tar xzf /tmp/inferforge.tar.gz -C /content/inferforge-validation
     %cd /content/inferforge-validation/InferForge-<branch>/projects/01-gpu-system-inspector
     ```
     (GitHub may name the top folder `InferForge-<short-sha>` when fetching by commit; adjust the path.)
   - Or, for unpublished / offline work, upload a tarball made on the development machine:
     ```bash
     cd projects && tar czf /tmp/gsi-project.tar.gz --exclude=__pycache__ \
         --exclude=.pytest_cache --exclude=.coverage --exclude='*.pyc' \
         01-gpu-system-inspector
     ```
     Upload it with the Colab **Files** pane, then `!tar xzf gsi-project.tar.gz` and
     `%cd 01-gpu-system-inspector`.
3. **Install into Colab's own Python** from that local tree only (never PyPI), so the PyTorch
   probe inspects Colab's PyTorch:
   ```
   !pip install -q . pytest jsonschema
   !python -m gpu_system_inspector version
   ```
4. **Collect the evidence** (about one to three minutes):
   ```
   !python validation/hardware_evidence.py --label "Colab T4" --out /content/gsi-evidence
   ```
   Optional, and only if you want them: `--cuda-smoke-test`, `--allow-cuda-allocation`.
5. **Read and download:**
   ```
   !cat /content/gsi-evidence/SUMMARY.md
   !tar czf /content/gsi-evidence.tar.gz -C /content gsi-evidence
   from google.colab import files; files.download("/content/gsi-evidence.tar.gz")
   ```

Colab limitations are listed in §0.5. If `pip` reports a conflict with an old `pynvml`
distribution, note it: the report records which binding was imported
(`driver.nvml_binding`, `driver.nvml_binding_distribution`), and D05 reports a third-party
`pynvml` (fixture FX20).

## 3. Native Linux with an NVIDIA GPU

Use the Python environment that has the PyTorch you want inspected (Python 3.11 or newer):

```bash
git clone https://github.com/nawalkumar/InferForge.git      # or unpack the tarball above
cd InferForge/projects/01-gpu-system-inspector
python -m pip install . pytest jsonschema                     # into the active environment
python validation/hardware_evidence.py --label "<machine, GPU>" --out ~/gsi-evidence
```

In the uv workspace, `uv run python validation/hardware_evidence.py …` works too, but the
workspace environment has no PyTorch, so the PyTorch tests skip.

Other machines the hardware table needs, with the same commands: WSL2 with the Windows NVIDIA
driver (HW-07), two or more GPUs (HW-09), Windows (HW-08, run from PowerShell with `python`),
macOS (HW-11).

## 4. Reading the result

`SUMMARY.md` starts with the verdict:

| Verdict | Meaning |
| ------- | ------- |
| `BLOCKED: no NVIDIA GPU is accessible in this environment` | NVML counted no device; nothing NVIDIA-specific can be validated here. |
| `EVIDENCE_COLLECTED: owner review required …` | An NVIDIA GPU was inventoried; the evidence is ready for review. It is **not** a validation by itself. |

Each feature the inspector reports is classified from the Mode A report:

| Classification | Collection status | Meaning |
| -------------- | ----------------- | ------- |
| `SUPPORTED_AND_OBSERVED` | `COLLECTED` | The value was read from its source. |
| `SUPPORTED_BUT_UNAVAILABLE` | `NOT_DETECTED`, `NOT_INSTALLED`, `PERMISSION_DENIED` | The inspector supports it, but it wasn't there or wasn't accessible. |
| `NOT_SUPPORTED_ON_THIS_ENVIRONMENT` | `NOT_SUPPORTED` | The platform or driver doesn't offer it (for example PCI under WSL2). |
| `NOT_IMPLEMENTED` | `NOT_SUPPORTED` with code `NOT_IMPLEMENTED` | The inspector doesn't implement it. |
| `FAILED` | `COLLECTION_FAILED`, `UNKNOWN` | Collection was attempted and failed. |
| `SKIPPED_BY_OPTION`, `NOT_APPLICABLE` | `SKIPPED`, `NOT_APPLICABLE` | Neither available nor missing (an option, or meaningless here). |
| `NO_DATA` | — | Nothing to classify, e.g. no device was inventoried. |

`SUMMARY.md` also lists the features the inspector doesn't implement at all (graphics
processes KI-36, the CUDA runtime library version, MIG instances, clock event reasons, the
Windows default toolkit location KI-37, container detection), so their absence is never read
as "not present on the machine".

Hardware test outcomes: **a skipped test is not a pass**; its message says which
precondition was missing. HW-05 records both the detected toolkit version and `nvcc`'s
(assumption C12) and asserts that their release (major.minor) agrees, which is what D03 uses.

## 5. What to send back

To record a result, the owner needs:

1. `gsi-evidence.tar.gz` (privately: the unredacted files contain GPU UUIDs, PCI bus IDs and
   paths), or at least `SUMMARY.md`, `summary.json` and `hardware-tests.xml`;
2. who ran it, when, and on which machine (cloud provider and instance type, or hardware);
3. anything unusual (driver installed that day, other GPU workloads running, pip conflicts).

The owner then records each HW item in [app_tracker.md](app_tracker.md) §6 with date,
machine, OS, driver, GPU model, PyTorch version and outcome. Only a **passed** hardware test
with its evidence may move an item to `HARDWARE_VALIDATED`; failures become known issues.

## 6. Current status of each hardware item

| ID | Needs | Status |
| -- | ----- | ------ |
| HW-01, HW-03, HW-04, HW-05, HW-12 | Any Linux machine with an NVIDIA GPU (Colab is enough) | `BLOCKED`: no run yet |
| HW-02 | Native Linux (Colab gives VM/container evidence) | `BLOCKED` |
| HW-06 | PyTorch CUDA wheel and no CUDA Toolkit | `BLOCKED` |
| HW-07 | WSL2 with the Windows NVIDIA driver | `BLOCKED` |
| HW-08 | Windows with an NVIDIA GPU | `BLOCKED` |
| HW-09 | Two or more NVIDIA GPUs | `BLOCKED` |
| HW-10 | Any machine | No-driver path measured on the WSL2 machine (§1.14 of the tracker); NVML path `BLOCKED` |
| HW-11 | macOS | `BLOCKED` |
| HW-13 | A machine without an NVIDIA driver | Passed on the WSL2 development machine, 2026-11-11 |
