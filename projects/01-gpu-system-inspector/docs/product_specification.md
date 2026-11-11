# Product Specification — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. Nothing described here is implemented yet.
Implementation status lives only in [app_tracker.md](app_tracker.md).

## 1. Product definition

**GPU System Inspector** (command `inferforge-gpu`) is a lightweight, read-only,
developer-oriented command-line tool and Python library. It inspects the local machine across the system, GPU hardware, NVIDIA
driver, CUDA, framework (PyTorch) and runtime layers. It produces:

- a normalised inventory in which every value carries its **source, method, unit and
  collection status**;
- **evidence-based diagnostic findings** that separate direct observations, inferences
  and recommendations;
- an optional **per-GPU free-VRAM readiness assessment** against a user-supplied threshold;
- a **human-readable terminal report** and a **versioned JSON report**.

One-line positioning: *cross-layer GPU environment inspection + evidence-based
diagnostics + workload memory readiness + reproducible reports.*

It complements NVIDIA tools; it doesn't replace them
([existing_tools_analysis.md](existing_tools_analysis.md)).

### 1.1 Dependencies (approved 2026-11-04)

| Class | Component | Declared in `pyproject.toml` | If absent |
| ----- | --------- | ---------------------------- | --------- |
| Required Python dependency | `nvidia-ml-py` (import `pynvml`) | Yes, `dependencies` (DEC-05) | Only possible in a broken installation. NVML fields become `NOT_INSTALLED`, D05 `WARNING`, inspection continues. |
| Optional Python dependency | PyTorch (`torch`) | **No.** Detected at runtime and never installed (DEC-26) | `frameworks.pytorch.status = NOT_INSTALLED`; D05 `INFO`. |
| External system component | NVIDIA driver (kernel module, NVML library, `libcuda`) | No; never installed or modified | NVIDIA sections `NOT_DETECTED`; D04 explains; general inspection continues. |
| External tool | `nvidia-smi` | No | Fallback source unavailable; D05 `INFO` if NVML works. |
| External tool | CUDA Toolkit (`nvcc`) | No | `cuda_environment.toolkits = NOT_DETECTED`; D03 `INFO`. |

PyTorch is inspected **only in the Python environment that runs the inspector**: the
probe uses the same interpreter (DEC-07). To inspect an ML environment's PyTorch, install
and run the inspector in that environment. A different interpreter or venv is a future
option (`--python`, §7). So "PyTorch not installed" always means "not installed in this
environment" (KI-11).

## 2. Users and primary questions

| Question | Answered by |
| -------- | ----------- |
| What GPU hardware is available? Is an NVIDIA GPU detected? | GPU hardware layer, D04 |
| Is the NVIDIA driver available? Is NVML accessible? | Driver layer, D04, D05 |
| Which CUDA-related versions can be identified? | Driver + CUDA environment + framework layers, D03 |
| Is PyTorch installed? Can it access CUDA? | Framework layer, D01, D05 |
| How much GPU memory is available? | Runtime layer, readiness |
| What runtime metrics can be collected? | Runtime layer, D06, D07 |
| Are there identifiable configuration problems? | Diagnostics engine (D01–D08) |
| Does a GPU meet a basic free-memory requirement? | Readiness engine, D02 |
| What evidence supports the findings? | `evidence` on every finding |

Personas are described in [problem_statement.md](problem_statement.md) §2.

## 3. Product modes

### Mode A — General GPU inspection

```
inferforge-gpu inspect
inferforge-gpu inspect --json report.json     # also write the versioned JSON report
```

Runs every enabled collector, then the diagnostics engine, then renders a report.
Diagnostic findings never change the exit code in Mode A (exit `0` when the report is
produced; see [cli_specification.md](cli_specification.md) §6).

Layers and the information collected (full field list in
[data_model_specification.md](data_model_specification.md)):

| Layer | Information | Primary source |
| ----- | ----------- | -------------- |
| System | OS name, release, version; CPU architecture; processor model (where supported); logical CPU count; total and available system memory; Python version, implementation and executable; WSL2 detection | Python stdlib, `/proc`, Windows API via `ctypes`, `sysctl` |
| GPU hardware | Vendor; name; NVML index; UUID; total memory; compute capability; PCI bus ID and IDs; MIG mode; device count; Linux driver-independent PCI observations | NVML; `nvidia-smi` fallback; Linux sysfs |
| Driver & CUDA | Driver version; NVML library version; **driver-supported CUDA version** (labelled as such); CUDA driver library presence; `nvidia-ml-py` binding presence/version; `nvidia-smi` presence/path/version; **CUDA Toolkit** candidates and versions (`nvcc`, `version.json`); allowlisted CUDA environment variables | NVML, `nvidia-smi`, filesystem, `nvcc` |
| Framework | PyTorch distribution and import status; version; **build CUDA version**; cuDNN version; `cuda.is_available()` plus captured initialisation warnings; device count; per-device name, capability, total memory, SM count, UUID/PCI where exposed; NVML correlation; allocator statistics **only where meaningful** | PyTorch, probed in an isolated subprocess |
| Runtime | Per device: GPU utilization %, memory-bus utilization %, used/free/total/reserved memory, temperature, slowdown threshold, power draw, power limit, running processes (PID, type, memory, process name) | NVML; `nvidia-smi` fallback |

Three CUDA versions are always kept apart, because they have different meanings
(EV-03, EV-04, EV-05):

| Field | Meaning | Source |
| ----- | ------- | ------ |
| `driver.driver_supported_cuda_version` | Newest CUDA version the installed driver supports | NVML `nvmlSystemGetCudaDriverVersion`, or the `nvidia-smi` header |
| `cuda_environment.toolkits[].version` | Version of a locally installed CUDA Toolkit | `version.json`, `nvcc --version` |
| `frameworks.pytorch.build_cuda_version` | CUDA version PyTorch was built against (bundled runtime) | `torch.version.cuda` |

### Mode B — Workload-based free-VRAM readiness

```
inferforge-gpu inspect --required-vram 8
```

1. Performs the full Mode A inspection.
2. Interprets `8` as **8 GiB = 8 × 2³⁰ bytes** of **free** device memory.
3. Assesses **each** GPU independently using device-wide free memory from NVML (or the
   `nvidia-smi` fallback). The NVML snapshot is taken **before** the PyTorch probe, so
   the probe's own CUDA context can't distort the reading (DEC-12).
4. Gives each device `READY`, `NOT_READY` or `UNKNOWN`, with a reason code, the margin,
   and the evidence used.
5. Aggregates: overall `READY` if at least one device is `READY`; `NOT_READY` if every
   device was measured and none qualifies (or no NVIDIA device was positively observed to
   exist); otherwise `UNKNOWN`.
6. Always includes the disclaimer:
   > *Free memory at least equal to the requested amount was observed at the time of
   > inspection. This does not guarantee that a model will fit or run: actual usage depends
   > on weights, activations, KV cache, framework/allocator overhead, CUDA context size,
   > fragmentation and other processes that may allocate memory after this snapshot.*
7. Sets the exit code (`0` READY, `1` NOT_READY, `3` UNKNOWN; see
   [cli_specification.md](cli_specification.md) §6).

This isn't a model memory estimator, and it never states that a model "fits".

## 4. Platform support (V1 target)

| Platform | General system info | NVIDIA via NVML | `nvidia-smi` fallback | PCI presence check | CUDA Toolkit detection | PyTorch probe | Runtime metrics | Per-process GPU memory |
| -------- | ------------------- | --------------- | --------------------- | ------------------ | ---------------------- | ------------- | --------------- | ---------------------- |
| Linux x86_64 / aarch64 (bare metal, VM with passthrough) | Yes | Yes | Yes | Yes (sysfs) | Yes | Yes | Yes | Yes (privilege-dependent) |
| Windows 10/11 x86_64 | Yes | Yes | Yes | `NOT_SUPPORTED` (V1) | Yes | Yes | Yes | `NOT_SUPPORTED` under WDDM (EV-08) |
| WSL2 | Yes (flagged as WSL2) | Yes, when Windows driver exposes NVML (EV-09) | Yes (searches `/usr/lib/wsl/lib`) | `NOT_SUPPORTED` (GPU not on PCI, OBS-01) | Yes | Yes | Partial: utilization and processes expected `NOT_SUPPORTED` (EV-09) | `NOT_SUPPORTED` expected |
| macOS (arm64 / x86_64) | Yes | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | Yes (reports CUDA unavailable) | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| Containers | As for host OS; NVIDIA data only if the container runtime exposes the devices and libraries | | | | | | | |

Not supported in V1: Jetson/iGPU SoCs (NVML memory queries may be `NOT_SUPPORTED`,
EV-06), MIG instance enumeration (MIG mode is detected and reported as a limitation),
AMD/Intel GPUs (reported only as non-NVIDIA PCI observations on Linux, without metrics).

## 5. V1 scope (confirmed features)

| ID | Feature | Requirement refs |
| -- | ------- | ---------------- |
| F01 | System inventory | FR-02 |
| F02 | GPU hardware inventory via NVML, multi-GPU | FR-03, FR-25 |
| F03 | Linux driver-independent NVIDIA PCI presence check | FR-04 |
| F04 | Driver, NVML and driver-supported CUDA version inspection | FR-05 |
| F05 | `nvidia-smi` detection and fallback query | FR-06 |
| F06 | CUDA Toolkit detection and CUDA environment variables | FR-07, FR-08 |
| F07 | Optional PyTorch inspection (never installed automatically) | FR-09, FR-10 |
| F08 | Runtime GPU snapshot (utilization, memory, temperature, power, processes) | FR-11 |
| F09 | Per-value provenance and collection status | FR-12 |
| F10 | Collector isolation and graceful degradation | FR-13 |
| F11 | Evidence-based diagnostics D01–D08 | FR-14 |
| F12 | Basic free-VRAM readiness (Mode B) | FR-15 |
| F13 | Terminal report | FR-16 |
| F14 | Versioned JSON report (`--json PATH` / `--json -`) plus shipped JSON Schema | FR-17, FR-18 |
| F15 | Opt-in redaction | FR-19 |
| F16 | `version` and `--help` commands; documented exit codes | FR-20, FR-21 |
| F17 | Collector opt-outs (`--no-pytorch`, `--no-processes`) | FR-22 |
| F18 | Provisional Python API (`inspect()` + models) | FR-23 |
| F19 | Automated tests without GPU hardware; separate hardware tests | NFR-10, [testing_strategy.md](testing_strategy.md) |
| F20 | Documentation | NFR-17 |

## 6. Explicit non-goals for V1

| Non-goal | Use instead |
| -------- | ----------- |
| Continuous time-series monitoring, watch mode | `nvidia-smi dmon`, `gpustat --watch`, DCGM |
| GPU fleet monitoring, Prometheus export | DCGM, `dcgm-exporter` |
| DCGM replacement, active health diagnostics, stress tests | `dcgmi diag` |
| Kernel profiling | Nsight Compute |
| System/timeline profiling | Nsight Systems |
| Distributed GPU management, scheduling | Cluster tooling |
| Full model memory estimation ("will my model fit?") | Future InferForge project |
| GPU benchmarking suite | Future InferForge project |
| Web dashboard | — |
| Remote inspection (SSH, agents, network) | — |
| Modifying GPU state (clocks, power, persistence, ECC, killing processes) | `nvidia-smi` (admin) |
| Installing or repairing drivers, CUDA or PyTorch | Vendor installers; findings give recommendations only |
| AMD ROCm / Intel / Apple GPU metrics | Vendor tools |
| Thermal-throttling inference from temperature | NVML clock event reasons (future, see §7) |
| Demo or simulation mode (decided 2026-11-04) | — (tests use fixtures; users always see `LIVE` data) |
| Declaring PyTorch as a dependency or packaging extra | PyTorch's own install selector (CUDA-variant wheels) |

## 7. Future possibilities (recorded, not planned for V1) `[FUT]`

- Inspect a *different* Python environment's PyTorch via `--python <interpreter>`
  (the subprocess probe design makes this cheap).
- MIG instance enumeration and per-instance readiness.
- NVML clock event (throttle) reasons as direct evidence for D07.
- Cross-validation of NVML against `nvidia-smi` values.
- `torch.cuda.mem_get_info()` as an opt-in readiness source.
- Additional frameworks (JAX, TensorFlow, ONNX Runtime, TensorRT, vLLM).
- Container runtime detection (NVIDIA Container Toolkit, `NVIDIA_VISIBLE_DEVICES`).
- Windows and macOS driver-independent GPU presence checks.
- Report diffing (`inferforge-gpu diff a.json b.json`).
- Model-aware memory estimation, as a separate InferForge project.
- A `--quiet` option to suppress the text report when only `--json PATH` and the exit code are needed.
- Colour terminal output; Markdown/HTML report formats.

## 8. Key product rules

1. **Read-only.** The tool never changes GPU, driver or system state, and never needs root
   or Administrator rights (it reports `PERMISSION_DENIED` instead).
2. **No silent zeros.** A missing value is `null` with a status, never `0` or `""`.
3. **No overclaiming.** Findings state observations as observations and inferences as
   inferences, with a confidence and limitations. A root cause is never claimed without
   sufficient evidence.
4. **No fake data.** There is no demo or simulation mode. Mocked or fixture data exists
   only in tests, is always marked `data_origin = "FIXTURE"`, and can't be produced by the
   CLI (DEC-17).
5. **No network.** The tool makes no network connections.
6. **No automatic installation.** Missing dependencies, drivers or tools produce recommendations only.
7. **Partial results beat no results.** A failing collector never stops the others.
8. **Works without NVIDIA.** No NVIDIA GPU, driver or tool is needed for the inspector to run
   and produce a complete report.

## 9. Related documents

[problem_statement.md](problem_statement.md) ·
[requirements.md](requirements.md) ·
[architecture.md](architecture.md) ·
[data_flow.md](data_flow.md) ·
[data_model_specification.md](data_model_specification.md) ·
[cli_specification.md](cli_specification.md) ·
[diagnostics_specification.md](diagnostics_specification.md) ·
[testing_strategy.md](testing_strategy.md) ·
[acceptance_criteria.md](acceptance_criteria.md) ·
[existing_tools_analysis.md](existing_tools_analysis.md) ·
[technical_evidence.md](technical_evidence.md) ·
[app_tracker.md](app_tracker.md)
