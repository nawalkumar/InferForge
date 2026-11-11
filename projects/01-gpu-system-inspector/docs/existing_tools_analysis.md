# Existing Tools Analysis — Project 01

This analysis is about **positioning**. Each tool below is authoritative in its own domain,
and InferForge Project 01 doesn't try to outperform any of them there. Statements are
limited to what the cited documentation says ([technical_evidence.md](technical_evidence.md)).
Where a capability isn't documented in a cited source, the matrix says
"not documented here" instead of "no".

## 1. NVIDIA System Management Interface (`nvidia-smi`)

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | Monitoring and management of NVIDIA GPUs from the command line (EV-01). |
| Main capabilities | Human-readable device summary; detailed `-q` query; scripted `--query-gpu` / `--query-compute-apps` CSV; XML output with DTD; management operations (persistence mode, clocks, power limits, etc.) (EV-01, EV-02). |
| Data exposed | Driver (KMD) version, CUDA UMD version (newest CUDA supported by the driver), product name, UUID, PCI, memory, utilization, temperature, power, clocks, processes, ECC and more (EV-03). |
| Strengths | Ships with the driver; authoritative. Documented coverage of Tesla, Quadro, GRID and GeForce devices from Fermi onward, with limited information for some GeForce products. Supports Linux and 64-bit Windows (EV-01). |
| Limitations for our use case | Output "is not guaranteed to be backwards compatible" (EV-01). Has no PyTorch/framework layer and no toolkit (`nvcc`) detection. Its "CUDA Version" header is frequently misread as the toolkit version (EV-03). Doesn't explain *why* something is unavailable across layers. Not present on machines without the NVIDIA driver, and not on `PATH` by default in WSL2 (EV-09). |
| Integration | **Secondary source.** Detect presence, path and version. Use `--query-gpu ... --format=csv,noheader,nounits` as a fallback only when NVML via Python is unavailable (DEC-04). Every value from it is tagged `source = NVIDIA_SMI`. |
| Avoid duplicating | Its human-readable dashboard, management/set operations, `dmon`/`pmon` monitoring loops and exhaustive `-q` coverage. |

## 2. NVIDIA Management Library (NVML)

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | C API for monitoring and managing NVIDIA GPUs; the library underneath `nvidia-smi` (EV-01, EV-06). |
| Main capabilities | System queries (driver version, NVML version, CUDA driver version), device queries (name, UUID, PCI, memory, compute capability, utilization, temperature, thresholds, power, processes, MIG) and management operations (EV-04, EV-06). |
| Data exposed | Raw numeric values with documented units (bytes, °C, milliwatts, percent) and typed error codes (`NOT_SUPPORTED`, `NO_PERMISSION`, `LIBRARY_NOT_FOUND`, ...) (EV-06, EV-11). |
| Strengths | Backward compatible across driver releases (EV-01); official Python bindings `nvidia-ml-py` (EV-11); precise error codes that map directly onto our collection statuses. |
| Limitations for our use case | A library, not a diagnosis. Knows nothing about PyTorch or the CUDA Toolkit. Some queries are unsupported on WSL2 (EV-09), under Windows WDDM (EV-08), on iGPU SoCs, or in MIG mode without privileges (EV-06). |
| Integration | **Primary hardware/driver/runtime source** through `nvidia-ml-py` (DEC-04, DEC-05). Read-only calls only. Never call `nvmlDeviceSet*` or other mutating APIs (NFR-12). |
| Avoid duplicating | Re-implementing NVML queries by other means when NVML is available; wrapping the whole NVML API surface. |

## 3. NVIDIA Data Center GPU Manager (DCGM)

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | Administering datacenter GPUs in cluster environments (EV-15). |
| Main capabilities | Behaviour monitoring, configuration management, policy, health and diagnostics, accounting and process statistics, NVSwitch support. Active diagnostics via `dcgmi diag` at levels 1–4. Prometheus export via `dcgm-exporter` (EV-15, EV-16). |
| Data exposed | Continuous telemetry fields, health incidents, diagnostic test results, job statistics (EV-15). |
| Strengths | Designed for fleets and schedulers; active hardware validation; low-overhead continuous telemetry; C, Python and Go APIs (EV-15). |
| Limitations for our use case | Needs a host engine (embedded or standalone) and a separate installation. Higher diagnostic levels target datacenter GPUs, and non-datacenter GPUs support level 1 (EV-15). Active diagnostics "can consume substantial GPU, CPU, memory, power, and fabric resources" (EV-15). `dcgm-exporter` is Linux-only (EV-16). No PyTorch layer. |
| Integration | **No integration in V1.** When findings point to possible hardware faults, recommend DCGM diagnostics where the user has DCGM. Future: optionally detect DCGM presence `[FUT]`. |
| Avoid duplicating | Active diagnostics or stress tests, health watches, policy, fleet grouping, Prometheus export, continuous telemetry. |

## 4. `gpustat`

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | "A simple command-line utility for querying and monitoring GPU status", "just *less* than nvidia-smi" (EV-17). |
| Main capabilities | One-line-per-GPU status (index, name, temperature, utilization, memory, processes), watch mode, JSON output, user/command/PID display, Python API; NVIDIA via `nvidia-ml-py`, plus AMD and Ascend backends (EV-17). |
| Strengths | Concise one-line-per-GPU view; multi-vendor backends; JSON output; process/user/command view; Python API (EV-17). |
| Limitations for our use case | A status view, not an environment diagnosis. No PyTorch layer, no CUDA toolkit/driver version reasoning, no readiness verdict. Diagnostic explanations are not documented in the cited README. |
| Integration | **Complementary, no dependency.** We follow its documented advice to depend on `nvidia-ml-py`, not `pynvml` (EV-17). Users who want a live view should use `gpustat --watch`. |
| Avoid duplicating | Watch/refresh mode, compact status line, colourised live display, multi-vendor backends. |

## 5. PyTorch CUDA APIs (`torch.cuda`)

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | CUDA support inside PyTorch: device selection, streams, memory management (EV-12, EV-13). |
| Main capabilities | `is_available()`, `device_count()`, `get_device_properties()`, `get_device_capability()`, `mem_get_info()`, allocator statistics (`memory_allocated`, `memory_reserved`, `max_*`, `memory_stats`, `memory_snapshot`), build metadata (`torch.version.cuda`) (EV-12). |
| Data exposed | Framework-visible devices (affected by `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`, EV-10, EV-13); per-process allocator statistics; device-wide free/total via `cudaMemGetInfo`. |
| Strengths | Ground truth for "can *this* PyTorch use CUDA?"; detailed allocator introspection in-process. |
| Limitations for our use case | `is_available()` returns a boolean, not a reason. Allocator statistics describe only the calling process and are `0` before CUDA is initialised (EV-12). Importing `torch` is heavy. Device indices may differ from NVML (EV-10). |
| Integration | **Optional adapter** (DEC-07). Probed in an isolated subprocess using the same interpreter, without installing anything. Allocator stats are reported only where meaningful (DEC-08). Devices are correlated with NVML by UUID/PCI bus ID (DEC-09). |
| Avoid duplicating | Memory snapshots/visualisation, allocator tuning, CUDA graph tooling, profiler integration. |

## 6. NVIDIA Nsight Systems

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | System-wide performance analysis on a unified CPU/GPU/OS timeline (EV-18). |
| Main capabilities | Tracing CUDA, cuBLAS, cuDNN, TensorRT, graphics APIs, OS and network activity; NVTX ranges (EV-18). |
| Strengths | Shows *when* and *why* an application is slow across CPU and GPU (EV-18). |
| Limitations for our use case | Profiles a running workload; it isn't an environment readiness check. Needs a separate install. |
| Integration | **None in V1.** Mention it as the next step when users ask performance questions. |
| Avoid duplicating | Tracing, timelines, NVTX, any performance profiling. |

## 7. NVIDIA Nsight Compute

| Aspect | Notes |
| ------ | ----- |
| Primary purpose | Interactive kernel profiler for CUDA applications (EV-19). |
| Main capabilities | Per-kernel metrics, API stream stepping, report comparison, CLI and UI (EV-19). |
| Strengths | Deep per-kernel performance analysis (EV-19). |
| Limitations for our use case | Kernel-level, workload-attached; not an environment inspector. |
| Integration | **None in V1.** |
| Avoid duplicating | Kernel profiling, roofline/metric analysis. |

## 8. Feature comparison matrix

Legend: **Yes** = documented capability in the cited source; **Partial** = documented
but limited for our purpose (explained in the notes); **—** = not documented in the cited
source; **Planned** = InferForge V1 requirement (not yet implemented).

| Capability | nvidia-smi | NVML | DCGM | gpustat | torch.cuda | Nsight Systems | Nsight Compute | InferForge P01 (V1) |
| ---------- | ---------- | ---- | ---- | ------- | ---------- | -------------- | -------------- | ------------------- |
| GPU inventory (name, UUID, memory) | Yes | Yes | Yes | Partial¹ | Partial² | — | — | Planned |
| Driver version | Yes | Yes | Yes | — | — | — | — | Planned |
| Driver-supported CUDA version | Yes | Yes | — | — | — | — | — | Planned (labelled distinctly) |
| CUDA Toolkit (`nvcc`) version | — | — | — | — | — | — | — | Planned |
| Framework build CUDA version | — | — | — | — | Yes | — | — | Planned |
| "Can the framework use CUDA?" | — | — | — | — | Yes (boolean) | — | — | Planned (with evidence) |
| Runtime snapshot (util, memory, temp, power) | Yes | Yes | Yes | Yes | Partial³ | Yes (during trace) | — | Planned |
| Continuous monitoring / watch | Yes | Partial⁴ | Yes | Yes | — | — | — | Out of scope |
| Per-process GPU usage | Yes | Yes | Yes | Yes | — | Yes | — | Planned (subject to platform) |
| Active hardware diagnostics | — | — | Yes | — | — | — | — | Out of scope |
| Fleet / cluster management | — | — | Yes | — | — | — | — | Out of scope |
| Performance profiling | — | — | Partial⁵ | — | Partial⁶ | Yes | Yes | Out of scope |
| Cross-layer correlation (HW ↔ driver ↔ CUDA ↔ framework) | — | — | — | — | — | Partial⁷ | — | Planned |
| Evidence-based explanations of failures | — | — | Partial⁸ | — | — | — | — | Planned |
| Explicit per-value source + collection status | — | Partial⁹ | — | — | — | — | — | Planned |
| Per-GPU free-VRAM readiness verdict | — | — | — | — | — | — | — | Planned |
| Versioned JSON report schema | Partial¹⁰ | — | — | Partial¹¹ | — | — | — | Planned |
| Report redaction | — | — | — | — | — | — | — | Planned |
| Works without NVIDIA driver (general info) | — | — | — | — | Yes | — | — | Planned |

Notes:
1. `gpustat` shows name, memory and processes. UUID display isn't documented in the cited README.
2. Only devices visible to PyTorch, indexed in CUDA order.
3. `mem_get_info()` for device-wide memory; no temperature or power.
4. NVML provides queries and events; looping is up to the caller.
5. DCGM profiling metrics are telemetry fields, not kernel profiling (EV-15, EV-16).
6. PyTorch has its own profiler; it's outside the scope of this analysis.
7. Nsight Systems correlates CPU and GPU activity of a traced workload, not environment configuration.
8. DCGM health and diagnostics report hardware/software checks for the GPU subsystem, not framework or toolkit configuration.
9. NVML returns typed error codes per call, but callers must surface them.
10. XML output conforms to a DTD (EV-01). CSV/human output isn't guaranteed stable.
11. `--json` output exists. A versioned schema isn't documented in the cited README.

## 9. The actual product gap

No single cited tool combines **(a)** a hardware/driver/CUDA/framework inventory,
**(b)** explicit per-value provenance and collection status, **(c)** evidence-based findings
that separate observation, inference and recommendation, **(d)** a per-device free-VRAM
readiness verdict with honest limitations, and **(e)** a versioned, redactable JSON report.

InferForge Project 01 is the **glue and the explanation layer**. It reads NVML (and, if
needed, `nvidia-smi`), local CUDA toolkit metadata and the installed PyTorch, normalises
them into one model, and explains the result. When a question goes beyond a point-in-time
environment inspection, it points to the specialised tool: DCGM for health, Nsight for
performance, `gpustat`/`nvidia-smi` for live monitoring.
