# Problem Statement — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. No application code exists yet.
Tracker: [app_tracker.md](app_tracker.md).

## 1. The problem

AI/ML, inference and CUDA developers often need to answer a basic question before they
can work: *"Is this machine's GPU environment usable for my workload, and if not, why?"*

The answer is spread across several independent layers, each with its own tool, its own
vocabulary and its own failure modes:

| Layer | Typical question | Where the answer lives today |
| ----- | ---------------- | ---------------------------- |
| System | Which OS, CPU architecture, RAM and Python? | `platform`, `/proc`, OS tools |
| Hardware | Is there an NVIDIA GPU at all? | PCI enumeration, `lspci`, Device Manager |
| Driver | Is the NVIDIA driver loaded? Which version? | NVML, `nvidia-smi` |
| CUDA | Which CUDA versions are present? | `nvidia-smi` header, `nvcc --version`, `version.json`, framework build metadata |
| Framework | Can PyTorch use the GPU? | `torch.cuda.*` |
| Runtime | How much memory is free right now? How hot is it? | NVML, `nvidia-smi`, `gpustat` |

Common failures this causes:

1. **"PyTorch says CUDA is unavailable" with no explanation.** The cause may be a CPU-only
   PyTorch build, a missing or unloaded driver, a driver too old for the PyTorch build's
   CUDA version, `CUDA_VISIBLE_DEVICES` hiding all devices, or no NVIDIA GPU. Each layer
   has to be checked by hand.
2. **Version confusion.** The "CUDA Version" printed by `nvidia-smi` is the newest CUDA
   version the *driver* supports. It is not the installed CUDA Toolkit, and neither is the
   CUDA version a PyTorch wheel was built with. Developers routinely mix these up
   ([technical_evidence.md](technical_evidence.md) EV-03, EV-04, EV-05).
3. **Misread memory numbers.** NVML "memory utilization" is the percentage of time device
   memory was being read or written, not the share of memory in use (EV-07). PyTorch
   `memory_allocated()` describes only the calling process's allocator, and it returns `0`
   when CUDA has not been initialised in that process (EV-12). A `0` from it doesn't mean
   no memory is in use.
4. **Index mismatches.** CUDA orders devices `FASTEST_FIRST` by default, while NVML and
   `nvidia-smi` order them by PCI bus. On multi-GPU machines "GPU 0" can be a different
   physical device in PyTorch and in `nvidia-smi` (EV-10).
5. **Silent degradation.** On WSL2, on Windows (WDDM) and in restricted containers,
   some metrics are unsupported or need privileges (EV-08, EV-09). Tools often print
   `N/A` or `0` with no reason.
6. **Unreproducible bug reports.** Environment details pasted into issues are incomplete,
   unstructured and may contain private paths and identifiers.

## 2. Who has the problem

| Persona | Need |
| ------- | ---- |
| ML / inference engineer | Quickly confirm a machine can run a job and has enough free VRAM. |
| CUDA developer | Distinguish driver, toolkit and framework CUDA versions. |
| GPU infrastructure developer | Get a structured, scriptable snapshot of a host with explicit unsupported/failed states. |
| Maintainer receiving bug reports | Receive a reproducible, versioned, optionally redacted environment report. |

## 3. Product gap

The existing tools (analysed in [existing_tools_analysis.md](existing_tools_analysis.md))
are each authoritative in their own layer. None of them, alone, does all of the following:

- correlates hardware, driver, CUDA and framework evidence in **one report**;
- explains a failure as **evidence-backed findings** that keep observations, inferences
  and recommendations separate;
- gives a **per-device basic free-VRAM readiness** verdict with an explicit disclaimer;
- tags **every value with its source, method, unit and collection status**, so a missing
  value is never shown as `0`;
- emits a **versioned JSON contract** that can be diffed, archived and redacted.

InferForge Project 01 fills that gap. It does **not** replace `nvidia-smi`, NVML, DCGM,
`gpustat`, PyTorch's CUDA APIs or Nsight. It reads from some of them and points users to
the others when their questions go beyond inspection.

## 4. Problem boundaries

In scope: a single-host, point-in-time, read-only inspection of the local machine
(see [product_specification.md](product_specification.md) §5).

Out of scope for V1: continuous monitoring, fleet management, profiling, benchmarking,
full model-memory estimation, remote inspection and dashboards
(see [product_specification.md](product_specification.md) §6).

## 5. Success definition

Project 01 succeeds when a developer can run one command (`inferforge-gpu inspect`) on
Linux, Windows or WSL2 (and get general information on macOS) and receive, quickly (targets
in [requirements.md](requirements.md) NFR-11, to be measured rather than assumed):

1. a correct inventory of what is and isn't present, with reasons;
2. clearly labelled CUDA versions from each layer;
3. findings that explain the most common GPU-environment failures, with evidence and
   limitations;
4. an optional per-GPU free-VRAM readiness verdict;
5. a versioned JSON report suitable for automation and sharing.

The checklist in [acceptance_criteria.md](acceptance_criteria.md) makes this measurable.
