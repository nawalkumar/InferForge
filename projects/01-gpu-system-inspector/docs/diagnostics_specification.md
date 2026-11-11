# Diagnostics Specification — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. The finding structure is defined in
[data_model_specification.md](data_model_specification.md) §6.3. The engine design is in
[architecture.md](architecture.md) §4.8.

## 1. Principles

1. **Diagnostics consume only the normalised snapshot** (and the readiness result). Rules
   perform no I/O, which makes them deterministic and testable without hardware.
2. **Three kinds of statement, never mixed:**
   - *Observation* (`Evidence.kind = OBSERVATION`): a collected value, quoted with its
     source and JSON pointer. Example: "`torch.version.cuda` is `null`".
   - *Inference* (`Evidence.kind = INFERENCE`): a conclusion drawn from observations, with
     its reasoning. Example: "The PyTorch build has no CUDA support, so it can't use the GPU".
   - *Recommendation* (`Recommendation`): something the user may do. The tool never does it.
3. **No definitive root cause without decisive evidence.** Words such as "the cause is"
   appear only with `confidence = HIGH`, and only when supported by direct observations.
   Otherwise findings say "likely", "possible" or "consistent with", and list alternatives
   in `limitations`.
4. **Absence of evidence isn't evidence of absence.** A `NOT_SUPPORTED`, `PERMISSION_DENIED`
   or `COLLECTION_FAILED` input can never support a `PASS` or a negative conclusion. It
   leads to `UNKNOWN` or reduced confidence.
5. **One finding per (check, device).** Each rule emits at most one finding per
   `device_ref` (`null` for system-wide checks). This keeps output stable and sortable.
6. **Recommendations cite official documentation only** (`reference_url`), and never
   suggest anything destructive or privileged as the default action.

## 2. Status, severity and confidence

| Status | Meaning |
| ------ | ------- |
| `PASS` | Check performed; the healthy condition was positively observed. |
| `WARNING` | A condition that may cause problems, or a partially failed check. |
| `FAIL` | A condition that is observed or strongly inferred to prevent the intended use. |
| `INFO` | Informational context; nothing wrong (e.g. expected absence of a GPU on macOS). |
| `NOT_APPLICABLE` | Preconditions for the check don't hold (e.g. D02 in Mode A). |
| `UNKNOWN` | The check couldn't reach a conclusion because inputs were unavailable. |

| Severity | Use |
| -------- | --- |
| `NONE` | `PASS`, `NOT_APPLICABLE`, and `INFO` that needs no attention |
| `LOW` | Advisory `INFO` worth noticing (D05 non-recommended binding distribution; D08 rows 5–6 index differences); advisory `WARNING`; `UNKNOWN` with little impact |
| `MEDIUM` | `WARNING` likely to affect work; `UNKNOWN` on an important check |
| `HIGH` | `FAIL` that blocks GPU use or the requested workload |

**Valid status/severity combinations (normative; enforced by `DiagnosticFinding` and the
JSON Schema, Stage 3).** These are the only combinations that the decision tables in §3
produce:

| Status | Allowed severities | Rows that use the non-default severity |
| ------ | ------------------ | -------------------------------------- |
| `PASS` | `NONE` | — |
| `NOT_APPLICABLE` | `NONE` | — |
| `INFO` | `NONE`, `LOW` | D05 `pynvml` distribution (`LOW`); D08 rows 5 and 6 (`LOW`) |
| `WARNING` | `LOW`, `MEDIUM` | D03 rows 3–4 (`LOW`); D05 missing binding without NVIDIA evidence (`LOW`) |
| `UNKNOWN` | `LOW`, `MEDIUM` | All current rows use `MEDIUM`; `LOW` is reserved for low-impact checks |
| `FAIL` | `HIGH` | — |

Before Stage 3 the table above listed `INFO` only under `NONE`, which contradicted D05 and
D08 rows 5–6. The decision tables were the more specific statement, so the summary table
was corrected to match them.

| Confidence | Use |
| ---------- | --- |
| `HIGH` | The conclusion follows directly from decisive observations. |
| `MEDIUM` | Consistent with the evidence, but plausible alternatives exist (listed in `limitations`). |
| `LOW` | Weak or indirect evidence. |
| `NOT_APPLICABLE` | Status is `NOT_APPLICABLE`. |

## 3. Rule catalogue

| ID | Title | Scope | Modes | Possible statuses |
| -- | ----- | ----- | ----- | ----------------- |
| D01 | PyTorch can't use CUDA | system | A, B | PASS, FAIL, WARNING, INFO, NOT_APPLICABLE, UNKNOWN |
| D02 | Free GPU memory vs requirement | system (per-device evidence) | B | PASS, FAIL, UNKNOWN, NOT_APPLICABLE |
| D03 | CUDA version landscape | system | A, B | INFO, WARNING, NOT_APPLICABLE |
| D04 | NVIDIA GPU availability | system | A, B | PASS, INFO, WARNING, FAIL |
| D05 | Dependency availability (required, optional, external) | system | A, B | PASS, INFO, WARNING |
| D06 | Runtime metrics availability | per device | A, B | PASS, INFO, WARNING, FAIL, NOT_APPLICABLE |
| D07 | Temperature observation | per device | A, B | INFO, WARNING, NOT_APPLICABLE |
| D08 | CUDA device visibility and ordering | system | A, B | PASS, INFO, WARNING, NOT_APPLICABLE |

Field paths below use JSON Pointer notation relative to the report root.

### D01 — PyTorch can't use CUDA

**Purpose.** When PyTorch reports CUDA unavailable, correlate hardware, driver, CUDA and
framework evidence into a ranked list of likely causes.

**Inputs.** `/frameworks/pytorch/*`, `/gpu/device_count`, `/gpu/devices`,
`/gpu/pci_observations`, `/driver/nvml_version`, `/driver/driver_supported_cuda_version`,
`/driver/cuda_driver_library`, `/cuda_environment/environment/CUDA_VISIBLE_DEVICES`,
`/cuda_environment/environment/PYTORCH_NVML_BASED_CUDA_CHECK`, `/system/platform_profile`.

**Decision table (evaluated top to bottom; first matching row sets the status).**

| # | Condition | Status | Severity | Confidence |
| - | --------- | ------ | -------- | ---------- |
| 1 | `pytorch.status` ∈ {`NOT_INSTALLED`, `SKIPPED`} | NOT_APPLICABLE | NONE | NOT_APPLICABLE |
| 2 | `pytorch.status = COLLECTION_FAILED` with `IMPORT_ERROR` | FAIL | HIGH | HIGH ("PyTorch is installed but can't be imported") |
| 3 | `pytorch.status = COLLECTION_FAILED` (timeout/crash/protocol) | UNKNOWN | MEDIUM | NOT_APPLICABLE |
| 4 | `rocm_build = true` | NOT_APPLICABLE | NONE | NOT_APPLICABLE (ROCm out of V1 scope; stated in explanation) |
| 5 | `cuda_available = true` | PASS | NONE | HIGH |
| 6 | `cuda_available = false` and **NVIDIA GPU observed** (NVML `device_count ≥ 1`, or fallback inventory non-empty, or a PCI observation with `is_nvidia`) | FAIL | HIGH | from the strongest cause below, or LOW if none matched |
| 7 | `cuda_available = false` and NVIDIA GPU **observed absent** (macOS; or NVML `device_count = 0`; or PCI scan `COLLECTED` with no NVIDIA device and NVML `NOT_DETECTED`) | INFO | NONE | HIGH ("expected: no NVIDIA GPU") |
| 8 | `cuda_available = false`, GPU presence undeterminable | WARNING | MEDIUM | MEDIUM or lower |
| 9 | `cuda_available` not `COLLECTED` | UNKNOWN | MEDIUM | NOT_APPLICABLE |

**Cause analysis** (rows 6 and 8). Every matching cause is added as evidence, strongest
first:

| Cause | Observation(s) | Inference | Confidence |
| ----- | -------------- | --------- | ---------- |
| D01.1 CPU-only build | `cuda_build = false` | The installed PyTorch build has no CUDA support. | HIGH |
| D01.2 Devices hidden | `CUDA_VISIBLE_DEVICES` is `""`, or its first entry isn't a valid index/UUID/MIG ID of an observed device (EV-10) | CUDA applications in this environment see no devices. | HIGH |
| D01.3 Driver not bound | PCI observation `is_nvidia` with `kernel_driver` ≠ `nvidia` (e.g. `nouveau`, none) and NVML `NOT_DETECTED` | The NVIDIA kernel driver isn't controlling the GPU. | HIGH |
| D01.4 Driver user-mode library missing | NVML `NOT_DETECTED` (`LIBRARY_NOT_FOUND`) or `cuda_driver_library = NOT_DETECTED` while a GPU is observed | The CUDA driver library isn't loadable. | MEDIUM |
| D01.5 Driver older than build | `major(driver_supported_cuda_version) < major(build_cuda_version)` | The driver likely doesn't support the CUDA major version PyTorch was built with. | MEDIUM (forward compatibility may change this, EV-05) |
| D01.6 Permission | NVML `PERMISSION_DENIED` | Device access may be restricted for this user/container. | MEDIUM |
| D01.7 PyTorch's own message | `cuda_init_warnings` non-empty | Quoted verbatim (redactable) as an observation; not interpreted. | n/a |
| D01.8 NVML-based check active | `PYTORCH_NVML_BASED_CUDA_CHECK` set (EV-13) | `is_available()` used NVML instead of the CUDA runtime. | n/a (context) |

If no cause D01.1–D01.6 matches: the explanation says the cause couldn't be determined from the
collected evidence, confidence is `LOW`, and the recommendation is to run
`python -c "import torch; torch.cuda.init()"` to see PyTorch's full error.

**Recommendations** (only for matched causes): D01.1 install a CUDA-enabled build using
the official selector (`https://pytorch.org/get-started/locally/`); D01.2 unset or correct
`CUDA_VISIBLE_DEVICES`; D01.3/D01.4 install or repair the NVIDIA driver (on WSL2: on the
Windows host, never inside WSL2, EV-09); D01.5 update the driver or install a PyTorch build
for an older CUDA major (compatibility guide link, EV-05); D01.6 check device permissions
or the container GPU configuration.

**Limitations (always).** "Causes are inferred from the collected snapshot; other causes
(for example corrupted installations or unusual library paths) are possible."
Plus, for D01.5: "Same-major version differences are not reported as causes because of CUDA
minor version compatibility; forward-compatibility packages are not detected."

### D02 — Free GPU memory vs requirement

**Applicability.** Mode B only; in Mode A the status is `NOT_APPLICABLE` with evidence
"no `--required-vram` supplied".

**Mapping from readiness** (DEC-19):

| `readiness.overall_status` | Status | Severity | Confidence |
| -------------------------- | ------ | -------- | ---------- |
| `READY` | PASS | NONE | HIGH (direct comparison; see limitations) |
| `NOT_READY` | FAIL | HIGH | HIGH if every device's free memory was `COLLECTED`; MEDIUM when the verdict comes from no-device reasons |
| `UNKNOWN` | UNKNOWN | MEDIUM | NOT_APPLICABLE |

**Evidence.** For each device: observed free bytes (with source/method), total bytes,
required bytes, margin, device status and reason, and MIG mode if enabled.

**Recommendations (FAIL).** "Free GPU memory by closing other GPU workloads (processes
observed using memory are listed in the runtime section, where visible)"; "Use a GPU with
more free memory, if available"; "Reduce the workload's memory requirement". All are
generic, and none claims a specific amount will be sufficient.

**Limitations (always).** The fixed readiness disclaimer; "single snapshot at
`<sampled_at>`"; source precision (1 MiB for the `nvidia-smi` fallback); MIG limitation
when applicable.

### D03 — CUDA version landscape

**Purpose.** Clarify which CUDA versions exist in which layer and flag real
inconsistencies (EV-03, EV-04, EV-05).

**Inputs.** `driver_supported_cuda_version`, `cuda_environment.toolkits`,
`active_nvcc_version`, `CUDA_HOME`, `build_cuda_version`.

| # | Condition | Status | Severity | Confidence |
| - | --------- | ------ | -------- | ---------- |
| 1 | None of the three versions `COLLECTED` | NOT_APPLICABLE | NONE | NOT_APPLICABLE |
| 2 | `major(build_cuda) > major(driver_supported)` | WARNING | MEDIUM | MEDIUM |
| 3 | `major(toolkit or active nvcc) > major(driver_supported)` | WARNING | LOW | MEDIUM |
| 4 | Several toolkits with different versions, or the `CUDA_HOME` toolkit version ≠ active `nvcc` version | WARNING | LOW | HIGH (direct observation of a mismatch) |
| 5 | Otherwise | INFO | NONE | HIGH |

The highest-ranked matching row sets status and severity. Every matching row adds
evidence.

**Explanation (always included).** A three-line legend:
1. "Driver supports CUDA up to X.Y" — the newest CUDA version the installed driver
   supports. **Not** the installed toolkit (EV-03).
2. "CUDA Toolkit A.B.C" — a local development toolkit (`nvcc`). Needed to compile CUDA
   code. Not needed to run PyTorch binary packages `[ASM]` (verify: HW-06).
3. "PyTorch built with CUDA P.Q" — the CUDA version PyTorch was built with.

**Toolkit absent while the driver is present** (fixture FX08): an INFO note, never a
WARNING or FAIL, saying that a toolkit is only needed for compiling CUDA code.

**Limitations.** Only major-version comparisons are used (minor version compatibility,
EV-05). Forward-compatibility (`cuda-compat`) installations aren't detected. Toolkit
detection relies on `version.json`/`nvcc` `[ASM]` (EV-20).

### D04 — NVIDIA GPU availability

| # | Condition | Status | Severity | Confidence |
| - | --------- | ------ | -------- | ---------- |
| 1 | NVML `COLLECTED` and `device_count ≥ 1` | PASS | NONE | HIGH |
| 2 | Fallback (`nvidia-smi`) inventory non-empty, NVML unavailable | PASS | NONE | MEDIUM (limitation: NVML-based fields unavailable, see D05) |
| 3 | `platform_profile = MACOS` | INFO | NONE | HIGH ("NVIDIA CUDA GPUs are not supported on macOS; general information was collected") |
| 4 | PCI shows an NVIDIA device, but neither NVML nor `nvidia-smi` provided an inventory | FAIL | HIGH | HIGH ("NVIDIA GPU present on PCI, but no NVIDIA driver interface is accessible"; evidence includes `kernel_driver`) |
| 5 | NVML `PERMISSION_DENIED` | WARNING | MEDIUM | MEDIUM |
| 6 | NVML `COLLECTED` with `device_count = 0` | WARNING | MEDIUM | MEDIUM ("driver loaded but reports no devices") |
| 7 | PCI scan `COLLECTED` with no NVIDIA device, NVML/`nvidia-smi` `NOT_DETECTED` | INFO | NONE | HIGH ("no NVIDIA GPU found") |
| 8 | PCI scan not supported (WSL2/Windows) or failed, NVML/`nvidia-smi` `NOT_DETECTED`/`NOT_INSTALLED` | INFO | NONE | MEDIUM ("no NVIDIA GPU is accessible"; limitation: hardware absence can't be confirmed) |
| 9 | Anything else (for example NVML `COLLECTION_FAILED`) | UNKNOWN | MEDIUM | NOT_APPLICABLE |

The explanation always says which NVIDIA-specific sections were unavailable, and that the
rest of the report is still valid. Recommendations for rows 4 and 8 point to driver
installation (on WSL2: the Windows host driver, EV-09).

### D05 — Dependency availability

One system-wide finding with one evidence entry per dependency. Dependencies fall into the
classes defined in [product_specification.md](product_specification.md) §1.1: the
**required** Python dependency `nvidia-ml-py`, the **optional** Python dependency PyTorch,
and **external** components (`nvidia-smi`). The NVIDIA driver and NVML library are
external too, but their availability is reported by D04.

| Dependency | Class | Condition | Contribution |
| ---------- | ----- | --------- | ------------ |
| `nvidia-ml-py` (`pynvml`) | required | `NOT_INSTALLED` and NVIDIA evidence exists (`nvidia-smi` found, or a PCI NVIDIA device) | WARNING / MEDIUM: "The required NVML bindings are missing, so the installation is incomplete. NVML-based fields are unavailable." Recommendation: install `nvidia-ml-py` into this environment (for example `pip install nvidia-ml-py`), or reinstall GPU System Inspector. |
| `nvidia-ml-py` | required | `NOT_INSTALLED`, no NVIDIA evidence | WARNING / LOW: same explanation. This report is unaffected because no NVIDIA GPU is accessible. |
| `nvidia-ml-py` | required | module `pynvml` provided by a distribution other than `nvidia-ml-py` (e.g. `pynvml`) | INFO / LOW: note the documented recommendation to depend on `nvidia-ml-py` (EV-17) |
| PyTorch | optional | `NOT_INSTALLED` | INFO: "PyTorch is not installed in this Python environment; framework checks were skipped. PyTorch in other environments is not inspected." Recommendation: the official install selector, or run the inspector inside the environment that has PyTorch. The tool never installs it. |
| PyTorch | optional | `SKIPPED` | INFO: "skipped by `--no-pytorch`" |
| `nvidia-smi` | external | `NOT_DETECTED` while NVML works | INFO: "fallback source unavailable; no impact on this report" |
| `nvidia-smi` | external | `NOT_DETECTED` and NVML also unavailable | No contribution (GPU availability is D04's responsibility) |

Status = worst contribution (`WARNING` > `INFO` > `PASS`). If every dependency is present →
`PASS`. Confidence `HIGH`: these are direct observations. Severity = the severity of the
worst contribution. `INFO` contributions without a stated severity are `NONE`.

### D06 — Runtime metrics availability (per device)

**Inputs.** All `Observed` fields of `/runtime/<i>` except `sampled_at`.

| # | Condition (ignoring `SKIPPED` fields, which are listed as evidence) | Status | Severity | Confidence |
| - | ------------------------------------------------------------------- | ------ | -------- | ---------- |
| 1 | No inventoried devices | NOT_APPLICABLE (single finding, `device_ref = null`) | NONE | NOT_APPLICABLE |
| 2 | Any field `COLLECTION_FAILED` with `NVML_ERROR_GPU_IS_LOST` | FAIL | HIGH | HIGH ("NVML reports the GPU as lost") |
| 3 | Any field `PERMISSION_DENIED` or `COLLECTION_FAILED` | WARNING | MEDIUM | HIGH (direct observation) |
| 4 | Any field `NOT_SUPPORTED` (only) | INFO | NONE | HIGH, with the documented platform reason when one applies (WSL2: EV-09; WDDM process memory: EV-08; MIG aggregate memory: EV-06) |
| 5 | All fields `COLLECTED` | PASS | NONE | HIGH |

**Recommendations.** Row 2: check system logs and consider DCGM diagnostics where
available (EV-15). Row 3: rerun with an account or container configuration that has GPU
access, if those values are needed. Never recommend root as the default.

### D07 — Temperature observation (per device)

| # | Condition | Status | Severity | Confidence |
| - | --------- | ------ | -------- | ---------- |
| 1 | `temperature_gpu_celsius` not `COLLECTED` | NOT_APPLICABLE (availability is reported by D06) | NONE | NOT_APPLICABLE |
| 2 | Slowdown threshold `COLLECTED` and `temperature ≥ threshold` | WARNING | MEDIUM | MEDIUM |
| 3 | Otherwise | INFO | NONE | HIGH |

**Explanation.** Row 3: "Measured X °C at `<sampled_at>`" (plus "device-reported slowdown
threshold Y °C" when available). Row 2: "Measured X °C, at or above the device-reported
slowdown threshold of Y °C at the moment of sampling."

**Limitations (always).** "A single temperature sample cannot establish whether the GPU is
thermally throttling or how temperature changes under load (C15). For sustained
observation use `nvidia-smi` monitoring or DCGM." The words "throttling is occurring" or
"overheating" must never appear as conclusions. Row 2's recommendation: check cooling and
observe temperature over time.

### D08 — CUDA device visibility and ordering

| # | Condition | Status | Severity | Confidence |
| - | --------- | ------ | -------- | ---------- |
| 1 | No inventoried devices and `CUDA_VISIBLE_DEVICES` unset | NOT_APPLICABLE | NONE | NOT_APPLICABLE |
| 2 | `CUDA_VISIBLE_DEVICES = ""` (EV-10: no devices visible) | WARNING | MEDIUM | HIGH |
| 3 | `CUDA_VISIBLE_DEVICES` contains an entry that matches no observed index/UUID prefix/MIG ID (EV-10: later entries are dropped) | WARNING | MEDIUM | MEDIUM (MIG IDs and UUID prefixes are matched best-effort) |
| 4 | `CUDA_VISIBLE_DEVICES` is a valid subset or reordering | INFO | NONE | HIGH ("CUDA applications see N device(s), renumbered from 0") |
| 5 | Correlation shows a PyTorch index ≠ NVML index for the same device | INFO | LOW | HIGH ("PyTorch cuda:i is NVML device j") |
| 6 | ≥ 2 devices with different names and `CUDA_DEVICE_ORDER` ≠ `PCI_BUS_ID` (and no correlation data) | INFO | LOW | MEDIUM ("indices may differ between PyTorch and `nvidia-smi`"; recommend `CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| 7 | Otherwise | PASS | NONE | HIGH |

Rows 2 and 3 take precedence; rows 4–6 may contribute evidence together. In containers,
`NVIDIA_VISIBLE_DEVICES` is reported as context evidence only.

## 4. Engine behaviour

- Rules run in ID order. Each rule is guarded: an exception yields one `UNKNOWN` finding
  (`title = "Check could not be evaluated"`) and a `CollectionError`
  (`collector = "diagnostics.Dxx"`).
- Findings are sorted by `(check_id, device_ref ?? "")` in JSON. The text renderer orders
  by status rank (FAIL, WARNING, UNKNOWN, INFO, PASS, NOT_APPLICABLE) and then by the same key.
- Every rule always emits at least one finding (possibly `NOT_APPLICABLE`), so the
  absence of a check in a report is itself a defect.

## 5. Wording rules (enforced by tests)

| Rule | Test |
| ---- | ---- |
| Every non-`NOT_APPLICABLE` finding has ≥ 1 evidence item | AC10 |
| `WARNING`/`FAIL` have ≥ 1 recommendation and ≥ 1 limitation | AC10 |
| "cause is" / "caused by" only with `confidence = HIGH` | AC28 |
| D07 never contains "is throttling", "throttled" or "overheating" outside `limitations` | AC34 |
| D02 and readiness never contain "will fit", "fits" or "guarantee" except in the negated disclaimer | AC23 |
| Recommendations never contain install commands for PyTorch (only the selector link). No recommendation is ever executed by the tool | AC32 |
| `reference_url` values are on an allowlist of official documentation domains (`docs.nvidia.com`, `developer.nvidia.com`, `docs.pytorch.org`, `pytorch.org`, `docs.python.org`, `pypi.org`) | AC28 |

## 6. Expected findings per fixture scenario

Fixtures are defined in [testing_strategy.md](testing_strategy.md) §4. Dashes mean the
rule's default outcome for that scenario (usually `PASS` or `NOT_APPLICABLE`), which is
also asserted.

| Fixture | D01 | D02 | D03 | D04 | D05 | D06 | D07 | D08 |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| FX01 working CUDA | PASS | N/A | INFO | PASS | PASS | PASS | INFO | PASS |
| FX02a CPU-only build | FAIL (D01.1, HIGH) | N/A | INFO | PASS | PASS | PASS | INFO | PASS |
| FX02b driver older than build | FAIL (D01.5, MEDIUM) | N/A | WARNING | PASS | PASS | PASS | INFO | PASS |
| FX02c `CUDA_VISIBLE_DEVICES=""` | FAIL (D01.2, HIGH) | N/A | INFO | PASS | PASS | PASS | INFO | WARNING |
| FX02d unexplained + warning | FAIL (LOW) | N/A | INFO | PASS | PASS | PASS | INFO | PASS |
| FX03 no NVIDIA GPU (Linux) | INFO | N/A | N/A | INFO (row 7) | PASS | N/A | N/A | N/A |
| FX04a binding missing, `nvidia-smi` present | PASS | N/A | INFO | PASS (row 2) | WARNING | INFO | INFO | PASS |
| FX04b NVML library missing (WSL2) | N/A | N/A | N/A | INFO (row 8) | INFO | N/A | N/A | N/A |
| FX05 PyTorch missing | N/A | N/A | INFO | PASS | INFO | PASS | INFO | PASS |
| FX06 multi-GPU, order differs | PASS | N/A | INFO | PASS | PASS | PASS ×2 | INFO ×2 | INFO (row 5) |
| FX07 insufficient memory (B) | PASS | FAIL | INFO | PASS | PASS | PASS | INFO | PASS |
| FX08 no toolkit | PASS | N/A | INFO (toolkit note) | PASS | PASS | PASS | INFO | PASS |
| FX09 metrics inaccessible | PASS | N/A | INFO | PASS | PASS | WARNING | N/A | PASS |
| FX10 partial collector failure | PASS | N/A | INFO | PASS | PASS | FAIL (nvml:1, lost) / PASS (nvml:0) | INFO (nvml:0) / N/A (nvml:1) | PASS |
| FX11 WSL2 with NVIDIA | PASS | N/A | INFO | PASS | PASS | INFO | INFO | PASS |
| FX12 Windows WDDM | PASS | N/A | INFO | PASS | PASS | INFO | INFO | PASS |
| FX13 macOS | INFO | N/A | N/A | INFO (row 3) | PASS | N/A | N/A | N/A |
| FX14 MIG enabled (B) | PASS | UNKNOWN | INFO | PASS | PASS | INFO | INFO | PASS |
| FX15 PyTorch import fails | FAIL (row 2) | N/A | INFO | PASS | PASS | PASS | INFO | PASS |
| FX16 PyTorch probe timeout | UNKNOWN | N/A | INFO | PASS | PASS | PASS | INFO | PASS |
| FX17 PCI NVIDIA + nouveau | N/A | N/A | N/A | FAIL (row 4) | INFO | N/A | N/A | N/A |
| FX18 hot GPU | PASS | N/A | INFO | PASS | PASS | PASS | WARNING | PASS |
| FX19 free == required (B) | PASS | PASS | INFO | PASS | PASS | PASS | INFO | PASS |
| FX20 `pynvml` distribution | PASS | N/A | INFO | PASS | INFO | PASS | INFO | PASS |

"N/A" = `NOT_APPLICABLE`. Rows marked (B) run in Mode B. D02 for the additional Mode B
runs (FX01, FX03, FX04b, FX06, FX09, FX13) is listed in
[testing_strategy.md](testing_strategy.md) §4.1. Per-device checks show one value per
device where they differ.
The fixture inputs that produce these values (for example whether PyTorch is installed)
are pinned in [testing_strategy.md](testing_strategy.md) §4.
