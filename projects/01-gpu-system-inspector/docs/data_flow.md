# Data Flow — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. Components are defined in
[architecture.md](architecture.md); models in
[data_model_specification.md](data_model_specification.md).

## 1. End-to-end flow

```
argv
  │
  ▼
CLI ── parse + validate ──► usage error? ──► stderr message, exit 2
  │
  │ InspectionOptions
  ▼
Inspection Core   (implemented, IMP-K08/DEC-44, IMP-K03/K04/DEC-46, IMP-K05/DEC-47,
  │                 IMP-K06/DEC-48, IMP-K07/DEC-49: steps ①–⑦, then report assembly;
  │                 readiness, diagnostics and redaction are placeholders — §1.2)
  │  collectors get adapters + clock in their constructors (no CollectionContext)
  │
  ├─①  System collector ────────────────► SystemInfo
  ├─②  PCI presence: PciCollector, run `pci` (only if linux_environment is NATIVE_LINUX;
  │     otherwise NOT_SUPPORTED, no sysfs access) ► GPUInventory.pci_observations
  │     (merged by the GPU hardware collector, IMP-K02; architecture §4.4, §4.11)
  ├─③  NVML session open (run `nvml`) ─┐  (ImportError / LibraryNotFound / DriverNotLoaded /
  │                            │   NoPermission … recorded; session = unavailable)
  │      ├─ Driver collector ──► DriverInfo (driver, NVML, driver-supported CUDA, libcuda)
  │      ├─ GPU hardware coll. ► GPUInventory (NVML devices, or the nvidia-smi fallback
  │      │                       query when NVML can't count; PCI section from ②)
  │      └─ Runtime collector ─► RuntimeMetrics[] (memory snapshot FIRST, then util,
  │                            │   temperature, power, processes)
  │   NVML session close ◄─────┘  (always, in `finally`)
  ├─④  nvidia-smi: locate + version (always, run `nvidia_smi`); the query only if ③ produced
  │     no device inventory or driver version, and at most once (the adapter caches it)
  │                        ──────► DriverInfo.nvidia_smi_*, GPUInventory/RuntimeMetrics (fallback)
  ├─⑤  CUDA environment collector (run `cuda_environment`, DEC-47):
  │     PlatformAdapter: allowlisted variables (+ CUDA_PATH_V* names on Windows)
  │       └─ CUDA_HOME / CUDA_PATH / CUDA_PATH_V* values ─► ToolkitLocator (hints)
  │     ToolkitLocator: CommandRunner.which("nvcc") → realpath → <root> (NVCC_ON_PATH)
  │                     Linux/WSL2: list /usr/local once, cuda[-X.Y] (DEFAULT_LOCATION)
  │                     per root: version.json (≤ 64 KiB) else <root>/bin/nvcc --version
  │                     active nvcc --version, once (shared with its toolkit)
  │       └─────────────────────────────► CUDAEnvironmentInfo (+ CUDA.TOOLKIT on macOS)
  ├─⑥  PyTorch (run `pytorch`, DEC-48): metadata check ─► NOT_INSTALLED? stop
  │              └─ probe (subprocess, timeout; §5) ► FrameworkCollector ► FrameworkInfo.pytorch
  │                 (build, runtime answer and devices; no CUDA code run; no correlation)
  ├─⑦  Correlation (run `correlation`, DEC-09, DEC-49): observations only, nothing probed
  │       GPUInventory (③, NVML devices) ─────┐
  │                                            ├─ UUID, then PCI bus ID; never index
  │       FrameworkInfo.pytorch.devices (⑥) ──┘     ─► FrameworkInfo.pytorch with every
  │                                                    device's correlated_device_ref
  │
  ▼
InspectionSnapshot  (normalised models + collection_errors + collector_runs)
  │
  ├─⑧  Readiness engine (only if --required-vram) ► ReadinessResult
  ├─⑨  Diagnostics engine (D01–D08; D02 uses ⑧) ──► DiagnosticFinding[]
  ▼
Report assembly ──► InspectionReport (+ unsupported_capabilities, limitations, metadata)
  │
  ├─⑩  Redaction (only if --redact) ──► redacted InspectionReport
  ▼
Renderers
  │
  ├── --json PATH ─► JSON ─► atomic write (tmp + rename) ─► failure? remember exit 4
  ├── --json -    ─► JSON ─► stdout (no text report)
  └── otherwise   ─► text ─► stdout  (also printed when --json PATH is used)
  │
  ▼
Exit code (0 / 1 / 3 per readiness; 0 in Mode A; 4 JSON write failure; 5 internal error;
           130 interrupt)
```

### 1.1 Step ① in detail (implemented in Stage 5)

```
SystemCollector(platform, clock).collect()
  t0 = clock.monotonic_ms()
  platform.system_info()                       # OsPlatformAdapter over StdlibHostApi
    linux_environment ← (Linux only) classify(/proc/sys/kernel/osrelease, /proc/version)
                        per data model §4.1 → NATIVE_LINUX | WSL1 | WSL2 | UNKNOWN,
                        rationale in note; non-Linux → NOT_APPLICABLE; detected once
    profile  ← platform.system(); Linux is WSL2 only if linux_environment is WSL2
    each field read independently through HostApi:
      platform.* / sys / os.cpu_count()         → Source PLATFORM
      /proc/cpuinfo, /proc/meminfo (read once), os-release, registry, GlobalMemoryStatusEx,
      sysctlbyname (macOS, read-only)           → Source OS
      failure → Observed status (NOT_DETECTED / PERMISSION_DENIED / COLLECTION_FAILED /
                PARSE_ERROR / NOT_SUPPORTED); a bug → exception → collector guard
  run = PARTIAL if any field COLLECTION_FAILED / PERMISSION_DENIED / UNKNOWN else COMPLETED
  unsupported = UnsupportedCapability per NOT_SUPPORTED field (SYSTEM.*)
  → CollectorOutcome(section=SystemInfo, run, unsupported)

collect_environment(platform)                  # feeds CUDAEnvironmentInfo.environment
  names checked against the allowlist first, then platform.getenv(name) for each:
  set → COLLECTED (value as-is), unset → NOT_DETECTED / NOT_SET
```

No subprocess is started, nothing is written, and no environment variable outside the
allowlist is read. On Windows the CUDA environment collector also reads the set
`CUDA_PATH_V*` variables: `PlatformAdapter.versioned_cuda_path_names()` filters the names by
`CUDA_PATH_VERSIONED_REGEX` before any value is read (KI-28, DEC-47).

### 1.2 What runs today (DEC-44 to DEC-49: every collector of `COLLECTOR_ORDER`)

`api.inspect()` and `Inspector(adapters).run()` execute: ① `system`; fail-closed environment
resolution; the PCI scanner built from that environment; ② `pci`; ③ `nvml` (session open),
`driver`, `gpu_hardware` and `runtime`, then the session is closed; ④ `nvidia_smi`; ⑤
`cuda_environment` (files and `nvcc` only, no NVML); ⑥ `pytorch` (the probe child, after
everything else, so nothing before it touches CUDA); ⑦ `correlation` (from the GPU and
PyTorch sections already collected: no NVML session, no probe, no process); report
then readiness (⑧, Mode B only), diagnostics D01–D08 (⑨) and report assembly with schema
validation (DEC-50). The CLI then redacts (⑩, `--redact`, before anything leaves the
process), writes the JSON atomically and renders the text report. A toolkit found in
⑤ is an installation fact, and PyTorch's `cuda_available` in ⑥ is PyTorch's answer: neither
is proof that CUDA code runs. In
production the NVML, `nvidia-smi` and command adapters are real (DEC-45): `nvml` initialises
NVML once and the three NVML-backed collectors share the session; `nvidia-smi` is located
and its header read once (④), and its query runs at most once, only if NVML can't count
devices or report the driver version and the executable is found (architecture §4.4). Only
the CUDA Toolkit locator (DEC-47) and the PyTorch probe (DEC-48) are real too.

## 2. Why this order

| Step | Ordering constraint | Reason |
| ---- | ------------------- | ------ |
| ① first | none | Cheap. Platform facts (OS, WSL2) are needed by later adapters (for example the `nvidia-smi` search path). |
| ② before ③ | independent of driver | Gives driver-independent evidence even when NVML fails. |
| ③ one NVML session | init once, shut down once | Avoids repeated `nvmlInit` cost; guarantees `nvmlShutdown` on every path. |
| Runtime memory before the PyTorch probe | **required** | The probe may create a CUDA context that consumes device memory (C10/C11 `[ASM]`). Readiness must read the pre-probe value (DEC-12). |
| ④ after ③ | fallback is conditional | `nvidia-smi` is only queried for data when NVML couldn't provide it (DEC-04). |
| ⑥ late | slowest step | Lets `--no-pytorch` or a probe timeout affect only the framework section. |
| ⑦ after ③ and ⑥ | needs both inventories | Correlation by UUID, then PCI bus ID (DEC-09, DEC-49); it reads the two sections and probes neither source again. |
| ⑧ before ⑨ | D02 reads readiness | Single source of truth (DEC-19). |
| ⑩ after assembly | must cover evidence text | Redaction scans the whole report, including finding explanations and evidence statements (FR-19). |

## 3. Collector guard and error flow

Implemented as `core/guard.py` `run_guarded(collector, clock, *, report_path)` (DEC-44):

```
t0 = clock.monotonic_ms()
try:
    outcome = collector.collect()          # CollectorOutcome: section, run (COMPLETED or
    check outcome                          #   PARTIAL, rules below), errors, unsupported
    return outcome                         # wrong type / other collector / FAILED → raise
except Exception as exc:                   # unexpected only; KeyboardInterrupt propagates
    detail = ErrorDetail(code="UNHANDLED_EXCEPTION", message=f"{type(exc).__name__}: {exc}")
    err = CollectionError(collector=collector.name, device_ref=None,
                          status=COLLECTION_FAILED, error_code=detail.code,
                          message=detail.message, report_path=report_path)
    section = collector.unavailable(detail)   # every field: COLLECTION_FAILED
    run = CollectorRun(collector.name, FAILED, duration_ms=clock.monotonic_ms() - t0)
    return CollectorOutcome(section, run, errors=(err,))
```

The core concatenates every outcome's errors and unsupported capabilities into the
snapshot and keeps the runs in execution order; report assembly rejects a collector that
ran twice or a capability record with two owners.

- **Expected** failures (missing library, unsupported query, permission, timeout, malformed
  output) are converted to statuses *inside* adapters, following the outcome contract in
  architecture §4.5.2. They produce field-level `Observed` failures and, where they
  affect a whole capability, an entry in `unsupported_capabilities` or `collection_errors`
  (data model §3.4). They never raise.
- **Run status** (implemented by `SystemCollector` and `PciCollector`). `NOT_DETECTED` and
  `NOT_SUPPORTED` describe the host, so on their own they leave the run `COMPLETED`. A run
  is `PARTIAL` for collection problems, which come in three kinds:
  - *Field-level* failures (`COLLECTION_FAILED`, `PERMISSION_DENIED` or `UNKNOWN` inside a
    kept value, e.g. an unreadable PCI driver link) stay in their `Observed` value and add
    no `CollectionError`.
  - *Omitted devices* (a PCI entry listed but not observable, whatever its status) add one
    host-wide `CollectionError` each.
  - *Whole-capability failures* (e.g. PCI sysfs root `PERMISSION_DENIED` or
    `COLLECTION_FAILED`) add one `CollectionError`.

  `FAILED` comes only from the guard. Mapping for the PCI collector: architecture §4.4
  (DEC-43).
- **Fallback.** `unavailable(...)` takes an `ErrorDetail`, because the fallback section's
  `Observed.error` can only hold one (`Collector` contract, architecture §4.2, DEC-44). The
  GPU hardware fallback keeps the PCI section, which another collector produced.
- **Unexpected** exceptions are caught by the guard. Other collectors still run (FR-13).
  `AdapterUsageError` (a caller bug) is caught the same way but is never expected in a
  correct build; tests treat it as a failure.
- **Per-device isolation:** the hardware and runtime collectors loop over devices with a
  per-device guard, so device *n* failing leaves devices ≠ *n* intact (AC54).
- **Diagnostics** run each rule under the same kind of guard (architecture §4.8).

## 4. NVML session lifecycle

| Event | Handling |
| ----- | -------- |
| `import pynvml` raises `ImportError` (required dependency missing: broken installation) | `nvml_binding.status = NOT_INSTALLED`; session unavailable; `nvidia-smi` fallback attempted; D05 `WARNING` explains. |
| `nvmlInit` raises `NVMLError_LibraryNotFound` | `nvml_library.status = NOT_DETECTED` (driver user-mode library absent). |
| `nvmlInit` raises `NVMLError_DriverNotLoaded` | `nvml_library.status = NOT_DETECTED`, error code recorded (kernel driver not loaded). |
| `nvmlInit` raises `NVMLError_NoPermission` | `PERMISSION_DENIED`. |
| Any other `NVMLError` on init | `COLLECTION_FAILED` with NVML error name. |
| Init failed (any row above) | `open()` still returns a session. Its `init_result` is the failure, every query returns that failure with the field's unit, and no `nvmlShutdown` is issued. |
| Session open | Opened once by the core; all NVML-backed collectors share it. A second concurrent `open()` is a usage error (architecture §4.5.3). |
| Device handle | `session.device(i)` returns `Observed[NvmlDevice]`. A handle failure (e.g. `GPU_IS_LOST`) affects only device *i*. The GPU hardware and runtime collectors each acquire the handle; a GPU lost between them is enumerated but gets an unavailable runtime record and one error (KI-34, DEC-46). Handles are invalid after close. |
| Any outcome | `close()` in `finally` (context manager): `nvmlShutdown()` once, only after a successful init; repeated `close()` is a no-op; shutdown errors are logged at DEBUG and ignored. Exceptions from the `with` body propagate. |

The contract is pinned by `tests/unit/test_nvml_lifecycle.py` against `FakeNvml` and by
`tests/unit/test_nvml_adapter.py` against the production `PynvmlAdapter` over a stub
`pynvml` (IMP-A01). On the WSL2 development machine the production adapter has run for real
(host tests, demo): `nvmlInit` raised `NVMLError_LibraryNotFound` (no NVIDIA driver
library), so no session was initialised and no shutdown issued. A session on NVIDIA hardware
has never been opened (HW-01 `BLOCKED`).

## 5. PyTorch probe flow

```
importlib.metadata.version("torch")
   ├─ PackageNotFoundError ─► pytorch.status = NOT_INSTALLED   (no subprocess)
   └─ version string
        │  --no-pytorch? ─► pytorch.status = SKIPPED
        ▼
   run [sys.executable, "-m", "gpu_system_inspector.adapters._torch_probe_main"]
        with timeout = pytorch_timeout_s, env = os.environ (unchanged), shell = False
        ├─ TimeoutExpired ──────► COLLECTION_FAILED / TIMEOUT
        ├─ returncode != 0 ────► COLLECTION_FAILED / PROBE_EXIT_<n> (+ redacted stderr tail)
        ├─ stdout not valid JSON ► COLLECTION_FAILED / PROBE_PROTOCOL_ERROR
        └─ JSON {probe_protocol: 2, import_ok, import_error, torch_version, cuda_build,
                 build_cuda_version, hip_version, cudnn_version, cuda_available,
                 cuda_init_warnings[], device_count, devices[], device_error}
              ├─ import_ok = false ─► COLLECTION_FAILED / IMPORT_ERROR (installed, not importable)
              ├─ device_error ─► PyTorchInfo COLLECTED, device_count COLLECTION_FAILED, run PARTIAL
              └─ normalise into PyTorchInfo (each field Observed, source = PYTORCH)
                   → FrameworkCollector → InspectionSnapshot → assemble_report → schema
```

The probe never allocates tensors, never calls `mem_get_info`, and never reads allocator
statistics (DEC-08). The probe protocol version (`probe_protocol`) lets the parent reject
mismatched output.

**Protocol version 2 (DEC-48; version 1 was Stage 4's, without `device_error`;
`adapters/probe_protocol.py`; the child `adapters/_torch_probe_main.py` runs the probe
function in `adapters/torch_probe_payload.py`).** The parent decodes the
child's output with `decode_probe_output` into a `TorchProbeReport`. Every key must be
present, no other key is allowed, and JSON booleans never pass for numbers. Anything else
is `PROBE_PROTOCOL_ERROR`.

| Key | Type | Key (per device) | Type |
| --- | ---- | ---------------- | ---- |
| `probe_protocol` | integer `1` | `index` | int |
| `import_ok` | bool | `name` | str \| null |
| `import_error` | str (when `import_ok` is false), else null | `capability` | str \| null (`"major.minor"`) |
| `torch_version` | str | `total_memory` | int \| null (bytes) |
| `cuda_build` | bool | `multiprocessor_count` | int \| null |
| `build_cuda_version`, `hip_version`, `cudnn_version` | str \| null (the probe stringifies `cudnn.version()`) | `uuid` | str \| null |
| `cuda_available` | bool | `pci_bus_id` | str \| null |
| `cuda_init_warnings` | list of str | | |
| `device_count` | int \| null | | |
| `devices` | list of device objects | | |
| `device_error` | str (a failure reading device properties; `devices` is then empty), else null | | |

`null` for a device attribute means this PyTorch version doesn't expose it; the framework
collector maps it to `NOT_SUPPORTED`. Launch failures (timeout, missing interpreter) pass
through the decoder unchanged.

## 6. Readiness flow (Mode B)

```
required_bytes = round(required_vram_gib * 2**30)
for device in inventory.devices (NVML index order):
    m = runtime[device].memory_free_bytes        # Observed[int], from NVML or nvidia-smi
    decide status + reason (architecture §4.9) → DeviceReadiness(evidence = [m, total, mig])
aggregate → ReadinessResult(overall_status, overall_reason, devices, disclaimer, limitations)
D02 finding ← ReadinessResult
exit code ← overall_status  (READY 0, NOT_READY 1, UNKNOWN 3)
```

## 7. Output routing

| Condition | stdout | stderr | file |
| --------- | ------ | ------ | ---- |
| no `--json` (default) | text report | logs (WARNING+ by default) | — |
| `--json PATH` | text report | logs + one line `JSON report written to <PATH>` (printed regardless of log level) | JSON report |
| `--json -` | JSON document + `\n` | logs | — |
| usage error | nothing | usage + error | — |
| JSON write error (`--json PATH`) | text report | `error: cannot write JSON report to …` | none (temp file removed); exit 4 |

## 8. Timing budget (targets, to be measured; NFR-11)

| Step | Expected cost `[ASM]` |
| ---- | --------------------- |
| System + PCI + environment | < 100 ms |
| NVML init + queries (≤ 8 GPUs) | < 1 s |
| `nvidia-smi` locate + version | < 1 s (bounded by 10 s timeout) |
| Toolkit detection (`version.json`, optional `nvcc`) | < 1 s (bounded by 10 s timeout per call) |
| PyTorch probe | dominated by `import torch`; bounded by 60 s timeout |

Measured values are recorded in [app_tracker.md](app_tracker.md) §6 during hardware
validation. They are not claimed in advance.
