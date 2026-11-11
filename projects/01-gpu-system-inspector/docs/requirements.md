# Requirements — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. Priority keywords: **MUST** (V1 release
blocker), **SHOULD** (V1 target; may be deferred only through a recorded decision in
[app_tracker.md](app_tracker.md)), **MAY** (optional).

Every requirement is verified by the acceptance criteria listed in its "Verified by"
column ([acceptance_criteria.md](acceptance_criteria.md)).

## 1. Functional requirements

| ID | Priority | Requirement | Verified by |
| -- | -------- | ----------- | ----------- |
| FR-01 | MUST | `inferforge-gpu inspect` with no options performs a general inspection (Mode A) and renders a report on every supported platform, including machines with no NVIDIA GPU, no driver and no PyTorch, and even if the required `nvidia-ml-py` binding is missing from a broken installation. | AC01, AC02, AC03 |
| FR-02 | MUST | Collect system information: OS name, release, version; CPU architecture; processor model where determinable; logical CPU count; total system memory; available system memory where supported; Python version, implementation and executable path; WSL2 detection. | AC36, AC38 |
| FR-03 | MUST | Enumerate NVIDIA devices via NVML. For each, record NVML index, name, UUID, total memory, compute capability, PCI bus ID / device ID / subsystem ID and MIG mode, each where accessible. Record the device count. | AC06, AC54 |
| FR-04 | SHOULD | On bare-metal/VM Linux, record driver-independent PCI display/3D-controller observations from sysfs (vendor ID, device ID, class, bound kernel driver), so that an NVIDIA GPU with no loaded NVIDIA driver can be detected. Report `NOT_SUPPORTED` on WSL2, Windows and macOS. | AC57 |
| FR-05 | MUST | Collect driver information: driver version, NVML library version, driver-supported CUDA version (labelled `driver_supported_cuda_version`), CUDA driver library (`libcuda`) presence, `nvidia-ml-py` binding presence and version. | AC02, AC04 |
| FR-06 | MUST | Detect `nvidia-smi` (on `PATH`, plus `/usr/lib/wsl/lib` on WSL2) and record its path and version. When the NVML Python path can't enumerate devices, use `nvidia-smi --query-gpu … --format=csv,noheader,nounits` as a fallback inventory/memory source and tag the values `source = NVIDIA_SMI`. | AC56 |
| FR-07 | MUST | Detect CUDA Toolkit installations from `CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH` and platform default locations. Record each candidate's root, version and version source (`version.json` or `nvcc`). Report `NOT_DETECTED` when none is found. Never infer a toolkit version from the driver. | AC17, AC30 |
| FR-08 | MUST | Record an allowlist of CUDA/PyTorch-related environment variables (data model §5.5), reporting unset variables as `NOT_DETECTED`. Never dump the full environment. | AC35, AC41 |
| FR-09 | MUST | When PyTorch is installed, record distribution version, import status, `torch.__version__`, build CUDA version (`torch.version.cuda`), whether the build has CUDA support, ROCm build flag, cuDNN version, `torch.cuda.is_available()` plus warnings emitted during that call, device count, and per-device name, capability, total memory, SM count, UUID and PCI bus ID where exposed. Correlate each PyTorch device with an NVML device. Distinguish *not installed* from *installed but failed to import*. Never install PyTorch. | AC03, AC15, AC18, AC19 |
| FR-10 | MUST | Report PyTorch allocator statistics (allocated, reserved, peak allocated, peak reserved) only when CUDA is initialised in the calling process (in-process Python API). Otherwise report them `NOT_APPLICABLE` with an explanation. Keep them in a separate structure from device-wide memory. | AC05, AC16 |
| FR-11 | MUST | For each NVIDIA device, collect a runtime snapshot: GPU utilization %, memory-bus utilization %, memory used/free/total/reserved, GPU temperature, slowdown temperature threshold, power draw, enforced power limit, and running compute/graphics processes (PID, type, used memory, process name), each where supported. | AC05, AC06, AC33, AC37, AC38 |
| FR-12 | MUST | Every collected value carries `status`, `source`, `method` and `unit` (when it has one). `value` is non-null **if and only if** `status == COLLECTED`. Statuses come only from the vocabulary in data model §3.1. | AC13, AC14 |
| FR-13 | MUST | Each collector runs in isolation. An exception, timeout or error in one collector (or one device) is recorded in `collection_errors` and doesn't remove or corrupt results from other collectors or devices. | AC08, AC54 |
| FR-14 | MUST | Run the diagnostics engine (D01–D08, [diagnostics_specification.md](diagnostics_specification.md)) on the normalised snapshot. Each finding has check ID, status, severity, title, explanation, evidence (classified as observation or inference), detection sources, recommendations, confidence and limitations. | AC10, AC28, AC29–AC35 |
| FR-15 | MUST | With `--required-vram <GiB>`, assess each device's observed free memory against the threshold (`free >= required` ⇒ `READY`). Produce per-device and overall results with reason codes, margin, evidence and the fixed disclaimer. Use the exit codes in FR-21. | AC07, AC20–AC24 |
| FR-16 | MUST | Render a human-readable text report on stdout with fixed section order, explicit status labels for missing values and the fixture-data banner when not `LIVE`. | AC12, AC26 |
| FR-17 | MUST | Render a JSON report that follows the versioned schema (data model §7). Ship a JSON Schema document with the application. | AC09, AC25 |
| FR-18 | MUST | `--json <path>` writes the JSON report to a file atomically (temporary file then rename), in addition to the text report on stdout. `--json -` writes the JSON report to stdout instead of the text report. Failure to write gives exit code `4`, leaves no partial file, and still prints the text report. | AC27 |
| FR-19 | MUST | `--redact` replaces paths, process names, device UUIDs, PCI bus IDs and the local username/home directory with stable placeholders everywhere in the report, including free-text evidence. The report records which categories were redacted. | AC40 |
| FR-20 | MUST | `inferforge-gpu version` prints the application version, report schema version, Python version and `nvidia-ml-py` version (text, or JSON with `--json`). `--help` is available at every command level. | AC45, AC46 |
| FR-21 | MUST | Use the exit codes in [cli_specification.md](cli_specification.md) §6. | AC20, AC21, AC24, AC47 |
| FR-22 | SHOULD | `--no-pytorch` skips the PyTorch probe; `--no-processes` skips process enumeration. Skipped data is reported with status `SKIPPED`. | AC59 |
| FR-23 | SHOULD | Provide a provisional Python API: `inspect(options) -> InspectionReport`, the model classes, and `to_json()`. It may run the PyTorch probe in-process so that allocator stats are meaningful. The API is unstable (`0.x`). | AC58 |
| FR-24 | MUST | The report lists `unsupported_capabilities` (capabilities that can't be collected on this platform/configuration, with reason) and global `limitations` (snapshot nature, readiness disclaimer, platform caveats). | AC55 |
| FR-25 | SHOULD | Detect MIG mode per device. For MIG-enabled devices, report MIG as a limitation and give readiness `UNKNOWN` with reason `MIG_ENABLED_UNSUPPORTED`. | AC39 |

## 2. Non-functional requirements

| ID | Priority | Category | Requirement | Verified by |
| -- | -------- | -------- | ----------- | ----------- |
| NFR-01 | MUST | Cross-platform | Supported: Linux x86_64/aarch64, Windows 10/11 x86_64, WSL2, and macOS (general information only), on CPython at or above the workspace floor, `>=3.11` (DEC-21, WSD-01). Platform behaviour is as specified in [product_specification.md](product_specification.md) §4. | AC36, AC37, AC38, AC52 |
| NFR-02 | MUST | Graceful degradation | No expected environment condition (missing GPU, driver, library, binding, PyTorch, permission, unsupported query, timeout) causes a crash, a traceback on stdout, or a non-zero exit in Mode A. | AC01–AC03, AC08, AC48 |
| NFR-03 | MUST | Footprint and dependencies | The only required runtime dependency is `nvidia-ml-py` (DEC-05, approved). Everything else is the Python standard library. No compiled third-party dependencies. PyTorch is an optional dependency, detected at runtime and never declared as a dependency or extra (DEC-26). The tool never installs anything. | AC50, AC32 |
| NFR-04 | MUST | Typed interfaces | All public functions and models are fully type-annotated. The package ships `py.typed`. Static type checking passes in strict mode (tool choice OQ-08). | AC51 |
| NFR-05 | MUST | Stable JSON | JSON output follows the schema rules in data model §7: versioned, snake_case keys, every key always present, enums as upper-case strings, integer bytes, RFC 3339 UTC timestamps, deterministic ordering. | AC09, AC25 |
| NFR-06 | MUST | Error handling | Expected failures become statuses, not exceptions. Unexpected exceptions inside a collector are contained (FR-13). An unexpected exception outside collectors gives a one-line message on stderr and exit `5`. With `-vv`, the traceback goes to stderr. | AC48 |
| NFR-07 | MUST | Logging | Standard `logging` to **stderr only**. Default level WARNING; `-v` INFO; `-vv` DEBUG. Stdout carries only the report: the text report, or the JSON report with `--json -`. | AC26, AC60 |
| NFR-08 | MUST | Privacy | Never collect hostname, username, full command lines, the full environment or device serial numbers. Process names are basename only. Redaction is available (FR-19). | AC40, AC41 |
| NFR-09 | MUST | Reproducibility | Report includes schema version, application version, binding/framework versions, effective options, collector durations and `data_origin`. With identical inputs, output is byte-identical except for the volatile fields listed in data model §7.4. | AC25 |
| NFR-10 | MUST | Testability | All external access goes through adapter interfaces that tests can replace. The ordinary test suite needs no GPU, driver, NVML or PyTorch. Hardware tests are separate and skipped by default with a reason. | AC11, AC12 |
| NFR-11 | SHOULD | Performance | Targets (to be measured and recorded, not assumed): Mode A with `--no-pytorch` completes in **< 2 s** wall time on a machine without an NVIDIA driver, and in **< 3 s** with NVML and up to 8 GPUs. The PyTorch probe is bounded by its timeout (default 60 s). The inspector's main process creates no CUDA context. | AC49 |
| NFR-12 | MUST | Safety (read-only) | The tool never calls state-changing NVML functions (`nvmlDeviceSet*`, `nvmlDeviceReset*`, `nvmlDeviceClear*`, etc.), never sends signals to processes, never writes any file other than the `--json` path, never installs software, and never requires elevated privileges. | AC43, AC32 |
| NFR-13 | SHOULD | Extensibility | New collectors and diagnostic rules are added by registering a class/function. They don't require changes to the core loop or existing rules. Schema evolves additively within a major version. | Review (AC51) |
| NFR-14 | MUST | Maintainability | Code passes `ruff` (repository config) and `scripts/check_dependencies.py`. Line coverage ≥ 90 % for `models`, `diagnostics`, `readiness` and `reporting`, and ≥ 80 % overall, on the ordinary suite. | AC51 |
| NFR-15 | MUST | Offline | The tool makes no network connections. | AC42 |
| NFR-16 | MUST | Bounded external calls | Every subprocess call has a timeout (default 10 s for `nvidia-smi`/`nvcc` via `--timeout`, 60 s for the PyTorch probe via `--pytorch-timeout`). A timeout produces `COLLECTION_FAILED` with error code `TIMEOUT`. | AC44 |
| NFR-17 | MUST | Documentation | The README states purpose, installation, usage, platform support, dependencies, limitations and current status. The specification documents stay consistent with each other and with the implementation. The tracker reflects actual status. | AC53 |

## 3. Constraints from the existing repository

| ID | Constraint | Source |
| -- | ---------- | ------ |
| CON-01 | Project lives at `projects/01-gpu-system-inspector/` and is a uv workspace member. | Root `pyproject.toml` (`members = ["packages/*", "projects/*"]`) |
| CON-02 | The project must not depend on or import from another project. Shared code is extracted into `packages/` only when a second consumer exists. | `docs/architecture.md`, `docs/package-policy.md` |
| CON-03 | Dependencies on packages must be declared with a bounded version range and a workspace source. | `projects/README.md` |
| CON-04 | Import names must be unique across workspace members. | `scripts/check_dependencies.py` |
| CON-05 | Ruff `line-length = 100`, `target-version = "py311"` (follows the workspace floor), rules `E,F,I,B,UP`. | Root `pyproject.toml` |
| CON-06 | The project runs its own tests from its directory (`uv run pytest`). It must define its own `[tool.pytest.ini_options]` so the root `testpaths = ["scripts/tests"]` is not inherited. | Root `README.md`, root `pyproject.toml` |
| CON-07 | Licence Apache-2.0. Runtime dependency licences must be compatible (`nvidia-ml-py` is BSD, EV-11). | Root `LICENSE` |

## 4. Traceability summary

| Product feature ([product_specification.md](product_specification.md) §5) | Requirements |
| --- | --- |
| F01 System inventory | FR-02 |
| F02 GPU inventory | FR-03, FR-25 |
| F03 PCI check | FR-04 |
| F04 Driver/CUDA driver | FR-05 |
| F05 `nvidia-smi` | FR-06 |
| F06 Toolkit/env | FR-07, FR-08 |
| F07 PyTorch | FR-09, FR-10 |
| F08 Runtime | FR-11 |
| F09 Provenance | FR-12 |
| F10 Isolation | FR-13, NFR-02 |
| F11 Diagnostics | FR-14 |
| F12 Readiness | FR-15 |
| F13 Text report | FR-16 |
| F14 JSON report | FR-17, FR-18, NFR-05 |
| F15 Redaction | FR-19, NFR-08 |
| F16 CLI basics | FR-20, FR-21 |
| F17 Opt-outs | FR-22 |
| F18 Python API | FR-23 |
| F19 Tests | NFR-10, NFR-14 |
| F20 Docs | NFR-17 |
