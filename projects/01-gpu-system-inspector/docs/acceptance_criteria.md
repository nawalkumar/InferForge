# Acceptance Criteria — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. 60 criteria. AC01–AC12 keep the numbering
given in the product brief. §I maps every criterion to requirements, implementation
components and verification methods.

Verification methods: **UT** unit test · **IT** integration test · **HW-xx** hardware
test ([testing_strategy.md](testing_strategy.md) §6) · **MAN** manual verification by the
user · **REV** review of code or documents. Fixture IDs (FXnn) are defined in
[testing_strategy.md](testing_strategy.md) §4.

A criterion is *met* when every listed method has passed. Hardware-dependent criteria
reach `HARDWARE_VALIDATED` only through their HW test, and `USER_VERIFIED` only on
explicit user confirmation ([app_tracker.md](app_tracker.md)).

## A. Core robustness (brief AC01–AC12)

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC01 | The application runs without an NVIDIA GPU. | FR-01, NFR-02 | IT: FX03, FX04b, FX13 · HW-13 · MAN on the WSL2 workstation | Exit 0; full report produced; `gpu.devices = []`; D04 `INFO`; no traceback on stdout/stderr. |
| AC02 | Missing NVML (the required binding missing from a broken installation, or the driver library) doesn't crash general inspection. | FR-05, FR-13 | IT: FX04a (binding), FX04b (library), FX17 (driver not loaded) | Exit 0; `driver.nvml_binding` `NOT_INSTALLED` (FX04a) or `driver.nvml_version` `NOT_DETECTED` with NVML error code (FX04b, FX17); system, CUDA env and PyTorch sections still populated. |
| AC03 | Missing PyTorch doesn't crash general inspection. | FR-09 | IT: FX05 | Exit 0; `frameworks.pytorch.status = NOT_INSTALLED`; no probe subprocess started (asserted via `FakeCommandRunner` call log); D01 `NOT_APPLICABLE`; D05 `INFO`. |
| AC04 | Driver-reported CUDA compatibility is never labelled as a Toolkit version. | FR-05, FR-07 | UT (reporting, collectors) · IT: FX08 · REV | The value appears only in `driver.driver_supported_cuda_version` (and `nvidia_smi_cuda_version`) with the fixed `note`. With no toolkit, `cuda_environment.toolkits` is `NOT_DETECTED` with value `null`. Text output labels it "Driver supports CUDA up to X.Y". No code path copies it into toolkit fields (test asserts with distinct synthetic values). |
| AC05 | GPU memory metrics preserve their source and meaning. | FR-10, FR-11, FR-12 | UT · IT: FX01, FX04a | Device-wide memory lives only in `runtime[*]` / `gpu.devices[*].memory_total_bytes` with `source` NVML or NVIDIA_SMI and the exact `method`. PyTorch allocator data lives only in `frameworks.pytorch.allocator`. `memory_bus_utilization_percent` is never labelled as memory used. Fallback values carry the 1 MiB precision note. |
| AC06 | Multiple devices are represented independently. | FR-03, FR-11, FR-15 | IT: FX06, FX07, FX10 | One `GPUDeviceInfo`, one `RuntimeMetrics`, one `DeviceReadiness` (Mode B) and per-device D06/D07 findings per device, each keyed by a distinct `device_ref`. Values aren't shared or aggregated across devices. |
| AC07 | VRAM readiness is assessed against the requested threshold. | FR-15 | UT (readiness) · IT: all Mode B runs in testing strategy §4.1 | `required_bytes = round(GiB × 2³⁰)`; each device `READY` iff `free_bytes ≥ required_bytes`; per-device margin is correct; overall aggregation follows architecture §4.9. |
| AC08 | A failed collector doesn't destroy other collected results. | FR-13 | IT: FX10 · UT (guard) | Exit 0. `collection_errors` contains the `cuda_environment` failure (`UNHANDLED_EXCEPTION`) and the `nvml:1` failure (`NVML_ERROR_GPU_IS_LOST`). `cuda_environment` fields are `COLLECTION_FAILED`. All `nvml:0` data and every other section are complete. `collector_runs` shows `FAILED`/`PARTIAL` accordingly. |
| AC09 | JSON output follows a versioned schema. | FR-17, NFR-05 | IT: every fixture validated with `jsonschema` against the shipped schema · REV | Every report validates. `schema_name` and `schema_version = "1.0"` are present. Schema-breaking changes require a MAJOR bump (data model §7.3). |
| AC10 | Diagnostics include supporting evidence and limitations. | FR-14 | UT (invariants) · IT: all fixtures | Every finding with status ≠ `NOT_APPLICABLE` has ≥ 1 evidence item. Every `WARNING`/`FAIL`/`UNKNOWN` has ≥ 1 limitation. Every `WARNING`/`FAIL` has ≥ 1 recommendation. Every finding has check ID, status, severity, title, explanation, detection sources and confidence. |
| AC11 | Hardware tests are separate from regular tests. | NFR-10 | IT (pytest collection) · REV | `uv run pytest` deselects `tests/hardware/`. `uv run pytest -m hardware` on a machine without NVIDIA hardware reports every hardware test as *skipped* with an explicit reason string; none passes vacuously. |
| AC12 | Mocked metrics are never represented as actual measurements. | NFR-10, DEC-17 | UT · IT · REV (production package contains no fixture code) | Every fixture report has `data_origin = "FIXTURE"`, and its text output starts and ends with the fixture-data banner. Live CLI reports have `LIVE`. No demo mode or CLI option produces `FIXTURE`. The tracker never marks an item `HARDWARE_VALIDATED` from UT/IT evidence. |

## B. Data model and provenance

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC13 | No misleading zeros. | FR-12 | UT (`Observed` constructor rejects violations) · IT (invariant over all fixtures) | For every `Observed`, `value != null` ⇔ `status = COLLECTED`. No `0`, `-1`, `""` or `"N/A"` stands in for a missing value. |
| AC14 | Collection statuses follow the normative error mapping. | FR-12 | UT (one case per row of data model §3.3) | Each listed condition yields exactly the specified status and `error.code`. `UNKNOWN` appears in no fixture. |
| AC15 | "PyTorch installed but not importable" is distinguished from "not installed". | FR-09 | IT: FX15 vs FX05 | FX15: `status = COLLECTION_FAILED`, `error.code = IMPORT_ERROR`, error message preserved (redactable), D01 `FAIL`. FX05: `NOT_INSTALLED`. |
| AC16 | Allocator statistics aren't reported as zeros in the CLI. | FR-10 | UT · IT: FX01 via CLI | `frameworks.pytorch.allocator.status = NOT_APPLICABLE` with the specified note. In-process API with CUDA initialised (stubbed) returns `COLLECTED` values with `note = "Calling process only"`. |
| AC17 | CUDA Toolkit detection reports sources and absence correctly. | FR-07 | UT (toolkit trees) · HW-05 | Each candidate lists `root_path`, `discovered_via` and a version `method` (`version.json:cuda.version` or `nvcc --version`). Mismatched candidates are all listed. No candidates → `NOT_DETECTED`. HW-05: detected = `nvcc --version`. |
| AC18 | PyTorch build CUDA version is labelled separately. | FR-09 | UT · IT: FX01, FX02a | `build_cuda_version` carries its note. For a CPU-only build `cuda_build = false` and `build_cuda_version.status = NOT_APPLICABLE`. Text label "PyTorch built with CUDA P.Q". |
| AC19 | PyTorch devices are correlated with NVML devices by UUID/PCI, never by index alone. | FR-09, DEC-09 | UT (correlation) · IT: FX06 · HW-09 | FX06: `cuda:0` → `nvml:1` (`method = uuid`), `cuda:1` → `nvml:0`. Missing UUID falls back to PCI bus ID. No match → `NOT_DETECTED`. No NVML inventory → `NOT_APPLICABLE`. |

## C. Readiness (Mode B)

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC20 | Insufficient memory yields NOT_READY. | FR-15, FR-21 | IT: FX07 (CLI) | Overall `NOT_READY` / `NO_DEVICE_READY`; devices `INSUFFICIENT_FREE_MEMORY` with negative margins; D02 `FAIL`; exit code 1. |
| AC21 | Undeterminable readiness yields UNKNOWN. | FR-15, FR-21 | IT: FX14 (MIG), FX09 (B) (free memory `PERMISSION_DENIED`), FX04b (B) (no device-presence evidence), per testing strategy §4.1 | Overall `UNKNOWN` with the reason in §4.1; D02 `UNKNOWN`; exit code 3. |
| AC22 | Units and boundary are exact. | FR-15, DEC-11 | UT · IT: FX19 | `8` → `8589934592` bytes; `7.5` → `8053063680`; `free == required` → `READY`; a device with `total < required` → `INSUFFICIENT_TOTAL_MEMORY`. |
| AC23 | The readiness disclaimer always accompanies a verdict. | FR-15 | UT · IT: all Mode B fixtures | `readiness.disclaimer` equals the fixed text. The text READINESS section ends with it. Wording rule (diagnostics spec §5) holds. |
| AC24 | `--required-vram` input is validated. | FR-15, FR-21 | UT (CLI parsing) | `0`, `-1`, `nan`, `inf`, `abc`, `8GB`, `1e9` → exit 2 with the specified message, nothing on stdout, no inspection performed (fake adapters record no calls). `8`, `7.5`, `0.5` accepted. |

## D. Reporting and CLI

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC25 | JSON is deterministic, stable and reproducible. | NFR-05, NFR-09 | IT | Two runs of the same fixture produce byte-identical JSON. Keys appear in model order, every key is present, lists are sorted as specified, and there is no NaN/Infinity. The report records schema version, application version, `nvidia-ml-py` and PyTorch versions (when present), effective options, collector durations and `data_origin`. |
| AC26 | Output streams and text-report structure are correct. | FR-16, NFR-07, DEC-18 | IT (CLI with `-vv`) · golden text files (FX01, FX03, FX07, FX13) | `--json -` stdout parses with `json.loads` even at DEBUG verbosity, and no text report appears. Without `--json -`, stdout holds only the text report. All log lines appear on stderr. Text sections appear in the order of cli spec §5.2, and missing values render as status and reason, never as a number. |
| AC27 | `--json PATH` writes atomically and reports failures. | FR-18, DEC-25 | IT (tmp dirs incl. read-only, missing parent, path is a directory) | Success: the file equals the `--json -` output, the text report is on stdout, stderr has `JSON report written to …`, and the exit code follows readiness. Failure: exit 4, the text report is still printed, and no target or temp file is left behind. |
| AC28 | Evidence is classified; wording rules hold. | FR-14 | UT (rules) · IT | Every evidence item has `kind` `OBSERVATION` or `INFERENCE`. Observations reference a `report_path` that resolves in the report. "cause is"/"caused by" only appear with `confidence = HIGH`. Recommendation URLs are on the allowlist. |
| AC45 | `version` command works without NVIDIA or PyTorch. | FR-20 | IT | `inferforge-gpu version` and `inferforge-gpu version --json -` list application (including display name), schema, Python and `nvidia-ml-py` versions. NVML is not initialised and torch not imported (call logs). Exit 0. |
| AC46 | Help is available at every level. | FR-20 | IT | `inferforge-gpu --help`, `inferforge-gpu inspect --help` and `inferforge-gpu version --help` exit 0. The `inspect` help includes the options and the exit-status table of cli spec §4. |
| AC47 | Exit codes follow the specification. | FR-21, DEC-14 | IT (table-driven) | Each row of cli spec §6, including precedence, is produced by a test. |
| AC48 | Unexpected errors are handled cleanly. | NFR-06 | IT (inject exception in core) | Exit 5; nothing on stdout; one-line error on stderr; traceback only with `-vv`. Ctrl-C → exit 130. |
| AC55 | Unsupported capabilities and limitations are reported. | FR-24 | IT: FX11, FX12, FX13 | `unsupported_capabilities` lists the expected capability IDs with reasons citing the platform. `limitations` always includes the snapshot statement. |
| AC58 | Provisional Python API works. | FR-23 | IT | `inspect(InspectionOptions(...))` returns an `InspectionReport` equal (modulo volatile fields) to the CLI's JSON for the same fixture. |
| AC59 | Opt-outs are honoured. | FR-22 | IT | `--no-pytorch` → `pytorch.status = SKIPPED`, no probe started. `--no-processes` → `processes.status = SKIPPED`, no process NVML calls (call log). |
| AC60 | Verbosity controls logging only. | NFR-07 | IT | Default: no INFO lines. `-v`: INFO. `-vv`: DEBUG. Report content is identical across verbosity levels. |

## E. Diagnostics scenarios

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC29 | D01 explains PyTorch CUDA unavailability with evidence. | FR-14 | UT (every row/cause) · IT: FX02a–d, FX15, FX16 · HW-04 | Matrix in diagnostics spec §6 matched exactly: FX02a cause D01.1 HIGH; FX02b cause D01.5 MEDIUM with forward-compatibility limitation; FX02c cause D01.2 HIGH; FX02d confidence LOW with PyTorch's warning quoted; FX15 FAIL; FX16 UNKNOWN. |
| AC30 | D03 explains CUDA version layers without false alarms. | FR-14 | UT · IT: FX01, FX02b, FX08 | The three-line legend is always present when any version is collected. FX08 (no toolkit) is `INFO` and never `WARNING`/`FAIL`. Same-major differences raise nothing. |
| AC31 | D04 reports absence of NVIDIA GPUs gracefully. | FR-14 | IT: FX03, FX04b, FX13, FX17 | FX03 INFO HIGH; FX04b INFO MEDIUM with "cannot confirm" limitation; FX13 INFO (macOS); FX17 FAIL HIGH citing the `nouveau` binding. |
| AC32 | D05 explains missing dependencies and never installs anything. | FR-14, NFR-03, NFR-12 | IT: FX04a, FX05, FX20 · UT (no subprocess other than allowlisted inspection commands is ever run) | FX04a WARNING identifying the missing *required* `nvidia-ml-py` with the reinstall recommendation; FX05 INFO stating that PyTorch isn't installed *in this environment*, with the official selector link; FX20 INFO. The command log contains no `pip`, `uv`, `conda` or installer invocations. |
| AC33 | D06 reports runtime-metric unavailability per device with reasons. | FR-14 | IT: FX09, FX10, FX11, FX12 | FX09 WARNING; FX10 `nvml:1` FAIL (GPU lost), `nvml:0` PASS; FX11/FX12 INFO citing the documented platform limitation. |
| AC34 | D07 reports temperature without inferring throttling. | FR-14 | UT (wording) · IT: FX01, FX18 | FX01 INFO with value and threshold; FX18 WARNING "at or above the device-reported slowdown threshold". The single-sample limitation is always present. Forbidden phrases are absent. |
| AC35 | D08 reports device visibility and ordering effects. | FR-08, FR-14 | IT: FX02c, FX06 · HW-09 | FX02c WARNING (all devices hidden); FX06 INFO with the observed index mapping. |

## F. Platforms and environments

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC36 | System information is collected on every platform profile, and macOS identifies unsupported NVIDIA functionality. | FR-02, NFR-01 | IT: FX01 (Linux), FX11 (WSL2), FX12 (Windows), FX13 (macOS), FX13 (B) · HW-11 | Every `SystemInfo` field is `COLLECTED`, or has the status the data model §4.1 specifies for that platform (e.g. macOS available memory `NOT_SUPPORTED`). macOS: `NVIDIA.ALL` in `unsupported_capabilities`, NVIDIA fields `NOT_SUPPORTED`, exit 0 in Mode A, exit 1 with `PLATFORM_NOT_SUPPORTED` in Mode B. |
| AC37 | Windows WDDM per-process memory is reported as unsupported, not zero. | NFR-01, FR-11 | IT: FX12 · HW-08 | Processes listed; `used_memory_bytes.status = NOT_SUPPORTED` with `VALUE_NOT_AVAILABLE` and the WDDM note. |
| AC38 | WSL2 is detected and its documented limitations are handled. | NFR-01 | IT: FX04b, FX11 · HW-07 · MAN on the WSL2 workstation | `platform_profile = WSL2`. The PCI scan is `NOT_SUPPORTED` (not "no GPU"). The `nvidia-smi` search includes `/usr/lib/wsl/lib`. Utilization/processes `NOT_SUPPORTED` → D06 INFO. |
| AC39 | MIG-enabled devices are handled conservatively. | FR-25 | IT: FX14 | `mig_mode = ENABLED`; readiness device `UNKNOWN/MIG_ENABLED_UNSUPPORTED`; `READINESS.MIG` in `unsupported_capabilities`. |
| AC52 | Real-hardware behaviour matches NVIDIA's own tools. | NFR-01 | HW-01, HW-03, HW-04 | HW pass criteria met on at least one Linux machine with an NVIDIA GPU. Results recorded in the tracker. |
| AC57 | Linux PCI presence check works and is gated by platform. | FR-04 | UT (sysfs trees) · IT: FX03, FX17 · HW-02 | Observations list vendor/device/class/driver. NVIDIA detection uses vendor `0x10de`. Non-Linux and WSL2 → `NOT_SUPPORTED`. |
| AC56 | `nvidia-smi` fallback works and is labelled. | FR-06 | UT (CSV parsing) · IT: FX04a | Inventory and memory from CSV with `source = NVIDIA_SMI`, `inventory_source = NVIDIA_SMI`, `device_ref = smi:<n>`, and the MiB-precision note. Unknown/unsupported fields map per data model §3.3. |

## G. Privacy, safety and performance

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC40 | Redaction removes sensitive values everywhere. | FR-19 | IT (fixture with sensitive literals in paths, process names, UUIDs, PCI IDs, home directory, and inside warnings and error messages) | With `--redact`, no sensitive literal appears anywhere in the JSON, text or stderr log. Placeholders are consistent (the same UUID → the same `<redacted:uuid:N>` in device, PyTorch and findings). `redaction.applied = true` with categories. Without `--redact`, values are unchanged. |
| AC41 | Privacy defaults hold. | NFR-08, FR-08 | UT · REV | The report schema has no hostname, username, command-line, serial or MAC fields. Only allowlisted environment variables appear. Process names are basenames. |
| AC42 | No network access. | NFR-15 | IT (socket patched to raise) | Full inspection completes with no socket creation attempts. |
| AC43 | The tool is read-only. | NFR-12 | UT (NVML call log vs allowlist) · REV (grep for `nvmlDeviceSet`, `nvmlDeviceReset`, `nvmlDeviceClear`, `os.kill`, `signal`) | Only read-only NVML functions are called; no mutating API is referenced in `src/`; no file is written except the `--json` path. |
| AC44 | External calls are bounded by timeouts. | NFR-16 | UT (`FakeCommandRunner` timeouts) · IT: FX16 | Timed-out calls → `COLLECTION_FAILED`/`TIMEOUT`. With a real stub command that sleeps (`python -c "import time; time.sleep(5)"`) and `--timeout 1`, the call returns within 2 s. |
| AC49 | Performance targets are met and measured. | NFR-11 | HW-10 · MAN | Measured wall times recorded in the tracker. Mode A `--no-pytorch` < 2 s on a no-driver machine and < 3 s with NVML (≤ 8 GPUs), or a recorded decision explains any deviation. |
| AC50 | Dependencies match the approved classes and the footprint is minimal. | NFR-03, DEC-05, DEC-26 | REV (`pyproject.toml`) · IT (`importlib.metadata.requires`) | Runtime requirements are exactly `nvidia-ml-py` (with a lower bound). No compiled dependencies. PyTorch is neither a dependency nor an extra. The console script is `inferforge-gpu`. |
| AC51 | Workspace compliance, lint, typing and extensibility. | NFR-04, NFR-13, NFR-14, CON-02–CON-06 | REV · automated commands | `scripts/check_dependencies.py` passes; `ruff check` clean; strict type check clean; `py.typed` shipped; coverage gates of testing strategy §8 met; no imports from other members. A review confirms that adding a collector or diagnostic rule needs only a new class/function plus registration, with no edits to the core loop or existing rules. |
| AC54 | A failure on one device doesn't affect other devices. | FR-13 | IT: FX10 | `nvml:0` runtime fully `COLLECTED`; `nvml:1` fields `COLLECTION_FAILED` with `NVML_ERROR_GPU_IS_LOST`; one `CollectionError` with `device_ref = nvml:1`. |

## H. Documentation

| AC | Criterion | Req. | Verification | Expected result |
| -- | --------- | ---- | ------------ | --------------- |
| AC53 | Documentation is complete and consistent. | NFR-17 | REV · MAN | README states purpose, installation, usage, platform support, dependencies, limitations and current status. Every spec document exists. Statuses, IDs, exit codes and field names agree across documents. The tracker reflects reality. |

## Summary

| Group | Criteria | Count |
| ----- | -------- | ----- |
| A. Core robustness | AC01–AC12 | 12 |
| B. Data model | AC13–AC19 | 7 |
| C. Readiness | AC20–AC24 | 5 |
| D. Reporting & CLI | AC25–AC28, AC45–AC48, AC55, AC58–AC60 | 12 |
| E. Diagnostics | AC29–AC35 | 7 |
| F. Platforms | AC36–AC39, AC52, AC56, AC57 | 7 |
| G. Privacy, safety, performance | AC40–AC44, AC49–AC51, AC54 | 9 |
| H. Documentation | AC53 | 1 |
| **Total** | | **60** |

## I. Traceability matrix

Component IDs are the implementation items in [app_tracker.md](app_tracker.md) §4
(`IMP-M*` models, `IMP-C*` CLI/API, `IMP-K*` core and collectors, `IMP-A*` adapters,
`IMP-D*` diagnostics, `IMP-R01` readiness, `IMP-P*` reporting) plus `TST-01` (test
infrastructure), `ARC-05` (project configuration) and `DOC-*` (documents).

| AC | Requirement(s) | Implementation component(s) | Verification |
| -- | -------------- | --------------------------- | ------------ |
| AC01 | FR-01, NFR-02 | IMP-K08, IMP-A01, IMP-K02, IMP-D04 | IT, HW-13, MAN |
| AC02 | FR-05, FR-13 | IMP-A01, IMP-K03, IMP-K08 | IT |
| AC03 | FR-09 | IMP-A07, IMP-K06 | IT |
| AC04 | FR-05, FR-07 | IMP-K03, IMP-K05, IMP-M02, IMP-P03 | UT, IT, REV |
| AC05 | FR-10, FR-11, FR-12 | IMP-M03, IMP-M04, IMP-K04, IMP-A02 | UT, IT |
| AC06 | FR-03, FR-11, FR-15 | IMP-K02, IMP-K04, IMP-R01, IMP-D06, IMP-D07 | IT |
| AC07 | FR-15 | IMP-R01 | UT, IT |
| AC08 | FR-13 | IMP-K08 | UT, IT |
| AC09 | FR-17, NFR-05 | IMP-P02 | IT, REV |
| AC10 | FR-14 | IMP-D00–IMP-D08, IMP-M05 | UT, IT |
| AC11 | NFR-10 | TST-01 | IT, REV |
| AC12 | NFR-10 | IMP-M07, IMP-P03, TST-01 | UT, IT, REV |
| AC13 | FR-12 | IMP-M01 | UT, IT |
| AC14 | FR-12 | IMP-A01–IMP-A07 | UT |
| AC15 | FR-09 | IMP-A07, IMP-K06, IMP-D01 | IT |
| AC16 | FR-10 | IMP-A07, IMP-M03 | UT, IT |
| AC17 | FR-07 | IMP-A06, IMP-K05 | UT, HW-05 |
| AC18 | FR-09 | IMP-K06, IMP-M03, IMP-P03 | UT, IT |
| AC19 | FR-09 | IMP-K07 | UT, IT, HW-09 |
| AC20 | FR-15, FR-21 | IMP-R01, IMP-D02, IMP-C02 | IT |
| AC21 | FR-15, FR-21 | IMP-R01, IMP-D02, IMP-C02 | IT |
| AC22 | FR-15 | IMP-R01 | UT, IT |
| AC23 | FR-15 | IMP-R01, IMP-P03 | UT, IT |
| AC24 | FR-15, FR-21 | IMP-C01 | UT |
| AC25 | NFR-05, NFR-09 | IMP-P01, IMP-P02 | IT |
| AC26 | FR-16, NFR-07 | IMP-C03, IMP-P03 | IT |
| AC27 | FR-18 | IMP-P05, IMP-C03 | IT |
| AC28 | FR-14 | IMP-D00 | UT, IT |
| AC29 | FR-14 | IMP-D01 | UT, IT, HW-04 |
| AC30 | FR-14 | IMP-D03 | UT, IT |
| AC31 | FR-14 | IMP-D04 | IT |
| AC32 | FR-14, NFR-03, NFR-12 | IMP-D05, IMP-A05 | UT, IT |
| AC33 | FR-14 | IMP-D06 | IT |
| AC34 | FR-14 | IMP-D07 | UT, IT |
| AC35 | FR-08, FR-14 | IMP-D08, IMP-K05 | IT, HW-09 |
| AC36 | FR-02, NFR-01 | IMP-K01, IMP-A04, IMP-P01 | IT, HW-11 |
| AC37 | NFR-01, FR-11 | IMP-A01, IMP-K04 | IT, HW-08 |
| AC38 | NFR-01 | IMP-A04, IMP-A02, IMP-A03 | IT, HW-07, MAN |
| AC39 | FR-25 | IMP-K02, IMP-R01 | IT |
| AC40 | FR-19 | IMP-P04 | IT |
| AC41 | NFR-08, FR-08 | IMP-M01–IMP-M07, IMP-A04 | UT, REV |
| AC42 | NFR-15 | IMP-A01–IMP-A07 | IT |
| AC43 | NFR-12 | IMP-A01, IMP-P05 | UT, REV |
| AC44 | NFR-16 | IMP-A05, IMP-A07 | UT, IT |
| AC45 | FR-20 | IMP-C04 | IT |
| AC46 | FR-20 | IMP-C01 | IT |
| AC47 | FR-21 | IMP-C02 | IT |
| AC48 | NFR-06 | IMP-C02, IMP-K08 | IT |
| AC49 | NFR-11 | IMP-K08 (whole pipeline) | HW-10, MAN |
| AC50 | NFR-03 | ARC-05, IMP-C05 | REV, IT |
| AC51 | NFR-04, NFR-13, NFR-14 | All components; TST-15 | REV, automated |
| AC52 | NFR-01 | IMP-A01, IMP-K02, IMP-K04, IMP-A07 | HW-01, HW-03, HW-04 |
| AC53 | NFR-17 | DOC-01–DOC-16 | REV, MAN |
| AC54 | FR-13 | IMP-K02, IMP-K04 | IT |
| AC55 | FR-24 | IMP-P01 | IT |
| AC56 | FR-06 | IMP-A02, IMP-K02 | UT, IT |
| AC57 | FR-04 | IMP-A03 | UT, IT, HW-02 |
| AC58 | FR-23 | IMP-C06 | IT |
| AC59 | FR-22 | IMP-C01, IMP-K04, IMP-K06 | IT |
| AC60 | NFR-07 | IMP-C03 | IT |

Reverse coverage: every requirement FR-01–FR-25 and NFR-01–NFR-17 is cited by at least
one criterion above. Every criterion has at least one requirement, one implementation
component and one verification method.
