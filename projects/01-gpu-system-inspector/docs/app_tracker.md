# Project 01 Tracker — GPU System Inspector

**This file is the single source of truth for Project 01 status.** Other documents
describe *what* to build; only this file says *where it stands*.

Last updated: **2026-11-11 (pre-diagnostics governance reconciliation after the post-IMP-K07 review; earlier the same day: device correlation, IMP-K07)**.
Phase: **Stages 1–5.1 and 6 complete (Stage 6 synthetic and WSL2-host validation; native-Linux PCI hardware validation HW-02 `BLOCKED`). Data foundation, synthetic fixtures, adapter contracts and
the first production collector (system/platform facts from the real host OS, read-only;
macOS via `sysctlbyname`; conservative Linux-environment classification) are in place. The PCI
sysfs adapter (read-only, fail-closed: native Linux only) is implemented and tested
synthetically, with per-environment error codes and the `GPU.PCI_SCAN`/`NVIDIA.ALL` capability
policy. The PCI collector (run status, host-wide collection errors, runtime `GPU.PCI_SCAN`)
is implemented, tested synthetically, reproduces every fixture's PCI outcome, and is gated
correctly on this WSL2 host (DEC-43). **Integration milestone (DEC-44):** the inspection core
(one collector contract, the guard, fail-closed environment, NVML session), the GPU hardware
collector and report assembly run end to end; `api.inspect()` produces a validated `LIVE`
report on this host (§1.7). **Production NVIDIA adapters (DEC-45):** the command runner,
NVML adapter and `nvidia-smi` adapter are implemented, tested against a stub binding, fake
runner and real fake executables, and wired into `api.inspect()` (§1.8); on this machine they
honestly report no NVIDIA driver library and no `nvidia-smi`. **Driver and runtime collectors
(DEC-46):** `DriverInfo` and per-device `RuntimeMetrics` are collected from the shared NVML
session and the cached `nvidia-smi` results, and every fixture's whole driver and runtime
sections are reproduced, FX10's lost GPU included (§1.9). **CUDA environment (DEC-47):** the
toolkit locator and collector report toolkit candidates, the active `nvcc` and the
allowlisted variables, and reproduce every fixture's section (§1.10); toolkit discovery is
not CUDA execution validation. **PyTorch (DEC-48):** the probe (a subprocess of the
inspector's own interpreter, or in-process on request) and its collector report installation,
build, PyTorch's CUDA answer and devices, separately, and reproduce every fixture's section
(§1.11); no CUDA code is run. **Device correlation (DEC-09, DEC-49):** each PyTorch device is
related to an NVML device by UUID, then PCI bus ID, never by index, from the sections already
collected; ambiguity stays unresolved (§1.12). Every collector is implemented. Not
implemented: readiness, diagnostics,
redaction and the CLI `inspect` command (their report parts say `NOT_IMPLEMENTED`). No
NVIDIA hardware validation has been possible (HW-01, HW-02 `BLOCKED`).**

## Status vocabulary

| Status | Meaning |
| ------ | ------- |
| `NOT_STARTED` | Not yet specified or scheduled. |
| `PLANNED` | Specified in the docs and scheduled; no code yet. |
| `IN_PROGRESS` | Work has started. |
| `IMPLEMENTED` | Code exists. For documents: the document is written. |
| `TESTED` | Relevant **automated** tests passed (mock/fixture-based tests count here, and only here). |
| `HARDWARE_VALIDATED` | Verified on **actual compatible hardware** through a hardware test (testing strategy §6). |
| `USER_VERIFIED` | The user **personally confirmed** the behaviour (or approved the document). Never set without explicit user confirmation. |
| `BLOCKED` | Can't proceed; the blocker is recorded in §13. |
| `DEFERRED` | Intentionally moved out of V1; recorded in §14. |

Rules: planned work is never marked implemented; mocked tests never count as hardware
validation; user verification is recorded only with the user's explicit confirmation
(date and what was confirmed).

## 1. Project overview

| Item | Value |
| ---- | ----- |
| Project | InferForge Project 01 — **GPU System Inspector** (approved name) |
| Command | `inferforge-gpu` (`inspect`, `version`); approved (DEC-03) |
| Dependencies | Required: `nvidia-ml-py` (DEC-05). Optional: PyTorch, runtime-detected (DEC-26). No demo mode (DEC-17). |
| Python floor | `>=3.11` (workspace decision WSD-01, approved and applied 2026-11-04). Validated on CPython 3.12.12; direct 3.11 testing pending (§13). |
| Location | `projects/01-gpu-system-inspector/`. Editable uv workspace member: distribution `01-gpu-system-inspector`, import package `gpu_system_inspector` (src layout). |
| Purpose | Cross-layer GPU environment inspection, evidence-based diagnostics, basic free-VRAM readiness and versioned reports ([product_specification.md](product_specification.md)) |
| Application version | `0.1.0` (development; not published, marked `Private :: Do Not Upload`) |
| Report schema | `1.0` (draft until AC09 is TESTED and the spec is approved) |
| Acceptance criteria | 60 ([acceptance_criteria.md](acceptance_criteria.md)) |
| Development machine | WSL2 x86_64 without an NVIDIA GPU (OBS-01). Hardware validation not possible here (§13). |

### Category status overview

| Category | Status | Notes |
| -------- | ------ | ----- |
| Product requirements | `PLANNED` | 25 FR + 17 NFR specified and approved; implementation pending |
| Architecture | `PLANNED` | Approved. DEC-03, -05, -17, -21, -25, -26, -27 approved; project configuration (ARC-05) applied. Stage 4 adapter-boundary decisions DEC-28–DEC-34 applied (need review) |
| Data models | `TESTED` | Stage 2 (§1.2). Schema `1.0` still a draft until AC09 is `TESTED`. |
| CLI | `IN_PROGRESS` | Stage 1 bootstrap `TESTED` (§1.1); `inspect` is a placeholder |
| Collectors | `IN_PROGRESS` | System collector and environment collection `TESTED` (Stage 5, §1.5); PCI collector `TESTED` synthetically, with fixture parity and a WSL2 host run (Stages 6.3–6.5, §1.6.3–§1.6.5); GPU hardware collector `TESTED` synthetically and the core orchestrates all three (§1.7); driver, runtime, CUDA, framework and correlation collectors not written |
| Adapters | `IN_PROGRESS` | Interfaces `TESTED` with fakes (Stage 4); production `OsPlatformAdapter` `TESTED` on this Linux/WSL2 host (Stage 5); production `PciSysfsAdapter` `TESTED` synthetically (Stage 6.1, fail-closed native-Linux gate); production command runner, NVML and `nvidia-smi` adapters `TESTED` (stub binding, fake runner, real fake executables; live on this WSL2 host without NVIDIA, §1.8); PyTorch probe and toolkit adapters not written |
| Diagnostics | `PLANNED` | D01–D08 |
| Readiness | `PLANNED` | |
| Reporting | `IN_PROGRESS` | JSON serialisation and schema `TESTED` (Stage 2); report assembly `TESTED` (IMP-P01, §1.7); text report, redaction pending |
| Testing | `IN_PROGRESS` | 1985 tests pass (through the NVIDIA adapter milestone, including 17 host tests), coverage 97.17% (target 92%). Synthetic fixtures, conforming fakes and contract tests done; tests of real adapters/collectors/engines pending |
| Hardware validation | `BLOCKED` | No NVIDIA hardware on the development machine |
| Documentation | `USER_VERIFIED` | Specification approved by the user 2026-11-04 ("approved for implementation"). Stage 1 amendments listed in §7 need review. |
| Demonstration | `NOT_STARTED` | Live runs only, after implementation; no demo mode (DEC-17) |
| Final review | `NOT_STARTED` | After implementation |

### Implementation readiness

| Gate | State |
| ---- | ----- |
| Product decisions | **Finalized and approved** 2026-11-04 |
| Specification approval | **Approved** by the user 2026-11-04 |
| Workspace Python floor (WSD-01) | **Approved and applied** (`>=3.11`) |
| PyTorch "optional = no extra" (OQ-09) | **Resolved** by the user: runtime detection only, no dependency or extra |
| Stage 1 (workspace migration, package activation, CLI bootstrap) | **Complete** (§1.1) |
| Non-blocking open questions | OQ-03, OQ-06, OQ-08, OQ-11 |
| Stage 2 (data models, envelope, JSON, schema) | **Complete** (§1.2) |
| Stage 3 (fixtures, fakes, coverage, spec corrections) | **Complete** (§1.3) |
| Stage 4 (adapter interfaces, outcomes, lifecycle, `FakeCommandRunner`, conformance) | **Complete** (§1.4) |
| Stage 5 (system and platform inspection) | **Complete** (§1.5) |
| Stage 5.1 (FX13/macOS correction, Linux-environment classification) | **Complete** (§1.5.1) |
| Stage 6.1 (PCI sysfs seam and adapter) | **Complete** (§1.6.1) |
| Stage 6.2 (PCI eligibility policy, error codes, capability semantics) | **Complete** (§1.6.2) |
| Stage 6.3 (PCI collector) | **Complete** (§1.6.3) |
| Stage 6.4 (PCI fixture-level parity) | **Complete** (§1.6.4) |
| Stage 6.5 (host validation) | **Complete** for this WSL2 host (§1.6.5); native-Linux PCI hardware validation (HW-02) `BLOCKED` |
| Stage 6.6 (documentation consolidation, integration readiness) | **Complete** (§1.6.6); documentation awaits user review |
| Integration milestone (IMP-K08 core, IMP-K02 GPU hardware collector, IMP-P01 report assembly) | **Complete** (§1.7): `TESTED` synthetically and on this WSL2 host; documentation awaits user review |
| Production NVIDIA adapters (IMP-A05 command runner, IMP-A01 NVML, IMP-A02 `nvidia-smi`) | **Complete** (§1.8): `TESTED` synthetically and live on this WSL2 host (no NVIDIA); hardware validation `BLOCKED` (HW-01); documentation awaits user review |
| Driver and runtime collectors (IMP-K03, IMP-K04; KI-34, KI-16) | **Complete** (§1.9): `TESTED` synthetically and on this WSL2 host (no NVIDIA); hardware validation `BLOCKED` (HW-01); documentation awaits user review |
| CUDA environment (OQ-10 → DEC-47; IMP-A06 locator, IMP-K05 collector; KI-28) | **Complete** (§1.10): `TESTED` synthetically and on this WSL2 host (no toolkit); HW-05 not run; documentation awaits user review |
| PyTorch (IMP-A07 probe, IMP-K06 collector; DEC-48; KI-35) | **Complete** (§1.11): `TESTED` synthetically, with the probe child run for real against a stub `torch`, and on this WSL2 host (PyTorch not installed); HW-04, HW-06 `BLOCKED`; documentation awaits user review |
| Device correlation (IMP-K07; DEC-09 applied, DEC-49) | **Complete** (§1.12): `TESTED` synthetically and on this WSL2 host (nothing to correlate); HW-09 `BLOCKED`; documentation awaits user review |
| **Verdict** | **Every collector of the inspection pipeline is implemented and validated synthetically, with real NVML, `nvidia-smi`, toolkit and PyTorch adapters (system, PCI, GPU hardware, driver, runtime metrics, CUDA environment, PyTorch, correlation, report).** HW-01, HW-02 `BLOCKED`. Not yet a release: diagnostics, readiness, redaction and the CLI `inspect` command remain. Next recommended: the diagnostics engine (IMP-D*) or the CLI (IMP-C01, IMP-P03), on request. |

### 1.1 Stage 1 tracker (2026-11-04)

| ID | Item | Status | Evidence / notes |
| -- | ---- | ------ | ---------------- |
| S1-WSM | **Workspace migration to Python 3.11** | `TESTED` | Root `pyproject.toml` (`requires-python >=3.11`, `tomli` removed, ruff `py311`, `gpu_system_inspector` first-party); `scripts/_workspace.py` (`import tomllib`); `scripts/verify_isolated_install.py` (`--python` defaults to the running interpreter); `scripts/tests/conftest.py` (`>=3.11`); `README.md`, `projects/README.md`. Root tests 34 passed (incl. 3 slow isolated-install tests); `ruff check .` clean; `check_dependencies.py` passed. |
| S1-LOCK | Lockfile and environment | `TESTED` | `uv lock`: `requires-python >=3.11`; removed `tomli 2.5.0`, `exceptiongroup 1.3.1`, `typing-extensions 4.16.0`; added `nvidia-ml-py 13.615.71`; Project 01 changed from `virtual` to `editable`. `uv lock --check` clean. `uv sync` replaced `.venv` (system 3.10.12 → uv-managed 3.12.12; prompt unchanged). |
| S1-PKG | **Package activation** | `TESTED` | `pyproject.toml`: hatchling, `01-gpu-system-inspector` 0.1.0, `>=3.11`, `nvidia-ml-py>=12.535.108`, no PyTorch, `inferforge-gpu` script, `Private :: Do Not Upload`. Package `src/gpu_system_inspector/` (`__init__`, `__main__`, `py.typed`, `cli/`). Metadata tests pass. |
| S1-CLI | **CLI bootstrap** | `TESTED` | `inferforge-gpu --help`, `inspect --help`, `version` (text, `--json -`, `--json PATH`, atomic, exit 4 on write failure). `inspect` validates the spec's options, then prints that nothing was inspected (stderr) and exits 6 (`NOT_IMPLEMENTED`, development only). No GPU libraries are imported. |
| S1-TST | **Automated tests** | `TESTED` | `projects/01-gpu-system-inspector/tests/`: 47 passed in the workspace (CPython 3.12.12). Covers imports, entry point, help, version, placeholder, validation, metadata. |
| S1-ISO | **Isolated installation** | `TESTED` | The existing `verify_isolated_install.py` rejects projects by design (packages only; enforced by `test_projects_are_rejected`), so its procedure was run manually: wheel built; fresh CPython 3.12.12 venv in `/tmp/inferforge-isolated-01-gpu-system-inspector-Sz25Ry`; installed by path (+ `nvidia-ml-py` from PyPI); package and `pynvml` imported from that venv; console script ran; copied tests 47 passed. Script support for projects is a separate decision (KI-14). |
| S1-PY311 | **Python 3.11 compatibility validation** | `BLOCKED` | No 3.11 interpreter installed; installing one needs user approval. Floor-level code relies only on 3.11 features (`tomllib`). Validated on 3.12.12 only. |

None of these items implements or validates GPU inspection. No acceptance criterion is
complete.

### 1.2 Stage 2 tracker: data models, collection status, JSON foundation (2026-11-04)

Paths are relative to `src/gpu_system_inspector/` and `tests/`. All test evidence is from
the ordinary (mock-free, hardware-free) suite on CPython 3.12.12: **222 passed** in the
workspace, **222 passed** from a fresh wheel in an isolated venv outside the repository.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S2-MOD | **Normalised models** (system, GPU, PCI, driver, CUDA env, toolkit, PyTorch, runtime, processes, diagnostics, readiness) | `TESTED` | `models/{enums,system,gpu,driver,cuda,framework,runtime,diagnostics,readiness}.py` | `unit/test_models.py` (construction, type and unit rejection, immutability, `unavailable()` builders, multi-GPU, CUDA label rules, toolkit/environment rules, readiness and finding invariants) | Pure data. No collector produces them yet. Diagnostic and readiness *logic* not implemented. |
| S2-ENV | **Metric/status envelope** (`Observed[T]`, `ErrorDetail`, `CollectionStatus`) | `TESTED` | `models/observed.py`, `models/enums.py`, `models/_validation.py` | `unit/test_observed.py` (each status, never-zero placeholders, error/note rules, NaN rejection, units, immutability) | `NOT_APPLICABLE` requires `note`, `error` optional (data model §10 #1). |
| S2-COL | **Collection result models** (`CollectionError`, `UnsupportedCapability`, `CollectorRun`, internal `CollectorOutcome`) | `TESTED` | `models/collection.py` | `test_models.py::test_partial_collector_failure_keeps_unrelated_results`, `…collection_error…`, `…collector_outcome…` | `CollectorOutcome` is not serialised (internal to the future core). |
| S2-SNAP | **Inspection snapshot and report models** (`InspectionSnapshot`, `InspectionReport`, `InspectionOptions`, `ApplicationInfo`, `RedactionInfo`) | `TESTED` | `models/report.py` | `test_models.py` (mode/readiness/options agreement, ref resolution, ordering, snapshot limitation, `from_snapshot`/`snapshot` round trip) | Report assembly (limitations text, unsupported capabilities) is a later reporting task. |
| S2-JSON | **JSON serialisation / deserialisation** | `TESTED` | `reporting/json_report.py`, `reporting/__init__.py` | `unit/test_serialization.py` (valid deterministic JSON, stable key order, enums, timestamps, null-not-zero, multi-GPU, lossless round trips, strict decoding errors with JSON Pointer paths) | Decoder is strict (unknown keys and other schema versions rejected, data model §10 #11). Independent of the CLI. |
| S2-SCHEMA | **Versioned JSON Schema** (Draft 2020-12, `$id` `urn:inferforge:gpu-system-inspector:report:1.0`) | `TESTED` | `reporting/schema.py`, `reporting/__main__.py`, `reporting/schema/inspection_report.schema.json` | `unit/test_schema.py` (shipped == generated, meta-schema valid, valid reports pass, 13 mutation classes fail, optional data, per-device errors, version present) | Schema `1.0` stays a draft until AC09 is `TESTED` over FX fixtures. Cross-object rules beyond mode/readiness are enforced by the models, not the schema. |
| S2-UT | **Unit tests** | `TESTED` | `tests/conftest.py` (deterministic `FIXTURE` builders), `tests/unit/test_{observed,models,serialization,schema,boundaries}.py` | 175 new tests; all 222 pass | No coverage tool configured yet (NFR-14 gate pending, KI-17). |
| S2-VAL | **Integration / validation evidence** | `TESTED` | `tests/unit/test_boundaries.py`; isolated wheel run | AST import allowlist (no NVML/PyTorch/subprocess/OS modules in `models/` or `reporting/`); runtime check that importing them loads no `pynvml`, `torch`, `subprocess`, `ctypes`, `socket`; wheel ships the schema; 222 passed in isolated venv | Hardware validation not applicable to this stage and not performed. |

Dependency change: test-only `jsonschema>=4` added to the project's `test` group (testing
strategy §2). Runtime dependencies are unchanged (only `nvidia-ml-py`).

### 1.3 Stage 3 tracker: test fixtures and simulation infrastructure (2026-11-04)

All evidence is synthetic (`data_origin = FIXTURE`, `fixture:` methods). **None of it is
hardware validation, and no acceptance criterion is marked complete because of it.**
Suite: **700 passed** in the workspace (CPython 3.12.12). In an isolated wheel venv,
**675 passed, 25 skipped** (the spec-table comparisons need `docs/`). Paths are relative to
`tests/` unless they start with `src/` or `docs/`.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S3-SPEC | **Specification consistency corrections** (A: `NOT_APPLICABLE`; B: severity; C: device ordering; D: schema provisional) | `TESTED` | `docs/data_model_specification.md` §1, §3.2, §9, §10; `docs/diagnostics_specification.md` §2, D05; `src/…/models/diagnostics.py` (`VALID_SEVERITIES`); regenerated schema | `unit/test_spec_corrections.py` (50 tests: all 36 status/severity pairs in model and schema; `NOT_APPLICABLE` with error/without note; natural ordering stable over shuffled inputs) | Schema `1.0` still provisional (KI-20). |
| S3-FIX | **Fixture implementation**: 20 fixtures, 24 variants, 30 report cases (21 Mode A + 9 Mode B) | `TESTED` | `gsi_testkit/fixtures/{scenario,catalog_core,catalog_platform}.py`, `gsi_testkit/builders.py` | `integration/test_fixture_catalog.py` (exact ID list, descriptions, premises hold, scope of every check, provenance) | FX14 adds one assumption (KI-21). |
| S3-FAKE | **Fake adapter infrastructure** (`FakeNvml`, `FakeNvidiaSmi`, `FakePciScanner`, `FakeTorchProbe`, `FakePlatform`, `FixedClock`, `FakeCollector`, `FakeAdapterSet`) | `TESTED` | `gsi_testkit/fakes/` | `unit/test_fakes.py` (95 tests: fixture parity, read-only NVML call log, shutdown, faults, MiB precision, probe protocol and failure shapes, environment allowlist, determinism, no hardware/process/network imports) | Provisional interfaces until Stage 4 (KI-22). `FakeCommandRunner` deferred. |
| S3-DIAG | **Diagnostic expectation validation** (D01–D08 for every case) | `TESTED` | `gsi_testkit/expectations.py`, `gsi_testkit/spec_tables.py` | Expectations equal diagnostics spec §6 cell by cell; severities valid; every expected finding materialises into a valid `DiagnosticFinding`; a mutation check confirmed that a wrong expectation is detected | No diagnostics engine exists: this validates expectations and the data contract, not rule logic. |
| S3-READY | **Readiness expectation validation** | `TESTED` | `gsi_testkit/expectations.py` | 9 runs equal testing strategy §4.1 (status, reason, devices, D02, exit code); expectations obey the architecture §4.9 rules on each fixture's memory; margins checked | No readiness engine exists (expectation data only). |
| S3-SCHEMA | **JSON Schema validation** | `TESTED` | `src/…/reporting/schema/inspection_report.schema.json` | `integration/test_fixture_contract.py`: all 30 report cases validate; schema rejects every invalid severity | — |
| S3-RT | **Round-trip validation** | `TESTED` | `src/…/reporting/json_report.py` | All 30 cases: `parse(render(r)) == r`, byte-identical re-render, byte-identical rebuild | — |
| S3-MGPU | **Multi-GPU and partial-failure tests** | `TESTED` | `integration/test_partial_failure_multi_gpu.py` | Failed collector keeps other sections; guard simulation reproduces the fixture section; lost GPU doesn't erase the other; nulls not zeros; unrelated findings survive; stable identities and order; NVML↔PyTorch association; per-device readiness | — |
| S3-COV | **Coverage measurement** | `TESTED` | `pyproject.toml` (`[tool.coverage.*]`, test-only `pytest-cov>=5`), `unit/test_model_invariants.py` | Baseline 93% → **95.19%**; `fail_under = 92` enforced by `uv run pytest --cov` | Subprocess-only `__main__` modules are not measured; defensive branches in `_validation.py` at 89%. |

### 1.4 Stage 4 tracker: adapter interfaces and hardware integration boundaries (2026-11-04)

Contracts only, validated with fakes. **No real NVML session, `nvidia-smi` run, PCI read
or PyTorch probe happened, and no hardware behaviour is claimed.** Suite: **975 passed**
in the workspace (CPython 3.12.12). In an isolated wheel venv, **948 passed, 27 skipped**
(the doc-table tests). Coverage **95.85%** (target 92%, unchanged). Paths are relative to
`projects/01-gpu-system-inspector/`.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S4-PROTO | **Protocol definitions**: `NvmlAdapter`/`NvmlSession`/`NvmlDevice`, `NvidiaSmiAdapter`, `PciScanner`, `TorchProbe`, `PlatformAdapter`, `CommandRunner`, `Clock`, plus result types (`NvmlPciInfo`, `NvmlMemoryInfo`, `NvmlUtilization`, `SmiVersion`, `SmiGpuRow`, `TorchProbeReport`) and the read-only NVML allowlist | `TESTED` | `src/gpu_system_inspector/adapters/protocols.py`, `probe_protocol.py`, `__init__.py` | `tests/unit/test_adapter_contracts.py` (208 tests); `tests/unit/test_boundaries.py` (adapters import only `models` + pure stdlib; `models`/`reporting` never import `adapters`; no hardware modules loaded) | No `ToolkitLocator` protocol yet (OQ-10). Conformance is checked at runtime (members and signatures), not by a static type checker (OQ-08). |
| S4-OUT | **Normalised outcomes and error behaviour** (`Observed` as the outcome type; `AdapterUsageError`; `nvml_error_status`, `exception_outcome`, `require_success`, `decode_probe_output`; `CommandResult`) | `TESTED` | `src/gpu_system_inspector/adapters/outcomes.py`, `probe_protocol.py`; data model §3.3 (`OSError` row added) | `tests/unit/test_adapter_outcomes.py` (32 tests: mapping equals data model §3.3 row by row; four failure kinds distinct; decoder accepts the protocol and rejects 14 malformed shapes) | The NVML mapping is exercised with error *names*; translation of real `pynvml` exception classes comes with the NVML adapter. |
| S4-LIFE | **Lifecycle contracts** (NVML init/shutdown, cleanup after success and exceptions, balanced shutdown, failed init, one shared session; process ownership for runner and probe) | `TESTED` (contract, on fakes) | architecture §4.5.3, data flow §4; `tests/gsi_testkit/fakes/nvml.py` | `tests/unit/test_nvml_lifecycle.py` (10 tests) plus lifecycle assertions in `tests/unit/test_fakes.py` (`inits == shutdowns` on all 24 variants) | Proven on `FakeNvml` only; the real adapter must pass the same tests with patched `pynvml` in a later stage. The "core owns the session" rule takes effect when the core exists. |
| S4-CMD | **FakeCommandRunner** (executable availability, exit codes, stdout/stderr, timeout, execution failure, call log, exact/prefix matching, determinism, output cap, usage errors) | `TESTED` | `tests/gsi_testkit/fakes/commands.py` | `tests/unit/test_command_runner.py` (18 tests; `subprocess.Popen` patched to fail if any process starts): fixed `nvidia-smi` argv, canned CSV with `[N/A]`/`[Not Supported]`/`[Insufficient Permissions]`, `EXIT_9`, WSL2 path, probe through the runner | Canned CSV is hand-written, not captured from a real `nvidia-smi` (EV-02 format). |
| S4-CONF | **Fake conformance** (all existing fakes updated to the protocols; existing assertions kept or strengthened) | `TESTED` | `tests/gsi_testkit/fakes/` (`nvml.py`, `nvidia_smi.py`, `torch_probe.py`, `platform.py`, `__init__.py`) | `tests/unit/test_adapter_contracts.py`: `isinstance` + member/signature equality for every protocol (a drifted class is detected); behaviour on 24 variants (outcomes never exceptions, failure propagation, PCI gating, probe/fixture agreement, runner/`nvidia-smi` agreement); `tests/unit/test_fakes.py` (96) and `tests/integration/test_partial_failure_multi_gpu.py` still pass | Changed fake behaviour, by contract: NVML faults are now outcomes instead of exceptions (FX10 error equals the fixture's); `header()` → `version()`; grouped NVML reads; `getenv` returns `Observed`. |
| S4-DI | **Adapter construction design** (`AdapterSet`, explicit injection, null-object partial availability, platform selection, single factory, no registry) | `TESTED` (container); factory `PLANNED` | `src/gpu_system_inspector/adapters/adapter_set.py`; architecture §4.5.4 | `AdapterSet` built from fakes for all 24 variants; rejects non-conforming adapters and non-`DataOrigin` provenance; frozen; every field required | `build_production_adapters` and the core are not implemented (out of scope). |
| S4-CLOCK | **`SystemClock`** (the only production adapter; time only) | `TESTED` | `src/gpu_system_inspector/adapters/clock.py` | Clock contract test runs against `FixedClock` and `SystemClock` | — |
| S4-DOC | **Documentation** | `IMPLEMENTED` (needs review) | architecture §2 (layer roles), §3, §4.5 (rewritten: §4.5.1–4.5.5), §4.7, §5, §6 (DEC-28–DEC-34); data flow §3, §4, §5; data model §3.3; testing strategy §5, §8 | Spec-table test reads the amended data model §3.3 | — |

### 1.5 Stage 5 tracker: system and platform inspection (2026-11-04)

The first production collector. It reads the **real host OS, read-only, through the
standard library only**: no subprocess, no writes, no non-allowlisted environment reads.
No NVIDIA GPU, NVML, `nvidia-smi`, PCI or PyTorch was touched. The collector isn't wired
into `inferforge-gpu inspect` (that needs the orchestration core). Suite: **1097 passed**
in the workspace (CPython 3.12.12), including 9 `host` tests run on this machine (Linux,
WSL2, x86_64). In an isolated wheel venv, **1070 passed, 27 skipped**. Coverage **95.61%**
(target 92%, unchanged). Paths are relative to `projects/01-gpu-system-inspector/`.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S5-ADAPTER | **Production platform adapter** (`OsPlatformAdapter` over `HostApi`; `StdlibHostApi`) | `TESTED` (Linux/WSL2 host + fakes) | `src/gpu_system_inspector/adapters/platform_os.py` | `tests/unit/test_platform_adapter.py` (44 tests); conformance with `PlatformAdapter` and `HostApi` in `tests/unit/test_adapter_contracts.py`; `tests/integration/test_system_host.py` | Windows path tested only through `FakeHostApi`: the real `winreg`/`ctypes` calls have never run (no Windows host, KI-27). macOS processor/memory deferred (KI-26). |
| S5-COLLECTOR | **System information collector** (`SystemCollector`: section, run status, `Clock` duration, `SYSTEM.*` unsupported capabilities) | `TESTED` | `src/gpu_system_inspector/collectors/system.py`; `models/collection.py` (`CollectorOutcome.unsupported`, internal) | `tests/unit/test_system_collector.py` (61 tests: parity with all 24 fixture variants, PARTIAL rule, fixed-clock duration, `unavailable()`, bug propagation); host run `COMPLETED` | No orchestrator/guard yet; `CollectorOutcome` isn't consumed by report assembly yet. |
| S5-OS | **OS information** (`platform_profile`, `os_name`, `os_release`, `os_version`, `os_distribution`, `is_wsl`) | `TESTED` | `platform_os.py` | Fakes: all five profiles, `os-release` fallback/quoting/absence, `/proc/version` unreadable; host: equals `platform.*`, profile matches `sys.platform`, WSL flag agrees with `platform.release()` | No separate "platform identifier" field (covered by `platform_profile` + `os_name` + `architecture`); WSL1 is reported as `WSL2` (KI-29). |
| S5-CPU | **CPU information** (`architecture`, `processor_model`, `logical_cpu_count`) | `TESTED` | `platform_os.py` | Fakes: `/proc/cpuinfo` present/absent/empty, registry present/missing/denied, CPU count `None`; host: equals `platform.machine()` and `os.cpu_count()` | Physical cores not collected (by design). ARM Linux without `model name` → `NOT_DETECTED`. macOS `NOT_SUPPORTED`. |
| S5-MEM | **Physical memory information** (`memory_total_bytes`, `memory_available_bytes`, with meaning notes) | `TESTED` | `platform_os.py` | Fakes: `MemTotal`/`MemAvailable` absent, malformed, permission denied; `MemFree` never substituted; Windows API failure affects only memory; host: positive, available ≤ total, `MemTotal` agrees with `os.sysconf` within a page | WSL2 values describe the VM (stated in `note`). macOS/other `NOT_SUPPORTED`. |
| S5-PY | **Python runtime information** (`python_version`, `python_implementation`, `python_executable`) | `TESTED` | `platform_os.py` | Fakes: empty executable → `NOT_DETECTED`; host: equals `platform.*` | `python_executable` is privacy class P; redaction is a later stage. |
| S5-ENV | **Environment inspection** (`collect_environment`, `OsPlatformAdapter.getenv`) | `TESTED` | `src/gpu_system_inspector/collectors/environment.py`, `platform_os.py` | Reads exactly the 11 allowlisted names (a planted `AWS_SECRET_ACCESS_KEY` is never read); non-allowlisted names rejected before any read; `system_info()` reads no variable; host: matches `os.environ` for allowlisted names | Versioned `CUDA_PATH_V*` names not read (would require listing all names, KI-28). Feeds `CUDAEnvironmentInfo.environment`; the rest of that section (toolkits) is out of scope. |
| S5-UNIT | **Unit tests** | `TESTED` | `tests/gsi_testkit/fakes/host.py` (`FakeHostApi`), `tests/unit/test_platform_adapter.py`, `tests/unit/test_system_collector.py`, `tests/unit/test_boundaries.py` (adapter implementation and collector import rules) | 105 new unit tests plus updated contract/boundary tests; Stage 3/4 tests unchanged in substance and passing | — |
| S5-HOST | **Host integration tests** (`host` marker) | `TESTED` (this Linux/WSL2 machine only) | `tests/integration/test_system_host.py`; `pyproject.toml` (`host` marker) | 9 passed in the workspace and in the isolated wheel venv; process creation patched to fail (no subprocess ran); no machine-specific assumptions; no comparison with fixture data | Not run on native Linux, Windows or macOS. |
| S5-DOC | **Documentation** | `IMPLEMENTED` (needs review) | architecture §3, §4.3 (rewritten), §4.5, §6 (DEC-35–DEC-38); data flow §1.1; data model §4.1; testing strategy §2, §5, §8; project README status | — | — |
| S5-QG | **Quality gates** | `TESTED` | — | Project 1097 passed; coverage 95.61%; root 34 passed; `check_dependencies.py` passed; `ruff check .` and `ruff format --check .` clean; `uv lock --check` clean (no dependency change); isolated wheel (39 entries, no test kit) 1070 passed + 27 skipped | — |

Provenance: real host observations come from `StdlibHostApi` (`LIVE`) and never carry
`fixture:`. Fake-backed observations (`FakeHostApi`, `FakePlatform`) are `FIXTURE` and
always carry it. `AdapterSet` rejects any mix of the two (DEC-38).

**Corrected in Stage 5.1 (§1.5.1):** macOS processor/total memory are now collected
(S5-ADAPTER, S5-CPU and S5-MEM limitations about macOS are superseded); `is_wsl` was
replaced by `linux_environment` (S5-OS); WSL1 is no longer reported as WSL2.

### 1.5.1 Stage 5.1 tracker: platform correctness and documentation hardening (2026-11-05)

A corrective stage, with no new GPU capability. Host inspection stayed read-only; no WSL
distribution, Windows feature, kernel or configuration was touched. Suite: **1194 passed**
in the workspace (9 `host` tests on this machine). In an isolated wheel venv, **1167 passed,
27 skipped**. Coverage **95.77%** (was 95.61%; target 92%).

**Observed WSL evidence on the development machine** (read-only; environment values not
read): `/proc/sys/kernel/osrelease` and `/proc/version` both name the running kernel
`5.15.167.4-microsoft-standard-WSL2`. Corroborating signs, not used by the classifier, are
`/dev/dxg`, the `WSLInterop` binfmt entries, `/run/WSL`, the CPU `hypervisor` flag and
`systemd-detect-virt` → `wsl`; the WSL variables are set, checked by name only. **Verdict: WSL2 currently
running.** `/proc` is generated by the running kernel, so this can't be a residual trace of
the removed installation. On-disk leftovers (e.g. `/etc/wsl.conf`) could be residual and are
not consulted. Production classification on this host: `WSL2`, profile `WSL2`, rationale
"Kernel release '5.15.167.4-microsoft-standard-WSL2' carries the WSL2 kernel marker
'-microsoft-standard-WSL2' (both sources agree)."

| ID | Item | Status | Files | Evidence | Known limitations |
| -- | ---- | ------ | ----- | -------- | ----------------- |
| S51-FX13 | **FX13 discrepancy analysis** | `TESTED` | `tests/gsi_testkit/fixtures/catalog_platform.py` (FX13), `builders.py` | Root cause: Stage 5 deferred macOS processor/total memory (DEC-37) and edited data model §4.1 to match, but the approved §4.1, FR-02 ("total system memory", MUST) and AC36 required them; FX13 kept the approved expectation. FX13 now validates: both premises hold, 8 findings match diagnostics spec §6, Mode B `NOT_READY`, both report cases pass the schema and round-trip, `data_origin` `FIXTURE` | — |
| S51-MACDEC | **macOS support decision**: required for V1 (processor model "where determinable", total memory MUST); available memory stays `NOT_SUPPORTED` as approved | `IMPLEMENTED` (needs review) | architecture DEC-40 (supersedes DEC-37); data model §4.1 restored | No scope change, so requirements and acceptance criteria are unchanged | — |
| S51-MACIMPL | **macOS implementation**: `HostApi.mac_sysctl` → libc `sysctlbyname(3)` via `ctypes`, read-only; strict decoding | `TESTED` (synthetic only) | `src/gpu_system_inspector/adapters/platform_os.py` | Signature and types verified against Apple's `sysctl(3)` manual (`hw.memsize` `int64_t`, `machdep.cpu.brand_string` `char[]`, 0/-1 + `errno`). `tests/unit/test_platform_adapter.py`: collected fields, decode errors (unknown name, empty, invalid UTF-8, wrong size, zero), permission denied; `StdlibHostApi.mac_sysctl` against a simulated libc (argtypes/restype, two-call protocol, `newp` NULL/`newlen` 0, `ENOENT`, `ENOMEM`, missing libc) | **Never run on a Mac.** Apple Silicon `machdep.cpu.brand_string` assumed (not listed in the fetched manual). |
| S51-WSLMODEL | **WSL classification model**: `SystemInfo.linux_environment: Observed[LinuxEnvironment]` (`NATIVE_LINUX`, `WSL1`, `WSL2`, `UNKNOWN`; `NOT_APPLICABLE` status off Linux) replaces `is_wsl`; profile `WSL2` ⇔ `WSL2`; consistency enforced in model and schema | `TESTED` | `models/enums.py`, `models/system.py`, `reporting/schema.py` (`Observed[Enum]` definitions), regenerated schema | Model/schema consistency tests; serialisation rejects unknown members; all 30 report cases validate and round-trip | — |
| S51-WSLPOLICY | **WSL evidence policy** (data model §4.1 rule: unreadable/empty/malformed/conflicting → `UNKNOWN`; `-microsoft-standard-WSL2` → `WSL2`; `<x.y.z>-<build>-Microsoft` → `WSL1`; other Microsoft/WSL marker → `UNKNOWN`; no marker → `NATIVE_LINUX`, not a bare-metal claim) | `TESTED` | `platform_os.classify_linux_environment`; DEC-39 | Kernel-generated evidence only; the word "Microsoft" alone never yields WSL1/WSL2 | The old 4.19 `-microsoft-standard` WSL2 kernels classify as `UNKNOWN` (conservative). |
| S51-WSLTEST | **WSL synthetic test matrix** | `TESTED` | `tests/unit/test_linux_environment.py` (87 tests) | 21 evidence rows (native including Azure/cloud/KVM, WSL1, WSL2 including custom builds, case variations, ambiguous markers, conflicts, empty, blank, malformed), each tested directly and through the adapter; missing/denied/I/O-failing evidence on either file; Windows/macOS/other `NOT_APPLICABLE`; other fields still collected; host test agrees with `platform.release()` | Real native Linux, WSL1 and ambiguous kernels not available. |
| S51-SCHEMA | **Schema/serialisation impact** | `TESTED` | `reporting/schema/inspection_report.schema.json`; data model §7.3 | `is_wsl` removed, `linux_environment` + `LinuxEnvironment` + `ObservedLinuxEnvironment` added; version stays `1.0` because the draft may change until AC09 is `TESTED` (§7.3). No report has been published. Would be MAJOR after the freeze | — |
| S51-PARITY | **Fixture ↔ production parity** (new guard) | `TESTED` | `builders.SYSTEM_PROFILES` drives both `system_info()` and `FakeHostApi.for_profile()` | Production on each synthetic host equals the fixture section; no fixture field differs from production in status/source/method/unit (24 variants) | — |
| S51-DOC | **Documentation review** | `IMPLEMENTED` (needs review) | data model §4.1, §7.3; architecture §4.3, §6 (DEC-37 superseded, DEC-39, DEC-40); data flow §1.1; testing strategy §5, §8; project README | — | — |
| S51-QG | **Quality gates** | `TESTED` | — | Project 1194 passed; coverage 95.77%; 30 report cases (schema + round-trip); root 34 passed; `check_dependencies.py` passed; `ruff check .` and `ruff format --check .` clean; `uv lock --check` clean (no dependency change); isolated wheel (39 entries, no test kit) 1167 passed + 27 skipped | — |

### 1.6 Stage 6 tracker: PCI discovery (FR-04, AC57; DEC-22 and FR-04 approved 2026-11-05)

Stage 6 status: **complete for synthetic/software and WSL2-host validation**; native-Linux
PCI hardware validation (HW-02) `BLOCKED`. 6.1 (§1.6.1: adapter); 6.2 (§1.6.2: eligibility
codes/messages, `GPU.PCI_SCAN` policy, adapter-level fixture parity); 6.3 (§1.6.3: PCI
collector, DEC-43); 6.4 (§1.6.4: collector-level parity for all 24 fixture variants, including
FX03 and FX17); 6.5 (§1.6.5: host run on this WSL2 machine); 6.6 (§1.6.6: documentation
consolidation and integration readiness). FR-04 and AC57 stay `IN_PROGRESS` until an
inspection runs the collector (IMP-K08, IMP-K02) and HW-02 passes.

#### 1.6.1 Stage 6.1 — PCI sysfs seam and adapter (2026-11-05)

No real sysfs was accessed: adapter tests use a dict-based fake (FIXTURE), reader tests use
temporary trees with a guard that fails on any `/sys` path. Suite: **1304 passed**; isolated
wheel **1277 passed, 27 skipped**; coverage **96.04%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S61-SEAM | `SysfsReader` protocol + `StdlibSysfsReader` (listdir, read-only open, readlink; vendor/device/class allowlist; traversal guard) | `TESTED` (temporary trees only) | `src/gpu_system_inspector/adapters/pci_sysfs.py` | `tests/unit/test_pci_sysfs_reader.py` (27) | Real-host sysfs validation deferred to HW-02. |
| S61-GATE | Fail-closed eligibility: explicit `LinuxEnvironment`; only `NATIVE_LINUX` calls the reader | `TESTED` | `pci_sysfs.py`; DEC-41 | Zero reader calls for WSL1/WSL2/UNKNOWN/non-Linux (fake and real reader); invalid eligibility rejected | `UNKNOWN` code/message refinement and `GPU.PCI_SCAN` are Stage 6.2 (done: §1.6.2). |
| S61-ADAPTER | `PciSysfsAdapter` parsing: canonical addresses, lowercase hex IDs/class, base class 0x03 incl. 3D, `is_nvidia`, driver basename, `NO_DRIVER_BOUND`, address ordering | `TESTED` (synthetic) | `pci_sysfs.py`, `tests/gsi_testkit/fakes/sysfs.py` | `tests/unit/test_pci_sysfs_adapter.py` (78) | — |
| S61-FAIL | `PciScan` result: whole-scan outcomes (missing → `NOT_DETECTED`, denied → `PERMISSION_DENIED`, other → `COLLECTION_FAILED`) and typed `PciDeviceFailure`s | `TESTED` | `adapters/protocols.py`, `pci_sysfs.py` | Root failures; invalid address, unreadable/malformed attribute, unreadable driver link; readable devices kept | Mapping to `CollectionError`/`PARTIAL` is Stage 6.3. `PciDeviceFailure.address` holds the raw directory-entry name (privacy class D); it must be redacted by the later redaction stage (IMP-P04). |
| S61-CONTRACT | `PciScanner.scan() -> PciScan`; `FakePciScanner` updated; conformance and gating tests | `TESTED` | `protocols.py`, `fakes/platform.py`, `tests/unit/test_adapter_contracts.py`, `tests/unit/test_fakes.py`, `tests/unit/test_boundaries.py` (`pci_sysfs.py`: `os` only) | 24-variant contract tests; gating now on `linux_environment` | — |
| S61-DOC | `NO_DRIVER_BOUND` (data model §3.2, §4.2); architecture §3, §4.5, §4.5.1, §4.5.5, DEC-22 approved, DEC-41 | `IMPLEMENTED` (needs review) | `docs/data_model_specification.md`, `docs/architecture.md` | — | — |
| S61-QG | Quality gates | `TESTED` | — | 1304 passed; coverage 96.04%; root 34; `check_dependencies.py`, ruff, format, `uv lock --check` clean; isolated wheel 1277 + 27 skipped | — |

#### 1.6.2 Stage 6.2 — PCI eligibility policy, error codes and capability semantics (2026-11-05)

Approved policy (DEC-42). Synthetic only: no real sysfs was accessed, and nothing emits
`UnsupportedCapability` records at runtime yet (PCI collector Stage 6.3, report assembly
IMP-P01). Suite: **1384 passed**; isolated wheel **1357 passed, 27 skipped**; coverage
**96.10%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S62-CODE | **Error-code vocabulary**: `LINUX_ENVIRONMENT_UNKNOWN` (UNKNOWN Linux, so native-Linux-only PCI discovery can't safely run); `PLATFORM_NOT_SUPPORTED` kept for WSL1, WSL2, Windows, macOS | `TESTED` | `src/gpu_system_inspector/adapters/outcomes.py`; data model §3.2, §3.3 | `tests/unit/test_pci_capability_policy.py`: code table per environment; schema-valid `ObservedPciObservationList` with the new code | Error codes are open strings in model and schema (no enum), so no schema change was needed. |
| S62-OUTCOME | **Environment-specific unsupported outcomes**: `NOT_SUPPORTED` everywhere except `NATIVE_LINUX`; pinned, evidence-referenced reasons; zero reader calls | `TESTED` | `adapters/pci_sysfs.py` (`PCI_SCAN_UNSUPPORTED_REASONS`, `PCI_SCAN_UNSUPPORTED_CODES`, `pci_scan_eligible`) | Six environments; `NOT_SUPPORTED` ≠ `COLLECTED []`; adapter has no profile input; `tests/unit/test_pci_sysfs_adapter.py` codes pinned per environment | Windows and macOS share the non-Linux reason (the adapter receives no profile, by design). |
| S62-CAP | **`GPU.PCI_SCAN` policy**: supported only on `NATIVE_LINUX`; unsupported records for WSL1, WSL2, UNKNOWN, Windows and macOS with the scan's reason; macOS also `NVIDIA.ALL` | `TESTED` | `src/gpu_system_inspector/collectors/capabilities.py` (`unsupported_capabilities_for`, pure policy); data model §3.4 | Policy for all six environments; macOS lists both; a `LINUX` profile never makes UNKNOWN/WSL1 eligible; inconsistent environment/profile pairs rejected | Policy only: runtime emission is Stage 6.3 / IMP-P01. |
| S62-PARITY | **Fixture parity**: FX04b, FX11 (WSL2), FX12 (Windows) and FX13 (macOS) built from the production policy; FX13 gains `GPU.PCI_SCAN`; NVIDIA.ALL reason now carries an evidence reference | `TESTED` | `tests/gsi_testkit/builders.py` (`pci_not_supported`), `fixtures/catalog_core.py`, `fixtures/catalog_platform.py` | Adapter reproduces every fixture's `pci_observations` exactly (24 variants; FX03 Intel and FX17 `nouveau` collected); every fixture's PCI/NVIDIA capability records equal the policy; diagnostics, readiness, schema and round-trip tests unchanged and passing | No WSL1 or UNKNOWN fixtures exist; those environments are covered by synthetic unit tests only. |
| S62-CONTRACT | **Contract tests** | `TESTED` | `tests/unit/test_pci_capability_policy.py` (80), `tests/unit/test_pci_sysfs_adapter.py` | Protocol conformance, provenance, fake contracts, boundaries, schema and serialisation suites pass unchanged | — |
| S62-DOC | **Documentation corrections**: native-Linux eligibility wording (data model §4.2, data flow §1 ②, architecture §4.5.4, testing strategy §3.1, §5, §6 HW-02); `LINUX_ENVIRONMENT_UNKNOWN` (data model §3.2, §3.3); capability policy table (data model §3.4); §7.5 example wording; architecture §4.5.1, §4.5.5, DEC-42 | `IMPLEMENTED` (needs review) | `docs/data_model_specification.md`, `docs/data_flow.md`, `docs/architecture.md`, `docs/testing_strategy.md` | Spec-table tests read the amended §3.3 (they validate the NVML/exception rows; the gate rows are pinned in code, not doc-checked) | FR-04, AC57 and D04 row 8 still name only WSL2/Windows/macOS (unchanged, approved documents). Agent documentation review 2026-11-05 passed with observations; its corrections are applied (data model §3.3 trailing text and §3.4 `OTHER` row, architecture §4.5.4 factory bullet and DEC-42); awaiting user review. |
| S62-QG | **Quality gates** | `TESTED` | — | 1384 passed; coverage 96.10%; root 34; `check_dependencies.py`, ruff, format, `uv lock --check` clean (no dependency change); isolated wheel (41 entries, no test kit) 1357 passed + 27 skipped | — |

#### 1.6.3 Stage 6.3 — PCI collector (2026-11-10)

Approved contract (DEC-43). Synthetic only: every scan in the tests goes through the production
`PciSysfsAdapter` over a dict-based `FakeSysfsReader`, `FakePciScanner` or a stub scanner;
no real sysfs was accessed. Nothing runs the collector yet (core IMP-K08, `GPUInventory`
merge IMP-K02). Suite: **1426 passed**; isolated wheel **1399 passed, 27 skipped**; coverage
**96.20%** (`collectors/pci.py` 100%, statements and branches).

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S63-COLLECTOR | **`PciCollector`** (name `pci`): `PciCollector(scanner, clock, *, environment, profile).collect() -> CollectorOutcome`; the scanner's `Observed` section passed through unchanged; duration from the injected `Clock` | `TESTED` | `src/gpu_system_inspector/collectors/pci.py`, `collectors/__init__.py` (`PciCollector`, `PCI_REPORT_PATH`) | `tests/unit/test_pci_collector.py` (41): collected devices and empty scan `COMPLETED`; section identical to the scanner's; `NO_DRIVER_BOUND` `COMPLETED`; clock-controlled `duration_ms` | Not wired into an inspection (IMP-K08); not merged into `GPUInventory` (IMP-K02). |
| S63-SCAN | **Whole-scan mapping**: `NOT_DETECTED` (no sysfs root) → `COMPLETED`, no error, no capability; `PERMISSION_DENIED`/`COLLECTION_FAILED` → `PARTIAL` + one host-wide `CollectionError`; never a collector-owned `FAILED` | `TESTED` | `collectors/pci.py`; architecture §4.4 | Missing root, denied root, I/O-error root; the collector never returns `FAILED` across all mapped outcomes | Downstream handling of a `NOT_DETECTED` scan is open (KI-33). |
| S63-DEVICE | **Per-device failures**: each omitted device (`INVALID_ADDRESS`, `ATTRIBUTE_MALFORMED`, `ATTRIBUTE_UNREADABLE` with `NOT_DETECTED`/`PERMISSION_DENIED`/`COLLECTION_FAILED`) → one `CollectionError` (`pci`, `device_ref` null, the failure's status/code, `<address> <attribute or 'address'>: <message>`, `/gpu/pci_observations`) and a `PARTIAL` run; `DRIVER_LINK_UNREADABLE` → `PARTIAL`, device and failed `kernel_driver` kept, no `CollectionError` | `TESTED` | `collectors/pci.py` | Each omitted kind and status; readable devices kept; errors in the scan's `(address, attribute)` order, identical on repeat; three driver-link failure forms (denied, I/O error, unusable target) | `CollectionError.message` holds the raw PCI address (privacy class D); redaction is IMP-P04. A `PARTIAL` scan must not be read as proof of absence (KI-32). |
| S63-CAP | **Runtime `GPU.PCI_SCAN`**: ineligible environments → `COMPLETED` plus the PCI-only record from `unsupported_capabilities_for`; no `NVIDIA.ALL` | `TESTED` | `collectors/pci.py`, `collectors/capabilities.py` (unchanged) | WSL1, WSL2, `UNKNOWN`, Windows, macOS, `OTHER`: zero reader calls, exactly one `GPU.PCI_SCAN` with the pinned reason, no `NVIDIA.ALL` | `NVIDIA.ALL` emission belongs to IMP-K02/IMP-P01. |
| S63-GUARD | **Fallback and consistency**: `unavailable(error: ErrorDetail)` → `COLLECTION_FAILED`, `source`/`method` `None`; `AdapterUsageError` for inconsistent environment/profile, for a scan that contradicts the environment (status, code or reason), and for statuses outside the mapping; unexpected exceptions propagate | `TESTED` | `collectors/pci.py` | Fallback equals the shared convention; 4 mismatch cases; 4 inconsistent pairs; `UNKNOWN`/`NOT_INSTALLED`/`SKIPPED`/`NOT_APPLICABLE` scans rejected; scanner and reader exceptions propagate | The guard itself (`UNHANDLED_EXCEPTION`, `FAILED`) is IMP-K08. |
| S63-CONTRACT | **Model and schema contracts** | `TESTED` | `tests/unit/test_pci_collector.py`; `tests/unit/test_boundaries.py` (covers `collectors/pci.py` automatically) | Section, run, errors and capability records validate against the shipped schema (`ObservedPciObservationList`, `CollectorRun`, `CollectionError`, `UnsupportedCapability`); an `InspectionSnapshot` accepts them; collectors import only models, adapters and stdlib | No model or schema change. |
| S63-DOC | **Documentation**: DEC-43; architecture §3 layout, §4.2 (protocol marked aspirational), §4.4 (PCI ownership, mapping table, interface), §4.5.5 cross-reference; data flow §3 (run-status kinds, `ErrorDetail` fallback); this tracker (IMP-K02, IMP-K09, KI-32, KI-33) | `IMPLEMENTED` (needs review) | `docs/architecture.md`, `docs/data_flow.md`, `docs/app_tracker.md` | — | Data model §3.4 unchanged (no contradiction found). Consolidated in Stage 6.6 (§1.6.6). |
| S63-QG | **Quality gates** | `TESTED` | — | 1426 passed (including 9 host tests, which read `/proc` only); coverage 96.20% (gate 92%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (110 files), `uv lock --check` clean (no dependency change); isolated wheel (42 entries, `collectors/pci.py` shipped, no test kit) 1399 passed + 27 skipped | — |

#### 1.6.4 Stage 6.4 — PCI fixture-level parity (2026-11-10)

No fixture was changed. Synthetic only; no real sysfs was accessed.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S64-PARITY | **Collector-level parity, all 24 variants**: section, `pci` run (including its fixed duration), `pci` collection errors and the `GPU.PCI_SCAN` record equal the fixture's; ineligible fixtures make zero reader calls | `TESTED` | `tests/integration/test_pci_collector_fixtures.py` | `test_collector_reproduces_every_fixture_pci_outcome` (24); `test_collector_output_rebuilds_the_fixture_snapshot` (24: substituting the collector's output leaves the snapshot and its JSON unchanged) | — |
| S64-NAMED | **Named outcomes**: FX03 (one Intel controller) and FX17 (NVIDIA + `nouveau`) `COLLECTED`/`COMPLETED`, no errors or records; FX04b, FX11 (WSL2), FX12 (Windows), FX13 (macOS) `NOT_SUPPORTED`/`COMPLETED` with the PCI record only; FX13's `NVIDIA.ALL` stays with its later owner | `TESTED` | same | `test_named_fixture_outcomes` | — |
| S64-FAKES | **Test kit**: `fake_collectors()` gains a `pci` entry (section = the fixture's `pci_observations`, fallback = `PciCollector.unavailable`); `sysfs_reader_for()` builds a synthetic sysfs holding exactly a fixture's observations (now shared with the 6.2 policy test) | `TESTED` | `tests/gsi_testkit/fakes/collectors.py`, `fakes/sysfs.py`, `fakes/__init__.py`; `tests/unit/test_pci_capability_policy.py` (uses the helper) | `test_fake_scanner_and_production_adapter_agree` (24), `test_fake_pci_collector_matches_the_real_collector` (24); `test_fakes.py` unchanged and passing | `gpu_hardware` still returns the whole `GPUInventory`, `pci_observations` included, until IMP-K02. |
| S64-COVERAGE | **Catalogue coverage**: which PCI outcomes the fixtures pin | `TESTED` | same | `test_fixture_catalogue_pci_coverage`: collected with NVIDIA, collected non-NVIDIA, `NOT_SUPPORTED`; no fixture has a `pci` error or a non-`COMPLETED` `pci` run | No fixture pins a missing root (`NOT_DETECTED`), an empty scan or a partial enumeration. Those are covered synthetically in `unit/test_pci_collector.py`; no fixture was added (FX01–FX20 are approved specification). |

#### 1.6.5 Stage 6.5 — Host validation (2026-11-10)

Host: Windows 11 with Ubuntu 22.04 under WSL2, kernel `5.15.167.4-microsoft-standard-WSL2`,
Intel Iris Xe (no NVIDIA GPU). The production platform adapter classifies it as
`linux_environment = WSL2`, profile `WSL2` (both kernel sources agree). Nothing was installed
and no WSL, Docker or GPU configuration was changed.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| S65-SYSTEM | **System host suite** (platform identification, WSL2 and Linux-environment classification, system information, serialisation) | `TESTED` (this host) | `tests/integration/test_system_host.py` (unchanged) | 9 `host` tests passed | Windows and macOS host runs still pending (KI-27, KI-30). |
| S65-PCI | **PCI on this host**: production adapter (real reader) and collector, with the environment from `OsPlatformAdapter()`. `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED` with the WSL2 reason, `LIVE` method `/sys/bus/pci/devices` without prefix, `COMPLETED` run, only `GPU.PCI_SCAN` (no `NVIDIA.ALL`), schema-valid; a guard proved nothing under `/sys` was opened, listed or resolved | `TESTED` (this WSL2 host) | `tests/integration/test_pci_host.py` (new, `host`) | 4 `host` tests passed (13 `host` tests in total) | A native Linux host is skipped by design: its real scan is HW-02. |
| S65-HW | **Native-Linux PCI hardware validation** (HW-02) | `BLOCKED` | — | — | No native Linux machine with an NVIDIA GPU is available. WSL2 doesn't enumerate GPUs on PCI, so fake-reader and WSL2 results aren't native PCI validation. |

#### 1.6.6 Stage 6.6 — Documentation consolidation and integration readiness (2026-11-10)

| ID | Item | Status | Files | Evidence | Known limitations |
| -- | ---- | ------ | ----- | -------- | ----------------- |
| S66-DOC | **Consolidation**: architecture §4.5.5 heading (was "not written yet"), DEC table back in numeric order (DEC-38), §4.11 added; data flow §1 step ② names `PciCollector` and the IMP-K02 merge; data model §4.2 (`NOT_DETECTED` root; omitted devices → `pci` errors, so `COLLECTED` isn't proof of a complete enumeration); testing strategy §5 (`pci` fake, `sysfs_reader_for`, three new test modules, host note), §8 (Stage 6 coverage) | `IMPLEMENTED` (needs user review) | `docs/architecture.md`, `docs/data_flow.md`, `docs/data_model_specification.md`, `docs/testing_strategy.md` | Re-read against the source; spec-table tests unaffected (they read data model §3.3, testing strategy §4.1, diagnostics §6) | — |
| S66-READY | **Integration readiness for IMP-K08 and IMP-K02** (architecture §4.11): collector shape, `CollectorOutcome` invariants, guard-owned `FAILED`, duration, order and dependencies, fail-closed environment mapping, scanner construction, capability ownership, `GPUInventory` invariants and fallback, `NVIDIA.ALL` | `IMPLEMENTED` (needs user review) | `docs/architecture.md` §4.11 | Verified against source: `GPUInventory.unavailable()` overwrites `pci_observations`; a failed system collector leaves `linux_environment` `None` on a `LINUX` profile, which `unsupported_capabilities_for` rejects (`AdapterUsageError`) | Not implemented: IMP-K08 and IMP-K02 are the next milestones. |
| S66-TRACKER | **Tracker consolidation** (header, readiness, §1.6, FR-04, AC57, TST-04/05/15, HW-02, IMP-K02/K08/K09, IMP-D01/D04/R01/P01/P04, DOC-04/05/07/10, KI-32/33, §10–§12, §15, §16) | `IMPLEMENTED` | this file | — | — |
| S66-QG | **Final quality gates** | `TESTED` | — | 1528 passed (13 `host`); coverage 96.20% (gate 92%; `pci_sysfs.py`, `capabilities.py`, `pci.py` 100%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (112 files), `uv lock --check` clean (no dependency change); isolated wheel (42 entries, no test kit) 1501 passed + 27 skipped | — |

### 1.7 Integration milestone — IMP-K08, IMP-K02, IMP-P01 (2026-11-11)

Approved scope; design recorded as DEC-44 (architecture §4.2, §4.4, §4.10, §4.11). Unit and
integration tests use fakes only (no real `/sys`, NVML, process or hardware). The host tests
and the demonstration ran on this WSL2 machine (no NVIDIA GPU). Suite: **1807 passed**
(17 `host`); isolated wheel **1780 passed, 27 skipped**; coverage **96.63%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| K08-CONTRACT | **One collector contract** (`Collector`: `name`, constructor injection, `collect() -> CollectorOutcome`, `unavailable(ErrorDetail)`); `FakeCollector` now returns the fixture's `CollectorOutcome` | `TESTED` | `collectors/base.py`; `tests/gsi_testkit/fakes/collectors.py` | `unit/test_core_guard.py`: every production collector, the NVML step and every fake conform; fakes' outcomes accepted by the guard for all 24 fixtures | Architecture §4.2's `collect(ctx)` sketch replaced (DEC-44); no `CollectionContext`. |
| K08-GUARD | **Single guard**, only producer of `FAILED`: one host-wide `UNHANDLED_EXCEPTION` error with the collector's report path, `unavailable(ErrorDetail)`, guard-timed run; `AdapterUsageError` and contract violations contained; `KeyboardInterrupt`/`SystemExit` propagate | `TESTED` | `core/guard.py` | Pass-through, host states not failures, exception → one `FAILED` run, usage errors named, 3 contract violations, base exceptions propagate | — |
| K08-ORDER | **Order and fail-closed environment**: `system` → resolve (`LINUX` → `UNKNOWN`, `WSL2` → `WSL2`, else `None`) → scanner built from it → `pci` → `nvml` → `gpu_hardware`; `DeferredPciScanner` in the production set | `TESTED` | `core/inspector.py`, `core/environment.py`, `core/factory.py` | Every profile's fallback resolution; system failure on Linux and WSL2 fixtures (no scan, `LINUX_ENVIRONMENT_UNKNOWN`); pre-built native scanner refused; runs in `COLLECTOR_ORDER` for all 24 fixtures | — |
| K08-NVML | **NVML session**: `NvmlSessionStep` (run `nvml`) opens it, `FailedNvmlSession` if opening raised; closed after `gpu_hardware` (since DEC-46: after `runtime`) on every path | `TESTED` | `core/inspector.py` | Reproduces every fixture's `nvml` run and errors (FX04a `PARTIAL`); opened once and closed for all 24 fixtures and after a GPU-collector exception | — |
| K02-INVENTORY | **`GpuHardwareCollector`**: NVML devices, `nvidia-smi` fallback (`FALLBACK_FIELD_UNAVAILABLE` fields), per-device isolation, `PARTIAL` rules, `NVIDIA.ALL` on macOS | `TESTED` (synthetic) | `collectors/gpu_hardware.py` | `unit/test_gpu_hardware_collector.py`: parity with 23 fixtures (inventory, run, records); FX10 pinned (KI-34, since resolved: all 24, §1.9); unreachable handle; fallback; failed count/query reported | No production NVML or `nvidia-smi` adapter yet (IMP-A01, IMP-A02): live inventories report `NOT_IMPLEMENTED`. |
| K02-PCI | **One PCI scan; PCI kept through fallbacks**: no scanner parameter; `unavailable()` restores the PCI section | `TESTED` | same | No PCI call; fallback keeps the PCI section (4 fixtures); inspector: one scan per inspection, PCI kept after a GPU-collector exception | — |
| P01-ASSEMBLY | **Report assembly**: metadata, limitations (snapshot statement first, once), rejects duplicate runs and capability records; snapshot aggregation in the core | `TESTED` | `reporting/assemble.py`, `core/inspector.py` | `integration/test_inspector.py`: 24 fixture reports schema-valid, round-trip, byte-identical on repeat; duplicate rejection; one owner per capability (FX13) | Diagnostics, readiness (Mode B → `NotImplementedError`) and redaction not implemented. |
| P01-PLACEHOLDERS | **Sections without a collector**: driver and CUDA environment `NOT_SUPPORTED`/`NOT_IMPLEMENTED`, PyTorch `SKIPPED`, runtime `[]`, no run, a stated limitation | `TESTED` | `core/inspector.py`; data model §3.2, §9 | Every report: placeholder codes, `SKIPPED` PyTorch, no findings, no `FAILED` | Driver and runtime placeholders replaced by DEC-46 (§1.9); CUDA environment and PyTorch remain. |
| API | **Public API** `api.inspect(options)` with the production adapter set | `TESTED` (this WSL2 host) | `api.py`, `core/factory.py` | `integration/test_inspect_host.py` (4 `host`): valid round-tripping `LIVE` report, runs in order, PCI gated once, NVIDIA sections `NOT_IMPLEMENTED`; no `/sys` access, no process | Native Linux skipped (HW-02). CLI `inspect` still a placeholder (KI-15). |
| STANDINS | **Not-implemented production adapters** (NVML, `nvidia-smi`, command runner, PyTorch probe) | `TESTED` | `adapters/unimplemented.py`, `adapters/outcomes.py` (`NOT_IMPLEMENTED`) | Protocol conformance (`unit/test_adapter_contracts.py`); every operation `NOT_SUPPORTED`/`NOT_IMPLEMENTED`, schema-valid, never raises; can't enter a fixture set | Temporary; each replaced by IMP-A01, A02, A05, A07. |
| FIXTURE-FX13 | **FX13 correction**: `SYSTEM.MEMORY_AVAILABLE` reason taken from its own system section (it contradicted it); system collector test now compares whole records | `TESTED` | `tests/gsi_testkit/fixtures/catalog_platform.py`, `tests/unit/test_system_collector.py` | Catalogue, contract, schema and system-collector suites pass | The only fixture change; FX10 left as is (KI-34). |
| DEMO | **End-to-end demonstration** `uv run python examples/inspect_demo.py [--json -\|PATH]` | `TESTED` (run manually on this WSL2 host) | `examples/inspect_demo.py` (not shipped) | Exit 0; report valid against the schema and round-trips; WSL2, PCI `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`, `GPU.PCI_SCAN` once, GPU `NOT_IMPLEMENTED`, four runs `COMPLETED` | Not an automated test. |
| K-DOC | **Documentation**: DEC-44; architecture §3, §4.2, §4.4, §4.5.4, §4.5.5, §4.10, §4.11; data flow §1, §1.2, §3; data model §3.2, §9; testing strategy §4, §5, §8 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| K-QG | **Quality gates** | `TESTED` | — | 1807 passed (17 `host`); coverage 96.63% (all new modules 100%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (127 files), `uv lock --check` clean (no dependency change); isolated wheel (52 entries, `core/` and `api.py` shipped, no test kit or examples) 1780 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

### 1.8 Production NVIDIA adapters — IMP-A05, IMP-A01, IMP-A02 (2026-11-11)

Approved scope; design recorded as DEC-45 (architecture §4.4, §4.5.5). No new dependency: the
NVML adapter uses the existing required `nvidia-ml-py` binding (DEC-05; `uv.lock`
unchanged). Unit and integration tests use a stub `pynvml`, the fake command runner and tiny
fake executables; no NVIDIA library or tool was used in them. Suite: **1985 passed** (17
`host`); isolated wheel **1958 passed, 27 skipped**; coverage **97.17%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| A05-RUNNER | **`SubprocessCommandRunner`**: list argv, no shell, no stdin, bounded timeout (child killed), captured stdout/stderr, exit code, clock duration, output cap, environment never recorded; `which` on PATH plus directories | `TESTED` | `adapters/commands.py` | `unit/test_subprocess_command_runner.py` (36): real fake executables (success, verbatim shell metacharacters, non-zero exit, empty output, no stdin, timeout, missing, permission denied, cap) and a mocked boundary (`shell=False`, `stdin=DEVNULL`, no `env`, other `OSError`, malformed result, unexpected exception, 10 usage errors); no `subprocess` import at package import | POSIX fake executables; Windows process semantics not exercised (no Windows host). |
| A01-NVML | **`PynvmlAdapter`/`PynvmlSession`/`PynvmlDevice`**: lazy injectable binding, macOS gate, error mapping by NVML value, one init / at most one shutdown, swallowed shutdown errors, independent reads, `VALUE_NOT_AVAILABLE`/`PARSE_ERROR` instead of invented values, older-call fallbacks, read-only calls | `TESTED` (stub binding; live on this host: library not found) | `adapters/nvml.py` | `unit/test_nvml_adapter.py` (75): stub checked against the real binding's codes and messages; every init outcome; 0/1/n devices; properties, conversions, isolation, fallbacks, lifecycle, usage errors; parity with **every fixture's** NVML facts and GPU inventory (FX10 per KI-34, since resolved, §1.9); `unit/test_nvml_lifecycle.py` unchanged | Never run against an NVIDIA driver (HW-01 `BLOCKED`). KI-16 resolved in §1.9 (floor confirmed at the binding level). |
| A02-SMI | **`NvidiaSmiCli` + `parse_query_output`**: PATH and WSL2 lookup, macOS gate, header versions (old and new labels), CSV query with per-field outcomes, whole-query failure on structural damage, exit/unknown-field/timeout/truncation mapping | `TESTED` (fake runner and a fake executable) | `adapters/nvidia_smi.py` | `unit/test_nvidia_smi_adapter.py` (39), including a fake `nvidia-smi` script through the production runner and FX04a's fallback inventory reproduced by both production adapters | Never run against a real `nvidia-smi` (not installed here; HW-01). |
| A-POLICY | **NVML / `nvidia-smi` policy** (precedence, never merge; architecture §4.4 table) | `TESTED` | `collectors/gpu_hardware.py` (unchanged), DEC-45 | `integration/test_production_nvidia_pipeline.py` (20): the full matrix (NVML wins with CLI working, failing or absent; NVML failure + CLI; partial NVML not merged; both failing; zeros; both missing ≠ no GPU; missing binding) | — |
| A-WIRING | **Production factory and API**: real runner, NVML and `nvidia-smi` adapters; `subprocess_timeout_s` from the options; PyTorch probe still a stand-in | `TESTED` | `core/factory.py`, `api.py`, `adapters/__init__.py` | `unit/test_core_guard.py` (factory, timeout); `unit/test_adapter_contracts.py` (signature conformance of the production adapters); `unit/test_boundaries.py` (only `commands.py` imports `subprocess`/`shutil`; `pynvml` never imported statically) | — |
| A-INTEGRATION | **Pipeline guarantees with the production adapters**: one NVML session and shutdown, one PCI scan, PCI kept after a GPU-collector exception, `NVIDIA.ALL`/`GPU.PCI_SCAN` once, independent failures, failing shutdown harmless, schema-valid, round trip, deterministic | `TESTED` | — | `integration/test_production_nvidia_pipeline.py` | — |
| A-HOST | **Live on this WSL2 host** (Intel Iris Xe, no NVIDIA): `api.inspect()` and the demo | `TESTED` (this host only) | `tests/integration/test_inspect_host.py` (updated: honest-inventory check; only `nvidia-smi` may be started) | 17 `host` passed; demo exit 0: NVML `NOT_DETECTED`/`NVML_ERROR_LIBRARY_NOT_FOUND`, `nvidia-smi` not found (nothing run), four runs `COMPLETED`; same from the installed wheel | Not NVIDIA validation of any kind. |
| A-DOC | **Documentation**: DEC-45; architecture §3, §4.4 (policy table), §4.5, §4.5.4, §4.5.5, §4.11; data flow §1.2, §4; data model §3.2, §3.3 (`nvidia-smi` rows); testing strategy §3.1, §5, §8 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| A-QG | **Quality gates** | `TESTED` | — | 1985 passed (17 `host`); coverage 97.17% (`commands.py`, `nvml.py`, `nvidia_smi.py`, `factory.py`, `api.py` 100%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (135 files), `uv lock --check` clean, `uv.lock` byte-identical; isolated wheel (55 entries, the three adapters shipped, no test kit or examples) 1958 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

### 1.9 Driver and runtime collectors — IMP-K03, IMP-K04; KI-34, KI-16 (2026-11-11)

Design recorded as DEC-46 (architecture §4.4). No new dependency, no model or schema change
(`uv.lock` unchanged). Scope note: in this repository IMP-K04 is the *runtime metrics*
collector (per-device telemetry); CUDA Toolkit/`nvcc` discovery is IMP-K05 with the toolkit
locator IMP-A06 and waits for OQ-10. The CUDA runtime library and CUDA usability have no
model field and need the PyTorch probe (IMP-A07/K06); the report's limitation says so.
Suite: **2106 passed** (18 `host`); isolated wheel **2079 passed, 27 skipped**; coverage
**97.37%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| K03-DRIVER | **`DriverCollector`** (run `driver`): binding metadata, `init_result`, driver version, driver-supported CUDA (fixed note), `libcuda` version, from the shared session; system calls only | `TESTED` (synthetic; live on this host: NVML library not found) | `collectors/driver.py` | `unit/test_driver_collector.py` (63 with parametrisation): whole-section parity with all 24 fixtures; call list; field-level `PARTIAL`; no session → `COMPLETED`; guard fallback | Never run against an NVIDIA driver (HW-01). |
| K03-SMI | **`NvidiaSmiCollector`** (run `nvidia_smi`): location and header; precedence NVML → query → header; disagreement kept and noted; failed lookup/header `PARTIAL` with one error; not installed is a fact | `TESTED` | `collectors/driver.py` | Same file: precedence, disagreement, fallbacks, AC04 distinct values, failures, crash containment; `integration/test_production_nvidia_pipeline.py` (header run once, query never when NVML counts) | `nvidia_smi_path` keeps the runner's `shutil.which(nvidia-smi)` labels (source `OS`) in production. |
| K04-RUNTIME | **`RuntimeCollector`** (run `runtime`): one record per inventoried device from the inventory's source; process opt-out; lost device → unavailable + one error, no partial values; field failures field-level; `RUNTIME.*` records | `TESTED` (synthetic) | `collectors/runtime.py` | `unit/test_runtime_collector.py` (40): whole-section parity, runs, errors and records for all 24 fixtures; AC59; smi fallback; isolation; mid-read loss; notes the adapter can't know (WSL2, MIG, WDDM); production adapter conversions | Compute processes only (graphics not merged). Never run on NVIDIA hardware (HW-01). |
| K-SMI-CACHE | **`NvidiaSmiCli` caches `version()` and `query_gpus()`** (as `locate()`): each command at most once per inspection | `TESTED` | `adapters/nvidia_smi.py` | `unit/test_driver_collector.py::test_the_production_adapter_runs_each_command_once_per_inspection`; pipeline tests assert the exact command list | — |
| K-ORDER | **Order**: `nvml` → `driver` → `gpu_hardware` → `runtime` (shared session) → close → `nvidia_smi`; driver section from `nvidia_smi`; limitation text updated | `TESTED` | `core/inspector.py` | `integration/test_inspector.py` (93): whole system/GPU/driver/runtime parity for all 24 fixtures; runtime and `nvidia_smi` crashes contained; process opt-out; NVML closed on every path | — |
| KI34-FIX | **Device-loss lifecycle** in `FakeNvml` and `StubPynvml` (`lost_after_handles`, `lost_on`); enumeration total from the inventory; **FX10 fault input corrected** (`nvml_lost_after_handles={1: 1}`, expected snapshot unchanged) | `TESTED` | `tests/gsi_testkit/fakes/{nvml,pynvml_stub}.py`, `fixtures/{scenario,catalog_core}.py` | `unit/test_device_loss.py` (10): legacy-fault contradiction kept as evidence; one, several, all GPUs lost; lost before enumeration, during driver and during runtime observations; PCI kept, unique records, deterministic; production adapter reproduces FX10 | — |
| KI16-FLOOR | **`nvidia-ml-py` floor 12.535.108 checked** in a throwaway venv | `TESTED` (binding level) | `unit/test_nvml_adapter.py` | Floor has all 28 required names; lacks only `nvmlDeviceGetTemperatureV` (fallback); 212 NVML/driver/runtime/device-loss/pipeline tests pass with it; new binding-surface test | No live driver (HW-01). |
| K-HOST | **Live on this WSL2 host** (no NVIDIA): `api.inspect()` and the demo | `TESTED` (this host only) | `tests/integration/test_inspect_host.py` (5), `examples/inspect_demo.py` (driver and runtime lines added) | 18 `host` passed; demo exit 0: seven runs `COMPLETED`, driver fields `NOT_DETECTED` (NVML library not found, `nvidia-smi` not found), binding collected, no runtime records; same from the installed wheel | Not NVIDIA validation of any kind. |
| K-DOC | **Documentation**: DEC-46; architecture §3, §4.2, §4.4, §4.5.5, §4.11, §9; data flow §1, §1.2, §4; data model §4.3, §6.1; testing strategy §3.1, §4, §5 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| K-QG | **Quality gates** | `TESTED` | — | 2106 passed (18 `host`); coverage 97.37% (`collectors/driver.py`, `collectors/runtime.py`, `core/inspector.py` 100%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (140 files), `uv lock --check` clean; isolated wheel (57 entries, both collectors shipped, no test kit or examples) 2079 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

### 1.10 CUDA environment — OQ-10 (DEC-47), IMP-A06, IMP-K05 (2026-11-11)

OQ-10 resolved by DEC-47 before any code (architecture §4.6). No new dependency, no model
or schema change (`uv.lock` unchanged). **CUDA Toolkit discovery is not CUDA execution
validation:** driver compatibility, toolkit installation, the CUDA runtime, PyTorch
availability and GPU execution are separate facts; only toolkit installation is observed
here. Suite: **2245 passed** (19 `host`); isolated wheel **2218 passed, 27 skipped** (the 27
need the specification documents, which the isolated copy doesn't include); coverage
**97.54%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| OQ10-DEC | **DEC-47**: `ToolkitLocator` protocol in the `AdapterSet`, `ToolkitFiles` seam used only by the locator; sources, platforms, semantics, schema mapping and fixture corrections | `IMPLEMENTED` (needs user review) | `docs/architecture.md` §4.6, DEC-47 | — | — |
| A06-LOCATOR | **`CudaToolkitLocator`**: `CUDA_HOME`, `CUDA_PATH`, Windows `CUDA_PATH_V*`, `nvcc` on `PATH` (resolved), Linux/WSL2 `/usr/local/cuda[-X.Y]`; resolved roots, aliases merged, all candidates; `version.json` (≤ 64 KiB) else `nvcc --version`; failures keep their status | `TESTED` (synthetic; live on this host: nothing found) | `adapters/toolkit.py`, `adapters/protocols.py` | `unit/test_toolkit_locator.py` (47): every case of the decision (§ testing strategy 3.1), the production seam on `tmp_path` | No real toolkit or `nvcc` ever seen (HW-05); Windows only through fakes (KI-27); no Windows default location (not allowlisted). |
| K05-COLLECTOR | **`CudaEnvironmentCollector`** (run `cuda_environment`, after `nvidia_smi`): whole allowlist (+ Windows `CUDA_PATH_V*`), hints to the locator, `PARTIAL` rules, `CUDA.TOOLKIT` on macOS/`OTHER` | `TESTED` (synthetic and this host) | `collectors/cuda_env.py` | `unit/test_cuda_env_collector.py` (83): parity with all 24 fixtures (section, run, errors, records; FX10 through the guard), hints, statuses, capability, no file or process access | — |
| K28-NAMES | **`PlatformAdapter.versioned_cuda_path_names()`** (`HostApi.environ_names`): Windows only, filtered by `CUDA_PATH_VERSIONED_REGEX` before any value is read | `TESTED` (synthetic) | `adapters/platform_os.py`, `adapters/protocols.py` | Windows/Linux synthetic hosts; real `os.environ` names | Real Windows host not run (KI-27). |
| K05-WIRING | **Pipeline**: `AdapterSet.toolkit`, production factory (`options.subprocess_timeout_s`), `Inspector` after `nvidia_smi`, CUDA placeholder and its helper removed, limitation text updated | `TESTED` | `adapters/adapter_set.py`, `core/factory.py`, `core/inspector.py` | `integration/test_inspector.py` (93): whole `cuda_environment` for all 24 fixtures, NVML session unchanged, one owner per capability; `unit/test_adapter_contracts.py` (new protocol and fakes conform) | — |
| K05-FIXTURES | **Fixture corrections** (DEC-47 (7)): full allowlist in every `environment`; `DEFAULT_LOCATION` for `/usr/local/cuda-12.4`; FX12 `NVCC_ON_PATH`, `CUDA_PATH_VERSIONED`, `nvcc.exe` (its premise updated to the corrected signals); `FakeToolkitFiles`, `nvcc` in `FakeCommandRunner.from_snapshot`, `FailingToolkitLocator` for FX10 | `TESTED` | `tests/gsi_testkit/{builders.py, fixtures/catalog_platform.py, fakes/*}` | Catalogue, premise, contract, schema and parity suites pass | Each correction is a contradiction with the specification, recorded in DEC-47. |
| K05-HOST | **Live on this WSL2 host**: `api.inspect()`, demo, installed wheel | `TESTED` (this host only) | `tests/integration/test_inspect_host.py` (6), `examples/inspect_demo.py` (CUDA line) | `toolkits` `NOT_DETECTED`/`NO_TOOLKIT_FOUND` (no `CUDA_HOME`/`CUDA_PATH`, no `nvcc`, no `cuda*` under `/usr/local`), all 11 variables, eight runs `COMPLETED`, no `CUDA.TOOLKIT`; only `nvidia-smi`/`nvcc` may start (neither present) | Not toolkit or NVIDIA validation (HW-05, HW-01). |
| K05-DOC | **Documentation**: DEC-47; architecture §3, §4.2, §4.5, §4.5.4, §4.6, §4.11; data flow §1, §1.1, §1.2; data model §5.1, §5.2; testing strategy §3.1, §4, §5 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| K05-QG | **Quality gates** | `TESTED` | — | 2245 passed (19 `host`); coverage 97.54% (`collectors/cuda_env.py` 100%, `adapters/toolkit.py` 99%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (145 files), `uv lock --check` clean; isolated wheel (59 entries, both modules shipped, no test kit or examples; requires only `nvidia-ml-py`; no `subprocess` at import) 2218 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

### 1.11 PyTorch — IMP-A07 probe, IMP-K06 collector (DEC-48, 2026-11-11)

Design recorded as DEC-48 before code (architecture §4.7, data flow §5). PyTorch stays
optional and undeclared (DEC-07, DEC-26): no dependency change (`uv.lock` unchanged), no
model or schema change; probe protocol 2 adds `device_error`. **What is and isn't
observed:** installation, import, build (CUDA/ROCm, build CUDA version, cuDNN), PyTorch's
own `is_available()` answer with its warnings, and the devices it enumerates. **No CUDA code
is executed** (no tensor, no kernel, no `mem_get_info`), so nothing here shows that CUDA
works; device correlation is IMP-K07. Suite: **2347 passed** (20 `host`); isolated wheel
**2320 passed, 27 skipped** (the 27 need the specification documents, which the isolated copy
doesn't include); coverage **97.18%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| DEC48 | **DEC-48**: probes, protocol 2, no execution, correlation placeholder, collector statuses, allocator rules, in-process limits | `IMPLEMENTED` (needs user review) | `docs/architecture.md` §4.7, DEC-48 | — | — |
| A07-CHILD | **Probe child and function**: `_torch_probe_main` (entry) and `torch_probe_payload` (versions, build flags, `is_available()` with warnings, count, properties; `device_error`; import-failure document) | `TESTED` (stub `torch`, in-process and as a real child) | `adapters/_torch_probe_main.py`, `adapters/torch_probe_payload.py`, `adapters/probe_protocol.py` | `unit/test_torch_probe.py` (27), `unit/test_adapter_outcomes.py` (35: protocol 2 decoding) | Never run against a real PyTorch (HW-04). |
| A07-PROBES | **`SubprocessTorchProbe`** (shared runner, `sys.executable`, timeout, duration) and **`InProcessTorchProbe`** (same function; allocator only with CUDA already initialised) | `TESTED` | `adapters/torch_probe.py` | Same file: argv, timeout, every failure code, distribution mapping, laziness (no torch, no `subprocess`, no child module at import), clean child stderr | In-process mode can't enforce a timeout and may initialise CUDA in the caller (DEC-48 (7)). |
| K06-COLLECTOR | **`FrameworkCollector`** (run `pytorch`, last): `SKIPPED`/`NOT_INSTALLED`/`COLLECTION_FAILED`/`COLLECTED` rules, separate facts, enumeration failure keeps the build facts, correlation placeholder, allocator rules | `TESTED` | `collectors/framework.py` | `unit/test_framework_collector.py` (66): parity with all 24 fixtures, AC03, AC15, AC16, AC18, AC59, every failure path, guard containment | `correlated_device_ref` waits for IMP-K07. |
| K06-WIRING | **Pipeline**: factory picks the probe from `pytorch_probe_mode`; `Inspector` runs `pytorch` after `cuda_environment`, outside the NVML session; PyTorch placeholder removed; limitation and `PRODUCTION_LIMITATIONS` (KI-11) updated | `TESTED` | `core/factory.py`, `core/inspector.py` | `integration/test_inspector.py` (93): whole `frameworks.pytorch` for every fixture; `unit/test_core_guard.py` (factory) | — |
| K06-FIXTURES | **Test kit**: `FakeAdapterSet.torch_probe` is the production subprocess probe over the fake runner (`FakeTorchProbe.canned`, `distribution_lookup`); `torch_fake` keeps the fixture facts; `StubTorch` and `write_stub_torch_package` | `TESTED` | `tests/gsi_testkit/fakes/{torch_probe,torch_stub,__init__}.py` | Contract and fake suites | No fixture expectation changed; parity replaces only `correlated_device_ref` with the K07 placeholder (DEC-48 (4)). |
| K06-HOST | **Live on this WSL2 host**: `api.inspect()`, demo, installed wheel | `TESTED` (this host only) | `tests/integration/test_inspect_host.py` (7), `examples/inspect_demo.py` (PyTorch line) | `NOT_INSTALLED` (no `torch` distribution), no probe process, `torch` never imported, nine runs `COMPLETED`; the wheel's child prints `import_ok: false` with an empty stderr | Not PyTorch or GPU validation (HW-04, HW-06). |
| K06-DOC | **Documentation**: DEC-48; architecture §3, §4.2, §4.5, §4.5.4, §4.7, §4.11; data flow §1, §1.2, §5; data model §5.3, §5.4; testing strategy §3.1, §4, §5 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| K06-QG | **Quality gates** | `TESTED` | — | 2347 passed (20 `host`); coverage 97.18% (the four new modules 100%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (152 files), `uv lock --check` clean; isolated wheel (63 entries, probe modules shipped, no test kit or examples; requires only `nvidia-ml-py`) 2320 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

### 1.12 Device correlation — IMP-K07 (DEC-09 applied, DEC-49, 2026-11-11)

DEC-09 implemented as written (UUID, then PCI bus ID, never by index); DEC-49 records what it
left open: how NVML's and PyTorch's PCI formats compare, and what an ambiguous or
contradictory match reports. No dependency, model or schema change (`uv.lock` unchanged).
The collector reads the GPU and PyTorch sections already collected: **no NVML session, no
probe, no process**. Suite: **2410 passed** (20 `host`); isolated wheel **2383 passed, 27
skipped** (the 27 need the specification documents, which the isolated copy doesn't
include); coverage **97.24%**.

| ID | Item | Status | Files | Test evidence | Known limitations |
| -- | ---- | ------ | ----- | ------------- | ----------------- |
| DEC49 | **DEC-49**: inputs, identifier rules, order, outcome codes, K06 interim value | `IMPLEMENTED` (needs user review) | `docs/architecture.md` §4.7, DEC-49 | — | UUID normalisation is still `[ASM]` (data model §5.4). |
| K07-MATCH | **`CorrelationCollector`** (run `correlation`, last): UUID among usable UUIDs, then PCI bus ID among devices a UUID can't rule out; one-to-one; contradictions noted; `NOT_APPLICABLE` without an NVML inventory; `NO_MATCH`, `NO_IDENTIFIER`, `AMBIGUOUS_MATCH`, `UNIDENTIFIED_CANDIDATES`; no capability records | `TESTED` (synthetic) | `collectors/correlation.py` | `unit/test_correlation_collector.py` (62): parity with all 24 fixtures, FX06 and FX04a through the `Inspector`, every rule and outcome, order independence, guard fallback | No fixture exercises the PCI fallback or ambiguity (unit-tested only); real multi-GPU ordering is HW-09. |
| K07-WIRING | **Pipeline**: `Inspector` runs `correlation` after `pytorch`; its section is the report's PyTorch section; the K06 interim `correlated_device_ref` is `NOT_APPLICABLE` "Set by the correlation run (DEC-09)"; limitation text updated | `TESTED` | `core/inspector.py`, `collectors/framework.py` | `integration/test_inspector.py` (93): whole PyTorch section for every fixture, exactly; correlation crash contained | — |
| K07-HOST | **Live on this WSL2 host**: `api.inspect()`, demo, installed wheel | `TESTED` (this host only) | `examples/inspect_demo.py` | Ten runs `COMPLETED`; no PyTorch devices and no NVML inventory, so nothing to correlate | Not hardware validation (HW-09). |
| K07-DOC | **Documentation**: DEC-49, DEC-09 applied; architecture §3, §4.2, §4.7, §4.11; data flow §1, §1.2, §2; data model §5.4; testing strategy §4, §5 | `IMPLEMENTED` (needs user review) | `docs/*.md` | — | — |
| K07-QG | **Quality gates** | `TESTED` | — | 2410 passed (20 `host`); coverage 97.24% (`collectors/correlation.py` 99%); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .` (154 files), `uv lock --check` clean; isolated wheel (correlation shipped; requires only `nvidia-ml-py`; importing it loads no torch, `pynvml` or `subprocess`) 2383 passed + 27 skipped, `api.inspect()` OK from the wheel | — |

## 2. Requirements tracker

All requirements are specified ([requirements.md](requirements.md)); none is implemented.

| ID | Summary | Status |
| -- | ------- | ------ |
| FR-01 | General inspection runs everywhere | `PLANNED` |
| FR-02 | System information | `PLANNED` |
| FR-03 | NVML device inventory | `IN_PROGRESS` (GPU hardware collector and production NVML adapter `TESTED` synthetically, §1.7, §1.8; hardware validation HW-01 `BLOCKED`) |
| FR-04 | Linux PCI presence check | `IN_PROGRESS` (adapter done in Stage 6.1; eligibility codes and capability policy in Stage 6.2; PCI collector in Stage 6.3; collector-level parity for all fixtures in Stage 6.4; WSL2 host gate in Stage 6.5. Pending: an inspection that runs the collector (IMP-K08, IMP-K02) and native-Linux validation, HW-02 `BLOCKED`) |
| FR-05 | Driver / NVML / driver-supported CUDA | `PLANNED` |
| FR-06 | `nvidia-smi` detection and fallback | `IN_PROGRESS` (detection, header versions, the fallback inventory, the `DriverInfo.nvidia_smi_*` fields and the runtime fallback `TESTED` synthetically, §1.8, §1.9; HW-01 `BLOCKED`) |
| FR-07 | CUDA Toolkit detection | `IN_PROGRESS` (`CUDA_HOME`, `CUDA_PATH`, Windows `CUDA_PATH_V*`, `nvcc` on `PATH`, Linux/WSL2 `/usr/local` defaults; root, version and version source per candidate; `NOT_DETECTED` when none: `TESTED` synthetically and on this WSL2 host, §1.10, DEC-47; no Windows default location; HW-05 pending) |
| FR-08 | Environment variable allowlist | `IN_PROGRESS` (all 11 names plus Windows `CUDA_PATH_V*` in `cuda_environment.environment`, unset `NOT_DETECTED`, `TESTED` synthetically and on this host, §1.10; redaction pending, IMP-P04) |
| FR-09 | PyTorch inspection and correlation | `IN_PROGRESS` (inspection §1.11 and correlation §1.12 `TESTED` synthetically and on this host: not installed vs installed-but-broken, versions, build flags, `is_available()` with warnings, devices, and each device related to NVML by UUID then PCI bus ID; never on real hardware, HW-04, HW-09) |
| FR-10 | Allocator stats only where meaningful | `IN_PROGRESS` (subprocess `NOT_APPLICABLE` with the DEC-08 note; in-process collected only with CUDA already initialised, stubbed, §1.11; never on real CUDA, HW-04) |
| FR-11 | Runtime snapshot | `IN_PROGRESS` (utilization, memory used/free/total/reserved, temperature, slowdown threshold, power draw and limit, and compute processes with PID, type, used memory and name: `TESTED` synthetically for NVML and the `nvidia-smi` fallback, §1.9; **graphics processes aren't collected (KI-36)**, so the requirement isn't met; HW-03 `BLOCKED`) |
| FR-12 | Provenance and status on every value | `IN_PROGRESS` (every collector emits `Observed` values whose constructor enforces the status vocabulary and `value` ⇔ `COLLECTED`; AC13 `TESTED`, reconciled 2026-11-11; AC14's one-case-per-row mapping test not yet reconciled) |
| FR-13 | Collector isolation | `IN_PROGRESS` (all ten runs go through the one guard; contained failures `TESTED` for system, PCI, NVML, GPU hardware, runtime, `nvidia-smi`, CUDA environment, PyTorch and correlation, and per-device isolation in GPU hardware and runtime, §1.7–§1.12; AC54 `TESTED`; AC08 waits for the CLI exit code, IMP-C01) |
| FR-14 | Diagnostics D01–D08 | `PLANNED` |
| FR-15 | Free-VRAM readiness | `PLANNED` |
| FR-16 | Text report | `PLANNED` |
| FR-17 | JSON report + schema | `IN_PROGRESS` (assembled reports validate and round-trip, §1.7; `--json` output in the CLI pending) |
| FR-18 | `--json PATH` (atomic) / `--json -` | `PLANNED` |
| FR-19 | Redaction | `PLANNED` |
| FR-20 | `version`, `--help` | `IN_PROGRESS` (bootstrap implemented and tested) |
| FR-21 | Exit codes | `PLANNED` |
| FR-22 | `--no-pytorch`, `--no-processes` | `IN_PROGRESS` (`include_pytorch`/`include_processes` honoured by the collectors, §1.9, §1.11; CLI flags need IMP-C01) |
| FR-23 | Provisional Python API | `IN_PROGRESS` (`api.inspect(options)` with `pytorch_probe_mode = IN_PROCESS` selects the in-process probe, §1.11; API stability and docs pending, IMP-C06) |
| FR-24 | Unsupported capabilities & limitations | `PLANNED` |
| FR-25 | MIG detection | `PLANNED` |
| NFR-01 … NFR-17 | Cross-platform, degradation, footprint and dependencies, typing, stable JSON, errors, logging, privacy, reproducibility, testability, performance, read-only, extensibility, maintainability, offline, timeouts, documentation | `PLANNED` (all 17) |

## 3. Architecture tracker

| ID | Item | Status | Notes |
| -- | ---- | ------ | ----- |
| ARC-01 | Layered architecture and dependency direction | `PLANNED` | [architecture.md](architecture.md) §2 |
| ARC-02 | Module layout | `PLANNED` | §3; not created (no source files yet, by instruction) |
| ARC-03 | Component interfaces (10 components) | `PLANNED` | §4 |
| ARC-04 | Data flow and ordering | `PLANNED` | [data_flow.md](data_flow.md) |
| ARC-05 | Project configuration (real package) | `TESTED` | Applied in Stage 1 ([architecture.md](architecture.md) §9); metadata tests and isolated install pass. `nvidia-ml-py` floor provisional (OQ-07). |
| ARC-06 | Metadata-only `pyproject.toml` (spec phase) | Superseded | Replaced by ARC-05 on 2026-11-04 (DEC-01 superseded) |
| ARC-07 | Future reuse candidates recorded | `PLANNED` | §8; not added to `docs/component-reuse-map.md` until code exists |

## 4. Implementation tracker

### 4.1 Data models

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-M01 | Enums, `Observed[T]`, `ErrorDetail`, invariant enforcement | `TESTED` (Stage 2) |
| IMP-M02 | `SystemInfo`, `GPUInventory`, `GPUDeviceInfo`, `PciObservation`, `DriverInfo` | `TESTED` (Stage 2) |
| IMP-M03 | `CUDAEnvironmentInfo`, `FrameworkInfo`, `PyTorchInfo`, `TorchDeviceInfo`, `AllocatorStats` | `TESTED` (Stage 2) |
| IMP-M04 | `RuntimeMetrics`, `GpuProcess` | `TESTED` (Stage 2) |
| IMP-M05 | `DiagnosticFinding`, `Evidence`, `Recommendation` | `TESTED` (Stage 2; data structures only, no rules) |
| IMP-M06 | `ReadinessResult`, `DeviceReadiness` | `TESTED` (Stage 2; data structures only, no engine) |
| IMP-M07 | `InspectionReport`, `InspectionSnapshot`, `CollectionError`, `UnsupportedCapability`, `CollectorRun`, `CollectorOutcome`, `InspectionOptions`, `ApplicationInfo`, `RedactionInfo` | `TESTED` (Stage 2) |

### 4.2 CLI

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-C01 | Command tree, options, validation | `IN_PROGRESS`. Command tree and all `inspect` options and validation rules exist and are tested; the options have no effect yet. |
| IMP-C02 | Exit codes and error handling | `IN_PROGRESS`. `ExitCode` contract, usage errors (2), output error (4, `version`), internal error (5) and interrupt (130) handlers exist; readiness codes are unused until Stage 2+. Temporary code 6 for the placeholder. |
| IMP-C03 | Output routing, logging configuration | `IN_PROGRESS`. Atomic JSON writer exists (used by `version`); text/JSON report routing and logging configuration not implemented. |
| IMP-C04 | `version` command | `TESTED` |
| IMP-C05 | Console script `inferforge-gpu` + `python -m gpu_system_inspector` entry | `TESTED` |
| IMP-C06 | Provisional Python API (`api.inspect()`, `InspectionOptions`) | `TESTED` (§1.7, API: on this WSL2 host; Mode A only) |

### 4.3 Collectors

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-K01 | System collector (Linux, WSL2, Windows, macOS) | `TESTED` (Stage 5): Linux/WSL2 on a real host, Windows/macOS/other through fakes; macOS processor and memory deferred (KI-26). Stage 5.1: macOS processor/total memory implemented (synthetic only); `linux_environment` classification |
| IMP-K02 | GPU hardware collector (NVML + fallback; consumes the PCI section from `PciCollector` as `GPUInventory.pci_observations`, DEC-43) | `TESTED` (synthetic; §1.7 K02-*): never rescans PCI, restores the PCI section after its fallback, owns `NVIDIA.ALL`; fail-closed environment applied by the core. Live NVML/`nvidia-smi` need IMP-A01/A02 |
| IMP-K03 | Driver collector | `TESTED` (§1.9 K03-*, DEC-46: `DriverCollector` + `NvidiaSmiCollector`, synthetic parity with all 24 fixtures; live on this host without a driver). Not `HARDWARE_VALIDATED` (HW-01) |
| IMP-K04 | Runtime metrics collector | `TESTED` (§1.9 K04-*, DEC-46: `RuntimeCollector`, synthetic parity with all 24 fixtures, device loss, process opt-out). Not `HARDWARE_VALIDATED` (HW-01) |
| IMP-K05 | CUDA environment collector | `TESTED` (§1.10 K05-*, DEC-47: `CudaEnvironmentCollector`, synthetic parity with all 24 fixtures; live on this WSL2 host: no toolkit). Not `HARDWARE_VALIDATED` (HW-05) |
| IMP-K06 | Framework (PyTorch) collector | `TESTED` (§1.11 K06-*, DEC-48: `FrameworkCollector`, synthetic parity with all 24 fixtures, correlation left to IMP-K07; live on this host: PyTorch not installed). Not `HARDWARE_VALIDATED` (HW-04, HW-06) |
| IMP-K07 | Device correlation | `TESTED` (§1.12 K07-*, DEC-09/DEC-49: `CorrelationCollector`, synthetic parity with all 24 fixtures incl. FX06 and FX04a; live on this host: no devices to correlate). Not `HARDWARE_VALIDATED` (HW-09) |
| IMP-K08 | Core: options, context, guard, ordering | `TESTED` (§1.7 K08-*; DEC-44): one `Collector` contract, single guard owning `FAILED`, `COLLECTOR_ORDER`, fail-closed environment before the PCI scanner is built, NVML session owned by the core. No `CollectionContext` (constructor injection) |
| IMP-K09 | PCI collector (`PciCollector`, DEC-43) | `TESTED` (Stages 6.3–6.5: unit tests, parity with all 24 fixture variants, WSL2 host run; §1.6.3–§1.6.5). Not `HARDWARE_VALIDATED` (HW-02 `BLOCKED`) |

### 4.4 Adapters

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-A00 | Adapter interfaces, outcomes, lifecycle contracts, `AdapterSet` (Stage 4) | `TESTED` with fakes (§1.4) |
| IMP-A01 | `NvmlAdapter` (read-only, error mapping) | `TESTED` (§1.8 A01-NVML, DEC-45: `PynvmlAdapter` over a stub binding; live on this host without a driver). Not `HARDWARE_VALIDATED` (HW-01) |
| IMP-A02 | `NvidiaSmiAdapter` | `TESTED` (§1.8 A02-SMI, DEC-45: `NvidiaSmiCli` over the fake runner and a fake executable). Not `HARDWARE_VALIDATED` (HW-01) |
| IMP-A03 | `PciSysfsAdapter` | `TESTED` (Stage 6.1, synthetic; real sysfs untested, HW-02) |
| IMP-A04 | `PlatformAdapter` | `TESTED` (Stage 5: `OsPlatformAdapter`; real Windows calls not yet run, KI-27) |
| IMP-A05 | `CommandRunner` | `TESTED` (§1.8 A05-RUNNER, DEC-45: `SubprocessCommandRunner` with real child processes; POSIX only so far) |
| IMP-A06 | `ToolkitLocator` | `TESTED` (§1.10 A06-*, DEC-47: `CudaToolkitLocator` over `ToolkitFiles` and the shared runner; synthetic and live on this host). Not `HARDWARE_VALIDATED` (HW-05) |
| IMP-A07 | `TorchProbe` (subprocess + in-process) and probe entry module | `TESTED` (§1.11 A07-*, DEC-48: `SubprocessTorchProbe`, `InProcessTorchProbe`, `_torch_probe_main`, protocol 2; the child run for real with a stub `torch`). Not `HARDWARE_VALIDATED` (HW-04) |
| IMP-A08 | `SystemClock` | `TESTED` (Stage 4) |

### 4.5 Diagnostics

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-D00 | Engine, rule guard, evidence helpers | `PLANNED` |
| IMP-D01 | D01 PyTorch can't use CUDA | `PLANNED`. Must handle KI-32 (a `PARTIAL` PCI scan isn't proof of absence, row 7) and KI-33 (PCI `NOT_DETECTED`) |
| IMP-D02 | D02 Free GPU memory vs requirement | `PLANNED` |
| IMP-D03 | D03 CUDA version landscape | `PLANNED` |
| IMP-D04 | D04 NVIDIA GPU availability | `PLANNED`. Must handle KI-32 (row 7 on a `PARTIAL` scan) and KI-33 (PCI `NOT_DETECTED` isn't covered by rows 7–8) |
| IMP-D05 | D05 Dependency availability | `PLANNED` |
| IMP-D06 | D06 Runtime metrics availability | `PLANNED` |
| IMP-D07 | D07 Temperature observation | `PLANNED` |
| IMP-D08 | D08 CUDA device visibility and ordering | `PLANNED` |

### 4.6 Readiness

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-R01 | Readiness engine (device + overall rules) | `PLANNED`. `NO_NVIDIA_DEVICE` must not be concluded from a `PARTIAL` PCI scan (KI-32); under the architecture §4.9 rule a PCI `NOT_DETECTED` already falls to `DEVICE_PRESENCE_UNKNOWN` (conservative); confirm it when implementing (KI-33) |

### 4.7 Reporting

| ID | Item | Status |
| -- | ---- | ------ |
| IMP-P01 | Report assembly (limitations, unsupported capabilities) | `TESTED` (§1.7 P01-*): `assemble_report`; one owner per capability (`SYSTEM.*` system, `GPU.PCI_SCAN` PCI, `NVIDIA.ALL` GPU hardware), duplicates rejected; placeholder sections for collectors not yet written |
| IMP-P02 | JSON serialiser + shipped JSON Schema | `TESTED` (Stage 2; schema `1.0` still a draft until AC09 is `TESTED` over the FX fixtures) |
| IMP-P03 | Text renderer | `PLANNED` |
| IMP-P04 | Redaction (incl. logging filter) | `PLANNED`. Includes PCI addresses (class D) in `PciObservation.address`, `PciDeviceFailure.address` and `pci` `CollectionError.message` (S61-FAIL, S63-DEVICE) |
| IMP-P05 | Atomic writer | `PLANNED` |

### 4.8 Demonstration

| ID | Item | Status | Notes |
| -- | ---- | ------ | ----- |
| DEMO-01 | Live run on the WSL2 dev machine (no-GPU path) recorded in README | `NOT_STARTED` | After implementation; real `LIVE` output |
| DEMO-02 | Live run on an NVIDIA machine (Mode A + Mode B) | `BLOCKED` | Needs hardware (§13) |
| DEMO-03 | Illustrative report excerpts in docs | `NOT_STARTED` | Documentation only, never a product mode (DEC-17). Must be labelled "illustrative, not a real measurement" and carry `data_origin = FIXTURE`. |

### 4.9 Final review

| ID | Item | Status |
| -- | ---- | ------ |
| REV-01 | Spec ↔ implementation conformance review | `NOT_STARTED` |
| REV-02 | Acceptance checklist complete with evidence | `NOT_STARTED` |
| REV-03 | Workspace checks (`check_dependencies`, ruff, types, coverage) | `NOT_STARTED` |
| REV-04 | User sign-off | `NOT_STARTED` |

## 5. Testing tracker

| ID | Item | Status |
| -- | ---- | ------ |
| TST-01 | Test infrastructure (project pytest config, markers, fakes, `ScenarioBuilder`, `FixedClock`) | `TESTED` (Stages 3–4). Pytest config, `hardware` marker, `gsi_testkit` (builders, fixtures, expectations, fakes, `FixedClock`; `FakeAdapterSet` replaces `ScenarioBuilder`): S3-FAKE. Stage 4: `FakeCommandRunner` added (S4-CMD); fakes conform to the protocols (S4-CONF). |
| TST-00 | Stage 1 bootstrap tests (imports, metadata, entry point, help, version, placeholder) | `TESTED`. 47 passed in workspace and in isolated venv (CPython 3.12.12). |
| TST-02 | Fixtures FX01–FX20 | `TESTED` (Stage 3, synthetic): 24 variants, 30 report cases (S3-FIX, S3-DIAG, S3-READY) |
| TST-03 | Unit: models | `TESTED` (Stage 2) |
| TST-04 | Unit: adapters | `IN_PROGRESS`: contract, outcome, lifecycle and runner tests `TESTED` on fakes (S4-*); production platform adapter `TESTED` (S5-UNIT); production command runner, NVML and `nvidia-smi` adapters `TESTED` (§1.8); unit tests of the PyTorch-probe adapter pending; PCI sysfs adapter and reader primitives `TESTED` (S61-*); PCI eligibility policy and fixture parity `TESTED` (S62-*); PCI adapter gate on this WSL2 host `TESTED` (S65-PCI) |
| TST-05 | Unit: collectors | `IN_PROGRESS`: system collector and environment collection `TESTED` (S5-UNIT); PCI collector `TESTED` (S63-*), with collector-level parity for all 24 fixture variants (S64-*) and a WSL2 host run (S65-PCI); GPU hardware collector `TESTED` (§1.7); other collectors pending |
| TST-06 | Unit: diagnostics (all decision-table rows) | `PLANNED` |
| TST-07 | Unit: readiness | `PLANNED` |
| TST-08 | Unit: reporting, redaction, serialisation | `IN_PROGRESS` (serialisation and schema `TESTED`; text report and redaction not implemented) |
| TST-09 | Unit: CLI parsing and exit codes | `PLANNED` |
| TST-10 | Integration: full inspection per fixture | `IN_PROGRESS`: `Inspector.run()` over all 24 fixtures for the implemented collectors, plus 12 failure scenarios (`integration/test_inspector.py`, §1.7); the full findings matrix needs diagnostics and the remaining collectors |
| TST-11 | Integration: CLI end-to-end | `PLANNED` |
| TST-12 | Integration: JSON Schema validation and determinism | `TESTED` for all 30 synthetic fixture report cases (S3-SCHEMA, S3-RT). Reports assembled by the production core over all 24 fixtures, and the live host report, also validate, round-trip and are deterministic (§1.7). |
| TST-13 | Integration: PyTorch probe protocol with stub `torch` | `PLANNED` |
| TST-14 | Invariant tests (testing strategy §7) | `IN_PROGRESS` (value/status invariant, no-zero, NaN rejection, schema validity and determinism tested on builder reports and all fixture reports; provenance invariant added; banner and redaction invariants pending) |
| TST-15 | Coverage and quality gates | `IN_PROGRESS`: coverage measured and enforced (S3-COV; 96.63% ≥ 92% after the integration milestone, K-QG); ruff, `check_dependencies` run every stage. Type checking not configured (OQ-08). |

## 6. Hardware validation tracker

Results are recorded here with date, machine, OS, driver, GPU, PyTorch version and outcome.
**No hardware validation has been performed.**

| ID | Test | Status | Notes |
| -- | ---- | ------ | ----- |
| HW-01 | NVML vs `nvidia-smi` | `BLOCKED` | No NVIDIA GPU available (OBS-01). The production adapters exist (§1.8) and ran on this WSL2 host, where NVML's library and `nvidia-smi` are absent; that isn't hardware validation. |
| HW-02 | PCI scan on bare-metal Linux | `BLOCKED` | Needs bare-metal Linux + NVIDIA. Stage 6.5 ran the PCI host test on this WSL2 machine, where the scan is correctly gated; that isn't native PCI validation (S65-HW). |
| HW-03 | Runtime snapshot sanity | `BLOCKED` | |
| HW-04 | PyTorch probe and isolation (C10/C11) | `BLOCKED` | |
| HW-05 | Toolkit detection (C12) | `BLOCKED` | No toolkit installed on dev machine |
| HW-06 | PyTorch without local toolkit | `BLOCKED` | |
| HW-07 | WSL2 with NVIDIA driver | `BLOCKED` | Dev machine is WSL2 but has no NVIDIA driver exposed |
| HW-08 | Windows WDDM | `BLOCKED` | |
| HW-09 | Multi-GPU ordering | `BLOCKED` | |
| HW-10 | Performance measurements | `PLANNED` | Partially possible on dev machine (no-GPU path) after implementation |
| HW-11 | macOS | `BLOCKED` | No macOS machine identified |
| HW-12 | Readiness on real memory pressure | `BLOCKED` | |
| HW-13 | No-GPU real-machine run | `PLANNED` | Can run on the WSL2 dev machine once implemented |

## 7. Documentation tracker

| ID | Document | Status | Notes |
| -- | -------- | ------ | ----- |
`USER_VERIFIED` here means the user approved the specification ("approved for
implementation", 2026-11-04). It applies to the document content at approval time. The
Stage 1 amendments listed in the last column were made afterwards and haven't been
reviewed yet.

| ID | Document | Status | Notes / Stage 1 amendments (unreviewed) |
| -- | -------- | ------ | --------------------------------------- |
| DOC-01 | [problem_statement.md](problem_statement.md) | `USER_VERIFIED` | — |
| DOC-02 | [product_specification.md](product_specification.md) | `USER_VERIFIED` | — |
| DOC-03 | [requirements.md](requirements.md) | `USER_VERIFIED` | NFR-01/CON-05 now state the applied `>=3.11` / `py311` floor. |
| DOC-04 | [architecture.md](architecture.md) | `USER_VERIFIED` | DEC-01 superseded; DEC-21 applied; §9 rewritten as the applied configuration; whitespace-only reformat of one code example by `ruff format` (OBS-08). Stage 2: §3 module layout lists `collection.py`, `_validation.py`, `schema.py`, `reporting/__main__.py`. Stage 4: §2 layer roles; §3 adapters layout; §4.5 rewritten (interface catalogue, outcomes, lifecycle, construction, planned implementations); §4.7 `TorchProbe` signature; §5 time; DEC-28–DEC-34. Stage 5: §4.3 rewritten (implemented system collector, platform-mechanism table, memory semantics, WSL detection, provenance); §3 layout; §4.5 production adapters; DEC-35–DEC-38. Stage 5.1: §4.3 macOS `sysctlbyname` and Linux-environment classification; DEC-37 marked superseded; DEC-39, DEC-40. Stage 6.1: §3, §4.5, §4.5.1, §4.5.5 (`PciSysfsAdapter`, fail-closed gate), DEC-22 approved, DEC-41. Stage 6.2: §4.5.1, §4.5.4 (no profile-only PCI selection), §4.5.5, DEC-42. Stage 6.3: §3 layout (`collectors/pci.py`), §4.2 (`Collector` protocol marked aspirational), §4.4 (PCI ownership, mapping table, interface), §4.5.5 cross-reference, DEC-43. Stage 6.6: §4.5.5 heading, DEC table order, §4.11 integration readiness (IMP-K08, IMP-K02). Integration milestone: §3 layout, §4.2 (implemented contract, guard, order), §4.4 (GPU hardware collector), §4.5.4 (production factory), §4.5.5 (stand-ins), §4.10 (assembly), §4.11 (delivery table, follow-ups), DEC-44. NVIDIA adapters: §3, §4.4 (NVML/`nvidia-smi` policy table), §4.5 intro, §4.5.4 (factory), §4.5.5 (runner, NVML, `nvidia-smi`, stand-ins rows), §4.11, DEC-45. **Needs review.** |
| DOC-05 | [data_flow.md](data_flow.md) | `USER_VERIFIED` | Stage 4: §3 outcome contract and `AdapterUsageError`; §4 lifecycle rows (failed init, single session, handles, close semantics); §5 probe protocol v1 types and decoding. Stage 5: §1.1 step ① (system collection and environment) in detail. Stage 5.1: §1.1 classification and profile derivation. Stage 6.2: §1 step ② native-Linux eligibility. Stage 6.3: §3 run-status kinds (field-level, omitted device, whole capability) and the `ErrorDetail` fallback argument. Stage 6.6: §1 step ② names `PciCollector` and the IMP-K02 merge. Integration milestone: §1 (core implemented, ③ fallback), new §1.2 (what runs today), §3 guard as implemented. NVIDIA adapters: §1.2 (real NVML/`nvidia-smi`), §4 (production adapter evidence). **Needs review.** |
| DOC-06 | [existing_tools_analysis.md](existing_tools_analysis.md) | `USER_VERIFIED` | — |
| DOC-07 | [data_model_specification.md](data_model_specification.md) | `USER_VERIFIED` | Python-floor sentence updated to 3.11. Stage 2: new §10 "Implementation clarifications" (12 items, including the `NOT_APPLICABLE`/`error` inconsistency resolution). Stage 3: §3.2 `error` row rewritten with an explanation of why `NOT_APPLICABLE` needs a note but not an error; §1 natural device ordering; §10 #1, #4, #9 updated. Stage 4: §3.3 row "any other `OSError` → `COLLECTION_FAILED`" plus a note that `adapters/outcomes.py` implements the table. Stage 5: §4.1 sources updated (Windows CPU from the registry instead of `platform.processor()`; macOS processor/total memory deferred; `is_wsl` without `WSL_DISTRO_NAME`; memory meaning notes). Stage 5.1: macOS rows restored to the approved requirement (`sysctlbyname`); `is_wsl` replaced by `linux_environment` with a normative classification rule; `platform_profile` note; §7.3 draft-change log. Stage 6.1: `NO_DRIVER_BOUND` (§3.2, §4.2). Stage 6.2: `LINUX_ENVIRONMENT_UNKNOWN` (§3.2, §3.3), `GPU.PCI_SCAN`/`NVIDIA.ALL` policy (§3.4), §4.2 eligibility wording, §7.5 example wording. Stage 6.6: §4.2 `pci_observations` (missing root, omitted devices → `pci` errors). Integration milestone: §3.2 `FALLBACK_FIELD_UNAVAILABLE` and `NOT_IMPLEMENTED`; §9 guard fallback keeps PCI, placeholder sections. NVIDIA adapters: §3.2 code list, §3.3 `nvidia-smi` rows (`[Unknown Error]`/`[GPU is lost]`, unknown query field). **Needs review.** |
| DOC-08 | [cli_specification.md](cli_specification.md) | `USER_VERIFIED` | Temporary exit code 6 `NOT_IMPLEMENTED` (development only); argparse message format note; `version --json PATH` messages; status line. |
| DOC-09 | [diagnostics_specification.md](diagnostics_specification.md) | `USER_VERIFIED` | Stage 3: §2 severity table corrected (`INFO` may be `LOW`, matching D08) and made normative as a status/severity table; D05 severity note. **Needs review.** |
| DOC-10 | [testing_strategy.md](testing_strategy.md) | `USER_VERIFIED` | Stage 3: §2 coverage command, test kit layout and provenance rule; §4 implementation note (FX14 assumption); §5 fakes as implemented; §8 coverage baseline and target. Stage 4: §5 rewritten (fake ↔ protocol table, conformance tests, what a fake can't prove); §8 Stage 4 coverage. Stage 5: §2 `host` marker; §5 `FakeHostApi` and the system inspection test table; §8 Stage 5 coverage. Stage 5.1: §5 WSL matrix, parity guard, simulated `sysctlbyname`; §8 Stage 5.1 coverage. Stage 6.2: §3.1 PCI row (dict fake, WSL1/UNKNOWN, policy tests), §5 PCI gating wording, §6 HW-02 precondition. Stage 6.6: §5 `pci` fake collector, `sysfs_reader_for`, PCI collector/parity/host test modules, host note; §8 Stage 6 coverage. Integration milestone: §4 note (FX13 correction, FX10/KI-34), §5 `FakeCollector` conforming, four new test modules, demo; §8 coverage. NVIDIA adapters: §3.1 runner/NVML/`nvidia-smi` rows, §5 new test modules and host note, "what a fake can't prove" updated, §8 coverage. **Needs review.** |
| DOC-11 | [acceptance_criteria.md](acceptance_criteria.md) | `USER_VERIFIED` | — |
| DOC-12 | app_tracker.md (this file) | `IMPLEMENTED` | Living status document; not subject to approval. |
| DOC-13 | [technical_evidence.md](technical_evidence.md) | `USER_VERIFIED` | OBS-07, OBS-08 added. |
| DOC-14 | [../README.md](../README.md) | `IMPLEMENTED` | Rewritten for Stage 1 status and current development usage after the approval. Needs review. |
| DOC-15 | User-facing usage docs / examples | `NOT_STARTED` | After implementation |
| DOC-16 | `docs/component-reuse-map.md` candidates | `NOT_STARTED` | Only after code exists (package policy) |
| DOC-17 | [../../../docs/python-version-review.md](../../../docs/python-version-review.md) (workspace-level, WSD-01) | `IMPLEMENTED` | Decision approved (3.11). Status and application record (§9) added in Stage 1. |

### 7.1 Specification review status (audit 2, 2026-11-04)

| Check | Result |
| ----- | ------ |
| CLI naming (`inferforge-gpu`, `--json PATH\|-`) consistent everywhere | Pass (after corrections) |
| Dependency classes (required / optional / external) consistent | Pass (after corrections; product spec §1.1) |
| Architecture and data flow agree (ordering, routing, error flow) | Pass (output routing corrected in both) |
| All eight diagnostics specified with decision tables and fixture outcomes | Pass (D05 renamed "Dependency availability") |
| All 20 fixtures have pinned inputs and expected outcomes (Mode A and Mode B) | Pass (missing pins and a Mode B table added) |
| All 60 ACs have verification method and expected result | Pass |
| AC → requirement → component → verification mapping | Pass after adding NFR-17, IMP-C06 and broadened AC25/26/36/51 (acceptance criteria §I) |
| Error states and missing-value handling consistent | Pass (`PROBE_EXIT_<n>` naming harmonised) |
| Hardware validation separated from mocked tests | Pass |
| No unsupported performance/accuracy claims | Pass (three claims softened or sourced) |
| V1 scope and non-goals consistent | Pass (demo mode and umbrella command removed from future list) |
| ID cross-references (EV, DEC, FR/NFR, AC, FX, HW, OQ, KI, OBS) resolve | Pass (scripted check) |

## 8. Acceptance checklist

No criterion is met yet. "Target status" is the highest status the criterion needs.

| # | Criterion (short) | Methods | Current | Target |
| -- | ----------------- | ------- | ------- | ------ |
| AC01 | Runs without NVIDIA GPU | IT, HW-13, MAN | `PLANNED` | `USER_VERIFIED` |
| AC02 | Missing NVML doesn't crash | IT | `IN_PROGRESS` (missing binding, missing library and driver not loaded handled by the production NVML adapter and pipeline, incl. live on this host, §1.8; the driver section since IMP-K03, §1.9; the CUDA and PyTorch sections need IMP-K05, K06) | `TESTED` |
| AC03 | Missing PyTorch doesn't crash | IT | `IN_PROGRESS` (FX05 through the `Inspector`: `NOT_INSTALLED`, no probe process (runner call log), §1.11; exit code and D01/D05 need IMP-C01, IMP-D*) | `TESTED` |
| AC04 | Driver CUDA ≠ Toolkit label | UT, IT, REV | `IN_PROGRESS` (separate fields and enforced note; toolkit versions only from `version.json`/`nvcc`, tested for every fixture, §1.10; text label pending, IMP-P03) | `TESTED` |
| AC05 | Memory metrics keep source and meaning | UT, IT | `TESTED` (reconciled 2026-11-11: device memory only in `runtime[*]`/`gpu.devices[*].memory_total_bytes` with NVML or `NVIDIA_SMI` source and exact method, FX01 and FX04a whole-section parity through the `Inspector` (`integration/test_inspector.py`); allocator data only in `frameworks.pytorch.allocator` (`unit/test_framework_collector.py`); `memory_bus_utilization_percent` is its own field and method; the fallback's 1 MiB note, FX04a and `unit/test_nvidia_smi_adapter.py`) | `TESTED` |
| AC06 | Multiple devices independent | IT | `PLANNED` | `TESTED` (+ HW-09) |
| AC07 | Readiness vs threshold | UT, IT | `PLANNED` | `TESTED` (+ HW-12) |
| AC08 | Failed collector isolated | IT, UT | `IN_PROGRESS` (UT: guard; IT: system, PCI, NVML, GPU-hardware, runtime, `nvidia-smi`, CUDA environment (FX10's injected crash) and PyTorch failures stay contained, §1.7–§1.11; exit-code behaviour needs IMP-C01) | `TESTED` |
| AC09 | Versioned JSON schema | IT, REV | `IN_PROGRESS` (schema shipped; all 30 synthetic fixture report cases validate and round-trip; reports produced by the real pipeline over all 24 fixtures and live on this host validate too, §1.7. Not complete: the remaining sections' collectors and the freeze decision, KI-20) | `TESTED` |
| AC10 | Findings have evidence & limitations | UT, IT | `PLANNED` | `TESTED` |
| AC11 | Hardware tests separate | IT, REV | `PLANNED` | `TESTED` |
| AC12 | Mocked never presented as real | UT, IT, REV | `PLANNED` | `TESTED` |
| AC13 | No misleading zeros | UT, IT | `TESTED` (reconciled 2026-11-11: UT `unit/test_observed.py` (the constructor rejects every violation); IT: every fixture's report is built by the production pipeline (`integration/test_inspector.py`, 24 fixtures), so every value passed that check, `integration/test_partial_failure_multi_gpu.py::test_missing_runtime_metrics_serialise_as_null_not_zero`, and the adapters map `[N/A]`/`None` to statuses, never values; the post-IMP-K07 review checked all 1,764 observed values of the 24 reports, no violation) | `TESTED` |
| AC14 | Error→status mapping | UT | `PLANNED` | `TESTED` |
| AC15 | Installed-but-broken PyTorch | IT | `IN_PROGRESS` (FX15 `COLLECTION_FAILED`/`IMPORT_ERROR` with the message vs FX05 `NOT_INSTALLED`, §1.11; D01 `FAIL` needs IMP-D01) | `TESTED` |
| AC16 | Allocator stats `NOT_APPLICABLE` in CLI | UT, IT | `IN_PROGRESS` (subprocess probe `NOT_APPLICABLE` with the note; in-process with CUDA initialised (stubbed) `COLLECTED` with "Calling process only", §1.11; "via CLI" needs IMP-C01) | `TESTED` |
| AC17 | Toolkit detection | UT, HW-05 | `IN_PROGRESS` (`root_path`, `discovered_via`, version `method`; mismatched candidates all listed; none → `NOT_DETECTED`: `TESTED` synthetically, §1.10; HW-05 not run) | `HARDWARE_VALIDATED` |
| AC18 | PyTorch build CUDA labelled | UT, IT | `IN_PROGRESS` (enforced note, CPU-only `NOT_APPLICABLE`, both reproduced by the collector for FX01/FX02a, §1.11; text label needs IMP-P03) | `TESTED` |
| AC19 | UUID/PCI correlation | UT, IT, HW-09 | `IN_PROGRESS` (FX06 `cuda:0` → `nvml:1`, `cuda:1` → `nvml:0` by UUID through the `Inspector`; PCI fallback, `NOT_DETECTED` for no match, `NOT_APPLICABLE` without an NVML inventory (FX04a): `TESTED` synthetically, §1.12; HW-09 not run) | `HARDWARE_VALIDATED` |
| AC20 | NOT_READY + exit 1 | IT | `PLANNED` | `TESTED` |
| AC21 | UNKNOWN + exit 3 | IT | `PLANNED` | `TESTED` |
| AC22 | Units and boundary | UT, IT | `PLANNED` | `TESTED` |
| AC23 | Readiness disclaimer | UT, IT | `PLANNED` | `TESTED` |
| AC24 | `--required-vram` validation | UT | `PLANNED` | `TESTED` |
| AC25 | Deterministic JSON | IT | `TESTED` (reconciled 2026-11-11: `integration/test_inspector.py::test_every_report_is_valid_complete_and_deterministic` renders every fixture's production report twice, byte-identical, schema-valid and round-tripping; key order, completeness and no NaN/Infinity from `unit/test_serialization.py` and the `Observed` checks; the report records schema and application versions, `nvidia-ml-py` (`driver.nvml_binding`) and PyTorch versions when present, options, collector durations and `data_origin`) | `TESTED` |
| AC26 | stdout/stderr separation | IT | `PLANNED` | `TESTED` |
| AC27 | Atomic `--json PATH` | IT | `PLANNED` | `TESTED` |
| AC28 | Evidence classification & wording | UT, IT | `PLANNED` | `TESTED` |
| AC29 | D01 | UT, IT, HW-04 | `PLANNED` | `HARDWARE_VALIDATED` |
| AC30 | D03 | UT, IT | `PLANNED` | `TESTED` |
| AC31 | D04 | IT | `PLANNED` | `TESTED` |
| AC32 | D05, no auto-install | IT, UT | `PLANNED` | `TESTED` |
| AC33 | D06 | IT | `PLANNED` | `TESTED` |
| AC34 | D07, no throttling inference | UT, IT | `PLANNED` | `TESTED` |
| AC35 | D08 | IT, HW-09 | `PLANNED` | `HARDWARE_VALIDATED` |
| AC36 | macOS | IT, HW-11 | `PLANNED` | `HARDWARE_VALIDATED` |
| AC37 | Windows WDDM | IT, HW-08 | `PLANNED` | `HARDWARE_VALIDATED` |
| AC38 | WSL2 | IT, HW-07, MAN | `PLANNED` | `USER_VERIFIED` |
| AC39 | MIG | IT | `PLANNED` | `TESTED` |
| AC40 | Redaction | IT | `PLANNED` | `TESTED` |
| AC41 | Privacy defaults | UT, REV | `PLANNED` | `TESTED` |
| AC42 | No network | IT | `PLANNED` | `TESTED` |
| AC43 | Read-only | UT, REV | `IN_PROGRESS` (UT: the NVML adapter calls only `NVML_READ_ONLY_FUNCTIONS` (`unit/test_nvml_adapter.py`); only `adapters/commands.py` may start processes (`unit/test_boundaries.py`), never through a shell; the processes an inspection may start are `nvidia-smi` (header; `--query-gpu` only as the fallback), `nvcc --version` (the active `nvcc` and toolkits without `version.json`) and the PyTorch probe child (`sys.executable -m gpu_system_inspector.adapters._torch_probe_main`, or none in `IN_PROCESS` mode), §1.8–§1.11; the review grep for `nvmlDeviceSet`, `nvmlDeviceReset`, `nvmlDeviceClear`, `os.kill`, `signal` found nothing in `src/` (2026-11-11); REV by the owner and the `--json` writer (IMP-P05) pending) | `TESTED` |
| AC44 | Timeouts | UT, IT | `PLANNED` | `TESTED` |
| AC45 | `version` command | IT | `IN_PROGRESS` (bootstrap tests pass; criterion not signed off) | `TESTED` |
| AC46 | Help | IT | `IN_PROGRESS` (bootstrap tests pass; `inspect` help describes a placeholder) | `TESTED` |
| AC47 | Exit codes | IT | `PLANNED` | `TESTED` |
| AC48 | Internal error handling | IT | `PLANNED` | `TESTED` |
| AC49 | Performance measured | HW-10, MAN | `PLANNED` | `HARDWARE_VALIDATED` |
| AC50 | Dependency classes & footprint | REV, IT | `IN_PROGRESS` (metadata tests pass; `nvidia-ml-py` floor confirmed at the binding level, KI-16/OQ-07 resolved, §1.9) | `TESTED` |
| AC51 | Workspace, lint, typing, coverage | REV, automated | `PLANNED` | `TESTED` |
| AC52 | Matches NVIDIA tools on real HW | HW-01, HW-03, HW-04 | `PLANNED` | `HARDWARE_VALIDATED` |
| AC53 | Documentation complete & consistent | REV, MAN | `IN_PROGRESS` (specification approved; README usage for `inspect` pending implementation) | `USER_VERIFIED` |
| AC54 | Per-device failure isolation | IT | `TESTED` (reconciled 2026-11-11: FX10 through the `Inspector`: `nvml:0` runtime fully `COLLECTED`, `nvml:1` `COLLECTION_FAILED` with `NVML_ERROR_GPU_IS_LOST`, one `CollectionError` with `device_ref = nvml:1` (`integration/test_inspector.py`, `unit/test_device_loss.py`); synthetic only, real device loss is HW-01, `BLOCKED`) | `TESTED` |
| AC55 | Unsupported capabilities & limitations | IT | `PLANNED` | `TESTED` |
| AC56 | `nvidia-smi` fallback | UT, IT | `PLANNED` | `TESTED` (+ HW-01) |
| AC57 | Linux PCI check | UT, IT, HW-02 | `IN_PROGRESS` (UT: adapter, eligibility-policy and PCI collector tests pass; IT: FX03/FX17 parity at adapter, collector and full-inspection level passes (S64-*, §1.7); HW-02 `BLOCKED`) | `HARDWARE_VALIDATED` |
| AC58 | Python API | IT | `PLANNED` | `TESTED` |
| AC59 | Opt-outs | IT | `IN_PROGRESS` (`include_processes=False` → `SKIPPED` with no process NVML call, §1.9; `include_pytorch=False` → `SKIPPED`, nothing read or run, §1.11; the CLI flags need IMP-C01) | `TESTED` |
| AC60 | Verbosity | IT | `PLANNED` | `TESTED` |

## 9. Known issues and limitations

| ID | Issue / limitation | Impact | Handling |
| -- | ------------------ | ------ | -------- |
| KI-01 | The development machine has no NVIDIA GPU or driver (OBS-01). | No hardware validation possible locally. | §13 BLOCKED; HW-13 still runnable. |
| KI-02 | ~~The `inferforge` console-script name would be a shared workspace resource.~~ | — | **Resolved 2026-11-04:** dedicated `inferforge-gpu` script (DEC-03). |
| KI-03 | ~~Workspace `requires-python >=3.10` past EOL.~~ | — | **Resolved 2026-11-04:** floor raised to `>=3.11` (WSD-01, Stage 1). |
| KI-04 | No official NVIDIA documentation located for the toolkit `version.json` format (EV-20). | Toolkit detection rests on an assumption. | `nvcc --version` fallback; HW-05. |
| KI-05 | The unversioned NVML `nvmlUtilization_t` page returned HTTP 404 during research (EV-07). | Citation uses the versioned 615 page. | Re-check links before release. |
| KI-06 | PyTorch device `uuid`/`pci_*` attributes depend on the version (EV-14). | Correlation may be unavailable on older PyTorch. | Per-field `NOT_SUPPORTED`; D08 falls back to heuristics. |
| KI-07 | Assumptions C10–C14 unverified ([technical_evidence.md](technical_evidence.md) §4). | Design choices may need adjustment. | HW-02, HW-04, HW-05; unit test for C13. |
| KI-08 | `uv.lock` records Project 01 as an editable member with `nvidia-ml-py` (re-locked in Stage 1). | Root lockfile changes with this project. | Expected; recorded in §1.1. |
| KI-09 | ~~Ruff's first-party classification of `gpu_system_inspector` unverified.~~ | — | **Resolved:** added to root `known-first-party` (existing convention); `ruff check .` clean. |
| KI-10 | Product limitations: point-in-time snapshot, single host, MIG instances not enumerated, Jetson/iGPU unsupported, NVIDIA only. | Documented to users. | Product spec §4, §6. |
| KI-11 | PyTorch is inspected only in the interpreter running the inspector (DEC-07, DEC-26). | The inspector must be installed in each ML environment to inspect its PyTorch, so the Python floor limits which environments qualify. | Documented in product spec §1.1, D05 wording and the CLI help; `--python` deferred (§14). |
| KI-12 | ~~`verify_isolated_install.py` venv interpreter not deterministic.~~ | — | **Resolved in Stage 1:** `--python` defaults to the running interpreter; slow tests pass. |
| KI-13 | No Python 3.11 interpreter is installed on the development machine (OBS-04). | The `>=3.11` floor is declared but not directly tested on 3.11. | S1-PY311 `BLOCKED`; needs approval to run `uv python install 3.11`. |
| KI-14 | `scripts/verify_isolated_install.py` only accepts packages (by design, enforced by `test_projects_are_rejected`). | The isolated-install check for Project 01 had to be run manually (S1-ISO). | Possible workspace change: an explicit `--allow-project` option. Not made, because it changes repository policy; awaiting a decision. |
| KI-15 | `inspect` is a placeholder returning temporary exit code 6. | Scripts must not treat Stage 1 builds as able to inspect. | Removed when `inspect` is implemented (cli spec §6). |
| KI-16 | ~~The `nvidia-ml-py>=12.535.108` floor is provisional (OQ-07).~~ **Resolved 2026-11-11** (§1.9, DEC-46): the floor is kept. With `nvidia-ml-py==12.535.108` installed in a throwaway environment, every call the adapter needs is present (the only missing preferred call, `nvmlDeviceGetTemperatureV`, falls back to `nvmlDeviceGetTemperature`), and the NVML adapter, driver, runtime, device-loss and production pipeline suites pass (212 tests); `unit/test_nvml_adapter.py::test_the_installed_binding_has_every_call_the_adapter_needs` keeps checking it. | — | Binding-level only: behaviour against a live driver is HW-01. |
| KI-17 | ~~No coverage tool is configured.~~ **Resolved in Stage 3** (S3-COV): `pytest-cov` test-only, `fail_under = 92`. | — | — |
| KI-18 | ~~Spec inconsistency on `error` for `NOT_APPLICABLE`.~~ **Resolved in Stage 3:** data model §3.2 now says `note` required, `error` optional. | — | User review of the amendment (DOC-07). |
| KI-19 | ~~Spec inconsistency on `INFO` severity.~~ **Resolved in Stage 3:** `INFO` allows `NONE` or `LOW` (diagnostics spec §2); combinations enforced in model and schema. | — | User review of the amendment (DOC-09). |
| KI-20 | Report schema `1.0` is still provisional (data model §7.3). | It may change without a version bump. | All synthetic fixtures and expected outputs now validate (Stage 3). Freezing needs AC09 `TESTED` on real pipeline output plus user agreement. |
| KI-21 | FX14 pins "one field `NOT_SUPPORTED`"; the fixture chooses utilization. | A collector that reports a different field could still be correct. | Recorded in the fixture's limitations and testing strategy §4. |
| KI-22 | ~~Fake adapter method names have no production protocols.~~ **Resolved in Stage 4** (S4-PROTO, S4-CONF): protocols defined, fakes conform, signature drift is detected. | — | Real adapters must pass the same conformance tests when they're written. |
| KI-24 | Protocol conformance is checked at runtime (members, parameter names, kinds and defaults), not by a static type checker. | Return-type mismatches would be caught only by behavioural tests. | Decide OQ-08 (e.g. mypy as a test-only tool); not added, because it changes the dependency set. |
| KI-26 | ~~macOS processor model and total memory are `NOT_SUPPORTED` (the spec names `sysctl`, a subprocess).~~ **Resolved in Stage 5.1** (S51-MACIMPL): collected via read-only `sysctlbyname` through `ctypes`, no subprocess (DEC-40). | — | Real-Mac validation still pending (KI-30). |
| KI-30 | The macOS `sysctlbyname` path has run only against a simulated libc. | A `ctypes` or dyld issue would only show on a Mac. | Run `uv run pytest -m host` on a Mac. |
| KI-31 | Older WSL2 kernels named `4.19.x-microsoft-standard` (without `WSL2`) classify as `UNKNOWN`. | Conservative under-classification on very old WSL2 installs. | Accept, or add a documented rule if evidence (EV entry) justifies it. |
| KI-27 | Windows branches have run only against `FakeHostApi`; the real `winreg` and `GlobalMemoryStatusEx` calls have never executed. | A struct-layout or API error would only show on Windows. | Run `uv run pytest -m host` on a Windows machine (no NVIDIA hardware needed). |
| KI-28 | ~~Versioned `CUDA_PATH_V*` variables aren't read.~~ **Resolved 2026-11-11** (DEC-47): on Windows, `PlatformAdapter.versioned_cuda_path_names()` lists only names matching `CUDA_PATH_VERSIONED_REGEX` before any value is read; no other name is kept. | — | Real Windows host not run (KI-27). |
| KI-29 | ~~`/proc/version` containing `microsoft` is reported as `WSL2`, including WSL1.~~ **Resolved in Stage 5.1** (S51-WSLPOLICY): conservative `linux_environment` classification; WSL1 and ambiguous markers are no longer `WSL2` (DEC-39). | — | Native Linux and WSL1 validated synthetically only. |
| KI-25 | ~~Lifecycle and outcome mapping are proven on fakes only.~~ **Partly resolved (IMP-A01):** the production adapter is tested against a stub `pynvml` whose error codes and messages are checked against the real binding, and the real binding ran on this host (library not found). | An NVIDIA driver could still behave in ways a stub can't show. | Validate on hardware (HW-01…). |
| KI-23 | Spec-table tests read `docs/`, which isn't in the wheel. | 25 tests skip in isolated installs (with a reason). | Intended: the wheel stays docs-free; the workspace run covers them. |
| KI-32 | A `PARTIAL` PCI scan (some listed devices couldn't be observed) still has `pci_observations` `COLLECTED` (DEC-43). D04 row 7, D01 row 7 and the readiness `NO_NVIDIA_DEVICE` rule (architecture §4.9) treat a `COLLECTED` scan with no NVIDIA device as proof of absence. | An omitted device could be the NVIDIA GPU, so a "no NVIDIA GPU" conclusion could be wrong. | **Open** (reviewed in Stage 6.6, retained): IMP-D01, IMP-D04 and IMP-R01 must check the `pci` run status or its `collection_errors` before concluding absence. The PCI collector already supplies both (S63-DEVICE); nothing consumes them yet. No `note` added to the section. |
| KI-33 | A whole-scan PCI `NOT_DETECTED` (no `/sys/bus/pci/devices`, e.g. a kernel without PCI or a container without `/sys`) isn't covered by D04 rows 7–8 or D01, so it would fall to D04 row 9 (`UNKNOWN`). | A host without PCI sysfs could get an `UNKNOWN` availability finding instead of an explained limitation. | **Open** (reviewed in Stage 6.6, retained): IMP-D01 and IMP-D04 must handle `NOT_DETECTED` explicitly (it isn't a failure and isn't proof of absence); IMP-R01's current rule already gives `DEVICE_PRESENCE_UNKNOWN`. No fixture pins this case (S64-COVERAGE). |
| KI-34 | ~~FX10's `nvml:1` is lost *after* enumeration (collected inventory, lost runtime); one fake NVML session with static faults couldn't express that, and its session-wide faults on `nvmlDeviceGetMemoryInfo_v2` also failed the enumeration's memory read that FX10's own snapshot records as collected.~~ **Resolved 2026-11-11** (§1.9, DEC-46): `FakeNvml` and `StubPynvml` model device loss as a lifecycle (`lost_after_handles`, `lost_on`), the fake's enumeration total comes from the inventory, and FX10's fault input is `nvml_lost_after_handles={1: 1}` (expected snapshot unchanged; old faults kept as `LEGACY_FX10_FAULTS` with a test showing the contradiction). FX10 is back in hardware, NVML-adapter and inspector parity. | — | — |
| KI-35 | ~~The production adapter set holds a `NOT_IMPLEMENTED` stand-in for the PyTorch probe (DEC-44).~~ **Resolved 2026-11-11** (§1.11, DEC-48): `SubprocessTorchProbe` (or `InProcessTorchProbe` on request) is the production probe. | — | — |
| KI-36 | **Graphics processes aren't collected** (FR-11, data model §6.1, DEC-46). Current behaviour: `PynvmlDevice.processes()` reads only `nvmlDeviceGetComputeRunningProcesses_v3` (legacy fallback), so `runtime[*].processes` lists compute processes only; data model §6.1 specifies compute **and** graphics processes merged by PID. Evidence: `adapters/nvml.py` (only compute calls), `NVML_DEVICE_FUNCTIONS["processes"]` in `adapters/protocols.py`; data model §6.1 note added in §1.9. | The process list is incomplete for display or graphics workloads (e.g. a desktop compositor using the GPU); FR-11 (MUST) can't be complete. Never present the list as the complete process inventory. | **Open.** Proposed: add `nvmlDeviceGetGraphicsRunningProcesses_v3` (and its legacy fallback) to the read-only allowlist, merge by PID with `kind` from the source call (a PID in both → one entry; the merge rule needs a decision), and extend `StubPynvml`, `FakeNvml` and the runtime fixtures. Verification: unit tests for both calls and the merge; FX parity; HW-03 on real hardware. |
| KI-37 | **No Windows default CUDA Toolkit location** (FR-07, DEC-47 (2)). Current behaviour: on Windows the locator uses `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V*` and `nvcc` on `PATH`, but not `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*` (architecture §4.6), because `ProgramFiles` isn't in the environment-variable allowlist (data model §5.5, `ENVIRONMENT_ALLOWLIST`) and DEC-47 rejected a hard-coded path. Evidence: `adapters/toolkit.py` (`_defaults()` skipped when `windows`), DEC-47. | A Windows toolkit installed without `CUDA_PATH`/`CUDA_HOME` set and without `nvcc` on `PATH` is reported `NOT_DETECTED`; FR-07 (MUST, "platform default locations") isn't met on Windows. | **Open; needs an owner decision.** Options: allowlist `ProgramFiles` (its value is a path, privacy class **P**, redactable) and list `<ProgramFiles>/NVIDIA GPU Computing Toolkit/CUDA` non-recursively for `v<version>` entries; or keep Windows on explicit signals and amend FR-07. Security and path validation: the variable's value must be an absolute local path, resolved before use, never a UNC/network path, and the listing must stay one level deep. Verification: fake Windows trees in `unit/test_toolkit_locator.py`; a real Windows host (KI-27). The allowlist isn't widened without that decision. |
| KI-38 | **PCI domains wider than the parser accepts stay unresolved** (DEC-49 (2), AC19). Current behaviour: `normalise_pci_bus_id` accepts 4- or 8-hex-digit domains; the PyTorch probe formats the domain with `%04x`, so a domain above `0xffff` (e.g. Intel VMD domains such as `10000`) prints as 5 digits and is unusable, while NVML prints it as 8 digits (`00010000:…`). Evidence: `collectors/correlation.py` (`_PCI`), `adapters/torch_probe_payload.py` (`_device`); confirmed in the post-IMP-K07 review (`normalise_pci_bus_id("10000:01:00.0")` is `None`). | On such hosts the PCI fallback can't correlate a device whose UUID PyTorch doesn't expose; the device stays `NOT_DETECTED`/`NO_IDENTIFIER` or `UNIDENTIFIED_CANDIDATES`. Never a wrong match. | **Open.** Proposed: accept 4–8 hex-digit domains on both sides (or have the probe print 8 digits) once a source documents the formats; don't loosen validation without that evidence. Verification: unit cases for 5-digit domains; HW-09 on a VMD host. |
| KI-39 | **Project 01 isn't under version control** (post-IMP-K07 review, H1). Current behaviour: `projects/01-gpu-system-inspector/` (141 files once `.gitignore` applies) is untracked in the root repository (`git status`: `??`); the only commit (`3b8d4c8`) predates it. Evidence: `git status`, `git log`, `git add --dry-run` (2026-11-11, nothing staged). | No history, no recovery point and no diffable baseline for the work since Stage 1; reviews can't refer to commits. | **Open** until the owner establishes an authorized baseline (see the governance report of 2026-11-11 for the recommended first operation). Verification: the files appear in `git ls-files`. |

## 10. Decisions and change history

Decision details: [architecture.md](architecture.md) §6.

| Date | Entry |
| ---- | ----- |
| 2026-11-04 | Repository inspected: virtual uv workspace (`packages/*`, `projects/*`), both directories empty; validation scripts and policies read. No existing Project 01 files. |
| 2026-11-04 | OBS-03: a docs-only project directory breaks `uv lock`. **DEC-01 applied:** metadata-only `pyproject.toml` (`tool.uv.package = false`); `uv.lock` updated with the virtual member. |
| 2026-11-04 | Technical research completed; evidence register created (**DEC-23**). |
| 2026-11-04 | Decisions DEC-02, DEC-04, DEC-06 to DEC-22 and DEC-24 **proposed**. DEC-03 and DEC-05 **need explicit approval** (OQ-01, OQ-02). |
| 2026-11-04 | Specification documents DOC-01 to DOC-14 written and cross-checked for consistency (IDs, statuses, exit codes, field names, fixture matrix). |
| 2026-11-04 | **User-approved product decisions:** name "GPU System Inspector"; command `inferforge-gpu` (DEC-03); `nvidia-ml-py` required (DEC-05); PyTorch optional (DEC-26); no demo mode (DEC-17); read-only, never installs software or changes GPU state (DEC-27); must work without an NVIDIA GPU or driver. |
| 2026-11-04 | **DEC-25:** `--json PATH` / `--json -` replaces `--format`/`--output` (follows the approved example `inferforge-gpu inspect --json report.json`). |
| 2026-11-04 | Python compatibility review performed at workspace level. Proposal **WSD-01** written to `docs/python-version-review.md` (recommend `>=3.11` with a rolling floor). Not applied: root config, scripts, `uv.lock` and existing docs untouched. |
| 2026-11-04 | Final consistency audit (audit 2) completed; corrections listed in §11. |
| 2026-11-04 | **User approved the specification for implementation**, the workspace minimum Python 3.11 (WSD-01), and PyTorch as runtime-detection only (OQ-09). |
| 2026-11-04 | **Stage 1 executed** (§1.1): workspace migrated to 3.11; Project 01 activated as a package; CLI bootstrap; tests; manual isolated install. DEC-01 superseded. |
| 2026-11-04 | **Stage 2 executed** (§1.2): models, `Observed` envelope, collection/snapshot/report models, JSON serialisation, generated JSON Schema, 175 new tests. Test-only dependency `jsonschema>=4` added. Spec clarifications recorded in data model §10. |
| 2026-11-04 | **Stage 3 executed** (§1.3): spec corrections (A–D); status/severity rule enforced in model and schema (schema regenerated, still `1.0` provisional); 24 synthetic fixture variants, fakes, expectation and contract tests (478 new tests); test-only `pytest-cov>=5` added (lock: `coverage`, `pytest-cov`). |
| 2026-11-04 | **Stage 4 executed** (§1.4): adapter protocols, normalised outcomes, NVML lifecycle contract, `AdapterSet`, `SystemClock`, probe protocol decoder; `FakeCommandRunner`; fakes brought into conformance; DEC-28–DEC-34; data model §3.3 `OSError` row. No dependency or lock change. 275 new tests. |
| 2026-11-04 | **Stage 5 executed** (§1.5): production `OsPlatformAdapter` (stdlib only), `SystemCollector`, `collect_environment`, `FakeHostApi`, `host` test marker; `FIXTURE_METHOD_PREFIX` moved into `models`; `CollectorOutcome.unsupported` (internal); `AdapterSet` provenance check; DEC-35–DEC-38; data model §4.1 sources updated. No dependency or lock change. 122 new tests. |
| 2026-11-05 | **Stage 5.1 executed** (§1.5.1): FX13 root cause found (an unapproved Stage 5 downgrade); macOS processor/total memory via `sysctlbyname` (DEC-40, supersedes DEC-37); `is_wsl` → `linux_environment` with a conservative rule (DEC-39); draft schema `1.0` changed (no bump, §7.3); fixture/production parity guard; KI-26 and KI-29 resolved. Read-only investigation of the dev machine: WSL2 currently running. No dependency or lock change. 97 new tests. |
| 2026-11-05 | **Stage 6.1 executed** (§1.6.1): DEC-22 and FR-04 approved; `PciSysfsAdapter` + `SysfsReader` seam (read-only, attribute allowlist, fail-closed `NATIVE_LINUX` gate); `PciScan` with typed per-device failures (DEC-41); `NO_DRIVER_BOUND` documented. No real sysfs accessed; no dependency change. 110 new tests. |
| 2026-11-05 | **Stage 6.2 executed** (§1.6.2): approved PCI eligibility policy (DEC-42): `LINUX_ENVIRONMENT_UNKNOWN` for UNKNOWN Linux, `PLATFORM_NOT_SUPPORTED` for WSL1/WSL2/Windows/macOS, pinned evidence-referenced reasons, `GPU.PCI_SCAN`/`NVIDIA.ALL` policy as a pure function; fixtures FX04b/FX11/FX12/FX13 built from it; stale "Linux non-WSL" and profile-only wording corrected. No runtime capability emission, collector or report assembly; no real sysfs accessed; no dependency change. 80 new tests. |
| 2026-11-10 | **Stage 6.3 executed** (§1.6.3): approved PCI collector contract (DEC-43): separate `PciCollector` (`pci`), section passed through, whole-scan and per-device status mapping, host-wide `CollectionError` for omitted devices and root failures, runtime `GPU.PCI_SCAN` only, `ErrorDetail` fallback; architecture §3/§4.2/§4.4/§4.5.5, data flow §3, IMP-K02, IMP-K09, KI-32, KI-33. No orchestration, `GPUInventory` assembly, report assembly or diagnostics; no real sysfs accessed; no dependency change. 41 new tests. |
| 2026-11-10 | **Stages 6.4–6.6 executed** (§1.6.4–§1.6.6): collector-level PCI parity for all 24 fixture variants (FX03, FX17 named); `pci` fake collector and `sysfs_reader_for` in the test kit; PCI host test on this WSL2 machine (gated, no `/sys` access); documentation consolidated (architecture §4.5.5 heading, DEC order, §4.11 integration readiness; data flow §1 ②; data model §4.2; testing strategy §5, §8); KI-32/KI-33 reviewed and kept open. **Stage 6 complete** for synthetic/software and WSL2-host validation; HW-02 `BLOCKED`. No fixture, source, dependency or configuration change. 102 new tests. |
| 2026-11-11 | **Integration milestone executed** (§1.7, DEC-44): IMP-K08 core (one `Collector` contract, single guard owning `FAILED`, `COLLECTOR_ORDER`, fail-closed environment, scanner built after classification, NVML session), IMP-K02 `GpuHardwareCollector` (NVML + `nvidia-smi` fallback, PCI section consumed and kept, `NVIDIA.ALL`), IMP-P01 `assemble_report`, `api.inspect()`, `NOT_IMPLEMENTED` production stand-ins, `examples/inspect_demo.py`. FX13's `SYSTEM.MEMORY_AVAILABLE` reason corrected to match its own system section. KI-34, KI-35 added; KI-32/KI-33 stay open. No dependency or configuration change. 279 new tests. |
| 2026-11-11 | **Production NVIDIA adapters executed** (§1.8, DEC-45): IMP-A05 `SubprocessCommandRunner` (no shell, bounded, captured), IMP-A01 `PynvmlAdapter` (lazy injectable binding, lifecycle, error mapping, fallbacks, macOS gate), IMP-A02 `NvidiaSmiCli` (CSV query parser, header versions, WSL2 path, macOS gate); NVML/`nvidia-smi` precedence policy (never merge); production factory wired, only the PyTorch probe still a stand-in; `StubPynvml` added to the test kit; data model §3.3 `nvidia-smi` rows. KI-35 narrowed to the PyTorch probe; KI-25 partly resolved. No dependency change (`uv.lock` byte-identical). 178 new tests. |

### Finalized decisions

| Decision | Status | Date |
| -------- | ------ | ---- |
| Application name "GPU System Inspector" (DEC-02, display name) | Approved by user | 2026-11-04 |
| Command `inferforge-gpu` with `inspect`, `version` (DEC-03) | Approved by user | 2026-11-04 |
| `nvidia-ml-py` required dependency (DEC-05) | Approved by user | 2026-11-04 |
| PyTorch optional, runtime detection only, no dependency or extra (DEC-26, OQ-09) | Approved by user | 2026-11-04 |
| Workspace minimum Python 3.11 (WSD-01, DEC-21) | Approved by user; applied in Stage 1 | 2026-11-04 |
| Specification approved for implementation | Approved by user | 2026-11-04 |
| No demo mode in V1 (DEC-17) | Approved by user | 2026-11-04 |
| Read-only; never installs software or changes GPU state (DEC-27, NFR-12) | Approved by user | 2026-11-04 |
| Works gracefully without an NVIDIA GPU or driver (FR-01, NFR-02) | Approved by user | 2026-11-04 |
| `--json PATH` / `--json -` output design (DEC-25) | Example approved by user; `-` convention proposed | 2026-11-04 |

### Workspace-level decisions

| ID | Decision | Status | Remaining |
| -- | -------- | ------ | --------- |
| WSD-01 | Workspace minimum Python version ([../../../docs/python-version-review.md](../../../docs/python-version-review.md)) | **Approved (`>=3.11`) and applied** 2026-11-04 | Direct 3.11 test run (needs interpreter install approval). The rolling-floor rule hasn't been approved. |
| WSD-02 (proposed) | Let `verify_isolated_install.py` verify projects, e.g. an explicit `--allow-project` (KI-14) | **Not decided; not implemented** | Changes repository policy, so it needs maintainer approval. |

### Open questions

| ID | Question | Status / recommended default |
| -- | -------- | ---------------------------- |
| OQ-01 | Command name | **Resolved 2026-11-04:** `inferforge-gpu`. |
| OQ-02 | `nvidia-ml-py` required or extra | **Resolved 2026-11-04:** required. |
| OQ-03 | Distribution name if the app is ever published (currently `01-gpu-system-inspector`). | Open, non-blocking. Decide at first publication. |
| OQ-04 | Demo/simulation mode in V1 | **Resolved 2026-11-04:** no demo mode. |
| OQ-05 | Keep the workspace Python floor at 3.10? | **Superseded by WSD-01.** |
| OQ-06 | Should `--redact` become the default when `--json PATH` is used? | Open, non-blocking. Recommended: no; opt-in. |
| OQ-07 | Minimum `nvidia-ml-py` / driver version floor. | **Resolved 2026-11-11** (DEC-46, KI-16): binding floor `>=12.535.108` kept, confirmed at the binding level. A minimum *driver* version isn't enforced; the report states what it finds. |
| OQ-08 | Static type checker for the project's dev group. | Open, non-blocking. Recommended: mypy `--strict`. |
| OQ-09 | Confirm that "PyTorch optional" means runtime-detected with **no** packaging extra (DEC-26). | **Resolved 2026-11-04** by the user: runtime detection only; no dependency or extra. |
| OQ-10 | How should toolkit discovery (filesystem reads of `version.json`, default directories) be abstracted: a `ToolkitLocator` protocol, or a small read-only filesystem protocol shared with `/proc` reads? | **Resolved 2026-11-11** (DEC-47): a `ToolkitLocator` protocol in the `AdapterSet`, implemented by `CudaToolkitLocator` on the shared `CommandRunner` and a minimal `ToolkitFiles` seam used only by the locator (not shared with `/proc` reads). |
| OQ-11 | Should process names (`nvmlSystemGetProcessName` vs `/proc/<pid>/comm`) be resolved inside `NvmlDevice.processes()` or by the runtime collector? | Open, non-blocking (Stage 4). Current implementation (IMP-A01, reconciled 2026-11-11): `PynvmlDevice.processes()` resolves names itself, through `nvmlSystemGetProcessName` only (basename); the `/proc/<pid>/comm` alternative named in data model §6.1 isn't implemented. Owner decision on the fallback pending. |

## 11. Resolved work

| Date | Item | Evidence |
| ---- | ---- | -------- |
| 2026-11-04 | Workspace inspection and conflict analysis | Recorded in §10 and OBS-03. |
| 2026-11-04 | Workspace-safe project skeleton (`pyproject.toml`, README) — ARC-06 | `uv lock --offline` added only the 8-line virtual-member entry; `uv lock --check` clean; `scripts/check_dependencies.py --graph` passed (1 member, no edges); root `uv run pytest` 34 passed; `ruff check .` clean. |
| 2026-11-04 | Cross-document consistency check | All EV, DEC, FR/NFR, AC (60), FX, HW, OQ, KI and claim IDs referenced are defined exactly once; D01 cause labels renamed `D01.1`–`D01.8` to avoid clashing with claim IDs `C01`–`C15`. |
| 2026-11-04 | Specification documents DOC-01–DOC-14 | Files in `docs/`. |
| 2026-11-04 | Product decisions finalized (§10) | Applied across README, product spec, requirements, architecture, data flow, data model, CLI spec, diagnostics, testing strategy, acceptance criteria, evidence register. |
| 2026-11-04 | Audit 2 corrections | CLI renamed and `--json` design applied everywhere; D05 reclassified dependencies (required/optional/external); new NFR-17 (documentation) and IMP-C06 (Python API) close mapping gaps; AC25/AC26/AC36/AC51 broadened to cover NFR-09, FR-16, FR-02, NFR-13; AC21's undefined "FX09 variant" replaced by a defined Mode B run table (testing strategy §4.1); fixture pins completed (toolkit default, FX07/FX09/FX14–FX16/FX19); "fast, popular", "very broad" and "within seconds" claims removed or sourced; EV-02 section reference fixed; `PROBE_EXIT_<n>` harmonised; fixture banner renamed to avoid implying a demo mode; PyTorch-in-same-environment limitation documented (KI-11); traceability matrix added (acceptance criteria §I). |
| 2026-11-04 | Workspace Python review (WSD-01 proposal) | `docs/python-version-review.md`; read-only evidence OBS-04 to OBS-06. |
| 2026-11-04 | Stage 1: workspace migration (S1-WSM, S1-LOCK) | See §1.1; root tests 34 passed on CPython 3.12.12. |
| 2026-11-04 | Stage 1: package activation, CLI bootstrap, tests (S1-PKG, S1-CLI, S1-TST) | 47 passed; `check_dependencies.py` passed; `ruff check .` clean. |
| 2026-11-04 | Stage 1: isolated installation (S1-ISO) | Manual run of the script's procedure (OBS-07). |
| 2026-11-04 | Stage 2: data foundation (S2-MOD … S2-VAL) | 222 passed (workspace and isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage2-T2TNcH`); root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .`, `uv lock --check` all clean. |
| 2026-11-04 | Stage 3: fixtures and simulation (S3-SPEC … S3-COV) | 700 passed in the workspace; 675 passed + 25 skipped in isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage3-MTvvvy` (wheel has 29 entries, no test kit); coverage 95.19%; root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .`, `uv lock --check` all clean. |
| 2026-11-04 | Stage 4: adapter interfaces (S4-PROTO … S4-DOC) | 975 passed in the workspace; 948 passed + 27 skipped in isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage4-ntIecr` (wheel has 35 entries including the 6 `adapters` modules, no test kit); coverage 95.85%; root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .`, `uv lock --check` all clean. |
| 2026-11-04 | Stage 5: system and platform inspection (S5-ADAPTER … S5-QG) | 1097 passed in the workspace (9 `host` tests on this Linux/WSL2 machine); 1070 passed + 27 skipped in isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage5-ZuDFzl` (39 wheel entries, no test kit); coverage 95.61%; root 34 passed; `check_dependencies.py`, `ruff check .`, `ruff format --check .`, `uv lock --check` all clean. |
| 2026-11-05 | Stage 5.1: platform correctness (S51-FX13 … S51-QG) | 1194 passed in the workspace; 1167 passed + 27 skipped in isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage5_1-Mf4mer`; coverage 95.77%; root 34 passed; all lint, format, lock and dependency checks clean. |
| 2026-11-05 | Stage 6.1: PCI sysfs seam and adapter (S61-*) | 1304 passed; isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage6_1-AQTZbv` 1277 passed + 27 skipped; coverage 96.04%; root 34 passed; all lint, format, lock and dependency checks clean. |
| 2026-11-05 | Stage 6.2: PCI eligibility policy (S62-*) | 1384 passed; isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage6_2-AGeeUN` 1357 passed + 27 skipped; coverage 96.10%; root 34 passed; all lint, format, lock and dependency checks clean. |
| 2026-11-10 | Stage 6.3: PCI collector (S63-*) | 1426 passed; isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage6_3-rKjZTq` 1399 passed + 27 skipped; coverage 96.20%; root 34 passed; all lint, format, lock and dependency checks clean. |
| 2026-11-10 | Stages 6.4–6.6: PCI parity, host validation, consolidation (S64-*, S65-*, S66-*) | 1528 passed (13 `host`); isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-stage6_6-3nzcNt` 1501 passed + 27 skipped; coverage 96.20%; root 34 passed; all lint, format, lock and dependency checks clean. |
| 2026-11-11 | Integration milestone: IMP-K08, IMP-K02, IMP-P01 (§1.7) | 1807 passed (17 `host`); isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-k08-LdBqn8` 1780 passed + 27 skipped, `api.inspect()` OK from the wheel; coverage 96.63%; root 34 passed; all lint, format, lock and dependency checks clean; demo exit 0. |
| 2026-11-11 | Production NVIDIA adapters: IMP-A05, IMP-A01, IMP-A02 (§1.8) | 1985 passed (17 `host`); isolated venv `/tmp/inferforge-isolated-01-gpu-system-inspector-a01-T3dfjn` 1958 passed + 27 skipped, `api.inspect()` OK from the wheel; coverage 97.17%; root 34 passed; all lint, format, lock and dependency checks clean; demo exit 0 (NVML library not found, no `nvidia-smi`). |
| 2026-11-11 | Driver and runtime collectors: IMP-K03, IMP-K04; KI-34, KI-16 resolved (§1.9, DEC-46) | 2106 passed (18 `host`); coverage 97.37%; isolated wheel 2079 passed + 27 skipped; `uv.lock` unchanged |
| 2026-11-11 | CUDA environment: OQ-10 resolved (DEC-47), IMP-A06, IMP-K05, KI-28 (§1.10) | 2245 passed (19 `host`); coverage 97.54%; isolated wheel 2218 passed + 27 skipped; `uv.lock` unchanged |
| 2026-11-11 | PyTorch: IMP-A07 probe, IMP-K06 collector (DEC-48), KI-35 (§1.11) | 2347 passed (20 `host`); coverage 97.18%; isolated wheel 2320 passed + 27 skipped; `uv.lock` unchanged |
| 2026-11-11 | Device correlation: IMP-K07 (DEC-09 applied, DEC-49) (§1.12) | 2410 passed (20 `host`); coverage 97.24%; isolated wheel 2383 passed + 27 skipped; `uv.lock` unchanged |
| 2026-11-11 | Pre-diagnostics governance reconciliation (post-IMP-K07 review findings H1, M1–M4, M6, L5, L8) | Only change outside `docs/`: the `api.py` module docstring. KI-36 to KI-39 added; DEC-04/06/07/08 marked "Proposed; applied", owner approval not recorded; DEC-26 OQ-09 reference updated; DEC-48 (4) marked superseded by DEC-49 (5); DEC-43 row reordered; FR-11/12/13 and AC43 notes corrected; AC05, AC13, AC25, AC54 `TESTED` with their evidence; OQ-11 current state recorded; no hardware status changed |

## 12. In-progress work

| Item | Status | Next step |
| ---- | ------ | --------- |
| macOS host run of the system collector (KI-30) | `PLANNED` | Run `uv run pytest -m host` on any Mac. |
| Review of Stage 1–6 and integration-milestone documentation amendments (§7) | Awaiting user review | User reviews the post-approval changes, especially the Stage 5.1 macOS decision (DEC-40), the WSL rule (data model §4.1), the Stage 6 PCI decisions (DEC-41, DEC-42, DEC-43), the integration decisions (DEC-44: collector contract, guard, placeholders, stand-ins), the NVIDIA adapter decisions (DEC-45: runner safety, NVML lifecycle and mapping, CSV parsing, precedence policy) and the integration contract and delivery (architecture §4.11): data model §3.2/§3.3/§3.4/§4.1/§4.2/§7.3/§10, diagnostics spec §2, testing strategy §2/§3.1/§5/§6/§8, architecture §2/§3/§4.2/§4.3/§4.4/§4.5 (including §4.5.1, §4.5.4, §4.5.5)/§4.11/§6, data flow §1 step ②/§1.1/§3–§5, project README status. |
| Windows host run of the system collector (KI-27) | `PLANNED` | Run `uv run pytest -m host` on any Windows machine. |
| Stage 6 | Complete (software and WSL2 host); HW-02 `BLOCKED` | 6.1–6.6 complete (§1.6.1–§1.6.6). |
| Integration milestone (IMP-K08, IMP-K02, IMP-P01) | Complete (synthetic and WSL2 host) | §1.7. |
| Production NVIDIA adapters (IMP-A05, IMP-A01, IMP-A02) | Complete (synthetic and WSL2 host); HW-01 `BLOCKED` | §1.8. |
| Driver and runtime collectors (IMP-K03, IMP-K04; KI-34, KI-16) | Complete (synthetic and WSL2 host); HW-01 `BLOCKED` | §1.9. |
| CUDA environment (OQ-10/DEC-47, IMP-A06, IMP-K05) | Complete (synthetic and WSL2 host); HW-05 not run | §1.10. |
| PyTorch (IMP-A07, IMP-K06; DEC-48) | Complete (synthetic, stub child, WSL2 host); HW-04, HW-06 `BLOCKED` | §1.11. |
| Device correlation (IMP-K07; DEC-49) | Complete (synthetic and WSL2 host); HW-09 `BLOCKED` | §1.12. Next recommended: diagnostics (IMP-D*) or the CLI (IMP-C01, IMP-P03), on request. |

## 13. Blocked work

| Item | Blocker | Unblock condition |
| ---- | ------- | ----------------- |
| HW-01 to HW-09, HW-11, HW-12; DEMO-02 | No NVIDIA GPU, driver, toolkit or macOS machine available (OBS-01) | Access to a Linux machine with an NVIDIA GPU (ideally also WSL2-with-NVIDIA, Windows, multi-GPU and macOS machines). |
| S1-PY311: direct Python 3.11 validation | No 3.11 interpreter installed; installing one needs approval | User approves `uv python install 3.11`; then run root and project tests with `uv run --python 3.11 pytest`. |
| Isolated-install script support for projects (WSD-02) | Repository policy decision | Maintainer decides whether the script should accept projects. |

## 14. Deferred work (out of V1)

Recorded in [product_specification.md](product_specification.md) §7: `--python`
interpreter targeting, MIG instance enumeration, NVML clock event reasons, NVML↔`nvidia-smi`
cross-validation, `mem_get_info` readiness source, other frameworks, container detection,
Windows/macOS hardware presence checks, report diffing, model-memory estimation,
`--quiet`, colour/Markdown/HTML output. All `DEFERRED`.

Not planned (decided, not merely deferred): demo/simulation mode (DEC-17), an `inferforge`
umbrella command (DEC-03), and PyTorch as a dependency or extra (DEC-26).

## 15. Remaining work (recommended implementation sequence)

Steps 1–3 are done (Stage 1, 2026-11-04): specification approved; WSD-01 applied;
`pyproject.toml` converted. Open from Stage 1: direct 3.11 validation (S1-PY311).
Step 4 is done (Stage 2, 2026-11-04): models, `Observed` invariant, JSON serialisation,
JSON Schema. Step 5 is done (Stage 3, 2026-11-04): fakes, fixtures FX01–FX20 and the
Mode B runs (TST-01/02), coverage measurement. Stage 4 (2026-11-04) defined the adapter
interfaces, outcomes, lifecycle contracts and `AdapterSet`, and added `FakeCommandRunner`.

6. Adapters (implementations of the Stage 4 protocols): `CommandRunner`, `PlatformAdapter`, `NvmlAdapter`, `NvidiaSmiAdapter`,
   `PciSysfsAdapter`, `ToolkitLocator`, `TorchProbe` (IMP-A*), with unit tests.
7. Collectors + core guard/ordering + correlation (IMP-K*), with integration tests. Done: system (IMP-K01), PCI (IMP-K09), GPU hardware (IMP-K02), driver (IMP-K03), runtime metrics (IMP-K04), CUDA environment (IMP-K05) and PyTorch (IMP-K06) collectors, device correlation (IMP-K07), the core (IMP-K08) and report assembly (IMP-P01). Step 6 progress: the command runner, NVML and `nvidia-smi` adapters (IMP-A05, A01, A02) are done; the toolkit locator (IMP-A06) and the PyTorch probe (IMP-A07) are done. All collectors done (IMP-K01 to IMP-K09).
8. Readiness engine (IMP-R01).
9. Diagnostics engine and rules D01–D08 (IMP-D*), with the fixture matrix.
10. Reporting: assembly, text renderer, redaction, atomic writer (IMP-P*).
11. CLI (`inferforge-gpu`) and Python API (IMP-C01–C06).
12. Quality gates: ruff, types, coverage, `check_dependencies` (TST-15, AC51).
13. HW-13 and HW-10 on the WSL2 dev machine; user check of AC01/AC38 (MAN).
14. Hardware validation on NVIDIA machines (HW-01 to HW-12) when available.
15. README usage docs, demonstration, final review (DOC-15, DEMO-*, REV-*).

## 16. Final completion summary

| Measure | Value |
| ------- | ----- |
| Phase | Stages 1–5.1 and 6 complete (workspace, package, CLI bootstrap; data foundation; synthetic fixtures, fakes, coverage; adapter contracts; system/platform inspection and its correction; PCI sysfs seam and adapter; PCI eligibility policy; PCI collector, fixture parity, WSL2 host validation, documentation consolidation), plus the integration milestone (IMP-K08 core, IMP-K02 GPU hardware collector, IMP-P01 report assembly, `api.inspect()`). Production NVML, `nvidia-smi` and command-runner adapters (IMP-A01, A02, A05) wired into `api.inspect()`; never run on NVIDIA hardware (HW-01 `BLOCKED`). Native-Linux PCI hardware validation (HW-02) `BLOCKED`. |
| Product decisions | Finalized and approved (§10) |
| Acceptance criteria met | 0 / 60. `IN_PROGRESS`: AC04, AC05, AC09, AC13, AC18, AC25, AC45, AC46, AC50, AC53. |
| AC traceability | 60 / 60 mapped to requirement, component and verification |
| Code | CLI bootstrap (`cli/`); the data foundation: `models/` (13 modules) and `reporting/` (`json_report.py`, `schema.py`, `__main__.py`, shipped schema); `adapters/` (contracts, `SystemClock`, `OsPlatformAdapter`, `PciSysfsAdapter`); `collectors/` (`Collector` contract, `SystemCollector`, `collect_environment`, PCI capability policy, `PciCollector`, `GpuHardwareCollector`); `core/` (guard, inspector, environment resolution, production factory); `reporting/assemble.py`; `api.py`; `adapters/unimplemented.py` (stand-ins). Host access is read-only and limited to `OsPlatformAdapter` and the native-Linux-only PCI reader; no NVIDIA, process or network access. |
| Automated tests | Project 01: 1985 passed (workspace, including 17 host tests) / 1958 passed + 27 skipped (isolated venv), CPython 3.12.12; coverage 97.17%. Root: 34 passed. |
| Synthetic test kit | 24 fixture variants, 30 report cases, 9 fakes (including `FakeCommandRunner`) conforming to the adapter protocols, plus the `FakeHostApi` and `FakeSysfsReader` seam fakes and `FakeCollector` (with a `pci` entry), in `tests/gsi_testkit/` (not shipped). Synthetic evidence only: no AC or HW item completed because of it. |
| Adapter layer | Interfaces, outcomes, lifecycle contracts, `AdapterSet`, `SystemClock`, the production `OsPlatformAdapter`, `PciSysfsAdapter` (synthetic tests only), `SubprocessCommandRunner`, `PynvmlAdapter` and `NvidiaSmiCli` in `adapters/`; the PyTorch probe is a stand-in. On this WSL2 host the real NVML binding ran (`nvmlInit` → library not found, no session) and `nvidia-smi` wasn't found; no NVML session on NVIDIA hardware, real `nvidia-smi` run or PyTorch probe has ever been executed. |
| Production collectors | System/platform (Stage 5), read-only, stdlib only; PCI (Stage 6: synthetic tests, fixture parity, gated on this WSL2 host; never run against real native-Linux sysfs); GPU hardware (production NVML/`nvidia-smi` adapters since §1.8; synthetic and no-NVIDIA-host evidence only). All run through `api.inspect()`; none is reachable from the CLI yet (KI-15). |
| Hardware validation | None (blocked) |
| User verification | Specification documents approved 2026-11-04 (§7). No implementation or test item is user-verified. |
| Python 3.11 | Declared floor; not directly tested (S1-PY311 `BLOCKED`) |
| Workspace integrity after Stage 1 | `uv lock --check` clean; `check_dependencies.py` passed; `ruff check .` clean; `uv sync` OK (2026-11-04) |
| Files changed outside the project (cumulative) | `pyproject.toml`, `uv.lock`, `scripts/_workspace.py`, `scripts/verify_isolated_install.py`, `scripts/tests/conftest.py`, `README.md`, `projects/README.md`, `docs/python-version-review.md` (new); `.venv` recreated. |
