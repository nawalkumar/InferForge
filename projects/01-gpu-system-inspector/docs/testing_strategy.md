# Testing Strategy — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. No tests exist yet.

## 1. Principles

1. **The ordinary suite needs no GPU, driver, NVML, `nvidia-smi`, CUDA Toolkit or
   PyTorch.** Everything external sits behind adapters (architecture §4.5). Tests replace
   adapters with fakes.
2. **Fakes live at the adapter seam, not inside third-party modules.** Collectors and
   engines are tested against fake adapters. Adapters are tested against fake *modules*
   (a stub `pynvml` object) and canned command outputs.
3. **Mocked data is never real data.** Fixture-built reports have
   `data_origin = "FIXTURE"`. Nothing produced by the ordinary suite counts as hardware
   validation. Results from the hardware suite are recorded separately in the tracker
   (§6, [app_tracker.md](app_tracker.md) §6).
4. **Deterministic.** Fixed clock, fixed ordering, no network, no reliance on the host's
   real environment (tests clear or fake environment variables).
5. **Every acceptance criterion maps to at least one test or manual verification**
   ([acceptance_criteria.md](acceptance_criteria.md)).

## 2. Layout, markers and commands

```
tests/
├── unit/          # models, adapters, collectors, diagnostics, readiness, reporting, cli parsing
├── integration/   # full Inspector runs over fixture scenarios; CLI end-to-end via main()
├── hardware/      # real-hardware tests, marker `hardware`, deselected by default
└── fixtures/      # scenario builders, fake adapters, canned outputs, golden files
```

| Command (from `projects/01-gpu-system-inspector/`) | Runs |
| -------------------------------------------------- | ---- |
| `uv run pytest` | unit + integration (hardware deselected via `-m 'not hardware'` in project config) |
| `uv run pytest -m hardware` | hardware tests only. Each skips with an explicit reason if its preconditions are absent. |
| `uv run pytest -m host` | host integration tests only (`host` marker, Stage 5): read-only reads of the real OS through the production system adapter; no NVIDIA hardware needed. They also run in the default suite. |
| `uv run python ../../scripts/check_dependencies.py` (or from root) | workspace dependency rules |
| `uv run ruff check .` | lint (root config) |
| `uv run pytest --cov --cov-report=term-missing` | the ordinary suite with coverage; fails below the target in §8 |

Test-only dependencies (project `[dependency-groups] test`): `pytest>=8`, `jsonschema>=4`,
`pytest-cov>=5`. No GPU-related test dependency is needed. Coverage configuration lives
only in the project's `pyproject.toml` (`[tool.coverage.*]`), so root tests are unaffected.

**Test kit layout (Stage 3).** `tests/gsi_testkit/` is importable from tests
(`pythonpath = ["tests"]`) and is never shipped in the wheel:

| Module | Contents |
| ------ | -------- |
| `builders.py` | Synthetic primitives (`ok`, `failed`, …), component builders, and the free-form `Builders` used by model tests |
| `fixtures/` | `scenario.py` (the `FixtureScenario` record), `catalog_core.py` (FX01–FX10), `catalog_platform.py` (FX11–FX20) |
| `expectations.py` | Expected D01–D08 and readiness outcomes, with materialisation into models |
| `fakes/` | Fake adapters and collectors (§5) |
| `spec_tables.py` | Reads diagnostics spec §6 and §4.1 below so tests can prove the expectations match the approved tables |

**Provenance rule.** Every synthetic `Observed.method` starts with `fixture:`, and every
fixture report has `data_origin = FIXTURE` and a "Synthetic fixture …" limitation. Tests
enforce both.

## 3. Test levels

### 3.1 Unit tests

| Area | What is tested |
| ---- | -------------- |
| Models | `Observed` invariant (value non-null iff `COLLECTED`) enforced at construction; enum serialisation; `unavailable()` constructors; immutability. |
| Adapters: NVML | **Implemented (IMP-A01):** `unit/test_nvml_adapter.py` uses `StubPynvml` (test kit), a stub `pynvml` injected via the adapter's loader, whose error values and messages are checked against the real binding: every init failure → status mapping (data model §3.3); missing binding; macOS gate without loading; zero, one and several devices; every property and conversion; `None` → `VALUE_NOT_AVAILABLE`; unusable values → `PARSE_ERROR`; a lost handle or read isolated; v2/v1, `V`/legacy, `_v3`/legacy fallbacks and an older binding; `nvmlShutdown` once on every path, never after a failed init, shutdown errors swallowed; one session at a time; closed-session usage errors; only read-only functions (AC43); parity with every fixture's NVML facts and inventory (FX10 through the stub's device-loss lifecycle); the installed binding has every call the adapter needs (KI-16: with the floor `nvidia-ml-py==12.535.108` installed in a throwaway environment, this test and the NVML adapter, collector, device-loss and production pipeline suites pass, 212 tests; the floor lacks only `nvmlDeviceGetTemperatureV`, served by the fallback). |
| Adapters: `nvidia-smi` | **Implemented (IMP-A02):** `unit/test_nvidia_smi_adapter.py`: the CSV parser alone (one/many GPUs, quoted fields, empty output, every placeholder, unparsable values per field, wrong column counts, bad/duplicate index, partial output), the adapter over `FakeCommandRunner` (PATH and WSL2 lookup, macOS gate, missing executable, one bounded query, non-zero exit, unknown field, timeout, truncation, both header label sets), a fake `nvidia-smi` script through the production runner, and FX04a's fallback inventory reproduced by both production adapters. |
| Adapters: PCI sysfs | Dict-based `FakeSysfsReader` (FIXTURE) for the adapter, and `tmp_path` trees for the real reader's primitives only (Stage 6.1): NVIDIA + `nvidia`, NVIDIA + `nouveau`, NVIDIA + no driver, non-NVIDIA only, empty. Only `NATIVE_LINUX` scans; WSL1, WSL2, `UNKNOWN`, Windows and macOS → `NOT_SUPPORTED` with zero reader calls (`LINUX_ENVIRONMENT_UNKNOWN` for `UNKNOWN`, `PLATFORM_NOT_SUPPORTED` otherwise). Stage 6.2 (`unit/test_pci_capability_policy.py`): pinned reasons, `GPU.PCI_SCAN`/`NVIDIA.ALL` policy for all six environments, and parity with every fixture's PCI section and capability records. |
| Adapters: toolkit | **Implemented (IMP-A06, DEC-47):** `unit/test_toolkit_locator.py` runs `CudaToolkitLocator` over `FakeToolkitFiles` (in-memory tree with symlinks and injected `OSError`s) and `FakeCommandRunner`: no toolkit; one; several and a `/usr/local/cuda` alias (all reported, ordered, signals merged); duplicate signals; `nvcc` through a symlink and outside `bin/`; signals naming no directory; non-`cuda` default entries; missing compiler; `version.json` → `nvcc` fallback (run once) → `NO_VERSION_EVIDENCE`; eight malformed or unsupported metadata forms (`PARSE_ERROR`, no fallback); unreadable metadata, unstatable root and an unreadable `/usr/local` (their own status, never `NOT_DETECTED`); `nvcc` timeout, non-zero exit, unexpected and empty output, denied, vanished; Windows (`nvcc.exe`, `CUDA_PATH_V*`, no default location); macOS/`OTHER` (nothing read or run); determinism; nothing done at construction; `StdlibToolkitFiles` on a `tmp_path` tree. |
| Adapters: command runner | **Implemented (IMP-A05):** `unit/test_subprocess_command_runner.py` runs tiny fake executables in `tmp_path` (POSIX): PATH and extra-directory lookup, missing and non-executable files, success with stdout/stderr and clock duration, verbatim arguments with shell metacharacters (nothing expanded), non-zero exit and `require_success`, empty output, no stdin, timeout (child killed), missing program, permission denied, output cap. A mocked `subprocess.run` proves list argv, `shell=False`, `stdin=DEVNULL`, the timeout and no environment argument; other `OSError`s, `None` output, malformed results and unexpected exceptions; caller bugs start nothing; importing the module imports no `subprocess`. |
| Adapters: torch probe | **Implemented (IMP-A07, DEC-48):** `unit/test_adapter_outcomes.py` (protocol 2 decoding: versions 1 and 3 rejected, every missing/extra/mistyped key incl. `device_error`, crash, timeout, garbage, `import_ok = false`); `unit/test_torch_probe.py` (26): the probe function over `StubTorch` (CUDA build with devices, CPU-only, ROCm, captured warnings, missing attributes, enumeration failure with available and unavailable CUDA, import-failure document, no allocation or `mem_get_info`); the probe child in a real subprocess with a stub `torch` package, a failing import and no torch; importing the probe modules imports no torch and no `subprocess`; `SubprocessTorchProbe` (argv, timeout, duration, every failure code, `sys.executable` default, distribution lookup mapping, nothing at construction); `InProcessTorchProbe` (allocator only with CUDA already initialised, import and unexpected failures as outcomes). |
| Collectors | Each collector over fake adapters per platform profile; per-device isolation (one device raises, others intact); `SKIPPED` behaviour for opt-outs; correlation by UUID/PCI with prefix normalisation, uncorrelated devices. |
| Diagnostics | Table-driven tests for every row of every decision table in the diagnostics spec §3, plus §5 wording rules. |
| Readiness | Every device/overall reason code; `free == required` boundary → `READY`; GiB → bytes rounding (`7.5` → `8053063680`); negative margin; MIG; no-device cases per platform. |
| Reporting | JSON key order; every key present; `allow_nan=False`; schema validation; text section order and missing-value rendering; FIXTURE banner; redaction (no sensitive literal survives anywhere, placeholder stability); atomic write (failure leaves no file). |
| CLI parsing | All validation rules of [cli_specification.md](cli_specification.md) §2.2; exit-code table §6; help text content. |
| Error handling | Collector guard converts any exception into `CollectionError`; rule guard converts any exception into an `UNKNOWN` finding; internal error path → exit 5 and no stdout. |

### 3.2 Integration tests

| Test | Description |
| ---- | ----------- |
| Full inspection per fixture | For each FX scenario (§4): run `Inspector.run()` with the scenario's `AdapterSet`; assert key fields, statuses, `collection_errors`, `unsupported_capabilities`, and the expected findings matrix (diagnostics spec §6). |
| CLI end-to-end | `main([...])` (the `inferforge-gpu` entry point) with the fake `AdapterSet` installed through the factory seam; capture stdout/stderr; assert exit codes for Mode A, READY/NOT_READY/UNKNOWN, usage errors, JSON write errors, internal error and interrupt. |
| JSON reporting | Every fixture report validates against the shipped JSON Schema; determinism (two runs → identical bytes); `--json -` stdout parses with `json.loads` and nothing else appears on stdout; `--json PATH` file content equals the `--json -` output. |
| Golden text reports | A small set (FX01, FX03, FX07, FX13) compared with reviewed golden files; regenerate only intentionally. |
| PyTorch probe protocol | Run the real `_torch_probe_main` in a subprocess with a **stub `torch` package** on `PYTHONPATH` in a temp directory (no real PyTorch). Variants: CUDA available, CPU-only, import raises, `is_available()` warns. |
| No network | Run the full flow with `socket.socket` patched to raise. Any attempt fails the test (AC42). |
| Workspace | `scripts/check_dependencies.py` passes with the project present (AC51). |

## 4. Fixture scenarios

All fixtures use `FixedClock(2026-01-01T00:00:00Z)` and `data_origin = FIXTURE`. The
required `nvidia-ml-py` binding is *installed* unless stated (FX04a simulates a broken
installation). No CUDA Toolkit is present unless stated. "GPU A" = 24 GiB device,
"GPU B" = 12 GiB device. Values are synthetic. Fixtures run in Mode A unless marked (B).
§4.1 lists the additional Mode B runs.

| ID | Scenario (required) | Pinned inputs |
| -- | ------------------- | ------------- |
| FX01 | NVIDIA GPU with working CUDA | Linux; NVML OK; 1 × GPU A, free 20 GiB; driver supports CUDA 12.4; toolkit 12.4 via `version.json`; PyTorch 2.x, build CUDA 12.4, available, correlated by UUID; all runtime metrics collected. |
| FX02a | NVIDIA GPU, PyTorch CUDA unavailable: CPU-only build | As FX01, but `cuda_build = false`, `cuda_available = false`. |
| FX02b | …: driver older than build | Driver supports CUDA 11.8; PyTorch build CUDA 12.4; `cuda_available = false`. |
| FX02c | …: devices hidden | `CUDA_VISIBLE_DEVICES=""`; `cuda_available = false`, `device_count = 0`. |
| FX02d | …: unexplained | All layers healthy; `cuda_available = false`; `cuda_init_warnings = ["CUDA initialization: Unexpected error from cudaGetDeviceCount()…"]`. |
| FX03 | No NVIDIA GPU | Linux bare metal; NVML `LIBRARY_NOT_FOUND`; `nvidia-smi` not found; PCI scan collected: one Intel display controller (`0x8086`); PyTorch CPU-only installed. |
| FX04a | NVML missing (binding; broken installation of a required dependency) | `pynvml` import fails; `nvidia-smi` present with 1 × GPU A via CSV; toolkit 12.4; PyTorch CUDA available. |
| FX04b | NVML missing (library) | WSL2; binding installed; NVML `LIBRARY_NOT_FOUND`; no `nvidia-smi`; PCI `NOT_SUPPORTED`; PyTorch not installed. (Mirrors OBS-01.) |
| FX05 | PyTorch missing | As FX01, but `torch` distribution not installed. |
| FX06 | Multiple GPUs | 2 devices: NVML 0 = GPU B (bus 01), NVML 1 = GPU A (bus 02); PyTorch `cuda:0` = GPU A (UUID match to `nvml:1`), `cuda:1` = GPU B; `CUDA_DEVICE_ORDER` unset. |
| FX07 | Insufficient free memory | (B) `--required-vram 8`; otherwise as FX01 but 2 devices (GPU A, GPU B) with free 5 GiB and 7.99 GiB. Expected overall `NOT_READY`, exit 1. |
| FX08 | Driver present, toolkit unavailable | As FX01, no toolkit candidates, no `nvcc`. |
| FX09 | Runtime metrics inaccessible | As FX01, but memory info (total/used/free/reserved, from the same NVML call) `PERMISSION_DENIED`, processes `PERMISSION_DENIED`, utilization `NOT_SUPPORTED`, temperature and slowdown threshold `NOT_SUPPORTED`, power `PERMISSION_DENIED`. |
| FX10 | Partial collector failure | 2 devices; the CUDA-environment collector raises `RuntimeError`; `nvml:1` is lost after enumeration (fake: `nvml_lost_after_handles={1: 1}`), so its runtime queries raise `NVMLError_GpuIsLost`; all else healthy. |
| FX11 | WSL2 with NVIDIA | WSL2; NVML OK via `/usr/lib/wsl/lib`; utilization and processes `NOT_SUPPORTED`; toolkit 12.4; PyTorch CUDA available. |
| FX12 | Windows WDDM | Windows; NVML OK; processes listed with `used_memory_bytes = NOT_SUPPORTED`; PCI `NOT_SUPPORTED`; toolkit 12.4 via `CUDA_PATH`, `CUDA_PATH_V12_4` and `nvcc.exe` on `PATH` (DEC-47); PyTorch CUDA available. |
| FX13 | macOS | Darwin arm64; NVIDIA `NOT_SUPPORTED`; PyTorch installed CPU/MPS build, `cuda_available = false`. |
| FX14 | MIG enabled | (B) `--required-vram 8`; otherwise as FX01 but 1 × 80 GiB device with MIG `ENABLED`, free 40 GiB, aggregate memory `COLLECTED`. Expected `UNKNOWN`, exit 3. |
| FX15 | PyTorch import fails | As FX01, but the distribution is present and the probe returns `import_ok = false` with `OSError: libcudart.so.12: cannot open shared object file`. |
| FX16 | PyTorch probe timeout | As FX01, but the probe exceeds `pytorch_timeout_s`. |
| FX17 | NVIDIA on PCI, driver not loaded | Linux; PCI: `0x10de` device bound to `nouveau`; NVML `DRIVER_NOT_LOADED`; no `nvidia-smi`; PyTorch not installed. |
| FX18 | Hot GPU | As FX01, temperature 92 °C, slowdown threshold 90 °C. |
| FX19 | Readiness boundary | (B) `--required-vram 8`; as FX01 but free exactly `8589934592` bytes. Expected `READY`, exit 0. |
| FX20 | `pynvml` distribution | As FX01, but module `pynvml` provided by distribution `pynvml`. |

Required scenarios 1–10 from the product brief map to FX01, FX02a–d, FX03, FX04a–b, FX05,
FX06, FX07, FX08, FX09 and FX10. FX11–FX20 cover platform and edge cases needed by the
acceptance criteria. Expected diagnostic outcomes for every fixture are in
[diagnostics_specification.md](diagnostics_specification.md) §6.

**Implementation (Stage 3).** All 24 variants are implemented in `tests/gsi_testkit/fixtures/`
with their premises, expected D01–D08 outcomes (status, severity, confidence, rule basis)
and readiness runs. Tests prove that they match diagnostics spec §6 and §4.1 below cell by
cell. FX14 adds one assumption beyond its pinned inputs: utilization is the field reported
`NOT_SUPPORTED` that yields D06 `INFO`.

**Integration milestone (IMP-K08/K02/P01).** Running the production core and collectors over
each fixture's fakes reproduces its `system`, `gpu` and PCI/NVML records with two
exceptions. FX13's `SYSTEM.MEMORY_AVAILABLE` reason contradicted its own system section
("Not collected on macOS in V1." vs the field's message, which the system collector copies,
architecture §4.3); the fixture now takes the reason from its section, and the system
collector test compares whole records instead of `(capability, status)`. FX10's `nvml:1`
was lost *after* enumeration (collected inventory, lost runtime), which its static,
session-wide faults couldn't express (KI-34).

**Driver and runtime milestone (IMP-K03/K04, DEC-46).** KI-34 is resolved: `FakeNvml` and
`StubPynvml` model device loss as a lifecycle (`lost_after_handles`: lost once that many
handles were acquired; `lost_on`: lost at the first call of an NVML function, device or
system), and the fake's enumeration memory total comes from the inventory. FX10's fault
*input* changed from session-wide faults on the runtime functions (which included
`nvmlDeviceGetMemoryInfo_v2`, the enumeration's memory read that FX10's snapshot records as
collected) to `nvml_lost_after_handles={1: 1}`; its expected snapshot is unchanged. The old
faults are kept in `unit/test_device_loss.py` (`LEGACY_FX10_FAULTS`) with a test showing the
contradiction. Every fixture's whole `system`, `gpu`, `driver` and `runtime` sections, runs,
errors and owned records are now reproduced by the production core, FX10 included.

**PyTorch milestone (IMP-A07/K06, DEC-48).** `FakeAdapterSet.torch_probe` is the production
`SubprocessTorchProbe` over `FakeCommandRunner`, which `FakeTorchProbe.canned()` teaches the
fixture's probe child output (or its timeout); `torch_fake` keeps the fixture-derived fake,
and its `distribution_lookup` stands in for `importlib.metadata`. Every fixture's whole
PyTorch section, run and errors were reproduced with one deliberate difference, each
device's `correlated_device_ref` (IMP-K07's), which the parity helpers replaced and nothing
else; since IMP-K07 (below) the sections are compared exactly. The probe protocol is now version 2
(`device_error`); fixtures take the version from `PROBE_PROTOCOL_VERSION`.

**Correlation milestone (IMP-K07, DEC-49).** The fixtures already pinned the expected
correlation (FX06 cross-ordered by UUID, FX04a `NOT_APPLICABLE`, every other device by UUID),
and the production pipeline now reproduces every fixture's PyTorch section exactly, with no
field replaced; the PyTorch run's interim `correlated_device_ref` changed from
`NOT_IMPLEMENTED` to `NOT_APPLICABLE` "Set by the correlation run (DEC-09)". No fixture
exercises the PCI fallback or ambiguity, so those are unit-tested on synthetic inventories.
Real cross-tool correlation is HW-09.

**CUDA environment milestone (IMP-A06/K05, DEC-47).** Three fixture corrections, each a
contradiction with the specification: every fixture's `environment` now lists the whole
allowlist (data model §5.1/§5.5 and the implemented `collect_environment` read all 11 names,
unset → `NOT_DETECTED`; the fixtures listed 3–5, and `FakePlatform` refused the rest); the
`/usr/local/cuda-12.4` candidates add `DEFAULT_LOCATION` (§4.6 consults the default
locations and records every signal); FX12 adds `NVCC_ON_PATH` and `CUDA_PATH_VERSIONED` and
uses `nvcc.exe`. `FakeAdapterSet` builds the production `CudaToolkitLocator` over
`FakeToolkitFiles.from_snapshot` and `FakeCommandRunner.from_snapshot` (with canned
`nvcc --version`); a fixture's injected `cuda_environment` failure (FX10) becomes a
`FailingToolkitLocator`, so the real guard produces the `FAILED` run. Every fixture's whole
`cuda_environment` section, run, errors and `CUDA.*` records are reproduced.

### 4.1 Expected readiness outcomes (Mode B runs)

| Run | `--required-vram` | Overall status / reason | Device results | D02 | Exit |
| --- | ----------------- | ----------------------- | -------------- | --- | ---- |
| FX01 (B) | 8 | `READY` / `AT_LEAST_ONE_DEVICE_READY` | `nvml:0` READY, margin +12 GiB | PASS | 0 |
| FX03 (B) | 8 | `NOT_READY` / `NO_NVIDIA_DEVICE` | none | FAIL | 1 |
| FX04b (B) | 8 | `UNKNOWN` / `DEVICE_PRESENCE_UNKNOWN` | none | UNKNOWN | 3 |
| FX07 | 8 | `NOT_READY` / `NO_DEVICE_READY` | both `INSUFFICIENT_FREE_MEMORY` | FAIL | 1 |
| FX09 (B) | 8 | `UNKNOWN` / `SOME_DEVICES_UNKNOWN` | `nvml:0` `FREE_MEMORY_UNAVAILABLE` | UNKNOWN | 3 |
| FX13 (B) | 8 | `NOT_READY` / `PLATFORM_NOT_SUPPORTED` | none | FAIL | 1 |
| FX14 | 8 | `UNKNOWN` / `SOME_DEVICES_UNKNOWN` | `nvml:0` `MIG_ENABLED_UNSUPPORTED` | UNKNOWN | 3 |
| FX19 | 8 | `READY` / `AT_LEAST_ONE_DEVICE_READY` | `nvml:0` READY, margin 0 | PASS | 0 |
| FX06 (B) | 16 | `READY` / `AT_LEAST_ONE_DEVICE_READY` | `nvml:1` (GPU A, free 20 GiB) READY; `nvml:0` (GPU B, total 12 GiB) `INSUFFICIENT_TOTAL_MEMORY` | PASS | 0 |

For FX06 (B), GPU A has 20 GiB free and GPU B 10 GiB free. In every Mode B run, the
other checks (D01, D03–D08) keep their Mode A outcomes.

## 5. Fake adapters

Implemented under `tests/gsi_testkit/fakes/` (Stage 3) and brought into conformance with
the adapter protocols (Stage 4, architecture §4.5). Each fake is derived from an FX fixture
snapshot, is deterministic, records a call log, has `provenance = FIXTURE`, and never
touches NVML, processes, the OS or the network (enforced by an import check).

| Fake | Protocol | Capabilities | Status |
| ---- | -------- | ------------ | ------ |
| `FakeNvml` / `FakeNvmlSession` / `FakeNvmlDevice` | `NvmlAdapter` / `NvmlSession` / `NvmlDevice` | Read-only NVML answers from the fixture. When NVML is unavailable, returns the init failure with each field's unit. Per-(device, function) faults become mapped outcomes, e.g. FX10 `GPU_IS_LOST` → `COLLECTION_FAILED`, equal to the fixture's error. Enforces the §4.5.3 lifecycle (balanced init/shutdown, idempotent close, one open session, usage errors after close). Call log ⊆ `NVML_READ_ONLY_FUNCTIONS` (AC43); `shutdown_fault` simulates a failing shutdown. | Conforming |
| `FakeNvidiaSmi` | `NvidiaSmiAdapter` | `locate()`, `version()`, `query_gpus()` rows carrying every query field, with MiB precision; a failed `locate` or header propagates. | Conforming |
| `FakePciScanner` | `PciScanner` | Returns the fixture's PCI observations as a `PciScan` (no per-device failures). Per-device failures are produced with the production adapter over `FakeSysfsReader` (`sysfs_reader_for` builds a reader holding exactly a fixture's observations). | Conforming |
| `FakeTorchProbe` | `TorchProbe` | `distribution_version()`; `subprocess_output()` produces the data flow §5 JSON, an import-error payload (FX15) or a timeout (FX16); `probe()` decodes it with the production decoder. Refuses to run when PyTorch isn't installed (AC03). | Conforming |
| `FakePlatform` | `PlatformAdapter` | System facts; `getenv` returns the fixture's `Observed`, and a non-allowlisted name is a usage error. | Conforming |
| `FakeCommandRunner` | `CommandRunner` | Configurable executables (PATH map and extra-directory files); canned exit code, stdout and stderr; timeout and execution-failure simulation (`PermissionError`, `OSError`); exact and prefix argv matching, most specific wins; output cap; call log; unconfigured commands fail the test. Used for the future `nvidia-smi` and probe command interfaces. | Implemented (Stage 4) |
| `FixedClock` | `Clock` | Fixed `now_utc()`; `monotonic_ms()` advances by a fixed step. | Conforming |
| `FakeCollector` / `fake_collectors()` | `Collector` (`collectors/base.py`, DEC-44) | One per section, including the separate `pci` collector (section = the fixture's `pci_observations`, fallback = `PciCollector.unavailable`). `collect()` returns the fixture's `CollectorOutcome` (section, run, the collector's errors and the capability records it owns) or raises the fixture's configured exception (FX10); `unavailable(error: ErrorDetail)` for guard tests. `gpu_hardware` returns the whole `GPUInventory`, as the production collector does. | Conforming |
| `FakeAdapterSet` | — | `from_scenario()` bundles all fakes; `adapter_set()` returns the validated production `AdapterSet` with `FIXTURE` provenance. | Implemented |
| `FakeHostApi` | `HostApi` (seam under the production `OsPlatformAdapter`) | Synthetic Linux, WSL2, Windows, macOS and other-OS hosts, generated from `builders.SYSTEM_PROFILES` (Stage 5.1) so production on `FakeHostApi.for_profile(p)` yields exactly the fixture `system_info(p)`: `/proc` file contents including `/proc/sys/kernel/osrelease`, registry CPU name, `GlobalMemoryStatusEx` values, `mac_ver`, `sysctlbyname` values, CPU count, environment. `kernel(osrelease, proc_version)` swaps the kernel evidence for the WSL matrix. `failures` makes one primitive or file raise (`PermissionError`, `OSError`, …). A profile fails the test if the adapter calls another platform's API (no Linux paths on Windows, no Windows APIs on Linux). Records calls and environment reads. `FIXTURE` provenance makes the production adapter emit `fixture:` methods. | Implemented (Stage 5) |

**Conformance tests (Stage 4).**

| Test module | What it proves |
| ----------- | -------------- |
| `unit/test_adapter_contracts.py` | Every protocol has a fake. Each fake passes `isinstance` and has exactly the protocol's members and parameter names, kinds and defaults; a deliberately drifted class is detected. Across all 24 fixture variants, adapters return outcomes and never raise, NVML init failures propagate, `nvidia-smi` failures propagate and rows are complete, the PCI scan is gated to `linux_environment == NATIVE_LINUX` (not to the Linux profile), the probe agrees with the fixture, and the runner and `nvidia-smi` agree on availability. Also checks the `Clock` contract (fake and `SystemClock`) and `AdapterSet` validation. |
| `unit/test_nvml_lifecycle.py` | The §4.5.3 lifecycle: single init/shutdown, cleanup after exceptions, no shutdown after failed init, one shared session, refused concurrent session, usage errors after close, swallowed shutdown errors, handle failures isolated per device, fault status and unit mapping. |
| `unit/test_command_runner.py` | `FakeCommandRunner` capabilities, the fixed `nvidia-smi` argv, canned CSV with `[N/A]`/`[Not Supported]`/`[Insufficient Permissions]`, `require_success`, the WSL2 search path, and the probe through the runner. `subprocess.Popen` is patched to fail if anything tries to start a process. |
| `unit/test_adapter_outcomes.py` | The outcome mapping equals data model §3.3, row by row (skipped without `docs/`); the four failure kinds stay distinct; the probe decoder accepts the documented payload and rejects 14 malformed shapes. |
| `unit/test_boundaries.py` | Adapter contract modules import only `models` and pure stdlib; adapter implementations import only their declared OS modules (`platform_os.py`: `os`, `platform`, `sys`, `ctypes`, `winreg`) and never `pynvml`, `torch`, `subprocess` or network modules; collectors import only `models`, `adapters` and pure stdlib; `models`/`reporting` never import `adapters`; importing models, reporting, adapters and collectors loads no `pynvml`, `torch`, `subprocess`, `ctypes` or `socket`. |

**System and platform inspection tests (Stage 5).**

| Test module | What it proves |
| ----------- | -------------- |
| `unit/test_platform_adapter.py` | `OsPlatformAdapter` on each `FakeHostApi` profile: correct mechanism, source, method, unit and note per field. Unsupported platforms say `NOT_SUPPORTED` instead of guessing. Missing processor, CPU count, total or available memory → `NOT_DETECTED`, and `MemFree` is never substituted. Permission denied and I/O errors stay confined to their own fields; malformed `/proc/meminfo` → `PARSE_ERROR`; empty values are never fabricated; bugs propagate. Synthetic output always carries `fixture:`. `system_info()` reads no environment variable; `getenv` refuses non-allowlisted names before reading; the output serialises against the `SystemInfo` schema definition. Stage 5.1: macOS `sysctlbyname` decoding (unknown name, empty, invalid UTF-8, wrong size, zero, permission denied), and `StdlibHostApi.mac_sysctl` against a simulated libc `sysctlbyname`: argument and return types, the two-call sizing protocol, read-only arguments, errno mapping, missing libc. |
| `unit/test_linux_environment.py` (Stage 5.1) | The data model §4.1 classification rule, both directly and through the adapter: native (including Azure, cloud and KVM kernels), WSL1, WSL2 (including custom `+` builds), case variations, the old `-microsoft-standard` name and other ambiguous markers → `UNKNOWN`, conflicting releases or WSL1 + WSL2 text → `UNKNOWN`, empty, blank or malformed sources → `UNKNOWN`, missing, denied or I/O-failing evidence → `UNKNOWN`. Windows, macOS and other systems → `NOT_APPLICABLE` without reading `/proc`. Other fields are still collected when classification fails. Profile/classification consistency is enforced in the model and the schema. Parity: production on each synthetic host equals the fixture builder, and no fixture's `SystemInfo` field differs from production in status, source, method or unit. |
| `unit/test_system_collector.py` | `SystemCollector` reproduces every fixture's system section, run status and `SYSTEM.*` unsupported capabilities. Duration comes from `FixedClock`; the run is `PARTIAL` only for collection problems; `unavailable()` keeps the profile; `AdapterSet` refuses live/synthetic mixes. `collect_environment` reads exactly the allowlist (a planted secret is never read) and rejects non-allowlisted names before any read. |
| `integration/test_system_host.py` (`host`) | The real host through `OsPlatformAdapter()`: `LIVE` provenance, no `fixture:` methods; profile matches `sys.platform`; OS, Python and CPU facts equal the standard library's; memory is positive and available ≤ total; on Linux `MemTotal` agrees with `os.sysconf` within a page; `linux_environment` is consistent with `platform.release()` and the profile; schema-valid round trip; the collector runs `COMPLETED`/`PARTIAL` consistently; environment values match `os.environ` for allowlisted names. Process creation is patched to fail, proving no subprocess runs. Nothing assumes a machine model, OS version, RAM size or architecture, and nothing is compared with fixture data. |
| `unit/test_pci_collector.py` (Stage 6.3) | `PciCollector` (DEC-43) over the production adapter and `FakeSysfsReader`: every whole-scan and per-device mapping (collected, empty, `NO_DRIVER_BOUND`, driver-link failures, each omitted-device kind and status, missing/denied/failing root, six ineligible environments), deterministic error order, `ErrorDetail` fallback, clock-controlled duration, scanner/environment mismatch and out-of-contract statuses → `AdapterUsageError`, exceptions propagate, no collector-owned `FAILED`, schema-valid outputs. |
| `integration/test_pci_collector_fixtures.py` (Stage 6.4) | For all 24 variants: the collector reproduces the fixture's PCI section, `pci` run, `pci` errors and `GPU.PCI_SCAN` record (production adapter over `sysfs_reader_for`, and `FakePciScanner`, give identical outcomes); substituting its output into the snapshot changes nothing, JSON included; the `pci` fake collector matches it. FX03 (Intel only) and FX17 (`nouveau`) collected and `COMPLETED`; FX04b/FX11/FX12/FX13 `NOT_SUPPORTED` with the PCI record only, never `NVIDIA.ALL`. The catalogue pins no missing-root, empty or partial scan, so those stay unit-tested. |
| `unit/test_core_guard.py` (IMP-K08) | One `Collector` contract (all production collectors, the NVML step, every fake); the guard: valid outcomes pass through, `NOT_DETECTED`/`NOT_SUPPORTED` stay `COMPLETED`, an exception becomes one `FAILED` run with one host-wide `UNHANDLED_EXCEPTION` error, the fallback section and the guard's own timing; `AdapterUsageError` and contract violations (bare section, another collector's run, self-reported `FAILED`) contained and named; `KeyboardInterrupt`/`SystemExit` propagate. Fail-closed environment for every profile; the NVML step reproduces every fixture's `nvml` run; `FailedNvmlSession`; the production set, `DeferredPciScanner`, `production_pci_scanner` and the not-implemented stand-ins (schema-valid, never raise, can't enter a fixture set). |
| `unit/test_gpu_hardware_collector.py` (IMP-K02) | Parity with all 24 fixture inventories, runs and `NVIDIA.ALL` records (FX10 through the device-loss lifecycle; enumeration happens before the loss); an unreachable handle isolated with one `CollectionError`; fallback fields and live/fixture method labels; fallback-query and count failures reported only when they're problems; no scanner parameter and no PCI call; the fallback keeps the PCI section; `NVIDIA.ALL` on macOS only; exceptions propagate; schema-valid sections. |
| `unit/test_driver_collector.py` (IMP-K03) | Both driver runs reproduce every fixture's whole `driver` section, runs and errors; the query runs only when NVML has no driver version and `nvidia-smi` exists; system calls only; NVML authoritative with a disagreeing `nvidia-smi` kept and noted; query → header fallback, header CUDA with the fixed note; NVML's reason kept when nothing reports; no toolkit value (AC04); not installed is `COMPLETED`, a failed lookup or header `PARTIAL` with one error; NVML read failures field-level; guard fallbacks keep the other run's facts; the production adapter runs each command once. |
| `unit/test_runtime_collector.py` (IMP-K04) | Parity with every fixture's whole runtime section, run, errors and `RUNTIME.*` records; `include_processes=False` → `SKIPPED` with no process call (AC59); no inventory → no records; the `nvidia-smi` fallback reuses the query; a device missing from the query or reported lost by it; an unreachable handle isolated; a GPU lost mid-read stops its reads and keeps no partial values; field failures field-level; the guard fallback; capability reasons per platform and per device; the production adapter's conversions. |
| `unit/test_cuda_env_collector.py` (IMP-K05) | Parity with all 24 fixtures (section, run, errors, `CUDA.*`; FX10 through the guard); versions only from `version.json`/`nvcc`; determinism; the whole allowlist read, only toolkit variables as hints, an empty variable recorded but no hint; Windows `CUDA_PATH_V*`; no toolkit `COMPLETED`; a failed default listing one error and `PARTIAL`; a failing compiler field-level; `CUDA.TOOLKIT` on macOS and `OTHER` with nothing read or run; the fallback section; no file or process access by the collector; `OsPlatformAdapter.versioned_cuda_path_names()` (Windows only, filtered, no value read); the production locator on a synthetic Windows host. |
| `unit/test_toolkit_locator.py` (IMP-A06) | See the toolkit row of §3.1. |
| `unit/test_torch_probe.py` (IMP-A07) | See the torch probe row of §3.1. |
| `unit/test_correlation_collector.py` (IMP-K07) | Parity with all 24 fixtures (whole PyTorch section, run, errors); FX06 cross-ordered `cuda:0` → `nvml:1`, `cuda:1` → `nvml:0` by UUID through the `Inspector` (AC19); FX04a `NOT_APPLICABLE` with no error; UUID normalisation; PCI fallback when either side lacks a UUID (8- vs 4-digit domains); UUID precedence with a contradicting PCI noted; a PCI bus ID never overriding two different UUIDs; malformed UUIDs and PCI bus IDs unusable and never guessed; no identifier, same index, same name and memory never matched; duplicate UUIDs, duplicate PCI bus IDs and one device claimed twice ambiguous; NVML-only and PyTorch-only devices; an unidentifiable NVML device not ruled out without spoiling other matches; empty, `nvidia-smi` and missing inventories; order independence; live labels; no adapter parameter; the guard's fallback keeps the PyTorch facts; a failing correlation keeps every other section. |
| `unit/test_framework_collector.py` (IMP-K06) | Parity with all 24 fixtures (section, run, errors; `correlated_device_ref` the interim `NOT_APPLICABLE` value the correlation run replaces, DEC-49 (5)); no probe without a distribution (AC03); the opt-out reads and runs nothing (AC59); FX15 vs FX05 (AC15); the timeout message; exit and protocol failures keep the distribution; unreadable metadata is a failure; build, runtime answer and devices separate (AC18); CPU-only, ROCm, missing cuDNN, unavailable with a warning; an enumeration failure keeps everything else; missing attributes; no correlation; allocator `NOT_APPLICABLE` in subprocess mode and in-process without CUDA, collected with the scope note otherwise (AC16); a crashing probe guard-contained with the other sections intact. |
| `unit/test_device_loss.py` (KI-34) | Evidence that FX10's old faults contradict its own inventory; one GPU lost after enumeration reproduces FX10 end to end; several and all GPUs lost; lost before enumeration; lost during the driver observations; lost during the runtime observations; PCI kept, unique capability records, deterministic schema-valid reports; the production NVML adapter over the stub reproduces FX10 through the `Inspector`. |
| `integration/test_inspector.py` (IMP-K08/K02/P01, K03/K04) | The production core over every fixture's fakes: whole system, GPU, driver, runtime and CUDA environment sections, runs, errors and owned capability records match the fixture (all 24, FX10 included, its CUDA crash guard-contained); every report validates against the schema, round-trips and is byte-identical on a repeat run; NVML opened once and closed; one PCI scan. Scenarios: normal, no NVIDIA GPU, WSL2 gate, missing sysfs root, partial PCI scan, system failure (fail-closed `UNKNOWN`/`WSL2`), a pre-built native scanner refused, PCI exception, GPU hardware exception (PCI kept), multiple failures, the process opt-out, runtime and `nvidia-smi` exceptions contained, one owner per capability, duplicate rejection, Mode B refused. |
| `integration/test_inspect_host.py` (`host`) | `api.inspect()` with the production adapter set on this host: a valid, round-tripping `LIVE` report with no `fixture:` method, runs in order and none `FAILED`, PCI gated once with the policy reason, a GPU inventory that is either real (NVML or `nvidia-smi`) or the honest NVML outcome (never a stand-in, never a fabricated zero); driver fields real or the tool's own reason, and one runtime record per inventoried device; the CUDA environment real or `NO_TOOLKIT_FOUND`, the whole allowlist, no `CUDA.TOOLKIT` off macOS; `nvidia-smi` and `nvcc` the only processes allowed; guards prove no `/sys` access and that the only process allowed is `nvidia-smi`. Skipped on native Linux (HW-02). |
| `unit/test_subprocess_command_runner.py`, `unit/test_nvml_adapter.py`, `unit/test_nvidia_smi_adapter.py` | See the command runner, NVML and `nvidia-smi` rows of §3.1. |
| `integration/test_production_nvidia_pipeline.py` (IMP-A01/A02/A05) | The production NVML and `nvidia-smi` adapters behind the production `Inspector`, over `StubPynvml` and `FakeCommandRunner`: the fallback matrix of architecture §4.4 (NVML wins whether `nvidia-smi` works, fails or is absent; NVML failure + working CLI; partial NVML not merged; both failing → honest failure; zeros; both missing ≠ "no GPU"; missing binding), one session/one PCI scan/read-only calls, the header run once for `DriverInfo` and the query never when NVML counts, the driver and runtime sections from the fallback, telemetry read by the runtime run after the inventory, macOS gates and capability ownership, a GPU-collector exception (PCI kept, NVML still shut down), independent failures, a failing shutdown, and the report contract. |
| `integration/test_pci_host.py` (`host`, Stage 6.5) | The production PCI adapter (real reader) and collector with the environment classified by `OsPlatformAdapter()`. On every non-native host: `NOT_SUPPORTED` with the policy code and reason, `LIVE` method without prefix, `COMPLETED` run, only `GPU.PCI_SCAN`, schema-valid, and a guard proves nothing under `/sys` is opened, listed or resolved. A native Linux host is skipped: its real scan is HW-02. |

Host integration has run on Linux (WSL2) only; this host classifies as `WSL2`. The Windows
and macOS branches are covered by `FakeHostApi` unit tests (and a simulated libc for
`sysctlbyname`). Their real `winreg`/`ctypes`/`sysctlbyname` calls (`StdlibHostApi`) haven't
run on a Windows or macOS host yet, and native Linux, WSL1 and real ambiguous kernels have
only been exercised synthetically. The PCI host test (Stage 6.5) has likewise run only on
this WSL2 host, where the scan is correctly gated. A real native-Linux sysfs scan has never
run (HW-02). The public-API host test and `examples/inspect_demo.py` (the end-to-end
demonstration, run with `uv run python examples/inspect_demo.py [--json -|PATH]`) have run
on this WSL2 host only. Since IMP-A01/A02 they use the real NVML and `nvidia-smi` adapters:
here `nvmlInit` reports `NVML_ERROR_LIBRARY_NOT_FOUND` and `nvidia-smi` isn't found, so no
NVIDIA device has ever been inventoried live (HW-01 `BLOCKED`).

**What a fake can't prove.** These need the real implementations and are planned as unit
tests with patched libraries (next stages) and as hardware tests:

- ~~real `pynvml` exceptions being translated by the NVML adapter~~: now covered by
  `unit/test_nvml_adapter.py` (the stub's error values and messages are checked against the
  real binding) and, for `NVMLError_LibraryNotFound`, by the real binding on this host; real
  NVIDIA hardware remains untested (HW-01);
- ~~`subprocess.run` semantics~~: now covered by real child processes in
  `unit/test_subprocess_command_runner.py` (no shell, timeout kills the child, output cap);
- ~~the real probe child against a stub `torch` module~~: now covered by
  `unit/test_torch_probe.py` (the child run through the production runner with a stub
  `torch` package on `PYTHONPATH`);
- `SystemClock` beyond its contract (wall-clock accuracy isn't asserted).

No artificial test was written for the remaining items.

Fake adapters live only under `tests/`. The production package has no fixture code, no
demo mode and no flag that can produce `FIXTURE` data (DEC-17, AC12).

## 6. Hardware tests

Hardware tests verify behaviour on real systems and are the **only** way an item reaches
`HARDWARE_VALIDATED` in the tracker.

| ID | Purpose | Preconditions (skip reason if absent) | Pass criteria |
| -- | ------- | ------------------------------------- | ------------- |
| HW-01 | NVML inventory matches `nvidia-smi` | NVML init succeeds; `nvidia-smi` on PATH | Same device count; per device: name, UUID, driver version exactly equal; `memory.total` within 1 MiB. |
| HW-02 | PCI scan on bare-metal Linux | Native Linux (`linux_environment = NATIVE_LINUX`); NVIDIA GPU | Observation with `is_nvidia` and `kernel_driver = nvidia` (verifies C14). |
| HW-03 | Runtime snapshot sanity | NVML device | `0 ≤ util ≤ 100`; `used ≤ total`; `free ≤ total`; temperature 0–120 °C; power draw ≥ 0. Every field has a non-`UNKNOWN` status. |
| HW-04 | PyTorch probe and isolation | PyTorch with CUDA available | `cuda_available = true`; every PyTorch device correlated; NVML free memory sampled before the probe is recorded and compared with a sample after the probe (records C10/C11 findings; the main process has no CUDA context). |
| HW-05 | Toolkit detection | A CUDA Toolkit installed | Detected version equals `nvcc --version` (verifies C12). |
| HW-06 | PyTorch without local toolkit | PyTorch CUDA wheel; no toolkit | `cuda_available = true` and D03 shows the toolkit note (verifies D03 `[ASM]`). |
| HW-07 | WSL2 behaviour | WSL2 with Windows NVIDIA driver | `nvidia-smi` found at `/usr/lib/wsl/lib`; fields documented as unsupported report `NOT_SUPPORTED`, not failures. |
| HW-08 | Windows WDDM | Windows with NVIDIA GPU | Per-process memory `NOT_SUPPORTED` (EV-08). |
| HW-09 | Multi-GPU ordering | ≥ 2 NVIDIA GPUs | Correlation correct under default order, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and a `CUDA_VISIBLE_DEVICES` subset. |
| HW-10 | Performance | Any target platform | Record wall times for NFR-11 targets. |
| HW-11 | macOS | macOS machine | General info collected; NVIDIA sections `NOT_SUPPORTED`; exit 0. |
| HW-12 | Readiness on real memory | NVML device; ability to start a separate process that allocates GPU memory | Readiness switches from `READY` to `NOT_READY` once free memory drops below the threshold. |
| HW-13 | No-GPU machine run | Machine without NVIDIA driver (e.g. this WSL2 workstation, OBS-01) | Exit 0; D04 INFO; no traceback. (This validates graceful degradation on real hardware, not GPU behaviour.) |

Rules:
- Each hardware test checks its preconditions at runtime and calls `pytest.skip("<explicit
  reason, e.g. 'NVML unavailable: NVML_ERROR_LIBRARY_NOT_FOUND'>")`.
- Results are recorded in [app_tracker.md](app_tracker.md) §6 with date, machine, OS,
  driver, GPU model, PyTorch version and outcome.
- A skipped hardware test is **not** a pass.

## 7. Invariant tests (run across all fixtures)

1. `Observed.value is not None` ⇔ `status == COLLECTED` everywhere in the report (AC13).
2. No numeric `0` appears in the report for a field whose status isn't `COLLECTED`.
3. JSON contains no NaN/Infinity and validates against the schema (AC09).
4. Every finding obeys the diagnostics wording rules (diagnostics spec §5).
5. `data_origin == "FIXTURE"` for all fixture reports, and the text output contains the
   fixture-data banner (AC12).
6. Redacted output contains none of the fixture's sensitive literals (AC40).

## 8. Coverage and quality gates

| Gate | Target |
| ---- | ------ |
| Line coverage, `models`/`diagnostics`/`readiness`/`reporting` | ≥ 90 % |
| Line coverage, overall (ordinary suite) | ≥ 80 % |
| `ruff check` | clean |
| `scripts/check_dependencies.py` | passes |
| Type check (tool per OQ-08) | clean in strict mode |

**Coverage (Stage 3).** Measured with `pytest-cov` (branch coverage, `source =
gpu_system_inspector`). Subprocess-only entry points (`__main__.py` modules) aren't
measured.

| Measurement | Total (statements + branches) |
| ----------- | ----------------------------- |
| Baseline before Stage 3 tests (Stage 2 suite) | 93% |
| After Stage 3 | 95.19% |
| After Stage 4 (adapter contracts, all modules in `adapters/` at 100%) | 95.85% |
| After Stage 5 (`platform_os.py` 94%: only the real Windows/macOS host calls are unmeasured on Linux; `collectors/` 100%) | 95.61% |
| After Stage 5.1 (`platform_os.py` 96%: only the real Windows `winreg`/`ctypes` calls are unmeasured; macOS `sysctlbyname` is exercised against a simulated libc) | 95.77% |
| After Stage 6.1 (`pci_sysfs.py` 100%) | 96.04% |
| After Stage 6.2 (`collectors/capabilities.py` 100%) | 96.10% |
| After Stage 6.3 (`collectors/pci.py` 100%, statements and branches) | 96.20% |
| After IMP-K08/K02/P01 (`core/`, `api.py`, `collectors/base.py`, `collectors/gpu_hardware.py`, `reporting/assemble.py`, `adapters/unimplemented.py` all 100%) | 96.63% |
| After IMP-A05/A01/A02 (`adapters/commands.py`, `adapters/nvml.py`, `adapters/nvidia_smi.py` all 100%, statements and branches) | 97.17% |
| Enforced target (`fail_under`) | **92%** (raise towards the per-package targets above as stages add code) |

Known untested paths: defensive type-mismatch branches in `models/_validation.py` (89%),
and the CLI internal-error path (`cli/main.py` lines 156–165), which is exercised once the
core exists.

## 9. Acceptance traceability

Every AC in [acceptance_criteria.md](acceptance_criteria.md) names its verification
method: `UT` (unit), `IT` (integration), `HW` (hardware test ID), `MAN` (manual by the
user), or `REV` (document/code review).
