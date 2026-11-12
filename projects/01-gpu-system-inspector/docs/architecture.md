# Architecture — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. Interfaces below are design contracts for the
implementation; no code exists yet. Data structures are specified in
[data_model_specification.md](data_model_specification.md). Runtime ordering is in
[data_flow.md](data_flow.md).

## 1. Architectural principles

1. **Adapters isolate the outside world.** Only adapters touch NVML, subprocesses, the
   filesystem, environment variables, OS APIs or PyTorch. Everything above them is pure
   Python over normalised models, so it can be tested without hardware (NFR-10).
2. **Collectors normalise; engines reason; reporters render.** Collectors never draw
   conclusions. Diagnostics and readiness never perform I/O. Reporters never change data,
   except that redaction produces a redacted copy.
3. **Every value is an `Observed[T]`.** Value, status, source, method and unit travel
   together (DEC-10). There are no bare optional numbers in the models.
4. **Isolation by default.** The core runs each collector inside a guard. A failure
   becomes data (`CollectionError`) instead of an exception (FR-13).
5. **Read-only and offline.** No mutating NVML calls, no network, no installs (NFR-12, NFR-15).
6. **Project-owned.** All code lives in this project. Reuse candidates are recorded (§8)
   but not extracted (`docs/package-policy.md`, extraction policy).

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. CLI  (argparse, exit codes, output routing)                           │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. Inspection core  (Inspector, InspectionOptions, CollectionContext,     │
│                      collector guard, ordering)                          │
├───────────────┬───────────────┬──────────────┬──────────────┬────────────┤
│ 3. System     │ 4. GPU        │ 6. CUDA env  │ 7. Framework │ Runtime    │
│ collector     │ hardware +    │ collector    │ collector    │ metrics    │
│               │ driver colls. │              │ (PyTorch)    │ collector  │
├───────────────┴───────┬───────┴──────────────┴──────┬───────┴────────────┤
│ 5. Hardware & driver adapters                        │ 7. PyTorch adapter │
│   NvmlAdapter · NvidiaSmiAdapter · PciSysfsAdapter   │   TorchProbe       │
│   PlatformAdapter · CommandRunner · ToolkitLocator   │   (subprocess /    │
│                                                      │    in-process)     │
├──────────────────────────────────────────────────────┴────────────────────┤
│ Normalised inspection models  (InspectionSnapshot)                        │
├────────────────────────────────┬──────────────────────────────────────────┤
│ 9. Readiness engine            │ 8. Diagnostics engine (rules D01–D08)    │
├────────────────────────────────┴──────────────────────────────────────────┤
│ 10. Reporting  (report assembly · redaction · JSON · text · file writer)  │
└──────────────────────────────────────────────────────────────────────────┘
```

Allowed dependency direction inside the package: `cli → core → {collectors, readiness,
diagnostics, reporting} → models`, and `collectors → adapters → models`. `models` depends
on nothing internal. `diagnostics`, `readiness` and `reporting` never import `adapters`.

**Layer roles (Stage 4).** Five roles stay distinct:

| Role | Location | Responsibility | Never |
| ---- | -------- | -------------- | ----- |
| Adapter interfaces | `adapters/protocols.py`, `outcomes.py`, `probe_protocol.py`, `adapter_set.py` | Define operations, result types and outcome rules (§4.5) | Touch NVML, processes, the OS or PyTorch |
| Adapter implementations | `adapters/<mechanism>.py` (later stages); fakes in `tests/gsi_testkit/fakes/` | Wrap one external mechanism and map its errors to outcomes | Combine sources or draw conclusions |
| Collectors | `collectors/` | Turn adapter outcomes into section models (`SystemInfo`, `GPUInventory`, …), including fallbacks and correlation | Open or close NVML, call third-party libraries directly |
| Inspection orchestration | `core/` | Build the context, own the NVML session, run collectors under the guard in order, time them | Interpret data |
| Diagnostics and readiness | `diagnostics/`, `readiness/` | Pure reasoning over the snapshot | Perform I/O or import adapters |

## 3. Proposed module layout (created at implementation time, not now)

```
projects/01-gpu-system-inspector/
├── pyproject.toml                 # becomes a real package at implementation (§9)
├── README.md
├── docs/                          # this specification
├── src/gpu_system_inspector/
│   ├── __init__.py                # __version__, __all__ (provisional public API)
│   ├── __main__.py                # python -m gpu_system_inspector inspect …
│   ├── py.typed
│   ├── api.py                     # inspect() (implemented, DEC-44)
│   ├── cli/            main.py, commands.py, exit_codes.py
│   ├── core/           inspector.py, guard.py, environment.py, factory.py
│   │                   (IMP-K08, DEC-44: implemented; no context.py/ordering.py)
│   ├── models/         observed.py, enums.py, system.py, gpu.py, driver.py, cuda.py,
│   │                   framework.py, runtime.py, diagnostics.py, readiness.py, report.py,
│   │                   collection.py, _validation.py   (Stage 2: implemented)
│   ├── adapters/       protocols.py, outcomes.py, probe_protocol.py, adapter_set.py,
│   │                   clock.py   (Stage 4: interfaces + SystemClock implemented)
│   │                   platform_os.py   (Stage 5: production PlatformAdapter)
│   │                   pci_sysfs.py   (Stage 6.1: production PciScanner)
│   │                   unimplemented.py   (DEC-44: not-implemented stand-ins)
│   │                   commands.py, nvml.py, nvidia_smi.py   (IMP-A05/A01/A02, DEC-45)
│   │                   toolkit.py   (IMP-A06, DEC-47: CUDA Toolkit locator)
│   │                   torch_probe.py, torch_probe_payload.py,
│   │                   _torch_probe_main.py   (IMP-A07, DEC-48)
│   ├── collectors/     system.py, environment.py   (Stage 5: implemented)
│   │                   capabilities.py   (Stage 6.2: PCI/NVIDIA capability policy)
│   │                   pci.py   (Stage 6.3: PciCollector, DEC-43)
│   │                   base.py, gpu_hardware.py   (IMP-K02, DEC-44: implemented)
│   │                   driver.py, runtime.py   (IMP-K03/K04, DEC-46: implemented)
│   │                   cuda_env.py   (IMP-K05, DEC-47: implemented)
│   │                   framework.py   (IMP-K06, DEC-48: implemented)
│   │                   correlation.py   (IMP-K07, DEC-49: implemented)
│   ├── diagnostics/    engine.py, evidence.py, rules/d01_…d08_….py
│   ├── readiness/      engine.py
│   └── reporting/      assemble.py, redaction.py, json_report.py, text_report.py,
│                       writer.py, schema.py, __main__.py,
│                       schema/inspection_report.schema.json
│                       (Stage 2: json_report.py, schema.py, __main__.py and the schema;
│                        IMP-P01: assemble.py)
├── examples/           inspect_demo.py   (end-to-end demonstration; not shipped)
└── tests/
    ├── unit/  integration/  hardware/
    └── gsi_testkit/    builders, FX fixtures, expectations, fake adapters (data_origin = FIXTURE)
```

## 4. Components

Each component lists responsibility, inputs, outputs, dependencies, failure behaviour,
public interface and testing approach. Signatures are normative intent; exact names may be
refined during implementation, provided the tracker records the change.

### 4.1 CLI (`cli/`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Parse arguments, validate input, build `InspectionOptions`, configure logging, call the core, render the text report to stdout, write the JSON report when `--json` is given (file or stdout), map results to exit codes. |
| Inputs | `argv`, process environment (only for logging configuration), stdout/stderr streams. |
| Outputs | Text report on stdout (or the JSON report with `--json -`); JSON file with `--json PATH`; log lines on stderr; exit code. |
| Dependencies | `argparse`, `logging`, `core`, `reporting`. |
| Failure behaviour | Invalid usage → message on stderr, exit `2`, no inspection. JSON file write failure → text report still printed, exit `4`. Unexpected exception → one-line stderr message, exit `5` (traceback with `-vv`). `KeyboardInterrupt` → exit `130`. |
| Public interface | `main(argv: Sequence[str] \| None = None) -> int`; console script `inferforge-gpu` (DEC-03, approved); `python -m gpu_system_inspector`. |
| Testing | Integration tests calling `main([...])` with fake adapters injected through the core factory; assert stdout/stderr/exit code ([cli_specification.md](cli_specification.md)). |

### 4.2 Inspection core (`core/`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Own the execution order, run each collector through the guard, own the NVML session, assemble the `InspectionSnapshot`, then call (when they exist) readiness and diagnostics, and report assembly. **Implemented (IMP-K08, DEC-44)** for the `system`, `pci`, `nvml` and `gpu_hardware` steps. |
| Inputs | `InspectionOptions` and an `AdapterSet` (real or fake; its `clock` is the only clock). There is no `CollectionContext`: collectors get their adapters and the clock in the constructor. |
| Outputs | `InspectionReport` (unredacted; redaction is applied by reporting when requested). |
| Dependencies | `collectors`, `reporting.assemble`, `adapters`, `models`; later `readiness` and `diagnostics`. |
| Failure behaviour | The guard (`core/guard.py`, `run_guarded`) is the only producer of `FAILED`: it catches *any* `Exception` from a collector (including `AdapterUsageError` and contract violations), records one host-wide `CollectionError` (`status = COLLECTION_FAILED`, `error_code = "UNHANDLED_EXCEPTION"`, message `Type: message`, the collector's report path), substitutes `unavailable(ErrorDetail)` and times the run with the injected clock. `KeyboardInterrupt`/`SystemExit` propagate. Exceptions in the core itself propagate to the CLI (exit `5`). |
| Public interface | `Inspector(adapters: AdapterSet, *, pci_scanner_factory=None, limitations=()).run(options: InspectionOptions \| None = None) -> InspectionReport` and `.collect(options) -> InspectionSnapshot`; `api.inspect(options: InspectionOptions \| None = None) -> InspectionReport` (production adapters). Mode B raises `NotImplementedError` until IMP-R01. |
| Testing | Unit tests for the guard, contract, environment resolution and NVML step (`unit/test_core_guard.py`); integration tests over all fixture scenarios and the failure scenarios (`integration/test_inspector.py`); the public API on the host (`integration/test_inspect_host.py`). |

```python
@dataclass(frozen=True, slots=True)
class InspectionOptions:
    required_vram_gib: float | None = None  # None => Mode A
    include_pytorch: bool = True
    include_processes: bool = True
    redact: bool = False
    subprocess_timeout_s: float = 10.0
    pytorch_timeout_s: float = 60.0
    pytorch_probe_mode: Literal["SUBPROCESS", "IN_PROCESS"] = "SUBPROCESS"  # CLI always SUBPROCESS


@runtime_checkable
class Collector(Protocol[SectionT]):  # collectors/base.py (DEC-44)
    name: str  # e.g. "gpu_hardware"; the collector_runs and CollectionError.collector key

    def collect(self) -> CollectorOutcome[SectionT]: ...  # COMPLETED or PARTIAL, never FAILED
    def unavailable(self, error: ErrorDetail) -> SectionT: ...  # the guard's fallback
```

**One contract (DEC-44).** `SystemCollector`, `PciCollector`, `GpuHardwareCollector`, the
core's `NvmlSessionStep` and the test kit's `FakeCollector` all satisfy `Collector`. Adapters
and the `Clock` are constructor arguments (there is no `CollectionContext`). `unavailable`
takes an `ErrorDetail`, because the fallback section's `Observed.error` can only hold one.
This replaces the earlier `collect(ctx) -> SectionT` / `unavailable(CollectionError)` sketch.

**Order** (`COLLECTOR_ORDER` in the test kit): `system`, `pci`, `nvml`, `driver`,
`gpu_hardware`, `runtime`, `nvidia_smi` (implemented, DEC-46), `cuda_environment`
(implemented, DEC-47), `pytorch` (implemented, DEC-48) and `correlation` (implemented, DEC-49). The core resolves the Linux environment from the system section, fail-closed
(`core/environment.py`: collected value, else `LINUX` → `UNKNOWN`, `WSL2` → `WSL2`, otherwise
`None`), builds the PCI scanner from it (`pci_scanner_factory`) and passes the same value to
`PciCollector` and `GpuHardwareCollector`. The `nvml` step opens the one NVML session (run
`PARTIAL` with a `CollectionError` for a missing binding, permission or failure; `COMPLETED`
for `NOT_DETECTED`/`NOT_SUPPORTED`); `driver`, `gpu_hardware` and `runtime` share it; the
core closes it after `runtime` on every path, and substitutes `FailedNvmlSession` when
opening it raised. `nvidia_smi` runs after the session is closed and completes the
`driver` run's section (DEC-46). Every report section now has a collector (DEC-44 to
DEC-49), so no section uses the DEC-44 (7) placeholder any more; diagnostics and readiness
are separate engines, not implemented, and a limitation says so. Report assembly (§4.10) rejects a collector that ran twice or a capability
record with two owners.

### 4.3 System collectors (`collectors/system.py`)

**Implemented in Stage 5 (2026-11-04)**: `adapters/platform_os.py` (`OsPlatformAdapter`,
`StdlibHostApi`, `HostApi`), `collectors/system.py` (`SystemCollector`) and
`collectors/environment.py` (`collect_environment`). This is the first production
inspection capability. It isn't wired into `inferforge-gpu inspect` yet, because that
needs the orchestration core.

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Produce `SystemInfo`: OS, release, version, distribution, architecture, processor model, logical CPUs, total and available physical RAM, Python details, WSL flag, and the platform profile. "OS family" is `platform_profile`; there is no separate platform-identifier field (`os_name` + `architecture` serve that role). Physical core counts are not collected. |
| Inputs | `PlatformAdapter` and `Clock`. `OsPlatformAdapter` reads the host through `HostApi`, a thin seam of primitive stdlib calls, so every platform branch is unit-testable on any machine. |
| Outputs | `CollectorOutcome[SystemInfo]`: the section, a `CollectorRun` (duration from the injected `Clock`), and `UnsupportedCapability` records for fields that are `NOT_SUPPORTED` (`SYSTEM.PROCESSOR_MODEL`, `SYSTEM.MEMORY_TOTAL`, `SYSTEM.MEMORY_AVAILABLE`, `SYSTEM.OS_DISTRIBUTION`, …). |
| Dependencies | Stdlib only (DEC-06): `platform`, `os`, `sys`; `winreg` and `ctypes` imported lazily on Windows only. No subprocess. |
| Failure behaviour | Each field independently (one `/proc` read failing never affects OS or Python facts). Empty or absent value → `NOT_DETECTED`/`FIELD_NOT_FOUND` (EV-21, OBS-02); missing file → `NOT_DETECTED`; `PermissionError` → `PERMISSION_DENIED`; other `OSError` → `COLLECTION_FAILED`; malformed content → `COLLECTION_FAILED`/`PARSE_ERROR`; not implemented on this platform → `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`. The run is `PARTIAL` only for `COLLECTION_FAILED`, `PERMISSION_DENIED` or `UNKNOWN` fields. Field failures don't create `CollectionError` records (data model §3.4). Other exceptions are bugs and propagate to the collector guard. |
| Public interface | `SystemCollector(platform, clock).collect() -> CollectorOutcome[SystemInfo]`; `unavailable(error) -> SystemInfo`; `collect_environment(platform, names=ENVIRONMENT_ALLOWLIST) -> dict[str, Observed[str]]`. |
| Testing | Unit tests over `FakeHostApi` profiles (Linux, WSL2, Windows, macOS, other) and over `FakePlatform` for all 24 fixture variants; host integration tests (`host` marker) on the real machine. |

**Platform mechanisms.**

| Field | Linux / WSL2 | Windows | macOS | Other |
| ----- | ------------ | ------- | ----- | ----- |
| `os_name`, `os_release`, `os_version`, `architecture` | `platform.system/release/version/machine()` | same | same | same |
| `os_distribution` | `/etc/os-release` `PRETTY_NAME` (fallback `/usr/lib/os-release`) | `platform.win32_ver()` + `win32_edition()` | `platform.mac_ver()` | `NOT_SUPPORTED` |
| `processor_model` | `/proc/cpuinfo` `model name` (none on many ARM boards → `NOT_DETECTED`) | registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString` (read-only) | `sysctlbyname("machdep.cpu.brand_string")` | `NOT_SUPPORTED` |
| `logical_cpu_count` | `os.cpu_count()` (system-wide; the process may be restricted to fewer) | same | same | same |
| `memory_total_bytes` | `/proc/meminfo` `MemTotal` × 1024 | `GlobalMemoryStatusEx.ullTotalPhys` | `sysctlbyname("hw.memsize")` | `NOT_SUPPORTED` |
| `memory_available_bytes` | `/proc/meminfo` `MemAvailable` × 1024; never `MemFree` | `GlobalMemoryStatusEx.ullAvailPhys` | `NOT_SUPPORTED` (no sysctl equivalent; V1 scope) | `NOT_SUPPORTED` |
| `python_*` | `platform.python_version/implementation()`, `sys.executable` | same | same | same |
| `linux_environment` | classified from `/proc/sys/kernel/osrelease` + `/proc/version` (rule in data model §4.1) | `NOT_APPLICABLE` | `NOT_APPLICABLE` | `NOT_APPLICABLE` |

Memory values are physical memory in bytes, never swap, page file, virtual memory or process
memory. Each collected value carries a `note` stating its exact meaning, and inside WSL2 the
note says the values describe the WSL2 virtual machine, not the Windows host.
`platform.processor()` is deliberately not used: it runs `uname -p` on POSIX and reads the
non-allowlisted `PROCESSOR_IDENTIFIER` variable on Windows.

**macOS (Stage 5.1).** `StdlibHostApi.mac_sysctl(name)` calls libc
`int sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen)`
through `ctypes` (signature and value types checked against Apple's `sysctl(3)` manual:
`hw.memsize` is `int64_t`, `machdep.cpu.brand_string` is `char[]`). It's read-only: `newp`
is `NULL` and `newlen` 0. A first call with `oldp = NULL` sizes the buffer, and a second
fills it. A -1 return raises `OSError(errno)`, so `ENOENT` becomes `NOT_DETECTED` and
`EPERM`/`EACCES` become `PERMISSION_DENIED`. The adapter decodes strictly: a NUL-terminated
UTF-8 string, and an exactly 8-byte native-order signed integer that must be positive;
anything else is `PARSE_ERROR`. No subprocess, no new dependency. `ctypes` is imported only
on that path, and libc is located with `ctypes.util.find_library("c")`, which uses dyld on
macOS (no subprocess there). This replaces the Stage 5 deferral (DEC-37, superseded by
DEC-40).

**Linux environment classification (Stage 5.1, DEC-39).** The binary `is_wsl` field, which
was true whenever `/proc/version` contained "microsoft", is replaced by
`linux_environment` (`NATIVE_LINUX`, `WSL1`, `WSL2`, `UNKNOWN`; `NOT_APPLICABLE` off
Linux), using the conservative rule in data model §4.1. Only kernel-generated evidence is
used, because files left behind by a removed distribution (`/etc/wsl.conf`, `/usr/lib/wsl`)
can be residual. The profile is `WSL2` only for a `WSL2` classification; `WSL1` and
`UNKNOWN` are profile `LINUX` but are not native, so native-only capabilities (Stage 6 PCI
scan) must test `linux_environment == NATIVE_LINUX`. Unreadable, empty, malformed or
conflicting evidence gives `UNKNOWN`, with the reason in `note`. The rest of the section is
still collected.

**Provenance.** `OsPlatformAdapter.provenance` comes from its `HostApi`: `LIVE` for
`StdlibHostApi`, `FIXTURE` for the test kit's `FakeHostApi`. With `FIXTURE`, every method
gets the `fixture:` prefix (`FIXTURE_METHOD_PREFIX`, now defined in `models`). `AdapterSet`
rejects adapters whose provenance differs from the set's, so live and synthetic adapters
can't be mixed.

### 4.4 GPU hardware collectors (`collectors/gpu_hardware.py`, `collectors/driver.py`, `collectors/pci.py`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Produce `GPUInventory` (device list, count, PCI observations) from NVML, with the `nvidia-smi` fallback (**implemented**, IMP-K02, DEC-44), `DriverInfo` (**implemented**, IMP-K03, DEC-46) and the per-device `RuntimeMetrics` (**implemented**, IMP-K04, `collectors/runtime.py`, DEC-46). The PCI scan is **not** done here: `PciCollector` (`collectors/pci.py`, DEC-43) owns it, and `GpuHardwareCollector` puts its section into `GPUInventory.pci_observations` unchanged. |
| Inputs | `NvmlSession` (opened and closed by the core), `NvidiaSmiAdapter`, the PCI section produced by `PciCollector`, and the resolved `linux_environment` and profile (for `NVIDIA.ALL`). |
| Outputs | `GPUInventory`, `DriverInfo`, `RuntimeMetrics` per inventoried device. |
| Dependencies | Adapters only. |
| Failure behaviour | NVML can count devices → one `GPUDeviceInfo` per index (`name`, `uuid`, `pci_info()`, `memory_info().total`, `compute_capability`, `mig_mode`); an unreachable handle → that device's `unavailable()` section plus one `CollectionError` (`device_ref = nvml:<i>`, `/gpu/devices/<i>`), other devices unaffected (AC54). NVML can't count → the `nvidia-smi` query: collected rows give `smi:<i>` devices (fields the query lacks are `NOT_SUPPORTED`/`FALLBACK_FIELD_UNAVAILABLE`), otherwise `devices = []` with `device_count` = the NVML outcome (never `0` unless a source reported zero). Run `PARTIAL` for any `CollectionError` (failed device; a count failure after a good init, `/gpu/device_count`; a fallback query that failed rather than was absent, `/gpu/devices`) or a device field with `COLLECTION_FAILED`/`PERMISSION_DENIED`/`UNKNOWN`. `unavailable()` keeps the PCI section, because another collector produced it (FR-13). |
| Public interface | `GpuHardwareCollector(session, nvidia_smi, clock, *, pci, environment, profile)` (`Collector[GPUInventory]`, emits `NVIDIA.ALL` on macOS); `PciCollector(scanner, clock, *, environment, profile)` (`Collector[Observed[tuple[PciObservation, ...]]]`); `DriverCollector(nvml, session, clock)` and `NvidiaSmiCollector(nvidia_smi, clock, *, driver)` (`Collector[DriverInfo]`); `RuntimeCollector(session, nvidia_smi, clock, *, inventory, profile, include_processes)` (`Collector[tuple[RuntimeMetrics, ...]]`, owns `RUNTIME.*`). |

**NVML / `nvidia-smi` policy (DEC-45, refining DEC-04).** The two paths are independent
adapters; only the GPU hardware collector combines them, by precedence, never by merging:

| NVML | `nvidia-smi` | Inventory |
| ---- | ------------ | --------- |
| counts devices (any number, including 0) | not run | NVML (`nvml:<i>`); a device NVML can't reach is that device's failure, not a reason to ask `nvidia-smi` |
| can't count (not installed, library missing, driver not loaded, denied, failed) | rows collected (including none) | `nvidia-smi` (`smi:<i>`, its own zero counts) |
| can't count | not found / not supported | none: `device_count` = the NVML outcome (e.g. `NOT_DETECTED`/`NVML_ERROR_LIBRARY_NOT_FOUND`, about the library, not the GPU), no error |
| can't count | failed (exit, timeout, unparsable) | none: `device_count` = the NVML outcome, plus a `gpu_hardware` `CollectionError` (`/gpu/devices`), run `PARTIAL` |

Indices are never matched across the two tools, so no device can appear twice; both keep
their source in every value. A successful empty inventory comes only from a source that
counted zero.

**Driver collectors (IMP-K03, DEC-46).** `DriverInfo` is built in two runs. `driver` (after
`nvml`) reads the NVML facts from the shared session: `nvml_binding` and
`nvml_binding_distribution` (package metadata), `nvml_version` (`init_result`),
`driver_version`, `driver_supported_cuda_version` (with the fixed note) and
`cuda_driver_library` (`nvmlSystemGetCudaDriverVersion_v2`, the libcuda version). It never
opens NVML and calls only system functions. `nvidia_smi` (after `runtime`, session closed)
adds `nvidia_smi_path` (`locate()`), `nvidia_smi_driver_version` and
`nvidia_smi_cuda_version` (`version()`, the plain run's header), and fills the gaps by a
fixed precedence:

| Field | Precedence |
| ----- | ---------- |
| `driver_version` | NVML → the `nvidia-smi` query's `driver_version` (the cached rows) → the `nvidia-smi` header → NVML's own reason |
| `driver_supported_cuda_version` | NVML → the `nvidia-smi` header, with `DRIVER_SUPPORTED_CUDA_NOTE` → NVML's own reason |
| `nvidia_smi_*` | always what `nvidia-smi` said; when it disagrees with NVML, the value is kept and gets the note "Differs from NVML, which reports X." |

The query runs only if NVML gave no driver version and `nvidia-smi` was found; the adapter
caches it, so after the GPU hardware fallback it isn't run again. Runs: `driver` is
`PARTIAL` (no error, field-level) when NVML initialised but a driver read has a failure
status; without a session every read repeats the `nvml` run's outcome, already reported.
`nvidia_smi` is `PARTIAL` with one `CollectionError` for a lookup that failed
(`/driver/nvidia_smi_path`) or a found executable whose header failed
(`/driver/nvidia_smi_driver_version`); not installed or macOS is a fact, `COMPLETED`.
`unavailable()`: `driver` → `DriverInfo.unavailable`; `nvidia_smi` keeps the NVML facts and
fails only its three fields. Neither emits capability records. The CUDA Toolkit is never
read here (AC04); the CUDA runtime (`libcudart`) and CUDA usability aren't observed by any
collector yet (PyTorch probe, IMP-A07/K06), and the report's limitation says so.

**Runtime metrics collector (IMP-K04, DEC-46).** One `RuntimeMetrics` per inventoried device,
from the inventory's own source. NVML inventory: the device's handle again, then
`memory_info`, `utilization`, `temperature`, `slowdown_threshold`, `power_draw`,
`power_limit`, `processes` (`SKIPPED`, with no process calls, when `include_processes` is
false, AC59); `sampled_at` from the `Clock`. `nvidia-smi` inventory: the cached query rows
(fields it lacks: `memory_reserved_bytes`, `temperature_slowdown_threshold_celsius`,
`processes` → `NOT_SUPPORTED`/`FALLBACK_FIELD_UNAVAILABLE`); a device missing from the rows
→ unavailable plus `DEVICE_NOT_IN_QUERY`. No inventory → `[]`.

| Situation | Record | `CollectionError` | Run |
| --------- | ------ | ----------------- | --- |
| Handle can't be acquired | `RuntimeMetrics.unavailable(ref, at, status, error)` | one, `device_ref`, `/runtime/<i>` | `PARTIAL` |
| Any read reports `NVML_ERROR_GPU_IS_LOST` (or `[GPU is lost]`) | unavailable with that error; no further calls on the device; values read before the loss are dropped | one, `/runtime/<i>` | `PARTIAL` |
| Other read failure | field-level | none | `PARTIAL` for `COLLECTION_FAILED`/`PERMISSION_DENIED`/`UNKNOWN` |

Capability records (one per device, NVML devices only): `RUNTIME.GPU_UTILIZATION` for
`NOT_SUPPORTED` utilization; `RUNTIME.PROCESSES` for `NOT_SUPPORTED` or `PERMISSION_DENIED`
process enumeration (with that status); `RUNTIME.PROCESS_MEMORY` for a listed process whose
memory is `NOT_SUPPORTED`. Reasons: WSL2 (EV-09) and WDDM on Windows (EV-08) use their
documented notes, which are also put on the fields; MIG enabled → "Not reported while MIG
is enabled."; otherwise a plain NVML statement. Temperature or power gaps are field-level
only. `unavailable()` gives every inventoried device an unavailable record.

**PCI collector (Stage 6.3, DEC-43).** `PciCollector` (name `pci`) passes the scanner's
`Observed` section through unchanged and adds the bookkeeping. Its status mapping:

| Scan result | Run | `CollectionError` | Unsupported capability |
| ----------- | --- | ----------------- | ---------------------- |
| `COLLECTED`, no failures (including `()`), or only `NO_DRIVER_BOUND` drivers | `COMPLETED` | none | none |
| `COLLECTED` with a `DRIVER_LINK_UNREADABLE` failure (device kept, `kernel_driver` failed) | `PARTIAL` | none (field-level) | none |
| `COLLECTED` with an omitted device (`INVALID_ADDRESS`, `ATTRIBUTE_UNREADABLE` of any status, `ATTRIBUTE_MALFORMED`) | `PARTIAL` | one per omitted device | none |
| `NOT_DETECTED` (no PCI sysfs root) | `COMPLETED` | none | none |
| `NOT_SUPPORTED` (ineligible environment) | `COMPLETED` | none | `GPU.PCI_SCAN` only (from `unsupported_capabilities_for`) |
| `PERMISSION_DENIED` / `COLLECTION_FAILED` (root unreadable) | `PARTIAL` | one, host-wide | none |

Every `CollectionError` has `collector = "pci"`, `device_ref = null` (a PCI address isn't a
`device_ref`) and `report_path = "/gpu/pci_observations"`. A device error copies the
failure's status and code, with the message `<address> <attribute or 'address'>:
<error.message>`, in the scan's `(address, attribute)` order. The collector never reports
`FAILED`: exceptions propagate to the guard (§4.2). It raises `AdapterUsageError`
when the scan contradicts the supplied environment (an eligible environment reported
`NOT_SUPPORTED`, or an ineligible one that isn't `NOT_SUPPORTED` with the policy code and
reason), or reports a status outside this table. `NVIDIA.ALL` isn't emitted here.

| Testing | Unit tests with `FakeNvml` (configurable per-call results and exceptions) and canned `nvidia-smi` CSV outputs, including `[N/A]`, `[Not Supported]` and `[Insufficient Permissions]`. |

### 4.5 Hardware and driver adapters (`adapters/`)

Stage 4 (2026-11-04) defined the adapter **interfaces** in code and validated them with the
test fakes. Production adapters so far: `SystemClock` (Stage 4), `OsPlatformAdapter`
(Stage 5, §4.3), `PciSysfsAdapter` (Stage 6.1), `SubprocessCommandRunner` (IMP-A05),
`PynvmlAdapter` (IMP-A01), `NvidiaSmiCli` (IMP-A02) (§4.5.5, DEC-45) and
`CudaToolkitLocator` (IMP-A06, §4.6, DEC-47) and `SubprocessTorchProbe` / `InProcessTorchProbe`
(IMP-A07, §4.7, DEC-48). Interfaces live in
`src/gpu_system_inspector/adapters/`: `protocols.py` (the protocols and their result types),
`outcomes.py` (outcome mapping, `CommandResult`, `AdapterUsageError`),
`probe_protocol.py` (the PyTorch probe protocol) and `adapter_set.py` (`AdapterSet`). They
depend only on `models` and pure stdlib, and a test enforces that.

#### 4.5.1 Interface catalogue

Every operation is read-only. "Mandatory" means the adapter is always part of an
`AdapterSet`. Every adapter is mandatory in that sense, and "optional" describes the
*capability* behind it, which may be absent without being an error.

| Protocol | Operations → output | Lifecycle | Capability | Platform availability |
| -------- | ------------------- | --------- | ---------- | --------------------- |
| `NvmlAdapter` | `binding_version()`, `binding_distribution()` → `Observed[str]` (no `nvmlInit`); `open()` → `NvmlSession` | Creates sessions; at most one open at a time | Required dependency (DEC-05), but NVML itself may be unusable | All; NVML is absent on macOS (`NOT_DETECTED`/`NOT_SUPPORTED`) |
| `NvmlSession` | `init_result` (the NVML accessibility outcome, with the NVML version); `driver_version()`, `cuda_driver_version()`, `cuda_driver_library()` → `Observed[str]`; `device_count()` → `Observed[int]`; `device(index)` → `Observed[NvmlDevice]`; `close()`; context manager | §4.5.3 | — | — |
| `NvmlDevice` | `index`; `name()`, `uuid()`, `compute_capability()`, `mig_mode()` → `Observed[str]`; `pci_info()` → `NvmlPciInfo`; `memory_info()` → `NvmlMemoryInfo` (one read, so total/used/free are consistent); `utilization()` → `NvmlUtilization`; `temperature()`, `slowdown_threshold()` → `Observed[int]`; `power_draw()`, `power_limit()` → `Observed[float]`; `processes()` → `Observed[tuple[GpuProcess, ...]]` | Valid only while its session is open | — | Per-field `NOT_SUPPORTED` (WSL2, WDDM, older GPUs) |
| `NvidiaSmiAdapter` | `locate()` → `Observed[str]`; `version()` → `SmiVersion`; `query_gpus()` → `Observed[tuple[SmiGpuRow, ...]]` (every requested field present, partial rows allowed) | Stateless | Optional tool | All with an NVIDIA driver; WSL2 adds `/usr/lib/wsl/lib` to the search |
| `PciScanner` | `scan()` → `PciScan(observations: Observed[tuple[PciObservation, ...]], failures: tuple[PciDeviceFailure, ...])` (Stage 6.1) | Stateless | Optional evidence | `linux_environment == NATIVE_LINUX` only; WSL1, WSL2 and non-Linux → `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`, `UNKNOWN` → `NOT_SUPPORTED`/`LINUX_ENVIRONMENT_UNKNOWN`, all without touching sysfs |
| `TorchProbe` | `distribution_version()` → `Observed[str]` (never imports torch); `probe(timeout_s)` → `Observed[TorchProbeReport]` | Each `probe()` starts and finishes one child process | Optional framework (DEC-26). Callers must not probe when it's `NOT_INSTALLED` or skipped | All |
| `PlatformAdapter` | `profile` → `PlatformProfile`; `system_info()` → `SystemInfo`; `getenv(name)` → `Observed[str]` (allowlisted names only, unset → `NOT_DETECTED`) | Stateless | Required | All (per-field `NOT_SUPPORTED`, e.g. available RAM on macOS) |
| `CommandRunner` | `which(name, *, extra_paths)` → `Observed[str]`; `run(argv, *, timeout_s, source)` → `Observed[CommandResult]` | Each `run()` is self-contained; no process outlives it | Required | All |
| `Clock` | `now_utc()` → timezone-aware UTC `datetime`; `monotonic_ms()` → non-decreasing `int` | Stateless | Required | All |

No third-party object crosses an interface. NVML structs become `NvmlPciInfo`,
`NvmlMemoryInfo` and `NvmlUtilization`; process output becomes `CommandResult`; the probe's
JSON becomes `TorchProbeReport`. Report values reuse the existing models (`Observed`,
`GpuProcess`, `PciObservation`, `SystemInfo`). `NVML_READ_ONLY_FUNCTIONS` lists every NVML
function an implementation may call, so the read-only rule (NFR-12, AC43) is a checkable
contract.

#### 4.5.2 Normalised outcomes and error behaviour

`Observed[T]` is the single outcome type. Adapter-only values (device handles,
`CommandResult`, `TorchProbeReport`) may sit inside it but never reach the report.

| Condition | Outcome |
| --------- | ------- |
| Success | `COLLECTED` |
| Capability unsupported on this platform, device or driver | `NOT_SUPPORTED` (`NVML_ERROR_NOT_SUPPORTED`, `PLATFORM_NOT_SUPPORTED`, `NVIDIA_SMI_NOT_SUPPORTED`, …) |
| Python dependency not installed | `NOT_INSTALLED` (`ImportError`, `PackageNotFoundError`) |
| System component or executable missing | `NOT_DETECTED` (`NVML_ERROR_LIBRARY_NOT_FOUND`, `FileNotFoundError`) |
| Installed but unusable | `COLLECTION_FAILED` with a specific code (`IMPORT_ERROR`: torch present but import failed); a missing driver library under an installed binding is `NOT_DETECTED` for that library |
| Permission denied | `PERMISSION_DENIED` |
| Timeout | `COLLECTION_FAILED` / `TIMEOUT` |
| Failure during collection | `COLLECTION_FAILED` / NVML error name, `EXIT_<n>`, `PROBE_EXIT_<n>`, exception type name |
| Invalid or malformed output | `COLLECTION_FAILED` / `PARSE_ERROR` or `PROBE_PROTOCOL_ERROR` |
| Partial results | A `COLLECTED` container whose elements carry their own outcomes (`SmiGpuRow.values`, per-device NVML reads, per-field `SystemInfo`) |
| Unclassifiable | `UNKNOWN` / exception type name (a defect signal) |

The mapping functions (`nvml_error_status`, `exception_outcome`, `require_success`,
`decode_probe_output`) implement data model §3.3, and a test compares them with that table.

**What may raise.** Only two things:

1. `AdapterUsageError` (the only adapter exception) when the *caller* breaks a contract:
   a second concurrent NVML session, a query on a closed session or a stale device handle,
   a device index outside `0..count-1`, a non-allowlisted environment name, a shell string
   or empty `argv`, or a non-positive timeout. These are bugs and must not be caught by
   collectors.
2. Unexpected exceptions caused by implementation bugs. The collector guard turns these into
   `UNHANDLED_EXCEPTION` records (data flow §3).

An environmental condition (no driver, no GPU, no permission, a tool missing, a slow or
crashing child) never raises.

#### 4.5.3 Resource lifecycle contracts

| Concern | Contract |
| ------- | -------- |
| NVML initialisation | `NvmlAdapter.open()` imports the binding and calls `nvmlInit`. Expected failures don't raise: the session's `init_result` carries the failure, and every query returns it with the queried field's unit. |
| One shared session | The inspection core opens **one** session per inspection and passes the session, not the adapter, to NVML-backed collectors. Collectors never open or close NVML. A second `open()` while a session is open is a usage error. |
| Shutdown | `close()` is idempotent and calls `nvmlShutdown` once, and only after a successful `nvmlInit`, so shutdowns always balance initialisations. Shutdown errors are swallowed and logged at DEBUG. |
| After an exception | The context manager closes the session and lets the exception propagate. |
| After close | Session queries and device handles raise `AdapterUsageError`; a new inspection opens a new session. |
| Processes | `CommandRunner.run()` and `TorchProbe.probe()` own their child process completely: on timeout the child is killed before the call returns. |

#### 4.5.4 Adapter construction and dependency injection

`AdapterSet` is a frozen dataclass holding one implementation of each protocol plus
`provenance` (`LIVE` or `FIXTURE`), which becomes the report's `data_origin` (DEC-17). Its
constructor checks every field against its protocol. Assembly rules:

- **Explicit injection, no registries.** `Inspector(adapters)` receives the set and hands
  each collector the adapters it needs (and the set's `clock`) through its constructor.
  There is no module-level instance, registry or singleton. Tests pass
  `FakeAdapterSet.from_scenario(fx).adapter_set()`.
- **Production factory (implemented, DEC-44, DEC-45).** `core/factory.py`
  `build_production_adapters(options=None) -> AdapterSet` is the only place that names
  production classes; `api.inspect(options)` calls it and always gets `provenance = LIVE`.
  Its slots: `OsPlatformAdapter`, `SystemClock`, `SubprocessCommandRunner`,
  `PynvmlAdapter(platform=…)`, `NvidiaSmiCli(runner, platform,
  timeout_s=options.subprocess_timeout_s)`, the not-implemented PyTorch-probe stand-in
  (IMP-A07), and `DeferredPciScanner` for `pci`, which refuses to scan. Nothing is loaded or
  run when the set is built. The PCI scanner needs the Linux environment, which is known only
  after the system collector ran (profile `LINUX` covers `NATIVE_LINUX`, WSL1 and
  `UNKNOWN`), so the core builds it with `production_pci_scanner(environment)` from the
  resolved environment (§4.2) instead of guessing it up front.
- **Partial availability.** Every field is always filled. An unavailable capability is an
  adapter that reports `NOT_INSTALLED`/`NOT_DETECTED`/`NOT_SUPPORTED` (a null object), never
  `None`, so collectors have no missing-adapter branches.
- **Platform-specific selection.** The factory picks implementations from the platform
  facts, for example the WSL2 `nvidia-smi` search path. The PCI scanner is the exception
  that must not be selected from `PlatformProfile` alone. The core always builds
  `PciSysfsAdapter(environment=<resolved linux_environment, or None off Linux>)`, and the
  adapter itself applies the safety boundary: only `NATIVE_LINUX` touches sysfs, and every
  other environment (including `UNKNOWN` and WSL1, which have profile `LINUX`) answers
  `NOT_SUPPORTED`
  (DEC-41, DEC-42). Collectors receive platform facts as data and never branch on the OS
  themselves (§5).
- **Probe mode.** The CLI uses a subprocess `TorchProbe` (built on `CommandRunner`); the
  Python API may choose in-process probing (DEC-07, DEC-08).

#### 4.5.5 Adapter implementations (implemented and planned)

| Adapter | Wraps | Key behaviour |
| ------- | ----- | ------------- |
| Not-implemented stand-ins (`adapters/unimplemented.py`) | nothing | **Implemented (DEC-44); no longer used in production** since the PyTorch probe landed (DEC-48). They remain as conforming placeholders for tests. |
| `NvmlAdapter` / `NvmlSession` / `NvmlDevice` → `PynvmlAdapter` / `PynvmlSession` / `PynvmlDevice` (`adapters/nvml.py`) | `pynvml` from `nvidia-ml-py` (EV-11, DEC-05) | **Implemented (IMP-A01, DEC-45).** `open()` loads `pynvml` through an injectable loader (default `importlib.import_module("pynvml")`, so importing the package loads nothing; tests pass a stub module). A missing binding → `NOT_INSTALLED`/`ImportError`; on macOS the session is `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED` without loading anything. `nvmlInit()` once per session; an `NVMLError` maps by its numeric value to the binding's `NVML_ERROR_*` name and `nvml_error_status` (library not found / driver not loaded → `NOT_DETECTED`, no permission → `PERMISSION_DENIED`, not supported → `NOT_SUPPORTED`, the rest, e.g. GPU lost → `COLLECTION_FAILED`; an unknown value → `NVML_ERROR_UNKNOWN`). If init failed, every query returns the init outcome with the field's unit. `close()` is idempotent and calls `nvmlShutdown()` once, only after a successful init; a shutdown error is swallowed, logged at DEBUG and kept in `shutdown_errors`. Every query is independent (one failing field or device leaves the rest); `None` → `NOT_SUPPORTED`/`VALUE_NOT_AVAILABLE`; an unusable value → `COLLECTION_FAILED`/`PARSE_ERROR`, never a zero. Conversions: bus ID text, PCI IDs `0x%08x`, compute capability `major.minor`, MIG `ENABLED`/`DISABLED`, power mW → W, CUDA versions `1000·major + 10·minor` → `"12.4"`, process names basename only. Prefers `_v2`/`_v3`/`V` variants and falls back on `FUNCTION_NOT_FOUND`/version mismatch or a binding without the newer call (`nvmlDeviceGetMemoryInfo` v2 → v1 with reserved `VALUE_NOT_AVAILABLE`; `nvmlDeviceGetTemperatureV` → `nvmlDeviceGetTemperature`; `nvmlDeviceGetPciInfo_v3` → `nvmlDeviceGetPciInfo`; compute processes `_v3` → legacy). A second concurrent session, an index outside `0..count-1` or any query on a closed session is `AdapterUsageError`. Calls only `NVML_READ_ONLY_FUNCTIONS` (NFR-12). |
| `NvidiaSmiAdapter` → `NvidiaSmiCli` (`adapters/nvidia_smi.py`) | `nvidia-smi` executable via `CommandRunner` | **Implemented (IMP-A02, DEC-45).** `locate()` (once per adapter) searches `PATH` and, on WSL2, `/usr/lib/wsl/lib` (EV-09); macOS → `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED` without a lookup; not found → `NOT_DETECTED` and nothing runs. `version()` parses the header of plain `nvidia-smi` (`KMD Version`/`Driver Version`, `CUDA UMD Version`/`CUDA Version`, EV-03; no label → `FIELD_NOT_FOUND`). `query_gpus()` runs `nvidia_smi_query_argv` (EV-02) with `options.subprocess_timeout_s` and parses it with `parse_query_output` (CSV via `csv`, never table columns): per value `[N/A]`/`[Not Supported]` → `NOT_SUPPORTED`, `[Insufficient Permissions]` → `PERMISSION_DENIED`, `[Unknown Error]`/`[GPU is lost]` → `COLLECTION_FAILED`, empty → `NOT_DETECTED`/`FIELD_NOT_FOUND`, unparsable → `PARSE_ERROR` for that field only; MiB → bytes with the 1 MiB precision `note`. A wrong column count, bad or duplicate index, or truncated output fails the whole query (`PARSE_ERROR`), since a partial inventory would miscount devices. Non-zero exit → `COLLECTION_FAILED`/`EXIT_<n>` with the tool's message; an unknown query field (older driver) → `NOT_SUPPORTED`/`NVIDIA_SMI_NOT_SUPPORTED`; timeout → `TIMEOUT`. Empty output with exit 0 is zero GPUs. `locate()`, `version()` and `query_gpus()` are each computed once per adapter (one adapter set per inspection, DEC-46), so the GPU hardware fallback, the runtime fallback and the `nvidia_smi` run share one execution. |
| `PciSysfsAdapter` (`PciScanner`) | `/sys/bus/pci/devices/*/{vendor,device,class}` and the `driver` symlink (EV-26), through the `SysfsReader` seam (`StdlibSysfsReader`: `os.listdir`, read-only `open`, `os.readlink`) | **Implemented in Stage 6.1.** **Fail-closed eligibility:** built with an explicit `LinuxEnvironment` value; only `NATIVE_LINUX` calls the reader, and every other environment returns `NOT_SUPPORTED` without any sysfs access (the platform profile is never used for this). Stage 6.2: `UNKNOWN` uses `LINUX_ENVIRONMENT_UNKNOWN`, the others `PLATFORM_NOT_SUPPORTED`, with the pinned reasons of data model §3.4. The reader accepts only the `vendor`, `device` and `class` attributes (anything else is `AdapterUsageError`), so config space is unreachable. Keeps base class `0x03` (display controllers, including 3D controllers). Strict canonical parsing: address `<domain>:<bus>:<dev>.<fn>` in lowercase hex, vendor/device `0x` + 4 lowercase hex, class `0x` + 6. `is_nvidia` ⇔ vendor `0x10de`. The driver is the basename of the symlink target, and a missing link is `NOT_DETECTED`/`NO_DRIVER_BOUND`. A missing root is `NOT_DETECTED`, a denied root `PERMISSION_DENIED`, any other root error `COLLECTION_FAILED`. Per-device problems are typed `PciDeviceFailure`s (invalid address, unreadable or malformed attribute: device omitted; unreadable driver link: observation kept with a failed `kernel_driver`), so a collector can record `PARTIAL` with `CollectionError` (`PciCollector`, §4.4, DEC-43). Observations are sorted by address. |
| `OsPlatformAdapter` (`PlatformAdapter`) | `platform`, `os`, `sys`, `/proc`, `os-release`, `winreg`, `ctypes` through `StdlibHostApi` | **Implemented in Stage 5** (§4.3). Single seam for OS facts and environment variable reads (allowlisted names only). |
| `CommandRunner` → `SubprocessCommandRunner` (`adapters/commands.py`) | `subprocess.run(list(argv), shell=False, stdin=DEVNULL, capture_output=True, text=True, errors="replace", timeout=…)`; `shutil.which` | **Implemented (IMP-A05, DEC-45).** The only adapter that may start processes (`subprocess`/`shutil` imported on first use). Never uses a shell; argv is passed verbatim; the child gets no input and inherits the environment, which is never recorded. Returns `COLLECTED` `CommandResult(argv, returncode, stdout, stderr, duration_ms, truncated)` for any process that ran, whatever its exit (callers apply `require_success`); missing → `NOT_DETECTED`, denied → `PERMISSION_DENIED`, timeout → `COLLECTION_FAILED`/`TIMEOUT` (`subprocess.run` kills the child first), other `OSError` → `COLLECTION_FAILED` (§4.5.2, EV-22). A string or empty argv, a NUL, or a non-positive timeout is `AdapterUsageError` and starts nothing; any other exception (including a malformed result) propagates to the guard. `which(name, extra_paths=dirs)`: `PATH`, then each directory (executable files only). Output kept up to `MAX_COMMAND_OUTPUT_CHARS` (1 Mi characters), `truncated` set beyond. Duration from the injected `Clock`. |
| `SubprocessTorchProbe` / `InProcessTorchProbe` (`adapters/torch_probe.py`) | `CommandRunner` + `_torch_probe_main` / direct call of the same probe function | **Implemented (IMP-A07, DEC-48).** §4.7; output decoded by `decode_probe_output` / `decode_probe_payload` (protocol 2). |
| `ToolkitLocator` → `CudaToolkitLocator` (`adapters/toolkit.py`) | `ToolkitFiles` (`StdlibToolkitFiles`: `os.stat`, `os.listdir`, bounded `open`, `os.path`) + `nvcc` via `CommandRunner` | **Implemented (IMP-A06, DEC-47).** §4.6. `AdapterSet.toolkit`; the files seam is the locator's own, not a general filesystem layer. |
| `SystemClock` (`Clock`) | `datetime.now(UTC)`, `time.monotonic_ns()` | **Implemented in Stage 4.** |

### 4.6 CUDA environment inspection (`collectors/cuda_env.py`, `adapters/toolkit.py`)

**Implemented (IMP-A06, IMP-K05; DEC-47, which resolves OQ-10).** CUDA Toolkit discovery is
not CUDA execution validation: driver compatibility (`driver.driver_supported_cuda_version`),
toolkit installation (this section), the CUDA runtime, PyTorch availability and successful
GPU execution are separate facts, and only the second is observed here.

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Produce `CUDAEnvironmentInfo`: toolkit candidates, the active `nvcc`, allowlisted environment variables. |
| Inputs | `ToolkitLocator` (`AdapterSet.toolkit`), `PlatformAdapter` (profile and allowlisted variables). |
| Outputs | `CUDAEnvironmentInfo`; `CUDA.TOOLKIT` on macOS and `OTHER`. |
| Public interface | `CudaEnvironmentCollector(locator, platform, clock)` (`Collector[CUDAEnvironmentInfo]`, run `cuda_environment`, after `nvidia_smi`); `ToolkitLocator.discover(hints) -> ToolkitDiscoveryResult`; `CudaToolkitLocator(commands, platform, files=None, *, timeout_s)`; `ToolkitFiles` / `StdlibToolkitFiles`. |

**Discovery signals (DEC-47).** Every signal is consulted; each one that points at an
existing directory adds that directory as a candidate, and `discovered_via` lists all the
signals that pointed at it. There's no winner: all candidates are reported (AC17), the
active compiler is reported separately.

| Signal | Platforms | How |
| ------ | --------- | --- |
| `CUDA_HOME` | Linux, WSL2, Windows | the variable's value (read through `PlatformAdapter.getenv`) |
| `CUDA_PATH` | Linux, WSL2, Windows | same |
| `CUDA_PATH_VERSIONED` | Windows | every set `CUDA_PATH_V*` variable; names come from `PlatformAdapter.versioned_cuda_path_names()`, which filters by `CUDA_PATH_VERSIONED_REGEX` before reading any value |
| `NVCC_ON_PATH` | Linux, WSL2, Windows | `CommandRunner.which("nvcc")`, symlinks resolved; the root is the parent of its `bin/` directory (no root if the resolved file isn't in a `bin/` directory) |
| `DEFAULT_LOCATION` | Linux, WSL2 | one non-recursive listing of `/usr/local`, entries named `cuda` or `cuda-<version>` (`^cuda(-[0-9][0-9.]*)?$`) |

Not sources: a configured path (no option exists), `version.txt`, any library
(`libcudart`, `libcuda`), NVML or `nvidia-smi` CUDA versions (FR-07: never inferred from the
driver), the Windows `%ProgramFiles%` location (`ProgramFiles` isn't an allowlisted
variable), recursive searches.

**Per candidate.** `root_path` is the directory's resolved path (symlink aliases such as
`/usr/local/cuda` → `/usr/local/cuda-12.4` merge into one candidate, their signals united);
candidates are ordered by `root_path`. A signalled path that isn't an existing directory
isn't a candidate (the variable stays visible in `environment`); a root whose type can't be
checked is a candidate whose version carries that failure. `version`: `<root>/version.json`
(read up to 64 KiB) → `.cuda.version`, which must match `^[0-9]+(\.[0-9]+)+$` (method
`version.json:cuda.version`); a malformed file or value is `COLLECTION_FAILED`/`PARSE_ERROR`
with no fallback, an unreadable one `PERMISSION_DENIED`/`COLLECTION_FAILED`; without the
file, `<root>/bin/nvcc --version` (method `nvcc --version`); with neither,
`NOT_DETECTED`/`NO_VERSION_EVIDENCE`. A directory name is never a version. `nvcc_path`:
`<root>/bin/nvcc` (`nvcc.exe` on Windows) if it's a file (method `toolkit layout`), else
`NOT_DETECTED`/`NVCC_NOT_IN_TOOLKIT`: root and compiler are separate evidence.

**Active `nvcc`.** `active_nvcc_path` is what `which("nvcc")` returns (`NOT_DETECTED`,
`FileNotFoundError` "nvcc not found" otherwise); `active_nvcc_version` runs it once with
`--version` (`release X.Y, VX.Y.Z` → `X.Y.Z`, `release X.Y` → `X.Y`; otherwise
`PARSE_ERROR`; non-zero exit `EXIT_<n>`; timeout `TIMEOUT`). A candidate needing the same
`nvcc` for its version reuses that run. A missing `nvcc` isn't a missing toolkit.

| Platform | Behaviour |
| -------- | --------- |
| Linux (`NATIVE_LINUX`, `WSL1`, `UNKNOWN`), WSL2 | the Linux signals above. Toolkit files are user-space facts; nothing here depends on native hardware access, so the environment classification doesn't gate them, and no hardware or usability claim is made |
| Windows | `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V*`, `nvcc` on `PATH`; no default location |
| macOS, `OTHER` | all three fields `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`, nothing read or run; one `CUDA.TOOLKIT` record; the environment map is still collected |

| Outcome | `toolkits` | Run |
| ------- | ---------- | --- |
| at least one candidate | `COLLECTED` (method `toolkit discovery`) | `COMPLETED`; `PARTIAL` if a field has a failure status, or the `/usr/local` listing failed (one `CollectionError` at `/cuda_environment/toolkits`) |
| none, every signal answered | `NOT_DETECTED`/`NO_TOOLKIT_FOUND` (informational: PyTorch wheels need no toolkit, D03) | `COMPLETED` |
| none, and the `/usr/local` listing failed | that failure (`PERMISSION_DENIED`, `COLLECTION_FAILED`), never `NOT_DETECTED` | `PARTIAL`, the same `CollectionError` |
| locator or collector raised | guard: `CUDAEnvironmentInfo.unavailable` (FX10) | `FAILED` |

**Boundaries.** Files only through `ToolkitFiles` (`kind`, which follows symlinks, returns
`None` only for a missing path and raises every other `OSError`; `list_dir`; `read_text`
with a limit; `realpath`; `join`/`dirname`/`basename`); commands only through the shared
`CommandRunner` (argument lists, `options.subprocess_timeout_s`, no shell). Nothing runs at
import or construction. Testing: `FakeToolkitFiles` trees and canned `nvcc` output;
`FakeAdapterSet` builds the production locator over them from each fixture.

### 4.7 PyTorch adapter (`adapters/torch_probe.py`, `collectors/framework.py`)

**Implemented (IMP-A07, IMP-K06; DEC-48).** PyTorch's facts are kept apart: installed
(`distribution_version`), importable and which build (`version`, `cuda_build`,
`build_cuda_version`, `rocm_build`, `cudnn_version`), what PyTorch says about CUDA at
runtime (`cuda_available`, its warnings) and which devices it enumerates. None of them
shows that CUDA code runs: the probe runs none. The build's CUDA version is neither the
toolkit (§4.6) nor the driver's (§4.4).

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Determine whether PyTorch is installed, importable and CUDA-capable, and collect device/build facts without disturbing the main process. |
| Inputs | `InspectionOptions` (`include_pytorch`, `pytorch_probe_mode`, `pytorch_timeout_s`), the `TorchProbe` (`AdapterSet.torch_probe`). |
| Outputs | `FrameworkInfo.pytorch` (`PyTorchInfo`) with `TorchDeviceInfo[]`. Correlation to NVML is `collectors/correlation.py`'s (IMP-K07, below); the PyTorch run leaves `correlated_device_ref` `NOT_APPLICABLE` "Set by the correlation run (DEC-09)" for it (DEC-49 (5)). |
| Design | **Step 1** (main process): `importlib.metadata.version("torch")` (EV-22). `PackageNotFoundError` → `NOT_INSTALLED`, with no subprocess. **Step 2**: `SubprocessTorchProbe` runs `sys.executable -m gpu_system_inspector.adapters._torch_probe_main` through the shared `CommandRunner` with `pytorch_timeout_s` (DEC-07). The probe imports torch, records versions and `is_available()` inside `warnings.catch_warnings(record=True)`, counts devices, reads each device's name, capability and properties, and prints one JSON document (protocol 2, data flow §5). A crash, non-zero exit, timeout or invalid JSON → `COLLECTION_FAILED` with `error_code` and stderr excerpt. `InProcessTorchProbe` (Python API only, `IN_PROCESS`) runs the same probe function (`adapters/torch_probe_payload.py`, kept apart from the child's entry module so importing the package never pre-imports it) in the calling process and reads allocator stats if `torch.cuda.is_initialized()` was already true (DEC-08); it can't enforce a timeout, and reading device properties may initialise CUDA in the caller. The factory picks the probe from the options. |
| Why a subprocess | (a) A crash or hang in CUDA initialisation can't take down the inspector; (b) the main process never creates a CUDA context, so NVML readings and readiness aren't affected by the inspector; (c) the import cost is bounded by a timeout; (d) it enables a future `--python` option `[FUT]`. |
| Failure behaviour | Attributes missing on older PyTorch versions (`uuid`, `pci_bus_id`, EV-14) → `NOT_SUPPORTED`/`ATTRIBUTE_MISSING` per field. A failure reading device properties → `device_error` (protocol 2): section `COLLECTED` with every build fact, `device_count` `COLLECTION_FAILED`/`DEVICE_ENUMERATION_FAILED`, run `PARTIAL`, one error at `/frameworks/pytorch/devices`. A probe that ran but failed → section `COLLECTION_FAILED` with `distribution_version` kept, run `PARTIAL`, one error at `/frameworks/pytorch` (a timeout reads "PyTorch probe exceeded pytorch_timeout_s (N s)"). Unreadable metadata → `COLLECTION_FAILED`, error at `/frameworks/pytorch/distribution_version`. `include_pytorch = false` → `SKIPPED`, nothing read or run. An exception escaping the collector → the guard's `FAILED` with `PyTorchInfo.unavailable`. `torch.cuda.mem_get_info()` is **not** called (C10 `[ASM]`: it would create a context), no tensor is allocated and no kernel runs. |
| Fields | `build_cuda_version` always carries `BUILD_CUDA_NOTE`; for a non-CUDA build it and `cudnn_version` are `NOT_APPLICABLE` ("CPU-only PyTorch build", or "Non-CUDA (ROCm) PyTorch build"); a CUDA build whose cuDNN is `None` → `NOT_DETECTED`/`CUDNN_NOT_AVAILABLE`. `allocator`: subprocess → `NOT_APPLICABLE` with the DEC-08 note; in-process → `COLLECTED` (note "Calling process only") only if CUDA was initialised before the probe, else `NOT_APPLICABLE` ("CUDA isn't initialised in the calling process…"). `probe` = `{mode, protocol_version: 2, duration_ms}` (the child's run time; `null` when no probe ran or it failed). |
| Public interface | `TorchProbe` protocol (§4.5.1): `distribution_version() -> Observed[str]`, `probe(timeout_s: float) -> Observed[TorchProbeReport]`; implementations `SubprocessTorchProbe(commands, python_executable=None, *, distribution)` and `InProcessTorchProbe(loader, *, distribution, provenance)`. `FrameworkCollector(probe, clock, *, options)` (`Collector[FrameworkInfo]`, run `pytorch`, last, after `cuda_environment` and outside the NVML session; owns no capability records). |
| Testing | `unit/test_torch_probe.py`: the probe function over `StubTorch`; the probe child in a real subprocess with a stub `torch` package on its path (verifies the protocol without real PyTorch); both probes over the fake runner / stub. `unit/test_framework_collector.py`: parity with every fixture and every failure path. Real PyTorch only in hardware tests (HW-04, HW-06). |

**Device correlation (`collectors/correlation.py`; IMP-K07, DEC-09, DEC-49).** Run
`correlation`, last. `CorrelationCollector(clock, *, inventory, frameworks, synthetic)` takes
the `gpu_hardware` run's `GPUInventory` and the `pytorch` run's `FrameworkInfo` and returns
the PyTorch section with each device's `correlated_device_ref` set. It takes no adapter:
no NVML session, no probe, no process, no file. It owns no capability records.

| Step | Rule |
| ---- | ---- |
| Identifiers | UUID: lower-case, strip a leading `GPU-`, must be 8-4-4-4-12 hex (data model §5.4). PCI bus ID: `domain:bus:device.function`, domain 4 or 8 hex digits, compared by value (`00000000:01:00.0` = `0000:01:00.0`). Missing, not collected or unparsable → unusable; never a wildcard, never completed by guessing. |
| 1. UUID | among NVML devices with a usable UUID: one → match (`uuid`); several → `AMBIGUOUS_MATCH`; none → step 2 |
| 2. PCI bus ID | among NVML devices a UUID can't rule out (both usable and different rules a device out): one → match (`pci_bus_id`); several → `AMBIGUOUS_MATCH` |
| Never | index, name, memory, enumeration position |
| One-to-one | an NVML device matched by several PyTorch devices → `AMBIGUOUS_MATCH` for each; none is chosen |
| Contradiction | a UUID match whose PCI bus IDs differ (both usable) keeps the UUID match and says so in the `note` |

| Situation | `correlated_device_ref` |
| --------- | ----------------------- |
| unique match | `COLLECTED` = the NVML `device_ref`, source `INSPECTOR`, method `uuid` / `pci_bus_id` |
| `inventory_source` isn't NVML (none, or the `nvidia-smi` fallback, FX04a) | `NOT_APPLICABLE` "No NVML inventory to correlate with" |
| no candidate (incl. an empty NVML inventory) | `NOT_DETECTED`/`NO_MATCH` |
| PyTorch exposes no usable UUID or PCI bus ID | `NOT_DETECTED`/`NO_IDENTIFIER` |
| several candidates or claims | `NOT_DETECTED`/`AMBIGUOUS_MATCH`, the candidates named |
| no match, but an NVML device has no usable identifier | `NOT_DETECTED`/`UNIDENTIFIED_CANDIDATES`, that device named |

An unmatched device is a fact, not an error: the run is `COMPLETED` with no
`CollectionError`, and an NVML device without a PyTorch counterpart isn't reported at all.
The output follows the PyTorch devices' order (`torch_index`) and doesn't depend on the NVML
enumeration order. An exception → the guard's `FAILED`; `unavailable()` keeps every PyTorch
fact and fails only `correlated_device_ref` (`COLLECTION_FAILED`).

### 4.8 Diagnostics engine (`diagnostics/`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Evaluate rules D01–D08 over the snapshot (and readiness result) and return `DiagnosticFinding[]`. |
| Inputs | `DiagnosticInput(snapshot: InspectionSnapshot, readiness: ReadinessResult \| None, options: InspectionOptions)`. |
| Outputs | Findings sorted by `(check_id, device_ref)`. |
| Dependencies | `models` only. Rules are pure functions: no I/O, no clock, no randomness. |
| Failure behaviour | A rule that raises produces one finding `status = UNKNOWN`, `title = "Check could not be evaluated"`, plus a `CollectionError` with `collector = "diagnostics.<check_id>"`. Other rules still run. |
| Public interface | `Rule = Callable[[DiagnosticInput], Sequence[DiagnosticFinding]]`; `DiagnosticsEngine(rules: Sequence[RegisteredRule] = DEFAULT_RULES).evaluate(input) -> DiagnosticsResult(findings, errors)` (the `diagnostics.Dxx` errors join the report's `collection_errors`, DEC-50); `RegisteredRule(check_id: str, title: str, fn: Rule)`. |
| Implementation (IMP-D00–D08, DEC-50) | `diagnostics/engine.py` (guard, ordering, at least one and at most one finding per device per rule), `evidence.py` (observation/inference helpers, reference-URL allowlist), `context.py` (shared snapshot facts: absence only from a complete, `COLLECTED` PCI scan — KI-32, KI-33), `rules.py` (D01–D08, one function per decision table). `Inspector.run` evaluates them after collection. |
| Testing | Table-driven unit tests per rule over snapshot builders; golden findings per fixture scenario ([diagnostics_specification.md](diagnostics_specification.md)). |

### 4.9 Readiness engine (`readiness/`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Mode B: compare each device's observed free memory with `required_bytes`, then aggregate. |
| Inputs | `InspectionSnapshot` (runtime free/total memory per device, MIG mode, inventory status), `required_vram_gib`. |
| Outputs | `ReadinessResult` (data model §6.2). |
| Rules | `required_bytes = round(required_vram_gib × 2³⁰)` (DEC-11). Device: MIG enabled → `UNKNOWN/MIG_ENABLED_UNSUPPORTED`; free not `COLLECTED` → `UNKNOWN/FREE_MEMORY_UNAVAILABLE`; total `COLLECTED` and `total < required` → `NOT_READY/INSUFFICIENT_TOTAL_MEMORY`; `free >= required` → `READY/SUFFICIENT_FREE_MEMORY`; else `NOT_READY/INSUFFICIENT_FREE_MEMORY`. Overall: any `READY` → `READY`; else any `UNKNOWN` → `UNKNOWN`; else, if devices exist and all are `NOT_READY` → `NOT_READY`; with no devices: platform is macOS → `NOT_READY/PLATFORM_NOT_SUPPORTED`; NVML initialised successfully and reported 0 devices, or NVML/driver `NOT_DETECTED` **and** the PCI scan `COLLECTED` with no NVIDIA device → `NOT_READY/NO_NVIDIA_DEVICE`; otherwise `UNKNOWN/DEVICE_PRESENCE_UNKNOWN`. |
| Failure behaviour | Pure function; it never raises for missing data. |
| Public interface | `ReadinessEngine().assess(snapshot: InspectionSnapshot, required_vram_gib: float) -> ReadinessResult`. |
| Testing | Unit tests for every reason code, the boundary (`free == required` → `READY`), GiB conversion, multi-GPU aggregation, MIG and no-device cases. |
| Implementation (IMP-R01, DEC-50) | `readiness/engine.py`. Free and total come from the device's runtime record (total falls back to the inventory's); a `PARTIAL` PCI scan never yields `NO_NVIDIA_DEVICE` (KI-32). Limitations: the sampling time, the 1 MiB `nvidia-smi` precision, MIG. `Inspector.run` runs it when `required_vram_gib` is set. |

### 4.10 Reporting layer (`reporting/`)

| Aspect | Specification |
| ------ | ------------- |
| Responsibility | Assemble `InspectionReport` (metadata, limitations, unsupported capabilities), optionally redact, serialise to JSON or text, write atomically. |
| Inputs | Snapshot, findings, readiness, options, application metadata, clock. |
| Outputs | `InspectionReport`; `str` (JSON or text); file on disk. |
| Dependencies | `json`, `textwrap`, `models`; only `writer.py` uses `os`/`tempfile`. |
| Failure behaviour | Serialisation of a model is total (every enum/value has a JSON form). Unsupported types are a programming error caught by tests. Write errors raise `OutputError` → CLI exit `4`; the temp file is removed. |
| Assembly (IMP-P01, implemented) | `reporting/assemble.py` `assemble_report(snapshot, *, options, generated_at, duration_ms, data_origin, limitations=())` adds the metadata and limitations (the snapshot statement first, each stated once) and rejects a snapshot in which a collector ran twice or a `(capability, device_ref)` record appears twice (one owner each: `SYSTEM.*` system, `GPU.PCI_SCAN` PCI, `NVIDIA.ALL` GPU hardware). It also takes the findings, the readiness result and the diagnostics errors (DEC-50). |
| Text, redaction, writer (IMP-P03–P05, DEC-50) | `text_report.py` `render_text` (CLI spec §5.2); `redaction.py` `redact(report, RedactionPolicy(categories, user, home))` and `Redactor` (also used for CLI error messages); `writer.py` `write_atomic` and `OutputError(path, reason)`. The CLI owns resolving the local user name and home directory (`cli.main.local_redaction_policy`). |
| Public interface | `assemble_report(...) -> InspectionReport`; `redact(report, policy=DEFAULT_REDACTION) -> InspectionReport`; `to_json_dict(report) -> dict[str, Any]`; `render_json(report) -> str`; `render_text(report) -> str`; `write_atomic(text: str, path: Path) -> None`; `REPORT_SCHEMA_PATH`. |
| Testing | Schema validation of every fixture report (`jsonschema`, test-only dependency); golden text snapshots; determinism test; redaction leak tests; atomic write tests with read-only directories. |

### 4.11 Integration contract for IMP-K08 and IMP-K02 (Stage 6.6) and its delivery

The two tables below are the contract recorded at the end of Stage 6 (left as written). Both
milestones, together with IMP-P01, are now implemented (DEC-44); how each item was met, and
the test that proves it, follows the tables.

**IMP-K08 (core: context, guard, ordering).**

| Topic | Current contract (source) | IMP-K08 must |
| ----- | ------------------------- | ------------ |
| Collector shape | `SystemCollector(platform, clock)` and `PciCollector(scanner, clock, *, environment, profile)`: constructor injection, `name: ClassVar[str]`, `collect() -> CollectorOutcome[SectionT]`, `unavailable(error: ErrorDetail) -> SectionT`. `FakeCollector.collect(ctx=None)` returns the bare section. §4.2's protocol (`collect(ctx) -> SectionT`, `unavailable(CollectionError)`) isn't in code (DEC-43). | Define one `Collector` protocol that the two production collectors and `FakeCollector` satisfy, then correct §4.2 and data flow §3 to match. Keep `unavailable(ErrorDetail)`: the fallback's `Observed.error` can only hold an `ErrorDetail`. |
| `CollectorOutcome` | `COMPLETED` can't carry errors; `FAILED` needs at least one error; every error's `collector` equals the run's; `PARTIAL` may have none (`models/collection.py`). | Build the guard's `FAILED` outcome as `unavailable(detail)` + run `FAILED` + one `CollectionError` (`UNHANDLED_EXCEPTION`, `COLLECTION_FAILED`). Collectors never return `FAILED`. |
| Exceptions | Expected conditions are outcomes. Collectors let everything else propagate, including `AdapterUsageError`, which tests treat as a bug. | Catch them in one guard only (data flow §3); don't add a second guard elsewhere. |
| Duration | Collectors time themselves with the injected `Clock`. | Use the collector's own run on success, and time `FAILED` runs in the guard with the same `Clock`. |
| Order and dependencies | `COLLECTOR_ORDER` in the test kit is `system`, `pci`, `nvml`, `driver`, `gpu_hardware`, `runtime`, `nvidia_smi`, `cuda_environment`, `pytorch`, `correlation`, and every fixture records a run for each. `PciCollector` needs `linux_environment` and `platform_profile` from the system section. | Run `system` first and pass its facts on. If the system run failed, `linux_environment` isn't collected (`None` with a `LINUX` profile, which `unsupported_capabilities_for` rejects). Map that fail-closed: `UNKNOWN` for a `LINUX` profile, `WSL2` for `WSL2`, `None` off Linux. |
| PCI scanner construction | `PciSysfsAdapter(environment=…)` needs the environment when it's built, and `AdapterSet.pci` holds it. | Classify before building the scanner (the factory can call `OsPlatformAdapter.system_info()`; the adapter caches its classification), and give the collector the same environment, otherwise `PciCollector` raises `AdapterUsageError`. |
| Snapshot assembly | `InspectionSnapshot` sorts `collection_errors` and `unsupported_capabilities` but doesn't deduplicate them. | Concatenate every outcome's errors and records, and keep `collector_runs` in execution order. Each capability has one owner: `SYSTEM.*` system collector, `GPU.PCI_SCAN` PCI collector, `NVIDIA.ALL` IMP-K02/IMP-P01. |
| Test kit | `fake_collectors()` has a `pci` entry (Stage 6.4); `gpu_hardware` still returns the whole `GPUInventory`. | Use the `pci` fake for guard tests of the PCI section. |

**IMP-K02 (GPU hardware collector, `GPUInventory`).**

| Topic | Current contract (source) | IMP-K02 must |
| ----- | ------------------------- | ------------ |
| PCI ownership | `PciCollector` owns the only PCI scan (DEC-43); `AdapterSet.pci` is used only by it. | Take the PCI section as an input and put it into `GPUInventory.pci_observations` unchanged. Never call `PciScanner.scan()` again. |
| `GPUInventory` | `device_count`, `inventory_source`, `devices`, `pci_observations`. A collected count needs a source and must equal `len(devices)`. `pci_observations` is re-sorted by address and must have unique addresses (`models/gpu.py`). | Fill the NVML/`nvidia-smi` fields (§4.4) and keep the invariants. |
| Fallback | `GPUInventory.unavailable()` sets every `Observed` to the failure status, `pci_observations` included. | After a `gpu_hardware` guard failure, put the PCI collector's section back, so one collector's failure doesn't erase another's result (FR-13). |
| `NVIDIA.ALL` | `unsupported_capabilities_for` returns it on macOS. FX13 expects exactly one record with `NVIDIA_ALL_MACOS_REASON`. The PCI collector filters it out. | Emit it once, together with IMP-P01, and never re-emit `GPU.PCI_SCAN`. |
| Fail-closed environment | An unclassified or unavailable Linux environment is never `NATIVE_LINUX` (DEC-41, DEC-43). | Keep this in the factory (see the IMP-K08 table). |
| Downstream | Diagnostics and readiness read `/gpu/pci_observations`. A `PARTIAL` scan isn't proof of absence (KI-32), and a whole-scan `NOT_DETECTED` needs explicit handling (KI-33). | Leave PCI interpretation to IMP-D01, IMP-D04 and IMP-R01. Keep the `pci` run and errors visible in the snapshot. |
| Privacy | PCI addresses appear in `PciObservation.address` and in `pci` `CollectionError.message` (class D). | Nothing here; redaction is IMP-P04. |

**Delivered (IMP-K08, IMP-K02, IMP-P01; DEC-44).**

| Contract item | Implementation | Evidence |
| ------------- | -------------- | -------- |
| One collector protocol | `collectors/base.py` `Collector`; satisfied by `SystemCollector`, `PciCollector`, `GpuHardwareCollector`, `NvmlSessionStep` and `FakeCollector` (which now returns the fixture's `CollectorOutcome`) | `unit/test_core_guard.py::test_every_collector_satisfies_the_one_contract`, `…fake_collectors_return_outcomes_the_guard_accepts` |
| Single guard, guard-owned `FAILED` | `core/guard.py` `run_guarded`: one host-wide `UNHANDLED_EXCEPTION` error with the collector's report path, `unavailable(ErrorDetail)`, guard-timed run; `AdapterUsageError` and contract violations contained and named; `KeyboardInterrupt`/`SystemExit` propagate | `unit/test_core_guard.py` (guard tests) |
| Order, environment first, fail-closed | `core/inspector.py`: `system` → resolve (`core/environment.py`) → build scanner → `pci` → `nvml` → `driver` → `gpu_hardware` → `runtime` → close the session → `nvidia_smi` (DEC-46) → `cuda_environment` (DEC-47) → `pytorch` (DEC-48) → `correlation` (DEC-49) | `unit/test_core_guard.py::test_unavailable_classification_fails_closed`; `integration/test_inspector.py::test_a_system_failure_fails_closed` |
| Scanner built from the classified environment | `pci_scanner_factory` (production: `production_pci_scanner`); the production set's `DeferredPciScanner` refuses to scan; a pre-built scanner for another environment is refused by `PciCollector` | `integration/test_inspector.py::test_wsl2_is_gated_and_reported_once`, `…prebuilt_native_scanner_is_refused_after_a_system_failure` |
| One PCI scan, PCI kept through fallbacks | `GpuHardwareCollector` takes the PCI section, has no scanner; `unavailable()` restores the PCI section | `unit/test_gpu_hardware_collector.py::test_the_collector_has_no_way_to_scan_pci`, `…fallback_section_keeps_the_pci_collectors_result`; `integration/test_inspector.py::test_a_gpu_hardware_failure_keeps_the_pci_section`, `…nvml_session_is_opened_once_and_always_closed` |
| NVML and fallback | §4.4; production `PynvmlAdapter` and `NvidiaSmiCli` since IMP-A01/A02 (DEC-45) | `unit/test_gpu_hardware_collector.py` (parity with all 24 fixtures since KI-34 was resolved, device isolation, fallback) |
| Device correlation (IMP-K07, DEC-09, DEC-49) | §4.7: `CorrelationCollector` | `unit/test_correlation_collector.py` (parity with all 24 fixtures, the DEC-09/DEC-49 rules, ambiguity, missing inventories, the guard); `integration/test_inspector.py` (whole `frameworks.pytorch`, correlation included, for every fixture) |
| PyTorch (IMP-A07/K06, DEC-48) | §4.7: `SubprocessTorchProbe`, `InProcessTorchProbe`, `_torch_probe_main`, `FrameworkCollector` | `unit/test_torch_probe.py`, `unit/test_framework_collector.py` (parity with all 24 fixtures); `integration/test_inspector.py` (whole `frameworks.pytorch` for every fixture, `correlated_device_ref` the K07 placeholder) |
| CUDA environment (IMP-A06/K05, DEC-47) | §4.6: `CudaToolkitLocator`, `CudaEnvironmentCollector`, `AdapterSet.toolkit` | `unit/test_toolkit_locator.py`, `unit/test_cuda_env_collector.py` (parity with all 24 fixtures); `integration/test_inspector.py` (whole `cuda_environment` section for every fixture, FX10's crash through the guard) |
| Driver and runtime (IMP-K03/K04, DEC-46) | §4.4: `DriverCollector`, `NvidiaSmiCollector`, `RuntimeCollector` | `unit/test_driver_collector.py`, `unit/test_runtime_collector.py` (parity with all 24 fixtures: whole sections, runs, errors, `RUNTIME.*` records); `unit/test_device_loss.py` (KI-34); `integration/test_inspector.py` (whole `driver` and `runtime` sections for every fixture) |
| Capability ownership | `SYSTEM.*` system, `GPU.PCI_SCAN` PCI, `NVIDIA.ALL` GPU hardware, `RUNTIME.*` runtime, `CUDA.TOOLKIT` CUDA environment; the PyTorch and correlation collectors own none; `assemble_report` rejects duplicates | `integration/test_inspector.py::test_capabilities_have_one_owner_each`, `…assembly_rejects_duplicate_capability_records_and_runs` |
| Snapshot and report | `Inspector.collect()` aggregates; `reporting/assemble.py`; `api.inspect()` | `integration/test_inspector.py` (24 fixtures: schema-valid, round trip, byte-identical repeat runs); `integration/test_inspect_host.py` (live) |

**Follow-ups recorded, not done here:** diagnostics (IMP-D*), readiness (IMP-R01, Mode B);
redaction (IMP-P04); the CLI `inspect` command and text renderer (IMP-C01, C03, IMP-P03,
still placeholders). Every collector of `COLLECTOR_ORDER` is now implemented (DEC-44 to
DEC-49).

## 5. Cross-cutting concerns

| Concern | Design |
| ------- | ------ |
| Logging | Logger hierarchy `gpu_system_inspector.*`; handler on stderr configured only by the CLI; library use adds a `NullHandler`. Log messages never contain raw values that redaction would remove when `--redact` is active: the redactor is also installed as a logging filter. |
| Time | The `Clock` protocol (`now_utc()`, `monotonic_ms()`, §4.5.1) is injected through `AdapterSet`. Production uses `SystemClock`; fixtures use `FixedClock` so reports are deterministic. |
| Privacy | Collection-time minimisation (NFR-08) plus report-time redaction (FR-19). |
| Concurrency | V1 is sequential. Parallel collectors are deferred: they add complexity, and NVML queries are fast `[ASM]`. |
| Platform branching | Only `PlatformAdapter`, `PciSysfsAdapter`, `ToolkitLocator` and `NvidiaSmiAdapter.locate()` branch on OS. Collectors receive OS facts as data. |
| Versioning | `__version__` (application, SemVer `0.x` until V1), `REPORT_SCHEMA_VERSION` (data model §7). |

## 6. Key design decisions

Decisions are also logged with dates in [app_tracker.md](app_tracker.md) §10.

| ID | Decision | Rationale | Status |
| -- | -------- | --------- | ------ |
| DEC-01 | During the specification phase the project has a metadata-only `pyproject.toml` with `tool.uv.package = false`, and `uv.lock` records it as a virtual member. | A `projects/*` directory without `pyproject.toml` breaks `uv lock`/`sync`/`run` for the whole workspace (OBS-03, EV-23). | Applied, then **superseded 2026-11-04** by the real package (§9) |
| DEC-02 | Application display name **GPU System Inspector** (approved). Distribution name `01-gpu-system-inspector` (follows `projects/README.md`: `name = "<dir>"`); import package `gpu_system_inspector`; report `application.name = "gpu-system-inspector"`. | The directory name isn't a valid Python identifier. The `inferforge_` import prefix is reserved for extracted packages (`docs/architecture.md`). | Display name **approved** 2026-11-04; distribution/import names proposed (publication name OQ-03) |
| DEC-03 | Console script **`inferforge-gpu`** with sub-commands `inspect` and `version`; `python -m gpu_system_inspector …` is equivalent. No `inferforge` umbrella command in V1. | A dedicated script name can't collide with other projects' commands in the shared workspace venv (EV-25), so no dispatcher package is needed. | **Approved 2026-11-04** |
| DEC-04 | NVML via `nvidia-ml-py` is the primary GPU source; `nvidia-smi` is used for detection/version and as a fallback inventory/memory source. | NVML is backward compatible; `nvidia-smi` output isn't (EV-01). | Proposed; applied by DEC-45 (IMP-A01, A02; NVML primary, `nvidia-smi` detection, header and fallback). Owner approval not recorded |
| DEC-05 | `nvidia-ml-py` is a **required** runtime dependency. The import is still guarded, so a broken installation degrades instead of crashing. Never depend on the `pynvml` PyPI package. | Pure-Python `ctypes` bindings with no compiled code, BSD-licensed (EV-11). Harmless to install without a driver. gpustat documents the `pynvml` pitfall (EV-17). | **Approved 2026-11-04** (version floor: OQ-07) |
| DEC-06 | Otherwise stdlib only: `argparse`, `dataclasses`, `enum`, `json`, `logging`, `subprocess`, `ctypes`. No `psutil`, `click`/`typer`, `rich`, `pydantic`. | Low footprint (NFR-03). | Proposed; applied (runtime dependencies are only `nvidia-ml-py`, AC50 metadata tests). Owner approval not recorded |
| DEC-07 | PyTorch is never a dependency. CLI probes it in a subprocess of the same interpreter with a timeout. | Isolation, no CUDA context in the main process, bounded cost (§4.7). | Proposed; applied by DEC-48 (IMP-A07: `SubprocessTorchProbe`). Owner approval not recorded |
| DEC-08 | Allocator stats are `NOT_APPLICABLE` unless CUDA is initialised in the calling process (in-process API only). | `memory_allocated()` returns `0` before initialisation (EV-12); a fresh probe process has no allocations. | Proposed; applied by DEC-48 (6) (IMP-K06: allocator rules). Owner approval not recorded |
| DEC-09 | Correlate PyTorch and NVML devices by normalised UUID, then by PCI bus ID; never by index alone. Uncorrelated devices are reported as such. | CUDA's default ordering and `CUDA_VISIBLE_DEVICES` re-indexing (EV-10). | Applied (IMP-K07; details DEC-49) |
| DEC-10 | `Observed[T]` envelope with the invariant `value is not None ⇔ status == COLLECTED`. | No misleading zeros (AC13). | Proposed |
| DEC-11 | Units: bytes (int), °C, W (float, converted from mW), % (0–100), ms. `--required-vram` is in GiB (2³⁰). | Unambiguous and lossless. | Proposed |
| DEC-12 | Readiness uses NVML (or `nvidia-smi`) device-wide free memory, sampled before the PyTorch probe. | The probe may create a context (C10/C11 `[ASM]`). | Proposed |
| DEC-13 | Overall readiness = `READY` if any device is `READY`; MIG-enabled devices are `UNKNOWN` in V1. | A single-device workload needs one suitable GPU; MIG instances have smaller memory slices. | Proposed |
| DEC-14 | Exit codes `0/1/2/3/4/5/130` ([cli_specification.md](cli_specification.md) §6). | Scriptable readiness gating. | Proposed |
| DEC-15 | Redaction is opt-in (`--redact`). Privacy by default comes from not collecting hostname, username, command lines or the full environment. | Local diagnostics need real paths; sharing needs redaction. | Proposed (OQ-06) |
| DEC-16 | Report schema `MAJOR.MINOR`, every key always present, shipped JSON Schema (Draft 2020-12). | Stable contract (NFR-05). | Proposed |
| DEC-17 | No demo or simulation mode in V1. `data_origin` ∈ {`LIVE`, `FIXTURE`}; only test fixtures can produce `FIXTURE`, and no CLI option can. | Mocked data must never look real (AC12). | **Approved 2026-11-04** (no demo mode) |
| DEC-18 | Logs only on stderr; stdout carries only the report. | Safe piping of JSON (AC26). | Proposed |
| DEC-19 | Readiness runs before diagnostics; D02 is derived from the readiness result. | Single source of truth for the memory verdict. | Proposed |
| DEC-20 | Hardware tests live in `tests/hardware/`, are marked `hardware`, and are deselected by default. | Ordinary suite needs no GPU (NFR-10). | Proposed |
| DEC-21 | Project 01's `requires-python` always equals the workspace floor: `>=3.11` (WSD-01). | Workspace consistency; the floor is a repository-level decision. | **Approved and applied 2026-11-04** |
| DEC-22 | The Linux sysfs PCI scan is in V1; other OSes get `NOT_SUPPORTED`. | Cheap, driver-independent evidence for D01/D04. Windows/macOS equivalents need slow subprocesses. | **Approved 2026-11-05** (with FR-04); Stage 6.1 restricts it to `NATIVE_LINUX` (DEC-41) |
| DEC-23 | `technical_evidence.md` added as the single evidence register. | Avoids duplicating citations across 12 documents. | Applied |
| DEC-24 | No colour in V1 text output. | Simplicity, deterministic golden tests. | Proposed |
| DEC-25 | Output options: the text report always goes to stdout. `--json PATH` additionally writes the JSON report atomically to `PATH`, and `--json -` writes JSON to stdout instead of text. This replaces the earlier `--format`/`--output` pair. | Matches the approved example `inferforge-gpu inspect --json report.json`. One option covers both scripting (`--json -`) and archiving. | **Approved 2026-11-04** (example); `-` convention proposed |
| DEC-26 | PyTorch is an optional dependency: detected at runtime in the inspector's own interpreter, never declared in `dependencies` or as an extra, never installed. | PyTorch wheels are CUDA-variant specific and selected through PyTorch's own index/selector. An extra could install a wrong variant. | Optional status **approved**; "no extra" interpretation **approved 2026-11-04** (OQ-09 resolved by the user, tracker §10); applied (DEC-48) |
| DEC-27 | Read-only and installation-free guarantees are product invariants (NFR-12, AC32, AC43). | Approved product rule. | **Approved 2026-11-04** |
| DEC-28 | Adapter interfaces are `typing.Protocol` classes (runtime-checkable) in `adapters/protocols.py`, separate from their implementations. | Real and fake adapters become interchangeable without inheritance. The fakes stay in `tests/`, and a conformance test compares signatures. | Applied (Stage 4) |
| DEC-29 | `Observed[T]` is also the adapter outcome type; no separate result or exception hierarchy. | One status vocabulary from the adapter to the JSON report; outcomes flow into models without translation (§4.5.2). | Applied (Stage 4) |
| DEC-30 | A single exception, `AdapterUsageError`, for caller contract violations; environmental conditions never raise. | Keeps "the system lacks X" (data) apart from "the code is wrong" (bug), without an elaborate hierarchy. | Applied (Stage 4) |
| DEC-31 | Grouped NVML reads (`memory_info`, `utilization`, `pci_info`) return one struct per NVML call. | Total, used and free come from one sample, so readiness can't mix readings taken at different moments. | Applied (Stage 4) |
| DEC-32 | The core owns the single NVML session; collectors receive the session, not the adapter. Shutdown runs only after a successful init and at most once. | Lifecycle stays out of collectors, and initialisations and shutdowns always balance (§4.5.3). | Applied (Stage 4, contract) |
| DEC-33 | `CommandRunner.run` returns `COLLECTED` for any process that ran; callers apply `require_success` or interpret the exit code. | Some tools report useful states through exit codes (for example `nvidia-smi` exit 9: driver not loaded). | Applied (Stage 4) |
| DEC-34 | `AdapterSet` always contains every adapter (null objects for unavailable capabilities), carries `provenance`, and is built by one factory function with no registry. | No `None` branches in collectors; `data_origin` has a single source; no hidden global state (§4.5.4). | Applied (Stage 4) |
| DEC-35 | `OsPlatformAdapter` reads the host only through `HostApi` (primitive stdlib calls), injected in tests. | Windows, macOS and other-platform branches are unit-tested deterministically on any machine, without patching the stdlib. | Applied (Stage 5) |
| DEC-36 | No `platform.processor()`, no `WSL_DISTRO_NAME`; Windows CPU name from the registry, WSL from `/proc/version`. | `platform.processor()` spawns `uname -p` on POSIX and reads a non-allowlisted variable on Windows; avoiding both keeps the collector subprocess-free and inside the environment allowlist. | Applied (Stage 5) |
| DEC-37 | macOS processor model and total memory are `NOT_SUPPORTED` in Stage 5 (the spec named `sysctl`). | `sysctl` needs a subprocess and the production `CommandRunner` doesn't exist yet; no macOS machine to validate on. FX13's synthetic expectation keeps the V1 target. | Applied (Stage 5); **superseded by DEC-40** (Stage 5.1): it downgraded an approved requirement (FR-02, AC36, data model §4.1) without approval |
| DEC-38 | Adapter provenance is enforced: an adapter's own `provenance` must match its `AdapterSet`; synthetic hosts make the production adapter emit `fixture:` methods. | A test can never wrap synthetic values in a `LIVE` set, and live values can't enter a fixture set. | Applied (Stage 5) |
| DEC-39 | `SystemInfo.is_wsl` (bool) is replaced by `linux_environment: Observed[LinuxEnvironment]`, classified by a documented conservative rule over `/proc/sys/kernel/osrelease` and `/proc/version`. | A boolean can't express WSL1 or "can't tell"; the word "Microsoft" alone isn't evidence of WSL2. Draft schema `1.0` may change without a bump (data model §7.3). | Applied (Stage 5.1) |
| DEC-40 | macOS processor model and total memory via read-only `sysctlbyname(3)` through `ctypes`, behind `HostApi.mac_sysctl`. Available memory stays `NOT_SUPPORTED`. | Meets the approved requirement without a subprocess or dependency; the native interface was verified against Apple's manual. | Applied (Stage 5.1); real macOS run pending |
| DEC-41 | `PciScanner.scan()` returns `PciScan` (observations plus typed per-device failures); `PciSysfsAdapter` requires an explicit `LinuxEnvironment` and is fail-closed (only `NATIVE_LINUX` touches sysfs); the `SysfsReader` seam allowlists `vendor`/`device`/`class`. | A bare `Observed` can't carry per-device failures for `PARTIAL`/`CollectionError`. The safety boundary lives in the adapter, so no caller can scan WSL or an unclassified host by mistake. The attribute allowlist makes config-space reads structurally impossible. | Approved and applied (Stage 6.1) |
| DEC-42 | PCI eligibility outcomes and capability policy: `UNKNOWN` Linux → `NOT_SUPPORTED`/`LINUX_ENVIRONMENT_UNKNOWN`; WSL1, WSL2, Windows, macOS and other non-Linux profiles (`OTHER`) → `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`; `GPU.PCI_SCAN` unsupported everywhere except `NATIVE_LINUX`, with one pinned, evidence-referenced reason per environment (shared by the scan message and the capability record); macOS also lists `NVIDIA.ALL`. The policy is a pure function (`collectors/capabilities.py`); runtime emission belongs to the PCI collector (Stage 6.3) and report assembly (IMP-P01). | An unclassified host isn't a known platform limitation, so it gets its own code. One reason string per environment keeps fixtures and production identical (data model §3.4). | Approved and applied (Stage 6.2) |
| DEC-43 | PCI collector contract. **(1)** A separate `collectors/pci.py` `PciCollector`, name `pci`, owns PCI discovery; `GpuHardwareCollector` only consumes its section (IMP-K02). **(2)** Section `Observed[tuple[PciObservation, ...]]`, passed through from the scanner unchanged; `unavailable(error: ErrorDetail)` returns `COLLECTION_FAILED` with `source`/`method` `None` and that error, following `SystemCollector` and `_validation.unavailable()`. This diverges from the unimplemented §4.2 protocol (`CollectionError` argument, `collect(ctx)`), which is reconciled in IMP-K08. **(3)** `PciCollector(scanner, clock, *, environment, profile)`; `unsupported_capabilities_for` keeps its environment/profile check; the collector raises `AdapterUsageError` if the scan contradicts the environment (eligible but `NOT_SUPPORTED`, or ineligible without the policy's `NOT_SUPPORTED` code and reason) or reports a status outside the mapping. **(4)** Whole scan: `COLLECTED` → `COMPLETED`; `NOT_DETECTED` (no sysfs root) → `COMPLETED`, no error, no capability; `NOT_SUPPORTED` → `COMPLETED` plus `GPU.PCI_SCAN`; `PERMISSION_DENIED`/`COLLECTION_FAILED` → `PARTIAL` plus one host-wide `CollectionError`; never a collector-owned `FAILED`. **(5)** Each omitted device (`INVALID_ADDRESS`, `ATTRIBUTE_UNREADABLE`, `ATTRIBUTE_MALFORMED`) → one `CollectionError` (`pci`, `device_ref` null, the failure's status and code, `<address> <attribute or 'address'>: <message>`, `report_path` `/gpu/pci_observations`) in the scan's order. **(6)** `DRIVER_LINK_UNREADABLE` keeps the device and its failed `kernel_driver`: run `PARTIAL`, no `CollectionError` (field-level, data model §3.4); `NO_DRIVER_BOUND` isn't a failure. **(7)** Any omitted device makes the run `PARTIAL`, even with status `NOT_DETECTED`. **(8)** The PCI collector emits only `GPU.PCI_SCAN`; `NVIDIA.ALL` belongs to IMP-K02/IMP-P01. **(9)** Deferred: the core/factory (IMP-K02, IMP-K08) must fail closed, so an unclassified or unavailable Linux environment is never treated as `NATIVE_LINUX`; diagnostics and readiness must not read a `PARTIAL` scan as proof of absence and must handle a `NOT_DETECTED` scan explicitly (KI-32, KI-33). | A listed device that can't be observed means the enumeration is incomplete, while a missing driver link is a field the observation already reports. `NOT_DETECTED` and `NOT_SUPPORTED` describe the host, as in `SystemCollector` and the NVML fixtures (FX03, FX04b, FX17). `FAILED` stays reserved for the guard (data flow §3). A PCI address can't be a `device_ref`, so PCI errors are host-wide. | Approved and applied (Stage 6.3) |
| DEC-44 | Integration milestone (IMP-K08, IMP-K02, IMP-P01). **(1)** One `Collector` contract (`collectors/base.py`): `name`, constructor-injected adapters and `Clock`, `collect() -> CollectorOutcome`, `unavailable(ErrorDetail)`; replaces the §4.2 sketch; `FakeCollector` conforms. **(2)** `run_guarded` is the only producer of `FAILED`; it contains every `Exception` (including `AdapterUsageError` and contract violations) as one host-wide `UNHANDLED_EXCEPTION` error with the collector's report path. **(3)** Order `system`, `pci`, `nvml`, `gpu_hardware`; the Linux environment comes from the system section, fail-closed (`LINUX` → `UNKNOWN`, `WSL2` → `WSL2`, else `None`), and the PCI scanner is built from it; the production set holds `DeferredPciScanner` until then. **(4)** The core owns the NVML session (`NvmlSessionStep`, run `nvml`; `FailedNvmlSession` if opening raised) and closes it on every path. **(5)** `GpuHardwareCollector` consumes the PCI section, never scans, keeps it in its fallback, owns `NVIDIA.ALL`; NVML count → NVML devices, else the `nvidia-smi` query. **(6)** `assemble_report` rejects duplicate runs and capability records. **(7)** Collectors and adapters that don't exist yet are explicit: production stand-ins report `NOT_SUPPORTED`/`NOT_IMPLEMENTED` (`adapters/unimplemented.py`), sections without a collector are placeholders (`NOT_SUPPORTED`/`NOT_IMPLEMENTED`; PyTorch `SKIPPED`, the only status its model allows besides failures) with no run and a stated limitation; Mode B raises `NotImplementedError`. **(8)** `api.inspect()` is the public entry point; the CLI stays a placeholder. | The tested collectors become a working, validated pipeline without inventing results: absent adapters and collectors are reported as not implemented, never as an absent GPU or a failure. Building the PCI scanner after classification makes a guessed environment impossible. | Applied (IMP-K08/K02/P01); the NVML, `nvidia-smi` and command-runner stand-ins replaced by DEC-45, the PyTorch probe still a stand-in |
| DEC-45 | Production NVIDIA adapters (IMP-A05, A01, A02). **(1)** `SubprocessCommandRunner` is the only code that starts processes: list argv, no shell, no stdin, bounded timeout, captured output, environment never recorded; malformed calls are `AdapterUsageError`. **(2)** `PynvmlAdapter` loads the existing required binding (`nvidia-ml-py`, DEC-05; no new dependency) lazily through an injectable loader, maps `NVMLError`s by value to the data model §3.3 codes, keeps one init/at most one shutdown per session, swallows shutdown errors into `shutdown_errors`, and falls back to older NVML calls on `FUNCTION_NOT_FOUND`. **(3)** `NvidiaSmiCli` uses the runner and the documented CSV query (EV-02); a structurally bad output fails the whole query, a bad value only its field; an unknown query field is `NOT_SUPPORTED`. **(4)** macOS is gated in both adapters (`PLATFORM_NOT_SUPPORTED`, nothing loaded or run), matching FX13. **(5)** Fallback by precedence, never merge (§4.4): NVML whenever it can count devices; `nvidia-smi` only when it can't; indices never matched across tools. **(6)** The production factory wires them; only the PyTorch probe stays a stand-in (IMP-A07). | NVML is the authoritative, structured source (DEC-04); merging two inventories by index would risk duplicate or mismatched devices, and partial NVML evidence is already per-device. A single runner keeps process creation auditable. | Applied (IMP-A05, A01, A02) |
| DEC-46 | Driver and runtime collectors (IMP-K03, IMP-K04), KI-34, KI-16. **(1)** `DriverInfo` is built in two runs: `driver` (NVML facts from the shared session, system calls only) and `nvidia_smi` (location and header, after the session is closed), with the precedence of §4.4: NVML first, then the `nvidia-smi` query, then its header; `nvidia_smi_*` always hold what `nvidia-smi` said and a disagreement with NVML is noted, never resolved silently. Neither run emits capability records. **(2)** `RuntimeCollector` samples every inventoried device from the inventory's own source; a device whose handle fails or that reports `GPU is lost` gets an unavailable record and one `CollectionError` (`/runtime/<i>`), with no further calls and no partial values; other failures stay field-level. It owns `RUNTIME.*` (rules in §4.4, taken from the fixtures). **(3)** `NvidiaSmiCli` computes `locate()`, `version()` and `query_gpus()` once per adapter, so each `nvidia-smi` command runs at most once per inspection. **(4)** Order: `driver`, `gpu_hardware`, `runtime` inside the NVML session window, then `nvidia_smi`. **(5)** Test kit: `FakeNvml` and `StubPynvml` model device loss as a lifecycle (`lost_after_handles`, `lost_on`) instead of static faults; FX10's fault *input* is corrected to "lost after enumeration" (its expected snapshot is unchanged), resolving KI-34. **(6)** OQ-07/KI-16: the `nvidia-ml-py>=12.535.108` floor is kept, confirmed at the binding level. **(7)** Scope: CUDA Toolkit/`nvcc` discovery stays with IMP-K05/IMP-A06 (OQ-10); the CUDA runtime library and CUDA usability are not observed (PyTorch probe, IMP-A07/K06) and the report's limitation says so. No model or schema change. | Reuses the one NVML session and the existing adapters, keeps every source's own evidence visible, and makes a lost GPU a per-device fact instead of a run failure (FR-13, AC54). The old FX10 faults failed the enumeration's memory read that FX10's own snapshot records as collected, so no implementation could reproduce it; the lifecycle expresses the scenario the fixture describes. | Applied (IMP-K03, IMP-K04); KI-34 and KI-16 resolved |
| DEC-47 | OQ-10 and the CUDA environment (IMP-A06, IMP-K05). **(1) Abstraction:** a `ToolkitLocator` protocol in the `AdapterSet` (`toolkit`), implemented by `CudaToolkitLocator` on the shared `CommandRunner` and a minimal read-only `ToolkitFiles` seam (`kind`, `list_dir`, bounded `read_text`, `realpath`, path joins; `kind` raises instead of turning a permission error into "absent") that only the locator uses; not shared with `/proc` reads (`HostApi` stays private to the platform adapter). The collector reads variables through `PlatformAdapter` and passes them to the locator as hints. **(2) Sources:** `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V*` (Windows), `nvcc` on `PATH`, and on Linux/WSL2 one non-recursive listing of `/usr/local` (`cuda`, `cuda-<version>`), as FR-07 and §4.6 require; all signals recorded, all candidates reported. Rejected: `version.txt` (not specified), libraries, NVML/`nvidia-smi` CUDA versions (FR-07), Windows `%ProgramFiles%` (not allowlisted), recursive search, a configured path (no option). **(3) KI-28:** `PlatformAdapter.versioned_cuda_path_names()` lists only names matching `CUDA_PATH_VERSIONED_REGEX`, Windows only (§5.5); no other name is kept or any other value read. **(4) Semantics:** resolved roots, aliases merged, ordered by path; `version.json` (≤ 64 KiB, `X.Y[.Z…]`) else `<root>/bin/nvcc --version`, never a directory name; malformed metadata is `PARSE_ERROR`, unreadable is its own status, never `NOT_DETECTED`; root and compiler are separate evidence; the active `nvcc` is run once and reused. **(5) Platforms:** Linux (any classification) and WSL2 use the Linux signals (user-space files, no hardware claim); Windows without a default location; macOS/`OTHER` `NOT_SUPPORTED` with `CUDA.TOOLKIT`. **(6) Schema:** no change. **(7) Fixture corrections** (inputs and expected values that contradicted the spec): every fixture's `environment` now lists the whole allowlist (data model §5.1/§5.5 and the implemented `collect_environment` read all 11 names, unset → `NOT_DETECTED`; the fixtures listed 3–5); the `/usr/local/cuda-12.4` candidates add `DEFAULT_LOCATION` (§4.6 consults the defaults and records every signal); FX12 adds `NVCC_ON_PATH` (its active `nvcc` is under the root, as in FX01) and `CUDA_PATH_VERSIONED` (`CUDA_PATH_V12_4` points at the root), and its `nvcc` paths end in `.exe`. | The specification already named the sources and the model; what was open was the seam, which the OQ-10 recommendation and the PCI precedent (a production adapter over a fake reader) settle. Reporting every candidate keeps conflicting installations visible for D03. | Applied (IMP-A06, IMP-K05); OQ-10 and KI-28 resolved |
| DEC-48 | PyTorch probe and collector (IMP-A07, IMP-K06). **(1) Probes:** `SubprocessTorchProbe` (default; the CLI always) runs `[sys.executable, "-m", "gpu_system_inspector.adapters._torch_probe_main"]` through the shared `CommandRunner` with `pytorch_timeout_s` (DEC-07); `InProcessTorchProbe` (Python API opt-in, `pytorch_probe_mode = IN_PROCESS`) runs the same probe function in the calling process. The factory picks one from the options. Both read `importlib.metadata.version("torch")` first and never start or import anything when it is missing. **(2) Protocol 2:** the child reports a device-enumeration failure in a new `device_error` key, so package and build facts survive it; protocol 1 had no place for it. `PROBE_PROTOCOL_VERSION` becomes 2; the decoder still rejects any other version. **(3) No execution:** the probe imports torch, reads versions and build flags, calls `is_available()` (warnings captured), `device_count()` and the device properties; it allocates no tensor, runs no kernel and never calls `mem_get_info` (data flow §5, C10). `cuda_available` is PyTorch's own answer, never "CUDA works". **(4) Correlation (IMP-K07):** `correlated_device_ref` is `NOT_SUPPORTED`/`NOT_IMPLEMENTED` until the correlation collector exists; no PCI, UUID or index matching here. The source facts it needs (`uuid`, `pci_bus_id`) are collected. **(5) Collector:** `FrameworkCollector` (run `pytorch`, after `cuda_environment`, outside the NVML session) owns no capability records. Not installed → `NOT_INSTALLED` (run `COMPLETED`); `include_pytorch = false` → `SKIPPED`, nothing read or run (AC59); a probe that ran but failed (import error, timeout, exit, protocol) → section `COLLECTION_FAILED` with the distribution version kept, run `PARTIAL`, one error at `/frameworks/pytorch`; a device-enumeration failure → section `COLLECTED`, `device_count` failed, run `PARTIAL`, one error at `/frameworks/pytorch/devices`. A timeout reads "PyTorch probe exceeded pytorch_timeout_s (N s)". **(6) Allocator (FR-10, DEC-08):** subprocess → `NOT_APPLICABLE` with the DEC-08 note; in-process → collected (note "Calling process only") only if CUDA was already initialised in the caller before the probe, else `NOT_APPLICABLE`. **(7) In-process limits:** no timeout can be enforced, and reading device properties may initialise CUDA in the caller; the CLI never uses it. No dependency, model or schema change. | The specification already prescribes the subprocess, its protocol and the facts; what it left open was how an enumeration failure keeps the build facts, and how K06 reports a field only K07 can fill. Reporting the placeholder keeps the K07 boundary visible instead of guessing a match. | Applied (IMP-A07, IMP-K06); KI-35 resolved. Point **(4)** (the `NOT_IMPLEMENTED` correlation placeholder) **superseded by DEC-49 (5)** |
| DEC-49 | Device correlation (IMP-K07), refining DEC-09 without changing its order. **(1) Inputs:** the `gpu_hardware` run's `GPUInventory` and the `pytorch` run's `FrameworkInfo`, nothing else: no NVML, no probe, no process; the collector (run `correlation`, last) returns the PyTorch section with each device's `correlated_device_ref` set, and owns no capability records. **(2) Identifiers:** UUIDs normalised as data model §5.4 states (lower-case, strip a leading `GPU-`, must then be 8-4-4-4-12 hex); PCI bus IDs parsed as `domain:bus:device.function` (domain 4 or 8 hex digits, as PyTorch and NVML print it; bus 2, device 2, function 1) and compared by value, so `00000000:01:00.0` = `0000:01:00.0`. An identifier that is missing, not collected or doesn't parse is unusable, never a wildcard, and no component is guessed. **(3) Order:** by UUID among NVML devices with a usable UUID; only if that finds nothing, by PCI bus ID among the NVML devices a UUID can't rule out (both UUIDs usable and different means different devices). Never by index, name or memory. **(4) Results:** a unique match → `COLLECTED` = the NVML `device_ref`, source `INSPECTOR`, method `uuid` or `pci_bus_id` (with a note if the other identifier contradicts it; UUID still wins); no NVML inventory (`inventory_source` isn't NVML) → `NOT_APPLICABLE` "No NVML inventory to correlate with" (FX04a); otherwise `NOT_DETECTED` with a code: `NO_MATCH`, `NO_IDENTIFIER` (PyTorch exposes no usable UUID or PCI bus ID), `AMBIGUOUS_MATCH` (several NVML candidates, or one NVML device claimed by several PyTorch devices: none of them is chosen), `UNIDENTIFIED_CANDIDATES` (no match, but some NVML device has no usable identifier and can't be ruled out). These are field-level facts: the run is `COMPLETED` with no `CollectionError`; the guard's fallback keeps the PyTorch facts and fails only `correlated_device_ref`. **(5) K06 adjustment:** the PyTorch collector's interim value becomes `NOT_APPLICABLE` "Set by the correlation run (DEC-09)" (it was `NOT_IMPLEMENTED`, DEC-48 (4)); the correlation run always replaces it. No model or schema change. | DEC-09 fixed the order but not how the two tools' PCI formats compare, nor what an ambiguous or contradictory match reports; choosing nothing when the evidence doesn't single out one device keeps every reported relationship provable. | Applied (IMP-K07) |
| DEC-50 | Demo-ready completion sprint (IMP-D00–D08, IMP-R01, IMP-P03–P05, IMP-C01–C03). **(1) Engines:** diagnostics and readiness are pure functions of the snapshot (`models` only, enforced by `test_boundaries`); `Inspector.run` runs readiness (Mode B) and then the diagnostics engine, so `api.inspect` and the CLI return complete reports. The `NotImplementedError` for Mode B and the not-implemented limitation are gone; `SCOPE_LIMITATION` states that findings and readiness come from one snapshot and that CUDA execution isn't checked. **(2) Absence:** "no NVIDIA GPU" (D01 row 7, D04 row 7, `NO_NVIDIA_DEVICE`) is only concluded from macOS, NVML counting zero devices, or a `COLLECTED` **and complete** PCI scan without an NVIDIA device while NVML is `NOT_DETECTED`; a `PARTIAL` scan (KI-32) or a `NOT_DETECTED`/`NOT_SUPPORTED` scan (KI-33) leads to D04 row 8 (`INFO`, MEDIUM, "no NVIDIA GPU is accessible") and `DEVICE_PRESENCE_UNKNOWN`. **(3) Fixture correction:** D02's mapping table gives `UNKNOWN` confidence `NOT_APPLICABLE`, like every `UNKNOWN` row of the specification; the fixture expectations for FX04b, FX09 and FX14 (Mode B) had `HIGH` only because `expect()` defaults to it. `readiness_run()` now defaults `UNKNOWN` D02 to `NOT_APPLICABLE`; the specification is unchanged. **(4) Engine output:** `evaluate()` returns `DiagnosticsResult(findings, errors)`; a rule that raises, returns nothing, returns another check's finding or two findings for one device is contained as `UNKNOWN` plus a `diagnostics.Dxx` collection error. **(5) Redaction:** applied to the validated report before rendering, so JSON, text and CLI messages agree; implemented over the JSON tree (every string except structural keys such as statuses, codes and pointers), re-validated with the strict decoder. Path placeholders are numbered (`<redacted:path:N>`) like device IDs, because the models require unique toolkit roots (and PCI addresses), which a single `<redacted:path>` would violate; data model §8 updated. Device ordinals normalise UUID (`GPU-` prefix, case) and PCI (domain width) spellings, so NVML, sysfs and PyTorch forms of one device share a number. The user name and home directory are inputs supplied by the CLI and never stored. **(6) CLI:** `inspect` is implemented per the CLI specification; exit code 6 (`NOT_IMPLEMENTED`) is removed as §6 required. `run_inspection` is the seam tests and the synthetic showcase replace (fixture adapters, `data_origin = FIXTURE`, banner). **(7) Text:** reference URLs print on their own line so lines stay ≤ 100 characters. **(8) `READINESS.MIG` owner:** FX14's expected snapshot (and AC39) contain `READINESS.MIG` for a MIG-enabled device, but no collector emitted it; the GPU hardware collector, which reads `mig_mode`, now emits it per MIG-enabled device ("MIG instances are not enumerated in V1."), in both modes, matching the fixture. Needs owner review. |

## 7. Security and privacy considerations

- Subprocesses are run without a shell, with fixed argument vectors, timeouts and output caps.
- Only allowlisted environment variables are read into the report.
- Process names come from NVML `nvmlSystemGetProcessName` or `/proc/<pid>/comm` (basename only), never full command lines.
- Hostname, username, MAC addresses and serial numbers are never collected. The username/home directory is read only to drive redaction and is never emitted.
- No network access and no telemetry.

## 8. Future reuse candidates (recorded only; not extracted)

Under `docs/package-policy.md`, nothing is extracted until a second InferForge project
needs it. When that happens, these interfaces are the likely seams. They should be added
to `docs/component-reuse-map.md` "Extraction candidates" **once the code exists**.

| Capability | Module | Possible future package | Likely consumers |
| ---------- | ------ | ----------------------- | ---------------- |
| `Observed[T]`, `CollectionStatus`, `Source`, `CollectionError` | `models/observed.py`, `models/enums.py` | `inferforge-observation` | Any project reporting environment or telemetry facts |
| NVML adapter (read-only, typed, status-mapped) | `adapters/nvml.py` | `inferforge-nvml` | Inference server metrics, benchmarking projects |
| PyTorch subprocess probe | `adapters/torch_probe.py` | `inferforge-framework-probe` | Benchmarking, model-loading projects |
| Readiness engine (threshold vs observed capacity) | `readiness/` | `inferforge-gpu-readiness` | Model serving / scheduling projects |
| Report envelope, schema versioning, redaction, atomic writer | `reporting/` | `inferforge-reporting` | Every CLI project emitting reports |
| CLI conventions (exit codes, `--json PATH\|-`, logging setup) | `cli/` | `inferforge-cli-common` | Other InferForge command-line projects (each keeps its own script name, as DEC-03 does) |

## 9. Project configuration (applied in Stage 1, 2026-11-04)

The metadata-only configuration (DEC-01) has been replaced by a real package. The
authoritative file is `../pyproject.toml`. Its key decisions:

| Setting | Value | Reason |
| ------- | ----- | ------ |
| Build backend | `hatchling>=1.27` | Workspace default; 1.27+ supports the SPDX `license = "Apache-2.0"` expression. |
| `name` / import package | `01-gpu-system-inspector` / `gpu_system_inspector` | DEC-02. The wheel target lists `src/gpu_system_inspector` explicitly because the two names differ. |
| `requires-python` | `>=3.11` | DEC-21 / WSD-01. |
| `dependencies` | `nvidia-ml-py>=12.535.108` | DEC-05. The floor is **confirmed at the binding level** (OQ-07, KI-16, DEC-46): every call the adapter needs exists in 12.535.108, whose only missing preferred call (`nvmlDeviceGetTemperatureV`) falls back to `nvmlDeviceGetTemperature`, and the NVML adapter, collector and pipeline tests pass with it installed. It matches the floor documented by gpustat (EV-17) so the two can share an environment. Behaviour against a live driver is HW-01. |
| PyTorch | not declared | DEC-26. |
| `[project.scripts]` | `inferforge-gpu = "gpu_system_inspector.cli.main:main"` | DEC-03. |
| Classifier `Private :: Do Not Upload` | set | PyPI rejects uploads carrying this classifier, which guards against accidental publication. |
| `[dependency-groups] test` | `pytest>=8` (`jsonschema` added when the schema exists) | `projects/README.md` convention. |
| `[tool.uv] default-groups` | `["test"]` | So `uv run pytest` from the project directory has pytest without extra flags. |
| `[tool.pytest.ini_options]` | `testpaths = ["tests"]`, importlib mode, `-m 'not hardware'`, `hardware` marker | CON-06, DEC-20. |

Ruff uses the root configuration. `gpu_system_inspector` was added to the root
`known-first-party` list, following the existing convention for the script modules.
`scripts/check_dependencies.py` passes with the project as an editable member.
