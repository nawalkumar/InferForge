# Data Model Specification — Project 01: GPU System Inspector

Status: **Specification (approval pending)**. Report schema version defined here: **`1.0`**
(frozen when AC09 reaches TESTED and the specification is approved; see §7.3).

## 1. Conventions

- Models are immutable `@dataclass(frozen=True, slots=True)` classes. They target the
  workspace Python floor, 3.11 (DEC-21), so 3.11 features such as `enum.StrEnum` are
  available. Nothing newer than 3.11 may be used.
- Python attribute names = JSON keys = `snake_case`.
- `Observed[T]` (§3.2) wraps every value that is *collected from the environment*. Plain
  types are used only for values the inspector itself defines (indices it enumerated,
  identifiers, enums it derives, configuration echoes).
- Units (DEC-11): bytes as `int`; temperatures in degrees Celsius (`celsius`); power in
  watts as `float` (NVML reports milliwatts; converted ÷1000); percentages `0–100`
  (`percent`); durations in milliseconds (`ms`); timestamps RFC 3339 UTC with `Z`.
- Privacy classes (§8) are marked in the tables: **P** = path, **N** = process name,
  **D** = device identifier, **–** = not sensitive.
- `device_ref` is a report-internal device key: `"nvml:<index>"` for NVML-enumerated
  devices, `"smi:<index>"` for devices known only from the `nvidia-smi` fallback. Runtime
  metrics, findings and readiness use it to refer to devices.
- **Device ordering is natural-numeric.** Wherever a list is "sorted by `device_ref`", the
  key is (prefix, integer index), so `nvml:2` precedes `nvml:10`. `null` refs sort first
  where a list mixes system-wide and per-device entries. Models apply the order at
  construction, so it doesn't depend on the order producers supply entries, and repeated
  serialisations are byte-identical.

## 2. Shared enumerations

All enumerations serialise as their upper-case member name.

| Enum | Members | Used by |
| ---- | ------- | ------- |
| `Source` | `PLATFORM` (Python stdlib), `OS` (`/proc`, sysfs, Windows API, `sysctl`), `ENVIRONMENT` (env vars), `NVML`, `NVIDIA_SMI`, `CUDA_TOOLKIT` (`version.json`, `nvcc`), `PYTORCH`, `INSPECTOR` (derived by this tool) | `Observed.source`, evidence |
| `DataOrigin` | `LIVE`, `FIXTURE` | report (DEC-17) |
| `InspectionMode` | `GENERAL`, `READINESS` | report |
| `PlatformProfile` | `LINUX`, `WSL2`, `WINDOWS`, `MACOS`, `OTHER` | `SystemInfo` |
| `DiagnosticStatus` | `PASS`, `WARNING`, `FAIL`, `INFO`, `NOT_APPLICABLE`, `UNKNOWN` | findings |
| `Severity` | `NONE`, `LOW`, `MEDIUM`, `HIGH` | findings |
| `Confidence` | `HIGH`, `MEDIUM`, `LOW`, `NOT_APPLICABLE` | findings |
| `EvidenceKind` | `OBSERVATION`, `INFERENCE` | evidence |
| `ReadinessStatus` | `READY`, `NOT_READY`, `UNKNOWN` | readiness |
| `ReadinessReason` | device: `SUFFICIENT_FREE_MEMORY`, `INSUFFICIENT_FREE_MEMORY`, `INSUFFICIENT_TOTAL_MEMORY`, `FREE_MEMORY_UNAVAILABLE`, `MIG_ENABLED_UNSUPPORTED`; overall: `AT_LEAST_ONE_DEVICE_READY`, `NO_DEVICE_READY`, `SOME_DEVICES_UNKNOWN`, `NO_NVIDIA_DEVICE`, `PLATFORM_NOT_SUPPORTED`, `DEVICE_PRESENCE_UNKNOWN` | readiness |
| `CollectorRunStatus` | `COMPLETED`, `PARTIAL`, `FAILED`, `SKIPPED` | `collector_runs` |
| `ProbeMode` | `SUBPROCESS`, `IN_PROCESS` | PyTorch probe |
| `ProcessKind` | `COMPUTE`, `GRAPHICS`, `COMPUTE_AND_GRAPHICS` | processes |
| `ToolkitDiscovery` | `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_VERSIONED`, `NVCC_ON_PATH`, `DEFAULT_LOCATION` | toolkit candidates |

## 3. Observed values and failures

### 3.1 `CollectionStatus` (the only status vocabulary for collected values)

| Status | Meaning | Example |
| ------ | ------- | ------- |
| `COLLECTED` | Value obtained successfully. **Only status with a non-null value.** | NVML returned total memory. |
| `NOT_INSTALLED` | A required *Python* dependency or tool package is not installed. | `nvidia-ml-py` or `torch` distribution absent. |
| `NOT_DETECTED` | The thing being looked for isn't present on this system (searched and not found). | No NVML shared library or loaded driver; no `nvidia-smi`; no CUDA toolkit; env var unset. |
| `NOT_SUPPORTED` | Present, but this query isn't supported on this platform, device, driver or configuration. | NVML `NOT_SUPPORTED`; utilization on WSL2; per-process memory under WDDM; NVIDIA on macOS. |
| `PERMISSION_DENIED` | The query exists but the caller lacks privileges. | NVML `NO_PERMISSION`; `[Insufficient Permissions]` from `nvidia-smi`. |
| `COLLECTION_FAILED` | An attempt was made and failed unexpectedly (error, timeout, malformed output, crash). | NVML `GPU_IS_LOST`; probe timeout; unparsable `nvcc` output. |
| `SKIPPED` | Deliberately not collected because of a user option. | `--no-pytorch`, `--no-processes`. |
| `NOT_APPLICABLE` | Not meaningful in this context because a precondition doesn't hold. | Allocator stats in the CLI (DEC-08); `build_cuda_version` for a CPU-only build. |
| `UNKNOWN` | The inspector can't classify the outcome. Should be rare. Its presence is itself a defect signal. | An error that matches no mapping rule. |

`COLLECTED`, `SKIPPED` and `NOT_APPLICABLE` are additions to the six failure statuses
required by the product brief. `COLLECTED` is needed for successful values; `SKIPPED` and
`NOT_APPLICABLE` prevent user opt-outs and meaningless metrics being misreported as
failures.

### 3.2 `Observed[T]`

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `value` | `T \| null` | yes | The collected value. **Non-null iff `status == COLLECTED`** (DEC-10). Never a placeholder such as `0`, `-1`, `""` or `"N/A"`. |
| `status` | `CollectionStatus` | yes | See §3.1. |
| `source` | `Source \| null` | yes | Where the value came from (or would have come from). `null` only when no source was attempted (for example `SKIPPED`). |
| `method` | `str \| null` | yes | Specific API/command/file, e.g. `"nvmlDeviceGetMemoryInfo_v2"`, `"nvidia-smi --query-gpu=memory.free"`, `"/proc/meminfo:MemTotal"`, `"torch.version.cuda"`. |
| `unit` | `str \| null` | yes | `bytes`, `celsius`, `watts`, `percent`, `ms` or `null` for unitless values. |
| `error` | `ErrorDetail \| null` | yes | **Required** for `NOT_INSTALLED`, `NOT_DETECTED`, `NOT_SUPPORTED`, `PERMISSION_DENIED`, `COLLECTION_FAILED` and `UNKNOWN`. **Must be `null`** for `COLLECTED` and `SKIPPED`. **Optional** for `NOT_APPLICABLE`, which instead requires `note` (see below). |
| `note` | `str \| null` | yes | Clarification of meaning or precision (e.g. `"Converted from MiB; precision 1 MiB"`, `"Newest CUDA version supported by the driver; not the installed toolkit"`). |

**Why `NOT_APPLICABLE` needs a `note` but not an `error`.** `NOT_APPLICABLE` says that a
value is meaningless in this context because a precondition doesn't hold. Nothing was
attempted and nothing failed. Examples: allocator statistics in a fresh probe process
(DEC-08), or `build_cuda_version` for a CPU-only PyTorch build. An `error` would wrongly
suggest a collection failure, so the explanation goes in `note`, which is mandatory. An
`error` may still be attached when the inapplicability *results from* a failure elsewhere
and that failure is worth quoting, for example `correlated_device_ref` being
`NOT_APPLICABLE` because the NVML inventory itself failed. The models, the decoder and
the JSON Schema all enforce these rules.

`ErrorDetail = { code: str, message: str }`. `code` is a stable machine string: NVML error
names (`NVML_ERROR_NOT_SUPPORTED`), Python exception names (`ImportError`,
`FileNotFoundError`), or inspector codes (`TIMEOUT`, `PARSE_ERROR`, `PROBE_EXIT_<n>`,
`PROBE_PROTOCOL_ERROR`, `IMPORT_ERROR`, `UNHANDLED_EXCEPTION`, `PLATFORM_NOT_SUPPORTED`,
`VALUE_NOT_AVAILABLE`, `FIELD_NOT_FOUND`, `NO_DRIVER_BOUND`, `LINUX_ENVIRONMENT_UNKNOWN`,
`FALLBACK_FIELD_UNAVAILABLE`, `NOT_IMPLEMENTED`, and the `nvidia-smi` codes of §3.3). `message` is human-readable and is redacted
with the rest of the report.

`FALLBACK_FIELD_UNAVAILABLE` (IMP-K02): a `GPUDeviceInfo` field that the `nvidia-smi`
fallback query doesn't carry (`pci_device_id`, `pci_subsystem_id`, `compute_capability`,
`mig_mode`); always `NOT_SUPPORTED`, source `NVIDIA_SMI`.

`NOT_IMPLEMENTED` (DEC-44): the mechanism or collector isn't part of this version yet, so
nothing was attempted (`method` is `null`); always `NOT_SUPPORTED`. It never means a GPU,
driver or tool is absent. Used by the production stand-in adapters and the placeholder
sections of collectors that don't exist yet; the report states it in `limitations`.

`NO_DRIVER_BOUND` (Stage 6.1): `PciObservation.kernel_driver` is `NOT_DETECTED` with this
code when the device's sysfs `driver` symlink doesn't exist, meaning no kernel driver is
bound. This is an observation about the device, not a collection failure. An unreadable
link is a failure with its own status instead (`PERMISSION_DENIED`, `COLLECTION_FAILED`).

`LINUX_ENVIRONMENT_UNKNOWN` (Stage 6.2): the host is classified as `UNKNOWN` Linux
(`linux_environment`, §4.1), so a native-Linux-only check (the sysfs PCI scan) can't safely
be performed. The status is always `NOT_SUPPORTED`. Known unsupported environments (WSL1,
WSL2, Windows, macOS and other non-Linux profiles) keep `PLATFORM_NOT_SUPPORTED`.

JSON example:

```json
{"value": 25757220864, "status": "COLLECTED", "source": "NVML",
 "method": "nvmlDeviceGetMemoryInfo_v2", "unit": "bytes", "error": null, "note": null}
```

```json
{"value": null, "status": "NOT_SUPPORTED", "source": "NVML",
 "method": "nvmlDeviceGetUtilizationRates", "unit": "percent",
 "error": {"code": "NVML_ERROR_NOT_SUPPORTED", "message": "Not Supported"},
 "note": "GPU utilization queries are documented as not yet supported on WSL2"}
```

### 3.3 Error-to-status mapping (normative)

| Origin | Condition | Status | `error.code` |
| ------ | --------- | ------ | ------------ |
| Python import | `ImportError` / `ModuleNotFoundError` for `pynvml` | `NOT_INSTALLED` | `ImportError` |
| `importlib.metadata` | `PackageNotFoundError` for `torch` | `NOT_INSTALLED` | `PackageNotFoundError` |
| PyTorch probe | distribution present but `import torch` failed | `COLLECTION_FAILED` | `IMPORT_ERROR` |
| NVML | `NVMLError_LibraryNotFound` | `NOT_DETECTED` | `NVML_ERROR_LIBRARY_NOT_FOUND` |
| NVML | `NVMLError_DriverNotLoaded` | `NOT_DETECTED` | `NVML_ERROR_DRIVER_NOT_LOADED` |
| NVML | `NVMLError_NoPermission` | `PERMISSION_DENIED` | `NVML_ERROR_NO_PERMISSION` |
| NVML | `NVMLError_NotSupported` | `NOT_SUPPORTED` | `NVML_ERROR_NOT_SUPPORTED` |
| NVML | `NVMLError_FunctionNotFound` (driver older than API) | `NOT_SUPPORTED` | `NVML_ERROR_FUNCTION_NOT_FOUND` |
| NVML binding | field returned as `None` (binding maps `NVML_VALUE_NOT_AVAILABLE` to `None`, EV-11) | `NOT_SUPPORTED` | `VALUE_NOT_AVAILABLE` |
| NVML | `NVMLError_GpuIsLost`, `NVMLError_Timeout`, `NVMLError_Unknown`, any other | `COLLECTION_FAILED` | NVML error name |
| Subprocess | executable not found | `NOT_DETECTED` | `FileNotFoundError` |
| Subprocess | `PermissionError` | `PERMISSION_DENIED` | `PermissionError` |
| Subprocess | `TimeoutExpired` | `COLLECTION_FAILED` | `TIMEOUT` |
| Subprocess | non-zero exit | `COLLECTION_FAILED` | `EXIT_<n>` |
| Parsers | unparsable output | `COLLECTION_FAILED` | `PARSE_ERROR` |
| Parsers | well-formed source without the expected entry (e.g. no `model name` in `/proc/cpuinfo`) | `NOT_DETECTED` | `FIELD_NOT_FOUND` |
| `nvidia-smi` | `[N/A]` or `[Not Supported]` | `NOT_SUPPORTED` | `NVIDIA_SMI_NA` / `NVIDIA_SMI_NOT_SUPPORTED` |
| `nvidia-smi` | `[Insufficient Permissions]` | `PERMISSION_DENIED` | `NVIDIA_SMI_INSUFFICIENT_PERMISSIONS` |
| `nvidia-smi` | `[Unknown Error]` or `[GPU is lost]` | `COLLECTION_FAILED` | `NVIDIA_SMI_UNKNOWN_ERROR` / `NVIDIA_SMI_GPU_IS_LOST` |
| `nvidia-smi` | query field unknown to an older driver | `NOT_SUPPORTED` | `NVIDIA_SMI_NOT_SUPPORTED` |
| Platform gate | capability not available on this OS profile or known Linux environment (e.g. the PCI scan on WSL1, WSL2, Windows, macOS) | `NOT_SUPPORTED` | `PLATFORM_NOT_SUPPORTED` |
| Linux-environment gate | native-Linux-only capability on a host whose `linux_environment` is `UNKNOWN` | `NOT_SUPPORTED` | `LINUX_ENVIRONMENT_UNKNOWN` |
| Files | `FileNotFoundError` for a probed path | `NOT_DETECTED` | `FileNotFoundError` |
| Files | `PermissionError` | `PERMISSION_DENIED` | `PermissionError` |
| Subprocess / files | any other `OSError` (for example an exec format error) | `COLLECTION_FAILED` | exception type name |
| Anything else | unclassified | `UNKNOWN` | exception type name |

Adapters apply the NVML, subprocess, file and exception rows through `adapters/outcomes.py`
(`nvml_error_status`, `exception_outcome`, `require_success`). The gate rows are applied
where the gate lives: the PCI scan uses `PCI_SCAN_UNSUPPORTED_CODES` in
`adapters/pci_sysfs.py`, and OS facts that a profile doesn't provide get
`PLATFORM_NOT_SUPPORTED` in `adapters/platform_os.py`. A test reads this table and checks
only the NVML and exception rows against `outcomes.py`. The PCI gate codes are pinned in code
by `tests/unit/test_pci_capability_policy.py`, not checked against this document.
The `OSError` row (Stage 4) covers a process or file access that was attempted and failed for
an operating-system reason other than "not found" or "permission". That matches the
definition of `COLLECTION_FAILED`, and `UNKNOWN` stays reserved for exceptions that aren't
I/O errors at all.

### 3.4 `CollectionError` and `UnsupportedCapability`

`CollectionError`: one record per failed collector, failed device, or failed capability
that a user should know about. Field-level `Observed` failures inside an otherwise
successful section do **not** each create a `CollectionError`.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `collector` | `str` | `system`, `pci`, `nvml`, `driver`, `gpu_hardware`, `runtime`, `nvidia_smi`, `cuda_environment`, `pytorch`, `correlation`, `readiness`, `diagnostics.D0x` |
| `device_ref` | `str \| null` | Affected device, if any. |
| `status` | `CollectionStatus` | Never `COLLECTED`. |
| `error_code` | `str` | As in §3.2. |
| `message` | `str` | Redactable. |
| `report_path` | `str \| null` | JSON Pointer (RFC 6901) to the affected part of the report. |

`UnsupportedCapability`: a capability that isn't available on this platform or
configuration. It is expected rather than a failure.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `capability` | `str` | Dotted ID, e.g. `NVIDIA.ALL`, `GPU.PCI_SCAN`, `RUNTIME.GPU_UTILIZATION`, `RUNTIME.PROCESSES`, `RUNTIME.PROCESS_MEMORY`, `SYSTEM.MEMORY_AVAILABLE`, `CUDA.TOOLKIT`, `READINESS.MIG`. |
| `status` | `CollectionStatus` | Usually `NOT_SUPPORTED`; `PERMISSION_DENIED` or `NOT_INSTALLED` when that is the blocking reason. |
| `device_ref` | `str \| null` | Device scope, if any. |
| `reason` | `str` | Human explanation with evidence reference (e.g. "Documented as unsupported on WSL2 (CUDA on WSL User Guide)"). |

**`GPU.PCI_SCAN` and `NVIDIA.ALL` policy (Stage 6.2, normative).** `GPU.PCI_SCAN` is the
driver-independent sysfs PCI presence check (FR-04, DEC-22). It is supported only when
`linux_environment` is `NATIVE_LINUX` (DEC-41); the platform profile never changes this.
Every other environment has a host-wide (`device_ref = null`) `NOT_SUPPORTED` record whose
`reason` is exactly the PCI scan's error message:

| Environment | `pci_observations` error code | `GPU.PCI_SCAN` reason | Also unsupported |
| ----------- | ----------------------------- | --------------------- | ---------------- |
| `NATIVE_LINUX` | — (scan runs) | — (supported) | — |
| `WSL1` | `PLATFORM_NOT_SUPPORTED` | "The PCI presence check runs only on native Linux, and this host is WSL1 (FR-04; DEC-41; data_model_specification.md §4.1)." | — |
| `WSL2` | `PLATFORM_NOT_SUPPORTED` | "GPUs are not enumerated on PCI under WSL2 (FR-04; product_specification.md §4, OBS-01)." | — |
| `UNKNOWN` | `LINUX_ENVIRONMENT_UNKNOWN` | "The Linux environment couldn't be classified as native Linux, so the PCI presence check doesn't run (DEC-41; data_model_specification.md §4.1)." | — |
| Windows (non-Linux) | `PLATFORM_NOT_SUPPORTED` | "PCI presence check is Linux-only in V1 (FR-04; DEC-22; product_specification.md §4)." | — |
| macOS (non-Linux) | `PLATFORM_NOT_SUPPORTED` | same as Windows | `NVIDIA.ALL`: "NVIDIA CUDA GPUs are not supported on macOS (product_specification.md §4)." |
| Other non-Linux (`OTHER`) | `PLATFORM_NOT_SUPPORTED` | same as Windows | — |

The policy is a pure function (`collectors/capabilities.py`: `unsupported_capabilities_for`)
shared by production code and the fixtures. Nothing emits these records at runtime yet: the
PCI collector (Stage 6.3) and report assembly (IMP-P01) will.

## 4. System and GPU models

### 4.1 `SystemInfo`

| Field | Type | Unit | Source / method | Privacy | Notes |
| ----- | ---- | ---- | --------------- | ------- | ----- |
| `platform_profile` | `PlatformProfile` | – | derived (`INSPECTOR`) | – | `WSL2` only when `linux_environment` is `WSL2`; every other Linux classification (`NATIVE_LINUX`, `WSL1`, `UNKNOWN`) is `LINUX`. Capabilities that need native Linux (the sysfs PCI scan, DEC-22) must check `linux_environment == NATIVE_LINUX`, not the profile. |
| `os_name` | `Observed[str]` | – | `platform.system()` | – | `Linux`, `Windows`, `Darwin`. |
| `os_release` | `Observed[str]` | – | `platform.release()` | – | |
| `os_version` | `Observed[str]` | – | `platform.version()` | – | |
| `os_distribution` | `Observed[str]` | – | Linux `/etc/os-release:PRETTY_NAME` (fallback `/usr/lib/os-release`); Windows `platform.win32_ver()` + `platform.win32_edition()`; macOS `platform.mac_ver()` | – | Other OSes: `NOT_SUPPORTED`. |
| `architecture` | `Observed[str]` | – | `platform.machine()` | – | e.g. `x86_64`, `AMD64`, `arm64`, `aarch64`. |
| `processor_model` | `Observed[str]` | – | Linux `/proc/cpuinfo:model name`; Windows registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0:ProcessorNameString` (Stage 5; replaces `platform.processor()`, which reads a non-allowlisted variable); macOS `sysctlbyname("machdep.cpu.brand_string")` (`char[]`, via `ctypes`; Stage 5.1) | – | Empty or absent entry → `NOT_DETECTED` (EV-21, OBS-02). |
| `logical_cpu_count` | `Observed[int]` | – | `os.cpu_count()` | – | `None` → `NOT_DETECTED`. System-wide logical CPUs; physical cores are not collected. |
| `memory_total_bytes` | `Observed[int]` | bytes | Linux `/proc/meminfo:MemTotal` (KiB × 1024); Windows `GlobalMemoryStatusEx.ullTotalPhys`; macOS `sysctlbyname("hw.memsize")` (`int64_t`, via `ctypes`; Stage 5.1) | – | Physical RAM usable by the OS; never swap/page file/virtual memory. In WSL2: the VM's memory, not the host's (stated in `note`). |
| `memory_available_bytes` | `Observed[int]` | bytes | Linux `/proc/meminfo:MemAvailable`; Windows `ullAvailPhys` | – | Memory available to new work without swapping; never `MemFree` and never process memory. Missing `MemAvailable` → `NOT_DETECTED`. macOS: `NOT_SUPPORTED` in V1. |
| `python_version` | `Observed[str]` | – | `platform.python_version()` | – | |
| `python_implementation` | `Observed[str]` | – | `platform.python_implementation()` | – | |
| `python_executable` | `Observed[str]` | – | `sys.executable` | **P** | |
| `linux_environment` | `Observed[LinuxEnvironment]` | – | derived (`INSPECTOR`) from `/proc/sys/kernel/osrelease` and `/proc/version`; method `classify(/proc/sys/kernel/osrelease, /proc/version)` | – | Linux: always `COLLECTED` with `NATIVE_LINUX`, `WSL1`, `WSL2` or `UNKNOWN`, and `note` = the rationale. Non-Linux: `NOT_APPLICABLE` with a note. Replaced `is_wsl: Observed[bool]` in Stage 5.1. Rule below. |

**`LinuxEnvironment` classification (Stage 5.1, normative).** Inputs are the kernel release
in `/proc/sys/kernel/osrelease` and in `/proc/version` (`Linux version <release> …`). Both
are generated live by the running kernel, so they can't be a leftover of a removed WSL
distribution or kernel. Files that *can* be residual (`/etc/wsl.conf`, `/usr/lib/wsl`,
`/mnt/wslg`) and environment variables (`WSL_DISTRO_NAME`, which isn't on the allowlist)
are deliberately not consulted. Matching ignores case. The first rule that applies wins:

| # | Evidence | Value |
| - | -------- | ----- |
| 1 | Either source unreadable (missing, permission denied, I/O error), empty or malformed, or the two name different releases | `UNKNOWN` |
| 2 | Release contains `-microsoft-standard-WSL2` (not followed by a letter or digit) | `WSL2` |
| 3 | Release has the WSL1 format `<x.y.z>-<Windows build>-Microsoft` (e.g. `4.4.0-19041-Microsoft`) and `/proc/version` doesn't mention `WSL2` | `WSL1` |
| 3a | WSL1 format but `/proc/version` mentions `WSL2` | `UNKNOWN` (conflicting) |
| 4 | Any other `microsoft` or `wsl` marker in either source (e.g. the older `4.19.x-microsoft-standard` naming, custom builds) | `UNKNOWN` (version can't be determined) |
| 5 | No marker at all | `NATIVE_LINUX` |

`NATIVE_LINUX` means "a Linux kernel that isn't WSL". It doesn't claim bare metal:
generic virtualisation (KVM, Azure or cloud kernels) is still `NATIVE_LINUX`. The word
"Microsoft" alone never yields `WSL1` or `WSL2`. The `UNKNOWN` value is a classification
result. It isn't the `CollectionStatus.UNKNOWN` defect signal, and the reason (including an
unreadable source's exception type) is always in `note`.

### 4.2 `GPUInventory`, `GPUDeviceInfo`, `PciObservation`

`GPUInventory`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `device_count` | `Observed[int]` | NVML `nvmlDeviceGetCount` (or fallback row count). `0` only when the source actually reported zero devices. |
| `inventory_source` | `Source \| null` | `NVML`, `NVIDIA_SMI` (fallback) or `null` (no inventory). |
| `devices` | `list[GPUDeviceInfo]` | Sorted by `nvml_index`. Empty when no inventory. |
| `pci_observations` | `Observed[list[PciObservation]]` | Scanned only when `linux_environment` is `NATIVE_LINUX`; otherwise `NOT_SUPPORTED` (`LINUX_ENVIRONMENT_UNKNOWN` for `UNKNOWN`, else `PLATFORM_NOT_SUPPORTED`; §3.4). `COLLECTED` with `[]` means the scan ran and found no display/3D controllers, which is different from `NOT_SUPPORTED`. A missing sysfs root is `NOT_DETECTED`. Devices that were listed but couldn't be observed are left out and reported as host-wide `collection_errors` (collector `pci`, run `PARTIAL`; architecture §4.4, DEC-43), so `COLLECTED` doesn't guarantee a complete enumeration. |

`GPUDeviceInfo`

| Field | Type | Unit | Source / method | Privacy |
| ----- | ---- | ---- | --------------- | ------- |
| `device_ref` | `str` | – | derived | – |
| `nvml_index` | `int` | – | enumeration index (NVML or `nvidia-smi` `index`) | – |
| `vendor` | `str` | – | constant `"NVIDIA"` (devices come only from NVIDIA sources) | – |
| `name` | `Observed[str]` | – | `nvmlDeviceGetName` / `name` | – |
| `uuid` | `Observed[str]` | – | `nvmlDeviceGetUUID` / `uuid` | **D** |
| `pci_bus_id` | `Observed[str]` | – | `nvmlDeviceGetPciInfo_v3.busId` / `pci.bus_id` | **D** |
| `pci_device_id` | `Observed[str]` | – | `nvmlDeviceGetPciInfo_v3.pciDeviceId` (hex string, as reported) | – |
| `pci_subsystem_id` | `Observed[str]` | – | `nvmlDeviceGetPciInfo_v3.pciSubSystemId` (hex) | – |
| `memory_total_bytes` | `Observed[int]` | bytes | `nvmlDeviceGetMemoryInfo[_v2].total` / `memory.total` (MiB→bytes) | – |
| `compute_capability` | `Observed[str]` | – | `nvmlDeviceGetCudaComputeCapability` → `"major.minor"` | – |
| `mig_mode` | `Observed[str]` | – | `nvmlDeviceGetMigMode` current → `ENABLED`/`DISABLED` | – |

`PciObservation`

| Field | Type | Description | Privacy |
| ----- | ---- | ----------- | ------- |
| `address` | `str` | sysfs device name, e.g. `0000:01:00.0` | **D** |
| `vendor_id` | `str` | e.g. `0x10de` | – |
| `device_id` | `str` | e.g. `0x2684` | – |
| `class_code` | `str` | e.g. `0x030000` (display) / `0x030200` (3D controller) | – |
| `is_nvidia` | `bool` | `vendor_id == 0x10de` (EV-26) | – |
| `kernel_driver` | `Observed[str]` | basename of the `driver` symlink (`nvidia`, `nouveau`, `vfio-pci`, …); `NOT_DETECTED` with `NO_DRIVER_BOUND` if no driver bound | – |

### 4.3 `DriverInfo`

| Field | Type | Source / method | Privacy | Notes |
| ----- | ---- | --------------- | ------- | ----- |
| `nvml_binding` | `Observed[str]` | `importlib.metadata.version(<dist>)` after a successful `import pynvml` | – | Value = binding version. `NOT_INSTALLED` if import fails with `ImportError`. That can only happen in a broken installation, because `nvidia-ml-py` is a required dependency (DEC-05). |
| `nvml_binding_distribution` | `Observed[str]` | `importlib.metadata.packages_distributions()["pynvml"]` | – | Expected `nvidia-ml-py`. A different distribution (e.g. `pynvml`) is recorded for D05. |
| `nvml_version` | `Observed[str]` | `nvmlSystemGetNVMLVersion` | – | **Its status is the NVML accessibility status** ([data_flow.md](data_flow.md) §4): `COLLECTED` ⇒ NVML initialised. |
| `driver_version` | `Observed[str]` | `nvmlSystemGetDriverVersion` / `nvidia-smi` `driver_version` | – | Precedence (DEC-46): NVML, then the `nvidia-smi` query, then the `nvidia-smi` header; with none, NVML's own reason. |
| `driver_supported_cuda_version` | `Observed[str]` | `nvmlSystemGetCudaDriverVersion` → `"{v//1000}.{(v%1000)//10}"` (EV-04) / `nvidia-smi` header | – | `note` always: *"Newest CUDA version supported by the installed driver. Not the installed CUDA Toolkit version."* (EV-03) |
| `cuda_driver_library` | `Observed[str]` | `nvmlSystemGetCudaDriverVersion_v2` (calls `cuDriverGetVersion`, EV-04) | – | `NOT_DETECTED` on `NVML_ERROR_LIBRARY_NOT_FOUND` (no `libcuda`). `NOT_SUPPORTED` if the function is absent. `[ASM]` binding exposes `_v2`. |
| `nvidia_smi_path` | `Observed[str]` | `shutil.which` / WSL2 path | **P** | |
| `nvidia_smi_driver_version` | `Observed[str]` | header `KMD Version` or `Driver Version` (EV-03) | – | Evidence only. When it differs from NVML's `driver_version`, its `note` says so (DEC-46). |
| `nvidia_smi_cuda_version` | `Observed[str]` | header `CUDA UMD Version` or `CUDA Version` (EV-03) | – | Same meaning as `driver_supported_cuda_version`, which takes it (with the fixed note) when NVML can't report one. |

No `DriverInfo` field describes the CUDA runtime library (`libcudart`) or whether CUDA is
usable: `cuda_driver_library` is the driver's `libcuda`, and usability comes from the
PyTorch probe (`pytorch.cuda_available`, IMP-A07/K06).

## 5. CUDA environment and framework models

### 5.1 `CUDAEnvironmentInfo`

| Field | Type | Source | Notes |
| ----- | ---- | ------ | ----- |
| `toolkits` | `Observed[list[CudaToolkitCandidate]]` | `CUDA_TOOLKIT`, method `toolkit discovery` | `NOT_DETECTED`/`NO_TOOLKIT_FOUND` (value `null`) when no candidates; a failed `/usr/local` listing with no candidate keeps its own status (never `NOT_DETECTED`); macOS and `OTHER` `NOT_SUPPORTED`/`PLATFORM_NOT_SUPPORTED`. Candidates sorted by `root_path`. |
| `active_nvcc_path` | `Observed[str]` (**P**) | `shutil.which("nvcc")`, source `CUDA_TOOLKIT` | First `nvcc` on `PATH`, as found (unresolved). Missing → `NOT_DETECTED`/`FileNotFoundError`, which doesn't make the toolkits absent. |
| `active_nvcc_version` | `Observed[str]` | `nvcc --version` | Parsed `X.Y.Z` from `VX.Y.Z` (or `X.Y` from `release X.Y`); unrecognised output `COLLECTION_FAILED`/`PARSE_ERROR`, non-zero exit `EXIT_<n>`, timeout `TIMEOUT`. Missing `nvcc` → the path's failure. |
| `environment` | `dict[str, Observed[str]]` | `ENVIRONMENT` | Allowlist only (§5.5). Keys sorted. Unset → `NOT_DETECTED`. Set but empty → `COLLECTED` with `""` (meaningful for `CUDA_VISIBLE_DEVICES`, EV-10). |

### 5.2 `CudaToolkitCandidate`

| Field | Type | Privacy | Notes |
| ----- | ---- | ------- | ----- |
| `root_path` | `str` | **P** | Resolved path (symlinks followed; aliases merge into one candidate). |
| `discovered_via` | `list[ToolkitDiscovery]` | – | All signals that pointed here (de-duplicated, enum order). |
| `version` | `Observed[str]` | – | `method` = `"version.json:cuda.version"` or `"nvcc --version"`. `[ASM]` EV-20. Malformed `version.json` or a value not matching `X.Y[.Z…]` → `COLLECTION_FAILED`/`PARSE_ERROR` (no fallback); unreadable → `PERMISSION_DENIED`/`COLLECTION_FAILED`; neither file nor compiler → `NOT_DETECTED`/`NO_VERSION_EVIDENCE`. Never a directory name. |
| `nvcc_path` | `Observed[str]` | **P** | `<root>/bin/nvcc[.exe]` if present (method `toolkit layout`); else `NOT_DETECTED`/`NVCC_NOT_IN_TOOLKIT`. |

Signals, platforms and the discovery boundaries: architecture §4.6 (DEC-47). Toolkit
discovery is not CUDA execution validation: nothing in this section says CUDA works.

### 5.3 `FrameworkInfo` and `PyTorchInfo`

`FrameworkInfo = { pytorch: PyTorchInfo }`. Other frameworks are future additions under new keys (additive schema change).

| Field | Type | Source / method | Notes |
| ----- | ---- | --------------- | ----- |
| `status` | `CollectionStatus` | – | Section status: `COLLECTED`, `NOT_INSTALLED` (no `torch` distribution; no probe ran), `SKIPPED` (`include_pytorch = false`; nothing read), `COLLECTION_FAILED` (the probe ran and failed: `IMPORT_ERROR`, `TIMEOUT`, `PROBE_EXIT_<n>`, `PROBE_PROTOCOL_ERROR`, `PROBE_EXCEPTION`; `distribution_version` kept). DEC-48. |
| `distribution_version` | `Observed[str]` | `importlib.metadata.version("torch")` | Main process, without importing torch. Scope: the inspector's own Python environment only (DEC-26, KI-11). |
| `version` | `Observed[str]` | `torch.__version__` | Probe. |
| `cuda_build` | `Observed[bool]` | `torch.version.cuda is not None` | `false` = CPU-only (or non-CUDA) build. |
| `build_cuda_version` | `Observed[str]` | `torch.version.cuda` | `NOT_APPLICABLE` when `cuda_build` is `false`. `note`: *"CUDA version PyTorch was built with; not the toolkit and not the driver."* |
| `rocm_build` | `Observed[bool]` | `torch.version.hip is not None` | ROCm builds are out of V1 analysis scope (D01 reports `NOT_APPLICABLE` reasoning). |
| `cudnn_version` | `Observed[str]` | `torch.backends.cudnn.version()` | `NOT_APPLICABLE` for non-CUDA builds; `NOT_DETECTED`/`CUDNN_NOT_AVAILABLE` if `None`. |
| `cuda_available` | `Observed[bool]` | `torch.cuda.is_available()` | PyTorch's own answer. Not proof that CUDA code runs: the probe runs none (DEC-48). |
| `cuda_init_warnings` | `list[str]` | warnings captured during `is_available()` | Redactable free text; may be empty. |
| `device_count` | `Observed[int]` | `torch.cuda.device_count()` | Respects `CUDA_VISIBLE_DEVICES` (EV-10). `COLLECTION_FAILED`/`DEVICE_ENUMERATION_FAILED` when device properties couldn't be read (probe protocol 2 `device_error`); the section stays `COLLECTED` and `devices` is empty. |
| `devices` | `list[TorchDeviceInfo]` | probe | Sorted by `torch_index`. |
| `allocator` | `Observed[list[AllocatorStats]]` | probe (in-process only) | CLI: `NOT_APPLICABLE` with `note` *"Allocator statistics describe only the calling process; the inspector's probe process has made no allocations."* (DEC-08) |
| `probe` | `ProbeInfo` | – | `{ mode: ProbeMode, protocol_version: int, duration_ms: int \| null }`. `protocol_version` is 2 (DEC-48); `duration_ms` is the probe's run time, `null` when no probe ran or it failed. |

### 5.4 `TorchDeviceInfo` and `AllocatorStats`

`TorchDeviceInfo`

| Field | Type | Unit | Method | Privacy |
| ----- | ---- | ---- | ------ | ------- |
| `torch_index` | `int` | – | CUDA ordinal in the probe process | – |
| `name` | `Observed[str]` | – | `get_device_name` | – |
| `compute_capability` | `Observed[str]` | – | `get_device_capability` | – |
| `total_memory_bytes` | `Observed[int]` | bytes | `get_device_properties().total_memory` | – |
| `multiprocessor_count` | `Observed[int]` | – | `.multi_processor_count` | – |
| `uuid` | `Observed[str]` | – | `.uuid` (version-dependent, EV-14) | **D** |
| `pci_bus_id` | `Observed[str]` | – | `.pci_domain_id/.pci_bus_id/.pci_device_id` formatted (version-dependent) | **D** |
| `correlated_device_ref` | `Observed[str]` | – | Set by the correlation run (IMP-K07, DEC-09, DEC-49), source `INSPECTOR`. A unique match: the NVML `device_ref`, `method` = `"uuid"` or `"pci_bus_id"` (`note` if the other identifier contradicts a UUID match). `NOT_APPLICABLE` *"No NVML inventory to correlate with"* when the inventory isn't NVML's. Otherwise `NOT_DETECTED` with `NO_MATCH`, `NO_IDENTIFIER`, `AMBIGUOUS_MATCH` (several candidates, or one NVML device claimed twice: none chosen) or `UNIDENTIFIED_CANDIDATES` (an NVML device without a usable identifier can't be ruled out). If the correlation run fails, `COLLECTION_FAILED` (the guard's error). | – |

A device attribute this PyTorch version doesn't expose is `NOT_SUPPORTED`/`ATTRIBUTE_MISSING`.

PCI bus ID comparison (DEC-49): parse `domain:bus:device.function` (domain 4 or 8 hex digits,
bus and device 2, function 1; device ≤ `0x1f`) and compare the four numbers, so NVML's
`00000000:01:00.0` equals PyTorch's `0000:01:00.0`; anything else is unusable. Index, name
and memory are never identity evidence.

UUID normalisation for correlation `[ASM]`: lower-case, strip a leading `GPU-` prefix
(NVML format), compare as 8-4-4-4-12 hex.

In-process mode without CUDA initialised in the caller, `allocator` is `NOT_APPLICABLE` with
*"CUDA isn't initialised in the calling process, so it has no allocator statistics."*

`AllocatorStats` (in-process API only): `torch_index: int`; `allocated_bytes`,
`reserved_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`: `Observed[int]`
(`source = PYTORCH`, `note` = *"Calling process only"*).

### 5.5 Environment variable allowlist

| Variable | Why | Privacy |
| -------- | --- | ------- |
| `CUDA_VISIBLE_DEVICES` | Device visibility (EV-10, D01, D08) | – |
| `CUDA_DEVICE_ORDER` | Index ordering (EV-10, D08) | – |
| `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V*` (Windows) | Toolkit discovery | **P** |
| `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES` | Container GPU exposure | – |
| `PYTORCH_NVML_BASED_CUDA_CHECK` | Changes `is_available()` behaviour (EV-13) | – |
| `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF`, `PYTORCH_NO_CUDA_MEMORY_CACHING` | Allocator behaviour (EV-13) | – |
| `LD_LIBRARY_PATH` (Linux only) | Library resolution for `libcuda`/`libnvidia-ml` | **P** |

## 6. Runtime, readiness and diagnostics models

### 6.1 `RuntimeMetrics` (one per inventoried device)

| Field | Type | Unit | Method (NVML / `nvidia-smi`) | Notes |
| ----- | ---- | ---- | ---------------------------- | ----- |
| `device_ref` | `str` | – | – | |
| `sampled_at` | `str` | RFC 3339 | clock | Volatile (§7.4). |
| `gpu_utilization_percent` | `Observed[int]` | percent | `nvmlDeviceGetUtilizationRates.gpu` / `utilization.gpu` | % of time ≥ 1 kernel executing over the last sample period (EV-07). |
| `memory_bus_utilization_percent` | `Observed[int]` | percent | `nvmlDeviceGetUtilizationRates.memory` / `utilization.memory` | % of time device memory was read/written. **Not** occupancy (EV-07, C04). |
| `memory_total_bytes` | `Observed[int]` | bytes | `nvmlDeviceGetMemoryInfo[_v2].total` / `memory.total` | |
| `memory_used_bytes` | `Observed[int]` | bytes | `.used` / `memory.used` | With v2, includes reserved (EV-06). Don't assume `total − free`. |
| `memory_free_bytes` | `Observed[int]` | bytes | `.free` / `memory.free` | Device-wide; the readiness input. |
| `memory_reserved_bytes` | `Observed[int]` | bytes | `nvmlDeviceGetMemoryInfo_v2.reserved` | `NOT_SUPPORTED` when only v1 is available. |
| `temperature_gpu_celsius` | `Observed[int]` | celsius | `nvmlDeviceGetTemperatureV` → fallback `nvmlDeviceGetTemperature(GPU)` / `temperature.gpu` | Single sample. |
| `temperature_slowdown_threshold_celsius` | `Observed[int]` | celsius | `nvmlDeviceGetTemperatureThreshold(SLOWDOWN)` | Device-reported. |
| `power_draw_watts` | `Observed[float]` | watts | `nvmlDeviceGetPowerUsage` ÷ 1000 / `power.draw` | |
| `power_limit_watts` | `Observed[float]` | watts | `nvmlDeviceGetEnforcedPowerLimit` ÷ 1000 / `power.limit` | |
| `processes` | `Observed[list[GpuProcess]]` | – | `nvmlDeviceGetComputeRunningProcesses_v3` + `…GraphicsRunningProcesses_v3` (merged by PID) | `SKIPPED` with `--no-processes`; WSL2 expected `NOT_SUPPORTED` (EV-09). |

A device that can't be reached again, or reports `GPU is lost` during sampling, gets
`RuntimeMetrics.unavailable(...)` (every metric failed with that error, no partial values)
and one `CollectionError` at `/runtime/<i>` (architecture §4.4, DEC-46). The implementation
reads compute processes only (`…ComputeRunningProcesses_v3`, legacy fallback); graphics
processes aren't merged yet.

`GpuProcess`: `pid: int`; `kind: ProcessKind`; `used_memory_bytes: Observed[int]`
(WDDM → `NOT_SUPPORTED`, EV-08); `name: Observed[str]` (**N**, basename from
`nvmlSystemGetProcessName` or `/proc/<pid>/comm`; `PERMISSION_DENIED`/`NOT_DETECTED` when
unavailable, e.g. PID-namespace mismatch in containers).

### 6.2 `ReadinessResult` (present only in `READINESS` mode; otherwise the report field is `null`)

| Field | Type | Description |
| ----- | ---- | ----------- |
| `required_vram_gib` | `float` | As supplied. |
| `required_bytes` | `int` | `round(required_vram_gib × 2³⁰)`. |
| `criterion` | `str` | Constant `"DEVICE_FREE_MEMORY_GTE_REQUIRED"`. |
| `overall_status` | `ReadinessStatus` | Aggregation rules in [architecture.md](architecture.md) §4.9. |
| `overall_reason` | `ReadinessReason` | One of the overall reason codes. |
| `devices` | `list[DeviceReadiness]` | One per inventoried device, sorted by `device_ref`. |
| `disclaimer` | `str` | Fixed text ([product_specification.md](product_specification.md) §3, Mode B). |
| `limitations` | `list[str]` | E.g. "Snapshot taken at `<sampled_at>`", "Memory source: `nvidia-smi` fallback, 1 MiB precision". |

`DeviceReadiness`: `device_ref: str`; `status: ReadinessStatus`; `reason: ReadinessReason`;
`free_bytes: Observed[int]` (copied from runtime, with its source); `total_bytes:
Observed[int]`; `margin_bytes: int | null` (`free − required`, may be negative; `null`
unless free is `COLLECTED`); `explanation: str`.

### 6.3 `DiagnosticFinding`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `check_id` | `str` | `D01`–`D08` ([diagnostics_specification.md](diagnostics_specification.md)). |
| `device_ref` | `str \| null` | `null` for system-wide findings. |
| `status` | `DiagnosticStatus` | |
| `severity` | `Severity` | Impact if the status isn't `PASS` (mapping in the diagnostics spec §2). |
| `title` | `str` | Short, specific. |
| `explanation` | `str` | What was found and what it means. |
| `evidence` | `list[Evidence]` | ≥ 1 item for every status other than `NOT_APPLICABLE`. `NOT_APPLICABLE` carries the precondition evidence. |
| `detection_sources` | `list[Source]` | Sorted, unique. |
| `recommendations` | `list[Recommendation]` | Required for `WARNING`/`FAIL`; optional otherwise. |
| `confidence` | `Confidence` | Confidence in the finding's *conclusion*. |
| `limitations` | `list[str]` | ≥ 1 item for `WARNING`, `FAIL`, `UNKNOWN`. |

`Evidence`: `kind: EvidenceKind`; `statement: str`; `source: Source | null`;
`report_path: str | null` (JSON Pointer to the supporting field); `value: JSON | null`
(copy of the observed value for convenience; `null` for inferences).

`Recommendation`: `action: str` (imperative, never executed by the tool);
`rationale: str`; `reference_url: str | null` (official documentation only).

## 7. `InspectionReport` and JSON contract

### 7.1 `InspectionReport` (top-level JSON object, keys in this order)

| Key | Type | Description |
| --- | ---- | ----------- |
| `schema_name` | `str` | Constant `"inferforge.gpu_system_inspector.report"`. |
| `schema_version` | `str` | `"1.0"` (§7.3). |
| `application` | `{name: str, display_name: str, distribution: str, version: str}` | `name = "gpu-system-inspector"`, `display_name = "GPU System Inspector"`, `distribution = "01-gpu-system-inspector"`. |
| `generated_at` | `str` | RFC 3339 UTC. Volatile. |
| `duration_ms` | `int` | Volatile. |
| `data_origin` | `DataOrigin` | `LIVE` for real runs (DEC-17). |
| `mode` | `InspectionMode` | |
| `options` | object | Echo of effective `InspectionOptions`. |
| `system` | `SystemInfo` | |
| `gpu` | `GPUInventory` | |
| `driver` | `DriverInfo` | |
| `cuda_environment` | `CUDAEnvironmentInfo` | |
| `frameworks` | `FrameworkInfo` | |
| `runtime` | `list[RuntimeMetrics]` | Sorted by `device_ref`. |
| `diagnostics` | `list[DiagnosticFinding]` | Sorted by `(check_id, device_ref ?? "")`. |
| `readiness` | `ReadinessResult \| null` | |
| `collection_errors` | `list[CollectionError]` | Sorted by `(collector, device_ref ?? "", error_code)`. |
| `unsupported_capabilities` | `list[UnsupportedCapability]` | Sorted by `(capability, device_ref ?? "")`. |
| `limitations` | `list[str]` | Global limitations (always includes the snapshot statement). |
| `collector_runs` | `list[{collector: str, status: CollectorRunStatus, duration_ms: int}]` | In execution order. Durations volatile. |
| `redaction` | `{applied: bool, categories: list[str]}` | Categories: `PATH`, `PROCESS_NAME`, `DEVICE_ID`, `USER`. |

### 7.2 Serialisation rules

1. UTF-8, `json.dumps(..., ensure_ascii=False, indent=2, allow_nan=False)` plus a trailing newline.
2. Keys appear in model field order (deterministic; not alphabetical).
3. **Every key is always present.** Absent data is `null` (inside `Observed`) or `[]` (lists), never omitted.
4. Enums are upper-case strings; booleans are JSON booleans; bytes are integers; floats are
   finite (NaN/Infinity are forbidden and rejected by `allow_nan=False`).
5. Lists are sorted as specified. Dictionaries (only `environment`) are sorted by key.
6. No Python-specific types (`repr`, tuples, bytes) appear in output.
7. The JSON Schema (Draft 2020-12) ships at
   `src/gpu_system_inspector/reporting/schema/inspection_report.schema.json`, and every test
   fixture report must validate against it (AC09). It uses `additionalProperties: false` for
   schema `1.x` within the application's own tests. Consumers must tolerate unknown keys
   for forward compatibility.

### 7.3 Schema versioning

- `schema_version` is `MAJOR.MINOR`, independent of the application version.
- **MINOR** bump: additive changes (new optional section, new key, new enum member,
  new check ID). Consumers must ignore unknown keys and treat unknown enum members as
  `UNKNOWN`.
- **MAJOR** bump: removing or renaming a key, changing a type, unit or meaning, or changing
  an invariant.
- Until the specification is approved and AC09 is TESTED, `1.0` may change without a bump.
  After that freeze, every change follows the rules above and is recorded in
  [app_tracker.md](app_tracker.md) §10.
- Draft changes so far (no report has been published, so no consumer is affected): Stage 3
  severity rules; **Stage 5.1** `SystemInfo.is_wsl` (boolean) replaced by
  `SystemInfo.linux_environment` (`ObservedLinuxEnvironment`, new `LinuxEnvironment` enum),
  plus profile/classification consistency rules. After the freeze, this would be a MAJOR
  change.

### 7.4 Determinism and volatile fields

With the same inputs (same fixture adapters and clock), two serialisations are
byte-identical. Against live systems, comparisons must ignore the **volatile fields**:
`generated_at`, `duration_ms`, `collector_runs[*].duration_ms`, `runtime[*].sampled_at`,
`frameworks.pytorch.probe.duration_ms`, and naturally changing runtime values
(utilization, used/free memory, temperature, power, processes).

### 7.5 Illustrative example (abridged)

> **Illustrative only. This is not a real measurement.** It shows the shape of a report
> for a WSL2 machine with no NVIDIA driver exposed and no PyTorch, similar to OBS-01.
> `data_origin` is `FIXTURE` because it was authored by hand.

```json
{
  "schema_name": "inferforge.gpu_system_inspector.report",
  "schema_version": "1.0",
  "application": {"name": "gpu-system-inspector", "display_name": "GPU System Inspector", "distribution": "01-gpu-system-inspector", "version": "0.1.0"},
  "generated_at": "2026-11-04T09:00:00Z",
  "duration_ms": 412,
  "data_origin": "FIXTURE",
  "mode": "GENERAL",
  "options": {"required_vram_gib": null, "include_pytorch": true, "include_processes": true,
              "redact": false, "subprocess_timeout_s": 10.0, "pytorch_timeout_s": 60.0,
              "pytorch_probe_mode": "SUBPROCESS"},
  "system": {
    "platform_profile": "WSL2",
    "os_name": {"value": "Linux", "status": "COLLECTED", "source": "PLATFORM", "method": "platform.system()", "unit": null, "error": null, "note": null},
    "processor_model": {"value": null, "status": "NOT_DETECTED", "source": "OS", "method": "/proc/cpuinfo:model name", "unit": null,
                        "error": {"code": "FIELD_NOT_FOUND", "message": "No 'model name' entry"}, "note": null}
  },
  "gpu": {
    "device_count": {"value": null, "status": "NOT_DETECTED", "source": "NVML", "method": "nvmlInit", "unit": null,
                     "error": {"code": "NVML_ERROR_LIBRARY_NOT_FOUND", "message": "NVML Shared Library Not Found"}, "note": null},
    "inventory_source": null,
    "devices": [],
    "pci_observations": {"value": null, "status": "NOT_SUPPORTED", "source": "OS", "method": "/sys/bus/pci/devices", "unit": null,
                         "error": {"code": "PLATFORM_NOT_SUPPORTED", "message": "GPUs are not enumerated on PCI under WSL2 (FR-04; product_specification.md §4, OBS-01)."}, "note": null}
  },
  "frameworks": {"pytorch": {"status": "NOT_INSTALLED"}},
  "runtime": [],
  "diagnostics": [
    {"check_id": "D04", "device_ref": null, "status": "INFO", "severity": "NONE",
     "title": "No NVIDIA GPU is accessible on this system",
     "explanation": "NVML could not be loaded and nvidia-smi was not found. NVIDIA-specific inspection was skipped; general system information was collected.",
     "evidence": [{"kind": "OBSERVATION", "statement": "nvmlInit failed: NVML_ERROR_LIBRARY_NOT_FOUND", "source": "NVML", "report_path": "/driver/nvml_version", "value": null},
                  {"kind": "OBSERVATION", "statement": "nvidia-smi not found on PATH or /usr/lib/wsl/lib", "source": "NVIDIA_SMI", "report_path": "/driver/nvidia_smi_path", "value": null}],
     "detection_sources": ["NVIDIA_SMI", "NVML"],
     "recommendations": [{"action": "If this machine should have an NVIDIA GPU, install or update the NVIDIA driver on the Windows host (not inside WSL2).",
                          "rationale": "On WSL2 the Windows driver is mapped into the Linux environment.",
                          "reference_url": "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"}],
     "confidence": "MEDIUM",
     "limitations": ["Driver-independent PCI detection is not supported under WSL2, so the absence of a GPU cannot be confirmed from hardware enumeration."]}
  ],
  "readiness": null,
  "collection_errors": [],
  "unsupported_capabilities": [{"capability": "GPU.PCI_SCAN", "status": "NOT_SUPPORTED", "device_ref": null, "reason": "GPUs are not enumerated on PCI under WSL2 (FR-04; product_specification.md §4, OBS-01)."}],
  "limitations": ["This report is a point-in-time snapshot."],
  "collector_runs": [{"collector": "system", "status": "COMPLETED", "duration_ms": 12}],
  "redaction": {"applied": false, "categories": []}
}
```

(Abridged: a real report contains every key listed in §4–§7.1.)

## 8. Privacy and redaction

| Category | Fields | Placeholder |
| -------- | ------ | ----------- |
| `PATH` | `python_executable`, `nvidia_smi_path`, toolkit `root_path`/`nvcc_path`, `active_nvcc_path`, `CUDA_HOME`/`CUDA_PATH*`/`LD_LIBRARY_PATH` values | `<redacted:path>` |
| `PROCESS_NAME` | `GpuProcess.name` | `<redacted:process>` |
| `DEVICE_ID` | GPU `uuid`, `pci_bus_id`, PCI `address`, torch `uuid`/`pci_bus_id` | `<redacted:uuid:N>`, `<redacted:pci:N>` (N = stable per-report ordinal, so correlation stays visible) |
| `USER` | local username and home directory wherever they occur in free text | `<redacted:user>`, `~` |

Rules:
1. Redaction runs on the assembled report and covers **all strings**, including
   `explanation`, `evidence.statement`, `error.message`, `cuda_init_warnings` and
   `collection_errors.message`. Every sensitive literal collected is replaced wherever it
   appears.
2. Redacted reports keep structure, statuses and non-sensitive values unchanged.
3. Never collected at all: hostname, username (except as a redaction input), full command
   lines, full environment, MAC addresses, serial numbers.

## 9. Missing-value representation summary

| Situation | Representation |
| --------- | -------------- |
| Value obtained | `Observed(value=…, status=COLLECTED)` |
| Value not obtainable | `Observed(value=null, status=<reason>, error={code, message})` |
| User opted out | `Observed(value=null, status=SKIPPED, error=null)` |
| Meaningless in context | `Observed(value=null, status=NOT_APPLICABLE, note=<why>)`; `error` optional (§3.2) |
| Whole device missing | Device absent from `gpu.devices`; `device_count` carries the status |
| Whole section failed unexpectedly | Section's `unavailable()` model (all `Observed` = `COLLECTION_FAILED`) + `CollectionError` (the guard's `UNHANDLED_EXCEPTION`; the GPU section keeps the PCI collector's `pci_observations`) |
| Collector not implemented in this version | Section placeholder: every `Observed` `NOT_SUPPORTED` with code `NOT_IMPLEMENTED` (PyTorch: section `SKIPPED`, its model's only non-failure option), no `collector_runs` entry, and a `limitations` statement (DEC-44) |
| Mode A readiness | `"readiness": null` |
| Zero | Only ever a real observed zero (e.g. 0 % utilization, 0 processes reported by NVML) |

## 10. Implementation clarifications (Stage 2, 2026-11-04)

The models in `src/gpu_system_inspector/models/` implement this specification. Where the
specification was silent or internally inconsistent, the implementation chose the reading
below. None of them changes the JSON contract in §7.

| # | Topic | Clarification |
| - | ----- | ------------- |
| 1 | `NOT_APPLICABLE` and `error` | **Resolved in the specification (Stage 3):** §3.2 now states the rule and the reason. `NOT_APPLICABLE` requires `note`, and `error` is optional. |
| 2 | `source` | Required (with a non-empty `method`) when `COLLECTED`. Otherwise it may be `null`, for example when a whole section is marked unavailable because its collector crashed. |
| 3 | Python types | Timestamps are timezone-aware `datetime` values normalised to UTC. JSON uses RFC 3339 with `Z`, with 6-digit fractional seconds only when non-zero. Arrays are tuples; `environment` is a read-only mapping; `Evidence.value` is frozen JSON (objects key-sorted). Integers given for `float` fields are stored as floats. |
| 4 | `device_ref` ordering | **Moved into §1 conventions (Stage 3).** Natural order (prefix, then numeric index). |
| 5 | Unspecified orderings | Processes by `pid`; toolkit candidates by `root_path`; PCI observations by `address`; `discovered_via` in enum order; `detection_sources` and redaction categories alphabetical; `collector_runs` in execution order (unchanged). |
| 6 | `RedactionCategory` | An enum for the four categories listed in §7.1. Same JSON values. |
| 7 | Enforced constant texts | `driver_supported_cuda_version.note` (when collected), `build_cuda_version.note` (when collected), readiness `criterion` and `disclaimer`, and the report's snapshot limitation `"This report is a point-in-time snapshot."`. |
| 8 | Enforced consistency | Device refs resolve to inventoried devices (runtime, readiness, findings); `device_count` equals the number of listed devices when collected; ref prefix matches `inventory_source`; `mode`, `options.required_vram_gib` and `readiness` agree; readiness covers exactly the inventoried devices; readiness status/reason pairs follow architecture §4.9; `required_bytes` and `margin_bytes` arithmetic; `cuda_build = false` forces `build_cuda_version`/`cudnn_version` to `NOT_APPLICABLE`; one finding per `(check_id, device_ref)`. |
| 9 | Severity and wording rules | **Stage 3:** the status/severity combinations in diagnostics spec §2 (corrected) are enforced by `DiagnosticFinding` and the JSON Schema. The recommendation URL allowlist and JSON Pointer resolution (AC28) remain diagnostics-stage checks. |
| 10 | Placement | `InspectionOptions` lives in `models/report.py` because the report echoes it; the core will reuse it. `CollectionError`, `UnsupportedCapability`, `CollectorRun` and the internal, never-serialised `CollectorOutcome` live in `models/collection.py`. Sections offer `unavailable(...)` builders (§9). |
| 11 | Decoding | `parse_report_json` / `from_json_dict` are strict: all keys required, unknown keys rejected, only `schema_version` `"1.0"` accepted. Forward-compatible tolerance (§7.3) remains a rule for external consumers. |
| 12 | Schema | Generated from the models (`build_report_schema()`) and shipped at the §7.2 path, with `$id` `urn:inferforge:gpu-system-inspector:report:1.0`. Per-field units are `const`; the `Observed` invariants and the mode/readiness rule are encoded with `if`/`then`. Regenerate with `uv run python -m gpu_system_inspector.reporting > src/gpu_system_inspector/reporting/schema/inspection_report.schema.json`. |
