# Roadmap — Project 01: GPU System Inspector

This is a reader's view of [app_tracker.md](app_tracker.md), which stays the single source of
truth for status. Last reviewed 2026-11-11. "Tested" means automated tests (fixtures and/or
the WSL2 development machine), never hardware validation.

## Implemented

| Area | What | Evidence |
| ---- | ---- | -------- |
| Collection | System, PCI (native Linux, read-only sysfs), NVML session, GPU hardware, driver, `nvidia-smi` (detection + fallback), runtime snapshot, CUDA environment, PyTorch probe (child process), UUID/PCI correlation; one guard per collector | Tracker §1.5–§1.12; fixture parity over 24 fixtures |
| Diagnostics | D01–D08 with evidence, confidence and limitations | Golden tests over 30 report cases (§1.13) |
| Readiness | Mode B free-memory check with disclaimer, exit codes 0/1/3 | All 9 Mode B fixture runs (§1.13) |
| Reports | Text, versioned JSON + shipped schema `1.0` (draft), redaction, atomic writer | §1.13; AC25–AC27, AC40 |
| CLI and API | `inferforge-gpu inspect`/`version`, `python -m gpu_system_inspector`, `api.inspect()` | AC46–AC48, AC58 |
| Packaging | Wheel with only the package; installs and passes the suite outside the repository on CPython 3.11 and 3.12 | Tracker §1.14 |
| No network access | Full inspection with sockets patched to raise | `tests/integration/test_no_network.py` (AC42, in-process) |
| Hardware validation tooling | Hardware tests HW-01 to HW-13, evidence bundle script, procedure | [hardware_validation.md](hardware_validation.md) |

## In progress

| Item | State | Next step |
| ---- | ----- | --------- |
| FR-11 runtime snapshot | Compute processes only; graphics processes missing (KI-36) | Needs the owner's merge rule first (below) |
| Acceptance checklist | 32 of 60 criteria met (tracker §16) | Remaining criteria are hardware-bound, owner reviews, or listed as good first issues |
| Documentation review | Post-approval amendments to architecture, data flow, data model, testing strategy await owner review (tracker §7) | Owner review |
| Python API stability (FR-23, IMP-C06) | `api.inspect()` works; stability not reviewed | Decide what's public before any release |

## Hardware validation required

Blocked until someone runs [the procedure](hardware_validation.md) on the right machine.

| ID | What it proves | Machine |
| -- | -------------- | ------- |
| HW-01 | NVML inventory equals `nvidia-smi` | Any NVIDIA Linux machine (Colab works) |
| HW-02 | PCI scan sees the NVIDIA device and its driver (C14) | Native Linux |
| HW-03 | Runtime values are plausible | Any NVIDIA machine |
| HW-04 | PyTorch probe isolation and correlation (C10, C11) | PyTorch with CUDA |
| HW-05 | Toolkit version detection (C12) | A CUDA Toolkit |
| HW-06 | D03's "toolkit not needed for PyTorch wheels" note | PyTorch CUDA wheel, no toolkit |
| HW-07 | WSL2 with the Windows driver | WSL2 + NVIDIA |
| HW-08 | Windows WDDM per-process memory | Windows + NVIDIA |
| HW-09 | Correlation under every device order | 2+ GPUs |
| HW-10 | Performance targets (NFR-11), NVML path | Any NVIDIA machine |
| HW-11 | macOS general information | A Mac |
| HW-12 | Readiness follows real memory pressure | NVIDIA + PyTorch (opt-in allocation) |

Also host-only checks that need no GPU: the real Windows registry and memory calls (KI-27)
and the macOS `sysctlbyname` path (KI-30): `uv run pytest -m host` on those systems.

## Owner decision required

None of these is resolved in code; the current behaviour is the conservative one.

| ID | Decision | Current behaviour |
| -- | -------- | ----------------- |
| KI-36 | How to merge compute and graphics processes (a PID in both) | Only compute processes listed; never presented as complete |
| KI-37 | Allowlist `ProgramFiles` for the Windows default toolkit location, or amend FR-07 | Windows uses `CUDA_HOME`, `CUDA_PATH`, `CUDA_PATH_V*`, `nvcc` on `PATH` only |
| KI-38 | Accept PCI domains wider than 16 bits once a source documents the formats | Such devices stay uncorrelated by PCI; never a wrong match |
| OQ-11 | Process-name fallback (`/proc/<pid>/comm`) | Names from `nvmlSystemGetProcessName` only |
| L4 | UUID/PCI one-to-one correlation policy | Ambiguity stays unresolved |
| DEC-04/06/07/08 | Approve the proposed-and-applied decisions | Applied, approval not recorded |
| DEC-44 to DEC-50 | Approve the implementation decisions | Applied, approval not recorded |
| AC43 | Review the read-only evidence | `TESTED` evidence recorded, review pending |
| AC09 | Freeze report schema `1.0` | Draft; may change without a version bump (KI-20) |
| OQ-03 | Distribution name for publication | `01-gpu-system-inspector`, `Private :: Do Not Upload` |
| OQ-06 | Make `--redact` the default for `--json PATH`? | Opt-in |
| OQ-08 | Static type checker | None configured |
| WSD-02 | Let `scripts/verify_isolated_install.py` accept projects | Project isolated installs are done by hand |
| Release | Publish at all; license text in the wheel; project URLs, authors and classifiers | Not published; repository license Apache-2.0, wheel lacks the license file |

## Future enhancements (deferred, not V1)

From [product_specification.md](product_specification.md) §7: `--python` to inspect another
interpreter's PyTorch, MIG instance enumeration, NVML clock event reasons, NVML ↔ `nvidia-smi`
cross-validation, `mem_get_info` as a readiness source, other frameworks, container detection,
Windows/macOS hardware presence checks, report diffing, model-memory estimation, `--quiet`,
colour/Markdown/HTML output.

Decided against (not deferred): a demo or simulation mode in the CLI (DEC-17), an
`inferforge` umbrella command (DEC-03), PyTorch as a dependency or extra (DEC-26).

Out of scope for this project: CUDA or Triton kernel development, GPU performance
optimisation, inference optimisation and model serving belong to later InferForge projects.
