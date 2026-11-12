# Demonstration Guide — Project 01: GPU System Inspector

Seven journeys that show what the tool does, each with copy-paste commands, the expected
result and the result actually recorded. Live results come from the development machine
(WSL2 on Windows, Ubuntu 22.04, **no NVIDIA GPU exposed**), 2026-11-11. Synthetic results come
from fixture data and are labelled as such everywhere they appear.

Run from `projects/01-gpu-system-inspector/` after `uv sync` at the repository root.

| Journey | Command | Expected | Recorded 2026-11-11 |
| ------- | ------- | -------- | ------------------- |
| A. Live inspection | `uv run inferforge-gpu inspect` | Report, exit 0; D04 `INFO` "no NVIDIA GPU is accessible", no claim that none exists | Exit 0, D04 `INFO` MEDIUM, no traceback |
| A. Live readiness | `uv run inferforge-gpu inspect --required-vram 8` | Exit 3, `UNKNOWN` (`DEVICE_PRESENCE_UNKNOWN`) | Exit 3 |
| B. Synthetic, ready | `uv run python examples/synthetic_showcase.py FX06 --required-vram 16` | Fixture banner; one device `READY`, overall `READY`, exit 0 | Exit 0 |
| B. Synthetic, not ready | `uv run python examples/synthetic_showcase.py FX07 --required-vram 8` | `NOT_READY` (`NO_DEVICE_READY`), D02 `FAIL`, exit 1 | Exit 1 |
| B. Synthetic, unknown | `uv run python examples/synthetic_showcase.py FX09 --required-vram 8` | `UNKNOWN`, exit 3 | Exit 3 |
| C. JSON and schema | `uv run inferforge-gpu inspect --json - > r.json`, then validate (below) | Only JSON on stdout; valid against the shipped schema; `data_origin` `LIVE` | Valid, `LIVE`, schema `1.0` |
| D. Redacted output | `uv run inferforge-gpu inspect --redact --json /tmp/r.json` | No user name, home directory, path, UUID or PCI ID in text, JSON or stderr | Exit 0; no user name or home path in any output |
| E. Unwritable output | `uv run inferforge-gpu inspect --json /nonexistent/r.json` (or a read-only directory) | Text still printed; `error: cannot write JSON report …`; exit 4; no partial file | Exit 4 for both; no file left |
| F. Installed wheel | Build, install into a fresh venv outside the repository, run (below) | CLI and suite work without the checkout | 3.12.12 and 3.11.15: CLI exit 0/3; 2691 passed, 28 skipped |
| G. NVIDIA hardware | Colab notebook [`notebooks/inferforge_nvidia_gpu_validation.ipynb`](../notebooks/inferforge_nvidia_gpu_validation.ipynb) or `python validation/hardware_evidence.py` on a GPU machine ([hardware_validation.md](hardware_validation.md)) | Evidence bundle, verdict `EVIDENCE_COLLECTED` pending owner review | **Not run on NVIDIA hardware.** On this machine: `BLOCKED: no NVIDIA GPU is accessible` |

Schema check for journey C:

```bash
uv run python -c "import json, jsonschema; from gpu_system_inspector.reporting import load_report_schema; jsonschema.validate(json.load(open('r.json')), load_report_schema()); print('valid')"
```

Journey F (outside the repository; `uv` creates the environment):

```bash
uv build projects/01-gpu-system-inspector --out-dir /tmp/gsi-dist          # from the repository root
uv venv --python 3.11 /tmp/gsi-venv
uv pip install --python /tmp/gsi-venv/bin/python /tmp/gsi-dist/*.whl
cd /tmp && /tmp/gsi-venv/bin/inferforge-gpu inspect
```

To run the test suite against the installed wheel, copy `tests/` and `pyproject.toml` next to
the venv and run its `python -m pytest` (`pytest`, `jsonschema` and `pytest-cov` installed).
The 27 tests that read `docs/` skip there (KI-23), as does the repository-only evidence-script
module.

## What the synthetic showcase is, and isn't

`examples/synthetic_showcase.py` runs the production CLI with fake adapters built from a test
fixture. Collectors, diagnostics, readiness, rendering, redaction, writing and exit codes are
the real code. The values are invented. Every report it produces says so: `data_origin` is
`FIXTURE`, the first and last lines are `*** FIXTURE DATA — NOT A REAL MEASUREMENT ***`,
device names read "Synthetic GPU … (fixture)", and stderr names the fixture. Never present
its output as a measurement, and never crop the banner out of a screenshot.

## Recording plan (about three minutes)

Record a terminal at 100+ columns, real commands only, no edited output.

| Step | Show | Say |
| ---- | ---- | --- |
| 1. Install (20 s) | `uv sync`, `uv run inferforge-gpu version` (or the pip route) | One command, Python 3.11+, read-only, no GPU needed to try it |
| 2. Live inspection (45 s) | `uv run inferforge-gpu inspect`, scroll SYSTEM → DRIVER & CUDA → FINDINGS | This laptop has no accessible NVIDIA GPU. Every missing value has a status and reason, never a zero; D04 says "not accessible", not "absent" |
| 3. Readiness (15 s) | `uv run inferforge-gpu inspect --required-vram 8; echo $?` | Exit 3: readiness can't be determined, so it isn't claimed |
| 4. Diagnostic findings, synthetic (40 s) | `uv run python examples/synthetic_showcase.py FX02b`, scroll to D01 and D03 | **Fixture data** (point at the banner): D01 ranks a cause with confidence; D03 keeps driver, toolkit and PyTorch CUDA versions apart |
| 5. Readiness, synthetic (25 s) | `… FX06 --required-vram 16; echo $?` then `… FX07 --required-vram 8; echo $?` | **Fixture data**: one GPU READY with margin, exit 0; then NOT_READY, exit 1 |
| 6. JSON (20 s) | `uv run inferforge-gpu inspect --json - \| head -40`, then the schema check | Versioned, schema-validated, every value with source and status |
| 7. Sharing safely (15 s) | `uv run inferforge-gpu inspect --redact --json /tmp/r.json` and `grep -c "$USER" /tmp/r.json` | Redaction before anything leaves the process |

Don't show a synthetic report without its banner, describe fixture numbers as measurements,
or imply that the tool has run on NVIDIA hardware until the tracker records a hardware run.
