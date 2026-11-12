# Good first issue candidates — Project 01: GPU System Inspector

Each item below is real, bounded and needs no owner decision. Each one has been checked
against the code on 2026-11-11. Owner decisions (KI-36, KI-37, KI-38, OQ-11, L4, the schema
freeze, …) are deliberately **not** here: see [roadmap.md](roadmap.md#owner-decision-required).
Read [CONTRIBUTING.md](../CONTRIBUTING.md) first.

## 1. Long environment-variable labels break the text report's alignment

- **Where:** `reporting/text_report.py` (`_Out.field`, `LABEL = 24`).
- **Observed:** in a live report the labels `NVIDIA_DRIVER_CAPABILITIES`,
  `PYTORCH_NO_CUDA_MEMORY_CACHING` and `PYTORCH_NVML_BASED_CUDA_CHECK` are wider than the
  label column, so their values start at a different column and wrap under the wrong indent
  ([example](../examples/reports/live-wsl2-no-nvidia.redacted.txt), CUDA TOOLKIT section).
- **Done when:** values of over-long labels start on the next line at the value column (or
  another consistent rule), lines stay ≤ 100 characters, the golden files are regenerated
  deliberately (`GSI_UPDATE_GOLDEN=1`, see CONTRIBUTING) and the diff is reviewed.

## 2. Make `-v` observable: INFO log lines (AC60)

- **Where:** `cli/main.py` and/or `core/inspector.py`.
- **Observed:** the source has no `log.info` call, so `-v` and the default level produce the
  same stderr; AC60 ("`-v`: INFO") can't be tested (tracker AC60 `IN_PROGRESS`).
- **Done when:** one INFO line per collector run (name, run status, duration) appears on
  stderr with `-v` and not by default. The lines carry no paths, identifiers or values, so
  redaction isn't affected. A CLI test shows that stdout (text and `--json -`) is byte-identical
  across verbosity levels.

## 3. One test per row of the error-mapping table (AC14)

- **Where:** a new `tests/unit/test_error_mapping.py`, against `adapters/outcomes.py`.
- **Observed:** AC14 asks for "one case per row of data model §3.3"; the mappings are
  exercised across many adapter tests but there's no row-by-row test (tracker AC14
  `PLANNED`).
- **Done when:** every row of [data_model_specification.md](data_model_specification.md)
  §3.3 has a parametrised case asserting the status and `error.code`, and a test asserts
  that no fixture report contains the `UNKNOWN` status.

## 4. Write the demo script's JSON atomically

- **Where:** `examples/inspect_demo.py` (`Path(args.json).write_text(...)`).
- **Observed:** the CLI writes JSON with `reporting.write_atomic`; the demo doesn't, so an
  interrupted demo can leave a partial file.
- **Done when:** the demo uses `write_atomic` and reports an `OutputError` the way the CLI
  does.

## 5. Run the host tests on Windows or macOS (no GPU needed)

- **Where:** `uv run pytest -m host` on a Windows or macOS machine.
- **Observed:** the Windows registry and memory calls (KI-27) and the macOS `sysctlbyname`
  path (KI-30) have only run against simulated APIs.
- **Done when:** the output is attached to an issue (with `inferforge-gpu version` and the
  OS version), and any failure is turned into a known issue with a fix.

## 6. Produce hardware evidence on a GPU machine

- **Where:** [hardware_validation.md](hardware_validation.md); Google Colab with a GPU
  runtime is enough for HW-01, HW-03, HW-04, HW-05 and HW-10.
- **Observed:** no NVIDIA hardware run exists; HW-01 to HW-12 are `BLOCKED`.
- **Done when:** the evidence bundle (`SUMMARY.md`, `summary.json`, `hardware-tests.xml`, and
  privately the unredacted files) is attached to an issue. The owner records the results.
