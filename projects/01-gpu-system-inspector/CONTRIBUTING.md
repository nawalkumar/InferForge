# Contributing to GPU System Inspector (InferForge Project 01)

Thanks for helping. This guide is enough to set up, find your way around, make a change and
get it reviewed without asking the original developer. Status lives in one place,
[docs/app_tracker.md](docs/app_tracker.md); plans and open decisions in
[docs/roadmap.md](docs/roadmap.md).

## Ground rules

These keep the reports trustworthy. A change that breaks one of them won't be merged.

1. **Read-only.** The inspector never installs software, changes GPU state, writes outside
   `--json PATH`, or opens network connections. Only `adapters/commands.py` may start
   processes, never through a shell (`tests/unit/test_boundaries.py` enforces it).
2. **Missing is not zero.** Every value is an `Observed` with a status. `value` is set exactly
   when the status is `COLLECTED`; anything else carries a reason. The models reject
   violations.
3. **Absence is not inaccessibility.** "No NVIDIA GPU" is concluded only from NVML counting
   zero, macOS, or a complete native-Linux PCI scan (`diagnostics/context.py`).
4. **Versions are facts, not proofs.** A driver, toolkit or PyTorch build version never shows
   that CUDA works. No CUDA code runs during an inspection.
5. **Fixture data is always labelled.** Fixture reports carry `data_origin = FIXTURE` and the
   banner; the CLI can't produce them (DEC-17).
6. **Specification first.** Behaviour is specified in `docs/` (approved by the owner). If the
   code needs to differ, change the specification in the same pull request and say so.
7. **Owner decisions stay open until the owner decides.** The items under
   [Owner decision required](docs/roadmap.md#owner-decision-required) aren't
   implementation tasks.
8. **No new dependencies** without an agreed need (runtime: only `nvidia-ml-py`; PyTorch is
   detected, never required).
9. **No hardware claims without hardware.** Fakes never count as hardware validation. Only a
   passed hardware test with its evidence moves a tracker item to `HARDWARE_VALIDATED`.

## Development setup

Requirements: Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), Linux, WSL2 or macOS
(Windows works for the code; its host tests have never run, KI-27). No GPU is needed.

```bash
git clone https://github.com/nawalkumar/InferForge.git && cd InferForge
uv sync                                     # whole workspace, editable, with test tools
cd projects/01-gpu-system-inspector
uv run inferforge-gpu inspect               # smoke test
```

## Tests and quality gates

Run these before opening a pull request (from `projects/01-gpu-system-inspector/` unless
noted):

```bash
uv run pytest                                # unit + integration + host; ~5 minutes
uv run pytest --cov                          # coverage; fails below 92 % (pyproject.toml)
uv run ruff check . && uv run ruff format --check .
cd ../.. && uv run python scripts/check_dependencies.py && uv lock --check && uv run pytest
```

| Marker | Runs by default | What it needs |
| ------ | --------------- | ------------- |
| (none) | yes | nothing: fakes and fixtures |
| `host` | yes | reads the real OS read-only; no NVIDIA hardware (`uv run pytest -m host` alone) |
| `hardware` | **no** | a real NVIDIA machine: `uv run pytest -m hardware tests/hardware -rA` ([docs/hardware_validation.md](docs/hardware_validation.md)) |

Useful single runs: `uv run pytest tests/unit/test_diagnostics_rules.py -k D04`,
`uv run pytest tests/integration/test_diagnostics_golden.py -k FX07`.

Don't weaken a test or the coverage threshold to make a change pass. If a test pins
behaviour the specification changed, update both and say why in the pull request.

## Where things live

```
src/gpu_system_inspector/
  cli/            argparse front end, validation (exit 2), exit codes, output routing
  api.py          inspect(options) -> InspectionReport: the provisional Python API
  core/           Inspector (collector order, guard, NVML session), production adapter factory
  collectors/     one collector per report section; turn adapter outcomes into Observed values
  adapters/       the only code that touches the host: NVML, nvidia-smi, sysfs, files, processes
  models/         frozen dataclasses for every report part; the invariants are enforced here
  diagnostics/    D01–D08 rules: pure functions of the snapshot (context.py: shared facts)
  readiness/      Mode B: pure function of the snapshot
  reporting/      assembly, JSON + shipped schema, text renderer, redaction, atomic writer
tests/
  unit/           one module per component; no host access except host-marked tests
  integration/    the real Inspector and CLI over every fixture; golden text files
  hardware/       HW-01 to HW-13; never run by default
  gsi_testkit/    builders, the FX fixture catalogue, expected findings, fakes of every adapter
examples/         inspect_demo.py, synthetic_showcase.py, reports/ (example outputs)
validation/       hardware_evidence.py: the evidence bundle for hardware validation
docs/             specifications (approved), tracker, procedures
```

Dependencies only point downwards: `cli → api/core → collectors → adapters → models`;
`diagnostics`, `readiness` and `reporting` use `models` only. `tests/unit/test_boundaries.py`
fails if an import crosses a layer. Read [docs/architecture.md](docs/architecture.md) §2–§4
for the full design and [docs/data_flow.md](docs/data_flow.md) for the execution order.

## How to add or change a diagnostic rule

1. **Specify it.** Add or change the decision table in
   [docs/diagnostics_specification.md](docs/diagnostics_specification.md) §3: inputs, rows,
   status, severity, confidence, wording. Follow the wording rules (§5): `observed:` vs
   `inferred:`, causal language only at `HIGH` confidence, references only to the allowed
   official domains (`diagnostics/evidence.py`, `ALLOWED_REFERENCE_DOMAINS`).
2. **Implement it** in `diagnostics/rules.py` as `dNN(data: DiagnosticInput) ->
   list[DiagnosticFinding]`, using the helpers in `diagnostics/evidence.py` (`finding`,
   `observed_field`, `inference`, `advice`, `not_applicable`) and the shared facts in
   `diagnostics/context.py`. Register it in `DEFAULT_RULES`. A rule may read only the
   snapshot (and readiness); the engine turns an exception into an `UNKNOWN` finding plus a
   `diagnostics.DNN` collection error.
3. **Pin the expected outcome for every fixture** in the catalogue
   (`tests/gsi_testkit/fixtures/catalog_*.py`, `expected_diagnostics`).
   `tests/integration/test_diagnostics_golden.py` checks status, severity, confidence and
   device for every fixture and Mode B run, and the wording rules over every finding.
4. **Unit-test the rows no fixture reaches** in `tests/unit/test_diagnostics_rules.py`.
5. If the text report changes, regenerate the golden files on purpose and review the diff:
   `GSI_UPDATE_GOLDEN=1 uv run pytest tests/integration/test_cli.py -k golden`.

## How to add a fixture

Fixtures are **expected snapshots**: the fakes are derived from them
(`FakeAdapterSet.from_scenario`), and parity tests check that the production collectors
reproduce them.

1. Describe it in [docs/testing_strategy.md](docs/testing_strategy.md) §4 (inputs, expected
   findings, Mode B runs).
2. Add a factory returning a `FixtureScenario` to `tests/gsi_testkit/fixtures/catalog_core.py`
   or `catalog_platform.py`, and list it in `CORE_FIXTURES`/`PLATFORM_FIXTURES`. Build the
   snapshot with `gsi_testkit.builders`; set `expected_diagnostics`, `readiness_runs`, and
   any `premises` the outcome depends on. Device names must say they're synthetic.
3. Run the whole suite. Catalogue tests count the report cases
   (`tests/integration/test_fixture_catalog.py`): update the count deliberately.
4. Use the showcase to look at it: `uv run python examples/synthetic_showcase.py FXnn`.

## How to add a collector

1. Specify the section in [docs/data_model_specification.md](docs/data_model_specification.md)
   and the order in [docs/data_flow.md](docs/data_flow.md).
2. Implement the contract in `collectors/base.py`: a `name`, adapters and `Clock` via the
   constructor, `collect() -> CollectorOutcome` (`COMPLETED` or `PARTIAL`, never `FAILED`;
   let unexpected exceptions propagate) and `unavailable(error)` for the guard's fallback.
   Map adapter outcomes to statuses with `adapters/outcomes.py`; never invent a value.
3. Add a `run_guarded(...)` step at the right place in `Inspector.run`
   (`core/inspector.py`); the order is that sequence. The guard (`core/guard.py`) is what
   produces `FAILED` and the collection error.
4. If it emits unsupported capabilities, give each capability exactly one owner: report
   assembly (`reporting/assemble.py`) rejects a `(capability, device_ref)` pair reported twice.
5. Tests: unit tests with fakes, fixture parity over all fixtures, a failure-isolation case,
   and update the host tests' expected collector order.

## How to add a platform adapter

1. Define or extend the protocol in `adapters/protocols.py`. Adapters return normalised
   outcomes (`adapters/outcomes.py`), not exceptions, for expected host conditions.
2. Implement it in `adapters/`: read-only, standard library or `nvidia-ml-py` only, OS
   modules imported only where `tests/unit/test_boundaries.py` allows. Processes only through
   `CommandRunner`.
3. Wire it in `core/factory.py` (the only place that names production adapter classes) and
   `AdapterSet`.
4. Add a fake to `tests/gsi_testkit/fakes/` that conforms to the protocol;
   `tests/unit/test_adapter_contracts.py` checks members, parameter names and defaults.
5. Remember what a fake can't prove: plan the host test or hardware test that would.

## Reporting a bug

Please include:

1. `inferforge-gpu version`;
2. the exact command and its exit code;
3. a **redacted** report: `inferforge-gpu inspect --redact --json report.json` (paths,
   process names, GPU UUIDs, PCI bus IDs, user name and home directory are replaced);
4. what you expected and why (for example the `nvidia-smi` output you compared with).

Use the bug report template on GitHub ("GPU System Inspector: bug report").

## Reproducing a reported bug

1. Load the report: `parse_report_json(Path("report.json").read_text())` gives the same
   models the code uses; the status, `error.code` and `source` of the fields involved
   usually locate the collector or rule.
2. Find the closest fixture (`uv run python examples/synthetic_showcase.py --list`) and run it
   with the reporter's options. A rule or rendering bug usually reproduces there.
3. Write the failing test first: a unit test with a builder snapshot for a rule, a fixture
   (or a variant) for a collector, a fake-adapter case for an adapter.
4. Host-specific problems (a real driver, a real WSL kernel) need the reporter's machine:
   ask them to run `pytest -m host` or the hardware procedure, and record the result in the
   tracker.

## Pull requests

- Keep them small and focused; reference the tracker IDs (FR, AC, KI, DEC) involved.
- Update the tracker ([docs/app_tracker.md](docs/app_tracker.md)) with what changed, its
  status and the evidence (test names, commands). Never set `USER_VERIFIED` or
  `HARDWARE_VALIDATED` without that evidence.
- Pull requests are reviewed by the project owner. Nothing is published automatically.
