"""SYNTHETIC showcase: the real `inferforge-gpu inspect` pipeline over a test fixture.

Every value printed comes from a synthetic fixture scenario in `tests/gsi_testkit` (fake
NVML, fake `nvidia-smi`, fake PyTorch probe), not from this machine and not from real NVIDIA
hardware. The report says so: `data_origin` is `FIXTURE` and the text report opens and
closes with `*** FIXTURE DATA — NOT A REAL MEASUREMENT ***`. Collectors, diagnostics,
readiness, redaction, rendering, atomic writing and exit codes are the production code,
exactly as `inferforge-gpu inspect` runs them; only the adapters are fakes.

    uv run python examples/synthetic_showcase.py --list
    uv run python examples/synthetic_showcase.py FX06 --required-vram 16
    uv run python examples/synthetic_showcase.py FX02b
    uv run python examples/synthetic_showcase.py FX07 --required-vram 8 --json -

Any `inspect` option may follow the fixture ID. The exit code is the CLI's. Needs the
repository checkout (the test kit isn't shipped in the wheel). Not part of the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from gsi_testkit.fakes import FakeAdapterSet  # noqa: E402
from gsi_testkit.fixtures import ALL_FIXTURES, FIXTURES_BY_ID  # noqa: E402

from gpu_system_inspector.cli import main as cli_main  # noqa: E402
from gpu_system_inspector.core import Inspector  # noqa: E402
from gpu_system_inspector.models import DataOrigin, InspectionOptions  # noqa: E402


def _list() -> int:
    for fixture in ALL_FIXTURES:
        runs = ", ".join(f"--required-vram {r.required_vram_gib:g}" for r in fixture.readiness_runs)
        extra = f"  [Mode B: {runs}]" if runs else ""
        print(f"{fixture.fixture_id:6} {fixture.name}{extra}")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        return _list()
    fixture = FIXTURES_BY_ID.get(argv[0])
    if fixture is None:
        print(f"unknown fixture {argv[0]!r}; use --list", file=sys.stderr)
        return 2

    def run_fixture(options: InspectionOptions):
        adapters = FakeAdapterSet.from_scenario(fixture).adapter_set()
        report = Inspector(adapters).run(options)
        if report.data_origin is not DataOrigin.FIXTURE:
            raise RuntimeError("synthetic showcase produced a non-FIXTURE report")
        return report

    cli_main.run_inspection = run_fixture
    print(f"[synthetic showcase: fixture {fixture.fixture_id} — {fixture.name}]", file=sys.stderr)
    return cli_main.main(["inspect", *argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
