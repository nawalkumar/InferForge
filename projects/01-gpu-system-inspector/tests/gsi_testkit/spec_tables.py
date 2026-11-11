"""Read the expected-outcome tables from the approved specification documents.

Used to prove that the fixture expectations match diagnostics spec §6 and testing strategy
§4.1 exactly. The documents are not part of an installed wheel, so callers skip when
`DOCS_DIR` doesn't exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

_STATUS_WORDS = {"PASS", "FAIL", "WARNING", "INFO", "UNKNOWN", "N/A"}
_DEVICE = re.compile(r"\b((?:nvml|smi):\d+)\b")


def _table_rows(text: str, heading: str) -> list[list[str]]:
    section = text.split(heading, 1)[1]
    rows = []
    for line in section.splitlines()[1:]:
        if line.startswith("#"):
            break
        if line.startswith("|") and not set(line) <= set("|- "):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]  # drop the header row


def _status(token: str) -> str:
    word = token.split()[0]
    if word not in _STATUS_WORDS:
        raise ValueError(f"unexpected status cell {token!r}")
    return "NOT_APPLICABLE" if word == "N/A" else word


def parse_cell(cell: str) -> dict[str | None, str]:
    """`PASS` → {None: PASS}; `FAIL (nvml:1, lost) / PASS (nvml:0)` → per-device statuses."""
    parts = [p.strip() for p in cell.split(" / ")]
    result: dict[str | None, str] = {}
    for part in parts:
        device = _DEVICE.search(part)
        result[device.group(1) if device else None] = _status(part)
    return result


def diagnostics_matrix() -> dict[str, dict[str, dict[str | None, str]]]:
    """fixture id → check id → {device_ref or None: status} (diagnostics spec §6)."""
    text = (DOCS_DIR / "diagnostics_specification.md").read_text(encoding="utf-8")
    checks = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
    matrix = {}
    for row in _table_rows(text, "## 6. Expected findings per fixture scenario"):
        fixture_id = row[0].split()[0]
        matrix[fixture_id] = {
            check: parse_cell(cell) for check, cell in zip(checks, row[1:], strict=True)
        }
    return matrix


@dataclass(frozen=True)
class ReadinessRow:
    label: str
    required_vram_gib: float
    overall_status: str
    overall_reason: str
    device_text: str
    d02: str
    exit_code: int


def readiness_table() -> dict[str, ReadinessRow]:
    """Run label → row (testing strategy §4.1)."""
    text = (DOCS_DIR / "testing_strategy.md").read_text(encoding="utf-8")
    rows = {}
    for row in _table_rows(text, "### 4.1 Expected readiness outcomes (Mode B runs)"):
        label, vram, overall, devices, d02, exit_code = row
        status, reason = re.findall(r"`([A-Z_]+)`", overall)
        rows[label] = ReadinessRow(
            label, float(vram), status, reason, devices, _status(d02), int(exit_code)
        )
    return rows


@dataclass(frozen=True)
class ErrorMappingRow:
    origin: str
    condition: str
    status: str
    code: str


def error_mapping_table() -> list[ErrorMappingRow]:
    """Rows of data model §3.3, with backticks removed."""
    text = (DOCS_DIR / "data_model_specification.md").read_text(encoding="utf-8")
    rows = _table_rows(text, "### 3.3 Error-to-status mapping (normative)")
    return [ErrorMappingRow(*(cell.replace("`", "") for cell in row)) for row in rows]


def device_results(text: str) -> dict[str, str] | str:
    """`nvml:0` READY …; … → {ref: reason}; `both X` → X for every device; `none` → {}."""
    if text == "none":
        return {}
    both = re.fullmatch(r"both `([A-Z_]+)`", text)
    if both:
        return both.group(1)
    result = {}
    for segment in text.split(";"):
        ref = _DEVICE.search(segment).group(1)
        reasons = re.findall(r"`([A-Z_]{4,})`", segment)
        result[ref] = reasons[0] if reasons else "SUFFICIENT_FREE_MEMORY"
    return result
