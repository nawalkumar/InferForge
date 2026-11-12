"""Report assembly, serialisation, the versioned JSON Schema, the text renderer, redaction and
the atomic writer (independent of the CLI)."""

from gpu_system_inspector.reporting.assemble import assemble_report
from gpu_system_inspector.reporting.json_report import (
    SerializationError,
    format_timestamp,
    from_json_dict,
    parse_report_json,
    parse_timestamp,
    render_json,
    to_json_dict,
)
from gpu_system_inspector.reporting.redaction import (
    DEFAULT_REDACTION,
    RedactionPolicy,
    Redactor,
    redact,
)
from gpu_system_inspector.reporting.schema import (
    build_report_schema,
    load_report_schema,
    render_report_schema,
)
from gpu_system_inspector.reporting.text_report import FIXTURE_BANNER, render_text
from gpu_system_inspector.reporting.writer import OutputError, write_atomic

__all__ = [
    "DEFAULT_REDACTION",
    "FIXTURE_BANNER",
    "OutputError",
    "RedactionPolicy",
    "Redactor",
    "SerializationError",
    "assemble_report",
    "build_report_schema",
    "format_timestamp",
    "from_json_dict",
    "load_report_schema",
    "parse_report_json",
    "parse_timestamp",
    "redact",
    "render_json",
    "render_report_schema",
    "render_text",
    "to_json_dict",
    "write_atomic",
]
