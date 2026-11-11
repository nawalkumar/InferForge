"""Report assembly, serialisation and the versioned JSON Schema (independent of the CLI)."""

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
from gpu_system_inspector.reporting.schema import (
    build_report_schema,
    load_report_schema,
    render_report_schema,
)

__all__ = [
    "SerializationError",
    "assemble_report",
    "build_report_schema",
    "format_timestamp",
    "from_json_dict",
    "load_report_schema",
    "parse_report_json",
    "parse_timestamp",
    "render_json",
    "render_report_schema",
    "to_json_dict",
]
