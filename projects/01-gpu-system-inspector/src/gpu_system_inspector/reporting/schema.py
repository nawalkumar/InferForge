"""JSON Schema (Draft 2020-12) for the inspection report.

The shipped file `schema/inspection_report.schema.json` is generated from the models by
`build_report_schema()`; a test fails if the two differ. Regenerate with:

    uv run python -m gpu_system_inspector.reporting \\
        > src/gpu_system_inspector/reporting/schema/inspection_report.schema.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from importlib import resources
from typing import Any, get_args, get_origin

from gpu_system_inspector import REPORT_SCHEMA_NAME, REPORT_SCHEMA_VERSION
from gpu_system_inspector.models import InspectionReport, Observed
from gpu_system_inspector.models._validation import is_optional, model_hints, strip_optional
from gpu_system_inspector.models.cuda import ENVIRONMENT_NAME_REGEX
from gpu_system_inspector.models.enums import CollectionStatus, Source
from gpu_system_inspector.models.observed import ERROR_REQUIRED_STATUSES, UNITS, ErrorDetail
from gpu_system_inspector.reporting.json_report import TIMESTAMP_REGEX

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = f"urn:inferforge:gpu-system-inspector:report:{REPORT_SCHEMA_VERSION}"
SCHEMA_FILE = "inspection_report.schema.json"

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}
_OBSERVED_KEYS = ("value", "status", "source", "method", "unit", "error", "note")
_PROPERTY_NAMES = {"environment": {"pattern": ENVIRONMENT_NAME_REGEX}}


def build_report_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    root = _object_schema(InspectionReport, defs)
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": "GPU System Inspector report",
        "description": (
            f"{REPORT_SCHEMA_NAME} version {REPORT_SCHEMA_VERSION}. Generated from the "
            "gpu_system_inspector.models dataclasses."
        ),
        **root,
        "$defs": {name: defs[name] for name in sorted(defs)},
    }


def render_report_schema() -> str:
    return json.dumps(build_report_schema(), ensure_ascii=False, indent=2) + "\n"


def load_report_schema() -> dict[str, Any]:
    """The schema shipped with the package."""
    text = resources.files(__package__).joinpath("schema", SCHEMA_FILE).read_text("utf-8")
    return json.loads(text)


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def _object_schema(cls: type, defs: dict[str, Any]) -> dict[str, Any]:
    hints = model_hints(cls)
    properties: dict[str, Any] = {}
    for f in fields(cls):
        constraints = dict(f.metadata.get("schema", {}))
        if f.name in _PROPERTY_NAMES:
            constraints["propertyNames"] = _PROPERTY_NAMES[f.name]
        schema = _type_schema(hints[f.name], defs, constraints)
        unit = f.metadata.get("unit", ...)
        if unit is not ...:
            unit_rule = {"properties": {"unit": {"const": unit}}}
            if get_origin(hints[f.name]) is Mapping:
                schema["additionalProperties"] = {
                    "allOf": [schema["additionalProperties"], unit_rule]
                }
            else:
                schema = {"allOf": [schema, unit_rule]}
        properties[f.name] = schema
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [f.name for f in fields(cls)],
        "properties": properties,
    }
    schema.update(getattr(cls, "_SCHEMA_EXTRA", {}))
    return schema


def _type_schema(hint: Any, defs: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    if hint is Any:
        return {}
    if is_optional(hint):
        return {"anyOf": [_type_schema(strip_optional(hint), defs, constraints), {"type": "null"}]}
    origin = get_origin(hint)
    if origin is tuple:
        return {"type": "array", "items": _type_schema(get_args(hint)[0], defs, {})}
    if origin is Mapping:
        schema = {
            "type": "object",
            "additionalProperties": _type_schema(get_args(hint)[1], defs, {}),
        }
        schema.update(constraints)
        return schema
    if origin is Observed:
        return _ref(_observed_def(get_args(hint)[0], defs))
    if isinstance(hint, type) and issubclass(hint, Enum):
        defs.setdefault(hint.__name__, {"type": "string", "enum": [m.value for m in hint]})
        return _ref(hint.__name__)
    if is_dataclass(hint):
        if hint.__name__ not in defs:
            defs[hint.__name__] = {}
            defs[hint.__name__] = _object_schema(hint, defs)
        return _ref(hint.__name__)
    if hint is datetime:
        return {"type": "string", "format": "date-time", "pattern": TIMESTAMP_REGEX}
    if hint in _SCALARS:
        return {"type": _SCALARS[hint], **constraints}
    raise TypeError(f"no JSON Schema mapping for {hint!r}")


def _observed_def(value_hint: Any, defs: dict[str, Any]) -> str:
    if get_origin(value_hint) is tuple:
        name = f"Observed{get_args(value_hint)[0].__name__}List"
    elif isinstance(value_hint, type) and issubclass(value_hint, Enum):
        name = f"Observed{value_hint.__name__}"
    else:
        name = f"Observed{_SCALARS[value_hint].capitalize()}"
    if name in defs:
        return name
    defs[name] = {}
    value_schema = _type_schema(value_hint, defs, {})
    collected = CollectionStatus.COLLECTED.value
    defs[name] = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_OBSERVED_KEYS),
        "properties": {
            "value": {"anyOf": [value_schema, {"type": "null"}]},
            "status": _type_schema(CollectionStatus, defs, {}),
            "source": {"anyOf": [_type_schema(Source, defs, {}), {"type": "null"}]},
            "method": {"type": ["string", "null"]},
            "unit": {"enum": [*UNITS, None]},
            "error": {"anyOf": [_type_schema(ErrorDetail, defs, {}), {"type": "null"}]},
            "note": {"type": ["string", "null"]},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": collected}}},
                "then": {
                    "properties": {
                        "value": {"not": {"type": "null"}},
                        "source": {"not": {"type": "null"}},
                        "method": {"type": "string", "minLength": 1},
                        "error": {"type": "null"},
                    }
                },
                "else": {"properties": {"value": {"type": "null"}}},
            },
            {
                "if": {
                    "properties": {
                        "status": {"enum": sorted(s.value for s in ERROR_REQUIRED_STATUSES)}
                    }
                },
                "then": {"properties": {"error": {"type": "object"}}},
            },
            {
                "if": {"properties": {"status": {"const": CollectionStatus.SKIPPED.value}}},
                "then": {"properties": {"error": {"type": "null"}}},
            },
            {
                "if": {"properties": {"status": {"const": CollectionStatus.NOT_APPLICABLE.value}}},
                "then": {"properties": {"note": {"type": "string", "minLength": 1}}},
            },
        ],
    }
    return name
