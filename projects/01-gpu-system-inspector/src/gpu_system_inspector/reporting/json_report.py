"""JSON serialisation of the report models (docs/data_model_specification.md §7.2).

Serialisation is driven by the model type hints, so field names, order and types always
match the dataclasses. Decoding is strict: every key must be present, unknown keys are
rejected, and every value is validated by the model constructors.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar, get_args, get_origin

from gpu_system_inspector import REPORT_SCHEMA_NAME, REPORT_SCHEMA_VERSION
from gpu_system_inspector.models import InspectionReport, InvalidModelError, Observed
from gpu_system_inspector.models._validation import model_hints, strip_optional

ModelT = TypeVar("ModelT")

TIMESTAMP_REGEX = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z$"
_TIMESTAMP = re.compile(TIMESTAMP_REGEX)


class SerializationError(ValueError):
    """Raised for values that can't be serialised or documents that don't match the models."""


def format_timestamp(value: datetime) -> str:
    """RFC 3339 UTC with `Z`; microseconds only when non-zero."""
    if value.tzinfo is None:
        raise SerializationError("timestamps must be timezone-aware")
    utc = value.astimezone(UTC)
    text = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        text += f".{utc.microsecond:06d}"
    return text + "Z"


def parse_timestamp(text: str) -> datetime:
    if not isinstance(text, str) or _TIMESTAMP.match(text) is None:
        raise SerializationError(f"expected an RFC 3339 UTC timestamp ending in 'Z', got {text!r}")
    base, _, fraction = text[:-1].partition(".")
    parsed = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return parsed.replace(microsecond=int(fraction.ljust(6, "0"))) if fraction else parsed


def to_json_dict(model: Any) -> dict[str, Any]:
    """Convert a model to plain JSON types (keys in model field order)."""
    if not is_dataclass(model) or isinstance(model, type):
        raise SerializationError(f"expected a model instance, got {type(model).__name__}")
    return _encode(model, "")


def render_json(model: Any) -> str:
    """Deterministic JSON text: UTF-8, 2-space indent, no NaN/Infinity, trailing newline."""
    return json.dumps(to_json_dict(model), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def from_json_dict(data: Any, model_type: type[ModelT]) -> ModelT:
    """Rebuild a model of `model_type` from plain JSON data."""
    if model_type is InspectionReport:
        _check_report_header(data)
    return _decode(model_type, data, "")


def parse_report_json(text: str | bytes) -> InspectionReport:
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"invalid JSON: {exc}") from exc
    return from_json_dict(data, InspectionReport)


def _reject_constant(name: str) -> Any:
    raise SerializationError(f"invalid JSON: {name} is not allowed")


def _check_report_header(data: Any) -> None:
    if not isinstance(data, dict):
        raise SerializationError("/: a report must be a JSON object")
    if data.get("schema_name") != REPORT_SCHEMA_NAME:
        raise SerializationError(
            f"/schema_name: expected {REPORT_SCHEMA_NAME!r}, got {data.get('schema_name')!r}"
        )
    if data.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise SerializationError(
            f"/schema_version: unsupported version {data.get('schema_version')!r} "
            f"(this build reads {REPORT_SCHEMA_VERSION!r})"
        )


def _pointer(path: str, key: str | int) -> str:
    return f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}"


def _encode(value: Any, path: str) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _encode(getattr(value, f.name), _pointer(path, f.name)) for f in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return format_timestamp(value)
    if isinstance(value, (tuple, list)):
        return [_encode(item, _pointer(path, i)) for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {key: _encode(value[key], _pointer(path, key)) for key in sorted(value)}
    if isinstance(value, float) and not math.isfinite(value):
        raise SerializationError(f"{path or '/'}: NaN and Infinity can't be serialised")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise SerializationError(f"{path or '/'}: {type(value).__name__} can't be serialised")


def _decode(hint: Any, data: Any, path: str) -> Any:
    where = path or "/"
    if hint is Any:
        return data
    if data is None:
        if strip_optional(hint) is not hint:
            return None
        raise SerializationError(f"{where}: null is not allowed here")
    hint = strip_optional(hint)
    origin = get_origin(hint)

    if origin is Observed:
        return _decode_model(Observed, data, path, value_hint=get_args(hint)[0] | None)
    if is_dataclass(hint):
        return _decode_model(hint, data, path)
    if origin is tuple:
        if not isinstance(data, list):
            raise SerializationError(f"{where}: expected an array")
        item_hint = get_args(hint)[0]
        return tuple(_decode(item_hint, item, _pointer(path, i)) for i, item in enumerate(data))
    if origin is Mapping:
        if not isinstance(data, dict):
            raise SerializationError(f"{where}: expected an object")
        item_hint = get_args(hint)[1]
        return {key: _decode(item_hint, item, _pointer(path, key)) for key, item in data.items()}
    if isinstance(hint, type) and issubclass(hint, Enum):
        try:
            return hint(data)
        except ValueError:
            allowed = ", ".join(member.value for member in hint)
            raise SerializationError(
                f"{where}: {data!r} is not a valid {hint.__name__} ({allowed})"
            ) from None
    if hint is datetime:
        try:
            return parse_timestamp(data)
        except SerializationError as exc:
            raise SerializationError(f"{where}: {exc}") from None
    if hint is bool:
        if type(data) is not bool:
            raise SerializationError(f"{where}: expected a boolean, got {data!r}")
        return data
    if hint is int:
        if type(data) is not int:
            raise SerializationError(f"{where}: expected an integer, got {data!r}")
        return data
    if hint is float:
        if type(data) not in (int, float):
            raise SerializationError(f"{where}: expected a number, got {data!r}")
        return float(data)
    if hint is str:
        if not isinstance(data, str):
            raise SerializationError(f"{where}: expected a string, got {data!r}")
        return data
    raise SerializationError(f"{where}: unsupported type {hint!r}")


def _decode_model(cls: type, data: Any, path: str, *, value_hint: Any = None) -> Any:
    where = path or "/"
    if not isinstance(data, dict):
        raise SerializationError(f"{where}: expected an object for {cls.__name__}")
    hints = dict(model_hints(cls))
    if value_hint is not None:
        hints["value"] = value_hint
    names = [f.name for f in fields(cls)]
    missing = [name for name in names if name not in data]
    unknown = sorted(set(data) - set(names))
    if missing:
        raise SerializationError(f"{where}: missing key(s) {', '.join(missing)} in {cls.__name__}")
    if unknown:
        raise SerializationError(f"{where}: unknown key(s) {', '.join(unknown)} in {cls.__name__}")
    values = {name: _decode(hints[name], data[name], _pointer(path, name)) for name in names}
    try:
        return cls(**values)
    except (InvalidModelError, TypeError) as exc:
        raise SerializationError(f"{where}: invalid {cls.__name__}: {exc}") from exc
