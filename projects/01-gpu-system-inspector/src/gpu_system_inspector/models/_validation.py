"""Construction-time validation shared by every report model.

Each model calls `validate_model(self)` from `__post_init__`. Field metadata declares the
unit an `Observed` field must carry (`unit`) and scalar constraints (`schema`: `pattern`,
`minimum`, `maximum`, `exclusiveMinimum`, `minLength`, `const`). The JSON Schema generator
reads the same metadata, so validation and schema can't drift apart.
"""

from __future__ import annotations

import math
import re
import types
from collections.abc import Mapping
from dataclasses import Field, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from functools import cache
from typing import Any, Union, get_args, get_origin, get_type_hints

from gpu_system_inspector.models.observed import ErrorDetail, InvalidModelError, Observed

DEVICE_REF_REGEX = r"^(nvml|smi):(0|[1-9][0-9]*)$"
JSON_POINTER_REGEX = r"^(/([^~/]|~[01])*)*$"
_DEVICE_REF = re.compile(DEVICE_REF_REGEX)
_MISSING = object()

_UNION_TYPES = (Union, types.UnionType)


def observed(unit: str | None = None) -> Any:
    """Declare an `Observed` field and the unit its values must carry."""
    return field(metadata={"unit": unit})


def constrained(*, default: Any = _MISSING, **schema: Any) -> Any:
    """Declare a scalar field with JSON-Schema-style constraints."""
    if default is _MISSING:
        return field(metadata={"schema": schema})
    return field(default=default, metadata={"schema": schema})


def device_ref_key(ref: str) -> tuple[str, int]:
    """Natural ordering for device refs: `nvml:2` sorts before `nvml:10`."""
    match = _DEVICE_REF.match(ref)
    if match is None:
        raise InvalidModelError(f"invalid device_ref {ref!r}")
    return match.group(1), int(match.group(2))


def optional_ref_key(ref: str | None) -> tuple[int, str, int]:
    return (0, "", 0) if ref is None else (1, *device_ref_key(ref))


def device_ref_index(ref: str) -> int:
    return device_ref_key(ref)[1]


@cache
def model_hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def is_optional(hint: Any) -> bool:
    return get_origin(hint) in _UNION_TYPES and type(None) in get_args(hint)


def strip_optional(hint: Any) -> Any:
    if get_origin(hint) in _UNION_TYPES:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(args) != 1:
            raise TypeError(f"only `X | None` unions are supported, got {hint!r}")
        return args[0]
    return hint


def validate_model(obj: Any) -> None:
    """Normalise sequences to tuples and check every field against its type hint."""
    hints = model_hints(type(obj))
    for f in fields(obj):
        where = f"{type(obj).__name__}.{f.name}"
        hint = hints[f.name]
        value = _normalise(hint, getattr(obj, f.name))
        object.__setattr__(obj, f.name, value)
        check_value(hint, value, where)
        _check_metadata(f, value, where)


def set_field(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def unavailable(
    cls: type, fill_status: Any, fill_error: ErrorDetail | None, /, **given: Any
) -> Any:
    """Build `cls` with every `Observed` field set to the same non-collected status.

    Fields that aren't `Observed` must be supplied in `given`.
    """
    hints = model_hints(cls)
    values: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in given:
            values[f.name] = given[f.name]
        elif get_origin(hints[f.name]) is Observed:
            values[f.name] = Observed(
                value=None,
                status=fill_status,
                source=None,
                method=None,
                unit=f.metadata.get("unit"),
                error=fill_error,
            )
        else:
            raise TypeError(f"{cls.__name__}.unavailable() needs a value for {f.name!r}")
    return cls(**values)


def freeze_json(value: Any, where: str) -> Any:
    """Validate a JSON value and make it immutable (lists → tuples, objects → read-only)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidModelError(f"{where}: JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, f"{where}[{i}]") for i, item in enumerate(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise InvalidModelError(f"{where}: JSON object keys must be strings")
        return types.MappingProxyType(
            {key: freeze_json(value[key], f"{where}.{key}") for key in sorted(value)}
        )
    raise InvalidModelError(f"{where}: {type(value).__name__} is not a JSON value")


def check_value(hint: Any, value: Any, where: str) -> None:
    if hint is Any:
        freeze_json(value, where)
        return
    if get_origin(hint) in _UNION_TYPES:
        if value is None and is_optional(hint):
            return
        check_value(strip_optional(hint), value, where)
        return
    origin = get_origin(hint)
    if origin is tuple:
        item_hint = get_args(hint)[0]
        if not isinstance(value, tuple):
            raise InvalidModelError(f"{where}: expected a sequence, got {type(value).__name__}")
        for i, item in enumerate(value):
            check_value(item_hint, item, f"{where}[{i}]")
        return
    if origin is Mapping:
        _key_hint, item_hint = get_args(hint)
        if not isinstance(value, Mapping):
            raise InvalidModelError(f"{where}: expected a mapping, got {type(value).__name__}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidModelError(f"{where}: mapping keys must be strings")
            check_value(item_hint, item, f"{where}[{key!r}]")
        return
    if origin is Observed:
        if not isinstance(value, Observed):
            raise InvalidModelError(f"{where}: expected Observed, got {type(value).__name__}")
        if value.value is not None:
            check_value(get_args(hint)[0], value.value, f"{where}.value")
        return
    if isinstance(hint, type) and issubclass(hint, Enum):
        if not isinstance(value, hint):
            raise InvalidModelError(f"{where}: expected {hint.__name__}, got {value!r}")
        return
    if hint is bool:
        if type(value) is not bool:
            raise InvalidModelError(f"{where}: expected bool, got {value!r}")
        return
    if hint is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidModelError(f"{where}: expected int, got {value!r}")
        return
    if hint is float:
        if not isinstance(value, float) or not math.isfinite(value):
            raise InvalidModelError(f"{where}: expected a finite float, got {value!r}")
        return
    if hint is str:
        if not isinstance(value, str):
            raise InvalidModelError(f"{where}: expected str, got {value!r}")
        return
    if hint is datetime:
        if not isinstance(value, datetime) or value.tzinfo is not UTC:
            raise InvalidModelError(f"{where}: expected a timezone-aware datetime")
        return
    if is_dataclass(hint):
        if not isinstance(value, hint):
            raise InvalidModelError(f"{where}: expected {hint.__name__}, got {value!r}")
        return
    raise TypeError(f"{where}: unsupported type hint {hint!r}")


def _normalise(hint: Any, value: Any) -> Any:
    hint = strip_optional(hint) if value is not None else hint
    origin = get_origin(hint)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if origin is Mapping and isinstance(value, Mapping):
        return types.MappingProxyType({key: value[key] for key in sorted(value)})
    if origin is Observed and isinstance(value, Observed):
        inner = get_args(hint)[0]
        if inner is float and type(value.value) is int:
            return replace(value, value=float(value.value))
        return value
    if hint is float and type(value) is int:
        return float(value)
    if hint is Any:
        return freeze_json(value, "value")
    if hint is datetime and isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC)
    return value


def _check_metadata(f: Field[Any], value: Any, where: str) -> None:
    unit = f.metadata.get("unit", _MISSING)
    if unit is not _MISSING:
        items = value.values() if isinstance(value, Mapping) else [value]
        for item in items:
            if isinstance(item, Observed) and item.unit != unit:
                raise InvalidModelError(f"{where}: unit must be {unit!r}, got {item.unit!r}")
    schema: dict[str, Any] = f.metadata.get("schema", {})
    if value is None or not schema:
        return
    if "const" in schema and value != schema["const"]:
        raise InvalidModelError(f"{where}: must be {schema['const']!r}, got {value!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise InvalidModelError(f"{where}: must not be empty")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise InvalidModelError(f"{where}: {value!r} doesn't match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise InvalidModelError(f"{where}: must be >= {schema['minimum']}, got {value!r}")
        if "maximum" in schema and value > schema["maximum"]:
            raise InvalidModelError(f"{where}: must be <= {schema['maximum']}, got {value!r}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise InvalidModelError(
                f"{where}: must be > {schema['exclusiveMinimum']}, got {value!r}"
            )
