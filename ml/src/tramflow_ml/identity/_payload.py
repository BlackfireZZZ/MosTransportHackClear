"""Typed extraction from untrusted dict payloads; loaders fail fast with the key path."""

import math
from collections.abc import Mapping, Sequence

from tramflow_ml.identity.types import IdentityError


def require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise IdentityError(f"{path} must be an object with string keys")
    return value


def optional_mapping(payload: Mapping[str, object], key: str, path: str) -> Mapping[str, object]:
    if key not in payload:
        return {}
    return require_mapping(payload[key], f"{path}.{key}")


def require_sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise IdentityError(f"{path} must be an array")
    return value


def optional_sequence(payload: Mapping[str, object], key: str, path: str) -> Sequence[object]:
    if key not in payload:
        return ()
    return require_sequence(payload[key], f"{path}.{key}")


def require_str(payload: Mapping[str, object], key: str, path: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise IdentityError(f"{path}.{key} must be a non-empty string")
    return value


def optional_str(payload: Mapping[str, object], key: str, path: str) -> str | None:
    if payload.get(key) is None:
        return None
    return require_str(payload, key, path)


def str_mapping(payload: Mapping[str, object], key: str, path: str) -> dict[str, str]:
    table = optional_mapping(payload, key, path)
    for source, target in table.items():
        if not source or not isinstance(target, str) or not target:
            raise IdentityError(f"{path}.{key}[{source!r}] must map to a non-empty string")
    return {source: target for source, target in table.items() if isinstance(target, str)}


def require_float(payload: Mapping[str, object], key: str, path: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise IdentityError(f"{path}.{key} must be a finite number")
    return float(value)


def optional_int(payload: Mapping[str, object], key: str, path: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityError(f"{path}.{key} must be an integer")
    return value
