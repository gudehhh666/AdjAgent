"""Production pure codec for the opt-in v0.9 tagged-presence layout.

The public API accepts the original host schema. ``compile_node`` returns the
closed wire schema, while ``encode`` and ``decode`` strictly validate both
sides of the reversible transformation. Provider dispatch and layout
selection remain owned by the adapter.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from adj_partner.tools.catalog import validate_schema, validator


def _legacy_helpers():
    """Load the existing bounded compiler lazily to avoid adapter cycles."""
    from adj_partner.models.codex_app_server import (
        SCHEMA_LIMITS,
        _check_schema_limits,
        _strict_node,
    )
    return SCHEMA_LIMITS, _check_schema_limits, _strict_node


_SCALAR_TYPES = {
    type(None): "null",
    bool: "boolean",
    str: "string",
    int: "number",
    float: "number",
}


def _finite_json(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("value must be finite JSON") from exc


def _admit(schema: dict) -> dict:
    """Apply the existing strict compiler's bounded, reference-free admission."""
    if not isinstance(schema, dict):
        raise ValueError("schema root must be an object")
    try:
        source = deepcopy(schema)
        validate_schema(source)
        _, _, strict_node = _legacy_helpers()
        expanded = strict_node(source, depth=4)
    except (RecursionError, TypeError, KeyError, StopIteration) as exc:
        raise ValueError("schema is not a valid bounded JSON schema") from exc
    _, check_schema_limits, _ = _legacy_helpers()
    check_schema_limits(expanded)
    return source


def _normalized(node: dict) -> dict:
    """Add the scalar type inferred by the existing compiler for enum-only nodes."""
    if "type" in node or "enum" not in node:
        return deepcopy(node)
    types = {_SCALAR_TYPES.get(type(value)) for value in node["enum"]}
    if not types or None in types:
        # _strict_node has already rejected mixed or unrecognized enum domains;
        # keep a defensive error here so this helper cannot silently broaden it.
        raise ValueError("enum-only schema has no single scalar type")
    result = deepcopy(node)
    if "null" in types:
        if len(types) == 1:
            # Preserve the existing compiler's enum-only null inference.
            result["type"] = "null"
        elif len(types) == 2:
            result["type"] = [next(item for item in types if item != "null"), "null"]
        else:
            raise ValueError("enum-only schema has more than one non-null type")
    elif len(types) == 1:
        result["type"] = next(iter(types))
    else:
        raise ValueError("enum-only schema has more than one scalar type")
    return result


def _kind(node: dict) -> str:
    normalized = _normalized(node)
    kind = normalized.get("type")
    if isinstance(kind, list):
        return next(item for item in kind if item != "null")
    return kind


def _wire_node(node: dict) -> dict:
    node = _normalized(node)
    kind = _kind(node)
    if kind == "object":
        properties = node.get("properties", {})
        required = list(node.get("required", []))
        result = {key: deepcopy(value) for key, value in node.items()
                  if key not in {"properties", "required", "additionalProperties"}}
        result["type"] = deepcopy(node.get("type", "object"))
        result["properties"] = {}
        required_set = set(required)
        for name, child in properties.items():
            if name in required_set:
                result["properties"][name] = _wire_node(child)
            else:
                result["properties"][name] = {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "present": {"type": "boolean", "enum": [False]},
                            },
                            "required": ["present"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "present": {"type": "boolean", "enum": [True]},
                                "value": _wire_node(child),
                            },
                            "required": ["present", "value"],
                            "additionalProperties": False,
                        },
                    ]
                }
        # Every original property is present on the wire. Optionality moves
        # inside the property's closed wrapper, so omission is represented by
        # {"present": false} rather than by omitting the property itself.
        result["required"] = list(properties)
        result["additionalProperties"] = False
        return result
    if kind == "array":
        result = _normalized(node)
        # As in the expanded layout, uniqueness is checked by host validation of the original schema.
        result.pop("uniqueItems", None)
        result["items"] = _wire_node(node["items"])
        return result
    return _normalized(node)


def _check_wire_depth(node: dict, depth: int = 4) -> None:
    """Apply the native envelope's parameter-schema depth offset to wire nodes."""
    if not isinstance(node, dict):
        raise ValueError("generated wire schema node must be an object")
    kind = node.get("type")
    container = (kind in ("object", "array") or
                 isinstance(kind, list) and any(item in ("object", "array") for item in kind))
    schema_limits, _, _ = _legacy_helpers()
    if container and depth > schema_limits["depth"]:
        raise ValueError("generated wire schema nesting depth limit exceeded")
    for child in node.get("properties", {}).values():
        _check_wire_depth(child, depth + 1)
    if "items" in node:
        _check_wire_depth(node["items"], depth + 1)
    for child in node.get("anyOf", []):
        _check_wire_depth(child, depth)


def compile_node(schema: dict) -> dict:
    """Compile an admitted original node to the closed presence-wrapper schema."""
    source = _admit(schema)
    wire = _wire_node(source)
    # Wrappers add branches/properties/depth; apply the same bounded limits to
    # the generated wire tree rather than assuming the original admission is
    # sufficient.
    _, check_schema_limits, _ = _legacy_helpers()
    check_schema_limits(wire)
    _check_wire_depth(wire)
    return wire


def _encode_node(value: Any, schema: dict) -> Any:
    kind = _kind(schema)
    if value is None:
        return None
    if kind == "object":
        result = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name, child in properties.items():
            if name in required:
                result[name] = _encode_node(value[name], child)
            elif name in value:
                result[name] = {"present": True, "value": _encode_node(value[name], child)}
            else:
                result[name] = {"present": False}
        return result
    if kind == "array":
        return [_encode_node(item, schema["items"]) for item in value]
    return deepcopy(value)


def _decode_node(wire: Any, schema: dict) -> Any:
    kind = _kind(schema)
    if wire is None:
        return None
    if kind == "object":
        result = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name, child in properties.items():
            if name in required:
                result[name] = _decode_node(wire[name], child)
            else:
                wrapper = wire[name]
                if wrapper["present"]:
                    result[name] = _decode_node(wrapper["value"], child)
        return result
    if kind == "array":
        return [_decode_node(item, schema["items"]) for item in wire]
    return deepcopy(wire)


def encode(value: Any, schema: dict) -> Any:
    """Validate a host value and encode optional properties as presence wrappers."""
    source = _admit(schema)
    wire_schema = compile_node(source)
    _finite_json(value)
    validator(source).validate(value)
    wire = _encode_node(value, source)
    validator(wire_schema).validate(wire)
    return wire


def decode(wire: Any, schema: dict) -> Any:
    """Validate a wire value, remove wrappers, and return the host value."""
    source = _admit(schema)
    wire_schema = compile_node(source)
    _finite_json(wire)
    validator(wire_schema).validate(wire)
    value = _decode_node(wire, source)
    validator(source).validate(value)
    return value
