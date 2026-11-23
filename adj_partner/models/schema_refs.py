"""Pure, opt-in compaction of an already validated expanded JSON schema.

The input is deliberately treated as a reference-free schema tree.  Only the
schema-bearing positions used by the native compiler are traversed; annotations
and enum values remain opaque JSON values.  Generated references are local to
the returned root and can therefore be expanded without changing validation.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any


_REFERENCE_OR_SCOPE_KEYS = {
    "$ref", "$dynamicRef", "$recursiveRef", "$id", "$anchor",
    "$dynamicAnchor", "$defs", "$dynamicDefs", "$schema", "$vocabulary",
}
_UNSUPPORTED_COMPOSITION_KEYS = {
    "allOf", "oneOf", "not", "if", "then", "else", "dependentSchemas",
    "prefixItems", "contains", "propertyNames", "unevaluatedProperties",
    "unevaluatedItems",
}


def _freeze_json(value: Any, active: set[int] | None = None):
    """Make an insertion-order-sensitive, hashable JSON identity."""
    active = set() if active is None else active
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise ValueError("schema contains a cycle")
        active.add(identity)
        try:
            pairs = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("schema object keys must be strings")
                pairs.append((key, _freeze_json(item, active)))
            return ("object", tuple(pairs))
        finally:
            active.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise ValueError("schema contains a cycle")
        active.add(identity)
        try:
            return ("array", tuple(_freeze_json(item, active) for item in value))
        finally:
            active.remove(identity)
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("schema values must be finite JSON")
        # Use the exact JSON token so -0.0 and 0.0 remain distinct.  Python
        # numeric equality would otherwise merge two schemas whose expanded
        # serialized trees differ.
        token = json.dumps(value, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":"))
        return ("number", token)
    if isinstance(value, str):
        return ("string", value)
    raise ValueError("schema values must be JSON-compatible")


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("schema must be finite JSON") from exc


def _collect_nodes(node: dict, counts: dict, representatives: dict,
                   order: list, *, active_nodes: set[int] | None = None) -> None:
    if not isinstance(node, dict):
        raise ValueError("schema nodes must be objects")
    active_nodes = set() if active_nodes is None else active_nodes
    object_id = id(node)
    if object_id in active_nodes:
        raise ValueError("schema contains a cycle in a schema position")
    active_nodes.add(object_id)
    try:
        forbidden = (_REFERENCE_OR_SCOPE_KEYS | _UNSUPPORTED_COMPOSITION_KEYS) & set(node)
        if forbidden:
            raise ValueError(f"unsupported schema reference, scope, or composition: {sorted(forbidden)[0]}")
        identity = _freeze_json(node)
        if identity not in counts:
            counts[identity] = 0
            representatives[identity] = node
            order.append(identity)
        counts[identity] += 1

        if "properties" in node:
            properties = node["properties"]
            if not isinstance(properties, dict):
                raise ValueError("schema properties must be an object")
            for name, child in properties.items():
                if not isinstance(name, str):
                    raise ValueError("schema property names must be strings")
                _collect_nodes(child, counts, representatives, order,
                               active_nodes=active_nodes)
        if "items" in node:
            if not isinstance(node["items"], dict):
                raise ValueError("schema items must be one schema object")
            _collect_nodes(node["items"], counts, representatives, order,
                           active_nodes=active_nodes)
        if "anyOf" in node:
            branches = node["anyOf"]
            if not isinstance(branches, list) or any(not isinstance(item, dict) for item in branches):
                raise ValueError("schema anyOf must contain schema objects")
            for child in branches:
                _collect_nodes(child, counts, representatives, order,
                               active_nodes=active_nodes)
    finally:
        active_nodes.remove(object_id)


def compact_schema(schema: dict) -> dict:
    """Return a smaller expanded-schema equivalent using generated local refs.

    The caller must provide the already compiled, reference-free schema.  The
    function never mutates it, preserves insertion order in identities and in
    the expanded tree, and returns a deep copy of the original when references
    do not reduce serialized bytes.
    """
    if not isinstance(schema, dict):
        raise ValueError("schema root must be an object")
    original_bytes = _json_bytes(schema)
    counts, representatives, order = {}, {}, []
    _collect_nodes(schema, counts, representatives, order)
    root_identity = _freeze_json(schema)
    interned = [identity for identity in order
                if identity != root_identity and counts[identity] > 1]
    if not interned:
        return deepcopy(schema)
    names = {identity: f"s{index}" for index, identity in enumerate(interned)}

    def transform(node: dict, *, keep_identity=None):
        identity = _freeze_json(node)
        name = names.get(identity)
        if name is not None and identity != keep_identity:
            return {"$ref": f"#/$defs/{name}"}
        result = {}
        for key, value in node.items():
            if key == "properties":
                result[key] = {name: transform(child) for name, child in value.items()}
            elif key == "items":
                result[key] = transform(value)
            elif key == "anyOf":
                result[key] = [transform(child) for child in value]
            else:
                result[key] = deepcopy(value)
        return result

    compacted = transform(schema, keep_identity=root_identity)
    compacted["$defs"] = {
        names[identity]: transform(representatives[identity], keep_identity=identity)
        for identity in interned
    }
    if len(_json_bytes(compacted)) < len(original_bytes):
        return compacted
    return deepcopy(schema)

