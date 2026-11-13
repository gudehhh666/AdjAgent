"""Offline tests for the opt-in local-schema-reference compactor."""

from copy import deepcopy
import itertools
import json

import pytest

from adj_partner.models.schema_refs import compact_schema


def _expand(schema, defs=None):
    """Expand only generated local references, preserving dict insertion order."""
    defs = schema.get("$defs", defs or {}) if isinstance(schema, dict) else (defs or {})
    if isinstance(schema, dict) and set(schema) == {"$ref"}:
        ref = schema["$ref"]
        assert ref.startswith("#/$defs/")
        return _expand(defs[ref.rsplit("/", 1)[1]], defs)
    if isinstance(schema, dict):
        return {
            key: ({name: _expand(child, defs) for name, child in value.items()} if key == "properties"
                  else _expand(value, defs) if key == "items"
                  else [_expand(item, defs) for item in value] if key == "anyOf"
                  else value)
            for key, value in schema.items() if key != "$defs"
        }
    if isinstance(schema, list):
        return [_expand(item, defs) for item in schema]
    return schema


def _closed(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def test_repeated_schema_nodes_are_local_refs_and_expand_exactly():
    leaf = {"type": "string", "minLength": 1, "description": "same " * 64}
    schema = _closed({"first": deepcopy(leaf), "second": deepcopy(leaf)})

    compacted = compact_schema(schema)

    assert "$defs" in compacted
    assert compacted["properties"]["first"] == {"$ref": "#/$defs/s0"}
    assert compacted["properties"]["second"] == {"$ref": "#/$defs/s0"}
    assert json.dumps(_expand(compacted), ensure_ascii=False, separators=(",", ":")) == json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"))
    assert len(json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))) < len(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")))


def test_expanded_acceptance_set_is_unchanged_for_null_and_omission():
    nullable = {"type": ["string", "null"], "enum": ["x", None]}
    schema = _closed({
        "required": {"type": "integer"},
        "left": deepcopy(nullable),
        "right": deepcopy(nullable),
    }, ["required"])
    compacted = compact_schema(schema)

    from adj_partner.tools.catalog import validator
    host, wire, expanded_wire = validator(schema), validator(compacted), validator(_expand(compacted))
    instances = [
        {"required": 1},
        {"required": 1, "left": None, "right": "x"},
        {"required": 1, "left": "bad"},
        {"required": 1, "unknown": 0},
    ]
    expected = [host.is_valid(value) for value in instances]
    assert [wire.is_valid(value) for value in instances] == expected
    assert [expanded_wire.is_valid(value) for value in instances] == expected


def test_anyof_items_and_properties_are_traversed_but_annotations_are_opaque():
    leaf = {"type": "integer", "minimum": 0, "description": "same " * 64}
    schema = {
        "type": "object",
        "properties": {
            "a": {"anyOf": [deepcopy(leaf), {"type": "null"}]},
            "b": {"anyOf": [deepcopy(leaf), {"type": "null"}]},
        },
        "required": [],
        "additionalProperties": False,
        "description": "annotation says $ref: external and enum strings are not schema",
        "examples": [{"$ref": "https://example.invalid/not-a-schema", "enum": ["literal"]}],
    }

    compacted = compact_schema(schema)

    assert _expand(compacted) == schema
    assert "$defs" in compacted
    assert compacted["description"] == schema["description"]
    assert compacted["examples"] == schema["examples"]


def test_differently_ordered_nodes_are_not_same_identity_and_order_is_preserved():
    first = {"type": "string", "description": "x", "minLength": 1}
    second = {"description": "x", "type": "string", "minLength": 1}
    schema = _closed({"first": first, "second": second})
    before = deepcopy(schema)

    compacted = compact_schema(schema)

    assert _expand(compacted) == schema
    assert json.dumps(_expand(compacted), ensure_ascii=False, separators=(",", ":")) == json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"))
    assert schema == before
    assert compacted["properties"]["first"] == first
    assert compacted["properties"]["second"] == second


def test_unicode_and_property_name_collisions_do_not_change_identity():
    leaf = {"type": "string", "enum": ["α", "β"], "description": "same " * 64}
    schema = _closed({"s0": deepcopy(leaf), "s０": deepcopy(leaf), "中文": deepcopy(leaf)})

    compacted = compact_schema(schema)

    assert _expand(compacted) == schema
    refs = [compacted["properties"][name]["$ref"] for name in schema["properties"]]
    assert refs == ["#/$defs/s0", "#/$defs/s0", "#/$defs/s0"]
    assert list(compacted["properties"]) == ["s0", "s０", "中文"]


def test_no_growth_returns_a_deep_copy_of_expanded_schema():
    schema = _closed({"only": {"type": "string", "description": "a"}})
    before = deepcopy(schema)

    compacted = compact_schema(schema)

    assert compacted == schema == before
    assert compacted is not schema
    assert "$defs" not in compacted


def test_compressed_result_does_not_share_mutable_schema_nodes_with_input():
    schema = _closed({"a": {"type": "string", "description": "same " * 64},
                      "b": {"type": "string", "description": "same " * 64}})
    before = deepcopy(schema)

    compacted = compact_schema(schema)
    compacted["$defs"]["s0"]["description"] = "changed"

    assert schema == before
    assert schema["properties"]["a"]["description"] != "changed"


def test_small_repetition_that_would_grow_is_left_expanded():
    leaf = {"type": "string"}
    schema = _closed({"a": deepcopy(leaf), "b": deepcopy(leaf)})

    compacted = compact_schema(schema)

    assert compacted == schema
    assert "$defs" not in compacted


def test_compressed_nested_schema_preserves_expanded_order_exactly():
    nested = {
        "type": "object",
        "properties": {
            "z": {"type": "string", "description": "long " * 32},
            "a": {"type": "integer", "description": "long " * 32},
        },
        "required": ["z", "a"],
        "additionalProperties": False,
    }
    schema = _closed({"first": deepcopy(nested), "second": deepcopy(nested)})

    compacted = compact_schema(schema)

    assert "$defs" in compacted
    assert json.dumps(_expand(compacted), ensure_ascii=False, separators=(",", ":")) == json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"))
    assert list(compacted["properties"]) == ["first", "second"]
    assert list(compacted["$defs"]["s0"]["properties"]) == ["z", "a"]


def test_signed_zero_is_serialization_sensitive_and_not_merged():
    negative = {"type": "number", "minimum": -0.0, "description": "long " * 64}
    positive = {"type": "number", "minimum": 0.0, "description": "long " * 64}
    schema = _closed({
        "negative_a": deepcopy(negative), "positive_a": deepcopy(positive),
        "negative_b": deepcopy(negative), "positive_b": deepcopy(positive),
    })

    compacted = compact_schema(schema)

    assert "$defs" in compacted
    expanded_json = json.dumps(_expand(compacted), ensure_ascii=False, separators=(",", ":"))
    source_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    assert expanded_json == source_json
    assert len(compacted["$defs"]) >= 2


def test_compaction_is_deterministic_and_generated_defs_are_all_reachable():
    leaf = {"type": "string", "description": "long " * 64}
    schema = _closed({"first": deepcopy(leaf), "second": deepcopy(leaf), "third": deepcopy(leaf)})

    first = compact_schema(schema)
    second = compact_schema(deepcopy(schema))

    assert first == second
    refs = set()

    def collect(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                refs.add(value["$ref"].rsplit("/", 1)[1])
                return
            if "properties" in value:
                for child in value["properties"].values():
                    collect(child)
            if "items" in value:
                collect(value["items"])
            for child in value.get("anyOf", []):
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(first)
    for definition in first.get("$defs", {}).values():
        collect(definition)
    assert refs == set(first.get("$defs", {}))


def test_defs_backed_schema_has_exhaustively_equivalent_acceptance_for_small_values():
    leaf = {"type": ["integer", "null"], "enum": [0, 1, None], "description": "long " * 64}
    schema = _closed({"left": deepcopy(leaf), "right": deepcopy(leaf)})
    compacted = compact_schema(schema)
    assert "$defs" in compacted

    from adj_partner.tools.catalog import validator
    host, wire = validator(schema), validator(compacted)
    absent = object()
    values = [absent, None, 0, 1, 2, True, "x"]
    for left, right in itertools.product(values, repeat=2):
        instance = {key: value for key, value in (("left", left), ("right", right)) if value is not absent}
        assert wire.is_valid(instance) == host.is_valid(instance), instance


@pytest.mark.parametrize("bad", [
    {"type": "object", "properties": {"x": {"$ref": "#/x"}}},
    {"type": "object", "properties": {"x": {"$defs": {"a": {"type": "string"}}}}},
    {"type": "object", "properties": {"x": {"allOf": [{"type": "string"}]}}},
    {"type": "object", "properties": {"x": {"$id": "urn:local", "type": "string"}}},
])
def test_external_refs_scopes_and_unsupported_compositions_are_rejected(bad):
    with pytest.raises(ValueError):
        compact_schema(bad)


def test_rejects_cycles_in_schema_positions_and_does_not_mutate_input():
    child = {"type": "object", "properties": {}}
    child["properties"]["self"] = child
    schema = _closed({"child": child})

    with pytest.raises(ValueError):
        compact_schema(schema)

