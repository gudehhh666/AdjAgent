"""Synthetic tests for the production v0.9 presence codec."""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ITERATION_ROOT = Path(__file__).resolve().parents[2]
if str(ITERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(ITERATION_ROOT))

import pytest
from jsonschema import ValidationError

from adj_partner.tools.catalog import validator
from adj_partner.models.schema_presence import compile_node, decode, encode


OPTIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "required_text": {"type": "string", "minLength": 1},
        "flag": {"type": "boolean"},
        "count": {"type": "integer", "minimum": 0},
        "maybe": {"type": ["string", "null"]},
        "nested": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "values": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["required_text"],
    "additionalProperties": False,
}


def test_compile_node_wraps_optional_keys_and_keeps_required_direct():
    wire = compile_node(OPTIONAL_SCHEMA)
    assert list(wire["properties"]) == list(OPTIONAL_SCHEMA["properties"])
    assert wire["required"] == list(OPTIONAL_SCHEMA["properties"])
    assert wire["properties"]["required_text"] == OPTIONAL_SCHEMA["properties"]["required_text"]
    optional = wire["properties"]["flag"]
    assert optional["anyOf"][0]["properties"]["present"]["enum"] == [False]
    assert optional["anyOf"][1]["properties"]["present"]["enum"] == [True]
    assert optional["anyOf"][1]["required"] == ["present", "value"]
    assert optional["anyOf"][1]["properties"]["value"] == {"type": "boolean"}


def test_roundtrip_distinguishes_omission_null_false_zero_and_nested_values():
    for flag, count, maybe, nested, values in itertools.product(
        (None, False, True), (None, 0, 4), (None, "text"), (None, {"name": "N", "active": False}),
        (None, [0.0, -0.0, 2.5]),
    ):
        value = {"required_text": "code ✓"}
        if flag is not None:
            value["flag"] = flag
        if count is not None:
            value["count"] = count
        if maybe is not None:
            value["maybe"] = maybe
        if nested is not None:
            value["nested"] = nested
        if values is not None:
            value["values"] = values
        wire = encode(value, OPTIONAL_SCHEMA)
        assert decode(wire, OPTIONAL_SCHEMA) == value
        assert wire["flag"]["present"] is ("flag" in value)
        assert wire["count"]["present"] is ("count" in value)
    explicit_null = encode({"required_text": "x", "maybe": None}, OPTIONAL_SCHEMA)
    assert explicit_null["maybe"] == {"present": True, "value": None}
    omitted = encode({"required_text": "x"}, OPTIONAL_SCHEMA)
    assert omitted["maybe"] == {"present": False}


def test_original_and_wire_validators_accept_roundtrip_and_reject_invalid():
    value = {"required_text": "x", "count": 0, "flag": False}
    wire = encode(value, OPTIONAL_SCHEMA)
    validator(OPTIONAL_SCHEMA).validate(value)
    validator(compile_node(OPTIONAL_SCHEMA)).validate(wire)
    with pytest.raises(ValidationError):
        encode({"required_text": ""}, OPTIONAL_SCHEMA)
    with pytest.raises(ValidationError):
        encode({"required_text": "x", "maybe": None}, {
            **OPTIONAL_SCHEMA,
            "properties": {**OPTIONAL_SCHEMA["properties"], "maybe": {"type": "string"}},
        })


@pytest.mark.parametrize("bad_flag", [
    {"required_text": "x", "flag": {"present": True}},
    {"required_text": "x", "flag": {"present": False, "value": False}},
    {"required_text": "x", "flag": {"present": "yes"}},
    {"required_text": "x", "flag": {"present": 0}},
    {"required_text": "x", "flag": {"present": 1, "value": False}},
    {"required_text": "x", "flag": {"present": True, "value": False, "extra": 1}},
])
def test_malformed_presence_wrappers_are_rejected(bad_flag):
    wire = encode({"required_text": "x"}, OPTIONAL_SCHEMA)
    wire["flag"] = bad_flag["flag"]
    with pytest.raises((ValidationError, ValueError)):
        decode(wire, OPTIONAL_SCHEMA)


def test_missing_optional_wrapper_is_rejected():
    wire = encode({"required_text": "x"}, OPTIONAL_SCHEMA)
    del wire["flag"]
    with pytest.raises((ValidationError, ValueError)):
        decode(wire, OPTIONAL_SCHEMA)


def test_nested_optional_values_and_wrapper_name_collisions_are_data_not_control():
    schema = {
        "type": "object",
        "properties": {
            "present": {"type": "string"},
            "value": {"type": "number"},
            "π": {"type": "object", "properties": {
                "present": {"type": "boolean"},
                "value": {"type": "string"},
            }, "required": [], "additionalProperties": False},
        },
        "required": [],
        "additionalProperties": False,
    }
    value = {"present": "ordinary", "value": -0.0,
             "π": {"present": False, "value": "code\n✓"}}
    wire = encode(value, schema)
    assert wire["present"] == {"present": True, "value": "ordinary"}
    assert wire["value"] == {"present": True, "value": -0.0}
    assert decode(wire, schema) == value
    assert math.copysign(1.0, decode(wire, schema)["value"]) < 0


def test_deterministic_and_non_mutating_with_enum_and_code_string():
    schema = {
        "type": "object", "description": "keep annotation",
        "properties": {"mode": {"enum": ["α", "β"]}, "code": {"type": "string"}},
        "required": [], "additionalProperties": False,
    }
    before = deepcopy(schema)
    first = compile_node(schema)
    second = compile_node(schema)
    assert first == second and schema == before
    assert first["description"] == "keep annotation"
    assert decode(encode({"mode": "α", "code": "print(0)"}, schema), schema) == {
        "mode": "α", "code": "print(0)"
    }


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {}, "required": [], "additionalProperties": True},
    {"type": "object", "properties": {}, "required": [], "additionalProperties": False, "default": {}},
    {"type": "object", "properties": {"x": {"$ref": "#/defs/x"}}, "required": [], "additionalProperties": False},
])
def test_unsupported_scope_defaults_open_objects_and_keywords_are_rejected(schema):
    with pytest.raises((ValueError, TypeError)):
        compile_node(schema)


def test_unique_items_is_dropped_from_wire_and_left_to_host_validation():
    schema = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
              "required": ["ids"], "additionalProperties": False}
    wire = compile_node(schema)
    assert "uniqueItems" not in wire["properties"]["ids"]
    assert validator(wire).is_valid({"ids": ["a", "a"]})
    with pytest.raises(Exception, match="non-unique"):
        decode({"ids": ["a", "a"]}, schema)


def test_generated_wire_schema_is_independently_closed_and_optional_subset_is_exhaustive():
    schema = {
        "type": "object",
        "properties": {f"x{i}": {"type": "integer"} for i in range(3)},
        "required": [], "additionalProperties": False,
    }
    wire = compile_node(schema)
    wire_validator = validator(wire)
    for mask in range(8):
        value = {f"x{i}": i for i in range(3) if mask & (1 << i)}
        encoded = encode(value, schema)
        wire_validator.validate(encoded)
        assert decode(encoded, schema) == value
    with pytest.raises(ValidationError):
        wire_validator.validate({"x0": {"present": True, "value": 1, "extra": 2},
                                 "x1": {"present": False}, "x2": {"present": False}})


def test_nested_nullable_object_and_array_roundtrip():
    schema = {
        "type": "object",
        "properties": {
            "obj": {"type": ["object", "null"], "properties": {
                "label": {"type": "string"},
            }, "required": [], "additionalProperties": False},
            "rows": {"type": ["array", "null"], "items": {
                "type": "object", "properties": {"n": {"type": "integer"}},
                "required": ["n"], "additionalProperties": False,
            }},
        },
        "required": [], "additionalProperties": False,
    }
    for value in (
        {"obj": None, "rows": None},
        {"obj": {}, "rows": []},
        {"obj": {"label": "✓"}, "rows": [{"n": 0}, {"n": 2}]},
    ):
        encoded = encode(value, schema)
        validator(compile_node(schema)).validate(encoded)
        assert decode(encoded, schema) == value


def test_original_optional_branch_limit_is_admitted_before_codec_generation():
    schema = {
        "type": "object",
        "properties": {f"x{i}": {"type": "integer"} for i in range(9)},
        "required": [], "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="branch"):
        compile_node(schema)


def test_generated_wire_depth_uses_native_envelope_offset():
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    node = schema
    for index in range(6):
        child = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        node["properties"][f"a{index}"] = child
        node["required"].append(f"a{index}")
        node = child
    node["properties"]["optional_leaf"] = {"type": "string"}
    with pytest.raises(ValueError, match="depth"):
        compile_node(schema)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_are_rejected_before_jsonschema_number_validation(bad):
    schema = {"type": "object", "properties": {"x": {"type": "number"}},
              "required": [], "additionalProperties": False}
    with pytest.raises(ValueError, match="finite"):
        encode({"x": bad}, schema)


def test_recursive_schema_is_rejected_as_a_bounded_admission_error():
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    schema["properties"]["self"] = schema
    with pytest.raises(ValueError, match="cycle|depth|schema"):
        compile_node(schema)


@pytest.mark.parametrize("bad_schema", [
    {"type": "string", "minLength": -1},
    {"type": "object", "properties": {"x": {"type": "string"}},
     "required": ["x", "x"], "additionalProperties": False},
    {"type": "string", "pattern": "["},
])
def test_existing_catalog_schema_admission_rejects_invalid_constraints(bad_schema):
    with pytest.raises(ValueError):
        compile_node(bad_schema)


def test_enum_only_nullable_schema_matches_existing_strict_compiler():
    schema = {"type": "object", "properties": {"x": {"enum": ["ok", None]}},
              "required": [], "additionalProperties": False}
    wire = encode({"x": None}, schema)
    assert decode(wire, schema) == {"x": None}
    assert compile_node(schema)["properties"]["x"]["anyOf"][1]["properties"]["value"]["type"] == ["string", "null"]


@pytest.mark.parametrize("enum, expected_type, value", [
    ([None], "null", None),
    ([False], "boolean", False),
    (["ok"], "string", "ok"),
    ([1], "number", 1),
    ([1, None], ["number", "null"], None),
    ([False, None], ["boolean", "null"], None),
])
def test_enum_only_type_inference_preserves_old_allowed_domains(enum, expected_type, value):
    schema = {"type": "object", "properties": {"x": {"enum": enum}},
              "required": [], "additionalProperties": False}
    wire_schema = compile_node(schema)
    assert wire_schema["properties"]["x"]["anyOf"][1]["properties"]["value"]["type"] == expected_type
    assert decode(encode({"x": value}, schema), schema) == {"x": value}


def test_lone_surrogate_is_rejected_by_utf8_json_admission():
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": [], "additionalProperties": False}
    with pytest.raises(ValueError, match="finite|UTF-8"):
        encode({"text": "\ud800"}, schema)
