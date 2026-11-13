"""Public-code acceptance is a separate, structural task contract."""

import json

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.research.acceptance import (
    REQUIRE_BOUND_EXECUTION_WITNESS,
    RESPONSE_SCHEMA,
    requirements,
    research_acceptance,
)
from adj_partner.runtime.context import ContextView
from adj_partner.runtime.journal import RunJournal
from adj_partner.storage import Store


SCHEMA = {
    "type": "object",
    "properties": {
        "generated_code": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": ["generated_code"],
    "additionalProperties": False,
}


def contract(*, enforce=True):
    return {
        "manifest_artifact_id": "artifact_public_manifest_01",
        "manifest_sha256": "a" * 64,
        "checker_version": "public-code-check-v1",
        "max_attempts": 2,
        "enforce": enforce,
    }


def acceptance(*, enforce=True):
    return {RESPONSE_SCHEMA: SCHEMA, "public_code_validation": contract(enforce=enforce)}


@pytest.mark.parametrize("enforce", [True, False])
def test_public_code_validation_is_standalone_and_does_not_require_report(enforce):
    accepted = requirements(acceptance(enforce=enforce))

    assert accepted["require_report"] is False
    assert accepted["public_code_validation"] == contract(enforce=enforce)
    assert not research_acceptance(acceptance(enforce=enforce))


@pytest.mark.parametrize("field,value", [
    ("manifest_artifact_id", ""),
    ("manifest_artifact_id", "x" * 129),
    ("manifest_artifact_id", "../artifact"),
    ("manifest_sha256", "A" * 64),
    ("manifest_sha256", "a" * 63),
    ("checker_version", "public-code-check-v2"),
    ("max_attempts", True),
    ("max_attempts", 1),
    ("max_attempts", 2.0),
    ("enforce", 1),
    ("enforce", "true"),
])
def test_public_code_contract_rejects_invalid_field_values(field, value):
    bad = contract()
    bad[field] = value

    with pytest.raises(ValueError, match="public_code_validation"):
        requirements({RESPONSE_SCHEMA: SCHEMA, "public_code_validation": bad})


def test_public_code_contract_rejects_unknown_fields_and_incompatible_modes():
    extra = contract()
    extra["checker_module"] = "model.supplied"
    with pytest.raises(ValueError, match="public_code_validation"):
        requirements({RESPONSE_SCHEMA: SCHEMA, "public_code_validation": extra})

    with pytest.raises(ValueError, match="cannot be combined"):
        requirements({RESPONSE_SCHEMA: SCHEMA,
                      "public_code_validation": contract(),
                      REQUIRE_BOUND_EXECUTION_WITNESS: True})


def test_public_code_contract_requires_schema_with_generated_code_mapping():
    with pytest.raises(ValueError, match="response_json_schema"):
        requirements({"public_code_validation": contract()})
    with pytest.raises(ValueError, match="generated_code"):
        requirements({RESPONSE_SCHEMA: {"type": "object", "properties": {},
                                        "additionalProperties": True},
                      "public_code_validation": contract()})
    with pytest.raises(ValueError, match="generated_code"):
        requirements({RESPONSE_SCHEMA: {"type": "object", "properties": {
            "generated_code": {"type": "array"}}, "required": ["generated_code"]},
            "public_code_validation": contract()})
    with pytest.raises(ValueError, match="generated_code"):
        requirements({RESPONSE_SCHEMA: {"type": "object", "properties": {
            "generated_code": {"type": "object"}}},
            "public_code_validation": contract()})


def test_existing_schema_and_execution_witness_contracts_keep_their_behavior():
    schema_only = requirements({RESPONSE_SCHEMA: SCHEMA})
    witness = requirements({RESPONSE_SCHEMA: SCHEMA, REQUIRE_BOUND_EXECUTION_WITNESS: True})

    assert schema_only["require_report"] is False
    assert witness["require_report"] is False
    assert not research_acceptance({RESPONSE_SCHEMA: SCHEMA})
    assert not research_acceptance({RESPONSE_SCHEMA: SCHEMA,
                                    REQUIRE_BOUND_EXECUTION_WITNESS: True})
    assert research_acceptance({"require_evidence": True})


def _acceptance_from_system_message(messages):
    system = next(message["content"] for message in messages
                  if message["role"] == "system" and "Host task policy:" in message["content"])
    policy = json.loads(system.split("Host task policy:\n", 1)[1])
    return policy["acceptance"]


def test_context_hints_public_check_without_disclosing_enforce(tmp_path):
    store = Store(tmp_path / "store")
    budget = Budget().to_dict()
    sid = store.create_session(SessionConfig(budget=Budget()).to_dict())["id"]
    run = RunJournal(store).start(sid, "Return the generated program", acceptance(enforce=True), budget)["id"]
    snapshot = {"revision": 0, "tools": [], "definition_hashes": {}}
    view = ContextView(store)
    first = view.build(sid, run, snapshot, max_input_tokens=30000)

    public_acceptance = _acceptance_from_system_message(first["messages"])
    assert "enforce" not in public_acceptance["public_code_validation"]
    hint = "\n".join(message["content"] for message in first["messages"]
                     if message["role"] == "system")
    assert "public_code_check" in hint
    assert "generated_code" in hint
    assert "at most 2" in hint
    assert "revise once" in hint
    assert "hidden oracle" in hint
    assert "enforce" not in hint

    task = store.get_task(store.get_run(run)["task_id"])
    changed = dict(task["acceptance"])
    changed["public_code_validation"] = contract(enforce=False)
    store.update_task(task["id"], {"acceptance": changed}, expected_revision=task["revision"])
    second = view.build(sid, run, snapshot, max_input_tokens=30000)

    assert first["messages"] == second["messages"]
    store.close()
