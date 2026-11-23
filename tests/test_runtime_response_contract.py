"""Opt-in structural contracts for ordinary model final responses."""

import json

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig
from adj_partner.application import AgentService
from adj_partner.models.base import ProviderFailure
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog


SCHEMA = {
    "type": "object",
    "properties": {
        "complete_pairs": {"type": "integer"},
        "mean_difference": {"type": "number"},
    },
    "required": ["complete_pairs", "mean_difference"],
    "additionalProperties": False,
}


class ScriptedModel:
    metadata = {"provider": "fixture", "model": "scripted"}
    secrets = ()

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds):
        self.requests.append({"messages": messages, "max_output_tokens": max_output_tokens,
                              "timeout_seconds": timeout_seconds})
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_engine(tmp_path, replies):
    store = Store(tmp_path / "store")
    catalog = ToolCatalog()
    sid = store.create_session(SessionConfig().to_dict())["id"]
    model = ScriptedModel(replies)
    return store, sid, model, AgentEngine(store, catalog, model)


def test_invalid_final_gets_one_format_repair_and_keeps_rejected_response(tmp_path):
    store, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='Here you go: ```json\n{"complete_pairs": 4, "mean_difference": 2.0}\n```',
                   usage={"input_tokens": 10, "output_tokens": 8}),
        ModelReply(content='{"complete_pairs": 4, "mean_difference": 2.0}',
                   usage={"input_tokens": 12, "output_tokens": 10}),
    ])

    run = engine.start(sid, "Return the counts", acceptance={"response_json_schema": SCHEMA})
    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert "original object-key or array-index path" in done["state"]["delivery"]["assurance"]
    assert len(model.requests) == 2
    assert done["state"]["response_format_repairs"] == 1
    assert done["usage"]["input_tokens"] == 22
    assert done["usage"]["output_tokens"] == 18
    assert done["state"]["rejected_final_response"].startswith("Here you go:")
    assert any(event["kind"] == "runtime.response_schema_rejected" for event in store.events(sid, run["id"]))
    repair_context = "\n".join(message["content"] for message in model.requests[1]["messages"]
                              if isinstance(message.get("content"), str))
    assert "response_json_schema" in repair_context
    assert "format-only" in repair_context
    assert "every typed scalar" in repair_context
    assert "Do not invent values" in repair_context
    assert "original object-key and array-index path" in repair_context


def test_second_invalid_final_is_explicitly_rejected(tmp_path):
    store, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs": [4], "mean_difference": 2.0}'),
        ModelReply(content='still malformed'),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == "still malformed"
    assert done["state"]["response_format_repairs"] == 1
    assert len(model.requests) == 2
    assert not any(message.get("role") == "assistant" and message.get("content") == "still malformed"
                   for message in store.messages(sid))


def test_non_json_inability_answer_is_rejected_without_format_repair(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content="I cannot determine these statistics from the available information."),
        ModelReply(content='{"complete_pairs": 0, "mean_difference": 0.0}'),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"].startswith("I cannot determine")
    assert "format_repair_unavailable" in done["state"]["response_schema_errors"][-1]
    assert len(model.requests) == 1


def test_multiple_distinct_embedded_objects_are_too_ambiguous_to_repair(tmp_path):
    text = ('The first draft was {"complete_pairs": 1, "mean_difference": 2.0}; '
            'the revised draft was {"complete_pairs": 2, "mean_difference": 3.0}.')
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content=text),
        ModelReply(content='{"complete_pairs": 2, "mean_difference": 3.0}'),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["rejected_final_response"] == text
    assert "format_repair_unavailable" in done["state"]["response_schema_errors"][-1]
    assert len(model.requests) == 1


def test_format_repair_cannot_invent_scalar_values(tmp_path):
    first = '{"status":"unable to determine"}'
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content=first),
        ModelReply(content='{"complete_pairs": 0, "mean_difference": 0.0}'),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["response_format_repairs"] == 1
    assert done["state"]["response_format_repair"]["rejected_response"] == first
    assert "repair_changed_scalar_values" in done["state"]["response_schema_errors"]
    assert len(model.requests) == 2


def test_format_repair_preserves_small_decimal_values_exactly(tmp_path):
    first = '{"complete_pairs": 4, "mean_difference": [1e-400]}'
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content=first),
        ModelReply(content='{"complete_pairs": 4, "mean_difference": 0.0}'),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert "repair_changed_scalar_values" in done["state"]["response_schema_errors"]


def test_format_repair_cannot_swap_values_between_object_fields(tmp_path):
    first = '{"complete_pairs": [4], "mean_difference": 2.0}'
    repair = '{"complete_pairs": 2, "mean_difference": 4.0}'
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first), ModelReply(content=repair)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == repair
    assert "repair_changed_scalar_values" in done["state"]["response_schema_errors"]
    assert len(model.requests) == 2


def test_format_repair_cannot_move_boolean_and_number_between_fields(tmp_path):
    schema = {
        "type": "object",
        "properties": {"left": {"type": ["number", "boolean"]},
                       "right": {"type": ["number", "boolean"]}},
        "required": ["left", "right"], "additionalProperties": False,
    }
    first = '{"left": [1], "right": true}'
    repair = '{"left": true, "right": 1}'
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first), ModelReply(content=repair)])
    run = engine.start(sid, "Return the fields", acceptance={"response_json_schema": schema})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert "repair_changed_scalar_values" in done["state"]["response_schema_errors"]
    assert len(model.requests) == 2


def test_format_repair_cannot_reorder_array_values(tmp_path):
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"type": "number"}},
                       "ready": {"type": "boolean"}},
        "required": ["values", "ready"], "additionalProperties": False,
    }
    first = '{"values": [1, 2], "ready": [true]}'
    repair = '{"values": [2, 1], "ready": true}'
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first), ModelReply(content=repair)])
    run = engine.start(sid, "Return values", acceptance={"response_json_schema": schema})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert "repair_changed_scalar_values" in done["state"]["response_schema_errors"]
    assert len(model.requests) == 2


def test_deep_valid_json_is_rejected_without_starting_format_repair(tmp_path):
    deep_array = "[" * 500 + "0" + "]" * 500
    first = '{"complete_pairs": 4, "mean_difference": ' + deep_array + "}"
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == first
    assert done["state"].get("response_format_repairs", 0) == 0
    assert len(model.requests) == 1


def test_deep_embedded_json_cannot_escape_the_terminal_failure_path(tmp_path):
    deep_array = "[" * 500 + "0" + "]" * 500
    first = "Draft: {\"complete_pairs\": 4, \"mean_difference\": " + deep_array + "}"
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == first
    assert len(model.requests) == 1


def test_embedded_json_repair_scan_has_a_fixed_parse_attempt_limit(tmp_path):
    first = ("{" * 40) + ' "complete_pairs": 4, "mean_difference": 2.0}'
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == first
    assert len(model.requests) == 1


def test_repair_scan_does_not_accept_a_candidate_when_the_attempt_limit_hides_more_text(tmp_path):
    first = '{"complete_pairs": [4], "mean_difference": 2.0} ' + ("{" * 40) + "unfinished"
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"].get("response_format_repairs", 0) == 0
    assert len(model.requests) == 1


@pytest.mark.parametrize("first", [
    '{"complete_pairs": 4, "mean_difference": 2.0, "padding": "' + ("x" * (64 * 1024)) + '"}',
    '{"complete_pairs": 4, "mean_difference": [' + ",".join("0" for _ in range(10_001)) + "]}",
])
def test_oversized_or_high_node_count_json_is_not_used_as_a_repair_seed(tmp_path, first):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=first)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == first
    assert done["state"].get("response_format_repairs", 0) == 0
    assert len(model.requests) == 1


def test_response_repair_fingerprint_exhaustion_is_a_terminal_failure(tmp_path, monkeypatch):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs": [4], "mean_difference": 2.0}')
    ])

    def exhausted(_value):
        raise RecursionError("signature depth exceeded")

    monkeypatch.setattr("adj_partner.runtime.engine.response_scalar_fingerprint", exhausted)
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"].startswith("{")
    assert any("format_repair_unavailable" in issue for issue in done["state"]["response_schema_errors"])
    assert len(model.requests) == 1


def test_schema_acceptance_is_structural_and_does_not_require_research_report(tmp_path):
    _, sid, _, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs": 999, "mean_difference": -1000.0}')
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert json.loads(done["state"]["delivery"]["content"]) == {
        "complete_pairs": 999, "mean_difference": -1000.0}


def test_absent_final_response_is_not_format_repaired(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content="")])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "empty_reply"
    assert len(model.requests) == 1


def test_unknown_cost_provider_failure_is_not_format_repaired(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [ProviderFailure("provider_http", "request outcome unknown")])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "provider_error"
    assert len(model.requests) == 1


def test_unknown_cost_provider_failure_during_repair_stops_without_another_retry(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs": [4], "mean_difference": 2.0}'),
        ProviderFailure("provider_http", "request outcome unknown"),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "provider_error"
    assert len(model.requests) == 2
    assert done["state"]["response_format_repairs"] == 1


def test_truncated_final_is_not_format_repaired(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs":', finish_reason="length", usage={"input_tokens": 10, "output_tokens": 5}),
        ModelReply(content='{"complete_pairs": [4], "mean', finish_reason="length", usage={"input_tokens": 10, "output_tokens": 5}),
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "model_length"
    assert len(model.requests) == 2
    assert done["state"]["output_length_continuations"] == 1
    assert "response_format_repairs" not in done["state"]


def test_truncated_reply_gets_one_larger_retry_with_an_explanation(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content="A long answer that was cut", finish_reason="length", usage={"input_tokens": 10, "output_tokens": 5}),
        ModelReply(content="A concise complete answer.", usage={"input_tokens": 10, "output_tokens": 5}),
    ])
    run = engine.start(sid, "Explain")

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded" and done["state"]["delivery"]["content"] == "A concise complete answer."
    first, second = model.requests
    assert second["max_output_tokens"] == 2 * first["max_output_tokens"]
    notes = [m["content"] for m in second["messages"] if m["role"] == "system"]
    assert any("reached the output-token limit" in note for note in notes)
    assert not any("reached the output-token limit" in m.get("content") or ""
                   for m in first["messages"] if m["role"] == "system")


def test_truncation_retry_respects_the_step_budget(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content="cut", finish_reason="length", usage={"input_tokens": 10, "output_tokens": 5}),
    ])
    run = engine.start(sid, "Explain", budget=Budget(max_steps=1).to_dict())

    done = engine.resume(run["id"])

    assert done["state"]["stop_reason"] == "model_length" and len(model.requests) == 1


def test_truncation_retry_is_withheld_when_cost_is_unknown(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content="cut", finish_reason="length")])
    done = engine.resume(engine.start(sid, "Explain")["id"])
    assert done["state"]["stop_reason"] == "model_length" and len(model.requests) == 1


def test_repair_is_skipped_when_task_has_no_remaining_step(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [
        ModelReply(content='{"complete_pairs": [4], "mean_difference": 2.0}')
    ])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA},
                       budget=Budget(max_steps=1).to_dict())

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert "response_format_repairs" not in done["state"]
    assert done["state"]["rejected_final_response"].startswith("{")
    assert len(model.requests) == 1


def test_missing_response_contract_preserves_ordinary_success_behavior(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content="plain response")])
    run = engine.start(sid, "Answer normally")

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "model_stop"
    assert done["state"]["delivery"]["content"] == "plain response"
    assert len(model.requests) == 1


def test_invalid_response_schema_is_rejected_at_task_submission(tmp_path):
    _, sid, _, engine = make_engine(tmp_path, [])

    with pytest.raises(ValueError, match="response_json_schema"):
        engine.start(sid, "Return counts", acceptance={"response_json_schema": {"type": "nonsense"}})


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"x": {"$dynamicRef": "https://example.invalid/schema"}}},
    {"type": "object", "properties": {"x": {"$recursiveRef": "https://example.invalid/schema"}}},
    {"$ref": "https://example.invalid/schema"},
])
def test_response_schema_rejects_external_references_at_task_submission(tmp_path, schema):
    _, sid, _, engine = make_engine(tmp_path, [])

    with pytest.raises(ValueError, match="local references"):
        engine.start(sid, "Return counts", acceptance={"response_json_schema": schema})


@pytest.mark.parametrize("schema", [
    {"$ref": "#/definitions/missing"},
    {"type": "object", "properties": {"x": {"$dynamicRef": "#/definitions/missing"}}},
])
def test_response_schema_rejects_unresolved_local_references_at_task_submission(tmp_path, schema):
    _, sid, _, engine = make_engine(tmp_path, [])

    with pytest.raises(ValueError, match="response_json_schema.*reference"):
        engine.start(sid, "Return counts", acceptance={"response_json_schema": schema})


@pytest.mark.parametrize("schema", [
    {"$defs": {"count": {"type": "integer"}}, "$ref": "#/$defs/count"},
    {"$id": "urn:example:root", "$defs": {"count": {"$anchor": "count", "type": "integer"}},
     "$ref": "#count"},
])
def test_response_schema_accepts_resolved_local_references(tmp_path, schema):
    _, sid, _, engine = make_engine(tmp_path, [ModelReply(content="4")])
    run = engine.start(sid, "Return count", acceptance={"response_json_schema": schema})

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"


def test_recursive_schema_validation_failure_is_terminal_and_auditable(tmp_path):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content="{}"), ModelReply(content="{}")])
    run = engine.start(sid, "Return a value", acceptance={"response_json_schema": {"$ref": "#"}})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["response_schema_errors"]
    assert len(model.requests) == 2


@pytest.mark.parametrize("invalid_json", [
    '{"complete_pairs": NaN, "mean_difference": 2.0}',
    '{"complete_pairs": Infinity, "mean_difference": 2.0}',
    '{"complete_pairs": -Infinity, "mean_difference": 2.0}',
    '{"complete_pairs": 1, "complete_pairs": 2, "mean_difference": 2.0}',
])
def test_nonstandard_constants_and_duplicate_keys_cannot_start_repair(tmp_path, invalid_json):
    _, sid, model, engine = make_engine(tmp_path, [ModelReply(content=invalid_json)])
    run = engine.start(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "response_schema_rejected"
    assert done["state"]["rejected_final_response"] == invalid_json
    assert done["state"]["response_schema_errors"][0].startswith("invalid_json:")
    assert "format_repair_unavailable" in done["state"]["response_schema_errors"][-1]
    assert len(model.requests) == 1


def test_service_submit_persists_the_public_response_contract(tmp_path):
    model = ScriptedModel([])
    service = AgentService(tmp_path / "service", model=model, project_root=tmp_path)
    try:
        sid = service.create_session()["id"]

        run = service.submit(sid, "Return counts", acceptance={"response_json_schema": SCHEMA})

        task = service.task(run["task_id"])
        assert task["acceptance"]["response_json_schema"] == SCHEMA
        assert model.requests == []
    finally:
        service.close()


def test_oversized_latest_tool_result_is_compacted_instead_of_failing(tmp_path):
    from adj_partner.runtime.context import ContextView
    group = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1",
              "content": json.dumps({"status": "ok", "data": {"rows": [{"text": "x" * 5000} for _ in range(40)]}})}]
    compacted = ContextView._compact_tool_results(group, {"max_string": 60, "max_list": 1, "max_dict": 6})
    assert compacted[0] == group[0]
    payload = json.loads(compacted[1]["content"])
    assert payload["status"] == "ok" and payload["compacted_for_context"] is True
    assert payload["original_bytes"] > 200_000 and len(compacted[1]["content"]) < 2_000
