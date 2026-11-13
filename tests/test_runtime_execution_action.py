"""Offline runtime tests for the opt-in required execution-action stage."""

import json

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall, ToolResult
from adj_partner.research.acceptance import (
    EXECUTION_BOUND_DELIVERY,
    REQUIRE_BOUND_EXECUTION_WITNESS,
    REQUIRE_EXECUTION_ACTION,
    RESPONSE_SCHEMA,
    requirements,
    research_acceptance,
)
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
    "additionalProperties": False,
}

ACTION_ACCEPTANCE = {
    RESPONSE_SCHEMA: SCHEMA,
    REQUIRE_BOUND_EXECUTION_WITNESS: True,
    REQUIRE_EXECUTION_ACTION: True,
}


class RecordingPort:
    secrets = ()

    def __init__(self, replies, *, supports=True):
        self.replies = list(replies)
        self.requests = []
        self.metadata = {"provider": "fixture", "model": "scripted"}
        if supports:
            self.metadata["tool_choice_modes"] = ["auto", "required_or_refuse"]

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds, **options):
        self.requests.append({"messages": messages, "tools": tools,
                              "max_output_tokens": max_output_tokens,
                              "timeout_seconds": timeout_seconds, **options})
        return self.replies.pop(0)


def reply(content="", *, calls=(), refusal=None, usage=None):
    return ModelReply(content=content, tool_calls=list(calls), refusal=refusal,
                      finish_reason="tool_calls" if calls else "stop",
                      usage=usage if usage is not None else {
                          "input_tokens": 11, "output_tokens": 3, "total_tokens": 14})


def make_runtime(tmp_path, replies, *, supports=True, analysis_tool=True, budget=None,
                 acceptance=None):
    store = Store(tmp_path / "state")
    catalog = ToolCatalog()
    name = "code_interpreter" if analysis_tool else "echo"
    catalog.register(ToolSpec(name, "Synthetic authorized tool", {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    }), lambda _ctx, code: ToolResult(data={"observed": code}))
    port = RecordingPort(replies, supports=supports)
    engine = AgentEngine(store, catalog, port)
    public = tmp_path / "public.csv"
    public.write_text("value\n3\n")
    session = store.create_session(SessionConfig(
        enabled_tools=catalog.names(), allowed_tools=catalog.names(),
        allowed_roots=[str(tmp_path)], budget=budget or Budget()).to_dict())
    run = engine.start(session["id"], "Analyze the bound public input",
        acceptance=ACTION_ACCEPTANCE if acceptance is None else acceptance,
        execution_binding={"allowed_roots": [str(tmp_path)], "input_paths": [str(public)]})
    return store, engine, run, port


def host_policy(request):
    message = next(item for item in request["messages"]
                   if item.get("role") == "system" and "\nHost task policy:\n" in item.get("content", ""))
    return json.loads(message["content"].split("\nHost task policy:\n", 1)[1])


def test_execution_action_contract_requires_schema_and_bound_witness():
    with pytest.raises(ValueError, match="require_execution_action|witness"):
        requirements({"require_execution_action": True})
    with pytest.raises(ValueError, match="require_execution_action|witness"):
        requirements({"response_json_schema": SCHEMA, "require_execution_action": True})
    with pytest.raises(ValueError, match="boolean"):
        requirements({"response_json_schema": SCHEMA,
                      "require_bound_execution_witness": True,
                      "require_execution_action": 1})

    contract = {
        "response_json_schema": SCHEMA,
        "require_bound_execution_witness": True,
        "require_execution_action": True,
        "execution_bound_delivery": True,
    }
    accepted = requirements(contract)

    assert accepted["require_execution_action"] is True
    assert accepted["require_report"] is False
    assert not research_acceptance(contract)


def test_required_action_rejects_unsupported_port_before_model_call(tmp_path):
    _store, engine, run, port = make_runtime(
        tmp_path, [reply('{"value":3}')], supports=False)

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_unsupported"
    assert port.requests == []
    assert done["usage"] == {}


def test_required_action_requires_an_exposed_trusted_analysis_tool(tmp_path):
    _store, engine, run, port = make_runtime(
        tmp_path, [reply('{"value":3}')], analysis_tool=False)

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_tool_unavailable"
    assert port.requests == []
    assert done["usage"] == {}


def test_required_action_empty_final_is_terminal_after_real_usage_is_counted(tmp_path):
    store, engine, run, port = make_runtime(
        tmp_path, [reply('{"value":3}', usage={
            "input_tokens": 17, "output_tokens": 5, "total_tokens": 22})])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_missing"
    assert done["state"]["reply_tool_choice"] == "required_or_refuse"
    assert done["usage"]["input_tokens"] == 17
    assert done["usage"]["output_tokens"] == 5
    assert done["usage"]["unknown_requests"] == 0
    assert len(port.requests) == 1
    assert port.requests[0]["tool_choice"] == "required_or_refuse"
    assert "execution_continuations" not in done["state"]
    assert done["state"].get("response_format_repairs", 0) == 0
    requested = [event for event in store.events(run["session_id"], run["id"])
                 if event["kind"] == "model.requested"]
    assert len(requested) == 1
    assert requested[0]["payload"]["tool_choice"] == "required_or_refuse"


def test_required_action_explicit_refusal_is_terminal_without_retry_or_tool(tmp_path):
    store, engine, run, port = make_runtime(
        tmp_path, [reply(refusal="I cannot perform the authorized analysis.")])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "model_refusal"
    assert len(port.requests) == 1
    assert store.list_actions(run["id"]) == []
    assert done["usage"]["input_tokens"] == 11
    assert "execution_continuations" not in done["state"]


def test_required_action_whitespace_refusal_matches_transport_nonempty_contract(tmp_path):
    store, engine, run, port = make_runtime(tmp_path, [reply(refusal="   ")])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "model_refusal"
    assert len(port.requests) == 1
    assert store.list_actions(run["id"]) == []


def test_required_action_empty_refusal_is_still_missing(tmp_path):
    _store, engine, run, _port = make_runtime(tmp_path, [reply(refusal="")])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_missing"


def test_required_action_mixed_tool_and_refusal_is_protocol_failure(tmp_path):
    store, engine, run, _port = make_runtime(tmp_path, [reply(
        refusal="I cannot proceed.",
        calls=[ToolCall("must-not-run", "code_interpreter", {"code": "return 3"})],
    )])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_protocol"
    assert store.list_actions(run["id"]) == []


def test_required_action_stage_is_explicit_in_model_visible_policy(tmp_path):
    _store, engine, run, port = make_runtime(
        tmp_path, [reply(refusal="Execution is unavailable.")])

    engine.resume(run["id"])

    policy = host_policy(port.requests[0])
    assert policy["request_stage"] == "execution_action_or_refusal"
    assert policy["tool_choice"] == "required_or_refuse"
    assert policy["final_response_allowed"] is False
    assert "code_interpreter" in policy["visible_tool_names"]


def test_current_verification_clears_stale_witness_before_projecting_stage(tmp_path, monkeypatch):
    store, engine, run, port = make_runtime(
        tmp_path, [reply(refusal="The analysis cannot be performed.")])
    state = dict(run["state"])
    state["execution_witness"] = {"kind": "stale_fixture"}
    store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_witness",
                        lambda *_args, **_kwargs: None)

    done = engine.resume(run["id"])

    assert done["state"]["stop_reason"] == "model_refusal"
    assert "execution_witness" not in done["state"]
    assert port.requests[0]["tool_choice"] == "required_or_refuse"
    assert host_policy(port.requests[0])["request_stage"] == "execution_action_or_refusal"


def test_verified_execution_switches_the_next_request_back_to_auto(tmp_path, monkeypatch):
    store, engine, run, port = make_runtime(tmp_path, [
        reply(calls=[ToolCall("analysis-call", "code_interpreter", {"code": "return 3"})]),
        reply('{"value":3}'),
    ])
    witness = {
        "kind": "execution_witness",
        "witnesses": [{"call_id": "analysis-call", "analysis_id": "analysis", "job_id": "job"}],
        "witness_count": 1,
        "semantic_validation": "not_performed",
        "narrative_derivation_validated": False,
    }

    def verify(_store, run_id, _binding, *, response_schema=None):
        assert response_schema == SCHEMA
        return witness if _store.list_actions(run_id) else None

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_witness", verify)

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert done["state"]["execution_witness"] == witness
    assert len(store.list_actions(run["id"])) == 1
    assert len(port.requests) == 2
    assert port.requests[0]["tool_choice"] == "required_or_refuse"
    assert "tool_choice" not in port.requests[1]
    final_policy = host_policy(port.requests[1])
    assert final_policy["request_stage"] == "final_or_more_tools"
    assert final_policy["tool_choice"] == "auto"
    assert final_policy["final_response_allowed"] is True


def test_verified_execution_can_complete_by_existing_bound_delivery(tmp_path, monkeypatch):
    acceptance = {**ACTION_ACCEPTANCE, EXECUTION_BOUND_DELIVERY: True}
    store, engine, run, port = make_runtime(tmp_path, [
        reply(calls=[ToolCall("analysis-call", "code_interpreter", {"code": "return 3"})]),
    ], acceptance=acceptance)
    witness = {
        "kind": "execution_witness",
        "witnesses": [{"call_id": "analysis-call", "analysis_id": "analysis", "job_id": "job"}],
        "witness_count": 1,
        "semantic_validation": "not_performed",
        "narrative_derivation_validated": False,
    }

    monkeypatch.setattr(
        "adj_partner.runtime.engine.verify_execution_witness",
        lambda _store, run_id, _binding, **_kwargs:
            witness if _store.list_actions(run_id) else None,
    )

    def delivery(_store, run_id, response_schema):
        if not _store.list_actions(run_id):
            return None
        assert response_schema == SCHEMA
        return {
            "kind": "execution_bound_json",
            "content": '{"value":3}',
            "provenance": {"call_id": "analysis-call", "run_id": run_id,
                           "tool_name": "code_interpreter"},
            "assurance": "Synthetic unit-test delivery.",
        }

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", delivery)

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "execution_bound_delivery"
    assert done["state"]["delivery"]["execution_witness"] == witness
    assert len(port.requests) == 1
    assert len(store.list_actions(run["id"])) == 1


def test_recovered_saved_reply_keeps_required_action_gate_and_usage(tmp_path, monkeypatch):
    store, engine, run, port = make_runtime(
        tmp_path, [reply('{"value":3}', usage={
            "input_tokens": 19, "output_tokens": 7, "total_tokens": 26})])
    original = engine.journal.checkpoint

    def crash_after_reply(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed":
            raise KeyboardInterrupt("synthetic crash after durable reply")
        return value

    monkeypatch.setattr(engine.journal, "checkpoint", crash_after_reply)
    with pytest.raises(KeyboardInterrupt, match="durable reply"):
        engine.resume(run["id"])

    checkpoint = store.get_run(run["id"])
    assert checkpoint["state"]["phase"] == "reply"
    assert checkpoint["state"]["reply_tool_choice"] == "required_or_refuse"
    assert checkpoint["usage"]["input_tokens"] == 19
    assert len(port.requests) == 1

    monkeypatch.setattr(engine.journal, "checkpoint", original)
    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_action_missing"
    assert done["usage"]["input_tokens"] == 19
    assert done["usage"]["output_tokens"] == 7
    assert len(port.requests) == 1


def test_required_action_respects_existing_step_budget_without_retry(tmp_path, monkeypatch):
    store, engine, run, port = make_runtime(
        tmp_path,
        [reply(calls=[ToolCall("analysis-call", "code_interpreter", {"code": "return 3"})])],
        budget=Budget(max_steps=1),
    )
    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_witness",
                        lambda *_args, **_kwargs: None)

    done = engine.resume(run["id"])

    assert done["status"] == "budget_exhausted"
    assert done["state"]["stop_reason"] == "budget_exhausted"
    assert len(port.requests) == 1
    assert len(store.list_actions(run["id"])) == 1
    assert done["state"].get("execution_continuations", 0) == 0


def test_required_action_unknown_usage_blocks_tool_before_execution(tmp_path):
    store, engine, run, port = make_runtime(tmp_path, [reply(
        calls=[ToolCall("must-not-run", "code_interpreter", {"code": "return 3"})],
        usage={},
    )])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert done["usage"]["unknown_requests"] == 1
    assert len(port.requests) == 1
    assert store.list_actions(run["id"]) == []


def test_recovered_required_action_unknown_usage_still_blocks_tool(tmp_path, monkeypatch):
    store, engine, run, port = make_runtime(tmp_path, [reply(
        calls=[ToolCall("must-not-run", "code_interpreter", {"code": "return 3"})],
        usage={},
    )])
    original = engine.journal.checkpoint

    def crash_after_reply(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed":
            raise KeyboardInterrupt("synthetic crash after unknown-usage reply")
        return value

    monkeypatch.setattr(engine.journal, "checkpoint", crash_after_reply)
    with pytest.raises(KeyboardInterrupt, match="unknown-usage reply"):
        engine.resume(run["id"])
    assert store.get_run(run["id"])["state"]["reply_tool_choice"] == "required_or_refuse"

    monkeypatch.setattr(engine.journal, "checkpoint", original)
    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert done["usage"]["unknown_requests"] == 1
    assert len(port.requests) == 1
    assert store.list_actions(run["id"]) == []


def test_required_action_partial_usage_is_unknown_and_blocks_tool(tmp_path):
    store, engine, run, _port = make_runtime(tmp_path, [reply(
        calls=[ToolCall("must-not-run", "code_interpreter", {"code": "return 3"})],
        usage={"input_tokens": 7},
    )])

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert done["usage"]["input_tokens"] == 7
    assert done["usage"]["output_tokens"] == 0
    assert done["usage"]["unknown_requests"] == 1
    assert done["usage"]["accounted_output_tokens"] > 0
    assert store.list_actions(run["id"]) == []


def test_default_auto_path_keeps_legacy_port_signature_unchanged(tmp_path):
    class LegacyPort:
        secrets = ()
        metadata = {"provider": "fixture", "model": "legacy"}

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, *, max_output_tokens, timeout_seconds):
            self.calls += 1
            return reply('{"value":3}')

    store = Store(tmp_path / "legacy-state")
    catalog = ToolCatalog()
    port = LegacyPort()
    engine = AgentEngine(store, catalog, port)
    session = store.create_session(SessionConfig().to_dict())
    run = engine.start(session["id"], "Return a schema-valid value",
                       acceptance={RESPONSE_SCHEMA: SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert "reply_tool_choice" not in done["state"]
    assert port.calls == 1
