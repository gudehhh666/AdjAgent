"""Offline tests for the opt-in execution-bound finalization path."""

import pytest
from threading import Event

from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall, ToolResult
from adj_partner.research.acceptance import (
    EXECUTION_BOUND_DELIVERY,
    REQUIRE_BOUND_EXECUTION_WITNESS,
    RESPONSE_SCHEMA,
    requirements,
    research_acceptance,
)
from adj_partner.runtime.execution_delivery import VerificationFailure
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec
from test_runtime_engine import reply


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
    "additionalProperties": False,
}
ACCEPTANCE = {RESPONSE_SCHEMA: SCHEMA, "execution_bound_delivery": True}
COMBINED_ACCEPTANCE = {
    RESPONSE_SCHEMA: SCHEMA,
    EXECUTION_BOUND_DELIVERY: True,
    REQUIRE_BOUND_EXECUTION_WITNESS: True,
}


def test_execution_bound_acceptance_requires_schema_and_rejects_report_contracts():
    with pytest.raises(ValueError, match="response_json_schema"):
        requirements({"execution_bound_delivery": True})
    with pytest.raises(ValueError, match="research|report|execution_bound_delivery"):
        requirements({**ACCEPTANCE, "require_evidence": False})
    with pytest.raises(ValueError, match="object"):
        requirements({"response_json_schema": {"type": "integer"},
                      "execution_bound_delivery": True})
    with pytest.raises(ValueError, match="boolean"):
        requirements({"response_json_schema": SCHEMA, "execution_bound_delivery": 1})


def test_false_or_absent_execution_flag_keeps_schema_acceptance_behavior():
    absent = requirements({"response_json_schema": SCHEMA})
    disabled = requirements({"response_json_schema": SCHEMA, "execution_bound_delivery": False})

    assert not research_acceptance({"response_json_schema": SCHEMA})
    assert not research_acceptance({"response_json_schema": SCHEMA,
                                    "execution_bound_delivery": False})
    assert absent["require_report"] == disabled["require_report"] is False


def test_execution_bound_delivery_composes_with_witness_but_not_report_acceptance():
    combined = requirements(COMBINED_ACCEPTANCE)

    assert combined[EXECUTION_BOUND_DELIVERY] is True
    assert combined[REQUIRE_BOUND_EXECUTION_WITNESS] is True
    assert combined["require_report"] is False
    assert not research_acceptance(COMBINED_ACCEPTANCE)

    witness_disabled = requirements({
        RESPONSE_SCHEMA: SCHEMA,
        EXECUTION_BOUND_DELIVERY: True,
        REQUIRE_BOUND_EXECUTION_WITNESS: False,
    })
    assert witness_disabled[REQUIRE_BOUND_EXECUTION_WITNESS] is False
    assert witness_disabled["require_report"] is False

    with pytest.raises(ValueError, match="research|report|execution_bound_delivery"):
        requirements({**COMBINED_ACCEPTANCE, "require_evidence": False})


def _engine(tmp_path, replies, *, budget=None):
    store = Store(tmp_path / "state")
    catalog = ToolCatalog()
    catalog.register(ToolSpec("echo", "Echo a value", {
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False,
    }), lambda _ctx, text: ToolResult(data={"text": text}))
    port = _ScriptedPort(replies)
    engine = AgentEngine(store, catalog, port)
    session = store.create_session(SessionConfig(
        enabled_tools=catalog.names(), allowed_tools=catalog.names(),
        allowed_roots=[str(tmp_path)], budget=budget or Budget()).to_dict())
    return store, engine, session["id"], port


class _ScriptedPort:
    metadata = {"provider": "fixture", "model": "scripted"}
    secrets = ()

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools, **limits):
        self.requests.append({"messages": messages, "tools": tools, **limits})
        return self.replies.pop(0)


def _tool_turn():
    return reply(calls=[ToolCall("compute", "echo", {"text": "finished"})])


def _verified_delivery(store, run_id, schema):
    return {"kind": "execution_bound_json", "content": '{"value":3}',
            "assurance": "Host analysis completed; computation semantic correctness was not independently validated.",
            "provenance": {"run_id": run_id, "response_schema": schema,
                           "semantic_validation": "not_performed"}}


def test_verified_delivery_follows_checkpoint_and_skips_a_second_model_call(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [_tool_turn()])
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE)
    observed = {}

    def verify(candidate_store, run_id, schema):
        observed["messages"] = candidate_store.messages(sid)
        observed["events"] = [event["kind"] for event in candidate_store.events(sid, run_id)]
        if "runtime.batch_finished" not in observed["events"]:
            return None
        assert any(item["role"] == "tool" for item in observed["messages"])
        assert schema == SCHEMA
        return _verified_delivery(candidate_store, run_id, schema)

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", verify)

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "execution_bound_delivery"
    assert done["state"]["delivery"]["kind"] == "execution_bound_json"
    assert done["state"]["delivery"]["provenance"]["semantic_validation"] == "not_performed"
    assert len(port.requests) == 1
    first_context = "\n".join(message["content"] for message in port.requests[0]["messages"]
                               if isinstance(message.get("content"), str))
    assert "Execution-bound delivery is enabled" in first_context
    assert any(item.get("function", {}).get("name") == "echo" for item in port.requests[0]["tools"])
    assert store.messages(sid)[-1]["content"] == '{"value":3}'
    assert "runtime.execution_bound_delivery_verified" in [
        event["kind"] for event in store.events(sid, run["id"])]


def test_combined_witness_and_direct_delivery_finalize_the_same_tool_run(tmp_path, monkeypatch):
    public_input = tmp_path / "public.csv"
    public_input.write_text("value\n3\n")
    store, engine, sid, port = _engine(tmp_path, [_tool_turn()])
    run = engine.start(
        sid,
        "Compute value from the bound input",
        acceptance=COMBINED_ACCEPTANCE,
        execution_binding={
            "allowed_roots": [str(tmp_path)],
            "input_paths": [str(public_input)],
        },
    )
    witness = {
        "kind": "execution_witness",
        "witnesses": [{"call_id": "compute", "analysis_id": "analysis", "job_id": "job"}],
        "witness_count": 1,
        "semantic_validation": "not_performed",
        "narrative_derivation_validated": False,
        "assurance": "Synthetic host witness for the combined runtime regression.",
    }
    observed = {"witness_calls": 0, "delivery_calls": 0}

    def verify_witness(candidate_store, run_id, binding, *, response_schema=None):
        observed["witness_calls"] += 1
        assert response_schema == SCHEMA
        assert binding["inputs"][0]["relative_path"] == "public.csv"
        if not any(event["kind"] == "runtime.batch_finished"
                   for event in candidate_store.events(sid, run_id)):
            return None
        return witness

    def verify_delivery(candidate_store, run_id, schema):
        observed["delivery_calls"] += 1
        assert schema == SCHEMA
        assert any(event["kind"] == "runtime.execution_witness_verified"
                   for event in candidate_store.events(sid, run_id))
        return _verified_delivery(candidate_store, run_id, schema)

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_witness", verify_witness)
    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", verify_delivery)

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "execution_bound_delivery"
    assert done["state"]["execution_witness"] == witness
    assert done["state"]["delivery"]["execution_witness"] == witness
    assert done["state"]["delivery"]["kind"] == "execution_bound_json"
    assert observed["witness_calls"] >= 2
    assert observed["delivery_calls"] == 1
    assert len(port.requests) == 1
    events = [event["kind"] for event in store.events(sid, run["id"])]
    assert events.index("runtime.execution_witness_verified") < events.index(
        "runtime.execution_bound_delivery_verified")


def test_no_unique_or_schema_compatible_candidate_continues_the_model_loop(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [_tool_turn(), reply('{"value":4}')])
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE)
    checks = []

    def no_candidate(_store, _run_id, _schema):
        checks.append(True)
        return None

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", no_candidate)

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert len(checks) >= 1
    assert len(port.requests) == 2


def test_integrity_failure_is_terminal_after_preserving_tool_group(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [_tool_turn(), reply('{"value":4}')])
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE)

    def invalid(candidate_store, run_id, _schema):
        if not any(event["kind"] == "runtime.batch_finished"
                   for event in candidate_store.events(sid, run_id)):
            return None
        raise VerificationFailure("hash mismatch")

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", invalid)

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "execution_bound_verification_failed"
    assert "hash mismatch" in done["state"]["error"]["message"]
    assert len(port.requests) == 1
    assert any(item["role"] == "tool" for item in store.messages(sid))


def test_cancellation_after_verification_prevents_success_delivery(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [_tool_turn()])
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE)

    def cancel_then_deliver(candidate_store, run_id, schema):
        if not any(event["kind"] == "runtime.batch_finished"
                   for event in candidate_store.events(sid, run_id)):
            return None
        engine.journal.cancel(run_id)
        return _verified_delivery(store, run_id, schema)

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", cancel_then_deliver)

    done = engine.resume(run["id"])

    assert done["status"] == "cancelled"
    assert done["state"].get("delivery") is None
    assert len(port.requests) == 1


def test_wall_budget_expiry_during_verification_prevents_delivery(tmp_path, monkeypatch):
    budget = Budget(max_steps=4, max_input_tokens=10_000, max_output_tokens=10_000,
                    wall_time_seconds=0.2)
    store, engine, sid, port = _engine(tmp_path, [_tool_turn()], budget=budget)
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE, budget=budget.to_dict())

    def exhaust_then_deliver(candidate_store, run_id, schema):
        if not any(event["kind"] == "runtime.batch_finished"
                   for event in candidate_store.events(sid, run_id)):
            return None
        import time
        time.sleep(0.25)
        return _verified_delivery(store, run_id, schema)

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery", exhaust_then_deliver)

    done = engine.resume(run["id"])

    assert done["status"] == "budget_exhausted"
    assert done["state"].get("delivery") is None
    assert len(port.requests) == 1


def test_resume_after_tool_group_checkpoint_verifies_without_replaying_or_reasking(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [_tool_turn()])
    run = engine.start(sid, "Compute value", acceptance=ACCEPTANCE)
    detached = Event()
    original = engine.journal.checkpoint

    def detach_after_batch(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.batch_finished":
            detached.set()
        return value

    monkeypatch.setattr(engine.journal, "checkpoint", detach_after_batch)
    waiting = engine.resume(run["id"], stop_event=detached)

    assert waiting["status"] == "waiting"
    assert waiting["state"]["waiting"]["kind"] == "client_detached"
    assert len(port.requests) == 1

    monkeypatch.setattr(engine.journal, "checkpoint", original)
    delivery = _verified_delivery(store, run["id"], SCHEMA)
    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_delivery",
                        lambda _store, _run_id, _schema: delivery)
    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "execution_bound_delivery"
    assert len(port.requests) == 1
    assert sum(event["kind"] == "runtime.batch_finished"
               for event in store.events(sid, run["id"])) == 1
