from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall, ToolResult
from adj_partner.research.acceptance import RESPONSE_SCHEMA
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec


SCHEMA = {
    "type": "object",
    "properties": {"hypothesis": {"type": "string"}},
    "required": ["hypothesis"],
    "additionalProperties": False,
}


def _reply(content, usage=None):
    return ModelReply(content=content, tool_calls=[], finish_reason="stop",
                      usage=usage if usage is not None else {
                          "input_tokens": 1, "output_tokens": 1, "total_tokens": 2})


class _Port:
    metadata = {"provider": "fixture", "model": "scripted"}
    secrets = ()

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools, **limits):
        self.requests.append({"messages": messages, "tools": tools, **limits})
        return self.replies.pop(0)


def _engine(tmp_path, replies, *, budget=None):
    store = Store(tmp_path / "state")
    catalog = ToolCatalog()
    catalog.register(ToolSpec("echo", "Echo", {
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False,
    }), lambda _ctx, text: ToolResult(data={"text": text}))
    port = _Port(replies)
    engine = AgentEngine(store, catalog, port)
    public = tmp_path / "public.csv"
    public.write_text("x\n1\n")
    session = store.create_session(SessionConfig(
        enabled_tools=catalog.names(), allowed_tools=catalog.names(), allowed_roots=[str(tmp_path)],
        budget=budget or Budget()).to_dict())
    return store, engine, session["id"], port


def test_schema_valid_premature_final_gets_one_continuation_then_fails(tmp_path):
    store, engine, sid, port = _engine(tmp_path, [
        _reply('{"hypothesis":"No computation was performed."}'),
        _reply('{"hypothesis":"Still unable to compute."}'),
    ])
    run = engine.start(sid, "Analyze the bound CSV", acceptance={
        RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True},
        execution_binding={"allowed_roots": [str(tmp_path)], "input_paths": [str(tmp_path / "public.csv")]})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_execution_missing_after_continuation"
    assert done["state"]["execution_continuations"] == 1
    assert done["state"].get("delivery") is None
    assert len(port.requests) == 2
    assert any('{"summary":{"metric":123}}' in message["content"]
               for message in port.requests[0]["messages"] if message["role"] == "system")


def test_absent_witness_flag_keeps_schema_only_success(tmp_path):
    store, engine, sid, port = _engine(tmp_path, [_reply('{"hypothesis":"answer"}')])
    run = engine.start(sid, "Answer", acceptance={RESPONSE_SCHEMA: SCHEMA})

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert len(port.requests) == 1


def test_required_witness_without_host_binding_fails_before_model_call(tmp_path):
    store, engine, sid, port = _engine(tmp_path, [_reply('{"hypothesis":"answer"}')])
    run = engine.start(sid, "Analyze", acceptance={
        RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "required_execution_binding_missing"
    assert not port.requests


def test_format_repair_fingerprint_is_scoped_before_execution_continuation(tmp_path, monkeypatch):
    store, engine, sid, port = _engine(tmp_path, [
        _reply('prefix {"hypothesis":"premature"}'),
        _reply('{"hypothesis":"premature"}'),
        _reply('{"hypothesis":"computed narrative"}'),
    ])
    witness_calls = []

    def witness(_store, _run_id, _binding, *, response_schema=None):
        witness_calls.append(response_schema)
        return None if len(witness_calls) < 4 else {
            "kind": "execution_witness", "witnesses": [{"call_id": "call", "analysis_id": "analysis", "job_id": "job"}],
            "witness_count": 1, "semantic_validation": "not_performed",
            "narrative_derivation_validated": False,
        }

    monkeypatch.setattr("adj_partner.runtime.engine.verify_execution_witness", witness)
    run = engine.start(sid, "Analyze the bound CSV", acceptance={
        RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True},
        execution_binding={"allowed_roots": [str(tmp_path)], "input_paths": [str(tmp_path / "public.csv")]})

    done = engine.resume(run["id"])

    assert done["status"] == "succeeded"
    assert done["state"]["stop_reason"] == "response_schema_valid"
    assert done["state"]["response_format_repairs"] == 1
    assert done["state"]["response_format_repair_consumed"] is True
    assert done["state"]["response_repair_scalar_path_fingerprint"]
    assert done["state"]["execution_continuations"] == 1
    assert done["state"]["delivery"]["content"] == '{"hypothesis":"computed narrative"}'
    assert len(port.requests) == 3
    assert not any("Host format-only repair" in message["content"]
                   for message in port.requests[-1]["messages"] if message["role"] == "system")


def test_budget_exhaustion_precedes_missing_continuation_reason(tmp_path):
    store, engine, sid, port = _engine(tmp_path, [_reply('{"hypothesis":"premature"}')],
                                        budget=Budget(max_steps=1))
    run = engine.start(sid, "Analyze the bound CSV", acceptance={
        RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True},
        execution_binding={"allowed_roots": [str(tmp_path)], "input_paths": [str(tmp_path / "public.csv")]})

    done = engine.resume(run["id"])

    assert done["status"] == "budget_exhausted"
    assert done["state"]["stop_reason"] == "budget_exhausted"
    assert len(port.requests) == 1


def test_unknown_provider_usage_stops_before_execution_continuation(tmp_path):
    store, engine, sid, port = _engine(tmp_path, [_reply('{"hypothesis":"premature"}', usage={}),
                                                   _reply('{"hypothesis":"should not be requested"}')])
    run = engine.start(sid, "Analyze the bound CSV", acceptance={
        RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True},
        execution_binding={"allowed_roots": [str(tmp_path)], "input_paths": [str(tmp_path / "public.csv")]})

    done = engine.resume(run["id"])

    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert done["usage"]["unknown_requests"] == 1
    assert len(port.requests) == 1

