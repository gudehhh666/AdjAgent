"""Offline mode identity, predispatch compatibility and accounting boundaries."""

import json
from types import SimpleNamespace

import pytest

from adj_partner.contracts import ModelReply, ToolCall
from adj_partner.models.base import ProviderFailure
from adj_partner.models.openai_compatible import OpenAICompatiblePort
from adj_partner.runtime.model_invocations import InvocationLedger, digest, leased_run
from test_model_invocations import setup


MESSAGES = [{"role": "user", "content": "Use the authorized fixture tool."}]
TOOLS = [{"type": "function", "function": {
    "name": "fixture", "parameters": {
        "type": "object", "properties": {}, "additionalProperties": False}}}]
REQUIRED = "required_or_refuse"


class LegacyPort:
    secrets = ()
    metadata = {"provider_kind": "scripted", "model": "legacy-fixture"}

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds):
        self.calls += 1
        return ModelReply(content="answer", usage={"input_tokens": 4, "output_tokens": 2})


class RequiredPort(LegacyPort):
    metadata = {"provider_kind": "scripted", "model": "required-fixture",
                "tool_choice_modes": ["auto", REQUIRED]}

    def __init__(self, outcome=None):
        super().__init__()
        self.modes = []
        self.outcome = outcome

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds, tool_choice="auto"):
        self.calls += 1
        self.modes.append(tool_choice)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome or ModelReply(
            tool_calls=[ToolCall(id="fixture-call", name="fixture", arguments={})],
            finish_reason="tool_calls", usage={"input_tokens": 7, "output_tokens": 3})


def invoke(ledger, sid, rid, port, guard, **kwargs):
    return ledger.complete(sid, rid, port, MESSAGES, TOOLS, role="critic", key="one",
                           max_output_tokens=100, timeout_seconds=10, guard=guard, **kwargs)


def row(store, rid):
    with store.transaction() as conn:
        result = conn.execute("SELECT * FROM model_invocations_v1 WHERE run_id=?", (rid,)).fetchone()
    return dict(result) if result else None


def test_default_and_explicit_auto_keep_old_identity_and_legacy_signature(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), LegacyPort()
    with leased_run(store, rid) as guard:
        first = invoke(ledger, sid, rid, port, guard)
        second = invoke(ledger, sid, rid, port, guard, tool_choice="auto")
    expected = {"messages": MESSAGES, "tools": TOOLS, "model": port.metadata,
                "role": "critic", "max_output_tokens": 100}
    stored = row(store, rid)
    assert stored["identity"] == digest(expected)
    assert "tool_choice" not in json.loads(stored["reservation"])
    assert second.content == first.content and port.calls == 1
    assert ledger.usage(rid)["input_tokens"] == 4


def test_required_mode_is_forwarded_durable_and_cached_without_second_call(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), RequiredPort()
    with leased_run(store, rid) as guard:
        first = invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
        second = invoke(InvocationLedger(store), sid, rid, port, guard, tool_choice=REQUIRED)
    expected = {"messages": MESSAGES, "tools": TOOLS, "model": port.metadata,
                "role": "critic", "max_output_tokens": 100, "tool_choice": REQUIRED}
    stored = row(store, rid)
    assert stored["identity"] == digest(expected)
    assert json.loads(stored["reservation"])["tool_choice"] == REQUIRED
    assert second.tool_calls == first.tool_calls and port.modes == [REQUIRED]
    assert ledger.usage(rid) == {"requests": 1, "input_tokens": 7, "output_tokens": 3,
                               "accounted_input_tokens": 7, "accounted_output_tokens": 3,
                               "unknown_requests": 0}


@pytest.mark.parametrize("first,second", [("auto", REQUIRED), (REQUIRED, "auto")])
def test_same_key_mode_change_is_identity_drift_not_cached_replay(tmp_path, first, second):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), RequiredPort()
    with leased_run(store, rid) as guard:
        invoke(ledger, sid, rid, port, guard, tool_choice=first)
        before = row(store, rid)
        with pytest.raises(ValueError, match="identity"):
            invoke(ledger, sid, rid, port, guard, tool_choice=second)
    assert row(store, rid) == before and port.calls == 1


def test_mode_drift_precedes_unsupported_capability_for_existing_key(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), LegacyPort()
    with leased_run(store, rid) as guard:
        invoke(ledger, sid, rid, port, guard)
        with pytest.raises(ValueError, match="identity"):
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    assert port.calls == 1


def test_unsupported_mode_fails_before_dispatch_reservation_and_budget_clock(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), LegacyPort()
    with leased_run(store, rid) as guard:
        with pytest.raises(ProviderFailure) as error:
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    assert error.value.code == "unsupported_tool_choice"
    assert error.value.cost_unknown is False and error.value.usage == {}
    assert row(store, rid) is None and port.calls == 0
    assert ledger.usage(rid)["requests"] == 0
    with store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM model_budget_clocks_v1").fetchone()[0] == 0


@pytest.mark.parametrize("tools,mode", [([], REQUIRED), (TOOLS, "required"), (TOOLS, None)])
def test_invalid_required_request_never_reserves_or_dispatches(tmp_path, tools, mode):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), RequiredPort()
    with leased_run(store, rid) as guard:
        with pytest.raises(ProviderFailure) as error:
            ledger.complete(sid, rid, port, MESSAGES, tools, role="critic", key="one",
                            max_output_tokens=100, timeout_seconds=10, guard=guard, tool_choice=mode)
    assert error.value.code == "invalid_request" and error.value.cost_unknown is False
    assert row(store, rid) is None and port.calls == 0


def test_required_reservation_is_committed_before_port_effect(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger = InvocationLedger(store)

    class InspectingPort(RequiredPort):
        def complete(self, *args, **kwargs):
            # A separately opened Store observes the intent before the external boundary.
            from adj_partner.storage import Store
            stored = row(Store(tmp_path / "state"), rid)
            assert stored["status"] == "pending"
            assert json.loads(stored["reservation"])["tool_choice"] == REQUIRED
            assert json.loads(stored["usage"])["unknown_requests"] == 1
            return super().complete(*args, **kwargs)

    port = InspectingPort()
    with leased_run(store, rid) as guard:
        invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    assert row(store, rid)["status"] == "completed" and port.calls == 1


@pytest.mark.parametrize("usage,unknown", [({"input_tokens": 13, "output_tokens": 5}, False), ({}, True)])
def test_required_failure_keeps_known_or_unknown_usage_and_refuses_replay(tmp_path, usage, unknown):
    store, sid, rid = setup(tmp_path)
    failure = ProviderFailure("fixture_failure", "Offline failure", usage=usage, cost_unknown=unknown)
    ledger, port = InvocationLedger(store), RequiredPort(failure)
    with leased_run(store, rid) as guard:
        with pytest.raises(ProviderFailure):
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
        stored = row(store, rid)
        with pytest.raises(RuntimeError, match="cannot be replayed"):
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    totals = ledger.usage(rid)
    assert stored["status"] == "failed" and row(store, rid) == stored
    assert json.loads(stored["reservation"])["tool_choice"] == REQUIRED
    assert port.calls == totals["requests"] == 1
    assert totals["unknown_requests"] == int(unknown)
    if unknown:
        assert totals["accounted_input_tokens"] > 0 and totals["accounted_output_tokens"] == 100
    else:
        assert totals["input_tokens"] == totals["accounted_input_tokens"] == 13
        assert totals["output_tokens"] == totals["accounted_output_tokens"] == 5


@pytest.mark.parametrize("refusal", [None, ""])
@pytest.mark.parametrize("usage", [{"input_tokens": 13, "output_tokens": 5}, {}, {"input_tokens": 13}])
def test_required_premature_final_fails_durably_with_real_usage(tmp_path, refusal, usage):
    store, sid, rid = setup(tmp_path)
    ledger = InvocationLedger(store)
    port = RequiredPort(ModelReply(content="premature final", refusal=refusal, usage=usage))
    with leased_run(store, rid) as guard:
        with pytest.raises(ProviderFailure) as error:
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
        failed = row(store, rid)
        with pytest.raises(RuntimeError, match="cannot be replayed"):
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    unknown = not {"input_tokens", "output_tokens"} <= usage.keys()
    assert error.value.code == "required_action_missing"
    assert error.value.usage == usage and error.value.cost_unknown is unknown
    assert failed["status"] == "failed" and failed["receipt"] is None
    assert row(store, rid) == failed and port.calls == 1
    totals = ledger.usage(rid)
    assert totals["requests"] == 1 and totals["unknown_requests"] == int(unknown)
    for name in ("input_tokens", "output_tokens"):
        assert totals[name] == usage.get(name, 0)
        if name in usage:
            assert totals["accounted_" + name] == usage[name]
        else:
            assert totals["accounted_" + name] > 0


@pytest.mark.parametrize("refusal", [" ", "Cannot perform this task."])
def test_required_nonempty_refusal_is_preserved_without_semantic_filter(tmp_path, refusal):
    store, sid, rid = setup(tmp_path)
    ledger = InvocationLedger(store)
    port = RequiredPort(ModelReply(refusal=refusal, usage={"input_tokens": 7, "output_tokens": 2}))
    with leased_run(store, rid) as guard:
        first = invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
        second = invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    assert first.refusal == second.refusal == refusal and port.calls == 1
    assert row(store, rid)["status"] == "completed"


def test_saved_required_premature_final_is_rejected_without_dispatch(tmp_path):
    store, sid, rid = setup(tmp_path)
    ledger, port = InvocationLedger(store), RequiredPort()
    with leased_run(store, rid) as guard:
        invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
        before = row(store, rid)
        # Simulate a persisted receipt from an earlier writer lacking the required-action gate.
        raw = json.loads(before["receipt"])
        raw.update(tool_calls=[], refusal=None, content="legacy premature final")
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE model_invocations_v1 SET receipt=? WHERE run_id=?",
                         (json.dumps(raw), rid))
        with pytest.raises(ProviderFailure) as error:
            invoke(InvocationLedger(store), sid, rid, port, guard, tool_choice=REQUIRED)
        failed = row(store, rid)
        with pytest.raises(RuntimeError, match="cannot be replayed"):
            invoke(ledger, sid, rid, port, guard, tool_choice=REQUIRED)
    assert error.value.code == "required_action_missing" and port.calls == 1
    assert failed["status"] == "failed" and failed["usage"] == before["usage"]
    assert row(store, rid) == failed and ledger.usage(rid)["requests"] == 1


def test_openai_compatible_rejects_required_before_public_projection_or_sdk(monkeypatch):
    import adj_partner.models.openai_compatible as module

    # No configured client is needed for a capability rejection; touching one is a bug.
    port = OpenAICompatiblePort.__new__(OpenAICompatiblePort)
    port._model, port.secrets = "fixture-model", ()
    monkeypatch.setattr(module, "public_request", lambda *a, **k: pytest.fail("projection entered"))
    with pytest.raises(ProviderFailure) as error:
        port.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout_seconds=10, tool_choice=REQUIRED)
    assert error.value.code == "unsupported_tool_choice"
    assert error.value.cost_unknown is False and error.value.usage == {}
    assert port.metadata == {"provider_kind": "openai_compatible", "model": "fixture-model"}


def test_openai_compatible_explicit_auto_does_not_add_sdk_argument():
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(content=json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}}).encode())

    port = OpenAICompatiblePort.__new__(OpenAICompatiblePort)
    port._model, port.secrets = "fixture-model", ()
    port._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))
    result = port.complete(MESSAGES, [], max_output_tokens=100, timeout_seconds=10, tool_choice="auto")
    assert result.content == "answer" and len(seen) == 1
    assert "tool_choice" not in seen[0] and "tools" not in seen[0]
