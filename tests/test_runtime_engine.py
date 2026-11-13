"""Real journal/executor loop with explicit offline model decisions."""

import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall, ToolResult
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor, ToolSpec
from adj_partner.tools.management import register_management_tools


class Decisions:
    """Test-only scripted decisions, not represented as a live model."""
    secrets = ()
    metadata = {"provider_kind": "test_scripted", "model": "fixture"}
    def __init__(self, replies): self.replies, self.requests = list(replies), []
    def complete(self, messages, tools, **limits):
        self.requests.append({"messages": messages, "tools": tools, **limits})
        reply = self.replies.pop(0)
        if callable(reply): return reply(messages, tools, **limits)
        if isinstance(reply, Exception): raise reply
        return reply


def reply(text="", calls=(), **kwargs):
    return ModelReply(content=text, tool_calls=list(calls), usage={"input_tokens": 50, "output_tokens": 20},
                      finish_reason="tool_calls" if calls else "stop", **kwargs)


def setup(tmp_path, replies, *, budget=None, allowed=None, approve=False):
    assert importlib.util.find_spec("adj_partner.runtime.engine"), "single durable engine must exist"
    from adj_partner.runtime.engine import AgentEngine
    store, catalog, effects = Store(tmp_path / "state"), ToolCatalog(), []
    def echo(ctx, text):
        effects.append(text)
        return ToolResult(data={"text": text})
    catalog.register(ToolSpec("echo", "Echo", {"type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False}, requires_approval=approve), echo)
    register_management_tools(catalog, store)
    port = Decisions(replies)
    engine = AgentEngine(store, catalog, port)
    config = SessionConfig(enabled_tools=catalog.names(), allowed_tools=catalog.names() if allowed is None else allowed,
                           budget=budget or Budget()).to_dict()
    sid = store.create_session(config)["id"]
    return store, engine, sid, port, effects


def test_ordinary_chat_has_no_research_pipeline_and_followup_keeps_history(tmp_path):
    store, engine, sid, port, _ = setup(tmp_path, [reply("你好"), reply("follow up")])
    first = engine.start(sid, "hello")
    done = engine.resume(first["id"])
    assert done["status"] == "succeeded" and done["state"]["delivery"]["content"] == "你好"
    second = engine.start(sid, "explain")
    assert second["task_id"] != first["task_id"]
    assert engine.resume(second["id"])["status"] == "succeeded"
    assert any(message.get("content") == "你好" for message in port.requests[1]["messages"])
    events = store.events(sid, first["id"])
    assert any(event["kind"] == "model.requested" for event in events)
    assert any(event["kind"] == "model.replied" for event in events)


def test_detach_after_request_checkpoint_does_not_issue_or_charge_request(tmp_path, monkeypatch):
    store, engine, sid, port, _ = setup(tmp_path, [reply("after reconnect")])
    stop = threading.Event()
    original = engine.journal.checkpoint
    def detach(*args, **kwargs):
        value = original(*args, **kwargs)
        if value["state"].get("phase") == "requesting": stop.set()
        return value
    monkeypatch.setattr(engine.journal, "checkpoint", detach)
    rid = engine.start(sid, "hello")["id"]
    waiting = engine.resume(rid, stop_event=stop)
    assert waiting["status"] == "waiting" and waiting["state"]["phase"] == "ready"
    assert not port.requests and not waiting["usage"] and waiting["state"]["steps"] == 0
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    assert engine.resume(rid)["status"] == "succeeded" and len(port.requests) == 1


@pytest.mark.parametrize("after_claim", [False, True])
def test_detach_before_effect_is_resumable_and_not_uncertain(tmp_path, monkeypatch, after_claim):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "effect"})]), reply("reconnected")])
    stop = threading.Event()
    target, name = (store, "claim_action") if after_claim else (engine.executor, "execute")
    original = getattr(target, name)
    def detach(*args, **kwargs):
        if not after_claim: stop.set()
        value = original(*args, **kwargs)
        if after_claim: stop.set()
        return value
    monkeypatch.setattr(target, name, detach)
    rid = engine.start(sid, "effect")["id"]
    waiting = engine.resume(rid, stop_event=stop)
    assert waiting["status"] == "waiting" and waiting["state"]["phase"] == "tools"
    assert not effects and len(port.requests) == 1
    if after_claim:
        action = store.get_action(rid, "a")
        assert action["finished_at"] and action["result"]["error"]["code"] == "client_detached_not_invoked"
    monkeypatch.setattr(target, name, original)
    assert engine.resume(rid)["status"] == "succeeded"
    assert effects == ([] if after_claim else ["effect"])
    assert len(port.requests) == 2


def test_detach_after_approval_consumption_records_known_not_invoked(tmp_path, monkeypatch):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "effect"})]), reply("replan")], approve=True)
    rid = engine.start(sid, "effect")["id"]
    assert engine.resume(rid)["status"] == "waiting"
    engine.executor.decide_approval(sid, rid, "a", True)
    stop, original = threading.Event(), store.append_event
    def detach(sid, kind, *args, **kwargs):
        value = original(sid, kind, *args, **kwargs)
        if kind == "tool.approval_consumed": stop.set()
        return value
    monkeypatch.setattr(store, "append_event", detach)
    assert engine.resume(rid, stop_event=stop)["status"] == "waiting" and not effects
    assert store.get_action(rid, "a")["result"]["error"]["code"] == "client_detached_not_invoked"
    assert engine.resume(rid)["status"] == "succeeded" and not effects


def test_batched_tools_have_original_ids_and_one_complete_protocol_group(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "one"}),
        ToolCall("b", "echo", {"text": "two"})]), reply("finished")])
    rid = engine.start(sid, "test tools")["id"]
    assert engine.resume(rid)["status"] == "succeeded" and effects == ["one", "two"]
    assert [a["call_id"] for a in store.list_actions(rid)] == ["a", "b"]
    messages = port.requests[1]["messages"]
    assistant = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
    assert [m["tool_call_id"] for m in messages[assistant+1:assistant+3]] == ["a", "b"]


def test_tool_enable_does_not_authorize_stale_later_batch_call(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("enable", "tools_enable", {"name": "echo"}),
        ToolCall("stale", "echo", {"text": "must not happen"})]), reply("replan")])
    rid = engine.start(sid, "test revisions")["id"]
    engine.resume(rid)
    assert effects == []
    assert store.get_action(rid, "stale")["result"]["error"]["code"] == "stale_revision"


def test_host_approval_suspends_and_reopen_resumes_without_duplicate_decision(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "yes"})]), reply("ok")], approve=True)
    rid = engine.start(sid, "needs approval")["id"]
    assert engine.resume(rid)["status"] == "waiting" and effects == []
    assert len(port.requests) == 1 and not any(m.get("tool_calls") for m in store.messages(sid))
    engine.executor.decide_approval(sid, rid, "a", approve=True)
    from adj_partner.runtime.engine import AgentEngine
    reopened = AgentEngine(Store(store.root), engine.catalog, port)
    assert reopened.resume(rid)["status"] == "succeeded" and effects == ["yes"]
    assert len(port.requests) == 2


def test_real_user_answer_is_only_clarification_resolution(tmp_path):
    store, engine, sid, port, _ = setup(tmp_path, [reply(calls=[ToolCall("ask", "ask_user", {"question": "Which sample?"})]), reply("thanks")])
    rid = engine.start(sid, "analyze")["id"]
    waiting = engine.resume(rid)
    assert waiting["status"] == "waiting" and waiting["state"]["waiting"]["kind"] == "user"
    pending = engine.journal.pending_inputs(rid)
    engine.journal.answer(rid, pending[0]["id"], "sample A")
    assert engine.resume(rid)["status"] == "succeeded"
    action = store.get_action(rid, "ask")
    assert action["result"]["data"]["answer"] == "sample A" and len(port.requests) == 2


def test_unknown_started_effect_is_not_replayed(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "effect"})])])
    rid = engine.start(sid, "effect")["id"]
    def interrupted(call, *args, **kwargs):
        store.claim_action(rid, call.id, call.name, call.arguments, 0)
        raise KeyboardInterrupt("simulated process interruption")
    engine.executor.execute = interrupted
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    from adj_partner.runtime.engine import AgentEngine
    reopened = AgentEngine(Store(store.root), engine.catalog, port)
    done = reopened.resume(rid)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "uncertain_tool_effect"
    assert effects == [] and len(port.requests) == 1


def test_interrupted_model_request_is_unknown_not_automatically_reissued(tmp_path):
    def interrupted(*args, **kwargs): raise KeyboardInterrupt("crash during request")
    store, engine, sid, port, _ = setup(tmp_path, [interrupted])
    rid = engine.start(sid, "hello")["id"]
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    done = engine.resume(rid)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "interrupted_model_request"
    assert done["usage"]["unknown_requests"] == 1 and len(port.requests) == 1


@pytest.mark.parametrize("bad,reason", [(ModelReply(content="", finish_reason="stop"), "empty_reply"),
    (ModelReply(content="partial", finish_reason="length"), "model_length"),
    (ModelReply(content="filtered", finish_reason="content_filter"), "model_content_filter")])
def test_non_successful_model_stops_are_explicit(tmp_path, bad, reason):
    store, engine, sid, port, _ = setup(tmp_path, [bad])
    done = engine.resume(engine.start(sid, "question")["id"])
    assert done["status"] == "failed" and done["state"]["stop_reason"] == reason
    assert len(port.requests) == 1


def test_strict_task_cannot_pass_with_only_free_text(tmp_path):
    store, engine, sid, _, _ = setup(tmp_path, [reply("Trust me, perfect evidence")])
    done = engine.resume(engine.start(sid, "research", acceptance={"require_report": True, "require_evidence": True})["id"])
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "acceptance_not_met"


def test_step_budget_prevents_another_model_request(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "one"})]), reply("never")],
        budget=Budget(max_steps=1))
    done = engine.resume(engine.start(sid, "test")["id"])
    assert done["status"] == "budget_exhausted" and len(port.requests) == 1 and effects == ["one"]


def test_output_allowance_is_capped_by_remaining_budget_and_actual_overrun_stops_effects(tmp_path):
    bad = ModelReply(tool_calls=[ToolCall("a", "echo", {"text": "never"})], finish_reason="tool_calls",
                     usage={"input_tokens": 10, "output_tokens": 21})
    store, engine, sid, port, effects = setup(tmp_path, [bad], budget=Budget(max_output_tokens=20))
    done = engine.resume(engine.start(sid, "test")["id"])
    assert port.requests[0]["max_output_tokens"] == 20
    assert done["status"] == "budget_exhausted" and effects == []


def test_known_provider_secrets_primed_before_user_input_and_requests(tmp_path):
    store, engine, sid, port, _ = setup(tmp_path, [reply("okay")])
    port.secrets = ("private-token",)
    done = engine.resume(engine.start(sid, "private-token")["id"])
    assert "private-token" not in json.dumps([store.messages(sid), store.events(sid), port.requests])


def test_concurrent_resume_has_only_one_model_request_and_cancel_is_immediate(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return reply(calls=[ToolCall("a", "echo", {"text": "must not run"})])
    store, engine, sid, port, effects = setup(tmp_path, [slow])
    rid = engine.start(sid, "slow")["id"]
    from adj_partner.runtime.journal import RunBusy
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(engine.resume, rid)
        assert entered.wait(5)
        with pytest.raises(RunBusy): engine.resume(rid)
        engine.journal.cancel(rid)
        assert store.get_run(rid)["status"] == "cancelled"
        release.set()
        assert future.result(timeout=5)["status"] == "cancelled"
    assert effects == [] and len(port.requests) == 1


def test_closing_client_yields_without_hidden_continued_model_calls(tmp_path):
    stop = threading.Event()
    def closing(*args, **kwargs):
        stop.set()
        return reply(calls=[ToolCall("a", "echo", {"text": "later"})])
    store, engine, sid, port, effects = setup(tmp_path, [closing, reply("done")])
    rid = engine.start(sid, "test")["id"]
    waiting = engine.resume(rid, stop_event=stop)
    assert waiting["status"] == "waiting" and effects == [] and len(port.requests) == 1
    stop.clear()
    assert engine.resume(rid, stop_event=stop)["status"] == "succeeded" and effects == ["later"]


@pytest.mark.parametrize("finish", [None, 1, [], ""])
def test_malformed_finish_reason_is_a_durable_protocol_failure(tmp_path, finish):
    bad = ModelReply(content="text", finish_reason=finish)
    store, engine, sid, _, _ = setup(tmp_path, [bad])
    done = engine.resume(engine.start(sid, "test")["id"])
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "model_protocol"


def test_reopened_engine_uses_actual_store_for_late_secret_report_gate(tmp_path):
    store, engine, sid, port, _ = setup(tmp_path, [reply("safe")])
    from adj_partner.runtime.engine import AgentEngine
    reopened_store = Store(store.root, secrets=["newly-known-secret"])
    reopened = AgentEngine(reopened_store, engine.catalog, port, research=engine.research, executor=engine.executor)
    assert reopened.research.store is reopened_store and reopened.executor.store is reopened_store


def test_wrong_database_injected_services_are_rejected(tmp_path):
    store, engine, sid, port, _ = setup(tmp_path, [])
    from adj_partner.runtime.engine import AgentEngine
    with pytest.raises(ValueError): AgentEngine(Store(tmp_path / "other"), engine.catalog, port, research=engine.research)


def test_completed_action_is_not_replayed_after_checkpoint_crash(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "once"})]), reply("done")])
    rid = engine.start(sid, "effect")["id"]
    execute = engine.executor.execute
    def crash(*args, **kwargs):
        execute(*args, **kwargs)
        raise KeyboardInterrupt("after action committed")
    engine.executor.execute = crash
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    engine.executor.execute = execute
    assert engine.resume(rid)["status"] == "succeeded" and effects == ["once"]


def test_refusal_does_not_execute_a_simultaneous_proposed_action(tmp_path):
    bad = ModelReply(content="", refusal="I cannot help", tool_calls=[ToolCall("a", "echo", {"text": "never"})], finish_reason="stop")
    store, engine, sid, _, effects = setup(tmp_path, [bad])
    done = engine.resume(engine.start(sid, "test")["id"])
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "model_refusal" and effects == []


def test_duplicate_call_ids_never_execute_and_known_usage_is_preserved(tmp_path):
    bad = reply(calls=[ToolCall("same", "echo", {"text": "one"}), ToolCall("same", "echo", {"text": "two"})])
    store, engine, sid, _, effects = setup(tmp_path, [bad])
    done = engine.resume(engine.start(sid, "test")["id"])
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "model_protocol"
    assert done["usage"]["input_tokens"] == 50 and effects == []


@pytest.mark.parametrize("case,reason", [("refusal", "model_refusal"),
    ("sensitive", "sensitive_tool_arguments_not_replayable"), ("over_budget", "budget_exhausted")])
def test_crash_after_reply_checkpoint_never_bypasses_pre_effect_gates(tmp_path, case, reason):
    decision = reply(calls=[ToolCall("a", "echo", {"text": "must-not-run"})])
    if case == "refusal": decision.refusal = "Cannot assist"
    if case == "sensitive": decision.tool_calls[0].arguments["text"] = "sensitive-credential"
    if case == "over_budget": decision.usage["output_tokens"] = 24001
    store, engine, sid, port, effects = setup(tmp_path, [decision, reply("must not be called")])
    if case == "sensitive": port.secrets = ("sensitive-credential",)
    rid = engine.start(sid, "test gates")["id"]
    checkpoint = engine.journal.checkpoint
    def crash(*args, **kwargs):
        result = checkpoint(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed": raise KeyboardInterrupt("after reply checkpoint")
        return result
    engine.journal.checkpoint = crash
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    engine.journal.checkpoint = checkpoint
    done = engine.resume(rid)
    assert done["status"] in {"failed", "budget_exhausted"} and done["state"]["stop_reason"] == reason
    assert effects == [] and len(port.requests) == 1


def test_crash_after_truncated_reply_retries_once_without_running_its_tool_calls(tmp_path):
    first = reply(calls=[ToolCall("a", "echo", {"text": "must-not-run"})])
    second = reply(calls=[ToolCall("b", "echo", {"text": "must-not-run-either"})])
    first.finish_reason = second.finish_reason = "length"
    store, engine, sid, port, effects = setup(tmp_path, [first, second, reply("must not be called")])
    rid = engine.start(sid, "test gates")["id"]
    checkpoint = engine.journal.checkpoint
    def crash(*args, **kwargs):
        result = checkpoint(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed": raise KeyboardInterrupt("after reply checkpoint")
        return result
    engine.journal.checkpoint = crash
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    engine.journal.checkpoint = checkpoint
    done = engine.resume(rid)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "model_length"
    assert effects == [] and len(port.requests) == 2
    assert done["state"]["output_length_continuations"] == 1


@pytest.mark.parametrize("changes", [{"acceptance": {"require_report": True, "require_evidence": True}}, {"goal": "Different task"}])
def test_current_task_changes_during_request_invalidate_old_reply(tmp_path, changes):
    store, engine, sid, port, effects = setup(tmp_path, [])
    rid = engine.start(sid, "Original")["id"]
    tid = store.get_run(rid)["task_id"]
    def changed(*args, **kwargs):
        store.update_task(tid, changes)
        return reply("Old unvalidated answer")
    port.replies = [changed]
    done = engine.resume(rid)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "task_changed"
    assert not done["state"].get("delivery")


def test_successful_report_gate_that_crosses_wall_budget_cannot_succeed(tmp_path):
    store, engine, sid, _, _ = setup(tmp_path, [reply("ready")], budget=Budget(wall_time_seconds=.15))
    rid = engine.start(sid, "Proposal", acceptance={"require_report": True})["id"]
    claim = engine.research.record_claim(sid, rid, text="A proposal", kind="proposal", evidence_ids=[])
    report = engine.research.create_report(sid, rid, title="Proposal", sections=[{"heading": "Finding", "claim_ids": [claim["id"]]}], limitations=["Unverified"])
    engine.research.validate_report(sid, rid, report["id"])
    accept = engine.research.accepted_report
    def delayed(*args, **kwargs):
        value = accept(*args, **kwargs)
        time.sleep(.2)
        return value
    engine.research.accepted_report = delayed
    done = engine.resume(rid)
    assert done["status"] == "budget_exhausted" and not done["state"].get("delivery")


def test_lease_lost_at_result_entry_does_not_start_effect(tmp_path):
    store, engine, sid, _, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "never"})])])
    rid = engine.start(sid, "test")["id"]
    result = engine._result
    owners = []
    def expired(*args, **kwargs):
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE runtime_leases SET expires=0,expires_monotonic=0 WHERE run_id=?", (rid,))
        owners.append(engine.journal.acquire(rid))
        return result(*args, **kwargs)
    engine._result = expired
    from adj_partner.runtime.journal import LeaseLost
    with pytest.raises(LeaseLost): engine.resume(rid)
    assert effects == [] and store.list_actions(rid) == []
    engine.journal.release(rid, owners[0])


def test_lease_lost_during_executor_validation_cannot_claim_or_invoke(tmp_path):
    store, engine, sid, _, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "never"})])])
    rid = engine.start(sid, "test")["id"]
    validate, owners = engine.executor._arguments, []
    def expire(*args, **kwargs):
        answer = validate(*args, **kwargs)
        with store.transaction(write=True) as conn: conn.execute(
            "UPDATE runtime_leases SET expires=0,expires_monotonic=0 WHERE run_id=?", (rid,))
        owners.append(engine.journal.acquire(rid))
        return answer
    engine.executor._arguments = expire
    from adj_partner.runtime.journal import LeaseLost
    with pytest.raises(LeaseLost): engine.resume(rid)
    assert effects == [] and store.list_actions(rid) == []
    engine.journal.release(rid, owners[0])


def test_cancellation_after_request_checkpoint_does_not_send_the_model_request(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [reply("must not send")])
    rid = engine.start(sid, "test")["id"]
    checkpoint = engine.journal.checkpoint
    def cancel_at_boundary(*args, **kwargs):
        value = checkpoint(*args, **kwargs)
        if kwargs.get("state", {}).get("phase") == "requesting": engine.journal.cancel(rid)
        return value
    engine.journal.checkpoint = cancel_at_boundary
    assert engine.resume(rid)["status"] == "cancelled"
    assert port.requests == [] and effects == []


@pytest.mark.parametrize("phase", ["reply", "tools"])
def test_late_secret_registration_cannot_authorize_previously_saved_arguments(tmp_path, phase):
    stop = threading.Event()
    store, engine, sid, port, effects = setup(tmp_path, [reply(calls=[ToolCall("a", "echo", {"text": "later-known-secret"})]), reply("never")])
    rid = engine.start(sid, "test")["id"]
    checkpoint = engine.journal.checkpoint
    def detach(*args, **kwargs):
        result = checkpoint(*args, **kwargs)
        if kwargs.get("state", {}).get("phase") == phase: stop.set()
        return result
    engine.journal.checkpoint = detach
    assert engine.resume(rid, stop_event=stop)["status"] == "waiting"
    assert effects == []
    port.secrets = ("later-known-secret",)
    engine.journal.checkpoint = checkpoint
    stop.clear()
    done = engine.resume(rid, stop_event=stop)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "sensitive_tool_arguments_not_replayable"
    assert effects == [] and len(port.requests) == 1
