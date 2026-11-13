"""Real catalog/policy/executor tests: no live models or simulated storage."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from threading import Barrier, Event

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall, ToolResult
from adj_partner.storage import Store


def api():
    assert importlib.util.find_spec("adj_partner.tools"), "shared tool module must exist"
    from adj_partner import tools
    return tools


def make_scope(tmp_path, names=("echo",), *, enabled=None, network=False, store=None):
    store = store or Store(tmp_path / "state")
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    config = SessionConfig(allowed_tools=list(names), enabled_tools=list(names if enabled is None else enabled),
                           allow_network=network, allowed_roots=[str(root)])
    session = store.create_session(config.to_dict())
    task = store.create_task(session["id"], "Exercise tools")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    return store, session["id"], run["id"], root


def echo_catalog(handler=None, **spec_kwargs):
    t = api()
    catalog = t.ToolCatalog()
    catalog.register(t.ToolSpec("echo", "Echo text", {
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False,
    }, **spec_kwargs), handler or (lambda context, text: ToolResult(data={"text": text})))
    return catalog


def test_definition_is_defensively_immutable_and_same_schema_is_exposed():
    t = api()
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    spec = t.ToolSpec("counter", "Count", schema, output_schema={"type": "integer"})
    digest = spec.definition_hash
    schema["properties"]["n"]["type"] = "string"
    spec.input_schema["properties"]["n"]["type"] = "boolean"
    spec.model_schema()["function"]["parameters"]["properties"].clear()
    assert spec.input_schema["properties"]["n"]["type"] == "integer"
    assert spec.definition_hash == digest
    assert spec.model_schema() == {"type": "function", "function": {
        "name": "counter", "description": "Count", "parameters": spec.input_schema}}
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "different"
    assert t.ToolSpec("counter", "Count", spec.input_schema, {"type": "integer"}).definition_hash == digest
    assert t.ToolSpec("counter", "Count", spec.input_schema, {"type": "string"}).definition_hash != digest


@pytest.mark.parametrize("name", ["", "a.b", "a b", "x" * 65, "中文"])
def test_model_unsafe_names_rejected(name):
    with pytest.raises(ValueError):
        api().ToolSpec(name, "Description", {"type": "object"})


@pytest.mark.parametrize("schema", [{"type": "nonsense"}, {"$ref": "https://invalid.test/schema"},
    {"properties": {"x": {"$dynamicRef": "https://invalid.test/schema"}}}])
def test_invalid_or_remote_schemas_rejected(schema):
    with pytest.raises(ValueError):
        api().ToolSpec("bad", "Description", schema)


def test_local_schema_refs_accepted_and_catalog_duplicates_rejected():
    t = api()
    spec = t.ToolSpec("echo", "Echo with local reference", {
        "type": "object", "$defs": {"text": {"type": "string"}},
        "properties": {"text": {"$ref": "#/$defs/text"}}})
    catalog = t.ToolCatalog()
    handler = lambda context, **kwargs: ToolResult(data=kwargs)
    catalog.register(spec, handler)
    assert catalog.get("echo").handler is handler
    with pytest.raises(ValueError, match="Duplicate"):
        catalog.register(spec, handler)
    assert catalog.names() == ["echo"]
    assert catalog.search("local")[0]["name"] == "echo"
    assert catalog.search("not present") == []
    with pytest.raises(ValueError):
        catalog.search("", limit=-1)


def test_sessions_expose_only_known_enabled_authorized_tools_and_persist_changes(tmp_path):
    t = api()
    catalog = echo_catalog()
    store, session, _, _ = make_scope(tmp_path, names=("echo", "unknown"), enabled=("unknown",))
    _, second, _, _ = make_scope(tmp_path, names=(), store=store)
    toolset = t.SessionToolset(store, catalog, session)
    assert toolset.snapshot() == {"revision": 0, "tools": [], "definition_hashes": {}}
    toolset.enable("echo")
    snap = toolset.snapshot()
    assert snap["revision"] == 1
    assert snap["tools"][0]["function"]["name"] == "echo"
    assert snap["definition_hashes"] == {"echo": catalog.get("echo").spec.definition_hash}
    assert t.SessionToolset(store, catalog, second).snapshot()["tools"] == []
    with pytest.raises(PermissionError):
        t.SessionToolset(store, catalog, second).enable("echo")
    with pytest.raises(KeyError):
        toolset.enable("unknown")
    toolset.disable("echo")
    assert toolset.snapshot()["tools"] == []
    assert store.get_session(session)["config"]["allowed_tools"] == ["echo", "unknown"]
    assert any(event["kind"] == "toolset.changed" for event in store.events(session))


def test_stale_toolset_cannot_overwrite_concurrent_session_change(tmp_path):
    t = api()
    store, session, _, _ = make_scope(tmp_path, enabled=())
    toolset = t.SessionToolset(store, echo_catalog(), session)
    assert toolset.snapshot()["revision"] == 0
    current = store.get_session(session)
    current["config"]["instructions"] = "Host edit"
    store.update_session(session, current["config"], 0)
    with pytest.raises(ValueError, match="Stale"):
        toolset.enable("echo")
    assert store.get_session(session)["config"]["instructions"] == "Host edit"


def test_context_paths_reject_escape_symlinks_and_credentials(tmp_path):
    t = api()
    store, session, run, root = make_scope(tmp_path)
    ordinary = root / "notes.txt"
    ordinary.write_text("safe")
    context = t.ToolContext(store, session, run, 0)
    assert context.check_path(ordinary) == ordinary
    assert context.check_path(root / "new.txt", write=True, must_exist=False) == root / "new.txt"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    for path in (outside, root / ".." / "outside", root / "escape" / "new.txt",
                 root / ".env", root / ".env.production", root / ".ssh" / "config",
                 root / "id_rsa", root / "private.pem", root / "credentials.json"):
        with pytest.raises(PermissionError):
            context.check_path(path, must_exist=False)
    (root / "safe-link").symlink_to(ordinary)
    assert context.check_path(root / "safe-link") == ordinary
    with pytest.raises(FileNotFoundError):
        context.check_path(root / "absent.txt")
    _, other, other_run, _ = make_scope(tmp_path, store=store)
    with pytest.raises(ValueError, match="belong"):
        t.ToolContext(store, other, run, 0)
    signal = Event()
    signal.set()
    with pytest.raises(t.ToolCancelled):
        t.ToolContext(store, other, other_run, 0, signal).check_cancelled()


def executor_api():
    t = api()
    assert hasattr(t, "ToolExecutor"), "durable shared executor must exist"
    return t


def test_executor_calls_once_and_reuses_completed_result(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    catalog = echo_catalog(lambda context, text: seen.append((context.session_id, text)) or ToolResult(data=text))
    executor = t.ToolExecutor(store, catalog)
    call = ToolCall("c1", "echo", {"text": "hello"})
    first = executor.execute(call, session, run, expected_revision=0,
                             expected_definition_hash=catalog.get("echo").spec.definition_hash)
    assert first.status == "ok" and first.data == "hello"
    assert executor.execute(call, session, run).to_dict() == first.to_dict()
    assert seen == [(session, "hello")]
    assert store.get_action(run, "c1")["finished_at"]
    assert any(event["kind"] == "tool.attempt" for event in store.events(session))


@pytest.mark.parametrize(("config_change", "code"), [
    ({"enabled_tools": []}, "tool_disabled"), ({"allowed_tools": []}, "tool_not_allowed"),
])
def test_policy_denials_are_durable_without_invocation(tmp_path, config_change, code):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    current = store.get_session(session)
    current["config"].update(config_change)
    store.update_session(session, current["config"], 0)
    seen = []
    result = t.ToolExecutor(store, echo_catalog(lambda *a, **k: seen.append(1))).execute(
        ToolCall("denied", "echo", {"text": "x"}), session, run)
    assert result.status == "denied" and result.error["code"] == code
    assert not seen
    assert store.get_action(run, "denied")["result"] == result.to_dict()


def test_unknown_network_stale_and_schema_drift_fail_without_execution(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    executor = t.ToolExecutor(store, echo_catalog(network=True))
    for identifier, name, kwargs, code in [
        ("unknown", "missing", {}, "unknown_tool"),
        ("network", "echo", {}, "network_denied"),
        ("revision", "echo", {"expected_revision": 10}, "stale_revision"),
        ("hash", "echo", {"expected_definition_hash": "old"}, "schema_drift"),
    ]:
        result = executor.execute(ToolCall(identifier, name, {"text": "x"}), session, run, **kwargs)
        assert result.status == "denied" and result.error["code"] == code
        assert store.get_action(run, identifier)["finished_at"]


def test_input_output_and_json_failures_are_explicit(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    executor = t.ToolExecutor(store, echo_catalog(output_schema={"type": "integer"}))
    result = executor.execute(ToolCall("input", "echo", {"text": 3}), session, run)
    assert result.error["code"] == "input_schema"
    result = executor.execute(ToolCall("output", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_schema"
    invalid = t.ToolExecutor(store, echo_catalog(lambda context, text: ToolResult(data=float("nan"))))
    result = invalid.execute(ToolCall("json", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_json"
    assert all(action["finished_at"] for action in store.list_actions(run))


def test_started_actions_and_changed_call_ids_are_never_replayed(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    store.claim_action(run, "started", "echo", {"text": "x"}, 0)
    seen = []
    executor = t.ToolExecutor(store, echo_catalog(lambda *a, **k: seen.append(1)))
    result = executor.execute(ToolCall("started", "echo", {"text": "x"}), session, run)
    assert result.status == "uncertain" and result.error["code"] == "action_uncertain"
    result = executor.execute(ToolCall("started", "echo", {"text": "changed"}), session, run)
    assert result.error["code"] == "call_id_conflict"
    assert seen == [] and store.get_action(run, "started")["finished_at"] is None


def test_concurrent_execution_has_one_atomic_winner(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    entered, release = Event(), Event()
    seen = []
    def handler(context, text):
        seen.append(text)
        entered.set()
        assert release.wait(3), "test must release the bounded adapter"
        return ToolResult(data=text)
    catalog = echo_catalog(handler)
    call = ToolCall("race", "echo", {"text": "one"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(t.ToolExecutor(store, catalog).execute, call, session, run)
        assert entered.wait(3)
        second = pool.submit(t.ToolExecutor(Store(store.root), catalog).execute, call, session, run)
        loser = second.result(timeout=3)
        release.set()
        winner = first.result(timeout=3)
    assert loser.status == "uncertain" and winner.status == "ok"
    assert seen == ["one"]


def test_exceptions_cancellation_and_scope_are_structured(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    def broken(context, text):
        raise RuntimeError("provider exploded")
    executor = t.ToolExecutor(store, echo_catalog(broken))
    result = executor.execute(ToolCall("fail", "echo", {"text": "x"}), session, run)
    assert result.error == {"code": "provider_error", "message": "RuntimeError: provider exploded"}
    signal = Event()
    signal.set()
    result = executor.execute(ToolCall("cancel", "echo", {"text": "x"}), session, run, cancel_event=signal)
    assert result.status == "cancelled" and result.error["code"] == "cancelled"
    _, other, _, _ = make_scope(tmp_path, store=store)
    result = executor.execute(ToolCall("scope", "echo", {"text": "x"}), other, run)
    assert result.status == "denied" and result.error["code"] == "scope_denied"
    assert store.get_action(run, "scope") is None


def test_malformed_call_is_rejected_and_trace_is_sanitized(tmp_path):
    t = executor_api()
    store = Store(tmp_path / "state", secrets=["super-private-value"])
    store, session, run, _ = make_scope(tmp_path, store=store)
    seen = []
    executor = t.ToolExecutor(store, echo_catalog(lambda context, text: seen.append(text) or ToolResult(data={"text": text})))
    invalid = executor.execute(ToolCall("../bad", "echo", {"text": "x"}), session, run)
    assert invalid.error["code"] == "invalid_call"
    result = executor.execute(ToolCall("secret", "echo", {"text": "super-private-value"}), session, run)
    assert seen == ["super-private-value"]  # Original ephemeral arguments, not redacted input.
    assert result.data == {"text": "[REDACTED]"}  # Model/UI share the safe durable result.
    assert "super-private-value" not in json.dumps(store.events(session))
    assert store.get_action(run, "secret")["args"]["text"] == "[REDACTED]"
    assert any(event["kind"] == "tool.rejected" for event in store.events(session))


def approval_scope(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    catalog = echo_catalog(lambda context, text: seen.append(text) or ToolResult(data=text), requires_approval=True)
    executor = t.ToolExecutor(store, catalog)
    return t, store, session, run, seen, catalog, executor


def test_approval_is_exact_host_only_one_shot_and_durable(tmp_path):
    t, store, session, run, seen, catalog, executor = approval_scope(tmp_path)
    call = ToolCall("approval", "echo", {"text": "once"})
    assert executor.execute(call, session, run).status == "approval_required"
    assert not seen and not store.get_action(run, call.id)["finished_at"]
    pending = executor.pending_approvals(session)
    assert len(pending) == 1 and pending[0]["call_id"] == call.id
    assert pending[0]["definition_hash"] == catalog.get("echo").spec.definition_hash
    assert executor.resume_approved(session, run, call.id).status == "approval_required"
    executor.decide_approval(session, run, call.id, True)
    restarted = t.ToolExecutor(Store(store.root), catalog)
    assert restarted.resume_approved(session, run, call.id).data == "once"
    assert restarted.resume_approved(session, run, call.id).data == "once"
    assert restarted.execute(call, session, run).data == "once"
    assert seen == ["once"] and executor.pending_approvals(session) == []
    with pytest.raises(ValueError):
        restarted.decide_approval(session, run, call.id, True)


def test_approval_denial_staleness_and_wrong_scope_never_execute(tmp_path):
    t, store, session, run, seen, catalog, executor = approval_scope(tmp_path)
    for identifier in ("denied", "stale"):
        executor.execute(ToolCall(identifier, "echo", {"text": identifier}), session, run)
    executor.decide_approval(session, run, "denied", False)
    assert executor.resume_approved(session, run, "denied").error["code"] == "approval_denied"
    executor.decide_approval(session, run, "stale", True)
    current = store.get_session(session)
    current["config"]["allowed_tools"] = []
    store.update_session(session, current["config"], 0)
    assert executor.resume_approved(session, run, "stale").error["code"] == "stale_approval"
    _, other, _, _ = make_scope(tmp_path, store=store)
    with pytest.raises(ValueError):
        executor.decide_approval(other, run, "stale", True)
    assert not seen


def test_concurrent_approval_resume_has_one_consumer(tmp_path):
    t, store, session, run, seen, catalog, executor = approval_scope(tmp_path)
    executor.execute(ToolCall("approval", "echo", {"text": "once"}), session, run)
    executor.decide_approval(session, run, "approval", True)
    barrier = Barrier(2)
    def resume():
        other = t.ToolExecutor(Store(store.root), catalog)
        barrier.wait(timeout=3)
        return other.resume_approved(session, run, "approval")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=5) for future in [pool.submit(resume), pool.submit(resume)]]
    assert all(result.status in {"ok", "uncertain"} for result in results)
    assert seen == ["once"]


def test_consumed_approval_crash_is_not_a_replay_grant(tmp_path):
    t, store, session, run, seen, catalog, executor = approval_scope(tmp_path)
    executor.execute(ToolCall("approval", "echo", {"text": "once"}), session, run)
    executor.decide_approval(session, run, "approval", True)
    # A durable consumed decision represents death after permission consumption.
    with store.transaction(write=True) as conn:
        conn.execute("UPDATE tool_approvals SET status='consumed' WHERE run_id=? AND call_id=?", (run, "approval"))
    assert t.ToolExecutor(Store(store.root), catalog).resume_approved(session, run, "approval").status == "uncertain"
    assert not seen


def test_approval_recovery_cannot_execute_sanitized_arguments(tmp_path):
    t, store, session, run, seen, catalog, executor = approval_scope(tmp_path)
    store.register_secrets(["original-private-input"])
    executor.execute(ToolCall("secret", "echo", {"text": "original-private-input"}), session, run)
    executor.decide_approval(session, run, "secret", True)
    restarted = t.ToolExecutor(Store(store.root), catalog)
    result = restarted.resume_approved(session, run, "secret")
    assert result.error["code"] == "approval_arguments_unavailable" and not seen


def test_management_tools_cannot_self_authorize(tmp_path):
    t = executor_api()
    assert hasattr(t, "register_management_tools"), "management capabilities use ordinary execution"
    store, session, run, _ = make_scope(tmp_path, names=("catalog_search", "tools_enable"))
    catalog = echo_catalog()
    t.register_management_tools(catalog, store)
    executor = t.ToolExecutor(store, catalog)
    result = executor.execute(ToolCall("find", "catalog_search", {"query": "echo"}), session, run)
    assert result.data[0]["name"] == "echo"
    denied = executor.execute(ToolCall("enable", "tools_enable", {"name": "echo"}), session, run)
    assert denied.status == "denied"
    assert "echo" not in store.get_session(session)["config"]["allowed_tools"]
    current = store.get_session(session)
    current["config"]["allowed_tools"].append("echo")
    store.update_session(session, current["config"], current["revision"])
    success = executor.execute(ToolCall("enable2", "tools_enable", {"name": "echo"}), session, run)
    assert success.status == "ok"
    assert "echo" in store.get_session(session)["config"]["enabled_tools"]


@pytest.mark.parametrize("status", ["pending", "running", "waiting"])
def test_known_pending_job_result_is_reused_without_reexecution(tmp_path, status):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    executor = t.ToolExecutor(store, echo_catalog(lambda context, text: seen.append(text) or ToolResult(status=status, job_id="job_known")))
    call = ToolCall("job", "echo", {"text": "x"})
    original = executor.execute(call, session, run)
    result = executor.execute(call, session, run)
    assert result.to_dict() == original.to_dict()
    assert result.job_id == "job_known" and seen == ["x"]


@pytest.mark.parametrize("bad_result", [ToolResult(status="made_up", data="oops"),
    ToolResult(status="ok", error={"code": "oops", "message": "contradictory"}),
    ToolResult(status="ok", artifacts=["not-an-artifact"]), ToolResult(status="error")])
def test_invalid_result_envelopes_are_explicit_contract_errors(tmp_path, bad_result):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    result = t.ToolExecutor(store, echo_catalog(lambda context, text: bad_result)).execute(
        ToolCall("bad", "echo", {"text": "x"}), session, run)
    assert result.status == "error" and result.error["code"] == "output_contract"


@pytest.mark.parametrize("terminal", ["cancelled", "succeeded", "failed", "budget_exhausted"])
def test_persisted_terminal_run_blocks_new_effects_across_store_instances(tmp_path, terminal):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    catalog = echo_catalog(lambda context, text: seen.append(text) or ToolResult(data=text))
    executor = t.ToolExecutor(store, catalog)
    old_call = ToolCall("old", "echo", {"text": "past"})
    assert executor.execute(old_call, session, run).data == "past"
    Store(store.root).update_run(run, {"status": terminal})
    result = executor.execute(ToolCall("new", "echo", {"text": "future"}), session, run)
    assert result.status in {"denied", "cancelled"}
    assert executor.execute(old_call, session, run).data == "past"
    assert seen == ["past"]


def test_waiting_run_allows_exact_approval_reconciliation(tmp_path):
    t, store, session, run, seen, _, executor = approval_scope(tmp_path)
    executor.execute(ToolCall("approval", "echo", {"text": "one"}), session, run)
    store.update_run(run, {"status": "waiting"})
    executor.decide_approval(session, run, "approval", True)
    assert executor.resume_approved(session, run, "approval").status == "ok"
    assert seen == ["one"]


def test_spec_has_no_mutable_instance_dictionary():
    spec = api().ToolSpec("immutable", "Immutable definition", {"type": "object"})
    assert not hasattr(spec, "__dict__"), "the immutable definition must not expose mutable backing fields"


def test_approval_same_process_uses_original_args_without_returning_secrets(tmp_path):
    _, store, session, run, seen, _, executor = approval_scope(tmp_path)
    store.register_secrets(["private-original-argument"])
    executor.execute(ToolCall("private", "echo", {"text": "private-original-argument"}), session, run)
    executor.decide_approval(session, run, "private", True)
    result = executor.resume_approved(session, run, "private")
    assert seen == ["private-original-argument"]
    assert result.data == "[REDACTED]"


def test_approval_cannot_execute_after_definition_changes(tmp_path):
    t, store, session, run, seen, _, executor = approval_scope(tmp_path)
    executor.execute(ToolCall("changed", "echo", {"text": "no"}), session, run)
    executor.decide_approval(session, run, "changed", True)
    changed = echo_catalog(lambda context, text: seen.append(text) or ToolResult(data=text),
                           requires_approval=True, version="2")
    result = t.ToolExecutor(Store(store.root), changed).resume_approved(session, run, "changed")
    assert result.error["code"] == "stale_approval" and not seen


def test_unserializable_provider_id_is_a_safe_rejection_event(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    result = t.ToolExecutor(store, echo_catalog()).execute(ToolCall(object(), "echo", {}), session, run)
    assert result.error["code"] == "invalid_call"
    assert any(event["kind"] == "tool.rejected" for event in store.events(session))


def test_adapter_cannot_fabricate_host_approval_state(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    bogus = ToolResult(status="approval_required", error={"code": "approval_required", "message": "adapter wants approval"})
    result = t.ToolExecutor(store, echo_catalog(lambda context, text: bogus)).execute(
        ToolCall("fabricate", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_contract"


def test_successful_output_cannot_contain_non_json_mapping_keys(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    result = t.ToolExecutor(store, echo_catalog(lambda context, text: ToolResult(data={1: "not a JSON key"}))).execute(
        ToolCall("keys", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_json"


@pytest.mark.parametrize("approve", [False, True])
def test_host_can_resolve_approval_immediately_when_request_event_is_observed(tmp_path, approve):
    t = executor_api()
    class ImmediateHostStore(Store):
        on_request = None
        def append_event(self, session_id, kind, payload, run_id=None):
            event = super().append_event(session_id, kind, payload, run_id)
            if kind == "tool.approval_requested":
                self.on_request(payload["call_id"])
            return event
    store = ImmediateHostStore(tmp_path / "state")
    store, session, run, _ = make_scope(tmp_path, store=store)
    seen = []
    executor = t.ToolExecutor(store, echo_catalog(
        lambda context, text: seen.append(text) or ToolResult(data=text), requires_approval=True))
    def host_decision(call_id):
        executor.decide_approval(session, run, call_id, approve)
        if approve:
            executor.resume_approved(session, run, call_id)
    store.on_request = host_decision
    result = executor.execute(ToolCall("immediate", "echo", {"text": "one"}), session, run)
    assert result.status == ("ok" if approve else "denied")
    assert seen == (["one"] if approve else [])


def test_concurrent_stale_approval_observer_cannot_overwrite_owner_failure(tmp_path):
    t = executor_api()
    entered, release = Event(), Event()
    class PausedFinishStore(Store):
        def finish_action(self, run_id, call_id, result):
            if (result.get("error") or {}).get("code") == "approval_arguments_unavailable":
                entered.set()
                assert release.wait(3)
            return super().finish_action(run_id, call_id, result)
    store, session, run, _ = make_scope(tmp_path)
    store.register_secrets(["not-recoverable-value"])
    catalog = echo_catalog(requires_approval=True)
    executor = t.ToolExecutor(store, catalog)
    executor.execute(ToolCall("stale", "echo", {"text": "not-recoverable-value"}), session, run)
    executor.decide_approval(session, run, "stale", True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(t.ToolExecutor(PausedFinishStore(store.root), catalog).resume_approved, session, run, "stale")
        assert entered.wait(3)
        try:
            observer = t.ToolExecutor(Store(store.root), catalog).resume_approved(session, run, "stale")
            assert observer.status == "uncertain"
        finally:
            release.set()
        owner = first.result(timeout=3)
    assert owner.error["code"] == "approval_arguments_unavailable"


@pytest.mark.parametrize(("status", "code", "details", "expected"), [
    ("approval_required", "adapter_error", None, "output_contract"),
    ("made_up", "adapter_error", None, "output_contract"),
    ("ok", "adapter_error", None, "output_contract"),
    ("error", object(), None, "output_contract"),
    ("error", "adapter_error", {"bad": object()}, "output_json"),
    ("error", "adapter_error", {"bad": float("nan")}, "output_json"),
])
def test_raised_tool_failure_must_pass_the_same_result_contract(tmp_path, status, code, details, expected):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    def broken(context, text):
        raise t.ToolFailure(code, "adapter failure", status=status, details=details)
    executor = t.ToolExecutor(store, echo_catalog(broken))
    call = ToolCall("raised", "echo", {"text": "x"})
    result = executor.execute(call, session, run)
    assert result.status == "error" and result.error["code"] == expected
    assert store.get_action(run, call.id)["finished_at"] is not None
    assert executor.pending_approvals(session) == []
    assert executor.execute(call, session, run).to_dict() == result.to_dict()


def test_raised_failure_with_recursive_details_finishes_as_output_json(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    details = {}
    details["cycle"] = details
    def broken(context, text):
        raise t.ToolFailure("bad_details", "cycle", details=details)
    result = t.ToolExecutor(store, echo_catalog(broken)).execute(ToolCall("cycle", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_json"
    assert store.get_action(run, "cycle")["finished_at"] is not None


def test_result_serialization_failure_cannot_escape_validation(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")
    def broken(context, text):
        raise t.ToolFailure("bad_details", "uncopyable", details={"bad": Uncopyable()})
    result = t.ToolExecutor(store, echo_catalog(broken)).execute(ToolCall("copy", "echo", {"text": "x"}), session, run)
    assert result.error["code"] == "output_json"
    assert store.get_action(run, "copy")["finished_at"] is not None


def test_handler_context_carries_exact_provider_call_id_for_job_idempotency(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    def handler(context, text):
        seen.append((context.session_id, context.run_id, context.call_id))
        return ToolResult(data=text)
    call = ToolCall("exact_provider_call", "echo", {"text": "x"})
    result = t.ToolExecutor(store, echo_catalog(handler)).execute(call, session, run)
    assert result.status == "ok"
    assert seen == [(session, run, "exact_provider_call")]


def test_approval_resume_context_preserves_exact_call_id_across_restart(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    def handler(context, text):
        seen.append(context.call_id)
        return ToolResult(data=text)
    catalog = echo_catalog(handler, requires_approval=True)
    executor = t.ToolExecutor(store, catalog)
    executor.execute(ToolCall("approved_provider_call", "echo", {"text": "x"}), session, run)
    executor.decide_approval(session, run, "approved_provider_call", True)
    result = t.ToolExecutor(Store(store.root), catalog).resume_approved(session, run, "approved_provider_call")
    assert result.status == "ok" and seen == ["approved_provider_call"]


def test_standalone_context_call_id_is_optional_keyword(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    assert t.ToolContext(store, session, run, 0).call_id is None
    assert t.ToolContext(store, session, run, 0, call_id="host_call").call_id == "host_call"


@pytest.mark.parametrize("status", ["pending", "running", "waiting"])
def test_approved_known_job_result_survives_repeated_resume_after_restart(tmp_path, status):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    seen = []
    def handler(context, text):
        seen.append(context.call_id)
        return ToolResult(status=status, job_id="job_exact_action", data={"progress": "recorded"})
    catalog = echo_catalog(handler, requires_approval=True)
    executor = t.ToolExecutor(store, catalog)
    call = ToolCall("approved_job", "echo", {"text": "x"})
    assert executor.execute(call, session, run).status == "approval_required"
    executor.decide_approval(session, run, call.id, True)
    original = executor.resume_approved(session, run, call.id)
    assert original.status == status and original.job_id == "job_exact_action"
    recorded = store.get_action(run, call.id)
    assert recorded["finished_at"] is None
    restarted = t.ToolExecutor(Store(store.root), catalog)
    assert restarted.resume_approved(session, run, call.id).to_dict() == original.to_dict()
    assert restarted.resume_approved(session, run, call.id).to_dict() == original.to_dict()
    assert restarted.execute(call, session, run).to_dict() == original.to_dict()
    assert store.get_action(run, call.id) == recorded
    assert seen == [call.id]


def test_catalog_secret_registration_is_atomic_and_host_only(tmp_path):
    t = executor_api()
    catalog = echo_catalog()
    assert hasattr(catalog, "prepare_store"), "catalog must prepare the actual execution Store"
    store = Store(tmp_path / "state")
    with pytest.raises(ValueError, match="Duplicate"):
        catalog.register_many([(catalog.get("echo").spec, lambda context: ToolResult())], secrets=["not-registered"])
    catalog.prepare_store(store)
    assert store.redact("not-registered") == "not-registered"
    with pytest.raises((TypeError, ValueError)):
        catalog.register_many([], secrets="not-an-iterable-of-values")
    with pytest.raises((TypeError, ValueError)):
        catalog.register_many([], secrets=["must-not-partially-register", object()])
    catalog.prepare_store(store)
    assert store.redact("must-not-partially-register") == "must-not-partially-register"
    catalog.register_many([], secrets=["host-only-credential"])
    catalog.prepare_store(store)
    assert store.redact("host-only-credential") == "[REDACTED]"
    assert "host-only-credential" not in repr(catalog)
    assert "host-only-credential" not in json.dumps(catalog.search(""))
    assert "host-only-credential" not in json.dumps(catalog.get("echo").spec.model_schema())


def test_approval_resume_prepares_store_for_newly_registered_provider_secrets(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    secret = "registered-between-approval-and-resume"
    catalog = echo_catalog(lambda context, text: ToolResult(data=secret), requires_approval=True)
    assert hasattr(catalog, "prepare_store"), "catalog must prepare approval recovery Store"
    executor = t.ToolExecutor(store, catalog)
    executor.execute(ToolCall("later_secret", "echo", {"text": "ordinary argument"}), session, run)
    executor.decide_approval(session, run, "later_secret", True)
    catalog.register_many([], secrets=[secret])
    result = executor.resume_approved(session, run, "later_secret")
    assert result.data == "[REDACTED]"
    assert secret not in json.dumps(store.events(session))


@pytest.mark.parametrize("call", [
    ToolCall("unicode_value", "echo", {"text": "\ud800"}),
    ToolCall("unicode_key", "echo", {"\ud800": "x"}),
    ToolCall("unicode_name", "\ud800", {"text": "x"}),
    ToolCall("\ud800", "echo", {"text": "x"}),
])
def test_unpaired_surrogate_calls_are_typed_rejections_with_safe_events(tmp_path, call):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    result = t.ToolExecutor(store, echo_catalog()).execute(call, session, run)
    assert result.status == "error" and result.error["code"] == "invalid_call"
    events = store.events(session)
    assert any(event["kind"] == "tool.rejected" for event in events)
    json.dumps(events, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("bad_result", [ToolResult(data="\ud800"),
    ToolResult(data={"\ud800": "bad key"}),
    ToolResult(status="error", error={"code": "bad_unicode", "message": "\ud800"})])
def test_unpaired_surrogate_outputs_finish_as_output_json(tmp_path, bad_result):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    result = t.ToolExecutor(store, echo_catalog(lambda context, text: bad_result)).execute(
        ToolCall("unicode_output", "echo", {"text": "x"}), session, run)
    assert result.status == "error" and result.error["code"] == "output_json"
    assert store.get_action(run, "unicode_output")["finished_at"] is not None


@pytest.mark.parametrize("kwargs", [{"description": "\ud800"}, {"version": "\ud800"},
    {"input_schema": {"type": "object", "properties": {"\ud800": {"type": "string"}}}},
    {"output_schema": {"type": "string", "enum": ["\ud800"]}}])
def test_tool_definitions_reject_unpaired_surrogates_upfront(kwargs):
    values = {"name": "valid", "description": "Ordinary", "input_schema": {"type": "object"}}
    values.update(kwargs)
    with pytest.raises(ValueError):
        api().ToolSpec(**values)


def test_normal_chinese_unicode_remains_round_trippable(tmp_path):
    t = executor_api()
    store, session, run, _ = make_scope(tmp_path)
    catalog = t.ToolCatalog()
    catalog.register(t.ToolSpec("echo", "保留中文说明", {"type": "object"}, version="版本一"),
                     lambda context, **arguments: ToolResult(data=arguments))
    result = t.ToolExecutor(store, catalog).execute(ToolCall("chinese", "echo", {"问题": "你好世界"}), session, run)
    assert result.data == {"问题": "你好世界"}
    assert store.get_action(run, "chinese")["args"] == {"问题": "你好世界"}
    assert catalog.get("echo").spec.definition_hash
