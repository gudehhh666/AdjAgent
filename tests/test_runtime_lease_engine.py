"""Engine surfaces heartbeat failures without replaying uncertain model work."""

import threading
import time

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig
from adj_partner.runtime.engine import AgentEngine
from adj_partner.runtime.journal import LeaseLost
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog
from adj_partner.tools.management import register_management_tools


class SlowOfflinePort:
    secrets = ()
    metadata = {"provider_kind": "test_scripted", "model": "offline"}

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, **kwargs):
        self.calls += 1
        time.sleep(1.2)  # let the actual heartbeat thread perform a renewal
        return ModelReply(content="offline result", usage={"input_tokens": 1, "output_tokens": 1},
                          finish_reason="stop")


def test_engine_surfaces_sanitized_heartbeat_failure_and_keeps_request_pending(tmp_path, monkeypatch):
    store, catalog, port = Store(tmp_path / "state"), ToolCatalog(), SlowOfflinePort()
    register_management_tools(catalog, store)
    sid = store.create_session(SessionConfig(enabled_tools=catalog.names(), allowed_tools=[]).to_dict())["id"]
    engine = AgentEngine(store, catalog, port)
    run = engine.start(sid, "offline request", budget=Budget(max_steps=2).to_dict())
    original = engine.journal.renew
    sentinel = "/private/opaque-heartbeat-detail"

    def failing_heartbeat(run_id, owner, ttl=15):
        if threading.current_thread().name == "adj-runtime-lease":
            raise OSError(sentinel)
        return original(run_id, owner, ttl)

    monkeypatch.setattr(engine.journal, "renew", failing_heartbeat)
    with pytest.raises(LeaseLost, match="heartbeat_renewal_error") as caught:
        engine.resume(run["id"])

    assert isinstance(caught.value.__cause__, OSError)
    assert sentinel not in str(caught.value)
    assert port.calls == 1
    pending = store.get_run(run["id"])
    assert pending["status"] == "running" and pending["state"]["phase"] == "requesting"
    assert "pending_request" in pending["state"]
    assert len(engine.journal.diagnostics(run["id"])) == 1
    assert sentinel not in str(engine.journal.diagnostics(run["id"]))


def test_lost_heartbeat_is_visible_before_blocked_diagnostic_and_effect_guard(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from adj_partner.contracts import ToolCall, ToolResult
    from adj_partner.tools import ToolSpec

    effects = []

    class ToolPort:
        secrets = ()
        metadata = {"provider_kind": "test_scripted", "model": "offline"}

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ModelReply(tool_calls=[ToolCall("do-it", "effect", {"value": "should not run"})],
                                  usage={"input_tokens": 1, "output_tokens": 1}, finish_reason="tool_calls")
            return ModelReply(content="done", usage={"input_tokens": 1, "output_tokens": 1}, finish_reason="stop")

    store, catalog, port = Store(tmp_path / "state"), ToolCatalog(), ToolPort()
    catalog.register(ToolSpec("effect", "Effect", {"type": "object", "properties": {"value": {"type": "string"}},
        "required": ["value"], "additionalProperties": False}),
        lambda ctx, value: (effects.append(value) or ToolResult(data={"value": value})))
    register_management_tools(catalog, store)
    sid = store.create_session(SessionConfig(enabled_tools=catalog.names(), allowed_tools=catalog.names()).to_dict())["id"]
    engine = AgentEngine(store, catalog, port)
    run = engine.start(sid, "effect must be fenced", budget=Budget(max_steps=3).to_dict())

    diagnostic_entered, release_diagnostic = threading.Event(), threading.Event()
    original_renew = engine.journal.renew
    original_record = engine.journal.record_heartbeat_failure
    original_checkpoint = engine.journal.checkpoint

    def failing_heartbeat(run_id, owner, ttl=15):
        if threading.current_thread().name == "adj-runtime-lease":
            raise OSError("private renewal failure")
        return original_renew(run_id, owner, ttl)

    def blocked_diagnostic(run_id, owner, exc):
        diagnostic_entered.set()
        release_diagnostic.wait(5)
        return original_record(run_id, owner, exc)

    def wait_for_heartbeat(*args, **kwargs):
        value = original_checkpoint(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_eligible":
            assert diagnostic_entered.wait(3), "heartbeat did not reach its diagnostic write"
        return value

    monkeypatch.setattr(engine.journal, "renew", failing_heartbeat)
    monkeypatch.setattr(engine.journal, "record_heartbeat_failure", blocked_diagnostic)
    monkeypatch.setattr(engine.journal, "checkpoint", wait_for_heartbeat)
    pool = ThreadPoolExecutor(1)
    try:
        future = pool.submit(engine.resume, run["id"])
        with pytest.raises(LeaseLost, match="heartbeat"):
            future.result(timeout=5)
    finally:
        release_diagnostic.set()
        pool.shutdown(wait=True)

    assert effects == []
    assert port.calls == 1
    assert store.get_run(run["id"])["state"]["phase"] == "tools"
