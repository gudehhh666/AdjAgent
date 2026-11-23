"""Real local stdio transport checks. Live remote HTTP is intentionally not used."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from threading import Event
import time

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.storage import Store
from adj_partner import tools


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


def mcp_api():
    assert hasattr(tools, "MCPServerConfig"), "real MCP configuration adapter must exist"
    assert hasattr(tools, "MCPProvider"), "real MCP SDK provider must exist"
    return tools


def config(**kwargs):
    defaults = {"name": "local", "command": sys.executable,
                "args": ("-B", str(FIXTURE)), "network": False,
                "requires_approval": False, "connection_timeout": 3, "call_timeout": 2}
    defaults.update(kwargs)
    return mcp_api().MCPServerConfig(**defaults)


def setup_provider(tmp_path, cfg):
    t = mcp_api()
    store = Store(tmp_path / "state")
    provider = t.MCPProvider(cfg)
    catalog = t.ToolCatalog()
    names = provider.register(catalog, store)
    session = store.create_session(SessionConfig(allowed_tools=names, enabled_tools=names).to_dict())
    task = store.create_task(session["id"], "MCP integration")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    return store, provider, catalog, session["id"], run["id"]


def test_actual_stdio_discovery_execution_and_remote_errors(tmp_path):
    t = mcp_api()
    store, provider, catalog, session, run = setup_provider(tmp_path, config())
    assert catalog.names() == ["local__echo", "local__environment", "local__fail", "local__large", "local__slow"]
    assert provider.discover()[0].name == "local__echo"
    executor = t.ToolExecutor(store, catalog)
    result = executor.execute(ToolCall("echo", "local__echo", {"text": "real protocol"}), session, run)
    assert result.status == "ok" and result.data == {"text": "real protocol"}
    failure = executor.execute(ToolCall("fail", "local__fail", {}), session, run)
    assert failure.status == "error" and failure.error["code"] == "mcp_tool_error"
    assert "fixture rejected" in failure.error["message"]


def test_remote_annotations_never_grant_permission(tmp_path):
    t = mcp_api()
    cfg = t.MCPServerConfig("local", command=sys.executable, args=("-B", str(FIXTURE)))
    specs = t.MCPProvider(cfg).discover()
    assert all(spec.network and spec.requires_approval for spec in specs)


def test_stdio_does_not_inherit_parent_api_secrets_and_redacts_host_values(tmp_path, monkeypatch):
    t = mcp_api()
    monkeypatch.setenv("HARNESS_PARENT_API_SECRET", "must-never-be-inherited")
    store, _, catalog, session, run = setup_provider(tmp_path, config(env={"HARNESS_EXPLICIT_VALUE": "explicit-private-value"}))
    result = t.ToolExecutor(store, catalog).execute(ToolCall("env", "local__environment", {}), session, run)
    assert result.status == "ok"
    assert result.data["inherited_value"] is None
    assert result.data["configured_value"] == "[REDACTED]"
    assert "explicit-private-value" not in json.dumps(store.events(session))
    assert "explicit-private-value" not in json.dumps(t.SessionToolset(store, catalog, session).snapshot())


def test_mcp_disconnected_server_is_an_explicit_bounded_failure(tmp_path):
    t = mcp_api()
    provider = t.MCPProvider(config(command="/no/such/mcp-server"))
    started = time.monotonic()
    with pytest.raises(t.ToolFailure) as exc:
        provider.discover()
    assert exc.value.code == "mcp_connection"
    assert time.monotonic() - started < 5
    store, _, catalog, session, run = setup_provider(tmp_path, config())
    # Discover normally, then the configured executable disappears (copied fixture wrapper).
    wrapper = tmp_path / "server.py"
    wrapper.write_text(FIXTURE.read_text())
    other = t.MCPProvider(config(name="gone", args=("-B", str(wrapper))))
    other.register(catalog, store)
    wrapper.unlink()
    current = store.get_session(session)
    current["config"]["enabled_tools"].append("gone__echo")
    current["config"]["allowed_tools"].append("gone__echo")
    store.update_session(session, current["config"], 0)
    result = t.ToolExecutor(store, catalog).execute(ToolCall("gone", "gone__echo", {"text": "x"}), session, run)
    assert result.status == "error" and result.error["code"] == "mcp_connection"


def test_mcp_schema_drift_is_rejected_before_call(tmp_path):
    t = mcp_api()
    schema = tmp_path / "server-schema.json"
    schema.write_text('{"text_type":"string"}')
    store, _, catalog, session, run = setup_provider(tmp_path, config(args=("-B", str(FIXTURE), "--schema-file", str(schema))))
    schema.write_text('{"text_type":"integer"}')
    result = t.ToolExecutor(store, catalog).execute(ToolCall("drift", "local__echo", {"text": "x"}), session, run)
    assert result.status == "denied" and result.error["code"] == "schema_drift"


@pytest.mark.parametrize("mode", ["unsafe", "duplicate"])
def test_mcp_unsafe_names_and_collisions_are_rejected(mode):
    t = mcp_api()
    with pytest.raises(t.ToolFailure) as exc:
        t.MCPProvider(config(args=("-B", str(FIXTURE), "--mode", mode))).discover()
    assert exc.value.code == "mcp_schema"


def test_mcp_catalog_registration_collision_preserves_existing_bindings(tmp_path):
    t = mcp_api()
    store = Store(tmp_path / "state")
    catalog = t.ToolCatalog()
    existing = t.ToolSpec("local__environment", "Host capability", {"type": "object"})
    handler = lambda context: None
    catalog.register(existing, handler)
    with pytest.raises(ValueError, match="Duplicate"):
        t.MCPProvider(config()).register(catalog, store)
    assert catalog.names() == ["local__environment"]
    assert catalog.get("local__environment").handler is handler


def test_mcp_call_timeout_and_cancellation_are_bounded(tmp_path):
    t = mcp_api()
    store, _, catalog, session, run = setup_provider(tmp_path, config(call_timeout=0.3))
    executor = t.ToolExecutor(store, catalog)
    started = time.monotonic()
    result = executor.execute(ToolCall("timeout", "local__slow", {}), session, run)
    assert result.error["code"] == "mcp_timeout"
    assert time.monotonic() - started < 6
    signal = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(executor.execute, ToolCall("cancel", "local__slow", {}), session, run,
                              cancel_event=signal)
        signal.set()
        result = pending.result(timeout=6)
    assert result.status == "cancelled"


def test_http_config_is_host_only_conservative_and_defensive():
    t = mcp_api()
    headers = {"Authorization": "Bearer not-a-real-secret"}
    cfg = t.MCPServerConfig("remote", transport="http", url="https://example.invalid/mcp", headers=headers,
                            network=False)
    headers["Authorization"] = "changed"
    assert cfg.headers["Authorization"] == "Bearer not-a-real-secret"
    assert cfg.network is True
    assert "not-a-real-secret" not in repr(cfg)
    with pytest.raises((ValueError, TypeError)):
        t.MCPServerConfig("remote", transport="http", url="file:///tmp/mcp")
    with pytest.raises(ValueError):
        config(call_timeout=0)


def test_active_mcp_call_cancels_and_reaps_its_subprocess(tmp_path):
    t = mcp_api()
    marker = tmp_path / "started.txt"
    store, _, catalog, session, run = setup_provider(tmp_path, config(
        args=("-B", str(FIXTURE), "--started-file", str(marker)), call_timeout=5))
    executor = t.ToolExecutor(store, catalog)
    signal = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, ToolCall("active_cancel", "local__slow", {}), session, run,
                             cancel_event=signal)
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "real MCP server must have begun the call"
        signal.set()
        result = future.result(timeout=6)
    assert result.status == "cancelled"
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_mcp_output_size_ceiling_is_explicit_not_silent_truncation(tmp_path):
    t = mcp_api()
    store, _, catalog, session, run = setup_provider(tmp_path, config())
    result = t.ToolExecutor(store, catalog).execute(ToolCall("large", "local__large", {}), session, run)
    assert result.status == "error" and result.error["code"] == "output_too_large"
    assert result.data is None
    assert len(json.dumps(store.get_action(run, "large")["result"])) < 2048


def test_actual_mcp_protocol_error_preserves_remote_code_and_details(tmp_path):
    t = mcp_api()
    schema = tmp_path / "server-schema.json"
    schema.write_text("{}")
    store, _, catalog, session, run = setup_provider(tmp_path, config(args=("-B", str(FIXTURE), "--schema-file", str(schema))))
    schema.write_text('{"protocol_error":true}')
    result = t.ToolExecutor(store, catalog).execute(ToolCall("protocol", "local__echo", {"text": "x"}), session, run)
    assert result.error["code"] == "mcp_protocol"
    assert result.error["details"] == {"provider_code": -32055, "provider_data": {"reason": "fixture"}}


def test_persisted_cancel_after_actual_discovery_prevents_remote_invocation(tmp_path):
    t = mcp_api()
    marker = tmp_path / "invoked.txt"
    store = Store(tmp_path / "state")
    class CancelAfterDiscovery(t.MCPProvider):
        run_id = None
        async def _list(self, session):
            discovered = await super()._list(session)  # Real installed SDK/local subprocess.
            if self.run_id is not None:
                Store(store.root).update_run(self.run_id, {"status": "cancelled"})
            return discovered
    provider = CancelAfterDiscovery(config(args=("-B", str(FIXTURE), "--invoked-file", str(marker))))
    catalog = t.ToolCatalog()
    names = provider.register(catalog, store)
    session = store.create_session(SessionConfig(allowed_tools=names, enabled_tools=names).to_dict())
    task = store.create_task(session["id"], "Cancellation during discovery")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    provider.run_id = run["id"]
    result = t.ToolExecutor(store, catalog).execute(
        ToolCall("cancel_discovery", "local__echo", {"text": "must not invoke"}), session["id"], run["id"])
    assert result.status == "cancelled"
    assert not marker.exists(), "persisted cancellation must prevent tools/call after discovery"
    assert store.get_action(run["id"], "cancel_discovery")["finished_at"] is not None


def test_persisted_cancel_during_actual_mcp_io_reaps_subprocess(tmp_path):
    t = mcp_api()
    marker = tmp_path / "started.txt"
    store, _, catalog, session, run = setup_provider(tmp_path, config(
        args=("-B", str(FIXTURE), "--started-file", str(marker)), call_timeout=3))
    executor = t.ToolExecutor(store, catalog)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, ToolCall("persisted_cancel", "local__slow", {}), session, run)
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "real MCP server must have begun the call"
        Store(store.root).update_run(run, {"status": "cancelled"})
        result = future.result(timeout=6)
    assert result.status == "cancelled", "cross-process cancellation must not wait for the MCP timeout"
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    assert store.get_action(run, "persisted_cancel")["finished_at"] is not None


def test_mcp_credentials_are_redacted_by_reopened_execution_store(tmp_path):
    t = mcp_api()
    secret = "synthetic-reopened-provider-credential"
    registration_store, _, catalog, session, run = setup_provider(tmp_path, config(env={"HARNESS_EXPLICIT_VALUE": secret}))
    execution_store = Store(registration_store.root)
    result = t.ToolExecutor(execution_store, catalog).execute(
        ToolCall("reopened_credentials", "local__environment", {}), session, run)
    assert result.status == "ok" and result.data["configured_value"] == "[REDACTED]"
    assert secret not in json.dumps(execution_store.get_action(run, "reopened_credentials"))
    assert secret not in json.dumps(execution_store.events(session))
    assert secret not in json.dumps(t.SessionToolset(execution_store, catalog, session).snapshot())
    assert secret not in json.dumps(catalog.search(""))
    assert secret not in repr(catalog)


def test_dynamic_mcp_registration_prepares_execution_store_before_argument_journaling(tmp_path):
    t = mcp_api()
    secret = "synthetic-dynamic-provider-credential"
    registration_store = Store(tmp_path / "state")
    execution_store = Store(registration_store.root)
    catalog = t.ToolCatalog()
    executor = t.ToolExecutor(execution_store, catalog)  # Before provider registration.
    names = t.MCPProvider(config(env={"HARNESS_EXPLICIT_VALUE": secret})).register(catalog, registration_store)
    session = registration_store.create_session(SessionConfig(allowed_tools=names, enabled_tools=names).to_dict())
    task = registration_store.create_task(session["id"], "Dynamic provider registration")
    run = registration_store.create_run(session["id"], task["id"], Budget().to_dict())
    result = executor.execute(ToolCall("dynamic_credentials", "local__environment", {"value": secret}),
                              session["id"], run["id"])
    assert result.status == "ok" and result.data["configured_value"] == "[REDACTED]"
    action = execution_store.get_action(run["id"], "dynamic_credentials")
    assert action["args"]["value"] == "[REDACTED]"
    attempt = next(event for event in execution_store.events(session["id"]) if event["kind"] == "tool.attempt")
    assert attempt["payload"]["arguments"]["value"] == "[REDACTED]"
    assert secret not in json.dumps(execution_store.events(session["id"]))
