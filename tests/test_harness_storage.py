"""Real persistence contract tests; no replacement stores or fake databases."""

import importlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


def module(name):
    try:
        return importlib.import_module(f"adj_partner.{name}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Persistence module is not implemented: {exc.name}")


def test_contracts_round_trip_without_sharing_mutable_defaults():
    contracts = module("contracts")
    config = contracts.SessionConfig.from_dict({
        "instructions": "Preserve evidence", "enabled_tools": ["read"],
        "allowed_roots": ["/tmp/project"], "budget": {"max_steps": 5},
    })
    assert config.budget.max_steps == 5
    assert config.to_dict()["budget"]["max_input_tokens"] == 120000
    assert contracts.SessionConfig.from_dict(config.to_dict()) == config
    config.enabled_tools.append("write")
    assert contracts.SessionConfig().enabled_tools == []
    assert contracts.ModelReply().tool_calls == []
    reply = contracts.ModelReply(tool_calls=[contracts.ToolCall("call_1", "read", {"path": "a"})])
    assert json.loads(json.dumps(reply.to_dict()))["tool_calls"][0]["id"] == "call_1"
    result = contracts.ToolResult(data={"answer": 1})
    assert result.to_dict() == {
        "status": "ok", "data": {"answer": 1}, "artifacts": [], "error": None, "job_id": None,
    }
    first_id = contracts.new_id("session")
    assert first_id.startswith("session_") and first_id != contracts.new_id("session")


@pytest.mark.parametrize("field,value", [
    ("max_steps", 0), ("max_input_tokens", -1), ("max_output_tokens", 0),
    ("wall_time_seconds", 0), ("wall_time_seconds", float("inf")),
    ("max_steps", 1.5), ("max_steps", True),
])
def test_budget_rejects_nonpositive_or_invalid_limits(field, value):
    with pytest.raises(ValueError):
        module("contracts").Budget(**{field: value})


def test_store_reopens_sessions_tasks_runs_and_native_messages(tmp_path):
    Store = module("storage").Store
    store = Store(tmp_path / "state")
    session = store.create_session({"instructions": "test", "budget": {"max_steps": 2}})
    task = store.create_task(session["id"], "Compute a mean", {"metric": "mean", "target": 3})
    run = store.create_run(session["id"], task["id"], {"max_steps": 2})
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_provider_1", "type": "function", "function": {"name": "mean", "arguments": "{\"x\": [2, 4]}"}},
    ]}
    store.append_message(session["id"], message, run_id=run["id"])
    event = store.append_event(session["id"], "test.observed", {"value": 3}, run_id=run["id"])
    store.close()
    reopened = Store(tmp_path / "state")
    assert reopened.root == (tmp_path / "state").resolve()
    assert reopened.get_session(session["id"]) == session
    assert reopened.get_task(task["id"]) == task
    assert reopened.get_run(run["id"]) == run
    assert task["status"] == "open" and task["revision"] == 0
    assert run["status"] == "created" and run["state"] == {} and run["usage"] == {}
    assert reopened.messages(session["id"]) == [message]
    assert reopened.list_sessions() == [session]
    assert reopened.list_runs() == [run]
    assert reopened.list_runs(session["id"]) == [run]
    assert reopened.events(session["id"])[-1] == event
    assert event["seq"] == event["id"]
    assert event["kind"] == "test.observed" and event["created_at"]
    assert (reopened.root / "db.sqlite3").is_file()
    assert (reopened.root / "artifacts").is_dir()


def seeded_store(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    session = store.create_session({"instructions": "test"})
    task = store.create_task(session["id"], "task")
    run = store.create_run(session["id"], task["id"], {"max_steps": 3})
    return store, session, task, run


def test_revision_updates_are_visible_to_another_store_and_reject_stale_writes(tmp_path):
    store, session, task, run = seeded_store(tmp_path)
    observer = module("storage").Store(store.root)
    updated = store.update_session(session["id"], {"instructions": "new"}, expected_revision=0)
    assert observer.get_session(session["id"]) == updated and updated["revision"] == 1
    with pytest.raises(ValueError, match="revision"):
        store.update_session(session["id"], {}, expected_revision=0)
    task_update = store.update_task(task["id"], {"acceptance": {"required": True}}, expected_revision=0)
    assert task_update["acceptance"] == {"required": True} and task_update["revision"] == 1
    with pytest.raises(ValueError, match="revision"):
        store.update_task(task["id"], {"goal": "stale"}, expected_revision=0)
    run_update = store.update_run(run["id"], {"status": "running", "state": {"step": 1},
                                             "usage": {"input_tokens": 12}}, expected_revision=0)
    assert run_update["revision"] == 1 and observer.get_run(run["id"]) == run_update
    with pytest.raises(ValueError, match="revision"):
        store.update_run(run["id"], {"status": "failed"}, expected_revision=0)
    kinds = [event["kind"] for event in observer.events(session["id"])]
    assert kinds[-3:] == ["session.updated", "task.updated", "run.updated"]
    assert [event["kind"] for event in observer.events(session["id"], run["id"])] == ["run.created", "run.updated"]
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == module("storage").SCHEMA_VERSION


def test_update_and_host_event_are_one_atomic_transaction(tmp_path):
    store, session, _, run = seeded_store(tmp_path)
    before = store.events(session["id"])
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("CREATE TRIGGER reject_event BEFORE INSERT ON events "
                           "BEGIN SELECT RAISE(ABORT, 'event unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError, match="event unavailable"):
        store.update_run(run["id"], {"status": "running"})
    assert store.get_run(run["id"]) == run
    assert store.events(session["id"]) == before


def test_parallel_revision_writers_have_exactly_one_winner(tmp_path):
    store, session, _, _ = seeded_store(tmp_path)

    def write(value):
        try:
            return store.update_session(session["id"], {"writer": value}, expected_revision=0)["revision"]
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [1, 2]))
    assert sorted(results, key=str) == [1, "conflict"]
    assert len([e for e in store.events(session["id"]) if e["kind"] == "session.updated"]) == 1


def test_updates_reject_undeclared_fields_and_invalid_terminal_transitions(tmp_path):
    store, _, task, run = seeded_store(tmp_path)
    with pytest.raises(ValueError, match="field"):
        store.update_task(task["id"], {"session_id": "anything"})
    with pytest.raises(ValueError, match="field"):
        store.update_run(run["id"], {"task_id": "anything"})
    with pytest.raises(ValueError, match="status"):
        store.update_run(run["id"], {"status": "success_maybe"})
    terminal = store.update_run(run["id"], {"status": "succeeded"})
    for active in ("created", "running", "waiting"):
        with pytest.raises(ValueError, match="terminal"):
            store.update_run(run["id"], {"status": active})
    assert store.get_run(run["id"]) == terminal


def test_scope_checks_prevent_cross_session_references(tmp_path):
    store, _, task, run = seeded_store(tmp_path)
    other = store.create_session({})
    with pytest.raises(ValueError, match="belong"):
        store.create_run(other["id"], task["id"], {})
    with pytest.raises(ValueError, match="belong"):
        store.append_event(other["id"], "bad", {}, run["id"])
    with pytest.raises(ValueError, match="belong"):
        store.append_message(other["id"], {"role": "user", "content": "bad"}, run["id"])
    with pytest.raises(ValueError, match="identifier"):
        store.get_session("../elsewhere")
    with pytest.raises(KeyError):
        store.get_run("run_unknown")


def test_action_intent_is_durable_and_completed_calls_are_idempotent(tmp_path):
    store, session, _, run = seeded_store(tmp_path)
    first = store.record_action(run["id"], "call_1", "calculate", {"x": 2}, 7)
    assert first["status"] == "started" and first["result"] is None
    assert first["finished_at"] is None and first["toolset_revision"] == 7
    reopened = module("storage").Store(store.root)
    assert reopened.get_action(run["id"], "call_1") == first
    result = {"status": "ok", "data": {"answer": 4}}
    completed = store.finish_action(run["id"], "call_1", result)
    assert completed["status"] == "completed" and completed["result"] == result
    assert completed["finished_at"]
    assert store.record_action(run["id"], "call_1", "calculate", {"x": 2}, 8) == completed
    assert store.finish_action(run["id"], "call_1", result) == completed
    assert reopened.list_actions(run["id"]) == [completed]
    assert reopened.get_action(run["id"], "missing_call") is None
    kinds = [event["kind"] for event in store.events(session["id"], run["id"])]
    assert kinds[-2:] == ["action.started", "action.finished"]
    with pytest.raises(ValueError, match="result"):
        store.finish_action(run["id"], "call_1", {"status": "ok", "data": {"answer": 9}})


def test_action_reused_call_id_cannot_change_tool_or_arguments(tmp_path):
    store, _, _, run = seeded_store(tmp_path)
    store.record_action(run["id"], "call_1", "calculate", {"x": 2}, 0)
    for tool, arguments in [("read", {"x": 2}), ("calculate", {"x": 3})]:
        with pytest.raises(ValueError, match="reused"):
            store.record_action(run["id"], "call_1", tool, arguments, 0)
    with pytest.raises(ValueError, match="identifier"):
        store.record_action(run["id"], "../invalid", "calculate", {}, 0)
    with pytest.raises(KeyError):
        store.finish_action(run["id"], "call_missing", {"status": "ok"})


@pytest.mark.parametrize("status", ["pending", "approval_required", "waiting"])
def test_pending_actions_do_not_claim_a_finished_side_effect(tmp_path, status):
    store, _, _, run = seeded_store(tmp_path)
    store.record_action(run["id"], "call_1", "submit_job", {}, 0)
    waiting = store.finish_action(run["id"], "call_1", {"status": status, "job_id": "job_1"})
    assert waiting["status"] == status and waiting["finished_at"] is None
    assert store.record_action(run["id"], "call_1", "submit_job", {}, 0) == waiting
    failed = store.finish_action(run["id"], "call_1", {"status": "error", "error": {"message": "no approval"}})
    assert failed["status"] == "failed" and failed["finished_at"]


def test_simultaneous_action_intents_share_one_journal_row(tmp_path):
    store, session, _, run = seeded_store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: store.record_action(run["id"], "call_1", "read", {}, 0), range(8)))
    assert all(value == values[0] for value in values)
    assert len(store.list_actions(run["id"])) == 1
    assert len([e for e in store.events(session["id"]) if e["kind"] == "action.started"]) == 1


def test_concurrent_action_claims_grant_exactly_one_execution_owner(tmp_path):
    store, _, _, run = seeded_store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: store.claim_action(run["id"], "call_1", "read", {}, 0), range(8)))
    assert sum(created for _, created in claims) == 1
    assert len({record["id"] for record, _ in claims}) == 1
    reopened = module("storage").Store(store.root)
    record, created = reopened.claim_action(run["id"], "call_1", "read", {}, 0)
    assert created is False and record["status"] == "started"


def test_artifacts_are_content_addressed_immutable_and_reopened(tmp_path):
    store, session, _, run = seeded_store(tmp_path)
    first = store.put_artifact(b"scientific bytes\x00", "result.bin", provenance={"source": "calculation"},
                               session_id=session["id"], run_id=run["id"])
    second = store.put_artifact(b"scientific bytes\x00", "duplicate.bin", session_id=session["id"])
    assert first["sha256"] == hashlib.sha256(b"scientific bytes\x00").hexdigest()
    assert first["path"] == second["path"] and first["id"] != second["id"]
    assert first["size"] == 17 and first["name"] == "result.bin"
    reopened = module("storage").Store(store.root)
    assert reopened.read_artifact(first["id"]) == b"scientific bytes\x00"
    assert reopened.artifacts(session["id"], run["id"]) == [first]
    assert reopened.artifacts(session["id"]) == [first, second]
    assert reopened.artifacts() == [first, second]
    different = store.put_artifact("changed", "result.bin")
    assert different["path"] != first["path"]
    assert store.read_artifact(first["id"]) == b"scientific bytes\x00"
    assert store.read_artifact(different["id"]) == b"changed"


@pytest.mark.parametrize("name", ["../evil", "a/b", "a\\b", "/absolute", ".", "..", "", "a\x00b"])
def test_artifact_names_cannot_traverse_paths(tmp_path, name):
    store = module("storage").Store(tmp_path / "state")
    with pytest.raises(ValueError, match="name"):
        store.put_artifact(b"bad", name)
    assert store.artifacts() == []


def test_artifact_integrity_is_checked_on_read_and_duplicate_write(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    artifact = store.put_artifact(b"original", "original.txt")
    Path(artifact["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        store.read_artifact(artifact["id"])
    with pytest.raises(ValueError, match="integrity"):
        store.put_artifact(b"original", "duplicate.txt")
    assert len(store.artifacts()) == 1


def test_artifact_scope_and_symlink_escape_are_rejected(tmp_path):
    store, session, _, run = seeded_store(tmp_path)
    other = store.create_session({})
    with pytest.raises(ValueError, match="belong"):
        store.put_artifact(b"x", "result", session_id=other["id"], run_id=run["id"])
    with pytest.raises(ValueError, match="identifier"):
        store.read_artifact("../../outside")
    # An unscoped session can be derived from a validated run, not from arbitrary metadata.
    artifact = store.put_artifact(b"x", "result", run_id=run["id"])
    assert artifact["session_id"] == session["id"]
    destination = Path(artifact["path"])
    destination.unlink()
    destination.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink|integrity"):
        store.read_artifact(artifact["id"])
    with pytest.raises(ValueError, match="symlink|integrity"):
        store.put_artifact(b"x", "result")
    assert not (tmp_path / "outside").exists()


def test_parallel_artifact_puts_never_overwrite_content(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = list(pool.map(lambda i: store.put_artifact(b"same bytes", f"result-{i}"), range(8)))
    assert len({artifact["path"] for artifact in artifacts}) == 1
    assert len({artifact["id"] for artifact in artifacts}) == 8
    assert all(store.read_artifact(artifact["id"]) == b"same bytes" for artifact in artifacts)


def test_recursive_redaction_preserves_usage_and_native_tool_ids(tmp_path):
    supplied = "supplied-sensitive-value"
    store = module("storage").Store(tmp_path / "state", secrets=[supplied])
    payload = {"api_key": "fake-api-value", "nested": [{"Authorization": "Bearer fake-bearer-value",
               "password": "fake-password-value", "refreshToken": "fake-refresh-value",
               "credentials": {"file": "/private/key"}, "cookies": "fake-cookie-value"}],
               "usage": {"input_tokens": 31, "output_tokens": 4, "total_tokens": 35, "max_tokens": 100},
               "content": f"before {supplied} after", "token": 123456}
    clean = store.redact(payload)
    assert clean["api_key"] == "[REDACTED]"
    assert clean["token"] == "[REDACTED]"
    assert all(value == "[REDACTED]" for value in clean["nested"][0].values())
    assert clean["usage"] == payload["usage"]
    assert supplied not in clean["content"] and supplied in payload["content"]
    message = {"role": "assistant", "content": "key sk-fake_credential_0123456789", "tool_calls": [
        {"id": "call_native_1", "type": "function", "function": {"name": "remote", "arguments": json.dumps({"api_key": "fake-api-value", "x": 1})}},
    ]}
    cleaned_message = store.redact(message)
    call = cleaned_message["tool_calls"][0]
    assert call["id"] == "call_native_1"
    assert json.loads(call["function"]["arguments"]) == {"api_key": "[REDACTED]", "x": 1}
    assert "sk-fake_credential_0123456789" not in cleaned_message["content"]


def test_secrets_never_enter_sqlite_metadata_and_original_action_args_stay_ephemeral(tmp_path):
    secret = "supplied-sensitive-value"
    store = module("storage").Store(tmp_path / "state", secrets=[secret])
    session = store.create_session({"instructions": secret, "api_key": "fake-api-value"})
    task = store.create_task(session["id"], f"goal {secret}", {"password": "fake-password-value"})
    run = store.create_run(session["id"], task["id"], {"note": secret})
    store.update_session(session["id"], {"instructions": f"updated {secret}"}, expected_revision=0)
    store.update_task(task["id"], {"goal": f"updated {secret}", "acceptance": {"note": secret}})
    store.update_run(run["id"], {"state": {"secret": "fake-state-value"}, "usage": {"input_tokens": 11, "note": secret}})
    store.append_event(session["id"], "model.request", {"input": secret, "authorization": "fake-auth-value"}, run["id"])
    store.append_message(session["id"], {"role": "user", "content": secret}, run["id"])
    args = {"api_key": "fake-action-key", "message": secret}
    store.record_action(run["id"], "call_1", "remote", args, 0)
    assert args == {"api_key": "fake-action-key", "message": secret}
    store.finish_action(run["id"], "call_1", {"status": "ok", "data": {"password": "fake-result-value"}})
    store.finish_action(run["id"], "call_1", {"status": "ok", "data": {"password": "fake-result-value"}})
    artifact = store.put_artifact(secret, f"{secret}.txt", provenance={"source": secret, "secret": "fake-source-value"},
                                  session_id=session["id"], run_id=run["id"])
    # Exact artifact bytes remain private and unchanged; only metadata is a redacted audit record.
    assert store.read_artifact(artifact["id"]) == secret.encode()
    with sqlite3.connect(store.db_path) as connection:
        persisted = "\n".join(connection.iterdump())
    for value in (secret, "fake-api-value", "fake-password-value", "fake-state-value", "fake-auth-value",
                  "fake-action-key", "fake-result-value", "fake-source-value"):
        assert value not in persisted
    assert store.get_run(run["id"])["usage"]["input_tokens"] == 11
    with pytest.raises(ValueError, match="reused"):
        store.record_action(run["id"], "call_1", "remote", {"api_key": "different", "message": secret}, 0)


def test_public_transaction_extension_commits_rolls_back_and_closes(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    with store.transaction(write=True) as connection:
        connection.execute("CREATE TABLE research_extension (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO research_extension VALUES(?,?)", ("one", json.dumps(store.redact({"token": "fake-secret"}))))
    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction(write=True) as connection:
            connection.execute("INSERT INTO research_extension VALUES('two', '{}')")
            raise RuntimeError("abort")
    with store.transaction() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
        rows = connection.execute("SELECT * FROM research_extension").fetchall()
        assert len(rows) == 1 and json.loads(rows[0]["payload"])["token"] == "[REDACTED]"
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_trace_export_is_ordered_readable_redacted_and_hash_manifested(tmp_path):
    secret = "supplied-sensitive-value"
    store = module("storage").Store(tmp_path / "state", secrets=[secret])
    session = store.create_session({"instructions": "Evidence first"})
    task = store.create_task(session["id"], "Calculate a mean")
    run = store.create_run(session["id"], task["id"], {"max_steps": 3})
    store.append_message(session["id"], {"role": "user", "content": f"Mean please. {secret}"}, run["id"])
    store.record_action(run["id"], "call_native_1", "calculate", {"api_key": "fake-credential", "x": [2, 4]}, 2)
    store.finish_action(run["id"], "call_native_1", {"status": "ok", "data": {"mean": 3}})
    artifact = store.put_artifact(f"private scientific file {secret}", "result.txt", session_id=session["id"], run_id=run["id"])
    store.append_event(session["id"], "acceptance.checked", {"passed": True, "input_tokens": 10}, run["id"])
    store.update_run(run["id"], {"status": "succeeded"})
    exported = store.export_trace(session["id"])
    assert set(exported) == {"jsonl", "markdown", "manifest"}
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in exported.values())
    assert all(Path(path).parent == store.root / "traces" / session["id"] for path in exported.values())
    raw = {key: Path(path).read_text() for key, path in exported.items()}
    assert all(secret not in text and "fake-credential" not in text for text in raw.values())
    records = [json.loads(line) for line in raw["jsonl"].splitlines()]
    events = [record for record in records if record.get("record_type") == "event"]
    assert [event["seq"] for event in events] == [event["seq"] for event in store.events(session["id"])]
    assert any(record.get("record_type") == "action" and record["data"]["call_id"] == "call_native_1" for record in records)
    assert "# Session trace" in raw["markdown"] and "Mean please." in raw["markdown"]
    assert "calculate" in raw["markdown"] and "acceptance.checked" in raw["markdown"]
    assert artifact["sha256"] in raw["markdown"]
    manifest = json.loads(raw["manifest"])
    for kind in ("jsonl", "markdown"):
        entry = manifest["files"][kind]
        assert entry["sha256"] == hashlib.sha256(Path(exported[kind]).read_bytes()).hexdigest()
        assert entry["size"] == Path(exported[kind]).stat().st_size
    assert manifest["artifacts"] == [artifact]
    assert len(list(Path(exported["manifest"]).parent.iterdir())) == 3
    # Re-export creates a fresh snapshot and never overwrites earlier trace files.
    second = store.export_trace(session["id"])
    assert second != exported
    assert Path(exported["jsonl"]).read_text() == raw["jsonl"]


def test_run_trace_export_filters_other_runs_and_rejects_traversal(tmp_path):
    store, session, task, run = seeded_store(tmp_path)
    other_run = store.create_run(session["id"], task["id"], {})
    store.append_message(session["id"], {"role": "user", "content": "other-run-private"}, other_run["id"])
    store.record_action(other_run["id"], "call_other", "read", {}, 0)
    store.put_artifact("other bytes", "other.txt", run_id=other_run["id"])
    exported = store.export_trace(session["id"], run_id=run["id"])
    assert all(Path(path).parent == store.root / "traces" / run["id"] for path in exported.values())
    assert "other-run-private" not in Path(exported["jsonl"]).read_text()
    assert "call_other" not in Path(exported["jsonl"]).read_text()
    assert json.loads(Path(exported["manifest"]).read_text())["artifacts"] == []
    custom = store.export_trace(session["id"], destination=tmp_path / "explicit-export")
    assert Path(custom["manifest"]).parent == tmp_path / "explicit-export"
    with pytest.raises(ValueError, match="traversal"):
        store.export_trace(session["id"], destination="../outside")
    with pytest.raises(ValueError, match="identifier"):
        store.export_trace("../outside")
    destination = tmp_path / "symlink-export"
    destination.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        store.export_trace(session["id"], destination=destination)
    assert not (tmp_path / "elsewhere").exists()


def test_redaction_cannot_hide_an_action_tool_name_mismatch(tmp_path):
    store = module("storage").Store(tmp_path / "state", secrets=["private_tool_a", "private_tool_b"])
    session = store.create_session({})
    task = store.create_task(session["id"], "test")
    run = store.create_run(session["id"], task["id"], {})
    store.record_action(run["id"], "call_1", "private_tool_a", {}, 0)
    with pytest.raises(ValueError, match="reused"):
        store.record_action(run["id"], "call_1", "private_tool_b", {}, 0)


def test_read_transaction_is_query_only_and_schema_upgrade_is_not_silent(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    with store.transaction() as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema version"):
        module("storage").Store(store.root)


def test_terminal_run_outcome_cannot_be_rewritten(tmp_path):
    store, _, _, run = seeded_store(tmp_path)
    terminal = store.update_run(run["id"], {"status": "cancelled"})
    with pytest.raises(ValueError, match="terminal"):
        store.update_run(run["id"], {"status": "succeeded"})
    assert store.get_run(run["id"]) == terminal
    assert store.update_run(run["id"], {"status": "cancelled", "state": {"cleanup": "recorded"}})["status"] == "cancelled"


def test_artifact_blob_io_does_not_hold_a_database_write_lock(tmp_path, monkeypatch):
    store, session, _, run = seeded_store(tmp_path)
    real_put = store._put_blob

    def put_with_lock_probe(content):
        # Instrument only the filesystem boundary. The real blob writer still
        # runs, and the assertion is another real connection acquiring a lock.
        with sqlite3.connect(store.db_path, timeout=0.05, isolation_level=None) as observer:
            observer.execute("BEGIN IMMEDIATE")
            observer.execute("ROLLBACK")
        return real_put(content)

    monkeypatch.setattr(store, "_put_blob", put_with_lock_probe)
    artifact = store.put_artifact(b"bytes", "result.bin", session_id=session["id"], run_id=run["id"])
    assert store.read_artifact(artifact["id"]) == b"bytes"


def test_token_accounting_exemption_cannot_expose_credential_containers(tmp_path):
    store = module("storage").Store(tmp_path / "state")
    clean = store.redact({"refresh_tokens": {"primary": "fake-refresh-credential"},
                          "access_tokens": 123456, "input_tokens": 17,
                          "input_tokens_details": {"cached_tokens": 3}})
    assert clean["refresh_tokens"] == "[REDACTED]"
    assert clean["access_tokens"] == "[REDACTED]"
    assert clean["input_tokens"] == 17
    assert clean["input_tokens_details"] == {"cached_tokens": 3}


@pytest.mark.parametrize("text,credential", [
    ("Authorization: Basic dXNlcjpzdXBlcnNlY3JldA==", "dXNlcjpzdXBlcnNlY3JldA=="),
    ('tool output: {"api_key":"fictional-credential-value"}', "fictional-credential-value"),
])
def test_reviewed_credential_forms_never_enter_database_or_trace(tmp_path, text, credential):
    store = module("storage").Store(tmp_path / "state")
    session = store.create_session({})
    store.append_message(session["id"], {"role": "user", "content": text})
    store.append_event(session["id"], "tool.output", {"text": text})
    exported = store.export_trace(session["id"])
    with sqlite3.connect(store.db_path) as connection:
        persisted = "\n".join(connection.iterdump())
    assert credential not in persisted
    assert all(credential not in Path(path).read_text() for path in exported.values())


def test_reviewed_prompt_and_completion_token_counts_remain_numeric(tmp_path):
    store, _, _, run = seeded_store(tmp_path)
    usage = {"prompt_token_count": 10, "completion_token_count": 3, "total_token_count": 13}
    updated = store.update_run(run["id"], {"usage": usage})
    assert updated["usage"] == usage
    reopened = module("storage").Store(store.root)
    assert reopened.get_run(run["id"])["usage"] == usage
    assert store.redact({"refresh_tokens": {"value": "fake-secret"}, "access_tokens": 123456}) == {
        "refresh_tokens": "[REDACTED]", "access_tokens": "[REDACTED]",
    }


@pytest.mark.parametrize("operation", ["read", "duplicate_put"])
def test_fifo_artifact_corruption_is_rejected_without_blocking(tmp_path, operation):
    store = module("storage").Store(tmp_path / "state")
    artifact = store.put_artifact(b"original", "original.txt")
    path = Path(artifact["path"])
    path.unlink()
    os.mkfifo(path)
    # Use a real child process so a regression cannot leave a blocked worker in
    # pytest. subprocess.run kills/reaps it on timeout, bounding both operations.
    script = """
import sys
from adj_partner.storage import Store
store = Store(sys.argv[1])
try:
    if sys.argv[3] == 'read':
        store.read_artifact(sys.argv[2])
    else:
        store.put_artifact(b'original', 'duplicate.txt')
except ValueError as exc:
    assert 'integrity' in str(exc), str(exc)
    print('rejected nonregular artifact')
else:
    raise AssertionError('FIFO artifact was not rejected')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(store.root), artifact["id"], operation],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=2,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Artifact {operation} blocked on a FIFO instead of rejecting corruption")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rejected nonregular artifact"
    assert len(store.artifacts()) == 1
