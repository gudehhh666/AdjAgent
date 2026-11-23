"""Independent non-author acceptance controls for the composed application."""

import json
from pathlib import Path
from threading import Event

import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.models import ScriptedPort
from adj_partner.runtime.context import ContextView
from adj_partner.tools.catalog import canonical_json


def test_application_predrive_detach_issues_no_provider_request_and_recovers(tmp_path):
    port = ScriptedPort([ModelReply(content="after reconnect")])
    service = AgentService(tmp_path / "state", model=port)
    sid = service.create_session()["id"]
    rid = service.submit(sid, "question")["id"]
    stop = Event(); stop.set()
    waiting = service.drive(rid, stop_event=stop)
    assert waiting["status"] == "waiting" and not port.requests and waiting["state"]["steps"] == 0
    assert service.drive(rid)["status"] == "succeeded" and len(port.requests) == 1
    service.close()


def test_current_secret_never_survives_old_preview_in_any_export_record(tmp_path):
    port = ScriptedPort([])
    service = AgentService(tmp_path / "state", model=port)
    sid = service.create_session()["id"]; rid = service.submit(sid, "question")["id"]
    secret = "REVIEW_LATE_CREDENTIAL_" + "abcdefghijk" * 6
    prefix = len(canonical_json({"data": {"long": ""}, "status": "ok"}).split('"long":"')[0]) + len('"long":"')
    message = ContextView(service.store).tool_message(sid, rid, "call", {"status": "ok", "data": {"long": "x" * (3000 - prefix - len(secret) + 1) + secret + "z" * 10000}})
    service.store.append_message(sid, message, rid)
    nested = service.store.put_artifact(canonical_json({"messages": [message]}), "request.json", "application/json",
        {"kind": "public_model_request"}, session_id=sid, run_id=rid)
    original = service.store.read_artifact(nested["id"])
    service.store.register_secrets([secret])
    result = service.trace(sid, tmp_path / "export")
    for path in Path(result["manifest"]).parent.rglob("*"):
        if path.is_file(): assert secret[:-1].encode() not in path.read_bytes()
    assert service.store.read_artifact(nested["id"]) == original
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert next(a for a in manifest["artifacts"] if a["id"] == nested["id"])["export_status"] == "omitted"
    service.close()


def test_trace_metadata_remains_one_snapshot_when_copy_phase_sees_new_event(tmp_path, monkeypatch):
    service = AgentService(tmp_path / "state", model=ScriptedPort([]))
    sid = service.create_session()["id"]
    service.store.put_artifact("public", "public.txt", "text/plain", {"visibility": "public"}, session_id=sid)
    original = service.store.read_artifact
    def concurrent(*args, **kwargs):
        service.store.append_event(sid, "review.after_snapshot", {"marker": "AFTER_SNAPSHOT_ONLY"})
        return original(*args, **kwargs)
    monkeypatch.setattr(service.store, "read_artifact", concurrent)
    result = service.trace(sid, tmp_path / "export")
    assert any(e["kind"] == "review.after_snapshot" for e in service.events(sid))
    assert "AFTER_SNAPSHOT_ONLY" not in Path(result["jsonl"]).read_text()
    assert json.loads(Path(result["manifest"]).read_text())["snapshot_consistency"] == "single_sql_read_transaction"
    service.close()


def test_reopening_requires_explicit_model_identity_while_readonly_still_works(tmp_path):
    port = ScriptedPort([ModelReply(content="stored answer")])
    first = AgentService(tmp_path / "state", model=port)
    sid = first.create_session()["id"]; rid = first.submit(sid, "hello")["id"]
    first.drive(rid); first.close()
    second = AgentService(tmp_path / "state")
    assert second.run(rid)["status"] == "succeeded"
    assert second.messages(sid)[-1]["content"] == "stored answer"
    # Read-only trace does not silently instantiate a network model or infer
    # that the old scripted model is a valid production backend.
    assert second.model_info(sid)["loaded"] is False
    assert Path(second.trace(sid, tmp_path / "trace")["manifest"]).is_file()
    second.close()


def test_activity_snapshot_does_not_mix_in_a_concurrent_later_event(tmp_path, monkeypatch):
    from contextlib import contextmanager
    service = AgentService(tmp_path / "state", model=ScriptedPort([]))
    sid = service.create_session()["id"]; rid = service.submit(sid, "question")["id"]
    other = AgentService(tmp_path / "state", model=ScriptedPort([]))
    original, inserted = service.store.transaction, []
    class Connection:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, *args, **kwargs):
            if "FROM events" in sql and not inserted:
                inserted.append(True)
                other.store.append_event(sid, "job.progress", {"message": "AFTER_ACTIVITY_SNAPSHOT"}, rid)
            return self.conn.execute(sql, *args, **kwargs)
    @contextmanager
    def wrapped(*args, **kwargs):
        with original(*args, **kwargs) as conn: yield Connection(conn)
    monkeypatch.setattr(service.store, "transaction", wrapped)
    try:
        first = service.activity_after(sid, rid)
        assert inserted and "AFTER_ACTIVITY_SNAPSHOT" not in json.dumps(first)
        later = service.activity_after(sid, rid, first["next_cursor"])
        assert "AFTER_ACTIVITY_SNAPSHOT" in json.dumps(later)
    finally: other.close(); service.close()


def test_activity_paging_never_reloads_large_payloads_and_keeps_all_event_ids(tmp_path):
    service = AgentService(tmp_path / "state", model=ScriptedPort([]))
    sid = service.create_session()["id"]; rid = service.submit(sid, "question")["id"]
    payload = canonical_json({"history_only": "x" * 32768})
    with service.store.transaction(write=True) as conn:
        conn.executemany("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
            [(sid, rid, "large_history", payload, "2026-08-31T00:00:00Z")] * 512)
        expected = [row[0] for row in conn.execute("SELECT seq FROM events WHERE session_id=? AND run_id=? ORDER BY seq", (sid, rid))]
    found, cursor = [], 0
    try:
        while True:
            page = service.activity_after(sid, rid, cursor)
            assert len(canonical_json(page)) < 16384 and len(page["events"]) <= 64
            assert all("payload" not in event for event in page["events"])
            found.extend(event["seq"] for event in page["events"])
            if not page["events"]: break
            cursor = page["next_cursor"]
        assert found == expected and len(found) == len(set(found))
    finally: service.close()
