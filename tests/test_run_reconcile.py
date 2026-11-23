"""Closing runs left by dead drivers, and resuming campaigns after a crash."""
import json
import sqlite3

import pytest

from adj_partner import doctor
from adj_partner.application import AgentService
from adj_partner.contracts import Budget, SessionConfig
from adj_partner.models.base import ProviderFailure, ScriptedPort
from adj_partner.orchestration import campaign as campaign_module
from adj_partner.orchestration.controller import AgentFramework
from adj_partner.runtime.journal import RunJournal
from adj_partner.storage import Store
from test_framework_cli import port


def stranded(tmp_path, *, in_flight, age="2026-01-01T00:00:00+00:00"):
    store = Store(tmp_path / "state")
    sid = store.create_session(SessionConfig().to_dict())["id"]
    journal = RunJournal(store)
    run = journal.start(sid, "hello", {}, Budget().to_dict())
    state = dict(run["state"], phase="requesting" if in_flight else "ready")
    if in_flight:
        state["pending_request"] = {"estimated_input_tokens": 100, "max_output_tokens": 50}
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE runs SET status='running', state=? WHERE id=?", (json.dumps(state), run["id"]))
        conn.execute("UPDATE runs SET created_at=? WHERE id=?", (age, run["id"]))
        conn.execute("UPDATE events SET created_at=? WHERE run_id=?", (age, run["id"]))
    return store, journal, run["id"]


def test_abandoned_in_flight_run_is_closed_with_conservative_cost(tmp_path):
    store, journal, rid = stranded(tmp_path, in_flight=True)
    run = journal.abandon(rid, idle_seconds=3600)
    assert run["status"] == "failed" and run["state"]["stop_reason"] == "abandoned"
    assert "unknown completion and cost" in run["state"]["error"]["message"]
    assert run["usage"]["accounted_input_tokens"] == 100 and run["usage"]["accounted_output_tokens"] == 50
    assert run["usage"]["unknown_requests"] == 1
    assert [e["kind"] for e in store.events(run["session_id"], run_id=rid)][-1] == "runtime.abandoned"
    assert journal.abandon(rid, idle_seconds=3600) is None


def test_recent_or_leased_runs_are_left_alone(tmp_path):
    store, journal, rid = stranded(tmp_path, in_flight=False, age="2999-01-01T00:00:00+00:00")
    assert journal.abandon(rid, idle_seconds=3600) is None
    store, journal, rid = stranded(tmp_path / "other", in_flight=False)
    journal.acquire(rid, ttl=60)
    assert journal.abandon(rid, idle_seconds=3600) is None
    assert store.get_run(rid)["status"] == "running"


def test_reconcile_is_dry_by_default_and_refuses_sealed_evidence(tmp_path):
    store, _, rid = stranded(tmp_path, in_flight=False)
    dry = doctor.reconcile(store.root, root=tmp_path)
    assert dry["running_candidates"] == [rid] and store.get_run(rid)["status"] == "running"
    applied = doctor.reconcile(store.root, apply=True, root=tmp_path)
    assert applied["abandoned"] == [rid] and store.get_run(rid)["status"] == "failed"
    sealed = tmp_path / "outputs" / "x" / "state"
    sealed.mkdir(parents=True)
    with pytest.raises(ValueError, match="sealed"):
        doctor.reconcile(sealed, root=tmp_path)


def queue(tmp_path, model):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"id": "a", "question": "Q"}]))
    q = campaign_module.Campaign(AgentFramework(AgentService(tmp_path / "state", model=model)))
    return q, q.prepare(dataset, task_type="qa", model="offline-script")


def test_item_left_running_by_a_dead_process_is_reclaimed(tmp_path, monkeypatch):
    p = port()
    q, manifest = queue(tmp_path, p)
    item = manifest["items"][0]
    item.update(status="running", claimed_by={"host": "elsewhere", "pid": 1, "at": 0})
    q._save_item(manifest["id"], item)
    result = q.execute(manifest["id"])
    assert result["items"][0]["status"] == "succeeded" and result["items"][0]["reclaimed"] == 1


def test_live_claim_is_not_reclaimed(tmp_path):
    q, manifest = queue(tmp_path, port())
    item = manifest["items"][0]
    item.update(status="running", claimed_by={"host": "elsewhere", "pid": 1, "at": 9e18})
    q._save_item(manifest["id"], item)
    assert q.execute(manifest["id"])["items"][0]["status"] == "running"


def test_only_transient_failures_are_retried_and_only_on_request(tmp_path):
    q, manifest = queue(tmp_path, port())
    item = manifest["items"][0]
    item.update(status="failed", error={"code": "provider_connection", "message": "Provider connection failed"})
    q._save_item(manifest["id"], item)
    assert q.execute(manifest["id"])["items"][0]["status"] == "failed"
    retried = q.execute(manifest["id"], retry_failed=True)["items"][0]
    assert retried["status"] == "succeeded" and retried["retries"] == 1


def test_uncertain_failure_is_not_retried_even_when_requested(tmp_path):
    p = ScriptedPort([ProviderFailure("timeout", "Unknown external completion")])
    q, manifest = queue(tmp_path, p)
    first = q.execute(manifest["id"])
    assert first["items"][0]["status"] in {"failed", "needs_attention"}
    again = q.execute(manifest["id"], retry_failed=True)
    assert again["items"][0]["status"] in {"failed", "needs_attention"} and len(p.requests) == 1
