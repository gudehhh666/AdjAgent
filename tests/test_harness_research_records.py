"""Synthetic research ledger tests; no live retrieval, models or scientific data."""

import importlib.util
import json

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store


def api():
    assert importlib.util.find_spec("adj_partner.research"), "research package must exist"
    from adj_partner import research
    return research


def scope(store, acceptance=None):
    session = store.create_session(SessionConfig().to_dict())["id"]
    task = store.create_task(session, "Synthetic research fixture", acceptance)["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return session, run, task


def setup(tmp_path, assessor=None, acceptance=None):
    store = Store(tmp_path / "research")
    sid, rid, tid = scope(store, acceptance)
    return store, api().ResearchStore(store, assessor=assessor), sid, rid, tid


def source(store, research, sid, rid, *, content="First finding. Second finding.", kind="literature", key="fixture", provenance=None):
    provenance = provenance or {"synthetic_fixture": True, "text_availability": "abstract",
                                "limitations": ["Synthetic material, not scientific evidence."]}
    artifact = store.put_artifact(content, "fixture.json" if isinstance(content, bytes) else "fixture.txt",
                                  "application/json" if isinstance(content, bytes) else "text/plain",
                                  provenance=provenance, session_id=sid, run_id=rid)
    record = research.register_source(sid, rid, artifact_id=artifact["id"], kind=kind,
                                      title="Synthetic source", provenance=provenance, idempotency_key=key)
    return record, artifact


def test_reopen_persists_sources_selections_claim_versions_and_journal(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    c = r.record_claim(sid, rid, text="A tentative idea", kind="proposal", evidence_ids=[e["id"]])
    c1 = r.revise_claim(sid, rid, c["id"], expected_revision=0,
                        text="A corrected idea", kind="proposal", evidence_ids=[e["id"]])
    reopened = api().ResearchStore(Store(store.root))
    assert c1["revision"] == 1
    assert reopened.get_claim(sid, rid, c["id"], revision=0)["text"] == "A tentative idea"
    assert reopened.get_claim(sid, rid, c["id"])["text"] == "A corrected idea"
    assert reopened.get_evidence(sid, rid, e["id"])["text"] == "First finding."
    assert len(reopened.events(sid, rid)) == 4
    assert len(reopened.snapshot(sid, rid)["history"]) == 4


def test_source_registration_is_idempotent_and_conflicts_fail(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, a = source(store, r, sid, rid)
    args = dict(artifact_id=a["id"], kind=s["kind"], title=s["title"],
                provenance=s["provenance"], idempotency_key="fixture")
    assert r.register_source(sid, rid, **args)["id"] == s["id"]
    args["title"] = "Conflicting title"
    with pytest.raises(ValueError, match="[Ii]dempotency"):
        r.register_source(sid, rid, **args)
    assert len(r.list_sources(sid, rid)) == 1


def test_artifact_scope_producing_run_and_corruption_are_checked(tmp_path):
    store, r, sid, rid, tid = setup(tmp_path)
    s, a = source(store, r, sid, rid)
    later = store.create_run(sid, tid, Budget().to_dict())["id"]
    with pytest.raises((ValueError, PermissionError)):
        r.register_source(sid, later, artifact_id=a["id"], kind="computed", title="Fake recomputation",
                          provenance={}, idempotency_key="other")
    (store.root / "artifacts" / a["sha256"]).write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="[Ii]ntegrity|hash"):
        r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    with pytest.raises(ValueError, match="[Ii]ntegrity|hash"):
        r.register_source(sid, rid, artifact_id=a["id"], kind="literature", title="Other",
                          provenance={}, idempotency_key="new")


def test_multiple_evidence_from_one_source_have_distinct_identity(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    first = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    second = r.select_evidence(sid, rid, s["id"], {"passage": "Second finding."})
    assert first["id"] != second["id"]
    assert first["source_id"] == second["source_id"] == s["id"]
    assert first["source_hash"] == s["artifact_hash"]
    assert first["locator"] == {"passage": "First finding.", "start": 0, "end": 14}
    assert len(r.list_evidence(sid, rid)) == 2


@pytest.mark.parametrize("selector", [{}, {"passage": "invented"}, {"passage": ""},
    {"pointer": "/missing"}, {"passage": "First", "pointer": ""}, {"passage": 1}, {"finding": "First"}])
def test_selectors_cannot_invent_evidence(tmp_path, selector):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    with pytest.raises((ValueError, TypeError)):
        r.select_evidence(sid, rid, s["id"], selector)
    assert r.list_evidence(sid, rid) == []


def test_json_pointer_extracts_actual_value_and_retains_userfile_kind(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid, content=b'{"a/b": [{"~name": 2.5}]}', kind="userfile")
    e = r.select_evidence(sid, rid, s["id"], {"pointer": "/a~1b/0/~0name"})
    assert e["value"] == 2.5 and e["kind"] == "userfile"
    assert e["locator"] == {"pointer": "/a~1b/0/~0name"}


def test_evidence_revision_is_optimistic_and_old_selection_is_retained(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    changed = r.reselect_evidence(sid, rid, e["id"], expected_revision=0,
                                  selector={"passage": "Second finding."})
    assert changed["revision"] == 1
    assert r.get_evidence(sid, rid, e["id"], revision=0)["text"] == "First finding."
    with pytest.raises(ValueError, match="[Ss]tale"):
        r.reselect_evidence(sid, rid, e["id"], expected_revision=0, selector={"passage": "First"})


def test_same_session_material_reuse_does_not_allow_cross_task_claim_access(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    c = r.record_claim(sid, rid, text="Proposal", kind="proposal", evidence_ids=[e["id"]])
    other_task = store.create_task(sid, "Another task")["id"]
    later = store.create_run(sid, other_task, Budget().to_dict())["id"]
    assert r.get_source(sid, later, s["id"])["run_id"] == rid
    assert r.get_evidence(sid, later, e["id"])["run_id"] == rid
    assert r.record_claim(sid, later, text="Reusing material", kind="proposal", evidence_ids=[e["id"]])["task_id"] == other_task
    with pytest.raises((PermissionError, KeyError)):
        r.get_claim(sid, later, c["id"])
    with pytest.raises((PermissionError, KeyError)):
        r.revise_claim(sid, later, c["id"], expected_revision=0, text="No", kind="rejected", evidence_ids=[])


def test_every_read_write_and_export_checks_session_and_run_scope(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First"})
    c = r.record_claim(sid, rid, text="Proposal", kind="proposal", evidence_ids=[e["id"]])
    foreign_sid, foreign_rid, _ = scope(store)
    operations = [lambda: r.get_source(foreign_sid, foreign_rid, s["id"]),
                  lambda: r.get_evidence(foreign_sid, foreign_rid, e["id"]),
                  lambda: r.get_claim(foreign_sid, foreign_rid, c["id"]),
                  lambda: r.select_evidence(foreign_sid, foreign_rid, s["id"], {"passage": "First"}),
                  lambda: r.record_claim(foreign_sid, foreign_rid, text="No", kind="proposal", evidence_ids=[e["id"]]),
                  lambda: r.list_sources(sid, foreign_rid), lambda: r.list_evidence(sid, foreign_rid),
                  lambda: r.list_claims(sid, foreign_rid), lambda: r.snapshot(sid, foreign_rid),
                  lambda: r.events(sid, foreign_rid)]
    for operation in operations:
        with pytest.raises((ValueError, PermissionError, KeyError)):
            operation()


@pytest.mark.parametrize("fields", [{"text": "", "kind": "proposal", "evidence_ids": []},
    {"text": "X", "kind": "supported", "evidence_ids": []},
    {"text": "X", "kind": "observation", "evidence_ids": ["missing"]},
    {"text": "X", "kind": "proposal", "evidence_ids": "not-list"}])
def test_claim_schema_and_evidence_reference_validation(tmp_path, fields):
    _, r, sid, rid, _ = setup(tmp_path)
    with pytest.raises((ValueError, TypeError, KeyError)):
        r.record_claim(sid, rid, **fields)
    assert r.list_claims(sid, rid) == []


def test_current_secrets_redact_reads_and_journal_and_mirror_failure_is_recoverable(tmp_path, monkeypatch):
    store, r, sid, rid, _ = setup(tmp_path)
    secret = "synthetic-long-secret-value"
    c = r.record_claim(sid, rid, text=f"Proposal {secret}", kind="proposal", evidence_ids=[])
    store.register_secrets([secret])
    assert secret not in json.dumps(r.get_claim(sid, rid, c["id"]))
    assert secret not in json.dumps(r.snapshot(sid, rid))
    def unavailable(*args, **kwargs):
        raise OSError("synthetic mirror unavailable")
    monkeypatch.setattr(store, "append_event", unavailable)
    r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Revised", kind="proposal", evidence_ids=[])
    assert len(r.events(sid, rid)) == 2
    assert len(r.snapshot(sid, rid)["history"]) == 2


@pytest.mark.parametrize("provenance", [{"n": float("nan")}, {1: "bad key"}, {"limitations": "not-list"},
    {"text_availability": "pretend-full-paper"}])
def test_source_provenance_rejects_malformed_or_nonfinite_json(tmp_path, provenance):
    store, r, sid, rid, _ = setup(tmp_path)
    artifact = store.put_artifact("fixture", "fixture.txt", session_id=sid, run_id=rid)
    with pytest.raises((ValueError, TypeError)):
        r.register_source(sid, rid, artifact_id=artifact["id"], kind="literature", title="Fixture",
                          provenance=provenance, idempotency_key="bad")
    assert r.list_sources(sid, rid) == []


def test_later_run_corrections_journal_actual_mutation_scope_and_time(tmp_path):
    from adj_partner.contracts import Budget
    store, r, sid, rid, tid = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First"})
    c = r.record_claim(sid, rid, text="Proposal", kind="proposal", evidence_ids=[e["id"]])
    later = store.create_run(sid, tid, Budget().to_dict())["id"]
    updated_e = r.reselect_evidence(sid, later, e["id"], expected_revision=0, selector={"passage": "Second"})
    updated_c = r.revise_claim(sid, later, c["id"], expected_revision=0, text="Revised", kind="proposal", evidence_ids=[e["id"]])
    journal = r.events(sid, later)
    assert len(journal) == 2
    assert all(event["run_id"] == later for event in journal)
    assert journal[0]["created_at"] > e["created_at"]
    assert journal[1]["created_at"] > c["created_at"]
    assert updated_e["run_id"] == rid and updated_c["run_id"] == rid
    mirrored = [event for event in store.events(sid, later) if event["kind"].startswith("research.")]
    assert len(mirrored) == 2


def test_source_registration_validates_actual_artifact_provenance_too(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    artifact = store.put_artifact("Synthetic", "fixture.txt", provenance={"limitations": "wrong type"}, session_id=sid, run_id=rid)
    with pytest.raises(ValueError):
        r.register_source(sid, rid, artifact_id=artifact["id"], kind="literature", title="Synthetic",
                          provenance={}, idempotency_key="fixture")


def test_journal_failure_rolls_back_state_write_atomically(tmp_path):
    import sqlite3
    store, r, sid, rid, _ = setup(tmp_path)
    with store.transaction(write=True) as conn:
        conn.execute("CREATE TRIGGER synthetic_journal_failure BEFORE INSERT ON research_events BEGIN SELECT RAISE(ABORT, 'synthetic journal failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="synthetic journal failure"):
        r.record_claim(sid, rid, text="Must roll back", kind="proposal", evidence_ids=[])
    assert r.list_claims(sid, rid) == [] and r.events(sid, rid) == []


def test_concurrent_revision_writers_cannot_both_win(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    _, r, sid, rid, _ = setup(tmp_path)
    c = r.record_claim(sid, rid, text="Initial", kind="proposal", evidence_ids=[])
    barrier = Barrier(2)
    def revise(label):
        barrier.wait()
        try:
            return r.revise_claim(sid, rid, c["id"], expected_revision=0, text=label, kind="proposal", evidence_ids=[])
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(revise, ["First", "Second"]))
    assert sum(result is not None for result in results) == 1
    assert r.get_claim(sid, rid, c["id"])["revision"] == 1


def test_idempotent_ingestion_accepts_equivalent_new_artifact_reference(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    original, artifact = source(store, r, sid, rid)
    same = store.put_artifact(store.read_artifact(artifact["id"]), artifact["name"], artifact["media_type"],
                              provenance=artifact["provenance"], session_id=sid, run_id=rid)
    repeated = r.register_source(sid, rid, artifact_id=same["id"], kind=original["kind"], title=original["title"],
                                  provenance=original["provenance"], idempotency_key="fixture")
    assert repeated["id"] == original["id"] and repeated["artifact_id"] == artifact["id"]
    assert len(r.list_sources(sid, rid)) == 1
