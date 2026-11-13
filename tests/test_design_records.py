"""Real SQLite and artifact tests for immutable design lineage (no providers)."""
from copy import deepcopy
import json
import pytest
from adj_partner.contracts import Budget
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from test_harness_research_records import setup, source
from test_research_design import design


def fixture(tmp_path):
    store, r, sid, rid, tid = setup(tmp_path)
    src, artifact = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, src['id'], {'passage': 'First finding.'})
    c = r.record_claim(sid, rid, text='Tentative', kind='proposal', evidence_ids=[e['id']])
    return store, r, sid, rid, tid, src, artifact, e, c


def chain(tmp_path):
    values = fixture(tmp_path)
    store, r, sid, rid, tid, src, artifact, e, c = values
    assert hasattr(r, 'create_finding'), 'Versioned discovery/design API is missing'
    f = r.create_finding(sid, rid, observation='Tentative finding', evidence_ids=[e['id']], limitations=['Synthetic'])
    b = r.create_design_brief(sid, rid, goal='Compare A/B', requirements=['Test outcome'], finding_ids=[f['id']])
    d = r.create_design_revision(sid, rid, brief_id=b['id'], design=design(c['id']))
    return values, f, b, d


def test_persistent_immutable_lineage_and_feedback_closure(tmp_path):
    (store, r, sid, rid, tid, _, _, e, c), f, b, d = chain(tmp_path)
    fb = r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='decision', issue='Clarify mechanism', affected_fields=['mechanism.text'], evidence_ids=[e['id']])
    before = r.get_design_record(sid, rid, d['id'])
    with pytest.raises(ValueError, match='changed'):
        r.create_design_revision(sid, rid, brief_id=b['id'], parent_id=d['id'], design=design(c['id']), feedback_ids=[fb['id']])
    changed = design(c['id']); changed['mechanism']['text'] = 'Revised testable explanation'
    d2 = r.create_design_revision(sid, rid, brief_id=b['id'], parent_id=d['id'], design=changed, feedback_ids=[fb['id']])
    assert d2['id'] != d['id'] and r.get_design_record(sid, rid, d['id']) == before
    reopened = ResearchStore(Store(store.root))
    lineage = reopened.design_lineage(sid, rid)
    assert lineage['feedback'][0]['resolved_by'] == [d2['id']]
    assert lineage['feedback'][0]['resolved'] is True
    assert len(reopened.list_design_records(sid, rid, kind='design_revision')) == 2
    assert reopened.validate_design_record(sid, rid, d2['id']) == []
    json.dumps(lineage)
    assert all({'id', 'kind', 'task_id', 'session_id', 'run_id', 'created_at', 'payload', 'hash'} <= set(x) for x in lineage['records'])
    with store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_records WHERE entity LIKE 'design%' OR entity='finding'").fetchone()[0] == 0
    later = store.create_run(sid, tid, Budget().to_dict())['id']
    assert reopened.get_design_record(sid, later, d2['id']) == d2


def test_scope_missing_references_and_artifact_integrity(tmp_path):
    (store, r, sid, rid, _, _, artifact, e, c), f, b, d = chain(tmp_path)
    tid2 = store.create_task(sid, 'Other task')['id']
    rid2 = store.create_run(sid, tid2, Budget().to_dict())['id']
    with pytest.raises((KeyError, PermissionError, ValueError)):
        r.create_finding(sid, rid2, observation='Foreign', evidence_ids=[e['id']], limitations=[])
    with pytest.raises(KeyError): r.get_design_record(sid, rid2, d['id'])
    assert r.list_design_records(sid, rid2) == []
    with pytest.raises((KeyError, ValueError)):
        r.create_design_revision(sid, rid, brief_id=b['id'], design=design('claim_missing'))
    b2 = r.create_design_brief(sid, rid, goal='Other', requirements=['Other'])
    with pytest.raises(ValueError, match='brief'):
        r.create_design_revision(sid, rid, brief_id=b2['id'], parent_id=d['id'], design=design(c['id']))
    (store.root / 'artifacts' / artifact['sha256']).write_bytes(b'tampered')
    with pytest.raises(ValueError):
        r.create_finding(sid, rid, observation='Bad', evidence_ids=[e['id']], limitations=[])
    assert r.validate_design_record(sid, rid, f['id'])


def test_upstream_evidence_and_claim_changes_are_detected(tmp_path):
    (_, r, sid, rid, _, _, _, e, c), f, b, d = chain(tmp_path)
    r.reselect_evidence(sid, rid, e['id'], expected_revision=0, selector={'passage': 'Second finding.'})
    for rec in (f, b, d):
        assert any('stale' in x.lower() for x in r.validate_design_record(sid, rid, rec['id']))
    r.revise_claim(sid, rid, c['id'], expected_revision=0, text='Changed', kind='proposal', evidence_ids=[])
    assert any(c['id'] in x for x in r.validate_design_record(sid, rid, d['id']))


def test_feedback_paths_and_lineage_targets_cannot_fake_resolution(tmp_path):
    (_, r, sid, rid, _, _, _, e, c), f, b, d = chain(tmp_path)
    with pytest.raises(ValueError):
        r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='decision', issue='Fake', affected_fields=['arbitrary'])
    with pytest.raises(ValueError):
        r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='decision', issue='Fake', affected_fields=[])
    fb = r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='decision', issue='Both', affected_fields=['mechanism', 'verification'])
    changed = design(c['id']); changed['mechanism']['text'] = 'Changed'
    with pytest.raises(ValueError, match='changed'):
        r.create_design_revision(sid, rid, brief_id=b['id'], parent_id=d['id'], design=changed, feedback_ids=[fb['id']])
    changed['verification']['text'] = 'Changed verification'
    with pytest.raises(ValueError):
        r.create_design_revision(sid, rid, brief_id=b['id'], design=changed, feedback_ids=[fb['id']])
    assert r.design_lineage(sid, rid)['feedback'][0]['resolved'] is False


def test_upstream_race_rolls_back_record_and_journal(tmp_path, monkeypatch):
    (store, r, sid, rid, _, _, _, e, c), f, b, d = chain(tmp_path)
    count = len(r.list_design_records(sid, rid))
    events = len(r.events(sid, rid))
    verify = r._verify_source
    fired = []
    def race(src):
        content = verify(src)
        if not fired:
            fired.append(True)
            r.reselect_evidence(sid, rid, e['id'], expected_revision=0,
                               selector={'passage': 'Second finding.'})
        return content
    monkeypatch.setattr(r, '_verify_source', race)
    with pytest.raises(ValueError, match='Stale'):
        r.create_finding(sid, rid, observation='Racing', evidence_ids=[e['id']], limitations=[])
    assert len(r.list_design_records(sid, rid)) == count
    assert len(r.events(sid, rid)) == events + 1  # Only evidence update committed.


def test_invalid_feedback_stage_and_unrelated_branch_are_rejected(tmp_path):
    (_, r, sid, rid, _, _, _, e, c), f, b, d = chain(tmp_path)
    with pytest.raises(ValueError, match='stage'):
        r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='arbitrary',
                                 issue='Fix', affected_fields=['mechanism'])
    fb = r.record_design_feedback(sid, rid, design_id=d['id'], target_stage='hypothesis',
                                 issue='Fix', affected_fields=['mechanism'])
    branch_design = design(c['id']); branch_design['mechanism']['text'] = 'A distinct unrelated branch'
    unrelated = r.create_design_revision(sid, rid, brief_id=b['id'], design=branch_design)
    changed = design(c['id']); changed['mechanism']['text'] = 'New mechanism'
    with pytest.raises(ValueError, match='ancestor'):
        r.create_design_revision(sid, rid, brief_id=b['id'], parent_id=unrelated['id'],
                                 design=changed, feedback_ids=[fb['id']])


def test_new_design_cannot_silently_refresh_a_stale_brief(tmp_path):
    (_, r, sid, rid, _, _, _, e, c), f, b, d = chain(tmp_path)
    r.reselect_evidence(sid, rid, e['id'], expected_revision=0, selector={'passage': 'Second finding.'})
    with pytest.raises(ValueError, match='[Ss]tale'):
        r.create_design_revision(sid, rid, brief_id=b['id'], parent_id=d['id'], design=design(c['id']))


def test_feedback_can_target_whole_design(tmp_path):
    from test_harness_research_records import setup,source
    from test_research_design import design
    store,r,sid,rid,_=setup(tmp_path)
    s,_=source(store,r,sid,rid);e=r.select_evidence(sid,rid,s['id'],{'passage':'First finding.'})
    c=r.observe_evidence(sid,rid,e['id'])
    brief=r.create_design_brief(sid,rid,goal='Goal',requirements=['Choose'],evidence_ids=[e['id']])
    old=r.create_design_revision(sid,rid,brief_id=brief['id'],design=design(c['id']))
    feedback=r.record_design_feedback(sid,rid,design_id=old['id'],target_stage='decision',issue='Change selection',affected_fields=['$'],evidence_ids=[e['id']])
    changed=design(c['id']);changed['selected_candidate']='B'
    new=r.create_design_revision(sid,rid,brief_id=brief['id'],design=changed,parent_id=old['id'],feedback_ids=[feedback['id']])
    assert r.design_lineage(sid,rid)['feedback'][0]['resolved']


def test_record_replay_is_idempotent_for_every_kind(tmp_path):
    (store, r, sid, rid, tid, _, _, e, c), f, b, d = chain(tmp_path)
    fb_args = dict(design_id=d['id'], target_stage='decision', issue='Clarify mechanism',
                   affected_fields=['mechanism.text'], evidence_ids=[e['id']])
    fb = r.record_design_feedback(sid, rid, **fb_args)
    before = len(r.events(sid, rid))
    assert r.create_finding(sid, rid, observation='Tentative finding', evidence_ids=[e['id']], limitations=['Synthetic']) == f
    assert r.create_design_brief(sid, rid, goal='Compare A/B', requirements=['Test outcome'], finding_ids=[f['id']]) == b
    assert r.create_design_revision(sid, rid, brief_id=b['id'], design=design(c['id'])) == d
    assert r.record_design_feedback(sid, rid, **fb_args) == fb
    assert len(r.events(sid, rid)) == before
    assert len(r.list_design_records(sid, rid)) == 4
    later = store.create_run(sid, tid, Budget().to_dict())['id']
    distinct = r.create_finding(sid, later, observation='Tentative finding', evidence_ids=[e['id']], limitations=['Synthetic'])
    assert distinct['id'] != f['id']  # Deliberate new run remains a new attempt.


def test_commit_before_checkpoint_failure_reuses_record_after_reopen(tmp_path, monkeypatch):
    store, r, sid, rid, tid, _, _, e, c = fixture(tmp_path)
    before = len(r.events(sid, rid))
    def interrupted(*args):
        raise RuntimeError('Simulated post-commit process failure')
    monkeypatch.setattr(r, '_mirror', interrupted)
    args = dict(observation='Crash finding', evidence_ids=[e['id']], limitations=[])
    with pytest.raises(RuntimeError, match='post-commit'):
        r.create_finding(sid, rid, **args)
    stored = r.list_design_records(sid, rid, kind='finding')[0]
    reopened = ResearchStore(Store(store.root))
    assert reopened.create_finding(sid, rid, **args) == stored
    assert len(reopened.events(sid, rid)) == before + 1
    assert len(reopened.list_design_records(sid, rid, kind='finding')) == 1


def test_concurrent_record_replays_create_single_record_and_event(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    store, r, sid, rid, _, _, _, e, _ = fixture(tmp_path)
    other = ResearchStore(Store(store.root))
    before = len(r.events(sid, rid))
    barrier = threading.Barrier(2)
    for research in (r, other):
        original = research._verify_source
        def verify(source, original=original):
            data = original(source)
            barrier.wait(timeout=5)
            return data
        monkeypatch.setattr(research, '_verify_source', verify)
    def create(research):
        return research.create_finding(sid, rid, observation='Concurrent', evidence_ids=[e['id']], limitations=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(create, (r, other)))
    assert first == second
    assert len(r.events(sid, rid)) == before + 1
    assert len(r.list_design_records(sid, rid, kind='finding')) == 1
