from copy import deepcopy
import pytest
from adj_partner.contracts import Budget
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.repository import WorkflowRepository
from test_model_invocations import setup


def test_create_attempt_interruption_rolls_back_every_record(tmp_path,monkeypatch):
    store,sid,rid=setup(tmp_path);run=store.get_run(rid);repo=WorkflowRepository(store)
    spec=TaskSpec('qa','Q');item=repo.bind(spec,sid,run['task_id'],rid)
    state=deepcopy(item['state']);before=deepcopy(state)
    original=store._event;old_task=store.get_task(run['task_id']);old_events=store.events(sid)
    def interrupted(conn,sid,kind,payload,run_id=None):
        original(conn,sid,kind,payload,run_id)
        if kind=='message.appended':raise KeyboardInterrupt('interrupt before attempt binding commits')
    with repo.owned(spec.fingerprint) as guard:
        monkeypatch.setattr(store,'_event',interrupted)
        with pytest.raises(KeyboardInterrupt):
            repo.create_attempt(spec.fingerprint,state,'stage_run',Budget().to_dict(),guard=guard,message={'role':'user','content':'Next'})
        monkeypatch.setattr(store,'_event',original)
        assert len(store.list_runs(sid))==1
        assert store.get_task(run['task_id'])==old_task
        assert store.events(sid)==old_events
        assert store.messages(sid)==[]
        assert repo.get(spec.fingerprint)['state']==before==state
        new=repo.create_attempt(spec.fingerprint,state,'stage_run',Budget().to_dict(),guard=guard)
        assert repo.get(spec.fingerprint)['state']['attempts']==[rid,new]
    store.close()


def test_create_attempt_after_commit_interruption_reuses_bound_run(tmp_path):
    store,sid,rid=setup(tmp_path);run=store.get_run(rid);repo=WorkflowRepository(store)
    spec=TaskSpec('qa','Q');item=repo.bind(spec,sid,run['task_id'],rid)
    class InterruptedState(dict):
        def clear(self):raise KeyboardInterrupt('after SQL commit before caller state update')
    stale=deepcopy(item['state'])
    with repo.owned(spec.fingerprint) as guard:
        with pytest.raises(KeyboardInterrupt):
            repo.create_attempt(spec.fingerprint,InterruptedState(stale),'stage_run',Budget().to_dict(),guard=guard)
        committed=repo.get(spec.fingerprint)['state']['stage_run']
        recovered=repo.create_attempt(spec.fingerprint,stale,'stage_run',Budget().to_dict(),guard=guard)
        assert recovered==committed
        assert len(store.list_runs(sid))==2
        assert stale['attempts']==[rid,committed]
    store.close()
