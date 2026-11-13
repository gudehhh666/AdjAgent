import json
import os
import pytest
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ProviderFailure
from adj_partner.research.staged_design import run_staged
from adj_partner.runtime.model_invocations import leased_run
from adj_partner.runtime.journal import RunBusy
from adj_partner.storage import Store
from test_staged_design import staged_fixture


def crashed_run(tmp_path,monkeypatch):
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[])
    output=tmp_path/'run'
    child=os.fork()
    if child==0:
        port.complete=lambda *a,**kw:os._exit(71)
        try:run_staged(dataset,'qa_001',output,bundle)
        finally:os._exit(72)
    _,status=os.waitpid(child,0)
    assert os.waitstatus_to_exitcode(status)==71
    # Simulate lease TTL expiry after a dead process without sleeping.
    store=Store(output/'state')
    with store.transaction(write=True) as conn:
        conn.execute('UPDATE runtime_leases SET expires=0,expires_monotonic=0')
    store.close()
    return dataset,bundle,port,output


def test_hard_crash_has_resume_identity_and_does_not_replay_unknown_request(tmp_path,monkeypatch):
    dataset,bundle,port,out=crashed_run(tmp_path,monkeypatch)
    assert (out/'manifest.json').exists()
    assert (out/'checkpoint.json').exists()
    result=run_staged(dataset,'qa_001',out,bundle,resume=True)
    assert result['status']=='failed'
    assert not port.requests
    assert 'unknown' in result['reason'].lower() or 'failed request' in result['reason'].lower()


def test_staged_resume_cannot_bypass_active_owner(tmp_path,monkeypatch):
    dataset,bundle,port,out=crashed_run(tmp_path,monkeypatch)
    manifest=json.loads((out/'manifest.json').read_text())
    before=(out/'manifest.json').read_bytes()
    store=Store(out/'state')
    with leased_run(store,manifest['run_id']):
        with pytest.raises(RunBusy):run_staged(dataset,'qa_001',out,bundle,resume=True)
    assert (out/'manifest.json').read_bytes()==before
    assert not port.requests
    store.close()


def test_terminal_failure_requires_explicit_new_attempt(tmp_path,monkeypatch):
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[ProviderFailure('fixture','Disconnected')])
    out=tmp_path/'run';first=run_staged(dataset,'qa_001',out,bundle)
    assert first['status']=='failed'
    with pytest.raises(ValueError,match='terminal|Terminal'):
        run_staged(dataset,'qa_001',out,bundle,resume=True)
    assert len(port.requests)==1


def test_staged_shared_budget_never_double_counts_synthesis(tmp_path,monkeypatch):
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[ModelReply(content='{}',usage={'input_tokens':20,'output_tokens':10}) for _ in range(12)])
    out=tmp_path/'run';result=run_staged(dataset,'qa_001',out,bundle)
    store=Store(out/'state')
    run=store.get_run(result['run_id'])
    from adj_partner.runtime.model_invocations import InvocationLedger
    assert InvocationLedger(store).usage(run['id'])['requests']==12
    assert run['usage'].get('accounted_input_tokens',0)==0
    assert run['state'].get('steps',0)==0
    store.close()


def test_resume_at_final_checkpoint_does_not_grant_eight_more_requests(tmp_path,monkeypatch):
    import adj_partner.research.staged_design as module
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[ModelReply(content='{}',usage={'input_tokens':20,'output_tokens':10}) for _ in range(12)])
    out=tmp_path/'run'
    child=os.fork()
    if child==0:
        module.export_reviews=lambda *a,**kw:os._exit(71)
        try:run_staged(dataset,'qa_001',out,bundle)
        finally:os._exit(72)
    _,status=os.waitpid(child,0)
    assert os.waitstatus_to_exitcode(status)==71
    store=Store(out/'state')
    with store.transaction(write=True) as conn:conn.execute('UPDATE runtime_leases SET expires=0,expires_monotonic=0')
    store.close()
    result=run_staged(dataset,'qa_001',out,bundle,resume=True)
    assert result['status']=='incomplete'
    assert result['synthesis_accounting']['requests']==12
    assert result['model_accounting']['requests']==12
    assert not port.requests


def test_lost_staged_owner_cannot_finalize_over_new_owner(tmp_path,monkeypatch):
    from adj_partner.runtime.journal import LeaseLost
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[])
    out=tmp_path/'run'
    def lose_owner(*args,**kwargs):
        manifest=json.loads((out/'manifest.json').read_text())
        store=Store(out/'state')
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE runtime_leases SET owner='replacement-driver' WHERE run_id=?",(manifest['run_id'],))
        store.close()
        return ModelReply(content='{}',usage={'input_tokens':20,'output_tokens':10})
    port.complete=lose_owner
    with pytest.raises(LeaseLost):run_staged(dataset,'qa_001',out,bundle)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='running'
    store=Store(out/'state')
    assert store.get_run(manifest['run_id'])['status']=='created'
    store.close()


def test_completed_critic_survives_crash_before_advancing_stage(tmp_path,monkeypatch):
    from adj_partner.research.critic import ResearchCritic
    from test_research_critic import verdict
    evidence=ModelReply(content=json.dumps({'brief':'Compare candidates','evidence_keys':[f'T{i}' for i in range(1,9)]}),usage={'input_tokens':20,'output_tokens':10})
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,[evidence,verdict()])
    out=tmp_path/'run'
    child=os.fork()
    if child==0:
        original=ResearchCritic.review
        def crash_after_review(*a,**kw):
            original(*a,**kw)
            os._exit(71)
        ResearchCritic.review=crash_after_review
        try:run_staged(dataset,'qa_001',out,bundle)
        finally:os._exit(72)
    _,status=os.waitpid(child,0)
    assert os.waitstatus_to_exitcode(status)==71
    store=Store(out/'state')
    with store.transaction(write=True) as conn:conn.execute('UPDATE runtime_leases SET expires=0,expires_monotonic=0')
    store.close()
    port._replies=iter([ProviderFailure('fixture','Stop after next decision',usage={'input_tokens':1,'output_tokens':0},cost_unknown=False)])
    result=run_staged(dataset,'qa_001',out,bundle,resume=True)
    assert len(port.requests)==1
    assert json.loads(port.requests[0]['messages'][1]['content'])['stage']=='decision'
    assert result['model_accounting']['requests']==3
