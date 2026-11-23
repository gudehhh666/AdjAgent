import pytest
from adj_partner.contracts import Budget
from adj_partner.runtime.model_invocations import InvocationLedger, leased_run
from adj_partner.runtime.journal import RunBusy
from test_research_critic import fixture, verdict


def test_critic_uses_configured_model_and_shared_ledger(tmp_path):
    from adj_partner.models.base import ScriptedPort
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict()])
    p=ScriptedPort([verdict()]);critic.port_factory=lambda sid:p
    with leased_run(r.store,rid) as guard:
        result=critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']],_guard=guard)
    assert result['model']==p.metadata.get('model')
    assert InvocationLedger(r.store).usage(rid)['requests']==1


def test_standalone_critic_cannot_bypass_existing_owner(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict()])
    with leased_run(r.store,rid):
        with pytest.raises(RunBusy):
            critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert not p.requests


def test_critic_obeys_shared_remaining_budget(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict()])
    r.store.update_run(rid,{'budget':Budget(max_output_tokens=100).to_dict()})
    with pytest.raises(RuntimeError,match='budget'):
        critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert not p.requests


def test_engine_counts_auxiliary_requests_before_next_decision(tmp_path):
    from test_runtime_engine import setup,reply
    store,engine,sid,p,_=setup(tmp_path,[reply('Must not run')],budget=Budget(max_steps=1))
    rid=engine.start(sid,'Bound all model work')['id']
    from adj_partner.models.base import ScriptedPort
    with leased_run(store,rid) as guard:
        InvocationLedger(store).complete(sid,rid,ScriptedPort([reply('Auxiliary')]),[{'role':'user','content':'Auxiliary'}],[],role='critic',key='critic-1',max_output_tokens=100,timeout_seconds=10,guard=guard)
    result=engine.resume(rid)
    assert result['status']=='budget_exhausted'
    assert not p.requests


def test_critic_unknown_prior_request_is_not_retried_with_new_key(tmp_path):
    from adj_partner.models.base import ProviderFailure
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[ProviderFailure('connection','Unknown effect',cost_unknown=True),verdict()])
    with pytest.raises(ProviderFailure):
        critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    with pytest.raises(RuntimeError,match='unknown|Unknown'):
        critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert len(p.requests)==1


def test_ninth_critic_receipt_recovers_without_consuming_tenth_attempt(tmp_path,monkeypatch):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(False) for _ in range(8)]+[verdict()])
    for _ in range(8):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    original=r.store.append_event
    def crash(sid,kind,payload,run_id=None):
        if kind=='research.critic.review':raise KeyboardInterrupt('crash after durable model receipt')
        return original(sid,kind,payload,run_id)
    monkeypatch.setattr(r.store,'append_event',crash)
    with pytest.raises(KeyboardInterrupt):
        critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']],_request_key='recover-9')
    monkeypatch.setattr(r.store,'append_event',original)
    result=critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']],_request_key='recover-9')
    assert result['verdict']=='pass'
    assert len(p.requests)==9
    assert sum(e['kind']=='research.critic.attempt' for e in r.store.events(sid,rid))==9
