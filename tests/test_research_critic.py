import json
from copy import deepcopy
import pytest
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ScriptedPort,ProviderFailure
from test_harness_research_records import setup,source
from test_research_design import design

class Luna(ScriptedPort):
    @property
    def metadata(self):return {'model':'gpt-5.6-luna','provider_kind':'scripted'}

def verdict(ok=True,back='evidence'):
    return ModelReply(content=json.dumps({'verdict':'pass' if ok else 'revise','return_to':'none' if ok else back,'issues':[] if ok else ['Need direct evidence'],'rationale':'Fixture review, not scientific validation.'}),usage={'input_tokens':20,'output_tokens':10})

def fixture(tmp_path,replies):
    from adj_partner.research.critic import ResearchCritic
    store,research,sid,rid,_=setup(tmp_path,acceptance={'require_report':True,'require_evidence':True,'require_critic':True})
    s,_=source(store,research,sid,rid);e=research.select_evidence(sid,rid,s['id'],{'passage':'First finding.'});c=research.observe_evidence(sid,rid,e['id'])
    port=Luna(replies);critic=ResearchCritic(research,lambda sid:port)
    return research,sid,rid,e,c,critic,port

def test_three_gates_exact_report_and_upstream_rejection(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(),verdict(),verdict(),verdict(False)])
    d=design(c['id']);kw={'title':'Fixture','sections':[{'heading':'Choice','claim_ids':[c['id']]}],'limitations':['Fixture'],'research_design':d}
    with pytest.raises(ValueError,match='critic'):r.create_report(sid,rid,**kw)
    with pytest.raises(ValueError,match='evidence'):critic.review(sid,rid,stage='decision',brief='Choice',evidence_ids=[e['id']],research_design=d)
    critic.review(sid,rid,stage='evidence',brief='Two candidates, compare evidence',evidence_ids=[e['id']])
    critic.review(sid,rid,stage='decision',brief='Conditional choice',evidence_ids=[e['id']],research_design=d)
    report=r.create_report(sid,rid,**kw)
    assert not r.validate_report(sid,rid,report['id'])['accepted']
    critic.review(sid,rid,stage='report',brief='Audit',report_id=report['id'])
    assert r.validate_report(sid,rid,report['id'])['accepted']
    changed=r.create_report(sid,rid,**(kw|{'title':'New bytes'}))
    assert not r.validate_report(sid,rid,changed['id'])['accepted']
    critic.review(sid,rid,stage='evidence',brief='New evidence comparison',evidence_ids=[e['id']])
    assert not r.validate_report(sid,rid,report['id'])['accepted']
    assert all(len(x['messages'])==2 and x['tools']==[] for x in p.requests)

def test_failed_request_recorded_and_never_passes(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[ProviderFailure('provider_connection','Failed')])
    with pytest.raises(ProviderFailure):critic.review(sid,rid,stage='evidence',brief='Candidates',evidence_ids=[e['id']])
    events=r.store.events(sid,rid)
    assert any(x['kind']=='research.critic.error' for x in events)
    assert any(x['kind']=='research.critic.attempt' for x in events)


def test_changed_evidence_and_design_require_reassessment(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(),verdict()])
    d=design(c['id']);critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    critic.review(sid,rid,stage='decision',brief='Choose',evidence_ids=[e['id']],research_design=d)
    kw={'title':'Fixture','sections':[{'heading':'Choice','claim_ids':[c['id']]}],'limitations':['Fixture'],'research_design':d}
    altered=deepcopy(d);altered['selected_candidate']='B'
    with pytest.raises(ValueError,match='critic_decision_changed'):r.create_report(sid,rid,**(kw|{'research_design':altered}))
    r.reselect_evidence(sid,rid,e['id'],expected_revision=0,selector={'passage':'Second finding.'})
    with pytest.raises(ValueError,match='critic_evidence_changed'):r.create_report(sid,rid,**kw)


def test_bad_json_and_refusal_do_not_approve(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[ModelReply(content='not json'),ModelReply(refusal='Declined')])
    with pytest.raises(ValueError):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    with pytest.raises(ValueError,match='refusal'):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    with pytest.raises(ValueError,match='previously refused'):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert len(p.requests)==2


def test_new_workflow_is_explicit_and_old_workflow_unchanged():
    from adj_partner.workflows import load_workflow
    old=load_workflow('adjuvant-design-v1');new=load_workflow('adjuvant-design-critic-v1')
    assert old['version']==2 and not old['acceptance'].get('require_critic')
    assert new['version']==3 and new['acceptance']['require_critic']


def test_single_candidate_cannot_enter_decision(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(),verdict()])
    critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    d=design(c['id']);d['candidates']=d['candidates'][:1]
    with pytest.raises(ValueError,match='at least two'):
        critic.review(sid,rid,stage='decision',brief='Choose',evidence_ids=[e['id']],research_design=d)
    assert len(p.requests)==1


def test_critic_budget_and_separate_usage_export(tmp_path):
    from adj_partner.research.critic import export_reviews
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(False) for _ in range(9)])
    for _ in range(9):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    with pytest.raises(ValueError,match='budget exhausted'):
        critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    accounting=export_reviews(r.store,sid,rid)['accounting']
    assert accounting['requests']==9 and accounting['unknown_requests']==0
    assert accounting['input_tokens']==180 and accounting['output_tokens']==90
    assert accounting['separate_from_generator']


def test_source_preview_does_not_relabel_exact_fulltext_evidence(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict()])
    s,_=source(r.store,r,sid,rid,key='fulltext',provenance={'text_availability':'derived_full_text','synthetic_fixture':True})
    selected=r.select_evidence(sid,rid,s['id'],{'passage':'First finding.'})
    critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[selected['id']])
    packet=json.loads(p.requests[0]['messages'][1]['content'])
    assert packet['sources'][s['id']]['text_availability']=='derived_full_text'
    assert packet['sources'][s['id']]['source_preview_is_partial'] is True
    assert 'preview_only' not in packet['sources'][s['id']]
    assert packet['evidence'][0]['text']=='First finding.'

@pytest.mark.parametrize('malformed',[False,True])
def test_empty_critic_usage_is_unknown(malformed,tmp_path):
    from adj_partner.research.critic import export_reviews
    reply=verdict();reply.usage={}
    if malformed:reply.content='invalid json'
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[reply])
    if malformed:
        with pytest.raises(ValueError):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    else:critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert export_reviews(r.store,sid,rid)['accounting']['unknown_requests']==1


def test_explicit_critic_budget_extension_retains_original_attempts(tmp_path):
    r,sid,rid,e,c,critic,p=fixture(tmp_path,[verdict(False) for _ in range(10)])
    for _ in range(9):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    with pytest.raises(ValueError,match='budget'):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    r.store.append_event(sid,'research.critic.budget_extension',{'additional_requests':1,'reason':'Explicit fixture extension'},rid)
    critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    assert len(p.requests)==10
    with pytest.raises(ValueError,match='budget'):critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])

def test_packet_compaction_only_drops_supplementary_previews(tmp_path,monkeypatch):
    import adj_partner.research.critic as module
    r,sid,rid,e,c,critic,port=fixture(tmp_path,[verdict(),verdict()])
    critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    before=json.loads(port.requests[-1]['messages'][1]['content'])
    encoded=port.requests[-1]['messages'][1]['content'].encode()
    # A long supplementary preview is removable; evidence is not.
    source_id=next(iter(before['sources']))
    original=r._verify_source
    monkeypatch.setattr(r,'_verify_source',lambda s:original(s)+b' supplementary context'*150)
    monkeypatch.setattr(module,'MAX_INPUT_BYTES',len(encoded)+200)
    critic.review(sid,rid,stage='evidence',brief='Compare',evidence_ids=[e['id']])
    after=json.loads(port.requests[-1]['messages'][1]['content'])
    assert after['evidence']==before['evidence']
    assert after['sources'][source_id]['sha256']==before['sources'][source_id]['sha256']
    assert 'source_preview' not in after['sources'][source_id]
