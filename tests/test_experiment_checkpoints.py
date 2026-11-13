"""Offline control-flow and evidence-contract tests; not scientific validation."""
import json
from copy import deepcopy
import pytest
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.research.design import validate_design
from test_framework_design import seeded


def checkpoint(i, branch='mechanism'):
    return dict(id=f'C{i}', branch=branch, hypothesis='BHMT-dependent explanation under the supplied cell context',
                current_evidence='RNA does not establish protein activity', readout='Protein and catalytic activity',
                controls='Matched vehicle, positive assay reference and sample quality',
                quality_rule='Below sensitivity or failed control is inconclusive, not a biological negative',
                supports='Detectable catalytic capability permits further mechanism testing, not a causal conclusion',
                opposes='Reliable absence weakens this mechanism, not all candidate effects',
                inconclusive='Resolve assay sensitivity before changing the mechanism claim',
                priority_reason='Test a necessary condition before expensive perturbation',
                depends_on=[], claim_ids=['P1'])


def experimental(tmp_path, missing=False):
    s,f,t,p=seeded(tmp_path, workflow_version='unified-experiment-v1')
    original=p.complete
    def complete(messages,tools,**kwargs):
        result=original(messages,tools,**kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            value=json.loads(result.content)
            if missing:value['research_design']['verification'].pop('checkpoints',None)
            else:value['research_design']['verification']['checkpoints']=[checkpoint(1),checkpoint(2),checkpoint(3,'function')]
            from adj_partner.contracts import ModelReply
            return ModelReply(content=json.dumps(value),usage=result.usage)
        return result
    p.complete=complete
    return s,f,t,p


def test_new_default_and_explicit_legacy_identity():
    assert TaskSpec('design','Q').workflow_version=='unified-experiment-v1'
    assert TaskSpec.from_dict({'task_type':'design','question':'Q','workflow_version':'unified-v1'}).workflow_version=='unified-v1'


def test_no_critic_and_checkpoints_survive_report_and_resume(tmp_path):
    s,f,t,p=experimental(tmp_path)
    try:
        assert 'research_critic' not in s.store.get_session(t['session_id'])['config']['allowed_tools']
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='succeeded',result.get('error')
        assert result['review_status']=='not_run'
        assert result['empirical_validation_status']=='not_performed'
        for req in p.requests:
            if not req['tools']:assert 'stage' not in json.loads(req['messages'][-1]['content'])
        records=[r for r in result['lineage']['records'] if r['kind']=='design_revision']
        assert len(records)==1
        cards=records[0]['payload']['design']['verification']['checkpoints']
        assert len(cards)==3 and cards[2]['branch']=='function'
        assert cards[0]['claim_ids'][0].startswith('claim_')
        for view in ('content','readable_content'):
            assert 'BHMT-dependent explanation' in result['delivery'][view]
            assert 'Below sensitivity' in result['delivery'][view]
        assert result['delivery']['experiment_status']=='planned'
        calls=len(p.requests)
        assert f.run(t['fingerprint'])==result and len(p.requests)==calls
    finally:s.close()


def test_missing_cards_stops_without_critic_or_rewrite(tmp_path):
    s,f,t,p=experimental(tmp_path,missing=True)
    try:
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='failed'
        assert 'checkpoints' in result['error']['message']
        assert p.decisions==1 and p.decision_reviews==0
        assert result['delivery_status']=='none'
    finally:s.close()


@pytest.mark.parametrize('mutation', ['unknown_claim','cycle','missing_quality','function_missing','too_few'])
def test_checkpoint_contract_rejects_untraceable_or_undecidable_cards(mutation):
    from adj_partner.task_modes.design import decision_prompt
    d=json.loads(decision_prompt('Q',{},None)[1]['content'])['schema']['research_design']
    d['requirements'][0]['status']='addressed'
    d['verification']['checkpoints']=[checkpoint(1),checkpoint(2),checkpoint(3,'function')]
    cards=d['verification']['checkpoints']
    if mutation=='unknown_claim':cards[0]['claim_ids']=['outside']
    if mutation=='cycle':cards[0]['depends_on']=['C2'];cards[1]['depends_on']=['C1']
    if mutation=='missing_quality':cards[0]['quality_rule']=''
    if mutation=='function_missing':cards[2]['branch']='mechanism'
    if mutation=='too_few':cards.pop()
    with pytest.raises(ValueError,match='checkpoint'):
        validate_design(d,['P1'])


@pytest.mark.parametrize('phase',['report','report_validate'])
def test_experiment_resume_after_persisted_stage_is_idempotent(tmp_path,phase):
    s,f,t,p=experimental(tmp_path)
    save=f.repository.save
    interrupted=False
    def crash(fp,state,**kwargs):
        nonlocal interrupted
        value=save(fp,state,**kwargs)
        if state['phase']==phase and not interrupted:
            interrupted=True
            raise KeyboardInterrupt('simulated interruption after persistence')
        return value
    try:
        f.repository.save=crash
        with pytest.raises(KeyboardInterrupt):f.run(t['fingerprint'])
        before=p.decisions
        f.repository.save=save
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='succeeded',result.get('error')
        assert p.decisions==before==1
        assert result['review_status']=='not_run'
        from adj_partner.orchestration.exports import export_task
        exported=export_task(f,t['fingerprint'],tmp_path/'export')
        assert exported['artifact_verification']['status']=='verified'
        assert 'BHMT-dependent explanation' in (tmp_path/'export/report.md').read_text()
    finally:s.close()


def test_checkpoint_claims_are_in_design_evidence_map(tmp_path):
    s,f,t,p=experimental(tmp_path)
    try:
        result=f.run(t['fingerprint'])
        rows=result['delivery']['research_evidence_map']['items']
        cards=[v for v in rows if v['kind']=='checkpoint']
        assert [v['label'] for v in cards]==['C1','C2','C3']
        assert all(v['source_ids'] and v['evidence_ids'] for v in cards)
        report=s.research.get_report(t['session_id'],f.repository.get(t['fingerprint'])['state']['stage_run'],result['delivery']['report_id'])
        assert all(v['claim_ids']==report['research_design']['verification']['claim_ids'] for v in cards)
    finally:s.close()


def test_checkpoint_cannot_bind_only_an_unsupported_proposal(tmp_path):
    s,f,t,p=experimental(tmp_path)
    original=p.complete
    def complete(messages,tools,**kwargs):
        reply=original(messages,tools,**kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            value=json.loads(reply.content)
            value['proposals'][0]['evidence_keys']=[]
            from adj_partner.contracts import ModelReply
            return ModelReply(content=json.dumps(value),usage=reply.usage)
        return reply
    p.complete=complete
    try:
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='failed'
        assert 'checkpoint' in result['error']['message']
        assert 'evidence' in result['error']['message']
        assert p.decisions==1
    finally:s.close()
