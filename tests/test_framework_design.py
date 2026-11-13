import json
from copy import deepcopy
from pathlib import Path
import pytest
from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework
from test_harness_research_records import source


class DesignPort:
    secrets=()
    metadata={'model':'gpt-5.6-luna','provider_kind':'scripted'}
    def __init__(self,revise=False):self.requests=[];self.revise=revise;self.decisions=0;self.decision_reviews=0
    def complete(self,messages,tools,**kwargs):
        self.requests.append({'messages':messages,'tools':tools})
        system=messages[0].get('content','')
        if tools:return ModelReply(content='Evidence brief comparing candidate A and B; synthetic fixture only.',usage={'input_tokens':20,'output_tokens':10})
        data=json.loads(messages[-1]['content'])
        if 'schema' in data:
            self.decisions+=1
            d=deepcopy(data['schema']);keys=list(data['evidence'])
            d['proposals']=[{'key':'P1','text':'Conditional synthetic design.', 'evidence_keys':keys}]
            design=d['research_design'];design['requirements'][0]['status']='addressed'
            design['selected_candidate']='B' if self.decisions>1 else 'A'
            return ModelReply(content=json.dumps(d),usage={'input_tokens':20,'output_tokens':10})
        if 'stage' in data:
            if data['stage']=='decision':self.decision_reviews+=1
            back=self.revise and data['stage']=='decision' and self.decision_reviews==1
            d={'verdict':'revise' if back else 'pass','return_to':'decision' if back else 'none','issues':['Select B given counterevidence'] if back else [],'rationale':'Synthetic review only'}
            return ModelReply(content=json.dumps(d),usage={'input_tokens':20,'output_tokens':10})
        claims=[c['id'] for c in data['claims']]
        d={'title':'Synthetic conceptual design','sections':[{'heading':h,'paragraphs':[{'text':'Conditional proposal bounded by synthetic observations; comparison and uncertainty are explicit.','claim_ids':claims}]} for h in data['required_headings']],'limitations':['Synthetic fixture, no scientific validation.']}
        return ModelReply(content=json.dumps(d),usage={'input_tokens':20,'output_tokens':10})


def seeded(tmp_path,revise=False,mode='design',dataset=None,workflow_version='unified-v1',allow_network=False):
    p=DesignPort(revise);s=AgentService(tmp_path/'state',model=p,dataset_dir=dataset);f=AgentFramework(s,allow_frozen=True)
    spec=TaskSpec(mode,'Synthetic conceptual design comparison',workflow_version=workflow_version,allow_network=allow_network,allowed_roots=(str(dataset),) if dataset else ())
    t=f.submit(spec)
    for i in range(6):
        raw=json.dumps({'abstract':f'Synthetic finding {i}. Second bounded observation. Third bounded observation.'}).encode()
        prov={'synthetic_fixture':True,'doi':f'10.1234/synthetic{i}','text_availability':'abstract'}
        a=s.store.put_artifact(raw,'synthetic.json','application/json',prov,session_id=t['session_id'],run_id=t['initial_run_id'])
        s.research.register_source(t['session_id'],t['initial_run_id'],artifact_id=a['id'],kind='literature',title=f'Synthetic distinct work {i}',provenance=prov,idempotency_key='lit'+str(i))
    return s,f,t,p


@pytest.mark.parametrize('revise',[False,True])
def test_design_delivers_versioned_report_and_resolves_actual_feedback(tmp_path,revise):
    s,f,t,p=seeded(tmp_path,revise)
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='succeeded',result
    assert result['delivery_status']=='report'
    assert result['empirical_validation_status']=='not_performed'
    records=result['lineage']['records'];designs=[r for r in records if r['kind']=='design_revision']
    assert len(designs)==(2 if revise else 1)
    if revise:assert result['lineage']['feedback'][0]['resolved']
    old_calls=len(p.requests);assert f.run(t['fingerprint'])==result;assert len(p.requests)==old_calls
    s.close()


@pytest.mark.parametrize('phase',['decision_review','report_review'])
def test_resume_after_record_commit_keeps_model_receipts(tmp_path,phase):
    s,f,t,p=seeded(tmp_path)
    save=f.repository.save;crashed=False
    def crash(fp,state,**kwargs):
        nonlocal crashed
        result=save(fp,state,**kwargs)
        if state['phase']==phase and not crashed:
            crashed=True
            raise KeyboardInterrupt('simulated process interruption after durable record')
        return result
    f.repository.save=crash
    with pytest.raises(KeyboardInterrupt):f.run(t['fingerprint'])
    before=p.decisions
    f.repository.save=save
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='succeeded',result.get('error')
    assert p.decisions==before
    assert len([r for r in result['lineage']['records'] if r['kind']=='design_revision'])==1
    s.close()


def test_evidence_feedback_returns_to_retrieval_automatically(tmp_path):
    s,f,t,p=seeded(tmp_path)
    complete=p.complete;evidence_reviews=0
    def review(messages,tools,**kwargs):
        nonlocal evidence_reviews
        if not tools:
            data=json.loads(messages[-1]['content'])
            if data.get('stage')=='evidence':
                evidence_reviews+=1
                if evidence_reviews==1:
                    p.requests.append({'messages':messages,'tools':tools})
                    return ModelReply(content=json.dumps({'verdict':'revise','return_to':'evidence','issues':['Retrieve candidate counterevidence'],'rationale':'Missing comparison'}),usage={'input_tokens':20,'output_tokens':10})
        return complete(messages,tools,**kwargs)
    p.complete=review
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='succeeded',result.get('error')
    assert evidence_reviews==2
    assert sum(bool(req['tools']) for req in p.requests)==2
    assert len(result['attempt_refs'])==3
    s.close()


def test_real_gene_data_enters_finding_and_design_lineage(tmp_path):
    dataset=Path(__file__).resolve().parents[1]/'data/gene_diff'
    if not (dataset/'cache').is_dir():pytest.skip('Reviewed local dataset unavailable')
    s,f,t,p=seeded(tmp_path,mode='discovery_design',dataset=dataset)
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='succeeded',result.get('error')
    findings=[r for r in result['lineage']['records'] if r['kind']=='finding']
    assert len(findings)==3
    assert result['empirical_validation_status']=='not_performed'
    assert any(edge['relation']=='finding_ids' for edge in result['lineage']['edges'])
    s.close()


def test_evidence_refresh_prioritizes_new_readable_sources(tmp_path):
    s,f,t,p=seeded(tmp_path)
    old=t['initial_run_id'];sid=t['session_id']
    for i in range(30):
        source(s.store,s.research,sid,old,content=f'Old synthetic excerpt {i}.',key=f'old_{i}')
    fresh=s.store.create_run(sid,t['task_id'],dict(TaskSpec('design','Q').budget))['id']
    recent,_=source(s.store,s.research,sid,fresh,content='Newly retrieved counterevidence.',key='new')
    selected,_=f._materialize(sid,fresh,latest_run_id=fresh)
    actual=[s.research.get_evidence(sid,fresh,e['id'])['source_id'] for e in selected.values()]
    assert recent['id'] in actual
    s.close()
