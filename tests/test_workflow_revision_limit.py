import json
from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework
from test_framework_design import DesignPort


def test_revision_limit_is_durable_and_resume_spends_no_more_requests(tmp_path):
    p=DesignPort();s=AgentService(tmp_path/'state',model=p);f=AgentFramework(s,allow_frozen=True)
    t=f.submit(TaskSpec('design','Synthetic question',workflow_version='unified-v1',allow_network=False,parameters={'max_revision_rounds':1}))
    for i in range(6):
        prov={'doi':f'10.1234/source{i}','text_availability':'abstract'}
        a=s.store.put_artifact(json.dumps({'abstract':f'Finding {i}. Other.'}),'a.json','application/json',prov,session_id=t['session_id'],run_id=t['initial_run_id'])
        s.research.register_source(t['session_id'],t['initial_run_id'],artifact_id=a['id'],kind='literature',title=f'Source {i}',provenance=prov,idempotency_key=str(i))
    original=p.complete
    def complete(messages,tools,**kwargs):
        if not tools and json.loads(messages[-1]['content']).get('stage')=='evidence':
            p.requests.append({'messages':messages,'tools':tools})
            return ModelReply(content=json.dumps({'verdict':'revise','return_to':'evidence','issues':['Need more sources'],'rationale':'Fixture'}),usage={'input_tokens':1,'output_tokens':1})
        return original(messages,tools,**kwargs)
    p.complete=complete
    result=f.run(t['fingerprint']);requests=len(p.requests)
    assert result['execution_status']=='waiting'
    assert result['error']['code']=='revision_limit'
    assert result['delivery_status']=='report' and result['content_status']=='incomplete'
    assert result['delivery']['kind']=='incomplete_report'
    from adj_partner.orchestration.exports import export_task
    export_task(f,t['fingerprint'],tmp_path/'export')
    assert 'Need more sources' in (tmp_path/'export/report.md').read_text()
    assert not (tmp_path/'export/answer.md').exists()
    assert f.repository.get(t['fingerprint'])['state']['phase']=='revision_limit'
    for _ in range(3):assert f.run(t['fingerprint'])==result
    assert len(p.requests)==requests==2
    assert len(s.store.list_runs(t['session_id']))==2
    s.close()
