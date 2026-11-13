import json
from adj_partner.application import AgentService
from adj_partner.models.base import ScriptedPort
from adj_partner.orchestration.controller import AgentFramework
from adj_partner.orchestration.contracts import TaskSpec


def test_computed_selections_expose_primary_and_sensitivity_parameters(tmp_path):
    service=AgentService(tmp_path/'state',model=ScriptedPort([]));f=AgentFramework(service)
    task=f.submit(TaskSpec('design','Synthetic design',model='offline-script',allow_network=False))
    sid,rid=task['session_id'],task['initial_run_id'];sources=[]
    for top_n in (250,10):
        provenance={'text_availability':'computed','parameters':{'top_n':top_n,'n_permutations':500}}
        artifact=service.store.put_artifact(json.dumps({'summary':{'rank':'A' if top_n==250 else 'B'}}),'screen.json','application/json',provenance,session_id=sid,run_id=rid)
        source=service.research.register_source(sid,rid,artifact_id=artifact['id'],kind='computed',title='Screen',provenance=provenance,idempotency_key=str(top_n));sources.append(source)
    selected,_=f._materialize(sid,rid,baseline_source_id=sources[0]['id'])
    values=list(selected.values())
    assert values[0]['analysis_role']=='primary_screen'
    assert values[0]['analysis_parameters']['top_n']==250
    assert values[0]['source_sha256']==sources[0]['artifact_hash']
    assert values[1]['analysis_role']=='sensitivity_screen'
    assert values[1]['analysis_parameters']['top_n']==10
    service.close()


def test_materialization_bounds_total_evidence_bytes_before_critic(tmp_path):
    service=AgentService(tmp_path/'state',model=ScriptedPort([]));f=AgentFramework(service)
    task=f.submit(TaskSpec('design','Q',model='offline-script',allow_network=False));sid,rid=task['session_id'],task['initial_run_id']
    for i in range(8):
        p={'text_availability':'computed','parameters':{'compound':str(i)}}
        a=service.store.put_artifact(json.dumps({'summary':{'text':str(i)+'x'*6000}}),'a.json','application/json',p,session_id=sid,run_id=rid)
        service.research.register_source(sid,rid,artifact_id=a['id'],kind='computed',title='Bounded summary',provenance=p,idempotency_key=str(i))
    selected,_=f._materialize(sid,rid,evidence_budget_bytes=20000)
    assert 0<len(selected)<8
    assert sum(len(json.dumps(v,ensure_ascii=False).encode()) for v in selected.values())<20000
    service.close()


def test_computed_followup_is_not_starved_by_literature_prefix(tmp_path):
    service=AgentService(tmp_path/'state',model=ScriptedPort([]));f=AgentFramework(service)
    task=f.submit(TaskSpec('design','Q',model='offline-script',allow_network=False));sid,rid=task['session_id'],task['initial_run_id']
    for i in range(35):
        p={'text_availability':'abstract'}
        a=service.store.put_artifact(json.dumps({'abstract':f'Independent public evidence {i}.'}),'paper.json','application/json',p,session_id=sid,run_id=rid)
        service.research.register_source(sid,rid,artifact_id=a['id'],kind='literature',title=f'Paper {i}',provenance=p,idempotency_key=f'paper-{i}')
    p={'text_availability':'computed','parameters':{'compound':'candidate','gene_set_library':'GO'}}
    a=service.store.put_artifact(json.dumps({'summary':{'immune_response_hits':101}}),'go.json','application/json',p,session_id=sid,run_id=rid)
    source=service.research.register_source(sid,rid,artifact_id=a['id'],kind='computed',title='Follow-up GO',provenance=p,idempotency_key='go')
    selected,_=f._materialize(sid,rid,latest_run_id=rid)
    assert any(v.get('source_id')==source['id'] for v in selected.values())
    assert any(v['scope']=='abstract' for v in selected.values())
    assert len(selected)<=30
    service.close()


def test_repeated_computation_does_not_consume_multiple_evidence_slots(tmp_path):
    service=AgentService(tmp_path/'state',model=ScriptedPort([]));f=AgentFramework(service)
    task=f.submit(TaskSpec('design','Q',model='offline-script',allow_network=False));sid,rid=task['session_id'],task['initial_run_id']
    p={'text_availability':'computed','parameters':{'top_n':250}}
    for i in range(3):
        a=service.store.put_artifact(json.dumps({'summary':{'rank':'A'},'execution_id':i}),'repeat.json','application/json',p,session_id=sid,run_id=rid)
        service.research.register_source(sid,rid,artifact_id=a['id'],kind='computed',title='Repeated screen',provenance=p,idempotency_key=str(i))
    selected,_=f._materialize(sid,rid)
    assert len(selected)==1
    service.close()
