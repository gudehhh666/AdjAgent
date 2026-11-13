"""Offline acceptance for the opt-in design resource handoff, no provider calls."""
import json
from types import SimpleNamespace

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework
from adj_partner.research import ResearchStore
from adj_partner.resources.pack import ResourcePack
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from test_adjuvant_resource_pack import make_pack


@pytest.fixture
def rig(tmp_path):
    pack=ResourcePack(make_pack(tmp_path/'pack'))
    pack.records['VO_1']['fields'].update(small=0.25, missing=None,
        long='prefix '*4000+' TARGET literal observation. '+' suffix'*4000)
    store=Store(tmp_path/'state');research=ResearchStore(store);catalog=ToolCatalog()
    NativeCapabilities(store,research=research,resource_pack=pack).register(catalog)
    names=['adjuvant_read_fields','adjuvant_cite','adjuvant_lookup']
    sid=store.create_session(SessionConfig(allowed_tools=names,enabled_tools=names).to_dict())['id']
    tid=store.create_task(sid,'Compare two proposals using exact available observations',{})['id']
    rid=store.create_run(sid,tid,Budget().to_dict())['id']
    framework=AgentFramework(SimpleNamespace(store=store,research=research))
    yield framework,research,ToolExecutor(store,catalog),sid,rid,store
    store.close()


def read(rig,tool,args,call='read'):
    f,r,executor,sid,rid,store=rig
    result=executor.execute(ToolCall(call,tool,{'record_id':'VO_1',**args}),sid,rid)
    assert result.status=='ok',result.to_dict()
    return result.data['summary']


def test_short_read_values_survive_oversize_record_with_original_selector(rig):
    packet=read(rig,'adjuvant_read_fields',{'fields':['small','missing']})
    f,r,_,sid,rid,_=rig
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert {v['text'] for v in selected.values()}=={0.25,None}
    for v in selected.values():
        e=r.get_evidence(sid,rid,v['id'])
        assert e['source_id']==packet['source_id']
        assert e['locator']['pointer'] in {'/record/fields/small','/record/fields/missing'}


def test_focused_citation_carries_only_read_range_and_parent_lineage(rig):
    packet=read(rig,'adjuvant_cite',{'field':'long','focus':'TARGET literal','length':300})
    f,r,_,sid,rid,_=rig
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert len(selected)==1
    v=next(iter(selected.values()));assert v['text']==packet['citation']['quote']
    source=r.get_source(sid,rid,v['source_id'])
    assert source['provenance']['parent_source_id']==packet['source_id']
    assert source['provenance']['read_selector']==packet['selector']
    assert source['provenance']['read_range']=={'start':packet['start'],'end':packet['end']}


def test_receipts_do_not_cross_tasks_or_accept_failed_actions(rig):
    packet=read(rig,'adjuvant_read_fields',{'fields':['small']})
    f,r,_,sid,rid,store=rig
    tid=store.create_task(sid,'Other task',{})['id']
    other=store.create_run(sid,tid,Budget().to_dict())['id']
    with pytest.raises(ValueError,match='task'):
        f._materialize(sid,other,receipt_run_ids=[rid])
    store.claim_action(rid,'failed','adjuvant_cite',{},0)
    store.finish_action(rid,'failed',{'status':'error','data':{'summary':packet}})
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert len(selected)==1


def test_legacy_materialize_does_not_implicitly_adopt_new_handoff(rig):
    read(rig,'adjuvant_read_fields',{'fields':['small']})
    f,_,_,sid,rid,_=rig
    assert f._materialize(sid,rid)[0]=={}


def test_optional_resource_identity_preserves_legacy_serialization():
    legacy=TaskSpec('design','Q')
    assert 'resource_manifest_sha256' not in legacy.to_dict()
    bound=TaskSpec.from_dict({**legacy.to_dict(),'workflow_version':'resource-guided-design-v1',
                             'resource_manifest_sha256':'a'*64})
    assert bound.to_dict()['resource_manifest_sha256']=='a'*64
    assert TaskSpec.from_dict(bound.to_dict()).fingerprint==bound.fingerprint
    with pytest.raises(ValueError,match='resource'):
        TaskSpec.from_dict({**legacy.to_dict(),'resource_manifest_sha256':'bad'})


def test_resource_workflow_is_opt_in_experimental_and_records_identity(tmp_path):
    from test_framework_design import seeded
    s,f,t,p=seeded(tmp_path,workflow_version='resource-guided-design-v1')
    try:
        config=s.store.get_session(t['session_id'])['config']
        assert config['profile']=='research-resource-aware'
        assert 'research_critic' not in config['allowed_tools']
        assert len(t['spec']['resource_manifest_sha256'])==64
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='succeeded',result.get('error')
        assert result['review_status']=='not_run'
        assert result['delivery']['experiment_status']=='planned'
        for request in p.requests:
            if not request['tools']:
                assert 'This synthesis stage has no tools' in request['messages'][0]['content']
                assert 'stage' not in json.loads(request['messages'][-1]['content'])
        assert f.run(t['fingerprint'])==result
    finally:s.close()


def test_resource_runtime_drift_refuses_even_cached_delivery(tmp_path):
    from test_framework_design import seeded
    s,f,t,p=seeded(tmp_path,workflow_version='resource-guided-design-v1')
    try:
        result=f.run(t['fingerprint']);assert result['execution_status']=='succeeded',result
        if s.native.resource_pack is None:pytest.skip('needs the local resource pack')
        s.native.resource_pack.manifest_hash='f'*64
        with pytest.raises(ValueError,match='Resource runtime changed'):
            f.run(t['fingerprint'])
    finally:s.close()


def test_lookup_visible_fragment_survives_same_task_stage_handoff(rig):
    f,r,executor,sid,rid,store=rig
    result=executor.execute(ToolCall('lookup','adjuvant_lookup',{'dataset':'vo','query':'VO_1','fields':['small']}),sid,rid)
    assert result.status=='ok',result.to_dict()
    stage=store.create_run(sid,store.get_run(rid)['task_id'],Budget().to_dict())['id']
    selected,_=f._materialize(sid,stage,receipt_run_ids=[rid])
    assert [v['text'] for v in selected.values()]==[0.25]
    assert next(iter(selected.values()))['source_id']==result.data['summary']['fragments'][0]['source_id']


def test_forged_value_is_excluded_and_omitted_budget_is_auditable(rig):
    packet=read(rig,'adjuvant_read_fields',{'fields':['small']})
    f,_,_,sid,rid,store=rig
    packet['values']['small']=999
    store.claim_action(rid,'forged','adjuvant_read_fields',{},0)
    store.finish_action(rid,'forged',{'status':'ok','data':{'summary':packet}})
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid],evidence_budget_bytes=1)
    assert selected=={}
    events=store.events(sid,rid)
    receipts=[v for event in events if event['kind']=='workflow.resource_handoff' for v in event['payload']['receipts']]
    assert any(v.get('reason')=='Receipt value mismatch' for v in receipts)
    assert any(e['kind']=='workflow.resource_handoff.omitted' and e['payload']['reason']=='packet_budget' for e in events)


def test_resource_manifest_includes_actual_registered_file_drift(tmp_path):
    from test_framework_design import seeded
    from adj_partner.orchestration.resource_handoff import resource_manifest
    s,_,_,_=seeded(tmp_path)
    try:
        host=tmp_path/'host';(host/'config').mkdir(parents=True)
        data=host/'fact.txt';data.write_text('old')
        (host/'config/research-resource-index.json').write_text(json.dumps({'resources':[{'path':'fact.txt','sha256':'a'*64}]}))
        s.project_root=host
        before=resource_manifest(s)[1]
        data.write_text('new')
        assert resource_manifest(s)[1]!=before
    finally:s.close()


def test_opt_in_discovery_keeps_prespecified_host_analysis_callable(tmp_path):
    from test_framework_design import seeded
    dataset=tmp_path/'reviewed';(dataset/'cache').mkdir(parents=True)
    (dataset/'cache/identity.parquet').write_bytes(b'identity-only fixture, no computation')
    s,_,t,_=seeded(tmp_path,mode='discovery_design',dataset=dataset,workflow_version='resource-guided-design-v1')
    try:
        config=s.store.get_session(t['session_id'])['config']
        assert {'deg_analyze','deg_explore'}<=set(config['enabled_tools'])
        assert config['allowed_roots']==[str(dataset)]
    finally:s.close()


def test_partial_read_never_promotes_unread_small_record_contents(rig):
    # Even a small full record must not be substituted for its actually read excerpt.
    packet=read(rig,'adjuvant_cite',{'field':'definition','length':7})
    f,r,_,sid,rid,_=rig
    from adj_partner.orchestration.resource_handoff import read_selections
    selectors,_=read_selections(f.service,sid,rid,[rid])
    assert selectors.get(packet['source_id'])==[]
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert [v['text'] for v in selected.values()]==[packet['citation']['quote']]


def test_fully_omitted_read_cannot_fall_back_to_whole_record(rig):
    packet=read(rig,'adjuvant_read_fields',{'fields':['long']})
    assert packet['values']=={} and packet['omitted_fields']==['long']
    f,_,_,sid,rid,_=rig
    from adj_partner.orchestration.resource_handoff import read_selections
    selectors,_=read_selections(f.service,sid,rid,[rid])
    assert selectors.get(packet['source_id'])==[]


def window_receipt(rig, *, content=None, query='decisive observation', source_backed=True):
    from adj_partner.evidence_access import read_evidence
    f,r,_,sid,rid,store=rig
    content=content or {'sections':[{'text':'unread prefix '*2000+'decisive observation here'+' unread suffix'*2000}]}
    provenance={'text_availability':'derived_full_text','url':'https://example.org/source'}
    artifact=store.put_artifact(json.dumps(content),'window-source.json','application/json',provenance,session_id=sid,run_id=rid)
    src=r.register_source(sid,rid,artifact_id=artifact['id'],kind='literature',title='Window fixture',provenance=provenance,idempotency_key='window-fixture') if source_backed else None
    args={'source_id':src['id']} if src else {'artifact_id':artifact['id']}
    args['query']=query
    context=SimpleNamespace(store=store,session_id=sid,run_id=rid,check_cancelled=lambda:None)
    result=read_evidence(context,**args)
    return src,artifact,args,result


def test_evidence_window_handoff_excludes_unread_siblings_and_preserves_lineage(rig):
    src,artifact,args,result=window_receipt(rig)
    f,r,_,sid,rid,store=rig
    store.claim_action(rid,'window-read','evidence_read',args,0)
    store.finish_action(rid,'window-read',result.to_dict())
    stage=store.create_run(sid,store.get_run(rid)['task_id'],Budget().to_dict())['id']
    selected,_=f._materialize(sid,stage,receipt_run_ids=[rid])
    assert [v['text'] for v in selected.values()]==[result.data['text']]
    derived=r.get_source(sid,stage,next(iter(selected.values()))['source_id'])
    p=derived['provenance']
    assert p['parent_source_id']==src['id'] and p['parent_source_sha256']==src['artifact_hash']
    assert p['parent_artifact_id']==artifact['id'] and p['read_projection_sha256']==result.data['projection_hash']
    assert p['read_selector']==result.data['selector']
    assert p['read_range']=={k:result.data[k] for k in ('start','end','total')}
    assert p['text_availability']=='derived_full_text'


@pytest.mark.parametrize('field,bad',[('artifact_hash','f'*64),('projection_hash','e'*64),('start',0),('selector',{'pointer':'/sections'}),('text','invented finding')])
def test_forged_evidence_window_is_excluded_without_fallback(rig,field,bad):
    from adj_partner.orchestration.resource_handoff import read_selections
    src,_,args,result=window_receipt(rig)
    f,_,_,sid,rid,store=rig
    result.data[field]=bad
    store.claim_action(rid,'forged-window','evidence_read',args,0)
    store.finish_action(rid,'forged-window',result.to_dict())
    selected,audit=read_selections(f.service,sid,rid,[rid])
    assert selected=={src['id']:[]}
    assert audit[-1]['status']=='excluded' and 'mismatch' in audit[-1]['reason'].lower()


def test_evidence_miss_cannot_promote_complete_source(rig):
    src,_,args,result=window_receipt(rig,query='not in source')
    f,_,_,sid,rid,store=rig
    store.claim_action(rid,'miss','evidence_read',args,0)
    store.finish_action(rid,'miss',result.to_dict())
    assert f._materialize(sid,rid,receipt_run_ids=[rid])[0]=={}


def test_unregistered_artifact_window_remains_user_material(rig):
    _,artifact,args,result=window_receipt(rig,source_backed=False)
    f,r,_,sid,rid,store=rig
    store.claim_action(rid,'artifact-window','evidence_read',args,0)
    store.finish_action(rid,'artifact-window',result.to_dict())
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert [v['text'] for v in selected.values()]==[result.data['text']]
    derived=r.get_source(sid,rid,next(iter(selected.values()))['source_id'])
    assert derived['kind']=='userfile' and derived['provenance']['parent_source_id'] is None
    assert derived['provenance']['parent_artifact_id']==artifact['id']


def test_redacted_window_handoff_carries_only_projection_not_private_bytes(rig):
    f,r,_,sid,rid,store=rig
    store.register_secrets(['synthetic-secret-123'])
    src,_,args,result=window_receipt(rig,content={'body':'before synthetic-secret-123 decisive observation after'})
    store.claim_action(rid,'redacted-window','evidence_read',args,0)
    store.finish_action(rid,'redacted-window',result.to_dict())
    selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
    assert [v['text'] for v in selected.values()]==[result.data['text']]
    assert 'synthetic-secret-123' not in json.dumps(selected)
    assert '[REDACTED]' in next(iter(selected.values()))['text']
    derived=r.get_source(sid,rid,next(iter(selected.values()))['source_id'])
    assert derived['provenance']['parent_source_sha256']==src['artifact_hash']
    assert derived['provenance']['read_projection_sha256']!=src['artifact_hash']


def test_new_redaction_invalidates_saved_window_instead_of_reusing_stale_text(rig):
    from adj_partner.orchestration.resource_handoff import read_selections
    src,_,args,result=window_receipt(rig,content={'body':'decisive observation then newly-private-string'})
    f,_,_,sid,rid,store=rig
    store.claim_action(rid,'stale-window','evidence_read',args,0)
    store.finish_action(rid,'stale-window',result.to_dict())
    store.register_secrets(['newly-private-string'])
    selected,audit=read_selections(f.service,sid,rid,[rid])
    assert selected=={src['id']:[]}
    assert audit[-1]['status']=='excluded' and 'mismatch' in audit[-1]['reason']


def test_failed_window_never_falls_back_to_existing_source(rig):
    src,_,args,_=window_receipt(rig)
    f,_,_,sid,rid,store=rig
    store.claim_action(rid,'failed-window','evidence_read',args,0)
    store.finish_action(rid,'failed-window',{'status':'error','data':None,'error':{'code':'test_failure'}})
    assert f._materialize(sid,rid,receipt_run_ids=[rid])[0]=={}


def test_resource_manifest_binds_reader_algorithms_and_optional_profile(tmp_path):
    from test_framework_design import seeded
    from adj_partner.orchestration.resource_handoff import resource_manifest
    from adj_partner.profiles import get_profile
    s,_,_,_=seeded(tmp_path)
    try:
        old,_=resource_manifest(s)
        assert old['workflow']=='resource-guided-design-v1'
        assert old['profile']==get_profile('research-resource-aware')
        value,_=resource_manifest(s,workflow='evidence-grounded-design-v2',profile='research-evidence-first')
        assert value['workflow']=='evidence-grounded-design-v2'
        assert value['profile']==get_profile('research-evidence-first')
        assert {'adj_partner/evidence_access.py','adj_partner/matrix_access.py','adj_partner/quality_entry.py'}<=set(value['files'])
    finally:s.close()
