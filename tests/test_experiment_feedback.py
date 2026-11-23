"""Synthetic assay reports test engineering boundaries, not scientific truth."""
from copy import deepcopy
import json
import pytest
from adj_partner.contracts import Budget
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from test_harness_research_records import setup,source
from test_research_design import design
from test_experiment_checkpoints import checkpoint


def prepared(tmp_path):
    store,r,sid,rid,tid=setup(tmp_path)
    src,_=source(store,r,sid,rid)
    evidence=r.select_evidence(sid,rid,src['id'],{'passage':'First finding.'})
    claim=r.observe_evidence(sid,rid,evidence['id'])
    d=design(claim['id']);cards=[checkpoint(1),checkpoint(2),checkpoint(3,'function')]
    for card,level in zip(cards,['molecular_activity','causal_dependence','antigen_function']):
        card['claim_ids']=[claim['id']];card['decision_level']=level
    d['verification']['checkpoints']=cards
    brief=r.create_design_brief(sid,rid,goal='Synthetic feedback',requirements=['Function and mechanism'],evidence_ids=[evidence['id']])
    revision=r.create_design_revision(sid,rid,brief_id=brief['id'],design=d)
    experimental,_=source(store,r,sid,rid,kind='userfile',key='experimental',content=json.dumps({'value':'synthetic measurement'}).encode(),provenance={'experiment_data':True,'text_availability':'userfile','synthetic_fixture':True})
    ee=r.select_evidence(sid,rid,experimental['id'],{'pointer':'/value'})
    return store,r,sid,rid,tid,revision,ee,evidence


def arguments(d,e,**overrides):
    return dict(result_key='lab_001',design_id=d['id'],checkpoint_id='C1',evidence_ids=[e['id']],
        measurement_level='molecular_activity',quality='pass',context_status='matched',outcome='opposes',
        interpretation='Synthetic valid absence weakens this mechanism only.',
        context='Matched synthetic cells, antigen, vehicle and time; not an actual experiment.',
        reported_by='synthetic fixture',**overrides)


def test_quality_level_and_branch_do_not_upgrade_to_mechanism_or_efficacy(tmp_path):
    store,r,sid,rid,tid,d,e,_=prepared(tmp_path)
    a=arguments(d,e)
    result=r.record_experiment_result(sid,rid,**a)
    assert result['payload']['effective_outcome']=='opposes'
    state=r.experiment_state(sid,rid,design_id=d['id'])
    assert state['checkpoints'][0]['status']=='opposes'
    assert state['decision']=='functional_benefit_unresolved'
    assert state['scientific_validation']=='reported_interpretations_not_independently_validated'
    a.update(result_key='lab_002',checkpoint_id='C3',measurement_level='antigen_function',outcome='supports')
    r.record_experiment_result(sid,rid,**a)
    assert r.experiment_state(sid,rid,design_id=d['id'])['decision']=='retain_function_reconsider_mechanism'
    a.update(result_key='lab_003',outcome='opposes',quality='fail')
    failed=r.record_experiment_result(sid,rid,**a)
    assert failed['payload']['effective_outcome']=='inconclusive'
    assert r.experiment_state(sid,rid,design_id=d['id'])['checkpoints'][2]['status']=='supports'


@pytest.mark.parametrize('field,value', [('quality','unknown'),('context_status','mismatched'),('measurement_level','transcript')])
def test_uninterpretable_or_wrong_level_remains_inconclusive(tmp_path,field,value):
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    a=arguments(d,e);a[field]=value
    record=r.record_experiment_result(sid,rid,**a)
    assert record['payload']['effective_outcome']=='inconclusive'
    assert record['payload']['outcome']=='opposes'


def test_conflicting_repeats_and_task_wide_idempotency(tmp_path):
    store,r,sid,rid,tid,d,e,_=prepared(tmp_path)
    a=arguments(d,e);first=r.record_experiment_result(sid,rid,**a)
    later=store.create_run(sid,tid,Budget().to_dict())['id']
    reopened=ResearchStore(Store(store.root))
    assert reopened.record_experiment_result(sid,later,**a)==first
    a['outcome']='supports'
    with pytest.raises(ValueError,match='[Cc]onflict'):reopened.record_experiment_result(sid,later,**a)
    a['result_key']='independent_repeat'
    reopened.record_experiment_result(sid,later,**a)
    assert reopened.experiment_state(sid,later,design_id=d['id'])['checkpoints'][0]['status']=='conflicting'


def test_cannot_use_literature_as_new_experiment_or_cross_task(tmp_path):
    store,r,sid,rid,_,d,e,literature=prepared(tmp_path)
    a=arguments(d,literature)
    with pytest.raises(ValueError,match='experimental'):r.record_experiment_result(sid,rid,**a)
    tid=store.create_task(sid,'Other task')['id'];other=store.create_run(sid,tid,Budget().to_dict())['id']
    with pytest.raises((KeyError,ValueError,PermissionError)):r.record_experiment_result(sid,other,**arguments(d,e))


def test_updated_plan_binds_actual_results_and_preserves_old_version(tmp_path):
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    result=r.record_experiment_result(sid,rid,**arguments(d,e))
    new=deepcopy(d['payload']['design'])
    with pytest.raises(ValueError,match='changed|evidence'):
        r.revise_design_from_experiments(sid,rid,design_id=d['id'],result_ids=[result['id']],design=new)
    observation=r.observe_evidence(sid,rid,e['id'])
    new['verification']['claim_ids'].append(observation['id'])
    new['verification']['checkpoints'][0]['claim_ids'].append(observation['id'])
    new['verification']['checkpoints'][0]['current_evidence']='New synthetic activity result opposes this mechanism; function remains unresolved.'
    revised=r.revise_design_from_experiments(sid,rid,design_id=d['id'],result_ids=[result['id']],design=new)
    assert revised['payload']['parent_id']==d['id']
    assert revised['payload']['result_ids']==[result['id']]
    assert r.get_design_record(sid,rid,d['id'])==d
    assert r.experiment_state(sid,rid,design_id=revised['id'])['checkpoints'][0]['status']=='opposes'
    assert any(edge['relation']=='result_ids' for edge in r.design_lineage(sid,rid)['edges'])


def test_changed_checkpoint_contract_does_not_inherit_old_result(tmp_path):
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    result=r.record_experiment_result(sid,rid,**arguments(d,e))
    changed=deepcopy(d['payload']['design']);changed['verification']['checkpoints'][0]['hypothesis']='Different cell context'
    new=r.create_design_revision(sid,rid,brief_id=d['payload']['brief_id'],parent_id=d['id'],design=changed)
    assert r.experiment_state(sid,rid,design_id=new['id'])['checkpoints'][0]['status']=='unmeasured'


def test_tool_state_is_compact_and_result_forgery_not_exposed(tmp_path):
    from adj_partner.research.tools import register_research_tools
    from adj_partner.tools import ToolCatalog,ToolExecutor
    from adj_partner.contracts import ToolCall,SessionConfig
    store,r,sid,rid,_,d,e,_=prepared(tmp_path)
    catalog=ToolCatalog();register_research_tools(catalog,r)
    assert 'experiment_state' in catalog.names()
    assert 'experiment_result_record' not in catalog.names()
    r.record_experiment_result(sid,rid,**arguments(d,e))
    with store.transaction(write=True) as c:
        row=c.execute('SELECT config FROM sessions WHERE id=?',(sid,)).fetchone();config=json.loads(row[0])
        config['allowed_tools']=['experiment_state'];config['enabled_tools']=['experiment_state']
        c.execute('UPDATE sessions SET config=? WHERE id=?',(json.dumps(config),sid))
    result=ToolExecutor(store,catalog).execute(ToolCall('state','experiment_state',{'design_id':d['id']}),sid,rid)
    assert result.status=='ok',result.to_dict()
    assert result.artifacts
    from adj_partner.runtime.context import ContextView
    preview=json.loads(ContextView(store).tool_message(sid,rid,'state',result.to_dict())['content'])
    assert preview['data']['summary']['decision']=='functional_benefit_unresolved'


def test_importer_binds_local_bytes_and_replays_without_duplicates(tmp_path):
    from adj_partner.experiment_feedback import import_result
    store,r,sid,rid,_,d,e,_=prepared(tmp_path)
    file=tmp_path/'measured.json';file.write_text(json.dumps({'measurement':{'BHMT_activity':None,'quality':'failed positive control'}}))
    bundle=arguments(d,e);bundle.pop('evidence_ids')
    bundle.update(result_file=str(file),selector={'pointer':'/measurement'},quality='fail')
    first=import_result(r,sid,rid,bundle)
    assert first['payload']['effective_outcome']=='inconclusive'
    second=import_result(r,sid,rid,bundle)
    assert first==second
    file.write_text(json.dumps({'measurement':{'BHMT_activity':10}}))
    with pytest.raises(ValueError,match='[Cc]onflict'):import_result(r,sid,rid,bundle)


def test_stale_result_cannot_update_or_silently_remain_supporting(tmp_path):
    store,r,sid,rid,_,d,e,_=prepared(tmp_path)
    result=r.record_experiment_result(sid,rid,**arguments(d,e))
    src=r.get_source(sid,rid,e['source_id'])
    (store.root/'artifacts'/src['artifact_hash']).write_bytes(b'corrupt')
    state=r.experiment_state(sid,rid,design_id=d['id'])
    assert state['checkpoints'][0]['status']=='inconclusive'
    assert state['checkpoints'][0]['stale_results']


def test_importer_rejects_empty_selector_even_on_replay(tmp_path):
    from adj_partner.experiment_feedback import import_result
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    file=tmp_path/'result.json';file.write_text('{"measurement": 1}')
    bundle=arguments(d,e);bundle.pop('evidence_ids');bundle.update(result_file=str(file),selector={'pointer':'/measurement'})
    import_result(r,sid,rid,bundle)
    bundle['selector']={}
    with pytest.raises(ValueError):import_result(r,sid,rid,bundle)


def test_explicit_correction_supersedes_report_without_faking_independent_repeat(tmp_path):
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    a=arguments(d,e);a['outcome']='supports'
    original=r.record_experiment_result(sid,rid,**a)
    a.update(result_key='lab_001_correction',supersedes=original['id'],quality='fail',outcome='inconclusive')
    correction=r.record_experiment_result(sid,rid,**a)
    state=r.experiment_state(sid,rid,design_id=d['id'])
    assert state['checkpoints'][0]['status']=='inconclusive'
    assert state['checkpoints'][0]['result_ids']==[correction['id']]
    assert state['checkpoints'][0]['superseded_result_ids']==[original['id']]
    a.update(result_key='bad_correction',checkpoint_id='C3',measurement_level='antigen_function')
    with pytest.raises(ValueError,match='same checkpoint'):r.record_experiment_result(sid,rid,**a)


def test_next_step_comes_from_bound_checkpoint_branch_not_new_inference(tmp_path):
    _,r,sid,rid,_,d,e,_=prepared(tmp_path)
    r.record_experiment_result(sid,rid,**arguments(d,e))
    state=r.experiment_state(sid,rid,design_id=d['id'])
    card=d['payload']['design']['verification']['checkpoints'][0]
    assert state['checkpoints'][0]['next_step']==card['opposes']
    assert state['checkpoints'][1]['next_step']==d['payload']['design']['verification']['checkpoints'][1]['inconclusive']


def test_new_reports_do_not_label_all_assays_unperformed_and_old_rendering_is_stable(tmp_path):
    from adj_partner.research.rendering import render_report
    _,r,sid,rid,_,d,_,_=prepared(tmp_path)
    data=d['payload']['design'];refs=data['verification']['claim_ids']
    report=r.create_report(sid,rid,title='Synthetic updated design',
        sections=[{'heading':'Decision','claim_ids':refs,'paragraphs':[{'text':'Read recorded outcomes separately.','claim_ids':refs}]}],
        limitations=['Synthetic fixture only'],research_design=data)
    assert report['renderer_version']==5
    assert '实际执行状态见实验结果记录' in render_report(report).decode()
    legacy=deepcopy(report);legacy['renderer_version']=4
    assert '关键实验判据（计划，未执行）' in render_report(legacy).decode()
