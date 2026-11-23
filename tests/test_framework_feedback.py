import pytest
from adj_partner.orchestration.feedback import update_feedback, repair_tasks, claim_coverage, source_priority


def review(stage='evidence', issues=None):
    return {'verdict':'revise','return_to':stage,'issues':issues or ['Need GO evidence for Zincgluconate'],'rationale':'fixture'}


def test_issue_identity_progress_and_stage_scoped_closure():
    selected={'E1':{'id':'evidence_one','text':{'rank':1},'scope':'computed'}}
    a=update_feedback([],review(),'evidence',0,selected,{'requests':1})
    b=update_feedback(a,review(),'evidence',1,selected,{'requests':3})
    assert b[-1]['tasks'][0]['id']==a[-1]['tasks'][0]['id']
    assert b[-1]['no_progress_streak']==2
    passed={'verdict':'pass','return_to':'none','issues':[]}
    c=update_feedback(b,passed,'decision',2,selected,{})
    assert c[-1]['tasks'][0]['status']=='open'
    d=update_feedback(c,passed,'evidence',3,selected,{})
    assert d[-1]['tasks'][0]['status']=='closed_by_stage_review'
    assert not repair_tasks(d)
    assert len(update_feedback(d,passed,'evidence',3,selected,{}))==len(d)


def test_new_evidence_resets_stall_and_removed_issue_not_auto_closed():
    a=update_feedback([],review(),'evidence',0,{}, {})
    b=update_feedback(a,review(),'evidence',1,{'E1':{'text':'new'}},{})
    assert b[-1]['no_progress_streak']==1
    c=update_feedback(b,review(issues=['Different missing support']),'evidence',2,{}, {})
    assert c[-1]['tasks'][0]['status']=='not_repeated_unverified'


def test_explicit_id_priority_and_scope_not_semantic_validation():
    tasks=[{'issue':'Need source_exact GO Zincgluconate','status':'open'}]
    exact={'id':'source_exact','title':'x','provenance':{}}
    related={'id':'source_other','title':'GO Zincgluconate','provenance':{}}
    assert source_priority(exact,tasks)>source_priority(related,tasks)>0
    report=claim_coverage([{'id':'claim_a','kind':'proposal','evidence_ids':['evidence_missing']},{'id':'claim_b','kind':'proposal','evidence_ids':['evidence_ok']}],{'E1':{'id':'evidence_ok'}})
    assert report['claims'][0]['status']=='missing_binding'
    assert report['claims'][1]['status']=='bound_unverified'
    assert report['scientific_support_assessed'] is False


def test_stall_is_durable_and_exported(tmp_path):
    import json
    from adj_partner.contracts import ModelReply
    from adj_partner.orchestration.exports import export_task
    from test_framework_design import seeded
    s,f,t,p=seeded(tmp_path)
    original=p.complete
    def complete(messages,tools,**kwargs):
        if not tools and json.loads(messages[-1]['content']).get('stage')=='evidence':
            p.requests.append({'messages':messages,'tools':tools})
            return ModelReply(content=json.dumps(review(issues=['Need exact candidate comparison'])),usage={'input_tokens':1,'output_tokens':1})
        return original(messages,tools,**kwargs)
    p.complete=complete
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='waiting',result
    assert result['error']['code']=='no_progress'
    calls=len(p.requests)
    assert f.run(t['fingerprint'])==result
    assert len(p.requests)==calls
    export_task(f,t['fingerprint'],tmp_path/'export')
    assert json.loads((tmp_path/'export/feedback.json').read_text())[-1]['no_progress_streak']==3
    s.close()


def test_explicit_evidence_selection_survives_budget_prioritization(tmp_path):
    from test_framework_design import seeded
    s,f,t,p=seeded(tmp_path)
    sid,rid=t['session_id'],t['initial_run_id']
    sources=s.research.list_sources(sid,rid)
    e=s.research.select_evidence(sid,rid,sources[-1]['id'],{'pointer':'/abstract'})
    selected,_=f._materialize(sid,rid,evidence_budget_bytes=5000,repair_items=[{'issue':'Need '+e['id'],'status':'open'}])
    assert e['id'] in [v['id'] for v in selected.values()]
    s.close()


def test_database_records_enter_qa_and_design_packets(tmp_path):
    import json
    from test_framework_design import seeded
    from adj_partner.research.qa_review_context import select_evidence
    s,f,t,p=seeded(tmp_path);sid,rid=t['session_id'],t['initial_run_id']
    prov={'text_availability':'database_record','url':'https://example.test/record'}
    a=s.store.put_artifact(json.dumps({'record':{'molecule':'candidateZ','activity':3.2}}),'db.json','application/json',prov,session_id=sid,run_id=rid)
    src=s.research.register_source(sid,rid,artifact_id=a['id'],kind='web',title='candidateZ record',provenance=prov,idempotency_key='db')
    design,_=f._materialize(sid,rid)
    assert any(v['scope']=='database_record' and v['source_id']==src['id'] for v in design.values())
    qa=select_evidence('candidateZ activity','',f._documents(sid,rid))
    assert any(v['text_availability']=='database_record' and '3.2' in str(v) for v in qa)
    s.close()


def test_requested_original_source_retains_its_identity(tmp_path):
    from test_framework_design import seeded
    s,f,t,p=seeded(tmp_path);sid,rid=t['session_id'],t['initial_run_id']
    original=s.research.list_sources(sid,rid)[-1]
    selected,_=f._materialize(sid,rid,repair_items=[{'issue':'Need '+original['id'],'status':'open'}])
    assert original['id'] in [v['source_id'] for v in selected.values()]
    s.close()


def test_changed_design_is_measurable_progress():
    selected={'E1':{'text':'same'}}
    a=update_feedback([],review('decision'),'decision',0,selected,{},design={'selected_candidate':'A'})
    b=update_feedback(a,review('decision'),'decision',1,selected,{},design={'selected_candidate':'B'})
    assert b[-1]['no_progress_streak']==1
    assert b[-1]['design_changed'] is True
