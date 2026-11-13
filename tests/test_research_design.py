from copy import deepcopy
import pytest
from adj_partner.research import reports


def design(cid):
    return {'requirements':[{'requirement':'Explain a testable research choice','answer':'Candidate A is a hypothesis','status':'addressed','claim_ids':[cid]}],
            'candidates':[{'name':'A','rationale':'Source-bounded choice','claim_ids':[cid]}, {'name':'B','rationale':'Comparator with a different limitation','claim_ids':[cid]}],
            'selected_candidate':'A','mechanism':{'text':'A conditional explanation','claim_ids':[cid]},
            'verification':{'text':'Compare the primary outcome and stop if the assumption fails','claim_ids':[cid]},'blocking_gaps':[]}


def test_design_validation_and_declared_readiness():
    assert hasattr(reports, 'validate_design'), 'Structured research decisions are not checked'
    d=design('claim_x');assert reports.validate_design(d,{'claim_x'})==d
    from adj_partner.research.design import design_status
    assert design_status(d)=='documented_design'
    d['requirements'][0]['status']='open';assert design_status(d)=='incomplete_design'
    d['requirements'][0]['status']='addressed';d['blocking_gaps']=['Missing direct comparison'];assert design_status(d)=='incomplete_design'
    d['blocking_gaps']=[];d['selected_candidate']='';assert design_status(d)=='incomplete_design'
    with pytest.raises(ValueError):reports.validate_design(design('claim_foreign'),{'claim_x'})
    d=design('claim_x');d['selected_candidate']='unlisted'
    with pytest.raises(ValueError):reports.validate_design(d,{'claim_x'})
    d=design('claim_x');d['requirements'][0]['status']='proved'
    with pytest.raises(ValueError):reports.validate_design(d,{'claim_x'})


def test_real_report_binds_design_and_preserves_legacy_bytes(tmp_path):
    from test_harness_research_records import setup
    from adj_partner.research.acceptance import requirements
    assert 'require_research_design' not in requirements({})
    store,research,sid,rid,_=setup(tmp_path,acceptance={'report_layout':'review'})
    c=research.record_claim(sid,rid,text='Unverified material-science hypothesis.',kind='proposal',evidence_ids=[])
    kw={'title':'Research','sections':[{'heading':'Decision','claim_ids':[c['id']], 'paragraphs':[{'text':'A provisional decision.','claim_ids':[c['id']]}]}],'limitations':['No experimental confirmation.']}
    old=research.create_report(sid,rid,**kw);original=store.read_artifact(old['artifact_id']);research.validate_report(sid,rid,old['id'])
    new=research.create_report(sid,rid,**kw,research_design=design(c['id']))
    assert new['renderer_version']==4
    assert research.validate_report(sid,rid,new['id'])['accepted']
    assert research.accepted_report(sid,rid,old['id'])['content']==original
    from adj_partner.research.design import readable_report
    view=readable_report(new).decode()
    assert 'A provisional decision.' in view and 'No experimental confirmation.' in view
    assert '## 证据摘录' not in view and 'report.md#c1' in view
    assert new['artifact_hash'] in view
    assert 'documented_design' in store.read_artifact(new['artifact_id']).decode()
    from adj_partner.runtime.engine import AgentEngine
    from types import SimpleNamespace
    delivery=AgentEngine._accepted(SimpleNamespace(research=research),sid,rid)
    assert delivery['report_id']==new['id']
    assert delivery['research_design_status']=='documented_design'
    assert delivery['readable_content']==view


def test_workflow_requires_research_decisions_not_exact_paragraph_count():
    from adj_partner.workflows import load_workflow
    a=load_workflow('adjuvant-design-v1')['acceptance']
    assert a.get('require_research_design') is True
    assert 'paragraphs_per_section' not in a


def test_runtime_does_not_deliver_earlier_run_report():
    from adj_partner.runtime.engine import AgentEngine
    from types import SimpleNamespace
    def unexpected(*args):
        pytest.fail('Earlier run report was used as the new run delivery')
    fake=SimpleNamespace(research=SimpleNamespace(list_reports=lambda *a:[{'id':'old','run_id':'previous'}],accepted_report=unexpected))
    assert AgentEngine._accepted(fake,'session','current') is None


def test_missing_design_rejected_but_honest_gap_kept_as_incomplete(tmp_path):
    from test_harness_research_records import setup
    from adj_partner.research.design import design_status
    store,research,sid,rid,_=setup(tmp_path,acceptance={'require_research_design':True})
    c=research.record_claim(sid,rid,text='An unresolved research question.',kind='proposal',evidence_ids=[])
    kw={'title':'Incomplete','sections':[{'heading':'Decision','claim_ids':[c['id']]}],'limitations':['Evidence is missing.']}
    missing=research.create_report(sid,rid,**kw)
    assert 'research_design_required' in research.validate_report(sid,rid,missing['id'])['issues']
    d=design(c['id']);d['requirements'][0]['status']='open';d['blocking_gaps']=['Selection evidence missing']
    gap=research.create_report(sid,rid,**kw,research_design=d)
    assert research.validate_report(sid,rid,gap['id'])['accepted']
    assert design_status(gap['research_design'])=='incomplete_design'
    assert 'Selection evidence missing' in store.read_artifact(gap['artifact_id']).decode()


def test_structured_design_tool_field_remains_optional_for_legacy():
    from adj_partner.research.design import tool_schema
    from adj_partner.research.tools import ID
    from adj_partner.tools.catalog import validate_schema, validator
    # Use the same host argument checker as report_create.
    schema=tool_schema(ID);validate_schema(schema);validator(schema).validate(design('claim_x'))


def test_readable_view_preserves_blocking_decisions(tmp_path):
    from test_harness_research_records import setup
    from adj_partner.research.design import readable_report
    store,research,sid,rid,_=setup(tmp_path,acceptance={})
    c=research.record_claim(sid,rid,text='Conditional hypothesis.',kind='proposal',evidence_ids=[])
    d=design(c['id']);d['blocking_gaps']=['A crucial contradiction remains unresolved.']
    r=research.create_report(sid,rid,title='Decision',sections=[{'heading':'Summary','claim_ids':[c['id']]}],limitations=['Provisional.'],research_design=d)
    v=readable_report(r).decode()
    assert d['blocking_gaps'][0] in v
    assert d['mechanism']['text'] in v and d['verification']['text'] in v


def test_evidence_map_uses_only_bound_claim_sources(tmp_path):
    from test_harness_research_records import setup, source
    from adj_partner.research import design as module
    assert hasattr(module,'design_evidence_map')
    store,research,sid,rid,_=setup(tmp_path,acceptance={})
    s,_=source(store,research,sid,rid,content='A literal material observation.')
    e=research.select_evidence(sid,rid,s['id'],{'passage':'A literal material observation.'})
    c=research.record_claim(sid,rid,text='A conditional choice.',kind='proposal',evidence_ids=[e['id']])
    r=research.create_report(sid,rid,title='Map',sections=[{'heading':'Decision','claim_ids':[c['id']]}],limitations=['Scope limited.'],research_design=design(c['id']))
    mapped=module.design_evidence_map(r)
    assert mapped['scientific_adequacy']=='not_assessed'
    assert mapped['items'][0]['evidence_ids']==[e['id']]
    assert mapped['items'][0]['source_ids']==[s['id']]
    assert mapped['report_sha256']==r['artifact_hash']


def test_literal_observation_policy_avoids_unassessed_authored_fact_loop(tmp_path):
    from test_harness_research_records import setup, source
    from adj_partner.research.acceptance import requirements
    assert 'require_literal_observations' not in requirements({})
    assert requirements({'require_literal_observations':True})['require_literal_observations']
    store,research,sid,rid,_=setup(tmp_path,acceptance={'require_literal_observations':True})
    s,_=source(store,research,sid,rid,content='Observed in this material only.')
    e=research.select_evidence(sid,rid,s['id'],{'passage':'Observed in this material only.'})
    with pytest.raises(ValueError,match='claim_observe'):
        research.record_claim(sid,rid,text='An authored generalization.',kind='observation',evidence_ids=[e['id']])
    c=research.observe_evidence(sid,rid,e['id'])
    assert c['origin']=='host_observation'
    with pytest.raises(ValueError,match='claim_observe'):
        research.revise_claim(sid,rid,c['id'],expected_revision=0,text='Rewritten assertion.',kind='observation',evidence_ids=[e['id']])


def test_unavailable_assessor_is_disclosed_before_tool_call(tmp_path):
    from test_harness_research_records import setup
    from test_harness_research_tools import tools
    store,research,sid,rid,_=setup(tmp_path)
    catalog,_=tools(store,research,sid)
    assert 'No semantic assessor is configured' in catalog.get('claim_assess').spec.description
    schema=catalog.get('report_create').spec.input_schema
    assert 'research_design' in schema['properties']
    assert 'research_design' not in schema['required']
