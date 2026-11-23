"""Opt-in fixed paragraph count, with legacy acceptance and request-budget compatibility."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from adj_partner.research.acceptance import requirements, structural_issues

@pytest.mark.parametrize('value',[True,0,101,'2',2.0])
def test_paragraph_count_rejects_nonbounded_integers(value):
    with pytest.raises(ValueError,match='paragraphs_per_section'):
        requirements({'paragraphs_per_section':value})

def test_paragraph_count_is_opt_in_and_checks_each_section():
    assert 'paragraphs_per_section' not in requirements({})
    report={'artifact_id':'artifact_test','sections':[{'heading':'A','claim_ids':['c'],'paragraphs':[{'text':'First','claim_ids':['c']}]}],
            'claim_snapshots':[{'claim':{'id':'c','kind':'proposal'},'assessment':None,'assessment_current':False,'evidence':[],'sources':[]}],
            'limitations':['Concept only.']}
    assert structural_issues(report,{'acceptance':{}},{})==[]
    task={'acceptance':{'paragraphs_per_section':2}}
    assert structural_issues(report,task,{})==['paragraphs_per_section_mismatch: A: 1/2']
    report['sections'][0]['paragraphs'].append({'text':'Second','claim_ids':['c']})
    assert structural_issues(report,task,{})==[]
    report['sections'].append(deepcopy(report['sections'][0]))
    report['sections'][1]['heading']='B';report['sections'][1]['paragraphs'].append({'text':'Third','claim_ids':['c']})
    assert structural_issues(report,task,{})==['paragraphs_per_section_mismatch: B: 3/2']

def test_design_runner_snapshots_format_and_larger_request_space(tmp_path):
    from adj_partner.open_research_case import run_case
    from test_open_research_case import _FakeService
    data=tmp_path/'input.json';data.write_text('[{"id":"x","question":"Design a concept"}]')
    result=run_case(data,'x',tmp_path/'out',workflow='adjuvant-design-v1',service_factory=_FakeService)
    instance=_FakeService.instances[-1]
    assert instance.kwargs['request_output_tokens']==12000
    assert instance.acceptance['require_research_design'] is True
    manifest=json.loads(Path(result['manifest_path']).read_text())
    assert manifest['workflow']['acceptance']['require_research_design'] is True
    run_case(data,'x',tmp_path/'legacy',service_factory=_FakeService)
    assert 'request_output_tokens' not in _FakeService.instances[-1].kwargs

def test_real_report_delivery_rejects_wrong_count_without_rewriting_artifact(tmp_path):
    from test_harness_research_records import setup
    from adj_partner.research.acceptance import ReportNotAccepted
    store,research,sid,rid,_=setup(tmp_path,acceptance={'paragraphs_per_section':2})
    claim=research.record_claim(sid,rid,text='Unverified concept.',kind='proposal',evidence_ids=[])
    sections=[{'heading':'Decision','claim_ids':[claim['id']],
               'paragraphs':[{'text':'Proposed concept.','claim_ids':[claim['id']]}]}]
    first=research.create_report(sid,rid,title='Fixture',sections=sections,limitations=['Not validated.'])
    original=store.read_artifact(first['artifact_id'])
    assert not research.validate_report(sid,rid,first['id'])['accepted']
    with pytest.raises(ReportNotAccepted):research.accepted_report(sid,rid,first['id'])
    sections[0]['paragraphs'].append({'text':'Evidence is missing.','claim_ids':[claim['id']]})
    second=research.create_report(sid,rid,title='Fixture',sections=sections,limitations=['Not validated.'])
    assert research.validate_report(sid,rid,second['id'])['accepted']
    assert research.accepted_report(sid,rid,second['id'])['content']==store.read_artifact(second['artifact_id'])
    assert store.read_artifact(first['artifact_id'])==original
