import pytest

def test_alias_binding_is_exact_and_only_changes_claim_id_fields():
    from adj_partner.research.staged_design import bind_claims
    value={'text':'O1 is prose','claim_ids':['O1'],'nested':[{'claim_ids':['P1']}]}
    assert bind_claims(value,{'O1':'claim_observed','P1':'claim_proposal'})=={'text':'O1 is prose','claim_ids':['claim_observed'],'nested':[{'claim_ids':['claim_proposal']}]}
    with pytest.raises(ValueError,match='Unknown claim alias'):
        bind_claims({'claim_ids':['invented']},{})

def test_abstract_parts_are_exact_nonoverlapping_passages():
    from adj_partner.research.staged_design import abstract_parts
    text='First observation. Second observation. Third observation.'
    parts=abstract_parts(text)
    assert len(parts)==3 and all(p in text for p in parts)
    assert len(set(parts))==3


def staged_fixture(tmp_path,monkeypatch,replies):
    import json,hashlib
    import adj_partner.application as application
    from adj_partner.contracts import ToolResult
    from test_research_critic import Luna
    real=application.AgentService;port=Luna(replies)
    def factory(*args,**kwargs):
        svc=real(*args,**kwargs);svc.set_model=lambda *a:None
        svc.models.port=lambda sid:port
        svc.critic.port_factory=lambda sid:port
        svc.executor.execute=lambda *a:ToolResult(status='error',error={'code':'fixture','message':'No network in fixture'})
        return svc
    monkeypatch.setattr(application,'AgentService',factory)
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps([{'id':'qa_001','question':'Fixture question'}]))
    sources=[]
    for i in range(6):
        raw=json.dumps({'abstract':f'Finding {i} one. Finding {i} two. Finding {i} three.'})
        sources.append({'title':f'Fixture {i}','content_utf8':raw,'sha256':hashlib.sha256(raw.encode()).hexdigest(),'provenance':{'doi':f'10.1234/fixture{i}','text_availability':'abstract'}})
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'question':'Fixture question','sources':sources}))
    return dataset,bundle,port


@pytest.mark.parametrize('mode',['ordinary','invalid','resume'])
def test_staged_success_performs_real_structural_acceptance(tmp_path,monkeypatch,mode):
    invalid_first=mode=='invalid'
    import json
    from adj_partner.contracts import ModelReply
    from adj_partner.research.staged_design import run_staged
    from adj_partner.workflows import load_workflow
    from test_research_critic import verdict
    from test_research_design import design
    def reply(value):return ModelReply(content=json.dumps(value),usage={'input_tokens':100,'output_tokens':50})
    evidence={'brief':'Synthetic comparison A versus B','evidence_keys':[f'T{i}' for i in range(1,9)]}
    decision={'proposals':[{'key':'P1','text':'Conditional synthetic proposal','evidence_keys':[f'E{i}' for i in range(1,9)]}],'research_design':design('P1')}
    report={'title':'Fixture','sections':[{'heading':h,'paragraphs':[{'text':'Synthetic conditional decision with explicit evidence boundary.','claim_ids':['P1']}]} for h in load_workflow('adjuvant-design-critic-v1')['acceptance']['required_sections']],'limitations':['Synthetic fixture only']}
    from copy import deepcopy
    invalid=deepcopy(decision);invalid['research_design']['candidates']=invalid['research_design']['candidates'][:1]
    replies=[reply(evidence),verdict()]+([reply(invalid)] if invalid_first else [])+[reply(decision),verdict(),reply(report),verdict()]
    if mode=='resume':
        from adj_partner.models.base import ProviderFailure
        replies=replies[:4]+[ProviderFailure('fixture_connection','Fixture connection',cost_unknown=True)]
    dataset,bundle,port=staged_fixture(tmp_path,monkeypatch,replies)
    result=run_staged(dataset,'qa_001',tmp_path/'run',bundle)
    if mode=='resume':
        assert result['status']=='failed'
        port._replies=iter([reply(report),verdict()])
        before=(tmp_path/'run/manifest.json').read_bytes()
        with pytest.raises(ValueError,match='Terminal'):
            run_staged(dataset,'qa_001',tmp_path/'run',bundle,resume=True)
        assert (tmp_path/'run/manifest.json').read_bytes()==before
        assert result['synthesis_accounting']['unknown_requests']==1
        assert len(port.requests)==5
        return
    assert result['status']=='succeeded',result
    assert (tmp_path/'run/report.md').exists()
    assert len(port.requests)==6+int(mode!='ordinary')


def test_staged_retains_failed_request_known_usage(tmp_path,monkeypatch):
    from adj_partner.models.base import ProviderFailure
    from adj_partner.research.staged_design import run_staged
    dataset,bundle,_=staged_fixture(tmp_path,monkeypatch,[ProviderFailure('connection','Fixture failure',usage={'input_tokens':123,'output_tokens':7},cost_unknown=False)])
    result=run_staged(dataset,'qa_001',tmp_path/'run',bundle)
    assert result['status']=='failed'
    assert result['synthesis_accounting']['input_tokens']==123
    assert result['synthesis_accounting']['unknown_requests']==0


def test_staged_empty_success_usage_is_unknown(tmp_path,monkeypatch):
    from adj_partner.contracts import ModelReply
    from adj_partner.research.staged_design import run_staged
    dataset,bundle,_=staged_fixture(tmp_path,monkeypatch,[ModelReply(refusal='Fixture refusal',usage={})])
    result=run_staged(dataset,'qa_001',tmp_path/'run',bundle)
    assert result['status']=='failed'
    assert result['synthesis_accounting']['requests']==1
    assert result['synthesis_accounting']['unknown_requests']==1


def test_report_cannot_repair_frozen_design_without_upstream_review():
    from adj_partner.research.staged_design import revision_stage
    assert revision_stage('report',{'return_to':'report','issues':['修正研究决策记录的Tc1比较']})=='decision'
    assert revision_stage('report',{'return_to':'report','issues':['删除正文重复句']})=='report'
    assert revision_stage('report',{'return_to':'evidence','issues':['来源不足']})=='evidence'

def test_resume_port_initialization_failure_preserves_accounting(tmp_path,monkeypatch):
    import json
    import adj_partner.application as application
    from adj_partner.models.base import ProviderFailure
    from adj_partner.research.staged_design import run_staged
    dataset,bundle,_=staged_fixture(tmp_path,monkeypatch,[ProviderFailure('fixture','Fixture',usage={'input_tokens':9,'output_tokens':1},cost_unknown=False)])
    first=run_staged(dataset,'qa_001',tmp_path/'run',bundle)
    factory=application.AgentService
    def failing(*args,**kwargs):
        svc=factory(*args,**kwargs)
        def unavailable(sid):raise ValueError('Fixture port initialization failure')
        svc.models.port=unavailable
        return svc
    monkeypatch.setattr(application,'AgentService',failing)
    with pytest.raises(ValueError,match='Terminal'):
        run_staged(dataset,'qa_001',tmp_path/'run',bundle,resume=True)
    second=json.loads((tmp_path/'run/manifest.json').read_text())
    assert second['synthesis_accounting']==first['synthesis_accounting']
    assert json.loads((tmp_path/'run/manifest.json').read_text())['synthesis_accounting']['input_tokens']==9
