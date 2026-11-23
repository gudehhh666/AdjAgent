import importlib.util
import pytest
from adj_partner.storage import Store
from adj_partner.contracts import SessionConfig, Budget


def api():
    assert importlib.util.find_spec('adj_partner.orchestration.contracts'), 'TaskSpec missing'
    from adj_partner.orchestration.contracts import TaskSpec
    from adj_partner.orchestration.repository import WorkflowRepository
    return TaskSpec,WorkflowRepository


def test_identity_changes_for_question_model_and_workflow():
    T,_=api();a=T('qa','Question',dataset_sha256='a'*64,row_id='1')
    assert a.fingerprint==T.from_dict(a.to_dict()).fingerprint
    assert len({a.fingerprint,T('qa','Other',dataset_sha256='a'*64,row_id='1').fingerprint,T('qa','Question',model='other').fingerprint,T('qa','Question',workflow_version='v2').fingerprint})==4
    with pytest.raises(ValueError):T.from_dict({'task_type':'qa','question':'Q','answer':'GOLD'})
    with pytest.raises(ValueError):T('unknown','Q')


def test_repository_reopen_and_fenced_ownership(tmp_path):
    T,R=api();s=Store(tmp_path/'s');sid=s.create_session(SessionConfig().to_dict())['id'];tid=s.create_task(sid,'Q')['id'];rid=s.create_run(sid,tid,Budget().to_dict())['id']
    r=R(s);spec=T('qa','Q');r.bind(spec,sid,tid,rid)
    with r.owned(spec.fingerprint) as g:
        with pytest.raises(RuntimeError):
            with R(Store(tmp_path/'s')).owned(spec.fingerprint):pass
        r.save(spec.fingerprint,{'phase':'review','evidence':'artifact'},guard=g)
    assert R(Store(tmp_path/'s')).get(spec.fingerprint)['state']['phase']=='review'


def test_result_never_infers_empirical_validation():
    T,_=api()
    from adj_partner.orchestration.contracts import result_record
    r=result_record(T('design','Q'),'succeeded',{'kind':'accepted_report','research_design_status':'incomplete_design','content':'report'},review_status='passed')
    assert r['content_status']=='incomplete' and r['empirical_validation_status']=='not_performed'
    assert r['delivery_status']=='report'


def test_qa_full_workflow_reuses_reviewed_answer_without_new_requests(tmp_path):
    assert importlib.util.find_spec('adj_partner.orchestration.controller'), 'Workflow controller missing'
    from adj_partner.orchestration.controller import AgentFramework
    from adj_partner.application import AgentService
    from adj_partner.models.base import ScriptedPort
    from adj_partner.contracts import ModelReply
    import json
    T,_=api()
    port=ScriptedPort([ModelReply(content='Qualified draft',usage={'input_tokens':10,'output_tokens':4}),ModelReply(content=json.dumps({'answer':'Evidence unavailable; qualified answer.','sources':[],'review':'Removed unsupported certainty.','uncertainty':'No primary source confirmed.'}),usage={'input_tokens':20,'output_tokens':10})])
    svc=AgentService(tmp_path/'state',model=port)
    f=AgentFramework(svc);task=f.submit(T('qa','A synthetic question',model='offline-script',allow_network=False))
    result=f.run(task['fingerprint'])
    assert result['execution_status']=='succeeded',result
    assert result['delivery']['content'].startswith('Evidence unavailable')
    assert result['empirical_validation_status']=='not_performed'
    assert f.run(task['fingerprint'])==result and len(port.requests)==2
    assert result['attempt_refs'][0]!=result['attempt_refs'][1]
    svc.close()


def test_review_transport_failure_does_not_repeat_retrieval(tmp_path):
    assert importlib.util.find_spec('adj_partner.orchestration.controller'), 'Workflow controller missing'
    from adj_partner.orchestration.controller import AgentFramework
    from adj_partner.application import AgentService
    from adj_partner.models.base import ScriptedPort,ProviderFailure
    from adj_partner.contracts import ModelReply
    T,_=api();p=ScriptedPort([ModelReply(content='Draft',usage={'input_tokens':4,'output_tokens':2}),ProviderFailure('timeout','unknown')])
    s=AgentService(tmp_path/'state',model=p);f=AgentFramework(s);t=f.submit(T('qa','Q',model='offline-script',allow_network=False))
    assert f.run(t['fingerprint'])['execution_status']=='failed'
    assert f.run(t['fingerprint'])['execution_status']=='failed'
    assert len(p.requests)==2
    s.close()


def test_queue_submission_needs_no_provider_credentials(tmp_path,monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.orchestration.controller import AgentFramework
    for key in ('ADJ_LLM_API_KEY','ADJ_LLM_BASE_URL','ADJ_LLM_MODEL'):monkeypatch.delenv(key,raising=False)
    s=AgentService(tmp_path/'state');f=AgentFramework(s);T,_=api()
    item=f.submit(T('qa','Public queued question'))
    assert item['state']['phase']=='retrieval'
    assert s.models.ports=={}
    s.close()


def test_provider_change_creates_a_different_execution_identity(tmp_path,monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.orchestration.controller import AgentFramework
    for key in ('ADJ_LLM_API_KEY','ADJ_LLM_BASE_URL','ADJ_LLM_MODEL'):monkeypatch.delenv(key,raising=False)
    s=AgentService(tmp_path/'state');f=AgentFramework(s);T,_=api()
    monkeypatch.setenv('ADJ_LLM_BASE_URL','https://one.example/v1')
    a=f.submit(T('qa','Public question'))
    monkeypatch.setenv('ADJ_LLM_BASE_URL','https://two.example/v1')
    b=f.submit(T('qa','Public question'))
    assert a['fingerprint']!=b['fingerprint']
    assert a['spec']['provider_sha256'] and a['spec']['provider_sha256']!=b['spec']['provider_sha256']
    assert 'one.example' not in str(a['spec'])
    s.close()
