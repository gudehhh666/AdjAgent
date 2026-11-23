"""Offline CLI/campaign acceptance with the real service, database and artifacts."""
import hashlib
import importlib.util
import json
import pytest
from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ScriptedPort, ProviderFailure
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework


def modules():
    assert importlib.util.find_spec('adj_partner.orchestration.cli'), 'Unified framework CLI is missing'
    from adj_partner.orchestration import cli, campaign, exports
    return cli, campaign, exports


def port():
    return ScriptedPort([
        ModelReply(content='Qualified draft', usage={'input_tokens':10,'output_tokens':4}),
        ModelReply(content=json.dumps({'answer':'Qualified answer.', 'sources':[], 'review':'No unsupported certainty.',
                                      'uncertainty':'No primary evidence.'}), usage={'input_tokens':20,'output_tokens':8})])


def test_run_status_export_and_resume_use_one_persisted_workflow(tmp_path, monkeypatch, capsys):
    cli, _, _ = modules()
    p = port(); svc = AgentService(tmp_path/'state', model=p)
    monkeypatch.setattr(cli, 'AgentService', lambda root: svc)
    monkeypatch.setattr(svc, 'close', lambda: None)
    spec = tmp_path/'spec.json'; spec.write_text(json.dumps(TaskSpec('qa','Synthetic?',model='offline-script',allow_network=False).to_dict()))
    assert cli.main(['--state', str(tmp_path/'state'), 'run', '--spec', str(spec)]) == 0
    run = json.loads(capsys.readouterr().out); fp = run['task_fingerprint']
    assert len(p.requests) == 2
    assert cli.main(['--state', str(tmp_path/'state'), 'resume', fp]) == 0
    assert json.loads(capsys.readouterr().out)['execution_status'] == 'succeeded'
    assert len(p.requests) == 2
    assert cli.main(['status', fp]) == 0
    assert json.loads(capsys.readouterr().out)['fingerprint'] == fp
    target = tmp_path/'export'
    assert cli.main(['export', fp, '--output', str(target)]) == 0
    json.loads(capsys.readouterr().out)
    result = json.loads((target/'result.json').read_text())
    assert result['execution_status']=='succeeded' and result['delivery_status']=='answer'
    assert result['review_status']=='passed' and result['empirical_validation_status']=='not_performed'
    assert (target/'answer.md').read_text() == 'Qualified answer.'
    assert result['artifact_verification']['status']=='verified'


def test_campaign_projects_only_question_and_id_and_reopens_queue(tmp_path):
    _, campaign, _ = modules()
    dataset=tmp_path/'data.jsonl'
    raw='\n'.join(json.dumps(x) for x in [{'id':'a','question':'First?','answer':'SECRET_GOLD','rubric':'SECRET_RUBRIC'},
                                       {'id':'b','question':'Second?','answer':'SECRET_GOLD'},
                                       {'id':'c','answer':'SECRET_GOLD'}])
    dataset.write_text(raw)
    p=port();svc=AgentService(tmp_path/'state',model=p);framework=AgentFramework(svc)
    q=campaign.Campaign(framework)
    manifest=q.prepare(dataset,task_type='qa',model='offline-script')
    assert len(p.requests)==0
    assert len(manifest['items'])==3 and manifest['items'][2]['status']=='invalid'
    serialized=json.dumps(manifest)+json.dumps(framework.repository.list())
    assert 'SECRET_' not in serialized and 'rubric' not in serialized
    assert manifest['dataset_sha256']==hashlib.sha256(raw.encode()).hexdigest()
    fingerprints=[i['fingerprint'] for i in manifest['items'][:2]]
    specs=[framework.repository.get(fp)['spec'] for fp in fingerprints]
    assert specs[0]['row_id'] != specs[1]['row_id']
    assert manifest['implementation_sha256'] == specs[0]['implementation_sha256']
    assert q.prepare(dataset,task_type='qa',model='offline-script')['id']==manifest['id']
    reopened=campaign.Campaign(AgentFramework(AgentService(tmp_path/'state',model=p)))
    assert reopened.get(manifest['id'])==manifest


def test_campaign_execute_is_bounded_and_does_not_retry_uncertain_call(tmp_path):
    _, campaign, _ = modules()
    dataset=tmp_path/'data.json';dataset.write_text(json.dumps([{'id':'a','question':'Q'},{'id':'b','question':'Other'}]))
    p=ScriptedPort([ProviderFailure('timeout','Unknown external completion')])
    svc=AgentService(tmp_path/'state',model=p);q=campaign.Campaign(AgentFramework(svc))
    manifest=q.prepare(dataset,task_type='qa',model='offline-script',limit=1)
    result=q.execute(manifest['id'])
    assert result['items'][0]['status'] in {'failed','needs_attention'}
    assert len(p.requests)==1
    assert q.execute(manifest['id'])==result and len(p.requests)==1


def test_export_refuses_tampered_artifact_without_writing_delivery(tmp_path):
    _, _, exports = modules()
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    item=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False));result=f.run(item['fingerprint'])
    (svc.store.root/'artifacts'/result['delivery']['sha256']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash|integrity'):
        exports.export_task(f,item['fingerprint'],tmp_path/'bad-export')
    assert not (tmp_path/'bad-export'/'answer.md').exists()


def test_cli_rejects_unknown_spec_fields_without_submit(tmp_path, monkeypatch, capsys):
    cli, _, _ = modules()
    spec=tmp_path/'bad.json';spec.write_text(json.dumps({'task_type':'qa','question':'Q','answer':'GOLD'}))
    svc=AgentService(tmp_path/'state',model=port())
    monkeypatch.setattr(cli,'AgentService',lambda root:svc)
    monkeypatch.setattr(svc,'close',lambda:None)
    assert cli.main(['run','--spec',str(spec)])==2
    assert 'GOLD' not in capsys.readouterr().err
    assert AgentFramework(svc).repository.list()==[]


def test_design_export_verifies_report_and_emits_lineage(tmp_path):
    _, _, exports = modules()
    from test_framework_design import seeded
    svc, framework, item, p = seeded(tmp_path)
    result = framework.run(item['fingerprint'])
    assert result['execution_status'] == 'succeeded', result
    target = tmp_path/'report-export'
    exported = exports.export_task(framework, item['fingerprint'], target)
    assert exported['artifact_verification']['status'] == 'verified'
    assert hashlib.sha256((target/'report.md').read_bytes()).hexdigest() == result['delivery']['sha256']
    assert json.loads((target/'lineage.json').read_text()) == result['lineage']


def test_resume_blocks_unknown_request_without_a_second_call(tmp_path, monkeypatch, capsys):
    cli, _, _ = modules()
    p = ScriptedPort([ProviderFailure('timeout','Unknown completion')])
    svc = AgentService(tmp_path/'state', model=p)
    f = AgentFramework(svc)
    item = f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False))
    f.run(item['fingerprint'])
    monkeypatch.setattr(cli,'AgentService',lambda root:svc)
    monkeypatch.setattr(svc,'close',lambda:None)
    assert cli.main(['resume',item['fingerprint']]) == 2
    assert len(p.requests)==1
    assert 'unknown' in capsys.readouterr().err.lower()


def test_cached_resume_cannot_return_corrupt_success(tmp_path, monkeypatch, capsys):
    cli, _, _ = modules()
    p=port();svc=AgentService(tmp_path/'state',model=p);f=AgentFramework(svc)
    item=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False))
    result=f.run(item['fingerprint'])
    (svc.store.root/'artifacts'/result['delivery']['sha256']).write_bytes(b'corrupt cached bytes')
    monkeypatch.setattr(cli,'AgentService',lambda root:svc)
    monkeypatch.setattr(svc,'close',lambda:None)
    assert cli.main(['resume',item['fingerprint']])==2
    assert 'succeeded' not in capsys.readouterr().out
    assert len(p.requests)==2


def test_campaign_repeated_execute_validates_cached_success(tmp_path):
    _, campaign, _ = modules()
    dataset=tmp_path/'data.json';dataset.write_text('[{"id":"a","question":"Q"}]')
    p=port();svc=AgentService(tmp_path/'state',model=p);f=AgentFramework(svc);q=campaign.Campaign(f)
    m=q.prepare(dataset,task_type='qa',model='offline-script');completed=q.execute(m['id'])
    assert q.execute(m['id'])==completed and len(p.requests)==2
    result=completed['items'][0]['result']
    (svc.store.root/'artifacts'/result['delivery']['sha256']).write_bytes(b'corrupt')
    checked=q.execute(m['id'])
    assert checked['items'][0]['status']=='needs_attention'
    assert len(p.requests)==2


def test_export_rejects_output_directory_from_another_task(tmp_path):
    _, _, exports=modules()
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    first=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False));f.run(first['fingerprint'])
    target=tmp_path/'export';exports.export_task(f,first['fingerprint'],target)
    second=f.submit(TaskSpec('qa','Other',model='offline-script',allow_network=False))
    with pytest.raises(ValueError,match='directory'):
        exports.export_task(f,second['fingerprint'],target)
    assert json.loads((target/'result.json').read_text())['task_fingerprint']==first['fingerprint']


def test_old_implementation_resume_requires_current_code(tmp_path, monkeypatch, capsys):
    cli, _, _=modules()
    p=port();svc=AgentService(tmp_path/'state',model=p);f=AgentFramework(svc)
    item=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False,implementation_sha256='a'*64))
    monkeypatch.setattr(cli,'AgentService',lambda root:svc)
    monkeypatch.setattr(svc,'close',lambda:None)
    assert cli.main(['resume',item['fingerprint']])==2
    assert len(p.requests)==0


def test_export_preserves_incomplete_content_and_independent_statuses(tmp_path):
    _, _, exports=modules()
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    item=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False));result=f.run(item['fingerprint'])
    result.update(content_status='incomplete',review_status='needs_revision',empirical_validation_status='not_performed')
    current=f.repository.get(item['fingerprint'])
    with f.repository.owned(item['fingerprint']) as guard:
        f.repository.save(item['fingerprint'],current['state'],guard=guard,result=result)
    target=tmp_path/'incomplete';exports.export_task(f,item['fingerprint'],target)
    saved=json.loads((target/'result.json').read_text())
    assert [saved[k] for k in ('execution_status','delivery_status','content_status','review_status','empirical_validation_status')]==['succeeded','answer','incomplete','needs_revision','not_performed']


def test_export_rejects_result_saved_before_run_finalization(tmp_path):
    _, _, exports=modules()
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    item=f.submit(TaskSpec('qa','Q',model='offline-script',allow_network=False));f.run(item['fingerprint'])
    current=f.repository.get(item['fingerprint']);rid=current['state']['stage_run']
    with svc.store.transaction(write=True) as conn:
        conn.execute("UPDATE runs SET status='running' WHERE id=?",(rid,))
    with pytest.raises(ValueError,match='finalization'):
        exports.export_task(f,item['fingerprint'],tmp_path/'pending')
    assert not (tmp_path/'pending').exists()


def test_campaign_identity_changes_with_implementation_or_exact_input(tmp_path, monkeypatch):
    _, campaign, _=modules()
    dataset=tmp_path/'data.json';dataset.write_text('[{"id":"a","question":"Q"}]')
    svc=AgentService(tmp_path/'state',model=port());q=campaign.Campaign(AgentFramework(svc))
    original=campaign.TaskSpec
    monkeypatch.setattr(campaign,'TaskSpec',lambda *a,**kw:original(*a,**{'implementation_sha256':'a'*64,**kw}))
    first=q.prepare(dataset,task_type='qa',model='offline-script')
    assert q.prepare(dataset,task_type='qa',model='offline-script')==first
    monkeypatch.setattr(campaign,'TaskSpec',lambda *a,**kw:original(*a,**{'implementation_sha256':'b'*64,**kw}))
    second=q.prepare(dataset,task_type='qa',model='offline-script')
    dataset.write_text(dataset.read_text()+'\n')
    third=q.prepare(dataset,task_type='qa',model='offline-script')
    assert len({m['id'] for m in (first,second,third)})==3
    assert len({m['items'][0]['fingerprint'] for m in (first,second,third)})==3


@pytest.mark.parametrize('dataset_name,extra',[
    ('adj_common_qa',{}),('adj_common_design',{}),('adj_common_design_L1',{'difficulty_level':'L1'})])
def test_repository_dataset_schemas_project_only_public_question(tmp_path,dataset_name,extra):
    _, campaign, _=modules()
    # Exact key schema of the three repository datasets, synthetic values only.
    keys=['Reasoning_Process:','Location:','Basis:','Answer:','Model_Level:','Q_diffusion:',
          'Q:','R:','A:','Q_level:','A_level:','Reason for Error:','generated_model','level','verification','png_path']
    row={k:'HIDDEN_VALUE' for k in keys}
    row.update({'Question:':'Public synthetic question','original_question':'HIDDEN_VALUE',**extra})
    dataset=tmp_path/(dataset_name+'.jsonl');dataset.write_text(json.dumps(row)+'\n')
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    manifest=campaign.Campaign(f).prepare(dataset,task_type='qa',model='offline-script')
    assert manifest['items'][0]['status']=='queued'
    assert manifest['items'][0]['question']=='Public synthetic question'
    assert 'HIDDEN_VALUE' not in json.dumps(manifest)+json.dumps(f.repository.list())


def test_original_question_is_never_a_public_fallback(tmp_path):
    _,campaign,_=modules()
    dataset=tmp_path/'hidden.json';dataset.write_text('[{"original_question":"HIDDEN"}]')
    svc=AgentService(tmp_path/'state',model=port());f=AgentFramework(svc)
    manifest=campaign.Campaign(f).prepare(dataset,task_type='qa',model='offline-script')
    assert manifest['items'][0]['status']=='invalid' and f.repository.list()==[]


def test_report_export_rechecks_upstream_bindings_without_writing_validation(tmp_path):
    _,_,exports=modules()
    from test_framework_design import seeded
    svc,f,item,p=seeded(tmp_path);result=f.run(item['fingerprint'])
    current=f.repository.get(item['fingerprint']);rid=current['state']['stage_run'];sid=item['session_id']
    report=svc.research.get_report(sid,rid,result['delivery']['report_id'])
    claim=report['claim_snapshots'][0]['claim']
    svc.research.revise_claim(sid,rid,claim['id'],expected_revision=claim['revision'],
                             text='Changed upstream claim',kind='proposal',evidence_ids=claim['evidence_ids'])
    before=len(svc.research.events(sid,rid))
    with pytest.raises(ValueError):
        exports.export_task(f,item['fingerprint'],tmp_path/'stale-report')
    assert len(svc.research.events(sid,rid))==before
    assert not (tmp_path/'stale-report').exists()


def test_cli_passes_only_explicit_host_configuration(tmp_path,monkeypatch,capsys):
    cli,_,_=modules()
    calls=[]
    svc=AgentService(tmp_path/'state',model=port())
    def factory(root,**kwargs):
        calls.append((root,kwargs));return svc
    monkeypatch.setattr(cli,'AgentService',factory)
    monkeypatch.setattr(svc,'close',lambda:None)
    assert cli.main(['status'])==0
    assert calls[-1][1]=={}
    assert cli.main(['--env-file','.env','--project-root',str(tmp_path),
                     '--dataset-dir','data/gene_diff','status'])==0
    assert calls[-1][1]=={'env_file':'.env','project_root':str(tmp_path),'dataset_dir':'data/gene_diff'}
    capsys.readouterr()
