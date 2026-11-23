import pytest
from adj_partner import open_research_case


def test_finalize_is_bounded_and_never_retries_policy_or_arbitrary_failures():
    assert hasattr(open_research_case,'check_finalization_parent')
    check=open_research_case.check_finalization_parent
    for status in ['running','failed','cancelled','succeeded']:
        with pytest.raises(ValueError):check({'status':status,'state':{}}, {})
    with pytest.raises(ValueError):check({'status':'budget_exhausted','state':{}},{'completion_of':'prior'})
    with pytest.raises(ValueError):check({'status':'budget_exhausted','state':{'error':{'code':'provider_http'}}},{})
    check({'status':'budget_exhausted','state':{'error':{'code':'input_budget'}}},{})


def test_real_finalization_preserves_parent_and_is_single_use(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from adj_partner.application import AgentService
    from adj_partner.research_finalize import finalize_case
    from test_harness_research_records import source
    from test_research_design import design
    parent=tmp_path/'parent';parent.mkdir();svc=AgentService(parent/'state',project_root=Path.cwd())
    sid=svc.create_session(profile='adjuvant',allow_network=False)['id']
    from adj_partner.runtime.journal import RunJournal
    from adj_partner.open_research_case import OPEN_RESEARCH_BUDGET
    old=RunJournal(svc.store).start(sid,'Synthetic research fixture',{'require_research_design':True,'require_literal_observations':True},OPEN_RESEARCH_BUDGET)
    s,_=source(svc.store,svc.research,sid,old['id'],content='A literal fixture.')
    svc.research.select_evidence(sid,old['id'],s['id'],{'passage':'A literal fixture.'})
    svc.store.update_run(old['id'],{'status':'budget_exhausted','state':{'error':{'code':'context_budget'}}})
    original=svc.store.get_run(old['id'])
    foreign=svc.store.create_task(sid,'Unrelated task')['id']
    foreign_run=svc.store.create_run(sid,foreign,OPEN_RESEARCH_BUDGET)['id']
    fs,_=source(svc.store,svc.research,sid,foreign_run,content='Unrelated evidence.',key='foreign')
    svc.research.select_evidence(sid,foreign_run,fs['id'],{'passage':'Unrelated evidence.'})
    svc.cancel(foreign_run)
    svc.close()
    (parent/'manifest.json').write_text(json.dumps({'run_id':old['id'],'session_id':sid,'status':'budget_exhausted'}))
    def drive(self,rid):
        run=self.store.get_run(rid);c=self.research.list_claims(sid,rid)[-1]
        assert c['origin']=='host_observation'
        assert all(x['task_id']==old['task_id'] for x in self.research.list_claims(sid,rid))
        config=self.store.get_session(sid)['config'];assert 'literature_search' not in config['enabled_tools']
        r=self.research.create_report(sid,rid,title='Fixture',sections=[{'heading':'Decision','claim_ids':[c['id']]}],limitations=['Synthetic only.'],research_design=design(c['id']))
        assert self.research.validate_report(sid,rid,r['id'])['accepted']
        self.store.update_run(rid,{'status':'succeeded','state':{**run['state'],'delivery':{'report_id':r['id'],'sha256':r['artifact_hash']}}})
        return self.store.get_run(rid)
    monkeypatch.setattr(AgentService,'drive',drive)
    result=finalize_case(parent,tmp_path/'final',env_file=None,project_root=Path.cwd())
    assert result['status']=='succeeded'
    assert (tmp_path/'final/report-readable.md').exists()
    with pytest.raises(ValueError,match='one finalization'):finalize_case(parent,tmp_path/'duplicate',env_file=None,project_root=Path.cwd())
    reopened=AgentService(parent/'state',project_root=Path.cwd())
    try:assert reopened.store.get_run(old['id'])==original
    finally:reopened.close()


def test_finalization_exception_leaves_terminal_run_and_diagnostic(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from adj_partner.application import AgentService
    from adj_partner.research_finalize import finalize_case
    from adj_partner.runtime.journal import RunJournal
    from adj_partner.open_research_case import OPEN_RESEARCH_BUDGET
    parent=tmp_path/'parent';parent.mkdir()
    svc=AgentService(parent/'state',project_root=Path.cwd())
    sid=svc.create_session(profile='adjuvant',allow_network=False)['id']
    old=RunJournal(svc.store).start(sid,'Failure fixture',{'require_research_design':True},OPEN_RESEARCH_BUDGET)
    svc.store.update_run(old['id'],{'status':'budget_exhausted','state':{'error':{'code':'context_budget'}}})
    svc.close()
    (parent/'manifest.json').write_text(json.dumps({'run_id':old['id'],'session_id':sid}))
    def fail(self,rid):raise RuntimeError('Injected connection failure')
    monkeypatch.setattr(AgentService,'drive',fail)
    out=tmp_path/'final'
    with pytest.raises(RuntimeError,match='Injected'):finalize_case(parent,out,env_file=None,project_root=Path.cwd())
    diagnostic=json.loads((out/'manifest.json').read_text())
    assert diagnostic['error']['code']=='finalization_exception'
    svc=AgentService(parent/'state',project_root=Path.cwd())
    try:
        run=svc.store.get_run(diagnostic['run_id'])
        assert run['status']=='cancelled'
        assert svc.store.get_task(run['task_id'])['status']=='cancelled'
    finally:svc.close()
