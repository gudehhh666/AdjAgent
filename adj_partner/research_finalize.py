"""Single explicit, separately budgeted synthesis phase over a case's existing evidence."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
from .application import AgentService
from .application_config import PROJECT_ROOT
from .open_research_case import _fresh_directory, _write, check_finalization_parent
from .tools.catalog import canonical_json
from .workflows import load_workflow
from .research.rendering import render_report
from .research.design import readable_report, design_evidence_map, design_status

FINALIZATION_BUDGET={'max_steps':24,'max_input_tokens':300000,'max_output_tokens':60000,'wall_time_seconds':600.0}
FINALIZATION_TOOLS={'update_plan','artifact_read','document_read','evidence_select','claim_observe','claim_record','report_create','report_validate'}


def finalize_case(parent, output, *, env_file, project_root=PROJECT_ROOT):
    parent=Path(parent).resolve();prior=json.loads((parent/'manifest.json').read_text())
    state_dir=Path(prior.get('state_dir',parent/'state')).resolve()
    svc=AgentService(state_dir,project_root=project_root,env_file=env_file,request_output_tokens=12000)
    dest = None
    rid = None
    m = {}
    try:
        with (state_dir/'.finalization.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            store=svc.store;old=store.get_run(prior['run_id']);sid=old['session_id']
            check_finalization_parent(old,prior)
            runs=store.list_runs(sid)
            if any(r.get('state',{}).get('completion_of')==old['id'] for r in runs):
                raise ValueError('This parent already has its one finalization attempt')
            if any(r['status'] not in {'succeeded','failed','budget_exhausted','cancelled'} for r in runs):
                raise ValueError('Cannot finalize while another session run is active')
            task=store.get_task(old['task_id'])
            if not task['acceptance'].get('require_research_design'):
                raise ValueError('Explicit structured research workflow required')
            dest=_fresh_directory(output);session=store.get_session(sid);config=deepcopy(session['config'])
            for key in ('allowed_tools','enabled_tools'):
                config[key]=[t for t in config[key] if t in FINALIZATION_TOOLS]
            config['instructions']=load_workflow('adjuvant-design-v1')['instructions']+'\nHost已有材料综合阶段：不再检索新文献，不需要重审原文观察。当前任务要求不变。优先使用已选证据及Host原文观察ID，至多补读现有材料中决定性的缺口。完成一个新report_create并校验；正文真正回答原题、比较候选与机制，缺少关键依据则声明open与blocking_gaps。不得补造参数、不得把缺口描述成已解决。'
            store.update_session(sid,config,session['revision'])
            run=store.create_run(sid,old['task_id'],dict(FINALIZATION_BUDGET));rid=run['id']
            store.update_run(rid,{'state':{'schema_version':1,'phase':'ready','steps':0,'elapsed_active_seconds':0.0,'completion_of':old['id']}},expected_revision=run['revision'])
            m={'completion_of':old['id'],'session_id':sid,'run_id':rid,'state_dir':str(state_dir),'budget':FINALIZATION_BUDGET,'status':'initializing'}
            _write(dest/'phase-start.json',(canonical_json(m)+'\n').encode())
            aliases=[]
            for source in svc.research.list_sources(sid,old['id']):
                if source['task_id'] == old['task_id'] and source.get('provenance',{}).get('text_availability') in {'abstract','full_text','derived_full_text'}:
                    alias=svc.research.handoff_source(sid,rid,source['id'])
                    aliases.append({'previous_source_id':source['id'],'current_source_id':alias['id'],'title':source['title']})
            observed=[];skipped=[]
            # These were selected by the agent, not new host scientific choices.
            for e in svc.research.list_evidence(sid,old['id']):
                if e['task_id'] != old['task_id']:
                    continue
                try:
                    c=svc.research.observe_evidence(sid,rid,e['id'])
                    observed.append({'claim_id':c['id'],'evidence_id':e['id'],'source_id':e['source_id']})
                except (ValueError,KeyError) as exc:
                    skipped.append({'evidence_id':e['id'],'error':str(exc)})
            store.append_event(sid,'host.bounded_research_finalization',{'completion_of':old['id'],'budget':FINALIZATION_BUDGET,'observations':observed,'skipped':skipped,'source_handoffs':aliases,'tools':sorted(FINALIZATION_TOOLS)},rid)
            store.append_message(sid,{'role':'assistant','content':'Host已建立下列同题原文观察，仅证明选定材料所述，不证明科学有效。请根据原问题与当前研究索引进行有界综合，不重复检索。真实映射：'+canonical_json(observed)+' 同题现有材料已由Host校验哈希并交接为当前run来源；读取时使用current_source_id：'+canonical_json(aliases)},rid)
            final=svc.drive(rid);delivery=final.get('state',{}).get('delivery',{})
            m.update(status=final['status'],usage=final['usage'],steps=final['state'].get('steps'),error=final['state'].get('error'),observations=observed,skipped=skipped)
            if final['status']=='succeeded':
                a=svc.research.accepted_report(sid,rid,delivery['report_id']);r=a['report'];raw=a['content'];assert raw==render_report(r)
                assert hashlib.sha256(raw).hexdigest()==delivery['sha256']
                _write(dest/'report.md',raw);_write(dest/'report-readable.md',readable_report(r))
                _write(dest/'research-evidence-map.json',(canonical_json(design_evidence_map(r))+'\n').encode())
                m.update(report_id=r['id'],report_sha256=r['artifact_hash'],research_design_status=design_status(r['research_design']))
            _write(dest/'sources.json',(canonical_json({'sources':[s for s in svc.research.list_sources(sid,rid) if s['task_id']==old['task_id']]})+'\n').encode())
            m['trace']=svc.trace(sid,dest/'trace',rid)
            _write(dest/'manifest.json',(canonical_json(m)+'\n').encode())
            return m
    except Exception as exc:
        if rid is not None:
            current=svc.store.get_run(rid)
            error={"code":"finalization_exception","type":type(exc).__name__,"message":str(exc)}
            if current["status"] not in {"succeeded","failed","budget_exhausted","cancelled"}:
                current=svc.cancel(rid)
                svc.store.update_run(rid,{"state":{**current["state"],"phase":"finished","error":error}},expected_revision=current["revision"])
            m.update(status="failed",error=error,run_status=svc.store.get_run(rid)["status"])
            if dest is not None and not (dest/'manifest.json').exists():
                _write(dest/'manifest.json',(canonical_json(svc.store.redact(m))+'\n').encode())
        raise
    finally:svc.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);p.add_argument('--env-file',required=True);a=p.parse_args()
    r=finalize_case(a.parent,a.output,env_file=a.env_file)
    print(json.dumps({k:r.get(k) for k in ('status','research_design_status','run_id','completion_of')},ensure_ascii=False))
    return 0 if r['status']=='succeeded' else 1


if __name__=='__main__':raise SystemExit(main())
