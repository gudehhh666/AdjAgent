"""Persistent three-mode controller over AgentService and the shared leased runner."""
from copy import deepcopy
import hashlib
import json
import re
import time
from pathlib import Path

from .contracts import TaskSpec, result_record, implementation_hash
from .identity import discovery_input_hash, provider_identity
from .repository import WorkflowRepository
from .feedback import update_feedback, repair_tasks, source_priority, claim_coverage
from . import graph_grounding_runtime, qa_baseline_patch_runtime
from . import registry, resource_handoff
from ..contracts import Budget
from ..runtime.model_invocations import InvocationLedger, leased_run, inherit_guard
from ..storage import TERMINAL_STATUSES
from ..tools.catalog import canonical_json
from ..research.qa_review_context import select_evidence
from ..research.staged_design import bind_claims, abstract_parts
from ..research.design import design_status, readable_report, design_evidence_map
from ..research import graph_seed
from ..task_modes import qa, qa_question_first, design, discovery_design
from ..workflows import load_workflow


class AgentFramework:
    def __init__(self,service,*,allow_frozen=False):
        self.service=service;self.store=service.store;self.allow_frozen=allow_frozen
        self.repository=WorkflowRepository(self.store);self.invocations=InvocationLedger(self.store)

    def submit(self,spec):
        spec=spec if isinstance(spec,TaskSpec) else TaskSpec.from_dict(spec)
        workflow=registry.resolve(spec)
        resource_profile=workflow.resource_profile or None
        resource_aware=resource_profile is not None
        resource_manifest=None
        if resource_aware:
            resource_manifest,identity=resource_handoff.resource_manifest(self.service,workflow=spec.workflow_version,profile=resource_profile)
            if spec.resource_manifest_sha256 and spec.resource_manifest_sha256!=identity:
                raise ValueError('Resource runtime changed; create a new versioned task')
            spec=TaskSpec.from_dict({**spec.to_dict(),'resource_manifest_sha256':identity})
        if spec.task_type=='discovery_design':
            actual=discovery_input_hash(self.service.native.dataset_dir)
            if spec.dataset_sha256 and spec.dataset_sha256!=actual:raise ValueError('Discovery dataset identity mismatch')
            spec=TaskSpec.from_dict({**spec.to_dict(),'dataset_sha256':actual})
        provider=provider_identity(self.service)
        if spec.provider_sha256 and provider and spec.provider_sha256!=provider:raise ValueError('Configured provider identity differs from TaskSpec')
        if provider:spec=TaskSpec.from_dict({**spec.to_dict(),'provider_sha256':provider})
        found=self.repository.find(spec.fingerprint)
        if found:return found
        registry.resolve(spec,allow_frozen=self.allow_frozen)
        instructions=qa.INSTRUCTIONS if spec.task_type=='qa' else design.RETRIEVAL
        question_first=spec.task_type=='qa' and workflow.session=='question_first'
        if question_first:instructions=qa_question_first.INSTRUCTIONS
        experimental=spec.task_type!='qa' and workflow.session=='experimental'
        if experimental:instructions+='\n'+design.EXPERIMENT_GUIDANCE
        limits=dict(spec.budget)
        limits['max_steps']=min(spec.parameters.get('retrieval_steps',32),max(1,limits['max_steps']//2))
        for k in ('max_input_tokens','max_output_tokens'):limits[k]=max(1,limits[k]//2)
        limits['wall_time_seconds']=max(.1,limits['wall_time_seconds']/2)
        allowed_tools=(sorted(qa_question_first.RETRIEVAL_TOOLS & self.service._benign_tools) if question_first else
                       sorted(set(self.service._benign_tools)-{'research_critic'}) if experimental else None)
        sid=self.service.create_session(profile=resource_profile if resource_aware else 'adjuvant',instructions=instructions,allowed_roots=spec.allowed_roots,
                                        allow_network=spec.allow_network,budget=limits,
                                        allowed_tools=allowed_tools)['id']
        if resource_aware and spec.task_type=='discovery_design':
            # Existing prespecified host discovery, not model-selected delegation.
            for name in ('deg_analyze','deg_explore'):self.service.enable_tool(sid,name)
        self.service.set_model(sid,spec.model)
        self.service.catalog.prepare_store(self.store)
        run=self.service.journal.start(sid,spec.question,{},limits)
        if resource_manifest is not None:
            self.store.put_artifact(canonical_json(resource_manifest),'resource-runtime-manifest.json','application/json',
                {'kind':'resource_runtime_manifest','sha256':spec.resource_manifest_sha256},session_id=sid,run_id=run['id'])
        bound=self.repository.bind(spec,sid,run['task_id'],run['id'])
        if bound['initial_run_id']!=run['id']:self.service.cancel(run['id'])
        return bound

    def _usage(self,state):
        total=dict(input_tokens=0,output_tokens=0,accounted_input_tokens=0,accounted_output_tokens=0,unknown_requests=0,requests=0)
        roles={}
        for rid in state['attempts']:
            r=self.store.get_run(rid);aux=self.invocations.usage(rid)
            for k in total:total[k]+=r['usage'].get(k,0)+aux.get(k,0)
            total['requests']+=r['state'].get('steps',0)
            roles[rid]={'runtime':r['usage'],'auxiliary':aux}
        return {**total,'by_attempt':roles}

    def _remaining(self,spec,state,*,admission=False):
        used=self._usage(state)
        budget={**spec.budget,'max_steps':spec.budget['max_steps']-used['requests'],
                'max_input_tokens':spec.budget['max_input_tokens']-used['accounted_input_tokens'],
                'max_output_tokens':spec.budget['max_output_tokens']-used['accounted_output_tokens'],
                'wall_time_seconds':spec.budget['wall_time_seconds']-(time.time()-state['started_at'])}
        # A reservation may consume the final step/token allowance before the
        # provider is called. Boundary guards and receipt recovery permit zero;
        # only new-request admission requires strictly positive remaining units.
        exhausted=any(v<0 for v in budget.values()) or budget['wall_time_seconds']<=0
        if exhausted or (admission and any(v<=0 for v in budget.values())):
            raise RuntimeError('Workflow cumulative budget exhausted')
        return Budget(**budget).to_dict() if admission else budget

    def _stage_run(self,item,spec,state,guard):
        if state.get('stage_run'):return state['stage_run']
        budget=self._remaining(spec,state,admission=True)
        # Model changes only at the completed-run boundary, never inside a lease.
        critic=spec.task_type=='qa' and registry.resolve(spec).stage_model=='critic'
        selected=spec.critic_model or spec.model if critic else spec.model
        self.service.set_model(item['session_id'],selected)
        return self.repository.create_attempt(item['fingerprint'],state,'stage_run',budget,guard=guard)

    def _call(self,item,state,guard,messages,key,role='synthesis',maximum=6000,json_response=True):
        rid=state['stage_run'];port=self.service.models.port(item['session_id'])
        if item['spec']['task_type']!='qa' and registry.WORKFLOWS[item['spec']['workflow_version']].resource_profile:
            messages=deepcopy(messages)
            messages[0]['content']+='\n'+resource_handoff.STAGE_GUIDANCE
        self.invocations.checkpoint(rid,state['phase'],state,guard=guard)
        reply=self.invocations.complete(item['session_id'],rid,port,messages,[],role=role,key=key,
                                       max_output_tokens=maximum,timeout_seconds=180,guard=guard)
        if reply.refusal or reply.finish_reason not in {'stop','end_turn'} or reply.tool_calls:
            raise ValueError('Refused or incomplete model response')
        return json.loads(reply.content) if json_response else reply.content

    def _review(self,item,spec,**kwargs):
        sid=item['session_id'];rid=kwargs.pop('rid')
        if not spec.critic_model or spec.critic_model==spec.model:
            return self.service.critic.review(sid,rid,**kwargs)
        if self.service.models.injected is not None:
            raise ValueError('A fixed injected port cannot provide a different critic model')
        from ..models import OpenAICompatiblePort
        from ..research.critic import ResearchCritic
        port=OpenAICompatiblePort.from_environment(project_root=self.service.models.project_root,
            env_file=self.service.models.env_file,model=spec.critic_model)
        try:
            return ResearchCritic(self.service.research,lambda _:port).review(sid,rid,**kwargs)
        finally:port.close()

    def _documents(self,sid,rid):
        values=[]
        for source in self.service.research.list_sources(sid,rid):
            if source['task_id']!=self.store.get_run(rid)['task_id']:continue
            raw=self.service.research._verify_source(source)
            try:document=json.loads(raw) if source['media_type']=='application/json' else {'text':raw.decode()}
            except (ValueError,UnicodeError):continue
            values.append({**source,'document':document})
        return values

    def _materialize(self,sid,rid,*,latest_run_id=None,baseline_source_id=None,primary_source_ids=(),evidence_budget_bytes=90000,repair_items=(),receipt_run_ids=None):
        selected={};observations={};r=self.service.research
        read_selectors={}
        if receipt_run_ids is not None:
            read_selectors,audit=resource_handoff.read_selections(self.service,sid,rid,receipt_run_ids)
            primary_source_ids=tuple(set(primary_source_ids)|set(read_selectors))
            self.store.append_event(sid,'workflow.resource_handoff',{'receipts':audit,'scientific_support_assessed':False},rid)
        packet_bytes=0;included_sources=set();included_computations=set()
        sources=list(r.list_sources(sid,rid))
        mentioned=set(re.findall(r'(?:source|evidence)_[A-Za-z0-9_-]+',' '.join(t['issue'] for t in repair_items)))
        exact_evidence=[e for e in r.list_evidence(sid,rid) if e['id'] in mentioned]
        required_sources={e['source_id'] for e in exact_evidence}|{s['id'] for s in sources if s['id'] in mentioned}
        primary_source_ids=tuple(set(primary_source_ids)|required_sources)
        # Fresh readable evidence must not be starved by an earlier 30-item prefix.
        scope_rank={'computed':2,'full_text':4,'derived_full_text':4,'abstract':3}
        sources.sort(key=lambda s:(6 if s['id']==baseline_source_id else 5 if s['id'] in primary_source_ids else scope_rank.get(s['provenance'].get('text_availability'),0),s['run_id']==latest_run_id),reverse=True)
        # Keep the fixed analysis first, then share the bounded packet between
        # computed follow-ups and literature. Either family alone can fill it.
        primary=[s for s in sources if s['id']==baseline_source_id or s['id'] in primary_source_ids]
        remaining=[s for s in sources if s not in primary]
        relevant=sorted([s for s in remaining if source_priority(s,repair_items)>0],key=lambda s:source_priority(s,repair_items),reverse=True)
        remaining=[s for s in remaining if s not in relevant]
        computed=[s for s in remaining if s['kind']=='computed']
        literature=[s for s in remaining if s['kind']!='computed']
        sources=primary+relevant
        for i in range(max(len(computed),len(literature))):
            if i<len(computed):sources.append(computed[i])
            if i<len(literature):sources.append(literature[i])
        for source in sources:
            if source['task_id']!=self.store.get_run(rid)['task_id']:continue
            if source['provenance'].get('projection')=='exact abstract text' and source['id'] not in required_sources:continue
            raw=r._verify_source(source)
            try:doc=json.loads(raw) if source['media_type']=='application/json' else None
            except ValueError:continue
            selectors=[]
            computation_key=None
            pinned=[{k:v for k,v in e['locator'].items() if k in {'pointer','passage'}} for e in exact_evidence if e['source_id']==source['id'] and e['source_hash']==source['artifact_hash']]
            if source['id'] in read_selectors:
                selectors=read_selectors[source['id']]
            elif pinned:
                selectors=pinned
            elif source['id'] in required_sources and isinstance(doc,dict) and isinstance(doc.get('abstract'),str) and doc['abstract'].strip():
                selectors=[{'pointer':'/abstract'}]
            elif source['kind']=='computed' and isinstance(doc,dict) and 'summary' in doc:
                computation_key=canonical_json({'title':source['title'],'parameters':source['provenance'].get('parameters',{}),'summary':doc['summary']})
                if computation_key in included_computations:continue
                selectors=([{'pointer':'/summary/top_centered'},{'pointer':'/summary/controls'}]
                    if source['id']!=baseline_source_id and 'top_n' in source['provenance'].get('parameters',{}) and 'top_centered' in doc['summary'] else [{'pointer':'/summary'}])
            elif isinstance(doc,dict) and isinstance(doc.get('abstract'),str) and doc['abstract'].strip():
                abstract=doc['abstract'];provenance={**source['provenance'],'parent_source_id':source['id'],'text_availability':'abstract','projection':'exact abstract text'}
                a=self.store.put_artifact(abstract,'workflow-abstract.txt','text/plain',provenance,session_id=sid,run_id=rid)
                source=r.register_source(sid,rid,artifact_id=a['id'],kind='literature',title=source['title'],provenance=provenance,idempotency_key='workflow-abstract:'+rid+':'+source['id'])
                selectors=[{'passage':p} for p in abstract_parts(abstract)]
            elif isinstance(doc,dict):
                if source['provenance'].get('text_availability')=='database_record' and 'record' in doc:selectors=[{'pointer':'/record'}]
                elif doc.get('sections'):selectors=[{'pointer':'/sections'}]
                elif doc.get('abstract'):selectors=[{'pointer':'/abstract'}]
                elif doc.get('snippet'):selectors=[{'pointer':'/snippet'}]
            elif source['media_type'].startswith('text/'):
                selectors=[{'passage':raw.decode()[:8000]}]
            for selector in selectors:
                e=next((e for e in r.list_evidence(sid,rid) if e['source_id']==source['id'] and e['source_hash']==source['artifact_hash'] and all(e['locator'].get(k)==v for k,v in selector.items())),None)
                e=e or r.select_evidence(sid,rid,source['id'],selector)
                # The actual selection is immutable and critic input remains bounded.
                evidence_bytes=len(canonical_json(e).encode())
                if evidence_bytes>18000:
                    if receipt_run_ids is not None:self.store.append_event(sid,'workflow.resource_handoff.omitted',{'source_id':source['id'],'selector':selector,'reason':'single_evidence_budget'},rid)
                    continue
                source_bytes=0 if source['id'] in included_sources else len(canonical_json({'title':source['title'],'provenance':source['provenance']}).encode())
                needed=evidence_bytes+source_bytes+512
                if packet_bytes+needed>evidence_budget_bytes:
                    if receipt_run_ids is not None:self.store.append_event(sid,'workflow.resource_handoff.omitted',{'source_id':source['id'],'selector':selector,'reason':'packet_budget'},rid)
                    continue
                if any(v['id']==e['id'] for v in selected.values()):continue
                packet_bytes+=needed;included_sources.add(source['id'])
                if computation_key is not None:included_computations.add(computation_key)
                c=next((c for c in r.list_claims(sid,rid) if c.get('origin')=='host_observation' and c['evidence_ids']==[e['id']]),None)
                c=c or r.observe_evidence(sid,rid,e['id']);key='E'+str(len(selected)+1)
                selected[key]={'id':e['id'],'source_id':source['id'],'source_sha256':source['artifact_hash'],'title':source['title'],'scope':source['provenance'].get('text_availability','unknown'),'text':e.get('text') if e.get('text') is not None else e.get('value')}
                if source['kind']=='computed':
                    params=source['provenance'].get('parameters',{})
                    role='primary_screen' if source['id']==baseline_source_id else 'sensitivity_screen' if 'top_n' in params else 'candidate_exploration'
                    selected[key].update(source_id=source['id'],source_sha256=source['artifact_hash'],analysis_parameters=params,analysis_role=role)
                observations['O'+str(len(observations)+1)]={'id':c['id'],'text':c['text'],'evidence_keys':[key]}
                if len(selected)>=30:return selected,observations
        return selected,observations

    def _finish(self,item,spec,state,execution,delivery=None,review='not_run',error=None,guard=None,run_guard=None):
        if spec.task_type!='qa' and not delivery and error and error.get('code') in {'revision_limit','no_progress'}:
            gaps=state.get('open_gaps') or ['Evidence or design review remains unresolved.']
            lines=['# Design review: unresolved evidence and no-go decision','',
                'Status: incomplete_design. This is a host-generated gap report, not an accepted design or experimental validation.','',
                '## Requested goal','',spec.question,'','## Unresolved review findings','']
            lines.extend('- '+str(gap) for gap in gaps)
            lines.extend(['','## Outcome','','Do not treat the proposed design as accepted. Resolve the recorded evidence gaps before advancing. The configured revision limit has been reached; automatic retries are stopped.'])
            content='\n'.join(lines)+'\n'
            artifact=self.store.put_artifact(content,'design-gaps.md','text/markdown',
                {'workflow':spec.fingerprint,'kind':'incomplete_design','open_gaps':gaps},
                session_id=item['session_id'],run_id=state['stage_run'])
            delivery={'kind':'incomplete_report','content':content,'artifact_id':artifact['id'],
                'sha256':artifact['sha256'],'research_design_status':'incomplete_design'}
        result=result_record(spec,execution,delivery,review_status=review,attempt_refs=state['attempts'],
                             usage=self._usage(state),open_gaps=state.get('open_gaps',[]),error=error)
        result['feedback_history']=state.get('feedback_history',[])
        result['claim_coverage']=state.get('claim_coverage',claim_coverage([],{}))
        result['selection_audit']=state.get('selection_audit',{})
        if spec.task_type!='qa' and state.get('stage_run'):
            result['lineage']=self.service.research.design_lineage(item['session_id'],state['stage_run'])
        if execution=='succeeded':
            state['phase']='done'
            current=self.store.get_run(state['stage_run'])
            self.invocations.finish(state['stage_run'],status='succeeded',state={**current['state'],'phase':'finished','delivery':result['delivery']},guard=run_guard,
                on_commit=lambda conn:self.repository.save_in_transaction(conn,item['fingerprint'],state,result,guard=guard))
        else:self.repository.save(item['fingerprint'],state,guard=guard,result=result)
        self.store.append_event(item['session_id'],'workflow.result',result,state.get('stage_run') or state['retrieval_run'])
        return result

    def _more_evidence(self,item,spec,state,owned):
        # Resume a distinct retrieval attempt under the same task identity.
        sid=item['session_id']
        if not state.get('extra_retrieval'):
            budget=self._remaining(spec,state,admission=True)
            budget['max_steps']=min(spec.parameters.get('retrieval_steps',32),max(1,budget['max_steps']//2))
            for key in ('max_input_tokens','max_output_tokens'):budget[key]=max(1,budget[key]//2)
            budget['wall_time_seconds']=max(.1,budget['wall_time_seconds']/2)
            task=self.store.get_task(item['task_id'])
            self.store.update_task(task['id'],{'acceptance':{}},expected_revision=task['revision'])
            message={'role':'user','content':canonical_json({'question':spec.question,'evidence_feedback':state['feedback'],'repair_tasks':repair_tasks(state.get('feedback_history',[])),'primary_analysis':state.get('discovery',{}).get('baseline'),'instruction':'Resolve each repair task explicitly. First reuse exact existing source/evidence IDs when available; otherwise retrieve or compute missing support. Report task IDs addressed and remaining gaps. Matching an evidence ID does not establish scientific support. Return a bounded evidence brief.'})}
            self.repository.create_attempt(item['fingerprint'],state,'extra_retrieval',budget,guard=owned,message=message)
        run=self.service.drive(state['extra_retrieval'])
        if run['status']!='succeeded':
            return self._finish(item,spec,state,run['status'],review='revision_required',error=run['state'].get('error'),guard=owned)
        state['draft']=run['state']['delivery']['content'];state['retrieval_run']=run['id']
        state.pop('extra_retrieval',None);state.pop('selected',None);state['phase']='evidence'
        self.repository.save(item['fingerprint'],state,guard=owned)
        return None

    def _cached(self,item,spec,state,owned):
        result=item['result'];delivery=result['delivery'];sid=item['session_id'];rid=state['stage_run']
        if spec.task_type=='qa':
            raw=self.store.read_artifact(delivery['artifact_id'])
            if hashlib.sha256(raw).hexdigest()!=delivery['sha256'] or raw.decode()!=delivery['content']:
                raise ValueError('Cached answer artifact integrity mismatch')
        else:
            if self.service.research.validate_design_record(sid,rid,state['design_id']):
                raise ValueError('Cached design bindings became stale')
            accepted=self.service.research.accepted_report(sid,rid,delivery['report_id'])
            if hashlib.sha256(accepted['content']).hexdigest()!=delivery['sha256']:
                raise ValueError('Cached report artifact integrity mismatch')
        # A crash after delivery commit and before the run terminal update is local-only recovery.
        current=self.store.get_run(rid)
        if current['status'] not in TERMINAL_STATUSES:
            with leased_run(self.store,rid) as execution:
                self.invocations.finish(rid,status='succeeded',state={**current['state'],'phase':'finished','delivery':delivery},guard=execution)
        return result

    def run(self,fingerprint):
        with self.repository.owned(fingerprint) as owned:
            item=self.repository.get(fingerprint);spec=TaskSpec.from_dict(item['spec']);state=item['state']
            sid=item['session_id'];workflow=registry.resolve(spec)
            if workflow.resource_profile:
                _,identity=resource_handoff.resource_manifest(self.service,workflow=spec.workflow_version,
                    profile=workflow.resource_profile)
                if identity!=spec.resource_manifest_sha256:
                    raise ValueError('Resource runtime changed; create a new versioned task')
            if not state.get('started_at'):state['started_at']=time.time();self.repository.save(fingerprint,state,guard=owned)
            if state['phase']=='done':
                return self._cached(item,spec,state,owned)
            if state['phase'] in registry.done_phases():
                return qa_baseline_patch_runtime.cached(self,item)
            if state['phase'] in {'revision_limit','no_progress'}:
                return item['result']
            if spec.implementation_sha256!=implementation_hash():raise ValueError('Implementation changed; create a new versioned task instead of reusing this execution')
            if spec.task_type=='discovery_design' and spec.dataset_sha256!=discovery_input_hash(self.service.native.dataset_dir):raise ValueError('Discovery dataset identity changed')
            provider=provider_identity(self.service)
            bound_provider=spec.provider_sha256 or state.get('provider_sha256')
            if bound_provider and bound_provider!=provider:raise ValueError('Configured provider changed; create a new versioned execution')
            if provider and not state.get('provider_sha256'):
                state['provider_sha256']=provider;self.repository.save(fingerprint,state,guard=owned)
            if workflow.delegated:
                return workflow.delegated.run(self,item,spec,state,owned)
            try:
                while True:
                    self._remaining(spec,state)
                    if state['phase']=='discovery':
                        with leased_run(self.store,state['retrieval_run']) as execution:
                            def guard():owned();execution();self._remaining(spec,state)
                            inherit_guard(guard,execution)
                            state['discovery']=discovery_design.discover(self.service,sid,state['retrieval_run'],spec,guard)
                            self.store.append_message(sid,{'role':'assistant','content':'Actual host-computed evidence (exploratory, not validated benefit): '+canonical_json(state['discovery'])},state['retrieval_run'])
                        state['phase']='retrieval';self.repository.save(fingerprint,state,guard=owned)
                    if state['phase']=='research_more':
                        result=self._more_evidence(item,spec,state,owned)
                        if result is not None:return result
                    if state['phase']=='retrieval':
                        initial=self.service.drive(state['retrieval_run'])
                        if initial['status']!='succeeded':
                            return self._finish(item,spec,state,initial['status'],error=initial['state'].get('error'),guard=owned)
                        state['draft']=initial['state']['delivery']['content'];state['phase']='review' if spec.task_type=='qa' else 'evidence'
                        if spec.task_type=='qa' and workflow.session=='question_first':state['phase']='answer'
                        self.repository.save(fingerprint,state,guard=owned)
                    rid=self._stage_run(item,spec,state,owned)
                    with leased_run(self.store,rid) as execution:
                        def guard():owned();execution();self._remaining(spec,state)
                        inherit_guard(guard,execution)
                        # Subtract other attempts from this run's envelope after upstream retrieval.
                        current=self.store.get_run(rid);remaining=self._remaining(spec,state)
                        own=self.invocations.usage(rid)
                        budget={**current['budget'],'max_steps':remaining['max_steps']+own['requests'],
                            'max_input_tokens':remaining['max_input_tokens']+own['accounted_input_tokens'],
                            'max_output_tokens':remaining['max_output_tokens']+own['accounted_output_tokens']}
                        self.store.update_run(rid,{'budget':budget},expected_revision=current['revision'])
                        if spec.task_type=='qa':
                            question_first=workflow.session=='question_first'
                            if question_first:
                                packet=state.get('answer_packet')
                                if packet is None:
                                    packet=qa_question_first.evidence_packet(spec.question,self._documents(sid,state['retrieval_run']))
                                    state['answer_packet']=packet;self.repository.save(fingerprint,state,guard=owned)
                                value=self._call(item,state,guard,qa_question_first.answer_messages(spec.question,packet),'qa-question-first-answer',maximum=2400,json_response=False)
                                answer=qa_question_first.parse_answer(value,packet)
                            else:
                                packet=state.get('review_packet')
                                if packet is None:
                                    packet=select_evidence(spec.question,state['draft'],self._documents(sid,state['retrieval_run']))
                                    state['review_packet']=packet;self.repository.save(fingerprint,state,guard=owned)
                                value=self._call(item,state,guard,qa.review_prompt(spec.question,state['draft'],packet),'qa-review',role='critic')
                                answer=qa.parse_answer(value,packet)
                            content=answer['answer']
                            a=self.store.put_artifact(content,'answer.md','text/markdown',{'workflow':spec.fingerprint,'review':answer},session_id=sid,run_id=rid)
                            delivery={'kind':'answer','content':content,'sha256':a['sha256'],'artifact_id':a['id'],**answer}
                            result=self._finish(item,spec,state,'succeeded',delivery,'not_run' if question_first else 'passed',guard=owned,run_guard=guard)
                        else:
                            result=self._design(item,spec,state,guard,owned)
                        if result is not None:return result
            except Exception as exc:
                code=getattr(exc,'code',type(exc).__name__)
                status='budget_exhausted' if 'budget' in str(exc).lower() else 'failed'
                if state['phase']=='done':state['phase']='delivery_invalid'
                return self._finish(item,spec,state,status,error={'code':code,'message':self.store.redact(str(exc))},guard=owned)

    def _design(self,item,spec,state,guard,owned):
        r=self.service.research;sid=item['session_id'];rid=state['stage_run'];fp=item['fingerprint']
        workflow=registry.resolve(spec)
        resource_aware=bool(workflow.resource_profile)
        experimental=workflow.session=='experimental'
        graph=workflow.graph
        acceptance={**load_workflow('adjuvant-design-v1' if experimental else 'adjuvant-design-critic-v1')['acceptance'],'require_concise_discovery':False}
        if experimental:acceptance['required_sections'][-1]='当前判断与下一步实验'
        task=self.store.get_task(item['task_id'])
        if task['acceptance']!=acceptance:self.store.update_task(task['id'],{'acceptance':acceptance},expected_revision=task['revision'])
        if not state.get('selected'):
            repairs=repair_tasks(state.get('feedback_history',[]))
            state['selected'],state['observations']=self._materialize(sid,rid,latest_run_id=state['retrieval_run'],baseline_source_id=state.get('discovery',{}).get('baseline',{}).get('source_id'),primary_source_ids=[r.get_evidence(sid,rid,eid)['source_id'] for eid in state.get('discovery',{}).get('evidence_ids',[])],repair_items=repairs,receipt_run_ids=state['attempts'] if resource_aware else None)
            required=set(re.findall(r'(?:source|evidence)_[A-Za-z0-9_-]+',' '.join(t['issue'] for t in repairs)))
            if state.get('discovery',{}).get('baseline'):required.add(state['discovery']['baseline']['source_id'])
            present={x for v in state['selected'].values() for x in (v['id'],v['source_id'])}
            state['selection_audit']={'explicit_required_ids':sorted(required),'missing_required_ids':sorted(required-present),'selected_count':len(state['selected']),'scope':'Explicit ID coverage only; lexical matching does not verify claims.'}
            if not state['selected']:raise ValueError('No retrievable evidence for conceptual design')
            state['brief_id']=state.get('brief_id') or r.create_design_brief(sid,rid,goal=spec.question,requirements=[spec.question],
                finding_ids=state.get('discovery',{}).get('finding_ids',[]),evidence_ids=[v['id'] for v in state['selected'].values()])['id']
            self.repository.save(fp,state,guard=owned)
        max_rounds=spec.parameters.get('max_revision_rounds',4)
        while state['round']<max_rounds:
            guard();round_no=state['round'];stage=state['phase']
            selected=state['selected'];eids=[v['id'] for v in selected.values()]
            if stage=='evidence':
                if experimental:
                    state['phase']='decision';self.repository.save(fp,state,guard=owned);continue
                prefix=(('Host primary analysis contract: '+canonical_json(state['discovery']['baseline'])+'\n') if state.get('discovery',{}).get('baseline') else '')
                prefix+='Selection audit: '+canonical_json(state['selection_audit'])+'\n'
                verdict=self._review(item,spec,rid=rid,stage='evidence',brief=prefix+state['draft'][:max(0,24000-len(prefix))],evidence_ids=eids,_guard=guard,_request_key=f'workflow-evidence-{round_no}')
                stopped=self._feedback_checkpoint(item,spec,state,verdict,'evidence',owned)
                if stopped is not None:return stopped
                if verdict['verdict']!='pass':
                    state['feedback']=verdict;state['open_gaps']=verdict['issues'];state['round']+=1
                    state['phase']='research_more'
                    self.repository.save(fp,state,guard=owned)
                    if state['round']>=max_rounds:break
                    return None
                state['phase']='decision';self.repository.save(fp,state,guard=owned);continue
            if stage=='decision':
                if workflow.predictive_graph and 'graph_seed' not in state:
                    state['graph_seed']=graph_seed.build(getattr(self.service.native,'resource_pack',None),spec.question)
                    self.repository.save(fp,state,guard=owned)
                prompt=(design.experiment_decision_prompt(spec.question,selected,graph=graph,predictive=workflow.predictive_graph,
                                                          seed=state.get('graph_seed')) if experimental else
                        design.decision_prompt(spec.question,selected,state.get('feedback'),state.get('design')))
                value=self._call(item,state,guard,prompt,f'decision-{round_no}',maximum=16000 if workflow.predictive_graph else 14000 if graph else 10000)
                if experimental and 'checkpoints' not in value.get('research_design',{}).get('verification',{}):
                    raise ValueError('Experimental design requires structured checkpoints')
                if graph and 'mechanism_graph' not in value['research_design']['verification']:
                    raise ValueError('Mechanism-graph design requires verification.mechanism_graph')
                if workflow.predictive_graph != ('predictions' in value['research_design']['verification'].get('mechanism_graph',{})):
                    raise ValueError('Predictive mechanism graphs carry predictions only in the predictive workflow')
                if 'literature_edges' in value['research_design'].get('verification',{}):
                    raise ValueError('literature_edges are host-written, not part of a model decision')
                claims=deepcopy(state['observations'])
                for proposal in value['proposals']:
                    if not isinstance(proposal.get('key'),str) or proposal['key'] in claims:raise ValueError('Duplicate proposal alias')
                    if any(k not in selected for k in proposal['evidence_keys']):raise ValueError('Unknown proposal evidence')
                    proposal_eids=[selected[k]['id'] for k in proposal['evidence_keys']]
                    c=next((c for c in r.list_claims(sid,rid) if c['kind']=='proposal' and c['text']==proposal['text'] and c['evidence_ids']==proposal_eids),None)
                    c=c or r.record_claim(sid,rid,text=proposal['text'],kind='proposal',evidence_ids=proposal_eids)
                    claims[proposal['key']]={'id':c['id'],'text':c['text'],'evidence_keys':proposal['evidence_keys']}
                d=bind_claims(value['research_design'],{k:v['id'] for k,v in claims.items()})
                revision=r.create_design_revision(sid,rid,brief_id=state['brief_id'],design=d,parent_id=state.get('design_id'),feedback_ids=state.get('feedback_ids',[]))
                state.update(design=d,design_id=revision['id'],claims=claims,feedback_ids=[],
                             phase='edge_grounding' if workflow.literature_edges else 'report' if experimental else 'decision_review')
                state['claim_coverage']=claim_coverage([r.get_claim(sid,rid,v['id']) for v in claims.values()],selected)
                self.repository.save(fp,state,guard=owned)
                continue
            if stage=='edge_grounding':
                graph_grounding_runtime.ground(self,item,spec,state,guard,owned)
                continue
            if stage=='decision_review':
                d=state['design']
                verdict=self._review(item,spec,rid=rid,stage='decision',brief='Evaluate source-bounded conceptual design and counterevidence.',evidence_ids=eids,research_design=d,_guard=guard,_request_key=f'workflow-decision-{round_no}')
            elif stage=='report':
                if r.validate_design_record(sid,rid,state['design_id']):raise ValueError('Design evidence bindings became stale')
                claims=[r.get_claim(sid,rid,v['id']) for v in state['claims'].values()]
                value=self._call(item,state,guard,design.report_prompt(spec.question,state['design'],claims,acceptance['required_sections'],state.get('feedback')),f'report-{round_no}',maximum=12000)
                for section in value['sections']:
                    section['claim_ids']=list(dict.fromkeys(cid for p in section.get('paragraphs',[]) for cid in p['claim_ids']))
                report=r.create_report(sid,rid,title=value['title'],sections=value['sections'],limitations=value['limitations'],research_design=state['design'])
                state['report_id']=report['id'];state['phase']='report_validate' if experimental else 'report_review';self.repository.save(fp,state,guard=owned)
                continue
            elif stage=='report_review' or (experimental and stage=='report_validate'):
                report=r.get_report(sid,rid,state['report_id'])
                if not experimental:
                    verdict=self._review(item,spec,rid=rid,stage='report',brief='Review exact report and frozen decision.',report_id=report['id'],_guard=guard,_request_key=f'workflow-report-{round_no}')
                    stopped=self._feedback_checkpoint(item,spec,state,verdict,'report',owned)
                    if stopped is not None:return stopped
                if experimental or verdict['verdict']=='pass':
                    validation=r.validate_report(sid,rid,report['id'])
                    if not validation['accepted']:raise ValueError('Report acceptance: '+', '.join(validation['issues']))
                    accepted=r.accepted_report(sid,rid,report['id'])
                    delivery={'kind':'accepted_report','report_id':report['id'],'content':accepted['content'].decode(),
                              'sha256':report['artifact_hash'],'design_revision_id':state['design_id'],
                              'research_design_status':design_status(state['design']),
                              'readable_content':readable_report(report).decode(),'research_evidence_map':design_evidence_map(report)}
                    if experimental:delivery['experiment_status']='planned'
                    state['open_gaps']=state['design']['blocking_gaps']
                    return self._finish(item,spec,state,'succeeded',delivery,'not_run' if experimental else 'passed',guard=owned,run_guard=guard)
            else:raise ValueError('Unknown persisted design stage')
            if stage=='decision_review':
                stopped=self._feedback_checkpoint(item,spec,state,verdict,'decision',owned)
                if stopped is not None:return stopped
            state['feedback']=verdict
            if verdict['verdict']=='pass':state['phase']='report'
            else:
                state['round']+=1;state['open_gaps']=verdict['issues']
                back=verdict['return_to']
                if back=='evidence':
                    state['phase']='research_more'
                    feedback=r.record_design_feedback(sid,rid,design_id=state['design_id'],target_stage='evidence',issue='; '.join(verdict['issues']),affected_fields=['$'],evidence_ids=eids)
                    state['feedback_ids']=[feedback['id']]
                    self.repository.save(fp,state,guard=owned)
                    if state['round']>=max_rounds:break
                    return None
                state['phase']=back
                if back=='decision':
                    feedback=r.record_design_feedback(sid,rid,design_id=state['design_id'],target_stage='decision',
                        issue='; '.join(verdict['issues']),affected_fields=['$'],evidence_ids=eids)
                    state['feedback_ids']=[feedback['id']]
            self.repository.save(fp,state,guard=owned)
        state['resume_stage']=state['phase']
        state['phase']='revision_limit'
        return self._finish(item,spec,state,'waiting',review='revision_required',error={'code':'revision_limit','message':'Bounded revision limit reached'},guard=owned)

    def _feedback_checkpoint(self,item,spec,state,verdict,stage,owned):
        state['feedback_history']=update_feedback(state.get('feedback_history',[]),verdict,stage,state['round'],state['selected'],self._usage(state),design=state.get('design'))
        self.repository.save(item['fingerprint'],state,guard=owned)
        if state['feedback_history'][-1]['no_progress_streak']>=spec.parameters.get('no_progress_limit',3):
            state.update(phase='no_progress',feedback=verdict,open_gaps=verdict['issues'])
            return self._finish(item,spec,state,'waiting',review='revision_required',error={'code':'no_progress','message':'Same-stage issues and selected evidence repeated without measurable progress'},guard=owned)
        return None
