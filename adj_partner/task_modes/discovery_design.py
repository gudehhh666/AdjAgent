"""Translate actual computed evidence into findings; never manufacture a design."""
from ..contracts import ToolCall
import time


def complete_tool(service,sid,rid,name,arguments,call_id,guard):
    guard()
    result=service.executor.execute(ToolCall(call_id,name,arguments),sid,rid,execution_guard=guard)
    while result.job_id and result.status in {'pending','running','waiting'}:
        guard()
        time.sleep(.1)
        result=service.native.resolve_job(sid,rid,result.job_id)
    if result.status!='ok':raise RuntimeError('Analysis failed: '+str(result.error))
    return result


def discover(service,sid,rid,spec,guard):
    p=spec.parameters
    result=complete_tool(service,sid,rid,'deg_analyze',{'top_n':p.get('top_n',250),'n_permutations':p.get('n_permutations',500)},'workflow_deg_screen',guard)
    evidence=[];findings=[]
    def record(value,description):
        source_id=value.data['source_id']
        e=next((e for e in service.research.list_evidence(sid,rid) if e['source_id']==source_id and e['locator']=={'pointer':'/summary'}),None)
        e=e or service.research.select_evidence(sid,rid,source_id,{'pointer':'/summary'})
        evidence.append(e['id'])
        existing=service.research.list_design_records(sid,rid,kind='finding')
        previous=next((f for f in existing if f['payload']['evidence_ids']==[e['id']]),None)
        f=previous or service.research.create_finding(sid,rid,observation=description,evidence_ids=[e['id']],limitations=value.data['summary'].get('limitations',[]))
        findings.append(f['id'])
    record(result,'Observed panel-centered transcriptomic comparison in the host-reviewed dataset; candidate prioritization is exploratory.')
    candidates=result.data['summary']['top_centered'][:min(5,p.get('max_candidates',2))]
    for index,candidate in enumerate(candidates):
        compound=candidate['compound']
        v=complete_tool(service,sid,rid,'deg_explore',{'compound':compound,'direction':'both','annotation':'KEGG','top_genes':10,'top_terms':8},f'workflow_deg_candidate_{index}',guard)
        record(v,f'Observed gene and pathway summary for {compound}; enrichment does not establish a causal mechanism.')
    source=service.research.get_source(sid,rid,result.data['source_id'])
    baseline={'source_id':source['id'],'source_sha256':source['artifact_hash'],'parameters':source['provenance']['parameters'],
        'role':'primary_screen','interpretation':'This prespecified analysis determines the primary ranking. Other screening parameters are sensitivity analyses and must be compared explicitly, never substituted silently.'}
    return {'finding_ids':findings,'evidence_ids':evidence,'summary':result.data['summary'],'baseline':baseline}
