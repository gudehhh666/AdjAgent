"""Deterministic repair bookkeeping; matching/binding never asserts scientific support."""
from copy import deepcopy
import hashlib
import re
from ..tools.catalog import canonical_json


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _normalized(text):
    return ' '.join(text.casefold().split())


def _action(issue, stage):
    if stage == 'decision':
        return 'revise_candidate_decision'
    if stage == 'report':
        return 'revise_report_with_frozen_decision'
    if re.search(r'GO|KEGG|Reactome|top_n|permutation|敏感性|计算|富集|交集', issue, re.I):
        return 'select_existing_computation_or_compute_missing_result'
    return 'retrieve_exact_support_or_narrow_claim'


def repair_tasks(history):
    if not history:
        return []
    return [deepcopy(t) for t in history[-1]['tasks'] if t['status'] != 'closed_by_stage_review']


def source_priority(source, tasks):
    """Lexical routing hint only. Explicit source IDs rank above term matches."""
    issues=' '.join(t['issue'] for t in tasks if t['status'] != 'closed_by_stage_review')
    if source['id'] in re.findall(r'source_[A-Za-z0-9_-]+', issues):
        return 10000
    text=canonical_json({'title':source.get('title',''),'parameters':source.get('provenance',{}).get('parameters',{})}).casefold()
    terms=set(re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{1,80}', issues.casefold()))
    terms-={'the','for','and','need','evidence','source','with','from','this','that','more','missing'}
    return sum(len(t) for t in terms if t in text)


def update_feedback(history, verdict, stage, round_no, selected, usage, *, design=None):
    """Append one replay-idempotent snapshot; only a same-stage pass closes issues."""
    result=deepcopy(history)
    key=f'{stage}:{round_no}'
    # Request recovery may return the same review after a saved checkpoint.
    existing=next((s for s in result if s['review_key']==key),None)
    if existing:
        if existing['verdict'] != verdict['verdict'] or existing['issues'] != verdict['issues']:
            raise ValueError('Recovered feedback changed for the same review key')
        return result
    tasks=deepcopy(result[-1]['tasks']) if result else []
    target=verdict['return_to'] if verdict['verdict']=='revise' else stage
    current_ids=[]
    for issue in verdict['issues']:
        identifier='repair_'+_digest({'stage':target,'issue':_normalized(issue)})[:20]
        current_ids.append(identifier)
        found=next((t for t in tasks if t['id']==identifier),None)
        if found:
            found.update(status='open',last_seen_round=round_no,occurrences=found['occurrences']+1)
        else:
            tasks.append({'id':identifier,'issue':issue,'target_stage':target,'action':_action(issue,target),
                'status':'open','first_seen_round':round_no,'last_seen_round':round_no,'occurrences':1,
                'completion_rule':'Only a passing review at the target stage closes this issue; evidence matching is not validation.'})
    for task in tasks:
        if verdict['verdict']=='pass' and task['target_stage']==stage:
            task['status']='closed_by_stage_review'
        elif verdict['verdict']=='revise' and task['target_stage']==target and task['id'] not in current_ids and task['status']=='open':
            task['status']='not_repeated_unverified'
    # Ignore aliases, generated IDs and source receipt timestamps when checking progress.
    evidence_hashes=sorted(set(_digest({'scope':v.get('scope'),'text':v.get('text'),
        'parameters':v.get('analysis_parameters')}) for v in selected.values()))
    design_hash=_digest(design) if design is not None else None
    signature=_digest({'stage':stage,'target':target,'issues':sorted(current_ids),'evidence':evidence_hashes,'design':design_hash})
    prior=next((s for s in reversed(result) if s['stage']==stage),None)
    streak=(prior['no_progress_streak']+1 if prior and prior['signature']==signature else 1) if verdict['verdict']=='revise' else 0
    result.append({'review_key':key,'stage':stage,'round':round_no,'verdict':verdict['verdict'],
        'issues':deepcopy(verdict['issues']),'tasks':tasks,'signature':signature,'no_progress_streak':streak,
        'selected_evidence_count':len(selected),'evidence_hashes':evidence_hashes,
        'design_hash':design_hash,'design_changed':bool(prior and prior.get('design_hash')!=design_hash),
        'new_evidence_count':len(set(evidence_hashes)-set(prior['evidence_hashes'] if prior else [])),
        'request_count':usage.get('requests'),'closed_issue_count':sum(t['status']=='closed_by_stage_review' for t in tasks)})
    return result


def claim_coverage(claims, selected):
    present={v['id'] for v in selected.values()}
    rows=[]
    for claim in claims:
        ids=claim.get('evidence_ids',[])
        missing=sorted(set(ids)-present)
        rows.append({'claim_id':claim['id'],'kind':claim.get('kind'),'evidence_ids':ids,
                     'missing_evidence_ids':missing,'status':'missing_binding' if missing or not ids else 'bound_unverified'})
    return {'schema_version':1,'scientific_support_assessed':False,'claims':rows,
            'binding_complete':all(r['status']=='bound_unverified' for r in rows) if rows else None}
