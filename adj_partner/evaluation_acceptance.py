"""Deterministic acceptance inputs and honest summaries, without model or expert scoring."""
import hashlib
from .orchestration.contracts import MODES
from .tools.catalog import canonical_json


def build_manifest(rows, task_type, limit):
    if task_type not in MODES or type(limit) is not int or limit<1:
        raise ValueError('Known task mode and positive limit required')
    items=[];seen=set();questions=set()
    for index,row in enumerate(rows):
        if not isinstance(row,dict):raise ValueError('Input row must be an object')
        question=next((row[k] for k in ('question','Question:') if isinstance(row.get(k),str) and row[k].strip()),None)
        if question is None:raise ValueError('Public question required')
        normalized=' '.join(question.split())
        qhash=hashlib.sha256(normalized.encode()).hexdigest()
        identity=row.get('id',qhash)
        if type(identity) not in (str,int) or (isinstance(identity,str) and not identity.strip()):
            raise ValueError('Task ID must be a nonempty string or integer')
        identity=str(identity)
        if identity in seen or qhash in questions:raise ValueError('Duplicate task identity or question')
        seen.add(identity);questions.add(qhash)
        key=hashlib.sha256(canonical_json({'mode':task_type,'id':identity,'question':normalized}).encode()).hexdigest()
        items.append({'case_id':key,'row_id':identity,'task_type':task_type,'question':question,
                      'split':'holdout' if int(qhash[:8],16)%5==0 else 'development',
                      'expert_review':None,'empirical_validation_status':'not_performed'})
    ordered=sorted(items,key=lambda r:r['case_id'])
    holdout=[r for r in ordered if r['split']=='holdout'][:max(1,limit//5) if limit>1 else 0]
    development=[r for r in ordered if r['split']=='development'][:limit-len(holdout)]
    chosen=holdout+development
    chosen_ids={r['case_id'] for r in chosen}
    chosen.extend([r for r in ordered if r['case_id'] not in chosen_ids][:limit-len(chosen)])
    items=sorted(chosen,key=lambda r:r['case_id'])
    return {'schema_version':1,'task_type':task_type,'selection':'stable hash order; question-hash holdout, not a powered study',
            'items':items,'manifest_sha256':hashlib.sha256(canonical_json(items).encode()).hexdigest()}


def validate_metadata(value):
    if not isinstance(value,dict):return ['metadata must be an object']
    issues=[]
    for key in ('dataset_id','species','tissue','assay','expression_scale','input_sha256'):
        if not isinstance(value.get(key),str) or not value[key].strip():issues.append(f'{key}: missing')
    samples=value.get('samples');contrasts=value.get('contrasts')
    if not isinstance(samples,list) or not samples:return issues+['samples: nonempty list required']
    ids=set();groups={}
    for sample in samples:
        if not isinstance(sample,dict):issues.append('sample: object required');continue
        sid=sample.get('sample_id');condition=sample.get('condition');unit=sample.get('biological_unit')
        if not isinstance(sid,str) or not sid:issues.append('sample_id: missing')
        elif sid in ids:issues.append(f'duplicate sample_id: {sid}')
        else:ids.add(sid)
        if not isinstance(condition,str) or not condition:issues.append('condition: missing');continue
        if not isinstance(unit,str) or not unit:issues.append(f'{sid}: biological replicate identity missing')
        else:groups.setdefault(condition,set()).add(unit)
        if not isinstance(sample.get('batch'),str) or not sample['batch']:issues.append(f'{sid}: batch missing')
    for group,units in groups.items():
        if len(units)<2:issues.append(f'{group}: fewer than two independent biological replicates; exploratory only')
    if not isinstance(contrasts,list) or not contrasts:return issues+['contrasts: nonempty list required']
    for contrast in contrasts:
        if not isinstance(contrast,dict):issues.append('contrast: object required');continue
        for role in ('treatment','control'):
            if contrast.get(role) not in groups:issues.append(f"contrast {contrast.get('name')}: {role} {contrast.get(role)} missing from samples")
        if contrast.get('treatment')==contrast.get('control'):issues.append('contrast treatment and control must differ')
        if type(contrast.get('shared_control')) is not bool:issues.append('contrast shared_control must be explicitly true or false')
    return issues


def summarize_results(results):
    groups={};seen=set()
    for result in results:
        fingerprint=result.get('task_fingerprint');mode=result.get('task_type')
        if not fingerprint or mode not in MODES:raise ValueError('Result requires task identity and known mode')
        if fingerprint in seen:raise ValueError('Duplicate result fingerprint')
        seen.add(fingerprint)
        row=groups.setdefault(mode,{'total':0,'succeeded':0,'waiting':0,'failed':0,'other_execution':0,
            'review_passed':0,'known_requests':0,'request_count_missing':0,'empirical_validated':0,
            'expert_acceptance_rate':None,'quality_improvement':None})
        row['total']+=1
        status=result.get('execution_status')
        row[status if status in ('succeeded','waiting','failed') else 'other_execution']+=1
        row['review_passed']+=int(result.get('review_status')=='passed')
        row['empirical_validated']+=int(result.get('empirical_validation_status')=='validated')
        requests=result.get('usage',{}).get('requests')
        if type(requests) is int and requests>=0:row['known_requests']+=requests
        else:row['request_count_missing']+=1
    return groups
