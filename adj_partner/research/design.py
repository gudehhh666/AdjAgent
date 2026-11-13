"""Declared research decisions with checked references, never scientific certification."""
from copy import deepcopy
from .validation import bounded_json, ids, text, strings


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('research_design object requires: ' + ', '.join(keys))


def validate_design(value, claim_ids):
    _object(value, ('requirements','candidates','selected_candidate','mechanism','verification','blocking_gaps'))
    bounded_json(value)
    def linked(item, fields):
        _object(item, (*fields, 'claim_ids'))
        for field in fields:
            text(item[field], 'research_design ' + field, 4000)
        ids(item['claim_ids'])
        if not set(item['claim_ids']) <= set(claim_ids):
            raise ValueError('research_design references claims outside report')
    for key in ('requirements', 'candidates'):
        if not isinstance(value[key], list) or not 1 <= len(value[key]) <= 30:
            raise ValueError('research_design ' + key + ' must have 1..30 entries')
    for item in value['requirements']:
        linked(item, ('requirement','answer','status'))
        if item['status'] not in ('addressed','open'):
            raise ValueError('Requirement status must be addressed or open')
    for item in value['candidates']:
        linked(item, ('name','rationale'))
    names=[x['name'] for x in value['candidates']]
    if len(set(names))!=len(names):
        raise ValueError('Candidate names must be unique')
    if not isinstance(value['selected_candidate'],str) or (value['selected_candidate'] and value['selected_candidate'] not in names):
        raise ValueError('Selected candidate must name a listed candidate or be empty')
    linked(value['mechanism'], ('text',))
    verification=value['verification']
    if isinstance(verification,dict) and 'checkpoints' in verification:
        from .checkpoints import validate_checkpoints
        linked({k:v for k,v in verification.items() if k not in ('checkpoints','mechanism_graph','literature_edges')}, ('text',))
        validate_checkpoints(verification['checkpoints'], verification['claim_ids'])
        if 'mechanism_graph' in verification:
            from .mechanism_graph import validate_mechanism_graph
            validate_mechanism_graph(verification['mechanism_graph'], verification['checkpoints'], verification['claim_ids'])
        if 'literature_edges' in verification:
            if 'mechanism_graph' not in verification:
                raise ValueError('literature_edges require a mechanism_graph')
            from .mechanism_graph import validate_literature_edges
            validate_literature_edges(verification['literature_edges'], verification['mechanism_graph'])
    else:
        linked(verification, ('text',))
    strings(value['blocking_gaps'], 'blocking_gaps', maximum=30, item_max=4000)
    return deepcopy(value)


def design_status(value):
    """Documentary completeness only; authored status is not independently proved."""
    if value is None:
        return 'not_recorded'
    linked=[*value['requirements'], *value['candidates'], value['mechanism'], value['verification']]
    complete=(value['selected_candidate'] and len(value['candidates'])>=2
              and not value['blocking_gaps'] and all(x['status']=='addressed' for x in value['requirements'])
              and all(x['claim_ids'] for x in linked))
    return 'documented_design' if complete else 'incomplete_design'


def render_design(value, claim_numbers=None, *, execution_note=False):
    from .rendering import authored
    def refs(item):
        if claim_numbers is None:
            return authored(', '.join(item['claim_ids']) or '无')
        return ' '.join(f"[C{claim_numbers[key]}](report.md#c{claim_numbers[key]})" for key in item['claim_ids']) or '无'
    lines=['## 研究决策记录','', '记录完整性：' + design_status(value) + '。仅检查声明和引用完整性，不是科学质量认证。','',
           '主候选：' + authored(value['selected_candidate'] or '尚未选定'), '']
    for item in value['requirements']:
        lines.extend(['- ' + authored(item['requirement']) + ' [' + item['status'] + ']：' + authored(item['answer']),
                      '  绑定论断：' + refs(item)])
    lines.extend(['','### 候选比较',''])
    for item in value['candidates']:
        lines.extend(['- ' + authored(item['name']) + '：' + authored(item['rationale']),
                      '  绑定论断：' + refs(item)])
    for key,label in [('mechanism','机制依据'),('verification','验证逻辑')]:
        item=value[key];lines.extend(['','### '+label,'',authored(item['text']),
                                     '绑定论断：'+refs(item)])
    if 'checkpoints' in value['verification']:
        from .checkpoints import render_checkpoints
        lines.extend(render_checkpoints(value['verification']['checkpoints'], refs, execution_note=execution_note))
        if 'mechanism_graph' in value['verification']:
            from .mechanism_graph import render_mechanism_graph
            lines.extend(render_mechanism_graph(value['verification']['mechanism_graph'], value['verification']['checkpoints'], refs,
                                                value['verification'].get('literature_edges')))
    lines.extend(['','### 阻塞缺口',''])
    lines.extend('- '+authored(x) for x in value['blocking_gaps'])
    if not value['blocking_gaps']:lines.append('- 作者未声明阻塞缺口；不代表没有未知。')
    return '\n'.join(lines)+'\n\n'


def readable_report(report):
    """Derived view links the immutable audit; must not be used as a silent eval replacement."""
    from .rendering_review import render_review_report
    import re
    content=render_review_report(report).decode()
    body=content.split('## 论断与证据审计\n',1)[0]
    body=re.sub(r'\]\(#c(\d+)\)',r'](report.md#c\1)',body)
    sources=content.split('## Sources\n',1)[1]
    heading='> 派生可读视图；[完整审计](report.md)。原始SHA256：'+report['artifact_hash']+'\n\n'
    if report.get('research_design'):
        heading+='> 研究记录状态：'+design_status(report['research_design'])+'；非科学认证。\n\n'
        numbers={x['claim']['id']:i for i,x in enumerate(report['claim_snapshots'],1)}
        body+=render_design(report['research_design'], numbers, execution_note=report.get('renderer_version',0)>=5)
    return (heading+body+'## Sources\n'+sources).encode()


def tool_schema(id_schema):
    def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    txt={'type':'string','minLength':1,'maxLength':4000}
    refs={'type':'array','items':id_schema,'maxItems':100,'uniqueItems':True}
    req=obj({'requirement':txt,'answer':txt,'status':{'enum':['addressed','open']},'claim_ids':refs})
    candidate=obj({'name':txt,'rationale':txt,'claim_ids':refs})
    linked=obj({'text':txt,'claim_ids':refs})
    from .checkpoints import checkpoint_schema
    verification=obj({'text':txt,'claim_ids':refs})
    verification['properties']['checkpoints']=checkpoint_schema(id_schema)
    from .mechanism_graph import graph_schema
    verification['properties']['mechanism_graph']=graph_schema(id_schema)
    return obj({'requirements':{'type':'array','minItems':1,'maxItems':30,'items':req},
                'candidates':{'type':'array','minItems':1,'maxItems':30,'items':candidate},
                'selected_candidate':{'type':'string','maxLength':4000},'mechanism':linked,'verification':verification,
                'blocking_gaps':{'type':'array','maxItems':30,'items':txt}})


def design_evidence_map(report):
    """Expose actual field-to-claim-to-source links for review; no relevance inference."""
    from .assessments import qualifications
    snapshots={x['claim']['id']:x for x in report['claim_snapshots']}
    d=report['research_design'];items=[]
    fields=[*(('requirement', x['requirement'], x) for x in d['requirements']),
            *(('candidate', x['name'], x) for x in d['candidates']),
            ('mechanism','机制',d['mechanism']),('verification','验证',d['verification'])]
    fields.extend(('checkpoint', c['id'], c) for c in d['verification'].get('checkpoints', []))
    for kind,label,item in fields:
        ss=[snapshots[cid] for cid in item['claim_ids']]
        evidence={e['id']:e for s in ss for e in s['evidence']}
        used={e['source_id'] for e in evidence.values()}
        sources={s['id']:s for snap in ss for s in snap['sources'] if s['id'] in used}
        items.append({'kind':kind,'label':label,'claim_ids':item['claim_ids'],
                      'evidence_ids':sorted(evidence),'source_ids':sorted(sources),
                      'source_scope':{sid:qualifications(s) for sid,s in sources.items()}})
    return {'report_sha256':report['artifact_hash'],'research_design_status':design_status(d),
            'scientific_adequacy':'not_assessed','items':items,
            'note':'Only report-bound IDs and recorded material scope. Links do not establish relevance, independence, causal support or correctness.'}
