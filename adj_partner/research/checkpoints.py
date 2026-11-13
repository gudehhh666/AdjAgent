"""Experimental decision plans; structural checks never establish biological support."""
from .validation import text, ids, identifier

TEXT_FIELDS = ('hypothesis', 'current_evidence', 'readout', 'controls', 'quality_rule',
               'supports', 'opposes', 'inconclusive', 'priority_reason')
LEVELS = ('transcript', 'molecular_presence', 'molecular_activity', 'metabolic_link',
          'causal_dependence', 'antigen_function')


def validate_checkpoints(cards, claim_ids):
    if not isinstance(cards, list) or not 3 <= len(cards) <= 5:
        raise ValueError('checkpoints require 3..5 decision cards')
    seen = set()
    for card in cards:
        required=set(TEXT_FIELDS) | {'id', 'branch', 'depends_on', 'claim_ids'}
        if not isinstance(card, dict) or set(card) not in (required, required | {'decision_level'}):
            raise ValueError('checkpoint fields are missing or unknown')
        identifier(card['id'])
        if card['id'] in seen:
            raise ValueError('checkpoint IDs must be unique')
        if card['branch'] not in ('mechanism', 'function'):
            raise ValueError('checkpoint branch must be mechanism or function')
        if 'decision_level' in card:
            if card['decision_level'] not in LEVELS:
                raise ValueError('Unknown checkpoint decision level')
            if (card['decision_level']=='antigen_function') != (card['branch']=='function'):
                raise ValueError('checkpoint decision level does not match branch')
        for key in TEXT_FIELDS:
            text(card[key], 'checkpoint '+key, 4000)
        ids(card['claim_ids']); ids(card['depends_on'])
        if not card['claim_ids'] or not set(card['claim_ids']) <= set(claim_ids):
            raise ValueError('checkpoint claims must be bound in verification')
        # Ordered dependencies make cyclic or nonexistent prerequisite plans impossible.
        if not set(card['depends_on']) <= seen:
            raise ValueError('checkpoint dependencies must refer to earlier cards')
        seen.add(card['id'])
    if {c['branch'] for c in cards} != {'mechanism', 'function'}:
        raise ValueError('checkpoints require separate mechanism and function branches')


def render_checkpoints(cards, refs, *, execution_note=False):
    from .rendering import authored
    heading='### 关键实验判据（实际执行状态见实验结果记录）' if execution_note else '### 关键实验判据（计划，未执行）'
    lines = ['', heading, '']
    labels = dict(hypothesis='要判断的假设与适用条件', current_evidence='当前依据与缺口',
                  readout='关键读出', controls='对照', quality_rule='质量与未判定规则',
                  supports='支持时的判断与下一步', opposes='反对时的判断与下一步',
                  inconclusive='未判定时的下一步', priority_reason='优先级理由')
    for c in cards:
        lines.extend(['#### '+authored(c['id'])+' · '+('机制' if c['branch']=='mechanism' else '功能'), ''])
        if 'decision_level' in c:lines.extend(['所需证据层级：'+authored(c['decision_level']), ''])
        for key, label in labels.items():
            lines.extend([label+'：'+authored(c[key]), ''])
        lines.extend(['依赖：'+authored(', '.join(c['depends_on']) or '无前置检测，可独立推进'),
                      '绑定论断：'+refs(c), ''])
    return lines


def checkpoint_schema(id_schema):
    txt = {'type':'string', 'minLength':1, 'maxLength':4000}
    fields = {k:txt for k in TEXT_FIELDS}
    fields.update(id=id_schema, branch={'enum':['mechanism','function']},
                  depends_on={'type':'array','items':id_schema,'maxItems':4,'uniqueItems':True},
                  claim_ids={'type':'array','items':id_schema,'minItems':1,'maxItems':100,'uniqueItems':True})
    required=list(fields)
    fields['decision_level']={'enum':list(LEVELS)}
    return {'type':'array','minItems':3,'maxItems':5,'items':{
        'type':'object','properties':fields,'required':required,'additionalProperties':False}}
