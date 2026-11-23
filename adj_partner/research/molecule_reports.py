"""Single-molecule decision plans; dose numbers are host-owned evidence fields."""
from copy import deepcopy
import html
import math
import re
from urllib.parse import urlsplit

from .validation import bounded_json, identifier, strings, text
from .rendering import authored

VERSION = 'molecule-reports-v2'
DECISIONS = {'priority', 'exploratory', 'defer', 'comparison'}
LABELS = {'priority': '优先验证', 'exploratory': '探索验证', 'defer': '暂缓推进', 'comparison': '比较材料'}
STAGES = {'exposure', 'molecular', 'function', 'mechanism'}
STAGE_LABELS = {'exposure': '条件与暴露', 'molecular': '分子复现', 'function': '抗原功能', 'mechanism': '机制依赖'}


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + ' requires exactly: ' + ', '.join(keys))


def _numbers(values):
    return (isinstance(values, list) and len(values) <= 12
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values))


def validate_dose(profile):
    dose = profile['dose_strategy']
    _exact(dose, ('status','unit','pilot_series','anchor','reference_doses','limitations'), 'dose')
    if dose['status'] not in {'literature_anchored_pilot','requires_original_concentration','identity_blocked'}:
        raise ValueError('Unknown dose status')
    text(dose['anchor'], 'dose anchor', 2000)
    strings(dose['limitations'], 'dose limitations', maximum=12, item_max=2000)
    if not _numbers(dose['pilot_series']): raise ValueError('Invalid dose series')
    references = dose['reference_doses']
    if not isinstance(references, list) or len(references) > 12: raise ValueError('Invalid dose references')
    seen = set()
    for ref in references:
        fields=('id','url','title','scope','compound','concentrations','unit','context','locator','transfer_limit')
        if not isinstance(ref,dict) or set(ref) not in (set(fields),set(fields)|{'material'}):
            raise ValueError('Invalid dose reference fields')
        if 'material' in ref: text(ref['material'],'dose reference material',20000)
        identifier(ref['id'])
        if ref['id'] in seen or ref['id'] == 'local': raise ValueError('Duplicate dose source')
        seen.add(ref['id'])
        if ref['compound'] != profile['compound']: raise ValueError('Cross-molecule dose reference')
        for key in ('title','context','locator','transfer_limit'): text(ref[key], 'dose '+key, 4000)
        parsed = urlsplit(ref['url'])
        if parsed.scheme not in {'https','http'} or not parsed.netloc or any(c.isspace() for c in ref['url']):
            raise ValueError('Invalid dose URL')
        if ref['scope'] not in {'full_text','derived_full_text','abstract'}: raise ValueError('Dose source must have readable material')
        if not _numbers(ref['concentrations']) or not ref['concentrations']: raise ValueError('Invalid reference dose')
        if ref['unit'] not in {'nM','uM','mM','ug/mL'}: raise ValueError('Unsupported reference dose unit')
    if dose['pilot_series']:
        if dose['status'] != 'literature_anchored_pilot' or not references or dose['unit'] not in {'nM','uM','mM','ug/mL'}:
            raise ValueError('Numerical dose proposal requires traceable reference')
        if any(ref['unit'] != dose['unit'] for ref in references): raise ValueError('Dose units must be explicitly reconciled')
    elif dose['status'] == 'literature_anchored_pilot':
        raise ValueError('Literature dose strategy needs a pilot series')
    elif dose['unit'] is not None: raise ValueError('Unknown dose requires null unit')
    return dose


def validate_plan(value, profile):
    _exact(value, ('alias','compound','decision','summary','mechanism_hypothesis','dose_rationale','focus_gene_ids','data_observations','experiments','limitations','source_ids'), 'molecule plan')
    bounded_json(value)
    if value['alias'] != profile['alias'] or value['compound'] != profile['compound']:
        raise ValueError('Molecule identity mismatch')
    identifier(value['alias'])
    dose = validate_dose(profile)
    if value['decision'] not in DECISIONS: raise ValueError('Invalid molecule decision')
    for key in ('summary','mechanism_hypothesis','dose_rationale'): text(value[key], key, 4000)
    strings(value['focus_gene_ids'],'focus genes',maximum=8,item_max=128,unique=True)
    available=[g for g in profile['genes'] if profile['compound']=='Betaine' or g['gene_name']!='Bhmt']
    if len(value['focus_gene_ids']) < min(3,len(available)) or not set(value['focus_gene_ids']) <= {g['gene_id'] for g in available}:
        raise ValueError('Focus genes must select actual molecule input IDs')
    observed=value['data_observations']
    if not isinstance(observed,list) or len(observed)!=len(value['focus_gene_ids']):raise ValueError('Observation coverage mismatch')
    by_id={g['gene_id']:g for g in profile['genes']}
    if {o.get('gene_id') for o in observed if isinstance(o,dict)}!=set(value['focus_gene_ids']):raise ValueError('Observation gene coverage mismatch')
    for observation in observed:
        _exact(observation,('gene_id','direction_vs_vehicle'),'data observation')
        if observation['direction_vs_vehicle']!=direction_vs_vehicle(by_id[observation['gene_id']]):
            raise ValueError('Observation direction contradicts actual vehicle comparison')
    # Numeric doses belong in the immutable host table, including experiment
    # prose and Chinese numeral/unit expressions, not merely dose_rationale.
    narrative=canonical_text(value)
    if profile['compound']!='Betaine' and re.search(r'Bhmt|Betaine|甜菜碱',narrative,re.I):
        raise ValueError('Unrelated molecule must not copy Betaine/BHMT template')
    pattern=r'(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万零点两]+)\s*(?:nM\b|[uµμ]M\b|mM\b|mg\b|[uµμ]g\b|nmol\b|mmol\b|[微毫纳](?:摩尔|克))'
    if re.search(pattern,narrative,re.I):
        raise ValueError('Numeric dose must use the host-owned dose table')
    allowed = {'local', *(r['id'] for r in dose['reference_doses'])}
    def refs(ids):
        strings(ids, 'source ids', maximum=15, item_max=128, unique=True)
        if not ids or not set(ids) <= allowed: raise ValueError('Unknown or missing source reference')
    refs(value['source_ids'])
    if 'local' not in value['source_ids']: raise ValueError('Local molecule source required')
    cards = value['experiments']
    if not isinstance(cards, list) or not 3 <= len(cards) <= 5: raise ValueError('Need 3..5 experiment cards')
    seen = set()
    for card in cards:
        keys=('id','stage','question','groups','readouts','quality_rule','continue_if','stop_if','inconclusive_if','source_ids')
        if not isinstance(card,dict):raise ValueError('Invalid experiment object')
        if card.get('stage')=='function':
            if 'function_design' not in card:raise ValueError('Explicit function design is required')
            _exact(card,(*keys,'function_design'),'function experiment')
            fields=('antigen_positive_group','antigen_negative_group','washout_restimulation','live_cell_output','primary_endpoint')
            _exact(card['function_design'],fields,'function design')
            for key in fields:text(card['function_design'][key], 'function design '+key,2000)
        else:
            if 'function_design' in card:raise ValueError('function design belongs only to the function stage')
            _exact(card,keys,'experiment')
        identifier(card['id'])
        if card['id'] in seen: raise ValueError('Duplicate experiment id')
        seen.add(card['id'])
        if card['stage'] not in STAGES: raise ValueError('Invalid experiment stage')
        for key in ('question','quality_rule','continue_if','stop_if','inconclusive_if'): text(card[key], key, 3000)
        for key in ('groups','readouts'):
            strings(card[key], key, maximum=10, item_max=2000)
            if not card[key]: raise ValueError('Empty experiment '+key)
        refs(card['source_ids'])
        if card['stage']=='function' and 'IFN' not in card['function_design']['primary_endpoint'].upper():
            raise ValueError('Function primary endpoint must name the proposed IFN-gamma assay')
    if not {'exposure','molecular','function'} <= {c['stage'] for c in cards}:
        raise ValueError('Every molecule needs exposure, molecular and function cards')
    strings(value['limitations'], 'limitations', maximum=15, item_max=3000)
    if not value['limitations']: raise ValueError('Evidence limitations required')
    return deepcopy(value)


def canonical_text(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return '\n'.join(canonical_text(v) for v in value)
    if isinstance(value,dict):return '\n'.join(canonical_text(v) for k,v in value.items() if k not in {'alias','compound','source_ids'})
    return ''


def direction_vs_vehicle(gene):
    a=gene['treatment_fpkm'];b=gene['vehicle_fpkm']
    return 'up' if a>b else 'down' if a<b else 'unchanged'


def _fmt(value):
    if value is None: return '未知'
    if isinstance(value, bool): return str(value).lower()
    return f'{value:.6g}' if isinstance(value, (float,int)) else authored(str(value))


def dose_label(profile):
    d = validate_dose(profile)
    return (' / '.join(_fmt(x) for x in d['pilot_series']) + ' ' + d['unit']) if d['pilot_series'] else '先恢复C0并校准'


def render_molecule(profile, plan):
    plan = validate_plan(plan, profile); d = profile['dose_strategy']
    lines = [f"# {authored(profile['compound'])}｜独立研究报告", '',
             '[返回总索引](../INDEX.md)', '', '## 结论与下一步', '',
             '**当前建议：'+LABELS[plan['decision']]+'。** '+authored(plan['summary']), '',
             '## 本分子的转录观察', '', authored(profile['context']), '',
             '下面数值来自冻结输入；FPKM不是独立动物均值，供应商q/flag未被重新解释为可复现显著性。', '',
             '| 基因 | DMSO FPKM | 本分子 FPKM | 原表log2FC | 原表q | 原表flag | 相对DMSO方向 |',
             '|---|---:|---:|---:|---:|---|---|']
    by_id={g['gene_id']:g for g in profile['genes']}
    for gid in plan['focus_gene_ids']:
        gene=by_id[gid]
        direction={'up':'上升','down':'下降','unchanged':'未变化'}[direction_vs_vehicle(gene)]
        lines.append('| '+' | '.join(_fmt(gene[k]) for k in ('gene_name','vehicle_fpkm','treatment_fpkm','vendor_log2fc','vendor_qval','vendor_flag'))+' | '+direction+' |')
    counts=profile['counts']
    count_text=f"全表{counts['genes']:,}个基因；供应商flag为true的有{counts['vendor_flagged']:,}个，其中表达门槛合格的有{counts['eligible_flagged']:,}个。"
    if 'common_core' in counts:count_text+=f"共同核心为{counts['common_core']:,}个基因。"
    lines += ['', count_text+'面板相对值只描述内部比较，不能读作相对DMSO的下降。', '',
              '## 机制假说与竞争解释', '', authored(plan['mechanism_hypothesis']), '',
              '## 剂量建议', '', '**拟议体外先导浓度：'+authored(dose_label(profile))+'。**', '',
              authored(d['anchor']), '', authored(plan['dose_rationale']), '',
              '原始测序浓度、暴露时间和最终vehicle浓度仍待原始记录确认。此建议不是历史实验剂量或已验证有效剂量，不从体外浓度换算动物/人体剂量。', '']
    for ref in d['reference_doses']:
        lines += [f"- [{ref['id']}：{authored(ref['title'])}]({ref['url']})；已读范围：{ref['scope']}。",
                  '  文献浓度：'+ ' / '.join(_fmt(x) for x in ref['concentrations'])+' '+ref['unit']+'；'+authored(ref['context']),
                  '  定位：'+authored(ref['locator'])+'；迁移限制：'+authored(ref['transfer_limit'])]
    if not d['reference_doses']: lines += ['- 当前没有已核实且可作为绝对浓度锚点的文献；使用原实验C0校准策略。']
    lines += ['', *('- '+authored(x) for x in d['limitations']), '', '## 实验推荐', '']
    for card in plan['experiments']:
        lines += [f"### {card['id']}｜{STAGE_LABELS[card['stage']]}：{authored(card['question'])}", '',
                  '**比较分组：** '+ '；'.join(authored(x) for x in card['groups']), '',
                  '**读出：** '+ '；'.join(authored(x) for x in card['readouts']), '',
                  '**可判定条件：** '+authored(card['quality_rule']), '',
                  '**继续：** '+authored(card['continue_if']), '', '**停止／转向：** '+authored(card['stop_if']), '',
                  '**不可判定：** '+authored(card['inconclusive_if']), '',
                  '设计依据：'+', '.join(card['source_ids'])+'。以上均为计划，未执行。', '']
        if card['stage']=='function':
            fd=card['function_design']
            lines += ['**抗原阳性比较：** '+authored(fd['antigen_positive_group']),'',
                      '**抗原阴性比较：** '+authored(fd['antigen_negative_group']),'',
                      '**洗脱与再刺激：** '+authored(fd['washout_restimulation']),'',
                      '**活细胞产出：** '+authored(fd['live_cell_output']),'',
                      '**主要终点：** '+authored(fd['primary_endpoint']),'']
    lines += ['## 来源与未解决事项', '',
              '本地来源：'+authored(profile['source']['source_name'])+'；SHA-256：'+profile['source']['source_sha256']+'。', '',
              '完整数值、稳定基因ID与来源记录见 [本分子输入](../inputs/'+profile['alias']+'.json)。', '',
              *('- '+authored(x) for x in plan['limitations']), '']
    return '\n'.join(lines).encode()


def render_molecule_html(profile, plan):
    # Disable raw HTML: all authored text remains escaped and links are checked
    # by markdown-it's normal URL validation. Tables remain actual readable tables.
    from markdown_it import MarkdownIt
    md = render_molecule(profile, plan).decode()
    reader=MarkdownIt('commonmark', {'html':False}).enable('table')
    return _html(profile['compound'], '<a href="../index.html">← 总索引</a> · <a href="'+profile['alias']+'.md">Markdown原文</a>'+reader.render(md))


def _html(title, body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>'+html.escape(title)+'</title><style>body{font:17px/1.8 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#25313a;background:#fafbf9}'
            'h1{font-size:30px}h2{font-size:23px;margin-top:36px}h3{font-size:19px}a{color:#246859}table{border-collapse:collapse;width:100%;font-size:15px}'
            'td,th{padding:10px;border-bottom:1px solid #dce5df;text-align:left}.data{font:13px/1.8 monospace;white-space:pre;overflow:auto}p{overflow-wrap:anywhere}</style>'
            '<main>'+body+'</main></html>').encode()


def render_index(profiles, plans):
    aliases=[p['alias'] for p in profiles]
    if len(aliases) != len(set(aliases)) or set(aliases) != set(plans): raise ValueError('Molecule coverage mismatch')
    lines=['# 小分子独立报告总索引', '', '每个分子一份完整文件；全部为探索分析与未执行的实验设计。', '',
           '| 分子 | 当前建议 | 体外剂量策略 | 下一步 |','|---|---|---|---|']
    rows=[]
    for p in profiles:
        a=validate_plan(plans[p['alias']],p); alias=p['alias']
        lines.append(f"| [{authored(p['compound'])}](molecules/{alias}.md) | {LABELS[a['decision']]} | {authored(dose_label(p))} | {authored(a['experiments'][0]['question'])} |")
        rows.append('<tr><td><a href="molecules/'+alias+'.html">'+html.escape(p['compound'])+'</a></td><td>'+LABELS[a['decision']]+'</td><td>'+html.escape(dose_label(p))+'</td><td>'+html.escape(a['experiments'][0]['question'])+'</td></tr>')
    body='<h1>小分子独立研究报告</h1><p>'+str(len(profiles))+'个分子各自阅读。剂量为有条件的体外先导建议；实验均未执行。</p><p><a href="DELIVERY.md">交付说明</a> · <a href="INDEX.md">Markdown索引</a></p><table><thead><tr><th>分子</th><th>当前建议</th><th>体外剂量策略</th><th>下一步</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    return ('\n'.join(lines)+'\n').encode(), _html('小分子独立报告',body)


def plan_schema(profile):
    return {'alias':profile['alias'],'compound':profile['compound'],'decision':'priority/exploratory/defer/comparison',
            'summary':'短段落，当前结论与下一步；只讨论本分子',
            'mechanism_hypothesis':'本地观察、未验证机制、相反解释，区分C→M和M→F',
            'dose_rationale':'解释宿主剂量策略的迁移条件，不写数值剂量',
            'focus_gene_ids':['选择本分子输入中最影响决策的3–8个实际gene_id，必须同时保留关键反证'],
            'data_observations':[{'gene_id':'对应每个focus_gene_id','direction_vs_vehicle':'up/down/unchanged；必须匹配两列FPKM'}],
            'experiments':[{'id':'E1','stage':'exposure/molecular/function/mechanism','question':'明确的决策问题',
                            'groups':['本分子和匹配对照，明确抗原/细胞条件'],'readouts':['具体指标'],
                            'quality_rule':'独立单位、测量有效性和混杂排查',
                            'continue_if':'什么结果允许继续及其范围','stop_if':'什么结果停止或修改机制',
                            'inconclusive_if':'什么结果无信息及补救','source_ids':['local']}],
            'limitations':['具体缺口'],'source_ids':['local']}


def plan_json_schema(profile):
    """Explicit JSON schema avoids slash-separated enum placeholders in output."""
    def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    txt={'type':'string','minLength':1,'maxLength':3000}
    texts={'type':'array','items':txt,'minItems':1,'maxItems':10}
    source_ids={'type':'array','items':{'type':'string','enum':['local',*(r['id'] for r in profile['dose_strategy']['reference_doses'])]},'minItems':1,'uniqueItems':True}
    base={'id':txt,'stage':{'type':'string','enum':['exposure','molecular','mechanism']},'question':txt,
          'groups':texts,'readouts':texts,'quality_rule':txt,'continue_if':txt,'stop_if':txt,'inconclusive_if':txt,'source_ids':source_ids}
    functional={**base,'stage':{'const':'function'},'function_design':obj({k:txt for k in
        ('antigen_positive_group','antigen_negative_group','washout_restimulation','live_cell_output','primary_endpoint')})}
    gene_ids=[g['gene_id'] for g in profile['genes'] if profile['compound']=='Betaine' or g['gene_name']!='Bhmt']
    return obj({'alias':{'const':profile['alias']},'compound':{'const':profile['compound']},
                'decision':{'type':'string','enum':['priority','exploratory','defer','comparison']},
                'summary':txt,'mechanism_hypothesis':txt,'dose_rationale':txt,
                'focus_gene_ids':{'type':'array','items':{'type':'string','enum':gene_ids},'minItems':min(3,len(gene_ids)),'maxItems':8,'uniqueItems':True},
                'data_observations':{'type':'array','minItems':min(3,len(gene_ids)),'maxItems':8,
                    'items':obj({'gene_id':{'type':'string','enum':gene_ids},'direction_vs_vehicle':{'type':'string','enum':['up','down','unchanged']}})},
                'experiments':{'type':'array','minItems':3,'maxItems':5,'items':{'oneOf':[obj(base),obj(functional)]}},
                'limitations':texts,'source_ids':source_ids})
