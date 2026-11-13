"""Bounded opt-in navigation to authorized files and installed pack records.

Cards contain identities and schemas, never record field values. Matching is
lexical over public names/aliases rather than task IDs, answers or private rubrics.
Installed pack tools have their own host authorization; exposing those tools
does not grant the model filesystem access to the pack directory.
"""
import json
import re

from .research_entry import initial_resource_cards
from .resources.usage import USAGE
from .tools.catalog import canonical_json

NAVIGATION_BOUNDARY = (
    'Navigation metadata only; names, field lists and hashes are not read evidence. '
    'Enable the authorized reader if needed, then execute open. Verify entity and '
    'setting in returned fields. Omitted fields remain unread; use the focused '
    'reader or state the gap. No new filesystem or network permissions.')
_STOP = set('a an and are as at be by can compare data do does for from how in is it '
            'of on or read source study studies the this to what which with'.split())
_CONTENT_FIELDS = ('annotations','definition','text','equation','Description','description','scope')


def _tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.casefold())) - _STOP


def _request_text(goal):
    """Extract the request, not attachment values or envelope bookkeeping.

    A trailing JSON task envelope has one explicit question. Plain text uses the
    prefix before source/data blocks. This is conservative navigation filtering,
    not instruction parsing or a security boundary; the solver still receives
    the complete original input through its normal context.
    """
    decoder=json.JSONDecoder(); end=len(goal.rstrip())
    for match in re.finditer(r'\{',goal):
        try:
            value, stop=decoder.raw_decode(goal,match.start())
        except ValueError:
            continue
        if stop==end and isinstance(value,dict) and 'question' in value:
            goal=value['question'] if isinstance(value['question'],str) else ''
            break
    boundary=re.search(r'(?im)^\s*(?:```|~~~|(?:input(?: data|_text)?|source(?: material| text| content)?|data|输入(?:数据|材料)?|资料|附件)\s*[:：])',goal)
    if boundary:
        goal=goal[:boundary.start()]
    # Paths mention files to inspect, not external record identities. In
    # particular inputs/VO_1.csv must not trigger the VO_1 identifier exception.
    goal=re.sub(r'https?://\S+',' ',goal)
    goal=re.sub(r'`([^`]+)`',lambda m:' ' if '/' in m[1] or re.search(r'\.[A-Za-z0-9]{1,8}$',m[1]) else m[1],goal)
    goal=re.sub(r'(?<![\w:])(?:[\w.~/-]+/)?[\w.-]+\.[A-Za-z]{1,8}(?!\w)',' ',goal)
    return goal.strip()


def _literal_match(request, name):
    name=name.casefold()
    # Avoid compiling/scanning a regex for every name in the installed pack.
    return name in request and re.search(r'(?<![a-z0-9_])'+re.escape(name)+r'(?![a-z0-9_])',request) is not None


def _identity_match(request, record):
    rid=record['id']
    names=[rid]
    suffix=rid.rsplit(':',1)[-1]
    if len(suffix)>=4 and any(c.isalpha() for c in suffix) and any(c.isdigit() for c in suffix):
        names.append(suffix)
    return any(any(c.isalpha() for c in name) and _literal_match(request,name) for name in names)


def _match(goal, record):
    names=[record['id'], record['title'], *record.get('aliases', [])]
    query=_tokens(goal)
    overlap=query & _tokens(' '.join(names))
    exact=[]
    for name in names:
        if not _tokens(name):
            continue
        if _literal_match(goal.casefold(),name):
            exact.append(len(_tokens(name)))
    if exact:
        return 100 + max(exact) + len(overlap)
    # Incidental word overlap is appropriate for an explicit search request,
    # not for an unsolicited pinned record that consumes the initial context.
    return 0


def _available(tool, config):
    allowed=set(config.get('allowed_tools', []))
    return tool in allowed and (tool in config.get('enabled_tools', []) or 'tools_enable' in allowed)


def _pack_cards(service, goal, config, *, has_task_inputs=False):
    pack=getattr(getattr(service,'native',None),'resource_pack',None)
    if pack is None or not _available('adjuvant_read_fields',config):
        return []
    requested=[r for r in pack.records.values() if _identity_match(goal.casefold(),r)]
    if has_task_inputs and not requested:
        return []
    ranked=[]
    # An explicit accession defines the candidate set. Never fill spare slots
    # with another named entity or a weakly related record from the same dataset.
    for record in requested or pack.records.values():
        score=_match(goal,record)
        if score and record['fields']:
            ranked.append((-score,record['id'],record))
    cards=[]
    for _, rid, record in sorted(ranked,key=lambda item:item[:2])[:4]:
        # Names and container types describe the schema; never rank on or emit
        # source values, nested values, or measurements in a navigation card.
        fields=list(record['fields'])
        composites={name:'object' if isinstance(value,dict) else 'array'
                    for name,value in record['fields'].items() if isinstance(value,(dict,list))}
        matching=sorted((f for f in fields if _tokens(f) & _tokens(goal)),
                        key=lambda f:(-len(_tokens(f)&_tokens(goal)),f))
        selected=list(dict.fromkeys(matching+[f for f in _CONTENT_FIELDS if f in fields]+sorted(fields)))[:3]
        selected=[f for f in selected if len(f)<=256]
        if not selected:
            continue
        card={'kind':'pack_record','record_id':rid,'dataset':record['dataset'],
              'title':record['title'][:120],'pack_sha256':pack.manifest_hash,
              'field_names':fields,'composite_fields':composites,
              'open':{'tool':'adjuvant_read_fields','arguments':{'record_id':rid,'fields':selected}}}
        if 'adjuvant_read_fields' not in config.get('enabled_tools',[]):
            card['enable']={'tool':'tools_enable','arguments':{'name':'adjuvant_read_fields'}}
        if _available('adjuvant_cite',config):
            composite=next((field for field in matching+selected if field in composites),None)
            card['long_field_reader']={'tool':'adjuvant_cite','arguments':{
                'record_id':rid,'field':composite or selected[0],'length':600}}
            card['read_hint']='Use original field_names only. Enable adjuvant_cite for omitted fields; follow truncated ranges, or add focus for a specific passage.'
        elif record['dataset'] in USAGE and _available('adjuvant_lookup',config):
            card['long_field_reader']={'tool':'adjuvant_lookup','arguments':{
                'dataset':record['dataset'],'query':rid,'fields':selected,'limit':1}}
            card['read_hint']='For omitted long fields enable adjuvant_lookup, add a short task-relevant focus, and read its literal fragments.'
        cards.append(card)
    return cards


def initial_navigation_cards(service, goal, config, *, max_cards=2, max_bytes=2200, has_task_inputs=False):
    """Return a bounded metadata directory without changing session configuration.

    Existing registry files are admitted only after their hash and allowed-root
    checks. The installed pack is exposed only via an authorized actual reader.
    Passing a nonexistent or unauthorized path in the task never grants access.
    Caller should attach NAVIGATION_BOUNDARY to the projected directory and pass
    has_task_inputs when named files were bound. Such tasks need only file cards
    unless the request explicitly identifies a pack record to read.
    """
    if not isinstance(goal,str):
        raise ValueError('goal must be text')
    if type(has_task_inputs) is not bool:
        raise ValueError('has_task_inputs must be boolean')
    if type(max_cards) is not int or not 0<=max_cards<=4 or type(max_bytes) is not int or not 0<=max_bytes<=5000:
        raise ValueError('Invalid navigation card budget')
    request=_request_text(goal)
    files=[] if has_task_inputs else [{'kind':'authorized_file',**card}
           for card in initial_resource_cards(service,request,config)]
    for card in files:
        reader=card['tools'][0]
        if reader=='file_inspect':
            card['open']={'tool':reader,'arguments':{'path':card['path']}}
        else:
            card['read_hint']='Pass this regular file path in code_interpreter inputs; read its staged inputs/<basename> file before computing.'
    packs=_pack_cards(service,request,config,has_task_inputs=has_task_inputs)
    # Preserve both entry kinds when both exist, within the same visible budget.
    candidates=files[:1]+packs[:1]+files[1:]+packs[1:]
    result=[]
    for card in candidates:
        if len(result)>=max_cards:
            break
        if len(canonical_json(result+[card]).encode())<=max_bytes:
            result.append(card)
    return result
