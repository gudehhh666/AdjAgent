"""Opt-in research entry: bounded resource discovery, not permission or evidence."""
import hashlib
import json
import re
from pathlib import Path
from .contracts import ToolResult
from .profiles import PROFILES, load_trusted_guidance
from .tools import ToolSpec
from .tools.catalog import canonical_json

PROFILE='research-resource-aware'
NEEDS=('all','papers','mechanism','assay','expression','clinical','table','dataset')
LOCAL_NEEDS={'vo':'mechanism','uniprot_adjuvant':'mechanism','rhea_adjuvant':'mechanism','ncl_protocols':'assay','immgen_studies':'expression','viget':'expression','ctgov_adjuvant':'clinical'}
INITIAL_TOOLS=['catalog_search','tools_enable','research_resources','research_skill','update_plan','literature_search','web_search','article_resolve','fulltext_fetch','document_read','artifact_read','job_status']
GUIDANCE='''First identify what the question requires and which unresolved facts could change the conclusion. For an evidence gap, research_resources lists currently permitted local and online entry points by need; use it when useful, not as a mandatory checklist. Enable only the relevant returned tools with tools_enable. Read actual primary text or exact fields before using a resource in a decision. Start document_read with limit=1 so the relevant passage stays visible; when text_truncated or context_truncated is true, read the returned full-result artifact in bounded slices if that missing text is decisive. Never count parser output hidden from context as read evidence. Metadata, annotations, protocols and measurements answer different questions. For an unfamiliar research step, research_skill supplies a short optional method guide; domain-specific adjuvant_skill is conditional. Keep the entity, study, conditions, endpoint and comparator matched. After two distinct unsuccessful attempts at a gap, preserve the limitation and move on. Reserve at least the last third of the budget for synthesis. Compare alternatives and counterevidence; no mandatory review rounds. Finish with an answer to the question, cited observations, decision-relevant uncertainty and what evidence would change the conclusion. Never claim a resource was read merely because it appears in a catalog.'''


def install_profile():
    definition={'instructions':GUIDANCE,'tools':list(INITIAL_TOOLS)}
    if PROFILE in PROFILES and PROFILES[PROFILE]!=definition:raise ValueError('Research entry profile drift')
    PROFILES[PROFILE]=definition


def resource_entries(pack,allowed,network):
    allowed=set(allowed);out=[]
    if pack is not None and {'adjuvant_resource_guide','adjuvant_lookup'}<=allowed:
        for name,meta in sorted(pack.datasets.items()):
            out.append({'id':name,'need':LOCAL_NEEDS.get(name,'table'),'readiness':'installed_snapshot',
                'tools':['adjuvant_resource_guide','adjuvant_lookup'],
                'open':{'tool':'adjuvant_resource_guide','arguments':{'dataset':name}},
                'scope':str(meta.get('evidence_scope','Local records; inspect scope before inference.'))[:230]})
    if network:
        definitions=[('literature','papers','literature_search','Configured Europe PMC/PubMed/Crossref discovery; metadata/available abstracts, not full text.'),
            ('web','papers','web_search','Web discovery snippets; search results are not source text.'),
            ('chembl','mechanism','chembl_compound','Curated compound/assay context, not efficacy in the current setting.'),
            ('reactome','mechanism','reactome_pathway','Curated pathway records by exact stable ID, not current-cell activity.'),
            ('geo','dataset','geo_search','Public study candidates; catalog records are not expression matrices.')]
        for name,need,tool,scope in definitions:
            if tool in allowed:out.append({'id':name,'need':need,'tools':[tool],'readiness':'configured_not_live_verified','scope':scope})
    return out


def resource_page(entries,need='all',offset=0,query=''):
    if need not in NEEDS or type(offset) is not int or offset<0:raise ValueError('Invalid resource query')
    selected=[e for e in entries if need=='all' or e['need']==need or need=='dataset' and e['need'] in ('table','expression')]
    if not isinstance(query,str) or len(query)>1000:raise ValueError('Invalid resource query text')
    if query.strip():
        terms=query.casefold().split()
        selected=[e for e in selected if (e.get('readiness')=='configured_not_live_verified' and e['need']=='papers') or all(t in (e['id']+' '+e['scope']).casefold() for t in terms)]
    page=[]
    for e in selected[offset:offset+3]:
        if len(canonical_json(page+[e]).encode())>1250:break
        page.append(e)
    return {'resources':page,'total_matches':len(selected),'next_offset':offset+len(page) if offset+len(page)<len(selected) else None,'topic_search_executed':False,
        'boundary':'Catalog metadata only, not evidence or new permissions. Enable named tools, then read task-relevant content. A failed request is not a negative result.'}


def file_resources(project_root,items,config):
    roots=[Path(p).resolve() for p in config.get('allowed_roots',[])];out=[]
    for item in items:
        path=Path(project_root)/item['path'];reader=item.get('reader','file_inspect')
        if reader not in ('file_inspect','code_interpreter') or reader not in config['allowed_tools']:continue
        if not path.is_file() or path.is_symlink() or not any(path.resolve().is_relative_to(root) for root in roots):continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:continue
        card={'id':item['id'],'need':item['need'],'readiness':'installed_snapshot','tools':[reader],
            'path':str(path),'sha256':item['sha256'],'scope':item['scope'][:230]}
        if reader=='file_inspect':card['open']={'tool':reader,'arguments':{'path':str(path)}}
        else:card['read_hint']='Use code_interpreter inputs to stage this exact path; read inputs/ with Python gzip/csv/pandas. Emit only relevant fields.'
        out.append(card)
    return out


def initial_resource_cards(service,goal,config):
    """Literal public entity matches only; no task IDs, answers or hidden rubric."""
    registry=service.project_root/'config/research-resource-index.json'
    if not registry.is_file():return []
    entities={x.casefold() for x in re.findall(r'[A-Za-z][A-Za-z0-9-]{2,}',goal)
              if any(c.isdigit() for c in x) or x.isupper()}
    if not entities:return []
    items=json.loads(load_trusted_guidance(registry)).get('resources',[])
    cards=file_resources(service.project_root,items,config)
    candidates=[]
    for card in cards:
        tokens=set(re.findall(r'[a-z][a-z0-9-]{2,}',(card['id']+' '+card['scope']).casefold()))
        if entities & tokens:candidates.append(card)
    candidates.sort(key=lambda x:('metadata' not in x['id']))
    result=[]
    for card in candidates:
        short={k:card[k] for k in ('id','path','tools','sha256')}
        raw=card['scope'].encode();short['scope']=raw[:180].decode('utf-8','ignore')+('...' if len(raw)>180 else '')
        if len(canonical_json(result+[short]).encode())>1150:continue
        result.append(short)
        if len(result)==2:break
    return result


def register_research_entry(service):
    install_profile()
    def resources(context,need='all',offset=0,query=''):
        config=context.store.get_session(context.session_id)['config']
        entries=resource_entries(service.native.resource_pack,config['allowed_tools'],config.get('allow_network',False))
        # Optional explicit host registry: expose only intact files inside authorized roots.
        registry=service.project_root/'config/research-resource-index.json'
        if registry.is_file():
            value=json.loads(load_trusted_guidance(registry))
            entries=file_resources(service.project_root,value.get('resources',[]),config)+entries
        return ToolResult(data={'summary':resource_page(entries,need,offset,query)})
    def skill(context,stage):
        path=service.project_root/'config/skills'/('research-'+stage)/'SKILL.md'
        text=load_trusted_guidance(path);body=text.split('---',2)[2].strip()
        if len(body.encode())>1500:raise ValueError('Research skill exceeds visible budget')
        return ToolResult(data={'summary':{'stage':stage,'guidance':body,'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'boundary':'Host-authored method guide; does not grant tools, permissions or scientific validity.'}})
    def schema(props,required):return {'type':'object','properties':props,'required':required,'additionalProperties':False}
    service.catalog.register(ToolSpec('research_resources','Discover permitted resource entry points, not paper contents. For local records query a short entity/accession such as a gene or study ID; use need=dataset for data and expression for expression studies. Topic queries do not search papers: configured paper-search endpoints remain listed for you to call. No evidence or new permissions.',schema({'need':{'enum':list(NEEDS)},'offset':{'type':'integer','minimum':0},'query':{'type':'string','maxLength':1000}},[]),version='2'),resources)
    service.catalog.register(ToolSpec('research_skill','Read one short optional research method guide: plan, evidence, synthesis, design or molecules (multi-molecule dossiers). Fixed host paths; never a permission grant.',schema({'stage':{'enum':['plan','evidence','synthesis','design','molecules']}},['stage']),version='2'),skill)
