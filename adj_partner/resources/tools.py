"""Native catalog integration preserving source scope and literal selectors."""
from ..contracts import ToolResult
from ..tools import ToolFailure, ToolSpec
from ..tools.catalog import canonical_json


def _identity_fields(record):
    keys = ('Vaxjo ID', 'VO ID', 'Catalog', 'PME Code', 'biosample_id', 'gsm', 'Adjuvant', 'Adjuvant Name',
            'accession', 'gene', 'taxon_id', 'nct_id', 'protocol_id', 'GeoAccession')
    return {k: record['fields'][k] for k in keys if k in record['fields']}


def register_resource_tools(native, catalog):
    def schema(properties, required):
        return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}
    limit={'type':'integer','minimum':1,'maximum':20}
    text={'type':'string','minLength':1,'maxLength':256}
    datasets=sorted(native.resource_pack.datasets) if native.resource_pack else []
    kinds=sorted({r['kind'] for r in native.resource_pack.records.values()}) if native.resource_pack else ['assay','ontology']
    from .skills import SKILLS, read_skill
    definitions=[
        ToolSpec('adjuvant_skill', 'Read one short evidence skill at the relevant entry point: paper-specific details, mechanism plausibility, assay choice, expression context, clinical registry or exact source tables. Guidance does not grant permissions.', schema({'skill':{'enum':list(SKILLS)}},['skill']),version='1'),
        ToolSpec('adjuvant_excerpt', 'Read a literal bounded character window of one field from a known resource record. Use after inspecting actual field names; useful for protocol page text omitted from summaries. Strings retain their text; structured fields use canonical JSON. Offsets are Unicode characters of that representation. Source selector identifies the whole original field; excerpt offsets identify the shown range.',schema({'record_id':text,'field':text,'start':{'type':'integer','minimum':0},'length':{'type':'integer','minimum':1,'maximum':400}},['record_id','field']),version='1'),
        ToolSpec('adjuvant_search', 'Search frozen source records by query words or dataset plus exact scalar filters on original fields, e.g. dataset=vaxjo_master, filters={"Vaxjo ID":178}. External database IDs are NOT worksheet row numbers: never infer a pack record ID from an external numeric ID. Filters preserve original types and null; query, if present, further narrows matches. Trial registrations, published protocols and curated protein/reaction annotations are distinct evidence types, not proof of efficacy.',
                 schema({'query':{'type':'string','minLength':1,'maxLength':500},'kind':{'enum':kinds},
                         'dataset':{'enum':datasets} if datasets else text,'limit':limit,
                         'match_mode':{'enum':['all','ranked'],'description':'Default all: every query word required. Ranked: partial lexical discovery; exact filters still apply; read exact records before making claims.'},
                         'filters':{'type':'object','minProperties':1,'maxProperties':8,'additionalProperties':{'type':['string','number','boolean','null']}}},[]),version='7'),
        ToolSpec('adjuvant_record', 'Read an exact record ID directly, including IDs supplied in the question; search is unnecessary for a known ID. Use fields to request exact assay columns. A compact summary includes title, requested values and original-source hash; full record remains cited. Database/assay evidence does not validate vaccine efficacy.',
                 schema({'record_id':text,'fields':{'type':'array','items':text,'maxItems':8,'uniqueItems':True}},['record_id']),version='3'),
        ToolSpec('adjuvant_relations', 'Read bounded explicit ontology or study/workbook membership relations for an exact pack record. These are not experimental causal links. Refer to source_record with adjuvant_record for evidence.',
                 schema({'record_id':text,'direction':{'enum':['outgoing','incoming']},'predicate':text,'limit':limit},['record_id']),version='3')
    ]

    def handler(name):
        def invoke(context, **args):
            bound=native._bind(context)
            if name=='adjuvant_skill':
                try:return ToolResult(data={'summary':read_skill(args['skill'],bound.resource_pack.datasets if bound.resource_pack else {})})
                except (KeyError,ValueError) as exc:raise ToolFailure('resource_skill',str(exc)) from exc
            if bound.resource_pack is None:
                raise ToolFailure('resource_pack_unconfigured','Install a host resource pack before querying.')
            pack=bound.resource_pack
            try:
                if name=='adjuvant_relations':
                    return ToolResult(data=pack.neighbors(**args))
                if name=='adjuvant_search':
                    response=pack.search(**args)
                    records=response.pop('results')
                else:
                    records=[pack.read(args['record_id'])]; response={}
                    unknown=set(args.get('fields',[]))-set(records[0]['fields'])
                    if unknown:
                        raise ValueError('Unknown fields; available: '+', '.join(records[0]['fields']))
                    if name=='adjuvant_excerpt':
                        value=records[0]['fields'][args['field']]
                        if not isinstance(value,(str,list,dict)):raise ValueError('Use adjuvant_record for scalar values')
                        value=value if isinstance(value,str) else canonical_json(value)
                        start=args.get('start',0);length=args.get('length',400)
                        if type(start) is not int or start<0 or start>len(value) or type(length) is not int or not 1<=length<=400:raise ValueError('Invalid excerpt range')
            except (KeyError,ValueError) as exc:
                raise ToolFailure('resource_pack_query',str(exc)) from exc
            imported=[]; artifacts=[]
            for record in records:
                context.check_cancelled()
                safe=context.store.redact(record)
                provenance={**safe['provenance'], 'provider':'AdjuvantResourcePack',
                            'record_id':safe['id'], 'url':safe['provenance']['source_url']}
                artifact=context.store.put_artifact(canonical_json({'record':safe}), 'adjuvant-record.json',
                    'application/json',provenance,session_id=context.session_id,run_id=context.run_id)
                source=bound.research.register_source(context.session_id,context.run_id,artifact_id=artifact['id'],
                    kind='web',title=safe['title'],provenance=provenance,
                    idempotency_key=f'adjuvant-pack:{context.call_id}:{pack.manifest_hash}:{safe["id"]}')
                artifacts.append(artifact)
                imported.append({'record':safe,'source_id':source['id'],'artifact_id':artifact['id'],
                                 'selector':{'pointer':'/record'}})
            if name=='adjuvant_search':
                data={**response,'results':imported}
                summary={'hit_count':response['hit_count'],'results':[],
                    'applied_filters':response['applied_filters'],
                    'limitations':'Frozen source lookup, not efficacy ranking. Read exact IDs for fields and citations.'}
                for item in imported:
                    r=item['record']
                    candidate={k:r[k] for k in ['id','title','dataset','kind']}
                    candidate['identity_fields']=_identity_fields(r)
                    proposed={**summary,'results':summary['results']+[candidate]}
                    if len(canonical_json(proposed).encode())>1300: break
                    summary=proposed
                if args.get('match_mode')=='ranked':summary['match_mode']='ranked partial matches; verify relevance with exact record reads'
                summary['results_omitted_from_summary']=len(imported)-len(summary['results'])
            elif name=='adjuvant_excerpt':
                data=imported[0];r=data['record'];field=args['field'];value=r['fields'][field]
                representation='text' if isinstance(value,str) else 'canonical_json'
                value=value if isinstance(value,str) else canonical_json(value)
                start=args.get('start',0);end=min(len(value),start+args.get('length',400))
                summary={'record_id':r['id'],'field':field,'representation':representation,'text':value[start:end],'start':start,'end':end,'total_chars':len(value),
                    'has_more':end<len(value),'source_id':data['source_id'],'raw_sha256':r['provenance']['raw_sha256'],
                    'selector':{'pointer':'/record/fields/'+field.replace('~','~0').replace('/','~1')},
                    'scope':'Literal field excerpt; not efficacy evidence.'}
            else:
                data=imported[0];r=data['record']
                summary={'record_id':r['id'],'title':r['title'],'dataset':r['dataset'],
                    'raw_sha256':r['provenance']['raw_sha256'],
                    'source_id':data['source_id'],'selector':data['selector'],
                    'values':{},'limitations':'Literal source records; metadata/worksheet cells do not establish mechanism or vaccine efficacy.'}
                requested=args.get('fields',[])
                summary['available_fields']=list(r['fields']) if not requested else requested
                summary['fields_omitted_from_summary']=[]
                for key in requested:
                    candidate={**summary,'values':{**summary['values'],key:r['fields'][key]}}
                    if len(canonical_json(candidate).encode())<=1300:
                        summary=candidate
                    else:
                        summary['fields_omitted_from_summary'].append(key)
            data['summary']=summary
            return ToolResult(data=data,artifacts=artifacts)
        return invoke
    catalog.register_many([(d,handler(d.name)) for d in definitions])
    from .usage_tools import register_usage_tools
    register_usage_tools(native,catalog)
    from .citations import register_citation_tool
    register_citation_tool(native,catalog)
    from .field_packets import register_field_tool
    register_field_tool(native,catalog)
