"""Model-visible per-resource guides and bounded, source-bound lookup packets."""
from ..contracts import ToolResult
from ..tools import ToolSpec,ToolFailure
from ..tools.catalog import canonical_json
from .usage import USAGE,resource_guide,excerpt_fields


def register_usage_tools(native,catalog):
    def schema(props,required):return {'type':'object','properties':props,'required':required,'additionalProperties':False}
    text={'type':'string','minLength':1,'maxLength':256}
    datasets=sorted(native.resource_pack.datasets) if native.resource_pack else []
    dataset={'enum':datasets} if datasets else text
    specs=[
        ToolSpec('adjuvant_resource_guide','Read actual resource scope, record kinds, typed fields and query/read recipe. Fields are paginated six at a time with next_offset; never guess field names. For protocol text choose kind=assay_protocol_page; for workbook indices inspect kinds. This is guidance, not new permissions.',schema({'dataset':dataset,'kind':text,'offset':{'type':'integer','minimum':0}},['dataset']),version='1'),
        ToolSpec('adjuvant_lookup','Find and READ a bounded evidence packet in one step. query is a short exact all-word anchor (entity, accession or readout); focus separately ranks relevant records and centres excerpts (e.g. query=endotoxin, focus=interference recovery). Exact typed filters remain exact. Pick fields from the resource guide, not invented nested names. Returned fragments have precise nested selectors, literal text and typed scalar values. Candidate matches do not certify relevance or efficacy.',schema({'dataset':dataset,'kind':text,'query':{'type':'string','minLength':1,'maxLength':200},'focus':{'type':'string','maxLength':200},
            'filters':{'type':'object','minProperties':1,'maxProperties':8,'additionalProperties':{'type':['string','number','boolean','null']}},
            'fields':{'type':'array','items':text,'minItems':1,'maxItems':8,'uniqueItems':True},'limit':{'type':'integer','minimum':1,'maximum':2}},['dataset']),version='1')]

    def guide(context,**args):
        bound=native._bind(context)
        if bound.resource_pack is None:raise ToolFailure('resource_pack_unconfigured','No local resource pack')
        try:return ToolResult(data={'summary':resource_guide(bound.resource_pack,**args)})
        except (ValueError,KeyError) as exc:raise ToolFailure('resource_guide',str(exc)) from exc

    def lookup(context,**args):
        bound=native._bind(context);pack=bound.resource_pack
        if pack is None:raise ToolFailure('resource_pack_unconfigured','No local resource pack')
        try:
            dataset=args['dataset']
            if dataset not in pack.datasets or dataset not in USAGE:raise ValueError('Unconfigured resource')
            requested=args.get('fields')
            available=set().union(*(r['fields'] for r in pack.records.values() if r['dataset']==dataset))
            if requested and any(f not in available for f in requested):raise ValueError('Unknown fields; use adjuvant_resource_guide')
            response=pack.search(args.get('query'),dataset=dataset,kind=args.get('kind'),filters=args.get('filters'),focus=args.get('focus'),limit=args.get('limit',1))
            fragments=[];artifacts=[];bindings=[];missing={}
            for r in response['results']:
                context.check_cancelled();safe=context.store.redact(r)
                provenance={**safe['provenance'],'provider':'AdjuvantResourcePack','record_id':safe['id'],'url':safe['provenance']['source_url']}
                artifact=context.store.put_artifact(canonical_json({'record':safe}),'adjuvant-record.json','application/json',provenance,session_id=context.session_id,run_id=context.run_id)
                source=bound.research.register_source(context.session_id,context.run_id,artifact_id=artifact['id'],kind='web',title=safe['title'],provenance=provenance,idempotency_key=f'adjuvant-pack:{context.call_id}:{pack.manifest_hash}:{safe["id"]}')
                artifacts.append(artifact);bindings.append({'record_id':safe['id'],'source_id':source['id'],'artifact_id':artifact['id']})
                fields=requested or [f for f in USAGE[dataset][3] if f in safe['fields']][:4] or list(safe['fields'])[:2]
                if requested:missing[safe['id']]=[f for f in requested if f not in safe['fields']]
                fields=[f for f in fields if f in safe['fields']]
                if not fields:continue
                for x in excerpt_fields(safe,fields,args.get('focus',''),limit=2):
                    if isinstance(x.get('value'),str):x.pop('value')
                    fragments.append({**x,'record_id':safe['id'],'source_id':source['id'],'raw_sha256':safe['provenance']['raw_sha256']})
            summary={'dataset':dataset,'hit_count':response['hit_count'],'fragments':[],
                'boundary':USAGE[dataset][2],'missing_requested_fields':{k:v for k,v in missing.items() if v}}
            for f in fragments:
                proposed={**summary,'fragments':summary['fragments']+[f]}
                if len(canonical_json(proposed).encode())>1650:break
                summary=proposed
            summary['fragments_omitted']=len(fragments)-len(summary['fragments'])
            summary['gap']='no_matching_record' if not response['hit_count'] else 'requested_fields_unavailable' if not fragments else None
            return ToolResult(data={'summary':summary,'bindings':bindings,'all_fragments':fragments,'pack_sha256':pack.manifest_hash},artifacts=artifacts)
        except (ValueError,KeyError) as exc:raise ToolFailure('resource_lookup',str(exc)) from exc
    catalog.register_many([(specs[0],guide),(specs[1],lookup)])
