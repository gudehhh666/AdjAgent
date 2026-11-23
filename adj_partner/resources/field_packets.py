"""Whole, typed multi-field reads with bounded model-visible citations."""
from ..contracts import ToolResult
from ..tools import ToolFailure,ToolSpec
from ..tools.catalog import canonical_json


def field_packet(record,fields):
    if (not isinstance(fields,list) or not 1<=len(fields)<=8 or
        any(not isinstance(f,str) or f not in record['fields'] for f in fields) or len(set(fields))!=len(fields)):
        raise ValueError('Request 1..8 distinct original fields; unknown fields are not substituted')
    base={'record_id':record['id'],'values':{},'citations':[],'selectors':{},
        'omitted_fields':list(fields),'complete':False,
        'next_action':'For omitted fields use adjuvant_cite with focus, or request a smaller batch.',
        'boundary':'Literal source values, not independent evidence of mechanism or efficacy.'}
    if len(canonical_json(base).encode())>1300:
        raise ValueError('Field names exceed visible packet budget; request a smaller batch')
    for field in fields:
        value=record['fields'][field];quote=value if isinstance(value,str) else canonical_json(value)
        cite={'record_id':record['id'],'field':field,'quote':quote}
        candidate={**base,'values':{**base['values'],field:value},
            'citations':base['citations']+([cite] if quote else []),
            'selectors':{**base['selectors'],field:{'pointer':'/record/fields/'+field.replace('~','~0').replace('/','~1')}},
            'omitted_fields':[f for f in base['omitted_fields'] if f!=field]}
        if len(canonical_json(candidate).encode())<=1300:base=candidate
    base['complete']=not base['omitted_fields']
    if base['complete']:base['next_action']=None
    return base


def register_field_tool(native,catalog):
    text={'type':'string','minLength':1,'maxLength':256}
    spec=ToolSpec('adjuvant_read_fields',
        'Read multiple exact original fields and ready-to-copy citations in ONE call for a known record ID. '
        'Preferred for scalar tables and short fields. No prior guide required when ID/fields are known. '
        'Returned values are whole and retain null, bool, number, formula text and cached objects. '
        'Explicit omitted_fields means not returned, never zero; use adjuvant_cite with focus for long text. '
        'Copy citations directly, never quote wrapper values keys or infer mechanism/efficacy from metadata.',
        {'type':'object','properties':{'record_id':text,'fields':{'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,'items':text}},'required':['record_id','fields'],'additionalProperties':False},version='1')
    def read(context,record_id,fields):
        bound=native._bind(context);pack=bound.resource_pack
        if pack is None:raise ToolFailure('resource_pack_unconfigured','No local resource pack')
        try:
            safe=context.store.redact(pack.read(record_id));packet=field_packet(safe,fields)
        except (KeyError,ValueError,TypeError) as exc:raise ToolFailure('resource_fields',str(exc)) from exc
        context.check_cancelled()
        provenance={**safe['provenance'],'provider':'AdjuvantResourcePack','record_id':safe['id'],'url':safe['provenance']['source_url']}
        artifact=context.store.put_artifact(canonical_json({'record':safe}),'adjuvant-record.json','application/json',provenance,session_id=context.session_id,run_id=context.run_id)
        source=bound.research.register_source(context.session_id,context.run_id,artifact_id=artifact['id'],kind='web',title=safe['title'],provenance=provenance,idempotency_key=f'adjuvant-fields:{context.call_id}:{pack.manifest_hash}:{safe["id"]}')
        packet.update(source_id=source['id'],raw_sha256=provenance['raw_sha256'])
        return ToolResult(data={'summary':packet,'artifact_id':artifact['id'],'pack_sha256':pack.manifest_hash},artifacts=[artifact])
    catalog.register_many([(spec,read)])
