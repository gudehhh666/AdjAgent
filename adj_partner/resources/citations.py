"""Literal source citation packets. Existence is not scientific claim support."""
from ..contracts import ToolResult
from ..tools import ToolFailure, ToolSpec
from ..tools.catalog import canonical_json
from ..research.records import extract
from .usage import excerpt_fields


def citation_packet(record, field, *, pointer=None, start=None, focus=None, length=400):
    if field not in record['fields']:
        raise ValueError('Unknown original field; inspect the record or guide')
    if type(length) is not int or not 1 <= length <= 600:
        raise ValueError('length must be 1..600 characters')
    if focus is not None and (start is not None or pointer is not None):
        focus=None
    root='/record/fields/'+field.replace('~','~0').replace('/','~1')
    pointer=pointer or root
    if not isinstance(pointer,str) or not (pointer==root or pointer.startswith(root+'/')):
        raise ValueError('Pointer must stay within the named original field')
    matched=[]
    if focus is not None:
        fragments=excerpt_fields(record,[field],focus,limit=1,width=400)
        if fragments:
            chosen=fragments[0];pointer=chosen['selector']['pointer'];start=chosen['start'];matched=chosen['matched_terms']
    value=extract(canonical_json({'record':record}).encode(),{'pointer':pointer})['value']
    text=value if isinstance(value,str) else canonical_json(value)
    start=0 if start is None else start
    if type(start) is not int or not 0 <= start <= len(text):
        raise ValueError('Invalid start character offset')
    end=min(len(text),start+length)
    def build(end):
        quote=text[start:end]
        result={'citation':{'record_id':record['id'],'field':field,'quote':quote} if quote else None,
            'selector':{'pointer':pointer},'representation':'text' if isinstance(value,str) else 'canonical_json',
            'start':start,'end':end,'total_chars':len(text),'truncated':start>0 or end<len(text),
            'boundary':'Literal source only; not verification of mechanism, efficacy or claim support.'}
        if focus is not None:result['matched_terms']=matched
        if start==0 and end==len(text) and len(canonical_json(value).encode())<=160:
            result['value']=value
        return result
    packet=build(end)
    # Bound UTF-8 JSON bytes, not character count. Never splice or alter source text.
    while len(canonical_json(packet).encode())>1300 and end>start:
        end-=1;packet=build(end)
    if len(canonical_json(packet).encode())>1300 or (text and end==start):
        raise ValueError('Citation metadata exceeds visible packet budget; choose a shorter selector')
    return packet


def register_citation_tool(native,catalog):
    text={'type':'string','minLength':1,'maxLength':256}
    spec=ToolSpec('adjuvant_cite',
        'Read one original field of a known resource record and return a ready-to-copy citation. '
        'No guide/search needed for known record+field. Scalars preserve types; formula text is never executed. '
        'Use focus to find a relevant continuous leaf passage, OR an exact returned pointer and start/length. '
        'Copy citation.record_id/field/quote verbatim; do not cite wrapper values, gap status, or join fragments. '
        'Source existence does not verify scientific claims. An unmatched focus is not evidence of absence.',
        {'type':'object','properties':{'record_id':text,'field':text,
            'pointer':{'type':'string','minLength':1,'maxLength':2000},
            'focus':{'type':'string','minLength':1,'maxLength':200},
            'start':{'type':'integer','minimum':0},'length':{'type':'integer','minimum':1,'maximum':600}},
         'required':['record_id','field'],'additionalProperties':False},version='1')
    def cite(context,**args):
        bound=native._bind(context);pack=bound.resource_pack
        if pack is None:raise ToolFailure('resource_pack_unconfigured','No local resource pack')
        try:
            record=pack.read(args.pop('record_id'));safe=context.store.redact(record)
            packet=citation_packet(safe,**args)
        except (KeyError,ValueError,TypeError) as exc:
            raise ToolFailure('resource_citation',str(exc)) from exc
        context.check_cancelled()
        provenance={**safe['provenance'],'provider':'AdjuvantResourcePack','record_id':safe['id'],'url':safe['provenance']['source_url']}
        artifact=context.store.put_artifact(canonical_json({'record':safe}),'adjuvant-record.json','application/json',provenance,session_id=context.session_id,run_id=context.run_id)
        source=bound.research.register_source(context.session_id,context.run_id,artifact_id=artifact['id'],kind='web',title=safe['title'],provenance=provenance,idempotency_key=f'adjuvant-cite:{context.call_id}:{pack.manifest_hash}:{safe["id"]}')
        packet.update(source_id=source['id'],raw_sha256=provenance['raw_sha256'])
        return ToolResult(data={'summary':packet,'artifact_id':artifact['id'],'pack_sha256':pack.manifest_hash},artifacts=[artifact])
    catalog.register_many([(spec,cite)])
