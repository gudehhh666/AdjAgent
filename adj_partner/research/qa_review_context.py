"""Select claim-relevant retrieved text for QA review, without benchmark labels."""
import json
import re

STOP=set('the a an and or of in to for with by is are was were how what why did does from that this these study evidence supplied answer draft'.split())


def _tokens(text):
    return {t.lower() for t in re.findall(r'[\w]+(?:[-+][\w]+)*',text) if len(t)>2 and t.lower() not in STOP}


def _passages(doc):
    """Keep paragraph and caption boundaries; reference lists are not evidence."""
    def walk(value,locator='source',kind='text'):
        if isinstance(value,str):
            if value.strip():yield locator,kind,value
        elif isinstance(value,list):
            for i,v in enumerate(value):yield from walk(v,f'{locator}/{i}',kind)
        elif isinstance(value,dict):
            loc=value.get('locator',locator)
            for k,v in value.items():
                if k=='record' and isinstance(v,dict):
                    yield loc+'/record','database_record',json.dumps(v,ensure_ascii=False,sort_keys=True)
                elif k in ('abstract','sections','paragraphs','text','body','content','tables','rows','cells','caption'):
                    yield from walk(v,loc,'caption_text' if k=='caption' else kind)
                elif k=='figures':yield from walk(v,loc,'caption_text')
                elif k=='snippet':yield from walk(v,loc,'search_snippet')
    yield from walk(doc)


def select_evidence(question,draft,sources,budget=60000,per_source=16000):
    """Rank across the whole retrieved document, then pack valid JSON to budget.

    sources contain document objects from retrieval, never reference answers.
    Draft terms locate claims for verification; lexical matches do not prove them.
    """
    qterms=_tokens(question);dterms=_tokens(draft)
    candidates=[];seen=set();records={}
    for i,s in enumerate(sources):
        p=s.get('provenance') or {};doc=s.get('document')
        if doc is None:continue
        url=p.get('url') or p.get('request_url') or s.get('url')
        availability=p.get('text_availability','unknown')
        for locator,kind,text in _passages(doc):
            # Long paragraphs get overlapping windows, not prefix-only truncation.
            for offset in range(0,len(text),1000):
                chunk=text[offset:offset+1400].strip()
                identity=(url,chunk)
                if not chunk or identity in seen:continue
                seen.add(identity);terms=_tokens(chunk)
                relevance=4*len(qterms&terms)+len(dterms&terms)
                if relevance==0:continue
                rank=(relevance,availability in ('full_text','derived_full_text'),-i,-offset)
                candidates.append((rank,i,{'locator':locator,'offset':offset,'evidence_type':kind,'text':chunk}))
        records[i]={'source_id':s.get('id',str(i)),'url':url,'text_availability':availability,
                    'selection':'query_and_draft_relevant_excerpts_not_full_document','excerpts':[]}
    packet=[];included={};source_size={}
    for rank,i,excerpt in sorted(candidates,key=lambda x:x[0],reverse=True):
        size=len(excerpt['text'])
        if source_size.get(i,0)+size>per_source:continue
        fresh=i not in included
        if fresh:
            included[i]=records[i];packet.append(included[i])
        included[i]['excerpts'].append(excerpt)
        if len(json.dumps(packet,ensure_ascii=False))>budget:
            included[i]['excerpts'].pop()
            if fresh:packet.pop();included.pop(i)
            continue
        source_size[i]=source_size.get(i,0)+size
    return packet
