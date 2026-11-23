"""Explicit host-assisted discovery input; no benchmark-answer access."""
import hashlib
import json
from .validation import text


def seed_bundle(service,sid,rid,question,path):
    from ..open_research_case import _read_dataset,_object
    _,content=_read_dataset(path)
    bundle=json.loads(content,object_pairs_hook=_object)
    if not isinstance(bundle,dict) or bundle.get('question')!=question:
        raise ValueError('Discovery bundle question mismatch')
    sources=bundle.get('sources')
    if not isinstance(sources,list) or not 1<=len(sources)<=20:
        raise ValueError('Discovery bundle needs 1..20 source candidates')
    # Preflight every member before any registration.
    for s in sources:
        if not isinstance(s,dict):raise ValueError('Invalid discovery source')
        text(s.get('title'),'discovery title',2000)
        raw=s.get('content_utf8')
        if not isinstance(raw,str) or len(raw.encode())>4*1024*1024:raise ValueError('Invalid discovery content')
        if hashlib.sha256(raw.encode()).hexdigest()!=s.get('sha256'):raise ValueError('Discovery source hash mismatch')
        json.loads(raw,object_pairs_hook=_object)
        if not isinstance(s.get('provenance'),dict):raise ValueError('Invalid discovery provenance')
    bindings=[]
    for i,s in enumerate(sources):
        provenance={**s['provenance'],'host_assisted_discovery':True,'discovery_bundle_sha256':hashlib.sha256(content).hexdigest()}
        a=service.store.put_artifact(s['content_utf8'].encode(),'discovery-source.json','application/json',provenance,session_id=sid,run_id=rid)
        source=service.research.register_source(sid,rid,artifact_id=a['id'],kind='literature',title=s['title'],provenance=provenance,idempotency_key=f'discovery-{rid}-{i}')
        bindings.append({'source_id':source['id'],'title':source['title'][:160],'scope':provenance.get('text_availability','unknown')})
    index=service.store.put_artifact(json.dumps(bindings,ensure_ascii=False),'discovery-index.json','application/json',{'scope':'unreviewed candidate index'},session_id=sid,run_id=rid)
    result={'host_assisted_public_discovery':True,'bundle_sha256':hashlib.sha256(content).hexdigest(),'source_ids':[b['source_id'] for b in bindings],'index_artifact_id':index['id']}
    service.store.append_event(sid,'research.discovery.seeded',result,rid)
    # Latest sources appear first in the bounded runtime index; complete list is immutable.
    service.store.append_message(sid,{'role':'assistant','content':'Host提供以下同题公开检索候选资料，尚未审查，不是参考答案。请先读取其中与关键物性最相关的原始研究，再独立比较至少两个候选，按evidence→decision→report三阶段critic迭代。不要把发现摘要当作阅读全文，也不要仅凭已有材料数量跳过审查。完整索引artifact：'+index['id']+'；近期候选：'+json.dumps(bindings[-3:],ensure_ascii=False)},rid)
    return result
