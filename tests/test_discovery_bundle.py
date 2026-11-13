import json,hashlib
import pytest
from types import SimpleNamespace
from test_harness_research_records import setup

def test_seed_bundle_checks_question_and_source_hash_before_writes(tmp_path):
    from adj_partner.research.discovery_bundle import seed_bundle
    store,r,sid,rid,tid=setup(tmp_path)
    raw=json.dumps({'abstract':'Public fixture only.'});h=hashlib.sha256(raw.encode()).hexdigest()
    data={'question':'Fixture question','sources':[{'title':'Fixture','content_utf8':raw,'sha256':h,'provenance':{'text_availability':'abstract','synthetic_fixture':True}}]}
    p=tmp_path/'bundle.json';p.write_text(json.dumps(data))
    service=SimpleNamespace(store=store,research=r)
    with pytest.raises(ValueError,match='question'):seed_bundle(service,sid,rid,'Different question',p)
    assert r.list_sources(sid,rid)==[]
    result=seed_bundle(service,sid,rid,'Fixture question',p)
    assert result['host_assisted_public_discovery'] and len(result['source_ids'])==1
    assert store.read_artifact(r.get_source(sid,rid,result['source_ids'][0])['artifact_id']).decode()==raw
    data['sources'][0]['sha256']='0'*64;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):seed_bundle(service,sid,rid,'Fixture question',p)
    assert len(r.list_sources(sid,rid))==1
