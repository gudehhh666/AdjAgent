"""Exercise the real native tool boundary with deterministic provider fixtures."""
import httpx
from test_native_integration import setup,call
from test_native_literature import payload
from adj_partner.native.literature import EuropePMC

def test_initial_discovery_bounds_queries_before_network_then_allows_followup(tmp_path):
    requests=[]
    def handle(req):
        requests.append(req);return httpx.Response(200,json=payload([]))
    store,r,n,ex,sid,rid=setup(tmp_path,network=True,literature=EuropePMC(transport=httpx.MockTransport(handle)))
    task=store.get_task(store.get_run(rid)['task_id'])
    store.update_task(task['id'],{'acceptance':{'require_concise_discovery':True}})
    long='one two three four five six seven eight nine'
    blocked=call(ex,sid,rid,'literature_search',{'query':long},'long')
    assert blocked.status=='error' and blocked.error['code']=='discovery_query_too_broad'
    assert not requests
    for i in range(3):
        assert call(ex,sid,rid,'literature_search',{'query':'distinctive constraint vaccine'},f'short{i}').status=='ok'
    assert call(ex,sid,rid,'literature_search',{'query':long},'followup').status=='ok'
    assert len(requests)==4

def test_legacy_discovery_queries_are_unchanged(tmp_path):
    provider=EuropePMC(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=payload([]))))
    store,r,n,ex,sid,rid=setup(tmp_path,network=True,literature=provider)
    assert call(ex,sid,rid,'literature_search',{'query':'one two three four five six seven eight nine'}).status=='ok'
