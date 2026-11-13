"""Offline integration for explicit public URL access; no frozen solver edits."""
import json
from types import SimpleNamespace

import httpx
import pytest

from adj_partner.web_access import WebPageFetcher, register_web_access
from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.evidence_access import register_evidence_access
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.research import ResearchStore
from adj_partner.runtime.context import ContextView
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


URL='https://docs.example/tutorial/'
HTML=b'<html><body><h1>Official-looking fixture, not verified authority</h1><p>Plain public static evidence.</p></body></html>'


@pytest.fixture
def rig(tmp_path):
    store=Store(tmp_path/'state');research=ResearchStore(store);catalog=ToolCatalog();calls=[]
    native=NativeCapabilities(store,research=research)
    service=SimpleNamespace(store=store,research=research,catalog=catalog,native=native)
    def respond(request):
        calls.append(str(request.url));return httpx.Response(200,headers={'content-type':'text/html'},content=HTML)
    fetcher=WebPageFetcher(transport=httpx.MockTransport(respond),resolver=lambda h:['93.184.216.34'])
    register_web_access(service,fetcher=fetcher);register_evidence_access(service)
    sid=store.create_session(SessionConfig(allowed_tools=['web_page_read','evidence_read'],enabled_tools=['web_page_read','evidence_read'],allow_network=True).to_dict())['id']
    tid=store.create_task(sid,'Synthetic public document reading',{})['id'];rid=store.create_run(sid,tid,Budget().to_dict())['id']
    yield service,ToolExecutor(store,catalog),sid,rid,calls
    store.close()


def call(rig,args,identifier='read'):
    _,executor,sid,rid,_=rig
    return executor.execute(ToolCall(identifier,'web_page_read',args),sid,rid)


def discovery(rig):
    service,_,sid,rid,_=rig
    prov={'provider':'synthetic_search','text_availability':'search_snippet','url':URL,'response_artifact_id':'artifact_fixture','result_index':0}
    a=service.store.put_artifact(json.dumps({'title':'Discovery title','url':URL,'snippet':'Discovery only'}),'result.json','application/json',prov,session_id=sid,run_id=rid)
    return service.research.register_source(sid,rid,artifact_id=a['id'],kind='web',title='Discovery title',provenance=prov,idempotency_key='discovery')


def test_direct_url_flat_schema_exact_provenance_and_visible_evidence(rig):
    service,executor,sid,rid,calls=rig
    schema=service.catalog.get('web_page_read').spec.input_schema
    assert set(schema['properties'])=={'source_id','url'} and not schema.get('required')
    assert 'oneOf' not in schema and 'anyOf' not in schema
    result=call(rig,{'url':URL})
    assert result.status=='ok',result.to_dict()
    source=service.research.get_source(sid,rid,result.data['source_id'])
    assert source['provenance']['locator_origin']=='model_supplied_public_url'
    assert 'parent_source_id' not in source['provenance']
    assert 'response_artifact_id' not in source['provenance']
    assert source['provenance']['requested_url']==URL and source['provenance']['url']==URL
    assert source['kind']=='web' and source['provenance']['text_availability']=='derived_full_text'
    assert calls==[URL]
    visible=ContextView(service.store).tool_message(sid,rid,'read',result.to_dict())
    assert len(visible['content'].encode())<2400 and 'context_truncated' not in visible['content']
    evidence=executor.execute(ToolCall('window','evidence_read',{'source_id':source['id'],'query':'Plain public'}),sid,rid)
    assert evidence.status=='ok' and 'Plain public static evidence.' in evidence.data['text']


@pytest.mark.parametrize('args',[{}, {'source_id':'source_fake','url':URL}, {'url':''}, {'source_id':''}])
def test_exactly_one_locator_fails_before_fetch(rig,args):
    assert call(rig,args).status!='ok'
    assert rig[-1]==[]


def test_discovery_source_mode_preserves_original_scope(rig):
    service,executor,sid,rid,calls=rig;original=discovery(rig)
    result=call(rig,{'source_id':original['id']})
    assert result.status=='ok',result.to_dict()
    source=service.research.get_source(sid,rid,result.data['source_id'])
    assert source['provenance']['parent_source_id']==original['id']
    assert source['provenance']['locator_origin']=='web_search_discovery'
    assert source['title']=='Discovery title'
    assert call(rig,{'source_id':source['id']},'not-discovery').status!='ok'
    other=service.store.create_task(sid,'Other task',{})
    run=service.store.create_run(sid,other['id'],Budget().to_dict())
    assert executor.execute(ToolCall('cross','web_page_read',{'source_id':original['id']}),sid,run['id']).status!='ok'
    assert calls==[URL]


@pytest.mark.parametrize('url',['http://docs.example/x','https://127.0.0.1/x','https://u:p@docs.example/x','https://docs.example/.env','https://docs.example/?token=secret'])
def test_direct_url_uses_existing_target_denials(rig,url):
    assert call(rig,{'url':url}).status!='ok'
    assert rig[-1]==[]


def test_direct_url_still_requires_host_network_permission(rig):
    service,executor,sid,rid,calls=rig
    session=service.store.get_session(sid);config=session['config'];config['allow_network']=False
    service.store.update_session(sid,config,session['revision'])
    assert call(rig,{'url':URL}).status!='ok' and calls==[]


@pytest.mark.parametrize('status,body',[(403,b'Blocked'),(404,b'Missing'),(200,b'<html><script>dynamic()</script></html>')])
def test_no_retry_or_source_on_http_denial_or_javascript_only(tmp_path,status,body):
    store=Store(tmp_path/'s');research=ResearchStore(store);catalog=ToolCatalog();calls=[]
    service=SimpleNamespace(store=store,research=research,catalog=catalog,native=NativeCapabilities(store,research=research))
    def respond(request):calls.append(str(request.url));return httpx.Response(status,headers={'content-type':'text/html'},content=body)
    register_web_access(service,fetcher=WebPageFetcher(transport=httpx.MockTransport(respond),resolver=lambda h:['93.184.216.34']))
    sid=store.create_session(SessionConfig(allowed_tools=['web_page_read'],enabled_tools=['web_page_read'],allow_network=True).to_dict())['id']
    tid=store.create_task(sid,'Reading fixture',{})['id'];rid=store.create_run(sid,tid,Budget().to_dict())['id']
    try:
        result=ToolExecutor(store,catalog).execute(ToolCall('denied','web_page_read',{'url':URL}),sid,rid)
        assert result.status!='ok' and len(calls)==1
        assert research.list_sources(sid,rid)==[]
    finally:store.close()
