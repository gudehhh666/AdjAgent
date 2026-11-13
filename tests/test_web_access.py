import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from adj_partner.web_access import WebPageFetcher, html_paragraphs, register_web_access
from adj_partner.tools.executor import ToolFailure


HTML = b'<html><head><title>Title</title><script>secret_script()</script></head><body><nav>navigation</nav><main><h1>Public documentation</h1><p>First useful paragraph.</p><p>Second <b>useful</b> paragraph &amp; evidence.</p><div hidden>hidden material</div><style>style material</style><footer>footer text</footer></main></body></html>'
URL = 'https://docs.example/public'


def test_extract_static_visible_paragraphs_without_scripts_navigation():
    paragraphs = html_paragraphs(HTML.decode())
    text = ' '.join(paragraphs)
    assert 'First useful paragraph.' in text and 'Second useful paragraph & evidence.' in text
    assert all(x not in text for x in ('navigation', 'secret_script', 'hidden material', 'style material', 'footer text'))


@pytest.mark.parametrize('url', ['http://docs.example/x', 'https://127.0.0.1/x', 'https://[::1]/x', 'https://169.254.169.254/latest/meta-data', 'https://u:p@docs.example/x', 'https://docs.example/?api_key=private', 'https://224.0.0.1/x', 'https://%31%32%37.0.0.1/x', 'https://文档.example/x'])
def test_denies_unsafe_target_before_request(url):
    calls=[]
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:calls.append(r)),resolver=lambda h:['93.184.216.34'])
    with pytest.raises((ToolFailure,ValueError)): f.fetch(url)
    assert not calls


def test_public_html_fetch_and_private_redirect_rejected():
    calls=[]
    def reply(request):
        calls.append(str(request.url))
        return httpx.Response(302,headers={'location':'https://127.0.0.1/private'})
    f=WebPageFetcher(transport=httpx.MockTransport(reply),resolver=lambda h:['93.184.216.34'])
    with pytest.raises(ToolFailure): f.fetch(URL)
    assert calls == [URL]
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/html; charset=utf-8'},content=HTML)),resolver=lambda h:['93.184.216.34'])
    assert f.fetch(URL).body == HTML


def test_size_media_and_mixed_dns_fail_closed():
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/html'},content=b'x'*1025)),resolver=lambda h:['93.184.216.34'],maximum_bytes=1024)
    with pytest.raises(ToolFailure): f.fetch(URL)
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'application/pdf'},content=b'%PDF-')),resolver=lambda h:['93.184.216.34'])
    with pytest.raises(ToolFailure): f.fetch(URL)
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:pytest.fail('network must not run')),resolver=lambda h:['93.184.216.34','10.0.0.1'])
    with pytest.raises(ToolFailure): f.fetch(URL)


def test_rebinding_redirect_is_revalidated_and_production_transport_cannot_be_unpinned():
    resolutions=iter([['93.184.216.34'],['127.0.0.1']]);calls=[]
    def reply(request):
        calls.append(str(request.url))
        return httpx.Response(302,headers={'location':'/next'})
    f=WebPageFetcher(transport=httpx.MockTransport(reply),resolver=lambda h:next(resolutions))
    with pytest.raises(ToolFailure): f.fetch(URL)
    assert calls==[URL]
    with pytest.raises(ValueError): WebPageFetcher(transport=object())


def test_dns_elapsed_timeout_is_bounded():
    import time
    def slow(host):time.sleep(.15);return ['93.184.216.34']
    f=WebPageFetcher(transport=httpx.MockTransport(lambda r:pytest.fail('network must not run')),resolver=slow,timeout_seconds=.02)
    start=time.monotonic()
    with pytest.raises(ToolFailure): f.fetch(URL)
    assert time.monotonic()-start < .12


def test_production_curl_pins_dns_and_disables_local_configuration(monkeypatch):
    captured={}
    def run(args,**kwargs):
        captured['args']=args
        Path(args[args.index('--output')+1]).write_bytes(HTML)
        Path(args[args.index('--dump-header')+1]).write_text('HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
        return SimpleNamespace(returncode=0,stdout=b'200',stderr=b'')
    monkeypatch.setattr('adj_partner.web_access.subprocess.run',run)
    assert WebPageFetcher(resolver=lambda h:['93.184.216.34']).fetch(URL).body == HTML
    args=captured['args']
    assert args[1]=='-q' and '--no-netrc' in args and '--no-location' in args
    assert args[args.index('--resolve')+1]=='docs.example:443:93.184.216.34'
    assert args[args.index('--noproxy')+1]=='*'


def test_real_source_scope_projection_and_evidence_reader(tmp_path, monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.models import ScriptedPort
    from adj_partner.contracts import ToolCall
    from adj_partner.tools.policy import SessionToolset
    from adj_partner.runtime.context import ContextView
    from adj_partner.native.web_search import WebSearchResponse
    class Search:
        secrets=();provider='serper'
        def search(self,*a,**k):
            return WebSearchResponse([{'title':'Public documentation','url':URL,'snippet':'Snippet only','text_availability':'search_snippet'}],'https://google.serper.dev/search','today',b'{}')
    fetcher=WebPageFetcher(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={'content-type':'text/html'},content=HTML)),resolver=lambda h:['93.184.216.34'])
    monkeypatch.setattr('adj_partner.web_access.WebPageFetcher',lambda:fetcher)
    svc=AgentService(tmp_path/'state',project_root=Path(__file__).resolve().parents[1],model=ScriptedPort([]))
    svc.native.web_search=Search()
    # AgentService may already register web_search from environment; replace its adapter only.
    if 'web_search' not in svc.catalog.names():
        from adj_partner.tools import ToolSpec
        svc.catalog.register(ToolSpec('web_search','fixture search',{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False},network=True),svc.native._web)
    if 'web_page_read' not in svc.catalog.names(): register_web_access(svc,fetcher=fetcher)
    sid=svc.create_session(allow_network=True,allowed_tools=['web_search','web_page_read','evidence_read'])['id']
    for name in ('web_search','web_page_read','evidence_read'): SessionToolset(svc.store,svc.catalog,sid).enable(name)
    run=svc.submit(sid,'read public web source')['id']
    def call(cid,name,args):return svc.executor.execute(ToolCall(cid,name,args),sid,run)
    try:
        found=call('search','web_search',{'query':'public documentation'})
        assert found.status=='ok',found.to_dict()
        source=found.data['results'][0]['source_id']
        r=call('fetch','web_page_read',{'source_id':source})
        assert r.status=='ok',r.to_dict()
        v=ContextView(svc.store).tool_message(sid,run,'fetch',r.to_dict())
        assert len(v['content'].encode())<2400 and not json.loads(v['content']).get('context_truncated')
        e=call('read','evidence_read',{'source_id':r.data['source_id'],'query':'Second useful'})
        assert e.status=='ok' and 'Second useful paragraph & evidence.' in e.data['text']
        assert call('badkind','web_page_read',{'source_id':r.data['source_id']}).status!='ok'
        svc.cancel(run)
        later=svc.submit(sid,'another task')['id']
        assert svc.executor.execute(ToolCall('cross','web_page_read',{'source_id':source}),sid,later).status!='ok'
    finally:svc.close()
