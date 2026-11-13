"""Task-scoped public HTML reading from discovery sources or explicit public URLs."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import queue
import re
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

import httpx

from adj_partner.contracts import ToolResult
from adj_partner.evidence_access import _material
from adj_partner.native.fulltext import FullTextFetcher, public_target
from adj_partner.tools import ToolSpec, ToolFailure
from adj_partner.tools.catalog import canonical_json
from adj_partner.tools.policy import _credential_path


SCOPE = ('Static HTML text extraction; whitespace normalized. Scripts, navigation and hidden elements excluded. '
         'No browser rendering, JavaScript, OCR, images or table-layout interpretation. Retrieved text is not verified truth.')


JINA_SCOPE = ('Page rendered and converted to Markdown by Jina Reader via the AI Hub gateway; whitespace normalized per block. '
              'Third-party extraction may drop or reorder content and may not match the static HTML. Retrieved text is not verified truth.')
_CREDENTIAL_KEYS = {'token','access_token','api_key','apikey','key','password','passwd','secret','authorization','auth',
                    'signature','x-amz-signature'}


@dataclass(frozen=True)
class WebPageResponse:
    body: bytes
    url: str
    received_at: str
    media: str = 'text/html'
    fallback_from: tuple = ()


def _credential_target(parsed):
    return (_credential_path(Path(unquote(parsed.path))) or
            any(k.casefold() in _CREDENTIAL_KEYS for k, _ in parse_qsl(parsed.query)))


class FallbackFetcher:
    """Try readers in order; a ToolFailure moves to the next reader."""

    def __init__(self, fetchers):
        self.fetchers = fetchers

    @property
    def secrets(self):
        return tuple(secret for fetcher in self.fetchers for secret in getattr(fetcher, 'secrets', ()))

    def fetch(self, url, *, check_cancelled=lambda: None):
        failures = []
        for fetcher in self.fetchers:
            try:
                response = fetcher.fetch(url, check_cancelled=check_cancelled)
            except ToolFailure as exc:
                failures.append(f'{type(fetcher).__name__}: {exc.code}')
                continue
            return WebPageResponse(response.body, response.url, response.received_at, response.media, tuple(failures))
        raise ToolFailure('web_page_all_failed', 'All web page readers failed: ' + '; '.join(failures))


class JinaGatewayReader:
    """r.jina.ai through the AI Hub gateway. The page is fetched by Jina, not by this host."""

    def __init__(self, client, *, maximum_bytes=2*1024*1024):
        self.client, self.maximum_bytes = client, maximum_bytes

    @property
    def secrets(self):
        return self.client.secrets

    def fetch(self, url, *, check_cancelled=lambda: None):
        parsed = urlsplit(url)
        if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ToolFailure('web_page_target', 'Web target must be a public http(s) URL without credentials')
        try: ipaddress.ip_address(parsed.hostname)
        except ValueError: pass
        else: raise ToolFailure('web_page_target', 'Literal IP web targets are denied for the gateway reader')
        if _credential_target(parsed):
            raise ToolFailure('web_page_credentials', 'Credential-bearing web targets are denied')
        check_cancelled()
        _, raw = self.client.post('/', {'url': url}, headers=('X-Return-Format: markdown',),
                                  failure_prefix='web_page')
        check_cancelled()
        try:
            body = json.loads(raw)
            data = body.get('data') if isinstance(body, dict) else None
            if not isinstance(data, dict) or body.get('code') != 200:
                raise ValueError('Jina response has no successful data object')
            content = data.get('content')
            final = data.get('url') or data.get('finalUrl') or url
            if not isinstance(content, str) or not content.strip() or not isinstance(final, str):
                raise ValueError('Jina response content is empty')
        except (ValueError, TypeError, RecursionError) as exc:
            raise ToolFailure('web_page_response', f'Malformed Jina reader response: {str(exc)[:300]}') from exc
        encoded = content.encode()
        if len(encoded) > self.maximum_bytes:
            raise ToolFailure('web_page_size', 'Web page byte limit exceeded')
        return WebPageResponse(encoded, final, datetime.now(timezone.utc).isoformat(), 'text/markdown')


def web_page_fetcher_from_environment(project_root, env_file=None):
    from adj_partner.native.gateway import GatewayClient, settings
    values = settings(project_root, env_file, ('ADJ_WEB_PAGE_READER', 'ADJ_AIHUB_GATEWAY_URL',
                                               'ADJ_JINA_GATEWAY_APP_ID', 'ADJ_JINA_GATEWAY_APP_KEY'))
    reader = values['ADJ_WEB_PAGE_READER'].lower()
    if reader in {'', 'curl'}:
        return None
    if reader != 'jina_gateway':
        raise ValueError('ADJ_WEB_PAGE_READER must be empty, curl or jina_gateway')
    jina = JinaGatewayReader(GatewayClient(values['ADJ_JINA_GATEWAY_APP_ID'], values['ADJ_JINA_GATEWAY_APP_KEY'],
                                           'jina_ai', base_url=values['ADJ_AIHUB_GATEWAY_URL'] or None))
    return FallbackFetcher([jina, WebPageFetcher()])


class WebPageFetcher(FullTextFetcher):
    """Reuse existing public-target validation and bounded transport semantics.

    Production always uses pinned curl; injected transports must be offline mocks.
    Redirects are separately validated. No cookies, auth, proxy or user curl config.
    """
    def __init__(self, *, transport=None, resolver=socket.getaddrinfo, timeout_seconds=20,
                 maximum_bytes=2*1024*1024, maximum_redirects=2):
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError('Only offline mock transports may bypass pinned curl')
        if maximum_bytes > 2*1024*1024 or timeout_seconds > 20:
            raise ValueError('Web page budgets cannot exceed 2 MiB / 20 seconds')
        super().__init__(transport=transport, resolver=resolver, timeout_seconds=timeout_seconds,
                         maximum_bytes=maximum_bytes, maximum_redirects=maximum_redirects)

    def fetch(self, url, *, check_cancelled=lambda: None):
        deadline = time.monotonic() + self.timeout_seconds
        def resolve(host):
            replies = queue.Queue(maxsize=1)
            def worker():
                try:
                    try: value = self.resolver(host)
                    except TypeError: value = self.resolver(host, 443, type=socket.SOCK_STREAM)
                    replies.put((True, value))
                except Exception: replies.put((False, None))
            threading.Thread(target=worker, daemon=True).start()
            try: ok, value = replies.get(timeout=max(.001, min(2, deadline-time.monotonic())))
            except queue.Empty: raise ToolFailure('web_page_dns_timeout', 'Public DNS lookup timed out') from None
            if not ok: raise ToolFailure('web_page_dns', 'Public DNS lookup failed')
            return value
        for redirect in range(self.maximum_redirects+1):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure('web_page_timeout', 'Web page time budget exhausted')
            parsed = urlsplit(url)
            hostname = parsed.hostname or ''
            # Avoid parser/curl IDNA and percent-decoding disagreements that
            # could defeat a hostname-specific --resolve pin. ASCII IDNA
            # labels are accepted; callers must discover that exact URL.
            try: ipaddress.ip_address(hostname)
            except ValueError:
                if not re.fullmatch(r'[A-Za-z0-9.-]+', hostname):
                    raise ToolFailure('web_page_target', 'Web hostname must be an unambiguous ASCII DNS name')
            if _credential_target(parsed):
                raise ToolFailure('web_page_credentials', 'Credential-bearing web targets are denied')
            target = public_target(url, resolve)
            if any(ipaddress.ip_address(a).is_multicast or ipaddress.ip_address(a).is_reserved for a in target.addresses):
                raise ToolFailure('web_page_target', 'Web targets must resolve only to public unicast addresses')
            status, headers, body = self._request(target, deadline, check_cancelled)
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure('web_page_timeout', 'Web page time budget exhausted')
            if status in {301,302,303,307,308}:
                if redirect == self.maximum_redirects or not headers.get('location'):
                    raise ToolFailure('web_page_redirect', 'Web page redirect budget exhausted or location missing')
                url = urljoin(url, headers['location'])
                continue
            if status != 200:
                raise ToolFailure('web_page_http', f'Public web host returned HTTP {status}; no retry')
            media = headers.get('content-type','').split(';',1)[0].strip().lower()
            if media not in {'text/html','application/xhtml+xml'} or not body.strip():
                raise ToolFailure('web_page_media', 'Expected nonempty public HTML; PDFs and non-HTML are unsupported')
            if len(body) > self.maximum_bytes:
                raise ToolFailure('web_page_size', 'Web page byte limit exceeded')
            return WebPageResponse(body, url, datetime.now(timezone.utc).isoformat())
        raise ToolFailure('web_page_redirect', 'Web page redirect budget exhausted')

    def _request_with_curl(self, target, deadline):
        # Same validated-IP/SNI pattern as native.fulltext, with curl startup
        # configuration disabled and HTML Accept. Never invoke a shell.
        remaining = max(.1, deadline-time.monotonic())
        selected = next((a for a in target.addresses if ipaddress.ip_address(a).version == 4),target.addresses[0])
        address = f'[{selected}]' if ipaddress.ip_address(selected).version == 6 else selected
        with tempfile.TemporaryDirectory(prefix='adjagent-web-') as directory:
            body_path, header_path = Path(directory)/'body', Path(directory)/'headers'
            args = ['/usr/bin/curl','-q','--silent','--show-error','--no-location','--no-netrc',
                    '--proto','=https','--noproxy','*','--resolve',f'{target.hostname}:{target.port}:{address}',
                    '--connect-timeout',str(min(5,remaining)),'--max-time',str(remaining),
                    '--max-filesize',str(self.maximum_bytes),'--header','Accept-Encoding: identity',
                    '--header','Accept: text/html, application/xhtml+xml;q=0.9',
                    '--output',str(body_path),'--dump-header',str(header_path),'--write-out','%{http_code}',target.url]
            try: response = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=remaining+.5,check=False)
            except subprocess.TimeoutExpired: raise ToolFailure('web_page_timeout','Web page transport timeout') from None
            except OSError: raise ToolFailure('web_page_transport','Public web transport could not start') from None
            if response.returncode:
                raise ToolFailure('web_page_transport',f'Public web transport failed (exit {response.returncode}); no retry')
            status = response.stdout.decode('ascii',errors='strict')
            if len(status)!=3 or not status.isdigit(): raise ToolFailure('web_page_transport','Missing HTTP status')
            if body_path.stat().st_size > self.maximum_bytes or header_path.stat().st_size > 256*1024:
                raise ToolFailure('web_page_size','Web response byte limit exceeded')
            body=body_path.read_bytes();headers={}
            for line in header_path.read_text(encoding='iso-8859-1').splitlines()[1:]:
                if ':' in line:
                    k,v=line.split(':',1);headers[k.strip().lower()]=v.strip()
            if headers.get('content-encoding','identity').lower()!='identity':
                raise ToolFailure('web_page_encoding','Compressed responses are unsupported')
            return int(status),headers,body


class _Paragraphs(HTMLParser):
    EXCLUDE={'head','script','style','nav','footer','header','form','noscript','svg','template','iframe','canvas'}
    BLOCK={'p','div','section','article','main','li','h1','h2','h3','h4','h5','h6','pre','blockquote','tr','br'}
    VOID={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self):
        super().__init__(convert_charrefs=True);self.stack=[];self.parts=[];self.paragraphs=[]
    def flush(self):
        text=' '.join(''.join(self.parts).split());self.parts=[]
        if text:
            for start in range(0,len(text),1200): self.paragraphs.append(text[start:start+1200])
            if len(self.paragraphs)>10000: raise ValueError('HTML paragraph limit exceeded')
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs);style=attrs.get('style','').replace(' ','').lower()
        hidden=(tag in self.EXCLUDE or 'hidden' in attrs or attrs.get('aria-hidden','').lower()=='true'
                or 'display:none' in style or 'visibility:hidden' in style)
        if tag in self.BLOCK: self.flush()
        if tag not in self.VOID:
            self.stack.append((tag,hidden))
            if len(self.stack)>256: raise ValueError('HTML nesting limit exceeded')
    def handle_endtag(self,tag):
        if tag in self.BLOCK: self.flush()
        for index in range(len(self.stack)-1,-1,-1):
            if self.stack[index][0]==tag:
                del self.stack[index:];break
    def handle_data(self,data):
        if not any(hidden for _,hidden in self.stack): self.parts.append(data)


def html_paragraphs(text):
    parser=_Paragraphs();parser.feed(text);parser.close();parser.flush()
    if not parser.paragraphs: raise ValueError('No static visible HTML text extracted')
    return parser.paragraphs


def markdown_paragraphs(text):
    paragraphs=[]
    for block in re.split(r'\n\s*\n', text):
        clean=' '.join(block.split())
        for start in range(0,len(clean),1200): paragraphs.append(clean[start:start+1200])
        if len(paragraphs)>10000: raise ValueError('Markdown paragraph limit exceeded')
    if not paragraphs: raise ValueError('No text extracted from reader Markdown')
    return paragraphs


def register_web_access(service, *, fetcher=None):
    if fetcher is None:
        fetcher = (web_page_fetcher_from_environment(getattr(service, 'project_root', None), getattr(service, 'env_file', None))
                   or WebPageFetcher())
    def read(context,source_id=None,url=None):
        if source_id is not None:
            url=None
        if source_id is None and url is None:
            raise ValueError('Provide exactly one source_id or public HTTPS url')
        bound=service.native._bind(context)
        if source_id is not None:
            raw,source,_=_material(context,source_id,None)
            provenance=source['provenance']
            if (source['kind']!='web' or provenance.get('text_availability') not in {'search_snippet','title_only'}
                    or not provenance.get('response_artifact_id') or 'result_index' not in provenance):
                raise PermissionError('source_id must identify a same-task web_search discovery source')
            record=json.loads(raw)
            if record.get('url_kind')=='google_redirect_unresolved':
                raise ToolFailure('web_page_target','Search result is an unresolved Google redirect link; search again later')
            url=record.get('url')
            if not isinstance(url,str) or url!=provenance.get('url') or context.store.redact(url)!=url:
                raise PermissionError('Web discovery URL is missing, inconsistent or credential-bearing')
            locator={'locator_origin':'web_search_discovery','parent_source_id':source_id}
            title=source['title']
        else:
            if not isinstance(url,str) or not url.strip() or len(url)>4096 or context.store.redact(url)!=url:
                raise PermissionError('Explicit URL is missing, too large or credential-bearing')
            locator={'locator_origin':'model_supplied_public_url'}
            title='Public HTML: '+url[:400]
        response=fetcher.fetch(url,check_cancelled=context.check_cancelled)
        # UTF-8 only, fail closed instead of silently corrupting source text.
        text=response.body.decode('utf-8-sig')
        if response.media=='text/markdown':
            paragraphs,parser,scope,name=markdown_paragraphs(context.store.redact(text)),'jina_reader_v1',JINA_SCOPE,'web-page.md'
        else:
            paragraphs,parser,scope,name=html_paragraphs(context.store.redact(text)),'static_html_v1',SCOPE,'web-page.html'
        base={**locator,'url':response.url,'requested_url':url,'retrieved_at':response.received_at,
              'raw_sha256':hashlib.sha256(response.body).hexdigest(),'parser':parser,'limitations':[scope]}
        if response.fallback_from: base['fallback_from']=list(response.fallback_from)
        original=context.store.put_artifact(response.body,name,response.media,{**base,'text_availability':'full_text'},
                    session_id=context.session_id,run_id=context.run_id)
        projection={'paragraphs':[{'text':p} for p in paragraphs]}
        derived_provenance={**base,'raw_artifact_id':original['id'],'text_availability':'derived_full_text'}
        artifact=context.store.put_artifact(canonical_json(projection),'web-page-paragraphs.json','application/json',derived_provenance,
                    session_id=context.session_id,run_id=context.run_id)
        derived=bound.research.register_source(context.session_id,context.run_id,artifact_id=artifact['id'],kind='web',
                    title=title,provenance=derived_provenance,idempotency_key='web-page:'+context.call_id)
        data={'source_id':derived['id'],'raw_sha256':original['sha256'],'projection_sha256':artifact['sha256'],
              'locator_origin':locator['locator_origin'],
              'url':response.url[:400],'url_truncated':len(response.url)>400,'paragraph_count':len(paragraphs),
              'text_availability':'derived_full_text','scope':scope,'paragraphs':[],
              'read_next':'Use evidence_read with this source_id and paragraph selector, or query a literal phrase.'}
        for index,p in enumerate(paragraphs[:2]):
            excerpt=p.encode()[:500].decode('utf-8','ignore')
            data['paragraphs'].append({'selector':{'pointer':f'/paragraphs/{index}/text'},'text':excerpt,
                                      'characters':len(p),'shown':len(excerpt)})
        result=ToolResult(data=data)
        while len(canonical_json(result.to_dict()).encode())>=2300 and data['paragraphs']:
            data['paragraphs'].pop()
        if len(canonical_json(result.to_dict()).encode())>=2300: raise ValueError('Web locator exceeds context budget')
        return result
    service.catalog.register_many([(ToolSpec('web_page_read',
        'Fetch static public HTTPS HTML using exactly one source_id OR url. source_id must be a same-task web_search discovery source; url is a model-supplied public locator, not search discovery or evidence of source trust. Extract visible paragraph text with hash/provenance; use evidence_read on the returned source and selectors. No file paths, JS, images, OCR or PDF; snippets remain a separate source. Same public DNS/redirect checks and IP pinning; no credentials/cookies or retries on denial.',
        {'type':'object','properties':{'source_id':{'type':'string','minLength':1,'maxLength':128},
             'url':{'type':'string','minLength':1,'maxLength':4096}},
         'additionalProperties':False},network=True,version='2'),read)],secrets=getattr(fetcher,'secrets',()))
