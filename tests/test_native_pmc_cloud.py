"""Source-bound PMC Cloud supplements, fixed destinations, integrity and archive budgets."""
from hashlib import md5, sha256
from io import BytesIO
import json
import os
import stat
from types import SimpleNamespace
import zipfile

import httpx
from pypdf import PdfWriter
import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall, new_id
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.native.documents import DocumentReader
from adj_partner.native.fulltext import FullTextFetcher
from adj_partner.native.pmc_cloud import _archive_members, fetch_pmc_article
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor, ToolFailure


PMCID = 'PMC13211340'
VERSION = PMCID + '.1'
ATTACHMENT = 'vaccines-14-00385-s001.zip'
IDENTITY = {'pmcid': PMCID, 'doi': '10.3390/vaccines14050385', 'pmid': '42188758'}
JATS = (f'<article><front><article-meta><article-id pub-id-type="pmcid">{PMCID}</article-id>'
        f'<article-id pub-id-type="pmcid-ver">{VERSION}</article-id></article-meta></front>'
        '<body><fig><graphic href="figure.jpg"/></fig></body><back><supplementary-material>'
        f'<media href="{ATTACHMENT}"/></supplementary-material></back></article>').encode()


def _zip(entries):
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def _pdf():
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def _fixture():
    pdf = _pdf()
    members = [('nested/one.pdf', pdf), ('two.xml', b'<article><body><p>Fixture</p></body></article>'),
               ('notes.txt', b'fixture notes')]
    supplement = _zip(members)
    def uri(name, content):
        return f's3://pmc-oa-opendata/{VERSION}/{name}?md5=' + md5(content).hexdigest()
    manifest = {**IDENTITY, 'version': 1, 'is_pmc_openaccess': True, 'xml_url': uri(VERSION + '.xml', JATS),
                'media_urls': [uri('figure.jpg', b'figure'), uri(ATTACHMENT, supplement)]}
    versions = [1]
    paths = {f'/{VERSION}/{VERSION}.xml': JATS, f'/{VERSION}/{ATTACHMENT}': supplement}
    requests = []
    def handle(request):
        requests.append(request)
        assert request.url.host == 'pmc-oa-opendata.s3.amazonaws.com'
        path = request.url.path
        if path == '/':
            assert request.url.params['prefix'] == PMCID + '.'
            prefixes = ''.join(f'<CommonPrefixes><Prefix>{PMCID}.{v}/</Prefix></CommonPrefixes>' for v in versions)
            listing = (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>pmc-oa-opendata</Name>'
                       f'<Prefix>{PMCID}.</Prefix><Delimiter>/</Delimiter><IsTruncated>false</IsTruncated>'
                       f'{prefixes}</ListBucketResult>').encode()
            return httpx.Response(200, content=listing)
        if path.endswith('.json'):
            return httpx.Response(200, json=manifest)
        return httpx.Response(200, content=paths[path])
    fetcher = FullTextFetcher(transport=httpx.MockTransport(handle), resolver=lambda _: ['8.8.8.8'])
    return SimpleNamespace(fetcher=fetcher, manifest=manifest, archive=supplement, paths=paths, requests=requests,
                           members=members, versions=versions)


def _scope(tmp_path, fetcher):
    store = Store(tmp_path / 'state')
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research, documents=DocumentReader(), fulltext=fetcher)
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allow_network=True, allowed_tools=catalog.names(),
                                                enabled_tools=catalog.names()).to_dict())['id']
    task = store.create_task(session, 'Read PMC supplement')['id']
    run = store.create_run(session, task, Budget().to_dict())['id']
    candidate = dict(IDENTITY, acquisition=[{'kind': 'pmc_cloud', 'provider': 'pmc_cloud', 'pmcid': PMCID}])
    artifact = store.put_artifact(json.dumps({'candidates': [candidate]}), 'resolution.json', 'application/json',
                                  session_id=session, run_id=run)
    source = research.register_source(session, run, artifact_id=artifact['id'], kind='literature', title='Article',
        provenance={'provider': 'scholarly_resolution', 'text_availability': 'title_only'}, idempotency_key='resolution')
    executor = ToolExecutor(store, catalog)
    def call(tool, **arguments):
        result = executor.execute(ToolCall(new_id('call'), tool, arguments), session, run)
        assert result.status == 'ok', result.to_dict()
        return result.data
    return SimpleNamespace(call=call, store=store, research=research, source=source, session=session, run=run,
                           executor=executor, task=task)


def test_inventory_excludes_figures_and_every_readable_zip_member_is_selectable(tmp_path):
    f = _fixture()
    s = _scope(tmp_path, f.fetcher)
    listing = s.call('pmc_attachment', source_id=s.source['id'])
    assert [item['filename'] for item in listing['attachments']] == [ATTACHMENT]
    assert all(not str(request.url).endswith(ATTACHMENT) for request in f.requests)
    selected = s.call('pmc_attachment', source_id=s.source['id'], action='fetch',
                      attachment_id=listing['attachments'][0]['attachment_id'])
    assert s.store.read_artifact(selected['archive_artifact_id']) == f.archive
    assert selected['archive_sha256'] == sha256(f.archive).hexdigest()
    assert [member['readable'] for member in selected['members']] == [True, True, False]
    for row, (path, content) in zip(selected['members'], f.members, strict=True):
        assert (row['name'], row['bytes'], row['sha256']) == (path, len(content), sha256(content).hexdigest())
        if row['readable']:
            source = s.research.get_source(s.session, s.run, row['source_id'])
            assert s.store.read_artifact(source['artifact_id']) == content
            assert source['provenance']['document_scope'] == 'supplement'
            assert source['provenance']['archive_member'] == path
            assert source['provenance']['archive_sha256'] == selected['archive_sha256']
            read = s.call('document_read', source_id=row['source_id'])
            assert read['matches'][0]['locator'] in {'page:1', 'section:1/paragraph:1'}
            assert read['document_scope'] == 'supplement'
    article = fetch_pmc_article(f.fetcher, PMCID, identity=IDENTITY, check_cancelled=lambda: None)
    assert article.body == JATS and article.acquisition['provider'] == 'pmc_cloud'
    assert article.acquisition['manifest_sha256'] == listing['manifest_sha256']


def test_raw_parsed_and_listed_sources_retain_their_exact_version(tmp_path):
    f = _fixture()
    s = _scope(tmp_path, f.fetcher)
    raw = s.call('fulltext_fetch', source_id=s.source['id'])
    parsed = s.call('document_read', source_id=raw['source_id'])
    listing = s.call('pmc_attachment', source_id=s.source['id'])
    f.versions.append(2)
    for source_id in (raw['source_id'], parsed['source_id']):
        pinned = s.call('pmc_attachment', source_id=source_id)
        assert pinned['pmc_version'] == 1 and pinned['attachments'] == listing['attachments']
    fetched = s.call('pmc_attachment', source_id=s.source['id'], action='fetch',
                     attachment_id=listing['attachments'][0]['attachment_id'])
    assert fetched['pmc_version'] == 1
    assert not any(PMCID + '.2/' in str(req.url) for req in f.requests)


def test_cloud_refuses_redirect_without_contacting_location():
    calls = []
    def redirect(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={'location': 'https://redirected.example/file'})
    fetcher = FullTextFetcher(transport=httpx.MockTransport(redirect), resolver=lambda _: ['8.8.8.8'])
    with pytest.raises(ToolFailure, match='Redirect') as error:
        fetch_pmc_article(fetcher, PMCID, check_cancelled=lambda: None)
    assert error.value.code == 'pmc_cloud' and len(calls) == 1


@pytest.mark.parametrize('key,value', [('version', 2), ('pmcid', 'PMC1'), ('doi', '10.1000/wrong'),
    ('is_pmc_openaccess', False), ('xml_url', f's3://evil/{VERSION}/{VERSION}.xml?md5=' + '0'*32),
    ('xml_url', 'https://untrusted.example/article.xml'),
    ('xml_url', f's3://pmc-oa-opendata/{VERSION}/../{VERSION}.xml?md5=' + '0'*32)])
def test_article_identity_oa_and_remote_keys_verified_before_downloading(key, value):
    f = _fixture()
    f.manifest[key] = value
    with pytest.raises(ToolFailure):
        fetch_pmc_article(f.fetcher, PMCID, identity=IDENTITY, check_cancelled=lambda: None)
    assert len(f.requests) == 2


def test_checksum_and_jats_identity_checked_even_for_official_url():
    f = _fixture()
    path = f'/{VERSION}/{VERSION}.xml'
    f.paths[path] = JATS.replace(PMCID.encode(), b'PMC1')
    with pytest.raises(ToolFailure, match='checksum'):
        fetch_pmc_article(f.fetcher, PMCID, check_cancelled=lambda: None)
    f.manifest['xml_url'] = f.manifest['xml_url'].split('?')[0] + '?md5=' + md5(f.paths[path]).hexdigest()
    with pytest.raises(ToolFailure, match='identity'):
        fetch_pmc_article(f.fetcher, PMCID, check_cancelled=lambda: None)


@pytest.mark.parametrize('entries,reason', [([('../escape.pdf', b'%PDF-')], 'path'),
    ([('x.pdf', b'%PDF-'), ('X.pdf', b'%PDF-')], 'colliding'),
    ([(str(i), b'x') for i in range(33)], 'directory'),
    ([('x.pdf', b'x' * 65536)], 'oversized'), ([('fake.pdf', b'<!DOCTYPE html>')], 'signature')])
def test_zip_rejects_unsafe_or_unbounded_material(entries, reason):
    with pytest.raises(ToolFailure, match=reason):
        _archive_members(_zip(entries), lambda: None)


def test_zip_rejects_symbolic_links():
    entry = zipfile.ZipInfo('link.pdf')
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ToolFailure, match='symlinks'):
        _archive_members(_zip([(entry, b'target')]), lambda: None)


def test_attachment_denies_wrong_run_unknown_provider_and_free_url_before_network(tmp_path):
    f = _fixture()
    s = _scope(tmp_path, f.fetcher)
    source = s.research.register_source(s.session, s.run, artifact_id=s.source['artifact_id'], kind='literature',
        title='Unverified', provenance={'provider': 'anything', 'pmcid': PMCID}, idempotency_key='wrong')
    run2 = s.store.create_run(s.session, s.task, Budget().to_dict())['id']
    for run, args in [(run2, {'source_id': s.source['id']}), (s.run, {'source_id': source['id']}),
            (s.run, {'source_id': s.source['id'], 'attachment_id': 'https://evil/private', 'action': 'fetch'})]:
        result = s.executor.execute(ToolCall(new_id('call'), 'pmc_attachment', args), s.session, run)
        assert result.status != 'ok'
    assert not f.requests


@pytest.mark.skipif(os.getenv('ADJ_PMC_LIVE') != '1', reason='Opt-in live PMC Cloud verification')
def test_live_public_bounded_zip_and_six_supplement_pages(tmp_path):
    s = _scope(tmp_path, FullTextFetcher(timeout_seconds=60))
    listing = s.call('pmc_attachment', source_id=s.source['id'])
    item, = [row for row in listing['attachments'] if row['filename'] == ATTACHMENT]
    fetched = s.call('pmc_attachment', source_id=s.source['id'], action='fetch', attachment_id=item['attachment_id'])
    archive = s.store.read_artifact(fetched['archive_artifact_id'])
    member, = fetched['members']
    read = s.call('document_read', source_id=member['source_id'])
    assert read['document_type'] == 'pdf' and [row['locator'] for row in read['matches']] == [f'page:{i}' for i in range(1, 7)]
    page = s.call('document_read', source_id=read['source_id'], page=1)['matches'][0]
    evidence = s.research.select_evidence(s.session, s.run, read['source_id'], page['selector'])
    assert evidence['value'] == page['text'] and len(page['text']) > 100
    print(json.dumps({'pmcid': PMCID, 'version': listing['pmc_version'], 'filename': item['filename'],
        'archive_bytes': len(archive), 'archive_sha256': fetched['archive_sha256'], 'member': member,
        'pages': len(read['matches']), 'page1_characters': len(evidence['value']), 'page1_selector': page['selector'],
        'page1_preview': evidence['value'][:180]}))
