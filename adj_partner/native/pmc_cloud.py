"""Versioned open-access PMC Cloud objects and source-bound supplementary material."""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5, sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
import struct
import time
from urllib.parse import quote, urlencode, urlsplit
import zipfile
import zlib

from lxml import etree

from ..contracts import ToolResult
from ..tools import ToolFailure, ToolSpec
from .fulltext import FullTextFetcher, FullTextResponse, public_target
from .jats import inert_jats_content, matched_article_ids
from .sequence_inventory import _member_path


BUCKET = 'pmc-oa-opendata'
BASE = 'https://' + BUCKET + '.s3.amazonaws.com/'
PMCID = re.compile(r'PMC[1-9][0-9]{0,11}')
FILENAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.()+, -]{0,239}')
SELECTOR = re.compile(r'v([1-9][0-9]{0,5})_[0-9a-f]{24}')
S3_NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
MAX_METADATA = 512 * 1024
MAX_OBJECT = 32 * 1024 * 1024
MAX_MEMBERS = 32
MAX_MEMBER = 16 * 1024 * 1024
READABLE = {'.pdf': 'application/pdf', '.xml': 'application/xml', '.nxml': 'application/xml'}


def _error(message):
    return ToolFailure('pmc_cloud', message)


def _xml(body, *, jats=False):
    try:
        # Only JATS permits an inert external article declaration. Reject other DTDs/entities.
        if not jats and (b'\x00' in body or re.search(br'<!\s*(?:DOCTYPE|ENTITY)\b', body, re.I)):
            raise ValueError('Unexpected declaration in listing')
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True,
                                 remove_comments=True, huge_tree=False, recover=False)
        root = etree.fromstring(inert_jats_content(body) if jats else body, parser)
    except (ValueError, etree.XMLSyntaxError) as exc:
        raise _error('Malformed or unsafe PMC XML') from exc
    if jats and (etree.QName(root).localname != 'article' or sum(1 for _ in root.iter()) > 200_000):
        raise _error('Expected a bounded JATS article')
    return root


def _jats_identity(root, identity, *, require_pmcid=False):
    try:
        matched, version = matched_article_ids(root, identity)
    except ValueError as exc:
        raise _error(str(exc)) from exc
    if require_pmcid and 'pmcid' not in matched:
        raise _error('PMC article XML lacks a PMCID')
    return version


def _request(fetcher, url, limit, deadline, check_cancelled):
    # URLs are constructed only after host, prefix and object-name validation. No redirects.
    check_cancelled()
    if time.monotonic() >= deadline:
        raise ToolFailure('fulltext_timeout', 'PMC Cloud request budget exhausted')
    constrained = FullTextFetcher(transport=fetcher.transport, resolver=fetcher.resolver,
        timeout_seconds=fetcher.timeout_seconds, maximum_bytes=min(limit, fetcher.maximum_bytes), maximum_redirects=0)
    if not url.startswith(BASE):
        raise _error('PMC Cloud object must use the official bucket host')
    try:
        target = public_target(url, constrained.resolver)
        status, headers, body = constrained._request(target, deadline, check_cancelled)
    except OSError as exc:
        raise ToolFailure('fulltext_transport', 'PMC Cloud public host could not be reached') from exc
    check_cancelled()
    if 300 <= status < 400:
        raise _error('Redirect from the official PMC bucket was refused')
    if status != 200:
        raise ToolFailure('pmc_cloud_http', f'PMC Cloud returned HTTP {status}')
    return body, headers


@dataclass(frozen=True)
class _Object:
    name: str
    url: str
    md5: str


class _Article:
    def __init__(self, fetcher, identity, check_cancelled, version=None, deadline=None):
        if not isinstance(fetcher, FullTextFetcher):
            raise _error('The host must configure a bounded full-text fetcher for PMC Cloud')
        self.fetcher = fetcher
        self.check_cancelled = check_cancelled
        self.deadline = deadline if deadline is not None else time.monotonic() + fetcher.timeout_seconds
        self.pmcid = identity['pmcid']
        if not isinstance(self.pmcid, str) or not PMCID.fullmatch(self.pmcid):
            raise _error('Invalid PMCID')
        query = urlencode({'list-type': '2', 'prefix': self.pmcid + '.', 'delimiter': '/', 'max-keys': 65})
        listing = _xml(self.get(BASE + '?' + query, MAX_METADATA)[0])
        prefix = self.pmcid + '.'
        if (listing.tag != S3_NS + 'ListBucketResult' or listing.findtext(S3_NS + 'Name') != BUCKET
                or listing.findtext(S3_NS + 'Prefix') != prefix or listing.findtext(S3_NS + 'Delimiter') != '/'
                or listing.findtext(S3_NS + 'IsTruncated') != 'false'):
            raise _error('Incomplete or mismatched PMC Cloud version listing')
        prefixes = listing.findall(f'{S3_NS}CommonPrefixes/{S3_NS}Prefix')
        if not prefixes:
            raise ToolFailure('pmc_cloud_unavailable', 'No publicly listed PMC version for this article')
        if len(prefixes) > 64:
            raise _error('Article exceeds the version listing limit')
        versions = set()
        for node in prefixes:
            match = re.fullmatch(re.escape(prefix) + r'([1-9][0-9]{0,5})/', node.text or '')
            if not match:
                raise _error('Unrelated prefix in PMC Cloud version listing')
            versions.add(int(match[1]))
        self.version = max(versions) if version is None else version
        if type(self.version) is not int or self.version not in versions:
            raise _error('Selected article version is unavailable')
        self.prefix = f'{self.pmcid}.{self.version}'
        self.manifest_url = BASE + self.prefix + '/' + self.prefix + '.json'
        content = self.get(self.manifest_url, MAX_METADATA)[0]
        self.manifest_hash = sha256(content).hexdigest()
        try:
            data = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise _error('Malformed PMC version metadata') from exc
        if (not isinstance(data, dict) or data.get('pmcid') != self.pmcid or type(data.get('version')) is not int
                or data['version'] != self.version):
            raise _error('PMC metadata must match the article and version')
        if data.get('is_pmc_openaccess') is not True:
            raise ToolFailure('pmc_cloud_unavailable', 'PMC metadata does not confirm open-access status')
        for key in ('doi', 'pmid'):
            if identity.get(key) and data.get(key) and str(identity[key]).lower() != str(data[key]).lower():
                raise _error(f'PMC metadata conflicts with the source {key}')
        self.data = data

    def get(self, url, limit=MAX_OBJECT):
        return _request(self.fetcher, url, limit, self.deadline, self.check_cancelled)

    def object(self, uri, *, expected=None):
        if not isinstance(uri, str) or len(uri) > 1024 or any(ord(c) < 32 or ord(c) == 127 for c in uri):
            raise _error('Malformed PMC metadata object URI')
        try:
            parsed = urlsplit(uri)
        except ValueError as exc:
            raise _error('Malformed PMC metadata object URI') from exc
        path_prefix = '/' + self.prefix + '/'
        name = parsed.path.removeprefix(path_prefix)
        checksum = re.fullmatch(r'md5=([a-f0-9]{32})', parsed.query)
        if (parsed.scheme != 's3' or parsed.netloc != BUCKET or parsed.fragment or not checksum
                or not parsed.path.startswith(path_prefix) or not FILENAME.fullmatch(name)
                or (expected is not None and name != expected)):
            raise _error('Unverified PMC object key, bucket, version or checksum')
        return _Object(name, BASE + self.prefix + '/' + quote(name, safe=''), checksum[1])

    def download(self, item):
        body, headers = self.get(item.url)
        if md5(body, usedforsecurity=False).hexdigest() != item.md5:
            raise _error('Downloaded object does not match the official PMC manifest checksum')
        return body, headers

    def jats(self):
        item = self.object(self.data.get('xml_url'), expected=self.prefix + '.xml')
        body, _ = self.download(item)
        root = _xml(body, jats=True)
        version = _jats_identity(root, self.data, require_pmcid=True)
        if version is not None and version != self.version:
            raise _error('PMC article XML version disagrees with the manifest')
        return item, body, root

    def provenance(self):
        fields = {'pmcid': self.pmcid, 'pmc_version': self.version, 'manifest_url': self.manifest_url,
                  'manifest_sha256': self.manifest_hash}
        if isinstance(self.data.get('license_code'), str):
            fields['license'] = self.data['license_code'][:256]
        return fields

    def attachments(self):
        if not self.data.get('xml_url'):
            raise _error('PMC has no article XML to distinguish supplements from figure media')
        xml_item, xml_body, root = self.jats()
        media = self.data.get('media_urls')
        if not isinstance(media, list) or len(media) > 2048:
            raise _error('Missing or oversized PMC media inventory')
        items = {}
        for uri in media:
            item = self.object(uri)
            if item.name in items:
                raise _error('Duplicate media key in PMC metadata')
            items[item.name] = item
        targets = set()
        supplements = root.xpath("//*[local-name()='supplementary-material' or local-name()='inline-supplementary-material']")
        if len(supplements) > 256:
            raise _error('Article exceeds the supplementary-material element limit')
        unresolved = 0
        for supplement in supplements:
            refs = supplement.xpath('descendant-or-self::*/@*[local-name()="href"]')
            matched = {str(ref) for ref in refs if str(ref) in items}
            targets.update(matched)
            if not matched:
                unresolved += 1
        if len(targets) > 64:
            raise _error('Supplement inventory exceeds 64 material objects')
        attachments = {}
        for name in sorted(targets):
            item = items[name]
            key = sha256(f'{item.url}\n{item.md5}'.encode()).hexdigest()[:24]
            attachments[f'v{self.version}_{key}'] = item
        return attachments, unresolved, {'article_xml_url': xml_item.url, 'article_xml_sha256': sha256(xml_body).hexdigest()}


def fetch_pmc_article(fetcher, pmcid, *, check_cancelled, identity=None, deadline=None):
    """Retrieve manifest-bound article text for a PMCID already resolved by the host."""
    if identity and identity.get('pmcid', pmcid) != pmcid:
        raise _error('Conflicting source PMCID')
    article = _Article(fetcher, {**(identity or {}), 'pmcid': pmcid}, check_cancelled, deadline=deadline)
    if article.data.get('xml_url'):
        item, body, _ = article.jats()
        kind, media_type = 'jats', 'application/xml'
    elif article.data.get('pdf_url'):
        item = article.object(article.data['pdf_url'], expected=article.prefix + '.pdf')
        body, headers = article.download(item)
        kind, media_type = 'pdf', fetcher._validate_body(body, headers, {'kind': 'pdf'})
    else:
        raise ToolFailure('pmc_cloud_unavailable', 'No article XML or PDF listed in PMC metadata')
    acquisition = dict(article.provenance(), kind=kind, provider='pmc_cloud', is_oa=True, url=item.url,
                       version=article.version, manifest_md5=item.md5)
    return FullTextResponse(body, item.url, media_type, datetime.now(timezone.utc).isoformat(), acquisition)


def _parent_article(bound, context, source_id):
    """Follow only host source types with the expected parent chain, verifying actual bytes."""
    def load(source_id):
        source = bound.research.get_source(context.session_id, context.run_id, source_id)
        if source['run_id'] != context.run_id or source['kind'] != 'literature':
            raise PermissionError('PMC attachments require a current-run literature source')
        return source, bound.research._verify_source(source)

    original, body = load(source_id)
    source = original
    prov = source['provenance']
    if prov.get('document_scope') == 'supplement':
        raise PermissionError('Use the host article source for supplementary material')
    if prov.get('provider') == 'document_parser':
        if (prov.get('document_type') not in {'pdf', 'jats', 'bioc'} or prov.get('text_availability') != 'derived_full_text'
                or not prov.get('parent_source_id')):
            raise PermissionError('Expected a host-parsed article full-text source')
        source, body = load(prov['parent_source_id'])
        prov = source['provenance']
        if prov.get('provider') == 'scholarly_resolution':
            raise PermissionError('Parsed article must originate from host-acquired full text')
    pmcid, version, acquisition = None, None, None
    if prov.get('provider') != 'scholarly_resolution':
        acquisition = prov.get('acquisition')
        if (prov.get('provider') not in {'europe_pmc', 'pmc_cloud', 'ncbi_bioc', 'unpaywall'}
                or prov.get('text_availability') != 'full_text' or not isinstance(acquisition, dict)
                or acquisition.get('provider') != prov['provider'] or not prov.get('parent_source_id')
                or prov.get('document_scope') == 'supplement'):
            raise PermissionError('Expected an article resolution or host-acquired article full text')
        pmcid = prov.get('pmcid')
        if not isinstance(pmcid, str) or not PMCID.fullmatch(pmcid):
            raise _error('Article full-text source lacks a bound PMCID')
        version = prov.get('pmc_version', acquisition.get('version'))
        if source['media_type'] == 'application/xml':
            xml_version = _jats_identity(_xml(body, jats=True), prov)
            if version is not None and xml_version is not None and version != xml_version:
                raise _error('Stored article version disagrees with article XML')
            if xml_version is not None:
                version = xml_version
        source, body = load(prov['parent_source_id'])
    if source['provenance'].get('provider') != 'scholarly_resolution' or source['media_type'] != 'application/json':
        raise PermissionError('Missing the host article resolution for this source')
    try:
        candidates = json.loads(body)['candidates']
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        raise _error('Malformed article resolution') from exc
    if not isinstance(candidates, list):
        raise _error('Malformed article resolution candidates')
    choices = [c for c in candidates if isinstance(c, dict) and (pmcid is None or c.get('pmcid') == pmcid)]
    if len(choices) != 1 or not isinstance(choices[0].get('pmcid'), str) or not PMCID.fullmatch(choices[0]['pmcid']):
        raise _error('A single article with an unambiguous resolved PMCID is required')
    identity = {key: choices[0][key] for key in ('pmcid', 'pmid', 'doi') if choices[0].get(key)}
    if original['provenance'].get('pmcid', identity['pmcid']) != identity['pmcid']:
        raise _error('Source and resolved article identifiers disagree')
    if acquisition is not None:
        options = choices[0].get('acquisition', [])
        if not isinstance(options, list) or not any(
                isinstance(option, dict) and option.get('provider') == acquisition['provider'] and (
                    option.get('kind') == 'pmc_cloud' and acquisition['provider'] == 'pmc_cloud' and option.get('pmcid') == pmcid
                    or option.get('kind') == acquisition.get('kind') and option.get('url') == acquisition.get('url'))
                for option in options):
            raise _error('Full-text acquisition is not bound to the resolved article')
    return original, identity, version


def _member_type(name, body):
    media = READABLE.get(PurePosixPath(name).suffix.lower())
    if media == 'application/pdf' and not body.startswith(b'%PDF-'):
        raise _error('Supplement labelled PDF does not have a PDF signature')
    if media == 'application/xml':
        _xml(body, jats=True)
    return media


def _archive_members(body, check_cancelled):
    """Read ordinary ZIP files into bounded memory; never extract or recurse into members."""
    # Preflight the central directory before ZipFile allocates one object per entry.
    end = body.rfind(b'PK\x05\x06', max(0, len(body) - 65557))
    if end < 0 or end + 22 > len(body):
        raise _error('Malformed supplement ZIP directory')
    disk, cd_disk, disk_entries, entries, cd_size, cd_start, comment = struct.unpack_from('<4H2LH', body, end + 4)
    if (disk or cd_disk or disk_entries != entries or not 1 <= entries <= MAX_MEMBERS
            or cd_size > 128 * 1024 or cd_start + cd_size != end or end + 22 + comment != len(body)):
        raise _error('Unsupported or oversized supplement ZIP directory')
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            infos = archive.infolist()
            if len(infos) != entries or sum(i.file_size for i in infos) > MAX_OBJECT:
                raise _error('Supplement ZIP exceeds aggregate size or member limit')
            members, seen = [], set()
            for info in infos:
                check_cancelled()
                name = _member_path(info, seen)
                if (len(name) > 512 or info.file_size > MAX_MEMBER or (info.is_dir() and info.file_size)
                        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or info.file_size > max(1024, 200 * info.compress_size)):
                    raise _error('Unsupported or oversized supplement ZIP member')
                if info.is_dir():
                    continue
                with archive.open(info) as stream:
                    content = bytearray()
                    while chunk := stream.read(65536):
                        check_cancelled()
                        content.extend(chunk)
                        if len(content) > min(info.file_size, MAX_MEMBER):
                            raise _error('Supplement ZIP member exceeds its byte budget')
                if len(content) != info.file_size:
                    raise _error('Supplement ZIP member size mismatch')
                raw = bytes(content)
                members.append((name, raw, _member_type(name, raw)))
            return members
    except (zipfile.BadZipFile, ValueError, zlib.error, NotImplementedError) as exc:
        raise _error(f'Invalid supplement ZIP: {exc}') from exc


def _attachment(native, context, source_id, action='list', attachment_id=None):
    if (action == 'fetch') != (attachment_id is not None) or action not in {'list', 'fetch'}:
        raise _error('List supplements first; fetch requires an attachment_id from that list')
    bound = native._bind(context)
    parent, identity, version = _parent_article(bound, context, source_id)
    if attachment_id is not None:
        match = SELECTOR.fullmatch(attachment_id)
        if not match or (version is not None and int(match[1]) != version):
            raise _error('Invalid or mismatched attachment version')
        version = int(match[1])
    article = _Article(bound.fulltext, identity, context.check_cancelled, version)
    available, unresolved, xml_provenance = article.attachments()
    common = {**article.provenance(), **xml_provenance, 'parent_article_source_id': source_id}
    if action == 'list':
        inventory = [{'attachment_id': key, 'filename': obj.name,
                      'fetch_supported': PurePosixPath(obj.name).suffix.lower() in {'.zip', *READABLE}}
                     for key, obj in available.items()]
        return ToolResult(data={**common, 'attachments': inventory, 'unresolved_supplement_elements': unresolved,
            'scope': 'Versioned PMC media referenced by JATS supplementary-material elements',
            'summary': {'attachments': inventory, 'unresolved_supplement_elements': unresolved}})
    item = available.get(attachment_id)
    if item is None:
        raise _error('Selected attachment does not belong to this versioned article inventory')
    suffix = PurePosixPath(item.name).suffix.lower()
    is_zip = suffix == '.zip'
    if not is_zip and suffix not in READABLE:
        raise _error('This attachment format cannot be ingested; supported types are ZIP, PDF, and JATS XML')
    body, _ = article.download(item)
    members = _archive_members(body, context.check_cancelled) if is_zip else [(item.name, body, _member_type(item.name, body))]
    scope = {'session_id': context.session_id, 'run_id': context.run_id}
    provenance = {**common, **identity, 'parent_source_id': source_id, 'provider': 'pmc_cloud_attachment',
        'document_scope': 'supplement', 'attachment': item.name, 'attachment_id': attachment_id,
        'url': item.url, 'manifest_md5': item.md5, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'limitations': ['This source contains supplementary material associated with the parent article.']}
    context.check_cancelled()
    archive = None
    if is_zip:
        archive = context.store.put_artifact(body, item.name, 'application/zip', provenance, **scope)
    artifacts = [archive] if archive else []
    results = []
    for name, raw, media_type in members:
        digest = sha256(raw).hexdigest()
        entry = {'name': name, 'bytes': len(raw), 'sha256': digest, 'readable': bool(media_type)}
        if media_type:
            fields = dict(provenance, text_availability='full_text', member_sha256=digest)
            if archive:
                fields.update(archive_artifact_id=archive['id'], archive_sha256=archive['sha256'], archive_member=name)
            artifact = context.store.put_artifact(raw, PurePosixPath(name).name, media_type, fields, **scope)
            source = bound.research.register_source(**scope, artifact_id=artifact['id'], kind='literature',
                title=(f'Supplement {name} — ' + parent['title'])[:2000], provenance=fields,
                idempotency_key='pmc-supplement:' + artifact['id'])
            entry.update(media_type=media_type, source_id=source['id'], artifact_id=artifact['id'])
            artifacts.append(artifact)
        results.append(entry)
    summary = {'attachment': item.name, 'members': [{k: row[k] for k in ('name', 'readable', 'source_id') if k in row}
                                                    for row in results]}
    data = {**common, 'attachment': item.name, 'url': item.url, 'attachment_sha256': sha256(body).hexdigest(),
            'members': results, 'summary': summary}
    if archive:
        data.update(archive_artifact_id=archive['id'], archive_sha256=archive['sha256'])
    return ToolResult(data=data, artifacts=artifacts)


def register_pmc_attachments(native, catalog):
    from .capabilities import ID, schema
    spec = ToolSpec('pmc_attachment',
        'List supplements for a current-run resolved PMC article, article full text or parsed article. '
        'Fetch a listed attachment_id to ingest its original PDF, JATS or bounded ZIP. '
        'Use each returned readable member source_id with document_read; page locators refer to the supplement.',
        schema({'source_id': ID, 'action': {'enum': ['list', 'fetch']},
                'attachment_id': {'type': 'string', 'pattern': '^' + SELECTOR.pattern + '$'}}, ['source_id']), network=True)
    catalog.register(spec, lambda context, **arguments: _attachment(native, context, **arguments))
