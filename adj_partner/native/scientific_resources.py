"""Bounded public database retrieval; records are not computed/verified findings.

The host may inject an HTTP transport for fixtures. Tool inputs never configure
an origin, local file, credentials, retries, redirects or arbitrary API path.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import threading
import time

import httpx

from ..contracts import ToolResult
from ..tools import ToolFailure
from ..tools.catalog import canonical_json

CHEMBL = 'https://www.ebi.ac.uk/chembl/api/data/'
REACTOME = 'https://reactome.org/ContentService/data/query/'
NCBI = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 16000
_NCBI_LOCK = threading.Lock()
_NCBI_LAST_REQUEST = 0.0

LIMITATIONS = {
    'ChEMBL': ['Database compound and assay records only; reported activity is assay-specific and does not establish efficacy, safety or a mechanism in the user setting.'],
    'Reactome': ['Curated or inferred pathway content only; pathway membership does not demonstrate compound modulation, efficacy or experimental validation.'],
    'GEO': ['Candidate dataset metadata only; no expression matrices were retrieved or analyzed, cohort independence checked, or independent replication established.'],
}


@dataclass(frozen=True)
class RawResponse:
    request_url: str
    received_at: str
    raw: bytes = field(repr=False)


@dataclass(frozen=True)
class ResourceResponse:
    provider: str
    records: list[dict]
    hit_count: int
    responses: list[RawResponse]


def _query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError('Query must be 1..500 characters')
    canonical_json(query)
    return query.strip()


def _limit(value):
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError('max_results must be an integer in 1..10')


def _identity(value, pattern):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError('Invalid database identifier')
    return value


def _failure(message):
    return ToolFailure('scientific_resource_response', message)


def _record(provider, identity, title, url, fields, scope):
    if not isinstance(title, str) or not title.strip() or len(title) > 2000:
        raise _failure('Missing or oversized database record title')
    record = {**fields, 'provider': provider, 'id': identity, 'title': title, 'url': url,
              'text_availability': 'database_record', 'evidence_scope': scope}
    if len(canonical_json(record).encode()) > MAX_RECORD_BYTES:
        raise _failure('Database record exceeds 16 KB normalized limit; no partial record imported')
    return record


def _fields(body, names):
    return {key: body[key] for key in names.split() if key in body}


class ScientificResources:
    def __init__(self, *, transport=None, timeout_seconds=20):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30):
            raise ValueError('Resource timeout must be finite and in (0,30] seconds')
        self.transport, self.timeout_seconds = transport, timeout_seconds

    def _checkpoint(self, check_cancelled):
        deadline = time.monotonic() + self.timeout_seconds
        def checkpoint():
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure('scientific_resource_timeout', 'Resource elapsed response budget exhausted')
        return checkpoint

    def _get(self, url, params, checkpoint):
        checkpoint()
        if not (url.startswith(CHEMBL) or url.startswith(REACTOME) or url.startswith(NCBI)):
            raise ValueError('Resource URL is not a fixed supported API origin')
        if url.startswith(NCBI):
            # Process-wide public NCBI pacing, also across the search/summary pair.
            global _NCBI_LAST_REQUEST
            while True:
                checkpoint()
                with _NCBI_LOCK:
                    delay = .35 - (time.monotonic() - _NCBI_LAST_REQUEST)
                    if delay <= 0:
                        _NCBI_LAST_REQUEST = time.monotonic()
                        break
                time.sleep(min(.05, delay))
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(min(5, self.timeout_seconds)),
                              follow_redirects=False, trust_env=False) as client:
                with client.stream('GET', url, params=params, headers={
                        'Accept': 'application/json', 'Accept-Encoding': 'identity',
                        'User-Agent': 'AdjPartner-PublicResources/1.0'}) as response:
                    checkpoint()
                    if response.status_code != 200:
                        raise ToolFailure('scientific_resource_http', f'HTTP {response.status_code}; no redirects or retries followed')
                    if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                        raise ToolFailure('scientific_resource_encoding', 'Compressed response rejected')
                    length = response.headers.get('content-length')
                    if length is not None and (len(length) > 10 or not length.isascii()
                            or not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                        raise ToolFailure('scientific_resource_size', 'Invalid or oversized response length')
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        checkpoint()
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ToolFailure('scientific_resource_size', 'Response exceeds 2 MiB')
                        chunks.append(chunk)
                    raw = b''.join(chunks)
                    result = RawResponse(str(response.request.url), datetime.now(timezone.utc).isoformat(), raw)
        except httpx.HTTPError as exc:
            raise ToolFailure('scientific_resource_transport', f'Request failed without retry: {type(exc).__name__}') from exc
        checkpoint()
        try:
            body = json.loads(raw)
            canonical_json(body)
            if not isinstance(body, dict) or body.get('error') or body.get('ERROR'):
                raise ValueError('Expected successful JSON object')
        except (ValueError, TypeError, RecursionError) as exc:
            raise _failure('Malformed database JSON response') from exc
        return body, result

    @staticmethod
    def _page(body, key, max_results):
        records = body.get(key)
        meta = body.get('page_meta')
        count = meta.get('total_count') if isinstance(meta, dict) else None
        if (not isinstance(records, list) or len(records) > max_results
                or type(count) is not int or count < len(records)
                or any(not isinstance(item, dict) for item in records)):
            raise _failure('Invalid bounded database result page')
        return records, count

    @staticmethod
    def _result(provider, records, count, responses, checkpoint):
        checkpoint()
        if len({r['id'] for r in records}) != len(records):
            raise _failure('Duplicate database result identities')
        return ResourceResponse(provider, records, count, responses)

    def chembl_compound(self, query, *, max_results=5, check_cancelled=lambda: None):
        query = _query(query)
        _limit(max_results)
        checkpoint = self._checkpoint(check_cancelled)
        exact = re.fullmatch(r'CHEMBL[0-9]{1,12}', query)
        body, raw = self._get(CHEMBL + ('molecule/' + query + '.json' if exact else 'molecule/search.json'),
                              {} if exact else {'q': query, 'limit': max_results, 'offset': 0}, checkpoint)
        items, count = ([body], 1) if exact else self._page(body, 'molecules', max_results)
        records = []
        for item in items:
            identity = item.get('molecule_chembl_id')
            if not isinstance(identity, str) or not re.fullmatch(r'CHEMBL[0-9]{1,12}', identity) or (exact and identity != query):
                raise _failure('ChEMBL molecule identity mismatch')
            fields = _fields(item, 'molecule_chembl_id pref_name molecule_type max_phase first_approval oral parenteral topical black_box_warning withdrawn_flag therapeutic_flag molecule_structures molecule_properties molecule_synonyms molecule_hierarchy')
            records.append(_record('ChEMBL', identity, item.get('pref_name') or identity,
                'https://www.ebi.ac.uk/chembl/explore/compound/' + identity, fields, 'compound_database_record'))
        return self._result('ChEMBL', records, count, [raw], checkpoint)

    def chembl_activity(self, molecule_chembl_id, *, max_results=5, check_cancelled=lambda: None):
        _identity(molecule_chembl_id, r'CHEMBL[0-9]{1,12}')
        _limit(max_results)
        checkpoint = self._checkpoint(check_cancelled)
        body, raw = self._get(CHEMBL + 'activity.json', {'molecule_chembl_id': molecule_chembl_id,
                                'limit': max_results, 'offset': 0}, checkpoint)
        items, count = self._page(body, 'activities', max_results)
        records = []
        for item in items:
            activity_id = item.get('activity_id')
            if (type(activity_id) not in (str, int) or not re.fullmatch(r'[0-9]{1,20}', str(activity_id))
                    or item.get('molecule_chembl_id') != molecule_chembl_id):
                raise _failure('ChEMBL activity identity mismatch')
            # Keep all supplied assay/measurement context together; values and
            # relation operators are preserved without potency reinterpretation.
            fields = _fields(item, 'activity_id molecule_chembl_id molecule_pref_name assay_chembl_id assay_description assay_type assay_variant_accession assay_variant_mutation bao_endpoint bao_format bao_label target_chembl_id target_pref_name target_organism target_tax_id document_chembl_id document_journal document_year src_id type relation value units standard_type standard_relation standard_value standard_units standard_flag pchembl_value activity_comment data_validity_comment data_validity_description potential_duplicate standard_text_value text_value record_id')
            identity = str(activity_id)
            records.append(_record('ChEMBL', identity, f'{molecule_chembl_id} activity {identity}',
                CHEMBL + 'activity/' + identity + '.json', fields, 'assay_specific_activity_record'))
        return self._result('ChEMBL', records, count, [raw], checkpoint)

    def reactome_pathway(self, stable_id, *, check_cancelled=lambda: None):
        _identity(stable_id, r'R-[A-Z]{3}-[0-9]{1,12}')
        checkpoint = self._checkpoint(check_cancelled)
        body, raw = self._get(REACTOME + stable_id, {}, checkpoint)
        if body.get('stId') != stable_id or body.get('schemaClass') != 'Pathway':
            raise _failure('Reactome response must match the exact stable ID and Pathway class')
        fields = _fields(body, 'dbId stId stIdVersion displayName name schemaClass speciesName species isInferred isInDisease summation literatureReference hasEvent disease releaseDate created modified')
        record = _record('Reactome', stable_id, body.get('displayName'),
                         'https://reactome.org/content/detail/' + stable_id, fields, 'pathway_database_record')
        return self._result('Reactome', [record], 1, [raw], checkpoint)

    def geo_search(self, query, *, max_results=5, check_cancelled=lambda: None):
        query = _query(query)
        _limit(max_results)
        checkpoint = self._checkpoint(check_cancelled)
        body, search_raw = self._get(NCBI + 'esearch.fcgi', {'db': 'gds', 'term': f'({query}) AND (gse[ETYP] OR gds[ETYP])',
            'retmax': max_results, 'retmode': 'json', 'tool': 'AdjPartner'}, checkpoint)
        search = body.get('esearchresult')
        if not isinstance(search, dict) or search.get('errorlist'):
            raise _failure('Invalid GEO discovery response')
        ids, count = search.get('idlist'), search.get('count')
        if (not isinstance(ids, list) or len(ids) > max_results
                or any(not isinstance(i, str) or not re.fullmatch(r'[0-9]{1,12}', i) for i in ids)
                or len(set(ids)) != len(ids) or not isinstance(count, str)
                or not re.fullmatch(r'[0-9]{1,12}', count) or int(count) < len(ids)):
            raise _failure('Invalid GEO bounded discovery identities')
        if not ids:
            return self._result('GEO', [], int(count), [search_raw], checkpoint)
        body, summary_raw = self._get(NCBI + 'esummary.fcgi', {'db': 'gds', 'id': ','.join(ids),
            'retmode': 'json', 'tool': 'AdjPartner'}, checkpoint)
        summary = body.get('result')
        if not isinstance(summary, dict) or summary.get('uids') != ids:
            raise _failure('GEO summary identities differ from discovery')
        records = []
        for uid in ids:
            item = summary.get(uid)
            if not isinstance(item, dict) or item.get('error') or str(item.get('uid')) != uid:
                raise _failure('Missing GEO summary record')
            accession = item.get('accession')
            if not isinstance(accession, str) or not re.fullmatch(r'G(?:SE|DS)[0-9]{1,12}', accession):
                raise _failure('Invalid GEO accession')
            fields = _fields(item, 'uid accession title summary taxon gdsType entryType gpl gse n_samples pdat suppFile samples relations pubmedids')
            records.append(_record('GEO', uid, item.get('title'),
                'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + accession, fields, 'candidate_dataset_metadata'))
        return self._result('GEO', records, int(count), [search_raw, summary_raw], checkpoint)


def import_response(context, research, response):
    """Preserve exact provider bytes privately; publish redacted scoped records."""
    context.check_cancelled()
    limitations = LIMITATIONS[response.provider]
    provenance = {'provider': response.provider, 'text_availability': 'database_record',
        'request_urls': [r.request_url for r in response.responses],
        'retrieved_at': response.responses[-1].received_at,
        'response_sha256': [hashlib.sha256(r.raw).hexdigest() for r in response.responses],
        'limitations': limitations}
    artifacts = []
    for index, raw in enumerate(response.responses):
        context.check_cancelled()
        artifacts.append(context.store.put_artifact(raw.raw, f'{response.provider.lower()}-response-{index}.json',
            'application/json', {**provenance, 'request_url': raw.request_url, 'retrieved_at': raw.received_at},
            session_id=context.session_id, run_id=context.run_id))
    provenance['response_artifact_ids'] = [a['id'] for a in artifacts]
    results = []
    for original in response.records:
        context.check_cancelled()
        record = context.store.redact(original)
        item_provenance = {**provenance, 'provider_id': record['id'], 'url': record['url'],
            'evidence_scope': record['evidence_scope'], 'public_text_redacted': record != original}
        artifact = context.store.put_artifact(canonical_json({'record': record}),
            f'{response.provider.lower()}-record.json', 'application/json', item_provenance,
            session_id=context.session_id, run_id=context.run_id)
        source = research.register_source(context.session_id, context.run_id, artifact_id=artifact['id'],
            kind='web', title=record['title'], provenance=item_provenance,
            idempotency_key=f'native-resource:{context.call_id}:{response.provider}:{record["id"]}')
        artifacts.append(artifact)
        results.append({'source_id': source['id'], 'artifact_id': artifact['id'], 'provider_id': record['id'],
            'title': record['title'], 'text_availability': 'database_record',
            'evidence_scope': record['evidence_scope'], 'selector': {'pointer': '/record'}})
    return ToolResult(data={'provider': response.provider, 'hit_count': response.hit_count,
        'returned': len(results), 'results': results, 'limitations': limitations,
        'request_urls': provenance['request_urls']}, artifacts=artifacts)
