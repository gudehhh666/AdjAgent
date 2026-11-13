"""Public database adapters: fixed fixtures plus production import/policy checks."""
import hashlib
import importlib
import importlib.util
import json

import httpx
import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor, ToolFailure


def client(handler, **kwargs):
    assert importlib.util.find_spec('adj_partner.native.scientific_resources'), 'ScientificResources adapter is missing'
    module = importlib.import_module('adj_partner.native.scientific_resources')
    return module.ScientificResources(transport=httpx.MockTransport(handler), **kwargs)


MOLECULE = {'molecule_chembl_id': 'CHEMBL25', 'pref_name': 'ASPIRIN', 'max_phase': '4',
            'molecule_structures': {'canonical_smiles': 'CC(=O)Oc1ccccc1C(=O)O'}}


def test_compound_exact_and_search_are_bounded_and_keep_exact_raw_bytes():
    requests = []
    raw = b'{ "molecule_chembl_id": "CHEMBL25", "pref_name": "ASPIRIN" }'
    def respond(request):
        requests.append(request)
        if request.url.path.endswith('/CHEMBL25.json'):
            return httpx.Response(200, content=raw)
        return httpx.Response(200, json={'molecules': [MOLECULE], 'page_meta': {'total_count': 12}})
    service = client(respond)
    exact = service.chembl_compound('CHEMBL25')
    assert exact.records[0]['id'] == 'CHEMBL25'
    assert exact.responses[0].raw == raw
    assert exact.records[0]['text_availability'] == 'database_record'
    found = service.chembl_compound('aspirin', max_results=1)
    assert found.hit_count == 12 and len(found.records) == 1
    assert requests[1].url.params['limit'] == '1'
    assert requests[1].url.path.endswith('/molecule/search.json')
    assert all(r.url.host == 'www.ebi.ac.uk' for r in requests)


def test_activity_retains_assay_target_and_measurement_context():
    activity = {'activity_id': 42, 'molecule_chembl_id': 'CHEMBL25', 'assay_chembl_id': 'CHEMBL123',
        'assay_description': 'Binding assay in human cells', 'assay_type': 'B',
        'target_chembl_id': 'CHEMBL777', 'target_organism': 'Homo sapiens',
        'standard_type': 'IC50', 'standard_relation': '>', 'standard_value': '100',
        'standard_units': 'nM', 'document_chembl_id': 'CHEMBL999', 'data_validity_comment': 'Outside typical range'}
    service = client(lambda r: httpx.Response(200, json={'activities': [activity], 'page_meta': {'total_count': 1}}))
    record = service.chembl_activity('CHEMBL25').records[0]
    assert record['assay_description'] == activity['assay_description']
    assert record['standard_relation'] == '>' and record['target_organism'] == 'Homo sapiens'
    assert record['data_validity_comment'] == activity['data_validity_comment']


def test_reactome_requires_matching_pathway_identity():
    body = {'stId': 'R-HSA-109581', 'displayName': 'Apoptosis', 'schemaClass': 'Pathway',
            'speciesName': 'Homo sapiens', 'isInferred': False, 'summation': [{'text': 'Curated pathway description.'}]}
    service = client(lambda r: httpx.Response(200, json=body))
    record = service.reactome_pathway('R-HSA-109581').records[0]
    assert record['id'] == body['stId'] and record['summation'] == body['summation']
    body['schemaClass'] = 'Reaction'
    with pytest.raises(ToolFailure):
        service.reactome_pathway('R-HSA-109581')
    body['schemaClass'], body['stId'] = 'Pathway', 'R-HSA-999'
    with pytest.raises(ToolFailure):
        service.reactome_pathway('R-HSA-109581')


def test_geo_two_responses_candidate_metadata_no_replication_claim():
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'count': '7', 'idlist': ['200000001']}})
        return httpx.Response(200, json={'result': {'uids': ['200000001'], '200000001': {
            'uid': '200000001', 'accession': 'GSE1', 'title': 'Expression fixture',
            'summary': 'A submitted dataset', 'taxon': 'Homo sapiens', 'n_samples': 4}}})
    result = client(respond).geo_search('public fixture', max_results=1)
    assert result.hit_count == 7 and len(result.responses) == 2
    assert result.records[0]['accession'] == 'GSE1'
    assert result.records[0]['evidence_scope'] == 'candidate_dataset_metadata'
    assert all(r.url.params['db'] == 'gds' for r in requests)
    assert requests[0].url.params['retmax'] == '1'
    assert requests[0].url.params['term'] == '(public fixture) AND (gse[ETYP] OR gds[ETYP])'


@pytest.mark.parametrize('status,headers,body', [
    (302, {'location': 'https://untrusted.example/'}, b''),
    (429, {}, b'{}'), (200, {'content-encoding': 'gzip'}, b'bad'),
    (200, {'content-length': '999999999'}, b'{}'),
    (200, {}, b'not json'), (200, {}, b'{"molecules": []}'),
])
def test_http_errors_do_not_retry_or_become_records(status, headers, body):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, headers=headers, content=body)
    with pytest.raises(ToolFailure):
        client(respond).chembl_compound('aspirin')
    assert len(calls) == 1


def test_validation_and_cancel_before_network():
    calls = []
    service = client(lambda request: calls.append(request))
    for method, args in [(service.chembl_activity, ('https://example.com',)),
                         (service.reactome_pathway, ('R-HSA-1/../../secret',)),
                         (service.chembl_compound, (' ',)), (service.geo_search, ('x' * 501,))]:
        with pytest.raises(ValueError):
            method(*args)
    with pytest.raises(ValueError):
        service.chembl_compound('aspirin', max_results=True)
    def cancelled():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        service.geo_search('public', check_cancelled=cancelled)
    assert calls == []


def test_oversized_result_and_duplicate_ids_rejected():
    body = {'molecules': [MOLECULE, MOLECULE], 'page_meta': {'total_count': 2}}
    service = client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(ToolFailure):
        service.chembl_compound('aspirin', max_results=1)
    with pytest.raises(ToolFailure):
        service.chembl_compound('aspirin', max_results=2)


def test_normalized_records_fit_critic_selection_budget():
    body = {**MOLECULE, 'molecule_synonyms': [{'molecule_synonym': 'x' * 17000}]}
    with pytest.raises(ToolFailure, match='normalized limit'):
        client(lambda r: httpx.Response(200, json=body)).chembl_compound('CHEMBL25')


def test_streaming_limit_cancellation_and_elapsed_timeout():
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'x' * (2 * 1024 * 1024)
            yield b'x'
    with pytest.raises(ToolFailure, match='2 MiB'):
        client(lambda r: httpx.Response(200, stream=Stream())).chembl_compound('CHEMBL25')
    state = {'cancel': False}
    class CancelStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{'
            state['cancel'] = True
            yield b'}'
    def check():
        if state['cancel']:
            raise RuntimeError('stream cancelled')
    with pytest.raises(RuntimeError, match='stream cancelled'):
        client(lambda r: httpx.Response(200, stream=CancelStream())).chembl_compound('CHEMBL25', check_cancelled=check)
    with pytest.raises(ToolFailure, match='elapsed'):
        client(lambda r: httpx.Response(200, json=MOLECULE), timeout_seconds=1e-12).chembl_compound('CHEMBL25')


def test_geo_empty_search_skips_summary_and_mismatch_fails():
    requests = []
    def empty(request):
        requests.append(request)
        return httpx.Response(200, json={'esearchresult': {'count': '0', 'idlist': []}})
    assert client(empty).geo_search('no match').records == []
    assert len(requests) == 1
    def mismatch(request):
        if request.url.path.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'count': '1', 'idlist': ['1']}})
        return httpx.Response(200, json={'result': {'uids': ['2'], '2': {'uid': '2'}}})
    with pytest.raises(ToolFailure, match='identities'):
        client(mismatch).geo_search('query')


def setup(tmp_path, network, service):
    store = Store(tmp_path / 'state')
    research = ResearchStore(store)
    catalog = ToolCatalog()
    NativeCapabilities(store, research=research, scientific_resources=service).register(catalog)
    sid = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)], allow_network=network,
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())['id']
    task = store.create_task(sid, 'Resource fixture', {})['id']
    rid = store.create_run(sid, task, Budget().to_dict())['id']
    return store, research, ToolExecutor(store, catalog), sid, rid


def test_native_network_gate_and_raw_provenance_import(tmp_path):
    raw = json.dumps(MOLECULE).encode()
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=raw)
    store, research, executor, sid, rid = setup(tmp_path, False, client(respond))
    call = ToolCall('deny', 'chembl_compound', {'query': 'CHEMBL25'})
    assert executor.execute(call, sid, rid).error['code'] == 'network_denied'
    assert requests == []
    store, research, executor, sid, rid = setup(tmp_path / 'allowed', True, client(respond))
    result = executor.execute(ToolCall('ok', 'chembl_compound', {'query': 'CHEMBL25'}), sid, rid)
    assert result.status == 'ok', result.to_dict()
    source = research.get_source(sid, rid, result.data['results'][0]['source_id'])
    assert source['kind'] == 'web'
    provenance = source['provenance']
    assert provenance['text_availability'] == 'database_record'
    assert provenance['response_sha256'] == [hashlib.sha256(raw).hexdigest()]
    assert store.read_artifact(provenance['response_artifact_ids'][0]) == raw
    assert json.loads(store.read_artifact(source['artifact_id']))['record']['id'] == 'CHEMBL25'
    assert result.data['results'][0]['selector'] == {'pointer': '/record'}
