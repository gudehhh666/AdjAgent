import json

import pytest

from adj_partner.library import LiteratureLibrary, normalize_ids

JATS = b'''<?xml version="1.0"?><article><front><article-meta><title-group><article-title>Alum activates
the NLRP3 inflammasome in dendritic cells</article-title></title-group><article-id pub-id-type="pmid">18566365</article-id>
<abstract><p>Alum is the only adjuvant approved for routine human use and its mechanism is debated.</p></abstract>
</article-meta></front><body><sec><title>Results</title><p>Alum-induced IL-1 beta secretion required NLRP3,
ASC and caspase-1 in bone marrow derived dendritic cells.</p></sec></body></article>'''

RECORD = {'pmid': '18566365', 'doi': 'https://doi.org/10.4049/JIMMUNOL.181.1.17',
          'title': 'Cutting edge: inflammasome activation by alum and alum adjuvant effect are mediated by NLRP3',
          'year': 2008, 'abstract': 'Alum is the only adjuvant approved for routine use in humans; NLRP3 mediates it.',
          'authors': ['Li H', 'Willingham SB']}


def library(tmp_path):
    return LiteratureLibrary(tmp_path / 'lib', adjuvants=['Alum', 'Squalene oil-in-water emulsion'])


def origin(ref, project=None):
    return {'kind': 'import', 'ref': ref, 'project': project}


def test_identifiers_are_normalized():
    assert normalize_ids({'pmcid': '123', 'doi': 'doi:10.1/AB'}, {'pmid': '9'}) == \
        {'pmcid': 'PMC123', 'doi': '10.1/ab', 'pmid': '9'}


def test_metadata_record_then_fulltext_files_one_paper_with_passages(tmp_path):
    lib = library(tmp_path)
    content = json.dumps(RECORD).encode()
    [paper_id] = lib.ingest(content, 'application/json', {'provider': 'scholarly_multi', 'query': 'alum NLRP3'},
                            origin=origin('run-a', 'alum-study'))
    assert paper_id == 'pmid:18566365'
    assert lib.ingest(JATS, 'application/xml', {'provider': 'europe_pmc', 'pmid': '18566365', 'pmcid': 'PMC1'},
                      origin=origin('run-b')) == [paper_id]
    paper = lib.paper('10.4049/jimmunol.181.1.17')
    assert paper['paper_id'] == paper_id and paper['best_text'] == 'full_text'
    assert paper['pmcid'] == 'PMC1' and paper['year'] == '2008'
    assert {d['kind'] for d in paper['documents']} == {'metadata_record', 'raw_fulltext'}
    assert {'kind': 'adjuvant', 'tag': 'Alum'} in paper['tags']
    assert {'kind': 'project', 'tag': 'alum-study'} in paper['tags']
    raw = next(d for d in paper['documents'] if d['kind'] == 'raw_fulltext')
    assert (lib.blobs / raw['blob']).read_bytes() == JATS


def test_reingesting_identical_bytes_only_adds_a_sighting(tmp_path):
    lib = library(tmp_path)
    content = json.dumps(RECORD).encode()
    for ref in ('run-a', 'run-a', 'run-c'):
        lib.ingest(content, 'application/json', {'provider': 'scholarly_multi'}, origin=origin(ref))
    stats = lib.stats()
    assert stats['papers'] == 1 and stats['documents'] == 1 and stats['sightings'] == 2
    assert stats['passages'] == 1


def test_records_known_by_different_ids_merge_once_linked(tmp_path):
    lib = library(tmp_path)
    lib.ingest(json.dumps({'doi': '10.9/x', 'title': 'A study of saponin adjuvants in mice models'}).encode(),
               'application/json', {'provider': 'scholarly_multi'}, origin=origin('r1'))
    lib.ingest(json.dumps({'pmid': '42', 'title': 'Another unrelated paper about emulsions'}).encode(),
               'application/json', {'provider': 'scholarly_multi'}, origin=origin('r2'))
    lib.ingest(json.dumps({'pmid': '42', 'doi': '10.9/X'}).encode(), 'application/json',
               {'provider': 'scholarly_resolution'}, origin=origin('r3'))
    assert lib.stats()['papers'] == 1
    assert {i['scheme'] for i in lib.paper('42')['identifiers']} >= {'pmid', 'doi'}


def test_search_ranks_passages_and_filters(tmp_path):
    lib = library(tmp_path)
    lib.ingest(JATS, 'application/xml', {'provider': 'europe_pmc', 'pmid': '18566365'}, origin=origin('r', 'p1'))
    hit = lib.search('NLRP3 caspase dendritic')[0]
    assert hit['pmid'] == '18566365' and 'caspase-1' in hit['text']
    assert lib.search('NLRP3', project='p1') and not lib.search('NLRP3', project='other')
    assert lib.search('NLRP3', adjuvant='Alum')
    assert not lib.search('NLRP3', max_year=2000)
    assert lib.search('NLRP3 nonexistentword')  # falls back from AND to OR


def test_host_grounding_passages_and_parsed_documents(tmp_path):
    lib = library(tmp_path)
    grounding = {'paper': {'pmid': '7', 'title': 'Squalene oil-in-water emulsion recruits monocytes to muscle',
                           'year': '2011', 'text_availability': 'full_text'},
                 'passages': [{'section': 'Results', 'text': 'MF59 recruited CD11b monocytes into injected muscle.'}]}
    lib.ingest(json.dumps(grounding).encode(), 'application/json', {'provider': 'host_grounding', 'pmid': '7'},
               origin=origin('qa'))
    parsed = {'document_type': 'jats', 'title': 'Parsed', 'abstract': [{'locator': 'a1', 'text': 'x' * 30}],
              'sections': [{'title': 'Methods', 'paragraphs': [{'locator': 's1', 'text': 'Mice were immunized twice.'}]}]}
    lib.ingest(json.dumps(parsed).encode(), 'application/json', {'provider': 'document_parser', 'pmid': '8'},
               origin=origin('qa'))
    assert lib.paper('pmid:7')['best_text'] == 'full_text'
    assert {'kind': 'adjuvant', 'tag': 'Squalene oil-in-water emulsion'} in lib.paper('pmid:7')['tags']
    assert lib.paper('pmid:8')['passages'] == 2


def test_unusable_material_is_ignored_and_failures_are_recorded(tmp_path):
    lib = library(tmp_path)
    assert lib.ingest(b'[1,2]', 'application/json', {'provider': 'scholarly_multi'}, origin=origin('r')) == []
    assert lib.ingest_safely(b'not xml', 'application/xml', {'provider': 'europe_pmc', 'pmid': '1'},
                             origin=origin('bad')) == []
    assert lib.stats()['failures'] == 1 and lib.stats()['papers'] == 0


def test_pack_citations_become_linked_papers_and_catalog_groups_them(tmp_path):
    pack = tmp_path / 'pack'
    pack.mkdir()
    (pack / 'manifest.json').write_text('{}')
    (pack / 'records.jsonl').write_text(json.dumps(
        {'id': 'vaxjo:1', 'dataset': 'vaxjo_master', 'title': 'Alum', 'fields': {'ref': 'PMID: 18566365'}}) + '\n')
    lib = library(tmp_path)
    assert lib.link_pack(pack) == 1
    lib.ingest(json.dumps(RECORD).encode(), 'application/json', {'provider': 'scholarly_multi'}, origin=origin('r'))
    paper = lib.paper('pmid:18566365')
    assert paper['pack_links'][0]['record_id'] == 'vaxjo:1' and paper['best_text'] == 'abstract'
    catalog = lib.catalog()
    assert '### Alum' in catalog and '### vaxjo_master' in catalog


def test_live_literature_sources_are_filed_and_library_faults_never_fail_registration(tmp_path, monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.contracts import Budget
    from adj_partner.models.base import ScriptedPort
    from adj_partner import library as library_module

    root = tmp_path / 'shared-lib'
    monkeypatch.setenv(library_module.ENV, str(root))
    svc = AgentService(tmp_path / 'state', model=ScriptedPort([]))
    try:
        sid = svc.create_session()['id']
        tid = svc.store.create_task(sid, 'alum')['id']
        rid = svc.store.create_run(sid, tid, Budget().to_dict())['id']
        def register(content, provider, key):
            artifact = svc.store.put_artifact(content, 'r.json', 'application/json',
                                              {'provider': provider, 'text_availability': 'abstract'},
                                              session_id=sid, run_id=rid)
            return svc.research.register_source(sid, rid, artifact_id=artifact['id'], kind='literature',
                                                title='Alum NLRP3', provenance={'provider': provider},
                                                idempotency_key=key)
        register(json.dumps(RECORD).encode(), 'scholarly_multi', 'k1')
        assert library_module.shared().paper('pmid:18566365')['sightings'] == 1

        def broken(*args, **kwargs):
            raise RuntimeError('disk full')
        monkeypatch.setattr(library_module.LiteratureLibrary, 'ingest', broken)
        assert register(json.dumps({**RECORD, 'pmid': '5'}).encode(), 'scholarly_multi', 'k2')['kind'] == 'literature'
        assert library_module.shared().stats()['failures'] == 1
    finally:
        svc.close()
        library_module._SHARED.clear()


def test_tests_never_write_to_the_project_library():
    from adj_partner.library import shared
    assert shared() is None


def _epmc_transport(calls):
    import httpx

    def handler(request):
        url = str(request.url)
        calls.append(url)
        if url.endswith('/PMC9/fullTextXML'):
            return httpx.Response(200, content=JATS)
        if '/fullTextXML' in url:
            return httpx.Response(404)
        if '/MED/18566365/references' in url:
            return httpx.Response(200, json={'referenceList': {'reference': [
                {'id': '111', 'source': 'MED', 'title': 'Alum adjuvant depot revisited'},
                {'id': '222', 'source': 'MED', 'title': 'Unrelated cardiology trial'}]}})
        if '/MED/18566365/citations' in url:
            return httpx.Response(200, json={'citationList': {'citation': [
                {'id': '42', 'source': 'MED', 'title': 'already filed'}]}})
        if 'elink.fcgi' in url:
            return httpx.Response(200, json={'linksets': [{'linksetdbs': [{'linkname': 'pubmed_pubmed', 'links': [
                {'id': '18566365', 'score': 999}, {'id': '333', 'score': 50}]}]}]})
        if '/search' in url:
            query = request.url.params['query']
            results = []
            if 'EXT_ID:111' in query:
                results.append({'pmid': '111', 'title': 'Alum adjuvant depot revisited', 'pubYear': '2012',
                                'abstractText': 'Alum retains antigen at the injection site briefly.'})
            if 'EXT_ID:333' in query:
                results.append({'pmid': '333', 'pmcid': 'PMC9', 'title': 'Similar but about heart failure',
                                'abstractText': 'No vaccine content here at all, only cardiology.'})
            if 'DOI:"10.9/t"' in query:
                results.append({'pmid': '77', 'doi': '10.9/T', 'title': 'Title only paper about saponin adjuvants',
                                'abstractText': 'QS-21 saponin adjuvant study abstract text here.'})
            return httpx.Response(200, json={'resultList': {'result': results}})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_enrichment_fills_abstracts_and_open_full_text_once(tmp_path):
    from adj_partner.library.enrich import Enricher, Fetcher
    lib = library(tmp_path)
    lib.ingest(json.dumps({'doi': '10.9/t', 'title': 'Title only paper about saponin adjuvants'}).encode(),
               'application/json', {'provider': 'scholarly_multi'}, origin=origin('r'))
    lib.ingest(json.dumps({'pmid': '5', 'pmcid': 'PMC9', 'title': 'Abstract paper with open full text available',
                           'abstract': 'An abstract long enough to be stored.'}).encode(),
               'application/json', {'provider': 'scholarly_multi'}, origin=origin('r'))
    calls = []
    enricher = Enricher(lib, Fetcher(transport=_epmc_transport(calls), interval=0), log=lambda m: None)
    assert enricher.abstracts() == {'checked': 1, 'filled': 1}
    assert lib.paper('10.9/t')['best_text'] == 'abstract' and lib.paper('10.9/t')['pmid'] == '77'
    assert enricher.fulltext() == {'checked': 1, 'filled': 1}
    assert lib.paper('pmid:5')['best_text'] == 'full_text'
    before = len(calls)
    assert enricher.abstracts()['checked'] == 0 and enricher.fulltext()['checked'] == 0
    assert len(calls) == before


def test_expansion_follows_links_keeps_relevant_papers_and_records_relations(tmp_path):
    from adj_partner.library.enrich import Enricher, Fetcher
    lib = library(tmp_path)
    lib.ingest(JATS, 'application/xml', {'provider': 'europe_pmc', 'pmid': '18566365'}, origin=origin('r'))
    lib.ingest(json.dumps({'pmid': '42', 'title': 'A previously filed paper on emulsions'}).encode(),
               'application/json', {'provider': 'scholarly_multi'}, origin=origin('r'))
    enricher = Enricher(lib, Fetcher(transport=_epmc_transport([]), interval=0), log=lambda m: None)
    assert [s['pmid'] for s in enricher.seeds()] == ['18566365']
    result = enricher.expand()
    assert result['new_papers'] == 1  # 111 kept; 222 filtered by title; 333 filtered after abstract lookup
    related = {(r['relation'], r['paper_id']) for r in lib.related('pmid:18566365')}
    assert related == {('cites', 'pmid:111'), ('cited_by', 'pmid:42')}
    assert lib.paper('pmid:111')['best_text'] == 'abstract' and lib.paper('pmid:333') is None
    assert enricher.seeds() == [] and enricher.expand()['seeds'] == 0
