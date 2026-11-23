import json

import httpx
import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ProviderFailure, ScriptedPort
from adj_partner.native.grounding import GroundingRetriever, jats_passages
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework
from adj_partner.task_modes import qa_grounded

QUESTION = 'What role does Mn2+ play when combined with Ca2+-alginate hydrogel and microwave ablation?'
DRAFT = 'Mn2+ likely acts as an innate immune adjuvant.'
REVISED = 'Mn2+ activates STING and promotes dendritic-cell maturation in this study [P1].'
QUERIES = json.dumps({'specific_study': True, 'queries': ['manganese alginate hydrogel microwave ablation']})
PARAGRAPH = ('Mn2+ ions released from the alginate hydrogel activated the STING pathway and promoted dendritic cell '
             'maturation after microwave ablation of tumors.')
JATS = f'''<?xml version="1.0"?><article><front><article-meta><title-group><article-title>Metallo-alginate
hydrogel potentiates microwave ablation</article-title></title-group><abstract><p>Alginate hydrogel with Mn2+.</p>
</abstract></article-meta></front><body><sec><title>Results</title><p>{PARAGRAPH}</p></sec>
<sec><title>Methods</title><p>{"Mice were housed under standard conditions for the entire experimental period. " * 2}</p>
</sec></body><back><ref-list><p>{"Unrelated reference text about manganese hydrogel ablation. " * 3}</p></ref-list></back></article>'''


def transport(calls):
    def handler(request):
        calls.append(request)
        path = request.url.path
        if path.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'idlist': ['111']}})
        if path.endswith('esummary.fcgi'):
            return httpx.Response(200, json={'result': {'111': {'title': 'Metallo-alginate hydrogel', 'pubdate': '2022 Aug',
                                  'articleids': [{'idtype': 'pmc', 'value': 'PMC1'}]}}})
        if path.endswith('PMC1/fullTextXML'):
            return httpx.Response(200, content=JATS.encode())
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def reply(text):
    return ModelReply(content=text, usage={'input_tokens': 5, 'output_tokens': 2})


def setup(tmp_path, replies, *, network=True):
    port = ScriptedPort(replies)
    service = AgentService(tmp_path / 'state', model=port)
    framework = AgentFramework(service, allow_frozen=True)
    calls = []
    framework.grounding = GroundingRetriever(transport=transport(calls))
    task = framework.submit(TaskSpec('qa', QUESTION, model='offline-script', workflow_version=qa_grounded.WORKFLOW_VERSION,
                                     allow_network=network))
    return service, framework, task, port, calls


def test_jats_passages_keep_sections_and_skip_references():
    doc = jats_passages(JATS.encode())
    assert doc['title'].startswith('Metallo-alginate')
    assert [p['section'] for p in doc['passages']] == ['Abstract', 'Results', 'Methods']
    assert all('Unrelated reference' not in p['text'] for p in doc['passages'])


def test_gather_ranks_by_relevance_and_records_requests():
    calls = []
    found = GroundingRetriever(transport=transport(calls)).gather(QUESTION, ['manganese alginate hydrogel'])
    assert calls[0].url.params['sort'] == 'relevance'
    assert found['papers'][0]['text_availability'] == 'full_text'
    assert found['passages'][0]['section'] == 'Results'
    assert found['passages'][0]['pmid'] == '111'


def test_zero_hit_query_retries_with_one_term_dropped():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith('esearch.fcgi'):
            ids = [] if 'manganese' in request.url.params['term'] else ['111']
            return httpx.Response(200, json={'esearchresult': {'idlist': ids}})
        return httpx.Response(200, json={'result': {'111': {'title': 'Metallo-alginate hydrogel', 'pubdate': '2022'}}})

    records = GroundingRetriever(transport=httpx.MockTransport(handler)).relaxed('manganese alginate hydrogel microwave ablation')
    assert [r['pmid'] for r in records] == ['111']
    assert calls[1].url.params['term'] == 'alginate hydrogel microwave ablation'


def test_max_year_limits_pubmed_publication_date():
    calls = []
    GroundingRetriever(transport=transport(calls), max_year=2007).pubmed('alum adjuvant')
    params = calls[0].url.params
    assert (params['datetype'], params['mindate'], params['maxdate']) == ('pdat', '1800', '2007')
    GroundingRetriever(transport=transport(calls)).pubmed('alum adjuvant')
    assert 'maxdate' not in calls[-2].url.params
    with pytest.raises(ValueError):
        GroundingRetriever(max_year='2007')


def test_transport_error_is_retried_once_then_raised(monkeypatch):
    monkeypatch.setattr('adj_partner.native.grounding.time.sleep', lambda s: None)
    calls, inner = [], transport([])

    def flaky(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError('nodename nor servname provided')
        return inner.handle_request(request)

    records = GroundingRetriever(transport=httpx.MockTransport(flaky)).pubmed('manganese alginate')
    assert [r['pmid'] for r in records] == ['111'] and len(calls) == 3

    def down(request):
        raise httpx.ConnectError('nodename nor servname provided')

    with pytest.raises(httpx.ConnectError):
        GroundingRetriever(transport=httpx.MockTransport(down)).pubmed('manganese alginate')


def test_parse_queries_rejects_malformed_or_general_output():
    assert qa_grounded.parse_queries('not json') == (False, [])
    assert qa_grounded.parse_queries('{"specific_study": true, "queries": []}') == (False, [])
    assert qa_grounded.parse_queries('```json\n{"specific_study": true, "queries": ["a \\"b\\" (c)"]}\n```') == (True, ['a  b   c'])


def test_applied_revision_cites_registered_source_and_caches(tmp_path):
    service, framework, task, port, calls = setup(tmp_path, [reply(DRAFT), reply(QUERIES), reply(REVISED)])
    try:
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'succeeded', result
        assert result['enhancement_status'] == 'applied'
        assert result['delivery']['content'] == REVISED
        assert result['delivery']['sources'] == ['https://pubmed.ncbi.nlm.nih.gov/111/']
        assert port.requests[0]['tools'] == [] and len(port.requests) == 3
        assert PARAGRAPH in port.requests[2]['messages'][1]['content']
        assert DRAFT in port.requests[2]['messages'][1]['content']
        sources = service.research.list_sources(task['session_id'], task['initial_run_id'])
        assert [s['provenance']['provider'] for s in sources] == ['host_grounding']
        assert service.store.get_run(task['initial_run_id'])['status'] == 'succeeded'
        assert framework.run(task['fingerprint']) == result
        assert len(port.requests) == 3
    finally:
        service.close()


def test_keep_draft_and_abstaining_revision_deliver_baseline(tmp_path):
    for text, status in ((qa_grounded.KEEP, 'unchanged'),
                         ('I cannot identify the specific study, so no answer can be given.', 'rejected_abstention')):
        service, framework, task, port, _ = setup(tmp_path / status, [reply(DRAFT), reply(QUERIES), reply(text)])
        try:
            result = framework.run(task['fingerprint'])
            assert result['execution_status'] == 'succeeded'
            assert result['enhancement_status'] == status
            assert result['delivery']['content'] == DRAFT
        finally:
            service.close()


def test_general_question_skips_retrieval(tmp_path):
    general = json.dumps({'specific_study': False, 'queries': []})
    service, framework, task, port, calls = setup(tmp_path, [reply(DRAFT), reply(general)])
    try:
        result = framework.run(task['fingerprint'])
        assert result['enhancement_status'] == 'not_attempted'
        assert result['fallback_reason']['code'] == 'general_question'
        assert result['delivery']['content'] == DRAFT
        assert calls == [] and len(port.requests) == 2
    finally:
        service.close()


def test_revision_failure_keeps_baseline_and_unknown_cost(tmp_path):
    service, framework, task, port, _ = setup(tmp_path, [reply(DRAFT), reply(QUERIES), ProviderFailure('timeout', 'Unknown result')])
    try:
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'failed'
        assert result['enhancement_status'] == 'failed'
        assert result['fallback_reason']['stage'] == 'revise'
        assert result['delivery']['content'] == DRAFT
        assert result['usage']['unknown_requests'] == 1
        assert framework.run(task['fingerprint']) == result
        assert len(port.requests) == 3
    finally:
        service.close()


def test_without_network_only_the_baseline_runs(tmp_path):
    service, framework, task, port, calls = setup(tmp_path, [reply(DRAFT)], network=False)
    try:
        result = framework.run(task['fingerprint'])
        assert result['fallback_reason']['code'] == 'network_not_allowed'
        assert result['delivery']['content'] == DRAFT
        assert calls == [] and len(port.requests) == 1
    finally:
        service.close()
