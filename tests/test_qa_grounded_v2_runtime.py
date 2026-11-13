import json
from types import SimpleNamespace

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ProviderFailure, ScriptedPort
from adj_partner.native.grounding import GroundingRetriever
from adj_partner.native.literature_search import HybridRetriever, SerperScholarWeb, blocked
from adj_partner.orchestration import registry
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework
from adj_partner.task_modes import qa_grounded

from test_qa_grounded_runtime import DRAFT, PARAGRAPH, QUERIES, QUESTION, REVISED, reply, transport

REFLECTED = 'Mn2+ activates STING, which drives dendritic-cell maturation after ablation [P1].'
WEB_TEXT = 'Manganese alginate hydrogel combined with microwave ablation enhanced dendritic cell maturation via STING.'


class FakeWeb:
    def __init__(self):
        self.calls = []

    def search(self, endpoint, query, num=8):
        self.calls.append((endpoint, query))
        if endpoint == '/scholar':
            return [dict(title='Manganese hydrogel ablation', url='https://journal.example/mn', snippet='Mn hydrogel',
                         pdf=None, year='2023'),
                    dict(title='Benchmark dump', url='https://huggingface.co/datasets/x', snippet=WEB_TEXT, pdf=None,
                         year=None)][:1 if 'huggingface' in query else 2]
        return []


class FakeReader:
    def __init__(self):
        self.urls = []

    def fetch(self, url):
        self.urls.append(url)
        return SimpleNamespace(media='text/markdown', body=WEB_TEXT.encode())


def setup(tmp_path, replies, *, network=True, web=True):
    port = ScriptedPort(replies)
    service = AgentService(tmp_path / 'state', model=port)
    framework = AgentFramework(service)
    calls, search, reader = [], FakeWeb() if web else None, FakeReader() if web else None
    framework.literature = HybridRetriever(GroundingRetriever(transport=transport(calls)), search, reader)
    task = framework.submit(TaskSpec('qa', QUESTION, model='offline-script', workflow_version=qa_grounded.WORKFLOW_V2,
                                     allow_network=network))
    return service, framework, task, port, calls, search, reader


def test_v2_is_the_active_grounded_qa_workflow_and_v1_is_frozen():
    assert registry.WORKFLOWS[qa_grounded.WORKFLOW_V2].status == 'active'
    assert registry.WORKFLOWS[qa_grounded.WORKFLOW_VERSION].status == 'frozen'


def test_blocklist_drops_benchmark_mirrors():
    assert blocked('https://huggingface.co/datasets/adj') and blocked('https://raw.githubusercontent.com/a/b')
    assert not blocked('https://pmc.ncbi.nlm.nih.gov/articles/PMC1/')


def test_serper_hits_skip_blocked_hosts_and_retry_transient_failures():
    attempts = []

    class Client:
        def post(self, path, payload, failure_prefix):
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError('HTTP 403')
            return {}, json.dumps({'organic': [{'title': 'A', 'link': 'https://huggingface.co/x'},
                                               {'title': 'B', 'link': 'https://journal.example/b', 'snippet': 's'}]})

    hits = SerperScholarWeb(Client(), sleep=lambda s: None).search('/scholar', 'alum')
    assert [h['url'] for h in hits] == ['https://journal.example/b'] and attempts == ['/scholar', '/scholar']


def test_hybrid_pool_mixes_pubmed_and_web_and_ranks_with_draft():
    calls, web, reader = [], FakeWeb(), FakeReader()
    found = HybridRetriever(GroundingRetriever(transport=transport(calls)), web, reader).gather(
        QUESTION, ['manganese alginate hydrogel microwave ablation'], DRAFT)
    urls = {p['url'] for p in found['passages']}
    assert 'https://pubmed.ncbi.nlm.nih.gov/111/' in urls and 'https://journal.example/mn' in urls
    assert not any('huggingface' in u for u in urls)
    assert {e for e, _ in web.calls} == {'/scholar', '/search'}
    assert reader.urls == ['https://journal.example/mn']


def test_revised_answer_is_self_checked_and_keeps_its_citations(tmp_path):
    service, framework, task, port, *_ = setup(tmp_path, [reply(DRAFT), reply(QUERIES), reply(REVISED), reply(REFLECTED)])
    try:
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'succeeded', result
        assert result['enhancement_status'] == 'applied' and result['reflection_status'] == 'applied'
        assert result['delivery']['content'] == REFLECTED
        assert REVISED in port.requests[3]['messages'][1]['content']
        assert len(result['delivery']['sources']) == 1
        assert PARAGRAPH in port.requests[2]['messages'][1]['content'] or WEB_TEXT in port.requests[2]['messages'][1]['content']
        assert framework.run(task['fingerprint']) == result and len(port.requests) == 4
    finally:
        service.close()


def test_general_question_still_gets_the_self_check(tmp_path):
    general = json.dumps({'specific_study': False, 'queries': []})
    service, framework, task, port, calls, web, _ = setup(tmp_path, [reply(DRAFT), reply(general), reply(REFLECTED)])
    try:
        result = framework.run(task['fingerprint'])
        assert result['enhancement_status'] == 'not_attempted' and result['reflection_status'] == 'applied'
        assert result['delivery']['content'] == REFLECTED
        assert calls == [] and web.calls == []
    finally:
        service.close()


def test_failed_or_abstaining_self_check_delivers_the_prior_answer(tmp_path):
    for name, last in (('failed', ProviderFailure('timeout', 'Unknown result')),
                       ('rejected_abstention', reply('I cannot identify the specific study, so no answer can be given.'))):
        service, framework, task, port, *_ = setup(tmp_path / name, [reply(DRAFT), reply(QUERIES), reply(qa_grounded.KEEP), last])
        try:
            result = framework.run(task['fingerprint'])
            assert result['execution_status'] == 'succeeded'
            assert result['enhancement_status'] == 'unchanged' and result['reflection_status'] == name
            assert result['delivery']['content'] == DRAFT
        finally:
            service.close()


def test_without_gateway_credentials_v2_uses_pubmed_only(tmp_path):
    service, framework, task, port, calls, *_ = setup(tmp_path, [reply(DRAFT), reply(QUERIES), reply(REVISED), reply(REFLECTED)],
                                                      web=False)
    try:
        result = framework.run(task['fingerprint'])
        assert result['delivery']['sources'] == ['https://pubmed.ncbi.nlm.nih.gov/111/']
        assert PARAGRAPH in port.requests[2]['messages'][1]['content']
    finally:
        service.close()
