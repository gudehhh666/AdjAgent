import importlib.util
import json

import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply
from adj_partner.models.base import ScriptedPort, ProviderFailure
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework


def helper():
    assert importlib.util.find_spec('adj_partner.task_modes.qa_question_first'), 'question-first helper missing'
    from adj_partner.task_modes import qa_question_first
    return qa_question_first


def run_fixture(tmp_path, final):
    port = ScriptedPort([ModelReply(content='OLD_DRAFT_SENTINEL', usage={'input_tokens':4,'output_tokens':2}), final])
    service = AgentService(tmp_path/'state', model=port)
    framework = AgentFramework(service, allow_frozen=True)
    task = framework.submit(TaskSpec('qa', 'What changes antigen surface charge?', model='offline-script',
                                    workflow_version='qa-question-first-v1', allow_network=False))
    return service, framework, task, port


def test_opt_in_plain_answer_uses_ledger_and_replays_without_critic(tmp_path):
    service, framework, task, port = run_fixture(tmp_path, ModelReply(content='Adsorption can change surface charge.', usage={'input_tokens':10,'output_tokens':6}))
    try:
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'succeeded', result
        assert result['delivery']['content'] == 'Adsorption can change surface charge.'
        assert result['review_status'] == 'not_run'
        assert result['delivery']['sources'] == []
        assert result['usage']['input_tokens'] == 14
        assert result['usage']['output_tokens'] == 8
        assert result['usage']['requests'] == 2
        assert 'OLD_DRAFT_SENTINEL' not in str(port.requests[-1])
        assert framework.run(task['fingerprint']) == result
        assert len(port.requests) == 2
    finally:
        service.close()


def test_opt_in_rejects_invented_url_and_does_not_request_again(tmp_path):
    service, framework, task, port = run_fixture(tmp_path, ModelReply(content='A result [paper](https://invented.example/paper)', usage={'input_tokens':10,'output_tokens':6}))
    try:
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'failed'
        assert 'URL' in result['error']['message']
        assert result['delivery_status'] == 'none'
        assert framework.run(task['fingerprint'])['execution_status'] == 'failed'
        assert len(port.requests) == 2
    finally:
        service.close()


def test_opt_in_unknown_provider_failure_is_not_replayed(tmp_path):
    service, framework, task, port = run_fixture(tmp_path, ProviderFailure('timeout', 'unknown'))
    try:
        assert framework.run(task['fingerprint'])['execution_status'] == 'failed'
        assert framework.run(task['fingerprint'])['execution_status'] == 'failed'
        assert len(port.requests) == 2
    finally:
        service.close()


def test_question_packet_has_titles_and_cannot_select_draft_only_terms():
    module = helper()
    documents = [{'id':'s1','title':'Charge study','provenance':{'url':'https://example.org/paper','text_availability':'full_text'},
                  'document':{'text':'Antigen surface charge changes after adsorption.'}},
                 {'id':'s2','title':'Unrelated','provenance':{'url':'https://example.org/other'},
                  'document':{'text':'UniqueDraftOnlySentinel unrelated detail.'}}]
    packet = module.evidence_packet('What changes antigen surface charge?', documents)
    assert [item['source_id'] for item in packet] == ['s1']
    assert packet[0]['title'] == 'Charge study'
    assert len(json.dumps(packet, ensure_ascii=False)) <= 24000


def test_source_contract_allows_exact_packet_urls_and_rejects_markdown_invention():
    module = helper()
    packet = [{'url':'https://example.org/paper'}]
    result = module.parse_answer('Supported [paper](https://example.org/paper).', packet)
    assert result['sources'] == ['https://example.org/paper']
    with pytest.raises(ValueError, match='URL'):
        module.parse_answer('Claim [paper](https://invented.example/paper).', packet)
    with pytest.raises(ValueError, match='URL'):
        module.parse_answer('Claim [paper](//invented.example/paper).', packet)
    with pytest.raises(ValueError, match='URL'):
        module.parse_answer('Claim [paper](ftp://invented.example/paper).', packet)


def test_source_contract_rejects_schemeless_and_html_link_targets():
    module = helper()
    for content in ('Read invented.example/paper for the result.', '<a href="/invented-paper">paper</a>'):
        with pytest.raises(ValueError, match='URL'):
            module.parse_answer(content, [])


def test_opt_in_pins_the_evaluated_skill(tmp_path, monkeypatch):
    module = helper()
    from adj_partner.task_modes import qa
    altered = tmp_path/'SKILL.md'
    altered.write_text('---\nname: changed\n---\nChanged policy')
    monkeypatch.setattr(qa, 'ANSWER_SKILL', altered)
    with pytest.raises(ValueError, match='skill'):
        module.answer_messages('Question', [])


def test_legacy_workflow_keeps_reviewed_json_contract(tmp_path):
    port = ScriptedPort([ModelReply(content='Draft', usage={'input_tokens':4,'output_tokens':2}),
                         ModelReply(content=json.dumps({'answer':'Reviewed answer','sources':[], 'review':'Checked','uncertainty':''}), usage={'input_tokens':8,'output_tokens':4})])
    service = AgentService(tmp_path/'legacy', model=port)
    try:
        framework = AgentFramework(service, allow_frozen=True)
        task = framework.submit(TaskSpec('qa','Public question',model='offline-script',allow_network=False))
        result = framework.run(task['fingerprint'])
        assert result['execution_status'] == 'succeeded'
        assert result['review_status'] == 'passed'
        assert result['delivery']['content'] == 'Reviewed answer'
        assert 'Draft' in str(port.requests[-1])
    finally:
        service.close()


def test_opt_in_context_fits_same_remaining_budget_after_resolver_batch(tmp_path):
    """Reproduce live smoke: fixed 30-tool schema plus latest group exhausted 28k."""
    from adj_partner.runtime.context import ContextView
    from adj_partner.tools.policy import SessionToolset
    from adj_partner.tools.catalog import canonical_json
    service = AgentService(tmp_path/'context', model=ScriptedPort([]))
    try:
        framework = AgentFramework(service, allow_frozen=True)
        task = framework.submit(TaskSpec('qa', 'What changes antigen surface charge?', model='offline-script',
            workflow_version='qa-question-first-v1', allow_network=True,
            budget={'max_steps':12,'max_input_tokens':80000,'max_output_tokens':12000,'wall_time_seconds':300}))
        sid, rid = task['session_id'], task['initial_run_id']
        calls = [{'id':f'call_{i}', 'type':'function', 'function':{'name':'article_resolve', 'arguments':json.dumps({'source_id':f'source_{i}'})}} for i in range(4)]
        service.store.append_message(sid, {'role':'assistant','content':None,'tool_calls':calls}, rid)
        for call in calls:
            service.store.append_message(sid, {'role':'tool','tool_call_id':call['id'],
                'content':canonical_json({'status':'success','data':{'metadata':'x'*2100}})}, rid)
        snapshot = SessionToolset(service.store, service.catalog, sid).snapshot()
        projection = ContextView(service.store).build(sid, rid, snapshot, max_input_tokens=28484)
        assert projection['estimated_input_tokens'] <= 28484
        names = {t['function']['name'] for t in projection['tools']}
        expected = {'literature_search','web_search','article_resolve','fulltext_fetch','document_read','artifact_read','job_status'}
        assert names <= expected
        assert expected - {'web_search'} <= names  # web search may be unconfigured offline
        assert len(projection['tools']) <= 7
        assert {m.get('tool_call_id') for m in projection['messages'] if m['role']=='tool'} == {c['id'] for c in calls}
        # Default QA keeps the broad profile/tool contract.
        legacy = framework.submit(TaskSpec('qa','Legacy QA',model='offline-script',allow_network=False))
        legacy_config = service.store.get_session(legacy['session_id'])['config']
        assert 'code_interpreter' in legacy_config['enabled_tools']
        assert 'tools_enable' in legacy_config['allowed_tools']
    finally:
        service.close()
