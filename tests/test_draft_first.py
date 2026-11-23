"""Draft-first orchestration and its host fallbacks; scripted models only."""
from pathlib import Path

import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import Budget, ModelReply
from adj_partner.draft_first import RETRIEVAL_MISS_GUIDANCE, abstains, run_draft_first
from adj_partner.models import ScriptedPort
from adj_partner.models.base import ProviderFailure

PROJECT = Path(__file__).resolve().parents[1]
BUDGET = Budget(max_steps=6, max_input_tokens=100000, max_output_tokens=10000, wall_time_seconds=120)
DRAFT = 'Advax increased Tfh cells and germinal-center B cells, supporting affinity maturation.'
ABSTAIN = 'The supplied evidence does not identify a study-specific result for Advax on Tfh cells.'
USAGE = {'input_tokens': 20, 'output_tokens': 10}


def run(tmp_path, replies, **options):
    port = ScriptedPort(replies)
    service = AgentService(tmp_path / 'state', project_root=PROJECT, model=port)
    try:
        return port, run_draft_first(service, port, 'How does Advax affect Tfh and GC B cells?',
                                     system='Baseline', budget=BUDGET,
                                     session_options=dict(allowed_tools=[], instructions='Host', **options))
    finally:
        service.close()


@pytest.mark.parametrize('text,expected', [
    (ABSTAIN, True),
    ("I can't identify the study or what sRCCs stands for.", True),
    ('Please provide the paper, study title or DOI.', True),
    ('', True),
    (DRAFT, False),
    (DRAFT + ' ' + 'x' * 300 + ' The exact effect size could not be confirmed.', False),
])
def test_abstention_only_reads_the_opening(text, expected):
    assert abstains(text) is expected


def test_agent_answer_kept_and_sees_draft_with_guidance(tmp_path):
    port, out = run(tmp_path, [ModelReply(content=DRAFT, usage=USAGE),
                               ModelReply(content='Advax increased both populations in mice.', usage=USAGE)])
    assert out['source'] == 'agent' and out['answer'] == 'Advax increased both populations in mice.'
    system = next(m['content'] for m in port.requests[1]['messages'] if m['role'] == 'system')
    assert 'Host' in system and RETRIEVAL_MISS_GUIDANCE in system
    assert DRAFT in str(port.requests[1]['messages'])
    assert port.requests[0]['tools'] == []


def test_abstaining_agent_falls_back_to_substantive_draft(tmp_path):
    _, out = run(tmp_path, [ModelReply(content=DRAFT, usage=USAGE), ModelReply(content=ABSTAIN, usage=USAGE)])
    assert out['source'] == 'draft_after_abstention' and out['answer'] == DRAFT
    assert out['agent']['answer'] == ABSTAIN and out['delivered']


def test_abstention_kept_when_draft_also_abstains(tmp_path):
    _, out = run(tmp_path, [ModelReply(content='Please provide the study title.', usage=USAGE),
                            ModelReply(content=ABSTAIN, usage=USAGE)])
    assert out['source'] == 'agent' and out['answer'] == ABSTAIN


def test_failed_verification_delivers_draft_and_records_failure(tmp_path):
    _, out = run(tmp_path, [ModelReply(content=DRAFT, usage=USAGE), ProviderFailure('timeout', 'slow', cost_unknown=True)])
    assert out['source'] == 'draft_after_agent_failure' and out['answer'] == DRAFT
    assert not out['agent']['delivered']


def test_length_truncated_draft_is_used_as_coverage(tmp_path):
    port, out = run(tmp_path, [ModelReply(content='partial', finish_reason='length', usage=USAGE),
                               ModelReply(content='Plain agent answer.', usage=USAGE)])
    assert out['source'] == 'agent' and out['answer'] == 'Plain agent answer.'
    assert out['draft']['truncated'] is True
    system = next(m['content'] for m in port.requests[1]['messages'] if m['role'] == 'system')
    assert RETRIEVAL_MISS_GUIDANCE in system
    assert 'partial' in str(port.requests[1]['messages'])


def test_invalid_draft_runs_plain_agent_on_remaining_budget(tmp_path):
    port, out = run(tmp_path, [ModelReply(content='', usage=USAGE),
                               ModelReply(content='Plain agent answer.', usage=USAGE)])
    assert out['source'] == 'agent_without_draft' and out['draft_error'] == 'invalid_draft'
    assert out['answer'] == 'Plain agent answer.'
    system = next(m['content'] for m in port.requests[1]['messages'] if m['role'] == 'system')
    assert RETRIEVAL_MISS_GUIDANCE not in system
