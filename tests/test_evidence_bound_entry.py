import json
from pathlib import Path

from adj_partner.application import AgentService
from adj_partner.models import ScriptedPort
from adj_partner.contracts import ModelReply, ToolCall


def test_bound_entry_runs_with_authorized_input_and_no_answer_injection(tmp_path):
    data = tmp_path / 'measurements.csv'
    data.write_text('sample,value\na,987654321\n')
    port = ScriptedPort([ModelReply(content='No calculation has run yet.')])
    service = AgentService(tmp_path / 'state', model=port)
    try:
        sid = service.create_session(profile='research-evidence-bound', allowed_roots=[str(tmp_path)],
                                     allowed_tools=['code_interpreter', 'tools_enable'])['id']
        rid = service.submit(sid, f'Count rows in `{data}`.')['id']
        assert service.drive(rid)['status'] == 'succeeded'
        messages = port.requests[0]['messages']
        packet = next(m['content'] for m in messages if m['content'].startswith('Pinned resource directory'))
        assert str(data) in packet and 'sha256' in packet
        assert '987654321' not in packet
        assert 'summary' in messages[0]['content']
        assert service.session(sid)['config']['allowed_roots'] == [str(tmp_path)]
    finally:
        service.close()


def test_bound_entry_cli_registered_and_old_profile_unchanged(tmp_path):
    from adj_partner.cli import parser
    from adj_partner.quality_entry import GUIDANCE, TOOLS
    from adj_partner.profiles import get_profile
    assert parser().parse_args(['new', '--profile', 'research-evidence-bound']).profile == 'research-evidence-bound'
    assert get_profile('research-evidence-first') == {'instructions': GUIDANCE, 'tools': TOOLS}


def test_bound_entry_never_exposes_unpermitted_input(tmp_path):
    data = tmp_path / 'denied.csv'; data.write_text('private field\n')
    port = ScriptedPort([ModelReply(content='Input not accessible.')])
    service = AgentService(tmp_path / 'state', model=port)
    try:
        sid = service.create_session(profile='research-evidence-bound', allowed_roots=[],
                                     allowed_tools=['code_interpreter'])['id']
        rid = service.submit(sid, f'Inspect `{data}`.')['id']
        assert service.drive(rid)['status'] == 'succeeded'
        cards = [m['content'] for m in port.requests[0]['messages'] if m['content'].startswith('Pinned resource directory')]
        assert not cards
    finally:
        service.close()


def test_bound_entry_preserves_large_task_once_without_changing_history(tmp_path):
    goal = 'Task input start\n' + ('0123456789' * 3200) + '\nTask input end'
    port = ScriptedPort([ModelReply(content='No calculation performed.')])
    service = AgentService(tmp_path / 'state', model=port)
    try:
        sid = service.create_session(profile='research-evidence-bound', allowed_tools=[],
                                     budget={'max_input_tokens': 100000})['id']
        rid = service.submit(sid, goal)['id']
        assert service.drive(rid)['status'] == 'succeeded'
        assert sum(goal in m.get('content', '') for m in port.requests[0]['messages']) == 1
        assert any(m.get('content') == goal for m in service.messages(sid))
    finally:
        service.close()


def test_bound_entry_projects_authorized_snapshot_readers_without_enable_roundtrip(tmp_path):
    port = ScriptedPort([ModelReply(content='No source read yet.')])
    service = AgentService(tmp_path / 'state', model=port)
    try:
        sid = service.create_session(profile='research-evidence-bound', allowed_roots=[],
                                     allowed_tools=['adjuvant_read_fields', 'adjuvant_cite'])['id']
        rid = service.submit(sid, 'Inspect a specified installed snapshot record.')['id']
        assert service.drive(rid)['status'] == 'succeeded'
        names = {t['function']['name'] for t in port.requests[0]['tools']}
        assert names == {'adjuvant_read_fields', 'adjuvant_cite'}
        assert service.session(sid)['config']['allowed_roots'] == []
    finally:
        service.close()


def test_qa_coverage_survives_tool_read_and_final_synthesis(tmp_path):
    replies = []
    port = ScriptedPort(replies)
    service = AgentService(tmp_path/'state', model=port)
    try:
        sid = service.create_session(allowed_tools=['artifact_read'], budget={'max_steps': 2})['id']
        fact = 'The day-seven readout increased relative to vehicle.'
        source = service.store.put_artifact(fact, 'observation.txt', 'text/plain', {}, session_id=sid)
        question = 'Which endpoint changed versus vehicle, and what can be concluded about duration?'
        rid = service.submit(sid, question)['id']
        replies.extend([ModelReply(tool_calls=[ToolCall('read', 'artifact_read', {'artifact_id': source['id']})],
                                   finish_reason='tool_calls'),
                        ModelReply(content='The day-seven readout increased. Duration is unknown.')])
        assert service.drive(rid)['status'] == 'succeeded'
        first, final = port.requests
        assert first['tools'] and final['tools'] == []
        assert any(m['role']=='tool' and fact in m['content'] for m in final['messages'])
        for request in (first, final):
            assert any(m['role']=='user' and question in m['content'] for m in request['messages'])
            system = request['messages'][0]['content']
            assert 'each part of the question' in system
            assert 'comparator' in system and 'endpoint' in system
        assert 'not a reason to withhold' in final['messages'][0]['content']
        assert 'study-specific' not in final['messages'][0]['content']
        assert 'withholding a substantive answer' in first['messages'][0]['content']
        assert 'Synthesis window' in final['messages'][0]['content']
    finally:
        service.close()
