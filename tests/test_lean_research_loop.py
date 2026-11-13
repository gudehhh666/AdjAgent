"""Offline regressions for a smaller model-facing loop, preserving host gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply, ToolCall, ToolResult
from adj_partner.models import ScriptedPort
from adj_partner.tools import ToolSpec
from adj_partner.tools.catalog import canonical_json
from adj_partner.tools.executor import ToolFailure
from adj_partner.tools.policy import SessionToolset


def service(tmp_path, replies=()):
    return AgentService(tmp_path/'state', project_root=tmp_path, model=ScriptedPort(list(replies)),
                        web_search=SimpleNamespace(secrets=()))


def projection(s, sid, rid):
    snapshot = SessionToolset(s.store, s.catalog, sid).snapshot()
    value = s._engine(sid).context.build(sid, rid, snapshot, max_input_tokens=120000)
    return value, snapshot


def test_small_default_request_retains_discovery_and_direct_answer_path(tmp_path):
    s = service(tmp_path, [ModelReply(content='A bounded answer.')])
    try:
        sid = s.create_session()['id']
        rid = s.submit(sid, 'Explain a basic scientific concept.')['id']
        assert s.drive(rid)['status'] == 'succeeded'
        req = s.models.injected.requests[0]
        names = {t['function']['name'] for t in req['tools']}
        assert {'catalog_search', 'tools_enable', 'ask_user', 'artifact_read'} <= names
        assert len(names) <= 9
        assert len(canonical_json({'messages':req['messages'], 'tools':req['tools']}).encode()) < 12000
        assert 'molecule-reports' not in req['messages'][0]['content']
        assert 'geo_matrix_metadata' not in req['messages'][0]['content']
        assert s.session(sid)['config']['allow_network'] is False
        assert s.session(sid)['config']['allowed_roots'] == []
    finally:
        s.close()


def test_specialist_enable_receipt_is_small_and_new_schema_is_real(tmp_path):
    s = service(tmp_path)
    try:
        sid = s.create_session()['id']; rid = s.submit(sid, 'Read a named source if needed.')['id']
        before,_ = projection(s,sid,rid)
        assert 'document_read' not in {t['function']['name'] for t in before['tools']}
        result = s.executor.execute(ToolCall('enable_doc','tools_enable',{'name':'document_read'}),sid,rid)
        assert result.status == 'ok'
        assert len(canonical_json(result.to_dict()).encode()) < 512
        after,_ = projection(s,sid,rid)
        assert 'document_read' in {t['function']['name'] for t in after['tools']}
        assert 'matches[].selector' in after['messages'][0]['content']
        assert s.session(sid)['config']['allow_network'] is False
    finally:
        s.close()


def quota_service(tmp_path, replies=()):
    s=service(tmp_path, replies); called=[]
    # An injected public tool fixture, not a live network/provider call.
    def run(context, query):
        called.append(query)
        raise ToolFailure('web_search_quota','Credits unavailable')
    s.catalog._bindings.pop('web_search')
    s.catalog.register(ToolSpec('web_search','Synthetic search',{'type':'object','properties':{
        'query':{'type':'string'}},'required':['query'],'additionalProperties':False}),run)
    return s,called


def test_quota_suspension_blocks_repeat_preserves_replay_and_resets_next_run(tmp_path):
    s,called=quota_service(tmp_path)
    try:
        sid=s.create_session()['id'];rid=s.submit(sid,'Look up a public source.')['id']
        first=ToolCall('first','web_search',{'query':'one'})
        failed=s.executor.execute(first,sid,rid)
        assert failed.error['code']=='web_search_quota'
        second=s.executor.execute(ToolCall('second','web_search',{'query':'two'}),sid,rid)
        assert second.error['code']=='tool_suspended_for_run'
        assert called==['one']
        assert s.executor.execute(first,sid,rid).to_dict()==failed.to_dict()
        assert len(s.actions(rid))==2
        p,snapshot=projection(s,sid,rid)
        assert 'web_search' not in {t['function']['name'] for t in p['tools']}
        assert 'web_search' not in snapshot['definition_hashes']
        assert 'web_search_quota' in p['messages'][0]['content']
        assert 'web_search' in s.session(sid)['config']['enabled_tools']
        s.journal.cancel(rid)
        next_run=s.submit(sid,'A later task after configuration repair.')['id']
        s.executor.execute(ToolCall('third','web_search',{'query':'three'}),sid,next_run)
        assert called==['one','three']
    finally:
        s.close()


def test_quota_like_source_data_and_http_errors_do_not_disable_search(tmp_path):
    s=service(tmp_path)
    try:
        sid=s.create_session()['id'];rid=s.submit(sid,'Task')['id']
        s.store.claim_action(rid,'source','literature_search',{},0)
        s.store.finish_action(rid,'source',ToolResult(data={'code':'web_search_quota'}).to_dict())
        s.store.claim_action(rid,'http','web_search',{},0)
        s.store.finish_action(rid,'http',ToolResult(status='error',error={
            'code':'web_search_http','message':'HTTP error'}).to_dict())
        s.store.claim_action(rid,'other','fulltext_fetch',{},0)
        s.store.finish_action(rid,'other',ToolResult(status='error',error={
            'code':'web_search_quota','message':'Wrong tool for breaker'}).to_dict())
        s.store.claim_action(rid,'unknown','web_search',{},0)
        p,_=projection(s,sid,rid)
        assert 'web_search' in {t['function']['name'] for t in p['tools']}
    finally:
        s.close()


def test_paragraph_only_report_derives_union_without_mutating_input(tmp_path):
    from test_harness_research_records import setup
    store,r,sid,rid,_=setup(tmp_path)
    first=r.record_claim(sid,rid,text='Unverified first option.',kind='proposal',evidence_ids=[])
    second=r.record_claim(sid,rid,text='Unverified alternative.',kind='proposal',evidence_ids=[])
    sections=[{'heading':'Decision','paragraphs':[
        {'text':'Options remain proposals.','claim_ids':[second['id'],first['id']]},
        {'text':'No experiment has run.','claim_ids':[second['id']]}]}]
    original=deepcopy(sections)
    report=r.create_report(sid,rid,title='Proposal',sections=sections,limitations=['No validation.'])
    assert sections==original
    assert report['sections'][0]['claim_ids']==[second['id'],first['id']]
    assert r.validate_report(sid,rid,report['id'])['accepted']
    rendered=store.read_artifact(report['artifact_id'])
    assert b'proposal' in rendered and b'unassessed' in rendered
    bad=deepcopy(sections);bad[0]['claim_ids']=[first['id']]
    with pytest.raises(ValueError,match='declared'):
        r.create_report(sid,rid,title='No silent repair',sections=bad,limitations=[])


def test_report_schema_allows_derived_ids_but_unknown_ids_remain_invalid(tmp_path):
    s=service(tmp_path)
    try:
        sid=s.create_session(profile='adjuvant')['id'];rid=s.submit(sid,'Write a proposal.')['id']
        claim=s.research.record_claim(sid,rid,text='Not validated.',kind='proposal',evidence_ids=[])
        args={'title':'Concept','sections':[{'heading':'Decision','paragraphs':[
            {'text':'Concept only.','claim_ids':[claim['id']]}]}],'limitations':['Unvalidated.']}
        good=s.executor.execute(ToolCall('report_good','report_create',args),sid,rid)
        assert good.status=='ok'
        args['sections'][0]['paragraphs'][0]['claim_ids']=['claim_missing']
        assert s.executor.execute(ToolCall('report_bad','report_create',args),sid,rid).status=='error'
        other_sid=s.create_session()['id'];other_rid=s.submit(other_sid,'Unrelated task.')['id']
        foreign=s.research.record_claim(other_sid,other_rid,text='Other task.',kind='proposal',evidence_ids=[])
        args['sections'][0]['paragraphs'][0]['claim_ids']=[foreign['id']]
        assert s.executor.execute(ToolCall('report_foreign','report_create',args),sid,rid).status=='error'
    finally:
        s.close()


def test_engine_quota_in_one_batch_has_one_provider_attempt_and_complete_history(tmp_path):
    s,called=quota_service(tmp_path,[
        ModelReply(tool_calls=[ToolCall('a','web_search',{'query':'one'}),
                               ToolCall('b','web_search',{'query':'two'})]),
        ModelReply(content='Search is unavailable; this result remains unresolved.')])
    try:
        sid=s.create_session()['id'];rid=s.submit(sid,'Find a public source.')['id']
        assert s.drive(rid)['status']=='succeeded'
        assert called==['one']
        requests=s.models.injected.requests
        assert len(requests)==2
        assert 'web_search' not in {t['function']['name'] for t in requests[1]['tools']}
        tool_messages=[m for m in requests[1]['messages'] if m['role']=='tool']
        assert {m['tool_call_id'] for m in tool_messages}=={'a','b'}
        assert 'web_search_quota' in tool_messages[0]['content']
        assert 'tool_suspended_for_run' in tool_messages[1]['content']
    finally:
        s.close()


def test_on_demand_method_guide_is_executable_after_enable(tmp_path):
    target=tmp_path/'config/skills/research-molecules/SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_bytes((Path(__file__).resolve().parents[1]/'config/skills/research-molecules/SKILL.md').read_bytes())
    s=service(tmp_path,[ModelReply(tool_calls=[ToolCall('enable','tools_enable',{'name':'research_skill'})]),
        ModelReply(tool_calls=[ToolCall('guide','research_skill',{'stage':'molecules'})]),
        ModelReply(content='The dossier has not been generated.')])
    try:
        sid=s.create_session()['id'];rid=s.submit(sid,'Prepare a multi-molecule dossier.')['id']
        assert s.drive(rid)['status']=='succeeded'
        guide=next(a for a in s.actions(rid) if a['call_id']=='guide')
        assert guide['result']['status']=='ok'
        body=guide['result']['data']['summary']['guidance']
        assert 'molecule-reports' in body and len(body.encode())<=1500
        assert 'research_skill' not in {t['function']['name'] for t in s.models.injected.requests[0]['tools']}
        assert 'research_skill' in {t['function']['name'] for t in s.models.injected.requests[1]['tools']}
    finally:
        s.close()


def test_specialist_enable_cannot_expand_host_permissions(tmp_path):
    s=service(tmp_path)
    try:
        sid=s.create_session(allowed_tools=['catalog_search','tools_enable'])['id']
        rid=s.submit(sid,'Restricted task.')['id']
        result=s.executor.execute(ToolCall('denied','tools_enable',{'name':'document_read'}),sid,rid)
        assert result.status=='denied'
        assert s.session(sid)['config']['enabled_tools']==['catalog_search','tools_enable']
    finally:
        s.close()


@pytest.mark.parametrize('authorize_all', [True, False])
def test_article_readers_enable_in_one_revision_or_none(tmp_path, authorize_all):
    names = ['article_resolve', 'fulltext_fetch', 'document_read']
    enable = ToolCall('group', 'tools_enable', {'name': names[0], 'also': names[1:]})
    s = service(tmp_path, [ModelReply(tool_calls=[enable]), ModelReply(content='Finished.')])
    try:
        allowed = ['tools_enable'] + (names if authorize_all else names[:2])
        sid = s.create_session(allowed_tools=allowed)['id']
        before = s.session(sid)
        rid = s.submit(sid, 'Read a specific study.')['id']
        assert s.drive(rid)['status'] == 'succeeded'
        action = s.actions(rid)[0]['result']
        assert action['status'] == ('ok' if authorize_all else 'denied')
        after = s.session(sid)
        assert after['revision'] == before['revision'] + int(authorize_all)
        projected = {t['function']['name'] for t in s.models.injected.requests[1]['tools']}
        assert set(names) & projected == (set(names) if authorize_all else set())
        assert before['config']['allowed_tools'] == after['config']['allowed_tools']
    finally:
        s.close()


def test_strict_bound_task_keeps_existing_report_acceptance(tmp_path):
    from adj_partner.runtime.context import BASE_INSTRUCTIONS
    s=service(tmp_path,[ModelReply(content='This text is not an accepted report.')])
    try:
        sid=s.create_session()['id'];rid=s.submit(sid,'Deliver a report.',acceptance={'require_report':True})['id']
        p,_=projection(s,sid,rid)
        assert BASE_INSTRUCTIONS in p['messages'][0]['content']
        assert s.drive(rid)['status']!='succeeded'
    finally:
        s.close()


@pytest.mark.parametrize('section',[
    {'heading':'Missing payload'},
    {'heading':'Unknown field','paragraphs':[],'extra':True},
    {'heading':'Null is not omitted','paragraphs':[],'claim_ids':None},
    {'heading':'Too many distinct IDs','paragraphs':[
        {'text':'First group.','claim_ids':[f'claim_{i}' for i in range(100)]},
        {'text':'Overflow.','claim_ids':['claim_100']}]}])
def test_derived_report_references_keep_shape_and_size_limits(section):
    from adj_partner.research.reports import report_fields
    with pytest.raises(ValueError):
        report_fields('Invalid',[section],[])
