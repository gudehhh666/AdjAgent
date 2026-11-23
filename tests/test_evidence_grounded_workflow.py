"""Scripted end-to-end evidence-grounded design; no model or network calls."""
import json

import pytest

from adj_partner.contracts import ModelReply, ToolCall
from adj_partner.orchestration.contracts import TaskSpec
from test_framework_design import seeded


WORKFLOW = 'evidence-grounded-design-v2'


def test_only_new_design_workflow_accepts_explicit_roots(tmp_path):
    root=str(tmp_path)
    assert TaskSpec('design','A conceptual comparison',workflow_version=WORKFLOW,allowed_roots=(root,)).allowed_roots==(root,)
    assert TaskSpec('design','A conceptual comparison',workflow_version=WORKFLOW).allowed_roots==()
    for mode,workflow in [('qa',WORKFLOW),('design','unified-v1'),('design','unified-experiment-v1'),('design','resource-guided-design-v1')]:
        with pytest.raises(ValueError):TaskSpec(mode,'A conceptual comparison',workflow_version=workflow,allowed_roots=(root,))
    assert TaskSpec('discovery_design','A comparison',allowed_roots=(root,)).allowed_roots==(root,)


def test_opt_in_workflow_reads_authorized_metadata_and_exact_window_into_decision(tmp_path):
    root=tmp_path/'public';root.mkdir()
    matrix=root/'synthetic_series_matrix.txt'
    matrix.write_text('!Sample_geo_accession\t"GSM1"\t"GSM2"\n!Sample_characteristics_ch1\t"subject: A"\t"subject: A"\n!Sample_characteristics_ch1\t"day: 0"\t"day: 2"\n')
    s,f,t,p=seeded(tmp_path,dataset=root,workflow_version=WORKFLOW)
    try:
        cfg=s.store.get_session(t['session_id'])['config']
        assert cfg['profile']=='research-evidence-first' and cfg['allowed_roots']==[str(root)]
        assert {'evidence_read','geo_matrix_metadata'}<=set(cfg['enabled_tools'])
        assert 'research_critic' not in cfg['allowed_tools'] and not cfg['allow_network']
        sid,rid=t['session_id'],t['initial_run_id']
        text='Unselected early material. '*1400+'DECISIVE_WINDOW synthetic fixed observation.'+' unread end.'*1400
        prov={'text_availability':'derived_full_text','url':'https://example.org/synthetic'}
        a=s.store.put_artifact(json.dumps({'sections':[{'text':text}]}),'parsed-synthetic.json','application/json',prov,session_id=sid,run_id=rid)
        src=s.research.register_source(sid,rid,artifact_id=a['id'],kind='literature',title='Synthetic exact-window source',provenance=prov,idempotency_key='exact-window')
        complete=p.complete;retrieval_calls=0
        def scripted(messages,tools,**kwargs):
            nonlocal retrieval_calls
            if tools:
                retrieval_calls+=1
                if retrieval_calls==1:
                    p.requests.append({'messages':messages,'tools':tools})
                    return ModelReply(tool_calls=[ToolCall('window','evidence_read',{'source_id':src['id'],'query':'DECISIVE_WINDOW'}),ToolCall('schema','geo_matrix_metadata',{'path':str(matrix)})],usage={'input_tokens':20,'output_tokens':10})
                if retrieval_calls==2:
                    p.requests.append({'messages':messages,'tools':tools})
                    return ModelReply(tool_calls=[ToolCall('rows','geo_matrix_metadata',{'path':str(matrix),'mode':'rows','fields':['f1','f2']})],usage={'input_tokens':20,'output_tokens':10})
            return complete(messages,tools,**kwargs)
        p.complete=scripted
        result=f.run(t['fingerprint'])
        assert result['execution_status']=='succeeded',result
        assert result['review_status']=='not_run' and result['delivery']['experiment_status']=='planned'
        decision=next(json.loads(x['messages'][-1]['content']) for x in p.requests if not x['tools'] and 'schema' in json.loads(x['messages'][-1]['content']))
        selected=decision['evidence']
        window=s.store.get_action(rid,'window')['result']['data']
        texts=[e['text'] for e in selected.values()]
        assert window['text'] in texts and text not in texts
        matrix_page=s.store.get_action(rid,'rows')['result']['data']['summary']
        assert matrix_page in texts and matrix_page['items']==[['subject: A','day: 0'],['subject: A','day: 2']]
        assert next(e for e in selected.values() if e['text']==matrix_page)['scope']=='userfile'
        assert all('This synthesis stage has no tools' in x['messages'][0]['content'] for x in p.requests if not x['tools'])
        events=s.store.events(sid)
        receipts=[r for e in events if e['kind']=='workflow.resource_handoff' for r in e['payload']['receipts']]
        assert {'evidence_read','geo_matrix_metadata'}<=set(r['tool'] for r in receipts if r['status']=='verified_literal_read')
        count=len(p.requests);assert f.run(t['fingerprint'])==result and len(p.requests)==count
        if s.native.resource_pack is None:pytest.skip('needs the local resource pack')
        s.native.resource_pack.manifest_hash='f'*64
        with pytest.raises(ValueError,match='Resource runtime changed'):f.run(t['fingerprint'])
    finally:s.close()


def test_new_workflow_without_roots_does_not_gain_file_permission(tmp_path):
    s,f,t,_=seeded(tmp_path,workflow_version=WORKFLOW)
    try:
        path=tmp_path/'private-matrix.txt';path.write_text('!Sample_geo_accession\tGSM1\n')
        cfg=s.store.get_session(t['session_id'])['config']
        assert cfg['allowed_roots']==[]
        result=s.executor.execute(ToolCall('denied','geo_matrix_metadata',{'path':str(path)}),t['session_id'],t['initial_run_id'])
        assert result.status!='ok'
    finally:s.close()


def test_metadata_page_handoff_does_not_expand_unread_rows_and_rejects_forgery(tmp_path):
    from copy import deepcopy
    from adj_partner.orchestration.resource_handoff import read_selections
    root=tmp_path/'public';root.mkdir()
    path=root/'synthetic_series_matrix.txt'
    path.write_text('!Sample_geo_accession\tGSM1\tGSM2\n!Sample_characteristics_ch1\tsubject:A\tsubject:B\n')
    s,f,t,_=seeded(tmp_path,dataset=root,workflow_version=WORKFLOW)
    try:
        sid,rid=t['session_id'],t['initial_run_id']
        args={'path':str(path),'mode':'rows','fields':['f1'],'limit':1}
        value=s.executor.execute(ToolCall('page','geo_matrix_metadata',args),sid,rid)
        assert value.status=='ok',value.to_dict()
        page=value.data['summary'];assert page['items']==[['subject:A']] and page['next_offset']==1
        selected,_=f._materialize(sid,rid,receipt_run_ids=[rid])
        assert page in [e['text'] for e in selected.values()]
        assert 'subject:B' not in json.dumps(selected)
        forged=deepcopy(value.to_dict());forged['data']['summary']['items']=[['invented']]
        s.store.claim_action(rid,'forged-page','geo_matrix_metadata',args,0)
        s.store.finish_action(rid,'forged-page',forged)
        _,audit=read_selections(s,sid,rid,[rid])
        item=next(a for a in audit if a['call_id']=='forged-page')
        assert item['status']=='excluded' and 'mismatch' in item['reason']
    finally:s.close()


def test_resource_v1_keeps_its_original_profile_and_root_restriction(tmp_path):
    s,_,t,_=seeded(tmp_path,workflow_version='resource-guided-design-v1')
    try:
        config=s.store.get_session(t['session_id'])['config']
        assert config['profile']=='research-resource-aware' and config['allowed_roots']==[]
        assert 'evidence_read' not in config['enabled_tools'] and 'geo_matrix_metadata' not in config['enabled_tools']
    finally:s.close()
