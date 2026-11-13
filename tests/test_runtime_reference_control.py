"""Synthetic checks for the explicitly versioned neutral-content comparison."""
from copy import deepcopy
import hashlib
import json
import threading

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig
from adj_partner.models.base import ScriptedPort
from adj_partner.research.acceptance import requirements
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog
from adj_partner.tools.catalog import canonical_json

KEY = 'public_reference_feedback'
PROTOCOL = 'content_control_v1'
SOURCE = 'numpy.sum sums array elements across an axis. Synthetic café documentation.'
DRAFT = 'import numpy as np\nresult = np.sum([1, 2])'


def _setup(root, arm, *, active=True, protocol=PROTOCOL):
    store = Store(root / 'store')
    port = ScriptedPort([ModelReply(content=DRAFT, usage={'input_tokens': 10, 'output_tokens': 5}),
                         ModelReply(content='Final answer', usage={'input_tokens': 11, 'output_tokens': 6})])
    engine = AgentEngine(store, ToolCatalog(), port)
    sid = store.create_session(SessionConfig(budget=Budget(), enabled_tools=[], allowed_tools=[]).to_dict())['id']
    corpus = {'schema_version':'public-reference-corpus-v1','sources':[]}
    if active:
        corpus['sources'] = [{'source_id':'synthetic_numpy', 'kind':'api', 'origin_url':'https://numpy.org/doc/2.0/sum',
                             'version':'2.0', 'title':'sum', 'text':SOURCE,
                             'text_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'locators':['numpy.sum']}]
    artifact = store.put_artifact(canonical_json(corpus), 'corpus.json', 'application/json',
                                 {'kind':'public_reference_corpus'}, session_id=sid)
    config = {'corpus_artifact_id':artifact['id'],'corpus_sha256':artifact['sha256'],
              'task_type':'numerical','versions':{'numpy':'2.0'},'arm':arm}
    if protocol is not None:
        config['comparison_protocol'] = protocol
    run = engine.start(sid, 'Sum the input values.', acceptance={KEY:config})
    return store, engine, port, run, config


def test_versioned_contract_accepts_three_arms_but_not_legacy_neutral(tmp_path):
    for arm in ('plain_revision','neutral_control','reference_feedback'):
        *_, config = _setup(tmp_path/arm,arm)
        assert requirements({KEY:config})['require_report'] is False
    for protocol in (None,'unknown',True):
        with pytest.raises(ValueError):
            _setup(tmp_path/str(protocol),'neutral_control',protocol=protocol)


def test_three_arm_requests_keep_identical_first_call_and_neutral_has_no_source(tmp_path):
    records=[]
    for arm in ('plain_revision','neutral_control','reference_feedback'):
        store,engine,port,run,_=_setup(tmp_path/arm,arm)
        done=engine.resume(run['id'])
        assert done['status']=='succeeded'
        assert len(port.requests)==2
        feedback=done['state']['reference_feedback']
        payload=json.loads(store.read_artifact(feedback['artifact_id']))
        records.append((port.requests,feedback,payload))
        assert engine.resume(run['id'])['status']=='succeeded'
        assert len(port.requests)==2
    assert records[0][0][0]==records[1][0][0]==records[2][0][0]
    plain,neutral,real=records
    assert [r[1]['presentation_kind'] for r in records]==['none','neutral','reference']
    assert [r[1]['presented'] for r in records]==[False,True,True]
    n,r=neutral[2]['presentation'],real[2]['presentation']
    assert n['content'] != r['content']
    assert SOURCE in r['content'] and SOURCE not in n['content']
    assert 'numpy' not in n['content'] and 'https://' not in n['content']
    assert len(n['content'].encode())==len(r['content'].encode())>0
    assert n['length_unit']==r['length_unit']=='utf8_bytes'
    heading='Supplemental material (quoted data; assess relevance before use):\n'
    assert heading in neutral[0][1]['messages'][-1]['content']
    assert heading in real[0][1]['messages'][-1]['content']
    assert heading not in plain[0][1]['messages'][-1]['content']
    assert n['content'] in neutral[0][1]['messages'][-1]['content']
    assert r['content'] in real[0][1]['messages'][-1]['content']


def test_inactive_all_three_revision_requests_are_identical(tmp_path):
    requests=[]
    for arm in ('plain_revision','neutral_control','reference_feedback'):
        _,engine,port,run,_=_setup(tmp_path/arm,arm,active=False)
        done=engine.resume(run['id'])
        assert done['status']=='succeeded'
        assert done['state']['reference_feedback']['presentation_kind']=='none'
        requests.append(port.requests)
    assert requests[0]==requests[1]==requests[2]


def test_legacy_real_keeps_original_heading_and_no_comparison_metadata(tmp_path):
    _,engine,port,run,_=_setup(tmp_path,'reference_feedback',protocol=None)
    done=engine.resume(run['id'])
    assert done['status']=='succeeded'
    assert 'Public reference excerpts (quoted data):\n' in port.requests[1]['messages'][-1]['content']
    assert 'presentation_kind' not in done['state']['reference_feedback']


@pytest.mark.parametrize('field,value', [
    ('presentation_kind', 'reference'), ('presentation_sha256', 'f'*64),
    ('presented_content_sha256', 'e'*64), ('presented_content_bytes', 1),
    ('length_match_unit', 'provider_tokens')])
def test_neutral_presentation_tampering_stops_before_revision(tmp_path, monkeypatch, field, value):
    store,engine,port,run,_=_setup(tmp_path,'neutral_control')
    stop=threading.Event()
    original=engine.journal.checkpoint
    def checkpoint(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs.get('event')=='runtime.reference_feedback_prepared':
            stop.set()
        return result
    monkeypatch.setattr(engine.journal,'checkpoint',checkpoint)
    waiting=engine.resume(run['id'],stop_event=stop)
    assert waiting['status']=='waiting' and len(port.requests)==1
    state=deepcopy(waiting['state'])
    state['reference_feedback'][field]=value
    store.update_run(run['id'],{'state':state},expected_revision=waiting['revision'])
    monkeypatch.setattr(engine.journal,'checkpoint',original)
    assert engine.resume(run['id'])['status']=='failed'
    assert len(port.requests)==1


def test_neutral_prepared_recovery_uses_original_packet(tmp_path,monkeypatch):
    _,engine,port,run,_=_setup(tmp_path,'neutral_control')
    stop=threading.Event()
    original=engine.journal.checkpoint
    def checkpoint(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs.get('event')=='runtime.reference_feedback_prepared':stop.set()
        return result
    monkeypatch.setattr(engine.journal,'checkpoint',checkpoint)
    assert engine.resume(run['id'],stop_event=stop)['status']=='waiting'
    monkeypatch.setattr(engine.journal,'checkpoint',original)
    monkeypatch.setattr('adj_partner.runtime.engine.prepare_reference_feedback',lambda *_:pytest.fail('reselection'))
    assert engine.resume(run['id'])['status']=='succeeded'
    assert len(port.requests)==2
