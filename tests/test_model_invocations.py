import importlib.util
import pytest
from adj_partner.contracts import Budget, ModelReply, SessionConfig
from adj_partner.models.base import ScriptedPort, ProviderFailure
from adj_partner.storage import Store


def api():
    assert importlib.util.find_spec('adj_partner.runtime.model_invocations'), 'shared invocation ledger missing'
    from adj_partner.runtime.model_invocations import InvocationLedger, leased_run
    return InvocationLedger, leased_run


def setup(tmp_path, steps=3):
    s=Store(tmp_path/'state');sid=s.create_session(SessionConfig().to_dict())['id']
    tid=s.create_task(sid,'fixture')['id'];rid=s.create_run(sid,tid,Budget(max_steps=steps,max_input_tokens=10000,max_output_tokens=1000).to_dict())['id']
    return s,sid,rid


def test_cached_reply_bound_to_request_and_no_reissue(tmp_path):
    L,lease=api();s,sid,rid=setup(tmp_path);p=ScriptedPort([ModelReply(content='answer',usage={'input_tokens':4,'output_tokens':2})]);l=L(s)
    with lease(s,rid) as guard:
        r=l.complete(sid,rid,p,[{'role':'user','content':'question'}],[],role='critic',key='one',max_output_tokens=100,timeout_seconds=10,guard=guard)
        assert l.complete(sid,rid,p,[{'role':'user','content':'question'}],[],role='critic',key='one',max_output_tokens=100,timeout_seconds=10,guard=guard).content==r.content
        with pytest.raises(ValueError,match='identity'):
            l.complete(sid,rid,p,[{'role':'user','content':'DIFFERENT'}],[],role='critic',key='one',max_output_tokens=100,timeout_seconds=10,guard=guard)
    assert len(p.requests)==1 and l.usage(rid)['input_tokens']==4


def test_unknown_failure_stops_replay_and_budget_is_cumulative(tmp_path):
    L,lease=api();s,sid,rid=setup(tmp_path,steps=1);l=L(s)
    p=ScriptedPort([ProviderFailure('timeout','unknown')])
    with lease(s,rid) as g:
        with pytest.raises(ProviderFailure):l.complete(sid,rid,p,[{'role':'user','content':'Q'}],[],role='synthesis',key='a',max_output_tokens=50,timeout_seconds=10,guard=g)
    with lease(s,rid) as g:
        with pytest.raises(RuntimeError):l.complete(sid,rid,p,[{'role':'user','content':'Q'}],[],role='synthesis',key='a',max_output_tokens=50,timeout_seconds=10,guard=g)
        with pytest.raises(RuntimeError,match='budget'):l.complete(sid,rid,p,[{'role':'user','content':'Q'}],[],role='synthesis',key='b',max_output_tokens=50,timeout_seconds=10,guard=g)
    assert len(p.requests)==1 and l.usage(rid)['unknown_requests']==1


def test_second_owner_cannot_enter_and_checkpoint_survives_reopen(tmp_path):
    L,lease=api();s,sid,rid=setup(tmp_path);l=L(s)
    with lease(s,rid) as g:
        with pytest.raises(RuntimeError):
            with lease(s,rid):pass
        l.checkpoint(rid,'decision',{'candidate':'A'},guard=g)
    reopened=L(Store(tmp_path/'state'))
    assert reopened.state(rid)=={'stage':'decision','payload':{'candidate':'A'}}
