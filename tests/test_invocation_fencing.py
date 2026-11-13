from types import SimpleNamespace
import pytest
from adj_partner.contracts import Budget,ModelReply
from adj_partner.models.base import ScriptedPort
from adj_partner.runtime.model_invocations import InvocationLedger,leased_run
from adj_partner.runtime.journal import LeaseLost
from test_model_invocations import setup


def test_checkpoint_fences_owner_inside_write_transaction(tmp_path):
    s,sid,rid=setup(tmp_path);ledger=InvocationLedger(s)
    with leased_run(s,rid) as owner:
        assert callable(getattr(owner,'fence',None))
        def race():
            owner()
            with s.transaction(write=True) as conn:
                conn.execute("UPDATE runtime_leases SET owner='new-owner' WHERE run_id=?",(rid,))
        race.fence=owner.fence
        race.run_id=rid
        with pytest.raises(LeaseLost):ledger.checkpoint(rid,'stale',{},guard=race)
    assert ledger.state(rid) is None


def test_finish_commits_run_task_and_preserves_terminal_outcome(tmp_path):
    s,sid,rid=setup(tmp_path);ledger=InvocationLedger(s)
    with leased_run(s,rid) as guard:
        assert callable(getattr(ledger,'finish',None))
        done=ledger.finish(rid,status='succeeded',state={'delivery':{'content':'final'}},guard=guard)
        assert done['status']=='succeeded'
        assert s.get_task(done['task_id'])['status']=='succeeded'
        with pytest.raises((ValueError,RuntimeError)):
            ledger.finish(rid,status='failed',state={},guard=guard)
        assert s.get_run(rid)['state']['delivery']['content']=='final'


def test_standalone_model_requests_share_persistent_wall_clock(tmp_path,monkeypatch):
    import adj_partner.runtime.model_invocations as module
    s,sid,rid=setup(tmp_path)
    s.update_run(rid,{'budget':Budget(max_input_tokens=10000,max_output_tokens=1000,wall_time_seconds=1).to_dict()})
    clock=[100.];monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:clock[0]))
    ledger=InvocationLedger(s);p=ScriptedPort([ModelReply(content='one',usage={'input_tokens':1,'output_tokens':1}),ModelReply(content='two')])
    with leased_run(s,rid) as guard:
        ledger.complete(sid,rid,p,[{'role':'user','content':'Q'}],[],role='critic',key='one',max_output_tokens=10,timeout_seconds=10,guard=guard)
    clock[0]+=1.1
    with leased_run(s,rid) as guard:
        with pytest.raises(RuntimeError,match='budget'):
            ledger.complete(sid,rid,p,[{'role':'user','content':'Q2'}],[],role='critic',key='two',max_output_tokens=10,timeout_seconds=10,guard=guard)
    assert len(p.requests)==1
    assert p.requests[0]['timeout_seconds']<=1


def test_finish_callback_failure_rolls_back_run_task_and_result(tmp_path):
    s,sid,rid=setup(tmp_path);ledger=InvocationLedger(s)
    before=s.get_run(rid);before_task=s.get_task(before['task_id']);before_events=s.events(sid,rid)
    def save_result(conn):
        assert conn.execute('SELECT status FROM runs WHERE id=?',(rid,)).fetchone()[0]=='succeeded'
        assert conn.execute('SELECT status FROM tasks WHERE id=?',(before['task_id'],)).fetchone()[0]=='succeeded'
        s._event(conn,sid,'fixture.result',{'delivery':'must roll back'},rid)
        raise RuntimeError('Workflow result owner was lost')
    with leased_run(s,rid) as guard:
        with pytest.raises(RuntimeError,match='owner was lost'):
            ledger.finish(rid,status='succeeded',state={'delivery':{'content':'answer'}},guard=guard,on_commit=save_result)
    assert s.get_run(rid)==before
    assert s.get_task(before['task_id'])==before_task
    assert s.events(sid,rid)==before_events
