"""Bounded read-only UI activity over real Store rows, never a new truth ledger."""

from contextlib import contextmanager
import json

import pytest

from adj_partner.application import AgentService
from adj_partner.models import ScriptedPort
from adj_partner.tools.catalog import canonical_json


@pytest.fixture
def current(tmp_path):
    service = AgentService(tmp_path/"state",model=ScriptedPort([]))
    sid = service.create_session()["id"]
    rid = service.submit(sid,"Synthetic activity projection")["id"]
    yield service,sid,rid
    service.close()


def test_activity_pages_are_incremental_scoped_read_only_and_reopenable(current):
    service,sid,rid = current
    for index in range(5): service.store.append_event(sid,"fixture",{"index":index},rid)
    expected = [item["seq"] for item in service.events(sid,rid)]
    found,cursor = [],0
    while True:
        page=service.activity_after(sid,rid,cursor,limit=2)
        found.extend(event["seq"] for event in page["events"])
        assert all(set(event)=={"seq","kind"} for event in page["events"])
        if not page["events"]: break
        assert page["next_cursor"]>cursor
        cursor=page["next_cursor"]
    assert found==expected and page["next_cursor"]==cursor
    assert page["actions"]==[] and page["active_job_count"]==0
    assert [item["seq"] for item in service.events(sid,rid)]==expected
    reopened = AgentService(service.store.root,model=ScriptedPort([]))
    try: assert reopened.activity_after(sid,rid,cursor)["events"]==[]
    finally: reopened.close()


@pytest.mark.parametrize("after_seq,limit", [(-1,64),(True,64),(0,0),(0,129),(0,True),(0,1.2)])
def test_activity_rejects_invalid_cursors_and_limits(current,after_seq,limit):
    service,sid,rid=current
    with pytest.raises(ValueError): service.activity_after(sid,rid,after_seq,limit)


def test_activity_cannot_cross_session_or_unknown_run(current):
    service,sid,rid=current
    other=service.create_session()["id"]
    with pytest.raises(ValueError): service.activity_after(other,rid)
    with pytest.raises(KeyError): service.activity_after(sid,"run_missing")
    with pytest.raises(ValueError): service.activity_after(sid,"bad/id")


def test_progress_and_action_errors_redact_whole_current_secrets_before_projection(current):
    service,sid,rid=current
    secret="LATE_ACTIVITY_SECRET_"+"abcdefghijk"*6
    prefix=canonical_json({"message":"MARK"}).index("MARK")
    message="x"*(2048-prefix-len(secret)+1)+secret+"z"*8000
    service.store.append_event(sid,"job.progress",{"message":message},rid)
    service.store.claim_action(rid,"fixture_action","fixture_tool",{},0)
    service.store.finish_action(rid,"fixture_action",{"status":"error","data":"LARGE_RESULT_"+"x"*100000,
        "error":{"message":message}})
    service.catalog.register_many([],secrets=[secret])
    page=service.activity_after(sid,rid)
    progress=next(event["payload"] for event in page["events"] if event["kind"]=="job.progress")
    assert progress["truncated"] is True and page["actions"][0]["error"]["truncated"] is True
    rendered=json.dumps(page)
    assert secret[:-1] not in rendered and "[REDACTED]" in rendered
    assert "LARGE_RESULT_" not in rendered and len(rendered)<10000


def test_oversize_progress_and_error_are_omitted_not_partially_loaded(current,monkeypatch):
    service,sid,rid=current
    service.store.append_event(sid,"job.progress",{"message":"oversize_progress_"+"x"*100000},rid)
    service.store.claim_action(rid,"fixture_action","fixture_tool",{},0)
    service.store.finish_action(rid,"fixture_action",{"status":"error","error":{"message":"oversize_error_"+"y"*100000}})
    original,observed=service.store.transaction,[]
    class Cursor:
        def __init__(self,cursor): self.cursor=cursor
        def fetchone(self):
            row=self.cursor.fetchone()
            if row is not None: observed.extend(value for value in row if isinstance(value,str))
            return row
        def fetchall(self):
            rows=self.cursor.fetchall()
            for row in rows: observed.extend(value for value in row if isinstance(value,str))
            return rows
    class Connection:
        def __init__(self,conn):self.conn=conn
        def execute(self,*args,**kwargs):return Cursor(self.conn.execute(*args,**kwargs))
    @contextmanager
    def measured(*args,**kwargs):
        assert not kwargs.get("write",False)
        with original(*args,**kwargs) as conn: yield Connection(conn)
    monkeypatch.setattr(service.store,"transaction",measured)
    page=service.activity_after(sid,rid)
    assert max(map(len,observed))<16384
    assert next(event["payload"] for event in page["events"] if event["kind"]=="job.progress")["omitted"] is True
    assert page["actions"][0]["error"]["omitted"] is True


def test_unrelated_large_event_payloads_are_never_selected_into_python(current,monkeypatch):
    service,sid,rid=current
    payload=canonical_json({"large":"x"*32768})
    with service.store.transaction(write=True) as conn:
        conn.executemany("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
            [(sid,rid,"history_fixture",payload,"2026-08-31T00:00:00Z")]*512)
    original,transactions=service.store.transaction,[]
    @contextmanager
    def measured(*args,**kwargs):
        transactions.append(kwargs.get("write",False))
        with original(*args,**kwargs) as conn: yield conn
    monkeypatch.setattr(service.store,"transaction",measured)
    monkeypatch.setattr(service,"events",lambda *args:pytest.fail("Do not read the full event service view"))
    monkeypatch.setattr(service,"actions",lambda *args:pytest.fail("Do not read full action results"))
    monkeypatch.setattr(service,"jobs",lambda *args:pytest.fail("Do not read full job results"))
    page=service.activity_after(sid,rid,limit=64)
    assert len(page["events"])==64 and len(canonical_json(page))<16000
    assert transactions==[False]


def test_actions_are_recent_bounded_with_truncation_and_jobs_are_metadata_only(current):
    service,sid,rid=current
    with service.store.transaction(write=True) as conn:
        for index in range(103):
            conn.execute("INSERT INTO actions VALUES(?,?,?,?,?,?,?,'completed',?,?,?)",
                (f"action_fixture_{index}",rid,f"call_{index}","fixture_tool","{}","hash",0,
                 canonical_json({"status":"ok","data":"PRIVATE_RESULT_"+"x"*5000}),"created","finished"))
        for index in range(6):
            conn.execute("""INSERT INTO execution_jobs(id,session_id,run_id,tool_name,handler_ref,definition,
                definition_hash,arguments,idempotency_key,fingerprint,timeout_seconds,deadline,created_at,workspace,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"job_fixture_{index}",sid,rid,"synthetic_job","not_executed","{}","hash",
                 '{"private":"PRIVATE_JOB_ARGUMENT"}',f"key_{index}",f"hash_{index}",10,9999999999,index,
                 "synthetic-not-created","succeeded" if index==0 else "queued"))
    page=service.activity_after(sid,rid)
    assert page["actions_truncated"] is True and len(page["actions"])==100
    assert page["actions"][0]["call_id"]=="call_3" and page["actions"][-1]["call_id"]=="call_102"
    assert page["active_job_count"]==5 and len(page["active_jobs"])==3
    assert all(set(job)=={"tool_name","status"} for job in page["active_jobs"])
    assert "PRIVATE_RESULT_" not in json.dumps(page) and "PRIVATE_JOB_ARGUMENT" not in json.dumps(page)


@pytest.mark.parametrize("error", [1,True])
def test_malformed_historical_scalar_error_is_an_omission_not_a_poll_failure(current,error):
    service,sid,rid=current
    service.store.claim_action(rid,"historical_action","fixture_tool",{},0)
    service.store.finish_action(rid,"historical_action",{"status":"error","error":error})
    page=service.activity_after(sid,rid)
    assert page["actions"][0]["error"]=={"omitted":True,"reason":"invalid_activity_payload"}
