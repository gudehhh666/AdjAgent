"""Atomic session turns and fenced runtime checkpoints, with no model required."""

import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store


def make(tmp_path, **clocks):
    assert importlib.util.find_spec("adj_partner.runtime"), "durable runtime package must exist"
    from adj_partner.runtime.journal import RunJournal
    store = Store(tmp_path / "state")
    session = store.create_session(SessionConfig().to_dict())
    return store, RunJournal(store, **clocks), session["id"]


def test_start_commits_task_run_and_user_message_together(tmp_path):
    store, journal, sid = make(tmp_path)
    run = journal.start(sid, "你好", {}, Budget().to_dict())
    assert run["status"] == "created" and run["state"]["phase"] == "ready"
    assert store.get_task(run["task_id"])["goal"] == "你好"
    assert store.messages(sid) == [{"role": "user", "content": "你好"}]


@pytest.mark.parametrize("goal", ["", " ", 23, "x" * 100001])
def test_invalid_turn_creates_nothing(tmp_path, goal):
    store, journal, sid = make(tmp_path)
    with pytest.raises(ValueError): journal.start(sid, goal, {}, Budget().to_dict())
    assert store.list_tasks(sid) == [] and store.list_runs(sid) == [] and store.messages(sid) == []


def test_two_concurrent_turn_starts_have_only_one_owner(tmp_path):
    store, journal, sid = make(tmp_path)
    from adj_partner.runtime.journal import RunBusy
    barrier = threading.Barrier(2)
    def start(_):
        barrier.wait()
        try: return journal.start(sid, "hello", {}, Budget().to_dict())["id"]
        except RunBusy: return None
    with ThreadPoolExecutor(2) as pool: results = list(pool.map(start, range(2)))
    assert sum(item is not None for item in results) == 1
    assert len(store.list_tasks(sid)) == len(store.messages(sid)) == 1


def test_lease_fences_a_second_driver_and_stale_owner(tmp_path):
    # A real boot-ID probe forks sysctl, which can outlast a 20ms TTL in a large test process.
    clock = [1000.0]
    store, journal, sid = make(tmp_path, wall_clock=lambda: clock[0], monotonic_clock=lambda: clock[0],
                               boot_id_provider=lambda: "boot-fixed-0001")
    from adj_partner.runtime.journal import RunBusy, LeaseLost
    rid = journal.start(sid, "hello", {}, Budget().to_dict())["id"]
    owner = journal.acquire(rid, ttl=.02)
    with pytest.raises(RunBusy): journal.acquire(rid)
    clock[0] += .03
    other = journal.acquire(rid)
    with pytest.raises(LeaseLost): journal.checkpoint(rid, owner, state={"phase": "oops"})
    with pytest.raises(LeaseLost): journal.renew(rid, owner)
    journal.release(rid, owner)  # must not remove other owner's lease
    with pytest.raises(RunBusy): journal.acquire(rid)
    journal.checkpoint(rid, other, state={"phase": "requesting"}, status="running")
    assert store.get_run(rid)["state"]["phase"] == "requesting"
    journal.release(rid, other)


def test_checkpoint_messages_and_state_have_one_atomic_boundary(tmp_path):
    store, journal, sid = make(tmp_path)
    rid = journal.start(sid, "hi", {}, Budget().to_dict())["id"]
    owner = journal.acquire(rid)
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "{\"status\":\"ok\"}"}]
    journal.checkpoint(rid, owner, state={"phase": "ready"}, messages=messages,
                       usage={"input_tokens": 12}, event="runtime.batch_finished")
    assert store.messages(sid)[1:] == messages and store.get_run(rid)["usage"]["input_tokens"] == 12
    assert store.events(sid, rid)[-1]["kind"] == "runtime.batch_finished"
    with pytest.raises(ValueError):
        journal.checkpoint(rid, owner, state={"value": float("nan")}, messages=[{"role": "assistant", "content": "must not commit"}])
    assert len(store.messages(sid)) == 3


def test_persisted_host_cancel_cannot_be_reversed_by_running_driver(tmp_path):
    store, journal, sid = make(tmp_path)
    rid = journal.start(sid, "hi", {}, Budget().to_dict())["id"]
    owner = journal.acquire(rid)
    journal.cancel(rid)
    with pytest.raises(ValueError): journal.checkpoint(rid, owner, status="running")
    assert store.get_run(rid)["status"] == "cancelled"
    journal.release(rid, owner)
    follow = journal.start(sid, "new turn", {}, Budget().to_dict())
    assert follow["task_id"] != store.get_run(rid)["task_id"]


def test_known_secrets_are_removed_before_runtime_persistence(tmp_path):
    store, journal, sid = make(tmp_path)
    store.register_secrets(["opaque-private-value"])
    rid = journal.start(sid, "opaque-private-value", {}, Budget().to_dict())["id"]
    owner = journal.acquire(rid)
    journal.checkpoint(rid, owner, state={"answer": "opaque-private-value"}, event="runtime.test")
    assert "opaque-private-value" not in json.dumps([store.get_run(rid), store.messages(sid), store.events(sid)])


def test_real_user_input_is_durable_and_single_resolution(tmp_path):
    store, journal, sid = make(tmp_path)
    rid = journal.start(sid, "hi", {}, Budget().to_dict())["id"]
    request = journal.ask(rid, "call-1", "Which temperature?")
    assert journal.pending_inputs(rid)[0]["id"] == request["id"]
    assert journal.ask(rid, "call-1", "Which temperature?")["id"] == request["id"]
    with pytest.raises(ValueError): journal.ask(rid, "call-1", "Different question")
    journal.answer(rid, request["id"], "25 C")
    assert journal.input_result(rid, "call-1") == {"answer": "25 C", "source": "host_user", "request_id": request["id"]}
    with pytest.raises(ValueError): journal.answer(rid, request["id"], "40 C")
    assert journal.pending_inputs(rid) == []


def test_plan_is_public_working_state_not_acceptance_or_permissions(tmp_path):
    store, journal, sid = make(tmp_path)
    rid = journal.start(sid, "hi", {}, Budget().to_dict())["id"]
    journal.set_plan(rid, [{"step": "Inspect", "status": "in_progress"}])
    assert journal.get_plan(rid)[0]["step"] == "Inspect"
    assert store.get_task(store.get_run(rid)["task_id"])["acceptance"] == {}
    assert store.get_session(sid)["config"]["allowed_tools"] == []
    with pytest.raises(ValueError): journal.set_plan(rid, [{"step": "evil", "status": "approved"}])


def test_success_commit_rechecks_current_task_binding_in_writer_transaction(tmp_path):
    store, journal, sid = make(tmp_path)
    from adj_partner.runtime import journal as module
    assert hasattr(module, "task_digest")
    run = journal.start(sid, "old", {}, Budget().to_dict())
    expected = module.task_digest(store.get_task(run["task_id"]))
    owner = journal.acquire(run["id"])
    store.update_task(run["task_id"], {"goal": "new"})
    done = journal.checkpoint(run["id"], owner, status="succeeded", state={"delivery": {"content": "old"}},
        messages=[{"role": "assistant", "content": "old"}], expected_task_binding=expected)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "task_changed"
    assert not done["state"].get("delivery") and len(store.messages(sid)) == 1


def test_success_commit_rechecks_expired_active_deadline_inside_writer_transaction(tmp_path):
    store, journal, sid = make(tmp_path)
    run = journal.start(sid, "old", {}, Budget().to_dict())
    owner = journal.acquire(run["id"])
    done = journal.checkpoint(run["id"], owner, status="succeeded", state={"delivery": {"content": "late"}},
        messages=[{"role": "assistant", "content": "late"}], active_deadline=time.monotonic() - .01)
    assert done["status"] == "budget_exhausted" and not done["state"].get("delivery")
    assert len(store.messages(sid)) == 1
