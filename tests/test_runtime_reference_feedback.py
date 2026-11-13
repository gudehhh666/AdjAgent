"""Synthetic Store + Engine integration for bounded host reference feedback."""
from copy import deepcopy
import hashlib
import json
import threading

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall
from adj_partner.research.acceptance import requirements, research_acceptance
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog
from adj_partner.tools.catalog import canonical_json

KEY = "public_reference_feedback"
DRAFT = "import numpy as np\nanswer = np.sum([1, 2])"
SOURCE = "numpy.sum(a, axis=None): Sum of array elements over a given axis."


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def corpus():
    return {"schema_version": "public-reference-corpus-v1", "sources": [{
        "source_id": "synthetic_numpy", "kind": "api", "origin_url": "https://numpy.org/doc/2.0/reference/generated/numpy.sum.html",
        "version": "2.0", "title": "numpy.sum", "text": SOURCE, "text_sha256": sha(SOURCE),
        "locators": ["numpy.sum"]}]}


class Port:
    metadata = {"provider": "scripted", "model": "synthetic"}
    secrets = ()

    def __init__(self, replies=None, unknown=False):
        self.requests = []
        self.replies = list(replies or [DRAFT, "Revised ordinary answer"])
        self.unknown = unknown

    def complete(self, messages, tools, **limits):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools), **limits})
        answer = self.replies.pop(0)
        if callable(answer):
            answer = answer()
        if isinstance(answer, ModelReply):
            return answer
        return ModelReply(content=answer, tool_calls=[], finish_reason="stop", usage={} if self.unknown else {
            "input_tokens": 11, "output_tokens": 7})


def setup(tmp_path, *, arm="reference_feedback", pack=None, unknown=False, budget=None,
          replies=None, enabled=False, foreign=False, no_pack=False, default=False, goal="Sum the input values."):
    store = Store(tmp_path / "store")
    port = Port(replies, unknown)
    engine = AgentEngine(store, ToolCatalog(), port)
    sid = store.create_session(SessionConfig(budget=budget or Budget(),
        enabled_tools=["ask_user"] if enabled else [], allowed_tools=["ask_user"] if enabled else []).to_dict())["id"]
    pack_sid = store.create_session(SessionConfig().to_dict())["id"] if foreign else sid
    artifact = None if no_pack else store.put_artifact(canonical_json(corpus() if pack is None else pack),
        "synthetic-corpus.json", "application/json", {"kind": "public_reference_corpus"}, session_id=pack_sid)
    config = {"corpus_artifact_id": artifact["id"] if artifact else "artifact_missing",
              "corpus_sha256": artifact["sha256"] if artifact else "a" * 64,
              "task_type": "numerical", "versions": {"numpy": "2.0"}, "arm": arm}
    run = engine.start(sid, goal, acceptance={} if default else {KEY: config})
    return store, engine, run, port, artifact


def test_reference_contract_is_standalone_and_not_research():
    config = {"corpus_artifact_id": "artifact_fixture", "corpus_sha256": "a" * 64,
              "task_type": "numerical", "versions": {"numpy": "2.0"}, "arm": "reference_feedback"}
    assert requirements({KEY: config})["require_report"] is False
    assert research_acceptance({KEY: config}) is False
    with pytest.raises(ValueError):
        requirements({KEY: config, "response_json_schema": {"type": "string"}})
    for field, value in [("arm", "extra"), ("task_type", "hidden"), ("corpus_sha256", "bad"),
                         ("versions", {"numpy": ""})]:
        with pytest.raises(ValueError):
            requirements({KEY: {**config, field: value}})


@pytest.mark.parametrize("version", ["v" * 65, "é" * 33])
def test_oversized_version_is_rejected_before_first_model_call(tmp_path, version):
    store = Store(tmp_path / "store")
    port = Port()
    engine = AgentEngine(store, ToolCatalog(), port)
    sid = store.create_session(SessionConfig(enabled_tools=[], allowed_tools=[]).to_dict())["id"]
    artifact = store.put_artifact(canonical_json(corpus()), "corpus.json", "application/json",
        {"kind": "public_reference_corpus"}, session_id=sid)
    config = {"corpus_artifact_id": artifact["id"], "corpus_sha256": artifact["sha256"],
              "task_type": "numerical", "versions": {"numpy": version}, "arm": "reference_feedback"}
    with pytest.raises(ValueError, match="versions"):
        engine.start(sid, "Synthetic numerical task", acceptance={KEY: config})
    assert not port.requests


@pytest.mark.parametrize("arm", ["plain_revision", "reference_feedback"])
def test_two_calls_without_tools_or_schema_and_terminal_resume_is_idle(tmp_path, arm):
    store, engine, run, port, _ = setup(tmp_path, arm=arm)
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert done["state"]["delivery"]["kind"] == "conversation"
    assert done["state"]["delivery"]["content"] == "Revised ordinary answer"
    assert len(port.requests) == 2 and all(not r["tools"] for r in port.requests)
    assert done["usage"]["input_tokens"] == 22
    assert done["usage"]["output_tokens"] == 14
    assert not store.list_actions(run["id"])
    assert engine.resume(run["id"])["status"] == "succeeded" and len(port.requests) == 2


def test_default_stays_one_call_and_retains_model_visible_task_id(tmp_path):
    store, engine, run, port, _ = setup(tmp_path, default=True)
    assert engine.resume(run["id"])["status"] == "succeeded" and len(port.requests) == 1
    policy = json.loads(port.requests[0]["messages"][0]["content"].split("Host task policy:\n", 1)[1])
    assert policy["task_id"] == run["task_id"] == store.get_task(run["task_id"])["id"]
    assert policy["acceptance"] == {}


@pytest.mark.parametrize("case", ["missing", "foreign", "damage", "enabled"])
def test_bad_binding_stops_before_any_model_call(tmp_path, case):
    store, engine, run, port, artifact = setup(tmp_path, no_pack=case == "missing", foreign=case == "foreign", enabled=case == "enabled")
    if case == "damage":
        (store.root / "artifacts" / artifact["sha256"]).write_bytes(b"damage")
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "reference_binding_invalid"
    assert not port.requests


def test_unknown_usage_prevents_retrieval_and_revision(tmp_path):
    _, engine, run, port, _ = setup(tmp_path, unknown=True)
    done = engine.resume(run["id"])
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert "reference_feedback" not in done["state"] and len(port.requests) == 1


def test_maxsteps_one_stops_with_accounted_first_call(tmp_path):
    _, engine, run, port, _ = setup(tmp_path, budget=Budget(max_steps=1))
    done = engine.resume(run["id"])
    assert done["status"] == "budget_exhausted"
    assert done["usage"]["input_tokens"] == 11 and len(port.requests) == 1


def test_unsolicited_tool_call_is_never_executed(tmp_path):
    reply = ModelReply(content="", tool_calls=[ToolCall("x", "ask_user", {"question": "x"})],
                       finish_reason="tool_calls", usage={"input_tokens": 1, "output_tokens": 1})
    store, engine, run, port, _ = setup(tmp_path, replies=[reply])
    done = engine.resume(run["id"])
    assert done["state"]["stop_reason"] == "reference_model_protocol"
    assert len(port.requests) == 1 and not store.list_actions(run["id"])


def test_arm_has_no_first_request_leak_and_active_packet_reaches_second_request(tmp_path):
    requests, host_task_ids, projection_hashes = [], [], []
    for arm in ("plain_revision", "reference_feedback"):
        store, engine, run, port, artifact = setup(tmp_path / arm, arm=arm)
        done = engine.resume(run["id"])
        feedback = done["state"]["reference_feedback"]
        assert feedback["feedback_active"] is True
        assert feedback["host_triggered"] is True
        assert feedback["presented"] == (arm == "reference_feedback")
        pointer = feedback["revision_request"]
        record = json.loads(store.read_artifact(pointer["artifact_id"]))
        assert pointer["projection_hash"] == sha(canonical_json({"messages": port.requests[1]["messages"], "tools": []}))
        assert record["messages"] == port.requests[1]["messages"]
        assert record["max_output_tokens"] == port.requests[0]["max_output_tokens"]
        assert port.requests[1]["messages"][:-1] == port.requests[0]["messages"]
        assert port.requests[1]["messages"][-1]["role"] == "user"
        assert (SOURCE in port.requests[1]["messages"][-1]["content"]) == (arm == "reference_feedback")
        first = canonical_json(port.requests[0])
        assert arm not in first and artifact["id"] not in first and "synthetic_numpy" not in first
        policy = json.loads(port.requests[0]["messages"][0]["content"].split("Host task policy:\n", 1)[1])
        assert "task_id" not in policy
        binding = done["state"]["reference_feedback_binding"]
        assert binding["task_binding"]["id"] == run["task_id"]
        assert store.get_task(run["task_id"])["id"] == run["task_id"]
        assert binding["contract"]["arm"] == arm
        host_task_ids.append(run["task_id"])
        projection_hashes.append(done["state"]["reference_initial_request"]["projection_hash"])
        requests.append(port.requests)
    assert len(set(host_task_ids)) == 2
    # Compare the complete actual ModelPort call, including messages, tools and
    # both limits. No normalization or exception for generated IDs is allowed.
    assert canonical_json(requests[0][0]) == canonical_json(requests[1][0])
    assert projection_hashes[0] == projection_hashes[1]


@pytest.mark.parametrize("duplicate", [False, True])
def test_inactive_candidate_matches_plain_revision_exactly(tmp_path, duplicate):
    outputs = []
    for arm in ("plain_revision", "reference_feedback"):
        _, engine, run, port, _ = setup(tmp_path / arm, arm=arm,
            pack=corpus() if duplicate else {"schema_version": "public-reference-corpus-v1", "sources": []},
            goal=SOURCE if duplicate else "Sum the input values.")
        done = engine.resume(run["id"])
        assert done["status"] == "succeeded"
        assert done["state"]["reference_feedback"]["feedback_active"] is False
        outputs.append(canonical_json(port.requests[1]))
    assert outputs[0] == outputs[1]


def pause_after_prepared(engine, monkeypatch):
    stop = threading.Event()
    original = engine.journal.checkpoint
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reference_feedback_prepared":
            stop.set()
        return result
    monkeypatch.setattr(engine.journal, "checkpoint", checkpoint)
    return stop, original


def test_prepared_feedback_recovers_without_new_lookup(tmp_path, monkeypatch):
    store, engine, run, port, _ = setup(tmp_path)
    stop, original = pause_after_prepared(engine, monkeypatch)
    waiting = engine.resume(run["id"], stop_event=stop)
    assert waiting["status"] == "waiting" and len(port.requests) == 1
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    monkeypatch.setattr("adj_partner.runtime.engine.prepare_reference_feedback", lambda *_: pytest.fail("lookup replay"))
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded" and len(port.requests) == 2


@pytest.mark.parametrize("change", ["pack", "packet", "first_request", "task", "core"])
def test_recovery_rechecks_pack_feedback_original_task_and_core(tmp_path, monkeypatch, change):
    store, engine, run, port, artifact = setup(tmp_path)
    stop, original = pause_after_prepared(engine, monkeypatch)
    waiting = engine.resume(run["id"], stop_event=stop)
    if change in {"pack", "packet", "first_request"}:
        digest = artifact["sha256"] if change == "pack" else waiting["state"][
            "reference_feedback" if change == "packet" else "reference_initial_request"]["sha256"]
        (store.root / "artifacts" / digest).write_bytes(b"changed")
    elif change == "task":
        task = store.get_task(run["task_id"])
        store.update_task(task["id"], {"goal": "Different synthetic task"}, expected_revision=task["revision"])
    else:
        monkeypatch.setattr("adj_partner.runtime.reference_feedback._source_hash", lambda: "f" * 64)
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    done = engine.resume(run["id"])
    assert done["status"] == "failed" and len(port.requests) == 1


def test_context_overflow_keeps_first_call_accounted(tmp_path, monkeypatch):
    _, engine, run, port, _ = setup(tmp_path, replies=["x" * 10000])
    original = engine.context.max_context_tokens
    assert original >= 10000
    engine.context.max_context_tokens = 8000
    done = engine.resume(run["id"])
    assert done["status"] == "budget_exhausted" and done["state"]["stop_reason"] == "context_budget"
    assert len(port.requests) == 1 and done["usage"]["input_tokens"] == 11


def test_interrupted_issued_revision_is_never_replayed(tmp_path, monkeypatch):
    class Crash(BaseException):
        pass
    store, engine, run, port, _ = setup(tmp_path)
    port.replies[1] = lambda: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        engine.resume(run["id"])
    assert len(port.requests) == 2
    saved = store.get_run(run["id"])
    assert saved["state"]["phase"] == "requesting"
    done = engine.resume(run["id"])
    assert done["state"]["stop_reason"] == "interrupted_model_request"
    assert done["usage"]["unknown_requests"] == 1 and len(port.requests) == 2


def test_revision_request_pointer_tamper_fails_on_reply_recovery(tmp_path, monkeypatch):
    stop = threading.Event()
    store, engine, run, port, _ = setup(tmp_path)
    original = engine.journal.checkpoint
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed" and result["state"]["steps"] == 2:
            stop.set()
        return result
    monkeypatch.setattr(engine.journal, "checkpoint", checkpoint)
    waiting = engine.resume(run["id"], stop_event=stop)
    state = waiting["state"]
    state["reference_feedback"]["revision_request"]["sha256"] = "f" * 64
    store.update_run(run["id"], {"state": state}, expected_revision=waiting["revision"])
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    done = engine.resume(run["id"])
    assert done["status"] == "failed" and len(port.requests) == 2


def test_prepared_packet_is_not_claimed_as_presented_before_second_call(tmp_path, monkeypatch):
    _, engine, run, port, _ = setup(tmp_path)
    stop, _ = pause_after_prepared(engine, monkeypatch)
    waiting = engine.resume(run["id"], stop_event=stop)
    assert waiting["state"]["reference_feedback"]["presented"] is False
    assert len(port.requests) == 1


def test_query_uses_task_goal_while_dedup_sees_full_original_context(tmp_path, monkeypatch):
    import adj_partner.public_reference_feedback as core
    original = core.build_feedback
    calls = []
    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(core, "build_feedback", observed)
    store, engine, run, port, _ = setup(tmp_path)
    session = store.get_session(run["session_id"])
    config = session["config"]
    config["instructions"] = "Synthetic unrelated system DOI 10.1234/system."
    store.update_session(session["id"], config, expected_revision=session["revision"])
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert calls[0]["public_text"] == "Sum the input values."
    assert "10.1234/system" in calls[0]["visible_context"]


def test_output_allowance_cannot_shrink_to_enable_extra_call(tmp_path):
    _, engine, run, port, _ = setup(tmp_path, budget=Budget(max_output_tokens=4096))
    done = engine.resume(run["id"])
    assert done["status"] == "budget_exhausted" and len(port.requests) == 1


def test_source_binding_includes_runtime_and_acceptance_code(tmp_path):
    _, _, run, _, _ = setup(tmp_path)
    sources = run["state"]["reference_feedback_binding"]["runtime_source_sha256"]
    assert set(sources) == {"reference_feedback.py", "engine.py", "context.py", "acceptance.py"}
    assert all(len(value) == 64 for value in sources.values())


def test_protocol_finish_without_calls_is_rejected(tmp_path):
    reply = ModelReply(content="No call", tool_calls=[], finish_reason="tool_calls", usage={"input_tokens": 1, "output_tokens": 1})
    _, engine, run, port, _ = setup(tmp_path, replies=[reply])
    assert engine.resume(run["id"])["state"]["stop_reason"] == "reference_model_protocol"
    assert len(port.requests) == 1


def test_detached_unissued_revision_is_resumable_without_usage_charge(tmp_path, monkeypatch):
    stop = threading.Event()
    _, engine, run, port, _ = setup(tmp_path)
    original = engine.journal.checkpoint
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["state"]["phase"] == "requesting" and result["state"]["steps"] == 2:
            stop.set()
        return result
    monkeypatch.setattr(engine.journal, "checkpoint", checkpoint)
    waiting = engine.resume(run["id"], stop_event=stop)
    assert waiting["status"] == "waiting" and waiting["state"]["steps"] == 1
    assert waiting["usage"]["input_tokens"] == 11 and len(port.requests) == 1
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    assert engine.resume(run["id"])["status"] == "succeeded" and len(port.requests) == 2


@pytest.mark.parametrize("phase", ["ready", "reply"])
def test_missing_feedback_checkpoint_cannot_create_a_third_call(tmp_path, monkeypatch, phase):
    store, engine, run, port, _ = setup(tmp_path)
    stop = threading.Event()
    original = engine.journal.checkpoint
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed" and result["state"]["steps"] == 2:
            stop.set()
        return result
    monkeypatch.setattr(engine.journal, "checkpoint", checkpoint)
    waiting = engine.resume(run["id"], stop_event=stop)
    state = waiting["state"]
    state.pop("reference_feedback")
    state["phase"] = phase
    store.update_run(run["id"], {"state": state}, expected_revision=waiting["revision"])
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    done = engine.resume(run["id"])
    assert done["status"] == "failed" and len(port.requests) == 2
    assert done["state"]["stop_reason"] == "reference_revision_state_invalid"


def test_source_text_hash_is_checked_before_first_dispatch(tmp_path):
    bad = corpus()
    bad["sources"][0]["text_sha256"] = "0" * 64
    _, engine, run, port, _ = setup(tmp_path, pack=bad)
    assert engine.resume(run["id"])["state"]["stop_reason"] == "reference_binding_invalid"
    assert not port.requests


def test_valid_second_reply_recovery_delivers_without_a_third_call(tmp_path, monkeypatch):
    _, engine, run, port, _ = setup(tmp_path)
    stop = threading.Event()
    original = engine.journal.checkpoint
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("event") == "runtime.reply_checkpointed" and result["state"]["steps"] == 2:
            stop.set()
        return result
    monkeypatch.setattr(engine.journal, "checkpoint", checkpoint)
    assert engine.resume(run["id"], stop_event=stop)["status"] == "waiting"
    monkeypatch.setattr(engine.journal, "checkpoint", original)
    assert engine.resume(run["id"])["status"] == "succeeded"
    assert len(port.requests) == 2
