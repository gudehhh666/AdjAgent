"""Bounded final-code checking is a runtime treatment, not a new model loop."""

import json

import pytest

from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolResult
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec


SCHEMA = {"type": "object", "properties": {"generated_code": {
    "type": "object", "additionalProperties": {"type": "string"}}},
    "required": ["generated_code"], "additionalProperties": False}


class Port:
    metadata = {"provider": "scripted", "model": "none"}
    secrets = ()

    def __init__(self, codes, unknown=False):
        self.codes, self.requests, self.unknown = list(codes), [], unknown

    def complete(self, messages, tools, **limits):
        self.requests.append(messages)
        return ModelReply(content=json.dumps({"generated_code": {"s": self.codes.pop(0)}}),
                          tool_calls=[], finish_reason="stop", usage={} if self.unknown else {
                              "input_tokens": 1, "output_tokens": 1})


def setup(tmp_path, monkeypatch, codes, *, enforce=True, unknown=False, budget=None,
          check_status=None, pending=False):
    import adj_partner.runtime.engine as runtime
    store = Store(tmp_path / "store")
    catalog, checked = ToolCatalog(), []

    def check(context, generated_code, timeout_seconds=30):
        checked.append(dict(generated_code))
        return ToolResult(status="pending", job_id="job_fixture") if pending else ToolResult(data={"ok": True})

    catalog.register(ToolSpec("public_code_check", "Public checks", {
        "type": "object", "properties": {"generated_code": {"type": "object"}},
        "required": ["generated_code"], "additionalProperties": False}), check)
    monkeypatch.setattr(runtime, "bind_public_code_validation", lambda *_: {"fixture": True})

    def verify(_store, _rid, generated_code):
        matching = [item for item in checked if item == generated_code]
        status = "missing" if not matching else check_status or (
            "passed" if generated_code["s"] == "good" else "failed")
        return {"status": status, "attempt_count": len(checked),
                "generated_code_sha256": generated_code["s"], "checks": []}

    monkeypatch.setattr(runtime, "verify_public_code_validation", verify)
    port = Port(codes, unknown=unknown)
    engine = AgentEngine(store, catalog, port)
    session = store.create_session(SessionConfig(enabled_tools=catalog.names(),
        allowed_tools=catalog.names(), budget=budget or Budget()).to_dict())
    run = engine.start(session["id"], "Produce code", acceptance={"response_json_schema": SCHEMA,
        "public_code_validation": {"manifest_artifact_id": "artifact_fixture",
            "manifest_sha256": "a" * 64, "checker_version": "public-code-check-v1",
            "max_attempts": 2, "enforce": enforce}})
    return store, engine, run, port, checked


def test_failed_draft_gets_one_feedback_then_changed_source_is_rechecked(tmp_path, monkeypatch):
    store, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad", "good"])
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert checked == [{"s": "bad"}, {"s": "good"}]
    assert done["state"]["public_code_repairs"] == 1
    assert done["state"]["delivery"]["public_code_validation"]["status"] == "passed"
    assert json.loads(done["state"]["delivery"]["content"])["generated_code"]["s"] == "good"
    assert len(port.requests) == 2
    assert any("bad" in message["content"] and "public" in message["content"].lower()
               for message in port.requests[1] if isinstance(message.get("content"), str))
    assert len(store.list_actions(run["id"])) == 2


def test_passing_draft_needs_no_repair(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["good"])
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert len(checked) == len(port.requests) == 1
    assert done["state"].get("public_code_repairs", 0) == 0


def test_baseline_gets_same_tool_but_no_forced_check(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad"], enforce=False)
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert checked == []
    assert len(port.requests) == 1
    assert done["state"]["delivery"]["public_code_validation"]["status"] == "missing"


def test_second_failed_draft_is_retained_without_third_attempt(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad", "still_bad"])
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "public_code_validation_failed"
    assert len(checked) == len(port.requests) == 2
    assert "still_bad" in done["state"]["rejected_final_response"]


@pytest.mark.parametrize("status", ["unavailable", "unknown"])
def test_unavailable_check_does_not_grant_free_repair(tmp_path, monkeypatch, status):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad"], check_status=status)
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "public_code_validation_unavailable"
    assert len(checked) == len(port.requests) == 1


def test_unknown_model_usage_stops_before_check_or_repair(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad"], unknown=True)
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "unknown_usage"
    assert not checked and len(port.requests) == 1


def test_budget_prevents_feedback_model_call(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad"], budget=Budget(max_steps=1))
    done = engine.resume(run["id"])
    assert done["status"] == "budget_exhausted"
    assert len(checked) == len(port.requests) == 1


def test_pending_host_check_resumes_same_action_without_replay(tmp_path, monkeypatch):
    store, engine, run, port, checked = setup(tmp_path, monkeypatch, ["good"], pending=True)
    waiting = engine.resume(run["id"])
    assert waiting["status"] == "waiting"
    assert waiting["state"]["phase"] == "public_code_check"
    engine.resolve_job = lambda *_: ToolResult(data={"ok": True})
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    assert len(checked) == len(port.requests) == len(store.list_actions(run["id"])) == 1


def test_completed_check_without_matching_receipt_is_not_replayed(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["good"])
    monkeypatch.setattr("adj_partner.runtime.engine.verify_public_code_validation",
                        lambda *_: {"status": "missing", "attempt_count": 0, "checks": []})
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "public_code_check_receipt_missing"
    assert len(checked) == len(port.requests) == 1


def test_old_pass_does_not_certify_changed_final_source(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["changed_bad"])
    checked.append({"s": "good"})
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert checked == [{"s": "good"}, {"s": "changed_bad"}]
    assert len(port.requests) == 1


def test_task_change_while_check_waits_stops_without_replay(tmp_path, monkeypatch):
    store, engine, run, port, checked = setup(tmp_path, monkeypatch, ["good"], pending=True)
    waiting = engine.resume(run["id"])
    task = store.get_task(waiting["task_id"])
    store.update_task(task["id"], {"goal": "Different task"}, expected_revision=task["revision"])
    engine.resolve_job = lambda *_: pytest.fail("Stale task must not resolve or dispatch a check")
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "task_changed"
    assert len(checked) == len(port.requests) == 1


def test_bad_host_binding_fails_before_model_or_tool(tmp_path, monkeypatch):
    store, engine, run, port, checked = setup(tmp_path, monkeypatch, ["good"])
    current = store.get_run(run["id"])
    state = dict(current["state"], public_code_binding_error="manifest identity mismatch")
    store.update_run(run["id"], {"state": state}, expected_revision=current["revision"])
    done = engine.resume(run["id"])
    assert done["state"]["stop_reason"] == "public_code_binding_invalid"
    assert not checked and not port.requests


def test_baseline_verification_error_is_recorded_without_gating_delivery(tmp_path, monkeypatch):
    _, engine, run, port, checked = setup(tmp_path, monkeypatch, ["bad"], enforce=False)
    def broken(*_):
        raise ValueError("damaged checker receipt")
    monkeypatch.setattr("adj_partner.runtime.engine.verify_public_code_validation", broken)
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded"
    diagnostic = done["state"]["delivery"]["public_code_validation"]
    assert diagnostic["status"] == "unavailable" and "damaged checker receipt" in diagnostic["verification_error"]
    assert not checked and len(port.requests) == 1
