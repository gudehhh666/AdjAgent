"""Verify real durable job/action/artifact chains without model or OS execution."""
import copy

import pytest

from adj_partner.analysis import public_code_capabilities as capability
from adj_partner.analysis.public_code_checker import submission_sha256, validate_manifest
from adj_partner.contracts import Budget, ModelReply, SessionConfig, ToolCall
from adj_partner.execution import JobManager
from adj_partner.execution.context import JobContext
from adj_partner.research.acceptance import task_binding
from adj_partner.runtime.engine import AgentEngine
from adj_partner.runtime.public_code_validation import PublicValidationFailure, verify_public_code_validation
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


MANIFEST = {"schema_version": "public-code-check-v1", "source_path": "synthetic/contract.json",
    "source_sha256": "a" * 64, "ordered_steps": [{"step_id": "s", "function_name": "solve",
        "signature": "def solve(x):", "checks": [
            {"id": "compile", "kind": "compile", "args": [], "kwargs": {},
             "constraints": {}, "public_basis": "Must compile."},
            {"id": "signature", "kind": "signature", "args": [], "kwargs": {},
             "constraints": {}, "public_basis": "Use stated signature."}]}]}
CODE = {"s": "def solve(x):\n    return x\n"}
SCHEMA = {"type": "object", "properties": {"generated_code": {"type": "object"}},
          "required": ["generated_code"], "additionalProperties": False}


class Port:
    secrets = ()
    metadata = {"provider": "scripted", "model": "none"}

    def complete(self, *args, **kwargs):
        import json
        return ModelReply(content=json.dumps({"generated_code": CODE}), tool_calls=[], finish_reason="stop",
                          usage={"input_tokens": 1, "output_tokens": 1})


def fixture(tmp_path, monkeypatch, *, false_status=False, enforce=True):
    store, catalog = Store(tmp_path / "store"), ToolCatalog()
    jobs = JobManager(store)
    monkeypatch.setattr(jobs, "_launch_if_unclaimed", lambda *_: None)
    caps = capability.PublicCodeCapabilities(store, jobs=jobs)
    caps.register(catalog)
    sid = store.create_session(SessionConfig(enabled_tools=catalog.names(), allowed_tools=catalog.names(),
                                             budget=Budget()).to_dict())["id"]
    artifact = capability.install_public_check_manifest(store, sid, copy.deepcopy(MANIFEST))
    engine = AgentEngine(store, catalog, Port(), resolve_job=caps.resolve_job, recover_action=caps.recover_action)
    run = engine.start(sid, "Generate solve", acceptance={"response_json_schema": SCHEMA,
        "public_code_validation": {"manifest_artifact_id": artifact["id"], "manifest_sha256": artifact["sha256"],
            "checker_version": "public-code-check-v1", "max_attempts": 2, "enforce": enforce}})

    def fake_check(generated_code, manifest, workspace, **kwargs):
        checks = [{"id": check["id"], "kind": check["kind"], "message": "scripted test fixture",
                   "status": "unavailable" if check["kind"] == "unavailable" else "passed"}
                  for step in validate_manifest(manifest) for check in step["checks"]]
        if false_status:
            checks[0]["status"] = "failed"
        return {"status": "passed", "schema_version": "public-code-check-result-v1", "checks": checks,
                "generated_code_sha256": submission_sha256(generated_code),
                "manifest_sha256": artifact["sha256"], "checker_source_sha256": kwargs["checker_source_sha256"]}

    monkeypatch.setattr(capability, "run_public_code_checks", fake_check)
    return store, jobs, engine, run


def complete_pending(store, jobs, run, index=0):
    attempts = capability.list_attempts(store, run["id"])
    job = jobs._claim(attempts[index]["job_id"])
    jobs._prepare_workspace(job)
    context = JobContext(jobs, job)
    result = capability.public_code_check_job(context, **job["arguments"])
    jobs._finish(job, "succeeded", result)
    return result


def test_exact_final_code_matches_durable_action_job_and_artifact(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch)
    waiting = engine.resume(run["id"])
    assert waiting["status"] == "waiting"
    complete_pending(store, jobs, run)
    done = engine.resume(run["id"])
    assert done["status"] == "succeeded", done["state"]
    verified = verify_public_code_validation(store, run["id"], CODE)
    assert verified["status"] == "passed"
    assert verified["declared_unavailable_assertions"] == 1
    assert verified["attempt_count"] == 1
    assert verified["identity"]["task_binding"] == done["state"]["public_code_validation_binding"]["task_binding"]
    assert verified["selected_attempt"] == verified["latest_attempt"] == 1
    changed = verify_public_code_validation(store, run["id"], {"s": "def solve(x):\n    return 0\n"})
    assert changed["status"] == "missing"


def test_forged_report_pass_disagrees_with_individual_assertions(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch, false_status=True)
    engine.resume(run["id"])
    complete_pending(store, jobs, run)
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "public_code_verification_failed"
    with pytest.raises(PublicValidationFailure, match="summary disagrees"):
        verify_public_code_validation(store, run["id"], CODE)


def test_checker_source_drift_after_completion_fails_closed(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch)
    engine.resume(run["id"])
    complete_pending(store, jobs, run)
    engine.resume(run["id"])
    monkeypatch.setattr("adj_partner.runtime.public_code_validation.checker_source_digest", lambda: "b" * 64)
    with pytest.raises(PublicValidationFailure, match="binding or checker source changed"):
        verify_public_code_validation(store, run["id"], CODE)


def test_manifest_in_another_session_cannot_be_bound(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch)
    sid2 = store.create_session(SessionConfig(budget=Budget()).to_dict())["id"]
    acceptance = store.get_task(run["task_id"])["acceptance"]
    other = engine.start(sid2, "Other task", acceptance=acceptance)
    assert "public_code_binding_error" in other["state"]
    done = engine.resume(other["id"])
    assert done["state"]["stop_reason"] == "public_code_binding_invalid"


def test_reverting_after_later_checked_source_cannot_reuse_old_pass(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch)
    engine.resume(run["id"])
    complete_pending(store, jobs, run)
    second = ToolExecutor(store, engine.catalog).execute(ToolCall(id="later_check", name="public_code_check",
        arguments={"generated_code": {"s": "def solve(x):\n    return x + 1\n"}}),
        run["session_id"], run["id"])
    assert second.status == "pending"
    complete_pending(store, jobs, run, index=1)
    reverted = verify_public_code_validation(store, run["id"], CODE)
    assert reverted["status"] == "missing"
    assert reverted["selected_attempt"] is None and reverted["latest_attempt"] == 2
    done = engine.resume(run["id"])
    assert done["status"] == "failed"
    assert done["state"]["stop_reason"] == "public_code_check_receipt_missing"


def test_post_run_task_edit_is_not_status_only_terminal_increment(tmp_path, monkeypatch):
    store, jobs, engine, run = fixture(tmp_path, monkeypatch)
    engine.resume(run["id"])
    complete_pending(store, jobs, run)
    engine.resume(run["id"])
    task = store.get_task(run["task_id"])
    store.update_task(task["id"], {"goal": "Edited task"}, expected_revision=task["revision"])
    with pytest.raises(PublicValidationFailure, match="binding or checker source changed"):
        verify_public_code_validation(store, run["id"], CODE)
