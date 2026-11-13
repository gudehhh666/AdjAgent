import hashlib
import json
from pathlib import Path

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall, ToolResult
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.tools.catalog import canonical_json


MANIFEST = {
    "schema_version": "public-code-check-v1",
    "source_path": "public/sample.py",
    "source_sha256": "a" * 64,
    "ordered_steps": [{
        "step_id": "step_1", "function_name": "solve", "signature": "def solve(x):",
        "checks": [{"id": "compile_1", "kind": "compile", "args": [], "kwargs": {},
                    "constraints": {}, "public_basis": "Compile the submitted public function."}],
    }],
}


class FakeJobs:
    def __init__(self, store):
        self.store = store
        self.registrations = {}
        self.submissions = []

    def register(self, spec, handler_ref):
        self.registrations[spec.name] = (spec, handler_ref)

    def submit(self, session_id, run_id, tool_name, arguments, idempotency_key,
               timeout_seconds=300):
        row = {"id": f"job_{len(self.submissions) + 1}", "status": "queued",
               "session_id": session_id, "run_id": run_id, "tool_name": tool_name,
               "handler_ref": self.registrations[tool_name][1], "arguments": arguments}
        self.submissions.append({**row, "idempotency_key": idempotency_key,
                                 "timeout_seconds": timeout_seconds})
        return row


class FakeJobContext:
    def __init__(self, store, session_id, run_id, workspace, *, job_id="job_fixture", call_id="check_1"):
        self.store, self.session_id, self.run_id = store, session_id, run_id
        self.workspace, self.job_id, self.call_id = Path(workspace), job_id, call_id
        self._job = {"deadline": 10**20}
        self.checks = 0

    def check_cancelled(self):
        self.checks += 1


def contract(enforce=True):
    return {"manifest_artifact_id": "artifact_manifest", "manifest_sha256": "b" * 64,
            "checker_version": "public-code-check-v1", "max_attempts": 2, "enforce": enforce}


def setup(tmp_path, *, bind=True):
    from adj_partner.analysis.public_code_capabilities import (
        PublicCodeCapabilities, install_public_check_manifest,
    )
    from adj_partner.research.acceptance import task_binding

    store = Store(tmp_path / "state")
    jobs = FakeJobs(store)
    capabilities = PublicCodeCapabilities(store, jobs=jobs)
    catalog = ToolCatalog()
    capabilities.register(catalog)
    session_id = store.create_session(SessionConfig(
        allowed_roots=[str(tmp_path)], allowed_tools=catalog.names(),
        enabled_tools=catalog.names()).to_dict())["id"]
    manifest_artifact = install_public_check_manifest(store, session_id, MANIFEST)
    acceptance = {"public_code_validation": {
        **contract(), "manifest_artifact_id": manifest_artifact["id"],
        "manifest_sha256": manifest_artifact["sha256"]}, "response_json_schema": {
        "type": "object", "properties": {"generated_code": {"type": "object"}},
        "required": ["generated_code"], "additionalProperties": True}}
    task = store.create_task(session_id, "Implement solve", acceptance=acceptance)
    run = store.create_run(session_id, task["id"], Budget().to_dict())
    checker_sha = capabilities.checker_source_digest()
    handler_sha = capabilities.handler_source_digest()
    binding = {"task_binding": task_binding(task),
               "manifest_artifact_id": manifest_artifact["id"], "manifest_sha256": manifest_artifact["sha256"],
               "checker_version": "public-code-check-v1", "checker_source_sha256": checker_sha,
               "handler_source_sha256": handler_sha, "max_attempts": 2}
    if bind:
        state = run["state"]
        state["public_code_validation_binding"] = binding
        store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
    return store, jobs, capabilities, catalog, ToolExecutor(store, catalog), session_id, run["id"], binding


def test_registers_separate_public_checker_job_and_tool(tmp_path):
    _, jobs, _, catalog, _, _, _, _ = setup(tmp_path)

    assert "public_code_check" in catalog.names()
    assert jobs.registrations["public_code_check"][1] == (
        "adj_partner.analysis.public_code_capabilities:public_code_check_job")
    spec = catalog.get("public_code_check").spec
    assert spec.network is False and spec.requires_approval is False
    assert spec.input_schema["required"] == ["generated_code"]


def test_manifest_install_and_load_are_session_scoped_and_hash_pinned(tmp_path):
    from adj_partner.analysis.public_code_capabilities import install_public_check_manifest, load_manifest

    store = Store(tmp_path / "state")
    one = store.create_session(SessionConfig().to_dict())["id"]
    two = store.create_session(SessionConfig().to_dict())["id"]
    artifact = install_public_check_manifest(store, one, MANIFEST)
    binding = {"manifest_artifact_id": artifact["id"], "manifest_sha256": artifact["sha256"],
               "checker_version": "public-code-check-v1"}

    assert load_manifest(store, one, binding) == MANIFEST
    with pytest.raises((KeyError, PermissionError, ValueError)):
        load_manifest(store, two, binding)
    with pytest.raises(ValueError, match="hash"):
        load_manifest(store, one, {**binding, "manifest_sha256": "0" * 64})


def test_read_only_attempt_ledger_does_not_create_table(tmp_path):
    from adj_partner.analysis.public_code_capabilities import list_attempts

    store = Store(tmp_path / "state")
    assert list_attempts(store, "run_missing") == []
    with store.transaction() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='public_code_check_attempts'").fetchone() is None


def test_tool_reserves_attempt_and_submits_exact_candidate_as_durable_job(tmp_path, monkeypatch):
    import adj_partner.analysis.public_code_capabilities as public_caps

    store, jobs, caps, _, executor, session, run, binding = setup(tmp_path)
    monkeypatch.setattr(public_caps, "checker_source_digest", lambda: binding["checker_source_sha256"])
    monkeypatch.setattr(public_caps, "run_public_code_checks", lambda code, manifest, workspace, **kw: {
        "status": "passed", "generated_code_sha256": "e" * 64,
        "checks": [{"id": "compile_1", "kind": "compile", "status": "passed", "message": "ok"}],
        "unavailable_assertions": [], "coverage": {"executable_checks": 1, "passed": 1,
            "failed": 0, "unavailable": 0, "declared_unavailable": 0},
    })

    result = executor.execute(ToolCall("check_call_1", "public_code_check", {
        "generated_code": {"step_1": "def solve(x):\n return x\n"}, "timeout_seconds": 5,
    }), session, run)

    assert result.status == "pending"
    assert len(jobs.submissions) == 1
    assert jobs.submissions[0]["arguments"]["generated_code"] == {"step_1": "def solve(x):\n return x\n"}
    rows = caps.list_attempts(store, run)
    assert len(rows) == 1 and rows[0]["call_id"] == "check_call_1" and rows[0]["attempt"] == 1
    assert rows[0]["manifest_artifact_id"] == binding["manifest_artifact_id"]
    artifact = store.artifacts(session, run)[-1]
    assert artifact["provenance"]["kind"] == "public_code_submission"
    repeated = executor.execute(ToolCall("check_call_1", "public_code_check", {
        "generated_code": {"step_1": "def solve(x):\n return x\n"}, "timeout_seconds": 5,
    }), session, run)
    assert repeated.status == "pending" and len(jobs.submissions) == 1


def test_checker_job_persists_failed_check_as_successful_validation_report(tmp_path, monkeypatch):
    import adj_partner.analysis.public_code_capabilities as public_caps

    store, jobs, caps, _, executor, session, run, binding = setup(tmp_path)
    monkeypatch.setattr(public_caps, "checker_source_digest", lambda: binding["checker_source_sha256"])
    monkeypatch.setattr(public_caps, "run_public_code_checks", lambda code, manifest, workspace, **kw: {
        "status": "failed", "generated_code_sha256": hashlib.sha256(canonical_json(code).encode()).hexdigest(),
        "checks": [{"id": "compile_1", "kind": "compile", "status": "failed", "message": "syntax"}],
        "unavailable_assertions": [], "coverage": {"executable_checks": 1, "passed": 0,
            "failed": 1, "unavailable": 0, "declared_unavailable": 0},
    })
    code = {"step_1": "def solve(:"}
    queued = executor.execute(ToolCall("call_job", "public_code_check", {
        "generated_code": code, "timeout_seconds": 5,
    }), session, run)
    submitted = jobs.submissions[0]
    identity = {key: value for key, value in submitted["arguments"].items()
                if key not in {"generated_code", "timeout_seconds"}}
    ctx = FakeJobContext(store, session, run, tmp_path / "job", job_id=submitted["id"], call_id="call_job")
    ctx.workspace.mkdir()
    assert queued.status == "pending"

    result = public_caps.public_code_check_job(ctx, **submitted["arguments"])

    assert result.status == "ok"
    assert result.data["validation"]["status"] == "failed"
    assert result.data["identity"] == {**identity, "job_id": submitted["id"]}
    assert ctx.checks >= 2


def test_checker_aborts_if_task_binding_changes_during_sandbox_poll(tmp_path, monkeypatch):
    import adj_partner.analysis.public_code_capabilities as public_caps
    from adj_partner.tools import ToolFailure

    store, jobs, _, _, executor, session, run, binding = setup(tmp_path)
    monkeypatch.setattr(public_caps, "checker_source_digest", lambda: binding["checker_source_sha256"])
    code = {"step_1": "def solve(x): return x"}
    queued = executor.execute(ToolCall("binding_poll", "public_code_check", {
        "generated_code": code, "timeout_seconds": 5,
    }), session, run)
    assert queued.status == "pending"
    submitted = jobs.submissions[0]
    task = store.get_task(store.get_run(run)["task_id"])
    ctx = FakeJobContext(store, session, run, tmp_path / "job", job_id=submitted["id"], call_id="binding_poll")
    ctx.workspace.mkdir()

    def mutate_then_poll(*args, cancel_check, **kwargs):
        changed = store.get_task(task["id"])
        changed["acceptance"]["public_code_validation"]["enforce"] = False
        store.update_task(changed["id"], {"acceptance": changed["acceptance"]},
                          expected_revision=changed["revision"])
        # Simulate a checker loop that catches callback errors while cleaning
        # up. The handler must still notice and fail closed after it returns.
        try:
            cancel_check()
        except ToolFailure:
            pass
        return {"status": "passed", "generated_code_sha256": public_caps.checker.submission_sha256(code),
                "checks": [], "unavailable_assertions": [],
                "coverage": {"executable_checks": 0, "passed": 0, "failed": 0,
                             "unavailable": 0, "declared_unavailable": 0}}

    monkeypatch.setattr(public_caps, "run_public_code_checks", mutate_then_poll)
    with pytest.raises(ToolFailure, match="task no longer matches"):
        public_caps.public_code_check_job(ctx, **submitted["arguments"])


def test_recovery_of_missing_job_becomes_uncertain_without_replay(tmp_path, monkeypatch):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    original = ToolCall("missing_public_job", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    })
    pending = executor.execute(original, session, run)
    assert pending.status == "pending" and len(jobs.submissions) == 1

    monkeypatch.setattr(caps.jobs, "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("Unknown job")), raising=False)
    recovered = caps.recover_action(session, run, original)

    assert recovered.status == "uncertain"
    assert recovered.error["code"] == "public_code_job_missing"
    assert len(jobs.submissions) == 1
    assert caps.list_attempts(store, run)[0]["status"] == "uncertain"


@pytest.mark.parametrize("failure", [KeyError("missing result"), ValueError("result hash mismatch")],
                         ids=["missing-result-artifact", "corrupt-result-artifact"])
def test_recovery_of_missing_or_corrupt_result_becomes_uncertain(tmp_path, failure):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    original = ToolCall("bad_public_result", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    })
    pending = executor.execute(original, session, run)
    job = jobs.submissions[0]
    assert pending.status == "pending"

    class DamagedResultJobs:
        def get(self, job_id, session_id=None):
            assert job_id == job["id"] and session_id == session
            return {**job, "status": "succeeded"}

        def reconcile(self, job_id, session_id=None):
            return self.get(job_id, session_id)

        def result(self, job_id, session_id=None):
            raise failure

    caps.jobs = DamagedResultJobs()
    recovered = caps.recover_action(session, run, original)

    assert recovered.status == "uncertain"
    assert recovered.error["code"] == "public_code_result_unavailable"
    assert caps.list_attempts(store, run)[0]["status"] == "uncertain"


def test_recovery_of_malformed_result_payload_becomes_uncertain(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    original = ToolCall("malformed_public_result", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    })
    pending = executor.execute(original, session, run)
    job = jobs.submissions[0]
    assert pending.status == "pending"

    class MalformedResultJobs:
        def get(self, job_id, session_id=None):
            return {**job, "status": "succeeded"}

        def reconcile(self, job_id, session_id=None):
            return self.get(job_id, session_id)

        @staticmethod
        def result(job_id, session_id=None):
            return ToolResult(status="ok", data=None)

    caps.jobs = MalformedResultJobs()
    recovered = caps.recover_action(session, run, original)

    assert recovered.status == "uncertain"
    assert recovered.error["code"] == "public_code_result_unavailable"
    assert caps.list_attempts(store, run)[0]["status"] == "uncertain"


def test_same_call_id_with_changed_source_cannot_recover_original_result(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    original = ToolCall("reused_public_call", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    })
    pending = executor.execute(original, session, run)
    assert pending.status == "pending"

    changed = ToolCall("reused_public_call", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return 0"},
    })
    recovered = caps.recover_action(session, run, changed)

    assert recovered.status == "uncertain"
    assert recovered.error["code"] == "public_code_recovery_mismatch"
    assert len(jobs.submissions) == 1
    assert caps.list_attempts(store, run)[0]["status"] == "submitted"


def test_recovery_rejects_job_args_changed_after_original_action(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    original = ToolCall("tampered_public_job", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    })
    pending = executor.execute(original, session, run)
    job = jobs.submissions[0]
    assert pending.status == "pending"

    class ChangedJobArgs:
        def get(self, job_id, session_id=None):
            return {**job, "status": "queued", "arguments": {
                **job["arguments"], "generated_code": {"step_1": "def solve(x): return 99"}}}

        def reconcile(self, *args, **kwargs):
            pytest.fail("Mismatched durable job arguments must be rejected before reconcile")

    caps.jobs = ChangedJobArgs()
    recovered = caps.recover_action(session, run, original)

    assert recovered.status == "uncertain"
    assert recovered.error["code"] == "public_code_recovery_mismatch"
    assert len(jobs.submissions) == 1
    assert caps.list_attempts(store, run)[0]["status"] == "uncertain"


def test_unbound_or_task_mismatched_tool_is_rejected_before_job_and_attempt(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path, bind=False)

    rejected = executor.execute(ToolCall("unbound_call", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    }), session, run)

    assert rejected.status in {"denied", "error"}
    assert jobs.submissions == []
    assert caps.list_attempts(store, run) == []


def test_changed_task_contract_rejects_before_attempt_reservation(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    task = store.get_task(store.get_run(run)["task_id"])
    task["acceptance"]["public_code_validation"]["manifest_sha256"] = "0" * 64
    store.update_task(task["id"], {"acceptance": task["acceptance"]}, expected_revision=task["revision"])

    rejected = executor.execute(ToolCall("changed_task", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    }), session, run)

    assert rejected.status == "denied"
    assert rejected.error["code"] == "public_code_binding_mismatch"
    assert jobs.submissions == [] and caps.list_attempts(store, run) == []


def test_attempt_budget_is_atomic_and_capped_at_two(tmp_path):
    store, jobs, caps, _, executor, session, run, binding = setup(tmp_path)
    assert callable(caps.list_attempts)

    for n in range(2):
        result = executor.execute(ToolCall(f"check_{n}", "public_code_check", {
            "generated_code": {"step_1": f"def solve(x): return {n}"},
        }), session, run)
        assert result.status == "pending"
    third = executor.execute(ToolCall("check_3", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return 3"},
    }), session, run)

    assert third.status in {"denied", "error"}
    assert len(jobs.submissions) == 2
    assert [row["attempt"] for row in caps.list_attempts(store, run)] == [1, 2]


def test_reserved_attempt_identity_and_attached_job_cannot_be_rewritten(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)
    result = executor.execute(ToolCall("immutable", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"},
    }), session, run)
    assert result.status == "pending"
    with pytest.raises(Exception, match="identity is immutable"):
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE public_code_check_attempts SET generated_code_sha256=? WHERE run_id=? AND call_id=?",
                         ("0" * 64, run, "immutable"))
    with pytest.raises(Exception, match="job identity is immutable"):
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE public_code_check_attempts SET job_id=? WHERE run_id=? AND call_id=?",
                         ("job_other", run, "immutable"))


def test_checker_timeout_is_clamped_to_runtime_guard_budget(tmp_path):
    store, jobs, caps, _, executor, session, run, _ = setup(tmp_path)

    class Guard:
        def __call__(self):
            return None

        @staticmethod
        def remaining_seconds():
            return 6.7

    result = executor.execute(ToolCall("guarded_check", "public_code_check", {
        "generated_code": {"step_1": "def solve(x): return x"}, "timeout_seconds": 20,
    }), session, run, execution_guard=Guard())

    assert result.status == "pending"
    assert caps.list_attempts(store, run)[0]["timeout_seconds"] == 4
    # Checker receives the floor(remaining - 2s) budget; the detached job gets
    # only the same 2s cleanup overhead, so the job itself cannot outlive the
    # run's guard budget.
    assert jobs.submissions[0]["timeout_seconds"] == 6
    assert jobs.submissions[0]["timeout_seconds"] <= 6.7


def test_application_registers_and_routes_public_checker_jobs(tmp_path, monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.models.base import ScriptedPort

    service = AgentService(tmp_path / "service", model=ScriptedPort([]), project_root=tmp_path)
    try:
        assert "public_code_check" in service.catalog.names()
        sid = service.create_session()["id"]
        assert "public_code_check" in service.session(sid)["config"]["allowed_tools"]
        from adj_partner.tools.policy import SessionToolset
        SessionToolset(service.store, service.catalog, sid).enable("public_code_check")
        assert "public_code_check" in service.session(sid)["config"]["enabled_tools"]
        monkeypatch.setattr(service.native.jobs, "get", lambda job_id, session_id=None: {
            "handler_ref": "adj_partner.analysis.public_code_capabilities:public_code_check_job"})
        sentinel = ToolResult(data={"routed": True})
        monkeypatch.setattr(service.public_code, "resolve_job", lambda *args: sentinel)

        assert service._resolve_job("session_fixture", "run_fixture", "job_fixture") is sentinel
    finally:
        service.close()


def test_profile_exposes_public_checker_in_opted_in_toolset():
    from adj_partner.profiles import PROFILES

    assert "public_code_check" in PROFILES["general"]["tools"]

