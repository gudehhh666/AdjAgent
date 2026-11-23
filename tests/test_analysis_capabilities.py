import hashlib
import json

import pytest

from adj_partner.analysis.history import AnalysisHistory
from adj_partner.contracts import Budget, SessionConfig, ToolCall, ToolResult
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


ANALYSIS_TOOLS = {
    "code_interpreter", "bash", "analysis_method_register",
    "analysis_method_run", "analysis_history", "analysis_history_read",
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
        self.submissions.append({"session_id": session_id, "run_id": run_id,
            "tool_name": tool_name, "arguments": arguments,
            "idempotency_key": idempotency_key, "timeout_seconds": timeout_seconds})
        return {"id": "job_analysis_fixture", "status": "queued"}


class TerminalJobs(FakeJobs):
    def __init__(self, store):
        super().__init__(store)
        self.status = "queued"

    def _job(self):
        submitted = self.submissions[0]
        return {"id": "job_analysis_fixture", "status": self.status,
            "session_id": submitted["session_id"], "run_id": submitted["run_id"],
            "tool_name": submitted["tool_name"], "handler_ref": "adj_partner.analysis.worker:analysis_job",
            "arguments": submitted["arguments"]}

    def get(self, job_id, session_id=None):
        return self._job()

    def reconcile(self, job_id, session_id=None):
        return self._job()

    def result(self, job_id, session_id=None):
        status = {"cancelled": "cancelled", "uncertain": "uncertain"}.get(self.status, "error")
        return ToolResult(status=status,
            error={"code": self.status, "message": f"Outer job became {self.status}"}, job_id=job_id)


class FailingJobs(FakeJobs):
    def submit(self, *args, **kwargs):
        raise RuntimeError("synthetic submission failure with /private/path")


class FinalizingJobs(FakeJobs):
    """Simulate a worker that commits its result before submit() returns."""

    def submit(self, session_id, run_id, tool_name, arguments, idempotency_key,
               timeout_seconds=300):
        from adj_partner.analysis.history import finalize_analysis_job

        job = super().submit(session_id, run_id, tool_name, arguments, idempotency_key,
                             timeout_seconds)
        durable = {**job, "session_id": session_id, "run_id": run_id,
                   "tool_name": tool_name, "arguments": arguments}
        history = AnalysisHistory(self.store)
        history.mark_running(arguments["analysis_id"], job["id"])
        result = ToolResult(data={"analysis_id": arguments["analysis_id"],
            "summary": {"fast": True}, "sandbox_backend": "fixture"}, job_id=job["id"])
        with self.store.transaction(write=True) as conn:
            finalize_analysis_job(conn, self.store, durable, "succeeded", result)
        return {**job, "status": "succeeded"}


def setup(tmp_path, jobs_type=FakeJobs):
    from adj_partner.analysis.capabilities import AnalysisCapabilities

    store = Store(tmp_path / "state")
    jobs = jobs_type(store)
    capabilities = AnalysisCapabilities(store, jobs=jobs)
    catalog = ToolCatalog()
    capabilities.register(catalog)
    session = store.create_session(SessionConfig(
        allowed_roots=[str(tmp_path)], allowed_tools=catalog.names(),
        enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Analyze a workbook")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return store, jobs, capabilities, catalog, ToolExecutor(store, catalog), session, run


def execute(executor, session_id, run_id, name, arguments, call_id="analysis_call"):
    return executor.execute(ToolCall(call_id, name, arguments), session_id, run_id)


def test_registers_exact_public_analysis_tool_names_and_trusted_jobs(tmp_path):
    _, jobs, _, catalog, _, _, _ = setup(tmp_path)

    assert ANALYSIS_TOOLS <= set(catalog.names())
    assert set(jobs.registrations) == {"code_interpreter", "bash", "analysis_method_run"}
    assert all(handler == "adj_partner.analysis.worker:analysis_job"
               for _, handler in jobs.registrations.values())
    assert catalog.get("code_interpreter").spec.network is False
    assert catalog.get("bash").spec.network is False
    assert "requires Linux Bubblewrap" in catalog.get("bash").spec.description
    assert "macOS returns sandbox_unavailable" in catalog.get("bash").spec.description


def test_code_interpreter_snapshots_authorized_input_manifest_and_code(tmp_path):
    source = tmp_path / "table.csv"
    source.write_bytes(b"gene,value\nA,1\n")
    store, jobs, _, _, executor, session, run = setup(tmp_path)
    code = "print('analyze')\n"

    result = execute(executor, session, run, "code_interpreter", {
        "code": code, "inputs": [str(source)], "title": "Count table rows",
        "timeout_seconds": 45,
    })

    assert result.status == "pending" and result.job_id == "job_analysis_fixture"
    submitted = jobs.submissions[0]
    assert submitted["tool_name"] == "code_interpreter"
    assert submitted["idempotency_key"] == "analysis_call"
    assert submitted["timeout_seconds"] == 55
    manifest = submitted["arguments"]["inputs"][0]
    assert manifest == {"path": str(source), "name": "table.csv", "size": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    record = AnalysisHistory(store).get_record(submitted["arguments"]["analysis_id"])
    assert record["job_id"] == "job_analysis_fixture" and record["status"] == "pending"
    assert record["title"] == "Count table rows" and record["inputs"] == [manifest]
    assert store.read_artifact(record["code_artifact_id"]).decode() == code


def test_duplicate_input_basenames_are_denied_before_job_submission(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir(); second.mkdir()
    (first / "same.csv").write_text("a\n", encoding="utf-8")
    (second / "same.csv").write_text("b\n", encoding="utf-8")
    _, jobs, _, _, executor, session, run = setup(tmp_path)

    result = execute(executor, session, run, "bash", {
        "script": "true\n", "inputs": [str(first / "same.csv"), str(second / "same.csv")],
        "title": "collision",
    })

    assert result.status == "denied"
    assert result.error["code"] == "analysis_input_denied"
    assert jobs.submissions == []


def successful_record(store, session, run):
    history = AnalysisHistory(store)
    code = store.put_artifact("print('method')\n", "analysis.py", "text/x-python", {}, session, run)
    record = history.create_record(session, run, language="python", title="source",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})
    history.finish(record["id"], status="succeeded", summary={"ok": True}, artifacts=[],
                   sandbox_backend="fixture")
    return history, record, code


def test_register_and_run_versioned_method_in_a_later_session(tmp_path):
    store, jobs, _, _, executor, session, run = setup(tmp_path)
    history, record, original_code = successful_record(store, session, run)
    registered = execute(executor, session, run, "analysis_method_register", {
        "record_id": record["id"], "name": "count_rows", "description": "Count rows",
        "parameter_schema": {"type": "object", "properties": {"minimum": {"type": "integer"}},
                             "required": ["minimum"], "additionalProperties": False},
    }, "register_method")
    assert registered.status == "ok" and registered.data["method"]["version"] == 1

    later = store.create_session(SessionConfig(
        allowed_roots=[str(tmp_path)], allowed_tools=sorted(ANALYSIS_TOOLS),
        enabled_tools=sorted(ANALYSIS_TOOLS)).to_dict())["id"]
    later_task = store.create_task(later, "Reuse method")["id"]
    later_run = store.create_run(later, later_task, Budget().to_dict())["id"]
    later_executor = ToolExecutor(store, executor.catalog)
    result = execute(later_executor, later, later_run, "analysis_method_run", {
        "method_id": registered.data["method"]["id"], "parameters": {"minimum": 2},
        "inputs": [], "title": "Repeat row count",
    }, "run_method")

    assert result.status == "pending"
    submitted = jobs.submissions[-1]["arguments"]
    rerun = history.get_record(submitted["analysis_id"])
    assert rerun["method_id"] == registered.data["method"]["id"]
    assert rerun["session_id"] == later and rerun["run_id"] == later_run
    assert rerun["code_artifact_id"] != original_code["id"]
    assert store.read_artifact(rerun["code_artifact_id"]) == store.read_artifact(original_code["id"])


def test_method_parameter_schema_is_enforced_and_history_is_project_wide(tmp_path):
    store, jobs, _, _, executor, session, run = setup(tmp_path)
    history, record, _ = successful_record(store, session, run)
    method = history.register_method(session, record["id"], name="requires_number",
        description="Requires n", parameter_schema={"type": "object",
            "properties": {"n": {"type": "number"}}, "required": ["n"],
            "additionalProperties": False})

    invalid = execute(executor, session, run, "analysis_method_run", {
        "method_id": method["id"], "parameters": {"n": "wrong"}, "title": "invalid",
    }, "invalid_method")
    found = execute(executor, session, run, "analysis_history", {
        "query": "source", "method_name": "requires_number"}, "history_search")

    assert invalid.status == "error" and invalid.error["code"] == "analysis_method_schema"
    assert jobs.submissions == []
    assert found.status == "ok" and found.data["records"][0]["id"] == record["id"]

    oversized = execute(executor, session, run, "analysis_method_run", {
        "method_id": method["id"], "parameters": {"n": "x" * 70000}, "title": "oversized",
    }, "oversized_method")
    assert oversized.status == "error" and oversized.error["code"] == "analysis_method_schema"
    assert jobs.submissions == []


def test_analysis_history_read_only_accepts_analysis_linked_text(tmp_path):
    store, _, _, _, executor, session, run = setup(tmp_path)
    history, record, code = successful_record(store, session, run)
    readable = execute(executor, session, run, "analysis_history_read", {
        "artifact_id": code["id"], "max_characters": 200,
    }, "read_history")
    unrelated = store.put_artifact(json.dumps({"secret": False}), "unrelated.json",
        "application/json", {}, session, run)
    denied = execute(executor, session, run, "analysis_history_read", {
        "artifact_id": unrelated["id"]}, "read_unrelated")

    assert readable.status == "ok" and "method" in readable.data["content"]
    assert denied.status == "denied" and denied.error["code"] == "analysis_history_denied"


def test_durable_recovery_matches_the_original_public_call_exactly(tmp_path, monkeypatch):
    from adj_partner.analysis.capabilities import AnalysisCapabilities

    store = Store(tmp_path / "state")
    capability = AnalysisCapabilities(store)
    catalog = ToolCatalog(); capability.register(catalog)
    monkeypatch.setattr(capability.jobs, "_launch_if_unclaimed", lambda job: None)
    session = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)],
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "recover analysis")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    executor = ToolExecutor(store, catalog)
    original = ToolCall("recover_call", "code_interpreter", {
        "code": "print('original')\n", "title": "recoverable"})
    pending = executor.execute(original, session, run)

    recovered = capability.recover_action(session, run, original)

    assert pending.job_id == recovered.job_id and recovered.status == "pending"
    with pytest.raises(ValueError, match="conflicts"):
        capability.recover_action(session, run, ToolCall("recover_call", "code_interpreter", {
            "code": "print('changed')\n", "title": "recoverable"}))


@pytest.mark.parametrize("outer_status", ["failed", "cancelled", "timed_out", "uncertain"])
def test_outer_terminal_job_always_terminalizes_its_analysis_history(tmp_path, outer_status):
    store, jobs, capability, _, executor, session, run = setup(tmp_path, TerminalJobs)
    pending = execute(executor, session, run, "code_interpreter", {
        "code": "print('never entered')\n", "title": f"outer {outer_status}"})
    jobs.status = outer_status

    result = capability.resolve_job(session, run, pending.job_id)
    record = capability.history.get_record(pending.data["analysis_id"])

    assert result.status in {"error", "cancelled", "uncertain"}
    assert record["status"] == outer_status
    assert record["sandbox_backend"] == "outer_job_supervisor"
    assert record["error"]["code"] == outer_status


def test_submission_failure_records_a_stable_terminal_outcome(tmp_path):
    _, _, capability, _, executor, session, run = setup(tmp_path, FailingJobs)

    result = execute(executor, session, run, "code_interpreter", {
        "code": "print('not submitted')\n", "title": "submission failure"})
    record = capability.history.search(session, query="submission failure")["records"][0]

    assert result.status == "error"
    assert result.error == {"code": "analysis_submission_failed",
                            "message": "The analysis job could not be submitted durably"}
    assert record["status"] == "failed"
    assert record["error"] == result.error
    assert "/private/path" not in repr(record)


def test_worker_finalization_can_win_submitter_attachment_race(tmp_path):
    store, _, capability, _, executor, session, run = setup(tmp_path, FinalizingJobs)

    pending = execute(executor, session, run, "code_interpreter", {
        "code": "print('fast')\n", "title": "finalization race"})
    record = capability.history.get_record(pending.data["analysis_id"])

    assert pending.status == "pending" and pending.data["status"] == "succeeded"
    assert record["status"] == "succeeded"
    assert record["job_id"] == pending.job_id
    assert record["summary"] == {"fast": True}
