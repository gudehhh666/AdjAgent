"""Real local subprocess diagnostics for durable jobs; no network/model/service."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import sqlite3

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolResult
from adj_partner.storage import Store
from adj_partner.tools import ToolSpec


PROJECT = Path(__file__).resolve().parents[1]
HANDLER = "tests.fixtures.job_diagnostics:exercise"
TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "uncertain"}


def manager_type():
    assert importlib.util.find_spec("adj_partner.execution"), "durable local execution package must exist"
    from adj_partner.execution import JobManager
    return JobManager


def spec(version="1", output_schema=None, name="diagnostic"):
    return ToolSpec(name, "Explicit diagnostic fixture", {
        "type": "object", "properties": {"mode": {"type": "string"}, "text": {"type": "string"},
            "duration": {"type": "number", "minimum": 0, "maximum": 10}},
        "additionalProperties": False,
    }, output_schema=output_schema, version=version)


def scope(tmp_path, store=None):
    store = store or Store(tmp_path / "state with spaces")
    config = SessionConfig(allowed_tools=["diagnostic"], enabled_tools=["diagnostic"],
                           allowed_roots=[str(tmp_path)])
    session = store.create_session(config.to_dict())
    task = store.create_task(session["id"], "Durable diagnostic")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    return store, session["id"], run["id"]


def setup(tmp_path, **spec_kwargs):
    store, session, run = scope(tmp_path)
    manager = manager_type()(store)
    manager.register(spec(**spec_kwargs), HANDLER)
    return manager, session, run


def wait_for(manager, job, predicate=lambda value: value["status"] in TERMINAL, timeout=10):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        value = manager.reconcile(job)
        if predicate(value):
            return value
        time.sleep(0.04)
    pytest.fail(f"bounded wait expired: {manager.get(job)}")


def wait_child(manager, job):
    return wait_for(manager, job, lambda value: (Path(value["workspace"]) / "child.pid").exists())


def assert_child_stopped(job):
    pid_path = Path(job["workspace"]) / "child.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_actual_subprocess_completes_and_persists_hashed_result(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"text": "space safe"}, "call-1")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "succeeded"
    assert finished["arguments"] == {"text": "space safe"}
    assert finished["handler_ref"] == HANDLER
    assert finished["definition_hash"] == spec().definition_hash
    assert finished["deadline"] > finished["created_at"]
    result = manager.result(job["id"], session)
    assert result.status == "ok" and result.data == {"text": "space safe"}
    assert manager.store.read_artifact(result.artifacts[0]["id"]) == b"exact diagnostic bytes"
    envelope = json.loads(manager.store.read_artifact(finished["result_artifact"]["id"]))
    assert envelope == result.to_dict()
    assert "diagnostic stdout" in finished["stdout"] and "diagnostic stderr" in finished["stderr"]
    assert any(item["kind"] == "job.progress" for item in manager.store.events(session))
    snapshot = manager.result(job["id"]).to_dict()
    manager.cancel(job["id"])
    assert manager.result(job["id"]).to_dict() == snapshot
    assert_child_stopped(finished)


def test_duplicate_reuses_and_changed_identity_conflicts_after_restart(tmp_path):
    manager, session, run = setup(tmp_path)
    first = manager.submit(session, run, "diagnostic", {"duration": 0.2}, "same")
    reopened = manager_type()(Store(manager.store.root))
    same = reopened.submit(session, run, "diagnostic", {"duration": 0.2}, "same")
    assert same["id"] == first["id"]
    with pytest.raises(ValueError, match="[Ii]dempotency"):
        reopened.submit(session, run, "diagnostic", {"duration": 0.3}, "same")
    reopened.register(spec(version="2"), HANDLER)
    with pytest.raises(ValueError, match="[Ii]dempotency"):
        reopened.submit(session, run, "diagnostic", {"duration": 0.2}, "same")
    finished = wait_for(reopened, first["id"])
    assert finished["status"] == "succeeded"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_parallel_submitters_and_worker_claims_execute_once(tmp_path):
    manager, session, run = setup(tmp_path)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = list(pool.map(lambda _: manager_type()(Store(manager.store.root)).submit(
            session, run, "diagnostic", {"duration": 0.3}, "parallel"), range(5)))
    assert len({job["id"] for job in jobs}) == 1
    finished = wait_for(manager, jobs[0]["id"])
    assert finished["status"] == "succeeded"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_worker_outlives_submitter_and_new_manager_reattaches(tmp_path):
    manager, session, run = setup(tmp_path)
    program = """import json,sys
from adj_partner.execution import JobManager
from adj_partner.storage import Store
m=JobManager(Store(sys.argv[1]))
j=m.submit(sys.argv[2],sys.argv[3],'diagnostic',{'duration':1.0},'orphan-parent')
print(j['id'],flush=True)
"""
    parent = subprocess.run([sys.executable, "-c", program, str(manager.store.root), session, run],
                            cwd=PROJECT, capture_output=True, text=True, timeout=5)
    assert parent.returncode == 0, parent.stderr
    job_id = parent.stdout.strip()
    reopened = manager_type()(Store(manager.store.root))
    assert reopened.result(job_id) is None
    finished = wait_for(reopened, job_id)
    assert finished["status"] == "succeeded"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


@pytest.mark.parametrize("mode,code", [("malformed", "output_contract"), ("bad_envelope", "output_contract"),
    ("nan", "output_json"), ("surrogate", "output_json"), ("pending", "output_contract"),
    ("bad_schema", "output_schema"), ("fake_artifact", "output_artifact"), ("exception", "handler_error")])
def test_malformed_terminal_results_fail_explicitly(tmp_path, mode, code):
    manager, session, run = setup(tmp_path, output_schema={"type": "object"})
    job = manager.submit(session, run, "diagnostic", {"mode": mode}, mode)
    assert wait_for(manager, job["id"])["status"] == "failed"
    result = manager.result(job["id"])
    assert result.status == "error" and result.error["code"] == code
    assert "sk-diagnosticsecret999" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("operation,expected", [("cancel", "cancelled"), ("timeout", "timed_out")])
def test_supervisor_stops_noncooperating_child_before_acknowledgement(tmp_path, operation, expected):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": "cpu", "duration": 8}, operation,
                         timeout_seconds=2.5 if operation == "timeout" else 6)
    wait_child(manager, job["id"])
    if operation == "cancel":
        requested = manager.cancel(job["id"], session)
        assert requested["status"] == "cancel_requested"
        assert manager.result(job["id"]) is None
    finished = wait_for(manager, job["id"])
    assert finished["status"] == expected
    assert_child_stopped(finished)
    assert manager.result(job["id"]).status == ("cancelled" if operation == "cancel" else "error")


def test_run_cancellation_reaches_supervisor_without_manager_cancel(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": "cpu", "duration": 8}, "run-cancel")
    wait_child(manager, job["id"])
    manager.store.update_run(run, {"status": "cancelled"})
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "cancelled"
    assert_child_stopped(finished)


@pytest.mark.parametrize("mode", ["crash", "supervisor_crash"])
def test_crashed_claimed_work_becomes_uncertain_and_never_replays(tmp_path, mode):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": mode}, mode)
    finished = wait_for(manager, job["id"], timeout=12)
    assert finished["status"] == "uncertain"
    reopened = manager_type()(Store(manager.store.root))
    assert reopened.submit(session, run, "diagnostic", {"mode": mode}, mode)["id"] == job["id"]
    assert reopened.reconcile(job["id"])["status"] == "uncertain"
    assert reopened.result(job["id"]).status == "uncertain"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_worker_environment_and_private_path_boundary(tmp_path, monkeypatch):
    manager, session, run = setup(tmp_path)
    for name in ("OPENAI_API_KEY", "AZURE_TOKEN", "PRIVATE_TEST_SENTINEL", "PYTHONPATH"):
        monkeypatch.setenv(name, "parent-only-secret")
    job = manager.submit(session, run, "diagnostic", {"mode": "env"}, "env")
    assert wait_for(manager, job["id"])["status"] == "succeeded"
    data = manager.result(job["id"]).data
    assert not any(name in data["env"] for name in ("HOME", "OPENAI_API_KEY", "AZURE_TOKEN", "PRIVATE_TEST_SENTINEL", "PYTHONPATH"))
    assert Path(data["cwd"]) == Path(job["workspace"])
    paths = manager.submit(session, run, "diagnostic", {"mode": "paths"}, "paths")
    assert wait_for(manager, paths["id"])["status"] == "succeeded"
    checked = manager.result(paths["id"]).data
    assert len(checked["denied"]) == 2
    assert Path(checked["own"]).parent == Path(paths["workspace"])


def test_scoped_reads_and_foreign_or_terminal_runs_rejected(tmp_path):
    manager, session, run = setup(tmp_path)
    _, other, other_run = scope(tmp_path, manager.store)
    with pytest.raises(ValueError, match="belong"):
        manager.submit(other, run, "diagnostic", {}, "foreign")
    job = manager.submit(session, run, "diagnostic", {}, "scoped")
    for method in (manager.get, manager.result, manager.cancel, manager.reconcile):
        with pytest.raises(KeyError):
            method(job["id"], session_id=other)
    assert manager.list(other) == []
    assert manager.list(session)[0]["id"] == job["id"]
    wait_for(manager, job["id"])
    manager.store.update_run(other_run, {"status": "succeeded"})
    with pytest.raises(PermissionError):
        manager.submit(other, other_run, "diagnostic", {}, "terminal")


@pytest.mark.parametrize("arguments", [{"text": "sk-testsecret123456"}, {"text": "opaque-value"},
    {"text": float("nan")}, {"text": "\ud800"}, {"text": 123}, {"unknown": "value"}])
def test_invalid_or_secret_arguments_rejected_without_submission(tmp_path, arguments):
    manager, session, run = setup(tmp_path)
    manager.store.register_secrets(["opaque-value"])
    with pytest.raises((TypeError, ValueError)):
        manager.submit(session, run, "diagnostic", arguments, "invalid")
    assert manager.list() == []


def test_bounded_redacted_logs_and_progress(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": "log"}, "logs")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "succeeded"
    assert len(finished["stdout"].encode()) <= 32768 and len(finished["stderr"].encode()) <= 32768
    assert finished["stdout_truncated"] and finished["stderr_truncated"]
    serialized = json.dumps(manager.store.events(session)) + json.dumps(finished)
    assert "sk-diagnosticsecret999" not in serialized and "notpublic" not in serialized
    progress = [event for event in manager.store.events(session) if event["kind"] == "job.progress"]
    assert 1 <= len(progress) <= 100
    assert max(len(json.dumps(event["payload"])) for event in progress) <= 10000


def test_host_registration_rejects_non_reference_values(tmp_path):
    manager, _, _ = setup(tmp_path)
    for reference in (lambda: None, "os.system('touch /tmp/no')", "/tmp/handler.py:f", "os:system:extra"):
        with pytest.raises((TypeError, ValueError)):
            manager.register(spec(), reference)


def test_later_host_secret_registration_redacts_public_reads_not_artifact_bytes(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"text": "later-sensitive-value"}, "late-secret")
    finished = wait_for(manager, job["id"])
    original = manager.store.read_artifact(finished["result_artifact"]["id"])
    assert b"later-sensitive-value" in original
    manager.store.register_secrets(["later-sensitive-value"])
    for value in (manager.get(job["id"]), manager.list(), manager.cancel(job["id"]),
                  manager.result(job["id"]).to_dict()):
        assert "later-sensitive-value" not in json.dumps(value)
    assert manager.store.read_artifact(finished["result_artifact"]["id"]) == original


def test_idempotency_conflict_takes_precedence_over_changed_kind_or_schema(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"text": "original"}, "identity")
    for name, arguments in (("unregistered", {"text": "original"}), ("diagnostic", {"text": 123})):
        with pytest.raises(ValueError, match="[Ii]dempotency"):
            manager.submit(session, run, name, arguments, "identity")
    wait_for(manager, job["id"])


def test_launch_failure_keeps_durable_unclaimed_intent_and_relaunches(tmp_path, monkeypatch):
    manager, session, run = setup(tmp_path)
    def unavailable(*args, **kwargs):
        # A second connection obtains the writer lock during launch: persistence
        # already committed and launch cannot be under the submission transaction.
        with Store(manager.store.root).transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM execution_jobs").fetchone()
            assert row["status"] == "queued" and row["owner_id"] is None
        raise OSError("diagnostic launch unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", unavailable)
        job = manager.submit(session, run, "diagnostic", {}, "launch-retry")
    assert job["status"] == "queued" and manager.result(job["id"]) is None
    reopened = manager_type()(Store(manager.store.root))
    finished = wait_for(reopened, job["id"])
    assert finished["status"] == "succeeded"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_independent_competing_supervisors_have_exactly_one_claim(tmp_path, monkeypatch):
    from adj_partner.execution.manager import worker_command
    manager, session, run = setup(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("paused launch")))
        job = manager.submit(session, run, "diagnostic", {"duration": 0.3}, "claim-race")
    workers = [subprocess.Popen(worker_command("supervise", manager.store.root, job["id"]),
               cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for _ in range(4)]
    try:
        for process in workers:
            assert process.wait(timeout=10) == 0
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
    finished = manager.get(job["id"])
    assert finished["status"] == "succeeded"
    with manager.store.transaction() as conn:
        assert conn.execute("SELECT count(*) FROM execution_events WHERE job_id=? AND kind='job.claimed'",
                            (job["id"],)).fetchone()[0] == 1
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_queued_cancel_acknowledges_without_ever_invoking_handler(tmp_path, monkeypatch):
    manager, session, run = setup(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("paused launch")))
        job = manager.submit(session, run, "diagnostic", {}, "cancel-queued")
        assert manager.cancel(job["id"])["status"] == "cancel_requested"
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "cancelled"
    assert not (Path(finished["workspace"]) / "invocations.txt").exists()


def test_deadline_before_claim_never_invokes_handler(tmp_path, monkeypatch):
    manager, session, run = setup(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("paused launch")))
        job = manager.submit(session, run, "diagnostic", {}, "expire-queued", timeout_seconds=0.1)
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "timed_out"
    assert not (Path(finished["workspace"]) / "invocations.txt").exists()


def test_terminal_result_and_intent_are_database_immutable(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {}, "immutable")
    with pytest.raises(sqlite3.IntegrityError, match="intent"):
        with manager.store.transaction(write=True) as conn:
            conn.execute("UPDATE execution_jobs SET arguments='{}' WHERE id=?", (job["id"],))
    wait_for(manager, job["id"])
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with manager.store.transaction(write=True) as conn:
            conn.execute("UPDATE execution_jobs SET status='queued' WHERE id=?", (job["id"],))


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan")])
def test_timeout_must_be_finite_and_positive(tmp_path, timeout):
    manager, session, run = setup(tmp_path)
    with pytest.raises(ValueError):
        manager.submit(session, run, "diagnostic", {}, "bad-timeout", timeout_seconds=timeout)
    assert manager.list() == []


@pytest.mark.parametrize("cancel,expected", [(False, "timed_out"), (True, "cancelled")])
def test_permanently_unavailable_launch_does_not_strand_expired_or_cancelled_intent(tmp_path, monkeypatch, cancel, expected):
    manager, session, run = setup(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("permanent launch failure")))
    job = manager.submit(session, run, "diagnostic", {}, "never-launch", timeout_seconds=0.2 if not cancel else 300)
    if cancel:
        assert manager.cancel(job["id"])["status"] == "cancel_requested"
    finished = wait_for(manager, job["id"], timeout=2)
    assert finished["status"] == expected
    assert not (Path(finished["workspace"]) / "invocations.txt").exists()


def test_handler_can_read_approved_source_but_cannot_write_it(tmp_path):
    manager, session, run = setup(tmp_path)
    source = tmp_path / "approved input with spaces.txt"
    source.write_text("unchanged input")
    job = manager.submit(session, run, "diagnostic", {"mode": "read_path", "text": str(source)}, "data-path")
    assert wait_for(manager, job["id"])["status"] == "succeeded"
    assert manager.result(job["id"]).data == {"text": "unchanged input", "write_denied": True}
    assert source.read_text() == "unchanged input"


def test_result_cannot_smuggle_an_actual_foreign_session_artifact(tmp_path):
    manager, session, run = setup(tmp_path)
    _, other, other_run = scope(tmp_path, manager.store)
    artifact = manager.store.put_artifact(b"private foreign data", "private.txt", session_id=other, run_id=other_run)
    job = manager.submit(session, run, "diagnostic", {"mode": "foreign_artifact", "text": artifact["id"]}, "foreign-artifact")
    assert wait_for(manager, job["id"])["status"] == "failed"
    assert manager.result(job["id"]).error["code"] == "output_artifact"


def test_result_artifact_corruption_is_detected_without_replay(tmp_path):
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {}, "corruption")
    finished = wait_for(manager, job["id"])
    Path(finished["result_artifact"]["path"]).write_bytes(b"tampered private test artifact")
    with pytest.raises(ValueError, match="hash"):
        manager.result(job["id"])
    assert manager.reconcile(job["id"])["status"] == "succeeded"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_canonical_argument_order_is_same_identity_but_new_call_id_is_new_work(tmp_path):
    manager, session, run = setup(tmp_path)
    first = manager.submit(session, run, "diagnostic", {"text": "x", "duration": 0}, "call-one")
    same = manager.submit(session, run, "diagnostic", {"duration": 0, "text": "x"}, "call-one")
    later = manager.submit(session, run, "diagnostic", {"text": "x", "duration": 0}, "call-two")
    assert first["id"] == same["id"] and later["id"] != first["id"]
    assert wait_for(manager, first["id"])["status"] == "succeeded"
    assert wait_for(manager, later["id"])["status"] == "succeeded"


def test_atomic_job_events_are_scoped_and_redacted_on_read(tmp_path):
    manager, session, run = setup(tmp_path)
    _, other, _ = scope(tmp_path, manager.store)
    job = manager.submit(session, run, "diagnostic", {}, "journal")
    wait_for(manager, job["id"])
    assert hasattr(manager, "events"), "job journal needs a public scoped reader"
    events = manager.events(job["id"], session_id=session)
    assert {"job.queued", "job.claimed", "job.progress", "job.succeeded"} <= {item["kind"] for item in events}
    assert [item["seq"] for item in events] == sorted(item["seq"] for item in events)
    with pytest.raises(KeyError):
        manager.events(job["id"], session_id=other)
    manager.store.register_secrets(["diagnostic progress"])
    assert "diagnostic progress" not in json.dumps(manager.events(job["id"], session_id=session))


@pytest.mark.parametrize("expansion", ["artifact_metadata", "redaction"])
def test_final_normalized_result_enforces_cap_after_expansion(tmp_path, expansion):
    from adj_partner.execution.manager import MAX_RESULT_BYTES, validate_result
    from adj_partner.tools.catalog import canonical_json
    store, session, run = scope(tmp_path)
    job = {"id": "job_diagnostic_probe", "session_id": session, "run_id": run,
           "definition": {"output_schema": None}}
    if expansion == "artifact_metadata":
        artifact = store.put_artifact(b"intact", "expanded-metadata.txt",
            provenance={"diagnostic_note": "x" * MAX_RESULT_BYTES}, session_id=session, run_id=run)
        incoming = ToolResult(data="small", artifacts=[{"id": artifact["id"]}])
    else:
        store.register_secrets(["qz"])
        incoming = ToolResult(data="qz" * (MAX_RESULT_BYTES // 3))
    assert len(canonical_json(incoming.to_dict()).encode()) < MAX_RESULT_BYTES
    normalized = validate_result(store, job, incoming)
    assert normalized.status == "error" and normalized.error["code"] == "output_json"
    assert len(canonical_json(normalized.to_dict()).encode()) <= MAX_RESULT_BYTES
    if expansion == "artifact_metadata":
        assert store.read_artifact(artifact["id"]) == b"intact"


@pytest.mark.parametrize("mode", ["metadata_expansion", "redaction_expansion"])
def test_expanded_result_fails_explicitly_without_uncertainty_or_replay(tmp_path, mode):
    from adj_partner.execution.manager import MAX_RESULT_BYTES
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": mode}, "expanded-result")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "failed"
    result = manager.result(job["id"])
    assert result.status == "error" and result.error["code"] == "output_json"
    assert len(manager.store.read_artifact(finished["result_artifact"]["id"])) <= MAX_RESULT_BYTES
    if mode == "metadata_expansion":
        artifacts = [item for item in manager.store.artifacts(session, run) if item["name"] == "metadata-diagnostic.txt"]
        assert len(artifacts) == 1
        assert manager.store.read_artifact(artifacts[0]["id"]) == b"genuine diagnostic bytes"
    reopened = manager_type()(Store(manager.store.root))
    assert reopened.submit(session, run, "diagnostic", {"mode": mode}, "expanded-result")["id"] == job["id"]
    assert reopened.reconcile(job["id"])["status"] == "failed"
    assert len((Path(finished["workspace"]) / "invocations.txt").read_text().splitlines()) == 1


def test_truncated_quoted_password_assignment_is_scrubbed_without_losing_ordinary_prefix(tmp_path):
    from adj_partner.execution.manager import MAX_LOG_BYTES
    manager, session, run = setup(tmp_path)
    synthetic_prefix = "QUALITY_SYNTHETIC_PASSWORD_VALUE_"
    complete_assignment = 'password="' + synthetic_prefix + "x" * 40000 + '"'
    assert synthetic_prefix not in manager.store.redact(complete_assignment)
    job = manager.submit(session, run, "diagnostic", {"mode": "truncated_assignment"}, "truncated-secret")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "succeeded"
    for stream in ("stdout", "stderr"):
        assert finished[stream + "_truncated"]
        assert f"ordinary {stream} prefix" in finished[stream]
        assert len(finished[stream].encode()) <= MAX_LOG_BYTES
        assert synthetic_prefix not in finished[stream]
    assert synthetic_prefix not in json.dumps(manager.events(job["id"], session))
    assert synthetic_prefix not in json.dumps(manager.store.events(session))


@pytest.mark.parametrize("variant", ["escaped", "nested", "nested_key", "token", "unicode_key"])
def test_truncated_encoded_credentials_never_reach_public_records(tmp_path, variant):
    from adj_partner.execution.manager import MAX_LOG_BYTES
    from tests.fixtures.job_diagnostics import encoded_credential_payload
    manager, session, run = setup(tmp_path)
    marker = "QUALITY_SYNTHETIC_ESCAPED_VALUE_"
    payload = encoded_credential_payload(variant, marker + "x" * 40000)
    assert marker not in manager.store.redact(payload)
    job = manager.submit(session, run, "diagnostic", {"mode": "escaped_assignment", "text": variant}, "escaped-log")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "succeeded"
    for stream in ("stdout", "stderr"):
        assert finished[stream + "_truncated"]
        assert f"ordinary {stream} prefix" in finished[stream]
        assert '"note": "completed ordinary record"' in finished[stream]
        assert len(finished[stream].encode()) <= MAX_LOG_BYTES
        assert marker not in finished[stream]
    assert marker not in json.dumps(manager.events(job["id"], session))
    assert marker not in json.dumps(manager.store.events(session))


def test_truncation_preserves_bounded_ordinary_log_content(tmp_path):
    from adj_partner.execution.manager import MAX_LOG_BYTES
    manager, session, run = setup(tmp_path)
    job = manager.submit(session, run, "diagnostic", {"mode": "ordinary_truncated"}, "ordinary-log")
    finished = wait_for(manager, job["id"])
    assert finished["status"] == "succeeded"
    for stream in ("stdout", "stderr"):
        assert finished[stream + "_truncated"]
        assert len(finished[stream].encode()) == MAX_LOG_BYTES
        assert f"ordinary {stream} prefix\nordinary data ordinary data" in finished[stream]
