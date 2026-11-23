import hashlib
import json
import time

import pytest

from adj_partner.analysis.history import AnalysisHistory
from adj_partner.contracts import Budget, SessionConfig, ToolResult
from adj_partner.execution.manager import JobManager
from adj_partner.runtime.execution_delivery import VerificationFailure, verify_execution_delivery
from adj_partner.storage import Store
from adj_partner.tools.catalog import canonical_json


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
    "additionalProperties": False,
}


def _fixture(tmp_path, *, value=3, tool_name="code_interpreter", run=None, secrets=None,
             job_id="job_fixture", runtime_action_shape=False):
    store = Store(tmp_path / "state", secrets=secrets or [])
    history = AnalysisHistory(store)
    JobManager(store)  # creates the durable job tables; the verifier itself is read-only.
    config = SessionConfig(allowed_roots=[str(tmp_path)], allowed_tools=[tool_name],
                           enabled_tools=[tool_name]).to_dict()
    session = store.create_session(config)["id"]
    task = store.create_task(session, "Analyze a fixture")["id"]
    run_id = run or store.create_run(session, task, Budget().to_dict())["id"]
    code_bytes = b"print('fixture')\n"
    code = store.put_artifact(code_bytes, "analysis.py", "text/x-python",
                              {"kind": "model_authored_analysis", "language": "python",
                               "sha256": hashlib.sha256(code_bytes).hexdigest()}, session, run_id)
    record_inputs = [] if runtime_action_shape else [{
                                       "name": "table.csv", "size": 8,
                                       "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
                                       "path": str(tmp_path / "table.csv"),
                                   }]
    record = history.create_record(session, run_id, language="python", title="fixture",
                                   code_artifact_id=code["id"], inputs=record_inputs,
                                   parameters={}, limits={"timeout_seconds": 120})
    analysis_id = record["id"]
    result_body = json.dumps({"summary": {"value": value}},
                             separators=(",", ":"), ensure_ascii=False).encode()
    result_artifact = store.put_artifact(
        result_body, "analysis_result.json", "application/json",
        {"analysis_id": analysis_id, "role": "result", "relative_path": "analysis_result.json",
         "sha256": hashlib.sha256(result_body).hexdigest()}, session, run_id)
    history.attach_job(analysis_id, job_id)
    history.finish(analysis_id, status="succeeded", summary={"value": value},
                   artifacts=[("result", result_artifact["id"])], sandbox_backend="fixture")
    tool_result = ToolResult(
        status="ok",
        data={"analysis_id": analysis_id, **({"job_id": job_id} if not runtime_action_shape else {}),
              "summary": {"value": value},
              "outputs": [{"role": "result", "artifact_id": result_artifact["id"],
                           "name": "analysis_result.json", "media_type": "application/json",
                           "size": len(result_body), "sha256": result_artifact["sha256"]}]},
        artifacts=[{"id": result_artifact["id"], "sha256": result_artifact["sha256"]}],
        job_id=job_id,
    )
    job_result = store.put_artifact(json.dumps(tool_result.to_dict(), separators=(",", ":")),
                                    "job-result.json", "application/json",
                                    {"job_id": job_id, "definition_hash": "definition-hash",
                                     "role": "job_result"}, session, run_id)
    job_arguments = {"analysis_id": analysis_id, "language": "python",
                     "code_artifact_id": code["id"], "inputs": record["inputs"],
                     "parameters": {}, "timeout_seconds": 120}
    job_fingerprint = hashlib.sha256(canonical_json({
        "tool_name": tool_name, "arguments": job_arguments,
        "definition_hash": "definition-hash",
        "handler_ref": "adj_partner.analysis.worker:analysis_job",
        "timeout_seconds": 30.0,
    }).encode()).hexdigest()
    now = time.time()
    with store.transaction(write=True) as conn:
        conn.execute("""INSERT INTO execution_jobs(
            id,session_id,run_id,tool_name,handler_ref,definition,definition_hash,
            arguments,idempotency_key,fingerprint,timeout_seconds,deadline,created_at,
            workspace,status,finished_at,result_artifact)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                job_id, session, run_id, tool_name,
                "adj_partner.analysis.worker:analysis_job",
                json.dumps({"name": tool_name}), "definition-hash",
                json.dumps(job_arguments), "call_fixture", job_fingerprint,
                30.0, now + 30.0, now, str(tmp_path / "workspace"), "succeeded", now,
                json.dumps({"id": job_result["id"], "sha256": job_result["sha256"]}),
            ))
    action_args = {"code": "print('fixture')\n", "title": "fixture", "timeout_seconds": 120}
    if not runtime_action_shape:
        action_args["inputs"] = [str(tmp_path / "table.csv")]
    store.record_action(run_id, "call_fixture", tool_name, action_args, 1)
    store.finish_action(run_id, "call_fixture", tool_result.to_dict())
    return {"store": store, "run": run_id, "session": session, "job": job_id,
            "analysis": analysis_id, "result_artifact": result_artifact["id"]}


def test_delivers_only_canonical_verified_summary_with_provenance(tmp_path):
    fixture = _fixture(tmp_path)

    delivery = verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)

    assert delivery["kind"] == "execution_bound_json"
    assert delivery["content"] == '{"value":3}'
    assert delivery["provenance"]["run_id"] == fixture["run"]
    assert delivery["provenance"]["job_id"] == fixture["job"]
    assert delivery["provenance"]["analysis_id"] == fixture["analysis"]
    assert delivery["provenance"]["call_id"] == "call_fixture"
    assert delivery["provenance"]["semantic_validation"] == "not_performed"
    assert "semantic correctness was not independently validated" in delivery["assurance"]
    assert "path" not in json.dumps(delivery)


def test_real_worker_action_shape_uses_envelope_job_id_and_expands_optional_defaults(tmp_path):
    fixture = _fixture(tmp_path, runtime_action_shape=True)

    delivery = verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)

    assert delivery["content"] == '{"value":3}'
    assert delivery["provenance"]["job_id"] == fixture["job"]


def test_failed_unknown_and_non_analysis_actions_are_not_deliverable(tmp_path):
    fixture = _fixture(tmp_path)
    store = fixture["store"]
    store.record_action(fixture["run"], "call_failed", "code_interpreter", {}, 1)
    store.finish_action(fixture["run"], "call_failed", {"status": "error"})
    store.record_action(fixture["run"], "call_unknown", "browser", {}, 1)
    store.finish_action(fixture["run"], "call_unknown", {"status": "unknown"})
    store.record_action(fixture["run"], "call_native", "analysis_history", {}, 1)
    store.finish_action(fixture["run"], "call_native", {"status": "ok", "data": {}})
    store.record_action(fixture["run"], "call_noresult", "bash", {}, 1)
    store.finish_action(fixture["run"], "call_noresult", {"status": "ok", "data": None})
    with store.transaction(write=True) as conn:
        conn.execute("DELETE FROM actions WHERE run_id=? AND call_id=?", (fixture["run"], "call_fixture"))

    assert verify_execution_delivery(store, fixture["run"], SUMMARY_SCHEMA) is None


def test_ambiguous_successful_analysis_actions_are_not_deliverable(tmp_path):
    fixture = _fixture(tmp_path)
    store = fixture["store"]
    store.record_action(fixture["run"], "call_second", "bash", {}, 1)
    store.finish_action(fixture["run"], "call_second", {
        "status": "ok", "data": {"analysis_id": fixture["analysis"], "job_id": fixture["job"]},
        "job_id": fixture["job"],
    })

    assert verify_execution_delivery(store, fixture["run"], SUMMARY_SCHEMA) is None


def test_tampered_durable_result_hash_raises_verification_failure(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("UPDATE artifacts SET sha256=? WHERE id=?", ("0" * 64, fixture["result_artifact"]))

    with pytest.raises(VerificationFailure, match="hash|integrity|provenance"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_action_job_from_another_run_raises_verification_failure(tmp_path):
    first = _fixture(tmp_path / "first")
    # Keep both jobs in one durable store so the verifier can observe the
    # authoritative job row and reject its run identity, rather than treating
    # a missing job as the test's primary condition.
    second = _fixture(tmp_path / "first", value=4, job_id="job_second")
    store = first["store"]
    # The action is in the current run, while its claimed durable job belongs to another run.
    store.record_action(first["run"], "call_wrong_run", "code_interpreter", {}, 1)
    store.finish_action(first["run"], "call_wrong_run", {
        "status": "ok", "data": {"analysis_id": first["analysis"], "job_id": second["job"]},
        "job_id": second["job"],
    })

    # The original valid action is still present, so this case must be explicitly isolated.
    with store.transaction(write=True) as conn:
        conn.execute("DELETE FROM actions WHERE run_id=? AND call_id=?", (first["run"], "call_fixture"))
    with pytest.raises(VerificationFailure, match="run|identity"):
        verify_execution_delivery(store, first["run"], SUMMARY_SCHEMA)


def test_redacted_summary_is_rejected(tmp_path):
    fixture = _fixture(tmp_path, value="supersecret", secrets=["supersecret"])

    with pytest.raises(VerificationFailure, match="summary|redact|protocol"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_summary_schema_mismatch_returns_no_delivery(tmp_path):
    fixture = _fixture(tmp_path)
    incompatible = {"type": "object", "required": ["missing"], "additionalProperties": False}

    assert verify_execution_delivery(fixture["store"], fixture["run"], incompatible) is None


def test_action_arguments_must_match_durable_analysis_request(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("UPDATE actions SET args=? WHERE run_id=? AND call_id=?",
                     (json.dumps({"code": "tampered", "inputs": [], "title": "fixture",
                                  "timeout_seconds": 120}), fixture["run"], "call_fixture"))

    with pytest.raises(VerificationFailure, match="action|request|identity"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_action_args_hash_and_job_request_fingerprint_are_verified(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("UPDATE actions SET args_hash=? WHERE run_id=? AND call_id=?",
                     ("0" * 64, fixture["run"], "call_fixture"))

    with pytest.raises(VerificationFailure, match="fingerprint|request"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


@pytest.mark.parametrize("field,value", [
    ("name", "stdout.txt"),
    ("media_type", "text/plain"),
])
def test_result_artifact_metadata_must_match_worker_protocol(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        conn.execute(f"UPDATE artifacts SET {field}=? WHERE id=?", (value, fixture["result_artifact"]))

    with pytest.raises(VerificationFailure, match="metadata|protocol|artifact"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_result_artifact_provenance_must_bind_analysis_and_role(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("UPDATE artifacts SET provenance=? WHERE id=?",
                     (json.dumps({"analysis_id": fixture["analysis"], "role": "stdout",
                                  "relative_path": "analysis_result.json",
                                  "sha256": "0" * 64}), fixture["result_artifact"]))

    with pytest.raises(VerificationFailure, match="provenance|artifact"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_job_result_artifact_must_be_current_run_scoped_and_protocol_named(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["store"].transaction(write=True) as conn:
        row = conn.execute("SELECT result_artifact FROM execution_jobs WHERE id=?",
                           (fixture["job"],)).fetchone()
        job_result_id = json.loads(row["result_artifact"])["id"]
        conn.execute("UPDATE artifacts SET name=? WHERE id=?", ("stdout.txt", job_result_id))

    with pytest.raises(VerificationFailure, match="Job result|metadata|artifact"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_bool_and_number_summary_mismatch_is_not_python_equal(tmp_path):
    fixture = _fixture(tmp_path)
    action = fixture["store"].get_action(fixture["run"], "call_fixture")
    with fixture["store"].transaction(write=True) as conn:
        result = action["result"]
        result["data"]["summary"] = {"value": True}
        conn.execute("UPDATE actions SET result=? WHERE run_id=? AND call_id=?",
                     (json.dumps(result), fixture["run"], "call_fixture"))

    with pytest.raises(VerificationFailure, match="summary|agree"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)


def test_deep_summary_fails_closed_as_verification_failure(tmp_path):
    fixture = _fixture(tmp_path)
    deep = "0"
    for _ in range(300):
        deep = "[" + deep + "]"
    body = json.dumps({"summary": {"value": json.loads(deep)}}).encode()
    with fixture["store"].transaction(write=True) as conn:
        digest = hashlib.sha256(body).hexdigest()
        conn.execute("UPDATE artifacts SET sha256=?,size=?,provenance=? WHERE id=?",
                     (digest, len(body), json.dumps({"analysis_id": fixture["analysis"],
                                                      "role": "result",
                                                      "relative_path": "analysis_result.json",
                                                      "sha256": digest}), fixture["result_artifact"]))
    fixture["store"]._put_blob(body)

    with pytest.raises(VerificationFailure, match="complex|depth|protocol|summary"):
        verify_execution_delivery(fixture["store"], fixture["run"], SUMMARY_SCHEMA)
