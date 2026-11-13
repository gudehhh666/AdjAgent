"""Read-only verification of a durable analysis result for direct delivery.

This module deliberately has no execution, model, budget, or acceptance side
effects.  It only follows already durable action, job, history, and artifact
records and returns a delivery when every identity and integrity check agrees.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePath

from ..analysis.history import AnalysisHistory
from ..contracts import ToolResult
from ..execution.manager import JobManager
from ..research.acceptance import (_response_value_within_limits, _strict_response_json,
    response_schema_issues)
from ..tools.catalog import canonical_json


TRUSTED_ANALYSIS_TOOLS = frozenset({"code_interpreter", "bash", "analysis_method_run"})
TRUSTED_ANALYSIS_HANDLER = "adj_partner.analysis.worker:analysis_job"
MAX_SOURCE_BYTES = 128 * 1024
MAX_SUMMARY_BYTES = 64 * 1024
MAX_JOB_RESULT_BYTES = 1024 * 1024


class VerificationFailure(ValueError):
    """Durable identity, integrity, or protocol evidence is inconsistent."""


def _fail(message: str, cause: Exception | None = None):
    if cause is None:
        raise VerificationFailure(message)
    raise VerificationFailure(message) from cause


def _read_only_manager(store):
    # JobManager.__init__ creates tables.  A verifier must not write, so attach
    # to the official read methods without invoking that constructor.
    manager = JobManager.__new__(JobManager)
    manager.store = store
    return manager


def _read_only_history(store):
    # AnalysisHistory.__init__ also creates tables; get_record itself is read-only.
    history = AnalysisHistory.__new__(AnalysisHistory)
    history.store = store
    return history


def _successful_candidates(store, run_id):
    candidates = []
    for action in store.list_actions(run_id):
        if action.get("tool_name") not in TRUSTED_ANALYSIS_TOOLS:
            continue
        result = action.get("result")
        if action.get("finished_at") is None or action.get("status") != "completed":
            continue
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        data = result.get("data")
        if not isinstance(data, dict):
            # A host may have durably recorded an action without a result
            # envelope (for example after an interrupted handoff).  That is
            # ineligible for direct delivery, rather than evidence to repair.
            continue
        analysis_id = data.get("analysis_id")
        # The durable ToolResult envelope owns job_id.  The analysis worker's
        # data payload intentionally carries analysis_id and output metadata,
        # but does not duplicate the job identity there.
        job_id = result.get("job_id")
        if (not isinstance(analysis_id, str) or not analysis_id
                or not isinstance(job_id, str) or not job_id):
            _fail("Successful trusted analysis action has incomplete identity")
        if data.get("job_id") not in {None, job_id}:
            _fail("Action and data job identities disagree")
        candidates.append((action, result, data))
    return candidates


def _bounded_canonical(value, label, *, max_bytes=MAX_SUMMARY_BYTES):
    """Canonical finite JSON with explicit traversal and byte limits."""
    try:
        if not _response_value_within_limits(value):
            _fail(f"{label} exceeds depth or node limits")
        encoded = canonical_json(value).encode("utf-8")
    except VerificationFailure:
        raise
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        _fail(f"{label} is not bounded finite JSON", exc)
    if len(encoded) > max_bytes:
        _fail(f"{label} exceeds its byte limit")
    return encoded


def _equal_json(left, right, label):
    if _bounded_canonical(left, label) != _bounded_canonical(right, label):
        _fail(f"{label} values disagree")


def _action_hash(store, run_id, call_id, action):
    # Store.claim_action uses its private durable JSON encoding (sorted keys,
    # UTF-8, and the default separators), so reproduce that exact fingerprint
    # without invoking a write-side Store helper.
    encoded = json.dumps({"tool_name": action.get("tool_name"), "args": action.get("args")},
                         ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    try:
        with store.transaction() as conn:
            row = conn.execute("SELECT args_hash FROM actions WHERE run_id=? AND call_id=?",
                               (run_id, call_id)).fetchone()
    except (KeyError, TypeError, ValueError) as exc:
        _fail("Action request fingerprint cannot be loaded", exc)
    if row is None or row["args_hash"] != expected:
        _fail("Action request fingerprint is invalid")


def _analysis_request(action, record, code_bytes):
    args = action.get("args")
    if not isinstance(args, dict):
        _fail("Analysis action arguments are malformed")
    try:
        limits = record["limits"]
        timeout = limits["timeout_seconds"]
    except (KeyError, TypeError) as exc:
        _fail("Analysis history limits are malformed", exc)
    inputs = record.get("inputs")
    if not isinstance(inputs, list):
        _fail("Analysis history input manifest is malformed")
    names = [item.get("path") for item in inputs if isinstance(item, dict)]
    if len(names) != len(inputs) or any(not isinstance(item, str) for item in names):
        _fail("Analysis history input paths are malformed")
    tool_name = action.get("tool_name")
    if tool_name == "code_interpreter":
        expected = {"code": code_bytes.decode("utf-8"), "inputs": names,
                    "title": record.get("title"), "timeout_seconds": timeout}
    elif tool_name == "bash":
        expected = {"script": code_bytes.decode("utf-8"), "inputs": names,
                    "title": record.get("title"), "timeout_seconds": timeout}
    elif tool_name == "analysis_method_run":
        expected = {"method_id": record.get("method_id"), "parameters": record.get("parameters"),
                    "inputs": names, "title": record.get("title"), "timeout_seconds": timeout}
    else:
        _fail("Analysis action tool is not trusted")
    # Tool handlers expand optional model arguments before creating the
    # durable analysis record/job.  The action journal retains the original
    # model request, so compare its canonical meaning after applying the
    # handler defaults instead of requiring omitted optional keys to be
    # present in the action row.
    normalized = dict(args)
    normalized.setdefault("inputs", [])
    normalized.setdefault("timeout_seconds", 120)
    _equal_json(normalized, expected, "Analysis action request")


def _job_request(job, record, analysis_id):
    arguments = job.get("arguments")
    expected = {
        "analysis_id": analysis_id,
        "language": record.get("language"),
        "code_artifact_id": record.get("code_artifact_id"),
        "inputs": record.get("inputs"),
        "parameters": record.get("parameters"),
        "timeout_seconds": record.get("limits", {}).get("timeout_seconds")
            if isinstance(record.get("limits"), dict) else None,
    }
    if record.get("method_id") is not None:
        expected["method_id"] = record["method_id"]
    _equal_json(arguments, expected, "Analysis job request")


def _job_fingerprint(job):
    try:
        expected = hashlib.sha256(canonical_json({
            "tool_name": job["tool_name"], "arguments": job["arguments"],
            "definition_hash": job["definition_hash"], "handler_ref": job["handler_ref"],
            "timeout_seconds": float(job["timeout_seconds"]),
        }).encode()).hexdigest()
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        _fail("Analysis job request fingerprint is malformed", exc)
    if job.get("fingerprint") != expected:
        _fail("Analysis job request fingerprint is invalid")


def _artifact_by_role(record, role):
    matches = [item for item in record.get("artifacts", []) if item.get("role") == role]
    if len(matches) != 1:
        _fail(f"Analysis history must contain exactly one {role} artifact")
    return matches[0]


def _check_artifact(store, artifact, label, *, session_id, run_id,
                    expected_name=None, expected_media_type=None,
                    expected_provenance=None, max_bytes=MAX_SUMMARY_BYTES):
    identifier, expected = artifact.get("artifact_id"), artifact.get("sha256")
    if not isinstance(identifier, str) or not isinstance(expected, str):
        _fail(f"{label} artifact identity is malformed")
    metadata = [item for item in store.artifacts(run_id=run_id) if item.get("id") == identifier]
    if (len(metadata) != 1 or metadata[0].get("session_id") != session_id
            or metadata[0].get("run_id") != run_id or metadata[0].get("sha256") != expected):
        _fail(f"{label} artifact scope or metadata is invalid")
    metadata = metadata[0]
    if (expected_name is not None and metadata.get("name") != expected_name
            or expected_media_type is not None and metadata.get("media_type") != expected_media_type):
        _fail(f"{label} artifact metadata is invalid")
    if expected_provenance is not None:
        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            _fail(f"{label} artifact provenance is malformed")
        for key, value in expected_provenance.items():
            if provenance.get(key) != value:
                _fail(f"{label} artifact provenance is invalid")
        if provenance.get("sha256") not in {None, expected}:
            _fail(f"{label} artifact provenance hash is invalid")
    try:
        content = store.read_artifact(identifier, max_bytes=max_bytes)
    except (KeyError, OSError, ValueError) as exc:
        _fail(f"{label} artifact integrity check failed", exc)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        _fail(f"{label} artifact hash mismatch")
    if metadata.get("size") != len(content):
        _fail(f"{label} artifact size mismatch")
    return content, metadata


def _safe_input_manifest(inputs):
    if not isinstance(inputs, list):
        _fail("Analysis input manifest is malformed")
    result = []
    for item in inputs:
        if not isinstance(item, dict):
            _fail("Analysis input manifest entry is malformed")
        name, size, sha256 = item.get("name"), item.get("size"), item.get("sha256")
        if (not isinstance(name, str) or not name or PurePath(name).name != name
                or "/" in name or "\\" in name or type(size) is not int
                or size < 0 or not isinstance(sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", sha256)):
            _fail("Analysis input manifest contains unsafe fields")
        result.append({"name": name, "size": size, "sha256": sha256})
    return result


def verify_execution_delivery(store, run_id, response_schema):
    """Return a directly deliverable verified summary, or ``None``.

    ``None`` means there is no unique eligible successful analysis (including
    failed, unknown, pending, schema-incompatible, and ambiguous outcomes).
    ``VerificationFailure`` means durable evidence claims success but its
    identity, hashes, redaction, or protocol is internally inconsistent.
    """
    if not isinstance(run_id, str) or not isinstance(response_schema, dict):
        _fail("Execution delivery request is malformed")
    try:
        run = store.get_run(run_id)
        session_id = run["session_id"]
    except (KeyError, TypeError) as exc:
        _fail("Current run identity cannot be loaded", exc)

    candidates = _successful_candidates(store, run_id)
    if len(candidates) != 1:
        return None
    action, action_result, action_data = candidates[0]
    call_id = action.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        _fail("Analysis action call identity is malformed")
    job_id = action_result["job_id"]
    analysis_id = action_data["analysis_id"]
    jobs = _read_only_manager(store)
    try:
        raw_job = jobs._read(job_id, session_id)
    except (KeyError, ValueError, TypeError) as exc:
        _fail("Analysis job identity cannot be resolved", exc)
    if (raw_job.get("run_id") != run_id or raw_job.get("tool_name") != action.get("tool_name")
            or raw_job.get("handler_ref") != TRUSTED_ANALYSIS_HANDLER
            or raw_job.get("status") != "succeeded"
            or raw_job.get("idempotency_key") != call_id):
        _fail("Analysis job identity or trusted handler protocol is invalid")
    _job_fingerprint(raw_job)
    job = jobs.get(job_id, session_id)
    try:
        job_result = jobs.result(job_id, session_id)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail("Analysis job result integrity check failed", exc)
    if not isinstance(job_result, ToolResult) or job_result.status != "ok":
        return None
    if job_result.job_id not in {None, job_id}:
        _fail("Job result and job identities disagree")
    if not isinstance(job_result.data, dict):
        _fail("Job result data envelope is malformed")
    if job_result.data.get("analysis_id") != analysis_id:
        _fail("Job result names a different analysis record")

    history = _read_only_history(store)
    try:
        record = history.get_record(analysis_id)
    except (KeyError, TypeError) as exc:
        _fail("Analysis history identity cannot be resolved", exc)
    if (record.get("session_id") != session_id or record.get("run_id") != run_id
            or record.get("job_id") != job_id or record.get("status") != "succeeded"):
        _fail("Analysis history identity or status is invalid")

    code = _artifact_by_role(record, "code")
    result_artifact = _artifact_by_role(record, "result")
    source_name = "analysis.py" if record.get("language") == "python" else "analysis.sh"
    source_media = "text/x-python" if record.get("language") == "python" else "text/x-shellscript"
    code_bytes, _code_metadata = _check_artifact(
        store, code, "Analysis source", session_id=session_id, run_id=run_id,
        expected_name=source_name, expected_media_type=source_media,
        expected_provenance={"kind": "model_authored_analysis", "language": record.get("language"),
                             "sha256": code.get("sha256")},
        max_bytes=MAX_SOURCE_BYTES)
    _action_hash(store, run_id, call_id, action)
    _analysis_request(action, record, code_bytes)
    _job_request(job, record, analysis_id)
    result_bytes, result_metadata = _check_artifact(
        store, result_artifact, "Analysis result", session_id=session_id, run_id=run_id,
        expected_name="analysis_result.json", expected_media_type="application/json",
        expected_provenance={"analysis_id": analysis_id, "role": "result",
                             "relative_path": "analysis_result.json",
                             "sha256": result_artifact.get("sha256")},
        max_bytes=MAX_SUMMARY_BYTES)
    job_result_ref = raw_job.get("result_artifact")
    if not isinstance(job_result_ref, dict):
        _fail("Job result artifact reference is malformed")
    _check_artifact(
        store, {"artifact_id": job_result_ref.get("id"), "sha256": job_result_ref.get("sha256")},
        "Job result", session_id=session_id, run_id=run_id,
        expected_name="job-result.json", expected_media_type="application/json",
        expected_provenance={"job_id": job_id, "definition_hash": raw_job.get("definition_hash"),
                             "role": "job_result"}, max_bytes=MAX_JOB_RESULT_BYTES)
    try:
        result_document = _strict_response_json(result_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        _fail("Analysis result artifact is not strict JSON", exc)
    if (not isinstance(result_document, dict) or set(result_document) - {"summary", "metrics", "limitations"}
            or not isinstance(result_document.get("summary"), dict)):
        _fail("Analysis result data envelope is malformed")
    summary = result_document["summary"]
    _bounded_canonical(result_document, "Analysis result", max_bytes=MAX_SUMMARY_BYTES)
    redacted_summary = store.redact(summary)
    _equal_json(summary, redacted_summary, "Analysis summary redaction")
    try:
        summary_json = _bounded_canonical(summary, "Analysis summary").decode("utf-8")
        schema_json = _bounded_canonical(response_schema, "Response schema").decode("utf-8")
    except VerificationFailure:
        raise
    if response_schema_issues(summary_json, response_schema):
        return None
    if not isinstance(action_data.get("summary"), dict):
        _fail("Action summary is missing from the successful result")
    if not isinstance(job_result.data.get("summary"), dict):
        _fail("Job summary is missing from the successful result")
    if not isinstance(record.get("summary"), dict):
        _fail("Analysis history summary is missing from the successful result")
    _equal_json(action_data["summary"], summary, "Action summary")
    _equal_json(job_result.data["summary"], summary, "Job summary")
    _equal_json(record["summary"], summary, "Analysis history summary")

    outputs = job_result.data.get("outputs")
    result_outputs = [item for item in outputs or []
                      if isinstance(item, dict) and item.get("role") == "result"]
    if len(result_outputs) != 1:
        _fail("Job result output descriptors are malformed")
    output = result_outputs[0]
    if (output.get("artifact_id") != result_artifact.get("artifact_id")
            or output.get("sha256") != result_artifact.get("sha256")
            or output.get("name") != result_metadata.get("name")
            or output.get("media_type") != result_metadata.get("media_type")
            or output.get("size") != result_metadata.get("size")):
        _fail("Job result and history artifact identities disagree")
    claimed = [item for item in job_result.artifacts if isinstance(item, dict)
               and item.get("id") == result_artifact.get("artifact_id")]
    if len(claimed) != 1 or claimed[0].get("sha256") != result_artifact.get("sha256"):
        _fail("Job result artifact claims are malformed")

    inputs = _safe_input_manifest(record.get("inputs"))
    provenance = {
        "call_id": call_id,
        "run_id": run_id,
        "session_id": session_id,
        "analysis_id": analysis_id,
        "job_id": job_id,
        "tool_name": action.get("tool_name"),
        "handler_ref": TRUSTED_ANALYSIS_HANDLER,
        "analysis_source_artifact_id": code["artifact_id"],
        "analysis_source_sha256": code["sha256"],
        "result_artifact_id": result_artifact["artifact_id"],
        "result_artifact_sha256": result_artifact["sha256"],
        "job_result_sha256": job["result_artifact"]["sha256"],
        "input_manifest": inputs,
        "sandbox_backend": record.get("sandbox_backend"),
        "summary_sha256": hashlib.sha256(summary_json.encode("utf-8")).hexdigest(),
        "response_schema_sha256": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
        "semantic_validation": "not_performed",
        "verification": [
            "job_terminal_succeeded",
            "analysis_record_terminal_succeeded",
            "result_artifact_hash_verified",
            "summary_matches_action_result",
            "summary_matches_public_schema",
        ],
    }
    if store.redact(provenance) != provenance:
        _fail("Execution delivery provenance contains redacted data")
    return {"kind": "execution_bound_json", "content": summary_json,
            "assurance": "Host analysis completed and its summary matched the public JSON Schema; "
                         "computation semantic correctness was not independently validated.",
            "provenance": provenance}
