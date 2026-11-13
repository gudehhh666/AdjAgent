"""Read-only final-source verification of fixed public-check jobs.

This verifies durable provenance and the outcomes of the available public
assertions. It is neither an official benchmark grader nor an adversarial proof
that an arbitrary candidate cannot interfere with its child-side serialization.
"""

import hashlib
import json

from ..analysis.public_code_checker import (
    MAX_MANIFEST_BYTES, MAX_SUBMISSION_BYTES, canonical_submission,
    checker_source_digest, submission_sha256, validate_manifest,
)
from ..research.acceptance import task_binding
from ..storage import TERMINAL_STATUSES
from ..tools.catalog import canonical_json


KEY = "public_code_validation"
BINDING_KEY = "public_code_validation_binding"
HANDLER = "adj_partner.analysis.public_code_capabilities:public_code_check_job"
TOOL = "public_code_check"


class PublicValidationFailure(ValueError):
    pass


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def bind_public_code_validation(store, run_id):
    from ..analysis.public_code_capabilities import handler_source_digest, load_manifest
    run = store.get_run(run_id)
    task = store.get_task(run["task_id"])
    contract = task["acceptance"][KEY]
    binding = {key: value for key, value in contract.items() if key != "enforce"}
    binding.update(task_binding=task_binding(task), checker_source_sha256=checker_source_digest(),
                   handler_source_sha256=handler_source_digest())
    load_manifest(store, run["session_id"], binding)
    return binding


def _binding(store, run):
    from ..analysis.public_code_capabilities import handler_source_digest, load_manifest
    binding = run["state"].get(BINDING_KEY)
    task = store.get_task(run["task_id"])
    expected = {key: value for key, value in task["acceptance"][KEY].items() if key != "enforce"}
    current_task_binding = task_binding(task)
    original = binding.get("task_binding", {}) if isinstance(binding, dict) else {}
    # RunJournal atomically increments task revision once when terminalizing a
    # run. Allow only that exact status-only increment for post-run inspection;
    # a live gate, edited requirements, or any further revision still fails.
    if (run["status"] in TERMINAL_STATUSES and task["status"] == run["status"]
            and current_task_binding.get("id") == original.get("id")
            and current_task_binding.get("requirements_hash") == original.get("requirements_hash")
            and current_task_binding["observed_revision"] == original.get("observed_revision", -2) + 1):
        current_task_binding = original
    expected.update(task_binding=current_task_binding, checker_source_sha256=checker_source_digest(),
                    handler_source_sha256=handler_source_digest())
    if binding != expected:
        raise PublicValidationFailure("Public validation binding or checker source changed")
    manifest = load_manifest(store, run["session_id"], binding)
    return binding, manifest


def _artifact(store, sid, rid, identifier, *, maximum, kind=None):
    artifact = next((item for item in store.artifacts(sid, rid) if item["id"] == identifier), None)
    if artifact is None or not 0 < artifact["size"] <= maximum:
        raise PublicValidationFailure("Validation artifact scope or size mismatch")
    if kind and artifact["provenance"].get("kind") != kind:
        raise PublicValidationFailure("Validation artifact role mismatch")
    content = store.read_artifact(identifier, max_bytes=maximum)
    if _sha(content) != artifact["sha256"] or len(content) != artifact["size"]:
        raise PublicValidationFailure("Validation artifact content mismatch")
    return artifact, content


def _public_status(report, manifest):
    expected = {check["id"]: check["kind"] for step in validate_manifest(manifest)
                for check in step["checks"]}
    checks = report.get("checks")
    if not isinstance(checks, list) or len(checks) != len(expected):
        raise PublicValidationFailure("Public checker omitted or added assertions")
    seen, executable = set(), []
    for item in checks:
        if not isinstance(item, dict):
            raise PublicValidationFailure("Public check result is malformed")
        identifier, kind = item.get("id"), item.get("kind")
        if identifier in seen or expected.get(identifier) != kind:
            raise PublicValidationFailure("Public check identities differ from the frozen suite")
        seen.add(identifier)
        status = item.get("status")
        if status not in {"passed", "failed", "unavailable"}:
            raise PublicValidationFailure("Unknown public check status")
        if kind == "unavailable":
            if status != "unavailable":
                raise PublicValidationFailure("An unavailable public assertion was relabeled")
        else:
            executable.append(status)
    # Infrastructure gaps take precedence over a partial failure. Declared
    # exclusions remain explicit and never join the executed-check denominator.
    status = ("unavailable" if not executable or "unavailable" in executable else
              "failed" if "failed" in executable else "passed")
    if report.get("status") != status:
        raise PublicValidationFailure("Public check summary disagrees with assertion outcomes")
    return status


def verify_public_code_validation(store, run_id, generated_code):
    """Verify the latest attempt for these exact final canonical code bytes."""
    from ..analysis.public_code_capabilities import list_attempts
    run = store.get_run(run_id)
    sid = run["session_id"]
    binding, manifest = _binding(store, run)
    source_bytes = canonical_submission(generated_code)
    source_hash = submission_sha256(generated_code)
    attempts = list_attempts(store, run_id)
    if len(attempts) > binding["max_attempts"]:
        raise PublicValidationFailure("Public checker attempt bound exceeded")
    if [item["attempt"] for item in attempts] != list(range(1, len(attempts) + 1)):
        raise PublicValidationFailure("Public checker attempt sequence is malformed")
    summary = {"status": "missing", "attempt_count": len(attempts),
        "generated_code_sha256": source_hash, "manifest_sha256": binding["manifest_sha256"],
        "checker_source_sha256": binding["checker_source_sha256"],
        "official_correctness": "not_evaluated", "checks": []}
    summary["latest_attempt"] = attempts[-1]["attempt"] if attempts else None
    summary["selected_attempt"] = None
    if not attempts or attempts[-1]["generated_code_sha256"] != source_hash:
        return summary
    attempt = attempts[-1]
    summary["selected_attempt"] = attempt["attempt"]
    for key in ("task_binding", "manifest_artifact_id", "manifest_sha256",
                "checker_source_sha256", "handler_source_sha256", "checker_version"):
        if attempt.get(key) != binding[key]:
            raise PublicValidationFailure("Public checker attempt binding differs from this task")
    artifact, stored_source = _artifact(store, sid, run_id, attempt["submission_artifact_id"],
        maximum=MAX_SUBMISSION_BYTES, kind="public_code_submission")
    if stored_source != source_bytes or artifact["sha256"] != source_hash:
        raise PublicValidationFailure("Final source differs from the checked source artifact")
    with store.transaction() as conn:
        row = conn.execute("SELECT * FROM execution_jobs WHERE run_id=? AND idempotency_key=?",
                           (run_id, attempt["call_id"])).fetchone()
    if row is None:
        return {**summary, "status": "unknown", "call_id": attempt["call_id"]}
    job = dict(row)
    for key in ("arguments", "definition", "result_artifact"):
        if job[key] is not None:
            job[key] = json.loads(job[key])
    if (job["session_id"] != sid or job["tool_name"] != TOOL or job["handler_ref"] != HANDLER
            or _sha(canonical_json(job["definition"]).encode()) != job["definition_hash"]):
        raise PublicValidationFailure("Public checker job definition or scope mismatch")
    fingerprint = _sha(canonical_json({"tool_name": TOOL, "arguments": job["arguments"],
        "definition_hash": job["definition_hash"], "handler_ref": HANDLER,
        "timeout_seconds": float(job["timeout_seconds"])}).encode())
    if fingerprint != job["fingerprint"]:
        raise PublicValidationFailure("Public checker durable intent fingerprint differs")
    identity = {key: attempt[key] for key in (
        "call_id", "attempt", "submission_artifact_id", "generated_code_sha256", "task_binding",
        "manifest_artifact_id", "manifest_sha256", "checker_source_sha256", "handler_source_sha256",
        "checker_version")}
    identity.update(run_id=run_id, session_id=sid, job_id=job["id"])
    if job["arguments"].get("generated_code") != generated_code:
        raise PublicValidationFailure("Public checker job source differs from the final source")
    for key, value in identity.items():
        if key not in {"job_id"} and job["arguments"].get(key) != value:
            raise PublicValidationFailure("Public checker job arguments differ from reserved intent")
    action = store.get_action(run_id, attempt["call_id"])
    if (action is None or action["tool_name"] != TOOL
            or action["args"].get("generated_code") != generated_code):
        raise PublicValidationFailure("Public checker action is not the checked code")
    if job["status"] != "succeeded":
        return {**summary, "status": "unavailable", "job_id": job["id"],
                "job_status": job["status"], "call_id": attempt["call_id"]}
    pointer = job["result_artifact"]
    if not isinstance(pointer, dict):
        raise PublicValidationFailure("Public checker has no result artifact")
    result_artifact, content = _artifact(store, sid, run_id, pointer["id"], maximum=1024 * 1024)
    if (result_artifact["sha256"] != pointer["sha256"]
            or result_artifact["provenance"] != {"job_id": job["id"],
                "definition_hash": job["definition_hash"], "role": "job_result"}):
        raise PublicValidationFailure("Public checker result artifact identity mismatch")
    result = json.loads(content)
    if action["finished_at"] is None or action["result"] != result:
        raise PublicValidationFailure("Public checker action and job terminal results disagree")
    if result.get("status") != "ok":
        # ToolResult defaults to ok. Execution/transport errors cannot become a
        # completed public-contract failure or a scientific correctness zero.
        return {**summary, "status": "unavailable", "job_id": job["id"]}
    data = result.get("data")
    if not isinstance(data, dict) or data.get("identity") != identity:
        raise PublicValidationFailure("Public checker result provenance mismatch")
    report = data.get("validation")
    if (not isinstance(report, dict)
            or report.get("generated_code_sha256") != source_hash
            or report.get("manifest_sha256") != binding["manifest_sha256"]
            or report.get("checker_source_sha256") != binding["checker_source_sha256"]
            or report.get("schema_version") != "public-code-check-result-v1"):
        raise PublicValidationFailure("Public checker report hashes or version mismatch")
    status = _public_status(report, manifest)
    return {**summary, "status": status, "checks": report["checks"], "identity": identity,
            "result_artifact_id": pointer["id"], "result_sha256": pointer["sha256"],
            "declared_unavailable_assertions": sum(item["kind"] == "unavailable" for item in report["checks"]),
            "assurance": "Outcomes on available public assertions only; excluded assertions and official correctness are unvalidated."}
