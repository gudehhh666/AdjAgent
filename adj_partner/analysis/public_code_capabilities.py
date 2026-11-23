"""Durable host capability for checking one bound generated-code submission.

The model supplies code only. The run checkpoint supplies the immutable public
manifest and source pins; this fixed handler runs the checker through the
existing OS sandbox and records every attempted check before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from ..contracts import ToolResult
from ..execution import JobManager
from ..execution.manager import TERMINAL
from ..tools import ToolFailure, ToolSpec
from ..tools.catalog import canonical_json, validator
from . import public_code_checker as checker
from .sandbox import SandboxUnavailable


PUBLIC_CODE_TOOL = "public_code_check"
PUBLIC_CODE_TOOLS = (PUBLIC_CODE_TOOL,)
PUBLIC_CODE_HANDLER = "adj_partner.analysis.public_code_capabilities:public_code_check_job"
PUBLIC_CODE_VERSION = "1"
CHECKER_VERSION = "public-code-check-v1"
MAX_ATTEMPTS = 2
MAX_TIMEOUT_SECONDS = 30
MAX_SUBMISSION_BYTES = checker.MAX_SUBMISSION_BYTES
HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def _public_schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checker_source_digest():
    return checker.checker_source_digest()


run_public_code_checks = checker.run_public_code_checks


def handler_source_digest():
    return _hash_bytes(Path(__file__).read_bytes())


def _strict_json(content: bytes):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(content.decode("utf-8"), object_pairs_hook=object_pairs,
                      parse_constant=constant)


def install_public_check_manifest(store, session_id: str, manifest: dict) -> dict:
    """Store and return host-owned immutable checker input for one session."""
    checker.validate_manifest(manifest)
    content = canonical_json(manifest).encode("utf-8")
    if len(content) > checker.MAX_MANIFEST_BYTES:
        raise ValueError("Public code checker manifest exceeds its byte limit")
    digest = _hash_bytes(content)
    return store.put_artifact(content, "public-code-check-manifest.json", "application/json",
        {"kind": "public_code_check_manifest", "sha256": digest,
         "checker_version": CHECKER_VERSION}, session_id=session_id)


def load_manifest(store, session_id: str, binding: dict) -> dict:
    """Load only the exact session-scoped manifest pinned by the run binding."""
    identifier, expected = binding.get("manifest_artifact_id"), binding.get("manifest_sha256")
    if (not isinstance(identifier, str) or not identifier or not isinstance(expected, str)
            or not HASH_RE.fullmatch(expected)):
        raise ValueError("Public checker manifest identity is malformed")
    metadata = next((item for item in store.artifacts(session_id=session_id)
                     if item["id"] == identifier), None)
    if metadata is None:
        raise ValueError("Public checker manifest is unavailable in this session")
    provenance = metadata.get("provenance", {})
    if (metadata.get("session_id") != session_id or metadata.get("run_id") is not None
            or provenance.get("kind") != "public_code_check_manifest"
            or provenance.get("checker_version") != CHECKER_VERSION
            or metadata.get("sha256") != expected or provenance.get("sha256") != expected):
        raise ValueError("Public checker manifest scope, kind, or hash does not match the run binding")
    content = store.read_artifact(identifier, max_bytes=checker.MAX_MANIFEST_BYTES)
    if _hash_bytes(content) != expected:
        raise ValueError("Public checker manifest content hash mismatch")
    manifest = _strict_json(content)
    checker.validate_manifest(manifest)
    if canonical_json(manifest).encode("utf-8") != content:
        raise ValueError("Public checker manifest is not canonical JSON")
    return manifest


def _ensure_attempts_table(store):
    with store.transaction(write=True) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS public_code_check_attempts (
            run_id TEXT NOT NULL REFERENCES runs(id),
            session_id TEXT NOT NULL REFERENCES sessions(id),
            call_id TEXT NOT NULL,
            attempt INTEGER NOT NULL CHECK(attempt BETWEEN 1 AND 2),
            job_id TEXT,
            submission_artifact_id TEXT NOT NULL,
            generated_code_sha256 TEXT NOT NULL,
            task_binding TEXT NOT NULL,
            manifest_artifact_id TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            checker_source_sha256 TEXT NOT NULL,
            handler_source_sha256 TEXT NOT NULL,
            checker_version TEXT NOT NULL,
            timeout_seconds REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'reserved',
            created_at REAL NOT NULL,
            finished_at REAL,
            PRIMARY KEY(run_id, call_id),
            UNIQUE(run_id, attempt))""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS public_code_attempt_identity_immutable
            BEFORE UPDATE OF run_id,session_id,call_id,attempt,submission_artifact_id,
                generated_code_sha256,task_binding,manifest_artifact_id,manifest_sha256,
                checker_source_sha256,handler_source_sha256,checker_version,timeout_seconds,created_at
            ON public_code_check_attempts
            BEGIN SELECT RAISE(ABORT,'Public checker attempt identity is immutable'); END""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS public_code_attempt_job_id_immutable
            BEFORE UPDATE OF job_id ON public_code_check_attempts
            WHEN OLD.job_id IS NOT NULL AND NEW.job_id IS NOT OLD.job_id
            BEGIN SELECT RAISE(ABORT,'Public checker job identity is immutable'); END""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS public_code_attempt_terminal_immutable
            BEFORE UPDATE OF status,finished_at ON public_code_check_attempts
            WHEN OLD.status IN ('succeeded','failed','cancelled','timed_out','uncertain')
                AND (NEW.status IS NOT OLD.status OR NEW.finished_at IS NOT OLD.finished_at)
            BEGIN SELECT RAISE(ABORT,'Terminal public checker attempt is immutable'); END""")


def _attempt_table_exists(store):
    with store.transaction() as conn:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='public_code_check_attempts'").fetchone() is not None


def list_attempts(store, run_id: str) -> list[dict]:
    """Return the durable attempt ledger for host verification and audit."""
    if not _attempt_table_exists(store):
        return []
    with store.transaction() as conn:
        rows = conn.execute("SELECT * FROM public_code_check_attempts WHERE run_id=? ORDER BY attempt",
                            (run_id,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["task_binding"] = json.loads(item["task_binding"])
        result.append(store.redact(item))
    return result


def _binding_for(store, session_id, run_id):
    from ..research.acceptance import task_binding

    run = store.get_run(run_id)
    if run["session_id"] != session_id:
        raise PermissionError("Public checker run belongs to another session")
    binding = run.get("state", {}).get("public_code_validation_binding")
    if not isinstance(binding, dict):
        raise ToolFailure("public_code_binding_missing", "This run has no host-bound public checker contract", "denied")
    task = store.get_task(run["task_id"])
    contract = task.get("acceptance", {}).get("public_code_validation")
    if not isinstance(contract, dict) or binding.get("task_binding") != task_binding(task):
        raise ToolFailure("public_code_binding_mismatch", "The task no longer matches the host checker binding", "denied")
    if (binding.get("manifest_artifact_id") != contract.get("manifest_artifact_id")
            or binding.get("manifest_sha256") != contract.get("manifest_sha256")
            or binding.get("checker_version") != contract.get("checker_version")
            or binding.get("max_attempts") != MAX_ATTEMPTS
            or contract.get("max_attempts") != MAX_ATTEMPTS
            or contract.get("checker_version") != CHECKER_VERSION):
        raise ToolFailure("public_code_binding_mismatch", "The host checker contract does not match task acceptance", "denied")
    if binding.get("checker_source_sha256") != checker_source_digest():
        raise ToolFailure("public_code_checker_changed", "The pinned public checker source has changed", "denied")
    if binding.get("handler_source_sha256") != handler_source_digest():
        raise ToolFailure("public_code_handler_changed", "The pinned public checker handler has changed", "denied")
    for key in ("checker_source_sha256", "handler_source_sha256"):
        if not isinstance(binding.get(key), str) or not HASH_RE.fullmatch(binding[key]):
            raise ToolFailure("public_code_binding_malformed", "The host checker source pin is malformed", "denied")
    return run, task, binding


def _identity(run_id, session_id, call_id, attempt, record, binding, job_id=None):
    value = {"run_id": run_id, "session_id": session_id, "call_id": call_id,
        "attempt": attempt, "submission_artifact_id": record["submission_artifact_id"],
        "generated_code_sha256": record["generated_code_sha256"],
        "task_binding": binding["task_binding"],
        "manifest_artifact_id": binding["manifest_artifact_id"],
        "manifest_sha256": binding["manifest_sha256"],
        "checker_source_sha256": binding["checker_source_sha256"],
        "handler_source_sha256": binding["handler_source_sha256"],
        "checker_version": binding["checker_version"]}
    if job_id is not None:
        value["job_id"] = job_id
    return value


def _row_for_call(store, run_id, call_id):
    if not _attempt_table_exists(store):
        return None
    with store.transaction() as conn:
        row = conn.execute("SELECT * FROM public_code_check_attempts WHERE run_id=? AND call_id=?",
                           (run_id, call_id)).fetchone()
    if row is None:
        return None
    value = dict(row)
    value["task_binding"] = json.loads(value["task_binding"])
    return value


class PublicCodeCapabilities:
    def __init__(self, store, *, jobs=None):
        self.store = store
        if jobs is not None and (jobs.store.root != store.root or jobs.store.db_path != store.db_path):
            raise ValueError("Public checker jobs must share the same Store root/database")
        self.jobs = jobs if jobs is not None and jobs.store is store else JobManager(store)
        _ensure_attempts_table(store)

    checker_source_digest = staticmethod(checker_source_digest)
    handler_source_digest = staticmethod(handler_source_digest)
    list_attempts = staticmethod(list_attempts)

    @staticmethod
    def definitions():
        return [ToolSpec(PUBLIC_CODE_TOOL,
            "Run the host-pinned public structural/fixture checker on a generated_code mapping. "
            "The checker may return bounded diagnostics and does not establish scientific truth.",
            _public_schema({"generated_code": {"type": "object", "maxProperties": checker.MAX_STEPS,
                "additionalProperties": {"type": "string", "minLength": 1,
                    "maxLength": checker.MAX_SOURCE_BYTES}},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECONDS}},
                ["generated_code"]), version=PUBLIC_CODE_VERSION)]

    @staticmethod
    def _job_spec():
        identity_properties = {
            "run_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "session_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "call_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "attempt": {"type": "integer", "minimum": 1, "maximum": 2},
            "submission_artifact_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "generated_code_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "task_binding": {"type": "object"},
            "manifest_artifact_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "manifest_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "checker_source_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "handler_source_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "checker_version": {"const": CHECKER_VERSION},
        }
        identity_fields = ["run_id", "session_id", "call_id", "attempt", "submission_artifact_id",
            "generated_code_sha256", "task_binding", "manifest_artifact_id", "manifest_sha256",
            "checker_source_sha256", "handler_source_sha256", "checker_version"]
        return ToolSpec(PUBLIC_CODE_TOOL, "Host-controlled public code validation job.",
            _public_schema({"generated_code": {"type": "object", "maxProperties": checker.MAX_STEPS,
                "additionalProperties": {"type": "string", "minLength": 1,
                    "maxLength": checker.MAX_SOURCE_BYTES}},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECONDS},
                **identity_properties}, ["generated_code", "timeout_seconds", *identity_fields]),
            version=PUBLIC_CODE_VERSION)

    def register(self, catalog):
        specs = self.definitions()
        job_spec = self._job_spec()
        for spec in specs:
            self.jobs.register(job_spec, PUBLIC_CODE_HANDLER)
        catalog.register_many((spec, self._check) for spec in specs)

    def _reserve(self, context, generated_code, timeout_seconds):
        context.check_cancelled()
        guard = getattr(context, "execution_guard", None)
        if callable(guard):
            guard()
        if not context.call_id or not isinstance(context.call_id, str):
            raise ToolFailure("call_identity_required", "Public checker requires the original tool call ID")
        _, _, binding = _binding_for(self.store, context.session_id, context.run_id)
        manifest = load_manifest(self.store, context.session_id, binding)
        step_ids = {step["step_id"] for step in manifest["ordered_steps"]}
        if (not isinstance(generated_code, dict)
                or len(generated_code) > checker.MAX_STEPS
                or any(not isinstance(key, str) or not isinstance(value, str) or not value
                       for key, value in generated_code.items())):
            raise ToolFailure("public_code_submission_invalid", "generated_code must map step IDs to nonempty source strings")
        unknown = set(generated_code) - step_ids
        if unknown:
            raise ToolFailure("public_code_submission_invalid", "generated_code contains an unknown public step ID")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
            timeout_seconds = 10
        if callable(guard):
            remaining = getattr(guard, "remaining_seconds", None)
            if callable(remaining):
                remaining_seconds = remaining()
                if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                        or not math.isfinite(remaining_seconds)):
                    raise ToolFailure("public_code_budget_unknown", "Remaining runtime budget is unavailable")
                available = int(math.floor(remaining_seconds - 2.0))
                if available < 1:
                    raise ToolFailure("public_code_budget_exhausted", "Insufficient remaining runtime budget to start a public check")
                timeout_seconds = min(timeout_seconds, available)
        try:
            generated_hash = checker.submission_sha256(generated_code)
            content = canonical_json(generated_code).encode("utf-8")
        except Exception as exc:
            raise ToolFailure("public_code_submission_invalid", "generated_code is not bounded finite JSON") from exc
        if len(content) > MAX_SUBMISSION_BYTES:
            raise ToolFailure("public_code_submission_invalid", "generated_code exceeds the checker input limit")

        # A repeated call ID can only recover the same immutable intent. It is
        # never turned into a second reservation or checker execution.
        existing = _row_for_call(self.store, context.run_id, context.call_id)
        if existing is not None:
            if (existing["generated_code_sha256"] != generated_hash
                    or existing["timeout_seconds"] != float(timeout_seconds)):
                raise ToolFailure("public_code_attempt_conflict", "This call ID already names a different checker attempt", "uncertain")
            if not existing["job_id"]:
                raise ToolFailure("public_code_attempt_uncertain", "The prior checker reservation has no recoverable job; do not replay", "uncertain")
            return self.resolve_job(context.session_id, context.run_id, existing["job_id"])

        submission = self.store.put_artifact(content, "public-code-submission.json", "application/json",
            {"kind": "public_code_submission", "call_id": context.call_id,
             "generated_code_sha256": generated_hash}, session_id=context.session_id, run_id=context.run_id)
        attempt_id = _identity(context.run_id, context.session_id, context.call_id, 1,
            {"submission_artifact_id": submission["id"], "generated_code_sha256": generated_hash}, binding)
        with self.store.transaction(write=True) as conn:
            prior = conn.execute("SELECT MAX(attempt) AS maximum FROM public_code_check_attempts WHERE run_id=?",
                                 (context.run_id,)).fetchone()["maximum"] or 0
            if prior >= MAX_ATTEMPTS:
                raise ToolFailure("public_code_attempt_limit", "The public checker attempt limit has been reached", "denied")
            attempt = int(prior) + 1
            attempt_id["attempt"] = attempt
            try:
                conn.execute("""INSERT INTO public_code_check_attempts(
                    run_id,session_id,call_id,attempt,submission_artifact_id,generated_code_sha256,
                    task_binding,manifest_artifact_id,manifest_sha256,checker_source_sha256,
                    handler_source_sha256,checker_version,timeout_seconds,status,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'reserved',?)""",
                    (context.run_id, context.session_id, context.call_id, attempt, submission["id"],
                     generated_hash, canonical_json(binding["task_binding"]), binding["manifest_artifact_id"],
                     binding["manifest_sha256"], binding["checker_source_sha256"],
                     binding["handler_source_sha256"], binding["checker_version"],
                     float(timeout_seconds), time.time()))
            except Exception as exc:
                raise ToolFailure("public_code_attempt_uncertain", "The checker attempt could not be reserved atomically", "uncertain") from exc
        attempt_id["attempt"] = attempt
        arguments = {"generated_code": generated_code, "timeout_seconds": timeout_seconds,
                     **attempt_id}
        # The detached supervisor gets two seconds of cleanup/serialization
        # overhead beyond the caller's remaining checker budget.
        job_timeout = float(timeout_seconds) + 2.0
        try:
            context.check_cancelled()
            job = self.jobs.submit(context.session_id, context.run_id, PUBLIC_CODE_TOOL,
                arguments, context.call_id, timeout_seconds=job_timeout)
        except Exception as exc:
            job = self._find_submitted_job(context.session_id, context.run_id, context.call_id,
                                            arguments, job_timeout)
            if job is None:
                self._set_attempt_status(context.run_id, context.call_id, "uncertain")
                raise ToolFailure("public_code_submission_uncertain",
                    "The checker intent may have been submitted; it will not be replayed", "uncertain") from exc
        with self.store.transaction(write=True) as conn:
            conn.execute("UPDATE public_code_check_attempts SET job_id=?,status='submitted' WHERE run_id=? AND call_id=? AND job_id IS NULL",
                         (job["id"], context.run_id, context.call_id))
        return ToolResult(status="pending", data={"job_id": job["id"], "status": job["status"]}, job_id=job["id"])

    def _check(self, context, generated_code, timeout_seconds=10):
        if context.store.root != self.store.root or context.store.db_path != self.store.db_path:
            raise ToolFailure("public_code_store_mismatch", "Public checker is bound to another Store", "denied")
        return self._reserve(context, generated_code, timeout_seconds)

    def _find_submitted_job(self, session_id, run_id, call_id, arguments, timeout_seconds):
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM execution_jobs WHERE run_id=? AND idempotency_key=?",
                               (run_id, call_id)).fetchone()
        if row is None:
            return None
        if (row["session_id"] != session_id or row["tool_name"] != PUBLIC_CODE_TOOL
                or row["handler_ref"] != PUBLIC_CODE_HANDLER
                or row["arguments"] != canonical_json(arguments)
                or float(row["timeout_seconds"]) != float(timeout_seconds)):
            raise ValueError("Existing public checker job identity conflicts with the reserved attempt")
        return {"id": row["id"], "status": row["status"]}

    def _set_attempt_status(self, run_id, call_id, status):
        with self.store.transaction(write=True) as conn:
            conn.execute("""UPDATE public_code_check_attempts SET status=?,finished_at=?
                WHERE run_id=? AND call_id=?
                  AND status NOT IN ('succeeded','failed','cancelled','timed_out','uncertain')""",
                         (status, time.time(), run_id, call_id))

    def _uncertain(self, run_id, call_id, code, message):
        self._set_attempt_status(run_id, call_id, "uncertain")
        return ToolResult(status="uncertain", error={"code": code, "message": message})

    def _row_for_job(self, run_id, job_id):
        if not _attempt_table_exists(self.store):
            return None
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM public_code_check_attempts WHERE run_id=? AND job_id=?",
                               (run_id, job_id)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["task_binding"] = json.loads(value["task_binding"])
        return value

    def resolve_job(self, session_id, run_id, job_id):
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise PermissionError("Public checker job run/session mismatch")
        try:
            job = self.jobs.get(job_id, session_id)
        except (KeyError, ValueError, OSError):
            row = self._row_for_job(run_id, job_id)
            if row is not None:
                return self._uncertain(run_id, row["call_id"], "public_code_job_missing",
                    "The reserved public checker job is missing or unreadable; it will not be replayed")
            return ToolResult(status="uncertain", error={"code": "public_code_job_missing",
                "message": "The public checker job is missing or unreadable; it will not be replayed"})
        if (job["run_id"] != run_id or job["tool_name"] != PUBLIC_CODE_TOOL
                or job["handler_ref"] != PUBLIC_CODE_HANDLER):
            raise PermissionError("Job is not this run's public checker capability")
        row = _row_for_call(self.store, run_id, job["idempotency_key"])
        expected_args = None
        if row is not None:
            try:
                _, _, pinned = _binding_for(self.store, session_id, run_id)
                expected_args = _identity(run_id, session_id, job["idempotency_key"],
                    row["attempt"], row, pinned)
            except Exception:
                expected_args = None
        action = self.store.get_action(run_id, job["idempotency_key"])
        action_args = action.get("args") if isinstance(action, dict) else None
        expected_job_args = None
        if isinstance(action_args, dict) and row is not None and expected_args is not None:
            expected_job_args = {"generated_code": action_args.get("generated_code"),
                "timeout_seconds": row["timeout_seconds"], **expected_args}
        try:
            submitted_hash = checker.submission_sha256(job.get("arguments", {}).get("generated_code", {}))
        except (TypeError, ValueError, RecursionError):
            submitted_hash = None
        if (row is None or row["job_id"] not in (None, job_id) or expected_args is None
                or action is None or action.get("tool_name") != PUBLIC_CODE_TOOL
                or expected_job_args is None or job.get("arguments") != expected_job_args
                or submitted_hash != row["generated_code_sha256"]):
            call_id = row["call_id"] if row is not None else job.get("idempotency_key", "")
            return self._uncertain(run_id, call_id, "public_code_recovery_mismatch",
                "The checker job arguments do not match the immutable action and attempt reservation")
        try:
            job = self.jobs.reconcile(job_id, session_id)
        except (KeyError, ValueError, OSError):
            return self._uncertain(run_id, row["call_id"], "public_code_job_missing",
                "The reserved public checker job disappeared or became unreadable during recovery")
        if job["status"] not in TERMINAL:
            return ToolResult(status="pending", data={"job_id": job_id, "status": job["status"]}, job_id=job_id)
        try:
            result = self.jobs.result(job_id, session_id)
        except (KeyError, ValueError, OSError, TypeError):
            return self._uncertain(run_id, row["call_id"], "public_code_result_unavailable",
                "The terminal public checker result is missing or failed integrity validation; it will not be replayed")
        if result is None:
            return self._uncertain(run_id, row["call_id"], "public_code_result_unavailable",
                "The terminal public checker job has no readable result; it will not be replayed")
        if not isinstance(result, ToolResult) or not isinstance(result.status, str):
            return self._uncertain(run_id, row["call_id"], "public_code_result_unavailable",
                "The terminal public checker result has an invalid structure")
        if result.status == "ok":
            if (not isinstance(result.data, dict)
                    or not isinstance(result.data.get("validation"), dict)
                    or not isinstance(result.data.get("identity"), dict)):
                return self._uncertain(run_id, row["call_id"], "public_code_result_unavailable",
                    "The terminal public checker result has a malformed payload")
            try:
                _, _, binding = _binding_for(self.store, session_id, run_id)
                expected = _identity(run_id, session_id, job["idempotency_key"], row["attempt"], row, binding, job_id)
                if result.data.get("identity") != expected:
                    raise ValueError("terminal identity differs from the durable reservation")
                if result.data.get("validation", {}).get("generated_code_sha256") != row["generated_code_sha256"]:
                    raise ValueError("terminal report is for a different source mapping")
            except Exception as exc:
                return self._uncertain(run_id, row["call_id"], "public_code_recovery_mismatch",
                    f"The terminal checker result could not be verified: {type(exc).__name__}")
        self._set_attempt_status(run_id, job["idempotency_key"], job["status"])
        return result

    def recover_action(self, session_id, run_id, call):
        if call.name != PUBLIC_CODE_TOOL:
            return None
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise PermissionError("Public checker action run/session mismatch")
        row = _row_for_call(self.store, run_id, call.id)
        if row is None:
            return None
        action = self.store.get_action(run_id, call.id)
        try:
            proposed_matches = (action is not None and action.get("tool_name") == PUBLIC_CODE_TOOL
                and canonical_json(action.get("args")) == canonical_json(call.arguments))
        except (TypeError, ValueError):
            proposed_matches = False
        if not proposed_matches:
            return ToolResult(status="uncertain", error={"code": "public_code_recovery_mismatch",
                "message": "The proposed checker call differs from the immutable recorded action"})
        if not row["job_id"]:
            return self._uncertain(run_id, call.id, "public_code_attempt_uncertain",
                "The public checker intent has no recoverable job and will not be replayed")
        return self.resolve_job(session_id, run_id, row["job_id"])


def _verify_worker_identity(context, generated_code, timeout_seconds, attempt_identity):
    if not isinstance(attempt_identity, dict):
        raise ToolFailure("public_code_identity_invalid", "Public checker job identity is missing", "denied")
    call_id = context.call_id
    run, task, binding = _binding_for(context.store, context.session_id, context.run_id)
    row = _row_for_call(context.store, context.run_id, call_id)
    if row is None:
        raise ToolFailure("public_code_attempt_missing", "The durable checker attempt reservation is missing", "denied")
    attempt = row["attempt"]
    expected = _identity(context.run_id, context.session_id, call_id, attempt, row, binding)
    if attempt_identity != expected:
        raise ToolFailure("public_code_identity_mismatch", "The checker job identity does not match its durable reservation", "denied")
    if row["job_id"] not in (None, context.job_id):
        raise ToolFailure("public_code_identity_mismatch", "The checker job ID conflicts with its durable reservation", "denied")
    if (row["session_id"] != context.session_id or row["generated_code_sha256"] != checker.submission_sha256(generated_code)
            or float(row["timeout_seconds"]) != float(timeout_seconds)):
        raise ToolFailure("public_code_identity_mismatch", "The checker submission differs from its reserved intent", "denied")
    with context.store.transaction(write=True) as conn:
        changed = conn.execute("UPDATE public_code_check_attempts SET job_id=? WHERE run_id=? AND call_id=? AND job_id IS NULL",
                               (context.job_id, context.run_id, call_id)).rowcount
        if not changed:
            current = conn.execute("SELECT job_id FROM public_code_check_attempts WHERE run_id=? AND call_id=?",
                                   (context.run_id, call_id)).fetchone()
            if current is None or current["job_id"] != context.job_id:
                raise ToolFailure("public_code_identity_mismatch", "The checker job could not claim its reservation", "denied")
    artifacts = {item["id"]: item for item in context.store.artifacts(context.session_id, context.run_id)}
    submission = artifacts.get(row["submission_artifact_id"])
    if (submission is None or submission["provenance"].get("kind") != "public_code_submission"
            or submission["provenance"].get("call_id") != call_id
            or submission["provenance"].get("generated_code_sha256") != row["generated_code_sha256"]):
        raise ToolFailure("public_code_submission_integrity", "The stored generated-code submission is missing or mismatched", "denied")
    content = context.store.read_artifact(submission["id"], max_bytes=MAX_SUBMISSION_BYTES)
    stored_code = _strict_json(content)
    if stored_code != generated_code or checker.submission_sha256(stored_code) != row["generated_code_sha256"]:
        raise ToolFailure("public_code_submission_integrity", "The stored source does not match the dispatched checker submission", "denied")
    manifest = load_manifest(context.store, context.session_id, binding)
    identity = _identity(context.run_id, context.session_id, call_id, attempt, row, binding, context.job_id)
    return binding, row, manifest, identity


def public_code_check_job(context, generated_code, timeout_seconds, run_id, session_id,
                          call_id, attempt, submission_artifact_id, generated_code_sha256,
                          task_binding, manifest_artifact_id, manifest_sha256,
                          checker_source_sha256, handler_source_sha256, checker_version):
    """Fixed durable handler; model input cannot choose checker, imports, or fixtures."""
    context.check_cancelled()
    attempt_identity = {"run_id": run_id, "session_id": session_id, "call_id": call_id,
        "attempt": attempt, "submission_artifact_id": submission_artifact_id,
        "generated_code_sha256": generated_code_sha256, "task_binding": task_binding,
        "manifest_artifact_id": manifest_artifact_id, "manifest_sha256": manifest_sha256,
        "checker_source_sha256": checker_source_sha256, "handler_source_sha256": handler_source_sha256,
        "checker_version": checker_version}
    binding, row, manifest, identity = _verify_worker_identity(
        context, generated_code, timeout_seconds, attempt_identity)
    deadline = context._job.get("deadline") if hasattr(context, "_job") else None
    if isinstance(deadline, (int, float)) and math.isfinite(deadline):
        remaining = deadline - time.time() - 1.0
        if remaining <= 0:
            raise ToolFailure("public_code_checker_deadline", "No execution time remains for the public checker")
        effective_timeout = min(float(timeout_seconds), remaining)
    else:
        effective_timeout = float(timeout_seconds)
    callback_failure = []

    def poll_cancel_and_binding():
        try:
            context.check_cancelled()
            _, _, current = _binding_for(context.store, context.session_id, context.run_id)
            if current != binding:
                raise ToolFailure("public_code_binding_changed", "The bound public checker contract changed during execution", "denied")
        except BaseException as exc:
            callback_failure.append(exc)
            raise

    try:
        validation = run_public_code_checks(generated_code, manifest, context.workspace,
            checker_source_sha256=binding["checker_source_sha256"],
            timeout_seconds=effective_timeout, cancel_check=poll_cancel_and_binding)
    except SandboxUnavailable as exc:
        raise ToolFailure("public_code_checker_unavailable", str(exc)[:500]) from exc
    except checker.PublicCodeCheckError as exc:
        raise ToolFailure("public_code_checker_invalid", str(exc)[:500]) from exc
    if callback_failure:
        raise callback_failure[0]
    context.check_cancelled()
    current_binding, current_row, _, current_identity = _verify_worker_identity(
        context, generated_code, timeout_seconds, attempt_identity)
    if current_identity != identity or current_row["generated_code_sha256"] != row["generated_code_sha256"]:
        raise ToolFailure("public_code_identity_changed", "The checker binding changed during execution", "denied")
    if validation.get("generated_code_sha256") != row["generated_code_sha256"]:
        raise ToolFailure("public_code_checker_integrity", "Checker report is for a different generated-code mapping", "denied")
    report = canonical_json(validation)
    if len(report.encode("utf-8")) > checker.MAX_RESULT_BYTES:
        raise ToolFailure("public_code_checker_output_invalid", "The bounded checker report exceeds its output limit")
    return ToolResult(data={"validation": json.loads(report), "identity": identity},
        artifacts=[{"id": row["submission_artifact_id"], "sha256": _hash_bytes(
            context.store.read_artifact(row["submission_artifact_id"], max_bytes=MAX_SUBMISSION_BYTES))}],
        job_id=context.job_id)


def public_code_check_job_finalize(conn, store, job, status, result):
    """Record the worker's terminal disposition in the same job transaction."""
    conn.execute("UPDATE public_code_check_attempts SET status=?,finished_at=? WHERE run_id=? AND call_id=?",
                 (status, time.time(), job["run_id"], job["idempotency_key"]))

