"""Durable LOCAL execution intent and recovery, never a distributed scheduler.

The host registers trusted code; model arguments cannot select imports or commands.
The shared executor remains responsible for capability authorization and approvals.
This module performs scope/input checks but does not grant an approval by itself.
"""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from ..contracts import ToolResult, new_id
from ..tools.catalog import ToolSpec, canonical_json, validator


TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "uncertain"})
ACTIVE = frozenset({"queued", "running", "cancel_requested"})
HEARTBEAT_STALE_SECONDS = 5.0
MAX_LOG_BYTES = 32768
MAX_RESULT_BYTES = 1024 * 1024
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", re.ASCII)


def failure(code, message, status="error"):
    return ToolResult(status=status, error={"code": code, "message": message})


def bounded_text(store, text, limit=MAX_LOG_BYTES):
    text = str(text).encode("utf-8", errors="replace").decode("utf-8")
    return store.redact(text).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def worker_command(*arguments):
    # -I excludes inherited Python path/user site and the process cwd. Only the
    # host installation root is added, never a path taken from model arguments.
    bootstrap = f"import sys;sys.path.insert(0,{str(PACKAGE_ROOT)!r});from adj_partner.execution.worker import main;main()"
    return [sys.executable, "-I", "-c", bootstrap, *map(str, arguments)]


def worker_environment(workspace):
    temporary = Path(workspace) / "tmp"
    temporary.mkdir(mode=0o700, exist_ok=True)
    return {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
            "TMPDIR": str(temporary), "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1", "LANG": "C.UTF-8"}


def _definition(spec):
    return {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema,
            "output_schema": spec.output_schema, "version": spec.version,
            "network": spec.network, "requires_approval": spec.requires_approval}


def _decode(row):
    if row is None:
        raise KeyError("Unknown job")
    value = dict(row)
    for key in ("arguments", "definition", "result_artifact"):
        if key in value and value[key] is not None:
            value[key] = json.loads(value[key])
    for key in ("stdout_truncated", "stderr_truncated"):
        if key in value:
            value[key] = bool(value[key])
    return value


class JobManager:
    """Attach to durable jobs using only a Store; no in-memory handlers required."""

    def __init__(self, store):
        self.store = store
        with store.transaction(write=True) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS execution_handlers (
                tool_name TEXT PRIMARY KEY, handler_ref TEXT NOT NULL,
                definition TEXT NOT NULL, definition_hash TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS execution_jobs (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                run_id TEXT NOT NULL REFERENCES runs(id), tool_name TEXT NOT NULL,
                handler_ref TEXT NOT NULL, definition TEXT NOT NULL, definition_hash TEXT NOT NULL,
                arguments TEXT NOT NULL, idempotency_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                timeout_seconds REAL NOT NULL, deadline REAL NOT NULL, created_at REAL NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed',
                    'cancel_requested','cancelled','timed_out','uncertain')),
                owner_id TEXT, claimed_at REAL, heartbeat REAL, launch_at REAL,
                finished_at REAL, result_artifact TEXT, stdout TEXT NOT NULL DEFAULT '',
                stderr TEXT NOT NULL DEFAULT '', stdout_truncated INTEGER NOT NULL DEFAULT 0,
                stderr_truncated INTEGER NOT NULL DEFAULT 0, progress_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(run_id,idempotency_key))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS execution_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES execution_jobs(id),
                kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS execution_terminal_immutable
                BEFORE UPDATE ON execution_jobs
                WHEN OLD.status IN ('succeeded','failed','cancelled','timed_out','uncertain')
                BEGIN SELECT RAISE(ABORT,'Terminal job is immutable'); END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS execution_intent_immutable
                BEFORE UPDATE OF session_id,run_id,tool_name,handler_ref,definition,definition_hash,
                    arguments,idempotency_key,fingerprint,timeout_seconds,deadline,created_at,workspace
                ON execution_jobs BEGIN SELECT RAISE(ABORT,'Job intent is immutable'); END""")

    def register(self, tool_spec, handler_ref):
        """Host-only: publish an importable trusted module:function and ToolSpec.

        Updating a registration changes future submissions, not existing snapshots.
        The host must bump ToolSpec.version when changing handler implementation.
        """
        if not isinstance(tool_spec, ToolSpec):
            raise TypeError("Job registration requires a ToolSpec")
        if not isinstance(handler_ref, str) or not _REFERENCE.fullmatch(handler_ref):
            raise ValueError("Handler must be a trusted module:function reference")
        definition = _definition(tool_spec)
        # Definitions are durable public metadata, not a credential transport.
        if self.store.redact(definition) != definition or self.store.redact(handler_ref) != handler_ref:
            raise ValueError("Secret-bearing handler metadata cannot be persisted")
        with self.store.transaction(write=True) as conn:
            conn.execute("""INSERT INTO execution_handlers VALUES(?,?,?,?)
                ON CONFLICT(tool_name) DO UPDATE SET handler_ref=excluded.handler_ref,
                definition=excluded.definition,definition_hash=excluded.definition_hash""",
                (tool_spec.name, handler_ref, canonical_json(definition), tool_spec.definition_hash))

    def _event(self, conn, job, kind, payload):
        value = self.store.redact({"job_id": job["id"], **payload})
        conn.execute("INSERT INTO execution_events(job_id,kind,payload,created_at) VALUES(?,?,?,?)",
                     (job["id"], kind, canonical_json(value), time.time()))
        return value

    def _mirror(self, job, kind, payload):
        # Atomic namespaced lifecycle journal is authoritative. Mirror to the
        # common trace only after releasing its writer lock.
        self.store.append_event(job["session_id"], kind, payload, job["run_id"])

    def submit(self, session_id, run_id, tool_name, arguments, idempotency_key, timeout_seconds=300):
        self.store.get_session(session_id)
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise ValueError("Run does not belong to session")
        if (not isinstance(idempotency_key, str) or not idempotency_key or len(idempotency_key) > 256
                or self.store.redact(idempotency_key) != idempotency_key):
            raise ValueError("Idempotency key must be bounded non-secret text")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Timeout must be finite and positive")
        if not isinstance(arguments, dict):
            raise ValueError("Job arguments must be a JSON object")
        argument_json = canonical_json(arguments)
        arguments = json.loads(argument_json)
        if canonical_json(self.store.redact(arguments)) != argument_json:
            raise ValueError("Secret-bearing arguments need host references; redacted values cannot be executed")
        canonical_json(idempotency_key)
        with self.store.transaction() as conn:
            binding = conn.execute("SELECT * FROM execution_handlers WHERE tool_name=?", (tool_name,)).fetchone()
            previous = conn.execute("SELECT * FROM execution_jobs WHERE run_id=? AND idempotency_key=?",
                                    (run_id, idempotency_key)).fetchone()
        if previous is not None and (previous["tool_name"] != tool_name
                or previous["arguments"] != argument_json or previous["timeout_seconds"] != timeout_seconds):
            raise ValueError("Idempotency conflict: identity already names different work")
        if binding is None:
            raise KeyError("No host-registered handler for tool")
        binding = dict(binding)
        definition = json.loads(binding["definition"])
        fingerprint = hashlib.sha256(canonical_json({"tool_name": tool_name, "arguments": arguments,
            "definition_hash": binding["definition_hash"], "handler_ref": binding["handler_ref"],
            "timeout_seconds": float(timeout_seconds)}).encode()).hexdigest()
        if previous is not None and previous["fingerprint"] != fingerprint:
            raise ValueError("Idempotency conflict: identity already names a different capability definition")
        try:
            validator(definition["input_schema"]).validate(arguments)
        except Exception as exc:
            raise ValueError("Job arguments violate the capability input schema") from exc
        now, identifier = time.time(), new_id("job")
        workspace = str(self.store.root / "jobs" / identifier / "work")
        created = False
        with self.store.transaction(write=True) as conn:
            existing = conn.execute("SELECT * FROM execution_jobs WHERE run_id=? AND idempotency_key=?",
                                    (run_id, idempotency_key)).fetchone()
            if existing is not None:
                job = _decode(existing)
                if job["fingerprint"] != fingerprint:
                    raise ValueError("Idempotency conflict: identity already names different work")
            else:
                run = conn.execute("SELECT session_id,status FROM runs WHERE id=?", (run_id,)).fetchone()
                if run["session_id"] != session_id:
                    raise ValueError("Run does not belong to session")
                if run["status"] in {"succeeded", "failed", "cancelled", "budget_exhausted"}:
                    raise PermissionError("A terminal run cannot submit new jobs")
                conn.execute("""INSERT INTO execution_jobs(id,session_id,run_id,tool_name,handler_ref,
                    definition,definition_hash,arguments,idempotency_key,fingerprint,timeout_seconds,
                    deadline,created_at,workspace,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (identifier, session_id, run_id, tool_name, binding["handler_ref"], binding["definition"],
                     binding["definition_hash"], argument_json, idempotency_key, fingerprint, timeout_seconds,
                     now + timeout_seconds, now, workspace, "queued"))
                job = _decode(conn.execute("SELECT * FROM execution_jobs WHERE id=?", (identifier,)).fetchone())
                payload = self._event(conn, job, "job.queued", {"tool_name": tool_name,
                    "definition_hash": job["definition_hash"], "deadline": job["deadline"]})
                created = True
        if created:
            self._mirror(job, "job.queued", payload)
        self._launch_if_unclaimed(job)
        return self.get(job["id"], session_id)

    def _read(self, job_id, session_id=None):
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM execution_jobs WHERE id=?", (job_id,)).fetchone()
        job = _decode(row)
        if session_id is not None and job["session_id"] != session_id:
            raise KeyError("Unknown job in session")
        return job

    def get(self, job_id, session_id=None):
        job = self._read(job_id, session_id)
        # Ownership is an internal compare-and-set guard, not model metadata.
        job.pop("owner_id", None)
        job.pop("fingerprint", None)
        return self.store.redact(job)

    def list(self, session_id=None):
        if session_id is not None:
            self.store.get_session(session_id)
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT id FROM execution_jobs" +
                (" WHERE session_id=?" if session_id is not None else "") + " ORDER BY created_at,id",
                (session_id,) if session_id is not None else ()).fetchall()
        return [self.get(row["id"], session_id) for row in rows]

    def events(self, job_id, session_id=None):
        """Read the authoritative atomic lifecycle journal in durable order."""
        self._read(job_id, session_id)
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT * FROM execution_events WHERE job_id=? ORDER BY seq", (job_id,)).fetchall()
        return self.store.redact([{**dict(row), "payload": json.loads(row["payload"])} for row in rows])

    def result(self, job_id, session_id=None):
        job = self._read(job_id, session_id)
        if job["status"] not in TERMINAL:
            return None
        artifact = job["result_artifact"]
        content = self.store.read_artifact(artifact["id"])
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ValueError("Job result integrity mismatch")
        return ToolResult(**self.store.redact(json.loads(content)))

    def cancel(self, job_id, session_id=None):
        job = self._read(job_id, session_id)
        with self.store.transaction(write=True) as conn:
            changed = conn.execute("UPDATE execution_jobs SET status='cancel_requested' WHERE id=? AND status IN ('queued','running')",
                                   (job_id,)).rowcount
            if changed:
                payload = self._event(conn, job, "job.cancel_requested", {})
            job = _decode(conn.execute("SELECT * FROM execution_jobs WHERE id=?", (job_id,)).fetchone())
        if changed:
            self._mirror(job, "job.cancel_requested", payload)
        self._launch_if_unclaimed(job)
        # Return the durable request snapshot, not an unearned acknowledgement.
        job.pop("owner_id", None)
        job.pop("fingerprint", None)
        return self.store.redact(job)

    def reconcile(self, job_id, session_id=None):
        job = self._read(job_id, session_id)
        if job["status"] in ACTIVE:
            if job["owner_id"] is None:
                self._launch_if_unclaimed(job)
            elif time.time() - job["heartbeat"] > HEARTBEAT_STALE_SECONDS:
                # No PID probing/signalling: missing heartbeat proves only that
                # outcome cannot be confirmed, not that the child has stopped.
                self._finish(job, "uncertain", failure("supervisor_lost",
                    "Supervisor heartbeat lost; execution may have effects and must not be replayed", "uncertain"),
                    expected_heartbeat=job["heartbeat"])
        return self.get(job_id, session_id)

    def _prepare_workspace(self, job):
        workspace = Path(job["workspace"])
        for directory in (self.store.root / "jobs", workspace.parent, workspace, workspace.parent / "control"):
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or directory.resolve() != directory:
                raise ValueError("Job workspace cannot traverse a symlink")
        return workspace

    def _launch_if_unclaimed(self, job):
        if job["status"] not in ACTIVE or job.get("owner_id") is not None:
            return
        now = time.time()
        if job["status"] == "cancel_requested" or self.store.get_run(job["run_id"])["status"] == "cancelled":
            self._finish(job, "cancelled", failure("cancelled", "Unclaimed job cancellation acknowledged", "cancelled"))
            return
        if now >= job["deadline"]:
            self._finish(job, "timed_out", failure("timed_out", "Unclaimed job deadline elapsed"))
            return
        with self.store.transaction(write=True) as conn:
            reserved = conn.execute("""UPDATE execution_jobs SET launch_at=? WHERE id=? AND owner_id IS NULL
                AND status IN ('queued','cancel_requested') AND (launch_at IS NULL OR launch_at<?)""",
                (now, job["id"], now - 1)).rowcount
        if not reserved:
            return
        try:
            workspace = self._prepare_workspace(job)
            process = subprocess.Popen(worker_command("supervise", self.store.root, job["id"]),
                cwd=workspace, env=worker_environment(workspace), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            # Reap when the UI remains alive; the daemon thread does not keep a
            # submitting process alive or make recovery depend on its memory.
            threading.Thread(target=process.wait, daemon=True).start()
        except (OSError, ValueError) as exc:
            payload = {"job_id": job["id"], "message": bounded_text(self.store, exc, 2000)}
            with self.store.transaction(write=True) as conn:
                self._event(conn, job, "job.launch_failed", payload)
            self._mirror(job, "job.launch_failed", payload)

    def _claim(self, job_id):
        owner, now = uuid4().hex, time.time()
        with self.store.transaction(write=True) as conn:
            changed = conn.execute("""UPDATE execution_jobs SET owner_id=?,claimed_at=?,heartbeat=?,
                status=CASE status WHEN 'queued' THEN 'running' ELSE status END
                WHERE id=? AND owner_id IS NULL AND status IN ('queued','cancel_requested')""",
                (owner, now, now, job_id)).rowcount
            if not changed:
                return None
            job = _decode(conn.execute("SELECT * FROM execution_jobs WHERE id=?", (job_id,)).fetchone())
            payload = self._event(conn, job, "job.claimed", {})
        self._mirror(job, "job.claimed", payload)
        return job

    def _heartbeat(self, job):
        with self.store.transaction(write=True) as conn:
            return bool(conn.execute("""UPDATE execution_jobs SET heartbeat=? WHERE id=? AND owner_id=?
                AND status IN ('running','cancel_requested')""", (time.time(), job["id"], job["owner_id"])).rowcount)

    def _finish(self, job, status, result, *, expected_heartbeat=None, logs=None):
        if status not in TERMINAL:
            raise ValueError("Completion requires a terminal job status")
        if job["owner_id"] is None and status not in {"cancelled", "timed_out"}:
            raise ValueError("Unclaimed work can only acknowledge cancellation or deadline")
        clean = self.store.redact(result.to_dict())
        module_name, handler_name = job["handler_ref"].split(":")
        try:
            finalizer = getattr(importlib.import_module(module_name), handler_name + "_finalize", None)
        except (ImportError, AttributeError):
            # Handler import failures are already represented by the worker's
            # terminal result. Recovery adapters may reconcile auxiliary state.
            finalizer = None
        if finalizer is not None and not callable(finalizer):
            raise TypeError("Trusted job finalizer must be callable")
        artifact = self.store.put_artifact(canonical_json(clean), "job-result.json", "application/json",
            {"job_id": job["id"], "definition_hash": job["definition_hash"], "role": "job_result"},
            session_id=job["session_id"], run_id=job["run_id"])
        logs = logs or {"stdout": "", "stderr": "", "stdout_truncated": False, "stderr_truncated": False}
        with self.store.transaction(write=True) as conn:
            values = [status, time.time(), canonical_json(artifact), logs["stdout"], logs["stderr"],
                      logs["stdout_truncated"], logs["stderr_truncated"], job["id"]]
            if job["owner_id"] is None:
                # A launcher racing this close either claims first or sees the
                # terminal row. There can never be an acknowledged live child.
                guard = " AND owner_id IS NULL AND status IN ('queued','cancel_requested')"
                if status == "cancelled":
                    guard += " AND (status='cancel_requested' OR EXISTS (SELECT 1 FROM runs WHERE runs.id=execution_jobs.run_id AND runs.status='cancelled'))"
                else:
                    guard += " AND deadline<=?"
                    values.append(time.time())
            else:
                guard = " AND owner_id=? AND status IN ('running','cancel_requested')"
                values.append(job["owner_id"])
            if expected_heartbeat is not None:
                guard += " AND heartbeat=?"
                values.append(expected_heartbeat)
            changed = conn.execute("""UPDATE execution_jobs SET status=?,finished_at=?,result_artifact=?,
                stdout=?,stderr=?,stdout_truncated=?,stderr_truncated=? WHERE id=?""" + guard, values).rowcount
            if changed:
                if finalizer is not None:
                    finalizer(conn, self.store, job, status, result)
                payload = self._event(conn, job, "job." + status, {"result_artifact": artifact,
                    "stdout": logs["stdout"], "stderr": logs["stderr"],
                    "stdout_truncated": logs["stdout_truncated"], "stderr_truncated": logs["stderr_truncated"]})
        if changed:
            self._mirror(job, "job." + status, payload)
        return bool(changed)


def validate_result(store, job, result):
    """Validate terminal envelopes, finite JSON, schema, and actual scoped blobs."""
    try:
        if (not isinstance(result, ToolResult) or result.status not in {"ok", "error", "denied", "cancelled", "uncertain"}
                or not isinstance(result.artifacts, list) or any(not isinstance(item, dict) for item in result.artifacts)
                or result.job_id not in (None, job["id"])
                or (result.status == "ok" and result.error is not None)
                or (result.status != "ok" and result.error is None)
                or (result.error is not None and (not isinstance(result.error, dict)
                    or not isinstance(result.error.get("code"), str) or not isinstance(result.error.get("message"), str)))):
            return failure("output_contract", "Handler must return a valid terminal ToolResult; nested pending jobs are not supported")
    except Exception:
        return failure("output_contract", "Handler result envelope could not be inspected")
    try:
        content = canonical_json(result.to_dict())
        if len(content.encode()) > MAX_RESULT_BYTES:
            raise ValueError("Result exceeds bounded JSON envelope")
        result = ToolResult(**json.loads(content))
    except Exception:
        return failure("output_json", "Result must be bounded finite UTF-8 JSON")
    if result.status == "ok" and job["definition"]["output_schema"] is not None:
        try:
            validator(job["definition"]["output_schema"]).validate(result.data)
        except Exception:
            return failure("output_schema", "Result data violates the capability output schema")
    try:
        artifacts = {item["id"]: item for item in store.artifacts(job["session_id"], job["run_id"])}
        actual = []
        for claimed in result.artifacts:
            item = artifacts[claimed["id"]]
            if claimed.get("sha256", item["sha256"]) != item["sha256"]:
                raise ValueError("Artifact hash does not match")
            store.read_artifact(item["id"])
            actual.append(item)
        result.artifacts = actual
    except Exception:
        return failure("output_artifact", "Result must reference intact artifacts belonging to this session and run")
    try:
        # Artifact IDs expand to authoritative metadata, and redaction markers
        # can be longer than their inputs. Bound the actual outgoing envelope,
        # not just the handler's pre-normalization claim.
        final_json = canonical_json(store.redact(result.to_dict()))
        if len(final_json.encode()) > MAX_RESULT_BYTES:
            raise ValueError("Normalized result exceeds bounded JSON envelope")
        cleaned = ToolResult(**json.loads(final_json))
    except Exception:
        return failure("output_json", "Normalized result must be bounded finite UTF-8 JSON")
    if cleaned.status == "ok" and job["definition"]["output_schema"] is not None:
        try:
            validator(job["definition"]["output_schema"]).validate(cleaned.data)
        except Exception:
            return failure("output_schema", "Redacted result no longer satisfies the output schema")
    return cleaned
