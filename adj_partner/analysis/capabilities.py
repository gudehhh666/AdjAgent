"""Host-owned catalog adapters for sandboxed project analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..contracts import ToolResult
from ..execution import JobManager
from ..execution.manager import TERMINAL
from ..native.safe_files import read_input
from ..tools import ToolFailure, ToolSpec
from ..tools.catalog import canonical_json, validator
from .history import AnalysisHistory, LANGUAGES, validate_method_parameters
from .worker import MAX_INPUT_BYTES, MAX_INPUT_FILES, MAX_SOURCE_BYTES, MAX_TOTAL_INPUT_BYTES


ANALYSIS_HANDLER = "adj_partner.analysis.worker:analysis_job"
ANALYSIS_VERSION = "2"
ANALYSIS_JOB_TOOLS = frozenset({"code_interpreter", "bash", "analysis_method_run"})
ANALYSIS_TOOLS = (
    "code_interpreter", "bash", "analysis_method_register",
    "analysis_method_run", "analysis_history", "analysis_history_read",
)

PATH = {"type": "string", "minLength": 1, "maxLength": 4096}
ID = {"type": "string", "minLength": 1, "maxLength": 128,
      "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"}
TITLE = {"type": "string", "minLength": 1, "maxLength": 500}
INPUTS = {"type": "array", "items": PATH, "maxItems": MAX_INPUT_FILES}
TIMEOUT = {"type": "integer", "minimum": 1, "maximum": 300}
PARAMETERS = {"type": "object"}


def schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _job_schema():
    manifest = schema({"path": PATH, "name": {"type": "string", "minLength": 1, "maxLength": 255},
        "size": {"type": "integer", "minimum": 0, "maximum": MAX_INPUT_BYTES},
        "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
        ["path", "name", "size", "sha256"])
    return schema({
        "analysis_id": ID, "language": {"enum": sorted(LANGUAGES)}, "code_artifact_id": ID,
        "inputs": {"type": "array", "items": manifest, "maxItems": MAX_INPUT_FILES},
        "parameters": PARAMETERS, "timeout_seconds": TIMEOUT, "method_id": ID,
    }, ["analysis_id", "language", "code_artifact_id", "inputs", "parameters", "timeout_seconds"])


class AnalysisCapabilities:
    def __init__(self, store, *, jobs=None):
        self.store = store
        if jobs is not None and (jobs.store.root != store.root or jobs.store.db_path != store.db_path):
            raise ValueError("Analysis jobs must share the same Store root/database")
        self.jobs = jobs if jobs is not None and jobs.store is store else JobManager(store)
        self.history = AnalysisHistory(store)

    @staticmethod
    def definitions():
        workspace = ("The script runs without network in an OS sandbox. Read only copied files in inputs/, "
            "write deliverables beneath outputs/, and optionally write outputs/analysis_result.json with "
            "an object summary plus optional metrics and limitations. This is local containment, not a "
            "defense against kernel or malicious native-library vulnerabilities.")
        return [
            ToolSpec("code_interpreter", "Run model-authored Python as a durable, auditable analysis job. " + workspace,
                schema({"code": {"type": "string", "minLength": 1, "maxLength": MAX_SOURCE_BYTES},
                    "inputs": INPUTS, "title": TITLE, "timeout_seconds": TIMEOUT},
                    ["code", "title"]), version=ANALYSIS_VERSION),
            ToolSpec("bash", "Run Bash via /bin/bash --noprofile --norc as a durable analysis job. "
                "Execution requires Linux Bubblewrap; macOS returns sandbox_unavailable because Seatbelt "
                "cannot safely contain forking Bash workloads. " + workspace,
                schema({"script": {"type": "string", "minLength": 1, "maxLength": MAX_SOURCE_BYTES},
                    "inputs": INPUTS, "title": TITLE, "timeout_seconds": TIMEOUT},
                    ["script", "title"]), version=ANALYSIS_VERSION),
            ToolSpec("analysis_method_register", "Promote one successful project analysis into a new immutable, versioned method.",
                schema({"record_id": ID, "name": {"type": "string", "minLength": 1, "maxLength": 64,
                            "pattern": "^[a-z][a-z0-9_]*$"},
                        "description": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "parameter_schema": {"type": "object"}},
                       ["record_id", "name", "description", "parameter_schema"]), version=ANALYSIS_VERSION),
            ToolSpec("analysis_method_run", "Run an immutable registered analysis method with newly authorized inputs and schema-validated parameters.",
                schema({"method_id": ID, "parameters": PARAMETERS, "inputs": INPUTS,
                        "title": TITLE, "timeout_seconds": TIMEOUT},
                       ["method_id", "parameters", "title"]), version=ANALYSIS_VERSION),
            ToolSpec("analysis_history", "Search immutable analysis records and registered methods across this project Store.",
                schema({"query": {"type": "string", "maxLength": 500},
                        "language": {"enum": sorted(LANGUAGES)},
                        "status": {"enum": ["pending", "running", "succeeded", "failed", "cancelled", "timed_out", "uncertain"]},
                        "method_name": {"type": "string", "minLength": 1, "maxLength": 64,
                                        "pattern": "^[a-z][a-z0-9_]*$"},
                        "created_after": {"type": "string", "minLength": 1, "maxLength": 64},
                        "created_before": {"type": "string", "minLength": 1, "maxLength": 64},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
                version=ANALYSIS_VERSION),
            ToolSpec("analysis_history_read", "Read one bounded text page from an artifact linked to project analysis history.",
                schema({"artifact_id": ID, "start": {"type": "integer", "minimum": 0},
                        "max_characters": {"type": "integer", "minimum": 1, "maximum": 16000}},
                       ["artifact_id"]), version=ANALYSIS_VERSION),
        ]

    def register(self, catalog):
        definitions = self.definitions()
        public = {spec.name: spec for spec in definitions}
        internal_schema = _job_schema()
        for name in ANALYSIS_JOB_TOOLS:
            self.jobs.register(ToolSpec(name, public[name].description, internal_schema,
                                        version=public[name].version), ANALYSIS_HANDLER)
        handlers = {
            "code_interpreter": self._python,
            "bash": self._bash,
            "analysis_method_register": self._register_method,
            "analysis_method_run": self._run_method,
            "analysis_history": self._history,
            "analysis_history_read": self._history_read,
        }
        catalog.register_many((spec, handlers[spec.name]) for spec in definitions)

    def _bind(self, context):
        context.check_cancelled()
        if context.store.root != self.store.root or context.store.db_path != self.store.db_path:
            raise ToolFailure("analysis_store_mismatch", "Analysis tools are bound to another Store", "denied")
        if context.store is self.store:
            return self
        return type(self)(context.store)

    @staticmethod
    def _manifests(context, inputs):
        if inputs is None:
            inputs = []
        manifests, names, total = [], set(), 0
        for path in inputs:
            try:
                item = read_input(context, path, maximum_bytes=MAX_INPUT_BYTES)
            except PermissionError as exc:
                raise ToolFailure("analysis_input_denied", str(exc), "denied") from exc
            except (OSError, ValueError) as exc:
                raise ToolFailure("analysis_input_denied", str(exc), "denied") from exc
            name = Path(item.path).name
            if name in names:
                raise ToolFailure("analysis_input_denied",
                    "Analysis input basenames must be unique", "denied")
            names.add(name)
            total += item.size
            if total > MAX_TOTAL_INPUT_BYTES:
                raise ToolFailure("analysis_input_denied",
                    "Analysis inputs exceed the total byte limit", "denied")
            manifests.append({"path": item.path, "name": name, "size": item.size,
                              "sha256": item.sha256})
        return manifests

    @staticmethod
    def _code_artifact(context, source, language, *, method_id=None, expected_hash=None):
        content = source.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        if len(content) > MAX_SOURCE_BYTES:
            raise ToolFailure("analysis_source_invalid", "Analysis source exceeds the byte limit")
        if expected_hash is not None and digest != expected_hash:
            raise ToolFailure("analysis_source_invalid", "Registered analysis method code failed integrity validation")
        suffix, media = ("py", "text/x-python") if language == "python" else ("sh", "text/x-shellscript")
        return context.store.put_artifact(content, f"analysis.{suffix}", media,
            {"kind": "model_authored_analysis", "language": language, "method_id": method_id,
             "sha256": digest}, session_id=context.session_id, run_id=context.run_id)

    def _start(self, context, *, language, source, inputs, title, timeout_seconds,
               parameters, method_id=None, expected_hash=None):
        bound = self._bind(context)
        if not context.call_id:
            raise ToolFailure("call_identity_required", "Durable analysis requires its original tool call ID")
        manifests = self._manifests(context, inputs)
        artifact = self._code_artifact(context, source, language,
            method_id=method_id, expected_hash=expected_hash)
        record = bound.history.create_record(context.session_id, context.run_id,
            language=language, title=title, code_artifact_id=artifact["id"], inputs=manifests,
            parameters=parameters, limits={"timeout_seconds": timeout_seconds}, method_id=method_id)
        arguments = {"analysis_id": record["id"], "language": language,
            "code_artifact_id": artifact["id"], "inputs": manifests,
            "parameters": parameters, "timeout_seconds": timeout_seconds}
        if method_id is not None:
            arguments["method_id"] = method_id
        tool_name = "analysis_method_run" if method_id is not None else (
            "code_interpreter" if language == "python" else "bash")
        try:
            job = bound.jobs.submit(context.session_id, context.run_id,
                tool_name,
                arguments, context.call_id, timeout_seconds=timeout_seconds + 10)
            bound.history.attach_job(record["id"], job["id"])
        except Exception as exc:
            # submit() may have durably inserted the intent before a launcher or
            # response-path failure. Recover that exact row instead of replaying.
            job = bound._find_submitted_job(context.session_id, context.run_id,
                context.call_id, tool_name, arguments, timeout_seconds + 10)
            if job is not None:
                try:
                    bound.history.attach_job(record["id"], job["id"])
                except Exception as attach_exc:
                    error = {"code": "analysis_submission_uncertain",
                             "message": "The durable analysis intent exists but could not be attached to history"}
                    bound.history.terminalize_if_active(record["id"], status="uncertain",
                        sandbox_backend="outer_job_supervisor", error=error)
                    raise ToolFailure(error["code"], error["message"], "uncertain") from attach_exc
            else:
                error = {"code": "analysis_submission_failed",
                         "message": "The analysis job could not be submitted durably"}
                bound.history.terminalize_if_active(record["id"], status="failed",
                    sandbox_backend="not_started", error=error)
                raise ToolFailure(error["code"], error["message"]) from exc
        return ToolResult(status="pending", data={"analysis_id": record["id"],
            "job_id": job["id"], "status": job["status"]}, job_id=job["id"])

    def _find_submitted_job(self, session_id, run_id, idempotency_key, tool_name,
                            arguments, timeout_seconds):
        with self.store.transaction() as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_jobs'").fetchone()
            if exists is None:
                return None
            row = conn.execute("""SELECT * FROM execution_jobs
                WHERE session_id=? AND run_id=? AND idempotency_key=?""",
                (session_id, run_id, idempotency_key)).fetchone()
        if row is None:
            return None
        value = dict(row)
        if (value["tool_name"] != tool_name or value["handler_ref"] != ANALYSIS_HANDLER
                or value["arguments"] != canonical_json(arguments)
                or float(value["timeout_seconds"]) != float(timeout_seconds)):
            raise ValueError("Durable analysis submission identity conflicts with the requested action")
        return {"id": value["id"], "status": value["status"]}

    def _python(self, context, code, title, inputs=None, timeout_seconds=120):
        return self._start(context, language="python", source=code, inputs=inputs or [],
            title=title, timeout_seconds=timeout_seconds, parameters={})

    def _bash(self, context, script, title, inputs=None, timeout_seconds=120):
        return self._start(context, language="bash", source=script, inputs=inputs or [],
            title=title, timeout_seconds=timeout_seconds, parameters={})

    def _register_method(self, context, record_id, name, description, parameter_schema):
        bound = self._bind(context)
        try:
            method = bound.history.register_method(context.session_id, record_id,
                name=name, description=description, parameter_schema=parameter_schema)
        except KeyError as exc:
            raise ToolFailure("analysis_method_not_found", str(exc)) from exc
        except ValueError as exc:
            raise ToolFailure("analysis_method_schema", str(exc)) from exc
        return ToolResult(data={"method": method})

    def _run_method(self, context, method_id, parameters, title, inputs=None, timeout_seconds=120):
        bound = self._bind(context)
        try:
            method = bound.history.get_method(method_id)
        except KeyError as exc:
            raise ToolFailure("analysis_method_not_found", "Unknown analysis method") from exc
        try:
            validate_method_parameters(parameters)
            validator(method["parameter_schema"]).validate(parameters)
        except Exception as exc:
            raise ToolFailure("analysis_method_schema",
                f"Parameters violate the registered method schema: {exc}") from exc
        try:
            source = context.store.read_artifact(method["code_artifact_id"], max_bytes=MAX_SOURCE_BYTES).decode("utf-8")
        except (KeyError, OSError, UnicodeDecodeError, ValueError) as exc:
            raise ToolFailure("analysis_source_invalid", "Registered analysis method code is unavailable") from exc
        return bound._start(context, language=method["language"], source=source,
            inputs=inputs or [], title=title, timeout_seconds=timeout_seconds,
            parameters=parameters, method_id=method_id, expected_hash=method["code_sha256"])

    def _history(self, context, query="", language=None, status=None, method_name=None,
                 created_after=None, created_before=None, limit=20):
        bound = self._bind(context)
        try:
            result = bound.history.search(context.session_id, query=query,
                language=language, status=status, method_name=method_name,
                created_after=created_after, created_before=created_before, limit=limit)
        except (PermissionError, ValueError) as exc:
            raise ToolFailure("analysis_history_denied", str(exc), "denied") from exc
        return ToolResult(data=result)

    def _history_read(self, context, artifact_id, start=0, max_characters=8000):
        bound = self._bind(context)
        try:
            result = bound.history.read_text(context.session_id, artifact_id,
                start=start, max_characters=max_characters)
        except (KeyError, PermissionError, ValueError) as exc:
            raise ToolFailure("analysis_history_denied", str(exc), "denied") from exc
        return ToolResult(data=result)

    def _scope(self, session_id, run_id):
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise PermissionError("Run does not belong to session")
        self.store.get_session(session_id)

    def resolve_job(self, session_id, run_id, job_id):
        self._scope(session_id, run_id)
        job = self.jobs.get(job_id, session_id)
        if (job["run_id"] != run_id or job["tool_name"] not in ANALYSIS_JOB_TOOLS
                or job["handler_ref"] != ANALYSIS_HANDLER):
            raise PermissionError("Job is not this run's analysis capability")
        job = self.jobs.reconcile(job_id, session_id)
        if job["status"] not in TERMINAL:
            return ToolResult(status="pending", data={"job_id": job_id, "status": job["status"]}, job_id=job_id)
        result = self.jobs.result(job_id, session_id)
        analysis_id = job.get("arguments", {}).get("analysis_id")
        if isinstance(analysis_id, str):
            current = self.history.get_record(analysis_id)
            if current["status"] not in TERMINAL:
                if job["status"] == "succeeded":
                    error = {"code": "analysis_history_incomplete",
                             "message": "The analysis job succeeded without an atomic history outcome"}
                    self.history.terminalize_if_active(analysis_id, status="uncertain",
                        sandbox_backend="outer_job_supervisor", error=error)
                    return ToolResult(status="uncertain", error=error, job_id=job_id)
                error = {"code": job["status"],
                         "message": f"The outer analysis job became {job['status']}"}
                self.history.terminalize_if_active(analysis_id, status=job["status"],
                    sandbox_backend="outer_job_supervisor", error=error)
        return result

    def recover_action(self, session_id, run_id, call):
        self._scope(session_id, run_id)
        if call.name not in ANALYSIS_JOB_TOOLS:
            return None
        action = self.store.get_action(run_id, call.id)
        if action is not None and (action["tool_name"] != call.name
                or canonical_json(action["args"]) != canonical_json(call.arguments)):
            raise ValueError("Existing analysis action identity conflicts with proposed action")
        with self.store.transaction() as conn:
            row = conn.execute("""SELECT id,tool_name,handler_ref FROM execution_jobs
                WHERE session_id=? AND run_id=? AND idempotency_key=?""",
                (session_id, run_id, call.id)).fetchone()
        if row is None:
            return None
        if row["tool_name"] != call.name or row["handler_ref"] != ANALYSIS_HANDLER:
            raise ValueError("Existing analysis job identity conflicts with proposed action")
        return self.resolve_job(session_id, run_id, row["id"])
