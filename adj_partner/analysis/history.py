"""Immutable, project-wide metadata for sandboxed analysis runs and methods."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re

from ..contracts import new_id
from ..tools.catalog import canonical_json, validate_schema


TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "uncertain"})
LANGUAGES = frozenset({"python", "bash"})
TEXT_MEDIA_TYPES = frozenset({
    "application/json", "application/x-ndjson", "application/xml",
    "text/csv", "text/markdown", "text/plain", "text/tab-separated-values",
    "text/x-python", "text/x-shellscript",
})
MAX_HISTORY_ARTIFACT = 8 * 1024 * 1024
MAX_HISTORY_PAGE_BYTES = 6000
MAX_METHOD_SCHEMA_BYTES = 16 * 1024
MAX_METHOD_PARAMETERS_BYTES = 64 * 1024
MAX_METHOD_SCHEMA_NODES = 512
MAX_METHOD_SCHEMA_DEPTH = 12

_SAFE_SCHEMA_KEYWORDS = frozenset({
    "$schema", "additionalProperties", "default", "description", "enum", "exclusiveMaximum",
    "exclusiveMinimum", "items", "maxItems", "maxLength", "maxProperties", "maximum", "minItems",
    "minLength", "minProperties", "minimum", "multipleOf", "properties", "required", "title", "type",
    "uniqueItems",
})


SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_records (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    run_id TEXT NOT NULL REFERENCES runs(id),
    job_id TEXT,
    language TEXT NOT NULL CHECK(language IN ('python','bash')),
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed','cancelled','timed_out','uncertain')),
    code_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    inputs TEXT NOT NULL,
    parameters TEXT NOT NULL,
    limits TEXT NOT NULL,
    method_id TEXT,
    sandbox_backend TEXT,
    summary TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS analysis_records_created ON analysis_records(created_at, id);
CREATE INDEX IF NOT EXISTS analysis_records_title ON analysis_records(title);
CREATE TABLE IF NOT EXISTS analysis_record_artifacts (
    record_id TEXT NOT NULL REFERENCES analysis_records(id),
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    PRIMARY KEY(record_id, ordinal),
    UNIQUE(record_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS analysis_record_artifact_lookup ON analysis_record_artifacts(artifact_id);
CREATE TABLE IF NOT EXISTS analysis_methods (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    description TEXT NOT NULL,
    language TEXT NOT NULL CHECK(language IN ('python','bash')),
    code_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    code_sha256 TEXT NOT NULL,
    parameter_schema TEXT NOT NULL,
    source_record_id TEXT NOT NULL REFERENCES analysis_records(id),
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);
CREATE INDEX IF NOT EXISTS analysis_methods_name ON analysis_methods(name, version);
CREATE TABLE IF NOT EXISTS analysis_schema_meta (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
INSERT OR IGNORE INTO analysis_schema_meta(component,version) VALUES('analysis',1);
CREATE TRIGGER IF NOT EXISTS analysis_terminal_immutable
BEFORE UPDATE ON analysis_records
WHEN OLD.status IN ('succeeded','failed','cancelled','timed_out','uncertain')
BEGIN SELECT RAISE(ABORT, 'Terminal analysis record is immutable'); END;
CREATE TRIGGER IF NOT EXISTS analysis_method_immutable
BEFORE UPDATE ON analysis_methods
BEGIN SELECT RAISE(ABORT, 'Analysis methods are immutable'); END;
CREATE TRIGGER IF NOT EXISTS analysis_terminal_no_delete
BEFORE DELETE ON analysis_records
WHEN OLD.status IN ('succeeded','failed','cancelled','timed_out','uncertain')
BEGIN SELECT RAISE(ABORT, 'Terminal analysis record cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS analysis_terminal_artifact_no_insert
BEFORE INSERT ON analysis_record_artifacts
WHEN (SELECT status FROM analysis_records WHERE id=NEW.record_id)
    IN ('succeeded','failed','cancelled','timed_out','uncertain')
BEGIN SELECT RAISE(ABORT, 'Terminal analysis artifacts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS analysis_terminal_artifact_no_update
BEFORE UPDATE ON analysis_record_artifacts
WHEN (SELECT status FROM analysis_records WHERE id=OLD.record_id)
    IN ('succeeded','failed','cancelled','timed_out','uncertain')
BEGIN SELECT RAISE(ABORT, 'Terminal analysis artifacts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS analysis_terminal_artifact_no_delete
BEFORE DELETE ON analysis_record_artifacts
WHEN (SELECT status FROM analysis_records WHERE id=OLD.record_id)
    IN ('succeeded','failed','cancelled','timed_out','uncertain')
BEGIN SELECT RAISE(ABORT, 'Terminal analysis artifacts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS analysis_method_no_delete
BEFORE DELETE ON analysis_methods
BEGIN SELECT RAISE(ABORT, 'Analysis methods are immutable'); END;
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.loads(value) if value is not None else None


def _record(row):
    if row is None:
        raise KeyError("Unknown analysis record")
    value = dict(row)
    for key in ("inputs", "parameters", "limits", "summary", "error"):
        value[key] = _json(value[key])
    return value


def _method(row):
    if row is None:
        raise KeyError("Unknown analysis method")
    value = dict(row)
    value["parameter_schema"] = json.loads(value["parameter_schema"])
    return value


def _bounded_tree(value, *, maximum_bytes, maximum_nodes, maximum_depth, label):
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    nodes = 0

    def walk(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > maximum_nodes or depth > maximum_depth:
            raise ValueError(f"{label} is too complex")
        if isinstance(item, dict):
            for key, nested in item.items():
                walk(key, depth + 1)
                walk(nested, depth + 1)
        elif isinstance(item, list):
            for nested in item:
                walk(nested, depth + 1)

    walk(value, 0)
    return encoded


def validate_method_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Analysis method parameter schema must be an object")
    _bounded_tree(schema, maximum_bytes=MAX_METHOD_SCHEMA_BYTES,
                  maximum_nodes=MAX_METHOD_SCHEMA_NODES,
                  maximum_depth=MAX_METHOD_SCHEMA_DEPTH,
                  label="Analysis method parameter schema")

    def walk(value, *, property_map=False):
        if isinstance(value, dict):
            if not property_map:
                unsupported = set(value) - _SAFE_SCHEMA_KEYWORDS
                if unsupported:
                    if "$ref" in unsupported:
                        raise ValueError("Only a bounded local JSON-schema subset without references is allowed")
                    raise ValueError("Analysis method schema uses unsupported keywords: " +
                                     ", ".join(sorted(unsupported)))
            for key, nested in value.items():
                if property_map:
                    walk(nested)
                elif key == "properties":
                    if not isinstance(nested, dict):
                        raise ValueError("Analysis method schema properties must be an object")
                    walk(nested, property_map=True)
                elif key == "items":
                    if not isinstance(nested, dict):
                        raise ValueError("Analysis method schema items must be one object schema")
                    walk(nested)
                elif key == "additionalProperties":
                    if not isinstance(nested, (bool, dict)):
                        raise ValueError("Analysis method additionalProperties must be boolean or an object schema")
                    if isinstance(nested, dict):
                        walk(nested)
        elif property_map:
            raise ValueError("Analysis method schema properties must be an object")

    walk(schema)
    validate_schema(schema)


def validate_method_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("Analysis method parameters must be an object")
    _bounded_tree(parameters, maximum_bytes=MAX_METHOD_PARAMETERS_BYTES,
                  maximum_nodes=2048, maximum_depth=24,
                  label="Analysis method parameters")


def finalize_analysis_job(conn, store, job, status, result):
    """Finalize analysis history in the same SQLite transaction as its job."""
    analysis_id = job.get("arguments", {}).get("analysis_id")
    if not isinstance(analysis_id, str):
        raise ValueError("Analysis job is missing its history identifier")
    row = conn.execute("SELECT * FROM analysis_records WHERE id=?", (analysis_id,)).fetchone()
    current = _record(row)
    if current["job_id"] not in {None, job["id"]}:
        raise ValueError("Analysis history is attached to another job")
    if current["status"] in TERMINAL:
        if current["status"] != status:
            raise ValueError("Analysis history and job terminal states conflict")
        return
    if status not in TERMINAL:
        raise ValueError("Analysis job finalizer requires a terminal state")

    data = result.data if isinstance(result.data, dict) else {}
    if data.get("analysis_id", analysis_id) != analysis_id:
        raise ValueError("Analysis result identifies another history record")
    if status == "succeeded" and result.status != "ok":
        raise ValueError("A successful analysis job requires an ok result")
    summary = data.get("summary", {}) if status == "succeeded" else {}
    if not isinstance(summary, dict):
        raise ValueError("Analysis result summary is malformed")
    error = None if status == "succeeded" else result.error
    if status != "succeeded" and (error is None or not isinstance(error, dict)):
        raise ValueError("Terminal analysis failure requires an error")
    backend = data.get("sandbox_backend") or current["sandbox_backend"] or "outer_job_supervisor"
    if not isinstance(backend, str) or not backend:
        raise ValueError("Analysis sandbox backend is malformed")

    ordinal = conn.execute("SELECT COALESCE(MAX(ordinal),-1) FROM analysis_record_artifacts WHERE record_id=?",
                           (analysis_id,)).fetchone()[0] + 1
    seen = set()
    if len(result.artifacts) > 40:
        raise ValueError("Analysis result references too many artifacts")
    for claimed in result.artifacts:
        artifact_id = claimed.get("id") if isinstance(claimed, dict) else None
        if not isinstance(artifact_id, str) or artifact_id in seen:
            raise ValueError("Analysis result artifact list is malformed")
        seen.add(artifact_id)
        artifact = conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if artifact is None or artifact["session_id"] != current["session_id"] or artifact["run_id"] != current["run_id"]:
            raise PermissionError("Analysis result artifact is outside its originating run")
        provenance = json.loads(artifact["provenance"])
        role = provenance.get("role")
        if provenance.get("analysis_id") != analysis_id or not isinstance(role, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", role):
            raise ValueError("Analysis result artifact provenance is malformed")
        conn.execute("INSERT INTO analysis_record_artifacts VALUES(?,?,?,?)",
                     (analysis_id, ordinal, role, artifact_id))
        ordinal += 1
    conn.execute("""UPDATE analysis_records SET job_id=?,status=?,sandbox_backend=?,summary=?,error=?,finished_at=?
        WHERE id=? AND status IN ('pending','running')""",
        (job["id"], status, store.redact(backend), canonical_json(store.redact(summary)),
         canonical_json(store.redact(error)), _now(), analysis_id))


class AnalysisHistory:
    def __init__(self, store):
        self.store = store
        with store.transaction(write=True) as conn:
            conn.executescript(SCHEMA)
            version = conn.execute("SELECT version FROM analysis_schema_meta WHERE component='analysis'").fetchone()[0]
            if version != 1:
                raise ValueError(f"Unsupported analysis schema version: {version}")

    def _artifact(self, conn, identifier):
        row = conn.execute("SELECT * FROM artifacts WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Unknown analysis artifact")
        return dict(row)

    def create_record(self, session_id, run_id, *, language, title, code_artifact_id,
                      inputs, parameters, limits, method_id=None, record_id=None):
        self.store.get_session(session_id)
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise ValueError("Run does not belong to analysis session")
        if language not in LANGUAGES:
            raise ValueError("Analysis language must be python or bash")
        if not isinstance(title, str) or not 1 <= len(title) <= 500:
            raise ValueError("Analysis title must contain 1..500 characters")
        for name, value in (("inputs", inputs), ("parameters", parameters), ("limits", limits)):
            if name == "inputs" and not isinstance(value, list):
                raise ValueError("Analysis inputs must be a list")
            if name != "inputs" and not isinstance(value, dict):
                raise ValueError(f"Analysis {name} must be an object")
            canonical_json(value)
        identifier = record_id or new_id("analysis")
        with self.store.transaction(write=True) as conn:
            artifact = self._artifact(conn, code_artifact_id)
            if artifact["session_id"] != session_id or artifact["run_id"] != run_id:
                raise PermissionError("Analysis code artifact must belong to its originating run")
            if method_id is not None and conn.execute(
                    "SELECT 1 FROM analysis_methods WHERE id=?", (method_id,)).fetchone() is None:
                raise KeyError("Unknown analysis method")
            conn.execute("""INSERT INTO analysis_records(
                id,session_id,run_id,job_id,language,title,status,code_artifact_id,
                inputs,parameters,limits,method_id,sandbox_backend,summary,error,created_at,finished_at)
                VALUES(?,?,?,NULL,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,NULL)""",
                (identifier, session_id, run_id, language, self.store.redact(title), "pending",
                 code_artifact_id, canonical_json(self.store.redact(inputs)),
                 canonical_json(self.store.redact(parameters)), canonical_json(limits), method_id, _now()))
            conn.execute("INSERT INTO analysis_record_artifacts VALUES(?,?,?,?)",
                         (identifier, 0, "code", code_artifact_id))
            return _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (identifier,)).fetchone())

    def attach_job(self, record_id, job_id):
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Analysis job ID is required")
        with self.store.transaction(write=True) as conn:
            current = _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            # The trusted worker may finish and atomically attach the job before
            # submit() returns to the caller. Re-attaching that same durable job
            # is therefore an idempotent success, even after terminalization.
            if current["status"] in TERMINAL and current["job_id"] == job_id:
                return current
            if current["status"] not in {"pending", "running"} or current["job_id"] not in {None, job_id}:
                raise ValueError("Analysis record cannot attach this job")
            conn.execute("UPDATE analysis_records SET job_id=? WHERE id=?", (job_id, record_id))
            return _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())

    def mark_running(self, record_id, job_id):
        with self.store.transaction(write=True) as conn:
            current = _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            if current["status"] not in {"pending", "running"} or current["job_id"] not in {None, job_id}:
                raise ValueError("Analysis record cannot start this job")
            conn.execute("UPDATE analysis_records SET job_id=?,status='running' WHERE id=?", (job_id, record_id))
            return _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())

    def finish(self, record_id, *, status, summary, artifacts, sandbox_backend, error=None):
        if status not in TERMINAL:
            raise ValueError("Analysis finish status must be terminal")
        if not isinstance(summary, dict) or not isinstance(artifacts, list):
            raise ValueError("Analysis summary and artifacts are malformed")
        if error is not None and not isinstance(error, dict):
            raise ValueError("Analysis error must be an object")
        canonical_json(summary)
        canonical_json(error)
        if not isinstance(sandbox_backend, str) or not sandbox_backend:
            raise ValueError("Analysis sandbox backend is required")
        with self.store.transaction(write=True) as conn:
            current = _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            if current["status"] in TERMINAL:
                raise ValueError("Analysis record is already terminal")
            ordinal = conn.execute("SELECT COALESCE(MAX(ordinal),-1) FROM analysis_record_artifacts WHERE record_id=?",
                                   (record_id,)).fetchone()[0] + 1
            for role, artifact_id in artifacts:
                if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role):
                    raise ValueError("Invalid analysis artifact role")
                artifact = self._artifact(conn, artifact_id)
                if artifact["session_id"] != current["session_id"] or artifact["run_id"] != current["run_id"]:
                    raise PermissionError("Analysis output artifact must belong to the originating run")
                conn.execute("INSERT INTO analysis_record_artifacts VALUES(?,?,?,?)",
                             (record_id, ordinal, role, artifact_id))
                ordinal += 1
            conn.execute("""UPDATE analysis_records SET status=?,sandbox_backend=?,summary=?,error=?,finished_at=?
                WHERE id=?""", (status, self.store.redact(sandbox_backend),
                canonical_json(self.store.redact(summary)), canonical_json(self.store.redact(error)), _now(), record_id))
            return _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())

    def terminalize_if_active(self, record_id, *, status, sandbox_backend, error):
        """Idempotently map an authoritative outer-job terminal state into history."""
        if status not in TERMINAL or not isinstance(error, dict):
            raise ValueError("Analysis outer-job outcome is malformed")
        canonical_json(error)
        if not isinstance(sandbox_backend, str) or not sandbox_backend:
            raise ValueError("Analysis sandbox backend is required")
        with self.store.transaction(write=True) as conn:
            current = _record(conn.execute(
                "SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            if current["status"] in TERMINAL:
                return current
            conn.execute("""UPDATE analysis_records SET status=?,sandbox_backend=?,summary=?,error=?,finished_at=?
                WHERE id=? AND status IN ('pending','running')""",
                (status, self.store.redact(sandbox_backend), canonical_json({}),
                 canonical_json(self.store.redact(error)), _now(), record_id))
            return _record(conn.execute(
                "SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())

    def get_record(self, record_id):
        with self.store.transaction() as conn:
            value = _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            value["artifacts"] = [dict(row) for row in conn.execute("""SELECT ara.ordinal,ara.role,
                a.id AS artifact_id,a.name,a.media_type,a.size,a.sha256 FROM analysis_record_artifacts ara
                JOIN artifacts a ON a.id=ara.artifact_id WHERE ara.record_id=? ORDER BY ara.ordinal""", (record_id,))]
            return value

    def register_method(self, session_id, record_id, *, name, description, parameter_schema):
        self.store.get_session(session_id)
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ValueError("Analysis method name must be lowercase snake_case")
        if not isinstance(description, str) or not 1 <= len(description) <= 2000:
            raise ValueError("Analysis method description must contain 1..2000 characters")
        validate_method_schema(parameter_schema)
        with self.store.transaction(write=True) as conn:
            record = _record(conn.execute("SELECT * FROM analysis_records WHERE id=?", (record_id,)).fetchone())
            if record["status"] != "succeeded":
                raise ValueError("Only a successful analysis record can become a method")
            if record["job_id"] is not None:
                table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_jobs'").fetchone()
                job = None if table is None else conn.execute(
                    "SELECT status,handler_ref,arguments FROM execution_jobs WHERE id=?",
                    (record["job_id"],)).fetchone()
                if job is None or job["status"] != "succeeded" or job["handler_ref"] != (
                        "adj_partner.analysis.worker:analysis_job"):
                    raise ValueError("Analysis method source does not have a matching successful durable job")
                try:
                    analysis_id = json.loads(job["arguments"])["analysis_id"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Analysis method source job metadata is malformed") from exc
                if analysis_id != record_id:
                    raise ValueError("Analysis method source job names another analysis record")
            artifact = self._artifact(conn, record["code_artifact_id"])
            version = conn.execute("SELECT COALESCE(MAX(version),0)+1 FROM analysis_methods WHERE name=?",
                                   (name,)).fetchone()[0]
            identifier = new_id("method")
            conn.execute("INSERT INTO analysis_methods VALUES(?,?,?,?,?,?,?,?,?,?)",
                (identifier, name, version, self.store.redact(description), record["language"],
                 record["code_artifact_id"], artifact["sha256"], canonical_json(parameter_schema),
                 record_id, _now()))
            return _method(conn.execute("SELECT * FROM analysis_methods WHERE id=?", (identifier,)).fetchone())

    def get_method(self, method_id):
        with self.store.transaction() as conn:
            return _method(conn.execute("SELECT * FROM analysis_methods WHERE id=?", (method_id,)).fetchone())

    def search(self, session_id, *, query="", language=None, status=None,
               method_name=None, created_after=None, created_before=None, limit=20):
        self.store.get_session(session_id)
        if not isinstance(query, str) or len(query) > 500:
            raise ValueError("Analysis history query is too long")
        if language is not None and language not in LANGUAGES:
            raise ValueError("Unknown analysis history language")
        if status is not None and status not in TERMINAL | {"pending", "running"}:
            raise ValueError("Unknown analysis history status")
        if method_name is not None and (not isinstance(method_name, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", method_name)):
            raise ValueError("Invalid analysis history method name")
        timestamps = []
        for label, value in (("created_after", created_after), ("created_before", created_before)):
            if value is None:
                timestamps.append(None)
                continue
            if not isinstance(value, str) or len(value) > 64:
                raise ValueError(f"Invalid {label} timestamp")
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"Invalid {label} timestamp") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(f"Invalid {label} timestamp")
            timestamps.append(parsed.astimezone(timezone.utc).isoformat())
        created_after, created_before = timestamps
        if created_after is not None and created_before is not None and created_after > created_before:
            raise ValueError("Analysis history creation window is reversed")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Analysis history limit must be 1..50")
        pattern = f"%{query}%"
        with self.store.transaction() as conn:
            clauses, parameters = ["title LIKE ?"], [pattern]
            if language is not None:
                clauses.append("language=?")
                parameters.append(language)
            if status is not None:
                clauses.append("status=?")
                parameters.append(status)
            if created_after is not None:
                clauses.append("created_at>=?")
                parameters.append(created_after)
            if created_before is not None:
                clauses.append("created_at<=?")
                parameters.append(created_before)
            records = [_record(row) for row in conn.execute(
                "SELECT * FROM analysis_records WHERE " + " AND ".join(clauses) +
                " ORDER BY created_at DESC,id DESC LIMIT ?", (*parameters, limit))]
            method_clauses, method_parameters = ["(name LIKE ? OR description LIKE ?)"], [pattern, pattern]
            if method_name is not None:
                method_clauses.append("name=?")
                method_parameters.append(method_name)
            if created_after is not None:
                method_clauses.append("created_at>=?")
                method_parameters.append(created_after)
            if created_before is not None:
                method_clauses.append("created_at<=?")
                method_parameters.append(created_before)
            methods = [_method(row) for row in conn.execute(
                "SELECT * FROM analysis_methods WHERE " + " AND ".join(method_clauses) +
                " ORDER BY created_at DESC,id DESC LIMIT ?", (*method_parameters, limit))]
            if records:
                placeholders = ",".join("?" for _ in records)
                grouped = {record["id"]: [] for record in records}
                for row in conn.execute(f"""SELECT ara.record_id,ara.ordinal,ara.role,
                    a.id AS artifact_id,a.name,a.media_type,a.size,a.sha256
                    FROM analysis_record_artifacts ara JOIN artifacts a ON a.id=ara.artifact_id
                    WHERE ara.record_id IN ({placeholders}) ORDER BY ara.record_id,ara.ordinal""",
                    tuple(grouped)):
                    item = dict(row)
                    grouped[item.pop("record_id")].append(item)
                for record in records:
                    record["artifacts"] = grouped[record["id"]]
        for record in records:
            record.pop("parameters", None)
            record.pop("inputs", None)
            record.pop("limits", None)
        return {"records": records, "methods": methods}

    def read_text(self, session_id, artifact_id, start=0, max_characters=8000):
        self.store.get_session(session_id)
        if type(start) is not int or start < 0 or type(max_characters) is not int or not 1 <= max_characters <= 16000:
            raise ValueError("Invalid analysis history text page")
        with self.store.transaction() as conn:
            row = conn.execute("""SELECT a.* FROM artifacts a JOIN analysis_record_artifacts ara
                ON ara.artifact_id=a.id WHERE a.id=? LIMIT 1""", (artifact_id,)).fetchone()
            if row is None:
                raise PermissionError("Artifact is not part of project analysis history")
            artifact = dict(row)
        if artifact["media_type"] not in TEXT_MEDIA_TYPES and not artifact["media_type"].startswith("text/"):
            raise ValueError("Analysis history artifact is not readable text")
        if artifact["size"] > MAX_HISTORY_ARTIFACT:
            raise ValueError("Analysis history text artifact exceeds the read limit")
        try:
            content = self.store.read_artifact(artifact_id, max_bytes=MAX_HISTORY_ARTIFACT).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Analysis history artifact is not valid UTF-8 text") from exc
        content = self.store.redact(content)
        if start > len(content):
            raise ValueError("Analysis history character offset is beyond content")
        requested = min(len(content), start + max_characters)
        low, high = start, requested
        while low < high:
            middle = (low + high + 1) // 2
            if len(canonical_json(content[start:middle]).encode("utf-8")) <= MAX_HISTORY_PAGE_BYTES:
                low = middle
            else:
                high = middle - 1
        return {"artifact_id": artifact_id, "content": content[start:low], "start": start,
                "next_start": low if low < len(content) else None, "total_characters": len(content),
                "sha256": artifact["sha256"], "media_type": artifact["media_type"]}
