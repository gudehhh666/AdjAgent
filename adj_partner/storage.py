"""Durable host state and ordered audit events using short SQLite transactions.

Every operation opens its own connection, so a Store may safely be used from
worker threads. Files are owned by this store; legacy source files are untouched.
"""

from contextlib import contextmanager
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import threading

from .contracts import new_id


SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"created", "running", "waiting", "succeeded", "failed", "cancelled", "budget_exhausted"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "budget_exhausted"})
PENDING_ACTION_STATUSES = frozenset({
    "pending", "approval_required", "requires_approval", "awaiting_approval", "needs_approval", "waiting", "running",
})
_SENSITIVE_KEY = re.compile(r"api_?key|authorization|token|password|secret|credential|cookies?", re.I)
_TOKEN_COUNTER = re.compile(
    r"(?:max_|min_|remaining_|used_|estimated_|accounted_|cache_read_|cache_creation_)?"
    r"(?:input_|output_|prompt_|completion_|total_|cached_|reasoning_|audio_|text_|image_|video_)?tokens(?:_used)?$"
    r"|(?:input_|output_|prompt_|completion_|total_)?token_(?:count|usage)$"
)
_TOKEN_DETAILS = frozenset({"input_tokens_details", "output_tokens_details", "prompt_tokens_details", "completion_tokens_details"})
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    re.compile(r"""(?i)\bauthorization["']?\s*[:=]\s*["']?Basic\s+[A-Za-z0-9+/]+=*"""),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)"
               r"""["']?\s*[=:]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s"',;}]+)"""),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _validate_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Invalid identifier")
    return value


class Store:
    def __init__(self, root: Path | str, secrets: list[str] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        self.db_path = self.root / "db.sqlite3"
        self._secrets: tuple[str, ...] = ()
        self._secrets_lock = threading.Lock()
        self.register_secrets([] if secrets is None else secrets)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError(f"Unsupported store schema version: {version}")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    goal TEXT NOT NULL, acceptance TEXT NOT NULL, status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    task_id TEXT NOT NULL REFERENCES tasks(id), status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, budget TEXT NOT NULL,
                    usage TEXT NOT NULL, state TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    run_id TEXT REFERENCES runs(id), kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_session ON events(session_id, seq);
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    run_id TEXT REFERENCES runs(id), message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    call_id TEXT NOT NULL, tool_name TEXT NOT NULL, args TEXT NOT NULL,
                    args_hash TEXT NOT NULL, toolset_revision INTEGER NOT NULL,
                    status TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL,
                    finished_at TEXT, UNIQUE(run_id, call_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, path TEXT NOT NULL,
                    name TEXT NOT NULL, media_type TEXT NOT NULL, size INTEGER NOT NULL,
                    provenance TEXT NOT NULL, session_id TEXT REFERENCES sessions(id),
                    run_id TEXT REFERENCES runs(id), created_at TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self, write: bool = False):
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except BaseException:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self, write: bool = False):
        """Extension surface for namespaced research/job tables in this database.

        Keep transactions short: never execute tools or wait for external work
        inside this block. Extension writers must call redact on their payloads.
        Read transactions are query-only consistent snapshots; every connection
        is closed on exit. Do not nest Store writes inside a write transaction.
        """
        with self._connection(write=write) as conn:
            if not write:
                conn.execute("PRAGMA query_only=ON")
                conn.execute("BEGIN")
            yield conn

    def register_secrets(self, values: Iterable[str]) -> None:
        """Add host-configured secrets before persisting provider material.

        This never replaces prior secrets or rewrites existing database history.
        Future writes and exports use the expanded set. Registration must finish
        before the corresponding provider can emit events or results.
        """
        if isinstance(values, (str, bytes)):
            raise TypeError("Secret values must be an iterable of strings")
        incoming = tuple(values)
        if any(not isinstance(value, str) for value in incoming):
            raise TypeError("Every secret value must be a string")
        with self._secrets_lock:
            self._secrets = tuple(dict.fromkeys((*self._secrets, *(value for value in incoming if value))))

    def _redact_text(self, value: str) -> str:
        # Native provider tool arguments are JSON encoded inside a string. Keep
        # unchanged provider text byte-for-byte when no redaction is necessary.
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                cleaned = self.redact(parsed)
                # Recursion sanitized keys and values. Do not apply free-text
                # assignment regexes to valid provider JSON: that would remove
                # its quoted keys/values and break the native argument envelope.
                return _json(cleaned) if cleaned != parsed else value
        with self._secrets_lock:
            known_secrets = self._secrets
        for secret in sorted(known_secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        for pattern in _CREDENTIAL_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value

    def redact(self, value):
        """Return a recursive sanitized copy for durable metadata, never for execution.

        Known secret strings and credential shapes are removed as defense in
        depth. This is not a proof that arbitrary free text contains no secrets.
        Numeric token usage/budget counters remain usable for accounting.
        """
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower().replace("-", "_")
                counter = (_TOKEN_COUNTER.fullmatch(normalized) and isinstance(item, (int, float))
                           and not isinstance(item, bool)) or (normalized in _TOKEN_DETAILS and isinstance(item, dict))
                cleaned[self._redact_text(str(key))] = (
                    "[REDACTED]" if _SENSITIVE_KEY.search(normalized) and not counter else self.redact(item)
                )
            return cleaned
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _encode(self, value) -> str:
        return _json(self.redact(value))

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        value = dict(row)
        for key in ("config", "acceptance", "budget", "usage", "state", "payload", "message", "args", "result", "provenance"):
            if key in value and value[key] is not None:
                value[key] = json.loads(value[key])
        value.pop("args_hash", None)
        if "kind" in value:
            value["id"] = value["seq"]
        return value

    def _get(self, conn, table: str, identifier: str) -> dict:
        _validate_id(identifier)
        row = conn.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown {table.rstrip('s')} identifier: {identifier}")
        return self._decode(row)

    def _scope(self, conn, session_id: str, run_id: str | None = None):
        self._get(conn, "sessions", session_id)
        if run_id is not None and self._get(conn, "runs", run_id)["session_id"] != session_id:
            raise ValueError("Run does not belong to session")

    def _event(self, conn, session_id, kind, payload, run_id=None) -> dict:
        cursor = conn.execute(
            "INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
            (session_id, run_id, self.redact(kind), self._encode(payload), _now()),
        )
        return self._decode(conn.execute("SELECT * FROM events WHERE seq=?", (cursor.lastrowid,)).fetchone())

    def create_session(self, config: dict) -> dict:
        identifier = new_id("session")
        with self._connection(write=True) as conn:
            conn.execute("INSERT INTO sessions VALUES(?,?,0,?)", (identifier, self._encode(config), _now()))
            value = self._get(conn, "sessions", identifier)
            self._event(conn, identifier, "session.created", value)
        return value

    def get_session(self, identifier: str) -> dict:
        with self._connection() as conn:
            return self._get(conn, "sessions", identifier)

    def list_sessions(self) -> list[dict]:
        with self._connection() as conn:
            return [self._decode(row) for row in conn.execute("SELECT * FROM sessions ORDER BY rowid")]

    def _update(self, table: str, identifier: str, changes: dict, expected_revision: int | None) -> dict:
        allowed = {"sessions": {"config"}, "tasks": {"goal", "acceptance", "status"},
                   "runs": {"status", "state", "usage", "budget"}}[table]
        if not isinstance(changes, dict) or set(changes) - allowed:
            raise ValueError("Unknown or immutable update field")
        with self._connection(write=True) as conn:
            current = self._get(conn, table, identifier)
            if expected_revision is not None and current["revision"] != expected_revision:
                raise ValueError("Stale revision; reload state before updating")
            if table == "runs" and "status" in changes:
                status = changes["status"]
                if status not in RUN_STATUSES:
                    raise ValueError("Invalid run status")
                if current["status"] in TERMINAL_STATUSES and status != current["status"]:
                    raise ValueError("A terminal run outcome cannot be changed")
            values = {key: self._encode(value) if key in {"config", "acceptance", "state", "usage", "budget"}
                      else self.redact(value) for key, value in changes.items()}
            values["revision"] = current["revision"] + 1
            assignments = ",".join(f"{key}=?" for key in values)
            conn.execute(f"UPDATE {table} SET {assignments} WHERE id=?", [*values.values(), identifier])
            result = self._get(conn, table, identifier)
            session_id = identifier if table == "sessions" else result["session_id"]
            self._event(conn, session_id, f"{table.rstrip('s')}.updated",
                        {"id": identifier, "revision": result["revision"], "changes": changes},
                        identifier if table == "runs" else None)
        return result

    def update_session(self, identifier: str, config: dict, expected_revision: int) -> dict:
        return self._update("sessions", identifier, {"config": config}, expected_revision)

    def create_task(self, session_id: str, goal: str, acceptance: dict | None = None) -> dict:
        identifier = new_id("task")
        with self._connection(write=True) as conn:
            self._scope(conn, session_id)
            conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,0,?)",
                         (identifier, session_id, self.redact(goal), self._encode(acceptance or {}), "open", _now()))
            value = self._get(conn, "tasks", identifier)
            self._event(conn, session_id, "task.created", value)
        return value

    def get_task(self, identifier: str) -> dict:
        with self._connection() as conn:
            return self._get(conn, "tasks", identifier)

    def list_tasks(self, session_id: str | None = None) -> list[dict]:
        """List durable tasks, including those without a run, in creation order."""
        with self._connection() as conn:
            if session_id is None:
                rows = conn.execute("SELECT * FROM tasks ORDER BY rowid")
            else:
                self._scope(conn, session_id)
                rows = conn.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY rowid", (session_id,))
            return [self._decode(row) for row in rows]

    def update_task(self, identifier: str, changes: dict, expected_revision: int | None = None) -> dict:
        return self._update("tasks", identifier, changes, expected_revision)

    def create_run(self, session_id: str, task_id: str, budget: dict) -> dict:
        identifier = new_id("run")
        with self._connection(write=True) as conn:
            self._scope(conn, session_id)
            if self._get(conn, "tasks", task_id)["session_id"] != session_id:
                raise ValueError("Task does not belong to session")
            conn.execute("INSERT INTO runs VALUES(?,?,?,?,0,?,?,?,?)",
                         (identifier, session_id, task_id, "created", self._encode(budget), "{}", "{}", _now()))
            value = self._get(conn, "runs", identifier)
            self._event(conn, session_id, "run.created", value, identifier)
        return value

    def get_run(self, identifier: str) -> dict:
        with self._connection() as conn:
            return self._get(conn, "runs", identifier)

    def update_run(self, identifier: str, changes: dict, expected_revision: int | None = None) -> dict:
        return self._update("runs", identifier, changes, expected_revision)

    def list_runs(self, session_id: str | None = None) -> list[dict]:
        with self._connection() as conn:
            if session_id is None:
                rows = conn.execute("SELECT * FROM runs ORDER BY rowid")
            else:
                self._scope(conn, session_id)
                rows = conn.execute("SELECT * FROM runs WHERE session_id=? ORDER BY rowid", (session_id,))
            return [self._decode(row) for row in rows]

    def append_event(self, session_id: str, kind: str, payload: dict, run_id: str | None = None) -> dict:
        with self._connection(write=True) as conn:
            self._scope(conn, session_id, run_id)
            return self._event(conn, session_id, kind, payload, run_id)

    def events(self, session_id: str, run_id: str | None = None) -> list[dict]:
        with self._connection() as conn:
            self._scope(conn, session_id, run_id)
            query = "SELECT * FROM events WHERE session_id=?"
            params = [session_id]
            if run_id is not None:
                query += " AND run_id=?"
                params.append(run_id)
            return [self._decode(row) for row in conn.execute(query + " ORDER BY seq", params)]

    def append_message(self, session_id: str, message: dict, run_id: str | None = None) -> dict:
        message = self.redact(message)
        with self._connection(write=True) as conn:
            self._scope(conn, session_id, run_id)
            conn.execute("INSERT INTO messages(session_id,run_id,message,created_at) VALUES(?,?,?,?)",
                         (session_id, run_id, self._encode(message), _now()))
            self._event(conn, session_id, "message.appended", {"message": message}, run_id)
        return message

    def messages(self, session_id: str) -> list[dict]:
        with self._connection() as conn:
            self._scope(conn, session_id)
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT message FROM messages WHERE session_id=? ORDER BY seq", (session_id,))]

    def record_action(self, run_id: str, call_id: str, tool_name: str, args: dict,
                      toolset_revision: int) -> dict:
        """Commit intent before executing a tool; a returned existing row is not a retry grant."""
        return self.claim_action(run_id, call_id, tool_name, args, toolset_revision)[0]

    def claim_action(self, run_id: str, call_id: str, tool_name: str, args: dict,
                     toolset_revision: int) -> tuple[dict, bool]:
        """Atomically journal intent and grant execution to exactly one new claimant.

        Existing started rows have uncertain effects and must not be reexecuted.
        The ownership boolean is ephemeral, never a persisted permission token.
        """
        _validate_id(call_id)
        fingerprint = hashlib.sha256(_json({"tool_name": tool_name, "args": args}).encode("utf-8")).hexdigest()
        with self._connection(write=True) as conn:
            run = self._get(conn, "runs", run_id)
            existing = conn.execute("SELECT * FROM actions WHERE run_id=? AND call_id=?",
                                    (run_id, call_id)).fetchone()
            if existing is not None:
                if existing["tool_name"] != self.redact(tool_name) or existing["args_hash"] != fingerprint:
                    raise ValueError("Tool call ID reused with a different tool or arguments")
                return self._decode(existing), False
            identifier = new_id("action")
            conn.execute("INSERT INTO actions VALUES(?,?,?,?,?,?,?, 'started',NULL,?,NULL)",
                         (identifier, run_id, call_id, self.redact(tool_name), self._encode(args), fingerprint, toolset_revision, _now()))
            value = self._get(conn, "actions", identifier)
            self._event(conn, run["session_id"], "action.started", value, run_id)
        return value, True

    def finish_action(self, run_id: str, call_id: str, result: dict) -> dict:
        """Save a result; pending/approval replies have no finished-side-effect timestamp."""
        _validate_id(call_id)
        result = self.redact(result)
        with self._connection(write=True) as conn:
            run = self._get(conn, "runs", run_id)
            row = conn.execute("SELECT * FROM actions WHERE run_id=? AND call_id=?", (run_id, call_id)).fetchone()
            if row is None:
                raise KeyError(f"Unknown action: {call_id}")
            current = self._decode(row)
            if current["finished_at"] is not None:
                if current["result"] != result:
                    raise ValueError("Cannot replace a completed action result")
                return current
            result_status = result.get("status", "ok")
            pending = result_status in PENDING_ACTION_STATUSES
            status = result_status if pending else ("failed" if result_status == "error" else "completed")
            conn.execute("UPDATE actions SET status=?, result=?, finished_at=? WHERE id=?",
                         (status, self._encode(result), None if pending else _now(), current["id"]))
            value = self._get(conn, "actions", current["id"])
            self._event(conn, run["session_id"], "action.pending" if pending else "action.finished", value, run_id)
        return value

    def get_action(self, run_id: str, call_id: str) -> dict | None:
        _validate_id(call_id)
        with self._connection() as conn:
            self._get(conn, "runs", run_id)
            row = conn.execute("SELECT * FROM actions WHERE run_id=? AND call_id=?", (run_id, call_id)).fetchone()
            return self._decode(row) if row is not None else None

    def list_actions(self, run_id: str) -> list[dict]:
        with self._connection() as conn:
            self._get(conn, "runs", run_id)
            return [self._decode(row) for row in conn.execute(
                "SELECT * FROM actions WHERE run_id=? ORDER BY rowid", (run_id,))]

    def _artifact_path(self, digest: str) -> Path:
        directory = self.root / "artifacts"
        if directory.is_symlink() or directory.resolve() != directory:
            raise ValueError("Artifact directory cannot be a symlink")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Artifact integrity: invalid hash")
        return directory / digest

    def _read_blob(self, digest: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("Artifact read limit must be a positive integer")
        path = self._artifact_path(digest)
        if path.is_symlink():
            raise ValueError("Artifact integrity: symlink is not allowed")
        try:
            # Open nonblocking so a corrupted digest path replaced by a FIFO
            # cannot wait for a writer before the descriptor's type is checked.
            # Regular-file reads retain their ordinary semantics.
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise ValueError("Artifact integrity: content unavailable") from exc
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Artifact integrity: content is not a regular file")
            if max_bytes is not None and os.fstat(stream.fileno()).st_size > max_bytes:
                raise ValueError("Artifact exceeds read limit")
            content = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
            if max_bytes is not None and len(content) > max_bytes:
                raise ValueError("Artifact exceeds read limit")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Artifact integrity: content hash mismatch")
        return content

    def _put_blob(self, content: bytes) -> tuple[str, Path]:
        digest = hashlib.sha256(content).hexdigest()
        path = self._artifact_path(digest)
        # Hard-link publication is atomic and exclusive: it never replaces an
        # existing object. The temporary file is fully flushed before publishing.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".artifact-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    self._read_blob(digest)
            finally:
                temporary.unlink(missing_ok=True)
        return digest, path

    def put_artifact(self, content: bytes | str, name: str,
                     media_type: str = "application/octet-stream", provenance: dict | None = None,
                     session_id: str | None = None, run_id: str | None = None) -> dict:
        """Store exact bytes privately; callers choose whether source data needs scrubbing.

        Artifact contents are not logs and are never copied into trace exports.
        Metadata receives log redaction, while exact bytes keep scientific hashes.
        """
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or "/" in name or "\\" in name or any(ord(char) < 32 for char in name)):
            raise ValueError("Invalid artifact name")
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, bytes):
            raise TypeError("Artifact content must be bytes or text")
        with self._connection() as conn:
            if run_id is not None:
                run = self._get(conn, "runs", run_id)
                if session_id is None:
                    session_id = run["session_id"]
            if session_id is not None:
                self._scope(conn, session_id, run_id)
        # The content-addressed write may be large. Publish exact bytes outside
        # SQLite's writer lock, then atomically insert metadata and its event.
        # A crash before metadata commit can leave an unreferenced private blob;
        # it can never expose a partial referenced blob or overwrite old content.
        digest, path = self._put_blob(content)
        with self._connection(write=True) as conn:
            if session_id is not None:
                self._scope(conn, session_id, run_id)
            identifier = new_id("artifact")
            conn.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (identifier, digest, self.redact(str(path)), self.redact(name), self.redact(media_type), len(content),
                          self._encode(provenance or {}), session_id, run_id, _now()))
            value = self._get(conn, "artifacts", identifier)
            if session_id is not None:
                self._event(conn, session_id, "artifact.created", value, run_id)
        return value

    def read_artifact(self, identifier: str, *, max_bytes: int | None = None) -> bytes:
        with self._connection() as conn:
            metadata = self._get(conn, "artifacts", identifier)
        return self._read_blob(metadata["sha256"], max_bytes=max_bytes)

    def artifacts(self, session_id: str | None = None, run_id: str | None = None) -> list[dict]:
        with self._connection() as conn:
            if run_id is not None and session_id is None:
                session_id = self._get(conn, "runs", run_id)["session_id"]
            if session_id is not None:
                self._scope(conn, session_id, run_id)
            clauses, params = [], []
            for column, value in (("session_id", session_id), ("run_id", run_id)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            query = "SELECT * FROM artifacts" + (" WHERE " + " AND ".join(clauses) if clauses else "")
            return [self._decode(row) for row in conn.execute(query + " ORDER BY rowid", params)]

    def _trace_snapshot(self, session_id: str, run_id: str | None) -> dict:
        # One read snapshot prevents concurrent writes from producing a trace
        # whose session revision and event/action tables disagree.
        with self.transaction() as conn:
            self._scope(conn, session_id, run_id)
            session = self._get(conn, "sessions", session_id)
            run = self._get(conn, "runs", run_id) if run_id else None
            tasks_query = "SELECT * FROM tasks WHERE session_id=?"
            task_params = [session_id]
            if run:
                tasks_query += " AND id=?"
                task_params.append(run["task_id"])
            tasks = [self._decode(row) for row in conn.execute(tasks_query + " ORDER BY rowid", task_params)]
            runs = [run] if run else [self._decode(row) for row in conn.execute(
                "SELECT * FROM runs WHERE session_id=? ORDER BY rowid", (session_id,))]
            snapshot = {"session": session, "tasks": tasks, "runs": runs}
            for table in ("events", "messages", "artifacts"):
                query = f"SELECT * FROM {table} WHERE session_id=?"
                params = [session_id]
                if run_id:
                    query += " AND run_id=?"
                    params.append(run_id)
                snapshot[table] = [self._decode(row) for row in conn.execute(query + " ORDER BY rowid", params)]
            query = "SELECT actions.* FROM actions JOIN runs ON actions.run_id=runs.id WHERE runs.session_id=?"
            params = [session_id]
            if run_id:
                query += " AND actions.run_id=?"
                params.append(run_id)
            snapshot["actions"] = [self._decode(row) for row in conn.execute(query + " ORDER BY actions.rowid", params)]
        return self.redact(snapshot)

    @staticmethod
    def _markdown_block(value) -> str:
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        runs = re.findall(r"`+", content)
        fence = "`" * max(3, 1 + max((len(run) for run in runs), default=0))
        return f"{fence}\n{content}\n{fence}\n"

    def _trace_markdown(self, snapshot: dict, run_id: str | None) -> str:
        lines = ["# Session trace", "", f"Session: `{snapshot['session']['id']}`", ""]
        if run_id:
            lines.extend([f"Run: `{run_id}`", ""])
        lines.extend(["This audit record contains public messages, host state, tool actions, and evidence references; "
                      "it does not contain private model reasoning or copies of artifact bytes.", "", "## Tasks", ""])
        for task in snapshot["tasks"]:
            lines.extend([f"### {task['id']} — {task['status']}", "", self._markdown_block(task["goal"]),
                          "Acceptance:", "", self._markdown_block(task["acceptance"])])
        lines.extend(["## Runs", ""])
        for run in snapshot["runs"]:
            lines.extend([f"### {run['id']} — {run['status']}", "", self._markdown_block(run)])
        lines.extend(["## Messages", ""])
        for record in snapshot["messages"]:
            message = record["message"]
            lines.extend([f"### Message {record['seq']}", "", self._markdown_block(message)])
        lines.extend(["## Ordered events", ""])
        for event in snapshot["events"]:
            lines.extend([f"### {event['seq']} — {event['kind']}", "", event["created_at"], "",
                          self._markdown_block(event["payload"])])
        lines.extend(["## Actions", ""])
        for action in snapshot["actions"]:
            lines.extend([f"### {action['call_id']} — {action['status']}", "", self._markdown_block(action)])
        lines.extend(["## Artifact references", "", "Artifact bytes remain in the private store. Review and scrub "
                      "content explicitly before sharing it.", ""])
        for artifact in snapshot["artifacts"]:
            lines.append(self._markdown_block(artifact))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_exclusive(path: Path, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".trace-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def export_trace(self, session_id: str, destination: Path | str | None = None,
                     run_id: str | None = None) -> dict[str, str]:
        """Export an immutable redacted snapshot and its hash manifest.

        A destination is an explicit directory (relative paths are beneath the
        store); traversal and symlinks are refused. Each export gets fresh names,
        preserving earlier snapshots. Artifact bytes are referenced, never copied.
        """
        snapshot = self._trace_snapshot(session_id, run_id)
        directory = Path(destination) if destination is not None else self.root / "traces" / (run_id or session_id)
        if ".." in directory.parts:
            raise ValueError("Trace destination cannot contain path traversal")
        if not directory.is_absolute():
            directory = self.root / directory
        if any(parent.is_symlink() for parent in (directory, *directory.parents)):
            raise ValueError("Trace destination cannot be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        directory = directory.resolve()
        export_id = new_id("trace")
        paths = {"jsonl": directory / f"{export_id}.jsonl", "markdown": directory / f"{export_id}.md",
                 "manifest": directory / f"{export_id}.manifest.json"}
        records = [{"record_type": "event", **event} for event in snapshot["events"]]
        records.append({"record_type": "session", "data": snapshot["session"]})
        for collection, kind in (("tasks", "task"), ("runs", "run"), ("messages", "message"),
                                 ("actions", "action"), ("artifacts", "artifact")):
            records.extend({"record_type": kind, "data": record} for record in snapshot[collection])
        contents = {"jsonl": ("\n".join(_json(record) for record in records) + "\n").encode("utf-8"),
                    "markdown": self._trace_markdown(snapshot, run_id).encode("utf-8")}
        manifest = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "run_id": run_id,
                    "created_at": _now(), "artifacts": snapshot["artifacts"],
                    "artifact_content": "private-reference-only", "files": {}}
        for kind, content in contents.items():
            self._write_exclusive(paths[kind], content)
            manifest["files"][kind] = {"path": str(paths[kind]), "size": len(content),
                                       "sha256": hashlib.sha256(content).hexdigest()}
        self._write_exclusive(paths["manifest"], (self._encode(manifest) + "\n").encode("utf-8"))
        return {kind: str(path) for kind, path in paths.items()}

    def close(self) -> None:
        """No retained connections exist; safe and idempotent for context owners."""
