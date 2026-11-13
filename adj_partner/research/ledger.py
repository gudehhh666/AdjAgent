"""Namespaced version journal in the host Store, never a second state database.

The research journal commits with each state change. Common host events are a
best-effort post-commit mirror; snapshot/events are authoritative for recovery.
Artifact I/O and host providers must remain outside writer transactions.
"""

from datetime import datetime, timezone
import json

from ..contracts import new_id
from .validation import bounded_json, identifier, revision


ENTITIES = frozenset({"source", "evidence", "claim", "assessment", "report", "validation"})
TASK_ENTITIES = frozenset({"claim", "assessment", "report", "validation"})


class Ledger:
    def __init__(self, store, assessor=None):
        self.store, self.assessor = store, assessor
        with store.transaction(write=True) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS research_records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                run_id TEXT NOT NULL REFERENCES runs(id), task_id TEXT NOT NULL REFERENCES tasks(id),
                subject_id TEXT, payload TEXT NOT NULL, UNIQUE(entity,id,revision)
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS research_scope ON research_records(session_id,entity,task_id)")
            conn.execute("""CREATE TABLE IF NOT EXISTS research_source_keys (
                session_id TEXT NOT NULL REFERENCES sessions(id), key_hash TEXT NOT NULL,
                fingerprint TEXT NOT NULL, source_id TEXT NOT NULL,
                PRIMARY KEY(session_id,key_hash)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id), run_id TEXT NOT NULL REFERENCES runs(id),
                kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
            )""")

    def _scope(self, conn, session_id, run_id):
        identifier(session_id)
        identifier(run_id)
        session = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if session is None or run is None:
            raise KeyError("Unknown session or run")
        if run["session_id"] != session_id:
            raise PermissionError("Run does not belong to session")
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (run["task_id"],)).fetchone()
        if task is None or task["session_id"] != session_id:
            raise PermissionError("Task does not belong to session")
        result = dict(task)
        result["acceptance"] = json.loads(result["acceptance"])
        return result

    def _get(self, conn, entity, session_id, task_id, record_id, version=None):
        identifier(record_id)
        if version is not None:
            revision(version)
        query = "SELECT payload FROM research_records WHERE entity=? AND id=? AND session_id=?"
        params = [entity, record_id, session_id]
        if entity in TASK_ENTITIES:
            query += " AND task_id=?"
            params.append(task_id)
        if version is not None:
            query += " AND revision=?"
            params.append(version)
        row = conn.execute(query + " ORDER BY revision DESC LIMIT 1", params).fetchone()
        if row is None:
            raise KeyError(f"Unknown or out-of-scope {entity}: {record_id}")
        return json.loads(row["payload"])

    def _list(self, conn, entity, session_id, task_id, *, history=False):
        query = "SELECT r.payload FROM research_records r WHERE r.entity=? AND r.session_id=?"
        params = [entity, session_id]
        if entity in TASK_ENTITIES:
            query += " AND r.task_id=?"
            params.append(task_id)
        if not history:
            query += " AND r.revision=(SELECT MAX(v.revision) FROM research_records v WHERE v.entity=r.entity AND v.id=r.id)"
        return [json.loads(row[0]) for row in conn.execute(query + " ORDER BY r.seq", params)]

    def _head(self, conn, entity, session_id, task_id, subject_id):
        row = conn.execute("""SELECT payload FROM research_records
            WHERE entity=? AND session_id=? AND task_id=? AND subject_id=? ORDER BY seq DESC LIMIT 1""",
            (entity, session_id, task_id, subject_id)).fetchone()
        return json.loads(row[0]) if row else None

    def _record(self, entity, session_id, run_id, task_id, **fields):
        return {"id": new_id(entity), "revision": 0, "session_id": session_id,
                "run_id": run_id, "task_id": task_id,
                "created_at": datetime.now(timezone.utc).isoformat(), **fields}

    def _save(self, conn, entity, record, *, subject_id=None):
        if entity not in ENTITIES:
            raise ValueError("Unknown research record entity")
        identifier(record["id"])
        revision(record["revision"])
        clean = self.store.redact(record)
        encoded = bounded_json(clean)
        conn.execute("""INSERT INTO research_records(entity,id,revision,session_id,run_id,task_id,subject_id,payload)
            VALUES(?,?,?,?,?,?,?,?)""", (entity, record["id"], record["revision"], record["session_id"],
                                       record["run_id"], record["task_id"], subject_id, encoded))
        conn.execute("INSERT INTO research_events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
                     (record["session_id"], record.get("last_modified_run_id", record["run_id"]),
                      f"research.{entity}.recorded", encoded, record.get("updated_at", record["created_at"])))
        return clean

    def _mirror(self, entity, record):
        try:
            self.store.append_event(record["session_id"], f"research.{entity}.recorded", record,
                                    record.get("last_modified_run_id", record["run_id"]))
        except Exception:
            # A journaled mutation remains successful if its duplicate host-event
            # mirror fails. Recovery consumers read research_events directly.
            pass
        return self.store.redact(record)

    def _read(self, entity, session_id, run_id, record_id, version=None):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            result = self._get(conn, entity, session_id, task["id"], record_id, version)
        return self.store.redact(result)

    def _read_list(self, entity, session_id, run_id):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            result = self._list(conn, entity, session_id, task["id"])
        return self.store.redact(result)

    def _journal(self, conn, session_id, run_id=None):
        query = "SELECT * FROM research_events WHERE session_id=?"
        params = [session_id]
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        records = []
        for row in conn.execute(query + " ORDER BY seq", params):
            record = dict(row)
            record["payload"] = json.loads(record["payload"])
            records.append(record)
        return records

    def events(self, session_id, run_id=None):
        with self.store.transaction() as conn:
            if run_id is not None:
                self._scope(conn, session_id, run_id)
            elif conn.execute("SELECT id FROM sessions WHERE id=?", (identifier(session_id),)).fetchone() is None:
                raise KeyError("Unknown session")
            result = self._journal(conn, session_id, run_id)
        return self.store.redact(result)

    def snapshot(self, session_id, run_id):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            result = {"session_id": session_id, "run_id": run_id, "task": task}
            result["history"] = []
            for entity in sorted(ENTITIES):
                result[entity + "s"] = self._list(conn, entity, session_id, task["id"])
                result["history"].extend({"entity": entity, **record} for record in
                                         self._list(conn, entity, session_id, task["id"], history=True))
            result["events"] = self._journal(conn, session_id, run_id)
        return self.store.redact(result)

    def _artifact(self, conn, session_id, artifact_id, producing_run=None):
        row = conn.execute("SELECT * FROM artifacts WHERE id=?", (identifier(artifact_id),)).fetchone()
        if row is None:
            raise KeyError("Unknown source artifact")
        if row["session_id"] != session_id or (producing_run is not None and row["run_id"] != producing_run):
            raise PermissionError("Artifact does not belong to the producing session/run")
        result = dict(row)
        result["provenance"] = json.loads(result["provenance"])
        return result

    def _verify_source(self, source):
        # Exact private bytes are used only for selection/integrity. Public
        # record/export paths apply the executing Store's current redaction.
        import hashlib
        content = self.store.read_artifact(source["artifact_id"])
        if hashlib.sha256(content).hexdigest() != source["artifact_hash"]:
            raise ValueError("Source artifact hash integrity mismatch")
        return content

    def _source_metadata(self, conn, source):
        metadata = self._artifact(conn, source["session_id"], source["artifact_id"], source["run_id"])
        if metadata["sha256"] != source["artifact_hash"]:
            raise ValueError("Source artifact metadata hash mismatch")
        return metadata
