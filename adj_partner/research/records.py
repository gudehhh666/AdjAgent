"""Immutable actual-material sources, literal selections and versioned claims."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from .validation import KINDS, bounded_json, claim_fields, digest, revision, text


def extract(content, selector):
    if not isinstance(selector, dict) or set(selector) not in ({"passage"}, {"pointer"}):
        raise ValueError("Selector must contain exactly a passage or pointer")
    material = content.decode("utf-8")
    if "passage" in selector:
        passage = text(selector["passage"], "passage")
        start = material.find(passage)
        if start < 0:
            raise ValueError("Passage does not exist in source material")
        return {"text": passage, "value": None,
                "locator": {"passage": passage, "start": start, "end": start + len(passage)}}
    pointer = text(selector["pointer"], "JSON pointer", 2000, empty=True)
    if pointer and not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with slash")
    if re.search(r"~(?![01])", pointer):
        raise ValueError("Invalid JSON pointer escape")
    try:
        value = json.loads(material)
        bounded_json(value)
        for raw_part in pointer.split("/")[1:] if pointer else []:
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", part):
                    raise ValueError("Invalid JSON array index")
                value = value[int(part)]
            elif isinstance(value, dict):
                value = value[part]
            else:
                raise ValueError("JSON pointer traverses a scalar")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("JSON pointer does not exist in source") from exc
    bounded_json(value, 100000)
    return {"text": None, "value": value, "locator": {"pointer": pointer}}


class Records:
    def register_source(self, session_id, run_id, *, artifact_id, kind, title, provenance, idempotency_key):
        """Host only: native adapters must check successful retrieval/job first.

        This verifies artifact origin/integrity, not the scientific correctness
        of trusted Python's material. There is deliberately no source-create tool.
        """
        text(title, "source title", 2000)
        text(idempotency_key, "idempotency key", 512)
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Unknown source kind")
        # Avoid a name collision with the keyword public API.
        from .validation import provenance as check_provenance
        check_provenance(provenance)
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            artifact = self._artifact(conn, session_id, artifact_id, run_id)
        check_provenance(artifact["provenance"])
        fields = dict(artifact_id=artifact_id, artifact_hash=artifact["sha256"], kind=kind,
                      title=title, provenance=deepcopy(provenance), artifact_provenance=artifact["provenance"],
                      media_type=artifact["media_type"])
        record = self._record("source", session_id, run_id, task["id"], **fields)
        self._verify_source(record)
        # A retry may publish another immutable reference to identical bytes.
        # Identity is material+provenance, not the incidental artifact row ID.
        fingerprint = digest({"run_id": run_id, **{key: value for key, value in fields.items() if key != "artifact_id"}})
        key_hash = digest(idempotency_key)
        with self.store.transaction(write=True) as conn:
            self._scope(conn, session_id, run_id)
            current_artifact = self._artifact(conn, session_id, artifact_id, run_id)
            if current_artifact != artifact:
                raise ValueError("Stale artifact metadata")
            existing = conn.execute("SELECT * FROM research_source_keys WHERE session_id=? AND key_hash=?",
                                    (session_id, key_hash)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ValueError("Idempotency key conflicts with existing source material/provenance")
                return self.store.redact(self._get(conn, "source", session_id, task["id"], existing["source_id"]))
            record = self._save(conn, "source", record)
            conn.execute("INSERT INTO research_source_keys VALUES(?,?,?,?)",
                         (session_id, key_hash, fingerprint, record["id"]))
        if kind == "literature":
            self._file_in_library(record)
        return self._mirror("source", record)

    def _file_in_library(self, record):
        from ..library import shared
        try:
            library = shared()
        except Exception:
            return
        if library is None:
            return
        try:
            content = self._verify_source(record)
        except Exception:
            return
        library.ingest_safely(content, record["media_type"],
                              {**record["artifact_provenance"], **record["provenance"]},
                              origin={"kind": "run", "ref": f"{record['session_id']}/{record['run_id']}/{record['id']}",
                                      "project": None})

    def handoff_source(self, session_id, run_id, source_id):
        """Host-only same-task material handoff; native current-run checks stay strict."""
        original = self.get_source(session_id, run_id, source_id)
        target = self.store.get_run(run_id)
        if original["task_id"] != target["task_id"] or original["session_id"] != session_id:
            raise ValueError("Source handoff requires the same session and task")
        if original["run_id"] == run_id:
            return original
        for existing in self.list_sources(session_id, run_id):
            if existing["run_id"] == run_id and existing.get("artifact_provenance", {}).get("handoff_source_id") == source_id:
                self._verify_source(existing)
                return existing
        content = self._verify_source(original)
        artifact = self.store.put_artifact(content, "handed-off-source", original["media_type"],
            {**original["artifact_provenance"], "handoff_source_id": source_id, "handoff_run_id": original["run_id"],
             "handoff_sha256": original["artifact_hash"]}, session_id=session_id, run_id=run_id)
        return self.register_source(session_id, run_id, artifact_id=artifact["id"], kind=original["kind"],
            title=original["title"], provenance=original["provenance"], idempotency_key="handoff:"+run_id+":"+source_id)

    def get_source(self, session_id, run_id, source_id):
        return self._read("source", session_id, run_id, source_id)

    def current_source_by_key(self, session_id, run_id, key):
        """Look up an immutable derived source in this run before recomputing it."""
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            row = conn.execute("SELECT source_id FROM research_source_keys WHERE session_id=? AND key_hash=?",
                               (session_id, digest(key))).fetchone()
            if row is None:
                return None
            source = self._get(conn, "source", session_id, task["id"], row["source_id"])
            return self.store.redact(source) if source["run_id"] == run_id else None

    def list_sources(self, session_id, run_id):
        return self._read_list("source", session_id, run_id)

    def get_evidence(self, session_id, run_id, evidence_id, *, revision=None):
        return self._read("evidence", session_id, run_id, evidence_id, revision)

    def list_evidence(self, session_id, run_id):
        return self._read_list("evidence", session_id, run_id)

    def select_evidence(self, session_id, run_id, source_id, selector):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            source = self._get(conn, "source", session_id, task["id"], source_id)
        selected = extract(self._verify_source(source), selector)
        record = self._record("evidence", session_id, run_id, task["id"], source_id=source_id,
                              source_hash=source["artifact_hash"], kind=source["kind"], **selected)
        with self.store.transaction(write=True) as conn:
            self._scope(conn, session_id, run_id)
            self._source_metadata(conn, source)
            record = self._save(conn, "evidence", record)
        return self._mirror("evidence", record)

    def reselect_evidence(self, session_id, run_id, evidence_id, *, expected_revision, selector):
        revision(expected_revision)
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            current = self._get(conn, "evidence", session_id, task["id"], evidence_id)
            source = self._get(conn, "source", session_id, task["id"], current["source_id"])
        if current["revision"] != expected_revision:
            raise ValueError("Stale evidence revision")
        selected = extract(self._verify_source(source), selector)
        record = {**current, **selected, "revision": expected_revision + 1,
                  "last_modified_run_id": run_id, "updated_at": datetime.now(timezone.utc).isoformat()}
        with self.store.transaction(write=True) as conn:
            self._scope(conn, session_id, run_id)
            latest = self._get(conn, "evidence", session_id, task["id"], evidence_id)
            if latest != current:
                raise ValueError("Stale evidence revision")
            self._source_metadata(conn, source)
            record = self._save(conn, "evidence", record)
        return self._mirror("evidence", record)

    def get_claim(self, session_id, run_id, claim_id, *, revision=None):
        return self._read("claim", session_id, run_id, claim_id, revision)

    def list_claims(self, session_id, run_id):
        return self._read_list("claim", session_id, run_id)

    def record_claim(self, session_id, run_id, *, text, kind, evidence_ids):
        claim_fields(text, kind, evidence_ids)
        with self.store.transaction(write=True) as conn:
            task = self._scope(conn, session_id, run_id)
            if kind == "observation" and task["acceptance"].get("require_literal_observations"):
                raise ValueError("Use claim_observe for literal observations; authored synthesis must remain a qualified proposal")
            for evidence_id in evidence_ids:
                self._get(conn, "evidence", session_id, task["id"], evidence_id)
            record = self._record("claim", session_id, run_id, task["id"],
                                  text=text, kind=kind, evidence_ids=list(evidence_ids), origin="authored")
            record = self._save(conn, "claim", record)
        return self._mirror("claim", record)

    def revise_claim(self, session_id, run_id, claim_id, *, expected_revision, text, kind, evidence_ids):
        revision(expected_revision)
        claim_fields(text, kind, evidence_ids)
        with self.store.transaction(write=True) as conn:
            task = self._scope(conn, session_id, run_id)
            if kind == "observation" and task["acceptance"].get("require_literal_observations"):
                raise ValueError("Use claim_observe for literal observations; authored synthesis must remain a qualified proposal")
            current = self._get(conn, "claim", session_id, task["id"], claim_id)
            if current["revision"] != expected_revision:
                raise ValueError("Stale claim revision")
            for evidence_id in evidence_ids:
                self._get(conn, "evidence", session_id, task["id"], evidence_id)
            record = {**current, "text": text, "kind": kind, "evidence_ids": list(evidence_ids),
                      "revision": expected_revision + 1, "origin": "authored", "last_modified_run_id": run_id,
                      "updated_at": datetime.now(timezone.utc).isoformat()}
            record = self._save(conn, "claim", record)
        return self._mirror("claim", record)
