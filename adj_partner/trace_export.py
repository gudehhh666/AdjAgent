"""One SQL snapshot, followed by separate verified immutable public-file copies.

Filesystem bytes are not part of SQLite's transaction. Copies are hash/size checked
against that snapshot, never rewritten to sanitize them. Unknown/private artifacts
remain references with explicit omission reasons. Free-text redaction is defensive,
not a guarantee that arbitrary text is safe to share.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from .contracts import new_id
from .tools.catalog import canonical_json
from .tools.policy import _credential_path


JSON_FIELDS = {"config", "acceptance", "budget", "usage", "state", "payload", "message", "provenance",
               "arguments", "result", "definition", "result_artifact", "args", "steps"}
KINDS = {"tasks": "task", "runs": "run", "messages": "message", "events": "event", "artifacts": "artifact",
         "actions": "action", "research_records": "research_record", "research_events": "research_event",
         "execution_jobs": "job", "execution_events": "job_event", "tool_approvals": "approval",
         "runtime_plans": "plan", "runtime_inputs": "input", "application_models": "model_preference"}
MEDIA = {"text/plain": ".txt", "text/markdown": ".md", "text/html": ".html", "application/json": ".json",
         "text/csv": ".csv", "text/tab-separated-values": ".tsv"}
PUBLIC_KINDS = {"public_model_request", "public_model_reply", "context_payload", "structured_research_report", "computed"}
MAX_ARTIFACT = 8 * 1024 * 1024
MAX_EXPORT_BYTES = 64 * 1024 * 1024


def _decode(row):
    value = dict(row)
    for key in JSON_FIELDS & value.keys():
        if value[key] is not None: value[key] = json.loads(value[key])
    value.pop("owner_id", None)
    value.pop("fingerprint", None)
    return value


def _strings(value):
    if isinstance(value, str): yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from _strings(item)
    elif isinstance(value, list):
        for item in value: yield from _strings(item)


def public_projection(value, depth=0):
    """Remove cached fragments even inside JSON-encoded messages/events.

    Complete records retain their identity and full-result references. Truncated
    fragments cannot be re-certified by matching a newly known whole secret;
    immutable files needing this projection must instead be omitted entirely.
    """
    if depth > 100: return {"export_projection_omitted": "Nested content exceeds the safe projection depth"}
    if isinstance(value, dict):
        if value.get("context_truncated") is True and "full_result_artifact_id" in value:
            identifier, digest = value.get("full_result_artifact_id"), value.get("sha256")
            result = {"context_truncated": True, "preview_omitted_for_export": True,
                "full_result_artifact_id": identifier if isinstance(identifier,str) and re.fullmatch(r"artifact_[A-Za-z0-9_-]{1,119}",identifier) else None,
                "sha256": digest if isinstance(digest,str) and re.fullmatch(r"[a-f0-9]{64}",digest) else None,
                "note": "Stored preview/error fragments omitted; consult the full artifact's export status. Immutable bytes were not rewritten."}
            status = value.get("status",value.get("recorded_status"))
            if isinstance(status,str) and status in {"ok","error","denied","cancelled","pending","running","waiting","approval_required","uncertain"}:
                result["recorded_status"] = status
            return result
        return {key: public_projection(item, depth+1) for key,item in value.items()}
    if isinstance(value,list): return [public_projection(item,depth+1) for item in value]
    if isinstance(value,str) and value.lstrip("\ufeff \t\r\n").startswith(("{","[")):
        try: parsed = json.loads(value.lstrip("\ufeff \t\r\n"))
        except (ValueError,RecursionError): return value
        changed = public_projection(parsed,depth+1)
        return canonical_json(changed) if changed != parsed else value
    return value


def snapshot(store, session_id, run_id=None):
    """Public extension-table snapshot; no nested service snapshot calls."""
    for identifier in (session_id, run_id):
        if identifier is not None and (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier)):
            raise ValueError("Invalid trace scope identifier")
    with store.transaction() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None: raise KeyError("Unknown session")
        result = {"session": _decode(row)}
        for table in ("tasks", "runs", "events", "messages", "artifacts", "research_records", "research_events", "execution_jobs", "tool_approvals", "application_models"):
            result[table] = [_decode(row) for row in conn.execute(f"SELECT * FROM {table} WHERE session_id=? ORDER BY rowid", (session_id,))]
        for table in ("actions", "runtime_plans", "runtime_inputs"):
            result[table] = [_decode(row) for row in conn.execute(
                f"SELECT t.* FROM {table} t JOIN runs r ON t.run_id=r.id WHERE r.session_id=? ORDER BY t.rowid", (session_id,))]
        result["execution_events"] = [_decode(row) for row in conn.execute("""SELECT e.* FROM execution_events e
            JOIN execution_jobs j ON e.job_id=j.id WHERE j.session_id=? ORDER BY e.seq""", (session_id,))]
    if run_id is not None:
        run = next((row for row in result["runs"] if row["id"] == run_id), None)
        if run is None: raise ValueError("Run is not in this session")
        result["tasks"] = [row for row in result["tasks"] if row["id"] == run["task_id"]]
        result["runs"] = [run]
        for table in ("events", "messages", "actions", "execution_jobs", "tool_approvals", "runtime_plans", "runtime_inputs"):
            result[table] = [row for row in result[table] if row["run_id"] == run_id]
        jobs = {row["id"] for row in result["execution_jobs"]}
        result["execution_events"] = [row for row in result["execution_events"] if row["job_id"] in jobs]
        records = result["research_records"]
        selected = {row["id"] for row in records if row["run_id"] == run_id or
                    row["entity"] in {"claim", "assessment", "report", "validation"} and row["task_id"] == run["task_id"]}
        # Same-session provenance closure includes all versions of referenced
        # prior material; unrelated task conversation/history is not exported.
        by_id = {}
        for record in records: by_id.setdefault(record["id"], []).append(record)
        pending = list(selected)
        while pending:
            identifier = pending.pop()
            references = set(_strings(by_id[identifier])) & by_id.keys() - selected
            selected.update(references)
            pending.extend(references)
        result["research_records"] = [row for row in records if row["id"] in selected]
        result["research_events"] = [row for row in result["research_events"] if row["run_id"] == run_id or row["payload"].get("id") in selected]
        references = set(_strings({key: rows for key, rows in result.items() if key != "artifacts"}))
        result["artifacts"] = [row for row in result["artifacts"] if row["run_id"] == run_id or row["id"] in references]
    canonical_json(result)
    return result


def _public_reason(artifact, successful_jobs):
    provenance = artifact["provenance"]
    if _credential_path(Path(artifact["name"])): return "credential_filename"
    if not isinstance(provenance, dict): return "malformed_artifact_classification"
    for key in ("visibility", "kind", "text_availability", "provider", "job_id", "role"):
        if provenance.get(key) is not None and not isinstance(provenance[key], str):
            return "malformed_artifact_classification"
    if provenance.get("private") is not None and type(provenance["private"]) is not bool:
        return "malformed_artifact_classification"
    if provenance.get("visibility") == "private" or provenance.get("private") is True: return "private_artifact"
    if provenance.get("text_availability") == "userfile" or provenance.get("kind") in {"userfile", "raw_userfile"}: return "raw_userfile"
    if artifact["media_type"] not in MEDIA: return "nontext_artifact"
    if type(artifact["size"]) is not int or not 0 <= artifact["size"] <= MAX_ARTIFACT: return "artifact_size_limit"
    public = provenance.get("visibility") == "public" or provenance.get("kind") in PUBLIC_KINDS
    public = public or provenance.get("provider") == "Europe PMC" and provenance.get("text_availability") in {"title_only", "abstract", "full_text"}
    public = public or provenance.get("job_id") in successful_jobs and (provenance.get("text_availability") == "computed" or provenance.get("role") == "job_result")
    return None if public else "unclassified_or_raw_private_artifact"


def _safe_directory(path):
    if ".." in path.parts or _credential_path(path) or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise PermissionError("Trace destination cannot use traversal, symlinks or credential paths")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.resolve() != path.absolute(): raise PermissionError("Trace destination changed")
    return path.absolute()


def _write(path, content):
    # Fresh export directories and exclusive files never overwrite prior traces.
    if path.parent.resolve() != path.parent: raise PermissionError("Export directory changed")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def export_trace(store, session_id, destination=None, run_id=None):
    raw = snapshot(store, session_id, run_id)
    clean = store.redact(public_projection(raw))
    base = Path(destination).expanduser() if destination is not None else store.root / "traces" / (run_id or session_id)
    if not base.is_absolute(): base = store.root / base
    base = _safe_directory(base)
    directory = base / new_id("trace")
    directory.mkdir(mode=0o700)
    manifest = {"schema_version": 1, "session_id": session_id, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "snapshot_consistency": "single_sql_read_transaction",
        "artifact_consistency": "Post-snapshot I/O verified against snapshot hash/size; not a database/filesystem atomic transaction",
        "privacy": "Explicit public allowlist and current-secret checks; arbitrary text is not guaranteed secret-free. Immutable bytes are never rewritten.",
        "files": {}, "artifacts": []}
    remaining = MAX_EXPORT_BYTES
    successful = {job["id"] for job in raw["execution_jobs"] if job["status"] == "succeeded"}
    userfiles = {row["payload"]["artifact_id"] for row in raw["research_records"]
                 if row["entity"] == "source" and row["payload"].get("kind") == "userfile"}
    artifact_dir = directory / "artifacts"
    for artifact in raw["artifacts"]:
        record = store.redact(public_projection(artifact))
        reason = "raw_userfile_source" if artifact["id"] in userfiles else _public_reason(artifact, successful)
        if reason is None and artifact["size"] > remaining: reason = "aggregate_export_size_limit"
        if reason is None:
            # Reserve attempts, including corrupt/private results that cannot
            # be copied, so rejected artifacts cannot multiply the I/O budget.
            remaining -= artifact["size"]
            try:
                # Scope is derived only from the one session snapshot; later DB
                # metadata cannot substitute different bytes or a foreign object.
                content = store.read_artifact(artifact["id"], max_bytes=max(1, artifact["size"]))
                if len(content) != artifact["size"] or hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                    raise ValueError("Snapshot artifact identity changed")
                text = content.decode("utf-8")
                if public_projection(text) != text:
                    reason = "stored_preview_requires_export_projection"
                elif store.redact(text) != text or store.redact(artifact) != artifact:
                    reason = "current_secret_redaction_required"
                else:
                    artifact_dir.mkdir(exist_ok=True, mode=0o700)
                    relative = Path("artifacts") / (artifact["id"] + MEDIA[artifact["media_type"]])
                    _write(directory / relative, content)
                    record.update(export_status="copied", export_path=str(relative))
            except (ValueError, OSError, UnicodeError): reason = "integrity_encoding_or_read_failure"
        if reason is not None: record.update(export_status="omitted", reason=reason)
        manifest["artifacts"].append(record)
    records = [{"record_type": "session", "data": clean["session"]}]
    for table, kind in KINDS.items(): records.extend({"record_type": kind, "data": item} for item in clean[table])
    jsonl = ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")
    # JSON code blocks keep all model-authored strings visibly quoted; choose a
    # fence longer than any untrusted backtick run so text cannot forge sections.
    markdown = ["# Public session trace", "", manifest["artifact_consistency"], "", manifest["privacy"], ""]
    for record in records:
        rendered = json.dumps(record, ensure_ascii=True, allow_nan=False, indent=2)
        fence = "`" * max(3, 1 + max((len(value) for value in re.findall(r"`+", rendered)), default=0))
        markdown.extend([f"## {record['record_type']}", "", fence + "json", rendered, fence, ""])
    for name, content in (("jsonl", jsonl), ("markdown", "\n".join(markdown).encode("utf-8"))):
        filename = "trace.jsonl" if name == "jsonl" else "trace.md"
        _write(directory / filename, content)
        manifest["files"][name] = {"path": filename, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    _write(directory / "manifest.json", (canonical_json(manifest) + "\n").encode("utf-8"))
    return {"jsonl": str(directory / "trace.jsonl"), "markdown": str(directory / "trace.md"), "manifest": str(directory / "manifest.json")}
