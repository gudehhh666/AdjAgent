"""Bounded current UI projection, not a journal or a replacement for full trace.

One read transaction selects event identities after a cursor and current action/
job metadata. Ordinary event payloads and full action/job results never cross the
SQLite/Python boundary. Oversize progress/errors are omitted before loading;
bounded complete values are current-secret-redacted before display truncation.
"""

import json
import re

from .tools.catalog import canonical_json


RAW_LIMIT = 16384
PREVIEW_LIMIT = 2048
LABEL_LIMIT = 1024


def _payload(store, raw, size):
    if size is None: return None
    if raw is None: return {"omitted": True, "reason": "activity_payload_read_limit", "original_bytes": size}
    try: value = store.redact(json.loads(raw))
    except (TypeError, ValueError, RecursionError): return {"omitted": True, "reason": "invalid_activity_payload"}
    rendered = canonical_json(value)
    if len(rendered) <= PREVIEW_LIMIT: return value
    return {"preview": rendered[:PREVIEW_LIMIT], "truncated": True,
            "reason": "activity_display_limit", "original_bytes": size}


def activity_after(store, session_id, run_id, after_seq=0, limit=64):
    """Read same-run events/actions and same-session active-job metadata.

    Cursor is the last included event seq (unchanged for an empty page). Events
    are ascending and never silently skipped. Actions are the latest100 in
    creation order, with actions_truncated; jobs are the latest3 active jobs in
    creation order plus the exact active count. Full history remains in trace.
    """
    for identifier in (session_id, run_id):
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier):
            raise ValueError("Invalid activity scope identifier")
    if type(after_seq) is not int or not 0 <= after_seq <= 9223372036854775807:
        raise ValueError("Activity cursor must be a nonnegative SQLite sequence")
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("Activity page limit must be an integer from1 to128")
    with store.transaction() as conn:
        if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
            raise KeyError("Unknown session")
        run = conn.execute("SELECT session_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if run is None: raise KeyError("Unknown run")
        if run[0] != session_id: raise ValueError("Run does not belong to session")
        events = conn.execute("""SELECT seq,
            CASE WHEN length(CAST(kind AS BLOB))<=? THEN kind END AS kind,
            CASE WHEN kind='job.progress' THEN length(CAST(payload AS BLOB)) END AS payload_size,
            CASE WHEN kind='job.progress' AND length(CAST(payload AS BLOB))<=? THEN payload END AS payload
            FROM events WHERE session_id=? AND run_id=? AND seq>? ORDER BY seq LIMIT ?""",
            (LABEL_LIMIT, RAW_LIMIT, session_id, run_id, after_seq, limit)).fetchall()
        actions = conn.execute("""WITH recent AS (
            SELECT rowid AS record_seq,call_id,tool_name,status,finished_at,result
                FROM actions WHERE run_id=? ORDER BY rowid DESC LIMIT 101
            ), projected AS (
            SELECT record_seq,call_id,tool_name,status,finished_at,
                CASE WHEN json_valid(result) THEN json_extract(result,'$.error') END AS error FROM recent
            ) SELECT call_id,CASE WHEN length(CAST(tool_name AS BLOB))<=? THEN tool_name END AS tool_name,
                status,finished_at,length(CAST(error AS BLOB)) AS error_size,
                CASE WHEN length(CAST(error AS BLOB))<=? THEN error END AS error
                FROM projected ORDER BY record_seq DESC""",
            (run_id, LABEL_LIMIT, RAW_LIMIT)).fetchall()
        active_count = conn.execute("""SELECT COUNT(*) FROM execution_jobs
            WHERE session_id=? AND status IN ('queued','running','cancel_requested')""", (session_id,)).fetchone()[0]
        jobs = conn.execute("""SELECT CASE WHEN length(CAST(tool_name AS BLOB))<=? THEN tool_name END AS tool_name,status
            FROM execution_jobs WHERE session_id=? AND status IN ('queued','running','cancel_requested')
            ORDER BY created_at DESC,rowid DESC LIMIT 3""", (LABEL_LIMIT, session_id)).fetchall()
    label = "[omitted: activity_label_read_limit]"
    public_events = []
    for row in events:
        item = {"seq": row["seq"], "kind": store.redact(row["kind"] or label)}
        if row["payload_size"] is not None: item["payload"] = _payload(store, row["payload"], row["payload_size"])
        public_events.append(item)
    public_actions = []
    for row in reversed(actions[:100]):
        item = {key: row[key] for key in ("call_id", "tool_name", "status", "finished_at")}
        item["tool_name"] = item["tool_name"] or label
        item = store.redact(item)
        item["error"] = _payload(store, row["error"], row["error_size"])
        public_actions.append(item)
    return {"events": public_events, "next_cursor": events[-1]["seq"] if events else after_seq,
            "actions": public_actions, "actions_truncated": len(actions)>100,
            "active_jobs": [store.redact({"tool_name": row["tool_name"] or label, "status": row["status"]}) for row in reversed(jobs)],
            "active_job_count": active_count}
