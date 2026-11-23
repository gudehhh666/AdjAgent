"""Bounded model projections; complete host snapshot/events remain separate."""

import json

from .assessments import qualifications
from .validation import bounded_json, digest


def summary(entity, record):
    result = {"entity": entity, **{key: record[key] for key in
               ("id", "revision", "run_id", "task_id", "created_at")}, "truncated_fields": [],
              "detail_retained_in_host_ledger": True}

    def preview(key, value, limit=1200):
        if len(value) > limit:
            result["truncated_fields"].append(key)
        result[key] = value[:limit]

    def references(key, values):
        result[key] = values[:10]
        result[key + "_count"] = len(values)
        if len(values) > 10:
            result["truncated_fields"].append(key)

    for key in ("kind", "source_id", "source_hash", "artifact_id", "artifact_hash", "claim_id", "report_id",
                "report_hash", "status", "reason_code", "provisional", "accepted", "last_modified_run_id", "updated_at"):
        if key in record:
            result[key] = record[key]
    for key in ("title", "text", "rationale"):
        if isinstance(record.get(key), str):
            preview(key, record[key])
    if "evidence_ids" in record:
        references("evidence_ids", record["evidence_ids"])
    if entity == "source":
        q = qualifications(record)
        result["text_availability"] = q["text_availability"]
        result["source_limitation_count"] = len(q["limitations"])
        preview("source_limitations_preview", bounded_json(q["limitations"]))
    if entity == "evidence":
        preview("locator_preview", bounded_json(record["locator"]), 500)
        if record["text"] is None:
            preview("value_preview", bounded_json(record["value"]))
    if entity == "report":
        result.update(section_count=len(record["sections"]), claim_count=len(record["claim_snapshots"]), validation_needed=True)
        preview("section_headings_preview", bounded_json([s["heading"] for s in record["sections"]]))
    if "bindings" in record:
        result["bindings_hash"] = digest(record["bindings"])
    if entity in {"assessment", "validation"}:
        result.update(current_eligibility_checked=False, authority="recorded_outcome_not_handoff_permission")
    if entity == "validation":
        # All issues are bounded by the validated report record. Preserve them
        # on validation output; state pages expose a preview with explicit count.
        result["issue_count"] = len(record["issues"])
        preview("issues_preview", bounded_json(record["issues"]))
    return result


class StateViews:
    def state_page(self, session_id, run_id, *, cursor=0, limit=25):
        """Current-record summaries ordered by stable mutation sequence.

        A cursor is an exclusive research_records.seq boundary, not an offset.
        Concurrent new revisions appear at new sequence IDs; clients deduplicate
        by record ID+revision. Start at zero for a fresh complete current view.
        No history/event replay is copied into model pages. Host snapshot is full.
        """
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("State cursor must be nonnegative; limit must be 1..50")
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            rows = list(conn.execute("""SELECT r.seq,r.entity,r.payload FROM research_records r
                WHERE r.session_id=? AND (r.entity IN ('source','evidence') OR r.task_id=?)
                AND r.seq>? AND r.revision=(SELECT MAX(v.revision) FROM research_records v
                    WHERE v.entity=r.entity AND v.id=r.id)
                ORDER BY r.seq LIMIT ?""", (session_id, task["id"], cursor, limit + 1)))
        visible, more = rows[:limit], len(rows) > limit
        result = {"session_id": session_id, "run_id": run_id, "task_id": task["id"],
                  "records": [{"cursor": row["seq"], **summary(row["entity"], self.store.redact(json.loads(row["payload"])))} for row in visible],
                  "has_more": more, "next_cursor": visible[-1]["seq"] if more else None,
                  "projection": "current_record_summaries; complete detail/history retained in host ledger"}
        return self.store.redact(result)
