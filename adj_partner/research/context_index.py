"""Small, task-scoped reference projection; never a replacement evidence ledger."""

import hashlib
import json
import re

from ..tools.catalog import canonical_json


def reference_index(store, session_id, task_id):
    with store.transaction() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='research_records'").fetchone():
            return None

        def latest(entity, count, extra=""):
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT r.payload FROM research_records r WHERE r.session_id=? AND r.task_id=? "
                "AND r.entity=? AND r.revision=(SELECT MAX(v.revision) FROM research_records v "
                "WHERE v.session_id=r.session_id AND v.task_id=r.task_id AND v.entity=r.entity AND v.id=r.id) "
                + extra + " ORDER BY r.seq DESC LIMIT ?", (session_id, task_id, entity, count))]

        task = conn.execute("SELECT acceptance FROM tasks WHERE id=? AND session_id=?", (task_id, session_id)).fetchone()
        expanded = bool(task and json.loads(task[0]).get("min_distinct_sources", 0) > 4)
        source_limit = 8 if expanded else 4
        claims = latest("claim", 13)
        evidence = latest("evidence", 9)
        source_scan_limit = 64 if expanded else source_limit + 1
        candidates = latest("source", source_scan_limit, "AND json_extract(r.payload,'$.provenance.text_availability') "
                         "IN ('abstract','derived_full_text','full_text')")
        sources, seen = [], set()
        for s in candidates:
            p = s["provenance"]
            key = ("doi", re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(p["doi"]).strip().casefold())) if p.get("doi") else (
                ("pmid", str(p["pmid"])) if p.get("pmid") else ("artifact", s["artifact_hash"]))
            if expanded and key in seen:
                continue
            seen.add(key)
            sources.append(s)
        sizes = {s["artifact_id"]: conn.execute("SELECT size FROM artifacts WHERE id=? AND session_id=?",
                 (s["artifact_id"], session_id)).fetchone() for s in sources[:source_limit]}
    result = {"claims": [], "evidence": [], "sources": [],
              "claims_omitted": len(claims) > 12, "evidence_omitted": len(evidence) > 8,
              "sources_omitted": len(sources) > source_limit or len(candidates) == source_scan_limit,
              "note": "Read-only current-task IDs, not scientific approval. IDs are complete; prose previews may be truncated. "
                      "Source IDs cannot be used as evidence IDs. Select real evidence before recording source-backed claims. "
                      "Use this index instead of scanning an entire source catalogue to recover IDs."}
    for c in claims[:12]:
        result["claims"].append({"id": c["id"], "revision": c["revision"], "kind": c["kind"],
            "text_preview": c["text"][:100], "evidence_ids": c["evidence_ids"][:4],
            "evidence_ids_omitted": len(c["evidence_ids"]) > 4})
    for e in evidence[:8]:
        result["evidence"].append({"id": e["id"], "source_id": e["source_id"], "revision": e["revision"]})
    for s in sources[:source_limit]:
        entry = {"id": s["id"], "artifact_id": s["artifact_id"], "title_preview": s["title"][:100],
                 "text_availability": s["provenance"]["text_availability"]}
        # Only bounded ingested JSON is inspected for an ACTUAL abstract selector.
        # Larger/fulltext documents remain behind their existing reading tools.
        size = sizes[s["artifact_id"]]
        if s["media_type"] == "application/json" and size and size[0] <= 32768:
            raw = store.read_artifact(s["artifact_id"])
            if hashlib.sha256(raw).hexdigest() == s["artifact_hash"]:
                try:
                    parsed = json.loads(raw)
                    abstract = parsed.get("abstract", []) if isinstance(parsed, dict) else []
                    if isinstance(abstract, str) and abstract.strip():
                        entry.update(selector={"pointer": "/abstract"}, abstract_preview=abstract[:200])
                    elif isinstance(abstract, list):
                        for index, part in enumerate(abstract):
                            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                                entry.update(selector={"pointer": f"/abstract/{index}/text"},
                                             abstract_preview=part["text"][:200])
                                break
                except (ValueError, UnicodeError):
                    pass
        result["sources"].append(entry)
    # Drop whole entries, never shorten identifiers. Explicit flags preserve gaps.
    while len(canonical_json(result).encode()) > 8000:
        for key in ("sources", "claims", "evidence"):
            if result[key]:
                result[key].pop()
                result[key + "_omitted"] = True
                break
    return result


def research_progress(run, references):
    """Advisory stages from recorded budgets and a bounded index, not quality gates."""
    budget, usage, state = run.get('budget', {}), run.get('usage', {}), run.get('state', {})
    ratios = []
    for used, cap in [(state.get('steps', 0), budget.get('max_steps')),
                      (usage.get('accounted_input_tokens', usage.get('input_tokens', 0)), budget.get('max_input_tokens')),
                      (usage.get('accounted_output_tokens', usage.get('output_tokens', 0)), budget.get('max_output_tokens')),
                      (state.get('elapsed_active_seconds', 0), budget.get('wall_time_seconds'))]:
        if isinstance(cap, (int, float)) and cap > 0:
            ratios.append(used / cap)
    fraction = max(ratios, default=0)
    refs = references or {}
    if fraction >= 0.6:
        stage = 'synthesize_or_report_gaps'
        guidance = ('Reserve remaining budget for decisions, research_design, report_create and validation. '
                    'Read only decision-critical missing evidence; avoid new broad searches. '
                    'If key questions remain unanswered, record open requirements and blocking_gaps honestly.')
    elif refs.get('evidence') and not refs.get('claims'):
        stage = 'record_claims'
        guidance = 'Turn already selected evidence into qualified observations/proposals before collecting more material.'
    elif refs.get('sources') and not refs.get('evidence'):
        stage = 'select_evidence'
        guidance = 'Select relevant literal evidence from available readable sources before more broad searches; titles alone are insufficient.'
    else:
        stage = 'resolve_decision_gaps'
        guidance = 'Use the requirement/candidate plan to identify the next decision-critical gap; compare evidence and alternatives.'
    return {'next_stage': stage, 'budget_fraction_used': round(fraction, 4),
            'scientific_adequacy': 'not_assessed',
            'guidance': guidance + ' This is not a minimum-source quality verdict. Never invent evidence or bypass a refusal.'}
