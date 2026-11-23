"""Trusted guidance and tool-name defaults, not domain-owned implementations."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import time

from .storage import TERMINAL_STATUSES
from .tools.catalog import canonical_json
from .tools.policy import _credential_path
from .runtime.journal import RunBusy


ANALYSIS = ["code_interpreter", "bash", "analysis_method_register", "analysis_method_run",
            "analysis_history", "analysis_history_read", "public_code_check"]
COMMON = ["catalog_search", "tools_enable", "ask_user", "update_plan", "file_inspect", "artifact_read", "literature_search", "article_resolve", "fulltext_fetch", "document_read", "web_search",
          "chembl_compound", "chembl_activity", "reactome_pathway", "geo_search",
          "job_status", "research_state", "evidence_select", "evidence_reselect", "claim_record", "claim_revise",
          "claim_observe", "claim_assess", "research_critic", "report_create", "report_validate"] + ANALYSIS
PROFILES = {
    "general": {"instructions": "Assist naturally. Use tools only when useful; do not impose a fixed research workflow.", "tools": COMMON},
    "adjuvant": {"instructions": "For adjuvant questions, separate transcriptomic association, mechanistic hypothesis and validated vaccine benefit. "
        "Preserve ageing, vehicle and shared-control limitations where applicable. Use actual replicate and pooling metadata; missing replication details remain unknown. DEG is only the host-configured reviewed experiment. "
        "For candidate prioritization, use deg_analyze summary.top_centered or positive_reversers ordered by centered_reversal_r; "
        "immune-module scores are descriptive and must not replace that ranking. Use deg_explore for candidate-specific genes and pathway terms. "
        "If exact values are unavailable, call the appropriate data tool or report the missing evidence; do not weaken the prompt or invent a result. "
        "Use web_search for broad discovery and literature_search for available abstracts where relevant; search snippets are not page or paper text. "
        "Retain abstract/title availability limits, and keep proposed formulations explicit as proposals.",
        "tools": COMMON + ["deg_analyze", "deg_explore", "adjuvant_search", "adjuvant_record", "adjuvant_relations",
                           "design_record_read", "experiment_state", "experiment_design_revise"]},
    "protein": {"instructions": "For protein questions, compute available sequence/file properties with real tools. "
        "Sequence composition, theoretical properties and parsed coordinates do not establish function, affinity, safety or efficacy. "
        "No structure-prediction/docking backend is configured by this profile.",
        "tools": COMMON + ["protein_sequence", "protein_fasta", "protein_structure"]},
}


PROFILES['adjuvant-evidence'] = {
    'instructions': PROFILES['adjuvant']['instructions'] +
        ' At the point where evidence is needed, read adjuvant_skill once for the relevant need: '
        'paper (specific article facts), mechanism (protein/reaction), assay (protocol/QC), '
        'expression (cell measurements vs catalog), clinical (registry), or table (exact cells). '
        'Use short entity queries in the indicated dataset; after a narrow miss ranked discovery may help. '
        'Read exact returned IDs/fields before citing; use adjuvant_excerpt for long string fields. '
        'Two searches with no relevant new evidence should end that local gap search; report the gap or use authorized literature tools. '
        'No need to run every skill or repeat review rounds.',
    'tools': PROFILES['adjuvant']['tools'] + ['adjuvant_skill','adjuvant_excerpt'],
}


PROFILES['adjuvant-resource'] = {
    'instructions': PROFILES['adjuvant']['instructions'] +
        ' For resource evidence first use adjuvant_resource_guide for the relevant dataset and kind. '
        'It exposes actual typed fields and applicability limits. Do not guess field names. '
        'Use adjuvant_lookup to search and read in one step: query is a short all-word entity/readout anchor; '
        'focus separately selects the needed content (e.g. interference recovery); fields come from the guide. '
        'Prefer one best record and its returned literal fragments before opening more candidates. '
        'For protocols select assay_protocol_page and text; for raw workbooks first inspect kind/index and coordinates. '
        'Copy source_id and selector exactly; fragments may point to nested leaf values. '
        'Retrieved metadata can explain a gap but cannot replace the requested measurement. '
        'After two local misses preserve the gap and finish or use separately authorized literature tools. '
        'Reserve output budget; no mandatory multi-round review. Use adjuvant_skill for additional domain-specific interpretation.',
    'tools': [x for x in PROFILES['adjuvant-evidence']['tools'] if x != 'adjuvant_search'] +
        ['adjuvant_resource_guide','adjuvant_lookup'],
}


PROFILES['adjuvant-grounded'] = {
    'instructions': PROFILES['adjuvant-resource']['instructions'] +
        ' For a known record ID and original field, read directly with adjuvant_cite; no guide or repeated search is needed. '
        'For long text use a short focus, then copy the returned citation object exactly. '
        'A citation is only literal source provenance, not claim support. Never cite wrapper keys, retrieval gaps, or joined fragments. '
        'If a specific paper result was not read, report that result as unknown; do not assert its direction then append a missing-source disclaimer.',
    'tools': PROFILES['adjuvant-resource']['tools'] + ['adjuvant_cite'],
}


PROFILES['adjuvant-batch'] = {
    'instructions': PROFILES['adjuvant-grounded']['instructions'],
    'tools': PROFILES['adjuvant-grounded']['tools'] + ['adjuvant_read_fields'],
}

PROFILES['adjuvant-routed'] = {
    'instructions': PROFILES['adjuvant']['instructions'] +
        ' Choose the shortest evidence path for the actual task. '
        'Known record ID and fields: call adjuvant_read_fields once for all requested short fields; '
        'for a long original text field use adjuvant_cite with a task-specific focus. '
        'No guide or search is needed before reading known IDs/fields. '
        'Unknown record identity: inspect only the relevant resource guide if needed, then adjuvant_lookup with '
        'a short entity anchor and exact typed filters. Use returned IDs; never derive row IDs from external identifiers. '
        'An authorized original file can be read directly with file/data tools; do not search unrelated databases for it. '
        'Check complete/omitted_fields and retain original types. Long field omitted: use focused cite, not repeated full-artifact paging. '
        'After two distinct local searches without relevant evidence, report that local evidence gap; '
        'do not repeat the same query across unrelated databases. This is not evidence of biological absence. '
        'Copy returned citations exactly; source existence is not claim support. Never cite wrapper fields or gap status. '
        'For paper-specific findings, state what was actually read; missing results remain unknown, without guessed directions. '
        'Designs are proposals: identify decision-changing readouts, controls and support/reject/undetermined branches. '
        'Use a relevant evidence skill only when interpretation needs it. No mandatory review rounds.',
    'tools': PROFILES['adjuvant-batch']['tools'][:],
}


def get_profile(name):
    if not isinstance(name, str) or name not in PROFILES: raise ValueError("Unknown profile")
    return deepcopy(PROFILES[name])


def apply_profile(store, catalog, session_id, name):
    definition = get_profile(name)
    with store.transaction(write=True) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None: raise KeyError("Unknown session")
        runs = conn.execute("SELECT status FROM runs WHERE session_id=?", (session_id,)).fetchall()
        # This atomic writer boundary excludes a concurrent RunJournal.start.
        if any(run[0] not in TERMINAL_STATUSES for run in runs): raise RunBusy("Change profiles between completed/cancelled runs")
        config = json.loads(row["config"])
        config["profile"] = name
        config["enabled_tools"] = sorted(set(definition["tools"]) & set(config.get("allowed_tools", [])) & set(catalog.names()))
        conn.execute("UPDATE sessions SET config=?,revision=revision+1 WHERE id=?", (canonical_json(store.redact(config)), session_id))
        conn.execute("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,NULL,?,?,?)",
            (session_id, "profile.changed", canonical_json({"profile": name, "revision": row["revision"] + 1}),
             datetime.now(timezone.utc).isoformat()))
    return store.get_session(session_id)


def load_trusted_guidance(path):
    """Host-only explicit file selection; no discovery, interpolation or execution."""
    selected = Path(path).expanduser().absolute()
    if ".." in selected.parts or _credential_path(selected) or selected.resolve() != selected:
        raise PermissionError("Trusted guidance must be an explicit non-credential, non-symlink file")
    descriptor = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1: raise PermissionError("Guidance requires a regular non-aliased file")
        if before.st_size > 65536: raise ValueError("Guidance exceeds64KiB")
        with os.fdopen(descriptor, "rb", closefd=False) as handle: data = handle.read(65537)
        after = os.fstat(descriptor)
        # Reads may change atime. Compare content/identity/security metadata only.
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        changed = any(getattr(before, key) != getattr(after, key) for key in stable)
        if (len(data) > 65536 or changed or selected.resolve() != selected
                or selected.stat().st_ino != before.st_ino): raise ValueError("Guidance changed or exceeded64KiB while reading")
        return data.decode("utf-8-sig")
    finally:
        os.close(descriptor)
