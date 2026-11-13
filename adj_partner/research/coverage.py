"""Conservative bibliographic coverage, not independence or semantic assurance.

Only cited readable literature/web material counts. Shared identifiers, normalized
full titles or identical artifacts merge transitively; unavailable metadata may
still hide duplicates, and distinct works may reuse the same study population.
"""

import re
import unicodedata

from .validation import digest


def _aliases(source):
    provenance = source.get("provenance", {})
    aliases = set()
    for field in ("doi", "pmid", "pmcid"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip().casefold()
        if field == "doi":
            value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
        aliases.add((field, value))
    title = unicodedata.normalize("NFKC", source.get("title", "")).casefold()
    title = "".join(c for c in title if c.isalnum())
    if title:
        aliases.add(("title", title))
    if source.get("artifact_hash"):
        aliases.add(("artifact_hash", source["artifact_hash"]))
    return aliases


def evidence_coverage(report):
    sources, evidence = {}, {}
    for snapshot in report["claim_snapshots"]:
        sources.update((s["id"], s) for s in snapshot["sources"])
        evidence.update((e["id"], e) for e in snapshot["evidence"])
    cited = {e["source_id"] for e in evidence.values()}
    eligible = {key: s for key, s in sources.items() if key in cited
                and s["kind"] in {"literature", "web"}
                and s.get("provenance", {}).get("text_availability")
                in {"abstract", "full_text", "derived_full_text"}}
    parent = {key: key for key in eligible}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for key, source in eligible.items():
        for alias in _aliases(source):
            if alias in seen:
                parent[find(key)] = find(seen[alias])
            else:
                seen[alias] = key
    excerpts = set()
    for e in evidence.values():
        if e["source_id"] not in eligible:
            continue
        value = e.get("text") if e.get("text") is not None else e.get("value")
        if isinstance(value, str):
            value = " ".join(value.split())
        if value is not None and value != "":
            excerpts.add((find(e["source_id"]), digest(value)))
    works = {work for work, _ in excerpts}
    return {"distinct_sources": len(works), "distinct_evidence": len(excerpts),
            "source_groups": {key: find(key) for key in eligible},
            "scope": "Bibliographic deduplication only; not independent-study or semantic verification."}
