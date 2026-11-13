"""Literature retrieval: PubMed (E-utilities) and Europe PMC.

Both are free APIs (verified reachable). Each hit becomes an EvidenceRecord.
No embedding — plain keyword queries against the source's own search index.
"""
from __future__ import annotations

from typing import Any

import requests

from core.config import env, load_yaml
from core.ratelimit import throttle
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec

_UA = {"User-Agent": "Adj_Partner/0.1 (adjuvant science agent)"}
# NCBI: 3 req/s without key, 10 req/s with key. Europe PMC is generous.
_NCBI_INTERVAL_NO_KEY = 0.34
_NCBI_INTERVAL_KEY = 0.11


def _cfg(*keys: str, default: Any = None) -> Any:
    node: Any = load_yaml("sources.yaml")
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def search_pubmed(query: str, max_results: int = 8) -> list[EvidenceRecord]:
    base = _cfg("literature", "pubmed", "base_url",
                default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    api_key = env("ADJ_NCBI_API_KEY")
    common = {"db": "pubmed", "retmode": "json"}
    if api_key:
        common["api_key"] = api_key
    interval = _NCBI_INTERVAL_KEY if api_key else _NCBI_INTERVAL_NO_KEY

    throttle("ncbi", interval)
    esearch = requests.get(
        f"{base}/esearch.fcgi",
        params={**common, "term": query, "retmax": max_results},
        headers=_UA, timeout=20,
    )
    esearch.raise_for_status()
    ids = esearch.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    throttle("ncbi", interval)
    summ = requests.get(
        f"{base}/esummary.fcgi",
        params={**common, "id": ",".join(ids)},
        headers=_UA, timeout=20,
    )
    summ.raise_for_status()
    result = summ.json().get("result", {})

    records: list[EvidenceRecord] = []
    for pid in ids:
        item = result.get(pid, {})
        if not item:
            continue
        title = item.get("title", "").strip()
        journal = item.get("fulljournalname", "") or item.get("source", "")
        year = (item.get("pubdate", "") or "")[:4]
        records.append(EvidenceRecord(
            claim=title,
            source_type="literature",
            source_id=f"PMID:{pid}",
            snippet=f"{title} — {journal} ({year})",
            date=year,
            authority_weight=0.8,
            extra={"url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/", "journal": journal},
        ))
    return records


def search_europepmc(query: str, max_results: int = 8) -> list[EvidenceRecord]:
    base = _cfg("literature", "europe_pmc", "base_url",
                default="https://www.ebi.ac.uk/europepmc/webservices/rest")
    resp = requests.get(
        f"{base}/search",
        params={"query": query, "format": "json", "pageSize": max_results,
                "resultType": "core"},
        headers=_UA, timeout=20,
    )
    resp.raise_for_status()
    hits = resp.json().get("resultList", {}).get("result", [])

    records: list[EvidenceRecord] = []
    for h in hits:
        title = (h.get("title") or "").strip()
        abstract = (h.get("abstractText") or "")[:400]
        year = str(h.get("pubYear", ""))
        src = h.get("source", "")
        ext_id = h.get("id", "")
        pmid = h.get("pmid")
        sid = f"PMID:{pmid}" if pmid else f"{src}:{ext_id}"
        records.append(EvidenceRecord(
            claim=title,
            source_type="literature",
            source_id=sid,
            snippet=abstract or title,
            date=year,
            authority_weight=0.75,
            extra={"citedByCount": h.get("citedByCount", 0)},
        ))
    return records


def search_literature(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Combined literature search across enabled sources; returns dicts."""
    records: list[EvidenceRecord] = []
    if _cfg("literature", "pubmed", "enabled", default=True):
        try:
            records += search_pubmed(query, max_results)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully per source
            records.append(EvidenceRecord(
                claim=f"[pubmed error] {exc}", source_type="literature",
                source_id="error", authority_weight=0.0))
    if _cfg("literature", "europe_pmc", "enabled", default=True):
        try:
            records += search_europepmc(query, max_results)
        except Exception as exc:  # noqa: BLE001
            records.append(EvidenceRecord(
                claim=f"[europepmc error] {exc}", source_type="literature",
                source_id="error", authority_weight=0.0))
    return [r.to_dict() for r in records]


REGISTRY.register(
    ToolSpec(
        name="search_literature",
        description="Search biomedical literature (PubMed + Europe PMC) for a query; "
                    "returns EvidenceRecord dicts with title, PMID, year.",
        domain="retrieval",
        parameters={"query": "search string", "max_results": "per-source cap (int)"},
        returns="list[EvidenceRecord as dict]",
    ),
    search_literature,
)
