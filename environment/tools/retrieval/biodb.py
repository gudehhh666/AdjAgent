"""Biological database retrieval: UniProt and KEGG (free REST, verified).

Turns entity/pathway lookups into EvidenceRecords so they fuse with literature
and data-driven evidence uniformly.
"""
from __future__ import annotations

from typing import Any

import requests

from core.config import load_yaml
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec

_UA = {"User-Agent": "Adj_Partner/0.1 (adjuvant science agent)"}


def _cfg(*keys: str, default: Any = None) -> Any:
    node: Any = load_yaml("sources.yaml")
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def search_uniprot(query: str, max_results: int = 5) -> list[EvidenceRecord]:
    base = _cfg("biodb", "uniprot", "base_url", default="https://rest.uniprot.org")
    resp = requests.get(
        f"{base}/uniprotkb/search",
        params={"query": query, "format": "json", "size": max_results,
                "fields": "accession,id,protein_name,gene_names,organism_name,cc_function"},
        headers=_UA, timeout=20,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    records: list[EvidenceRecord] = []
    for r in results:
        acc = r.get("primaryAccession", "")
        prot = (r.get("proteinDescription", {})
                 .get("recommendedName", {})
                 .get("fullName", {}).get("value", ""))
        genes = [g.get("geneName", {}).get("value", "")
                 for g in r.get("genes", []) if g.get("geneName")]
        func = ""
        for c in r.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    func = texts[0].get("value", "")
                break
        records.append(EvidenceRecord(
            claim=f"{prot} ({', '.join(genes)})" if genes else prot,
            source_type="biodb",
            source_id=f"UniProt:{acc}",
            snippet=func[:400] or prot,
            entities=genes,
            authority_weight=0.85,
            extra={"url": f"https://www.uniprot.org/uniprotkb/{acc}"},
        ))
    return records


def search_kegg(query: str, max_results: int = 5) -> list[EvidenceRecord]:
    """KEGG text find over the pathway database."""
    base = _cfg("biodb", "kegg", "base_url", default="https://rest.kegg.jp")
    resp = requests.get(f"{base}/find/pathway/{query}", headers=_UA, timeout=20)
    resp.raise_for_status()
    records: list[EvidenceRecord] = []
    for line in resp.text.strip().splitlines()[:max_results]:
        if "\t" not in line:
            continue
        pid, desc = line.split("\t", 1)
        records.append(EvidenceRecord(
            claim=desc.strip(),
            source_type="biodb",
            source_id=pid.strip(),
            snippet=desc.strip(),
            authority_weight=0.8,
            extra={"url": f"https://www.kegg.jp/entry/{pid.strip().split(':')[-1]}"},
        ))
    return records


def search_biodb(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    records: list[EvidenceRecord] = []
    if _cfg("biodb", "uniprot", "enabled", default=True):
        try:
            records += search_uniprot(query, max_results)
        except Exception as exc:  # noqa: BLE001
            records.append(EvidenceRecord(claim=f"[uniprot error] {exc}",
                           source_type="biodb", source_id="error", authority_weight=0.0))
    if _cfg("biodb", "kegg", "enabled", default=True):
        try:
            records += search_kegg(query, max_results)
        except Exception as exc:  # noqa: BLE001
            records.append(EvidenceRecord(claim=f"[kegg error] {exc}",
                           source_type="biodb", source_id="error", authority_weight=0.0))
    return [r.to_dict() for r in records]


REGISTRY.register(
    ToolSpec(
        name="search_biodb",
        description="Look up genes/proteins/pathways in UniProt + KEGG; "
                    "returns EvidenceRecord dicts with function annotations.",
        domain="retrieval",
        parameters={"query": "gene/protein/pathway term", "max_results": "per-source cap (int)"},
        returns="list[EvidenceRecord as dict]",
    ),
    search_biodb,
)
