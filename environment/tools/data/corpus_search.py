"""Local corpus full-text search (SQLite FTS5). No embedding by design.

PDFs are extracted page-by-page into an FTS5 index; queries return matching
passages as EvidenceRecords. Rebuild the index with build_index().
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from core.config import load_yaml, resolve_path
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec


def _index_path() -> Path:
    rel = load_yaml("sources.yaml").get("local_corpus", {}).get(
        "index_path", "environment/corpus_index/corpus.db")
    p = resolve_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _iter_pdf_paths() -> list[Path]:
    cfg = load_yaml("sources.yaml").get("local_corpus", {})
    paths: list[Path] = []
    for entry in cfg.get("pdf_dirs", []):
        p = resolve_path(entry)
        if p.is_dir():
            paths += sorted(p.glob("*.pdf"))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            paths.append(p)
    return paths


def _extract_pages(pdf_path: Path) -> list[str]:
    """Prefer PyMuPDF (robust); fall back to pypdf."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(pdf_path)) as doc:
            return [page.get_text() or "" for page in doc]
    except Exception:  # noqa: BLE001 — fall back
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                pages.append("")
        return pages
    except Exception:  # noqa: BLE001
        return []


def build_index() -> dict[str, Any]:
    db = _index_path()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE IF EXISTS passages")
    conn.execute(
        "CREATE VIRTUAL TABLE passages USING fts5(doc, page, text)")
    n_docs = n_pages = 0
    for pdf in _iter_pdf_paths():
        pages = _extract_pages(pdf)
        if not pages:
            continue
        n_docs += 1
        for i, text in enumerate(pages):
            if text.strip():
                conn.execute("INSERT INTO passages(doc, page, text) VALUES (?,?,?)",
                             (pdf.name, str(i + 1), text))
                n_pages += 1
    conn.commit()
    conn.close()
    return {"indexed_docs": n_docs, "indexed_pages": n_pages, "index": str(db)}


def search_corpus(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    db = _index_path()
    if not db.exists():
        info = build_index()
        if info["indexed_pages"] == 0:
            return [EvidenceRecord(claim="[local corpus empty] add PDFs and rebuild",
                    source_type="local_corpus", source_id="empty",
                    authority_weight=0.0).to_dict()]
    conn = sqlite3.connect(db)
    # FTS5 MATCH; fall back to gracefully handle special chars
    safe = " ".join(w for w in query.replace('"', " ").split() if w)
    try:
        rows = conn.execute(
            "SELECT doc, page, snippet(passages, 2, '[', ']', ' … ', 15) "
            "FROM passages WHERE passages MATCH ? LIMIT ?",
            (safe, max_results),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    records = [EvidenceRecord(
        claim=snip.strip()[:200], source_type="local_corpus",
        source_id=f"{doc}#p{page}", snippet=snip.strip(),
        authority_weight=0.6, extra={"doc": doc, "page": page})
        for doc, page, snip in rows]
    if not records:
        records = [EvidenceRecord(claim="[no local match]", source_type="local_corpus",
                   source_id="no_match", authority_weight=0.0)]
    return [r.to_dict() for r in records]


REGISTRY.register(
    ToolSpec(
        name="search_corpus",
        description="Full-text search the local PDF corpus (SQLite FTS5, no embedding). "
                    "Returns EvidenceRecord dicts with matching passages.",
        domain="retrieval",
        parameters={"query": "keywords", "max_results": "cap (int)"},
        returns="list[EvidenceRecord as dict]",
    ),
    search_corpus,
)
