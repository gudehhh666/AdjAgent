"""Read one PMC BioC article with original passage indexes and offsets."""

import json
import re

from ..tools.executor import ToolFailure


NON_BODY_SECTIONS = {"REF", "REFERENCES", "FIG", "TABLE", "SUPPL", "APPENDIX",
                     "AUTH_CONT", "ACK_FUND", "COMP_INT", "ABBR", "BACK"}


def parse_bioc(content, expected_pmcid=None):
    if not isinstance(content, bytes) or not 0 < len(content) <= 32 * 1024 * 1024:
        raise ToolFailure("fulltext_media", "BioC response is empty or exceeds its byte limit")
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ToolFailure("fulltext_media", "Provider did not return a BioC JSON article") from exc
    collections = payload if isinstance(payload, list) else [payload]
    if len(collections) != 1 or not isinstance(collections[0], dict):
        raise ToolFailure("fulltext_media", "Expected one BioC collection")
    docs = collections[0].get("documents")
    if not isinstance(docs, list) or len(docs) != 1 or not isinstance(docs[0], dict):
        raise ToolFailure("fulltext_media", "Expected exactly one BioC document")
    doc = docs[0]
    pmcid = str(doc.get("id", "")).upper()
    if not re.fullmatch(r"PMC[0-9]{1,12}", pmcid) or (expected_pmcid and expected_pmcid != pmcid):
        raise ToolFailure("fulltext_identity", "BioC article identity differs from resolved PMCID")
    passages = doc.get("passages")
    if not isinstance(passages, list) or len(passages) > 50000:
        raise ToolFailure("fulltext_media", "Invalid BioC passages")
    sections, abstract, title, total = {}, [], "", 0
    has_body = False
    for index, passage in enumerate(passages):
        if not isinstance(passage, dict) or not isinstance(passage.get("infons"), dict):
            raise ToolFailure("fulltext_media", "Malformed BioC passage")
        value = passage.get("text")
        if not isinstance(value, str) or not value.strip():
            continue
        total += len(value)
        if total > 4_000_000 or len(value) > 200_000:
            raise ToolFailure("fulltext_media", "BioC text exceeds its extraction limit")
        infons = passage["infons"]
        section = str(infons.get("section_type", "")).upper()
        typ = str(infons.get("type", "")).lower()
        locator = f"bioc:passage:{index}"
        item = {"locator": locator, "text": value, "passage_index": index, "bioc_type": typ}
        if type(passage.get("offset")) is int:
            item["offset"] = passage["offset"]
        if section == "TITLE":
            if typ == "front" or not title:
                title = value
        elif section == "ABSTRACT":
            item["locator"] = f"abstract:{locator}"
            abstract.append(item)
        elif section:
            if section not in NON_BODY_SECTIONS and typ == "paragraph":
                has_body = True
            group = sections.setdefault(section, {"title": section, "paragraphs": []})
            group["paragraphs"].append(item)
    if not has_body:
        raise ToolFailure("fulltext_media", "BioC response contains no identifiable body paragraph")
    return {"document_type": "bioc", "pmcid": pmcid, "title": title, "abstract": abstract,
            "sections": list(sections.values()), "limitations": ["BioC is a text representation of the article."]}
