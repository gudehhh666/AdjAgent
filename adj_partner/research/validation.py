"""Bounded JSON and domain validation for every research write boundary."""

import hashlib
import re

from ..tools.catalog import canonical_json


KINDS = frozenset({"literature", "web", "computed", "userfile"})
CLAIM_KINDS = frozenset({"observation", "proposal", "rejected"})
AVAILABILITY = frozenset({"title_only", "search_snippet", "abstract", "full_text", "derived_full_text", "database_record", "userfile", "computed", "unknown"})


def bounded_json(value, maximum=1_000_000):
    try:
        encoded = canonical_json(value)
    except (RecursionError, OverflowError) as exc:
        raise ValueError("JSON nesting exceeds limits") from exc
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError("Research JSON exceeds size limit")
    return encoded


def digest(value):
    return hashlib.sha256(bounded_json(value).encode("utf-8")).hexdigest()


def text(value, name, maximum=20000, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'a' if not empty else 'possibly empty'} bounded text value")
    bounded_json(value)
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Invalid research identifier")
    return value


def revision(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected revision must be a nonnegative integer")
    return value


def strings(value, name, maximum=100, *, item_max=20000, unique=False):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded list")
    for item in value:
        text(item, name, item_max)
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique entries")
    return value


def ids(value):
    strings(value, "evidence/claim references", item_max=128, unique=True)
    for item in value:
        identifier(item)
    return value


def provenance(value):
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    bounded_json(value)
    if "limitations" in value:
        strings(value["limitations"], "source limitations")
    if "text_availability" in value and value["text_availability"] not in AVAILABILITY:
        raise ValueError("Unknown source text_availability")
    return value


def claim_fields(claim_text, kind, evidence_ids):
    text(claim_text, "claim text")
    if not isinstance(kind, str) or kind not in CLAIM_KINDS:
        raise ValueError("Unknown claim kind")
    ids(evidence_ids)
