"""Bounded deterministic windows over already ingested, task-scoped evidence.

This is a reader, not retrieval, evidence certification, or a permissions grant.
Coordinates refer to redacted scalar text; the immutable source hash is retained
separately from the redacted projection hash. JSON containers are never emitted.
"""
import hashlib
import json
import re

from .contracts import ToolResult
from .research import ResearchStore
from .research.records import extract
from .research.validation import bounded_json, identifier
from .tools import ToolSpec
from .tools.catalog import canonical_json

MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 2300
BOUNDARY = "Exact redacted window only; scope is inherited, not full-source review or scientific validation. A query miss is not evidence of absence."


def _placeholder(value):
    return isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "n/a"}


def _number(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid {name}")


def _leaves(value, pointer="", depth=0, remaining=None):
    """Bounded literal JSON traversal, no attributes, evaluation, or I/O."""
    remaining = [20000] if remaining is None else remaining
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 32:
        raise ValueError("JSON traversal exceeds reader bounds; select a narrower pointer")
    if isinstance(value, dict):
        for key, item in value.items():
            part = key.replace("~", "~0").replace("/", "~1")
            yield from _leaves(item, pointer + "/" + part, depth + 1, remaining)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, pointer + "/" + str(index), depth + 1, remaining)
    else:
        yield pointer, value


def _material(context, source_id, artifact_id):
    store = context.store
    research = ResearchStore(store)
    with store.transaction() as conn:
        task = research._scope(conn, context.session_id, context.run_id)
        source = None
        if source_id:
            source = research._get(conn, "source", context.session_id, task["id"], source_id)
            if source["task_id"] != task["id"]:
                raise PermissionError("Evidence source belongs to another task")
            metadata = research._source_metadata(conn, source)
        else:
            metadata = research._artifact(conn, context.session_id, artifact_id)
            if not metadata["run_id"]:
                raise PermissionError("Evidence artifact lacks a producing task")
            producing = research._scope(conn, context.session_id, metadata["run_id"])
            if producing["id"] != task["id"]:
                raise PermissionError("Evidence artifact belongs to another task")
            # Same-task source association is a provenance convenience, not a
            # way to turn a raw artifact into a scientifically certified source.
            for item in research._list(conn, "source", context.session_id, task["id"]):
                if item["task_id"] == task["id"] and item["artifact_id"] == artifact_id:
                    source = item
                    research._source_metadata(conn, source)
                    break
    if metadata["size"] > MAX_INPUT_BYTES:
        raise ValueError("Evidence exceeds the 8 MiB text limit; use a bounded computation")
    raw = store.read_artifact(metadata["id"], max_bytes=MAX_INPUT_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata["sha256"] or source and digest != source["artifact_hash"]:
        raise ValueError("Evidence source artifact hash integrity mismatch")
    return raw, source, metadata


def read_evidence(context, source_id=None, artifact_id=None, selector=None,
                  query=None, start=0, max_characters=1200, match_index=0):
    """Read one scalar/window; exactly one current-task source/artifact required."""
    context.check_cancelled()
    source_id, artifact_id = (None if _placeholder(x) else x for x in (source_id, artifact_id))
    if source_id and artifact_id:
        artifact_id = None
    if bool(source_id) == bool(artifact_id):
        raise ValueError("Provide exactly one source_id or artifact_id")
    identifier(source_id or artifact_id)
    _number(start, "start", 100000000)
    _number(max_characters, "max_characters", 4000)
    _number(match_index, "match_index", 10000)
    if max_characters == 0:
        raise ValueError("max_characters must be positive")
    if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 160):
        raise ValueError("Query must be 1–160 nonblank characters")
    if query is None and match_index:
        raise ValueError("match_index requires query")
    if selector is not None and (not isinstance(selector, dict) or set(selector) != {"pointer"}):
        raise ValueError("Selector must contain only a JSON pointer")
    raw, source, metadata = _material(context, source_id, artifact_id)
    text = raw.decode("utf-8-sig")
    is_json = metadata["media_type"].split(";", 1)[0] == "application/json"
    if is_json:
        parsed = json.loads(text)
        bounded_json(parsed)
        material = context.store.redact(parsed)
        encoded = canonical_json(material).encode("utf-8")
        pointer = selector["pointer"] if selector else ""
        value = extract(encoded, {"pointer": pointer})["value"] if selector else material
    else:
        if selector is not None and selector["pointer"]:
            raise ValueError("JSON selector requires a JSON artifact")
        material = context.store.redact(text)
        encoded = material.encode("utf-8")
        value, pointer = material, None
    provenance = context.store.redact(source["provenance"] if source else metadata["provenance"])
    url = provenance.get("url") or provenance.get("request_url")
    url = url if isinstance(url, str) else None
    data = {"source_id": source["id"] if source else None, "artifact_id": metadata["id"],
            "artifact_hash": metadata["sha256"], "projection_hash": hashlib.sha256(encoded).hexdigest(),
            "projection": "redacted_scalar_character_offsets", "text_availability": str(provenance.get("text_availability", "unknown"))[:40],
            "url": url[:400] if url else None, "url_truncated": bool(url and len(url) > 400),
            "boundary": BOUNDARY}
    for key in ("document_scope", "parse_mode", "page", "archive_member", "pmc_version"):
        if key in provenance:
            data[key] = provenance[key]
    found = None
    if query is not None:
        count = 0
        for path, leaf in _leaves(value, pointer or ""):
            if not isinstance(leaf, str):
                continue
            for match in re.finditer(re.escape(query), leaf, flags=re.IGNORECASE):
                if match.start() < start:
                    continue
                if count == match_index:
                    found = (path if is_json else None, leaf, match.start())
                    break
                count += 1
            if found:
                break
        if found is None:
            missed = ToolResult(data={**data, "matched": False, "selector": selector, "start": None,
                                     "end": None, "total": None, "next": None, "text": ""})
            if len(canonical_json(missed.to_dict()).encode("utf-8")) >= MAX_RESULT_BYTES:
                raise ValueError("Evidence locator exceeds visible budget; select a shorter pointer")
            return missed
        pointer, value, position = found
        # A short lead-in leaves enough byte budget for the matched term, even
        # for multibyte scripts. Case-insensitive search retains original offsets.
        start = max(0, position - 48)
    elif isinstance(value, (dict, list)):
        raise ValueError("Select a scalar JSON pointer or use query; containers are not emitted")
    scalar_text = value if isinstance(value, str) else canonical_json(value)
    if start > len(scalar_text):
        raise ValueError("Character offset is beyond selected scalar")
    data.update(matched=True, selector={"pointer": pointer} if is_json else None,
                start=start, end=start, total=len(scalar_text), next=None, text="")
    if not isinstance(value, str):
        if start or max_characters < len(scalar_text):
            raise ValueError("Non-string scalar must be read whole")
        data.update(value=value, value_type="null" if value is None else "boolean" if isinstance(value, bool) else "number")
    requested_end = min(len(scalar_text), start + max_characters)

    def result(end):
        payload = {**data, "end": end, "text": scalar_text[start:end],
                   "next": {"selector": data["selector"], "start": end} if end < len(scalar_text) else None}
        return ToolResult(data=payload)

    low, high = start, requested_end
    while low < high:
        middle = (low + high + 1) // 2
        if len(canonical_json(result(middle).to_dict()).encode("utf-8")) < MAX_RESULT_BYTES:
            low = middle
        else:
            high = middle - 1
    selected = result(low)
    if len(canonical_json(selected.to_dict()).encode("utf-8")) >= MAX_RESULT_BYTES or low == start and start < len(scalar_text):
        raise ValueError("Evidence locator exceeds visible budget; select a shorter pointer")
    if found and low < position + len(query):
        raise ValueError("Query window exceeds visible budget; shorten query or locator")
    context.check_cancelled()
    return selected


def register_evidence_access(service):
    """Register before AgentService snapshots its benign tools; enable separately."""
    properties = {
        "source_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "artifact_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "selector": {"type": "object", "properties": {"pointer": {"type": "string", "maxLength": 2000}}, "required": ["pointer"], "additionalProperties": False},
        "query": {"type": "string", "minLength": 1, "maxLength": 160},
        "start": {"type": "integer", "minimum": 0, "maximum": 100000000},
        "max_characters": {"type": "integer", "minimum": 1, "maximum": 4000},
        "match_index": {"type": "integer", "minimum": 0, "maximum": 10000},
    }
    spec = ToolSpec("evidence_read", "Read exact visible text from one already ingested current-task source_id OR artifact_id. Use document_read's returned derived source_id with selector={pointer: ...}, or query a literal short phrase to locate a window anywhere in text/JSON strings (case-insensitive, match_index starts at 0). A table cell uses its exact JSON pointer. Follow returned next selector/start without query for continuation. No network, file paths, new evidence, or full-source review; scope and hashes are preserved.",
                    {"type": "object", "properties": properties, "additionalProperties": False}, version="2")
    service.catalog.register(spec, read_evidence)
