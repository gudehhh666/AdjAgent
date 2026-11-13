"""Bounded fixed-origin Europe PMC core search, with explicit text availability.

This retrieves metadata/available abstracts, never claims that a full paper was
read. Host-only HTTP transport injection exists for deterministic fixture tests.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
import re
import time

import httpx

from ..tools.catalog import canonical_json
from ..tools.executor import ToolFailure


ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class SearchResponse:
    records: list[dict]
    hit_count: int
    request_url: str
    received_at: str
    raw: bytes = field(repr=False)


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignored += 1
        if tag in {"p", "br", "div", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.ignored:
            self.ignored -= 1
        if tag in {"p", "div", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def _plain(value, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("Provider text is not bounded text")
    parser = _PlainText()
    parser.feed(value)
    parser.close()
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def _record(item):
    if not isinstance(item, dict):
        raise ValueError("Provider record must be an object")
    identifier, source = item.get("id"), item.get("source")
    if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier)
            or not isinstance(source, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", source)):
        raise ValueError("Provider result has an invalid identifier/source")
    title = _plain(item.get("title"), 20000)
    if not title:
        raise ValueError("Provider result has no title")
    # Absent/null optional fields mean unavailable. Explicit false/zero/list
    # values are malformed provider metadata, not evidence that text is absent.
    abstract_value = item.get("abstractText")
    abstract = _plain("" if abstract_value is None else abstract_value, 200000)
    authors, year = item.get("authorString"), item.get("pubYear")
    authors = "" if authors is None else authors
    year = "" if year is None else year
    if not isinstance(authors, str) or len(authors) > 20000 or not isinstance(year, str) or len(year) > 8:
        raise ValueError("Malformed provider author/year metadata")
    result = {"id": identifier, "source": source, "title": title, "abstract": abstract,
              "authors": authors, "year": year,
              "text_availability": "abstract" if abstract else "title_only",
              "url": f"https://europepmc.org/article/{source}/{identifier}"}
    doi, pmcid = item.get("doi"), item.get("pmcid")
    if isinstance(doi, str) and re.fullmatch(r"10\.[0-9]{4,9}/[^\s]{1,256}", doi):
        result["doi"] = doi
    if isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid):
        result["pmcid"] = pmcid
    return result


class EuropePMC:
    def __init__(self, *, transport=None, timeout_seconds=20):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30):
            raise ValueError("Literature timeout must be finite and in (0,30] seconds")
        self.transport, self.timeout_seconds = transport, timeout_seconds

    def search(self, query, *, page_size=5, check_cancelled=lambda: None):
        """Check elapsed budget on every received chunk, with no hidden retries.

        A stalled network operation can take up to its I/O timeout (at most 5s)
        beyond a cancellation/deadline; this is not an OS-level kill guarantee.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        canonical_json(query)
        if type(page_size) is not int or not 1 <= page_size <= 10:
            raise ValueError("Page size must be an integer in 1..10")
        deadline = time.monotonic() + self.timeout_seconds

        def checkpoint():
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure("literature_timeout", "Europe PMC elapsed response budget exhausted")

        checkpoint()
        io_timeout = min(5, self.timeout_seconds)
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(io_timeout),
                              follow_redirects=False, trust_env=False) as client:
                with client.stream("GET", ENDPOINT, params={"query": query, "pageSize": page_size,
                                   "resultType": "core", "format": "json"},
                                   headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                    checkpoint()
                    if response.status_code != 200:
                        raise ToolFailure("literature_http", f"Europe PMC returned HTTP {response.status_code}; no redirects/retries followed")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ToolFailure("literature_encoding", "Unexpected compressed response; no decompression performed")
                    length = response.headers.get("content-length")
                    if length is not None and (len(length) > 10 or not length.isascii()
                                               or not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                        raise ToolFailure("literature_size", "Europe PMC response exceeds size limit or has invalid length")
                    chunks, size = [], 0
                    # No re-chunking: a dripping response must not postpone
                    # cancellation until 64 KiB has accumulated in the client.
                    for chunk in response.iter_bytes():
                        checkpoint()
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ToolFailure("literature_size", "Europe PMC response exceeds 2 MiB limit")
                        chunks.append(chunk)
                    raw, request_url = b"".join(chunks), str(response.request.url)
        except httpx.HTTPError as exc:
            raise ToolFailure("literature_transport", f"Europe PMC request failed without retry: {type(exc).__name__}") from exc
        checkpoint()
        try:
            body = json.loads(raw)
            canonical_json(body)
            if not isinstance(body, dict) or type(body.get("hitCount")) is not int or body["hitCount"] < 0:
                raise ValueError("Malformed hitCount")
            raw_records = body["resultList"]["result"]
            if not isinstance(raw_records, list) or len(raw_records) > page_size or body["hitCount"] < len(raw_records):
                raise ValueError("Malformed result list")
            records = [_record(item) for item in raw_records]
            if len({(record["source"], record["id"]) for record in records}) != len(records):
                raise ValueError("Provider returned duplicate result identities")
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            raise ToolFailure("literature_response", f"Malformed Europe PMC response: {str(exc)[:1000]}") from exc
        return SearchResponse(records, body["hitCount"], request_url, datetime.now(timezone.utc).isoformat(), raw)
