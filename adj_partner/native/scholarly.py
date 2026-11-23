"""Normalized multi-provider scholarly discovery and OA identity resolution.

Every adapter has a fixed origin and returns only bounded metadata/abstracts.
Provider failures remain diagnostics; they are never converted to negative
scientific evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import re
import time
import unicodedata
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from lxml import etree

import httpx

from ..tools.catalog import canonical_json
from ..tools.executor import ToolFailure
from ..cli_defaults import _explicit_values
from . import ncbi
from .literature import EuropePMC, _plain


PUBMED_SEARCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
PMC_ID_CONVERTER_ENDPOINT = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2/"
MAX_METADATA_BYTES = 2 * 1024 * 1024
_PRIVATE_REQUEST_PARAMETERS = {"api_key", "key", "token", "access_token", "email", "mailto"}


@dataclass(frozen=True)
class ScholarlyResponse:
    records: list[dict]
    diagnostics: list[dict]
    request_urls: list[str]
    received_at: str
    raw: bytes = field(repr=False)
    hit_count: int = 0

    @property
    def request_url(self):
        """Compatibility projection for the former single-provider adapter."""
        return self.request_urls[0] if self.request_urls else ""


def normalize_doi(value):
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean)
    clean = re.sub(r"^doi:\s*", "", clean)
    clean = clean.rstrip(".,;)")
    return clean if re.fullmatch(r"10\.[0-9]{4,9}/[^\s<>\"]{1,256}", clean) else None


def _stored_request_url(value):
    """Strip credentials/contact metadata from provider URLs before persistence."""
    if not isinstance(value, str) or len(value) > 16_384:
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return ""
        query = urlencode([(key, item) for key, item in parse_qsl(
            parsed.query, keep_blank_values=True, max_num_fields=128)
            if key.casefold() not in _PRIVATE_REQUEST_PARAMETERS])
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
    except (TypeError, ValueError, UnicodeError):
        return ""


def _title_key(value):
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


_QUERY_STOP_WORDS = {"and", "or", "not", "the", "for", "from", "with", "using",
                     "into", "that", "this", "study", "effect", "effects"}


def _terms(value):
    if not isinstance(value, str):
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold())
            if len(token) >= 2 and token not in _QUERY_STOP_WORDS}


def _relevance(record, query):
    """Small deterministic lexical reranker across already-ranked providers.

    PubMed is queried with Best Match while Europe PMC keyword results skew to recent,
    loosely matching papers, so a PubMed hit earns less than one title term but more
    than one abstract term.
    """
    wanted = _terms(query)
    title = _terms(record.get("title"))
    abstract = _terms(record.get("abstract"))
    title_hits, abstract_hits = len(wanted & title), len(wanted & abstract)
    best_match = 4 if "pubmed" in record.get("providers", ()) else 0
    return (title_hits * 20 + abstract_hits * 3 + best_match,
            title_hits, abstract_hits, int(bool(record.get("abstract"))),
            int(bool(record.get("pmcid"))), int(bool(record.get("pmid") or record.get("doi"))))


def _provider_name(record, fallback):
    value = record.get("provider") if isinstance(record, dict) else None
    return value if isinstance(value, str) and value else fallback


def _bounded_request(transport, endpoint, *, params, accept, timeout_seconds,
                     check_cancelled, error_prefix):
    deadline = time.monotonic() + timeout_seconds

    def checkpoint():
        check_cancelled()
        if time.monotonic() >= deadline:
            raise ToolFailure(f"{error_prefix}_timeout", "Scholarly provider response budget exhausted")

    checkpoint()
    if "ncbi.nlm.nih.gov" in endpoint:
        ncbi.wait(params.get("api_key"))
        checkpoint()
    try:
        with httpx.Client(transport=transport, timeout=httpx.Timeout(min(5, timeout_seconds)),
                          follow_redirects=False, trust_env=False) as client:
            with client.stream("GET", endpoint, params=params,
                               headers={"Accept": accept, "Accept-Encoding": "identity",
                                        "User-Agent": "AdjAgent/1.0 (scholarly metadata retrieval)"}) as response:
                checkpoint()
                if response.status_code != 200:
                    raise ToolFailure(f"{error_prefix}_http",
                        f"Scholarly provider returned HTTP {response.status_code}; no redirects/retries followed")
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ToolFailure(f"{error_prefix}_encoding", "Unexpected compressed scholarly response")
                length = response.headers.get("content-length")
                if length is not None and (not length.isascii() or not length.isdigit()
                                           or int(length) > MAX_METADATA_BYTES):
                    raise ToolFailure(f"{error_prefix}_size", "Scholarly response exceeds 2 MiB")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    checkpoint()
                    size += len(chunk)
                    if size > MAX_METADATA_BYTES:
                        raise ToolFailure(f"{error_prefix}_size", "Scholarly response exceeds 2 MiB")
                    chunks.append(chunk)
                return b"".join(chunks), str(response.request.url)
    except httpx.HTTPError as exc:
        raise ToolFailure(f"{error_prefix}_transport",
            f"Scholarly request failed without retry: {type(exc).__name__}") from exc


def _valid_timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 30):
        raise ValueError("Scholarly timeout must be finite and in (0,30] seconds")
    return value


def _xml_root(raw, error_prefix):
    # PubMed legitimately emits an external NLM DOCTYPE. Accept its declaration
    # while disabling DTD loading, network access and entity resolution. Internal
    # entity declarations remain forbidden so no expansion payload reaches lxml.
    if re.search(br"<!\s*ENTITY\b", raw, flags=re.IGNORECASE):
        raise ToolFailure(f"{error_prefix}_response", "Unsafe XML declaration in scholarly response")
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                                 recover=False, huge_tree=False)
        return etree.fromstring(raw, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ToolFailure(f"{error_prefix}_response", "Malformed scholarly XML response") from exc


class PubMedClient:
    """NCBI E-utilities search followed by one batched article fetch."""

    def __init__(self, *, transport=None, timeout_seconds=20, api_key=None, email=None):
        self.transport, self.timeout_seconds = transport, _valid_timeout(timeout_seconds)
        self.api_key = api_key if isinstance(api_key, str) and api_key else None
        self.email = email if isinstance(email, str) and email else None

    @property
    def secrets(self):
        return (self.api_key,) if self.api_key else ()

    def _common(self):
        result = {"tool": "adjagent"}
        if self.api_key:
            result["api_key"] = self.api_key
        if self.email:
            result["email"] = self.email
        return result

    def search(self, query, *, page_size=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        if type(page_size) is not int or not 1 <= page_size <= 10:
            raise ValueError("Page size must be an integer in 1..10")
        raw_search, search_url = _bounded_request(self.transport, PUBMED_SEARCH_ENDPOINT,
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": page_size,
                    "sort": "relevance", **self._common()},
            accept="application/json", timeout_seconds=self.timeout_seconds,
            check_cancelled=check_cancelled, error_prefix="pubmed")
        try:
            body = json.loads(raw_search)
            canonical_json(body)
            result = body["esearchresult"]
            total = int(result["count"])
            identifiers = result["idlist"]
            if (not isinstance(identifiers, list) or len(identifiers) > page_size
                    or any(not isinstance(item, str) or not re.fullmatch(r"[1-9][0-9]{0,11}", item) for item in identifiers)):
                raise ValueError("Malformed PubMed identifiers")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise ToolFailure("pubmed_response", "Malformed PubMed search response") from exc
        if not identifiers:
            aggregate = canonical_json({"search": json.loads(raw_search), "fetch": None}).encode()
            return ScholarlyResponse([], [], [search_url], datetime.now(timezone.utc).isoformat(), aggregate, total)
        raw_fetch, fetch_url = _bounded_request(self.transport, PUBMED_FETCH_ENDPOINT,
            params={"db": "pubmed", "id": ",".join(identifiers), "retmode": "xml", **self._common()},
            accept="application/xml", timeout_seconds=self.timeout_seconds,
            check_cancelled=check_cancelled, error_prefix="pubmed")
        root = _xml_root(raw_fetch, "pubmed")
        records = []
        for article in root.findall("./PubmedArticle"):
            pmid = (article.findtext("./MedlineCitation/PMID") or "").strip()
            title_node = article.find("./MedlineCitation/Article/ArticleTitle")
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""
            if not re.fullmatch(r"[1-9][0-9]{0,11}", pmid) or not title:
                raise ToolFailure("pubmed_response", "Malformed PubMed article identity")
            abstract_parts = []
            for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
                value = "".join(node.itertext()).strip()
                label = (node.get("Label") or "").strip()
                if value:
                    abstract_parts.append(f"{label}: {value}" if label else value)
            authors = []
            for author in article.findall("./MedlineCitation/Article/AuthorList/Author"):
                name = " ".join(part for part in ((author.findtext("ForeName") or "").strip(),
                                                   (author.findtext("LastName") or "").strip()) if part)
                if name:
                    authors.append(name)
            year = (article.findtext("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year") or "").strip()
            identifiers_by_type = {node.get("IdType"): (node.text or "").strip()
                for node in article.findall("./PubmedData/ArticleIdList/ArticleId")}
            abstract = "\n".join(abstract_parts)
            record = {"id": pmid, "source": "MED", "provider": "pubmed", "pmid": pmid,
                      "title": title, "abstract": abstract, "authors": ", ".join(authors), "year": year,
                      "text_availability": "abstract" if abstract else "title_only",
                      "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
            doi = normalize_doi(identifiers_by_type.get("doi"))
            pmcid = identifiers_by_type.get("pmc")
            if doi:
                record["doi"] = doi
            if isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid):
                record["pmcid"] = pmcid
            records.append(record)
        aggregate = canonical_json({"search": json.loads(raw_search), "fetch_xml_sha256":
            __import__("hashlib").sha256(raw_fetch).hexdigest(), "records": records}).encode()
        return ScholarlyResponse(records, [], [search_url, fetch_url],
            datetime.now(timezone.utc).isoformat(), aggregate, total)


class CrossrefClient:
    """Crossref public REST metadata search."""

    def __init__(self, *, transport=None, timeout_seconds=20, email=None):
        self.transport, self.timeout_seconds = transport, _valid_timeout(timeout_seconds)
        self.email = email if isinstance(email, str) and email else None

    def search(self, query, *, page_size=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        if type(page_size) is not int or not 1 <= page_size <= 10:
            raise ValueError("Page size must be an integer in 1..10")
        params = {"query.bibliographic": query, "rows": page_size, "select":
                  "DOI,title,abstract,author,published,URL"}
        if self.email:
            params["mailto"] = self.email
        raw, request_url = _bounded_request(self.transport, CROSSREF_ENDPOINT, params=params,
            accept="application/json", timeout_seconds=self.timeout_seconds,
            check_cancelled=check_cancelled, error_prefix="crossref")
        try:
            body = json.loads(raw)
            canonical_json(body)
            message, records = body["message"], []
            items = message["items"]
            total = int(message["total-results"])
            if not isinstance(items, list) or len(items) > page_size:
                raise ValueError("Malformed Crossref items")
            for item in items:
                doi = normalize_doi(item.get("DOI"))
                titles = item.get("title")
                if not doi or not isinstance(titles, list) or not titles or not isinstance(titles[0], str):
                    raise ValueError("Malformed Crossref identity")
                abstract = _plain(item.get("abstract") or "", 200000)
                author_items = item.get("author") or []
                if not isinstance(author_items, list):
                    raise ValueError("Malformed Crossref authors")
                authors = ", ".join(" ".join(part for part in (person.get("given", ""), person.get("family", ""))
                    if isinstance(part, str) and part).strip() for person in author_items if isinstance(person, dict))
                date_parts = (item.get("published") or {}).get("date-parts") or []
                year = str(date_parts[0][0]) if date_parts and isinstance(date_parts[0], list) and date_parts[0] else ""
                records.append({"id": doi, "source": "DOI", "provider": "crossref", "doi": doi,
                    "title": _plain(titles[0], 20000), "abstract": abstract, "authors": authors,
                    "year": year, "text_availability": "abstract" if abstract else "title_only",
                    "url": f"https://doi.org/{doi}"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise ToolFailure("crossref_response", "Malformed Crossref response") from exc
        return ScholarlyResponse(records, [], [request_url], datetime.now(timezone.utc).isoformat(), raw, total)


def _identifier_from(record):
    for key in ("pmcid", "pmid", "doi"):
        value = record.get(key) if isinstance(record, dict) else None
        if key == "doi":
            value = normalize_doi(value)
        if isinstance(value, str) and value:
            return value
    return None


class PMCIdConverter:
    """NCBI PMC ID Converter for DOI/PMID/PMCID mapping."""

    def __init__(self, *, transport=None, timeout_seconds=20, api_key=None, email=None):
        self.transport, self.timeout_seconds = transport, _valid_timeout(timeout_seconds)
        self.api_key = api_key if isinstance(api_key, str) and api_key else None
        self.email = email if isinstance(email, str) and email else None

    @property
    def secrets(self):
        return (self.api_key,) if self.api_key else ()

    def resolve(self, record, *, check_cancelled=lambda: None):
        identifier = _identifier_from(record)
        if identifier is None:
            return {}
        params = {"ids": identifier, "format": "json", "tool": "adjagent"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        raw, _ = _bounded_request(self.transport, PMC_ID_CONVERTER_ENDPOINT, params=params,
            accept="application/json", timeout_seconds=self.timeout_seconds,
            check_cancelled=check_cancelled, error_prefix="pmc_id")
        try:
            body = json.loads(raw)
            canonical_json(body)
            rows = body["records"]
            if not isinstance(rows, list) or len(rows) > 10:
                raise ValueError("Malformed converter records")
            if not rows:
                return {}
            row = rows[0]
            if not isinstance(row, dict):
                raise ValueError("Malformed converter record")
            result = {}
            doi = normalize_doi(row.get("doi"))
            pmid, pmcid = row.get("pmid"), row.get("pmcid")
            if type(pmid) is int and 0 < pmid <= 999_999_999_999:
                pmid = str(pmid)
            if doi:
                result["doi"] = doi
            if isinstance(pmid, str) and re.fullmatch(r"[1-9][0-9]{0,11}", pmid):
                result["pmid"] = pmid
            if isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid):
                result["pmcid"] = pmcid
            return result
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise ToolFailure("pmc_id_response", "Malformed PMC ID Converter response") from exc


class UnpaywallClient:
    """Optional free OA-location lookup using a required contact email."""

    def __init__(self, email, *, transport=None, timeout_seconds=20):
        if (not isinstance(email, str) or len(email) > 320 or not re.fullmatch(r"[^\s@]+@[^\s@]+", email)):
            raise ValueError("Unpaywall requires a valid contact email")
        self.email, self.transport = email, transport
        self.timeout_seconds = _valid_timeout(timeout_seconds)

    @property
    def secrets(self):
        return ()

    def resolve(self, record, *, check_cancelled=lambda: None):
        doi = normalize_doi(record.get("doi") if isinstance(record, dict) else None)
        if not doi:
            return {}
        endpoint = UNPAYWALL_ENDPOINT + quote(doi, safe="/")
        raw, _ = _bounded_request(self.transport, endpoint, params={"email": self.email},
            accept="application/json", timeout_seconds=self.timeout_seconds,
            check_cancelled=check_cancelled, error_prefix="unpaywall")
        try:
            body = json.loads(raw)
            canonical_json(body)
            response_doi = normalize_doi(body.get("doi"))
            location = body.get("best_oa_location")
            if response_doi != doi or type(body.get("is_oa")) is not bool:
                raise ValueError("Malformed Unpaywall identity")
            result = {"doi": doi, "is_oa": body["is_oa"], "license": None,
                      "best_oa_location": None}
            if location is not None:
                if not isinstance(location, dict):
                    raise ValueError("Malformed Unpaywall location")
                url = location.get("url_for_pdf")
                host_type, license_name = location.get("host_type"), location.get("license")
                parsed = urlsplit(url) if isinstance(url, str) else None
                if (parsed is None or parsed.scheme != "https" or not parsed.hostname
                        or parsed.username or parsed.password):
                    url = None
                result["license"] = license_name if isinstance(license_name, str) else None
                result["best_oa_location"] = {"url_for_pdf": url,
                    "host_type": host_type if isinstance(host_type, str) else None}
            return result
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise ToolFailure("unpaywall_response", "Malformed Unpaywall response") from exc


def _merge_record(existing, incoming, provider):
    merged = dict(existing)
    providers = list(merged.get("providers", []))
    if provider not in providers:
        providers.append(provider)
    merged["providers"] = providers
    for identifier in ("doi", "pmid", "pmcid"):
        value = normalize_doi(incoming.get(identifier)) if identifier == "doi" else incoming.get(identifier)
        if value and identifier not in merged:
            merged[identifier] = value
    for field in ("abstract", "authors", "year", "url"):
        candidate = incoming.get(field)
        current = merged.get(field)
        if isinstance(candidate, str) and (not isinstance(current, str) or len(candidate) > len(current)):
            merged[field] = candidate
    merged["text_availability"] = "abstract" if merged.get("abstract") else "title_only"
    return merged


class ScholarlyGateway:
    """Merge bounded metadata from Europe PMC, PubMed, and Crossref."""

    def __init__(self, *, europe_pmc=None, pubmed=None, crossref=None,
                 id_converter=None, unpaywall=None, secrets=()):
        self.europe_pmc = europe_pmc or EuropePMC()
        self.pubmed = pubmed or PubMedClient()
        self.crossref = crossref or CrossrefClient()
        self.id_converter = id_converter or PMCIdConverter()
        self.unpaywall = unpaywall
        combined = list(secrets)
        for client in (self.pubmed, self.id_converter, self.unpaywall):
            for secret in getattr(client, "secrets", ()) if client is not None else ():
                if secret not in combined:
                    combined.append(secret)
        self.secrets = tuple(combined)

    @classmethod
    def from_environment(cls, project_root, env_file=None):
        try:
            selected = _explicit_values(project_root, env_file)
            def value(name):
                return os.environ.get(name) or selected.get(name) or ""
            api_key = value("ADJ_NCBI_API_KEY")
            ncbi_email = value("ADJ_NCBI_EMAIL")
            unpaywall_email = value("ADJ_UNPAYWALL_EMAIL")
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            raise ValueError("Scholarly provider configuration is missing, malformed or unsafe") from None
        if api_key and (len(api_key) > 4096 or any(not 33 <= ord(char) <= 126 for char in api_key)):
            raise ValueError("ADJ_NCBI_API_KEY is malformed")
        for name, email in (("ADJ_NCBI_EMAIL", ncbi_email), ("ADJ_UNPAYWALL_EMAIL", unpaywall_email)):
            if email and (len(email) > 320 or not re.fullmatch(r"[^\s@]+@[^\s@]+", email)):
                raise ValueError(f"{name} is malformed")
        pubmed = PubMedClient(api_key=api_key or None, email=ncbi_email or None)
        converter = PMCIdConverter(api_key=api_key or None, email=ncbi_email or None)
        unpaywall = UnpaywallClient(unpaywall_email) if unpaywall_email else None
        return cls(pubmed=pubmed, crossref=CrossrefClient(email=ncbi_email or None),
                   id_converter=converter, unpaywall=unpaywall)

    def _search_clients(self):
        return [(name, client) for name, client in (
            ("europe_pmc", self.europe_pmc),
            ("pubmed", self.pubmed),
            ("crossref", self.crossref),
        ) if client is not None]

    def search(self, query, *, page_size=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        if type(page_size) is not int or not 1 <= page_size <= 10:
            raise ValueError("Page size must be an integer in 1..10")
        canonical_json(query)
        diagnostics, request_urls, provider_rows = [], [], []
        raw_providers = []
        total_hits = 0
        for provider, client in self._search_clients():
            check_cancelled()
            try:
                response = client.search(query, page_size=page_size, check_cancelled=check_cancelled)
            except ToolFailure as exc:
                diagnostics.append({"provider": provider, "code": exc.code})
                continue
            response_urls = getattr(response, "request_urls", None)
            if not isinstance(response_urls, list):
                response_urls = [getattr(response, "request_url", "")]
            stored_urls = [stored for stored in map(_stored_request_url, response_urls) if stored]
            request_urls.extend(stored_urls)
            request_url = stored_urls[0] if stored_urls else ""
            total_hits += getattr(response, "hit_count", len(response.records))
            raw_providers.append({
                "provider": provider,
                "request_url": request_url,
                "response_sha256": __import__("hashlib").sha256(response.raw).hexdigest(),
            })
            for original in response.records:
                record = dict(original)
                record["provider"] = _provider_name(record, provider)
                doi = normalize_doi(record.get("doi"))
                if doi:
                    record["doi"] = doi
                provider_rows.append(record)

        merged, identities = [], {}
        for record in provider_rows:
            identity = ("doi", record["doi"]) if record.get("doi") else ("title", _title_key(record.get("title")))
            if not identity[1]:
                identity = ("provider", record.get("provider"), record.get("id"))
            provider = record["provider"]
            if identity in identities:
                index = identities[identity]
                merged[index] = _merge_record(merged[index], record, provider)
                continue
            normalized = _merge_record({}, record, provider)
            normalized.setdefault("id", record.get("id", ""))
            normalized.setdefault("source", record.get("source", provider))
            normalized.setdefault("title", record.get("title", ""))
            normalized.setdefault("abstract", record.get("abstract", ""))
            merged.append(normalized)
            identities[identity] = len(merged) - 1

        # Every upstream provider already applies its own bounded ranking. Merge
        # the at-most 3*page_size candidates first, then rerank the small pool so
        # one provider cannot crowd relevant records from the others out merely
        # because it is queried first. Stable original order remains the tie-break.
        merged = [record for _, record in sorted(enumerate(merged),
            key=lambda pair: tuple(-value for value in _relevance(pair[1], query)) + (pair[0],))][:page_size]
        aggregate = canonical_json({"providers": raw_providers, "records": merged,
                                    "diagnostics": diagnostics}).encode("utf-8")
        return ScholarlyResponse(merged, diagnostics, request_urls,
            datetime.now(timezone.utc).isoformat(), aggregate,
            hit_count=max(total_hits, len(merged)))

    def resolve(self, record, *, check_cancelled=lambda: None):
        if not isinstance(record, dict) or not isinstance(record.get("title"), str):
            raise ValueError("A normalized scholarly record is required")
        check_cancelled()
        identity = {key: record[key] for key in ("doi", "pmid", "pmcid") if isinstance(record.get(key), str)}
        if "doi" in identity:
            normalized = normalize_doi(identity["doi"])
            if normalized:
                identity["doi"] = normalized
        diagnostics = []
        if self.id_converter is not None:
            try:
                converted = self.id_converter.resolve(record, check_cancelled=check_cancelled)
                if isinstance(converted, dict):
                    ids = {k: v for k, v in converted.items()
                           if k in {"doi", "pmid", "pmcid"} and isinstance(v, str) and v}
                    if "doi" in ids:
                        ids["doi"] = normalize_doi(ids["doi"])
                    if any(identity[k] != v for k, v in ids.items() if k in identity and v):
                        diagnostics.append({"provider":"pmc_id_converter", "code":"identifier_conflict"})
                    else:
                        identity.update({k: v for k, v in ids.items() if v})
            except ToolFailure as exc:
                diagnostics.append({"provider": "pmc_id_converter", "code": exc.code})
        acquisition = []
        pmcid = identity.get("pmcid")
        if isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid):
            acquisition.append({"kind": "jats",
                "url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                "provider": "europe_pmc", "access_status": "availability_unverified"})
        if self.unpaywall is not None and identity.get("doi"):
            try:
                oa = self.unpaywall.resolve({**record, **identity}, check_cancelled=check_cancelled)
                location = oa.get("best_oa_location") if isinstance(oa, dict) else None
                pdf_url = location.get("url_for_pdf") if isinstance(location, dict) else None
                if oa.get("is_oa") is True and isinstance(pdf_url, str):
                    acquisition.append({"kind": "pdf", "url": pdf_url, "provider": "unpaywall",
                        "is_oa": True, "license": oa.get("license"),
                        "host_type": location.get("host_type")})
            except ToolFailure as exc:
                diagnostics.append({"provider": "unpaywall", "code": exc.code})
        if isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid):
            acquisition.extend([
                {"kind": "pmc_cloud", "provider": "pmc_cloud", "pmcid": pmcid},
                {"kind": "bioc", "provider": "ncbi_bioc",
                 "url": f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"},
            ])
        candidate = {"title": record["title"], **identity,
                     "acquisition": acquisition, "diagnostics": diagnostics,
                     "signals": {"identifier_count": len(identity),
                                 "has_fulltext_candidate": bool(acquisition)}}
        return [candidate]
