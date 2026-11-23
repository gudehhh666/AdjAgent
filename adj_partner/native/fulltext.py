"""Bounded retrieval of lawfully exposed JATS XML or OA PDF bytes."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import ipaddress
import math
import re
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.parse import urljoin, urlsplit

import httpx
from lxml import etree

from ..tools.executor import ToolFailure
from .jats import inert_jats_content, matched_article_ids


CURL_EXECUTABLE = "/usr/bin/curl"


@dataclass(frozen=True)
class PublicTarget:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


@dataclass(frozen=True)
class FullTextResponse:
    body: bytes = field(repr=False)
    url: str
    media_type: str
    received_at: str
    acquisition: dict
    attempts: tuple[dict, ...] = ()


def _resolved_addresses(hostname, resolver):
    try:
        answer = resolver(hostname)
    except TypeError:
        answer = resolver(hostname, 443, type=socket.SOCK_STREAM)
    addresses = []
    for item in answer:
        if isinstance(item, str):
            addresses.append(item)
        elif isinstance(item, tuple) and len(item) >= 5 and isinstance(item[4], tuple):
            addresses.append(item[4][0])
    return tuple(dict.fromkeys(addresses))


def public_target(url, resolver=socket.getaddrinfo):
    """Validate one destination before a request and return pinned addresses."""
    message = "Full text must use a credential-free public HTTPS target"
    if not isinstance(url, str) or not 1 <= len(url) <= 4096 or any(ord(char) < 32 for char in url):
        raise ToolFailure("fulltext_target", message)
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.port not in {None, 443}):
        raise ToolFailure("fulltext_target", message)
    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = (str(literal),)
    except ValueError:
        addresses = _resolved_addresses(parsed.hostname, resolver)
    if not addresses:
        raise ToolFailure("fulltext_target", message)
    try:
        if any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ToolFailure("fulltext_target", message)
    except ValueError as exc:
        raise ToolFailure("fulltext_target", message) from exc
    return PublicTarget(url, parsed.hostname, parsed.port or 443, addresses)


def _acquisitions(resolution):
    candidates = resolution.get("candidates") if isinstance(resolution, dict) else None
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ToolFailure("fulltext_resolution", "A normalized article resolution is required")
    article = candidates[0]
    options = article.get("acquisition")
    if not isinstance(options, list):
        raise ToolFailure("fulltext_resolution", "Missing acquisition options")
    pmcid = article.get("pmcid")
    has_pmcid = isinstance(pmcid, str) and re.fullmatch(r"PMC[0-9]{1,12}", pmcid)
    trusted = {
        "europe_pmc": ("jats", f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"),
        "ncbi_bioc": ("bioc", f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"),
    }
    selected, seen = [], set()
    for option in options:
        if not isinstance(option, dict):
            continue
        kind, url, provider = option.get("kind"), option.get("url"), option.get("provider")
        if not isinstance(provider, str):
            continue
        if kind == "pmc_cloud":
            allowed = has_pmcid and provider == "pmc_cloud" and option.get("pmcid") == pmcid
        else:
            official = has_pmcid and trusted.get(provider) == (kind, url)
            legacy_europe = (provider == "europe_pmc" and not pmcid and kind == "jats"
                             and isinstance(url, str) and re.fullmatch(
                                 r"https://www\.ebi\.ac\.uk/europepmc/webservices/rest/PMC[0-9]{1,12}/fullTextXML", url))
            declared_oa = (kind in ("jats", "pdf") and option.get("is_oa") is True
                           and (provider not in trusted or legacy_europe))
            allowed = isinstance(url, str) and (official or declared_oa)
        key = url if isinstance(url, str) else kind
        if allowed and key not in seen:
            selected.append(option)
            seen.add(key)
        if len(selected) >= 4:
            break
    if not selected:
        raise ToolFailure("fulltext_unavailable", "No eligible acquisition for the resolved article")
    return article, selected


def _verify_jats_identity(body, article, *, require_identifier=False):
    try:
        root = etree.fromstring(inert_jats_content(body), etree.XMLParser(
            resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False))
    except (ValueError, etree.XMLSyntaxError) as exc:
        raise ToolFailure("fulltext_media", "Malformed JATS response") from exc
    if etree.QName(root).localname != "article":
        raise ToolFailure("fulltext_media", "JATS root is not an article")
    try:
        matched, version = matched_article_ids(root, article)
    except ValueError as exc:
        raise ToolFailure("fulltext_identity", str(exc)) from exc
    if require_identifier and not matched:
        raise ToolFailure("fulltext_media", "Official JATS has no matching primary article identifier")
    body_nodes = root.xpath("./*[local-name()='body']")
    if not body_nodes or not any(
            body.xpath(".//*[local-name()='fig' or local-name()='table-wrap']") or
            any(text.strip() for text in body.itertext()) for body in body_nodes):
        raise ToolFailure("fulltext_media", "JATS response contains no article body")
    return version


class FullTextFetcher:
    def __init__(self, *, transport=None, resolver=socket.getaddrinfo, timeout_seconds=30,
                 maximum_bytes=32 * 1024 * 1024, maximum_redirects=2):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
            raise ValueError("Full-text timeout must be finite and in (0,60] seconds")
        if type(maximum_bytes) is not int or not 1024 <= maximum_bytes <= 64 * 1024 * 1024:
            raise ValueError("Full-text byte limit is invalid")
        if type(maximum_redirects) is not int or not 0 <= maximum_redirects <= 3:
            raise ValueError("Full-text redirect limit is invalid")
        self.transport, self.resolver = transport, resolver
        self.timeout_seconds, self.maximum_bytes = timeout_seconds, maximum_bytes
        self.maximum_redirects = maximum_redirects

    def fetch(self, resolution, *, check_cancelled=lambda: None):
        article, options = _acquisitions(resolution)
        deadline = time.monotonic() + self.timeout_seconds
        attempts = []
        for option in options:
            check_cancelled()
            step = {"provider": option["provider"], "kind": option["kind"]}
            try:
                if time.monotonic() >= deadline:
                    raise ToolFailure("fulltext_timeout", "Article acquisition deadline exhausted")
                if option["kind"] == "pmc_cloud":
                    from .pmc_cloud import fetch_pmc_article
                    result = fetch_pmc_article(self, article["pmcid"], identity=article,
                                               deadline=deadline, check_cancelled=check_cancelled)
                else:
                    result = self._fetch_option(option, deadline, check_cancelled)
                if result.media_type == "application/xml":
                    version = _verify_jats_identity(result.body, article, require_identifier=bool(
                        article.get("pmcid") and option["provider"] in {"europe_pmc", "pmc_cloud"}))
                    if version is not None:
                        result = replace(result, acquisition={**result.acquisition, "pmc_version": version})
                elif option["kind"] == "bioc":
                    from .bioc import parse_bioc
                    parse_bioc(result.body, article["pmcid"])
                attempts.append({**step, "status": "selected"})
                return replace(result, attempts=tuple(attempts))
            except ToolFailure as exc:
                attempts.append({**step, "status": "failed", "code": exc.code, "message": str(exc)})
                exc.details = {**(exc.details or {}), "attempts": list(attempts)}
                if option is options[-1] or exc.status != "error" or exc.code not in {
                        "fulltext_media", "fulltext_transport", "fulltext_http", "fulltext_size",
                        "pmc_cloud_unavailable", "pmc_cloud_http"}:
                    raise

    def _fetch_option(self, option, deadline, check_cancelled):
        url = option["url"]
        for redirect_number in range(self.maximum_redirects + 1):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure("fulltext_timeout", "Full-text response budget exhausted")
            try:
                target = public_target(url, self.resolver)
            except OSError as exc:
                raise ToolFailure("fulltext_transport", "Public hostname resolution failed") from exc
            status, headers, body = self._request(target, deadline, check_cancelled)
            if status in {301, 302, 303, 307, 308}:
                if redirect_number == self.maximum_redirects:
                    raise ToolFailure("fulltext_redirect", "Full-text redirect limit exceeded")
                location = headers.get("location")
                if not isinstance(location, str) or not location:
                    raise ToolFailure("fulltext_redirect", "Full-text redirect lacks a location")
                url = urljoin(url, location)
                continue
            if status != 200:
                raise ToolFailure("fulltext_http", f"Full-text host returned HTTP {status}; no retry performed")
            media_type = self._validate_body(body, headers, option)
            return FullTextResponse(body, url, media_type, datetime.now(timezone.utc).isoformat(), option)
        raise ToolFailure("fulltext_redirect", "Full-text redirect limit exceeded")

    def _request(self, target, deadline, check_cancelled):
        if self.transport is None:
            return self._request_with_curl(target, deadline)
        try:
            with httpx.Client(transport=self.transport,
                    timeout=httpx.Timeout(min(5, max(.1, deadline - time.monotonic()))),
                    follow_redirects=False, trust_env=False) as client:
                with client.stream("GET", target.url, headers={"Accept":
                        "application/xml, application/pdf, application/json", "Accept-Encoding": "identity"}) as response:
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ToolFailure("fulltext_encoding", "Unexpected compressed full-text response")
                    length = response.headers.get("content-length")
                    if length is not None and (not length.isascii() or not length.isdigit()
                                               or int(length) > self.maximum_bytes):
                        raise ToolFailure("fulltext_size", "Full text exceeds configured byte limit")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        check_cancelled()
                        if time.monotonic() >= deadline:
                            raise ToolFailure("fulltext_timeout", "Full-text response budget exhausted")
                        size += len(chunk)
                        if size > self.maximum_bytes:
                            raise ToolFailure("fulltext_size", "Full text exceeds configured byte limit")
                        chunks.append(chunk)
                    return response.status_code, {key.lower(): value for key, value in response.headers.items()}, b"".join(chunks)
        except httpx.HTTPError as exc:
            raise ToolFailure("fulltext_transport", f"Full-text request failed: {type(exc).__name__}") from exc

    def _request_with_curl(self, target, deadline):
        """Pin the validated IP while retaining TLS hostname verification/SNI."""
        remaining = max(.1, deadline - time.monotonic())
        selected = next((address for address in target.addresses if ipaddress.ip_address(address).version == 4),
                        target.addresses[0])
        rendered_address = f"[{selected}]" if ipaddress.ip_address(selected).version == 6 else selected
        pinned = f"{target.hostname}:{target.port}:{rendered_address}"
        with tempfile.TemporaryDirectory(prefix="adjagent-fulltext-") as directory:
            body_path, header_path = Path(directory) / "body", Path(directory) / "headers"
            command = [CURL_EXECUTABLE, "--disable", "--silent", "--show-error", "--no-location",
                "--proto", "=https", "--noproxy", "*", "--resolve", pinned,
                "--connect-timeout", str(min(5, remaining)), "--max-time", str(remaining),
                "--max-filesize", str(self.maximum_bytes), "--header", "Accept-Encoding: identity",
                "--header", "Accept: application/xml, application/pdf, application/json",
                "--output", str(body_path), "--dump-header", str(header_path),
                "--write-out", "%{http_code}", target.url]
            try:
                completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           timeout=remaining + 2, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ToolFailure("fulltext_timeout", "Full-text curl response budget exhausted") from exc
            except OSError as exc:
                raise ToolFailure("fulltext_transport", f"Full-text curl could not start: {type(exc).__name__}") from exc
            if completed.returncode != 0:
                code = "fulltext_size" if completed.returncode == 63 else "fulltext_transport"
                raise ToolFailure(code, f"Full-text curl failed without retry (exit {completed.returncode})")
            status_text = completed.stdout.decode("ascii", errors="strict")
            if len(status_text) != 3 or not status_text.isdigit():
                raise ToolFailure("fulltext_transport", "Full-text curl returned no valid HTTP status")
            body = body_path.read_bytes()
            if len(body) > self.maximum_bytes:
                raise ToolFailure("fulltext_size", "Full text exceeds configured byte limit")
            header_lines = header_path.read_text(encoding="iso-8859-1").splitlines()
            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            if headers.get("content-encoding", "identity").lower() != "identity":
                raise ToolFailure("fulltext_encoding", "Unexpected compressed full-text response")
            return int(status_text), headers, body

    @staticmethod
    def _validate_body(body, headers, option):
        declared = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        prefix = body.lstrip()[:32].lower()
        if option.get("kind") == "jats":
            if declared not in {"application/xml", "text/xml", "application/jats+xml", ""}:
                raise ToolFailure("fulltext_media", "JATS candidate returned a non-XML media type")
            try:
                prefix = inert_jats_content(body).removeprefix(b"\xef\xbb\xbf").lstrip()[:32].lower()
            except ValueError as exc:
                raise ToolFailure("fulltext_media", "JATS candidate contains unsupported XML declarations") from exc
            if not (prefix.startswith(b"<?xml") or prefix.startswith(b"<article")):
                raise ToolFailure("fulltext_media", "JATS candidate did not return XML article bytes")
            return "application/xml"
        if option.get("kind") == "bioc":
            # The fixed NIH endpoint may wrap valid BioC JSON in a text media type.
            # The caller validates actual document structure and article identity.
            return "application/bioc+json"
        if option.get("kind") == "pdf":
            if declared not in {"application/pdf", "application/octet-stream", ""} or not body.startswith(b"%PDF-"):
                raise ToolFailure("fulltext_media", "OA PDF candidate did not return PDF bytes")
            return "application/pdf"
        raise ToolFailure("fulltext_media", "Unsupported full-text candidate kind")
