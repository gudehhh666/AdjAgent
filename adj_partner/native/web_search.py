"""Bounded fixed-origin web search. Provider snippets are discovery material."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import threading
import time
from urllib.parse import urlencode, urlsplit

import httpx

from ..cli_defaults import _explicit_values
from ..tools.catalog import canonical_json
from ..tools.executor import ToolFailure


ENDPOINT = "https://google.serper.dev/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
MWMBL_ENDPOINT = "https://api.mwmbl.org/api/v2/search/"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
CURL_EXECUTABLE = "/usr/bin/curl"
_MWMBL_LOCK = threading.Lock()
_MWMBL_LAST_REQUEST = -math.inf


@dataclass(frozen=True)
class WebSearchResponse:
    records: list[dict]
    request_url: str
    received_at: str
    raw: bytes = field(repr=False)
    provider: str = ""
    fallback_from: tuple = ()


def _text(value, limit):
    if not isinstance(value, str) or len(value) > limit * 4:
        raise ValueError("Search result text is malformed or too large")
    clean = " ".join(value.split())
    clean = "".join(char for char in clean if ord(char) >= 32 and ord(char) != 127)
    return clean[:limit]


def _url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096 or any(ord(char) < 32 for char in value):
        raise ValueError("Search result URL is malformed")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Search result URL is not a public HTTP(S) locator")
    return value


def _record(item):
    if not isinstance(item, dict):
        raise ValueError("Search result must be an object")
    title, url = _text(item.get("title"), 1000), _url(item.get("link"))
    if not title:
        raise ValueError("Search result has no title")
    snippet = _text(item.get("snippet", ""), 4000)
    result = {"title": title, "url": url, "snippet": snippet,
              "text_availability": "search_snippet" if snippet else "title_only"}
    position = item.get("position")
    if type(position) is int and 1 <= position <= 10000:
        result["position"] = position
    date = item.get("date")
    if isinstance(date, str) and date:
        result["date_label"] = _text(date, 200)
    return result


def _serper_records(raw, max_results):
    try:
        body = json.loads(raw)
        canonical_json(body)
        organic = body.get("organic", [])
        if not isinstance(body, dict) or not isinstance(organic, list) or len(organic) > 1000:
            raise ValueError("Malformed organic result list")
        records, seen = [], set()
        for item in organic:
            record = _record(item)
            if record["url"] in seen:
                continue
            seen.add(record["url"])
            records.append(record)
            if len(records) >= max_results:
                break
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ToolFailure("web_search_response", f"Malformed Serper response: {str(exc)[:1000]}") from exc
    return records


class SerperGatewaySearch:
    """Serper through the AI Hub gateway; the gateway resolves Google goto links by default."""

    provider = "serper_gateway"

    def __init__(self, client):
        self.client = client

    @property
    def secrets(self):
        return self.client.secrets

    def search(self, query, *, max_results=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Web search query must be 1..2000 characters")
        if type(max_results) is not int or not 1 <= max_results <= 10:
            raise ValueError("max_results must be an integer in 1..10")
        canonical_json(query)
        check_cancelled()
        headers, raw = self.client.post("/search", {"q": query, "num": max_results}, failure_prefix="web_search")
        check_cancelled()
        records = _serper_records(raw, max_results)
        # Unresolved google.com/goto links are not readable page locators.
        if headers.get("x-serper-link-resolve") in {"partial", "failed", "raw", "disabled"}:
            for record in records:
                if urlsplit(record["url"]).hostname in {"www.google.com", "google.com"} and "/goto" in record["url"]:
                    record["url_kind"] = "google_redirect_unresolved"
        return WebSearchResponse(records, self.client.base_url + "/search",
                                 datetime.now(timezone.utc).isoformat(), raw)


class SerperSearch:
    def __init__(self, api_key, *, transport=None, timeout_seconds=20):
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                or any(not 33 <= ord(char) <= 126 for char in api_key)):
            raise ValueError("Serper API key is missing or malformed")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30):
            raise ValueError("Serper timeout must be finite and in (0,30] seconds")
        self.api_key, self.transport, self.timeout_seconds = api_key, transport, timeout_seconds

    @property
    def secrets(self):
        return (self.api_key,)

    @property
    def provider(self):
        return "serper"

    @classmethod
    def from_environment(cls, project_root, env_file=None):
        try:
            selected = _explicit_values(project_root, env_file)
            provider = os.environ.get("ADJ_WEB_SEARCH_PROVIDER")
            key = os.environ.get("ADJ_WEB_SEARCH_API_KEY")
            if not provider:
                provider = selected.get("ADJ_WEB_SEARCH_PROVIDER")
            if not key:
                key = selected.get("ADJ_WEB_SEARCH_API_KEY")
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            raise ValueError("Serper configuration is missing, malformed or unsafe") from None
        provider, key = (provider or "").strip().lower(), key or ""
        if not provider and not key:
            return None
        if provider != "serper" or not key:
            raise ValueError("Serper requires ADJ_WEB_SEARCH_PROVIDER=serper and ADJ_WEB_SEARCH_API_KEY")
        return cls(key)

    def search(self, query, *, max_results=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Web search query must be 1..2000 characters")
        if type(max_results) is not int or not 1 <= max_results <= 10:
            raise ValueError("max_results must be an integer in 1..10")
        canonical_json(query)
        deadline = time.monotonic() + self.timeout_seconds

        def checkpoint():
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure("web_search_timeout", "Serper elapsed response budget exhausted")

        checkpoint()
        if self.transport is None:
            raw = self._search_with_curl(query, max_results)
        else:
            raw = self._search_with_httpx(query, max_results, checkpoint)
        checkpoint()
        return WebSearchResponse(_serper_records(raw, max_results), ENDPOINT,
                                 datetime.now(timezone.utc).isoformat(), raw)

    def _search_with_curl(self, query, max_results):
        """Use the system curl TLS stack while keeping the credential out of argv."""
        payload = canonical_json({"q": query, "num": max_results})
        arguments = [
            CURL_EXECUTABLE,
            "--silent",
            "--show-error",
            "--no-location",
            "--proto",
            "=https",
            "--connect-timeout",
            str(min(5, self.timeout_seconds)),
            "--max-time",
            str(self.timeout_seconds),
            "--max-filesize",
            str(MAX_RESPONSE_BYTES),
            "--request",
            "POST",
            "--header",
            "Accept: application/json",
            "--header",
            "Accept-Encoding: identity",
            "--header",
            "Content-Type: application/json",
            "--header",
            "@-",
            "--data-binary",
            payload,
            "--write-out",
            "\n%{http_code}",
            ENDPOINT,
        ]
        try:
            completed = subprocess.run(
                arguments,
                input=f"X-API-KEY: {self.api_key}\n".encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds + 2,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolFailure("web_search_timeout", "Serper curl response budget exhausted") from exc
        except OSError as exc:
            raise ToolFailure("web_search_transport", f"Serper curl could not start: {type(exc).__name__}") from exc
        raw, separator, status = completed.stdout.rpartition(b"\n")
        if completed.returncode != 0:
            raise ToolFailure("web_search_transport", f"Serper curl failed without retry (exit {completed.returncode})")
        if not separator or len(status) != 3 or not status.isdigit():
            raise ToolFailure("web_search_transport", "Serper curl returned no valid HTTP status")
        if status != b"200":
            try:
                failure = json.loads(raw)
            except (json.JSONDecodeError, UnicodeError):
                failure = None
            if (status == b"400" and isinstance(failure, dict)
                    and failure.get("message") == "Not enough credits"):
                raise ToolFailure("web_search_quota", "Serper account has insufficient search credits")
            raise ToolFailure("web_search_http", f"Serper returned HTTP {status.decode()}; no redirects/retries followed")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ToolFailure("web_search_size", "Serper response exceeds 2 MiB")
        return raw

    def _search_with_httpx(self, query, max_results, checkpoint):
        """Injected transports keep unit tests deterministic and network-free."""
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(min(5, self.timeout_seconds)),
                              follow_redirects=False, trust_env=False) as client:
                with client.stream("POST", ENDPOINT, json={"q": query, "num": max_results},
                                   headers={"Accept": "application/json", "Accept-Encoding": "identity",
                                            "Content-Type": "application/json", "X-API-KEY": self.api_key}) as response:
                    checkpoint()
                    if response.status_code != 200:
                        raise ToolFailure("web_search_http", f"Serper returned HTTP {response.status_code}; no redirects/retries followed")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ToolFailure("web_search_encoding", "Unexpected compressed Serper response")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        checkpoint()
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ToolFailure("web_search_size", "Serper response exceeds 2 MiB")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise ToolFailure("web_search_transport", f"Serper request failed without retry: {type(exc).__name__}") from exc


def _tavily_record(item):
    if not isinstance(item, dict):
        raise ValueError("Tavily search result must be an object")
    title, url = _text(item.get("title"), 1000), _url(item.get("url"))
    if not title:
        raise ValueError("Tavily search result has no title")
    snippet = _text(item.get("content", ""), 4000)
    result = {"title": title, "url": url, "snippet": snippet}
    score = item.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1:
        result["score"] = score
    result["text_availability"] = "search_snippet" if snippet else "title_only"
    return result


class TavilySearch:
    """Bounded Tavily basic search; result content remains a provider snippet."""

    def __init__(self, api_key, *, transport=None, timeout_seconds=20):
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                or any(not 33 <= ord(char) <= 126 for char in api_key)):
            raise ValueError("Tavily API key is missing or malformed")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30):
            raise ValueError("Tavily timeout must be finite and in (0,30] seconds")
        self.api_key, self.transport, self.timeout_seconds = api_key, transport, timeout_seconds

    @property
    def secrets(self):
        return (self.api_key,)

    @property
    def provider(self):
        return "tavily"

    def search(self, query, *, max_results=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Web search query must be 1..2000 characters")
        if type(max_results) is not int or not 1 <= max_results <= 10:
            raise ValueError("max_results must be an integer in 1..10")
        payload = {"api_key": self.api_key, "query": query, "max_results": max_results,
                   "search_depth": "basic", "include_answer": False,
                   "include_raw_content": False, "include_images": False}
        deadline = time.monotonic() + self.timeout_seconds

        def checkpoint():
            check_cancelled()
            if time.monotonic() >= deadline:
                raise ToolFailure("web_search_timeout", "Tavily elapsed response budget exhausted")

        checkpoint()
        raw = self._search_with_curl(payload) if self.transport is None else self._search_with_httpx(payload, checkpoint)
        checkpoint()
        try:
            body = json.loads(raw)
            canonical_json(body)
            items = body.get("results", [])
            if not isinstance(body, dict) or not isinstance(items, list) or len(items) > 1000:
                raise ValueError("Malformed Tavily result list")
            records, seen = [], set()
            for item in items:
                record = _tavily_record(item)
                if record["url"] in seen:
                    continue
                seen.add(record["url"])
                records.append(record)
                if len(records) >= max_results:
                    break
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeError, RecursionError) as exc:
            raise ToolFailure("web_search_response", f"Malformed Tavily response: {str(exc)[:1000]}") from exc
        return WebSearchResponse(records, TAVILY_ENDPOINT, datetime.now(timezone.utc).isoformat(), raw)

    def _search_with_curl(self, payload):
        arguments = [CURL_EXECUTABLE, "--silent", "--show-error", "--no-location", "--proto", "=https",
            "--connect-timeout", str(min(5, self.timeout_seconds)), "--max-time", str(self.timeout_seconds),
            "--max-filesize", str(MAX_RESPONSE_BYTES), "--request", "POST",
            "--header", "Accept: application/json", "--header", "Accept-Encoding: identity",
            "--header", "Content-Type: application/json", "--data-binary", "@-",
            "--write-out", "\n%{http_code}", TAVILY_ENDPOINT]
        try:
            completed = subprocess.run(arguments, input=canonical_json(payload).encode(), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, timeout=self.timeout_seconds + 2, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ToolFailure("web_search_timeout", "Tavily curl response budget exhausted") from exc
        except OSError as exc:
            raise ToolFailure("web_search_transport", f"Tavily curl could not start: {type(exc).__name__}") from exc
        raw, separator, status = completed.stdout.rpartition(b"\n")
        if completed.returncode != 0 or not separator or len(status) != 3 or not status.isdigit():
            raise ToolFailure("web_search_transport", "Tavily curl failed or returned no valid HTTP status")
        if status != b"200":
            raise ToolFailure("web_search_http", f"Tavily returned HTTP {status.decode()}; no redirects/retries followed")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ToolFailure("web_search_size", "Tavily response exceeds 2 MiB")
        return raw

    def _search_with_httpx(self, payload, checkpoint):
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(min(5, self.timeout_seconds)),
                              follow_redirects=False, trust_env=False) as client:
                with client.stream("POST", TAVILY_ENDPOINT, json=payload,
                    headers={"Accept": "application/json", "Accept-Encoding": "identity",
                             "Content-Type": "application/json"}) as response:
                    checkpoint()
                    if response.status_code != 200:
                        raise ToolFailure("web_search_http", f"Tavily returned HTTP {response.status_code}; no redirects/retries followed")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ToolFailure("web_search_encoding", "Unexpected compressed Tavily response")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        checkpoint()
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ToolFailure("web_search_size", "Tavily response exceeds 2 MiB")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise ToolFailure("web_search_transport", f"Tavily request failed without retry: {type(exc).__name__}") from exc


def _mwmbl_slot(deadline, check_cancelled):
    """Pace anonymous requests across all sessions in this host process."""
    global _MWMBL_LAST_REQUEST
    while True:
        check_cancelled()
        with _MWMBL_LOCK:
            now = time.monotonic()
            if now >= deadline:
                raise ToolFailure("web_search_timeout", "Mwmbl request budget exhausted while waiting")
            delay = _MWMBL_LAST_REQUEST + 1 - now
            if delay <= 0:
                _MWMBL_LAST_REQUEST = now
                return
        time.sleep(min(delay, 0.05))


def _mwmbl_value(value):
    if not isinstance(value, list):
        return value
    # Highlighted fragments already include spaces between words.
    if any(not isinstance(part, dict) or not isinstance(part.get("value"), str) for part in value):
        raise ValueError("Malformed Mwmbl highlighted text")
    return "".join(part["value"] for part in value)


def _mwmbl_http_failure(status, raw):
    if status == b"429":
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, str):
            if detail.startswith("Monthly quota exceeded:"):
                return ToolFailure("web_search_quota", "Mwmbl reported monthly search quota exhaustion")
            if detail.startswith("Rate limit exceeded:"):
                return ToolFailure("web_search_rate_limit", "Mwmbl reported a request rate limit")
    return ToolFailure("web_search_http", f"Mwmbl returned HTTP {status.decode()}; no retries or redirects followed")


class MwmblSearch:
    """Anonymous v2 search. The requested result count is applied locally."""

    provider = "mwmbl"
    secrets = ()

    def search(self, query, *, max_results=5, check_cancelled=lambda: None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Web search query must be 1..2000 characters")
        if type(max_results) is not int or not 1 <= max_results <= 10:
            raise ValueError("max_results must be an integer in 1..10")
        request_url = MWMBL_ENDPOINT + "?" + urlencode({"q": query})
        deadline = time.monotonic() + 20
        _mwmbl_slot(deadline, check_cancelled)
        remaining = max(0.001, deadline - time.monotonic())
        command = [CURL_EXECUTABLE, "--disable", "--silent", "--no-location", "--proto", "=https",
                   "--connect-timeout", str(min(5, remaining)), "--max-time", str(remaining),
                   "--max-filesize", str(MAX_RESPONSE_BYTES), "--header", "Accept-Encoding: identity",
                   "--header", "Accept: application/json", "--write-out", "\n%{http_code}", request_url]
        try:
            # Ignore user curl defaults and ambient credentials for this anonymous endpoint.
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, env={}, timeout=remaining + 1, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ToolFailure("web_search_timeout", "Mwmbl request timed out") from exc
        except OSError as exc:
            raise ToolFailure("web_search_transport", "Mwmbl curl could not start") from exc
        check_cancelled()
        if completed.returncode == 28 or time.monotonic() >= deadline:
            raise ToolFailure("web_search_timeout", "Mwmbl request timed out")
        if completed.returncode == 63:
            raise ToolFailure("web_search_size", "Mwmbl response exceeds 2 MiB")
        raw, separator, status = completed.stdout.rpartition(b"\n")
        if completed.returncode or not separator or len(status) != 3 or not status.isdigit():
            raise ToolFailure("web_search_transport", f"Mwmbl curl failed without retry (exit {completed.returncode})")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ToolFailure("web_search_size", "Mwmbl response exceeds 2 MiB")
        if status != b"200":
            raise _mwmbl_http_failure(status, raw)
        try:
            body = json.loads(raw)
            canonical_json(body)
            items = body.get("results") if isinstance(body, dict) else None
            if not isinstance(items, list) or len(items) > 1000:
                raise ValueError("Malformed Mwmbl result list")
            records, seen = [], set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("Malformed Mwmbl result")
                record = _record({"link": item.get("url"), "title": _mwmbl_value(item.get("title")),
                                  "snippet": _mwmbl_value(item.get("content", item.get("extract", "")))})
                if record["url"] not in seen:
                    records.append(record)
                    seen.add(record["url"])
                if len(records) == max_results:
                    break
        except (ValueError, TypeError, RecursionError) as exc:
            raise ToolFailure("web_search_response", "Malformed Mwmbl search response") from exc
        return WebSearchResponse(records, request_url, datetime.now(timezone.utc).isoformat(), raw)


def web_search_from_environment(project_root, env_file=None):
    try:
        selected = _explicit_values(project_root, env_file)
        provider = os.environ.get("ADJ_WEB_SEARCH_PROVIDER") or selected.get("ADJ_WEB_SEARCH_PROVIDER")
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("Web-search configuration is missing, malformed or unsafe") from None
    provider = (provider or "").strip().lower()
    if "," in provider:
        names = [name.strip() for name in provider.split(",") if name.strip()]
        if not names:
            raise ValueError("ADJ_WEB_SEARCH_PROVIDER fallback list is empty")
        chain = [_single_provider(names[0], selected, project_root, env_file)]
        for name in names[1:]:
            # Fallbacks without credentials are skipped rather than blocking the preferred provider.
            try:
                chain.append(_single_provider(name, selected, project_root, env_file))
            except ValueError:
                continue
        return FallbackSearch([search for search in chain if search is not None])
    return _single_provider(provider, selected, project_root, env_file)


class FallbackSearch:
    """Try providers in order; a ToolFailure moves to the next provider."""

    def __init__(self, providers):
        if not providers:
            raise ValueError("Web search fallback chain has no configured provider")
        self.providers = providers

    @property
    def provider(self):
        return self.providers[0].provider

    @property
    def secrets(self):
        return tuple(secret for search in self.providers for secret in search.secrets)

    def search(self, query, *, max_results=5, check_cancelled=lambda: None):
        failures = []
        for search in self.providers:
            try:
                response = search.search(query, max_results=max_results, check_cancelled=check_cancelled)
            except ToolFailure as exc:
                failures.append(f"{search.provider}: {exc.code}")
                continue
            return WebSearchResponse(response.records, response.request_url, response.received_at, response.raw,
                                     provider=search.provider, fallback_from=tuple(failures))
        raise ToolFailure("web_search_all_failed", "All web search providers failed: " + "; ".join(failures))


def _single_provider(provider, selected, project_root, env_file):
    if provider == "mwmbl":
        return MwmblSearch()
    if provider == "serper_gateway":
        from .gateway import GatewayClient, settings
        values = settings(project_root, env_file, ("ADJ_AIHUB_GATEWAY_URL", "ADJ_SERPER_GATEWAY_APP_ID",
                                                   "ADJ_SERPER_GATEWAY_APP_KEY"))
        return SerperGatewaySearch(GatewayClient(values["ADJ_SERPER_GATEWAY_APP_ID"],
                                                 values["ADJ_SERPER_GATEWAY_APP_KEY"], "serper",
                                                 base_url=values["ADJ_AIHUB_GATEWAY_URL"] or None))
    specific_name = {"serper": "ADJ_SERPER_API_KEY", "tavily": "ADJ_TAVILY_API_KEY"}.get(provider)
    specific = (os.environ.get(specific_name) or selected.get(specific_name)) if specific_name else None
    key = specific or os.environ.get("ADJ_WEB_SEARCH_API_KEY") or selected.get("ADJ_WEB_SEARCH_API_KEY") or ""
    if not provider and not key:
        return None
    if not key:
        raise ValueError("Configured web search requires ADJ_WEB_SEARCH_API_KEY")
    if provider == "serper":
        return SerperSearch(key)
    if provider == "tavily":
        return TavilySearch(key)
    raise ValueError("ADJ_WEB_SEARCH_PROVIDER must be serper, serper_gateway, tavily or mwmbl")
