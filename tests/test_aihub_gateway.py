"""AI Hub gateway wrappers for web_search (Serper) and web_page_read (Jina)."""

import json
from pathlib import Path

import pytest

from adj_partner.native.gateway import GatewayClient
from adj_partner.native.web_search import (FallbackSearch, MwmblSearch, SerperGatewaySearch, WebSearchResponse,
                                           web_search_from_environment)
from adj_partner.tools.executor import ToolFailure
from adj_partner.web_access import (FallbackFetcher, JinaGatewayReader, WebPageFetcher, WebPageResponse,
                                    markdown_paragraphs, web_page_fetcher_from_environment)

GATEWAY_NAMES = ("ADJ_WEB_SEARCH_PROVIDER", "ADJ_AIHUB_GATEWAY_URL", "ADJ_SERPER_GATEWAY_APP_ID",
                 "ADJ_SERPER_GATEWAY_APP_KEY", "ADJ_WEB_PAGE_READER", "ADJ_JINA_GATEWAY_APP_ID",
                 "ADJ_JINA_GATEWAY_APP_KEY")
GW = "http://gw.test:8080"


def fake_curl(monkeypatch, body, *, status=b"200", headers=""):
    captured = {}

    class Completed:
        returncode, stderr = 0, b""
        stdout = body + b"\n" + status

    def run(arguments, **kwargs):
        captured["arguments"], captured["input"] = arguments, kwargs["input"]
        dump = arguments[arguments.index("--dump-header") + 1]
        Path(dump).write_text("HTTP/1.1 200 OK\r\n" + headers)
        return Completed()

    monkeypatch.setattr("adj_partner.native.gateway.subprocess.run", run)
    return captured


def test_serper_gateway_keeps_credential_off_argv_and_parses_organic(monkeypatch):
    body = json.dumps({"organic": [{"title": "Apple", "link": "https://www.apple.com/", "snippet": "S", "position": 1}]})
    captured = fake_curl(monkeypatch, body.encode(), headers="X-Serper-Link-Resolve: skipped\r\n")
    search = SerperGatewaySearch(GatewayClient("fixture_id", "fixture-secret", "serper", base_url="http://gw.test:8080"))
    response = search.search("apple", max_results=3)
    assert response.records[0]["url"] == "https://www.apple.com/"
    assert response.request_url == "http://gw.test:8080/search"
    assert "fixture-secret" not in json.dumps(captured["arguments"])
    assert b"Bearer fixture_id:fixture-secret?provider=serper&timeout=60" in captured["input"]
    assert search.secrets == ("fixture-secret",)


def test_serper_gateway_marks_unresolved_google_redirects(monkeypatch):
    link = "https://www.google.com/goto?url=CAES123"
    body = json.dumps({"organic": [{"title": "T", "link": link, "snippet": "S"}]})
    fake_curl(monkeypatch, body.encode(), headers="X-Serper-Link-Resolve: partial\r\n")
    search = SerperGatewaySearch(GatewayClient("fixture_id", "fixture-secret", "serper", base_url=GW))
    assert search.search("q").records[0]["url_kind"] == "google_redirect_unresolved"


def test_gateway_http_error_is_tool_failure(monkeypatch):
    fake_curl(monkeypatch, b"{}", status=b"401", headers='X-Error-Data: {"message": "auth"}\r\n')
    search = SerperGatewaySearch(GatewayClient("fixture_id", "fixture-secret", "serper", base_url=GW))
    with pytest.raises(Exception) as failure:
        search.search("q")
    assert getattr(failure.value, "code", None) == "web_search_http"


def test_jina_reader_returns_markdown_and_denies_credential_urls(monkeypatch):
    body = json.dumps({"code": 200, "data": {"url": "https://example.com/", "title": "Example",
                                             "content": "Heading\n\nFirst paragraph.\n\nSecond."}})
    captured = fake_curl(monkeypatch, body.encode())
    reader = JinaGatewayReader(GatewayClient("fixture_id", "fixture-secret", "jina_ai", base_url=GW))
    response = reader.fetch("https://example.com")
    assert response.media == "text/markdown" and response.url == "https://example.com/"
    assert markdown_paragraphs(response.body.decode()) == ["Heading", "First paragraph.", "Second."]
    assert b"provider=jina_ai" in captured["input"]
    with pytest.raises(Exception) as failure:
        reader.fetch("https://example.com/x?api_key=abc")
    assert getattr(failure.value, "code", None) == "web_page_credentials"


def test_environment_selects_gateway_backends(tmp_path, monkeypatch):
    for name in GATEWAY_NAMES:
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / "configured.env"
    configured.write_text(f"ADJ_AIHUB_GATEWAY_URL={GW}\nADJ_WEB_SEARCH_PROVIDER=serper_gateway\n"
                          "ADJ_SERPER_GATEWAY_APP_ID=a_id\nADJ_SERPER_GATEWAY_APP_KEY=a-key\n"
                          "ADJ_WEB_PAGE_READER=jina_gateway\nADJ_JINA_GATEWAY_APP_ID=b_id\nADJ_JINA_GATEWAY_APP_KEY=b-key\n")
    assert isinstance(web_search_from_environment(tmp_path, configured), SerperGatewaySearch)
    reader = web_page_fetcher_from_environment(tmp_path, configured)
    assert isinstance(reader, FallbackFetcher) and isinstance(reader.fetchers[0], JinaGatewayReader)
    assert isinstance(reader.fetchers[1], WebPageFetcher)
    configured.write_text("ADJ_WEB_PAGE_READER=\n")
    assert web_page_fetcher_from_environment(tmp_path, configured) is None
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=serper_gateway\n")
    with pytest.raises(ValueError, match="APP-ID"):
        web_search_from_environment(tmp_path, configured)
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=serper_gateway\nADJ_SERPER_GATEWAY_APP_ID=a_id\n"
                          "ADJ_SERPER_GATEWAY_APP_KEY=a-key\n")
    with pytest.raises(ValueError, match="ADJ_AIHUB_GATEWAY_URL"):
        web_search_from_environment(tmp_path, configured)


def test_comma_list_builds_ordered_chain_and_skips_unconfigured_fallbacks(tmp_path, monkeypatch):
    for name in GATEWAY_NAMES + ("ADJ_SERPER_API_KEY", "ADJ_WEB_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / "configured.env"
    configured.write_text(f"ADJ_AIHUB_GATEWAY_URL={GW}\nADJ_WEB_SEARCH_PROVIDER=serper_gateway,serper,mwmbl\n"
                          "ADJ_SERPER_GATEWAY_APP_ID=a_id\nADJ_SERPER_GATEWAY_APP_KEY=a-key\n")
    chain = web_search_from_environment(tmp_path, configured)
    assert isinstance(chain, FallbackSearch)
    assert [type(s) for s in chain.providers] == [SerperGatewaySearch, MwmblSearch]
    assert chain.provider == "serper_gateway" and chain.secrets == ("a-key",)


class _Failing:
    provider, secrets = "first", ("s1",)
    def search(self, *a, **k): raise ToolFailure("web_search_http", "down")
    def fetch(self, *a, **k): raise ToolFailure("web_page_http", "down")


class _Working:
    provider, secrets = "second", ()
    def search(self, *a, **k): return WebSearchResponse([], "https://second.test", "now", b"{}")
    def fetch(self, url, **k): return WebPageResponse(b"<p>x</p>", url, "now")


def test_search_and_reader_fall_back_on_tool_failure():
    response = FallbackSearch([_Failing(), _Working()]).search("q")
    assert response.provider == "second" and response.fallback_from == ("first: web_search_http",)
    page = FallbackFetcher([_Failing(), _Working()]).fetch("https://example.com")
    assert page.media == "text/html" and page.fallback_from == ("_Failing: web_page_http",)
    with pytest.raises(ToolFailure) as failure:
        FallbackSearch([_Failing()]).search("q")
    assert failure.value.code == "web_search_all_failed"
