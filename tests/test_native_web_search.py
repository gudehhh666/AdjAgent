"""Serper is a bounded host retrieval tool, not evidence that a page was read."""

import json

import httpx
import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.native.web_search import SerperSearch
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


def scope(tmp_path, search, *, network=True):
    store = Store(tmp_path / "state")
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research, web_search=search)
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allow_network=network,
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Serper fixture")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return store, research, catalog, ToolExecutor(store, catalog), session, run


def test_serper_search_registers_bounded_snippet_sources_without_claiming_page_text(tmp_path):
    requests = []
    body = {"organic": [{"title": "Fixture paper", "link": "https://example.org/paper",
        "snippet": "A search-engine snippet only.", "position": 1}]}
    transport = httpx.MockTransport(lambda request: (requests.append(request), httpx.Response(200, json=body))[1])
    search = SerperSearch("fixture-serper-key", transport=transport)
    store, research, catalog, executor, sid, rid = scope(tmp_path, search)
    assert catalog.get("web_search").spec.network is True
    result = executor.execute(ToolCall("search-1", "web_search",
        {"query": "zinc vaccine adjuvant", "max_results": 5}), sid, rid)
    assert result.status == "ok", result.to_dict()
    assert result.data["provider"] == "serper" and result.data["returned"] == 1
    hit = result.data["results"][0]
    assert hit["text_availability"] == "search_snippet" and hit["snippet"] == "A search-engine snippet only."
    source = research.get_source(sid, rid, hit["source_id"])
    record = json.loads(store.read_artifact(source["artifact_id"]))
    assert record["url"] == "https://example.org/paper" and record["text_availability"] == "search_snippet"
    assert source["kind"] == "web" and source["provenance"]["provider"] == "Serper"
    assert source["provenance"]["response_artifact_id"] in {a["id"] for a in store.artifacts(sid)}
    assert requests[0].headers["X-API-KEY"] == "fixture-serper-key"
    assert "fixture-serper-key" not in json.dumps(result.to_dict())


def test_serper_search_respects_session_network_policy(tmp_path):
    requests = []
    transport = httpx.MockTransport(lambda request: (requests.append(request), httpx.Response(200, json={"organic": []}))[1])
    _, research, _, executor, sid, rid = scope(tmp_path, SerperSearch("fixture-key", transport=transport), network=False)
    result = executor.execute(ToolCall("search-1", "web_search", {"query": "fixture"}), sid, rid)
    assert result.status == "denied" and requests == [] and research.list_sources(sid, rid) == []


def test_serper_configuration_requires_both_explicit_provider_and_key(tmp_path, monkeypatch):
    for name in ("ADJ_WEB_SEARCH_PROVIDER", "ADJ_WEB_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("ADJ_WEB_SEARCH_PROVIDER=\nADJ_WEB_SEARCH_API_KEY=\n")
    assert SerperSearch.from_environment(tmp_path, empty) is None
    configured = tmp_path / "configured.env"
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=serper\nADJ_WEB_SEARCH_API_KEY=fixture-key\n")
    assert SerperSearch.from_environment(tmp_path, configured).secrets == ("fixture-key",)
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=tavily\nADJ_WEB_SEARCH_API_KEY=fixture-key\n")
    with pytest.raises(ValueError, match="Serper"):
        SerperSearch.from_environment(tmp_path, configured)


def test_production_curl_transport_keeps_api_key_out_of_process_arguments(monkeypatch):
    captured = {}
    body = json.dumps({"organic": [{"title": "Fixture", "link": "https://example.org", "snippet": "Snippet"}]}).encode()
    class Completed:
        returncode, stdout, stderr = 0, body + b"\n200", b""
    def fake_run(arguments, **kwargs):
        captured["arguments"], captured["kwargs"] = arguments, kwargs
        return Completed()
    monkeypatch.setattr("adj_partner.native.web_search.subprocess.run", fake_run)
    response = SerperSearch("fixture-curl-secret").search("fixture", max_results=1)
    assert response.records[0]["title"] == "Fixture"
    assert "fixture-curl-secret" not in json.dumps(captured["arguments"])
    assert b"fixture-curl-secret" in captured["kwargs"]["input"]


def test_production_curl_transport_reports_exhausted_serper_credits(monkeypatch):
    class Completed:
        returncode, stdout, stderr = 0, b'{"message":"Not enough credits","statusCode":400}\n400', b""
    monkeypatch.setattr("adj_partner.native.web_search.subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(Exception) as failure:
        SerperSearch("fixture-key").search("fixture")
    assert getattr(failure.value, "code", None) == "web_search_quota"
    assert "credit" in str(failure.value).lower()


def test_tavily_configuration_and_results_remain_search_snippets(tmp_path, monkeypatch):
    from adj_partner.native.web_search import TavilySearch, web_search_from_environment

    for name in ("ADJ_WEB_SEARCH_PROVIDER", "ADJ_WEB_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / "configured.env"
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=tavily\nADJ_WEB_SEARCH_API_KEY=fixture-key\n")
    search = web_search_from_environment(tmp_path, configured)
    assert isinstance(search, TavilySearch)
    body = {"results": [{"title": "Paper", "url": "https://example.org/paper",
                          "content": "Provider excerpt only.", "score": 0.91}]}
    search.transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))

    response = search.search("fixture", max_results=2)

    assert response.records == [{"title": "Paper", "url": "https://example.org/paper",
        "snippet": "Provider excerpt only.", "score": 0.91, "text_availability": "search_snippet"}]


def test_provider_specific_web_key_precedes_legacy_generic_key(tmp_path, monkeypatch):
    from adj_partner.native.web_search import web_search_from_environment

    for name in ("ADJ_WEB_SEARCH_PROVIDER", "ADJ_WEB_SEARCH_API_KEY",
                 "ADJ_SERPER_API_KEY", "ADJ_TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / "configured.env"
    configured.write_text("ADJ_WEB_SEARCH_PROVIDER=serper\n"
                          "ADJ_WEB_SEARCH_API_KEY=legacy-key\n"
                          "ADJ_SERPER_API_KEY=provider-key\n")

    search = web_search_from_environment(tmp_path, configured)

    assert isinstance(search, SerperSearch)
    assert search.secrets == ("provider-key",)


def test_tavily_native_sources_record_tavily_provenance(tmp_path):
    from adj_partner.native.web_search import TavilySearch

    body = {"results": [{"title": "Paper", "url": "https://example.org/paper",
                          "content": "Snippet", "score": 0.5}]}
    search = TavilySearch("fixture-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    _, research, _, executor, sid, rid = scope(tmp_path, search)

    result = executor.execute(ToolCall("tavily", "web_search", {"query": "fixture"}), sid, rid)

    assert result.status == "ok", result.to_dict()
    assert result.data["provider"] == "tavily"
    source = research.get_source(sid, rid, result.data["results"][0]["source_id"])
    assert source["provenance"]["provider"] == "Tavily"
