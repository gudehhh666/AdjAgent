"""The anonymous provider uses the same host permissions and source artifacts."""

import json
from subprocess import CompletedProcess
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ToolCall
from adj_partner.models import ScriptedPort
from adj_partner.native import web_search
from adj_partner.native.web_search import web_search_from_environment
from adj_partner.tools import ToolFailure
from adj_partner.web_access import WebPageFetcher


def mock_response(monkeypatch, body, status=200, exit_code=0):
    calls = []
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return CompletedProcess(args, exit_code, raw + f"\n{status}".encode(), b"")

    monkeypatch.setattr("adj_partner.native.web_search.subprocess.run", run)
    return calls, raw


@pytest.fixture
def clock(monkeypatch):
    # Exercise the host-wide limiter without adding real sleeps to the tests.
    now = 10.0

    def advance(seconds):
        nonlocal now
        now += seconds

    clock = SimpleNamespace(monotonic=lambda: now, sleep=advance)
    monkeypatch.setattr(web_search, "time", clock)
    monkeypatch.setattr(web_search, "_MWMBL_LAST_REQUEST", float("-inf"))
    return clock


@pytest.fixture
def anonymous(monkeypatch, tmp_path, clock):
    for name in ("ADJ_WEB_SEARCH_API_KEY", "ADJ_SERPER_API_KEY", "ADJ_TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ADJ_WEB_SEARCH_PROVIDER", "mwmbl")
    return web_search_from_environment(tmp_path)


def test_mwmbl_configured_native_call_is_anonymous_and_archives_exact_response(tmp_path, monkeypatch):
    monkeypatch.delenv("ADJ_WEB_SEARCH_PROVIDER", raising=False)
    env_file = tmp_path / "search.env"
    env_file.write_text("ADJ_WEB_SEARCH_PROVIDER=mwmbl\nADJ_WEB_SEARCH_API_KEY=fixture-generic-key\n")
    monkeypatch.setenv("ADJ_SERPER_API_KEY", "fixture-serper-key")
    monkeypatch.setenv("ADJ_TAVILY_API_KEY", "fixture-tavily-key")
    body = {"results": [
        {"title": "Result 1", "url": "https://example.org/one", "content": "long " * 1200},
        {"title": "Duplicate", "url": "https://example.org/one", "content": "duplicate"},
        {"title": [{"value": "Result "}, {"value": "2"}], "url": "https://example.org/two",
         "extract": [{"value": "Another "}, {"value": "excerpt."}]},
        {"title": "More", "url": "https://example.org/more", "extract": "Not projected."},
    ]}
    calls, raw = mock_response(monkeypatch, body)
    html = b"<main><p>Read from the linked page.</p></main>"
    page_fetcher = WebPageFetcher(
        resolver=lambda _: ["93.184.216.34"],
        transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/html"}, content=html)),
    )
    monkeypatch.setattr("adj_partner.web_access.WebPageFetcher", lambda: page_fetcher)
    service = AgentService(tmp_path / "state", project_root=tmp_path, env_file=env_file, model=ScriptedPort([]))
    try:
        assert service.native.web_search.secrets == ()
        query = "vaccine & adjuvant?api_key=x"
        args = {"query": query, "max_results": 2}
        sid = service.create_session()["id"]
        rid = service.submit(sid, "Offline search policy")["id"]
        denied = service.executor.execute(ToolCall("denied", "web_search", args), sid, rid)
        assert denied.status == "denied" and not calls
        sid = service.create_session(allow_network=True)["id"]
        rid = service.submit(sid, "Search for a public source")["id"]
        result = service.executor.execute(ToolCall("allowed", "web_search", args), sid, rid)
        assert result.status == "ok", result.to_dict()
        assert result.data["provider"] == "mwmbl" and result.data["returned"] == 2
        first, second = result.data["results"]
        assert first["text_availability"] == "search_snippet" and len(first["snippet"]) == 4000
        assert (second["title"], second["snippet"]) == ("Result 2", "Another excerpt.")
        source = service.research.get_source(sid, rid, first["source_id"])
        assert source["provenance"]["provider"] == "Mwmbl"
        assert service.store.read_artifact(source["provenance"]["response_artifact_id"]) == raw
        assert len(calls) == 1
        cmd, options = calls[0]
        target = urlsplit(cmd[-1])
        assert (target.scheme, target.netloc, target.path) == ("https", "api.mwmbl.org", "/api/v2/search/")
        assert parse_qs(target.query) == {"q": [query]} and "--no-location" in cmd
        assert cmd[1] == "--disable" and options["env"] == {}
        assert not any("fixture-" in str(value) for value in (cmd, options, result.to_dict()))
        assert set(service.catalog.get("web_search").spec.input_schema["properties"]) == {"query", "max_results"}
        for tool in ("web_page_read", "evidence_read"):
            enabled = service.executor.execute(ToolCall(f"enable-{tool}", "tools_enable", {"name": tool}), sid, rid)
            assert enabled.status == "ok", enabled.to_dict()
        page = service.executor.execute(ToolCall("page", "web_page_read", {"source_id": first["source_id"]}), sid, rid)
        assert page.status == "ok", page.to_dict()
        derived = service.research.get_source(sid, rid, page.data["source_id"])
        assert derived["provenance"]["parent_source_id"] == first["source_id"]
        assert page.data["text_availability"] == "derived_full_text"
        evidence = service.executor.execute(ToolCall("read", "evidence_read", {
            "source_id": derived["id"], "selector": page.data["paragraphs"][0]["selector"]}), sid, rid)
        assert evidence.status == "ok" and evidence.data["text"] == "Read from the linked page."
    finally:
        service.close()


@pytest.mark.parametrize(("body", "status", "exit_code", "code"), [
    ({"detail": "Rate limit exceeded: please slow down."}, 429, 0, "web_search_rate_limit"),
    ({"detail": "Monthly quota exceeded: your account allows 1,000 requests per month."}, 429, 0, "web_search_quota"),
    (b"Too Many Requests", 429, 0, "web_search_http"),
    ({"detail": "Temporary failure"}, 503, 0, "web_search_http"),
    (b"", 301, 0, "web_search_http"),
    ({"message": "unexpected response"}, 200, 0, "web_search_response"),
    ({"results": [{"title": "Bad URL", "url": "file:///etc/passwd"}]}, 200, 0, "web_search_response"),
    (b"", 0, 63, "web_search_size"),
    (b"", 0, 28, "web_search_timeout"),
])
def test_mwmbl_failures_remain_failures_without_retry_or_paid_fallback(anonymous, monkeypatch, body, status, exit_code, code):
    calls, _ = mock_response(monkeypatch, body, status, exit_code)
    with pytest.raises(ToolFailure) as failure:
        anonymous.search("fixture")
    assert failure.value.code == code and len(calls) == 1


def test_mwmbl_empty_results_and_title_only_are_distinct(anonymous, monkeypatch):
    mock_response(monkeypatch, {"results": []})
    assert anonymous.search("rare query").records == []
    mock_response(monkeypatch, {"results": [{"url": "https://example.org", "title": "Title", "extract": []}]})
    record, = anonymous.search("fixture").records
    assert record["text_availability"] == "title_only" and record["snippet"] == ""


def test_mwmbl_rate_is_shared_and_wait_is_cancellable(anonymous, clock, tmp_path, monkeypatch):
    calls, _ = mock_response(monkeypatch, {"results": []})
    start = clock.monotonic()
    anonymous.search("first")
    other = web_search_from_environment(tmp_path)

    def cancel_during_wait():
        if clock.monotonic() > start:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        other.search("cancelled", check_cancelled=cancel_during_wait)
    assert len(calls) == 1
    other.search("second")
    assert len(calls) == 2 and clock.monotonic() >= start + 1


def test_provider_keys_alone_do_not_enable_mwmbl_or_paid_search(tmp_path, monkeypatch):
    for name in ("ADJ_WEB_SEARCH_PROVIDER", "ADJ_WEB_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ADJ_SERPER_API_KEY", "fixture-serper-key")
    monkeypatch.setenv("ADJ_TAVILY_API_KEY", "fixture-tavily-key")
    assert web_search_from_environment(tmp_path) is None
