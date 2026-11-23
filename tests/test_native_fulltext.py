"""Lawful bounded full-text acquisition and source-chain tests."""

from dataclasses import dataclass
import json
from pathlib import Path

import httpx
import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.tools.executor import ToolFailure


JATS_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12/fullTextXML"
PDF_URL = "https://archive.example/article.pdf"
JATS_BYTES = b"<?xml version='1.0'?><article><body><p>Fixture</p></body></article>"


@pytest.mark.parametrize("url", [
    "http://example.org/a.pdf",
    "https://127.0.0.1/a.pdf",
    "https://[::1]/a.pdf",
    "https://user:pass@example.org/a.pdf",
    "https://169.254.169.254/latest/meta-data",
])
def test_public_target_rejects_unsafe_urls(url):
    from adj_partner.native.fulltext import public_target

    with pytest.raises(ToolFailure, match="public HTTPS"):
        public_target(url, resolver=lambda host: ["127.0.0.1"])


def test_fetch_accepts_jats_and_preserves_exact_bytes():
    from adj_partner.native.fulltext import FullTextFetcher

    calls = []
    transport = httpx.MockTransport(lambda request: (calls.append(request),
        httpx.Response(200, headers={"content-type": "application/xml"}, content=JATS_BYTES))[1])
    fetcher = FullTextFetcher(transport=transport, resolver=lambda host: ["93.184.216.34"])

    response = fetcher.fetch({"candidates": [{"acquisition": [
        {"kind": "jats", "url": JATS_URL, "provider": "europe_pmc", "is_oa": True}
    ]}]})

    assert len(calls) == 1
    assert response.body == JATS_BYTES
    assert response.media_type == "application/xml"
    assert response.acquisition["provider"] == "europe_pmc"


def test_fetch_revalidates_redirect_and_rejects_private_target():
    from adj_partner.native.fulltext import FullTextFetcher

    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})

    fetcher = FullTextFetcher(transport=httpx.MockTransport(handler),
        resolver=lambda host: ["93.184.216.34"] if host == "archive.example" else ["127.0.0.1"])

    with pytest.raises(ToolFailure) as error:
        fetcher.fetch({"candidates": [{"acquisition": [
            {"kind": "pdf", "url": PDF_URL, "provider": "unpaywall", "is_oa": True}
        ]}]})

    assert error.value.code == "fulltext_target"
    assert calls == [PDF_URL]


def test_production_curl_pins_validated_ipv6_with_curl_resolve_syntax(monkeypatch):
    from adj_partner.native.fulltext import FullTextFetcher

    captured = {}

    class Completed:
        returncode, stdout, stderr = 0, b"200", b""

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        body = Path(arguments[arguments.index("--output") + 1])
        headers = Path(arguments[arguments.index("--dump-header") + 1])
        body.write_bytes(JATS_BYTES)
        headers.write_text("HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\n\r\n", encoding="iso-8859-1")
        return Completed()

    monkeypatch.setattr("adj_partner.native.fulltext.subprocess.run", fake_run)
    fetcher = FullTextFetcher(resolver=lambda host: ["2606:4700:4700::1111"])

    response = fetcher.fetch({"candidates": [{"acquisition": [
        {"kind": "jats", "url": JATS_URL, "provider": "europe_pmc", "is_oa": True}
    ]}]})

    pinned = captured["arguments"][captured["arguments"].index("--resolve") + 1]
    assert pinned.endswith(":[2606:4700:4700::1111]")
    assert response.body == JATS_BYTES


@dataclass(frozen=True)
class _Fetched:
    body: bytes
    url: str
    media_type: str
    received_at: str
    acquisition: dict


class _Fetcher:
    def fetch(self, resolution, *, check_cancelled):
        check_cancelled()
        assert resolution["candidates"][0]["pmcid"] == "PMC12"
        return _Fetched(JATS_BYTES, JATS_URL, "application/xml", "2026-09-01T00:00:00+00:00",
            {"kind": "jats", "provider": "europe_pmc", "url": JATS_URL, "is_oa": True})


def _scope(tmp_path):
    from adj_partner.native.capabilities import NativeCapabilities

    store = Store(tmp_path / "state")
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research, fulltext=_Fetcher())
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allow_network=True,
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Fixture")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    resolution = {"parent_source_id": "source_parent", "candidates": [{
        "title": "Fixture", "doi": "10.1000/a", "pmid": "12", "pmcid": "PMC12", "acquisition": [
            {"kind": "jats", "url": JATS_URL, "provider": "europe_pmc", "is_oa": True}]}]}
    artifact = store.put_artifact(json.dumps(resolution), "resolution.json", "application/json", {},
                                  session_id=session, run_id=run)
    source = research.register_source(session, run, artifact_id=artifact["id"], kind="literature",
        title="Fixture", provenance={"provider": "scholarly_resolution", "text_availability": "title_only"},
        idempotency_key="resolution")
    return store, research, ToolExecutor(store, catalog), session, run, source


def test_fulltext_fetch_registers_exact_full_text_source(tmp_path):
    store, research, executor, session, run, resolved = _scope(tmp_path)

    result = executor.execute(ToolCall("fetch", "fulltext_fetch", {"source_id": resolved["id"]}), session, run)

    assert result.status == "ok", result.to_dict()
    source = research.get_source(session, run, result.data["source_id"])
    assert source["media_type"] == "application/xml"
    assert store.read_artifact(source["artifact_id"]) == JATS_BYTES
    assert source["provenance"]["text_availability"] == "full_text"
    assert source["provenance"]["parent_source_id"] == resolved["id"]
    assert source["provenance"]["doi"] == "10.1000/a"
    assert source["provenance"]["pmid"] == "12" and source["provenance"]["pmcid"] == "PMC12"
