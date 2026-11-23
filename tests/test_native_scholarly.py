"""Normalized scholarly discovery and identity-resolution behavior."""

from dataclasses import dataclass
import json
import os

import httpx

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.literature import EuropePMC
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.tools.catalog import canonical_json

from adj_partner.tools.executor import ToolFailure


@dataclass(frozen=True)
class _Envelope:
    records: list[dict]
    request_url: str = "https://fixture.invalid/search"
    received_at: str = "2026-09-01T00:00:00+00:00"
    raw: bytes = b"{}"
    hit_count: int = 1


class _Provider:
    def __init__(self, records=None, failure=None):
        self.records = records or []
        self.failure = failure

    def search(self, query, *, page_size, check_cancelled):
        check_cancelled()
        if self.failure:
            raise ToolFailure("fixture_failure", self.failure)
        return _Envelope(self.records, hit_count=len(self.records))


def _record(provider, title, *, doi=None, pmid=None, abstract=""):
    result = {
        "id": pmid or title.lower().replace(" ", "-"),
        "source": provider,
        "provider": provider,
        "title": title,
        "abstract": abstract,
        "authors": "Fixture Author",
        "year": "2025",
        "url": f"https://example.org/{provider}",
        "text_availability": "abstract" if abstract else "title_only",
    }
    if doi:
        result["doi"] = doi
    if pmid:
        result["pmid"] = pmid
    return result


def test_search_merges_and_deduplicates_by_doi_without_promoting_text_scope():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([_record("europe_pmc", "A study", doi="https://doi.org/10.1000/A", abstract="Full abstract")]),
        pubmed=_Provider([_record("pubmed", "A study", doi="10.1000/a", pmid="12")]),
        crossref=_Provider([_record("crossref", "A study", doi="10.1000/a")]),
    )

    response = gateway.search("fixture", page_size=5)

    assert len(response.records) == 1
    assert response.records[0]["doi"] == "10.1000/a"
    assert response.records[0]["pmid"] == "12"
    assert response.records[0]["abstract"] == "Full abstract"
    assert response.records[0]["text_availability"] == "abstract"
    assert response.records[0]["providers"] == ["europe_pmc", "pubmed", "crossref"]


def test_search_keeps_successes_when_one_provider_fails():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([_record("europe_pmc", "Usable")]),
        pubmed=_Provider(failure="HTTP 503"),
        crossref=_Provider([_record("crossref", "Second")]),
    )

    response = gateway.search("fixture", page_size=5)

    assert {item["title"] for item in response.records} == {"Usable", "Second"}
    assert response.diagnostics == [{"provider": "pubmed", "code": "fixture_failure"}]


def test_gateway_never_persists_provider_credentials_or_contact_query_parameters():
    from adj_partner.native.scholarly import ScholarlyGateway

    secret = "fixture-ncbi-private-key"
    provider = _Provider([_record("pubmed", "Usable")])
    provider.search = lambda query, page_size, check_cancelled: _Envelope(
        provider.records,
        request_url=("https://eutils.ncbi.nlm.nih.gov/search?term=fixture"
                     f"&api_key={secret}&email=private%40example.org"),
        hit_count=1,
    )
    gateway = ScholarlyGateway(europe_pmc=_Provider([]), pubmed=provider,
                               crossref=_Provider([]), secrets=(secret,))

    response = gateway.search("fixture", page_size=5)

    assert secret not in response.raw.decode() + " ".join(response.request_urls)
    assert "api_key=" not in response.raw.decode() + " ".join(response.request_urls)
    assert "email=" not in response.raw.decode() + " ".join(response.request_urls)


def test_search_ranks_relevant_cross_provider_results_before_truncation():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([
            _record("europe_pmc", "Unrelated liver assay"),
            _record("europe_pmc", "Generic antibody methods"),
        ]),
        pubmed=_Provider([_record("pubmed", "HBsAg alum CpG hepatitis B vaccine",
                                  abstract="IgG2a and IFN gamma responses")]),
        crossref=_Provider([_record("crossref", "Alum vaccine formulation")]),
    )

    response = gateway.search("hepatitis B HBsAg alum CpG IgG2a IFN gamma", page_size=2)

    assert len(response.records) == 2
    assert response.records[0]["title"] == "HBsAg alum CpG hepatitis B vaccine"
    assert {item["providers"][0] for item in response.records} == {"pubmed", "crossref"}


def test_pubmed_best_match_outranks_one_extra_abstract_term_but_not_a_title_term():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([
            _record("europe_pmc", "Alum adjuvant review", abstract="NLRP3 inflammasome"),
            _record("europe_pmc", "Alum NLRP3 adjuvant"),
        ]),
        pubmed=_Provider([_record("pubmed", "Alum adjuvant mechanism", pmid="7")]),
        crossref=_Provider([]),
    )

    titles = [r["title"] for r in gateway.search("alum NLRP3 adjuvant", page_size=3).records]
    assert titles == ["Alum NLRP3 adjuvant", "Alum adjuvant mechanism", "Alum adjuvant review"]


def test_search_uses_conservative_title_fallback_when_identifiers_are_absent():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([_record("europe_pmc", "  Same: Study!  ")]),
        pubmed=_Provider([_record("pubmed", "same study")]),
        crossref=_Provider([]),
    )

    response = gateway.search("fixture", page_size=5)

    assert len(response.records) == 1
    assert response.records[0]["providers"] == ["europe_pmc", "pubmed"]


def test_pubmed_search_batches_esearch_and_efetch_into_normalized_records():
    from adj_partner.native.scholarly import PubMedClient

    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"count": "1", "idlist": ["12"]}})
        assert request.url.path.endswith("/efetch.fcgi")
        return httpx.Response(200, content=b"""<?xml version='1.0'?>
        <!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN"
          "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">
        <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12</PMID><Article>
        <ArticleTitle>A PubMed title</ArticleTitle><Abstract><AbstractText Label='RESULTS'>Observed effect.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author></AuthorList>
        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        </Article></MedlineCitation><PubmedData><ArticleIdList>
        <ArticleId IdType='doi'>10.1000/PUB</ArticleId><ArticleId IdType='pmc'>PMC12</ArticleId>
        </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>""")

    response = PubMedClient(transport=httpx.MockTransport(handler)).search("fixture", page_size=3)

    assert len(calls) == 2
    assert calls[0].url.params["sort"] == "relevance"
    assert response.hit_count == 1
    assert response.records == [{
        "id": "12", "source": "MED", "provider": "pubmed", "pmid": "12",
        "pmcid": "PMC12", "doi": "10.1000/pub", "title": "A PubMed title",
        "abstract": "RESULTS: Observed effect.", "authors": "Jane Doe", "year": "2024",
        "text_availability": "abstract", "url": "https://pubmed.ncbi.nlm.nih.gov/12/",
    }]


def test_pubmed_xml_rejects_internal_entity_declarations():
    from adj_partner.native.scholarly import _xml_root

    with __import__("pytest").raises(ToolFailure, match="Unsafe XML"):
        _xml_root(b'<!DOCTYPE x [<!ENTITY leak SYSTEM "file:///etc/passwd">]><x>&leak;</x>',
                  "pubmed")


def test_crossref_search_returns_metadata_without_calling_it_full_text():
    from adj_partner.native.scholarly import CrossrefClient

    body = {"message": {"total-results": 1, "items": [{
        "DOI": "10.1000/CROSS", "title": ["Crossref title"],
        "abstract": "<jats:p>Metadata abstract.</jats:p>",
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "published": {"date-parts": [[2023, 1, 1]]},
        "URL": "https://doi.org/10.1000/CROSS",
    }]}}
    client = CrossrefClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    response = client.search("fixture", page_size=2)

    assert response.records[0]["doi"] == "10.1000/cross"
    assert response.records[0]["abstract"] == "Metadata abstract."
    assert response.records[0]["text_availability"] == "abstract"


class _Resolver:
    def __init__(self, result):
        self.result = result

    def resolve(self, record, *, check_cancelled):
        check_cancelled()
        return self.result


def test_resolve_combines_identifiers_and_ranks_jats_before_oa_pdf():
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([]), pubmed=_Provider([]), crossref=_Provider([]),
        id_converter=_Resolver({"pmid": "12", "pmcid": "PMC12", "doi": "10.1000/a"}),
        unpaywall=_Resolver({"doi": "10.1000/a", "is_oa": True, "license": "cc-by",
                             "best_oa_location": {"url_for_pdf": "https://archive.example/a.pdf", "host_type": "repository"}}),
    )

    candidates = gateway.resolve({"title": "A", "doi": "10.1000/A"})

    assert candidates[0]["pmcid"] == "PMC12"
    assert candidates[0]["acquisition"][0]["kind"] == "jats"
    assert candidates[0]["acquisition"][1] == {
        "kind": "pdf", "url": "https://archive.example/a.pdf", "provider": "unpaywall",
        "is_oa": True, "license": "cc-by", "host_type": "repository",
    }


def _native_scope(tmp_path, literature):
    from adj_partner.native.capabilities import NativeCapabilities

    store = Store(tmp_path / "state")
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research, literature=literature)
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allow_network=True,
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Fixture")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return store, research, ToolExecutor(store, catalog), session, run


def test_article_resolve_requires_current_run_literature_source_and_records_parent(tmp_path):
    from adj_partner.native.scholarly import ScholarlyGateway

    gateway = ScholarlyGateway(
        europe_pmc=_Provider([_record("europe_pmc", "A study", doi="10.1000/a", pmid="12")]),
        pubmed=_Provider([]), crossref=_Provider([]),
        id_converter=_Resolver({"pmid": "12", "pmcid": "PMC12", "doi": "10.1000/a"}),
    )
    store, research, executor, session, run = _native_scope(tmp_path, gateway)
    found = executor.execute(ToolCall("find", "literature_search", {"query": "fixture"}), session, run)
    assert found.status == "ok", found.to_dict()
    discovered_source = research.get_source(session, run, found.data["results"][0]["source_id"])
    assert found.data["provider"] == "scholarly_multi"
    assert discovered_source["provenance"]["doi"] == "10.1000/a"
    assert discovered_source["provenance"]["pmid"] == "12"
    assert discovered_source["provenance"]["providers"] == ["europe_pmc"]

    resolved = executor.execute(ToolCall("resolve", "article_resolve",
        {"source_id": found.data["results"][0]["source_id"]}), session, run)

    assert resolved.status == "ok", resolved.to_dict()
    source = research.get_source(session, run, resolved.data["source_id"])
    assert source["provenance"]["parent_source_id"] == found.data["results"][0]["source_id"]
    assert resolved.data["candidates"][0]["pmcid"] == "PMC12"
    assert json.loads(store.read_artifact(source["artifact_id"])) == resolved.data["resolution"]


def test_article_resolve_rejects_missing_or_non_literature_sources(tmp_path):
    gateway = __import__("adj_partner.native.scholarly", fromlist=["ScholarlyGateway"]).ScholarlyGateway(
        europe_pmc=_Provider([]), pubmed=_Provider([]), crossref=_Provider([]))
    store, research, executor, session, run = _native_scope(tmp_path, gateway)
    artifact = store.put_artifact("fixture", "web.txt", "text/plain", {}, session_id=session, run_id=run)
    web = research.register_source(session, run, artifact_id=artifact["id"], kind="web",
        title="Web", provenance={"text_availability": "search_snippet"}, idempotency_key="web")

    missing = executor.execute(ToolCall("missing", "article_resolve", {"source_id": "source_missing"}), session, run)
    wrong_kind = executor.execute(ToolCall("wrong", "article_resolve", {"source_id": web["id"]}), session, run)

    assert missing.status == "error"
    assert wrong_kind.status == "denied" and wrong_kind.error["code"] == "permission_denied"


def test_general_and_adjuvant_profiles_enable_the_resolution_chain():
    from adj_partner.profiles import get_profile

    for profile in ("general", "adjuvant"):
        assert {"literature_search", "article_resolve"} <= set(get_profile(profile)["tools"])


def test_default_gateway_has_all_free_metadata_providers():
    from adj_partner.native.scholarly import CrossrefClient, PubMedClient, ScholarlyGateway

    gateway = ScholarlyGateway()

    assert isinstance(gateway.europe_pmc, EuropePMC)
    assert isinstance(gateway.pubmed, PubMedClient)
    assert isinstance(gateway.crossref, CrossrefClient)


def test_pmc_id_converter_normalizes_doi_pmid_and_pmcid():
    from adj_partner.native.scholarly import PMCIdConverter

    requests = []
    body = {"records": [{"doi": "10.1000/A", "pmid": 12, "pmcid": "PMC12"}]}
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body)
    converter = PMCIdConverter(transport=httpx.MockTransport(handler))

    result = converter.resolve({"title": "A", "doi": "10.1000/a"})

    assert result == {"doi": "10.1000/a", "pmid": "12", "pmcid": "PMC12"}
    assert requests[0].url.host == "pmc.ncbi.nlm.nih.gov"
    assert requests[0].url.path == "/tools/idconv/api/v1/articles/"


def test_unpaywall_retains_only_valid_oa_location_metadata():
    from adj_partner.native.scholarly import UnpaywallClient

    body = {"doi": "10.1000/A", "is_oa": True, "best_oa_location": {
        "url_for_pdf": "https://archive.example/a.pdf", "host_type": "repository", "license": "cc-by"}}
    client = UnpaywallClient("researcher@example.org",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    result = client.resolve({"title": "A", "doi": "10.1000/a"})

    assert result["doi"] == "10.1000/a" and result["license"] == "cc-by"
    assert result["best_oa_location"]["url_for_pdf"] == "https://archive.example/a.pdf"


def test_gateway_environment_configuration_is_explicit_and_registers_api_key(tmp_path, monkeypatch):
    from adj_partner.native.scholarly import PMCIdConverter, ScholarlyGateway, UnpaywallClient

    for name in ("ADJ_NCBI_API_KEY", "ADJ_NCBI_EMAIL", "ADJ_UNPAYWALL_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / "providers.env"
    configured.write_text("ADJ_NCBI_API_KEY=fixture-ncbi-key\nADJ_NCBI_EMAIL=ncbi@example.org\n"
                          "ADJ_UNPAYWALL_EMAIL=oa@example.org\n")

    gateway = ScholarlyGateway.from_environment(tmp_path, configured)

    assert isinstance(gateway.id_converter, PMCIdConverter)
    assert isinstance(gateway.unpaywall, UnpaywallClient)
    assert gateway.pubmed.api_key == "fixture-ncbi-key"
    assert gateway.secrets == ("fixture-ncbi-key",)


def test_runtime_guidance_explains_open_research_text_scope_chain():
    from adj_partner.runtime.context import BASE_INSTRUCTIONS

    for name in ("literature_search", "article_resolve", "fulltext_fetch", "document_read"):
        assert name in BASE_INSTRUCTIONS
    assert "search snippet" in BASE_INSTRUCTIONS.lower()
    assert "PDF" in BASE_INSTRUCTIONS and "OCR" in BASE_INSTRUCTIONS
    assert "returned derived source_id" in BASE_INSTRUCTIONS
    assert "matches[].selector" in BASE_INSTRUCTIONS


def test_model_summary_remains_small_for_parallel_literature_batches(tmp_path):
    from adj_partner.native.scholarly import ScholarlyGateway

    records = [_record("europe_pmc", f"Study {index}", doi=f"10.1000/{index}",
                       abstract=("Long abstract passage. " * 80)) for index in range(10)]
    gateway = ScholarlyGateway(europe_pmc=_Provider(records), pubmed=_Provider([]), crossref=_Provider([]))
    _, _, executor, session, run = _native_scope(tmp_path, gateway)

    result = executor.execute(ToolCall("search", "literature_search",
        {"query": "fixture", "page_size": 10}), session, run)

    assert result.status == "ok", result.to_dict()
    assert len(canonical_json(result.data["summary"]).encode()) <= 2000
    assert result.data["summary"]["results_shown"] == 5
    assert all({"source_id", "title", "text_availability"} <= set(item)
               for item in result.data["summary"]["results"])
    assert all("abstract_preview" in item for item in result.data["summary"]["results"][:2])
    assert all("abstract_preview" not in item for item in result.data["summary"]["results"][2:])


def test_matched_abstract_and_paper_id_reach_the_model_after_search(tmp_path):
    from adj_partner.native.scholarly import ScholarlyGateway
    from adj_partner.runtime.context import ContextView

    abstract = "Background on the formulation. " * 9 + "The groups differed at the final endpoint."
    records = [_record("europe_pmc", "Formulation in the trial", doi="10.1000/trial", abstract=abstract)]
    records += [_record("europe_pmc", f"Another long candidate title {i} " + "x" * 180) for i in range(9)]
    gateway = ScholarlyGateway(europe_pmc=_Provider(records), pubmed=_Provider([]), crossref=_Provider([]))
    store, _, executor, session, run = _native_scope(tmp_path, gateway)
    result = executor.execute(ToolCall("find", "literature_search", {"query": "formulation trial", "page_size": 10}), session, run)
    message = ContextView(store).tool_message(session, run, "find", result.to_dict())
    packet = json.loads(message["content"])
    first = packet["data"]["summary"]["results"][0]
    assert first["abstract_preview"] == abstract and first["abstract_truncated"] is False
    assert first["doi"] == "10.1000/trial" and first["source_id"] == result.data["results"][0]["source_id"]
    assert len(message["content"].encode()) <= ContextView.PREVIEW_MAX_BYTES
    assert len(result.data["results"]) == 10


def test_resolution_summary_distinguishes_oa_routes_from_no_location(tmp_path):
    from adj_partner.native.scholarly import ScholarlyGateway

    for label, identity, count in [("open", {"pmcid": "PMC12"}, 3), ("unavailable", {}, 0)]:
        gateway = ScholarlyGateway(europe_pmc=_Provider([_record("europe_pmc", "A study", doi="10.1000/trial")]),
            pubmed=_Provider([]), crossref=_Provider([]), id_converter=_Resolver(identity))
        _, _, executor, session, run = _native_scope(tmp_path / label, gateway)
        found = executor.execute(ToolCall("find", "literature_search", {"query": "study"}), session, run)
        res = executor.execute(ToolCall("resolve", "article_resolve",
            {"source_id": found.data["results"][0]["source_id"]}), session, run)
        summary = res.data["summary"]
        assert summary["source_id"] == res.data["source_id"]
        assert summary["acquisition_count"] == count
        assert summary["next_tool"] == ("fulltext_fetch" if count else None)
