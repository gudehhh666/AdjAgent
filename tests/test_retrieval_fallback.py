import json

import httpx
import pytest

from adj_partner.native.documents import parse_document, query_document
from adj_partner.native.fulltext import FullTextFetcher
from adj_partner.tools.executor import ToolFailure


def bioc(pmcid="PMC12", section="METHODS"):
    return json.dumps([{"documents": [{"id": pmcid, "passages": [
        {"infons": {"section_type": section, "type": "paragraph"},
         "offset": 527, "text": "Observed dendritic cells."}]}]}]).encode()


def test_same_paper_bioc_fallback_and_exact_passage():
    seen = []
    def serve(request):
        seen.append(request.url.host)
        if request.url.host == "www.ebi.ac.uk":
            return httpx.Response(404)
        return httpx.Response(200, content=bioc(), headers={"Content-Type": "application/json"})
    fetcher = FullTextFetcher(transport=httpx.MockTransport(serve), resolver=lambda host: ["8.8.8.8"])
    first = {"pmcid": "PMC12", "acquisition": [
        {"provider": "europe_pmc", "kind": "jats", "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12/fullTextXML"},
        {"provider": "ncbi_bioc", "kind": "bioc", "url": "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC12/unicode"}]}
    second = {"pmcid": "PMC13", "acquisition": [{"kind":"pdf","is_oa":True,"url":"https://wrong.example/other.pdf"}]}
    result = fetcher.fetch({"candidates": [first, second]})
    assert seen == ["www.ebi.ac.uk", "www.ncbi.nlm.nih.gov"]
    assert [a["status"] for a in result.attempts] == ["failed", "selected"]
    passage = query_document(parse_document(result.body, result.media_type), query="dendritic")["matches"][0]
    assert passage["offset"] == 527
    assert passage["selector"]["pointer"] == "/sections/0/paragraphs/0/text"


@pytest.mark.parametrize("body", [bioc("PMC42"), bioc(section="ABSTRACT"), b"[Error] No result"])
def test_bioc_success_requires_matching_full_text(body):
    fetcher = FullTextFetcher(transport=httpx.MockTransport(lambda req: httpx.Response(
        200, content=body, headers={"Content-Type":"text/plain"})), resolver=lambda host: ["8.8.8.8"])
    with pytest.raises(ToolFailure) as exc:
        fetcher.fetch({"candidates":[{"pmcid":"PMC12","acquisition":[{"kind":"bioc","provider":"ncbi_bioc",
            "url":"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC12/unicode"}]}]})
    assert exc.value.code in {"fulltext_identity", "fulltext_media"}


def test_jats_other_pmcid_rejected():
    data = b'<article><front><article-meta><article-id pub-id-type="pmc">42</article-id></article-meta></front><body><p>Body</p></body></article>'
    fetcher = FullTextFetcher(transport=httpx.MockTransport(lambda req: httpx.Response(
        200, content=data, headers={"content-type":"text/xml"})), resolver=lambda host: ["8.8.8.8"])
    with pytest.raises(ToolFailure) as exc:
        fetcher.fetch({"candidates":[{"pmcid":"PMC12","acquisition":[{"kind":"jats","provider":"europe_pmc",
            "url":"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12/fullTextXML"}]}]})
    assert exc.value.code == "fulltext_identity"


def test_parent_section_filter_includes_nested_paragraphs():
    doc = parse_document(b"<article><body><sec><title>Results</title><sec><title>Cohort</title><p>A nested observation.</p></sec></sec></body></article>", "application/xml")
    matched = query_document(doc, section="results")["matches"]
    assert [item["text"] for item in matched] == ["A nested observation."]
    assert matched[0]["section"] == "Cohort"


def test_official_provider_cannot_be_redirected_by_oa_resolver_flag():
    fetcher = FullTextFetcher(resolver=lambda h: pytest.fail("No network expected"))
    with pytest.raises(ToolFailure) as exc:
        fetcher.fetch({"candidates":[{"pmcid":"PMC12","acquisition":[
            {"kind":"jats", "provider":"europe_pmc", "url":"https://example.org/different.xml", "is_oa":True}]}]})
    assert exc.value.code == "fulltext_unavailable"


def test_jats_retains_exact_pmc_version_for_later_supplement_lookup():
    from adj_partner.native.fulltext import _verify_jats_identity
    xml = b'<article><front><article-meta><article-id pub-id-type="pmcid-ver">PMC12.7</article-id></article-meta></front><body><p>Text</p></body></article>'
    assert _verify_jats_identity(xml, {"pmcid": "PMC12"}) == 7
    with pytest.raises(ToolFailure, match="version"):
        _verify_jats_identity(xml, {"pmcid": "PMC13"})


def test_abstract_only_xml_cannot_be_registered_as_full_text():
    from adj_partner.native.fulltext import _verify_jats_identity
    xml = b"<article><front><abstract><p>Abstract</p></abstract></front></article>"
    with pytest.raises(ToolFailure) as exc:
        _verify_jats_identity(xml, {})
    assert exc.value.code == "fulltext_media"


def test_unidentified_official_jats_does_not_satisfy_known_identity():
    from adj_partner.native.fulltext import _verify_jats_identity
    with pytest.raises(ToolFailure) as exc:
        _verify_jats_identity(b"<article><body><p>Anonymous body</p></body></article>",
                              {"pmcid": "PMC12"}, require_identifier=True)
    assert exc.value.code == "fulltext_media"


def test_expensive_page_is_parsed_once_and_scope_survives_repeated_queries(tmp_path, monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.contracts import Budget, ToolCall
    from adj_partner.models.base import ScriptedPort
    from adj_partner.native import pdf_enhanced

    svc = AgentService(tmp_path, model=ScriptedPort([]))
    try:
        sid = svc.create_session()["id"]
        tid = svc.store.create_task(sid, "Read supplementary material")["id"]
        rid = svc.store.create_run(sid, tid, Budget().to_dict())["id"]
        provenance = {"text_availability": "full_text", "document_scope":"supplement", "pmc_version":7}
        artifact = svc.store.put_artifact(b"%PDF-fixture", "sup.pdf", "application/pdf",
                                          provenance, session_id=sid, run_id=rid)
        raw = svc.research.register_source(sid, rid, artifact_id=artifact["id"], kind="literature",
                                           title="Fixture", provenance=provenance, idempotency_key="pdf")
        calls = []
        def worker(content, page, mode, check_cancelled):
            calls.append((page, mode))
            return {"document_type":"pdf", "mode":mode, "backend":"fixture", "pages":[
                {"locator":"page:2", "page":2, "text":"SAMPLE", "text_availability":"ocr_text"}]}
        monkeypatch.setattr(pdf_enhanced, "parse_pdf_page", worker)
        def execute(call_id, tool, args):
            result = svc.executor.execute(ToolCall(call_id, tool, args), sid, rid)
            assert result.status == "ok", result.error
            return result
        for name in ("document_read", "evidence_read"):
            execute("enable-" + name, "tools_enable", {"name":name})
        args = {"source_id":raw["id"], "page":2, "mode":"ocr"}
        parsed = execute("parse", "document_read", args)
        cached = execute("reuse", "document_read", args)
        assert calls == [(2, "ocr")] and not cached.artifacts
        assert cached.data["source_id"] == parsed.data["source_id"]
        assert cached.data["artifact_id"] == parsed.data["artifact_id"]
        evidence = execute("read", "evidence_read", {"source_id":cached.data["source_id"],
                           "selector":cached.data["matches"][0]["selector"]})
        assert evidence.data["text"] == "SAMPLE" and evidence.data["page"] == 2
        assert evidence.data["pmc_version"] == 7 and evidence.data["document_scope"] == "supplement"
    finally:
        svc.close()


def test_resolution_rejects_conflicting_conversion_and_preserves_unknown_license():
    from adj_partner.native.scholarly import ScholarlyGateway
    class Conflict:
        def resolve(self, record, *, check_cancelled):
            return {"doi":"10.1234/wrong", "pmcid":"PMC123", "pmid":"100"}
    article = ScholarlyGateway(id_converter=Conflict()).resolve({
        "title":"Original", "doi":"10.1234/correct", "pmcid":"PMC42"})[0]
    assert article["pmcid"] == "PMC42" and "pmid" not in article
    assert article["diagnostics"] == [{"provider":"pmc_id_converter", "code":"identifier_conflict"}]
    assert all(item.get("is_oa") is not True for item in article["acquisition"])
