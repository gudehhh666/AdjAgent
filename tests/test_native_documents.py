"""Bounded JATS/PDF parsing and document-query behavior."""

import io
import json

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.tools.executor import ToolFailure


JATS_BYTES = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta><title-group><article-title>Fixture article</article-title></title-group>
    <abstract><p>Abstract passage.</p></abstract></article-meta></front>
  <body><sec><title>Results</title><p>Observed immune response in the treated group.</p>
    <fig id="f1"><label>Figure 1</label><caption><p>Figure fixture caption.</p></caption>
      <graphic xlink:href="figure1.png"/></fig>
    <table-wrap id="t1"><label>Table 1</label><caption><p>Table fixture caption.</p></caption>
      <table><tr><th>Group</th><th>Value</th></tr><tr><td>Treated</td><td>7</td></tr></table></table-wrap>
  </sec></body>
  <back><ref-list><ref id="r1"><element-citation><article-title>Reference title</article-title>
    <pub-id pub-id-type="doi">10.1000/ref</pub-id></element-citation></ref></ref-list></back>
</article>"""


def test_parse_jats_preserves_sections_tables_figures_and_references():
    from adj_partner.native.documents import parse_jats

    parsed = parse_jats(JATS_BYTES)

    assert parsed["document_type"] == "jats"
    assert parsed["title"] == "Fixture article"
    assert parsed["sections"][0]["title"] == "Results"
    assert parsed["sections"][0]["paragraphs"][0]["locator"] == "section:1/paragraph:1"
    assert parsed["figures"][0]["caption"] == "Figure fixture caption."
    assert parsed["figures"][0]["graphic_href"] == "figure1.png"
    assert parsed["tables"][0]["rows"] == [["Group", "Value"], ["Treated", "7"]]
    assert parsed["references"][0]["doi"] == "10.1000/ref"


def test_parse_jats_rejects_doctype_and_entities():
    from adj_partner.native.documents import parse_jats

    hostile = b'<!DOCTYPE article [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><article><body><p>&xxe;</p></body></article>'

    with pytest.raises(ToolFailure) as error:
        parse_jats(hostile)

    assert error.value.code == "document_xml_unsafe"


def test_parse_pdf_preserves_page_locators_and_discloses_empty_page():
    from pypdf import PdfWriter
    from adj_partner.native.documents import parse_pdf

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    target = io.BytesIO()
    writer.write(target)

    parsed = parse_pdf(target.getvalue())

    assert parsed["document_type"] == "pdf"
    assert parsed["pages"][0]["locator"] == "page:1"
    assert parsed["pages"][0]["text"] == ""
    assert parsed["pages"][0]["text_availability"] == "image_only_or_empty"


def test_query_document_returns_stable_matches_and_cursor():
    from adj_partner.native.documents import parse_jats, query_document

    parsed = parse_jats(JATS_BYTES)
    result = query_document(parsed, query="immune response", cursor=0, limit=1)

    assert result["matches"][0]["locator"] == "section:1/paragraph:1"
    assert result["matches"][0]["selector"] == {"pointer": "/sections/0/paragraphs/0/text"}
    assert "immune response" in result["matches"][0]["text"]
    assert result["next_cursor"] is None


def test_query_document_ranks_partial_term_matches_instead_of_requiring_every_term():
    from adj_partner.native.documents import query_document

    document = {"document_type": "jats", "abstract": [], "figures": [], "tables": [],
        "references": [], "sections": [
            {"title": "Results", "paragraphs": [{"locator": "section:1/paragraph:1",
                "text": "Alum increased IgG2a."}]},
            {"title": "Discussion", "paragraphs": [{"locator": "section:2/paragraph:1",
                "text": "Alum increased IFN-gamma responses."}]},
        ]}

    result = query_document(document, query="alum IFN gamma neutralizing", limit=10)

    assert [item["locator"] for item in result["matches"]] == [
        "section:2/paragraph:1", "section:1/paragraph:1"]
    assert result["matches"][0]["matched_terms"] == ["alum", "ifn", "gamma"]


def test_query_document_ignores_pdf_page_filter_for_jats_and_matches_section_terms():
    from adj_partner.native.documents import parse_jats, query_document

    result = query_document(parse_jats(JATS_BYTES), query="immune missing",
                            page=1, section="abstract results discussion", limit=10)

    assert result["matches"][0]["locator"] == "section:1/paragraph:1"
    assert result["ignored_filters"] == ["page"]


def test_exact_subsection_filters_out_other_sections_with_shared_words():
    from adj_partner.native.documents import query_document

    heading = "3.1. Targets of the formulation"
    doc = {"document_type": "jats", "sections": [
        {"title": title, "section_path": path, "paragraphs": [{"locator": f"section:{i}/paragraph:1",
          "text": "Reported target for this experiment."}]}
        for i, (title, path) in enumerate([
            ("2.1. Materials of the study", ["Methods", "2.1. Materials of the study"]),
            (heading, ["Results", heading]),
            ("Sensitivity analysis", ["Results", heading, "Sensitivity analysis"])], 1)]}
    hits = query_document(doc, query="target", section=heading)["matches"]
    assert [m["section"] for m in hits] == [heading, "Sensitivity analysis"]


class _Documents:
    def parse(self, content, media_type):
        assert content == JATS_BYTES and media_type == "application/xml"
        return {"document_type": "jats", "title": "Fixture", "abstract": [], "sections": [{
            "title": "Results", "locator": "section:1", "paragraphs": [{
                "locator": "section:1/paragraph:1", "text": "Observed response."}]}],
            "figures": [], "tables": [], "references": [], "limitations": []}

    def query(self, document, **arguments):
        assert arguments["query"] == "response"
        return {"matches": document["sections"][0]["paragraphs"], "next_cursor": None,
                "returned": 1, "total_matches": 1, "ignored_filters": ["fixture_filter"]}


def test_document_read_registers_derived_source_and_returns_passages(tmp_path):
    from adj_partner.native.capabilities import NativeCapabilities

    store = Store(tmp_path / "state")
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research, documents=_Documents())
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allowed_tools=catalog.names(),
        enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Fixture")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    artifact = store.put_artifact(JATS_BYTES, "article.xml", "application/xml", {},
                                  session_id=session, run_id=run)
    fulltext = research.register_source(session, run, artifact_id=artifact["id"], kind="literature",
        title="Fixture", provenance={"text_availability": "full_text", "doi": "10.1000/a",
                                     "pmid": "12", "pmcid": "PMC12", "url": "https://example.org/article"},
        idempotency_key="fulltext")
    executor = ToolExecutor(store, catalog)

    result = executor.execute(ToolCall("read", "document_read",
        {"source_id": fulltext["id"], "query": "response"}), session, run)

    assert result.status == "ok", result.to_dict()
    derived = research.get_source(session, run, result.data["source_id"])
    assert derived["provenance"]["text_availability"] == "derived_full_text"
    assert derived["provenance"]["parent_source_id"] == fulltext["id"]
    assert derived["provenance"]["doi"] == "10.1000/a" and derived["provenance"]["pmcid"] == "PMC12"
    assert json.loads(store.read_artifact(derived["artifact_id"]))["document_type"] == "jats"
    assert result.data["matches"][0]["locator"] == "section:1/paragraph:1"
    assert result.data["summary"]["matches"][0]["text"] == "Observed response."
    assert result.data["summary"]["ignored_filters"] == ["fixture_filter"]


def test_document_read_description_explains_exact_evidence_handoff(tmp_path):
    from adj_partner.native.capabilities import NativeCapabilities

    store = Store(tmp_path / "state")
    catalog = ToolCatalog()
    NativeCapabilities(store, research=ResearchStore(store), documents=_Documents()).register(catalog)

    description = catalog.get("document_read").spec.description
    assert "returned source_id" in description
    assert "matches[].selector" in description


def test_document_read_can_query_derived_and_abstract_without_promoting_scope(tmp_path):
    from adj_partner.native.capabilities import NativeCapabilities
    from adj_partner.native.documents import DocumentReader
    store=Store(tmp_path/'state');research=ResearchStore(store);catalog=ToolCatalog()
    NativeCapabilities(store,research=research,documents=DocumentReader()).register(catalog)
    sid=store.create_session(SessionConfig(allowed_tools=catalog.names(),enabled_tools=catalog.names()).to_dict())['id']
    task=store.create_task(sid,'Read materials')['id'];rid=store.create_run(sid,task,Budget().to_dict())['id'];executor=ToolExecutor(store,catalog)
    artifact=store.put_artifact(json.dumps({'title':'Abstract fixture','abstract':'Observed immune response in a limited abstract.'}),'abstract.json','application/json',{},session_id=sid,run_id=rid)
    src=research.register_source(sid,rid,artifact_id=artifact['id'],kind='literature',title='Abstract',provenance={'text_availability':'abstract'},idempotency_key='abstract')
    result=executor.execute(ToolCall('abstract-read','document_read',{'source_id':src['id'],'query':'immune response'}),sid,rid)
    assert result.status=='ok', result.to_dict()
    parsed=research.get_source(sid,rid,result.data['source_id'])
    assert parsed['provenance']['text_availability']=='abstract'
    selected=result.data['matches'][0]['selector'];e=research.select_evidence(sid,rid,parsed['id'],selected)
    assert 'limited abstract' in str(e)
    again=executor.execute(ToolCall('abstract-again','document_read',{'source_id':parsed['id'],'query':'immune response'}),sid,rid)
    assert again.status=='ok' and again.data['source_id']==parsed['id']
    raw=store.put_artifact(JATS_BYTES,'article.xml','application/xml',{},session_id=sid,run_id=rid)
    full=research.register_source(sid,rid,artifact_id=raw['id'],kind='literature',title='Full',provenance={'text_availability':'full_text'},idempotency_key='full')
    first=executor.execute(ToolCall('full-read','document_read',{'source_id':full['id'],'query':'immune response'}),sid,rid)
    assert first.status=='ok'
    second=executor.execute(ToolCall('derived-read','document_read',{'source_id':first.data['source_id'],'query':'immune response'}),sid,rid)
    assert second.status=='ok' and second.data['source_id']==first.data['source_id']


@pytest.mark.parametrize("different_task", [False, True])
def test_document_read_requires_explicit_handoff_for_other_runs(tmp_path, different_task):
    from adj_partner.native.capabilities import NativeCapabilities
    from adj_partner.native.documents import DocumentReader
    store=Store(tmp_path/'state');research=ResearchStore(store);catalog=ToolCatalog()
    NativeCapabilities(store,research=research,documents=DocumentReader()).register(catalog)
    sid=store.create_session(SessionConfig(allowed_tools=catalog.names(),enabled_tools=catalog.names()).to_dict())['id']
    tid=store.create_task(sid,'Original task')['id'];old=store.create_run(sid,tid,Budget().to_dict())['id']
    artifact=store.put_artifact(JATS_BYTES,'article.xml','application/xml',{},session_id=sid,run_id=old)
    source=research.register_source(sid,old,artifact_id=artifact['id'],kind='literature',title='Fixture',provenance={'text_availability':'full_text'},idempotency_key='original')
    target_tid=store.create_task(sid,'Other task')['id'] if different_task else tid
    current=store.create_run(sid,target_tid,Budget().to_dict())['id'];executor=ToolExecutor(store,catalog)
    result=executor.execute(ToolCall('foreign-read','document_read',{'source_id':source['id']}),sid,current)
    assert result.status != 'ok'
    if not different_task:
        handed=research.handoff_source(sid,current,source['id'])
        result=executor.execute(ToolCall('handed-read','document_read',{'source_id':handed['id']}),sid,current)
        assert result.status=='ok', result.to_dict()
