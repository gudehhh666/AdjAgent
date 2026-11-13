"""Narrative Markdown report rendering with host-owned citation labels."""

from copy import deepcopy

from test_harness_research_records import setup, source


def _report_fixture(tmp_path):
    store, research, session, run, _ = setup(tmp_path)
    provenance = {"provider": "fixture", "url": "https://example.org/paper",
        "doi": "10.1000/a", "pmid": "12", "text_availability": "full_text",
        "limitations": ["Synthetic fixture only."]}
    material, _ = source(store, research, session, run, content="Observed response.",
        provenance=provenance)
    evidence = research.select_evidence(session, run, material["id"], {"passage": "Observed response."})
    claim = research.observe_evidence(session, run, evidence["id"])
    return store, research, session, run, claim


def test_markdown_report_renders_narrative_with_host_generated_citations(tmp_path):
    store, research, session, run, claim = _report_fixture(tmp_path)

    report = research.create_report(session, run, title="Fixture report", sections=[{
        "heading": "Evidence synthesis", "claim_ids": [claim["id"]],
        "paragraphs": [{"text": "The formulation improved the measured response.",
                         "claim_ids": [claim["id"]]}],
    }], limitations=["No causal conclusion."])
    rendered = store.read_artifact(report["artifact_id"]).decode()
    artifact = next(item for item in store.artifacts(session) if item["id"] == report["artifact_id"])

    assert rendered.startswith("# Research report")
    assert "The formulation improved the measured response. [S1]" in rendered
    assert "[S1](https://example.org/paper)" in rendered
    assert "DOI: [10.1000/a](https://doi.org/10.1000/a)" in rendered
    assert "PMID: [12](https://pubmed.ncbi.nlm.nih.gov/12/)" in rendered
    assert "Selected evidence [S1]:" in rendered
    assert "> Observed response." in rendered
    assert r"Text availability: full\_text; limitations: Synthetic fixture only." in rendered
    assert "session_id" not in rendered and "run_id" not in rendered
    assert report["renderer_version"] == 2
    from adj_partner.research.rendering import render_report
    legacy = deepcopy(research.get_report(session, run, report["id"]))
    legacy.pop("renderer_version")
    legacy_rendered = render_report(legacy).decode()
    assert "Selected evidence: " in legacy_rendered and r"session\_id" in legacy_rendered
    assert artifact["media_type"] == "text/markdown"


def test_authored_markdown_cannot_forge_host_labels_links_or_html(tmp_path):
    store, research, session, run, claim = _report_fixture(tmp_path)
    hostile = "# HOST VERIFIED\n[S1](https://evil.invalid)<script>alert(1)</script>\u202e"

    report = research.create_report(session, run, title=hostile, sections=[{
        "heading": hostile, "claim_ids": [claim["id"]],
        "paragraphs": [{"text": hostile, "claim_ids": [claim["id"]]}],
    }], limitations=[hostile])
    rendered = store.read_artifact(report["artifact_id"]).decode()

    assert "\n# HOST VERIFIED" not in rendered
    assert "https://evil.invalid" not in rendered
    assert "<script>" not in rendered
    assert "\\u202e" in rendered and "\u202e" not in rendered
    assert rendered.count("[S1](https://example.org/paper)") == 1


def test_old_section_shape_remains_valid_and_is_markdown(tmp_path):
    store, research, session, run, claim = _report_fixture(tmp_path)

    report = research.create_report(session, run, title="Legacy caller", sections=[{
        "heading": "Findings", "claim_ids": [claim["id"]]
    }], limitations=["Fixture only."])

    assert store.read_artifact(report["artifact_id"]).startswith(b"# Research report\n")


def test_paragraph_citation_includes_page_locator_when_evidence_has_one(tmp_path):
    store, research, session, run, _ = setup(tmp_path)
    material, _ = source(store, research, session, run,
        content=b'{"document_type":"pdf","pages":[{"text":"First"},{"text":"Page finding"}]}',
        provenance={"url": "https://example.org/paper.pdf", "doi": "10.1000/page",
                    "text_availability": "derived_full_text", "limitations": []})
    evidence = research.select_evidence(session, run, material["id"], {"pointer": "/pages/1/text"})
    claim = research.observe_evidence(session, run, evidence["id"])

    report = research.create_report(session, run, title="Page report", sections=[{
        "heading": "Finding", "claim_ids": [claim["id"]],
        "paragraphs": [{"text": "The paper reports a page-specific finding.", "claim_ids": [claim["id"]]}],
    }], limitations=["Fixture only."])

    assert "The paper reports a page-specific finding. [S1, p. 2]" in store.read_artifact(report["artifact_id"]).decode()
