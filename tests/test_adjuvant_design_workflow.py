"""Host-selected workflow and exact-byte review-layout integration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_harness_research_records import setup, source
from test_open_research_case import _FakeService


def test_runner_injects_host_workflow_and_snapshots_it_without_hidden_data(tmp_path):
    from adj_partner.open_research_case import run_case
    dataset = tmp_path / "input.json"
    dataset.write_text(json.dumps([{"id": "a", "question": "Design with Alum?",
                                    "answer": "HIDDEN_ANSWER"}]))
    result = run_case(dataset, "a", tmp_path / "out", workflow="adjuvant-design-v1",
                      model="gpt-5.6-luna", service_factory=_FakeService)
    instance = _FakeService.instances[-1]
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    workflow = manifest["workflow"]
    assert instance.session_kwargs["instructions"] == workflow["instructions"]
    assert workflow["instructions_sha256"] == hashlib.sha256(workflow["instructions"].encode()).hexdigest()
    assert instance.acceptance == workflow["acceptance"]
    assert instance.acceptance["report_layout"] == "review"
    assert len(instance.acceptance["required_sections"]) == 8
    assert instance.session_kwargs["allowed_roots"] == []
    assert "HIDDEN_ANSWER" not in json.dumps(manifest) + instance.goal + workflow["instructions"]
    assert "Do not assume" not in instance.goal  # no conflict with explicit chapter contract


def test_unknown_workflow_rejected_before_output_or_service(tmp_path):
    from adj_partner.open_research_case import run_case
    dataset = tmp_path / "input.json"
    dataset.write_text('[{"question":"Question"}]')
    count = len(_FakeService.instances)
    with pytest.raises(ValueError, match="workflow"):
        run_case(dataset, "1", tmp_path / "out", workflow="../../secret", service_factory=_FakeService)
    assert not (tmp_path / "out").exists()
    assert len(_FakeService.instances) == count


@pytest.mark.parametrize("layout", [True, {}, "unknown"])
def test_invalid_layout_is_rejected(layout):
    from adj_partner.research.acceptance import requirements
    with pytest.raises(ValueError, match="layout"):
        requirements({"report_layout": layout})


def test_review_layout_keeps_body_contiguous_deduplicates_and_preserves_binding(tmp_path):
    from adj_partner.research.rendering import render_report
    from adj_partner.research.acceptance import ReportNotAccepted
    store, research, sid, rid, _ = setup(tmp_path, acceptance={
        "require_evidence": True, "require_limitations": True, "report_layout": "review"})
    material, _ = source(store, research, sid, rid, content="Literal evidence excerpt.")
    evidence = research.select_evidence(sid, rid, material["id"], {"passage": "Literal evidence excerpt."})
    claims = [research.record_claim(sid, rid, text=t, kind="proposal", evidence_ids=[evidence["id"]])
              for t in ["Proposed interpretation.", "Alternative interpretation."]]
    sections = [{"heading": heading, "claim_ids": [c["id"] for c in claims],
                 "paragraphs": [{"text": body, "claim_ids": [claims[0]["id"]]}]}
                for heading, body in [("Decision", "Body A."), ("Validation", "Body B.")]]
    report = research.create_report(sid, rid, title="Review fixture", sections=sections,
                                    limitations=["Synthetic only."])
    content = store.read_artifact(report["artifact_id"])
    rendered = content.decode()
    assert report["renderer_version"] == 3
    assert rendered.index("Body A.") < rendered.index("Body B.") < rendered.index("## 论断与证据审计")
    assert rendered.count("Literal evidence excerpt.") == 1
    assert rendered.count("Proposed interpretation.") == 1
    assert "proposal" in rendered and "unassessed" in rendered
    assert claims[0]["id"] not in rendered
    assert render_report(report) == content
    assert research.validate_report(sid, rid, report["id"])["accepted"] is True
    research.revise_claim(sid, rid, claims[0]["id"], expected_revision=0,
                          text="Changed interpretation.", kind="proposal", evidence_ids=[evidence["id"]])
    assert research.validate_report(sid, rid, report["id"])["accepted"] is False
    with pytest.raises(ReportNotAccepted):
        research.accepted_report(sid, rid, report["id"])


def test_review_layout_escapes_authored_links_and_keeps_legacy_bytes(tmp_path):
    from adj_partner.research.rendering import render_report
    from test_research_markdown_reports import _report_fixture
    store, research, sid, rid, claim = _report_fixture(tmp_path)
    hostile = "<script>bad</script>\n[S1](https://evil.invalid)"
    report = research.create_report(sid, rid, title=hostile, sections=[{
        "heading": hostile, "claim_ids": [claim["id"]],
        "paragraphs": [{"text": hostile, "claim_ids": [claim["id"]]}]}], limitations=["Fixture."])
    old_bytes = store.read_artifact(report["artifact_id"])
    proposed = deepcopy(report)
    proposed["renderer_version"] = 3
    view = render_report(proposed).decode()
    assert "<script>" not in view and "https://evil.invalid" not in view
    assert "[S1](https://example.org/paper)" in view
    assert render_report(report) == old_bytes
    assert report["renderer_version"] == 2


def test_workflow_acceptance_is_independent_between_loads():
    from adj_partner.workflows import load_workflow
    first = load_workflow("adjuvant-design-v1")
    first["acceptance"]["required_sections"].clear()
    assert len(load_workflow("adjuvant-design-v1")["acceptance"]["required_sections"]) == 8
