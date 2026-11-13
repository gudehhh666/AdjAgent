"""Structured report safety, actual artifact handoff and fail-closed acceptance."""

import json

import pytest

from adj_partner.contracts import Budget
from adj_partner.storage import Store
from test_harness_research_records import api, scope, setup, source
from test_harness_research_assessments import evidence_claim, verdict


def report(r, sid, rid, claim_id, *, title="Fixture report", heading="Findings", limitations=None):
    assert hasattr(r, "create_report"), "structured report operation must exist"
    return r.create_report(sid, rid, title=title, sections=[{"heading": heading, "claim_ids": [claim_id]}],
                           limitations=["Synthetic fixture; efficacy and safety remain unknown."] if limitations is None else limitations)


def observed(tmp_path, acceptance=None, *, kind="computed", content=b'{"n":1,"score":-0.3}'):
    store, r, sid, rid, tid = setup(tmp_path, acceptance=acceptance)
    s, a = source(store, r, sid, rid, kind=kind, content=content,
                  provenance={"synthetic_fixture": True, "text_availability": "computed" if kind == "computed" else "userfile",
                              "limitations": ["n=1", "shared-control", "Not efficacy or safety evidence."]})
    e = r.select_evidence(sid, rid, s["id"], {"pointer": "/score"})
    c = r.observe_evidence(sid, rid, e["id"])
    return store, r, sid, rid, tid, s, a, e, c


def test_positive_exact_immutable_handoff_reopens_and_status_only_task_change_is_allowed(tmp_path):
    requirements = {"require_report": True, "require_evidence": True, "require_limitations": True,
                    "require_data_artifact": True, "required_sections": ["Findings"]}
    store, r, sid, rid, tid, _, _, _, c = observed(tmp_path, requirements)
    rp = report(r, sid, rid, c["id"])
    validation = r.validate_report(sid, rid, rp["id"])
    assert validation["accepted"] is True and validation["issues"] == []
    store.update_task(tid, {"status": "complete"}, expected_revision=0)
    reopened = api().ResearchStore(Store(store.root))
    result = reopened.accepted_report(sid, rid, rp["id"])
    assert result["content"] == store.read_artifact(rp["artifact_id"])
    assert result["artifact"]["sha256"] == rp["artifact_hash"]
    assert result["validation"]["id"] == validation["id"]
    assert result["report"]["bindings"]["task"]["observed_revision"] == 0
    assert len(reopened.list_reports(sid, rid)) == 1


def test_source_and_method_caveats_are_automatic_and_distinct_from_authored_limitations(tmp_path):
    store, r, sid, rid, _, _, _, _, c = observed(tmp_path)
    rp = report(r, sid, rid, c["id"], limitations=[])
    rendered = store.read_artifact(rp["artifact_id"]).decode()
    assert "n=1" in rendered and "shared-control" in rendered
    assert "Source/method qualifications" in rendered and "Author-supplied limitations" in rendered
    assert "computed" in rendered and "observation" in rendered and "supported" in rendered


def test_all_authored_text_is_escaped_and_cannot_forge_host_html_or_labels(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    hostile = '</pre><h1>HOST SUPPORTED</h1><script>alert(1)</script>\n# accepted\u202e'
    c = r.record_claim(sid, rid, text=hostile, kind="proposal", evidence_ids=[])
    rp = report(r, sid, rid, c["id"], title=hostile, heading=hostile, limitations=[hostile])
    rendered = store.read_artifact(rp["artifact_id"]).decode()
    assert "<script>" not in rendered and "<h1>HOST SUPPORTED" not in rendered
    assert "&lt;script&gt;" in rendered and "\\u202e" in rendered
    assert "\n# accepted" not in rendered
    assert "Author-supplied title" in rendered and "Author-supplied heading" in rendered
    assert "Claim kind: proposal" in rendered and "Assessment status: unassessed" in rendered
    with pytest.raises(TypeError):
        r.create_report(sid, rid, title="Title", sections=[], limitations=[], final_paragraph="Unsupported certainty")


@pytest.mark.parametrize("kind", ["proposal", "rejected"])
def test_explicit_proposals_and_rejections_can_pass_structurally_with_limitations(tmp_path, kind):
    store, r, sid, rid, _ = setup(tmp_path, acceptance={"require_report": True, "require_limitations": True})
    c = r.record_claim(sid, rid, text="An uncertain candidate", kind=kind, evidence_ids=[])
    rp = report(r, sid, rid, c["id"])
    validation = r.validate_report(sid, rid, rp["id"])
    assert validation["accepted"] is True
    rendered = r.accepted_report(sid, rid, rp["id"])["content"].decode()
    assert f"Claim kind: {kind}" in rendered and "unassessed" in rendered
    assert "not scientific assurance" in rendered


@pytest.mark.parametrize("status", [None, "partial", "contradicted", "unassessed"])
def test_unsupported_observations_cannot_pass_as_established(tmp_path, status):
    store, r, sid, rid, _ = setup(tmp_path, assessor=None if status is None else lambda request: verdict(request, status))
    _, _, c = evidence_claim(store, r, sid, rid)
    if status:
        r.assess_claim(sid, rid, c["id"])
    rp = report(r, sid, rid, c["id"])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False
    assert any("observation_not_supported" in issue for issue in v["issues"])
    with pytest.raises(ValueError):
        r.accepted_report(sid, rid, rp["id"])


def test_provisional_supported_semantic_review_has_a_positive_path(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path, assessor=lambda request: verdict(request))
    _, _, c = evidence_claim(store, r, sid, rid)
    r.assess_claim(sid, rid, c["id"])
    rp = report(r, sid, rid, c["id"])
    assert r.validate_report(sid, rid, rp["id"])["accepted"] is True
    assert "provisional" in r.accepted_report(sid, rid, rp["id"])["content"].decode()


@pytest.mark.parametrize("requirements", [{"unknown_rubric": True}, {"require_evidence": "yes"},
    {"require_report": 1}, {"require_data_artifact": None}, {"required_sections": "Findings"},
    {"required_sections": ["Findings", "Findings"]}, {"required_sections": [""]}])
def test_unknown_or_malformed_requirements_veto_acceptance(tmp_path, requirements):
    store, r, sid, rid, _, _, _, _, c = observed(tmp_path, requirements)
    rp = report(r, sid, rid, c["id"])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False and any("invalid_requirements" in item for item in v["issues"])


@pytest.mark.parametrize("sections", [[], [{"heading": "Findings", "claim_ids": []}],
    [{"heading": "Findings", "claim_ids": []}, {"heading": "Findings", "claim_ids": []}],
    [{"heading": "Other", "claim_ids": []}]])
def test_missing_duplicate_empty_sections_or_references_fail(tmp_path, sections):
    _, r, sid, rid, _ = setup(tmp_path, acceptance={"required_sections": ["Findings"]})
    assert hasattr(r, "create_report"), "structured report operation must exist"
    rp = r.create_report(sid, rid, title="Empty", sections=sections, limitations=[])
    assert r.validate_report(sid, rid, rp["id"])["accepted"] is False


@pytest.mark.parametrize("kind,requirements", [("proposal", {}), ("rejected", {}),
                                              ("observation", {"require_limitations": True})])
def test_explicit_limitations_are_required_for_unknown_claims_and_requested_reports(tmp_path, kind, requirements):
    store, r, sid, rid, _ = setup(tmp_path, acceptance=requirements)
    c = r.record_claim(sid, rid, text="Uncertain", kind=kind, evidence_ids=[])
    rp = report(r, sid, rid, c["id"], limitations=[])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False and "explicit_limitations_required" in v["issues"]


def test_evidence_required_needs_referenced_actual_material_not_unrelated_artifacts(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path, acceptance={"require_evidence": True})
    source(store, r, sid, rid)
    c = r.record_claim(sid, rid, text="Proposal", kind="proposal", evidence_ids=[])
    rp = report(r, sid, rid, c["id"])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False and "referenced_evidence_required" in v["issues"]


def test_data_required_rejects_userfile_json_even_with_deterministic_observation(tmp_path):
    store, r, sid, rid, _, _, _, _, c = observed(tmp_path, {"require_data_artifact": True}, kind="userfile")
    rp = report(r, sid, rid, c["id"])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False and "referenced_computed_data_required" in v["issues"]


@pytest.mark.parametrize("change", ["claim", "evidence", "assessment", "goal", "requirements", "report_bytes", "source_bytes"])
def test_acceptance_fails_closed_after_any_binding_or_artifact_change(tmp_path, change):
    store, r, sid, rid, tid, _, artifact, e, c = observed(tmp_path)
    rp = report(r, sid, rid, c["id"])
    assert r.validate_report(sid, rid, rp["id"])["accepted"] is True
    if change == "claim":
        r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Altered", kind="proposal", evidence_ids=[e["id"]])
    elif change == "evidence":
        r.reselect_evidence(sid, rid, e["id"], expected_revision=0, selector={"pointer": "/n"})
    elif change == "assessment":
        r.assess_claim(sid, rid, c["id"])
    elif change == "goal":
        store.update_task(tid, {"goal": "New requirement context"}, expected_revision=0)
    elif change == "requirements":
        store.update_task(tid, {"acceptance": {"require_evidence": True}}, expected_revision=0)
    else:
        digest = rp["artifact_hash"] if change == "report_bytes" else artifact["sha256"]
        (store.root / "artifacts" / digest).write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError):
        r.accepted_report(sid, rid, rp["id"])
    assert r.validate_report(sid, rid, rp["id"])["accepted"] is False


def test_reports_and_validations_are_task_scoped_and_later_same_task_run_can_handoff(tmp_path):
    store, r, sid, rid, tid, _, _, _, c = observed(tmp_path)
    rp = report(r, sid, rid, c["id"])
    validation = r.validate_report(sid, rid, rp["id"])
    later = store.create_run(sid, tid, Budget().to_dict())["id"]
    assert r.accepted_report(sid, later, rp["id"])["report"]["run_id"] == rid
    other_task = store.create_task(sid, "Other task")["id"]
    other_run = store.create_run(sid, other_task, Budget().to_dict())["id"]
    for operation in [lambda: r.get_report(sid, other_run, rp["id"]),
                      lambda: r.validate_report(sid, other_run, rp["id"]),
                      lambda: r.accepted_report(sid, other_run, rp["id"]),
                      lambda: r.get_validation(sid, other_run, validation["id"]),
                      lambda: report(r, sid, other_run, c["id"])]:
        with pytest.raises((ValueError, PermissionError, KeyError)):
            operation()


def test_newly_registered_secret_redacts_export_and_blocks_exact_handoff_instead_of_rewriting(tmp_path):
    store, r, sid, rid, _, _, _, _, c = observed(tmp_path)
    secret = "synthetic-report-secret-6655"
    rp = report(r, sid, rid, c["id"], title=secret)
    assert r.validate_report(sid, rid, rp["id"])["accepted"] is True
    exact_before = store.read_artifact(rp["artifact_id"])
    store.register_secrets([secret])
    assert secret not in json.dumps(r.get_report(sid, rid, rp["id"]))
    assert secret not in json.dumps(r.snapshot(sid, rid))
    with pytest.raises(ValueError):
        r.accepted_report(sid, rid, rp["id"])
    assert store.read_artifact(rp["artifact_id"]) == exact_before


def test_claim_revision_during_handoff_artifact_io_cannot_pass_cached_acceptance(tmp_path, monkeypatch):
    store, r, sid, rid, _, _, _, e, c = observed(tmp_path)
    rp = report(r, sid, rid, c["id"])
    r.validate_report(sid, rid, rp["id"])
    original_read = store.read_artifact
    changed = []
    def racing_read(artifact_id):
        content = original_read(artifact_id)
        if not changed:
            changed.append(True)
            r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Concurrent mutation",
                           kind="proposal", evidence_ids=[e["id"]])
        return content
    monkeypatch.setattr(store, "read_artifact", racing_read)
    with pytest.raises(ValueError):
        r.accepted_report(sid, rid, rp["id"])


def test_chinese_authored_material_is_readable_but_bidi_and_html_remain_escaped(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    c = r.record_claim(sid, rid, text="这是一项待验证的假设。<script>伪造</script>\u202e", kind="proposal", evidence_ids=[])
    rp = report(r, sid, rid, c["id"], title="中文研究报告", heading="研究发现", limitations=["疗效与安全性尚不明确。"])
    rendered = store.read_artifact(rp["artifact_id"]).decode()
    assert "中文研究报告" in rendered and "研究发现" in rendered
    assert "待验证的假设" in rendered and "疗效与安全性尚不明确" in rendered
    assert "<script>" not in rendered and "\\u202e" in rendered and "\u202e" not in rendered


def test_data_requirement_does_not_count_unreferenced_data_or_computed_prose(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path, acceptance={"require_data_artifact": True})
    source(store, r, sid, rid, kind="computed", content=b'{"real_data": 1}', key="unreferenced")
    s, _ = source(store, r, sid, rid, kind="computed", content="A computation narrative.", key="prose")
    e = r.select_evidence(sid, rid, s["id"], {"passage": "computation narrative"})
    c = r.observe_evidence(sid, rid, e["id"])
    rp = report(r, sid, rid, c["id"])
    v = r.validate_report(sid, rid, rp["id"])
    assert v["accepted"] is False and "referenced_computed_data_required" in v["issues"]
