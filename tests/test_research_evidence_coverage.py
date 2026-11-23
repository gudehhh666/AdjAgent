"""Coverage gates count cited works and excerpts, not repeated ledger IDs."""
from copy import deepcopy

import pytest

from adj_partner.research.acceptance import requirements, structural_issues


def report_fixture():
    snapshots = []
    for i in range(5):
        source = {"id": f"s{i}", "kind": "literature", "title": f"Distinct study {i}",
                  "artifact_id": f"a{i}", "artifact_hash": f"hash{i}",
                  "provenance": {"doi": f"10.1234/{i}", "text_availability": "abstract"}}
        evidence = [{"id": f"e{i}-{j}", "source_id": source["id"], "text": f"Finding {i} {j}",
                     "value": None, "locator": {"passage": f"Finding {i} {j}"}} for j in range(2)]
        snapshots.append({"claim": {"id": f"c{i}", "kind": "proposal"}, "assessment": None,
                          "assessment_current": False, "sources": [source], "evidence": evidence})
    return {"artifact_id": "report", "sections": [{"heading": "Decision", "claim_ids": ["c0"]}],
            "claim_snapshots": snapshots, "limitations": ["Synthetic only"]}


def check(report):
    return structural_issues(report, {"acceptance": {"min_distinct_sources": 5,
                                                    "min_distinct_evidence": 8}}, {})


def test_sufficient_bound_works_pass_but_one_work_many_excerpts_fails():
    report = report_fixture()
    assert check(report) == []
    report["claim_snapshots"] = report["claim_snapshots"][:1] * 8
    issues = check(report)
    assert "distinct_sources_below_minimum: 1/5" in issues
    assert "distinct_evidence_below_minimum: 2/8" in issues


def test_aliases_and_identical_excerpts_do_not_inflate_coverage():
    report = report_fixture()
    original, duplicate = report["claim_snapshots"][:2]
    duplicate["sources"][0]["provenance"]["doi"] = "https://doi.org/10.1234/0"
    duplicate["evidence"][0]["text"] = original["evidence"][0]["text"]
    duplicate["evidence"][1]["text"] = original["evidence"][1]["text"]
    issues = check(report)
    assert "distinct_sources_below_minimum: 4/5" in issues
    # Duplicate source IDs / different locators still represent the same work/excerpt.
    duplicate["evidence"].append({**duplicate["evidence"][0], "id": "new", "locator": {"pointer": "/text"}})
    assert check(report) == issues


def test_shared_identifier_bridge_and_title_alias_merge_transitively():
    from adj_partner.research.coverage import evidence_coverage
    report = report_fixture()
    a, b, c = [s["sources"][0] for s in report["claim_snapshots"][:3]]
    b["provenance"] = {"pmid": "123", "text_availability": "abstract"}
    c["provenance"] = {"doi": a["provenance"]["doi"], "pmid": "123", "text_availability": "full_text"}
    assert evidence_coverage(report)["distinct_sources"] == 3
    report["claim_snapshots"][3]["sources"][0]["title"] = a["title"].upper() + "."
    assert evidence_coverage(report)["distinct_sources"] == 2


def test_uncited_sources_title_only_and_computed_do_not_count():
    from adj_partner.research.coverage import evidence_coverage
    report = report_fixture()
    report["claim_snapshots"][0]["evidence"] = []
    report["claim_snapshots"][1]["sources"][0]["provenance"]["text_availability"] = "title_only"
    report["claim_snapshots"][2]["sources"][0]["kind"] = "computed"
    result = evidence_coverage(report)
    assert result["distinct_sources"] == 2
    assert result["distinct_evidence"] == 4


@pytest.mark.parametrize("value", [True, -1, 0, 101, 1.2, "5", None])
@pytest.mark.parametrize("field", ["min_distinct_sources", "min_distinct_evidence"])
def test_coverage_thresholds_require_bounded_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        requirements({field: value})


def test_legacy_requirements_do_not_acquire_new_hash_fields():
    assert set(requirements({})) == {"require_report", "require_evidence", "require_limitations",
                                     "require_data_artifact", "required_sections"}


def test_real_report_acceptance_enforces_cited_material_and_keeps_old_report(tmp_path):
    from test_harness_research_records import setup
    from adj_partner.research.acceptance import ReportNotAccepted
    store, research, sid, rid, _ = setup(tmp_path, acceptance={
        "report_layout": "review", "min_distinct_sources": 2, "min_distinct_evidence": 2})
    claims = []
    for i in range(2):
        provenance = {"doi": f"10.1234/integration-{i}", "text_availability": "abstract"}
        artifact = store.put_artifact(f"Fixture finding {i}.", f"source{i}.txt", "text/plain",
                                      provenance, session_id=sid, run_id=rid)
        source = research.register_source(sid, rid, artifact_id=artifact["id"], kind="literature",
            title=f"Separate fixture study {i}", provenance=provenance, idempotency_key=f"source{i}")
        evidence = research.select_evidence(sid, rid, source["id"], {"passage": f"Fixture finding {i}."})
        claims.append(research.record_claim(sid, rid, text=f"Hypothesis {i}", kind="proposal",
                                             evidence_ids=[evidence["id"]]))
    def create(ids):
        return research.create_report(sid, rid, title="Fixture", sections=[{"heading": "Decision",
            "claim_ids": ids}], limitations=["Synthetic only"])
    old = create([claims[0]["id"]])
    old_bytes = store.read_artifact(old["artifact_id"])
    assert research.validate_report(sid, rid, old["id"])["accepted"] is False
    with pytest.raises(ReportNotAccepted):
        research.accepted_report(sid, rid, old["id"])
    new = create([c["id"] for c in claims])
    assert research.validate_report(sid, rid, new["id"])["accepted"] is True
    research.accepted_report(sid, rid, new["id"])
    assert store.read_artifact(old["artifact_id"]) == old_bytes
