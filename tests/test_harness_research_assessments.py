"""Host-controlled assessment boundaries with small explicit synthetic providers."""

import json

import pytest

from test_harness_research_records import api, scope, setup, source


def evidence_claim(store, r, sid, rid, kind="observation"):
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    c = r.record_claim(sid, rid, text="First finding proves safe efficacy and mechanism.",
                       kind=kind, evidence_ids=[e["id"]])
    return s, e, c


def verdict(snapshot, status="supported"):
    return {"status": status, "rationale": "Synthetic fixture reviewer verdict; not scientific assurance.",
            "evidence_ids": [item["id"] for item in snapshot["evidence"]]}


def test_citations_and_exact_overlap_are_not_semantic_support_without_provider(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    _, _, c = evidence_claim(store, r, sid, rid)
    assert hasattr(r, "assess_claim"), "assessment operation must exist"
    a = r.assess_claim(sid, rid, c["id"])
    assert a["status"] == "unassessed"
    assert a["reason_code"] == "assessor_unavailable"
    assert a["provisional"] is True
    assert r.current_assessment(sid, rid, c["id"])["status"] == "unassessed"


@pytest.mark.parametrize("kind,content,selector,phrase", [
    ("literature", "The article claims efficacy.", {"passage": "claims efficacy"}, "Source quotation"),
    ("computed", b'{"n": 1, "score": -0.3}', {"pointer": "/score"}, "Computed output"),
    ("userfile", b'{"efficacy": true}', {"pointer": "/efficacy"}, "User-supplied content"),
])
def test_narrow_observations_are_host_generated_and_bound_to_exact_material(tmp_path, kind, content, selector, phrase):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid, content=content, kind=kind)
    e = r.select_evidence(sid, rid, s["id"], selector)
    assert hasattr(r, "observe_evidence"), "narrow observation operation must exist"
    c = r.observe_evidence(sid, rid, e["id"])
    assert c["kind"] == "observation" and c["origin"] == "host_observation"
    assert phrase in c["text"] and "not" in c["text"]
    a = r.current_assessment(sid, rid, c["id"])
    assert a["status"] == "supported" and a["provisional"] is False
    assert a["method"] == "literal_material_observation"
    assert a["bindings"]["evidence"] == {e["id"]: 0}
    assert a["bindings"]["sources"] == {s["id"]: s["artifact_hash"]}
    assert "Synthetic material" in c["text"]


@pytest.mark.parametrize("status", ["supported", "partial", "contradicted", "unassessed"])
def test_injected_review_is_explicit_provisional_and_persists_public_io(tmp_path, status):
    requests = []
    def reviewer(snapshot):
        requests.append(snapshot)
        return verdict(snapshot, status)
    assert hasattr(api(), "HostAssessor"), "host assessment metadata wrapper must exist"
    provider = api().HostAssessor(reviewer, reviewer_id="fixture-reviewer", method="synthetic-semantic-review")
    store, r, sid, rid, tid = setup(tmp_path, provider)
    _, e, c = evidence_claim(store, r, sid, rid)
    a = r.assess_claim(sid, rid, c["id"])
    assert a["status"] == status
    assert a["reviewer_id"] == "fixture-reviewer" and a["method"] == "synthetic-semantic-review"
    assert a["provisional"] is True
    assert len(requests) == 1
    request = requests[0]
    assert (request["session_id"], request["run_id"], request["task_id"]) == (sid, rid, tid)
    assert request["claim"]["id"] == c["id"] and request["evidence"][0]["id"] == e["id"]
    assert request["sources"][0]["provenance"]["synthetic_fixture"] is True
    assert a["request"] == request and a["result"] == verdict(request, status)


@pytest.mark.parametrize("response", [None, {}, {"status": "accepted"},
    {"status": "supported", "rationale": "", "evidence_ids": []},
    {"status": "supported", "rationale": "Yes", "evidence_ids": ["unretrieved"]},
    {"status": "supported", "rationale": "Yes", "evidence_ids": []},
    {"status": "supported", "rationale": 1, "evidence_ids": []},
    {"status": "supported", "rationale": "Yes", "evidence_ids": [], "private_reasoning": "DO NOT RETAIN"},
    {"status": "partial", "rationale": float("nan"), "evidence_ids": []}])
def test_malformed_host_verdict_never_becomes_support(tmp_path, response):
    store, r, sid, rid, _ = setup(tmp_path, lambda request: response)
    _, _, c = evidence_claim(store, r, sid, rid)
    assert hasattr(r, "assess_claim"), "assessment operation must exist"
    a = r.assess_claim(sid, rid, c["id"])
    assert a["status"] == "unassessed" and a["reason_code"] == "malformed_assessment"
    assert "DO NOT RETAIN" not in json.dumps(r.snapshot(sid, rid))


def test_provider_error_is_unassessed_without_retry_and_secrets_are_redacted(tmp_path):
    calls = []
    secret = "fixture-provider-secret-4321"
    def reviewer(request):
        calls.append(request)
        raise RuntimeError(f"synthetic error {secret}")
    assert hasattr(api(), "HostAssessor"), "host assessment metadata wrapper must exist"
    store, r, sid, rid, _ = setup(tmp_path, api().HostAssessor(reviewer, secrets=(secret,)))
    _, _, c = evidence_claim(store, r, sid, rid)
    a = r.assess_claim(sid, rid, c["id"])
    assert a["status"] == "unassessed" and a["reason_code"] == "assessor_error"
    assert len(calls) == 1
    assert secret not in json.dumps(r.snapshot(sid, rid))


def test_assessment_is_outside_transaction_and_concurrent_claim_revision_cannot_commit_support(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    _, e, c = evidence_claim(store, r, sid, rid)
    def reviewer(request):
        r.revise_claim(sid, rid, c["id"], expected_revision=0,
                       text="Concurrent new claim", kind="observation", evidence_ids=[e["id"]])
        return verdict(request)
    r.assessor = reviewer
    assert hasattr(r, "assess_claim"), "assessment operation must exist"
    a = r.assess_claim(sid, rid, c["id"])
    assert a["status"] == "unassessed" and a["reason_code"] == "stale_inputs"
    assert a["bindings"]["claim_revision"] == 0
    assert r.get_claim(sid, rid, c["id"])["revision"] == 1


def test_evidence_reselection_and_claim_edits_invalidate_deterministic_assessment(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, _ = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First finding."})
    assert hasattr(r, "observe_evidence"), "narrow observation operation must exist"
    c = r.observe_evidence(sid, rid, e["id"])
    a = r.current_assessment(sid, rid, c["id"])
    r.reselect_evidence(sid, rid, e["id"], expected_revision=0, selector={"passage": "Second finding."})
    assert r.current_assessment(sid, rid, c["id"])["status"] == "unassessed"
    assert r.get_assessment(sid, rid, a["id"])["status"] == "supported"
    r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Safe efficacy", kind="observation", evidence_ids=[e["id"]])
    assert r.current_assessment(sid, rid, c["id"])["status"] == "unassessed"


def test_assessment_scope_is_checked_before_provider_call(tmp_path):
    requests = []
    store, r, sid, rid, _ = setup(tmp_path, lambda request: requests.append(request))
    _, _, c = evidence_claim(store, r, sid, rid)
    foreign_sid, foreign_rid, _ = scope(store)
    assert hasattr(r, "assess_claim"), "assessment operation must exist"
    with pytest.raises((KeyError, PermissionError)):
        r.assess_claim(foreign_sid, foreign_rid, c["id"])
    assert requests == []


def test_current_assessment_checks_actual_artifact_integrity(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    s, artifact = source(store, r, sid, rid)
    e = r.select_evidence(sid, rid, s["id"], {"passage": "First"})
    c = r.observe_evidence(sid, rid, e["id"])
    (store.root / "artifacts" / artifact["sha256"]).write_bytes(b"tampered")
    assert r.current_assessment(sid, rid, c["id"])["status"] == "unassessed"


def test_narrow_observation_can_quote_public_redacted_selection_without_secret_leak(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    secret = "synthetic-literal-secret-753"
    store.register_secrets([secret])
    s, _ = source(store, r, sid, rid, content=f"A literal {secret} passage.")
    e = r.select_evidence(sid, rid, s["id"], {"passage": f"literal {secret}"})
    c = r.observe_evidence(sid, rid, e["id"])
    assert secret not in json.dumps(c)
    assert "[REDACTED]" in c["text"]
    assert r.current_assessment(sid, rid, c["id"])["status"] == "supported"
