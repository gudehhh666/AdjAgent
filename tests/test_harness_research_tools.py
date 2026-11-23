"""Actual shared executor integration; research tools cannot manufacture authority."""

import json

import pytest

from adj_partner.contracts import ToolCall, new_id
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from test_harness_research_records import api, scope, setup, source
from test_harness_research_assessments import verdict


def tools(store, research, sid):
    catalog = ToolCatalog()
    assert hasattr(api(), "register_research_tools"), "shared research tool registration must exist"
    api().register_research_tools(catalog, research)
    session = store.get_session(sid)
    config = {**session["config"], "allowed_tools": catalog.names(), "enabled_tools": catalog.names()}
    store.update_session(sid, config, expected_revision=session["revision"])
    return catalog, ToolExecutor(store, catalog)


def invoke(executor, sid, rid, name, **arguments):
    return executor.execute(ToolCall(new_id("call"), name, arguments), sid, rid)


def test_shared_tools_real_executor_builds_and_validates_actual_report(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path, acceptance={"require_report": True, "require_evidence": True})
    s, _ = source(store, r, sid, rid)
    catalog, executor = tools(store, r, sid)
    assert set(catalog.names()) == {"research_state", "evidence_select", "evidence_reselect", "claim_record", "claim_revise",
                                     "claim_observe", "claim_assess", "report_create", "report_validate",
                                     "design_record_read", "experiment_state", "experiment_design_revise"}
    description = catalog.get("claim_record").spec.description
    assert "previous" in description.lower() and "evidence_" in description and "source_" in description
    e = invoke(executor, sid, rid, "evidence_select", source_id=s["id"], selector={"passage": "First finding."})
    assert e.status == "ok"
    c = invoke(executor, sid, rid, "claim_observe", evidence_id=e.data["id"])
    assert c.status == "ok"
    rp = invoke(executor, sid, rid, "report_create", title="Tool report",
                sections=[{"heading": "Results", "claim_ids": [c.data["id"]]}], limitations=["Synthetic fixture only."])
    assert rp.status == "ok" and rp.artifacts[0]["id"] == rp.data["artifact_id"]
    v = invoke(executor, sid, rid, "report_validate", report_id=rp.data["id"])
    assert v.status == "ok" and v.data["accepted"] is True
    state = invoke(executor, sid, rid, "research_state")
    assert state.status == "ok" and any(item["id"] == rp.data["id"] for item in state.data["records"])
    assert r.accepted_report(sid, rid, rp.data["id"])["content"] == store.read_artifact(rp.data["artifact_id"])
    assert len(store.list_actions(rid)) == 5


@pytest.mark.parametrize("name,args", [
    ("claim_record", {"text": "Fake", "kind": "observation", "evidence_ids": [], "supported": True}),
    ("claim_record", {"text": "Fake", "kind": "proposal", "evidence_ids": [], "session_id": "override"}),
    ("claim_assess", {"claim_id": "claim_fake", "result": {"status": "supported"}}),
    ("research_state", {"run_id": "override"}),
    ("evidence_select", {"source_id": "source_fake", "selector": {"passage": "x", "pointer": ""}}),
    ("report_create", {"title": "X", "sections": [], "limitations": [], "final_paragraph": "Safe"}),
    ("claim_revise", {"claim_id": "claim_fake", "expected_revision": True, "text": "X", "kind": "proposal", "evidence_ids": []}),
])
def test_schemas_reject_model_authority_scope_and_unchecked_prose(tmp_path, name, args):
    store, r, sid, rid, _ = setup(tmp_path)
    _, executor = tools(store, r, sid)
    result = invoke(executor, sid, rid, name, **args)
    assert result.status == "error" and result.error["code"] == "input_schema"
    assert r.events(sid, rid) == []


def test_missing_assessor_is_honest_unassessed_tool_result_and_no_source_creation_tool(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    catalog, executor = tools(store, r, sid)
    c = invoke(executor, sid, rid, "claim_record", text="Unknown mechanism", kind="observation", evidence_ids=[])
    a = invoke(executor, sid, rid, "claim_assess", claim_id=c.data["id"])
    assert a.status == "ok" and a.data["status"] == "unassessed"
    assert "source_create" not in catalog.names() and "accepted_report" not in catalog.names()
    assert invoke(executor, sid, rid, "source_create").status == "denied"


def test_cross_store_catalog_reuse_is_rejected_without_foreign_writes(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    catalog, _ = tools(store, r, sid)
    foreign = Store(tmp_path / "foreign")
    fsid, frid, _ = scope(foreign)
    session = foreign.get_session(fsid)
    config = {**session["config"], "allowed_tools": catalog.names(), "enabled_tools": catalog.names()}
    foreign.update_session(fsid, config, 0)
    result = invoke(ToolExecutor(foreign, catalog), fsid, frid, "claim_record", text="Forbidden", kind="proposal", evidence_ids=[])
    assert result.status == "denied" and result.error["code"] == "research_store_mismatch"
    assert r.list_claims(sid, rid) == []
    assert api().ResearchStore(foreign).list_claims(fsid, frid) == []


def test_reopened_same_store_uses_actual_context_redaction_and_primes_assessor_secrets(tmp_path):
    secret = "synthetic-assessor-credential-9012"
    def provider(request):
        result = verdict(request)
        result["rationale"] += secret
        return result
    store, r, sid, rid, _ = setup(tmp_path, api().HostAssessor(provider, secrets=(secret,)))
    s, _ = source(store, r, sid, rid)
    catalog, _ = tools(store, r, sid)
    reopened = Store(store.root)
    reopened.register_secrets(["new-context-secret-321"])
    executor = ToolExecutor(reopened, catalog)
    e = invoke(executor, sid, rid, "evidence_select", source_id=s["id"], selector={"passage": "First"})
    c = invoke(executor, sid, rid, "claim_record", text="new-context-secret-321", kind="observation", evidence_ids=[e.data["id"]])
    a = invoke(executor, sid, rid, "claim_assess", claim_id=c.data["id"])
    assert a.status == "ok" and a.data["status"] == "supported"
    assert secret not in json.dumps(a.to_dict())
    exported = json.dumps(api().ResearchStore(reopened).snapshot(sid, rid))
    assert secret not in exported and "new-context-secret-321" not in exported
    assert secret not in json.dumps(reopened.list_actions(rid))


def test_real_executor_preserves_optimistic_revisions_and_replay_does_not_duplicate(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    _, executor = tools(store, r, sid)
    call = ToolCall("unique_claim_call", "claim_record", {"text": "Proposal", "kind": "proposal", "evidence_ids": []})
    first = executor.execute(call, sid, rid)
    second = executor.execute(call, sid, rid)
    assert first.data["id"] == second.data["id"]
    revised = invoke(executor, sid, rid, "claim_revise", claim_id=first.data["id"], expected_revision=0,
                     text="Revised", kind="proposal", evidence_ids=[])
    assert revised.status == "ok" and revised.data["revision"] == 1
    stale = invoke(executor, sid, rid, "claim_revise", claim_id=first.data["id"], expected_revision=0,
                   text="Stale", kind="proposal", evidence_ids=[])
    assert stale.status == "error" and stale.error["code"] == "research_invalid"
    assert len(r.events(sid, rid)) == 2


def test_long_state_is_paginated_without_repeating_history_or_silently_dropping_records(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    catalog, executor = tools(store, r, sid)
    expected = set()
    for index in range(30):
        c = r.record_claim(sid, rid, text=f"Claim {index} " + "x" * 14000, kind="proposal", evidence_ids=[])
        r.revise_claim(sid, rid, c["id"], expected_revision=0, text=f"Current {index} " + "y" * 14000,
                       kind="proposal", evidence_ids=[])
        expected.add(c["id"])
    assert len(json.dumps(r.snapshot(sid, rid))) > 1_000_000
    page = invoke(executor, sid, rid, "research_state")
    assert page.status == "ok"
    assert len(json.dumps(page.data)) < 100_000
    assert "history" not in page.data and "events" not in page.data
    assert page.data["has_more"] is True and page.data["next_cursor"] is not None
    seen = set()
    while True:
        seen.update(item["id"] for item in page.data["records"])
        assert all(item["revision"] == 1 and "text" in item["truncated_fields"] for item in page.data["records"])
        if not page.data["has_more"]:
            break
        page = invoke(executor, sid, rid, "research_state", cursor=page.data["next_cursor"], limit=7)
        assert page.status == "ok"
    assert seen == expected
    assert len(r.snapshot(sid, rid)["history"]) == 60


def test_model_report_result_is_compact_while_host_retains_exact_full_snapshot(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    _, executor = tools(store, r, sid)
    claims = [r.record_claim(sid, rid, text="Proposal " + "z" * 12000, kind="proposal", evidence_ids=[])["id"] for _ in range(8)]
    result = invoke(executor, sid, rid, "report_create", title="Large report", sections=[{"heading": "Findings", "claim_ids": claims}],
                    limitations=["Synthetic unknown claims."])
    assert result.status == "ok"
    assert len(json.dumps(result.data)) < 10000
    assert "claim_snapshots" not in result.data and result.data["validation_needed"] is True
    host = r.get_report(sid, rid, result.data["id"])
    assert len(host["claim_snapshots"]) == 8
    validation = invoke(executor, sid, rid, "report_validate", report_id=result.data["id"])
    assert validation.status == "ok" and validation.data["accepted"] is True
    assert validation.data["report_hash"] == host["artifact_hash"]


def test_state_page_labels_assessments_and_validation_as_recorded_not_current_eligibility(tmp_path):
    store, r, sid, rid, _ = setup(tmp_path)
    _, executor = tools(store, r, sid)
    c = r.record_claim(sid, rid, text="An unknown proposal", kind="proposal", evidence_ids=[])
    r.assess_claim(sid, rid, c["id"])
    rp = r.create_report(sid, rid, title="Proposal", sections=[{"heading": "Ideas", "claim_ids": [c["id"]]}],
                          limitations=["Unknown efficacy."])
    r.validate_report(sid, rid, rp["id"])
    r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Revised idea", kind="proposal", evidence_ids=[])
    state = invoke(executor, sid, rid, "research_state")
    recorded = [item for item in state.data["records"] if item["entity"] in {"assessment", "validation"}]
    assert len(recorded) == 2
    assert all(item.get("current_eligibility_checked") is False for item in recorded)
    assert all(item.get("authority") == "recorded_outcome_not_handoff_permission" for item in recorded)
