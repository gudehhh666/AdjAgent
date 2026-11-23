"""Review-task references survive context trimming without crossing task scope."""

import json

from test_harness_research_records import setup, source, scope


def test_index_has_real_selectors_current_revisions_and_is_task_scoped(tmp_path):
    from adj_partner.research.context_index import reference_index
    store, r, sid, rid, tid = setup(tmp_path)
    s, _ = source(store, r, sid, rid, content=b'{"abstract":[{"text":"Bounded abstract."}]}',
                  provenance={"text_availability": "abstract"})
    e = r.select_evidence(sid, rid, s["id"], {"pointer": "/abstract/0/text"})
    c = r.record_claim(sid, rid, text="Old", kind="proposal", evidence_ids=[e["id"]])
    r.revise_claim(sid, rid, c["id"], expected_revision=0, text="Current", kind="proposal", evidence_ids=[e["id"]])
    sid2, rid2, _ = scope(store)
    r.record_claim(sid2, rid2, text="OTHER_TASK_PRIVATE", kind="proposal", evidence_ids=[])
    index = reference_index(store, sid, tid)
    assert index["claims"][0]["id"] == c["id"]
    assert index["claims"][0]["revision"] == 1
    assert index["claims"][0]["evidence_ids"] == [e["id"]]
    assert index["sources"][0]["selector"] == {"pointer": "/abstract/0/text"}
    assert index["sources"][0]["id"] == s["id"]
    assert "OTHER_TASK_PRIVATE" not in json.dumps(index)


def test_index_is_bounded_and_does_not_truncate_identifiers(tmp_path):
    from adj_partner.research.context_index import reference_index
    from adj_partner.tools.catalog import canonical_json
    store, r, sid, rid, tid = setup(tmp_path)
    ids = [r.record_claim(sid, rid, text="x" * 12000, kind="proposal", evidence_ids=[])["id"] for _ in range(20)]
    index = reference_index(store, sid, tid)
    assert len(index["claims"]) <= 12
    assert index["claims_omitted"] is True
    assert all(c["id"] in ids for c in index["claims"])
    assert len(canonical_json(index).encode()) <= 8000


def test_only_review_tasks_project_reference_index(tmp_path):
    from adj_partner.runtime.context import ContextView
    store, r, sid, rid, tid = setup(tmp_path, acceptance={"report_layout": "review"})
    claim = r.record_claim(sid, rid, text="Test", kind="proposal", evidence_ids=[])
    request = ContextView(store).build(sid, rid, {"tools": [], "revision": 0}, max_input_tokens=64000)
    assert any("Current research reference index" in m["content"] and claim["id"] in m["content"]
               for m in request["messages"])
    sid2, rid2, _ = scope(store)
    plain = ContextView(store).build(sid2, rid2, {"tools": [], "revision": 0}, max_input_tokens=64000)
    assert not any("Current research reference index" in m["content"] for m in plain["messages"])


def test_string_abstract_has_real_selector_and_expanded_task_keeps_more_sources(tmp_path):
    from adj_partner.research.context_index import reference_index
    store, r, sid, rid, tid = setup(tmp_path, acceptance={"report_layout": "review", "min_distinct_sources": 5})
    created = []
    for i in range(6):
        material, _ = source(store, r, sid, rid,
            content=json.dumps({"abstract": f"Actual string abstract {i}."}).encode(), key=f"source-{i}",
            provenance={"text_availability": "abstract", "doi": f"10.1234/{i}"})
        created.append(material["id"])
    index = reference_index(store, sid, tid)
    assert {s["id"] for s in index["sources"]} == set(created)
    for item in index["sources"]:
        assert item["selector"] == {"pointer": "/abstract"}
        evidence = r.select_evidence(sid, rid, item["id"], item["selector"])
        assert evidence["value"].startswith("Actual string abstract")


def test_expanded_source_index_does_not_let_recent_duplicate_dois_hide_other_works(tmp_path):
    from adj_partner.research.context_index import reference_index
    store, r, sid, rid, tid = setup(tmp_path, acceptance={"report_layout": "review", "min_distinct_sources": 5})
    for i in range(6):
        source(store, r, sid, rid, content=json.dumps({"abstract": f"Work {i}"}).encode(), key=f"work{i}",
               provenance={"doi": f"10.1234/{i}", "text_availability": "abstract"})
    for i in range(10):
        source(store, r, sid, rid, content=json.dumps({"abstract": f"Alias of work 5, record {i}"}).encode(), key=f"alias{i}",
               provenance={"doi": "https://doi.org/10.1234/5", "text_availability": "abstract"})
    index = reference_index(store, sid, tid)
    assert len(index["sources"]) == 6
