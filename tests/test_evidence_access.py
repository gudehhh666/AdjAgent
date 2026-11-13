"""Deterministic synthetic evidence access; no scientific gold or model calls."""
import json
from types import SimpleNamespace

import pytest

from adj_partner.contracts import Budget
from adj_partner.runtime.context import ContextView
from adj_partner.tools.catalog import canonical_json
from test_harness_research_records import setup, source, scope


def fixture(tmp_path, content, provenance=None):
    from adj_partner.evidence_access import read_evidence
    store, research, sid, rid, _ = setup(tmp_path)
    s, a = source(store, research, sid, rid, content=content,
                  provenance=provenance or {"text_availability": "derived_full_text", "url": "https://example.org/study"})
    context = SimpleNamespace(store=store, session_id=sid, run_id=rid, check_cancelled=lambda: None)
    return read_evidence, context, s, a


def test_query_reaches_middle_of_long_paragraph_and_survives_context(tmp_path):
    text = "Introductory text. " * 900 + "Critical outcome was unchanged under the control." + " Follow-up." * 900
    read, ctx, src, _ = fixture(tmp_path, json.dumps({"sections": [{"text": text}]}).encode())
    result = read(ctx, source_id=src["id"], query="Critical outcome")
    d = result.data
    assert "Critical outcome was unchanged under the control." in d["text"]
    assert d["start"] > 500 and d["selector"] == {"pointer": "/sections/0/text"}
    assert d["text"] == text[d["start"]:d["end"]]
    assert d["total"] == len(text) and d["next"]["start"] == d["end"]
    assert d["artifact_hash"] == src["artifact_hash"]
    assert d["text_availability"] == "derived_full_text"
    projected = ContextView(ctx.store).tool_message(ctx.session_id, ctx.run_id, "test", result.to_dict())
    assert len(projected["content"].encode()) < 2400
    assert not json.loads(projected["content"]).get("context_truncated")
    assert "Critical outcome was unchanged" in projected["content"]


def test_table_exact_cell_and_escaped_pointer(tmp_path):
    value = {"table/a": {"rows": [["donor", "time"], ["S1", "D0"], ["S1", "D2"]]}, "dose": 3.5}
    read, ctx, src, _ = fixture(tmp_path, json.dumps(value).encode())
    d = read(ctx, source_id=src["id"], selector={"pointer": "/table~1a/rows/2/1"}).data
    assert d["text"] == "D2" and d["end"] == d["total"] == 2 and d["next"] is None
    d = read(ctx, source_id=src["id"], selector={"pointer": "/dose"}).data
    assert d["value"] == 3.5 and d["value_type"] == "number"


def test_plain_artifact_query_and_empty_query_not_absence(tmp_path):
    read, ctx, src, a = fixture(tmp_path, "Header\n" * 2000 + "individual: subject-one\ntime: day-zero\n")
    d = read(ctx, artifact_id=a["id"], query="individual:").data
    assert "individual: subject-one" in d["text"] and d["source_id"] == src["id"]
    d = read(ctx, source_id=src["id"], query="missing-token").data
    assert d["matched"] is False and "not evidence of absence" in d["boundary"]


def test_placeholder_identifier_and_empty_pointer_are_ignored(tmp_path):
    read, ctx, src, _ = fixture(tmp_path, "Header\nindividual: subject-one\n")
    d = read(ctx, source_id=src["id"], artifact_id="none", selector={"pointer": ""}, query="individual:").data
    assert "individual: subject-one" in d["text"]
    d = read(ctx, source_id=src["id"], artifact_id="x", query="individual:").data
    assert d["source_id"] == src["id"]
    with pytest.raises(ValueError, match="exactly one"):
        read(ctx, source_id="none", artifact_id="")
    with pytest.raises(ValueError, match="JSON selector"):
        read(ctx, source_id=src["id"], selector={"pointer": "/x"})


def test_query_occurrence_and_unicode_windows(tmp_path):
    read, ctx, src, _ = fixture(tmp_path, ("中文文字" * 300 + "target first" + "中文文字" * 300 + "target second"))
    d = read(ctx, source_id=src["id"], query="target", match_index=1).data
    assert "target second" in d["text"] and "target first" not in d["text"]
    assert len(canonical_json(read(ctx, source_id=src["id"], query="target", match_index=1).to_dict()).encode()) < 2400


def test_other_session_and_other_task_are_denied_for_both_ids(tmp_path):
    read, ctx, src, a = fixture(tmp_path, "Visible only to the authorized task.")
    original_sid, original_rid = ctx.session_id, ctx.run_id
    other_sid, other_rid, _ = scope(ctx.store)
    task = ctx.store.create_task(original_sid, "Other task", {})
    other_task_run = ctx.store.create_run(original_sid, task["id"], Budget().to_dict())
    for sid, rid in [(other_sid, other_rid), (original_sid, other_task_run["id"])]:
        ctx.session_id, ctx.run_id = sid, rid
        for args in [{"source_id": src["id"]}, {"artifact_id": a["id"]}]:
            with pytest.raises((PermissionError, KeyError, ValueError)):
                read(ctx, **args)
    ctx.session_id, ctx.run_id = original_sid, original_rid
    assert "authorized" in read(ctx, source_id=src["id"]).data["text"]


def test_tampered_artifact_is_never_read(tmp_path):
    read, ctx, src, a = fixture(tmp_path, "Immutable original.")
    path = ctx.store.root / a["path"]
    path.write_bytes(b"Tampered content.")
    for args in [{"source_id": src["id"]}, {"artifact_id": a["id"]}]:
        with pytest.raises((ValueError, OSError)):
            read(ctx, **args)


@pytest.mark.parametrize("args", [{"start": -1}, {"start": True}, {"start": 999}, {"selector": {"pointer": "/../secret"}}, {"selector": {"pointer": "/x~2"}}, {"max_characters": 0}, {"match_index": -1}, {"query": ""}])
def test_invalid_boundaries_fail(tmp_path, args):
    read, ctx, src, _ = fixture(tmp_path, json.dumps({"x": "ok"}).encode())
    with pytest.raises(ValueError):
        read(ctx, source_id=src["id"], **args)


def test_container_not_serialized_as_hidden_payload(tmp_path):
    read, ctx, src, _ = fixture(tmp_path, json.dumps({"rows": [["x"]] * 10000}).encode())
    with pytest.raises(ValueError, match="scalar"):
        read(ctx, source_id=src["id"], selector={"pointer": "/rows"})


def test_registration_executor_permissions_and_real_projection(tmp_path):
    from pathlib import Path
    from adj_partner.application import AgentService
    from adj_partner.contracts import ToolCall
    from adj_partner.evidence_access import register_evidence_access
    from adj_partner.models import ScriptedPort
    service = AgentService(tmp_path / "service", project_root=Path(__file__).resolve().parents[1], model=ScriptedPort([]))
    try:
        if "evidence_read" not in service.catalog.names():
            register_evidence_access(service)
        sid = service.create_session(profile="general", allowed_tools=["evidence_read", "tools_enable"])["id"]
        run = service.submit(sid, "Synthetic evidence access")
        s, _ = source(service.store, service.research, sid, run["id"], content="prefix " * 1000 + "exact-window-target")
        denied = service.executor.execute(ToolCall("denied", "evidence_read", {"source_id": s["id"]}), sid, run["id"])
        assert denied.status != "ok"
        enabled = service.executor.execute(ToolCall("enable", "tools_enable", {"name": "evidence_read"}), sid, run["id"])
        assert enabled.status == "ok", enabled.to_dict()
        result = service.executor.execute(ToolCall("read", "evidence_read", {"source_id": s["id"], "query": "exact-window-target"}), sid, run["id"])
        assert result.status == "ok", result.to_dict()
        message = ContextView(service.store).tool_message(sid, run["id"], "read", result.to_dict())
        assert "exact-window-target" in message["content"] and "context_truncated" not in message["content"]
    finally:
        service.close()


def test_sensitive_fields_and_secret_strings_are_redacted_before_search(tmp_path):
    secret = "synthetic-private-value-abc987"
    raw = json.dumps({"api_key": "hidden-key-value", "body": "Finding " + secret + " public observation"}).encode()
    read, ctx, src, _ = fixture(tmp_path, raw)
    ctx.store.register_secrets([secret])
    assert read(ctx, source_id=src["id"], query=secret).data["matched"] is False
    assert read(ctx, source_id=src["id"], query="hidden-key-value").data["matched"] is False
    d = read(ctx, source_id=src["id"], selector={"pointer": "/body"}).data
    assert secret not in d["text"] and "[REDACTED]" in d["text"]
    assert d["artifact_hash"] != d["projection_hash"]


def test_same_task_later_run_allowed_but_unscoped_artifact_denied(tmp_path):
    read, ctx, src, _ = fixture(tmp_path, "Actual same-task evidence.")
    task = ctx.store.get_run(ctx.run_id)["task_id"]
    ctx.run_id = ctx.store.create_run(ctx.session_id, task, Budget().to_dict())["id"]
    assert "same-task" in read(ctx, source_id=src["id"]).data["text"]
    a = ctx.store.put_artifact("Host-only material", "private.txt", "text/plain")
    with pytest.raises((PermissionError, KeyError)):
        read(ctx, artifact_id=a["id"])


def test_continuation_reconstructs_unicode_without_gaps(tmp_path):
    text = "甲乙丙丁\\\"\n" * 500
    read, ctx, src, _ = fixture(tmp_path, text)
    pieces, start = [], 0
    while True:
        result = read(ctx, source_id=src["id"], start=start)
        d = result.data
        assert len(canonical_json(result.to_dict()).encode()) < 2400
        pieces.append(d["text"])
        if d["next"] is None:
            break
        assert d["next"]["start"] > start
        start = d["next"]["start"]
    assert "".join(pieces) == text


def test_portable_flat_schema_still_enforces_exactly_one_identifier(tmp_path):
 from pathlib import Path
 from adj_partner.application import AgentService
 from adj_partner.models import ScriptedPort
 from adj_partner.contracts import ToolCall
 service=AgentService(tmp_path/'flat',model=ScriptedPort([]))
 try:
  schema=service.catalog.get('evidence_read').spec.input_schema
  assert 'oneOf' not in schema and 'anyOf' not in schema
  sid=service.create_session(profile='research-evidence-first')['id']
  rid=service.submit(sid,'Scope validation')['id']
  for i,args in enumerate(({}, {'source_id':'x','artifact_id':'y'})):
   result=service.executor.execute(ToolCall('bad'+str(i),'evidence_read',args),sid,rid)
   assert result.status!='ok'
 finally:service.close()
