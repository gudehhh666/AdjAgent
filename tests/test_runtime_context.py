"""Context is a bounded projection; tool protocol groups remain whole."""

import importlib.util
import json

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store
from adj_partner.runtime.journal import RunJournal
from adj_partner.tools.catalog import canonical_json


def context(tmp_path):
    assert importlib.util.find_spec("adj_partner.runtime.context"), "ContextView must exist"
    from adj_partner.runtime.context import ContextView
    store = Store(tmp_path / "context")
    sid = store.create_session(SessionConfig(instructions="Trusted host guidance").to_dict())["id"]
    rid = RunJournal(store).start(sid, "hello", {}, Budget().to_dict())["id"]
    return store, ContextView(store), sid, rid


def pair(call_id, content):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content}]


def test_context_is_deterministic_and_current_task_is_preserved(tmp_path):
    store, view, sid, rid = context(tmp_path)
    snapshot = {"revision": 0, "tools": [], "definition_hashes": {}}
    first = view.build(sid, rid, snapshot, max_input_tokens=20000)
    assert first == view.build(sid, rid, snapshot, max_input_tokens=20000)
    assert first["messages"][-1] == {"role": "user", "content": "hello"}
    assert "Trusted host guidance" in first["messages"][0]["content"]
    assert first["estimated_input_tokens"] > 0 and first["projection_hash"]


def test_old_history_compaction_never_splits_tool_groups(tmp_path):
    store, view, sid, rid = context(tmp_path)
    for n in range(10):
        for message in pair(str(n), "x" * 1000): store.append_message(sid, message, rid)
    store.append_message(sid, {"role": "user", "content": "latest safe boundary"}, rid)
    projection = view.build(sid, rid, {"revision": 0, "tools": []}, max_input_tokens=4000)
    assert projection["omitted_message_count"] > 0
    calls = [call["id"] for m in projection["messages"] for call in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in projection["messages"] if m["role"] == "tool"]
    assert calls == results and projection["messages"][-1]["content"] == "latest safe boundary"
    assert len(store.messages(sid)) == 22


def test_default_context_capacity_accepts_one_bounded_parallel_tool_batch(tmp_path):
    store, view, sid, rid = context(tmp_path)
    RunJournal(store).set_plan(rid, [{"step": "Use the returned source IDs", "status": "in_progress"}])
    calls = [{"id": f"call-{index}", "type": "function",
              "function": {"name": "inspect", "arguments": "{}"}} for index in range(4)]
    store.append_message(sid, {"role": "assistant", "content": "", "tool_calls": calls}, rid)
    for index in range(4):
        store.append_message(sid, {"role": "tool", "tool_call_id": f"call-{index}",
            "content": canonical_json({"status": "ok", "data": {"text": "x" * 1900}})}, rid)
    snapshot = {"revision": 0, "definition_hashes": {}, "tools": [{"type": "function",
        "function": {"name": "inspect", "description": "x" * 23000,
                     "parameters": {"type": "object", "properties": {}}}}]}

    projection = view.build(sid, rid, snapshot, max_input_tokens=100000)

    assert 32000 < projection["estimated_input_tokens"] <= 64000
    assert projection["messages"][-4:][0]["tool_call_id"] == "call-0"


@pytest.mark.parametrize("messages", [[{"role": "tool", "tool_call_id": "x", "content": "orphan"}],
    pair("a", "ok")[:1], [pair("a", "ok")[0], {"role": "user", "content": "interrupted"}, pair("a", "ok")[1]]])
def test_invalid_durable_protocol_history_fails_closed(tmp_path, messages):
    store, view, sid, rid = context(tmp_path)
    for message in messages: store.append_message(sid, message, rid)
    with pytest.raises(ValueError, match="protocol"):
        view.build(sid, rid, {"tools": []}, max_input_tokens=10000)


def test_full_tool_result_is_immutable_artifact_with_explicit_preview(tmp_path):
    store, view, sid, rid = context(tmp_path)
    result = {"status": "ok", "data": {"long": "x" * 20000}}
    message = view.tool_message(sid, rid, "call-1", result)
    envelope = json.loads(message["content"])
    assert envelope["context_truncated"] is True and envelope["full_result_artifact_id"]
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result
    assert len(message["content"]) < 6000


def test_medium_tool_result_is_also_bounded_for_parallel_batches(tmp_path):
    store, view, sid, rid = context(tmp_path)
    result = {"status": "ok", "data": {"text": "x" * 7000}}

    message = view.tool_message(sid, rid, "medium-call", result)
    envelope = json.loads(message["content"])

    assert envelope["context_truncated"] is True
    assert len(message["content"].encode("utf-8")) <= view.PREVIEW_MAX_BYTES
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result


def _handoff_tool(name):
    if name == "tools_enable":
        parameters = {"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"], "additionalProperties": False}
    elif name == "artifact_read":
        parameters = {"type": "object", "properties": {
            "artifact_id": {"type": "string"},
            "start": {"type": "integer", "minimum": 0, "maximum": 100000000},
            "max_characters": {"type": "integer", "minimum": 1, "maximum": 16000}},
            "required": ["artifact_id"], "additionalProperties": False}
    else:
        raise AssertionError(name)
    return {"type": "function", "function": {"name": name, "description": name,
        "parameters": parameters}}


def _append_truncated_tool_result(store, view, sid, rid):
    call_id = "search-result"
    summary = {"query": "public question", "results": [{"source_id": "source_public"}]}
    result = {"status": "ok", "data": {"summary": summary},
              "artifacts": [{"id": f"artifact-{n}", "name": f"source-{n}.json",
                  "media_type": "application/json", "size": 5000, "sha256": f"{n:064x}"}
                  for n in range(50)]}
    for message in [pair(call_id, "")[0], view.tool_message(sid, rid, call_id, result)]:
        store.append_message(sid, message, rid)
    return call_id, summary


def _set_tool_permissions(store, sid, *, allowed, enabled):
    session = store.get_session(sid)
    config = dict(session["config"])
    config["allowed_tools"] = list(allowed)
    config["enabled_tools"] = list(enabled)
    store.update_session(sid, config, session["revision"])


def test_truncated_result_handoff_enables_authorized_disabled_artifact_read(tmp_path):
    store, view, sid, rid = context(tmp_path)
    _set_tool_permissions(store, sid, allowed=["tools_enable", "artifact_read"], enabled=["tools_enable"])
    _call_id, summary = _append_truncated_tool_result(store, view, sid, rid)

    projection = view.build(sid, rid, {"revision": 1, "tools": [_handoff_tool("tools_enable")]},
                             max_input_tokens=20000)
    envelope = json.loads(next(message["content"] for message in projection["messages"]
                               if message["role"] == "tool"))
    handoff = envelope["artifact_read_handoff"]

    assert handoff["status"] == "enable_required"
    assert handoff["enable"] == {"tool": "tools_enable", "arguments": {"name": "artifact_read"}}
    assert handoff["read"] == {"tool": "artifact_read", "arguments": {
        "artifact_id": envelope["full_result_artifact_id"], "start": 0, "max_characters": 16000}}
    assert "next_start" in handoff["pagination"]
    assert envelope["data"]["summary"] == summary
    assert {tool["function"]["name"] for tool in projection["tools"]} == {"tools_enable"}


def test_truncated_result_handoff_reads_directly_when_artifact_read_is_enabled(tmp_path):
    store, view, sid, rid = context(tmp_path)
    _set_tool_permissions(store, sid, allowed=["tools_enable", "artifact_read"],
                          enabled=["tools_enable", "artifact_read"])
    _call_id, summary = _append_truncated_tool_result(store, view, sid, rid)

    projection = view.build(sid, rid, {"revision": 1, "tools": [
        _handoff_tool("tools_enable"), _handoff_tool("artifact_read")]}, max_input_tokens=20000)
    envelope = json.loads(next(message["content"] for message in projection["messages"]
                               if message["role"] == "tool"))
    handoff = envelope["artifact_read_handoff"]

    assert handoff["status"] == "ready"
    assert handoff["read"] == {"tool": "artifact_read", "arguments": {
        "artifact_id": envelope["full_result_artifact_id"], "start": 0, "max_characters": 16000}}
    assert "enable" not in handoff
    assert envelope["data"]["summary"] == summary


def test_truncated_result_handoff_does_not_offer_disallowed_artifact_read(tmp_path):
    store, view, sid, rid = context(tmp_path)
    _set_tool_permissions(store, sid, allowed=["tools_enable"], enabled=["tools_enable"])
    _append_truncated_tool_result(store, view, sid, rid)

    projection = view.build(sid, rid, {"revision": 1, "tools": [_handoff_tool("tools_enable")]},
                             max_input_tokens=20000)
    envelope = json.loads(next(message["content"] for message in projection["messages"]
                               if message["role"] == "tool"))
    handoff = envelope["artifact_read_handoff"]

    assert handoff["status"] == "unavailable"
    assert "not authorized" in handoff["reason"]
    assert "enable" not in handoff and "read" not in handoff
    assert "use artifact_read" not in envelope["note"].lower()


def test_truncated_result_handoff_stays_unavailable_without_enabled_tool_management(tmp_path):
    store, view, sid, rid = context(tmp_path)
    _set_tool_permissions(store, sid, allowed=["artifact_read"], enabled=[])
    _append_truncated_tool_result(store, view, sid, rid)

    projection = view.build(sid, rid, {"revision": 1, "tools": []}, max_input_tokens=20000)
    envelope = json.loads(next(message["content"] for message in projection["messages"]
                               if message["role"] == "tool"))

    assert envelope["artifact_read_handoff"]["status"] == "unavailable"
    assert "tools_enable is not currently available" in envelope["artifact_read_handoff"]["reason"]


def test_large_scientific_result_prioritizes_structured_summary_over_artifact_provenance(tmp_path):
    store, view, sid, rid = context(tmp_path)
    summary = {"ranking_metric": "centered_reversal_r", "top_centered": [
        {"compound": "Zincgluconate", "centered_reversal_r": 0.3567, "perm_q": 0.00472},
        {"compound": "Saxagliptin", "centered_reversal_r": 0.3485, "perm_q": 0.00472},
    ], "limitations": ["exploratory only"]}
    result = {"status": "ok", "data": {"source_id": "source_1", "summary": summary,
        "source": {"provenance": {"inputs": [{"path": str(n), "sha256": "a" * 64} for n in range(60)]}}},
        "artifacts": [{"id": "artifact_result", "name": "result.json",
            "provenance": {"inputs": [{"path": str(n), "sha256": "b" * 64} for n in range(60)]}}]}
    message = view.tool_message(sid, rid, "deg-call", result)
    envelope = json.loads(message["content"])
    assert envelope["context_truncated"] is True
    assert envelope["data"] == {"source_id": "source_1", "summary": summary}
    assert "Zincgluconate" in message["content"]
    assert "provenance" not in message["content"]
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result
    assert len(message["content"].encode("utf-8")) <= 8000


def test_large_result_preview_bounds_artifact_metadata_without_losing_full_result(tmp_path):
    store, view, sid, rid = context(tmp_path)
    artifacts = [{"id": f"artifact_{index}", "name": f"source-{index}.json",
        "media_type": "application/json", "size": 100 + index,
        "sha256": f"{index:064x}"} for index in range(100)]
    result = {"status": "ok", "data": {"summary": {"results": [
        {"source_id": f"source_{index}", "title": "x" * 100} for index in range(3)]}},
        "artifacts": artifacts}

    message = view.tool_message(sid, rid, "batch-call", result)
    envelope = json.loads(message["content"])

    assert envelope["artifact_count"] == 100
    assert envelope["artifacts_omitted"] == 99
    assert envelope["artifacts"] == artifacts[:1]
    assert len(message["content"].encode("utf-8")) <= 2400
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result


def test_oversize_structured_summary_is_compacted_but_remains_structured(tmp_path):
    store, view, sid, rid = context(tmp_path)
    summary = {"provider": "fixture", "query": "research query", "results": [
        {"source_id": f"source_{index}", "title": "title " + "x" * 400,
         "abstract_preview": "abstract " + "y" * 800} for index in range(3)]}
    result = {"status": "ok", "data": {"summary": summary},
        "artifacts": [{"id": f"artifact_{index}", "name": "result.json",
                       "media_type": "application/json", "size": 9999,
                       "sha256": f"{index:064x}"} for index in range(50)]}

    message = view.tool_message(sid, rid, "summary-call", result)
    envelope = json.loads(message["content"])

    assert envelope["summary_compacted"] is True
    assert isinstance(envelope["data"]["summary"], dict)
    assert len(envelope["data"]["summary"]["results"]) <= 2
    assert len(message["content"].encode("utf-8")) <= 2400
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result


def test_known_secrets_redacted_before_all_request_projection(tmp_path):
    store, view, sid, rid = context(tmp_path)
    store.register_secrets(["opaque-secret"])
    snapshot = {"revision": 0, "tools": [{"type": "function", "function": {"name": "safe",
        "description": "opaque-secret", "parameters": {"type": "object"}}}]}
    store.append_message(sid, {"role": "user", "content": "opaque-secret"}, rid)
    assert "opaque-secret" not in json.dumps(view.build(sid, rid, snapshot, max_input_tokens=10000))


def test_public_plan_and_retrieved_text_do_not_become_system_instructions(tmp_path):
    store, view, sid, rid = context(tmp_path)
    RunJournal(store).set_plan(rid, [{"step": "UNTRUSTED_PLAN", "status": "pending"}])
    for message in pair("a", "UNTRUSTED_RESULT"): store.append_message(sid, message, rid)
    messages = view.build(sid, rid, {"tools": []}, max_input_tokens=10000)["messages"]
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    assert "UNTRUSTED_PLAN" not in system and "UNTRUSTED_RESULT" not in system
    assert "UNTRUSTED_PLAN" in json.dumps(messages)


def test_ordinary_task_is_told_to_answer_without_research_record_workflow(tmp_path):
    store, view, sid, rid = context(tmp_path)
    messages = view.build(sid, rid, {"tools": []}, max_input_tokens=10000)["messages"]
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    assert "strict_acceptance is false" in system
    assert "answer directly" in system
    assert "research-record tools are not projected" in system


def test_research_record_tools_are_projected_only_for_strict_tasks(tmp_path):
    store, view, sid, rid = context(tmp_path)
    def tool(name):
        return {"type": "function", "function": {"name": name, "description": name,
            "parameters": {"type": "object", "properties": {}}}}
    snapshot = {"revision": 0, "tools": [tool("deg_analyze"), tool("research_state"),
        tool("evidence_select"), tool("claim_record"), tool("report_create")]}
    ordinary = view.build(sid, rid, snapshot, max_input_tokens=20000)
    ordinary_names = {item["function"]["name"] for item in ordinary["tools"]}
    assert ordinary_names == {"deg_analyze"}

    task = store.get_task(store.get_run(rid)["task_id"])
    store.update_task(task["id"], {"acceptance": {"require_report": True}}, task["revision"])
    strict = view.build(sid, rid, snapshot, max_input_tokens=20000)
    assert {item["function"]["name"] for item in strict["tools"]} == {
        "deg_analyze", "research_state", "evidence_select", "claim_record", "report_create"}


def test_too_large_current_goal_or_tool_schema_does_not_silently_truncate(tmp_path):
    store, view, sid, rid = context(tmp_path)
    from adj_partner.runtime.context import ContextOverflow
    store.append_message(sid, {"role": "user", "content": "x" * 50000}, rid)
    with pytest.raises(ContextOverflow): view.build(sid, rid, {"tools": []}, max_input_tokens=1000)


def test_oversized_latest_tool_result_is_compacted_rather_than_overflowing(tmp_path):
    store, view, sid, rid = context(tmp_path)
    call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    store.append_message(sid, {"role": "assistant", "content": "", "tool_calls": [call]}, rid)
    huge = json.dumps({"status": "ok", "data": {"rows": [{"text": "y" * 1000} for _ in range(30)]}})
    store.append_message(sid, {"role": "tool", "tool_call_id": "call_1", "content": huge}, rid)
    built = view.build(sid, rid, {"tools": []}, max_input_tokens=6000)
    assert built["latest_group_compaction_level"] >= 1 and built["estimated_input_tokens"] <= 6000
    tool = next(m for m in built["messages"] if m["role"] == "tool")
    assert json.loads(tool["content"])["compacted_for_context"] is True
    roomy = view.build(sid, rid, {"tools": []}, max_input_tokens=60000)
    assert "latest_group_compaction_level" not in roomy


def test_cross_session_projection_is_rejected(tmp_path):
    store, view, sid, rid = context(tmp_path)
    other = store.create_session(SessionConfig().to_dict())["id"]
    with pytest.raises(ValueError): view.build(other, rid, {"tools": []}, max_input_tokens=10000)


def test_oversize_error_envelope_is_bounded_without_losing_full_error(tmp_path):
    store, view, sid, rid = context(tmp_path)
    result = {"status": "error", "error": {"code": "huge", "message": "x" * 20000, "details": "z" * 20000}}
    message = view.tool_message(sid, rid, "call", result)
    assert len(message["content"]) < 6000
    envelope = json.loads(message["content"])
    assert json.loads(store.read_artifact(envelope["full_result_artifact_id"])) == result


@pytest.mark.parametrize("field,cutoff", [("data", 400), ("error", 600)])
def test_late_secret_reprojection_precedes_preview_truncation(tmp_path, field, cutoff):
    from adj_partner.tools.catalog import canonical_json
    store, view, sid, rid = context(tmp_path)
    secret = "LATE_PRIVATE_VALUE_" + "abcdefghijk" * 6
    if field == "data":
        prefix = len(canonical_json({"data": {"long": ""}, "status": "ok"}).split('"long":"')[0]) + len('"long":"')
        result = {"status": "ok", "data": {"long": "x" * (cutoff - prefix - len(secret) + 1) + secret + "z" * 10000}}
    else:
        result = {"status": "error", "error": {"code": "large", "message": "x" * (cutoff - len(secret) + 1) + secret + "z" * 10000}}
    message = view.tool_message(sid, rid, "a", result)
    assert secret[:-1] in message["content"]
    for item in [pair("a", "")[0], message]: store.append_message(sid, item, rid)
    store.register_secrets([secret])
    first = view.build(sid, rid, {"tools": []}, max_input_tokens=20000)
    assert secret[:-1] not in json.dumps(first)
    projected = json.loads(next(m["content"] for m in first["messages"] if m["role"] == "tool"))
    assert "preview" in projected and not projected.get("preview_unavailable")
    before = store.artifacts(sid)
    assert first == view.build(sid, rid, {"tools": []}, max_input_tokens=20000)
    assert store.artifacts(sid) == before


def test_cross_session_full_result_is_not_read(tmp_path, monkeypatch):
    store, view, sid, rid = context(tmp_path)
    other = store.create_session(SessionConfig().to_dict())["id"]
    other_run = RunJournal(store).start(other, "other", {}, Budget().to_dict())["id"]
    message = view.tool_message(other, other_run, "a", {"status": "ok", "data": "FOREIGN" + "x" * 10000})
    for item in [pair("a", "")[0], message]: store.append_message(sid, item, rid)
    def forbidden(*args, **kwargs): pytest.fail("Cross-session artifact must not be read")
    monkeypatch.setattr(store, "read_artifact", forbidden)
    result = view.build(sid, rid, {"tools": []}, max_input_tokens=20000)
    assert "FOREIGN" not in json.dumps(result) and "unavailable" in json.dumps(result)


def test_bounded_artifact_read_rejects_before_unbounded_allocation(tmp_path):
    store, _, sid, rid = context(tmp_path)
    artifact = store.put_artifact("x" * 10000, "payload.txt", session_id=sid, run_id=rid)
    with pytest.raises(ValueError, match="read limit"): store.read_artifact(artifact["id"], max_bytes=100)
    assert len(store.read_artifact(artifact["id"], max_bytes=10000)) == 10000


def test_missing_full_result_never_reuses_stale_preview(tmp_path, monkeypatch):
    store, view, sid, rid = context(tmp_path)
    message = view.tool_message(sid, rid, "a", {"status": "ok", "data": "STALE_PREVIEW" + "x" * 10000})
    for item in [pair("a", "")[0], message]: store.append_message(sid, item, rid)
    def unavailable(*args, **kwargs): raise ValueError("corrupt artifact")
    monkeypatch.setattr(store, "read_artifact", unavailable)
    result = view.build(sid, rid, {"tools": []}, max_input_tokens=20000)
    assert "STALE_PREVIEW" not in json.dumps(result) and "unavailable" in json.dumps(result)


def test_compaction_serialization_work_is_linear_in_history(tmp_path, monkeypatch):
    import adj_partner.runtime.context as module
    store, view, sid, rid = context(tmp_path)
    history = [{"role": "user", "content": str(n) + "x" * 2000} for n in range(512)]
    monkeypatch.setattr(store, "messages", lambda _sid: history)
    original, work = module.canonical_json, [0]
    def measured(value):
        result = original(value)
        work[0] += len(result)
        return result
    monkeypatch.setattr(module, "canonical_json", measured)
    projection = view.build(sid, rid, {"tools": []}, max_input_tokens=8000)
    assert projection["omitted_message_count"] > 500
    assert projection["omitted_history"] == view._omitted(history[:projection["omitted_message_count"]])
    assert work[0] < len(original(history)) * 10


def test_full_preview_refresh_has_aggregate_io_budget_and_prioritizes_latest(tmp_path, monkeypatch):
    store, view, sid, rid = context(tmp_path)
    for n in range(12):
        message = view.tool_message(sid, rid, str(n), {"status": "ok", "data": "x" * 1000000})
        for item in [pair(str(n), "")[0], message]: store.append_message(sid, item, rid)
    original, counts = store.read_artifact, [0, 0]
    def measured(*args, **kwargs):
        content = original(*args, **kwargs)
        counts[0] += 1
        counts[1] += len(content)
        return content
    monkeypatch.setattr(store, "read_artifact", measured)
    projection = view.build(sid, rid, {"tools": []}, max_input_tokens=8000)
    assert counts[1] <= 8 * 1024 * 1024 and counts[0] <= 128
    assert "preview" in json.loads(projection["messages"][-1]["content"])
