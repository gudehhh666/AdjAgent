"""End-to-end offline decisions with real jobs, sources, evidence and report gate."""

import json
import time

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.research import ResearchStore
from adj_partner.research.tools import register_research_tools
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog
from test_runtime_engine import Decisions, reply


def scientific_engine(tmp_path, decisions):
    store, catalog = Store(tmp_path / "science"), ToolCatalog()
    research = ResearchStore(store)
    native = NativeCapabilities(store, research=research)
    native.register(catalog)
    register_research_tools(catalog, research)
    port = Decisions(decisions)
    engine = AgentEngine(store, catalog, port, research=research,
                         resolve_job=native.resolve_job, recover_action=native.recover_action)
    sid = store.create_session(SessionConfig(enabled_tools=catalog.names(), allowed_tools=catalog.names(),
                                              budget=Budget(max_steps=12)).to_dict())["id"]
    return store, engine, sid, port, native


def last_data(messages):
    message = next(item for item in reversed(messages) if item["role"] == "tool")
    return json.loads(message["content"])["data"]


def test_strict_protein_sequence_runs_real_worker_and_delivers_exact_report_without_final_rewrite(tmp_path):
    def evidence(messages, tools, **limits):
        return reply(calls=[ToolCall("evidence", "evidence_select", {"source_id": last_data(messages)["source_id"], "selector": {"pointer": "/length"}})])
    def observe(messages, tools, **limits):
        return reply(calls=[ToolCall("observe", "claim_observe", {"evidence_id": last_data(messages)["id"]})])
    def report(messages, tools, **limits):
        return reply(calls=[ToolCall("report", "report_create", {"title": "Sequence inspection", "sections": [
            {"heading": "Findings", "claim_ids": [last_data(messages)["id"]]}], "limitations": ["Theoretical sequence properties, no function or efficacy inference."]})])
    def validate(messages, tools, **limits):
        return reply(calls=[ToolCall("validate", "report_validate", {"report_id": last_data(messages)["id"]})])
    store, engine, sid, port, native = scientific_engine(tmp_path, [reply(calls=[
        ToolCall("protein", "protein_sequence", {"sequence": "MKWVTFISLLFLFSSAYS"})]), evidence, observe, report, validate])
    rid = engine.start(sid, "Compute actual length, qualify limits", acceptance={"require_report": True,
        "require_evidence": True, "require_data_artifact": True, "require_limitations": True})["id"]
    waiting = engine.resume(rid)
    assert waiting["status"] == "waiting" and waiting["state"]["waiting"]["kind"] == "job"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        done = engine.resume(rid)
        if done["status"] != "waiting": break
        time.sleep(.05)
    assert done["status"] == "succeeded", done
    delivery = done["state"]["delivery"]
    assert delivery["content"].encode() == store.read_artifact(delivery["artifact_id"])
    assert delivery["kind"] == "accepted_report" and len(port.requests) == 5
    assert len(native.jobs.list(sid)) == 1 and len(engine.research.list_sources(sid, rid)) == 1
    source = engine.research.list_sources(sid, rid)[0]
    assert json.loads(store.read_artifact(source["artifact_id"]))["length"] == 18
    assert store.messages(sid)[-1]["content"] == delivery["content"]


def test_completed_job_survives_crash_before_outer_action_result_is_recorded(tmp_path):
    store, engine, sid, port, native = scientific_engine(tmp_path, [reply(calls=[
        ToolCall("protein", "protein_sequence", {"sequence": "AAAA"})]), reply("done")])
    rid = engine.start(sid, "Inspect")["id"]
    original = store.finish_action
    def crash(run_id, call_id, result):
        if result.get("status") == "pending": raise KeyboardInterrupt("crash after actual job launch")
        return original(run_id, call_id, result)
    store.finish_action = crash
    with pytest.raises(KeyboardInterrupt): engine.resume(rid)
    assert len(native.jobs.list(sid)) == 1
    store.finish_action = original
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        done = engine.resume(rid)
        if done["status"] != "waiting": break
        time.sleep(.05)
    assert done["status"] == "succeeded" and len(native.jobs.list(sid)) == 1 and len(port.requests) == 2


def test_budget_counters_survive_real_job_suspension_and_reopen(tmp_path):
    store, engine, sid, port, native = scientific_engine(tmp_path, [reply(calls=[
        ToolCall("protein", "protein_sequence", {"sequence": "AAAA"})]), reply("done")])
    rid = engine.start(sid, "Inspect")["id"]
    assert engine.resume(rid)["status"] == "waiting"
    before = store.get_run(rid)["usage"]
    reopened_store = Store(store.root)
    reopened_native = NativeCapabilities(reopened_store)
    reopened = AgentEngine(reopened_store, engine.catalog, port,
        resolve_job=reopened_native.resolve_job, recover_action=reopened_native.recover_action)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        done = reopened.resume(rid)
        if done["status"] != "waiting": break
        time.sleep(.05)
    assert done["status"] == "succeeded" and done["usage"]["input_tokens"] == before["input_tokens"] + 50
