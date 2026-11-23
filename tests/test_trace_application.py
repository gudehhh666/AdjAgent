"""Consistent authoritative traces and explicit immutable artifact privacy."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import pytest

from adj_partner.contracts import ModelReply
from adj_partner.models import ScriptedPort
from test_application import PROJECT, service_type


@pytest.fixture(autouse=True)
def no_ambient_provider_credentials(monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def make(tmp_path):
    service = service_type()(tmp_path / "state", model=ScriptedPort([ModelReply(content="done")]), project_root=PROJECT)
    sid = service.create_session()["id"]
    rid = service.submit(sid, "trace subject")["id"]
    return service, sid, rid


def load(result):
    return json.loads(Path(result["manifest"]).read_text()), [json.loads(line) for line in Path(result["jsonl"]).read_text().splitlines()]


def test_trace_snapshot_contains_authoritative_extensions_and_manifest_hashes(tmp_path):
    service, sid, rid = make(tmp_path)
    service.journal.set_plan(rid, [{"step": "Public step", "status": "pending"}])
    service.journal.ask(rid, "ask", "Actual pending question")
    claim = service.research.record_claim(sid, rid, text="Tentative proposal", kind="proposal", evidence_ids=[])
    exported = service.trace(sid, tmp_path / "export", rid)
    manifest, records = load(exported)
    assert Path(exported["manifest"]).name == "manifest.json"
    assert manifest["snapshot_consistency"] == "single_sql_read_transaction"
    kinds = {record["record_type"] for record in records}
    assert {"session", "task", "run", "message", "plan", "input", "research_record", "research_event", "model_preference"} <= kinds
    assert claim["id"] in Path(exported["jsonl"]).read_text()
    for item in manifest["files"].values():
        content = (Path(exported["manifest"]).parent / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"] and len(content) == item["size"]


def test_public_unchanged_artifact_copied_private_raw_binary_and_secret_omitted(tmp_path):
    service, sid, rid = make(tmp_path)
    def artifact(content, name, media="text/plain", provenance=None):
        return service.store.put_artifact(content, name, media, provenance or {}, session_id=sid, run_id=rid)
    public = artifact("Public exact 中文", "public.txt", provenance={"visibility": "public"})
    raw = artifact("Raw private user data", "sample.txt", provenance={"text_availability": "userfile", "visibility": "public"})
    credential = artifact("Credential data", ".env", provenance={"visibility": "public"})
    binary = artifact(b"\xff\x00", "file.bin", "application/octet-stream", {"visibility": "public"})
    secret = artifact("late-known-secret", "formerly-public.txt", provenance={"visibility": "public"})
    private = artifact("Unclassified contents", "private.txt")
    service.store.register_secrets(["late-known-secret"])
    result = service.trace(sid, tmp_path / "export", rid)
    manifest, _ = load(result)
    by_id = {item["id"]: item for item in manifest["artifacts"]}
    assert by_id[public["id"]]["export_status"] == "copied"
    assert (Path(result["manifest"]).parent / by_id[public["id"]]["export_path"]).read_bytes() == b"Public exact \xe4\xb8\xad\xe6\x96\x87"
    for item in (raw, credential, binary, secret, private):
        assert by_id[item["id"]]["export_status"] == "omitted" and by_id[item["id"]]["reason"]
    assert service.store.read_artifact(secret["id"]) == b"late-known-secret"
    assert "late-known-secret" not in Path(result["jsonl"]).read_text()


@pytest.mark.parametrize("corruption", ["hash", "symlink", "size"])
def test_corrupt_or_oversize_artifact_is_omitted_not_copied(tmp_path, corruption):
    service, sid, rid = make(tmp_path)
    item = service.store.put_artifact("safe", "public.txt", "text/plain", provenance={"visibility": "public"}, session_id=sid, run_id=rid)
    path = Path(item["path"])
    if corruption == "hash": path.write_text("evil")
    elif corruption == "symlink":
        other = tmp_path / "other.txt"; other.write_text("safe")
        path.unlink(); path.symlink_to(other)
    else:
        with service.store.transaction(write=True) as conn: conn.execute("UPDATE artifacts SET size=? WHERE id=?", (64 * 1024 * 1024, item["id"]))
    manifest, _ = load(service.trace(sid, tmp_path / "export", rid))
    target = next(a for a in manifest["artifacts"] if a["id"] == item["id"])
    assert target["export_status"] == "omitted" and "export_path" not in target
    assert target["reason"] == ("artifact_size_limit" if corruption == "size" else "integrity_encoding_or_read_failure")


def test_trace_snapshot_is_one_sql_read_transaction_and_does_not_call_separate_snapshots(tmp_path, monkeypatch):
    service, sid, rid = make(tmp_path)
    original, transactions = service.store.transaction, []
    @contextmanager
    def measured(*args, **kwargs):
        transactions.append(kwargs.get("write", False))
        with original(*args, **kwargs) as conn: yield conn
    monkeypatch.setattr(service.store, "transaction", measured)
    monkeypatch.setattr(service.research, "snapshot", lambda *a: pytest.fail("Not a shared SQL snapshot"))
    monkeypatch.setattr(service.native.jobs, "list", lambda *a: pytest.fail("Not a shared SQL snapshot"))
    service.trace(sid, tmp_path / "export", rid)
    assert transactions == [False]


def test_run_scoped_trace_keeps_referenced_prior_material_not_other_session(tmp_path):
    service, sid, rid = make(tmp_path)
    artifact = service.store.put_artifact('{"value":3}', "fixture.json", "application/json", {"kind": "computed"}, session_id=sid, run_id=rid)
    source = service.research.register_source(sid, rid, artifact_id=artifact["id"], kind="computed", title="Synthetic fixture", provenance={"kind":"computed"}, idempotency_key="fixture")
    evidence = service.research.select_evidence(sid, rid, source["id"], {"pointer": "/value"})
    service.cancel(rid)
    second = service.submit(sid, "Reuse source")["id"]
    service.research.observe_evidence(sid, second, evidence["id"])
    other = service.create_session()["id"]
    foreign = service.store.put_artifact("FOREIGN", "foreign.txt", provenance={"visibility":"public"}, session_id=other)
    result = service.trace(sid, tmp_path / "export", second)
    manifest, records = load(result)
    assert artifact["id"] in {a["id"] for a in manifest["artifacts"]}
    assert foreign["id"] not in json.dumps(records) + json.dumps(manifest)
    with pytest.raises((ValueError, PermissionError)): service.trace(other, tmp_path / "bad", second)


def test_destinations_preserve_prior_exports_and_reject_aliases(tmp_path):
    service, sid, rid = make(tmp_path)
    first = service.trace(sid, tmp_path / "export", rid)
    original = Path(first["manifest"]).read_bytes()
    second = service.trace(sid, tmp_path / "export", rid)
    assert first["manifest"] != second["manifest"] and Path(first["manifest"]).read_bytes() == original
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path / "export", target_is_directory=True)
    with pytest.raises((ValueError, PermissionError)): service.trace(sid, alias, rid)


@pytest.mark.parametrize("field", ["data", "error"])
def test_late_secret_fragments_in_nested_messages_events_and_requests_are_omitted(tmp_path, field):
    from adj_partner.runtime.context import ContextView
    from adj_partner.tools.catalog import canonical_json
    service, sid, rid = make(tmp_path)
    secret = "TRACE_LATE_PRIVATE_" + "abcdefghijk" * 6
    if field == "data":
        prefix = canonical_json({"status":"ok","data":{"long":"MARK"}}).index("MARK")
        # Keep a late fragment inside the bounded structured string preview;
        # the complete result remains much larger in its immutable artifact.
        result = {"status":"ok","data":{"long":"x"*(400-prefix-len(secret)+1)+secret+"z"*10000}}
    else:
        result = {"status":"error","error":{"code":"large","message":"x"*(600-len(secret)+1)+secret+"z"*10000}}
    message = ContextView(service.store).tool_message(sid,rid,"call",result)
    assert secret[:-1] in message["content"]
    service.store.append_message(sid,message,rid)
    service.store.append_event(sid,"nested.fixture",{"nested":canonical_json({"messages":[message]})},rid)
    request = service.store.put_artifact(canonical_json({"messages":[message]}),"model-request.json","application/json",
        {"kind":"public_model_request"},session_id=sid,run_id=rid)
    original = service.store.read_artifact(request["id"])
    service.store.register_secrets([secret])
    exported = service.trace(sid,tmp_path/"export",rid)
    manifest, _ = load(exported)
    for path in (exported["jsonl"],exported["markdown"],exported["manifest"]):
        assert secret[:-1] not in Path(path).read_text()
    item = next(a for a in manifest["artifacts"] if a["id"] == request["id"])
    assert item["export_status"] == "omitted" and "preview" in item["reason"]
    assert "preview_omitted_for_export" in Path(exported["jsonl"]).read_text()
    assert service.store.read_artifact(request["id"]) == original


def test_raw_source_kind_vetoes_public_artifact_marker(tmp_path):
    service,sid,rid = make(tmp_path)
    artifact = service.store.put_artifact("User private file", "sample.txt", "text/plain", {"visibility":"public"},session_id=sid,run_id=rid)
    service.research.register_source(sid,rid,artifact_id=artifact["id"],kind="userfile",title="Synthetic user file",
        provenance={"text_availability":"userfile"},idempotency_key="fixture-userfile")
    manifest,_ = load(service.trace(sid,tmp_path/"export",rid))
    assert next(item for item in manifest["artifacts"] if item["id"]==artifact["id"])["export_status"]=="omitted"


def test_json_bom_cannot_hide_credential_shaped_content_from_export(tmp_path):
    service,sid,rid = make(tmp_path)
    artifact = service.store.put_artifact(b'\xef\xbb\xbf{"api_key":"synthetic-unknown-key"}',"public.json","application/json",
        {"visibility":"public"},session_id=sid,run_id=rid)
    manifest,_ = load(service.trace(sid,tmp_path/"export",rid))
    assert next(item for item in manifest["artifacts"] if item["id"]==artifact["id"])["export_status"]=="omitted"


def test_aggregate_export_read_budget_counts_omitted_secret_artifacts(tmp_path, monkeypatch):
    from adj_partner import trace_export
    service,sid,rid = make(tmp_path)
    for index in range(5):
        service.store.put_artifact("late-private"+"x"*700,f"public-{index}.txt","text/plain",provenance={"visibility":"public"},session_id=sid,run_id=rid)
    service.store.register_secrets(["late-private"])
    monkeypatch.setattr(trace_export,"MAX_EXPORT_BYTES",1024)
    original, sizes = service.store.read_artifact, []
    def measured(*args,**kwargs):
        content = original(*args,**kwargs); sizes.append(len(content)); return content
    monkeypatch.setattr(service.store,"read_artifact",measured)
    service.trace(sid,tmp_path/"export",rid)
    assert 0 < sum(sizes) <= 1024


def test_single_snapshot_resists_concurrent_task_update(tmp_path, monkeypatch):
    from adj_partner.storage import Store
    service,sid,rid = make(tmp_path)
    tid = service.run(rid)["task_id"]
    original, changed = service.store.transaction, []
    class Connection:
        def __init__(self, conn): self.conn=conn
        def execute(self, sql, *args):
            if "SELECT * FROM tasks" in sql and not changed:
                Store(service.store.root).update_task(tid,{"goal":"Concurrent new goal"})
                changed.append(True)
            return self.conn.execute(sql,*args)
    @contextmanager
    def concurrent(*args,**kwargs):
        with original(*args,**kwargs) as conn: yield Connection(conn)
    monkeypatch.setattr(service.store,"transaction",concurrent)
    _,records = load(service.trace(sid,tmp_path/"export",rid))
    task = next(row["data"] for row in records if row["record_type"]=="task")
    assert task["goal"]=="trace subject" and service.store.get_task(tid)["goal"]=="Concurrent new goal"


def test_real_native_job_journal_and_computed_bytes_survive_missing_common_mirror(tmp_path):
    from adj_partner.contracts import ToolCall
    service = service_type()(tmp_path/"state",project_root=PROJECT,model=ScriptedPort([
        ModelReply(tool_calls=[ToolCall("p","protein_sequence",{"sequence":"AAAA"})],finish_reason="tool_calls",usage={"input_tokens":1,"output_tokens":1}),
        ModelReply(content="done",usage={"input_tokens":1,"output_tokens":1})]))
    sid = service.create_session(profile="protein")["id"]
    rid = service.submit(sid,"Actual computation")["id"]
    assert service.drive(rid)["status"]=="succeeded"
    # Simulate loss of only the duplicate common mirror in this temporary fixture.
    with service.store.transaction(write=True) as conn:
        conn.execute("DELETE FROM events WHERE session_id=? AND kind LIKE 'job.%'",(sid,))
    manifest,records = load(service.trace(sid,tmp_path/"export",rid))
    assert any(row["record_type"]=="job_event" and row["data"]["kind"]=="job.succeeded" for row in records)
    source = service.research.list_sources(sid,rid)[0]
    assert next(item for item in manifest["artifacts"] if item["id"]==source["artifact_id"])["export_status"]=="copied"


def test_untrusted_malformed_preview_envelope_cannot_crash_or_smuggle_nested_fragments(tmp_path):
    service,sid,rid = make(tmp_path)
    message = {"context_truncated":True,"full_result_artifact_id":{"preview":"HIDDEN_FRAGMENT"},"status":{},"sha256":[]}
    service.store.append_message(sid,{"role":"user","content":json.dumps(message)},rid)
    exported = service.trace(sid,tmp_path/"export",rid)
    assert "HIDDEN_FRAGMENT" not in Path(exported["jsonl"]).read_text()


def test_bom_prefixed_json_string_preview_is_projected(tmp_path):
    service,sid,rid = make(tmp_path)
    message = {"context_truncated":True,"full_result_artifact_id":"artifact_synthetic", "preview":"OLD_PRIVATE_FRAGMENT"}
    service.store.append_event(sid,"fixture",{"json_string":"\ufeff"+json.dumps(message)},rid)
    exported = service.trace(sid,tmp_path/"export",rid)
    assert "OLD_PRIVATE_FRAGMENT" not in Path(exported["jsonl"]).read_text()


def test_relative_export_destination_anchors_to_service_project_not_cwd_or_store(tmp_path,monkeypatch):
    project = tmp_path/"source"; project.mkdir()
    other = tmp_path/"elsewhere"; other.mkdir()
    service = service_type()(tmp_path/"state",project_root=project,model=ScriptedPort([]))
    sid = service.create_session()["id"]
    monkeypatch.chdir(other)
    exported = service.trace(sid,"exports")
    assert Path(exported["manifest"]).is_relative_to(project/"exports")


@pytest.mark.parametrize("provenance", [["not-an-object"], {"kind":[]}, {"job_id":{}},
    {"provider":"Europe PMC","text_availability":[]}])
def test_malformed_artifact_classification_is_omitted_without_aborting_trace(tmp_path, provenance):
    service,sid,rid = make(tmp_path)
    artifact = service.store.put_artifact("Unclassified bytes", "sample.txt", "text/plain", provenance,
        session_id=sid,run_id=rid)
    manifest,_ = load(service.trace(sid,tmp_path/"export",rid))
    item = next(row for row in manifest["artifacts"] if row["id"]==artifact["id"])
    assert item["export_status"]=="omitted" and item["reason"]=="malformed_artifact_classification"
