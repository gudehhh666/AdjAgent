"""Native tools through production executor/jobs/research; only HTTP is a fixture."""

import importlib
import importlib.util
import json
from pathlib import Path
import time

import httpx
import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.native.literature import EuropePMC


PROJECT = Path(__file__).resolve().parents[1]


def native_type():
    assert importlib.util.find_spec("adj_partner.native.capabilities"), "shared native registration/service must exist"
    return importlib.import_module("adj_partner.native.capabilities").NativeCapabilities


def setup(tmp_path, *, network=False, literature=None):
    store = Store(tmp_path / "state")
    research = ResearchStore(store)
    native = native_type()(store, research=research, literature=literature)
    catalog = ToolCatalog()
    native.register(catalog)
    session = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)], allow_network=network,
                    allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(session, "Actual native fixture", {"require_data_artifact": True})["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return store, research, native, ToolExecutor(store, catalog), session, run


def call(executor, sid, rid, name, arguments, call_id="call1"):
    return executor.execute(ToolCall(call_id, name, arguments), sid, rid)


def complete(native, sid, rid, job_id, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = native.resolve_job(sid, rid, job_id)
        if result.status not in {"pending", "running", "waiting"}:
            return result
        time.sleep(0.04)
    pytest.fail("Native job did not finish within fixture timeout")


def test_real_protein_job_imports_actual_artifact_once_and_recovers_call_identity(tmp_path):
    store, research, native, executor, sid, rid = setup(tmp_path)
    arguments = {"sequence": "MKWVTFISLLFLFSSAYS"}
    pending = call(executor, sid, rid, "protein_sequence", arguments)
    assert pending.status == "pending" and pending.job_id
    result = complete(native, sid, rid, pending.job_id)
    assert result.status == "ok", result.to_dict()
    source = research.get_source(sid, rid, result.data["source_id"])
    output = json.loads(store.read_artifact(source["artifact_id"]))
    assert output["length"] == 18 and output["molecular_weight_da"] == pytest.approx(2140.5405)
    assert source["kind"] == "computed" and source["provenance"]["job_id"] == pending.job_id
    assert source["provenance"]["code_hashes"] and source["provenance"]["limitations"]
    reopened = native_type()(Store(store.root))
    recovered = reopened.recover_action(sid, rid, ToolCall("call1", "protein_sequence", arguments))
    assert recovered.data["source_id"] == result.data["source_id"]
    assert reopened.resolve_job(sid, rid, pending.job_id).data["source_id"] == result.data["source_id"]
    assert len(research.list_sources(sid, rid)) == 1
    assert call(executor, sid, rid, "protein_sequence", arguments).job_id == pending.job_id
    with pytest.raises(ValueError):
        reopened.recover_action(sid, rid, ToolCall("call1", "protein_sequence", {"sequence": "AAAA"}))


def test_failed_job_and_unconfigured_deg_do_not_become_sources(tmp_path):
    _, research, native, executor, sid, rid = setup(tmp_path)
    pending = call(executor, sid, rid, "protein_sequence", {"sequence": "AX"})
    result = complete(native, sid, rid, pending.job_id)
    assert result.status == "error" and result.error
    missing = call(executor, sid, rid, "deg_analyze", {}, "missing_deg")
    assert missing.status == "error" and missing.error["code"] == "dataset_unconfigured"
    assert research.list_sources(sid, rid) == []


@pytest.mark.parametrize("tool,fixture,expected", [("protein_fasta", "protein_example.fasta", "records"),
                                                  ("protein_structure", "structure_example.pdb", "atoms")])
def test_real_file_protein_jobs_preserve_actual_input_hashes(tmp_path, tool, fixture, expected):
    path = tmp_path / fixture
    path.write_bytes((PROJECT / "tests/fixtures" / fixture).read_bytes())
    store, research, native, executor, sid, rid = setup(tmp_path)
    pending = call(executor, sid, rid, tool, {"path": str(path)})
    result = complete(native, sid, rid, pending.job_id)
    assert result.status == "ok", result.to_dict()
    source = research.get_source(sid, rid, result.data["source_id"])
    assert source["provenance"]["inputs"][0]["path"] == str(path)
    data = json.loads(store.read_artifact(source["artifact_id"]))
    assert expected in data


def test_file_inspection_and_scoped_artifact_reads_preserve_userfile_authority(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"n":1,"api_key":"synthetic-sensitive-value"}')
    store, research, native, executor, sid, rid = setup(tmp_path)
    result = call(executor, sid, rid, "file_inspect", {"path": str(path)})
    assert result.status == "ok", result.to_dict()
    source = research.get_source(sid, rid, result.data["source_id"])
    assert source["kind"] == "userfile" and source["provenance"]["text_availability"] == "userfile"
    evidence = research.select_evidence(sid, rid, source["id"], {"pointer": "/n"})
    assert evidence["value"] == 1
    read = call(executor, sid, rid, "artifact_read", {"artifact_id": source["artifact_id"], "max_characters": 100}, "read1")
    assert read.status == "ok"
    assert "synthetic-sensitive-value" not in json.dumps(read.to_dict())
    assert read.data["artifact_hash"] == source["artifact_hash"]
    other = store.create_session(SessionConfig(allowed_tools=["artifact_read"], enabled_tools=["artifact_read"]).to_dict())["id"]
    task = store.create_task(other, "Foreign")["id"]
    run = store.create_run(other, task, Budget().to_dict())["id"]
    assert call(executor, other, run, "artifact_read", {"artifact_id": source["artifact_id"]}).status == "denied"


def test_artifact_read_page_is_directly_model_readable_without_nested_context_artifact(tmp_path):
    from adj_partner.runtime.context import ContextView
    store, _, _, executor, sid, rid = setup(tmp_path)
    artifact = store.put_artifact("数" * 20000, "large.txt", "text/plain", {}, session_id=sid, run_id=rid)
    result = call(executor, sid, rid, "artifact_read",
                  {"artifact_id": artifact["id"], "max_characters": 16000}, "large-read")
    assert result.status == "ok"
    assert result.data["next_start"] is not None and result.data["next_start"] < result.data["total_characters"]
    message = ContextView(store).tool_message(sid, rid, "large-read", result.to_dict())
    projected = json.loads(message["content"])
    assert projected["data"]["projection"] == "redacted_text_character_slice"
    assert projected["data"]["content"]
    assert projected.get("context_truncated") is not True
    assert len(message["content"].encode("utf-8")) <= 8000


def test_artifact_read_tool_tells_model_to_follow_host_pagination(tmp_path):
    _, _, _, executor, _, _ = setup(tmp_path)
    description = executor.catalog.get("artifact_read").spec.description
    assert "next_start" in description and "guess" in description.lower()


def test_csv_inspection_is_real_and_denied_inputs_produce_no_source(tmp_path):
    data = tmp_path / "sample.csv"
    data.write_text("name,value\na,1\nb,2\n")
    _, research, _, executor, sid, rid = setup(tmp_path)
    result = call(executor, sid, rid, "file_inspect", {"path": str(data)})
    assert result.status == "ok" and result.data["inspection"]["rows"] == 2
    assert result.data["inspection"]["columns"] == ["name", "value"]
    forbidden = call(executor, sid, rid, "file_inspect", {"path": "/etc/passwd"}, "forbidden")
    assert forbidden.status == "denied" and len(research.list_sources(sid, rid)) == 1


def test_fixture_literature_imports_available_material_not_an_id_as_truth(tmp_path):
    body = {"hitCount": 1, "resultList": {"result": [{"id": "123456", "source": "MED", "title": "Fixture",
                                                   "abstractText": "Synthetic abstract, not real evidence."}]}}
    literature = EuropePMC(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    store, research, _, executor, sid, rid = setup(tmp_path, network=True, literature=literature)
    result = call(executor, sid, rid, "literature_search", {"query": "fixture", "page_size": 1})
    assert result.status == "ok" and result.data["hit_count"] == 1
    assert result.data["summary"]["results"][0]["title"] == "Fixture"
    assert result.data["summary"]["results"][0]["abstract_preview"] == "Synthetic abstract, not real evidence."
    source = research.get_source(sid, rid, result.data["results"][0]["source_id"])
    assert source["kind"] == "literature" and source["provenance"]["text_availability"] == "abstract"
    assert source["provenance"]["request_url"] and source["provenance"]["retrieved_at"]
    evidence = research.select_evidence(sid, rid, source["id"], {"pointer": "/abstract"})
    assert evidence["value"] == "Synthetic abstract, not real evidence."
    assert source["provenance"]["response_artifact_id"] in {item["id"] for item in store.artifacts(sid)}


def test_network_policy_prevents_fixture_call_and_cross_run_job_access_is_denied(tmp_path):
    requests = []
    literature = EuropePMC(transport=httpx.MockTransport(lambda req: requests.append(req)))
    store, research, native, executor, sid, rid = setup(tmp_path, literature=literature)
    assert call(executor, sid, rid, "literature_search", {"query": "fixture"}).status == "denied"
    assert requests == [] and research.list_sources(sid, rid) == []
    pending = call(executor, sid, rid, "protein_sequence", {"sequence": "AAAA"}, "protein")
    complete(native, sid, rid, pending.job_id)
    other_task = store.create_task(sid, "Other")["id"]
    other_run = store.create_run(sid, other_task, Budget().to_dict())["id"]
    with pytest.raises((PermissionError, ValueError, KeyError)):
        native.resolve_job(sid, other_run, pending.job_id)


def test_injected_reopened_services_write_with_actual_executing_store_secrets(tmp_path):
    from adj_partner.execution import JobManager
    secret = "synthetic-current-store-secret-451"
    active = Store(tmp_path / "state", secrets=[secret])
    stale = Store(active.root)
    native = native_type()(active, research=ResearchStore(stale), jobs=JobManager(stale))
    catalog = ToolCatalog()
    native.register(catalog)
    sid = active.create_session(SessionConfig(allowed_tools=catalog.names(), enabled_tools=catalog.names(), allowed_roots=[str(tmp_path)]).to_dict())["id"]
    tid = active.create_task(sid, "Fixture")["id"]
    rid = active.create_run(sid, tid, Budget().to_dict())["id"]
    path = tmp_path / (secret + ".txt")
    path.write_text("Ordinary synthetic text")
    result = call(ToolExecutor(active, catalog), sid, rid, "file_inspect", {"path": str(path)})
    assert result.status == "ok"
    with active.transaction() as conn:
        raw_records = "".join(row[0] for row in conn.execute("SELECT payload FROM research_records"))
    assert secret not in raw_records
    assert native.research.store is active and native.jobs.store is active


@pytest.mark.parametrize("kind,limit", [("csv", 200), ("literature_abstract", 4000), ("literature_title", 2000)])
def test_known_secret_crossing_preview_cutoff_is_redacted_before_truncation(tmp_path, kind, limit):
    secret = "synthetic-preview-secret-9812"
    value = "x" * (limit - len(secret) + 1) + secret
    body = {"hitCount": 1, "resultList": {"result": [{"id": "123456", "source": "MED", "title": "Fixture", "abstractText": "Fixture"}]}}
    if kind.startswith("literature"):
        body["resultList"]["result"][0]["title" if kind.endswith("title") else "abstractText"] = value
    literature = EuropePMC(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    store, _, _, executor, sid, rid = setup(tmp_path, network=True, literature=literature)
    store.register_secrets([secret])
    if kind == "csv":
        path = tmp_path / "sample.csv"
        path.write_text("column\n" + value + "\n")
        result = call(executor, sid, rid, "file_inspect", {"path": str(path)})
    else:
        result = call(executor, sid, rid, "literature_search", {"query": "fixture"})
    assert result.status == "ok", result.to_dict()
    assert secret[:-1] not in json.dumps(result.to_dict())
    with store.transaction() as conn:
        records = "".join(row[0] for row in conn.execute("SELECT payload FROM research_records"))
    assert secret[:-1] not in records


@pytest.mark.parametrize("extension", ["json", "csv"])
def test_inspection_key_and_header_previews_have_character_bounds(tmp_path, extension):
    _, _, _, executor, sid, rid = setup(tmp_path)
    # CSV's standard parser limits individual fields independently of preview.
    key = "x" * (100000 if extension == "csv" else 200000)
    path = tmp_path / ("sample." + extension)
    path.write_text(json.dumps({key: 1}) if extension == "json" else key + "\n1\n")
    result = call(executor, sid, rid, "file_inspect", {"path": str(path)})
    assert result.status == "ok", result.to_dict()
    assert len(json.dumps(result.to_dict())) < 12000
    assert result.data["inspection"]["labels_truncated"] is True


def test_later_secret_registration_does_not_change_immutable_job_import_identity(tmp_path):
    secret = "later-known-file-marker-654"
    path = tmp_path / (secret + ".fasta")
    path.write_text(">fixture\nAAAA\n")
    store, research, native, executor, sid, rid = setup(tmp_path)
    arguments = {"path": str(path)}
    pending = call(executor, sid, rid, "protein_fasta", arguments)
    first = complete(native, sid, rid, pending.job_id)
    assert first.status == "ok"
    store.register_secrets([secret])
    again = native.resolve_job(sid, rid, pending.job_id)
    assert again.data["source_id"] == first.data["source_id"]
    assert secret not in json.dumps(again.to_dict())
    assert native.recover_action(sid, rid, ToolCall("call1", "protein_fasta", arguments)).data["source_id"] == first.data["source_id"]
    assert len(research.list_sources(sid, rid)) == 1
