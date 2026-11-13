import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest

from adj_partner.analysis import AnalysisCapabilities
from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.contracts import ModelReply
from adj_partner.models import ScriptedPort
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


PROJECT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT / "data/gene_diff/AminosalicylicVSDMSO_Gene_differential_expression.xlsx"
EXPECTED_HASH = "58834bb4fea2e436e1aecf4abd8b2801468fd1d4319dfe509f2feb9c0d1d9055"


def wait_for(capability, session_id, run_id, job_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = capability.resolve_job(session_id, run_id, job_id)
        if result.status not in {"pending", "running", "waiting"}:
            return result
        time.sleep(.05)
    pytest.fail("Analysis job did not finish in time")


def make_run(store, tools, root):
    session = store.create_session(SessionConfig(allowed_roots=[str(root)],
        allowed_tools=tools, enabled_tools=tools).to_dict())["id"]
    task = store.create_task(session, "Aminosalicylic workbook acceptance")["id"]
    run = store.create_run(session, task, Budget().to_dict())["id"]
    return session, run


def wait_until(path, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return
        time.sleep(.02)
    pytest.fail(f"Timed out waiting for {Path(path).name}")


def assert_process_exits(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.02)
    pytest.fail(f"Analysis process {pid} survived cancellation")


@pytest.mark.skipif(bool(os.environ.get("CODEX_SANDBOX")),
                    reason="A nested sandbox cannot exercise macOS sandbox-exec")
@pytest.mark.real_sandbox
def test_agent_service_drives_code_interpreter_job_to_completion(tmp_path):
    from adj_partner.application import AgentService

    code = ('from pathlib import Path\nimport json\n'
            'Path("outputs/analysis_result.json").write_text('
            'json.dumps({"summary":{"answer":42}}))\n')
    model = ScriptedPort([
        ModelReply(tool_calls=[ToolCall("run_python", "code_interpreter", {
            "code": code, "title": "agent service analysis", "timeout_seconds": 20})],
            finish_reason="tool_calls", usage={"input_tokens": 1, "output_tokens": 1}),
        ModelReply(content="Analysis completed.", usage={"input_tokens": 1, "output_tokens": 1}),
    ])
    service = AgentService(tmp_path / "state", model=model, project_root=PROJECT)
    session = service.create_session("general", allowed_roots=[tmp_path])
    run = service.submit(session["id"], "Compute the fixture value")

    completed = service.drive(run["id"])

    assert completed["status"] == "succeeded"
    action = service.actions(run["id"])[0]
    assert action["tool_name"] == "code_interpreter"
    assert action["result"]["data"]["summary"] == {"answer": 42}
    assert service.analysis.history.search(session["id"], query="agent service")["records"][0]["status"] == "succeeded"


@pytest.mark.skipif(bool(os.environ.get("CODEX_SANDBOX")) or sys.platform == "darwin",
                    reason="Nested Seatbelt is unavailable and macOS rejects forking Bash workloads")
@pytest.mark.real_sandbox
def test_bash_tool_uses_the_same_real_sandbox_and_history_pipeline(tmp_path):
    store = Store(tmp_path / "state")
    capability = AnalysisCapabilities(store)
    catalog = ToolCatalog(); capability.register(catalog)
    session, run = make_run(store, catalog.names(), tmp_path)
    executor = ToolExecutor(store, catalog)
    pending = executor.execute(ToolCall("run_bash", "bash", {
        "script": "printf '%s\\n' '{\"summary\":{\"shell\":\"ok\"}}' > outputs/analysis_result.json\n",
        "title": "bash sandbox acceptance", "timeout_seconds": 20,
    }), session, run)

    result = wait_for(capability, session, run, pending.job_id)

    assert result.status == "ok", result.to_dict()
    assert result.data["summary"] == {"shell": "ok"}
    assert result.data["sandbox_backend"] in {"macos-sandbox-exec", "linux-bubblewrap"}
    assert capability.history.search(session, query="bash sandbox")["records"][0]["status"] == "succeeded"


@pytest.mark.skipif(bool(os.environ.get("CODEX_SANDBOX")),
                    reason="A nested sandbox cannot exercise macOS sandbox-exec")
@pytest.mark.real_sandbox
def test_durable_cancellation_kills_the_analysis_process_and_atomically_records_history(tmp_path):
    store = Store(tmp_path / "state")
    capability = AnalysisCapabilities(store)
    catalog = ToolCatalog(); capability.register(catalog)
    session, run = make_run(store, catalog.names(), tmp_path)
    executor = ToolExecutor(store, catalog)
    code = ("import os,time\nfrom pathlib import Path\n"
            "Path('outputs/pid.txt').write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
    pending = executor.execute(ToolCall("cancel_python", "code_interpreter", {
        "code": code, "title": "cancellable analysis", "timeout_seconds": 60,
    }), session, run)
    job = capability.jobs.get(pending.job_id, session)
    marker = Path(job["workspace"]) / "outputs" / "pid.txt"
    wait_until(marker)
    sandbox_pid = int(marker.read_text())

    capability.jobs.cancel(pending.job_id, session)
    result = wait_for(capability, session, run, pending.job_id)

    assert result.status == "cancelled"
    record = capability.history.get_record(pending.data["analysis_id"])
    assert record["status"] == "cancelled"
    assert record["error"]["code"] == "cancelled"
    assert_process_exits(sandbox_pid)


@pytest.mark.skipif(bool(os.environ.get("CODEX_SANDBOX")),
                    reason="A nested sandbox cannot exercise macOS sandbox-exec")
@pytest.mark.real_sandbox
def test_original_workbook_analysis_history_and_method_replay(tmp_path):
    from scripts.analyze_aminosalicylic import ANALYSIS_CODE

    assert WORKBOOK.is_file()
    before = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest()
    assert before == EXPECTED_HASH

    store = Store(tmp_path / "state")
    capability = AnalysisCapabilities(store)
    catalog = ToolCatalog(); capability.register(catalog)
    executor = ToolExecutor(store, catalog)
    first_session, first_run = make_run(store, catalog.names(), WORKBOOK.parent)
    pending = executor.execute(ToolCall("aminosalicylic_analysis", "code_interpreter", {
        "code": ANALYSIS_CODE, "inputs": [str(WORKBOOK)],
        "title": "Aminosalicylic vs DMSO workbook integrity and DEG summary",
        "timeout_seconds": 60,
    }), first_session, first_run)
    result = wait_for(capability, first_session, first_run, pending.job_id)

    assert result.status == "ok", result.to_dict()
    summary = result.data["summary"]
    assert summary["data_rows"] == 34899 and summary["columns"] == 20
    assert summary["missing_gene_ids"] == 0 and summary["duplicate_gene_ids"] == 0
    assert summary["significant_yes"] == 1114
    assert summary["nonfinite_numeric_values"] == 0
    assert summary["nonpositive_fc_values"] == 0
    assert summary["log2_fc_rows_checked"] == 34899
    assert summary["regulation_counts_all"] == {"down": 13931, "no": 9115, "up": 11853}
    assert summary["max_log2_fc_recalculation_error"] < 1e-10
    assert {item["name"] for item in result.data["outputs"]} >= {
        "analysis_result.json", "significant_genes.csv", "significant_regulation_counts.csv"}

    registered = executor.execute(ToolCall("register_aminosalicylic", "analysis_method_register", {
        "record_id": result.data["analysis_id"], "name": "analyze_gene_differential_workbook",
        "description": "Validate and summarize a gene differential-expression workbook.",
        "parameter_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }), first_session, first_run)
    assert registered.status == "ok"

    later_session, later_run = make_run(store, catalog.names(), WORKBOOK.parent)
    later_executor = ToolExecutor(store, catalog)
    replay = later_executor.execute(ToolCall("replay_aminosalicylic", "analysis_method_run", {
        "method_id": registered.data["method"]["id"], "parameters": {},
        "inputs": [str(WORKBOOK)], "title": "Replayed Aminosalicylic validation",
        "timeout_seconds": 60,
    }), later_session, later_run)
    replayed = wait_for(capability, later_session, later_run, replay.job_id)
    assert replayed.status == "ok", replayed.to_dict()
    assert replayed.data["summary"]["data_rows"] == summary["data_rows"]
    assert replayed.data["summary"]["significant_yes"] == summary["significant_yes"]

    found = later_executor.execute(ToolCall("find_history", "analysis_history", {
        "query": "Aminosalicylic", "status": "succeeded"}), later_session, later_run)
    assert found.status == "ok" and len(found.data["records"]) >= 2
    original_record = next(item for item in found.data["records"]
                           if item["id"] == result.data["analysis_id"])
    result_artifact = next(item for item in original_record["artifacts"] if item["role"] == "result")
    page = later_executor.execute(ToolCall("read_history", "analysis_history_read", {
        "artifact_id": result_artifact["artifact_id"]}), later_session, later_run)
    assert json.loads(page.data["content"])["summary"]["significant_yes"] == 1114

    assert hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() == before
