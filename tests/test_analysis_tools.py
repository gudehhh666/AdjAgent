import hashlib
import json
import os
from pathlib import Path

import pytest

from adj_partner.analysis.history import AnalysisHistory
from adj_partner.analysis.sandbox import SandboxResult
from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store


class FakeContext:
    def __init__(self, store, session_id, run_id, workspace, job_id="job_fixture"):
        self.store = store
        self.session_id = session_id
        self.run_id = run_id
        self.workspace = Path(workspace)
        self.job_id = job_id
        self.call_id = job_id
        self.messages = []

    def check_cancelled(self):
        return None

    def check_path(self, path, write=False, must_exist=True):
        selected = Path(path).resolve()
        if write and not selected.is_relative_to(self.workspace.resolve()):
            raise PermissionError("write denied")
        if must_exist and not selected.exists():
            raise FileNotFoundError(selected)
        return selected

    def progress(self, message, **details):
        self.messages.append((message, details))


def setup_context(tmp_path):
    store = Store(tmp_path / "state")
    session = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)]).to_dict())
    task = store.create_task(session["id"], "analysis")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    workspace = tmp_path / "work"
    workspace.mkdir()
    return store, session["id"], run["id"], FakeContext(store, session["id"], run["id"], workspace)


def manifest(path):
    content = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "name": Path(path).name,
            "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def finalize(context, analysis_id, result, status=None):
    from adj_partner.analysis.worker import analysis_job_finalize

    if status is None:
        if result.status in {"ok", "cancelled", "uncertain"}:
            status = {"ok": "succeeded", "cancelled": "cancelled", "uncertain": "uncertain"}[result.status]
        else:
            status = "timed_out" if result.error["code"] == "analysis_timeout" else "failed"
    job = {"id": context.job_id, "arguments": {"analysis_id": analysis_id}}
    with context.store.transaction(write=True) as conn:
        analysis_job_finalize(conn, context.store, job, status, result)


def test_workspace_preparation_accepts_job_manager_owned_empty_tmp(tmp_path):
    from adj_partner.analysis.worker import _prepare_workspace

    _, _, _, context = setup_context(tmp_path)
    (context.workspace / "tmp").mkdir(mode=0o700)

    source = _prepare_workspace(context, "print('ok')\n", "python", {})

    assert source.name == "analysis.py"
    assert (context.workspace / "inputs").is_dir()
    assert (context.workspace / "outputs").is_dir()
    assert (context.workspace / "tmp").is_dir()


def test_snapshot_inputs_rejects_content_changed_after_manifest(tmp_path):
    from adj_partner.analysis.worker import snapshot_inputs
    from adj_partner.tools import ToolFailure

    _, _, _, context = setup_context(tmp_path)
    source = tmp_path / "input.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    expected = manifest(source)
    source.write_text("a\n2\n", encoding="utf-8")
    (context.workspace / "inputs").mkdir()

    with pytest.raises(ToolFailure) as caught:
        snapshot_inputs(context, [expected])

    assert caught.value.code == "analysis_input_changed"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo"])
def test_output_collector_rejects_non_regular_files(tmp_path, unsafe_kind):
    from adj_partner.analysis.worker import collect_outputs
    from adj_partner.tools import ToolFailure

    _, _, _, context = setup_context(tmp_path)
    outputs = context.workspace / "outputs"
    outputs.mkdir()
    target = outputs / "unsafe"
    if unsafe_kind == "symlink":
        target.symlink_to(tmp_path / "outside")
    else:
        os.mkfifo(target)

    with pytest.raises(ToolFailure) as caught:
        collect_outputs(context, outputs)

    assert caught.value.code == "analysis_output_invalid"


def test_analysis_job_publishes_validated_outputs_and_history(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    source = tmp_path / "input.csv"
    source.write_text("value\n1\n2\n", encoding="utf-8")
    code = store.put_artifact("print('analyzed')\n", "analysis.py", "text/x-python", {}, session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="python", title="fixture analysis",
        code_artifact_id=code["id"], inputs=[manifest(source)], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)

    class SuccessfulSandbox:
        def run(self, command, workspace, *, timeout_seconds, cancel_check=None):
            output = Path(workspace) / "outputs"
            (output / "analysis_result.json").write_text(json.dumps({
                "summary": {"rows": 2}, "metrics": {"mean": 1.5},
                "limitations": ["fixture only"],
            }), encoding="utf-8")
            (output / "table.csv").write_text("value\n1\n2\n", encoding="utf-8")
            return SandboxResult("fixture-sandbox", 0, "analyzed\n", "", False, .1)

    monkeypatch.setattr(worker, "select_sandbox", lambda: SuccessfulSandbox())
    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[manifest(source)], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    saved = history.get_record(record["id"])
    assert result.status == "ok"
    assert result.data["summary"]["rows"] == 2
    assert saved["status"] == "succeeded"
    assert {item["role"] for item in saved["artifacts"]} >= {"code", "result", "output", "stdout"}
    assert all(item["sha256"] for item in saved["artifacts"])
    output_ids = {item["artifact_id"] for item in saved["artifacts"] if item["role"] in {"result", "output"}}
    assert all(item["provenance"]["analysis_id"] == record["id"]
               for item in store.artifacts(session_id, run_id) if item["id"] in output_ids)


def test_analysis_job_preserves_nonzero_exit_as_stable_error(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    code = store.put_artifact("raise SystemExit(2)\n", "analysis.py", "text/x-python", {}, session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="python", title="failure",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)

    class FailedSandbox:
        def run(self, command, workspace, *, timeout_seconds, cancel_check=None):
            return SandboxResult("fixture-sandbox", 2, "", "bad input", False, .1)

    monkeypatch.setattr(worker, "select_sandbox", lambda: FailedSandbox())
    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    assert result.status == "error"
    assert result.error["code"] == "analysis_exit_nonzero"
    assert history.get_record(record["id"])["error"]["code"] == "analysis_exit_nonzero"


@pytest.mark.parametrize(("sandbox_result", "expected_code"), [
    (SandboxResult("fixture-sandbox", -15, "", "", False, .1), "analysis_signal"),
    (SandboxResult("fixture-sandbox", -9, "", "", False, .1, "memory_limit"),
     "analysis_resource_limit"),
])
def test_analysis_job_distinguishes_signals_from_enforced_limits(
        tmp_path, monkeypatch, sandbox_result, expected_code):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    code = store.put_artifact("print('run')\n", "analysis.py", "text/x-python", {}, session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="python", title="classified failure",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)

    class ClassifiedSandbox:
        def run(self, *args, **kwargs):
            return sandbox_result

    monkeypatch.setattr(worker, "select_sandbox", lambda: ClassifiedSandbox())
    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    assert result.error["code"] == expected_code
    assert history.get_record(record["id"])["error"]["code"] == expected_code


def test_unexpected_analysis_failure_does_not_expose_private_paths(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    code = store.put_artifact("print('never')\n", "analysis.py", "text/x-python", {}, session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="python", title="internal failure",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)
    private_detail = "/private/store/jobs/secret-internal-path"
    monkeypatch.setattr(worker, "_load_source", lambda *args: (_ for _ in ()).throw(RuntimeError(private_detail)))

    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    assert result.error == {"code": "analysis_internal", "message": "Analysis handler failed internally"}
    assert private_detail not in json.dumps(history.get_record(record["id"]))


def test_sandbox_launch_failure_has_stable_nonrevealing_error(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    code = store.put_artifact("print('never')\n", "analysis.py", "text/x-python", {}, session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="python", title="sandbox launch",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)

    class UnstartableSandbox:
        def run(self, *args, **kwargs):
            raise OSError("/private/host/detail")

    monkeypatch.setattr(worker, "select_sandbox", lambda: UnstartableSandbox())
    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    assert result.error == {"code": "sandbox_unavailable",
                            "message": "The host could not start the analysis sandbox"}
    assert "/private/host/detail" not in json.dumps(history.get_record(record["id"]))


def test_macos_bash_fails_closed_before_selecting_a_forking_backend(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    store, session_id, run_id, context = setup_context(tmp_path)
    code = store.put_artifact("printf ok\n", "analysis.sh", "text/x-shellscript", {},
                              session_id, run_id)
    history = AnalysisHistory(store)
    record = history.create_record(session_id, run_id, language="bash", title="mac bash",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 20})
    history.attach_job(record["id"], context.job_id)
    monkeypatch.setattr(worker.sys, "platform", "darwin")
    monkeypatch.setattr(worker, "select_sandbox",
                        lambda: pytest.fail("unsafe macOS Bash backend was selected"))

    result = worker.analysis_job(context, analysis_id=record["id"], language="bash",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)
    finalize(context, record["id"], result)

    assert result.error["code"] == "sandbox_unavailable"
    assert "Linux Bubblewrap" in result.error["message"]
    assert history.get_record(record["id"])["status"] == "failed"
