import importlib.util
import subprocess
from pathlib import Path

import pytest

from adj_partner.analysis.history import AnalysisHistory
from adj_partner.analysis.sandbox import SandboxResult
from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store


class FakeContext:
    def __init__(self, store, session_id, run_id, workspace):
        self.store = store
        self.session_id = session_id
        self.run_id = run_id
        self.workspace = Path(workspace)
        self.job_id = "job_serialization_hint"

    def check_cancelled(self):
        return None

    def progress(self, message, **details):
        pass


def _analysis_fixture(tmp_path, source):
    store = Store(tmp_path / "state")
    session = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)]).to_dict())
    task = store.create_task(session["id"], "analysis")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    workspace = tmp_path / "work"
    workspace.mkdir()
    context = FakeContext(store, session["id"], run["id"], workspace)
    code = store.put_artifact(source, "analysis.py", "text/x-python", {}, session["id"], run["id"])
    record = AnalysisHistory(store).create_record(session["id"], run["id"], language="python",
        title="serialization failure", code_artifact_id=code["id"], inputs=[], parameters={},
        limits={"timeout_seconds": 20})
    AnalysisHistory(store).attach_job(record["id"], context.job_id)
    return context, record, code


class LocalProcessSandbox:
    """Run the test-authored repro source to capture Python's real traceback."""

    def run(self, command, workspace, *, timeout_seconds, cancel_check=None):
        completed = subprocess.run(command, cwd=workspace, capture_output=True, text=True,
            timeout=timeout_seconds, check=False)
        return SandboxResult("test-local-process", completed.returncode, completed.stdout,
            completed.stderr, False, 0.0)


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None,
                    reason="NumPy is needed for the real serialization traceback")
def test_numpy_scalar_json_traceback_keeps_error_and_adds_repair_hint(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    context, record, code = _analysis_fixture(tmp_path,
        "import json\nimport numpy as np\njson.dumps({'count': np.int64(1)})\n")
    monkeypatch.setattr(worker, "select_sandbox", lambda: LocalProcessSandbox())

    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)

    assert result.error["code"] == "analysis_exit_nonzero"
    details = result.error["details"]
    assert result.error["message"].startswith("Analysis exited with status ")
    assert details["repair_hint"] in result.error["message"]
    assert "Object of type int64 is not JSON serializable" in details["stderr_preview"]
    stderr_artifact = next(item for item in result.artifacts if item["name"] == "stderr.txt")
    raw_stderr = context.store.read_artifact(stderr_artifact["id"]).decode("utf-8")
    assert raw_stderr.startswith(details["stderr_preview"])
    assert "TypeError" in raw_stderr
    assert "item()" in details["repair_hint"]
    assert "int(" in details["repair_hint"] and "float(" in details["repair_hint"]
    assert ".tolist()" in details["repair_hint"]
    assert "analysis_result.json" not in details["repair_hint"]
    from adj_partner.runtime.context import ContextView
    tool_message = ContextView(context.store).tool_message(context.session_id,
        context.run_id, "call_numpy_failure", result.to_dict())
    assert details["repair_hint"] in tool_message["content"]


def test_unrelated_traceback_keeps_generic_error_without_serialization_hint(tmp_path, monkeypatch):
    import adj_partner.analysis.worker as worker

    context, record, code = _analysis_fixture(tmp_path, "raise RuntimeError('unrelated failure')\n")
    monkeypatch.setattr(worker, "select_sandbox", lambda: LocalProcessSandbox())

    result = worker.analysis_job(context, analysis_id=record["id"], language="python",
        code_artifact_id=code["id"], inputs=[], parameters={}, timeout_seconds=20)

    assert result.error["code"] == "analysis_exit_nonzero"
    assert "unrelated failure" in result.error["details"]["stderr_preview"]
    assert "repair_hint" not in result.error["details"]
    assert "Repair hint:" not in result.error["message"]
