"""Public application contract; all clients must compose the same real engine."""

import importlib.util
import json
from pathlib import Path
import threading

import pytest

from adj_partner.contracts import ModelReply
from adj_partner.models import ScriptedPort


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_ambient_provider_credentials(monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def service_type():
    assert importlib.util.find_spec("adj_partner.application"), "single AgentService application must exist"
    from adj_partner.application import AgentService
    return AgentService


def test_one_service_creates_converses_reopens_and_exports_trace(tmp_path):
    port = ScriptedPort([ModelReply(content="你好", usage={"input_tokens": 1, "output_tokens": 1})])
    service = service_type()(tmp_path / "state", model=port, project_root=PROJECT)
    sid = service.create_session(profile="general")["id"]
    run = service.submit(sid, "hello")
    assert len(port.requests) == 0
    assert service.drive(run["id"])["status"] == "succeeded"
    assert service.messages(sid)[-1]["content"] == "你好"
    service.close()
    reopened = service_type()(tmp_path / "state", project_root=PROJECT)
    assert reopened.session(sid)["id"] == sid and reopened.run(run["id"])["status"] == "succeeded"
    trace = reopened.trace(sid, tmp_path / "export")
    for key in ("jsonl", "markdown", "manifest"): assert Path(trace[key]).is_file()
    assert "hello" in Path(trace["jsonl"]).read_text()
    reopened.close()


def test_profile_does_not_increase_authorization_and_tools_are_session_scoped(tmp_path):
    service = service_type()(tmp_path / "state", model=ScriptedPort([]), project_root=PROJECT)
    sid = service.create_session(profile="protein", allowed_tools=["catalog_search"])["id"]
    with pytest.raises(PermissionError): service.enable_tool(sid, "protein_sequence")
    service.set_profile(sid, "adjuvant")
    assert service.session(sid)["config"]["allowed_tools"] == ["catalog_search"]
    assert all(not row["enabled"] for row in service.tools(sid) if row["name"] != "catalog_search")
    service.close()


def test_read_only_application_use_needs_no_model_key_or_implicit_dotenv(tmp_path, monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"): monkeypatch.delenv(key, raising=False)
    project = tmp_path / "source"
    project.mkdir()
    (project / ".env").write_text("ADJ_LLM_API_KEY=synthetic-must-not-autoload")
    service = service_type()(tmp_path / "state", project_root=project)
    assert service.sessions() == []
    sid = service.create_session()["id"]
    assert service.session(sid)["id"] == sid
    with pytest.raises(Exception): service.submit(sid, "requires configured model")
    assert service.messages(sid) == []
    service.close()


def test_pending_user_input_and_cancellation_use_actual_runtime_state(tmp_path):
    from adj_partner.contracts import ToolCall
    port = ScriptedPort([ModelReply(tool_calls=[ToolCall("ask", "ask_user", {"question": "Which sample?"})],
        finish_reason="tool_calls", usage={"input_tokens": 1, "output_tokens": 1})])
    service = service_type()(tmp_path / "state", model=port, project_root=PROJECT)
    sid = service.create_session()["id"]
    rid = service.submit(sid, "inspect")["id"]
    assert service.drive(rid)["status"] == "waiting"
    assert service.pending_inputs(rid)[0]["question"] == "Which sample?"
    assert service.cancel(rid)["status"] == "cancelled"
    assert service.run(rid)["status"] == "cancelled"
    service.close()


def test_host_relative_paths_anchor_to_project_not_current_directory(tmp_path, monkeypatch):
    project = tmp_path / "source"
    project.mkdir()
    (project / "inputs").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    service = service_type()(tmp_path / "state", model=ScriptedPort([]), project_root=project, dataset_dir="inputs")
    sid = service.create_session(allowed_roots=["inputs"])["id"]
    assert service.session(sid)["config"]["allowed_roots"] == [str(project / "inputs")]
    assert service.native.dataset_dir == str(project / "inputs")
    service.close()


def test_submit_forwards_optional_execution_binding_to_engine(tmp_path, monkeypatch):
    service = service_type()(tmp_path / "state", model=ScriptedPort([]), project_root=PROJECT)
    sid = service.create_session(profile="general")['id']
    seen = {}

    class FakeEngine:
        def start(self, session_id, goal, *, acceptance=None, execution_binding=None):
            seen.update(session_id=session_id, goal=goal, acceptance=acceptance,
                        execution_binding=execution_binding)
            return {"id": "synthetic-run"}

    monkeypatch.setattr(service, "_engine", lambda session_id: FakeEngine())
    binding = {"allowed_roots": [str(tmp_path)], "input_paths": [str(tmp_path / "public.csv")]}
    result = service.submit(sid, "compute", acceptance={"response_json_schema": {"type": "object"}},
                            execution_binding=binding)

    assert result["id"] == "synthetic-run"
    assert seen["execution_binding"] == binding
    assert seen["acceptance"]["response_json_schema"] == {"type": "object"}
    service.close()
