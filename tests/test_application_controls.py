"""Real application composition and host boundaries; no live provider calls."""

import json
from pathlib import Path
import sys
from threading import Event

import pytest

from adj_partner.contracts import Budget, ModelReply, ToolCall, ToolResult
from adj_partner.models import ScriptedPort
from adj_partner.tools import ToolSpec
from test_application import PROJECT, service_type


@pytest.fixture(autouse=True)
def no_ambient_provider_credentials(monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def response(text="done", calls=()):
    return ModelReply(content=text, tool_calls=list(calls), finish_reason="tool_calls" if calls else "stop",
                      usage={"input_tokens": 1, "output_tokens": 1})


def make(tmp_path, replies=()):
    return service_type()(tmp_path / "state", model=ScriptedPort(replies), project_root=PROJECT)


def test_services_share_actual_store_and_default_permissions_are_bounded(tmp_path):
    service = make(tmp_path)
    assert service.research.store is service.native.store is service.analysis.store is service.executor.store is service.store
    sid = service.create_session()["id"]
    config = service.session(sid)["config"]
    assert config["allow_network"] is False and config["allowed_roots"] == []
    assert "ask_user" in config["enabled_tools"] and "protein_sequence" in config["allowed_tools"]
    analysis = {"code_interpreter", "bash", "analysis_method_register",
                "analysis_method_run", "analysis_history", "analysis_history_read"}
    assert analysis <= set(config["allowed_tools"])
    assert set(config["enabled_tools"]) <= set(config["allowed_tools"])
    from adj_partner.profiles import get_profile
    assert set(config["enabled_tools"]) == set(get_profile(config["profile"])["tools"]) & set(config["allowed_tools"])
    service.catalog.register(ToolSpec("external", "Host-only extra", {"type": "object"}), lambda ctx: ToolResult())
    newer = service.create_session()["id"]
    assert "external" not in service.session(newer)["config"]["allowed_tools"]
    with pytest.raises((KeyError, ValueError)): service.create_session(allowed_tools=["unknown"])


def test_host_authorizes_analysis_for_legacy_session_only_between_runs(tmp_path):
    from adj_partner.contracts import SessionConfig
    from adj_partner.runtime.journal import RunJournal

    service = make(tmp_path)
    legacy = service.store.create_session(SessionConfig(
        allowed_tools=["ask_user"], enabled_tools=["ask_user"]).to_dict())
    upgraded = service.authorize_analysis(legacy["id"])
    names = {"code_interpreter", "bash", "analysis_method_register",
             "analysis_method_run", "analysis_history", "analysis_history_read"}
    assert names <= set(upgraded["config"]["allowed_tools"])
    assert names <= set(upgraded["config"]["enabled_tools"])
    assert upgraded["revision"] == legacy["revision"] + 1
    assert service.events(legacy["id"])[-1]["kind"] == "analysis.authorized"

    active = service.store.create_session(SessionConfig().to_dict())
    RunJournal(service.store).start(active["id"], "active", {}, active["config"]["budget"])
    before = service.store.get_session(active["id"])
    with pytest.raises(RuntimeError, match="between completed"):
        service.authorize_analysis(active["id"])
    assert service.store.get_session(active["id"]) == before


def test_read_only_views_redact_current_secrets_and_do_not_load_sdk(tmp_path, monkeypatch):
    service = make(tmp_path, [response("newly-known-secret")])
    sid = service.create_session()["id"]
    rid = service.submit(sid, "newly-known-secret")["id"]
    service.drive(rid)
    service.close()
    monkeypatch.setenv("ADJ_LLM_API_KEY", "newly-known-secret")
    from adj_partner.models import OpenAICompatiblePort
    monkeypatch.setattr(OpenAICompatiblePort, "from_environment", lambda **kwargs: pytest.fail("Read-only use must not create provider"))
    reopened = service_type()(tmp_path / "state", project_root=PROJECT)
    public = [reopened.messages(sid), reopened.run(rid), reopened.events(sid), reopened.actions(rid), reopened.model_info(sid)]
    assert "newly-known-secret" not in json.dumps(public)
    assert reopened.model_info(sid)["loaded"] is False


def test_host_model_preference_is_durable_safe_boundary_not_sessionconfig(tmp_path):
    service = service_type()(tmp_path / "state", project_root=PROJECT)
    sid = service.create_session()["id"]
    assert service.set_model(sid, "host-model-v2")["model"] == "host-model-v2"
    assert "model" not in service.session(sid)["config"]
    service.close()
    reopened = service_type()(tmp_path / "state", project_root=PROJECT)
    assert reopened.model_info(sid)["model"] == "host-model-v2"
    assert any(item["kind"] == "model.changed" for item in reopened.events(sid))
    injected = make(tmp_path / "injected")
    iid = injected.create_session()["id"]
    with pytest.raises(ValueError): injected.set_model(iid, "pretend-live-model")
    rid = injected.submit(iid, "outstanding")["id"]
    with pytest.raises(RuntimeError): injected.set_model(iid, injected.model_info(iid)["model"])
    injected.cancel(rid)


@pytest.mark.parametrize("name", ["", " ", "x" * 257, "model\nname"])
def test_invalid_model_name_does_not_mutate_preferences(tmp_path, name):
    service = make(tmp_path)
    sid = service.create_session()["id"]
    before = service.events(sid)
    with pytest.raises(ValueError): service.set_model(sid, name)
    assert service.events(sid) == before


def test_explicit_dotenv_priming_is_anchored_and_no_secret_in_model_info(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "explicit.env").write_text("ADJ_LLM_API_KEY=synthetic-explicit-key\nADJ_LLM_MODEL=fixture-model\n")
    monkeypatch.chdir(tmp_path)
    service = service_type()(tmp_path / "state", project_root=project, env_file="explicit.env")
    sid = service.create_session(instructions="synthetic-explicit-key")["id"]
    assert "synthetic-explicit-key" not in json.dumps([service.session(sid), service.model_info(sid)])
    assert service.model_info(sid)["model"] == "fixture-model"


def test_drive_runs_real_native_job_and_reopen_keeps_actual_store(tmp_path):
    service = make(tmp_path, [response("", [ToolCall("p", "protein_sequence", {"sequence": "AAAA"})]), response()])
    sid = service.create_session(profile="protein")["id"]
    rid = service.submit(sid, "Inspect sequence")["id"]
    done = service.drive(rid)
    assert done["status"] == "succeeded"
    assert len(service.jobs(sid)) == 1 and len(service.research.list_sources(sid, rid)) == 1
    assert service.actions(rid)[0]["finished_at"] is not None
    reopened = service_type()(tmp_path / "state", model=ScriptedPort([]), project_root=PROJECT)
    assert reopened.native.store is reopened.store and reopened.run(rid)["usage"] == done["usage"]


def test_drive_detach_does_not_initiate_and_cancel_requests_associated_job(tmp_path):
    service = make(tmp_path, [response("", [ToolCall("p", "protein_sequence", {"sequence": "AAAA"})])])
    sid = service.create_session(profile="protein")["id"]
    rid = service.submit(sid, "Inspect")["id"]
    stop = Event(); stop.set()
    assert service.drive(rid, stop_event=stop)["status"] == "waiting" and service.jobs(sid) == []
    stop.clear()
    assert service.resume(rid)["status"] == "waiting"
    assert service.cancel(rid)["status"] == "cancelled"
    assert service.jobs(sid)[0]["status"] in {"cancel_requested", "cancelled", "succeeded"}


def test_actual_approval_and_user_answer_resume_original_calls(tmp_path):
    service = make(tmp_path, [response("", [ToolCall("a", "approved", {})]), response("", [ToolCall("ask", "ask_user", {"question": "Sample?"})]), response()])
    effects = []
    service.catalog.register(ToolSpec("approved", "Host configured", {"type": "object"}, requires_approval=True),
        lambda ctx: effects.append("once") or ToolResult(data="done"))
    sid = service.create_session(allowed_tools=["approved", "ask_user"])["id"]
    service.enable_tool(sid, "approved")
    rid = service.submit(sid, "Inspect")["id"]
    assert service.drive(rid)["status"] == "waiting" and effects == []
    assert service.approvals(sid)[0]["call_id"] == "a"
    service.decide_approval(sid, rid, "a", True)
    assert service.drive(rid)["status"] == "waiting" and effects == ["once"]
    question = service.pending_inputs(rid)[0]
    service.answer(rid, question["id"], "actual sample")
    assert service.drive(rid)["status"] == "succeeded" and effects == ["once"]


def test_research_acceptance_and_budgets_are_explicit(tmp_path):
    service = make(tmp_path, [response("unchecked conclusion")])
    sid = service.create_session(budget=Budget(max_steps=2))["id"]
    rid = service.submit(sid, "Research", acceptance={"require_report": True})["id"]
    done = service.drive(rid)
    assert done["status"] == "failed" and done["state"]["stop_reason"] == "acceptance_not_met"
    assert done["budget"]["max_steps"] == 2


def test_task_getters_expose_actual_acceptance_without_mutation(tmp_path):
    service = make(tmp_path)
    sid = service.create_session()["id"]
    rid = service.submit(sid, "Requirements", acceptance={"require_report": True})["id"]
    tid = service.run(rid)["task_id"]
    assert service.task(tid)["acceptance"] == {"require_report": True}
    assert service.tasks(sid) == [service.task(tid)]


def test_close_is_idempotent_for_injected_client_and_does_not_cancel_jobs(tmp_path):
    class Client(ScriptedPort):
        closed = 0
        def close(self): self.closed += 1
    client = Client([])
    service = service_type()(tmp_path/"state",model=client,project_root=PROJECT)
    sid = service.create_session()["id"]
    rid = service.submit(sid,"Unstarted work")["id"]
    service.close(); service.close()
    assert client.closed == 1 and service.store.get_run(rid)["status"] == "created"


def test_explicit_local_mcp_registration_is_real_but_not_default_authorization(tmp_path):
    from adj_partner.tools.mcp import MCPServerConfig
    fixture = PROJECT/"tests/fixtures/mcp_echo_server.py"
    port = ScriptedPort([response("",[ToolCall("m","local__environment",{})]),response()])
    service = service_type()(tmp_path/"state",project_root=PROJECT,model=port,mcp_config=[MCPServerConfig(
        "local",command=sys.executable,args=("-B",str(fixture)),network=False,requires_approval=False,
        env={"HARNESS_EXPLICIT_VALUE":"APP_MCP_PRIVATE"},connection_timeout=3,call_timeout=2)])
    default = service.create_session()["id"]
    assert "local__environment" not in service.session(default)["config"]["allowed_tools"]
    sid = service.create_session(allowed_tools=["local__environment"])["id"]
    service.enable_tool(sid,"local__environment")
    rid = service.submit(sid,"Explicit local integration")["id"]
    assert service.drive(rid)["status"]=="succeeded"
    action = service.actions(rid)[0]
    assert action["result"]["data"]["configured_value"]=="[REDACTED]"
    assert "APP_MCP_PRIVATE" not in json.dumps([service.events(sid),service.session(sid),service.tools(sid)])


@pytest.mark.parametrize("operation", ["display", "change"])
def test_model_controls_prime_new_catalog_secrets_before_display_or_persistence(tmp_path, operation):
    service = service_type()(tmp_path/"state",project_root=PROJECT)
    sid = service.create_session()["id"]
    secret = "SYNTHETIC_CATALOG_PRIVATE"
    if operation == "display": service.set_model(sid,secret)
    service.catalog.register_many([],secrets=[secret])
    if operation == "display":
        assert secret not in json.dumps(service.model_info(sid))
    else:
        before = service.store.events(sid)
        with pytest.raises(ValueError): service.set_model(sid,secret)
        assert service.store.events(sid)==before
