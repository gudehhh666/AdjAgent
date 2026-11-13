"""Actual subprocess entrypoints from outside the source checkout."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "scripts/adj-partner"


def invoke(tmp_path, *args, offline=True):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("ADJ_LLM_", "OPENAI_"))}
    environment["ADJ_PARTNER_PYTHON"] = sys.executable
    command = [str(LAUNCHER), "--state-dir", str(tmp_path / "state"), "--json"]
    if offline: command.append("--offline-demo")
    assert LAUNCHER.is_file(), "source-checkout launcher must exist"
    return subprocess.run(command + list(args), cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)


def test_help_and_read_only_sessions_work_outside_checkout_without_key(tmp_path):
    help_result = invoke(tmp_path, "--help", offline=False)
    assert help_result.returncode == 0 and "--env-file" in help_result.stdout and "resume" in help_result.stdout
    result = invoke(tmp_path, "sessions", offline=False)
    assert result.returncode == 0 and json.loads(result.stdout) == []


def test_cli_create_offline_run_followup_status_and_trace(tmp_path):
    created = invoke(tmp_path, "new", "--profile", "protein", "--max-steps", "3", "--max-input-tokens", "80000", "--max-output-tokens", "6000", "--wall-time-seconds", "60")
    assert created.returncode == 0, created.stderr
    sid = json.loads(created.stdout)["id"]
    first = invoke(tmp_path, "run", sid, "hello")
    assert first.returncode == 0, first.stderr
    run = json.loads(first.stdout)
    assert run["status"] == "succeeded" and "Offline" in run["state"]["delivery"]["content"]
    follow = invoke(tmp_path, "run", sid, "follow up")
    assert follow.returncode == 0 and json.loads(follow.stdout)["task_id"] != run["task_id"]
    status = invoke(tmp_path, "status", run["id"], offline=False)
    assert status.returncode == 0 and json.loads(status.stdout)["run"]["budget"]["max_steps"] == 3
    exported = invoke(tmp_path, "trace", sid, "--destination", str(tmp_path / "export"), offline=False)
    result = json.loads(exported.stdout)
    assert exported.returncode == 0 and Path(result["manifest"]).name == "manifest.json"


def test_explicit_research_mode_cannot_pass_offline_free_text(tmp_path):
    sid = json.loads(invoke(tmp_path, "new").stdout)["id"]
    result = invoke(tmp_path, "run", sid, "Research", "--research")
    assert result.returncode != 0
    run = json.loads(result.stdout)
    assert run["status"] == "failed" and run["state"]["stop_reason"] == "acceptance_not_met"


def test_cli_host_permissions_model_profile_and_unknown_criteria(tmp_path):
    root = tmp_path / "inputs"; root.mkdir()
    created = invoke(tmp_path, "new", "--allow-root", str(root), "--allow-network", "--allow-tool", "catalog_search", offline=False)
    assert created.returncode == 0, created.stderr
    sid = json.loads(created.stdout)["id"]
    assert json.loads(created.stdout)["config"]["allowed_roots"] == [str(root)]
    denied = invoke(tmp_path, "tools", sid, "--enable", "protein_sequence", offline=False)
    assert denied.returncode != 0 and "error" in json.loads(denied.stdout)
    assert invoke(tmp_path, "profile", sid, "adjuvant", offline=False).returncode == 0
    choice = invoke(tmp_path, "model", sid, "host-choice", offline=False)
    assert choice.returncode == 0 and json.loads(choice.stdout)["model"] == "host-choice"
    bad = tmp_path / "criteria.json"; bad.write_text('{"unknown_requirement":true}')
    # Separate offline session keeps the fixed fixture identity honest.
    offline_sid = json.loads(invoke(tmp_path, "new").stdout)["id"]
    rejected = invoke(tmp_path, "run", offline_sid, "Research", "--acceptance", str(bad))
    assert rejected.returncode != 0 and "error" in json.loads(rejected.stdout)


def test_cli_explicitly_authorizes_analysis_for_a_legacy_session(tmp_path):
    from adj_partner.contracts import SessionConfig
    from adj_partner.storage import Store

    store = Store(tmp_path / "state")
    legacy = store.create_session(SessionConfig(
        allowed_tools=["ask_user"], enabled_tools=["ask_user"]).to_dict())
    store.close()

    result = invoke(tmp_path, "analysis-authorize", legacy["id"])

    assert result.returncode == 0, result.stdout
    config = json.loads(result.stdout)["config"]
    assert {"bash", "code_interpreter", "analysis_history"} <= set(config["allowed_tools"])
    assert {"bash", "code_interpreter", "analysis_history"} <= set(config["enabled_tools"])


def test_module_entrypoint_and_terminal_projection_escape_controls(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ADJ_LLM_", "OPENAI_"))}
    env["PYTHONPATH"] = str(PROJECT)
    result = subprocess.run([sys.executable, "-m", "adj_partner", "--help"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    from adj_partner.cli import terminal_text
    rendered = terminal_text("中文\x1b]52;c;secret\x07\u202eFAKE\nnext")
    assert "中文" in rendered and "\x1b" not in rendered and "\x07" not in rendered and "\u202e" not in rendered
    assert "\\x1b" in rendered and "\\u202e" in rendered


def test_cli_real_pending_approval_answer_jobs_and_cancel(tmp_path, monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"): monkeypatch.delenv(key, raising=False)
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply, ToolCall, ToolResult
    from adj_partner.models import ScriptedPort
    from adj_partner.tools import ToolSpec
    port = ScriptedPort([ModelReply(tool_calls=[ToolCall("a", "approved", {})], finish_reason="tool_calls")])
    service = AgentService(tmp_path / "state", model=port, project_root=PROJECT)
    service.catalog.register(ToolSpec("approved", "Fixture", {"type":"object"}, requires_approval=True), lambda ctx: ToolResult())
    sid = service.create_session(allowed_tools=["approved"])["id"]
    service.enable_tool(sid,"approved")
    rid = service.submit(sid,"Ask approval")["id"]
    assert service.resume(rid)["status"] == "waiting"
    approved = invoke(tmp_path,"approve",sid,rid,"a")
    assert approved.returncode == 0 and json.loads(approved.stdout)["status"] == "approved"
    question = service.journal.ask(rid,"q","Question")
    answer = invoke(tmp_path,"answer",rid,question["id"],"Actual user")
    assert answer.returncode == 0 and service.journal.input_result(rid,"q")["answer"] == "Actual user"
    assert json.loads(invoke(tmp_path,"jobs",sid).stdout) == []
    cancelled = invoke(tmp_path,"cancel",rid)
    assert cancelled.returncode != 0 and json.loads(cancelled.stdout)["status"] == "cancelled"


def test_cli_resume_reconciles_real_native_job_once(tmp_path, monkeypatch):
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"): monkeypatch.delenv(key, raising=False)
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply, ToolCall
    from adj_partner.models import ScriptedPort
    service = AgentService(tmp_path / "state", project_root=PROJECT, model=ScriptedPort([
        ModelReply(tool_calls=[ToolCall("p", "protein_sequence", {"sequence":"AAAA"})],finish_reason="tool_calls",usage={"input_tokens":1,"output_tokens":1})]))
    sid = service.create_session(profile="protein")["id"]
    rid = service.submit(sid,"Inspect")["id"]
    assert service.resume(rid)["status"] == "waiting"
    result = invoke(tmp_path,"resume",rid)
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "succeeded"
    assert len(service.jobs(sid)) == 1


def test_cli_explicit_request_limits_reach_actual_model_port(tmp_path, monkeypatch, capsys):
    from adj_partner import cli
    from adj_partner.application import AgentService
    from adj_partner.models import ScriptedPort
    for key in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"): monkeypatch.delenv(key,raising=False)
    service = AgentService(tmp_path/"state",model=ScriptedPort([]),project_root=PROJECT)
    sid = service.create_session()["id"]
    ports = []
    def factory(replies):
        port=ScriptedPort(replies); ports.append(port); return port
    monkeypatch.setattr(cli,"ScriptedPort",factory)
    code = cli.main(["--state-dir",str(tmp_path/"state"),"--offline-demo","--json",
        "--request-output-tokens","777","--request-timeout-seconds","33.5","run",sid,"Hello"])
    assert code==0 and json.loads(capsys.readouterr().out)["status"]=="succeeded"
    assert ports[0].requests[0]["max_output_tokens"]==777
    assert ports[0].requests[0]["timeout_seconds"]==33.5
    defaults = cli.parser().parse_args(["sessions"])
    assert defaults.request_output_tokens==4096 and defaults.request_timeout_seconds==120


RUNTIME_DEFAULT_ENV = (
    "ADJ_PARTNER_REQUEST_OUTPUT_TOKENS", "ADJ_PARTNER_REQUEST_TIMEOUT_SECONDS",
    "ADJ_PARTNER_MAX_STEPS", "ADJ_PARTNER_MAX_INPUT_TOKENS",
    "ADJ_PARTNER_MAX_OUTPUT_TOKENS", "ADJ_PARTNER_WALL_TIME_SECONDS",
)


def test_cli_explicit_dotenv_sets_runtime_defaults_and_reaches_model_port(tmp_path, monkeypatch, capsys):
    from adj_partner import cli
    from adj_partner.models import ScriptedPort
    for name in RUNTIME_DEFAULT_ENV: monkeypatch.delenv(name, raising=False)
    chosen = tmp_path / "runtime.env"
    chosen.write_text("\n".join([
        "ADJ_PARTNER_REQUEST_OUTPUT_TOKENS=777",
        "ADJ_PARTNER_REQUEST_TIMEOUT_SECONDS=33.5",
        "ADJ_PARTNER_MAX_STEPS=7",
        "ADJ_PARTNER_MAX_INPUT_TOKENS=70000",
        "ADJ_PARTNER_MAX_OUTPUT_TOKENS=7000",
        "ADJ_PARTNER_WALL_TIME_SECONDS=77.5",
    ]), encoding="utf-8")
    ports = []
    def factory(replies):
        port = ScriptedPort(replies); ports.append(port); return port
    monkeypatch.setattr(cli, "ScriptedPort", factory)
    common = ["--state-dir", str(tmp_path / "state"), "--env-file", str(chosen), "--offline-demo", "--json"]
    assert cli.main(common + ["new"]) == 0
    session = json.loads(capsys.readouterr().out)
    assert session["config"]["budget"] == {
        "max_steps": 7, "max_input_tokens": 70000, "max_output_tokens": 7000,
        "wall_time_seconds": 77.5,
    }
    assert cli.main(common + ["run", session["id"], "Hello"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"
    assert ports[-1].requests[0]["max_output_tokens"] == 777
    assert ports[-1].requests[0]["timeout_seconds"] == 33.5


def test_runtime_default_precedence_is_cli_then_process_then_dotenv(tmp_path, monkeypatch, capsys):
    from adj_partner import cli
    for name in RUNTIME_DEFAULT_ENV: monkeypatch.delenv(name, raising=False)
    chosen = tmp_path / "runtime.env"
    chosen.write_text("ADJ_PARTNER_MAX_STEPS=3\nADJ_PARTNER_MAX_OUTPUT_TOKENS=3000\n", encoding="utf-8")
    monkeypatch.setenv("ADJ_PARTNER_MAX_STEPS", "5")
    common = ["--state-dir", str(tmp_path / "state"), "--env-file", str(chosen), "--offline-demo", "--json"]
    assert cli.main(common + ["new"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["budget"]["max_steps"] == 5
    assert cli.main(common + ["new", "--max-steps", "9", "--max-output-tokens", "9000"]) == 0
    budget = json.loads(capsys.readouterr().out)["config"]["budget"]
    assert budget["max_steps"] == 9 and budget["max_output_tokens"] == 9000


@pytest.mark.parametrize("name,value", [
    ("ADJ_PARTNER_REQUEST_OUTPUT_TOKENS", "0"),
    ("ADJ_PARTNER_REQUEST_OUTPUT_TOKENS", "1.5"),
    ("ADJ_PARTNER_REQUEST_TIMEOUT_SECONDS", "nan"),
    ("ADJ_PARTNER_MAX_STEPS", "-1"),
    ("ADJ_PARTNER_MAX_INPUT_TOKENS", "not-an-integer"),
    ("ADJ_PARTNER_MAX_OUTPUT_TOKENS", "inf"),
    ("ADJ_PARTNER_WALL_TIME_SECONDS", "0"),
])
def test_runtime_default_environment_rejects_invalid_values_without_running(name, value, tmp_path, monkeypatch, capsys):
    from adj_partner import cli
    for item in RUNTIME_DEFAULT_ENV: monkeypatch.delenv(item, raising=False)
    chosen = tmp_path / "runtime.env"
    chosen.write_text(f"{name}={value}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--state-dir", str(tmp_path / "state"), "--env-file", str(chosen), "--offline-demo", "sessions"])
    assert error.value.code == 2 and name in capsys.readouterr().err


@pytest.mark.parametrize("flag,value", [("--request-output-tokens","0"),("--request-output-tokens","-1"),
    ("--request-output-tokens","1.2"),("--request-timeout-seconds","0"),
    ("--request-timeout-seconds","nan"),("--request-timeout-seconds","inf")])
def test_cli_request_limits_reject_nonpositive_nonfinite_or_noninteger_values(flag,value,capsys):
    from adj_partner.cli import parser
    with pytest.raises(SystemExit): parser().parse_args([flag,value,"sessions"])
    assert "positive" in capsys.readouterr().err
