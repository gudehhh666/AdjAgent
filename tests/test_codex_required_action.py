"""Opt-in required host-action transport contract (offline only)."""
import io
import json
from copy import deepcopy

import pytest

from adj_partner.models.base import ProviderFailure
from adj_partner.models.codex_app_server import (
    CodexAppServerPort, decode_envelope, envelope_schema, run_native_turn, thread_params,
)
from adj_partner.tools.catalog import validator


USAGE = {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}


def tool(name="echo"):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }}}


def request_messages():
    return [{"role": "user", "content": "Use the authorized host action."}]


def required_call(value=7):
    return {"decision": {"tool_calls": [{"name": "echo", "arguments": {"value": value}}]}}


def required_refusal(text="Cannot perform the action."):
    return {"decision": {"refusal": text}}


def runner_result(text, **extra):
    return {"model": "gpt-6-luna", "status": "completed", "usage": USAGE,
            "text": json.dumps(text), "native_activity": False, **extra}


def test_required_schema_is_nested_and_auto_schema_stays_unchanged():
    tools = [tool()]
    auto = envelope_schema(tools)
    assert set(auto["properties"]) == {"content", "tool_calls"}
    assert auto["properties"]["tool_calls"]["maxItems"] == 128

    required = envelope_schema(tools, tool_choice="required_or_refuse")
    assert set(required) == {"type", "properties", "required", "additionalProperties"}
    assert required["required"] == ["decision"]
    assert set(required["properties"]) == {"decision"}
    decision = required["properties"]["decision"]
    branches = decision["anyOf"]
    assert len(branches) == 2
    call_branch = next(branch for branch in branches if "tool_calls" in branch["properties"])
    refusal_branch = next(branch for branch in branches if "refusal" in branch["properties"])
    assert call_branch["properties"]["tool_calls"]["minItems"] == 1
    assert call_branch["properties"]["tool_calls"]["maxItems"] == 128
    assert refusal_branch["properties"]["refusal"]["minLength"] == 1
    compiled_validator = validator(required)
    assert compiled_validator.is_valid(required_call())
    assert compiled_validator.is_valid(required_refusal())
    assert not compiled_validator.is_valid({"decision": {"tool_calls": []}})
    assert not compiled_validator.is_valid({"content": "premature", "tool_calls": []})
    assert not compiled_validator.is_valid({"decision": {"refusal": ""}})


def test_decode_required_call_and_refusal_preserves_usage():
    reply = decode_envelope(json.dumps(required_call()), USAGE, [tool()],
                            tool_choice="required_or_refuse")
    assert reply.tool_calls[0].name == "echo"
    assert reply.tool_calls[0].arguments == {"value": 7}
    assert reply.usage == USAGE and reply.refusal is None

    refusal = decode_envelope(json.dumps(required_refusal()), USAGE, [tool()],
                              tool_choice="required_or_refuse")
    assert refusal.tool_calls == []
    assert refusal.refusal == "Cannot perform the action."
    assert refusal.content == "" and refusal.usage == USAGE


@pytest.mark.parametrize("payload", [
    {"decision": {"refusal": ""}},
    {"decision": {"refusal": "no", "tool_calls": []}},
    {"decision": {"tool_calls": [{"name": "echo", "arguments": {"value": "bad"}}]}},
])
def test_required_decoder_rejects_empty_mixed_or_invalid_action(payload):
    with pytest.raises(ProviderFailure) as exc:
        decode_envelope(json.dumps(payload), USAGE, [tool()],
                        tool_choice="required_or_refuse")
    assert exc.value.code == "required_action_missing"


def test_required_decoder_rejects_unexposed_action_without_downgrading():
    payload = {"decision": {"tool_calls": [{"name": "not_authorized", "arguments": {}}]}}
    with pytest.raises(ProviderFailure) as exc:
        decode_envelope(json.dumps(payload), USAGE, [tool()],
                        tool_choice="required_or_refuse")
    assert exc.value.code == "codex_unexposed_tool"


def test_thread_params_use_mode_specific_instructions():
    auto = thread_params("gpt-6-luna", "/tmp", tool_choice="auto")
    required = thread_params("gpt-6-luna", "/tmp", tool_choice="required_or_refuse")
    assert "content" in auto["baseInstructions"] and "tool_calls" in auto["baseInstructions"]
    assert "decision" not in auto["baseInstructions"]
    assert "decision" in required["baseInstructions"]
    assert "refusal" in required["baseInstructions"]
    assert "content" not in required["baseInstructions"]
    assert "authorized host tools" in required["baseInstructions"]


def test_required_final_answer_fails_once_with_known_usage_and_receipt(tmp_path):
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        return runner_result({"content": "premature", "tool_calls": []})
    port = CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, runner=runner)
    with pytest.raises(ProviderFailure) as exc:
        port.complete(request_messages(), [tool()], max_output_tokens=128,
                      timeout_seconds=10, tool_choice="required_or_refuse")
    assert exc.value.code == "required_action_missing"
    assert exc.value.usage == USAGE and exc.value.cost_unknown is False
    assert len(calls) == 1
    receipt = json.loads(next(tmp_path.glob("*/receipt.json")).read_text())
    assert receipt["status"] == "failed"
    assert receipt["tool_choice"] == "required_or_refuse"
    assert receipt["usage"] == USAGE


def test_required_refusal_is_a_typed_model_reply_and_records_decision(tmp_path):
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        return runner_result(required_refusal())
    port = CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, runner=runner)
    reply = port.complete(request_messages(), [tool()], max_output_tokens=128,
                          timeout_seconds=10, tool_choice="required_or_refuse")
    assert reply.refusal == "Cannot perform the action." and not reply.tool_calls
    assert len(calls) == 1
    receipt = json.loads(next(tmp_path.glob("*/receipt.json")).read_text())
    assert receipt["status"] == "completed"
    assert receipt["tool_choice"] == "required_or_refuse"
    assert receipt["decision"] == "refusal"


def test_invalid_required_mode_and_empty_tools_fail_before_runner(tmp_path):
    calls = []
    port = CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
                              runner=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ProviderFailure) as invalid:
        port.complete(request_messages(), [tool()], max_output_tokens=128,
                      timeout_seconds=10, tool_choice="required")
    assert invalid.value.code == "invalid_request" and not calls
    with pytest.raises(ProviderFailure) as empty:
        port.complete(request_messages(), [], max_output_tokens=128,
                      timeout_seconds=10, tool_choice="required_or_refuse")
    assert empty.value.code == "invalid_request" and not calls
    assert not list(tmp_path.rglob("receipt.json"))


def test_metadata_declares_version_and_explicit_tool_choice_capability(tmp_path):
    metadata = CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path).metadata
    assert metadata["transport_version"] == "0.7"
    assert metadata["tool_choice_modes"] == ["auto", "required_or_refuse"]


class _FakeStdin:
    def __init__(self):
        self.lines = []
    def write(self, text):
        self.lines.append(text)
    def flush(self):
        pass
    def close(self):
        pass


class _FakeProcess:
    def __init__(self, events):
        self.stdout = io.StringIO("".join(json.dumps(item) + "\n" for item in events))
        self.stdin = _FakeStdin()
    def terminate(self):
        pass
    def kill(self):
        pass
    def wait(self, **kwargs):
        return 0


def test_native_turn_uses_required_schema_and_rejects_early_final(monkeypatch, tmp_path):
    events = [
        {"id": 1, "result": {"userAgent": "test"}},
        {"id": 2, "result": {"model": "gpt-6-luna", "modelProvider": "openai",
            "thread": {"id": "thread1"}, "runtimeWorkspaceRoots": [],
            "instructionSources": [], "sandbox": {"type": "readOnly", "networkAccess": False},
            "approvalPolicy": "never", "multiAgentMode": "explicitRequestOnly"}},
        {"id": 3, "result": {"turn": {"id": "turn1"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread1", "turnId": "turn1",
            "tokenUsage": {"total": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14}}}},
        {"method": "turn/completed", "params": {"threadId": "thread1",
            "turn": {"id": "turn1", "status": "completed", "items": [
                {"type": "agentMessage", "id": "msg1", "phase": "final_answer",
                 "text": json.dumps({"content": "premature", "tool_calls": []})}]}}},
    ]
    process = _FakeProcess(events)
    monkeypatch.setattr("adj_partner.models.codex_app_server.subprocess.Popen",
                        lambda *args, **kwargs: process)
    request = {"messages": request_messages(), "tools": [tool()],
               "max_output_tokens": 128, "timeout_seconds": 10,
               "tool_choice": "required_or_refuse"}
    with pytest.raises(ProviderFailure) as exc:
        run_native_turn(model="gpt-6-luna", effort="low", request=request,
                        trace_path=tmp_path / "events.jsonl", cli="/fake",
                        config_path=tmp_path / "missing.toml")
    assert exc.value.code == "required_action_missing"
    turns = [json.loads(line) for line in process.stdin.lines
             if json.loads(line).get("method") == "turn/start"]
    assert len(turns) == 1
    schema = turns[0]["params"]["outputSchema"]
    assert schema["required"] == ["decision"]
    assert "anyOf" in schema["properties"]["decision"]


def _native_events(payload, status="completed"):
    item = {"type": "agentMessage", "id": "msg1", "phase": "final_answer",
            "text": json.dumps(payload)}
    return [
        {"id": 1, "result": {"userAgent": "test"}},
        {"id": 2, "result": {"model": "gpt-6-luna", "modelProvider": "openai",
            "thread": {"id": "thread1"}, "runtimeWorkspaceRoots": [],
            "instructionSources": [], "sandbox": {"type": "readOnly", "networkAccess": False},
            "approvalPolicy": "never", "multiAgentMode": "explicitRequestOnly"}},
        {"id": 3, "result": {"turn": {"id": "turn1"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread1", "turnId": "turn1",
            "tokenUsage": {"total": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14}}}},
        {"method": "turn/completed", "params": {"threadId": "thread1",
            "turn": {"id": "turn1", "status": status, "items": [item]}}},
    ]


@pytest.mark.parametrize("payload, expected", [
    (required_call(9), {"tool": "echo", "refusal": None}),
    (required_refusal("No safe action."), {"tool": None, "refusal": "No safe action."}),
])
def test_native_turn_selects_required_action_or_refusal(monkeypatch, tmp_path, payload, expected):
    process = _FakeProcess(_native_events(payload))
    monkeypatch.setattr("adj_partner.models.codex_app_server.subprocess.Popen",
                        lambda *args, **kwargs: process)
    request = {"messages": request_messages(), "tools": [tool()],
               "max_output_tokens": 128, "timeout_seconds": 10,
               "tool_choice": "required_or_refuse"}
    output = run_native_turn(model="gpt-6-luna", effort="low", request=request,
                             trace_path=tmp_path / "events.jsonl", cli="/fake",
                             config_path=tmp_path / "missing.toml")
    assert output["selection"] == "first_required_decision"
    decoded = decode_envelope(output["text"], USAGE, [tool()],
                              tool_choice="required_or_refuse")
    assert (decoded.tool_calls[0].name if decoded.tool_calls else None) == expected["tool"]
    assert decoded.refusal == expected["refusal"]


def test_native_failed_turn_precedes_required_action_missing(monkeypatch, tmp_path):
    process = _FakeProcess(_native_events(required_refusal(), status="failed"))
    monkeypatch.setattr("adj_partner.models.codex_app_server.subprocess.Popen",
                        lambda *args, **kwargs: process)
    request = {"messages": request_messages(), "tools": [tool()],
               "max_output_tokens": 128, "timeout_seconds": 10,
               "tool_choice": "required_or_refuse"}
    with pytest.raises(ProviderFailure) as exc:
        run_native_turn(model="gpt-6-luna", effort="low", request=request,
                        trace_path=tmp_path / "events.jsonl", cli="/fake",
                        config_path=tmp_path / "missing.toml")
    assert exc.value.code == "codex_turn_failed"
    assert exc.value.usage == USAGE and exc.value.cost_unknown is False


def test_required_output_limit_is_explicit_and_charged(tmp_path):
    def runner(**kwargs):
        return runner_result(required_call(), usage={"input_tokens": 11, "output_tokens": 129,
                                                     "total_tokens": 140})
    port = CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, runner=runner)
    with pytest.raises(ProviderFailure) as exc:
        port.complete(request_messages(), [tool()], max_output_tokens=128,
                      timeout_seconds=10, tool_choice="required_or_refuse")
    assert exc.value.code == "codex_output_limit"
    assert exc.value.usage["output_tokens"] == 129
