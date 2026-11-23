"""Installed SDK -> single engine -> real executor, with fixture HTTP only."""

import json

import httpx
import pytest

from adj_partner.contracts import SessionConfig, ToolResult
from adj_partner.models import OpenAICompatiblePort
from adj_partner.runtime.engine import AgentEngine
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec


def response(content="done", calls=None, usage=None):
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": {
        "role": "assistant", "content": content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3} if usage is None else usage}


@pytest.mark.parametrize("malformed", [False, True])
def test_installed_sdk_reply_flows_through_durable_action_and_failure_accounting(tmp_path, monkeypatch, malformed):
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_WEBHOOK_SECRET", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.setenv(key, "")
    requests, effects = [], []
    call = {"id": "original-id", "type": "function", "function": {"name": "echo", "arguments": "{" if malformed else '{"text":"actual"}'}}
    def http(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(None, [call]) if len(requests) == 1 else response())
    port = OpenAICompatiblePort(api_key="synthetic-host-credential", base_url="https://fixture.invalid/v1", model="fixture",
                               transport=httpx.MockTransport(http))
    store, catalog = Store(tmp_path / "store"), ToolCatalog()
    def echo(ctx, text):
        effects.append(text)
        return ToolResult(data=text)
    catalog.register(ToolSpec("echo", "fixture", {"type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False}), echo)
    engine = AgentEngine(store, catalog, port)
    sid = store.create_session(SessionConfig(enabled_tools=catalog.names(), allowed_tools=catalog.names()).to_dict())["id"]
    rid = engine.start(sid, "Check synthetic-host-credential is scrubbed")["id"]
    done = engine.resume(rid)
    assert "synthetic-host-credential" not in json.dumps([store.events(sid), requests, done])
    if malformed:
        assert done["status"] == "failed" and effects == [] and len(requests) == 1
        assert done["usage"]["input_tokens"] == 7 and done["usage"]["unknown_requests"] == 0
        assert done["state"]["error"]["code"] == "malformed_reply"
    else:
        assert done["status"] == "succeeded" and effects == ["actual"] and len(requests) == 2
        assert done["usage"]["input_tokens"] == 14
        assert store.get_action(rid, "original-id")["finished_at"]
    port.close()
