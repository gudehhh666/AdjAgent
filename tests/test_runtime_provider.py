"""Offline provider contracts and real installed-SDK HTTP fixture boundaries."""

import importlib
import importlib.util
import json

import httpx

import pytest

from adj_partner.contracts import ModelReply, ToolCall


@pytest.fixture(autouse=True)
def synthetic_sdk_environment(monkeypatch):
    # No actual ambient credentials are inspected by SDK fixture construction.
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_WEBHOOK_SECRET"):
        monkeypatch.setenv(name, "synthetic-ambient-unused")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer synthetic-ambient-unused\nX-Private: synthetic-ambient-unused")


def models():
    assert importlib.util.find_spec("adj_partner.models"), "new provider boundary must exist"
    return importlib.import_module("adj_partner.models")


MESSAGES = [{"role": "user", "content": "Synthetic fixture"}]


def test_scripted_port_is_explicit_records_independent_public_requests_and_exhausts():
    api = models()
    reply = ModelReply(content="Fixture", tool_calls=[ToolCall("call_1", "read", {"n": 1})],
                       usage={"input_tokens": 3, "output_tokens": 2})
    port = api.ScriptedPort([reply])
    messages = [{"role": "user", "content": "Original"}]
    result = port.complete(messages, [], max_output_tokens=17, timeout_seconds=2)
    messages[0]["content"] = "Changed"
    reply.tool_calls[0].arguments["n"] = 2
    assert result.tool_calls[0].arguments == {"n": 1}
    assert port.requests == [{"messages": [{"role": "user", "content": "Original"}], "tools": [],
                              "max_output_tokens": 17, "timeout_seconds": 2}]
    assert port.metadata == {"provider_kind": "scripted", "model": "offline-script",
                             "tool_choice_modes": ["auto", "required_or_refuse"]}
    assert port.secrets == () and isinstance(port, api.ModelPort)
    with pytest.raises(api.ProviderFailure) as error:
        port.complete(MESSAGES, [], max_output_tokens=17, timeout_seconds=2)
    assert error.value.code == "scripted_exhausted" and not error.value.cost_unknown


def test_refusal_is_additive_and_preserves_existing_defaults():
    assert hasattr(ModelReply(), "refusal"), "ModelReply must preserve explicit refusal"
    assert ModelReply().refusal is None
    assert ModelReply(refusal="Cannot comply").to_dict()["refusal"] == "Cannot comply"


@pytest.mark.parametrize("limit,value", [("max_output_tokens", 0), ("max_output_tokens", True),
    ("max_output_tokens", 1.2), ("timeout_seconds", 0), ("timeout_seconds", False),
    ("timeout_seconds", float("nan")), ("timeout_seconds", float("inf"))])
def test_invalid_budgets_fail_before_consuming_script(limit, value):
    api = models()
    port = api.ScriptedPort([ModelReply(content="untouched")])
    limits = {"max_output_tokens": 10, "timeout_seconds": 1, limit: value}
    with pytest.raises(api.ProviderFailure) as error:
        port.complete(MESSAGES, [], **limits)
    assert error.value.code == "invalid_request" and not error.value.cost_unknown
    assert port.requests == []
    assert port.complete(MESSAGES, [], max_output_tokens=10, timeout_seconds=1).content == "untouched"


@pytest.mark.parametrize("messages,tools", [([], []), ([{"role": "user", "content": float("nan")}], []),
    ([{"role": "user", "content": "ok", "reasoning_content": "not public"}], []),
    (MESSAGES, [{"type": "function", "function": {"name": "tool", "parameters": {"default": float("inf")}}}])])
def test_invalid_or_nonpublic_requests_fail_locally(messages, tools):
    api = models()
    port = api.ScriptedPort([ModelReply(content="untouched")])
    with pytest.raises(api.ProviderFailure) as error:
        port.complete(messages, tools, max_output_tokens=10, timeout_seconds=1)
    assert error.value.code == "invalid_request" and not error.value.cost_unknown
    assert port.requests == []


def test_scripted_failures_are_explicit_not_successful_placeholder_replies():
    api = models()
    expected = api.ProviderFailure("offline_error", "Fixture failure", usage={"input_tokens": 7}, cost_unknown=True)
    port = api.ScriptedPort([expected])
    with pytest.raises(api.ProviderFailure) as error:
        port.complete(MESSAGES, [], max_output_tokens=10, timeout_seconds=1)
    assert error.value.code == "offline_error" and error.value.usage == {"input_tokens": 7}
    assert error.value.cost_unknown


def completion(**changes):
    result = {"id": "chatcmpl_fixture", "object": "chat.completion", "created": 1700000000,
              "model": "fixture-model", "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None,
                  "message": {"role": "assistant", "content": "Fixture reply"}}],
              "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    result.update(changes)
    return result


def sdk_port(response=None, handler=None, **options):
    api = models()
    assert hasattr(api, "OpenAICompatiblePort"), "single-call installed SDK adapter must exist"
    requests = []

    def respond(request):
        requests.append(request)
        return handler(request) if handler else httpx.Response(200, json=response or completion())

    port = api.OpenAICompatiblePort(api_key="synthetic-key-not-real", base_url="https://provider.invalid/v1",
                                    model="fixture-model", transport=httpx.MockTransport(respond), **options)
    return port, requests


def invoke(port):
    return port.complete(MESSAGES, [], max_output_tokens=123, timeout_seconds=1.25)


def test_actual_sdk_serializes_tools_auth_and_limits_once_without_private_projection():
    body = completion()
    body["reasoning_content"] = "PRIVATE_TOP_LEVEL"
    body["choices"][0]["message"].update(content=None, reasoning_content="PRIVATE_MESSAGE",
        tool_calls=[{"id": "call-real-id", "type": "function", "function": {"name": "read", "arguments": '{"n":2,"word":"中文"}'}}])
    body["choices"][0]["finish_reason"] = "tool_calls"
    port, requests = sdk_port(body)
    schema = [{"type": "function", "function": {"name": "read", "description": "Fixture", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}}, "additionalProperties": False}}}]
    result = port.complete(MESSAGES, schema, max_output_tokens=123, timeout_seconds=1.25)
    assert len(requests) == 1
    request = requests[0]
    sent = json.loads(request.content)
    assert sent["model"] == "fixture-model" and sent["messages"] == MESSAGES and sent["tools"] == schema
    assert sent["max_completion_tokens"] == 123 and "max_tokens" not in sent and not sent.get("stream")
    assert set(request.extensions["timeout"].values()) == {1.25}
    assert request.headers["authorization"] == "Bearer synthetic-key-not-real"
    assert "x-private" not in request.headers and "openai-organization" not in request.headers
    assert str(request.url) == "https://provider.invalid/v1/chat/completions"
    assert result.tool_calls == [ToolCall("call-real-id", "read", {"n": 2, "word": "中文"})]
    assert result.finish_reason == "tool_calls" and result.content == ""
    assert result.usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert "PRIVATE" not in json.dumps(result.to_dict())
    assert port.secrets == ("synthetic-key-not-real",)
    assert port.metadata == {"provider_kind": "openai_compatible", "model": "fixture-model"}
    assert "provider.invalid" not in repr(port.metadata)
    port.close()


def test_refusal_and_finish_are_preserved_separately():
    body = completion()
    body["choices"][0].update(finish_reason="content_filter")
    body["choices"][0]["message"].update(content=None, refusal="Cannot assist")
    port, _ = sdk_port(body)
    result = invoke(port)
    assert result.refusal == "Cannot assist" and result.finish_reason == "content_filter" and result.content == ""
    port.close()


@pytest.mark.parametrize("arguments", ["", "{", "[]", "null", '{"x":NaN}', '{"x":Infinity}',
    '{"x":1,"x":2}', '{"x":"\\ud800"}', '{"x":1e400}'])
def test_malformed_tool_json_is_failure_preserving_usage_not_empty_arguments(arguments):
    body = completion()
    body["choices"][0].update(finish_reason="tool_calls")
    body["choices"][0]["message"].update(content=None, tool_calls=[{
        "id": "call_1", "type": "function", "function": {"name": "read", "arguments": arguments}}])
    port, requests = sdk_port(body)
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.code == "malformed_reply" and not error.value.cost_unknown
    assert error.value.usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert len(requests) == 1
    port.close()


@pytest.mark.parametrize("case", ["empty_id", "duplicate_id", "control_id", "bad_name", "wrong_type",
    "missing_finish", "wrong_content", "wrong_refusal", "empty_choices", "multiple_choices", "wrong_role"])
def test_malformed_public_reply_shapes_fail_closed_with_known_usage(case):
    body = completion()
    choice = body["choices"][0]
    message = choice["message"]
    call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    message["tool_calls"] = [call]
    choice["finish_reason"] = "tool_calls"
    if case == "empty_id": call["id"] = ""
    elif case == "duplicate_id": message["tool_calls"].append(dict(call))
    elif case == "control_id": call["id"] = "id\nforged"
    elif case == "bad_name": call["function"]["name"] = "not/a/tool"
    elif case == "wrong_type": call["type"] = "custom"
    elif case == "missing_finish": choice.pop("finish_reason")
    elif case == "wrong_content": message["content"] = ["not text"]
    elif case == "wrong_refusal": message["refusal"] = 17
    elif case == "empty_choices": body["choices"] = []
    elif case == "multiple_choices": body["choices"].append(choice.copy())
    elif case == "wrong_role": message["role"] = "user"
    port, _ = sdk_port(body)
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.code == "malformed_reply" and error.value.usage["input_tokens"] == 7
    port.close()


@pytest.mark.parametrize("usage,expected", [(None, {}), ({}, {}),
    ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    ({"prompt_tokens": 7}, {"input_tokens": 7}),
    ({"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": "7"}, {"output_tokens": 2}),
    ({"prompt_tokens": True, "completion_tokens": 2.5}, {})])
def test_missing_or_invalid_usage_is_not_invented_zero(usage, expected):
    port, _ = sdk_port(completion(usage=usage))
    assert invoke(port).usage == expected
    port.close()


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (429, 4), (500, 1), (503, 4)])
def test_http_errors_retry_only_when_unprocessed_and_never_leak(status, attempts):
    body = {"error": {"message": "synthetic-key-not-real at https://provider.invalid PRIVATE_REASONING"}}
    port, requests = sdk_port(handler=lambda _: httpx.Response(status, json=body, headers={"retry-after": "0"}),
                              sleep=lambda _: None)
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert len(requests) == attempts and error.value.code == "provider_http"
    assert error.value.usage == {} and error.value.cost_unknown
    assert "synthetic-key" not in str(error.value) and "provider.invalid" not in str(error.value)
    assert "PRIVATE" not in str(error.value)
    port.close()


def test_error_body_known_usage_is_preserved_without_other_body_fields():
    port, requests = sdk_port(handler=lambda _: httpx.Response(400, json={
        "error": {"message": "PRIVATE"}, "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}}))
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.usage == {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13}
    assert not error.value.cost_unknown and len(requests) == 1
    port.close()


def test_sdk_timeout_is_single_attempt_with_unknown_remote_cost():
    def timeout(request):
        raise httpx.ReadTimeout("synthetic-key-not-real PRIVATE", request=request)
    port, requests = sdk_port(handler=timeout)
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.code == "provider_timeout" and error.value.cost_unknown and len(requests) == 1
    assert "synthetic-key" not in str(error.value)
    port.close()


def test_redirect_is_not_followed_to_another_endpoint():
    port, requests = sdk_port(handler=lambda _: httpx.Response(307, headers={"location": "https://other.invalid/target"}))
    with pytest.raises(models().ProviderFailure):
        invoke(port)
    assert len(requests) == 1
    port.close()


@pytest.mark.parametrize("status", [200, 500])
def test_wire_body_limit_stops_reading_before_full_sdk_buffering(status):
    class Body(httpx.SyncByteStream):
        chunks = 0
        closed = False

        def __iter__(self):
            for _ in range(256):
                self.chunks += 1
                yield b"x" * 65536

        def close(self):
            self.closed = True
    body = Body()
    port, requests = sdk_port(handler=lambda _: httpx.Response(status, stream=body))
    with pytest.raises(models().ProviderFailure):
        invoke(port)
    assert body.chunks <= 129 and body.closed and len(requests) == 1
    port.close()


def test_elapsed_response_deadline_stops_trickle_even_when_each_read_is_fast(monkeypatch):
    import time
    elapsed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])

    class Body(httpx.SyncByteStream):
        chunks = 0
        closed = False

        def __iter__(self):
            content = json.dumps(completion()).encode()
            for offset in range(0, len(content), 30):
                elapsed[0] += .4
                self.chunks += 1
                yield content[offset:offset+30]

        def close(self):
            self.closed = True
    body = Body()
    port, requests = sdk_port(handler=lambda _: httpx.Response(200, stream=body))
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.code == "provider_timeout" and error.value.cost_unknown
    assert body.chunks <= 4 and body.closed and len(requests) == 1
    port.close()


def test_response_requires_identity_encoding_to_bound_decompressed_memory():
    import gzip
    port, requests = sdk_port(handler=lambda _: httpx.Response(200,
        headers={"content-encoding": "gzip"}, content=gzip.compress(json.dumps(completion()).encode())))
    with pytest.raises(models().ProviderFailure):
        invoke(port)
    assert len(requests) == 1 and requests[0].headers["accept-encoding"] == "identity"
    port.close()


@pytest.mark.parametrize("messages,tools", [
    ([{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "reasoning_content": "PRIVATE",
       "function": {"name": "read", "arguments": "{}"}}]}], []),
    (MESSAGES, [{"type": "function", "function": {"name": "read", "parameters": {}, "private_reasoning": "PRIVATE"}}]),
    (MESSAGES, [{"type": "custom", "function": {"name": "read", "parameters": {}}}]),
    (MESSAGES, [{"type": "function", "function": {"name": "read", "parameters": []}}]),
    ([{"role": "tool", "content": "orphan without identity"}], []),
    ([{"role": "user", "content": None}], []),
])
def test_request_nested_protocol_fields_are_allowlisted_before_any_sdk_request(messages, tools):
    port, requests = sdk_port()
    with pytest.raises(models().ProviderFailure) as error:
        port.complete(messages, tools, max_output_tokens=10, timeout_seconds=1)
    assert error.value.code == "invalid_request" and not error.value.cost_unknown and requests == []
    port.close()


def test_valid_complete_tool_group_serializes_without_changing_id_or_arguments():
    port, requests = sdk_port()
    messages = [*MESSAGES, {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"n":1}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'}]
    port.complete(messages, [], max_output_tokens=10, timeout_seconds=1)
    assert json.loads(requests[0].content)["messages"] == messages
    port.close()


def test_http_error_nested_known_usage_is_not_erased_by_missing_outer_usage():
    port, _ = sdk_port(handler=lambda _: httpx.Response(400, json={"error": {"message": "PRIVATE",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}}}))
    with pytest.raises(models().ProviderFailure) as error:
        invoke(port)
    assert error.value.usage == {"input_tokens": 5, "output_tokens": 2}
    assert not error.value.cost_unknown
    port.close()
