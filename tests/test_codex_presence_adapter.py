"""The opt-in wire codec preserves host semantics and matched control inputs."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from adj_partner.models import codex_app_server as native
from adj_partner.models.base import ProviderFailure
from adj_partner.models.schema_presence import encode
from adj_partner.tools.catalog import canonical_json, validator

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
LAYOUTS = ("expanded-codec-v1", "presence-v1")


def tool():
    return {"type": "function", "function": {"name": "probe", "parameters": {
        "type": "object", "properties": {
            "code": {"type": "string"},
            "missing": {"type": "string"},
            "maybe": {"type": ["string", "null"]},
            "nested": {"type": "array", "items": {"type": "object", "properties": {
                "flag": {"type": "boolean"}, "number": {"type": "number"},
                "absent": {"type": "string"}}, "required": ["flag"], "additionalProperties": False}}},
        "required": ["code"], "additionalProperties": False}}}


def args():
    return {"code": 'print("quoted\\path")\n# Ω\n', "maybe": None,
            "nested": [{"flag": False, "number": -0.0}]}


def body(layout="presence-v1", choice="auto", arguments=None):
    original = args() if arguments is None else arguments
    wire = encode(original, tool()["function"]["parameters"]) if layout == "presence-v1" else original
    calls = [{"name": "probe", "arguments": wire}]
    return {"content": "", "tool_calls": calls} if choice == "auto" else {"decision": {"tool_calls": calls}}


@pytest.mark.parametrize("choice", ["auto", "required_or_refuse"])
def test_full_wire_schema_and_decoded_host_arguments_are_exact(choice):
    tools = [tool()]
    old = deepcopy(tools)
    for layout in LAYOUTS:
        wire = body(layout, choice)
        schema = native.envelope_schema(tools, tool_choice=choice, schema_layout=layout)
        assert validator(schema).is_valid(wire)
        reply = native.decode_envelope(json.dumps(wire), USAGE, tools,
                                       tool_choice=choice, schema_layout=layout)
        assert canonical_json(reply.tool_calls[0].arguments) == canonical_json(args())
        assert math.copysign(1, reply.tool_calls[0].arguments["nested"][0]["number"]) == -1
        assert reply.usage == USAGE
    assert tools == old
    assert native.envelope_schema(tools, tool_choice=choice, schema_layout=LAYOUTS[0]) == native.envelope_schema(tools, tool_choice=choice)


@pytest.mark.parametrize("mutation", ["omit_wrapper", "false_has_value", "numeric_present", "unknown_tool", "duplicate_key", "nonfinite", "surrogate"])
def test_malformed_codec_reply_keeps_known_usage(mutation):
    wire = body()
    arguments = wire["tool_calls"][0]["arguments"]
    if mutation == "omit_wrapper": del arguments["missing"]
    if mutation == "false_has_value": arguments["missing"] = {"present": False, "value": "x"}
    if mutation == "numeric_present": arguments["maybe"]["present"] = 1
    if mutation == "unknown_tool": wire["tool_calls"][0]["name"] = "other"
    if mutation == "nonfinite": arguments["nested"]["value"][0]["number"]["value"] = float("nan")
    if mutation == "surrogate": arguments["code"] = "\ud800"
    text = json.dumps(wire)
    if mutation == "duplicate_key": text = text.replace('"present": false', '"present": false, "present": true', 1)
    with pytest.raises(ProviderFailure) as failure:
        native.decode_envelope(text, USAGE, [tool()], schema_layout="presence-v1")
    assert failure.value.cost_unknown is False and failure.value.usage == USAGE


@pytest.mark.parametrize("layout", LAYOUTS)
def test_original_aggregate_limit_still_blocks_before_dispatch(tmp_path, layout):
    parameters = {"type": "object", "properties": {f"x{i}": {"type": "integer"} for i in range(8)},
                  "required": [], "additionalProperties": False}
    tools = [{"type": "function", "function": {"name": "too_many", "parameters": parameters}}]
    calls = []
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
                                    schema_layout=layout, runner=lambda **kw: calls.append(kw))
    with pytest.raises(ProviderFailure) as failure:
        port.complete([{"role": "user", "content": "synthetic"}], tools,
                      max_output_tokens=128, timeout_seconds=5)
    assert failure.value.code == "invalid_request" and failure.value.cost_unknown is False
    assert not calls and not list(tmp_path.rglob("receipt.json"))


def test_generated_extra_depth_is_rejected_instead_of_fallback():
    schema = {"type": "object", "properties": {"optional": {"type": "string"}}, "required": [], "additionalProperties": False}
    for _ in range(6):
        schema = {"type": "object", "properties": {"child": schema}, "required": ["child"], "additionalProperties": False}
    tools = [{"type": "function", "function": {"name": "deep", "parameters": schema}}]
    native.envelope_schema(tools, schema_layout="expanded")
    with pytest.raises(ProviderFailure) as failure:
        native.envelope_schema(tools, schema_layout="presence-v1")
    assert failure.value.code == "invalid_request"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_receipt_binds_raw_wire_decoded_reply_and_codec_identity(tmp_path, layout):
    text = json.dumps(body(layout))
    def run(**kwargs):
        return {"model": "gpt-6-luna", "status": "completed", "native_activity": False,
                "usage": USAGE, "text": text}
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, schema_layout=layout, runner=run)
    reply = port.complete([{"role": "user", "content": "synthetic"}], [tool()], max_output_tokens=128, timeout_seconds=5)
    receipt_path = next(tmp_path.glob("*/receipt.json")); receipt = json.loads(receipt_path.read_text())
    for filename, key in (("wire_reply.json", "wire_reply_sha256"), ("reply.json", "decoded_reply_sha256")):
        assert hashlib.sha256((receipt_path.parent / filename).read_bytes()).hexdigest() == receipt[key]
    assert json.loads((receipt_path.parent / "wire_reply.json").read_text()) == {"text": text}
    assert reply.tool_calls[0].arguments == args()
    assert receipt["transport_version"] == port.metadata["transport_version"] == "0.9"
    assert receipt["instruction_profile"] == port.metadata["instruction_profile"] == "codec-aware-v1"
    assert receipt["argument_codec"] == ("identity" if layout == LAYOUTS[0] else "presence-v1")
    assert receipt["schema_layout_effective"] == ("expanded" if layout == LAYOUTS[0] else "presence-v1")


def test_invalid_wire_is_preserved_with_failed_known_cost_receipt(tmp_path):
    text = json.dumps(body()).replace('"present": false', '"present": 0', 1)
    def run(**kwargs):
        return {"model": "gpt-6-luna", "status": "completed", "native_activity": False, "usage": USAGE, "text": text}
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, schema_layout="presence-v1", runner=run)
    with pytest.raises(ProviderFailure):
        port.complete([{"role": "user", "content": "synthetic"}], [tool()], max_output_tokens=128, timeout_seconds=5)
    receipt_path = next(tmp_path.glob("*/receipt.json")); receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed" and receipt["cost_unknown"] is False and receipt["usage"] == USAGE
    assert receipt["wire_reply_sha256"] == hashlib.sha256((receipt_path.parent / "wire_reply.json").read_bytes()).hexdigest()
    assert not (receipt_path.parent / "reply.json").exists()


@pytest.mark.parametrize("choice", ["auto", "required_or_refuse"])
def test_new_control_and_codec_match_native_input_instructions_and_selection(monkeypatch, tmp_path, choice):
    from test_codex_app_server_port import fake_process
    captured = []; original_dumps = json.dumps
    def record(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("method") in {"thread/start", "turn/start"}:
            captured.append(deepcopy(value))
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(native.json, "dumps", record)
    for layout in LAYOUTS:
        item = {"id": "action", "type": "agentMessage", "phase": "final_answer", "text": json.dumps(body(layout, choice))}
        kwargs = fake_process(monkeypatch, tmp_path, [item])
        kwargs["request"].update(tools=[tool()], schema_layout=layout, tool_choice=choice)
        output = native.run_native_turn(**kwargs)
        assert output["selected_message_id"] == "action"
        assert output["selection"] == ("first_required_decision" if choice == "required_or_refuse" else "first_host_tool_request")
    threads = [v["params"] for v in captured if v["method"] == "thread/start"]
    turns = [v["params"] for v in captured if v["method"] == "turn/start"]
    assert threads[0]["baseInstructions"] == threads[1]["baseInstructions"]
    assert threads[0]["developerInstructions"] == threads[1]["developerInstructions"]
    assert "present" in threads[0]["baseInstructions"]
    assert turns[0]["input"] == turns[1]["input"]
    assert turns[0]["outputSchema"] != turns[1]["outputSchema"]


def test_presence_refusal_and_final_no_tools_are_identity_outputs():
    refusal = native.decode_envelope('{"decision":{"refusal":"missing input"}}', USAGE, [tool()], tool_choice="required_or_refuse", schema_layout="presence-v1")
    assert refusal.refusal == "missing input" and not refusal.tool_calls
    final = native.decode_envelope('{"content":"done","tool_calls":[]}', USAGE, [], schema_layout="presence-v1")
    assert final.content == "done" and not final.tool_calls


def test_wire_arguments_size_bound_is_not_bypassed_by_codec():
    text = json.dumps(body(arguments={"code": "x" * (1024 * 1024)}))
    with pytest.raises(ProviderFailure) as failure:
        native.decode_envelope(text, USAGE, [tool()], schema_layout="presence-v1")
    assert failure.value.cost_unknown is False


@pytest.mark.parametrize("choice", ["auto", "required_or_refuse"])
def test_invalid_unicode_in_final_content_or_refusal_is_known_cost_failure(tmp_path, choice):
    wire = ({"content": "\ud800", "tool_calls": []} if choice == "auto" else
            {"decision": {"refusal": "\ud800"}})
    def run(**kwargs):
        return {"model": "gpt-6-luna", "status": "completed", "native_activity": False,
                "usage": USAGE, "text": json.dumps(wire)}
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
                                    schema_layout="presence-v1", runner=run)
    with pytest.raises(ProviderFailure) as failure:
        port.complete([{"role": "user", "content": "synthetic"}], [tool()],
                      max_output_tokens=128, timeout_seconds=5, tool_choice=choice)
    assert failure.value.cost_unknown is False and failure.value.usage == USAGE
    receipt = json.loads(next(tmp_path.glob("*/receipt.json")).read_text())
    assert receipt["cost_unknown"] is False and receipt["status"] == "failed"
