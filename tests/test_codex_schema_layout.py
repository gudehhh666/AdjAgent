"""Layout treatment is explicit, fail-closed and absent from model input text."""
from copy import deepcopy
import hashlib
import json

import pytest

from adj_partner.models import codex_app_server as native
from adj_partner.models.base import ProviderFailure
from adj_partner.tools.catalog import canonical_json, validator


def tools():
    return [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {
            "query": {"type": "string", "description": "Synthetic description. " * 20},
            **{f"option{i}": {"type": ["string", "null"], "description": "Optional metadata. " * 12}
               for i in range(4)}},
        "required": ["query"], "additionalProperties": False}}}]


@pytest.mark.parametrize("choice", ["auto", "required_or_refuse"])
def test_opt_in_layout_is_smaller_and_preserves_host_argument_semantics(choice):
    original = tools()
    before = deepcopy(original)
    baseline = native.envelope_schema(original, tool_choice=choice)
    explicit = native.envelope_schema(original, tool_choice=choice, schema_layout="expanded")
    compact = native.envelope_schema(original, tool_choice=choice, schema_layout="refs-v1")
    assert baseline == explicit
    assert "$defs" in compact
    assert len(canonical_json(compact)) < len(canonical_json(baseline)) / 2
    for arguments in ({"query": "x"}, {"query": "x", "option0": None},
                      {"query": "x", "option0": ""}, {"query": None},
                      {"query": "x", "extra": 1}, {"query": "x", "option0": 0}):
        calls = [{"name": "lookup", "arguments": arguments}]
        body = {"content": "", "tool_calls": calls} if choice == "auto" else {"decision": {"tool_calls": calls}}
        assert validator(compact).is_valid(body) == validator(baseline).is_valid(body)
    assert original == before


@pytest.mark.parametrize("layout", ["unknown", None, {}, 1])
def test_bad_layout_rejected_before_any_receipt_or_dispatch(tmp_path, layout):
    calls = []
    with pytest.raises(ProviderFailure) as error:
        native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
            schema_layout=layout, runner=lambda **kw: calls.append(kw))
    assert error.value.code == "invalid_request"
    assert error.value.cost_unknown is False
    assert not calls and not list(tmp_path.rglob("receipt.json"))


def test_receipt_binds_layout_compiler_and_exact_emitted_schema(tmp_path):
    captured = []
    def run(**kwargs):
        captured.append(kwargs)
        return {"model": "gpt-6-luna", "status": "completed", "native_activity": False,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "text": '{"content":"done","tool_calls":[]}'}
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
                                    schema_layout="refs-v1", runner=run)
    port.complete([{"role": "user", "content": "Synthetic."}], tools(),
                  max_output_tokens=128, timeout_seconds=5)
    request = captured[0]["request"]
    receipt = json.loads(next(tmp_path.glob("*/receipt.json")).read_text())
    encoded = canonical_json(native.envelope_schema(request["tools"], schema_layout="refs-v1")).encode()
    assert request["schema_layout"] == receipt["schema_layout"] == "refs-v1"
    assert receipt["output_schema_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert receipt["output_schema_bytes"] == len(encoded)
    assert receipt["output_schema_encoding"] == "canonical-json-utf8"
    assert receipt["schema_layout_effective"] == "refs-v1"
    assert receipt["schema_compiler"] == port.metadata["schema_compiler"] == "closed-exact-keyset-local-refs-v1"


def test_no_saving_receipt_discloses_expanded_fallback(tmp_path):
    def run(**kwargs):
        return {"model": "gpt-6-luna", "status": "completed", "native_activity": False,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "text": '{"content":"done","tool_calls":[]}'}
    port = native.CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path,
                                    schema_layout="refs-v1", runner=run)
    port.complete([{"role": "user", "content": "Synthetic."}], [],
                  max_output_tokens=128, timeout_seconds=5)
    receipt = json.loads(next(tmp_path.glob("*/receipt.json")).read_text())
    assert receipt["schema_layout"] == "refs-v1"
    assert receipt["schema_layout_effective"] == "expanded"


def test_both_layouts_send_identical_public_text_and_instructions(monkeypatch, tmp_path):
    from test_codex_app_server_port import fake_process, FINAL
    captured = []
    original_dumps = json.dumps
    def record(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("method") in {"thread/start", "turn/start"}:
            captured.append(deepcopy(value))
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(native.json, "dumps", record)
    for layout in ("expanded", "refs-v1"):
        kwargs = fake_process(monkeypatch, tmp_path, [FINAL])
        kwargs["request"]["schema_layout"] = layout
        kwargs["request"]["tools"] = tools()
        native.run_native_turn(**kwargs)
    threads = [v["params"] for v in captured if v["method"] == "thread/start"]
    turns = [v["params"] for v in captured if v["method"] == "turn/start"]
    assert threads[0]["baseInstructions"] == threads[1]["baseInstructions"]
    assert threads[0]["developerInstructions"] == threads[1]["developerInstructions"]
    assert turns[0]["input"] == turns[1]["input"]
    assert "schema_layout" not in json.loads(turns[0]["input"][0]["text"])
    assert turns[0]["outputSchema"] != turns[1]["outputSchema"]


def test_refs_cannot_bypass_original_expansion_limits():
    parameters = {"type": "object", "properties": {
        "required": {"type": "string", "description": "x" * 20000},
        **{f"o{i}": {"type": "integer"} for i in range(7)}},
        "required": ["required"], "additionalProperties": False}
    with pytest.raises(ProviderFailure) as error:
        native.envelope_schema([{"type": "function", "function": {
            "name": "large", "parameters": parameters}}], schema_layout="refs-v1")
    assert error.value.code == "invalid_request" and error.value.cost_unknown is False
