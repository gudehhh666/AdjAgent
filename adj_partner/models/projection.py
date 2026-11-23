"""Allowlisted SDK reply projection; never dump a provider model wholesale."""

import json
import re

from ..contracts import ModelReply, ToolCall
from ..tools.catalog import canonical_json
from .base import ProviderFailure


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def public_usage(value):
    result = {}
    for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                           ("total_tokens", "total_tokens")):
        count = field(value, source)
        if type(count) is int and 0 <= count <= 2**63 - 1:
            result[target] = count
    return result


def cost_unknown(usage):
    return not {"input_tokens", "output_tokens"} <= usage.keys()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _invalid_constant(_):
    raise ValueError("nonfinite JSON")


def response_body(content):
    """Inspect bounded wire JSON before SDK coercion; never return it to callers."""
    if not isinstance(content, bytes) or len(content) > 8 * 1024 * 1024:
        raise ValueError("bounded response body required")
    value = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("response object required")
    return value


def arguments(value):
    if not isinstance(value, str) or len(value.encode("utf-8")) > 1024 * 1024:
        raise ValueError("bounded arguments string required")
    parsed = json.loads(value, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be an object")
    canonical_json(parsed)
    return parsed


def project_reply(response):
    usage = public_usage(field(response, "usage"))
    try:
        choices = field(response, "choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("one choice required")
        choice = choices[0]
        finish = field(choice, "finish_reason")
        if not isinstance(finish, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", finish):
            raise ValueError("finish reason required")
        message = field(choice, "message")
        if field(message, "role") != "assistant" or field(message, "function_call") is not None:
            raise ValueError("structured assistant message required")
        content, refusal = field(message, "content"), field(message, "refusal")
        if content is not None and not isinstance(content, str):
            raise ValueError("public content must be text")
        if refusal is not None and not isinstance(refusal, str):
            raise ValueError("refusal must be text")
        raw_calls = field(message, "tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list) or len(raw_calls) > 128:
            raise ValueError("bounded structured calls required")
        calls, seen = [], set()
        for call in raw_calls:
            call_id, function = field(call, "id"), field(call, "function")
            name = field(function, "name")
            if (field(call, "type") != "function" or not isinstance(call_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", call_id) or call_id in seen
                    or not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)):
                raise ValueError("valid distinct call identities required")
            seen.add(call_id)
            calls.append(ToolCall(call_id, name, arguments(field(function, "arguments"))))
        reply = ModelReply(content=content or "", tool_calls=calls, usage=usage, finish_reason=finish, refusal=refusal)
        if len(canonical_json(reply.to_dict()).encode()) > 4 * 1024 * 1024:
            raise ValueError("public reply exceeds size limit")
        return reply
    except (ValueError, TypeError, OverflowError, RecursionError, AttributeError):
        raise ProviderFailure("malformed_reply", "Provider returned malformed public reply fields or tool arguments",
                              usage=usage, cost_unknown=cost_unknown(usage)) from None
