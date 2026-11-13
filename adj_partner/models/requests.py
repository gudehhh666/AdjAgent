"""Bounded finite public request snapshots shared by real and scripted ports."""

import json
import math
import re

from ..tools.catalog import canonical_json
from .base import ProviderFailure
from .projection import arguments


def _identifier(value, pattern=r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _calls(calls):
    if not isinstance(calls, list) or len(calls) > 128:
        raise ValueError("bounded tool calls")
    seen = set()
    for call in calls:
        if (not isinstance(call, dict) or set(call) != {"id", "type", "function"}
                or call["type"] != "function" or not _identifier(call["id"]) or call["id"] in seen):
            raise ValueError("public call")
        seen.add(call["id"])
        function = call["function"]
        if (not isinstance(function, dict) or set(function) != {"name", "arguments"}
                or not _identifier(function["name"], r"[A-Za-z0-9_-]{1,64}")):
            raise ValueError("public function call")
        arguments(function["arguments"])


def _tools(tools):
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
            raise ValueError("public tool definition")
        function = tool["function"]
        if (not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}
                or not _identifier(function.get("name"), r"[A-Za-z0-9_-]{1,64}")
                or not isinstance(function.get("parameters"), dict)
                or "description" in function and not isinstance(function["description"], str)
                or "strict" in function and type(function["strict"]) is not bool):
            raise ValueError("public function definition")


def public_request(messages, tools, max_output_tokens, timeout_seconds, *, tool_choice='auto'):
    try:
        if (not isinstance(tool_choice, str) or tool_choice not in ('auto', 'required_or_refuse')
                or tool_choice == 'required_or_refuse' and not tools):
            raise ValueError('tool choice requires a supported mode and available tools')
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("output limit")
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("timeout")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 4096:
            raise ValueError("messages")
        if not isinstance(tools, list) or len(tools) > 256:
            raise ValueError("tools")
        for message in messages:
            if (not isinstance(message, dict) or set(message) - {"role", "content", "name", "tool_calls", "tool_call_id", "refusal"}
                    or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
                    or message.get("content") is not None and not isinstance(message["content"], str)):
                raise ValueError("public message")
            role = message["role"]
            allowed = {"role", "content", "name"}
            if role == "assistant":
                allowed |= {"tool_calls", "refusal"}
                if message.get("tool_calls") is not None:
                    _calls(message["tool_calls"])
                if message.get("refusal") is not None and not isinstance(message["refusal"], str):
                    raise ValueError("public refusal")
            elif not isinstance(message.get("content"), str):
                raise ValueError("text content required")
            if role == "tool":
                allowed.add("tool_call_id")
                if not _identifier(message.get("tool_call_id")):
                    raise ValueError("tool result identity required")
            if set(message) - allowed or "name" in message and not isinstance(message["name"], str):
                raise ValueError("role-specific public fields")
        _tools(tools)
        request = {"messages": messages, "tools": tools, "max_output_tokens": max_output_tokens,
                   "timeout_seconds": timeout_seconds}
        if tool_choice != 'auto':
            request['tool_choice'] = tool_choice
        encoded = canonical_json(request)
        if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("request size")
        return json.loads(encoded)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ProviderFailure("invalid_request", "Provider request must contain bounded public finite JSON and positive finite limits", cost_unknown=False) from None
