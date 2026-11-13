"""Small public provider contracts; no execution, persistence or retry loop."""

from copy import deepcopy
from typing import Protocol, runtime_checkable

from ..contracts import ModelReply


class ProviderFailure(Exception):
    """Public safe failure, with independently known token counts if available.

    Adapter callers should persist only these fields, never SDK exceptions or
    their response bodies. An unknown bill is not a zero-cost request.
    """

    def __init__(self, code, message, *, usage=None, cost_unknown=True):
        super().__init__(message)
        self.code = code
        self.usage = {key: value for key, value in (usage or {}).items()
                      if key in {"input_tokens", "output_tokens", "total_tokens"}
                      and type(value) is int and value >= 0}
        self.cost_unknown = bool(cost_unknown)


def supports_tool_choice(port, tool_choice):
    """Required decisions are opt-in; a wrapper may not silently drop them."""
    if tool_choice == 'auto':
        return True
    if tool_choice != 'required_or_refuse':
        return False
    metadata = getattr(port, 'metadata', {})
    modes = metadata.get('tool_choice_modes') if isinstance(metadata, dict) else None
    return isinstance(modes, (list, tuple)) and 'required_or_refuse' in modes


@runtime_checkable
class ModelPort(Protocol):
    secrets: tuple[str, ...]

    @property
    def metadata(self) -> dict: ...

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds,
                 tool_choice='auto') -> ModelReply: ...


class ScriptedPort:
    """Explicit offline-only fixture; exhausted scripts fail, never improvise."""

    secrets = ()

    def __init__(self, replies):
        self._replies = iter(replies)
        self.requests = []

    @property
    def metadata(self):
        return {"provider_kind": "scripted", "model": "offline-script",
                "tool_choice_modes": ['auto', 'required_or_refuse']}

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds, tool_choice='auto'):
        from .requests import public_request
        request = public_request(messages, tools, max_output_tokens, timeout_seconds, tool_choice=tool_choice)
        self.requests.append(deepcopy(request))
        try:
            result = next(self._replies)
        except StopIteration:
            raise ProviderFailure("scripted_exhausted", "Explicit offline reply script is exhausted", cost_unknown=False) from None
        if isinstance(result, ProviderFailure):
            raise ProviderFailure(result.code, str(result), usage=result.usage, cost_unknown=result.cost_unknown) from None
        if not isinstance(result, ModelReply):
            raise ProviderFailure("scripted_reply", "Offline script must contain ModelReply or ProviderFailure", cost_unknown=False)
        return deepcopy(result)
