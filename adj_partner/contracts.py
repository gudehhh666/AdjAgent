"""Small JSON-serializable contracts shared by the host, providers, and tools."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any
from uuid import uuid4

from .defaults import RUNTIME_DEFAULTS


def new_id(prefix: str) -> str:
    """Generate host-owned identifiers; never accept model-supplied object IDs."""
    if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", prefix):
        raise ValueError("ID prefix must be a short lowercase identifier")
    return f"{prefix}_{uuid4().hex}"


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCall(_Serializable):
    id: str
    name: str
    arguments: dict


@dataclass
class ModelReply(_Serializable):
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    refusal: str | None = None


@dataclass
class ToolResult(_Serializable):
    status: str = "ok"
    data: Any = None
    artifacts: list[dict] = field(default_factory=list)
    error: dict | None = None
    job_id: str | None = None


@dataclass
class Budget(_Serializable):
    max_steps: int = RUNTIME_DEFAULTS["max_steps"]
    max_input_tokens: int = RUNTIME_DEFAULTS["max_input_tokens"]
    max_output_tokens: int = RUNTIME_DEFAULTS["max_output_tokens"]
    wall_time_seconds: float = RUNTIME_DEFAULTS["wall_time_seconds"]

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        value = self.wall_time_seconds
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("wall_time_seconds must be finite and positive")


@dataclass
class SessionConfig(_Serializable):
    instructions: str = ""
    enabled_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    profile: str = "general"
    allow_network: bool = False
    allowed_roots: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)

    @classmethod
    def from_dict(cls, value: dict) -> "SessionConfig":
        config = deepcopy(value)
        config["budget"] = Budget(**config.get("budget", {}))
        return cls(**config)
