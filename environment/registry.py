"""Tool Registry (Biomni-E1 style).

Every capability — retrieval, fusion, reliability, analysis, memory access — is
registered here as a Tool with a declarative schema. The control layer uses the
schemas for dynamic tool retrieval and invokes tools by name. Nothing about
*when* a tool runs lives here; that is the coordinator's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    domain: str                       # retrieval | fusion | reliability | analysis | data | memory | exec
    parameters: dict[str, str] = field(default_factory=dict)  # name -> description
    returns: str = ""


@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, spec: ToolSpec, func: Callable[..., Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = Tool(spec=spec, func=func)

    def tool(self, name: str, description: str, domain: str,
             parameters: dict[str, str] | None = None, returns: str = ""):
        """Decorator form."""
        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                ToolSpec(name=name, description=description, domain=domain,
                         parameters=parameters or {}, returns=returns),
                func,
            )
            return func
        return deco

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def invoke(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def list_specs(self, domain: str | None = None) -> list[ToolSpec]:
        specs = [t.spec for t in self._tools.values()]
        return [s for s in specs if domain is None or s.domain == domain]

    def catalog(self, domain: str | None = None) -> str:
        """Compact text catalog for the LLM tool retriever."""
        lines = []
        for s in self.list_specs(domain):
            params = ", ".join(f"{k}: {v}" for k, v in s.parameters.items())
            lines.append(f"- {s.name} [{s.domain}]: {s.description}"
                         + (f" (params: {params})" if params else ""))
        return "\n".join(lines)


# Global registry instance; tool modules register on import.
REGISTRY = ToolRegistry()
