"""Procedural memory access: list and read Know-How cards (SOPs, immune modules)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import get_setting, resolve_path
from environment.registry import REGISTRY, ToolSpec


def _dir() -> Path:
    return resolve_path(get_setting("knowhow_dir", default="environment/knowhow"))


def get_knowhow(name: str | None = None) -> dict[str, Any]:
    d = _dir()
    if not d.exists():
        return {"cards": [], "content": ""}
    cards = sorted(p.name for p in d.iterdir() if p.is_file())
    if name is None:
        return {"cards": cards, "content": ""}
    target = d / name
    if not target.exists():
        # fuzzy match by stem
        for p in d.iterdir():
            if name.lower() in p.name.lower():
                target = p
                break
    content = target.read_text(encoding="utf-8") if target.exists() else ""
    return {"cards": cards, "content": content, "name": target.name if target.exists() else None}


REGISTRY.register(
    ToolSpec(name="get_knowhow",
             description="List or read Know-How cards (adjuvant SOPs, immune modules). "
                         "Call without name to list; with a name to read a card.",
             domain="memory",
             parameters={"name": "card filename (optional)"},
             returns="dict with cards list and content"),
    get_knowhow,
)
