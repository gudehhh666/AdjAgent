"""Central config loader: merges YAML config with environment variables.

Secrets are never stored in YAML; only the *names* of env vars are. This keeps
keys out of git while the repo syncs to GitHub.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv optional; env may be set externally
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load_env() -> None:
    env_file = ROOT / ".env"
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file)


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (ROOT / relative).resolve()


def env(name: str, default: str | None = None) -> str | None:
    _load_env()
    return os.environ.get(name, default)


def get_setting(*keys: str, default: Any = None) -> Any:
    """Nested lookup into settings.yaml, e.g. get_setting('orchestration', 'effort')."""
    node: Any = load_yaml("settings.yaml")
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
