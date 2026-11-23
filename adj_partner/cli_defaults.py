"""Allowlisted numeric CLI defaults from process or an explicit dotenv file."""

import io
import math
import os
from pathlib import Path

from dotenv import dotenv_values

from .defaults import RUNTIME_DEFAULTS
from .models.configuration import _read


BUILTIN_DEFAULTS = RUNTIME_DEFAULTS

ENVIRONMENT_NAMES = {
    "request_output_tokens": "ADJ_PARTNER_REQUEST_OUTPUT_TOKENS",
    "request_timeout_seconds": "ADJ_PARTNER_REQUEST_TIMEOUT_SECONDS",
    "max_steps": "ADJ_PARTNER_MAX_STEPS",
    "max_input_tokens": "ADJ_PARTNER_MAX_INPUT_TOKENS",
    "max_output_tokens": "ADJ_PARTNER_MAX_OUTPUT_TOKENS",
    "wall_time_seconds": "ADJ_PARTNER_WALL_TIME_SECONDS",
}

_INTEGER_DEFAULTS = frozenset({"request_output_tokens", "max_steps", "max_input_tokens", "max_output_tokens"})


def _explicit_values(project_root, env_file):
    if env_file is None:
        return {}
    root = Path(project_root).expanduser()
    if not root.is_absolute():
        raise ValueError("Runtime defaults require an absolute project root")
    root = root.resolve(strict=True)
    chosen = Path(env_file).expanduser()
    if not chosen.is_absolute():
        chosen = root / chosen
    return dotenv_values(stream=io.StringIO(_read(chosen)), interpolate=False)


def _positive_integer(name, value):
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_seconds(name, value):
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and positive") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def load_cli_defaults(project_root, env_file=None):
    """Resolve built-in < explicit dotenv < process environment defaults."""
    try:
        selected = _explicit_values(project_root, env_file)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("Runtime defaults configuration is missing, malformed or unsafe") from None
    result = dict(BUILTIN_DEFAULTS)
    for destination, environment_name in ENVIRONMENT_NAMES.items():
        raw = os.environ.get(environment_name)
        if raw is None or raw == "":
            raw = selected.get(environment_name)
        # Empty optional template entries mean "use the next/default source".
        # A nonempty malformed value is rejected instead of silently falling back.
        if raw is None or raw == "":
            continue
        result[destination] = (_positive_integer if destination in _INTEGER_DEFAULTS else _positive_seconds)(
            environment_name, raw)
    return result
