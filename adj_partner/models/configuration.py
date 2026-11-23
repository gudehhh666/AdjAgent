"""Explicit host-only model configuration; never discover or mutate dotenv state."""

import io
import ipaddress
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit

from dotenv import dotenv_values
import yaml

from .base import ProviderFailure


def configuration_failure():
    return ProviderFailure("provider_configuration", "Provider configuration is missing, malformed or unsafe", cost_unknown=False)


def validate_configuration(api_key, base_url, model):
    try:
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                or any(not 33 <= ord(char) <= 126 for char in api_key)):
            raise ValueError("credential")
        if (not isinstance(model, str) or not 1 <= len(model) <= 256
                or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in model)):
            raise ValueError("model")
        model.encode("utf-8")
        if (not isinstance(base_url, str) or not 1 <= len(base_url) <= 2048
                or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in base_url)
                or any(char in base_url for char in "\\?#")):
            raise ValueError("endpoint")
        url = urlsplit(base_url)
        if (not url.hostname or url.username is not None or url.password is not None
                or "%" in url.netloc or url.port is not None and not 1 <= url.port <= 65535):
            raise ValueError("endpoint authority")
        loopback = url.hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            pass
        if url.scheme != "https" and not (url.scheme == "http" and loopback):
            raise ValueError("endpoint scheme")
    except (ValueError, TypeError, UnicodeError):
        raise configuration_failure() from None


def _read(path, *, optional=False):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        if optional:
            return None
        raise
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
            raise ValueError("bounded regular config required")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            content = handle.read(128 * 1024 + 1)
        after = os.fstat(fd)
        if (len(content) > 128 * 1024 or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("config changed")
        return content.decode("utf-8")
    finally:
        os.close(fd)


def from_environment(project_root, env_file=None, model=None):
    try:
        root = Path(project_root).expanduser()
        if not root.is_absolute():
            raise ValueError("explicit absolute project root required")
        root = root.resolve(strict=True)
        config_text = _read(root / "config/llm.yaml", optional=True)
        config = yaml.safe_load(config_text) if config_text is not None else {}
        if not isinstance(config, dict) or not isinstance(config.get("main", {}), dict):
            raise ValueError("public config shape")
        main, selected = config.get("main", {}), {}
        if env_file is not None:
            chosen = Path(env_file).expanduser()
            if not chosen.is_absolute():
                chosen = root / chosen
            selected = dotenv_values(stream=io.StringIO(_read(chosen)), interpolate=False)
        values = {name: os.environ.get(name, selected.get(name)) for name in
                  ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL")}
        return {"api_key": values["ADJ_LLM_API_KEY"],
                "base_url": values["ADJ_LLM_BASE_URL"] if values["ADJ_LLM_BASE_URL"] is not None else main.get("default_base_url"),
                "model": model if model is not None else values["ADJ_LLM_MODEL"] if values["ADJ_LLM_MODEL"] is not None else main.get("default_model")}
    except (OSError, ValueError, TypeError, yaml.YAMLError, RecursionError):
        raise configuration_failure() from None
