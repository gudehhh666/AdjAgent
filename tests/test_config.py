from pathlib import Path

from core.config import env, get_setting, resolve_path


def test_env_default_and_override(monkeypatch):
    monkeypatch.delenv("ADJ_TEST_VAR", raising=False)
    assert env("ADJ_TEST_VAR", "fallback") == "fallback"
    monkeypatch.setenv("ADJ_TEST_VAR", "value")
    assert env("ADJ_TEST_VAR") == "value"


def test_get_setting_missing_returns_default():
    assert get_setting("no_such_section", "no_key", default="d") == "d"


def test_get_setting_reads_workspace_root():
    # settings.yaml ships with workspace_root; if absent, default kicks in.
    val = get_setting("workspace_root", default="workspace")
    assert isinstance(val, str) and val


def test_resolve_path_absolute_passthrough():
    assert resolve_path("/tmp/x") == Path("/tmp/x")


def test_resolve_path_relative_under_root():
    p = resolve_path("config")
    assert p.name == "config" and p.is_absolute()
