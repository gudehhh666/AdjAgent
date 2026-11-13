"""Only synthetic host configuration; no actual dotenv or credentials read."""

import json
from pathlib import Path

import pytest

from test_runtime_provider import models


@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    for name in ("ADJ_LLM_API_KEY", "ADJ_LLM_BASE_URL", "ADJ_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_WEBHOOK_SECRET", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.setenv(name, "synthetic-unused")


def project(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config/llm.yaml").write_text("main:\n  default_base_url: https://fixture.invalid/v1\n  default_model: fixture-model\n  max_retries: 99\n", encoding="utf-8")
    return root


def load(root, **kwargs):
    cls = models().OpenAICompatiblePort
    assert hasattr(cls, "from_environment"), "explicit anchored host configuration must exist"
    return cls.from_environment(project_root=root, **kwargs)


def test_public_project_defaults_and_env_key_are_used_without_implicit_dotenv(tmp_path, monkeypatch):
    root = project(tmp_path)
    (root / ".env").write_text("ADJ_LLM_API_KEY=synthetic-must-not-load\n")
    with pytest.raises(models().ProviderFailure) as error:
        load(root)
    assert error.value.code == "provider_configuration" and not error.value.cost_unknown
    monkeypatch.setenv("ADJ_LLM_API_KEY", "synthetic-key")
    monkeypatch.chdir(tmp_path)
    port = load(root)
    assert port.metadata == {"provider_kind": "openai_compatible", "model": "fixture-model"}
    assert port.secrets == ("synthetic-key",)
    assert port._client.max_retries == 0  # Public SDK property, legacy config cannot add retries.
    port.close()


def test_explicit_dotenv_is_project_anchored_and_does_not_mutate_environment(tmp_path, monkeypatch):
    import os
    root = project(tmp_path)
    (root / "chosen.env").write_text("ADJ_LLM_API_KEY=synthetic-file-key\nADJ_LLM_MODEL=file-model\nADJ_LLM_BASE_URL=https://file.invalid/v1\n")
    monkeypatch.chdir(tmp_path)
    port = load(root, env_file="chosen.env")
    assert port.metadata["model"] == "file-model" and port.secrets == ("synthetic-file-key",)
    assert "ADJ_LLM_API_KEY" not in os.environ
    assert "file.invalid" not in json.dumps(port.metadata)
    port.close()


def test_process_env_precedes_explicit_file_and_explicit_model_precedes_both(tmp_path, monkeypatch):
    root = project(tmp_path)
    (root / "chosen.env").write_text("ADJ_LLM_API_KEY=synthetic-file-key\nADJ_LLM_MODEL=file-model\nADJ_LLM_BASE_URL=https://file.invalid/v1\n")
    monkeypatch.setenv("ADJ_LLM_API_KEY", "synthetic-process-key")
    monkeypatch.setenv("ADJ_LLM_MODEL", "process-model")
    monkeypatch.setenv("ADJ_LLM_BASE_URL", "https://process.invalid/v1")
    port = load(root, env_file="chosen.env", model="explicit-model")
    assert port.secrets == ("synthetic-process-key",) and port.metadata["model"] == "explicit-model"
    assert str(port._client.base_url) == "https://process.invalid/v1/"
    port.close()


def test_no_project_config_is_needed_when_all_three_named_vars_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ADJ_LLM_API_KEY", "synthetic-key")
    monkeypatch.setenv("ADJ_LLM_MODEL", "fixture-model")
    monkeypatch.setenv("ADJ_LLM_BASE_URL", "https://fixture.invalid/v1")
    port = load(tmp_path)
    assert port.metadata["model"] == "fixture-model"
    port.close()


@pytest.mark.parametrize("endpoint", ["http://example.com/v1", "ftp://fixture.invalid/v1", "https://user:synthetic-password@example.com/v1",
    "https://example.com/v1?api_key=synthetic", "https://example.com/v1#fragment", "https:///v1", "https://example.com:bogus/v1",
    "https://example.com\\@evil.invalid/v1", " https://example.com/v1", "https://example.com/v1\n", "http://127.0.0.1.evil.invalid/v1"])
def test_unsafe_endpoints_rejected_before_sdk_or_network(endpoint):
    with pytest.raises(models().ProviderFailure) as error:
        models().OpenAICompatiblePort(api_key="synthetic-key", base_url=endpoint, model="fixture")
    assert error.value.code == "provider_configuration" and not error.value.cost_unknown
    assert endpoint not in str(error.value)


@pytest.mark.parametrize("endpoint", ["https://example.com/v1", "http://127.0.0.1:1234/v1", "http://localhost:1234/v1", "http://[::1]:1234/v1"])
def test_https_and_exact_loopback_fixture_endpoints_supported(endpoint):
    port = models().OpenAICompatiblePort(api_key="synthetic-key", base_url=endpoint, model="fixture")
    assert port.metadata["model"] == "fixture"
    port.close()


@pytest.mark.parametrize("key,model", [(None, "fixture"), ("", "fixture"), ("bad\nkey", "fixture"), ("synthetic-key", ""),
    ("synthetic-key", "bad\nmodel"), ("synthetic-key", None)])
def test_missing_or_invalid_credential_model_inputs_fail_safely(key, model):
    with pytest.raises(models().ProviderFailure) as error:
        models().OpenAICompatiblePort(api_key=key, base_url="https://fixture.invalid/v1", model=model)
    assert error.value.code == "provider_configuration" and not error.value.cost_unknown


def test_malformed_or_missing_explicit_config_files_raise_safe_errors(tmp_path):
    root = project(tmp_path)
    with pytest.raises(models().ProviderFailure) as error:
        load(root, env_file="missing-file.env")
    assert "missing-file" not in str(error.value) and error.value.code == "provider_configuration"
    (root / "config/llm.yaml").write_text("main: [broken: synthetic-private-value")
    with pytest.raises(models().ProviderFailure) as error:
        load(root)
    assert "synthetic-private" not in str(error.value)


def test_dotenv_interpolation_does_not_read_unselected_ambient_credentials(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.setenv("UNSELECTED_SECRET", "synthetic-unselected-secret")
    (root / "chosen.env").write_text("ADJ_LLM_API_KEY=${UNSELECTED_SECRET}\n")
    port = load(root, env_file="chosen.env")
    assert port.secrets == ("${UNSELECTED_SECRET}",)
    port.close()


def test_invalid_endpoint_encoding_cannot_escape_as_raw_sdk_configuration_error():
    with pytest.raises(models().ProviderFailure) as error:
        models().OpenAICompatiblePort(api_key="synthetic-key", base_url="https://synthetic-key.\ud800.invalid/v1", model="fixture")
    assert error.value.code == "provider_configuration" and "synthetic-key" not in str(error.value)
