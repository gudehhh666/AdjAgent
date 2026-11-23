"""Portable checkout startup; never calls a model or installs during tests."""

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def checkout(tmp_path, script):
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    source = ROOT / "scripts" / script
    assert source.is_file(), f"Missing {script} checkout script"
    target = root / "scripts" / script
    shutil.copy2(source, target)
    return root, target


def invoke(script, tmp_path, *args, override=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ADJ_", "OPENAI_"))}
    if override:
        env["ADJ_PARTNER_PYTHON"] = str(override)
    return subprocess.run([str(script), *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)


def test_launcher_uses_venv_inside_checkout_from_external_cwd(tmp_path):
    root, script = checkout(tmp_path, "adj-partner")
    (root / ".venv").symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    (root / "adj_partner").symlink_to(ROOT / "adj_partner", target_is_directory=True)
    result = invoke(script, tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    assert "--env-file" in result.stdout


def test_launcher_missing_interpreter_has_setup_guidance(tmp_path):
    _, script = checkout(tmp_path, "adj-partner")
    result = invoke(script, tmp_path, "--help")
    assert result.returncode != 0
    assert "scripts/setup" in result.stderr
    assert "ADJ_PARTNER_PYTHON" in result.stderr


def test_launcher_keeps_explicit_interpreter_override(tmp_path):
    result = invoke(ROOT / "scripts/adj-partner", tmp_path, "--help", override=sys.executable)
    assert result.returncode == 0, result.stderr


def test_public_provider_defaults_are_empty():
    from dotenv import dotenv_values
    assert (ROOT / ".env.example").is_file()
    values = dotenv_values(ROOT / ".env.example", interpolate=False)
    assert values["ADJ_LLM_API_KEY"] == values["ADJ_LLM_BASE_URL"] == values["ADJ_LLM_MODEL"] == ""
    assert all(not value for key, value in values.items() if key != "ADJ_ALLOW_CODE_EXEC")
    main = yaml.safe_load((ROOT / "config/llm.yaml").read_text())["main"]
    assert main.get("default_base_url") is None and main.get("default_model") is None


def test_default_dataset_lives_under_checkout_not_a_sibling(tmp_path):
    from adj_partner.application import AgentService
    root = tmp_path / "project"
    dataset = root / "data/gene_diff"
    dataset.mkdir(parents=True)
    service = AgentService(tmp_path / "state", project_root=root)
    try:
        assert service.native.dataset_dir == str(dataset)
    finally:
        service.close()


def test_public_data_paths_match_preserved_original_layout():
    # No private files are required on CI: this checks the public path contract.
    settings = yaml.safe_load((ROOT / "config/settings.yaml").read_text())
    assert settings["data_lake"] == {
        "otg_dataset": "data/open-tggates/data/Adjuvant_Discovery_Dataset.csv",
        "otg_dataset_small": ["data/raw/ADJ/toxygates-2026-6-24-2040.csv",
                              "data/raw/ADJ/toxygates-2026-6-24-4044.csv"],
        "gene_mapping": "data/open-tggates/data/gene_mapping.json",
        "gene_diff_dir": "data/gene_diff",
    }


def test_setup_help_is_side_effect_free(tmp_path):
    root, script = checkout(tmp_path, "setup")
    result = invoke(script, tmp_path, "--help")
    assert result.returncode == 0 and "--python" in result.stdout
    assert not (root / ".venv").exists() and not (root / ".env").exists()


def test_setup_rejects_bad_python_before_mutation(tmp_path):
    root, script = checkout(tmp_path, "setup")
    result = invoke(script, tmp_path, "--python", "/usr/bin/false", "--no-install")
    assert result.returncode != 0 and "Python 3.12" in result.stderr
    assert not (root / ".venv").exists() and not (root / ".env").exists()


@pytest.mark.parametrize("existing_env", [False, True])
def test_setup_creates_real_environment_without_overwriting_private_config(tmp_path, existing_env):
    root, script = checkout(tmp_path, "setup")
    shutil.copy2(ROOT / ".env.example", root / ".env.example")
    private = root / ".env"
    if existing_env:
        private.write_text("LOCAL_SETTING=preserve-this-local-value\n")
    result = invoke(script, tmp_path, "--python", sys.executable, "--no-install")
    assert result.returncode == 0, result.stderr
    assert (root / ".venv/bin/python").is_file()
    if existing_env:
        assert private.read_text() == "LOCAL_SETTING=preserve-this-local-value\n"
    else:
        assert private.read_bytes() == (root / ".env.example").read_bytes()
        assert stat.S_IMODE(private.stat().st_mode) == 0o600


def test_setup_rejects_non_environment_existing_directory(tmp_path):
    root, script = checkout(tmp_path, "setup")
    (root / ".venv").mkdir()
    sentinel = root / ".venv/user-file"
    sentinel.write_text("keep")
    result = invoke(script, tmp_path, "--python", sys.executable, "--no-install")
    assert result.returncode != 0 and "existing" in result.stderr.lower()
    assert sentinel.read_text() == "keep" and not (root / ".venv/bin").exists()


def test_checkout_dependencies_include_legacy_llm_import_used_by_tests():
    # The migrated full suite imports llm.provider through legacy Coordinator.
    names = {line.split("==")[0].strip().lower() for line in
             (ROOT / "requirements-harness.txt").read_text().splitlines()}
    assert {"tenacity", "requests", "matplotlib", "seaborn"} <= names


def test_setup_uses_complete_validated_dependency_constraints():
    from packaging.requirements import Requirement
    path = ROOT / "requirements-lock.txt"
    assert path.is_file()
    locked = {Requirement(line).name.lower(): Requirement(line) for line in path.read_text().splitlines()
              if line and not line.startswith("#")}
    for line in (ROOT / "requirements-harness.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            req = Requirement(line)
            assert str(locked[req.name.lower()].specifier) == str(req.specifier)
    assert len(locked) > 50
    assert '-c "$adj_root/requirements-lock.txt"' in (ROOT / "scripts/setup").read_text()
