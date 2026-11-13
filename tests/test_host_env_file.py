"""The private key file .einwang_env: resolution, default loading by `adj`, and exclusion everywhere."""
import importlib.util
from pathlib import Path

import pytest

from adj_partner import application_config, entry
from adj_partner.application_config import host_env_file
from adj_partner.tools.policy import _credential_path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_variable_wins_then_project_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ADJ_ENV_FILE", raising=False)
    assert host_env_file(tmp_path) is None
    (tmp_path / ".einwang_env").write_text("A=1\n")
    assert host_env_file(tmp_path) == tmp_path / ".einwang_env"
    monkeypatch.setenv("ADJ_ENV_FILE", "other.env")
    assert host_env_file(tmp_path) == (tmp_path / "other.env").resolve()


@pytest.fixture
def captured(tmp_path, monkeypatch):
    calls = []
    keyfile = tmp_path / ".einwang_env"
    keyfile.write_text("A=1\n")
    monkeypatch.setattr(application_config, "host_env_file", lambda *a: keyfile)
    import adj_partner.cli as session_cli
    import adj_partner.orchestration.cli as task_cli
    import adj_partner.doctor as doctor
    monkeypatch.setattr(session_cli, "main", lambda argv: calls.append(("session", argv)) or 0)
    monkeypatch.setattr(task_cli, "main", lambda argv: calls.append(("task", argv)) or 0)
    monkeypatch.setattr(doctor, "main", lambda argv: calls.append(("doctor", argv)) or 0)
    return calls, str(keyfile)


def test_adj_loads_the_key_file_for_session_and_task_commands(captured):
    calls, keyfile = captured
    entry.main(["run", "SESSION", "hello"])
    entry.main(["task", "status", "fp"])
    entry.main(["doctor", "runs"])
    assert calls == [("session", ["--env-file", keyfile, "run", "SESSION", "hello"]),
                     ("task", ["--env-file", keyfile, "status", "fp"]),
                     ("doctor", ["runs"])]


def test_explicit_choice_or_offline_demo_is_respected(captured):
    calls, _ = captured
    entry.main(["--env-file", "x.env", "tui"])
    entry.main(["--offline-demo", "tui"])
    assert calls == [("session", ["--env-file", "x.env", "tui"]), ("session", ["--offline-demo", "tui"])]


def test_agent_tools_publication_and_delivery_all_refuse_the_key_file():
    assert _credential_path(Path("/project/.einwang_env"))
    assert _credential_path(Path("/project/Eval_Data/AdjQA/bench/questions.jsonl"))
    assert _credential_path(Path("/w/references.judge_only.json"))
    assert not _credential_path(Path("/project/eval/knowledge_anchor_eval.py"))
    publication = load_script("check-publication.py")
    assert not publication.allowed(".einwang_env") and publication.allowed("scripts/with-env")
    delivery = load_script("build_delivery.py")
    assert not delivery._allowed(Path(".einwang_env"))
