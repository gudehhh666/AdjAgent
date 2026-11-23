"""Check actual staged bytes and historical trees, not just .gitignore promises."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts/check-publication.py"


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    return root


def stage(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    git(root, "add", "-f", "--", path)


def scan(root, *args):
    assert SCANNER.is_file(), "Publication scanner must exist"
    result = subprocess.run([sys.executable, str(SCANNER), "--repo", str(root), *args],
                            capture_output=True, text=True, timeout=30)
    return result, json.loads(result.stdout)


def test_scanner_accepts_source_and_blank_template(tmp_path):
    root = repository(tmp_path)
    stage(root, "adj_partner/example.py", "answer = 42\n")
    stage(root, ".env.example", "ADJ_LLM_API_KEY=\nADJ_LLM_BASE_URL=\nADJ_LLM_MODEL=\n")
    result, report = scan(root)
    assert result.returncode == 0 and report["files_checked"] == 2
    assert report["issues"] == []


def test_scanner_accepts_reviewed_synthetic_protein_test_assets(tmp_path):
    root = repository(tmp_path)
    stage(root, "tests/fixtures/protein_example.fasta", ">synthetic\nACDE\n")
    stage(root, "tests/fixtures/structure_example.pdb", "HEADER SYNTHETIC\nEND\n")
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []


def test_scanner_accepts_all_required_numerical_baselines(tmp_path):
    root = repository(tmp_path)
    for name in ("data-baseline.json", "protein-baseline.json", "dataset-before.sha256"):
        path = "docs/refactor_2026-08-31/" + name
        stage(root, path, (ROOT / path).read_text())
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []


def test_scanner_accepts_architecture_docs_eval_clients_and_dashboard(tmp_path):
    root = repository(tmp_path)
    for name in ("docs/architecture/note.md", "docs/architecture/examples/qa.json", "eval/llm/client.py",
                 "eval/review_eval/prompts/LICENSE", "trace_dashboard/app.js", "scripts/review_eval"):
        stage(root, name, "public\n")
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []


def test_scanner_accepts_freeze_citation_and_experiment_index(tmp_path):
    root = repository(tmp_path)
    for name in ("CITATION.cff", "LICENSE", "HARNESS_FREEZE.md", "harness_freeze.json", "experiments/README.md",
                 "experiments/SOURCES.json", "experiments/sync_sources.py", "experiments/figures/numbers.json",
                 "experiments/figures/make_figures.py", "experiments/figures/fig2.pdf", "experiments/figures/fig2.png",
                 "experiments/p2_adjqa/run.py", "experiments/p2_adjqa/README.md",
                 "docs/architecture/figures/overview.png"):
        stage(root, name, "public\n")
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []


def test_scanner_accepts_skill_guides_but_not_other_skill_files(tmp_path):
    root = repository(tmp_path)
    stage(root, "config/skills/research-plan/SKILL.md", "---\nname: research-plan\n---\nPlan.\n")
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []
    stage(root, "config/skills/research-plan/notes.txt", "draft\n")
    result, report = scan(root)
    assert report["issues"] == [{"path": "config/skills/research-plan/notes.txt", "reason": "non-public-path"}]


def test_rendered_figures_may_be_binary_but_other_files_may_not(tmp_path):
    root = repository(tmp_path)
    for name in ("experiments/figures/fig2.png", "experiments/figures/fig2.pdf", "docs/architecture/figures/a.png"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"\x89PNG\x00\x01public")
        git(root, "add", "-f", "--", name)
    result, report = scan(root)
    assert result.returncode == 0 and report["issues"] == []
    (root / "experiments/p2_adjqa").mkdir(parents=True)
    (root / "experiments/p2_adjqa/run.py").write_bytes(b"x = 1\x00")
    git(root, "add", "-f", "--", "experiments/p2_adjqa/run.py")
    result, report = scan(root)
    assert report["issues"] == [{"path": "experiments/p2_adjqa/run.py", "reason": "binary-content"}]


def test_known_secret_in_rendered_figure_is_still_blocked(tmp_path):
    root = repository(tmp_path)
    secret = "synthetic-" + "CredentialInFigureBytes123456789"
    secret_file = tmp_path / "private.env"
    secret_file.write_text("ADJ_LLM_API_KEY=" + secret + "\n")
    (root / "experiments/figures").mkdir(parents=True)
    (root / "experiments/figures/fig.png").write_bytes(b"\x89PNG\x00" + secret.encode())
    git(root, "add", "-f", "--", "experiments/figures/fig.png")
    result, report = scan(root, "--secret-file", str(secret_file))
    assert report["issues"] == [{"path": "experiments/figures/fig.png", "reason": "known-private-value"}]


@pytest.mark.parametrize("name", ["eval/deep_research_bench/results/x.jsonl", "eval/cot.txt",
                                  "docs/architecture/private.txt", "trace_dashboard/data.json",
                                  "experiments/p1_resource_data/references.private.json",
                                  "experiments/p1_resource_data/tasks.public.jsonl",
                                  "experiments/p2_adjqa/runs/q1/answer.json", "experiments/figures/raw.json",
                                  "experiments/summary.json", "docs/architecture/figures/raw/x.png"])
def test_scanner_rejects_paths_outside_the_extended_allowlist(tmp_path, name):
    root = repository(tmp_path)
    stage(root, name, "public\n")
    result, report = scan(root)
    assert result.returncode != 0 and report["issues"][0]["path"] == name


@pytest.mark.parametrize("name", [".env", "config/provider.env", "adj_partner/secrets.json",
    "config/api.local.yaml", "Qwen-Agent-main/source.py", "tmp/trace.json", "docs/private.txt",
    "environment/data_lake/private.json", "environment/corpus_index/private.json"])
def test_scanner_rejects_forced_sensitive_or_unrelated_paths(tmp_path, name):
    root = repository(tmp_path)
    stage(root, name, "nothing secret here\n")
    result, report = scan(root)
    assert result.returncode != 0 and report["issues"][0]["path"] == name


def test_known_secret_detected_in_index_even_if_worktree_is_clean(tmp_path):
    root = repository(tmp_path)
    secret = "synthetic-" + "CredentialForIndexBoundary987654321"
    secret_file = tmp_path / "private.env"
    secret_file.write_text("ADJ_LLM_API_KEY=" + secret + "\n")
    stage(root, "adj_partner/leak.py", 'value = "' + secret + '"\n')
    (root / "adj_partner/leak.py").write_text("value = None\n")
    result, report = scan(root, "--secret-file", str(secret_file))
    assert result.returncode != 0
    assert any(item["reason"] == "known-private-value" for item in report["issues"])
    assert secret not in result.stdout + result.stderr


def test_patterns_detect_unregistered_token_without_echoing_it(tmp_path):
    root = repository(tmp_path)
    token = "ghp_" + "aB9" * 12
    stage(root, "adj_partner/leak.py", 'token = "' + token + '"\n')
    result, report = scan(root)
    assert result.returncode != 0 and any(item["reason"] == "credential-pattern" for item in report["issues"])
    assert token not in result.stdout + result.stderr


def test_nonblank_public_template_rejected(tmp_path):
    root = repository(tmp_path)
    stage(root, ".env.example", "ADJ_LLM_BASE_URL=https://private.example/v1\n")
    result, report = scan(root)
    assert result.returncode != 0 and report["issues"][0]["reason"] == "nonblank-provider-template"


def test_gitignore_excludes_private_material_but_keeps_source_template(tmp_path):
    root = repository(tmp_path)
    shutil.copy2(ROOT / ".gitignore", root / ".gitignore")
    for name in [".env", "config/provider.env", "config/api.local.yaml", "adj_partner/secrets.json",
                 "Qwen-Agent-main/Adj_Partner/main.py", "OpenAI4S/README.md", "data/a.json",
                 "harness-state/session.json", "docs/refactor_2026-08-31/evaluations/trace.json"]:
        result = subprocess.run(["git", "-C", str(root), "check-ignore", "--", name], capture_output=True)
        assert result.returncode == 0, name
    for name in [".env.example", "scripts/setup", "adj_partner/application.py", "tests/test_root_checkout.py"]:
        result = subprocess.run(["git", "-C", str(root), "check-ignore", "--", name], capture_output=True)
        assert result.returncode == 1, name


def test_history_scan_catches_secret_removed_in_later_commit(tmp_path):
    root = repository(tmp_path)
    token = "github_pat_" + "xyZ38" * 15
    stage(root, "adj_partner/example.py", 'token = "' + token + '"\n')
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
    stage(root, "adj_partner/example.py", "token = None\n")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "clean")
    clean, _ = scan(root, "--revision", "HEAD")
    assert clean.returncode == 0
    historical, report = scan(root, "--all-history")
    assert historical.returncode != 0 and report["issues"]
    assert token not in historical.stdout + historical.stderr


@pytest.mark.parametrize("allowed_root", ["adj_partner", "tmp"])
def test_known_secret_in_git_filename_is_blocked_without_echoing(tmp_path, allowed_root):
    root = repository(tmp_path)
    secret = "synthetic-" + "FilenameCredential987654321"
    secret_file = tmp_path / "private.env"
    secret_file.write_text("ADJ_LLM_API_KEY=" + secret + "\n")
    stage(root, allowed_root + "/" + secret + ".py", "value = None\n")
    result, report = scan(root, "--secret-file", str(secret_file))
    assert result.returncode != 0 and report["issues"]
    assert secret not in result.stdout + result.stderr


def test_unregistered_credential_pattern_in_filename_is_not_echoed(tmp_path):
    root = repository(tmp_path)
    token = "ghp_" + "aB4" * 12
    stage(root, "adj_partner/" + token + ".py", "value = None\n")
    result, report = scan(root)
    assert result.returncode != 0 and report["issues"]
    assert token not in result.stdout + result.stderr


def test_public_provider_selector_is_not_treated_as_a_credential(tmp_path):
    root = repository(tmp_path)
    private = tmp_path / "private.env"
    private.write_text("ADJ_WEB_SEARCH_PROVIDER=serper\nADJ_ALLOW_CODE_EXEC=0\n")
    stage(root, "adj_partner/example.py", 'supported_providers = ["serper"]\n')
    result, report = scan(root, "--secret-file", str(private))
    assert result.returncode == 0 and report["issues"] == []


def test_public_runtime_budget_values_are_not_treated_as_credentials(tmp_path):
    root = repository(tmp_path)
    private = tmp_path / "private.env"
    private.write_text("ADJ_PARTNER_MAX_INPUT_TOKENS=90000\nADJ_LLM_MODEL=gpt-fixture\n")
    stage(root, "adj_partner/example.py", 'defaults = {"tokens": 90000, "model": "gpt-fixture"}\n')
    result, report = scan(root, "--secret-file", str(private))
    assert result.returncode == 0 and report["issues"] == []
