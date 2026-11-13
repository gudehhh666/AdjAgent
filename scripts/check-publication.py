#!/usr/bin/env python3
"""Audit Git index/tree bytes; never print matched credential values.

Run with --secret-file for in-memory matching of known local dotenv values.
No network calls or Git changes. Use --all-history before first publication.
Not a proof that arbitrary source prose contains no unregistered secret.
"""
import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from urllib.parse import quote

CODE = {"adj_partner", "core", "environment", "legacy", "tests", "scripts"}
TOP = {".gitignore", ".env.example", "README.md", "README_HARNESS.md", "requirements.txt", "requirements-harness.txt",
       "requirements-lock.txt", "pytest.ini", "__init__.py", "CITATION.cff", "HARNESS_FREEZE.md", "harness_freeze.json",
       "LICENSE"}
EXPERIMENT_INDEX = {"experiments/README.md", "experiments/PENDING.md", "experiments/SOURCES.json",
                    "experiments/sync_sources.py"}
FIGURE_DIRS = {PurePosixPath("experiments/figures"), PurePosixPath("docs/architecture/figures")}
SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".tcss", ".html", ".css", ".svg"}
RECORDS = PurePosixPath("docs/refactor_2026-08-31")
BLOCKED = {".git", ".venv", "__pycache__", "workspace", "workspaces", "harness-state", ".harness-state", "state",
           "results", "evaluations", "deliveries", "node_modules", ".ssh", ".aws", ".gnupg", ".netrc", ".npmrc",
           ".pypirc", ".git-credentials", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
PATTERNS = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|"
                      rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}|"
                      rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)")
MAX_FILE = 16 * 1024 * 1024
CREDENTIAL_KEY = re.compile(r"(?:API_KEY|SECRET|PASSWORD|PRIVATE_KEY|ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN|BEARER_TOKEN|TOKEN)$")


def allowed(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or any(ord(c) < 32 for c in name):
        return False
    if name == ".env.example":
        return True
    if name in {"tests/fixtures/protein_example.fasta", "tests/fixtures/structure_example.pdb"}:
        return True
    if path.parts[:2] in {("environment", "data_lake"), ("environment", "corpus_index")}:
        return name == "environment/data_lake/.gitkeep"
    for part in path.parts:
        lower = part.lower()
        if (lower in BLOCKED or lower.startswith((".env", ".einwang_env", "credentials", "secrets")) or ".local." in lower
                or lower.endswith((".env", ".key", ".pem", ".p12", ".pfx", ".keystore", ".db", ".sqlite", ".sqlite3"))):
            return False
    if len(path.parts) == 1:
        return name in TOP
    if name in {"scripts/adj-partner", "scripts/adj", "scripts/with-env", "scripts/setup", "scripts/review_eval", "docs/ROOT_MIGRATION.md",
                "docs/superpowers/plans/2026-08-31-root-publication.md",
                "docs/superpowers/plans/2026-08-31-tui-ctrl-c-exit.md",
                "docs/superpowers/specs/2026-08-31-tui-ctrl-c-exit-design.md",
                "docs/superpowers/plans/2026-09-02-open-research-trace-dashboard.md",
                "docs/superpowers/specs/2026-09-02-open-research-trace-dashboard-design.md",
                "trace_dashboard/app.js", "trace_dashboard/index.html", "trace_dashboard/styles.css",
                "scripts/progress_canvas.template.tsx",
                "eval/review_eval/prompts/LICENSE",
                ".github/workflows/tests.yml"}:
        return True
    if path.parts[0] == "docs" and path.parts[1] == "architecture":
        if len(path.parts) == 4 and path.parts[2] == "figures":
            return path.suffix == ".png"
        return path.suffix in {".md", ".json"}
    # Experiment directories hold run-script copies and aggregate numbers; task texts, gold and runs stay local.
    if path.parts[0] == "experiments":
        if path.parts[1] == "figures":
            return len(path.parts) == 3 and (path.suffix in {".py", ".pdf", ".png"} or path.name == "numbers.json")
        if len(path.parts) == 2:
            return name in EXPERIMENT_INDEX
        return len(path.parts) == 3 and (path.suffix == ".py" or path.name == "README.md")
    if path.parent == RECORDS:
        return path.name.endswith(("_CONTRACT.md", "_API.md")) or path.name in {"data-baseline.json", "protein-baseline.json", "dataset-before.sha256"}
    if path.parts[0] == "config":
        if name in {"config/skills/adjuvant-design/SKILL.md",
                    "config/skills/adjuvant-design/acceptance.json",
                    "config/skills/adjuvant-design/critic-workflow.md",
                    "config/skills/adjuvant-design/references/standards.md"}:
            return True
        if len(path.parts) == 4 and path.parts[1] == "skills" and path.name == "SKILL.md":
            return True
        return len(path.parts) == 2 and path.name in {"llm.yaml", "settings.yaml", "sources.yaml"}
    if name == "docs/superpowers/specs/2026-09-06-adjuvant-design-workflow.md":
        return True
    if path.parts[0] == "eval":
        if path.parts[1] in {"llm", "prompts", "review_eval"}:
            return path.suffix in {".py", ".md"}
        return (len(path.parts) == 2 and path.suffix in {".py", ".json"}) or (path.parts[1] == "fixtures" and path.suffix in SUFFIXES)
    return path.parts[0] in CODE and (path.suffix in SUFFIXES or path.name == ".gitkeep")


def private_values(paths):
    from dotenv import dotenv_values
    import io
    values = set()
    for path in paths:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 128 * 1024:
                raise ValueError("Private input must be a bounded regular dotenv file")
            raw = stream.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                raise ValueError("Private input too large")
        for key, value in dotenv_values(stream=io.StringIO(raw.decode()), interpolate=False).items():
            # Runtime budgets, model IDs, URLs and provider selectors are
            # public configuration even when their values happen to occur in
            # source. Only credential-shaped dotenv keys feed exact matching.
            if not isinstance(key, str) or not CREDENTIAL_KEY.search(key.upper()):
                continue
            if value and len(value) >= 6:
                values.update((value.encode(), quote(value, safe="").encode(),
                               base64.b64encode(value.encode()), json.dumps(value)[1:-1].encode()))
    return values


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("Git read failed (details suppressed to protect configuration)")
    return result.stdout


def entries(repo, revision=None):
    command = ("ls-tree", "-rz", "--full-tree", revision) if revision else ("ls-files", "--stage", "-z")
    for raw in git(repo, *command).split(b"\0"):
        if not raw:
            continue
        meta, raw_name = raw.split(b"\t", 1)
        mode, middle, last = meta.decode().split()
        oid = last if revision else middle
        if not revision and last != "0":
            raise ValueError("Unmerged index cannot be published")
        yield raw_name.decode("utf-8"), mode, oid


def inspect(repo, *, revision=None, all_history=False, secrets=()):
    commits = git(repo, "rev-list", "--all").decode().splitlines() if all_history else [revision]
    if not commits or len(commits) > 1000:
        raise ValueError("Expected 1..1000 revisions for bounded audit")
    issues, seen = [], set()
    total = 0
    for commit in commits:
        for name, mode, oid in entries(repo, commit):
            if (name, mode, oid) in seen:
                continue
            seen.add((name, mode, oid))
            reason = None
            private_name = any(value in name.encode() for value in secrets) or bool(PATTERNS.search(name.encode()))
            if len(seen) > 10000:
                raise ValueError("Too many files for bounded audit")
            if private_name:
                reason = "credential-in-path"
            elif not allowed(name):
                reason = "non-public-path"
            elif mode not in {"100644", "100755"}:
                reason = "non-regular-git-entry"
            else:
                size = int(git(repo, "cat-file", "-s", oid))
                total += size
                if size > MAX_FILE or total > 128 * 1024 * 1024:
                    raise ValueError("Publication exceeds size bounds")
                raw = git(repo, "cat-file", "blob", oid)
                if any(value in raw for value in secrets):
                    reason = "known-private-value"
                elif PATTERNS.search(raw):
                    reason = "credential-pattern"
                elif b"\0" in raw and not (PurePosixPath(name).parent in FIGURE_DIRS
                                           and PurePosixPath(name).suffix in {".png", ".pdf"}):
                    reason = "binary-content"
                elif name == ".env.example":
                    for line in raw.decode().splitlines():
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        key, equal, value = line.partition("=")
                        if not equal or (value.strip().strip("\"'") and not (key.strip() == "ADJ_ALLOW_CODE_EXEC" and value.strip() == "0")):
                            reason = "nonblank-provider-template"
                            break
            if reason:
                issues.append({"path": "[redacted-path]" if private_name else name, "reason": reason})
    return {"files_checked": len(seen), "revisions_checked": len(commits), "issues": issues,
            "scope": "Git blob bytes, path allowlist, known private values and credential patterns; not a guarantee against every unknown secret"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revision")
    group.add_argument("--all-history", action="store_true")
    parser.add_argument("--secret-file", action="append", default=[])
    args = parser.parse_args()
    try:
        report = inspect(args.repo, revision=args.revision, all_history=args.all_history,
                         secrets=private_values(args.secret_file) if args.secret_file else ())
    except Exception:
        print(json.dumps({"error": "Audit could not complete safely; no publication authorized"}))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
