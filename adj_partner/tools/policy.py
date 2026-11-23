"""Session capabilities and cooperative resource checks, not a Python sandbox."""

from pathlib import Path


class ToolCancelled(Exception):
    """Cooperative cancellation: adapters must check before side effects."""


class SessionToolset:
    def __init__(self, store, catalog, session_id):
        self.store, self.catalog, self.session_id = store, catalog, session_id
        self._session = store.get_session(session_id)

    def snapshot(self):
        self._session = self.store.get_session(self.session_id)
        config = self._session["config"]
        names = sorted(set(config.get("enabled_tools", [])) & set(config.get("allowed_tools", []))
                       & set(self.catalog.names()))
        specs = [self.catalog.get(name).spec for name in names]
        return {"revision": self._session["revision"], "tools": [spec.model_schema() for spec in specs],
                "definition_hashes": {spec.name: spec.definition_hash for spec in specs}}

    def _change(self, name, enable, *, also=(), expected_revision=None):
        names = (name, *also)
        config = dict(self._session["config"])
        for tool in names:
            self.catalog.get(tool)
        if enable and not set(names).issubset(config.get("allowed_tools", [])):
            raise PermissionError("Tool is not in the host-authorized subset")
        enabled = set(config.get("enabled_tools", []))
        if enable:
            enabled.update(names)
        else:
            enabled.discard(name)
        config["enabled_tools"] = sorted(enabled)
        revision = self._session["revision"] if expected_revision is None else expected_revision
        self._session = self.store.update_session(self.session_id, config, revision)
        self.store.append_event(self.session_id, "toolset.changed", {
            "name": name, **({"also": list(also)} if also else {}),
            "enabled": enable, "revision": self._session["revision"]})
        return self.snapshot()

    def enable(self, name, *, also=(), expected_revision=None):
        return self._change(name, True, also=also, expected_revision=expected_revision)

    def disable(self, name):
        return self._change(name, False)


def _credential_path(path):
    for part in path.parts:
        lower = part.lower()
        if (lower in {".ssh", ".aws", ".gnupg", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
                      "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
                      "secrets.json", "secrets.yaml", "secrets.yml", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
                or lower in {".env", ".einwang_env"} or lower.startswith(".env.")
                # Benchmark answer keys are as off-limits to agents as credentials.
                or lower in {"eval_data", "references.judge_only.json"}
                or lower.endswith((".pem", ".key", ".p12", ".pfx", ".keystore"))):
            return True
    return False


class ToolContext:
    def __init__(self, store, session_id, run_id, toolset_revision, cancel_event=None, *, call_id=None):
        store.get_session(session_id)
        if store.get_run(run_id)["session_id"] != session_id:
            raise ValueError("Run does not belong to session")
        self.store, self.session_id, self.run_id = store, session_id, run_id
        self.toolset_revision, self.cancel_event = toolset_revision, cancel_event
        self.call_id = call_id

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ToolCancelled("Tool operation cancelled")
        status = self.store.get_run(self.run_id)["status"]
        if status == "cancelled":
            raise ToolCancelled("Run cancellation was persisted by the host")
        if status in {"succeeded", "failed", "budget_exhausted"}:
            raise PermissionError("A terminal run cannot initiate another tool operation")

    def check_path(self, path, write=False, must_exist=True):
        """Return a checked path; callers still own safe opening/TOCTOU handling.

        Roots authorize reads and writes equally in the current SessionConfig.
        This method cannot contain arbitrary native code or a trusted MCP server.
        """
        self.check_cancelled()
        candidate = Path(path).expanduser()
        if ".." in candidate.parts or _credential_path(candidate):
            raise PermissionError("Traversal and credential paths are not accessible")
        resolved = candidate.resolve(strict=False)
        roots = [Path(root).expanduser().resolve() for root in
                 self.store.get_session(self.session_id)["config"].get("allowed_roots", [])]
        if _credential_path(resolved) or not any(resolved.is_relative_to(root) for root in roots):
            raise PermissionError("Path escapes the host-approved roots or targets credentials")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
