"""Host-only paths and durable model preferences; credentials remain in memory."""

from datetime import datetime, timezone
from pathlib import Path
import time

from .models import OpenAICompatiblePort
from .models.configuration import from_environment, configuration_failure
from .runtime.journal import RunBusy
from .storage import TERMINAL_STATUSES
from .tools.catalog import canonical_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST_ENV_NAME = ".einwang_env"


def host_env_file(project_root=PROJECT_ROOT):
    """The project's private key file: $ADJ_ENV_FILE if set, else <root>/.einwang_env when present."""
    import os
    explicit = os.environ.get("ADJ_ENV_FILE")
    if explicit:
        return host_path(explicit, Path(project_root))
    candidate = Path(project_root) / HOST_ENV_NAME
    return candidate if candidate.is_file() else None


def host_path(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def model_name(name):
    if (not isinstance(name, str) or not 1 <= len(name) <= 256
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError("Model name must be nonempty text without whitespace/control characters, at most256 characters")
    canonical_json(name)
    return name


class HostModels:
    def __init__(self, store, project_root, env_file, injected):
        self.store, self.project_root, self.env_file, self.injected = store, project_root, env_file, injected
        self.ports = {}
        if injected is not None and (not callable(getattr(injected, "complete", None)) or not isinstance(injected.metadata, dict)):
            raise TypeError("Injected model must be an explicit ModelPort")
        with store.transaction(write=True) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS application_models (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id), model TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0)""")
        self.prime()

    def prime(self):
        if self.injected is not None:
            self.store.register_secrets(getattr(self.injected, "secrets", ()))
            return {"model": self.injected.metadata.get("model")}
        config = from_environment(self.project_root, self.env_file)
        secret = config.get("api_key")
        if secret is not None:
            if not isinstance(secret, str) or len(secret) > 4096: raise configuration_failure()
            if secret: self.store.register_secrets([secret])
        return config

    def selected(self, session_id):
        self.store.get_session(session_id)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT model FROM application_models WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else None

    def info(self, session_id):
        config = self.prime()
        selected = self.selected(session_id)
        port = self.injected or self.ports.get(session_id)
        return self.store.redact({"model": selected or (port.metadata.get("model") if port else config.get("model")),
            "provider_kind": port.metadata.get("provider_kind", "injected") if port else "openai_compatible",
            "loaded": port is not None, "override": selected})

    def select(self, session_id, name):
        model_name(name)
        self.prime()
        if self.store.redact(name) != name: raise ValueError("A model name cannot contain configured credentials")
        with self.store.transaction(write=True) as conn:
            if conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone() is None: raise KeyError("Unknown session")
            runs = conn.execute("SELECT status FROM runs WHERE session_id=?", (session_id,)).fetchall()
            lease = conn.execute("SELECT expires FROM runtime_leases WHERE session_id=?", (session_id,)).fetchone()
            if any(row[0] not in TERMINAL_STATUSES for row in runs) or lease and lease[0] > time.time():
                raise RunBusy("Change models only between completed/cancelled runs")
            if self.injected is not None and name != self.injected.metadata.get("model"):
                raise ValueError("A fixed injected ModelPort cannot switch models")
            conn.execute("""INSERT INTO application_models(session_id,model) VALUES(?,?)
                ON CONFLICT(session_id) DO UPDATE SET model=excluded.model,revision=revision+1""", (session_id, name))
            conn.execute("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,NULL,?,?,?)",
                (session_id, "model.changed", canonical_json({"model": name, "source": "host"}), datetime.now(timezone.utc).isoformat()))
        previous = self.ports.pop(session_id, None)
        if previous is not None: previous.close()
        return self.info(session_id)

    def port(self, session_id):
        self.prime()
        selected = self.selected(session_id)
        if self.injected is not None:
            if selected is not None and selected != self.injected.metadata.get("model"):
                raise ValueError("Injected ModelPort differs from the session model choice")
            return self.injected
        previous = self.ports.get(session_id)
        if previous is not None and (selected is None or previous.metadata["model"] == selected): return previous
        if previous is not None: previous.close()
        port = OpenAICompatiblePort.from_environment(project_root=self.project_root, env_file=self.env_file, model=selected)
        self.store.register_secrets(port.secrets)
        self.ports[session_id] = port
        return port

    def close(self):
        ports = list(self.ports.values()) + ([self.injected] if self.injected else [])
        self.ports.clear()
        for port in ports:
            close = getattr(port, "close", None)
            if close: close()
