"""One host service for CLI, TUI and evaluations; no second decision loop."""

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event, RLock

from .analysis import ANALYSIS_TOOLS, AnalysisCapabilities
from .analysis.capabilities import ANALYSIS_HANDLER, ANALYSIS_JOB_TOOLS
from .analysis.public_code_capabilities import PUBLIC_CODE_HANDLER, PUBLIC_CODE_TOOLS, PublicCodeCapabilities
from .application_config import HostModels, PROJECT_ROOT, host_path
from .contracts import Budget, SessionConfig
from .defaults import DEFAULT_ENGLISH_INSTRUCTIONS, DEFAULT_PROFILE, RUNTIME_DEFAULTS
from .execution.manager import TERMINAL as JOB_TERMINAL
from .native.capabilities import NativeCapabilities
from .profiles import apply_profile, get_profile
from .research import ResearchStore
from .research.tools import register_research_tools
from .runtime.engine import AgentEngine
from .runtime.journal import RunBusy, RunJournal
from .runtime.tools import register_runtime_tools
from .storage import Store, TERMINAL_STATUSES
from .tools import ToolCatalog, ToolExecutor
from .tools.catalog import canonical_json
from .tools.management import register_management_tools
from .tools.policy import SessionToolset


class _Stop:
    def __init__(self, closed, external): self.closed, self.external = closed, external
    def is_set(self): return self.closed.is_set() or self.external is not None and self.external.is_set()
    def wait(self, seconds):
        # Never busy-spin; external stop latency is bounded by the250ms interval.
        if self.is_set(): return True
        self.closed.wait(seconds)
        return self.is_set()


class AgentService:
    def __init__(self, root, *, model=None, project_root=None, env_file=None, dataset_dir=None, web_search=None,
                 mcp_config=None, request_output_tokens=RUNTIME_DEFAULTS["request_output_tokens"],
                 request_timeout_seconds=RUNTIME_DEFAULTS["request_timeout_seconds"], resource_pack=None):
        self.project_root = host_path(project_root or PROJECT_ROOT, PROJECT_ROOT)
        self.env_file = env_file
        self.store = Store(host_path(root, self.project_root))
        self.catalog, self._lock, self._closed = ToolCatalog(), RLock(), Event()
        self.journal = RunJournal(self.store)
        self.models = HostModels(self.store, self.project_root, env_file, model)
        self.research = ResearchStore(self.store)
        default_data = self.project_root / "data/gene_diff"
        dataset = host_path(dataset_dir, self.project_root) if dataset_dir is not None else default_data if default_data.is_dir() else None
        if web_search is None:
            from .native.web_search import web_search_from_environment
            web_search = web_search_from_environment(self.project_root, env_file)
        from .native.scholarly import ScholarlyGateway
        literature = ScholarlyGateway.from_environment(self.project_root, env_file)
        from .native.fulltext import FullTextFetcher
        fulltext = FullTextFetcher()
        from .native.documents import DocumentReader
        documents = DocumentReader()
        from .resources import ResourcePack
        default_pack = self.project_root / 'data/adjuvant_resources_v4'
        if not (default_pack / 'manifest.json').is_file():
            default_pack = self.project_root / 'data/adjuvant_resources_v3'
        if not (default_pack / 'manifest.json').is_file():
            default_pack = self.project_root / 'data/adjuvant_resources_v2'
        if not (default_pack / 'manifest.json').is_file():
            default_pack = self.project_root / 'data/adjuvant_resources'
        pack_path = host_path(resource_pack, self.project_root) if resource_pack is not None else default_pack
        installed_pack = ResourcePack(pack_path) if resource_pack is not None or (pack_path / 'manifest.json').is_file() else None
        self.native = NativeCapabilities(self.store, research=self.research, dataset_dir=dataset,
                                         literature=literature, fulltext=fulltext, documents=documents,
                                         web_search=web_search, resource_pack=installed_pack)
        self.analysis = AnalysisCapabilities(self.store, jobs=self.native.jobs)
        self.public_code = PublicCodeCapabilities(self.store, jobs=self.native.jobs)
        register_management_tools(self.catalog, self.store)
        register_research_tools(self.catalog, self.research)
        from .research.critic import ResearchCritic
        self.critic = ResearchCritic(self.research, self.models.port)
        self.critic.register(self.catalog)
        self.native.register(self.catalog)
        self.analysis.register(self.catalog)
        self.public_code.register(self.catalog)
        register_runtime_tools(self.catalog)
        from .research_entry import register_research_entry
        register_research_entry(self)
        from .quality_entry import register_quality_entry
        register_quality_entry(self)
        from .evidence_bound_entry import install_bound_profile
        install_bound_profile()
        self._benign_tools = frozenset(self.catalog.names())
        if mcp_config is not None:
            from .tools.mcp import MCPProvider, MCPServerConfig
            if not isinstance(mcp_config, (list, tuple)) or any(not isinstance(item, MCPServerConfig) for item in mcp_config):
                raise TypeError("MCP configuration must be explicit host MCPServerConfig objects")
            for config in mcp_config: MCPProvider(config).register(self.catalog, self.store)
        self.executor = ToolExecutor(self.store, self.catalog)
        self.request_output_tokens, self.request_timeout_seconds = request_output_tokens, request_timeout_seconds

    def _prime(self):
        self.models.prime()
        self.catalog.prepare_store(self.store)

    def _public(self, value):
        self._prime()
        return self.store.redact(value)

    def create_session(self, profile=DEFAULT_PROFILE, instructions="", allowed_roots=(), allow_network=False, allowed_tools=None, budget=None):
        self._prime()
        definition = get_profile(profile)
        names = sorted(self._benign_tools) if allowed_tools is None else list(allowed_tools)
        if any(not isinstance(name, str) or name not in self.catalog.names() for name in names): raise ValueError("Unknown authorized tool name")
        if not isinstance(instructions, str) or type(allow_network) is not bool: raise ValueError("Invalid host session configuration")
        if profile == DEFAULT_PROFILE:
            instructions += ("\n\n" if instructions else "") + DEFAULT_ENGLISH_INSTRUCTIONS
        limits = budget if isinstance(budget, Budget) else Budget(**(budget or {}))
        config = SessionConfig(profile=profile, instructions=instructions,
            allowed_tools=sorted(set(names)), enabled_tools=sorted(set(definition["tools"]) & set(names)),
            allowed_roots=[str(host_path(path, self.project_root)) for path in allowed_roots],
            allow_network=allow_network, budget=limits)
        session = self.store.create_session(config.to_dict())
        selected = self.models.info(session["id"])["model"]
        if selected: self.models.select(session["id"], selected)
        return self._public(session)

    def sessions(self): return self._public(self.store.list_sessions())
    def session(self, identifier): return self._public(self.store.get_session(identifier))
    def runs(self, session_id): return self._public(self.store.list_runs(session_id))
    def run(self, identifier): return self._public(self.store.get_run(identifier))
    def tasks(self, session_id): return self._public(self.store.list_tasks(session_id))
    def task(self, identifier): return self._public(self.store.get_task(identifier))
    def messages(self, session_id): return self._public(self.store.messages(session_id))
    def actions(self, run_id): return self._public(self.store.list_actions(run_id))
    def events(self, session_id, run_id=None): return self._public(self.store.events(session_id, run_id))
    def activity_after(self, session_id, run_id, after_seq=0, limit=64):
        self._prime()
        from .application_activity import activity_after
        return activity_after(self.store, session_id, run_id, after_seq, limit)
    def model_info(self, session_id):
        self._prime(); return self.models.info(session_id)

    def _engine(self, session_id):
        if self._closed.is_set(): raise RuntimeError("Application service is closed")
        with self._lock:
            model = self.models.port(session_id)
            engine = AgentEngine(self.store, self.catalog, model, executor=self.executor, research=self.research,
                resolve_job=self._resolve_job, recover_action=self._recover_action,
                max_output_tokens=self.request_output_tokens, max_request_seconds=self.request_timeout_seconds)
            if self.store.get_session(session_id)['config'].get('profile') == 'research-resource-aware':
                from .research_context import ResearchContextView
                from .research_entry import initial_resource_cards
                engine.context = ResearchContextView(self.store,
                    initial_cards=lambda goal, config: initial_resource_cards(self, goal, config))
            elif self.store.get_session(session_id)['config'].get('profile') == 'research-evidence-first':
                from .quality_entry import make_evidence_context
                engine.context = make_evidence_context(self)
            elif self.store.get_session(session_id)['config'].get('profile') == 'research-evidence-bound':
                from .evidence_bound_entry import make_bound_context
                engine.context = make_bound_context(self)
            return engine

    def _resolve_job(self, session_id, run_id, job_id):
        job = self.native.jobs.get(job_id, session_id)
        if job["handler_ref"] == ANALYSIS_HANDLER:
            return self.analysis.resolve_job(session_id, run_id, job_id)
        if job["handler_ref"] == PUBLIC_CODE_HANDLER:
            return self.public_code.resolve_job(session_id, run_id, job_id)
        return self.native.resolve_job(session_id, run_id, job_id)

    def _recover_action(self, session_id, run_id, call):
        if call.name in PUBLIC_CODE_TOOLS:
            return self.public_code.recover_action(session_id, run_id, call)
        if call.name in ANALYSIS_JOB_TOOLS:
            return self.analysis.recover_action(session_id, run_id, call)
        return self.native.recover_action(session_id, run_id, call)

    def submit(self, session_id, text, acceptance=None, *, execution_binding=None):
        with self._lock:
            engine = self._engine(session_id)
            selected = self.models.selected(session_id)
            result = engine.start(session_id, text, acceptance=acceptance,
                                 execution_binding=execution_binding)
            if self.models.selected(session_id) != selected:
                self.journal.cancel(result["id"])
                raise RuntimeError("Model choice changed concurrently; cancelled the unissued run")
            return self._public(result)

    def resume(self, run_id, stop_event=None):
        run = self.store.get_run(run_id)
        if run["status"] in TERMINAL_STATUSES: return self._public(run)
        return self._public(self._engine(run["session_id"]).resume(run_id, stop_event=_Stop(self._closed, stop_event)))

    def drive(self, run_id, stop_event=None):
        stop = _Stop(self._closed, stop_event)
        while True:
            result = self.resume(run_id, stop_event=stop)
            waiting = result["state"].get("waiting", {})
            if result["status"] != "waiting" or waiting.get("kind") != "job" or stop.is_set(): return result
            identifier = waiting.get("job_id")
            if not identifier: return result
            # Only reconcile a known native job. It does not make a new model
            # decision or resubmit an unknown effect; AgentEngine owns recovery.
            job = self.native.jobs.get(identifier, result["session_id"])
            if job["run_id"] != run_id: raise PermissionError("Waiting job is not this run's job")
            while job["status"] not in JOB_TERMINAL:
                if stop.wait(.25): return self._public(result)
                if self.store.get_run(run_id)["status"] in TERMINAL_STATUSES: return self.run(run_id)
                job = self.native.jobs.reconcile(identifier, result["session_id"])

    def tools(self, session_id):
        session = self.store.get_session(session_id)
        return self._public([{**self.catalog.get(name).spec.model_schema()["function"], "name": name,
            "enabled": name in session["config"].get("enabled_tools", []),
            "authorized": name in session["config"].get("allowed_tools", []),
            "network": self.catalog.get(name).spec.network,
            "requires_approval": self.catalog.get(name).spec.requires_approval} for name in self.catalog.names()])

    def enable_tool(self, session_id, name):
        self._prime(); return self._public(SessionToolset(self.store, self.catalog, session_id).enable(name))
    def disable_tool(self, session_id, name):
        self._prime(); return self._public(SessionToolset(self.store, self.catalog, session_id).disable(name))
    def set_profile(self, session_id, name):
        self._prime(); return self._public(apply_profile(self.store, self.catalog, session_id, name))
    def authorize_analysis(self, session_id):
        """Host-only opt-in for sessions created before analysis tools existed."""
        self._prime()
        with self.store.transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown session")
            runs = conn.execute("SELECT status FROM runs WHERE session_id=?", (session_id,)).fetchall()
            if any(run[0] not in TERMINAL_STATUSES for run in runs):
                raise RunBusy("Authorize analysis between completed/cancelled runs")
            config = json.loads(row["config"])
            selected_analysis_tools = set(ANALYSIS_TOOLS) | set(PUBLIC_CODE_TOOLS)
            config["allowed_tools"] = sorted(set(config.get("allowed_tools", [])) | selected_analysis_tools)
            config["enabled_tools"] = sorted(set(config.get("enabled_tools", [])) | selected_analysis_tools)
            revision = row["revision"] + 1
            conn.execute("UPDATE sessions SET config=?,revision=? WHERE id=?",
                (canonical_json(self.store.redact(config)), revision, session_id))
            conn.execute("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,NULL,?,?,?)",
                (session_id, "analysis.authorized", canonical_json({"tools": sorted(selected_analysis_tools),
                 "revision": revision}), datetime.now(timezone.utc).isoformat()))
        return self._public(self.store.get_session(session_id))
    def set_model(self, session_id, name):
        with self._lock:
            self._prime(); return self.models.select(session_id, name)
    def approvals(self, session_id): return self._public(self.executor.pending_approvals(session_id))
    def decide_approval(self, session_id, run_id, call_id, approve):
        self._prime(); return self._public(self.executor.decide_approval(session_id, run_id, call_id, approve))
    def pending_inputs(self, run_id): return self._public(self.journal.pending_inputs(run_id))
    def answer(self, run_id, request_id, text):
        self._prime(); self.journal.answer(run_id, request_id, text)
        return {"run_id": run_id, "request_id": request_id, "status": "answered"}
    def jobs(self, session_id): return self._public(self.native.jobs.list(session_id))
    def cancel_job(self, session_id, job_id):
        self._prime(); return self._public(self.native.jobs.cancel(job_id, session_id))
    def cancel(self, run_id):
        self._prime()
        result = self.journal.cancel(run_id)
        for job in self.native.jobs.list(result["session_id"]):
            if job["run_id"] == run_id and job["status"] not in JOB_TERMINAL: self.native.jobs.cancel(job["id"], result["session_id"])
        return self._public(result)
    def trace(self, session_id, destination=None, run_id=None):
        self._prime()
        from .trace_export import export_trace
        if destination is not None:
            destination = Path(destination).expanduser()
            if not destination.is_absolute(): destination = self.project_root / destination
        return export_trace(self.store, session_id, destination, run_id)
    def close(self):
        with self._lock:
            if self._closed.is_set(): return
            self._closed.set()
            self.models.close()
            self.store.close()
