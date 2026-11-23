"""Thin argparse client for the shared AgentService; no model/tool orchestration."""

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
from threading import Event
import unicodedata

from .application import AgentService
from .application_config import PROJECT_ROOT
from .cli_defaults import BUILTIN_DEFAULTS, ENVIRONMENT_NAMES, load_cli_defaults
from .contracts import Budget, ModelReply
from .defaults import DEFAULT_PROFILE
from .models import ScriptedPort
from .profiles import PROFILES
from .tools.catalog import canonical_json
from .tools.policy import _credential_path


def terminal_text(value):
    """Display-only escaping; never rewrite accepted report artifacts."""
    result = []
    for char in str(value):
        code = ord(char)
        if char not in "\n\t" and (code < 32 or 127 <= code <= 159 or unicodedata.category(char) in {"Cf", "Cs"}):
            result.append(f"\\x{code:02x}" if code <= 255 else f"\\u{code:04x}")
        else: result.append(char)
    return "".join(result)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("Duplicate JSON configuration key")
        result[key] = value
    return result


def _read_json(value):
    selected = Path(value).expanduser()
    if not selected.is_absolute(): selected = PROJECT_ROOT / selected
    if ".." in selected.parts or _credential_path(selected) or any(path.is_symlink() for path in (selected, *selected.parents)):
        raise PermissionError("Configuration requires an explicit noncredential, non-symlink JSON file")
    descriptor = os.open(selected, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 131072:
            raise ValueError("Configuration requires a regular file of at most128KiB")
        raw = handle.read(131073)
        after = os.fstat(handle.fileno())
        if len(raw) > 131072 or (before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError("Configuration changed or exceeded128KiB")
    result = json.loads(raw, object_pairs_hook=_object)
    canonical_json(result)
    return result


def _positive_integer(value):
    try: parsed = int(value)
    except ValueError: raise argparse.ArgumentTypeError("Must be a positive integer") from None
    if parsed <= 0: raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def _positive_seconds(value):
    try: parsed = float(value)
    except ValueError: raise argparse.ArgumentTypeError("Must be finite and positive") from None
    if not math.isfinite(parsed) or parsed <= 0: raise argparse.ArgumentTypeError("Must be finite and positive")
    return parsed


def parser(defaults=None):
    from .research_entry import install_profile
    install_profile()
    from .quality_entry import install_quality_profile
    install_quality_profile()
    from .evidence_bound_entry import install_bound_profile
    install_bound_profile()
    effective = dict(BUILTIN_DEFAULTS)
    if defaults is not None:
        if not isinstance(defaults, dict) or set(defaults) - set(effective): raise ValueError("Unknown CLI defaults")
        effective.update(defaults)
    command = argparse.ArgumentParser(prog="adj-partner", description="One durable scientific conversation service; ordinary chat is not scientific validation.",
        epilog="Examples:\n  adj-partner --state-dir ./private-state new --profile protein\n  adj-partner --state-dir ./private-state run SESSION 'Inspect sequence'\n  adj-partner --env-file /explicit/provider.env resume RUN\n  adj-partner --offline-demo new\nRelative host/config paths anchor to the source checkout; dotenv is never discovered automatically.\nConnecting --mcp-config executes explicitly configured trusted local servers; it does not authorize their tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument("--state-dir", default=str(Path.home() / ".local/state/adj-partner"), help="Private state directory (prefer an explicit path)")
    command.add_argument("--env-file", help="Explicit provider/runtime dotenv file, never auto-discovered")
    command.add_argument("--dataset-dir", help="Explicit reviewed DEG dataset root; access also requires --allow-root")
    command.add_argument("--mcp-config", help="Explicit trusted MCPServerConfig JSON list; may execute configured servers")
    command.add_argument("--model", dest="selected_model", help="Explicit model choice for new/run/resume/tui, only at a safe boundary")
    command.add_argument("--request-output-tokens", type=_positive_integer, default=effective["request_output_tokens"],
        help=f"Per-model-request output allowance (default:{effective['request_output_tokens']}; env:{ENVIRONMENT_NAMES['request_output_tokens']}; also limited by remaining run budget)")
    command.add_argument("--request-timeout-seconds", type=_positive_seconds, default=effective["request_timeout_seconds"],
        help=f"Per-model-request timeout (default:{effective['request_timeout_seconds']:g}; env:{ENVIRONMENT_NAMES['request_timeout_seconds']}; also limited by remaining active run time)")
    command.add_argument("--offline-demo", action="store_true", help="Explicit finite scripted demonstration; no live model")
    command.add_argument("--json", action="store_true", help="Machine-readable escaped JSON output")
    sub = command.add_subparsers(dest="command")
    ui = sub.add_parser("tui", help="Interactive Textual client (default)"); ui.add_argument("--session")
    new = sub.add_parser("new", help="Create a session with explicit host permissions/budget")
    new.add_argument("--profile", choices=list(PROFILES), default=DEFAULT_PROFILE)
    new.add_argument("--instructions", default="")
    new.add_argument("--allow-root", action="append", default=[])
    new.add_argument("--allow-network", action="store_true")
    new.add_argument("--allow-tool", action="append")
    for name, destination, kind in (("max-steps","max_steps",int),("max-input-tokens","max_input_tokens",int),
            ("max-output-tokens","max_output_tokens",int),("wall-time-seconds","wall_time_seconds",float)):
        value = effective[destination]
        shown = f"{value:g}" if isinstance(value, float) else str(value)
        new.add_argument("--"+name,type=kind,default=value,
            help=f"Default:{shown}; env:{ENVIRONMENT_NAMES[destination]}")
    sub.add_parser("sessions", help="List/reopen durable session identities")
    molecules = sub.add_parser("molecule-reports", help="Generate one dose and experiment report per molecule in a frozen bundle")
    molecules.add_argument("--bundle", required=True)
    molecules.add_argument("--output", required=True)
    molecules.add_argument("--workers", type=int, choices=range(1,9), default=4)
    molecules.add_argument("--retry-failed", action="store_true", help="Create new attempts for terminal failures; preserve earlier receipts")
    run = sub.add_parser("run", help="Submit and drive one conversation turn")
    run.add_argument("session_id"); run.add_argument("text")
    criteria = run.add_mutually_exclusive_group()
    criteria.add_argument("--research", action="store_true", help="Require a source-backed, limited, structurally accepted report")
    criteria.add_argument("--acceptance", help="Exact strict acceptance JSON file")
    run.add_argument("--required-section", action="append", default=[])
    run.add_argument("--require-data-artifact", action="store_true")
    resume = sub.add_parser("resume", help="Continue the same run and original pending calls"); resume.add_argument("run_id")
    status = sub.add_parser("status", help="Inspect a session or run plus actual task requirements"); status.add_argument("identifier")
    tools = sub.add_parser("tools", help="List or enable/disable within host authorization"); tools.add_argument("session_id")
    changes = tools.add_mutually_exclusive_group(); changes.add_argument("--enable"); changes.add_argument("--disable")
    analysis_authorize = sub.add_parser("analysis-authorize",
        help="Host-authorize the six built-in analysis tools for a legacy session between runs")
    analysis_authorize.add_argument("session_id")
    for name in ("profile", "model"):
        item = sub.add_parser(name, help=f"Inspect/change {name} between runs"); item.add_argument("session_id"); item.add_argument("name", nargs="?")
    for name in ("approve", "deny"):
        item = sub.add_parser(name, help="Explicit host decision for the exact requested call")
        for field in ("session_id","run_id","call_id"): item.add_argument(field)
    answer = sub.add_parser("answer", help="Provide actual host/user clarification")
    for field in ("run_id","request_id","text"): answer.add_argument(field)
    jobs = sub.add_parser("jobs", help="Inspect jobs or request cancellation"); jobs.add_argument("session_id"); jobs.add_argument("--cancel", dest="job_id")
    cancel = sub.add_parser("cancel", help="Persist run cancellation, then request associated local-job cancellation"); cancel.add_argument("run_id")
    trace = sub.add_parser("trace", help="Export one consistent public trace with fixed manifest.json")
    trace.add_argument("session_id"); trace.add_argument("--run-id"); trace.add_argument("--destination")
    return command


def _acceptance(args):
    if args.acceptance:
        if args.required_section or args.require_data_artifact: raise ValueError("Use either an exact criteria file or research flags")
        return _read_json(args.acceptance)
    if args.research or args.required_section or args.require_data_artifact:
        return {"require_report": True, "require_evidence": True, "require_limitations": True,
                "require_data_artifact": args.require_data_artifact, "required_sections": args.required_section}
    return None


def _dispatch(service, args, stop):
    action = args.command or "tui"
    if action == "new":
        session = service.create_session(profile=args.profile, instructions=args.instructions, allowed_roots=args.allow_root,
            allow_network=args.allow_network, allowed_tools=args.allow_tool,
            budget=Budget(args.max_steps,args.max_input_tokens,args.max_output_tokens,args.wall_time_seconds))
        if args.selected_model: service.set_model(session["id"],args.selected_model)
        return service.session(session["id"])
    if action == "sessions": return service.sessions()
    if action == "molecule-reports":
        if args.offline_demo: raise ValueError("Molecule scientific delivery requires a real model or an explicit injected test service")
        from .molecule_reports import run_bundle
        return run_bundle(args.bundle,args.output,model=args.selected_model,workers=args.workers,
                          retry_failed=args.retry_failed,service=service)
    if action == "run":
        if args.selected_model: service.set_model(args.session_id,args.selected_model)
        run = service.submit(args.session_id,args.text,acceptance=_acceptance(args))
        return service.drive(run["id"],stop_event=stop)
    if action == "resume":
        if args.selected_model:
            sid = service.run(args.run_id)["session_id"]
            if service.model_info(sid)["model"] != args.selected_model: service.set_model(sid,args.selected_model)
        return service.drive(args.run_id,stop_event=stop)
    if action == "status":
        if args.identifier.startswith("run_"):
            run = service.run(args.identifier)
            return {"run":run,"task":service.task(run["task_id"]),"model":service.model_info(run["session_id"]),
                    "pending_inputs":service.pending_inputs(run["id"]),"approvals":service.approvals(run["session_id"])}
        return {"session":service.session(args.identifier),"tasks":service.tasks(args.identifier),
                "runs":service.runs(args.identifier),"model":service.model_info(args.identifier)}
    if action == "tools":
        if args.enable: service.enable_tool(args.session_id,args.enable)
        if args.disable: service.disable_tool(args.session_id,args.disable)
        return service.tools(args.session_id)
    if action == "analysis-authorize": return service.authorize_analysis(args.session_id)
    if action == "profile": return service.set_profile(args.session_id,args.name) if args.name else service.session(args.session_id)
    if action == "model": return service.set_model(args.session_id,args.name) if args.name else service.model_info(args.session_id)
    if action in {"approve","deny"}: return service.decide_approval(args.session_id,args.run_id,args.call_id,action=="approve")
    if action == "answer": return service.answer(args.run_id,args.request_id,args.text)
    if action == "jobs": return service.cancel_job(args.session_id,args.job_id) if args.job_id else service.jobs(args.session_id)
    if action == "cancel": return service.cancel(args.run_id)
    if action == "trace": return service.trace(args.session_id,args.destination,args.run_id)
    if args.json: raise ValueError("Interactive TUI is not JSON mode; select a command")
    sid = getattr(args,"session",None)
    if args.selected_model:
        sid = sid or service.create_session()["id"]
        service.set_model(sid,args.selected_model)
    from .tui import AdjPartnerApp
    AdjPartnerApp(service,session_id=sid).run()
    return None


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--env-file")
    selected, _ = bootstrap.parse_known_args(raw_argv)
    try: defaults = load_cli_defaults(PROJECT_ROOT, selected.env_file)
    except ValueError as exc: parser().error(str(exc))
    args = parser(defaults).parse_args(raw_argv)
    service, stop = None, Event()
    try:
        port = ScriptedPort([ModelReply(content="Offline demonstration only: no live model was called; this is not scientific validation.",
            usage={"input_tokens":0,"output_tokens":0}) for _ in range(16)]) if args.offline_demo else None
        config = None
        if args.mcp_config:
            from .tools.mcp import MCPServerConfig
            raw = _read_json(args.mcp_config)
            if not isinstance(raw,list) or len(raw)>32: raise ValueError("MCP configuration requires at most32 explicit server objects")
            config = [MCPServerConfig(**item) for item in raw]
        service = AgentService(args.state_dir, model=port, env_file=args.env_file, dataset_dir=args.dataset_dir, mcp_config=config,
            request_output_tokens=args.request_output_tokens, request_timeout_seconds=args.request_timeout_seconds)
        result = _dispatch(service,args,stop)
        if result is not None:
            output = json.dumps(result, ensure_ascii=args.json, allow_nan=False, indent=None if args.json else 2)
            print(output if args.json else terminal_text(output))
        status = result.get("status") if isinstance(result,dict) else None
        return {"waiting":2,"failed":3,"budget_exhausted":4,"cancelled":5}.get(status,0)
    except KeyboardInterrupt:
        stop.set()
        print(json.dumps({"status":"detached","note":"No further model loop continues. Independent jobs may remain; reconnect with resume RUN."}))
        return 130
    except Exception as exc:
        message = service.store.redact(str(exc)) if service else "Application setup failed; check explicit host configuration."
        result = {"error":{"code":getattr(exc,"code",type(exc).__name__),"message":message}}
        print(json.dumps(result,ensure_ascii=True,allow_nan=False))
        return 1
    finally:
        stop.set()
        if service: service.close()
