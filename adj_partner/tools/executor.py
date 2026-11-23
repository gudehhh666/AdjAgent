"""Exactly-once intent ownership, policy checks, and one-shot host approvals.

Exactly-once invocation is not exactly-once external effect: a dead worker after
intent/approval consumption has an unknown outcome and is never silently replayed.
Native adapters execute cooperatively; slow work belongs in bounded adapters/jobs.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from threading import RLock

from ..contracts import ToolCall, ToolResult
from .catalog import canonical_json, validator
from .policy import ToolCancelled, ToolContext


RESULT_STATUSES = frozenset({"ok", "error", "denied", "cancelled", "uncertain", "approval_required",
                             "pending", "running", "waiting"})


def failure(code, message, status="error"):
    return ToolResult(status=status, error={"code": code, "message": message})


class ToolFailure(Exception):
    """Adapter error retaining a stable code/status through the shared executor."""

    def __init__(self, code, message, status="error", details=None):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details


class ToolNotInvoked(Exception):
    """Trusted host guard stopped before invocation; never an adapter outcome.

    Only the guard's pre-invocation boundary may finalize this known no-effect
    outcome. Arbitrary exceptions after invocation remain uncertain on recovery.
    """


def _not_invoked():
    return failure("client_detached_not_invoked",
        "Client detached before invocation; no effect started. Replan after reconnect.", "cancelled")


def _fingerprint(name, args):
    return hashlib.sha256(canonical_json({"name": name, "arguments": args}).encode()).hexdigest()


class ToolExecutor:
    def __init__(self, store, catalog):
        self.store, self.catalog = store, catalog
        catalog.prepare_store(store)
        self._ephemeral = {}
        self._lock = RLock()
        with store.transaction(write=True) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS tool_approvals (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                run_id TEXT NOT NULL REFERENCES runs(id), call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL, args TEXT NOT NULL, fingerprint TEXT NOT NULL,
                toolset_revision INTEGER NOT NULL, definition_hash TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(run_id,call_id),
                FOREIGN KEY(run_id,call_id) REFERENCES actions(run_id,call_id)
            )""")

    def _scope(self, session_id, run_id):
        session = self.store.get_session(session_id)
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise ValueError("Run does not belong to session")
        return session

    def _policy(self, name, session, expected_revision=None, expected_hash=None):
        if expected_revision is not None and session["revision"] != expected_revision:
            return failure("stale_revision", "Session policy changed; obtain a fresh tool proposal", "denied")
        try:
            spec = self.catalog.get(name).spec
        except KeyError:
            return failure("unknown_tool", f"Unknown tool: {name}", "denied")
        if expected_hash is not None and spec.definition_hash != expected_hash:
            return failure("schema_drift", "Tool definition changed; obtain a fresh tool proposal", "denied")
        config = session["config"]
        if name not in config.get("allowed_tools", []):
            return failure("tool_not_allowed", "Tool is not host-authorized for this session", "denied")
        if name not in config.get("enabled_tools", []):
            return failure("tool_disabled", "Tool is not enabled in this session", "denied")
        if spec.network and not config.get("allow_network", False):
            return failure("network_denied", "Network access is not host-authorized", "denied")
        return None

    def _arguments(self, spec, arguments):
        if not isinstance(arguments, dict):
            return failure("input_schema", "Tool arguments must be a JSON object")
        try:
            canonical_json(arguments)
            validator(spec.input_schema).validate(arguments)
        except Exception as exc:
            return failure("input_schema", f"Invalid tool arguments: {exc}")
        return None

    def _reject_event(self, session_id, run_id, call, result):
        identifier, name = getattr(call, "id", None), getattr(call, "name", None)
        self.store.append_event(session_id, "tool.rejected", {
            "call_id": identifier.encode("utf-8", errors="backslashreplace").decode("utf-8") if isinstance(identifier, str) else None,
            "tool_name": name.encode("utf-8", errors="backslashreplace").decode("utf-8") if isinstance(name, str) else None,
            "result": result.to_dict()}, run_id)
        return result

    def _existing(self, action):
        if action["finished_at"] is not None:
            return ToolResult(**action["result"])
        if action["status"] == "approval_required":
            with self.store.transaction() as conn:
                row = conn.execute("SELECT status FROM tool_approvals WHERE run_id=? AND call_id=?",
                                   (action["run_id"], action["call_id"])).fetchone()
            if row and row["status"] in {"pending", "approved"}:
                return ToolResult(**action["result"])
        if action["status"] in {"pending", "running", "waiting"} and action["result"] is not None:
            return ToolResult(**action["result"])
        return failure("action_uncertain", "An earlier invocation may have started; reconcile its outcome, do not replay", "uncertain")

    def _finish(self, run_id, call_id, result):
        action = self.store.finish_action(run_id, call_id, result.to_dict())
        return ToolResult(**action["result"])

    def _owned_guard(self, guard, run_id, call_id):
        try:
            if guard is not None: guard()
        except ToolNotInvoked:
            self._finish(run_id, call_id, _not_invoked())
            raise

    def _validate_result(self, binding, result):
        """One non-recursive validation boundary for returned and raised outcomes."""
        try:
            if not isinstance(result, ToolResult):
                return failure("output_contract", "Adapter must return ToolResult")
            if (not isinstance(result.status, str) or result.status not in RESULT_STATUSES
                    or result.status == "approval_required"  # Host-owned state, never an adapter grant.
                    or not isinstance(result.artifacts, list)
                    or any(not isinstance(item, dict) for item in result.artifacts)
                    or (result.job_id is not None and not isinstance(result.job_id, str))
                    or (result.status == "ok" and result.error is not None)
                    or (result.status in {"error", "denied", "cancelled", "uncertain", "approval_required"}
                        and result.error is None)
                    or (result.error is not None and (not isinstance(result.error, dict)
                        or not isinstance(result.error.get("code"), str)
                        or not isinstance(result.error.get("message"), str)))):
                return failure("output_contract", "Adapter returned an invalid or inconsistent ToolResult envelope")
        except Exception:
            return failure("output_contract", "Adapter result envelope could not be inspected")
        try:
            # Parse back to plain JSON values so persistence cannot re-run an
            # adapter object's failing/custom deepcopy or mapping behavior.
            normalized = ToolResult(**json.loads(canonical_json(result.to_dict())))
        except Exception:
            return failure("output_json", "Tool result is not finite, serializable JSON")
        if normalized.status == "ok" and binding.spec.output_schema is not None:
            try:
                validator(binding.spec.output_schema).validate(normalized.data)
            except Exception as exc:
                return failure("output_schema", f"Tool result violates its output schema: {exc}")
        return normalized

    def _invoke(self, binding, context, arguments):
        try:
            context.check_cancelled()
            result = binding.handler(context, **deepcopy(arguments))
        except ToolCancelled as exc:
            result = failure("cancelled", str(exc), "cancelled")
        except ToolFailure as exc:
            result = failure(exc.code, str(exc), exc.status)
            if exc.details is not None:
                result.error["details"] = exc.details
        except PermissionError as exc:
            result = failure("permission_denied", str(exc), "denied")
        except Exception as exc:
            result = failure("provider_error", f"{type(exc).__name__}: {exc}")
        return self._validate_result(binding, result)

    def execute(self, call, session_id, run_id, expected_revision=None,
                cancel_event=None, expected_definition_hash=None, execution_guard=None):
        if execution_guard is not None: execution_guard()
        self.catalog.prepare_store(self.store)
        try:
            session = self._scope(session_id, run_id)
        except (ValueError, KeyError) as exc:
            return failure("scope_denied", str(exc), "denied")
        try:
            if (not isinstance(call, ToolCall) or not isinstance(call.id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", call.id)
                    or not isinstance(call.name, str) or not call.name):
                raise ValueError("Malformed tool call or provider call identifier")
            arguments = deepcopy(call.arguments)
            canonical_json({"id": call.id, "name": call.name, "arguments": arguments})
        except (TypeError, ValueError, RecursionError) as exc:
            return self._reject_event(session_id, run_id, call, failure("invalid_call", str(exc)))
        self.store.append_event(session_id, "tool.attempt", {
            "call_id": call.id, "tool_name": call.name, "arguments": arguments,
            "expected_revision": expected_revision, "expected_definition_hash": expected_definition_hash}, run_id)
        # Validation computes a prospective outcome, but only the journal winner
        # may save it or invoke a side effect. Existing outcomes remain immutable.
        outcome = self._policy(call.name, session, expected_revision, expected_definition_hash)
        if outcome is None:
            outcome = self._arguments(self.catalog.get(call.name).spec, arguments)
        if execution_guard is not None: execution_guard()
        try:
            action, created = self.store.claim_action(run_id, call.id, call.name, arguments, session["revision"])
        except ValueError as exc:
            return self._reject_event(session_id, run_id, call, failure("call_id_conflict", str(exc)))
        if not created:
            return self._existing(action)
        if outcome is not None:
            return self._finish(run_id, call.id, outcome)
        binding = self.catalog.get(call.name)
        # Recheck current authorization after acquiring durable intent ownership.
        outcome = self._policy(call.name, self.store.get_session(session_id), session["revision"], binding.spec.definition_hash)
        if outcome is not None:
            return self._finish(run_id, call.id, outcome)
        context = ToolContext(self.store, session_id, run_id, session["revision"], cancel_event, call_id=call.id)
        context.execution_guard = execution_guard
        try:
            context.check_cancelled()
        except ToolCancelled as exc:
            return self._finish(run_id, call.id, failure("cancelled", str(exc), "cancelled"))
        except PermissionError as exc:
            return self._finish(run_id, call.id, failure("run_terminal", str(exc), "denied"))
        if call.name == "web_search":
            from .health import suspended_tools
            if call.name in suspended_tools(self.store, run_id):
                return self._finish(run_id, call.id, failure("tool_suspended_for_run",
                    "web_search returned web_search_quota in this run. Use available evidence or another "
                    "authorized source; a new run may try again after host configuration repair.", "denied"))
        if binding.spec.requires_approval:
            self._owned_guard(execution_guard, run_id, call.id)
            return self._request_approval(call, arguments, session, run_id, binding.spec)
        self._owned_guard(execution_guard, run_id, call.id)
        return self._finish(run_id, call.id, self._invoke(binding, context, arguments))

    def _request_approval(self, call, arguments, session, run_id, spec):
        # Publish the pending action before making a host decision discoverable.
        # A host can resolve the request as soon as its row/event appears; never
        # overwrite that final result with a late approval_required response.
        self._finish(run_id, call.id, failure(
            "approval_required", "Host approval is required for this exact operation", "approval_required"))
        with self._lock:
            self._ephemeral[(run_id, call.id)] = deepcopy(arguments)
        with self.store.transaction(write=True) as conn:
            conn.execute("INSERT INTO tool_approvals VALUES(?,?,?,?,?,?,?,?, 'pending',?)", (
                session["id"], run_id, call.id, spec.name, canonical_json(self.store.redact(arguments)),
                _fingerprint(spec.name, arguments), session["revision"], spec.definition_hash,
                datetime.now(timezone.utc).isoformat()))
        self.store.append_event(session["id"], "tool.approval_requested", {
            "call_id": call.id, "tool_name": spec.name, "arguments": arguments,
            "toolset_revision": session["revision"], "definition_hash": spec.definition_hash}, run_id)
        return self._existing(self.store.get_action(run_id, call.id))

    def pending_approvals(self, session_id):
        self.store.get_session(session_id)
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT * FROM tool_approvals WHERE session_id=? AND status='pending' ORDER BY rowid",
                                (session_id,)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["args"] = json.loads(value["args"])
            result.append(self.store.redact(value))
        return result

    def decide_approval(self, session_id, run_id, call_id, approve):
        self._scope(session_id, run_id)
        if type(approve) is not bool:
            raise ValueError("Approval must be an explicit host boolean decision")
        status = "approved" if approve else "denied"
        with self.store.transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM tool_approvals WHERE session_id=? AND run_id=? AND call_id=?",
                               (session_id, run_id, call_id)).fetchone()
            if row is None or row["status"] != "pending":
                raise ValueError("Approval is absent or already decided/consumed")
            conn.execute("UPDATE tool_approvals SET status=? WHERE run_id=? AND call_id=?", (status, run_id, call_id))
        self.store.append_event(session_id, "tool.approval_decided", {"call_id": call_id, "status": status}, run_id)
        if not approve:
            with self._lock:
                self._ephemeral.pop((run_id, call_id), None)
            self._finish(run_id, call_id, failure("approval_denied", "Host denied this operation", "denied"))
        return {"session_id": session_id, "run_id": run_id, "call_id": call_id, "status": status}

    def resume_approved(self, session_id, run_id, call_id, cancel_event=None, execution_guard=None):
        if execution_guard is not None: execution_guard()
        self.catalog.prepare_store(self.store)
        try:
            self._scope(session_id, run_id)
            action = self.store.get_action(run_id, call_id)
        except (KeyError, ValueError) as exc:
            return failure("scope_denied", str(exc), "denied")
        if action is None:
            return failure("approval_missing", "No operation exists for this approval", "denied")
        if (action["finished_at"] is not None or
                (action["status"] in {"pending", "running", "waiting"} and action["result"] is not None)):
            # An approved adapter may already have handed off a durable job.
            # Its recorded outcome is authoritative even though it is nonterminal;
            # consumed permission is uncertain only when no such outcome exists.
            return ToolResult(**action["result"])
        with self.store.transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM tool_approvals WHERE session_id=? AND run_id=? AND call_id=?",
                               (session_id, run_id, call_id)).fetchone()
            if row is None:
                return failure("approval_missing", "No matching host approval exists", "denied")
            approval = dict(row)
            if approval["status"] == "pending":
                return failure("approval_required", "Host approval is still pending", "approval_required")
            if approval["status"] == "consumed":
                # Another consumer may be running; do not finalize its action.
                return failure("action_uncertain", "Approval was consumed; reconcile the earlier operation", "uncertain")
            if approval["status"] == "denied":
                outcome = failure("approval_denied", "Host denied this operation", "denied")
            elif approval["status"] == "stale":
                # The process that invalidated the approval owns finalization.
                # Another observer must not race it with a different error result.
                return failure("action_uncertain", "Another consumer is finalizing the stale approval; reload its outcome", "uncertain")
            else:
                raw_session = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
                session = dict(raw_session)
                session["config"] = json.loads(session["config"])
                outcome = self._policy(approval["tool_name"], session, approval["toolset_revision"], approval["definition_hash"])
                if outcome is not None:
                    outcome = failure("stale_approval", "Approval no longer matches the current tool definition or session policy", "denied")
                with self._lock:
                    arguments = deepcopy(self._ephemeral.get((run_id, call_id), json.loads(approval["args"])))
                if outcome is None and _fingerprint(approval["tool_name"], arguments) != approval["fingerprint"]:
                    outcome = failure("approval_arguments_unavailable", "Sanitized persisted arguments cannot reproduce the approved operation; submit a new call", "denied")
                if outcome is None:
                    outcome = self._arguments(self.catalog.get(approval["tool_name"]).spec, arguments)
                conn.execute("UPDATE tool_approvals SET status=? WHERE run_id=? AND call_id=?",
                             ("stale" if outcome is not None else "consumed", run_id, call_id))
        with self._lock:
            self._ephemeral.pop((run_id, call_id), None)
        if outcome is not None:
            return self._finish(run_id, call_id, outcome)
        self.store.append_event(session_id, "tool.approval_consumed", {"call_id": call_id}, run_id)
        binding = self.catalog.get(approval["tool_name"])
        outcome = self._policy(binding.spec.name, self.store.get_session(session_id),
                               approval["toolset_revision"], approval["definition_hash"])
        if outcome is None:
            context = ToolContext(self.store, session_id, run_id, approval["toolset_revision"], cancel_event, call_id=call_id)
            context.execution_guard = execution_guard
            self._owned_guard(execution_guard, run_id, call_id)
            outcome = self._invoke(binding, context, arguments)
        return self._finish(run_id, call_id, outcome)
