"""One bounded durable loop: model decision -> host tool -> observed result.

The loop owns no domain workflow. Every request/reply checkpoint precedes effects;
tool protocol groups commit together. Unknown effects are never silently replayed.
"""

from copy import deepcopy
import json
import math
import re
from threading import Event, Thread
import time

from ..contracts import ModelReply, ToolCall, ToolResult, new_id
from ..models.base import ProviderFailure, supports_tool_choice
from ..research import ResearchStore
from ..research.acceptance import (EXECUTION_BOUND_DELIVERY, REQUIRE_BOUND_EXECUTION_WITNESS,
    REQUIRE_EXECUTION_ACTION, ReportNotAccepted, RESPONSE_SCHEMA, research_acceptance, task_binding,
    response_repair_seed, response_scalar_fingerprint, response_schema_issues)
from ..storage import TERMINAL_STATUSES
from ..tools import ToolExecutor
from ..tools.catalog import canonical_json
from ..tools.executor import ToolNotInvoked, failure
from ..tools.policy import SessionToolset
from .context import ContextOverflow, ContextView
from .journal import LeaseLost, RunJournal, task_digest
from .model_invocations import InvocationLedger, fenced_guard, inherit_guard
from .tools import register_runtime_tools
from .execution_delivery import verify_execution_delivery
from .execution_witness import (bind_execution_inputs, validate_execution_binding,
                                verify_execution_witness)
from .public_code_validation import bind_public_code_validation, verify_public_code_validation
from .reference_feedback import (KEY as REFERENCE_KEY, BINDING_KEY as REFERENCE_BINDING,
    bind_reference_feedback, verify_reference_binding, prepare_reference_feedback,
    revision_projection, initial_projection, sha as reference_sha)


PENDING = {"pending", "running", "waiting", "approval_required"}


class _BoundaryStop(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason


class _ClientDetached(ToolNotInvoked):
    pass


class AgentEngine:
    def __init__(self, store, catalog, model, *, executor=None, research=None,
                 resolve_job=None, recover_action=None, max_request_seconds=120, max_output_tokens=4096):
        if (isinstance(max_request_seconds, bool) or not isinstance(max_request_seconds, (int, float))
                or not math.isfinite(max_request_seconds) or not 0 < max_request_seconds <= 600):
            raise ValueError("Request timeout must be finite and within (0,600]")
        if type(max_output_tokens) is not int or max_output_tokens <= 0: raise ValueError("Invalid output allowance")
        self.store, self.catalog, self.model = store, catalog, model
        for service in (executor, research):
            if service is not None and (service.store.root != store.root or service.store.db_path != store.db_path):
                raise ValueError("Runtime services must share one Store root/database")
        self._prime()
        self.journal = RunJournal(store)
        register_runtime_tools(catalog)
        self.executor = executor if executor is not None and executor.store is store else ToolExecutor(store, catalog)
        self.research = research if research is not None and research.store is store else ResearchStore(
            store, assessor=research.assessor if research is not None else None)
        self.context = ContextView(store)
        self.resolve_job, self.recover_action = resolve_job, recover_action
        self.max_request_seconds, self.max_output_tokens = max_request_seconds, max_output_tokens

    def _prime(self):
        self.store.register_secrets(getattr(self.model, "secrets", ()))
        self.catalog.prepare_store(self.store)

    def start(self, session_id, goal, *, acceptance=None, budget=None, execution_binding=None):
        self._prime()
        session = self.store.get_session(session_id)
        run = self.journal.start(session_id, goal, {} if acceptance is None else acceptance,
                                 budget or session["config"].get("budget", {}))
        accepted = run["task_id"] and self.store.get_task(run["task_id"])
        if REFERENCE_KEY in accepted["acceptance"]:
            state = deepcopy(run["state"])
            try:
                state[REFERENCE_BINDING] = bind_reference_feedback(self.store, run["id"])
            except Exception as exc:
                state["reference_binding_error"] = type(exc).__name__
            run = self.store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
        if "public_code_validation" in accepted["acceptance"]:
            state = deepcopy(run["state"])
            try:
                state["public_code_validation_binding"] = bind_public_code_validation(self.store, run["id"])
            except Exception as exc:
                state["public_code_binding_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
            run = self.store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
        if accepted["acceptance"].get(REQUIRE_BOUND_EXECUTION_WITNESS) is True and execution_binding is None:
            state = deepcopy(run["state"])
            state["execution_binding_missing"] = True
            return self.store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
        if execution_binding is not None:
            response_schema = accepted["acceptance"].get(RESPONSE_SCHEMA)
            # The host may provide a complete binding, or a task-start
            # descriptor. The latter lets the journal allocate the task ID
            # before the binding records its authoritative task digest.
            if (isinstance(execution_binding, dict)
                    and set(execution_binding) == {"allowed_roots", "input_paths"}):
                execution_binding = bind_execution_inputs(
                    task_binding(accepted),
                    execution_binding["allowed_roots"], execution_binding["input_paths"], response_schema=response_schema)
            validate_execution_binding(execution_binding,
                                       task_binding=task_binding(accepted),
                                       response_schema=response_schema,
                                       session_roots=session["config"].get("allowed_roots", []))
            state = deepcopy(run["state"])
            state["execution_binding"] = execution_binding
            run = self.store.update_run(run["id"], {"state": state}, expected_revision=run["revision"])
        return run

    @staticmethod
    def _usage(previous, actual, request):
        usage = deepcopy(previous)
        complete = True
        for key, fallback in (("input_tokens", request["estimated_input_tokens"]), ("output_tokens", request["max_output_tokens"])):
            value = actual.get(key) if isinstance(actual, dict) else None
            known = type(value) is int and value >= 0
            usage[key] = usage.get(key, 0) + (value if known else 0)
            usage["accounted_" + key] = usage.get("accounted_" + key, 0) + (value if known else fallback)
            if not known: complete = False
        usage["estimated_input_tokens"] = usage.get("estimated_input_tokens", 0) + request["estimated_input_tokens"]
        usage["unknown_requests"] = usage.get("unknown_requests", 0) + int(not complete)
        usage["accounting_policy"] = "Known provider counters; missing counters reserve input estimate/full output allowance, not zero cost"
        return usage

    def _accepted(self, sid, rid):
        for report in reversed(self.research.list_reports(sid, rid)):
            if report["run_id"] != rid:
                continue
            try:
                value = self.research.accepted_report(sid, rid, report["id"])
                delivery = {"kind": "accepted_report", "content": value["content"].decode("utf-8"), "report_id": report["id"],
                    "artifact_id": value["artifact"]["id"], "sha256": value["artifact"]["sha256"],
                    "assurance": "Structural acceptance only; scientific claims retain their recorded qualifications"}
                if value["report"].get("research_design"):
                    from ..research.design import design_status, readable_report, design_evidence_map
                    delivery.update(research_design_status=design_status(value["report"]["research_design"]),
                                    readable_content=readable_report(value["report"]).decode("utf-8"),
                                    research_evidence_map=design_evidence_map(value["report"]))
                return delivery
            except ReportNotAccepted:
                continue
        return None

    def _result(self, call, sid, rid, snapshot, execution_guard):
        execution_guard()
        def invocation_guard():
            execution_guard()
            if self.store.redact(call.arguments) != call.arguments:
                raise _BoundaryStop("failed", "sensitive_tool_arguments_not_replayable")
        inherit_guard(invocation_guard, execution_guard)
        action = self.store.get_action(rid, call.id)
        if action and action["finished_at"] is not None:
            return ToolResult(**self.store.redact(action["result"]))
        if action:
            if action["status"] == "approval_required":
                return self.executor.resume_approved(sid, rid, call.id, execution_guard=invocation_guard)
            result = ToolResult(**action["result"]) if action["result"] else None
            if result and result.job_id and self.resolve_job:
                result = self.resolve_job(sid, rid, result.job_id)
            elif call.name == "ask_user":
                answer = self.journal.input_result(rid, call.id)
                pending = [item for item in self.journal.pending_inputs(rid) if item["call_id"] == call.id]
                if answer is not None: result = ToolResult(data=answer)
                elif pending: result = ToolResult(status="waiting", data={"kind": "user", "request_id": pending[0]["id"]})
            if result is None and self.recover_action:
                result = self.recover_action(sid, rid, call)
            if result is None:
                result = failure("action_uncertain", "Earlier tool intent has no safely recoverable outcome; it was not replayed", "uncertain")
            if result.status not in PENDING: self.store.finish_action(rid, call.id, result.to_dict())
            return result
        invocation_guard()
        if call.name not in snapshot["definition_hashes"]:
            self.store.claim_action(rid, call.id, call.name, call.arguments, snapshot["revision"])
            result = failure("tool_not_exposed", "This tool was not in the model request's registered toolset", "denied")
            self.store.finish_action(rid, call.id, result.to_dict())
            return result
        return self.executor.execute(call, sid, rid, expected_revision=snapshot["revision"],
                                     expected_definition_hash=snapshot["definition_hashes"][call.name], execution_guard=invocation_guard)

    @staticmethod
    def _valid_reply(reply, used_ids):
        if not isinstance(reply, ModelReply) or not isinstance(reply.content, str) or not isinstance(reply.tool_calls, list):
            raise ValueError("Model reply contract malformed")
        if (not isinstance(reply.finish_reason, str) or not reply.finish_reason or len(reply.finish_reason) > 64
                or (reply.refusal is not None and not isinstance(reply.refusal, str))):
            raise ValueError("Malformed model finish/refusal")
        if len(reply.tool_calls) > 32: raise ValueError("Too many tool calls in one reply")
        identifiers = set()
        for call in reply.tool_calls:
            if (not isinstance(call, ToolCall) or not isinstance(call.id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", call.id)
                    or call.id in identifiers or call.id in used_ids or not isinstance(call.name, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", call.name)
                    or not isinstance(call.arguments, dict)):
                raise ValueError("Malformed, duplicate or reused model call ID/arguments")
            identifiers.add(call.id)
        public = reply.to_dict()
        if len(canonical_json(public).encode()) > 2 * 1024 * 1024: raise ValueError("Model public reply exceeds2MiB")
        return public

    def resume(self, run_id, *, stop_event=None):
        self._prime()
        owner = self.journal.acquire(run_id)
        stopped, lost = Event(), Event()
        heartbeat_failure = []

        def raise_heartbeat_failure():
            failure, cause = heartbeat_failure[0] if heartbeat_failure else (
                LeaseLost("Runtime lease heartbeat failed"), None)
            if cause is not None:
                raise failure from cause
            raise failure

        def heartbeat():
            while not stopped.wait(1):
                try: self.journal.renew(run_id, owner)
                except Exception as exc:
                    failure = self.journal.heartbeat_failure_error(exc)
                    heartbeat_failure.append((failure, exc))
                    # Make ownership loss visible before diagnostic I/O, which
                    # may wait on SQLite. Diagnostics are evidence, never a gate.
                    lost.set()
                    try:
                        self.journal.record_heartbeat_failure(run_id, owner, exc)
                    except Exception:
                        pass
                    return
        thread = Thread(target=heartbeat, name="adj-runtime-lease", daemon=True)
        thread.start()
        started = time.monotonic()
        initial = self.store.get_run(run_id)
        elapsed_base = initial["state"].get("elapsed_active_seconds", 0.0)
        sid, state, usage = initial["session_id"], deepcopy(initial["state"]), deepcopy(initial["usage"])
        budget = initial["budget"]
        auxiliary = InvocationLedger(self.store)
        def current_task(): return self.store.get_task(initial["task_id"])
        def strict(): return research_acceptance(current_task()["acceptance"])
        def reference_enabled(): return REFERENCE_KEY in current_task()["acceptance"] or REFERENCE_BINDING in state
        def reference_guard():
            if not reference_enabled():
                return
            try:
                if state.get("reference_binding_error") or REFERENCE_BINDING not in state:
                    raise ValueError("Reference corpus was not bound at start")
                verify_reference_binding(self.store, run_id)
                if state.get("reference_feedback"):
                    projection = revision_projection(self.store, run_id)
                    feedback = state["reference_feedback"]
                    bound = feedback.get("revision_projection_hash")
                    if bound is not None and reference_sha(canonical_json(projection)) != bound:
                        raise ValueError("Reference revision request binding changed")
                    pointer = feedback.get("revision_request")
                    if feedback["revision_state"] in {"requesting", "replied", "completed"} and pointer is None:
                        raise ValueError("Missing reference revision request binding")
                    if pointer is not None:
                        record = initial_projection(self.store, self.store.get_run(run_id), pointer)
                        if {key: record[key] for key in ("messages", "tools")} != projection:
                            raise ValueError("Reference revision artifact differs from intended projection")
            except Exception as exc:
                state["reference_binding_error"] = type(exc).__name__
                raise _BoundaryStop("failed", "reference_binding_invalid") from exc
        def task_changed():
            return state.get("request_task_binding") is not None and task_digest(current_task()) != state["request_task_binding"]

        def elapsed(): return elapsed_base + time.monotonic() - started
        def save(*, status=None, messages=(), event="runtime.checkpoint"):
            if lost.is_set():
                raise_heartbeat_failure()
            state["elapsed_active_seconds"] = elapsed()
            current = self.store.get_run(run_id)
            if current["status"] in TERMINAL_STATUSES:
                status, messages = None, ()
            return self.journal.checkpoint(run_id, owner, state=state, status=status, usage=usage, messages=messages, event=event,
                expected_task_binding=state.get("request_task_binding"), active_deadline=started + budget["wall_time_seconds"] - elapsed_base)
        def finish(status, reason, delivery=None):
            if status == "succeeded" and task_changed(): status, reason, delivery = "failed", "task_changed", None
            if status == "succeeded" and exceeded(): status, reason, delivery = "budget_exhausted", "budget_exhausted", None
            state.update(phase="finished", stop_reason=reason)
            if delivery is not None: state["delivery"] = delivery
            messages = [{"role": "assistant", "content": delivery["content"]}] if delivery else ()
            return save(status=status, messages=messages, event="runtime.finished")
        def exceeded():
            extra = auxiliary.usage(run_id)
            return (elapsed() >= budget["wall_time_seconds"]
                or usage.get("accounted_input_tokens", 0) + extra["accounted_input_tokens"] > budget["max_input_tokens"]
                or usage.get("accounted_output_tokens", 0) + extra["accounted_output_tokens"] > budget["max_output_tokens"])
        def execution_guard():
            if lost.is_set(): raise_heartbeat_failure()
            self.journal.renew(run_id, owner)
            if lost.is_set(): raise_heartbeat_failure()
            if self.store.get_run(run_id)["status"] in TERMINAL_STATUSES: raise _BoundaryStop("cancelled", "host_stopped")
            if exceeded(): raise _BoundaryStop("budget_exhausted", "budget_exhausted")
            if task_changed(): raise _BoundaryStop("failed", "task_changed")
            if stop_event is not None and stop_event.is_set(): raise _ClientDetached()
        fenced_guard(execution_guard, self.journal, run_id, owner)
        execution_guard.remaining_seconds = lambda: budget["wall_time_seconds"] - elapsed()
        def required_witness():
            binding = state.get("execution_binding")
            if binding is None:
                return None
            try:
                return verify_execution_witness(self.store, run_id, binding,
                                                response_schema=current_task()["acceptance"].get(RESPONSE_SCHEMA))
            except Exception as exc:
                state["error"] = {"code": "execution_witness_verification",
                                   "message": f"{type(exc).__name__}: {str(exc)[:800]}"}
                raise _BoundaryStop("failed", "required_execution_verification_failed")

        def continuation_after_missing_witness():
            if state.get("execution_continuations", 0) >= 1:
                return False
            extra = auxiliary.usage(run_id)
            if (state.get("steps", 0) + extra["requests"] >= budget["max_steps"]
                    or exceeded()):
                state["execution_continuation_blocked_by_budget"] = True
                return False
            if usage.get("unknown_requests", 0) > 0:
                state["execution_continuation_blocked_by_unknown_usage"] = True
                return False
            state["execution_continuations"] = 1
            state["execution_witness_continuation"] = {
                "reason": "no_verified_bound_execution_witness",
                "procedural_only": True,
            }
            for key in ("reply", "snapshot", "results", "reply_arguments_unchanged"):
                state.pop(key, None)
            state["phase"] = "ready"
            save(status="running", event="runtime.execution_witness_required")
            return True

        def continuation_after_length(clean):
            """One retry after an output-limit cut; the truncated reply and any partial tool calls are discarded."""
            if state.get("output_length_continuations", 0) >= 1 or reference_enabled():
                return False
            extra = auxiliary.usage(run_id)
            if (state.get("steps", 0) + extra["requests"] >= budget["max_steps"] or exceeded()
                    or usage.get("unknown_requests", 0) > 0):
                return False
            state["output_length_continuations"] = 1
            state["output_length_continuation"] = {"truncated_content_chars": len(clean.get("content") or ""),
                                                   "discarded_tool_calls": len(clean.get("tool_calls") or [])}
            for key in ("reply", "snapshot", "results", "reply_arguments_unchanged"):
                state.pop(key, None)
            state["phase"] = "ready"
            save(status="running", event="runtime.output_length_continuation")
            return True

        def json_delivery(content, witness=None):
            return {"kind": "json_response", "content": content,
                "assurance": "JSON Schema structure accepted; any format repair preserved each typed scalar at its original object-key or array-index path. Scientific correctness was not validated.",
                **({"execution_witness": witness} if witness is not None else {}),
                **({"public_code_validation": state["public_code_validation"]}
                   if "public_code_validation" in current_task()["acceptance"] else {})}

        def public_code_gate(content, *, after_check=False):
            """Return True to deliver, False after a durable phase transition."""
            contract = current_task()["acceptance"].get("public_code_validation")
            if contract is None:
                return True
            execution_guard()
            state["rejected_final_response"] = content
            if contract["enforce"] and usage.get("unknown_requests", 0):
                raise _BoundaryStop("failed", "unknown_usage")
            try:
                generated_code = json.loads(content)["generated_code"]
                checked = verify_public_code_validation(self.store, run_id, generated_code)
            except Exception as exc:
                if not contract["enforce"]:
                    state["public_code_validation"] = {"status": "unavailable",
                        "verification_error": f"{type(exc).__name__}: {str(exc)[:800]}",
                        "official_correctness": "not_evaluated", "checks": []}
                    state.pop("rejected_final_response", None)
                    return True
                state["error"] = {"code": "public_code_verification",
                                  "message": f"{type(exc).__name__}: {str(exc)[:800]}"}
                raise _BoundaryStop("failed", "public_code_verification_failed") from exc
            state["public_code_validation"] = checked
            if after_check and checked["status"] == "missing":
                raise _BoundaryStop("failed", "public_code_check_receipt_missing")
            if not contract["enforce"] or checked["status"] == "passed":
                state.pop("rejected_final_response", None)
                return True
            if checked["status"] not in {"missing", "failed"}:
                raise _BoundaryStop("failed", "public_code_validation_unavailable")
            if checked["attempt_count"] >= contract["max_attempts"]:
                raise _BoundaryStop("failed", "public_code_validation_failed")
            if checked["status"] == "failed":
                if state.get("public_code_repairs", 0) >= 1:
                    raise _BoundaryStop("failed", "public_code_validation_failed")
                extra = auxiliary.usage(run_id)
                if state.get("steps", 0) + extra["requests"] >= budget["max_steps"] or exceeded():
                    raise _BoundaryStop("budget_exhausted", "budget_exhausted")
                state["public_code_repairs"] = 1
                state["public_code_feedback"] = {"previous_response": content, "validation": checked}
                for key in ("reply", "snapshot", "results", "reply_arguments_unchanged", "waiting",
                            "public_code_pending"):
                    state.pop(key, None)
                state["phase"] = "ready"
                save(status="running", event="runtime.public_code_feedback")
                return False
            # A host action uses the ordinary authorized executor and the same
            # durable journal. It is explicitly not a fabricated model tool call.
            snapshot = SessionToolset(self.store, self.catalog, sid).snapshot()
            call = ToolCall(id=new_id("publiccheck"), name="public_code_check",
                            arguments={"generated_code": generated_code})
            state["public_code_pending"] = {"call": call.to_dict(), "snapshot": snapshot,
                                            "final_content": content}
            state["phase"] = "public_code_check"
            save(status="running", event="runtime.public_code_check_requested")
            return False
        try:
            if initial["status"] in TERMINAL_STATUSES: return initial
            if state.get("phase") == "requesting":
                request = state["pending_request"]
                usage = self._usage(usage, {}, request)
                state["interrupted_request"] = request
                state.pop("pending_request", None)
                return finish("failed", "interrupted_model_request")
            while True:
                if self.store.get_run(run_id)["status"] in TERMINAL_STATUSES: return save()
                self.journal.renew(run_id, owner)
                if stop_event is not None and stop_event.is_set():
                    state["waiting"] = {"kind": "client_detached", "note": "Resume this run; no background model loop continues"}
                    return save(status="waiting", event="runtime.client_detached")
                if exceeded(): return finish("budget_exhausted", "budget_exhausted")
                if state.get("phase") in {"reply", "tools"} and task_changed(): return finish("failed", "task_changed")
                if state.get("phase") == "reply":
                    # This gate is durable and identical on fresh/recovered
                    # execution. A saved reply is never itself execution permission.
                    clean = state["reply"]
                    state.pop("output_length_continuation", None)
                    reference_guard()
                    if reference_enabled():
                        if usage.get("unknown_requests", 0):
                            return finish("failed", "unknown_usage")
                        if clean["tool_calls"] or clean["finish_reason"] == "tool_calls":
                            return finish("failed", "reference_model_protocol")
                    if (state.get("reply_arguments_unchanged") is not True
                            or self.store.redact(clean["tool_calls"]) != clean["tool_calls"]):
                        return finish("failed", "sensitive_tool_arguments_not_replayable")
                    if state.get("reply_tool_choice", "auto") == "required_or_refuse":
                        if usage.get("unknown_requests", 0) > 0:
                            return finish("failed", "unknown_usage")
                        refusal = clean.get("refusal")
                        calls = clean["tool_calls"]
                        exposed = state.get("snapshot", {}).get("definition_hashes", {})
                        if refusal:
                            if calls or clean["content"]:
                                state["error"] = {"code": "required_action_protocol",
                                    "message": "Required action refusal must be nonempty and exclusive"}
                                return finish("failed", "required_action_protocol")
                        elif refusal is not None and calls:
                            state["error"] = {"code": "required_action_protocol",
                                "message": "Required action reply mixed an empty refusal with tool calls"}
                            return finish("failed", "required_action_protocol")
                        if not refusal and not calls:
                            state["error"] = {"code": "required_action_missing",
                                "message": "Required action reply contained neither a tool call nor a refusal"}
                            return finish("failed", "required_action_missing")
                        if (refusal is None and (clean["finish_reason"] != "tool_calls"
                                or any(call.get("name") not in exposed for call in calls))):
                            state["error"] = {"code": "required_action_protocol",
                                "message": "Required action reply did not name only exposed tools"}
                            return finish("failed", "required_action_protocol")
                    if clean.get("refusal"): return finish("failed", "model_refusal")
                    if clean["finish_reason"] == "length" and continuation_after_length(clean):
                        continue
                    if clean["finish_reason"] not in {"stop", "tool_calls"}:
                        return finish("failed", "model_" + clean["finish_reason"])
                    if clean["tool_calls"]:
                        state["phase"] = "tools"
                        save(status="running", event="runtime.reply_eligible")
                        continue
                    if not clean["content"].strip(): return finish("failed", "empty_reply")
                    if reference_enabled():
                        feedback = state.get("reference_feedback")
                        if state.get("steps") != (2 if feedback else 1):
                            return finish("failed", "reference_revision_state_invalid")
                        if feedback is None:
                            extra = auxiliary.usage(run_id)
                            if state.get("steps", 0) + extra["requests"] >= budget["max_steps"] or exceeded():
                                return finish("budget_exhausted", "budget_exhausted")
                            # Intent is durable before the single local lookup. An
                            # interrupted lookup is never silently performed again.
                            state["phase"] = "reference_lookup"
                            save(status="running", event="runtime.reference_lookup_started")
                            execution_guard()
                            try:
                                state["reference_feedback"] = prepare_reference_feedback(self.store, run_id, clean["content"])
                            except Exception as exc:
                                state["reference_lookup_error"] = type(exc).__name__
                                return finish("failed", "reference_lookup_failed")
                            for key in ("reply", "snapshot", "results", "reply_arguments_unchanged"):
                                state.pop(key, None)
                            state["phase"] = "ready"
                            save(status="running", event="runtime.reference_feedback_prepared")
                            continue
                        if feedback.get("revision_state") != "replied" or state.get("steps") != 2:
                            return finish("failed", "reference_revision_state_invalid")
                        feedback["revision_state"] = "completed"
                    acceptance = current_task()["acceptance"]
                    response_schema = acceptance.get(RESPONSE_SCHEMA)
                    if response_schema is not None:
                        issues = response_schema_issues(clean["content"], response_schema)
                        if issues:
                            state["rejected_final_response"] = clean["content"]
                            state["response_schema_errors"] = issues
                            extra = auxiliary.usage(run_id)
                            repair_has_budget = (state.get("steps", 0) + extra["requests"] < budget["max_steps"]
                                and not exceeded()
                                and budget["max_input_tokens"] - usage.get("accounted_input_tokens", 0)
                                    - extra["accounted_input_tokens"] > 0
                                and budget["max_output_tokens"] - usage.get("accounted_output_tokens", 0)
                                    - extra["accounted_output_tokens"] > 0)
                            try:
                                has_seed, seed = response_repair_seed(clean["content"])
                                seed_fingerprint = response_scalar_fingerprint(seed) if has_seed else None
                            except Exception as exc:
                                state["response_schema_errors"].append(
                                    f"format_repair_unavailable: response value safety check failed ({type(exc).__name__})")
                                return finish("failed", "response_schema_rejected")
                            if state.get("response_format_repairs", 0) == 0 and repair_has_budget and has_seed:
                                state["response_format_repairs"] = 1
                                state["response_format_repair"] = {
                                    "rejected_response": clean["content"], "errors": issues}
                                state["response_repair_scalar_path_fingerprint"] = seed_fingerprint
                                for key in ("reply", "snapshot", "results", "reply_arguments_unchanged"):
                                    state.pop(key, None)
                                state["phase"] = "ready"
                                save(status="running", event="runtime.response_schema_rejected")
                                continue
                            if not has_seed and state.get("response_format_repairs", 0) == 0:
                                state["response_schema_errors"].append(
                                    "format_repair_unavailable: no unique strict JSON value was found")
                            return finish("failed", "response_schema_rejected")
                        if (state.get("response_format_repairs", 0)
                                and not state.get("response_format_repair_consumed")):
                            safety_error = None
                            try:
                                has_seed, repaired_value = response_repair_seed(clean["content"])
                                expected = state.get("response_repair_scalar_path_fingerprint")
                                unchanged = (has_seed and expected is not None
                                    and response_scalar_fingerprint(repaired_value) == expected)
                            except Exception as exc:
                                has_seed = False
                                unchanged = False
                                safety_error = f"repair_scalar_check_failed: {type(exc).__name__}"
                            if not unchanged:
                                state["rejected_final_response"] = clean["content"]
                                state["response_schema_errors"] = list(state.get("response_schema_errors", []))
                                state["response_schema_errors"].append(
                                    safety_error or "repair_changed_scalar_values")
                                return finish("failed", "response_schema_rejected")
                            # The scalar fingerprint belongs only to this
                            # repaired response. Keep it in state for audit,
                            # but never apply it to a later execution
                            # continuation or narrative.
                            state["response_format_repair_consumed"] = True
                        if not strict():
                            if not public_code_gate(clean["content"]):
                                continue
                            witness = None
                            if acceptance.get(REQUIRE_BOUND_EXECUTION_WITNESS) is True:
                                if "execution_binding" not in state:
                                    return finish("failed", "required_execution_binding_missing")
                                witness = required_witness()
                                if witness is None:
                                    if continuation_after_missing_witness():
                                        continue
                                    if state.pop("execution_continuation_blocked_by_budget", False):
                                        return finish("budget_exhausted", "budget_exhausted")
                                    if state.pop("execution_continuation_blocked_by_unknown_usage", False):
                                        return finish("failed", "unknown_usage")
                                    return finish("failed", "required_execution_missing_after_continuation")
                                state["execution_witness"] = witness
                            return finish("succeeded", "response_schema_valid", json_delivery(clean["content"], witness))
                    if strict():
                        delivery = self._accepted(sid, run_id)
                        if delivery: return finish("succeeded", "accepted_report", delivery)
                        state["unaccepted_model_text"] = clean["content"]
                        return finish("failed", "acceptance_not_met")
                    return finish("succeeded", "model_stop", {"kind": "conversation", "content": clean["content"],
                        "assurance": "Ordinary model response, not scientific validation"})
                if state.get("phase") == "public_code_check":
                    pending = state["public_code_pending"]
                    call = ToolCall(**pending["call"])
                    result = self._result(call, sid, run_id, pending["snapshot"], execution_guard)
                    if result.status in PENDING:
                        state["waiting"] = {"kind": "job" if result.job_id else "tool", "call_id": call.id,
                                            "job_id": result.job_id, "result": result.to_dict()}
                        return save(status="waiting", event="runtime.public_code_check_waiting")
                    state.pop("waiting", None)
                    if result.status != "ok":
                        state["public_code_check_error"] = result.to_dict()
                        return finish("failed", "public_code_validation_unavailable")
                    content = pending["final_content"]
                    # Check durable results independently; never trust the
                    # model's or tool-return prose as a final pass boolean.
                    if public_code_gate(content, after_check=True):
                        return finish("succeeded", "response_schema_valid", json_delivery(content))
                    continue
                if state.get("phase") == "tools":
                    if state.get("reply_arguments_unchanged") is not True:
                        return finish("failed", "invalid_runtime_checkpoint")
                    raw_reply, snapshot = state["reply"], state["snapshot"]
                    calls = [ToolCall(**value) for value in raw_reply["tool_calls"]]
                    results = state.setdefault("results", [])
                    if len(results) < len(calls):
                        call = calls[len(results)]
                        result = self._result(call, sid, run_id, snapshot, execution_guard)
                        if result.status in PENDING:
                            state["waiting"] = {"kind": "approval" if result.status == "approval_required" else
                                "job" if result.job_id else "user" if call.name == "ask_user" else "tool",
                                "call_id": call.id, "job_id": result.job_id, "result": result.to_dict()}
                            return save(status="waiting", event="runtime.waiting")
                        results.append({"call_id": call.id, "result": result.to_dict()})
                        if result.status == "uncertain":
                            state["uncertain_call_id"] = call.id
                            # Preserve the partial batch in the run checkpoint; it
                            # was never inserted as an orphan into shared messages.
                            return finish("failed", "uncertain_tool_effect")
                        save(status="running", event="runtime.tool_observed")
                        continue
                    group = [{"role": "assistant", "content": raw_reply["content"] or None,
                              "tool_calls": [{"id": call.id, "type": "function", "function": {
                                  "name": call.name, "arguments": canonical_json(call.arguments)}} for call in calls]}]
                    group.extend(self.context.tool_message(sid, run_id, item["call_id"], item["result"]) for item in results)
                    state.update(phase="ready")
                    for key in ("reply", "snapshot", "results", "waiting"): state.pop(key, None)
                    save(status="running", messages=group, event="runtime.batch_finished")
                    if strict():
                        delivery = self._accepted(sid, run_id)
                        if delivery: return finish("succeeded", "accepted_report", delivery)
                    continue

                if state.get("phase") != "ready": return finish("failed", "invalid_runtime_checkpoint")
                reference_guard()
                if reference_enabled() and state.get("steps") != (1 if state.get("reference_feedback") else 0):
                    return finish("failed", "reference_revision_state_invalid")
                if reference_enabled() and state.get("reference_feedback", {}).get("revision_state", "prepared") != "prepared":
                    return finish("failed", "reference_revision_state_invalid")
                acceptance = current_task()["acceptance"]
                if "public_code_validation" in acceptance and (
                        state.get("public_code_binding_error") or "public_code_validation_binding" not in state):
                    return finish("failed", "public_code_binding_invalid")
                if (acceptance.get(REQUIRE_BOUND_EXECUTION_WITNESS) is True
                        and (state.get("execution_binding_missing")
                             or "execution_binding" not in state)):
                    return finish("failed", "required_execution_binding_missing")
                witness = None
                if acceptance.get(REQUIRE_BOUND_EXECUTION_WITNESS) is True:
                    witness = required_witness()
                    if witness is not None:
                        state["execution_witness"] = witness
                        save(status="running", event="runtime.execution_witness_verified")
                    else:
                        # Only the current host verification controls the next
                        # request stage; never project a stale cached witness.
                        if "execution_witness" in state:
                            state.pop("execution_witness", None)
                            save(status="running", event="runtime.execution_witness_cleared")
                if acceptance.get(EXECUTION_BOUND_DELIVERY) is True and (
                        acceptance.get(REQUIRE_BOUND_EXECUTION_WITNESS) is not True or witness is not None):
                    execution_guard()
                    try:
                        execution_delivery = verify_execution_delivery(
                            self.store, run_id, current_task()["acceptance"][RESPONSE_SCHEMA])
                    except Exception as exc:
                        state["error"] = {"code": "execution_bound_verification",
                                           "message": f"{type(exc).__name__}: {str(exc)[:800]}"}
                        return finish("failed", "execution_bound_verification_failed")
                    if execution_delivery is not None:
                        if (not isinstance(execution_delivery, dict)
                                or execution_delivery.get("kind") != "execution_bound_json"
                                or not isinstance(execution_delivery.get("content"), str)
                                or not isinstance(execution_delivery.get("provenance"), dict)):
                            state["error"] = {"code": "execution_bound_verification",
                                               "message": "Verifier returned a malformed direct delivery"}
                            return finish("failed", "execution_bound_verification_failed")
                        provenance = execution_delivery["provenance"]
                        state["execution_bound_delivery_candidate"] = {
                            key: provenance.get(key) for key in (
                                "call_id", "run_id", "session_id", "analysis_id", "job_id",
                                "tool_name", "analysis_source_artifact_id", "analysis_source_sha256",
                                "result_artifact_id", "result_artifact_sha256", "job_result_sha256",
                                "summary_sha256", "response_schema_sha256")}
                        if witness is not None:
                            execution_delivery["execution_witness"] = witness
                            execution_delivery["assurance"] += " A bound execution witness was also verified; " \
                                "the computation's semantic correctness remains unvalidated."
                        save(status="running", event="runtime.execution_bound_delivery_candidate")
                        execution_guard()
                        save(status="running", event="runtime.execution_bound_delivery_verified")
                        execution_guard()
                        return finish("succeeded", "execution_bound_delivery", execution_delivery)
                action_required = acceptance.get(REQUIRE_EXECUTION_ACTION) is True and witness is None
                if (action_required
                        and not supports_tool_choice(self.model, "required_or_refuse")):
                    return finish("failed", "required_action_unsupported")
                extra = auxiliary.usage(run_id)
                remaining_input = budget["max_input_tokens"] - usage.get("accounted_input_tokens", 0) - extra["accounted_input_tokens"]
                remaining_output = budget["max_output_tokens"] - usage.get("accounted_output_tokens", 0) - extra["accounted_output_tokens"]
                if state.get("steps", 0) + extra["requests"] >= budget["max_steps"] or min(remaining_input, remaining_output) <= 0:
                    return finish("budget_exhausted", "budget_exhausted")
                reference_output_limit = None
                if reference_enabled() and state.get("reference_feedback"):
                    original_request = initial_projection(self.store, self.store.get_run(run_id),
                                                          state["reference_initial_request"])
                    reference_output_limit = original_request["max_output_tokens"]
                    if remaining_output < reference_output_limit:
                        return finish("budget_exhausted", "budget_exhausted")
                snapshot = SessionToolset(self.store, self.catalog, sid).snapshot()
                if (action_required and not set(snapshot["definition_hashes"])
                        & set(state["execution_binding"]["trusted_tools"])):
                    return finish("failed", "required_action_tool_unavailable")
                try: projection = self.context.build(sid, run_id, snapshot, max_input_tokens=remaining_input)
                except ContextOverflow as exc:
                    state["error"] = {"code": "context_budget", "message": str(exc)}
                    return finish("budget_exhausted", "context_budget")
                except ValueError:
                    if reference_enabled():
                        return finish("failed", "reference_binding_invalid")
                    raise
                if reference_enabled() and projection["tools"]:
                    return finish("failed", "reference_binding_invalid")
                request = {"estimated_input_tokens": projection["estimated_input_tokens"],
                    "task_binding": projection["task_binding"],
                    "max_output_tokens": reference_output_limit or min(
                        remaining_output, self.max_output_tokens * (2 if state.get("output_length_continuation") else 1)),
                    "timeout_seconds": min(self.max_request_seconds, budget["wall_time_seconds"] - elapsed()),
                    "model": self.store.redact(getattr(self.model, "metadata", {})),
                    "projection_hash": projection["projection_hash"], "started_at_unix": time.time()}
                if action_required:
                    request["tool_choice"] = "required_or_refuse"
                if request["timeout_seconds"] <= 0: return finish("budget_exhausted", "budget_exhausted")
                artifact = self.store.put_artifact(canonical_json({**projection, **request, "toolset": snapshot}),
                    "model-request.json", "application/json", {"kind": "public_model_request"}, session_id=sid, run_id=run_id)
                request["artifact_id"] = artifact["id"]
                if reference_enabled():
                    pointer = {"artifact_id": artifact["id"], "sha256": artifact["sha256"],
                               "projection_hash": projection["projection_hash"]}
                    if state.get("reference_feedback"):
                        state["reference_feedback"].update(revision_state="requesting",
                            revision_request=pointer, revision_projection_hash=projection["projection_hash"])
                    else:
                        state["reference_initial_request"] = pointer
                self.store.append_event(sid, "model.requested", request, run_id)
                state.update(phase="requesting", pending_request=request, steps=state.get("steps", 0) + 1,
                             request_task_binding=request["task_binding"])
                save(status="running")
                execution_guard()
                reference_guard()
                try:
                    completion = {"max_output_tokens": request["max_output_tokens"],
                                  "timeout_seconds": request["timeout_seconds"]}
                    if "tool_choice" in request:
                        completion["tool_choice"] = request["tool_choice"]
                    reply = self.model.complete(projection["messages"], projection["tools"], **completion)
                except Exception as exc:
                    usage = self._usage(usage, getattr(exc, "usage", {}), request)
                    state["error"] = self.store.redact({"code": getattr(exc, "code", "provider_error"),
                        "message": str(exc), "cost_unknown": getattr(exc, "cost_unknown", True)})
                    state.pop("pending_request", None)
                    if (isinstance(exc, ProviderFailure) and exc.code == "malformed_reply"
                            and not exc.cost_unknown
                            and {"input_tokens", "output_tokens"} <= exc.usage.keys()
                            and current_task()["acceptance"].get("report_layout") == "review"
                            and state.get("reply_format_repairs", 0) == 0):
                        # Projection failed before any tool dispatch. Permit one
                        # new model decision inside the SAME remaining budget;
                        # transport/unknown-cost failures still stop immediately.
                        state.update(phase="ready", reply_format_repairs=1,
                                     last_reply_format_error=state.pop("error"))
                        save(status="running", event="runtime.model_reply_repair", messages=[{
                            "role": "assistant", "content": "Host format-repair notice: the last model response "
                            "was rejected before any tool execution because its public fields or tool arguments "
                            "were malformed. Return valid JSON arguments with complete IDs from the current "
                            "reference index. Continue the current task; do not repeat completed research. "
                            "Only one format repair is allowed and its usage counts toward the same budget."}])
                        continue
                    return finish("failed", "provider_error")
                usage = self._usage(usage, getattr(reply, "usage", {}), request)
                try: public = self._valid_reply(reply, {a["call_id"] for a in self.store.list_actions(run_id)})
                except (ValueError, TypeError) as exc:
                    state["error"] = {"code": "model_protocol", "message": str(exc)}
                    state.pop("pending_request", None)
                    return finish("failed", "model_protocol")
                clean = self.store.redact(public)
                artifact = self.store.put_artifact(canonical_json(clean), "model-reply.json", "application/json",
                    {"kind": "public_model_reply"}, session_id=sid, run_id=run_id)
                self.store.append_event(sid, "model.replied", {"artifact_id": artifact["id"], "usage": clean["usage"],
                    "finish_reason": clean["finish_reason"], "request_artifact_id": request["artifact_id"]}, run_id)
                state.pop("pending_request", None)
                state.update(phase="reply", reply=clean, snapshot=snapshot, results=[],
                             reply_arguments_unchanged=clean["tool_calls"] == public["tool_calls"])
                if reference_enabled() and state.get("reference_feedback"):
                    state["reference_feedback"]["revision_state"] = "replied"
                    state["reference_feedback"]["presented"] = state["reference_feedback"]["presentation_planned"]
                if "tool_choice" in request:
                    state["reply_tool_choice"] = request["tool_choice"]
                else:
                    state.pop("reply_tool_choice", None)
                save(status="running", event="runtime.reply_checkpointed")
                continue
        except _ClientDetached:
            if state.get("phase") == "requesting":
                state["request_not_issued"] = state.pop("pending_request", None)
                state.update(phase="ready", steps=state["steps"] - 1)
                if state.get("reference_feedback"):
                    state["reference_feedback"]["revision_state"] = "prepared"
            state["waiting"] = {"kind": "client_detached", "note": "Resume this run; no background model loop continues"}
            return save(status="waiting", event="runtime.client_detached")
        except _BoundaryStop as stop:
            if state.get("phase") == "requesting":
                state["request_not_issued"] = state.pop("pending_request", None)
            return finish(stop.status, stop.reason)
        finally:
            stopped.set()
            thread.join(timeout=2)
            self.journal.release(run_id, owner)
