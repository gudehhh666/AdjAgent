"""Shared public-plan and actual-user clarification capabilities."""

from ..contracts import ToolResult
from ..tools import ToolSpec
from .journal import RunJournal


def register_runtime_tools(catalog):
    def ask(context, question):
        request = RunJournal(context.store).ask(context.run_id, context.call_id, question)
        return ToolResult(status="waiting", data={"request_id": request["id"], "question": request["question"], "kind": "user"})
    def plan(context, steps):
        return ToolResult(data=RunJournal(context.store).set_plan(context.run_id, steps))
    bindings = [
        (ToolSpec("ask_user", "Ask the actual user for missing information. Cannot authorize tools or fabricate a user answer.", {
            "type": "object", "properties": {"question": {"type": "string", "minLength": 1, "maxLength": 4000}},
            "required": ["question"], "additionalProperties": False}), ask),
        (ToolSpec("update_plan", "Update a short public working plan, not a mandatory workflow or private reasoning trace.", {
            "type": "object", "properties": {"steps": {"type": "array", "maxItems": 30, "items": {
                "type": "object", "properties": {"step": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "status": {"enum": ["pending", "in_progress", "completed"]}},
                "required": ["step", "status"], "additionalProperties": False}}},
            "required": ["steps"], "additionalProperties": False}), plan)]
    # Reopening the same host catalog must not duplicate registrations.
    for spec, handler in bindings:
        if spec.name not in catalog.names(): catalog.register(spec, handler)
