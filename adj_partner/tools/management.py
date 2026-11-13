"""Model capabilities can discover tools and enable only host-authorized ones."""

from ..contracts import ToolResult
from .catalog import ToolSpec
from .policy import SessionToolset


def register_management_tools(catalog, store):
    def search(context, query, limit=10):
        return ToolResult(data=catalog.search(query, limit))

    def enable(context, name, also=()):
        snapshot = SessionToolset(context.store, catalog, context.session_id).enable(
            name, also=also, expected_revision=context.toolset_revision)
        return ToolResult(data={"enabled": name, **({"also": list(also)} if also else {}),
                                "toolset_revision": snapshot["revision"],
                                "available_next_request": True})

    catalog.register(ToolSpec("catalog_search", "Search the shared catalog; discovery does not grant access", {
        "type": "object", "properties": {"query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["query"], "additionalProperties": False}), search)
    catalog.register(ToolSpec("tools_enable", "Enable authorized tools in ONE call. For a group, put additional names in also. Use them on the next request.", {
        "type": "object", "properties": {"name": {"type": "string"},
            "also": {"type": "array", "items": {"type": "string"}, "maxItems": 3, "uniqueItems": True}},
        "required": ["name"], "additionalProperties": False}), enable)
