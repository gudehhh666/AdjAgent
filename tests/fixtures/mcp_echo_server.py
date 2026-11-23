"""An actual official-SDK stdio server used only by bounded integration tests."""

import argparse
import json
import os
from pathlib import Path

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
import mcp.types as types


parser = argparse.ArgumentParser()
parser.add_argument("--schema-file")
parser.add_argument("--mode", default="normal")
parser.add_argument("--started-file")
parser.add_argument("--invoked-file")
args = parser.parse_args()
server = Server("harness-fixture", version="1")


@server.list_tools()
async def list_tools():
    config = json.loads(Path(args.schema_file).read_text()) if args.schema_file else {}
    if config.get("protocol_error"):
        raise McpError(types.ErrorData(code=-32055, message="fixture protocol refusal", data={"reason": "fixture"}))
    name = "bad.remote.name" if args.mode == "unsafe" else "echo"
    tools = [types.Tool(name=name, description="Echo actual MCP protocol input", inputSchema={
        "type": "object", "properties": {"text": {"type": config.get("text_type", "string")}},
        "required": ["text"], "additionalProperties": False}, outputSchema={
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)),
        types.Tool(name="environment", description="Check inherited test environment", inputSchema={"type": "object"}),
        types.Tool(name="fail", description="Return a genuine MCP tool error", inputSchema={"type": "object"}),
        types.Tool(name="slow", description="Wait cooperatively for timeout/cancel tests", inputSchema={"type": "object"}),
        types.Tool(name="large", description="Large output exercises the explicit result ceiling", inputSchema={"type": "object"})]
    if args.mode == "duplicate":
        tools.append(tools[0])
    return tools


@server.call_tool()
async def call_tool(name, arguments):
    if args.invoked_file:
        Path(args.invoked_file).write_text(name)
    if name == "echo":
        return {"text": arguments["text"]}
    if name == "environment":
        return {"inherited_value": os.environ.get("HARNESS_PARENT_API_SECRET"),
                "configured_value": os.environ.get("HARNESS_EXPLICIT_VALUE")}
    if name == "fail":
        raise ValueError("fixture rejected the request")
    if name == "slow":
        if args.started_file:
            Path(args.started_file).write_text(str(os.getpid()))
        await anyio.sleep(30)
        return {"unexpected": True}
    if name == "large":
        return {"text": "x" * (1024 * 1024 + 1)}
    raise ValueError("Unknown fixture tool")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
