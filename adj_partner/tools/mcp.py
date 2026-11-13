"""Official-SDK MCP transport adapters configured only by the host.

Connections are opened per discovery/call and closed before returning. Each call
lists current definitions before invoking, rejecting contract drift. Stdio servers
are trusted local programs, not sandboxed code. SDK-safe default environment plus
explicit configuration is inherited; the full parent environment is never copied.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import math
import os
import re
from types import MappingProxyType
from urllib.parse import urlparse

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

from ..contracts import ToolResult
from .catalog import ToolSpec
from .executor import ToolFailure
from .policy import ToolCancelled


MAX_MCP_RESULT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = field(default=None, repr=False)
    env: dict | None = field(default=None, repr=False)
    headers: dict | None = field(default=None, repr=False)
    connection_timeout: float = 10
    call_timeout: float = 30
    network: bool = True
    requires_approval: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", self.name):
            raise ValueError("MCP server name must be a short model-compatible namespace")
        if self.transport not in {"stdio", "http"}:
            raise ValueError("MCP transport must be stdio or http (Streamable HTTP)")
        if self.transport == "stdio" and (not isinstance(self.command, str) or not self.command):
            raise ValueError("Stdio requires an explicit host command")
        if self.transport == "http":
            parsed = urlparse(self.url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("HTTP requires an http(s) URL without userinfo; configure auth in headers")
            object.__setattr__(self, "network", True)
        if isinstance(self.args, str) or any(not isinstance(value, str) for value in self.args):
            raise ValueError("MCP command arguments must be an explicit string sequence")
        object.__setattr__(self, "args", tuple(self.args))
        for name in ("env", "headers"):
            value = getattr(self, name) or {}
            if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
                raise ValueError(f"MCP {name} must contain host-configured strings")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        for value in (self.connection_timeout, self.call_timeout):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("MCP timeouts must be positive finite seconds")
        if type(self.network) is not bool or type(self.requires_approval) is not bool:
            raise ValueError("MCP risk metadata must be host-configured booleans")


def _leaves(exc):
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for child in exc.exceptions for leaf in _leaves(child)]
    return [exc]


class MCPProvider:
    def __init__(self, config):
        if not isinstance(config, MCPServerConfig):
            raise TypeError("MCPProvider requires host-owned MCPServerConfig")
        self.config = config

    @asynccontextmanager
    async def _transport(self):
        cfg = self.config
        if cfg.transport == "stdio":
            params = StdioServerParameters(command=cfg.command, args=list(cfg.args), env=dict(cfg.env))
            # Raw stderr is untrusted and may contain credentials. Protocol errors
            # remain structured; server stderr is not copied to user logs/traces.
            with open(os.devnull, "w") as errors:
                async with stdio_client(params, errlog=errors) as streams:
                    yield streams
        else:
            async with streamablehttp_client(cfg.url, headers=dict(cfg.headers),
                                            timeout=cfg.connection_timeout,
                                            sse_read_timeout=cfg.call_timeout) as streams:
                yield streams[:2]

    def _specs(self, tools):
        specs = {}
        try:
            for remote in tools:
                name = f"{self.config.name}__{remote.name}"
                spec = ToolSpec(name, remote.description or "", remote.inputSchema, remote.outputSchema,
                                network=self.config.network, requires_approval=self.config.requires_approval)
                if name in specs:
                    raise ValueError("Duplicate names in MCP discovery")
                specs[name] = (remote.name, spec)
        except (ValueError, TypeError) as exc:
            raise ToolFailure("mcp_schema", f"Unsafe MCP definition: {exc}") from exc
        return specs

    async def _list(self, session):
        all_tools, cursor, seen = [], None, set()
        # Bound dishonest/broken pagination in addition to the operation timeout.
        for _ in range(100):
            response = await session.list_tools(cursor=cursor)
            all_tools.extend(response.tools)
            cursor = response.nextCursor
            if cursor is None:
                return self._specs(all_tools)
            if cursor in seen:
                raise ToolFailure("mcp_protocol", "MCP discovery repeated a pagination cursor")
            seen.add(cursor)
        raise ToolFailure("mcp_protocol", "MCP discovery exceeded pagination limit")

    async def _operate(self, expected=None, arguments=None, cancel_event=None, check_cancelled=None):
        cfg = self.config
        outcome = None
        def ensure_active():
            # A session callback checks fresh persisted lifecycle state as well
            # as the process-local signal; standalone discovery has no callback.
            if check_cancelled is not None:
                check_cancelled()
            elif cancel_event is not None and cancel_event.is_set():
                raise ToolCancelled("MCP operation cancelled; remote effects may require reconciliation")
        ensure_active()
        async with anyio.create_task_group() as group:
            async def watch_cancel():
                while True:
                    ensure_active()
                    await anyio.sleep(0.05)
            if check_cancelled is not None or cancel_event is not None:
                group.start_soon(watch_cancel)
            try:
                # Outer deadline also bounds connection setup/teardown. No timeout
                # worker thread can outlive this call and continue executing tools.
                with anyio.fail_after(cfg.connection_timeout + 2 * cfg.call_timeout):
                    async with self._transport() as (read, write):
                        async with ClientSession(read, write, read_timeout_seconds=timedelta(
                                seconds=max(cfg.connection_timeout, cfg.call_timeout))) as session:
                            with anyio.fail_after(cfg.connection_timeout):
                                await session.initialize()
                            with anyio.fail_after(cfg.call_timeout):
                                specs = await self._list(session)
                            if expected is None:
                                outcome = [spec for _, spec in specs.values()]
                            else:
                                current = specs.get(expected.name)
                                if current is None or current[1].definition_hash != expected.definition_hash:
                                    raise ToolFailure("schema_drift", "MCP tool definition changed since registration", "denied")
                                # Discovery can yield for arbitrary I/O. Recheck
                                # persisted cancellation immediately before call.
                                ensure_active()
                                with anyio.fail_after(cfg.call_timeout):
                                    result = await session.call_tool(current[0], arguments,
                                                                     read_timeout_seconds=timedelta(seconds=cfg.call_timeout))
                                if len(result.model_dump_json().encode("utf-8")) > MAX_MCP_RESULT_BYTES:
                                    raise ToolFailure("output_too_large", "MCP result exceeds the 1 MiB serialized-result ceiling; request a bounded result or artifact")
                                content = [item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in result.content]
                                if result.isError:
                                    details = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
                                    outcome = ToolResult(status="error", data={"content": content}, error={
                                        "code": "mcp_tool_error", "message": details or "MCP tool reported an error"})
                                else:
                                    outcome = ToolResult(data=result.structuredContent if result.structuredContent is not None else {"content": content})
            finally:
                group.cancel_scope.cancel()
        ensure_active()
        return outcome

    def _run(self, expected=None, arguments=None, cancel_event=None, check_cancelled=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ToolFailure("mcp_async_context", "Run synchronous MCP adapters in the host worker thread")
        try:
            return anyio.run(self._operate, expected, arguments, cancel_event, check_cancelled)
        except Exception as exc:
            leaves = _leaves(exc)
            for leaf in leaves:
                if isinstance(leaf, (ToolFailure, ToolCancelled, PermissionError)):
                    raise leaf
            if any(isinstance(leaf, TimeoutError) or "timed out" in str(leaf).lower() for leaf in leaves):
                raise ToolFailure("mcp_timeout", "MCP operation timed out; a remote call may have an unknown outcome") from exc
            for leaf in leaves:
                if isinstance(leaf, McpError):
                    code = "mcp_connection" if leaf.error.code == CONNECTION_CLOSED else "mcp_protocol"
                    raise ToolFailure(code, leaf.error.message, details={
                        "provider_code": leaf.error.code, "provider_data": leaf.error.data}) from exc
            details = "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)
            raise ToolFailure("mcp_connection", f"MCP connection/protocol failed: {details}") from exc

    def discover(self):
        """Return real remote ToolSpecs. Host config only; does not enable tools."""
        return self._run()

    def register(self, catalog, store):
        """Register an atomic discovery batch, adding known secrets before any I/O.

        Secret registration affects future persistence/export, not old raw bytes.
        Values stay out of definitions; never pass this configuration as model args.
        """
        secret_values = [*self.config.env.values(), *self.config.headers.values()]
        for value in self.config.headers.values():
            if value.lower().startswith(("bearer ", "basic ")):
                secret_values.append(value.split(" ", 1)[1])
        store.register_secrets(secret_values)
        specs = self.discover()
        def handler(spec):
            def call(context, **arguments):
                context.check_cancelled()
                return self._run(spec, arguments, context.cancel_event, context.check_cancelled)
            return call
        catalog.register_many([(spec, handler(spec)) for spec in specs], secrets=secret_values)
        return [spec.name for spec in specs]
