"""
MCP discovery and lightweight client adapters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import httpx

from .types import MCPServerConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    return None


def _normalize_server_name(name: str) -> str:
    clean = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in name
    )
    return clean.strip("-_") or "server"


def _resolve_placeholders(value: str, base_path: Path) -> str:
    resolved = value.replace("${extensionPath}", str(base_path))
    resolved = os.path.expandvars(os.path.expanduser(resolved))
    return resolved


def _parse_mcp_server_entry(
    server_name: str,
    server_payload: dict[str, Any],
    *,
    source: str,
    base_path: Path | None = None,
) -> MCPServerConfig | None:
    normalized_name = _normalize_server_name(server_name)
    payload = dict(server_payload)

    server_type = payload.get("type")
    url = payload.get("url")
    http_url = payload.get("httpUrl")
    command = payload.get("command")
    args_raw = payload.get("args", [])
    env_raw = payload.get("env", {})
    cwd_raw = payload.get("cwd")

    if isinstance(server_type, str) and server_type.lower() == "http" and isinstance(
        url, str
    ):
        return MCPServerConfig(
            name=normalized_name,
            source=source,
            transport="http",
            url=url,
        )

    if isinstance(url, str):
        return MCPServerConfig(
            name=normalized_name,
            source=source,
            transport="http",
            url=url,
        )

    if isinstance(http_url, str):
        return MCPServerConfig(
            name=normalized_name,
            source=source,
            transport="http",
            url=http_url,
        )

    if isinstance(command, str):
        base = base_path or Path.cwd()
        args: list[str] = []
        if isinstance(args_raw, list):
            for item in args_raw:
                if isinstance(item, str):
                    args.append(_resolve_placeholders(item, base))

        env: dict[str, str] = {}
        if isinstance(env_raw, dict):
            for key, value in env_raw.items():
                if isinstance(key, str) and isinstance(value, str):
                    env[key] = _resolve_placeholders(value, base)

        cwd: str | None = None
        if isinstance(cwd_raw, str):
            cwd = _resolve_placeholders(cwd_raw, base)

        return MCPServerConfig(
            name=normalized_name,
            source=source,
            transport="stdio",
            command=_resolve_placeholders(command, base),
            args=tuple(args),
            env=env,
            cwd=cwd,
        )

    return None


def _collect_codex_mcp_servers(codex_home: Path) -> list[MCPServerConfig]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return []

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    if not isinstance(data, dict):
        return []
    servers_raw = data.get("mcp_servers")
    if not isinstance(servers_raw, dict):
        return []

    discovered: list[MCPServerConfig] = []
    for server_name, payload in servers_raw.items():
        if not isinstance(server_name, str) or not isinstance(payload, dict):
            continue
        parsed = _parse_mcp_server_entry(
            server_name,
            cast(dict[str, Any], payload),
            source="codex",
        )
        if parsed:
            discovered.append(parsed)
    return discovered


def _extract_mcp_server_maps(payload: Any) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "mcpServers" and isinstance(value, dict):
                maps.append(cast(dict[str, Any], value))
            maps.extend(_extract_mcp_server_maps(value))
    elif isinstance(payload, list):
        for item in payload:
            maps.extend(_extract_mcp_server_maps(item))
    return maps


def _collect_claude_mcp_servers(claude_home: Path, workspace: Path) -> list[MCPServerConfig]:
    discovered: list[MCPServerConfig] = []

    claude_global = _read_json(claude_home.parent / ".claude.json")
    if claude_global:
        maps = _extract_mcp_server_maps(claude_global)
        for server_map in maps:
            for server_name, server_payload in server_map.items():
                if not isinstance(server_name, str) or not isinstance(
                    server_payload, dict
                ):
                    continue
                parsed = _parse_mcp_server_entry(
                    server_name,
                    cast(dict[str, Any], server_payload),
                    source="claude",
                )
                if parsed:
                    discovered.append(parsed)

        # Also parse project-specific mcpServers if present.
        projects = claude_global.get("projects")
        if isinstance(projects, dict):
            workspace_str = str(workspace.resolve())
            for project_path, project_payload in projects.items():
                if not isinstance(project_path, str) or not isinstance(
                    project_payload, dict
                ):
                    continue
                if workspace_str.startswith(project_path):
                    mcp_servers = project_payload.get("mcpServers")
                    if isinstance(mcp_servers, dict):
                        for server_name, server_payload in mcp_servers.items():
                            if not isinstance(server_name, str) or not isinstance(
                                server_payload, dict
                            ):
                                continue
                            parsed = _parse_mcp_server_entry(
                                server_name,
                                cast(dict[str, Any], server_payload),
                                source="claude",
                            )
                            if parsed:
                                discovered.append(parsed)

    installed_plugins = _read_json(
        claude_home / "plugins" / "installed_plugins.json"
    )
    if installed_plugins and isinstance(installed_plugins.get("plugins"), dict):
        plugins = cast(dict[str, Any], installed_plugins["plugins"])
        for plugin_entries in plugins.values():
            if not isinstance(plugin_entries, list):
                continue
            for plugin_entry in plugin_entries:
                if not isinstance(plugin_entry, dict):
                    continue
                install_path_raw = plugin_entry.get("installPath")
                if not isinstance(install_path_raw, str):
                    continue
                install_path = Path(install_path_raw)
                mcp_json_path = install_path / ".mcp.json"
                mcp_json = _read_json(mcp_json_path)
                if mcp_json:
                    for server_name, server_payload in mcp_json.items():
                        if not isinstance(server_name, str) or not isinstance(
                            server_payload, dict
                        ):
                            continue
                        parsed = _parse_mcp_server_entry(
                            server_name,
                            cast(dict[str, Any], server_payload),
                            source="claude-plugin",
                            base_path=install_path,
                        )
                        if parsed:
                            discovered.append(parsed)

                gemini_extension_path = install_path / "gemini-extension.json"
                extension_manifest = _read_json(gemini_extension_path)
                if extension_manifest and isinstance(
                    extension_manifest.get("mcpServers"), dict
                ):
                    mcp_servers = cast(
                        dict[str, Any], extension_manifest["mcpServers"]
                    )
                    for server_name, server_payload in mcp_servers.items():
                        if not isinstance(server_name, str) or not isinstance(
                            server_payload, dict
                        ):
                            continue
                        parsed = _parse_mcp_server_entry(
                            server_name,
                            cast(dict[str, Any], server_payload),
                            source="claude-plugin",
                            base_path=install_path,
                        )
                        if parsed:
                            discovered.append(parsed)

    return discovered


def _collect_gemini_mcp_servers(gemini_home: Path) -> list[MCPServerConfig]:
    extensions_root = gemini_home / "extensions"
    if not extensions_root.exists():
        return []

    discovered: list[MCPServerConfig] = []
    for extension_dir in extensions_root.iterdir():
        if not extension_dir.is_dir():
            continue
        manifest = _read_json(extension_dir / "gemini-extension.json")
        if not manifest:
            continue
        mcp_servers = manifest.get("mcpServers")
        if not isinstance(mcp_servers, dict):
            continue
        for server_name, payload in mcp_servers.items():
            if not isinstance(server_name, str) or not isinstance(payload, dict):
                continue
            parsed = _parse_mcp_server_entry(
                server_name,
                cast(dict[str, Any], payload),
                source="gemini",
                base_path=extension_dir,
            )
            if parsed:
                discovered.append(parsed)
    return discovered


def discover_mcp_servers(
    *,
    codex_home: Path,
    claude_home: Path,
    gemini_home: Path,
    sources: tuple[str, ...],
    workspace: Path,
) -> list[MCPServerConfig]:
    """Discover MCP server definitions from selected sources."""
    normalized_sources = {item.lower().strip() for item in sources}
    discovered: list[MCPServerConfig] = []
    if "codex" in normalized_sources:
        discovered.extend(_collect_codex_mcp_servers(codex_home))
    if "claude" in normalized_sources:
        discovered.extend(_collect_claude_mcp_servers(claude_home, workspace))
    if "gemini" in normalized_sources:
        discovered.extend(_collect_gemini_mcp_servers(gemini_home))

    unique: dict[str, MCPServerConfig] = {}
    for server in discovered:
        key = f"{server.source}:{server.name}:{server.transport}:{server.url}:{server.command}:{server.args}"
        if key not in unique:
            unique[key] = server
    return list(unique.values())


class MCPHttpClient:
    """Minimal HTTP/SSE MCP client."""

    def __init__(self, server: MCPServerConfig, timeout_seconds: float) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self._request_id = 1
        self._session_id: str | None = None

    def _next_id(self) -> int:
        current = self._request_id
        self._request_id += 1
        return current

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse_sse_payload(response_text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in response_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            data_raw = stripped[len("data:") :].strip()
            if not data_raw:
                continue
            try:
                payload = json.loads(data_raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(cast(dict[str, Any], payload))
        return events

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.server.url:
            raise RuntimeError("MCP HTTP server missing URL.")

        request_id = self._next_id()
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.server.url,
                headers=self._headers(),
                json=payload,
            )
            if session_id := response.headers.get("mcp-session-id"):
                self._session_id = session_id
            response.raise_for_status()
            events = self._parse_sse_payload(response.text)
            for event in events:
                if event.get("id") == request_id:
                    return event
            if events:
                return events[-1]
            raise RuntimeError("MCP HTTP response did not include a JSON-RPC event.")


class MCPStdioClient:
    """Minimal stdio MCP client using Content-Length framing."""

    def __init__(self, server: MCPServerConfig, timeout_seconds: float) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self._request_id = 1

    def _next_id(self) -> int:
        current = self._request_id
        self._request_id += 1
        return current

    async def _write_message(
        self, writer: asyncio.StreamWriter, payload: dict[str, Any]
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        writer.write(header + body)
        await writer.drain()

    async def _read_message(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line:
                raise RuntimeError("MCP stdio server closed the stream.")
            if line in {b"\r\n", b"\n"}:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", maxsplit=1)
            headers[key.strip().lower()] = value.strip()

        content_length_raw = headers.get("content-length")
        if not content_length_raw:
            raise RuntimeError("Invalid MCP stdio frame: missing Content-Length.")
        content_length = int(content_length_raw)
        raw_body = await reader.readexactly(content_length)
        parsed = json.loads(raw_body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("Invalid MCP stdio frame: payload is not an object.")
        return cast(dict[str, Any], parsed)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.server.command:
            raise RuntimeError("MCP stdio server missing command.")

        env = os.environ.copy()
        env.update(self.server.env)

        process = await asyncio.create_subprocess_exec(
            self.server.command,
            *self.server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.server.cwd or None,
            env=env,
        )

        if process.stdin is None or process.stdout is None:
            process.kill()
            await process.wait()
            raise RuntimeError("Failed to open stdio pipes for MCP server.")

        try:
            initialize_id = self._next_id()
            await self._write_message(
                process.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": initialize_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "ai-orchestrator",
                            "version": "2.0.0",
                        },
                    },
                },
            )
            await asyncio.wait_for(
                self._wait_for_id(process.stdout, initialize_id),
                timeout=self.timeout_seconds,
            )
            await self._write_message(
                process.stdin,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            )

            request_id = self._next_id()
            request_payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                request_payload["params"] = params
            await self._write_message(process.stdin, request_payload)

            response = await asyncio.wait_for(
                self._wait_for_id(process.stdout, request_id),
                timeout=self.timeout_seconds,
            )
            return response
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    async def _wait_for_id(
        self,
        reader: asyncio.StreamReader,
        request_id: int,
    ) -> dict[str, Any]:
        while True:
            message = await self._read_message(reader)
            if message.get("id") == request_id:
                return message


def _extract_tool_text(result_payload: dict[str, Any]) -> str:
    content = result_payload.get("content")
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    lines.append(text_value)
            elif isinstance(item, str):
                lines.append(item)
        if lines:
            return "\n".join(lines).strip()

    if isinstance(content, str):
        return content

    structured = result_payload.get("structuredContent")
    if isinstance(structured, dict):
        return json.dumps(structured, ensure_ascii=False, indent=2)

    if isinstance(result_payload, dict):
        return json.dumps(result_payload, ensure_ascii=False, indent=2)
    return str(result_payload)


async def list_server_tools(
    server: MCPServerConfig,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """List tools for a discovered MCP server."""
    client: MCPHttpClient | MCPStdioClient
    if server.transport == "http":
        client = MCPHttpClient(server, timeout_seconds)
    else:
        client = MCPStdioClient(server, timeout_seconds)

    response = await client.request("tools/list", {})
    result = response.get("result")
    if not isinstance(result, dict):
        return []
    tools_raw = result.get("tools")
    if not isinstance(tools_raw, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in tools_raw:
        if isinstance(item, dict):
            tools.append(cast(dict[str, Any], item))
    return tools


async def call_server_tool(
    server: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
) -> str:
    """Execute a tool call against an MCP server."""
    client: MCPHttpClient | MCPStdioClient
    if server.transport == "http":
        client = MCPHttpClient(server, timeout_seconds)
    else:
        client = MCPStdioClient(server, timeout_seconds)

    response = await client.request(
        "tools/call",
        {"name": tool_name, "arguments": arguments},
    )
    if "error" in response:
        error_payload = response.get("error")
        return json.dumps({"error": error_payload}, ensure_ascii=False, indent=2)

    result = response.get("result")
    if isinstance(result, dict):
        return _extract_tool_text(cast(dict[str, Any], result))
    return json.dumps({"result": result}, ensure_ascii=False, indent=2)
