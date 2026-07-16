"""
Agent tool registry and built-in tool handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from .config import AgentConfig, AgentLimits
from .mcp import call_server_tool, list_server_tools
from .memory import MemoryFileStore, truncate_text
from .skills import find_skill_by_name, serialize_skill_catalog
from .types import MCPServerConfig, SkillDoc, ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool metadata and execution handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def to_model_manifest(self) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for tool in sorted(self._tools.values(), key=lambda item: item.name):
            manifest.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "dangerous": tool.dangerous,
                    "source": tool.source,
                }
            )
        return manifest

    async def execute(self, tool_call: ToolCall, timeout_seconds: float) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_name=tool_call.name,
                content="",
                success=False,
                error=f"Unknown tool: {tool_call.name}",
            )

        try:
            result = await asyncio.wait_for(
                tool.handler(tool_call.arguments),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_call.name,
                content="",
                success=False,
                error=f"Tool timed out after {timeout_seconds:.1f}s.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=tool_call.name,
                content="",
                success=False,
                error=str(exc),
            )
        return result


def _safe_path(raw_path: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(raw_path))
    return Path(expanded).resolve()


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _extract_text_from_html(html: str) -> str:
    # Keep extraction intentionally simple and dependency-light.
    text = html
    for pattern in (
        r"(?is)<script.*?>.*?</script>",
        r"(?is)<style.*?>.*?</style>",
        r"(?is)<noscript.*?>.*?</noscript>",
    ):
        text = re.sub(pattern, " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return min(maximum, max(minimum, value))
    return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    return default


async def register_builtin_tools(
    registry: ToolRegistry,
    *,
    config: AgentConfig,
    limits: AgentLimits,
    memory_store: MemoryFileStore,
    skills: list[SkillDoc],
    enable_web_tools: bool,
    enable_browser_automation: bool,
) -> None:
    """Register built-in tools according to runtime feature toggles."""

    async def fs_read_tool(arguments: dict[str, Any]) -> ToolResult:
        path_raw = _coerce_str(arguments.get("path"))
        max_chars = _coerce_int(
            arguments.get("max_chars"),
            limits.max_tool_output_chars,
            1,
            max(limits.max_tool_output_chars, 1),
        )
        if not path_raw:
            return ToolResult("fs_read", "", False, "Missing required argument: path")
        path = _safe_path(path_raw)
        if not path.exists():
            return ToolResult("fs_read", "", False, f"Path does not exist: {path}")
        if not path.is_file():
            return ToolResult("fs_read", "", False, f"Path is not a file: {path}")
        content = path.read_text(encoding="utf-8")
        truncated_content, truncated = truncate_text(content, max_chars)
        payload = {
            "path": str(path),
            "truncated": truncated,
            "content": truncated_content,
        }
        return ToolResult(
            "fs_read",
            json.dumps(payload, ensure_ascii=False, indent=2),
            truncated=truncated,
        )

    async def fs_write_tool(arguments: dict[str, Any]) -> ToolResult:
        path_raw = _coerce_str(arguments.get("path"))
        content = _coerce_str(arguments.get("content"))
        append = _coerce_bool(arguments.get("append"), False)
        if not path_raw:
            return ToolResult("fs_write", "", False, "Missing required argument: path")
        path = _safe_path(path_raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return ToolResult(
            "fs_write",
            json.dumps(
                {
                    "path": str(path),
                    "bytes_written": len(content.encode("utf-8")),
                    "append": append,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    async def fs_list_tool(arguments: dict[str, Any]) -> ToolResult:
        path_raw = _coerce_str(arguments.get("path"), ".")
        limit = _coerce_int(arguments.get("limit"), 100, 1, 5000)
        path = _safe_path(path_raw)
        if not path.exists():
            return ToolResult("fs_list", "", False, f"Path does not exist: {path}")
        if not path.is_dir():
            return ToolResult("fs_list", "", False, f"Path is not a directory: {path}")

        entries: list[dict[str, Any]] = []
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower())[:limit]:
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return ToolResult(
            "fs_list",
            json.dumps(
                {"path": str(path), "entries": entries}, ensure_ascii=False, indent=2
            ),
        )

    async def shell_tool(arguments: dict[str, Any]) -> ToolResult:
        command = _coerce_str(arguments.get("command"))
        if not command:
            return ToolResult("shell", "", False, "Missing required argument: command")
        timeout_s = min(
            float(
                _coerce_int(
                    arguments.get("timeout"), int(limits.shell_timeout_seconds), 1, 600
                )
            ),
            limits.shell_timeout_seconds,
        )

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                "shell",
                "",
                False,
                f"Command timed out after {timeout_s:.1f}s",
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        output_payload = {
            "return_code": process.returncode,
            "stdout": truncate_text(stdout_text, limits.max_shell_output_chars)[0],
            "stderr": truncate_text(stderr_text, limits.max_shell_output_chars)[0],
        }
        return ToolResult(
            "shell",
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            success=process.returncode == 0,
            truncated=(
                len(stdout_text) > limits.max_shell_output_chars
                or len(stderr_text) > limits.max_shell_output_chars
            ),
        )

    async def memory_read_tool(arguments: dict[str, Any]) -> ToolResult:
        max_chars = _coerce_int(
            arguments.get("max_chars"),
            limits.max_memory_context_chars,
            1,
            max(limits.max_memory_context_chars, 1),
        )
        content = memory_store.read(max_chars)
        return ToolResult(
            "memory_read",
            json.dumps(
                {
                    "path": str(memory_store.memory_file),
                    "content": content,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    async def memory_write_tool(arguments: dict[str, Any]) -> ToolResult:
        note = _coerce_str(arguments.get("note"))
        category = _coerce_str(arguments.get("category"), "general")
        source = _coerce_str(arguments.get("source"), "model")
        if not note:
            return ToolResult(
                "memory_write",
                "",
                False,
                "Missing required argument: note",
            )
        memory_store.remember(note, category=category, source=source)
        return ToolResult(
            "memory_write",
            json.dumps(
                {
                    "path": str(memory_store.memory_file),
                    "saved": True,
                    "category": category,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    async def memory_search_tool(arguments: dict[str, Any]) -> ToolResult:
        query = _coerce_str(arguments.get("query"))
        limit = _coerce_int(arguments.get("limit"), 5, 1, 25)
        if not query:
            return ToolResult(
                "memory_search", "", False, "Missing required argument: query"
            )
        matches = memory_store.search(query, limit=limit)
        return ToolResult(
            "memory_search",
            json.dumps(
                {"query": query, "matches": matches}, ensure_ascii=False, indent=2
            ),
        )

    async def skills_list_tool(arguments: dict[str, Any]) -> ToolResult:
        _ = arguments
        catalog = serialize_skill_catalog(skills)
        return ToolResult(
            "skills_list",
            json.dumps({"skills": catalog}, ensure_ascii=False, indent=2),
        )

    async def skills_open_tool(arguments: dict[str, Any]) -> ToolResult:
        name = _coerce_str(arguments.get("name"))
        max_chars = _coerce_int(
            arguments.get("max_chars"),
            limits.max_tool_output_chars,
            200,
            max(limits.max_tool_output_chars, 200),
        )
        if not name:
            return ToolResult(
                "skills_open", "", False, "Missing required argument: name"
            )
        skill = find_skill_by_name(skills, name)
        if not skill:
            return ToolResult("skills_open", "", False, f"Skill not found: {name}")
        body, truncated = truncate_text(skill.body, max_chars)
        return ToolResult(
            "skills_open",
            json.dumps(
                {
                    "name": skill.name,
                    "source": skill.source,
                    "path": skill.path,
                    "description": skill.description,
                    "body": body,
                    "truncated": truncated,
                },
                ensure_ascii=False,
                indent=2,
            ),
            truncated=truncated,
        )

    registry.register(
        ToolSpec(
            name="fs_read",
            description="Read a UTF-8 text file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=fs_read_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="fs_write",
            description="Write UTF-8 text to a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
            handler=fs_write_tool,
            dangerous=True,
        )
    )
    registry.register(
        ToolSpec(
            name="fs_list",
            description="List directory entries.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            handler=fs_list_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="shell",
            description="Execute a shell command on the local machine.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
            handler=shell_tool,
            dangerous=True,
        )
    )
    registry.register(
        ToolSpec(
            name="memory_read",
            description="Read long-term memory notes.",
            input_schema={
                "type": "object",
                "properties": {"max_chars": {"type": "integer"}},
            },
            handler=memory_read_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="memory_write",
            description=(
                "Persist important facts/preferences to memory. "
                "Use when user asks to remember something."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "category": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["note"],
            },
            handler=memory_write_tool,
            dangerous=True,
        )
    )
    registry.register(
        ToolSpec(
            name="memory_search",
            description="Search memory notes by keyword.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=memory_search_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="skills_list",
            description="List available skills imported from Codex/Claude/Gemini.",
            input_schema={"type": "object", "properties": {}},
            handler=skills_list_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="skills_open",
            description="Read the body of a discovered skill by name.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["name"],
            },
            handler=skills_open_tool,
        )
    )

    if enable_web_tools:

        async def web_search_tool(arguments: dict[str, Any]) -> ToolResult:
            query = _coerce_str(arguments.get("query"))
            if not query:
                return ToolResult(
                    "web_search", "", False, "Missing required argument: query"
                )

            num_results = _coerce_int(
                arguments.get("num_results"),
                min(config.limits.max_web_results, 8),
                1,
                limits.max_web_results,
            )
            endpoint = f"{config.searxng_base_url}/search"
            params = {
                "q": query,
                "format": "json",
                "language": config.searxng_language,
            }
            timeout = httpx.Timeout(limits.web_search_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    endpoint,
                    params=params,
                    headers={"user-agent": config.user_agent},
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return ToolResult(
                    "web_search", "", False, "Unexpected search response format."
                )

            raw_results = payload.get("results", [])
            if not isinstance(raw_results, list):
                raw_results = []

            results: list[dict[str, str]] = []
            for raw_item in raw_results[:num_results]:
                if not isinstance(raw_item, dict):
                    continue
                title = _coerce_str(raw_item.get("title"))
                url = _coerce_str(raw_item.get("url"))
                snippet = _coerce_str(raw_item.get("content"))
                if not url:
                    continue
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )

            content = json.dumps(
                {
                    "query": query,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            trimmed, truncated = truncate_text(content, limits.max_tool_output_chars)
            return ToolResult(
                "web_search",
                trimmed,
                truncated=truncated,
                metadata={"results_count": len(results)},
            )

        async def web_fetch_tool(arguments: dict[str, Any]) -> ToolResult:
            url = _coerce_str(arguments.get("url"))
            if not url:
                return ToolResult(
                    "web_fetch", "", False, "Missing required argument: url"
                )
            if not _is_http_url(url):
                return ToolResult("web_fetch", "", False, f"Invalid URL: {url}")

            timeout = httpx.Timeout(limits.web_fetch_timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                response = await client.get(
                    url, headers={"user-agent": config.user_agent}
                )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            body = response.text
            if "html" in content_type.lower():
                extracted = _extract_text_from_html(body)
            else:
                extracted = body

            extracted, truncated = truncate_text(extracted, limits.max_fetched_chars)
            payload = {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "content": extracted,
                "truncated": truncated,
            }
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            serialized, payload_truncated = truncate_text(
                serialized, limits.max_tool_output_chars
            )
            return ToolResult(
                "web_fetch",
                serialized,
                truncated=truncated or payload_truncated,
            )

        registry.register(
            ToolSpec(
                name="web_search",
                description="Search the web via SearxNG and return ranked results.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                handler=web_search_tool,
            )
        )
        registry.register(
            ToolSpec(
                name="web_fetch",
                description="Fetch and extract content from a web page URL.",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
                handler=web_fetch_tool,
            )
        )

    if enable_browser_automation:
        playwright_script = (
            config.codex_home
            / "skills"
            / "playwright"
            / "scripts"
            / "playwright_cli.sh"
        )
        playwright_command = (
            str(playwright_script) if playwright_script.exists() else "playwright-cli"
        )

        async def browser_tool(arguments: dict[str, Any]) -> ToolResult:
            action = _coerce_str(arguments.get("action"))
            if not action:
                return ToolResult(
                    "browser_action", "", False, "Missing required argument: action"
                )
            args = arguments.get("args", [])
            if not isinstance(args, list):
                return ToolResult(
                    "browser_action",
                    "",
                    False,
                    "Field 'args' must be a list of strings.",
                )
            str_args: list[str] = [str(item) for item in args]

            process = await asyncio.create_subprocess_exec(
                playwright_command,
                action,
                *str_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=limits.shell_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    "browser_action", "", False, "Browser command timed out."
                )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            payload = {
                "command": [playwright_command, action, *str_args],
                "return_code": process.returncode,
                "stdout": truncate_text(stdout_text, limits.max_shell_output_chars)[0],
                "stderr": truncate_text(stderr_text, limits.max_shell_output_chars)[0],
            }
            return ToolResult(
                "browser_action",
                json.dumps(payload, ensure_ascii=False, indent=2),
                success=process.returncode == 0,
            )

        registry.register(
            ToolSpec(
                name="browser_action",
                description=(
                    "Run a Playwright CLI action to automate browser tasks. "
                    "Use only when browser automation is explicitly enabled."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["action"],
                },
                handler=browser_tool,
                dangerous=True,
            )
        )


async def register_mcp_tools(
    registry: ToolRegistry,
    *,
    mcp_servers: list[MCPServerConfig],
    limits: AgentLimits,
) -> list[str]:
    """
    Discover MCP tools from configured servers and register them.

    Returns a list of warnings for servers that could not be initialized.
    """
    warnings: list[str] = []
    for server in mcp_servers:
        try:
            tools = await list_server_tools(
                server, timeout_seconds=limits.tool_timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001
            warning = (
                f"Failed to list tools for MCP server '{server.name}' "
                f"({server.source}): {exc}"
            )
            logger.warning(warning)
            warnings.append(warning)
            continue

        for tool in tools:
            tool_name = tool.get("name")
            if not isinstance(tool_name, str):
                continue
            remote_tool = tool_name
            input_schema = tool.get("inputSchema", {})
            if not isinstance(input_schema, dict):
                input_schema = {}
            description = tool.get("description")
            if not isinstance(description, str):
                description = f"MCP tool {remote_tool}"

            local_name = (
                f"mcp__{_sanitize_segment(server.name)}__"
                f"{_sanitize_segment(remote_tool)}"
            )

            async def _handler(
                arguments: dict[str, Any],
                *,
                server_config: MCPServerConfig = server,
                remote_tool_name: str = remote_tool,
                local_tool_name: str = local_name,
            ) -> ToolResult:
                try:
                    content = await call_server_tool(
                        server_config,
                        remote_tool_name,
                        arguments,
                        timeout_seconds=limits.tool_timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(
                        tool_name=local_tool_name,
                        content="",
                        success=False,
                        error=str(exc),
                    )

                trimmed, truncated = truncate_text(
                    content, limits.max_tool_output_chars
                )
                return ToolResult(
                    tool_name=local_tool_name,
                    content=trimmed,
                    success=True,
                    truncated=truncated,
                    metadata={
                        "server": server_config.name,
                        "source": server_config.source,
                        "remote_tool": remote_tool_name,
                    },
                )

            registry.register(
                ToolSpec(
                    name=local_name,
                    description=description,
                    input_schema=cast(dict[str, Any], input_schema),
                    handler=_handler,
                    source=f"mcp:{server.source}:{server.name}",
                )
            )

    return warnings


def _sanitize_segment(value: str) -> str:
    clean = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return clean.strip("_-") or "tool"
