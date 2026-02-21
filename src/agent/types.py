"""
Shared types for agent-mode orchestration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A validated tool invocation emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Standardized result returned by a tool handler."""

    tool_name: str
    content: str
    success: bool = True
    error: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    """A tool exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    dangerous: bool = False
    source: str = "builtin"


@dataclass(frozen=True)
class SkillDoc:
    """A discovered skill-like instruction document."""

    name: str
    description: str
    body: str
    source: str
    path: str


@dataclass(frozen=True)
class MCPServerConfig:
    """A discovered MCP server definition."""

    name: str
    source: str
    transport: str  # "http" or "stdio"
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
