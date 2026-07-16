"""
Agent-mode runner: local model + tools + MCP + skills.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..orchestrator import (
    AgentStatus,
    AIOrchestrator,
    APIResponse,
    StatusCallback,
    StatusStage,
)
from .config import AgentConfig, AgentLimits, build_agent_config
from .mcp import discover_mcp_servers
from .memory import ConversationStore, MemoryFileStore, truncate_text
from .parser import parse_tool_calls, strip_attribution
from .skills import discover_skills, render_skill_context, select_skills_for_prompt
from .tools import ToolRegistry, register_builtin_tools, register_mcp_tools
from .types import MCPServerConfig, SkillDoc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRunOptions:
    """Per-request overrides for agent execution."""

    session_id: str
    model_override: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.2
    enable_web_tools: bool | None = None
    enable_mcp: bool | None = None
    enable_skills: bool | None = None
    enable_browser_automation: bool | None = None
    incognito: bool = False
    system_prompt: str | None = None


class AgentRunner:
    """Codex-like tool-using agent loop over the existing AIOrchestrator."""

    def __init__(
        self, orchestrator: AIOrchestrator, workspace: Path | None = None
    ) -> None:
        self.orchestrator = orchestrator
        self.workspace = (workspace or Path.cwd()).resolve()
        user_config = getattr(orchestrator, "_user_config", {})
        if not isinstance(user_config, dict):
            user_config = {}
        self.config: AgentConfig = build_agent_config(user_config)
        self.memory_store = MemoryFileStore(self.config.memory_file)
        self.conversation_store = ConversationStore(self.config.conversation_dir)
        self._skills_cache: list[SkillDoc] | None = None
        self._mcp_server_cache: list[MCPServerConfig] | None = None

    async def _emit_status(
        self,
        callback: StatusCallback | None,
        stage: StatusStage,
        message: str,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return
        status = AgentStatus(
            stage=stage,
            message=message,
            model=model,
            metadata=metadata or {},
        )
        maybe_coro = callback(status)
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro

    def _should_auto_remember(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(
            pattern.lower() in lowered for pattern in self.config.auto_remember_patterns
        )

    def _format_transcript(
        self, transcript: list[dict[str, str]], max_chars: int
    ) -> str:
        lines: list[str] = []
        for item in transcript:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"[{role.upper()}]\n{content}")
        serialized = "\n\n".join(lines)
        if len(serialized) <= max_chars:
            return serialized
        return serialized[-max_chars:]

    def _build_agent_prompt(
        self,
        *,
        prompt: str,
        transcript: list[dict[str, str]],
        tools_manifest: list[dict[str, Any]],
        memory_context: str,
        skill_context: str,
        limits: AgentLimits,
        step: int,
        max_steps: int,
        browser_enabled: bool,
    ) -> str:
        tools_json = json.dumps(tools_manifest, ensure_ascii=False, indent=2)

        instruction_lines = [
            "You are a local coding agent with tool access.",
            "If you need tools, respond ONLY with JSON in this exact shape:",
            '{"tool_calls":[{"id":"call-1","name":"tool_name","arguments":{}}]}',
            "Do not wrap tool JSON in markdown fences.",
            "If you already have enough information, respond with a normal final answer.",
            "When the user asks to remember facts/preferences, call memory_write.",
            "Cite sources as plain URLs when using web tools.",
            f"Current loop step: {step}/{max_steps}.",
        ]
        if browser_enabled:
            instruction_lines.append(
                "Browser automation is enabled; use browser_action only when web interaction is explicitly required."
            )

        instruction_block = "\n".join(instruction_lines)

        memory_block = memory_context.strip() or "(no memory entries)"
        memory_block, _ = truncate_text(memory_block, limits.max_memory_context_chars)

        skill_block = skill_context.strip() or "(no selected skills)"
        skill_block, _ = truncate_text(
            skill_block, max(2000, limits.max_prompt_chars // 3)
        )

        static_prefix = (
            f"{instruction_block}\n\n"
            f"## Available Tools\n{tools_json}\n\n"
            f"## Long-Term Memory\n{memory_block}\n\n"
            f"## Active Skills\n{skill_block}\n\n"
            "## Conversation Transcript\n"
        )

        remaining = max(1000, limits.max_prompt_chars - len(static_prefix))
        transcript_serialized = self._format_transcript(transcript, remaining)
        final_prompt = f"{static_prefix}{transcript_serialized}\n\n## Current User Request\n{prompt}"
        if len(final_prompt) <= limits.max_prompt_chars:
            return final_prompt
        return final_prompt[: limits.max_prompt_chars]

    def _default_model(self, override: str | None) -> str:
        if override:
            return override
        default_model = self.config.default_model
        return default_model or "mlx-qwen3-coder-30b"

    async def _discover_skills_if_needed(self, use_skills: bool) -> list[SkillDoc]:
        if not use_skills:
            return []
        if self._skills_cache is None:
            self._skills_cache = discover_skills(
                codex_home=self.config.codex_home,
                claude_home=self.config.claude_home,
                gemini_home=self.config.gemini_home,
                sources=self.config.skill_sources,
            )
        return self._skills_cache

    async def _discover_mcp_servers_if_needed(
        self,
        use_mcp: bool,
    ) -> list[MCPServerConfig]:
        if not use_mcp:
            return []
        if self._mcp_server_cache is None:
            self._mcp_server_cache = discover_mcp_servers(
                codex_home=self.config.codex_home,
                claude_home=self.config.claude_home,
                gemini_home=self.config.gemini_home,
                sources=self.config.mcp_sources,
                workspace=self.workspace,
            )
        return self._mcp_server_cache

    async def run(
        self,
        prompt: str,
        *,
        options: AgentRunOptions,
        status_callback: StatusCallback | None = None,
    ) -> APIResponse:
        """
        Execute a complete agent loop for one prompt.
        """
        if not prompt.strip():
            return APIResponse(
                content="",
                model="",
                provider="agent",
                usage={},
                latency_ms=0,
                success=False,
                error="Prompt must be non-empty.",
            )

        limits = self.config.limits
        use_web_tools = (
            self.config.enable_web_tools
            if options.enable_web_tools is None
            else options.enable_web_tools
        )
        use_mcp = (
            self.config.enable_mcp if options.enable_mcp is None else options.enable_mcp
        )
        use_skills = (
            self.config.enable_skills
            if options.enable_skills is None
            else options.enable_skills
        )
        use_browser = (
            self.config.enable_browser_automation
            if options.enable_browser_automation is None
            else options.enable_browser_automation
        )

        await self._emit_status(
            status_callback,
            StatusStage.VALIDATING,
            "Preparing agent runtime...",
        )

        skills = await self._discover_skills_if_needed(use_skills)
        selected_skills = select_skills_for_prompt(prompt, skills, max_auto=4)
        skill_context = render_skill_context(
            selected_skills, max_chars=max(2000, limits.max_prompt_chars // 3)
        )

        memory_context = self.memory_store.read(limits.max_memory_context_chars)

        history = (
            []
            if options.incognito
            else self.conversation_store.load_recent(
                options.session_id,
                limits.max_history_messages,
            )
        )
        transcript: list[dict[str, str]] = [
            *history,
            {"role": "user", "content": prompt},
        ]

        registry = ToolRegistry()
        await register_builtin_tools(
            registry,
            config=self.config,
            limits=limits,
            memory_store=self.memory_store,
            skills=skills,
            enable_web_tools=use_web_tools,
            enable_browser_automation=use_browser,
        )

        mcp_warnings: list[str] = []
        if use_mcp:
            await self._emit_status(
                status_callback,
                StatusStage.ROUTING,
                "Discovering MCP servers...",
            )
            mcp_servers = await self._discover_mcp_servers_if_needed(use_mcp)
            mcp_warnings = await register_mcp_tools(
                registry,
                mcp_servers=mcp_servers,
                limits=limits,
            )

        model_key = self._default_model(options.model_override)
        model_start = datetime.now(timezone.utc)
        model_usage_input = 0
        model_usage_output = 0
        tool_log: list[dict[str, Any]] = []
        memory_write_called = False
        final_content = ""
        final_model_name = model_key

        original_incognito = self.orchestrator.incognito
        self.orchestrator.set_incognito(True)
        try:
            for step in range(1, limits.max_steps + 1):
                await self._emit_status(
                    status_callback,
                    StatusStage.GENERATING,
                    f"Agent reasoning step {step}/{limits.max_steps}...",
                    model=model_key,
                )

                agent_prompt = self._build_agent_prompt(
                    prompt=prompt,
                    transcript=transcript,
                    tools_manifest=registry.to_model_manifest(),
                    memory_context=memory_context,
                    skill_context=skill_context,
                    limits=limits,
                    step=step,
                    max_steps=limits.max_steps,
                    browser_enabled=use_browser,
                )

                response = await asyncio.wait_for(
                    self.orchestrator.query(
                        agent_prompt,
                        model_override=model_key,
                        max_tokens=options.max_tokens,
                        temperature=options.temperature,
                        system_prompt=options.system_prompt,
                        enable_web_search=False,
                    ),
                    timeout=limits.model_timeout_seconds,
                )

                model_usage_input += response.usage.get("input_tokens", 0)
                model_usage_output += response.usage.get("output_tokens", 0)
                if not response.success:
                    return APIResponse(
                        content="",
                        model=response.model or model_key,
                        provider="agent",
                        usage={
                            "input_tokens": model_usage_input,
                            "output_tokens": model_usage_output,
                        },
                        latency_ms=(
                            datetime.now(timezone.utc) - model_start
                        ).total_seconds()
                        * 1000,
                        success=False,
                        error=response.error or "Model request failed in agent loop.",
                        metadata={"mcp_warnings": mcp_warnings},
                    )

                final_model_name = response.model or model_key
                response_without_attr = strip_attribution(response.content)
                tool_calls = parse_tool_calls(response_without_attr)
                if not tool_calls:
                    final_content = response.content
                    break

                transcript.append(
                    {"role": "assistant", "content": response_without_attr}
                )

                for tool_call in tool_calls:
                    await self._emit_status(
                        status_callback,
                        StatusStage.CHAINING,
                        f"Running tool: {tool_call.name}",
                        model=model_key,
                    )
                    result = await registry.execute(
                        tool_call, limits.tool_timeout_seconds
                    )
                    if tool_call.name == "memory_write" and result.success:
                        memory_write_called = True

                    tool_log.append(
                        {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "success": result.success,
                            "error": result.error,
                            "truncated": result.truncated,
                        }
                    )
                    tool_result_payload = {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "success": result.success,
                        "error": result.error,
                        "content": result.content,
                    }
                    transcript.append(
                        {
                            "role": "tool",
                            "content": json.dumps(
                                tool_result_payload,
                                ensure_ascii=False,
                                indent=2,
                            ),
                        }
                    )

            if not final_content:
                final_content = (
                    "Reached max tool-iteration steps before producing a final answer. "
                    "Try narrowing the request or increasing agent.limits.maxSteps."
                )

            if self._should_auto_remember(prompt) and not memory_write_called:
                self.memory_store.remember(
                    prompt,
                    category="auto-remember",
                    source="heuristic",
                )

            if not options.incognito:
                self.conversation_store.append(options.session_id, "user", prompt)
                self.conversation_store.append(
                    options.session_id,
                    "assistant",
                    strip_attribution(final_content),
                )

            if mcp_warnings:
                warnings_block = "\n".join(f"- {item}" for item in mcp_warnings)
                final_content = (
                    f"{final_content}\n\n---\n*[MCP Warnings]*\n{warnings_block}"
                )

            latency_ms = (
                datetime.now(timezone.utc) - model_start
            ).total_seconds() * 1000
            return APIResponse(
                content=final_content,
                model=final_model_name,
                provider="agent",
                usage={
                    "input_tokens": model_usage_input,
                    "output_tokens": model_usage_output,
                },
                latency_ms=latency_ms,
                success=True,
                metadata={
                    "tool_calls": tool_log,
                    "session_id": options.session_id,
                    "memory_file": str(self.memory_store.memory_file),
                    "browser_automation": use_browser,
                    "mcp_enabled": use_mcp,
                    "skills_enabled": use_skills,
                },
            )
        except asyncio.TimeoutError:
            return APIResponse(
                content="",
                model=model_key,
                provider="agent",
                usage={
                    "input_tokens": model_usage_input,
                    "output_tokens": model_usage_output,
                },
                latency_ms=(datetime.now(timezone.utc) - model_start).total_seconds()
                * 1000,
                success=False,
                error=(
                    "Agent model step timed out. Increase "
                    "agent.limits.modelTimeoutSeconds for long-running tasks."
                ),
            )
        finally:
            self.orchestrator.set_incognito(original_incognito)
