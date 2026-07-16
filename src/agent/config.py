"""
Agent-mode configuration and limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..credentials import CONFIG_DIR


def _expand_path(value: str, default: Path) -> Path:
    stripped = value.strip()
    if not stripped:
        return default
    return Path(os.path.expandvars(os.path.expanduser(stripped))).resolve()


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_int(value: Any, default: int, *, minimum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(minimum, value)
    return default


def _as_float(value: Any, default: float, *, minimum: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(minimum, float(value))
    return default


def _as_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        if cleaned:
            return cleaned
    return list(default)


@dataclass(frozen=True)
class AgentLimits:
    """Execution policy for model/tool loops."""

    profile: str
    max_steps: int
    model_timeout_seconds: float
    tool_timeout_seconds: float
    shell_timeout_seconds: float
    web_search_timeout_seconds: float
    web_fetch_timeout_seconds: float
    max_prompt_chars: int
    max_tool_output_chars: int
    max_shell_output_chars: int
    max_fetched_chars: int
    max_web_results: int
    max_memory_context_chars: int
    max_history_messages: int


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent-mode configuration."""

    enabled_by_default: bool
    default_model: str
    default_session_id: str
    enable_web_tools: bool
    enable_mcp: bool
    enable_skills: bool
    enable_browser_automation: bool
    codex_home: Path
    claude_home: Path
    gemini_home: Path
    memory_file: Path
    conversation_dir: Path
    searxng_base_url: str
    searxng_language: str
    user_agent: str
    skill_sources: tuple[str, ...]
    mcp_sources: tuple[str, ...]
    auto_remember_patterns: tuple[str, ...]
    limits: AgentLimits


_DEFAULT_LIMITS_BY_PROFILE: dict[str, dict[str, Any]] = {
    "fast": {
        "maxSteps": 6,
        "modelTimeoutSeconds": 45,
        "toolTimeoutSeconds": 20,
        "shellTimeoutSeconds": 15,
        "webSearchTimeoutSeconds": 10,
        "webFetchTimeoutSeconds": 12,
        "maxPromptChars": 12000,
        "maxToolOutputChars": 5000,
        "maxShellOutputChars": 3000,
        "maxFetchedChars": 6000,
        "maxWebResults": 5,
        "maxMemoryContextChars": 4000,
        "maxHistoryMessages": 12,
    },
    "balanced": {
        "maxSteps": 10,
        "modelTimeoutSeconds": 90,
        "toolTimeoutSeconds": 40,
        "shellTimeoutSeconds": 30,
        "webSearchTimeoutSeconds": 20,
        "webFetchTimeoutSeconds": 25,
        "maxPromptChars": 24000,
        "maxToolOutputChars": 10000,
        "maxShellOutputChars": 8000,
        "maxFetchedChars": 12000,
        "maxWebResults": 8,
        "maxMemoryContextChars": 9000,
        "maxHistoryMessages": 24,
    },
    "deep": {
        "maxSteps": 16,
        "modelTimeoutSeconds": 180,
        "toolTimeoutSeconds": 90,
        "shellTimeoutSeconds": 60,
        "webSearchTimeoutSeconds": 35,
        "webFetchTimeoutSeconds": 45,
        "maxPromptChars": 48000,
        "maxToolOutputChars": 20000,
        "maxShellOutputChars": 16000,
        "maxFetchedChars": 24000,
        "maxWebResults": 12,
        "maxMemoryContextChars": 16000,
        "maxHistoryMessages": 48,
    },
}


def build_agent_config(user_config: dict[str, Any]) -> AgentConfig:
    """Build strongly-typed agent config from orchestrator user config."""
    agent_raw = user_config.get("agent", {})
    if not isinstance(agent_raw, dict):
        agent_raw = {}
    agent_config = cast(dict[str, Any], agent_raw)

    env_codex_home = os.environ.get("CODEX_HOME")
    codex_home_default = (
        Path(env_codex_home).expanduser() if env_codex_home else Path.home() / ".codex"
    )
    claude_home_default = Path.home() / ".claude"
    gemini_home_default = Path.home() / ".gemini"
    memory_default = CONFIG_DIR / "agent_memory.md"
    conversation_default = CONFIG_DIR / "agent_conversations"

    profile = _as_str(agent_config.get("profile"), "balanced").lower()
    if profile not in _DEFAULT_LIMITS_BY_PROFILE:
        profile = "balanced"

    profile_values = dict(_DEFAULT_LIMITS_BY_PROFILE[profile])
    custom_limits_raw = agent_config.get("limits", {})
    if isinstance(custom_limits_raw, dict):
        profile_values.update(custom_limits_raw)

    limits = AgentLimits(
        profile=profile,
        max_steps=_as_int(profile_values.get("maxSteps"), 10, minimum=1),
        model_timeout_seconds=_as_float(
            profile_values.get("modelTimeoutSeconds"), 90.0, minimum=1.0
        ),
        tool_timeout_seconds=_as_float(
            profile_values.get("toolTimeoutSeconds"), 40.0, minimum=1.0
        ),
        shell_timeout_seconds=_as_float(
            profile_values.get("shellTimeoutSeconds"), 30.0, minimum=1.0
        ),
        web_search_timeout_seconds=_as_float(
            profile_values.get("webSearchTimeoutSeconds"), 20.0, minimum=1.0
        ),
        web_fetch_timeout_seconds=_as_float(
            profile_values.get("webFetchTimeoutSeconds"), 25.0, minimum=1.0
        ),
        max_prompt_chars=_as_int(
            profile_values.get("maxPromptChars"), 24000, minimum=2000
        ),
        max_tool_output_chars=_as_int(
            profile_values.get("maxToolOutputChars"), 10000, minimum=500
        ),
        max_shell_output_chars=_as_int(
            profile_values.get("maxShellOutputChars"), 8000, minimum=200
        ),
        max_fetched_chars=_as_int(
            profile_values.get("maxFetchedChars"), 12000, minimum=500
        ),
        max_web_results=_as_int(profile_values.get("maxWebResults"), 8, minimum=1),
        max_memory_context_chars=_as_int(
            profile_values.get("maxMemoryContextChars"), 9000, minimum=500
        ),
        max_history_messages=_as_int(
            profile_values.get("maxHistoryMessages"), 24, minimum=1
        ),
    )

    memory_raw = agent_config.get("memory", {})
    if not isinstance(memory_raw, dict):
        memory_raw = {}
    memory_cfg = cast(dict[str, Any], memory_raw)

    web_raw = agent_config.get("web", {})
    if not isinstance(web_raw, dict):
        web_raw = {}
    web_cfg = cast(dict[str, Any], web_raw)

    compatibility_raw = agent_config.get("compatibility", {})
    if not isinstance(compatibility_raw, dict):
        compatibility_raw = {}
    compatibility_cfg = cast(dict[str, Any], compatibility_raw)

    return AgentConfig(
        enabled_by_default=_as_bool(agent_config.get("enabledByDefault"), False),
        default_model=_as_str(agent_config.get("defaultModel"), "mlx-qwen3-coder-30b"),
        default_session_id=_as_str(agent_config.get("defaultSessionId"), "default"),
        enable_web_tools=_as_bool(agent_config.get("enableWebTools"), True),
        enable_mcp=_as_bool(agent_config.get("enableMcp"), True),
        enable_skills=_as_bool(agent_config.get("enableSkills"), True),
        enable_browser_automation=_as_bool(
            agent_config.get("enableBrowserAutomation"), False
        ),
        codex_home=_expand_path(
            _as_str(agent_config.get("codexHome"), str(codex_home_default)),
            codex_home_default,
        ),
        claude_home=_expand_path(
            _as_str(agent_config.get("claudeHome"), str(claude_home_default)),
            claude_home_default,
        ),
        gemini_home=_expand_path(
            _as_str(agent_config.get("geminiHome"), str(gemini_home_default)),
            gemini_home_default,
        ),
        memory_file=_expand_path(
            _as_str(memory_cfg.get("filePath"), str(memory_default)),
            memory_default,
        ),
        conversation_dir=_expand_path(
            _as_str(memory_cfg.get("conversationDir"), str(conversation_default)),
            conversation_default,
        ),
        searxng_base_url=_as_str(
            web_cfg.get("searxngBaseUrl"), "http://127.0.0.1:8080"
        ).rstrip("/"),
        searxng_language=_as_str(web_cfg.get("language"), "en-US"),
        user_agent=_as_str(
            web_cfg.get("userAgent"),
            "ai-orchestrator-agent/1.0 (+local-model)",
        ),
        skill_sources=tuple(
            _as_str_list(
                compatibility_cfg.get("skillSources"),
                ["codex", "claude", "gemini"],
            )
        ),
        mcp_sources=tuple(
            _as_str_list(
                compatibility_cfg.get("mcpSources"),
                ["codex", "claude", "gemini"],
            )
        ),
        auto_remember_patterns=tuple(
            _as_str_list(
                memory_cfg.get("autoRememberPatterns"),
                [
                    "remember this",
                    "don't forget",
                    "do not forget",
                    "save this",
                    "note this",
                    "keep this in memory",
                ],
            )
        ),
        limits=limits,
    )
