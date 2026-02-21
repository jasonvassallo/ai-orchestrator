#!/usr/bin/env python3
"""
AI Orchestrator Terminal UI
============================

A beautiful terminal-based interface using Textual.

Usage:
    python -m src.tui.app

Or after installation:
    ai-chat
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, ScrollableContainer
    from textual.screen import Screen
    from textual.timer import Timer
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        Markdown,
        Select,
        Static,
        Switch,
    )
except ImportError:
    print("Error: textual and rich are required for the TUI.")
    print("Install them with: pip install textual rich")
    sys.exit(1)


from ..agent.runner import AgentRunner, AgentRunOptions
from ..orchestrator import LOCAL_PROVIDERS, AgentStatus, ModelRegistry


def _build_model_options() -> list[tuple[str, str]]:
    options = [("Auto (Best for Task)", "auto")]
    models = sorted(
        ModelRegistry.MODELS.items(),
        key=lambda item: (item[1].provider, item[1].name),
    )
    for key, model in models:
        tags = []
        if model.provider in LOCAL_PROVIDERS:
            tags.append("Local")
        provider_tag = model.provider
        if model.provider == "vertex-ai" and "vertex" in model.name.lower():
            provider_tag = ""
        if provider_tag:
            tags.append(provider_tag)
        label = f"{model.name} ({', '.join(tags)})"
        options.append((label, key))
    return options


# Available models (dynamic from registry)
MODELS = _build_model_options()


class MessageWidget(Static):
    """A chat message widget."""

    def __init__(
        self, role: str, content: str, model_info: str = "", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.text_content = content
        self.model_info = model_info

    def compose(self) -> ComposeResult:
        role_label = "You" if self.role == "user" else "Assistant"
        role_style = "bold cyan" if self.role == "user" else "bold green"

        header_text = f"[{role_style}]{role_label}[/]"
        if self.model_info:
            header_text += f" {self.model_info}"

        yield Static(header_text)
        yield Markdown(self.text_content)


class StatusWidget(Static):
    """Animated status indicator widget with spinner."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._frame_index = 0
        self._status_text = "Thinking..."
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        """Start animation on mount."""
        self._timer = self.set_interval(0.1, self._animate)
        self._update_display()

    def on_unmount(self) -> None:
        """Stop animation on unmount."""
        if self._timer:
            self._timer.stop()

    def _animate(self) -> None:
        """Advance animation frame."""
        self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current frame and status."""
        spinner = self.SPINNER_FRAMES[self._frame_index]
        self.update(f"[cyan]{spinner}[/] [dim]{self._status_text}[/]")

    def set_status(self, text: str) -> None:
        """Update the status text."""
        self._status_text = text
        # Use call_later to ensure update happens in Textual's event loop
        self.call_later(self._update_display)


class ExportScreen(Screen):
    """Screen for exporting conversation."""

    CSS = """
    ExportScreen {
        align: center middle;
    }

    #export-dialog {
        width: 60;
        height: auto;
        padding: 2;
        background: $surface;
        border: thick $primary;
    }

    #export-buttons {
        margin-top: 1;
    }

    #export-buttons Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, messages: list[tuple[str, str]]) -> None:
        super().__init__()
        self.messages = messages

    def compose(self) -> ComposeResult:
        with Container(id="export-dialog"):
            yield Static("[bold]Export Conversation[/]\n")
            yield Label("Filename:")
            yield Input(value="conversation.md", id="filename-input")
            yield Static("")
            yield Label("Format:")
            yield Select(
                [("Markdown (.md)", "md"), ("JSON (.json)", "json")],
                id="format-select",
                value="md",
            )
            with Horizontal(id="export-buttons"):
                yield Button("Export", variant="primary", id="export-btn")
                yield Button("Cancel", id="cancel-btn")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-btn":
            await self._do_export()
        else:
            self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    async def _do_export(self) -> None:
        import json
        from datetime import datetime
        from pathlib import Path

        filename = self.query_one("#filename-input", Input).value
        format_select = self.query_one("#format-select", Select)
        export_format = format_select.value

        # Generate content
        if export_format == "json":
            content = json.dumps(
                {
                    "exported_at": datetime.now().isoformat(),
                    "messages": [
                        {"role": role, "content": text} for role, text in self.messages
                    ],
                },
                indent=2,
            )
            if not filename.endswith(".json"):
                filename = filename.rsplit(".", 1)[0] + ".json"
        else:
            lines = ["# AI Orchestrator Conversation\n"]
            for role, text in self.messages:
                role_text = "**You:**" if role == "user" else "**Assistant:**"
                lines.append(f"{role_text}\n\n{text}\n\n---\n")
            content = "\n".join(lines)
            if not filename.endswith(".md"):
                filename = filename.rsplit(".", 1)[0] + ".md"

        # Write file to Downloads
        try:
            filepath = Path.home() / "Downloads" / filename
            filepath.write_text(content, encoding="utf-8")
            self.app.notify(f"Exported to {filepath}", severity="information")
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")

        self.app.pop_screen()


class RuntimeSettingsScreen(Screen):
    """Screen showing live effective runtime settings."""

    CSS = """
    RuntimeSettingsScreen {
        align: center middle;
    }

    #runtime-dialog {
        width: 100;
        height: 36;
        padding: 1;
        background: $surface;
        border: thick $primary;
    }

    #runtime-title {
        margin-bottom: 1;
    }

    #runtime-markdown {
        height: 1fr;
        overflow-y: auto;
        border: round $boost;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(self, snapshot_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self._snapshot_provider = snapshot_provider
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Container(id="runtime-dialog"):
            yield Static("[bold]Runtime Settings (Live)[/]", id="runtime-title")
            yield Markdown("", id="runtime-markdown")
            yield Button("Close", id="runtime-close", variant="primary")

    def on_mount(self) -> None:
        self._update_snapshot()
        self._refresh_timer = self.set_interval(1.0, self._update_snapshot)

    def on_unmount(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()

    def _format_snapshot(self, snapshot: dict[str, Any]) -> str:
        runtime = snapshot.get("runtime", {})
        toggles = snapshot.get("toggles", {})
        effective = snapshot.get("effective", {})

        if not isinstance(runtime, dict):
            runtime = {}
        if not isinstance(toggles, dict):
            toggles = {}
        if not isinstance(effective, dict):
            effective = {}

        lines = [
            "## Runtime Defaults",
            f"- max_tokens: `{runtime.get('max_tokens')}`",
            f"- temperature: `{runtime.get('temperature')}`",
            f"- agent_default_session: `{runtime.get('agent_default_session')}`",
            f"- agent_default_model: `{runtime.get('agent_default_model')}`",
            f"- agent_web_tools: `{runtime.get('agent_web_tools')}`",
            f"- agent_mcp_enabled: `{runtime.get('agent_mcp_enabled')}`",
            f"- agent_skills_enabled: `{runtime.get('agent_skills_enabled')}`",
            f"- agent_browser_enabled: `{runtime.get('agent_browser_enabled')}`",
            "",
            "## Current Toggles",
            f"- model: `{toggles.get('model')}`",
            f"- think: `{toggles.get('think')}`",
            f"- web: `{toggles.get('web')}`",
            f"- agent: `{toggles.get('agent')}`",
            f"- browser: `{toggles.get('browser')}`",
            f"- image: `{toggles.get('image')}`",
            f"- incognito: `{toggles.get('incognito')}`",
            "",
            "## Effective Agent Flags",
            f"- web_tools: `{effective.get('web_tools')}`",
            f"- mcp: `{effective.get('mcp')}`",
            f"- skills: `{effective.get('skills')}`",
            f"- browser_automation: `{effective.get('browser_automation')}`",
        ]
        return "\n".join(lines)

    def _update_snapshot(self) -> None:
        markdown = self.query_one("#runtime-markdown", Markdown)
        snapshot = self._snapshot_provider()
        markdown.update(self._format_snapshot(snapshot))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "runtime-close":
            self.app.pop_screen()

    def action_close(self) -> None:
        self.app.pop_screen()


class ChatScreen(Screen):
    """Main chat screen."""

    CSS = """
    ChatScreen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto auto;
    }

    #chat-container {
        height: 100%;
        overflow-y: auto;
        padding: 1 2;
        background: $surface;
    }

    .message {
        margin-bottom: 1;
        padding: 1;
    }

    .user-message {
        background: #2B5278;
        border-left: thick #569CD6;
    }

    .assistant-message {
        background: #2D2D2D;
        border-left: thick #4EC9B0;
    }

    .system-message {
        background: #1E3A5F;
        border-left: thick #569CD6;
    }

    #loading, #compact-status {
        padding: 1;
        margin-bottom: 1;
        background: #2B3E50;
        border-left: thick #FFD700;
    }

    #controls {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border-top: solid $primary;
    }

    #model-select {
        width: 30;
    }

    .toggle-container {
        height: auto;
        width: auto;
        margin-right: 2;
    }

    .toggle-label {
        width: auto;
        margin-right: 1;
    }

    #input-container {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
    }

    #message-input {
        width: 1fr;
    }

    #send-button {
        width: 10;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("ctrl+e", "export_chat", "Export"),
        Binding("ctrl+k", "compact_chat", "Compact"),
        Binding("ctrl+r", "runtime_settings", "Runtime"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self, orchestrator: Any = None) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.agent_runner = AgentRunner(orchestrator) if orchestrator else None
        self.runtime_settings: dict[str, Any] = self._default_runtime_settings()
        self._applying_default_toggles = False
        self._load_runtime_settings_from_orchestrator()
        self.messages: list[tuple[str, str]] = []
        self.is_processing = False

    @staticmethod
    def _default_runtime_settings() -> dict[str, Any]:
        return {
            "max_tokens": 4096,
            "temperature": 0.7,
            "agent_enabled_default": False,
            "agent_default_session": "tui-default",
            "agent_default_model": "mlx-qwen3-coder-30b",
            "agent_web_tools": True,
            "agent_mcp_enabled": True,
            "agent_skills_enabled": True,
            "agent_browser_enabled": False,
        }

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        return default

    @staticmethod
    def _coerce_int(value: Any, default: int, *, minimum: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return max(minimum, value)
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float, *, minimum: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return max(minimum, float(value))
        return default

    @staticmethod
    def _coerce_str(value: Any, default: str) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        return default

    def _load_runtime_settings_from_orchestrator(self) -> None:
        settings = self._default_runtime_settings()
        if self.orchestrator is None:
            self.runtime_settings = settings
            return

        user_config = getattr(self.orchestrator, "_user_config", {})
        if not isinstance(user_config, dict):
            self.runtime_settings = settings
            return

        defaults = user_config.get("defaults", {})
        if isinstance(defaults, dict):
            settings["max_tokens"] = self._coerce_int(
                defaults.get("maxTokens"),
                settings["max_tokens"],
                minimum=100,
            )
            settings["temperature"] = self._coerce_float(
                defaults.get("temperature"),
                settings["temperature"],
                minimum=0.0,
            )

        agent_cfg = user_config.get("agent", {})
        if isinstance(agent_cfg, dict):
            settings["agent_enabled_default"] = self._coerce_bool(
                agent_cfg.get("enabledByDefault"),
                settings["agent_enabled_default"],
            )
            settings["agent_default_session"] = self._coerce_str(
                agent_cfg.get("defaultSessionId"),
                settings["agent_default_session"],
            )
            settings["agent_default_model"] = self._coerce_str(
                agent_cfg.get("defaultModel"),
                settings["agent_default_model"],
            )
            settings["agent_web_tools"] = self._coerce_bool(
                agent_cfg.get("enableWebTools"),
                settings["agent_web_tools"],
            )
            settings["agent_mcp_enabled"] = self._coerce_bool(
                agent_cfg.get("enableMcp"),
                settings["agent_mcp_enabled"],
            )
            settings["agent_skills_enabled"] = self._coerce_bool(
                agent_cfg.get("enableSkills"),
                settings["agent_skills_enabled"],
            )
            settings["agent_browser_enabled"] = self._coerce_bool(
                agent_cfg.get("enableBrowserAutomation"),
                settings["agent_browser_enabled"],
            )

        self.runtime_settings = settings

    def _apply_runtime_settings_to_controls(self) -> None:
        model_select = self.query_one("#model-select", Select)
        default_model = self.runtime_settings.get("agent_default_model")
        available_models = {value for _, value in MODELS}
        if isinstance(default_model, str) and default_model in available_models:
            model_select.value = default_model

        self._applying_default_toggles = True
        try:
            self.query_one("#agent-toggle", Switch).value = bool(
                self.runtime_settings.get("agent_enabled_default", False)
            )
            self.query_one("#browser-toggle", Switch).value = bool(
                self.runtime_settings.get("agent_browser_enabled", False)
            )
        finally:
            self._applying_default_toggles = False

    def compose(self) -> ComposeResult:
        yield Header()

        # Chat messages container
        with ScrollableContainer(id="chat-container"):
            yield Static(
                "[bold]Welcome to AI Orchestrator[/]\n\n"
                "Start typing to chat with AI. Use the controls above to select a model.\n\n"
                "[dim]Keyboard shortcuts:[/]\n"
                "  • [bold]Enter[/] - Send message\n"
                "  • [bold]Ctrl+N[/] - New chat\n"
                "  • [bold]Ctrl+E[/] - Export conversation\n"
                "  • [bold]Ctrl+K[/] - Compact history (summarize)\n"
                "  • [bold]Ctrl+R[/] - Runtime settings\n"
                "  • [bold]Ctrl+Q[/] - Quit\n",
                id="welcome",
            )

        # Controls
        with Horizontal(id="controls"):
            yield Label("Model:", classes="toggle-label")
            yield Select(
                [(name, value) for name, value in MODELS],
                id="model-select",
                value="auto",
            )

            with Horizontal(classes="toggle-container"):
                yield Label("Think", classes="toggle-label")
                yield Switch(id="think-toggle")

            with Horizontal(classes="toggle-container"):
                yield Label("Web", classes="toggle-label")
                yield Switch(id="web-toggle")

            with Horizontal(classes="toggle-container"):
                yield Label("Agent", classes="toggle-label")
                yield Switch(id="agent-toggle")

            with Horizontal(classes="toggle-container"):
                yield Label("Browser", classes="toggle-label")
                yield Switch(id="browser-toggle")

            with Horizontal(classes="toggle-container"):
                yield Label("Image", classes="toggle-label")
                yield Switch(id="image-toggle")

            with Horizontal(classes="toggle-container"):
                yield Label("Incognito", classes="toggle-label")
                yield Switch(id="incognito-toggle")

        # Input area
        with Horizontal(id="input-container"):
            yield Input(placeholder="Message AI Orchestrator...", id="message-input")
            yield Button("Send", id="send-button", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        """Focus input on mount."""
        self._apply_runtime_settings_to_controls()
        self.query_one("#message-input", Input).focus()

    def action_focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#message-input", Input).focus()

    def action_new_chat(self) -> None:
        """Clear chat and start new."""
        self.messages.clear()
        if self.orchestrator:
            self.orchestrator.clear_history()
        container = self.query_one("#chat-container", ScrollableContainer)
        for child in list(container.children):
            if child.id != "welcome":
                child.remove()
        welcome = self.query_one("#welcome", Static)
        welcome.display = True

    def action_export_chat(self) -> None:
        """Export the current conversation."""
        if not self.messages:
            self.app.notify("No messages to export", severity="warning")
            return
        self.app.push_screen(ExportScreen(self.messages))

    def _runtime_snapshot(self) -> dict[str, Any]:
        model_select = self.query_one("#model-select", Select)
        think_toggle = self.query_one("#think-toggle", Switch)
        web_toggle = self.query_one("#web-toggle", Switch)
        agent_toggle = self.query_one("#agent-toggle", Switch)
        browser_toggle = self.query_one("#browser-toggle", Switch)
        image_toggle = self.query_one("#image-toggle", Switch)
        incognito_toggle = self.query_one("#incognito-toggle", Switch)

        web_enabled = bool(web_toggle.value)
        browser_enabled = bool(browser_toggle.value)
        return {
            "runtime": dict(self.runtime_settings),
            "toggles": {
                "model": model_select.value,
                "think": bool(think_toggle.value),
                "web": web_enabled,
                "agent": bool(agent_toggle.value),
                "browser": browser_enabled,
                "image": bool(image_toggle.value),
                "incognito": bool(incognito_toggle.value),
            },
            "effective": {
                "web_tools": web_enabled
                and bool(self.runtime_settings.get("agent_web_tools", True)),
                "mcp": bool(self.runtime_settings.get("agent_mcp_enabled", True)),
                "skills": bool(self.runtime_settings.get("agent_skills_enabled", True)),
                "browser_automation": browser_enabled
                and bool(self.runtime_settings.get("agent_browser_enabled", False)),
            },
        }

    def action_runtime_settings(self) -> None:
        """Show effective runtime settings."""
        self.app.push_screen(RuntimeSettingsScreen(self._runtime_snapshot))

    async def action_compact_chat(self) -> None:
        """Compact conversation history into a summary."""
        if not self.orchestrator:
            self.app.notify("Orchestrator not available", severity="error")
            return

        if len(self.messages) < 2:
            self.app.notify("Not enough messages to compact", severity="warning")
            return

        if self.is_processing:
            self.app.notify("Please wait for current operation", severity="warning")
            return

        self.is_processing = True
        container = self.query_one("#chat-container", ScrollableContainer)

        # Show status while compacting
        status_widget = StatusWidget(id="compact-status")
        await container.mount(status_widget)
        status_widget.set_status("Compacting conversation...")
        container.scroll_end()

        try:
            result = await self.orchestrator.compact_conversation()

            # Remove status widget
            status_widget.remove()

            if result:
                # Add a visual separator showing the compaction
                summary_msg = MessageWidget(
                    role="system",
                    content=f"**Conversation Compacted**\n\n"
                    f"*{result.original_message_count} messages summarized "
                    f"(~{result.tokens_saved_estimate} tokens saved)*\n\n"
                    f"**Summary:**\n{result.summary}",
                    model_info=f"[dim]by {result.model_used}[/]",
                    classes="message system-message",
                )
                await container.mount(summary_msg)
                container.scroll_end()
                self.app.notify(
                    f"Compacted {result.original_message_count} messages",
                    severity="information",
                )
            else:
                self.app.notify("Failed to compact conversation", severity="error")

        except Exception as e:
            status_widget.remove()
            self.app.notify(f"Error: {e}", severity="error")
        finally:
            self.is_processing = False

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggle changes."""
        if self._applying_default_toggles:
            return

        if event.switch.id == "incognito-toggle":
            if self.orchestrator:
                self.orchestrator.set_incognito(event.value)
                if event.value:
                    self.app.notify("Incognito mode enabled", severity="information")
                else:
                    self.app.notify("Incognito mode disabled", severity="information")
        if event.switch.id == "browser-toggle" and event.value:
            self.app.notify(
                "Warning: Browser automation is enabled and can interact with real websites.",
                severity="warning",
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "message-input":
            await self._send_message(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "send-button":
            input_widget = self.query_one("#message-input", Input)
            await self._send_message(input_widget.value)

    async def _send_message(self, message: str) -> None:
        """Send a message."""
        message = message.strip()
        if not message or self.is_processing:
            return

        self.is_processing = True

        # Hide welcome message
        self.query_one("#welcome", Static).display = False

        # Clear input
        input_widget = self.query_one("#message-input", Input)
        input_widget.value = ""

        # Add user message
        container = self.query_one("#chat-container", ScrollableContainer)
        user_msg = MessageWidget(
            role="user", content=message, classes="message user-message"
        )
        await container.mount(user_msg)
        container.scroll_end()

        # Get settings
        model_select = self.query_one("#model-select", Select)
        model = model_select.value if model_select.value != "auto" else None

        think_enabled = self.query_one("#think-toggle", Switch).value
        web_enabled = self.query_one("#web-toggle", Switch).value
        agent_enabled = self.query_one("#agent-toggle", Switch).value
        browser_enabled = self.query_one("#browser-toggle", Switch).value
        image_enabled = self.query_one("#image-toggle", Switch).value
        max_tokens = self._coerce_int(
            self.runtime_settings.get("max_tokens"),
            4096,
            minimum=100,
        )
        temperature = self._coerce_float(
            self.runtime_settings.get("temperature"),
            0.7,
            minimum=0.0,
        )
        effective_web_tools = web_enabled and bool(
            self.runtime_settings.get("agent_web_tools", True)
        )
        effective_browser_enabled = browser_enabled and bool(
            self.runtime_settings.get("agent_browser_enabled", False)
        )

        # Show animated status indicator
        status_widget = StatusWidget(id="loading")
        await container.mount(status_widget)
        container.scroll_end()

        # Status callback for orchestrator
        def on_status(status: AgentStatus) -> None:
            status_widget.set_status(status.message)

        try:
            if self.orchestrator:
                # Prepare prompt
                prompt = message
                system_prompt = None

                if think_enabled:
                    system_prompt = (
                        "Think through problems step by step, showing your reasoning."
                    )
                if web_enabled:
                    prompt = f"[Web search enabled] {prompt}"
                if image_enabled:
                    prompt = f"[Image generation requested] {prompt}"

                if agent_enabled and self.agent_runner:
                    if browser_enabled and not effective_browser_enabled:
                        self.app.notify(
                            "Browser automation toggle is on, but disabled in config.",
                            severity="warning",
                        )

                    response = await self.agent_runner.run(
                        prompt,
                        options=AgentRunOptions(
                            session_id=self._coerce_str(
                                self.runtime_settings.get("agent_default_session"),
                                "tui-default",
                            ),
                            model_override=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            enable_web_tools=effective_web_tools,
                            enable_mcp=bool(
                                self.runtime_settings.get("agent_mcp_enabled", True)
                            ),
                            enable_skills=bool(
                                self.runtime_settings.get("agent_skills_enabled", True)
                            ),
                            enable_browser_automation=effective_browser_enabled,
                            incognito=self.query_one("#incognito-toggle", Switch).value,
                            system_prompt=system_prompt,
                        ),
                        status_callback=on_status,
                    )
                else:
                    # Query with status callback
                    response = await self.orchestrator.query(
                        prompt,
                        model_override=model,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        status_callback=on_status,
                    )

                if response.success:
                    response_text = response.content
                    model_used = f"[dim]({response.model})[/]"
                    role_style = "message assistant-message"
                else:
                    response_text = f"❌ Error: {response.error}"
                    model_used = "[dim](Error)[/]"
                    role_style = "message assistant-message"  # Or separate error style

            else:
                response_text = (
                    "Orchestrator not initialized. Please configure API keys.\n\n"
                    "Run `ai-configure` in terminal to set up your credentials."
                )
                model_used = ""
                role_style = "message assistant-message"

            # Remove status indicator
            status_widget.remove()

            # Add assistant message
            assistant_msg = MessageWidget(
                role="assistant",
                content=response_text,
                model_info=model_used,
                classes=role_style,
            )
            await container.mount(assistant_msg)
            container.scroll_end()

            self.messages.append(("user", message))
            self.messages.append(("assistant", response_text))

        except Exception as e:
            status_widget.remove()
            error_msg = MessageWidget(
                role="assistant",
                content=f"Error: {str(e)}",
                classes="message assistant-message",
            )
            await container.mount(error_msg)
            container.scroll_end()

        finally:
            self.is_processing = False
            input_widget.focus()


class AIChat(App):
    """AI Orchestrator Terminal UI Application."""

    TITLE = "AI Orchestrator"
    CSS = """
    Screen {
        background: #1E1E1E;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator: Any = None

    def on_mount(self) -> None:
        """Initialize on mount."""
        try:
            from ..orchestrator import AIOrchestrator

            self.orchestrator = AIOrchestrator(verbose=False)
        except Exception as e:
            self.notify(f"Could not initialize orchestrator: {e}", severity="warning")

        # Push chat screen
        self.push_screen(ChatScreen(self.orchestrator))


def main() -> int:
    """Main entry point."""
    app = AIChat()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
