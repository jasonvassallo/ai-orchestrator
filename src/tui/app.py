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
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer
    from textual.screen import Screen
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


from ..orchestrator import LOCAL_PROVIDERS, ModelRegistry


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
        tags.append(model.provider)
        label = f"{model.name} ({', '.join(tags)})"
        options.append((label, key))
    return options


# Available models (dynamic from registry)
MODELS = _build_model_options()


class MessageWidget(Static):
    """A chat message widget."""

    def __init__(self, role: str, content: str, model_info: str = "", **kwargs: Any) -> None:
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
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self, orchestrator: Any = None) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.messages: list[tuple[str, str]] = []
        self.is_processing = False

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
                yield Label("Image", classes="toggle-label")
                yield Switch(id="image-toggle")

        # Input area
        with Horizontal(id="input-container"):
            yield Input(placeholder="Message AI Orchestrator...", id="message-input")
            yield Button("Send", id="send-button", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        """Focus input on mount."""
        self.query_one("#message-input", Input).focus()

    def action_focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#message-input", Input).focus()

    def action_new_chat(self) -> None:
        """Clear chat and start new."""
        self.messages.clear()
        container = self.query_one("#chat-container", ScrollableContainer)
        for child in list(container.children):
            if child.id != "welcome":
                child.remove()
        welcome = self.query_one("#welcome", Static)
        welcome.display = True

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
            role="user",
            content=message,
            classes="message user-message"
        )
        await container.mount(user_msg)
        container.scroll_end()

        # Get settings
        model_select = self.query_one("#model-select", Select)
        model = model_select.value if model_select.value != "auto" else None

        think_enabled = self.query_one("#think-toggle", Switch).value
        web_enabled = self.query_one("#web-toggle", Switch).value
        image_enabled = self.query_one("#image-toggle", Switch).value

        # Show processing indicator
        loading = Static("[dim]Thinking...[/]", id="loading")
        await container.mount(loading)
        container.scroll_end()

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

                # Query
                response = await self.orchestrator.query(
                    prompt,
                    model_override=model,
                    system_prompt=system_prompt,
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

            # Remove loading indicator
            loading.remove()

            # Add assistant message
            assistant_msg = MessageWidget(
                role="assistant",
                content=response_text,
                model_info=model_used,
                classes=role_style
            )
            await container.mount(assistant_msg)
            container.scroll_end()

            self.messages.append(("user", message))
            self.messages.append(("assistant", response_text))

        except Exception as e:
            loading.remove()
            error_msg = MessageWidget(
                role="assistant",
                content=f"Error: {str(e)}",
                classes="message assistant-message"
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
