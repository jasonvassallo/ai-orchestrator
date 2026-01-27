#!/usr/bin/env python3
"""
AI Orchestrator Menu Bar App
=============================

A macOS menu bar application for quick AI access.

Usage:
    python -m src.menubar.app

Or after installation:
    ai-menubar
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Any

# Check for macOS
if sys.platform != "darwin":
    print("Error: Menu bar app is only available on macOS")
    sys.exit(1)

try:
    import rumps
except ImportError:
    print("Error: rumps is required for the menu bar app.")
    print("Install it with: pip install rumps")
    sys.exit(1)


class AIMenuBarApp(rumps.App):
    """Menu bar application for AI Orchestrator."""

    def __init__(self) -> None:
        super().__init__(
            "AI",
            icon=None,  # Use text instead of icon
            quit_button=None,  # We'll add our own
        )
        self.orchestrator: Any = None
        self._last_query: str | None = None
        self._last_response: str | None = None
        self._last_model: str | None = None
        self._original_title = "AI"
        self._init_orchestrator()
        self._build_menu()

    def _init_orchestrator(self) -> None:
        """Initialize the orchestrator."""
        try:
            from ..orchestrator import LOCAL_PROVIDERS, AIOrchestrator, ModelRegistry

            self.orchestrator = AIOrchestrator(verbose=False)
            self._local_providers = LOCAL_PROVIDERS
            self._model_registry = ModelRegistry
        except Exception as e:
            rumps.alert(
                title="Orchestrator Error",
                message=f"Could not initialize: {e}\n\nRun 'ai-configure' to set up API keys.",
            )

    def _build_menu(self) -> None:
        """Build the menu items."""
        models_menu = self._build_models_menu()
        self._incognito_item = rumps.MenuItem(
            "Incognito Mode", callback=self.toggle_incognito
        )
        self.menu = [
            rumps.MenuItem("Quick Query...", callback=self.quick_query),
            None,  # Separator
            rumps.MenuItem("Models", models_menu),
            self._incognito_item,
            None,
            rumps.MenuItem("Export Last Query", callback=self.export_last),
            rumps.MenuItem("Compact History", callback=self.compact_history),
            rumps.MenuItem("Copy Last Response", callback=self.copy_last),
            None,
            rumps.MenuItem("Open GUI App", callback=self.open_gui),
            rumps.MenuItem("Open Terminal UI", callback=self.open_tui),
            None,
            rumps.MenuItem("Configure API Keys", callback=self.configure_keys),
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

        self.selected_model: str | None = None
        self._incognito_enabled = False

    def _build_models_menu(self) -> list[rumps.MenuItem | None]:
        if not hasattr(self, "_model_registry"):
            return [
                rumps.MenuItem("Auto (Best)", callback=lambda _: self.set_model(None))
            ]

        models_by_provider: dict[str, list[tuple[str, str]]] = {}
        for key, model in self._model_registry.MODELS.items():
            models_by_provider.setdefault(model.provider, []).append((key, model.name))

        menu_items: list[rumps.MenuItem | None] = [
            rumps.MenuItem("Auto (Best)", callback=lambda _: self.set_model(None)),
            None,
        ]

        for provider in sorted(models_by_provider):
            entries = []
            for key, name in sorted(
                models_by_provider[provider], key=lambda item: item[1]
            ):
                tag = "Local, " if provider in self._local_providers else ""
                label = f"{name} ({tag}{provider})"
                entries.append(
                    rumps.MenuItem(label, callback=lambda _, m=key: self.set_model(m))
                )
            menu_items.append(rumps.MenuItem(provider.title(), entries))

        return menu_items

    def set_model(self, model: str | None) -> None:
        """Set the selected model."""
        self.selected_model = model
        model_name = model or "Auto"
        rumps.notification(
            title="AI Orchestrator",
            subtitle="Model Changed",
            message=f"Now using: {model_name}",
        )

    @rumps.clicked("Incognito Mode")
    def toggle_incognito(self, sender: Any) -> None:
        """Toggle incognito mode."""
        self._incognito_enabled = not self._incognito_enabled
        if self.orchestrator:
            self.orchestrator.set_incognito(self._incognito_enabled)

        # Update menu item with checkmark
        sender.state = self._incognito_enabled

        status = "enabled" if self._incognito_enabled else "disabled"
        rumps.notification(
            title="AI Orchestrator",
            subtitle="Incognito Mode",
            message=f"Incognito mode {status}",
        )

    @rumps.clicked("Quick Query...")
    def quick_query(self, _: Any) -> None:
        """Show quick query dialog."""
        # Create a simple input window
        response = rumps.Window(
            message="Enter your query:",
            title="AI Orchestrator",
            default_text="",
            ok="Send",
            cancel="Cancel",
            dimensions=(400, 100),
        )
        result = response.run()

        if result.clicked and result.text.strip():
            self._run_query(result.text.strip())

    def _run_query(self, query: str) -> None:
        """Run a query in a background thread."""
        from ..orchestrator import AgentStatus

        # Short status messages for menu bar
        status_messages = {
            "VALIDATING": "Validating...",
            "CLASSIFYING": "Classifying...",
            "ROUTING": "Routing...",
            "ROUTING_LLM": "Analyzing...",
            "SELECTING": "Selecting...",
            "CHAINING": "Chaining...",
            "GENERATING": "Generating...",
            "THINKING": "Thinking...",
            "SEARCHING": "Searching...",
        }

        def on_status(status: AgentStatus) -> None:
            """Handle status updates."""
            stage_name = status.stage.name
            msg = status_messages.get(stage_name, "Processing...")
            self.title = f"AI: {msg}"

        def run_async() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if self.orchestrator:
                    result = loop.run_until_complete(
                        self.orchestrator.query(
                            query,
                            model_override=self.selected_model,
                            status_callback=on_status,
                        )
                    )

                    # Store last query/response
                    self._last_query = query
                    self._last_response = result.content
                    self._last_model = result.model

                    # Show result in a dialog (truncated for notification)
                    content = result.content
                    if len(content) > 500:
                        content = content[:500] + "..."

                    rumps.alert(
                        title=f"Response from {result.model}",
                        message=content,
                    )
                else:
                    rumps.alert(
                        title="Error",
                        message="Orchestrator not initialized. Run 'ai-configure' first.",
                    )
            except Exception as e:
                rumps.alert(title="Error", message=str(e))
            finally:
                # Restore original title
                self.title = self._original_title
                loop.close()

        # Run in background thread
        thread = threading.Thread(target=run_async)
        thread.start()

    @rumps.clicked("Open GUI App")
    def open_gui(self, _: Any) -> None:
        """Open the GUI application."""
        import subprocess

        try:
            subprocess.Popen([sys.executable, "-m", "src.gui.app"])  # noqa: S603
            rumps.notification(
                title="AI Orchestrator",
                subtitle="",
                message="Opening GUI app...",
            )
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open GUI: {e}")

    @rumps.clicked("Open Terminal UI")
    def open_tui(self, _: Any) -> None:
        """Open the terminal UI in a new Terminal window."""
        import subprocess

        try:
            # Open Terminal and run the TUI
            script = f'''
            tell application "Terminal"
                do script "{sys.executable} -m src.tui.app"
                activate
            end tell
            '''
            subprocess.Popen(["osascript", "-e", script])  # noqa: S603
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open TUI: {e}")

    @rumps.clicked("Export Last Query")
    def export_last(self, _: Any) -> None:
        """Export last query/response to markdown file."""
        from datetime import datetime
        from pathlib import Path

        if not self._last_query or not self._last_response:
            rumps.alert(
                title="Nothing to Export",
                message="No query has been made yet. Use 'Quick Query...' first.",
            )
            return

        # Generate markdown content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"""# AI Orchestrator Query

**Date:** {timestamp}
**Model:** {self._last_model or "Unknown"}

---

## Prompt

{self._last_query}

---

## Response

{self._last_response}
"""

        # Save to Downloads folder
        try:
            filename = f"ai_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            filepath = Path.home() / "Downloads" / filename
            filepath.write_text(content, encoding="utf-8")
            rumps.notification(
                title="AI Orchestrator",
                subtitle="Export Complete",
                message=f"Saved to ~/Downloads/{filename}",
            )
        except Exception as e:
            rumps.alert(title="Export Error", message=f"Could not save file: {e}")

    @rumps.clicked("Compact History")
    def compact_history(self, _: Any) -> None:
        """Compact conversation history into a summary."""
        import asyncio

        if not self.orchestrator:
            rumps.alert(
                title="Not Available",
                message="Orchestrator not initialized.",
            )
            return

        # Check if there's enough history
        if len(self.orchestrator.conversation_history) < 2:
            rumps.alert(
                title="Nothing to Compact",
                message="Not enough conversation history to compact.",
            )
            return

        async def do_compact() -> None:
            result = await self.orchestrator.compact_conversation()
            if result:
                rumps.notification(
                    title="AI Orchestrator",
                    subtitle="History Compacted",
                    message=f"Compacted {result.original_message_count} messages "
                    f"(~{result.tokens_saved_estimate} tokens saved)",
                )
            else:
                rumps.alert(
                    title="Compaction Failed",
                    message="Failed to compact conversation history.",
                )

        # Run asynchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(do_compact())

    @rumps.clicked("Copy Last Response")
    def copy_last(self, _: Any) -> None:
        """Copy last response to clipboard."""
        import subprocess

        if not self._last_response:
            rumps.alert(
                title="Nothing to Copy",
                message="No response available. Use 'Quick Query...' first.",
            )
            return

        try:
            # Use pbcopy on macOS
            process = subprocess.Popen(
                ["pbcopy"],
                stdin=subprocess.PIPE,
                text=True,
            )
            process.communicate(input=self._last_response)
            rumps.notification(
                title="AI Orchestrator",
                subtitle="Copied",
                message="Response copied to clipboard!",
            )
        except Exception as e:
            rumps.alert(title="Copy Error", message=f"Could not copy to clipboard: {e}")

    @rumps.clicked("Configure API Keys")
    def configure_keys(self, _: Any) -> None:
        """Open terminal to configure API keys."""
        import subprocess

        try:
            script = f'''
            tell application "Terminal"
                do script "{sys.executable} -m src.credentials"
                activate
            end tell
            '''
            subprocess.Popen(["osascript", "-e", script])  # noqa: S603
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open configuration: {e}")


def main() -> int:
    """Main entry point."""
    app = AIMenuBarApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
