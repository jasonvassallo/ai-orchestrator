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
from typing import Callable

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

    def __init__(self):
        super().__init__(
            "AI",
            icon=None,  # Use text instead of icon
            quit_button=None,  # We'll add our own
        )
        self.orchestrator = None
        self._init_orchestrator()
        self._build_menu()

    def _init_orchestrator(self) -> None:
        """Initialize the orchestrator."""
        try:
            from ..orchestrator import AIOrchestrator
            self.orchestrator = AIOrchestrator(verbose=False)
        except Exception as e:
            rumps.alert(
                title="Orchestrator Error",
                message=f"Could not initialize: {e}\n\nRun 'ai-configure' to set up API keys.",
            )

    def _build_menu(self) -> None:
        """Build the menu items."""
        self.menu = [
            rumps.MenuItem("Quick Query...", callback=self.quick_query),
            None,  # Separator
            rumps.MenuItem("Models", [
                rumps.MenuItem("Auto (Best)", callback=lambda _: self.set_model(None)),
                None,
                rumps.MenuItem("Claude Opus 4.5", callback=lambda _: self.set_model("claude-opus-4.5")),
                rumps.MenuItem("Claude Sonnet 4.5", callback=lambda _: self.set_model("claude-sonnet-4.5")),
                rumps.MenuItem("GPT-4o", callback=lambda _: self.set_model("gpt-4o")),
                rumps.MenuItem("Gemini 2.0 Flash", callback=lambda _: self.set_model("gemini-2.0-flash")),
                rumps.MenuItem("Llama 3.2 (Local)", callback=lambda _: self.set_model("llama3.2")),
            ]),
            None,
            rumps.MenuItem("Open GUI App", callback=self.open_gui),
            rumps.MenuItem("Open Terminal UI", callback=self.open_tui),
            None,
            rumps.MenuItem("Configure API Keys", callback=self.configure_keys),
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

        self.selected_model = None

    def set_model(self, model: str | None) -> None:
        """Set the selected model."""
        self.selected_model = model
        model_name = model or "Auto"
        rumps.notification(
            title="AI Orchestrator",
            subtitle="Model Changed",
            message=f"Now using: {model_name}",
        )

    @rumps.clicked("Quick Query...")
    def quick_query(self, _) -> None:
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
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if self.orchestrator:
                    result = loop.run_until_complete(
                        self.orchestrator.query(query, model_override=self.selected_model)
                    )
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
                loop.close()

        # Run in background thread
        thread = threading.Thread(target=run_async)
        thread.start()

    @rumps.clicked("Open GUI App")
    def open_gui(self, _) -> None:
        """Open the GUI application."""
        import subprocess
        try:
            subprocess.Popen([sys.executable, "-m", "src.gui.app"])
            rumps.notification(
                title="AI Orchestrator",
                subtitle="",
                message="Opening GUI app...",
            )
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open GUI: {e}")

    @rumps.clicked("Open Terminal UI")
    def open_tui(self, _) -> None:
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
            subprocess.Popen(["osascript", "-e", script])
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open TUI: {e}")

    @rumps.clicked("Configure API Keys")
    def configure_keys(self, _) -> None:
        """Open terminal to configure API keys."""
        import subprocess
        try:
            script = f'''
            tell application "Terminal"
                do script "{sys.executable} -m src.credentials"
                activate
            end tell
            '''
            subprocess.Popen(["osascript", "-e", script])
        except Exception as e:
            rumps.alert(title="Error", message=f"Could not open configuration: {e}")


def main() -> int:
    """Main entry point."""
    app = AIMenuBarApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
