"""
Main Window
===========

The primary application window with chat interface.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..storage import Conversation, get_storage
from .chat_widget import ChatWidget, WelcomeWidget
from .input_widget import InputWidget
from .settings_dialog import SettingsDialog
from .sidebar import Sidebar
from .styles import STYLESHEET

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner
    from ..orchestrator import AgentStatus, AIOrchestrator


class AsyncWorker(QObject):
    """Worker for running async operations."""

    finished = Signal(object)  # result
    error = Signal(str)  # error message
    chunk = Signal(str)  # streaming chunk

    def __init__(self, coro, parent=None):
        super().__init__(parent)
        self.coro = coro

    async def run(self):
        try:
            result = await self.coro
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._storage = get_storage()
        self._current_conversation: Conversation | None = None
        self._orchestrator: AIOrchestrator | None = None
        self._agent_runner: AgentRunner | None = None
        self._runtime_settings: dict[str, Any] = self._default_runtime_settings()
        self._is_processing = False

        self._setup_ui()
        self._setup_menu_bar()
        self._load_conversations()
        self._init_orchestrator()

    @staticmethod
    def _default_runtime_settings() -> dict[str, Any]:
        return {
            "max_tokens": 4096,
            "temperature": 0.7,
            "agent_enabled_default": False,
            "agent_default_session": "gui-default",
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
        if self._orchestrator is None:
            self._runtime_settings = settings
            return

        user_config = getattr(self._orchestrator, "_user_config", {})
        if not isinstance(user_config, dict):
            self._runtime_settings = settings
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

        self._runtime_settings = settings

    def _apply_runtime_settings_to_input(self) -> None:
        self.input_widget.apply_agent_defaults(
            enabled=bool(self._runtime_settings.get("agent_enabled_default", False)),
            browser_automation=bool(
                self._runtime_settings.get("agent_browser_enabled", False)
            ),
        )
        default_model = self._runtime_settings.get("agent_default_model")
        if isinstance(default_model, str):
            self.input_widget.set_model_by_id(default_model)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("AI Orchestrator")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for sidebar and main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.newChatClicked.connect(self._new_chat)
        self.sidebar.conversationSelected.connect(self._load_conversation)
        self.sidebar.conversationDeleted.connect(self._delete_conversation)
        self.sidebar.conversationRenamed.connect(self._rename_conversation)
        self.sidebar.settingsClicked.connect(self._show_settings)
        splitter.addWidget(self.sidebar)

        # Main content area
        main_content = QWidget()
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Stacked widget for welcome screen and chat
        self.content_stack = QStackedWidget()

        # Welcome screen
        self.welcome_widget = WelcomeWidget()
        self.content_stack.addWidget(self.welcome_widget)

        # Chat widget
        self.chat_widget = ChatWidget()
        self.content_stack.addWidget(self.chat_widget)

        main_layout.addWidget(self.content_stack, 1)

        # Input widget
        self.input_widget = InputWidget()
        self.input_widget.messageSent.connect(self._on_message_sent)
        main_layout.addWidget(self.input_widget)

        splitter.addWidget(main_content)

        # Set splitter sizes (sidebar: 280, content: rest)
        splitter.setSizes([280, 920])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        layout.addWidget(splitter)

        # Focus input on start
        self.input_widget.focus_input()

    def _setup_menu_bar(self) -> None:
        """Set up the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_chat_action = QAction("New Chat", self)
        new_chat_action.setShortcut(QKeySequence("Ctrl+N"))
        new_chat_action.triggered.connect(self._new_chat)
        file_menu.addAction(new_chat_action)

        file_menu.addSeparator()

        export_action = QAction("Export Conversation...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_conversation)
        file_menu.addAction(export_action)

        compact_action = QAction("Compact History...", self)
        compact_action.setShortcut(QKeySequence("Ctrl+K"))
        compact_action.triggered.connect(self._compact_conversation)
        file_menu.addAction(compact_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        copy_all_action = QAction("Copy Entire Conversation", self)
        copy_all_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        copy_all_action.triggered.connect(self._copy_conversation)
        edit_menu.addAction(copy_all_action)

        copy_last_action = QAction("Copy Last Response", self)
        copy_last_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        copy_last_action.triggered.connect(self._copy_last_response)
        edit_menu.addAction(copy_last_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About AI Orchestrator", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _export_conversation(self) -> None:
        """Export the current conversation to a file."""
        self.chat_widget.export_to_file()

    def _compact_conversation(self) -> None:
        """Compact conversation history into a summary."""
        import asyncio

        if not self._orchestrator:
            QMessageBox.warning(
                self,
                "Not Available",
                "Orchestrator not initialized.",
            )
            return

        if len(self.chat_widget._messages) < 2:
            QMessageBox.information(
                self,
                "Nothing to Compact",
                "Not enough messages to compact. Continue the conversation first.",
            )
            return

        # Run compaction asynchronously
        async def do_compact() -> None:
            result = await self._orchestrator.compact_conversation()
            if result:
                # Add a visual separator showing the compaction
                self.chat_widget.add_message(
                    "system",
                    f"**Conversation Compacted**\n\n"
                    f"*{result.original_message_count} messages summarized "
                    f"(~{result.tokens_saved_estimate} tokens saved)*\n\n"
                    f"**Summary:**\n{result.summary}",
                )
                QMessageBox.information(
                    self,
                    "Compacted",
                    f"Compacted {result.original_message_count} messages.\n"
                    f"Estimated tokens saved: ~{result.tokens_saved_estimate}",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Failed",
                    "Failed to compact conversation. Please try again.",
                )

        asyncio.ensure_future(do_compact())

    def _copy_conversation(self) -> None:
        """Copy the entire conversation to clipboard."""
        if self.chat_widget.copy_all_to_clipboard():
            QMessageBox.information(
                self,
                "Copied",
                "Conversation copied to clipboard!",
            )
        else:
            QMessageBox.information(
                self,
                "No Conversation",
                "No messages to copy. Start a conversation first!",
            )

    def _copy_last_response(self) -> None:
        """Copy the last assistant response to clipboard."""
        from PySide6.QtWidgets import QApplication

        messages = self.chat_widget._messages
        for msg in reversed(messages):
            if msg.role == "assistant":
                clipboard = QApplication.clipboard()
                clipboard.setText(msg.content)
                QMessageBox.information(
                    self,
                    "Copied",
                    "Last response copied to clipboard!",
                )
                return
        QMessageBox.information(
            self,
            "No Response",
            "No assistant response to copy.",
        )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AI Orchestrator",
            "AI Orchestrator v2.0.0\n\n"
            "Intelligent multi-model AI assistant.\n\n"
            "Supports 25+ models across providers:\n"
            "OpenAI, Anthropic, Google, Mistral,\n"
            "Groq, xAI, Perplexity, DeepSeek, Moonshot, MLX\n\n"
            "github.com/jasonvassallo/ai-orchestrator",
        )

    def _init_orchestrator(self) -> None:
        """Initialize the AI orchestrator."""
        try:
            from ..agent.runner import AgentRunner
            from ..orchestrator import AIOrchestrator

            self._orchestrator = AIOrchestrator(verbose=False)
            self._agent_runner = AgentRunner(self._orchestrator)
            self._load_runtime_settings_from_orchestrator()
            self._apply_runtime_settings_to_input()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Orchestrator Error",
                f"Could not initialize AI orchestrator: {e}\n\n"
                "Please configure your API keys using: ai-configure",
            )

    def _load_conversations(self) -> None:
        """Load conversations from storage."""
        conversations = self._storage.list_conversations()
        self.sidebar.load_conversations(conversations)

    def _new_chat(self) -> None:
        """Start a new chat."""
        self._current_conversation = None
        self.chat_widget.clear()
        self.content_stack.setCurrentWidget(self.welcome_widget)
        self.input_widget.focus_input()

    def _load_conversation(self, conversation_id: str) -> None:
        """Load a conversation by ID."""
        conversation = self._storage.get_conversation(conversation_id)
        if conversation:
            self._current_conversation = conversation
            self.chat_widget.load_messages(conversation.messages)
            self.content_stack.setCurrentWidget(self.chat_widget)
            self.input_widget.focus_input()

    def _delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""
        self._storage.delete_conversation(conversation_id)
        if (
            self._current_conversation
            and self._current_conversation.id == conversation_id
        ):
            self._new_chat()

    def _rename_conversation(self, conversation_id: str, new_title: str) -> None:
        """Rename a conversation."""
        self._storage.update_conversation(conversation_id, title=new_title)

    def _show_settings(self) -> None:
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        dialog.settingsChanged.connect(self._apply_settings)
        dialog.exec()

    def _apply_settings(self, settings: dict) -> None:
        """Apply settings changes."""
        self._init_orchestrator()
        if isinstance(settings, dict):
            max_tokens = self._coerce_int(
                settings.get("max_tokens"),
                int(self._runtime_settings.get("max_tokens", 4096)),
                minimum=100,
            )
            temperature = self._coerce_float(
                settings.get("temperature"),
                float(self._runtime_settings.get("temperature", 0.7)),
                minimum=0.0,
            )
            self._runtime_settings["max_tokens"] = max_tokens
            self._runtime_settings["temperature"] = temperature

    def _on_message_sent(self, message: str, settings: dict) -> None:
        """Handle a message being sent."""
        if self._is_processing:
            return

        # Create conversation if needed
        if not self._current_conversation:
            self._current_conversation = self._storage.create_conversation(
                title="New Chat",
                model=settings.get("model"),
            )
            self.sidebar.add_conversation(self._current_conversation)
            self.sidebar.select_conversation(self._current_conversation.id)
            self.content_stack.setCurrentWidget(self.chat_widget)

        # Add user message to UI and storage
        self.chat_widget.add_message("user", message)
        self._storage.add_message(
            self._current_conversation.id,
            "user",
            message,
        )

        # Auto-title if this is the first message
        if len(self._current_conversation.messages) == 0:
            title = message[:50] + "..." if len(message) > 50 else message
            self._storage.update_conversation(
                self._current_conversation.id, title=title
            )

        # Start processing
        self._is_processing = True
        self.input_widget.set_enabled(False)

        # Run query in async
        asyncio.ensure_future(self._process_query(message, settings))

    async def _process_query(self, message: str, settings: dict) -> None:
        """Process the query asynchronously."""
        try:
            if not self._orchestrator:
                self._show_error("AI orchestrator not initialized")
                return

            # Start streaming bubble
            streaming_bubble = self.chat_widget.start_streaming()

            # Prepare query parameters
            model_override = settings.get("model")
            system_prompt = None
            max_tokens = self._coerce_int(
                self._runtime_settings.get("max_tokens"),
                4096,
                minimum=100,
            )
            temperature = self._coerce_float(
                self._runtime_settings.get("temperature"),
                0.7,
                minimum=0.0,
            )

            # Handle special modes
            if settings.get("thinking"):
                system_prompt = (
                    "You are a thoughtful AI assistant. Think through problems "
                    "step by step, showing your reasoning process. Consider multiple "
                    "angles and potential issues before providing your final answer."
                )

            if settings.get("web_search"):
                message = f"[Web search enabled] {message}"

            if settings.get("deep_research"):
                message = f"[Deep research mode] Please conduct thorough research on: {message}"

            if settings.get("image_generation"):
                message = f"[Image generation requested] {message}"

            if settings.get("music_generation"):
                music_params = settings["music_generation"]
                # Actually generate music files
                await self._generate_music_files(music_params, streaming_bubble)
                return  # Don't query LLM for music generation

            # Handle incognito mode
            if settings.get("incognito"):
                self._orchestrator.set_incognito(True)
            else:
                self._orchestrator.set_incognito(False)

            # Status callback for UI updates
            def on_status(status: AgentStatus) -> None:
                stage_name = status.stage.name.lower()
                self.chat_widget.update_status(
                    stage_name,
                    status.model,
                    status.progress,
                    status.message,
                )

            if settings.get("agent_mode") and self._agent_runner is not None:
                from ..agent.runner import AgentRunOptions

                if settings.get("browser_automation"):
                    self.chat_widget.add_message(
                        "system",
                        (
                            "Warning: Browser automation is enabled and may perform real "
                            "actions on live websites."
                        ),
                    )

                response = await self._agent_runner.run(
                    message,
                    options=AgentRunOptions(
                        session_id=self._coerce_str(
                            self._runtime_settings.get("agent_default_session"),
                            "gui-default",
                        ),
                        model_override=model_override,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        enable_web_tools=bool(settings.get("web_search"))
                        and bool(self._runtime_settings.get("agent_web_tools", True)),
                        enable_mcp=bool(
                            self._runtime_settings.get("agent_mcp_enabled", True)
                        ),
                        enable_skills=bool(
                            self._runtime_settings.get("agent_skills_enabled", True)
                        ),
                        enable_browser_automation=bool(
                            settings.get("browser_automation")
                        )
                        and bool(
                            self._runtime_settings.get("agent_browser_enabled", False)
                        ),
                        incognito=settings.get("incognito", False),
                        system_prompt=system_prompt,
                    ),
                    status_callback=on_status,
                )
            else:
                # Query the orchestrator
                response = await self._orchestrator.query(
                    message,
                    model_override=model_override,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    status_callback=on_status,
                )

            # Update the streaming bubble with final content
            streaming_bubble.content = response.content
            self.chat_widget.finish_streaming()

            # Save to storage
            if self._current_conversation:
                self._storage.add_message(
                    self._current_conversation.id,
                    "assistant",
                    response.content,
                    metadata={
                        "model": response.model,
                        "provider": response.provider,
                        "usage": response.usage,
                        "latency_ms": response.latency_ms,
                    },
                )

        except Exception as e:
            self.chat_widget.finish_streaming()
            self.chat_widget.add_message(
                "assistant",
                f"Error: {str(e)}\n\nPlease check your API keys and try again.",
            )

        finally:
            self._is_processing = False
            self.input_widget.set_enabled(True)
            self.input_widget.focus_input()

    async def _generate_music_files(self, params: dict, streaming_bubble) -> None:
        """Generate actual music files from parameters."""
        try:
            from ..music import MusicParameters, format_music_result, generate_music

            # Parse parameters
            music_params = MusicParameters.from_dict(params)

            # Show generating message
            streaming_bubble.content = "Generating music...\n\n"
            streaming_bubble.content += (
                f"Key: {music_params.key} {music_params.scale}\n"
            )
            streaming_bubble.content += f"BPM: {music_params.bpm}\n"
            streaming_bubble.content += f"Genre: {music_params.genre}\n"

            # Generate music
            result = await generate_music(music_params)

            # Format result
            streaming_bubble.content = format_music_result(result)
            self.chat_widget.finish_streaming()

            # Save to storage
            if self._current_conversation:
                self._storage.add_message(
                    self._current_conversation.id,
                    "assistant",
                    streaming_bubble.content,
                    metadata={
                        "type": "music_generation",
                        "files": result.get("files", []),
                    },
                )

        except ImportError as e:
            streaming_bubble.content = (
                "**Music generation requires additional packages.**\n\n"
                "Install with:\n"
                "```bash\n"
                "pip install midiutil\n"
                "```\n\n"
                f"Error: {e}"
            )
            self.chat_widget.finish_streaming()

        except Exception as e:
            streaming_bubble.content = f"**Error generating music:**\n\n{str(e)}"
            self.chat_widget.finish_streaming()

        finally:
            self._is_processing = False
            self.input_widget.set_enabled(True)
            self.input_widget.focus_input()

    def _build_music_prompt(self, params: dict) -> str:
        """Build a music generation prompt from parameters."""
        parts = []

        if params.get("prompt"):
            parts.append(f"Description: {params['prompt']}")
        if params.get("key"):
            parts.append(f"Key signature: {params['key']}")
        if params.get("genre"):
            parts.append(f"Genre: {params['genre']}")
        if params.get("artist"):
            parts.append(f"Similar to: {params['artist']}")
        if params.get("mood"):
            parts.append(f"Mood: {params['mood']}")
        if params.get("energy") is not None:
            parts.append(f"Energy level: {int(params['energy'] * 100)}%")
        if params.get("bpm"):
            parts.append(f"BPM: {params['bpm']}")
        if params.get("duration"):
            parts.append(f"Duration: {params['duration']} seconds")
        if params.get("format"):
            parts.append(f"Output format: {params['format']}")

        return "\n".join(parts)

    def _show_error(self, message: str) -> None:
        """Show an error message."""
        self.chat_widget.add_message("assistant", f"Error: {message}")
        self._is_processing = False
        self.input_widget.set_enabled(True)

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Clean up async tasks
        event.accept()
