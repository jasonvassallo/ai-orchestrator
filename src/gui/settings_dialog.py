"""
Settings Dialog
================

Application settings and preferences.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..credentials import CONFIG_DIR
from .styles import COLORS


class SettingsDialog(QDialog):
    """Application settings dialog."""

    settingsChanged = Signal(dict)  # Emitted when settings are saved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 600)
        self._setup_ui()
        self._load_settings()

    @staticmethod
    def _available_model_names(task_types: set[str] | None = None) -> list[str]:
        """Build display model names from the shared ModelRegistry."""
        from ..orchestrator import LOCAL_PROVIDERS, ModelRegistry

        entries: list[tuple[int, str, str]] = []
        for _, model in ModelRegistry.MODELS.items():
            if task_types is not None and not any(
                task.name in task_types for task in model.task_types
            ):
                continue
            local_rank = 0 if model.provider in LOCAL_PROVIDERS else 1
            entries.append((local_rank, model.provider, model.name))

        entries.sort(key=lambda item: (item[0], item[1], item[2].lower()))
        deduped: list[str] = []
        seen: set[str] = set()
        for _, _, name in entries:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_general_tab(), "General")
        tabs.addTab(self._create_models_tab(), "Models")
        tabs.addTab(self._create_api_keys_tab(), "API Keys")
        tabs.addTab(self._create_appearance_tab(), "Appearance")
        tabs.addTab(self._create_agent_tab(), "Agent")
        tabs.addTab(self._create_advanced_tab(), "Advanced")
        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._save_settings)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["button_primary"]};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS["button_primary_hover"]};
            }}
        """)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Default behavior group
        behavior_group = QGroupBox("Default Behavior")
        behavior_layout = QFormLayout(behavior_group)

        self.auto_select_model = QCheckBox("Automatically select best model for task")
        self.auto_select_model.setChecked(True)
        behavior_layout.addRow(self.auto_select_model)

        self.prefer_local = QCheckBox("Prefer local models (MLX) when available")
        behavior_layout.addRow(self.prefer_local)

        self.cost_optimize = QCheckBox(
            "Optimize for cost (use cheaper models when appropriate)"
        )
        behavior_layout.addRow(self.cost_optimize)

        self.save_history = QCheckBox("Save conversation history")
        self.save_history.setChecked(True)
        behavior_layout.addRow(self.save_history)

        layout.addWidget(behavior_group)

        # Generation defaults
        gen_group = QGroupBox("Generation Defaults")
        gen_layout = QFormLayout(gen_group)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(100, 32000)
        self.max_tokens.setValue(4096)
        gen_layout.addRow("Max Tokens:", self.max_tokens)

        self.temperature = QSlider(Qt.Orientation.Horizontal)
        self.temperature.setRange(0, 100)
        self.temperature.setValue(70)
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temperature)
        self.temp_label = QLabel("0.7")
        self.temperature.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v / 100:.1f}")
        )
        temp_layout.addWidget(self.temp_label)
        gen_layout.addRow("Temperature:", temp_layout)

        layout.addWidget(gen_group)

        layout.addStretch()
        return widget

    def _create_models_tab(self) -> QWidget:
        """Create models settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Default model selection
        default_group = QGroupBox("Default Model")
        default_layout = QFormLayout(default_group)

        self.default_model = QComboBox()
        self.default_model.addItems(
            ["Auto (Best for Task)", *self._available_model_names()]
        )
        default_layout.addRow("Default Model:", self.default_model)

        layout.addWidget(default_group)

        # Task routing
        routing_group = QGroupBox("Task Routing Preferences")
        routing_layout = QFormLayout(routing_group)

        self.code_model = QComboBox()
        self.code_model.addItems(
            ["Auto", *self._available_model_names({"CODE_GENERATION"})]
        )
        routing_layout.addRow("Coding Tasks:", self.code_model)

        self.reasoning_model = QComboBox()
        self.reasoning_model.addItems(
            [
                "Auto",
                *self._available_model_names({"REASONING", "DEEP_REASONING", "MATH"}),
            ]
        )
        routing_layout.addRow("Reasoning Tasks:", self.reasoning_model)

        self.creative_model = QComboBox()
        self.creative_model.addItems(
            ["Auto", *self._available_model_names({"CREATIVE_WRITING"})]
        )
        routing_layout.addRow("Creative Tasks:", self.creative_model)

        layout.addWidget(routing_group)

        layout.addStretch()
        return widget

    def _create_api_keys_tab(self) -> QWidget:
        """Create API keys settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        info_label = QLabel(
            "API keys are stored securely in your macOS Keychain.\n"
            "You can also set them via environment variables."
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(info_label)

        # API key status
        keys_group = QGroupBox("API Key Status")
        keys_layout = QVBoxLayout(keys_group)

        # Check key status
        try:
            from ..credentials import get_api_key

            providers = [
                ("OpenAI", "openai"),
                ("Anthropic", "anthropic"),
                ("Google", "google"),
                ("Perplexity", "perplexity"),
                ("Groq", "groq"),
                ("Mistral", "mistral"),
                ("xAI", "xai"),
                ("DeepSeek", "deepseek"),
            ]

            for name, provider_id in providers:
                row = QHBoxLayout()
                label = QLabel(name)
                label.setMinimumWidth(100)
                row.addWidget(label)

                key = get_api_key(provider_id)
                if key:
                    status = QLabel("✅ Configured")
                    status.setStyleSheet(f"color: {COLORS['success']};")
                    masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
                    masked_label = QLabel(masked)
                    masked_label.setStyleSheet(f"color: {COLORS['text_muted']};")
                    row.addWidget(status)
                    row.addWidget(masked_label)
                else:
                    status = QLabel("❌ Not configured")
                    status.setStyleSheet(f"color: {COLORS['text_muted']};")
                    row.addWidget(status)

                row.addStretch()
                keys_layout.addLayout(row)

        except Exception as e:
            error_label = QLabel(f"Error loading keys: {e}")
            keys_layout.addWidget(error_label)

        layout.addWidget(keys_group)

        # Configure button
        configure_btn = QPushButton("Configure API Keys in Terminal")
        configure_btn.clicked.connect(self._open_configure)
        layout.addWidget(configure_btn)

        layout.addStretch()
        return widget

    def _create_appearance_tab(self) -> QWidget:
        """Create appearance settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Theme (coming soon)
        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout(theme_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(
            ["Dark (Default)", "Light (Coming Soon)", "System (Coming Soon)"]
        )
        self.theme_combo.setCurrentIndex(0)
        theme_layout.addRow("Theme:", self.theme_combo)

        layout.addWidget(theme_group)

        # Font settings
        font_group = QGroupBox("Fonts")
        font_layout = QFormLayout(font_group)

        self.font_size = QSpinBox()
        self.font_size.setRange(10, 24)
        self.font_size.setValue(14)
        font_layout.addRow("Font Size:", self.font_size)

        self.code_font_size = QSpinBox()
        self.code_font_size.setRange(10, 24)
        self.code_font_size.setValue(13)
        font_layout.addRow("Code Font Size:", self.code_font_size)

        layout.addWidget(font_group)

        # Chat display
        chat_group = QGroupBox("Chat Display")
        chat_layout = QFormLayout(chat_group)

        self.show_timestamps = QCheckBox("Show message timestamps")
        chat_layout.addRow(self.show_timestamps)

        self.show_model_info = QCheckBox("Show model name in responses")
        self.show_model_info.setChecked(True)
        chat_layout.addRow(self.show_model_info)

        self.compact_mode = QCheckBox("Compact message display")
        chat_layout.addRow(self.compact_mode)

        layout.addWidget(chat_group)

        layout.addStretch()
        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Storage
        storage_group = QGroupBox("Storage")
        storage_layout = QFormLayout(storage_group)

        db_path = QLabel("~/.ai_orchestrator/conversations.db")
        db_path.setStyleSheet(f"color: {COLORS['text_muted']};")
        storage_layout.addRow("Database:", db_path)

        clear_btn = QPushButton("Clear Conversation History")
        clear_btn.clicked.connect(self._confirm_clear_history)
        storage_layout.addRow(clear_btn)

        layout.addWidget(storage_group)

        # Music generation
        music_group = QGroupBox("Music Generation")
        music_layout = QFormLayout(music_group)

        music_output = QLabel("~/Music/AI Orchestrator/")
        music_output.setStyleSheet(f"color: {COLORS['text_muted']};")
        music_layout.addRow("Output Folder:", music_output)

        # Check MIDI availability
        try:
            from midiutil import MIDIFile  # noqa: F401

            midi_status = QLabel("✅ MIDI generation available")
            midi_status.setStyleSheet(f"color: {COLORS['success']};")
        except ImportError:
            midi_status = QLabel("❌ Install midiutil for MIDI: pip install midiutil")
            midi_status.setStyleSheet(f"color: {COLORS['warning']};")
        music_layout.addRow("MIDI:", midi_status)

        # Check MusicGen availability
        try:
            import torch  # noqa: F401

            audio_status = QLabel("✅ Audio generation available (torch installed)")
            audio_status.setStyleSheet(f"color: {COLORS['success']};")
        except ImportError:
            audio_status = QLabel("⚠️ Install torch for audio: pip install torch")
            audio_status.setStyleSheet(f"color: {COLORS['warning']};")
        music_layout.addRow("Audio:", audio_status)

        layout.addWidget(music_group)

        # Debug
        debug_group = QGroupBox("Debug")
        debug_layout = QFormLayout(debug_group)

        self.verbose_mode = QCheckBox("Enable verbose logging")
        debug_layout.addRow(self.verbose_mode)

        self.show_tokens = QCheckBox("Show token usage in responses")
        debug_layout.addRow(self.show_tokens)

        layout.addWidget(debug_group)

        layout.addStretch()
        return widget

    def _create_agent_tab(self) -> QWidget:
        """Create dedicated agent settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Core agent options
        core_group = QGroupBox("Agent Core")
        core_layout = QFormLayout(core_group)

        self.agent_enabled_default = QCheckBox("Enable Agent mode by default")
        core_layout.addRow(self.agent_enabled_default)

        self.agent_default_session = QLineEdit()
        self.agent_default_session.setPlaceholderText("default")
        core_layout.addRow("Default Session ID:", self.agent_default_session)

        self.agent_profile = QComboBox()
        self.agent_profile.addItems(["fast", "balanced", "deep"])
        self.agent_profile.setCurrentText("balanced")
        core_layout.addRow("Execution Profile:", self.agent_profile)

        self.agent_default_model = QLineEdit()
        self.agent_default_model.setPlaceholderText("mlx-qwen3-coder-30b")
        core_layout.addRow("Default Agent Model:", self.agent_default_model)

        self.agent_web_tools = QCheckBox("Enable web_search/web_fetch tools")
        self.agent_web_tools.setChecked(True)
        core_layout.addRow(self.agent_web_tools)

        self.agent_mcp_enabled = QCheckBox("Enable MCP discovery and tools")
        self.agent_mcp_enabled.setChecked(True)
        core_layout.addRow(self.agent_mcp_enabled)

        self.agent_skills_enabled = QCheckBox("Enable skills discovery")
        self.agent_skills_enabled.setChecked(True)
        core_layout.addRow(self.agent_skills_enabled)

        self.agent_browser_enabled = QCheckBox("Enable browser automation (dangerous)")
        core_layout.addRow(self.agent_browser_enabled)

        layout.addWidget(core_group)

        # Compatibility sources
        compatibility_group = QGroupBox("Compatibility Sources")
        compatibility_layout = QFormLayout(compatibility_group)

        self.agent_source_codex = QCheckBox("Codex")
        self.agent_source_codex.setChecked(True)
        compatibility_layout.addRow(self.agent_source_codex)

        self.agent_source_claude = QCheckBox("Claude")
        self.agent_source_claude.setChecked(True)
        compatibility_layout.addRow(self.agent_source_claude)

        self.agent_source_gemini = QCheckBox("Gemini")
        self.agent_source_gemini.setChecked(True)
        compatibility_layout.addRow(self.agent_source_gemini)

        layout.addWidget(compatibility_group)

        # Limits and timeouts
        limits_group = QGroupBox("Limits & Timeouts")
        limits_layout = QFormLayout(limits_group)

        self.agent_max_steps = QSpinBox()
        self.agent_max_steps.setRange(1, 100)
        self.agent_max_steps.setValue(10)
        limits_layout.addRow("Max Steps:", self.agent_max_steps)

        self.agent_model_timeout = QSpinBox()
        self.agent_model_timeout.setRange(1, 600)
        self.agent_model_timeout.setValue(90)
        limits_layout.addRow("Model Timeout (s):", self.agent_model_timeout)

        self.agent_tool_timeout = QSpinBox()
        self.agent_tool_timeout.setRange(1, 600)
        self.agent_tool_timeout.setValue(40)
        limits_layout.addRow("Tool Timeout (s):", self.agent_tool_timeout)

        self.agent_shell_timeout = QSpinBox()
        self.agent_shell_timeout.setRange(1, 600)
        self.agent_shell_timeout.setValue(30)
        limits_layout.addRow("Shell Timeout (s):", self.agent_shell_timeout)

        self.agent_web_search_timeout = QSpinBox()
        self.agent_web_search_timeout.setRange(1, 600)
        self.agent_web_search_timeout.setValue(20)
        limits_layout.addRow("Web Search Timeout (s):", self.agent_web_search_timeout)

        self.agent_web_fetch_timeout = QSpinBox()
        self.agent_web_fetch_timeout.setRange(1, 600)
        self.agent_web_fetch_timeout.setValue(25)
        limits_layout.addRow("Web Fetch Timeout (s):", self.agent_web_fetch_timeout)

        self.agent_max_prompt_chars = QSpinBox()
        self.agent_max_prompt_chars.setRange(2000, 1000000)
        self.agent_max_prompt_chars.setValue(24000)
        limits_layout.addRow("Max Prompt Chars:", self.agent_max_prompt_chars)

        self.agent_max_tool_output_chars = QSpinBox()
        self.agent_max_tool_output_chars.setRange(500, 1000000)
        self.agent_max_tool_output_chars.setValue(10000)
        limits_layout.addRow("Max Tool Output Chars:", self.agent_max_tool_output_chars)

        self.agent_max_shell_output_chars = QSpinBox()
        self.agent_max_shell_output_chars.setRange(200, 1000000)
        self.agent_max_shell_output_chars.setValue(8000)
        limits_layout.addRow(
            "Max Shell Output Chars:", self.agent_max_shell_output_chars
        )

        self.agent_max_fetched_chars = QSpinBox()
        self.agent_max_fetched_chars.setRange(500, 1000000)
        self.agent_max_fetched_chars.setValue(12000)
        limits_layout.addRow("Max Fetched Chars:", self.agent_max_fetched_chars)

        self.agent_max_web_results = QSpinBox()
        self.agent_max_web_results.setRange(1, 100)
        self.agent_max_web_results.setValue(8)
        limits_layout.addRow("Max Web Results:", self.agent_max_web_results)

        self.agent_max_memory_context_chars = QSpinBox()
        self.agent_max_memory_context_chars.setRange(500, 1000000)
        self.agent_max_memory_context_chars.setValue(9000)
        limits_layout.addRow(
            "Max Memory Context Chars:", self.agent_max_memory_context_chars
        )

        self.agent_max_history_messages = QSpinBox()
        self.agent_max_history_messages.setRange(1, 500)
        self.agent_max_history_messages.setValue(24)
        limits_layout.addRow("Max History Messages:", self.agent_max_history_messages)

        layout.addWidget(limits_group)
        layout.addStretch()
        return widget

    def _load_settings(self) -> None:
        """Load settings from storage."""
        config_path = CONFIG_DIR / "config.json"
        if not config_path.exists():
            return

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception:
            return

        if not isinstance(loaded, dict):
            return

        defaults = loaded.get("defaults", {})
        if isinstance(defaults, dict):
            self.prefer_local.setChecked(bool(defaults.get("preferLocal", False)))
            self.cost_optimize.setChecked(bool(defaults.get("costOptimize", False)))
            self.max_tokens.setValue(int(defaults.get("maxTokens", 4096)))
            temp = float(defaults.get("temperature", 0.7))
            self.temperature.setValue(max(0, min(100, int(temp * 100))))

        logging_cfg = loaded.get("logging", {})
        if isinstance(logging_cfg, dict):
            level = logging_cfg.get("level", "INFO")
            self.verbose_mode.setChecked(str(level).upper() == "DEBUG")

        agent_cfg = loaded.get("agent", {})
        if isinstance(agent_cfg, dict):
            self.agent_enabled_default.setChecked(
                bool(agent_cfg.get("enabledByDefault", False))
            )
            self.agent_default_session.setText(
                str(agent_cfg.get("defaultSessionId", "default"))
            )
            self.agent_default_model.setText(
                str(agent_cfg.get("defaultModel", "mlx-qwen3-coder-30b"))
            )

            profile = str(agent_cfg.get("profile", "balanced"))
            profile_index = self.agent_profile.findText(profile)
            if profile_index >= 0:
                self.agent_profile.setCurrentIndex(profile_index)

            self.agent_web_tools.setChecked(bool(agent_cfg.get("enableWebTools", True)))
            self.agent_mcp_enabled.setChecked(bool(agent_cfg.get("enableMcp", True)))
            self.agent_skills_enabled.setChecked(
                bool(agent_cfg.get("enableSkills", True))
            )
            self.agent_browser_enabled.setChecked(
                bool(agent_cfg.get("enableBrowserAutomation", False))
            )

            compatibility = agent_cfg.get("compatibility", {})
            if isinstance(compatibility, dict):
                skill_sources = compatibility.get(
                    "skillSources", ["codex", "claude", "gemini"]
                )
                mcp_sources = compatibility.get(
                    "mcpSources", ["codex", "claude", "gemini"]
                )
                all_sources = set()
                if isinstance(skill_sources, list):
                    all_sources.update(
                        item for item in skill_sources if isinstance(item, str)
                    )
                if isinstance(mcp_sources, list):
                    all_sources.update(
                        item for item in mcp_sources if isinstance(item, str)
                    )
                self.agent_source_codex.setChecked("codex" in all_sources)
                self.agent_source_claude.setChecked("claude" in all_sources)
                self.agent_source_gemini.setChecked("gemini" in all_sources)

            limits = agent_cfg.get("limits", {})
            if isinstance(limits, dict):
                self.agent_max_steps.setValue(int(limits.get("maxSteps", 10)))
                self.agent_model_timeout.setValue(
                    int(limits.get("modelTimeoutSeconds", 90))
                )
                self.agent_tool_timeout.setValue(
                    int(limits.get("toolTimeoutSeconds", 40))
                )
                self.agent_shell_timeout.setValue(
                    int(limits.get("shellTimeoutSeconds", 30))
                )
                self.agent_web_search_timeout.setValue(
                    int(limits.get("webSearchTimeoutSeconds", 20))
                )
                self.agent_web_fetch_timeout.setValue(
                    int(limits.get("webFetchTimeoutSeconds", 25))
                )
                self.agent_max_prompt_chars.setValue(
                    int(limits.get("maxPromptChars", 24000))
                )
                self.agent_max_tool_output_chars.setValue(
                    int(limits.get("maxToolOutputChars", 10000))
                )
                self.agent_max_shell_output_chars.setValue(
                    int(limits.get("maxShellOutputChars", 8000))
                )
                self.agent_max_fetched_chars.setValue(
                    int(limits.get("maxFetchedChars", 12000))
                )
                self.agent_max_web_results.setValue(int(limits.get("maxWebResults", 8)))
                self.agent_max_memory_context_chars.setValue(
                    int(limits.get("maxMemoryContextChars", 9000))
                )
                self.agent_max_history_messages.setValue(
                    int(limits.get("maxHistoryMessages", 24))
                )

    def _save_settings(self) -> None:
        """Save settings and close dialog."""
        selected_sources = []
        if self.agent_source_codex.isChecked():
            selected_sources.append("codex")
        if self.agent_source_claude.isChecked():
            selected_sources.append("claude")
        if self.agent_source_gemini.isChecked():
            selected_sources.append("gemini")
        if not selected_sources:
            selected_sources = ["codex"]

        settings = {
            "auto_select_model": self.auto_select_model.isChecked(),
            "prefer_local": self.prefer_local.isChecked(),
            "cost_optimize": self.cost_optimize.isChecked(),
            "save_history": self.save_history.isChecked(),
            "max_tokens": self.max_tokens.value(),
            "temperature": self.temperature.value() / 100,
            "default_model": self.default_model.currentText(),
            "font_size": self.font_size.value(),
            "code_font_size": self.code_font_size.value(),
            "show_timestamps": self.show_timestamps.isChecked(),
            "show_model_info": self.show_model_info.isChecked(),
            "compact_mode": self.compact_mode.isChecked(),
            "verbose_mode": self.verbose_mode.isChecked(),
            "show_tokens": self.show_tokens.isChecked(),
            "agent_enabled_default": self.agent_enabled_default.isChecked(),
            "agent_default_session": self.agent_default_session.text().strip()
            or "default",
            "agent_profile": self.agent_profile.currentText(),
            "agent_default_model": self.agent_default_model.text().strip()
            or "mlx-qwen3-coder-30b",
            "agent_web_tools": self.agent_web_tools.isChecked(),
            "agent_mcp_enabled": self.agent_mcp_enabled.isChecked(),
            "agent_skills_enabled": self.agent_skills_enabled.isChecked(),
            "agent_browser_enabled": self.agent_browser_enabled.isChecked(),
            "agent_sources": selected_sources,
            "agent_limits": {
                "maxSteps": self.agent_max_steps.value(),
                "modelTimeoutSeconds": self.agent_model_timeout.value(),
                "toolTimeoutSeconds": self.agent_tool_timeout.value(),
                "shellTimeoutSeconds": self.agent_shell_timeout.value(),
                "webSearchTimeoutSeconds": self.agent_web_search_timeout.value(),
                "webFetchTimeoutSeconds": self.agent_web_fetch_timeout.value(),
                "maxPromptChars": self.agent_max_prompt_chars.value(),
                "maxToolOutputChars": self.agent_max_tool_output_chars.value(),
                "maxShellOutputChars": self.agent_max_shell_output_chars.value(),
                "maxFetchedChars": self.agent_max_fetched_chars.value(),
                "maxWebResults": self.agent_max_web_results.value(),
                "maxMemoryContextChars": self.agent_max_memory_context_chars.value(),
                "maxHistoryMessages": self.agent_max_history_messages.value(),
            },
        }

        config_path = CONFIG_DIR / "config.json"
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        loaded: dict = {}
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    maybe_loaded = json.load(handle)
                if isinstance(maybe_loaded, dict):
                    loaded = maybe_loaded
            except Exception:
                loaded = {}

        defaults = loaded.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
        defaults["preferLocal"] = settings["prefer_local"]
        defaults["costOptimize"] = settings["cost_optimize"]
        defaults["maxTokens"] = settings["max_tokens"]
        defaults["temperature"] = settings["temperature"]
        loaded["defaults"] = defaults

        logging_cfg = loaded.get("logging", {})
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
        logging_cfg["level"] = "DEBUG" if settings["verbose_mode"] else "INFO"
        loaded["logging"] = logging_cfg

        agent_cfg = loaded.get("agent", {})
        if not isinstance(agent_cfg, dict):
            agent_cfg = {}
        agent_cfg["enabledByDefault"] = settings["agent_enabled_default"]
        agent_cfg["defaultSessionId"] = settings["agent_default_session"]
        agent_cfg["profile"] = settings["agent_profile"]
        agent_cfg["defaultModel"] = settings["agent_default_model"]
        agent_cfg["enableWebTools"] = settings["agent_web_tools"]
        agent_cfg["enableMcp"] = settings["agent_mcp_enabled"]
        agent_cfg["enableSkills"] = settings["agent_skills_enabled"]
        agent_cfg["enableBrowserAutomation"] = settings["agent_browser_enabled"]
        agent_cfg["compatibility"] = {
            "skillSources": settings["agent_sources"],
            "mcpSources": settings["agent_sources"],
        }
        agent_cfg["limits"] = settings["agent_limits"]
        loaded["agent"] = agent_cfg

        if "version" not in loaded:
            loaded["version"] = "2.0.0"

        try:
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(loaded, handle, indent=2)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Settings Error",
                f"Failed to save settings: {exc}",
            )
            return

        self.settingsChanged.emit(settings)

        QMessageBox.information(
            self,
            "Settings Saved",
            "Your settings have been saved.",
        )
        self.accept()

    def _open_configure(self) -> None:
        """Open terminal to configure API keys."""
        try:
            script = """
            tell application "Terminal"
                do script "cd && python3 -m src.credentials"
                activate
            end tell
            """
            osascript_path = shutil.which("osascript")
            if osascript_path is None:
                raise FileNotFoundError("Could not locate 'osascript'")
            subprocess.Popen([osascript_path, "-e", script])  # noqa: S603  # nosec B603
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open Terminal: {e}\n\n"
                "Run manually: python3 -m src.credentials",
            )

    def _confirm_clear_history(self) -> None:
        """Confirm and clear conversation history."""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to delete ALL conversation history?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                import os

                from ..storage import get_storage_path

                db_path = get_storage_path() / "conversations.db"
                if db_path.exists():
                    os.remove(db_path)
                    QMessageBox.information(
                        self,
                        "History Cleared",
                        "All conversation history has been deleted.\n\n"
                        "Restart the app to see the changes.",
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Could not clear history: {e}",
                )
