"""
Settings Dialog
================

Application settings and preferences.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_general_tab(), "General")
        tabs.addTab(self._create_models_tab(), "Models")
        tabs.addTab(self._create_api_keys_tab(), "API Keys")
        tabs.addTab(self._create_appearance_tab(), "Appearance")
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
            [
                "Auto (Best for Task)",
                "Claude Opus 4.5",
                "Claude Sonnet 4.5",
                "Claude Haiku 4.5",
                "GPT-4o",
                "GPT-4o Mini",
                "o1",
                "Gemini 2.0 Flash",
                "Gemini 1.5 Pro",
                "DeepSeek Chat",
                "Llama 3.2 (Local)",
            ]
        )
        default_layout.addRow("Default Model:", self.default_model)

        layout.addWidget(default_group)

        # Task routing
        routing_group = QGroupBox("Task Routing Preferences")
        routing_layout = QFormLayout(routing_group)

        self.code_model = QComboBox()
        self.code_model.addItems(
            ["Auto", "Claude Sonnet 4.5", "GPT-4o", "DeepSeek Chat", "Codestral"]
        )
        routing_layout.addRow("Coding Tasks:", self.code_model)

        self.reasoning_model = QComboBox()
        self.reasoning_model.addItems(
            ["Auto", "Claude Opus 4.5", "o1", "DeepSeek Reasoner"]
        )
        routing_layout.addRow("Reasoning Tasks:", self.reasoning_model)

        self.creative_model = QComboBox()
        self.creative_model.addItems(
            ["Auto", "Claude Opus 4.5", "GPT-4o", "Gemini 1.5 Pro"]
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

    def _load_settings(self) -> None:
        """Load settings from storage."""
        # TODO: Load from config file or database
        pass

    def _save_settings(self) -> None:
        """Save settings and close dialog."""
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
        }

        self.settingsChanged.emit(settings)

        QMessageBox.information(
            self,
            "Settings Saved",
            "Your settings have been saved.",
        )
        self.accept()

    def _open_configure(self) -> None:
        """Open terminal to configure API keys."""
        import subprocess

        try:
            script = """
            tell application "Terminal"
                do script "cd && python3 -m src.credentials"
                activate
            end tell
            """
            subprocess.Popen(["osascript", "-e", script])  # noqa: S603
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
