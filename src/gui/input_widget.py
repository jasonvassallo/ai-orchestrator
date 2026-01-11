"""
Input Widget
============

Multi-line input area with model selector and feature toggles.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .styles import COLORS

# Import MusicGen models for the dropdown
try:
    from ..music import MUSICGEN_MODELS
except ImportError:
    MUSICGEN_MODELS = {
        "musicgen-small": {"description": "Fast, 300M params (recommended)"},
    }


class ExpandingTextEdit(QTextEdit):
    """Text edit that expands with content up to a max height."""

    returnPressed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText("Message AI Orchestrator...")
        self.setAcceptRichText(False)
        self.setMinimumHeight(50)
        self.setMaximumHeight(200)
        self.textChanged.connect(self._adjust_height)
        self._adjust_height()

    def _adjust_height(self) -> None:
        """Adjust height based on content."""
        doc = self.document()
        height = int(doc.size().height()) + 20
        height = max(50, min(height, 200))
        self.setFixedHeight(height)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press, emit signal on Cmd+Enter."""
        if event.key() == Qt.Key.Key_Return:
            if event.modifiers() == Qt.KeyboardModifier.MetaModifier:
                self.returnPressed.emit()
                return
        super().keyPressEvent(event)


class ToggleButton(QPushButton):
    """A toggle button with on/off state."""

    def __init__(self, text: str, tooltip: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self.setObjectName("toggleButton")
        self._update_style()
        self.toggled.connect(lambda: self._update_style())

    def _update_style(self) -> None:
        """Update style based on checked state."""
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["toggle_on"]};
                    color: white;
                    border-radius: 4px;
                    padding: 6px 10px;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["button_primary_hover"]};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["toggle_off"]};
                    color: {COLORS["text_secondary"]};
                    border-radius: 4px;
                    padding: 6px 10px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["bg_hover"]};
                    color: {COLORS["text_primary"]};
                }}
            """)


class MusicGenerationDialog(QDialog):
    """Dialog for music generation parameters."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Music Generation")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Grid for parameters
        grid = QGridLayout()
        grid.setSpacing(12)

        # Prompt
        grid.addWidget(QLabel("Prompt:"), 0, 0)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Describe the music you want...")
        grid.addWidget(self.prompt_edit, 0, 1)

        # Key signature
        grid.addWidget(QLabel("Key:"), 1, 0)
        self.key_combo = QComboBox()
        self.key_combo.addItems(
            [
                "C Major",
                "C Minor",
                "D Major",
                "D Minor",
                "E Major",
                "E Minor",
                "F Major",
                "F Minor",
                "G Major",
                "G Minor",
                "A Major",
                "A Minor",
                "B Major",
                "B Minor",
                "Auto",
            ]
        )
        self.key_combo.setCurrentText("Auto")
        grid.addWidget(self.key_combo, 1, 1)

        # Genre
        grid.addWidget(QLabel("Genre:"), 2, 0)
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(
            [
                "Auto",
                "Classical",
                "Electronic",
                "Jazz",
                "Rock",
                "Pop",
                "Hip Hop",
                "Ambient",
                "Orchestral",
                "Folk",
                "Blues",
                "Country",
                "R&B",
                "Metal",
                "Indie",
            ]
        )
        grid.addWidget(self.genre_combo, 2, 1)

        # Sounds like artist
        grid.addWidget(QLabel("Sounds Like:"), 3, 0)
        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("Artist name (optional)")
        grid.addWidget(self.artist_edit, 3, 1)

        # Mood
        grid.addWidget(QLabel("Mood:"), 4, 0)
        self.mood_combo = QComboBox()
        self.mood_combo.addItems(
            [
                "Auto",
                "Happy",
                "Sad",
                "Energetic",
                "Calm",
                "Dark",
                "Uplifting",
                "Mysterious",
                "Romantic",
                "Aggressive",
                "Peaceful",
                "Epic",
                "Nostalgic",
            ]
        )
        grid.addWidget(self.mood_combo, 4, 1)

        # Energy slider
        grid.addWidget(QLabel("Energy:"), 5, 0)
        energy_layout = QHBoxLayout()
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setRange(0, 100)
        self.energy_slider.setValue(50)
        self.energy_label = QLabel("50%")
        self.energy_slider.valueChanged.connect(
            lambda v: self.energy_label.setText(f"{v}%")
        )
        energy_layout.addWidget(self.energy_slider)
        energy_layout.addWidget(self.energy_label)
        grid.addLayout(energy_layout, 5, 1)

        # BPM
        grid.addWidget(QLabel("BPM:"), 6, 0)
        self.bpm_edit = QLineEdit()
        self.bpm_edit.setPlaceholderText("120 (or Auto)")
        grid.addWidget(self.bpm_edit, 6, 1)

        # Duration
        grid.addWidget(QLabel("Duration:"), 7, 0)
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(
            [
                "10 seconds",
                "15 seconds",
                "30 seconds",
                "60 seconds",
                "90 seconds",
                "120 seconds",
            ]
        )
        self.duration_combo.setCurrentText("30 seconds")
        grid.addWidget(self.duration_combo, 7, 1)

        # Output format
        grid.addWidget(QLabel("Format:"), 8, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV", "MIDI", "All"])
        grid.addWidget(self.format_combo, 8, 1)

        # MusicGen model selector
        grid.addWidget(QLabel("AI Model:"), 9, 0)
        self.model_combo = QComboBox()
        for key, info in MUSICGEN_MODELS.items():
            # Display format: "musicgen-small - Fast, 300M params"
            self.model_combo.addItem(f"{key} - {info['description']}", key)
        self.model_combo.setCurrentIndex(0)  # Default to first (small)
        self.model_combo.setToolTip(
            "Select the MusicGen model for audio generation.\n"
            "Larger models produce higher quality but are slower."
        )
        grid.addWidget(self.model_combo, 9, 1)

        layout.addLayout(grid)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_parameters(self) -> dict:
        """Get the music generation parameters."""
        bpm = self.bpm_edit.text().strip()
        bpm_val: int | None = None
        if bpm:
            try:
                bpm_val = int(bpm)
            except ValueError:
                pass

        duration_text = self.duration_combo.currentText()
        duration = int(duration_text.split()[0])

        return {
            "prompt": self.prompt_edit.text(),
            "key": self.key_combo.currentText()
            if self.key_combo.currentText() != "Auto"
            else None,
            "genre": self.genre_combo.currentText()
            if self.genre_combo.currentText() != "Auto"
            else None,
            "artist": self.artist_edit.text().strip() or None,
            "mood": self.mood_combo.currentText()
            if self.mood_combo.currentText() != "Auto"
            else None,
            "energy": self.energy_slider.value() / 100.0,
            "bpm": bpm_val,
            "duration": duration,
            "format": self.format_combo.currentText().lower(),
            "musicgen_model": self.model_combo.currentData(),
        }


class InputWidget(QFrame):
    """Input area with model selector and feature toggles."""

    messageSent = Signal(str, dict)  # message, settings

    # Available models (subset for display)
    MODELS = [
        ("Auto (Best for Task)", None),
        ("Claude Opus 4.5", "claude-opus-4.5"),
        ("Claude Sonnet 4.5", "claude-sonnet-4.5"),
        ("GPT-5 (Preview)", "gpt-5-preview"),
        ("GPT-4.5 (Preview)", "gpt-4.5-preview"),
        ("GPT-4o", "gpt-4o"),
        ("o1 (Reasoning)", "o1"),
        ("Gemini 3.0 Pro (Preview)", "gemini-3-pro"),
        ("Gemini 3.0 Flash (Preview)", "gemini-3-flash"),
        ("Gemini 2.0 Flash", "gemini-2.0-flash"),
        ("DeepSeek Chat", "deepseek-chat"),
        ("MLX Llama 3.2 11B Vision (Local)", "mlx-llama-vision-11b"),
        ("MLX Qwen3 4B (Local)", "mlx-qwen3-4b"),
        ("MLX Ministral 14B Reasoning (Local)", "mlx-ministral-14b-reasoning"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("inputContainer")
        self._music_params: dict | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(12)

        # Top row: Model selector and toggles
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Model selector
        model_label = QLabel("Model:")
        model_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        top_row.addWidget(model_label)

        self.model_combo = QComboBox()
        for display_name, _ in self.MODELS:
            self.model_combo.addItem(display_name)
        self.model_combo.setMinimumWidth(180)
        top_row.addWidget(self.model_combo)

        top_row.addSpacing(16)

        # Feature toggles
        self.think_toggle = ToggleButton("Think", "Enable extended thinking mode")
        self.web_toggle = ToggleButton("Web", "Enable web search")
        self.research_toggle = ToggleButton("Research", "Enable deep research mode")
        self.image_toggle = ToggleButton("Image", "Generate images")
        self.music_toggle = ToggleButton("Music", "Generate music")

        top_row.addWidget(self.think_toggle)
        top_row.addWidget(self.web_toggle)
        top_row.addWidget(self.research_toggle)
        top_row.addWidget(self.image_toggle)
        top_row.addWidget(self.music_toggle)

        # Music toggle opens dialog
        self.music_toggle.clicked.connect(self._on_music_toggle)

        top_row.addStretch()
        layout.addLayout(top_row)

        # Bottom row: Text input and send button
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        self.text_input = ExpandingTextEdit()
        self.text_input.returnPressed.connect(self._send_message)
        bottom_row.addWidget(self.text_input)

        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primaryButton")
        self.send_button.setFixedSize(80, 50)
        self.send_button.clicked.connect(self._send_message)
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["button_primary"]};
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS["button_primary_hover"]};
            }}
            QPushButton:disabled {{
                background-color: {COLORS["bg_tertiary"]};
                color: {COLORS["text_muted"]};
            }}
        """)
        bottom_row.addWidget(self.send_button)

        layout.addLayout(bottom_row)

        # Styling
        self.setStyleSheet(f"""
            QFrame#inputContainer {{
                background-color: {COLORS["bg_secondary"]};
                border-top: 1px solid {COLORS["border"]};
            }}
        """)

    def _on_music_toggle(self) -> None:
        """Handle music toggle click."""
        if self.music_toggle.isChecked():
            dialog = MusicGenerationDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._music_params = dialog.get_parameters()
            else:
                self.music_toggle.setChecked(False)
                self._music_params = None
        else:
            self._music_params = None

    def _send_message(self) -> None:
        """Send the current message."""
        text = self.text_input.toPlainText().strip()
        if not text:
            return

        # Get selected model
        model_idx = self.model_combo.currentIndex()
        _, model_id = self.MODELS[model_idx]

        settings = {
            "model": model_id,
            "thinking": self.think_toggle.isChecked(),
            "web_search": self.web_toggle.isChecked(),
            "deep_research": self.research_toggle.isChecked(),
            "image_generation": self.image_toggle.isChecked(),
            "music_generation": self._music_params
            if self.music_toggle.isChecked()
            else None,
        }

        self.text_input.clear()
        self.messageSent.emit(text, settings)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable input."""
        self.text_input.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)

    def focus_input(self) -> None:
        """Focus the text input."""
        self.text_input.setFocus()
