"""
Chat Widget
===========

Displays chat messages with markdown rendering and syntax highlighting.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .styles import COLORS

if TYPE_CHECKING:
    from ..storage import Message


class CodeBlockWidget(QFrame):
    """Widget for displaying code blocks with copy functionality."""

    def __init__(self, code: str, language: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.code = code
        self.language = language
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with language and copy button
        header = QFrame()
        header.setStyleSheet(f"""
            background-color: {COLORS["bg_hover"]};
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 4px 8px;
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        lang_label = QLabel(self.language or "code")
        lang_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        header_layout.addWidget(lang_label)

        header_layout.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS["text_secondary"]};
                border: none;
                padding: 2px 8px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                color: {COLORS["text_primary"]};
            }}
        """)
        copy_btn.clicked.connect(self._copy_code)
        header_layout.addWidget(copy_btn)

        layout.addWidget(header)

        # Code content
        code_edit = QTextEdit()
        code_edit.setPlainText(self.code)
        code_edit.setReadOnly(True)
        code_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS["bg_tertiary"]};
                border: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 12px;
                font-family: "SF Mono", "Menlo", "Monaco", monospace;
                font-size: 13px;
                color: {COLORS["text_primary"]};
            }}
        """)
        code_edit.setFont(QFont("SF Mono", 13))

        # Calculate height based on content
        doc = code_edit.document()
        doc.setDefaultFont(code_edit.font())
        height = int(doc.size().height()) + 30
        code_edit.setFixedHeight(min(height, 400))

        layout.addWidget(code_edit)

    def _copy_code(self) -> None:
        """Copy code to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code)


class MessageBubble(QFrame):
    """A single message bubble in the chat."""

    def __init__(
        self,
        role: str,
        content: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.role = role
        self.content = content
        self._setup_ui()
        self._setup_context_menu()

    def _setup_context_menu(self) -> None:
        """Set up right-click context menu."""
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos) -> None:
        """Show context menu with copy options."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS["bg_secondary"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
                color: {COLORS["text_primary"]};
            }}
            QMenu::item:selected {{
                background-color: {COLORS["bg_active"]};
            }}
        """)

        copy_action = QAction("Copy Message", self)
        copy_action.triggered.connect(self._copy_message)
        menu.addAction(copy_action)

        copy_markdown_action = QAction("Copy as Markdown", self)
        copy_markdown_action.triggered.connect(self._copy_as_markdown)
        menu.addAction(copy_markdown_action)

        menu.exec(self.mapToGlobal(pos))

    def _copy_message(self) -> None:
        """Copy plain text message to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.content)

    def _copy_as_markdown(self) -> None:
        """Copy message with role prefix as markdown."""
        clipboard = QApplication.clipboard()
        role_text = "**You:**" if self.role == "user" else "**Assistant:**"
        clipboard.setText(f"{role_text}\n\n{self.content}")

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        # Role indicator
        role_text = "You" if self.role == "user" else "Assistant"
        role_label = QLabel(role_text)
        role_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: 12px;
            font-weight: 600;
        """)
        layout.addWidget(role_label)

        # Message content - parse and render
        self._render_content(layout)

        # Styling based on role
        bg_color = COLORS["user_bg"] if self.role == "user" else COLORS["assistant_bg"]
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 12px;
            }}
        """)

    def _render_content(self, layout: QVBoxLayout) -> None:
        """Render content with markdown support."""
        # Split content into code blocks and text
        parts = re.split(r"(```[\w]*\n[\s\S]*?```)", self.content)

        for part in parts:
            if part.startswith("```"):
                # Code block
                match = re.match(r"```(\w*)\n([\s\S]*?)```", part)
                if match:
                    language = match.group(1)
                    code = match.group(2).rstrip()
                    code_widget = CodeBlockWidget(code, language)
                    layout.addWidget(code_widget)
            elif part.strip():
                # Regular text - render as HTML with basic markdown
                text_label = QLabel()
                text_label.setWordWrap(True)
                text_label.setTextFormat(Qt.TextFormat.RichText)
                text_label.setOpenExternalLinks(True)
                text_label.setStyleSheet(f"""
                    color: {COLORS["text_primary"]};
                    line-height: 1.5;
                """)

                # Basic markdown conversion
                html = self._markdown_to_html(part)
                text_label.setText(html)

                layout.addWidget(text_label)

    def _markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML."""
        # Escape HTML first
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")

        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

        # Italic: *text* or _text_
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+)`",
            rf'<code style="background-color: {COLORS["bg_tertiary"]}; '
            rf'padding: 2px 6px; border-radius: 4px; font-family: monospace;">\1</code>',
            text,
        )

        # Links: [text](url)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            rf'<a href="\2" style="color: {COLORS["text_accent"]};">\1</a>',
            text,
        )

        # Line breaks
        text = text.replace("\n", "<br>")

        return text

    def update_content(self, content: str) -> None:
        """Update the message content (for streaming)."""
        self.content = content
        # Clear and re-render
        layout = self.layout()
        while layout.count() > 1:  # Keep role label
            item = layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._render_content(layout)


class StreamingMessageBubble(MessageBubble):
    """A message bubble that supports streaming updates."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("assistant", "", parent)
        self._text_label: QLabel | None = None
        self._setup_streaming_ui()

    def _setup_streaming_ui(self) -> None:
        """Set up UI optimized for streaming."""
        # Clear existing content
        layout = self.layout()
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        # Add a simple text label for streaming
        self._text_label = QLabel()
        self._text_label.setWordWrap(True)
        self._text_label.setTextFormat(Qt.TextFormat.PlainText)
        self._text_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            line-height: 1.5;
        """)
        layout.addWidget(self._text_label)

    def append_text(self, text: str) -> None:
        """Append text during streaming."""
        self.content += text
        if self._text_label:
            self._text_label.setText(self.content)

    def finalize(self) -> None:
        """Finalize the message after streaming is complete."""
        # Re-render with full markdown support
        layout = self.layout()
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._text_label = None
        self._render_content(layout)


class ChatWidget(QScrollArea):
    """Main chat display widget."""

    messageClicked = Signal(str)  # message_id

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._messages: list[MessageBubble] = []
        self._streaming_bubble: StreamingMessageBubble | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("border: none; background-color: transparent;")

        # Container widget
        container = QWidget()
        container.setStyleSheet(f"background-color: {COLORS['bg_primary']};")

        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(16)
        self._layout.addStretch()

        self.setWidget(container)

    def add_message(self, role: str, content: str) -> MessageBubble:
        """Add a message to the chat."""
        bubble = MessageBubble(role, content)
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        self._messages.append(bubble)

        # Scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

        return bubble

    def start_streaming(self) -> StreamingMessageBubble:
        """Start a streaming response."""
        self._streaming_bubble = StreamingMessageBubble()
        self._streaming_bubble.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, self._streaming_bubble)
        self._messages.append(self._streaming_bubble)

        return self._streaming_bubble

    def finish_streaming(self) -> None:
        """Finalize the streaming response."""
        if self._streaming_bubble:
            self._streaming_bubble.finalize()
            self._streaming_bubble = None
            self._scroll_to_bottom()

    def clear(self) -> None:
        """Clear all messages."""
        for bubble in self._messages:
            bubble.deleteLater()
        self._messages.clear()
        self._streaming_bubble = None

    def load_messages(self, messages: list[Message]) -> None:
        """Load messages from storage."""
        self.clear()
        for msg in messages:
            self.add_message(msg.role, msg.content)

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the chat."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def copy_all_to_clipboard(self) -> bool:
        """Copy entire conversation to clipboard. Returns True on success."""
        if not self._messages:
            return False

        text_parts = []
        for bubble in self._messages:
            role_text = "You:" if bubble.role == "user" else "Assistant:"
            text_parts.append(f"{role_text}\n{bubble.content}")

        clipboard = QApplication.clipboard()
        clipboard.setText("\n\n---\n\n".join(text_parts))
        return True

    def export_to_markdown(self) -> str:
        """Export conversation as markdown string."""
        if not self._messages:
            return ""

        lines = ["# AI Orchestrator Conversation\n"]
        for bubble in self._messages:
            role_text = "**You:**" if bubble.role == "user" else "**Assistant:**"
            lines.append(f"{role_text}\n\n{bubble.content}\n")
            lines.append("---\n")

        return "\n".join(lines)

    def export_to_file(self, filepath: str | None = None) -> bool:
        """Export conversation to a file. Returns True on success."""
        if not self._messages:
            QMessageBox.information(
                self,
                "Export",
                "No messages to export.",
            )
            return False

        if filepath is None:
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export Conversation",
                "conversation.md",
                "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
            )

        if not filepath:
            return False

        try:
            content = self.export_to_markdown()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Conversation exported to:\n{filepath}",
            )
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not export conversation:\n{str(e)}",
            )
            return False

    def get_conversation_text(self) -> str:
        """Get plain text of entire conversation."""
        if not self._messages:
            return ""

        text_parts = []
        for bubble in self._messages:
            role_text = "You:" if bubble.role == "user" else "Assistant:"
            text_parts.append(f"{role_text}\n{bubble.content}")

        return "\n\n".join(text_parts)


class WelcomeWidget(QWidget):
    """Welcome screen shown when no conversation is active."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo/Title
        title = QLabel("AI Orchestrator")
        title.setStyleSheet(f"""
            font-size: 32px;
            font-weight: 600;
            color: {COLORS["text_primary"]};
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Intelligent multi-model AI assistant")
        subtitle.setStyleSheet(f"""
            font-size: 16px;
            color: {COLORS["text_secondary"]};
            margin-top: 8px;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        # Feature hints
        hints = [
            ("Select a model or let AI choose the best one", "text_secondary"),
            ("Enable thinking mode for complex reasoning", "text_secondary"),
            ("Use web search for current information", "text_secondary"),
            ("Generate images and music", "text_secondary"),
        ]

        for hint_text, color in hints:
            hint = QLabel(f"• {hint_text}")
            hint.setStyleSheet(f"""
                font-size: 14px;
                color: {COLORS[color]};
            """)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(hint)
