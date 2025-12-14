"""
Sidebar Widget
==============

Conversation history sidebar with search and management.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QInputDialog,
)

from .styles import COLORS

if TYPE_CHECKING:
    from ..storage import Conversation


class ConversationItem(QListWidgetItem):
    """A conversation item in the list."""

    def __init__(self, conversation: Conversation):
        super().__init__()
        self.conversation = conversation
        self.setText(conversation.title)
        self._update_tooltip()

    def _update_tooltip(self) -> None:
        """Update the tooltip with conversation info."""
        date_str = self.conversation.updated_at.strftime("%b %d, %Y at %I:%M %p")
        self.setToolTip(f"Last updated: {date_str}")


class ConversationList(QListWidget):
    """List of conversations with context menu."""

    conversationSelected = Signal(str)  # conversation_id
    conversationDeleted = Signal(str)  # conversation_id
    conversationRenamed = Signal(str, str)  # conversation_id, new_title

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        self._setup_style()

    def _setup_style(self) -> None:
        self.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 12px 16px;
                border-radius: 8px;
                margin: 2px 8px;
                color: {COLORS['text_primary']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['bg_active']};
            }}
        """)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item click."""
        if isinstance(item, ConversationItem):
            self.conversationSelected.emit(item.conversation.id)

    def _show_context_menu(self, pos) -> None:
        """Show context menu for conversation."""
        item = self.itemAt(pos)
        if not isinstance(item, ConversationItem):
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
                color: {COLORS['text_primary']};
            }}
            QMenu::item:selected {{
                background-color: {COLORS['bg_active']};
            }}
        """)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_conversation(item))
        menu.addAction(rename_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_conversation(item))
        menu.addAction(delete_action)

        menu.exec(self.mapToGlobal(pos))

    def _rename_conversation(self, item: ConversationItem) -> None:
        """Rename a conversation."""
        new_title, ok = QInputDialog.getText(
            self,
            "Rename Conversation",
            "New title:",
            text=item.conversation.title,
        )
        if ok and new_title.strip():
            self.conversationRenamed.emit(item.conversation.id, new_title.strip())
            item.setText(new_title.strip())
            item.conversation.title = new_title.strip()

    def _delete_conversation(self, item: ConversationItem) -> None:
        """Delete a conversation."""
        reply = QMessageBox.question(
            self,
            "Delete Conversation",
            f"Are you sure you want to delete '{item.conversation.title}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.conversationDeleted.emit(item.conversation.id)
            row = self.row(item)
            self.takeItem(row)


class Sidebar(QFrame):
    """Sidebar with conversation history."""

    newChatClicked = Signal()
    conversationSelected = Signal(str)
    conversationDeleted = Signal(str)
    conversationRenamed = Signal(str, str)
    settingsClicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with new chat button
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header_layout.setSpacing(12)

        # New chat button
        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['button_primary']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: 600;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['button_primary_hover']};
            }}
        """)
        self.new_chat_btn.clicked.connect(self.newChatClicked.emit)
        header_layout.addWidget(self.new_chat_btn)

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search conversations...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_tertiary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 12px;
                color: {COLORS['text_primary']};
            }}
            QLineEdit:focus {{
                border-color: {COLORS['border_focus']};
            }}
        """)
        self.search_input.textChanged.connect(self._on_search)
        header_layout.addWidget(self.search_input)

        layout.addWidget(header)

        # Conversation list
        self.conversation_list = ConversationList()
        self.conversation_list.conversationSelected.connect(self.conversationSelected.emit)
        self.conversation_list.conversationDeleted.connect(self.conversationDeleted.emit)
        self.conversation_list.conversationRenamed.connect(self.conversationRenamed.emit)
        layout.addWidget(self.conversation_list)

        # Footer with settings
        footer = QFrame()
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                padding: 8px;
            }}
            QPushButton:hover {{
                color: {COLORS['text_primary']};
            }}
        """)
        settings_btn.clicked.connect(self.settingsClicked.emit)
        footer_layout.addWidget(settings_btn)

        footer_layout.addStretch()

        layout.addWidget(footer)

        # Styling
        self.setStyleSheet(f"""
            QFrame#sidebar {{
                background-color: {COLORS['bg_secondary']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)

    def load_conversations(self, conversations: list[Conversation]) -> None:
        """Load conversations into the list."""
        self.conversation_list.clear()
        self._add_conversations_grouped(conversations)

    def _add_conversations_grouped(self, conversations: list[Conversation]) -> None:
        """Add conversations grouped by date."""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)

        groups = {
            "Today": [],
            "Yesterday": [],
            "This Week": [],
            "This Month": [],
            "Older": [],
        }

        for conv in conversations:
            conv_date = conv.updated_at.date()
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif conv_date > week_ago:
                groups["This Week"].append(conv)
            elif conv_date > month_ago:
                groups["This Month"].append(conv)
            else:
                groups["Older"].append(conv)

        for group_name, group_convs in groups.items():
            if group_convs:
                # Add group header
                header_item = QListWidgetItem(group_name)
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)
                header_item.setForeground(Qt.GlobalColor.darkGray)
                self.conversation_list.addItem(header_item)

                # Add conversations
                for conv in group_convs:
                    self.conversation_list.addItem(ConversationItem(conv))

    def add_conversation(self, conversation: Conversation) -> None:
        """Add a new conversation to the top of the list."""
        # Insert at the beginning (after "Today" header if exists)
        insert_pos = 0
        if self.conversation_list.count() > 0:
            first_item = self.conversation_list.item(0)
            if first_item and first_item.text() == "Today":
                insert_pos = 1
            elif first_item and not isinstance(first_item, ConversationItem):
                # First item is a different header, insert "Today" header first
                today_header = QListWidgetItem("Today")
                today_header.setFlags(Qt.ItemFlag.NoItemFlags)
                today_header.setForeground(Qt.GlobalColor.darkGray)
                self.conversation_list.insertItem(0, today_header)
                insert_pos = 1

        self.conversation_list.insertItem(insert_pos, ConversationItem(conversation))

    def select_conversation(self, conversation_id: str) -> None:
        """Select a conversation by ID."""
        for i in range(self.conversation_list.count()):
            item = self.conversation_list.item(i)
            if isinstance(item, ConversationItem) and item.conversation.id == conversation_id:
                self.conversation_list.setCurrentItem(item)
                break

    def _on_search(self, text: str) -> None:
        """Filter conversations by search text."""
        text = text.lower()
        for i in range(self.conversation_list.count()):
            item = self.conversation_list.item(i)
            if isinstance(item, ConversationItem):
                # Show/hide based on search
                matches = text in item.conversation.title.lower()
                item.setHidden(not matches)
            else:
                # Hide headers during search
                item.setHidden(bool(text))
