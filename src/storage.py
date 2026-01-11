"""
Conversation Storage Layer
==========================

SQLite-based persistent storage for conversation history.
Stores conversations and messages with metadata for the GUI app.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def get_storage_path() -> Path:
    """Get the storage directory path, creating it if needed."""
    storage_dir = Path.home() / ".ai_orchestrator"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@dataclass
class Message:
    """A single message in a conversation."""

    id: str
    conversation_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=lambda: {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Message:
        """Create from database row."""
        return cls(
            id=str(row[0]),
            conversation_id=str(row[1]),
            role=str(row[2]),
            content=str(row[3]),
            created_at=datetime.fromisoformat(str(row[4])),
            metadata=json.loads(str(row[5])) if row[5] else {},
        )


@dataclass
class Conversation:
    """A conversation with its messages."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str | None = None
    settings: dict[str, Any] = field(default_factory=lambda: {})
    messages: list[Message] = field(default_factory=lambda: [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "model": self.model,
            "settings": self.settings,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Conversation:
        """Create from database row (without messages)."""
        return cls(
            id=str(row[0]),
            title=str(row[1]),
            created_at=datetime.fromisoformat(str(row[2])),
            updated_at=datetime.fromisoformat(str(row[3])),
            model=str(row[4]) if row[4] else None,
            settings=json.loads(str(row[5])) if row[5] else {},
            messages=[],
        )


class ConversationStorage:
    """SQLite-based conversation storage."""

    def __init__(self, db_path: Path | None = None):
        """Initialize storage with optional custom database path."""
        if db_path is None:
            db_path = get_storage_path() / "conversations.db"
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT,
                    settings TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC)
            """)
            conn.commit()

    def create_conversation(
        self,
        title: str = "New Chat",
        model: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        now = datetime.now()
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now,
            model=model,
            settings=settings or {},
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, model, settings)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    conversation.model,
                    json.dumps(conversation.settings),
                ),
            )
            conn.commit()

        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID with all its messages."""
        with self._get_connection() as conn:
            # Get conversation
            cursor = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            conversation = Conversation.from_row(row)

            # Get messages
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            )
            conversation.messages = [Message.from_row(r) for r in cursor.fetchall()]

        return conversation

    def list_conversations(self, limit: int = 100) -> list[Conversation]:
        """List all conversations, most recent first."""
        with self._get_connection() as conn:
            # Optimize: Select explicit columns and skip loading settings JSON
            cursor = conn.execute(
                """
                SELECT id, title, created_at, updated_at, model, NULL as settings
                FROM conversations
                ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            )
            return [Conversation.from_row(row) for row in cursor.fetchall()]

    def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> bool:
        """Update conversation metadata."""
        updates: list[str] = []
        params: list[Any] = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if model is not None:
            updates.append("model = ?")
            params.append(model)
        if settings is not None:
            updates.append("settings = ?")
            params.append(json.dumps(settings))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(conversation_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",  # noqa: S608
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._get_connection() as conn:
            # Delete messages first (foreign key)
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        now = datetime.now()
        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            metadata=metadata or {},
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.created_at.isoformat(),
                    json.dumps(message.metadata),
                ),
            )
            # Update conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now.isoformat(), conversation_id),
            )
            conn.commit()

        return message

    def update_message(
        self,
        message_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a message's content or metadata."""
        updates: list[str] = []
        params: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        params.append(message_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE messages SET {', '.join(updates)} WHERE id = ?",  # noqa: S608
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def search_conversations(self, query: str, limit: int = 20) -> list[Conversation]:
        """Search conversations by title or message content."""
        with self._get_connection() as conn:
            # Search in titles
            cursor = conn.execute(
                """
                SELECT DISTINCT c.* FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            )
            return [Conversation.from_row(row) for row in cursor.fetchall()]

    def get_conversation_preview(self, conversation_id: str) -> str | None:
        """Get the first user message as a preview."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT content FROM messages
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY created_at
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row:
                # Truncate for preview
                content = str(row[0])
                return content[:100] + "..." if len(content) > 100 else content
            return None

    def auto_title_conversation(self, conversation_id: str) -> str | None:
        """Auto-generate a title from the first user message."""
        preview = self.get_conversation_preview(conversation_id)
        if preview:
            # Use first line or first 50 chars
            title = preview.split("\n")[0][:50]
            if len(preview) > 50:
                title += "..."
            self.update_conversation(conversation_id, title=title)
            return title
        return None


# Global storage instance
_storage: ConversationStorage | None = None


def get_storage() -> ConversationStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = ConversationStorage()
    return _storage
