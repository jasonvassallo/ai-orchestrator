"""
Persistent conversation and long-term memory helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    """Persist lightweight agent conversation history by session."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        sanitized = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in session_id.strip()
        )
        return sanitized or "default"

    def _session_file(self, session_id: str) -> Path:
        return self.base_dir / f"{self._sanitize_session_id(session_id)}.jsonl"

    def append(self, session_id: str, role: str, content: str) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        session_file = self._session_file(session_id)
        with session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def load_recent(self, session_id: str, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []

        session_file = self._session_file(session_id)
        if not session_file.exists():
            return []

        entries: list[dict[str, str]] = []
        with session_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                role = payload.get("role")
                content = payload.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    entries.append({"role": role, "content": content})

        return entries[-limit:]


class MemoryFileStore:
    """Append-only memory file that the agent can read and update."""

    def __init__(self, memory_file: Path) -> None:
        self.memory_file = memory_file
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("", encoding="utf-8")

    def read(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        try:
            text = self.memory_file.read_text(encoding="utf-8")
        except OSError:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def remember(
        self,
        note: str,
        *,
        category: str = "general",
        source: str = "user",
    ) -> None:
        cleaned = note.strip()
        if not cleaned:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = (
            f"\n## {timestamp}\n"
            f"- Category: {category}\n"
            f"- Source: {source}\n"
            f"- Note: {cleaned}\n"
        )
        with self.memory_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    def search(self, query: str, limit: int = 5) -> list[str]:
        query_lower = query.strip().lower()
        if not query_lower:
            return []

        try:
            lines = self.memory_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        matches: list[str] = []
        for line in lines:
            if query_lower in line.lower():
                stripped = line.strip()
                if stripped:
                    matches.append(stripped)
            if len(matches) >= limit:
                break
        return matches


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text and return truncation flag."""
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 3] + "...", True


def safe_json_dump(payload: dict[str, Any]) -> str:
    """Serialize JSON safely for model/tool exchange."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
