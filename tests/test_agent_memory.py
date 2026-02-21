from src.agent.memory import ConversationStore, MemoryFileStore


def test_conversation_store_roundtrip(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append("alpha", "user", "hello")
    store.append("alpha", "assistant", "hi there")

    recent = store.load_recent("alpha", 5)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert recent[1]["content"] == "hi there"


def test_memory_file_store_read_search_and_remember(tmp_path) -> None:
    memory_file = tmp_path / "memory.md"
    store = MemoryFileStore(memory_file)
    store.remember("Favorite editor is VS Code", category="preference", source="user")
    store.remember("Use pytest for tests", category="workflow", source="user")

    text = store.read(1000)
    assert "Favorite editor is VS Code" in text

    matches = store.search("pytest")
    assert matches
    assert any("pytest" in line.lower() for line in matches)
