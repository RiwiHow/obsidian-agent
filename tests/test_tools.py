from __future__ import annotations

import pytest

from obsidian_agent.rag import retrieve_notes_from_vault
from obsidian_agent.tools import list_notes_in_vault, read_note_from_vault


def test_retrieve_notes_matches_filename_and_body(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "AI Ideas.md").write_text("LLM tool calling notes", encoding="utf-8")
    (vault / "Travel.md").write_text("Paris plan", encoding="utf-8")

    filename_results = retrieve_notes_from_vault(vault, "AI", max_results=5)
    body_results = retrieve_notes_from_vault(vault, "calling", max_results=5)

    assert filename_results[0]["file_path"] == "AI Ideas.md"
    assert filename_results[0]["chunk_id"] == "AI Ideas.md#0"
    assert body_results[0]["file_path"] == "AI Ideas.md"
    assert "tool calling" in body_results[0]["snippet"]
    assert body_results[0]["score"] > 0


def test_read_note_reads_markdown_inside_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hello", encoding="utf-8")

    result = read_note_from_vault(vault, "note.md")

    assert result == {"file_path": "note.md", "content": "# Hello"}


def test_read_note_rejects_outside_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        read_note_from_vault(vault, outside)


def test_empty_search_returns_stable_structure(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("hello", encoding="utf-8")

    assert retrieve_notes_from_vault(vault, "missing", max_results=5) == []


def test_list_notes_returns_markdown_files(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("a", encoding="utf-8")
    (vault / "b.txt").write_text("b", encoding="utf-8")

    assert list_notes_in_vault(vault) == ["a.md"]
