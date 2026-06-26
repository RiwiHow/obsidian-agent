from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from obsidian_agent.rag import retrieve_notes_from_vault


def _resolve_vault(vault_path: str | Path) -> Path:
    return Path(vault_path).expanduser().resolve()


def _markdown_files(vault_path: str | Path) -> list[Path]:
    vault = _resolve_vault(vault_path)
    if not vault.exists():
        return []
    return sorted(path for path in vault.rglob("*.md") if path.is_file())


def _relative_posix(path: Path, vault: Path) -> str:
    return path.relative_to(vault).as_posix()


def _safe_note_path(vault_path: str | Path, file_path: str | Path) -> Path:
    vault = _resolve_vault(vault_path)
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = vault / candidate

    resolved = candidate.resolve()
    if not resolved.is_relative_to(vault):
        raise ValueError("Refusing to read a file outside the configured Obsidian vault.")
    if resolved.suffix.lower() != ".md":
        raise ValueError("Only Markdown notes can be read.")
    return resolved


def read_note_from_vault(vault_path: str | Path, file_path: str | Path) -> dict[str, str]:
    vault = _resolve_vault(vault_path)
    note_path = _safe_note_path(vault, file_path)
    content = note_path.read_text(encoding="utf-8")
    return {"file_path": _relative_posix(note_path, vault), "content": content}


def list_notes_in_vault(vault_path: str | Path) -> list[str]:
    vault = _resolve_vault(vault_path)
    return [_relative_posix(path, vault) for path in _markdown_files(vault)]


def create_note_tools(
    vault_path: str | Path,
    max_results: int = 5,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    @tool
    def retrieve_notes(query: str) -> list[dict[str, Any]]:
        """Retrieve relevant Markdown chunks from the Obsidian vault for RAG."""
        return retrieve_notes_from_vault(
            vault_path,
            query,
            max_results=max_results,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            markdown_files=_markdown_files,
            relative_path=_relative_posix,
        )

    @tool
    def read_note(file_path: str) -> dict[str, str]:
        """Read one Markdown note from inside the configured Obsidian vault."""
        return read_note_from_vault(vault_path, file_path)

    @tool
    def list_notes() -> list[str]:
        """List Markdown notes in the configured Obsidian vault. Useful for debugging."""
        return list_notes_in_vault(vault_path)

    return [retrieve_notes, read_note, list_notes]
